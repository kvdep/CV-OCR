from typing import Dict, Optional
import torch
import torch.nn.functional as F


def apply_structural_penalty_batch(
    logits: torch.Tensor,
    gn: torch.Tensor,
    t2i: Optional[Dict[str, int]],
    pf_empty: float = 1000.0,
    pf_redundant: float = 2.0,
    max_consecutive_run: int = 3,
) -> torch.Tensor:
    """Применяет синтаксические и структурные штрафы к логитам на шаге инференса.
    
    Ограничения:
    1. Запрет немедленного закрытия пустых фигурных скобок: {}
    2. Штраф за избыточное дублирование открывающих скобок: {{
    3. Запрет лишних закрывающих скобок: число '}' не может превышать число '{'
    4. Запрет повторения операторов индексации подряд: '__', '^^', '_^', '^_'
    5. Запрет закрытия скобки сразу после оператора индекса: '_}', '^}'
    6. Штраф за непрерывные циклические повторения одного и того же токена >= max_consecutive_run
    """
    if gn.size(1) < 1 or t2i is None:
        return logits

    o_id, c_id = t2i.get("{", -1), t2i.get("}", -1)
    last_ids = gn[:, -1]

    # Баланс и структура фигурных скобок
    if o_id != -1 and c_id != -1:
        e_mask = last_ids == o_id
        if e_mask.any():
            logits[e_mask, c_id] -= pf_empty
            v = logits[e_mask, o_id]
            logits[e_mask, o_id] = v - torch.abs(v) * pf_redundant - 5.0

        op = (gn == o_id).sum(dim=1)
        cl = (gn == c_id).sum(dim=1)
        inv = cl >= op
        if inv.any():
            logits[inv, c_id] -= pf_empty

    # Ограничения на операторы _ и ^
    pos_ids = [x for x in [t2i.get("_", -1), t2i.get("^", -1)] if x != -1]
    if pos_ids:
        for pid in pos_ids:
            p_mask = last_ids == pid
            if p_mask.any():
                for pid2 in pos_ids:
                    logits[p_mask, pid2] -= pf_empty
                if c_id != -1:
                    logits[p_mask, c_id] -= pf_empty

    # Штраф за циклические повторения
    if gn.size(1) >= max_consecutive_run:
        tail = gn[:, -max_consecutive_run:]
        is_repeating = (tail == last_ids.unsqueeze(1)).all(dim=1)
        if is_repeating.any():
            for idx in torch.where(is_repeating)[0]:
                repeated_tok = last_ids[idx].item()
                if repeated_tok not in [
                    t2i.get("<P>", -1),
                    t2i.get("<S>", -1),
                    t2i.get("<E>", -1),
                ]:
                    logits[idx, repeated_tok] -= 50.0

    return torch.clamp(logits, min=-1000.0, max=1000.0)


def apply_loss_penalties_vectorized(
    logits: torch.Tensor,
    inputs: torch.Tensor,
    ids: torch.Tensor,
    t2i: Dict[str, int],
    penalty_scale: float = 2.0,
    hard_penalty: float = 10000.0,
) -> torch.Tensor:
    """Векторизованный расчет штрафов к логитам модели во время обучения."""
    B, T, V = logits.shape
    device = logits.device
    penalty = torch.zeros_like(logits)

    o_id, c_id = t2i.get("{", -1), t2i.get("}", -1)

    # 1. Баланс скобок
    if o_id != -1 and c_id != -1:
        unbalanced = (inputs == o_id).cumsum(dim=1) <= (inputs == c_id).cumsum(dim=1)
        penalty[..., c_id] += unbalanced.float() * hard_penalty

    # 2. Недопустимые операторы после _ и ^
    pos_ids = [x for x in [t2i.get("_", -1), t2i.get("^", -1)] if x != -1]
    if pos_ids:
        pos_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        for pid in pos_ids:
            pos_mask |= inputs == pid
        for pid2 in pos_ids:
            penalty[..., pid2] += pos_mask.float() * hard_penalty
        if c_id != -1:
            penalty[..., c_id] += pos_mask.float() * hard_penalty

    # 3. Квадратичный штраф за превышение реальной частоты символов в последовательности
    target_counts = torch.zeros(B, V, device=device).scatter_add_(
        1, ids, torch.ones_like(ids, dtype=torch.float)
    )
    one_hot_inputs = F.one_hot(inputs, num_classes=V).float()
    current_counts = one_hot_inputs.cumsum(dim=1)

    excess = F.relu(current_counts + 1.0 - target_counts.unsqueeze(1))
    for ignore_tok in ["<P>", "<S>", "<E>", "<UNK>"]:
        tok_id = t2i.get(ignore_tok, -1)
        if tok_id != -1:
            excess[:, :, tok_id] = 0.0

    penalty += (excess ** 2) * penalty_scale
    penalty = torch.clamp(penalty, max=hard_penalty)

    # Защита от переполнения FP16
    return torch.clamp(logits - penalty, min=-60000.0, max=60000.0)
