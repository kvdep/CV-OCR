from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class FlashDecoderLayer(nn.Module):
    """Слой Transformer Decoder с поддержкой FlashAttention и инкрементального KV-кэша.
    
    Использует F.scaled_dot_product_attention для эффективного вычисления внимания.
    Поддерживает хранение ключей и значений (self_k, self_v, cross_k, cross_v)
    в словаре state для ускорения авторегрессионного инференса.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

        # Проекции Self-Attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Проекции Cross-Attention к выходу энкодера
        self.q_cross_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_cross_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_cross_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_cross_proj = nn.Linear(d_model, d_model, bias=False)

        # Полносвязный блок Feed-Forward Network
        self.ffn1 = nn.Linear(d_model, dim_feedforward)
        self.ffn2 = nn.Linear(dim_feedforward, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Прямой проход слоя декодера.
        
        Args:
            tgt: Тензор входных токенов [B, T, D]
            memory: Выход визуального энкодера [B, S, D]
            self_attn_mask: Причинно-следственная маска [B, 1, T, T]
            memory_mask: Маска визуальных токенов [B, 1, 1, S]
            state: Словарь кэша состояний (KV-кэш) для авторегрессионного шага
        """
        b, t, d = tgt.shape
        m_t = memory.shape[1]

        # 1. Masked Self-Attention
        x = self.ln1(tgt)
        q = self.q_proj(x).view(b, t, self.nhead, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.nhead, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.nhead, self.d_k).transpose(1, 2)

        if state is not None:
            if "self_k" in state:
                k = torch.cat([state["self_k"], k], dim=2)
                v = torch.cat([state["self_v"], v], dim=2)
            state["self_k"] = k
            state["self_v"] = v

        is_causal = (t > 1) and (self_attn_mask is None) and (state is None)
        s_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=self_attn_mask if state is None else None,
            is_causal=is_causal,
            dropout_p=0.0,
        )
        tgt = tgt + self.drop(self.out_proj(s_out.transpose(1, 2).contiguous().view(b, t, d)))

        # 2. Cross-Attention
        x = self.ln2(tgt)
        q_c = self.q_cross_proj(x).view(b, t, self.nhead, self.d_k).transpose(1, 2)

        if state is not None and "cross_k" in state:
            km = state["cross_k"]
            vm = state["cross_v"]
        else:
            km = self.k_cross_proj(memory).view(b, m_t, self.nhead, self.d_k).transpose(1, 2)
            vm = self.v_cross_proj(memory).view(b, m_t, self.nhead, self.d_k).transpose(1, 2)
            if state is not None:
                state["cross_k"] = km
                state["cross_v"] = vm

        c_out = F.scaled_dot_product_attention(
            q_c,
            km,
            vm,
            attn_mask=memory_mask,
            is_causal=False,
            dropout_p=0.0,
        )
        tgt = tgt + self.drop(self.out_cross_proj(c_out.transpose(1, 2).contiguous().view(b, t, d)))

        # 3. Position-wise Feed-Forward Network
        x = self.ln3(tgt)
        tgt = tgt + self.drop(self.ffn2(F.gelu(self.ffn1(x))))
        return tgt
