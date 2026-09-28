import math
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Conv, C3k2, C2PSA
from .positional import PositionalEncoding, PositionalEncoding2D
from .decoder import FlashDecoderLayer
from .penalties import apply_structural_penalty_batch


class FlexibleCNNTransformer(nn.Module):
    """Гибридная архитектура нейросети для оптического распознавания формул (Image-to-LaTeX).
    
    Компоненты:
    1. Энкодер: Сверточный экстрактор признаков с блоками C3k2 (CSP) и блоком внимания C2PSA.
    2. Проектор признаков: Свертка 1x1, 2D синусоидальное позиционное кодирование и LayerNorm.
    3. Декодер: Стек слоев FlashDecoderLayer со встроенным KV-кэшем и F.scaled_dot_product_attention.
    4. Генератор: Линейная классификационная голова с поддержкой жадного поиска и лучевого поиска (Beam Search).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_decoder_layers: int = 8,
        max_len: int = 250,
        version: int = 10,
    ):
        super().__init__()
        self.dm = d_model
        self.ml = max_len
        self.vs = vocab_size
        self.v = version
        self.t2i: Optional[Dict[str, int]] = None
        self.num_layers_cnt = num_decoder_layers
        self.nhead_cnt = nhead

        # 1. Сверточный энкодер
        self.stem = nn.Sequential(
            Conv(1, 32, k=3, s=2, p=1),
            Conv(32, 64, k=3, s=1, p=1),
        )
        self.stage1 = nn.Sequential(
            C3k2(64, 128, n=2, shortcut=True, e=0.5),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        self.stage2 = nn.Sequential(
            C3k2(128, 256, n=2, shortcut=True, e=0.5),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        self.stage3 = nn.Sequential(
            C3k2(256, 256, n=3, shortcut=True, e=0.5),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),
        )
        self.stage4 = nn.Sequential(
            C3k2(256, 256, n=3, shortcut=True, e=0.5),
        )
        self.psa = nn.Sequential(
            C2PSA(256, 256, n=2, e=0.5),
            Conv(256, 256, k=3, s=1, p=1),
        )

        # 2. Адаптация карты признаков
        self.fusion = nn.Conv2d(256, d_model, kernel_size=1, stride=1, bias=False)
        self.pe2d = PositionalEncoding2D(d_model)
        self.ln_mem = nn.LayerNorm(d_model)

        # 3. Декодер
        self.em = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=2048)
        self.layers = nn.ModuleList(
            [FlashDecoderLayer(d_model, nhead) for _ in range(num_decoder_layers)]
        )
        self.final_ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)
        nn.init.xavier_uniform_(self.fc.weight)

    def get_architecture_config(self) -> Dict[str, Any]:
        """Возвращает метаданные конфигурации архитектуры для сохранения в чекпоинт."""
        return {
            "vocab_size": self.vs,
            "d_model": self.dm,
            "nhead": self.nhead_cnt,
            "num_decoder_layers": self.num_layers_cnt,
            "max_len": self.ml,
            "version": self.v,
        }

    def encode(self, img: torch.Tensor):
        """Извлечение визуальных признаков энкодером."""
        x1 = self.stage1(self.stem(img))
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        xf = self.psa(x4)

        ft = self.pe2d(self.fusion(xf))
        mem = self.ln_mem(ft.flatten(2).permute(0, 2, 1))
        mm = F.adaptive_max_pool2d((img > 0).float(), xf.shape[2:]).view(img.size(0), 1, 1, -1).bool()
        return mem, mm

    def forward(self, img: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Прямой проход модели при обучении.
        
        Args:
            img: Тензор нормализованных изображений [B, 1, 128, 1024]
            tgt: Тензор целевых токенов префикса [B, T]
        Returns:
            Логиты вероятностей токенов [B, T, vocab_size]
        """
        mem, mm = self.encode(img)

        out = self.pos_enc(self.em(tgt) * math.sqrt(self.dm))
        b, t = tgt.shape
        self_attn_mask = None
        if self.t2i is not None and "<P>" in self.t2i:
            causal_mask = torch.ones(t, t, device=tgt.device, dtype=torch.bool).tril()
            padding_mask = tgt.ne(self.t2i["<P>"]).unsqueeze(1).unsqueeze(2)
            self_attn_mask = causal_mask.unsqueeze(0).unsqueeze(1) & padding_mask

        for layer in self.layers:
            out = layer(out, mem, self_attn_mask=self_attn_mask, memory_mask=mm)

        return self.fc(self.final_ln(out))

    @torch.no_grad()
    def predict(
        self,
        img: torch.Tensor,
        sos_id: int,
        eos_id: int,
        max_len: int = 250,
        beam_width: int = 1,
    ) -> torch.Tensor:
        """Авторегрессионная генерация последовательности токенов формулы.
        
        При beam_width == 1 используется жадный поиск (Greedy Search).
        При beam_width > 1 используется поиск по лучу (Beam Search) с нормализацией длины.
        """
        self.eval()
        b, d = img.size(0), img.device
        mem, mm = self.encode(img)

        # 1. Жадный поиск
        if beam_width == 1:
            gn = torch.full((b, 1), sos_id, dtype=torch.long, device=d)
            finished = torch.zeros(b, dtype=torch.bool, device=d)
            states = [{} for _ in range(len(self.layers))]

            for step_idx in range(max_len):
                curr_token = gn[:, -1:]
                out = self.pos_enc(
                    self.em(curr_token) * math.sqrt(self.dm), start_pos=step_idx
                )
                for layer_idx, layer in enumerate(self.layers):
                    out = layer(
                        out,
                        mem,
                        self_attn_mask=None,
                        memory_mask=mm,
                        state=states[layer_idx],
                    )
                lg = self.fc(self.final_ln(out)[:, -1, :])
                if self.t2i is not None:
                    lg = apply_structural_penalty_batch(lg, gn, self.t2i)
                nt = torch.argmax(lg, dim=-1, keepdim=True)
                gn = torch.cat([gn, nt], dim=1)
                finished |= nt.squeeze(-1) == eos_id
                if finished.all():
                    break
            return gn

        # 2. Лучевой поиск (Beam Search)
        mem = mem.repeat_interleave(beam_width, dim=0)
        mm = mm.repeat_interleave(beam_width, dim=0)
        gn = torch.full((b * beam_width, 1), sos_id, dtype=torch.long, device=d)
        scores = torch.zeros(b, beam_width, device=d)
        scores[:, 1:] = -1e9
        finished = torch.zeros(b, beam_width, dtype=torch.bool, device=d)
        states = [{} for _ in range(len(self.layers))]

        for step_idx in range(max_len):
            curr_token = gn[:, -1:]
            out = self.pos_enc(
                self.em(curr_token) * math.sqrt(self.dm), start_pos=step_idx
            )
            for layer_idx, layer in enumerate(self.layers):
                out = layer(
                    out,
                    mem,
                    self_attn_mask=None,
                    memory_mask=mm,
                    state=states[layer_idx],
                )
            lg = self.fc(self.final_ln(out)[:, -1, :])
            if self.t2i is not None:
                lg = apply_structural_penalty_batch(lg, gn, self.t2i)

            log_probs = F.log_softmax(lg, dim=-1).view(b, beam_width, -1)
            if finished.any():
                log_probs[finished] = -1e9
                log_probs[..., eos_id] = torch.where(
                    finished, 0.0, log_probs[..., eos_id]
                )

            # Штраф за длину гипотезы
            length_penalty = ((5.0 + step_idx + 1) / 6.0) ** 0.7
            next_scores = (scores.unsqueeze(-1) + (log_probs / length_penalty)).view(b, -1)
            top_scores, top_indices = torch.topk(next_scores, beam_width, dim=-1)
            scores = top_scores
            beam_indices = top_indices // self.vs
            token_indices = top_indices % self.vs

            batch_indices = torch.arange(b, device=d).unsqueeze(1).expand(-1, beam_width)
            flat_gather_indices = (batch_indices * beam_width + beam_indices).view(-1)

            gn = torch.cat([gn[flat_gather_indices], token_indices.view(-1, 1)], dim=1)
            finished = finished[batch_indices, beam_indices] | (token_indices == eos_id)

            for state in states:
                if "self_k" in state:
                    state["self_k"] = state["self_k"][flat_gather_indices]
                    state["self_v"] = state["self_v"][flat_gather_indices]

            if finished.all():
                break

        return gn.view(b, beam_width, -1)[:, 0, :]
