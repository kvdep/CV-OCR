import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Одномерное синусоидальное позиционное кодирование последовательности токенов."""

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Добавляет позиционный эмбеддинг к входному тензору x [B, T, D]."""
        return x + self.pe[:, start_pos : start_pos + x.size(1), :]


class PositionalEncoding2D(nn.Module):
    """Двумерное синусоидальное позиционное кодирование для карты признаков [B, C, H, W].
    
    Первая половина каналов кодирует координату по ширине W,
    вторая половина каналов кодирует координату по высоте H.
    """

    def __init__(self, d_model: int, max_h: int = 32, max_w: int = 256):
        super().__init__()
        pe = torch.zeros(d_model, max_h, max_w)
        d_half = d_model // 2
        div_term = torch.exp(
            torch.arange(0, d_half, 2).float() * (-math.log(10000.0) / d_half)
        )
        pos_w = torch.arange(0, max_w).float().unsqueeze(1)
        pos_h = torch.arange(0, max_h).float().unsqueeze(1)

        pe[0:d_half:2, :, :] = (
            torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, max_h, 1)
        )
        pe[1:d_half:2, :, :] = (
            torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, max_h, 1)
        )
        pe[d_half::2, :, :] = (
            torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, max_w)
        )
        pe[d_half + 1 :: 2, :, :] = (
            torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, max_w)
        )
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Добавляет 2D позиционный эмбеддинг к тензору x [B, C, H, W]."""
        return x + self.pe[:, :, : x.size(2), : x.size(3)]
