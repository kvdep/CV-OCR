from typing import Tuple, Optional
import torch
import torch.nn as nn


class Conv(nn.Module):
    """Базовый сверточный блок: Conv2d + BatchNorm2d + SiLU."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: Optional[int] = None,
        g: int = 1,
        act: bool = True,
    ):
        super().__init__()
        self.cv = nn.Conv2d(
            c1,
            c2,
            kernel_size=k,
            stride=s,
            padding=p if p is not None else (k // 2),
            groups=g,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.cv(x)))


class Bottleneck(nn.Module):
    """Стандартный резидуальный блок Bottleneck (1x1 conv -> 3x3 conv)."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: Tuple[int, int] = (3, 3),
        e: float = 0.5,
    ):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


class C3k2(nn.Module):
    """CSP-модуль (Cross Stage Partial) с n блоками Bottleneck."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
        k: int = 3,
    ):
        super().__init__()
        self.c_ = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c_, 1, 1)
        self.cv2 = Conv((2 + n) * self.c_, c2, 1, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c_, self.c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1, y2 = self.cv1(x).chunk(2, 1)
        out = [y1, y2]
        for m in self.m:
            out.append(m(out[-1]))
        return self.cv2(torch.cat(out, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (быстрый пространственный пирамидальный пулинг)."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))


class PSAModule(nn.Module):
    """Pointwise / Position-Sensitive Attention модуль на базе MultiheadAttention."""

    def __init__(self, c: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        xf = x.view(b, c, h * w).permute(0, 2, 1)
        ao, _ = self.attn(xf, xf, xf)
        return (ao + xf).permute(0, 2, 1).view(b, c, h, w)


class C2PSA(nn.Module):
    """CSP-модуль с интеграцией блоков PSAModule внимания."""

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        super().__init__()
        self.c_ = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c_, 1, 1)
        self.cv2 = Conv(2 * self.c_, c2, 1, 1)
        self.m = nn.ModuleList(PSAModule(self.c_) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).chunk(2, 1)
        for m in self.m:
            b = m(b)
        return self.cv2(torch.cat((a, b), 1))
