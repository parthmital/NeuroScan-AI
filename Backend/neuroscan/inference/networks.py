"""Network architectures matching the trained weights."""

from typing import cast

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0


def efficientnet_b0_head(num_classes: int) -> nn.Module:
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return cast(nn.Module, model)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_convs: int) -> None:
        super().__init__()
        layers = []
        ch = in_ch
        for _ in range(n_convs):
            layers += [
                nn.Conv3d(ch, out_ch, 3, padding=1, bias=False),
                nn.InstanceNorm3d(out_ch, affine=True),
                nn.PReLU(out_ch),
            ]
            ch = out_ch
        self.convs = nn.Sequential(*layers)
        self.proj = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convs(x) + self.proj(x)  # type: ignore[no-any-return]


class VNet3D(nn.Module):
    """Four-channel (FLAIR, T1, T1CE, T2) input, three-channel (WT, TC, ET) output."""

    def __init__(self) -> None:
        super().__init__()
        self.e1 = ResBlock(4, 16, 1)
        self.d1 = nn.Conv3d(16, 16, 2, stride=2, bias=False)
        self.e2 = ResBlock(16, 32, 1)
        self.d2 = nn.Conv3d(32, 32, 2, stride=2, bias=False)
        self.e3 = ResBlock(32, 64, 2)
        self.d3 = nn.Conv3d(64, 64, 2, stride=2, bias=False)
        self.b = ResBlock(64, 128, 2)
        self.u3 = nn.ConvTranspose3d(128, 64, 2, 2)
        self.r3 = ResBlock(128, 64, 2)
        self.u2 = nn.ConvTranspose3d(64, 32, 2, 2)
        self.r2 = ResBlock(64, 32, 1)
        self.u1 = nn.ConvTranspose3d(32, 16, 2, 2)
        self.r1 = ResBlock(32, 16, 1)
        self.out = nn.Conv3d(16, 3, 1)
        self.ds2 = nn.Conv3d(32, 3, 1)
        self.ds3 = nn.Conv3d(64, 3, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.e1(x)
        e2 = self.e2(self.d1(e1))
        e3 = self.e3(self.d2(e2))
        b = self.b(self.d3(e3))
        d3 = self.r3(torch.cat([self.u3(b), e3], dim=1))
        d2 = self.r2(torch.cat([self.u2(d3), e2], dim=1))
        d1 = self.r1(torch.cat([self.u1(d2), e1], dim=1))
        return self.out(d1), self.ds2(d2), self.ds3(d3)
