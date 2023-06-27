import torch
import torch.nn as nn
import torch_redstone as rst


class PointNet(nn.Module):
    def __init__(self, in_dim, repr_dim, scaling) -> None:
        super().__init__()
        self.lift_1 = rst.MLP([in_dim, 64, 64], 1)
        self.lift_2 = rst.MLP([64, 64 * scaling, 128 * scaling, 1024 * scaling], 1, lambda x: torch.nn.functional.relu(x, True))
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.top = nn.Sequential(rst.MLP([1024 * scaling, 512 * scaling]), nn.Linear(512 * scaling, repr_dim))

    def forward(self, xyz: torch.Tensor, features: torch.Tensor, device=None, quantization_size=0.05):
        x = features.transpose(-1, -2).contiguous()
        # x: [B, C, N]
        x = self.lift_1(x)
        x = self.lift_2(x)
        x = self.maxpool(x).squeeze(-1)
        x = self.top(x)
        return x


def make(cfg):
    return PointNet(cfg.model.in_channel, cfg.model.out_channel, cfg.model.scaling)
