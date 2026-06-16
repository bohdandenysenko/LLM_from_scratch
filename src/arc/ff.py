import torch
from torch import nn
from src.arc.config import GPTConfig


class GELU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=config.emb_dim, out_features=4 * config.emb_dim),
            GELU(),
            nn.Linear(in_features=4 * config.emb_dim, out_features=config.emb_dim),
        )

    def forward(self, x):
        return self.layers(x)
