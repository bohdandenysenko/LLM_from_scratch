import torch
from torch import nn
from src.arc.attention import MultiHeadAttention, MultiHeadAttention_v2
from src.arc.ff import FeedForward
from src.arc.norm import LayerNorm
from src.arc.config import GPTConfig
from typing import Tuple, Optional


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=config.emb_dim,
            d_out=config.emb_dim,
            context_length=config.context_length,
            dropout=config.drop_rate,
            num_heads=config.n_heads,
            qkv_bias=config.qkv_bias,
        )
        self.ff = FeedForward(config=config)
        self.norm1 = LayerNorm(emb_dim=config.emb_dim)
        self.norm2 = LayerNorm(emb_dim=config.emb_dim)
        self.drop_shortcut = nn.Dropout(p=config.drop_rate)

    def forward(self, x):

        shortcut = x  # residual connection for attention
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x += shortcut

        shortcut = x  # residual connection for ff bloc
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x += shortcut

        return x


# Custom implementation with KV cache
class TransformerBlock_v2(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.att = MultiHeadAttention_v2(
            d_in=config.emb_dim,
            d_out=config.emb_dim,
            context_length=config.context_length,
            dropout=config.drop_rate,
            num_heads=config.n_heads,
            qkv_bias=config.qkv_bias,
        )
        self.ff = FeedForward(config=config)
        self.norm1 = nn.LayerNorm(config.emb_dim)
        self.norm2 = nn.LayerNorm(config.emb_dim)
        self.dropout = nn.Dropout(config.drop_rate)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        pos_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Attention sublayer with pre-norm:
        normed_x = self.norm1(x)
        attn_ret = self.att(
            normed_x, past_kv=past_kv, use_cache=use_cache, pos_offset=pos_offset
        )
        attn_out, new_kv = attn_ret
        x = x + self.dropout(attn_out)

        # Feed-forward sublayer with pre-norm:
        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout(ff_out)

        return (x, new_kv)
