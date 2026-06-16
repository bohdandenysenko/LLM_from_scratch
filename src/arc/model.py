import torch
from torch import nn
from src.arc.tf import TransformerBlock, TransformerBlock_v2
from src.arc.norm import LayerNorm
from src.arc.config import GPTConfig
from typing import Tuple, Optional, List

class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.tok_emb = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.emb_dim)
        self.pos_emb = nn.Embedding(num_embeddings=config.context_length, embedding_dim=config.emb_dim)
        self.drop_emb = nn.Dropout(config.drop_rate)


        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(config=config) for _ in range(config.n_layers)]
        )

        self.final_norm = LayerNorm(emb_dim=config.emb_dim)
        self.out_head = nn.Linear(
            in_features=config.emb_dim, out_features=config.vocab_size, bias=False
        )


    
    def forward(self, in_idx: torch.Tensor):
        batch_size, seq_len = in_idx.shape
        in_idx = in_idx.type(dtype=torch.long)
        tok_embeds = self.tok_emb(in_idx)

        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

# custom implementation with RoPE, weight tying, combined QKV projections, KV cache
class GPTModel_v2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.drop_emb = nn.Dropout(config.drop_rate)
        
        # Use ModuleList to allow per-layer cache handling
        self.trf_blocks = nn.ModuleList([
            TransformerBlock_v2(config) for _ in range(config.n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.emb_dim)
        self.out_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)
        # Weight tying:
        self.out_head.weight = self.tok_emb.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        in_idx: torch.Tensor,
        past_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        pos_offset: int = 0
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Args:
            in_idx: input tokens [batch, seq]
            past_kv: list of (key, value) tuples for each layer
            use_cache: whether to return new cache
            pos_offset: positional offset for RoPE
            
        Returns:
            logits: [batch, seq, vocab]
            new_kv: list of (key, value) tuples per layer (if use_cache)
        """
        batch_size, seq_len = in_idx.shape
        device = in_idx.device
        
        # Determine previous sequence length (for positional offsets)
        if past_kv is not None:
            prev_seq_len = past_kv[0][0].shape[2]
        else:
            prev_seq_len = 0
            
        # Embed tokens
        tok_embeds = self.tok_emb(in_idx.long())
        x = self.drop_emb(tok_embeds)
        
        # Initialize cache storage if caching is enabled
        new_kv = [] if use_cache else None
        
        # Process through each transformer block
        for layer_idx, block in enumerate(self.trf_blocks):
            layer_past_kv = past_kv[layer_idx] if past_kv is not None else None
            
            # Note: The pos_offset for each block is based on the accumulated sequence length.
            # Here we pass prev_seq_len (which is the same for all blocks).
            x, layer_kv = block(
                x,
                past_kv=layer_past_kv,
                use_cache=use_cache,
                pos_offset=prev_seq_len + pos_offset
            )
            
            if use_cache:
                new_kv.append(layer_kv)
        
        x = self.final_norm(x)
        logits = self.out_head(x)
        
        return (logits, new_kv) if use_cache else logits
