from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 50257  # Vocabulary size
    context_length: int = 1024  # Context length
    emb_dim: int = 768  # Embedding dimension
    n_heads: int = 12  # Number of attention heads
    n_layers: int = 12  # Number of layers
    drop_rate: float = 0.1  # Dropout rate
    qkv_bias: bool = False  # Query-Key-Value bias
