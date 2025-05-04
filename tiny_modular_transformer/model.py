import torch
import torch.nn as nn
from .embedding import TokenAndPositionEmbedding
from .transformer_block import TransformerBlock

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_len=512):
        super().__init__()
        self.embedding = TokenAndPositionEmbedding(vocab_size, embed_dim, max_len)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.head(x)
