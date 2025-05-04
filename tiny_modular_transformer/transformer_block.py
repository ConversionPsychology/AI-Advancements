import torch
import torch.nn as nn
from .attention import SelfAttention
from .ffn import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, use_layernorm=True):
        super().__init__()
        self.use_layernorm = use_layernorm
        self.attn = SelfAttention(embed_dim, num_heads)
        self.ffn = FeedForward(embed_dim, ff_dim)

        if use_layernorm:
            self.ln1 = nn.LayerNorm(embed_dim)
            self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        residual = x
        if self.use_layernorm:
            x = self.ln1(x)
        x = self.attn(x) + residual

        residual = x
        if self.use_layernorm:
            x = self.ln2(x)
        x = self.ffn(x) + residual
        return x
