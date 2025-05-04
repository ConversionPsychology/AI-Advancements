import torch
import math

def apply_rotary_positional_encoding(x):
    # x shape: (batch, seq_len, dim)
    seq_len = x.size(1)
    dim = x.size(2)

    freqs = torch.arange(0, dim, 2.0, device=x.device)
    freqs = 1.0 / (10000 ** (freqs / dim))
    positions = torch.arange(seq_len, device=x.device).unsqueeze(1)

    angles = positions * freqs.unsqueeze(0)
    sin = torch.sin(angles)
    cos = torch.cos(angles)

    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    x_rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rotated.flatten(-2)
