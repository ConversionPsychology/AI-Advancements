import torch
from tiny_modular_transformer.tokenizer import SimpleTokenizer
from tiny_modular_transformer.model import TinyTransformer
from tiny_modular_transformer.train import train

text = "hello tiny transformer! " * 10
tokenizer = SimpleTokenizer(text)

model = TinyTransformer(
    vocab_size=tokenizer.vocab_size,
    embed_dim=32,
    num_heads=4,
    ff_dim=64,
    num_layers=2,
    max_len=64
)

train(model, text, tokenizer, epochs=5, lr=1e-3, block_size=16)
