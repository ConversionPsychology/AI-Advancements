import torch
from tiny_modular_transformer.tokenizer import SimpleTokenizer
from tiny_modular_transformer.train import train
from tiny_modular_transformer.model import TinyTransformer
from tiny_modular_transformer.rotary import apply_rotary_positional_encoding

class RotaryTransformer(TinyTransformer):
    def forward(self, x):
        x = self.embedding(x)
        x = apply_rotary_positional_encoding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.head(x)

text = "rotary embeddings in action! " * 10
tokenizer = SimpleTokenizer(text)

model = RotaryTransformer(
    vocab_size=tokenizer.vocab_size,
    embed_dim=32,
    num_heads=4,
    ff_dim=64,
    num_layers=2,
    max_len=64
)

train(model, text, tokenizer, epochs=5, lr=1e-3, block_size=16)
