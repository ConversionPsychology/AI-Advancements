import torch
from tiny_modular_transformer.tokenizer import SimpleTokenizer
from tiny_modular_transformer.model import TinyTransformer
from tiny_modular_transformer.train import train
from tiny_modular_transformer.transformer_block import TransformerBlock

text = "layer norm off experiment! " * 10
tokenizer = SimpleTokenizer(text)

class NoLayerNormTransformer(TinyTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = torch.nn.ModuleList([
            TransformerBlock(
                embed_dim=kwargs['embed_dim'],
                num_heads=kwargs['num_heads'],
                ff_dim=kwargs['ff_dim'],
                use_layernorm=False
            ) for _ in range(kwargs['num_layers'])
        ])

model = NoLayerNormTransformer(
    vocab_size=tokenizer.vocab_size,
    embed_dim=32,
    num_heads=4,
    ff_dim=64,
    num_layers=2,
    max_len=64
)

train(model, text, tokenizer, epochs=5, lr=1e-3, block_size=16)
