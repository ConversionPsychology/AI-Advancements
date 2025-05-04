import streamlit as st
import torch
from tiny_modular_transformer.tokenizer import SimpleTokenizer
from tiny_modular_transformer.model import TinyTransformer

st.title("🧠 Tiny Modular Transformer Sandbox")

sample_text = st.text_area("Input Text", "hello tiny transformer!")
embed_dim = st.slider("Embedding Dimension", 16, 128, 32, step=16)
num_heads = st.slider("Number of Heads", 1, 8, 4)
ff_dim = st.slider("FeedForward Dimension", 32, 256, 64, step=32)
num_layers = st.slider("Number of Transformer Blocks", 1, 6, 2)
block_size = st.slider("Block Size", 8, 64, 16, step=8)

if st.button("Run Transformer"):
    tokenizer = SimpleTokenizer(sample_text)
    model = TinyTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        max_len=block_size
    )

    model.eval()
    tokens = tokenizer.encode(sample_text[:block_size])
    input_tensor = torch.tensor([tokens], dtype=torch.long)

    with torch.no_grad():
        output = model(input_tensor)

    predicted = torch.argmax(output, dim=-1).squeeze(0).tolist()
    decoded = tokenizer.decode(predicted)

    st.text_area("Model Output", decoded, height=100)
