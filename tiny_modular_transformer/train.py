import torch
import torch.nn as nn
import torch.optim as optim
from .tokenizer import SimpleTokenizer
from .model import TinyTransformer

def train(model, data, tokenizer, epochs=10, lr=1e-3, block_size=32):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        inputs = torch.tensor([tokenizer.encode(data[i:i+block_size]) for i in range(len(data)-block_size)], dtype=torch.long)
        targets = torch.tensor([tokenizer.encode(data[i+1:i+1+block_size]) for i in range(len(data)-block_size)], dtype=torch.long)

        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_fn(output.view(-1, tokenizer.vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
