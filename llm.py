import torch
import torch.nn as nn
import torch.nn.functional as F

# Sample dataset
text = "hello world"
chars = sorted(set(text))
vocab_size = len(chars)

# Character to index mapping
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Encode input and targets
data = torch.tensor(encode(text), dtype=torch.long)

# Define model
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_size)
        self.fc = nn.Linear(emb_size, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x = self.fc(x)
        return x

# Initialize model
model = MiniTransformer(vocab_size, 16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(500):
    for i in range(len(data)-1):
        x = data[i].unsqueeze(0)
        y = data[i+1].unsqueeze(0)

        out = model(x)
        loss = F.cross_entropy(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Generate text
idx = data[0].unsqueeze(0)  # start with first character
output = [idx.item()]

for _ in range(10):
    logits = model(idx.unsqueeze(0))      # shape: [1, 1, vocab_size]
    probs = F.softmax(logits[0, 0], dim=-1)  # shape: [vocab_size]
    idx = torch.multinomial(probs, num_samples=1)  # sample from distribution
    output.append(idx.item())

print("Generated:", decode(output))
