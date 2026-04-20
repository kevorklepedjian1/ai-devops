import torch
import torch.nn as nn
import torch.optim as optim

from data import logs, labels
from preprocess import encode_sequence, encoder
from model import LogModel


X = [encode_sequence(seq) for seq in logs]

y = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

max_len = max(len(seq) for seq in X)

X = [seq + [0] * (max_len - len(seq)) for seq in X]
X = torch.tensor(X, dtype=torch.long)


vocab_size = len(encoder.classes_)
model = LogModel(vocab_size)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


for epoch in range(50):
    model.train()

    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


def predict(seq):
    encoded = encode_sequence(seq)
    padded = encoded + [0] * (max_len - len(encoded))
    tensor = torch.tensor([padded], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        prob = model(tensor).item()

    return "ANOMALY 🚨" if prob > 0.5 else "NORMAL ✅", prob



print("\n--- TEST ---")
print(predict(["INFO start", "WARNING memory high", "ERROR crash"]))
print(predict(["INFO login", "INFO action", "INFO logout"]))

torch.save(model.state_dict(), "model.pth")
print("Model saved!")