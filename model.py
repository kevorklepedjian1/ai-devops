import torch
import torch.nn as nn

class LogModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 16)
        self.lstm = nn.LSTM(16, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)          
        _, (hidden, _) = self.lstm(x)  
        x = hidden[-1]                 
        x = self.fc(x)
        return self.sigmoid(x)