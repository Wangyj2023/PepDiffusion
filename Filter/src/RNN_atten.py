from torch import nn
import pickle
import torch.nn.functional as F

import torch

with open('params/peptide_vocab.pkl', 'rb') as f:
    w2i = pickle.load(f)
class RNN_attenModel(nn.Module):
    def __init__(self):
        super(RNN_attenModel, self).__init__()
        n_vocab = len(w2i)
        embeding_dim = 128
        hidden_size = 128
        hidden_size2 = 64
        num_layers = 3
        dropout = 0.1
        self.embedding = nn.Embedding(n_vocab, embeding_dim, padding_idx=w2i["_"])
        self.lstm = nn.LSTM(embeding_dim, hidden_size, num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.zeros(hidden_size * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size2)
        self.fc = nn.Linear(hidden_size2, 2)

    def forward(self, x):
        # x, _ = x
        emb = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        H, _ = self.lstm(emb)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]
        M = self.tanh1(H)  # [128, 32, 256]
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = H * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc(out)  # [128, 64]
        return out



