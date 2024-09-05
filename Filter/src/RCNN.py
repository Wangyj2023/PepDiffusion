from torch import nn
import pickle
import torch.nn.functional as F

import torch

with open('params/peptide_vocab.pkl', 'rb') as f:
    w2i = pickle.load(f)
class RCNNModel(nn.Module):
    def __init__(self):
        super(RCNNModel, self).__init__()
        n_vocab = len(w2i)
        embeding_dim = 128
        hidden_size = 256
        num_layers = 3
        dropout = 0.1
        pad_size = 64
        self.embedding = nn.Embedding(n_vocab, embeding_dim, padding_idx=w2i["_"])
        self.lstm = nn.LSTM(embeding_dim, hidden_size, num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.maxpool = nn.MaxPool1d(pad_size)
        self.fc = nn.Linear(hidden_size * 2 + embeding_dim, 2)
    def forward(self, x):
        # x, _ = x
        embed = self.embedding(x)  # [batch_size, seq_len, embeding]=[64, 32, 64]
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out





