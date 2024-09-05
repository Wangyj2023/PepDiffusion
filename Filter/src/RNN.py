from torch import nn
import pickle


with open('params/peptide_vocab.pkl', 'rb') as f:
    w2i = pickle.load(f)

class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        n_vocab = len(w2i)
        embeding_dim = 128
        hidden_size = 128
        num_layers = 3
        dropout = 0.1

        self.embedding = nn.Embedding(n_vocab, embeding_dim, padding_idx=w2i["_"])
        self.lstm = nn.LSTM(embeding_dim, hidden_size, num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, 2)

    def forward(self, x):
        # x, _ = x
        out = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out
















