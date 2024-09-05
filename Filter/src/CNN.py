from torch import nn
import pickle
import torch.nn.functional as F


with open('params/peptide_vocab.pkl', 'rb') as f:
    w2i = pickle.load(f)
class CNNModel(nn.Module):
    def __init__(self,num_classes = 2):
        super(CNNModel, self).__init__()
        n_vocab = len(w2i)
        embeding_dim = 128
        conv_num = 256
        num_classes = num_classes
        self.embedding = nn.Embedding(n_vocab, embeding_dim, padding_idx=w2i["_"])
        self.conv_region = nn.Conv2d(1, conv_num, (3, embeding_dim), stride=1)
        self.conv = nn.Conv2d(conv_num, conv_num, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(conv_num, num_classes)

    def forward(self, x):
        # x = x[0]
        x = self.embedding(x)
        x = x.unsqueeze(1)              # [batch_size, 250, seq_len, 1]
        x = self.conv_region(x)         # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)            # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)                # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)            # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)                # [batch_size, 250, seq_len-3+1, 1]
        while x.size()[2] > 2:
            x = self._block(x)
        x = x.squeeze()                 # [batch_size, num_filters(250)]
        x = self.fc(x)
        return x
    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)
        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)
        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)
        # Short Cut
        x = x + px
        return x

