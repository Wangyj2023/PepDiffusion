import torch
import pickle
import re
import copy
import numpy as np
import pandas as pd
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset


with open('params/peptide_vocab.pkl', 'rb') as f:
    w2i = pickle.load(f)
class TransformerModel(nn.Module):
    def __init__(self,num_classes = 2):
        super(TransformerModel, self).__init__()
        n_vocab = len(w2i)
        embeding_dim = 128
        hidden_dim = 256
        pad_size = 64
        dropout = 0.1
        num_layers = 6
        num_classes = num_classes
        self.embedding = nn.Embedding(n_vocab, embeding_dim, padding_idx=w2i["_"])
        self.postion_embedding = Positional_Encoding(embeding_dim, pad_size, dropout, torch.device("cuda"))
        self.encoder = Encoder(embeding_dim, 8, hidden_dim, dropout)
        self.encoders = nn.ModuleList([copy.deepcopy(self.encoder) for _ in range(num_layers)])
        self.fc1 = nn.Linear(pad_size * embeding_dim, num_classes)
    def forward(self, x):
        out = self.embedding(x)
        out = self.postion_embedding(out)
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out
class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)
    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out
class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out
class Scaled_Dot_Product_Attention(nn.Module):
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()
    def forward(self, Q, K, V, scale=None):
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context
class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)
    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        scale = K.size(-1) ** -0.5  
        context = self.attention(Q, K, V, scale)
        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x  
        out = self.layer_norm(out)
        return out
class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)
    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x 
        out = self.layer_norm(out)
        return out
def peptide_tokenizer(peptide):
    #need to remove "X", "B", "Z", "U", "O"
    pattern =  "(G|A|L|M|F|W|K|Q|E|S|P|V|I|C|Y|H|R|N|D|T)"
    regezz = re.compile(pattern)
    tokens = [token for token in regezz.findall(peptide)]
    assert peptide == ''.join(tokens), ("{} could not be joined".format(peptide))
    return tokens
def encode_seq(sequence, max_len, char_dict):
    "Converts tokenized sequences to list of token ids"
    for i in range(max_len - len(sequence)):
        if i == 0:
            sequence.append('<end>')
        else:
            sequence.append('_')
    seq_vec = [char_dict[c] for c in sequence]
    return seq_vec
def vae_data_gen(data, max_len=63, char_dict=w2i):
    seq_list = data[:,0] 
    seq_list = [peptide_tokenizer(x) for x in seq_list]     
    encoded_data = torch.empty((len(seq_list), max_len+1))
    for j, seq in enumerate(seq_list):
        encoded_seq = encode_seq(seq, max_len, char_dict)
        encoded_seq = [0] + encoded_seq
        encoded_data[j,:] = torch.tensor(encoded_seq)
    return encoded_data
class MyDataset_class(Dataset):
    def __init__(self,pos_path,neg_path):
        super().__init__()
        self.pos_data = pd.read_csv(pos_path,header=None)
        self.neg_data = pd.read_csv(neg_path,header=None)
        self.pos_num = len(self.pos_data)
        self.neg_num = len(self.neg_data)
        self.pos_data = self.pos_data.values.tolist()
        self.neg_data = self.neg_data.values.tolist()
        self.max_length = 64
    def __len__(self):
        return  self.pos_num + self.neg_num
    def __getitem__(self, index):
        # AMP 0 NAMP 1
        if self.pos_num > index:
            seq = self.pos_data[index]
            act = 0
        else:
            seq = self.neg_data[index-self.pos_num]
            act = 1
        seq_list = peptide_tokenizer(seq[0])
        encoded_seq = encode_seq(seq_list, self.max_length - 1, w2i)
        encoded_seq = [0] + encoded_seq
        return torch.tensor(encoded_seq),act
