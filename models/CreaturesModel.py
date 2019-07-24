from __future__ import print_function

import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.nn.utils.rnn as rnn_utils

class TextEmbedding(nn.Module):
    """ Embeds a |vocab_size| number

    """
    def __init__(self, vocab_size, hidden_dim=64):
        super(TextEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
    
    def forward(self, x):
        return self.embedding(x)
class Supervised(nn.Module):
    # Bi = Bidirectional (True/False)
    # Width = length of final batch (Skinny, Medium, Fat)
    # Number = number of hidden layers after concat (1,2,3)
    def __init__(self, embedding_module,
     bi = True, 
     width = "Medium", 
     number = 2, n_filters=64, img_size=32, channels=4, device=None):
        super(Supervised, self).__init__()
        self.embedding = embedding_module
        self.embedding_dim = embedding_module.embedding.embedding_dim

        self.hidden_dim = 256

        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, batch_first=True, bidirectional=bi)
        self.txt_lin = nn.Linear(self.hidden_dim, self.hidden_dim // 2)

        self.conv = nn.Sequential(
            nn.Conv2d(channels, n_filters, 2, 2, padding=0),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters * 2, 2, 2, padding=0),
            nn.ReLU(),
            nn.Conv2d(n_filters * 2, n_filters * 4, 2, 2, padding=0))
        cout = gen_32_conv_output_dim(img_size)
        self.fc = nn.Linear(n_filters * 4 * cout**2, self.hidden_dim)
        self.cout = cout
        self.n_filters = n_filters
        self.number = number
        if (number == 1):
            if (width=='Skinny'):
                self.sequential = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim // 4), \
                                nn.ReLU(),  \
                                nn.Linear(self.hidden_dim // 4, 1))
            elif (width=='Medium'):
                self.sequential = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim // 3), \
                                nn.ReLU(),  \
                                nn.Linear(self.hidden_dim // 3, 1))
            elif (width=='Fat'):
                self.sequential = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim // 2), \
                                nn.ReLU(),  \
                                nn.Linear(self.hidden_dim // 2, 1))
        if (number == 2):
            if (width=='Skinny'):
                self.sequential = nn.Sequential(
                                nn.Linear(self.hidden_dim, self.hidden_dim // 4), \
                                nn.ReLU(),  \
                                nn.Linear(self.hidden_dim // 4, self.hidden_dim // 16), \
                                nn.ReLU(), \
                                nn.Linear(self.hidden_dim // 16, 1))
            elif (width=='Medium'):
                self.sequential = nn.Sequential(
                                nn.Linear(self.hidden_dim, self.hidden_dim // 3), \
                                nn.ReLU(),  \
                                nn.Linear(self.hidden_dim // 3, self.hidden_dim // 9), \
                                nn.ReLU(), \
                                nn.Linear(self.hidden_dim // 9, 1))
            elif (width=='Fat'):
                self.sequential = nn.Sequential(
                                nn.Linear(self.hidden_dim, self.hidden_dim // 2), \
                                nn.ReLU(),  \
                                nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4), \
                                nn.ReLU(), \
                                nn.Linear(self.hidden_dim // 4, 1))
        if (number == 3):
            if (width=='Skinny'):
                self.sequential = nn.Sequential(
                                nn.Linear(self.hidden_dim, self.hidden_dim // 4), \
                                nn.ReLU(),  \
                                nn.Linear(self.hidden_dim // 4, self.hidden_dim // 16), \
                                nn.ReLU(), \
                                nn.Linear(self.hidden_dim // 16, self.hidden_dim // 64), \
                                nn.ReLU(), \
                                nn.Linear(self.hidden_dim // 64, 1))            
            elif (width=='Medium'):
                self.sequential = nn.Sequential(
                                nn.Linear(self.hidden_dim, self.hidden_dim // 3), \
                                nn.ReLU(),  \
                                nn.Linear(self.hidden_dim // 3, self.hidden_dim // 9), \
                                nn.ReLU(), \
                                nn.Linear(self.hidden_dim // 9, self.hidden_dim // 27), \
                                nn.ReLU(), \
                                nn.Linear(self.hidden_dim // 27, 1))        
            elif (width=='Fat'):
                self.sequential = nn.Sequential(
                                nn.Linear(self.hidden_dim, self.hidden_dim // 2), \
                                nn.ReLU(),  \
                                nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4), \
                                nn.ReLU(), \
                                nn.Linear(self.hidden_dim // 4, self.hidden_dim // 8), \
                                nn.ReLU(), \
                                nn.Linear(self.hidden_dim // 8, 1))        



    def forward(self, img, seq, length):
        batch_size = img.size(0)
        #print(batch_size)
        #print(img)

        out = self.conv(img)
        out = out.view(batch_size, self.n_filters * 4 * self.cout**2)
        #print(out)

        hidden_img = self.fc(out)
        hidden_img = self.txt_lin(hidden_img)
        batch_size = seq.size(0)

        if batch_size > 1:
            sorted_lengths, sorted_idx = torch.sort(length, descending=True)
            seq = seq[sorted_idx]

        # embed sequences
        embed_seq = self.embedding(seq)
        # pack padded sequences
        packed = rnn_utils.pack_padded_sequence(
            embed_seq,
            sorted_lengths.data.tolist() if batch_size > 1 else length.data.tolist(), batch_first=True)
        _, hidden = self.gru(packed)
        hidden = hidden[-1, ...]
        
        if batch_size > 1:
            _, reversed_idx = torch.sort(sorted_idx)
            hidden = hidden[reversed_idx]
        # print(hidden)
        txt_hidden = self.txt_lin(hidden)

        concat = torch.cat((txt_hidden, hidden_img), 1)
        # for layer in self.hidden:
        #     concat = F.relu(layer(concat))

        return self.sequential(concat)

def gen_32_conv_output_dim(s):
    s = get_conv_output_dim(s, 2, 0, 2)
    s = get_conv_output_dim(s, 2, 0, 2)
    s = get_conv_output_dim(s, 2, 0, 2)
    return s


def get_conv_output_dim(I, K, P, S):
    # I = input height/length
    # K = filter size
    # P = padding
    # S = stride
    # O = output height/length
    O = (I - K + 2*P)/float(S) + 1
    return int(O)