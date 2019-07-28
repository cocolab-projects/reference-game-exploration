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
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
    
    def forward(self, x):
        return self.embedding(x)


class Supervised(nn.Module):
    """
    x: text, y: image, z: latent
    Model p(z|x,y)
    @Param embedding_module: nn.Embedding
                             pass the embedding module (share with decoder)
    @Param z_dim: number of latent dimensions
    @Param hidden_dim: integer [default: 256]
                       number of hidden nodes in GRU
    """
    # Bi = Bidirectional (True/False)
    # Width = length of final batch (Skinny, Medium, Fat)
    # Number = number of hidden layers after concat (1,2,3)
    def __init__(
        self,
        embedding_module,
        bi = True, 
        width = "Medium", 
        number = 2, rgb_dim=3,
        device=None
    ):
        super().__init__()

        self.rgb_dim = rgb_dim
        self.embedding = embedding_module
        self.embedding_dim = embedding_module.embedding.embedding_dim

        self.hidden_dim = 256

        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, batch_first=True, bidirectional=bi)
        self.txt_lin = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.rgb_seq = nn.Sequential(
            nn.Linear(rgb_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        )

        self.rgb_to_rnn = nn.Sequential(
            nn.Linear(rgb_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        self.sequential = None
        self.hidden = []
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


    def forward(self, rgb, seq, length):
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

        rgb_hidden = self.rgb_seq(rgb)
        # layer = self.txt_lin(rgb_hidden)

        _, hidden = self.gru(packed)
        hidden = hidden[-1, ...]
        
        if batch_size > 1:
            _, reversed_idx = torch.sort(sorted_idx)
            hidden = hidden[reversed_idx]
        # print(hidden)

        concat = torch.cat((hidden, rgb_hidden), 1)

 

        return self.sequential(concat)
