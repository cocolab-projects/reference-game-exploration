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
    def __init__(self, vocab_size, hidden_dims):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
    
    def forward(self, x):
        return self.embedding(x)


class Test(nn.Module):
    def __init__(
        self,
        vocab_size,#Size of Vocab
        device=None #Device
    ):
        super().__init__()
        embedding_module = TextEmbedding(vocab_size)
    
    #rgb = img
    #seq = texts
    def forward(self, rgb, seq, length):
        # return layer
