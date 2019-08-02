from __future__ import print_function

import numpy as np
from copy import deepcopy
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.nn.utils.rnn as rnn_utils


class TextEmbedding(nn.Module):
    """ Embeds a |vocab_size| number

    """
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
    
    def forward(self, x):
        return self.embedding(x) 


class Supervised(nn.Module):
    # do not hardcode
    file_path = "/Users/user/Desktop/Models/Completed_Model_Refrence.py"

    registry = []
    plit = file_path.split('/', 0)
    name = os.path.basename(file_path)
    f_path = file_path.replace(name,"")

    # sys.path.append(os.path.dirname(f_path))

    
    # module = __import__(name)
    # sup = getattr(module, "Testing")

    #module = __import__(module_name)
    #class_ = getattr(module, class_name)
    #instance = class_()

    def __init__(
        self,
        vocab_size,#Size of Vocab
        device=None #Device
    ):
        super().__init__()
        embedding_module = TextEmbedding(vocab_size)
   
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.registry.append(cls)
    
    def get_all_subclasses(self, cls):
        all_subclasses = []

        for subclass in cls.__subclasses__():
            all_subclasses.append(subclass)
            all_subclasses.extend(get_all_subclasses(subclass))

        return all_subclasses

    #rgb = img
    #seq = texts
    def forward(self, rgb, seq, length):
        raise NotImplementedError

