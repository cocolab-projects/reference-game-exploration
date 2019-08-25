# RGE

RGE, Reference Game Engine, is a modular engine that was built for the sole perpose of streamlining and simplyfing the process of training models in the refrence game enviriement.
RGE takes care of all the time consuming activites that are needed to train models,
only requiring the model file and training details.

    $ rge
    usage: rge [-h] [--full_diagnostic FULL_DIAGNOSTIC] [--hot_start HOT_START]
            [--cuda]
            dir

## Installation

Install Pip:

> https://pip.pypa.io/en/stable/installing/

Install RGE by running (in terminal/cmd prompt):

> pip install rge

Clone the data repository (imgs and datasets)

> git clone https://github.com/OverAny/reference-game-exploration-data.git

*If you want to upgrade*:

> pip install rge -U


## Features


- Train a model based on training details (deiscribed in config file)
    - Including many debugging features (full diagnostic)
    - Details every 10 epochs
- Train from a certain checkpoint
- Option to run from gpu


### Terminal Execution:

> dir (directory of the config file)
> 
> --full_diagnostic (prints all given information in the terminal, for debugging)
> 
> --hot_start (directory of a checkpoint that has saved the epoch, optimizer, and model to beginning the training from said checkpoint)
> 
> --cuda (enables cuda, gpu usage)

### Config File (Layout):

    {
        "seed": 0,                                  (seed)
        "name": "ex",                               (name of run)
        "out_dir": "dir",                           (dir to checkpoint and loss saves)
        "gpu": "0",                                 (if cuda is available, which gpu)
        "dir": "dir",                               (dir to result saves)
        "training_per": ".64",                      (percentage used for training)
        "testing_per": ".20",                       (percentage used for testing)
        "data_dir": "dir"                           (dir of data downlaoded from Github) [see installation]
        "model": [
            {
            "type": "",                             (Color, Chairs, Creatures)
            "class_name": "",                       (name of class of the model in the model dir)
            "file_path": "dir",                     (dir of when model class is located)
            "dis": "far"                            (far [small difference in items]/close [large difference in items/all [both])
            }
        ],
        "training": [
            {
            "batch_size": 100,
            "learning_rate": 0.0002,
            "epochs": 20,
            "dim": 5,
            "log_interval": 10
            }
        ]
    }

## Model File (Layout):
    from __future__ import print_function

    import numpy as np
    import os
    import sys
    import torch
    import torch.nn as nn

>Basic Text Embedding

    class TextEmbedding(nn.Module):
        def __init__(self, vocab_size, hidden_dim):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        def forward(self, x):
            return self.embedding(x) 

>Basic Model Class (Named 'Test')

    class Test(nn.Module):

        def __init__(
            self,
            vocab_size,#Size of Vocab
            device=None #Device
        ):
            super().__init__()
            embedding_module = TextEmbedding(vocab_size)

        def forward(self, rgb, seq, length):
            #run layers
            raise NotImplementedError


## Reference Game Set-Up


### Dataset

In the refrence game environment. There are three objects. One of the is the target object. In RGE, each dataset sets up the first item in the output to be the **target**, followed by the first distration, the second distrantion, the inputs, and the length.

> return  (item_a, item_b, item_c, inputs, length)

### Checkpoints
> epoch - The epoch you are on
>
> sup_img - Model class's current state
> 
> track_loss - Loss
> 
> optimizer - Optimizer's current state 
>
> vocab - Vocab list
> 
> vocab_size - Size of vocab
> 
> is_best - True (best run), False (not best run)
>
> folder - Directory for the information

  
    save_checkpoint({
        'epoch': epoch,
        'sup_img': self.sup_img.state_dict(),
        'track_loss': track_loss,
        'optimizer': self.optimizer.state_dict(),
        'vocab': self.vocab,
        'vocab_size': self.vocab_size,
    }, is_best, folder=DIR_DATA)

## Results

This engine will return:
- Final loss
- Final runnning time
- Final accuracy
- Saved loss (as loss.npy)
- Best run (as a checkpoint)

## Vocab Generation

Color Dataset
- Color dataset requires specific processing:
        
        text = text.lower() 
        tokens = word_tokenize(text)
        i = 0
        while i < len(tokens):
            while (tokens[i] != '.' and '.' in tokens[i]):
                tokens[i] = tokens[i].replace('.','')

            while (tokens[i] != '\'' and '\'' in tokens[i]):
                tokens[i] = tokens[i].replace('\'','')

            while (tokens[i] != '~' and '~' in tokens[i]):
                tokens[i] = tokens[i].replace('~','')

            while('-' in tokens[i] or '/' in tokens[i]):
                if tokens[i] == '/' or tokens[i] == '-':
                    tokens.pop(i)
                    i -= 1
                if '/' in tokens[i]:
                    split = tokens[i].split('/')
                    tokens[i] = split[0]
                    i += 1
                    tokens.insert(i, split[1])
                if '-' in tokens[i]:
                    split = tokens[i].split('-')                
                    tokens[i] = split[0]
                    i += 1
                    tokens.insert(i, split[1])
                if tokens[i-1] == '/' or tokens[i-1] == '-':
                    tokens.pop(i-1)
                    i -= 1
                if '/' in tokens[i-1]:
                    split = tokens[i-1].split('/')
                    tokens[i-1] = split[0]
                    i += 1
                    tokens.insert(i-1, split[1])
                if '-' in tokens[i-1]:
                    split = tokens[i-1].split('-')                
                    tokens[i-1] = split[0]
                    i += 1
                    tokens.insert(i-1, split[1])
            if tokens[i].endswith('er'):
                tokens[i] = tokens[i][:-2]
                i += 1
                tokens.insert(i, 'er')
            if tokens[i].endswith('est'):
                tokens[i] = tokens[i][:-3]
                i += 1
                tokens.insert(i, 'est')
            if tokens[i].endswith('ish'):
                tokens[i] = tokens[i][:-3]
                i += 1
                tokens.insert(i, 'ish')
            if tokens[i-1].endswith('er'):
                tokens[i-1] = tokens[i-1][:-2]
                i += 1
                tokens.insert(i-1, 'er')
            if tokens[i-1].endswith('est'):
                tokens[i-1] = tokens[i-1][:-3]
                i += 1
                tokens.insert(i-1, 'est')
            if tokens[i-1].endswith('ish'):
                tokens[i-1] = tokens[i-1][:-3]
                i += 1
                tokens.insert(i-1, 'ish')
            i += 1
        replace = {'redd':'red', 'gren': 'green', 'whit':'white', 'biege':'beige', 'purp':'purple', 'olve':'olive', 'ca':'can', 'blu':'blue', 'orang':'orange', 'gray':'grey'}
        for i in range(len(tokens)):
            if tokens[i] in replace.keys():
                tokens[i] = replace[tokens[i]]
        while '' in tokens:
            tokens.remove('')
        return tokens

Chair Dataset
- Word Tokenizer: https://www.nltk.org/api/nltk.tokenize.html

Creatures Dataset
- Word Tokenizer: https://www.nltk.org/api/nltk.tokenize.html



## Bottlenecks

- Currently, only works on mac
    - (Please post on issue on the github if you want windows to be a feature!)
- Strict structure of applicable models

- The Creatures dataset always takes in all items (not far/close) and has a set testing and training rate
    - Due to the way the reference game was set up

## Contribute

- Issue Tracker: https://github.com/cocolab-projects/reference-game-exploration/issues
- Source Code: https://github.com/cocolab-projects/reference-game-exploration

## Support

If you are having issues, please let me know.
ronarel123@gmail.com

## More information

https://docs.google.com/document/d/1VJNZs63Lg2dKkoM7X6Dcxv5ca4qV_iUyqlIKqriRjqs/edit

License
-------

The project is licensed under the BSD license.
