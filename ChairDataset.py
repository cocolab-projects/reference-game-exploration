from __future__ import print_function

import os
import json
from glob import glob
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import colorsys


import nltk
from nltk import sent_tokenize, word_tokenize

import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import RegexpTokenizer

from collections import defaultdict

FILE_DIR = os.path.realpath(os.path.dirname(__file__))
RAW_DIR = os.path.join(FILE_DIR, 'data')
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
TRAINING_PERCENTAGE = 64 / 100
TESTING_PERCENTAGE = 20 / 100
MIN_USED = 2
MAX_LEN = 10

class ChairDataset(data.Dataset):
    def __init__(self, vocab=None, split='Train', dis='far'):
        
        self.names = np.load(os.path.join('chairs_img_npy/numpy/numpy/', 'names.npy'))
        self.images = np.load(os.path.join('chairs_img_npy/numpy/numpy/', 'names.npy'))
  
        with open(os.path.join(RAW_DIR, 'chairs2k_group_data.csv')) as fp:
            df = pd.read_csv(fp)
        df = df[df['correct'] == True]
        df = df[df['communication_role'] == 'speaker']
        # note that target_chair is always the chair 
        # so label is always 3
        df = df[df['context_condition'] == dis]


        self.image_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
        ])

        self.texts = []
        self.rounds = []
        self.images = []

        self.textsList = [text for text in df['text']]
        self.roundsList = [roundN for roundN in df['trial_num']]
        self.imagesList = [img for img in df['target_chair']]

        print(self.imagesList)
        length = len(self.textsList)
        train_len = int(length * TRAINING_PERCENTAGE)
        test_len = int(length * TESTING_PERCENTAGE)
        if split == 'Train':
            self.texts = self.textsList[:train_len]
            self.rounds = self.roundsList[:train_len]
            self.images = self.imagesList[:train_len]
        elif split == 'Validation':
            self.texts = self.textsList[train_len:-test_len]
            self.rounds = self.roundsList[train_len:-test_len]
            self.images = self.imagesList[train_len:-test_len]
        elif split == 'Test':
            self.texts = self.textsList[-test_len:]
            self.rounds = self.roundsList[-test_len:]
            self.images = self.imagesList[-test_len:]
        if vocab is None:
            self.vocab = self.build_vocab(self.texts)
        else:
            self.vocab = vocab

        df = df[['chair_a', 'chair_b', 'chair_c', 'target_chair', 'text']]

        self.data = df

        for c,i in enumerate(self.images):
            i = self.remake(c)
        

        self.vocab_size = len(self.vocab['w2i'])
        self.target, self.texts = self.concatenate_by_round(self.texts, self.images, self.rounds)
        self.inputs, self.lengths, self.max_len = self.process_texts(self.texts)

    def remake(self, index):
        _, _, _, chair_target, _ = self.data.iloc[index]
        
        chair_target = chair_target + '.png'
        chair_names = list(self.names)


        #ERROR
        index_target = chair_names.index(chair_target)
        chair_target_np = self.images[index_target][index]
        chair_target_pt = torch.from_numpy(chair_target_np)
        chair_target = transforms.ToPILImage()(chair_target_pt).convert('RGB')
        chair_target = self.image_transform(chair_target)

        return chair_target

    def process_texts(self, texts):
        inputs, lengths = [], []

        n = len(texts)
        for i in range(n):
            tokens = preprocess_text(texts[i])
            input_tokens = [SOS_TOKEN] + tokens
            if len(input_tokens) > MAX_LEN-1:
                input_tokens = input_tokens[:MAX_LEN-1] + [EOS_TOKEN]
                length = MAX_LEN
            else:
                input_tokens += [EOS_TOKEN]
                length = len(input_tokens)
                input_tokens.extend([PAD_TOKEN] * (MAX_LEN - length))
            input_indices = [self.vocab['w2i'].get(token, self.vocab['w2i'][UNK_TOKEN]) for token in input_tokens]
            assert(len(input_indices) == MAX_LEN), breakpoint()
            inputs.append(np.array(input_indices))
            lengths.append(length)
        
        inputs = np.array(inputs)
        lengths = np.array(lengths)
        return inputs, lengths, MAX_LEN

    def concatenate_by_round(self, texts, images, rounds):
        concat_texts, target = [], []
        concat = texts[0]
        for i in range(1, len(rounds)):
            if rounds[i] == rounds[i-1]:
                concat += " " + texts[i]
            else:
                target.append(self.images[i-1])
                concat_texts.append(concat)
                concat = texts[i]
        return target, concat_texts

    def build_vocab(self, texts):
        w2c = defaultdict(int)
        i2w, w2i = {}, {}
        for text in texts:
            tokens = preprocess_text(text)
            for token in tokens:
                w2c[token] += 1
        indexCount = 0
        for token in w2c.keys():
            if w2c[token] >= MIN_USED:
                w2i[token] = indexCount
                i2w[indexCount] = token
                indexCount += 1
        w2i[SOS_TOKEN] = indexCount
        w2i[EOS_TOKEN] = indexCount+1
        w2i[UNK_TOKEN] = indexCount+2
        w2i[PAD_TOKEN] = indexCount+3
        i2w[indexCount] = SOS_TOKEN
        i2w[indexCount+1] = EOS_TOKEN
        i2w[indexCount+2] = UNK_TOKEN
        i2w[indexCount+3] = PAD_TOKEN

        vocab = {'i2w': i2w, 'w2i': w2i}

        # print(i2w)
        print("total number of words used at least twice: %d" % len(w2i))
        print("total number of different words: %d" % len(w2c.keys()))
        print("max number of word usage: %d" % max(w2c.values()))
        return vocab

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.target[index], self.inputs[index], self.lengths[index]


class Chairs_ReferenceGame(data.Dataset):
    def __init__(self, vocab, split='Test', dis='far'):
        assert vocab is not None

        with open(os.path.join(RAW_DIR, 'filteredCorpus.csv')) as fp:
            df = pd.read_csv(fp)
        # Only pick out data with true outcomes, far(=easy) conditions, and speaker text
        df = df[df['correct'] == True]
        df = df[df['communication_role'] == 'speaker']
        # note that target_chair is always the chair 
        # so label is always 3
        df = df[df['context_condition'] == dis]


        self.image_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
        ])

        self.texts = []
        self.rounds = []
        self.tgt_images = []
        self.d1_images = []
        self.d2_image = []
        self.d3_image = []

        self.d1_images = [text for text in df['chair_a']]
        self.d2_image = [text for text in df['chair_b']]
        self.d3_image = [text for text in df['chair_c']]
        self.textsList = [text for text in df['text']]
        self.roundsList = [roundN for roundN in df['trial_num']]
        self.tgt_imagesList = [img for img in df['target_chair']]

        length = len(self.textsList)
        train_len = int(length * TRAINING_PERCENTAGE)
        test_len = int(length * TESTING_PERCENTAGE)
        if split == 'Train':
            self.texts = self.textsList[:train_len]
            self.rounds = self.roundsList[:train_len]
            self.tgt_images = self.tgt_imagesList[:train_len]
            self.d1_images = self.d1_imagesList[:train_len]
            self.d2_images = self.d2_imagesList[:train_len]
            self.d3_images = self.d3_imagesList[:train_len]

        elif split == 'Validation':
            self.texts = self.textsList[train_len:-test_len]
            self.rounds = self.roundsList[train_len:-test_len]
            self.tgt_images = self.tgt_imagesList[train_len:-test_len]
            self.d1_images = self.d1_imagesList[train_len:-test_len]
            self.d2_images = self.d2_imagesList[train_len:-test_len]
            self.d3_images = self.d3_imagesList[train_len:-test_len]

        elif split == 'Test':
            self.texts = self.textsList[-test_len:]
            self.rounds = self.roundsList[-test_len:]
            self.tgt_images = self.tgt_imagesList[-test_len:]
            self.d1_images = self.d1_imagesList[-test_len:]
            self.d2_images = self.d2_imagesList[-test_len:]
            self.d3_images = self.d3_imagesList[-test_len:]

        if vocab is None:
            self.vocab = self.build_vocab(self.texts)
        else:
            self.vocab = vocab


        for c,i in enumerate(self.images):
            i = self.remake(c)
        

        self.vocab_size = len(self.vocab['w2i'])
        self.tgt, self.d1, self.d2, self.d3, self.texts = \
                self.concatenate_by_round(self.texts, self.tgt_images, self.d1_images, self.d2_images, self.d3_images, self.rounds)
        self.inputs, self.lengths, self.max_len = self.process_texts(self.texts)

    def process_texts(self, texts):
        inputs, lengths = [], []
        n = len(texts)
        for i in range(n):
            tokens = preprocess_text(texts[i])
            input_tokens = [SOS_TOKEN] + tokens
            if len(input_tokens) > MAX_LEN-1:
                input_tokens = input_tokens[:MAX_LEN-1] + [EOS_TOKEN]
                length = MAX_LEN
            else:
                input_tokens += [EOS_TOKEN]
                length = len(input_tokens)
                input_tokens.extend([PAD_TOKEN] * (MAX_LEN - length))
            input_indices = [self.vocab['w2i'].get(token, self.vocab['w2i'][UNK_TOKEN]) for token in input_tokens]
            assert(len(input_indices) == MAX_LEN), breakpoint()
            inputs.append(np.array(input_indices))
            lengths.append(length)
        
        inputs = np.array(inputs)
        lengths = np.array(lengths)
        return inputs, lengths, MAX_LEN

    def remake(self, index):
        _, _, _, chair_target, _ = self.data.iloc[index]
        
        chair_target = chair_target + '.png'
        chair_names = list(self.names)
        print(self.names)
        index_target = chair_names.index(chair_target)
        chair_target_np = self.images[index_target][index]
        chair_target_pt = torch.from_numpy(chair_target_np)
        chair_target = transforms.ToPILImage()(chair_target_pt).convert('RGB')
        chair_target = self.image_transform(chair_target)

        return chair_target

    def concatenate_by_round(self, texts, tgt_images, d1_images, d2_images, d3_images, rounds):
        concat_texts, tgt, d1, d2 = [], [], [], []
        concat = texts[0]
        for i in range(1, len(rounds)):
            if rounds[i] == rounds[i-1]:
                concat += " " + texts[i]
            else:
                tgt_raw= np.array(tgt_images[i-1])
                tgt.append(tgt_raw)
                d1_raw = np.array(d1_images[i-1])
                d1.append(d1_raw)
                d2_raw = np.array(d2_images[i-1])
                d2.append(d2_raw)
                d3_raw = np.array(d3_images[i-1])
                d3.append(d2_raw)

                concat_texts.append(concat)
                concat = texts[i]
        return tgt, d1, d2, concat_texts

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.tgt[index], self.d1[index], self.d2[index], self.d3[index], self.inputs[index], self.lengths[index]


def preprocess_text(text):
    text = text.lower() 
    tokens = word_tokenize(text)
    i = 0
    while i < len(tokens):
        while (tokens[i] != '.' and '.' in tokens[i]):
            tokens[i] = tokens[i].replace('.','')
        while (tokens[i] != '\'' and '\'' in tokens[i]):
            tokens[i] = tokens[i].replace('\'','')
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
            tokens.insert(i, 'est')
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
            tokens.insert(i-1, 'est')
        i += 1
    replace = {'redd':'red', 'gren': 'green', 'whit':'white', 'biege':'beige', 'purp':'purple', 'olve':'olive', 'ca':'can', 'blu':'blue', 'orang':'orange', 'gray':'grey'}
    for i in range(len(tokens)):
        if tokens[i] in replace.keys():
            tokens[i] = replace[tokens[i]]
    while '' in tokens:
        tokens.remove('')
    return tokens

#Testing
chair = ChairDataset()