from __future__ import print_function

import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from utils import OrderedCounter
from nltk import sent_tokenize, word_tokenize

import torch
import torch.utils.data as data
from torchvision import transforms
from collections import defaultdict
import datasets
FILE_DIR = os.path.realpath(os.path.dirname(__file__))
DATA_DIR = os.path.join(FILE_DIR, '../data/pilot_coll1/')
RAW_DIR = os.path.join(FILE_DIR, '../data')

NUMPY_DIR = ''

SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
TRAINING_PERCENTAGE = 64 / 100
TESTING_PERCENTAGE = 20 / 100
MIN_USED = 1
MAX_LEN = 10

class ReferenceGame(data.Dataset):
    def __init__(self, vocab=None, split='Validation', train=True, context_condition='all', 
                 image_size=32, image_transform=None, dataVal=None):
        super(ReferenceGame, self).__init__()
        
        NUMPY_DIR = datasets.utils.DIR + '/crea_img_npy/'

        self.split = split
       
        print('loading CSV')
        if (self.split == "Train"):
            csv_path = os.path.join(DATA_DIR, 'train/msgs.tsv')
            csv_path_concat = os.path.join(DATA_DIR, 'train/msgs_concat.tsv')
            csv_path_data = os.path.join(DATA_DIR, 'train/vision/dataset.tsv')

        if (self.split == "Validation"):
            csv_path = os.path.join(DATA_DIR, 'val/msgs.tsv')
            csv_path_concat = os.path.join(DATA_DIR, 'val/msgs_concat.tsv')
            csv_path_data = os.path.join(DATA_DIR, 'val/vision/dataset.tsv')

        if (self.split == "Test"):
            csv_path = os.path.join(DATA_DIR, 'test/msgs.tsv')
            csv_path_concat = os.path.join(DATA_DIR, 'test/msgs_concat.tsv')
            csv_path_data = os.path.join(DATA_DIR, 'test/vision/dataset.tsv')
        
        df = pd.read_csv(csv_path_data, sep='\t')
        # note that target_chair is always the chair 
        # so label is always 3

        df = df.dropna()
        data = np.asarray(df)

        # print(data)
        # make sure rows reference existing images
        # print(data)


        self.data = data

        text = [d[1] for d in data]
    
        if vocab is None:
            print('building vocab.')
            self.vocab = self.build_vocab(text)
        else:
            self.vocab = vocab
        
        self.w2i, self.i2w = self.vocab['w2i'], self.vocab['i2w']
        self.vocab_size = len(self.w2i)

        self.sos_token = SOS_TOKEN
        self.eos_token = EOS_TOKEN
        self.pad_token = PAD_TOKEN
        self.unk_token = UNK_TOKEN

        self.sos_index = self.w2i[self.sos_token]
        self.eos_index = self.w2i[self.eos_token]
        self.pad_index = self.w2i[self.pad_token]
        self.unk_index = self.w2i[self.unk_token]

        self.inputs, self.targets, self.lengths, self.positions, self.max_length \
            = self.process_texts(text)

        self.image_transform = image_transform

        # print(self.vocab)

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


    def process_texts(self, texts):
        inputs, targets, lengths, positions = [], [], [], []

        n = len(texts)
        max_len = 0
        for i in range(n):
            text = texts[i]
            tokens = word_tokenize(text)
            input_tokens = [SOS_TOKEN] + tokens
            target_tokens = tokens + [EOS_TOKEN]
            assert len(input_tokens) == len(target_tokens)
            length = len(input_tokens)
            max_len = max(max_len, length)

            inputs.append(input_tokens)
            targets.append(target_tokens)
            lengths.append(length)

        for i in range(n):
            input_tokens = inputs[i]
            target_tokens = targets[i]
            length = lengths[i]
            input_tokens.extend([PAD_TOKEN] * (max_len - length))
            target_tokens.extend([PAD_TOKEN] * (max_len - length))
            input_tokens = [self.w2i.get(token, self.w2i[UNK_TOKEN]) for token in input_tokens]
            target_tokens = [self.w2i.get(token, self.w2i[UNK_TOKEN]) for token in target_tokens]
            pos = [pos_i+1 if w_i != self.pad_index else 0
                   for pos_i, w_i in enumerate(input_tokens)]
            inputs[i] = input_tokens
            targets[i] = target_tokens
            positions.append(pos)
        
        inputs = np.array(inputs)
        targets = np.array(targets)
        lengths = np.array(lengths)
        positions = np.array(positions)

        return inputs, targets, lengths, positions, max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        _, msg, distr1, distr2, target = self.data[index]
        # chair_target = chair_target + '.png'
        if self.split == "Train":
            for root, dirs, files in os.walk(DATA_DIR+'train/vision/imgs'):
                if distr1 in files:
                    image_np_1 = os.path.join(root, distr1)
                if distr2 in files:
                    image_np_2 = os.path.join(root, distr2)
                if target in files:
                    image_np_tgt = os.path.join(root, target)
        if self.split == "Validation":
            for root, dirs, files in os.walk(DATA_DIR+'val/vision/imgs'):
                if distr1 in files:
                    image_np_1 = os.path.join(root, distr1)
                if distr2 in files:
                    image_np_2 = os.path.join(root, distr2)
                if target in files:
                    image_np_tgt = os.path.join(root, target)
        if self.split == "Test":
            for root, dirs, files in os.walk(DATA_DIR+'test/vision/imgs'):
                if distr1 in files:
                    image_np_1 = os.path.join(root, distr1)
                if distr2 in files:
                    image_np_2 = os.path.join(root, distr2)
                if target in files:
                    image_np_tgt = os.path.join(root, target)
        image_np_1_PIL = Image.open(image_np_1)
        image_np_2_PIL = Image.open(image_np_2)
        image_np_tgt_PIL = Image.open(image_np_tgt)



        if self.image_transform is not None:
            image_np_1_PIL = self.image_transform(image_np_1_PIL)
            image_np_2_PIL = self.image_transform(image_np_2_PIL)
            image_np_tgt_PIL = self.image_transform(image_np_tgt_PIL)

        inputs = self.inputs[index]
        length = self.lengths[index]

        trans = transforms.ToTensor()

        return trans(image_np_tgt_PIL), trans(image_np_1_PIL), trans(image_np_2_PIL), inputs, length

def preprocess_text(text):
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


