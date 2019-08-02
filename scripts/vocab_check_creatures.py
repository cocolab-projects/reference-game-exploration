import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from nltk import sent_tokenize, word_tokenize

import torch
import torch.utils.data as data
from torchvision import transforms
from collections import defaultdict

MIN_USED = 1
DATA_DIR = os.path.join("../", 'data/pilot_coll1/')
csv_path_data = os.path.join(DATA_DIR, 'train/vision/dataset.tsv')


df = pd.read_csv(csv_path_data, sep='\t')

df = df.dropna()
data = np.asarray(df)


texts = [d[1] for d in data]

w2c = defaultdict(int)
i2w, w2i = {}, {}
for text in texts:
    text = text.lower() 
    tokens = word_tokenize(text)
    for token in tokens:
        w2c[token] += 1
indexCount = 0
for token in w2c.keys():
    if w2c[token] >= MIN_USED:
        w2i[token] = indexCount
        i2w[indexCount] = token
        indexCount += 1
