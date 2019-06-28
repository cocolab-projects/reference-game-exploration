import os
import sys
import numpy as np
from tqdm import tqdm
from itertools import chain

import json
from ColorTraining import test,train
from colorama import init 
from termcolor import colored 
from utils import (AverageMeter, save_checkpoint)
from ColorModel import TextEmbedding, Supervised
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from ColorDataset import ColorDataset

#Run through list in the data folder?

class Engine(object):
    
    def __init__ (self):
        print(" ")
        print(colored("==begining data (args put in)==", 'magenta'))
        
   
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('dir', type=str, help="directory of config file")
        parser.add_argument('--file_name', type=str, help="name of file [default: config.json]", default="config.json")
        # parser.add_argument('--dir', type=int, help="directory of config file" default="data/data.json")
        parser.add_argument('--cuda', action='store_true', help='Enable cuda')
        args = parser.parse_args()
        
        args.cuda = args.cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if args.cuda else 'cpu')

        self.parsed = {}
        self.dir = args.dir
        self.fileName = args.file_name
        print(colored("Directory: "+ self.dir, 'cyan'))
        print(colored("File Name: "+ self.fileName, 'cyan'))

        print(colored("==ending data (args put in)==", 'magenta'))
        print(" ")


        self.config = self.dir + "/" + self.fileName
        with open(self.config) as f:
            self.parsed = json.load(f)
            # print(self.parsed)

        assert self.parsed
        self.seed = self.parsed['seed']
        self.out_dir = self.parsed['out_dir']

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.name = self.parsed['name']
        self.modelDir = self.parsed['model'][0]
        self.trainDir = self.parsed['training'][0]
        
        self.bi = self.modelDir['bidir']
        self.filePath = self.modelDir['file_path']

        self.lr = self.trainDir['learning_rate']
        self.bs = self.trainDir['batch_size']
        self.epochs = self.trainDir['epochs']
        self.dim = self.trainDir['dim']
        self.log_interval = self.trainDir['log_interval']

        self.check_data()

        self.train_dataset = ColorDataset(split='Train')
        self.train_loader = DataLoader(self.train_dataset, shuffle=True, batch_size=self.bs)
        self.N_mini_batches = len(self.train_loader)
        self.vocab_size = self.train_dataset.vocab_size
        self.vocab = self.train_dataset.vocab

        self.test_dataset = ColorDataset(vocab=self.vocab, split='Validation')
        self.test_loader = DataLoader(self.test_dataset, shuffle=False, batch_size=self.bs)

        self.sup_emb = TextEmbedding(self.vocab_size)
        self.sup_img = Supervised(self.sup_emb, self.bi)
        
        self.sup_emb = self.sup_emb.to(self.device)
        self.sup_img = self.sup_img.to(self.device)
        self.optimizer = torch.optim.Adam(
            chain(
                self.sup_emb.parameters(), 
                self.sup_img.parameters(),
            ), lr=self.lr)

        # model
        self.__init_model(self.config)

        self.train()
        self.load_model()
        self.load_best()
    def __init_model(self, config):
        print("init model")
        #import file model
        #Model or Training class? Save to a self.var?
    def check_data(self):
        print(colored("==begining data (config)==", 'magenta'))
        print(colored("name: "+ self.name , 'cyan'))
        print(colored("bi: "+ str(self.bi) , 'cyan'))
        print(colored("filePath: "+ self.filePath , 'cyan'))
        print(colored("learning rate: "+ str(self.lr) , 'cyan'))
        print(colored("batch size: "+ str(self.bs) , 'cyan'))
        print(colored("epochs: "+ str(self.epochs) , 'cyan'))
        print(colored("dim: "+ str(self.dim) , 'cyan'))
        print(colored("log interval: "+ str(self.log_interval) , 'cyan'))
        print(colored("==ending data (config)==", 'magenta'))
        print(" ")

    def train(self):
        print("train")
        print("begin training...")
        best_loss = float('inf')
        track_loss = np.zeros((self.epochs, 2))

        for epoch in range(1, self.epochs + 1):
            t_loss = self.train_one_epoch(epoch)
            v_loss = self.validate_one_epoch(epoch)

            is_best = v_loss < best_loss
            best_loss = min(v_loss, best_loss)
            track_loss[epoch - 1, 0] = t_loss
            track_loss[epoch - 1, 1] = v_loss

            save_checkpoint({
                'epoch': epoch,
                'sup_emb': self.sup_emb.state_dict(),
                'sup_img': self.sup_img.state_dict(),
                'track_loss': track_loss,
                'optimizer': self.optimizer.state_dict(),
                'vocab': self.vocab,
                'vocab_size': self.vocab_size,
            }, is_best, folder=self.out_dir)
            np.save(os.path.join(self.out_dir, 'loss.npy'), track_loss)

    def train_one_epoch(self, epoch): 
        #train a single epoch 

        train_loss = train(epoch,self.sup_emb,self.sup_img,self.train_loader,self.device,self.optimizer)
        return train_loss

    def validate_one_epoch(self, epoch): 
        # validate a single epoch 
        test_loss = test(epoch,self.sup_emb,self.sup_img,self.test_loader,self.device,self.optimizer)
        return test_loss

    def load_model(self,folder='./color_data/', filename='checkpoint.pth.tar'):
        checkpoint = torch.load(folder + filename)
        epoch = checkpoint['epoch']
        track_loss = checkpoint['track_loss']
        sup_emb = checkpoint['sup_emb']
        sup_img = checkpoint['sup_img']
        vocab = checkpoint['vocab']
        vocab_size = checkpoint['vocab_size']

        print(colored("==begining data (loaded model)==", 'magenta'))
        print(colored("epoch: "+ str(epoch) , 'cyan'))
        print(colored("track loss: "+ str(track_loss) , 'cyan'))
        print(colored("sup emb: "+ str(sup_emb) , 'cyan'))
        print(colored("sup img: "+ str(sup_img) , 'cyan'))
        print(colored("vocab: "+ str(vocab) , 'cyan'))
        print(colored("vocab size: "+ str(vocab_size) , 'cyan'))
        print(colored("==begining data (loaded model)==", 'magenta'))
        print(" ")
        return epoch, track_loss, sup_emb, sup_img, vocab, vocab_size

    def load_best(self, folder='./color_data/', filename='model_best.pth.tar'):
        checkpoint = torch.load(folder + filename)
        epoch = checkpoint['epoch']
        
        print(colored("==begining data (best model)==", 'magenta'))
        print(colored("epoch: "+ str(epoch) , 'cyan'))
        print(colored("==begining data (best model)==", 'magenta'))
        print(" ")
        return epoch

color = Engine()