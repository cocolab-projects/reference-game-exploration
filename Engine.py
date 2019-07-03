import os
import os.path
from os import path

import random 
import sys
import numpy as np
from tqdm import tqdm
from itertools import chain
import matplotlib.pyplot as plt
import json
from ColorTraining import test,train
from colorama import init 
from termcolor import colored 
from utils import (AverageMeter, save_checkpoint,get_text)
from ColorModel import TextEmbedding, Supervised
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from ColorDataset import (ColorDataset, Colors_ReferenceGame)
import fileinput

#Run through list in the data folder?

class Engine(object):
    
    def __init__ (self):

        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('dir', type=str, help="directory of config file")
        parser.add_argument('--file_name', type=str, help="name of file [default: config.json]", default="config.json")
        parser.add_argument('--full_diagnostic', type=str, help="run full diagnostic (print) [default: false]", default=False)


        # parser.add_argument('--dir', type=int, help="directory of config file" default="data/data.json")
        parser.add_argument('--cuda', action='store_true', help='Enable cuda')
        args = parser.parse_args()
        
        args.cuda = args.cuda and torch.cuda.is_available()
        self.fd = args.full_diagnostic
        self.device = torch.device('cuda' if args.cuda else 'cpu')
        self.loss = None
        self.accuracy = None
        self.parsed = {}
        self.dir = args.dir
        self.fileName = args.file_name
        if (self.fd):
            print(" ")
            print(colored("==begining data (args put in)==", 'magenta'))
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

        
        if path.exists('seeds_save/' + 'seed.txt'):
            seedNew = random.randint(1,1000001)
            self.seed = seedNew 
            # val.write(str(val.read()) + "\n" + str(seedNew))
            
            with open('seeds_save/' + 'seed.txt','a') as f:
                f.write('\n' + str(self.seed))
                f.flush()
        else:
            completeName = os.path.join("seeds_save", 'seed.txt')         
            file1 = open(completeName, "w")
            file1.write(str(self.seed))
            file1.close()


        self.out_dir = self.parsed['out_dir']

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.name = self.parsed['name']
        self.modelDir = self.parsed['model'][0]
        self.trainDir = self.parsed['training'][0]
        
        self.bi = self.modelDir['bidir']
        self.distance = self.modelDir['dis']
        self.filePath = self.modelDir['file_path']

        self.lr = self.trainDir['learning_rate']
        self.bs = self.trainDir['batch_size']
        self.epochs = self.trainDir['epochs']
        self.dim = self.trainDir['dim']
        self.log_interval = self.trainDir['log_interval']


        self.train_dataset = ColorDataset(split='Train', dis=self.distance)
        self.train_loader = DataLoader(self.train_dataset, shuffle=True, batch_size=self.bs)
        self.N_mini_batches = len(self.train_loader)
        self.vocab_size = self.train_dataset.vocab_size
        self.vocab = self.train_dataset.vocab

        self.test_dataset = ColorDataset(vocab=self.vocab, split='Validation', dis=self.distance)
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
        if (self.fd):
            self.check_data()

        self.train()

        if (self.fd):
            self.load_model()
            self.load_best()
        self.final_loss()
        self.final_accuracy()

             
        if (self.distance == 'close'):
            if (self.bi):

                if path.exists('plot_data/' + 'plot_close_bi.txt'):    
                    with open('plot_data/' + 'plot_close_bi.txt','a') as f:
                        f.write('\n' + str(self.accuracy))
                        f.flush()
                else:
                    completeName = os.path.join("plot_data", 'plot_close_bi.txt')         
                    file1 = open(completeName, "w")
                    file1.write(str(self.accuracy))
                    file1.close()

            else:
            
                if path.exists('plot_data/' + 'plot_close_nonbi.txt'):    
                    with open('plot_data/' + 'plot_close_nonbi.txt','a') as f:
                        f.write('\n' + str(self.accuracy))
                        f.flush()
                else:
                    completeName = os.path.join("plot_data", 'plot_close_nonbi.txt')         
                    file1 = open(completeName, "w")
                    file1.write(str(self.accuracy))
                    file1.close()

        elif (self.distance == 'far'):
            if (self.bi):

                if path.exists('plot_data/' + 'plot_far_bi.txt'):    
                    with open('plot_data/' + 'plot_far_bi.txt','a') as f:
                        f.write('\n' + str(self.accuracy))
                        f.flush()
                else:
                    completeName = os.path.join("plot_data", 'plot_far_bi.txt')         
                    file1 = open(completeName, "w")
                    file1.write(str(self.accuracy))
                    file1.close()

            else:
            
                if path.exists('plot_data/' + 'plot_far_nonbi.txt'):    
                    with open('plot_data/' + 'plot_far_nonbi.txt','a') as f:
                        f.write('\n' + str(self.accuracy))
                        f.flush()
                else:
                    completeName = os.path.join("plot_data", 'plot_far_nonbi.txt')         
                    file1 = open(completeName, "w")
                    file1.write(str(self.accuracy))
                    file1.close()

        # ref_dataset = Colors_ReferenceGame(self.vocab, split='Test',dis=self.distance)
        # l3 = [x for x in ref_dataset.vocab if x not in self.train_dataset.vocab]
        # print(l3)

    

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
        print(colored("==ending data (loaded model)==", 'magenta'))
        print(" ")
        return epoch, track_loss, sup_emb, sup_img, vocab, vocab_size

    def load_best(self, folder='./color_data/', filename='model_best.pth.tar'):
        checkpoint = torch.load(folder + filename)
        epoch = checkpoint['epoch']
        
        print(colored("==begining data (best model)==", 'magenta'))
        print(colored("epoch: "+ str(epoch) , 'cyan'))
        print(colored("==ending data (best model)==", 'magenta'))
        print(" ")
        return epoch
        
    def final_accuracy(self):
        ref_dataset = Colors_ReferenceGame(self.vocab, split='Test',dis=self.distance)
        ref_loader = DataLoader(ref_dataset, shuffle=False, batch_size=self.bs)
        N_mini_batches = len(ref_loader)
        with torch.no_grad():

            total_count = 0
            correct_count = 0
            correct = False

            for batch_idx, (tgt_rgb, d1_rgb, d2_rgb, x_inp, x_len) in enumerate(ref_loader):
                batch_size = x_inp.size(0)
                tgt_rgb = tgt_rgb.float()
                d1_rgb = d1_rgb.float()
                d2_rgb = d2_rgb.float()

                pred_rgb = self.sup_img(x_inp, x_len)
                pred_rgb = torch.sigmoid(pred_rgb)

                for i in range(batch_size):
                    diff_tgt = torch.mean(torch.pow(pred_rgb[i] - tgt_rgb[i], 2))
                    diff_d1 = torch.mean(torch.pow(pred_rgb[i] - d1_rgb[i], 2))
                    diff_d2 = torch.mean(torch.pow(pred_rgb[i] - d2_rgb[i], 2))
                    total_count += 1
                    if diff_tgt.item() < diff_d1.item() and diff_tgt.item() < diff_d2.item():
                        correct_count += 1
                    else:
                        if x_len[i].item() == 3:
                            given_text = get_text(self.vocab['i2w'], np.array(x_inp[i]), x_len[i].item())
                            pred_RGB = (pred_rgb[i] * 255.0).long().tolist()
                            # print()
                            # print('{0} matches with text: {1}'.format(pred_RGB, given_text))
                            # print('{}, {}, {}'.format(tgt_rgb[i] * 255, d1_rgb[i] * 255, d2_rgb[i] * 255))
                            # print('{}, {}, {}'.format(diff_tgt, diff_d1, diff_d2))
                

            accuracy = correct_count / float(total_count) * 100
            # print('====> Final Test Loss: {:.4f}'.format(loss_meter.avg))
            print(colored("==begining data (final accuracy)==", 'magenta'))
            print(colored('====> Final Accuracy: {}/{} = {}%'.format(correct_count, total_count, accuracy), 'cyan'))
            print(colored("==ending data (final accuracy)==", 'magenta'))
            print("")
            self.accuracy = accuracy

    def final_loss(self):
        print(colored("==begining data (final loss)==", 'magenta'))
        test_dataset = ColorDataset(vocab=self.vocab, split='Test',dis=self.distance)
        test_loader = DataLoader(test_dataset, shuffle=True, batch_size=self.bs)
        N_mini_batches = len(test_loader)
        with torch.no_grad():
            loss_meter = AverageMeter()

            for batch_idx, (y_rgb, x_inp, x_len) in enumerate(test_loader):
                batch_size = x_inp.size(0)
                y_rgb = y_rgb.float()

                pred_rgb = self.sup_img(x_inp, x_len)
                pred_rgb = torch.sigmoid(pred_rgb)

                loss = torch.mean(torch.pow(pred_rgb - y_rgb, 2))
                self.loss = loss
                given_text = get_text(self.vocab['i2w'], np.array(x_inp[0]), x_len[0].item())
                pred_RGB = (pred_rgb[0] * 255.0).long().tolist()
                if (self.fd):
                    print(colored('{0} matches with text: {1}'.format(pred_RGB, given_text),'cyan'))

                loss_meter.update(loss.item(), batch_size)
            print(colored('====> Final Test Loss: {:.4f}'.format(loss_meter.avg),'cyan'))
            
        print(colored("==ending data (final loss)==", 'magenta'))
        print("")

color = Engine()