import os
import os.path
from os import path
import nltk, collections

import random 
import sys
import numpy as np
from tqdm import tqdm
from itertools import chain
import matplotlib.pyplot as plt
import json
from ChairTraining import test,train
from colorama import init 
from termcolor import colored 
from utils import (AverageMeter, save_checkpoint,get_text)
from ChairModel import TextEmbedding, Supervised
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from ChairDataset import (ChairDataset,Chairs_ReferenceGame)
import fileinput
import time 


DIR = '/mnt/fs5/rona03/'
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
        self.time = 0

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

        
        if path.exists(DIR+ 'seeds_save/' + 'seed.txt'):
            seedNew = random.randint(1,1000001)
            self.seed = seedNew 
            # val.write(str(val.read()) + "\n" + str(seedNew))
            
            with open(DIR+ 'seeds_save/' + 'seed.txt','a') as f:
                f.write('\n' + str(self.seed))
                f.flush()
        else:
            completeName = os.path.join(DIR+ "seeds_save", 'seed.txt')         
            file1 = open(completeName, "w+")
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
        self.num = self.trainDir['number']
        self.width = self.trainDir['width']

        self.bs = self.trainDir['batch_size']
        self.epochs = self.trainDir['epochs']
        self.dim = self.trainDir['dim']
        self.log_interval = self.trainDir['log_interval']


        self.train_dataset = Chairs_ReferenceGame(split='Train', context_condition=self.distance)
        self.data = self.train_dataset.data
        self.train_loader = DataLoader(self.train_dataset, shuffle=True, batch_size=self.bs)
        self.N_mini_batches = len(self.train_loader)
        self.vocab_size = self.train_dataset.vocab_size
        self.vocab = self.train_dataset.vocab
        self.ref_dataset = Chairs_ReferenceGame(vocab=self.vocab, split='Test', dataVal=self.data, context_condition=self.distance)
        self.test_dataset = Chairs_ReferenceGame(vocab=self.vocab, split='Validation', dataVal=self.data, context_condition=self.distance)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False, batch_size=self.bs)

        self.sup_emb = TextEmbedding(self.vocab_size)
        self.sup_img = Supervised(self.sup_emb, self.bi,self.width,self.num)
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
        self.final_time()
        # self.final_perplexity()
             

        if path.exists(DIR+'plot_data/' + 'plot_' +str(self.distance)+ '_'+str(self.bi)+'_'+str(self.width)+'_'+str(self.num)+'.txt'):    
            with open(DIR+'plot_data/' + 'plot_' +str(self.distance)+ '_'+str(self.bi)+'_'+str(self.width)+'_'+str(self.num)+'.txt','a') as f:
                f.write('\n' + str(self.accuracy))
                f.flush()
        else:
            completeName = os.path.join(DIR+'plot_data/' + 'plot_' +str(self.distance)+ '_'+str(self.bi)+'_'+str(self.width)+'_'+str(self.num)+'.txt')         
            file1 = open(completeName, "w")
            file1.write(str(self.accuracy))
            file1.close()
        
        # if (self.distance == 'close'):
        #     if (self.bi):

        #         if path.exists('plot_data/' + 'plot_close_bi.txt'):    
        #             with open('plot_data/' + 'plot_close_bi.txt','a') as f:
        #                 f.write('\n' + str(self.accuracy))
        #                 f.flush()
        #         else:
        #             completeName = os.path.join("plot_data", 'plot_close_bi.txt')         
        #             file1 = open(completeName, "w")
        #             file1.write(str(self.accuracy))
        #             file1.close()

        #     else:
            
        #         if path.exists('plot_data/' + 'plot_close_nonbi.txt'):    
        #             with open('plot_data/' + 'plot_close_nonbi.txt','a') as f:
        #                 f.write('\n' + str(self.accuracy))
        #                 f.flush()
        #         else:
        #             completeName = os.path.join("plot_data", 'plot_close_nonbi.txt')         
        #             file1 = open(completeName, "w")
        #             file1.write(str(self.accuracy))
        #             file1.close()

        # elif (self.distance == 'far'):
        #     if (self.bi):

        #         if path.exists('plot_data/' + 'plot_far_bi.txt'):    
        #             with open('plot_data/' + 'plot_far_bi.txt','a') as f:
        #                 f.write('\n' + str(self.accuracy))
        #                 f.flush()
        #         else:
        #             completeName = os.path.join("plot_data", 'plot_far_bi.txt')         
        #             file1 = open(completeName, "w")
        #             file1.write(str(self.accuracy))
        #             file1.close()

        #     else:
            
        #         if path.exists('plot_data/' + 'plot_far_nonbi.txt'):    
        #             with open('plot_data/' + 'plot_far_nonbi.txt','a') as f:
        #                 f.write('\n' + str(self.accuracy))
        #                 f.flush()
        #         else:
        #             completeName = os.path.join("plot_data", 'plot_far_nonbi.txt')         
        #             file1 = open(completeName, "w")
        #             file1.write(str(self.accuracy))
        #             file1.close()

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
        t0 = time.time()

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
            np.save(os.path.join(DIR+self.out_dir, 'loss.npy'), track_loss)
        self.time = time.time() - t0
    def train_one_epoch(self, epoch): 
        #train a single epoch 

        train_loss = train(epoch,self.sup_emb,self.sup_img,self.train_loader,self.device,self.optimizer)
        return train_loss

    def validate_one_epoch(self, epoch): 
        # validate a single epoch 
        test_loss = test(epoch,self.sup_emb,self.sup_img,self.test_loader,self.device,self.optimizer)
        return test_loss

    def load_model(self,folder=DIR+'color_data/', filename='checkpoint.pth.tar'):
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

    def load_best(self, folder=DIR+'color_data/', filename='model_best.pth.tar'):
        checkpoint = torch.load(folder + filename)
        epoch = checkpoint['epoch']
        
        print(colored("==begining data (best model)==", 'magenta'))
        print(colored("epoch: "+ str(epoch) , 'cyan'))
        print(colored("==ending data (best model)==", 'magenta'))
        print(" ")
        return epoch
        
    def final_accuracy(self):
        ref_dataset = Chairs_ReferenceGame(self.vocab, split='Test', dataVal=self.data, context_condition='self.distance')
        ref_loader = DataLoader(ref_dataset, shuffle=False, batch_size=self.bs)
        N_mini_batches = len(ref_loader)
        with torch.no_grad():

            total_count = 0
            correct_count = 0
            correct = False
            
            for batch_idx, (tgt_rgb, d1_rgb, d2_rgb, x_inp, x_len) in enumerate(ref_loader):
                batch_size = x_inp.size(0)
                tgt_rgb = tgt_rgb.to(self.device).float()
                d1_rgb = d1_rgb.to(self.device).float()
                d2_rgb = d2_rgb.to(self.device).float()
                
                tgt_score = self.sup_img(tgt_rgb, x_inp, x_len)
                d1_score = self.sup_img(d1_rgb, x_inp, x_len)
                d2_score = self.sup_img(d2_rgb, x_inp, x_len)
                soft = nn.Softmax(dim=1)
                loss = soft(torch.cat([tgt_score,d1_score,d2_score],1))
                softList = torch.argmax(loss, dim=1)

                correct_count += torch.sum(softList == 0).item()
                total_count += softList.size(0)
                

            accuracy = correct_count / float(total_count) * 100
            # print('====> Final Test Loss: {:.4f}'.format(loss_meter.avg))
            print(colored("==begining data (final accuracy)==", 'magenta'))
            print(colored('====> Final Accuracy: {}/{} = {}%'.format(correct_count, total_count, accuracy), 'cyan'))
            print(colored("==ending data (final accuracy)==", 'magenta'))
            print("")
            self.accuracy = accuracy

    def final_loss(self):
        print(colored("==begining data (final loss)==", 'magenta'))
        test_dataset = Chairs_ReferenceGame(vocab=self.vocab, split='Test', dataVal = self.data, context_condition=self.distance)
        test_loader = DataLoader(test_dataset, shuffle=True, batch_size=self.bs)
        N_mini_batches = len(test_loader)
        with torch.no_grad():
            loss_meter = AverageMeter()

            for batch_idx, (tgt_rgb, d1_rgb, d2_rgb, x_inp, x_len) in enumerate(test_loader):
                batch_size = x_inp.size(0)
                tgt_rgb = tgt_rgb.to(self.device).float()
                d1_rgb = d1_rgb.to(self.device).float()
                d2_rgb = d2_rgb.to(self.device).float()
                x_inp = x_inp.to(self.device)
                x_len = x_len.to(self.device)

                # obtain predicted rgb
                tgt_score = self.sup_img(tgt_rgb, x_inp, x_len)
                d1_score = self.sup_img(d1_rgb, x_inp, x_len)
                d2_score = self.sup_img(d2_rgb, x_inp, x_len)

                loss = F.cross_entropy(torch.cat([tgt_score,d1_score,d2_score], 1), torch.LongTensor(np.zeros(batch_size)).to(self.device))
                self.loss = loss
                given_text = get_text(self.vocab['i2w'], np.array(x_inp[0]), x_len[0].item())
             
                loss_meter.update(loss.item(), batch_size)
            print(colored('====> Final Test Loss: {:.4f}'.format(loss_meter.avg),'cyan'))
            
        print(colored("==ending data (final loss)==", 'magenta'))
        print("")
    def final_time(self):
        print(colored("==begining data (final time)==", 'magenta'))
        print(colored('====> Final Time: {:.4f}'.format(self.time),'cyan'))
        print(colored("==ending data (final time)==", 'magenta'))
        print("")
    def final_perplexity(self):
        corpus = ""
        perp = 0
        counter = 0

        # for i in self.train_dataset.get_textColor():
        #     corpus = corpus + " " + i
        # for i in self.ref_dataset.get_textColor():
        #     corpus = corpus + " " + i
        # for i in self.test_dataset.get_textColor():
        #     corpus = corpus + " " + i

        # print(corpus)

        model = self.unigram(self.train_dataset.get_textColor())
        model1 = self.unigram(self.ref_dataset.get_textColor())
        model2 = self.unigram(self.test_dataset.get_textColor())

        for i in self.train_dataset.get_textColor():
            if (self.perplexity(i, model) < 100):
                counter = counter + 1
                perp = perp + self.perplexity(i, model)
        for i in self.ref_dataset.get_textColor():
            if (self.perplexity(i, model1) < 100):
                counter = counter + 1
                perp = perp + self.perplexity(i, model1)
        for i in self.test_dataset.get_textColor():
            if (self.perplexity(i, model2) < 100):
                counter = counter + 1
                perp = perp + self.perplexity(i, model2)


        print(colored("==begining data (final perplexity)==", 'magenta'))
        print(colored('====> Final Average Perplexity: {:.4f}'.format(perp/counter),'cyan'))
        print(colored("==ending data (final perplexity)==", 'magenta'))
        print("")

    def unigram(self, tokens):    
        model = collections.defaultdict(lambda: 0.01)
        for f in tokens:
            try:
                model[f] += 1
            except KeyError:
                model [f] = 0
                continue
        N = float(sum(model.values()))
        for word in model:
            model[word] = model[word]/N
        return model
    
    def perplexity(self, testset, model):
        testset = testset.split()
        perplexity = 1
        N = 0
        for word in testset:
            N += 1
            perplexity = perplexity * (1/model[word])
        perplexity = pow(perplexity, 1/float(N)) 
        return perplexity

color = Engine()