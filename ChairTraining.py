from __future__ import print_function

import os
import sys
import numpy as np
from tqdm import tqdm
from itertools import chain

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from ChairDataset import ChairDataset, Chairs_ReferenceGame

from utils import (AverageMeter, save_checkpoint)
from ChairModel import TextEmbedding, Supervised



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', type=str, help='where to save checkpoints')
    parser.add_argument('--d-dim', type=int, default=100,
                        help='number of hidden dimensions [default: 100]')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='batch size [default=100]')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate [default=0.0002]')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs [default: 200]')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='interval to print results [default: 10]')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', action='store_true', help='Enable cuda')
    args = parser.parse_args()
    
    args.cuda = args.cuda and torch.cuda.is_available()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if args.cuda else 'cpu')

    # data_dir = get_data_dir(args.dataset)
    train_dataset = Chairs_ReferenceGame(split='Train')
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    N_mini_batches = len(train_loader)
    vocab_size = train_dataset.vocab_size
    vocab = train_dataset.vocab

    test_dataset = Chairs_ReferenceGame(vocab=vocab, split='Validation')
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)

    sup_emb = TextEmbedding(vocab_size)
    sup_img = Supervised(sup_emb)
    
    sup_emb = sup_emb.to(device)
    sup_img = sup_img.to(device)
    optimizer = torch.optim.Adam(
        chain(
            sup_emb.parameters(), 
            sup_img.parameters(),
        ), lr=args.lr)


def train(epoch, sup_emb, sup_img,train_loader,device,optimizer):
    sup_emb.train()
    sup_img.train()

    loss_meter = AverageMeter()
    pbar = tqdm(total=len(train_loader))
    for batch_idx, (tgt_rgb, d1_rgb, d2_rgb, x_inp, x_len) in enumerate(train_loader):
        batch_size = x_inp.size(0) 
        tgt_rgb = tgt_rgb.to(device).float()
        d1_rgb = d1_rgb.to(device).float()
        d2_rgb = d2_rgb.to(device).float()
        x_inp = x_inp.to(device)
        x_len = x_len.to(device)

        # obtain predicted rgb
        tgt_score = sup_img(tgt_rgb, x_inp, x_len)
        d1_score = sup_img(d1_rgb, x_inp, x_len)
        d2_score = sup_img(d2_rgb, x_inp, x_len)

        # loss between actual and predicted rgb: Mean Squared Loss !!
        loss = F.cross_entropy(torch.cat([tgt_score,d1_score,d2_score], 1), torch.LongTensor(np.zeros(batch_size)))
        

        loss_meter.update(loss.item(), batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix({'loss': loss_meter.avg})
        pbar.update()
    pbar.close()
        
    if epoch % 10 == 0:
        print('====> Train Epoch: {}\tLoss: {:.4f}'.format(epoch, loss_meter.avg))
    
    return loss_meter.avg


def test(epoch, sup_emb, sup_img, test_loader,device,optimizer):
    sup_emb.eval()
    sup_img.eval()

    with torch.no_grad():
        loss_meter = AverageMeter()

        pbar = tqdm(total=len(test_loader))
        for batch_idx, (tgt_rgb, d1_rgb, d2_rgb, x_inp, x_len) in enumerate(test_loader):
            batch_size = x_inp.size(0)
            tgt_rgb = tgt_rgb.to(device).float()
            d1_rgb = d1_rgb.to(device).float()
            d2_rgb = d2_rgb.to(device).float()
            x_inp = x_inp.to(device)
            x_len = x_len.to(device)

            # obtain predicted rgb
            tgt_score = sup_img(tgt_rgb, x_inp, x_len)
            d1_score = sup_img(d1_rgb, x_inp, x_len)
            d2_score = sup_img(d2_rgb, x_inp, x_len)

            loss = F.cross_entropy(torch.cat([tgt_score,d1_score,d2_score], 1), torch.LongTensor(np.zeros(batch_size)))

            loss_meter.update(loss.item(), batch_size)
            
            pbar.update()
        pbar.close()
        if epoch % 10 == 0:
            print('====> Test Epoch: {}\tLoss: {:.4f}'.format(epoch, loss_meter.avg))
                
    return loss_meter.avg

