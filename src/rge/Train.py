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

from utils import (AverageMeter, save_checkpoint)


def train(epoch, sup_img,train_loader,device,optimizer):
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
        loss = F.cross_entropy(torch.cat([tgt_score,d1_score,d2_score], 1), torch.LongTensor(np.zeros(batch_size)).to(device))

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


def test(epoch, sup_img, test_loader,device,optimizer):
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

            loss = F.cross_entropy(torch.cat([tgt_score,d1_score,d2_score], 1), torch.LongTensor(np.zeros(batch_size)).to(device))

            loss_meter.update(loss.item(), batch_size)
            
            pbar.update()
        pbar.close()
        if epoch % 10 == 0:
            print('====> Test Epoch: {}\tLoss: {:.4f}'.format(epoch, loss_meter.avg))
                
    return loss_meter.avg

