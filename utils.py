import inspect
import math
import os
import csv

import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np



def train(e, model, optimizer, train_loader, C):
    device = torch.device("cuda")
    model.train()
    
    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    #criterion = nn.MSELoss()
    criterion = criterion.to(device)

    total_loss = 0
    total_recon_loss = 0
    total_KL_loss = 0
    for step, batch in enumerate(train_loader):
        input, _ = batch
        input = input.to(device)
        optimizer.zero_grad()
        
        result, mu, log_var = model(input)
        recon_loss = criterion(result, input)
        #recon_loss = 0.5 * torch.sum(input - mu.pow(2)) / (2 * math.pow(C.std, 2)) + math.log(C.std)
        KL_loss = - C.KL_weight * 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + KL_loss

        loss.backward()
        optimizer.step()
        
        total_loss += loss
        total_recon_loss += recon_loss
        total_KL_loss += KL_loss
        print("[Epoch #{} [{}/{}]] loss: {} recon_loss: {} KL_loss: {}".format(e, step, len(train_loader), loss, recon_loss, KL_loss))

    total_loss = total_loss / len(train_loader)
    total_recon_loss = total_recon_loss / len(train_loader)
    total_KL_loss = total_KL_loss / len(train_loader)
    loss = {
        'recon': total_recon_loss,
        'KL': total_KL_loss,
        'total': total_loss
    }
    return loss

def gen_sample(e, z, x, model, C):
    if not os.path.exists(C.sample_dpath):
        os.makedirs(C.sample_dpath)

        os.makedirs(os.path.join(C.sample_dpath, 'sample'))
        os.makedirs(os.path.join(C.sample_dpath, 'reconst'))
        save_image(x, os.path.join(C.sample_dpath, 'real.png'))

    sample_dir = os.path.join(C.sample_dpath, 'sample')
    reconst_dir = os.path.join(C.sample_dpath, 'reconst')
    

    model.eval()
    out = model.decode(z).view(-1, 3, C.img_size, C.img_size)
    save_image(out, os.path.join(sample_dir, 'sampled-{}.png'.format(e)))
    
    out, _, _ = model(x)
    save_image(out, os.path.join(reconst_dir, 'reconstruction-{}.png'.format(e)))


def load_checkpoint(model, ckpt_fpath):
    checkpoint = torch.load(ckpt_fpath)
    model.load_state_dict(checkpoint['vae'])
    return model


def save_checkpoint(e, model, ckpt_fpath, config):
    ckpt_dpath = os.path.dirname(ckpt_fpath)
    if not os.path.exists(ckpt_dpath):
        os.makedirs(ckpt_dpath)

    torch.save({
        'epoch': e,
        'vae': model.state_dict(),
    }, ckpt_fpath)

def cal_accuracy(result, label):
    m = nn.Sigmoid()
    pred = (m(result) >= 0.5).float()
    acc = pred.eq(label).sum().float() / len(label)
    return acc

