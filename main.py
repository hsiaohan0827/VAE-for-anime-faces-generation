import os
import argparse
import importlib
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

from model import VAEModel
from utils import train, gen_sample, save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs.config", help="module path of the configuration file.")
    return parser.parse_args()

def build_model(C):
    VAE = VAEModel(C.img_size, C.vae.h_dim, C.vae.z_dim)
    print(VAE)
    return VAE

def log_train(C, summary_writer, e, loss, lr):
    summary_writer.add_scalar(C.tx_train_recon_loss, loss['recon'], e)
    summary_writer.add_scalar(C.tx_train_KL_loss, loss['KL'], e)
    summary_writer.add_scalar(C.tx_train_loss, loss['total'], e)
    print("[Epoch #{}] loss: {} recon_loss: {} KL_loss: {}".format(e, loss['total'], loss['recon'], loss['KL']))

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

if __name__ == '__main__':

    # set config
    args = parse_args()
    C = importlib.import_module(args.config).TrainConfig
    print("MODEL ID: {}".format(C.model_id))

    # create dataloader
    train_data = ImageFolder('./data',
                            transform = transforms.Compose([
                                        transforms.Resize((C.img_size, C.img_size)),
                                        transforms.RandomHorizontalFlip(0.5),
                                        transforms.ToTensor(),
                                        #transforms.Normalize(mean = (0.5, 0.5, 0.5), 
                                        #                     std = (0.5, 0.5, 0.5))
                                        ])
                            )
    train_loader = Data.DataLoader(dataset=train_data, batch_size=C.batch_size, shuffle=True)

    print('train data: '+str(len(train_data)))

    # logs
    summary_writer = SummaryWriter(C.log_dpath)

    if C.isTrain:

        # model
        model = build_model(C)

        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=C.lr, weight_decay=C.weight_decay)
        #optimizer = torch.optim.SGD(model.parameters(), lr=C.lr)
        
        device = torch.device("cuda")
        model = model.to(device)

        gen_sample_synth = torch.randn(C.batch_size, C.vae.z_dim).to(device)
        for first in train_loader:
            gen_sample_recon = first[0].to(device)
            break
        for e in range(1, C.epochs + 1):
            ckpt_fpath = C.ckpt_fpath_tpl.format(e)

            """ Train """
            print("\n")
            train_loss = train(e, model, optimizer, train_loader, C)
            log_train(C, summary_writer, e, train_loss, get_lr(optimizer))

            """ Sample """
            if e % C.sample_img_every == 0 or e == 1:            
                gen_sample(e, gen_sample_synth, gen_sample_recon, model, C)

            """ Save model """
            if e >= C.save_from and e % C.save_every == 0:
                print("Saving checkpoint at epoch={} to {}".format(e, ckpt_fpath))
                save_checkpoint(e, model, ckpt_fpath, C)

    else:
        # load model
        model = build_model(C)
        model = load_checkpoint(model, C.ckpt_fpath)
        
        device = torch.device("cuda")
        model = model.to(device)

        model.eval()

        transform = transforms.Compose([
                    transforms.Resize((C.img_size, C.img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean = (0.5, 0.5, 0.5), 
                                            std = (0.5, 0.5, 0.5))
                    ])

        print('----- {}.png | {}.png choosen -----'.format(C.img1_num, C.img2_num))
        img1 = Image.open(C.img1_path)
        img1 = transform(img1).to(device)

        img2 = Image.open(C.img2_path)
        img2 = transform(img2).to(device)

        mu1, log_var1 = model.encode(img1.unsqueeze(0))
        mu2, log_var2 = model.encode(img2.unsqueeze(0))
        img1_latent = model.reparameterize(mu1, log_var1)
        img2_latent = model.reparameterize(mu2, log_var2)

        input = img1_latent
        for i in range(C.interpolate_num-1):
            latent = torch.lerp(img1_latent, img2_latent, float(i+1)/C.interpolate_num)
            input = torch.cat((input, latent), 0)
        input = input.to(device)

        print('----- start generating interpolation -----')
        out = model.decode(input).view(-1, 3, C.img_size, C.img_size)
        save_image(out, 'interpolate-{}-{}.png'.format(C.img1_num, C.img2_num))
        print('----- FINISHED! -----')