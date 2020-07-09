import os
import time


class VAEConfig:
    model_type = 'VAE' 
    h_dim = 400
    z_dim = 32

class TrainConfig:

    isTrain = True
    img_size = 32

    """ Model """
    vae = VAEConfig
    std = 0.01

    """ Optimization """
    epochs = 500
    batch_size = 64
    optimizer = 'Adam'
    lr = 1e-3
    weight_decay = 1e-5
    lr_decay_gamma = 0.5
    lr_decay_patience = 5
    lr_decay_start_from = 0

    """ Loss """
    KL_weight = 1
    recon = 'BCE'

    """ Testing """
    img1_num = 323
    img2_num = 267
    img1_path = os.path.join('data', 'img', str(img1_num)+'.png')
    img2_path = os.path.join('data', 'img', str(img2_num)+'.png')
    interpolate_num = 8

    """ ID """
    exp_id = vae.model_type
    batch_id = 'bs-{}'.format(batch_size)
    optim_id = optimizer + '-lr-{}'.format(lr)
    KL_id = 'KL-{}'.format(KL_weight)
    img_id = 'img-{}'.format(img_size)
    loss_id = recon
    h_id = 'h-{}'.format(vae.h_dim)
    z_id = 'z-{}'.format(vae.z_dim)
    timestamp = time.strftime("%y%m%d-%H:%M:%S", time.gmtime())
    model_id = "|".join([ exp_id, batch_id, optim_id, loss_id, img_id, KL_id, h_id, z_id, timestamp ])

    """ Log """
    log_dpath = "logs/{}".format(model_id)
    ckpt_dpath = os.path.join("checkpoints", model_id)
    ckpt_fpath_tpl = os.path.join(ckpt_dpath, "{}.ckpt")
    sample_dpath = os.path.join("sample", model_id)
    save_from = 1
    save_every = 10
    sample_img_every = 1

    """ TensorboardX """
    tx_train_recon_loss = "loss/recon"
    tx_train_KL_loss = "loss/KL"
    tx_train_loss = "loss/total"

    tx_lr = "params/lr"

    """ predict """
    ckpt = 'VAE|bs-64|Adam-lr-0.001|img-32|KL-1|h-512|z-64'
    epoch = 1000
    
    ckpt_fpath = os.path.join("checkpoints", ckpt)
    ckpt_fpath = os.path.join(ckpt_fpath, str(epoch)+'.ckpt')