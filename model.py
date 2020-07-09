import torch
import torch.nn as nn
import torch.nn.functional as F

class VAEModel(nn.Module):
    def __init__(self, image_size=32, h_dim=512, z_dim=64):
        super(VAEModel, self).__init__()

        self.h_dim = h_dim
        self.z_dim = z_dim
        self.img_size = image_size
        
        self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 5, 2, 2),       # [3 x 32 x 32] -> [64 x 16 x 16]
                nn.BatchNorm2d(64),
                nn.ReLU(inplace = True),
                nn.Conv2d(64, 128, 5, 2, 2),     # [64 x 16 x 16] -> [128 x 8 x 8]
                nn.BatchNorm2d(128),
                nn.ReLU(inplace = True),
                nn.Conv2d(128, 256, 5, 2, 2),    # [128 x 8 x 8] -> [256 x 4 x 4]
                nn.BatchNorm2d(256),
                nn.ReLU(inplace = True),
                nn.Conv2d(256, 512, 5, 2, 2),    # [256 x 4 x 4] -> [512 x 2 x 2]
                nn.BatchNorm2d(512),
                nn.ReLU(inplace = True)
            )
        self.x2z = nn.Sequential(
                nn.Linear(512 * (self.img_size//16) * (self.img_size//16), self.h_dim*2),
                nn.Linear(self.h_dim*2, self.h_dim),
                nn.BatchNorm1d(self.h_dim)
            )
        self.mean = nn.Linear(self.h_dim, self.z_dim)
        self.var = nn.Linear(self.h_dim, self.z_dim)
        self.z2x = nn.Sequential(
                nn.Linear(self.z_dim, self.h_dim),
                nn.Linear(self.h_dim, 256*(self.img_size//8)*(self.img_size//8))
        )
        self.decoder = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 256, 5, 2, 2, output_padding=1),  # [256 x 4 x 4] -> [256 x 8 x 8]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 5, 2, 2, output_padding=1),  # [256 x 8 x 8] -> [128 x 16 x 16]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 32, 5, 2, 2, output_padding=1),   # [128 x 16 x 16] -> [32 x 32 x 32]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 5, 1, 2),     # [32 x 32 x 32] -> [3 x 32 x 32]
            #nn.Tanh()
        )
        
    # encoder
    def encode(self, x):
        h = self.encoder(x).view(x.shape[0], -1)
        h = self.x2z(h)
        return self.mean(h), self.var(h)
    
    # generate Gaussian distribution
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    # decoder
    def decode(self, z):
        x = self.z2x(z).view(-1, 256, (self.img_size//8), (self.img_size//8))
        x = self.decoder(x)
        return x
    
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var