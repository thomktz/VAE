# %%
import torch.nn as nn
import torch
import numpy as np
import torchvision
import time
import glob
import torch.nn.functional as F
import matplotlib.pyplot as plt
from data_treatment import batch_size, image_size

in_channels = 3
latent_dims = 200
print_every = 200 #in batches
save_every = 15 #in epochs
log_loss_every = 5 #in batches
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
maxpool_indexes = [0,1]
# %% Model


class VAE(nn.Module):
    def __init__(self, layer_count = 3, zsize = 512):
        super(VAE, self).__init__()
        self.layer_count = layer_count
        self.zsize = zsize
        inputs = in_channels
        
        mul = 1
        for i in range(self.layer_count):
            setattr(self, "conv%d" % (i + 1), nn.Conv2d(inputs, image_size * mul, 4, 2, 1))
            setattr(self, "conv%d_bn" % (i + 1), nn.BatchNorm2d(image_size * mul))
            if i in maxpool_indexes:
                setattr(self, "maxpool%d" % (i + 1), nn.MaxPool2d(2,2))
            inputs = image_size * mul
            mul *= 2
        self.d_max = inputs
        self.fc1 = nn.Linear(inputs * 4 * 4, zsize)
        self.fc2 = nn.Linear(inputs * 4 * 4, zsize)
        self.d1 = nn.Linear(zsize, inputs * 4 * 4)

        mul = inputs // image_size // 2

        for i in range(1, self.layer_count):
            setattr(self, "deconv%d" % (i + 1), nn.ConvTranspose2d(inputs, image_size * mul, 4, 2, 1))
            setattr(self, "deconv%d_bn" % (i + 1), nn.BatchNorm2d(image_size * mul))
            if i-1 in maxpool_indexes:
                setattr(self, "upsample%d" % (i + 1), nn.Upsample(scale_factor = 2))
            inputs = image_size * mul
            mul //= 2
        setattr(self, "deconv%d" % (self.layer_count + 1), nn.ConvTranspose2d(inputs, in_channels, 4, 2, 1))

    def encode(self, x):
        for i in range(self.layer_count):
            #print( torch.cuda.memory_summary(device=None, abbreviated=False))
            x = F.relu(getattr(self, "conv%d_bn" % (i + 1))(getattr(self, "conv%d" % (i + 1))(x)))
            if i in maxpool_indexes:
                x = getattr(self, "maxpool%d" % (i + 1))(x)
            #print(x.shape)

        x = x.view(x.shape[0], self.d_max * 4 * 4)
        h1 = self.fc1(x)
        h2 = self.fc2(x)
        return h1, h2

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, x):
        x = x.view(x.shape[0], self.zsize)
        x = self.d1(x)
        x = x.view(x.shape[0], self.d_max, 4, 4)
        #x = self.deconv1_bn(x)
        x = F.leaky_relu(x, 0.2)

        for i in range(1, self.layer_count):
            x = F.leaky_relu(getattr(self, "deconv%d_bn" % (i + 1))(getattr(self, "deconv%d" % (i + 1))(x)), 0.2)
            if i-1 in maxpool_indexes:
                x = getattr(self, "upsample%d" % (i + 1))(x)

        x = torch.tanh(getattr(self, "deconv%d" % (self.layer_count + 1))(x))
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        mu = mu.squeeze()
        logvar = logvar.squeeze()
        z = self.reparameterize(mu, logvar)
        return self.decode(z.view(-1, self.zsize, 1, 1)), mu, logvar

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()



# %% Save model state

