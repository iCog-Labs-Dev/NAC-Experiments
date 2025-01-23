import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
import sys, getopt as gopt, time
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import subprocess
import os

class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=360, latent_dim=20):
        super(Encoder, self).__init__()

        
        self.fc1 = nn.Linear(input_dim, hidden_dim)  
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  
        self.fc3 = nn.Linear(hidden_dim, hidden_dim) 
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  

        self._init_weights()

    def _init_weights(self, sigma=0.1):
        for layer in [self.fc1, self.fc2, self.fc3, self.fc_mu]:
            nn.init.normal_(layer.weight, mean=0.0, std=sigma)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu = self.fc_mu(x)
        return mu


class Decoder(nn.Module):
    def __init__(self, latent_dim=20, hidden_dim=360, output_dim=784):
        super(Decoder, self).__init__()

        
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self, sigma=0.1):
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.normal_(layer.weight, mean=0.0, std=sigma)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = torch.sigmoid(self.fc3(z))
        return z


# GVAE-CV
class GVAE_CV(nn.Module):
    def __init__(
        self,
        input_dim=784,
        hidden_dim=360,
        latent_dim=20,
        fixed_variance=torch.tensor(0.0),
    ):
        super(GVAE_CV, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.fixed_variance = fixed_variance

    def reparameterize(self, mu):
        std = torch.exp(0.5 * self.fixed_variance)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        mu = self.encoder(x)
        z = self.reparameterize(mu)
        recon_x = self.decoder(z)
        return recon_x, mu
