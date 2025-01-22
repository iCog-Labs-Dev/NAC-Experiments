import argparse
import os
import numpy as np
import math
import sys
import random
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import logging

# [Your provided setup code for argparse and data loading remains the same]
# Adjust sys.argv to remove unwanted Jupyter arguments
sys.argv = sys.argv[:1]  # Keep only the script name, remove Jupyter's arguments

# Now proceed with argparse as usual
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=200, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=8, help="number of CPU threads for data loading")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=2000, help="interval between image sampling")

# Parse the arguments
opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)
print("this is the image shape ", img_shape)
cuda = True if torch.cuda.is_available() else False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  
)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if cuda:
    torch.cuda.manual_seed(42)



# Data Loading Classes and Functions
class NumpyDataset(Dataset):
    def __init__(self, dataX, dataY=None):
        self.dataX = np.load(dataX)
        self.dataY = np.load(dataY) if dataY is not None else None

    def __len__(self):
        return len(self.dataX)

    def __getitem__(self, idx):
        data = torch.tensor(self.dataX[idx], dtype=torch.float32)
        label = (
            torch.tensor(self.dataY[idx], dtype=torch.long)
            if self.dataY is not None
            else None
        )
        return data, label


# Define file paths
dataX = "../../../data/mnist/trainX.npy"
dataY = "../../../data/mnist/trainY.npy"
devX = "../../../data/mnist/validX.npy"
devY = "../../../data/mnist/validY.npy"
testX = "../../../data/mnist/testX.npy"
testY = "../../../data/mnist/testY.npy"

# Create dataloaders
train_dataset = NumpyDataset(dataX, dataY)
train_loader = DataLoader(dataset=train_dataset, batch_size=200, shuffle=True)

dev_dataset = NumpyDataset(devX, devY)
dev_loader = DataLoader(dataset=dev_dataset, batch_size=200, shuffle=False)

test_dataset = NumpyDataset(testX, testY)
test_loader = DataLoader(dataset=test_dataset, batch_size=200, shuffle=False)



class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=20):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc_mean = nn.Linear(hidden_dim//2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim//2, latent_dim)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        mean = self.fc_mean(h2)
        logvar = self.fc_logvar(h2)
        return mean, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class Decoder(nn.Module):
    def __init__(self, latent_dim=20, hidden_dim=512, output_dim=784):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim//2)
        self.fc2 = nn.Linear(hidden_dim//2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        h1 = F.relu(self.fc1(z))
        h2 = F.relu(self.fc2(h1))
        output = torch.sigmoid(self.fc3(h2))
        return output.view(-1, opt.channels, opt.img_size, opt.img_size)

class Discriminator(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=512):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc3 = nn.Linear(hidden_dim//2, 1)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        h1 = F.relu(self.fc1(z))
        h2 = F.relu(self.fc2(h1))
        return torch.sigmoid(self.fc3(h2))

# Initialize models, optimizers, and loss functions
encoder = Encoder()
decoder = Decoder()
discriminator = Discriminator()

if cuda:
    encoder.cuda()
    decoder.cuda()
    discriminator.cuda()

# Optimizers
optimizer_E = torch.optim.Adam(encoder.parameters(), lr=0.002)
optimizer_D = torch.optim.Adam(decoder.parameters(), lr=0.002, weight_decay=1e-5)  # L2 regularization
optimizer_Disc = torch.optim.Adam(discriminator.parameters(), lr=0.002)

# Loss functions
BCE_loss = nn.BCELoss()
if cuda:
    BCE_loss = BCE_loss.cuda()

def train_epoch(epoch):
    encoder.train()
    decoder.train()
    discriminator.train()
    
    total_disc_loss = 0
    total_gen_loss = 0
    total_rec_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        batch_size = data.size(0)
        real = Variable(torch.ones(batch_size, 1))
        fake = Variable(torch.zeros(batch_size, 1))

        if cuda:
            data = data.cuda()
            real = real.cuda()
            fake = fake.cuda()

        # Train Discriminator
        optimizer_Disc.zero_grad()
        
        # Real latent vectors
        z_real = Variable(torch.randn(batch_size, 20))
        if cuda:
            z_real = z_real.cuda()
            
        # Generate fake latent vectors
        mu, logvar = encoder(data)
        z_fake = encoder.reparameterize(mu, logvar)
        
        # Discriminator losses
        d_real = discriminator(z_real)
        d_fake = discriminator(z_fake.detach())
        d_loss_real = BCE_loss(d_real, real)
        d_loss_fake = BCE_loss(d_fake, fake)
        d_loss = (d_loss_real + d_loss_fake) / 2
        
        d_loss.backward()
        optimizer_Disc.step()

        # Train Generator (Encoder)
        optimizer_E.zero_grad()
        optimizer_D.zero_grad()
        
        # Generate fake latent vectors
        mu, logvar = encoder(data)
        z_fake = encoder.reparameterize(mu, logvar)
        
        # Reconstruction
        recon = decoder(z_fake)
        
        # Generator loss (fool discriminator)
        g_loss = BCE_loss(discriminator(z_fake), real)
        
        # Reconstruction loss
        rec_loss = BCE_loss(recon, data)
        
        # Total generator loss
        total_loss = rec_loss + 0.001 * g_loss
        
        total_loss.backward()
        optimizer_E.step()
        optimizer_D.step()

        # Record losses
        total_disc_loss += d_loss.item()
        total_gen_loss += g_loss.item()
        total_rec_loss += rec_loss.item()

        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx}/{len(train_loader)}] '
                  f'D_loss: {d_loss.item():.4f} '
                  f'G_loss: {g_loss.item():.4f} '
                  f'Rec_loss: {rec_loss.item():.4f}')

    return total_disc_loss/len(train_loader), total_gen_loss/len(train_loader), total_rec_loss/len(train_loader)

def evaluate(loader):
    encoder.eval()
    decoder.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data, _ in loader:
            if cuda:
                data = data.cuda()
            
            mu, logvar = encoder(data)
            z = encoder.reparameterize(mu, logvar)
            recon = decoder(z)
            
            loss = BCE_loss(recon, data)
            total_loss += loss.item()
            
    return total_loss / len(loader)

# Training loop
for epoch in range(opt.n_epochs):
    train_losses = train_epoch(epoch)
    val_loss = evaluate(dev_loader)
    
    print(f'Epoch {epoch}: Validation Loss: {val_loss:.4f}')