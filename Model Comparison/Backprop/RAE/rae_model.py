import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten image
        z = self.model(x)
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.Linear(512, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, z):
        x_recon = self.model(z)
        x_recon = x_recon.view(-1, 1, 28, 28)  # Reshape to image dimensions
        return x_recon

class RegularizedAutoencoder(nn.Module):
    def __init__(self, latent_dim, learning_rate=0.001, l2_lambda=1e-5):
        super(RegularizedAutoencoder, self).__init__() 
        self.encoder = Encoder(latent_dim) 
        self.decoder = Decoder(latent_dim)
        self.reconstruction_loss = nn.MSELoss()
        self.reconstruction_loss = nn.BCELoss() 
        self.nll_loss = nn.NLLLoss()
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), 
            lr=learning_rate, 
            weight_decay=l2_lambda
        )
    def forward(self, x):
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x

    def compute_loss(self, recon_x, x):
        return self.reconstruction_loss(recon_x, x)