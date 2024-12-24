import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, latent_dim)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten image
        z = self.model(x)
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 28 * 28),
            nn.Sigmoid()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z):
        x_recon = self.model(z)
        x_recon = x_recon.view(-1, 1, 28, 28)  # Reshape to image dimensions
        return x_recon

class RegularizedAutoencoder(nn.Module):
    def __init__(self, latent_dim, learning_rate=0.001, l2_lambda=1e-5):
        super(RegularizedAutoencoder, self).__init__() 
        self.encoder = Encoder(latent_dim) 
        self.decoder = Decoder(latent_dim)
        self.l2_lambda = l2_lambda
        self.optimizer = torch.optim.SGD(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), 
            lr=learning_rate, 
            weight_decay = self.l2_lambda
        )
    def forward(self, x):
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x