import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim[0])  # Input to hidden layer 1
        self.fc2 = nn.Linear(
            hidden_dim[0], hidden_dim[1]
        )  # Hidden layer 1 to hidden layer 2
        self.fc_mu = nn.Linear(hidden_dim[1], latent_dim)  # Hidden layer 2 to mean
        self.fc_logvar = nn.Linear(
            hidden_dim[1], latent_dim
        )  # Hidden layer 2 to log-variance

        self._init_weights()

    def _init_weights(self, sigma=0.1):
        for layer in [self.fc1, self.fc2, self.fc_mu, self.fc_logvar]:
            nn.init.normal_(layer.weight, mean=0.0, std=sigma)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(
            latent_dim, hidden_dim[0]
        )  # Latent space to hidden layer 1
        self.fc2 = nn.Linear(
            hidden_dim[0], hidden_dim[1]
        )  # Hidden layer 1 to hidden layer 2
        self.fc3 = nn.Linear(hidden_dim[1], output_dim)  # Hidden layer 2 to output

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


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar
