import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=360, latent_dim=20):
        super(Encoder, self).__init__()

        # Four layers in the encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # Input to hidden layer 1
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Hidden layer 1 to hidden layer 2
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)  # Hidden layer 2 to hidden layer 3
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # Hidden layer 3 to mean

        self._init_weights()

    def _init_weights(self, sigma=0.1):
        # Gaussian initialization with tunable sigma
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

        # Four layers in the decoder
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self, sigma=0.1):
        # Gaussian initialization with tunable sigma
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.normal_(layer.weight, mean=0.0, std=sigma)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = torch.sigmoid(self.fc3(z))  # Sigmoid activation in the output layer
        return z


# GVAE-CV
class GVAE(nn.Module):
    def __init__(
        self,
        input_dim=784,
        hidden_dim=360,
        latent_dim=20,
        fixed_logvar=torch.tensor(0.0),
    ):
        super(GVAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.fixed_logvar = fixed_logvar

    def reparameterize(self, mu):
        std = torch.exp(0.5 * self.fixed_logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        mu = self.encoder(x)
        z = self.reparameterize(mu)
        recon_x = self.decoder(z)
        return recon_x, mu
