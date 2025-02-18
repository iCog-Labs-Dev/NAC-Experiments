import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim[0])
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc_mu = nn.Linear(hidden_dim[1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim[1], latent_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in [self.fc1, self.fc2, self.fc_mu, self.fc_logvar]:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.055)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, input_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim[1])
        self.fc2 = nn.Linear(hidden_dim[1], hidden_dim[0])
        self.fc3 = nn.Linear(hidden_dim[0], input_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in[self.fc1, self.fc2, self.fc3]:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.055)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = torch.sigmoid(self.fc3(z))
        return z
    
class Discriminator(nn.Module):
    def __init__(self, latent_dim, hidden_dims):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], 1)

        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.055)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = torch.sigmoid(self.fc3(z))
        return z
    
class GANAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, l2_lambda):
        super(GANAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, input_dim)
        self.discriminator = Discriminator(latent_dim, hidden_dims)
        self.l2_lambda = l2_lambda

    def compute_l2_penalty(self):
        l2_penalty = 0
        for param in self.decoder.parameters():
            if param.requires_grad:
                l2_penalty += torch.sum(param**2)
        return self.l2_lambda * l2_penalty
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        real_or_fake = self.discriminator(z)
        l2_penalty = self.compute_l2_penalty()
        return x_recon, real_or_fake, mu, logvar, l2_penalty