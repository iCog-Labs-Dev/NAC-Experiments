import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim[0])
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc_mu = nn.Linear(hidden_dim[1], latent_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in [self.fc1, self.fc2, self.fc_mu]:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        return mu

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, input_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim[1])
        self.fc2 = nn.Linear(hidden_dim[1], hidden_dim[0])
        self.fc3 = nn.Linear(hidden_dim[0], input_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in [self.fc1, self.fc2, self.fc3]:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = torch.sigmoid(self.fc3(z)) 
        return z
class GVAE_CV(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, l2_lambda, fixed_variance):
        super(GVAE_CV, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.l2_lambda = l2_lambda
        self.fixed_variance = fixed_variance
    
    def compute_l2_penalty(self):
        l2_penalty = 0
        for param in self.decoder.parameters():
            if param.requires_grad:
                l2_penalty += torch.sum(param**2)
        return self.l2_lambda * l2_penalty
    
    def reparameterize(self, mu):
        std = torch.exp(0.5 * self.fixed_variance)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        mu = self.encoder(x)
        z = self.reparameterize(mu)
        recon_x = self.decoder(z)
        l2_penalty = self.compute_l2_penalty()
        return recon_x, mu, z, l2_penalty
