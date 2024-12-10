import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE_GMM(nn.Module):
    def __init__(self, latent_dim, n_components):
        super(VAE_GMM, self).__init__()
        self.latent_dim = latent_dim
        self.n_components = n_components
        
        # Encoder
        self.fc1 = nn.Linear(28 * 28, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        
        # GMM parameters
        self.pi = nn.Parameter(torch.ones(n_components) / n_components)
        self.mu = nn.Parameter(torch.randn(n_components, latent_dim) * 0.01)
        self.logvar = nn.Parameter(torch.randn(n_components, latent_dim) * 0.01)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 28 * 28)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # GMM sampling
        gmm_idx = torch.multinomial(self.pi, z.size(0), replacement=True)
        gmm_mu = self.mu[gmm_idx]
        gmm_logvar = self.logvar[gmm_idx]
        gmm_std = torch.exp(0.5 * gmm_logvar)
        gmm_eps = torch.randn_like(gmm_std)
        z_gmm = gmm_mu + gmm_eps * gmm_std
        
        return z_gmm

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 28 * 28))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar