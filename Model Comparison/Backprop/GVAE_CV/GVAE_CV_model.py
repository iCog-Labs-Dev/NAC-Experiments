import torch
import torch.nn as nn

fixed_variance = 0.1

class CV_VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(CV_VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        mu = self.fc_mu(h1)
        return mu

    def reparameterize(self, mu):
        std = fixed_variance
        eps = torch.randn_like(mu)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu = self.encode(x)
        z = self.reparameterize(mu)
        return self.decode(z), mu