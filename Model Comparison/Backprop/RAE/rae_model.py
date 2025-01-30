import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], latent_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in [self.fc1, self.fc2, self.fc3]:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        z = F. relu(self.fc3(x))
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, input_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dims[1])
        self.fc2 = nn.Linear(hidden_dims[1], hidden_dims[0])
        self.fc3 = nn.Linear(hidden_dims[0], input_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in [self.fc1, self.fc2, self.fc3]:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        z = F. sigmoid(self.fc3(x))
        return z
    
class RegularizedAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, l2_lambda):
        super(RegularizedAutoencoder, self).__init__() 
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim) 
        self.decoder = Decoder(latent_dim, hidden_dims, input_dim)
        self.l2_lambda = l2_lambda

    def compute_l2_penalty(self):
        l2_penalty = 0
        for param in self.decoder.parameters():
            if param.requires_grad:
                l2_penalty += torch.sum(param**2)
        return self.l2_lambda * l2_penalty

    def forward(self, x):
        z = self.encoder(x)
        recon_x = self.decoder(z)
        l2_penalty = self.compute_l2_penalty()
        return recon_x, l2_penalty