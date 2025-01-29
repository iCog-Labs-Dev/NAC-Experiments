import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim, input_dim, hidden_dims):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], latent_dim),
            nn.Sigmoid()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.055)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  
        z = self.model(x)
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim, input_dim, hidden_dims):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim),
            nn.Sigmoid()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.055)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z):
        x_recon = self.model(z)
        x_recon = x_recon.view(-1, 1, 28, 28)  # Reshape to image dimensions
        return x_recon

class RegularizedAutoencoder(nn.Module):
    def __init__(self, latent_dim,input_dim, hidden_dims, learning_rate=0.1, l2_lambda=1e-3):
        super(RegularizedAutoencoder, self).__init__() 
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim) 
        self.decoder = Decoder(latent_dim, input_dim, hidden_dims)
        self.reconstruction_loss = nn.BCELoss()
        self.l2_lambda = l2_lambda
        self.optimizer = torch.optim.SGD(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), 
            lr=learning_rate
        )

    def compute_l2_penalty(self):
        l2_penalty = 0
        for param in self.decoder.parameters():
            if param.requires_grad:
                l2_penalty += torch.sum(param**2)
        return self.l2_lambda * l2_penalty

    def forward(self, x):
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x

    def compute_loss(self, x, recon_x):
        reconstruction_loss = self.reconstruction_loss(recon_x, x)
        l2_penalty = self.compute_l2_penalty()
        return -(reconstruction_loss + l2_penalty)