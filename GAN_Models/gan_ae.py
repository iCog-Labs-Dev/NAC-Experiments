import torch
import torch.nn as nn
import torch.optim as optim
import torchvision  # Add this line
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
matplotlib.use('Agg')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

latent_dim = 64
batch_size = 128
learning_rate = 0.0002
num_epochs = 1

# MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data',
                               train=True,
                               transform=transform,
                               download=True)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(True),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        x = x.view(-1, 28*28)
        z = self.model(x)
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 28*28),
            nn.Tanh()
        )

    def forward(self, z):
        x_recon = self.model(z)
        x_recon = x_recon.view(-1, 1, 28, 28)
        return x_recon

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 28*28)
        validity = self.model(x)
        return validity

encoder = Encoder(latent_dim).to(device)
decoder = Decoder(latent_dim).to(device)
discriminator = Discriminator().to(device)

adversarial_loss = nn.BCELoss()
reconstruction_loss = nn.MSELoss()

optimizer_G = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(train_loader):
        real_imgs = imgs.to(device)
        batch_size_i = real_imgs.size(0)

        valid = torch.ones(batch_size_i, 1, device=device)
        fake = torch.zeros(batch_size_i, 1, device=device)

        optimizer_G.zero_grad()

        z = encoder(real_imgs)
        recon_imgs = decoder(z)

        g_loss = 0.001 * adversarial_loss(discriminator(recon_imgs), valid) + \
                 0.999 * reconstruction_loss(recon_imgs, real_imgs)

        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()

        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(recon_imgs.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

        if (i+1) % 200 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch {i+1}/{len(train_loader)} \
                  Loss D: {d_loss.item():.4f}, loss G: {g_loss.item():.4f}")
            
    with torch.no_grad():
        sample = recon_imgs[:16]
        sample = sample * 0.5 + 0.5  
        grid = torchvision.utils.make_grid(sample, nrow=4)
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.title(f'Epoch {epoch+1}')
        plt.savefig(f'reconstructed_epoch_{epoch+1}.png')  
        plt.close()  

