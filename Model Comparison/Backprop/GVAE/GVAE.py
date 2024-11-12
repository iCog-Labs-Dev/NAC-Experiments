# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Function to set the seed for reproducibility
import random
def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # The below two lines are for deterministic algorithm behavior in CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the seed
set_seed()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=256, shuffle=True)

# Define the VAE model
class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(28 * 28, 400)
        self.fc21 = nn.Linear(400, latent_dim)  # Mean of latent variables
        self.fc22 = nn.Linear(400, latent_dim)  # Log-variance of latent variables
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 28 * 28)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 28 * 28))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function
beta = 1.0 
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 28 * 28), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    NLL = BCE
    return BCE, KLD, NLL

# Function to calculate accuracy
def calculate_accuracy(recon_x, x):
    # Convert probabilities to binary predictions
    preds = recon_x.view(-1, 28 * 28) > 0.5  # Thresholding at 0.5
    targets = x.view(-1, 28 * 28) > 0.5
    accuracy = (preds == targets).float().mean()  # Mean of correct predictions
    return accuracy.item()

# # Training the VAE with 2D latent space
latent_dim= 2           # <--note! only 2
# latent_dim= 4         
model = VAE(latent_dim=latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 100

# for epoch in range(num_epochs):
#     model.train()
#     train_loss = 0
#     for data, _ in trainloader:
#         data = data.to(device)
#         optimizer.zero_grad()
#         recon_batch, mu, logvar = model(data) ##
#         loss = loss_function(recon_batch, data, mu, logvar)
#         loss.backward()
#         train_loss += loss.item()
#         optimizer.step()
#     print(f'Epoch {epoch + 1}, Loss: {train_loss / len(trainloader.dataset)}')

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    total_accuracy = 0
    total_mse = 0
    total_bce = 0 
    total_kld = 0 
    total_nll = 0 
    for data, _ in trainloader:
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        bce, kld, nll = loss_function(recon_batch, data, mu, logvar)  # Get BCE, KLD, and NLL

        # Calculate accuracy and MSE
        accuracy = calculate_accuracy(recon_batch, data)
        mse = F.mse_loss(recon_batch, data.view(-1, 28 * 28), reduction='mean')

        total_accuracy += accuracy
        total_mse += mse.item()
        total_bce += bce.item()
        total_kld += kld.item()
        total_nll += nll.item()
        train_loss += bce + beta * kld 
        loss = bce + beta * kld
        loss.backward()
        optimizer.step()

    avg_loss = train_loss / len(trainloader.dataset)
    avg_accuracy = total_accuracy / len(trainloader)
    avg_mse = total_mse / len(trainloader)
    avg_bce = total_bce / len(trainloader.dataset) 
    avg_kld = total_kld / len(trainloader.dataset) 
    avg_nll = total_nll / len(trainloader.dataset) 

    print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}, '
          f'BCE: {avg_bce:.4f}, KLD: {avg_kld:.4f}, NLL: {avg_nll:.4f}, '
          f'Accuracy: {avg_accuracy:.4f}, MSE: {avg_mse:.4f}')

# Visualizing the latent space
model.eval()
all_z = []
all_labels = []
with torch.no_grad():
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    for data, labels in testloader:
        data = data.to(device)
        mu, logvar = model.encode(data.view(-1, 28 * 28))
        z = model.reparameterize(mu, logvar)
        z = z.cpu().numpy()

        all_z.append(z)
        all_labels.append(labels)

    all_z = np.concatenate(all_z)
    all_labels = np.concatenate(all_labels)

    plt.figure(figsize=(10, 8))
    plt.scatter(all_z[:, 0], all_z[:, 1], c=all_labels, cmap='tab10')
    plt.colorbar()
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.title('Latent Space Visualization')
    plt.show()

# Continuous value adjustment visualization
with torch.no_grad():
    grid_x = np.linspace(-3, 3, 20)
    grid_y = np.linspace(-3, 3, 20)
    figure = np.zeros((28 * 20, 28 * 20))
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            # Create a latent vector of size (1, 128) with all zeros
            z_sample = torch.zeros(1, latent_dim).to(device)
            # Set the first two dimensions to xi and yi
            z_sample[0, 0] = xi
            z_sample[0, 1] = yi

            # z_sample = torch.Tensor([[xi, yi]]).to(device)

            x_decoded = model.decode(z_sample)
            digit = x_decoded.view(28, 28).cpu().numpy()
            figure[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = digit

    plt.figure(figsize=(15, 15))
    plt.imshow(figure, cmap='gray')
    plt.title('Continuous Value Adjustment Visualization')
    plt.show()

