import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
import random
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Set Random Seeds for Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data
train_x = np.load('mnist/trainX.npy')
train_y = np.load('mnist/trainY.npy')
test_x = np.load('mnist/testX.npy')
test_y = np.load('mnist/testY.npy')
valid_x = np.load('mnist/validX.npy')
valid_y = np.load('mnist/validY.npy')

# Convert to PyTorch tensors
train_x = torch.tensor(train_x, dtype=torch.float32).to(device)
train_y = torch.tensor(train_y, dtype=torch.float32).to(device)
test_x = torch.tensor(test_x, dtype=torch.float32).to(device)
test_y = torch.tensor(test_y, dtype=torch.float32).to(device)
valid_x = torch.tensor(valid_x, dtype=torch.float32).to(device)
valid_y = torch.tensor(valid_y, dtype=torch.float32).to(device)

# Dataloaders
batch_size = 128
train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(TensorDataset(valid_x, valid_y), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=False)


# Hyperparameters
learning_rate = 0.0002
num_epochs = 100
latent_dim = 20
fixed_variance = 1

# Define the VAE Model with Constant Variance
class CV_VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=400):
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

# Initialize model
model = CV_VAE(input_dim=784, latent_dim=latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
bce_loss = nn.BCELoss(reduction='sum')

# Define the loss function for training
def loss_function(recon_x, x, mu):
    BCE = bce_loss(recon_x, x)
    KLD = 0.5 * torch.sum(mu.pow(2))
    return BCE + KLD, BCE, KLD

# Training Loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for x, _ in train_loader:
        x = x.to(device) 
        optimizer.zero_grad()
        recon_x, mu = model(x)
        loss, bce, kld = loss_function(recon_x, x, mu)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss / len(train_loader.dataset)}")

print("Training complete.")

def compute_ndb(real_data, recon_data, num_bins=20, threshold_ratio=0.05):
    real_data = real_data.view(real_data.size(0), -1).numpy()
    recon_data = recon_data.view(recon_data.size(0), -1).numpy()
    kmeans = KMeans(n_clusters=num_bins, random_state=0).fit(real_data)
    real_bins = kmeans.predict(real_data)
    recon_bins = kmeans.predict(recon_data)
    
    real_counts = np.bincount(real_bins, minlength=num_bins)
    recon_counts = np.bincount(recon_bins, minlength=num_bins)

    threshold = threshold_ratio * len(real_data)
    
    empty_real_bins = np.sum(real_counts < threshold)
    empty_recon_bins = np.sum(recon_counts < threshold)

    # NDB Score: Sum of empty bins in both real and recon data distributions
    ndb_score = empty_real_bins + empty_recon_bins

    return ndb_score




# Evaluation Function
def evaluate_model(model, data_loader, device, num_bins=20, threshold_ratio=0.05):
    model.eval()
    total_mse = 0
    total_bce = 0
    total_kld = 0
    all_real = []
    all_recon = []

    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            recon_x, mu = model(x)
            
            # MSE Loss
            mse = mean_squared_error(x.cpu().numpy(), recon_x.cpu().numpy())
            total_mse += mse * x.size(0)
          
            
            # BCE Loss
            bce = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum').item()
            total_bce += bce
            
            # KLD Loss
            kld = 0.5 * torch.sum(mu.pow(2)).item()
            total_kld += kld
            
            # NDB computation
            all_real.append(x.cpu())
            all_recon.append(recon_x.cpu())

    # Calculate average losses
    avg_mse = total_mse / len(data_loader.dataset)
    avg_bce = total_bce / len(data_loader.dataset)
    avg_kld = total_kld / len(data_loader.dataset)
    
 
    all_real = torch.cat(all_real, dim=0)
    all_recon = torch.cat(all_recon, dim=0)

    # Compute NDB
    ndb = compute_ndb(all_real, all_recon, num_bins=num_bins, threshold_ratio=threshold_ratio)

    print(f"Evaluation Results:\nMSE: {avg_mse:.6f}\nBCE: {avg_bce:.6f}\nKLD: {avg_kld:.6f}\nNDB: {ndb}")
    return avg_mse, avg_bce, avg_kld, ndb

def display_reconstructed_images(model, data_loader, device, num_images=10):
    model.eval()
    with torch.no_grad():
        # Get a batch of images
        for x, _ in data_loader:
            x = x.to(device)
            recon_x, _ = model(x)
            break  # Only take the first batch

    # Move to CPU and detach to convert to NumPy arrays for plotting
    original_images = x[:num_images].cpu().numpy()
    reconstructed_images = recon_x[:num_images].cpu().numpy()

    # Plot the original and reconstructed images side by side
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 1.5, 3))
    for i in range(num_images):
        # Display original images
        axes[0, i].imshow(original_images[i].reshape(28, 28), cmap="gray")
        axes[0, i].axis("off")
        axes[0, i].set_title("Original")

        # Display reconstructed images
        axes[1, i].imshow(reconstructed_images[i].reshape(28, 28), cmap="gray")
        axes[1, i].axis("off")
        axes[1, i].set_title("Reconstructed")

    plt.show()


evaluate_model(model, test_loader, device, num_bins=20, threshold_ratio=0.05)
display_reconstructed_images(model, data_loader=test_loader, device=device)        
