# General Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.cluster import KMeans
from scipy.stats import entropy
from sklearn.decomposition import PCA
from minst_data import get_mnist_loaders

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

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Hyperparameters
latent_dim = 64
batch_size = 128
learning_rate = 0.0002
num_epochs = 50  


train_loader, test_loader = get_mnist_loaders(batch_size=batch_size) 


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

# Initialize Models
encoder = Encoder(latent_dim).to(device)
decoder = Decoder(latent_dim).to(device)
discriminator = Discriminator().to(device)

# Function to Count Trainable Parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'Total parameters in Encoder: {count_parameters(encoder)}')
print(f'Total parameters in Decoder: {count_parameters(decoder)}')
print(f'Total parameters in Discriminator: {count_parameters(discriminator)}')

# Loss Functions
adversarial_loss = nn.BCELoss()
reconstruction_loss = nn.MSELoss()

# Optimizers
optimizer_G = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

# Lists to Store Losses
g_losses = []
d_losses = []
gkl_list = []
ndb_list = []

# Evaluation Functions

def compute_gkl_pca(real_samples, generated_samples, n_components=50, epsilon=1e-6):
    """
    Computes the Gaussian KL Divergence between real and generated samples after PCA.

    Args:
        real_samples (torch.Tensor): Real images tensor of shape [N, C, H, W].
        generated_samples (torch.Tensor): Generated images tensor of shape [N, C, H, W].
        n_components (int): Number of PCA components.
        epsilon (float): Small value to add to the diagonal for numerical stability.
        
    Returns:
        float: G-KL divergence value.
    """
    # Flatten the images and convert to NumPy
    real_flat = real_samples.view(real_samples.size(0), -1).cpu().numpy()
    gen_flat = generated_samples.view(generated_samples.size(0), -1).cpu().numpy()
    
    # Perform PCA
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(real_flat)
    real_pca = pca.transform(real_flat)
    gen_pca = pca.transform(gen_flat)
    
    # Compute means
    mean_real = np.mean(real_pca, axis=0)
    mean_gen = np.mean(gen_pca, axis=0)
    
    # Compute covariance matrices
    cov_real = np.cov(real_pca, rowvar=False) + epsilon * np.eye(n_components)
    cov_gen = np.cov(gen_pca, rowvar=False) + epsilon * np.eye(n_components)
    
    # Compute KL Divergence between two Gaussians
    try:
        inv_cov_gen = np.linalg.inv(cov_gen)
        diff = mean_gen - mean_real
        term1 = np.trace(inv_cov_gen @ cov_real)
        term2 = diff.T @ inv_cov_gen @ diff
        term3 = -n_components
        term4 = np.log((np.linalg.det(cov_gen) + epsilon) / (np.linalg.det(cov_real) + epsilon))
        gkl = 0.5 * (term1 + term2 + term3 + term4)
        return gkl
    except np.linalg.LinAlgError as e:
        print(f"LinAlgError during G-KL computation: {e}")
        return float('inf')
    except Exception as e:
        print(f"Error computing G-KL with numpy: {e}")
        return float('inf')

def compute_ndb(real_samples, generated_samples, num_bins=20, threshold_ratio=0.05):
    """
    Computes the Number of Statistically Different Bins (NDB) between real and generated samples.

    Args:
        real_samples (torch.Tensor): Real images tensor of shape [N, C, H, W].
        generated_samples (torch.Tensor): Generated images tensor of shape [N, C, H, W].
        num_bins (int): Number of clusters/bins for K-Means.
        threshold_ratio (float): Ratio to determine significant difference.

    Returns:
        int: Number of statistically different bins.
    """
    # Flatten the images and convert to NumPy
    real_flat = real_samples.view(real_samples.size(0), -1).cpu().numpy()
    gen_flat = generated_samples.view(generated_samples.size(0), -1).cpu().numpy()
    
    # Combine data for K-Means clustering
    combined = np.vstack((real_flat, gen_flat))
    
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=num_bins, random_state=42)
    kmeans.fit(combined)
    
    # Assign clusters
    real_clusters = kmeans.predict(real_flat)
    gen_clusters = kmeans.predict(gen_flat)
    
    # Count frequencies
    real_counts = np.bincount(real_clusters, minlength=num_bins)
    gen_counts = np.bincount(gen_clusters, minlength=num_bins)
    
    # Define a threshold for statistical difference
    threshold = threshold_ratio * real_counts
    differences = np.abs(real_counts - gen_counts)
    
    # Count bins where difference exceeds threshold
    ndb = np.sum(differences > threshold)
    return ndb

# Training Loop
for epoch in range(num_epochs):
    encoder.train()
    decoder.train()
    discriminator.train()
    for i, (imgs, _) in enumerate(train_loader):
        real_imgs = imgs.to(device)
        batch_size_i = real_imgs.size(0)

        # Labels
        valid = torch.ones(batch_size_i, 1, device=device)
        fake = torch.zeros(batch_size_i, 1, device=device)

        #  Train Generator
        optimizer_G.zero_grad()

        z = encoder(real_imgs)
        recon_imgs = decoder(z)

        g_adv_loss = adversarial_loss(discriminator(recon_imgs), valid)
        g_recon_loss = reconstruction_loss(recon_imgs, real_imgs)
        g_loss = 0.001 * g_adv_loss + 0.999 * g_recon_loss

        g_loss.backward()
        optimizer_G.step()

        #  Train Discriminator
        optimizer_D.zero_grad()

        # Discriminator losses
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(recon_imgs.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

        # Save losses for plotting
        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())

        # Print progress
        if (i+1) % 200 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch {i+1}/{len(train_loader)} \
                  Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}")

    # Save reconstructed images every epoch
    save_interval = 20  
    if (epoch + 1) % save_interval == 0 or epoch == num_epochs - 1:
        encoder.eval()
        decoder.eval()
        
        with torch.no_grad():
            sample = recon_imgs[:16]
            sample = sample * 0.5 + 0.5  
            grid = torchvision.utils.make_grid(sample, nrow=4)
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
            plt.title(f'Epoch {epoch+1}')
            plt.axis('off')
            plt.savefig(f'reconstructed_epoch_{epoch+1}.png')
            plt.close()  # Close the figure to free memory

# Plot loss curves after training
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(g_losses, label='Generator Loss')
plt.plot(d_losses, label='Discriminator Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_curves.png')
plt.close()

# Evaluation on test data
encoder.eval()
decoder.eval()
test_loss = 0
all_real = []
all_gen = []

with torch.no_grad():
    for imgs, _ in test_loader:
        real_imgs = imgs.to(device)
        z = encoder(real_imgs)
        recon_imgs = decoder(z)
        loss = reconstruction_loss(recon_imgs, real_imgs)
        test_loss += loss.item() * real_imgs.size(0)
        
        # Collect samples for G-KL and NDB
        all_real.append(real_imgs.cpu())
        all_gen.append(recon_imgs.cpu())

test_loss /= len(test_loader.dataset)
print(f'Test Reconstruction MSE Loss: {test_loss:.6f}')

# Concatenate all samples
all_real = torch.cat(all_real, dim=0)
all_gen = torch.cat(all_gen, dim=0)

# Compute G-KL Divergence using PCA
gkl = compute_gkl_pca(all_real, all_gen, n_components=50)
print(f'Gaussian KL Divergence (G-KL): {gkl:.6f}')

# Compute Number of Statistically Different Bins (NDB)
ndb = compute_ndb(all_real, all_gen, num_bins=20, threshold_ratio=0.05)
print(f'Number of Statistically Different Bins (NDB): {ndb}')

# Optionally Save Evaluation Metrics
with open('evaluation_metrics.txt', 'w') as f:
    f.write(f'Test Reconstruction MSE Loss: {test_loss:.6f}\n')
    f.write(f'Gaussian KL Divergence (G-KL): {gkl:.6f}\n')
    f.write(f'Number of Statistically Different Bins (NDB): {ndb}\n')

# Save a batch of original and reconstructed images from test set
with torch.no_grad():
    for imgs, _ in test_loader:
        real_imgs = imgs.to(device)
        z = encoder(real_imgs)
        recon_imgs = decoder(z)
        break  # Take only the first batch

    # Original images
    originals = real_imgs[:16]
    originals = originals * 0.5 + 0.5 
    grid_originals = torchvision.utils.make_grid(originals, nrow=4)
    plt.imshow(grid_originals.permute(1, 2, 0).cpu().numpy())
    plt.title('Original Images')
    plt.axis('off')
    plt.savefig('original_images.png')
    plt.close()

    # Reconstructed images
    reconstructions = recon_imgs[:16]
    reconstructions = reconstructions * 0.5 + 0.5  # Denormalize
    grid_reconstructions = torchvision.utils.make_grid(reconstructions, nrow=4)
    plt.imshow(grid_reconstructions.permute(1, 2, 0).cpu().numpy())
    plt.title('Reconstructed Images')
    plt.axis('off')
    plt.savefig('reconstructed_images_test.png')
    plt.close()

print("Training complete. Loss curves, evaluation metrics, and sample images have been saved.")
