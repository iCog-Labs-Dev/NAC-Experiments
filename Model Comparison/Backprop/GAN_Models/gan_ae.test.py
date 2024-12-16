import argparse
import os
import numpy as np
# import jax.numpy as jnp
import itertools
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn as nn
import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt 
from mnist_data import get_mnist_loaders
from sklearn.manifold import TSNE 
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.mixture import GaussianMixture


os.makedirs("images", exist_ok=True)
os.makedirs("models", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=200, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first-order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second-order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of CPU threads for data loading")
parser.add_argument("--latent_dim", type=int, default=10, help="dimensionality of the latent code")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=2000, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)
print("this is the imaage shape " ,img_shape)
cuda = True if torch.cuda.is_available() else False

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if cuda:
    torch.cuda.manual_seed(42)

def project_gradients_to_gaussian_ball(model, radius=5.0):
    for param in model.parameters():
        if param.grad is not None:
            norm = param.grad.norm(p=2)
            if norm > radius:
                param.grad.mul_(radius / norm)

def fit_gmm_to_latent_space(encoder, dataloader):
    encoder.eval()
    latents = []
    with torch.no_grad():
        for imgs, _ in dataloader:
            imgs = imgs.type(Tensor)
            _, mu, _ = encoder(imgs)  # mu is the latent code
            latents.append(mu.cpu().numpy())
    latents = np.concatenate(latents, axis=0)
    
    # Fit the Gaussian Mixture Model
    gmm = GaussianMixture(n_components=75, random_state=42)
    gmm.fit(latents)

    return gmm


class NumpyDataset(Dataset):
    def __init__(self, dataX, dataY=None):
        self.dataX = np.load(dataX).reshape(-1,1,28,28)
        self.dataY = np.load(dataY) if dataY is not None else None 

    def __len__(self):
        return len(self.dataX)

    def __getitem__(self, idx):
        data = torch.tensor(self.dataX[idx], dtype=torch.float32)
        label = torch.tensor(self.dataY[idx], dtype=torch.long) if self.dataY is not None else None
        return data, label
    


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Linear(512, opt.latent_dim)
        self.logvar = nn.Linear(512, opt.latent_dim)

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        x = self.model(img_flat)
        mu = self.mu(x)
        logvar = self.logvar(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(np.prod(img_shape))),
        )

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], *img_shape)
        return img 

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 256),  
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),            
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity

adversarial_loss = torch.nn.BCELoss()
nll_loss = torch.nn.BCEWithLogitsLoss()

# Initialize generator and discriminator
encoder = Encoder()
decoder = Decoder()
discriminator = Discriminator()

if cuda:
    encoder.cuda()
    decoder.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    nll_loss.cuda()

transform = transforms.Compose([
    transforms.Resize(opt.img_size),
    transforms.ToTensor(),
])
dataX = "../../../data/mnist/trainX.npy"
dataY = "../../../data/mnist/trainY.npy"
devX = "../../../data/mnist/validX.npy"
devY = "../../../data/mnist/validY.npy"
testX = "../../../data/mnist/testX.npy"
testY = "../../../data/mnist/testY.npy"
verbosity = 0  
train_dataset = NumpyDataset(dataX, dataY)
train_loader = DataLoader(dataset=train_dataset, batch_size=200, shuffle=True)
dev_dataset = NumpyDataset(devX, devY)
dev_loader = DataLoader(dataset=dev_dataset, batch_size=200, shuffle=False)

test_dataset = NumpyDataset(testX, testY)
test_loader = DataLoader(dataset=test_dataset, batch_size = 200, shuffle = False)


import torch

# print(f"Train Dataset Information:\n"
#       f"- Type: {type(train_loader.dataset)}\n"
#       f"- Length: {len(train_loader.dataset)}\n"
#       f"- Size (in bytes): {train_loader.dataset.__sizeof__()}\n"
#       f"- Dataset Object: {train_loader.dataset}\n"
#       f"- Data Shape: {train_loader.dataset.data.shape}\n"
#       f"- Data Type: {train_loader.dataset.data.dtype}\n"
#       f"- Target Shape: {train_loader.dataset.targets.shape}\n"
#       f"- Target Type: {train_loader.dataset.targets.dtype}\n"
#       f"- Example Data:\n{train_loader.dataset.data[0]}\n"
#       f"- Example Target: {train_loader.dataset.targets[0]}")

optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr * 0.5, betas=(opt.b1, opt.b2))

optimizer_G = torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=opt.lr * 1.5, betas=(opt.b1, opt.b2)
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Initialize lists to store losses
g_losses = []
d_losses = []
recon_losses = []
adv_losses = []
mse_losses = []
kld_losses = []
nll_losses = []
discriminator_accs = []

def sample_image(n_row, batches_done):
    z = Variable(Tensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    gen_imgs = decoder(z)
    gen_imgs = torch.sigmoid(gen_imgs) 
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)

def visualize_reconstructions(epoch, batches_done, real_imgs, decoded_imgs):
    real_imgs = real_imgs[:25]
    decoded_imgs = decoded_imgs[:25]
    decoded_imgs = torch.sigmoid(decoded_imgs)  
    comparison = torch.cat([real_imgs, decoded_imgs])
    save_image(comparison.data, "images/reconstruction_%d.png" % batches_done, nrow=5, normalize=True)

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(train_loader):

        batches_done = epoch * len(train_loader) + i

        # Adversarial ground truths with label smoothing
        valid = Variable(Tensor(imgs.size(0), 1).fill_(0.9), requires_grad=False) 
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.1), requires_grad=False)   

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        encoded_imgs, mu, logvar = encoder(real_imgs)
        decoded_imgs = decoder(encoded_imgs)

        # Adversarial loss
        g_loss_adv = adversarial_loss(discriminator(encoded_imgs), valid)

        # Reconstruction loss using NLL (BCE with logits)
        reconstruction_loss = nll_loss(decoded_imgs, real_imgs)

        # KL Divergence
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld_loss /= imgs.size(0) * np.prod(img_shape)  # Normalize

        # Total generator loss with adjusted loss weighting
        g_loss = 0.001 * g_loss_adv + 0.999 * reconstruction_loss + 0.1 * kld_loss

        g_loss.backward()

        # Apply gradient rescaling to the generator's gradients
        project_gradients_to_gaussian_ball(encoder, radius=5.0)
        project_gradients_to_gaussian_ball(decoder, radius=5.0)


        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as discriminator ground truth
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.size(0), opt.latent_dim))))

        # Add noise to discriminator inputs
        noise_factor = 0.1
        z_real = z + noise_factor * torch.randn_like(z)
        z_fake = encoded_imgs.detach() + noise_factor * torch.randn_like(encoded_imgs.detach())

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(z_real), valid)
        fake_loss = adversarial_loss(discriminator(z_fake), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()

        # Apply gradient rescaling to the generator's gradients
        project_gradients_to_gaussian_ball(encoder, radius=5.0)
        project_gradients_to_gaussian_ball(decoder, radius=5.0)

        optimizer_D.step()

        # Calculate discriminator accuracy
        with torch.no_grad():
            # Predictions for real and fake latent codes
            pred_real = discriminator(z_real)
            pred_fake = discriminator(z_fake)

            # Threshold predictions at 0.5
            acc_real = (pred_real >= 0.5).float()
            acc_fake = (pred_fake < 0.5).float()

            # Compute accuracy
            discriminator_acc = torch.mean(torch.cat((acc_real, acc_fake), 0))
            discriminator_accs.append(discriminator_acc.item())

        # Compute MSE for monitoring
        mse = F.mse_loss(torch.sigmoid(decoded_imgs), real_imgs)

        # Save losses for plotting
        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
        recon_losses.append(reconstruction_loss.item())
        adv_losses.append(g_loss_adv.item())
        mse_losses.append(mse.item())
        kld_losses.append(kld_loss.item())
        nll_losses.append(reconstruction_loss.item())

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Adv loss: %f] [Recon loss(NLL): %f] [KL-D: %f] [Acc: %.2f%%] [MSE: %f]"
            % (
                epoch + 1,
                opt.n_epochs,
                i + 1,
                len(train_loader),
                d_loss.item(),
                g_loss.item(),
                g_loss_adv.item(),
                reconstruction_loss.item(),
                kld_loss.item(),
                100 * discriminator_acc.item(),
                mse.item(),
            )
        )

        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)
            visualize_reconstructions(epoch, batches_done, real_imgs, decoded_imgs)

    print(f"[Epoch {epoch + 1}/{opt.n_epochs}] Training completed.")
print("Training completed.")

gmm = fit_gmm_to_latent_space(encoder, train_loader)



# Save the models
torch.save(encoder.state_dict(), "models/encoder.pth")
torch.save(decoder.state_dict(), "models/decoder.pth")
torch.save(discriminator.state_dict(), "models/discriminator.pth")

# Plot the training losses
def plot_losses(g_losses, d_losses, recon_losses, adv_losses, mse_losses, kld_losses, nll_losses):
    iterations = range(len(g_losses))
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(iterations, g_losses, label="G loss")
    plt.plot(iterations, d_losses, label="D loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("images/losses.png")
    # plt.show()

    plt.figure(figsize=(10, 5))
    plt.title("Reconstruction (NLL) and Adversarial Loss During Training")
    plt.plot(iterations, recon_losses, label="NLL loss")
    plt.plot(iterations, adv_losses, label="Adversarial loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("images/recon_adv_losses.png")
    # plt.show()

    plt.figure(figsize=(10, 5))
    plt.title("MSE and KL-Divergence During Training")
    plt.plot(iterations, mse_losses, label="MSE")
    plt.plot(iterations, kld_losses, label="KL-D")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("images/mse_kld.png")
    # plt.show()

# Visualize the latent space
def visualize_latent_space(encoder, dataloader):
    encoder.eval()
    latents = []
    labels = []
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = imgs.type(Tensor)
            _, mu, _ = encoder(imgs)
            latents.append(mu.cpu().numpy())
            labels.append(targets.numpy())
    latents = np.concatenate(latents, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Use t-SNE to reduce dimensionality to 2D
    tsne = TSNE(n_components=2, random_state=42)
    latents_2d = tsne.fit_transform(latents)

    # Plot the latent space
    plt.figure(figsize=(8, 6))
    plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap="tab10", alpha=0.6)
    plt.colorbar()
    plt.title("t-SNE of Latent Representations")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig("images/latent_space.png")
    # plt.show()


# ----------
#  Evaluation
# ----------

def evaluate_model(encoder, decoder, discriminator, test_loader, adversarial_loss, nll_loss, cuda):
    encoder.eval()
    decoder.eval()
    discriminator.eval()

    total_d_loss = 0
    total_g_loss = 0
    total_reconstruction_loss = 0
    total_adv_loss = 0
    total_kld_loss = 0
    total_mse_loss = 0
    total_accuracy = 0
    total_samples = 0

    with torch.no_grad():
        for imgs, _ in tqdm(test_loader, desc="Evaluating on Test Data"):
            # Prepare the inputs
            real_imgs = imgs.type(torch.FloatTensor).cuda() if cuda else imgs.type(torch.FloatTensor)

            # Forward pass through the encoder and decoder
            encoded_imgs, mu, logvar = encoder(real_imgs)
            decoded_imgs = decoder(encoded_imgs)

            # Adversarial loss
            valid = torch.ones(real_imgs.size(0), 1).cuda() if cuda else torch.ones(real_imgs.size(0), 1)
            fake = torch.zeros(real_imgs.size(0), 1).cuda() if cuda else torch.zeros(real_imgs.size(0), 1)
            g_loss_adv = adversarial_loss(discriminator(encoded_imgs), valid)

            # Reconstruction loss using NLL (BCE with logits)
            reconstruction_loss = nll_loss(decoded_imgs, real_imgs)

            # KL Divergence
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kld_loss /= real_imgs.size(0) * np.prod(img_shape)  # Normalize

            # MSE
            mse = F.mse_loss(torch.sigmoid(decoded_imgs), real_imgs)

            # Calculate the discriminator's performance
            z = torch.randn(real_imgs.size(0), opt.latent_dim).cuda() if cuda else torch.randn(real_imgs.size(0), opt.latent_dim)
            z_real = z + 0.1 * torch.randn_like(z)
            z_fake = encoded_imgs.detach() + 0.1 * torch.randn_like(encoded_imgs.detach())

            real_loss = adversarial_loss(discriminator(z_real), valid)
            fake_loss = adversarial_loss(discriminator(z_fake), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            # Calculate accuracy
            with torch.no_grad():
                pred_real = discriminator(z_real)
                pred_fake = discriminator(z_fake)

                acc_real = (pred_real >= 0.5).float()
                acc_fake = (pred_fake < 0.5).float()
                accuracy = torch.mean(torch.cat((acc_real, acc_fake), 0))

            # Accumulate loss and accuracy values
            total_d_loss += d_loss.item()
            total_g_loss += g_loss_adv.item()
            total_reconstruction_loss += reconstruction_loss.item()
            total_adv_loss += g_loss_adv.item()
            total_kld_loss += kld_loss.item()
            total_mse_loss += mse.item()
            total_accuracy += accuracy.item()
            total_samples += 1

    # Calculate average loss values
    avg_d_loss = total_d_loss / total_samples
    avg_g_loss = total_g_loss / total_samples
    avg_reconstruction_loss = total_reconstruction_loss / total_samples
    avg_adv_loss = total_adv_loss / total_samples
    avg_kld_loss = total_kld_loss / total_samples
    avg_mse_loss = total_mse_loss / total_samples
    avg_accuracy = total_accuracy / total_samples * 100

    print(f"Evaluation Results: \n"
          f"Discriminator Loss: {avg_d_loss:.4f} \n"
          f"Generator Loss: {avg_g_loss:.4f} \n"
          f"Reconstruction Loss: {avg_reconstruction_loss:.4f} \n"
          f"Adversarial Loss: {avg_adv_loss:.4f} \n"
          f"KL Divergence Loss: {avg_kld_loss:.4f} \n"
          f"MSE Loss: {avg_mse_loss:.4f} \n"
          f"Discriminator Accuracy: {avg_accuracy:.2f}%")
    
    return avg_d_loss, avg_g_loss, avg_reconstruction_loss, avg_adv_loss, avg_kld_loss, avg_mse_loss, avg_accuracy


# Call evaluation function
evaluate_model(encoder, decoder, discriminator, test_loader, adversarial_loss, nll_loss, cuda)
