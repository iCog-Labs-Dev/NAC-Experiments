import argparse
import os
import numpy as np
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

os.makedirs("images", exist_ok=True)
os.makedirs("models", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=200, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of CPU threads for data loading")
parser.add_argument("--latent_dim", type=int, default=10, help="dimensionality of the latent code")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=2000, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if cuda:
    torch.cuda.manual_seed(42)

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

        # Output both mean and log variance for latent variables
        self.mu = nn.Linear(512, opt.latent_dim)
        self.logvar = nn.Linear(512, opt.latent_dim)

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        x = self.model(img_flat)
        mu = self.mu(x)
        logvar = self.logvar(x)
        # Reparameterization trick
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
            # Output logits for NLL loss (no Sigmoid here)
        )

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], *img_shape)
        return img  # Output logits for NLL loss

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

# Loss functions
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


train_loader, test_loader = get_mnist_loaders(
    batch_size=opt.batch_size,
    image_size=opt.img_size,
    num_workers=opt.n_cpu,
    pin_memory=True,
    download=True,
    transform=transform
)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

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
    gen_imgs = torch.sigmoid(gen_imgs)  # Apply sigmoid to logits for visualization
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)

def visualize_reconstructions(epoch, batches_done, real_imgs, decoded_imgs):
    real_imgs = real_imgs[:25]
    decoded_imgs = decoded_imgs[:25]
    decoded_imgs = torch.sigmoid(decoded_imgs)  # Apply sigmoid to logits

    # Create a grid of original and reconstructed images
    comparison = torch.cat([real_imgs, decoded_imgs])
    save_image(comparison.data, "images/reconstruction_%d.png" % batches_done, nrow=5, normalize=True)

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(train_loader):

        batches_done = epoch * len(train_loader) + i

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

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

        # Total generator loss
        g_loss = 0.001 * g_loss_adv + 0.999 * reconstruction_loss + 0.1 * kld_loss

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as discriminator ground truth
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.size(0), opt.latent_dim))))

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(z), valid)
        fake_loss = adversarial_loss(discriminator(encoded_imgs.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

        # Calculate discriminator accuracy
        with torch.no_grad():
            # Predictions for real and fake latent codes
            pred_real = discriminator(z)
            pred_fake = discriminator(encoded_imgs.detach())

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

plot_losses(g_losses, d_losses, recon_losses, adv_losses, mse_losses, kld_losses, nll_losses)

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
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)
    latents_2d = tsne.fit_transform(latents)

    # Plot the latent space
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap="tab10", alpha=0.6)
    plt.colorbar()
    plt.title("t-SNE of Latent Representations")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig("images/latent_space.png")
    # plt.show()


visualize_latent_space(encoder, test_loader)
