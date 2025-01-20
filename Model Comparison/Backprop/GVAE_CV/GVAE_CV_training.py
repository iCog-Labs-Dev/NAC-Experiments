import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from GVAE_CV_model import GVAE
from sklearn.mixture import GaussianMixture
import scipy.stats as stats
import sys
import getopt as gopt
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Data Loading Classes and Functions
class NumpyDataset(Dataset):
    def __init__(self, dataX, dataY=None):
        self.dataX = np.load(dataX)
        self.dataY = np.load(dataY) if dataY is not None else None

    def __len__(self):
        return len(self.dataX)

    def __getitem__(self, idx):
        data = torch.tensor(self.dataX[idx], dtype=torch.float32)
        label = (
            torch.tensor(self.dataY[idx], dtype=torch.long)
            if self.dataY is not None
            else None
        )
        return data, label


options, remainder = gopt.getopt(
    sys.argv[1:],
    "",
    ["dataX=", "dataY=", "devX=", "devY=", "testX", "testY", "verbosity="],
)

dataX = "../../../data/mnist/trainX.npy"
dataY = "../../../data/mnist/trainY.npy"
devX = "../../../data/mnist/validX.npy"
devY = "../../../data/mnist/validY.npy"
testX = "../../../data/mnist/testX.npy"
testY = "../../../data/mnist/testY.npy"
verbosity = 0

for opt, arg in options:
    if opt in ("--dataX"):
        dataX = arg.strip()
    elif opt in ("--dataY"):
        dataY = arg.strip()
    elif opt in ("--devX"):
        devX = arg.strip()
    elif opt in ("--devY"):
        devY = arg.strip()
    elif opt in ("--testX"):
        testX = arg.strip()
    elif opt in ("--testY"):
        testY = arg.strip()
    elif opt in ("--verbosity"):
        verbosity = int(arg.strip())


# Create dataloaders
train_dataset = NumpyDataset(dataX, dataY)
train_loader = DataLoader(dataset=train_dataset, batch_size=200, shuffle=True)

dev_dataset = NumpyDataset(devX, devY)
dev_loader = DataLoader(dataset=dev_dataset, batch_size=200, shuffle=False)

test_dataset = NumpyDataset(testX, testY)
test_loader = DataLoader(dataset=test_dataset, batch_size=200, shuffle=False)


def vae_loss(recon_x, x, mu, fixed_logvar):
    recon_x = recon_x.view(recon_x.size(0), -1)
    x = x.view(x.size(0), -1)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")

    # KL Divergence
    kl_loss = -0.5 * torch.sum(1 + fixed_logvar - mu.pow(2) - torch.exp(fixed_logvar))
    kl_loss = kl_loss / 20
    return recon_loss + kl_loss


def rescale_gradients(model, max_norm=5.0):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)


def train_model(model, train_loader):

    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

    model.train()
    for epoch in range(50):
        total_loss = 0
        total_samples = 0
        for batch_idx, (data, _) in enumerate(train_loader):

            data = (data > 0.5).float()
            data = data.view(data.size(0), -1)
            optimizer.zero_grad()

            recon_data, mu = model(data)
            recon_data = recon_data.view(recon_data.size(0), -1)
            loss = vae_loss(recon_data, data, mu, fixed_logvar)
            loss.backward()
            rescale_gradients(model)
            optimizer.step()
            total_loss += loss.item()
            total_samples += data.size(0)
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader.dataset):.4f}")


# Function to calculate the error of the model
def bce_loss(model, loader):
    model.eval()
    total_bce = 0.0
    total_samples = 0
    with torch.no_grad():
        for data, _ in loader:
            data = data.view(data.size(0), -1)
            data = (data > 0.5).float()
            recon_data, _ = model(data)
            recon_data = recon_data.view(data.size(0), -1)

            bce = F.binary_cross_entropy(recon_data, data, reduction="sum")
            total_bce += bce.item()

            total_samples += data.size(0)

    # Normalize by the total number of elements
    avg_bce = total_bce / total_samples

    return avg_bce


# M-MSE Loss
def masked_mse_loss(model, loader):
    model.eval()
    total_mse = 0.0
    total_samples = 0
    total_masked_elements = 0
    with torch.no_grad():
        for data, _ in loader:

            data = data.view(data.size(0), -1)
            data = (data > 0.5).float()
            # Mask exactly half of the image columns
            mask = torch.ones_like(data, dtype=torch.bool)
            mask[:, : data.size(1) // 2] = 0

            masked_data = data * mask.float()
            masked_data = (masked_data > 0.5).float()
            reconstructed, _ = model(masked_data)
            reconstructed = reconstructed.view(data.size(0), -1)

            mse = F.mse_loss(reconstructed[~mask], data[~mask], reduction="sum")
            total_mse += mse.item() * data.size(0)
            total_samples += data.size(0)
            total_masked_elements += (~mask).sum().item()

    avg_mse = total_mse / (total_samples * data.size(1) // 2)

    return avg_mse


# Classification Loss


def classification_loss(model, data_loader, latent_dim, num_classes):
    """
    Fit a logistic regression classifier on the latent space representations and evaluate on the test set.

    Parameters:
        model (nn.Module): The trained GVAE model.
        data_loader (DataLoader): DataLoader for the dataset (training or test set).
        latent_dim (int): Dimensionality of the latent space.
        num_classes (int): Number of unique categories in the dataset.

    Returns:
        float: Classification error (percentage).
    """
    model.eval()
    latent_representations = []
    labels = []

    with torch.no_grad():
        for batch in data_loader:
            data, target = batch

            data = data.view(data.size(0), -1)

            # If the target is one-hot encoded, convert it to class indices
            if target.ndim > 1:
                target = torch.argmax(target, dim=1)

            # Extract latent representations and collect labels
            mu = model.encoder(data)
            latent_representations.append(mu.cpu().numpy())
            labels.append(target.cpu().numpy())

    # Stack all latent representations and labels into single arrays
    X = np.vstack(latent_representations)
    y = np.hstack(labels)

    print(f"Shape of latent representations (X): {X.shape}")
    print(f"Shape of labels (y): {y.shape}")

    # Ensure the number of samples in X and y are consistent
    assert (
        X.shape[0] == y.shape[0]
    ), "Mismatch in the number of samples between X and y!"

    classifier = LogisticRegression(max_iter=1000, multi_class="multinomial")
    classifier.fit(X, y)

    y_pred = classifier.predict(X)
    accuracy = accuracy_score(y, y_pred)

    error_percentage = 100 * (1 - accuracy)
    return error_percentage


# Density Sampling
# Function to fit GMM
def fit_gmm(latent_vectors, n_components=75):
    gmm = GaussianMixture(
        n_components=n_components, covariance_type="full", random_state=42
    )
    gmm.fit(latent_vectors)
    return gmm


# Function to calculate Monte Carlo log likelihood
def monte_carlo_log_likelihood(gmm, gvae, data_loader, n_samples=5000):
    """
    Calculate the Monte Carlo log likelihood:
    log p(x) ≈ log E_{z ~ GMM}[p(x|z) * p(z)]
    """
    gmm_samples, _ = gmm.sample(n_samples)
    z_samples = torch.tensor(gmm_samples, dtype=torch.float32)

    log_p_z = gmm.score_samples(gmm_samples)  # Log probability under GMM
    log_p_x_given_z = []

    # Decode z_samples to get p(x|z)
    with torch.no_grad():
        for i in range(0, n_samples, data_loader.batch_size):
            batch_z = z_samples[i : i + data_loader.batch_size]
            recon_x = gvae.decoder(batch_z)
            log_p_x_given_z.extend(
                -torch.nn.functional.binary_cross_entropy(
                    recon_x, recon_x, reduction="none"
                )
                .sum(dim=1)
                .cpu()
                .numpy()
            )  # Reconstruction likelihood

    log_p_x_given_z = np.array(log_p_x_given_z)

    # Combine log p(z) and log p(x|z)
    log_likelihood = np.mean(log_p_z + log_p_x_given_z)
    return log_likelihood


def density_modeling(model, loader):
    # Collect latent vectors from the dataset
    latent_vectors = []
    model.eval()
    print("Starting to process latent vectors...")

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(
            tqdm(train_loader, desc="Processing Latent Vectors")
        ):
            data = data.view(data.size(0), -1)
            mu = model.encoder(data)
            latent_vectors.append(mu.cpu().numpy())

    print("Finished processing latent vectors. Fitting GMM...")

    latent_vectors = np.vstack(latent_vectors)
    print(f"Latent vectors shape: {latent_vectors.shape}")

    # Fit the GMM
    gmm = fit_gmm(latent_vectors, n_components=75)
    print("GMM fitting complete. Calculating Monte Carlo log likelihood...")

    # Calculate Monte Carlo log likelihood
    log_likelihood = monte_carlo_log_likelihood(gmm, model, train_loader)
    # print(f"Monte Carlo Log Likelihood: {log_likelihood:.4f}")
    return log_likelihood


input_dim = 784
latent_dim = 20
hidden_dim = 360
fixed_logvar = torch.tensor(0.0)
gvae = GVAE(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    latent_dim=latent_dim,
    fixed_logvar=fixed_logvar,
)

sim_start_time = time.time()
print("--------------- Training ---------------")
train_model(gvae, train_loader)

sim_time = time.time() - sim_start_time
print(f"Training Time = {sim_time:.4f} seconds")

print("--------------- Testing ---------------")
inference_start_time = time.time()
bce_loss = bce_loss(gvae, test_loader)
classification_loss = classification_loss(
    gvae, test_loader, latent_dim=latent_dim, num_classes=10
)
masked_mse = masked_mse_loss(gvae, test_loader)
log_likelihood = density_modeling(gvae, test_loader)
# test_mse, test_loss, test_bce, test_accuracy = evaluate(model, test_loader)
inference_time = time.time() - inference_start_time
print(
    f"Test MSE: {masked_mse:.4f}, Test BCE: {bce_loss:.4f}, Error Percentage: {classification_loss:.2f}%, LogLikelihood: {log_likelihood}"
)
print(f"Inference Time = {inference_time:.4f} seconds")
