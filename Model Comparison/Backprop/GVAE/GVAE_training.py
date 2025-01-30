import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from GVAE_model import VAE
import os
import sys
import random
import logging
import getopt as gopt
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.numpy_dataset import NumpyDataset

# Set random seed for reproducibility
seed_value = 42
torch.manual_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  
)

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

train_dataset = NumpyDataset(dataX, dataY)
train_loader = DataLoader(dataset=train_dataset, batch_size=200, shuffle=True)

dev_dataset = NumpyDataset(devX, devY)
dev_loader = DataLoader(dataset=dev_dataset, batch_size=200, shuffle=False)

test_dataset = NumpyDataset(testX, testY)
test_loader = DataLoader(dataset=test_dataset, batch_size=200, shuffle=False)


def train(model, loader, optimizer):
    model.train()
    total_losses = []

    for batch_idx, (data, _) in enumerate(tqdm(loader)):
        data = (data > 0.5).float()
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()

        recon_data, mu, logvar, l2_penality = model(data)
        recon_data = recon_data.view(recon_data.size(0), -1)

        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))
        kl_loss = kl_loss / 20
        bce_loss = F.binary_cross_entropy(recon_data, data, reduction="sum")
        total_loss = bce_loss + kl_loss + l2_penality

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_losses.append(total_loss.item() / data.size(0))

    avg_bce = np.mean(total_losses)
    return avg_bce

def evaluate(model, loader):
    logging.info("Starting model evaluation...")
    inference_start_time = time.time()

    model.eval()
    logging.info("Calculating Binary Cross-Entropy (BCE) loss...")
    total_losses = []
    results = {}

    with torch.no_grad():
        for data, _ in loader:
            data = (data > 0.5).float()
            data = data.view(data.size(0), -1)

            recon_data, mu, logvar, l2_penality = model(data)
            recon_data = recon_data.view(data.size(0), -1)

            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))
            kl_loss = kl_loss / 20
            bce_loss = F.binary_cross_entropy(recon_data, data, reduction="sum")
            total_loss = bce_loss + kl_loss + l2_penality

            total_losses.append(total_loss.item() / data.size(0))

    results['Test_BCE'] = np.mean(total_losses)
    logging.info(f"Test BCE loss: {results['Test_BCE']:.4f}")

    results['Total_inference_time'] = time.time() - inference_start_time
    logging.info(f"Total inference time: {results['Total_inference_time']:.2f} sec")
    return results

input_dim = 28 * 28
hidden_dim = [360, 360]
latent_dim = 20
l2_lambda = 1e-3
model = VAE(input_dim, hidden_dim, latent_dim, l2_lambda)
optimizer = optim.SGD(model.parameters(), lr=0.01)
num_epochs = 50

# Training
logging.info("Starting model training...")
sim_start_time = time.time()
for epoch in range(1, num_epochs + 1):
    train_bce =train(model, train_loader, optimizer)
    logging.info(f'Epoch [{epoch}/{num_epochs}] BCE = {train_bce:.4f}')
sim_time = time.time() - sim_start_time
logging.info(f"Total training time: {sim_time:.2f} sec")  

# Evaluation
test_bce = evaluate(model, test_loader)

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
            reconstructed, _, _ = model(masked_data)
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
            mu, _ = model.encoder(data)
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
def monte_carlo_log_likelihood(gmm, vae, data_loader, n_samples=5000):
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
            recon_x = vae.decoder(batch_z)
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
            mu, _ = model.encoder(data)
            latent_vectors.append(mu.cpu().numpy())

    print("Finished processing latent vectors. Fitting GMM...")

    latent_vectors = np.vstack(latent_vectors)
    print(f"Latent vectors shape: {latent_vectors.shape}")

    # Fit the GMM
    gmm = fit_gmm(latent_vectors, n_components=75)
    print("GMM fitting complete. Calculating Monte Carlo log likelihood...")

    # Calculate Monte Carlo log likelihood
    log_likelihood = monte_carlo_log_likelihood(gmm, vae, train_loader)
    # print(f"Monte Carlo Log Likelihood: {log_likelihood:.4f}")
    return log_likelihood


input_dim = 784
latent_dim = 20
hidden_dim = [360, 360]
vae = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)

sim_start_time = time.time()
print("--------------- Training ---------------")
train_model(vae, train_loader)

sim_time = time.time() - sim_start_time
print(f"Training Time = {sim_time:.4f} seconds")

print("--------------- Testing ---------------")
inference_start_time = time.time()
bce_loss = bce_loss(vae, test_loader)
classification_loss = classification_loss(vae, test_loader)
masked_mse = masked_mse_loss(vae, test_loader)
log_likelihood = density_modeling(vae, test_loader)
test_mse, test_loss, test_bce, test_accuracy = evaluate(model, test_loader)
inference_time = time.time() - inference_start_time
print(
    f"Test MSE: {masked_mse:.4f}, Test BCE: {bce_loss:.4f}, Error Percentage: {classification_loss:.2f}%, LogLikelihood: {log_likelihood}"
)
print(f"Inference Time = {inference_time:.4f} seconds")
