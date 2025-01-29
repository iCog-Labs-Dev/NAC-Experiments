import torch
import numpy as np
import random
import logging
import sys
import os
import time
import getopt as gopt
from tqdm import tqdm
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from GVAE_CV_model import GVAE_CV
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.numpy_dataset import NumpyDataset

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

options, remainder = gopt.getopt(sys.argv[1:], '',
                                 ["dataX=", "dataY=", "devX=", "devY=", "testX", "testY","verbosity="]
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

print("Train-set: X: {} | Y: {}".format(dataX, dataY))
print("  Dev-set: X: {} | Y: {}".format(devX, devY))
print("  Test-set: X: {} | Y: {}".format(testX, testY))

train_dataset = NumpyDataset(dataX, dataY)
train_loader = DataLoader(dataset=train_dataset, batch_size=200, shuffle=True)

dev_dataset = NumpyDataset(devX, devY)
dev_loader = DataLoader(dataset=dev_dataset, batch_size=200, shuffle=False)

test_dataset = NumpyDataset(testX, testY)
test_loader = DataLoader(dataset=test_dataset, batch_size = 200, shuffle = False)

def train(model, loader, optimizer, fixed_variance):
    model.train()
    total_losses = []

    for batch_idx, (data, _) in enumerate(tqdm(loader)):
        data = (data > 0.5).float()
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()

        recon_x, mu, z, l2_penalty = model(data)
        recon_x = recon_x.view(recon_x.size(0), -1)

        kl_loss = -0.5 * torch.sum(1 + fixed_variance - mu.pow(2) - torch.exp(fixed_variance))
        kl_loss = kl_loss / 20
        reconstruction_loss = F.binary_cross_entropy(recon_x, data, reduction="sum")
        total_loss = reconstruction_loss + kl_loss + l2_penalty

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_losses.append(total_loss.item() / data.size(0))

    avg_loss = np.mean(total_losses)
    return avg_loss

def evaluate(model, loader, fixed_variance):
    logging.info("Starting model evaluation...")
    model.eval()
    logging.info("Calculating Binary Cross-Entropy (BCE) loss...")
    total_losses = [] 
    results = {}
    with torch.no_grad():
        for data, _ in loader:
            data = (data > 0.5).float()
            data = data.view(data.size(0), -1)

            x_recon, mu, z, l2_penalty = model(data)
            x_recon = x_recon.view(data.size(0), -1)

            kl_loss = -0.5 * torch.sum(1 + fixed_variance - mu.pow(2) - torch.exp(fixed_variance))
            kl_loss = kl_loss / 20 
            reconstruction_loss = F.binary_cross_entropy(x_recon, data, reduction="sum")
            total_loss = reconstruction_loss+ kl_loss + l2_penalty

            total_losses.append(total_loss.item() / data.size(0))

    results['Test_BCE'] = np.mean(total_losses)
    logging.info(f"Test BCE loss: {results['Test_BCE']:.4f}")
    return results

input_dim = 28 * 28
hidden_dim = [360, 360]
latent_dim = 20
l2_lambda = 1e-3
fixed_variance=torch.tensor(0.0)
model = GVAE_CV(input_dim, latent_dim, hidden_dim, l2_lambda, fixed_variance)
optimizer = optim.SGD(model.parameters(), lr=0.02)
num_epochs = 50

# Training
logging.info("Starting model training...")
sim_start_time = time.time()
for epoch in range(1, num_epochs + 1):
    avg_bce = train(model, train_loader, optimizer, fixed_variance)
    logging.info(f'Epoch [{epoch}/{num_epochs}]  Train BCE = {avg_bce:.4f}')
sim_time = time.time() - sim_start_time
logging.info(f"Total training time: {sim_time:.2f} sec")   

# Evaluation
test_bce = evaluate(model, test_loader, fixed_variance)

# GMM
def fit_gmm(latent_vectors, n_components=75):
    gmm = GaussianMixture(
        n_components=n_components, covariance_type="full", random_state=42
    )
    gmm.fit(latent_vectors)
    return gmm

# Classification error
def classification_error(model, data_loader, latent_dim, num_classes):
    model.eval()
    latent_representations = []
    labels = []

    with torch.no_grad():
        for batch in data_loader:
            data, target = batch

            data = data.view(data.size(0), -1)

            
            if target.ndim > 1:
                target = torch.argmax(target, dim=1)

            mu = model.encoder(data)
            latent_representations.append(mu.cpu().numpy())
            labels.append(target.cpu().numpy())

    X = np.vstack(latent_representations)
    y = np.hstack(labels)


    assert (
        X.shape[0] == y.shape[0]
    ), "Mismatch in the number of samples between X and y!"

    classifier = LogisticRegression(max_iter=1000, multi_class="multinomial")
    classifier.fit(X, y)

    y_pred = classifier.predict(X)
    accuracy = accuracy_score(y, y_pred)

    error_percentage = 100 * (1 - accuracy)
    return error_percentage
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

# log likelihood
def monte_carlo_log_likelihood(gmm, gvae, data_loader, n_samples=5000):
    
    gmm_samples, _ = gmm.sample(n_samples)
    z_samples = torch.tensor(gmm_samples, dtype=torch.float32)

    log_p_z = gmm.score_samples(gmm_samples)
    log_p_x_given_z = []

   
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
            )  

    log_p_x_given_z = np.array(log_p_x_given_z)
    log_likelihood = np.mean(log_p_z + log_p_x_given_z)
    
    return log_likelihood


def final_modeling(model, loader):
    latent_vectors = []
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(
            tqdm(train_loader, desc="Processing Latent Vectors")
        ):
            data = data.view(data.size(0), -1)
            mu = model.encoder(data)
            latent_vectors.append(mu.cpu().numpy())
    latent_vectors = np.vstack(latent_vectors)


    # Fit the GMM
    gmm = fit_gmm(latent_vectors, n_components=75)

    # Calculate log likelihood
    log_likelihood = monte_carlo_log_likelihood(gmm, model, train_loader)


    return log_likelihood


input_dim = 784
latent_dim = 20
hidden_dim = 360
fixed_variance = torch.tensor(0.0)
gvae = GVAE_CV(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    latent_dim=latent_dim,
    fixed_variance=fixed_variance
)
sim_start_time = time.time()  # Start time profiling

print("--------------- Training ---------------")
train_model(gvae, train_loader)

# Stop time profiling
sim_time = time.time() - sim_start_time
print(f"Training Time = {sim_time:.4f} seconds")



print("--------------- Testing ---------------")
bce_loss = bce_loss(gvae, test_loader)
classification_error = classification_error(
    gvae, test_loader, latent_dim=latent_dim, num_classes=10
)
masked_mse = masked_mse_loss(gvae, test_loader)
log_likelihood = final_modeling(gvae, test_loader)

print(f"Test M-MSE: {masked_mse:.4f}, Test BCE: {bce_loss:.4f}, Error(%): {classification_error:.2f}%, log p(x): {log_likelihood}")

