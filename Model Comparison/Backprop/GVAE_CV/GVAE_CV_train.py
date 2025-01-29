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
from utils.metrics import classification_error, masked_mse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from density.fit_gmm import fit_gmm
from density.eval_logpx import evaluate_logpx

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

def evaluate(model, loader, fixed_variance, n_components, num_samples):
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

            x_recon, mu, z, l2_penalty = model(data)
            x_recon = x_recon.view(data.size(0), -1)

            kl_loss = -0.5 * torch.sum(1 + fixed_variance - mu.pow(2) - torch.exp(fixed_variance))
            kl_loss = kl_loss / 20 
            reconstruction_loss = F.binary_cross_entropy(x_recon, data, reduction="sum")
            total_loss = reconstruction_loss+ kl_loss + l2_penalty

            total_losses.append(total_loss.item() / data.size(0))

    results['Test_BCE'] = np.mean(total_losses)
    logging.info(f"Test BCE loss: {results['Test_BCE']:.4f}")

    logging.info("Evaluating M-MSE...")
    results['M-MSE'] = masked_mse(model, test_loader)
    logging.info(f"M-MSE: {results['M-MSE']:.4f}")

    logging.info("Evaluating classification error...")
    results['%Err'] = classification_error(model, train_loader, test_loader)
    logging.info(f"Classification error: {results['%Err']:.4f}%")

    logging.info("Fitting GMM on latent space...")
    gmm = fit_gmm(train_loader, model, latent_dim=latent_dim, n_components=n_components)
    logging.info("Finished fitting GMM.")

    logging.info("Evaluating Monte Carlo log-likelihood...") 
    results['log_p(x)'] = evaluate_logpx(test_loader, model, gmm, latent_dim=latent_dim, num_samples=num_samples)
    logging.info(f"Monte Carlo log-likelihood: {results['log_p(x)']:.4f}")

    results['Total_inference_time'] = time.time() - inference_start_time
    logging.info(f"Total inference time: {results['Total_inference_time']:.2f} sec")

    return results

input_dim = 28 * 28
hidden_dim = [360, 360]
latent_dim = 20
l2_lambda = 1e-3
fixed_variance=torch.tensor(0.0)
model = GVAE_CV(input_dim, latent_dim, hidden_dim, l2_lambda, fixed_variance)
optimizer = optim.SGD(model.parameters(), lr=0.02)
num_epochs = 50
n_components=75
num_samples = 5000

# Training
logging.info("Starting model training...")
sim_start_time = time.time()
for epoch in range(1, num_epochs + 1):
    avg_bce = train(model, train_loader, optimizer, fixed_variance)
    logging.info(f'Epoch [{epoch}/{num_epochs}]  Train BCE = {avg_bce:.4f}')
sim_time = time.time() - sim_start_time
logging.info(f"Total training time: {sim_time:.2f} sec")   

# Evaluation
test_bce = evaluate(model, test_loader, fixed_variance, n_components, num_samples)