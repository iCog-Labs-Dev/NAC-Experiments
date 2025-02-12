import logging
import sys
import os
import time
import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
import getopt as gopt
from gan_ae_model import GANAE
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

print(f"Train-set: X: {dataX} | Y: {dataY}")
print(f"Dev-set: X: {devX} | Y: {devY}")
print(f"Test-set: X: {testX} | Y: {testY}")

# Load datasets
train_dataset = NumpyDataset(dataX, dataY)
train_loader = DataLoader(dataset=train_dataset, batch_size=200, shuffle=True)

dev_dataset = NumpyDataset(devX, devY)
dev_loader = DataLoader(dataset=dev_dataset, batch_size=200, shuffle=False)

test_dataset = NumpyDataset(testX, testY)
test_loader = DataLoader(dataset=test_dataset, batch_size = 200, shuffle = False)

def train(model, loader, optimizer, epoch):
    model.train()
    total_losses = []

    for batch_idx, (data, _) in enumerate(tqdm(loader)):
        data = (data > 0.5).float()
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        x_recon, real_or_fake, mu, logvar, l2_penalty = model(data)

        reconstruction_loss = F.binary_cross_entropy(x_recon, data, reduction="sum")
        discriminator_loss = F.binary_cross_entropy(real_or_fake, torch.ones_like(real_or_fake), reduction= "sum")
        total_loss = reconstruction_loss + discriminator_loss + l2_penalty

        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_losses.append(total_loss.item() / data.size(0))

    avg_loss = np.mean(total_losses)
    return avg_loss

def evaluate(model, loader, n_components=75, num_samples = 5000):
    logging.info("Starting model evaluation...")
    inference_start_time = time.time()
    results = {}

    model.eval()
    logging.info("Calculating Binary Cross-Entropy (BCE) loss...")
    total_losses = [] 
    with torch.no_grad():
        for data, _ in loader:
            data = (data > 0.5).float()
            data = data.view(data.size(0), -1)

            x_recon, real_or_fake, mu, logvar, l2_penalty = model(data)
            x_recon = x_recon.view(data.size(0), -1)

            reconstruction_loss = F.binary_cross_entropy(x_recon, data, reduction="sum")
            discriminator_loss = F.binary_cross_entropy(real_or_fake, torch.ones_like(real_or_fake), reduction= "sum")
            total_loss = reconstruction_loss + discriminator_loss + l2_penalty

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
hidden_dims = [360, 360]
latent_dim = 20
l2_lambda= 1e-3
model = GANAE(input_dim, hidden_dims, latent_dim, l2_lambda)
num_epochs = 50
optimizer = optim.Adam(model.parameters(), lr=0.02)

# Training
for epoch in range(1, num_epochs + 1):
    avg_loss = train(model, train_loader, optimizer, num_epochs)
    print(f'Epoch [{epoch}/{num_epochs}]')
    print(f'Avg Loss = {avg_loss:.4f}')

# Evaluation
results = evaluate(model, test_loader)