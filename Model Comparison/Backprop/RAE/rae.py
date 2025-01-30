import torch
import os
import time
import logging
import sys
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from rae_model import RegularizedAutoencoder  
import getopt as gopt
from torch import optim
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.numpy_dataset import NumpyDataset
from utils.metrics import classification_error, masked_mse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from density.fit_gmm import fit_gmm
from density.eval_logpx import evaluate_logpx

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

# Training
def train(model, loader, optimizer):

    model.train()
    total_losses = []

    for batch_idx, (data, _) in enumerate(tqdm(loader)):
        data = (data > 0.5).float().view(data.size(0), -1) 
        optimizer.zero_grad()
        reconstructed, l2_penality = model(data)  
        reconstructed = reconstructed.view(reconstructed.size(0), -1)  
        bce_loss = F.binary_cross_entropy(reconstructed, data, reduction="sum")
        total_loss = bce_loss + l2_penality

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_losses.append(total_loss.item() / data.size(0))

    return np.mean(total_losses)

# Evaluation
def evaluate_model(model, train_loader, test_loader, latent_dim, n_components, num_samples):
    logging.info("Starting model evaluation...")
    inference_start_time = time.time()
    results = {}
    model.eval()

    logging.info("Calculating Binary Cross-Entropy (BCE) loss...")
    total_losses = []
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = (data > 0.5).float().view(data.size(0), -1)
            reconstructed, l2_penality = model(data)  
            reconstructed = reconstructed.view(reconstructed.size(0), -1) 

            bce_loss = F.binary_cross_entropy(reconstructed, data, reduction="sum")
            total_loss = bce_loss + l2_penality
            total_losses.append(total_loss.item() / data.size(0))

    results['Test_BCE'] = np.mean(total_losses)
    logging.info(f"Test BCE loss: {results['Test_BCE']:.4f}")

    logging.info("Evaluating classification error...")
    results['%Err'] = classification_error(model, train_loader, test_loader)
    logging.info(f"Classification error: {results['%Err']:.4f}%")

    logging.info("Evaluating M-MSE...")
    results['M-MSE'] = masked_mse(model, test_loader)
    logging.info(f"M-MSE: {results['M-MSE']:.4f}")

    logging.info("Fitting GMM on latent space...")
    gmm = fit_gmm(train_loader, model, latent_dim, n_components)
    logging.info("Finished fitting GMM.")

    logging.info("Evaluating Monte Carlo log-likelihood...") 
    results['log_p(x)'] = evaluate_logpx(test_loader, model, gmm, latent_dim, num_samples)
    logging.info(f"Monte Carlo log-likelihood: {results['log_p(x)']:.4f}")

    results['Total_inference_time'] = time.time() - inference_start_time
    logging.info(f"Total inference time: {results['Total_inference_time']:.2f} sec")

    return results

input_dim = 28 * 28
hidden_dims = [360, 360]
latent_dim = 20
l2_lambda = 1e-3
model = RegularizedAutoencoder(input_dim, hidden_dims, latent_dim, l2_lambda)
optimizer = optim.SGD(model.parameters(), lr=0.1)
num_epochs = 50
n_components=75
num_samples=5000

logging.info("Starting model training...")
sim_start_time = time.time()
for epoch in range(1, num_epochs + 1): 
    train_bce = train(model, train_loader, optimizer)
    logging.info(f"Epoch [{epoch}/{num_epochs}] Train BCE Loss: {train_bce:.4f}")
sim_time = time.time() - sim_start_time
logging.info(f"Total training time: {sim_time:.2f} sec")    

# Evaluation
results = evaluate_model(model, train_loader, test_loader, latent_dim, n_components, num_samples)