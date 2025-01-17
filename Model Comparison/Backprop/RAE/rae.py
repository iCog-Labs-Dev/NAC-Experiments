import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from rae_model import RegularizedAutoencoder  
import sys, getopt as gopt, time
import logging
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.numpy_dataset import NumpyDataset
from utils.calc_perc_error import evaluate_perc_err
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from density.fit_gmm import fit_gmm
from density.eval_logpx import evaluate_logpx

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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  
)

def train(model, loader, optimizer, epoch):

    model.train()
    bce_losses = []

    for batch_idx, (data, _) in enumerate(tqdm(loader)):
        data = (data > 0.5).float()
        data = data.view(data.size(0), -1)  
        
        optimizer.zero_grad()
        reconstructed = model(data)
        reconstructed = reconstructed.view(reconstructed.size(0), -1) 
    
        # BCE Loss for reconstruction
        bce_loss = F.binary_cross_entropy(reconstructed, data, reduction="sum")
        bce_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bce_losses.append(bce_loss.item() / data.size(0))

    avg_bce = np.mean(bce_losses)
    return avg_bce

def evaluate_model(model, train_loader, test_loader, latent_dim, n_components=75, num_samples=5000):
    """
    Evaluate the model on various metrics including BCE loss, classification error, GMM fitting, 
    Monte Carlo log-likelihood, and inference time.

    Args:
        model: The trained model to evaluate.
        train_loader: DataLoader providing the training dataset.
        test_loader: DataLoader providing the test dataset.
        latent_dim: Dimension of the latent space.
        n_components: Number of GMM components for fitting.
        num_samples: Number of Monte Carlo samples for log-likelihood estimation.

    Returns:
        results: A dictionary containing evaluation metrics and total inference time.
    """
    logging.info("Starting model evaluation...")
    inference_start_time = time.time()

    results = {}
    model.eval()

    logging.info("Calculating Binary Cross-Entropy (BCE) loss...")
    bce_losses = []

    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = (data > 0.5).float()
            data = data.view(data.size(0), -1)
            reconstructed = model(data)
            reconstructed = reconstructed.view(reconstructed.size(0), -1)
            bce_loss = F.binary_cross_entropy(reconstructed, data, reduction="sum")
            bce_losses.append(bce_loss.item() / data.size(0))
            if i % 10 == 0:  # Log progress every 10 batches
                logging.info(f"Processed {i+1} batches for BCE loss.")
    
    avg_bce = np.mean(bce_losses)
    results['Test_BCE'] = avg_bce
    logging.info(f"Finished BCE loss calculation: {avg_bce:.4f}")

    logging.info("Evaluating classification error (%Err)...")
    err = evaluate_perc_err(model, train_loader, test_loader)
    results['Classification_Error'] = err
    logging.info(f"Classification error: {err:.4f}%")

    logging.info("Fitting GMM on latent space...")
    gmm = fit_gmm(train_loader, model, latent_dim=latent_dim, n_components=n_components)
    logging.info("Finished fitting GMM.")

    logging.info("Evaluating Monte Carlo log-likelihood...")
    test_logpx = evaluate_logpx(test_loader, model, gmm, latent_dim=latent_dim, num_samples=num_samples)
    results['Monte_Carlo_Log_Likelihood'] = test_logpx
    logging.info(f"Monte Carlo log-likelihood: {test_logpx:.4f}")

    total_inference_time = time.time() - inference_start_time
    results['Total_Inference_Time'] = total_inference_time
    logging.info(f"Total inference time: {total_inference_time:.2f} seconds")

    return results

def masked_mse(model, loader):
    model.eval()
    total_mse = 0.0
    total_samples = 0
    with torch.no_grad():
        for data, _ in loader:
            data = data / 255.0 
            data = data.view(data.size(0), -1)

            mask = torch.ones_like(data, dtype=torch.bool) 
            mask[:, : data.size(1) // 2] = 0  

            masked_data = data * mask.float()  

            reconstructed = model(masked_data).view(data.size(0), -1)

            mse = F.mse_loss(reconstructed[~mask], data[~mask], reduction="sum")
            total_mse += mse.item()
            total_samples += data.size(0)

    avg_mse = total_mse / (total_samples * data.size(1) // 2)
    return avg_mse

input_dim = 28 * 28
hidden_dims = [360, 360]
latent_dim = 20
model = RegularizedAutoencoder(latent_dim=latent_dim, input_dim=input_dim, hidden_dims=hidden_dims)
optimizer = optim.SGD(model.parameters(), lr=0.1)
num_epochs = 5

print("--------------- Training ---------------")
logging.info("Starting model training...")
sim_start_time = time.time()
for epoch in range(1, num_epochs + 1): 
    train_bce = train(model, train_loader, optimizer, epoch)
    print(f'Epoch [{epoch}/{num_epochs}]')
    print(f'Train BCE {train_bce:.4f}')
logging.info(f"Finished BCE loss calculation: {train_bce:.4f}")
sim_time = time.time() - sim_start_time
logging.info(f"Total training time: {sim_time:.2f} seconds")    

print("--------------- Evaluation ---------------")
results = evaluate_model(model, train_loader, test_loader, latent_dim=latent_dim, n_components=75)