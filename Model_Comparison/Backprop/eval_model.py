import sys
import os
import time
import logging
import argparse
from typing import Dict, Any

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.numpy_dataset import NumpyDataset
from utils.metrics import classification_error, masked_mse
from density.fit_gmm import fit_gmm
from density.eval_logpx import evaluate_logpx

from GVAE.GVAE_model import VAE
from GVAE_CV.GVAE_CV_model import GVAE_CV
from RAE.rae_model import RegularizedAutoencoder
from GAN_AE.gan_ae_model import GANAE

"""
Usage: python eval_model.py --model=model_name --config=path_to/config.py
"""

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--model", type=str, required=True, help="Model type to evaluate.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--dataX", type=str, default="../../data/mnist/trainX.npy", help="Path to training data X.")
    parser.add_argument("--dataY", type=str, default="../../data/mnist/trainY.npy", help="Path to training data Y.")
    parser.add_argument("--testX", type=str, default="../../data/mnist/testX.npy", help="Path to test data X.")
    parser.add_argument("--testY", type=str, default="../../data/mnist/testY.npy", help="Path to test data Y.")
    return parser.parse_args()

def load_config(config_file_path: str) -> Dict[str, Any]:
    """Load configuration from a Python file."""
    if not os.path.exists(config_file_path):
        raise ValueError(f"Configuration file not found: {config_file_path}")
    
    config = {}
    with open(config_file_path) as f:
        exec(f.read(), {}, config)
    return config

def load_model(model_type: str, config: Dict[str, Any], save_directory: str) -> torch.nn.Module:
    """Load a trained model based on the model type."""
    model_classes = {
        "GVAE_CV": GVAE_CV,
        "RAE": RegularizedAutoencoder,
        "GAN_AE": GANAE,
        "GVAE": VAE,
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Model type not recognized: {model_type}")
    
    model_class = model_classes[model_type]
    
    # Get only the parameters that the model's __init__ method accepts
    valid_params = {k: v for k, v in config.items() if k in model_class.__init__.__code__.co_varnames}

    model = model_class(**valid_params)
    model.load_state_dict(torch.load(save_directory))
    logging.info(f"Successfully loaded model from {save_directory}")
    return model

def compute_loss(model_type: str, model: torch.nn.Module, data: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
    """Compute the loss for a given model and data."""
    if model_type == "GVAE_CV":
        recon_x, mu, z, l2_penalty = model(data)
        kl_loss = (-0.5 * torch.sum(1 + config['fixed_variance'] - mu.pow(2) - torch.exp(config['fixed_variance']))) / 20
        reconstruction_loss = F.binary_cross_entropy(recon_x, data, reduction="sum")
        total_loss = reconstruction_loss + kl_loss + l2_penalty
    elif model_type == "RAE":
        reconstructed, l2_penalty = model(data)
        bce_loss = F.binary_cross_entropy(reconstructed, data, reduction="sum")
        total_loss = bce_loss + l2_penalty
    elif model_type == "GAN_AE":
        x_recon, real_or_fake, _, _, l2_penalty = model(data)
        reconstruction_loss = F.binary_cross_entropy(x_recon, data, reduction="sum")
        discriminator_loss = F.binary_cross_entropy(real_or_fake, torch.ones_like(real_or_fake), reduction="sum")
        total_loss = reconstruction_loss + discriminator_loss + l2_penalty
    else:
        recon_data, mu, logvar, l2_penalty = model(data)
        kl_loss = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))) / 20
        bce_loss = F.binary_cross_entropy(recon_data, data, reduction="sum")
        total_loss = bce_loss + kl_loss + l2_penalty
    
    return total_loss

def evaluate(
    model: torch.nn.Module, 
    train_loader: DataLoader, 
    test_loader: DataLoader,
    model_type: str, 
    latent_dim: int,
    n_components: int, 
    num_samples: int,
    config: Dict[str, Any]
    ) -> Dict[str, float]:
    """Evaluate the model on the given data loader."""
    logging.info("Starting model evaluation...")
    model.eval()
    total_losses = []
    results = {}

    with torch.no_grad():
        for data, _ in test_loader:
            data = (data > 0.5).float()
            data = data.view(data.size(0), -1)
            total_loss = compute_loss(model_type, model, data, config)
            total_losses.append(total_loss.item() / data.size(0))
    
    results = {'Test_BCE': np.mean(total_losses)}
    print(f"Test BCE loss: {results['Test_BCE']:.2f}")

    logging.info("Evaluating Masked-MSE...")
    results['M-MSE'] = masked_mse(model, test_loader)
    print(f"M-MSE: {results['M-MSE']:.2f}")

    logging.info("Evaluating classification error...")
    results['%Err'] = classification_error(model, train_loader, test_loader)
    print(f"Classification error: {results['%Err']:.2f}%")
    
    # logging.info("Fitting GMM on latent space...")
    # gmm = fit_gmm(train_loader, model, latent_dim, n_components)
    
    # logging.info("Evaluating Monte Carlo log-likelihood...") 
    # results['log_p(x)'] = evaluate_logpx(test_loader, model, gmm, latent_dim, num_samples)
    # print(f"Monte Carlo log-likelihood: {results['log_p(x)']:.2f}")

    return results

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Load datasets
    train_dataset = NumpyDataset(args.dataX, args.dataY)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_dataset = NumpyDataset(args.testX, args.testY)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Load the trained model
    save_directory = f'./{args.model}/model.pth'
    if not os.path.exists(save_directory):
        logging.warning(f"No saved model found at {save_directory}")
        return
    
    model = load_model(args.model, config, save_directory)
    
    # Evaluate the model
    test_results = evaluate(model, train_loader, test_loader, args.model, config['latent_dim'], config['n_components'], config['num_samples'], config)

if __name__ == "__main__":
    main()