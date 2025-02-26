import sys
import os
import time
import torch
import random
import logging
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from utils.numpy_dataset import NumpyDataset

from GVAE.GVAE_model import VAE 
from GVAE_CV.GVAE_CV_model import GVAE_CV
from RAE.rae_model import RegularizedAutoencoder
from GAN_AE.gan_ae_model import GANAE

"""
Usage: python3 train_model.py --model=model_name --config=path_to/config.py
"""

# Set random seed for reproducibility
seed_value = 69
torch.manual_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  
)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--model", type=str, required=True, help="Model type to train.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--dataX", type=str, default="../../data/mnist/trainX.npy", help="Path to training data X.")
    parser.add_argument("--dataY", type=str, default="../../data/mnist/trainY.npy", help="Path to training data Y.")
    parser.add_argument("--verbosity", type=int, default=0, help="Verbosity level.")
    return parser.parse_args()

def load_config(config_file_path: str) -> dict:
    """Load configuration from a Python file."""
    if not os.path.exists(config_file_path):
        raise ValueError(f"Configuration file not found: {config_file_path}")
    
    config = {}
    with open(config_file_path) as f:
        exec(f.read(), {}, config)
    return config

def load_model(model_type: str, config: dict) -> torch.nn.Module:
    """Initialize and return a model based on the type."""
    model_classes = {
        "GVAE_CV": GVAE_CV,
        "RAE": RegularizedAutoencoder,
        "GAN_AE": GANAE,
        "GVAE": VAE,
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Model type not recognized: {model_type}")
    
    model_class = model_classes[model_type]
    
    # Initialize model
    valid_params = {k: v for k, v in config.items() if k in model_class.__init__.__code__.co_varnames}
    model = model_class(**valid_params)
    return model

def compute_loss(model_type, model, data, config):
    """Compute the loss for a given model and data."""
    match model_type:
        case "GVAE_CV":
            recon_x, mu, _, l2_penalty = model(data)
            kl_loss = (-0.5 * torch.sum(1 + config['fixed_variance'] - mu.pow(2) - torch.exp(config['fixed_variance']))) / 20
            reconstruction_loss = F.binary_cross_entropy(recon_x.view(recon_x.size(0), -1), data, reduction="sum")
            total_loss = reconstruction_loss + kl_loss + l2_penalty
        case "RAE":
            reconstructed, l2_penalty = model(data)
            reconstructed = reconstructed.view(reconstructed.size(0), -1)
            bce_loss = F.binary_cross_entropy(reconstructed, data, reduction="sum")
            total_loss = bce_loss + l2_penalty
        case "GAN_AE":
            x_recon, real_or_fake, _, _, l2_penalty = model(data)
            reconstruction_loss = F.binary_cross_entropy(x_recon, data, reduction="sum")
            discriminator_loss = F.binary_cross_entropy(real_or_fake, torch.ones_like(real_or_fake), reduction="sum")
            total_loss = reconstruction_loss + discriminator_loss + l2_penalty
        case _:
            recon_data, mu, logvar, l2_penalty = model(data)
            kl_loss = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))) / 20
            bce_loss = F.binary_cross_entropy(recon_data.view(recon_data.size(0), -1), data, reduction="sum")
            total_loss = bce_loss + kl_loss + l2_penalty
    return total_loss

def train(model, loader, optimizer, model_type, config):
    """Training loop."""
    model.train()
    total_losses = []
    
    for batch_idx, (data, _) in enumerate(tqdm(loader)):
        data = (data > 0.5).float().view(data.size(0), -1)
        optimizer.zero_grad()
        
        total_loss = compute_loss(model_type, model, data, config)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_losses.append(total_loss.item() / data.size(0))

    avg_bce = np.mean(total_losses)
    return avg_bce

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Load dataset
    train_dataset = NumpyDataset(args.dataX, args.dataY)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Initialize the model
    model = load_model(args.model, config)
    
    # Set the optimizer based on model type
    if args.model == "GAN_AE":
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])

    # Training the model
    logging.info("Starting model training...")
    sim_start_time = time.time()
    
    for epoch in range(1, config['num_epochs'] + 1):
        avg_bce = train(model, train_loader, optimizer, args.model, config)
        logging.info(f'Epoch [{epoch}/{config["num_epochs"]}]  Train BCE = {avg_bce:.2f}')
    
    sim_time = time.time() - sim_start_time
    logging.info(f"Total training time: {sim_time:.2f} sec")
    
    # Save the trained model
    save_directory = f'./{args.model}/model.pth'
    torch.save(model.state_dict(), save_directory)

if __name__ == "__main__":
    main()