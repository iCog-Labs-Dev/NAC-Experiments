import os
import torch
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from rae_model import RegularizedAutoencoder
from rae import NumpyDataset
from torch.utils.data import DataLoader

# Paths to test data
testX = "../../../data/mnist/testX.npy"
testY = "../../../data/mnist/testY.npy"

# Validate paths
if not os.path.exists(testX) or not os.path.exists(testY):
    raise FileNotFoundError(f"Test data files not found: {testX}, {testY}")

# Model setup
latent_dim = 64  
model = RegularizedAutoencoder(latent_dim=latent_dim)
model_path = "trained_model.pth"

# Load model weights (ensure weights-only save)
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model weights not found at {model_path}")
model.load_state_dict(torch.load(model_path))  
model.eval()

# Load test dataset
test_dataset = NumpyDataset(testX, testY)
test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)