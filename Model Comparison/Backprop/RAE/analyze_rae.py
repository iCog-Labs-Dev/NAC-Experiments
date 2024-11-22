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

def extract_latent_representations(model, loader):
    model.eval()
    latent_representations = []
    labels = []
    with torch.no_grad():
        for data, label in loader:
            data = data.view(data.size(0), -1)  # Flatten the input
            latent = model.encoder(data)  # Get the latent representation
            latent_representations.append(latent.cpu().numpy())
            # Convert one-hot labels to scalar class indices
            labels.append(label.argmax(dim=1).cpu().numpy())
    latent_representations = np.concatenate(latent_representations, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    # Debugging size
    print(f"Latent representations shape: {latent_representations.shape}")
    print(f"Labels shape: {labels.shape}")
    
    assert latent_representations.shape[0] == labels.shape[0], "Mismatch between latent representations and labels"
    return latent_representations, labels
