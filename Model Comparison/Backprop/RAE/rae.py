import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from rae_model import RegularizedAutoencoder  
import sys, getopt as gopt, time
from ngclearn.utils.metric_utils import measure_KLD
class NumpyDataset(Dataset):
    def __init__(self, dataX, dataY=None):
        self.dataX = np.load(dataX) 
        self.dataY = np.load(dataY) if dataY is not None else None 

    def __len__(self):
        return len(self.dataX)

    def __getitem__(self, idx):
        data = torch.tensor(self.dataX[idx], dtype=torch.float32)
        label = torch.tensor(self.dataY[idx], dtype=torch.long) if self.dataY is not None else None
        return data, label


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

latent_dim = 64  
model = RegularizedAutoencoder(latent_dim=latent_dim)
optimizer = optim.SGD(model.parameters(), lr=0.001)

train_dataset = NumpyDataset(dataX, dataY)
train_loader = DataLoader(dataset=train_dataset, batch_size=200, shuffle=True)

dev_dataset = NumpyDataset(devX, devY)
dev_loader = DataLoader(dataset=dev_dataset, batch_size=200, shuffle=False)

test_dataset = NumpyDataset(testX, testY)
test_loader = DataLoader(dataset=test_dataset, batch_size = 200, shuffle = False)

# Function to rescale gradients
def rescale_gradients(model, radius=5):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    scale = radius / max(radius, total_norm)
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.mul_(scale)

def train(model, loader, optimizer, epoch):
    model.train()
    total_bce = 0.0

    for batch_idx, (data, _) in enumerate(tqdm(loader)):
        data = data / 255.0
        data = data.view(data.size(0), -1)  # Flatten the input data to shape (batch_size, input_dim)
        optimizer.zero_grad()

        reconstructed = model(data)
        reconstructed = reconstructed.view(reconstructed.size(0), -1)  # Flatten the output to (batch_size, input_dim)
    

        # Loss for reconstruction
        bce_loss = F.binary_cross_entropy(reconstructed, data)
        bce_loss.backward()
        rescale_gradients(model, radius=5)
        optimizer.step()

        total_bce += bce_loss.item()
        torch.save(model.state_dict(), "trained_model.pth")

    avg_bce = total_bce / len(loader)
    return avg_bce

def evaluate(model, loader):
    model.eval()
    total_bce = 0.0
    
    with torch.no_grad():
        for data, _ in loader:
            data = data / 255.0
            data = data.view(data.size(0), -1) 
            reconstructed = model(data)
            reconstructed = reconstructed.view(reconstructed.size(0), -1) 
            
            # Calculating BCE
            bce_loss = F.binary_cross_entropy(reconstructed, data)
            total_bce += bce_loss.item()

    avg_bce = total_bce / len(loader)
    return avg_bce

def fit_gmm_on_latent(model, loader, n_components=75):
    model.eval()
    latents = []
    with torch.no_grad():
        for data, _ in loader:
            data = data.view(data.size(0), -1)
            z = model.encoder(data)
            latents.append(z.cpu().numpy())
    latents = np.vstack(latents)

    gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    gmm.fit(latents)
    print("GMM fitted on latent representations.")
    return gmm

def masked_mse(model, loader):
    model.eval()
    total_mse = 0.0
    total_samples = 0
    with torch.no_grad():
        for data, _ in loader:
            data = data / 255.0  # Normalize input
            data = data.view(data.size(0), -1)

            # Mask exactly half of the image columns
            mask = torch.ones_like(data, dtype=torch.bool)  # Ensure mask is Boolean
            mask[:, : data.size(1) // 2] = 0  # Mask left half of the image

            masked_data = data * mask.float()  # Apply mask

            reconstructed = model(masked_data).view(data.size(0), -1)

            # Compute MSE over the masked (hidden) parts
            mse = F.mse_loss(reconstructed[~mask], data[~mask], reduction="sum")
            total_mse += mse.item()
            total_samples += data.size(0)

    # Normalize by the total number of masked elements
    avg_mse = total_mse / (total_samples * data.size(1) // 2)
    return avg_mse

num_epochs = 50 

# Start time profiling
sim_start_time = time.time()
print("--------------- Training ---------------")
for epoch in range(1, num_epochs + 1): 
    train_bce = train(model, train_loader, optimizer, epoch)
    print(f'Epoch [{epoch}/{num_epochs}]')
    print(f'Train BCE: {train_bce:.4f}')

# Stop time profiling
sim_time = time.time() - sim_start_time
print(f"Training Time = {sim_time:.4f} seconds")

print("--------------- Fitting GMM on latent space ---------------")
gmm_train = fit_gmm_on_latent(model, train_loader, n_components=75)
print("GMM fitting completed.")

print("--------------- Evaluating ---------------")
eval_bce = evaluate(model, dev_loader)
print(f'Eval BCE: {eval_bce:.4f}')

print("--------------- Testing ---------------")
inference_start_time = time.time()
test_bce = evaluate(model, test_loader)
inference_time = time.time() - inference_start_time
print(f'Test BCE: {test_bce:.4f}')
print(f"Inference Time = {inference_time:.4f} seconds")

print("--------------- Masked MSE ---------------")
masked_mse_result = masked_mse(model, test_loader)
print(f"Masked MSE: {masked_mse_result:.4f}")