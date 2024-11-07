import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from rae_model import RegularizedAutoencoder  
import sys, getopt as gopt
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
                                 ["dataX=", "dataY=", "devX=", "devY=", "verbosity="]
                                 )

dataX = "../../../data/mnist/trainX.npy"
dataY = "../../../data/mnist/trainY.npy"
devX = "../../../data/mnist/validX.npy"
devY = "../../../data/mnist/validY.npy"
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
    elif opt in ("--verbosity"):
        verbosity = int(arg.strip())
print("Train-set: X: {} | Y: {}".format(dataX, dataY))
print("  Dev-set: X: {} | Y: {}".format(devX, devY))


latent_dim = 64  
model = RegularizedAutoencoder(latent_dim=latent_dim)
optimizer = optim.Adam(model.parameters(), lr=0.0002)

train_dataset = NumpyDataset(dataX, dataY)
train_loader = DataLoader(dataset=train_dataset, batch_size=200, shuffle=True)

dev_dataset = NumpyDataset(devX, devY)
dev_loader = DataLoader(dataset=dev_dataset, batch_size=200, shuffle=False)

def train(model, loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, _) in enumerate(tqdm(loader)):
        data = data.view(data.size(0), -1)  # Flatten the input data to shape (batch_size, input_dim)
        optimizer.zero_grad()

        reconstructed = model(data)
        reconstructed = reconstructed.view(reconstructed.size(0), -1)  # Flatten the output to (batch_size, input_dim)

        # MSE loss for reconstruction
        loss = F.mse_loss(reconstructed, data)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    print(f'Epoch [{epoch}], MSE: {avg_loss:.4f}')
    return avg_loss

def evaluate(model, loader):
    model.eval()
    eval_loss = 0.0
    total_correct = 0
    total_samples = 0
    threshold = 0.1  
    total_kld = 0.0
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(loader):
            data = data.view(data.size(0), -1) 
            reconstructed = model(data)
            reconstructed = reconstructed.view(reconstructed.size(0), -1) 

            loss = F.mse_loss(reconstructed, data)  
            eval_loss += loss.item()
            
            data_np = data.cpu().numpy()
            reconstructed_np = reconstructed.cpu().numpy()

            # Calculating KLD
            kld = measure_KLD(data_np, reconstructed_np)
            total_kld = kld.item()

            # Calculating accuracy
            diff = torch.abs(reconstructed - data) 
            correct = torch.sum(diff < threshold, dim=1)  
            total_correct += correct.sum().item()  
            total_samples += data.size(0) 

    avg_loss = eval_loss / len(loader)
    accuracy = total_correct / (total_samples * data.size(1)) * 100
    avg_kld = total_kld / len(loader)
    
    print(f'MSE: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, KLD: {avg_kld:.4f}')

    return avg_loss, accuracy, avg_kld

num_epochs = 50

for epoch in range(1, num_epochs + 1):
    train_loss = train(model, train_loader, optimizer, epoch)
    eval_loss, eval_accuracy, eval_kld = evaluate(model, dev_loader)

    print(f'Epoch [{epoch}/{num_epochs}]')
    print(f'Train MSE: {train_loss:.4f}, Eval MSE: {eval_loss:.4f}, Eval Accuracy: {eval_accuracy:.2f}%, Eval KLD: {eval_kld:.4f}')