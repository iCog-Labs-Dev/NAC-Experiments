import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
from rae_model import RegularizedAutoencoder  
import sys, getopt as gopt, time
class NumpyDataset(Dataset):
    def __init__(self, dataX, dataY=None):
        self.dataX = np.load(dataX)
        if dataY is not None:
            self.dataY = np.load(dataY).reshape(-1)
        else:
            self.dataY = None

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

train_dataset = NumpyDataset(dataX, dataY)
train_loader = DataLoader(dataset=train_dataset, batch_size=200, shuffle=True)

dev_dataset = NumpyDataset(devX, devY)
dev_loader = DataLoader(dataset=dev_dataset, batch_size=200, shuffle=False)

test_dataset = NumpyDataset(testX, testY)
test_loader = DataLoader(dataset=test_dataset, batch_size = 200, shuffle = False)

def train(model, loader, optimizer, epoch):
    model.train()
    bce_losses = []

    for batch_idx, (data, _) in enumerate(tqdm(loader)):
        data = (data > 0.5).float()
        data = data.view(data.size(0), -1)  
        
        optimizer.zero_grad()
        reconstructed = model(data)
        reconstructed = reconstructed.view(reconstructed.size(0), -1) 
    
        # Loss for reconstruction
        bce_loss = F.binary_cross_entropy(reconstructed, data, reduction="sum")
        bce_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bce_losses.append(bce_loss.item() / data.size(0))

    avg_bce = np.mean(bce_losses)
    std_bce = np.std(bce_losses)
    return avg_bce, std_bce

def evaluate(model, loader):
    model.eval()
    bce_losses = []
    
    with torch.no_grad():
        for data, _ in loader:
            data = (data > 0.5).float()
            data = data.view(data.size(0), -1) 
            reconstructed = model(data)
            reconstructed = reconstructed.view(reconstructed.size(0), -1) 
            
            bce_loss = F.binary_cross_entropy(reconstructed, data, reduction="sum")
            bce_losses.append(bce_loss.item() / data.size(0))

    avg_bce = np.mean(bce_losses)
    std_bce = np.std(bce_losses)
    return avg_bce, std_bce

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

def evaluate_classification(model, train_loader, test_loader):
    model.eval()
    train_latents, train_labels = [], []
    test_latents, test_labels = [], []

    with torch.no_grad():
        for data, label in train_loader:
            data = data.view(data.size(0), -1)
            if label is not None:
                z = model.encoder(data)
                train_latents.append(z.cpu().numpy())
                train_labels.append(label.cpu().numpy())
    
    with torch.no_grad():
        for data, label in test_loader:
            data = data.view(data.size(0), -1)
            if label is not None:
                z = model.encoder(data)
                test_latents.append(z.cpu().numpy())
                test_labels.append(label.cpu().numpy())

    train_latents = np.vstack(train_latents)
    train_labels = np.hstack(train_labels).reshape(-1)
    test_latents = np.vstack(test_latents)
    test_labels = np.hstack(test_labels).reshape(-1)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_latents, train_labels)

    predictions = clf.predict(test_latents)
    err = 100 * (1 - accuracy_score(test_labels, predictions))
    return err

input_dim = 28 * 28
hidden_dims = [360, 360]
latent_dim = 20
model = RegularizedAutoencoder(latent_dim=latent_dim, input_dim=input_dim, hidden_dims=hidden_dims)
optimizer = optim.SGD(model.parameters(), lr=0.1)
num_epochs = 50 

# Start time profiling
sim_start_time = time.time()
print("--------------- Training ---------------")
for epoch in range(1, num_epochs + 1): 
    train_bce, train_std = train(model, train_loader, optimizer, epoch)
    print(f'Epoch [{epoch}/{num_epochs}]')
    print(f'Train BCE: {train_bce:.4f}, Deviation: {train_std}')

# Stop time profiling
sim_time = time.time() - sim_start_time
print(f"Training Time = {sim_time:.4f} seconds")

# print("--------------- Fitting GMM on latent space ---------------")
# gmm_train = fit_gmm_on_latent(model, train_loader, n_components=75)
# print("GMM fitting completed.")

print("--------------- Evaluating ---------------")
eval_bce, eval_std = evaluate(model, dev_loader)
print(f'Eval BCE: {eval_bce:.4f}, Deviation: {eval_std}')

print("--------------- Testing ---------------")
inference_start_time = time.time()
test_bce, test_std = evaluate(model, test_loader)
inference_time = time.time() - inference_start_time
print(f'Test BCE: {test_bce:.4f}, Deviation: {test_std}')
print(f"Inference Time = {inference_time:.4f} seconds")

# print("--------------- Masked MSE ---------------")
# masked_mse_result = masked_mse(model, test_loader)
# print(f"Masked MSE: {masked_mse_result:.4f}")

err = evaluate_classification(model, train_loader, test_loader)
print(f"Classifation Error: {err:.2f}")