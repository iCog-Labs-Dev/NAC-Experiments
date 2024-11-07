import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Custom Dataset class to handle .npy files with normalization
class NumpyMNISTDataset(Dataset):
    def __init__(self, data_path, label_path):
        # Load data and labels from .npy files
        self.data = np.load(data_path).astype(np.float32) / 255.0  # Normalize to [0, 1]
        self.labels = np.load(label_path).astype(np.int64)  # Convert labels to int64
        self.data = torch.tensor(self.data).unsqueeze(1)  # Add channel dimension for grayscale images

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Paths to your .npy files
trainX_path = "../../../data/mnist/trainX.npy"
trainY_path = "../../../data/mnist/trainY.npy"
validX_path = "../../../data/mnist/validX.npy"
validY_path = "../../../data/mnist/validY.npy"

# Create datasets
train_dataset = NumpyMNISTDataset(trainX_path, trainY_path)
valid_dataset = NumpyMNISTDataset(validX_path, validY_path)

# Create data loaders
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Define a simple model (e.g., a basic neural network for MNIST)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
        data, labels = data.to(device), labels.to(device)

        # Convert labels to class indices for training
        labels = torch.argmax(labels, dim=1) 
        outputs = model(data)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader):.4f}")

# Validation loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, labels in valid_loader:
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        # Convert labels to class indices for validation
        labels = torch.argmax(labels, dim=1) 
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Validation Accuracy: {accuracy:.2f}%")

# Save model
torch.save(model.state_dict(), "simple_nn_mnist.pth")