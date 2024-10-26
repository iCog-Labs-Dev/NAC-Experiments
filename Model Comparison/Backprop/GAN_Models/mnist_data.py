import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Set random seed for reproducibility
torch.manual_seed(42)

# Define hyperparameters
batch_size = 64
image_size = 28
image_channels = 1

# Define transformation to convert images to tensors and normalize them
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# Automatically download and load the MNIST dataset
mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

# Create DataLoader for batching during training
data_loader = DataLoader(dataset=mnist_dataset, batch_size=batch_size, shuffle=True)

# Check if dataset is loaded properly
if __name__ == "__main__":
    print("MNIST dataset loaded successfully!")
    for images, labels in data_loader:
        print(f"Batch of images has shape: {images.shape}")
        break
