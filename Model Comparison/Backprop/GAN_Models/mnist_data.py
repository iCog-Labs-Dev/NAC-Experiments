# minst_data.py

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_mnist_loaders(batch_size=64, image_size=28, image_channels=1, download=True):
    """
    Returns training and testing DataLoaders for the MNIST dataset.
    """
    # Define transformation to convert images to tensors and normalize them
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # Load training dataset
    train_dataset = datasets.MNIST(root='./data',
                                   train=True,
                                   transform=transform,
                                   download=download)

    # Load testing dataset
    test_dataset = datasets.MNIST(root='./data',
                                  train=False,
                                  transform=transform,
                                  download=download)

    # Create DataLoader for training
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    # Create DataLoader for testing
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = get_mnist_loaders()
    print("MNIST dataset loaded successfully!")
    for images, labels in train_loader:
        print(f"Batch of images has shape: {images.shape}")
        break
