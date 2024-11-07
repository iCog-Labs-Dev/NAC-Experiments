import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_mnist_loaders(batch_size=64, image_size=28, image_channels=1,
                      download=True, num_workers=2, pin_memory=True,
                      transform=None):
    if transform is None:
        # Define default transformation: resize and convert images to tensors
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

    # Load training dataset
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        transform=transform,
        download=download
    )

    # Load testing dataset
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        transform=transform,
        download=download
    )

    # Create DataLoader for training
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Create DataLoader for testing
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, test_loader

if __name__ == "__main__":
    # Example usage and verification
    train_loader, test_loader = get_mnist_loaders()
    print("MNIST dataset loaded successfully!")
    for images, labels in train_loader:
        print(f"Batch of images has shape: {images.shape}")
        # Verify pixel value range
        print(f"Min pixel value: {images.min().item()}")
        print(f"Max pixel value: {images.max().item()}")
        break
