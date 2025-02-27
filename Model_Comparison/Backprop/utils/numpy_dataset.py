import torch
import numpy as np
from torch.utils.data import Dataset

class NumpyDataset(Dataset):
    """
    This class provides an interface to load features (dataX) and optional labels (dataY) from 
    NumPy `.npy` files.

    Args:
        dataX (str): Path to the NumPy file containing the feature data.
        dataY (str, optional): Path to the NumPy file containing the label data

    Attributes:
        dataX (np.ndarray): Loaded feature data from the specified NumPy file.
        dataY (np.ndarray or None): Loaded label data from the specified NumPy file.

    Methods:
        __len__(): Returns the total number of samples in the dataset.
        __getitem__(idx): Returns a tuple of the feature tensor and label tensor for the given index.

    """
    def __init__(self, dataX, dataY=None):
        self.dataX = np.load(dataX)
        if dataY is not None:
             self.dataY = np.argmax(np.load(dataY_path), axis=1)  # Convert one-hot to class indices (0–9)
        else:
            self.dataY = None

    def __len__(self):
        return len(self.dataX)

    def __getitem__(self, idx):
        data = torch.tensor(self.dataX[idx], dtype=torch.float32)
        label = torch.tensor(self.dataY[idx], dtype=torch.long) if self.dataY is not None else None
        return data, label
