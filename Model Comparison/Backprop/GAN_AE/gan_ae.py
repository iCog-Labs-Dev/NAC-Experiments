import logging
import sys
import os
from torch.utils.data import DataLoader
import getopt as gopt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.numpy_dataset import NumpyDataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  
)

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

print(f"Train-set: X: {dataX} | Y: {dataY}")
print(f"Dev-set: X: {devX} | Y: {devY}")
print(f"Test-set: X: {testX} | Y: {testY}")

# Load datasets
train_dataset = NumpyDataset(dataX, dataY)
train_loader = DataLoader(dataset=train_dataset, batch_size=200, shuffle=True)

dev_dataset = NumpyDataset(devX, devY)
dev_loader = DataLoader(dataset=dev_dataset, batch_size=200, shuffle=False)

test_dataset = NumpyDataset(testX, testY)
test_loader = DataLoader(dataset=test_dataset, batch_size = 200, shuffle = False)