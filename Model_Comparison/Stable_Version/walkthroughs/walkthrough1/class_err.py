import numpy as np

train_acc = np.load("gncn_t1/train_acc.npy")
test_acc = np.load("gncn_t1/test_acc.npy")

train_err = (1.0 - train_acc) * 100.0
test_err = (1.0 - test_acc) * 100.0

print("Classification Error Train: {0:.2f}% || Test: {1:.2f}%".format(
    train_err[-1], test_err[-1]))