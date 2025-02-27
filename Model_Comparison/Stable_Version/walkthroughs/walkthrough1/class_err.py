import numpy as np
import os
import sys

"""
Prints the classification error for the NGC models

Usage:
$ python class_err.py <output_dir> (e.g., gncn_t1)

"""

if len(sys.argv) < 2:
    print("Please provide the output directory.")
    sys.exit(1)

out_dir = sys.argv[1].rstrip('/')

train_acc = np.load(os.path.join(out_dir, "train_acc.npy"))
test_acc = np.load(os.path.join(out_dir, "test_acc.npy"))

train_err = (1.0 - train_acc) * 100.0
test_err = (1.0 - test_acc) * 100.0

print(f"Classification Error for {out_dir}:")
print(f"Train: {train_err[-1]:.2f}% || Test: {test_err[-1]:.2f}%")
