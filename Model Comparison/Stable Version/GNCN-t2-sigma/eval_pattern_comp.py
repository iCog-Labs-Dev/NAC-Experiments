import os
import sys, getopt
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '../')

import ngclearn.utils.transform_utils as transform
import ngclearn.utils.metric_utils as metric
import ngclearn.utils.io_utils as io_tools
from ngclearn.utils.data_utils import DataLoader
from ngclearn.utils.config import Config

options, remainder = getopt.getopt(sys.argv[1:], '', ["config=","gpu_id=","trial="])
cfg_fname = None
use_gpu = False
trial_num = 0
gpu_id = -1

for opt, arg in options:
    if opt in ("--config"):
        cfg_fname = arg.strip()
    elif opt in ("--gpu_id"):
        gpu_id = int(arg.strip())
        use_gpu = True
    elif opt in ("--trial"):
        trial_num = int(arg.strip())

# Setup GPU
mid = gpu_id
if mid >= 0:
    print(" > Using GPU ID {0}".format(mid))
    os.environ["CUDA_VISIBLE_DEVICES"]="{0}".format(mid)
    gpu_tag = '/GPU:0'
else:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    gpu_tag = '/CPU:0'

args = Config(cfg_fname)
out_dir = args.getArg("out_dir")
dev_xfname = args.getArg("test_xfname")
dev_batch_size = int(args.getArg("dev_batch_size"))
model_path = args.getArg("model_fname")
beta = args.getArg("beta")

def to_numpy(tensor):
    return tensor.numpy() if hasattr(tensor, 'numpy') else tensor

def calculate_mse(model_path, data_path, config_path):
    X = transform.binarize(tf.cast(np.load(data_path), dtype=tf.float32)).numpy()
    dataset = DataLoader(design_matrices=[("z0", X)], batch_size=dev_batch_size, disable_shuffle=True)
    agent = io_tools.deserialize(model_path)

    total_squared_error = 0.0
    total_unmasked_pixels = 0  # Count of pixels *only* in the unmasked region
    first_batch = True # a flag to perform visualization only once

    for batch in dataset:
        x_name, x = batch[0]
        x = transform.binarize(x).numpy()
        batch_size = x.shape[0]

        x_2d = x.reshape(batch_size, 28, 28)

        mask = np.ones_like(x_2d, dtype=np.float32)
        mask[:, :, 14:] = 0  # Mask the right half
        mask_1d = mask.reshape(batch_size, 784) # Reshape the mask for easy calculation
        num_unmasked_pixels = np.sum(1 - mask_1d)

        x_masked = x_2d.reshape(batch_size, 784)  # Reshape here.

        x_hat = agent.settle(x_masked)  # Reconstructed image (batch_size, 784)

        # Calculate squared error for all 784 pixels per image using *your* mse function
        # Here the dimension of each image is N=784
        squared_errors = metric.mse(x_hat, x) # returns a vector (batch_size, )
        # Calculate the mean squared error over the *unmasked* regions only

        masked_errors = squared_errors
        total_squared_error += np.sum(masked_errors)

        total_unmasked_pixels += num_unmasked_pixels
        agent.clear()

        # Visualization (only for the first batch)
        if first_batch:
            plt.figure(figsize=(15, 5))

            # Plot the original image
            plt.subplot(1, 3, 1)
            plt.title('Original Image')
            plt.imshow(to_numpy(x[9]).reshape(28, 28), cmap='gray')
            plt.axis('off')

            # Plot the masked image
            plt.subplot(1, 3, 2)
            plt.title('Masked Image')
            plt.imshow(to_numpy(x_masked[9]).reshape(28, 28), cmap='gray')
            plt.axis('off')

            # Plot the reconstructed image
            plt.subplot(1, 3, 3)
            plt.title('Reconstructed Image')
            plt.imshow(to_numpy(x_hat[9]).reshape(28, 28), cmap='gray')
            plt.axis('off')

            # Save the visualization
            if os.path.exists('image_reconstruction.png'):
                os.remove('image_reconstruction.png')

            plt.tight_layout()
            plt.savefig('image_reconstruction.png')
            plt.close()
            first_batch = False # stop visualization for other batches

    # Calculate the MEAN squared error, averaged over ALL UNMASKED PIXELS
    MSE = (total_squared_error / total_unmasked_pixels) * batch_size
    return MSE

with tf.device(gpu_tag):
    mse = calculate_mse(model_path, dev_xfname, dev_batch_size)
    print(f"Mean Squared Error: {mse}")
