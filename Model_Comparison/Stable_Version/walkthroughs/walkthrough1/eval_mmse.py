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

"""
    Evaluates a neural generative completion model. Masks the right half of images,
    reconstructs masked images, and then calculates masked mean squared error (M-MSE).

    Usage: python eval_mmse.py --config=path_to_config/fit.cfg --gpu_id=-1
"""
def to_numpy(tensor):
    return tensor.numpy() if hasattr(tensor, 'numpy') else tensor

def calculate_mmse(model_path, data_path, config_path):
    X = transform.binarize(tf.cast(np.load(data_path), dtype=tf.float32)).numpy()
    dataset = DataLoader(design_matrices=[("z0", X)], batch_size=dev_batch_size, disable_shuffle=True)
    agent = io_tools.deserialize(model_path)

    total_squared_error = 0.0
    total_unmasked_pixels = 0  
    first_batch = True  

    for batch in dataset:
        x_name, x = batch[0]
        x = transform.binarize(x).numpy()
        batch_size = x.shape[0]

        x_2d = x.reshape(batch_size, 28, 28)
        M = np.zeros_like(x_2d, dtype=np.float32)
        M[:, :, 14:] = 1  

        masked_x = x_2d * (1 - M)
        masked_x_flat = masked_x.reshape(batch_size, 784)
        x_hat = agent.settle(masked_x_flat)
        x_flat = x_2d.reshape(batch_size, 784)
        M_flat = M.reshape(batch_size, 784)

        error = (x_hat - x_flat) * (1 - M_flat)
        squared_error_per_image = np.sum(error * error, axis=1)  
        total_squared_error += np.sum(squared_error_per_image)
        total_unmasked_pixels += np.sum(1 - M_flat)  

        agent.clear()

        if first_batch:
            plt.figure(figsize=(15, 5))

            # Plot the original, masked and reconstructed image
            plt.subplot(1, 3, 1)
            plt.title('Original Image')
            plt.imshow(x_2d[9], cmap='gray')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.title('Masked Image')
            plt.imshow(masked_x[9], cmap='gray')
            plt.axis('off')

            x_hat_np = to_numpy(x_hat)
            plt.subplot(1, 3, 3)
            plt.title('Reconstructed Image')
            plt.imshow(x_hat_np[9].reshape(28, 28), cmap='gray')
            plt.axis('off')

            # Save the visualization
            if os.path.exists('image_reconstruction.png'):
                os.remove('image_reconstruction.png')
            plt.tight_layout()
            plt.savefig('image_reconstruction.png')
            plt.close()
            first_batch = False

    mmse = (total_squared_error / total_unmasked_pixels)* batch_size
    return mmse

with tf.device(gpu_tag):
    mmse = calculate_mmse(model_path, dev_xfname, dev_batch_size)
    print(f"Masked MSE: {mmse:.2f}")