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