"""
Copyright (C) 2021 Alexander G. Ororbia II - All Rights Reserved
You may use, distribute and modify this code under the
terms of the BSD 3-clause license.

You should have received a copy of the BSD 3-clause license with
this file. If not, please write to: ago@cs.rit.edu
"""

import os
import sys, getopt, optparse
import pickle
sys.path.insert(0, '../')
import tensorflow as tf
import numpy as np
import time

# import general simulation utilities
from ngclearn.utils.config import Config
import ngclearn.utils.transform_utils as transform
import ngclearn.utils.metric_utils as metric
import ngclearn.utils.io_utils as io_tools
from ngclearn.utils.data_utils import DataLoader

# import model from museum to train
from ngclearn.museum.gncn_t1 import GNCN_t1
from ngclearn.museum.gncn_t1_sigma import GNCN_t1_Sigma
from ngclearn.museum.gncn_pdh import GNCN_PDH

# For classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

seed = 69
tf.random.set_seed(seed=seed)
np.random.seed(seed)

"""
################################################################################
Walkthrough #1 File:
Trains/fits an NGC model to a dataset of sensory patterns, e.g., the MNIST
database. Note that this script will sequentially run multiple trials/seeds if an
experimental multi-trial setup is required (the tutorial only requires 1 trial).

Usage:
$ python sim_train.py --config=/path/to/fit.cfg --gpu_id=0 --n_trials=1

@author Alexander Ororbia
################################################################################
"""

###############################################################################
# Classification Evaluation Helper
###############################################################################
def evaluate_classification(agent, train_loader, test_loader):
    train_latents = []
    train_labels  = []
    for batch in train_loader:
        x = batch[0][1]
        y = batch[1][1]  # shape might be (batch_size,) or (batch_size, 10)...

        # 1) Inference
        agent.settle(x, calc_update=False)
        z_top = agent.ngc_model.extract(node_name="z1", node_var_name="A")  # or "z2" if your model has that
        z_top_np = z_top.numpy()
        
        train_latents.append(z_top_np)

        # 2) Make sure labels are 1D
        if len(y.shape) > 1 and y.shape[1] > 1:
            # Probably one-hot -> convert to integer
            y = tf.argmax(y, axis=1)
        else:
            # Possibly shape [batch_size, 1] -> squeeze
            y = tf.reshape(y, [-1])

        y = y.numpy()
        train_labels.append(y)
        agent.clear()

    # Combine all into final arrays
    train_latents = np.vstack(train_latents)
    train_labels  = np.hstack(train_labels)  # => shape [N]

    # Fit logistic regression
    clf = LogisticRegression(penalty='none', max_iter=50, multi_class="multinomial", solver="lbfgs")
    clf.fit(train_latents, train_labels)

    # Do same for test set
    test_latents = []
    test_labels  = []
    for batch in test_loader:
        x = batch[0][1]
        y = batch[1][1]

        agent.settle(x, calc_update=False)
        z_top = agent.ngc_model.extract(node_name="z1", node_var_name="A")
        z_top_np = z_top.numpy()
        test_latents.append(z_top_np)

        if len(y.shape) > 1 and y.shape[1] > 1:
            y = tf.argmax(y, axis=1)
        else:
            y = tf.reshape(y, [-1])
        y = y.numpy()
        test_labels.append(y)

        agent.clear()

    test_latents = np.vstack(test_latents)
    test_labels  = np.hstack(test_labels)

    # Predict and compute error
    test_preds = clf.predict(test_latents)
    acc = accuracy_score(test_labels, test_preds)
    err = 100.0 * (1.0 - acc)

    print(f"Classification Error on test set: {err:.2f}%")
    return err


###############################################################################
# Main Training Script
###############################################################################
options, remainder = getopt.getopt(sys.argv[1:], '', ["config=","gpu_id=","n_trials="])
# GPU arguments
cfg_fname = None
use_gpu = False
n_trials = 1
gpu_id = -1
for opt, arg in options:
    if opt in ("--config"):
        cfg_fname = arg.strip()
    elif opt in ("--gpu_id"):
        gpu_id = int(arg.strip())
        use_gpu = True
    elif opt in ("--n_trials"):
        n_trials = int(arg.strip())

mid = gpu_id
if mid >= 0:
    print(" > Using GPU ID {0}".format(mid))
    os.environ["CUDA_VISIBLE_DEVICES"]="{0}".format(mid)
    gpu_tag = '/GPU:0'
else:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    gpu_tag = '/CPU:0'

save_marker = 1

args = Config(cfg_fname)

model_type = args.getArg("model_type")
out_dir = args.getArg("out_dir")
batch_size = int(args.getArg("batch_size")) # e.g., 128
dev_batch_size = int(args.getArg("dev_batch_size")) # e.g., 128

eta = float(args.getArg("eta"))    # learning rate/step size (optimzation)
num_iter = int(args.getArg("num_iter")) # number training iterations

# Filenames for data
train_xfname = args.getArg("train_xfname") # e.g. ../data/mnist/trainX.npy
train_yfname = args.getArg("train_yfname") # e.g. ../data/mnist/trainY.npy
dev_xfname   = args.getArg("dev_xfname")   # e.g. ../data/mnist/validX.npy
dev_yfname   = args.getArg("dev_yfname")   # e.g. ../data/mnist/validY.npy
test_xfname  = args.getArg("test_xfname")  # e.g. ../data/mnist/testX.npy
test_yfname  = args.getArg("test_yfname")  # e.g. ../data/mnist/testY.npy

# Load the data into memory
print(" >> Loading data into memory...")

X_train = transform.binarize(tf.cast(np.load(train_xfname), dtype=tf.float32)).numpy()
y_train = np.load(train_yfname)  # shape: (num_train,)

X_dev   = transform.binarize(tf.cast(np.load(dev_xfname), dtype=tf.float32)).numpy()
y_dev   = np.load(dev_yfname)

X_test  = transform.binarize(tf.cast(np.load(test_xfname), dtype=tf.float32)).numpy()
y_test  = np.load(test_yfname)

x_dim = X_train.shape[1]
args.setArg("x_dim", x_dim)

# Create data loaders
train_set = DataLoader(design_matrices=[("z0", X_train), ("y", y_train)],
                       batch_size=batch_size,
                       disable_shuffle=False)

dev_set   = DataLoader(design_matrices=[("z0", X_dev), ("y", y_dev)],
                       batch_size=dev_batch_size,
                       disable_shuffle=True)

test_set  = DataLoader(design_matrices=[("z0", X_test), ("y", y_test)],
                       batch_size=dev_batch_size,
                       disable_shuffle=True)

def eval_model(agent, dataset, calc_ToD, verbose=False):
    """
    Evaluates performance (ToD, Lx) of agent on a data sample.
    """
    ToD = 0.0 # total discrepancy over entire data pool
    Lx = 0.0  # metric/loss over entire data pool
    N = 0.0   # number of samples
    for batch in dataset:
        # batch is [("z0", X), ("y", Y)] but we only need X here
        x = batch[0][1]
        N += x.shape[0]
        x_hat = agent.settle(x, calc_update=False)  # iterative inference

        # update tracked fixed-point losses
        Lx_batch = tf.reduce_sum(metric.bce(x_hat, x))
        Lx += Lx_batch
        ToD_batch = calc_ToD(agent)
        ToD += ToD_batch

        agent.clear()
        if verbose:
            print("\r ToD {:.6f}  Lx {:.6f} over {:.0f} samples...".format(
                  (ToD/(N*1.0)), (Lx/(N*1.0)), N), end="")
    if verbose:
        print()
    Lx  = Lx / N
    ToD = ToD / N
    return ToD, Lx

################################################################################
# Start simulation
################################################################################
with tf.device(gpu_tag):
    def calc_ToD(agent):
        """
        Measures the total discrepancy (ToD) of a given NGC model
        by summing (negative) the energies in each layer's error node.
        """
        # For GNCN_t1: e0, e1, e2 exist
        # Adjust if your model has different naming or number of layers
        L2 = agent.ngc_model.extract(node_name="e2", node_var_name="L")
        L1 = agent.ngc_model.extract(node_name="e1", node_var_name="L")
        L0 = agent.ngc_model.extract(node_name="e0", node_var_name="L")
        ToD_val = -(L0 + L1 + L2)
        return float(ToD_val)

    for trial in range(n_trials):
        agent = None
        print(" >> Building ", model_type)
        if model_type == "GNCN_t1":
            agent = GNCN_t1(args)
        elif model_type == "GNCN_t1_Sigma":
            agent = GNCN_t1_Sigma(args)
        elif model_type == "GNCN_PDH" or model_type == "GNCN_t2_LSigma_PDH":
            agent = GNCN_PDH(args)

        eta_v  = tf.Variable(eta)
        opt = tf.keras.optimizers.Adam(eta_v)

        Lx_series   = []
        ToD_series  = []
        vLx_series  = []
        vToD_series = []

        # Evaluate initial performance
        ToD_0, Lx_0   = eval_model(agent, train_set, calc_ToD, verbose=True)
        vToD_0, vLx_0 = eval_model(agent, dev_set,   calc_ToD, verbose=True)
        print("{} | ToD = {}  Lx = {} ; vToD = {}  vLx = {}".format(
              -1, ToD_0, Lx_0, vToD_0, vLx_0))

        Lx_series.append(Lx_0)
        ToD_series.append(ToD_0)
        vLx_series.append(vLx_0)
        vToD_series.append(vToD_0)

        PATIENCE = 10
        impatience = 0
        vLx_best = vLx_0
        sim_start_time = time.time()

        ############################################################################
        # Training loop
        ############################################################################
        for i in range(num_iter):
            ToD_accum = 0.0
            Lx_accum  = 0.0
            n_s = 0.0

            # One epoch over training set
            for batch in train_set:
                x = batch[0][1]
                n_s += x.shape[0]

                x_hat = agent.settle(x)  # inference
                ToD_t = calc_ToD(agent)
                Lx_batch = tf.reduce_sum(metric.bce(x_hat, x))

                # accumulate
                ToD_accum += ToD_t
                Lx_accum  += Lx_batch

                # parameter update
                delta = agent.calc_updates()
                opt.apply_gradients(zip(delta, agent.ngc_model.theta))
                agent.ngc_model.apply_constraints()
                agent.clear()

            # average over all training samples
            ToD_epoch = ToD_accum / n_s
            Lx_epoch  = Lx_accum / n_s

            # Evaluate on dev/validation set
            vToD_epoch, vLx_epoch = eval_model(agent, dev_set, calc_ToD, verbose=False)

            print("-------------------------------------------------")
            print("{} | ToD = {}  Lx = {} ; vToD = {}  vLx = {}".format(
                  i, ToD_epoch, Lx_epoch, vToD_epoch, vLx_epoch))

            Lx_series.append(Lx_epoch)
            ToD_series.append(ToD_epoch)
            vLx_series.append(vLx_epoch)
            vToD_series.append(vToD_epoch)

            # Save partial training curves
            if i % save_marker == 0:
                np.save("{}Lx{}".format(out_dir, trial),   np.array(Lx_series))
                np.save("{}ToD{}".format(out_dir, trial),  np.array(ToD_series))
                np.save("{}vLx{}".format(out_dir, trial),  np.array(vLx_series))
                np.save("{}vToD{}".format(out_dir, trial), np.array(vToD_series))

            # Early stopping check
            if vLx_epoch < vLx_best:
                print(" -> Saving model checkpoint:  {} < {}".format(vLx_epoch, vLx_best))
                model_fname = "{}model{}.ngc".format(out_dir, trial)
                io_tools.serialize(model_fname, agent)

                vLx_best = vLx_epoch
                impatience = 0
            else:
                impatience += 1
                if impatience >= PATIENCE:
                    print(" > Executed early stopping!!")
                    break

        sim_end_time = time.time()
        sim_time = sim_end_time - sim_start_time
        print("------------------------------------")
        sim_time_hr = sim_time / 3600.0
        print(" Trial.sim_time = {} h  ({} sec)".format(sim_time_hr, sim_time))

        ############################################################################
        # After training completes or early-stops, evaluate classification error
        ############################################################################
        print("\n >> Training ended. Evaluating final classification error on test set...")
        test_err = evaluate_classification(agent, train_set, test_set)
        print(f"Final classification error on test data = {test_err:.2f}%")
        print("============================================================")
