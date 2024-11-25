from jax import numpy as jnp, random
import sys, getopt as gopt, optparse, time
from pcn_model import PCN ## bring in model from museum
## bring in ngc-learn analysis tools
from ngclearn.utils.metric_utils import measure_ACC, measure_CatNLL, measure_MSE, measure_KLD, measure_BCE

"""
################################################################################
Predictive Coding Network (PCN) Exhibit File:

Fits a PCN classifier to the MNIST database.

Usage:
$ python sim_pcn.py --dataX="/path/to/train_patterns.npy" \
                    --dataY="/path/to/train_labels.npy" \
                    --devX="/path/to/dev_patterns.npy" \
                    --devY="/path/to/dev_labels.npy" \
                    --verbosity=0

@author: The Neural Adaptive Computing Laboratory
################################################################################
"""

# read in general program arguments
options, remainder = gopt.getopt(sys.argv[1:], '',
                                 ["dataX=", "dataY=", "devX=", "devY=","testX","testY", "verbosity="]
                                 )
# external dataset arguments
dataX = "../../data/mnist/trainX.npy"
dataY = "../../data/mnist/trainY.npy"
devX = "../../data/mnist/validX.npy"
devY = "../../data/mnist/validY.npy"
testX = "../../../data/mnist/testX.npy"
testY = "../../../data/mnist/testY.npy"
verbosity = 0 ## verbosity level (0 - fairly minimal, 1 - prints multiple lines on I/O)
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
        devX = arg.strip()
    elif opt in ("--testY"):
        devX = arg.strip()
    elif opt in ("--verbosity"):
        verbosity = int(arg.strip())
print("Train-set: X: {} | Y: {}".format(dataX, dataY))
print("  Dev-set: X: {} | Y: {}".format(devX, devY))
print("  Test-set: X: {} | Y: {}".format(testX, testY))

_X = jnp.load(dataX)
_Y = jnp.load(dataY)
Xdev = jnp.load(devX)
Ydev = jnp.load(devY)
testX = jnp.load(testX)
testY = jnp.load(testY)
x_dim = _X.shape[1]
patch_shape = (int(jnp.sqrt(x_dim)), int(jnp.sqrt(x_dim)))
y_dim = _Y.shape[1]

n_iter = 2
mb_size = 200
n_batches = int(_X.shape[0]/mb_size)
save_point = 5 ## save model params every modulo "save_point"

## set up JAX seeding
dkey = random.PRNGKey(1234)
dkey, *subkeys = random.split(dkey, 10)

## build model
print("--- Building Model ---")
model = PCN(subkeys[1], x_dim, y_dim, hid1_dim=512, hid2_dim=512, T=20,
            dt=1., tau_m=20., act_fx="sigmoid", eta=0.001, exp_dir="exp",
            model_name="pcn")
model.save_to_disk() # save final state of synapses to disk
print("--- Starting Simulation ---")

# Define evaluation function with accuracy, NLL, and MSE logging
def eval_model(model, Xdev, Ydev, mb_size):
    n_batches = int(Xdev.shape[0] / mb_size)
    nll, acc, mse, kld, bce = 0, 0, 0, 0, 0
    n_samp_seen = 0

    for j in range(n_batches):
        idx = j * mb_size
        Xb = Xdev[idx: idx + mb_size, :]
        Yb = Ydev[idx: idx + mb_size, :]

        # Run model inference
        yMu_0, yMu, _ = model.process(obs=Xb, lab=Yb, adapt_synapses=False)

        # Record metrics
        nll += measure_CatNLL(yMu_0, Yb) * Xb.shape[0]
        acc += measure_ACC(yMu_0, Yb) * Yb.shape[0]
        mse += measure_MSE(yMu_0, Yb, preserve_batch=False) * Xb.shape[0]
        kld += measure_KLD(yMu_0, Yb, preserve_batch=False) * Xb.shape[0]
        bce += measure_BCE(yMu_0, Yb, offset=1e-7, preserve_batch=False) * Xb.shape[0]

        n_samp_seen += Yb.shape[0]

    nll /= Xdev.shape[0]
    acc /= Xdev.shape[0]
    mse /= Xdev.shape[0]
    kld /= Xdev.shape[0]
    bce /= Xdev.shape[0]
    return nll, acc, mse, kld, bce

# Logging metrics
trAcc_set, trNll_set, acc_set, efe_set, mse_set, kld_set, bce_set = [], [], [], [], [], [], []

print("--- Training the Model ---")
sim_start_time = time.time()

# Training loop
for i in range(n_iter):
    dkey, *subkeys = random.split(dkey, 2)
    ptrs = random.permutation(subkeys[0], _X.shape[0])
    X, Y = _X[ptrs, :], _Y[ptrs, :]

    train_EFE, n_samp_seen = 0, 0

    for j in range(int(_X.shape[0] / mb_size)):
        idx = j * mb_size
        Xb, Yb = X[idx: idx + mb_size, :], Y[idx: idx + mb_size, :]
        
        # Perform training step
        yMu_0, yMu, _EFE = model.process(obs=Xb, lab=Yb, adapt_synapses=True)
        train_EFE += _EFE * mb_size
        n_samp_seen += Yb.shape[0]

    # Periodic evaluation on dev set
    tr_nll, tr_acc, tr_mse, tr_kld, tr_bce = eval_model(model, _X, _Y, mb_size=1000)
    nll, acc, mse, kld, bce = eval_model(model, Xdev, Ydev, mb_size=1000)

    # Save and log metrics
    trAcc_set.append(tr_acc)
    trNll_set.append(tr_nll)
    acc_set.append(acc)
    mse_set.append(mse)
    kld_set.append(kld)
    bce_set.append(bce)
    efe_set.append(train_EFE / n_samp_seen)

    MSE = jnp.array(mse_set)
    KLD = jnp.array(kld_set)
    BCE = jnp.array(bce_set)
    print(f"{i}: Dev: Acc = {acc}, NLL = {nll}, KLD = {jnp.mean(KLD)}, MSE = {jnp.mean(MSE)}, BCE = {jnp.mean(BCE)} | "
          f"Tr: Acc = {tr_acc}, Tr: MSE = {jnp.mean(tr_mse)}, Tr: KLD = {jnp.mean(tr_kld)}, Tr: BCE = {jnp.mean(tr_bce)}, EFE = {train_EFE / n_samp_seen}")

    if (i + 1) % 5 == 0 or i == (n_iter - 1):
        model.save_to_disk(params_only=True)

_, tr_acc, tr_mse, tr_kld, tr_bce = eval_model(model, _X, _Y, mb_size=1000)
print("--- Testing the Model ---")
inference_start_time = time.time()
nll, acc, mse, kld, bce = eval_model(model, testX, testY, mb_size=1000)
inference_time = time.time() - inference_start_time
print("------------------------------------")
print(f"Inference Time = {inference_time:.4f} seconds")

print(f"-1: Dev: Acc = {acc}, NLL = {nll}, MSE = {jnp.mean(mse)}, KLD = {jnp.mean(kld)}, BCE = {jnp.mean(bce)} | "
      f"Tr: Acc = {tr_acc}, MSE = {jnp.mean(tr_mse)}, Tr: KLD = {jnp.mean(tr_kld)}, Tr: BCE = {jnp.mean(tr_bce)}")

testAcc_set = []
testNll_set = []
testMse_set = []
testBce_set = []
testKld_set = []

# Save the test values
testAcc_set.append(acc)
testNll_set.append(nll)
testMse_set.append(mse)
testBce_set.append(bce)
testKld_set.append(kld)

jnp.save("exp/testAcc.npy", jnp.asarray(testAcc_set))
jnp.save("exp/testNll.npy", jnp.asarray(testNll_set))
jnp.save("exp/testMse.npy", jnp.asarray(testMse_set))
jnp.save("exp/testBce.npy", jnp.asarray(testBce_set))
jnp.save("exp/testKld.npy", jnp.asarray(testKld_set))

# Stop time profiling
sim_time = time.time() - sim_start_time
print("------------------------------------")
print(f"Training Time = {sim_time:.4f} seconds")

print("--- Best Train Values ---")
print(f"Best Train Accuracy = {jnp.amax(jnp.asarray(trAcc_set))}")
print(f"Best Train NLL = {jnp.amin(jnp.asarray(trNll_set))}")
print(f"Best Train MSE = {jnp.amin(jnp.asarray(mse_set))}")

print("--- Best Test Values ---")
print(f"Best Test Accuracy = {jnp.amax(jnp.asarray(testAcc_set))}")
print(f"Best Test NLL = {jnp.amin(jnp.asarray(testNll_set))}")
print(f"Best Test MSE = {jnp.amin(jnp.asarray(testMse_set))}")
print(f"Best Test BCE = {jnp.amin(jnp.asarray(testBce_set))}")
print(f"Best Test KLD = {jnp.amin(jnp.asarray(testKld_set))}")

jnp.save("exp/trAcc.npy", jnp.asarray(trAcc_set))
jnp.save("exp/acc.npy", jnp.asarray(acc_set))
jnp.save("exp/efe.npy", jnp.asarray(efe_set))