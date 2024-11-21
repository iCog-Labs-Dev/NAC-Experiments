from jax import numpy as jnp, random, nn, jit
import sys, getopt as gopt, optparse, time
from bfasnn_model import BFA_SNN as Model  
from ngclearn.utils.metric_utils import measure_ACC, measure_CatNLL, measure_MSE, measure_BCE, measure_KLD

"""
################################################################################
BFA-Trained Spiking Neural Network (BFA-SNN) Exhibit File:

Fits a BFA-SNN (an SNN trained with broadcast feedback alignment) classifier
to the MNIST database.

Usage:
$ python train_bfasnn.py --dataX="/path/to/train_patterns.npy" \
                         --dataY="/path/to/train_labels.npy" \
                         --devX="/path/to/dev_patterns.npy" \
                         --devY="/path/to/dev_labels.npy" \
                         --verbosity=0

@author: The Neural Adaptive Computing Laboratory
################################################################################
"""

# read in general program arguments
options, remainder = gopt.getopt(sys.argv[1:], '',
                                 ["dataX=", "dataY=", "devX=", "devY=", "verbosity="]
                                 )
# external dataset arguments
dataX = "../../../data/mnist/trainX.npy"
dataY = "../../../data/mnist/trainY.npy"
devX = "../../../data/mnist/validX.npy"
devY = "../../../data/mnist/validY.npy"
testX = "../../../data/mnist/testX.npy"
testY = "../../../data/mnist/testY.npy"
verbosity = 0  # verbosity level (0 - fairly minimal, 1 - prints multiple lines on I/O)
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

lab_estimator = "current"  # "voltage" or "spike"
n_iter = 1  # number of discrete time steps to simulate
mb_size = 128
n_batches = int(_X.shape[0] / mb_size)
save_point = 5  # save model params every epoch/iteration modulo "save_point"

# set up JAX seeding
dkey = random.PRNGKey(1234)
dkey, *subkeys = random.split(dkey, 10)

# build/configure BFA-SNN model
hid_dim = 1000
T = 100  # number of discrete time steps to simulate
dt = 0.25  # integration time constant (set in accordance w/ Samadi et al., 2017)
tau_mem = 20.0  # membrane potential time constant (set as in Samadi et al., 2017)
print("--- Building Model ---")
model = Model(subkeys[1], in_dim=x_dim, out_dim=y_dim, hid_dim=hid_dim, T=T, dt=dt, tau_m=tau_mem)
model.save_to_disk()  # save initial state of synapses to disk
print("--- Starting Simulation ---")

# Evaluation model
def eval_model(model, Xdev, Ydev, mb_size, verbosity=1):  # evals model's test-time inference performance
    n_batches = int(Xdev.shape[0] / mb_size)

    n_samp_seen = 0
    nll = 0.0  # negative Categorical log likelihood
    acc = 0.0  # accuracy
    mse = 0.0  # mean squared error
    bce = 0.0
    kld = 0.0
    for j in range(n_batches):
        # extract data block/batch
        idx = j * mb_size
        Xb = Xdev[idx: idx + mb_size, :]
        Yb = Ydev[idx: idx + mb_size, :]
        # run model inference
        _S, yMu, yCnt = model.process(obs=Xb, lab=Yb, adapt_synapses=False, label_dist_estimator=lab_estimator)
        # record metric measurements
        _nll = measure_CatNLL(yMu, Yb) * Xb.shape[0]  # un-normalize score
        _acc = measure_ACC(yMu, Yb) * Yb.shape[0]  
        _mse = measure_MSE(yMu, Yb) * Yb.shape[0] 
        _bce = measure_BCE(yMu, Yb) * Yb.shape[0]
        _kld = measure_KLD(yMu, Yb) * Yb.shape[0]
        nll += _nll
        acc += _acc
        mse += _mse
        bce += _bce
        kld += _kld

        n_samp_seen += Yb.shape[0]
        if verbosity > 0:
            print("\r Acc = {}  NLL = {}  MSE = {} BCE = {} KLD = {} ({} samps)".format(acc / n_samp_seen,
                                                                       nll / n_samp_seen,
                                                                       mse / n_samp_seen,
                                                                       bce / n_samp_seen,
                                                                       kld / n_samp_seen,
                                                                       n_samp_seen), end="")
    if verbosity > 0:
        print()
    nll = nll / (Xdev.shape[0])  # calc full dev-set nll
    acc = acc / (Xdev.shape[0]) 
    mse = mse / (Xdev.shape[0]) 
    bce = bce / (Xdev.shape[0])
    kld = kld / (Xdev.shape[0])
    return nll, acc, mse, bce, kld


trAcc_set = []
trNll_set = []
acc_set = []
nll_set = []
mse_set = []
bce_set = []
kld_set = []

sim_start_time = time.time()  # start time profiling
print("--- Training the Model ---")

tr_acc = 0.1
nll, acc, mse, bce, kld = eval_model(model, Xdev, Ydev, mb_size=1000)
bestDevAcc = acc
print("-1: Dev: Acc = {}  NLL = {}  MSE = {} BCE = {} KLD = {} | Tr: Acc = {}".format(acc, nll, mse, bce, kld, tr_acc))
if verbosity >= 2:
    print(model._get_norm_string())
trAcc_set.append(tr_acc)  # random guessing is where models typically start
trNll_set.append(2.4)
acc_set.append(acc)
nll_set.append(nll)
mse_set.append(mse)
bce_set.append(bce)
kld_set.append(kld)
jnp.save("exp/trAcc.npy", jnp.asarray(trAcc_set))
jnp.save("exp/acc.npy", jnp.asarray(acc_set))
jnp.save("exp/trNll.npy", jnp.asarray(trNll_set))
jnp.save("exp/nll.npy", jnp.asarray(nll_set))
jnp.save("exp/mse.npy", jnp.asarray(mse_set))
jnp.save("exp/bce.npy", jnp.asarray(bce_set))
jnp.save("exp/kld.npy", jnp.asarray(kld_set))

# Training Loop
for i in range(n_iter):
    # shuffle data (to ensure i.i.d. assumption holds)
    dkey, *subkeys = random.split(dkey, 2)
    ptrs = random.permutation(subkeys[0], _X.shape[0])
    X = _X[ptrs, :]
    Y = _Y[ptrs, :]

    # begin a single epoch/iteration
    n_samp_seen = 0
    tr_nll = 0.0
    tr_acc = 0.0
    for j in range(n_batches):
        dkey, *subkeys = random.split(dkey, 2)

        # sample mini-batch of patterns
        idx = j * mb_size
        Xb = X[idx: idx + mb_size, :]
        Yb = Y[idx: idx + mb_size, :]
        # perform a step of inference/learning
        _S, yMu, yCnt = model.process(obs=Xb, lab=Yb, adapt_synapses=True, label_dist_estimator=lab_estimator)
        # track "online" training log likelihood and accuracy
        tr_nll += measure_CatNLL(yMu, Yb) * mb_size  # un-normalize score
        tr_acc += measure_ACC(yCnt, Yb) * mb_size 
        n_samp_seen += Yb.shape[0]
        if verbosity >= 1:
            wStats = ""  # model.get_synapse_stats()
            print("\r NLL = {} ACC = {} ({}) over {} samples ".format((tr_nll / n_samp_seen),
                                                                      (tr_acc / n_samp_seen),
                                                                      wStats, n_samp_seen), end="")
    if verbosity >= 1:
        print()

    # Evaluate of model on dev-set
    print("--- Evaluating the Model ---")
    nll, acc, mse, bce, kld  = eval_model(model, Xdev, Ydev, mb_size=1000)
    tr_acc = (tr_acc / n_samp_seen)
    tr_nll = (tr_nll / n_samp_seen)
    if acc >= bestDevAcc:
        model.save_to_disk(params_only=True)  # save final state of synapses to disk
        bestDevAcc = acc
    if (i + 1) % save_point == 0 or i == (n_iter - 1):
        jnp.save("exp/trAcc.npy", jnp.asarray(trAcc_set))
        jnp.save("exp/acc.npy", jnp.asarray(acc_set))
        jnp.save("exp/trNll.npy", jnp.asarray(trNll_set))
        jnp.save("exp/nll.npy", jnp.asarray(nll_set))
        jnp.save("exp/mse.npy", jnp.asarray(mse_set))
        jnp.save("exp/bce.npy", jnp.asarray(bce_set))
        jnp.save("exp/kld.npy", jnp.asarray(kld_set))

    print("{}: Dev: Acc = {}  NLL = {}  MSE = {} BCE = {} KLD = {} | Tr: Acc = {}".format(i, acc, nll, mse, bce, kld, tr_acc))
    if verbosity >= 2:
        print(model._get_norm_string())

    # update exp records
    trAcc_set.append(tr_acc)
    trNll_set.append(tr_nll)
    acc_set.append(acc)
    nll_set.append(nll)
    mse_set.append(mse)
    bce_set.append(bce)
    kld_set.append(kld)

#  Testing the model
print("--- Testing the Model ---")
inference_start_time = time.time()
test_nll, test_acc, test_mse, test_bce, test_kld = eval_model(
    model, testX, testY, mb_size=1000, verbosity=1
)
inference_time = time.time() - inference_start_time
print("------------------------------------")
print(f"Inference Time = {inference_time} seconds")
print("{}: Test: Acc = {}  NLL = {}  MSE = {} BCE = {} KLD = {}".format(i, test_acc, test_nll, test_mse, test_bce, test_kld))

testAcc_set = []
testNll_set = []
testMse_set = []
testBce_set = []
testKld_set = []

# Save the test values
testAcc_set.append(test_acc)
testNll_set.append(test_nll)
testMse_set.append(test_mse)
testBce_set.append(test_bce)
testKld_set.append(test_kld)

jnp.save("exp/testAcc.npy", jnp.asarray(testAcc_set))
jnp.save("exp/testNll.npy", jnp.asarray(testNll_set))
jnp.save("exp/testMse.npy", jnp.asarray(testMse_set))
jnp.save("exp/testBce.npy", jnp.asarray(testBce_set))
jnp.save("exp/testKld.npy", jnp.asarray(testKld_set))

# Stop time profiling
sim_time = time.time() - sim_start_time
print("------------------------------------")
print(f"Training Time = {sim_time} seconds")

print("--- Best Train Values ---")
print(f"Best Train Accuracy = {jnp.amax(jnp.asarray(trAcc_set))}")
print(f"Best Train NLL = {jnp.amin(jnp.asarray(trNll_set))}")

print("--- Best Development Values ---")
print(f"Best Dev Accuracy = {jnp.amax(jnp.asarray(acc_set))}")
print(f"Best Dev NLL = {jnp.amin(jnp.asarray(nll_set))}")
print(f"Best Dev MSE = {jnp.amin(jnp.asarray(mse_set))}")
print(f"Best Dev BCE = {jnp.amin(jnp.asarray(bce_set))}")
kld_set_array = jnp.asarray(kld_set)
if kld_set_array.ndim == 0:
    print(f"Best Dev KLD = {kld_set_array}")
else:
    print(f"Best Dev KLD = {jnp.amin(kld_set_array)}")

print("--- Best Test Values ---")
print(f"Best Test Accuracy = {jnp.amax(jnp.asarray(testAcc_set))}")
print(f"Best Test NLL = {jnp.amin(jnp.asarray(testNll_set))}")
print(f"Best Test MSE = {jnp.amin(jnp.asarray(testMse_set))}")
print(f"Best Test BCE = {jnp.amin(jnp.asarray(testBce_set))}")
print(f"Best Test KLD = {jnp.amin(jnp.asarray(testKld_set))}")