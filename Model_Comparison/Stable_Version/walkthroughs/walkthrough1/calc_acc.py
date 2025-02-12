import os
import sys, getopt, optparse
import pickle
sys.path.insert(0, '../')
import numpy as np
import time
from ngclearn.utils.config import Config
import ngclearn.utils.metric_utils as metric
import ngclearn.utils.io_utils as io_tools
from ngclearn.utils.data_utils import DataLoader

import tensorflow as tf
seed = 69
os.environ["CUDA_VISIBLE_DEVICES"]="0"
tf.random.set_seed(seed=seed)
np.random.seed(seed)

options, remainder = getopt.getopt(sys.argv[1:], '', ["config=","gpu_id="])
cfg_fname = None
use_gpu = False
gpu_id = -1
for opt, arg in options:
    if opt in ("--config"):
        cfg_fname = arg.strip()
    elif opt in ("--gpu_id"):
        gpu_id = int(arg.strip())
        use_gpu = True

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
model_fname = args.getArg("model_fname")
batch_size = int(args.getArg("batch_size"))

print(" Loading data...")
X = (tf.cast(np.load("../data/mnist/trainX.npy"), dtype=tf.float32)).numpy()
Y = (tf.cast(np.load("../data/mnist/trainY.npy"), dtype=tf.float32)).numpy()
test_X = (tf.cast(np.load("../data/mnist/testX.npy"), dtype=tf.float32)).numpy()
test_Y = (tf.cast(np.load("../data/mnist/testY.npy"), dtype=tf.float32)).numpy()

print(f"Loaded X shape: {X.shape}, Y shape: {Y.shape}")
print(f"Loaded test X shape: {test_X.shape}, test Y shape: {test_Y.shape}")
train_set = DataLoader(design_matrices=[("z0", X), ("z0_target", Y)], batch_size=batch_size, disable_shuffle=True)
test_set = DataLoader(design_matrices=[("z0", test_X), ("z0_target", test_Y)], batch_size=batch_size, disable_shuffle=True)

classifier = tf.keras.layers.Dense(10, use_bias=True)
batch = tf.random.normal([batch_size, 360])
_ = classifier(batch)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

def train_epoch(agent, classifier, dataset, optimizer, verbose=False):
    """
    Train classifier
    """
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    for batch in dataset:
        x_name, x = batch[0]
        y_name, y = batch[1]
        batch_size = x.shape[0]
        total_samples += batch_size

        with tf.GradientTape() as tape:
            _ = agent.settle(x)
            z3_latents = agent.ngc_model.extract("z3", "phi(z)")

            logits = classifier(z3_latents)
            probs = tf.nn.softmax(logits)
            loss = tf.reduce_mean(metric.cat_nll(probs, y))

        grads = tape.gradient(loss, classifier.trainable_variables)
        optimizer.apply_gradients(zip(grads, classifier.trainable_variables))

        y_ind = tf.cast(tf.argmax(y, 1), dtype=tf.int32)
        y_pred = tf.cast(tf.argmax(probs, 1), dtype=tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_ind), dtype=tf.float32))

        total_loss += loss * batch_size
        total_acc += acc * batch_size
        agent.clear()

        if verbose:
            print("\r Loss {:.4f}  Acc {:.4f} over {} samples...".format(
                total_loss/total_samples, total_acc/total_samples, total_samples), end="")

    if verbose:
        print()

    return total_loss/total_samples, total_acc/total_samples

def eval_model(agent, dataset, verbose=False):
    """
        Evaluates performance of model using latent representations
    """
    Ly = 0.0
    Acc = 0.0
    N = 0.0
    for batch in dataset:
        x_name, x = batch[0]
        y_name, y = batch[1]
        N += x.shape[0]
        _ = agent.settle(x)

        z3_latents = agent.ngc_model.extract("z3", "phi(z)")

        logits = classifier(z3_latents)
        probs = tf.nn.softmax(logits)
        Ly += tf.reduce_sum(metric.cat_nll(probs, y))

        # Compute accuracy
        y_ind = tf.cast(tf.argmax(y, 1), dtype=tf.int32)
        y_pred = tf.cast(tf.argmax(probs, 1), dtype=tf.int32)
        Acc += tf.reduce_sum(tf.cast(tf.equal(y_pred, y_ind), dtype=tf.float32))

        agent.clear()
        if verbose:
            print("\r Acc {}  Ly {} over {} samples...".format((Acc / N), (Ly / N), N), end="")
    if verbose:
        print()

    Ly /= N
    Acc /= N
    return Ly, Acc

with tf.device(gpu_tag):
    agent = io_tools.deserialize(model_fname)
    print(" > Loading model: ", model_fname)

    print("\nStarting training...")
    best_test_acc = 0.0
    patience = 5
    patience_counter = 0

    train_accs = []
    val_accs = []

    for epoch in range(50):
            train_loss, train_acc = train_epoch(agent, classifier, train_set, optimizer, verbose=True)
            test_loss, test_acc = eval_model(agent, test_set)

            train_accs.append(train_acc)
            val_accs.append(test_acc)

            print(f"Epoch {epoch+1}/50:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                patience_counter = 0
                with open('gncn_t1/classifier_weight.pkl', 'wb') as f:
                    pickle.dump(classifier.get_weights(), f)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered. Best test accuracy: {best_test_acc:.4f}")
                    break

    np.save("gncn_t1/train_acc.npy", np.array(train_accs))
    np.save("gncn_t1/test_acc.npy", np.array(val_accs))