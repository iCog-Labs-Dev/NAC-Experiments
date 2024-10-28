#!/bin/sh
################################################################################
# Simulate the PCN on the MNIST database
################################################################################
DATA_DIR="../../data/ag-news"
rm -r exp/* ## clear out experimental directory

# RUN

python3 train_pcn.py  --dataX="$DATA_DIR/train_x.npy" \
                     --dataY="$DATA_DIR/train_y.npy" \
                     --devX="$DATA_DIR/test_x.npy" \
                     --devY="$DATA_DIR/test_y.npy" \
                     --verbosity=1

