#!/bin/sh
################################################################################
# Simulate the BFA-SNN on the MNIST database
################################################################################
DATA_DIR="../../data/ag-news"

rm -r exp/* ## clear out experimental directory
python3 train_bfasnn.py  --dataX="$DATA_DIR/train_x.npy" \
                        --dataY="$DATA_DIR/train_y.npy" \
                        --devX="$DATA_DIR/val_x.npy" \
                        --devY="$DATA_DIR/val_y.npy" \
                        --verbosity=0
