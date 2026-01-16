#!/bin/bash

# Abort script if any command fails
set -e

echo "Starting Experiment Sequence..."

echo "Running Experiment 1: Baseline QF (10–70)"

python SwinIR/train.py \
    --wandb_name swinir_baseline_q40 \
    --qf_train 10 20 30 40 50 60 70 \
    --qf_test 10 20 30 40 50 60 70 \
    --cuda_id 1 \
    --batch_size 6 \
    --log_interval 100

echo "All experiments finished!"
