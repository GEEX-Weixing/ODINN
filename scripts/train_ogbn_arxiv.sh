#!/usr/bin/env bash
set -euo pipefail

# The generated split matches the few-shot protocol used by the original code.
# Change --split official to use the OGB public train/valid/test split.
for model in DeGroot Friedkin_Johnsen; do
  python train.py \
    --dataset ogbn-arxiv \
    --model_name "$model" \
    --split generated \
    --label_rate 1 \
    --num_layers 20 \
    --epochs 400 \
    --folds 10
 done
