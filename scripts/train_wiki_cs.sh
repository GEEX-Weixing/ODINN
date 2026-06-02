#!/usr/bin/env bash
set -euo pipefail

for model in DeGroot Friedkin_Johnsen; do
  python train.py \
    --dataset wiki_cs \
    --model_name "$model" \
    --split generated \
    --label_rate 1 \
    --num_layers 10 \
    --epochs 400 \
    --folds 10
 done
