#!/usr/bin/env bash
set -euo pipefail
python experiments/fewshot_homophily.py \
  --shots 1 3 5 \
  --hops 10 20 40 \
  --folds 30 \
  "$@"
