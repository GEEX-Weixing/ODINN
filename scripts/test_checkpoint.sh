#!/usr/bin/env bash
set -euo pipefail

: "${CHECKPOINT:?Set CHECKPOINT to a .pt file produced by train.py}"
python test.py --checkpoint "$CHECKPOINT"
