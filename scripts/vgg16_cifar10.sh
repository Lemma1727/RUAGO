#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# VGG16 & CIFAR-10
python src/main.py \
  --data_name cifar10 \
  --model_name VGG16 \
  --lr 1e-4 \
  --gamma_1 0.2 \
  --gamma_2 0.01 \
  --eps 32 \
  --alpha 1 \
  --iters 3 \
  --epoch 50 \
  "$@"


