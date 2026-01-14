#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# VGG16 & CIFAR-100
python src/main.py \
  --data_name cifar100 \
  --model_name VGG16 \
  --lr 5e-5 \
  --gamma_1 0.1 \
  --gamma_2 0.01 \
  --eps 32 \
  --alpha 2 \
  --iters 3 \
  --epoch 30 \
  "$@"


