#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# ResNet18 & CIFAR-100
python src/main.py \
  --data_name cifar100 \
  --model_name ResNet18 \
  --lr 1e-4 \
  --gamma_1 0.5 \
  --gamma_2 0.01 \
  --eps 32 \
  --alpha 4 \
  --iters 3 \
  --epoch 50 \
  "$@"


