#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# ResNet18 & CIFAR-10
python src/main.py \
  --data_name cifar10 \
  --model_name ResNet18 \
  --lr 5e-5 \
  --gamma_1 0.2 \
  --gamma_2 0.01 \
  --eps 32 \
  --alpha 4 \
  --iters 3 \
  --epoch 50 \
  "$@"


