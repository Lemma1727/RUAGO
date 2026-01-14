#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# ViT & CIFAR-10
python src/main.py \
  --data_name cifar10 \
  --model_name ViT \
  --lr 1e-5 \
  --gamma_1 0.15 \
  --gamma_2 0.01 \
  --eps 32 \
  --alpha 0.1 \
  --iters 2 \
  --epoch 2 \
  --batch_size 64 \
  --synthesis_batch_size 32 \
  "$@"


