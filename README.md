## RUAGO: Effective and Practical Retain-Free Unlearning via Adversarial Attack and OOD Generator

Official implementation of **"RUAGO: Effective and Practical Retain-Free Unlearning via Adversarial Attack and OOD Generator" (NeurIPS 2025)**.

### Overview

RUAGO integrates adversarial attacks, an OOD generator, model inversion, and a sample difficulty strategy to remove targeted information while maintaining model utility.

- **Forget**: adversarial attacks create uncertain probability labels for forget data.
- **Retain (retain-free)**: an OOD-trained generator replaces retain data, enabling unlearning without access to original retain samples.

### Installation

```bash
git clone https://github.com/Lemma1727/RUAGO.git
cd RUAGO

conda create -n ruago python=3.9.20 -y
conda activate ruago
pip install -r requirements.txt
```

### System requirements (recommended)

This repo includes StyleGAN2-ADA components (`src/dnnlib/`, `src/torch_utils/`) that build **custom CUDA/C++ ops at runtime** via `torch.utils.cpp_extension`.

- **Python**: 3.9.x
- **PyTorch**: `torch==1.12.1+cu113` (see `requirements.txt`)
- **NVIDIA GPU + driver**: driver compatible with CUDA 11.3 runtime (cu113)
- **CUDA toolkit (nvcc)**: recommended to match PyTorch CUDA version (**CUDA 11.3**) for building custom ops
- **C++ compiler**: GCC/Clang toolchain (Linux) for JIT compilation

If CUDA/C++ op compilation fails, the code will typically **fall back to a slower reference implementation** (you may see warnings). For best performance/reproducibility, we recommend having a working `nvcc` toolchain.

Quick checks:

```bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

### Quickstart

1) Download released assets (required for generator, optional for classifier checkpoints)

```bash
bash scripts/download_assets.sh
```

2) Run an experiment preset (model + dataset)

```bash
bash scripts/resnet18_cifar10.sh
```

### Data

This repo uses `torchvision.datasets` with `download=True`, so CIFAR-10/100 will be downloaded automatically under `--data_path` (default: `./data`).

**Datasets are not included in the GitHub repository.**

### Released assets (checkpoints + generators)

The following folders are runtime artifacts and are **ignored by Git** by default:

- `checkpoints/` (classifier checkpoints)
- `generators/` (OOD generator pickle files)

RUAGO requires pretrained generator pickle files under `generators/` (e.g., `gan_coco.pkl`, `gan_tiny.pkl`) to run.
We do **not** release generator training code in this repository; these generators were trained by us using the StyleGAN2-ADA codebase and are provided via the released assets bundle.

- Pretrained bundle: `RUAGO_checkpoints.zip` (Dropbox) — [`download link`](https://www.dropbox.com/scl/fi/syteb6jjm8a1y672lx4ol/RUAGO_checkpoints.zip?rlkey=z9wg6u3p2otkk9eylxkqldilw&st=fwoym6ng&dl=0)

You can download and unpack it with:

```bash
bash scripts/download_assets.sh
```

Place the contents under the project root so the directory structure looks like:

```text
RUAGO/
  checkpoints/
    original/<data_name>/<model_name>/original_model.pth
    retrain/<data_name>/<model_name>/retrain_model.pth
    unlearn/<data_name>/<model_name>/(unlearing_model.pth, unlearning_log.txt)
  generators/
    gan_coco.pkl
    gan_tiny.pkl
```

### Train base models

You can either **train** the classifier checkpoints or **download** them from the released assets bundle.

**Train original model (example)**

```bash
bash scripts/pretrain_resnet18_cifar10_original.sh
```

**Train retrain baseline (example)**

```bash
bash scripts/pretrain_resnet18_cifar10_retrain.sh
```

### Run RUAGO (unlearning)

Below scripts are “preset runs” for each model/dataset combination:

**VGG16 & CIFAR-10**

```bash
bash scripts/vgg16_cifar10.sh
```

**ResNet18 & CIFAR-10**

```bash
bash scripts/resnet18_cifar10.sh
```

**ViT & CIFAR-10**

```bash
bash scripts/vit_cifar10.sh
```

**VGG16 & CIFAR-100**

```bash
bash scripts/vgg16_cifar100.sh
```

**ResNet18 & CIFAR-100**

```bash
bash scripts/resnet18_cifar100.sh
```

**ViT & CIFAR-100**

```bash
bash scripts/vit_cifar100.sh
```

### Repository layout (current)

```text
RUAGO/
  src/
    adversarial.py
    dataset.py
    dfkd.py
    main.py
    pretrain.py
    models/
    utils.py
    dnnlib/
    torch_utils/
  scripts/
    download_assets.sh
    vgg16_cifar10.sh
    resnet18_cifar10.sh
    vit_cifar10.sh
    vgg16_cifar100.sh
    resnet18_cifar100.sh
    vit_cifar100.sh
    pretrain_resnet18_cifar10_original.sh
    pretrain_resnet18_cifar10_retrain.sh
  requirements.txt
```

### Notes for public release

- **Large assets**: `checkpoints/`, `data/`, `generators/` can be very large and should not be committed.
- **Third-party code**: this repo includes `src/dnnlib/` and `src/torch_utils/` with NVIDIA copyright headers.
  Before public release, please ensure the original license/notice requirements are satisfied (e.g., include the correct third-party license text and attribution).

### Third-party licenses

See `THIRD_PARTY_NOTICES.md` and `LICENSES/`. In particular, `src/dnnlib/` and `src/torch_utils/` are distributed under NVIDIA’s StyleGAN2-ADA license (non-commercial restriction).

### License

- **RUAGO code**: MIT License (see `LICENSE`)
- **Third-party components**: see `THIRD_PARTY_NOTICES.md` and `LICENSES/` (some components include a **non-commercial restriction**)

### Citation

If you use this code, please cite our NeurIPS 2025 paper.

```bibtex
@inproceedings{leeruago,
  title={RUAGO: Effective and Practical Retain-Free Unlearning via Adversarial Attack and OOD Generator},
  author={Lee, SangYong and Chung, Sangjun and Woo, Simon S},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}
}
```

