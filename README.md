# Paper ID 18256

### Overview

Our approach, RUAGO, integrates adversarial attacks, a generative model, model inversion, and a sample difficulty strategy to remove targeted information while maintaining model performance. Specifically, adversarial attacks create uncertain probability labels for the forget data, and an OOD-trained generative model replaces the retain data, ensuring effective unlearning without access to the original retained data.

We provide this code to reproduce the results presented in our paper.

### Installation

1. Clone the repository:
    
    Download repository from [here](https://anonymous.4open.science/r/RUAGO-41A2/README.md)

    ```sh
    cd RUAGO
    ```
2. Create a virtual environment:
    ```sh
    conda create -n ruago python=3.9.20
    ```
3. Install requirements:
    ```sh
    conda activate ruago
    pip install -r requirements.txt
    ```

### Training

**Training Original model:**
```sh
python pretrain.py --dataset cifar10 --model_name ResNet18
```

**Training Retrain model:**
```sh
python pretrain.py --dataset cifar10 --model_name ResNet18 --mode retrain
```

### Download Pretrained Models
You can download the pretrained checkpoints of both the generative model and the classification model [here](https://www.dropbox.com/scl/fi/syteb6jjm8a1y672lx4ol/RUAGO_checkpoints.zip?rlkey=z9wg6u3p2otkk9eylxkqldilw&st=fwoym6ng&dl=0).

Place the downloaded checkpoints in the RUGAO/ directory.

```bash
RUAGO
├── checkpoints  # Pretrained model checkpoints (train or copy from download link)
│   ├── original
│   ├── retrain
│   └── unlearn
├── data         # Dataset directory  
│   ├── cifar10
│   ├── cifar100
├── dnnlib       # Deep learning library code
├── generators   # Generative models (copy from download link)
├── models       # Model definition files
├── torch_utils  # PyTorch utility functions
├── adversarial.py # Adversarial attack implementation
├── dataset.py     # Dataset loading and preprocessing
├── dfkd.py      # model inversion code
├── main.py      # Main execution script
├── pretrain.py  # Pretraining script
├── README.md    # Project documentation
├── requirements.txt  # List of dependencies
└── utils  # Miscellaneous utility functions
```

### Run RAUGO

**VGG16 & CIFAR-10**
```sh
python main.py --data_name cifar10 --model_name VGG16 --lr 1e-4 --gamma_1 0.2 --gamma_2 0.01 --eps 32 --alpha 1 --iters 3 --epoch 50
```

**ResNet18 & CIFAR-10**
```sh
python main.py --data_name cifar10 --model_name ResNet18 --lr 5e-5 --gamma_1 0.2 --gamma_2 0.01 --eps 32 --alpha 4 --iters 3 --epoch 50
```

**ViT & CIFAR-10**
```sh
python main.py --data_name cifar10 --model_name ViT --lr 1e-5 --gamma_1 0.15 --gamma_2 0.01 --eps 32 --alpha 0.1 --iters 2 --epoch 2 --batch_size 64 --synthesis_batch_size 32
```

**VGG16 & CIFAR-100**
```sh
python main.py --data_name cifar100 --model_name VGG16 --lr 5e-5 --gamma_1 0.1 --gamma_2 0.01 --eps 32 --alpha 2 --iters 3 --epoch 30
```

**ResNet18 & CIFAR-100**
```sh
python main.py --data_name cifar100 --model_name ResNet18 --lr 1e-4 --gamma_1 0.5 --gamma_2 0.01 --eps 32 --alpha 4 --iters 3 --epoch 50
```

**ViT & CIFAR-100**
```sh
python main.py --data_name cifar100 --model_name ViT --lr 1e-5 --gamma_1 0.1 --gamma_2 0.01 --eps 32 --alpha 0.1 --iters 3 --epoch 2 --batch_size 64 --synthesis_batch_size 32
```

