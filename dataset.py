import os
import random
import numpy as np

import torch
from torch.utils.data import Subset, DataLoader, random_split
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms

DATASET_CONFIGS = {
    'cifar10': {
        'num_classes': 10,
        'dataset': CIFAR10,
        'size': (32, 32),
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2470, 0.2434, 0.2615)
    },
    'cifar100': {
        'num_classes': 100,
        'dataset': CIFAR100,
        'size': (32, 32),
        'mean': (0.5071, 0.4866, 0.4409),
        'std': (0.2673, 0.2564, 0.2762)
    },
}

def get_transform(model_name, config, is_train=True):
    if model_name == 'ViT' or config['size'][0] == 224:
        if model_name == 'ViT':
            config['mean'] = config['std'] = [0.5] * 3
            config['size'] = (224, 224)
        if is_train:
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(config['mean'], config['std'])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(config['mean'], config['std'])
            ])
    else:
        if is_train:
            return transforms.Compose([
                transforms.RandomCrop(config['size'][0], padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(config['mean'], config['std'])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(config['size'][0]),
                transforms.ToTensor(),
                transforms.Normalize(config['mean'], config['std'])
            ])

def get_dataset(data_name, data_path, model_name):
    if data_name not in DATASET_CONFIGS:
        raise NotImplementedError(f"Dataset {data_name} not implemented")
    
    data_path = os.path.join(data_path, data_name)

    config = DATASET_CONFIGS[data_name].copy()
    train_transform = get_transform(model_name, config, is_train=True)
    test_transform = get_transform(model_name, config, is_train=False)

    trainset = config['dataset'](root=data_path, train=True, download=True, transform=train_transform)
    testset = config['dataset'](root=data_path, train=False, download=True, transform=test_transform)

    return {
        'train': trainset,
        'test': testset,
        'num_classes': config['num_classes'],
        'channel': 3,
        'size': config['size'],
        'mean': config['mean'],
        'std': config['std']
    }

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dataloader(data_dict, batch_size, num_workers, seed, shuffle=True):
    g = torch.Generator()
    g.manual_seed(seed)
    common_args = {
        'num_workers': num_workers,
        'pin_memory': True,
        'worker_init_fn': seed_worker,
        'generator': g
    }
    train_loader = DataLoader(data_dict['train'], batch_size=batch_size, shuffle=shuffle, **common_args)
    test_loader = DataLoader(data_dict['test'], batch_size=batch_size, shuffle=False, **common_args)
    return train_loader, test_loader

def get_unlearn_dataset(data_name, data_path, forget_ratio, seed=42, save_path='./', is_train=False, model_name='ResNet18'):
    if data_name not in DATASET_CONFIGS:
        raise NotImplementedError(f"Dataset {data_name} not implemented")
    data_path = os.path.join(data_path, data_name)
    config = DATASET_CONFIGS[data_name].copy()
    
    transform = get_transform(model_name, config, is_train=is_train)

    save_path = os.path.join(save_path, 'dataset_indices')
    indices_file = os.path.join(save_path, f'{data_name}_indices_{seed}.pt')

    dataset_args = {'root': config.get('root', data_path), 'transform': transform}
    if data_name.startswith('cifar'):
        dataset_args.update({'train': True, 'download': True})

    full_dataset = config['dataset'](**dataset_args)

    if os.path.exists(indices_file):
        print(f'Load indices from {indices_file}')
        indices = torch.load(indices_file)
        forgetset = Subset(full_dataset, indices['forget'])
        retainset = Subset(full_dataset, indices['retain'])
    else:
        forget_size = int(len(full_dataset) * forget_ratio)
        retain_size = len(full_dataset) - forget_size
        retainset, forgetset = random_split(full_dataset, [retain_size, forget_size], 
                                            generator=torch.Generator().manual_seed(seed))
        indices = {
            'forget': forgetset.indices,
            'retain': retainset.indices
        }
        os.makedirs(save_path, exist_ok=True)
        torch.save(indices, indices_file)

    return {
        'forget': forgetset,
        'retain': retainset,
        'num_classes': config['num_classes'],
        'channel': 3,
        'size': config['size'],
        'mean': config['mean'],
        'std': config['std']
    }

def get_unlearn_dataloader(data_dict, forget_batch, retain_batch, num_workers, seed, shuffle=True):
    g = torch.Generator()
    g.manual_seed(seed)
    common_args = {
        'num_workers': num_workers,
        'pin_memory': True,
        'worker_init_fn': seed_worker,
        'generator': g,
        'shuffle': shuffle
    }
    forget_loader = DataLoader(data_dict['forget'], batch_size=forget_batch, **common_args)
    retain_loader = DataLoader(data_dict['retain'], batch_size=retain_batch, **common_args)
    return forget_loader, retain_loader