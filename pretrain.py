import os
import argparse
import torch
import torch.nn as nn
from time import time
from tqdm import tqdm

import models
from dataset import *
from utils import *

import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser(description='Unlearning')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--mode', type=str, default='original', choices=['original', 'retrain'], help='mode')

    parser.add_argument('--data_name', type=str, default='cifar10', help='dataset name')
    parser.add_argument('--data_path', type=str, default='./data', help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')

    parser.add_argument('--forget_ratio', type=float, default=0.1, help='forget ratio')

    parser.add_argument('--model_name', type=str, default='ResNet18', choices=['VGG16', 'ResNet18', 'ViT'], help='model name')

    parser.add_argument('--epoch', type=int, default=200, help='epoch')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')

    parser.add_argument('--save_path', type=str, default='./checkpoints', help='path to save')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return args

def main(args, logger):
    data_dict = get_dataset(args.data_name, args.data_path, args.model_name)
    trainloader, testloader = get_dataloader(data_dict, args.batch_size, args.num_workers, args.seed)

    unlearnset = get_unlearn_dataset(args.data_name, args.data_path, args.forget_ratio, args.seed, args.data_path, False, model_name=args.model_name)
    forgetloader, retainloader = get_unlearn_dataloader(unlearnset, args.batch_size, args.batch_size, args.num_workers, args.seed, shuffle=False)

    if args.mode == 'retrain':
        unlearnset_U = get_unlearn_dataset(args.data_name, args.data_path, args.forget_ratio, args.seed, args.data_path, True, args.model_name)
        _, trainloader = get_unlearn_dataloader(unlearnset_U, args.batch_size, args.batch_size, args.num_workers, args.seed)

    model = getattr(models, args.model_name)(num_classes=data_dict['num_classes'])
    model = model.to(args.device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

    start_time = time()
    for it in tqdm(range(args.epoch), desc='Epoch', leave=True):
        model.train()
        losses = AverageMeter()
        with tqdm(total=len(trainloader), desc='One epoch', leave=False) as pbar:
            for r_imgs, r_labs in trainloader:
                pbar.update(1)
                r_imgs, r_labs = r_imgs.to(args.device), r_labs.to(args.device)
                output = model(r_imgs)
                loss = nn.CrossEntropyLoss()(output, r_labs)
                optimizer.zero_grad()
                loss.backward()
                losses.update(loss.item(), r_imgs.size(0))
                optimizer.step()
                pbar.set_postfix({'loss': f'{losses.avg:.4f}'})
        scheduler.step()
    end_time = time()
    retain_acc, forget_acc, test_acc= overall_test(model, retainloader, forgetloader, testloader, nn.CrossEntropyLoss(), args.device)
    miaresult = get_membership_attack_prob(retainloader, forgetloader, testloader, model)
    
    logger.write(f'Current Time: {get_time()}')
    logger.write(f'Retain ACC: {retain_acc:.4f}')
    logger.write(f'Forget ACC: {forget_acc:.4f}')
    logger.write(f'Test ACC: {test_acc:.4f}')
    logger.write(f'MIA result: {miaresult:.8f}')
    logger.write(f'Consuming Time: {end_time - start_time}')

    torch.save(model.state_dict(), os.path.join(args.log_ckp_path, f'{args.mode}_model.pth'))

if __name__ == '__main__':
    args = get_args()
    logger = Logger()
    args.log_ckp_path = os.path.join(args.save_path, args.mode, args.data_name, args.model_name)
    os.makedirs(args.log_ckp_path, exist_ok=True)
    logger.open(os.path.join(args.log_ckp_path, f'{args.mode}_log.txt'))
    print_args(args, logger=logger)
    seed_everything(args.seed)
    main(args, logger)