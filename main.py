import os
import math
import pickle
from time import time
from copy import deepcopy
import argparse
import torch
import torch.nn as nn

import models
from dataset import *
from utils import *
from adversarial import PGD
from dfkd import ProbSynthesizer
import datetime

import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser(description='Unlearning')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--model_seed', type=int, default=42, help='model seed')

    parser.add_argument('--data_name', type=str, default='cifar10', help='dataset name')
    parser.add_argument('--data_path', type=str, default='./data', help='path to dataset')
    parser.add_argument('--synthesis_batch_size', type=int, default=64, help='sample batch size')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')

    parser.add_argument('--forget_ratio', type=float, default=0.1, help='forget ratio')
    
    parser.add_argument('--model_name', type=str, default='ResNet18', choices=['VGG16', 'ResNet18', 'ViT'], help='model name')

    parser.add_argument('--epoch', type=int, default=50, help='unlearn epoch')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')

    parser.add_argument('--eps', type=float, default=32, help='eps')
    parser.add_argument('--alpha', type=float, default=4, help='alpha')
    parser.add_argument('--iters', type=int, default=3, help='iters')

    parser.add_argument('--g_type', type=str, default='coco', help='choose a generator type')
    parser.add_argument('--g_res', type=int, default=32, help='generator resolution')
    parser.add_argument('--g_steps', type=int, default=1, help='g steps')
    parser.add_argument('--lr_g', type=float, default=1e-5, help='learning rate for generator')
    parser.add_argument('--adv', type=float, default=0.0, help='adv')
    parser.add_argument('--bn', type=int, default=1, help='bn')
    parser.add_argument('--oh', type=float, default=1, help='oh')
    parser.add_argument('--adv_type', type=str, default='kl', help='adv type')
    parser.add_argument('--T', type=int, default=20, help='T')
    parser.add_argument('--lambda_0', type=float, default=2.0, help='lambda 0')

    parser.add_argument('--gamma_1', type=float, default=0.2, help='hyperparameter for the forget loss')
    parser.add_argument('--gamma_2', type=float, default=0.01, help='hyperparameter for the retain loss')

    parser.add_argument('--load_path', type=str, default='./checkpoints', help='path to load')
    parser.add_argument('--save_path', type=str, default='./checkpoints', help='path to save')    

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return args

def main(args, logger):
    data_dict = get_dataset(args.data_name, args.data_path, args.model_name)
    _, testloader = get_dataloader(data_dict, args.batch_size, args.num_workers, args.seed)

    unlearnset = get_unlearn_dataset(args.data_name, args.data_path, args.forget_ratio, args.seed, args.data_path, False, model_name=args.model_name)
    forgetloader, retainloader = get_unlearn_dataloader(unlearnset, args.batch_size, args.batch_size, args.num_workers, args.seed)
    
    original_model = getattr(models, args.model_name)(num_classes=data_dict['num_classes'])
    original_model.load_state_dict(torch.load(os.path.join(args.load_path, 'original', args.data_name, args.model_name, 'original_model.pth')))
    original_model.to(args.device)

    unlearn_model = deepcopy(original_model)

    if args.g_type == 'coco':
        pretrained_GAN = './generators/gan_coco.pkl'
    elif args.g_type == 'tiny':
        pretrained_GAN = './generators/gan_tiny.pkl'
    else:
        raise ValueError('Invalid generator type')
    logger.write(f'Loading pretrained GAN from {pretrained_GAN}')
    with open(pretrained_GAN, 'rb') as f:
        generator = pickle.load(f)['G_ema'].cuda()
    for param in generator.parameters():
        param.requires_grad = True
            
    generator = generator.eval()
    generator = generator.to(args.device)

    synthesizer = ProbSynthesizer(
            teacher=original_model,
            student=unlearn_model,
            G=generator,
            nz=512,
            num_classes=data_dict['num_classes'],
            img_size=data_dict['size'][0],
            g_steps=args.g_steps,
            lr_g=args.lr_g,
            synthesis_batch_size=args.synthesis_batch_size,
            sample_batch_size=args.batch_size,
            normalizer=transforms.Normalize(unlearnset['mean'], unlearnset['std']),
            device=args.device,
            adv=args.adv,
            oh=args.oh,
            adv_type=args.adv_type,
            bn=args.bn,
            T=args.T,
        )

    attack = PGD(model=deepcopy(original_model), eps=args.eps, alpha=args.alpha, iters=args.iters, denorm=True)
    attack.set_normalization(unlearnset['mean'], unlearnset['std'])

    optimizer = torch.optim.Adam(unlearn_model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

    global_iter = 0
    alpha = 0.0001 if args.data_name == 'cifar10' else 0.00002

    unlearn_start = time()
    for it in range(args.epoch):
        synthesizer.synthesize()
        unlearn_model.train()
        original_model.eval()
        losses, ce_losses, kd_losses = AverageMeter(), AverageMeter(), AverageMeter()
        vs = AverageMeter()
        lbs, adv_lbs = torch.tensor([]), torch.tensor([])
        for f_imgs, f_lbs in forgetloader:
            unlearn_model.train()
            original_model.eval()
            _, adv_labels = attack(f_imgs, f_lbs, return_prob=True)
            lbs, adv_lbs = torch.cat([lbs, f_lbs]), torch.cat([adv_lbs, adv_labels.argmax(1)])

            f_imgs, f_lbs = f_imgs.to(args.device), f_lbs.to(args.device)
            f_outputs = unlearn_model(f_imgs)

            ce_loss = nn.CrossEntropyLoss()(f_outputs, adv_labels.to(args.device))

            syn_images = synthesizer.sample()

            lamda = args.lambda_0 + alpha * min(global_iter, 500000000)

            with torch.no_grad():
                t_out = original_model(syn_images.detach())
            s_out = unlearn_model(syn_images)

            kd_loss = kldiv(s_out, t_out.detach(), args.T, reduction='none').sum(1)

            with torch.no_grad():
                v = (1 + math.exp(-lamda)) / (1 + (kd_loss - lamda).exp())
            kd_loss = (v * kd_loss).sum()
            vs.update(v.mean().item())
            
            loss = args.gamma_1 * ce_loss + args.gamma_2 * kd_loss
            ce_losses.update(ce_loss.item(), f_imgs.size(0))
            kd_losses.update(kd_loss.item(), syn_images.size(0))
            losses.update(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_iter += 1
        scheduler.step()
        logger.write(f'{get_time()} | iter={it} | V mean={vs.avg:.4f} | loss={losses.avg:.4f} | ce loss={ce_losses.avg:.4f} | kd loss={kd_losses.avg:.4f} | ASR={1 - adv_lbs.eq(lbs).float().mean().item():.4f}')
    
    end_time = time()
    retain_acc, forget_acc, test_acc= overall_test(unlearn_model, retainloader, forgetloader, testloader, nn.CrossEntropyLoss(), args.device)
    logger.write('='*50)
    logger.write(' '*20 + 'Result')
    logger.write(f'Retain ACC: {retain_acc:.4f}')
    logger.write(f'Forget ACC: {forget_acc:.4f}')
    logger.write(f'Test ACC: {test_acc:.4f}')
    miaresult = get_membership_attack_prob(retainloader, forgetloader, testloader, unlearn_model)
    logger.write(f'MIA result: {miaresult:.4f}')
    logger.write(f'Unlearn time: {end_time-unlearn_start:.4f} sec')
    logger.write('######################################################')
    
    _best_ckpt = os.path.join(args.log_ckp_path, 'unlearing_model.pth')
    torch.save(unlearn_model.state_dict(), _best_ckpt)

if __name__ == '__main__':
    args = get_args()
    logger = Logger()
    tm = datetime.datetime.now()
    args.log_ckp_path = os.path.join(args.save_path, 'unlearn', args.data_name, args.model_name)
    os.makedirs(args.log_ckp_path, exist_ok=True)
    logger.open(os.path.join(args.log_ckp_path, 'unlearning_log.txt'))
    print_args(args, logger=logger)
    seed_everything(args.seed)
    main(args, logger)