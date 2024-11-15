# This code is sourced from the following GitHub repository:
# Repository: https://github.com/ljrprocc/DataFree

from torch import nn
import torch
import torch.nn.functional as F
from abc import ABC, abstractclassmethod
from typing import Dict
from utils import jsdiv, kldiv
from torchvision import transforms

class DeepInversionHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)
        self.r_feature = r_feature

    def remove(self):
        self.hook.remove()

def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

class BaseSynthesis(ABC):
    def __init__(self, teacher, student):
        super(BaseSynthesis, self).__init__()
        self.teacher = teacher
        self.student = student
    
    @abstractclassmethod
    def synthesize(self) -> Dict[str, torch.Tensor]:
        """ take several steps to synthesize new images and return an image dict for visualization. 
            Returned images should be normalized to [0, 1].
        """
        pass
    
    @abstractclassmethod
    def sample(self, n):
        """ fetch a batch of training data. 
        """
        pass

class ProbSynthesizer(BaseSynthesis):
    def __init__(self, teacher, student, G, num_classes, img_size, nz, g_steps=None, 
                lr_g=0.1, synthesis_batch_size=128, sample_batch_size=128, normalizer=None, device='cpu',
                adv=0.10, oh=0, adv_type='js', bn=0, T=5):
        super(ProbSynthesizer, self).__init__(teacher, student)
        self.img_size = img_size
        self.g_steps = g_steps
        self.lr_g = lr_g
        self.normalizer = normalizer
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size
        self.device = device
        self.num_classes = num_classes
        self.G = G
        self.adv_type = adv_type
        self.adv = adv
        self.nz = nz
        self.bn = bn
        self.oh = oh
        self.T = T

        self.optimizer = torch.optim.Adam(self.G.parameters(), self.lr_g, betas=[0.9, 0.99])

        self.hooks = []
        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append(DeepInversionHook(m))

    def synthesize(self):
        self.student.eval()
        self.teacher.eval()
        
        for i in range(self.g_steps):
            z = torch.randn(self.synthesis_batch_size, self.nz).to(self.device)
            self.G.train()
            self.optimizer.zero_grad()
            
            # Rec and variance
            mu_theta = self.G(z, c=None)
            if mu_theta.shape[2] != self.img_size:
                mu_theta = transforms.Resize(self.img_size)(mu_theta)
            mu_theta = mu_theta.clamp(-1, 1)
            mu_theta = (mu_theta + 1) / 2
            samples = self.normalizer(mu_theta)
            
            t_out = self.teacher(samples)
            loss_oh = F.cross_entropy(t_out, t_out.max(1)[1])

            if self.bn > 0:
                loss_bn = sum([h.r_feature for h in self.hooks])
            else:
                loss_bn = torch.zeros(1).to(self.device)
            
            # Negative Divergence.
            if self.adv > 0:
                s_out = self.student(samples)
                if self.adv_type == 'js':
                    l_js = jsdiv(s_out, t_out, T=3)
                    loss_adv = 1.0-torch.clamp(l_js, 0.0, 1.0)
                if self.adv_type == 'kl':
                    mask = (s_out.max(1)[1]==t_out.max(1)[1]).float()
                    loss_adv = -(kldiv(s_out, t_out, reduction='none', T=3).sum(1) * mask).mean()
            else:
                loss_adv = torch.zeros(1).to(self.device)
                            
            loss = self.adv * loss_adv + self.oh * loss_oh + self.bn * loss_bn
            loss.backward()
            self.optimizer.step()


    @torch.no_grad()
    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.sample_batch_size
        self.G.eval()
        z = torch.randn(size=(batch_size, self.nz), device=self.device)
        inputs = self.G(z, c=None)
        if inputs.shape[2] != self.img_size:
            inputs = transforms.Resize(self.img_size)(inputs)
        inputs = inputs.clamp(-1, 1)
        inputs = (inputs + 1) / 2
        inputs = self.normalizer(inputs)
        return inputs