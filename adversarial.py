import torch
from torch import nn

class PGD():
    def __init__(self, model=None, eps=8, alpha=2, iters=10, denorm=True):
        self.model = model
        self.eps = eps/255
        self.alpha = alpha/255
        self.iters = iters
        self.denorm = denorm
        self.device = next(model.parameters()).device

    def set_normalization(self, mean, std):
        n_channels = len(mean)
        self.mean = torch.tensor(mean).reshape(1, n_channels, 1, 1).to(self.device)
        self.std = torch.tensor(std).reshape(1, n_channels, 1, 1).to(self.device)

    def normalize(self, img):
        img = img.to(self.device)
        return (img - self.mean) / self.std

    def denormalize(self, img):
        img = img.to(self.device)
        return img * self.std + self.mean

    def forward(self, images, labels, target_labels=None):
        self.model.eval()
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        if target_labels is not None:
            target_labels = target_labels.clone().detach().to(self.device)
        criterion = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        for _ in range(self.iters):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            if target_labels is not None:
                loss = -criterion(outputs, target_labels)
            else:
                loss = criterion(outputs, labels)
            grad_sign = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
            adv_images = adv_images.detach() + self.alpha * grad_sign.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images.detach().cpu()
    
    def __call__(self, images, labels, target_labels=None, return_disrupted=False, return_prob=False):
        self.model.eval()
        if self.denorm:
            images = self.denormalize(images)
            adv_inputs = self.forward(images, labels, target_labels)
            adv_inputs = self.normalize(adv_inputs)
        else:
            adv_inputs = self.forward(images, labels, target_labels)
        
        if return_disrupted and return_prob:
            raise ValueError("return_disrupted and return_prob cannot be True at the same time")
        if return_prob:
            with torch.no_grad():
                adv_outputs = self.model(adv_inputs.to(self.device))
                adv_probs = torch.softmax(adv_outputs, dim=1)
            self.model.train()
            return adv_inputs.detach().cpu(), adv_probs.detach().cpu()
        if return_disrupted:
            with torch.no_grad():
                adv_outputs = self.model(adv_inputs.to(self.device))
                adv_labels = adv_outputs.argmax(dim=1)
            self.model.train()
            return adv_inputs.detach().cpu(), adv_labels.detach().cpu()
        self.model.train()
        return adv_inputs