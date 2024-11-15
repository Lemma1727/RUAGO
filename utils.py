import sys
import numpy as np
import random
import torch
import torch.nn.functional as F
from time import strftime, localtime

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.file = None

    def open(self, fp, mode=None):
        if mode is None: mode = 'w'
        self.file = open(fp, mode)

    def write(self, msg, is_terminal=1, is_file=1):
        if msg[-1] != '\n' : msg = msg + '\n'
        if '\r' in msg: is_file = 0
        if is_terminal == 1:
            self.terminal.write(msg)
            self.terminal.flush()
        if is_file == 1:
            self.file.write(msg)
            self.file.flush()
    def flush(self):
        pass

def print_args(args, logger=None):
    if logger is not None:
        logger.write("#### configurations ####\n")
    for k, v in vars(args).items():
        if logger is not None:
            logger.write('{}: {}\n'.format(k, v))
        else:
            print('{}: {}'.format(k, v))
    if logger is not None:
        logger.write("########################\n")

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    correct, total= 0, 0
    for data, target in dataloader:
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss=criterion(output, target)
        loss.backward()
        optimizer.step()
        correct += torch.eq(output.argmax(dim=1), target).sum().item()
        total += data.size(0)
    return correct/total*100

@torch.no_grad()
def test(model, dataloader, criterion, device):
    model.eval()
    correct, total= 0, 0
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        correct += torch.eq(output.argmax(dim=1), target).sum().item()
        total += data.size(0)
    return correct/total*100

def overall_test(model, retainloader, forgetloader, testloader, criterion, device):
    model.to(device)
    retain_acc = test(model, retainloader, criterion, device)
    forget_acc = test(model, forgetloader, criterion, device)
    test_acc= test(model, testloader, criterion, device)
    return retain_acc, forget_acc, test_acc

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_time():
    return str(strftime("[%Y-%m-%d %H:%M:%S]", localtime()))

def kldiv(logits, targets, T=1.0, reduction='batchmean'):
    q = F.log_softmax(logits/T, dim=1)
    p = F.softmax(targets/T, dim=1 )
    return F.kl_div(q+1e-8, p+1e-8, reduction=reduction ) * (T*T)

def jsdiv( logits, targets, T=1.0, reduction='batchmean' ):
    P = F.softmax(logits / T, dim=1)
    Q = F.softmax(targets / T, dim=1)
    M = 0.5 * (P + Q)
    P = torch.clamp(P, 0.01, 0.99)
    Q = torch.clamp(Q, 0.01, 0.99)
    M = torch.clamp(M, 0.01, 0.99)
    return 0.5 * F.kl_div(torch.log(P), M, reduction=reduction) + 0.5 * F.kl_div(torch.log(Q), M, reduction=reduction)
    
from sklearn.linear_model import LogisticRegression
def collect_prob(data_loader, model):
    model.eval()
    prob = []
    with torch.no_grad():
        for batch in data_loader:
            data, target = batch
            data = data.to(next(model.parameters()).device)
            output = model(data)
            prob.append(F.softmax(output, dim=-1).data)
    return torch.cat(prob)

def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)

# https://arxiv.org/abs/2205.08096
def get_membership_attack_data(retain_loader, forget_loader, test_loader, model):
    retain_prob = collect_prob(retain_loader, model)
    forget_prob = collect_prob(forget_loader, model)
    test_prob = collect_prob(test_loader, model)

    X_r = (
        torch.cat([entropy(retain_prob), entropy(test_prob)])
        .cpu()
        .numpy()
        .reshape(-1, 1)
    )
    Y_r = np.concatenate([np.ones(len(retain_prob)), np.zeros(len(test_prob))])

    X_f = entropy(forget_prob).cpu().numpy().reshape(-1, 1)
    Y_f = np.concatenate([np.ones(len(forget_prob))])
    return X_f, Y_f, X_r, Y_r


# https://arxiv.org/abs/2205.08096
def get_membership_attack_prob(retain_loader, forget_loader, test_loader, model):
    X_f, Y_f, X_r, Y_r = get_membership_attack_data(
        retain_loader, forget_loader, test_loader, model
    )
    # clf = SVC(C=3,gamma='auto',kernel='rbf')
    clf = LogisticRegression(
        class_weight="balanced", solver="lbfgs"
    )
    clf.fit(X_r, Y_r)
    results = clf.predict(X_f)
    return results.mean()

def attention(x):
        """
        Taken from https://github.com/szagoruyko/attention-transfer
        :param x = activations
        """
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


def attention_diff(x, y):
    """
    Taken from https://github.com/szagoruyko/attention-transfer
    :param x = activations
    :param y = activations
    """
    return (attention(x) - attention(y)).pow(2).mean()