# -*- coding: utf-8 -*-
import numpy as np
import sys
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from models.wrn import WideResNet

import utils.svhn_loader as svhn
from utils.display_results import get_measures, print_measures
from utils.tinyimages_80mn_loader import TinyImages

parser = argparse.ArgumentParser(description='DAL training procedure on the CIFAR benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100'],
                    help='Choose between CIFAR-10, CIFAR-100.')

# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--oe_batch_size', type=int, default=256, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# WRN Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
# DAL hyper parameters
parser.add_argument('--gamma', default=1, type=float)
parser.add_argument('--beta',  default=0.5, type=float)
parser.add_argument('--rho',   default=0.01, type=float)
parser.add_argument('--strength', default=0.01, type=float)
parser.add_argument('--warmup', type=int, default=0)
parser.add_argument('--iter', default=10, type=int)
# Others
parser.add_argument('--out_as_pos', action='store_true', help='OE define OOD data as positive.')
parser.add_argument('--seed', type=int, default=1, help='seed for np(tinyimages80M sampling); 1|2|8|100|107')


args = parser.parse_args()
torch.manual_seed(1)
np.random.seed(args.seed)
torch.cuda.manual_seed(1)

print(args.gamma, args.beta, args.rho)

cudnn.benchmark = True  # fire on all cylinders


# mean and standard deviation of channels of CIFAR-10 images
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                                trn.ToTensor(), trn.Normalize(mean, std)])
test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

if args.dataset == 'cifar10':
    train_data_in = dset.CIFAR10('../data/cifarpy', train=True, transform=train_transform)
    test_data = dset.CIFAR10('../data/cifarpy', train=False, transform=test_transform)
    cifar_data = dset.CIFAR100('../data/cifarpy', train=False, transform=test_transform) 
    num_classes = 10
else:
    train_data_in = dset.CIFAR100('../data/cifarpy', train=True, transform=train_transform)
    test_data = dset.CIFAR100('../data/cifarpy', train=False, transform=test_transform)
    cifar_data = dset.CIFAR10('../data/cifarpy', train=False, transform=test_transform)
    num_classes = 100

ood_data = TinyImages(transform=trn.Compose([trn.ToTensor(), trn.ToPILImage(), trn.RandomCrop(32, padding=4), trn.RandomHorizontalFlip(), trn.ToTensor(), trn.Normalize(mean, std)]), exclude_cifar = True)

train_loader_in = torch.utils.data.DataLoader(train_data_in, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False)
train_loader_out = torch.utils.data.DataLoader(ood_data, batch_size=args.oe_batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False)

texture_data = dset.ImageFolder(root="../data/dtd/images", transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)]))
svhn_data = svhn.SVHN(root='../data/svhn/', split="test",transform=trn.Compose( [trn.ToTensor(), trn.Normalize(mean, std)]), download=False)
places365_data = dset.ImageFolder(root="../data/places365_standard/", transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)]))
lsunc_data = dset.ImageFolder(root="../data/LSUN", transform=trn.Compose([trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)]))
lsunr_data = dset.ImageFolder(root="../data/LSUN_resize", transform=trn.Compose([trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)]))
isun_data = dset.ImageFolder(root="../data/iSUN",transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))

texture_loader = torch.utils.data.DataLoader(texture_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
svhn_loader = torch.utils.data.DataLoader(svhn_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
places365_loader = torch.utils.data.DataLoader(places365_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
lsunc_loader = torch.utils.data.DataLoader(lsunc_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
lsunr_loader = torch.utils.data.DataLoader(lsunr_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
isun_loader = torch.utils.data.DataLoader(isun_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
cifar_loader = torch.utils.data.DataLoader(cifar_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
ood_num_examples = len(test_data) // 5
expected_ap = ood_num_examples / (ood_num_examples + len(test_data))
concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()


def get_ood_scores(loader, in_dist=False):
    _score = []
    net.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
                break
            data, target = data.cuda(), target.cuda()
            output = net(data)
            smax = to_np(F.softmax(output, dim=1))
            _score.append(-np.max(smax, axis=1))
    if in_dist:
        return concat(_score).copy() # , concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()

def get_and_print_results(ood_loader, in_score, num_to_avg=1):
    net.eval()
    aurocs, auprs, fprs = [], [], []
    for _ in range(num_to_avg):
        out_score = get_ood_scores(ood_loader)
        if args.out_as_pos: # OE's defines out samples as positive
            measures = get_measures(out_score, in_score)
        else:
            measures = get_measures(-in_score, -out_score)
        aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    print_measures(auroc, aupr, fpr, '')
    return fpr, auroc, aupr

def train(epoch, gamma):

    net.train()

    loss_avg = 0.0
    train_loader_out.dataset.offset = np.random.randint(len(train_loader_in.dataset))
    for batch_idx, (in_set, out_set) in enumerate(zip(train_loader_in, train_loader_out)):

        data, target = torch.cat((in_set[0], out_set[0]), 0), in_set[1]
        data, target = data.cuda(), target.cuda()

        x, emb = net.pred_emb(data)
        l_ce = F.cross_entropy(x[:len(in_set[0])], target)
        l_oe_old = - (x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()

        emb_oe = emb[len(in_set[0]):].detach()
        emb_bias = torch.rand_like(emb_oe) * 0.0001

        for _ in range(args.iter):
            emb_bias.requires_grad_()

            x_aug = net.fc(emb_bias + emb_oe)
            l_sur = - (x_aug.mean(1) - torch.logsumexp(x_aug, dim=1)).mean()
            r_sur = (emb_bias.abs()).mean(-1).mean()
            l_sur = l_sur - r_sur * gamma
            grads = torch.autograd.grad(l_sur, [emb_bias])[0]
            grads /= (grads ** 2).sum(-1).sqrt().unsqueeze(1)
            
            emb_bias = emb_bias.detach() + args.strength * grads.detach() # + torch.randn_like(grads.detach()) * 0.000001
            optimizer.zero_grad()
        
        gamma -= args.beta * (args.rho - r_sur.detach())
        gamma = gamma.clamp(min=0.0, max=args.gamma)
        if epoch >= args.warmup:
            x_oe = net.fc(emb[len(in_set[0]):] + emb_bias)
        else:    
            x_oe = net.fc(emb[len(in_set[0]):])
        
        l_oe = - (x_oe.mean(1) - torch.logsumexp(x_oe, dim=1)).mean()
        loss = l_ce + .5 * l_oe
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_avg = loss_avg * 0.8 + float(loss) * 0.2
        sys.stdout.write('\r epoch %2d %d/%d loss %.2f' %(epoch, batch_idx + 1, len(train_loader_in), loss_avg))
        scheduler.step()
    return gamma

def test():
    net.eval()
    correct = 0
    y, c = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = net(data)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()
    return correct / len(test_loader.dataset) * 100



net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate).cuda()
if args.dataset == 'cifar10':
    model_path = './models/cifar10_wrn_pretrained_epoch_99.pt'
else:
    model_path = './models/cifar100_wrn_pretrained_epoch_99.pt'
optimizer = torch.optim.SGD(net.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.decay, nesterov=True)
def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: cosine_annealing(step, args.epochs * len(train_loader_in), 1, 1e-6 / args.learning_rate))
net.load_state_dict(torch.load(model_path))

gamma = 0.01
for epoch in range(args.epochs):
    gamma = train(epoch, gamma)
   
    if epoch % 10 == 9: 
        net.eval()
        in_score = get_ood_scores(test_loader, in_dist=True)
        metric_ll = []
        metric_ll.append(get_and_print_results(svhn_loader, in_score))
        metric_ll.append(get_and_print_results(lsunc_loader, in_score))
        metric_ll.append(get_and_print_results(isun_loader, in_score))
        metric_ll.append(get_and_print_results(texture_loader, in_score))
        metric_ll.append(get_and_print_results(places365_loader, in_score))
        print('\n & %.2f & %.2f & %.2f' % tuple((100 * torch.Tensor(metric_ll).mean(0)).tolist()))

