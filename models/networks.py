import torch
import torch.nn as nn

from torch.optim import lr_scheduler
import torch.nn.functional as F


def init_weights(net, init_type, init_gain):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


def init_net(net, init_type, init_gain, cuda_ids):
    if len(cuda_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda(cuda_ids[0])
        net = net.cuda()
        net = torch.nn.DataParallel(net, cuda_ids)
    if init_type != 'none':
        init_weights(net, init_type, init_gain)
    return net


def get_net(opt):
    net = None
    if opt.arch == 'meshconv':
        net = MeshGraph()
    else:
        raise NotImplementedError(
            'model name [%s] is not implemented' % opt.arch)
    return init_net(net, opt.init_type, opt.init_gain, opt.cuda)


def get_loss(opt):
    if opt.task == 'cls':
        loss = nn.CrossEntropyLoss()
    elif opt.task == 'seg':
        loss = nn.CrossEntropyLoss(ignore_index=-1)
    return loss


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'step':
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, gamma=opt.gamma, milestones=opt.milestones)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


class MeshGraph(nn.Module):
    """Some Information about MeshGraph"""

    def __init__(self):
        super(MeshGraph, self).__init__()

    def forward(self, x):

        return x
