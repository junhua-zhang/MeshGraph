import torch
import torch.nn

from torch.optim import lr_scheduler
import torch.nn.functional as F


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'step':
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, gamma=opt.gamma, milestones=opt.milestones)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler