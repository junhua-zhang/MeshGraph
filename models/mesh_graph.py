import torch
import torch.nn
import os.path as osp
from . import networks
from models.optimizer import adabound


class mesh_graph:
    def __init__(self, opt):
        self.opt = opt
        self.cuda = opt.cuda
        self.is_train = opt.is_train
        self.device = torch.device('cuda:{}'.format(
            self.cuda[0]) if self.cuda else 'cpu')
        self.save_dir = osp.join(opt.ckpt_root, opt.name)
        self.optimizer = None
        self.loss = None

        # init mesh data
        pass

        # init network
        self.net.train(self.is_train)
        pass

        # criterion
        self.loss = networks.get_loss(self.opt).to(self.device)

        if self.is_train:
            self.optimizer = adabound.AdaBound(
                self.net.parameters(), lr=self.opt.lr, final_lr=self.opt.final_lr)
            self.scheduler = networks.get_scheduler(self.optimizer, self.opt)
            
