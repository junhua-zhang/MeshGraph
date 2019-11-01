import torch
import torch.nn as nn
import os.path as osp
import numpy as np
from . import networks
import torch.optim as optim
from models.optimizer import adabound
from torch_geometric.utils import remove_self_loops, contains_self_loops, contains_isolated_nodes


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
        self.nclasses = opt.nclasses

        # init network
        self.net = networks.get_net(opt)
        self.net.train(self.is_train)

        # criterion
        self.loss = networks.get_loss(self.opt).to(self.device)

        if self.is_train:
            self.optimizer = adabound.AdaBound(
                params=self.net.parameters(), lr=self.opt.lr, final_lr=self.opt.final_lr)
            # self.optimizer = optim.SGD(self.net.parameters(
            # ), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
            self.scheduler = networks.get_scheduler(self.optimizer, self.opt)
        if not self.is_train or opt.continue_train:
            self.load_state(opt.last_epoch)

    def test(self):
        """tests model
        returns: number correct and total number
        """
        with torch.no_grad():
            out = self.forward()
            # compute number of correct
            pred_class = torch.max(out, dim=1)[1]
            # print(pred_class)
            label_class = self.labels
            correct = self.get_accuracy(pred_class, label_class)
        return correct, len(label_class)

    def get_accuracy(self, pred, labels):
        """computes accuracy for classification / segmentation """
        if self.opt.task == 'cls':
            correct = pred.eq(labels).sum()
        return correct

    def load_state(self, last_epoch):
        ''' load epoch '''
        load_file = '%s_net.pth' % last_epoch
        load_path = osp.join(self.save_dir, load_file)
        net = self.net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        net.load_state_dict(state_dict)

    def set_input_data(self, data):
        '''set input data'''

        gt_label = data.y
        edge_index = data.edge_index
        if self.opt.batch_size == 1:
            nodes_features = data.x.unsqueeze(0)
            # neigbour_index = data.pos.unsqueeze(0)
        else:
            nodes_features = data.x.view(self.opt.batch_size, 1024, -1)
            neigbour_index = data.pos.view(self.opt.batch_size, -1, 3)

        self.labels = gt_label.to(self.device).long()
        self.edge_index = edge_index.to(self.device).long()

        self.neigbour_index = neigbour_index.to(self.device).long()
        self.centers = nodes_features[:, :, :3].transpose(1, 2)
        self.corners = nodes_features[:, :, 3:12].transpose(1, 2)
        self.normals = nodes_features[:, :, 12:].transpose(1, 2)
        self.x = data.x[:, -3:].to(self.device).float()

    def forward(self):
        out = self.net(self.x, self.edge_index, self.centers,
                       self.corners, self.normals, self.neigbour_index)
        return out

    def backward(self, out):
        self.loss_val = self.loss(out, self.labels)
        self.loss_val.backward()

    def optimize(self):
        '''
        optimize paramater
        '''
        self.optimizer.zero_grad()
        out = self.forward()
        self.backward(out)
        nn.utils.clip_grad_norm_(self.net.parameters(), self.opt.grad_clip)
        self.optimizer.step()

    def save_network(self, which_epoch):
        '''
        save network to disk
        '''
        save_filename = '%s_net.pth' % (which_epoch)
        save_path = osp.join(self.save_dir, save_filename)
        if len(self.cuda) > 0 and torch.cuda.is_available():
            torch.save(self.net.module.cpu().state_dict(), save_path)
            self.net.cuda(self.cuda[0])
        else:
            torch.save(self.net.cpu().state_dict(), save_path)
