import torch
import torch.nn
import os.path as osp
from . import networks
from models.optimizer import adabound
from torch_geometric.utils import remove_self_loops,contains_self_loops,contains_isolated_nodes

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
            self.scheduler = networks.get_scheduler(self.optimizer, self.opt)

        if not self.is_train or opt.continue_train:
            self.load_state(opt.last_epoch)

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
        print(data)
        print(data.num_nodes)
        gt_label = data.y
        edge_index = data.edge_index
        vertex_features = data.pos

        print(contains_self_loops(edge_index))
        print(contains_isolated_nodes(edge_index,data.num_nodes))
        print(data.norm)
        self.labels = gt_label.to(self.device).long()
        self.vertexes = vertex_features.float().to(self.device)
        self.edges = edge_index.t().long().to(self.device)

        print(self.vertexes.size())
        print(self.edges.size())
        print(self.edges)
