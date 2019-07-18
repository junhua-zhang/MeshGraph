import argparse
import os
import torch
from util import util


class base_options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.is_init = False

    def initialize(self):
        # dataset
        self.parser.add_argument(
            '--datasets', required=True, help='dataset to be train/test'
        )
        self.parser.add_argument(
            '--task', choices={'cls', 'seg'}, default='cls', help='task for network to loader model'
        )
        # general arg
        self.parser.add_argument(
            '--cuda', type=str, default='0', help='cuda device number e.g. 0 0,1,2, 0,2. use -1 for CPU'
        )
        self.parser.add_argument(
            '--ckpt_root', type=str, default='./ckpt_root', help='model saved path'
        )
        self.parser.add_argument(
            '--ckpt', type=str, default='./ckpt', help='final model saved path'
        )
        self.parser.add_argument(
            '--name', type=str, default='debug', help='model saved path'
        )

    def parse(self):
        if not self.is_init:
            self.initialize()
        self.opt, _ = self.parser.parse_known_args()
        self.opt.is_train = self.is_train  # train or test

        cuda_ids = self.opt.cuda.split(',')
        self.opt.cuda = []
        self.opt.milestones = [int(x) for x in self.opt.milestones.split(',')]
        for cuda_id in cuda_ids:
            id = int(cuda_id)
            if id >= 0:
                self.opt.cuda.append(id)
        # set gpu id
        if len(self.opt.cuda) > 0:
            torch.cuda.set_device(self.opt.cuda[0])

        args = vars(self.opt)

        if self.is_train:
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')

            # check dir
            util.check_dir(os.path.join(self.opt.ckpt_root, self.opt.name))
            util.check_dir(os.path.join(self.opt.ckpt, self.opt.name))

            # save train options
            expr_dir = os.path.join(self.opt.ckpt_root, self.opt.name)
            file_name = os.path.join(expr_dir, 'opt.txt')

            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
