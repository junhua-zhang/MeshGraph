import time
from options.train_options import train_options
from torch_geometric.datasets import ModelNet
from torch_geometric.data import DataLoader
from models import create_model
from util.writer import Writer


if __name__ == '__main__':
    opt = train_options().parse()

    # load dataset
    dataset = ModelNet(root=opt.datasets, name='40')
    print('# training meshes = %d' % len(dataset))
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    model = create_model(opt)
    writer = Writer(opt)
    total_steps = 0

    for epoch in range(0, 100):
        pass
