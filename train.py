import time
from options.train_options import train_options
from torch_geometric.datasets import ModelNet
from preprocess.preprocess import FaceToGraph
from torch_geometric.data import DataLoader
from models import create_model
from util.writer import Writer


if __name__ == '__main__':
    opt = train_options().parse()

    # load dataset
    dataset = ModelNet(root=opt.datasets, name='40',
                       pre_transform=FaceToGraph(remove_faces=True))
    print('# training meshes = %d' % len(dataset))
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    model = create_model(opt)
    writer = Writer(opt)
    total_steps = 0

    for epoch in range(0, opt.epoch):
        start_time = time.time()
        count = 0

        for i, data in enumerate(loader):
            total_steps += opt.batch_size
            # model.set_input_data(data)
            if total_steps % opt.frequency == 0:
                pass
            break

        break
