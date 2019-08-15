import time
from options.train_options import train_options
from torch_geometric.datasets import ModelNet
from preprocess.preprocess import FaceToGraph, FaceToEdge
from torch_geometric.data import DataLoader
from models import create_model
from util.writer import Writer
from test import run_test

if __name__ == '__main__':
    opt = train_options().parse()

    # load dataset
    dataset = ModelNet(root=opt.datasets, name=str(opt.nclasses),
                       pre_transform=FaceToGraph(remove_faces=True))
    print('# training meshes = %d' % len(dataset))
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    model = create_model(opt)
    writer = Writer(opt)
    total_steps = 0

    for epoch in range(1, opt.epoch):
        start_time = time.time()
        count = 0

        for i, data in enumerate(loader):
            # break
            total_steps += opt.batch_size
            count += opt.batch_size
            model.set_input_data(data)
            model.optimize()
            if total_steps % opt.frequency == 0:
                loss_val = model.loss_val
                writer.print_loss(epoch, count, loss_val)
                writer.plot_loss(epoch, count, loss_val, len(dataset))

            if i % opt.loop_frequency == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_network('latest')
            # break

        if epoch % opt.epoch_frequency == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_network('latest')
            model.save_network(epoch)

        if epoch % opt.test_frequency == 0:
            acc = run_test(epoch)
            writer.plot_acc(acc, epoch)
        # break
    writer.close()
