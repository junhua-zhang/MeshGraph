from options.test_options import test_options
from torch_geometric.datasets import ModelNet
from preprocess.preprocess import FaceToGraph
from torch_geometric.data import DataLoader
from models import create_model
from util.writer import Writer


def run_test(epoch=-1):
    print('Running Test')
    opt = test_options().parse()
    dataset = ModelNet(root=opt.datasets, name='10', train=False,
                       pre_transform=FaceToGraph(remove_faces=True))
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)
    model = create_model(opt)
    writer = Writer(opt)
    writer.reset_counter()
    for i, data in enumerate(loader):
        model.set_input(data)
        ncorrect, nexamples = model.test()
        writer.update_counter(ncorrect, nexamples)
    writer.print_acc(epoch, writer.acc)
    return writer.acc


if __name__ == '__main__':
    run_test()
