import os
import time

try:
    from tensorboardX import SummaryWriter
except ImportError as error:
    print('tensorboardX is not available, please install it.')
    SummaryWriter = None


class Writer:
    def __init__(self, opt):
        self.opt = opt
        self.name = opt.name
        self.save_path = os.path.join(opt.ckpt_root, opt.name)
        self.train_loss = os.path.join(self.save_path, 'train_loss.txt')
        self.test_loss = os.path.join(self.save_path, 'test_loss.txt')

        # set display
        if opt.is_train and not SummaryWriter is not None:
            self.display = SummaryWriter(comment=opt.name)
        else:
            self.display = None

        self.start_logs()

    def start_logs(self):
        ''' create log file'''
        if self.opt.is_train:
            with open(self.train_loss, 'a') as train_loss:
                now = time.strftime('%c')
                train_loss.write(
                    '================ Training Loss (%s) ================\n' % now)
        else:
            with open(self.test_loss, 'a') as test_loss:
                now = time.strftime('%c')
                test_loss.write(
                    '================ Test Loss (%s) ================\n' % now)
