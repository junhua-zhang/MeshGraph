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
        self.nexamples = 0
        self.ncorrect = 0

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

    def reset_counter(self):
        """
        counts # of correct examples
        """
        self.ncorrect = 0
        self.nexamples = 0

    def update_counter(self, ncorrect, nexamples):
        self.ncorrect += ncorrect
        self.nexamples += nexamples

    def print_loss(self, epoch, iters, loss):
        print('epoch : %d, iter : %d , loss : %.3f' %
              (epoch, iters, loss.item()))
        with open(self.train_loss, 'a') as train_loss:
            train_loss.write('epoch : %d, iter : %d , loss : %.3f\n' %
                             (epoch, iters, loss.item()))

    def plot_loss(self, epoch, i, loss, n):
        train_data_iter = i + epoch * n
        if self.display:
            self.display.add_scalar(
                'data/train_loss', loss.item(), train_data_iter)

    def plot_acc(self, acc, epoch):
        if self.display:
            self.display.add_scalar('data/test_acc', acc, epoch)

    def print_acc(self, epoch, acc):
        """ prints test accuracy to terminal / file """
        message = 'epoch: {}, TEST ACC: [{:.5} %]\n' \
            .format(epoch, acc * 100)
        print(message)
        with open(self.test_loss, "a") as log_file:
            log_file.write('%s\n' % message)

    @property
    def acc(self):
        return float(self.ncorrect) / self.nexamples

    def close(self):
        if self.display is not None:
            self.display.close()
