import os
import time

try:
    from tensorboardX import SummaryWriter
except ImportError as error:
    print('tensorboardX is not available, please install it.')
    SummaryWriter = None


class Writer:
    def __init__(self):
        