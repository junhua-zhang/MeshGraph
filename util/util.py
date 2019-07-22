import os
import os.path as osp


def check_dir(dir):
    if not osp.exists(dir):
        os.makedirs(dir)


