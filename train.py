import os

import argparse

from torch.backends import cudnn

from dataloder import get_loader
from psgan.solver import Solver
from setup import setup_config, setup_argparser
import json

def train_net(config):
    # enable cudnn
    cudnn.benchmark = True

    data_loader = get_loader(config)
    solver = Solver(config, data_loader=data_loader)
    solver.train()


if __name__ == '__main__':
    args = setup_argparser().parse_args()
    config = setup_config(args)
    print("Call with args:")
    print(config)
    os.makedirs(config.LOG.LOG_PATH, exist_ok=True)
    with open(f'{config.LOG.LOG_PATH}/config.yaml', mode='w') as f:
        f.write(config.dump())

    train_net(config)
