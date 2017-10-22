import os
used_gpu = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = used_gpu

import argparse
from train import Asy_train


def create_config():
    parser = argparse.ArgumentParser(description='Basic settings for Pytorch implemented RL.')
    parser.add_argument('--env', type=str, dest="ENV_NAME", default='HalfCheetah-v1', help='the name of the environment')
    parser.add_argument('--agent', type=str, default='Ppo_clip_Agent', help='the kind of the agent')
    parser.add_argument('--timesteps_per_batch', type=int, default=25000, help='total number of steps between two updates')
    parser.add_argument("--n_worker", type=int, default=10, help="total number of workers")
    parser.add_argument("--update_threshold", type=int, default=10, help="update after how many workers have finished sampling")

    return parser.parse_args().__dict__

if __name__ == "__main__":
    cfg = create_config()
    Asy_train.train(cfg)
