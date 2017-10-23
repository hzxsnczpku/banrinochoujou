import os

used_gpu = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = used_gpu

import argparse
from train import Asy_train


def create_config():
    parser = argparse.ArgumentParser(description='Basic settings for Pytorch implemented RL.')

    # System Basic Setting
    parser.add_argument('--env', type=str, dest="ENV_NAME", default='Pendulum-v0', help='the name of the environment')
    parser.add_argument('--agent', type=str, default='Ppo_adapted_Agent', help='which kind of agent')
    parser.add_argument("--load_model", type=bool, default=False, help="whether load model or not")
    parser.add_argument("--save_every", type=int, default=None, help="number of steps between two saving operations")

    # RL General Setting
    parser.add_argument("--gamma", type=float, default=0.995, help="discount factor")
    parser.add_argument("--lambda", type=float, dest="lam", default=0.98,
                        help="lambda parameter from generalized advantage estimation")
    parser.add_argument("--batch_size", type=int, default=256, help="size of minibatches")
    parser.add_argument("--lr_optimizer", type=float, default=1e-3, help="learning rate of the baseline")
    parser.add_argument("--lr_updater", type=float, default=9e-4, help="learning rate of the policy")

    # Env Setting
    parser.add_argument("--consec_frames", type=int, default=4, help="how many frames to concatenete in a row")
    parser.add_argument("--image_size", type=tuple, default=(84, 84), help="the size of processed frames")
    parser.add_argument("--running_stat", type=bool, default=True,
                        help="whether to normalize the frames and rewards or not")
    parser.add_argument("--alpha", type=float, default=1 / 50000, help="factor of soft updating in running_stat mode")
    parser.add_argument("--use_mujoco_setting", type=bool, default=True,
                        help="whether to automatically design the net architecture and lr for the mujoco environment")

    # Asynchronous Setting
    parser.add_argument('--timesteps_per_batch_worker', type=int, default=4000,
                        help='total number of steps between two updates')
    parser.add_argument("--n_worker", type=int, default=1, help="total number of workers")
    parser.add_argument("--update_threshold", type=int, default=1,
                        help="update after how many workers have finished sampling")

    # TRPO Agent Setting
    parser.add_argument("--cg_damping", type=float, default=1e-3,
                        help="Add multiple of the identity to Fisher matrix during CG")
    parser.add_argument("--cg_iters", type=int, default=10, help="number of iterations in cg")
    parser.add_argument("--max_kl", type=float, default=1e-2, help="KL divergence between old and new policy")

    # PPO Agent Setting
    parser.add_argument("--kl_target", type=float, default=0.003,
                        help="KL divergence between old and new policy(used in PPO)")
    parser.add_argument("--kl_cutoff_coeff", type=float, default=50.0, help="penalty factor when kl is large")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="factor of clipped loss")
    parser.add_argument("--beta_upper", type=float, default=35.0, help="upper bound of the adapted penalty factor")
    parser.add_argument("--beta_lower", type=float, default=1 / 35.0, help="lower bound of the adapted penalty factor")
    parser.add_argument("--beta_adj_thres_u", type=float, default=2.0, help="threshold to magnify beta")
    parser.add_argument("--beta_adj_thres_l", type=float, default=0.5, help="threshold to cut off beta")

    return parser.parse_args().__dict__


if __name__ == "__main__":
    cfg = create_config()
    Asy_train.train(cfg)
