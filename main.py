import os

used_gpu = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = used_gpu

import argparse
from basic_utils.options import *
from train import Trainer
from basic_utils.utils import *
from basic_utils.layers import mujoco_layer_designer
from models.policies import Categorical
from models.net_builder import MLPs_v, MLPs_pol
from basic_utils.env_wrapper import Vec_env_wrapper
from models.agents import TRPO_Agent, A2C_Agent


def create_config():
    parser = argparse.ArgumentParser(description='Basic settings for Pytorch implemented RL.')

    # System Basic Setting
    parser.add_argument('--env', type=str, dest="ENV_NAME", default='CartPole-v0', help='the name of the environment')
    parser.add_argument('--agent', type=str, default='A2C_Agent', help='which kind of agent')
    parser.add_argument("--load_model", type=bool, default=False, help="whether to load model or not")
    parser.add_argument("--save_every", type=int, default=100, help="number of steps between two saving operations")
    parser.add_argument("--get_info", type=bool, default=True, help="whether to print update info or not")
    parser.add_argument('--disable_cuda', type=bool, default=False, help='whether to disable cuda')
    parser.add_argument('--disable_cudnn', type=bool, default=False, help='whether to disable cudnn')
    parser.add_argument("--print_every", type=int, default=10, help="print information after how many episodes")

    # RL General Setting
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--lambda", type=float, dest="lam", default=0.98,
                        help="lambda parameter from generalized advantage estimation")
    parser.add_argument("--batch_size_optimizer", type=int, default=256,
                        help="size of the minibatch when updating the baseline")
    parser.add_argument("--lr_optimizer", type=float, default=1e-3, help="learning rate of the baseline")
    parser.add_argument("--lr_updater", type=float, default=9e-4, help="learning rate of the policy")
    parser.add_argument("--epochs_updater", type=int, default=10,
                        help='epochs of a single updating process of the policy')
    parser.add_argument('--epoches_optimizer', type=int, default=10,
                        help='epochs of a single updating process of the baseline')
    parser.add_argument('--dist', type=str, default='DiagGauss', help='kind of distribution in continuous action space')
    parser.add_argument("--kl_target", type=float, default=0.003, help="KL divergence between old and new policy")

    # Env Setting
    parser.add_argument("--consec_frames", type=int, default=1, help="how many frames to concatenete in a row")
    parser.add_argument("--image_size", type=tuple, default=(84, 84), help="the size of processed frames")
    parser.add_argument("--running_stat", type=bool, default=False, help="whether to normalize the frames")
    parser.add_argument("--use_mujoco_setting", type=bool, default=False,
                        help="whether to automatically design the net architecture and lr for the mujoco environment")
    parser.add_argument('--render', type=bool, default=False, help='whether to render the environment')

    # Asynchronous Setting
    parser.add_argument('--timesteps_per_batch', type=int, default=5000,
                        help='total number of steps between two updates if not set None')
    parser.add_argument('--path_num', type=int, default=5,
                        help='fix the number of paths in every updating if not set None')
    parser.add_argument("--n_worker", type=int, default=5, help="total number of workers")

    # TRPO Setting
    parser.add_argument("--cg_damping", type=float, default=1e-3,
                        help="Add multiple of the identity to Fisher matrix during CG")
    parser.add_argument("--cg_iters", type=int, default=10, help="number of iterations in cg")

    # PPO Setting
    parser.add_argument("--kl_cutoff_coeff", type=float, default=50.0, help="penalty factor when kl is large")
    parser.add_argument("--clip_epsilon_init", type=float, default=0.2, help="factor of clipped loss")
    parser.add_argument("--beta_init", type=float, default=1.0, help="initialization of beta")
    parser.add_argument("--clip_range", type=tuple, default=(0.05, 0.3),
                        help="range of the adapted penalty factor")
    parser.add_argument("--adj_thres", type=tuple, default=(0.5, 2.0), help="threshold to magnify clip epsilon")
    parser.add_argument("--beta_range", type=tuple, default=(1 / 35.0, 35.0),
                        help="range of the adapted penalty factor")

    # Q Setting
    parser.add_argument("--batch_size_q", type=int, default=64, help="size of the minibatch in Q learning")
    parser.add_argument("--alpha", type=float, default=0.8, help="factor of the prioritize replay memory")
    parser.add_argument("--beta", type=float, default=0.6, help="factor of the prioritize replay memory")
    parser.add_argument("--memory_cap", type=int, default=200000, help="size of the replay mempry")
    parser.add_argument("--ini_epsilon", type=float, default=0.4, help="initial epsilon")
    parser.add_argument("--final_epsilon", type=float, default=0.01, help="final epsilon")
    parser.add_argument("--explore_len", type=float, default=10000, help="length of exploration")
    parser.add_argument("--rand_explore_len", type=float, default=1000, help="length of random exploration")
    parser.add_argument("--update_target_every", type=int, default=500, help="update the target after how many steps")

    # ES Setting
    parser.add_argument("--n_kid", type=int, default=10, help="number of kids")
    parser.add_argument("--sigma", type=float, default=0.05, help="scale of pertubation")
    parser.add_argument("--ES_lr", type=float, default=0.01, help="lr of ES")

    return parser.parse_args().__dict__


if __name__ == "__main__":
    cfg = create_config()
    cfg = update_default_config(MLP_OPTIONS, cfg)
    if cfg["use_mujoco_setting"]:
        cfg = mujoco_layer_designer(cfg)

    env = Vec_env_wrapper(name='CartPole-v0', consec_frames=1, running_stat=True)
    action_space = env.action_space
    observation_space = env.observation_space

    probtype = Categorical(env.action_space)
    pol_net = MLPs_pol(observation_space, net_topology_pol_vec, probtype.output_layers)
    v_net = MLPs_v(observation_space, net_topology_v_vec)

    if use_cuda:
        pol_net.cuda()
        v_net.cuda()

    agent = A2C_Agent(pol_net=pol_net, v_net=v_net, probtype=probtype)

    t = Trainer(agent=agent, env=env)
    t.train()
