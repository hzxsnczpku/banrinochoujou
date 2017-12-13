import os

used_gpu = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = used_gpu

import argparse
from basic_utils.options import *
from train import *
from basic_utils.utils import *
from basic_utils.layers import mujoco_layer_designer
from models.policies import *
from basic_utils.replay_memory import *
from models.net_builder import *
from basic_utils.env_wrapper import Vec_env_wrapper
from models.agents import *

if __name__ == "__main__":
    """cfg = create_config()
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

    agent = PPO_clip_Agent(pol_net=pol_net, v_net=v_net, probtype=probtype)

    if cfg['load_model']:
        agent.load_model("./save_model/" + env.name + "_" + agent.name)

    t = Path_Trainer(agent=agent, env=env)
    t.train()"""

    env = Vec_env_wrapper(name='CartPole-v0', consec_frames=1, running_stat=True)
    action_space = env.action_space
    observation_space = env.observation_space

    net = MLPs_q(observation_space, action_space, net_topology_q_vec)
    target_net = MLPs_q(observation_space, action_space, net_topology_q_vec)

    if use_cuda:
        net.cuda()
        target_net.cuda()

    agent = Double_DQN_Agent(net=net,
                             target_net=target_net,
                             gamma=0.95)
    memory = PrioritizedReplayBuffer(memory_cap=2000,
                                     batch_size_q=64)

    # agent.load_model("./save_model/" + env.name + "_" + agent.name)

    t = Mem_Trainer(agent=agent,
                    env=env,
                    memory=memory,
                    n_worker=1,
                    step_num=1,
                    explore_len=10000,
                    ini_epsilon=1.0,
                    final_epsilon=0.01,
                    rand_explore_len=1000,
                    print_every=50)
    t.train()
