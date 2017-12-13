from train import *
from models.net_builder import *
from basic_utils.env_wrapper import Vec_env_wrapper
from models.agents import *
from basic_utils.options import *


def train_CartPole_DQN(load_model=False, render=False, save_every=None, double=False, prioritized=False):
    env = Vec_env_wrapper(name='CartPole-v0', consec_frames=1, running_stat=True)
    action_space = env.action_space
    observation_space = env.observation_space

    net = MLPs_q(observation_space, action_space, net_topology_q_vec)
    target_net = MLPs_q(observation_space, action_space, net_topology_q_vec)

    if use_cuda:
        net.cuda()
        target_net.cuda()

    if double:
        agent = Double_DQN_Agent(net=net, target_net=target_net, gamma=0.95)
    else:
        agent = DQN_Agent(net=net, target_net=target_net, gamma=0.95)
    if prioritized:
        memory = PrioritizedReplayBuffer(memory_cap=2000, batch_size_q=64)
    else:
        memory = ReplayBuffer(memory_cap=2000, batch_size_q=64)

    if load_model:
        agent.load_model("./save_model/" + env.name + "_" + agent.name)

    t = Mem_Trainer(agent=agent,
                    env=env,
                    memory=memory,
                    n_worker=1,
                    step_num=1,
                    explore_len=10000,
                    ini_epsilon=1.0,
                    final_epsilon=0.01,
                    rand_explore_len=1000,
                    save_every=save_every,
                    render=render,
                    print_every=50)
    t.train()


if __name__ == '__main__':
    train_CartPole_DQN()
