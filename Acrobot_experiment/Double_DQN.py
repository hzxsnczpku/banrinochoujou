from train import *
from models.net_builder import *
from basic_utils.env_wrapper import Vec_env_wrapper
from models.agents import *
from basic_utils.options import *
from basic_utils.exploration_noise import *


def train_CartPole_DQN(load_model=False, render=False, save_every=None):
    torch.manual_seed(8833)
    env = Vec_env_wrapper(name='Acrobot-v1', consec_frames=1, running_stat=False, seed=23333)
    action_space = env.action_space
    observation_space = env.observation_space

    net = MLPs_q(observation_space, action_space, net_topology_q_vec)
    target_net = MLPs_q(observation_space, action_space, net_topology_q_vec)

    noise = EpsilonGreedy_Exploration(action_n=action_space.n,
                                      explore_len=10000,
                                      init_epsilon=0.05,
                                      final_epsilon=0.05)

    if use_cuda:
        net.cuda()
        target_net.cuda()

    agent = Double_DQN_Agent(net=net,
                             target_net=target_net,
                             gamma=0.95,
                             lr=1e-3,
                             update_target_every=1000,
                             get_info=True)

    memory = ReplayBuffer(memory_cap=10000, batch_size_q=64)

    if load_model:
        agent.load_model("./save_model/" + env.name + "_" + agent.name)

    t = Mem_Trainer(agent=agent,
                    env=env,
                    memory=memory,
                    n_worker=1,
                    step_num=1,
                    rand_explore_len=1000,
                    save_every=save_every,
                    render=render,
                    print_every=10,
                    noise=noise)
    t.train()


if __name__ == '__main__':
    train_CartPole_DQN(save_every=5)
