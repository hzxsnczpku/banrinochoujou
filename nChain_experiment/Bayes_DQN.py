from train import *
from models.net_builder import *
from basic_utils.env_wrapper import Vec_env_wrapper
from models.agents import *
from basic_utils.options import *
from basic_utils.exploration_noise import *


def train_CartPole_DQN(load_model=False, render=False, save_every=None, double=False, prioritized=False):
    torch.manual_seed(8833)
    np.random.seed(8833)
    env = Vec_env_wrapper(name='100Chain', consec_frames=1, running_stat=False, seed=23333)
    action_space = env.action_space
    observation_space = env.observation_space

    net = MLPs_q(observation_space, action_space, net_topology_q_vec)
    mean_net = MLPs_q(observation_space, action_space, net_topology_q_vec)
    std_net = MLPs_q(observation_space, action_space, net_topology_q_vec)
    target_net = MLPs_q(observation_space, action_space, net_topology_q_vec)
    target_mean_net = MLPs_q(observation_space, action_space, net_topology_q_vec)
    target_std_net = MLPs_q(observation_space, action_space, net_topology_q_vec)
    noise = NoNoise_Exploration()

    if use_cuda:
        net.cuda()
        mean_net.cuda()
        std_net.cuda()
        target_net.cuda()
        target_mean_net.cuda()
        target_std_net.cuda()

    agent = Bayesian_DQN_Agent(net,
                               mean_net,
                               std_net,
                               target_net,
                               target_mean_net,
                               target_std_net,
                               alpha=1,
                               beta=1e-4,
                               gamma=0.99,
                               lr=1e-3,
                               scale=1e-4,
                               update_target_every=500,
                               get_info=True)

    if prioritized:
        memory = PrioritizedReplayBuffer(memory_cap=10000,
                                         batch_size_q=64,
                                         alpha=0.8,
                                         beta=0.6)
    else:
        memory = ReplayBuffer(memory_cap=10000, batch_size_q=64)

    if load_model:
        agent.load_model("./save_model/" + env.name + "_" + agent.name)

    t = Mem_Trainer(agent=agent,
                    env=env,
                    memory=memory,
                    n_worker=1,
                    step_num=1,
                    rand_explore_len=100,
                    save_every=save_every,
                    render=render,
                    print_every=10,
                    noise=noise)
    t.train()


if __name__ == '__main__':
    train_CartPole_DQN(save_every=5)
