from train import *
from models.net_builder import *
from basic_utils.env_wrapper import Vec_env_wrapper
from models.agents import *
from basic_utils.options import *
from basic_utils.exploration_noise import *
from basic_utils.data_generator import *


def train_CartPole_DQN(load_model=False, render=False, save_every=None, double=False, prioritized=False, gamma=0.95):
    # set_seed
    # torch.manual_seed(8933)

    # set environment
    env = Vec_env_wrapper(name='CartPole-v1', consec_frames=1, running_stat=False, seed=23333)
    action_space = env.action_space
    observation_space = env.observation_space

    # set neural network
    net = MLPs_q(observation_space, action_space, net_topology_q_vec)
    target_net = MLPs_q(observation_space, action_space, net_topology_q_vec)

    if use_cuda:
        net.cuda()
        target_net.cuda()

    # set noise
    noise = EpsilonGreedy_Exploration(action_n=action_space.n,
                                      explore_len=100000,
                                      init_epsilon=0.1,
                                      final_epsilon=0.01)

    # set agent
    if double:
        agent = Double_DQN_Agent(net=net, target_net=target_net, gamma=0.95)
    else:
        agent = DQN_Agent(net=net,
                          target_net=target_net,
                          gamma=0.99,
                          lr=1e-3,
                          update_target_every=1000,
                          get_info=True)
    if load_model:
        agent.load_model("./save_model/" + env.name + "_" + agent.name)

    # set replay memory
    if prioritized:
        memory = PrioritizedReplayBuffer(memory_cap=100000, batch_size_q=128)
    else:
        memory = ReplayBuffer(memory_cap=100000, batch_size_q=128)

    # set data generator
    data_generator = Parallel_Memory_Data_Generator(agent=agent,
                                                    env=env,
                                                    memory=memory,
                                                    n_worker=1,
                                                    rand_explore_len=1000,
                                                    noise=noise,
                                                    step_num=1,
                                                    action_repeat=1,
                                                    render=render)

    # set data processor
    single_processors = [
        Scale_Reward(1 - gamma),
        Calculate_Next_Q_Value(agent.baseline, double=double),
        Calculate_Q_Target(gamma),
        Extract_Item_By_Name(["observation", "action", "y_targ", 'weights']),
        Concatenate_Paths()
    ]
    processor = Ensemble(single_processors)

    # set trainer
    t = Memory_Trainer(agent=agent,
                       env=env,
                       memory=memory,
                       data_generator=data_generator,
                       data_processor=processor,
                       save_every=save_every,
                       print_every=10)
    t.train()


if __name__ == '__main__':
    train_CartPole_DQN(prioritized=True)
