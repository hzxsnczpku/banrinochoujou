from train import *
from models.net_builder import *
from basic_utils.env_wrapper import Vec_env_wrapper
from models.agents import *
from basic_utils.options import *
from basic_utils.data_generator import *
from basic_utils.exploration_noise import OUNoise_Exploration


def train_MountainCarContinuous_DDPG(load_model=False, render=False, save_every=None, prioritized=False, gamma=0.99):
    # set_seed
    # torch.manual_seed(8933)

    # set environment
    env = Vec_env_wrapper(name='MountainCarContinuous-v0', consec_frames=1, running_stat=False)
    ob_space = env.observation_space
    ac_space = env.action_space

    probtype = Deterministic(env.action_space)

    # set neural network
    pol_net = MLPs_pol(ob_space, net_topology_pol_vec, probtype.output_layers)
    target_pol_net = MLPs_pol(ob_space, net_topology_pol_vec, probtype.output_layers)
    q_net = MLPs_q_deterministic(ob_space, ac_space, net_topology_det_vec)
    target_q_net = MLPs_q_deterministic(ob_space, ac_space, net_topology_det_vec)

    if use_cuda:
        pol_net.cuda()
        target_pol_net.cuda()
        q_net.cuda()
        target_q_net.cuda()

    # set noise
    noise = OUNoise_Exploration(ac_space.shape[0], init_epsilon=0.99, final_epsilon=0.05, explore_len=100000)

    # set agent
    agent = DDPG_Agent(policy_net=pol_net,
                       policy_target_net=target_pol_net,
                       q_net=q_net,
                       q_target_net=target_q_net,
                       probtype=probtype,
                       lr_updater=1e-3,
                       lr_optimizer=1e-3,
                       tau=0.01,
                       update_target_every=None,
                       get_info=True)

    if load_model:
        agent.load_model("./save_model/" + env.name + "_" + agent.name)

    # set replay memory
    if prioritized:
        memory = PrioritizedReplayBuffer(memory_cap=10000, batch_size_q=100)
    else:
        memory = ReplayBuffer(memory_cap=10000, batch_size_q=100)

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
        Predict_Next_Action(agent.policy, target=True),
        Calculate_Next_Q_Value_AS(agent.baseline),
        Calculate_Q_Target(gamma),
        Extract_Item_By_Name(["observation", "action", "y_targ"]),
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
                       print_every=5)
    t.train()


if __name__ == '__main__':
    train_MountainCarContinuous_DDPG()
