from train import *
from models.net_builder import *
from basic_utils.env_wrapper import Vec_env_wrapper
from models.agents import *
from basic_utils.options import *


def train_MountainCarContinuous_DDPG(load_model=False, render=False, save_every=None, prioritized=False):
    env = Vec_env_wrapper(name='MountainCarContinuous-v0', consec_frames=1, running_stat=True)
    ob_space = env.observation_space
    ac_space = env.action_space

    probtype = Deterministic(env.action_space)
    noise = OUNoise(ac_space.shape[0])

    pol_net = MLPs_pol(ob_space, net_topology_pol_vec, probtype.output_layers)
    target_pol_net = MLPs_pol(ob_space, net_topology_pol_vec, probtype.output_layers)
    q_net = MLPs_q_deterministic(ob_space, ac_space, net_topology_det_vec)
    target_q_net = MLPs_q_deterministic(ob_space, ac_space, net_topology_det_vec)
    if use_cuda:
        pol_net.cuda()
        target_pol_net.cuda()
        q_net.cuda()
        target_q_net.cuda()

    agent = DDPG_Agent(policy_net=pol_net,
                       policy_target_net=target_pol_net,
                       q_net=q_net,
                       q_target_net=target_q_net,
                       noise=noise,
                       probtype=probtype,
                       lr_updater=9e-4,
                       lr_optimizer=1e-3,
                       gamma=0.99,
                       tau=0.01,
                       update_target_every=None,
                       get_info=True)

    if load_model:
        agent.load_model("./save_model/" + env.name + "_" + agent.name)

    if prioritized:
        memory = PrioritizedReplayBuffer(memory_cap=20000, batch_size_q=64)
    else:
        memory = ReplayBuffer(memory_cap=20000, batch_size_q=64)

    t = Mem_Trainer(agent=agent,
                    env=env,
                    memory=memory,
                    n_worker=2,
                    step_num=4,
                    explore_len=10000,
                    ini_epsilon=0,
                    final_epsilon=0,
                    rand_explore_len=5000,
                    save_every=save_every,
                    render=render,
                    action_repeat=4,
                    print_every=10)
    t.train()


if __name__ == '__main__':
    train_MountainCarContinuous_DDPG()
