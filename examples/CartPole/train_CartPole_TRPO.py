from train import *
from models.net_builder import *
from basic_utils.env_wrapper import Vec_env_wrapper
from models.agents import *
from basic_utils.options import *


def train_CartPole_TRPO(load_model=False, render=False, save_every=None):
    torch.manual_seed(2)
    env = Vec_env_wrapper(name='MountainCarContinuous-v0', consec_frames=1, running_stat=True)
    ob_space = env.observation_space

    probtype = DiagGauss(env.action_space)

    pol_net = MLPs_pol(ob_space, net_topology_pol_vec, probtype.output_layers)
    v_net = MLPs_v(ob_space, net_topology_v_vec)
    if use_cuda:
        pol_net.cuda()
        v_net.cuda()

    agent = TRPO_Agent(pol_net=pol_net,
                       v_net=v_net,
                       probtype=probtype,
                       lr_optimizer=1e-3,
                       lam=0.98,
                       epochs_v=10,
                       gamma=0.99,
                       cg_iters=10,
                       max_kl=0.01,
                       batch_size=256,
                       cg_damping=1e-3,
                       get_info=True)

    if load_model:
        agent.load_model("./save_model/" + env.name + "_" + agent.name)

    t = Path_Trainer(agent,
                     env,
                     n_worker=10,
                     path_num=1,
                     save_every=save_every,
                     render=render,
                     action_repeat=1,
                     print_every=10)
    t.train()


if __name__ == '__main__':
    train_CartPole_TRPO()
