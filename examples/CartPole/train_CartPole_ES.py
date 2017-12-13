from train import *
from models.net_builder import *
from basic_utils.env_wrapper import Vec_env_wrapper
from models.agents import *
from basic_utils.options import *


def train_CartPole_ES(load_model=False, render=False, save_every=None):
    env = Vec_env_wrapper(name='CartPole-v1', consec_frames=1, running_stat=True)
    ob_space = env.observation_space

    probtype = Categorical(env.action_space)

    pol_net = MLPs_pol(ob_space, net_topology_pol_vec, probtype.output_layers)

    if use_cuda:
        pol_net.cuda()

    agent = Evolution_Agent(pol_net=pol_net,
                            probtype=probtype,
                            lr_updater=0.01,
                            sigma=0.05,
                            n_kid=10)

    if load_model:
        agent.load_model("./save_model/" + env.name + "_" + agent.name)

    t = ES_Trainer(agent,
                   env,
                   path_num=1,
                   n_worker=10,
                   save_every=save_every,
                   render=render,
                   action_repeat=1,
                   print_every=10)
    t.train()


if __name__ == '__main__':
    train_CartPole_ES()
