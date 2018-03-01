from train import *
from models.net_builder import *
from basic_utils.env_wrapper import Vec_env_wrapper
from models.agents import *
from basic_utils.options import *
from basic_utils.data_generator import *
from basic_utils.exploration_noise import OUNoise_Exploration


def train_Bullet_DDPG(env='HalfCheetahBulletEnv-v0', consec_frames=3, running_stat=False, load_model=False,
                      render=False, save_every=None, gamma=0.99, seed=None):
    # set_seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # set environment
    env = Vec_env_wrapper(name=env, consec_frames=consec_frames, running_stat=running_stat, seed=seed)
    ob_space = env.observation_space
    ac_space = env.action_space

    probtype = Deterministic(env.action_space)

    # set neural network
    pol_net = MLPs_pol(ob_space, net_topology_pol_vec, probtype.output_layers)
    q_net = MLPs_q_deterministic(ob_space, ac_space, net_topology_det_vec)

    if use_cuda:
        pol_net.cuda()
        q_net.cuda()

    # set noise
    noise = OUNoise_Exploration(ac_space.shape[0], init_epsilon=0.1, final_epsilon=0.1, explore_len=100000)

    # set agent
    agent = DDPG_Agent(policy_net=pol_net,
                       policy_target_net=None,
                       q_net=q_net,
                       q_target_net=None,
                       probtype=probtype,
                       lr_updater=1e-4,
                       lr_optimizer=1e-3,
                       tau=0.01,
                       update_target_every=None,
                       get_info=True)

    if load_model:
        agent.load_model("./save_model/" + env.name + "_" + agent.name)

    # set data generator
    data_generator = Parallel_Path_Data_Generator(agent=agent,
                                                  env=env,
                                                  n_worker=10,
                                                  path_num=1,
                                                  action_repeat=1,
                                                  render=render,
                                                  noise=noise)

    # set data processor
    single_processors = [
        Scale_Reward(1 - gamma),
        Calculate_Return(gamma),
        Extract_Item_By_Name(["observation", "action", "y_targ"]),
        Concatenate_Paths()
    ]
    processor = Ensemble(single_processors)

    # set trainer
    t = Path_Trainer(agent,
                     env,
                     data_generator=data_generator,
                     data_processor=processor,
                     save_every=save_every,
                     print_every=10,
                     log_dir_name='./train_log/base/')
    t.train()


if __name__ == '__main__':
    train_Bullet_DDPG()
