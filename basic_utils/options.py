net_topology_pol_vec = [
    {'kind': 'dense', 'units': 64},
    {'kind': 'Tanh'},
    {'kind': 'dense', 'units': 64},
    {'kind': 'Tanh'},
    {'kind': 'dense', 'units': 64},
]

net_topology_pol_fig = [
    {'kind': 'conv', 'filters': 32, 'stride': 4, 'ker_size': 8},
    {'kind': 'ReLU'},
    {'kind': 'conv', 'filters': 64, 'stride': 2, 'ker_size': 4},
    {'kind': 'ReLU'},
    {'kind': 'conv', 'filters': 64, 'stride': 1, 'ker_size': 3},
    {'kind': 'flatten'},
    {'kind': 'dense', 'units': 512},
]

net_topology_v_fig = [
    {'kind': 'conv', 'filters': 32, 'stride': 4, 'ker_size': 8},
    {'kind': 'ReLU'},
    {'kind': 'conv', 'filters': 64, 'stride': 2, 'ker_size': 4},
    {'kind': 'ReLU'},
    {'kind': 'conv', 'filters': 64, 'stride': 1, 'ker_size': 3},
    {'kind': 'flatten'},
    {'kind': 'dense', 'units': 512},
]

net_topology_v_vec = [
    {'kind': 'dense', 'units': 64},
    {'kind': 'Tanh'},
    {'kind': 'dense', 'units': 64},
    {'kind': 'Tanh'},
    {'kind': 'dense', 'units': 64},
]

net_topology_q_fig = [
    {'kind': 'conv', 'filters': 32, 'stride': 4, 'ker_size': 8},
    {'kind': 'ReLU'},
    {'kind': 'conv', 'filters': 64, 'stride': 2, 'ker_size': 4},
    {'kind': 'ReLU'},
    {'kind': 'conv', 'filters': 64, 'stride': 1, 'ker_size': 3},
    {'kind': 'flatten'},
    {'kind': 'dense', 'units': 512},
]

net_topology_q_vec = [
    {'kind': 'dense', 'units': 64},
    {'kind': 'ReLU'},
    {'kind': 'dense', 'units': 64},
    {'kind': 'ReLU'},
]

MLP_OPTIONS = [
    ("net_topology_pol_vec", list, net_topology_pol_vec, "Sizes of hidden layers of MLP"),
    ("net_topology_v_vec", list, net_topology_v_vec, "Sizes of hidden layers of MLP"),
    ("net_topology_q_vec", list, net_topology_q_vec, "Sizes of hidden layers of MLP"),
    ("net_topology_pol_fig", list, net_topology_pol_fig, "Sizes of hidden layers of MLP"),
    ("net_topology_v_fig", list, net_topology_v_fig, "Sizes of hidden layers of MLP"),
    ("net_topology_q_fig", list, net_topology_q_fig, "Sizes of hidden layers of MLP"),
]

PG_OPTIONS = [
    ("gamma", float, 0.995, "discount"),
    ("lam", float, 0.98, "lambda parameter from generalized advantage estimation"),
    ("load_model", bool, False, ""),
    ("save_every", int, None, ""),
    ("n_worker", int, 10, ""),
    ("update_threshold", int, 10, "")
]

ENV_OPTIONS = [
    ("ENV_NAME", str, "Swimmer-v1", ""),
    ("consec_frames", int, 4, ""),
    ("image_size", tuple, (84, 84), ""),
    ("running_stat", bool, False, ""),
    ("alpha", float, 1 / 50000, "")
]

TRPO_OPTIONS = [
    ("cg_damping", float, 1e-3, "Add multiple of the identity to Fisher matrix during CG"),
    ("cg_iters", int, 10, ""),
    ("max_kl", float, 1e-2, "KL divergence between old and new policy (averaged over state-space)"),
    ("batch_size", int, 256, ""),
    ("lr_optimizer", float, 1e-3, "learning rate"),
]

PPO_OPTIONS = [
    ("kl_target", float, 0.003, ""),
    ("batch_size", int, 200, ""),
    ("kl_cutoff_coeff", float, 50.0, ""),
    ("clip_epsilon", float, 0.2, ""),
    ("lr_optimizer", float, 1e-3, "learning rate"),
    ("lr_updater", float, 9e-4, "learning rate"),
    ("beta_upper", float, 35.0, ""),
    ("beta_lower", float, 1 / 35.0, ""),
    ("beta_adj_thres_u", float, 2.0, ""),
    ("beta_adj_thres_l", float, 0.5, "")
]

A3C_OPTIONS = [
    ("batch_size", int, 256, ""),
    ("lr_optimizer", float, 1e-3, "learning rate"),
    ("lr_updater", float, 8e-4, "learning rate"),
    ("kl_target", float, 1e-2, ""),
]

Q_OPTIONS = [
    ("alpha", float, 0.8, ""),
    ("beta", float, 0.6, ""),
    ("batch_size", int, 32, ""),
    ("memory_cap", int, 500000, ""),
    ("ini_epsilon", float, 0.2, ""),
    ("final_epsilon", float, 0.01, ""),
    ("gamma", float, 0.99, "discount"),
    ("explore_len", float, 1000000, ""),
    ("rand_explore_len", float, 5000, ""),
    ("update_target_every", int, 10000, ""),
    ("lr_optimizer", float, 1e-3, "learning rate"),
]

MOJOCO_ENVS = ["InvertedPendulum-v1", "InvertedDoublePendulum-v1", "Reacher-v1", "HalfCheetah-v1", "Swimmer-v1",
               "Hopper-v1", "Walker2d-v1", "Ant-v1", "Humanoid-v1", "HumanoidStandup-v1"]

CLASSICAL_CONTROL = ["CartPole-v0", "CartPole-v1", "Acrobot-v1", "MountainCar-v0", "MountainCarContinuous-v0",
                     "Pendulum-v0"]
