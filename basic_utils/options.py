net_topology_pol_vec = [
    {'kind': 'dense', 'units': 64},
    {'kind': 'Tanh'},
    {'kind': 'dense', 'units': 64},
    {'kind': 'Tanh'},
    {'kind': 'dense', 'units': 64},
    {'kind': 'Tanh'},
]

net_topology_v_vec = [
    {'kind': 'dense', 'units': 64},
    {'kind': 'Tanh'},
    {'kind': 'dense', 'units': 64},
    {'kind': 'Tanh'},
    {'kind': 'dense', 'units': 64},
    {'kind': 'Tanh'},
]

net_topology_q_vec = [
    {'kind': 'dense', 'units': 24},
    {'kind': 'ReLU'},
    {'kind': 'dense', 'units': 24},
    {'kind': 'ReLU'},
]

net_topology_pol_fig = [
    {'kind': 'conv', 'filters': 32, 'stride': 4, 'ker_size': 8},
    {'kind': 'Tanh'},
    {'kind': 'conv', 'filters': 64, 'stride': 2, 'ker_size': 4},
    {'kind': 'Tanh'},
    {'kind': 'conv', 'filters': 64, 'stride': 1, 'ker_size': 3},
    {'kind': 'flatten'},
    {'kind': 'dense', 'units': 512},
    {'kind': 'Tanh'},
]

net_topology_v_fig = [
    {'kind': 'conv', 'filters': 32, 'stride': 4, 'ker_size': 8},
    {'kind': 'Tanh'},
    {'kind': 'conv', 'filters': 64, 'stride': 2, 'ker_size': 4},
    {'kind': 'Tanh'},
    {'kind': 'conv', 'filters': 64, 'stride': 1, 'ker_size': 3},
    {'kind': 'flatten'},
    {'kind': 'dense', 'units': 512},
    {'kind': 'Tanh'},
]

net_topology_q_fig = [
    {'kind': 'conv', 'filters': 32, 'stride': 4, 'ker_size': 8},
    {'kind': 'Tanh'},
    {'kind': 'conv', 'filters': 64, 'stride': 2, 'ker_size': 4},
    {'kind': 'Tanh'},
    {'kind': 'conv', 'filters': 64, 'stride': 1, 'ker_size': 3},
    {'kind': 'flatten'},
    {'kind': 'dense', 'units': 512},
    {'kind': 'Tanh'},
]


MLP_OPTIONS = [
    ("net_topology_pol_vec", list, net_topology_pol_vec, "Sizes of hidden layers of MLP"),
    ("net_topology_v_vec", list, net_topology_v_vec, "Sizes of hidden layers of MLP"),
    ("net_topology_q_vec", list, net_topology_q_vec, "Sizes of hidden layers of MLP"),
    ("net_topology_pol_fig", list, net_topology_pol_fig, "Sizes of hidden layers of MLP"),
    ("net_topology_v_fig", list, net_topology_v_fig, "Sizes of hidden layers of MLP"),
    ("net_topology_q_fig", list, net_topology_q_fig, "Sizes of hidden layers of MLP"),
]

MOJOCO_ENVS = ["InvertedPendulum-v1", "InvertedDoublePendulum-v1", "Reacher-v1", "HalfCheetah-v1", "Swimmer-v1",
               "Hopper-v1", "Walker2d-v1", "Ant-v1", "Humanoid-v1", "HumanoidStandup-v1"]

CLASSICAL_CONTROL = ["CartPole-v0", "CartPole-v1", "Acrobot-v1", "MountainCar-v0", "MountainCarContinuous-v0",
                     "Pendulum-v0"]

POLICY_BASED_AGENT = ["TRPO_Agent", "A2C_Agent", "PPO_adapted_Agent", "PPO_clip_Agent"]

VALUE_BASED_AGENT = ["DQN_Agent", "Double_DQN_Agent", "Prioritized_DQN_Agent", "Prioritized_Double_DQN_Agent"]
