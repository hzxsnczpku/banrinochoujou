"""
Here are some default network structures as well as
some lists containing the names of different agents and environments.
"""

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
    {'kind': 'dense', 'units': 64},
    {'kind': 'Tanh'},
    {'kind': 'dense', 'units': 64},
    {'kind': 'Tanh'},
    # {'kind': 'dense', 'units': 64},
    # {'kind': 'ReLU'},
]

net_topology_q_dropout_vec = [
    {'kind': 'dense', 'units': 24},
    {'kind': 'ReLU'},
    {'kind': 'Dropout', 'p': 0.1},
    {'kind': 'dense', 'units': 24},
    {'kind': 'ReLU'},
    {'kind': 'Dropout', 'p': 0.1}
    # {'kind': 'dense', 'units': 64},
    # {'kind': 'ReLU'},
]

net_topology_pol_det_vec = [
    {'kind': 'dense', 'units': 64},
    {'kind': 'Tanh'},
]

net_topology_q_det_vec = [
    {'kind': 'dense', 'units': 64},
    {'kind': 'Tanh'},
]

net_topology_merge_det_vec = [
    {'kind': 'dense', 'units': 64},
    {'kind': 'Tanh'},
    {'kind': 'dense', 'units': 64},
    {'kind': 'Tanh'},
]

net_topology_det_vec = [net_topology_pol_det_vec, net_topology_q_det_vec, net_topology_merge_det_vec]

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
