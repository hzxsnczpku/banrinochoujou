from torch import nn
from gym.spaces import Box, Discrete
from models.policies import DiagGauss, Categorical, StochPolicy, DiagBeta, Deterministic
from models.baselines import ValueFunction, QValueFunction, QValueFunction_deterministic
from basic_utils.utils import *
from basic_utils.layers import *


class MLPs_pol(nn.Module):
    def __init__(self, ob_space, net_topology, output_layers):
        super(MLPs_pol, self).__init__()
        self.layers = nn.ModuleList([])
        inshp = ob_space.shape[0]
        for des in net_topology:
            l, inshp = get_layer(des, inshp)
            self.layers.append(l)
        self.layers += output_layers(inshp)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class MLPs_v(nn.Module):
    def __init__(self, ob_space, net_topology):
        super(MLPs_v, self).__init__()
        self.layers = nn.ModuleList([])
        inshp = ob_space.shape[0]
        for (i, des) in enumerate(net_topology):
            l, inshp = get_layer(des, inshp)
            self.layers.append(l)
        self.layers.append(nn.Linear(inshp, 1))

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class MLPs_q(nn.Module):
    def __init__(self, ob_space, ac_apace, net_topology):
        super(MLPs_q, self).__init__()
        self.layers = nn.ModuleList([])
        inshp = ob_space.shape[0]
        for (i, des) in enumerate(net_topology):
            l, inshp = get_layer(des, inshp)
            self.layers.append(l)
        self.layers.append(nn.Linear(inshp, ac_apace.n))

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class MLPs_q_deterministic(nn.Module):
    def __init__(self, ob_space, ac_space, net_topology):
        super(MLPs_q_deterministic, self).__init__()
        self.ac_layers = nn.ModuleList([])
        self.ob_layers = nn.ModuleList([])
        self.merge_layers = nn.ModuleList([])
        inshp_a = ac_space.shape[0]
        for (i, des) in enumerate(net_topology[0]):
            l, inshp_a = get_layer(des, inshp_a)
            self.ac_layers.append(l)

        inshp_o = ob_space.shape[0]
        for (i, des) in enumerate(net_topology[1]):
            l, inshp_o = get_layer(des, inshp_o)
            self.ob_layers.append(l)

        inshp_m = inshp_a + inshp_o
        for (i, des) in enumerate(net_topology[2]):
            l, inshp_m = get_layer(des, inshp_m)
            self.merge_layers.append(l)
        self.merge_layers.append(nn.Linear(inshp_m, 1))

    def forward(self, a, x):
        for l in self.ac_layers:
            a = l(a)
        for l in self.ob_layers:
            x = l(x)
        m = torch.cat([a, x], 1)
        for l in self.ob_layers:
            m = l(m)
        return m


def make_policy(updater, cfg):
    ob_space = cfg["observation_space"]
    ac_space = cfg["action_space"]
    assert isinstance(ob_space, Box)
    if len(ob_space.shape) == 1:
        net_topology_pol = cfg["net_topology_pol_vec"]
    else:
        net_topology_pol = cfg["net_topology_pol_fig"]
    if isinstance(ac_space, Box):
        if cfg['dist'] == 'DiagGauss':
            probtype = DiagGauss(ac_space)
        elif cfg['dist'] == 'DiagBeta':
            probtype = DiagBeta(ac_space)
    elif isinstance(ac_space, Discrete):
        probtype = Categorical(ac_space)
    net = MLPs_pol(ob_space, net_topology_pol, probtype.output_layers)
    if use_cuda:
        net.cuda()
    policy = StochPolicy(net, probtype, updater, cfg)
    return policy


def make_baseline(optimizer, cfg):
    ob_space = cfg["observation_space"]
    assert isinstance(ob_space, Box)
    if len(ob_space.shape) == 1:
        net_topology_v = cfg["net_topology_v_vec"]
    else:
        net_topology_v = cfg["net_topology_v_fig"]
    net = MLPs_v(ob_space, net_topology_v)
    if use_cuda:
        net.cuda()
    baseline = ValueFunction(net, optimizer, cfg)
    return baseline


def make_q_baseline(optimizer, cfg):
    ob_space = cfg["observation_space"]
    ac_space = cfg["action_space"]
    assert isinstance(ob_space, Box)
    assert isinstance(ac_space, Discrete)
    if len(ob_space.shape) == 1:
        net_topology_q = cfg["net_topology_q_vec"]
    else:
        net_topology_q = cfg["net_topology_q_fig"]
    net = MLPs_q(ob_space, ac_space, net_topology_q)
    net_target = MLPs_q(ob_space, ac_space, net_topology_q)
    net_target.load_state_dict(net.state_dict())
    if use_cuda:
        net.cuda()
        net_target.cuda()
    baseline = QValueFunction(net, net_target, optimizer, cfg)
    return baseline


def make_policy_deterministic(updater, cfg):
    ob_space = cfg["observation_space"]
    ac_space = cfg["action_space"]
    assert isinstance(ob_space, Box)
    if len(ob_space.shape) == 1:
        net_topology_pol = cfg["net_topology_pol_vec"]
    else:
        net_topology_pol = cfg["net_topology_pol_fig"]
    probtype = Deterministic(ac_space)
    net = MLPs_pol(ob_space, net_topology_pol, probtype.output_layers)
    net_target = MLPs_pol(ob_space, net_topology_pol, probtype.output_layers)
    net_target.load_state_dict(net.state_dict())
    if use_cuda:
        net.cuda()
        net_target.cuda()
    policy = StochPolicy(net, probtype, updater, cfg)
    policy_target = StochPolicy(net, probtype, updater, cfg)
    return policy, policy_target


def make_q_baseline_deterministic(optimizer, cfg):
    ob_space = cfg["observation_space"]
    ac_space = cfg["action_space"]
    assert isinstance(ob_space, Box)
    assert isinstance(ac_space, Box)
    if len(ob_space.shape) == 1:
        net_topology_q = cfg["net_topology_det_vec"]
    else:
        net_topology_q = cfg["net_topology_det_fig"]
    net = MLPs_q_deterministic(ob_space, ac_space, net_topology_q)
    net_target = MLPs_q_deterministic(ob_space, ac_space, net_topology_q)
    net_target.load_state_dict(net.state_dict())
    if use_cuda:
        net.cuda()
        net_target.cuda()
    baseline = QValueFunction_deterministic(net, net_target, optimizer, cfg)
    return baseline
