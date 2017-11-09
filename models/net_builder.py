from torch import nn
from gym.spaces import Box, Discrete
from models.policies import DiagGauss, Categorical, StochPolicy
from models.baselines import ValueFunction, QValueFunction
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


def make_policy(updater, cfg):
    ob_space = cfg["observation_space"]
    ac_space = cfg["action_space"]
    assert isinstance(ob_space, Box)
    if len(ob_space.shape) == 1:
        net_topology_pol = cfg["net_topology_pol_vec"]
    else:
        net_topology_pol = cfg["net_topology_pol_fig"]
    if isinstance(ac_space, Box):
        outdim = ac_space.shape[0]
        probtype = DiagGauss(outdim)
    elif isinstance(ac_space, Discrete):
        outdim = ac_space.n
        probtype = Categorical(outdim)
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
