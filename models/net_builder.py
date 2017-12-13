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
