from basic_utils.utils import *


class ConcatFixedStd(nn.Module):
    """
    Add a fixed standard err to the input vector.
    """
    def __init__(self, ishp):
        super(ConcatFixedStd, self).__init__()
        self.log_var = nn.Parameter(torch.zeros(1, ishp) - 1.0)

    def forward(self, x):
        Mean = x
        Std = torch.exp(self.log_var * 0.5) * Variable(torch.ones(x.size()))
        return torch.cat((Mean, Std), dim=1)


class Flatten(nn.Module):
    """
    The flatten module.
    """
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size()[0], -1)


class Add_One(nn.Module):
    """
    A module whose function is to add one to the input.
    """
    def __init__(self):
        super(Add_One, self).__init__()

    def forward(self, x):
        return x+1


class Softplus(nn.Module):
    """
    The softplus module.
    """
    def __init__(self):
        super(Softplus, self).__init__()

    def forward(self, x):
        return (1 + x.exp()).log()


def get_layer(des, inshp):
    """
    Get a torch layer according to the description.

    Args:
        des: the description of the layer.
        inshp: input shape

    Return:
        layer: the corresponding torch network
        inshp: the output shape
    """
    if des['kind'] == 'conv':
        return nn.Conv2d(in_channels=inshp, out_channels=des["filters"], kernel_size=des["ker_size"],
                         stride=des["stride"]), des["filters"]
    if des['kind'] == 'flatten':
        return Flatten(), 3136
    if des["kind"] == 'dense':
        return nn.Linear(in_features=inshp, out_features=des["units"]), des["units"]
    if des['kind'] == 'ReLU':
        return nn.ReLU(), inshp
    if des['kind'] == 'Tanh':
        return nn.Tanh(), inshp
    if des['kind'] == 'Dropout':
        return nn.Dropout(p=des['p']), inshp


def mujoco_layer_designer(ob_space, ac_space):
    """
    Design the network structure for the mujoco games.

    Args:
        ob_space: the observation space
        ac_space: the action space

    Return:
        net_topology_pol_vec: structure of the policy network
        lr_updater: learning rate for the policy network
        net_topology_v_vec: structure of the value network
        lr_optimizer: learning rate for the value network
    """
    assert len(ob_space.shape) == 1

    hid1_size = ac_space.shape[0] * 10
    hid3_size = ob_space.shape[0] * 10
    hid2_size = int(np.sqrt(hid1_size*hid3_size))
    net_topology_pol_vec = [
        {'kind': 'dense', 'units': hid1_size},
        {'kind': 'Tanh'},
        {'kind': 'dense', 'units': hid2_size},
        {'kind': 'Tanh'},
        {'kind': 'dense', 'units': hid3_size},
        {'kind': 'Tanh'},
    ]
    net_topology_pol_vec = net_topology_pol_vec
    lr_updater = 9e-4 / np.sqrt(hid2_size)

    hid1_size = ac_space.shape[0] * 10
    hid3_size = 5
    hid2_size = int(np.sqrt(hid1_size * hid3_size))
    net_topology_v_vec = [
        {'kind': 'dense', 'units': hid1_size},
        {'kind': 'Tanh'},
        {'kind': 'dense', 'units': hid2_size},
        {'kind': 'Tanh'},
        {'kind': 'dense', 'units': hid3_size},
        {'kind': 'Tanh'},
    ]
    net_topology_v_vec = net_topology_v_vec
    lr_optimizer = 1e-2 / np.sqrt(hid2_size)

    return net_topology_pol_vec, lr_updater, net_topology_v_vec, lr_optimizer
