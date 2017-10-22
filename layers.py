from utils import *


class ConcatFixedStd(nn.Module):
    def __init__(self, ishp):
        super(ConcatFixedStd, self).__init__()
        self.log_std = nn.Parameter(torch.zeros(1, ishp)-1.0)

    def forward(self, x):
        Mean = x
        Std = self.log_std.exp() * Variable(torch.ones(x.size()))
        return torch.cat((Mean, Std), dim=1)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size()[0], -1)


def get_layer(des, inshp):
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


def mujoco_layer_designer(ob_space, ac_space, cfg):
    assert len(ob_space.shape)==1
    hid1_size = ac_space.shape[0] * 10
    hid3_size = ob_space.shape[0] * 10
    hid2_size = int(np.sqrt(hid1_size*hid3_size))
    net_topology_pol_vec = [
        {'kind': 'dense', 'units': hid1_size},
        {'kind': 'Tanh'},
        {'kind': 'dense', 'units': hid2_size},
        {'kind': 'Tanh'},
        {'kind': 'dense', 'units': hid3_size},
    ]
    cfg["net_topology_pol_vec"] = net_topology_pol_vec

    hid1_size = ac_space.shape[0] * 10
    hid3_size = 5
    hid2_size = int(np.sqrt(hid1_size * hid3_size))
    net_topology_v_vec = [
        {'kind': 'dense', 'units': hid1_size},
        {'kind': 'Tanh'},
        {'kind': 'dense', 'units': hid2_size},
        {'kind': 'Tanh'},
        {'kind': 'dense', 'units': hid3_size},
    ]
    cfg["net_topology_v_vec"] = net_topology_v_vec