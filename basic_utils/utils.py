from collections import defaultdict, OrderedDict
from torch import nn
import numpy as np
import torch
from tabulate import tabulate
from basic_utils.env_wrapper import Env_wrapper
import scipy.signal
import time


def discount(x, gamma):
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


def update_default_config(tuples, usercfg):
    for (name, _, defval, _) in tuples:
        if name not in usercfg:
            usercfg[name] = defval
    return usercfg


def merge_before_after(info_before, info_after):
    info = OrderedDict()
    for k in info_before:
        info[k + '_before'] = info_before[k]
        info[k + '_after'] = info_after[k]
    return info


def compute_advantage(vf, paths, gamma, lam):
    for path in paths:
        rewards = path['reward'] * (1 - gamma) if gamma < 0.999 else path['reward']
        path['return'] = discount(rewards, gamma)
        values = vf.predict(path["observation"]).reshape((-1,))

        tds = rewards - values + np.append(values[1:] * gamma, 0)
        advantages = discount(tds, gamma * lam)
        path['advantage'] = advantages
    alladv = np.concatenate([path["advantage"] for path in paths])
    std = alladv.std()
    mean = alladv.mean()
    for path in paths:
        path["advantage"] = (path["advantage"] - mean) / std


def compute_target(qf, path, gamma, double=False):
    next_observations = path['next_observation']
    not_dones = path['not_done']
    rewards = path['reward'] * (1 - gamma) if gamma < 0.999 else path['reward']
    if not double:
        y_targ = qf.predict(next_observations, target=True).max(axis=1)
    else:
        ty = qf.predict(next_observations).argmax(axis=1)
        y_targ = qf.predict(next_observations, target=True)[np.arange(next_observations.shape[0]), ty]
    path['y_targ'] = y_targ * not_dones * gamma + rewards


class Callback:
    def __init__(self):
        self.counter = 0
        self.u_stats = dict()
        self.path_info = defaultdict(list)
        self.extra_info = dict()
        self.scores = []
        self.tstart = time.time()

    def print_table(self):
        self.counter += 1
        stats = OrderedDict()
        add_episode_stats(stats, self.path_info)
        for d in self.extra_info:
            stats[d] = self.extra_info[d]
        for di in self.u_stats:
            for k in self.u_stats[di]:
                self.u_stats[di][k] = np.mean(self.u_stats[di][k])
        for u in self.u_stats:
            add_prefixed_stats(stats, u, self.u_stats[u])
        stats["TimeElapsed"] = time.time() - self.tstart

        print("************ Iteration %i ************" % self.counter)
        print(tabulate(filter(lambda k: np.asarray(k[1]).size == 1, stats.items())))

        self.scores += self.path_info['episoderewards']
        self.u_stats = dict()
        self.path_info = defaultdict(list)
        self.extra_info = dict()

        return self.counter

    def num_batches(self):
        return len(self.path_info['episoderewards'])

    def add_update_info(self, u):
        if u is not None:
            for d in u:
                if d[0] not in self.u_stats:
                    self.u_stats[d[0]] = defaultdict(list)
                for k in d[1]:
                    self.u_stats[d[0]][k].append(d[1][k])

    def add_path_info(self, path_info, extra_info):
        self.path_info['episoderewards'] += [np.sum(p) for p in path_info]
        self.path_info['pathlengths'] += [len(p) for p in path_info]
        for d in extra_info:
            self.extra_info[d] = extra_info[d]


def add_episode_stats(stats, path_info):
    episoderewards = np.array(path_info['episoderewards'])
    pathlengths = np.array(path_info['pathlengths'])
    len_paths = len(episoderewards)

    stats["NumEpBatch"] = len_paths
    stats["EpRewMean"] = episoderewards.mean()
    stats["EpRewSEM"] = episoderewards.std() / np.sqrt(len_paths)
    stats["EpRewMax"] = episoderewards.max()
    stats["EpRewMin"] = episoderewards.min()
    stats["EpLenMean"] = pathlengths.mean()
    stats["EpLenMax"] = pathlengths.max()
    stats["EpLenMin"] = pathlengths.min()
    stats["RewPerStep"] = episoderewards.sum() / pathlengths.sum()

    return stats


def add_prefixed_stats(stats, prefix, d):
    if d is not None:
        for k in d:
            stats[prefix + "_" + k] = d[k]


use_cuda = torch.cuda.is_available()


def Variable(tensor, *args, **kwargs):
    if use_cuda:
        return torch.autograd.Variable(tensor, *args, **kwargs).cuda()
    else:
        return torch.autograd.Variable(tensor, *args, **kwargs)


def np_to_var(nparray):
    assert isinstance(nparray, np.ndarray)
    return torch.autograd.Variable(torch.from_numpy(nparray).float())


def pre_process_path(paths, keys):
    new_path = defaultdict(list)
    for k in keys:
        new_path[k] = np.concatenate([path[k] for path in paths])
        new_path[k] = np_to_var(np.array(new_path[k]))
        if len(new_path[k].size()) == 1:
            new_path[k] = new_path[k].view(-1, 1)
    return new_path


def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def get_env_info(cfg):
    env = Env_wrapper(cfg)
    cfg["observation_space"] = env.observation_space
    cfg["action_space"] = env.action_space
    env.close()
    return cfg


def turn_into_cuda(var):
    return var.cuda() if use_cuda else var


def log_gamma(xx):
    gamma_coeff = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ]
    magic1 = 1.000000000190015
    magic2 = 2.5066282746310005
    x = xx - 1.0
    t = x + 5.5
    t = t - (x + 0.5) * torch.log(t)
    ser = Variable(torch.ones(x.size())) * magic1
    for c in gamma_coeff:
        x = x + 1.0
        ser = ser + torch.pow(x / c, -1)
    return torch.log(ser * magic2) - t


def digamma(xx):
    gamma_coeff = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ]
    magic1 = 1.000000000190015
    x = xx - 1.0
    t = 5/(x+5.5) - torch.log(x+5.5)
    ser = Variable(torch.ones(x.size()), requires_grad=True) * magic1
    ser_p = Variable(torch.zeros(x.size()), requires_grad=True)
    for c in gamma_coeff:
        x = x + 1.0
        ser = ser + torch.pow(x / c, -1)
        ser_p = ser_p - c/(x*x)
    return ser_p/ser - t


def merge_dict(path, single_trans):
    for k in single_trans:
        path[k] += single_trans[k]
    return 1 - single_trans['not_done'][0]


def merge_train_data(u_stats):
    merged_dicts = dict()
    re = []
    for u in u_stats:
        for d in u:
            if d[0] not in merged_dicts:
                merged_dicts[d[0]] = defaultdict(list)
            for k in d[1]:
                merged_dicts[d[0]][k].append(d[1][k])
    for di in merged_dicts:
        for k in merged_dicts[di]:
            merged_dicts[di][k] = np.mean(merged_dicts[di][k])
        re.append((di, merged_dicts[di]))
    return re
