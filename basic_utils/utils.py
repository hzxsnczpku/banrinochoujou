from collections import defaultdict, OrderedDict
from torch import nn
import numpy as np
import torch
from tabulate import tabulate
from basic_utils.env_wrapper import Env_wrapper
import scipy.signal


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


class Callback:
    def __init__(self):
        self.counter = 0

    def __call__(self, stats):
        self.counter += 1
        # Print stats
        print("*********** Iteration %i ***********" % self.counter)
        print(tabulate(filter(lambda k: np.asarray(k[1]).size == 1, stats.items())))
        return self.counter


def pathlength(path):
    return len(path["action"])


def add_episode_stats(stats, paths):
    reward_key = "reward_raw" if "reward_raw" in paths[0] else "reward"
    episoderewards = np.array([np.sum(path[reward_key]) for path in paths])
    pathlengths = np.array([pathlength(path) for path in paths])

    stats["NumEpBatch"] = len(episoderewards)
    stats["EpRewMean"] = episoderewards.mean()
    stats["EpRewSEM"] = episoderewards.std() / np.sqrt(len(paths))
    stats["EpRewMax"] = episoderewards.max()
    stats["EpRewMin"] = episoderewards.min()
    stats["EpLenMean"] = pathlengths.mean()
    stats["EpLenMax"] = pathlengths.max()
    stats["EpLenMin"] = pathlengths.min()
    stats["RewPerStep"] = episoderewards.sum() / pathlengths.sum()

    return list(episoderewards)


def add_prefixed_stats(stats, prefix, d):
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


def derive_observation_data(path):
    observations = path["observation_raw"]
    new_data_var = np.var(observations, axis=0)
    new_data_mean = np.mean(observations, axis=0)
    new_data_mean_sq = np.square(new_data_mean)
    n = observations.shape[0]

    return {'new_data_var': new_data_var, 'new_data_mean': new_data_mean, 'new_data_mean_sq': new_data_mean_sq, 'n': n}
