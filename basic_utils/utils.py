from collections import defaultdict, OrderedDict
from torch import nn
import numpy as np
import torch
from tabulate import tabulate
from basic_utils.env_wrapper import Env_wrapper


def discount(x, gamma, last=0):
    assert x.ndim >= 1
    y = np.zeros_like(x)
    y[-1] = x[-1] + gamma * last
    for i in range(len(x) - 2, -1, -1):
        y[i] = x[i] + gamma * y[i + 1]
    return y


def update_default_config(tuples, usercfg):
    for (name, _, defval, _) in tuples:
        if name not in usercfg:
            usercfg[name] = defval
    return usercfg


def compute_advantage(vf, paths, gamma, lam):
    # Compute return, baseline, advantage
    for path in paths:
        last = 0 if path["not_done"][-1] == 0 else vf.predict(path["next_observation"][-1])
        path["return"] = discount(path["reward"], gamma, last)
        b = path["baseline"] = vf.predict(path["observation"])
        b1 = np.append(b, last)
        deltas = path["reward"] + gamma * b1[1:] - b1[:-1]
        path["advantage"] = discount(deltas, gamma * lam)
    alladv = np.concatenate([path["advantage"] for path in paths])
    # Standardize advantage
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
    stats["EpLenMean"] = pathlengths.mean()
    stats["EpLenMax"] = pathlengths.max()
    stats["RewPerStep"] = episoderewards.sum() / pathlengths.sum()


def merge_episode_stats(statslist):
    total_stats = OrderedDict()
    num = len(statslist)
    for stats in statslist:
        for k in stats:
            if k in total_stats:
                if k == "NumEpBatch":
                    total_stats[k] += stats[k]
                elif k == "EpRewMax" or k == "EpLenMax":
                    total_stats[k] = max(total_stats[k], stats[k])
                else:
                    total_stats[k] += stats[k] / num
            else:
                if k == "NumEpBatch" or k == "EpRewMax" or k == "EpLenMax":
                    total_stats[k] = stats[k]
                else:
                    total_stats[k] = stats[k] / num
    return total_stats


def add_fixed_stats(stats, prefix, suffix, d):
    for k in d:
        stats[prefix + "_" + k + "_" + suffix] = d[k]


use_cuda = torch.cuda.is_available()


def Variable(tensor, *args, **kwargs):
    if use_cuda:
        return torch.autograd.Variable(tensor, *args, **kwargs).cuda()
    else:
        return torch.autograd.Variable(tensor, *args, **kwargs)


def Tensor(nparray):
    if use_cuda:
        return torch.from_numpy(nparray).float().cuda()
    else:
        return torch.from_numpy(nparray).float()


def np_to_var(nparray):
    assert isinstance(nparray, np.ndarray)
    return Variable(Tensor(nparray))


def np_to_var_cpu(nparray):
    assert isinstance(nparray, np.ndarray)
    return torch.autograd.Variable(torch.from_numpy(nparray).float())


def pre_process_path(paths, keys):
    new_path = defaultdict(list)
    for k in keys:
        new_path[k] = np.concatenate([path[k] for path in paths])
        new_path[k] = np_to_var_cpu(np.array(new_path[k]))
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
    if cfg["timestep_limit"] == 0:
        cfg["timestep_limit"] = env.timestep_limit
    env.close()
    return cfg


def turn_into_cuda(var):
    return var.cuda() if use_cuda else var
