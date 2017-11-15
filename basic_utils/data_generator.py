from basic_utils.utils import *
from torch.multiprocessing import Queue
import torch.multiprocessing as mp
from models.agents import *
from basic_utils.env_wrapper import Scaler


class Whole_path_Data_generator:
    def __init__(self, cfg):
        mp.set_start_method('spawn')
        self.cfg = cfg
        self.scores = []
        self.workers = []
        self.senders = []
        self.receivers = []
        self.scaler = Scaler(self.cfg["observation_space"].shape)
        for i in range(self.cfg["n_worker"]):
            s = Queue()
            r = Queue()
            self.senders.append(s)
            self.receivers.append(r)
            self.workers.append(mp.Process(target=path_rollout, args=(self.cfg, r, s, i)))
        for worker in self.workers:
            worker.start()

    def set_param(self, params):
        for index in range(self.cfg["n_worker"]):
                self.senders[index].put((params, self.scaler.get()))

    def derive_data(self):
        paths = []
        complete_paths = []
        for i in range(self.cfg["n_worker"]):
            paths.append(defaultdict(list))
        counter = 0
        while True:
            for i in range(self.cfg["n_worker"]):
                while not self.receivers[i].empty():
                    single_trans = self.receivers[i].get()
                    if single_trans is None:
                        counter += 1
                    else:
                        done = merge_dict(paths[i], single_trans)
                        if done:
                            path = {k: np.array(paths[i][k]) for k in paths[i]}
                            self.scaler.update(path['observation_raw'])
                            complete_paths.append(path)
                            paths[i] = defaultdict(list)
            if counter == self.cfg["n_worker"]:
                stats = OrderedDict()
                self.scores += add_episode_stats(stats, complete_paths)
                return complete_paths, stats, self.scores


def path_rollout(cfg, require_q, recv_q, process_id=0):
    agent, cfg = get_agent(cfg)
    params, scale = recv_q.get()
    agent.set_params(params)

    env = Env_wrapper(cfg)
    timesteps_sofar = 0
    single_data = defaultdict(list)
    while True:
        count = 0
        env.set_scaler(scale)
        ob, info = env.reset()
        for k in info:
            single_data[k].append(info[k])
        done = False
        while not done:
            if cfg['render'] and process_id == 0:
                env.render()
            single_data["observation"].append(ob)
            action, info = agent.act(ob.reshape((1,) + ob.shape))
            single_data["action"].append(action[0])
            for k in info:
                single_data[k].append(info[k])
            ob, rew, done, info = env.step(agent.process_act(action[0]))
            for k in info:
                single_data[k].append(info[k])
            single_data["reward"].append(rew)
            single_data["not_done"].append(1 - done)
            single_data["next_observation"].append(ob)

            require_q.put(single_data)
            single_data = defaultdict(list)
            timesteps_sofar += 1
        count += 1

        if (cfg['path_num'] is not None and count >= cfg['path_num']) or timesteps_sofar >= cfg[
            "timesteps_per_batch_worker"]:
            require_q.put(None)
            params, scale = recv_q.get()
            agent.set_params(params)
            timesteps_sofar = 0


class Memory_Data_generator:
    def __init__(self, cfg, priorized=False):
        mp.set_start_method('spawn')
        self.cfg = cfg
        if not priorized:
            self.memory = ReplayBuffer(self.cfg)
        else:
            self.memory = PrioritizedReplayBuffer(self.cfg)

        self.workers = []
        self.senders = []
        self.receivers = []
        self.paths = []
        for i in range(self.cfg["n_worker"]):
            self.paths.append(defaultdict(list))
        self.complete_paths = []
        self.scaler = Scaler(self.cfg["observation_space"].shape)
        for i in range(self.cfg["n_worker"]):
            s = Queue()
            r = Queue()
            self.senders.append(s)
            self.receivers.append(r)
            self.workers.append(mp.Process(target=path_rollout, args=(self.cfg, r, s, i)))
        for worker in self.workers:
            worker.start()

    def set_param(self, params):
        for index in range(self.cfg["n_worker"]):
                self.senders[index].put((params, self.scaler.get()))

    def derive_data(self):
        for i in range(self.cfg["n_worker"]):
            while not self.receivers[i].empty():
                single_trans = self.receivers[i].get()
                done = merge_dict(self.paths[i], single_trans)
                if done:
                    path = {k: np.array(self.paths[i][k]) for k in self.paths[i]}
                    self.scaler.update(path['observation_raw'])
                    self.complete_paths.append(path)
                    self.paths[i] = defaultdict(list)


def Mem_path_rollout(cfg, require_q, recv_q, process_id=0):
    agent, cfg = get_agent(cfg)
    params, scale = recv_q.get()
    agent.set_params(params)

    env = Env_wrapper(cfg)
    timesteps_sofar = 0
    single_data = defaultdict(list)
    while True:
        count = 0
        env.set_scaler(scale)
        ob, info = env.reset()
        for k in info:
            single_data[k].append(info[k])
        done = False
        while not done:
            if cfg['render'] and process_id == 0:
                env.render()
            single_data["observation"].append(ob)
            action, info = agent.act(ob.reshape((1,) + ob.shape))
            single_data["action"].append(action[0])
            for k in info:
                single_data[k].append(info[k])
            ob, rew, done, info = env.step(agent.process_act(action[0]))
            for k in info:
                single_data[k].append(info[k])
            single_data["reward"].append(rew)
            single_data["not_done"].append(1 - done)
            single_data["next_observation"].append(ob)

            require_q.put(single_data)
            single_data = defaultdict(list)
            timesteps_sofar += 1
        count += 1

        if (cfg['path_num'] is not None and count >= cfg['path_num']) or timesteps_sofar >= cfg[
            "timesteps_per_batch_worker"]:
            require_q.put(None)
            params, scale = recv_q.get()
            agent.set_params(params)
            timesteps_sofar = 0

