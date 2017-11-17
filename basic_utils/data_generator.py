import torch.multiprocessing as mp
from torch.multiprocessing import Queue

from basic_utils.env_wrapper import Scaler
from models.agents import *


class Whole_path_Data_generator:
    def __init__(self, cfg):
        mp.set_start_method('spawn')
        self.cfg = cfg
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
                            del path['observation_raw']
                            complete_paths.append(path)
                            paths[i] = defaultdict(list)
            if counter == self.cfg["n_worker"]:
                path_info = [p['reward_raw'] for p in complete_paths]
                extra_info = {}
                return complete_paths, path_info, extra_info


def path_rollout(cfg, require_q, recv_q, process_id=0):
    agent = get_agent(cfg)
    params, scale = recv_q.get()
    agent.set_params(params)

    env = Env_wrapper(cfg)
    timesteps_sofar = 0
    single_data = defaultdict(list)
    count = 0
    while True:
        env.set_scaler(scale)
        ob, info = env.reset()
        for k in info:
            single_data[k].append(info[k])
        done = False
        while not done:
            if cfg['render'] and process_id == 0:
                env.render()
            single_data["observation"].append(ob)
            action = agent.act(ob.reshape((1,) + ob.shape))
            single_data["action"].append(action[0])
            ob, rew, done, info = env.step(agent.process_act(action[0]))
            for k in info:
                single_data[k].append(info[k])
            single_data["reward"].append(rew)
            single_data["not_done"].append(1 - done)

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
            count = 0


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
            self.workers.append(mp.Process(target=mem_path_rollout, args=(self.cfg, r, s, i)))
        for worker in self.workers:
            worker.start()

    def set_param(self, params):
        for index in range(self.cfg["n_worker"]):
            if self.senders[index].empty():
                self.senders[index].put((params, self.scaler.get()))

    def derive_data(self):
        for i in range(self.cfg["n_worker"]):
            while not self.receivers[i].empty():
                single_trans = self.receivers[i].get()
                self.memory.add((single_trans['observation'][0], single_trans['action'][0],
                                 single_trans['next_observation'][0], single_trans['reward'][0],
                                 single_trans['not_done'][0]))
                derived_data = {'reward_raw': single_trans['reward_raw'], 'not_done': single_trans['not_done'],
                                'observation_raw': single_trans['observation_raw'], 'epsilon': single_trans['epsilon']}
                done = merge_dict(self.paths[i], derived_data)

                if done:
                    path = {k: np.array(self.paths[i][k]) for k in self.paths[i]}
                    self.scaler.update(path['observation_raw'])
                    self.complete_paths.append(path)
                    self.paths[i] = defaultdict(list)

        sample_return = self.memory.sample() if len(self.memory) > self.cfg["rand_explore_len"] else None
        path_info = [p['reward_raw'] for p in self.complete_paths]
        extra_info = {"Memory_length": len(self.memory), 'Epsilon': np.mean([k['epsilon'][-1] for k in self.complete_paths])}
        self.complete_paths = []
        return sample_return, path_info, extra_info


def mem_path_rollout(cfg, require_q, recv_q, process_id=0):
    agent = get_agent(cfg)
    params, scale = recv_q.get()
    agent.set_params(params)

    env = Env_wrapper(cfg)
    single_data = defaultdict(list)
    while True:
        env.set_scaler(scale)
        ob, info = env.reset()
        for k in info:
            single_data[k].append(info[k])
        done = False
        while not done:
            if cfg['render'] and process_id == 0:
                env.render()
            single_data["observation"].append(ob)
            action = agent.act(ob.reshape((1,) + ob.shape))
            single_data["action"].append(action)
            ob, rew, done, info = env.step(action)
            for k in info:
                single_data[k].append(info[k])
            single_data["reward"].append(rew)
            single_data["not_done"].append(1 - done)
            single_data["next_observation"].append(ob)
            single_data['epsilon'].append(agent.epsilon)

            require_q.put(single_data)

            single_data = defaultdict(list)
            params, scale = recv_q.get()
            agent.set_params(params)


def get_generator(cfg):
    if cfg['agent'] in POLICY_BASED_AGENT:
        return Whole_path_Data_generator(cfg)
    elif cfg['agent'] in VALUE_BASED_AGENT:
        return Memory_Data_generator(cfg)
