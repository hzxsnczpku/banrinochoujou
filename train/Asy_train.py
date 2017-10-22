import time
from collections import deque

from torch import multiprocessing as mp
from torch.multiprocessing import Queue

from models.agents import *
from basic_utils.env_wrapper import Env_wrapper


class Master:
    def __init__(self, cfg):
        self.cfg = update_default_config(PG_OPTIONS + ENV_OPTIONS + MLP_OPTIONS, cfg)
        self.callback = Callback()
        env = Env_wrapper(self.cfg)
        self.cfg["observation_space"] = env.observation_space
        self.cfg["action_space"] = env.action_space
        self.cfg["timesteps_per_batch_worker"] = self.cfg["timesteps_per_batch"] / self.cfg["update_threshold"]
        if self.cfg["timestep_limit"] == 0:
            self.cfg["timestep_limit"] = env.timestep_limit
        self.agent, self.cfg = get_agent(self.cfg)
        if self.cfg["load_model"]:
            self.agent.load_model("./save_model/" + self.cfg["ENV_NAME"] + "_" + self.cfg["agent"])
        env.close()
        del env

    def train(self):
        mp.set_start_method('spawn')
        tstart = time.time()
        self.agent.policy.net.share_memory()
        require_q = Queue()
        data_q = Queue()
        workers = []
        senders = []
        for i in range(self.cfg["n_worker"]):
            r = Queue()
            senders.append(r)
            workers.append(mp.Process(target=run, args=(self.cfg, require_q, data_q, r, i)))
            r.put(self.agent.get_params())
        for worker in workers:
            worker.start()

        while True:
            req_sofar = 0
            end_sofar = 0
            indexes = []
            statslist = []
            while end_sofar < self.cfg["update_threshold"]:
                while req_sofar < self.cfg["update_threshold"]:
                    index, grads = require_q.get()
                    indexes.append(index)
                    if grads is not None:
                        req_sofar += 1
                        self.agent.set_update(grads)
                    else:
                        end_sofar += 1
                self.agent.step_update()
                req_sofar = 0
                for index in indexes:
                    senders[index].put(self.agent.get_params())
                indexes = []
            while req_sofar < self.cfg["update_threshold"]:
                statslist.append(data_q.get())
                req_sofar += 1
            total_stats = merge_episode_stats(statslist)
            total_stats["TimeElapsed"] = time.time() - tstart
            self.callback(total_stats)


def run(cfg, require_q, data_q, recv_q, process_id=0):
    agent, cfg = get_agent(cfg)

    params = recv_q.get()
    agent.set_params(params)

    paths = []
    timestep_limit = cfg["timestep_limit"]
    env = Env_wrapper(cfg)
    timesteps_sofar = 0
    while True:
        ob = env.reset()
        data = defaultdict(list)
        for _ in range(timestep_limit):
            data["observation"].append(ob)
            action, info = agent.act(ob.reshape((1,) + ob.shape))
            data["action"].append(action[0])
            for k in info:
                data[k].append(info[k])
            ob, rew, done, info = env.step(action[0])
            for k in info:
                data[k].append(info[k])
            data["reward"].append(rew)
            data["not_done"].append(1 - done)
            data["next_observation"].append(ob)
            if done:
                break
        data = {k: np.array(data[k]) for k in data}
        timesteps_sofar += pathlength(data)
        paths.append(data)

        if timesteps_sofar >= cfg["timesteps_per_batch_worker"]:
            info_before = agent.get_update_info(paths)
            timesteps_sofar = 0

            for grads in agent.update(paths):
                require_q.put((process_id, grads))
                params = recv_q.get()
                agent.set_params(params)
            require_q.put((process_id, None))
            info_after = agent.get_update_info(paths)
            stats = OrderedDict()
            add_episode_stats(stats, paths)
            for u in info_before:
                add_fixed_stats(stats, u[0], "before", u[1])
            for u in info_after:
                add_fixed_stats(stats, u[0], "After", u[1])
            data_q.put(stats)
            paths = []


def train(cfg):
    tr = Master(cfg)
    tr.train()
