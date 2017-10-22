import time
from queue import Queue
from threading import Thread

from agents import *
from env_wrapper import Env_wrapper
from options import *
from utils import *


def rollout(index, q, r, cfg):
    timestep_limit = cfg["timestep_limit"]
    env = Env_wrapper(cfg)
    while True:
        ob = env.reset()
        data = defaultdict(list)
        score = 0
        for _ in range(timestep_limit):
            data["observation"].append(ob)
            q.put((index, ob))
            action, info = r.get()
            data["action"].append(action)
            for k in info:
                data[k].append(info[k])
            ob, rew, done, _ = env.step(action)
            score += rew
            data["reward"].append(rew)
            data["not_done"].append(1 - done)
            data["next_observation"].append(ob)
            if done:
                break
        data = {k: np.array(data[k]) for k in data}
        data["score"] = score
        q.put(('path', data))


class Trainer:
    def __init__(self, cfg):
        self.cfg = update_default_config(PG_OPTIONS + ENV_OPTIONS + MLP_OPTIONS, cfg)
        self.callback = Callback()
        self.scores = []
        env = Env_wrapper(self.cfg)
        if self.cfg["timestep_limit"] == 0:
            self.cfg["timestep_limit"] = env.timestep_limit
        if self.cfg["agent"] == "Trpo_Agent":
            self.cfg = update_default_config(TRPO_OPTIONS, cfg)
            self.agent = Trpo_Agent(env.observation_space, env.action_space, cfg)
        elif self.cfg["agent"] == "A3C_Agent":
            self.cfg = update_default_config(A3C_OPTIONS, cfg)
            self.agent = A3C_Agent(env.observation_space, env.action_space, cfg)
        elif self.cfg["agent"] == "Ppo_adapted_Agent":
            self.cfg = update_default_config(PPO_OPTIONS, cfg)
            self.agent = Ppo_adapted_Agent(env.observation_space, env.action_space, cfg)
        elif self.cfg["agent"] == "Ppo_clip_Agent":
            self.cfg = update_default_config(PPO_OPTIONS, cfg)
            self.agent = Ppo_clip_Agent(env.observation_space, env.action_space, cfg)
        if self.cfg["load_model"]:
            self.agent.load_model("./save_model/" + self.cfg["ENV_NAME"] + "_" + self.cfg["agent"])
        env.close()
        del env

    def train(self):
        tstart = time.time()
        q = Queue()
        workers = []
        senders = []
        for i in range(self.cfg["n_worker"]):
            r = Queue()
            senders.append(r)
            workers.append(Thread(target=rollout, args=(i, q, r, self.cfg)))
        for worker in workers:
            worker.start()

        paths = []
        timesteps_sofar = 0
        while True:
            tasks = []
            indexes = []
            while not q.empty():
                t = q.get()
                if t[0] == 'path':
                    paths.append(t[1])
                    timesteps_sofar += pathlength(t[1])
                    if timesteps_sofar > self.cfg["timesteps_per_batch"]:
                        self.update(paths, tstart)
                        paths = []
                        timesteps_sofar = 0
                else:
                    tasks.append(t[1])
                    indexes.append(t[0])
            if len(tasks):
                ob = np.array(tasks)
                actions, info = self.agent.act(ob)
                assert len(indexes) == len(tasks)
                for i in range(len(indexes)):
                    info_w = {}
                    for k in info:
                        info_w[k] = info[k][i, :]
                    senders[indexes[i]].put((actions[i], info_w))

    def update(self, paths, tstart):
        for path in paths:
            self.scores.append(path["score"])
        stats = OrderedDict()
        add_episode_stats(stats, paths)
        for u in self.agent.update(paths):
            add_prefixed_stats(stats, u[0], u[1])
        stats["TimeElapsed"] = time.time() - tstart
        counter = self.callback(stats)
        if self.cfg["save_every"] is not None and counter % self.cfg["save_every"] == 0:
            self.agent.save_model("./save_model/" + self.cfg["ENV_NAME"] + "_" + self.cfg["agent"])
            np.save(np.array(self.scores), "./save_model/" + self.cfg["ENV_NAME"] + "_" + self.cfg["agent"] + ".npy")


if __name__ == "__main__":
    cfg = dict(ENV_NAME="InvertedDoublePendulum-v1", agent='A3C_Agent', timestep_limit=0, n_iter=200, timesteps_per_batch=25000,
               gamma=0.99, lam=1., cg_damping=0.1, n_worker=8, save_every=100)
    tr = Trainer(cfg)
    tr.train()
