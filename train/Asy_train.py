import time

from torch import multiprocessing as mp
from torch.multiprocessing import Queue

from models.agents import *
from basic_utils.env_wrapper import Env_wrapper
from basic_utils.layers import mujoco_layer_designer


class Master:
    def __init__(self, cfg):
        self.cfg = update_default_config(PG_OPTIONS + ENV_OPTIONS + MLP_OPTIONS, cfg)
        self.callback = Callback()
        self.cfg = get_env_info(self.cfg)
        if self.cfg["use_mujoco_setting"]:
            self.cfg = mujoco_layer_designer(self.cfg)
        self.agent, self.cfg = get_agent(self.cfg)
        if self.cfg["load_model"]:
            self.agent.load_model("./save_model/" + self.cfg["ENV_NAME"] + "_" + self.cfg["agent"])

    def train(self):
        # mp.set_start_method('spawn')
        tstart = time.time()
        require_q = Queue()
        workers = []
        senders = []
        for i in range(self.cfg["n_worker"]):
            r = Queue()
            senders.append(r)
            workers.append(mp.Process(target=run, args=(self.cfg, require_q, r, i)))
            r.put(self.agent.get_params())
        for worker in workers:
            worker.start()

        while True:
            req_sofar = 0
            indexes = []
            total_paths = []
            while req_sofar < self.cfg["update_threshold"]:
                index, paths = require_q.get()
                indexes.append(index)
                total_paths += paths
                req_sofar += 1

            stats = OrderedDict()
            add_episode_stats(stats, paths)
            for u in self.agent.update(paths):
                add_prefixed_stats(stats, u[0], u[1])
            stats["TimeElapsed"] = time.time() - tstart
            counter = self.callback(stats)

            for index in indexes:
                senders[index].put(self.agent.get_params())


def run(cfg, require_q, recv_q, process_id=0):
    agent, cfg = get_agent(cfg)
    params = recv_q.get()
    agent.set_params(params)

    paths = []
    env = Env_wrapper(cfg)
    timesteps_sofar = 0
    while True:
        ob = env.reset()
        data = defaultdict(list)
        done = False
        while not done:
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

        data = {k: np.array(data[k]) for k in data}
        timesteps_sofar += pathlength(data)
        paths.append(data)

        if timesteps_sofar >= cfg["timesteps_per_batch_worker"]:
            require_q.put((process_id, paths))
            agent.set_params(recv_q.get())
            timesteps_sofar = 0
            paths = []


def train(cfg):
    tr = Master(cfg)
    tr.train()
