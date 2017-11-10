import time

from torch import multiprocessing as mp
from torch.multiprocessing import Queue

from basic_utils.env_wrapper import Scaler
from basic_utils.layers import mujoco_layer_designer
from models.agents import *


class Asy_train:
    def __init__(self, cfg):
        self.cfg = update_default_config(MLP_OPTIONS, cfg)
        self.callback = Callback()
        self.cfg = get_env_info(self.cfg)
        self.scaler = Scaler(self.cfg["observation_space"].shape)
        self.cfg["timesteps_per_batch_worker"] = self.cfg["timesteps_per_batch"] / self.cfg["n_worker"]
        if self.cfg["use_mujoco_setting"]:
            self.cfg = mujoco_layer_designer(self.cfg)
        self.agent, self.cfg = get_agent(self.cfg)
        if self.cfg["load_model"]:
            self.agent.load_model("./save_model/" + self.cfg["ENV_NAME"] + "_" + self.cfg["agent"])

    def train(self):
        mp.set_start_method('spawn')
        tstart = time.time()
        require_q = Queue()
        workers = []
        senders = []
        scores = []
        for i in range(self.cfg["n_worker"]):
            r = Queue()
            senders.append(r)
            workers.append(mp.Process(target=run, args=(self.cfg, require_q, r, i)))
            r.put((self.agent.get_params(), self.scaler.get()))
        for worker in workers:
            worker.start()

        while True:
            req_sofar = 0
            indexes = []
            total_paths = []
            while req_sofar < self.cfg["n_worker"]:
                index, paths = require_q.get()
                if index is not None:
                    indexes.append(index)
                    req_sofar += 1
                if paths is not None:
                    for path in paths:
                        self.scaler.update(path['observation_raw'])
                    total_paths += paths

            stats = OrderedDict()
            rewards = add_episode_stats(stats, total_paths)
            scores += rewards
            for u in self.agent.update(total_paths):
                add_prefixed_stats(stats, u[0], u[1])
            stats["TimeElapsed"] = time.time() - tstart
            counter = self.callback(stats)

            for index in indexes:
                senders[index].put((self.agent.get_params(), self.scaler.get()))

            if self.cfg['save_every'] is not None and counter % self.cfg["save_every"] == 0:
                self.agent.save_model('./save_model/' + self.cfg['ENV_NAME'] + '_' + self.cfg["agent"])
                np.save('./save_score/' + self.cfg['ENV_NAME'] + '_' + self.cfg["agent"], scores)


def run(cfg, require_q, recv_q, process_id=0):
    agent, cfg = get_agent(cfg)
    params, scale = recv_q.get()
    agent.set_params(params)

    paths = []
    env = Env_wrapper(cfg)
    timesteps_sofar = 0
    while True:
        data = defaultdict(list)
        env.set_scaler(scale)
        ob, info = env.reset()
        for k in info:
            data[k].append(info[k])
        done = False
        while not done:
            if cfg['render'] and process_id==0:
                env.render()
            data["observation"].append(ob)
            action, info = agent.act(ob.reshape((1,) + ob.shape))
            data["action"].append(action[0])
            for k in info:
                data[k].append(info[k])
            ob, rew, done, info = env.step(agent.process_act(action[0]))
            for k in info:
                data[k].append(info[k])
            data["reward"].append(rew)
            data["not_done"].append(1 - done)
            data["next_observation"].append(ob)

        data = {k: np.array(data[k]) for k in data}
        timesteps_sofar += pathlength(data)
        paths.append(data)

        if (cfg['path_num'] is not None and len(paths) >= cfg['path_num']) or timesteps_sofar >= cfg["timesteps_per_batch_worker"]:
            for path in paths:
                require_q.put((None, [path]))
            require_q.put((process_id, None))
            params, scale = recv_q.get()
            agent.set_params(params)
            timesteps_sofar = 0
            paths = []
