from basic_utils.utils import *
from collections import deque
from basic_utils.options import *
from models.agents import *
from basic_utils.env_wrapper import Env_wrapper, Scaler
from collections import OrderedDict
import time


class Mem_train:
    def __init__(self, cfg):
        self.cfg = update_default_config(MLP_OPTIONS, cfg)
        self.callback = Callback()
        self.cfg = get_env_info(self.cfg)
        self.counter = 0
        self.datas = []
        self.agent, self.cfg = get_agent(self.cfg)
        self.scaler = Scaler(self.cfg["observation_space"].shape)

    def train(self):
        env = Env_wrapper(self.cfg)
        tstart = time.time()
        while True:
            data = defaultdict(list)
            env.set_scaler(self.scaler.get())
            ob, info = env.reset()
            for k in info:
                data[k].append(info[k])
            done = False
            while not done:
                action = self.agent.act(ob.reshape((1,)+ob.shape))
                data["action"].append(action)
                ob_new, rew, done, info = env.step(action)
                for k in info:
                    data[k].append(info[k])
                data["reward"].append(rew)
                self.agent.memorize((ob, action, ob_new, rew, 1-done))
                ob = ob_new
                self.counter += 1
                self.update(tstart)
                if done:
                    data = {k: np.array(data[k]) for k in data}
                    self.datas.append(data)
                    break

    def update(self, tstart):
        u_stats = self.agent.update()
        if u_stats is not None:
            if len(self.datas) >= 10:
                stats = OrderedDict()
                add_episode_stats(stats, self.datas)
                for sta in u_stats:
                    add_prefixed_stats(stats, sta[0], sta[1])
                stats["Memory_length"] = len(self.agent.memory)
                stats["TimeElapsed"] = time.time() - tstart
                self.callback(stats)
                self.datas = []
