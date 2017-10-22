import os
used_gpu = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = used_gpu

from utils import *
from collections import deque
from options import *
from models.agents import *
from utils.env_wrapper import Env_wrapper
from collections import OrderedDict
import time


class Trainer:
    def __init__(self, cfg):
        self.cfg = update_default_config(Q_OPTIONS + ENV_OPTIONS + MLP_OPTIONS, cfg)
        self.callback = Callback()
        self.counter = 0
        self.datas = []
        env = Env_wrapper(self.cfg)
        if self.cfg["timestep_limit"] == 0:
            self.cfg["timestep_limit"] = env.timestep_limit
        if self.cfg["agent"] == "DQN_Agent":
            self.agent = DQN_Agent(env.observation_space, env.action_space, cfg)
        elif self.cfg["agent"] == "Double_DQN_Agent":
            self.agent = Double_DQN_Agent(env.observation_space, env.action_space, cfg)
        elif self.cfg["agent"] == "Priorized_DQN_Agent":
            self.agent = Prioritized_DQN_Agent(env.observation_space, env.action_space, cfg)
        elif self.cfg["agent"] == "Priorized_Double_DQN_Agent":
            self.agent = Prioritized_Double_DQN_Agent(env.observation_space, env.action_space, cfg)
        env.close()

    def train(self):
        env = Env_wrapper(self.cfg)
        tstart = time.time()
        while True:
            ob = env.reset()
            data = defaultdict(list)
            for _ in range(self.cfg["timestep_limit"]):
                action = self.agent.act(ob.reshape((1,)+ob.shape))
                data["action"].append(action)
                ob_new, rew, done, _ = env.step(action)
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
            if self.counter % 2000 == 0:
                stats = OrderedDict()
                add_episode_stats(stats, self.datas)
                for sta in u_stats:
                    add_prefixed_stats(stats, sta[0], sta[1])
                stats["Memory_length"] = len(self.agent.memory)
                stats["TimeElapsed"] = time.time() - tstart
                self.callback(stats)
                self.datas = []


if __name__ == "__main__":
    cfg = dict(ENV_NAME="CartPole-v1", agent='DQN_Agent', timestep_limit=0, n_iter=200,
               timesteps_per_batch=10000, gamma=0.99, save_every=200)
    tr = Trainer(cfg)
    tr.train()
