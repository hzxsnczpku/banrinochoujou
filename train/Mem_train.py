from basic_utils.utils import *
from collections import deque
from basic_utils.options import *
from models.agents import *
from basic_utils.env_wrapper import Env_wrapper
from collections import OrderedDict
import time


class Mem_train:
    def __init__(self, cfg):
        self.cfg = update_default_config(Q_OPTIONS + ENV_OPTIONS + MLP_OPTIONS, cfg)
        self.callback = Callback()
        self.cfg = get_env_info(self.cfg)
        self.counter = 0
        self.datas = []
        self.agent, self.cfg = get_agent(self.cfg)

    def train(self):
        env = Env_wrapper(self.cfg)
        tstart = time.time()
        while True:
            ob = env.reset()
            done = False
            data = defaultdict(list)
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
