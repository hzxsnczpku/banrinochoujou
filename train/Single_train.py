import time
from basic_utils.env_wrapper import Scaler
from basic_utils.layers import mujoco_layer_designer
from models.agents import *


class Sin_train:
    def __init__(self, cfg):
        self.cfg = update_default_config(MLP_OPTIONS, cfg)
        self.callback = Callback()
        self.cfg = get_env_info(self.cfg)
        self.scaler = Scaler(self.cfg["observation_space"].shape)
        if self.cfg["use_mujoco_setting"]:
            self.cfg = mujoco_layer_designer(self.cfg)
        self.agent, self.cfg = get_agent(self.cfg)
        if self.cfg["load_model"]:
            self.agent.load_model("./save_model/" + self.cfg["ENV_NAME"] + "_" + self.cfg["agent"])
        self.env = Env_wrapper(cfg)

    def run_policy(self, episodes):
        paths = []
        self.env.set_scaler(self.scaler.get())
        for e in range(episodes):
            path = defaultdict(list)
            obs, info = self.env.reset()
            for k in info:
                path[k].append(info[k])
            done = False
            while not done:
                path['observation'].append(obs)
                action, info = self.agent.act(obs.reshape((1,) + obs.shape))
                path['action'].append(action[0])
                for k in info:
                    path[k].append(info[k])
                obs, reward, done, info = self.env.step(action[0])
                for k in info:
                    path[k].append(info[k])
                path['reward'].append(reward)

            path = {k: np.array(path[k]) for k in path}
            paths.append(path)
        unscaled = np.concatenate([t['observation_raw'] for t in paths])
        self.scaler.update(unscaled)
        return paths

    def train(self):
        tstart = time.time()
        scores = []
        self.run_policy(episodes=5)
        while True:
            paths = self.run_policy(episodes=5)
            stats = OrderedDict()
            rewards = add_episode_stats(stats, paths)
            scores += rewards
            for u in self.agent.update(paths):
                add_prefixed_stats(stats, u[0], u[1])
            stats["TimeElapsed"] = time.time() - tstart
            counter = self.callback(stats)
            if self.cfg['save_every'] is not None and counter % self.cfg["save_every"] == 0:
                self.agent.save_model('./save_model/' + self.cfg['ENV_NAME'] + '_' + self.cfg["agent"])
                np.save('./save_score/' + self.cfg['ENV_NAME'] + '_' + self.cfg["agent"], scores)

