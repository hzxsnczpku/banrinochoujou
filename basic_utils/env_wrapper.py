import gym
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from collections import deque
from gym.spaces import Box


class Scaler(object):
    def __init__(self, obs_dim):
        self.vars = np.zeros(obs_dim)
        self.means = np.zeros(obs_dim)
        self.m = 0
        self.first_pass = True

    def update(self, x):
        if self.first_pass:
            self.means = np.mean(x, axis=0)
            self.vars = np.var(x, axis=0)
            self.m = x.shape[0]
            self.first_pass = False
        else:
            n = x.shape[0]
            new_data_var = np.var(x, axis=0)
            new_data_mean = np.mean(x, axis=0)
            new_data_mean_sq = np.square(new_data_mean)
            new_means = ((self.means * self.m) + (new_data_mean * n)) / (self.m + n)
            self.vars = (((self.m * (self.vars + np.square(self.means))) +
                          (n * (new_data_var + new_data_mean_sq))) / (self.m + n) -
                         np.square(new_means))
            self.vars = np.maximum(0.0, self.vars)
            self.means = new_means
            self.m += n

    def get(self):
        return 1/(np.sqrt(self.vars) + 0.1)/3, self.means


class Vec_env_wrapper:
    def __init__(self, name, consec_frames, running_stat):
        self.env = gym.make(name)
        self.name = name
        self.consec_frames = consec_frames
        self.states = deque(maxlen=self.consec_frames)
        self.observation_space = Box(shape=(self.env.observation_space.shape[0] * self.consec_frames + 1,), low=0, high=1)
        self.observation_space_sca = Box(shape=(self.env.observation_space.shape[0] * self.consec_frames,), low=0,
                                     high=1)
        self.action_space = self.env.action_space

        self.timestep = 0
        self.running_stat = running_stat
        self.offset = None
        self.scale = None

    def set_scaler(self, scales):
        self.offset = scales[1]
        self.scale = scales[0]

    def _normalize_ob(self, ob):
        if not self.running_stat:
            return ob
        return (ob - self.offset) * self.scale

    def reset(self):
        ob = self.env.reset()
        self.timestep = 0
        for i in range(self.consec_frames):
            self.states.append(ob)
        history = np.concatenate(self.states, axis=-1)
        history_normalized = self._normalize_ob(history)
        history_normalized = np.append(history_normalized, [self.timestep])

        info = {"observation_raw": history}
        return history_normalized, info

    def step(self, action):
        ob, r, done, info = self.env.step(action)
        self.timestep += 1e-3
        self.states.append(ob)
        history = np.concatenate(self.states, axis=-1)
        history_normalized = self._normalize_ob(history)
        history_normalized = np.append(history_normalized, [self.timestep])
        info["reward_raw"] = r
        info["observation_raw"] = history
        return history_normalized, r, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


class Fig_env_wrapper:
    def __init__(self, cfg):
        self.env = gym.make(cfg["ENV_NAME"])
        self.consec_frames = cfg["consec_frames"]
        self.image_size = cfg["image_size"]
        self.states = deque(maxlen=self.consec_frames)
        self.observation_space = Box(shape=(self.consec_frames,) + self.image_size, low=0, high=1)
        self.observation_space_sca = Box(shape=(self.consec_frames,) + self.image_size, low=0, high=1)
        self.action_space = self.env.action_space

        self.running_stat = cfg["running_stat"]
        self.offset = None
        self.scale = None

    def set_scaler(self, scales):
        self.offset = scales[1]
        self.scale = scales[0]

    def _process(self, ob):
        processed_observe = resize(rgb2gray(ob), self.image_size, mode='constant')
        return np.reshape(processed_observe, newshape=self.image_size + (1,))

    def _normalize_ob(self, ob):
        if not self.running_stat:
            return ob
        return (ob - self.offset) * self.scale

    def reset(self):
        ob = self.env.reset()
        self.timestep = 0
        ob = self._process(ob)
        for i in range(self.consec_frames):
            self.states.append(ob)
        history = np.concatenate(self.states, axis=-1)
        history = np.transpose(history, (2, 0, 1))

        history_normalized = self._normalize_ob(history)

        info = {"observation_raw": history}
        return history_normalized, info

    def step(self, action):
        ob, r, done, info = self.env.step(action)
        ob = self._process(ob)
        self.states.append(ob)
        history = np.concatenate(self.states, axis=-1)
        history = np.transpose(history, (2, 0, 1))
        history_normalized = self._normalize_ob(history)
        info["reward_raw"] = r
        info["observation_raw"] = history
        return history_normalized, r, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
