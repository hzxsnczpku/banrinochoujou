import gym
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from collections import deque
from gym.spaces import Box


class Env_wrapper:
    def __init__(self, cfg):
        self.env = gym.make(cfg["ENV_NAME"])
        self.consec_frames = cfg["consec_frames"]
        self.image_size = cfg["image_size"]
        self.states = deque(maxlen=self.consec_frames)
        self.ob_len = len(self.env.observation_space.shape)
        if self.ob_len > 1:
            self.observation_space = Box(shape=(self.consec_frames,) + self.image_size, low=0, high=1)
        else:
            self.observation_space = Box(shape=(self.env.observation_space.shape[0] * self.consec_frames,), low=0, high=1)
        self.action_space = self.env.action_space

        self.running_stat = cfg["running_stat"]
        self.ob_running_mean = None
        self.ob_running_var = None
        self.alpha = cfg["smoothing_factor"]

    def _process(self, ob):
        processed_observe = resize(rgb2gray(ob), self.image_size, mode='constant')
        return np.reshape(processed_observe, newshape=self.image_size + (1,))

    def _normalize_ob(self, ob):
        if not self.running_stat:
            return ob
        if self.ob_running_mean is None:
            self.ob_running_mean = ob
            self.ob_running_var = np.ones_like(ob)
        else:
            self.ob_running_var = (1-self.alpha) * self.ob_running_var + self.alpha * np.square(ob - self.ob_running_mean)
            self.ob_running_mean = (1-self.alpha) * self.ob_running_mean + self.alpha * ob

        return (ob - self.ob_running_mean)/np.sqrt(self.ob_running_var)

    def reset(self):
        ob = self.env.reset()
        if self.ob_len > 1:
            ob = self._process(ob)
        ob = self._normalize_ob(ob)
        for i in range(self.consec_frames):
            self.states.append(ob)
        history = np.concatenate(self.states, axis=-1)
        if self.ob_len > 1:
            history = np.transpose(history, (2, 0, 1))
        return history

    def step(self, action):
        ob, r, done, info = self.env.step(action)
        if self.ob_len > 1:
            ob = self._process(ob)
        ob = self._normalize_ob(ob)
        self.states.append(ob)
        history = np.concatenate(self.states, axis=-1)
        if self.ob_len > 1:
            history = np.transpose(history, (2, 0, 1))
        info["reward_raw"] = r
        return history, r, done, info

    def close(self):
        self.env.close()
