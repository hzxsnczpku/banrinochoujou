from basic_utils.options import *
from basic_utils.utils import update_default_config, compute_advantage
from models.net_builder import make_policy, make_baseline, make_q_baseline
import numpy as np
from models.optimizers import *
from basic_utils.replay_memory import *


# ================================================================
# Abstract Class
# ================================================================
class BasicAgent:
    def act(self, ob_no):
        raise NotImplementedError

    def update(self, paths):
        raise NotImplementedError

    def save_model(self, name):
        raise NotImplementedError

    def load_model(self, name):
        raise NotImplementedError


# ================================================================
# Policy Based Agent
# ================================================================
class Policy_Based_Agent(BasicAgent):
    def __init__(self, updater, optimizer, usercfg):
        self.cfg = usercfg
        self.policy = make_policy(updater, self.cfg)
        self.baseline = make_baseline(optimizer, self.cfg)
        self.stochastic = True

    def act(self, ob_no):
        return self.policy.act(ob_no, stochastic=self.stochastic)

    def update(self, paths):
        compute_advantage(self.baseline, paths, gamma=self.cfg["gamma"], lam=self.cfg["lam"])
        keys = ["observation", "action", "advantage", "prob", "return"]
        processed_path = pre_process_path(paths, keys)
        pol_stats = self.policy.update(processed_path)
        vf_stats = self.baseline.fit(processed_path)
        a = [("v", vf_stats), ("pol", pol_stats)]
        return a

    def save_model(self, name):
        self.policy.save_model(name)
        self.baseline.save_model(name)

    def load_model(self, name):
        self.policy.load_model(name)
        self.baseline.load_model(name)

    def get_params(self):
        return self.policy.net.state_dict(), self.baseline.net.state_dict()

    def set_params(self, state_dicts):
        self.policy.net.load_state_dict(state_dicts[0])
        self.baseline.net.load_state_dict(state_dicts[1])


# ================================================================
# Value Based Agent
# ================================================================
class Value_Based_Agent(BasicAgent):
    def __init__(self, optimizer, usercfg, double=False, priorized=False):
        self.cfg = usercfg
        self.baseline = make_q_baseline(optimizer, usercfg)
        self.epsilon = self.cfg["ini_epsilon"]
        self.final_epsilon = self.cfg["final_epsilon"]
        self.epsilon_decay = (self.epsilon - self.final_epsilon)/self.cfg["explore_len"]
        self.action_dim = self.cfg["action_space"].n
        self.double = double
        if not priorized:
            self.memory = ReplayBuffer(self.cfg)
        else:
            self.memory = PrioritizedReplayBuffer(self.cfg)

    def act(self, ob_no):
        if np.random.rand() > self.epsilon:
            action = self.baseline.act(ob_no)
        else:
            action = np.random.randint(0, self.action_dim)
        if self.epsilon > self.final_epsilon:
            self.epsilon -= self.epsilon_decay
        return action

    def memorize(self, tuple):
        self.memory.add(tuple)

    def update(self, path=None):
        if path is None:
            if len(self.memory) > self.cfg["rand_explore_len"]:
                path = self.memory.sample()
            else:
                return None
        compute_target(self.baseline, path, gamma=self.cfg["gamma"], double=self.double)
        keys = ["observation", "action", "y_targ"]
        processed_path = pre_process_path([path], keys)
        vf_stats, info = self.baseline.fit(processed_path)
        self.memory.update_priorities(path["idxes"], info["td_err"])
        vf_stats["epsilon"] = self.epsilon
        return [("q", vf_stats)]

    def save_model(self, name):
        self.baseline.save_model(name)

    def load_model(self, name):
        self.baseline.load_model(name)


def get_agent(cfg):
    if cfg["agent"] == "TRPO_Agent":
        agent = TRPO_Agent(cfg)
    elif cfg["agent"] == "A2C_Agent":
        agent = A2C_Agent(cfg)
    elif cfg["agent"] == "PPO_adapted_Agent":
        agent = PPO_adapted_Agent(cfg)
    elif cfg["agent"] == "PPO_clip_Agent":
        agent = PPO_clip_Agent(cfg)
    elif cfg["agent"] == "DQN_Agent":
        agent = DQN_Agent(cfg)
    elif cfg["agent"] == "Double_DQN_Agent":
        agent = Double_DQN_Agent(cfg)
    elif cfg["agent"] == "Priorized_DQN_Agent":
        agent = Prioritized_DQN_Agent(cfg)
    elif cfg["agent"] == "Priorized_Double_DQN_Agent":
        agent = Prioritized_Double_DQN_Agent(cfg)
    return agent, cfg


# ================================================================
# Trust Region Policy Optimization
# ================================================================
class TRPO_Agent(Policy_Based_Agent):
    def __init__(self, usercfg):
        Policy_Based_Agent.__init__(self, TRPO_Updater, Adam_Optimizer, usercfg)


# ================================================================
# Advantage Actor-Critic
# ================================================================
class A2C_Agent(Policy_Based_Agent):
    def __init__(self, usercfg):
        Policy_Based_Agent.__init__(self, Adam_Updater, Adam_Optimizer, usercfg)


# ================================================================
# Proximal Policy Optimization
# ================================================================
class PPO_adapted_Agent(Policy_Based_Agent):
    def __init__(self, usercfg):
        Policy_Based_Agent.__init__(self, PPO_adapted_Updater, Adam_Optimizer, usercfg)


class PPO_clip_Agent(Policy_Based_Agent):
    def __init__(self, usercfg):
        Policy_Based_Agent.__init__(self, PPO_clip_Updater, Adam_Optimizer, usercfg)


# ================================================================
# Deep Q Learning
# ================================================================
class DQN_Agent(Value_Based_Agent):
    def __init__(self, usercfg):
        Value_Based_Agent.__init__(self, Adam_Q_Optimizer, usercfg)


# ================================================================
# Double Deep Q Learning
# ================================================================
class Double_DQN_Agent(Value_Based_Agent):
    def __init__(self, usercfg):
        Value_Based_Agent.__init__(self, Adam_Q_Optimizer, usercfg, double=True)


# ================================================================
# Deep Q Learning with Prioritized Experience Replay
# ================================================================
class Prioritized_DQN_Agent(Value_Based_Agent):
    def __init__(self, usercfg):
        Value_Based_Agent.__init__(self, Adam_Q_Optimizer, usercfg, priorized=True)


class Prioritized_Double_DQN_Agent(Value_Based_Agent):
    def __init__(self, usercfg):
        Value_Based_Agent.__init__(self, Adam_Q_Optimizer, usercfg, double=True, priorized=True)
