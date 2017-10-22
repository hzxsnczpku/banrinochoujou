from options import *
from utils import update_default_config, compute_advantage
from net_builder import make_policy, make_baseline, make_q_baseline
import numpy as np
from optimizers import *
from replay_memory import *


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
    options = MLP_OPTIONS + PG_OPTIONS

    def __init__(self, updater, optimizer, usercfg):
        self.cfg = update_default_config(self.options, usercfg)
        self.policy = make_policy(updater, self.cfg)
        self.baseline = make_baseline(optimizer, self.cfg)
        self.stochastic = True

    def act(self, ob_no):
        return self.policy.act(ob_no, stochastic=self.stochastic)

    def update(self, paths):
        compute_advantage(self.baseline, paths, gamma=self.cfg["gamma"], lam=self.cfg["lam"])
        for vf_stats, pol_stats in zip(self.baseline.fit(paths), self.policy.update(paths)):
            yield [pol_stats, vf_stats]

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

    def set_update(self, datas):
        self.policy.updater.set_update(datas[0])
        self.baseline.optimizer.set_update(datas[1])

    def step_update(self):
        self.policy.updater.step()
        self.baseline.optimizer.step()

    def get_update_info(self, paths):
        compute_advantage(self.baseline, paths, gamma=self.cfg["gamma"], lam=self.cfg["lam"])
        return ("pol", self.policy.updater.derive_data(paths)), ("v", self.baseline.optimizer.derive_data(paths))


# ================================================================
# Value Based Agent
# ================================================================
class Value_Based_Agent(BasicAgent):
    options = MLP_OPTIONS + Q_OPTIONS

    def __init__(self, optimizer, usercfg, double=False, priorized=True):
        self.cfg = update_default_config(self.options, usercfg)
        self.baseline = make_q_baseline(optimizer, self.cfg, double)
        self.epsilon = self.cfg["ini_epsilon"]
        self.final_epsilon = self.cfg["final_epsilon"]
        self.epsilon_decay = (self.epsilon - self.final_epsilon)/self.cfg["explore_len"]
        self.action_dim = self.cfg["action_space"].n
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

    def update(self, paths=None):
        if paths is None:
            if len(self.memory) > self.cfg["rand_explore_len"]:
                paths = self.memory.sample()
            else:
                return None
        vf_stats, info = self.baseline.fit(paths)
        self.memory.update_priorities(paths["idxes"], info["td_err"])
        vf_stats["epsilon"] = self.epsilon
        return [("q", vf_stats)]

    def save_model(self, name):
        self.baseline.save_model(name)

    def load_model(self, name):
        self.baseline.load_model(name)


def get_agent(cfg):
    if cfg["agent"] == "Trpo_Agent":
        cfg = update_default_config(TRPO_OPTIONS, cfg)
        agent = Trpo_Agent(cfg)
    elif cfg["agent"] == "A3C_Agent":
        cfg = update_default_config(A3C_OPTIONS, cfg)
        agent = A3C_Agent(cfg)
    elif cfg["agent"] == "Ppo_adapted_Agent":
        cfg = update_default_config(PPO_OPTIONS, cfg)
        agent = Ppo_adapted_Agent(cfg)
    elif cfg["agent"] == "Ppo_clip_Agent":
        cfg = update_default_config(PPO_OPTIONS, cfg)
        agent = Ppo_clip_Agent(cfg)
    return agent, cfg

# ================================================================
# Trust Region Policy Optimization
# ================================================================
class Trpo_Agent(Policy_Based_Agent):
    def __init__(self, usercfg):
        Policy_Based_Agent.__init__(self, Trpo_Updater, Adam_Optimizer, usercfg)


# ================================================================
# Asynchronous Advantage Actor-Critic
# ================================================================
class A3C_Agent(Policy_Based_Agent):
    def __init__(self, usercfg):
        Policy_Based_Agent.__init__(self, Adam_Updater, Adam_Optimizer, usercfg)


# ================================================================
# Proximal Policy Optimization
# ================================================================
class Ppo_adapted_Agent(Policy_Based_Agent):
    def __init__(self, usercfg):
        Policy_Based_Agent.__init__(self, Ppo_adapted_Updater, Adam_Optimizer, usercfg)


class Ppo_clip_Agent(Policy_Based_Agent):
    def __init__(self, usercfg):
        Policy_Based_Agent.__init__(self, Ppo_clip_Updater, Adam_Optimizer, usercfg)


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
