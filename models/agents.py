from basic_utils.options import *
from basic_utils.utils import update_default_config, compute_advantage
from models.net_builder import make_policy, make_baseline, make_q_baseline, make_policy_deterministic, \
    make_q_baseline_deterministic
import numpy as np
from models.net_builder import MLPs_pol, MLPs_v
from models.optimizers import *
from models.policies import StochPolicy
from models.baselines import ValueFunction
from basic_utils.replay_memory import *
from models.policies import probtypes


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
    def __init__(self, policy, baseline, gamma, lam):
        self.policy = policy
        self.baseline = baseline
        self.gamma = gamma
        self.lam = lam

    def act(self, ob_no):
        return self.policy.act(ob_no)

    def process_act(self, a):
        return self.policy.probtype.process_act(a)

    def update(self, paths):
        compute_advantage(self.baseline, paths, gamma=self.gamma, lam=self.lam)
        keys = ["observation", "action", "advantage", "return"]
        processed_path = pre_process_path(paths, keys)
        pol_stats = self.policy.update(processed_path)
        vf_stats = self.baseline.fit(processed_path)
        return [("v", vf_stats), ("pol", pol_stats)], {}

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
#  Evolution Strategy Based Agent
# ================================================================
class Evolution_Based_Agent(BasicAgent):
    def __init__(self, policy, n_kid):
        self.policy = policy
        self.n_kid = n_kid

    def act(self, ob_no):
        return self.policy.act(ob_no)

    def process_act(self, a):
        return self.policy.probtype.process_act(a)

    def update(self, paths):
        pol_stats = self.policy.update(paths)
        return [("pol", pol_stats)], {}

    def save_model(self, name):
        self.policy.save_model(name)

    def load_model(self, name):
        self.policy.load_model(name)

    def get_params(self):
        return get_flat_params_from(self.policy.net)

    def set_params(self, flat_params):
        set_flat_params_to(self.policy.net, flat_params)


# ================================================================
# Deterministic Policy Based Agent
# ================================================================
class Deterministic_Policy_Based_Agent(BasicAgent):
    def __init__(self, updater, optimizer):
        self.policy, self.target_policy = make_policy_deterministic(updater)
        self.baseline = make_q_baseline_deterministic(optimizer)
        self.memory = ReplayBuffer(self.cfg)
        self.stochastic = True

    def act(self, ob_no):
        return self.policy.act(ob_no, stochastic=self.stochastic)

    def update(self, paths):
        compute_advantage(self.baseline, paths, gamma=self.cfg["gamma"], lam=self.cfg["lam"])
        keys = ["observation", "action", "advantage", "return"]
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
    def __init__(self, optimizer, usercfg, double=False):
        self.cfg = usercfg
        self.baseline = make_q_baseline(optimizer, usercfg)
        self.epsilon = self.cfg["ini_epsilon"]
        self.final_epsilon = self.cfg["final_epsilon"]
        self.epsilon_decay = (self.epsilon - self.final_epsilon) / self.cfg["explore_len"]
        self.action_dim = self.cfg["action_space"].n
        self.double = double

    def act(self, ob_no):
        if np.random.rand() > self.epsilon:
            action = self.baseline.act(ob_no)
        else:
            action = np.random.randint(0, self.action_dim)
        if self.epsilon > self.final_epsilon:
            self.epsilon -= self.epsilon_decay
        return action

    def update(self, path=None):
        if path is None:
            return None, None
        compute_target(self.baseline, path, gamma=self.cfg["gamma"], double=self.double)
        keys = ["observation", "action", "y_targ"]
        processed_path = pre_process_path([path], keys)
        if 'weights' in path:
            processed_path['weights'] = path['weights']
        vf_stats, info = self.baseline.fit(processed_path)
        return [("q", vf_stats)], {'idxes': path["idxes"], 'td_err': info["td_err"]}

    def get_params(self):
        return self.baseline.net.state_dict()

    def set_params(self, state_dict):
        self.baseline.net.load_state_dict(state_dict)

    def save_model(self, name):
        self.baseline.save_model(name)

    def load_model(self, name):
        self.baseline.load_model(name)


# ================================================================
# Trust Region Policy Optimization
# ================================================================
class TRPO_Agent(Policy_Based_Agent):
    def __init__(self,
                 pol_net,
                 v_net,
                 probtype,
                 lr_optimizer=1e-3,
                 lam=0.98,
                 epochs_v=10,
                 gamma=0.99,
                 cg_iters=10,
                 max_kl=0.003,
                 batch_size=256,
                 cg_damping=1e-3,
                 get_info=True):
        updater = TRPO_Updater(net=pol_net,
                               probtype=probtype,
                               cg_damping=cg_damping,
                               cg_iters=cg_iters,
                               max_kl=max_kl,
                               get_info=get_info)

        optimizer = Adam_Optimizer(net=v_net,
                                   batch_size=batch_size,
                                   epochs=epochs_v,
                                   lr=lr_optimizer)

        policy = StochPolicy(net=pol_net, probtype=probtype, updater=updater)
        baseline = ValueFunction(net=v_net, optimizer=optimizer)

        self.name = 'TRPO_Agent'
        Policy_Based_Agent.__init__(self, baseline=baseline, policy=policy, gamma=gamma, lam=lam)


# ================================================================
# Advantage Actor-Critic
# ================================================================
class A2C_Agent(Policy_Based_Agent):
    def __init__(self,
                 pol_net,
                 v_net,
                 probtype,
                 epochs_v=10,
                 epochs_p=10,
                 lam=0.98,
                 gamma=0.99,
                 kl_target=0.003,
                 lr_updater=9e-4,
                 lr_optimizer=1e-3,
                 batch_size=256,
                 get_info=True):
        updater = Adam_Updater(net=pol_net,
                               epochs=epochs_p,
                               kl_target=kl_target,
                               lr=lr_updater,
                               probtype=probtype,
                               get_info=get_info)

        optimizer = Adam_Optimizer(net=v_net,
                                   batch_size=batch_size,
                                   epochs=epochs_v,
                                   lr=lr_optimizer)

        policy = StochPolicy(net=pol_net, probtype=probtype, updater=updater)
        baseline = ValueFunction(net=v_net, optimizer=optimizer)

        self.name = 'A2C_Agent'
        Policy_Based_Agent.__init__(self, baseline=baseline, policy=policy, gamma=gamma, lam=lam)


# ================================================================
# Proximal Policy Optimization
# ================================================================
class PPO_adapted_Agent(Policy_Based_Agent):
    def __init__(self,
                 pol_net,
                 v_net,
                 probtype,
                 epochs_v=10,
                 epochs_p=10,
                 lam=0.98,
                 gamma=0.99,
                 kl_target=0.003,
                 lr_updater=9e-4,
                 lr_optimizer=1e-3,
                 batch_size=256,
                 adj_thres=(0.5, 2.0),
                 beta=1.0,
                 beta_range=(1 / 35.0, 35.0),
                 kl_cutoff_coeff=50.0,
                 get_info=True):
        updater = PPO_adapted_Updater(adj_thres=adj_thres,
                                      beta=beta,
                                      beta_range=beta_range,
                                      epochs=epochs_p,
                                      kl_cutoff_coeff=kl_cutoff_coeff,
                                      kl_target=kl_target,
                                      lr=lr_updater,
                                      net=pol_net,
                                      probtype=probtype,
                                      get_info=get_info)

        optimizer = Adam_Optimizer(net=v_net,
                                   batch_size=batch_size,
                                   epochs=epochs_v,
                                   lr=lr_optimizer)

        policy = StochPolicy(net=pol_net, probtype=probtype, updater=updater)
        baseline = ValueFunction(net=v_net, optimizer=optimizer)

        self.name = 'PPO_adapted_Agent'
        Policy_Based_Agent.__init__(self, baseline=baseline, policy=policy, gamma=gamma, lam=lam)


class PPO_clip_Agent(Policy_Based_Agent):
    def __init__(self,
                 pol_net,
                 v_net,
                 probtype,
                 epochs_v=10,
                 epochs_p=10,
                 lam=0.98,
                 gamma=0.99,
                 kl_target=0.003,
                 lr_updater=9e-4,
                 lr_optimizer=1e-3,
                 batch_size=256,
                 adj_thres=(0.5, 2.0),
                 clip_range=(0.05, 0.3),
                 epsilon=0.2,
                 get_info=True):
        updater = PPO_clip_Updater(adj_thres=adj_thres,
                                   clip_range=clip_range,
                                   epsilon=epsilon,
                                   epochs=epochs_p,
                                   kl_target=kl_target,
                                   lr=lr_updater,
                                   net=pol_net,
                                   probtype=probtype,
                                   get_info=get_info)

        optimizer = Adam_Optimizer(net=v_net,
                                   batch_size=batch_size,
                                   epochs=epochs_v,
                                   lr=lr_optimizer)

        policy = StochPolicy(net=pol_net, probtype=probtype, updater=updater)
        baseline = ValueFunction(net=v_net, optimizer=optimizer)

        self.name = 'PPO_clip_Agent'
        Policy_Based_Agent.__init__(self, baseline=baseline, policy=policy, gamma=gamma, lam=lam)


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
        Value_Based_Agent.__init__(self, Adam_Q_Optimizer, usercfg)


class Prioritized_Double_DQN_Agent(Value_Based_Agent):
    def __init__(self, usercfg):
        Value_Based_Agent.__init__(self, Adam_Q_Optimizer, usercfg, double=True)


# ================================================================
# Evolution Strategies
# ================================================================
class Evolution_Agent(Evolution_Based_Agent):
    def __init__(self,
                 pol_net,
                 probtype,
                 lr_updater=0.01,
                 n_kid=10,
                 sigma=0.05):
        updater = Evolution_Updater(lr=lr_updater, n_kid=n_kid, net=pol_net, sigma=sigma)
        policy = StochPolicy(net=pol_net, probtype=probtype, updater=updater)

        self.name='Evolution_Agent'
        Evolution_Based_Agent.__init__(self, policy=policy, n_kid=n_kid)
