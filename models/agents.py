from basic_utils.options import *
import numpy as np
from models.net_builder import MLPs_pol, MLPs_v
from models.optimizers import *
from models.policies import StochPolicy
from models.baselines import *
from basic_utils.replay_memory import *
from models.data_processor import *


# ================================================================
# Abstract Class
# ================================================================
class BasicAgent:
    """
    This is the abstract class of the agent.
    """

    def act(self, ob_no):
        """
        Get the action given the observation.

        Args:
            ob_no: the observation

        Return:
            the corresponding action
        """
        raise NotImplementedError

    def update(self, paths):
        """
        Update the weights of the network.

        Args:
            paths: a dict containing the information for updating

        Return:
            information of the updating process, extra information
        """
        raise NotImplementedError

    def get_params(self):
        """
        Get the parameters of the agent.

        Return:
            the state dict of the agent
        """
        raise NotImplementedError

    def set_params(self, state_dicts):
        """
        Set the parameters to the agent.

        Args:
            state_dicts: the parameters to be set
        """
        raise NotImplementedError

    def save_model(self, name):
        """
        Save the model.
        """
        raise NotImplementedError

    def load_model(self, name):
        """
        Load the model.
        """
        raise NotImplementedError


# ================================================================
# Policy Based Agent
# ================================================================
class Policy_Based_Agent(BasicAgent):
    def __init__(self, policy, baseline):
        self.policy = policy
        self.baseline = baseline

    def act(self, observation):
        return self.policy.act(observation)

    def process_act(self, action):
        return self.policy.probtype.process_act(action)

    def update(self, processed_path):
        vf_name, vf_stats, info = self.baseline.fit(processed_path)
        pol_stats = self.policy.update(processed_path)
        return [(vf_name, vf_stats), ("pol", pol_stats)], info

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
    def __init__(self, policy, n_kid, sigma):
        self.policy = policy
        self.n_kid = n_kid
        self.sigma = sigma

    def act(self, ob_no):
        return self.policy.act(ob_no)

    def process_act(self, a):
        return self.policy.probtype.process_act(a)

    def update(self, paths):
        return [("pol", self.policy.update(paths, self.noise_seed))], {}

    def save_model(self, name):
        self.policy.save_model(name)

    def load_model(self, name):
        self.policy.load_model(name)

    def get_params(self):
        self.noise_seed = np.random.randint(0, 2 ** 32 - 1, size=self.n_kid, dtype=np.uint32).repeat(2)
        params = get_flat_params_from(self.policy.net)
        for index in range(2 * self.n_kid):
            np.random.seed(self.noise_seed[index])
            change = turn_into_cuda(
                torch.from_numpy(sign(index) * self.sigma * np.random.randn(params.numel()))).float()
            new_params = params + change
            yield new_params

    def set_params(self, flat_params):
        set_flat_params_to(self.policy.net, flat_params)


# ================================================================
# Value Based Agent
# ================================================================
class Value_Based_Agent(BasicAgent):
    def __init__(self, baseline, gamma, double=False):
        self.baseline = baseline
        self.gamma = gamma
        self.double = double

    def act(self, ob_no):
        return self.baseline.act(ob_no)

    def update(self, processed_path):
        vf_name, vf_stats, info = self.baseline.fit(processed_path)
        return [(vf_name, vf_stats)], {'td_err': info["td_err"]}

    def get_params(self):
        return [net.state_dict() for net in self.baseline.nets]

    def set_params(self, state_dict):
        for i in range(len(self.baseline.nets)):
            self.baseline.nets[i].load_state_dict(state_dict[i])

    def save_model(self, name):
        self.baseline.save_model(name)

    def load_model(self, name):
        self.baseline.load_model(name)


# ================================================================
# Trust Region Policy Optimization
# ================================================================
class TRPO_Agent(Policy_Based_Agent):
    name = 'TRPO_Agent'

    def __init__(self,
                 pol_net,
                 v_net,
                 probtype,
                 lr_optimizer=1e-3,
                 epochs_v=10,
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

        Policy_Based_Agent.__init__(self, baseline=baseline, policy=policy)


# ================================================================
# Advantage Actor-Critic
# ================================================================
class A2C_Agent(Policy_Based_Agent):
    name = 'A2C_Agent'

    def __init__(self,
                 pol_net,
                 v_net,
                 probtype,
                 epochs_v=10,
                 epochs_p=10,
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

        Policy_Based_Agent.__init__(self, baseline=baseline, policy=policy)


# ================================================================
# Proximal Policy Optimization
# ================================================================
class PPO_adapted_Agent(Policy_Based_Agent):
    name = 'PPO_adapted_Agent'

    def __init__(self,
                 pol_net,
                 v_net,
                 probtype,
                 epochs_v=10,
                 epochs_p=10,
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

        Policy_Based_Agent.__init__(self, baseline=baseline, policy=policy)


class PPO_clip_Agent(Policy_Based_Agent):
    def __init__(self,
                 pol_net,
                 v_net,
                 probtype,
                 epochs_v=10,
                 epochs_p=10,
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
        Policy_Based_Agent.__init__(self, baseline=baseline, policy=policy)


# ================================================================
# Deep Q Learning
# ================================================================
class DQN_Agent(Value_Based_Agent):
    name = 'DQN_Agent'

    def __init__(self,
                 net,
                 target_net,
                 gamma=0.99,
                 lr=1e-3,
                 update_target_every=500,
                 get_info=True):
        optimizer = Adam_Q_Optimizer(net=net,
                                     lr=lr,
                                     get_data=get_info)

        baseline = QValueFunction(net=net,
                                  target_net=target_net,
                                  optimizer=optimizer,
                                  update_target_every=update_target_every)

        Value_Based_Agent.__init__(self, baseline=baseline, gamma=gamma, double=False)


# ================================================================
# Bayesian Deep Q Learning
# ================================================================
class Bayesian_DQN_Agent(Value_Based_Agent):
    name = 'Bayesian_DQN_Agent'

    def __init__(self,
                 net,
                 mean_net,
                 std_net,
                 target_net,
                 target_mean_net,
                 target_std_net,
                 alpha=1,
                 beta=1e-4,
                 gamma=0.99,
                 lr=1e-3,
                 scale=1e-3,
                 update_target_every=500,
                 get_info=True):
        optimizer = Bayesian_Q_Optimizer(net=net,
                                         mean_net=mean_net,
                                         std_net=std_net,
                                         lr=lr,
                                         alpha=alpha,
                                         beta=beta,
                                         scale=scale,
                                         get_data=get_info)

        baseline = QValueFunction_Bayesian(net=net,
                                           mean_net=mean_net,
                                           std_net=std_net,
                                           target_net=target_net,
                                           target_mean_net=target_mean_net,
                                           target_std_net=target_std_net,
                                           optimizer=optimizer,
                                           scale=scale,
                                           tau=0.01,
                                           update_target_every=update_target_every)

        Value_Based_Agent.__init__(self, baseline=baseline, gamma=gamma, double=False)


# ================================================================
# Double Deep Q Learning
# ================================================================
class Double_DQN_Agent(Value_Based_Agent):
    name = 'Double_DQN_Agent'

    def __init__(self,
                 net,
                 target_net,
                 gamma=0.99,
                 lr=1e-3,
                 update_target_every=500,
                 get_info=True):
        optimizer = Adam_Q_Optimizer(net=net,
                                     lr=lr,
                                     get_data=get_info)

        baseline = QValueFunction(net=net,
                                  target_net=target_net,
                                  optimizer=optimizer,
                                  update_target_every=update_target_every)

        Value_Based_Agent.__init__(self, baseline=baseline, gamma=gamma, double=True)


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
        updater = Evolution_Updater(lr=lr_updater,
                                    n_kid=n_kid,
                                    net=pol_net,
                                    sigma=sigma)
        policy = StochPolicy(net=pol_net,
                             probtype=probtype,
                             updater=updater)

        self.name = 'Evolution_Agent'
        Evolution_Based_Agent.__init__(self, policy=policy, n_kid=n_kid, sigma=sigma)


# ================================================================
# Deep Deterministic Policy Gradient
# ================================================================
class DDPG_Agent(Policy_Based_Agent):
    def __init__(self,
                 policy_net,
                 policy_target_net,
                 q_net,
                 q_target_net,
                 probtype,
                 lr_updater,
                 lr_optimizer,
                 tau=0.01,
                 update_target_every=None,
                 get_info=True):
        updater = DDPG_Updater(net=policy_net,
                               lr=lr_updater,
                               q_net=q_net,
                               get_data=get_info)

        policy = StochPolicy(net=policy_net,
                             target_net=policy_target_net,
                             tau=tau,
                             update_target_every=update_target_every,
                             probtype=probtype,
                             updater=updater)

        # optimizer = Adam_Q_Optimizer(net=q_net,
        #                             lr=lr_optimizer,
        #                             get_data=get_info)

        optimizer = DDPG_Optimizer_v2(net=q_net,
                                      lr=lr_optimizer,
                                      batch_size=256,
                                      epochs=10,
                                      get_data=get_info)

        baseline = QValueFunction_deterministic(net=q_net,
                                                target_net=q_target_net,
                                                optimizer=optimizer,
                                                tau=tau,
                                                update_target_every=update_target_every)

        self.name = 'DDPG_Agent'
        Policy_Based_Agent.__init__(self, policy=policy, baseline=baseline)
