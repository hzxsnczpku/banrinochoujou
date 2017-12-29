from basic_utils.utils import *
from models.optimizers import Target_updater


class ValueFunction:
    def __init__(self, net, optimizer):
        self.net = net
        self.optimizer = optimizer

    def predict(self, ob):
        observations = turn_into_cuda(np_to_var(np.array(ob)))
        return self.net(observations).data.cpu().numpy()

    def fit(self, path):
        return self.optimizer(path)

    def save_model(self, name):
        torch.save(self.net, name + "_baseline.pkl")

    def load_model(self, name):
        net = torch.load(name + "_baseline.pkl")
        self.net.load_state_dict(net.state_dict())


class QValueFunction:
    def __init__(self, net, target_net, optimizer, tau=0.01, update_target_every=None):
        self.net = net
        self.target_net = target_net
        self.optimizer = optimizer
        self.target_updater = Target_updater(self.net, self.target_net, tau, update_target_every)

    def predict(self, ob_no, target=False):
        observations = turn_into_cuda(np_to_var(np.array(ob_no)))
        if not target:
            return self.net(observations).data.cpu().numpy()
        else:
            return self.target_net(observations).data.cpu().numpy()

    def act(self, ob_no):
        return self.predict(ob_no)

    def fit(self, paths):
        stat = self.optimizer(paths)
        self.target_updater.update()
        return stat

    def save_model(self, name):
        torch.save(self.net, name + "_baseline.pkl")

    def load_model(self, name):
        net = torch.load(name + "_baseline.pkl")
        self.net.load_state_dict(net.state_dict())
        self.target_net.load_state_dict(net.state_dict())


class QValueFunction_deterministic:
    def __init__(self, net, target_net, optimizer, tau=0.01, update_target_every=None):
        self.net = net
        self.target_net = target_net
        self.optimizer = optimizer
        self.target_updater = Target_updater(self.net, self.target_net, tau, update_target_every)

    def predict(self, ob_no, action, target=False):
        observations = turn_into_cuda(np_to_var(np.array(ob_no)))
        actions = turn_into_cuda(np_to_var(np.array(action)))
        if not target:
            return self.net(observations, actions).data.cpu().numpy()
        else:
            return self.target_net(observations, actions).data.cpu().numpy()

    def fit(self, paths):
        stat = self.optimizer(paths)
        self.target_updater.update()
        return stat

    def save_model(self, name):
        torch.save(self.net, name + "_baseline.pkl")

    def load_model(self, name):
        net = torch.load(name + "_baseline.pkl")
        self.net.load_state_dict(net.state_dict())
        self.target_net.load_state_dict(net.state_dict())


class QValueFunction_Bayesian:
    def __init__(self, net, mean_net, std_net, target_net, target_mean_net, target_std_net, optimizer, tau=0.01,
                 scale=1e-3, update_target_every=None):
        self.net = net
        self.mean_net = mean_net
        self.std_net = std_net
        self.target_net = target_net
        self.scale = scale
        self.target_mean_net = target_mean_net
        self.target_std_net = target_std_net
        self.optimizer = optimizer
        self.target_updater_mean = Target_updater(self.mean_net, self.target_mean_net, tau, update_target_every)
        self.target_updater_std = Target_updater(self.std_net, self.target_std_net, tau, update_target_every)

    def predict(self, ob_no, target=False):
        observations = turn_into_cuda(np_to_var(np.array(ob_no)))
        if not target:
            mean = get_flat_params_from(self.mean_net)
            std = torch.log(1 + torch.exp(get_flat_params_from(self.std_net)))
            sample_weight = mean + torch.randn(mean.size()) * std * self.scale
            set_flat_params_to(self.net, sample_weight)
            return self.net(observations).data.cpu().numpy()
        else:
            mean = get_flat_params_from(self.target_mean_net)
            std = torch.log(1 + torch.exp(get_flat_params_from(self.target_std_net)))
            sample_weight = mean + torch.randn(mean.size()) * std * self.scale
            set_flat_params_to(self.target_net, sample_weight)
            return self.target_net(observations).data.cpu().numpy()

    def act(self, ob_no):
        return self.predict(ob_no)

    def fit(self, paths):
        stat = self.optimizer(paths)
        self.target_updater_mean.update()
        self.target_updater_std.update()
        return stat

    def save_model(self, name):
        torch.save(self.net, name + "_baseline.pkl")

    def load_model(self, name):
        net = torch.load(name + "_baseline.pkl")
        self.net.load_state_dict(net.state_dict())
        self.target_net.load_state_dict(net.state_dict())
