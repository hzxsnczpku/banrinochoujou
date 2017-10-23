from basic_utils.utils import *


class ValueFunction:
    def __init__(self, net, optimizer, cfg):
        self.net = net
        self.optimizer = optimizer(self.net, cfg)

    def predict(self, ob_no):
        observations = np_to_var(np.array(ob_no))
        return self.net(observations).data.cpu().numpy()

    def fit(self, batch):
        return self.optimizer(batch)

    def save_model(self, name):
        torch.save(self.net, name + "_baseline.pkl")

    def load_model(self, name):
        net = torch.load(name + "_baseline.pkl")
        self.net.load_state_dict(net.state_dict())


class QValueFunction:
    def __init__(self, net, target_net, optimizer, cfg, double):
        self.net = net
        self.target_net = target_net
        self.optimizer = optimizer(self.net, self.target_net, cfg, double)

    def predict(self, ob_no):
        observations = np_to_var(np.array(ob_no))
        return self.net(observations).data.cpu().numpy()

    def act(self, ob_no):
        return np.argmax(self.predict(ob_no))

    def fit(self, paths):
        return self.optimizer(paths)

    def save_model(self, name):
        torch.save(self.net, name + "_baseline.pkl")

    def load_model(self, name):
        net = torch.load(name + "_baseline.pkl")
        self.net.load_state_dict(net.state_dict())
