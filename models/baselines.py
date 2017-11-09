from basic_utils.utils import *


class ValueFunction:
    def __init__(self, net, optimizer, cfg):
        self.net = net
        self.optimizer = optimizer(self.net, cfg)

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
    def __init__(self, net, target_net, optimizer, cfg):
        self.net = net
        self.target_net = target_net
        self.optimizer = optimizer(self.net, self.target_net, cfg)

    def predict(self, ob_no, target=False):
        observations = turn_into_cuda(np_to_var(np.array(ob_no)))
        if not target:
            return self.net(observations).data.cpu().numpy()
        else:
            return self.target_net(observations).data.cpu().numpy()

    def act(self, ob_no):
        return np.argmax(self.predict(ob_no))

    def fit(self, paths):
        return self.optimizer(paths)

    def save_model(self, name):
        torch.save(self.net, name + "_baseline.pkl")

    def load_model(self, name):
        net = torch.load(name + "_baseline.pkl")
        self.net.load_state_dict(net.state_dict())
