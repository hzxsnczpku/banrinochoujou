from basic_utils.layers import ConcatFixedStd
from basic_utils.utils import *


class StochPolicy:
    def __init__(self, net, probtype, updater, cfg):
        self.net = net
        self.probtype = probtype
        self.updater = updater(self.net, self.probtype, cfg)

    def act(self, ob, stochastic=True):
        ob = turn_into_cuda(np_to_var(ob))
        prob = self.net(ob).data.cpu().numpy()
        if stochastic:
            return self.probtype.sample(prob), {"prob": prob[0]}
        else:
            return self.probtype.maxprob(prob), {"prob": prob[0]}

    def update(self, batch):
        return self.updater(batch)

    def save_model(self, name):
        torch.save(self.net, name + "_policy.pkl")

    def load_model(self, name):
        net = torch.load(name + "_policy.pkl")
        self.net.load_state_dict(net.state_dict())


# ================================================================
# Abstract Class of Probtype
# ================================================================
class Probtype:
    def likelihood(self, a, prob):
        raise NotImplementedError

    def loglikelihood(self, a, prob):
        raise NotImplementedError

    def kl(self, prob0, prob1):
        raise NotImplementedError

    def entropy(self, prob0):
        raise NotImplementedError

    def sample(self, prob):
        raise NotImplementedError

    def maxprob(self, prob):
        raise NotImplementedError

    def output_layers(self, oshp):
        raise NotImplementedError


class Categorical(Probtype):
    def __init__(self, n):
        self.n = n

    def likelihood(self, a, prob):
        return prob.gather(1, a.long())

    def loglikelihood(self, a, prob):
        return self.likelihood(a, prob).log()

    def kl(self, prob0, prob1):
        return (prob0 * torch.log(prob0 / prob1)).sum(dim=1)

    def entropy(self, prob0):
        return - (prob0 * prob0.log()).sum(dim=1)

    def sample(self, prob):
        assert prob.ndim == 2
        N = prob.shape[0]
        csprob_nk = np.cumsum(prob, axis=1)
        return np.argmax(csprob_nk > np.random.rand(N, 1), axis=1)

    def maxprob(self, prob):
        return prob.argmax(axis=1)

    def output_layers(self, oshp):
        return [nn.Linear(oshp, self.n), nn.Softmax()]


class DiagGauss(Probtype):
    def __init__(self, d):
        self.d = d

    def loglikelihood(self, a, prob):
        mean0 = prob[:, :self.d]
        std0 = prob[:, self.d:]
        return - 0.5 * (((a - mean0) / std0).pow(2)).sum(dim=1, keepdim=True) - 0.5 * np.log(
            2.0 * np.pi) * self.d - std0.log().sum(dim=1, keepdim=True)

    def likelihood(self, a, prob):
        return self.loglikelihood(a, prob).exp()

    def kl(self, prob0, prob1):
        mean0 = prob0[:, :self.d]
        std0 = prob0[:, self.d:]
        mean1 = prob1[:, :self.d]
        std1 = prob1[:, self.d:]
        return ((std1 / std0).log()).sum(dim=1) + (
            (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2))).sum(dim=1) - 0.5 * self.d

    def entropy(self, prob):
        std_nd = prob[:, self.d:]
        return std_nd.log().sum(dim=1) + .5 * np.log(2 * np.pi * np.e) * self.d

    def sample(self, prob):
        mean_nd = prob[:, :self.d]
        std_nd = prob[:, self.d:]
        return np.random.randn(prob.shape[0], self.d) * std_nd + mean_nd

    def maxprob(self, prob):
        return prob[:, :self.d]

    def output_layers(self, oshp):
        return [nn.Linear(oshp, self.d), ConcatFixedStd(self.d)]
