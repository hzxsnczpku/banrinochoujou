from basic_utils.layers import ConcatFixedStd, Add_One, Softplus
from basic_utils.utils import *
from models.optimizers import Target_updater


class StochPolicy:
    def __init__(self, net, probtype, updater, target_net=None, tau=0.01, update_target_every=None):
        self.net = net
        self.target_net = target_net
        if target_net is not None:
            self.target_updater = Target_updater(self.net, self.target_net, tau, update_target_every)
        self.probtype = probtype
        self.updater = updater

    def act(self, ob, target=False):
        ob = turn_into_cuda(np_to_var(ob))
        if target:
            prob = self.target_net(ob).data.cpu().numpy()
        else:
            prob = self.net(ob).data.cpu().numpy()
        return self.probtype.sample(prob)

    def update(self, *args):
        stats = self.updater(*args)
        if self.target_net is not None:
            self.target_updater.update()
        return stats

    def save_model(self, name):
        torch.save(self.net, name + "_policy.pkl")

    def load_model(self, name):
        net = torch.load(name + "_policy.pkl")
        self.net.load_state_dict(net.state_dict())
        if self.target_net is not None:
            self.target_net.load_state_dict(net.state_dict())


# ================================================================
# Abstract Class of Probtype
# ================================================================
class Probtype:
    """
    This is the abstract class of probtype.
    """
    def likelihood(self, a, prob):
        """
        Output the likelihood of an action given the parameters of the probability.

        Args:
            a: the action
            prob: the parameters of the probability

        Return:
            likelihood: the likelihood of the action
        """
        raise NotImplementedError

    def loglikelihood(self, a, prob):
        """
        Output the log likelihood of an action given the parameters of the probability.

        Args:
            a: the action
            prob: the parameters of the probability

        Return:
            log_likelihood: the log likelihood of the action
        """
        raise NotImplementedError

    def kl(self, prob0, prob1):
        """
        Output the kl divergence of two given distributions

        Args:
            prob0: the parameter of the first distribution
            prob1: the parameter of the second distribution

        Return:
            kl: the kl divergence between the two distributions
        """
        raise NotImplementedError

    def entropy(self, prob0):
        """
        Output the entropy of one given distribution

        Args:
            prob0: the parameter of the distribution

        Return:
            entropy: the entropy of the distribution
        """
        raise NotImplementedError

    def sample(self, prob):
        """
        Sample action from the given distribution.

        Args:
            prob: the parameter of the distribution

        Return:
            action: the sampled action
        """
        raise NotImplementedError

    def maxprob(self, prob):
        """
        Sample action with the maximum likelihood.

        Args:
            prob: the parameter of the distribution

        Return:
            action: the sampled action
        """
        raise NotImplementedError

    def output_layers(self, oshp):
        """
        Set the output layer needed for the distribution.

        Args:
            oshp: the input shape

        Return:
            layer: the corresponding layer
        """
        raise NotImplementedError

    def process_act(self, a):
        """
        Optional action processer.
        Args:
            a: the action to be processed

        Return:
            processed_action: the processed action
        """
        return a


class Deterministic(Probtype):
    """
    The deterministic policy type for the continuous action space, which directly determines the output point.
    """
    def __init__(self, ac_space):
        self.d = ac_space.shape[0]

    def likelihood(self, a, prob):
        pass

    def loglikelihood(self, a, prob):
        pass

    def kl(self, prob0, prob1):
        pass

    def entropy(self, prob0):
        pass

    def sample(self, prob):
        return prob

    def maxprob(self, prob):
        return prob

    def output_layers(self, oshp):
        return [nn.Linear(oshp, self.d)]


class Categorical(Probtype):
    """
    The multinomial distribution for discrete action space. It gives
     a vector representing the probability for selecting each action.
    """
    def __init__(self, ac_space):
        self.n = ac_space.n

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
    """
    The diagonal Gauss distribution for continuous action space.
    It models the distribution of the action as independent Gaussian distribution.
    """
    def __init__(self, ac_space):
        self.d = ac_space.shape[0]

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


class DiagBeta(Probtype):
    """
    The diagonal Beta distribution for continuous action space.
    It models the distribution of the action as independent Beta distribution.
    """
    def __init__(self, ac_space):
        self.d = ac_space.shape[0]
        self.scale = ac_space.high - ac_space.low
        self.center = (ac_space.high + ac_space.low)/2

    def loglikelihood(self, a, prob):
        prob = prob.view(-1, self.d, 2)
        lbeta = log_gamma(prob[:, :, 0]) + log_gamma(prob[:, :, 1]) - log_gamma(prob[:, :, 0] + prob[:, :, 1])
        lp = -lbeta + (prob[:, :, 0] - 1) * torch.log(a) + (prob[:, :, 1] - 1) * torch.log(1 - a)
        lp = lp.sum(dim=-1, keepdim=True)
        return lp

    def likelihood(self, a, prob):
        return torch.exp(self.loglikelihood(a, prob))

    def kl(self, prob0, prob1):
        prob0 = prob0.view(-1, self.d, 2)
        prob1 = prob1.view(-1, self.d, 2)
        alpha0 = prob0[:, :, 0]
        beta0 = prob0[:, :, 1]
        alpha1 = prob1[:, :, 0]
        beta1 = prob1[:, :, 1]
        lbeta0 = log_gamma(alpha0) + log_gamma(beta0) - log_gamma(alpha0 + beta0)
        lbeta1 = log_gamma(alpha1) + log_gamma(beta1) - log_gamma(alpha1 + beta1)
        kl = -lbeta0 + lbeta1 + (alpha0 - alpha1) * (digamma(alpha0) - digamma(alpha0 + beta0)
        ) + (beta0 - beta1) * (digamma(beta0) - digamma(alpha0 + beta0))
        kl = kl.sum(dim=-1)
        return kl

    def entropy(self, prob):
        prob = prob.view(-1, self.d, 2)
        alpha = prob[:, :, 0]
        beta = prob[:, :, 1]
        lbeta = log_gamma(alpha) + log_gamma(beta) - log_gamma(alpha + beta)
        ent = -lbeta + (prob[:, :, 0] - 1) * (digamma(
            prob[:, :, 0]) - digamma(prob[:, :, 0] + prob[:, :, 1])) + (prob[:, :, 1] - 1) * (digamma(
            prob[:, :, 1]) - digamma(prob[:, :, 0] + prob[:, :, 1]))
        ent = ent.sum(dim=-1)
        return ent

    def sample(self, prob):
        prob = np.reshape(prob, (-1, self.d, 2))
        alpha = prob[:, :, 0]
        beta = prob[:, :, 1]
        return np.random.beta(alpha, beta)

    def process_act(self, a):
        a = a - 0.5
        return a * self.scale + self.center

    def output_layers(self, oshp):
        return [nn.Linear(oshp, 2 * self.d), Softplus(), Add_One()]
