import numpy as np


class Exploration_Noise:
    def reset(self):
        """
        Reset the noise generator.
        """
        pass

    def process_action(self, a):
        """
        Add noise to the given action.

        Args:
            a: the action to be processed

        Return:
            the processed action
        """
        raise NotImplementedError


class OUNoise_Exploration:
    """
    The OU noise.
    """
    def __init__(self, action_dimension, mu=0, theta=0.15, init_epsilon=0.2, final_epsilon=0.01, explore_len=100000):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = init_epsilon
        self.final_sigma = final_epsilon
        self.sigma_decay = (init_epsilon - final_epsilon) / explore_len
        self.state = np.ones(self.action_dimension) * self.mu
        self.extra_info = ['epsilon']
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def process_action(self, a):
        processed_a = a + self.noise()
        if self.sigma > self.final_sigma:
            self.sigma -= self.sigma_decay
        return processed_a, {'epsilon': self.sigma}

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class NoNoise_Exploration:
    def __init__(self):
        self.extra_info = []

    def process_action(self, a):
        return np.argmax(a), {}

    def reset(self):
        pass


class EpsilonGreedy_Exploration:
    """
    The epsilon greedy noise.
    """
    def __init__(self, action_n, init_epsilon, final_epsilon, explore_len):
        self.epsilon = init_epsilon
        self.epsilon_decay = (init_epsilon - final_epsilon) / explore_len
        self.final_epsilon = final_epsilon
        self.extra_info = ['epsilon']
        self.n = action_n

    def process_action(self, a):
        if np.random.rand() < self.epsilon:
            new_a = np.random.randint(0, self.n)
        else:
            new_a = np.argmax(a)

        if self.epsilon > self.final_epsilon:
            self.epsilon -= self.epsilon_decay
        return new_a, {'epsilon': self.epsilon}

    def reset(self):
        pass


class Boltzmann_Exploration:
    """
    The Boltzmann noise.
    """
    def __init__(self, action_n, init_epsilon, final_epsilon, explore_len):
        self.epsilon = init_epsilon
        self.epsilon_decay = (init_epsilon - final_epsilon) / explore_len
        self.final_epsilon = final_epsilon
        self.extra_info = ['epsilon']
        self.n = action_n

    def process_action(self, a):
        probs = np.exp(a / self.epsilon)
        new_a = np.argmax(np.random.multinomial(1, probs))

        if self.epsilon > self.final_epsilon:
            self.epsilon -= self.epsilon_decay
        return new_a, {'epsilon': self.epsilon}

    def reset(self):
        pass