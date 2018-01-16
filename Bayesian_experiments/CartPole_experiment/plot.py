import numpy as np
import pylab

length = 200
MEAN_LENGTH = 10

# distri = ['Bayesian Epsilon Greedy', 'Double DQN', 'Prioritized DQN']
distri = ['DQN No Exploration', 'DQN Epsilon Greedy', 'Bayesian Thompson Sampling', 'Bayesian Dropout', 'Double DQN', 'Prioritized DQN']

# pylab.title(NAME[index])
pylab.title('CartPole-v0')
pylab.xlabel("episode")
pylab.ylabel("score")
c = []

# names = ['CartPole_Bayesian_epsilon.npy']
# names = ['Acrobot-v1_Bayesian_DQN_Agent.npy', 'Acrobot-v1_DQN_Agent.npy']
names = ['DQN_no_noise.npy', 'DQN_eg.npy', 'Bayesian_DQN_ts.npy', 'Bayesian_Dropout.npy', 'Double_DQN.npy', 'Prioritized_DQN.npy']
for dis in names:
    # a = np.load(NAME[index] + "_" + dis + '_Agent' + ".npy")
    a = np.load(dis)
    b = []
    upper = []
    lower = []

    for i in range(min(a.shape[0], length)):
        fin = False
        # while not fin:
        if i < MEAN_LENGTH:
            mean = np.mean(a[0:i + 1])
            std = np.std(a[0:i + 1])
        else:
            mean = np.mean(a[i - MEAN_LENGTH: i])
            std = np.std(a[i - MEAN_LENGTH: i])
            if mean - a[i] > 3 * std:
                a[i] = mean
            else:
                fin = True
        b.append(mean)
        upper.append(mean + 0.5 * std)
        lower.append(mean - 0.5 * std)
    b = b[:length]
    lower = lower[:length]
    upper = upper[:length]
    c += pylab.plot(b)
    pylab.fill_between(range(len(b)), lower, upper, alpha=0.3)

pylab.legend(c, distri, loc=2, fontsize='x-small')
pylab.show()
