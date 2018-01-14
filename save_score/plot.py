import numpy as np
import pylab

length = 150
MEAN_LENGTH = length // 10

NAME = ["InvertedDoublePendulum-v1", "HalfCheetah-v1", "InvertedPendulum-v1", "Swimmer-v1", "Reacher-v1", "Walker2d-v1",
        "Humanoid-v1", "HumanoidStandup-v1", "Ant-v1", "Hopper-v1", "BipedalWalker-v2"]

index = -5

# distri = ["PPO_adapted", "PPO_clip", "TRPO", "A2C"]
# distri = ['DQN No Exploration', 'DQN Epsilon Greedy', 'Bayesian Thompson Sampling', 'Bayesian Epsilon Greedy', 'Double DQN',
#            'Prioritized DQN']
distri = ['DQN Epsilon Greedy', 'Bayesian Thompson Sampling']

# pylab.title(NAME[index])
pylab.title('CartPole-v1')
pylab.xlabel("episode")
pylab.ylabel("score")
c = []

# names = ['CartPole_DQN_Baseline.npy', 'CartPole_DQN_epsilon.npy', 'CartPole_Bayesian_TS.npy',
#          'CartPole_Bayesian_epsilon.npy', 'CartPole_DDQN.npy', 'CartPole_PDQN.npy']
# names = ['Acrobot-v1_Bayesian_DQN_Agent.npy', 'Acrobot-v1_DQN_Agent.npy']
names = ['CartPole-v1_DQN_Agent.npy', 'CartPole-v1_Bayesian_DQN_Agent.npy']
for dis in names:
    # a = np.load(NAME[index] + "_" + dis + '_Agent' + ".npy")
    a = np.load(dis)
    b = []
    upper = []
    lower = []

    for i in range(min(a.shape[0], length)):
        fin = False
        while not fin:
            if i < MEAN_LENGTH:
                mean = np.mean(a[0:i + 1])
                std = np.std(a[0:i + 1])
            else:
                mean = np.mean(a[i - MEAN_LENGTH: i])
                std = np.std(a[i - MEAN_LENGTH: i])
            if np.abs(a[i] - mean) > 100 * std:
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
    # pylab.fill_between(range(len(b)), lower, upper, alpha=0.3)

pylab.legend(c, distri, loc=2, fontsize='x-small')
pylab.show()
