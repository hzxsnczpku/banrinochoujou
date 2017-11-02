import numpy as np
import pylab

length = 100000
MEAN_LENGTH = length//25

NAME = ["InvertedDoublePendulum-v1", "HalfCheetah-v1", "InvertedPendulum-v1", "Swimmer-v1", "Reacher-v1", "Walker2d-v1",
        "Humanoid-v1", "HumanoidStandup-v1", "Ant-v1", "Hopper-v1", "BipedalWalker-v2"]

index = -3

distri = ["PPO_adapted", "PPO_clip", "TRPO"]
pylab.title(NAME[index])
pylab.xlabel("episode")
pylab.ylabel("score")
c = []

for dis in distri:
    a = np.load(NAME[index] + "_" + dis + '_Agent' + ".npy")
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
            if np.abs(a[i]-mean) > 100 * std:
                a[i] = mean
            else:
                fin = True
        b.append(mean)
        upper.append(mean + std)
        lower.append(mean - std)
    b = b[:length]
    lower = lower[:length]
    upper = upper[:length]
    c += pylab.plot(b)
    pylab.fill_between(range(len(b)), lower, upper, alpha=0.3)

pylab.legend(c, distri, loc=2)
pylab.show()
