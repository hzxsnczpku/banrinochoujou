import numpy as np
import pylab

length = 5000
MEAN_LENGTH = length//25

NAME = ["InvertedDoublePendulum-v1", "HalfCheetah-v1", "InvertedPendulum-v1", "Swimmer-v1", "Reacher-v1", "Walker2d-v1",
        "Humanoid-v1", "HumanoidStandup-v1", "Ant-v1", "Hopper-v1", "BipedalWalker-v2"]

index = 1

distri = ["PPO_adapted", "PPO_clip", "TRPO", "A2C"]
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
        not_passed = True
        while not_passed:
            if i < MEAN_LENGTH:
                mean = np.mean(a[0:i + 1])
                std = np.std(a[0:i + 1])
            else:
                mean = np.mean(a[i - MEAN_LENGTH: i])
                std = np.std(a[i - MEAN_LENGTH: i])
            b.append(mean)
            upper.append(mean + 0.5*std)
            lower.append(mean - 0.5*std)

            not_passed = False
    b = b[:length]
    lower = lower[:length]
    upper = upper[:length]
    c += pylab.plot(b)
    pylab.fill_between(range(len(b)), lower, upper, alpha=0.3)

pylab.legend(c, distri, loc=2)
pylab.show()
