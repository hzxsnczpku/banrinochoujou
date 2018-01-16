import numpy as np
from collections import defaultdict

prefix = ['20Chain_', '30Chain_', '50Chain_', '80Chain_', '100Chain_']
suffix = ['DQN_eg', 'Bayesian_DQN_Agent', 'DQN_nonoise', 'DQN_Agent']

res = defaultdict(list)
for s in suffix:
    for p in prefix:
        name = p + s + '.npy'
        n = int(p[:-6])
        a = np.load(name)
        for i in range(len(a)):
            if a[i]==0.1:
                a[i]=1
        print(a)
        res[s].append(np.mean(a[90:110]))
print(res)