import ConfigSpace.hyperparameters as CSH
import numpy as np
rnd = np.random.RandomState(19937)
a = CSH.NormalIntegerHyperparameter('a', mu=10, sigma=500, lower=1, upper=2147483647, log=True)
print(a)
print(rnd)

for i in range(1, 10000):
    a.get_neighbors(0.031249126501512327, rnd, number=8, std=0.05)
