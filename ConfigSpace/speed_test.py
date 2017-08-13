import time

import numpy as np

from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter, \
    ForbiddenEqualsClause
from ConfigSpace.io import pcs
from ConfigSpace.util import get_one_exchange_neighbourhood

#with open('/home/feurerm/projects/ConfigSpace/test/test_searchspaces/auto'
#          '-sklearn_2017_04.pcs') as fh:
#    cs = pcs.read(fh)


cs = ConfigurationSpace()
hp1 = cs.add_hyperparameter(CategoricalHyperparameter("hp1", [0, 1, 2, 3, 4, 5]))
cs.add_forbidden_clause(ForbiddenEqualsClause(hp1, 1))
cs.add_forbidden_clause(ForbiddenEqualsClause(hp1, 3))
cs.add_forbidden_clause(ForbiddenEqualsClause(hp1, 5))


times = []

for i in range(20):
    start_time = time.time()
    configs = cs.sample_configuration(500000)
    end_time = time.time()
    times.append(end_time - start_time)
print("all times:", times)
print('Sampling 500000 configurations took on average:', np.mean(times))

times = []
for config in configs[:100]:
    start_time = time.time()
    for i, n in enumerate(get_one_exchange_neighbourhood(config, 1)):
        if i == 100:
            break
    end_time = time.time()
    times.append((end_time - start_time) / 10)

print('Getting a nearest neighbor took on average:', np.mean(times))
