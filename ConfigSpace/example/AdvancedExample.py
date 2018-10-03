"""
Advanced Example
================

| ConfigSpace is able to realize constraints as well as forbidden clauses in the *configuration space*.
| This is often necessary, because some hyperparameters necessitate some other hyperparameters.
| We will explain you the conditions by showing you a simple example.

| It captures the topics:

1) EqualsCondition
2) NotEqualsCondition
3) LessThanCondition
4) GreaterThanCondition
5) InCondition
6) AndConjunction
7) OrConjunction
8) ForbiddenEqualsClause
9) ForbiddenInClause
10) ForbiddenAndConjunction

+------------------------+---------------+----------+---------------------------+
| Parameter              | Type          | values   |  condition                |
+========================+===============+==========+===========================+
| a                      | categorical   | 1, 2, 3  |  None                     |
+------------------------+---------------+----------+---------------------------+
| b                      | uniform float | 1.-8.    |  a == 1                   |
+------------------------+---------------+----------+---------------------------+
| c                      | uniform float | 10-100   |  a != 2                   |
+------------------------+---------------+----------+---------------------------+
| d                      | uniform int   | 10-100   |  b < 5 AND b > 2          |
+------------------------+---------------+----------+---------------------------+
| e                      | uniform int   | 10-100   | c in {25,26,27} OR a == 2 |
+------------------------+---------------+----------+---------------------------+
| f                      | categorical   | 1, 2, 3  | Forbidden: f = g = 2      |
+------------------------+---------------+----------+---------------------------+
| g                      | categorical   | 2, 5, 6  |  None                     |
+------------------------+---------------+----------+---------------------------+

"""

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace.read_and_write import json

# First, define the hyperparameter and add them to the configuration space
cs = CS.ConfigurationSpace()
a = CSH.CategoricalHyperparameter('a', choices=[1, 2, 3])
b = CSH.UniformFloatHyperparameter('b', lower=1., upper=8., log=False)
c = CSH.UniformIntegerHyperparameter('c', lower=10, upper=100, log=False)
d = CSH.UniformIntegerHyperparameter('d', lower=10, upper=100, log=False)
e = CSH.UniformIntegerHyperparameter('e', lower=10, upper=100, log=False)
f = CSH.CategoricalHyperparameter('f', [1, 2, 3])
g = CSH.CategoricalHyperparameter('g', [2, 5, 6])

cs.add_hyperparameters([a, b, c, d, e, f, g])

# 1) EqualsCondition:
#    'b' is only active if 'a' is equal to 1
cond = CS.EqualsCondition(b, a, 1)
cs.add_condition(cond)

# 2) NotEqualsCondition:
#    'c' is only active if 'a' is not equal to 2
cond = CS.NotEqualsCondition(c, a, 2)
cs.add_condition(cond)

# 3) LessThanCondition:
#    'd' is only active if 'b' is less than 5
# We do not add this condition here directly, because we will use it later in the 'and-conjunction'.
less_cond = CS.LessThanCondition(d, b, 5)

# 4) GreaterThanCondition:
#    'd' is only active if 'b' is greater than 2
greater_cond = CS.GreaterThanCondition(d, b, 2)

# 5) InCondition:
#    'e' is only active if 'c' is in the set [25, 26, 27]
in_cond = CS.InCondition(e, c, [25, 26, 27])

# 6) AndConjunction:
#    The 'and-conjunction' combines the conditions less_cond and greater_cond
cs.add_condition(CS.AndConjunction(less_cond, greater_cond))

# 7) OrConjunction:
#    The 'or-conjunction' works similar to the 'and-conjunction'
equals_cond = CS.EqualsCondition(e, a, 2)
cs.add_condition(CS.OrConjunction(in_cond, equals_cond))

# 8) ForbiddenEqualsClause:
#    This clause forbids the value 2 for the hyperparameter f
forbidden_clause_f = CS.ForbiddenEqualsClause(f, 2)

# 9) ForbiddenInClause
#    This clause forbids the value of the hyperparameter g to be in the set [2]
forbidden_clause_g = CS.ForbiddenInClause(g, [2])

# 10) ForbiddenAndConjunction
#     Now, we combine them with an 'and-conjunction' and add them to the ConfigurationSpace
forbidden_clause = CS.ForbiddenAndConjunction(forbidden_clause_f, forbidden_clause_g)
cs.add_forbidden_clause(forbidden_clause)

# To end this example, we store the defined configuration space to a json file
with open('configspace.json', 'w') as fh:
    fh.write(json.write(cs))
