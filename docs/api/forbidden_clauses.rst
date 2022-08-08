.. _Forbidden clauses:

Forbidden Clauses
=================

ConfigSpace contains *forbidden equal* and *forbidden in clauses*.
The *ForbiddenEqualsClause* and the *ForbiddenInClause* can forbid values to be
sampled from a configuration space if a certain condition is met. The
*ForbiddenAndConjunction* can be used to combine *ForbiddenEqualsClauses* and
the *ForbiddenInClauses*.

For a further example, please take a look in the :doc:`user guide <../guide>`.

ForbiddenEqualsClause
---------------------
.. autoclass:: ConfigSpace.ForbiddenEqualsClause(hyperparameter, value)


ForbiddenInClause
-----------------
.. autoclass:: ConfigSpace.ForbiddenInClause(hyperparameter, values)


ForbiddenAndConjunction
-----------------------
.. autoclass:: ConfigSpace.ForbiddenAndConjunction(*args)