## Forbidden Clauses

ConfigSpace contains *forbidden equal* and *forbidden in clauses*.
The *ForbiddenEqualsClause* and the *ForbiddenInClause* can forbid values to be
sampled from a configuration space if a certain condition is met. The
*ForbiddenAndConjunction* can be used to combine *ForbiddenEqualsClauses* and
the *ForbiddenInClauses*.

For a further example, please take a look in the [user guide](../guide.md)
or the API docs below:

### Static clauses
* [ForbiddenEqualsClause][ConfigSpace.forbidden.ForbiddenEqualsClause]
* [ForbiddenInClause][ConfigSpace.forbidden.ForbiddenInClause]

### Conjunctions
* [ForbiddenAndConjunction][ConfigSpace.forbidden.ForbiddenAndConjunction]

### Relational Clauses
* [ForbiddenLessThenRelation][ConfigSpace.forbidden.ForbiddenLessThanRelation]
* [ForbiddenGreaterThanRelation][ConfigSpace.forbidden.ForbiddenGreaterThanRelation]
* [ForbiddenEqualsRelation][ConfigSpace.forbidden.ForbiddenEqualsRelation]

