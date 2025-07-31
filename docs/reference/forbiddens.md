## Forbidden Clauses

ConfigSpace contains *forbidden equal* and *forbidden in clauses*.
The *ForbiddenEqualsClause* and the *ForbiddenInClause* can forbid values to be
sampled from a configuration space if a certain condition is met. The
*ForbiddenAndConjunction* and *ForbiddenOrConjunction* can be used to combine
*ForbiddenEqualsClauses* and the *ForbiddenInClauses*. ConfigSpace also
contains *relational clauses*, which can express relations such as less than
(equals) or greater than (equals).

For a further example, please take a look in the [user guide](../guide.md)
or the API docs below:

### Static clauses
* [ForbiddenEqualsClause][ConfigSpace.forbidden.ForbiddenEqualsClause]
* [ForbiddenInClause][ConfigSpace.forbidden.ForbiddenInClause]

### Conjunctions
* [ForbiddenAndConjunction][ConfigSpace.forbidden.ForbiddenAndConjunction]
* [ForbiddenOrConjunction][ConfigSpace.forbidden.ForbiddenOrConjunction]

### Relational Clauses
* [ForbiddenEqualsRelation][ConfigSpace.forbidden.ForbiddenEqualsRelation]
* [ForbiddenLessThenRelation][ConfigSpace.forbidden.ForbiddenLessThanRelation]
* [ForbiddenGreaterThanRelation][ConfigSpace.forbidden.ForbiddenGreaterThanRelation]
* [ForbiddenLessThanEqualsRelation][ConfigSpace.forbidden.ForbiddenLessThanEqualsRelation]
* [ForbiddenGreaterThanEqualsRelation][ConfigSpace.forbidden.ForbiddenGreaterThanEqualsRelation]
