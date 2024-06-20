## Conditions
ConfigSpace can realize *equal*, *not equal*, *less than*, *greater than* and
*in conditions*.

Conditions can be combined by using the conjunctions *and* and *or*.
To see how to use conditions, please take a look at the [user guide](../guide.md).

For now, please refer to the individual API docs for these classes:

* [EqualsCondition][ConfigSpace.conditions.EqualsCondition]
* [NotEqualsCondition][ConfigSpace.conditions.NotEqualsCondition]
* [LessThanCondition][ConfigSpace.conditions.LessThanCondition]
* [GreaterThanCondition][ConfigSpace.conditions.GreaterThanCondition]
* [InCondition][ConfigSpace.conditions.InCondition]

To combine conditions, you can use the following classes:
* [AndConjunction][ConfigSpace.conditions.AndConjunction]
* [OrConjunction][ConfigSpace.conditions.OrConjunction]

!!! warning
 
    We advise not  using the `EqualsCondition` or the `InCondition` on float hyperparameters.
    Due to numerical rounding that can occur, it can be the case that these conditions evaluate to
    `False` even if they should evaluate to `True`.
