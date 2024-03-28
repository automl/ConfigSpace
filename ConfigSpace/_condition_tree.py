from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator
from typing_extensions import Self

import numpy as np
from more_itertools import unique_everseen

from ConfigSpace import nx
from ConfigSpace.conditions import Condition, Conjunction
from ConfigSpace.exceptions import (
    AmbiguousConditionError,
    ChildNotFoundError,
    CyclicDependancyError,
    HyperparameterAlreadyExistsError,
    HyperparameterNotFoundError,
    ParentNotFoundError,
)
from ConfigSpace.forbidden import (
    ForbiddenClause,
    ForbiddenConjunction,
    ForbiddenLike,
    ForbiddenRelation,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from ConfigSpace.conditions import ConditionLike
    from ConfigSpace.hyperparameters import Hyperparameter


@dataclass
class _Node:
    hp: Hyperparameter
    idx: int
    maximum_depth: int
    children: dict[str, tuple[_Node, ConditionLike]] = field(default_factory=dict)
    forbiddens: list[ForbiddenLike] = field(default_factory=list)

    # NOTE: We have the restriction that a hyperparameter can only have one parent
    # condition but multiple parent from which these relationshops are dervied
    # This is to prevent ambiguity between AND and OR, in other words, what
    # do we do with the current Node if the condition for one parent is satisfied
    # but not the other.
    # * A useful assertion to make is that all nodes in `parents` will have the same
    #   parent condition, also accessible via `parent_condition`
    parents: dict[str, tuple[_Node, ConditionLike]] = field(
        default_factory=dict,
        # We explicitly don't compare parents to prevent recursion
        compare=False,
    )
    parent_condition: ConditionLike | None = None

    def __lt__(self, __value: object) -> bool:
        if not isinstance(__value, _Node):
            return NotImplemented

        return (self.maximum_depth, self.hp.name) < (
            __value.maximum_depth,
            __value.hp.name,
        )

    def __le__(self, __value: object) -> bool:
        if not isinstance(__value, _Node):
            return NotImplemented

        return (self.maximum_depth, self.hp.name) <= (
            __value.maximum_depth,
            __value.hp.name,
        )

    def __gt__(self, __value: object) -> bool:
        if not isinstance(__value, _Node):
            return NotImplemented

        return (self.maximum_depth, self.hp.name) > (
            __value.maximum_depth,
            __value.hp.name,
        )

    def __ge__(self, __value: object) -> bool:
        if not isinstance(__value, _Node):
            return NotImplemented

        return (self.maximum_depth, self.hp.name) >= (
            __value.maximum_depth,
            __value.hp.name,
        )

    @property
    def name(self) -> str:
        return self.hp.name

    def child_conditions(self) -> list[Condition | Conjunction]:
        return [condition for _, condition in self.children.values()]

    def parent_conditions(self) -> list[Condition | Conjunction]:
        if self.parent_condition is not None:
            return [self.parent_condition]
        return []

    def propogate_new_depth(self, depth: int) -> None:
        if depth <= self.maximum_depth:
            # There's no need to propogate the depth, this nodes
            # maximum depth already dominates it
            return

        self.maximum_depth = depth
        for child, _ in self.children.values():
            child.propogate_new_depth(depth + 1)


@dataclass
class DAG:
    # All relevant information is kept in these fields, rest is cached information
    nodes: dict[str, _Node] = field(default_factory=dict)
    unconditional_forbiddens: list[ForbiddenLike] = field(default_factory=list)
    conditional_forbiddens: list[ForbiddenLike] = field(default_factory=list)

    # Keep track of nodes
    roots: dict[str, _Node] = field(default_factory=dict, compare=False)
    non_roots: dict[str, _Node] = field(default_factory=dict, compare=False)

    # Keep track of conditions
    conditions: list[ConditionLike] = field(default_factory=list, compare=False)
    minimum_condition_span: list[
        tuple[ConditionLike, list[_Node], npt.NDArray[np.int64]]
    ] = field(
        default_factory=list,
        compare=False,
    )

    # Indexes into the states of the dag
    index_of: dict[str, int] = field(default_factory=dict, compare=False)
    at: list[str] = field(default_factory=list, compare=False)
    children_of: dict[str, list[Hyperparameter]] = field(
        default_factory=dict,
        compare=False,
    )
    parents_of: dict[str, list[Hyperparameter]] = field(
        default_factory=dict,
        compare=False,
    )
    child_conditions_of: dict[str, list[ConditionLike]] = field(
        default_factory=dict,
        compare=False,
    )
    parent_conditions_of: dict[str, list[ConditionLike]] = field(
        default_factory=dict,
        compare=False,
    )

    # Internal flag to keep track of whether we are currently in a transaction
    _updating: bool = False

    @contextmanager
    def update(self) -> Iterator[Self]:
        if self._updating:
            raise RuntimeError("Already in a pending update")

        self._updating = True
        try:
            yield self
        finally:
            # In case things crash and are caught, we need to make sure
            # we reset the transaction flag.
            # This is particularly useful in `pytest.raises()` situations
            # where we expect a crash to happen
            self._updating = False

        # Sort the nodes into roots, non-roots, and nodes, each sorted by (depth, name)
        roots: dict[str, _Node] = {}
        non_roots: dict[str, _Node] = {}
        nodes: dict[str, _Node] = {}
        at: list[str] = []
        index_of: dict[str, int] = {}

        nodes_sorted_by_depth_and_name = sorted(self.nodes.values())
        for i, n in enumerate(nodes_sorted_by_depth_and_name):
            if n.maximum_depth == 1:
                roots[n.name] = n
            else:
                non_roots[n.name] = n

            n.idx = i
            index_of[n.name] = i
            at.append(n.name)

            nodes[n.name] = n

        self.roots = roots
        self.non_roots = non_roots
        self.nodes = nodes
        self.at = at
        self.index_of = index_of

        # Sort out forbiddens based on whether they are unconditional or conditional
        unconditional_forbiddens = []
        conditional_forbiddens = []

        # Sort forbiddens so it's always same order
        for forbidden in sorted(self.forbiddens, key=str):
            if self._is_unconditional_forbidden(forbidden):
                unconditional_forbiddens.append(forbidden)
            else:
                conditional_forbiddens.append(forbidden)

            if isinstance(forbidden, ForbiddenClause):
                hp_name = forbidden.hyperparameter.name
                self.nodes[hp_name].forbiddens.append(forbidden)
            elif isinstance(forbidden, ForbiddenConjunction):
                dlcs = forbidden.get_descendant_literal_clauses()
                for dlc in sorted(dlcs, key=str):
                    hp_name = dlc.hyperparameter.name
                    self.nodes[hp_name].forbiddens.append(forbidden)
            elif isinstance(forbidden, ForbiddenRelation):
                left_name = forbidden.left.name
                right_name = forbidden.right.name
                self.nodes[left_name].forbiddens.append(forbidden)
                self.nodes[right_name].forbiddens.append(forbidden)
            else:
                raise NotImplementedError(type(forbidden))

        # Now we reduce the set of all forbiddens to ensure equivalence on nodes.
        for node in nodes.values():
            node.forbiddens = list(unique_everseen(node.forbiddens, key=id))

        self.unconditional_forbiddens = unconditional_forbiddens
        self.conditional_forbiddens = conditional_forbiddens

        # Sort conditions by their parents sort order
        conditions = []
        for node in self.nodes.values():
            if node.parent_condition is not None:
                conditions.append(node.parent_condition)

        self.conditions = list(unique_everseen(conditions, key=id))

        # Update indices of conditions and forbiddens
        for forbidden in self.forbiddens:
            forbidden.set_vector_idx(self.index_of)

        for condition in self.conditions:
            condition.set_vector_idx(self.index_of)

        # Create children and parent of dictionaries
        self.children_of = {}
        self.parents_of = {}
        self.child_conditions_of = {}
        self.parent_conditions_of = {}

        for n in self.nodes.values():
            self.children_of[n.name] = [child.hp for child, _ in n.children.values()]
            self.parents_of[n.name] = [parent.hp for parent, _ in n.parents.values()]
            self.child_conditions_of[n.name] = n.child_conditions()
            self.parent_conditions_of[n.name] = n.parent_conditions()

        # Cache out the minimum condition span used for sampling
        self.minimum_condition_span = self._generate_minimum_condition_span()

    @property
    def forbiddens(self) -> list[ForbiddenLike]:
        return self.unconditional_forbiddens + self.conditional_forbiddens

    def add(self, hp: Hyperparameter) -> None:
        existing = self.nodes.get(hp.name, None)
        if existing is not None:
            raise HyperparameterAlreadyExistsError(
                f"Hyperparameter {existing.name} already exists in space."
                f"\nExisting: {existing.hp}"
                f"\nNew one: {hp}",
            )

        # idx will get filled in post transaction
        node = _Node(hp, maximum_depth=1, idx=len(self.nodes))
        self.nodes[hp.name] = node
        self.roots[hp.name] = node

    def add_condition(self, condition: ConditionLike) -> None:
        if not self._updating:
            raise RuntimeError(
                "Cannot add conditions outside of transaction."
                "Please use `add_condition` inside `with dag.transaction():`",
            )
        # All condition/conjunctions have the same child
        child_hp = condition.child
        child_name = child_hp.name
        child = self.nodes.get(child_name, None)
        if child is None:
            raise ChildNotFoundError(
                f"Could not find child '{child_name}' for condition {condition}",
            )

        # This stems from the fact all nodes can only have one parent condition,
        # as this prevent ambiguity. If a node can have two parents it depends on,
        # is it inherently an AND or an OR condition? This
        if child.parent_condition is not None and condition != child.parent_condition:
            raise AmbiguousConditionError(
                "Adding a second parent condition for a for a hyperparameter"
                " is ambiguous and therefore forbidden. Use an `OrConjunction`"
                " or `AndConjunction` to combine conditions instead."
                f"\nAlready inserted: {child.parent_condition}"
                f"\nNew one: {condition}",
            )

        parent_names = (
            [dlc.parent.name for dlc in condition.dlcs]
            if isinstance(condition, Conjunction)
            else [condition.parent.name]
        )
        for parent_name in parent_names:
            parent = self.nodes.get(parent_name, None)

            if parent is None:
                raise ParentNotFoundError(
                    f"Could not find parent '{parent_name}' for condition {condition}",
                )

            # Ensure there is no existing condition between the parent and the child
            _, existing = parent.children.get(child.name, (None, None))
            if existing is not None and condition != existing:
                raise AmbiguousConditionError(existing, condition)

            # Now we are certain they exist and no existing condition between them,
            # update the nodes to point to each other
            parent.children[child.name] = (child, condition)

            child.parents[parent.name] = (parent, condition)
            child.parent_condition = condition

            # Make sure we don't have any cyclic dependancies
            self._check_cyclic_dependancy()

            # We also need to update the depths and all its children in a recursive
            # manner
            child.propogate_new_depth(parent.maximum_depth + 1)

            # Reorder the children and parents by their sort order (depth, name)
            parent.children = {
                c.name: (c, cond) for c, cond in sorted(parent.children.values())
            }
            child.parents = {
                p.name: (p, cond) for p, cond in sorted(child.parents.values())
            }

    def add_forbidden(self, forbidden: ForbiddenLike) -> None:
        if not self._updating:
            raise RuntimeError(
                "Cannot add forbidden outside of transaction."
                "Please use `add_forbidden` inside `with dag.transaction():`",
            )

        def _check_hp(tmp_clause: ForbiddenLike, hp: Hyperparameter) -> None:
            if hp.name not in self.nodes:
                raise HyperparameterNotFoundError(
                    f"Cannot add '{tmp_clause}' because it references '{hp.name}'"
                    f"\nHyperparameter {hp.name} not found in space.\n{self}",
                )

        if isinstance(forbidden, ForbiddenClause):
            _check_hp(forbidden, forbidden.hyperparameter)
        elif isinstance(forbidden, ForbiddenConjunction):
            dlcs = forbidden.get_descendant_literal_clauses()
            for clause in dlcs:
                _check_hp(clause, clause.hyperparameter)

        elif isinstance(forbidden, ForbiddenRelation):
            _check_hp(forbidden, forbidden.left)
            _check_hp(forbidden, forbidden.right)
        else:
            raise NotImplementedError(type(forbidden))

        if self._is_unconditional_forbidden(forbidden):
            self.unconditional_forbiddens.append(forbidden)
        else:
            self.conditional_forbiddens.append(forbidden)

    def dependancies(
        self,
        name: str | Hyperparameter,
    ) -> Iterator[tuple[_Node, ConditionLike]]:
        # child -> [root to this parameter]
        _name = name if isinstance(name, str) else name.name
        node = self.nodes[_name]
        seen: set[str] = set()
        for parent, condition in node.parents.values():
            if parent.name in seen:
                continue

            seen.add(parent.name)
            yield from self.dependancies(parent.name)
            yield (parent, condition)

    def _is_unconditional_forbidden(self, forbidden: ForbiddenLike) -> bool:
        if isinstance(forbidden, ForbiddenClause):
            name = forbidden.hyperparameter.name
            return name in self.roots

        if isinstance(forbidden, ForbiddenConjunction):
            dlcs = forbidden.get_descendant_literal_clauses()
            return all(dlc.hyperparameter.name in self.roots for dlc in dlcs)

        if isinstance(forbidden, ForbiddenRelation):
            return (
                forbidden.left.name in self.roots and forbidden.right.name in self.roots
            )

        raise NotImplementedError(type(forbidden))

    def _generate_minimum_condition_span(
        self,
    ) -> list[tuple[ConditionLike, list[_Node], npt.NDArray[np.int64]]]:
        # TODO: This can be improved by using knowledge of AND conjunctions
        # of conditionals

        # The minimum number of conditions required to determine whether all
        # hyperparameters are active or inactive. I.e. many hps' will rely on
        # a single choice being to a specific value.

        # First collect all conditions and their nodes
        def affected_children(condition: ConditionLike) -> list[_Node]:
            if isinstance(condition, Condition):
                return [self.nodes[condition.child.name]]

            if isinstance(condition, Conjunction):
                return [self.nodes[dlc.child.name] for dlc in condition.dlcs]

            raise NotImplementedError(type(condition))

        conditions_with_nodes_effected = [
            (condition, affected_children(condition)) for condition in self.conditions
        ]

        # We ensure they're sorted such that by traversing through them lineraly,
        # the condition being checked will have all the parents it relies on having
        # been fully checked. This is done by ensuring they'red order by have soon
        # in the heirarchy they need to be inferred.
        sorted_conditions = sorted(
            conditions_with_nodes_effected,
            key=lambda cond_nodes: min(
                c.maximum_depth for c in affected_children(cond_nodes[0])
            ),
        )

        # Now we deduplicate conditions, merging the effected children. This happens as
        # multiple hyperparameters may have the exact same condition to be activated,
        # i.e. random forest variables are active when classifier == "random_forest"
        # NOTE: The dict part is just to remove duplicate entries
        basis_conditions_with_effected_nodes: list[
            tuple[ConditionLike, dict[int, _Node]]
        ] = []
        for condition, nodes in sorted_conditions:
            for basis_condition, effected_nodes in basis_conditions_with_effected_nodes:
                if condition.conditionally_equal(basis_condition):
                    effected_nodes.update({n.idx: n for n in nodes})
                    break
            else:
                basis_conditions_with_effected_nodes.append(
                    (condition, {n.idx: n for n in nodes}),
                )

        # And as a final touch, sort and get ready for use
        basis_condition_with_index_masks: list[
            tuple[ConditionLike, list[_Node], npt.NDArray[np.int64]]
        ] = []
        for condition, effected_nodes in basis_conditions_with_effected_nodes:
            sorted_effected_nodes = sorted(effected_nodes.values(), key=lambda n: n.idx)
            indices = np.array([n.idx for n in sorted_effected_nodes], dtype=np.int64)
            basis_condition_with_index_masks.append(
                (condition, sorted_effected_nodes, indices),
            )

        return basis_condition_with_index_masks

    def _check_cyclic_dependancy(self) -> None:
        tmp_dag = nx.DiGraph()
        for node_name in self.nodes:
            tmp_dag.add_node(node_name)

        for node in self.nodes.values():
            for parent_name, _ in node.parents.items():
                tmp_dag.add_edge(parent_name, node.hp.name)

        for node in self.nodes.values():
            for parent_name, _ in node.parents.items():
                tmp_dag.add_edge(parent_name, node.hp.name)

        if not nx.is_directed_acyclic_graph(tmp_dag):
            cycles = list(nx.simple_cycles(tmp_dag))
            for cycle in cycles:
                cycle.sort()
            cycles.sort()
            raise CyclicDependancyError(cycles)
