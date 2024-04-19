from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import chain, product
from typing import TYPE_CHECKING, ClassVar, Iterator
from typing_extensions import Self

import numpy as np
from more_itertools import unique_everseen

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
    ForbiddenAndConjunction,
    ForbiddenClause,
    ForbiddenConjunction,
    ForbiddenEqualsClause,
    ForbiddenInClause,
    ForbiddenLike,
    ForbiddenRelation,
)
from ConfigSpace.types import f64

if TYPE_CHECKING:
    from ConfigSpace.conditions import ConditionLike
    from ConfigSpace.hyperparameters import Hyperparameter
    from ConfigSpace.types import Array


@dataclass
class ConditionNode:
    CACHED_NAN_ARRAY: ClassVar[Array[f64]] = np.array([np.nan], dtype=f64)

    condition: ConditionLike
    dependants: list[ConditionNode]
    unique_children: dict[int, HPNode]
    children_indices: Array[np.intp] = field(
        default_factory=lambda: np.array((), dtype=np.intp),
        compare=False,
    )
    nan_arr: Array[f64] = field(
        default_factory=lambda: ConditionNode.CACHED_NAN_ARRAY[:0],
        compare=False,
    )

    def node_parents(self) -> list[str]:
        if isinstance(self.condition, Condition):
            return [self.condition.parent.name]
        return [dlc.parent.name for dlc in self.condition.dlcs]

    def depends_on(self, other: ConditionNode) -> bool:
        return any(parent in other.dependant_names() for parent in self.node_parents())

    def dependant_names(self) -> set[str]:
        _dep = {node.name for node in self.unique_children.values()}
        for dep in self.dependants:
            _dep.update(dep.dependant_names())
        return _dep

    @classmethod
    def from_node(cls, node: HPNode) -> ConditionNode:
        assert node.parent_condition is not None
        return cls(
            condition=node.parent_condition,
            dependants=[],
            unique_children={node.idx: node},
            children_indices=np.array([node.idx], dtype=np.int64),
        )

    def has_equivalent_condition(self, node: HPNode) -> bool:
        assert node.parent_condition is not None
        return self.condition.equivalent_condition_on_parent(
            node.parent_condition,
        )

    def __str__(self, indent: int = 0) -> str:
        parts = ["  " * indent + "*" + str(self.condition)]
        for child in self.unique_children.values():
            parts.append("  " * (indent + 1) + f"Activates: {child.name}")
        for dominated in self.dependants:
            parts.append(dominated.__str__(indent + 1))
        return "\n".join(parts)


@dataclass
class HPNode:
    hp: Hyperparameter
    idx: int
    maximum_depth: int
    children: dict[str, tuple[HPNode, ConditionLike]] = field(default_factory=dict)
    forbiddens: list[ForbiddenLike] = field(default_factory=list)

    # NOTE: We have the restriction that a hyperparameter can only have one parent
    # condition but multiple parent from which these relationshops are dervied
    # This is to prevent ambiguity between AND and OR, in other words, what
    # do we do with the current Node if the condition for one parent is satisfied
    # but not the other.
    # * A useful assertion to make is that all nodes in `parents` will have the same
    #   parent condition, also accessible via `parent_condition`
    parents: dict[str, tuple[HPNode, ConditionLike]] = field(
        default_factory=dict,
        # We explicitly don't compare parents to prevent recursion
        compare=False,
    )
    parent_condition: ConditionLike | None = None

    def __lt__(self, __value: object) -> bool:
        if not isinstance(__value, HPNode):
            return NotImplemented

        return (self.maximum_depth, self.hp.name) < (
            __value.maximum_depth,
            __value.hp.name,
        )

    def __le__(self, __value: object) -> bool:
        if not isinstance(__value, HPNode):
            return NotImplemented

        return (self.maximum_depth, self.hp.name) <= (
            __value.maximum_depth,
            __value.hp.name,
        )

    def __gt__(self, __value: object) -> bool:
        if not isinstance(__value, HPNode):
            return NotImplemented

        return (self.maximum_depth, self.hp.name) > (
            __value.maximum_depth,
            __value.hp.name,
        )

    def __ge__(self, __value: object) -> bool:
        if not isinstance(__value, HPNode):
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
    nodes: dict[str, HPNode] = field(default_factory=dict)
    unconditional_forbiddens: list[ForbiddenLike] = field(default_factory=list)
    conditional_forbiddens: list[ForbiddenLike] = field(default_factory=list)

    # Keep track of nodes
    roots: dict[str, HPNode] = field(default_factory=dict, compare=False)
    non_roots: dict[str, HPNode] = field(default_factory=dict, compare=False)

    # Keep track of conditions
    conditions: list[ConditionLike] = field(default_factory=list, compare=False)
    minimum_conditions: list[ConditionNode] = field(default_factory=list, compare=False)
    change_hp_lookup: dict[str, list[ConditionNode]] = field(
        default_factory=dict,
        compare=False,
    )
    all_trees_needed_for_sampling: list[ConditionNode] = field(
        default_factory=list,
        compare=False,
    )
    normalized_defaults: Array[f64] = field(
        default_factory=lambda: np.array((), dtype=f64),
        compare=False,
    )
    # OPTIM: Mainly used for generating neighbors, do not use if validation
    # the underlying forbidden is required to be displayed to the user
    fast_forbidden_checks: list[ForbiddenLike] = field(
        default_factory=list,
        compare=False,
    )
    forbidden_lookup: dict[str, list[ForbiddenLike]] = field(
        default_factory=dict,
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
    hyperparameters: list[Hyperparameter] = field(default_factory=list, compare=False)

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
        roots: dict[str, HPNode] = {}
        non_roots: dict[str, HPNode] = {}
        nodes: dict[str, HPNode] = {}
        at: list[str] = []
        index_of: dict[str, int] = {}
        hyperparameters: list[Hyperparameter] = []

        nodes_sorted_by_depth_and_name = sorted(self.nodes.values())
        for i, n in enumerate(nodes_sorted_by_depth_and_name):
            if n.maximum_depth == 1:
                roots[n.name] = n
            else:
                non_roots[n.name] = n

            n.idx = i
            index_of[n.name] = i
            at.append(n.name)
            hyperparameters.append(n.hp)

            nodes[n.name] = n

        self.roots = roots
        self.non_roots = non_roots
        self.nodes = nodes
        self.at = at
        self.index_of = index_of
        self.hyperparameters = hyperparameters
        self.normalized_defaults = np.array(
            [hp._normalized_default_value for hp in hyperparameters],
            dtype=f64,
        )

        # Sort out forbiddens based on whether they are unconditional or conditional
        unconditional_forbiddens = []
        conditional_forbiddens = []

        def _sort_forbiddens(_f: ForbiddenLike, append: bool = True) -> None:
            if append:
                if self._is_unconditional_forbidden(_f):
                    unconditional_forbiddens.append(_f)
                else:
                    conditional_forbiddens.append(_f)

            if isinstance(_f, ForbiddenClause):
                hp_name = _f.hyperparameter.name
                self.nodes[hp_name].forbiddens.append(_f)
            elif isinstance(_f, ForbiddenConjunction):
                for dlc in sorted(_f.dlcs, key=str):
                    _sort_forbiddens(dlc, append=False)
            elif isinstance(_f, ForbiddenRelation):
                left_name = _f.left.name
                right_name = _f.right.name
                self.nodes[left_name].forbiddens.append(_f)
                self.nodes[right_name].forbiddens.append(_f)
            else:
                raise NotImplementedError(type(_f))

        # Sort forbiddens so it's always same order
        for forbidden in sorted(self.forbiddens, key=str):
            _sort_forbiddens(forbidden)

        # Now we reduce the set of all forbiddens to ensure equivalence on nodes.
        for node in nodes.values():
            node.forbiddens = list(unique_everseen(node.forbiddens, key=str))

        self.unconditional_forbiddens = unconditional_forbiddens
        self.conditional_forbiddens = conditional_forbiddens

        # Please check function for optimization applied, these are only
        # used to speed up internal verifications
        self.fast_forbidden_checks = self._optimized_forbiddens(
            unconditional_forbiddens + conditional_forbiddens,
        )

        def _parents(_f: ForbiddenLike) -> list[str]:
            if isinstance(_f, ForbiddenClause):
                return [_f.hyperparameter.name]
            if isinstance(_f, ForbiddenConjunction):
                return list(set(chain.from_iterable(_parents(dlc) for dlc in _f.dlcs)))
            if isinstance(_f, ForbiddenRelation):
                return [_f.left.name, _f.right.name]

            raise NotImplementedError(type(_f))

        forbidden_lookup: dict[str, list[ForbiddenLike]] = {}
        for forbidden in self.fast_forbidden_checks:
            for parent in _parents(forbidden):
                if parent not in forbidden_lookup:
                    forbidden_lookup[parent] = [forbidden]
                else:
                    forbidden_lookup[parent].append(forbidden)

        self.forbidden_lookup = forbidden_lookup

        # Sort conditions by their parents sort order
        conditions = []
        for node in self.nodes.values():
            if node.parent_condition is not None:
                conditions.append(node.parent_condition)

        self.conditions = list(unique_everseen(conditions, key=str))

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
        # This is useful for sampling
        minimum_conditions = self._minimum_conditions()

        for a, b in product(minimum_conditions, minimum_conditions):
            if a is not b:
                assert a.unique_children.keys().isdisjoint(b.unique_children.keys())

        def shallowest_parent(_x: ConditionNode) -> int:
            return min(self.nodes[_p].maximum_depth for _p in _x.node_parents())

        self.minimum_conditions = sorted(minimum_conditions, key=shallowest_parent)

        condition_trees = {id(c): c for c in self.minimum_conditions}
        updated = True
        while updated:
            updated = False
            values: list[ConditionNode] = list(condition_trees.values())
            for a, b in product(values, values):
                if a is b:
                    continue

                if b.depends_on(a):
                    a.dependants.append(b)
                    a.dependants = sorted(a.dependants, key=shallowest_parent)
                    if id(b) in condition_trees:
                        condition_trees.pop(id(b))
                    updated = True

        sorted_condition_trees = sorted(condition_trees.values(), key=shallowest_parent)

        # Now we go through the trees and update the lookup
        # Note that it's possible to have reconverging paths, i.e.
        # the same condition node might appear when starting from two
        # different roots.
        self.change_hp_lookup: dict[str, list[ConditionNode]] = {}
        trees: deque[ConditionNode] = deque(sorted_condition_trees)
        while trees:
            tree = trees.popleft()
            for parent in tree.node_parents():
                if parent not in self.change_hp_lookup:
                    self.change_hp_lookup[parent] = [tree, *tree.dependants]
                else:
                    self.change_hp_lookup[parent].extend([tree, *tree.dependants])
                    self.change_hp_lookup[parent] = sorted(
                        self.change_hp_lookup[parent],
                        key=shallowest_parent,
                    )

            trees.extend(tree.dependants)

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
        node = HPNode(hp, maximum_depth=1, idx=len(self.nodes))
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
            for clause in forbidden.dlcs:
                if isinstance(clause, ForbiddenRelation):
                    _check_hp(clause, clause.left)
                    _check_hp(clause, clause.right)
                else:
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

    def _is_unconditional_forbidden(self, forbidden: ForbiddenLike) -> bool:
        if isinstance(forbidden, ForbiddenClause):
            name = forbidden.hyperparameter.name
            return name in self.roots

        if isinstance(forbidden, ForbiddenRelation):
            return (
                forbidden.left.name in self.roots and forbidden.right.name in self.roots
            )

        if isinstance(forbidden, ForbiddenConjunction):
            return all(self._is_unconditional_forbidden(dlc) for dlc in forbidden.dlcs)

        raise NotImplementedError(type(forbidden))

    def _minimum_conditions(self) -> list[ConditionNode]:
        # First we group the conditions by the equivalence of their parent conditions,
        # i.e. two hyperparameters both rely on algorithm == "A"
        base_conditions: dict[int, ConditionNode] = {}
        for node in self.nodes.values():
            # This node has no parent as is a root
            if node.parent_condition is None:
                assert node.name in self.roots
                continue

            for a in base_conditions.values():
                if a.has_equivalent_condition(node):
                    a.unique_children[node.idx] = node
                    a.children_indices = np.array(
                        list(a.unique_children.keys()),
                        dtype=np.intp,
                    )
                    break
            else:
                _a = ConditionNode.from_node(node)
                base_conditions[id(_a)] = _a

        # OPTIM: `change_hp` relies on this to basically nan out the children
        # We can save some time by pre-creating the nan arrays
        if any(base_conditions):
            largest_required = max(
                len(v.children_indices) for v in base_conditions.values()
            )
            # this is a global
            ConditionNode.CACHED_NAN_ARRAY = np.full(
                largest_required,
                np.nan,
                dtype=f64,
            )
            for v in base_conditions.values():
                v.nan_arr = ConditionNode.CACHED_NAN_ARRAY[: len(v.children_indices)]

        # We return the base conditions such that conditions relying on
        # earlier shallower hps are first
        return list(base_conditions.values())

    def _optimized_forbiddens(
        self,
        forbiddens: list[ForbiddenLike],
    ) -> list[ForbiddenLike]:
        # OPTIM: Many time, forbiddens are an AND conjunction of multiple
        # clauses, where the clauses are all on the same hyperparameters.
        # (classifier== 'adaboost' && preprocessor== 'densifier'),
        # (classifier== 'adaboost' && preprocessor== 'kitchen_sinks'),
        # (classifier== 'adaboost' && preprocessor == 'nystroem_sampler')
        # When performing operations of forbiddens, only a single array at a time,
        # the **slowest part is actually indexing into a numpy array like[so_idx]**,
        # even if just retrieving one index.
        # https://stackoverflow.com/a/29311751/5332072
        # We can't get around indexing so the next best attempt is to reduce the
        # amount indexing required.
        # (classifier== 'adaboost' && preprocessor in ('nystroem_sampler', 'kitchen_sinks', 'densifier'))  # noqa: E501
        # We make the assumption that shared forbiddens are more likely to occure
        # on _more shallow_ nodes.
        to_optimize: dict[
            # First parent_name, with N-1 (hp_name, value)...
            tuple[str, tuple[tuple[str, f64], ...]],
            # unique parts of AND, list to for isin
            tuple[tuple[ForbiddenEqualsClause, ...], list[ForbiddenEqualsClause]],
        ] = {}
        forbiddens_to_return: list[ForbiddenLike] = []

        and_conjunction_parts: list[list[ForbiddenEqualsClause]] = []
        for f in forbiddens:
            if isinstance(f, ForbiddenAndConjunction) and all(
                isinstance(c, ForbiddenEqualsClause) for c in f.components
            ):
                and_conjunction_parts.append(
                    sorted(
                        f.components,  # type: ignore
                        key=lambda _x: self.index_of[_x.hyperparameter.name],  # type: ignore
                    ),
                )
            else:
                forbiddens_to_return.append(f)

        for *firsts, last in and_conjunction_parts:
            shallowest_key = tuple(
                (x.hyperparameter.name, x.vector_value) for x in firsts
            )
            parent_key = last.hyperparameter.name
            joint_key = (parent_key, shallowest_key)
            if joint_key in to_optimize:
                _, conjunctions = to_optimize[joint_key]
                conjunctions.append(last)
            else:
                to_optimize[joint_key] = (tuple(firsts), [last])

        for _and_parts, equal_components in to_optimize.values():
            # Didn't share first parts such that we could group it with anything else
            if len(equal_components) == 1:
                conj = ForbiddenAndConjunction(*_and_parts, equal_components[0])
                conj.set_vector_idx(self.index_of)
                forbiddens_to_return.append(conj)
                continue

            isin_clause = ForbiddenInClause(
                equal_components[0].hyperparameter,
                [x.value for x in equal_components],
            )
            conj = ForbiddenAndConjunction(*_and_parts, isin_clause)
            conj.set_vector_idx(self.index_of)
            forbiddens_to_return.append(conj)

        return forbiddens_to_return

    def _check_cyclic_dependancy(self) -> None:
        seen: set[str] = set()
        explored: set[str] = set()

        for node in self.nodes.values():
            if node.name in explored:
                continue
            fringe = [node]
            while fringe:
                w = fringe[-1]
                if w.name in explored:
                    fringe.pop()
                    continue
                seen.add(w.name)

                new_nodes = []
                children = [c for c, _ in w.children.values()]
                for c in children:
                    if c.name not in explored:
                        if c.name in seen:
                            raise CyclicDependancyError()

                        new_nodes.append(c)

                if new_nodes:
                    fringe.extend(new_nodes)
                else:
                    explored.add(w.name)
                    fringe.pop()
