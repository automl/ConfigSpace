from __future__ import annotations

import time

import numpy as np

from ConfigSpace import (
    CategoricalHyperparameter,
    ConfigurationSpace,
    EqualsCondition,
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
    InCondition,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from ConfigSpace.util import grid_generator

n_repeats = 3


def build_space() -> ConfigurationSpace:
    # Mix of all grid-able types, plus a condition and a forbidden so the
    # validity checks inside the generator are exercised too
    cs = ConfigurationSpace(seed=1)
    cs.add(
        CategoricalHyperparameter("cat", ["a", "b", "c"]),
        OrdinalHyperparameter("ord", ["low", "mid", "high"]),
        UniformFloatHyperparameter("float", -1, 1),
        UniformFloatHyperparameter("logfloat", 1e-4, 1, log=True),
        UniformIntegerHyperparameter("int", 1, 100),
        UniformIntegerHyperparameter("child", 1, 10),
    )
    cs.add(
        EqualsCondition(cs["child"], cs["cat"], "a"),
        ForbiddenAndConjunction(
            ForbiddenEqualsClause(cs["cat"], "b"),
            ForbiddenEqualsClause(cs["ord"], "high"),
        ),
    )
    return cs


def build_conditional_space() -> ConfigurationSpace:
    # 16 hyperparameters, most of them conditional (up to 2 levels deep), plus
    # forbiddens spanning parents and children
    cs = ConfigurationSpace(seed=1)
    cs.add(
        CategoricalHyperparameter("model", ["svm", "rf", "knn", "mlp"]),
        # svm
        UniformFloatHyperparameter("C", 1e-3, 1e3, log=True),
        CategoricalHyperparameter("kernel", ["rbf", "poly", "linear"]),
        UniformFloatHyperparameter("gamma", 1e-4, 1, log=True),
        UniformIntegerHyperparameter("degree", 2, 5),
        # rf
        UniformIntegerHyperparameter("n_estimators", 10, 500),
        UniformIntegerHyperparameter("max_depth", 1, 30),
        CategoricalHyperparameter("criterion", ["gini", "entropy"]),
        # knn
        UniformIntegerHyperparameter("k", 1, 50),
        CategoricalHyperparameter("weights", ["uniform", "distance"]),
        # mlp
        UniformIntegerHyperparameter("layers", 1, 4),
        UniformIntegerHyperparameter("units", 16, 512, log=True),
        CategoricalHyperparameter("activation", ["relu", "tanh"]),
        CategoricalHyperparameter("solver", ["adam", "sgd"]),
        UniformFloatHyperparameter("lr", 1e-5, 1e-1, log=True),
        UniformFloatHyperparameter("momentum", 0.0, 0.99),
    )
    model = cs["model"]
    cs.add(
        EqualsCondition(cs["C"], model, "svm"),
        EqualsCondition(cs["kernel"], model, "svm"),
        InCondition(cs["gamma"], cs["kernel"], ["rbf", "poly"]),
        EqualsCondition(cs["degree"], cs["kernel"], "poly"),
        EqualsCondition(cs["n_estimators"], model, "rf"),
        EqualsCondition(cs["max_depth"], model, "rf"),
        EqualsCondition(cs["criterion"], model, "rf"),
        EqualsCondition(cs["k"], model, "knn"),
        EqualsCondition(cs["weights"], model, "knn"),
        EqualsCondition(cs["layers"], model, "mlp"),
        EqualsCondition(cs["units"], model, "mlp"),
        EqualsCondition(cs["activation"], model, "mlp"),
        EqualsCondition(cs["solver"], model, "mlp"),
        EqualsCondition(cs["lr"], model, "mlp"),
        EqualsCondition(cs["momentum"], cs["solver"], "sgd"),
    )
    # Forbiddens use bounds / categories, which are always on the grid
    cs.add(
        ForbiddenAndConjunction(
            ForbiddenEqualsClause(cs["kernel"], "poly"),
            ForbiddenEqualsClause(cs["degree"], 5),
        ),
        ForbiddenAndConjunction(
            ForbiddenEqualsClause(cs["weights"], "distance"),
            ForbiddenEqualsClause(cs["k"], 1),
        ),
        ForbiddenAndConjunction(
            ForbiddenEqualsClause(cs["criterion"], "entropy"),
            ForbiddenEqualsClause(cs["max_depth"], 30),
        ),
        ForbiddenAndConjunction(
            ForbiddenEqualsClause(cs["activation"], "tanh"),
            ForbiddenEqualsClause(cs["solver"], "sgd"),
        ),
        ForbiddenAndConjunction(
            ForbiddenEqualsClause(cs["layers"], 4),
            ForbiddenEqualsClause(cs["units"], 512),
        ),
    )
    return cs


def run_test(cs: ConfigurationSpace, num_steps: int) -> None:
    steps = {
        hp.name: num_steps
        for hp in cs.values()
        if isinstance(hp, (UniformFloatHyperparameter, UniformIntegerHyperparameter))
    }

    times = []
    for _ in range(n_repeats):
        start_time = time.time()
        n_configs = sum(1 for _ in grid_generator(cs, steps))
        times.append(time.time() - start_time)

    mean = np.mean(times)
    print(
        f"num_steps={num_steps:>2}: {n_configs:>6} configs, "
        f"average total {mean:.4f}s, per config {mean / n_configs * 1e6:.1f}us",
        flush=True,
    )


print("Simple space (6 hyperparameters, 1 condition, 1 forbidden)")
for num_steps in (3, 5, 8, 12):
    run_test(build_space(), num_steps)

print("Conditional space (16 hyperparameters, 15 conditions, 5 forbiddens)")
for num_steps in (3, 4, 5, 6):
    run_test(build_conditional_space(), num_steps)
