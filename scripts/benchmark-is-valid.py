from __future__ import annotations

import os
import time

import numpy as np

from ConfigSpace.read_and_write.pcs import read as read_pcs
from ConfigSpace.util import get_one_exchange_neighbourhood


def run_test(configuration_space_path):
    if "2017_11" not in configuration_space_path:
        return

    with open(configuration_space_path) as fh:
        cs = read_pcs(fh)

    print("###")
    print(configuration_space_path, flush=True)

    configs = []
    times = []

    # Sample a little bit
    for i in range(20):
        cs.seed(i)
        configurations = cs.sample_configuration(size=10)
        for j, c in enumerate(configurations):
            neighborhood = get_one_exchange_neighbourhood(
                c,
                seed=i * j,
                num_neighbors=4,
            )
            configs.extend(list(neighborhood))

    for c in configs:
        t0 = time.time()
        c.is_valid_configuration()
        t1 = time.time()
        times.append(t1 - t0)

    print("Average time checking one configuration", np.mean(times))


this_file = os.path.abspath(__file__)
this_directory = os.path.dirname(this_file)
configuration_space_path = os.path.join(
    this_directory,
    "..",
    "test",
    "test_searchspaces",
)
configuration_space_path = os.path.abspath(configuration_space_path)
pcs_files = os.listdir(configuration_space_path)

for pcs_file in pcs_files:
    if ".pcs" in pcs_file:
        full_path = os.path.join(configuration_space_path, pcs_file)
        run_test(full_path)

# ------------
# Average time sampling 100 configurations 0.0115247011185
# Average time retrieving a nearest neighbor 0.00251974105835
# Average time checking one configuration 0.000194481347553

# is_close_integer
# Average time sampling 100 configurations 0.1998179078102112
# Average time retrieving a nearest neighbor 0.023387917677561442
# Average time checking one configuration 0.0012463332840478253

# /home/skantify/code/ConfigSpace/test/test_searchspaces/auto-sklearn_2017_11_17.pcs
# Average time sampling 100 configurations 0.05419049263000488
# Average time retrieving a nearest neighbor 0.01149404075410631
# Average time checking one configuration 0.0006455589667150331
