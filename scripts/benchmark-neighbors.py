# /home/feurerm/projects/ConfigSpace/test/test_searchspaces/auto-sklearn_2017_11_17.pcs
# Average time sampling 100 configurations 0.0115247011185
# Average time retrieving a nearest neighbor 0.00251974105835
# Average time checking one configuration 0.000194481347553
from __future__ import annotations

import os
import time

import numpy as np

import ConfigSpace
import ConfigSpace.read_and_write.pcs as pcs_parser
import ConfigSpace.util

n_configs = 100


def run_test(configuration_space_path):
    if "2017_11" not in configuration_space_path:
        return

    with open(configuration_space_path) as fh:
        cs = pcs_parser.read(fh)

    print("###")
    print(configuration_space_path, flush=True)

    neighborhood_time = []

    for i in range(3):
        cs.seed(i)
        rs = np.random.RandomState(i)
        configurations = cs.sample_configuration(size=n_configs)
        for c in configurations:
            c.check_valid_configuration()

        for _j, c in enumerate(configurations):
            start_time = time.time()
            neighborhood = ConfigSpace.util.get_one_exchange_neighbourhood(
                c,
                seed=rs,
                num_neighbors=4,
            )
            _ns = list(neighborhood)
            end_time = time.time()
            neighborhood_time.append(end_time - start_time)

    print(f"Average time retrieving a nearest neighbor {np.mean(neighborhood_time):f}")


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
