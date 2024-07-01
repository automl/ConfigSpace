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

    sampling_time = []
    neighborhood_time = []
    validation_times = []

    # Sample a little bit
    for i in range(20):
        cs.seed(i)
        start_time = time.time()
        configurations = cs.sample_configuration(size=n_configs)
        end_time = time.time()
        sampling_time.append(end_time - start_time)
        for c in configurations:
            c.check_valid_configuration()

        for j, c in enumerate(configurations):
            if i > 10:
                neighborhood = ConfigSpace.util.get_one_exchange_neighbourhood(
                    c,
                    seed=i * j,
                    num_neighbors=4,
                )

                start_time = time.time()
                validation_time = []
                for _shuffle, n in enumerate(neighborhood):
                    v_start_time = time.time()
                    n.check_valid_configuration()
                    v_end_time = time.time()
                    validation_time.append(v_end_time - v_start_time)
                end_time = time.time()
                neighborhood_time.append(
                    end_time - start_time - np.sum(validation_time),
                )
                validation_times.extend(validation_time)

    print("Average time sampling %d configurations" % n_configs, np.mean(sampling_time))
    print("Average time retrieving a nearest neighbor", np.mean(neighborhood_time))
    print("Average time checking one configuration", np.mean(validation_times))


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
