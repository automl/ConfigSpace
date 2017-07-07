import os
import time

import numpy as np

import ConfigSpace
import ConfigSpace.util
import ConfigSpace.io.pcs as pcs_parser


n_configs = 100


def run_test(configuration_space_path):
    if not '2017' in configuration_space_path:
        return

    with open(configuration_space_path) as fh:
        cs = pcs_parser.read(fh)

    print('###')
    print(configuration_space_path, flush=True)

    sampling_time = []
    neighborhood_time = []
    validation_times = []

    # Sample a little bit
    for i in range(10):
        cs.seed(i)
        start_time = time.time()
        configurations = cs.sample_configuration(size=n_configs)
        end_time = time.time()
        sampling_time.append(end_time - start_time)

        for j, c in enumerate(configurations):
            #c.is_valid_configuration()

            if i == 0:
                neighborhood = ConfigSpace.util.get_one_exchange_neighbourhood(
                    c, seed=i*j)

                start_time = time.time()
                validation_time = []
                for shuffle, n in enumerate(neighborhood):
                    v_start_time = time.time()
                    n.is_valid_configuration()
                    v_end_time = time.time()
                    validation_time.append(v_end_time - v_start_time)
                    if shuffle == 10:
                        break
                end_time = time.time()
                neighborhood_time.append(end_time - start_time - np.sum(validation_time))
                validation_times.extend(validation_time)

    print('Average time sampling %d configurations' % n_configs, np.mean(sampling_time))
    print('Average time retrieving a nearest neighbor', np.mean(neighborhood_time))
    print('Average time checking one configuration', np.mean(validation_times))


this_file = os.path.abspath(__file__)
this_directory = os.path.dirname(this_file)
configuration_space_path = os.path.join(this_directory, '..',
                                        "test", "test_searchspaces")
configuration_space_path = os.path.abspath(configuration_space_path)
pcs_files = os.listdir(configuration_space_path)

for pcs_file in pcs_files:
    if '.pcs' in pcs_file:
        full_path = os.path.join(configuration_space_path, pcs_file)
        run_test(full_path)
