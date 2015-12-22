import os
import unittest

import ConfigSpace
import ConfigSpace.io.pcs as pcs_parser


class ExampleSearchSpacesTest(unittest.TestCase):
    pass


def generate(configuration_space_path):
    def run_test(self):
        with open(configuration_space_path) as fh:
            cs = pcs_parser.read(fh)

        cs.seed(1)
        # Sample a little bit
        for i in range(10000):
            c = cs.sample_configuration()
            c.is_valid_configuration()
    return run_test


configuration_space_path = os.path.abspath(ConfigSpace.__file__)
configuration_space_path = os.path.dirname(configuration_space_path)
configuration_space_path = os.path.join(configuration_space_path,
                                        "..", "test",
                                        "test_searchspaces")
pcs_files = os.listdir(configuration_space_path)

for pcs_file in pcs_files:
    if 'spear-params.pcs' in pcs_file:
        pcs_file = os.path.join(configuration_space_path, pcs_file)
        setattr(ExampleSearchSpacesTest, 'test_%s' % pcs_file,
                generate(pcs_file))
