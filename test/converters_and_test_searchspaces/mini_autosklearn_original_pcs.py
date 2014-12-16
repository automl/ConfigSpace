import os
import unittest

import HPOlibConfigSpace
import HPOlibConfigSpace.converters.pcs_parser as pcs_parser
from HPOlibConfigSpace.random_sampler import RandomSampler


class TestPCSConverterOnMiniAutoSklearn(unittest.TestCase):
    # TODO test the other formats once they are ready!

    def test_read_and_write_pcs(self):
        configuration_space_path = os.path.abspath(HPOlibConfigSpace.__file__)
        configuration_space_path = os.path.dirname(configuration_space_path)
        configuration_space_path = os.path.join(configuration_space_path,
                                                "..", "test",
                                                "test_searchspaces",
                                                "mini_autosklearn_original.pcs")

        with open(configuration_space_path) as fh:
            cs = pcs_parser.read(fh)

        pcs = pcs_parser.write(cs)

        with open(configuration_space_path) as fh:
            lines = fh.readlines()

        num_asserts = 0
        for line in lines:
            line = line.replace("\n", "")
            line = line.split("#")[0]       # Remove comments
            line = line.strip()

            if line:
                num_asserts += 1
                self.assertIn(line, pcs)

        self.assertEqual(21, num_asserts)

        # Sample a little bit
        rs = RandomSampler(cs, 1)
        print cs
        for i in range(1000):
            c = rs.sample_configuration()