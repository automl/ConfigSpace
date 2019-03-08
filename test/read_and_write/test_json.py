import os
import unittest

from ConfigSpace.read_and_write.json import read, write
from ConfigSpace.read_and_write.pcs import read as read_pcs
from ConfigSpace.read_and_write.pcs_new import read as read_pcs_new
from ConfigSpace import (
    ForbiddenInClause,
    CategoricalHyperparameter,
    ConfigurationSpace,
)


class TestJson(unittest.TestCase):

    def test_serialize_forbidden_in_clause(self):
        cs = ConfigurationSpace()
        a = cs.add_hyperparameter(CategoricalHyperparameter('a', [0, 1, 2]))
        cs.add_forbidden_clause(ForbiddenInClause(a, [1, 2]))
        write(cs)

    def test_round_trip(self):
        this_file = os.path.abspath(__file__)
        this_directory = os.path.dirname(this_file)
        configuration_space_path = os.path.join(this_directory,
                                                "..", "test_searchspaces")
        configuration_space_path = os.path.abspath(configuration_space_path)
        pcs_files = os.listdir(configuration_space_path)

        for pcs_file in sorted(pcs_files):

            if '.pcs' in pcs_file:
                full_path = os.path.join(configuration_space_path, pcs_file)

                with open(full_path) as fh:
                    cs_string = fh.read().split('\n')
                try:
                    cs = read_pcs(cs_string)
                except Exception:
                    cs = read_pcs_new(cs_string)

                cs.name = pcs_file

                json_string = write(cs)
                new_cs = read(json_string)

                self.assertEqual(new_cs, cs)
