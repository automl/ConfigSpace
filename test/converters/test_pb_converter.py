#!/usr/bin/env python

##
# wrapping: A program making it easy to use hyperparameter
# optimization software.
# Copyright (C) 2013 Katharina Eggensperger and Matthias Feurer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"

import StringIO
import unittest

import HPOlibConfigSpace.converters.pb_parser as pb_parser
from HPOlibConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter

float_a = UniformFloatHyperparameter("float_a", -5.3, 10.5)
int_a = UniformIntegerHyperparameter("int_a", -5, 10)
cat_a = CategoricalHyperparameter("enum_a", ["5", "a", "b", "@/&%%"])
crazy = CategoricalHyperparameter("@.:;/\?!$%&_-<>*+1234567890", ["const"])
easy_space = {"float_a": float_a,
              "int_a": int_a,
              "enum_a": cat_a,
              "@.:;/\?!$%&_-<>*+1234567890": crazy,
              }


class TestPbConverter(unittest.TestCase):

    def test_read_configuration_space_easy(self):
        expected = StringIO.StringIO()
        expected.write("""
        language: PYTHON
        name: "HPOlib.cv"

        variable {
            name: "float_a"
            type: FLOAT
            size: 1
            min:  -5.3
            max:  10.5
        }

        variable {
            name: "enum_a"
            type: ENUM
            size: 1
            options: "5"
            options: "a"
            options: "b"
            options: "@/&%%"
        }

         variable {
            name: "int_a"
            type: INT
            size: 1
            min:  -5
            max:  10
         }

         variable {
            name: "@.:;/\?!$%&_-<>*+1234567890"
            type: ENUM
            size: 1
            options: "const"
        }
        """)

        cs = pb_parser.read(expected.getvalue())
        self.assertDictEqual(cs, easy_space)
