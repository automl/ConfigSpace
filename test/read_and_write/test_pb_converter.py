# Copyright (c) 2014-2016, ConfigSpace developers
# Matthias Feurer
# Katharina Eggensperger
# and others (see commit history).
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the <organization> nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# __authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
# __contact__ = "automl.org"
#
# import unittest
#
# import six
#
# import HPOlibConfigSpace.converters.pb_parser as pb_parser
# from HPOlibConfigSpace.hyperparameters import UniformFloatHyperparameter, \
#     UniformIntegerHyperparameter, CategoricalHyperparameter
#
# float_a = UniformFloatHyperparameter("float_a", -5.3, 10.5)
# int_a = UniformIntegerHyperparameter("int_a", -5, 10)
# cat_a = CategoricalHyperparameter("enum_a", ["5", "a", "b", "@/&%%"])
# crazy = CategoricalHyperparameter("@.:;/\?!$%&_-<>*+1234567890", ["const"])
# easy_space = {"float_a": float_a,
#               "int_a": int_a,
#               "enum_a": cat_a,
#               "@.:;/\?!$%&_-<>*+1234567890": crazy,
#               }
#
#
# class TestPbConverter(unittest.TestCase):
#
#     def test_read_configuration_space_easy(self):
#         expected = StringIO.StringIO()
#         expected.write("""
#         language: PYTHON
#         name: "HPOlib.cv"
#
#         variable {
#             name: "float_a"
#             type: FLOAT
#             size: 1
#             min:  -5.3
#             max:  10.5
#         }
#
#         variable {
#             name: "enum_a"
#             type: ENUM
#             size: 1
#             options: "5"
#             options: "a"
#             options: "b"
#             options: "@/&%%"
#         }
#
#          variable {
#             name: "int_a"
#             type: INT
#             size: 1
#             min:  -5
#             max:  10
#          }
#
#          variable {
#             name: "@.:;/\?!$%&_-<>*+1234567890"
#             type: ENUM
#             size: 1
#             options: "const"
#         }
#         """)
#
#         cs = pb_parser.read(expected.getvalue())
#         self.assertDictEqual(cs, easy_space)
