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

# import unittest
#
# from hyperopt import hp
# import hyperopt
# import numpy as np
# import six
#
# import HPOlib.format_converter.configuration_space as configuration_space
# import HPOlib.format_converter.pyll_parser as pyll_parser
# from HPOlibConfigSpace.hyperparameters import CategoricalHyperparameter, \
#     UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant, \
#     NormalFloatHyperparameter, NormalIntegerHyperparameter
#
# # More complex search space
# classifier = CategoricalHyperparameter("classifier", ["svm", "nn"])
# kernel = CategoricalHyperparameter("kernel", ["rbf", "linear"],
#     conditions=[["classifier == svm"]])
# C = UniformFloatHyperparameter("C", 0.03125, 32768, base=2,
#     conditions=[["classifier == svm"]])
# gamma = UniformFloatHyperparameter("gamma", 0.000030518, 8, base=2,
#     conditions=[["kernel == rbf"]])
# neurons = UniformIntegerHyperparameter("neurons", 16, 1024, q=16,
#     conditions=[["classifier == nn"]])
# lr = UniformFloatHyperparameter("lr", 0.0001, 1.0,
#     conditions=[["classifier == nn"]])
# preprocessing = CategoricalHyperparameter("preprocessing", [None, "pca"])
# config_space = {"classifier": classifier,
#                 "kernel": kernel,
#                 "C": C,
#                 "gamma": gamma,
#                 "neurons": neurons,
#                 "lr": lr,
#                 "preprocessing": preprocessing}
#
# # A search space where a hyperparameter depends on two others:
# gamma_2 = UniformFloatHyperparameter("gamma_2", 0.000030518, 8, base=2,
#     conditions=[["kernel == rbf", "classifier == svm"]])
#
# config_space_2 = {"classifier": classifier,
#                 "kernel": kernel,
#                 "C": C,
#                 "gamma_2": gamma_2,
#                 "neurons": neurons,
#                 "lr": lr,
#                 "preprocessing": preprocessing}
#
#
# class TestPyllReader(unittest.TestCase):
#     def setUp(self):
#         self.pyll_reader = pyll_parser.PyllReader()
#
#     def test_read_literal(self):
#         literal = hyperopt.pyll.as_apply("5....4....3....1! Off blast! ")
#         ret = self.pyll_reader.read_literal(literal, "pre_chorus")
#         expected = Constant("pre_chorus", "5....4....3....1! Off blast! ")
#         self.assertEqual(expected, ret)
#
#     def test_read_container(self):
#         #### Lists
#         # The Literal is added to the content of the list, but no method will
#         #  add it to the list of found hyperparameters
#         # hyperparameter
#         # 0 pos_args
#         # 1   float
#         # 2     hyperopt_param
#         # 3       Literal{a}
#         # 4       uniform
#         # 5         Literal{0}
#         # 6         Literal{10}
#         # 7   Literal{Alpha}
#         expected = {'a': UniformFloatHyperparameter('a', 0, 10),
#                     'Alpha': Constant('Alpha', 'Alpha')}
#         param = hyperopt.pyll.as_apply([hp.uniform("a", 0, 10), "Alpha"])
#         ret = self.pyll_reader.read_container(param)
#         self.assertEqual(expected, ret)
#         # 0 pos_args
#         # 1   float
#         # 2     hyperopt_param
#         # 3       Literal{a}
#         # 4       uniform
#         # 5         Literal{0}
#         # 6         Literal{10}
#         # 7   float
#         # 8     hyperopt_param
#         # 9       Literal{b}
#         # 10       uniform
#         # 11         Literal{5}
#         # 12         Literal{15}
#         container =  hyperopt.pyll.as_apply([hp.uniform("a", 0, 10),
#                                              hp.uniform("b", 5, 15)])
#         ret = self.pyll_reader.read_container(container)
#         expected = {"a": UniformFloatHyperparameter("a", 0, 10),
#                     "b": UniformFloatHyperparameter("b", 5, 15)}
#         self.assertEqual(expected, ret)
#
#     def test_read_dict(self):
#         #### Dictionaries
#         # 0 dict
#         # 1  Elektronik =
#         # 2   Literal{Supersonik}
#         # 3  a =
#         # 4   float
#         # 5     hyperopt_param
#         # 6       Literal{a}
#         # 7       uniform
#         # 8         Literal{0}
#         # 9         Literal{10}
#         container = hyperopt.pyll.as_apply({"a": hp.uniform("a", 0, 10),
#                                             "Elektronik": "Supersonik"})
#         ret = self.pyll_reader.read_dict(container)
#         expected = {"a": UniformFloatHyperparameter("a", 0, 10),
#                     "Elektronik": Constant("Elektronik", "Supersonik")}
#         self.assertEqual(expected, ret)
#
#         # 0 dict
#         # 1  @1:F:ASPEED:aspeed-opt =
#         # 2   switch
#         # 3     hyperopt_param
#         # 4       Literal{@1:F:ASPEED:aspeed-opt}
#         # 5       randint
#         # 6         Literal{1}
#         # 7     dict
#         # 8      @1:F:ASPEED:aspeed-opt =
#         # 9       Literal{yes}
#         # 10  @1:approach =
#         # 11   Literal{ASPEED}
#         param_6 = hp.choice("@1:F:ASPEED:aspeed-opt", [
#                                 {"@1:F:ASPEED:aspeed-opt": "yes", },
#                                 ])
#         container = hyperopt.pyll.as_apply(
#             {"@1:F:ASPEED:aspeed-opt": param_6,
#              "@1:approach": "ASPEED"})
#         ret = self.pyll_reader.read_dict(container)
#         expected = {"@1:F:ASPEED:aspeed-opt":
#                         CategoricalHyperparameter("@1:F:ASPEED:aspeed-opt", ['yes']),
#                     "@1:approach":
#                         Constant("@1:approach", "ASPEED")}
#         self.maxDiff = None
#         self.assertEqual(expected, ret)
#
#     def test_read_switch(self):
#         # 0 switch
#         # 1   hyperopt_param
#         # 2     Literal{dist1}
#         # 3     randint
#         # 4       Literal{2}
#         # 5   Literal{uniform}
#         # 6   Literal{normal}
#         dist = hp.choice('dist1', ['uniform', 'normal'])
#         ret = self.pyll_reader.read_switch(dist)
#         expected = CategoricalHyperparameter('dist1', ['uniform', 'normal'])
#         self.assertEqual(expected, ret)
#
#         bigger_choice = hp.choice('choice', [
#             {'choice': "zero", 'a': 0, 'b': hp.uniform('b', 0, 10)},
#             {'choice': "other", 'a': 1, 'b': hp.uniform('b', 0, 10)}])
#         ret = self.pyll_reader.read_switch(bigger_choice)
#         expected = CategoricalHyperparameter('choice', ['zero', 'other'])
#         self.assertEqual(expected, ret)
#         self.assertEqual(2, len(self.pyll_reader.constants))
#         # Only the hyperparameter b is put into pyll_reader.hyperparameters
#         self.assertEqual(1, len(self.pyll_reader.hyperparameters))
#
#     # TODO: duplicate these tests for Integer/care about integers + test if
#     # the warning of non-uniform parameters is actually printed
#     def test_read_uniform(self):
#         # 0 float
#         # 1   hyperopt_param
#         # 2     Literal{scale_mult1}
#         # 3     uniform
#         # 4       Literal{0.2}
#         # 5       Literal{2}
#         uniform = hp.uniform('scale_mult1', .2, 2).inputs()[0].inputs()[1]
#         ret = self.pyll_reader.read_uniform(uniform, 'scale_mult1')
#         expected = UniformFloatHyperparameter('scale_mult1', 0.2, 2)
#         self.assertEqual(expected, ret)
#
#     def test_read_loguniform(self):
#         # 0 float
#         # 1   hyperopt_param
#         # 2     Literal{colnorm_thresh}
#         # 3     loguniform
#         # 4       Literal{-20.7232658369}
#         # 5       Literal{-6.90775527898}
#         loguniform = hp.loguniform('colnorm_thresh', np.log(1e-9),
#             np.log(1e-3)).inputs()[0].inputs()[1]
#         ret = self.pyll_reader.read_loguniform(loguniform, 'colnorm_thresh')
#         expected = UniformFloatHyperparameter(
#             'colnorm_thresh', 1e-9, 1e-3, base=np.e)
#         self.assertEqual(expected, ret)
#
#     def test_read_quniform(self):
#         # TODO scope.int(hp.quniform('liblinear:LOG2_C', -5, 15, 1))
#         # 0 float
#         # 1   hyperopt_param
#         # 2     Literal{l0eg_fsize}
#         # 3     quniform
#         # 4       Literal{2.50001}
#         # 5       Literal{8.5}
#         # 6       Literal{1}
#         quniform = hp.quniform('l0eg_fsize', 2.50001, 8.5, 1). \
#             inputs()[0].inputs()[1]
#         ret = self.pyll_reader.read_quniform(quniform, 'l0eg_fsize')
#         expected = UniformIntegerHyperparameter(
#             'l0eg_fsize', 3, 8)
#         self.assertEqual(expected, ret)
#
#         l2_out_lp_psize = hp.quniform("l2_out_lp_psize", 0.50001, 5.5, 1). \
#             inputs()[0].inputs()[1]
#         ret = self.pyll_reader.read_quniform(l2_out_lp_psize, "l2_out_lp_psize")
#         expected = UniformIntegerHyperparameter(
#             "l2_out_lp_psize", 1, 5)
#         self.assertEqual(expected, ret)
#
#     def test_read_qloguniform(self):
#         # 0 float
#         # 1   hyperopt_param
#         # 2     Literal{nhid1}
#         # 3     qloguniform
#         # 4       Literal{2.77258872224}
#         # 5       Literal{6.9314718056}
#         # 6      q =
#         # 7       Literal{16}
#         qloguniform = hp.qloguniform('nhid1', np.log(16), np.log(1024), q=16). \
#             inputs()[0].inputs()[1]
#         ret = self.pyll_reader.read_qloguniform(qloguniform, 'nhid1')
#         expected = UniformFloatHyperparameter(
#             'nhid1', 16, 1024, q=16, base=np.e)
#         self.assertEqual(expected, ret)
#
#     def test_read_normal(self):
#         # 0 float
#         # 1   hyperopt_param
#         # 2     Literal{l0eg_alpha}
#         # 3     normal
#         # 4       Literal{0.0}
#         # 5       Literal{1.0}
#         normal = hp.normal("l0eg_alpha", 0.0, 1.0).inputs()[0].inputs()[1]
#         ret = self.pyll_reader.read_normal(normal, "l0eg_alpha")
#         expected = NormalFloatHyperparameter("l0eg_alpha", 0.0, 1.0)
#         self.assertEqual(expected, ret)
#
#
#     def test_read_lognormal(self):
#         # 0 float
#         # 1   hyperopt_param
#         # 2     Literal{lr}
#         # 3     lognormal
#         # 4       Literal{-4.60517018599}
#         # 5       Literal{3.0}
#         lognormal = hp.lognormal('lr', np.log(.01), 3.).inputs()[0].inputs()[1]
#         ret = self.pyll_reader.read_lognormal(lognormal, "lr")
#         expected = NormalFloatHyperparameter("lr", np.log(0.01), 3.0, base=np.e)
#         self.assertEqual(expected, ret)
#
#     def test_read_qnormal(self):
#         # 0 float
#         # 1   hyperopt_param
#         # 2     Literal{qnormal}
#         # 3     qnormal
#         # 4       Literal{0.0}
#         # 5       Literal{1.0}
#         # 6       Literal{0.5}
#         qnormal = hp.qnormal("qnormal", 0.0, 1.0, 0.5).inputs()[0].inputs()[1]
#         ret = self.pyll_reader.read_qnormal(qnormal, "qnormal")
#         expected = NormalFloatHyperparameter("qnormal", 0.0, 1.0, q=0.5)
#         self.assertEqual(expected, ret)
#
#     def test_read_qlognormal(self):
#         # 0 float
#         # 1   hyperopt_param
#         # 2     Literal{qlognormal}
#         # 3     qlognormal
#         # 4       Literal{0.0}
#         # 5       Literal{1.0}
#         # 6       Literal{0.5}
#         qlognormal = hp.qlognormal("qlognormal", 0.0, 1.0, 0.5).inputs()[0].inputs()[1]
#         ret = self.pyll_reader.read_qlognormal(qlognormal, "qlognormal")
#         expected = NormalFloatHyperparameter("qlognormal", 0.0, 1.0, q=0.5, base=np.e)
#         self.assertEqual(expected, ret)
#
#         qlognormal = hp.qlognormal("qlognormal", 1.0, 5.0, 1.0).inputs()[0].inputs()[1]
#         ret = self.pyll_reader.read_qlognormal(qlognormal, "qlognormal")
#         expected = NormalIntegerHyperparameter("qlognormal", 1.0, 5.0, base=np.e)
#         self.assertEqual(expected, ret)
#
#
# class TestPyllWriter(unittest.TestCase):
#     def setUp(self):
#         self.pyll_writer = pyll_parser.PyllWriter()
#
#     def test_convert_configuration_space(self):
#         a = UniformFloatHyperparameter("a", 0, 1)
#         b = UniformFloatHyperparameter("b", 0, 3, q=0.1)
#
#         expected = six.StringIO()
#         expected.write('from hyperopt import hp\nimport hyperopt.pyll as pyll')
#         expected.write('\n\n')
#         expected.write('param_0 = hp.uniform("a", 0.0, 1.0)\n')
#         expected.write('param_1 = hp.quniform("b", -0.0499, 3.05, 0.1)\n\n')
#         expected.write('space = {"a": param_0, "b": param_1}\n')
#         simple_space = {"a": a, "b": b}
#         cs = self.pyll_writer.write(simple_space)
#         self.assertEqual(expected.getvalue(), cs)
#
#     def test_convert_conditional_space(self):
#         a_or_b = CategoricalHyperparameter("a_or_b", ["a", "b"])
#         cond_a = UniformFloatHyperparameter(
#             'cond_a', 0, 1, conditions=[['a_or_b == a']])
#         cond_b = UniformFloatHyperparameter(
#             'cond_b', 0, 3, q=0.1, conditions=[['a_or_b == b']])
#         conditional_space = {"a_or_b": a_or_b, "cond_a": cond_a, "cond_b": cond_b}
#         cs = self.pyll_writer.write(conditional_space)
#         expected = six.StringIO()
#         expected.write('from hyperopt import hp\nimport hyperopt.pyll as pyll')
#         expected.write('\n\n')
#         expected.write('param_0 = hp.uniform("cond_a", 0.0, 1.0)\n')
#         expected.write('param_1 = hp.quniform("cond_b", -0.0499, 3.05, 0.1)\n')
#         expected.write('param_2 = hp.choice("a_or_b", [\n')
#         expected.write('    {"a_or_b": "a", "cond_a": param_0, },\n')
#         expected.write('    {"a_or_b": "b", "cond_b": param_1, },\n')
#         expected.write('    ])\n\n')
#         expected.write('space = {"a_or_b": param_2}\n')
#         self.assertEqual(expected.getvalue(), cs)
#
#     def test_convert_complex_space(self):
#         cs = self.pyll_writer.write(config_space)
#         expected = six.StringIO()
#         expected.write('from hyperopt import hp\nimport hyperopt.pyll as pyll')
#         expected.write('\n\n')
#         expected.write('param_0 = hp.uniform("LOG2_C", -5.0, 15.0)\n')
#         expected.write('param_1 = hp.uniform("LOG2_gamma", -14.9999800563, '
#                        '3.0)\n')
#         expected.write('param_2 = hp.choice("kernel", [\n')
#         expected.write('    {"kernel": "linear", },\n')
#         expected.write('    {"kernel": "rbf", "LOG2_gamma": param_1, },\n')
#         expected.write('    ])\n')
#         expected.write('param_3 = hp.uniform("lr", 0.0001, 1.0)\n')
#         expected.write('param_4 = pyll.scope.int(hp.quniform('
#                        '"neurons", 15.50001, 1024.5, 16.0))\n')
#         expected.write('param_5 = hp.choice("classifier", [\n')
#         expected.write('    {"classifier": "nn", "lr": param_3, "neurons": '
#                        'param_4, },\n')
#         expected.write('    {"classifier": "svm", "LOG2_C": param_0, '
#                        '"kernel": param_2, },\n')
#         expected.write('    ])\n')
#         expected.write('param_6 = hp.choice("preprocessing", [\n')
#         expected.write('    {"preprocessing": "None", },\n')
#         expected.write('    {"preprocessing": "pca", },\n')
#         expected.write('    ])\n\n')
#         expected.write('space = {"classifier": param_5, '
#                        '"preprocessing": param_6}\n')
#         self.assertEqual(expected.getvalue(), cs)
#
#         self.pyll_writer.reset_hyperparameter_countr()
#         expected.seek(0)
#         cs = self.pyll_writer.write(config_space_2)
#         self.assertEqual(expected.getvalue().replace("gamma", "gamma_2"), cs)
#
#     def test_operator_in(self):
#         a_or_b = CategoricalHyperparameter("a_or_b", ["a", "b"])
#         cond_a = UniformFloatHyperparameter(
#             'cond_a', 0, 1, conditions=[['a_or_b == a']])
#         cond_b = UniformFloatHyperparameter(
#             'cond_b', 0, 3, q=0.1, conditions=[['a_or_b == b']])
#         e = UniformFloatHyperparameter("e", 0, 5,
#                                      conditions=[['a_or_b in {a,b}']])
#         conditional_space_operator_in = {"a_or_b": a_or_b, "cond_a": cond_a,
#                                  "cond_b": cond_b, "e": e}
#         cs = self.pyll_writer.write(conditional_space_operator_in)
#         expected = six.StringIO()
#         expected.write('from hyperopt import hp\nimport hyperopt.pyll as pyll')
#         expected.write('\n\n')
#         expected.write('param_0 = hp.uniform("cond_a", 0.0, 1.0)\n')
#         expected.write('param_1 = hp.quniform("cond_b", -0.0499, 3.05, 0.1)\n')
#         expected.write('param_2 = hp.uniform("e", 0.0, 5.0)\n')
#         expected.write('param_3 = hp.choice("a_or_b", [\n')
#         expected.write('    {"a_or_b": "a", "cond_a": param_0, "e": param_2, '
#                        '},\n')
#         expected.write('    {"a_or_b": "b", "cond_b": param_1, "e": param_2, '
#                        '},\n')
#         expected.write('    ])\n\n')
#         expected.write('space = {"a_or_b": param_3}\n')
#         self.assertEqual(expected.getvalue(), cs)
#
#     def test_write_uniform(self):
#         a = UniformFloatHyperparameter("a", 0, 1)
#         expected = ('a', 'param_0 = hp.uniform("a", 0.0, 1.0)')
#         value = self.pyll_writer.write_hyperparameter(a, None)
#         self.assertEqual(expected, value)
#
#         # The hyperparameter name has to be converted seperately because
#         # otherwise the parameter values are converted at object costruction
#         # time
#         a = UniformFloatHyperparameter("a", 1, 10, base=10)
#         a.name = self.pyll_writer.convert_name(a)
#         expected = ('LOG10_a', 'param_1 = hp.uniform("LOG10_a", 0.0, 1.0)')
#         value = self.pyll_writer.write_hyperparameter(a, None)
#         self.assertEqual(expected, value)
#
#         nhid1 = UniformFloatHyperparameter(
#             "nhid1", 16, 1024, q=16, base=np.e)
#         expected = ('nhid1', 'param_2 = hp.qloguniform('
#                     '"nhid1", 2.0794540416, 6.93925394604, 16.0)')
#         value = self.pyll_writer.write_hyperparameter(nhid1, None)
#         self.assertEqual(expected, value)
#
#     def test_write_uniform_int(self):
#         a_int = UniformIntegerHyperparameter("a_int", 0, 1)
#         expected = ('a_int', 'param_0 = pyll.scope.int(hp.quniform('
#                              '"a_int", -0.49999, 1.5, 1.0))')
#         value = self.pyll_writer.write_hyperparameter(a_int, None)
#         self.assertEqual(expected, value)
#
#         # Test for the problem that if a parameter has Q not None and is on
#         # log scale, the Q must not be in the hp object, but the
#         # hyperparameter name. If this is done the other way round,
#         # the log-value of the hyperparameter is quantized
#         a_int = UniformIntegerHyperparameter(
#             "a_int", 1, 1000, base=10)
#         a_int.name = self.pyll_writer.convert_name(a_int)
#         expected = ('LOG10_Q1_a_int', 'param_1 = hp.uniform('
#                     '"LOG10_Q1_a_int", -0.301021309861, 3.00021709297)')
#         value = self.pyll_writer.write_hyperparameter(a_int, None)
#         self.assertEqual(expected, value)
#
#     def test_write_quniform(self):
#         b = UniformFloatHyperparameter("b", 0, 3, q=0.1)
#         expected = ("b", 'param_0 = hp.quniform("b", -0.0499, 3.05, 0.1)')
#         value = self.pyll_writer.write_hyperparameter(b, None)
#         self.assertEqual(expected, value)
#
#         b = UniformFloatHyperparameter(
#             "b", 0.1, 3, q=0.1, base=10)
#         b.name = self.pyll_writer.convert_name(b)
#         expected = ('LOG10_Q0.100000_b', 'param_1 = hp.uniform('
#                     '"LOG10_Q0.100000_b", -1.30016227413, 0.484299839347)')
#         value = self.pyll_writer.write_hyperparameter(b, None)
#         self.assertEqual(expected, value)
#
#     def test_write_quniform_int(self):
#         b_int_1 = UniformIntegerHyperparameter("b_int", 0, 3, q=1.0)
#         expected = ("b_int", 'param_0 = pyll.scope.int(hp.quniform('
#                     '"b_int", -0.49999, 3.5, 1.0))')
#         value = self.pyll_writer.write_hyperparameter(b_int_1, None)
#         self.assertEqual(expected, value)
#
#         # TODO: trying to add the same parameter name a second time, maybe an
#         #  error should be raised!
#         b_int_2 = UniformIntegerHyperparameter("b_int", 0, 3, q=2.0)
#         expected = ("b_int", 'param_1 = pyll.scope.int(hp.quniform('
#                     '"b_int", -0.49999, 3.5, 2.0))')
#         value = self.pyll_writer.write_hyperparameter(b_int_2, None)
#         self.assertEqual(expected, value)
#
#         b_int_3 = UniformIntegerHyperparameter( "b_int", 1, 100, base=10)
#         b_int_3.name = self.pyll_writer.convert_name(b_int_3)
#         # TODO: this is an example of non-uniform integer sampling!
#         expected = ('LOG10_Q1_b_int', 'param_1 = hp.uniform('
#                     '"LOG10_Q1_b_int", -0.301021309861, 2.00216606176)')
#         value = self.pyll_writer.write_hyperparameter(b_int_3, None)
#         self.assertEqual(expected, value)
#
#     def test_write_loguniform(self):
#         c = UniformFloatHyperparameter("c", 0.001, 1, base=np.e)
#         expected = ("c", 'param_0 = hp.loguniform("c", -6.90775527898, 0.0)')
#         value = self.pyll_writer.write_hyperparameter(c, None)
#         self.assertEqual(expected, value)
#
#     def test_write_loguniform_int(self):
#         c_int = UniformIntegerHyperparameter("c_int", 1, 10, base=np.e)
#         expected = ("c_int", 'param_0 = pyll.scope.int(hp.qloguniform('
#                     '"c_int", -0.69312718076, 2.35137525716, 1.0))')
#         value = self.pyll_writer.write_hyperparameter(c_int, None)
#         self.assertEqual(expected, value)
#
#     def test_write_qloguniform(self):
#         d = UniformFloatHyperparameter("d", 0.1, 3, q=0.1, base=np.e)
#         expected = ("d", 'param_0 = hp.qloguniform("d", -2.99373427089, '
#                          '1.11514159062, 0.1)')
#         value = self.pyll_writer.write_hyperparameter(d, None)
#         self.assertEqual(expected, value)
#
#     def test_write_qloguniform_int(self):
#         d_int_1 = UniformIntegerHyperparameter("d_int", 1, 3, q=1.0, base=np.e)
#         expected = ("d_int", 'param_0 = pyll.scope.int(hp.qloguniform('
#                     '"d_int", -0.69312718076, 1.2527629685, 1.0))')
#         value = self.pyll_writer.write_hyperparameter(d_int_1, None)
#         self.assertEqual(expected, value)
#
#         d_int_2 = UniformIntegerHyperparameter("d_int", 1, 3, q=2.0, base=np.e)
#         expected = ("d_int", 'param_1 = pyll.scope.int(hp.qloguniform('
#                     '"d_int", -0.69312718076, 1.2527629685, 2.0))')
#         value = self.pyll_writer.write_hyperparameter(d_int_2, None)
#         self.assertEqual(expected, value)
#
#     def test_write_normal(self):
#         parameter = NormalFloatHyperparameter("e", 0, 1)
#         expected = ('e', 'param_0 = hp.normal("e", 0.0, 1.0)')
#         value = self.pyll_writer.write_hyperparameter(parameter, None)
#         self.assertEqual(expected, value)
#
#         parameter = NormalFloatHyperparameter("e", 0, 1, base=10)
#         parameter.name = self.pyll_writer.convert_name(parameter)
#         expected = ('LOG10_e', 'param_1 = hp.normal("LOG10_e", 0.0, 1.0)')
#         value = self.pyll_writer.write_hyperparameter(parameter, None)
#         self.assertEqual(expected, value)
#
#     def test_write_normal_int(self):
#         parameter = NormalIntegerHyperparameter("e", 0, 1)
#         expected = ('e',
#                     'param_0 = pyll.scope.int(hp.qnormal("e", 0.0, 1.0, 1.0))')
#         value = self.pyll_writer.write_hyperparameter(parameter, None)
#         self.assertEqual(expected, value)
#
#         parameter = NormalIntegerHyperparameter("e", 0, 1, base=10)
#         parameter.name = self.pyll_writer.convert_name(parameter)
#         # TODO: this is an example of non-uniform sampling
#         expected = ('LOG10_Q1_e', 'param_1 = hp.normal("LOG10_Q1_e", 0.0, 1.0)')
#         value = self.pyll_writer.write_hyperparameter(parameter, None)
#         self.assertEqual(expected, value)
#
#     def test_write_qnormal(self):
#         parameter = NormalFloatHyperparameter("f", 0, 1, q=0.1)
#         expected = ('f', 'param_0 = hp.qnormal("f", 0.0, 1.0, 0.1)')
#         value = self.pyll_writer.write_hyperparameter(parameter, None)
#         self.assertEqual(expected, value)
#
#         parameter = NormalFloatHyperparameter("f", 0, 1, q=0.1, base=10)
#         parameter.name = self.pyll_writer.convert_name(parameter)
#         expected = ('LOG10_Q0.100000_f',
#                     'param_1 = hp.normal("LOG10_Q0.100000_f", 0.0, 1.0)')
#         value = self.pyll_writer.write_hyperparameter(parameter, None)
#         self.assertEqual(expected, value)
#
#     def test_write_qnormal_int(self):
#         parameter = NormalIntegerHyperparameter("f_int", 0, 1, q=1.0)
#         expected = ('f_int',
#                     'param_0 = pyll.scope.int(hp.qnormal("f_int", 0.0, 1.0, 1.0))')
#         value = self.pyll_writer.write_hyperparameter(parameter, None)
#         self.assertEqual(expected, value)
#
#         parameter = NormalIntegerHyperparameter("f_int", 0, 1, q=1.0, base=10)
#         parameter.name = self.pyll_writer.convert_name(parameter)
#         expected = ('LOG10_Q1.000000_f_int',
#                     'param_1 = hp.normal("LOG10_Q1.000000_f_int", 0.0, 1.0)')
#         value = self.pyll_writer.write_hyperparameter(parameter, None)
#         self.assertEqual(expected, value)
#
#     def test_write_lognormal(self):
#         parameter = NormalFloatHyperparameter("g", 0, 1, base=np.e)
#         expected = ('g', 'param_0 = hp.lognormal("g", 0.0, 1.0)')
#         value = self.pyll_writer.write_hyperparameter(parameter, None)
#         self.assertEqual(expected, value)
#
#     def test_write_lognormal_int(self):
#         parameter = NormalIntegerHyperparameter("g", 0, 1, base=np.e)
#         expected = ('g',
#                     'param_0 = pyll.scope.int(hp.qlognormal("g", 0.0, 1.0, 1.0))')
#         value = self.pyll_writer.write_hyperparameter(parameter, None)
#         self.assertEqual(expected, value)
#
#     def test_write_qlognormal(self):
#         parameter = NormalFloatHyperparameter("g", 0, 1, q=0.1, base=np.e)
#         expected = ('g', 'param_0 = hp.qlognormal("g", 0.0, 1.0, 0.1)')
#         value = self.pyll_writer.write_hyperparameter(parameter, None)
#         self.assertEqual(expected, value)
#
#     def test_write_qlognormal_int(self):
#         parameter = NormalIntegerHyperparameter("g_int", 0, 10, q=2.0, base=np.e)
#         expected = ('g_int',
#                     'param_0 = pyll.scope.int(hp.qlognormal("g_int", 0.0, 10.0, 2.0))')
#         value = self.pyll_writer.write_hyperparameter(parameter, None)
#         self.assertEqual(expected, value)
#
#     def test_get_bounds_as_exponent(self):
#         parameter = UniformFloatHyperparameter('a', 1, 1000, base=10)
#         lower, upper = self.pyll_writer.get_bounds_as_exponent(parameter)
#         name = self.pyll_writer.convert_name(parameter)
#         self.assertEqual(name, 'LOG10_a')
#         self.assertEqual(lower, 0)
#         self.assertEqual(upper, 3)
#
#         parameter = UniformFloatHyperparameter('a', 2, 128, base=2)
#         name = self.pyll_writer.convert_name(parameter)
#         lower, upper = self.pyll_writer.get_bounds_as_exponent(parameter)
#         self.assertEqual(name, 'LOG2_a')
#         self.assertEqual(lower, 1)
#         self.assertEqual(upper, 7)
#
#         parameter = UniformFloatHyperparameter('a', 128, 256, base=np.e)
#         name = self.pyll_writer.convert_name(parameter)
#         lower, upper = self.pyll_writer.get_bounds_as_exponent(parameter)
#         self.assertEqual(name, 'LOG_a')
#         self.assertAlmostEqual(lower, 4.852030264)
#         self.assertAlmostEqual(upper, 5.545177444)
#
#         parameter = UniformFloatHyperparameter('a', 10, 1000, base=5)
#         lower, upper = self.pyll_writer.get_bounds_as_exponent(parameter)
#         name = self.pyll_writer.convert_name(parameter)
#         self.assertEqual(name, 'LOG5_a')
#         self.assertAlmostEqual(lower, 1.430676558)
#         self.assertAlmostEqual(upper, 4.292029674)
#
#         parameter = UniformFloatHyperparameter('illegal',
#             0, 1000, base=10)
#         self.assertRaises(ValueError, self.pyll_writer.get_bounds_as_exponent,
#                           parameter)
