from collections import OrderedDict

import hpconvnet
import hpconvnet.cifar10
import hyperopt

import cStringIO

space = hpconvnet.cifar10.build_search_space(max_n_features=4500,
                                             bagging_fraction=0.5,
                                             n_unsup=2000,
                                             abort_on_rows_larger_than=50 * 1000,
                                             output_sizes=(32, 64))["pipeline"]

