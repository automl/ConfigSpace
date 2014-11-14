"""
Neural Network (NNet) and Deep Belief Network (DBN) search spaces used in [1]
and [2].

The functions in this file return pyll graphs that can be used as the `space`
argument to e.g. `hyperopt.fmin`. The pyll graphs include hyperparameter
constructs (e.g. `hyperopt.hp.uniform`) so `hyperopt.fmin` can perform
hyperparameter optimization.

See ./skdata_learning_algo.py for example usage of these functions.


[1] Bergstra, J.,  Bardenet, R., Bengio, Y., Kegl, B. (2011). Algorithms
for Hyper-parameter optimization, NIPS 2011.

[2] Bergstra, J., Bengio, Y. (2012). Random Search for Hyper-Parameter
Optimization, JMLR 13:281--305.

"""

"""
CHANGED TO WORK AS SEARCHSPACE IN THE BBoM Framework
"""

__author__ = "James Bergstra"
__license__ = "BSD-3"

import numpy as np

from hyperopt import hp

import hpnnet.nips2011

space = {'preproc': hp.choice('preproc', [{
                    'preproc' : 0}, {
                    'preproc' : 1, # Column Normalization
                    'colnorm_thresh' : hp.loguniform('colnorm_thresh', np.log(1e-9), np.log(1e-3)),
                    }, {
                    'preproc' : 2, # PCA
                    'pca_energy' : hp.uniform('pca_energy', .5, 1),
                    }]),
        'nhid1': hp.qloguniform('nhid1', np.log(16), np.log(1024), q=16),
        'dist1': hp.choice('dist1', [0, 1]), # 0 = Uniform, 1 = Normal
        'scale_heur1': hp.choice('scale_heur1', [{
                       'scale_heur1' : 0,   # Old
                       'scale_mult1': hp.uniform('scale_mult1', .2, 2)}, {
                       'scale_heur1': 1}]), # Glorot
        'squash' : hp.choice('squash', [0, 1]), # ['tanh', 'logistic']
        'iseed': hp.choice('iseed', [0, 1, 2, 3]), # Are 5, 6, 7, 8
        'batch_size': hp.choice('batch_size', [0, 1]), # 20 or 100
        'lr': hp.lognormal('lr', np.log(.01), 3.),
        'lr_anneal_start': hp.qloguniform('lr_anneal_start', np.log(100), np.log(10000), q=1),
        'l2_penalty': hp.choice('l2_penalty', [{
        'l2_penalty' : 0}, { # Zero
        'l2_penalty' : 1,    # notzero
        'l2_penalty_nz': hp.lognormal('l2_penalty_nz', np.log(1.0e-6), 2.)}])
        }

