"""
Deep Belief Network (DBN) search spaces used in [1] and [2].

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
CHANGED TO WORK AS A SEARCHSPACE IN THE BBoM Framework
"""

__author__ = "James Bergstra"
__license__ = "BSD-3"

import numpy as np

from hyperopt import hp

layer1 = { 'depth' : 1,
        'n_hid_0': hp.qloguniform('n_hid_0', np.log(2**7), np.log(2**12), q=16),
        'W_idist_0': hp.choice('W_idist_0', [0, 1]),   # ['uniform', 'normal']
        #The scale heuristic
        'W_ialgo_0': hp.choice('W_ialgo_0', [{
                'W_ialgo_0': 0,     # 'old'
                'W_imult_0': hp.lognormal('W_imult_0', 0, 1)},
                {'W_ialgo_0': 1}]), #'Glorot'
        'cd_lr_0': hp.lognormal('cd_lr_0', np.log(.01), 2),
        'cd_seed_0': hp.randint('cd_seed_0', 10),
        'cd_epochs_0': hp.qloguniform('cd_epochs_0', np.log(1), np.log(3000), q=1),
        'sample_v0s_0': hp.choice('sample_v0s_0', [0, 1]),    # [False, True]
        'lr_anneal_0': hp.qloguniform('lr_anneal_0',
                                               np.log(10), np.log(10000), q=1)}

layer2 = dict(layer1)
layer2.update({'depth' : 2,
        'n_hid_1': hp.qloguniform('n_hid_1', np.log(2**7), np.log(2**12), q=16),
        'W_idist_1': hp.choice('W_idist_1', [0, 1]),      # ['uniform', 'normal']
        'W_ialgo_1': hp.choice('W_ialgo_1', [{
                'W_ialgo_1': 0,     # 'old'
                'W_imult_1': hp.lognormal('W_imult_1', 0, 1)},
                {'W_ialgo_1': 1     # 'Glorot'
                }]),
        'cd_lr_1': hp.lognormal('cd_lr_1', np.log(.01), 2),
        'cd_seed_1': hp.randint('cd_seed_1', 10),
        'cd_epochs_1': hp.qloguniform('cd_epochs_1', np.log(1), np.log(2000), q=1),
        'sample_v0s_1': hp.choice('sample_v0s_1', [0, 1]),      # [False, True]
        'lr_anneal_1': hp.qloguniform('lr_anneal_1',
                                               np.log(10), np.log(10000), q=1)})

layer3 = dict(layer2)
layer3.update({'depth' : 3,
        'n_hid_2': hp.qloguniform('n_hid_2', np.log(2**7), np.log(2**12), q=16),
        'W_idist_2': hp.choice('W_idist_2', [0, 1]),          # ['uniform', 'normal']
        'W_ialgo_2': hp.choice('W_ialgo_2', [{
                'W_ialgo_2': 0,     # 'old'
                'W_imult_2': hp.lognormal('W_imult_2', 0, 1)},
                {'W_ialgo_2': 1     # 'Glorot'
                }]),
        'cd_lr_2': hp.lognormal('cd_lr_2', np.log(.01), 2),
        'cd_seed_2': hp.randint('cd_seed_2', 10),
        'cd_epochs_2': hp.qloguniform('cd_epochs_2', np.log(1), np.log(1500), q=1),
        'sample_v0s_2': hp.choice('sample_v0s_2', [0, 1]),  # [False, True]
        'lr_anneal_2': hp.qloguniform('lr_anneal_2',
                                               np.log(10), np.log(10000), q=1)})

space = {'preproc': hp.choice('preproc', [{
             'preproc': 0 }, {      #'no'
             'preproc': 1,          #'zca'
             'pca_energy': hp.uniform('pca_energy', .5, 1),
             }]),
         # Sqash is fixed at logistic, no need for a hyperparameter
        'iseed': hp.choice('iseed', [0, 1, 2, 3]),      # [5, 6, 7, 8]
                                               
        # This was called nnet_features
        'depth': hp.pchoice('depth', [(0.5, 0),
                                     (0.25, layer1),
                                     (0.125, layer2), 
                                     (0.125, layer3)]), # This is fine
        
        #'nnet_features': hp.pchoice('nnet_features', [(.5, 0), (.25, 1),(.125, 2), (.125, 3)]),
        'batch_size': hp.choice('batch_size', [0, 1]),      # [20, 100]
        'lr': hp.lognormal('lr', np.log(.01), 3.),
        'lr_anneal_start': hp.qloguniform('lr_anneal_start', np.log(100),
                                            np.log(10000),
                                            q=1),
        'l2_penalty': hp.choice('l2_penalty', [{
            'l2_penalty' : 0}, { # Zero
            'l2_penalty' : 1,    # notzero
            'l2_penalty_nz': hp.lognormal('l2_penalty_nz', np.log(1.0e-6), 2.)}])
    }


