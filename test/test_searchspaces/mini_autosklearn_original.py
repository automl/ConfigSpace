from hyperopt import hp
from hyperopt.pyll import scope

pca = {'preprocessing': 'pca', 'pca:keep_variance': scope.int(
    hp.quniform('pca:keep_variance', 0, 1, 1))}

penalty_and_loss = hp.choice('penalty_and_loss',
                             [{'liblinear:penalty': 'l1', 'liblinear:loss': 'l2'},
                              {'liblinear:penalty': 'l2', 'liblinear:loss': 'l1'},
                              {'liblinear:penalty': 'l2', 'liblinear:loss': 'l2'}])
liblinear_LOG2_C = scope.int(hp.quniform('liblinear:LOG2_C', -5, 15, 1))
liblinear = {'classifier': 'liblinear', 'liblinear:penalty_and_loss': penalty_and_loss, 'liblinear:LOG2_C': liblinear_LOG2_C}

libsvm_LOG2_C = scope.int(hp.quniform('libsvm_svc:LOG2_C', -5, 15, 1))
libsvm_LOG2_gamma = scope.int(hp.quniform('libsvm_svc:LOG2_gamma', -15, 3, 1))
libsvm_svc = {'classifier': 'libsvm_svc', 'libsvm_svc:LOG2_C': libsvm_LOG2_C, 'libsvm_svc:LOG2_gamma': libsvm_LOG2_gamma}
criterion = hp.choice('random_forest:criterion', ['gini', 'entropy'])
max_features = scope.int(hp.quniform('random_forest:max_features', 1, 10, 1))
min_samples_split = scope.int(hp.quniform('random_forest:min_samples_split', 0, 4, 1))
random_forest = {'classifier': 'random_forest', 'random_forest:criterion': criterion, 'random_forest:max_features': max_features, 'random_forest:min_samples_split': min_samples_split}

preprocessors = {'None': 'None', 'pca': pca}
classifiers = {'libsvm_svc': libsvm_svc,
               'liblinear': liblinear,
               'random_forest': random_forest}

space = {'classifier': hp.choice('classifier', classifiers.values()),
         'preprocessing': hp.choice('preprocessing', preprocessors.values())}