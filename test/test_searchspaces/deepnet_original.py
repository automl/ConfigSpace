from hyperopt import hp
from hyperopt.pyll import scope
import hyperopt


depth = hp.quniform('depth', 0, 5, 1)

num_units_1 = hp.quniform('num_units_1', 1, 20, 1)
num_units_2 = hp.quniform('num_units_2', 1, 20, 1)
num_units_3 = hp.quniform('num_units_3', 1, 20, 1)
num_units_4 = hp.quniform('num_units_4', 1, 20, 1)
num_units_5 = hp.quniform('num_units_5', 1, 20, 1)

log_base_epsilon_0 = hp.uniform('log_base_epsilon_0', -11.512925464970229, 0)
log_base_epsilon_1 = hp.uniform('log_base_epsilon_1', -11.512925464970229, 0)
log_base_epsilon_2 = hp.uniform('log_base_epsilon_2', -11.512925464970229, 0)
log_base_epsilon_3 = hp.uniform('log_base_epsilon_3', -11.512925464970229, 0)
log_base_epsilon_4 = hp.uniform('log_base_epsilon_4', -11.512925464970229, 0)
log_base_epsilon_5 = hp.uniform('log_base_epsilon_5', -11.512925464970229, 0)

weight_norm_0 = hp.uniform('weight_norm_0', 0.25, 8)
weight_norm_1 = hp.uniform('weight_norm_1', 0.25, 8)
weight_norm_2 = hp.uniform('weight_norm_2', 0.25, 8)
weight_norm_3 = hp.uniform('weight_norm_3', 0.25, 8)
weight_norm_4 = hp.uniform('weight_norm_4', 0.25, 8)
weight_norm_5 = hp.uniform('weight_norm_5', 0.25, 8)

dropout_0 = hp.uniform('dropout_0', 0, 0.8)
dropout_1 = hp.uniform('dropout_1', 0, 0.8)
dropout_2 = hp.uniform('dropout_2', 0, 0.8)
dropout_3 = hp.uniform('dropout_3', 0, 0.8)
dropout_4 = hp.uniform('dropout_4', 0, 0.8)
dropout_5 = hp.uniform('dropout_5', 0, 0.8)

space = scope.switch(scope.int(depth),
    {'depth': 0, 'log_base_epsilon_0': log_base_epsilon_0, 'weight_norm_0': weight_norm_0, 'dropout_0': dropout_0},

    {'depth': 1, 'log_base_epsilon_0': log_base_epsilon_0, 'weight_norm_0': weight_norm_0, 'dropout_0': dropout_0,
                     'log_base_epsilon_1': log_base_epsilon_1, 'weight_norm_1': weight_norm_1, 'dropout_1': dropout_1, 'num_units_1': num_units_1},

    {'depth': 2, 'log_base_epsilon_0': log_base_epsilon_0, 'weight_norm_0': weight_norm_0, 'dropout_0': dropout_0, 
                     'log_base_epsilon_1': log_base_epsilon_1, 'weight_norm_1': weight_norm_1, 'dropout_1': dropout_1, 'num_units_1': num_units_1,
                     'log_base_epsilon_2': log_base_epsilon_2, 'weight_norm_2': weight_norm_2, 'dropout_2': dropout_2, 'num_units_2': num_units_2},

    {'depth': 3, 'log_base_epsilon_0': log_base_epsilon_0, 'weight_norm_0': weight_norm_0, 'dropout_0': dropout_0, 
                     'log_base_epsilon_1': log_base_epsilon_1, 'weight_norm_1': weight_norm_1, 'dropout_1': dropout_1, 'num_units_1': num_units_1,
                     'log_base_epsilon_2': log_base_epsilon_2, 'weight_norm_2': weight_norm_2, 'dropout_2': dropout_2, 'num_units_2': num_units_2,
                     'log_base_epsilon_3': log_base_epsilon_3, 'weight_norm_3': weight_norm_3, 'dropout_3': dropout_3, 'num_units_3': num_units_3},

    {'depth': 4, 'log_base_epsilon_0': log_base_epsilon_0, 'weight_norm_0': weight_norm_0, 'dropout_0': dropout_0, 
                     'log_base_epsilon_1': log_base_epsilon_1, 'weight_norm_1': weight_norm_1, 'dropout_1': dropout_1, 'num_units_1': num_units_1,
                     'log_base_epsilon_2': log_base_epsilon_2, 'weight_norm_2': weight_norm_2, 'dropout_2': dropout_2, 'num_units_2': num_units_2,
                     'log_base_epsilon_3': log_base_epsilon_3, 'weight_norm_3': weight_norm_3, 'dropout_3': dropout_3, 'num_units_3': num_units_3,
                     'log_base_epsilon_4': log_base_epsilon_4, 'weight_norm_4': weight_norm_4, 'dropout_4': dropout_4, 'num_units_4': num_units_4},

    {'depth': 5, 'log_base_epsilon_0': log_base_epsilon_0, 'weight_norm_0': weight_norm_0, 'dropout_0': dropout_0, 
                     'log_base_epsilon_1': log_base_epsilon_1, 'weight_norm_1': weight_norm_1, 'dropout_1': dropout_1, 'num_units_1': num_units_1,
                     'log_base_epsilon_2': log_base_epsilon_2, 'weight_norm_2': weight_norm_2, 'dropout_2': dropout_2, 'num_units_2': num_units_2,
                     'log_base_epsilon_3': log_base_epsilon_3, 'weight_norm_3': weight_norm_3, 'dropout_3': dropout_3, 'num_units_3': num_units_3,
                     'log_base_epsilon_4': log_base_epsilon_4, 'weight_norm_4': weight_norm_4, 'dropout_4': dropout_4, 'num_units_4': num_units_4,
                     'log_base_epsilon_5': log_base_epsilon_5, 'weight_norm_5': weight_norm_5, 'dropout_5': dropout_5, 'num_units_5': num_units_5},
    )

