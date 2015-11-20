import copy

import numpy as np

from ConfigSpace import Configuration


def impute_inactive_values(configuration, strategy='default'):
    """Impute inactive parameters.

    Parameters
    ----------
    strategy : string, optional (default='default')
        The imputation strategy.

        - If 'default', replace inactive parameters by their default.
        - If float, replace inactive parameters by the given float value,
          which should be able to be splitted apart by a tree-based model.
    """
    values = dict()
    for hp_name in configuration:
        value = configuration[hp_name]
        if value is None:

            if strategy == 'default':
                hp = configuration.configuration_space.get_hyperparameter(
                    hp_name)
                new_value = hp.default

            elif isinstance(strategy, float):
                new_value = strategy

            else:
                raise ValueError('Unknown imputation strategy %s' % str(strategy))

            value = new_value

        values[hp_name] = value

    new_configuration = Configuration(configuration.configuration_space,
                                      values=values,
                                      allow_inactive_with_values=True)
    return new_configuration


def get_random_neighbor(configuration, seed):
    """Draw a random neighbor by changing one parameter of a configuration.

    * If the parameter is categorical, it changes it to another value.
    * If the parameter is ordinal, it changes it to the next higher or lower
      value.
    * If parameter is a float, draw a random sample

    If changing a parameter activates new parameters or deactivates
    previously active parameters, the configuration will be rejected. If more
    than 10000 configurations were rejected, this function raises a
    ValueError.

    Parameters
    ----------
    configuration : Configuration

    seed : int
        Used to generate a random state.

    Returns
    -------
    Configuration
        The new neighbor.

    """
    random = np.random.RandomState(seed)
    values = copy.deepcopy(configuration.get_dictionary())

    rejected = True
    while rejected:
        # First, choose an active hyperparameter
        active = False
        iteration = 0
        while not active:
            iteration += 1
            if configuration._num_hyperparameters > 1:
                rand_idx = random.randint(0,
                                          configuration._num_hyperparameters - 1)
            else:
                rand_idx = 0

            value = configuration.get_array()[rand_idx]
            if np.isfinite(value):
                active = True

                hp_name = configuration.configuration_space \
                    .get_hyperparameter_by_idx(rand_idx)
                hp = configuration.configuration_space.get_hyperparameter(hp_name)

                # Only choose if there is a possibility of finding a neigboor
                if not hp.has_neighbors():
                    active = False

            if iteration > 10000:
                raise ValueError('Probably caught in an infinite loop.')
        # Get a neighboor and adapt the rest of the configuration if necessary
        neighbor = hp.get_neighbor(value, random, transform=True)
        previous_value = values[hp.name]
        values[hp.name] = neighbor

        try:
            new_configuration = Configuration(
                configuration.configuration_space, values=values)
            rejected = False
        except ValueError:
            values[hp.name] = previous_value

    return new_configuration




