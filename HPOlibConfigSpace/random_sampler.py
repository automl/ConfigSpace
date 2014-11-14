import numpy as np
import random
import sys

from HPOlibConfigSpace.configuration_space import ConfigurationSpace, \
    Configuration
from HPOlibConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, NormalFloatHyperparameter, \
    NormalIntegerHyperparameter, CategoricalHyperparameter, \
    InactiveHyperparameter


class RandomSampler(object):
    # Sampling methods are in here to control the random seed!
    def __init__(self, configuration_space, seed):
        if not isinstance(configuration_space, ConfigurationSpace):
            raise TypeError("Configuration expects an instance of %s, "
                            "you provided '%s'" %
                            (ConfigurationSpace, type(configuration_space)))

        if not isinstance(seed, int):
            raise TypeError("Seed must be of type '%s', not '%s'" %
                            (int, type(seed)))

        self.configuration_space = configuration_space
        self.seed = seed
        self.random = random.Random(self.seed)

    def sample_configuration(self):
        # TODO: this is straightforward, but slow. It would make more sense
        # to have a list of conditions, which are sorted topologicoly by the
        # appearence of their children
        iteration = 0
        while True:
            instantiated_hyperparameters = {}
            hyperparameters = self.configuration_space.get_hyperparameters(
                order="topological")
            for hyperparameter in hyperparameters:
                conditions =  self.configuration_space.get_parents_of(
                    hyperparameter.name)
                # TODO this conditions should all be equal, are they actually?
                add = True
                for condition in conditions:
                    parent_names = [c.parent.name for c in
                                    condition.get_descendant_literal_conditions()]

                    parents = [instantiated_hyperparameters[parent_name] for
                               parent_name in parent_names]

                    if len(parents) == 1:
                        parents = parents[0]
                    if not condition.evaluate(parents):
                        add = False

                if add:
                    instantiated_hyperparameters[hyperparameter.name] = \
                        getattr(self, "_sample_%s" % type(hyperparameter).__name__)\
                            (hyperparameter)
                else:
                    instantiated_hyperparameters[hyperparameter.name] = \
                        InactiveHyperparameter(None, hyperparameter)

            try:
                return Configuration(self.configuration_space,
                                      **instantiated_hyperparameters)
            except ValueError as e:
                iteration += 1

                if iteration == 1000000:
                    raise ValueError("Cannot sample valid configuration for "
                                     "%s" % self.configuration_space)



    def _sample_UniformFloatHyperparameter(self, ufhp):
        if ufhp.log:
            lower = np.log(ufhp.lower)
            upper = np.log(ufhp.upper)
        else:
            lower = ufhp.lower
            upper = ufhp.upper
        value = self.random.uniform(lower, upper)
        if ufhp.log:
            value = np.exp(value)
        return ufhp.instantiate(value)

    def _sample_NormalFloatHyperparameter(self, nfhp):
        mu = nfhp.mu
        sigma = nfhp.sigma
        gauss = self.random.gauss(mu, sigma)
        if nfhp.log:
            gauss = np.exp(gauss)
        return nfhp.instantiate(gauss)

    def _sample_UniformIntegerHyperparameter(self, uihp):
        if uihp.log:
            lower = np.log(uihp.lower)
            upper = np.log(uihp.upper)
        else:
            lower = uihp.lower
            upper = uihp.upper
        value = self.random.randint(lower, upper)
        if uihp.log:
            value = np.exp(value)
        return uihp.instantiate(value)

    def _sample_NormalIntegerHyperparameter(self, nihp):
        mu = nihp.mu
        sigma = nihp.sigma
        value = self.random.gauss(mu, sigma)
        if nihp.log:
            gauss = np.exp(value)
        return nihp.instantiate(int(round(value)))

    def _sample_CategoricalHyperparameter(self, chp):
        choice = self.random.choice(chp.choices)
        return chp.instantiate(choice)

    def _sample_Constant(self, chp):
        return chp.instantiate(chp.value)

    def _sample_UnParametrizedHyperparameter(self, uphp):
        return self._sample_Constant(uphp)






