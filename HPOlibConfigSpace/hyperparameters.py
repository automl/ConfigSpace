__author__ = 'feurerm'


from abc import ABCMeta, abstractmethod
import StringIO
import types
import warnings


import numpy as np


class Hyperparameter(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, name):
        if not isinstance(name, types.StringTypes):
            raise TypeError("The name of a hyperparameter must be of in %s, "
                            "but is %s." % (str(types.StringTypes), type(name)))
        self.name = name

    @abstractmethod
    def __eq__(self, other):
        pass

    def __ne__(self, other):
        return not self.__eq__(other)

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def instantiate(self, value):
        pass

    @abstractmethod
    def is_legal(self, value):
        pass


class Constant(Hyperparameter):
    def __init__(self, name, value):
        super(Constant, self).__init__(name)
        allowed_types = (types.IntType, types.FloatType,
                         types.StringType, types.UnicodeType)

        if not isinstance(value, allowed_types) or \
                isinstance(value, bool):
            raise TypeError("Constant value is of type %s, but only the "
                            "following types are allowed: %s" % (type(value),
                                                                 allowed_types))

        self.value = value

    def __repr__(self):
        repr_str = ["%s" % self.name,
                    "Type: Constant",
                    "Value: %s" % self.value]
        return ", ".join(repr_str)

    def __eq__(self, other):
        if type(self) == type(other):
            if self.name != other.name:
                return False
            if self.value != other.value:
                return False
            return True
        else:
            return False

    def instantiate(self, value):
        if value != self.value:
            raise ValueError("Cannot instantiate a constant with a new value!")
        return InstantiatedConstant("True", self)

    def is_legal(self, value):
        return value == self.value


class NumericalHyperparameter(Hyperparameter):
    def __init__(self, name):
        super(NumericalHyperparameter, self).__init__(name)


class FloatHyperparameter(NumericalHyperparameter):
    pass


class IntegerHyperparameter(NumericalHyperparameter):
    def check_int(self, parameter, name):
        if abs(np.round(parameter, 5) - parameter) > 0.00001 and \
                        type(parameter) is not int:
            raise ValueError("For the Integer parameter %s, the value must be "
                             "an Integer, too. Right now it is a %s with value"
                             " %s" % (name, type(parameter), str(parameter)))
        return int(parameter)


class UniformMixin():
    def is_legal(self, value):
        if self.upper >= value >= self.lower:
            return True
        else:
            return False


class NormalMixin():
    pass


class UniformFloatHyperparameter(UniformMixin, FloatHyperparameter):
    def __init__(self, name, lower, upper, q=None, log=False):
        super(UniformFloatHyperparameter, self).__init__(name)
        self.lower = float(lower)
        self.upper = float(upper)
        self.q = float(q) if q is not None else None
        self.log = bool(log)

    def __repr__(self):
        repr_str = StringIO.StringIO()
        repr_str.write("%s, Type: UniformFloat, Range: [%s, %s]" %
                       (self.name, str(self.lower), str(self.upper)))
        if self.log:
            repr_str.write(", on log-scale")
        if self.q is not None:
            repr_str.write(", Q: %s" % str(self.q))
        repr_str.seek(0)
        return repr_str.getvalue()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return all([self.name == other.name,
                        abs(self.lower - other.lower) < 0.00000001,
                        abs(self.upper - other.upper) < 0.00000001,
                        self.log == other.log,
                        self.q is None and other.q is None or
                        self.q is not None and other.q is not None and
                        abs(self.q - other.q) < 0.00000001])
        else:
            return False

    def to_integer(self):
        return UniformIntegerHyperparameter(self.name, self.lower,
                                            self.upper, self.q, self.log)

    def instantiate(self, value):
        return InstantiatedUniformFloatHyperparameter(value, self)


class NormalFloatHyperparameter(NormalMixin, FloatHyperparameter):
    def __init__(self, name, mu, sigma, q=None, log=False):
        super(NormalFloatHyperparameter, self).__init__(name)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.q = float(q) if q is not None else None
        self.log = bool(log)

    def __repr__(self):
        repr_str = StringIO.StringIO()
        repr_str.write("%s, Type: NormalFloat, Mu: %s Sigma %s" %
                       (self.name, str(self.mu), str(self.sigma)))
        if self.log:
            repr_str.write(", on log-scale")
        if self.q is not None:
            repr_str.write(", Q: %s" % str(self.q))
        repr_str.seek(0)
        return repr_str.getvalue()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return all([self.name == other.name,
                        abs(self.mu - other.mu) < 0.00000001,
                        abs(self.sigma - other.sigma) < 0.00000001,
                        self.log == other.log,
                        self.q is None and other.q is None or
                        self.q is not None and other.q is not None and
                        abs(self.q - other.q) < 0.00000001])
        else:
            return False

    def to_uniform(self, z=3):
        return UniformFloatHyperparameter(self.name,
                                          self.mu - (z * self.sigma),
                                          self.mu + (z * self.sigma),
                                          q=self.q, log=self.log)

    def to_integer(self):
        return NormalIntegerHyperparameter(self.name, self.mu, self.sigma,
                                           q=self.q, log=self.log)

    def is_legal(self, value):
        if isinstance(value, (float, int)):
            return True
        else:
            return False

    def instantiate(self, value):
        return InstantiatedNormalFloatHyperparameter(value, self)


class UniformIntegerHyperparameter(UniformMixin, IntegerHyperparameter):
    def __init__(self, name, lower, upper, q=None, log=False):
        super(UniformIntegerHyperparameter, self).__init__(name)
        self.lower = self.check_int(lower, "lower")
        self.upper = self.check_int(upper, "upper")
        if q is not None:
            q = self.check_int(q, "q")
            if q < 1:
                warnings.warn("Setting quantization < 1 for Integer "
                              "Hyperparameter '%s' has no effect." %
                              self.name)
                self.q = None
            else:
                self.q = self.check_int(q, "q")
        else:
            self.q = None
        self.log = bool(log)

    def __repr__(self):
        repr_str = StringIO.StringIO()
        repr_str.write("%s, Type: UniformInteger, Range: [%s, %s]" %
                       (self.name, str(np.int(self.lower)),
                        str(np.int(self.upper))))
        if self.log:
            repr_str.write(", on log-scale")
        if self.q is not None:
            repr_str.write(", Q: %s" % str(np.int(self.q)))
        repr_str.seek(0)
        return repr_str.getvalue()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return all([self.name == other.name,
                        self.lower == other.lower,
                        self.upper == other.upper,
                        self.log == other.log,
                        self.q is None and other.q is None or
                        self.q is not None and other.q is not None and
                        self.q == other.q])
        else:
            return False

    def instantiate(self, value):
        return InstantiatedUniformIntegerHyperparameter(value, self)


class NormalIntegerHyperparameter(NormalMixin, IntegerHyperparameter):
    def __init__(self, name, mu, sigma, q=None, log=False):
        super(NormalIntegerHyperparameter, self).__init__(name)
        self.mu = mu
        self.sigma = sigma
        if q is not None:
            q = self.check_int(q, "q")
            if q < 1:
                warnings.warn("Setting quantization < 1 for Integer "
                              "Hyperparameter '%s' has no effect." %
                              self.name)
                self.q = None
            else:
                self.q = self.check_int(q, "q")
        else:
            self.q = None
        self.log = bool(log)

    def __repr__(self):
        repr_str = StringIO.StringIO()
        repr_str.write("%s, Type: NormalInteger, Mu: %s Sigma %s" %
                       (self.name, str(self.mu), str(self.sigma)))
        if self.log:
            repr_str.write(", on log-scale")
        if self.q is not None:
            repr_str.write(", Q: %s" % str(self.q))
        repr_str.seek(0)
        return repr_str.getvalue()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return all([self.name == other.name,
                        abs(self.mu - other.mu) < 0.00000001,
                        abs(self.sigma - other.sigma) < 0.00000001,
                        self.log == other.log,
                        self.q is None and other.q is None or
                        self.q is not None and other.q is not None and
                        self.q == other.q])
        else:
            return False

    def to_uniform(self, z=3):
        return UniformIntegerHyperparameter(self.name,
                                            self.mu - (z * self.sigma),
                                            self.mu + (z * self.sigma),
                                            q=self.q, log=self.log)

    def is_legal(self, value):
        if isinstance(value, int):
            return True
        else:
            return False

    def instantiate(self, value):
        return InstantiatedNormalIntegerHyperparameter(value, self)


class CategoricalHyperparameter(Hyperparameter):
    # TODO add more magic for automated type recognition
    def __init__(self, name, choices):
        super(CategoricalHyperparameter, self).__init__(name)
        # TODO check that there is no bullshit in the choices!
        self.choices = choices

    def __repr__(self):
        repr_str = StringIO.StringIO()
        repr_str.write("%s, Type: Categorical, Choices: {" % (self.name))
        for idx, choice in enumerate(self.choices):
            repr_str.write(str(choice))
            if idx < len(self.choices) - 1:
                repr_str.write(", ")
        repr_str.write("}")
        repr_str.seek(0)
        return repr_str.getvalue()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.name != other.name:
                return False
            if len(self.choices) != len(other.choices):
                return False
            else:
                for i in range(len(self.choices)):
                    if self.choices[i] != other.choices[i]:
                        return False
            return True
        else:
            return False

    def is_legal(self, value):
        if value in self.choices:
            return True
        else:
            return False

    def instantiate(self, value):
        return InstantiatedCategoricalHyperparameter(value, self)


class InstantiatedHyperparameter(object):
    __metaclass__ = ABCMeta

    def __init__(self, value, hyperparameter):
        if not isinstance(hyperparameter, Hyperparameter):
            raise TypeError("Argument 'hyperparameter' is not of type %s." %
                            Hyperparameter)
        if not hyperparameter.is_legal(value):
            raise ValueError("Value %s for instantiation of hyperparameter %s "
                             "is not a legal value." %
                             (str(value), hyperparameter.name))

        self.value = value
        self.hyperparameter = hyperparameter

    def __eq__(self, other):
        if type(self) == type(other):
            return False
        elif self.value != other.value:
            return False
        elif self.hyperparameter != other.hyperparameter:
            return False
        return True

    @abstractmethod
    def __repr__(self):
        pass

    def is_legal(self):
        return self.hyperparameter.is_legal(self.value)


class InactiveHyperparameter(InstantiatedHyperparameter):
    def __init__(self, value, hyperparameter):
        # TODO document that value is just a dummy argument here!
        self.value = value
        self.hyperparameter = hyperparameter

    # TODO implement a better equals function
    def __repr__(self):
        return "%s, Inactive" % self.hyperparameter.name

    def is_legal(self):
        return False


class InstantiatedConstant(InstantiatedHyperparameter):
    def __repr__(self):
        return "%s, Constant: %s" % (self.hyperparameter.name, str(self.value))


class InstantiatedNumericalHyperparameter(InstantiatedHyperparameter):
    __metaclass__ = ABCMeta


class InstantiatedFloatHyperparameter(InstantiatedNumericalHyperparameter):
    __metaclass__ = ABCMeta

    def __repr__(self):
        return "%s, Value: %f" % (self.hyperparameter.name, self.value)


class InstantiatedIntegerHyperparameter(InstantiatedNumericalHyperparameter):
    __metaclass__ = ABCMeta

    def __repr__(self):
        return "%s, Value: %d" % (self.hyperparameter.name, self.value)


class InstantiatedUniformFloatHyperparameter(InstantiatedFloatHyperparameter):
    pass


class InstantiatedNormalFloatHyperparameter(InstantiatedFloatHyperparameter):
    pass


class InstantiatedUniformIntegerHyperparameter(
    InstantiatedIntegerHyperparameter):
    pass


class InstantiatedNormalIntegerHyperparameter(
    InstantiatedIntegerHyperparameter):
    pass


class InstantiatedCategoricalHyperparameter(InstantiatedHyperparameter):
    def __repr__(self):
        return "%s, Value: %s" % (self.hyperparameter.name, str(self.value))
