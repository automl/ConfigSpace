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
        return InstantiatedConstant(self.value, self)

    def is_legal(self, value):
        return value == self.value


class UnParametrizedHyperparameter(Constant):
    pass


class NumericalHyperparameter(Hyperparameter):
    def __init__(self, name, default):
        super(NumericalHyperparameter, self).__init__(name)
        self.default = default


class FloatHyperparameter(NumericalHyperparameter):
    def is_legal(self, value):
        return isinstance(value, float) or isinstance(value, int)

    def check_default(self, default):
        return default


class IntegerHyperparameter(NumericalHyperparameter):
    def is_legal(self, value):
        return isinstance(value, int)

    def check_int(self, parameter, name):
        if abs(np.round(parameter, 0) - parameter) > 0.00000001 and \
                        type(parameter) is not int:
            raise ValueError("For the Integer parameter %s, the value must be "
                             "an Integer, too. Right now it is a %s with value"
                             " %s." % (name, type(parameter), str(parameter)))
        return int(np.round(parameter, 0))

    def check_default(self, default):
        return int(np.round(default, 0))


class UniformMixin(object):
    def is_legal(self, value):
        if not super(UniformMixin, self).is_legal(value):
            return False
        # Strange numerical issues!
        elif self.upper >= value >= (self.lower - 0.0000000001):
            return True
        else:
            return False

    def check_default(self, default):
        if default is None:
            if self.log:
                default = np.exp((np.log(self.lower) + np.log(self.upper)) / 2)
            else:
                default = (self.lower + self.upper) / 2
        default = super(UniformMixin, self).check_default(default)
        if self.is_legal(default):
            return default
        else:
            raise ValueError("Illegal default value %s" % str(default))


class NormalMixin(object):
    def check_default(self, default):
        if default is None:
            return self.mu
        elif self.is_legal(default):
            return default
        else:
            raise ValueError("Illegal default value %s" % str(default))


class UniformFloatHyperparameter(UniformMixin, FloatHyperparameter):
    def __init__(self, name, lower, upper, default=None, q=None, log=False):
        self.lower = float(lower)
        self.upper = float(upper)
        self.q = float(q) if q is not None else None
        self.log = bool(log)

        if self.lower >= self.upper:
            raise ValueError("Upper bound %f must be larger than lower bound "
                             "%f for hyperparameter %s" %
                             (self.lower, self.upper, name))
        elif log and self.lower <= 0:
            raise ValueError("Negative lower bound (%f) for log-scale "
                             "hyperparameter %s is forbidden." %
                             (self.lower, name))
        elif self.q is not None and \
                (abs(int(self.lower / self.q) - (self.lower / self.q)) >
                    0.00001):
            raise ValueError("If q is active, the lower bound %f "
                             "must be a multiple of q %f!" % (self.lower,
                                                              self.q))
        elif self.q is not None and \
                (abs(int(self.upper / self.q) - (self.upper / self.q)) >
                    0.00001):
            raise ValueError("If q is active, the upper bound %f "
                             "must be a multiple of q %f!" % (self.upper,
                                                              self.q))

        super(UniformFloatHyperparameter, self).\
            __init__(name, self.check_default(default))

    def __repr__(self):
        repr_str = StringIO.StringIO()
        repr_str.write("%s, Type: UniformFloat, Range: [%s, %s], Default: %s" %
                       (self.name, str(self.lower), str(self.upper),
                       str(self.default)))
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
        # TODO check if conversion makes sense at all (at least two integer
        # values possible!)
        return UniformIntegerHyperparameter(self.name, self.lower,
            self.upper, int(np.round(self.default)), self.q, self.log)

    def instantiate(self, value):
        return InstantiatedUniformFloatHyperparameter(value, self)


class NormalFloatHyperparameter(NormalMixin, FloatHyperparameter):
    def __init__(self, name, mu, sigma, default=None, q=None, log=False):
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.q = float(q) if q is not None else None
        self.log = bool(log)
        super(NormalFloatHyperparameter, self).\
            __init__(name, self.check_default(default))

    def __repr__(self):
        repr_str = StringIO.StringIO()
        repr_str.write("%s, Type: NormalFloat, Mu: %s Sigma: %s, Default: %s" %
                       (self.name, str(self.mu), str(self.sigma), str(self.default)))
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
            self.mu - (z * self.sigma), self.mu + (z * self.sigma),
            default=int(np.round(self.default, 0)), q=self.q, log=self.log)

    def to_integer(self):
        return NormalIntegerHyperparameter(self.name, self.mu, self.sigma,
            default=int(np.round(self.default, 0)), q=self.q, log=self.log)

    def is_legal(self, value):
        if isinstance(value, (float, int)):
            return True
        else:
            return False

    def instantiate(self, value):
        return InstantiatedNormalFloatHyperparameter(value, self)


class UniformIntegerHyperparameter(UniformMixin, IntegerHyperparameter):
    def __init__(self, name, lower, upper, default=None, q=None, log=False):
        self.lower = self.check_int(lower, "lower")
        self.upper = self.check_int(upper, "upper")
        if default is not None:
            default = self.check_int(default, name)
        if q is not None:
            if q < 1:
                warnings.warn("Setting quantization < 1 for Integer "
                              "Hyperparameter '%s' has no effect." %
                              name)
                self.q = None
            else:
                self.q = self.check_int(q, "q")
        else:
            self.q = None
        self.log = bool(log)

        if self.lower >= self.upper:
            raise ValueError("Upper bound %d must be larger than lower bound "
                             "%d for hyperparameter %s" %
                             (self.lower, self.upper, name))
        elif log and self.lower <= 0:
            raise ValueError("Negative lower bound (%d) for log-scale "
                             "hyperparameter %s is forbidden." %
                             (self.lower, name))
        elif self.q is not None and \
                (abs(int(float(self.lower) / self.q) -
                    (float(self.lower) / self.q)) > 0.00001):
            raise ValueError("If q is active, the lower bound %d "
                             "must be a multiple of q %d!" % (self.lower,
                                                              self.q))
        elif self.q is not None and \
                (abs(int(float(self.upper) / self.q) -
                    (float(self.upper) / self.q)) > 0.00001):
            raise ValueError("If q is active, the upper bound %d "
                             "must be a multiple of q %d!" % (self.upper,
                                                              self.q))

        super(UniformIntegerHyperparameter, self).\
            __init__(name, self.check_default(default))


    def __repr__(self):
        repr_str = StringIO.StringIO()
        repr_str.write("%s, Type: UniformInteger, Range: [%s, %s], Default: %s"
                       % (self.name, str(self.lower),
                          str(self.upper), str(self.default)))
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
    def __init__(self, name, mu, sigma, default=None, q=None, log=False):
        self.mu = mu
        self.sigma = sigma
        if default is not None:
            default = self.check_int(default, name)
        if q is not None:
            if q < 1:
                warnings.warn("Setting quantization < 1 for Integer "
                              "Hyperparameter '%s' has no effect." %
                              name)
                self.q = None
            else:
                self.q = self.check_int(q, "q")
        else:
            self.q = None
        self.log = bool(log)
        super(NormalIntegerHyperparameter, self).\
            __init__(name, self.check_default(default))

    def __repr__(self):
        repr_str = StringIO.StringIO()
        repr_str.write("%s, Type: NormalInteger, Mu: %s Sigma: %s, Default: "
                       "%s" % (self.name, str(self.mu),
                               str(self.sigma), str(self.default)))
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
                                            default=self.default,
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
    def __init__(self, name, choices, default=None):
        super(CategoricalHyperparameter, self).__init__(name)
        # TODO check that there is no bullshit in the choices!
        self.choices = choices
        self.default = self.check_default(default)

    def __repr__(self):
        repr_str = StringIO.StringIO()
        repr_str.write("%s, Type: Categorical, Choices: {" % (self.name))
        for idx, choice in enumerate(self.choices):
            repr_str.write(str(choice))
            if idx < len(self.choices) - 1:
                repr_str.write(", ")
        repr_str.write("}")
        repr_str.write(", Default: ")
        repr_str.write(str(self.default))
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

    def check_default(self, default):
        if default is None:
            return self.choices[0]
        elif self.is_legal(default):
            return default
        else:
            raise ValueError("Illegal default value %s" % str(default))

    def instantiate(self, value):
        return InstantiatedCategoricalHyperparameter(value, self)


class InstantiatedHyperparameter(object):
    __metaclass__ = ABCMeta

    def __init__(self, value, hyperparameter):
        if not isinstance(hyperparameter, Hyperparameter):
            raise TypeError("Argument 'hyperparameter' is not of type %s, "
                            "but %s" % (Hyperparameter, type(hyperparameter)))
        if not hyperparameter.is_legal(value):
            raise ValueError("Value %s, %s for instantiation of "
                             "hyperparameter '%s' is not a legal value." %
                             (str(value), str(type(value)), hyperparameter))

        self.value = value
        self.hyperparameter = hyperparameter

    def __eq__(self, other):
        if type(self) != type(other):
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

    def __init__(self, value, hyperparameter):
        value = hyperparameter.check_int(value, hyperparameter.name)
        super(InstantiatedIntegerHyperparameter, self).__init__(value, hyperparameter)

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
