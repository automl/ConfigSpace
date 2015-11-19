from abc import ABCMeta, abstractmethod
import warnings

import numpy as np
import six


class Hyperparameter(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, name):
        if not isinstance(name, six.string_types):
            raise TypeError(
                "The name of a hyperparameter must be an instance of"
                " %s, but is %s." % (str(six.string_types), type(name)))
        self.name = name

    # http://stackoverflow.com/a/25176504/4636294
    def __eq__(self, other):
        """Override the default Equals behavior"""
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __ne__(self, other):
        """Define a non-equality test"""
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        """Override the default hash behavior (that returns the id or the object)"""
        return hash(tuple(sorted(self.__dict__.items())))

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def is_legal(self, value):
        pass

    def sample(self, rs):
        vector = self._sample(rs)
        return self._transform(vector)

    @abstractmethod
    def _sample(self, rs, size):
        pass

    @abstractmethod
    def _transform(self, vector):
        pass


class Constant(Hyperparameter):
    def __init__(self, name, value):
        super(Constant, self).__init__(name)
        allowed_types = []
        allowed_types.extend(six.integer_types)
        allowed_types.append(float)
        allowed_types.extend(six.string_types)
        allowed_types.append(six.text_type)
        allowed_types = tuple(allowed_types)

        if not isinstance(value, allowed_types) or \
                isinstance(value, bool):
            raise TypeError("Constant value is of type %s, but only the "
                            "following types are allowed: %s" %
                            (type(value), allowed_types))

        self.value = value
        self._nan = -1

    def __repr__(self):
        repr_str = ["%s" % self.name,
                    "Type: Constant",
                    "Value: %s" % self.value]
        return ", ".join(repr_str)

    def is_legal(self, value):
        return value == self.value

    def _sample(self, rs, size=None):
        return 0 if size == 1 else np.zeros((size,))

    def _transform(self, vector):
        return self.value if vector == 0 else None


class UnParametrizedHyperparameter(Constant):
    pass


class NumericalHyperparameter(Hyperparameter):
    def __init__(self, name, default):
        super(NumericalHyperparameter, self).__init__(name)
        self.default = default


class FloatHyperparameter(NumericalHyperparameter):
    def __init__(self, name, default):
        self._nan = np.NaN
        super(FloatHyperparameter, self).__init__(name, default)

    def is_legal(self, value):
        return isinstance(value, float) or isinstance(value, int)

    def check_default(self, default):
        return np.round(float(default), 10)


class IntegerHyperparameter(NumericalHyperparameter):
    def __init__(self, name, default):
        self._nan = np.NaN
        super(IntegerHyperparameter, self).__init__(name, default)

    def is_legal(self, value):
        return isinstance(value, int)

    def check_int(self, parameter, name):
        if abs(int(parameter) - parameter) > 0.00000001 and \
                        type(parameter) is not int:
            raise ValueError("For the Integer parameter %s, the value must be "
                             "an Integer, too. Right now it is a %s with value"
                             " %s." % (name, type(parameter), str(parameter)))
        return int(parameter)

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

        super(UniformFloatHyperparameter, self). \
            __init__(name, self.check_default(default))

        if self.log:
            if self.q is not None:
                lower = self.lower - (np.float64(self.q) / 2 - 0.0001)
                upper = self.upper + (np.float64(self.q) / 2 - 0.0001)
            else:
                lower = self.lower
                upper = self.upper
            self._lower = np.log(lower)
            self._upper = np.log(upper)
        else:
            if self.q is not None:
                self._lower = self.lower - (self.q / 2 - 0.0001)
                self._upper = self.upper + (self.q / 2 - 0.0001)
            else:
                self._lower = self.lower
                self._upper = self.upper

    def __repr__(self):
        repr_str = six.StringIO()
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
                                            self.upper,
                                            int(np.round(self.default)), self.q,
                                            self.log)

    def _sample(self, rs, size=None):
        return rs.uniform(size=size)

    def _transform(self, vector):
        if np.isnan(vector):
            return None
        vector *= (self._upper - self._lower)
        vector += self._lower
        if self.log:
            vector = np.exp(vector)
        if self.q is not None:
            vector = int(np.round(vector / self.q, 0)) * self.q
        return vector


class NormalFloatHyperparameter(NormalMixin, FloatHyperparameter):
    def __init__(self, name, mu, sigma, default=None, q=None, log=False):
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.q = float(q) if q is not None else None
        self.log = bool(log)
        super(NormalFloatHyperparameter, self). \
            __init__(name, self.check_default(default))

    def __repr__(self):
        repr_str = six.StringIO()
        repr_str.write("%s, Type: NormalFloat, Mu: %s Sigma: %s, Default: %s" %
                       (self.name, str(self.mu), str(self.sigma),
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
                                          default=int(
                                              np.round(self.default, 0)),
                                          q=self.q, log=self.log)

    def to_integer(self):
        return NormalIntegerHyperparameter(self.name, self.mu, self.sigma,
                                           default=int(
                                               np.round(self.default, 0)),
                                           q=self.q, log=self.log)

    def is_legal(self, value):
        if isinstance(value, (float, int)):
            return True
        else:
            return False

    def _sample(self, rs, size=None):
        mu = self.mu
        sigma = self.sigma
        return rs.normal(mu, sigma, size=size)

    def _transform(self, vector):
        if np.isnan(vector):
            return None
        if self.log:
            vector = np.exp(vector)
        if self.q is not None:
            vector = int(np.round(vector / self.q, 0)) * self.q
        return vector


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

        super(UniformIntegerHyperparameter, self). \
            __init__(name, self.check_default(default))

        self.ufhp = UniformFloatHyperparameter(self.name,
                                               self.lower - 0.49999,
                                               self.upper + 0.49999,
                                               log=self.log, q=self.q,
                                               default=self.default)

    def __repr__(self):
        repr_str = six.StringIO()
        repr_str.write("%s, Type: UniformInteger, Range: [%s, %s], Default: %s"
                       % (self.name, str(self.lower),
                          str(self.upper), str(self.default)))
        if self.log:
            repr_str.write(", on log-scale")
        if self.q is not None:
            repr_str.write(", Q: %s" % str(np.int(self.q)))
        repr_str.seek(0)
        return repr_str.getvalue()

    def _sample(self, rs, size=None):
        value = self.ufhp._sample(rs, size=size)
        return value
        # if self.log is False and self.q is None:
        #    value = rs.randint(self.lower, self.upper + 1)
        #    return value
        #else:
        #    value = self.ufhp.sample(rs)
        #    if self.q is not None:
        #        value = int(np.round(value / self.q, 0)) * self.q
        #    return int(np.round(value, 0))

    def _transform(self, vector):
        if np.isnan(vector):
            return None
        vector = self.ufhp._transform(vector)
        if self.q is not None:
            vector = int(np.round(vector / self.q, 0)) * self.q
        return int(np.round(vector, 0))


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

        super(NormalIntegerHyperparameter, self). \
            __init__(name, self.check_default(default))

        self.nfhp = NormalFloatHyperparameter(self.name,
                                              self.mu,
                                              self.sigma,
                                              log=self.log,
                                              q=self.q,
                                              default=self.default)

    def __repr__(self):
        repr_str = six.StringIO()
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

    def _sample(self, rs, size=None):
        return self.nfhp._sample(rs, size=size)

    def _transform(self, vector):
        if np.isnan(vector):
            return None
        vector = self.nfhp._transform(vector)
        return int(np.round(vector, 0))


class CategoricalHyperparameter(Hyperparameter):
    # TODO add more magic for automated type recognition
    def __init__(self, name, choices, default=None):
        super(CategoricalHyperparameter, self).__init__(name)
        # TODO check that there is no bullshit in the choices!
        self.choices = choices
        self._num_choices = len(choices)
        self.default = self.check_default(default)
        self._nan = -1

    def __repr__(self):
        repr_str = six.StringIO()
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

    def _sample(self, rs, size=None):
        return rs.randint(0, self._num_choices, size=size)

    def _transform(self, vector):
        if vector == -1:
            return None
        return self.choices[vector]
