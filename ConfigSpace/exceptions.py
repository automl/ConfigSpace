#
# class SubValueError(ValueError):
#     def __init__(self, error_msg, **kwargs):
#         ValueError.__init__(self, error_msg)
#         self.vars = []
#         for k, v in kwargs.iteritems():
#             self.k = v
#             self.vars.append(k)
#             print("%s = %s" % (k, v))
#
#     def get_items(self):
#         return self.vars


class IntValueError(ValueError):
    pass


class BoundsValueError(ValueError):
    pass


class DefaultValueError(ValueError):
    pass


class LoopValueError(ValueError):
    pass


class IndexValueError(ValueError):
    pass


class IllegalValueError(ValueError):
    pass


class ForbiddenValueError(ValueError):
    pass


class HpConfigValueError(ValueError):
    pass


class ParentValueError(ValueError):
    pass


class ArgsValueError(ValueError):
    pass