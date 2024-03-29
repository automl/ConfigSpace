"""
**********
Exceptions
**********.

Base exceptions and errors for NetworkX.

"""
from __future__ import annotations

__author__ = """Aric Hagberg (hagberg@lanl.gov)
Pieter Swart (swart@lanl.gov)
Dan Schult(dschult@colgate.edu)
Loïc Séguin-C. <loicseguin@gmail.com>"""
# Copyright (C) 2004-2011 by
#    Aric Hagberg <hagberg@lanl.gov>
#    Dan Schult <dschult@colgate.edu>
#    Pieter Swart <swart@lanl.gov>
#    All rights reserved.
#    BSD license.
#

# Exception handling


# the root of all Exceptions
class NetworkXException(Exception):
    """Base class for exceptions in NetworkX."""


class NetworkXError(NetworkXException):
    """Exception for a serious error in NetworkX."""


class NetworkXPointlessConcept(NetworkXException):
    """Harary, F. and Read, R. "Is the Null Graph a Pointless Concept?"
    In Graphs and Combinatorics Conference, George Washington University.
    New York: Springer-Verlag, 1973.
    """


class NetworkXAlgorithmError(NetworkXException):
    """Exception for unexpected termination of algorithms."""


class NetworkXUnfeasible(NetworkXAlgorithmError):
    """Exception raised by algorithms trying to solve a problem
    instance that has no feasible solution.
    """


class NetworkXNoPath(NetworkXUnfeasible):
    """Exception for algorithms that should return a path when running
    on graphs where such a path does not exist.
    """


class NetworkXUnbounded(NetworkXAlgorithmError):
    """Exception raised by algorithms trying to solve a maximization
    or a minimization problem instance that is unbounded.
    """


class NetworkXNotImplemented(NetworkXException):
    """Exception raised by algorithms not implemented for a type of graph."""
