#    Copyright (C) 2004-2010 by
#    Aric Hagberg <hagberg@lanl.gov>
#    Dan Schult <dschult@colgate.edu>
#    Pieter Swart <swart@lanl.gov>
#    All rights reserved.
#    BSD license.
#
# Add platform dependent shared library path to sys.path
#
# Modified by Matthias Feurer for the package HPOlibConfigSpace

from __future__ import absolute_import

import sys
if sys.version_info[:2] < (2, 6):
    m = "Python version 2.6 or later is required for NetworkX (%d.%d detected)."
    raise ImportError(m % sys.version_info[:2])
del sys

# Release data
from HPOlibConfigSpace.nx.release import authors, license, date, version

__author__   = '%s <%s>\n%s <%s>\n%s <%s>' % \
              ( authors['Hagberg'] + authors['Schult'] + \
                authors['Swart'] )
__license__  = license

__date__ = date
__version__ = version

#These are import orderwise
from HPOlibConfigSpace.nx import *

import HPOlibConfigSpace.nx.exception
from HPOlibConfigSpace.nx.exception import *

import HPOlibConfigSpace.nx.classes
from HPOlibConfigSpace.nx.classes import *

import HPOlibConfigSpace.nx.algorithms
from HPOlibConfigSpace.nx.algorithms import *

