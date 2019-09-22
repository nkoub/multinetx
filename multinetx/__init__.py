#! /usr/bin/env python
# -*- coding: utf-8 -*-

########################################################################
#
#    multiNetX -- a python package for general multilayer graphs
#
#    (C) Copyright 2013-2019, Nikos E Kouvaris
#    multiNetX is part of the deliverables of the LASAGNE project
#    (multi-LAyer SpAtiotemporal Generalized NEtworks),
#    EU/FP7-2012-STREP-318132 (http://complex.ffn.ub.es/~lasagne/)
#
#    multiNetX is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published
#    by the Free Software Foundation, either version 3 of the License,
#    or (at your option) any later version.
#
#    multiNetX is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#    See the GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program. If not, see http://www.gnu.org/licenses/.
########################################################################

"""
multiNetX - a general multilayer graph manipulation python module based
on NetworkX
"""
from __future__ import division, absolute_import

try:
    from networkx import *
except ImportError:
    raise ImportError("NetworkX is required")

try:
    from scipy.sparse import lil_matrix as lil_matrix
except ImportError:
    raise ImportError("SciPy is required")

# import all modules of the packages
from multinetx.core.draw import *
from multinetx.core.exceptions import *
from multinetx.core.multilayer import *
from multinetx.core.utilities import *
from multinetx.core import *
from multinetx import *

# check for Python verion
import sys
if sys.version_info[:2] < (3, 6):
    raise ImportError("Python version 3.6 or later is required for multiNetX\
                        (%d.%d detected)." % sys.version_info[:2])
del sys

__author__ = "Nikos E. Kouvaris <nkouba@gmail.com>"
__copyright__ = "Copyright (C) 2013-2019 by \
                Nikos E. Kouvaris <nkouba@gmail.com>\
                Project LASAGNE \
                -- multi-LAyer SpAtiotemporal Generalized NEtworks"
__license__ = "GNU GPL"
__version__ = "2.1"


