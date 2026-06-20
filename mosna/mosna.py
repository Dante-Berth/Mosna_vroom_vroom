#! /usr/bin/env python3
"""
mosna.mosna — backward-compatibility shim
=========================================

The monolithic implementation that used to live in this file has been split
into thematic submodules (see :mod:`mosna`). This module is kept so that legacy
imports such as ``from mosna.mosna import mixing_matrix`` keep working.

Every public name from the package is re-exported here.
"""

from mosna import *          # noqa: F401,F403
from mosna._common import *  # noqa: F401,F403
from mosna import io         # noqa: F401
