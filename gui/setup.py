#!/usr/bin/env python

from numpy.distutils.misc_util import Configuration


def configuration(parent_package='', top_path=None):
    config = Configuration('gui', parent_package, top_path)

    # There is nothing special to do for this subpackage.
    # This setup.py file as it stands could be deleted.

    return config
