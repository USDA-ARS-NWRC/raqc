# -*- coding: utf-8 -*-

"""Top-level package for raqc."""

__author__ = """Zachary R Uhlmann"""
__email__ = 'zach.uhlmann@usda.gov'
__version__ = '0.1.1'

import os

__core_config__ = os.path.abspath(os.path.dirname(__file__) + '/CoreConfig.ini')


__recipes__ = os.path.abspath(os.path.dirname(__file__) + '/recipes.ini')

__config_titles__ = {'files': 'filepaths of input images and output locations',
                     'difference_arrays': 'specify actions to perform on arrays',
                     'flags': 'specify flags (outliers) desired in otput',
                     'histogram_outliers' : 'set parameters and thresholds to find histogram space outliers',
                     'block_behavior' : 'set parameters and thresholds to find large blocks of outliers in the image space'
                     }
