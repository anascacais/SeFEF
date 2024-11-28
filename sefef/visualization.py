# -*- coding: utf-8 -*-
"""
sefef.visualization
-------------------

This is a helper module for visualization.

:copyright: (c) 2024 by Ana Sofia Carmo
:license: BSD 3-clause License, see LICENSE for more details.
"""


COLOR_PALETTE = ['#4179A0', '#A0415D', '#44546A', '#44AA97', '#FFC000', '#0F3970', '#873C26']


def hex_to_rgba(h, alpha):
    '''
    converts color value in hex format to rgba format with alpha transparency
    '''
    return tuple([int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)] + [alpha])
