#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 13:25:38 2023

Some commonly used constants in the other calculations.

@author: Dakotah Martinez
"""

from mpmath import mp, mpf
import mpmath

loop_fac = 1 / (16 * mp.power(mpmath.pi, 2))
loop_fac_sq = mp.power(loop_fac, 2)
b_1l = [mpf(str(33))/5, mpf(str(1)), mpf(str(-3))]
b_2l = [[mpf(str(199/25)), mpf(str(27))/5, mpf(str(88))/5],
        [mpf(str(9))/5, mpf(str(25)), mpf(str(24))],
        [mpf(str(11))/5, mpf(str(9)), mpf(str(14))]]
c_2l = [[mpf(str(26))/5, mpf(str(14))/5, mpf(str(18))/5],
        [mpf(str(6)), mpf(str(6)), mpf(str(2))],
        [mpf(str(4)), mpf(str(4)), mpf(str(0))]]
