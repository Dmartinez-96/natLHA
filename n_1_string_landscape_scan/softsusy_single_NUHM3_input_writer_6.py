# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 07:57:59 2021

@author: Dakotah Martinez
"""

import numpy as np


def lin_dist_func(a, b):
    """
    Return linearly increasing probability distr'n function w/ hit-or-miss.

    Parameters
    ----------
    a : Lowest value of random variable.
    b : Highest value of random variable.

    Returns
    -------
    Array with linearly increasing probability distribution for soft term.

    """
    check_if_keep = True
    while check_if_keep:
        rand_val_1 = np.random.rand() * (b - a) + a
        rand_val_2 = np.random.rand() * (b - a) + a
        if abs(rand_val_2) > abs(rand_val_1):
            # Reject value and try again
            check_if_keep = True
        else:
            check_if_keep = False
    return rand_val_1


def unif_dist_func(a, b):
    """
    Return uniform probability dist.'n function on array data.

    Parameters
    ----------
    a : Lowest value of uniform random variable.
    b : Highest value of uniform random variable.

    Returns
    -------
    Array with uniform probability distribution for tanb.

    """
    unif_val = np.random.uniform(a, b)
    return unif_val


def main():
    """Create random NUHM3 input with linear draw on soft terms for testing."""
    m0_12_rand_val = lin_dist_func(100.0, 60000.0)
    m0_3_rand_val = lin_dist_func(100.0, np.min([20000.0, m0_12_rand_val]))
    mhf_rand_val = lin_dist_func(500.0, 10000.0)
    A0_rand_val = lin_dist_func(-50000.0, 0.0)
    mA_rand_val = lin_dist_func(300.0, 10000.0)
    tanb_rand_val = unif_dist_func(3, 60)
    print('Block MODSEL\n' + '    1   1\n'
          + 'Block SMINPUTS\n' + '    1   1.279340000e+02\n'
          + '    2   1.166370000e-05\n' + '    3   1.172000000e-01\n'
          + '    4   9.118760000e+01\n' + '    5   4.250000000e+00\n'
          + '    6   1.732000000e+02\n' + '    7   1.777000000e+00\n'
          + 'Block MINPAR\n' + '    2   '
          + '{:.9e}'.format(mhf_rand_val)
          + '\n    3   ' + '{:.9e}'.format(tanb_rand_val)
          + '\n    4   1.000000000e+00\n'
          + '    5   ' + '{:.9e}'.format(A0_rand_val)
          + '\nBlock SOFTSUSY\n'
          + '    0   0.000000000e+00\n' + '    1   1.000000000e-03\n'
          + '    2   0.000000000e+00\n' + '    3   0.000000000e+00\n'
          + '    4   1.000000000e+00\n' + '    6   1.000000000e-04\n'
          + '    7   3.000000000e+00\n' + '   10   0.000000000e+00\n'
          + '   11   1.000000000e+19\n' + '   12   1.000000000e+00\n'
          + '   13   0.000000000e+00\n' + '   19   1.000000000e+00\n'
          + '   20   3.100000000e+01\n' + '   21   1.000000000e+00\n'
          + 'Block EXTPAR\n' + '   23   2.000000000e+02\n'
          + '   26   ' + '{:.9e}'.format(mA_rand_val) + '\n   31   '
          + '{:.9e}'.format(m0_12_rand_val) + '\n   32   '
          + '{:.9e}'.format(m0_12_rand_val) + '\n   33   '
          + '{:.9e}'.format(m0_3_rand_val) + '\n   34   '
          + '{:.9e}'.format(m0_12_rand_val) + '\n   35   '
          + '{:.9e}'.format(m0_12_rand_val) + '\n   36   '
          + '{:.9e}'.format(m0_3_rand_val) + '\n   41   '
          + '{:.9e}'.format(m0_12_rand_val) + '\n   42   '
          + '{:.9e}'.format(m0_12_rand_val) + '\n   43   '
          + '{:.9e}'.format(m0_3_rand_val) + '\n   44   '
          + '{:.9e}'.format(m0_12_rand_val) + '\n   45   '
          + '{:.9e}'.format(m0_12_rand_val) + '\n   46   '
          + '{:.9e}'.format(m0_3_rand_val) + '\n   47   '
          + '{:.9e}'.format(m0_12_rand_val) + '\n   48   '
          + '{:.9e}'.format(m0_12_rand_val) + '\n   49   '
          + '{:.9e}'.format(m0_3_rand_val), file=open('test_in_6', "w"))


if __name__ == "__main__":
    main()
