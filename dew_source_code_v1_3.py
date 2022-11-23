#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 11:34:53 2022

Compute naturalness measures Delta_EW, Delta_BG, and Delta_HS,
 and individual radiative corrections to the Higgs potential.

Author: Dakotah Martinez
"""

import numpy as np
from scipy.special import spence
import pyslha
import time
from scipy.integrate import solve_ivp
from math import ceil
import os

def my_RGE_solver(BCs, inp_Q, target_Q_val=2000.0):
    """
    Use scipy.integrate to evolve MSSM RGEs and collect solution vectors.

    Parameters
    ----------
    BCs : Array of floats.
        GUT scale boundary conditions for RGEs.
    inp_Q : Float.
        Highest value for t parameter to run to in solution,
            typically unification scale from SoftSUSY.
    target_Q_val : Float.
        Lowest value for t parameter to run to in solution. Default is 2 TeV.

    Returns
    -------
    sol_arrs: [float, float, Array of floats, array of floats].
        Return solutions to system of RGEs and scales as
        [GUT scale, weak scale, weak scale solutions, GUT scale solutions].
        See before return statement for a comment on return array ordering.

    """
    def my_odes(t, x):
        """
        Define two-loop RGEs for soft terms.

        Parameters
        ----------
        x : Array of floats.
            Numerical solutions to RGEs. The order of entries in x is:
              (0: g1, 1: g2, 2: g3, 3: M1, 4: M2, 5: M3, 6: mu, 7: yt, 8: yc,
               9: yu, 10: yb, 11: ys, 12: yd, 13: ytau, 14: ymu, 15: ye,
               16: at, 17: ac, 18: au, 19: ab, 20: as, 21: ad, 22: atau,
               23: amu, 24: ae, 25: mHu^2, 26: mHd^2, 27: mQ1^2,
               28: mQ2^2, 29: mQ3^2, 30: mL1^2, 31: mL2^2, 32: mL3^2,
               33: mU1^2, 34: mU2^2, 35: mU3^2, 36: mD1^2, 37: mD2^2,
               38: mD3^2, 39: mE1^2, 40: mE2^2, 41: mE3^2, 42: b, 43: tanb)
        t : Array of evaluation renormalization scales.
            t = Q values for numerical solutions.

        Returns
        -------
        Array of floats.
            Return all soft RGEs evaluated at current t value.

        """
        # Unification scale is acquired from running a BM point through
        # SoftSUSY, then GUT scale boundary conditions are acquired from
        # SoftSUSY so that all three generations of Yukawas (assumed
        # to be diagonalized) are accounted for. A universal boundary condition
        # is used for soft scalar trilinear couplings a_i=y_i*A_i.
        # The soft b^(ij) mass^2 term is defined as b=B*mu, but is computed
        # in a later iteration.
        # Scalar mass matrices will also be written in diagonalized form such
        # that, e.g., mQ^2=((mQ1^2,0,0),(0,mQ2^2,0),(0,0,mQ3^2)).

        # Define all parameters in terms of solution vector x
        g1_val = x[0]
        g2_val = x[1]
        g3_val = x[2]
        M1_val = x[3]
        M2_val = x[4]
        M3_val = x[5]
        mu_val = x[6]
        yt_val = x[7]
        yc_val = x[8]
        yu_val = x[9]
        yb_val = x[10]
        ys_val = x[11]
        yd_val = x[12]
        ytau_val = x[13]
        ymu_val = x[14]
        ye_val = x[15]
        at_val = x[16]
        ac_val = x[17]
        au_val = x[18]
        ab_val = x[19]
        as_val = x[20]
        ad_val = x[21]
        atau_val = x[22]
        amu_val = x[23]
        ae_val = x[24]
        mHu_sq_val = x[25]
        mHd_sq_val = x[26]
        mQ1_sq_val = x[27]
        mQ2_sq_val = x[28]
        mQ3_sq_val = x[29]
        mL1_sq_val = x[30]
        mL2_sq_val = x[31]
        mL3_sq_val = x[32]
        mU1_sq_val = x[33]
        mU2_sq_val = x[34]
        mU3_sq_val = x[35]
        mD1_sq_val = x[36]
        mD2_sq_val = x[37]
        mD3_sq_val = x[38]
        mE1_sq_val = x[39]
        mE2_sq_val = x[40]
        mE3_sq_val = x[41]
        b_val = x[42]
        tanb_val = x[43]

        ##### Gauge couplings and gaugino masses #####
        # 1 loop parts
        dg1_dt_1l = b_1l[0] * np.power(g1_val, 3)

        dg2_dt_1l = b_1l[1] * np.power(g2_val, 3)

        dg3_dt_1l = b_1l[2] * np.power(g3_val, 3)

        dM1_dt_1l = b_1l[0] * np.power(g1_val, 2) * M1_val

        dM2_dt_1l = b_1l[1] * np.power(g2_val, 2) * M2_val

        dM3_dt_1l = b_1l[2] * np.power(g3_val, 2) * M3_val

        # 2 loop parts
        dg1_dt_2l = (np.power(g1_val, 3)
                     * ((b_2l[0][0] * np.power(g1_val, 2))
                        + (b_2l[0][1] * np.power(g2_val, 2))
                        + (b_2l[0][2] * np.power(g3_val, 2))# Tr(Yu^2)
                        - (c_2l[0][0] * (np.power(yt_val, 2)
                                         + np.power(yc_val, 2)
                                         + np.power(yu_val, 2)))# end trace, begin Tr(Yd^2)
                        - (c_2l[0][1] * (np.power(yb_val, 2)
                                         + np.power(ys_val, 2)
                                         + np.power(yd_val, 2)))# end trace, begin Tr(Ye^2)
                        - (c_2l[0][2] * (np.power(ytau_val, 2)
                                         + np.power(ymu_val, 2)
                                         + np.power(ye_val, 2)))))# end trace

        dg2_dt_2l = (np.power(g2_val, 3)
                     * ((b_2l[1][0] * np.power(g1_val, 2))
                        + (b_2l[1][1] * np.power(g2_val, 2))
                        + (b_2l[1][2] * np.power(g3_val, 2))# Tr(Yu^2)
                        - (c_2l[1][0] * (np.power(yt_val, 2)
                                         + np.power(yc_val, 2)
                                         + np.power(yu_val, 2)))# end trace, begin Tr(Yd^2)
                        - (c_2l[1][1] * (np.power(yb_val, 2)
                                         + np.power(ys_val, 2)
                                         + np.power(yd_val, 2)))# end trace, begin Tr(Ye^2)
                        - (c_2l[1][2] * (np.power(ytau_val, 2)
                                         + np.power(ymu_val, 2)
                                         + np.power(ye_val, 2)))))# end trace

        dg3_dt_2l = (np.power(g3_val, 3)
                     * ((b_2l[2][0] * np.power(g1_val, 2))
                        + (b_2l[2][1] * np.power(g2_val, 2))
                        + (b_2l[2][2] * np.power(g3_val, 2))# Tr(Yu^2)
                        - (c_2l[2][0] * (np.power(yt_val, 2)
                                         + np.power(yc_val, 2)
                                         + np.power(yu_val, 2)))# end trace, begin Tr(Yd^2)
                        - (c_2l[2][1] * (np.power(yb_val, 2)
                                         + np.power(ys_val, 2)
                                         + np.power(yd_val, 2)))# end trace, begin Tr(Ye^2)
                        - (c_2l[2][2] * (np.power(ytau_val, 2)
                                         + np.power(ymu_val, 2)
                                         + np.power(ye_val, 2)))))# end trace

        dM1_dt_2l = (2 * np.power(g1_val, 2)
                     * (((b_2l[0][0] * np.power(g1_val, 2) * (M1_val + M1_val))
                         + (b_2l[0][1] * np.power(g2_val, 2)
                            * (M1_val + M2_val))
                         + (b_2l[0][2] * np.power(g3_val, 2)
                            * (M1_val + M3_val)))# Tr(Yu*au)
                        + ((c_2l[0][0] * (((yt_val * at_val)
                                           + (yc_val * ac_val)
                                           + (yu_val * au_val))# end trace, begin Tr(Yu^2)
                                          - (M1_val * (np.power(yt_val, 2)
                                                       + np.power(yc_val, 2)
                                                       + np.power(yu_val, 2)))# end trace
                                          )))# Tr(Yd*ad)
                        + ((c_2l[0][1] * (((yb_val * ab_val)
                                           + (ys_val * as_val)
                                           + (yd_val * ad_val))# end trace, begin Tr(Yd^2)
                                          - (M1_val * (np.power(yb_val, 2)
                                                       + np.power(ys_val, 2)
                                                       + np.power(yd_val, 2)))# end trace
                                          )))# Tr(Ye*ae)
                        + ((c_2l[0][2] * (((ytau_val * atau_val)
                                           + (ymu_val * amu_val)
                                           + (ye_val * ae_val))# end trace, begin Tr(Ye^2)
                                          - (M1_val * (np.power(ytau_val, 2)
                                                       + np.power(ymu_val, 2)
                                                       + np.power(ye_val, 2)))
                                          )))))

        dM2_dt_2l = (2 * np.power(g2_val, 2)
                     * (((b_2l[1][0] * np.power(g1_val, 2) * (M2_val + M1_val))
                         + (b_2l[1][1] * np.power(g2_val, 2)
                            * (M2_val + M2_val))
                         + (b_2l[1][2] * np.power(g3_val, 2)
                            * (M2_val + M3_val)))# Tr(Yu*au)
                        + ((c_2l[1][0] * (((yt_val * at_val)
                                           + (yc_val * ac_val)
                                           + (yu_val * au_val))# end trace, begin Tr(Yu^2)
                                          - (M2_val * (np.power(yt_val, 2)
                                                       + np.power(yc_val, 2)
                                                       + np.power(yu_val, 2)))# end trace
                                          )))# Tr(Yd*ad)
                        + ((c_2l[1][1] * (((yb_val * ab_val)
                                           + (ys_val * as_val)
                                           + (yd_val * ad_val))# end trace, begin Tr(Yd^2)
                                          - (M2_val * (np.power(yb_val, 2)
                                                       + np.power(ys_val, 2)
                                                       + np.power(yd_val, 2)))# end trace
                                          )))# Tr(Ye*ae)
                        + ((c_2l[1][2] * (((ytau_val * atau_val)
                                           + (ymu_val * amu_val)
                                           + (ye_val * ae_val))# end trace, begin Tr(Ye^2)
                                          - (M2_val * (np.power(ytau_val, 2)
                                                       + np.power(ymu_val, 2)
                                                       + np.power(ye_val, 2)))# end trace
                                          )))))

        dM3_dt_2l = (2 * np.power(g3_val, 2)
                     * (((b_2l[2][0] * np.power(g1_val, 2) * (M3_val + M1_val))
                         + (b_2l[2][1] * np.power(g2_val, 2)
                            * (M3_val + M2_val))
                         + (b_2l[2][2] * np.power(g3_val, 2)
                            * (M3_val + M3_val)))# Tr(Yu*au)
                        + ((c_2l[2][0] * (((yt_val * at_val)
                                           + (yc_val * ac_val)
                                           + (yu_val * au_val))# end trace, begin Tr(Yu^2)
                                          - (M3_val * (np.power(yt_val, 2)
                                                       + np.power(yc_val, 2)
                                                       + np.power(yu_val, 2)))# end trace
                                          )))# Tr(Yd*ad)
                        + ((c_2l[2][1] * (((yb_val * ab_val)
                                           + (ys_val * as_val)
                                           + (yd_val * ad_val))# end trace, begin Tr(Yd^2)
                                          - (M3_val * (np.power(yb_val, 2)
                                                       + np.power(ys_val, 2)
                                                       + np.power(yd_val, 2)))# end trace
                                          )))# Tr(Ye*ae)
                        + ((c_2l[2][2] * (((ytau_val * atau_val)
                                           + (ymu_val * amu_val)
                                           + (ye_val * ae_val))# end trace, begin Tr(Ye^2)
                                          - (M3_val * (np.power(ytau_val, 2)
                                                       + np.power(ymu_val, 2)
                                                       + np.power(ye_val, 2)))# end trace
                                          )))))

        # Total gauge and gaugino mass beta functions
        dg1_dt = (1 / t) * ((loop_fac * dg1_dt_1l)
                            + (loop_fac_sq * dg1_dt_2l))

        dg2_dt = (1 / t) * ((loop_fac * dg2_dt_1l)
                            + (loop_fac_sq * dg2_dt_2l))

        dg3_dt = (1 / t) * ((loop_fac * dg3_dt_1l)
                            + (loop_fac_sq * dg3_dt_2l))

        dM1_dt = (2 / t) * ((loop_fac * dM1_dt_1l)
                             + (loop_fac_sq * dM1_dt_2l))

        dM2_dt = (2 / t) * ((loop_fac * dM2_dt_1l)
                             + (loop_fac_sq * dM2_dt_2l))

        dM3_dt = (2 / t) * ((loop_fac * dM3_dt_1l)
                             + (loop_fac_sq * dM3_dt_2l))

        ##### Higgsino mass parameter mu #####
        # 1 loop part
        dmu_dt_1l = (mu_val# Tr(3Yu^2 + 3Yd^2 + Ye^2)
                     * ((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2) + np.power(yb_val, 2)
                              + np.power(ys_val, 2) + np.power(yd_val, 2)))
                        + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                           + np.power(ye_val, 2))# end trace
                        - (3 * np.power(g2_val, 2))
                        - ((3 / 5) * np.power(g1_val, 2))))

        # 2 loop part
        dmu_dt_2l = (mu_val# Tr(3Yu^4 + 3Yd^4 + (2Yu^2*Yd^2) + Ye^4)
                     * ((-3 * ((3 * (np.power(yt_val, 4) + np.power(yc_val, 4)
                                     + np.power(yu_val, 4)
                                     + np.power(yb_val, 4)
                                     + np.power(ys_val, 4)
                                     + np.power(yd_val, 4)))
                               + (2 * ((np.power(yt_val, 2)
                                        * np.power(yb_val, 2))
                                       + (np.power(yc_val, 2)
                                          * np.power(ys_val, 2))
                                       + (np.power(yu_val, 2)
                                          * np.power(yd_val, 2))))
                               + (np.power(ytau_val, 4) + np.power(ymu_val, 4)
                                  + np.power(ye_val, 4))))# end trace
                        + (((16 * np.power(g3_val, 2))
                            + (4 * np.power(g1_val, 2) / 5))# Tr(Yu^2)
                           * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        + (((16 * np.power(g3_val, 2))
                            - (2 * np.power(g1_val, 2) / 5))# Tr(Yd^2)
                           * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))# end trace
                        + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                           * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))# end trace
                        + ((15 / 2) * np.power(g2_val, 4))
                        + ((9 / 5) * np.power(g1_val, 2)
                           * np.power(g2_val, 2))
                        + ((207 / 50) * np.power(g1_val, 4))))

        # Total mu beta function
        dmu_dt = (1 / t) * ((loop_fac * dmu_dt_1l)
                            + (loop_fac_sq * dmu_dt_2l))

        ##### Yukawa couplings for all 3 generations, assumed diagonalized#####
        # 1 loop parts
        dyt_dt_1l = (yt_val# Tr(3Yu^2)
                     * ((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        + (3 * (np.power(yt_val, 2)))
                        + np.power(yb_val, 2)
                        - ((16 / 3) * np.power(g3_val, 2))
                        - (3 * np.power(g2_val, 2))
                        - ((13 / 15) * np.power(g1_val, 2))))

        dyc_dt_1l = (yc_val# Tr(3Yu^2)
                     * ((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        + (3 * (np.power(yc_val, 2)))
                        + np.power(ys_val, 2)
                        - ((16 / 3) * np.power(g3_val, 2))
                        - (3 * np.power(g2_val, 2))
                        - ((13 / 15) * np.power(g1_val, 2))))

        dyu_dt_1l = (yu_val# Tr(3Yu^2)
                     * ((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        + (3 * (np.power(yu_val, 2)))
                        + np.power(yd_val, 2)
                        - ((16 / 3) * np.power(g3_val, 2))
                        - (3 * np.power(g2_val, 2))
                        - ((13 / 15) * np.power(g1_val, 2))))

        dyb_dt_1l = (yb_val# Tr(3Yd^2 + Ye^2)
                     * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))
                        + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                           + np.power(ye_val, 2))# end trace
                        + (3 * (np.power(yb_val, 2))) + np.power(yt_val, 2)
                        - ((16 / 3) * np.power(g3_val, 2))
                        - (3 * np.power(g2_val, 2))
                        - ((7 / 15) * np.power(g1_val, 2))))

        dys_dt_1l = (ys_val# Tr(3Yd^2 + Ye^2)
                     * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))
                        + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                           + np.power(ye_val, 2))# end trace
                        + (3 * (np.power(ys_val, 2))) + np.power(yc_val, 2)
                        - ((16 / 3) * np.power(g3_val, 2))
                        - (3 * np.power(g2_val, 2))
                        - ((7 / 15) * np.power(g1_val, 2))))

        dyd_dt_1l = (yd_val# Tr(3Yd^2 + Ye^2)
                     * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))
                        + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                           + np.power(ye_val, 2))# end trace
                        + (3 * (np.power(yd_val, 2))) + np.power(yu_val, 2)
                        - ((16 / 3) * np.power(g3_val, 2))
                        - (3 * np.power(g2_val, 2))
                        - ((7 / 15) * np.power(g1_val, 2))))

        dytau_dt_1l = (ytau_val# Tr(3Yd^2 + Ye^2)
                       * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                + np.power(yd_val, 2)))
                          + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                             + np.power(ye_val, 2))# end trace
                          + (3 * (np.power(ytau_val, 2)))
                          - (3 * np.power(g2_val, 2))
                          - ((9 / 5) * np.power(g1_val, 2))))

        dymu_dt_1l = (ymu_val# Tr(3Yd^2 + Ye^2)
                      * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                               + np.power(yd_val, 2)))
                         + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                            + np.power(ye_val, 2))# end trace
                         + (3 * (np.power(ymu_val, 2)))
                         - (3 * np.power(g2_val, 2))
                         - ((9 / 5) * np.power(g1_val, 2))))

        dye_dt_1l = (ye_val# Tr(3Yd^2 + Ye^2)
                     * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))
                        + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                           + np.power(ye_val, 2))# end trace
                        + (3 * (np.power(ye_val, 2)))
                        - (3 * np.power(g2_val, 2))
                        - ((9 / 5) * np.power(g1_val, 2))))

        # 2 loop parts
        dyt_dt_2l = (yt_val # Tr(3Yu^4 + (Yu^2*Yd^2))
                     * (((-3) * ((3 * (np.power(yt_val, 4)
                                       + np.power(yc_val, 4)
                                       + np.power(yu_val, 4)))
                                 + (np.power(yt_val, 2) * np.power(yb_val, 2))
                                 + (np.power(yc_val, 2) * np.power(ys_val, 2))
                                 + (np.power(yu_val, 2) * np.power(yd_val, 2)))
                         )# end trace
                        - (np.power(yb_val, 2)# Tr(3Yd^2 + Ye^2)
                           * ((3 * (np.power(yb_val, 2)
                                    + np.power(ys_val, 2)
                                    + np.power(yd_val, 2)))
                              + (np.power(ytau_val, 2)
                                 + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2))))# end trace
                        - (9 * np.power(yt_val, 2)# Tr(Yu^2)
                           * (np.power(yt_val, 2)
                              + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        - (4 * np.power(yt_val, 4))
                        - (2 * np.power(yb_val, 4))
                        - (2 * np.power(yb_val, 2) * np.power(yt_val, 2))
                        + (((16 * np.power(g3_val, 2))
                            + ((4 / 5) * np.power(g1_val, 2)))# Tr(Yu^2)
                           * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        + (((6 * np.power(g2_val, 2))
                            + ((2 / 5) * np.power(g1_val, 2)))
                           * np.power(yt_val, 2))
                        + ((2 / 5) * np.power(g1_val, 2) * np.power(yb_val, 2))
                        - ((16 / 9) * np.power(g3_val, 4))
                        + (8 * np.power(g3_val, 2) * np.power(g2_val, 2))
                        + ((136 / 45) * np.power(g3_val, 2)
                           * np.power(g1_val, 2))
                        + ((15 / 2) * np.power(g2_val, 4))
                        + (np.power(g2_val, 2) * np.power(g1_val, 2))
                        + ((2743 / 450) * np.power(g1_val, 4))))

        dyc_dt_2l = (yc_val # Tr(3Yu^4 + (Yu^2*Yd^2))
                     * (((-3) * ((3 * (np.power(yt_val, 4)
                                       + np.power(yc_val, 4)
                                       + np.power(yu_val, 4)))
                                 + (np.power(yt_val, 2) * np.power(yb_val, 2))
                                 + (np.power(yc_val, 2) * np.power(ys_val, 2))
                                 + (np.power(yu_val, 2) * np.power(yd_val, 2)))
                         )#end trace
                        - (np.power(ys_val, 2)# Tr(3Yd^2 + Ye^2)
                           * ((3 * (np.power(yb_val, 2)
                                    + np.power(ys_val, 2)
                                    + np.power(yd_val, 2)))
                              + (np.power(ytau_val, 2)
                                 + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2))))# end trace
                        - (9 * np.power(yc_val, 2)# Tr(Yu^2)
                           * (np.power(yt_val, 2)
                              + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        - (4 * np.power(yc_val, 4))
                        - (2 * np.power(ys_val, 4))
                        - (2 * np.power(ys_val, 2)
                           * np.power(yc_val, 2))
                        + (((16 * np.power(g3_val, 2))
                            + ((4 / 5) * np.power(g1_val, 2)))# Tr(Yu^2)
                           * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        + (((6 * np.power(g2_val, 2))
                            + ((2 / 5) * np.power(g1_val, 2)))
                           * np.power(yc_val, 2))
                        + ((2 / 5) * np.power(g1_val, 2) * np.power(ys_val, 2))
                        - ((16 / 9) * np.power(g3_val, 4))
                        + (8 * np.power(g3_val, 2) * np.power(g2_val, 2))
                        + ((136 / 45) * np.power(g3_val, 2)
                           * np.power(g1_val, 2))
                        + ((15 / 2) * np.power(g2_val, 4))
                        + (np.power(g2_val, 2) * np.power(g1_val, 2))
                        + ((2743 / 450) * np.power(g1_val, 4))))

        dyu_dt_2l = (yu_val# Tr(3Yu^4 + (Yu^2*Yd^2))
                     * (((-3) * ((3 * (np.power(yt_val, 4)
                                       + np.power(yc_val, 4)
                                       + np.power(yu_val, 4)))
                                 + (np.power(yt_val, 2) * np.power(yb_val, 2))
                                 + (np.power(yc_val, 2) * np.power(ys_val, 2))
                                 + (np.power(yu_val, 2) * np.power(yd_val, 2)))
                         )# end trace
                        - (np.power(yd_val, 2)# Tr(3Yd^2 + Ye^2)
                           * ((3 * (np.power(yb_val, 2)
                                    + np.power(ys_val, 2)
                                    + np.power(yd_val, 2)))
                              + (np.power(ytau_val, 2)
                                 + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2))))
                        - (9 * np.power(yu_val, 2)# Tr(Yu^2)
                           * (np.power(yt_val, 2)
                              + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))
                        - (4 * np.power(yu_val, 4))
                        - (2 * np.power(yd_val, 4))
                        - (2 * np.power(yd_val, 2) * np.power(yu_val, 2))
                        + (((16 * np.power(g3_val, 2))
                            + ((4 / 5) * np.power(g1_val, 2)))# Tr(Yu^2)
                           * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        + (((6 * np.power(g2_val, 2))
                            + ((2 / 5) * np.power(g1_val, 2)))
                           * np.power(yu_val, 2))
                        + ((2 / 5) * np.power(g1_val, 2) * np.power(yd_val, 2))
                        - ((16 / 9) * np.power(g3_val, 4))
                        + (8 * np.power(g3_val, 2) * np.power(g2_val, 2))
                        + ((136 / 45) * np.power(g3_val, 2)
                           * np.power(g1_val, 2))
                        + ((15 / 2) * np.power(g2_val, 4))
                        + (np.power(g2_val, 2) * np.power(g1_val, 2))
                        + ((2743 / 450) * np.power(g1_val, 4))))

        dyb_dt_2l = (yb_val# Tr(3Yd^4 + (Yu^2*Yd^2) + Ye^4)
                     * (((-3) * ((3 * (np.power(yb_val, 4)
                                       + np.power(ys_val, 4)
                                       + np.power(yd_val, 4)))
                                 + (np.power(yt_val, 2) * np.power(yb_val, 2))
                                 + (np.power(yc_val, 2) * np.power(ys_val, 2))
                                 + (np.power(yu_val, 2) * np.power(yd_val, 2))
                                 + np.power(ytau_val, 4) + np.power(ymu_val, 4)
                                 + np.power(ye_val, 4)))# end trace
                        - (3 * np.power(yt_val, 2)# Tr(Yu^2)
                           * (np.power(yt_val, 2)
                              + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        - (3 * np.power(yb_val, 2)# Tr(3Yd^2 + Ye^2)
                           * ((3 * (np.power(yb_val, 2)
                                    + np.power(ys_val, 2)
                                    + np.power(yd_val, 2)))
                              + np.power(ytau_val, 2)
                              + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))# end trace
                        - (4 * np.power(yb_val, 4))
                        - (2 * np.power(yt_val, 4))
                        - (2 * np.power(yt_val, 2) * np.power(yb_val, 2))
                        + (((16 * np.power(g3_val, 2))
                            - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                           * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))# end trace
                        + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                           * (np.power(ytau_val, 2)
                              + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))# end trace
                        + ((4 / 5) * np.power(g1_val, 2)
                           * np.power(yt_val, 2))
                        + (np.power(yb_val, 2)
                           * ((6 * np.power(g2_val, 2))
                              + ((4 / 5) * np.power(g1_val, 2))))
                        - ((16 / 9) * np.power(g3_val, 4))
                        + (8 * np.power(g3_val, 2) * np.power(g2_val, 2))
                        + ((8 / 9) * np.power(g3_val, 2) * np.power(g1_val, 2))
                        + ((15 / 2) * np.power(g2_val, 4))
                        + (np.power(g2_val, 2) * np.power(g1_val, 2))
                        + ((287 / 90) * np.power(g1_val, 4))))

        dys_dt_2l = (ys_val# Tr(3Yd^4 + (Yu^2*Yd^2) + Ye^4)
                     * (((-3) * ((3 * (np.power(yb_val, 4)
                                       + np.power(ys_val, 4)
                                       + np.power(yd_val, 4)))
                                 + (np.power(yt_val, 2) * np.power(yb_val, 2))
                                 + (np.power(yc_val, 2) * np.power(ys_val, 2))
                                 + (np.power(yu_val, 2) * np.power(yd_val, 2))
                                 + np.power(ytau_val, 4)
                                 + np.power(ymu_val, 4)
                                 + np.power(ye_val, 4)))# end trace
                        - (3 * np.power(yc_val, 2)# Tr(Yu^2)
                           * (np.power(yt_val, 2)
                              + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        - (3 * np.power(ys_val, 2)# Tr(3Yd^2 + Ye^2)
                           * ((3 * (np.power(yb_val, 2)
                                    + np.power(ys_val, 2)
                                    + np.power(yd_val, 2)))
                              + np.power(ytau_val, 2)
                              + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))# end trace
                        - (4 * np.power(ys_val, 4))
                        - (2 * np.power(yc_val, 4))
                        - (2 * np.power(yc_val, 2) * np.power(ys_val, 2))
                        + (((16 * np.power(g3_val, 2))
                            - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                           * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))# end trace
                        + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                           * (np.power(ytau_val, 2)
                              + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))# end trace
                        + ((4 / 5) * np.power(g1_val, 2) * np.power(yc_val, 2))
                        + (np.power(ys_val, 2)
                           * ((6 * np.power(g2_val, 2))
                              + ((4 / 5) * np.power(g1_val, 2))))
                        - ((16 / 9) * np.power(g3_val, 4))
                        + (8 * np.power(g3_val, 2) * np.power(g2_val, 2))
                        + ((8 / 9) * np.power(g3_val, 2) * np.power(g1_val, 2))
                        + ((15 / 2) * np.power(g2_val, 4))
                        + (np.power(g2_val, 2) * np.power(g1_val, 2))
                        + ((287 / 90) * np.power(g1_val, 4))))

        dyd_dt_2l = (yd_val# Tr(3Yd^4 + (Yu^2*Yd^2) + Ye^4)
                     * (((-3) * ((3 * (np.power(yb_val, 4)
                                       + np.power(ys_val, 4)
                                       + np.power(yd_val, 4)))
                                 + (np.power(yt_val, 2) * np.power(yb_val, 2))
                                 + (np.power(yc_val, 2) * np.power(ys_val, 2))
                                 + (np.power(yu_val, 2) * np.power(yd_val, 2))
                                 + np.power(ytau_val, 4) + np.power(ymu_val, 4)
                                 + np.power(ye_val, 4)))# end trace
                        - (3 * np.power(yu_val, 2)# Tr(Yu^2)
                           * (np.power(yt_val, 2)
                              + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        - (3 * np.power(yd_val, 2)# Tr(3Yd^2 + Ye^2)
                           * ((3 * (np.power(yb_val, 2)
                                    + np.power(ys_val, 2)
                                    + np.power(yd_val, 2)))
                              + np.power(ytau_val, 2)
                              + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))# end trace
                        - (4 * np.power(yd_val, 4))
                        - (2 * np.power(yu_val, 4))
                        - (2 * np.power(yd_val, 2) * np.power(yu_val, 2))
                        + (((16 * np.power(g3_val, 2))
                            - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                           * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))# end trace
                        + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                           * (np.power(ytau_val, 2)
                              + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))# end trace
                        + ((4 / 5) * np.power(g1_val, 2) * np.power(yu_val, 2))
                        + (np.power(yd_val, 2)
                           * ((6 * np.power(g2_val, 2))
                              + ((4 / 5) * np.power(g1_val, 2))))
                        - ((16 / 9) * np.power(g3_val, 4))
                        + (8 * np.power(g3_val, 2) * np.power(g2_val, 2))
                        + ((8 / 9) * np.power(g3_val, 2) * np.power(g1_val, 2))
                        + ((15 / 2) * np.power(g2_val, 4))
                        + (np.power(g2_val, 2) * np.power(g1_val, 2))
                        + ((287 / 90) * np.power(g1_val, 4))))

        dytau_dt_2l = (ytau_val# Tr(3Yd^4 + (Yu^2*Yd^2) + Ye^4)
                       * (((-3) * ((3 * (np.power(yb_val, 4)
                                         + np.power(ys_val, 4)
                                         + np.power(yd_val, 4)))
                                   + (np.power(yt_val, 2)
                                      * np.power(yb_val, 2))
                                   + (np.power(yc_val, 2)
                                      * np.power(ys_val, 2))
                                   + (np.power(yu_val, 2)
                                      * np.power(yd_val, 2))
                                   + np.power(ytau_val, 4)
                                   + np.power(ymu_val, 4)
                                   + np.power(ye_val, 4)))# end trace
                          - (3 * np.power(ytau_val, 2)# Tr(3Yd^2 + Ye^2)
                             * ((3 * (np.power(yb_val, 2)
                                      + np.power(ys_val, 2)
                                      + np.power(yd_val, 2)))
                                + np.power(ytau_val, 2)
                                + np.power(ymu_val, 2)
                                + np.power(ye_val, 2)))# end trace
                          - (4 * np.power(ytau_val, 4))
                          + (((16 * np.power(g3_val, 2))
                              - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                             * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                + np.power(yd_val, 2)))# end trace
                          + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                             * (np.power(ytau_val, 2)
                                + np.power(ymu_val, 2)
                                + np.power(ye_val, 2)))# end trace
                          + (6 * np.power(g2_val, 2) * np.power(ytau_val, 2))
                          + ((15 / 2) * np.power(g2_val, 4))
                          + ((9 / 5) * np.power(g2_val, 2)
                             * np.power(g1_val, 2))
                          + ((27 / 2) * np.power(g1_val, 4))))

        dymu_dt_2l = (ymu_val# Tr(3Yd^4 + (Yu^2*Yd^2) + Ye^4)
                      * (((-3) * ((3 * (np.power(yb_val, 4)
                                        + np.power(ys_val, 4)
                                        + np.power(yd_val, 4)))
                                  + (np.power(yt_val, 2)
                                     * np.power(yb_val, 2))
                                  + (np.power(yc_val, 2)
                                     * np.power(ys_val, 2))
                                  + (np.power(yu_val, 2)
                                     * np.power(yd_val, 2))
                                  + np.power(ytau_val, 4)
                                  + np.power(ymu_val, 4)
                                  + np.power(ye_val, 4)))# end trace
                         - (3 * np.power(ymu_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2)
                                     + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2)
                               + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (4 * np.power(ymu_val, 4))
                         + (((16 * np.power(g3_val, 2))
                             - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                            * (np.power(yb_val, 2) + np.power(ys_val, 2)
                               + np.power(yd_val, 2)))# end trace
                         + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                            * (np.power(ytau_val, 2)
                               + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         + (6 * np.power(g2_val, 2) * np.power(ymu_val, 2))
                         + ((15 / 2) * np.power(g2_val, 4))
                         + ((9 / 5) * np.power(g2_val, 2) * np.power(g1_val, 2))
                         + ((27 / 2) * np.power(g1_val, 4))))

        dye_dt_2l = (ye_val# Tr(3Yd^4 + (Yu^2*Yd^2) + Ye^4)
                     * (((-3) * ((3 * (np.power(yb_val, 4)
                                       + np.power(ys_val, 4)
                                       + np.power(yd_val, 4)))
                                 + (np.power(yt_val, 2)
                                    * np.power(yb_val, 2))
                                 + (np.power(yc_val, 2)
                                    * np.power(ys_val, 2))
                                 + (np.power(yu_val, 2)
                                    * np.power(yd_val, 2))
                                 + np.power(ytau_val, 4)
                                 + np.power(ymu_val, 4)
                                 + np.power(ye_val, 4)))# end trace
                        - (3 * np.power(ye_val, 2)# Tr(3Yd^2 + Ye^2)
                           * ((3 * (np.power(yb_val, 2)
                                    + np.power(ys_val, 2)
                                    + np.power(yd_val, 2)))
                              + np.power(ytau_val, 2)
                              + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))# end trace
                        - (4 * np.power(ye_val, 4))
                        + (((16 * np.power(g3_val, 2))
                            - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                           * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))# end trace
                        + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                           * (np.power(ytau_val, 2)
                              + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))
                        + (6 * np.power(g2_val, 2) * np.power(ye_val, 2))
                        + ((15 / 2) * np.power(g2_val, 4))
                        + ((9 / 5) * np.power(g2_val, 2) * np.power(g1_val, 2))
                        + ((27 / 2) * np.power(g1_val, 4))))

        # Total Yukawa coupling beta functions
        dyt_dt = (1 / t) * ((loop_fac * dyt_dt_1l)
                            + (loop_fac_sq * dyt_dt_2l))

        dyc_dt = (1 / t) * ((loop_fac * dyc_dt_1l)
                            + (loop_fac_sq * dyc_dt_2l))

        dyu_dt = (1 / t) * ((loop_fac * dyu_dt_1l)
                            + (loop_fac_sq * dyu_dt_2l))

        dyb_dt = (1 / t) * ((loop_fac * dyb_dt_1l)
                            + (loop_fac_sq * dyb_dt_2l))

        dys_dt = (1 / t) * ((loop_fac * dys_dt_1l)
                            + (loop_fac_sq * dys_dt_2l))

        dyd_dt = (1 / t) * ((loop_fac * dyd_dt_1l)
                            + (loop_fac_sq * dyd_dt_2l))

        dytau_dt = (1 / t) * ((loop_fac * dytau_dt_1l)
                            + (loop_fac_sq * dytau_dt_2l))

        dymu_dt = (1 / t) * ((loop_fac * dymu_dt_1l)
                            + (loop_fac_sq * dymu_dt_2l))

        dye_dt = (1 / t) * ((loop_fac * dye_dt_1l)
                            + (loop_fac_sq * dye_dt_2l))

        ##### Soft trilinear couplings for 3 gen, assumed diagonalized #####
        # 1 loop parts
        dat_dt_1l = ((at_val# Tr(Yu^2)
                      * ((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         + (5 * np.power(yt_val, 2)) + np.power(yb_val, 2)
                         - ((16 / 3) * np.power(g3_val, 2))
                         - (3 * np.power(g2_val, 2))
                         - ((13 / 15) * np.power(g1_val, 2))))
                     + (yt_val# Tr(au*Yu)
                        * ((6 * ((at_val * yt_val) + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           + (4 * yt_val * at_val)
                           + (2 * yb_val * ab_val)
                           + ((32 / 3) * np.power(g3_val, 2) * M3_val)
                           + (6 * np.power(g2_val, 2) * M2_val)
                           + ((26 / 15) * np.power(g1_val, 2) * M1_val))))

        dac_dt_1l = ((ac_val# Tr(Yu^2)
                      * ((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         + (5 * np.power(yc_val, 2)) + np.power(ys_val, 2)
                         - ((16 / 3) * np.power(g3_val, 2))
                         - (3 * np.power(g2_val, 2))
                         - ((13 / 15) * np.power(g1_val, 2))))
                     + (yc_val# Tr(au*Yu)
                        * ((6 * ((at_val * yt_val) + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           + (4 * yc_val * ac_val)
                           + (2 * ys_val * as_val)
                           + ((32 / 3) * np.power(g3_val, 2) * M3_val)
                           + (6 * np.power(g2_val, 2) * M2_val)
                           + ((26 / 15) * np.power(g1_val, 2) * M1_val))))

        dau_dt_1l = ((au_val# Tr(Yu^2)
                      * ((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         + (5 * np.power(yu_val, 2)) + np.power(yd_val, 2)
                         - ((16 / 3) * np.power(g3_val, 2))
                         - (3 * np.power(g2_val, 2))
                         - ((13 / 15) * np.power(g1_val, 2))))
                     + (yu_val# Tr(au*Yu)
                        * ((6 * ((at_val * yt_val) + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           + (4 * yu_val * au_val)
                           + (2 * yd_val * ad_val)
                           + ((32 / 3) * np.power(g3_val, 2) * M3_val)
                           + (6 * np.power(g2_val, 2) * M2_val)
                           + ((26 / 15) * np.power(g1_val, 2) * M1_val))))

        dab_dt_1l = ((ab_val# Tr(3Yd^2 + Ye^2)
                      * (((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                + np.power(yd_val, 2)))
                          + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                             + np.power(ye_val, 2)))# end trace
                         + (5 * np.power(yb_val, 2)) + np.power(yt_val, 2)
                         - ((16 / 3) * np.power(g3_val, 2))
                         - (3 * np.power(g2_val, 2))
                         - ((7 / 15) * np.power(g1_val, 2))))
                     + (yb_val# Tr(6ad*Yd + 2ae*Ye)
                        * ((6 * ((ab_val * yb_val) + (as_val * ys_val)
                                 + (ad_val * yd_val)))
                           + (2 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                   + (ae_val * ye_val)))# end trace
                           + (4 * yb_val * ab_val) + (2 * yt_val * at_val)
                           + ((32 / 3) * np.power(g3_val, 2) * M3_val)
                           + (6 * np.power(g2_val, 2) * M2_val)
                           + ((14 / 15) * np.power(g1_val, 2) * M1_val))))

        das_dt_1l = ((as_val# Tr(3Yd^2 + Ye^2)
                      * (((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                + np.power(yd_val, 2)))
                          + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                             + np.power(ye_val, 2)))# end trace
                         + (5 * np.power(ys_val, 2)) + np.power(yc_val, 2)
                         - ((16 / 3) * np.power(g3_val, 2))
                         - (3 * np.power(g2_val, 2))
                         - ((7 / 15) * np.power(g1_val, 2))))
                     + (ys_val# Tr(6ad*Yd + 2ae*Ye)
                        * ((6 * ((ab_val * yb_val) + (as_val * ys_val)
                                 + (ad_val * yd_val)))
                           + (2 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                   + (ae_val * ye_val)))
                           + (4 * ys_val * as_val) + (2 * yc_val * ac_val)
                           + ((32 / 3) * np.power(g3_val, 2) * M3_val)
                           + (6 * np.power(g2_val, 2) * M2_val)
                           + ((14 / 15) * np.power(g1_val, 2) * M1_val))))

        dad_dt_1l = ((ad_val# Tr(3Yd^2 + Ye^2)
                      * (((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                + np.power(yd_val, 2)))
                          + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                             + np.power(ye_val, 2)))# end trace
                         + (5 * np.power(yd_val, 2)) + np.power(yu_val, 2)
                         - ((16 / 3) * np.power(g3_val, 2))
                         - (3 * np.power(g2_val, 2))
                         - ((7 / 15) * np.power(g1_val, 2))))
                     + (yd_val# Tr(6ad*Yd + 2ae*Ye)
                        * ((6 * ((ab_val * yb_val) + (as_val * ys_val)
                                 + (ad_val * yd_val)))
                           + (2 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                   + (ae_val * ye_val)))# end trace
                           + (4 * yd_val * ad_val) + (2 * yu_val * au_val)
                           + ((32 / 3) * np.power(g3_val, 2) * M3_val)
                           + (6 * np.power(g2_val, 2) * M2_val)
                           + ((14 / 15) * np.power(g1_val, 2) * M1_val))))

        datau_dt_1l = ((atau_val# Tr(3Yd^2 + Ye^2)
                        * (((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                  + np.power(yd_val, 2)))
                            + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                           + (5 * np.power(ytau_val, 2))
                           - (3 * np.power(g2_val, 2))
                           - ((9 / 5) * np.power(g1_val, 2))))
                       + (ytau_val# Tr(6ad*Yd + 2ae*Ye)
                          * ((6 * ((ab_val * yb_val) + (as_val * ys_val)
                                   + (ad_val * yd_val)))
                             + (2 * ((atau_val * ytau_val)
                                     + (amu_val * ymu_val)
                                     + (ae_val * ye_val)))# end trace
                             + (4 * ytau_val * atau_val)
                             + (6 * np.power(g2_val, 2) * M2_val)
                             + ((18 / 5) * np.power(g1_val, 2) * M1_val))))

        damu_dt_1l = ((amu_val# Tr(3Yd^2 + Ye^2)
                       * (((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                 + np.power(yd_val, 2)))
                           + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))# end trace
                          + (5 * np.power(ymu_val, 2))
                          - (3 * np.power(g2_val, 2))
                          - ((9 / 5) * np.power(g1_val, 2))))
                      + (ymu_val# Tr(6ad*Yd + 2ae*Ye)
                         * ((6 * ((ab_val * yb_val) + (as_val * ys_val)
                                  + (ad_val * yd_val)))
                            + (2 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                    + (ae_val * ye_val)))# end trace
                            + (4 * ymu_val * amu_val)
                            + (6 * np.power(g2_val, 2) * M2_val)
                            + ((18 / 5) * np.power(g1_val, 2) * M1_val))))

        dae_dt_1l = ((ae_val# Tr(3Yd^2 + Ye^2)
                      * (((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                + np.power(yd_val, 2)))
                          + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                             + np.power(ye_val, 2)))# end trace
                         + (5 * np.power(ye_val, 2))
                         - (3 * np.power(g2_val, 2))
                         - ((9 / 5) * np.power(g1_val, 2))))
                     + (ye_val# Tr(6ad*Yd + 2ae*Ye)
                        * ((6 * ((ab_val * yb_val) + (as_val * ys_val)
                                 + (ad_val * yd_val)))
                           + (2 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                   + (ae_val * ye_val)))# end trace
                           + (4 * ye_val * ae_val)
                           + (6 * np.power(g2_val, 2) * M2_val)
                           + ((18 / 5) * np.power(g1_val, 2) * M1_val))))

        # 2 loop parts
        dat_dt_2l = ((at_val# Tr(3Yu^4 + (Yu^2*Yd^2))
                      * (((-3) * ((3 * (np.power(yt_val, 4)
                                        + np.power(yc_val, 4)
                                        + np.power(yu_val, 4)))
                                  + ((np.power(yt_val, 2) * np.power(yb_val,2))
                                     + (np.power(yc_val, 2)
                                        * np.power(ys_val, 2))
                                     + (np.power(yu_val, 2)
                                        * np.power(yd_val, 2)))))# end trace
                         - (np.power(yb_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2)
                                     + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (np.power(ytau_val, 2)
                                  + np.power(ymu_val, 2)
                                  + np.power(ye_val, 2))))# end trace
                         - (15 * np.power(yt_val, 2)# Tr(Yu^2)
                            * (np.power(yt_val, 2)
                               + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         - (6 * np.power(yt_val, 4))
                         - (2 * np.power(yb_val, 4))
                         - (4 * np.power(yb_val, 2) * np.power(yt_val, 2))
                         + (((16 * np.power(g3_val, 2))
                             + ((4 / 5) * np.power(g1_val, 2)))# Tr(Yu^2)
                            * (np.power(yt_val, 2)
                               + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         + (12 * np.power(g2_val, 2)
                            * np.power(yt_val, 2))
                         + ((2 / 5) * np.power(g1_val, 2)
                            * np.power(yb_val, 2))
                         - ((16 / 9) * np.power(g3_val, 4))
                         + (8 * np.power(g3_val, 2)
                            * np.power(g2_val, 2))
                         + ((136 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2))
                         + ((15 / 2) * np.power(g2_val, 4))
                         + (np.power(g2_val, 2) * np.power(g1_val, 2))
                         + ((2743 / 450) * np.power(g1_val, 4))))
                     + (yt_val# Tr(6au*Yu^3 + au*Yd^2*Yu + ad*Yu^2*Yd)
                        * (((-6) * ((6 * ((at_val * np.power(yt_val, 3))
                                          + (ac_val * np.power(yc_val, 3))
                                          + (au_val * np.power(yu_val, 3))))
                                    + (at_val * np.power(yb_val, 2) * yt_val)
                                    + (ac_val * np.power(ys_val, 2) * yc_val)
                                    + (au_val * np.power(yd_val, 2) * yu_val)
                                    + (ab_val * np.power(yt_val, 2) * yb_val)
                                    + (as_val * np.power(yc_val, 2) * ys_val)
                                    + (ad_val * np.power(yu_val, 2) * yd_val)))# end trace
                           - (18 * np.power(yt_val, 2)# Tr(au*Yu)
                              * ((at_val * yt_val)
                                 + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           - (np.power(yb_val, 2)# Tr(6ad*Yd + 2ae*Ye)
                              * ((6
                                  * ((ab_val * yb_val) + (as_val * ys_val)
                                     + (ad_val * yd_val)))
                                 + (2
                                    * ((atau_val * ytau_val)
                                       + (amu_val * ymu_val)
                                       + (ae_val * ye_val)))))# end trace
                           - (12 * yt_val * at_val# Tr(Yu^2)
                              * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                 + np.power(yu_val, 2)))# end trace
                           - (yb_val * ab_val# Tr(6Yd^2 + 2Ye^2)
                              * ((6 * (np.power(yb_val, 2)
                                       + np.power(ys_val, 2)
                                       + np.power(yd_val, 2)))
                                 + (2 * (np.power(ytau_val, 2)
                                         + np.power(ymu_val, 2)
                                         + np.power(ye_val, 2)))))# end trace
                           - (14 * np.power(yt_val, 3) * at_val)
                           - (8 * np.power(yb_val, 3) * ab_val)
                           - (2 * np.power(yb_val, 2) * yt_val * at_val)
                           - (4 * yb_val * ab_val * np.power(yt_val, 2))
                           + (((32 * np.power(g3_val, 2))
                               + ((8 / 5) * np.power(g1_val, 2)))# Tr(au*Yu)
                              * ((at_val * yt_val) + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           + (((6 * np.power(g2_val, 2))
                               + ((6 / 5) * np.power(g1_val, 2)))
                              * yt_val * at_val)
                           + ((4 / 5) * np.power(g1_val, 2) * yb_val * ab_val)
                           - (((32 * np.power(g3_val, 2) * M3_val)
                               + ((8 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yu^2)
                              * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                 + np.power(yu_val, 2)))# end trace
                           - (((12 * np.power(g2_val, 2) * M2_val)
                               + ((4 / 5) * np.power(g1_val, 2) * M1_val))
                              * np.power(yt_val, 2))
                           - ((4 / 5) * np.power(g1_val, 2) * M1_val
                              * np.power(yb_val, 2))
                           + ((64 / 9) * np.power(g3_val, 4) * M3_val)
                           - (16 * np.power(g3_val, 2) * np.power(g2_val, 2)
                              * (M3_val + M2_val))
                           - ((272 / 45) * np.power(g3_val, 2)
                              * np.power(g1_val, 2)
                              * (M3_val + M1_val))
                           - (30 * np.power(g2_val, 4) * M2_val)
                           - (2 * np.power(g2_val, 2) * np.power(g1_val, 2)
                              * (M2_val + M1_val))
                           - ((5486 / 225) * np.power(g1_val, 4) * M1_val))))

        dac_dt_2l = ((ac_val# Tr(3Yu^4 + (Yu^2*Yd^2))
                      * (((-3) * ((3 * (np.power(yt_val, 4)
                                        + np.power(yc_val, 4)
                                        + np.power(yu_val, 4)))
                                  + ((np.power(yt_val, 2) * np.power(yb_val,2))
                                     + (np.power(yc_val, 2)
                                        * np.power(ys_val, 2))
                                     + (np.power(yu_val, 2)
                                        * np.power(yd_val, 2)))))# end trace
                         - (np.power(ys_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2)
                                     + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (np.power(ytau_val, 2)
                                  + np.power(ymu_val, 2)
                                  + np.power(ye_val, 2))))# end trace
                         - (15 * np.power(yc_val, 2)# Tr(Yu^2)
                            * (np.power(yt_val, 2)
                               + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         - (6 * np.power(yc_val, 4))
                         - (2 * np.power(ys_val, 4))
                         - (4 * np.power(ys_val, 2) * np.power(yc_val, 2))
                         + (((16 * np.power(g3_val, 2))
                             + ((4 / 5) * np.power(g1_val, 2)))# Tr(Yu^2)
                            * (np.power(yt_val, 2)
                               + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         + (12 * np.power(g2_val, 2)
                            * np.power(yc_val, 2))
                         + ((2 / 5) * np.power(g1_val, 2)
                            * np.power(ys_val, 2))
                         - ((16 / 9) * np.power(g3_val, 4))
                         + (8 * np.power(g3_val, 2)
                            * np.power(g2_val, 2))
                         + ((136 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2))
                         + ((15 / 2) * np.power(g2_val, 4))
                         + (np.power(g2_val, 2) * np.power(g1_val, 2))
                         + ((2743 / 450) * np.power(g1_val, 4))))
                     + (yc_val# Tr(6au*Yu^3 + au*Yd^2*Yu + ad*Yu^2*Yd)
                        * (((-6) * ((6 * ((at_val * np.power(yt_val, 3))
                                          + (ac_val * np.power(yc_val, 3))
                                          + (au_val * np.power(yu_val, 3))))
                                    + (at_val * np.power(yb_val, 2) * yt_val)
                                    + (ac_val * np.power(ys_val, 2) * yc_val)
                                    + (au_val * np.power(yd_val, 2) * yu_val)
                                    + (ab_val * np.power(yt_val, 2) * yb_val)
                                    + (as_val * np.power(yc_val, 2) * ys_val)
                                    + (ad_val * np.power(yu_val, 2) * yd_val)))# end trace
                           - (18 * np.power(yc_val, 2)# Tr(au*Yu)
                              * ((at_val * yt_val)
                                 + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           - (np.power(ys_val, 2)# Tr(6ad*Yd + 2ae*Ye)
                              * ((6
                                  * ((ab_val * yb_val) + (as_val * ys_val)
                                     + (ad_val * yd_val)))
                                 + (2
                                    * ((atau_val * ytau_val)
                                       + (amu_val * ymu_val)
                                       + (ae_val * ye_val)))))# end trace
                           - (12 * yc_val * ac_val# Tr(Yu^2)
                              * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                 + np.power(yu_val, 2)))# end trace
                           - (ys_val * as_val# Tr(6Yd^2 + 2Ye^2)
                              * ((6 * (np.power(yb_val, 2)
                                       + np.power(ys_val, 2)
                                       + np.power(yd_val, 2)))
                                 + (2 * (np.power(ytau_val, 2)
                                         + np.power(ymu_val, 2)
                                         + np.power(ye_val, 2)))))# end trace
                           - (14 * np.power(yc_val, 3) * ac_val)
                           - (8 * np.power(ys_val, 3) * as_val)
                           - (2 * np.power(ys_val, 2) * yc_val * ac_val)
                           - (4 * ys_val * as_val * np.power(yc_val, 2))
                           + (((32 * np.power(g3_val, 2))
                               + ((8 / 5) * np.power(g1_val, 2)))# Tr(au*Yu)
                              * ((at_val * yt_val) + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           + (((6 * np.power(g2_val, 2))
                               + ((6 / 5) * np.power(g1_val, 2)))
                              * yc_val * ac_val)
                           + ((4 / 5) * np.power(g1_val, 2) * ys_val * as_val)
                           - (((32 * np.power(g3_val, 2) * M3_val)
                               + ((8 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yu^2)
                              * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                 + np.power(yu_val, 2)))# end trace
                           - (((12 * np.power(g2_val, 2) * M2_val)
                               + ((4 / 5) * np.power(g1_val, 2) * M1_val))
                              * np.power(yc_val, 2))
                           - ((4 / 5) * np.power(g1_val, 2) * M1_val
                              * np.power(ys_val, 2))
                           + ((64 / 9) * np.power(g3_val, 4) * M3_val)
                           - (16 * np.power(g3_val, 2) * np.power(g2_val, 2)
                              * (M3_val + M2_val))
                           - ((272 / 45) * np.power(g3_val, 2)
                              * np.power(g1_val, 2)
                              * (M3_val + M1_val))
                           - (30 * np.power(g2_val, 4) * M2_val)
                           - (2 * np.power(g2_val, 2) * np.power(g1_val, 2)
                              * (M2_val + M1_val))
                           - ((5486 / 225) * np.power(g1_val, 4) * M1_val))))

        dau_dt_2l = ((au_val# Tr(3Yu^4 + (Yu^2*Yd^2))
                      * (((-3) * ((3 * (np.power(yt_val, 4)
                                        + np.power(yc_val, 4)
                                        + np.power(yu_val, 4)))
                                  + ((np.power(yt_val, 2) * np.power(yb_val,2))
                                     + (np.power(yc_val, 2)
                                        * np.power(ys_val, 2))
                                     + (np.power(yu_val, 2)
                                        * np.power(yd_val, 2)))))# end trace
                         - (np.power(yd_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2)
                                     + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (np.power(ytau_val, 2)
                                  + np.power(ymu_val, 2)
                                  + np.power(ye_val, 2))))# end trace
                         - (15 * np.power(yu_val, 2)# Tr(Yu^2)
                            * (np.power(yt_val, 2)
                               + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         - (6 * np.power(yu_val, 4))
                         - (2 * np.power(yd_val, 4))
                         - (4 * np.power(yd_val, 2) * np.power(yu_val, 2))
                         + (((16 * np.power(g3_val, 2))
                             + ((4 / 5) * np.power(g1_val, 2)))# Tr(Yu^2)
                            * (np.power(yt_val, 2)
                               + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         + (12 * np.power(g2_val, 2)
                            * np.power(yu_val, 2))
                         + ((2 / 5) * np.power(g1_val, 2)
                            * np.power(yd_val, 2))
                         - ((16 / 9) * np.power(g3_val, 4))
                         + (8 * np.power(g3_val, 2)
                            * np.power(g2_val, 2))
                         + ((136 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2))
                         + ((15 / 2) * np.power(g2_val, 4))
                         + (np.power(g2_val, 2) * np.power(g1_val, 2))
                         + ((2743 / 450) * np.power(g1_val, 4))))
                     + (yu_val# Tr(6au*Yu^3 + au*Yd^2*Yu + ad*Yu^2*Yd)
                        * (((-6) * ((6 * ((at_val * np.power(yt_val, 3))
                                          + (ac_val * np.power(yc_val, 3))
                                          + (au_val * np.power(yu_val, 3))))
                                    + (at_val * np.power(yb_val, 2) * yt_val)
                                    + (ac_val * np.power(ys_val, 2) * yc_val)
                                    + (au_val * np.power(yd_val, 2) * yu_val)
                                    + (ab_val * np.power(yt_val, 2) * yb_val)
                                    + (as_val * np.power(yc_val, 2) * ys_val)
                                    + (ad_val * np.power(yu_val, 2) * yd_val)))# end trace
                           - (18 * np.power(yu_val, 2)# Tr(au*Yu)
                              * ((at_val * yt_val)
                                 + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           - (np.power(yd_val, 2)# Tr(6ad*Yd + 2ae*Ye)
                              * ((6
                                  * ((ab_val * yb_val) + (as_val * ys_val)
                                     + (ad_val * yd_val)))
                                 + (2
                                    * ((atau_val * ytau_val)
                                       + (amu_val * ymu_val)
                                       + (ae_val * ye_val)))))# end trace
                           - (12 * yu_val * au_val# Tr(Yu^2)
                              * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                 + np.power(yu_val, 2)))# end trace
                           - (yd_val * ad_val# Tr(6Yd^2 + 2Ye^2)
                              * ((6 * (np.power(yb_val, 2)
                                       + np.power(ys_val, 2)
                                       + np.power(yd_val, 2)))
                                 + (2 * (np.power(ytau_val, 2)
                                         + np.power(ymu_val, 2)
                                         + np.power(ye_val, 2)))))# end trace
                           - (14 * np.power(yu_val, 3) * au_val)
                           - (8 * np.power(yd_val, 3) * ad_val)
                           - (2 * np.power(yd_val, 2) * yu_val * au_val)
                           - (4 * yd_val * ad_val * np.power(yu_val, 2))
                           + (((32 * np.power(g3_val, 2))
                               + ((8 / 5) * np.power(g1_val, 2)))# Tr(au*Yu)
                              * ((at_val * yt_val) + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           + (((6 * np.power(g2_val, 2))
                               + ((6 / 5) * np.power(g1_val, 2)))
                              * yu_val * au_val)
                           + ((4 / 5) * np.power(g1_val, 2) * yd_val * ad_val)
                           - (((32 * np.power(g3_val, 2) * M3_val)
                               + ((8 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yu^2)
                              * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                 + np.power(yu_val, 2)))# end trace
                           - (((12 * np.power(g2_val, 2) * M2_val)
                               + ((4 / 5) * np.power(g1_val, 2) * M1_val))
                              * np.power(yu_val, 2))
                           - ((4 / 5) * np.power(g1_val, 2) * M1_val
                              * np.power(yd_val, 2))
                           + ((64 / 9) * np.power(g3_val, 4) * M3_val)
                           - (16 * np.power(g3_val, 2) * np.power(g2_val, 2)
                              * (M3_val + M2_val))
                           - ((272 / 45) * np.power(g3_val, 2)
                              * np.power(g1_val, 2)
                              * (M3_val + M1_val))
                           - (30 * np.power(g2_val, 4) * M2_val)
                           - (2 * np.power(g2_val, 2) * np.power(g1_val, 2)
                              * (M2_val + M1_val))
                           - ((5486 / 225) * np.power(g1_val, 4) * M1_val))))

        dab_dt_2l = ((ab_val# Tr(3Yd^4 + Yu^2*Yd^2 + Ye^4)
                      * (((-3) * ((3 * (np.power(yb_val, 4)
                                        + np.power(ys_val, 4)
                                        + np.power(yd_val, 4)))
                                  + ((np.power(yt_val, 2) * np.power(yb_val,2))
                                     + (np.power(yc_val, 2)
                                        * np.power(ys_val, 2))
                                     + (np.power(yu_val, 2)
                                        * np.power(yd_val, 2)))
                                  + np.power(ytau_val, 4)
                                  + np.power(ymu_val, 4)
                                  + np.power(ye_val, 4)))# end trace
                         - (3 * np.power(yt_val, 2)# Tr(Yu^2)
                            * (np.power(yt_val, 2) + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         - (5 * np.power(yb_val, 2)# Tr(3Yd^2+Ye^2)
                            * ((3 * (np.power(yb_val, 2)
                                     + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (np.power(ytau_val, 2)
                                  + np.power(ymu_val, 2)
                                  + np.power(ye_val, 2))))# end trace
                         - (6 * np.power(yb_val, 4))
                         - (2 * np.power(yt_val, 4))
                         - (4 * np.power(yb_val, 2) * np.power(yt_val, 2))
                         + (((16 * np.power(g3_val, 2))
                             - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                            * (np.power(yb_val, 2)
                               + np.power(ys_val, 2)
                               + np.power(yd_val, 2)))# end trace
                         + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                            * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         + ((4 / 5) * np.power(g1_val, 2)
                            * np.power(yt_val, 2))
                         + (((12 * np.power(g2_val, 2))
                             + ((6 / 5) * np.power(g1_val, 2)))
                            * np.power(yb_val, 2))
                         - ((16 / 9) * np.power(g3_val, 4))
                         + (8 * np.power(g3_val, 2)
                            * np.power(g2_val, 2))
                         + ((8 / 9) * np.power(g3_val, 2)
                            * np.power(g1_val, 2))
                         + ((15 / 2) * np.power(g2_val, 4))
                         + (np.power(g2_val, 2) * np.power(g1_val, 2))
                         + ((287 / 90) * np.power(g1_val, 4))))
                     + (yb_val# Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
                        * (((-6) * ((6 * ((ab_val * np.power(yb_val, 3))
                                          + (as_val * np.power(ys_val, 3))
                                          + (ad_val * np.power(yd_val, 3))))
                                    + (at_val * np.power(yb_val, 2) * yt_val)
                                    + (ac_val * np.power(ys_val, 2) * yc_val)
                                    + (au_val * np.power(yd_val, 2) * yu_val)
                                    + (ab_val * np.power(yt_val, 2) * yb_val)
                                    + (as_val * np.power(yc_val, 2) * ys_val)
                                    + (ad_val * np.power(yu_val, 2) * yd_val)
                                    + (2 * ((atau_val * np.power(ytau_val, 3))
                                            + (amu_val * np.power(ymu_val, 3))
                                            + (ae_val * np.power(ye_val, 3)))
                                       )))# end trace
                           - (6 * np.power(yt_val, 2)# Tr(au*Yu)
                              * ((at_val * yt_val)
                                 + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           - (6 * np.power(yb_val, 2)# Tr(3ad*Yd + ae*Ye)
                              * ((3
                                  * ((ab_val * yb_val) + (as_val * ys_val)
                                     + (ad_val * yd_val)))
                                 + ((atau_val * ytau_val) + (amu_val * ymu_val)
                                    + (ae_val * ye_val))))# end trace
                           - (6 * yt_val * at_val# Tr(Yu^2)
                              * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                 + np.power(yu_val, 2)))# end trace
                           - (4 * yb_val * ab_val# Tr(3Yd^2 + Ye^2)
                              * ((3 * (np.power(yb_val, 2)
                                       + np.power(ys_val, 2)
                                       + np.power(yd_val, 2)))
                                 + ((np.power(ytau_val, 2)
                                     + np.power(ymu_val, 2)
                                     + np.power(ye_val, 2)))))# end trace
                           - (14 * np.power(yb_val, 3) * ab_val)
                           - (8 * np.power(yt_val, 3) * at_val)
                           - (4 * np.power(yb_val, 2) * yt_val * at_val)
                           - (2 * yb_val * ab_val * np.power(yt_val, 2))
                           + (((32 * np.power(g3_val, 2))
                               - ((4 / 5) * np.power(g1_val, 2)))# Tr(ad*Yd)
                              * ((ab_val * yb_val) + (as_val * ys_val)
                                 + (ad_val * yd_val)))# end trace
                           + ((12 / 5) * np.power(g1_val, 2)# Tr(ae*Ye)
                              * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                 + (ae_val * ye_val)))# end trace
                           + ((8 / 5) * np.power(g1_val, 2) * yt_val * at_val)
                           + (((6 * np.power(g2_val, 2))
                               + ((6 / 5) * np.power(g1_val, 2)))
                              * yb_val * ab_val)
                           - (((32 * np.power(g3_val, 2) * M3_val)
                               - ((4 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yd^2)
                              * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                 + np.power(yd_val, 2)))# end trace
                           - ((12 / 5) * np.power(g1_val, 2) * M1_val# Tr(Ye^2)
                              * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2)))# end trace
                           - (((12 * np.power(g2_val, 2) * M2_val)
                               + ((8 / 5) * np.power(g1_val, 2) * M1_val))
                              * np.power(yb_val, 2))
                           - ((8 / 5) * np.power(g1_val, 2) * M1_val
                              * np.power(yt_val, 2))
                           + ((64 / 9) * np.power(g3_val, 4) * M3_val)
                           - (16 * np.power(g3_val, 2) * np.power(g2_val, 2)
                              * (M3_val + M2_val))
                           - ((16 / 9) * np.power(g3_val, 2)
                              * np.power(g1_val, 2)
                              * (M3_val + M1_val))
                           - (30 * np.power(g2_val, 4) * M2_val)
                           - (2 * np.power(g2_val, 2) * np.power(g1_val, 2)
                              * (M2_val + M1_val))
                           - ((574 / 45) * np.power(g1_val, 4) * M1_val))))

        das_dt_2l = ((as_val# Tr(3Yd^4 + Yu^2*Yd^2 + Ye^4)
                      * (((-3) * ((3 * (np.power(yb_val, 4)
                                        + np.power(ys_val, 4)
                                        + np.power(yd_val, 4)))
                                  + ((np.power(yt_val, 2) * np.power(yb_val,2))
                                     + (np.power(yc_val, 2)
                                        * np.power(ys_val, 2))
                                     + (np.power(yu_val, 2)
                                        * np.power(yd_val, 2)))
                                  + np.power(ytau_val, 4)
                                  + np.power(ymu_val, 4)
                                  + np.power(ye_val, 4)))# end trace
                         - (3 * np.power(yc_val, 2)# Tr(Yu^2)
                            * (np.power(yt_val, 2) + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         - (5 * np.power(ys_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2)
                                     + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (np.power(ytau_val, 2)
                                  + np.power(ymu_val, 2)
                                  + np.power(ye_val, 2))))# end trace
                         - (6 * np.power(ys_val, 4))
                         - (2 * np.power(yc_val, 4))
                         - (4 * np.power(ys_val, 2) * np.power(yc_val, 2))
                         + (((16 * np.power(g3_val, 2))
                             - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                            * (np.power(yb_val, 2)
                               + np.power(ys_val, 2)
                               + np.power(yd_val, 2)))# end trace
                         + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                            * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         + ((4 / 5) * np.power(g1_val, 2)
                            * np.power(yc_val, 2))
                         + (((12 * np.power(g2_val, 2))
                             + ((6 / 5) * np.power(g1_val, 2)))
                            * np.power(ys_val, 2))
                         - ((16 / 9) * np.power(g3_val, 4))
                         + (8 * np.power(g3_val, 2)
                            * np.power(g2_val, 2))
                         + ((8 / 9) * np.power(g3_val, 2)
                            * np.power(g1_val, 2))
                         + ((15 / 2) * np.power(g2_val, 4))
                         + (np.power(g2_val, 2) * np.power(g1_val, 2))
                         + ((287 / 90) * np.power(g1_val, 4))))
                     + (ys_val# Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
                        * (((-6) * ((6 * ((ab_val * np.power(yb_val, 3))
                                          + (as_val * np.power(ys_val, 3))
                                          + (ad_val * np.power(yd_val, 3))))
                                    + (at_val * np.power(yb_val, 2) * yt_val)
                                    + (ac_val * np.power(ys_val, 2) * yc_val)
                                    + (au_val * np.power(yd_val, 2) * yu_val)
                                    + (ab_val * np.power(yt_val, 2) * yb_val)
                                    + (as_val * np.power(yc_val, 2) * ys_val)
                                    + (ad_val * np.power(yu_val, 2) * yd_val)
                                    + (2 * ((atau_val * np.power(ytau_val, 3))
                                            + (amu_val * np.power(ymu_val, 3))
                                            + (ae_val * np.power(ye_val, 3)))
                                       )))# end trace
                           - (6 * np.power(yc_val, 2)# Tr(au*Yu)
                              * ((at_val * yt_val)
                                 + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           - (6 * np.power(ys_val, 2)# Tr(3ad*Yd + ae*Ye)
                              * ((3
                                  * ((ab_val * yb_val) + (as_val * ys_val)
                                     + (ad_val * yd_val)))
                                 + ((atau_val * ytau_val) + (amu_val * ymu_val)
                                    + (ae_val * ye_val))))# end trace
                           - (6 * yc_val * ac_val# Tr(Yu^2)
                              * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                 + np.power(yu_val, 2)))# end trace
                           - (4 * ys_val * as_val# Tr(3Yd^2 + Ye^2)
                              * ((3 * (np.power(yb_val, 2)
                                       + np.power(ys_val, 2)
                                       + np.power(yd_val, 2)))
                                 + ((np.power(ytau_val, 2)
                                     + np.power(ymu_val, 2)
                                     + np.power(ye_val, 2)))))# end trace
                           - (14 * np.power(ys_val, 3) * as_val)
                           - (8 * np.power(yc_val, 3) * ac_val)
                           - (4 * np.power(ys_val, 2) * yc_val * ac_val)
                           - (2 * ys_val * as_val * np.power(yc_val, 2))
                           + (((32 * np.power(g3_val, 2))
                               - ((4 / 5) * np.power(g1_val, 2)))# Tr(ad*Yd)
                              * ((ab_val * yb_val) + (as_val * ys_val)
                                 + (ad_val * yd_val)))# end trace
                           + ((12 / 5) * np.power(g1_val, 2)# Tr(ae*Ye)
                              * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                 + (ae_val * ye_val)))# end trace
                           + ((8 / 5) * np.power(g1_val, 2) * yc_val * ac_val)
                           + (((6 * np.power(g2_val, 2))
                               + ((6 / 5) * np.power(g1_val, 2)))
                              * ys_val * as_val)
                           - (((32 * np.power(g3_val, 2) * M3_val)
                               - ((4 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yd^2)
                              * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                 + np.power(yd_val, 2)))# end trace
                           - ((12 / 5) * np.power(g1_val, 2) * M1_val# Tr(Ye^2)
                              * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2)))# end trace
                           - (((12 * np.power(g2_val, 2) * M2_val)
                               + ((8 / 5) * np.power(g1_val, 2) * M1_val))
                              * np.power(ys_val, 2))
                           - ((8 / 5) * np.power(g1_val, 2) * M1_val
                              * np.power(yc_val, 2))
                           + ((64 / 9) * np.power(g3_val, 4) * M3_val)
                           - (16 * np.power(g3_val, 2) * np.power(g2_val, 2)
                              * (M3_val + M2_val))
                           - ((16 / 9) * np.power(g3_val, 2)
                              * np.power(g1_val, 2)
                              * (M3_val + M1_val))
                           - (30 * np.power(g2_val, 4) * M2_val)
                           - (2 * np.power(g2_val, 2) * np.power(g1_val, 2)
                              * (M2_val + M1_val))
                           - ((574 / 45) * np.power(g1_val, 4) * M1_val))))

        dad_dt_2l = ((ad_val# Tr(3Yd^4 + Yu^2*Yd^2 + Ye^4)
                      * (((-3) * ((3 * (np.power(yb_val, 4)
                                        + np.power(ys_val, 4)
                                        + np.power(yd_val, 4)))
                                  + ((np.power(yt_val, 2) * np.power(yb_val,2))
                                     + (np.power(yc_val, 2)
                                        * np.power(ys_val, 2))
                                     + (np.power(yu_val, 2)
                                        * np.power(yd_val, 2)))
                                  + np.power(ytau_val, 4)
                                  + np.power(ymu_val, 4)
                                  + np.power(ye_val, 4)))# end trace
                         - (3 * np.power(yu_val, 2)# Tr(Yu^2)
                            * (np.power(yt_val, 2) + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         - (5 * np.power(yd_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2)
                                     + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (np.power(ytau_val, 2)
                                  + np.power(ymu_val, 2)
                                  + np.power(ye_val, 2))))# end trace
                         - (6 * np.power(yd_val, 4))
                         - (2 * np.power(yu_val, 4))
                         - (4 * np.power(yd_val, 2) * np.power(yu_val, 2))
                         + (((16 * np.power(g3_val, 2))
                             - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                            * (np.power(yb_val, 2)
                               + np.power(ys_val, 2)
                               + np.power(yd_val, 2)))# end trace
                         + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                            * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         + ((4 / 5) * np.power(g1_val, 2)
                            * np.power(yu_val, 2))
                         + (((12 * np.power(g2_val, 2))
                             + ((6 / 5) * np.power(g1_val, 2)))
                            * np.power(yd_val, 2))
                         - ((16 / 9) * np.power(g3_val, 4))
                         + (8 * np.power(g3_val, 2)
                            * np.power(g2_val, 2))
                         + ((8 / 9) * np.power(g3_val, 2)
                            * np.power(g1_val, 2))
                         + ((15 / 2) * np.power(g2_val, 4))
                         + (np.power(g2_val, 2) * np.power(g1_val, 2))
                         + ((287 / 90) * np.power(g1_val, 4))))
                     + (yd_val# Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
                        * (((-6) * ((6 * ((ab_val * np.power(yb_val, 3))
                                          + (as_val * np.power(ys_val, 3))
                                          + (ad_val * np.power(yd_val, 3))))
                                    + (at_val * np.power(yb_val, 2) * yt_val)
                                    + (ac_val * np.power(ys_val, 2) * yc_val)
                                    + (au_val * np.power(yd_val, 2) * yu_val)
                                    + (ab_val * np.power(yt_val, 2) * yb_val)
                                    + (as_val * np.power(yc_val, 2) * ys_val)
                                    + (ad_val * np.power(yu_val, 2) * yd_val)
                                    + (2 * ((atau_val * np.power(ytau_val, 3))
                                            + (amu_val * np.power(ymu_val, 3))
                                            + (ae_val * np.power(ye_val, 3)))
                                       )))# end trace
                           - (6 * np.power(yu_val, 2)# Tr(au*Yu)
                              * ((at_val * yt_val)
                                 + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           - (6 * np.power(yd_val, 2)# Tr(3ad*Yd + ae*Ye)
                              * ((3
                                  * ((ab_val * yb_val) + (as_val * ys_val)
                                     + (ad_val * yd_val)))
                                 + ((atau_val * ytau_val) + (amu_val * ymu_val)
                                    + (ae_val * ye_val))))# end trace
                           - (6 * yu_val * au_val# Tr(Yu^2)
                              * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                 + np.power(yu_val, 2)))# end trace
                           - (4 * yd_val * ad_val# Tr(3Yd^2 + Ye^2)
                              * ((3 * (np.power(yb_val, 2)
                                       + np.power(ys_val, 2)
                                       + np.power(yd_val, 2)))
                                 + ((np.power(ytau_val, 2)
                                     + np.power(ymu_val, 2)
                                     + np.power(ye_val, 2)))))# end trace
                           - (14 * np.power(yd_val, 3) * ad_val)
                           - (8 * np.power(yu_val, 3) * au_val)
                           - (4 * np.power(yd_val, 2) * yu_val * au_val)
                           - (2 * yd_val * ad_val * np.power(yu_val, 2))
                           + (((32 * np.power(g3_val, 2))
                               - ((4 / 5) * np.power(g1_val, 2)))# Tr(ad*Yd)
                              * ((ab_val * yb_val) + (as_val * ys_val)
                                 + (ad_val * yd_val)))# end trace
                           + ((12 / 5) * np.power(g1_val, 2)# Tr(ae*Ye)
                              * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                 + (ae_val * ye_val)))# end trace
                           + ((8 / 5) * np.power(g1_val, 2) * yu_val * au_val)
                           + (((6 * np.power(g2_val, 2))
                               + ((6 / 5) * np.power(g1_val, 2)))
                              * yd_val * ad_val)
                           - (((32 * np.power(g3_val, 2) * M3_val)
                               - ((4 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yd^2)
                              * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                 + np.power(yd_val, 2)))# end trace
                           - ((12 / 5) * np.power(g1_val, 2) * M1_val# Tr(Ye^2)
                              * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2)))# end trace
                           - (((12 * np.power(g2_val, 2) * M2_val)
                               + ((8 / 5) * np.power(g1_val, 2) * M1_val))
                              * np.power(yd_val, 2))
                           - ((8 / 5) * np.power(g1_val, 2) * M1_val
                              * np.power(yu_val, 2))
                           + ((64 / 9) * np.power(g3_val, 4) * M3_val)
                           - (16 * np.power(g3_val, 2) * np.power(g2_val, 2)
                              * (M3_val + M2_val))
                           - ((16 / 9) * np.power(g3_val, 2)
                              * np.power(g1_val, 2)
                              * (M3_val + M1_val))
                           - (30 * np.power(g2_val, 4) * M2_val)
                           - (2 * np.power(g2_val, 2) * np.power(g1_val, 2)
                              * (M2_val + M1_val))
                           - ((574 / 45) * np.power(g1_val, 4) * M1_val))))

        datau_dt_2l = ((atau_val# Tr(3Yd^4 + Yu^2*Yd^2 + Ye^4)
                        * (((-3) * ((3 * (np.power(yb_val, 4)
                                          + np.power(ys_val, 4)
                                          + np.power(yd_val, 4)))
                                    + ((np.power(yt_val, 2)
                                        * np.power(yb_val,2))
                                       + (np.power(yc_val, 2)
                                          * np.power(ys_val, 2))
                                       + (np.power(yu_val, 2)
                                          * np.power(yd_val, 2)))
                                    + np.power(ytau_val, 4)
                                    + np.power(ymu_val, 4)
                                    + np.power(ye_val, 4)))# end trace
                           - (5 * np.power(ytau_val, 2)# Tr(3Yd^2 + Ye^2)
                              * ((3 * (np.power(yb_val, 2)
                                       + np.power(ys_val, 2)
                                       + np.power(yd_val, 2)))
                                 + (np.power(ytau_val, 2)
                                    + np.power(ymu_val, 2)
                                    + np.power(ye_val, 2))))# end trace
                           - (6 * np.power(ytau_val, 4))
                           + (((16 * np.power(g3_val, 2))
                               - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                              * (np.power(yb_val, 2)
                                 + np.power(ys_val, 2)
                                 + np.power(yd_val, 2)))# end trace
                           + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                              * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2)))# end trace
                           + (((12 * np.power(g2_val, 2))
                               - ((6 / 5) * np.power(g1_val, 2)))
                              * np.power(ytau_val, 2))
                           + ((15 / 2) * np.power(g2_val, 4))
                           + ((9 / 5) * np.power(g2_val, 2)
                              * np.power(g1_val, 2))
                           + ((27 / 2) * np.power(g1_val, 4))))
                       + (ytau_val# Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
                          * (((-6) * ((6 * ((ab_val * np.power(yb_val, 3))
                                            + (as_val * np.power(ys_val, 3))
                                            + (ad_val * np.power(yd_val, 3))))
                                      + (at_val * np.power(yb_val, 2) * yt_val)
                                      + (ac_val * np.power(ys_val, 2) * yc_val)
                                      + (au_val * np.power(yd_val, 2) * yu_val)
                                      + (ab_val * np.power(yt_val, 2) * yb_val)
                                      + (as_val * np.power(yc_val, 2) * ys_val)
                                      + (ad_val * np.power(yu_val, 2) * yd_val)
                                      + (2 * ((atau_val
                                               * np.power(ytau_val, 3))
                                              + (amu_val
                                                 * np.power(ymu_val, 3))
                                              + (ae_val
                                                 * np.power(ye_val, 3)))
                                         )))# end trace
                             - (4 * ytau_val * atau_val# Tr(3Yd^2 + Ye^2)
                                * ((3 * (np.power(yb_val, 2)
                                         + np.power(ys_val, 2)
                                         + np.power(yd_val, 2)))
                                   + ((np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                             - (6 * np.power(ytau_val, 2)# Tr(3ad*Yd + ae*Ye)
                                * ((3 * ((ab_val * yb_val) + (as_val * ys_val)
                                         + (ad_val * yd_val)))
                                   + (atau_val * ytau_val)
                                   + (amu_val * ymu_val)
                                   + (ae_val * ye_val)))# end trace
                             - (14 * np.power(ytau_val, 3) * atau_val)
                             + (((32 * np.power(g3_val, 2))
                                 - ((4 / 5) * np.power(g1_val, 2)))# Tr(ad*Yd)
                                * ((ab_val * yb_val) + (as_val * ys_val)
                                   + (ad_val * yd_val)))# end trace
                             + ((12 / 5) * np.power(g1_val, 2)# Tr(ae*Ye)
                                * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                   + (ae_val * ye_val)))# end trace
                             + (((6 * np.power(g2_val, 2))
                                 + ((6 / 5) * np.power(g1_val, 2)))
                                * ytau_val * atau_val)
                             - (((32 * np.power(g3_val, 2) * M3_val)
                                 - ((4 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yd^2)
                                * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                   + np.power(yd_val, 2)))# end trace
                             - ((12 / 5) * np.power(g1_val, 2) * M1_val# Tr(Ye^2)
                                * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                   + np.power(ye_val, 2)))# end trace
                             - (12 * np.power(g2_val, 2) * M2_val
                                * np.power(ytau_val, 2))
                             - (30 * np.power(g2_val, 4) * M2_val)
                             - ((18 / 5) * np.power(g2_val, 2)
                                * np.power(g1_val, 2)
                                * (M1_val + M2_val))
                             - (54 * np.power(g1_val, 4) * M1_val))))

        damu_dt_2l = ((amu_val# Tr(3Yd^4 + Yu^2*Yd^2 + Ye^4)
                       * (((-3) * ((3 * (np.power(yb_val, 4)
                                         + np.power(ys_val, 4)
                                         + np.power(yd_val, 4)))
                                   + ((np.power(yt_val, 2)
                                       * np.power(yb_val,2))
                                      + (np.power(yc_val, 2)
                                         * np.power(ys_val, 2))
                                      + (np.power(yu_val, 2)
                                         * np.power(yd_val, 2)))
                                   + np.power(ytau_val, 4)
                                   + np.power(ymu_val, 4)
                                   + np.power(ye_val, 4)))# end trace
                          - (5 * np.power(ymu_val, 2)# Tr(3Yd^2 + Ye^2)
                             * ((3 * (np.power(yb_val, 2)
                                      + np.power(ys_val, 2)
                                      + np.power(yd_val, 2)))
                                + (np.power(ytau_val, 2)
                                   + np.power(ymu_val, 2)
                                   + np.power(ye_val, 2))))# end trace
                          - (6 * np.power(ymu_val, 4))
                          + (((16 * np.power(g3_val, 2))
                              - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                             * (np.power(yb_val, 2)
                                + np.power(ys_val, 2)
                                + np.power(yd_val, 2)))# end trace
                          + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                             * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                + np.power(ye_val, 2)))# end trace
                          + (((12 * np.power(g2_val, 2))
                              - ((6 / 5) * np.power(g1_val, 2)))
                             * np.power(ymu_val, 2))
                          + ((15 / 2) * np.power(g2_val, 4))
                          + ((9 / 5) * np.power(g2_val, 2)
                             * np.power(g1_val, 2))
                          + ((27 / 2) * np.power(g1_val, 4))))
                      + (ymu_val# Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
                         * (((-6) * ((6 * ((ab_val * np.power(yb_val, 3))
                                           + (as_val * np.power(ys_val, 3))
                                           + (ad_val * np.power(yd_val, 3))))
                                     + (at_val * np.power(yb_val, 2) * yt_val)
                                     + (ac_val * np.power(ys_val, 2) * yc_val)
                                     + (au_val * np.power(yd_val, 2) * yu_val)
                                     + (ab_val * np.power(yt_val, 2) * yb_val)
                                     + (as_val * np.power(yc_val, 2) * ys_val)
                                     + (ad_val * np.power(yu_val, 2) * yd_val)
                                     + (2 * ((atau_val * np.power(ytau_val, 3))
                                             + (amu_val * np.power(ymu_val, 3))
                                             + (ae_val * np.power(ye_val, 3))))))# end trace
                            - (4 * ymu_val * amu_val# Tr(3Yd^2 + Ye^2)
                               * ((3 * (np.power(yb_val, 2)
                                        + np.power(ys_val, 2)
                                        + np.power(yd_val, 2)))
                                  + ((np.power(ytau_val, 2)
                                      + np.power(ymu_val, 2)
                                      + np.power(ye_val, 2)))))# end trace
                            - (6 * np.power(ymu_val, 2)# Tr(3ad*Yd + ae*Ye)
                               * ((3 * ((ab_val * yb_val) + (as_val * ys_val)
                                        + (ad_val * yd_val)))
                                  + (atau_val * ytau_val) + (amu_val * ymu_val)
                                  + (ae_val * ye_val)))# end trace
                            - (14 * np.power(ymu_val, 3) * amu_val)
                            + (((32 * np.power(g3_val, 2))
                                - ((4 / 5) * np.power(g1_val, 2)))# Tr(ad*Yd)
                               * ((ab_val * yb_val) + (as_val * ys_val)
                                  + (ad_val * yd_val)))# end trace
                            + ((12 / 5) * np.power(g1_val, 2)# Tr(ae*Ye)
                               * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                  + (ae_val * ye_val)))# end trace
                            + (((6 * np.power(g2_val, 2))
                                + ((6 / 5) * np.power(g1_val, 2)))
                               * ymu_val * amu_val)
                            - (((32 * np.power(g3_val, 2) * M3_val)
                                - ((4 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yd^2)
                               * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                  + np.power(yd_val, 2)))# end trace
                            - ((12 / 5) * np.power(g1_val, 2) * M1_val# Tr(Ye^2)
                               * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                  + np.power(ye_val, 2)))# end trace
                            - (12 * np.power(g2_val, 2) * M2_val
                               * np.power(ymu_val, 2))
                            - (30 * np.power(g2_val, 4) * M2_val)
                            - ((18 / 5) * np.power(g2_val, 2) * np.power(g1_val, 2)
                               * (M1_val + M2_val))
                            - (54 * np.power(g1_val, 4) * M1_val))))

        dae_dt_2l = ((ae_val# Tr(3Yd^4 + Yu^2*Yd^2 + Ye^4)
                      * (((-3) * ((3 * (np.power(yb_val, 4)
                                        + np.power(ys_val, 4)
                                        + np.power(yd_val, 4)))
                                  + ((np.power(yt_val, 2)
                                      * np.power(yb_val,2))
                                     + (np.power(yc_val, 2)
                                        * np.power(ys_val, 2))
                                     + (np.power(yu_val, 2)
                                        * np.power(yd_val, 2)))
                                  + np.power(ytau_val, 4)
                                  + np.power(ymu_val, 4)
                                  + np.power(ye_val, 4)))# end trace
                         - (5 * np.power(ye_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2)
                                     + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (np.power(ytau_val, 2)
                                  + np.power(ymu_val, 2)
                                  + np.power(ye_val, 2))))# end trace
                         - (6 * np.power(ye_val, 4))
                         + (((16 * np.power(g3_val, 2))
                             - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                            * (np.power(yb_val, 2)
                               + np.power(ys_val, 2)
                               + np.power(yd_val, 2)))# end trace
                         + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                            * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         + (((12 * np.power(g2_val, 2))
                             - ((6 / 5) * np.power(g1_val, 2)))
                            * np.power(ye_val, 2))
                         + ((15 / 2) * np.power(g2_val, 4))
                         + ((9 / 5) * np.power(g2_val, 2)
                            * np.power(g1_val, 2))
                         + ((27 / 2) * np.power(g1_val, 4))))
                     + (ye_val# Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
                        * (((-6) * ((6 * ((ab_val * np.power(yb_val, 3))
                                          + (as_val * np.power(ys_val, 3))
                                          + (ad_val * np.power(yd_val, 3))))
                                    + (at_val * np.power(yb_val, 2) * yt_val)
                                    + (ac_val * np.power(ys_val, 2) * yc_val)
                                    + (au_val * np.power(yd_val, 2) * yu_val)
                                    + (ab_val * np.power(yt_val, 2) * yb_val)
                                    + (as_val * np.power(yc_val, 2) * ys_val)
                                    + (ad_val * np.power(yu_val, 2) * yd_val)
                                    + (2 * ((atau_val * np.power(ytau_val, 3))
                                            + (amu_val * np.power(ymu_val, 3))
                                            + (ae_val * np.power(ye_val, 3))))))# end trace
                           - (4 * ye_val * ae_val# Tr(3Yd^2 + Ye^2)
                              * ((3 * (np.power(yb_val, 2)
                                       + np.power(ys_val, 2)
                                       + np.power(yd_val, 2)))
                                 + ((np.power(ytau_val, 2)
                                     + np.power(ymu_val, 2)
                                     + np.power(ye_val, 2)))))# end trace
                           - (6 * np.power(ye_val, 2)# Tr(3ad*Yd + ae*Ye)
                              * ((3 * ((ab_val * yb_val) + (as_val * ys_val)
                                       + (ad_val * yd_val)))
                                 + (atau_val * ytau_val) + (amu_val * ymu_val)
                                 + (ae_val * ye_val)))# end trace
                           - (14 * np.power(ye_val, 3) * ae_val)
                           + (((32 * np.power(g3_val, 2))
                               - ((4 / 5) * np.power(g1_val, 2)))# Tr(ad*Yd)
                              * ((ab_val * yb_val) + (as_val * ys_val)
                                 + (ad_val * yd_val)))# end trace
                           + ((12 / 5) * np.power(g1_val, 2)# Tr(ae*Ye)
                              * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                 + (ae_val * ye_val)))# end trace
                           + (((6 * np.power(g2_val, 2))
                               + ((6 / 5) * np.power(g1_val, 2)))
                              * ye_val * ae_val)
                           - (((32 * np.power(g3_val, 2) * M3_val)
                               - ((4 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yd^2)
                              * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                 + np.power(yd_val, 2)))# end trace
                           - ((12 / 5) * np.power(g1_val, 2) * M1_val# Tr(Ye^2)
                              * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2)))# end trace
                           - (12 * np.power(g2_val, 2) * M2_val
                              * np.power(ye_val, 2))
                           - (30 * np.power(g2_val, 4) * M2_val)
                           - ((18 / 5) * np.power(g2_val, 2) * np.power(g1_val, 2)
                              * (M1_val + M2_val))
                           - (54 * np.power(g1_val, 4) * M1_val))))

        # Total soft trilinear coupling beta functions
        dat_dt = (1 / t) * ((loop_fac * dat_dt_1l)
                            + (loop_fac_sq * dat_dt_2l))

        dac_dt = (1 / t) * ((loop_fac * dac_dt_1l)
                            + (loop_fac_sq * dac_dt_2l))

        dau_dt = (1 / t) * ((loop_fac * dau_dt_1l)
                            + (loop_fac_sq * dau_dt_2l))

        dab_dt = (1 / t) * ((loop_fac * dab_dt_1l)
                            + (loop_fac_sq * dab_dt_2l))

        das_dt = (1 / t) * ((loop_fac * das_dt_1l)
                            + (loop_fac_sq * das_dt_2l))

        dad_dt = (1 / t) * ((loop_fac * dad_dt_1l)
                            + (loop_fac_sq * dad_dt_2l))

        datau_dt = (1 / t) * ((loop_fac * datau_dt_1l)
                            + (loop_fac_sq * datau_dt_2l))

        damu_dt = (1 / t) * ((loop_fac * damu_dt_1l)
                            + (loop_fac_sq * damu_dt_2l))

        dae_dt = (1 / t) * ((loop_fac * dae_dt_1l)
                            + (loop_fac_sq * dae_dt_2l))

        ##### Soft bilinear coupling b=B*mu#####
        # 1 loop part
        db_dt_1l = ((b_val# Tr(3Yu^2 + 3Yd^2 + Ye^2)
                     * (((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                               + np.power(yu_val, 2) + np.power(yb_val, 2)
                               + np.power(ys_val, 2) + np.power(yd_val, 2)))
                         + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                         + np.power(ye_val, 2))# end trace
                        - (3 * np.power(g2_val, 2))
                        - ((3 / 5) * np.power(g1_val, 2))))
                    + (mu_val# Tr(6au*Yu + 6ad*Yd + 2ae*Ye)
                       * (((6 * ((at_val * yt_val) + (ac_val * yc_val)
                                 + (au_val * yu_val) + (ab_val * yb_val)
                                 + (as_val * ys_val) + (ad_val * yd_val)))
                           + (2 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                   + (ae_val * ye_val))))
                          + (6 * np.power(g2_val, 2) * M2_val)
                          + ((6 / 5) * np.power(g1_val, 2) * M1_val))))

        # 2 loop part
        db_dt_2l = ((b_val# Tr(3Yu^4 + 3Yd^4 + 2Yu^2*Yd^2 + Ye^4)
                     * (((-3) * ((3 * (np.power(yt_val, 4) + np.power(yc_val, 4)
                                       + np.power(yu_val, 4)
                                       + np.power(yb_val, 4)
                                       + np.power(ys_val, 4)
                                       + np.power(yd_val, 4)))
                                 + (2 * ((np.power(yt_val, 2)
                                          * np.power(yb_val, 2))
                                         + (np.power(yc_val, 2)
                                            * np.power(ys_val, 2))
                                         + (np.power(yu_val, 2)
                                            * np.power(yd_val, 2))))
                                 + np.power(ytau_val, 4) + np.power(ymu_val, 4)
                                 + np.power(ye_val, 4)))# end trace
                        + (((16 * np.power(g3_val, 2))
                            + ((4 / 5) * np.power(g1_val, 2)))# Tr(Yu^2)
                           * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        + (((16 * np.power(g3_val, 2))
                            - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                           * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))# end trace
                        + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                           * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))# end trace
                        + ((15 / 2) * np.power(g2_val, 4))
                        + ((9 / 5) * np.power(g1_val, 2) * np.power(g2_val, 2))
                        + ((207 / 50) * np.power(g1_val, 4))))
                    + (mu_val * (((-12)# Tr(3au*Yu^3 + 3ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + ae*Ye^3)
                          * ((3 * ((at_val * np.power(yt_val, 3))
                                   + (ac_val * np.power(yc_val, 3))
                                   + (au_val * np.power(yu_val, 3))
                                   + (ab_val * np.power(yb_val, 3))
                                   + (as_val * np.power(ys_val, 3))
                                   + (ad_val * np.power(yd_val, 3))))
                             + ((at_val * np.power(yb_val, 2) * yt_val)
                                + (ac_val * np.power(ys_val, 2) * yc_val)
                                + (au_val * np.power(yd_val, 2) * yu_val))
                             + ((ab_val * np.power(yt_val, 2) * yb_val)
                                + (as_val * np.power(yc_val, 2) * ys_val)
                                + (ad_val * np.power(yu_val, 2) * yd_val))
                             + ((atau_val * np.power(ytau_val, 3))
                                + (amu_val * np.power(ymu_val, 3))
                                + (ae_val * np.power(ye_val, 3)))))# end trace
                        + (((32 * np.power(g3_val, 2))
                             + ((8 / 5) * np.power(g1_val, 2)))# Tr(au*Yu)
                            * ((at_val * yt_val) + (ac_val * yc_val)
                               + (au_val * yu_val)))# end trace
                         + (((32 * np.power(g3_val, 2))
                             - ((4 / 5) * np.power(g1_val, 2)))# Tr(ad*Yd)
                            * ((ab_val * yb_val) + (as_val * ys_val)
                               + (ad_val * yd_val)))# end trace
                         + ((12 / 5) * np.power(g1_val, 2)# Tr(ae*Ye)
                            * ((atau_val * ytau_val) + (amu_val * ymu_val)
                               + (ae_val * ye_val)))# end trace
                         - (((32 * np.power(g3_val, 2) * M3_val)
                             + ((8 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yu^2)
                            * (np.power(yt_val, 2) + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))
                         - (((32 * np.power(g3_val, 2) * M3_val)# end trace
                             - ((4 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yd^2)
                            * (np.power(yb_val, 2) + np.power(ys_val, 2)
                               + np.power(yd_val, 2)))# end trace
                         - ((12 / 5) * np.power(g1_val, 2) * M1_val# Tr(Ye^2)
                            * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (30 * np.power(g2_val, 4) * M2_val)
                         - ((18 / 5) * np.power(g1_val, 2)
                            * np.power(g2_val, 2)
                            * (M1_val + M2_val))
                         - ((414 / 25) * np.power(g1_val, 4) * M1_val))))

        # Total b beta function
        db_dt = (1 / t) * ((loop_fac * db_dt_1l)
                           + (loop_fac_sq * db_dt_2l))

        ##### Scalar squared masses #####
        # Introduce S, S', and sigma terms
        S_val = (mHu_sq_val - mHd_sq_val + mQ3_sq_val + mQ2_sq_val + mQ1_sq_val
                 - mL3_sq_val - mL2_sq_val - mL1_sq_val
                 - (2 * (mU3_sq_val + mU2_sq_val + mU1_sq_val))
                 + mD3_sq_val + mD2_sq_val + mD1_sq_val
                 + mE3_sq_val + mE2_sq_val + mE1_sq_val)

        # Tr(-(3mHu^2 + mQ^2) * Yu^2 + 4Yu^2 * mU^2 + (3mHd^2 - mQ^2) * Yd^2
        #    - 2Yd^2 * mD^2 + (mHd^2 + mL^2) * Ye^2 - 2Ye^2 * mE^2)
        Spr_val = ((((-1) * ((((3 * mHu_sq_val) + mQ3_sq_val)
                              * np.power(yt_val, 2))
                             + (((3 * mHu_sq_val) + mQ2_sq_val)
                                * np.power(yc_val, 2))
                             + (((3 * mHu_sq_val) + mQ1_sq_val)
                                * np.power(yu_val, 2))))
                    + (4 * np.power(yt_val, 2) * mU3_sq_val)
                    + (4 * np.power(yc_val, 2) * mU2_sq_val)
                    + (4 * np.power(yu_val, 2) * mU1_sq_val)
                    + ((((3 * mHd_sq_val) - mQ3_sq_val) * np.power(yb_val, 2))
                       + (((3 * mHd_sq_val) - mQ2_sq_val)
                          * np.power(ys_val, 2))
                       + (((3 * mHd_sq_val) - mQ1_sq_val)
                          * np.power(yd_val, 2)))
                    - (2 * ((mD3_sq_val * np.power(yb_val, 2))
                            + (mD2_sq_val * np.power(ys_val, 2))
                            + (mD1_sq_val * np.power(yd_val, 2))))
                    + (((mHd_sq_val + mL3_sq_val) * np.power(ytau_val, 2))
                       + ((mHd_sq_val + mL2_sq_val) * np.power(ymu_val, 2))
                       + ((mHd_sq_val + mL1_sq_val) * np.power(ye_val, 2)))
                    - (2 * ((np.power(ytau_val, 2) * mE3_sq_val)
                            + (np.power(ymu_val, 2) * mE2_sq_val)
                            + (np.power(ye_val, 2) * mE1_sq_val))))# end trace
                   + ((((3 / 2) * np.power(g2_val, 2))
                       + ((3 / 10) * np.power(g1_val, 2)))
                      * (mHu_sq_val - mHd_sq_val# Tr(mL^2)
                         - (mL3_sq_val + mL2_sq_val + mL1_sq_val)))# end trace
                   + ((((8 / 3) * np.power(g3_val, 2))
                       + ((3 / 2) * np.power(g2_val, 2))
                       + ((1 / 30) * np.power(g1_val, 2)))# Tr(mQ^2)
                      * (mQ3_sq_val + mQ2_sq_val + mQ1_sq_val))# end trace
                   - ((((16 / 3) * np.power(g3_val, 2))
                       + ((16 / 15) * np.power(g1_val, 2)))# Tr (mU^2)
                      * (mU3_sq_val + mU2_sq_val + mU1_sq_val))# end trace
                   + ((((8 / 3) * np.power(g3_val, 2))
                      + ((2 / 15) * np.power(g1_val, 2)))# Tr(mD^2)
                      * (mD3_sq_val + mD2_sq_val + mD1_sq_val))# end trace
                   + ((6 / 5) * np.power(g1_val, 2)# Tr(mE^2)
                      * (mE3_sq_val + mE2_sq_val + mE1_sq_val)))# end trace

        sigma1 = ((1 / 5) * np.power(g1_val, 2)
                  * ((3 * (mHu_sq_val + mHd_sq_val))# Tr(mQ^2 + 3mL^2 + 8mU^2 + 2mD^2 + 6mE^2)
                     + mQ3_sq_val + mQ2_sq_val + mQ1_sq_val
                     + (3 * (mL3_sq_val + mL2_sq_val + mL1_sq_val))
                     + (8 * (mU3_sq_val + mU2_sq_val + mU1_sq_val))
                     + (2 * (mD3_sq_val + mD2_sq_val + mD1_sq_val))
                     + (6 * (mE3_sq_val + mE2_sq_val + mE1_sq_val))))# end trace

        sigma2 = (np.power(g2_val, 2)
                  * (mHu_sq_val + mHd_sq_val# Tr(3mQ^2 + mL^2)
                     + (3 * (mQ3_sq_val + mQ2_sq_val + mQ1_sq_val))
                     + mL3_sq_val + mL2_sq_val + mL1_sq_val))# end trace

        sigma3 = (np.power(g3_val, 2)# Tr(2mQ^2 + mU^2 + mD^2)
                  * ((2 * (mQ3_sq_val + mQ2_sq_val + mQ1_sq_val))
                     + mU3_sq_val + mU2_sq_val + mU1_sq_val
                     + mD3_sq_val + mD2_sq_val + mD1_sq_val))# end trace

        # 1 loop part of Higgs squared masses
        dmHu_sq_dt_1l = ((6# Tr((mHu^2 + mQ^2) * Yu^2 + Yu^2 * mU^2 + au^2)
                          * (((mHu_sq_val + mQ3_sq_val) * np.power(yt_val, 2))
                             + ((mHu_sq_val + mQ2_sq_val)
                                * np.power(yc_val, 2))
                             + ((mHu_sq_val + mQ1_sq_val)
                                * np.power(yu_val, 2))
                             + (mU3_sq_val * np.power(yt_val, 2))
                             + (mU2_sq_val * np.power(yc_val, 2))
                             + (mU1_sq_val * np.power(yu_val, 2))
                             + np.power(at_val, 2) + np.power(ac_val, 2)
                             + np.power(au_val, 2)))# end trace
                         - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))
                         - ((6 / 5) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((3 / 5) * np.power(g1_val, 2) * S_val))

        # Tr (6(mHd^2 + mQ^2) * Yd^2 + 6Yd^2*mD^2 + 2(mHd^2 + mL^2) * Ye^2
        #     + 2(Ye^2 * mE^2) + 6ad^2 + 2ae^2)
        dmHd_sq_dt_1l = ((6 * (((mHd_sq_val + mQ3_sq_val)
                                * np.power(yb_val, 2))
                               + ((mHd_sq_val + mQ2_sq_val)
                                  * np.power(ys_val, 2))
                               + ((mHd_sq_val + mQ1_sq_val)
                                  * np.power(yd_val, 2)))
                          + (6 * ((mD3_sq_val * np.power(yb_val, 2))
                                  + (mD2_sq_val * np.power(ys_val, 2))
                                  + (mD1_sq_val * np.power(yd_val, 2))))
                          + (2 * (((mHd_sq_val + mL3_sq_val)
                                   * np.power(ytau_val, 2))
                                  + ((mHd_sq_val + mL2_sq_val)
                                     * np.power(ymu_val, 2))
                                  + ((mHd_sq_val + mL1_sq_val)
                                     * np.power(ye_val, 2))))
                          + (2 * ((mE3_sq_val * np.power(ytau_val, 2))
                                  + (mE2_sq_val * np.power(ymu_val, 2))
                                  + (mE1_sq_val * np.power(ye_val, 2))))
                          + (6 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                  + np.power(ad_val, 2)))
                          + (2 * (np.power(atau_val, 2) + np.power(amu_val, 2)
                                  + np.power(ae_val, 2))))# end trace
                         - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))
                         - ((6 / 5) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         - ((3 / 5) * np.power(g1_val, 2) * S_val))

        # 2 loop part of Higgs squared masses
        dmHu_sq_dt_2l = (((-6) # Tr(6(mHu^2 + mQ^2)*Yu^4 + 6Yu^4 * mU^2 + (mHu^2 + mHd^2 + mQ^2) * Yu^2 * Yd^2 + Yu^2 * Yd^2 * mU^2 + Yu^2 * Yd^2 * mQ^2 + Yu^2 * Yd^2 * mD^2 + 12au^2 * Yu^2 + ad^2 * Yu^2 + Yd^2 * au^2 + 2ad * Yd * Yu * au)
                          * ((6 * (((mHu_sq_val + mQ3_sq_val)
                                    * np.power(yt_val, 4))
                                   + ((mHu_sq_val + mQ2_sq_val)
                                      * np.power(yc_val, 4))
                                   + ((mHu_sq_val + mQ1_sq_val)
                                      * np.power(yu_val, 4))))
                             + (6 * ((mU3_sq_val * np.power(yt_val, 4))
                                     + (mU2_sq_val * np.power(yc_val, 4))
                                     + (mU1_sq_val * np.power(yu_val, 4))))
                             + ((mHu_sq_val + mHd_sq_val + mQ3_sq_val)
                                * np.power(yt_val, 2) * np.power(yb_val, 2))
                             + ((mHu_sq_val + mHd_sq_val + mQ2_sq_val)
                                * np.power(yc_val, 2) * np.power(ys_val, 2))
                             + ((mHu_sq_val + mHd_sq_val + mQ1_sq_val)
                                * np.power(yu_val, 2) * np.power(yd_val, 2))
                             + ((mU3_sq_val + mQ3_sq_val + mD3_sq_val)
                                * np.power(yt_val, 2) * np.power(yb_val, 2))
                             + ((mU2_sq_val + mQ2_sq_val + mD2_sq_val)
                                * np.power(yc_val, 2) * np.power(ys_val, 2))
                             + ((mU1_sq_val + mQ1_sq_val + mD1_sq_val)
                                * np.power(yu_val, 2) * np.power(yd_val, 2))
                             + (12 * ((np.power(at_val, 2)
                                       * np.power(yt_val, 2))
                                      + (np.power(ac_val, 2)
                                         * np.power(yc_val, 2))
                                      + (np.power(au_val, 2)
                                         * np.power(yu_val, 2))))
                             + (np.power(ab_val, 2) * np.power(yt_val, 2))
                             + (np.power(as_val, 2) * np.power(yc_val, 2))
                             + (np.power(ad_val, 2) * np.power(yu_val, 2))
                             + (np.power(yb_val, 2) * np.power(at_val, 2))
                             + (np.power(ys_val, 2) * np.power(ac_val, 2))
                             + (np.power(yd_val, 2) * np.power(au_val, 2))
                             + (2 * ((yb_val * ab_val * at_val * yt_val)
                                     + (ys_val * as_val * ac_val * yc_val)
                                     + (yd_val * ad_val * au_val * yu_val)))))# end trace
                         + (((32 * np.power(g3_val, 2))
                             + ((8 / 5) * np.power(g1_val, 2))) # Tr((mHu^2 + mQ^2 + mU^2) * Yu^2 + au^2)
                            * (((mHu_sq_val + mQ3_sq_val + mU3_sq_val)
                                * np.power(yt_val, 2))
                               + ((mHu_sq_val + mQ2_sq_val + mU2_sq_val)
                                  * np.power(yc_val, 2))
                               + ((mHu_sq_val + mQ1_sq_val + mU1_sq_val)
                                  * np.power(yu_val, 2))
                               + np.power(at_val, 2) + np.power(ac_val, 2)
                               + np.power(au_val, 2)))# end trace
                         + (32 * np.power(g3_val, 2)
                            * ((2 * np.power(M3_val, 2)# Tr(Yu^2)
                                * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                               - (2 * M3_val# Tr(Yu*au)
                                  * ((yt_val * at_val) + (yc_val * ac_val)
                                     + (yu_val * au_val)))))# end trace
                         + ((8 / 5) * np.power(g1_val, 2)
                            * ((2 * np.power(M1_val, 2)# Tr(Yu^2)
                                * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                               - (2 * M1_val# Tr(Yu*au)
                                  * ((yt_val * at_val) + (yc_val * ac_val)
                                     + (yu_val * au_val)))))# end trace
                         + ((6 / 5) * np.power(g1_val, 2) * Spr_val)
                         + (33 * np.power(g2_val, 4) * np.power(M2_val, 2))
                         + ((18 / 5) * np.power(g2_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M2_val, 2) + np.power(M1_val, 2)
                               + (M1_val * M2_val)))
                         + ((621 / 25) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + (3 * np.power(g2_val, 2) * sigma2)
                         + ((3 / 5) * np.power(g1_val, 2) * sigma1))

        dmHd_sq_dt_2l = (((-6) # Tr(6(mHd^2 + mQ^2)*Yd^4 + 6Yd^4 * mD^2 + (mHu^2 + mHd^2 + mQ^2) * Yu^2 * Yd^2 + Yu^2 * Yd^2 * mU^2 + Yu^2 * Yd^2 * mQ^2 + Yu^2 * Yd^2 * mD^2 + 2(mHd^2 + mL^2) * Ye^4 + 2Ye^4 * mE^2 + 12ad^2 * Yd^2 + ad^2 * Yu^2 + Yd^2 * au^2 + 2ad * Yd * Yu * au + 4ae^2 * Ye^2)
                          * ((6 * (((mHd_sq_val + mQ3_sq_val)
                                    * np.power(yb_val, 4))
                                   + ((mHd_sq_val + mQ2_sq_val)
                                      * np.power(ys_val, 4))
                                   + ((mHd_sq_val + mQ1_sq_val)
                                      * np.power(yd_val, 4))))
                             + (6 * ((mD3_sq_val * np.power(yb_val, 4))
                                     + (mD2_sq_val * np.power(ys_val, 4))
                                     + (mD1_sq_val * np.power(yd_val, 4))))
                             + ((mHu_sq_val + mHd_sq_val + mQ3_sq_val)
                                * np.power(yt_val, 2) * np.power(yb_val, 2))
                             + ((mHu_sq_val + mHd_sq_val + mQ2_sq_val)
                                * np.power(yc_val, 2) * np.power(ys_val, 2))
                             + ((mHu_sq_val + mHd_sq_val + mQ1_sq_val)
                                * np.power(yu_val, 2) * np.power(yd_val, 2))
                             + ((mU3_sq_val + mQ3_sq_val + mD3_sq_val)
                                * np.power(yt_val, 2) * np.power(yb_val, 2))
                             + ((mU2_sq_val + mQ2_sq_val + mD2_sq_val)
                                * np.power(yc_val, 2) * np.power(ys_val, 2))
                             + ((mU1_sq_val + mQ1_sq_val + mD1_sq_val)
                                * np.power(yu_val, 2) * np.power(yd_val, 2))
                             + (2 * (((mHd_sq_val + mL3_sq_val + mE3_sq_val)
                                      * np.power(ytau_val, 4))
                                     + ((mHd_sq_val + mL2_sq_val + mE2_sq_val)
                                        * np.power(ymu_val, 4))
                                     + ((mHd_sq_val + mL1_sq_val + mE1_sq_val)
                                        * np.power(ye_val, 4))))
                             + (12 * ((np.power(ab_val, 2)
                                       * np.power(yb_val, 2))
                                      + (np.power(as_val, 2)
                                         * np.power(ys_val, 2))
                                      + (np.power(ad_val, 2)
                                         * np.power(yd_val, 2))))
                             + (np.power(ab_val, 2) * np.power(yt_val, 2))
                             + (np.power(as_val, 2) * np.power(yc_val, 2))
                             + (np.power(ad_val, 2) * np.power(yu_val, 2))
                             + (np.power(yb_val, 2) * np.power(at_val, 2))
                             + (np.power(ys_val, 2) * np.power(ac_val, 2))
                             + (np.power(yd_val, 2) * np.power(au_val, 2))
                             + (2 * ((yb_val * ab_val * at_val * yt_val)
                                     + (ys_val * as_val * ac_val * yc_val)
                                     + (yd_val * ad_val * au_val * yu_val)
                                     + (2 * ((np.power(atau_val, 2)
                                              * np.power(ytau_val, 2))
                                             + (np.power(amu_val, 2)
                                                * np.power(ymu_val, 2))
                                             + (np.power(ae_val, 2)
                                                * np.power(ye_val, 2))))))))# end trace
                         + (((32 * np.power(g3_val, 2))
                             - ((4 / 5) * np.power(g1_val, 2))) # Tr((mHd^2 + mQ^2 + mD^2) * Yd^2 + ad^2)
                            * (((mHu_sq_val + mQ3_sq_val + mD3_sq_val)
                                * np.power(yb_val, 2))
                               + ((mHu_sq_val + mQ2_sq_val + mD2_sq_val)
                                  * np.power(ys_val, 2))
                               + ((mHu_sq_val + mQ1_sq_val + mD1_sq_val)
                                  * np.power(yd_val, 2))
                               + np.power(ab_val, 2) + np.power(as_val, 2)
                               + np.power(ad_val, 2)))# end trace
                         + (32 * np.power(g3_val, 2)
                            * ((2 * np.power(M3_val, 2)# Tr(Yd^2)
                                * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                   + np.power(yd_val, 2)))# end trace
                               - (2 * M3_val # Tr(Yd*ad)
                                  * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))))# end trace
                         - ((4 / 5) * np.power(g1_val, 2)
                            * ((2 * np.power(M1_val, 2)# Tr(Yd^2)
                                * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                   + np.power(yd_val, 2)))# end trace
                               - (2 * M1_val # Tr(Yd*ad)
                                  * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))))# end trace
                         + ((12 / 5) * np.power(g1_val, 2)
                            * (# Tr((mHd^2 + mL^2 + mE^2) * Ye^2 + ae^2)
                               ((mHd_sq_val + mL3_sq_val + mE3_sq_val)
                                * np.power(ytau_val, 2))
                               + ((mHd_sq_val + mL2_sq_val + mE2_sq_val)
                                  * np.power(ymu_val, 2))
                               + ((mHd_sq_val + mL1_sq_val + mE1_sq_val)
                                  * np.power(ye_val, 2))
                               + np.power(atau_val, 2) + np.power(amu_val, 2)
                               + np.power(ae_val, 2)))# end trace
                         - ((6 / 5) * np.power(g1_val, 2) * Spr_val)
                         + (33 * np.power(g2_val, 4) * np.power(M2_val, 2))
                         + ((18 / 5) * np.power(g2_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M2_val, 2) + np.power(M1_val, 2)
                               + (M1_val * M2_val)))
                         + ((621 / 25) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + (3 * np.power(g2_val, 2) * sigma2)
                         + ((3 / 5) * np.power(g1_val, 2) * sigma1))

        # Total Higgs squared mass beta functions
        dmHu_sq_dt = (1 / t) * ((loop_fac * dmHu_sq_dt_1l)
                                + (loop_fac_sq * dmHu_sq_dt_2l))

        dmHd_sq_dt = (1 / t) * ((loop_fac * dmHd_sq_dt_1l)
                                + (loop_fac_sq * dmHd_sq_dt_2l))

        # 1 loop parts of scalar squared masses
        # Left squarks
        dmQ3_sq_dt_1l = (((mQ3_sq_val + (2 * mHu_sq_val))
                          * np.power(yt_val, 2))
                         + ((mQ3_sq_val + (2 * mHd_sq_val))
                            * np.power(yb_val, 2))
                         + ((np.power(yt_val, 2) + np.power(yb_val, 2))
                            * mQ3_sq_val)
                         + (2 * np.power(yt_val, 2) * mU3_sq_val)
                         + (2 * np.power(yb_val, 2) * mD3_sq_val)
                         + (2 * np.power(at_val, 2))
                         + (2 * np.power(ab_val, 2))
                         - ((32 / 3) * np.power(g3_val, 2)
                            * np.power(M3_val, 2))
                         - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))
                         - ((2 / 15) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((1 / 5) * np.power(g1_val, 2) * S_val))

        dmQ2_sq_dt_1l = (((mQ2_sq_val + (2 * mHu_sq_val))
                          * np.power(yc_val, 2))
                         + ((mQ2_sq_val + (2 * mHd_sq_val))
                            * np.power(ys_val, 2))
                         + ((np.power(yc_val, 2) + np.power(ys_val, 2))
                            * mQ2_sq_val)
                         + (2 * np.power(yc_val, 2) * mU2_sq_val)
                         + (2 * np.power(ys_val, 2) * mD2_sq_val)
                         + (2 * np.power(ac_val, 2))
                         + (2 * np.power(as_val, 2))
                         - ((32 / 3) * np.power(g3_val, 2)
                            * np.power(M3_val, 2))
                         - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))
                         - ((2 / 15) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((1 / 5) * np.power(g1_val, 2) * S_val))

        dmQ1_sq_dt_1l = (((mQ1_sq_val + (2 * mHu_sq_val))
                          * np.power(yu_val, 2))
                         + ((mQ1_sq_val + (2 * mHd_sq_val))
                            * np.power(yd_val, 2))
                         + ((np.power(yu_val, 2)
                             + np.power(yd_val, 2)) * mQ1_sq_val)
                         + (2 * np.power(yu_val, 2) * mU1_sq_val)
                         + (2 * np.power(yd_val, 2) * mD1_sq_val)
                         + (2 * np.power(au_val, 2))
                         + (2 * np.power(ad_val, 2))
                         - ((32 / 3) * np.power(g3_val, 2)
                            * np.power(M3_val, 2))
                         - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))
                         - ((2 / 15) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((1 / 5) * np.power(g1_val, 2) * S_val))

        # Left leptons
        dmL3_sq_dt_1l = (((mL3_sq_val + (2 * mHd_sq_val))
                          * np.power(ytau_val, 2))
                         + (2 * np.power(ytau_val, 2) * mE3_sq_val)
                         + (np.power(ytau_val, 2) * mL3_sq_val)
                         + (2 * np.power(atau_val, 2))
                         - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))
                         - ((6 / 5) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         - ((3 / 5) * np.power(g1_val, 2) * S_val))

        dmL2_sq_dt_1l = (((mL2_sq_val + (2 * mHd_sq_val))
                          * np.power(ymu_val, 2))
                         + (2 * np.power(ymu_val, 2) * mE2_sq_val)
                         + (np.power(ymu_val, 2) * mL2_sq_val)
                         + (2 * np.power(amu_val, 2))
                         - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))
                         - ((6 / 5) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         - ((3 / 5) * np.power(g1_val, 2) * S_val))

        dmL1_sq_dt_1l = (((mL1_sq_val + (2 * mHd_sq_val))
                          * np.power(ye_val, 2))
                         + (2 * np.power(ye_val, 2) * mE1_sq_val)
                         + (np.power(ye_val, 2) * mL1_sq_val)
                         + (2 * np.power(ae_val, 2))
                         - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))
                         - ((6 / 5) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         - ((3 / 5) * np.power(g1_val, 2) * S_val))

        # Right up-type squarks
        dmU3_sq_dt_1l = ((2 * (mU3_sq_val + (2 * mHd_sq_val))
                          * np.power(yt_val, 2))
                         + (4 * np.power(yt_val, 2) * mQ3_sq_val)
                         + (2 * np.power(yt_val, 2) * mU3_sq_val)
                         + (4 * np.power(at_val, 2))
                         - ((32 / 3) * np.power(g3_val, 2)
                            * np.power(M3_val, 2))
                         - ((32 / 15) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         - ((4 / 5) * np.power(g1_val, 2) * S_val))

        dmU2_sq_dt_1l = ((2 * (mU2_sq_val + (2 * mHd_sq_val))
                          * np.power(yc_val, 2))
                         + (4 * np.power(yc_val, 2) * mQ2_sq_val)
                         + (2 * np.power(yc_val, 2) * mU2_sq_val)
                         + (4 * np.power(ac_val, 2))
                         - ((32 / 3) * np.power(g3_val, 2)
                            * np.power(M3_val, 2))
                         - ((32 / 15) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         - ((4 / 5) * np.power(g1_val, 2) * S_val))

        dmU1_sq_dt_1l = ((2 * (mU1_sq_val + (2 * mHd_sq_val))
                          * np.power(yu_val, 2))
                         + (4 * np.power(yu_val, 2) * mQ1_sq_val)
                         + (2 * np.power(yu_val, 2) * mU1_sq_val)
                         + (4 * np.power(au_val, 2))
                         - ((32 / 3) * np.power(g3_val, 2)
                            * np.power(M3_val, 2))
                         - ((32 / 15) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         - ((4 / 5) * np.power(g1_val, 2) * S_val))

        # Right down-type squarks
        dmD3_sq_dt_1l = ((2 * (mD3_sq_val + (2 * mHd_sq_val))
                          * np.power(yb_val, 2))
                         + (4 * np.power(yb_val, 2) * mQ3_sq_val)
                         + (2 * np.power(yb_val, 2) * mD3_sq_val)
                         + (4 * np.power(ab_val, 2))
                         - ((32 / 3) * np.power(g3_val, 2)
                            * np.power(M3_val, 2))
                         - ((8 / 15) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((2 / 5) * np.power(g1_val, 2) * S_val))

        dmD2_sq_dt_1l = ((2 * (mD2_sq_val + (2 * mHd_sq_val))
                          * np.power(ys_val, 2))
                         + (4 * np.power(ys_val, 2) * mQ2_sq_val)
                         + (2 * np.power(ys_val, 2) * mD2_sq_val)
                         + (4 * np.power(as_val, 2))
                         - ((32 / 3) * np.power(g3_val, 2)
                            * np.power(M3_val, 2))
                         - ((8 / 15) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((2 / 5) * np.power(g1_val, 2) * S_val))

        dmD1_sq_dt_1l = ((2 * (mD1_sq_val + (2 * mHd_sq_val))
                          * np.power(yd_val, 2))
                         + (4 * np.power(yd_val, 2) * mQ1_sq_val)
                         + (2 * np.power(yd_val, 2) * mD1_sq_val)
                         + (4 * np.power(ad_val, 2))
                         - ((32 / 3) * np.power(g3_val, 2)
                            * np.power(M3_val, 2))
                         - ((8 / 15) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((2 / 5) * np.power(g1_val, 2) * S_val))

        # Right leptons
        dmE3_sq_dt_1l = ((2 * (mE3_sq_val + (2 * mHd_sq_val))
                          * np.power(ytau_val, 2))
                         + (4 * np.power(ytau_val, 2) * mL3_sq_val)
                         + (2 * np.power(ytau_val, 2) * mE3_sq_val)
                         + (4 * np.power(atau_val, 2))
                         - ((24 / 5) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((6 / 5) * np.power(g1_val, 2) * S_val))

        dmE2_sq_dt_1l = ((2 * (mE2_sq_val + (2 * mHd_sq_val))
                          * np.power(ymu_val, 2))
                         + (4 * np.power(ymu_val, 2) * mL2_sq_val)
                         + (2 * np.power(ymu_val, 2) * mE2_sq_val)
                         + (4 * np.power(amu_val, 2))
                         - ((24 / 5) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((6 / 5) * np.power(g1_val, 2) * S_val))

        dmE1_sq_dt_1l = ((2 * (mE1_sq_val + (2 * mHd_sq_val))
                          * np.power(ye_val, 2))
                         + (4 * np.power(ye_val, 2) * mL1_sq_val)
                         + (2 * np.power(ye_val, 2) * mE1_sq_val)
                         + (4 * np.power(ae_val, 2))
                         - ((24 / 5) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((6 / 5) * np.power(g1_val, 2) * S_val))

        # 2 loop parts of scalar squared masses
        # Left squarks
        dmQ3_sq_dt_2l = (((-8) * (mQ3_sq_val + mHu_sq_val + mU3_sq_val)
                          * np.power(yt_val, 4))
                         - (8 * (mQ3_sq_val + mHd_sq_val + mD3_sq_val)
                            * np.power(yb_val, 4))
                         - (np.power(yt_val, 2)
                            * ((2 * mQ3_sq_val) + (2 * mU3_sq_val)
                               + (4 * mHu_sq_val))# Tr(3Yu^2)
                            * 3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                         - (np.power(yb_val, 2)
                            * ((2 * mQ3_sq_val) + (2 * mD3_sq_val)
                               + (4 * mHd_sq_val))# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (6 * np.power(yt_val, 2)# Tr((mQ^2 + mU^2)*Yu^2)
                            * (((mQ3_sq_val + mU3_sq_val)
                                * np.power(yt_val, 2))
                               + ((mQ2_sq_val + mU2_sq_val)
                                  * np.power(yc_val, 2))
                               + ((mQ1_sq_val + mU1_sq_val)
                                  * np.power(yu_val, 2))))# end trace
                         - (np.power(yb_val, 2)# Tr(6(mQ^2 + mD^2)*Yd^2 + 2(mL^2 + mE^2)*Ye^2)
                            * ((6 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + (2 * (((mL3_sq_val + mE3_sq_val)
                                        * np.power(ytau_val, 2))
                                       + ((mL2_sq_val + mE2_sq_val)
                                          * np.power(ymu_val, 2))
                                       + ((mL1_sq_val + mE1_sq_val)
                                          * np.power(ye_val, 2))))# end trace
                               ))
                         - (16 * np.power(yt_val, 2) * np.power(at_val, 2))
                         - (16 * np.power(yb_val, 2) * np.power(ab_val, 2))
                         - (np.power(at_val, 2)# Tr(6Yu^2)
                            * 6 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                         - (np.power(yt_val, 2)# Tr(6au^2)
                            * 6 * (np.power(at_val, 2) + np.power(ac_val, 2)
                                   + np.power(au_val, 2)))# end trace
                         - (at_val * yt_val# Tr(12Yu*au)
                            * 12 * ((yt_val * at_val) + (yc_val * ac_val)
                                    + (yu_val * au_val)))# end trace
                         - (np.power(ab_val, 2)# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (np.power(yb_val, 2)# Tr(6ad^2 + 2ae^2)
                            * ((6 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + (2 * (np.power(atau_val, 2)
                                       + np.power(amu_val, 2)
                                       + np.power(ae_val, 2)))))# end trace
                         - (2 * ab_val * yb_val# Tr(6Yd*ad + 2Ye*ae)
                            * ((6 * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))
                               + (2 * ((ytau_val * atau_val)
                                       + (ymu_val * amu_val)
                                       + (ye_val * ae_val)))))# end trace
                         + ((2 / 5) * np.power(g1_val, 2)
                            * ((4 * (mQ3_sq_val + mHu_sq_val + mU3_sq_val)
                                * np.power(yt_val, 2))
                               + (4 * np.power(at_val, 2))
                               - (8 * M1_val * at_val * yt_val)
                               + (8 * np.power(M1_val, 2) * np.power(yt_val, 2))
                               + (2 * (mQ3_sq_val + mHd_sq_val + mD3_sq_val)
                                  * np.power(yb_val, 2))
                               + (2 * np.power(ab_val, 2))
                               - (4 * M1_val * ab_val * yb_val)
                               + (4 * np.power(M1_val, 2)
                                  * np.power(yb_val, 2))))
                         + ((2 / 5) * np.power(g1_val, 2) * Spr_val)
                         - ((128 / 3) * np.power(g3_val, 4)
                            * np.power(M3_val, 2))
                         + (32 * np.power(g3_val, 2) * np.power(g2_val, 2)
                            * (np.power(M3_val, 2) + np.power(M2_val, 2)
                               + (M2_val * M3_val)))
                         + ((32 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M3_val, 2) + np.power(M1_val, 2)
                               + (M3_val * M1_val)))
                         + (33 * np.power(g2_val, 4) * np.power(M2_val, 2))
                         + ((2 / 5) * np.power(g2_val, 2) * np.power(g1_val, 2)
                            * (np.power(M1_val, 2) + np.power(M2_val, 2)
                               + (M1_val * M2_val)))
                         + ((199 / 75) * np.power(g1_val, 4) * np.power(M1_val, 2))
                         + ((16 / 3) * np.power(g3_val, 2) * sigma3)
                         + (3 * np.power(g2_val, 2) * sigma2)
                         + ((1 / 15) * np.power(g1_val, 2) * sigma1))

        dmQ2_sq_dt_2l = (((-8) * (mQ2_sq_val + mHu_sq_val + mU2_sq_val)
                          * np.power(yc_val, 4))
                         - (8 * (mQ2_sq_val + mHd_sq_val + mD2_sq_val)
                            * np.power(ys_val, 4))
                         - (np.power(yc_val, 2)
                            * ((2 * mQ2_sq_val) + (2 * mU2_sq_val)
                               + (4 * mHu_sq_val))# Tr(3Yu^2)
                            * 3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                         - (np.power(ys_val, 2)
                            * ((2 * mQ2_sq_val) + (2 * mD2_sq_val)
                               + (4 * mHd_sq_val))# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (6 * np.power(yc_val, 2)# Tr((mQ^2 + mU^2)*Yu^2)
                            * (((mQ3_sq_val + mU3_sq_val)
                                * np.power(yt_val, 2))
                               + ((mQ2_sq_val + mU2_sq_val)
                                  * np.power(yc_val, 2))
                               + ((mQ1_sq_val + mU1_sq_val)
                                  * np.power(yu_val, 2))))# end trace
                         - (np.power(ys_val, 2)# Tr(6(mQ^2 + mD^2)*Yd^2 + 2(mL^2 + mE^2)*Ye^2)
                            * ((6 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + (2 * (((mL3_sq_val + mE3_sq_val)
                                        * np.power(ytau_val, 2))
                                       + ((mL2_sq_val + mE2_sq_val)
                                          * np.power(ymu_val, 2))
                                       + ((mL1_sq_val + mE1_sq_val)
                                          * np.power(ye_val, 2))))# end trace
                               ))
                         - (16 * np.power(yc_val, 2) * np.power(ac_val, 2))
                         - (16 * np.power(ys_val, 2) * np.power(as_val, 2))
                         - (np.power(ac_val, 2)# Tr(6Yu^2)
                            * 6 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                         - (np.power(yc_val, 2)# Tr(6au^2)
                            * 6 * (np.power(at_val, 2) + np.power(ac_val, 2)
                                   + np.power(au_val, 2)))# end trace
                         - (ac_val * yc_val# Tr(12Yu*au)
                            * 12 * ((yt_val * at_val) + (yc_val * ac_val)
                                    + (yu_val * au_val)))# end trace
                         - (np.power(as_val, 2)# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (np.power(ys_val, 2)# Tr(6ad^2 + 2ae^2)
                            * ((6 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + (2 * (np.power(atau_val, 2)
                                       + np.power(amu_val, 2)
                                       + np.power(ae_val, 2)))))# end trace
                         - (2 * as_val * ys_val# Tr(6Yd*ad + 2Ye*ae)
                            * ((6 * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))
                               + (2 * ((ytau_val * atau_val)
                                       + (ymu_val * amu_val)
                                       + (ye_val * ae_val)))))# end trace
                         + ((2 / 5) * np.power(g1_val, 2)
                            * ((4 * (mQ2_sq_val + mHu_sq_val + mU2_sq_val)
                                * np.power(yc_val, 2))
                               + (4 * np.power(ac_val, 2))
                               - (8 * M1_val * ac_val * yc_val)
                               + (8 * np.power(M1_val, 2) * np.power(yc_val, 2))
                               + (2 * (mQ2_sq_val + mHd_sq_val + mD2_sq_val)
                                  * np.power(ys_val, 2))
                               + (2 * np.power(as_val, 2))
                               - (4 * M1_val * as_val * ys_val)
                               + (4 * np.power(M1_val, 2)
                                  * np.power(ys_val, 2))))
                         + ((2 / 5) * np.power(g1_val, 2) * Spr_val)
                         - ((128 / 3) * np.power(g3_val, 4)
                            * np.power(M3_val, 2))
                         + (32 * np.power(g3_val, 2) * np.power(g2_val, 2)
                            * (np.power(M3_val, 2) + np.power(M2_val, 2)
                               + (M2_val * M3_val)))
                         + ((32 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M3_val, 2) + np.power(M1_val, 2)
                               + (M3_val * M1_val)))
                         + (33 * np.power(g2_val, 4) * np.power(M2_val, 2))
                         + ((2 / 5) * np.power(g2_val, 2) * np.power(g1_val, 2)
                            * (np.power(M1_val, 2) + np.power(M2_val, 2)
                               + (M1_val * M2_val)))
                         + ((199 / 75) * np.power(g1_val, 4) * np.power(M1_val, 2))
                         + ((16 / 3) * np.power(g3_val, 2) * sigma3)
                         + (3 * np.power(g2_val, 2) * sigma2)
                         + ((1 / 15) * np.power(g1_val, 2) * sigma1))

        dmQ1_sq_dt_2l = (((-8) * (mQ1_sq_val + mHu_sq_val + mU1_sq_val)
                          * np.power(yu_val, 4))
                         - (8 * (mQ1_sq_val + mHd_sq_val + mD1_sq_val)
                            * np.power(yd_val, 4))
                         - (np.power(yu_val, 2)
                            * ((2 * mQ1_sq_val) + (2 * mU1_sq_val)
                               + (4 * mHu_sq_val))# Tr(3Yu^2)
                            * 3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                         - (np.power(yd_val, 2)
                            * ((2 * mQ1_sq_val) + (2 * mD1_sq_val)
                               + (4 * mHd_sq_val))# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (6 * np.power(yu_val, 2)# Tr((mQ^2 + mU^2)*Yu^2)
                            * (((mQ3_sq_val + mU3_sq_val)
                                * np.power(yt_val, 2))
                               + ((mQ2_sq_val + mU2_sq_val)
                                  * np.power(yc_val, 2))
                               + ((mQ1_sq_val + mU1_sq_val)
                                  * np.power(yu_val, 2))))# end trace
                         - (np.power(yd_val, 2)# Tr(6(mQ^2 + mD^2)*Yd^2 + 2(mL^2 + mE^2)*Ye^2)
                            * ((6 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + (2 * (((mL3_sq_val + mE3_sq_val)
                                        * np.power(ytau_val, 2))
                                       + ((mL2_sq_val + mE2_sq_val)
                                          * np.power(ymu_val, 2))
                                       + ((mL1_sq_val + mE1_sq_val)
                                          * np.power(ye_val, 2))))# end trace
                               ))
                         - (16 * np.power(yu_val, 2) * np.power(au_val, 2))
                         - (16 * np.power(yd_val, 2) * np.power(ad_val, 2))
                         - (np.power(au_val, 2)# Tr(6Yu^2)
                            * 6 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                         - (np.power(yu_val, 2)# Tr(6au^2)
                            * 6 * (np.power(at_val, 2) + np.power(ac_val, 2)
                                   + np.power(au_val, 2)))# end trace
                         - (au_val * yu_val# Tr(12Yu*au)
                            * 12 * ((yt_val * at_val) + (yc_val * ac_val)
                                    + (yu_val * au_val)))# end trace
                         - (np.power(ad_val, 2)# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (np.power(yd_val, 2)# Tr(6ad^2 + 2ae^2)
                            * ((6 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + (2 * (np.power(atau_val, 2)
                                       + np.power(amu_val, 2)
                                       + np.power(ae_val, 2)))))# end trace
                         - (2 * ad_val * yd_val# Tr(6Yd*ad + 2Ye*ae)
                            * ((6 * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))
                               + (2 * ((ytau_val * atau_val)
                                       + (ymu_val * amu_val)
                                       + (ye_val * ae_val)))))# end trace
                         + ((2 / 5) * np.power(g1_val, 2)
                            * ((4 * (mQ1_sq_val + mHu_sq_val + mU1_sq_val)
                                * np.power(yu_val, 2))
                               + (4 * np.power(au_val, 2))
                               - (8 * M1_val * au_val * yu_val)
                               + (8 * np.power(M1_val, 2) * np.power(yu_val, 2))
                               + (2 * (mQ1_sq_val + mHd_sq_val + mD1_sq_val)
                                  * np.power(yd_val, 2))
                               + (2 * np.power(ad_val, 2))
                               - (4 * M1_val * ad_val * yd_val)
                               + (4 * np.power(M1_val, 2)
                                  * np.power(yd_val, 2))))
                         + ((2 / 5) * np.power(g1_val, 2) * Spr_val)
                         - ((128 / 3) * np.power(g3_val, 4)
                            * np.power(M3_val, 2))
                         + (32 * np.power(g3_val, 2) * np.power(g2_val, 2)
                            * (np.power(M3_val, 2) + np.power(M2_val, 2)
                               + (M2_val * M3_val)))
                         + ((32 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M3_val, 2) + np.power(M1_val, 2)
                               + (M3_val * M1_val)))
                         + (33 * np.power(g2_val, 4) * np.power(M2_val, 2))
                         + ((2 / 5) * np.power(g2_val, 2) * np.power(g1_val, 2)
                            * (np.power(M1_val, 2) + np.power(M2_val, 2)
                               + (M1_val * M2_val)))
                         + ((199 / 75) * np.power(g1_val, 4) * np.power(M1_val, 2))
                         + ((16 / 3) * np.power(g3_val, 2) * sigma3)
                         + (3 * np.power(g2_val, 2) * sigma2)
                         + ((1 / 15) * np.power(g1_val, 2) * sigma1))

        # Left leptons
        dmL3_sq_dt_2l = (((-8) * (mL3_sq_val + mHd_sq_val + mE3_sq_val)
                          * np.power(ytau_val, 4))
                         - (np.power(ytau_val, 2)
                            * ((2 * mL3_sq_val) + (2 * mE3_sq_val)
                               + (4 * mHd_sq_val))# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (np.power(ytau_val, 2)# Tr(6(mQ^2 + mD^2)*Yd^2 + 2(mL^2 + mE^2)*Ye^2)
                            * ((6 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + (2 * (((mL3_sq_val + mE3_sq_val)
                                        * np.power(ytau_val, 2))
                                       + ((mL2_sq_val + mE2_sq_val)
                                          * np.power(ymu_val, 2))
                                       + ((mL1_sq_val + mE1_sq_val)
                                          * np.power(ye_val, 2))))# end trace
                               ))
                         - (16 * np.power(ytau_val, 2) * np.power(atau_val, 2))
                         - (np.power(atau_val, 2)# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (np.power(ytau_val, 2)# Tr(6ad^2 + 2ae^2)
                            * ((6 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + (2 * (np.power(atau_val, 2)
                                       + np.power(amu_val, 2)
                                       + np.power(ae_val, 2)))))# end trace
                         - (2 * atau_val * ytau_val# Tr(6Yd*ad + 2Ye*ae)
                            * ((6 * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))
                               + (2 * ((ytau_val * atau_val)
                                       + (ymu_val * amu_val)
                                       + (ye_val * ae_val)))))# end trace
                         + ((6 / 5) * np.power(g1_val, 2)
                            * ((2 * (mL3_sq_val + mHd_sq_val + mE3_sq_val)
                                * np.power(ytau_val, 2))
                               + (2 * np.power(atau_val, 2))
                               - (4 * M1_val * atau_val
                                  * ytau_val)
                               + (4 * np.power(M1_val, 2)
                                  * np.power(ytau_val, 2))))
                         - ((6 / 5) * np.power(g1_val, 2) * Spr_val)
                         + (33 * np.power(g2_val, 4) * np.power(M2_val, 2))
                         + ((18 / 5) * np.power(g2_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M1_val, 2) + np.power(M2_val, 2)
                               + (M1_val * M2_val)))
                         + ((621 / 25) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + (3 * np.power(g2_val, 2) * sigma2)
                         + ((3 / 5) * np.power(g1_val, 2) * sigma1))

        dmL2_sq_dt_2l = (((-8) * (mL2_sq_val + mHd_sq_val + mE2_sq_val)
                          * np.power(ymu_val, 4))
                         - (np.power(ymu_val, 2)
                            * ((2 * mL2_sq_val) + (2 * mE2_sq_val)
                               + (4 * mHd_sq_val))# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (np.power(ymu_val, 2)# Tr(6(mQ^2 + mD^2)*Yd^2 + 2(mL^2 + mE^2)*Ye^2)
                            * ((6 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + (2 * (((mL3_sq_val + mE3_sq_val)
                                        * np.power(ytau_val, 2))
                                       + ((mL2_sq_val + mE2_sq_val)
                                          * np.power(ymu_val, 2))
                                       + ((mL1_sq_val + mE1_sq_val)
                                          * np.power(ye_val, 2))))# end trace
                               ))
                         - (16 * np.power(ymu_val, 2) * np.power(amu_val, 2))
                         - (np.power(amu_val, 2)# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (np.power(ymu_val, 2)# Tr(6ad^2 + 2ae^2)
                            * ((6 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + (2 * (np.power(atau_val, 2)
                                       + np.power(amu_val, 2)
                                       + np.power(ae_val, 2)))))# end trace
                         - (2 * amu_val * ymu_val# Tr(6Yd*ad + 2Ye*ae)
                            * ((6 * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))
                               + (2 * ((ytau_val * atau_val)
                                       + (ymu_val * amu_val)
                                       + (ye_val * ae_val)))))# end trace
                         + ((6 / 5) * np.power(g1_val, 2)
                            * ((2 * (mL2_sq_val + mHd_sq_val + mE2_sq_val)
                                * np.power(ymu_val, 2))
                               + (2 * np.power(amu_val, 2))
                               - (4 * M1_val * amu_val
                                  * ymu_val)
                               + (4 * np.power(M1_val, 2)
                                  * np.power(ymu_val, 2))))
                         - ((6 / 5) * np.power(g1_val, 2) * Spr_val)
                         + (33 * np.power(g2_val, 4) * np.power(M2_val, 2))
                         + ((18 / 5) * np.power(g2_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M1_val, 2) + np.power(M2_val, 2)
                               + (M1_val * M2_val)))
                         + ((621 / 25) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + (3 * np.power(g2_val, 2) * sigma2)
                         + ((3 / 5) * np.power(g1_val, 2) * sigma1))

        dmL1_sq_dt_2l = (((-8) * (mL1_sq_val + mHd_sq_val + mE1_sq_val)
                          * np.power(ye_val, 4))
                         - (np.power(ye_val, 2)
                            * ((2 * mL1_sq_val) + (2 * mE1_sq_val)
                               + (4 * mHd_sq_val))# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (np.power(ye_val, 2)# Tr(6(mQ^2 + mD^2)*Yd^2 + 2(mL^2 + mE^2)*Ye^2)
                            * ((6 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + (2 * (((mL3_sq_val + mE3_sq_val)
                                        * np.power(ytau_val, 2))
                                       + ((mL2_sq_val + mE2_sq_val)
                                          * np.power(ymu_val, 2))
                                       + ((mL1_sq_val + mE1_sq_val)
                                          * np.power(ye_val, 2))))# end trace
                               ))
                         - (16 * np.power(ye_val, 2) * np.power(ae_val, 2))
                         - (np.power(ae_val, 2)# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (np.power(ye_val, 2)# Tr(6ad^2 + 2ae^2)
                            * ((6 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + (2 * (np.power(atau_val, 2)
                                       + np.power(amu_val, 2)
                                       + np.power(ae_val, 2)))))# end trace
                         - (2 * ae_val * ye_val# Tr(6Yd*ad + 2Ye*ae)
                            * ((6 * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))
                               + (2 * ((ytau_val * atau_val)
                                       + (ymu_val * amu_val)
                                       + (ye_val * ae_val)))))# end trace
                         + ((6 / 5) * np.power(g1_val, 2)
                            * ((2 * (mL1_sq_val + mHd_sq_val + mE1_sq_val)
                                * np.power(ye_val, 2))
                               + (2 * np.power(ae_val, 2))
                               - (4 * M1_val * ae_val
                                  * ye_val)
                               + (4 * np.power(M1_val, 2)
                                  * np.power(ye_val, 2))))
                         - ((6 / 5) * np.power(g1_val, 2) * Spr_val)
                         + (33 * np.power(g2_val, 4) * np.power(M2_val, 2))
                         + ((18 / 5) * np.power(g2_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M1_val, 2) + np.power(M2_val, 2)
                               + (M1_val * M2_val)))
                         + ((621 / 25) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + (3 * np.power(g2_val, 2) * sigma2)
                         + ((3 / 5) * np.power(g1_val, 2) * sigma1))

        # Right up-type squarks
        dmU3_sq_dt_2l = (((-8) * (mQ3_sq_val + mHu_sq_val + mU3_sq_val)
                          * np.power(yt_val, 4))
                         - (4 * (mU3_sq_val + mHu_sq_val + mHd_sq_val
                                 + (2 * mQ3_sq_val) + mD3_sq_val)
                            * np.power(yb_val, 2) * np.power(yt_val, 2))
                         - (np.power(yt_val, 2)
                            * ((2 * mQ3_sq_val) + (2 * mU3_sq_val)
                               + (4 * mHu_sq_val))# Tr(6Yu^2)
                            * 6 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                         - (12 * np.power(yt_val, 2)# Tr((mQ^2 + mU^2)*Yu^2)
                            * (((mQ3_sq_val + mU3_sq_val)
                                * np.power(yt_val, 2))
                               + ((mQ2_sq_val + mU2_sq_val)
                                  * np.power(yc_val, 2))
                               + ((mQ1_sq_val + mU1_sq_val)
                                  * np.power(yu_val, 2))))# end trace
                         - (16 * np.power(yt_val, 2) * np.power(at_val, 2))
                         - (16 * at_val * ab_val * yb_val * yt_val)
                         - (12 * ((np.power(at_val, 2)# Tr(Yu^2)
                                   * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                      + np.power(yu_val, 2)))# end trace
                                  + (np.power(yt_val, 2) # Tr(au^2)
                                     * (np.power(at_val, 2)
                                        + np.power(ac_val, 2)
                                        + np.power(au_val, 2)))# end trace
                                  + (at_val * yt_val * 2# Tr(Yu*au)
                                     * ((yt_val * at_val) + (yc_val * ac_val)
                                        + (yu_val * au_val)))))# end trace
                         + (((6 * np.power(g2_val, 2))
                             - ((2 / 5) * np.power(g1_val, 2)))
                            * ((2 * (mQ3_sq_val + mHu_sq_val + mU3_sq_val)
                                * np.power(yt_val, 2))
                               + (2 * np.power(at_val, 2))))
                         + (12 * np.power(g2_val, 2)
                            * 2 * ((np.power(M2_val, 2) * np.power(yt_val, 2))
                                   - (M2_val * at_val * yt_val)))
                         - ((4 / 5) * np.power(g1_val, 2)
                            * 2 * ((np.power(M1_val, 2) * np.power(yt_val, 2))
                                   - (M1_val * at_val * yt_val)))
                         - ((8 / 5) * np.power(g1_val, 2) * Spr_val)
                         - ((128 / 3) * np.power(g3_val, 4)
                            * np.power(M3_val, 2))
                         + ((512 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M3_val, 2) + np.power(M1_val, 2)
                               + (M3_val * M1_val)))
                         + ((3424 / 75) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + ((16 / 3) * np.power(g3_val, 2) * sigma3)
                         + ((16 / 15) * np.power(g1_val, 2) * sigma1))

        dmU2_sq_dt_2l = (((-8) * (mQ2_sq_val + mHu_sq_val + mU2_sq_val)
                          * np.power(yc_val, 4))
                         - (4 * (mU2_sq_val + mHu_sq_val + mHd_sq_val
                                 + (2 * mQ2_sq_val)
                                 + mD2_sq_val)
                            * np.power(ys_val, 2) * np.power(yc_val, 2))
                         - (np.power(yc_val, 2)
                            * ((2 * mQ2_sq_val) + (2 * mU2_sq_val)
                               + (4 * mHu_sq_val))# Tr(6Yu^2)
                            * 6 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                         - (12 * np.power(yc_val, 2)# Tr((mQ^2 + mU^2)*Yu^2)
                            * (((mQ3_sq_val + mU3_sq_val)
                                * np.power(yt_val, 2))
                               + ((mQ2_sq_val + mU2_sq_val)
                                  * np.power(yc_val, 2))
                               + ((mQ1_sq_val + mU1_sq_val)
                                  * np.power(yu_val, 2))))# end trace
                         - (16 * np.power(yc_val, 2) * np.power(ac_val, 2))
                         - (16 * ac_val * as_val * ys_val * yc_val)
                         - (12 * ((np.power(ac_val, 2)# Tr(Yu^2)
                                   * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                      + np.power(yu_val, 2)))# end trace
                                  + (np.power(yc_val, 2) # Tr(au^2)
                                     * (np.power(at_val, 2)
                                        + np.power(ac_val, 2)
                                        + np.power(au_val, 2)))# end trace
                                  + (ac_val * yc_val * 2# Tr(Yu*au)
                                     * ((yt_val * at_val) + (yc_val * ac_val)
                                        + (yu_val * au_val)))))# end trace
                         + (((6 * np.power(g2_val, 2))
                             - ((2 / 5) * np.power(g1_val, 2)))
                            * ((2 * (mQ2_sq_val + mHu_sq_val + mU2_sq_val)
                                * np.power(yc_val, 2))
                               + (2 * np.power(ac_val, 2))))
                         + (12 * np.power(g2_val, 2)
                            * 2 * ((np.power(M2_val, 2) * np.power(yc_val, 2))
                                   - (M2_val * ac_val * yc_val)))
                         - ((4 / 5) * np.power(g1_val, 2)
                            * 2 * ((np.power(M1_val, 2) * np.power(yc_val, 2))
                                   - (M1_val * ac_val * yc_val)))
                         - ((8 / 5) * np.power(g1_val, 2) * Spr_val)
                         - ((128 / 3) * np.power(g3_val, 4)
                            * np.power(M3_val, 2))
                         + ((512 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M3_val, 2) + np.power(M1_val, 2)
                               + (M3_val * M1_val)))
                         + ((3424 / 75) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + ((16 / 3) * np.power(g3_val, 2) * sigma3)
                         + ((16 / 15) * np.power(g1_val, 2) * sigma1))

        dmU1_sq_dt_2l = (((-8) * (mQ1_sq_val + mHu_sq_val + mU1_sq_val)
                          * np.power(yu_val, 4))
                         - (4 * (mU1_sq_val + mHu_sq_val + mHd_sq_val
                                 + (2 * mQ1_sq_val)
                                 + mD1_sq_val)
                            * np.power(yd_val, 2) * np.power(yu_val, 2))
                         - (np.power(yu_val, 2)
                            * ((2 * mQ1_sq_val) + (2 * mU1_sq_val)
                               + (4 * mHu_sq_val))# Tr(6Yu^2)
                            * 6 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                         - (12 * np.power(yu_val, 2)# Tr((mQ^2 + mU^2)*Yu^2)
                            * (((mQ3_sq_val + mU3_sq_val)
                                * np.power(yt_val, 2))
                               + ((mQ2_sq_val + mU2_sq_val)
                                  * np.power(yc_val, 2))
                               + ((mQ1_sq_val + mU1_sq_val)
                                  * np.power(yu_val, 2))))# end trace
                         - (16 * np.power(yu_val, 2) * np.power(au_val, 2))
                         - (16 * au_val * ad_val * yd_val * yu_val)
                         - (12 * ((np.power(au_val, 2)# Tr(Yu^2)
                                   * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                      + np.power(yu_val, 2)))# end trace
                                  + (np.power(yu_val, 2) # Tr(au^2)
                                     * (np.power(at_val, 2)
                                        + np.power(ac_val, 2)
                                        + np.power(au_val, 2)))# end trace
                                  + (au_val * yu_val * 2# Tr(Yu*au)
                                     * ((yt_val * at_val) + (yc_val * ac_val)
                                        + (yu_val * au_val)))))# end trace
                         + (((6 * np.power(g2_val, 2))
                             - ((2 / 5) * np.power(g1_val, 2)))
                            * ((2 * (mQ1_sq_val + mHu_sq_val + mU1_sq_val)
                                * np.power(yu_val, 2))
                               + (2 * np.power(au_val, 2))))
                         + (12 * np.power(g2_val, 2)
                            * 2 * ((np.power(M2_val, 2) * np.power(yu_val, 2))
                                   - (M2_val * au_val * yu_val)))
                         - ((4 / 5) * np.power(g1_val, 2)
                            * 2 * ((np.power(M1_val, 2) * np.power(yu_val, 2))
                                   - (M1_val * au_val * yu_val)))
                         - ((8 / 5) * np.power(g1_val, 2) * Spr_val)
                         - ((128 / 3) * np.power(g3_val, 4)
                            * np.power(M3_val, 2))
                         + ((512 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M3_val, 2) + np.power(M1_val, 2)
                               + (M3_val * M1_val)))
                         + ((3424 / 75) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + ((16 / 3) * np.power(g3_val, 2) * sigma3)
                         + ((16 / 15) * np.power(g1_val, 2) * sigma1))

        # Right down-type squarks
        dmD3_sq_dt_2l = (((-8) * (mQ3_sq_val + mHd_sq_val + mD3_sq_val)
                          * np.power(yb_val, 4))
                         - (4 * (mU3_sq_val + mHu_sq_val + mHd_sq_val
                                 + (2 * mQ3_sq_val)
                                 + mD3_sq_val) * np.power(yb_val, 2)
                            * np.power(yt_val, 2))
                         - (np.power(yb_val, 2)
                            * (2 * (mD3_sq_val + mQ3_sq_val
                                    + (2 * mHd_sq_val)))# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (4 * np.power(yb_val, 2) # Tr(3(mQ^2 + mD^2) * Yd^2 + (mL^2 + mE^2) * Ye^2)
                            * ((3 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + (((mL3_sq_val + mE3_sq_val)
                                   * np.power(ytau_val, 2))
                                  + ((mL2_sq_val + mE2_sq_val)
                                     * np.power(ymu_val, 2))
                                  + ((mL1_sq_val + mE1_sq_val)
                                     * np.power(ye_val, 2)))# end trace
                               ))
                         - (16 * np.power(yb_val, 2) * np.power(ab_val, 2))
                         - (16 * at_val * ab_val * yb_val * yt_val)
                         - (4 * np.power(ab_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (4 * np.power(yb_val, 2) # Tr(3ad^2 + ae^2)
                            * ((3 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + np.power(atau_val, 2) + np.power(amu_val, 2)
                               + np.power(ae_val, 2)))# end trace
                         - (8 * ab_val * yb_val # Tr(3Yd * ad + Ye * ae)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         + (((6 * np.power(g2_val, 2))
                             + ((2 / 5) * np.power(g1_val, 2)))
                            * ((2 * (mQ3_sq_val + mHd_sq_val + mD3_sq_val)
                                * np.power(yb_val, 2))
                               + (2 * np.power(ab_val, 2))))
                         + (12 * np.power(g2_val, 2)
                            * 2 * ((np.power(M2_val, 2) * np.power(yb_val, 2))
                                   - (M2_val * ab_val * yb_val)))
                         + ((4 / 5) * np.power(g1_val, 2)
                            * 2 * ((np.power(M1_val, 2) * np.power(yb_val, 2))
                                   - (M1_val * ab_val * yb_val)))
                         + ((4 / 5) * np.power(g1_val, 2) * Spr_val)
                         - ((128 / 3) * np.power(g3_val, 4)
                            * np.power(M3_val, 2))
                         + ((128 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M3_val, 2) + np.power(M1_val, 2)
                               + (M3_val * M1_val)))
                         + ((808 / 75) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + ((16 / 3) * np.power(g3_val, 2) * sigma3)
                         + ((4 / 15) * np.power(g1_val, 2) * sigma1))

        dmD2_sq_dt_2l = (((-8) * (mQ2_sq_val + mHd_sq_val + mD2_sq_val)
                          * np.power(ys_val, 4))
                         - (4 * (mU2_sq_val + mHu_sq_val + mHd_sq_val
                                 + (2 * mQ2_sq_val)
                                 + mD2_sq_val) * np.power(ys_val, 2)
                            * np.power(yc_val, 2))
                         - (np.power(ys_val, 2)
                            * (2 * (mD2_sq_val + mQ2_sq_val
                                    + (2 * mHd_sq_val)))# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (4 * np.power(ys_val, 2) # Tr(3(mQ^2 + mD^2) * Yd^2 + (mL^2 + mE^2) * Ye^2)
                            * ((3 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + (((mL3_sq_val + mE3_sq_val)
                                   * np.power(ytau_val, 2))
                                  + ((mL2_sq_val + mE2_sq_val)
                                     * np.power(ymu_val, 2))
                                  + ((mL1_sq_val + mE1_sq_val)
                                     * np.power(ye_val, 2)))# end trace
                               ))
                         - (16 * np.power(ys_val, 2) * np.power(as_val, 2))
                         - (16 * ac_val * as_val * ys_val * yc_val)
                         - (4 * np.power(as_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (4 * np.power(ys_val, 2) # Tr(3ad^2 + ae^2)
                            * ((3 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + np.power(atau_val, 2) + np.power(amu_val, 2)
                               + np.power(ae_val, 2)))# end trace
                         - (8 * as_val * ys_val # Tr(3Yd * ad + Ye * ae)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         + (((6 * np.power(g2_val, 2))
                             + ((2 / 5) * np.power(g1_val, 2)))
                            * ((2 * (mQ2_sq_val + mHd_sq_val + mD2_sq_val)
                                * np.power(ys_val, 2))
                               + (2 * np.power(as_val, 2))))
                         + (12 * np.power(g2_val, 2)
                            * 2 * ((np.power(M2_val, 2) * np.power(ys_val, 2))
                                   - (M2_val * as_val * ys_val)))
                         + ((4 / 5) * np.power(g1_val, 2)
                            * 2 * ((np.power(M1_val, 2) * np.power(ys_val, 2))
                                   - (M1_val * as_val * ys_val)))
                         + ((4 / 5) * np.power(g1_val, 2) * Spr_val)
                         - ((128 / 3) * np.power(g3_val, 4)
                            * np.power(M3_val, 2))
                         + ((128 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M3_val, 2) + np.power(M1_val, 2)
                               + (M3_val * M1_val)))
                         + ((808 / 75) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + ((16 / 3) * np.power(g3_val, 2) * sigma3)
                         + ((4 / 15) * np.power(g1_val, 2) * sigma1))

        dmD1_sq_dt_2l = (((-8) * (mQ1_sq_val + mHd_sq_val + mD1_sq_val)
                          * np.power(yd_val, 4))
                         - (4 * (mU1_sq_val + mHu_sq_val + mHd_sq_val
                                 + (2 * mQ1_sq_val)
                                 + mD1_sq_val) * np.power(yd_val, 2)
                            * np.power(yu_val, 2))
                         - (np.power(yd_val, 2)
                            * (2 * (mD1_sq_val + mQ1_sq_val
                                    + (2 * mHd_sq_val)))# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (4 * np.power(yd_val, 2) # Tr(3(mQ^2 + mD^2) * Yd^2 + (mL^2 + mE^2) * Ye^2)
                            * ((3 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + (((mL3_sq_val + mE3_sq_val)
                                   * np.power(ytau_val, 2))
                                  + ((mL2_sq_val + mE2_sq_val)
                                     * np.power(ymu_val, 2))
                                  + ((mL1_sq_val + mE1_sq_val)
                                     * np.power(ye_val, 2)))# end trace
                               ))
                         - (16 * np.power(yd_val, 2) * np.power(ad_val, 2))
                         - (16 * au_val * ad_val * yd_val * yu_val)
                         - (4 * np.power(ad_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (4 * np.power(yd_val, 2) # Tr(3ad^2 + ae^2)
                            * ((3 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + np.power(atau_val, 2) + np.power(amu_val, 2)
                               + np.power(ae_val, 2)))# end trace
                         - (8 * ad_val * yd_val # Tr(3Yd * ad + Ye * ae)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         + (((6 * np.power(g2_val, 2))
                             + ((2 / 5) * np.power(g1_val, 2)))
                            * ((2 * (mQ1_sq_val + mHd_sq_val + mD1_sq_val)
                                * np.power(yd_val, 2))
                               + (2 * np.power(ad_val, 2))))
                         + (12 * np.power(g2_val, 2)
                            * 2 * ((np.power(M2_val, 2) * np.power(yd_val, 2))
                                   - (M2_val * ad_val * yd_val)))
                         + ((4 / 5) * np.power(g1_val, 2)
                            * 2 * ((np.power(M1_val, 2) * np.power(yd_val, 2))
                                   - (M1_val * ad_val * yd_val)))
                         + ((4 / 5) * np.power(g1_val, 2) * Spr_val)
                         - ((128 / 3) * np.power(g3_val, 4)
                            * np.power(M3_val, 2))
                         + ((128 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M3_val, 2) + np.power(M1_val, 2)
                               + (M3_val * M1_val)))
                         + ((808 / 75) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + ((16 / 3) * np.power(g3_val, 2) * sigma3)
                         + ((4 / 15) * np.power(g1_val, 2) * sigma1))

        # Right leptons
        dmE3_sq_dt_2l = (((-8) * (mL3_sq_val + mHd_sq_val + mE3_sq_val)
                         * np.power(ytau_val, 4))
            - (np.power(ytau_val, 2)
               * ((2 * mL3_sq_val) + (2 * mE3_sq_val)
                  + (4 * mHd_sq_val))# Tr(6Yd^2 + 2Ye^2)
               * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                        + np.power(yd_val, 2)))
                  + (2 * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                          + np.power(ye_val, 2)))))# end trace
            - (4 * np.power(ytau_val, 2) # Tr(3(mQ^2 + mD^2) * Yd^2 + (mL^2 + mE^2) * Ye^2)
               * ((3 * (((mQ3_sq_val + mD3_sq_val) * np.power(yb_val, 2))
                        + ((mQ2_sq_val + mD2_sq_val) * np.power(ys_val, 2))
                        + ((mQ1_sq_val + mD1_sq_val) * np.power(yd_val, 2))))
                  + ((((mL3_sq_val + mE3_sq_val) * np.power(ytau_val, 2))
                      + ((mL2_sq_val + mE2_sq_val) * np.power(ymu_val, 2))
                      + ((mL1_sq_val + mE1_sq_val) * np.power(ye_val, 2))))# end trace
                  ))
            - (16 * np.power(ytau_val, 2) * np.power(atau_val, 2))
            - (4 * np.power(atau_val, 2)# Tr(3Yd^2 + Ye^2)
               * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                        + np.power(yd_val, 2)))
                  + ((np.power(ytau_val, 2) + np.power(ymu_val, 2)
                      + np.power(ye_val, 2)))))# end trace
            - (4 * np.power(ytau_val, 2) # Tr(3ad^2 + ae^2)
               * ((3 * (np.power(ab_val, 2) + np.power(as_val, 2)
                        + np.power(ad_val, 2)))
                  + ((np.power(atau_val, 2) + np.power(amu_val, 2)
                      + np.power(ae_val, 2)))))# end trace
            - (8 * atau_val * ytau_val # Tr(3Yd * ad + Ye * ae)
               * ((3 * ((yb_val * ab_val) + (ys_val * as_val)
                         + (yd_val * ad_val)))
                  + (((ytau_val * atau_val) + (ymu_val * amu_val)
                      + (ye_val * ae_val)))))# end trace
            + (((6 * np.power(g2_val, 2)) - (6 / 5) * np.power(g1_val, 2))
               * ((2 * (mL3_sq_val + mHd_sq_val + mE3_sq_val)
                   * np.power(ytau_val, 2))
                  + (2 * np.power(atau_val, 2))))
            + (12 * np.power(g2_val, 2) * 2
               * ((np.power(M2_val, 2) * np.power(ytau_val, 2))
                  - (M2_val * atau_val * ytau_val)))
            - ((12 / 5) * np.power(g1_val, 2) * 2
               * ((np.power(M1_val, 2) * np.power(ytau_val, 2))
                  - (M1_val * atau_val * ytau_val)))
            + ((12 / 5) * np.power(g1_val, 2) * Spr_val)
            + ((2808 / 25) * np.power(g1_val, 4) * np.power(M1_val, 2))
            + ((12 / 5) * np.power(g1_val, 2) * sigma1))

        dmE2_sq_dt_2l = (((-8) * (mL2_sq_val + mHd_sq_val + mE2_sq_val)
                          * np.power(ymu_val, 4))
                         - (np.power(ymu_val, 2)
                            * ((2 * mL2_sq_val) + (2 * mE2_sq_val)
                               + (4 * mHd_sq_val))# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (4 * np.power(ymu_val, 2) # Tr(3(mQ^2 + mD^2) * Yd^2 + (mL^2 + mE^2) * Ye^2)
                            * ((3 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + ((((mL3_sq_val + mE3_sq_val)
                                    * np.power(ytau_val, 2))
                                   + ((mL2_sq_val + mE2_sq_val)
                                      * np.power(ymu_val, 2))
                                   + ((mL1_sq_val + mE1_sq_val)
                                      * np.power(ye_val, 2))))# end trace
                               ))
                         - (16 * np.power(ymu_val, 2) * np.power(amu_val, 2))
                         - (4 * np.power(amu_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + ((np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                   + np.power(ye_val, 2)))))# end trace
                         - (4 * np.power(ymu_val, 2) # Tr(3ad^2 + ae^2)
                            * ((3 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + ((np.power(atau_val, 2) + np.power(amu_val, 2)
                                   + np.power(ae_val, 2)))))# end trace
                         - (8 * amu_val * ymu_val # Tr(3Yd * ad + Ye * ae)
                            * ((3 * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))
                               + (((ytau_val * atau_val) + (ymu_val * amu_val)
                                   + (ye_val * ae_val)))))# end trace
                         + (((6 * np.power(g2_val, 2))
                             - (6 / 5) * np.power(g1_val, 2))
                            * ((2 * (mL2_sq_val + mHd_sq_val + mE2_sq_val)
                                * np.power(ymu_val, 2))
                               + (2 * np.power(amu_val, 2))))
                         + (12 * np.power(g2_val, 2) * 2
                            * ((np.power(M2_val, 2) * np.power(ymu_val, 2))
                               - (M2_val * amu_val * ymu_val)))
                         - ((12 / 5) * np.power(g1_val, 2) * 2
                            * ((np.power(M1_val, 2) * np.power(ymu_val, 2))
                               - (M1_val * amu_val * ymu_val)))
                         + ((12 / 5) * np.power(g1_val, 2) * Spr_val)
                         + ((2808 / 25) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + ((12 / 5) * np.power(g1_val, 2) * sigma1))

        dmE1_sq_dt_2l = (((-8) * (mL1_sq_val + mHd_sq_val + mE1_sq_val)
                          * np.power(ye_val, 4))
                         - (np.power(ye_val, 2)
                            * ((2 * mL1_sq_val) + (2 * mE1_sq_val)
                               + (4 * mHd_sq_val))# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (4 * np.power(ye_val, 2) # Tr(3(mQ^2 + mD^2) * Yd^2 + (mL^2 + mE^2) * Ye^2)
                            * ((3 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + ((((mL3_sq_val + mE3_sq_val)
                                    * np.power(ytau_val, 2))
                                   + ((mL2_sq_val + mE2_sq_val)
                                      * np.power(ymu_val, 2))
                                   + ((mL1_sq_val + mE1_sq_val)
                                      * np.power(ye_val, 2))))# end trace
                               ))
                         - (16 * np.power(ye_val, 2) * np.power(ae_val, 2))
                         - (4 * np.power(ae_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + ((np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                   + np.power(ye_val, 2)))))# end trace
                         - (4 * np.power(ye_val, 2) # Tr(3ad^2 + ae^2)
                            * ((3 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + ((np.power(atau_val, 2) + np.power(amu_val, 2)
                                   + np.power(ae_val, 2)))))# end trace
                         - (8 * ae_val * ye_val # Tr(3Yd * ad + Ye * ae)
                            * ((3 * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))
                               + (((ytau_val * atau_val) + (ymu_val * amu_val)
                                   + (ye_val * ae_val)))))# end trace
                         + (((6 * np.power(g2_val, 2))
                             - (6 / 5) * np.power(g1_val, 2))
                            * ((2 * (mL1_sq_val + mHd_sq_val + mE1_sq_val)
                                * np.power(ye_val, 2))
                               + (2 * np.power(ae_val, 2))))
                         + (12 * np.power(g2_val, 2) * 2
                            * ((np.power(M2_val, 2) * np.power(ye_val, 2))
                               - (M2_val * ae_val * ye_val)))
                         - ((12 / 5) * np.power(g1_val, 2) * 2
                            * ((np.power(M1_val, 2) * np.power(ye_val, 2))
                               - (M1_val * ae_val * ye_val)))
                         + ((12 / 5) * np.power(g1_val, 2) * Spr_val)
                         + ((2808 / 25) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + ((12 / 5) * np.power(g1_val, 2) * sigma1))

        # Total scalar squared mass beta functions
        dmQ3_sq_dt = (1 / t) * ((loop_fac * dmQ3_sq_dt_1l)
                                + (loop_fac_sq * dmQ3_sq_dt_2l))

        dmQ2_sq_dt = (1 / t) * ((loop_fac * dmQ2_sq_dt_1l)
                                + (loop_fac_sq * dmQ2_sq_dt_2l))

        dmQ1_sq_dt = (1 / t) * ((loop_fac * dmQ1_sq_dt_1l)
                                + (loop_fac_sq * dmQ1_sq_dt_2l))

        dmL3_sq_dt = (1 / t) * ((loop_fac * dmL3_sq_dt_1l)
                                + (loop_fac_sq * dmL3_sq_dt_2l))

        dmL2_sq_dt = (1 / t) * ((loop_fac * dmL2_sq_dt_1l)
                                + (loop_fac_sq * dmL2_sq_dt_2l))

        dmL1_sq_dt = (1 / t) * ((loop_fac * dmL1_sq_dt_1l)
                                + (loop_fac_sq * dmL1_sq_dt_2l))

        dmU3_sq_dt = (1 / t) * ((loop_fac * dmU3_sq_dt_1l)
                                + (loop_fac_sq * dmU3_sq_dt_2l))

        dmU2_sq_dt = (1 / t) * ((loop_fac * dmU2_sq_dt_1l)
                                + (loop_fac_sq * dmU2_sq_dt_2l))

        dmU1_sq_dt = (1 / t) * ((loop_fac * dmU1_sq_dt_1l)
                                + (loop_fac_sq * dmU1_sq_dt_2l))

        dmD3_sq_dt = (1 / t) * ((loop_fac * dmD3_sq_dt_1l)
                                + (loop_fac_sq * dmD3_sq_dt_2l))

        dmD2_sq_dt = (1 / t) * ((loop_fac * dmD2_sq_dt_1l)
                                + (loop_fac_sq * dmD2_sq_dt_2l))

        dmD1_sq_dt = (1 / t) * ((loop_fac * dmD1_sq_dt_1l)
                                + (loop_fac_sq * dmD1_sq_dt_2l))

        dmE3_sq_dt = (1 / t) * ((loop_fac * dmE3_sq_dt_1l)
                                + (loop_fac_sq * dmE3_sq_dt_2l))

        dmE2_sq_dt = (1 / t) * ((loop_fac * dmE2_sq_dt_1l)
                                + (loop_fac_sq * dmE2_sq_dt_2l))

        dmE1_sq_dt = (1 / t) * ((loop_fac * dmE1_sq_dt_1l)
                                + (loop_fac_sq * dmE1_sq_dt_2l))

        ##### Tanb RGE from arXiv:hep-ph/0112251 in R_xi=1 Feynman gauge #####
        # 1 loop part
        dtanb_dt_1l = 3 * (np.power(yt_val, 2) - np.power(yb_val, 2))

        # 2 loop part
        dtanb_dt_2l = (((-9) * (np.power(yt_val, 4) - np.power(yb_val, 4)))
                      + (6 * np.power(yt_val, 2)
                          * (((8 / 3) * np.power(g3_val, 2))
                            + ((6 / 45) * np.power(g1_val, 2))))
                      - (6 * np.power(yb_val, 2)
                          * (((8 / 3) * np.power(g3_val, 2))
                            - ((3 / 45) * np.power(g1_val, 2))))
                      - (3 * (np.power(yt_val, 2) - np.power(yb_val, 2))
                          * (((1 / np.sqrt(2))
                              * (((3 / 5) * np.power(g1_val, 2))
                                + np.power(g2_val, 2)))
                            + np.power(g2_val, 2))))

        # Total beta function for tanb
        dtanb_dt = (tanb_val / t) * ((loop_fac * dtanb_dt_1l)
                                    + (loop_fac_sq * dtanb_dt_2l))


        # Collect all for return
        dxdt = [dg1_dt, dg2_dt, dg3_dt, dM1_dt, dM2_dt, dM3_dt, dmu_dt, dyt_dt,
                dyc_dt, dyu_dt, dyb_dt, dys_dt, dyd_dt, dytau_dt, dymu_dt,
                dye_dt, dat_dt, dac_dt, dau_dt, dab_dt, das_dt, dad_dt,
                datau_dt, damu_dt, dae_dt, dmHu_sq_dt, dmHd_sq_dt,
                dmQ1_sq_dt, dmQ2_sq_dt, dmQ3_sq_dt, dmL1_sq_dt, dmL2_sq_dt,
                dmL3_sq_dt, dmU1_sq_dt, dmU2_sq_dt, dmU3_sq_dt, dmD1_sq_dt,
                dmD2_sq_dt, dmD3_sq_dt, dmE1_sq_dt, dmE2_sq_dt, dmE3_sq_dt,
                db_dt, dtanb_dt]
        return dxdt

    # Set up domains for solve_ivp
    if (inp_Q < target_Q_val):
        numpoints = ceil((np.log10(target_Q_val / inp_Q + 0.1)) * 1000)
        t_vals = np.logspace(np.log10(inp_Q), np.log10(target_Q_val),
                             numpoints)
        t_vals[0] = inp_Q + 0.1
        t_vals[-1] = target_Q_val
        t_span = np.array([inp_Q + 0.1, target_Q_val])
    
        # Now solve down to low scale
        sol = solve_ivp(my_odes, t_span, BCs, t_eval = t_vals,
                        dense_output=True, method='DOP853', atol=1e-9,
                        rtol=1e-9)
    else:
        numpoints = ceil((np.log10((inp_Q + 0.1) / target_Q_val)) * 1000)
        t_vals = np.logspace(np.log10(target_Q_val), np.log10(inp_Q),
                             numpoints)
        t_vals[0] = target_Q_val
        t_vals[-1] = inp_Q + 0.1
        t_span = np.array([inp_Q + 0.1, target_Q_val])

        # Now solve down to low scale
        sol = solve_ivp(my_odes, t_span, BCs, t_eval = t_vals[::-1],
                        dense_output=True, method='DOP853', atol=1e-9,
                        rtol=1e-9)
    myx1 = sol.y
    x1 = [myx1[0][-1], myx1[1][-1], myx1[2][-1], myx1[3][-1],
          myx1[4][-1], myx1[5][-1], myx1[6][-1], myx1[7][-1],
          myx1[8][-1], myx1[9][-1], myx1[10][-1], myx1[11][-1],
          myx1[12][-1], myx1[13][-1], myx1[14][-1], myx1[15][-1],
          myx1[16][-1], myx1[17][-1], myx1[18][-1], myx1[19][-1],
          myx1[20][-1], myx1[21][-1], myx1[22][-1], myx1[23][-1],
          myx1[24][-1], myx1[25][-1], myx1[26][-1], myx1[27][-1],
          myx1[28][-1], myx1[29][-1], myx1[30][-1], myx1[31][-1],
          myx1[32][-1], myx1[33][-1], myx1[34][-1], myx1[35][-1],
          myx1[36][-1], myx1[37][-1], myx1[38][-1], myx1[39][-1],
          myx1[40][-1], myx1[41][-1], myx1[42][-1], myx1[43][-1]]

    #print(x1[0])
    #print(x1[1])
    # Now evolve results up to 10^17 GeV, find approximate unification scale,
    # where g1(Q) is closest to g2(Q),
    # and return results for calculation of naturalness measures.
    numpoints2 = ceil((np.log10(1e17 / target_Q_val)) * 1000)
    t_vals2 = np.logspace(np.log10(target_Q_val), np.log10(1e17),
                          numpoints2)
    t_vals2[0] = target_Q_val
    t_vals2[-1] = 1e17
    t_span2 = np.array([target_Q_val, 1e17])
    sol2 = solve_ivp(my_odes, t_span2, x1, t_eval = t_vals2,
                     dense_output=True, method='DOP853', atol=1e-9,
                     rtol=1e-9)
    x2 = sol2.y
    # global GUT_check
    # global myg1s
    # myg1s = x2[0]
    # global myg2s
    # myg2s = x2[1]
    # GUT_check = np.abs(x2[0]-x2[1])
    GUT_idx = np.where(np.abs(x2[0] - x2[1])
                       == np.amin(np.abs(x2[0]-x2[1])))[0][0]
    approx_GUT_scale = t_vals2[GUT_idx]
    #print("Approx. GUT scale: " + str(approx_GUT_scale))

    # Results returned as:
    # (0: approximate GUT scale, 1: weak scale (2 TeV), 2: g1(weak),
    #  3: g2(weak), 4: g3(weak), 5: M1(weak), 6: M2(weak), 7: M3(weak),
    #  8: mu(weak), 9: yt(weak), 10: yc(weak), 11: yu(weak), 12: yb(weak),
    #  13: ys(weak), 14: yd(weak), 15: ytau(weak), 16: ymu(weak), 17: ye(weak),
    #  18: at(weak), 19: ac(weak), 20: au(weak), 21: ab(weak), 22: as(weak),
    #  23: ad(weak), 24: atau(weak), 25: amu(weak), 26: ae(weak),
    #  27: mHu^2(weak), 28: mHd^2(weak), 29: mQ1^2(weak), 30: mQ2^2(weak),
    #  31: mQ3^2(weak), 32: mL1^2(weak), 33: mL2^2(weak), 34: mL3^2(weak),
    #  35: mU1^2(weak), 36: mU2^2(weak), 37: mU3^2(weak), 38: mD1^2(weak),
    #  39: mD2^2(weak), 40: mD3^2(weak), 41: mE1^2(weak), 42: mE2^2(weak),
    #  43: mE3^2(weak), 44: b(weak), 45: g1(GUT), 46: g2(GUT), 47: g3(GUT),
    #  48: M1(GUT), 49: M2(GUT), 50: M3(GUT), 51: mu(GUT), 52: yt(GUT),
    #  53: yc(GUT), 54: yu(GUT), 55: yb(GUT), 56: ys(GUT), 57: yd(GUT),
    #  58: ytau(GUT), 59: ymu(GUT), 60: ye(GUT), 61: at(GUT), 62: ac(GUT),
    #  63: au(GUT), 64: ab(GUT), 65: as(GUT), 66: ad(GUT), 67: atau(GUT),
    #  68: amu(GUT), 69: ae(GUT), 70: mHu^2(GUT), 71: mHd^2(GUT),
    #  72: mQ1^2(GUT), 73: mQ2^2(GUT), 74: mQ3^2(GUT), 75: mL1^2(GUT),
    #  76: mL2^2(GUT), 77: mL3^2(GUT), 78: mU1^2(GUT), 79: mU2^2(GUT),
    #  80: mU3^2(GUT), 81: mD1^2(GUT), 82: mD2^2(GUT), 83: mD3^2(GUT),
    #  84: mE1^2(GUT), 85: mE2^2(GUT), 86: mE3^2(GUT), 87: b(GUT),
    #  88: tanb(weak), 89: tanb(GUT))
    sol_arrs = [approx_GUT_scale, target_Q_val, x2[0][0], x2[1][0], x2[2][0],
                x2[3][0], x2[4][0], x2[5][0], x2[6][0], x2[7][0], x2[8][0],
                x2[9][0], x2[10][0], x2[11][0], x2[12][0], x2[13][0],
                x2[14][0],
                x2[15][0], x2[16][0], x2[17][0], x2[18][0], x2[19][0],
                x2[20][0],
                x2[21][0], x2[22][0], x2[23][0], x2[24][0], x2[25][0],
                x2[26][0],
                x2[27][0], x2[28][0], x2[29][0], x2[30][0], x2[31][0],
                x2[32][0],
                x2[33][0], x2[34][0], x2[35][0], x2[36][0], x2[37][0],
                x2[38][0],
                x2[39][0], x2[40][0], x2[41][0], x2[42][0],
                x2[0][GUT_idx], x2[1][GUT_idx], x2[2][GUT_idx], x2[3][GUT_idx],
                x2[4][GUT_idx], x2[5][GUT_idx], x2[6][GUT_idx], x2[7][GUT_idx],
                x2[8][GUT_idx], x2[9][GUT_idx], x2[10][GUT_idx],
                x2[11][GUT_idx],
                x2[12][GUT_idx], x2[13][GUT_idx], x2[14][GUT_idx],
                x2[15][GUT_idx],
                x2[16][GUT_idx], x2[17][GUT_idx], x2[18][GUT_idx],
                x2[19][GUT_idx],
                x2[20][GUT_idx], x2[21][GUT_idx], x2[22][GUT_idx],
                x2[23][GUT_idx],
                x2[24][GUT_idx], x2[25][GUT_idx], x2[26][GUT_idx],
                x2[27][GUT_idx],
                x2[28][GUT_idx], x2[29][GUT_idx], x2[30][GUT_idx],
                x2[31][GUT_idx],
                x2[32][GUT_idx], x2[33][GUT_idx], x2[34][GUT_idx],
                x2[35][GUT_idx],
                x2[36][GUT_idx], x2[37][GUT_idx], x2[38][GUT_idx],
                x2[39][GUT_idx],
                x2[40][GUT_idx], x2[41][GUT_idx], x2[42][GUT_idx],
                x2[43][0], x2[43][GUT_idx]]
    return sol_arrs

def GUT_to_weak_runner(inpGUTBCs, GUT_Q, lowQ_val=2000.0):
    """
    Use scipy.integrate to evolve MSSM RGEs and collect solution vectors.

    Parameters
    ----------
    BCs : Array of floats.
        GUT scale boundary conditions for RGEs.
    GUT_Q : Float.
        Highest value for t parameter to run to in solution,
            typically unification scale from SoftSUSY.
    lowQ_val : Float.
        Lowest value for t parameter to run to in solution. Default is 2 TeV.

    Returns
    -------
    weakvals : Array of floats.
        Return solutions to system of RGEs as weak scale solutions.
        See before return statement for a comment on return array ordering.

    """
    def my_odes_rundown(t, x):
        """
        Define two-loop RGEs for soft terms.

        Parameters
        ----------
        x : Array of floats.
            Numerical solutions to RGEs. The order of entries in x is:
              (0: g1, 1: g2, 2: g3, 3: M1, 4: M2, 5: M3, 6: mu, 7: yt, 8: yc,
               9: yu, 10: yb, 11: ys, 12: yd, 13: ytau, 14: ymu, 15: ye,
               16: at, 17: ac, 18: au, 19: ab, 20: as, 21: ad, 22: atau,
               23: amu, 24: ae, 25: mHu^2, 26: mHd^2, 27: mQ1^2,
               28: mQ2^2, 29: mQ3^2, 30: mL1^2, 31: mL2^2, 32: mL3^2,
               33: mU1^2, 34: mU2^2, 35: mU3^2, 36: mD1^2, 37: mD2^2,
               38: mD3^2, 39: mE1^2, 40: mE2^2, 41: mE3^2, 42: b, 43: tanb)
        t : Array of evaluation renormalization scales.
            t = Q values for numerical solutions.

        Returns
        -------
        Array of floats.
            Return all soft RGEs evaluated at current t value.

        """
        # Unification scale is acquired from running a BM point through
        # SoftSUSY, then GUT scale boundary conditions are acquired from
        # SoftSUSY so that all three generations of Yukawas (assumed
        # to be diagonalized) are accounted for. A universal boundary condition
        # is used for soft scalar trilinear couplings a_i=y_i*A_i.
        # The soft b^(ij) mass^2 term is defined as b=B*mu, but is computed
        # in a later iteration.
        # Scalar mass matrices will also be written in diagonalized form such
        # that, e.g., mQ^2=((mQ1^2,0,0),(0,mQ2^2,0),(0,0,mQ3^2)).

        # Define all parameters in terms of solution vector x
        g1_val = x[0]
        g2_val = x[1]
        g3_val = x[2]
        M1_val = x[3]
        M2_val = x[4]
        M3_val = x[5]
        mu_val = x[6]
        yt_val = x[7]
        yc_val = x[8]
        yu_val = x[9]
        yb_val = x[10]
        ys_val = x[11]
        yd_val = x[12]
        ytau_val = x[13]
        ymu_val = x[14]
        ye_val = x[15]
        at_val = x[16]
        ac_val = x[17]
        au_val = x[18]
        ab_val = x[19]
        as_val = x[20]
        ad_val = x[21]
        atau_val = x[22]
        amu_val = x[23]
        ae_val = x[24]
        mHu_sq_val = x[25]
        mHd_sq_val = x[26]
        mQ1_sq_val = x[27]
        mQ2_sq_val = x[28]
        mQ3_sq_val = x[29]
        mL1_sq_val = x[30]
        mL2_sq_val = x[31]
        mL3_sq_val = x[32]
        mU1_sq_val = x[33]
        mU2_sq_val = x[34]
        mU3_sq_val = x[35]
        mD1_sq_val = x[36]
        mD2_sq_val = x[37]
        mD3_sq_val = x[38]
        mE1_sq_val = x[39]
        mE2_sq_val = x[40]
        mE3_sq_val = x[41]
        b_val = x[42]
        tanb_val = x[43]

        ##### Gauge couplings and gaugino masses #####
        # 1 loop parts
        dg1_dt_1l = b_1l[0] * np.power(g1_val, 3)

        dg2_dt_1l = b_1l[1] * np.power(g2_val, 3)

        dg3_dt_1l = b_1l[2] * np.power(g3_val, 3)

        dM1_dt_1l = b_1l[0] * np.power(g1_val, 2) * M1_val

        dM2_dt_1l = b_1l[1] * np.power(g2_val, 2) * M2_val

        dM3_dt_1l = b_1l[2] * np.power(g3_val, 2) * M3_val

        # 2 loop parts
        dg1_dt_2l = (np.power(g1_val, 3)
                     * ((b_2l[0][0] * np.power(g1_val, 2))
                        + (b_2l[0][1] * np.power(g2_val, 2))
                        + (b_2l[0][2] * np.power(g3_val, 2))# Tr(Yu^2)
                        - (c_2l[0][0] * (np.power(yt_val, 2)
                                         + np.power(yc_val, 2)
                                         + np.power(yu_val, 2)))# end trace, begin Tr(Yd^2)
                        - (c_2l[0][1] * (np.power(yb_val, 2)
                                         + np.power(ys_val, 2)
                                         + np.power(yd_val, 2)))# end trace, begin Tr(Ye^2)
                        - (c_2l[0][2] * (np.power(ytau_val, 2)
                                         + np.power(ymu_val, 2)
                                         + np.power(ye_val, 2)))))# end trace

        dg2_dt_2l = (np.power(g2_val, 3)
                     * ((b_2l[1][0] * np.power(g1_val, 2))
                        + (b_2l[1][1] * np.power(g2_val, 2))
                        + (b_2l[1][2] * np.power(g3_val, 2))# Tr(Yu^2)
                        - (c_2l[1][0] * (np.power(yt_val, 2)
                                         + np.power(yc_val, 2)
                                         + np.power(yu_val, 2)))# end trace, begin Tr(Yd^2)
                        - (c_2l[1][1] * (np.power(yb_val, 2)
                                         + np.power(ys_val, 2)
                                         + np.power(yd_val, 2)))# end trace, begin Tr(Ye^2)
                        - (c_2l[1][2] * (np.power(ytau_val, 2)
                                         + np.power(ymu_val, 2)
                                         + np.power(ye_val, 2)))))# end trace

        dg3_dt_2l = (np.power(g3_val, 3)
                     * ((b_2l[2][0] * np.power(g1_val, 2))
                        + (b_2l[2][1] * np.power(g2_val, 2))
                        + (b_2l[2][2] * np.power(g3_val, 2))# Tr(Yu^2)
                        - (c_2l[2][0] * (np.power(yt_val, 2)
                                         + np.power(yc_val, 2)
                                         + np.power(yu_val, 2)))# end trace, begin Tr(Yd^2)
                        - (c_2l[2][1] * (np.power(yb_val, 2)
                                         + np.power(ys_val, 2)
                                         + np.power(yd_val, 2)))# end trace, begin Tr(Ye^2)
                        - (c_2l[2][2] * (np.power(ytau_val, 2)
                                         + np.power(ymu_val, 2)
                                         + np.power(ye_val, 2)))))# end trace

        dM1_dt_2l = (2 * np.power(g1_val, 2)
                     * (((b_2l[0][0] * np.power(g1_val, 2) * (M1_val + M1_val))
                         + (b_2l[0][1] * np.power(g2_val, 2)
                            * (M1_val + M2_val))
                         + (b_2l[0][2] * np.power(g3_val, 2)
                            * (M1_val + M3_val)))# Tr(Yu*au)
                        + ((c_2l[0][0] * (((yt_val * at_val)
                                           + (yc_val * ac_val)
                                           + (yu_val * au_val))# end trace, begin Tr(Yu^2)
                                          - (M1_val * (np.power(yt_val, 2)
                                                       + np.power(yc_val, 2)
                                                       + np.power(yu_val, 2)))# end trace
                                          )))# Tr(Yd*ad)
                        + ((c_2l[0][1] * (((yb_val * ab_val)
                                           + (ys_val * as_val)
                                           + (yd_val * ad_val))# end trace, begin Tr(Yd^2)
                                          - (M1_val * (np.power(yb_val, 2)
                                                       + np.power(ys_val, 2)
                                                       + np.power(yd_val, 2)))# end trace
                                          )))# Tr(Ye*ae)
                        + ((c_2l[0][2] * (((ytau_val * atau_val)
                                           + (ymu_val * amu_val)
                                           + (ye_val * ae_val))# end trace, begin Tr(Ye^2)
                                          - (M1_val * (np.power(ytau_val, 2)
                                                       + np.power(ymu_val, 2)
                                                       + np.power(ye_val, 2)))
                                          )))))

        dM2_dt_2l = (2 * np.power(g2_val, 2)
                     * (((b_2l[1][0] * np.power(g1_val, 2) * (M2_val + M1_val))
                         + (b_2l[1][1] * np.power(g2_val, 2)
                            * (M2_val + M2_val))
                         + (b_2l[1][2] * np.power(g3_val, 2)
                            * (M2_val + M3_val)))# Tr(Yu*au)
                        + ((c_2l[1][0] * (((yt_val * at_val)
                                           + (yc_val * ac_val)
                                           + (yu_val * au_val))# end trace, begin Tr(Yu^2)
                                          - (M2_val * (np.power(yt_val, 2)
                                                       + np.power(yc_val, 2)
                                                       + np.power(yu_val, 2)))# end trace
                                          )))# Tr(Yd*ad)
                        + ((c_2l[1][1] * (((yb_val * ab_val)
                                           + (ys_val * as_val)
                                           + (yd_val * ad_val))# end trace, begin Tr(Yd^2)
                                          - (M2_val * (np.power(yb_val, 2)
                                                       + np.power(ys_val, 2)
                                                       + np.power(yd_val, 2)))# end trace
                                          )))# Tr(Ye*ae)
                        + ((c_2l[1][2] * (((ytau_val * atau_val)
                                           + (ymu_val * amu_val)
                                           + (ye_val * ae_val))# end trace, begin Tr(Ye^2)
                                          - (M2_val * (np.power(ytau_val, 2)
                                                       + np.power(ymu_val, 2)
                                                       + np.power(ye_val, 2)))# end trace
                                          )))))

        dM3_dt_2l = (2 * np.power(g3_val, 2)
                     * (((b_2l[2][0] * np.power(g1_val, 2) * (M3_val + M1_val))
                         + (b_2l[2][1] * np.power(g2_val, 2)
                            * (M3_val + M2_val))
                         + (b_2l[2][2] * np.power(g3_val, 2)
                            * (M3_val + M3_val)))# Tr(Yu*au)
                        + ((c_2l[2][0] * (((yt_val * at_val)
                                           + (yc_val * ac_val)
                                           + (yu_val * au_val))# end trace, begin Tr(Yu^2)
                                          - (M3_val * (np.power(yt_val, 2)
                                                       + np.power(yc_val, 2)
                                                       + np.power(yu_val, 2)))# end trace
                                          )))# Tr(Yd*ad)
                        + ((c_2l[2][1] * (((yb_val * ab_val)
                                           + (ys_val * as_val)
                                           + (yd_val * ad_val))# end trace, begin Tr(Yd^2)
                                          - (M3_val * (np.power(yb_val, 2)
                                                       + np.power(ys_val, 2)
                                                       + np.power(yd_val, 2)))# end trace
                                          )))# Tr(Ye*ae)
                        + ((c_2l[2][2] * (((ytau_val * atau_val)
                                           + (ymu_val * amu_val)
                                           + (ye_val * ae_val))# end trace, begin Tr(Ye^2)
                                          - (M3_val * (np.power(ytau_val, 2)
                                                       + np.power(ymu_val, 2)
                                                       + np.power(ye_val, 2)))# end trace
                                          )))))

        # Total gauge and gaugino mass beta functions
        dg1_dt = (1 / t) * ((loop_fac * dg1_dt_1l)
                            + (loop_fac_sq * dg1_dt_2l))

        dg2_dt = (1 / t) * ((loop_fac * dg2_dt_1l)
                            + (loop_fac_sq * dg2_dt_2l))

        dg3_dt = (1 / t) * ((loop_fac * dg3_dt_1l)
                            + (loop_fac_sq * dg3_dt_2l))

        dM1_dt = (2 / t) * ((loop_fac * dM1_dt_1l)
                             + (loop_fac_sq * dM1_dt_2l))

        dM2_dt = (2 / t) * ((loop_fac * dM2_dt_1l)
                             + (loop_fac_sq * dM2_dt_2l))

        dM3_dt = (2 / t) * ((loop_fac * dM3_dt_1l)
                             + (loop_fac_sq * dM3_dt_2l))

        ##### Higgsino mass parameter mu #####
        # 1 loop part
        dmu_dt_1l = (mu_val# Tr(3Yu^2 + 3Yd^2 + Ye^2)
                     * ((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2) + np.power(yb_val, 2)
                              + np.power(ys_val, 2) + np.power(yd_val, 2)))
                        + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                           + np.power(ye_val, 2))# end trace
                        - (3 * np.power(g2_val, 2))
                        - ((3 / 5) * np.power(g1_val, 2))))

        # 2 loop part
        dmu_dt_2l = (mu_val# Tr(3Yu^4 + 3Yd^4 + (2Yu^2*Yd^2) + Ye^4)
                     * ((-3 * ((3 * (np.power(yt_val, 4) + np.power(yc_val, 4)
                                     + np.power(yu_val, 4)
                                     + np.power(yb_val, 4)
                                     + np.power(ys_val, 4)
                                     + np.power(yd_val, 4)))
                               + (2 * ((np.power(yt_val, 2)
                                        * np.power(yb_val, 2))
                                       + (np.power(yc_val, 2)
                                          * np.power(ys_val, 2))
                                       + (np.power(yu_val, 2)
                                          * np.power(yd_val, 2))))
                               + (np.power(ytau_val, 4) + np.power(ymu_val, 4)
                                  + np.power(ye_val, 4))))# end trace
                        + (((16 * np.power(g3_val, 2))
                            + (4 * np.power(g1_val, 2) / 5))# Tr(Yu^2)
                           * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        + (((16 * np.power(g3_val, 2))
                            - (2 * np.power(g1_val, 2) / 5))# Tr(Yd^2)
                           * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))# end trace
                        + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                           * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))# end trace
                        + ((15 / 2) * np.power(g2_val, 4))
                        + ((9 / 5) * np.power(g1_val, 2)
                           * np.power(g2_val, 2))
                        + ((207 / 50) * np.power(g1_val, 4))))

        # Total mu beta function
        dmu_dt = (1 / t) * ((loop_fac * dmu_dt_1l)
                            + (loop_fac_sq * dmu_dt_2l))

        ##### Yukawa couplings for all 3 generations, assumed diagonalized#####
        # 1 loop parts
        dyt_dt_1l = (yt_val# Tr(3Yu^2)
                     * ((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        + (3 * (np.power(yt_val, 2)))
                        + np.power(yb_val, 2)
                        - ((16 / 3) * np.power(g3_val, 2))
                        - (3 * np.power(g2_val, 2))
                        - ((13 / 15) * np.power(g1_val, 2))))

        dyc_dt_1l = (yc_val# Tr(3Yu^2)
                     * ((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        + (3 * (np.power(yc_val, 2)))
                        + np.power(ys_val, 2)
                        - ((16 / 3) * np.power(g3_val, 2))
                        - (3 * np.power(g2_val, 2))
                        - ((13 / 15) * np.power(g1_val, 2))))

        dyu_dt_1l = (yu_val# Tr(3Yu^2)
                     * ((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        + (3 * (np.power(yu_val, 2)))
                        + np.power(yd_val, 2)
                        - ((16 / 3) * np.power(g3_val, 2))
                        - (3 * np.power(g2_val, 2))
                        - ((13 / 15) * np.power(g1_val, 2))))

        dyb_dt_1l = (yb_val# Tr(3Yd^2 + Ye^2)
                     * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))
                        + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                           + np.power(ye_val, 2))# end trace
                        + (3 * (np.power(yb_val, 2))) + np.power(yt_val, 2)
                        - ((16 / 3) * np.power(g3_val, 2))
                        - (3 * np.power(g2_val, 2))
                        - ((7 / 15) * np.power(g1_val, 2))))

        dys_dt_1l = (ys_val# Tr(3Yd^2 + Ye^2)
                     * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))
                        + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                           + np.power(ye_val, 2))# end trace
                        + (3 * (np.power(ys_val, 2))) + np.power(yc_val, 2)
                        - ((16 / 3) * np.power(g3_val, 2))
                        - (3 * np.power(g2_val, 2))
                        - ((7 / 15) * np.power(g1_val, 2))))

        dyd_dt_1l = (yd_val# Tr(3Yd^2 + Ye^2)
                     * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))
                        + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                           + np.power(ye_val, 2))# end trace
                        + (3 * (np.power(yd_val, 2))) + np.power(yu_val, 2)
                        - ((16 / 3) * np.power(g3_val, 2))
                        - (3 * np.power(g2_val, 2))
                        - ((7 / 15) * np.power(g1_val, 2))))

        dytau_dt_1l = (ytau_val# Tr(3Yd^2 + Ye^2)
                       * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                + np.power(yd_val, 2)))
                          + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                             + np.power(ye_val, 2))# end trace
                          + (3 * (np.power(ytau_val, 2)))
                          - (3 * np.power(g2_val, 2))
                          - ((9 / 5) * np.power(g1_val, 2))))

        dymu_dt_1l = (ymu_val# Tr(3Yd^2 + Ye^2)
                      * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                               + np.power(yd_val, 2)))
                         + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                            + np.power(ye_val, 2))# end trace
                         + (3 * (np.power(ymu_val, 2)))
                         - (3 * np.power(g2_val, 2))
                         - ((9 / 5) * np.power(g1_val, 2))))

        dye_dt_1l = (ye_val# Tr(3Yd^2 + Ye^2)
                     * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))
                        + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                           + np.power(ye_val, 2))# end trace
                        + (3 * (np.power(ye_val, 2)))
                        - (3 * np.power(g2_val, 2))
                        - ((9 / 5) * np.power(g1_val, 2))))

        # 2 loop parts
        dyt_dt_2l = (yt_val # Tr(3Yu^4 + (Yu^2*Yd^2))
                     * (((-3) * ((3 * (np.power(yt_val, 4)
                                       + np.power(yc_val, 4)
                                       + np.power(yu_val, 4)))
                                 + (np.power(yt_val, 2) * np.power(yb_val, 2))
                                 + (np.power(yc_val, 2) * np.power(ys_val, 2))
                                 + (np.power(yu_val, 2) * np.power(yd_val, 2)))
                         )# end trace
                        - (np.power(yb_val, 2)# Tr(3Yd^2 + Ye^2)
                           * ((3 * (np.power(yb_val, 2)
                                    + np.power(ys_val, 2)
                                    + np.power(yd_val, 2)))
                              + (np.power(ytau_val, 2)
                                 + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2))))# end trace
                        - (9 * np.power(yt_val, 2)# Tr(Yu^2)
                           * (np.power(yt_val, 2)
                              + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        - (4 * np.power(yt_val, 4))
                        - (2 * np.power(yb_val, 4))
                        - (2 * np.power(yb_val, 2) * np.power(yt_val, 2))
                        + (((16 * np.power(g3_val, 2))
                            + ((4 / 5) * np.power(g1_val, 2)))# Tr(Yu^2)
                           * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        + (((6 * np.power(g2_val, 2))
                            + ((2 / 5) * np.power(g1_val, 2)))
                           * np.power(yt_val, 2))
                        + ((2 / 5) * np.power(g1_val, 2) * np.power(yb_val, 2))
                        - ((16 / 9) * np.power(g3_val, 4))
                        + (8 * np.power(g3_val, 2) * np.power(g2_val, 2))
                        + ((136 / 45) * np.power(g3_val, 2)
                           * np.power(g1_val, 2))
                        + ((15 / 2) * np.power(g2_val, 4))
                        + (np.power(g2_val, 2) * np.power(g1_val, 2))
                        + ((2743 / 450) * np.power(g1_val, 4))))

        dyc_dt_2l = (yc_val # Tr(3Yu^4 + (Yu^2*Yd^2))
                     * (((-3) * ((3 * (np.power(yt_val, 4)
                                       + np.power(yc_val, 4)
                                       + np.power(yu_val, 4)))
                                 + (np.power(yt_val, 2) * np.power(yb_val, 2))
                                 + (np.power(yc_val, 2) * np.power(ys_val, 2))
                                 + (np.power(yu_val, 2) * np.power(yd_val, 2)))
                         )#end trace
                        - (np.power(ys_val, 2)# Tr(3Yd^2 + Ye^2)
                           * ((3 * (np.power(yb_val, 2)
                                    + np.power(ys_val, 2)
                                    + np.power(yd_val, 2)))
                              + (np.power(ytau_val, 2)
                                 + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2))))# end trace
                        - (9 * np.power(yc_val, 2)# Tr(Yu^2)
                           * (np.power(yt_val, 2)
                              + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        - (4 * np.power(yc_val, 4))
                        - (2 * np.power(ys_val, 4))
                        - (2 * np.power(ys_val, 2)
                           * np.power(yc_val, 2))
                        + (((16 * np.power(g3_val, 2))
                            + ((4 / 5) * np.power(g1_val, 2)))# Tr(Yu^2)
                           * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        + (((6 * np.power(g2_val, 2))
                            + ((2 / 5) * np.power(g1_val, 2)))
                           * np.power(yc_val, 2))
                        + ((2 / 5) * np.power(g1_val, 2) * np.power(ys_val, 2))
                        - ((16 / 9) * np.power(g3_val, 4))
                        + (8 * np.power(g3_val, 2) * np.power(g2_val, 2))
                        + ((136 / 45) * np.power(g3_val, 2)
                           * np.power(g1_val, 2))
                        + ((15 / 2) * np.power(g2_val, 4))
                        + (np.power(g2_val, 2) * np.power(g1_val, 2))
                        + ((2743 / 450) * np.power(g1_val, 4))))

        dyu_dt_2l = (yu_val# Tr(3Yu^4 + (Yu^2*Yd^2))
                     * (((-3) * ((3 * (np.power(yt_val, 4)
                                       + np.power(yc_val, 4)
                                       + np.power(yu_val, 4)))
                                 + (np.power(yt_val, 2) * np.power(yb_val, 2))
                                 + (np.power(yc_val, 2) * np.power(ys_val, 2))
                                 + (np.power(yu_val, 2) * np.power(yd_val, 2)))
                         )# end trace
                        - (np.power(yd_val, 2)# Tr(3Yd^2 + Ye^2)
                           * ((3 * (np.power(yb_val, 2)
                                    + np.power(ys_val, 2)
                                    + np.power(yd_val, 2)))
                              + (np.power(ytau_val, 2)
                                 + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2))))
                        - (9 * np.power(yu_val, 2)# Tr(Yu^2)
                           * (np.power(yt_val, 2)
                              + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))
                        - (4 * np.power(yu_val, 4))
                        - (2 * np.power(yd_val, 4))
                        - (2 * np.power(yd_val, 2) * np.power(yu_val, 2))
                        + (((16 * np.power(g3_val, 2))
                            + ((4 / 5) * np.power(g1_val, 2)))# Tr(Yu^2)
                           * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        + (((6 * np.power(g2_val, 2))
                            + ((2 / 5) * np.power(g1_val, 2)))
                           * np.power(yu_val, 2))
                        + ((2 / 5) * np.power(g1_val, 2) * np.power(yd_val, 2))
                        - ((16 / 9) * np.power(g3_val, 4))
                        + (8 * np.power(g3_val, 2) * np.power(g2_val, 2))
                        + ((136 / 45) * np.power(g3_val, 2)
                           * np.power(g1_val, 2))
                        + ((15 / 2) * np.power(g2_val, 4))
                        + (np.power(g2_val, 2) * np.power(g1_val, 2))
                        + ((2743 / 450) * np.power(g1_val, 4))))

        dyb_dt_2l = (yb_val# Tr(3Yd^4 + (Yu^2*Yd^2) + Ye^4)
                     * (((-3) * ((3 * (np.power(yb_val, 4)
                                       + np.power(ys_val, 4)
                                       + np.power(yd_val, 4)))
                                 + (np.power(yt_val, 2) * np.power(yb_val, 2))
                                 + (np.power(yc_val, 2) * np.power(ys_val, 2))
                                 + (np.power(yu_val, 2) * np.power(yd_val, 2))
                                 + np.power(ytau_val, 4) + np.power(ymu_val, 4)
                                 + np.power(ye_val, 4)))# end trace
                        - (3 * np.power(yt_val, 2)# Tr(Yu^2)
                           * (np.power(yt_val, 2)
                              + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        - (3 * np.power(yb_val, 2)# Tr(3Yd^2 + Ye^2)
                           * ((3 * (np.power(yb_val, 2)
                                    + np.power(ys_val, 2)
                                    + np.power(yd_val, 2)))
                              + np.power(ytau_val, 2)
                              + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))# end trace
                        - (4 * np.power(yb_val, 4))
                        - (2 * np.power(yt_val, 4))
                        - (2 * np.power(yt_val, 2) * np.power(yb_val, 2))
                        + (((16 * np.power(g3_val, 2))
                            - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                           * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))# end trace
                        + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                           * (np.power(ytau_val, 2)
                              + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))# end trace
                        + ((4 / 5) * np.power(g1_val, 2)
                           * np.power(yt_val, 2))
                        + (np.power(yb_val, 2)
                           * ((6 * np.power(g2_val, 2))
                              + ((4 / 5) * np.power(g1_val, 2))))
                        - ((16 / 9) * np.power(g3_val, 4))
                        + (8 * np.power(g3_val, 2) * np.power(g2_val, 2))
                        + ((8 / 9) * np.power(g3_val, 2) * np.power(g1_val, 2))
                        + ((15 / 2) * np.power(g2_val, 4))
                        + (np.power(g2_val, 2) * np.power(g1_val, 2))
                        + ((287 / 90) * np.power(g1_val, 4))))

        dys_dt_2l = (ys_val# Tr(3Yd^4 + (Yu^2*Yd^2) + Ye^4)
                     * (((-3) * ((3 * (np.power(yb_val, 4)
                                       + np.power(ys_val, 4)
                                       + np.power(yd_val, 4)))
                                 + (np.power(yt_val, 2) * np.power(yb_val, 2))
                                 + (np.power(yc_val, 2) * np.power(ys_val, 2))
                                 + (np.power(yu_val, 2) * np.power(yd_val, 2))
                                 + np.power(ytau_val, 4)
                                 + np.power(ymu_val, 4)
                                 + np.power(ye_val, 4)))# end trace
                        - (3 * np.power(yc_val, 2)# Tr(Yu^2)
                           * (np.power(yt_val, 2)
                              + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        - (3 * np.power(ys_val, 2)# Tr(3Yd^2 + Ye^2)
                           * ((3 * (np.power(yb_val, 2)
                                    + np.power(ys_val, 2)
                                    + np.power(yd_val, 2)))
                              + np.power(ytau_val, 2)
                              + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))# end trace
                        - (4 * np.power(ys_val, 4))
                        - (2 * np.power(yc_val, 4))
                        - (2 * np.power(yc_val, 2) * np.power(ys_val, 2))
                        + (((16 * np.power(g3_val, 2))
                            - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                           * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))# end trace
                        + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                           * (np.power(ytau_val, 2)
                              + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))# end trace
                        + ((4 / 5) * np.power(g1_val, 2) * np.power(yc_val, 2))
                        + (np.power(ys_val, 2)
                           * ((6 * np.power(g2_val, 2))
                              + ((4 / 5) * np.power(g1_val, 2))))
                        - ((16 / 9) * np.power(g3_val, 4))
                        + (8 * np.power(g3_val, 2) * np.power(g2_val, 2))
                        + ((8 / 9) * np.power(g3_val, 2) * np.power(g1_val, 2))
                        + ((15 / 2) * np.power(g2_val, 4))
                        + (np.power(g2_val, 2) * np.power(g1_val, 2))
                        + ((287 / 90) * np.power(g1_val, 4))))

        dyd_dt_2l = (yd_val# Tr(3Yd^4 + (Yu^2*Yd^2) + Ye^4)
                     * (((-3) * ((3 * (np.power(yb_val, 4)
                                       + np.power(ys_val, 4)
                                       + np.power(yd_val, 4)))
                                 + (np.power(yt_val, 2) * np.power(yb_val, 2))
                                 + (np.power(yc_val, 2) * np.power(ys_val, 2))
                                 + (np.power(yu_val, 2) * np.power(yd_val, 2))
                                 + np.power(ytau_val, 4) + np.power(ymu_val, 4)
                                 + np.power(ye_val, 4)))# end trace
                        - (3 * np.power(yu_val, 2)# Tr(Yu^2)
                           * (np.power(yt_val, 2)
                              + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        - (3 * np.power(yd_val, 2)# Tr(3Yd^2 + Ye^2)
                           * ((3 * (np.power(yb_val, 2)
                                    + np.power(ys_val, 2)
                                    + np.power(yd_val, 2)))
                              + np.power(ytau_val, 2)
                              + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))# end trace
                        - (4 * np.power(yd_val, 4))
                        - (2 * np.power(yu_val, 4))
                        - (2 * np.power(yd_val, 2) * np.power(yu_val, 2))
                        + (((16 * np.power(g3_val, 2))
                            - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                           * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))# end trace
                        + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                           * (np.power(ytau_val, 2)
                              + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))# end trace
                        + ((4 / 5) * np.power(g1_val, 2) * np.power(yu_val, 2))
                        + (np.power(yd_val, 2)
                           * ((6 * np.power(g2_val, 2))
                              + ((4 / 5) * np.power(g1_val, 2))))
                        - ((16 / 9) * np.power(g3_val, 4))
                        + (8 * np.power(g3_val, 2) * np.power(g2_val, 2))
                        + ((8 / 9) * np.power(g3_val, 2) * np.power(g1_val, 2))
                        + ((15 / 2) * np.power(g2_val, 4))
                        + (np.power(g2_val, 2) * np.power(g1_val, 2))
                        + ((287 / 90) * np.power(g1_val, 4))))

        dytau_dt_2l = (ytau_val# Tr(3Yd^4 + (Yu^2*Yd^2) + Ye^4)
                       * (((-3) * ((3 * (np.power(yb_val, 4)
                                         + np.power(ys_val, 4)
                                         + np.power(yd_val, 4)))
                                   + (np.power(yt_val, 2)
                                      * np.power(yb_val, 2))
                                   + (np.power(yc_val, 2)
                                      * np.power(ys_val, 2))
                                   + (np.power(yu_val, 2)
                                      * np.power(yd_val, 2))
                                   + np.power(ytau_val, 4)
                                   + np.power(ymu_val, 4)
                                   + np.power(ye_val, 4)))# end trace
                          - (3 * np.power(ytau_val, 2)# Tr(3Yd^2 + Ye^2)
                             * ((3 * (np.power(yb_val, 2)
                                      + np.power(ys_val, 2)
                                      + np.power(yd_val, 2)))
                                + np.power(ytau_val, 2)
                                + np.power(ymu_val, 2)
                                + np.power(ye_val, 2)))# end trace
                          - (4 * np.power(ytau_val, 4))
                          + (((16 * np.power(g3_val, 2))
                              - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                             * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                + np.power(yd_val, 2)))# end trace
                          + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                             * (np.power(ytau_val, 2)
                                + np.power(ymu_val, 2)
                                + np.power(ye_val, 2)))# end trace
                          + (6 * np.power(g2_val, 2) * np.power(ytau_val, 2))
                          + ((15 / 2) * np.power(g2_val, 4))
                          + ((9 / 5) * np.power(g2_val, 2)
                             * np.power(g1_val, 2))
                          + ((27 / 2) * np.power(g1_val, 4))))

        dymu_dt_2l = (ymu_val# Tr(3Yd^4 + (Yu^2*Yd^2) + Ye^4)
                      * (((-3) * ((3 * (np.power(yb_val, 4)
                                        + np.power(ys_val, 4)
                                        + np.power(yd_val, 4)))
                                  + (np.power(yt_val, 2)
                                     * np.power(yb_val, 2))
                                  + (np.power(yc_val, 2)
                                     * np.power(ys_val, 2))
                                  + (np.power(yu_val, 2)
                                     * np.power(yd_val, 2))
                                  + np.power(ytau_val, 4)
                                  + np.power(ymu_val, 4)
                                  + np.power(ye_val, 4)))# end trace
                         - (3 * np.power(ymu_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2)
                                     + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2)
                               + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (4 * np.power(ymu_val, 4))
                         + (((16 * np.power(g3_val, 2))
                             - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                            * (np.power(yb_val, 2) + np.power(ys_val, 2)
                               + np.power(yd_val, 2)))# end trace
                         + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                            * (np.power(ytau_val, 2)
                               + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         + (6 * np.power(g2_val, 2) * np.power(ymu_val, 2))
                         + ((15 / 2) * np.power(g2_val, 4))
                         + ((9 / 5) * np.power(g2_val, 2) * np.power(g1_val, 2))
                         + ((27 / 2) * np.power(g1_val, 4))))

        dye_dt_2l = (ye_val# Tr(3Yd^4 + (Yu^2*Yd^2) + Ye^4)
                     * (((-3) * ((3 * (np.power(yb_val, 4)
                                       + np.power(ys_val, 4)
                                       + np.power(yd_val, 4)))
                                 + (np.power(yt_val, 2)
                                    * np.power(yb_val, 2))
                                 + (np.power(yc_val, 2)
                                    * np.power(ys_val, 2))
                                 + (np.power(yu_val, 2)
                                    * np.power(yd_val, 2))
                                 + np.power(ytau_val, 4)
                                 + np.power(ymu_val, 4)
                                 + np.power(ye_val, 4)))# end trace
                        - (3 * np.power(ye_val, 2)# Tr(3Yd^2 + Ye^2)
                           * ((3 * (np.power(yb_val, 2)
                                    + np.power(ys_val, 2)
                                    + np.power(yd_val, 2)))
                              + np.power(ytau_val, 2)
                              + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))# end trace
                        - (4 * np.power(ye_val, 4))
                        + (((16 * np.power(g3_val, 2))
                            - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                           * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))# end trace
                        + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                           * (np.power(ytau_val, 2)
                              + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))
                        + (6 * np.power(g2_val, 2) * np.power(ye_val, 2))
                        + ((15 / 2) * np.power(g2_val, 4))
                        + ((9 / 5) * np.power(g2_val, 2) * np.power(g1_val, 2))
                        + ((27 / 2) * np.power(g1_val, 4))))

        # Total Yukawa coupling beta functions
        dyt_dt = (1 / t) * ((loop_fac * dyt_dt_1l)
                            + (loop_fac_sq * dyt_dt_2l))

        dyc_dt = (1 / t) * ((loop_fac * dyc_dt_1l)
                            + (loop_fac_sq * dyc_dt_2l))

        dyu_dt = (1 / t) * ((loop_fac * dyu_dt_1l)
                            + (loop_fac_sq * dyu_dt_2l))

        dyb_dt = (1 / t) * ((loop_fac * dyb_dt_1l)
                            + (loop_fac_sq * dyb_dt_2l))

        dys_dt = (1 / t) * ((loop_fac * dys_dt_1l)
                            + (loop_fac_sq * dys_dt_2l))

        dyd_dt = (1 / t) * ((loop_fac * dyd_dt_1l)
                            + (loop_fac_sq * dyd_dt_2l))

        dytau_dt = (1 / t) * ((loop_fac * dytau_dt_1l)
                            + (loop_fac_sq * dytau_dt_2l))

        dymu_dt = (1 / t) * ((loop_fac * dymu_dt_1l)
                            + (loop_fac_sq * dymu_dt_2l))

        dye_dt = (1 / t) * ((loop_fac * dye_dt_1l)
                            + (loop_fac_sq * dye_dt_2l))

        ##### Soft trilinear couplings for 3 gen, assumed diagonalized #####
        # 1 loop parts
        dat_dt_1l = ((at_val# Tr(Yu^2)
                      * ((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         + (5 * np.power(yt_val, 2)) + np.power(yb_val, 2)
                         - ((16 / 3) * np.power(g3_val, 2))
                         - (3 * np.power(g2_val, 2))
                         - ((13 / 15) * np.power(g1_val, 2))))
                     + (yt_val# Tr(au*Yu)
                        * ((6 * ((at_val * yt_val) + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           + (4 * yt_val * at_val)
                           + (2 * yb_val * ab_val)
                           + ((32 / 3) * np.power(g3_val, 2) * M3_val)
                           + (6 * np.power(g2_val, 2) * M2_val)
                           + ((26 / 15) * np.power(g1_val, 2) * M1_val))))

        dac_dt_1l = ((ac_val# Tr(Yu^2)
                      * ((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         + (5 * np.power(yc_val, 2)) + np.power(ys_val, 2)
                         - ((16 / 3) * np.power(g3_val, 2))
                         - (3 * np.power(g2_val, 2))
                         - ((13 / 15) * np.power(g1_val, 2))))
                     + (yc_val# Tr(au*Yu)
                        * ((6 * ((at_val * yt_val) + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           + (4 * yc_val * ac_val)
                           + (2 * ys_val * as_val)
                           + ((32 / 3) * np.power(g3_val, 2) * M3_val)
                           + (6 * np.power(g2_val, 2) * M2_val)
                           + ((26 / 15) * np.power(g1_val, 2) * M1_val))))

        dau_dt_1l = ((au_val# Tr(Yu^2)
                      * ((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         + (5 * np.power(yu_val, 2)) + np.power(yd_val, 2)
                         - ((16 / 3) * np.power(g3_val, 2))
                         - (3 * np.power(g2_val, 2))
                         - ((13 / 15) * np.power(g1_val, 2))))
                     + (yu_val# Tr(au*Yu)
                        * ((6 * ((at_val * yt_val) + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           + (4 * yu_val * au_val)
                           + (2 * yd_val * ad_val)
                           + ((32 / 3) * np.power(g3_val, 2) * M3_val)
                           + (6 * np.power(g2_val, 2) * M2_val)
                           + ((26 / 15) * np.power(g1_val, 2) * M1_val))))

        dab_dt_1l = ((ab_val# Tr(3Yd^2 + Ye^2)
                      * (((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                + np.power(yd_val, 2)))
                          + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                             + np.power(ye_val, 2)))# end trace
                         + (5 * np.power(yb_val, 2)) + np.power(yt_val, 2)
                         - ((16 / 3) * np.power(g3_val, 2))
                         - (3 * np.power(g2_val, 2))
                         - ((7 / 15) * np.power(g1_val, 2))))
                     + (yb_val# Tr(6ad*Yd + 2ae*Ye)
                        * ((6 * ((ab_val * yb_val) + (as_val * ys_val)
                                 + (ad_val * yd_val)))
                           + (2 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                   + (ae_val * ye_val)))# end trace
                           + (4 * yb_val * ab_val) + (2 * yt_val * at_val)
                           + ((32 / 3) * np.power(g3_val, 2) * M3_val)
                           + (6 * np.power(g2_val, 2) * M2_val)
                           + ((14 / 15) * np.power(g1_val, 2) * M1_val))))

        das_dt_1l = ((as_val# Tr(3Yd^2 + Ye^2)
                      * (((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                + np.power(yd_val, 2)))
                          + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                             + np.power(ye_val, 2)))# end trace
                         + (5 * np.power(ys_val, 2)) + np.power(yc_val, 2)
                         - ((16 / 3) * np.power(g3_val, 2))
                         - (3 * np.power(g2_val, 2))
                         - ((7 / 15) * np.power(g1_val, 2))))
                     + (ys_val# Tr(6ad*Yd + 2ae*Ye)
                        * ((6 * ((ab_val * yb_val) + (as_val * ys_val)
                                 + (ad_val * yd_val)))
                           + (2 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                   + (ae_val * ye_val)))
                           + (4 * ys_val * as_val) + (2 * yc_val * ac_val)
                           + ((32 / 3) * np.power(g3_val, 2) * M3_val)
                           + (6 * np.power(g2_val, 2) * M2_val)
                           + ((14 / 15) * np.power(g1_val, 2) * M1_val))))

        dad_dt_1l = ((ad_val# Tr(3Yd^2 + Ye^2)
                      * (((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                + np.power(yd_val, 2)))
                          + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                             + np.power(ye_val, 2)))# end trace
                         + (5 * np.power(yd_val, 2)) + np.power(yu_val, 2)
                         - ((16 / 3) * np.power(g3_val, 2))
                         - (3 * np.power(g2_val, 2))
                         - ((7 / 15) * np.power(g1_val, 2))))
                     + (yd_val# Tr(6ad*Yd + 2ae*Ye)
                        * ((6 * ((ab_val * yb_val) + (as_val * ys_val)
                                 + (ad_val * yd_val)))
                           + (2 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                   + (ae_val * ye_val)))# end trace
                           + (4 * yd_val * ad_val) + (2 * yu_val * au_val)
                           + ((32 / 3) * np.power(g3_val, 2) * M3_val)
                           + (6 * np.power(g2_val, 2) * M2_val)
                           + ((14 / 15) * np.power(g1_val, 2) * M1_val))))

        datau_dt_1l = ((atau_val# Tr(3Yd^2 + Ye^2)
                        * (((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                  + np.power(yd_val, 2)))
                            + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                           + (5 * np.power(ytau_val, 2))
                           - (3 * np.power(g2_val, 2))
                           - ((9 / 5) * np.power(g1_val, 2))))
                       + (ytau_val# Tr(6ad*Yd + 2ae*Ye)
                          * ((6 * ((ab_val * yb_val) + (as_val * ys_val)
                                   + (ad_val * yd_val)))
                             + (2 * ((atau_val * ytau_val)
                                     + (amu_val * ymu_val)
                                     + (ae_val * ye_val)))# end trace
                             + (4 * ytau_val * atau_val)
                             + (6 * np.power(g2_val, 2) * M2_val)
                             + ((18 / 5) * np.power(g1_val, 2) * M1_val))))

        damu_dt_1l = ((amu_val# Tr(3Yd^2 + Ye^2)
                       * (((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                 + np.power(yd_val, 2)))
                           + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))# end trace
                          + (5 * np.power(ymu_val, 2))
                          - (3 * np.power(g2_val, 2))
                          - ((9 / 5) * np.power(g1_val, 2))))
                      + (ymu_val# Tr(6ad*Yd + 2ae*Ye)
                         * ((6 * ((ab_val * yb_val) + (as_val * ys_val)
                                  + (ad_val * yd_val)))
                            + (2 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                    + (ae_val * ye_val)))# end trace
                            + (4 * ymu_val * amu_val)
                            + (6 * np.power(g2_val, 2) * M2_val)
                            + ((18 / 5) * np.power(g1_val, 2) * M1_val))))

        dae_dt_1l = ((ae_val# Tr(3Yd^2 + Ye^2)
                      * (((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                + np.power(yd_val, 2)))
                          + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                             + np.power(ye_val, 2)))# end trace
                         + (5 * np.power(ye_val, 2))
                         - (3 * np.power(g2_val, 2))
                         - ((9 / 5) * np.power(g1_val, 2))))
                     + (ye_val# Tr(6ad*Yd + 2ae*Ye)
                        * ((6 * ((ab_val * yb_val) + (as_val * ys_val)
                                 + (ad_val * yd_val)))
                           + (2 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                   + (ae_val * ye_val)))# end trace
                           + (4 * ye_val * ae_val)
                           + (6 * np.power(g2_val, 2) * M2_val)
                           + ((18 / 5) * np.power(g1_val, 2) * M1_val))))

        # 2 loop parts
        dat_dt_2l = ((at_val# Tr(3Yu^4 + (Yu^2*Yd^2))
                      * (((-3) * ((3 * (np.power(yt_val, 4)
                                        + np.power(yc_val, 4)
                                        + np.power(yu_val, 4)))
                                  + ((np.power(yt_val, 2) * np.power(yb_val,2))
                                     + (np.power(yc_val, 2)
                                        * np.power(ys_val, 2))
                                     + (np.power(yu_val, 2)
                                        * np.power(yd_val, 2)))))# end trace
                         - (np.power(yb_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2)
                                     + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (np.power(ytau_val, 2)
                                  + np.power(ymu_val, 2)
                                  + np.power(ye_val, 2))))# end trace
                         - (15 * np.power(yt_val, 2)# Tr(Yu^2)
                            * (np.power(yt_val, 2)
                               + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         - (6 * np.power(yt_val, 4))
                         - (2 * np.power(yb_val, 4))
                         - (4 * np.power(yb_val, 2) * np.power(yt_val, 2))
                         + (((16 * np.power(g3_val, 2))
                             + ((4 / 5) * np.power(g1_val, 2)))# Tr(Yu^2)
                            * (np.power(yt_val, 2)
                               + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         + (12 * np.power(g2_val, 2)
                            * np.power(yt_val, 2))
                         + ((2 / 5) * np.power(g1_val, 2)
                            * np.power(yb_val, 2))
                         - ((16 / 9) * np.power(g3_val, 4))
                         + (8 * np.power(g3_val, 2)
                            * np.power(g2_val, 2))
                         + ((136 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2))
                         + ((15 / 2) * np.power(g2_val, 4))
                         + (np.power(g2_val, 2) * np.power(g1_val, 2))
                         + ((2743 / 450) * np.power(g1_val, 4))))
                     + (yt_val# Tr(6au*Yu^3 + au*Yd^2*Yu + ad*Yu^2*Yd)
                        * (((-6) * ((6 * ((at_val * np.power(yt_val, 3))
                                          + (ac_val * np.power(yc_val, 3))
                                          + (au_val * np.power(yu_val, 3))))
                                    + (at_val * np.power(yb_val, 2) * yt_val)
                                    + (ac_val * np.power(ys_val, 2) * yc_val)
                                    + (au_val * np.power(yd_val, 2) * yu_val)
                                    + (ab_val * np.power(yt_val, 2) * yb_val)
                                    + (as_val * np.power(yc_val, 2) * ys_val)
                                    + (ad_val * np.power(yu_val, 2) * yd_val)))# end trace
                           - (18 * np.power(yt_val, 2)# Tr(au*Yu)
                              * ((at_val * yt_val)
                                 + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           - (np.power(yb_val, 2)# Tr(6ad*Yd + 2ae*Ye)
                              * ((6
                                  * ((ab_val * yb_val) + (as_val * ys_val)
                                     + (ad_val * yd_val)))
                                 + (2
                                    * ((atau_val * ytau_val)
                                       + (amu_val * ymu_val)
                                       + (ae_val * ye_val)))))# end trace
                           - (12 * yt_val * at_val# Tr(Yu^2)
                              * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                 + np.power(yu_val, 2)))# end trace
                           - (yb_val * ab_val# Tr(6Yd^2 + 2Ye^2)
                              * ((6 * (np.power(yb_val, 2)
                                       + np.power(ys_val, 2)
                                       + np.power(yd_val, 2)))
                                 + (2 * (np.power(ytau_val, 2)
                                         + np.power(ymu_val, 2)
                                         + np.power(ye_val, 2)))))# end trace
                           - (14 * np.power(yt_val, 3) * at_val)
                           - (8 * np.power(yb_val, 3) * ab_val)
                           - (2 * np.power(yb_val, 2) * yt_val * at_val)
                           - (4 * yb_val * ab_val * np.power(yt_val, 2))
                           + (((32 * np.power(g3_val, 2))
                               + ((8 / 5) * np.power(g1_val, 2)))# Tr(au*Yu)
                              * ((at_val * yt_val) + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           + (((6 * np.power(g2_val, 2))
                               + ((6 / 5) * np.power(g1_val, 2)))
                              * yt_val * at_val)
                           + ((4 / 5) * np.power(g1_val, 2) * yb_val * ab_val)
                           - (((32 * np.power(g3_val, 2) * M3_val)
                               + ((8 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yu^2)
                              * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                 + np.power(yu_val, 2)))# end trace
                           - (((12 * np.power(g2_val, 2) * M2_val)
                               + ((4 / 5) * np.power(g1_val, 2) * M1_val))
                              * np.power(yt_val, 2))
                           - ((4 / 5) * np.power(g1_val, 2) * M1_val
                              * np.power(yb_val, 2))
                           + ((64 / 9) * np.power(g3_val, 4) * M3_val)
                           - (16 * np.power(g3_val, 2) * np.power(g2_val, 2)
                              * (M3_val + M2_val))
                           - ((272 / 45) * np.power(g3_val, 2)
                              * np.power(g1_val, 2)
                              * (M3_val + M1_val))
                           - (30 * np.power(g2_val, 4) * M2_val)
                           - (2 * np.power(g2_val, 2) * np.power(g1_val, 2)
                              * (M2_val + M1_val))
                           - ((5486 / 225) * np.power(g1_val, 4) * M1_val))))

        dac_dt_2l = ((ac_val# Tr(3Yu^4 + (Yu^2*Yd^2))
                      * (((-3) * ((3 * (np.power(yt_val, 4)
                                        + np.power(yc_val, 4)
                                        + np.power(yu_val, 4)))
                                  + ((np.power(yt_val, 2) * np.power(yb_val,2))
                                     + (np.power(yc_val, 2)
                                        * np.power(ys_val, 2))
                                     + (np.power(yu_val, 2)
                                        * np.power(yd_val, 2)))))# end trace
                         - (np.power(ys_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2)
                                     + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (np.power(ytau_val, 2)
                                  + np.power(ymu_val, 2)
                                  + np.power(ye_val, 2))))# end trace
                         - (15 * np.power(yc_val, 2)# Tr(Yu^2)
                            * (np.power(yt_val, 2)
                               + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         - (6 * np.power(yc_val, 4))
                         - (2 * np.power(ys_val, 4))
                         - (4 * np.power(ys_val, 2) * np.power(yc_val, 2))
                         + (((16 * np.power(g3_val, 2))
                             + ((4 / 5) * np.power(g1_val, 2)))# Tr(Yu^2)
                            * (np.power(yt_val, 2)
                               + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         + (12 * np.power(g2_val, 2)
                            * np.power(yc_val, 2))
                         + ((2 / 5) * np.power(g1_val, 2)
                            * np.power(ys_val, 2))
                         - ((16 / 9) * np.power(g3_val, 4))
                         + (8 * np.power(g3_val, 2)
                            * np.power(g2_val, 2))
                         + ((136 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2))
                         + ((15 / 2) * np.power(g2_val, 4))
                         + (np.power(g2_val, 2) * np.power(g1_val, 2))
                         + ((2743 / 450) * np.power(g1_val, 4))))
                     + (yc_val# Tr(6au*Yu^3 + au*Yd^2*Yu + ad*Yu^2*Yd)
                        * (((-6) * ((6 * ((at_val * np.power(yt_val, 3))
                                          + (ac_val * np.power(yc_val, 3))
                                          + (au_val * np.power(yu_val, 3))))
                                    + (at_val * np.power(yb_val, 2) * yt_val)
                                    + (ac_val * np.power(ys_val, 2) * yc_val)
                                    + (au_val * np.power(yd_val, 2) * yu_val)
                                    + (ab_val * np.power(yt_val, 2) * yb_val)
                                    + (as_val * np.power(yc_val, 2) * ys_val)
                                    + (ad_val * np.power(yu_val, 2) * yd_val)))# end trace
                           - (18 * np.power(yc_val, 2)# Tr(au*Yu)
                              * ((at_val * yt_val)
                                 + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           - (np.power(ys_val, 2)# Tr(6ad*Yd + 2ae*Ye)
                              * ((6
                                  * ((ab_val * yb_val) + (as_val * ys_val)
                                     + (ad_val * yd_val)))
                                 + (2
                                    * ((atau_val * ytau_val)
                                       + (amu_val * ymu_val)
                                       + (ae_val * ye_val)))))# end trace
                           - (12 * yc_val * ac_val# Tr(Yu^2)
                              * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                 + np.power(yu_val, 2)))# end trace
                           - (ys_val * as_val# Tr(6Yd^2 + 2Ye^2)
                              * ((6 * (np.power(yb_val, 2)
                                       + np.power(ys_val, 2)
                                       + np.power(yd_val, 2)))
                                 + (2 * (np.power(ytau_val, 2)
                                         + np.power(ymu_val, 2)
                                         + np.power(ye_val, 2)))))# end trace
                           - (14 * np.power(yc_val, 3) * ac_val)
                           - (8 * np.power(ys_val, 3) * as_val)
                           - (2 * np.power(ys_val, 2) * yc_val * ac_val)
                           - (4 * ys_val * as_val * np.power(yc_val, 2))
                           + (((32 * np.power(g3_val, 2))
                               + ((8 / 5) * np.power(g1_val, 2)))# Tr(au*Yu)
                              * ((at_val * yt_val) + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           + (((6 * np.power(g2_val, 2))
                               + ((6 / 5) * np.power(g1_val, 2)))
                              * yc_val * ac_val)
                           + ((4 / 5) * np.power(g1_val, 2) * ys_val * as_val)
                           - (((32 * np.power(g3_val, 2) * M3_val)
                               + ((8 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yu^2)
                              * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                 + np.power(yu_val, 2)))# end trace
                           - (((12 * np.power(g2_val, 2) * M2_val)
                               + ((4 / 5) * np.power(g1_val, 2) * M1_val))
                              * np.power(yc_val, 2))
                           - ((4 / 5) * np.power(g1_val, 2) * M1_val
                              * np.power(ys_val, 2))
                           + ((64 / 9) * np.power(g3_val, 4) * M3_val)
                           - (16 * np.power(g3_val, 2) * np.power(g2_val, 2)
                              * (M3_val + M2_val))
                           - ((272 / 45) * np.power(g3_val, 2)
                              * np.power(g1_val, 2)
                              * (M3_val + M1_val))
                           - (30 * np.power(g2_val, 4) * M2_val)
                           - (2 * np.power(g2_val, 2) * np.power(g1_val, 2)
                              * (M2_val + M1_val))
                           - ((5486 / 225) * np.power(g1_val, 4) * M1_val))))

        dau_dt_2l = ((au_val# Tr(3Yu^4 + (Yu^2*Yd^2))
                      * (((-3) * ((3 * (np.power(yt_val, 4)
                                        + np.power(yc_val, 4)
                                        + np.power(yu_val, 4)))
                                  + ((np.power(yt_val, 2) * np.power(yb_val,2))
                                     + (np.power(yc_val, 2)
                                        * np.power(ys_val, 2))
                                     + (np.power(yu_val, 2)
                                        * np.power(yd_val, 2)))))# end trace
                         - (np.power(yd_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2)
                                     + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (np.power(ytau_val, 2)
                                  + np.power(ymu_val, 2)
                                  + np.power(ye_val, 2))))# end trace
                         - (15 * np.power(yu_val, 2)# Tr(Yu^2)
                            * (np.power(yt_val, 2)
                               + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         - (6 * np.power(yu_val, 4))
                         - (2 * np.power(yd_val, 4))
                         - (4 * np.power(yd_val, 2) * np.power(yu_val, 2))
                         + (((16 * np.power(g3_val, 2))
                             + ((4 / 5) * np.power(g1_val, 2)))# Tr(Yu^2)
                            * (np.power(yt_val, 2)
                               + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         + (12 * np.power(g2_val, 2)
                            * np.power(yu_val, 2))
                         + ((2 / 5) * np.power(g1_val, 2)
                            * np.power(yd_val, 2))
                         - ((16 / 9) * np.power(g3_val, 4))
                         + (8 * np.power(g3_val, 2)
                            * np.power(g2_val, 2))
                         + ((136 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2))
                         + ((15 / 2) * np.power(g2_val, 4))
                         + (np.power(g2_val, 2) * np.power(g1_val, 2))
                         + ((2743 / 450) * np.power(g1_val, 4))))
                     + (yu_val# Tr(6au*Yu^3 + au*Yd^2*Yu + ad*Yu^2*Yd)
                        * (((-6) * ((6 * ((at_val * np.power(yt_val, 3))
                                          + (ac_val * np.power(yc_val, 3))
                                          + (au_val * np.power(yu_val, 3))))
                                    + (at_val * np.power(yb_val, 2) * yt_val)
                                    + (ac_val * np.power(ys_val, 2) * yc_val)
                                    + (au_val * np.power(yd_val, 2) * yu_val)
                                    + (ab_val * np.power(yt_val, 2) * yb_val)
                                    + (as_val * np.power(yc_val, 2) * ys_val)
                                    + (ad_val * np.power(yu_val, 2) * yd_val)))# end trace
                           - (18 * np.power(yu_val, 2)# Tr(au*Yu)
                              * ((at_val * yt_val)
                                 + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           - (np.power(yd_val, 2)# Tr(6ad*Yd + 2ae*Ye)
                              * ((6
                                  * ((ab_val * yb_val) + (as_val * ys_val)
                                     + (ad_val * yd_val)))
                                 + (2
                                    * ((atau_val * ytau_val)
                                       + (amu_val * ymu_val)
                                       + (ae_val * ye_val)))))# end trace
                           - (12 * yu_val * au_val# Tr(Yu^2)
                              * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                 + np.power(yu_val, 2)))# end trace
                           - (yd_val * ad_val# Tr(6Yd^2 + 2Ye^2)
                              * ((6 * (np.power(yb_val, 2)
                                       + np.power(ys_val, 2)
                                       + np.power(yd_val, 2)))
                                 + (2 * (np.power(ytau_val, 2)
                                         + np.power(ymu_val, 2)
                                         + np.power(ye_val, 2)))))# end trace
                           - (14 * np.power(yu_val, 3) * au_val)
                           - (8 * np.power(yd_val, 3) * ad_val)
                           - (2 * np.power(yd_val, 2) * yu_val * au_val)
                           - (4 * yd_val * ad_val * np.power(yu_val, 2))
                           + (((32 * np.power(g3_val, 2))
                               + ((8 / 5) * np.power(g1_val, 2)))# Tr(au*Yu)
                              * ((at_val * yt_val) + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           + (((6 * np.power(g2_val, 2))
                               + ((6 / 5) * np.power(g1_val, 2)))
                              * yu_val * au_val)
                           + ((4 / 5) * np.power(g1_val, 2) * yd_val * ad_val)
                           - (((32 * np.power(g3_val, 2) * M3_val)
                               + ((8 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yu^2)
                              * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                 + np.power(yu_val, 2)))# end trace
                           - (((12 * np.power(g2_val, 2) * M2_val)
                               + ((4 / 5) * np.power(g1_val, 2) * M1_val))
                              * np.power(yu_val, 2))
                           - ((4 / 5) * np.power(g1_val, 2) * M1_val
                              * np.power(yd_val, 2))
                           + ((64 / 9) * np.power(g3_val, 4) * M3_val)
                           - (16 * np.power(g3_val, 2) * np.power(g2_val, 2)
                              * (M3_val + M2_val))
                           - ((272 / 45) * np.power(g3_val, 2)
                              * np.power(g1_val, 2)
                              * (M3_val + M1_val))
                           - (30 * np.power(g2_val, 4) * M2_val)
                           - (2 * np.power(g2_val, 2) * np.power(g1_val, 2)
                              * (M2_val + M1_val))
                           - ((5486 / 225) * np.power(g1_val, 4) * M1_val))))

        dab_dt_2l = ((ab_val# Tr(3Yd^4 + Yu^2*Yd^2 + Ye^4)
                      * (((-3) * ((3 * (np.power(yb_val, 4)
                                        + np.power(ys_val, 4)
                                        + np.power(yd_val, 4)))
                                  + ((np.power(yt_val, 2) * np.power(yb_val,2))
                                     + (np.power(yc_val, 2)
                                        * np.power(ys_val, 2))
                                     + (np.power(yu_val, 2)
                                        * np.power(yd_val, 2)))
                                  + np.power(ytau_val, 4)
                                  + np.power(ymu_val, 4)
                                  + np.power(ye_val, 4)))# end trace
                         - (3 * np.power(yt_val, 2)# Tr(Yu^2)
                            * (np.power(yt_val, 2) + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         - (5 * np.power(yb_val, 2)# Tr(3Yd^2+Ye^2)
                            * ((3 * (np.power(yb_val, 2)
                                     + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (np.power(ytau_val, 2)
                                  + np.power(ymu_val, 2)
                                  + np.power(ye_val, 2))))# end trace
                         - (6 * np.power(yb_val, 4))
                         - (2 * np.power(yt_val, 4))
                         - (4 * np.power(yb_val, 2) * np.power(yt_val, 2))
                         + (((16 * np.power(g3_val, 2))
                             - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                            * (np.power(yb_val, 2)
                               + np.power(ys_val, 2)
                               + np.power(yd_val, 2)))# end trace
                         + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                            * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         + ((4 / 5) * np.power(g1_val, 2)
                            * np.power(yt_val, 2))
                         + (((12 * np.power(g2_val, 2))
                             + ((6 / 5) * np.power(g1_val, 2)))
                            * np.power(yb_val, 2))
                         - ((16 / 9) * np.power(g3_val, 4))
                         + (8 * np.power(g3_val, 2)
                            * np.power(g2_val, 2))
                         + ((8 / 9) * np.power(g3_val, 2)
                            * np.power(g1_val, 2))
                         + ((15 / 2) * np.power(g2_val, 4))
                         + (np.power(g2_val, 2) * np.power(g1_val, 2))
                         + ((287 / 90) * np.power(g1_val, 4))))
                     + (yb_val# Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
                        * (((-6) * ((6 * ((ab_val * np.power(yb_val, 3))
                                          + (as_val * np.power(ys_val, 3))
                                          + (ad_val * np.power(yd_val, 3))))
                                    + (at_val * np.power(yb_val, 2) * yt_val)
                                    + (ac_val * np.power(ys_val, 2) * yc_val)
                                    + (au_val * np.power(yd_val, 2) * yu_val)
                                    + (ab_val * np.power(yt_val, 2) * yb_val)
                                    + (as_val * np.power(yc_val, 2) * ys_val)
                                    + (ad_val * np.power(yu_val, 2) * yd_val)
                                    + (2 * ((atau_val * np.power(ytau_val, 3))
                                            + (amu_val * np.power(ymu_val, 3))
                                            + (ae_val * np.power(ye_val, 3)))
                                       )))# end trace
                           - (6 * np.power(yt_val, 2)# Tr(au*Yu)
                              * ((at_val * yt_val)
                                 + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           - (6 * np.power(yb_val, 2)# Tr(3ad*Yd + ae*Ye)
                              * ((3
                                  * ((ab_val * yb_val) + (as_val * ys_val)
                                     + (ad_val * yd_val)))
                                 + ((atau_val * ytau_val) + (amu_val * ymu_val)
                                    + (ae_val * ye_val))))# end trace
                           - (6 * yt_val * at_val# Tr(Yu^2)
                              * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                 + np.power(yu_val, 2)))# end trace
                           - (4 * yb_val * ab_val# Tr(3Yd^2 + Ye^2)
                              * ((3 * (np.power(yb_val, 2)
                                       + np.power(ys_val, 2)
                                       + np.power(yd_val, 2)))
                                 + ((np.power(ytau_val, 2)
                                     + np.power(ymu_val, 2)
                                     + np.power(ye_val, 2)))))# end trace
                           - (14 * np.power(yb_val, 3) * ab_val)
                           - (8 * np.power(yt_val, 3) * at_val)
                           - (4 * np.power(yb_val, 2) * yt_val * at_val)
                           - (2 * yb_val * ab_val * np.power(yt_val, 2))
                           + (((32 * np.power(g3_val, 2))
                               - ((4 / 5) * np.power(g1_val, 2)))# Tr(ad*Yd)
                              * ((ab_val * yb_val) + (as_val * ys_val)
                                 + (ad_val * yd_val)))# end trace
                           + ((12 / 5) * np.power(g1_val, 2)# Tr(ae*Ye)
                              * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                 + (ae_val * ye_val)))# end trace
                           + ((8 / 5) * np.power(g1_val, 2) * yt_val * at_val)
                           + (((6 * np.power(g2_val, 2))
                               + ((6 / 5) * np.power(g1_val, 2)))
                              * yb_val * ab_val)
                           - (((32 * np.power(g3_val, 2) * M3_val)
                               - ((4 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yd^2)
                              * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                 + np.power(yd_val, 2)))# end trace
                           - ((12 / 5) * np.power(g1_val, 2) * M1_val# Tr(Ye^2)
                              * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2)))# end trace
                           - (((12 * np.power(g2_val, 2) * M2_val)
                               + ((8 / 5) * np.power(g1_val, 2) * M1_val))
                              * np.power(yb_val, 2))
                           - ((8 / 5) * np.power(g1_val, 2) * M1_val
                              * np.power(yt_val, 2))
                           + ((64 / 9) * np.power(g3_val, 4) * M3_val)
                           - (16 * np.power(g3_val, 2) * np.power(g2_val, 2)
                              * (M3_val + M2_val))
                           - ((16 / 9) * np.power(g3_val, 2)
                              * np.power(g1_val, 2)
                              * (M3_val + M1_val))
                           - (30 * np.power(g2_val, 4) * M2_val)
                           - (2 * np.power(g2_val, 2) * np.power(g1_val, 2)
                              * (M2_val + M1_val))
                           - ((574 / 45) * np.power(g1_val, 4) * M1_val))))

        das_dt_2l = ((as_val# Tr(3Yd^4 + Yu^2*Yd^2 + Ye^4)
                      * (((-3) * ((3 * (np.power(yb_val, 4)
                                        + np.power(ys_val, 4)
                                        + np.power(yd_val, 4)))
                                  + ((np.power(yt_val, 2) * np.power(yb_val,2))
                                     + (np.power(yc_val, 2)
                                        * np.power(ys_val, 2))
                                     + (np.power(yu_val, 2)
                                        * np.power(yd_val, 2)))
                                  + np.power(ytau_val, 4)
                                  + np.power(ymu_val, 4)
                                  + np.power(ye_val, 4)))# end trace
                         - (3 * np.power(yc_val, 2)# Tr(Yu^2)
                            * (np.power(yt_val, 2) + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         - (5 * np.power(ys_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2)
                                     + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (np.power(ytau_val, 2)
                                  + np.power(ymu_val, 2)
                                  + np.power(ye_val, 2))))# end trace
                         - (6 * np.power(ys_val, 4))
                         - (2 * np.power(yc_val, 4))
                         - (4 * np.power(ys_val, 2) * np.power(yc_val, 2))
                         + (((16 * np.power(g3_val, 2))
                             - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                            * (np.power(yb_val, 2)
                               + np.power(ys_val, 2)
                               + np.power(yd_val, 2)))# end trace
                         + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                            * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         + ((4 / 5) * np.power(g1_val, 2)
                            * np.power(yc_val, 2))
                         + (((12 * np.power(g2_val, 2))
                             + ((6 / 5) * np.power(g1_val, 2)))
                            * np.power(ys_val, 2))
                         - ((16 / 9) * np.power(g3_val, 4))
                         + (8 * np.power(g3_val, 2)
                            * np.power(g2_val, 2))
                         + ((8 / 9) * np.power(g3_val, 2)
                            * np.power(g1_val, 2))
                         + ((15 / 2) * np.power(g2_val, 4))
                         + (np.power(g2_val, 2) * np.power(g1_val, 2))
                         + ((287 / 90) * np.power(g1_val, 4))))
                     + (ys_val# Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
                        * (((-6) * ((6 * ((ab_val * np.power(yb_val, 3))
                                          + (as_val * np.power(ys_val, 3))
                                          + (ad_val * np.power(yd_val, 3))))
                                    + (at_val * np.power(yb_val, 2) * yt_val)
                                    + (ac_val * np.power(ys_val, 2) * yc_val)
                                    + (au_val * np.power(yd_val, 2) * yu_val)
                                    + (ab_val * np.power(yt_val, 2) * yb_val)
                                    + (as_val * np.power(yc_val, 2) * ys_val)
                                    + (ad_val * np.power(yu_val, 2) * yd_val)
                                    + (2 * ((atau_val * np.power(ytau_val, 3))
                                            + (amu_val * np.power(ymu_val, 3))
                                            + (ae_val * np.power(ye_val, 3)))
                                       )))# end trace
                           - (6 * np.power(yc_val, 2)# Tr(au*Yu)
                              * ((at_val * yt_val)
                                 + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           - (6 * np.power(ys_val, 2)# Tr(3ad*Yd + ae*Ye)
                              * ((3
                                  * ((ab_val * yb_val) + (as_val * ys_val)
                                     + (ad_val * yd_val)))
                                 + ((atau_val * ytau_val) + (amu_val * ymu_val)
                                    + (ae_val * ye_val))))# end trace
                           - (6 * yc_val * ac_val# Tr(Yu^2)
                              * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                 + np.power(yu_val, 2)))# end trace
                           - (4 * ys_val * as_val# Tr(3Yd^2 + Ye^2)
                              * ((3 * (np.power(yb_val, 2)
                                       + np.power(ys_val, 2)
                                       + np.power(yd_val, 2)))
                                 + ((np.power(ytau_val, 2)
                                     + np.power(ymu_val, 2)
                                     + np.power(ye_val, 2)))))# end trace
                           - (14 * np.power(ys_val, 3) * as_val)
                           - (8 * np.power(yc_val, 3) * ac_val)
                           - (4 * np.power(ys_val, 2) * yc_val * ac_val)
                           - (2 * ys_val * as_val * np.power(yc_val, 2))
                           + (((32 * np.power(g3_val, 2))
                               - ((4 / 5) * np.power(g1_val, 2)))# Tr(ad*Yd)
                              * ((ab_val * yb_val) + (as_val * ys_val)
                                 + (ad_val * yd_val)))# end trace
                           + ((12 / 5) * np.power(g1_val, 2)# Tr(ae*Ye)
                              * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                 + (ae_val * ye_val)))# end trace
                           + ((8 / 5) * np.power(g1_val, 2) * yc_val * ac_val)
                           + (((6 * np.power(g2_val, 2))
                               + ((6 / 5) * np.power(g1_val, 2)))
                              * ys_val * as_val)
                           - (((32 * np.power(g3_val, 2) * M3_val)
                               - ((4 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yd^2)
                              * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                 + np.power(yd_val, 2)))# end trace
                           - ((12 / 5) * np.power(g1_val, 2) * M1_val# Tr(Ye^2)
                              * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2)))# end trace
                           - (((12 * np.power(g2_val, 2) * M2_val)
                               + ((8 / 5) * np.power(g1_val, 2) * M1_val))
                              * np.power(ys_val, 2))
                           - ((8 / 5) * np.power(g1_val, 2) * M1_val
                              * np.power(yc_val, 2))
                           + ((64 / 9) * np.power(g3_val, 4) * M3_val)
                           - (16 * np.power(g3_val, 2) * np.power(g2_val, 2)
                              * (M3_val + M2_val))
                           - ((16 / 9) * np.power(g3_val, 2)
                              * np.power(g1_val, 2)
                              * (M3_val + M1_val))
                           - (30 * np.power(g2_val, 4) * M2_val)
                           - (2 * np.power(g2_val, 2) * np.power(g1_val, 2)
                              * (M2_val + M1_val))
                           - ((574 / 45) * np.power(g1_val, 4) * M1_val))))

        dad_dt_2l = ((ad_val# Tr(3Yd^4 + Yu^2*Yd^2 + Ye^4)
                      * (((-3) * ((3 * (np.power(yb_val, 4)
                                        + np.power(ys_val, 4)
                                        + np.power(yd_val, 4)))
                                  + ((np.power(yt_val, 2) * np.power(yb_val,2))
                                     + (np.power(yc_val, 2)
                                        * np.power(ys_val, 2))
                                     + (np.power(yu_val, 2)
                                        * np.power(yd_val, 2)))
                                  + np.power(ytau_val, 4)
                                  + np.power(ymu_val, 4)
                                  + np.power(ye_val, 4)))# end trace
                         - (3 * np.power(yu_val, 2)# Tr(Yu^2)
                            * (np.power(yt_val, 2) + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         - (5 * np.power(yd_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2)
                                     + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (np.power(ytau_val, 2)
                                  + np.power(ymu_val, 2)
                                  + np.power(ye_val, 2))))# end trace
                         - (6 * np.power(yd_val, 4))
                         - (2 * np.power(yu_val, 4))
                         - (4 * np.power(yd_val, 2) * np.power(yu_val, 2))
                         + (((16 * np.power(g3_val, 2))
                             - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                            * (np.power(yb_val, 2)
                               + np.power(ys_val, 2)
                               + np.power(yd_val, 2)))# end trace
                         + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                            * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         + ((4 / 5) * np.power(g1_val, 2)
                            * np.power(yu_val, 2))
                         + (((12 * np.power(g2_val, 2))
                             + ((6 / 5) * np.power(g1_val, 2)))
                            * np.power(yd_val, 2))
                         - ((16 / 9) * np.power(g3_val, 4))
                         + (8 * np.power(g3_val, 2)
                            * np.power(g2_val, 2))
                         + ((8 / 9) * np.power(g3_val, 2)
                            * np.power(g1_val, 2))
                         + ((15 / 2) * np.power(g2_val, 4))
                         + (np.power(g2_val, 2) * np.power(g1_val, 2))
                         + ((287 / 90) * np.power(g1_val, 4))))
                     + (yd_val# Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
                        * (((-6) * ((6 * ((ab_val * np.power(yb_val, 3))
                                          + (as_val * np.power(ys_val, 3))
                                          + (ad_val * np.power(yd_val, 3))))
                                    + (at_val * np.power(yb_val, 2) * yt_val)
                                    + (ac_val * np.power(ys_val, 2) * yc_val)
                                    + (au_val * np.power(yd_val, 2) * yu_val)
                                    + (ab_val * np.power(yt_val, 2) * yb_val)
                                    + (as_val * np.power(yc_val, 2) * ys_val)
                                    + (ad_val * np.power(yu_val, 2) * yd_val)
                                    + (2 * ((atau_val * np.power(ytau_val, 3))
                                            + (amu_val * np.power(ymu_val, 3))
                                            + (ae_val * np.power(ye_val, 3)))
                                       )))# end trace
                           - (6 * np.power(yu_val, 2)# Tr(au*Yu)
                              * ((at_val * yt_val)
                                 + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           - (6 * np.power(yd_val, 2)# Tr(3ad*Yd + ae*Ye)
                              * ((3
                                  * ((ab_val * yb_val) + (as_val * ys_val)
                                     + (ad_val * yd_val)))
                                 + ((atau_val * ytau_val) + (amu_val * ymu_val)
                                    + (ae_val * ye_val))))# end trace
                           - (6 * yu_val * au_val# Tr(Yu^2)
                              * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                 + np.power(yu_val, 2)))# end trace
                           - (4 * yd_val * ad_val# Tr(3Yd^2 + Ye^2)
                              * ((3 * (np.power(yb_val, 2)
                                       + np.power(ys_val, 2)
                                       + np.power(yd_val, 2)))
                                 + ((np.power(ytau_val, 2)
                                     + np.power(ymu_val, 2)
                                     + np.power(ye_val, 2)))))# end trace
                           - (14 * np.power(yd_val, 3) * ad_val)
                           - (8 * np.power(yu_val, 3) * au_val)
                           - (4 * np.power(yd_val, 2) * yu_val * au_val)
                           - (2 * yd_val * ad_val * np.power(yu_val, 2))
                           + (((32 * np.power(g3_val, 2))
                               - ((4 / 5) * np.power(g1_val, 2)))# Tr(ad*Yd)
                              * ((ab_val * yb_val) + (as_val * ys_val)
                                 + (ad_val * yd_val)))# end trace
                           + ((12 / 5) * np.power(g1_val, 2)# Tr(ae*Ye)
                              * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                 + (ae_val * ye_val)))# end trace
                           + ((8 / 5) * np.power(g1_val, 2) * yu_val * au_val)
                           + (((6 * np.power(g2_val, 2))
                               + ((6 / 5) * np.power(g1_val, 2)))
                              * yd_val * ad_val)
                           - (((32 * np.power(g3_val, 2) * M3_val)
                               - ((4 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yd^2)
                              * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                 + np.power(yd_val, 2)))# end trace
                           - ((12 / 5) * np.power(g1_val, 2) * M1_val# Tr(Ye^2)
                              * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2)))# end trace
                           - (((12 * np.power(g2_val, 2) * M2_val)
                               + ((8 / 5) * np.power(g1_val, 2) * M1_val))
                              * np.power(yd_val, 2))
                           - ((8 / 5) * np.power(g1_val, 2) * M1_val
                              * np.power(yu_val, 2))
                           + ((64 / 9) * np.power(g3_val, 4) * M3_val)
                           - (16 * np.power(g3_val, 2) * np.power(g2_val, 2)
                              * (M3_val + M2_val))
                           - ((16 / 9) * np.power(g3_val, 2)
                              * np.power(g1_val, 2)
                              * (M3_val + M1_val))
                           - (30 * np.power(g2_val, 4) * M2_val)
                           - (2 * np.power(g2_val, 2) * np.power(g1_val, 2)
                              * (M2_val + M1_val))
                           - ((574 / 45) * np.power(g1_val, 4) * M1_val))))

        datau_dt_2l = ((atau_val# Tr(3Yd^4 + Yu^2*Yd^2 + Ye^4)
                        * (((-3) * ((3 * (np.power(yb_val, 4)
                                          + np.power(ys_val, 4)
                                          + np.power(yd_val, 4)))
                                    + ((np.power(yt_val, 2)
                                        * np.power(yb_val,2))
                                       + (np.power(yc_val, 2)
                                          * np.power(ys_val, 2))
                                       + (np.power(yu_val, 2)
                                          * np.power(yd_val, 2)))
                                    + np.power(ytau_val, 4)
                                    + np.power(ymu_val, 4)
                                    + np.power(ye_val, 4)))# end trace
                           - (5 * np.power(ytau_val, 2)# Tr(3Yd^2 + Ye^2)
                              * ((3 * (np.power(yb_val, 2)
                                       + np.power(ys_val, 2)
                                       + np.power(yd_val, 2)))
                                 + (np.power(ytau_val, 2)
                                    + np.power(ymu_val, 2)
                                    + np.power(ye_val, 2))))# end trace
                           - (6 * np.power(ytau_val, 4))
                           + (((16 * np.power(g3_val, 2))
                               - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                              * (np.power(yb_val, 2)
                                 + np.power(ys_val, 2)
                                 + np.power(yd_val, 2)))# end trace
                           + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                              * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2)))# end trace
                           + (((12 * np.power(g2_val, 2))
                               - ((6 / 5) * np.power(g1_val, 2)))
                              * np.power(ytau_val, 2))
                           + ((15 / 2) * np.power(g2_val, 4))
                           + ((9 / 5) * np.power(g2_val, 2)
                              * np.power(g1_val, 2))
                           + ((27 / 2) * np.power(g1_val, 4))))
                       + (ytau_val# Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
                          * (((-6) * ((6 * ((ab_val * np.power(yb_val, 3))
                                            + (as_val * np.power(ys_val, 3))
                                            + (ad_val * np.power(yd_val, 3))))
                                      + (at_val * np.power(yb_val, 2) * yt_val)
                                      + (ac_val * np.power(ys_val, 2) * yc_val)
                                      + (au_val * np.power(yd_val, 2) * yu_val)
                                      + (ab_val * np.power(yt_val, 2) * yb_val)
                                      + (as_val * np.power(yc_val, 2) * ys_val)
                                      + (ad_val * np.power(yu_val, 2) * yd_val)
                                      + (2 * ((atau_val
                                               * np.power(ytau_val, 3))
                                              + (amu_val
                                                 * np.power(ymu_val, 3))
                                              + (ae_val
                                                 * np.power(ye_val, 3)))
                                         )))# end trace
                             - (4 * ytau_val * atau_val# Tr(3Yd^2 + Ye^2)
                                * ((3 * (np.power(yb_val, 2)
                                         + np.power(ys_val, 2)
                                         + np.power(yd_val, 2)))
                                   + ((np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                             - (6 * np.power(ytau_val, 2)# Tr(3ad*Yd + ae*Ye)
                                * ((3 * ((ab_val * yb_val) + (as_val * ys_val)
                                         + (ad_val * yd_val)))
                                   + (atau_val * ytau_val)
                                   + (amu_val * ymu_val)
                                   + (ae_val * ye_val)))# end trace
                             - (14 * np.power(ytau_val, 3) * atau_val)
                             + (((32 * np.power(g3_val, 2))
                                 - ((4 / 5) * np.power(g1_val, 2)))# Tr(ad*Yd)
                                * ((ab_val * yb_val) + (as_val * ys_val)
                                   + (ad_val * yd_val)))# end trace
                             + ((12 / 5) * np.power(g1_val, 2)# Tr(ae*Ye)
                                * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                   + (ae_val * ye_val)))# end trace
                             + (((6 * np.power(g2_val, 2))
                                 + ((6 / 5) * np.power(g1_val, 2)))
                                * ytau_val * atau_val)
                             - (((32 * np.power(g3_val, 2) * M3_val)
                                 - ((4 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yd^2)
                                * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                   + np.power(yd_val, 2)))# end trace
                             - ((12 / 5) * np.power(g1_val, 2) * M1_val# Tr(Ye^2)
                                * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                   + np.power(ye_val, 2)))# end trace
                             - (12 * np.power(g2_val, 2) * M2_val
                                * np.power(ytau_val, 2))
                             - (30 * np.power(g2_val, 4) * M2_val)
                             - ((18 / 5) * np.power(g2_val, 2)
                                * np.power(g1_val, 2)
                                * (M1_val + M2_val))
                             - (54 * np.power(g1_val, 4) * M1_val))))

        damu_dt_2l = ((amu_val# Tr(3Yd^4 + Yu^2*Yd^2 + Ye^4)
                       * (((-3) * ((3 * (np.power(yb_val, 4)
                                         + np.power(ys_val, 4)
                                         + np.power(yd_val, 4)))
                                   + ((np.power(yt_val, 2)
                                       * np.power(yb_val,2))
                                      + (np.power(yc_val, 2)
                                         * np.power(ys_val, 2))
                                      + (np.power(yu_val, 2)
                                         * np.power(yd_val, 2)))
                                   + np.power(ytau_val, 4)
                                   + np.power(ymu_val, 4)
                                   + np.power(ye_val, 4)))# end trace
                          - (5 * np.power(ymu_val, 2)# Tr(3Yd^2 + Ye^2)
                             * ((3 * (np.power(yb_val, 2)
                                      + np.power(ys_val, 2)
                                      + np.power(yd_val, 2)))
                                + (np.power(ytau_val, 2)
                                   + np.power(ymu_val, 2)
                                   + np.power(ye_val, 2))))# end trace
                          - (6 * np.power(ymu_val, 4))
                          + (((16 * np.power(g3_val, 2))
                              - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                             * (np.power(yb_val, 2)
                                + np.power(ys_val, 2)
                                + np.power(yd_val, 2)))# end trace
                          + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                             * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                + np.power(ye_val, 2)))# end trace
                          + (((12 * np.power(g2_val, 2))
                              - ((6 / 5) * np.power(g1_val, 2)))
                             * np.power(ymu_val, 2))
                          + ((15 / 2) * np.power(g2_val, 4))
                          + ((9 / 5) * np.power(g2_val, 2)
                             * np.power(g1_val, 2))
                          + ((27 / 2) * np.power(g1_val, 4))))
                      + (ymu_val# Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
                         * (((-6) * ((6 * ((ab_val * np.power(yb_val, 3))
                                           + (as_val * np.power(ys_val, 3))
                                           + (ad_val * np.power(yd_val, 3))))
                                     + (at_val * np.power(yb_val, 2) * yt_val)
                                     + (ac_val * np.power(ys_val, 2) * yc_val)
                                     + (au_val * np.power(yd_val, 2) * yu_val)
                                     + (ab_val * np.power(yt_val, 2) * yb_val)
                                     + (as_val * np.power(yc_val, 2) * ys_val)
                                     + (ad_val * np.power(yu_val, 2) * yd_val)
                                     + (2 * ((atau_val * np.power(ytau_val, 3))
                                             + (amu_val * np.power(ymu_val, 3))
                                             + (ae_val * np.power(ye_val, 3))))))# end trace
                            - (4 * ymu_val * amu_val# Tr(3Yd^2 + Ye^2)
                               * ((3 * (np.power(yb_val, 2)
                                        + np.power(ys_val, 2)
                                        + np.power(yd_val, 2)))
                                  + ((np.power(ytau_val, 2)
                                      + np.power(ymu_val, 2)
                                      + np.power(ye_val, 2)))))# end trace
                            - (6 * np.power(ymu_val, 2)# Tr(3ad*Yd + ae*Ye)
                               * ((3 * ((ab_val * yb_val) + (as_val * ys_val)
                                        + (ad_val * yd_val)))
                                  + (atau_val * ytau_val) + (amu_val * ymu_val)
                                  + (ae_val * ye_val)))# end trace
                            - (14 * np.power(ymu_val, 3) * amu_val)
                            + (((32 * np.power(g3_val, 2))
                                - ((4 / 5) * np.power(g1_val, 2)))# Tr(ad*Yd)
                               * ((ab_val * yb_val) + (as_val * ys_val)
                                  + (ad_val * yd_val)))# end trace
                            + ((12 / 5) * np.power(g1_val, 2)# Tr(ae*Ye)
                               * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                  + (ae_val * ye_val)))# end trace
                            + (((6 * np.power(g2_val, 2))
                                + ((6 / 5) * np.power(g1_val, 2)))
                               * ymu_val * amu_val)
                            - (((32 * np.power(g3_val, 2) * M3_val)
                                - ((4 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yd^2)
                               * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                  + np.power(yd_val, 2)))# end trace
                            - ((12 / 5) * np.power(g1_val, 2) * M1_val# Tr(Ye^2)
                               * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                  + np.power(ye_val, 2)))# end trace
                            - (12 * np.power(g2_val, 2) * M2_val
                               * np.power(ymu_val, 2))
                            - (30 * np.power(g2_val, 4) * M2_val)
                            - ((18 / 5) * np.power(g2_val, 2) * np.power(g1_val, 2)
                               * (M1_val + M2_val))
                            - (54 * np.power(g1_val, 4) * M1_val))))

        dae_dt_2l = ((ae_val# Tr(3Yd^4 + Yu^2*Yd^2 + Ye^4)
                      * (((-3) * ((3 * (np.power(yb_val, 4)
                                        + np.power(ys_val, 4)
                                        + np.power(yd_val, 4)))
                                  + ((np.power(yt_val, 2)
                                      * np.power(yb_val,2))
                                     + (np.power(yc_val, 2)
                                        * np.power(ys_val, 2))
                                     + (np.power(yu_val, 2)
                                        * np.power(yd_val, 2)))
                                  + np.power(ytau_val, 4)
                                  + np.power(ymu_val, 4)
                                  + np.power(ye_val, 4)))# end trace
                         - (5 * np.power(ye_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2)
                                     + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (np.power(ytau_val, 2)
                                  + np.power(ymu_val, 2)
                                  + np.power(ye_val, 2))))# end trace
                         - (6 * np.power(ye_val, 4))
                         + (((16 * np.power(g3_val, 2))
                             - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                            * (np.power(yb_val, 2)
                               + np.power(ys_val, 2)
                               + np.power(yd_val, 2)))# end trace
                         + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                            * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         + (((12 * np.power(g2_val, 2))
                             - ((6 / 5) * np.power(g1_val, 2)))
                            * np.power(ye_val, 2))
                         + ((15 / 2) * np.power(g2_val, 4))
                         + ((9 / 5) * np.power(g2_val, 2)
                            * np.power(g1_val, 2))
                         + ((27 / 2) * np.power(g1_val, 4))))
                     + (ye_val# Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
                        * (((-6) * ((6 * ((ab_val * np.power(yb_val, 3))
                                          + (as_val * np.power(ys_val, 3))
                                          + (ad_val * np.power(yd_val, 3))))
                                    + (at_val * np.power(yb_val, 2) * yt_val)
                                    + (ac_val * np.power(ys_val, 2) * yc_val)
                                    + (au_val * np.power(yd_val, 2) * yu_val)
                                    + (ab_val * np.power(yt_val, 2) * yb_val)
                                    + (as_val * np.power(yc_val, 2) * ys_val)
                                    + (ad_val * np.power(yu_val, 2) * yd_val)
                                    + (2 * ((atau_val * np.power(ytau_val, 3))
                                            + (amu_val * np.power(ymu_val, 3))
                                            + (ae_val * np.power(ye_val, 3))))))# end trace
                           - (4 * ye_val * ae_val# Tr(3Yd^2 + Ye^2)
                              * ((3 * (np.power(yb_val, 2)
                                       + np.power(ys_val, 2)
                                       + np.power(yd_val, 2)))
                                 + ((np.power(ytau_val, 2)
                                     + np.power(ymu_val, 2)
                                     + np.power(ye_val, 2)))))# end trace
                           - (6 * np.power(ye_val, 2)# Tr(3ad*Yd + ae*Ye)
                              * ((3 * ((ab_val * yb_val) + (as_val * ys_val)
                                       + (ad_val * yd_val)))
                                 + (atau_val * ytau_val) + (amu_val * ymu_val)
                                 + (ae_val * ye_val)))# end trace
                           - (14 * np.power(ye_val, 3) * ae_val)
                           + (((32 * np.power(g3_val, 2))
                               - ((4 / 5) * np.power(g1_val, 2)))# Tr(ad*Yd)
                              * ((ab_val * yb_val) + (as_val * ys_val)
                                 + (ad_val * yd_val)))# end trace
                           + ((12 / 5) * np.power(g1_val, 2)# Tr(ae*Ye)
                              * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                 + (ae_val * ye_val)))# end trace
                           + (((6 * np.power(g2_val, 2))
                               + ((6 / 5) * np.power(g1_val, 2)))
                              * ye_val * ae_val)
                           - (((32 * np.power(g3_val, 2) * M3_val)
                               - ((4 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yd^2)
                              * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                 + np.power(yd_val, 2)))# end trace
                           - ((12 / 5) * np.power(g1_val, 2) * M1_val# Tr(Ye^2)
                              * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2)))# end trace
                           - (12 * np.power(g2_val, 2) * M2_val
                              * np.power(ye_val, 2))
                           - (30 * np.power(g2_val, 4) * M2_val)
                           - ((18 / 5) * np.power(g2_val, 2) * np.power(g1_val, 2)
                              * (M1_val + M2_val))
                           - (54 * np.power(g1_val, 4) * M1_val))))

        # Total soft trilinear coupling beta functions
        dat_dt = (1 / t) * ((loop_fac * dat_dt_1l)
                            + (loop_fac_sq * dat_dt_2l))

        dac_dt = (1 / t) * ((loop_fac * dac_dt_1l)
                            + (loop_fac_sq * dac_dt_2l))

        dau_dt = (1 / t) * ((loop_fac * dau_dt_1l)
                            + (loop_fac_sq * dau_dt_2l))

        dab_dt = (1 / t) * ((loop_fac * dab_dt_1l)
                            + (loop_fac_sq * dab_dt_2l))

        das_dt = (1 / t) * ((loop_fac * das_dt_1l)
                            + (loop_fac_sq * das_dt_2l))

        dad_dt = (1 / t) * ((loop_fac * dad_dt_1l)
                            + (loop_fac_sq * dad_dt_2l))

        datau_dt = (1 / t) * ((loop_fac * datau_dt_1l)
                            + (loop_fac_sq * datau_dt_2l))

        damu_dt = (1 / t) * ((loop_fac * damu_dt_1l)
                            + (loop_fac_sq * damu_dt_2l))

        dae_dt = (1 / t) * ((loop_fac * dae_dt_1l)
                            + (loop_fac_sq * dae_dt_2l))

        ##### Soft bilinear coupling b=B*mu#####
        # 1 loop part
        db_dt_1l = ((b_val# Tr(3Yu^2 + 3Yd^2 + Ye^2)
                     * (((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                               + np.power(yu_val, 2) + np.power(yb_val, 2)
                               + np.power(ys_val, 2) + np.power(yd_val, 2)))
                         + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                         + np.power(ye_val, 2))# end trace
                        - (3 * np.power(g2_val, 2))
                        - ((3 / 5) * np.power(g1_val, 2))))
                    + (mu_val# Tr(6au*Yu + 6ad*Yd + 2ae*Ye)
                       * (((6 * ((at_val * yt_val) + (ac_val * yc_val)
                                 + (au_val * yu_val) + (ab_val * yb_val)
                                 + (as_val * ys_val) + (ad_val * yd_val)))
                           + (2 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                   + (ae_val * ye_val))))
                          + (6 * np.power(g2_val, 2) * M2_val)
                          + ((6 / 5) * np.power(g1_val, 2) * M1_val))))

        # 2 loop part
        db_dt_2l = ((b_val# Tr(3Yu^4 + 3Yd^4 + 2Yu^2*Yd^2 + Ye^4)
                     * (((-3) * ((3 * (np.power(yt_val, 4) + np.power(yc_val, 4)
                                       + np.power(yu_val, 4)
                                       + np.power(yb_val, 4)
                                       + np.power(ys_val, 4)
                                       + np.power(yd_val, 4)))
                                 + (2 * ((np.power(yt_val, 2)
                                          * np.power(yb_val, 2))
                                         + (np.power(yc_val, 2)
                                            * np.power(ys_val, 2))
                                         + (np.power(yu_val, 2)
                                            * np.power(yd_val, 2))))
                                 + np.power(ytau_val, 4) + np.power(ymu_val, 4)
                                 + np.power(ye_val, 4)))# end trace
                        + (((16 * np.power(g3_val, 2))
                            + ((4 / 5) * np.power(g1_val, 2)))# Tr(Yu^2)
                           * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        + (((16 * np.power(g3_val, 2))
                            - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                           * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))# end trace
                        + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                           * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))# end trace
                        + ((15 / 2) * np.power(g2_val, 4))
                        + ((9 / 5) * np.power(g1_val, 2) * np.power(g2_val, 2))
                        + ((207 / 50) * np.power(g1_val, 4))))
                    + (mu_val * (((-12)# Tr(3au*Yu^3 + 3ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + ae*Ye^3)
                          * ((3 * ((at_val * np.power(yt_val, 3))
                                   + (ac_val * np.power(yc_val, 3))
                                   + (au_val * np.power(yu_val, 3))
                                   + (ab_val * np.power(yb_val, 3))
                                   + (as_val * np.power(ys_val, 3))
                                   + (ad_val * np.power(yd_val, 3))))
                             + ((at_val * np.power(yb_val, 2) * yt_val)
                                + (ac_val * np.power(ys_val, 2) * yc_val)
                                + (au_val * np.power(yd_val, 2) * yu_val))
                             + ((ab_val * np.power(yt_val, 2) * yb_val)
                                + (as_val * np.power(yc_val, 2) * ys_val)
                                + (ad_val * np.power(yu_val, 2) * yd_val))
                             + ((atau_val * np.power(ytau_val, 3))
                                + (amu_val * np.power(ymu_val, 3))
                                + (ae_val * np.power(ye_val, 3)))))# end trace
                        + (((32 * np.power(g3_val, 2))
                             + ((8 / 5) * np.power(g1_val, 2)))# Tr(au*Yu)
                            * ((at_val * yt_val) + (ac_val * yc_val)
                               + (au_val * yu_val)))# end trace
                         + (((32 * np.power(g3_val, 2))
                             - ((4 / 5) * np.power(g1_val, 2)))# Tr(ad*Yd)
                            * ((ab_val * yb_val) + (as_val * ys_val)
                               + (ad_val * yd_val)))# end trace
                         + ((12 / 5) * np.power(g1_val, 2)# Tr(ae*Ye)
                            * ((atau_val * ytau_val) + (amu_val * ymu_val)
                               + (ae_val * ye_val)))# end trace
                         - (((32 * np.power(g3_val, 2) * M3_val)
                             + ((8 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yu^2)
                            * (np.power(yt_val, 2) + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))
                         - (((32 * np.power(g3_val, 2) * M3_val)# end trace
                             - ((4 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yd^2)
                            * (np.power(yb_val, 2) + np.power(ys_val, 2)
                               + np.power(yd_val, 2)))# end trace
                         - ((12 / 5) * np.power(g1_val, 2) * M1_val# Tr(Ye^2)
                            * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (30 * np.power(g2_val, 4) * M2_val)
                         - ((18 / 5) * np.power(g1_val, 2)
                            * np.power(g2_val, 2)
                            * (M1_val + M2_val))
                         - ((414 / 25) * np.power(g1_val, 4) * M1_val))))

        # Total b beta function
        db_dt = (1 / t) * ((loop_fac * db_dt_1l)
                           + (loop_fac_sq * db_dt_2l))

        ##### Scalar squared masses #####
        # Introduce S, S', and sigma terms
        S_val = (mHu_sq_val - mHd_sq_val + mQ3_sq_val + mQ2_sq_val + mQ1_sq_val
                 - mL3_sq_val - mL2_sq_val - mL1_sq_val
                 - (2 * (mU3_sq_val + mU2_sq_val + mU1_sq_val))
                 + mD3_sq_val + mD2_sq_val + mD1_sq_val
                 + mE3_sq_val + mE2_sq_val + mE1_sq_val)

        # Tr(-(3mHu^2 + mQ^2) * Yu^2 + 4Yu^2 * mU^2 + (3mHd^2 - mQ^2) * Yd^2
        #    - 2Yd^2 * mD^2 + (mHd^2 + mL^2) * Ye^2 - 2Ye^2 * mE^2)
        Spr_val = ((((-1) * ((((3 * mHu_sq_val) + mQ3_sq_val)
                              * np.power(yt_val, 2))
                             + (((3 * mHu_sq_val) + mQ2_sq_val)
                                * np.power(yc_val, 2))
                             + (((3 * mHu_sq_val) + mQ1_sq_val)
                                * np.power(yu_val, 2))))
                    + (4 * np.power(yt_val, 2) * mU3_sq_val)
                    + (4 * np.power(yc_val, 2) * mU2_sq_val)
                    + (4 * np.power(yu_val, 2) * mU1_sq_val)
                    + ((((3 * mHd_sq_val) - mQ3_sq_val) * np.power(yb_val, 2))
                       + (((3 * mHd_sq_val) - mQ2_sq_val)
                          * np.power(ys_val, 2))
                       + (((3 * mHd_sq_val) - mQ1_sq_val)
                          * np.power(yd_val, 2)))
                    - (2 * ((mD3_sq_val * np.power(yb_val, 2))
                            + (mD2_sq_val * np.power(ys_val, 2))
                            + (mD1_sq_val * np.power(yd_val, 2))))
                    + (((mHd_sq_val + mL3_sq_val) * np.power(ytau_val, 2))
                       + ((mHd_sq_val + mL2_sq_val) * np.power(ymu_val, 2))
                       + ((mHd_sq_val + mL1_sq_val) * np.power(ye_val, 2)))
                    - (2 * ((np.power(ytau_val, 2) * mE3_sq_val)
                            + (np.power(ymu_val, 2) * mE2_sq_val)
                            + (np.power(ye_val, 2) * mE1_sq_val))))# end trace
                   + ((((3 / 2) * np.power(g2_val, 2))
                       + ((3 / 10) * np.power(g1_val, 2)))
                      * (mHu_sq_val - mHd_sq_val# Tr(mL^2)
                         - (mL3_sq_val + mL2_sq_val + mL1_sq_val)))# end trace
                   + ((((8 / 3) * np.power(g3_val, 2))
                       + ((3 / 2) * np.power(g2_val, 2))
                       + ((1 / 30) * np.power(g1_val, 2)))# Tr(mQ^2)
                      * (mQ3_sq_val + mQ2_sq_val + mQ1_sq_val))# end trace
                   - ((((16 / 3) * np.power(g3_val, 2))
                       + ((16 / 15) * np.power(g1_val, 2)))# Tr (mU^2)
                      * (mU3_sq_val + mU2_sq_val + mU1_sq_val))# end trace
                   + ((((8 / 3) * np.power(g3_val, 2))
                      + ((2 / 15) * np.power(g1_val, 2)))# Tr(mD^2)
                      * (mD3_sq_val + mD2_sq_val + mD1_sq_val))# end trace
                   + ((6 / 5) * np.power(g1_val, 2)# Tr(mE^2)
                      * (mE3_sq_val + mE2_sq_val + mE1_sq_val)))# end trace

        sigma1 = ((1 / 5) * np.power(g1_val, 2)
                  * ((3 * (mHu_sq_val + mHd_sq_val))# Tr(mQ^2 + 3mL^2 + 8mU^2 + 2mD^2 + 6mE^2)
                     + mQ3_sq_val + mQ2_sq_val + mQ1_sq_val
                     + (3 * (mL3_sq_val + mL2_sq_val + mL1_sq_val))
                     + (8 * (mU3_sq_val + mU2_sq_val + mU1_sq_val))
                     + (2 * (mD3_sq_val + mD2_sq_val + mD1_sq_val))
                     + (6 * (mE3_sq_val + mE2_sq_val + mE1_sq_val))))# end trace

        sigma2 = (np.power(g2_val, 2)
                  * (mHu_sq_val + mHd_sq_val# Tr(3mQ^2 + mL^2)
                     + (3 * (mQ3_sq_val + mQ2_sq_val + mQ1_sq_val))
                     + mL3_sq_val + mL2_sq_val + mL1_sq_val))# end trace

        sigma3 = (np.power(g3_val, 2)# Tr(2mQ^2 + mU^2 + mD^2)
                  * ((2 * (mQ3_sq_val + mQ2_sq_val + mQ1_sq_val))
                     + mU3_sq_val + mU2_sq_val + mU1_sq_val
                     + mD3_sq_val + mD2_sq_val + mD1_sq_val))# end trace

        # 1 loop part of Higgs squared masses
        dmHu_sq_dt_1l = ((6# Tr((mHu^2 + mQ^2) * Yu^2 + Yu^2 * mU^2 + au^2)
                          * (((mHu_sq_val + mQ3_sq_val) * np.power(yt_val, 2))
                             + ((mHu_sq_val + mQ2_sq_val)
                                * np.power(yc_val, 2))
                             + ((mHu_sq_val + mQ1_sq_val)
                                * np.power(yu_val, 2))
                             + (mU3_sq_val * np.power(yt_val, 2))
                             + (mU2_sq_val * np.power(yc_val, 2))
                             + (mU1_sq_val * np.power(yu_val, 2))
                             + np.power(at_val, 2) + np.power(ac_val, 2)
                             + np.power(au_val, 2)))# end trace
                         - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))
                         - ((6 / 5) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((3 / 5) * np.power(g1_val, 2) * S_val))

        # Tr (6(mHd^2 + mQ^2) * Yd^2 + 6Yd^2*mD^2 + 2(mHd^2 + mL^2) * Ye^2
        #     + 2(Ye^2 * mE^2) + 6ad^2 + 2ae^2)
        dmHd_sq_dt_1l = ((6 * (((mHd_sq_val + mQ3_sq_val)
                                * np.power(yb_val, 2))
                               + ((mHd_sq_val + mQ2_sq_val)
                                  * np.power(ys_val, 2))
                               + ((mHd_sq_val + mQ1_sq_val)
                                  * np.power(yd_val, 2)))
                          + (6 * ((mD3_sq_val * np.power(yb_val, 2))
                                  + (mD2_sq_val * np.power(ys_val, 2))
                                  + (mD1_sq_val * np.power(yd_val, 2))))
                          + (2 * (((mHd_sq_val + mL3_sq_val)
                                   * np.power(ytau_val, 2))
                                  + ((mHd_sq_val + mL2_sq_val)
                                     * np.power(ymu_val, 2))
                                  + ((mHd_sq_val + mL1_sq_val)
                                     * np.power(ye_val, 2))))
                          + (2 * ((mE3_sq_val * np.power(ytau_val, 2))
                                  + (mE2_sq_val * np.power(ymu_val, 2))
                                  + (mE1_sq_val * np.power(ye_val, 2))))
                          + (6 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                  + np.power(ad_val, 2)))
                          + (2 * (np.power(atau_val, 2) + np.power(amu_val, 2)
                                  + np.power(ae_val, 2))))# end trace
                         - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))
                         - ((6 / 5) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         - ((3 / 5) * np.power(g1_val, 2) * S_val))

        # 2 loop part of Higgs squared masses
        dmHu_sq_dt_2l = (((-6) # Tr(6(mHu^2 + mQ^2)*Yu^4 + 6Yu^4 * mU^2 + (mHu^2 + mHd^2 + mQ^2) * Yu^2 * Yd^2 + Yu^2 * Yd^2 * mU^2 + Yu^2 * Yd^2 * mQ^2 + Yu^2 * Yd^2 * mD^2 + 12au^2 * Yu^2 + ad^2 * Yu^2 + Yd^2 * au^2 + 2ad * Yd * Yu * au)
                          * ((6 * (((mHu_sq_val + mQ3_sq_val)
                                    * np.power(yt_val, 4))
                                   + ((mHu_sq_val + mQ2_sq_val)
                                      * np.power(yc_val, 4))
                                   + ((mHu_sq_val + mQ1_sq_val)
                                      * np.power(yu_val, 4))))
                             + (6 * ((mU3_sq_val * np.power(yt_val, 4))
                                     + (mU2_sq_val * np.power(yc_val, 4))
                                     + (mU1_sq_val * np.power(yu_val, 4))))
                             + ((mHu_sq_val + mHd_sq_val + mQ3_sq_val)
                                * np.power(yt_val, 2) * np.power(yb_val, 2))
                             + ((mHu_sq_val + mHd_sq_val + mQ2_sq_val)
                                * np.power(yc_val, 2) * np.power(ys_val, 2))
                             + ((mHu_sq_val + mHd_sq_val + mQ1_sq_val)
                                * np.power(yu_val, 2) * np.power(yd_val, 2))
                             + ((mU3_sq_val + mQ3_sq_val + mD3_sq_val)
                                * np.power(yt_val, 2) * np.power(yb_val, 2))
                             + ((mU2_sq_val + mQ2_sq_val + mD2_sq_val)
                                * np.power(yc_val, 2) * np.power(ys_val, 2))
                             + ((mU1_sq_val + mQ1_sq_val + mD1_sq_val)
                                * np.power(yu_val, 2) * np.power(yd_val, 2))
                             + (12 * ((np.power(at_val, 2)
                                       * np.power(yt_val, 2))
                                      + (np.power(ac_val, 2)
                                         * np.power(yc_val, 2))
                                      + (np.power(au_val, 2)
                                         * np.power(yu_val, 2))))
                             + (np.power(ab_val, 2) * np.power(yt_val, 2))
                             + (np.power(as_val, 2) * np.power(yc_val, 2))
                             + (np.power(ad_val, 2) * np.power(yu_val, 2))
                             + (np.power(yb_val, 2) * np.power(at_val, 2))
                             + (np.power(ys_val, 2) * np.power(ac_val, 2))
                             + (np.power(yd_val, 2) * np.power(au_val, 2))
                             + (2 * ((yb_val * ab_val * at_val * yt_val)
                                     + (ys_val * as_val * ac_val * yc_val)
                                     + (yd_val * ad_val * au_val * yu_val)))))# end trace
                         + (((32 * np.power(g3_val, 2))
                             + ((8 / 5) * np.power(g1_val, 2))) # Tr((mHu^2 + mQ^2 + mU^2) * Yu^2 + au^2)
                            * (((mHu_sq_val + mQ3_sq_val + mU3_sq_val)
                                * np.power(yt_val, 2))
                               + ((mHu_sq_val + mQ2_sq_val + mU2_sq_val)
                                  * np.power(yc_val, 2))
                               + ((mHu_sq_val + mQ1_sq_val + mU1_sq_val)
                                  * np.power(yu_val, 2))
                               + np.power(at_val, 2) + np.power(ac_val, 2)
                               + np.power(au_val, 2)))# end trace
                         + (32 * np.power(g3_val, 2)
                            * ((2 * np.power(M3_val, 2)# Tr(Yu^2)
                                * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                               - (2 * M3_val# Tr(Yu*au)
                                  * ((yt_val * at_val) + (yc_val * ac_val)
                                     + (yu_val * au_val)))))# end trace
                         + ((8 / 5) * np.power(g1_val, 2)
                            * ((2 * np.power(M1_val, 2)# Tr(Yu^2)
                                * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                               - (2 * M1_val# Tr(Yu*au)
                                  * ((yt_val * at_val) + (yc_val * ac_val)
                                     + (yu_val * au_val)))))# end trace
                         + ((6 / 5) * np.power(g1_val, 2) * Spr_val)
                         + (33 * np.power(g2_val, 4) * np.power(M2_val, 2))
                         + ((18 / 5) * np.power(g2_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M2_val, 2) + np.power(M1_val, 2)
                               + (M1_val * M2_val)))
                         + ((621 / 25) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + (3 * np.power(g2_val, 2) * sigma2)
                         + ((3 / 5) * np.power(g1_val, 2) * sigma1))

        dmHd_sq_dt_2l = (((-6) # Tr(6(mHd^2 + mQ^2)*Yd^4 + 6Yd^4 * mD^2 + (mHu^2 + mHd^2 + mQ^2) * Yu^2 * Yd^2 + Yu^2 * Yd^2 * mU^2 + Yu^2 * Yd^2 * mQ^2 + Yu^2 * Yd^2 * mD^2 + 2(mHd^2 + mL^2) * Ye^4 + 2Ye^4 * mE^2 + 12ad^2 * Yd^2 + ad^2 * Yu^2 + Yd^2 * au^2 + 2ad * Yd * Yu * au + 4ae^2 * Ye^2)
                          * ((6 * (((mHd_sq_val + mQ3_sq_val)
                                    * np.power(yb_val, 4))
                                   + ((mHd_sq_val + mQ2_sq_val)
                                      * np.power(ys_val, 4))
                                   + ((mHd_sq_val + mQ1_sq_val)
                                      * np.power(yd_val, 4))))
                             + (6 * ((mD3_sq_val * np.power(yb_val, 4))
                                     + (mD2_sq_val * np.power(ys_val, 4))
                                     + (mD1_sq_val * np.power(yd_val, 4))))
                             + ((mHu_sq_val + mHd_sq_val + mQ3_sq_val)
                                * np.power(yt_val, 2) * np.power(yb_val, 2))
                             + ((mHu_sq_val + mHd_sq_val + mQ2_sq_val)
                                * np.power(yc_val, 2) * np.power(ys_val, 2))
                             + ((mHu_sq_val + mHd_sq_val + mQ1_sq_val)
                                * np.power(yu_val, 2) * np.power(yd_val, 2))
                             + ((mU3_sq_val + mQ3_sq_val + mD3_sq_val)
                                * np.power(yt_val, 2) * np.power(yb_val, 2))
                             + ((mU2_sq_val + mQ2_sq_val + mD2_sq_val)
                                * np.power(yc_val, 2) * np.power(ys_val, 2))
                             + ((mU1_sq_val + mQ1_sq_val + mD1_sq_val)
                                * np.power(yu_val, 2) * np.power(yd_val, 2))
                             + (2 * (((mHd_sq_val + mL3_sq_val + mE3_sq_val)
                                      * np.power(ytau_val, 4))
                                     + ((mHd_sq_val + mL2_sq_val + mE2_sq_val)
                                        * np.power(ymu_val, 4))
                                     + ((mHd_sq_val + mL1_sq_val + mE1_sq_val)
                                        * np.power(ye_val, 4))))
                             + (12 * ((np.power(ab_val, 2)
                                       * np.power(yb_val, 2))
                                      + (np.power(as_val, 2)
                                         * np.power(ys_val, 2))
                                      + (np.power(ad_val, 2)
                                         * np.power(yd_val, 2))))
                             + (np.power(ab_val, 2) * np.power(yt_val, 2))
                             + (np.power(as_val, 2) * np.power(yc_val, 2))
                             + (np.power(ad_val, 2) * np.power(yu_val, 2))
                             + (np.power(yb_val, 2) * np.power(at_val, 2))
                             + (np.power(ys_val, 2) * np.power(ac_val, 2))
                             + (np.power(yd_val, 2) * np.power(au_val, 2))
                             + (2 * ((yb_val * ab_val * at_val * yt_val)
                                     + (ys_val * as_val * ac_val * yc_val)
                                     + (yd_val * ad_val * au_val * yu_val)
                                     + (2 * ((np.power(atau_val, 2)
                                              * np.power(ytau_val, 2))
                                             + (np.power(amu_val, 2)
                                                * np.power(ymu_val, 2))
                                             + (np.power(ae_val, 2)
                                                * np.power(ye_val, 2))))))))# end trace
                         + (((32 * np.power(g3_val, 2))
                             - ((4 / 5) * np.power(g1_val, 2))) # Tr((mHd^2 + mQ^2 + mD^2) * Yd^2 + ad^2)
                            * (((mHu_sq_val + mQ3_sq_val + mD3_sq_val)
                                * np.power(yb_val, 2))
                               + ((mHu_sq_val + mQ2_sq_val + mD2_sq_val)
                                  * np.power(ys_val, 2))
                               + ((mHu_sq_val + mQ1_sq_val + mD1_sq_val)
                                  * np.power(yd_val, 2))
                               + np.power(ab_val, 2) + np.power(as_val, 2)
                               + np.power(ad_val, 2)))# end trace
                         + (32 * np.power(g3_val, 2)
                            * ((2 * np.power(M3_val, 2)# Tr(Yd^2)
                                * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                   + np.power(yd_val, 2)))# end trace
                               - (2 * M3_val # Tr(Yd*ad)
                                  * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))))# end trace
                         - ((4 / 5) * np.power(g1_val, 2)
                            * ((2 * np.power(M1_val, 2)# Tr(Yd^2)
                                * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                   + np.power(yd_val, 2)))# end trace
                               - (2 * M1_val # Tr(Yd*ad)
                                  * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))))# end trace
                         + ((12 / 5) * np.power(g1_val, 2)
                            * (# Tr((mHd^2 + mL^2 + mE^2) * Ye^2 + ae^2)
                               ((mHd_sq_val + mL3_sq_val + mE3_sq_val)
                                * np.power(ytau_val, 2))
                               + ((mHd_sq_val + mL2_sq_val + mE2_sq_val)
                                  * np.power(ymu_val, 2))
                               + ((mHd_sq_val + mL1_sq_val + mE1_sq_val)
                                  * np.power(ye_val, 2))
                               + np.power(atau_val, 2) + np.power(amu_val, 2)
                               + np.power(ae_val, 2)))# end trace
                         - ((6 / 5) * np.power(g1_val, 2) * Spr_val)
                         + (33 * np.power(g2_val, 4) * np.power(M2_val, 2))
                         + ((18 / 5) * np.power(g2_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M2_val, 2) + np.power(M1_val, 2)
                               + (M1_val * M2_val)))
                         + ((621 / 25) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + (3 * np.power(g2_val, 2) * sigma2)
                         + ((3 / 5) * np.power(g1_val, 2) * sigma1))

        # Total Higgs squared mass beta functions
        dmHu_sq_dt = (1 / t) * ((loop_fac * dmHu_sq_dt_1l)
                                + (loop_fac_sq * dmHu_sq_dt_2l))

        dmHd_sq_dt = (1 / t) * ((loop_fac * dmHd_sq_dt_1l)
                                + (loop_fac_sq * dmHd_sq_dt_2l))

        # 1 loop parts of scalar squared masses
        # Left squarks
        dmQ3_sq_dt_1l = (((mQ3_sq_val + (2 * mHu_sq_val))
                          * np.power(yt_val, 2))
                         + ((mQ3_sq_val + (2 * mHd_sq_val))
                            * np.power(yb_val, 2))
                         + ((np.power(yt_val, 2) + np.power(yb_val, 2))
                            * mQ3_sq_val)
                         + (2 * np.power(yt_val, 2) * mU3_sq_val)
                         + (2 * np.power(yb_val, 2) * mD3_sq_val)
                         + (2 * np.power(at_val, 2))
                         + (2 * np.power(ab_val, 2))
                         - ((32 / 3) * np.power(g3_val, 2)
                            * np.power(M3_val, 2))
                         - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))
                         - ((2 / 15) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((1 / 5) * np.power(g1_val, 2) * S_val))

        dmQ2_sq_dt_1l = (((mQ2_sq_val + (2 * mHu_sq_val))
                          * np.power(yc_val, 2))
                         + ((mQ2_sq_val + (2 * mHd_sq_val))
                            * np.power(ys_val, 2))
                         + ((np.power(yc_val, 2) + np.power(ys_val, 2))
                            * mQ2_sq_val)
                         + (2 * np.power(yc_val, 2) * mU2_sq_val)
                         + (2 * np.power(ys_val, 2) * mD2_sq_val)
                         + (2 * np.power(ac_val, 2))
                         + (2 * np.power(as_val, 2))
                         - ((32 / 3) * np.power(g3_val, 2)
                            * np.power(M3_val, 2))
                         - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))
                         - ((2 / 15) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((1 / 5) * np.power(g1_val, 2) * S_val))

        dmQ1_sq_dt_1l = (((mQ1_sq_val + (2 * mHu_sq_val))
                          * np.power(yu_val, 2))
                         + ((mQ1_sq_val + (2 * mHd_sq_val))
                            * np.power(yd_val, 2))
                         + ((np.power(yu_val, 2)
                             + np.power(yd_val, 2)) * mQ1_sq_val)
                         + (2 * np.power(yu_val, 2) * mU1_sq_val)
                         + (2 * np.power(yd_val, 2) * mD1_sq_val)
                         + (2 * np.power(au_val, 2))
                         + (2 * np.power(ad_val, 2))
                         - ((32 / 3) * np.power(g3_val, 2)
                            * np.power(M3_val, 2))
                         - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))
                         - ((2 / 15) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((1 / 5) * np.power(g1_val, 2) * S_val))

        # Left leptons
        dmL3_sq_dt_1l = (((mL3_sq_val + (2 * mHd_sq_val))
                          * np.power(ytau_val, 2))
                         + (2 * np.power(ytau_val, 2) * mE3_sq_val)
                         + (np.power(ytau_val, 2) * mL3_sq_val)
                         + (2 * np.power(atau_val, 2))
                         - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))
                         - ((6 / 5) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         - ((3 / 5) * np.power(g1_val, 2) * S_val))

        dmL2_sq_dt_1l = (((mL2_sq_val + (2 * mHd_sq_val))
                          * np.power(ymu_val, 2))
                         + (2 * np.power(ymu_val, 2) * mE2_sq_val)
                         + (np.power(ymu_val, 2) * mL2_sq_val)
                         + (2 * np.power(amu_val, 2))
                         - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))
                         - ((6 / 5) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         - ((3 / 5) * np.power(g1_val, 2) * S_val))

        dmL1_sq_dt_1l = (((mL1_sq_val + (2 * mHd_sq_val))
                          * np.power(ye_val, 2))
                         + (2 * np.power(ye_val, 2) * mE1_sq_val)
                         + (np.power(ye_val, 2) * mL1_sq_val)
                         + (2 * np.power(ae_val, 2))
                         - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))
                         - ((6 / 5) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         - ((3 / 5) * np.power(g1_val, 2) * S_val))

        # Right up-type squarks
        dmU3_sq_dt_1l = ((2 * (mU3_sq_val + (2 * mHd_sq_val))
                          * np.power(yt_val, 2))
                         + (4 * np.power(yt_val, 2) * mQ3_sq_val)
                         + (2 * np.power(yt_val, 2) * mU3_sq_val)
                         + (4 * np.power(at_val, 2))
                         - ((32 / 3) * np.power(g3_val, 2)
                            * np.power(M3_val, 2))
                         - ((32 / 15) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         - ((4 / 5) * np.power(g1_val, 2) * S_val))

        dmU2_sq_dt_1l = ((2 * (mU2_sq_val + (2 * mHd_sq_val))
                          * np.power(yc_val, 2))
                         + (4 * np.power(yc_val, 2) * mQ2_sq_val)
                         + (2 * np.power(yc_val, 2) * mU2_sq_val)
                         + (4 * np.power(ac_val, 2))
                         - ((32 / 3) * np.power(g3_val, 2)
                            * np.power(M3_val, 2))
                         - ((32 / 15) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         - ((4 / 5) * np.power(g1_val, 2) * S_val))

        dmU1_sq_dt_1l = ((2 * (mU1_sq_val + (2 * mHd_sq_val))
                          * np.power(yu_val, 2))
                         + (4 * np.power(yu_val, 2) * mQ1_sq_val)
                         + (2 * np.power(yu_val, 2) * mU1_sq_val)
                         + (4 * np.power(au_val, 2))
                         - ((32 / 3) * np.power(g3_val, 2)
                            * np.power(M3_val, 2))
                         - ((32 / 15) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         - ((4 / 5) * np.power(g1_val, 2) * S_val))

        # Right down-type squarks
        dmD3_sq_dt_1l = ((2 * (mD3_sq_val + (2 * mHd_sq_val))
                          * np.power(yb_val, 2))
                         + (4 * np.power(yb_val, 2) * mQ3_sq_val)
                         + (2 * np.power(yb_val, 2) * mD3_sq_val)
                         + (4 * np.power(ab_val, 2))
                         - ((32 / 3) * np.power(g3_val, 2)
                            * np.power(M3_val, 2))
                         - ((8 / 15) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((2 / 5) * np.power(g1_val, 2) * S_val))

        dmD2_sq_dt_1l = ((2 * (mD2_sq_val + (2 * mHd_sq_val))
                          * np.power(ys_val, 2))
                         + (4 * np.power(ys_val, 2) * mQ2_sq_val)
                         + (2 * np.power(ys_val, 2) * mD2_sq_val)
                         + (4 * np.power(as_val, 2))
                         - ((32 / 3) * np.power(g3_val, 2)
                            * np.power(M3_val, 2))
                         - ((8 / 15) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((2 / 5) * np.power(g1_val, 2) * S_val))

        dmD1_sq_dt_1l = ((2 * (mD1_sq_val + (2 * mHd_sq_val))
                          * np.power(yd_val, 2))
                         + (4 * np.power(yd_val, 2) * mQ1_sq_val)
                         + (2 * np.power(yd_val, 2) * mD1_sq_val)
                         + (4 * np.power(ad_val, 2))
                         - ((32 / 3) * np.power(g3_val, 2)
                            * np.power(M3_val, 2))
                         - ((8 / 15) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((2 / 5) * np.power(g1_val, 2) * S_val))

        # Right leptons
        dmE3_sq_dt_1l = ((2 * (mE3_sq_val + (2 * mHd_sq_val))
                          * np.power(ytau_val, 2))
                         + (4 * np.power(ytau_val, 2) * mL3_sq_val)
                         + (2 * np.power(ytau_val, 2) * mE3_sq_val)
                         + (4 * np.power(atau_val, 2))
                         - ((24 / 5) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((6 / 5) * np.power(g1_val, 2) * S_val))

        dmE2_sq_dt_1l = ((2 * (mE2_sq_val + (2 * mHd_sq_val))
                          * np.power(ymu_val, 2))
                         + (4 * np.power(ymu_val, 2) * mL2_sq_val)
                         + (2 * np.power(ymu_val, 2) * mE2_sq_val)
                         + (4 * np.power(amu_val, 2))
                         - ((24 / 5) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((6 / 5) * np.power(g1_val, 2) * S_val))

        dmE1_sq_dt_1l = ((2 * (mE1_sq_val + (2 * mHd_sq_val))
                          * np.power(ye_val, 2))
                         + (4 * np.power(ye_val, 2) * mL1_sq_val)
                         + (2 * np.power(ye_val, 2) * mE1_sq_val)
                         + (4 * np.power(ae_val, 2))
                         - ((24 / 5) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((6 / 5) * np.power(g1_val, 2) * S_val))

        # 2 loop parts of scalar squared masses
        # Left squarks
        dmQ3_sq_dt_2l = (((-8) * (mQ3_sq_val + mHu_sq_val + mU3_sq_val)
                          * np.power(yt_val, 4))
                         - (8 * (mQ3_sq_val + mHd_sq_val + mD3_sq_val)
                            * np.power(yb_val, 4))
                         - (np.power(yt_val, 2)
                            * ((2 * mQ3_sq_val) + (2 * mU3_sq_val)
                               + (4 * mHu_sq_val))# Tr(3Yu^2)
                            * 3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                         - (np.power(yb_val, 2)
                            * ((2 * mQ3_sq_val) + (2 * mD3_sq_val)
                               + (4 * mHd_sq_val))# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (6 * np.power(yt_val, 2)# Tr((mQ^2 + mU^2)*Yu^2)
                            * (((mQ3_sq_val + mU3_sq_val)
                                * np.power(yt_val, 2))
                               + ((mQ2_sq_val + mU2_sq_val)
                                  * np.power(yc_val, 2))
                               + ((mQ1_sq_val + mU1_sq_val)
                                  * np.power(yu_val, 2))))# end trace
                         - (np.power(yb_val, 2)# Tr(6(mQ^2 + mD^2)*Yd^2 + 2(mL^2 + mE^2)*Ye^2)
                            * ((6 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + (2 * (((mL3_sq_val + mE3_sq_val)
                                        * np.power(ytau_val, 2))
                                       + ((mL2_sq_val + mE2_sq_val)
                                          * np.power(ymu_val, 2))
                                       + ((mL1_sq_val + mE1_sq_val)
                                          * np.power(ye_val, 2))))# end trace
                               ))
                         - (16 * np.power(yt_val, 2) * np.power(at_val, 2))
                         - (16 * np.power(yb_val, 2) * np.power(ab_val, 2))
                         - (np.power(at_val, 2)# Tr(6Yu^2)
                            * 6 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                         - (np.power(yt_val, 2)# Tr(6au^2)
                            * 6 * (np.power(at_val, 2) + np.power(ac_val, 2)
                                   + np.power(au_val, 2)))# end trace
                         - (at_val * yt_val# Tr(12Yu*au)
                            * 12 * ((yt_val * at_val) + (yc_val * ac_val)
                                    + (yu_val * au_val)))# end trace
                         - (np.power(ab_val, 2)# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (np.power(yb_val, 2)# Tr(6ad^2 + 2ae^2)
                            * ((6 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + (2 * (np.power(atau_val, 2)
                                       + np.power(amu_val, 2)
                                       + np.power(ae_val, 2)))))# end trace
                         - (2 * ab_val * yb_val# Tr(6Yd*ad + 2Ye*ae)
                            * ((6 * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))
                               + (2 * ((ytau_val * atau_val)
                                       + (ymu_val * amu_val)
                                       + (ye_val * ae_val)))))# end trace
                         + ((2 / 5) * np.power(g1_val, 2)
                            * ((4 * (mQ3_sq_val + mHu_sq_val + mU3_sq_val)
                                * np.power(yt_val, 2))
                               + (4 * np.power(at_val, 2))
                               - (8 * M1_val * at_val * yt_val)
                               + (8 * np.power(M1_val, 2) * np.power(yt_val, 2))
                               + (2 * (mQ3_sq_val + mHd_sq_val + mD3_sq_val)
                                  * np.power(yb_val, 2))
                               + (2 * np.power(ab_val, 2))
                               - (4 * M1_val * ab_val * yb_val)
                               + (4 * np.power(M1_val, 2)
                                  * np.power(yb_val, 2))))
                         + ((2 / 5) * np.power(g1_val, 2) * Spr_val)
                         - ((128 / 3) * np.power(g3_val, 4)
                            * np.power(M3_val, 2))
                         + (32 * np.power(g3_val, 2) * np.power(g2_val, 2)
                            * (np.power(M3_val, 2) + np.power(M2_val, 2)
                               + (M2_val * M3_val)))
                         + ((32 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M3_val, 2) + np.power(M1_val, 2)
                               + (M3_val * M1_val)))
                         + (33 * np.power(g2_val, 4) * np.power(M2_val, 2))
                         + ((2 / 5) * np.power(g2_val, 2) * np.power(g1_val, 2)
                            * (np.power(M1_val, 2) + np.power(M2_val, 2)
                               + (M1_val * M2_val)))
                         + ((199 / 75) * np.power(g1_val, 4) * np.power(M1_val, 2))
                         + ((16 / 3) * np.power(g3_val, 2) * sigma3)
                         + (3 * np.power(g2_val, 2) * sigma2)
                         + ((1 / 15) * np.power(g1_val, 2) * sigma1))

        dmQ2_sq_dt_2l = (((-8) * (mQ2_sq_val + mHu_sq_val + mU2_sq_val)
                          * np.power(yc_val, 4))
                         - (8 * (mQ2_sq_val + mHd_sq_val + mD2_sq_val)
                            * np.power(ys_val, 4))
                         - (np.power(yc_val, 2)
                            * ((2 * mQ2_sq_val) + (2 * mU2_sq_val)
                               + (4 * mHu_sq_val))# Tr(3Yu^2)
                            * 3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                         - (np.power(ys_val, 2)
                            * ((2 * mQ2_sq_val) + (2 * mD2_sq_val)
                               + (4 * mHd_sq_val))# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (6 * np.power(yc_val, 2)# Tr((mQ^2 + mU^2)*Yu^2)
                            * (((mQ3_sq_val + mU3_sq_val)
                                * np.power(yt_val, 2))
                               + ((mQ2_sq_val + mU2_sq_val)
                                  * np.power(yc_val, 2))
                               + ((mQ1_sq_val + mU1_sq_val)
                                  * np.power(yu_val, 2))))# end trace
                         - (np.power(ys_val, 2)# Tr(6(mQ^2 + mD^2)*Yd^2 + 2(mL^2 + mE^2)*Ye^2)
                            * ((6 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + (2 * (((mL3_sq_val + mE3_sq_val)
                                        * np.power(ytau_val, 2))
                                       + ((mL2_sq_val + mE2_sq_val)
                                          * np.power(ymu_val, 2))
                                       + ((mL1_sq_val + mE1_sq_val)
                                          * np.power(ye_val, 2))))# end trace
                               ))
                         - (16 * np.power(yc_val, 2) * np.power(ac_val, 2))
                         - (16 * np.power(ys_val, 2) * np.power(as_val, 2))
                         - (np.power(ac_val, 2)# Tr(6Yu^2)
                            * 6 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                         - (np.power(yc_val, 2)# Tr(6au^2)
                            * 6 * (np.power(at_val, 2) + np.power(ac_val, 2)
                                   + np.power(au_val, 2)))# end trace
                         - (ac_val * yc_val# Tr(12Yu*au)
                            * 12 * ((yt_val * at_val) + (yc_val * ac_val)
                                    + (yu_val * au_val)))# end trace
                         - (np.power(as_val, 2)# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (np.power(ys_val, 2)# Tr(6ad^2 + 2ae^2)
                            * ((6 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + (2 * (np.power(atau_val, 2)
                                       + np.power(amu_val, 2)
                                       + np.power(ae_val, 2)))))# end trace
                         - (2 * as_val * ys_val# Tr(6Yd*ad + 2Ye*ae)
                            * ((6 * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))
                               + (2 * ((ytau_val * atau_val)
                                       + (ymu_val * amu_val)
                                       + (ye_val * ae_val)))))# end trace
                         + ((2 / 5) * np.power(g1_val, 2)
                            * ((4 * (mQ2_sq_val + mHu_sq_val + mU2_sq_val)
                                * np.power(yc_val, 2))
                               + (4 * np.power(ac_val, 2))
                               - (8 * M1_val * ac_val * yc_val)
                               + (8 * np.power(M1_val, 2) * np.power(yc_val, 2))
                               + (2 * (mQ2_sq_val + mHd_sq_val + mD2_sq_val)
                                  * np.power(ys_val, 2))
                               + (2 * np.power(as_val, 2))
                               - (4 * M1_val * as_val * ys_val)
                               + (4 * np.power(M1_val, 2)
                                  * np.power(ys_val, 2))))
                         + ((2 / 5) * np.power(g1_val, 2) * Spr_val)
                         - ((128 / 3) * np.power(g3_val, 4)
                            * np.power(M3_val, 2))
                         + (32 * np.power(g3_val, 2) * np.power(g2_val, 2)
                            * (np.power(M3_val, 2) + np.power(M2_val, 2)
                               + (M2_val * M3_val)))
                         + ((32 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M3_val, 2) + np.power(M1_val, 2)
                               + (M3_val * M1_val)))
                         + (33 * np.power(g2_val, 4) * np.power(M2_val, 2))
                         + ((2 / 5) * np.power(g2_val, 2) * np.power(g1_val, 2)
                            * (np.power(M1_val, 2) + np.power(M2_val, 2)
                               + (M1_val * M2_val)))
                         + ((199 / 75) * np.power(g1_val, 4) * np.power(M1_val, 2))
                         + ((16 / 3) * np.power(g3_val, 2) * sigma3)
                         + (3 * np.power(g2_val, 2) * sigma2)
                         + ((1 / 15) * np.power(g1_val, 2) * sigma1))

        dmQ1_sq_dt_2l = (((-8) * (mQ1_sq_val + mHu_sq_val + mU1_sq_val)
                          * np.power(yu_val, 4))
                         - (8 * (mQ1_sq_val + mHd_sq_val + mD1_sq_val)
                            * np.power(yd_val, 4))
                         - (np.power(yu_val, 2)
                            * ((2 * mQ1_sq_val) + (2 * mU1_sq_val)
                               + (4 * mHu_sq_val))# Tr(3Yu^2)
                            * 3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                         - (np.power(yd_val, 2)
                            * ((2 * mQ1_sq_val) + (2 * mD1_sq_val)
                               + (4 * mHd_sq_val))# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (6 * np.power(yu_val, 2)# Tr((mQ^2 + mU^2)*Yu^2)
                            * (((mQ3_sq_val + mU3_sq_val)
                                * np.power(yt_val, 2))
                               + ((mQ2_sq_val + mU2_sq_val)
                                  * np.power(yc_val, 2))
                               + ((mQ1_sq_val + mU1_sq_val)
                                  * np.power(yu_val, 2))))# end trace
                         - (np.power(yd_val, 2)# Tr(6(mQ^2 + mD^2)*Yd^2 + 2(mL^2 + mE^2)*Ye^2)
                            * ((6 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + (2 * (((mL3_sq_val + mE3_sq_val)
                                        * np.power(ytau_val, 2))
                                       + ((mL2_sq_val + mE2_sq_val)
                                          * np.power(ymu_val, 2))
                                       + ((mL1_sq_val + mE1_sq_val)
                                          * np.power(ye_val, 2))))# end trace
                               ))
                         - (16 * np.power(yu_val, 2) * np.power(au_val, 2))
                         - (16 * np.power(yd_val, 2) * np.power(ad_val, 2))
                         - (np.power(au_val, 2)# Tr(6Yu^2)
                            * 6 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                         - (np.power(yu_val, 2)# Tr(6au^2)
                            * 6 * (np.power(at_val, 2) + np.power(ac_val, 2)
                                   + np.power(au_val, 2)))# end trace
                         - (au_val * yu_val# Tr(12Yu*au)
                            * 12 * ((yt_val * at_val) + (yc_val * ac_val)
                                    + (yu_val * au_val)))# end trace
                         - (np.power(ad_val, 2)# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (np.power(yd_val, 2)# Tr(6ad^2 + 2ae^2)
                            * ((6 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + (2 * (np.power(atau_val, 2)
                                       + np.power(amu_val, 2)
                                       + np.power(ae_val, 2)))))# end trace
                         - (2 * ad_val * yd_val# Tr(6Yd*ad + 2Ye*ae)
                            * ((6 * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))
                               + (2 * ((ytau_val * atau_val)
                                       + (ymu_val * amu_val)
                                       + (ye_val * ae_val)))))# end trace
                         + ((2 / 5) * np.power(g1_val, 2)
                            * ((4 * (mQ1_sq_val + mHu_sq_val + mU1_sq_val)
                                * np.power(yu_val, 2))
                               + (4 * np.power(au_val, 2))
                               - (8 * M1_val * au_val * yu_val)
                               + (8 * np.power(M1_val, 2) * np.power(yu_val, 2))
                               + (2 * (mQ1_sq_val + mHd_sq_val + mD1_sq_val)
                                  * np.power(yd_val, 2))
                               + (2 * np.power(ad_val, 2))
                               - (4 * M1_val * ad_val * yd_val)
                               + (4 * np.power(M1_val, 2)
                                  * np.power(yd_val, 2))))
                         + ((2 / 5) * np.power(g1_val, 2) * Spr_val)
                         - ((128 / 3) * np.power(g3_val, 4)
                            * np.power(M3_val, 2))
                         + (32 * np.power(g3_val, 2) * np.power(g2_val, 2)
                            * (np.power(M3_val, 2) + np.power(M2_val, 2)
                               + (M2_val * M3_val)))
                         + ((32 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M3_val, 2) + np.power(M1_val, 2)
                               + (M3_val * M1_val)))
                         + (33 * np.power(g2_val, 4) * np.power(M2_val, 2))
                         + ((2 / 5) * np.power(g2_val, 2) * np.power(g1_val, 2)
                            * (np.power(M1_val, 2) + np.power(M2_val, 2)
                               + (M1_val * M2_val)))
                         + ((199 / 75) * np.power(g1_val, 4) * np.power(M1_val, 2))
                         + ((16 / 3) * np.power(g3_val, 2) * sigma3)
                         + (3 * np.power(g2_val, 2) * sigma2)
                         + ((1 / 15) * np.power(g1_val, 2) * sigma1))

        # Left leptons
        dmL3_sq_dt_2l = (((-8) * (mL3_sq_val + mHd_sq_val + mE3_sq_val)
                          * np.power(ytau_val, 4))
                         - (np.power(ytau_val, 2)
                            * ((2 * mL3_sq_val) + (2 * mE3_sq_val)
                               + (4 * mHd_sq_val))# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (np.power(ytau_val, 2)# Tr(6(mQ^2 + mD^2)*Yd^2 + 2(mL^2 + mE^2)*Ye^2)
                            * ((6 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + (2 * (((mL3_sq_val + mE3_sq_val)
                                        * np.power(ytau_val, 2))
                                       + ((mL2_sq_val + mE2_sq_val)
                                          * np.power(ymu_val, 2))
                                       + ((mL1_sq_val + mE1_sq_val)
                                          * np.power(ye_val, 2))))# end trace
                               ))
                         - (16 * np.power(ytau_val, 2) * np.power(atau_val, 2))
                         - (np.power(atau_val, 2)# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (np.power(ytau_val, 2)# Tr(6ad^2 + 2ae^2)
                            * ((6 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + (2 * (np.power(atau_val, 2)
                                       + np.power(amu_val, 2)
                                       + np.power(ae_val, 2)))))# end trace
                         - (2 * atau_val * ytau_val# Tr(6Yd*ad + 2Ye*ae)
                            * ((6 * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))
                               + (2 * ((ytau_val * atau_val)
                                       + (ymu_val * amu_val)
                                       + (ye_val * ae_val)))))# end trace
                         + ((6 / 5) * np.power(g1_val, 2)
                            * ((2 * (mL3_sq_val + mHd_sq_val + mE3_sq_val)
                                * np.power(ytau_val, 2))
                               + (2 * np.power(atau_val, 2))
                               - (4 * M1_val * atau_val
                                  * ytau_val)
                               + (4 * np.power(M1_val, 2)
                                  * np.power(ytau_val, 2))))
                         - ((6 / 5) * np.power(g1_val, 2) * Spr_val)
                         + (33 * np.power(g2_val, 4) * np.power(M2_val, 2))
                         + ((18 / 5) * np.power(g2_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M1_val, 2) + np.power(M2_val, 2)
                               + (M1_val * M2_val)))
                         + ((621 / 25) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + (3 * np.power(g2_val, 2) * sigma2)
                         + ((3 / 5) * np.power(g1_val, 2) * sigma1))

        dmL2_sq_dt_2l = (((-8) * (mL2_sq_val + mHd_sq_val + mE2_sq_val)
                          * np.power(ymu_val, 4))
                         - (np.power(ymu_val, 2)
                            * ((2 * mL2_sq_val) + (2 * mE2_sq_val)
                               + (4 * mHd_sq_val))# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (np.power(ymu_val, 2)# Tr(6(mQ^2 + mD^2)*Yd^2 + 2(mL^2 + mE^2)*Ye^2)
                            * ((6 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + (2 * (((mL3_sq_val + mE3_sq_val)
                                        * np.power(ytau_val, 2))
                                       + ((mL2_sq_val + mE2_sq_val)
                                          * np.power(ymu_val, 2))
                                       + ((mL1_sq_val + mE1_sq_val)
                                          * np.power(ye_val, 2))))# end trace
                               ))
                         - (16 * np.power(ymu_val, 2) * np.power(amu_val, 2))
                         - (np.power(amu_val, 2)# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (np.power(ymu_val, 2)# Tr(6ad^2 + 2ae^2)
                            * ((6 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + (2 * (np.power(atau_val, 2)
                                       + np.power(amu_val, 2)
                                       + np.power(ae_val, 2)))))# end trace
                         - (2 * amu_val * ymu_val# Tr(6Yd*ad + 2Ye*ae)
                            * ((6 * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))
                               + (2 * ((ytau_val * atau_val)
                                       + (ymu_val * amu_val)
                                       + (ye_val * ae_val)))))# end trace
                         + ((6 / 5) * np.power(g1_val, 2)
                            * ((2 * (mL2_sq_val + mHd_sq_val + mE2_sq_val)
                                * np.power(ymu_val, 2))
                               + (2 * np.power(amu_val, 2))
                               - (4 * M1_val * amu_val
                                  * ymu_val)
                               + (4 * np.power(M1_val, 2)
                                  * np.power(ymu_val, 2))))
                         - ((6 / 5) * np.power(g1_val, 2) * Spr_val)
                         + (33 * np.power(g2_val, 4) * np.power(M2_val, 2))
                         + ((18 / 5) * np.power(g2_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M1_val, 2) + np.power(M2_val, 2)
                               + (M1_val * M2_val)))
                         + ((621 / 25) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + (3 * np.power(g2_val, 2) * sigma2)
                         + ((3 / 5) * np.power(g1_val, 2) * sigma1))

        dmL1_sq_dt_2l = (((-8) * (mL1_sq_val + mHd_sq_val + mE1_sq_val)
                          * np.power(ye_val, 4))
                         - (np.power(ye_val, 2)
                            * ((2 * mL1_sq_val) + (2 * mE1_sq_val)
                               + (4 * mHd_sq_val))# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (np.power(ye_val, 2)# Tr(6(mQ^2 + mD^2)*Yd^2 + 2(mL^2 + mE^2)*Ye^2)
                            * ((6 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + (2 * (((mL3_sq_val + mE3_sq_val)
                                        * np.power(ytau_val, 2))
                                       + ((mL2_sq_val + mE2_sq_val)
                                          * np.power(ymu_val, 2))
                                       + ((mL1_sq_val + mE1_sq_val)
                                          * np.power(ye_val, 2))))# end trace
                               ))
                         - (16 * np.power(ye_val, 2) * np.power(ae_val, 2))
                         - (np.power(ae_val, 2)# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (np.power(ye_val, 2)# Tr(6ad^2 + 2ae^2)
                            * ((6 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + (2 * (np.power(atau_val, 2)
                                       + np.power(amu_val, 2)
                                       + np.power(ae_val, 2)))))# end trace
                         - (2 * ae_val * ye_val# Tr(6Yd*ad + 2Ye*ae)
                            * ((6 * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))
                               + (2 * ((ytau_val * atau_val)
                                       + (ymu_val * amu_val)
                                       + (ye_val * ae_val)))))# end trace
                         + ((6 / 5) * np.power(g1_val, 2)
                            * ((2 * (mL1_sq_val + mHd_sq_val + mE1_sq_val)
                                * np.power(ye_val, 2))
                               + (2 * np.power(ae_val, 2))
                               - (4 * M1_val * ae_val
                                  * ye_val)
                               + (4 * np.power(M1_val, 2)
                                  * np.power(ye_val, 2))))
                         - ((6 / 5) * np.power(g1_val, 2) * Spr_val)
                         + (33 * np.power(g2_val, 4) * np.power(M2_val, 2))
                         + ((18 / 5) * np.power(g2_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M1_val, 2) + np.power(M2_val, 2)
                               + (M1_val * M2_val)))
                         + ((621 / 25) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + (3 * np.power(g2_val, 2) * sigma2)
                         + ((3 / 5) * np.power(g1_val, 2) * sigma1))

        # Right up-type squarks
        dmU3_sq_dt_2l = (((-8) * (mQ3_sq_val + mHu_sq_val + mU3_sq_val)
                          * np.power(yt_val, 4))
                         - (4 * (mU3_sq_val + mHu_sq_val + mHd_sq_val
                                 + (2 * mQ3_sq_val) + mD3_sq_val)
                            * np.power(yb_val, 2) * np.power(yt_val, 2))
                         - (np.power(yt_val, 2)
                            * ((2 * mQ3_sq_val) + (2 * mU3_sq_val)
                               + (4 * mHu_sq_val))# Tr(6Yu^2)
                            * 6 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                         - (12 * np.power(yt_val, 2)# Tr((mQ^2 + mU^2)*Yu^2)
                            * (((mQ3_sq_val + mU3_sq_val)
                                * np.power(yt_val, 2))
                               + ((mQ2_sq_val + mU2_sq_val)
                                  * np.power(yc_val, 2))
                               + ((mQ1_sq_val + mU1_sq_val)
                                  * np.power(yu_val, 2))))# end trace
                         - (16 * np.power(yt_val, 2) * np.power(at_val, 2))
                         - (16 * at_val * ab_val * yb_val * yt_val)
                         - (12 * ((np.power(at_val, 2)# Tr(Yu^2)
                                   * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                      + np.power(yu_val, 2)))# end trace
                                  + (np.power(yt_val, 2) # Tr(au^2)
                                     * (np.power(at_val, 2)
                                        + np.power(ac_val, 2)
                                        + np.power(au_val, 2)))# end trace
                                  + (at_val * yt_val * 2# Tr(Yu*au)
                                     * ((yt_val * at_val) + (yc_val * ac_val)
                                        + (yu_val * au_val)))))# end trace
                         + (((6 * np.power(g2_val, 2))
                             - ((2 / 5) * np.power(g1_val, 2)))
                            * ((2 * (mQ3_sq_val + mHu_sq_val + mU3_sq_val)
                                * np.power(yt_val, 2))
                               + (2 * np.power(at_val, 2))))
                         + (12 * np.power(g2_val, 2)
                            * 2 * ((np.power(M2_val, 2) * np.power(yt_val, 2))
                                   - (M2_val * at_val * yt_val)))
                         - ((4 / 5) * np.power(g1_val, 2)
                            * 2 * ((np.power(M1_val, 2) * np.power(yt_val, 2))
                                   - (M1_val * at_val * yt_val)))
                         - ((8 / 5) * np.power(g1_val, 2) * Spr_val)
                         - ((128 / 3) * np.power(g3_val, 4)
                            * np.power(M3_val, 2))
                         + ((512 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M3_val, 2) + np.power(M1_val, 2)
                               + (M3_val * M1_val)))
                         + ((3424 / 75) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + ((16 / 3) * np.power(g3_val, 2) * sigma3)
                         + ((16 / 15) * np.power(g1_val, 2) * sigma1))

        dmU2_sq_dt_2l = (((-8) * (mQ2_sq_val + mHu_sq_val + mU2_sq_val)
                          * np.power(yc_val, 4))
                         - (4 * (mU2_sq_val + mHu_sq_val + mHd_sq_val
                                 + (2 * mQ2_sq_val)
                                 + mD2_sq_val)
                            * np.power(ys_val, 2) * np.power(yc_val, 2))
                         - (np.power(yc_val, 2)
                            * ((2 * mQ2_sq_val) + (2 * mU2_sq_val)
                               + (4 * mHu_sq_val))# Tr(6Yu^2)
                            * 6 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                         - (12 * np.power(yc_val, 2)# Tr((mQ^2 + mU^2)*Yu^2)
                            * (((mQ3_sq_val + mU3_sq_val)
                                * np.power(yt_val, 2))
                               + ((mQ2_sq_val + mU2_sq_val)
                                  * np.power(yc_val, 2))
                               + ((mQ1_sq_val + mU1_sq_val)
                                  * np.power(yu_val, 2))))# end trace
                         - (16 * np.power(yc_val, 2) * np.power(ac_val, 2))
                         - (16 * ac_val * as_val * ys_val * yc_val)
                         - (12 * ((np.power(ac_val, 2)# Tr(Yu^2)
                                   * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                      + np.power(yu_val, 2)))# end trace
                                  + (np.power(yc_val, 2) # Tr(au^2)
                                     * (np.power(at_val, 2)
                                        + np.power(ac_val, 2)
                                        + np.power(au_val, 2)))# end trace
                                  + (ac_val * yc_val * 2# Tr(Yu*au)
                                     * ((yt_val * at_val) + (yc_val * ac_val)
                                        + (yu_val * au_val)))))# end trace
                         + (((6 * np.power(g2_val, 2))
                             - ((2 / 5) * np.power(g1_val, 2)))
                            * ((2 * (mQ2_sq_val + mHu_sq_val + mU2_sq_val)
                                * np.power(yc_val, 2))
                               + (2 * np.power(ac_val, 2))))
                         + (12 * np.power(g2_val, 2)
                            * 2 * ((np.power(M2_val, 2) * np.power(yc_val, 2))
                                   - (M2_val * ac_val * yc_val)))
                         - ((4 / 5) * np.power(g1_val, 2)
                            * 2 * ((np.power(M1_val, 2) * np.power(yc_val, 2))
                                   - (M1_val * ac_val * yc_val)))
                         - ((8 / 5) * np.power(g1_val, 2) * Spr_val)
                         - ((128 / 3) * np.power(g3_val, 4)
                            * np.power(M3_val, 2))
                         + ((512 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M3_val, 2) + np.power(M1_val, 2)
                               + (M3_val * M1_val)))
                         + ((3424 / 75) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + ((16 / 3) * np.power(g3_val, 2) * sigma3)
                         + ((16 / 15) * np.power(g1_val, 2) * sigma1))

        dmU1_sq_dt_2l = (((-8) * (mQ1_sq_val + mHu_sq_val + mU1_sq_val)
                          * np.power(yu_val, 4))
                         - (4 * (mU1_sq_val + mHu_sq_val + mHd_sq_val
                                 + (2 * mQ1_sq_val)
                                 + mD1_sq_val)
                            * np.power(yd_val, 2) * np.power(yu_val, 2))
                         - (np.power(yu_val, 2)
                            * ((2 * mQ1_sq_val) + (2 * mU1_sq_val)
                               + (4 * mHu_sq_val))# Tr(6Yu^2)
                            * 6 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                         - (12 * np.power(yu_val, 2)# Tr((mQ^2 + mU^2)*Yu^2)
                            * (((mQ3_sq_val + mU3_sq_val)
                                * np.power(yt_val, 2))
                               + ((mQ2_sq_val + mU2_sq_val)
                                  * np.power(yc_val, 2))
                               + ((mQ1_sq_val + mU1_sq_val)
                                  * np.power(yu_val, 2))))# end trace
                         - (16 * np.power(yu_val, 2) * np.power(au_val, 2))
                         - (16 * au_val * ad_val * yd_val * yu_val)
                         - (12 * ((np.power(au_val, 2)# Tr(Yu^2)
                                   * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                      + np.power(yu_val, 2)))# end trace
                                  + (np.power(yu_val, 2) # Tr(au^2)
                                     * (np.power(at_val, 2)
                                        + np.power(ac_val, 2)
                                        + np.power(au_val, 2)))# end trace
                                  + (au_val * yu_val * 2# Tr(Yu*au)
                                     * ((yt_val * at_val) + (yc_val * ac_val)
                                        + (yu_val * au_val)))))# end trace
                         + (((6 * np.power(g2_val, 2))
                             - ((2 / 5) * np.power(g1_val, 2)))
                            * ((2 * (mQ1_sq_val + mHu_sq_val + mU1_sq_val)
                                * np.power(yu_val, 2))
                               + (2 * np.power(au_val, 2))))
                         + (12 * np.power(g2_val, 2)
                            * 2 * ((np.power(M2_val, 2) * np.power(yu_val, 2))
                                   - (M2_val * au_val * yu_val)))
                         - ((4 / 5) * np.power(g1_val, 2)
                            * 2 * ((np.power(M1_val, 2) * np.power(yu_val, 2))
                                   - (M1_val * au_val * yu_val)))
                         - ((8 / 5) * np.power(g1_val, 2) * Spr_val)
                         - ((128 / 3) * np.power(g3_val, 4)
                            * np.power(M3_val, 2))
                         + ((512 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M3_val, 2) + np.power(M1_val, 2)
                               + (M3_val * M1_val)))
                         + ((3424 / 75) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + ((16 / 3) * np.power(g3_val, 2) * sigma3)
                         + ((16 / 15) * np.power(g1_val, 2) * sigma1))

        # Right down-type squarks
        dmD3_sq_dt_2l = (((-8) * (mQ3_sq_val + mHd_sq_val + mD3_sq_val)
                          * np.power(yb_val, 4))
                         - (4 * (mU3_sq_val + mHu_sq_val + mHd_sq_val
                                 + (2 * mQ3_sq_val)
                                 + mD3_sq_val) * np.power(yb_val, 2)
                            * np.power(yt_val, 2))
                         - (np.power(yb_val, 2)
                            * (2 * (mD3_sq_val + mQ3_sq_val
                                    + (2 * mHd_sq_val)))# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (4 * np.power(yb_val, 2) # Tr(3(mQ^2 + mD^2) * Yd^2 + (mL^2 + mE^2) * Ye^2)
                            * ((3 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + (((mL3_sq_val + mE3_sq_val)
                                   * np.power(ytau_val, 2))
                                  + ((mL2_sq_val + mE2_sq_val)
                                     * np.power(ymu_val, 2))
                                  + ((mL1_sq_val + mE1_sq_val)
                                     * np.power(ye_val, 2)))# end trace
                               ))
                         - (16 * np.power(yb_val, 2) * np.power(ab_val, 2))
                         - (16 * at_val * ab_val * yb_val * yt_val)
                         - (4 * np.power(ab_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (4 * np.power(yb_val, 2) # Tr(3ad^2 + ae^2)
                            * ((3 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + np.power(atau_val, 2) + np.power(amu_val, 2)
                               + np.power(ae_val, 2)))# end trace
                         - (8 * ab_val * yb_val # Tr(3Yd * ad + Ye * ae)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         + (((6 * np.power(g2_val, 2))
                             + ((2 / 5) * np.power(g1_val, 2)))
                            * ((2 * (mQ3_sq_val + mHd_sq_val + mD3_sq_val)
                                * np.power(yb_val, 2))
                               + (2 * np.power(ab_val, 2))))
                         + (12 * np.power(g2_val, 2)
                            * 2 * ((np.power(M2_val, 2) * np.power(yb_val, 2))
                                   - (M2_val * ab_val * yb_val)))
                         + ((4 / 5) * np.power(g1_val, 2)
                            * 2 * ((np.power(M1_val, 2) * np.power(yb_val, 2))
                                   - (M1_val * ab_val * yb_val)))
                         + ((4 / 5) * np.power(g1_val, 2) * Spr_val)
                         - ((128 / 3) * np.power(g3_val, 4)
                            * np.power(M3_val, 2))
                         + ((128 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M3_val, 2) + np.power(M1_val, 2)
                               + (M3_val * M1_val)))
                         + ((808 / 75) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + ((16 / 3) * np.power(g3_val, 2) * sigma3)
                         + ((4 / 15) * np.power(g1_val, 2) * sigma1))

        dmD2_sq_dt_2l = (((-8) * (mQ2_sq_val + mHd_sq_val + mD2_sq_val)
                          * np.power(ys_val, 4))
                         - (4 * (mU2_sq_val + mHu_sq_val + mHd_sq_val
                                 + (2 * mQ2_sq_val)
                                 + mD2_sq_val) * np.power(ys_val, 2)
                            * np.power(yc_val, 2))
                         - (np.power(ys_val, 2)
                            * (2 * (mD2_sq_val + mQ2_sq_val
                                    + (2 * mHd_sq_val)))# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (4 * np.power(ys_val, 2) # Tr(3(mQ^2 + mD^2) * Yd^2 + (mL^2 + mE^2) * Ye^2)
                            * ((3 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + (((mL3_sq_val + mE3_sq_val)
                                   * np.power(ytau_val, 2))
                                  + ((mL2_sq_val + mE2_sq_val)
                                     * np.power(ymu_val, 2))
                                  + ((mL1_sq_val + mE1_sq_val)
                                     * np.power(ye_val, 2)))# end trace
                               ))
                         - (16 * np.power(ys_val, 2) * np.power(as_val, 2))
                         - (16 * ac_val * as_val * ys_val * yc_val)
                         - (4 * np.power(as_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (4 * np.power(ys_val, 2) # Tr(3ad^2 + ae^2)
                            * ((3 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + np.power(atau_val, 2) + np.power(amu_val, 2)
                               + np.power(ae_val, 2)))# end trace
                         - (8 * as_val * ys_val # Tr(3Yd * ad + Ye * ae)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         + (((6 * np.power(g2_val, 2))
                             + ((2 / 5) * np.power(g1_val, 2)))
                            * ((2 * (mQ2_sq_val + mHd_sq_val + mD2_sq_val)
                                * np.power(ys_val, 2))
                               + (2 * np.power(as_val, 2))))
                         + (12 * np.power(g2_val, 2)
                            * 2 * ((np.power(M2_val, 2) * np.power(ys_val, 2))
                                   - (M2_val * as_val * ys_val)))
                         + ((4 / 5) * np.power(g1_val, 2)
                            * 2 * ((np.power(M1_val, 2) * np.power(ys_val, 2))
                                   - (M1_val * as_val * ys_val)))
                         + ((4 / 5) * np.power(g1_val, 2) * Spr_val)
                         - ((128 / 3) * np.power(g3_val, 4)
                            * np.power(M3_val, 2))
                         + ((128 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M3_val, 2) + np.power(M1_val, 2)
                               + (M3_val * M1_val)))
                         + ((808 / 75) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + ((16 / 3) * np.power(g3_val, 2) * sigma3)
                         + ((4 / 15) * np.power(g1_val, 2) * sigma1))

        dmD1_sq_dt_2l = (((-8) * (mQ1_sq_val + mHd_sq_val + mD1_sq_val)
                          * np.power(yd_val, 4))
                         - (4 * (mU1_sq_val + mHu_sq_val + mHd_sq_val
                                 + (2 * mQ1_sq_val)
                                 + mD1_sq_val) * np.power(yd_val, 2)
                            * np.power(yu_val, 2))
                         - (np.power(yd_val, 2)
                            * (2 * (mD1_sq_val + mQ1_sq_val
                                    + (2 * mHd_sq_val)))# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (4 * np.power(yd_val, 2) # Tr(3(mQ^2 + mD^2) * Yd^2 + (mL^2 + mE^2) * Ye^2)
                            * ((3 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + (((mL3_sq_val + mE3_sq_val)
                                   * np.power(ytau_val, 2))
                                  + ((mL2_sq_val + mE2_sq_val)
                                     * np.power(ymu_val, 2))
                                  + ((mL1_sq_val + mE1_sq_val)
                                     * np.power(ye_val, 2)))# end trace
                               ))
                         - (16 * np.power(yd_val, 2) * np.power(ad_val, 2))
                         - (16 * au_val * ad_val * yd_val * yu_val)
                         - (4 * np.power(ad_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (4 * np.power(yd_val, 2) # Tr(3ad^2 + ae^2)
                            * ((3 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + np.power(atau_val, 2) + np.power(amu_val, 2)
                               + np.power(ae_val, 2)))# end trace
                         - (8 * ad_val * yd_val # Tr(3Yd * ad + Ye * ae)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         + (((6 * np.power(g2_val, 2))
                             + ((2 / 5) * np.power(g1_val, 2)))
                            * ((2 * (mQ1_sq_val + mHd_sq_val + mD1_sq_val)
                                * np.power(yd_val, 2))
                               + (2 * np.power(ad_val, 2))))
                         + (12 * np.power(g2_val, 2)
                            * 2 * ((np.power(M2_val, 2) * np.power(yd_val, 2))
                                   - (M2_val * ad_val * yd_val)))
                         + ((4 / 5) * np.power(g1_val, 2)
                            * 2 * ((np.power(M1_val, 2) * np.power(yd_val, 2))
                                   - (M1_val * ad_val * yd_val)))
                         + ((4 / 5) * np.power(g1_val, 2) * Spr_val)
                         - ((128 / 3) * np.power(g3_val, 4)
                            * np.power(M3_val, 2))
                         + ((128 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M3_val, 2) + np.power(M1_val, 2)
                               + (M3_val * M1_val)))
                         + ((808 / 75) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + ((16 / 3) * np.power(g3_val, 2) * sigma3)
                         + ((4 / 15) * np.power(g1_val, 2) * sigma1))

        # Right leptons
        dmE3_sq_dt_2l = (((-8) * (mL3_sq_val + mHd_sq_val + mE3_sq_val)
                         * np.power(ytau_val, 4))
            - (np.power(ytau_val, 2)
               * ((2 * mL3_sq_val) + (2 * mE3_sq_val)
                  + (4 * mHd_sq_val))# Tr(6Yd^2 + 2Ye^2)
               * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                        + np.power(yd_val, 2)))
                  + (2 * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                          + np.power(ye_val, 2)))))# end trace
            - (4 * np.power(ytau_val, 2) # Tr(3(mQ^2 + mD^2) * Yd^2 + (mL^2 + mE^2) * Ye^2)
               * ((3 * (((mQ3_sq_val + mD3_sq_val) * np.power(yb_val, 2))
                        + ((mQ2_sq_val + mD2_sq_val) * np.power(ys_val, 2))
                        + ((mQ1_sq_val + mD1_sq_val) * np.power(yd_val, 2))))
                  + ((((mL3_sq_val + mE3_sq_val) * np.power(ytau_val, 2))
                      + ((mL2_sq_val + mE2_sq_val) * np.power(ymu_val, 2))
                      + ((mL1_sq_val + mE1_sq_val) * np.power(ye_val, 2))))# end trace
                  ))
            - (16 * np.power(ytau_val, 2) * np.power(atau_val, 2))
            - (4 * np.power(atau_val, 2)# Tr(3Yd^2 + Ye^2)
               * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                        + np.power(yd_val, 2)))
                  + ((np.power(ytau_val, 2) + np.power(ymu_val, 2)
                      + np.power(ye_val, 2)))))# end trace
            - (4 * np.power(ytau_val, 2) # Tr(3ad^2 + ae^2)
               * ((3 * (np.power(ab_val, 2) + np.power(as_val, 2)
                        + np.power(ad_val, 2)))
                  + ((np.power(atau_val, 2) + np.power(amu_val, 2)
                      + np.power(ae_val, 2)))))# end trace
            - (8 * atau_val * ytau_val # Tr(3Yd * ad + Ye * ae)
               * ((3 * ((yb_val * ab_val) + (ys_val * as_val)
                         + (yd_val * ad_val)))
                  + (((ytau_val * atau_val) + (ymu_val * amu_val)
                      + (ye_val * ae_val)))))# end trace
            + (((6 * np.power(g2_val, 2)) - (6 / 5) * np.power(g1_val, 2))
               * ((2 * (mL3_sq_val + mHd_sq_val + mE3_sq_val)
                   * np.power(ytau_val, 2))
                  + (2 * np.power(atau_val, 2))))
            + (12 * np.power(g2_val, 2) * 2
               * ((np.power(M2_val, 2) * np.power(ytau_val, 2))
                  - (M2_val * atau_val * ytau_val)))
            - ((12 / 5) * np.power(g1_val, 2) * 2
               * ((np.power(M1_val, 2) * np.power(ytau_val, 2))
                  - (M1_val * atau_val * ytau_val)))
            + ((12 / 5) * np.power(g1_val, 2) * Spr_val)
            + ((2808 / 25) * np.power(g1_val, 4) * np.power(M1_val, 2))
            + ((12 / 5) * np.power(g1_val, 2) * sigma1))

        dmE2_sq_dt_2l = (((-8) * (mL2_sq_val + mHd_sq_val + mE2_sq_val)
                          * np.power(ymu_val, 4))
                         - (np.power(ymu_val, 2)
                            * ((2 * mL2_sq_val) + (2 * mE2_sq_val)
                               + (4 * mHd_sq_val))# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (4 * np.power(ymu_val, 2) # Tr(3(mQ^2 + mD^2) * Yd^2 + (mL^2 + mE^2) * Ye^2)
                            * ((3 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + ((((mL3_sq_val + mE3_sq_val)
                                    * np.power(ytau_val, 2))
                                   + ((mL2_sq_val + mE2_sq_val)
                                      * np.power(ymu_val, 2))
                                   + ((mL1_sq_val + mE1_sq_val)
                                      * np.power(ye_val, 2))))# end trace
                               ))
                         - (16 * np.power(ymu_val, 2) * np.power(amu_val, 2))
                         - (4 * np.power(amu_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + ((np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                   + np.power(ye_val, 2)))))# end trace
                         - (4 * np.power(ymu_val, 2) # Tr(3ad^2 + ae^2)
                            * ((3 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + ((np.power(atau_val, 2) + np.power(amu_val, 2)
                                   + np.power(ae_val, 2)))))# end trace
                         - (8 * amu_val * ymu_val # Tr(3Yd * ad + Ye * ae)
                            * ((3 * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))
                               + (((ytau_val * atau_val) + (ymu_val * amu_val)
                                   + (ye_val * ae_val)))))# end trace
                         + (((6 * np.power(g2_val, 2))
                             - (6 / 5) * np.power(g1_val, 2))
                            * ((2 * (mL2_sq_val + mHd_sq_val + mE2_sq_val)
                                * np.power(ymu_val, 2))
                               + (2 * np.power(amu_val, 2))))
                         + (12 * np.power(g2_val, 2) * 2
                            * ((np.power(M2_val, 2) * np.power(ymu_val, 2))
                               - (M2_val * amu_val * ymu_val)))
                         - ((12 / 5) * np.power(g1_val, 2) * 2
                            * ((np.power(M1_val, 2) * np.power(ymu_val, 2))
                               - (M1_val * amu_val * ymu_val)))
                         + ((12 / 5) * np.power(g1_val, 2) * Spr_val)
                         + ((2808 / 25) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + ((12 / 5) * np.power(g1_val, 2) * sigma1))

        dmE1_sq_dt_2l = (((-8) * (mL1_sq_val + mHd_sq_val + mE1_sq_val)
                          * np.power(ye_val, 4))
                         - (np.power(ye_val, 2)
                            * ((2 * mL1_sq_val) + (2 * mE1_sq_val)
                               + (4 * mHd_sq_val))# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (4 * np.power(ye_val, 2) # Tr(3(mQ^2 + mD^2) * Yd^2 + (mL^2 + mE^2) * Ye^2)
                            * ((3 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + ((((mL3_sq_val + mE3_sq_val)
                                    * np.power(ytau_val, 2))
                                   + ((mL2_sq_val + mE2_sq_val)
                                      * np.power(ymu_val, 2))
                                   + ((mL1_sq_val + mE1_sq_val)
                                      * np.power(ye_val, 2))))# end trace
                               ))
                         - (16 * np.power(ye_val, 2) * np.power(ae_val, 2))
                         - (4 * np.power(ae_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + ((np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                   + np.power(ye_val, 2)))))# end trace
                         - (4 * np.power(ye_val, 2) # Tr(3ad^2 + ae^2)
                            * ((3 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + ((np.power(atau_val, 2) + np.power(amu_val, 2)
                                   + np.power(ae_val, 2)))))# end trace
                         - (8 * ae_val * ye_val # Tr(3Yd * ad + Ye * ae)
                            * ((3 * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))
                               + (((ytau_val * atau_val) + (ymu_val * amu_val)
                                   + (ye_val * ae_val)))))# end trace
                         + (((6 * np.power(g2_val, 2))
                             - (6 / 5) * np.power(g1_val, 2))
                            * ((2 * (mL1_sq_val + mHd_sq_val + mE1_sq_val)
                                * np.power(ye_val, 2))
                               + (2 * np.power(ae_val, 2))))
                         + (12 * np.power(g2_val, 2) * 2
                            * ((np.power(M2_val, 2) * np.power(ye_val, 2))
                               - (M2_val * ae_val * ye_val)))
                         - ((12 / 5) * np.power(g1_val, 2) * 2
                            * ((np.power(M1_val, 2) * np.power(ye_val, 2))
                               - (M1_val * ae_val * ye_val)))
                         + ((12 / 5) * np.power(g1_val, 2) * Spr_val)
                         + ((2808 / 25) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + ((12 / 5) * np.power(g1_val, 2) * sigma1))

        # Total scalar squared mass beta functions
        dmQ3_sq_dt = (1 / t) * ((loop_fac * dmQ3_sq_dt_1l)
                                + (loop_fac_sq * dmQ3_sq_dt_2l))

        dmQ2_sq_dt = (1 / t) * ((loop_fac * dmQ2_sq_dt_1l)
                                + (loop_fac_sq * dmQ2_sq_dt_2l))

        dmQ1_sq_dt = (1 / t) * ((loop_fac * dmQ1_sq_dt_1l)
                                + (loop_fac_sq * dmQ1_sq_dt_2l))

        dmL3_sq_dt = (1 / t) * ((loop_fac * dmL3_sq_dt_1l)
                                + (loop_fac_sq * dmL3_sq_dt_2l))

        dmL2_sq_dt = (1 / t) * ((loop_fac * dmL2_sq_dt_1l)
                                + (loop_fac_sq * dmL2_sq_dt_2l))

        dmL1_sq_dt = (1 / t) * ((loop_fac * dmL1_sq_dt_1l)
                                + (loop_fac_sq * dmL1_sq_dt_2l))

        dmU3_sq_dt = (1 / t) * ((loop_fac * dmU3_sq_dt_1l)
                                + (loop_fac_sq * dmU3_sq_dt_2l))

        dmU2_sq_dt = (1 / t) * ((loop_fac * dmU2_sq_dt_1l)
                                + (loop_fac_sq * dmU2_sq_dt_2l))

        dmU1_sq_dt = (1 / t) * ((loop_fac * dmU1_sq_dt_1l)
                                + (loop_fac_sq * dmU1_sq_dt_2l))

        dmD3_sq_dt = (1 / t) * ((loop_fac * dmD3_sq_dt_1l)
                                + (loop_fac_sq * dmD3_sq_dt_2l))

        dmD2_sq_dt = (1 / t) * ((loop_fac * dmD2_sq_dt_1l)
                                + (loop_fac_sq * dmD2_sq_dt_2l))

        dmD1_sq_dt = (1 / t) * ((loop_fac * dmD1_sq_dt_1l)
                                + (loop_fac_sq * dmD1_sq_dt_2l))

        dmE3_sq_dt = (1 / t) * ((loop_fac * dmE3_sq_dt_1l)
                                + (loop_fac_sq * dmE3_sq_dt_2l))

        dmE2_sq_dt = (1 / t) * ((loop_fac * dmE2_sq_dt_1l)
                                + (loop_fac_sq * dmE2_sq_dt_2l))

        dmE1_sq_dt = (1 / t) * ((loop_fac * dmE1_sq_dt_1l)
                                + (loop_fac_sq * dmE1_sq_dt_2l))

        ##### Tanb RGE from arXiv:hep-ph/0112251 in R_xi=1 Feynman gauge #####
        # 1 loop part
        dtanb_dt_1l = 3 * (np.power(yt_val, 2) - np.power(yb_val, 2))

        # 2 loop part
        dtanb_dt_2l = (((-9) * (np.power(yt_val, 4) - np.power(yb_val, 4)))
                      + (6 * np.power(yt_val, 2)
                          * (((8 / 3) * np.power(g3_val, 2))
                            + ((6 / 45) * np.power(g1_val, 2))))
                      - (6 * np.power(yb_val, 2)
                          * (((8 / 3) * np.power(g3_val, 2))
                            - ((3 / 45) * np.power(g1_val, 2))))
                      - (3 * (np.power(yt_val, 2) - np.power(yb_val, 2))
                          * (((1 / np.sqrt(2))
                              * (((3 / 5) * np.power(g1_val, 2))
                                + np.power(g2_val, 2)))
                            + np.power(g2_val, 2))))

        # Total beta function for tanb
        dtanb_dt = (tanb_val / t) * ((loop_fac * dtanb_dt_1l)
                                    + (loop_fac_sq * dtanb_dt_2l))


        # Collect all for return
        dxdt = [dg1_dt, dg2_dt, dg3_dt, dM1_dt, dM2_dt, dM3_dt, dmu_dt, dyt_dt,
                dyc_dt, dyu_dt, dyb_dt, dys_dt, dyd_dt, dytau_dt, dymu_dt,
                dye_dt, dat_dt, dac_dt, dau_dt, dab_dt, das_dt, dad_dt,
                datau_dt, damu_dt, dae_dt, dmHu_sq_dt, dmHd_sq_dt,
                dmQ1_sq_dt, dmQ2_sq_dt, dmQ3_sq_dt, dmL1_sq_dt, dmL2_sq_dt,
                dmL3_sq_dt, dmU1_sq_dt, dmU2_sq_dt, dmU3_sq_dt, dmD1_sq_dt,
                dmD2_sq_dt, dmD3_sq_dt, dmE1_sq_dt, dmE2_sq_dt, dmE3_sq_dt,
                db_dt, dtanb_dt]
        return dxdt

    numpoints = int((np.log10(GUT_Q / lowQ_val)) * 1000)
    t_vals = np.logspace(np.log10(GUT_Q), np.log10(lowQ_val),
                         numpoints)
    t_vals[0] = GUT_Q
    t_vals[-1] = lowQ_val
    t_span = np.array([GUT_Q, lowQ_val])

    # Now solve down to low scale
    sol = solve_ivp(my_odes_rundown, t_span, inpGUTBCs, t_eval = t_vals,
                    dense_output=True, method='DOP853', atol=1e-9,
                    rtol=1e-9)
    myx1 = sol.y
    weakvals = [myx1[0][-1], myx1[1][-1], myx1[2][-1], myx1[3][-1],
                myx1[4][-1], myx1[5][-1], myx1[6][-1], myx1[7][-1],
                myx1[8][-1], myx1[9][-1], myx1[10][-1], myx1[11][-1],
                myx1[12][-1], myx1[13][-1], myx1[14][-1], myx1[15][-1],
                myx1[16][-1], myx1[17][-1], myx1[18][-1], myx1[19][-1],
                myx1[20][-1], myx1[21][-1], myx1[22][-1], myx1[23][-1],
                myx1[24][-1], myx1[25][-1], myx1[26][-1], myx1[27][-1],
                myx1[28][-1], myx1[29][-1], myx1[30][-1], myx1[31][-1],
                myx1[32][-1], myx1[33][-1], myx1[34][-1], myx1[35][-1],
                myx1[36][-1], myx1[37][-1], myx1[38][-1], myx1[39][-1],
                myx1[40][-1], myx1[41][-1], myx1[42][-1], myx1[43][-1]]
    return weakvals

def my_radcorr_calc(myQ, vHiggs_wk, mu_wk,
                    beta_wk, yt_wk, yc_wk, yu_wk, yb_wk, ys_wk, yd_wk,
                    ytau_wk, ymu_wk, ye_wk, g1_wk, g2_wk,
                    g3_wk, mQ3_sq_wk,
                    mQ2_sq_wk, mQ1_sq_wk,
                    mL3_sq_wk, mL2_sq_wk,
                    mL1_sq_wk, mU3_sq_wk,
                    mU2_sq_wk,
                    mU1_sq_wk, mD3_sq_wk,
                    mD2_sq_wk, mD1_sq_wk,
                    mE3_sq_wk, mE2_sq_wk,
                    mE1_sq_wk, M1_wk, M2_wk,
                    M3_wk, mHu_sq_wk,
                    mHd_sq_wk, at_wk, ac_wk, au_wk, ab_wk, as_wk, ad_wk,
                    atau_wk, amu_wk, ae_wk):
    """
    Compute 1-loop and some 2-loop radiative corrections to Higgs scalar
    potential for evaluation of b=B*mu soft SUSY-breaking bilinear parameter
    boundary value at weak scale.
    Parameters
    ----------
    myQ: Float.
        Renormalization scale for evaluation of radiative corrections.
    vHiggs_wk : Float.
        Weak-scale Higgs VEV.
    mu_wk : Float.
        Weak-scale Higgsino mass parameter mu.
    beta_wk : Float.
        Higgs mixing angle beta at the weak scale (from ratio of Higgs VEVs).
    yt_wk : Float.
        Weak-scale top Yukawa coupling.
    yc_wk : Float.
        Weak-scale charm Yukawa coupling.
    yu_wk : Float.
        Weak-scale up Yukawa coupling.
    yb_wk : Float.
        Weak-scale bottom Yukawa coupling.
    ys_wk : Float.
        Weak-scale strange Yukawa coupling.
    yd_wk : Float.
        Weak-scale down Yukawa coupling.
    ytau_wk : Float.
        Weak-scale tau Yukawa coupling.
    ymu_wk : Float.
        Weak-scale mu Yukawa coupling.
    ye_wk : Float.
        Weak-scale electron Yukawa coupling.
    g1_wk : Float.
        Weak-scale U(1) gauge coupling.
    g2_wk : Float.
        Weak-scale SU(2) gauge coupling.
    g3_wk : Float.
        Weak-scale SU(3) gauge coupling.
    mQ3_sq_wk : Float.
        Weak-scale 3rd gen left squark squared mass.
    mQ2_sq_wk : Float.
        Weak-scale 2nd gen left squark squared mass.
    mQ1_sq_wk : Float.
        Weak-scale 1st gen left squark squared mass.
    mL3_sq_wk : Float.
        Weak-scale 3rd gen left slepton squared mass.
    mL2_sq_wk : Float.
        Weak-scale 2nd gen left slepton squared mass.
    mL1_sq_wk : Float.
        Weak-scale 1st gen left slepton squared mass.
    mU3_sq_wk : Float.
        Weak-scale 3rd gen right up-type squark squared mass.
    mU2_sq_wk : Float.
        Weak-scale 2nd gen right up-type squark squared mass.
    mU1_sq_wk : Float.
        Weak-scale 1st gen right up-type squark squared mass.
    mD3_sq_wk : Float.
        Weak-scale 3rd gen right down-type squark squared mass.
    mD2_sq_wk : Float.
        Weak-scale 2nd gen right down-type squark squared mass.
    mD1_sq_wk : Float.
        Weak-scale 1st gen right down-type squark squared mass.
    mE3_sq_wk : Float.
        Weak-scale 3rd gen right slepton squared mass.
    mE2_sq_wk : Float.
        Weak-scale 2nd gen right slepton squared mass.
    mE1_sq_wk : Float.
        Weak-scale 1st gen right slepton squared mass.
    M1_wk : Float.
        Weak-scale bino mass parameter.
    M2_wk : Float.
        Weak-scale wino mass parameter.
    M3_wk : Float.
        Weak-scale gluino mass parameter.
    mHu_sq_wk : Float.
        Weak-scale up-type soft Higgs mass parameter.
    mHd_sq_wk : Float.
        Weak-scale down-type soft Higgs mass parameter.
    at_wk : Float.
        Weak-scale reduced top soft trilinear coupling.
    ac_wk : Float.
        Weak-scale reduced charm soft trilinear coupling.
    au_wk : Float.
        Weak-scale reduced up soft trilinear coupling.
    ab_wk : Float.
        Weak-scale reduced bottom soft trilinear coupling.
    as_wk : Float.
        Weak-scale reduced strange soft trilinear coupling.
    ad_wk : Float.
        Weak-scale reduced down soft trilinear coupling.
    atau_wk : Float.
        Weak-scale reduced tau soft trilinear coupling.
    amu_wk : Float.
        Weak-scale reduced mu soft trilinear coupling.
    ae_wk : Float.
        Weak-scale reduced electron soft trilinear coupling.
    Returns
    -------
    my_radcorrs : Array of floats.
        Individual and total radiative corrections of the types uu, dd, and ud.
        Return 42 Sigma_u^u corrections, 42 Sigma_d^d corrections, and 41
        Sigma_u^d corrections.
    """
    gpr_wk = g1_wk * np.sqrt(3. / 5.)
    gpr_sq = np.power(gpr_wk, 2)
    g2_sq = np.power(g2_wk, 2)
    mu_wk_sq = np.power(mu_wk, 2)

    ##### Fundamental equations: #####

    def logfunc(mass, Q_renorm_sq=np.power(myQ, 2)):
        """
        Return F = m^2 * (ln(m^2 / Q^2) - 1), where input mass term is linear.

        Parameters
        ----------
        mass : Float.
            Input mass to be evaluated.
        Q_renorm_sq : Float.
            Squared renormalization scale, read in from supplied SLHA file.

        Returns
        -------
        myf : Float.
            Return F = m^2 * (ln(m^2 / Q^2) - 1),
            where input mass term is linear.

        """
        myf = np.power(mass, 2) * (np.log((np.power(mass, 2))
                                          / (Q_renorm_sq)) - 1)
        return myf


    def logfunc2(masssq, Q_renorm_sq=np.power(myQ, 2)):
        """
        Return F = m^2 * (ln(m^2 / Q^2) - 1), where input mass term is
        quadratic.

        Parameters
        ----------
        mass : Float.
            Input mass to be evaluated.
        Q_renorm_sq : Float.
            Squared renormalization scale, read in from supplied SLHA file.

        Returns
        -------
        myf : Float.
            Return F = m^2 * (ln(m^2 / Q^2) - 1),
            where input mass term is quadratic.

        """
        #if masssq < 0:
        #    print("Warning! Negative mass squared aside from mHu^2(weak)"
        #          + " encountered.")
        myf2 = masssq * (np.log(np.abs((masssq) / (Q_renorm_sq))) - 1)
        return myf2


    sinsqb = np.power(np.sin(beta_wk), 2)
    cossqb = np.power(np.cos(beta_wk), 2)
    vu = vHiggs_wk * np.sqrt(sinsqb)
    vd = vHiggs_wk * np.sqrt(cossqb)
    vu_sq = np.power(vu, 2)
    vd_sq = np.power(vd, 2)
    v_sq = np.power(vHiggs_wk, 2)
    tan_th_w = gpr_wk / g2_wk
    theta_w = np.arctan(tan_th_w)
    sinsq_th_w = np.power(np.sin(theta_w), 2)
    cos2b = np.cos(2 * beta_wk)
    sin2b = np.sin(2 * beta_wk)
    gz_sq = (np.power(g2_wk, 2) + np.power(gpr_wk, 2)) / 8

    ##### Mass relations: #####

    # W-boson tree-level running squared mass
    m_w_sq = (np.power(g2_wk, 2) / 2) * v_sq
    #print("mW^2: " + str(m_w_sq))

    # Z-boson tree-level running squared mass
    mz_q_sq = v_sq * ((np.power(g2_wk, 2) + np.power(gpr_wk, 2)) / 2)
    #print("mZ^2: " + str(mz_q_sq))
    #print("mZ: " + str(np.sqrt(mz_q_sq)))

    # Higgs psuedoscalar tree-level running squared mass
    mA0sq = 2 * mu_wk_sq + mHu_sq_wk + mHd_sq_wk
    #print("mA0^2: " + str(mA0sq))

    # Top quark tree-level running mass
    mymt = yt_wk * vu
    mymtsq = np.power(mymt, 2)
    #print("mt^2: " + str(mymtsq))

    # Bottom quark tree-level running mass
    mymb = yb_wk * vd
    mymbsq = np.power(mymb, 2)

    # Tau tree-level running mass
    mymtau = ytau_wk * vd
    mymtausq = np.power(mymtau, 2)

    # Charm quark tree-level running mass
    mymc = yc_wk * vu
    mymcsq = np.power(mymc, 2)

    # Strange quark tree-level running mass
    myms = ys_wk * vd
    mymssq = np.power(myms, 2)

    # Muon tree-level running mass
    mymmu = ymu_wk * vd
    mymmusq = np.power(mymmu, 2)

    # Up quark tree-level running mass
    mymu = yu_wk * vu
    mymusq = np.power(mymu, 2)

    # Down quark tree-level running mass
    mymd = yd_wk * vd
    mymdsq = np.power(mymd, 2)

    # Electron tree-level running mass
    myme = ye_wk * vd
    mymesq = np.power(myme, 2)

    # Sneutrino running masses
    mselecneutsq = mL1_sq_wk + (0.25 * (gpr_sq + g2_sq) * (vd_sq - vu_sq))
    msmuneutsq = mL2_sq_wk + (0.25 * (gpr_sq + g2_sq) * (vd_sq - vu_sq))
    mstauneutsq = mL3_sq_wk + (0.25 * (gpr_sq + g2_sq) * (vd_sq - vu_sq))

    # Tree-level charged Higgs doublet running squared mass.
    mH_pmsq = mA0sq + m_w_sq

    # Some terms for mass eigenstate-basis eigenvalue computations.
    DeltauL = (vd_sq - vu_sq) * ((3 * g2_sq) - (2 * gpr_sq)) / 12.0
    DeltauR = (vu_sq - vd_sq) * ((2 / 3) * gpr_sq)
    DeltadL = (vu_sq - vd_sq) * ((3 * g2_sq) + (2 * gpr_sq)) / 12.0
    DeltadR = (vd_sq - vu_sq) * ((1 / 3) * gpr_sq)
    DeltaeL = (vu_sq - vd_sq) * (g2_sq - (2 * gpr_sq)) / 4.0
    DeltaeR = gpr_sq * (vd_sq - vu_sq)

    # Up-type squark mass eigenstate eigenvalues
    m_stop_1sq = (0.5)\
        * ((2 * mymtsq) + mQ3_sq_wk + mU3_sq_wk + DeltauL + DeltauR
           - np.sqrt(np.power((mQ3_sq_wk - mU3_sq_wk + DeltauL - DeltauR),
                              2)
                     + (4 * (np.power((at_wk * vu), 2)
                             - (2 * at_wk * vu * vd * yt_wk * mu_wk)
                             + (vd_sq * np.power(yt_wk, 2) * mu_wk_sq)))))
    m_stop_2sq = (0.5)\
        * ((2 * mymtsq) + mQ3_sq_wk + mU3_sq_wk + DeltauL + DeltauR
           + np.sqrt(np.power((mQ3_sq_wk - mU3_sq_wk + DeltauL - DeltauR),
                              2)
                     + (4 * (np.power((at_wk * vu), 2)
                             - (2 * at_wk * vu * vd * yt_wk * mu_wk)
                             + (vd_sq * np.power(yt_wk, 2) * mu_wk_sq)))))
    #print("mst1^2: " + str(m_stop_1sq))
    #print("mst2^2: " + str(m_stop_2sq))
    m_scharm_1sq = (0.5)\
        * ((2 * mymcsq) + mQ2_sq_wk + mU2_sq_wk + DeltauL + DeltauR
           - np.sqrt(np.power((mQ2_sq_wk - mU2_sq_wk + DeltauL - DeltauR),
                              2)
                     + (4 * (np.power((ac_wk * vu), 2)
                             - (2 * ac_wk * vu * vd * yc_wk * mu_wk)
                             + (vd_sq * np.power(yc_wk, 2) * mu_wk_sq)))))
    m_scharm_2sq = (0.5)\
        * ((2 * mymcsq) + mQ2_sq_wk + mU2_sq_wk + DeltauL + DeltauR
           + np.sqrt(np.power((mQ2_sq_wk - mU2_sq_wk + DeltauL - DeltauR),
                              2)
                     + (4 * (np.power((ac_wk * vu), 2)
                             - (2 * ac_wk * vu * vd * yc_wk * mu_wk)
                             + (vd_sq * np.power(yc_wk, 2) * mu_wk_sq)))))
    m_sup_1sq = (0.5)\
        * ((2 * mymusq) + mQ1_sq_wk + mU1_sq_wk + DeltauL + DeltauR
           - np.sqrt(np.power((mQ1_sq_wk - mU1_sq_wk + DeltauL - DeltauR),
                              2)
                     + (4 * (np.power((au_wk * vu), 2)
                             - (2 * au_wk * vu * vd * yu_wk * mu_wk)
                             + (vd_sq * np.power(yu_wk, 2) * mu_wk_sq)))))
    m_sup_2sq = (0.5)\
        * ((2 * mymusq) + mQ1_sq_wk + mU1_sq_wk + DeltauL + DeltauR
           + np.sqrt(np.power((mQ1_sq_wk - mU1_sq_wk + DeltauL - DeltauR),
                              2)
                     + (4 * (np.power((au_wk * vu), 2)
                             - (2 * au_wk * vu * vd * yu_wk * mu_wk)
                             + (vd_sq * np.power(yu_wk, 2) * mu_wk_sq)))))

    # Down-type squark mass eigenstate eigenvalues
    m_sbot_1sq = (0.5)\
        * ((2 * mymbsq) + mQ3_sq_wk + mD3_sq_wk + DeltadL + DeltadR
           - np.sqrt(np.power((mQ3_sq_wk - mD3_sq_wk + DeltadL - DeltadR),
                              2)
                     + (4 * (np.power((ab_wk * vd), 2)
                             - (2 * ab_wk * vu * vd * yb_wk * mu_wk)
                             + (vu_sq * np.power(yb_wk, 2) * mu_wk_sq)))))
    m_sbot_2sq = (0.5)\
        * ((2 * mymbsq) + mQ3_sq_wk + mD3_sq_wk + DeltadL + DeltadR
           + np.sqrt(np.power((mQ3_sq_wk - mD3_sq_wk + DeltadL - DeltadR),
                              2)
                     + (4 * (np.power((ab_wk * vd), 2)
                             - (2 * ab_wk * vu * vd * yb_wk * mu_wk)
                             + (vu_sq * np.power(yb_wk, 2) * mu_wk_sq)))))
    m_sstrange_1sq = (0.5)\
        * ((2 * mymssq) + mQ2_sq_wk + mD2_sq_wk + DeltadL + DeltadR
           - np.sqrt(np.power((mQ2_sq_wk - mD2_sq_wk + DeltadL - DeltadR),
                              2)
                     + (4 * (np.power((as_wk * vd), 2)
                             - (2 * as_wk * vu * vd * ys_wk * mu_wk)
                             + (vu_sq * np.power(ys_wk, 2) * mu_wk_sq)))))
    m_sstrange_2sq = (0.5)\
        * ((2 * mymssq) + mQ2_sq_wk + mD2_sq_wk + DeltadL + DeltadR
           + np.sqrt(np.power((mQ2_sq_wk - mD2_sq_wk + DeltadL - DeltadR),
                              2)
                     + (4 * (np.power((as_wk * vd), 2)
                             - (2 * as_wk * vu * vd * ys_wk * mu_wk)
                             + (vu_sq * np.power(ys_wk, 2) * mu_wk_sq)))))
    m_sdown_1sq = (0.5)\
        * ((2 * mymdsq) + mQ1_sq_wk + mD1_sq_wk + DeltadL + DeltadR
           - np.sqrt(np.power((mQ1_sq_wk - mD1_sq_wk + DeltadL - DeltadR),
                              2)
                     + (4 * (np.power((ad_wk * vd), 2)
                             - (2 * ad_wk * vu * vd * yd_wk * mu_wk)
                             + (vu_sq * np.power(yd_wk, 2) * mu_wk_sq)))))
    m_sdown_2sq = (0.5)\
        * ((2 * mymdsq) + mQ1_sq_wk + mD1_sq_wk + DeltadL + DeltadR
           + np.sqrt(np.power((mQ1_sq_wk - mD1_sq_wk + DeltadL - DeltadR),
                              2)
                     + (4 * (np.power((ad_wk * vd), 2)
                             - (2 * ad_wk * vu * vd * yd_wk * mu_wk)
                             + (vu_sq * np.power(yd_wk, 2) * mu_wk_sq)))))

    # Slepton mass eigenstate eigenvalues
    m_stau_1sq = (0.5)\
        * ((2 * mymtausq) + mL3_sq_wk + mE3_sq_wk + DeltaeL + DeltaeR
           - np.sqrt(np.power((mL3_sq_wk - mE3_sq_wk + DeltaeL - DeltaeR),
                              2)
                     + (4 * (np.power((atau_wk * vd), 2)
                             - (2 * atau_wk * vu * vd * ytau_wk * mu_wk)
                             + (vu_sq * np.power(ytau_wk, 2) * mu_wk_sq)))
                     ))
    m_stau_2sq = (0.5)\
        * ((2 * mymtausq) + mL3_sq_wk + mE3_sq_wk + DeltaeL + DeltaeR
           + np.sqrt(np.power((mL3_sq_wk - mE3_sq_wk + DeltaeL - DeltaeR),
                              2)
                     + (4 * (np.power((atau_wk * vd), 2)
                             - (2 * atau_wk * vu * vd * ytau_wk * mu_wk)
                             + (vu_sq * np.power(ytau_wk, 2) * mu_wk_sq)))
                     ))
    m_smu_1sq = (0.5)\
        * ((2 * mymmusq) + mL2_sq_wk + mE2_sq_wk + DeltaeL + DeltaeR
           - np.sqrt(np.power((mL2_sq_wk - mE2_sq_wk + DeltaeL - DeltaeR),
                              2)
                     + (4 * (np.power((amu_wk * vd), 2)
                             - (2 * amu_wk * vu * vd * ymu_wk * mu_wk)
                             + (vu_sq * np.power(ymu_wk, 2) * mu_wk_sq)))))
    m_smu_2sq = (0.5)\
        * ((2 * mymmusq) + mL2_sq_wk + mE2_sq_wk + DeltaeL + DeltaeR
           + np.sqrt(np.power((mL2_sq_wk - mE2_sq_wk + DeltaeL - DeltaeR),
                              2)
                     + (4 * (np.power((amu_wk * vd), 2)
                             - (2 * amu_wk * vu * vd * ymu_wk * mu_wk)
                             + (vu_sq * np.power(ymu_wk, 2) * mu_wk_sq)))))
    m_se_1sq = (0.5)\
        * ((2 * mymesq) + mL1_sq_wk + mE1_sq_wk + DeltaeL + DeltaeR
           - np.sqrt(np.power((mL1_sq_wk - mE1_sq_wk + DeltaeL - DeltaeR),
                              2)
                     + (4 * (np.power((ae_wk * vd), 2)
                             - (2 * ae_wk * vu * vd * ye_wk * mu_wk)
                             + (vu_sq * np.power(ye_wk, 2) * mu_wk_sq)))))
    m_se_2sq = (0.5)\
        * ((2 * mymesq) + mL1_sq_wk + mE1_sq_wk + DeltaeL + DeltaeR
           + np.sqrt(np.power((mL1_sq_wk - mE1_sq_wk + DeltaeL - DeltaeR),
                              2)
                     + (4 * (np.power((ae_wk * vd), 2)
                             - (2 * ae_wk * vu * vd * ye_wk * mu_wk)
                             + (vu_sq * np.power(ye_wk, 2) * mu_wk_sq)))))

    # Chargino mass eigenstate eigenvalues
    msC1sq = (0.5)\
        * ((g2_sq * v_sq) + mu_wk_sq + np.power(M2_wk, 2)
           - np.sqrt(np.power((np.power(M2_wk, 2) + (g2_sq * v_sq)
                               + mu_wk_sq), 2)
                     - (4 * (np.power(g2_sq * vu * vd, 2)
                             - (2 * g2_sq * M2_wk * vu * vd * mu_wk)
                             + np.power(M2_wk * mu_wk, 2)))))
    msC2sq = (0.5)\
        * ((g2_sq * v_sq) + mu_wk_sq + np.power(M2_wk, 2)
           + np.sqrt(np.power((np.power(M2_wk, 2) + (g2_sq * v_sq)
                               + mu_wk_sq), 2)
                     - (4 * (np.power(g2_sq * vu * vd, 2)
                             - (2 * g2_sq * M2_wk * vu * vd * mu_wk)
                             + np.power(M2_wk * mu_wk, 2)))))

    # Neutralino mass eigenstate eigenvalues
    # neut_mass_sq_mat = \
    #     np.array([[np.power(M1_wk, 2) + (v_sq * gpr_sq
    #                                      / 2),
    #                (-1 / 2) * v_sq * g2_wk * gpr_wk,
    #                (-1 / np.sqrt(2)) * gpr_wk
    #                * ((M1_wk * vd) + (vu * mu_wk)),
    #                (1 / np.sqrt(2)) * gpr_wk
    #                * ((M1_wk * vu) + (vd * mu_wk))],# end row 1
    #               [(-1 / 2) * v_sq * g2_wk * gpr_wk,
    #                np.power(M2_wk, 2) + (v_sq * g2_sq
    #                                                 / 2),
    #                (1 / np.sqrt(2)) * g2_wk
    #                * ((M2_wk * vd) + (vu * mu_wk)),
    #                (-1 / np.sqrt(2)) * g2_wk
    #                * ((M2_wk * vu) + (vd * mu_wk))],# end row 2
    #               [(-1 / np.sqrt(2)) * gpr_wk
    #                * ((M1_wk * vd)
    #                   - (vu * mu_wk)),
    #                (1 / np.sqrt(2)) * g2_wk
    #                * ((M2_wk * vd)
    #                   - (vu * mu_wk)),
    #                ((1 / 2) * (gpr_sq + g2_sq)
    #                 * vd_sq) - mu_wk_sq,
    #                ((-1 / 2) * (gpr_sq + g2_sq) * vd * vu)],# end row 3
    #               [(1 / np.sqrt(2)) * gpr_wk
    #                * ((M1_wk * vu)
    #                   - (vd * mu_wk)),
    #                (-1 / np.sqrt(2)) * g2_wk
    #                * ((M2_wk * vu)
    #                   - (vd * mu_wk)),
    #                ((-1 / 2) * (gpr_sq + g2_sq) * vd * vu),
    #                ((1 / 2) * (gpr_sq + g2_sq)
    #                 * vu_sq) - mu_wk_sq]],
    #                             dtype=float)
    # #print(neut_mass_sq_mat)
    # my_neut_eigvals, my_neut_eigvecs = np.linalg.eig(neut_mass_sq_mat)
    # sorted_eigvals = sorted(my_neut_eigvals, key=abs)
    # # Include signs on N_1,2,4
    # msN1sq = sorted_eigvals[0]
    # #print("msN1^2: " + str(msN1sq))
    # msN2sq = sorted_eigvals[1]
    # #print("msN2^2: " + str(msN2sq))
    # msN3sq = sorted_eigvals[2]
    # #print("msN3^2: " + str(msN3sq))
    # msN4sq = sorted_eigvals[3]
    #print("msN4^2: " + str(msN4sq))
    neut_mass_mat = \
        np.array([[M1_wk, 0, (-1) * gpr_wk * vd / np.sqrt(2),
                   gpr_wk * vu / np.sqrt(2)],
                  [0, M2_wk, g2_wk * vd / np.sqrt(2),
                   (-1) * g2_wk * vu / np.sqrt(2)],
                  [(-1) * gpr_wk * vd / np.sqrt(2),
                   g2_wk * vd / np.sqrt(2), 0, (-1) * mu_wk],
                  [gpr_wk * vu / np.sqrt(2), (-1) * g2_wk * vu / np.sqrt(2),
                   (-1) * mu_wk, 0]])
    my_neut_mass_eigvals, my_neut_mass_eigvecs = np.linalg.eig(neut_mass_mat)
    sorted_mass_eigvals = sorted(my_neut_mass_eigvals, key=abs)
    mneutrsq = np.power(sorted_mass_eigvals, 2)
    #print("neutralino masses:")
    #print(sorted_mass_eigvals)
    #print(mneutrsq)
    msN1sq = mneutrsq[0]
    msN2sq = mneutrsq[1]
    msN3sq = mneutrsq[2]
    msN4sq = mneutrsq[3]
    # Neutral Higgs doublet mass eigenstate running squared masses
    mh0sq = 0.5 * (mA0sq + mz_q_sq
                   - np.sqrt(np.power(mA0sq - mz_q_sq, 2)
                             + (4 * mz_q_sq * mA0sq
                                * np.power(np.sin(2 * beta_wk), 2))))
    mH0sq = 0.5 * (mA0sq + mz_q_sq
                   + np.sqrt(np.power(mA0sq - mz_q_sq, 2)
                             + (4 * mz_q_sq * mA0sq
                                * np.power(np.sin(2 * beta_wk), 2))))

    # mh0sq = (1.0 / 2.0) * (mA0sq + mz_q_sq
    #                        - np.sqrt(np.power((mA0sq - mz_q_sq), 2)
    #                                  + (4.0 * mz_q_sq * mA0sq
    #                                     * (np.power(np.sin(2 * beta_wk), 2)))))
    # mH0sq = (1.0 / 2.0) * (mA0sq + mz_q_sq
    #                        + np.sqrt(np.power((mA0sq - mz_q_sq), 2)
    #                                  + (4.0 * mz_q_sq * mA0sq
    #                                     * (np.power(np.sin(2 * beta_wk), 2)))))
    #print(np.sqrt(mh0sq))
    #print(np.sqrt(mH0sq))
    ##### Radiative corrections in stop squark sector #####

    stop_denom = np.sqrt(np.power((4 * mQ3_sq_wk) - (4 * mU3_sq_wk)
                                    + ((vd_sq - vu_sq)
                                       * (g2_sq + (2 * gpr_sq))), 2)
                           + (64 * np.power(at_wk, 2) * vu_sq)
                           + (64 * np.abs(mu_wk_sq) * vd_sq
                              * np.power(yt_wk, 2))
                           - (128 * at_wk * mu_wk * yt_wk * vd * vu))
    dmsq_st1_dvuvd = 8 * at_wk * yt_wk * mu_wk / stop_denom
    dmsq_st2_dvuvd = (-1) * dmsq_st1_dvuvd
    sigmauu_stop_1 = (3 * loop_fac) * logfunc2(m_stop_1sq) \
        * (np.power(yt_wk, 2) - (gz_sq)
           - (((np.power(at_wk, 2)) - (8 * gz_sq
                                       * ((1 / 4)
                                          - ((2 / 3) * sinsq_th_w))
                                       * (((mQ3_sq_wk - mU3_sq_wk) / 2)
                                          + (mz_q_sq * cos2b
                                             * ((1 / 4)
                                                - ((2 / 3) * sinsq_th_w))))))
              / (m_stop_2sq - m_stop_1sq)))
    sigmauu_stop_2 = (3 * loop_fac) * logfunc2(m_stop_2sq) \
        * (np.power(yt_wk, 2) - (gz_sq)
           + (((np.power(at_wk, 2)) - (8 * gz_sq
                                       * ((1 / 4)
                                          - ((2 / 3) * sinsq_th_w))
                                       * (((mQ3_sq_wk - mU3_sq_wk) / 2)
                                          + (mz_q_sq * cos2b
                                             * ((1 / 4)
                                                - ((2 / 3) * sinsq_th_w))))))
              / (m_stop_2sq - m_stop_1sq)))
    sigmadd_stop_1 = (3 * loop_fac) * logfunc2(m_stop_1sq) \
        * ((gz_sq)
           - (((np.power(yt_wk, 2)
                * mu_wk_sq)
               + (8 * gz_sq
                  * ((1 / 4) - ((2 / 3) * sinsq_th_w))
                  * (((mQ3_sq_wk - mU3_sq_wk) / 2)
                     + (mz_q_sq * cos2b * ((1 / 4)
                                           - ((2 / 3) * sinsq_th_w))))))
              / (m_stop_2sq - m_stop_1sq)))
    sigmadd_stop_2 = (3 * loop_fac) * logfunc2(m_stop_2sq) \
        * ((gz_sq)
           - (((np.power(yt_wk, 2)
                * mu_wk_sq)
               + (8 * gz_sq
                  * ((1 / 4) - ((2 / 3) * sinsq_th_w))
                  * (((mQ3_sq_wk - mU3_sq_wk) / 2)
                     + (mz_q_sq * cos2b * ((1 / 4)
                                           - ((2 / 3) * sinsq_th_w))))))
              / (m_stop_2sq - m_stop_1sq)))
    sigmaud_stop_1 = (3 * loop_fac) * logfunc2(m_stop_1sq) \
        * dmsq_st1_dvuvd
    sigmaud_stop_2 = (3 * loop_fac) * logfunc2(m_stop_2sq) \
        * dmsq_st2_dvuvd

    ##### Radiative corrections in sbottom squark sector #####
    sbot_denom = np.sqrt(np.power((4 * mD3_sq_wk) - (4 * mQ3_sq_wk)
                                    + ((vd_sq - vu_sq)
                                       * (g2_sq + (2 * gpr_sq))), 2)
                           + (64 * np.power(ab_wk, 2) * vd_sq)
                           + (64 * mu_wk_sq * vu_sq
                              * np.power(yb_wk, 2)))
    dmsq_sb1_dvuvd = 8 * ab_wk * yb_wk * mu_wk / sbot_denom
    dmsq_sb2_dvuvd = (-1) * dmsq_sb1_dvuvd
    sigmauu_sbot_1 = (3 * loop_fac) * logfunc2(m_sbot_1sq) \
        * ((gz_sq)
           - (((np.power(yb_wk, 2)
                * mu_wk_sq)
               - (8 * gz_sq
                  * ((1 / 4) - ((1 / 3) * sinsq_th_w))
                  * (((mQ3_sq_wk - mD3_sq_wk) / 2)
                     - (mz_q_sq * cos2b * ((1 / 4)
                                           - ((1 / 3) * sinsq_th_w))))))
              / (m_sbot_2sq - m_sbot_1sq)))
    sigmauu_sbot_2 = (3 * loop_fac) * logfunc2(m_sbot_2sq) \
        * ((gz_sq)
           + (((np.power(yb_wk, 2)
                * mu_wk_sq)
               - (8 * gz_sq
                  * ((1 / 4) - ((1 / 3) * sinsq_th_w))
                  * (((mQ3_sq_wk - mD3_sq_wk) / 2)
                     - (mz_q_sq * cos2b * ((1 / 4)
                                           - ((1 / 3) * sinsq_th_w))))))
              / (m_sbot_2sq - m_sbot_1sq)))
    sigmadd_sbot_1 = (3 * loop_fac) * logfunc2(m_sbot_1sq) \
        * (np.power(yb_wk, 2) - (gz_sq)
           - (((np.power(ab_wk, 2))
               - (8 * gz_sq
                  * ((1 / 4) - ((1 / 3) * sinsq_th_w))
                  * (((mQ3_sq_wk - mD3_sq_wk) / 2)
                     - (mz_q_sq * cos2b * ((1 / 4)
                                           - ((1 / 3) * sinsq_th_w))))))
              / (m_sbot_2sq - m_sbot_1sq)))
    sigmadd_sbot_2 = (3 * loop_fac) * logfunc2(m_sbot_2sq) \
        * (np.power(yb_wk, 2) - (gz_sq)
           + (((np.power(ab_wk, 2))
               - (8 * gz_sq
                  * ((1 / 4) - ((1 / 3) * sinsq_th_w))
                  * (((mQ3_sq_wk - mD3_sq_wk) / 2)
                     - (mz_q_sq * cos2b * ((1 / 4)
                                           - ((1 / 3) * sinsq_th_w))))))
              / (m_sbot_2sq - m_sbot_1sq)))
    sigmaud_sbot_1 = (3 * loop_fac) * logfunc2(m_sbot_1sq) \
        * dmsq_sb1_dvuvd
    sigmaud_sbot_2 = (3 * loop_fac) * logfunc2(m_sbot_2sq) \
        * dmsq_sb2_dvuvd
    
    ##### Radiative corrections in stau slepton sector #####
    stau_denom = np.sqrt(np.power((4 * mE3_sq_wk) - (4 * mL3_sq_wk)
                                  + ((vd_sq - vu_sq)
                                     * (g2_sq + (2 * gpr_sq))), 2)
                         + (64 * np.power(atau_wk, 2) * vd_sq)
                         + (64 * np.abs(mu_wk_sq) * vu_sq
                            * np.power(ytau_wk, 2))
                         - (128 * atau_wk * mu_wk * ytau_wk * vd * vu))
    dmsq_stau1_dvuvd = 8 * atau_wk * ytau_wk * mu_wk / stau_denom
    dmsq_stau2_dvuvd = (-1) * dmsq_stau1_dvuvd
    sigmauu_stau_1 = (loop_fac) * logfunc2(m_stau_1sq) \
        * ((gz_sq)
           - (((np.power(ytau_wk, 2)
                * mu_wk_sq)
               - (8 * gz_sq
                  * ((1 / 4) - (sinsq_th_w))
                  * (((mL3_sq_wk - mE3_sq_wk) / 2)
                     - (mz_q_sq * cos2b * ((1 / 4)
                                           - (sinsq_th_w))))))
              / (m_stau_2sq - m_stau_1sq)))
    sigmauu_stau_2 = (loop_fac) * logfunc2(m_stau_2sq) \
        * ((gz_sq)
           + (((np.power(ytau_wk, 2)
                * mu_wk_sq)
               - (8 * gz_sq
                  * ((1 / 4) - (sinsq_th_w))
                  * (((mL3_sq_wk - mE3_sq_wk) / 2)
                     - (mz_q_sq * cos2b * ((1 / 4)
                                           - (sinsq_th_w))))))
              / (m_stau_2sq - m_stau_1sq)))
    sigmadd_stau_1 = (loop_fac) * logfunc2(m_stau_1sq) \
        * (np.power(ytau_wk, 2) - (gz_sq)
           - (((np.power(atau_wk, 2))
               - (8 * gz_sq
                  * ((1 / 4) - (sinsq_th_w))
                  * (((mL3_sq_wk - mE3_sq_wk) / 2)
                     - (mz_q_sq * cos2b * ((1 / 4)
                                           - (sinsq_th_w))))))
              / (m_stau_2sq - m_stau_1sq)))
    sigmadd_stau_2 = (loop_fac) * logfunc2(m_stau_1sq) \
        * (np.power(ytau_wk, 2) - (gz_sq)
           + (((np.power(atau_wk, 2))
               - (8 * gz_sq
                  * ((1 / 4) - (sinsq_th_w))
                  * (((mL3_sq_wk - mE3_sq_wk) / 2)
                     - (mz_q_sq * cos2b * ((1 / 4)
                                           - (sinsq_th_w))))))
              / (m_stau_2sq - m_stau_1sq)))
    sigmaud_stau_1 = (loop_fac) * logfunc2(m_stau_1sq) \
        * dmsq_stau1_dvuvd
    sigmaud_stau_2 = (loop_fac) * logfunc2(m_stau_2sq) \
        * dmsq_stau2_dvuvd
    dmstauneutsq_dvusq = ((-1) / 4) * (gpr_sq + g2_sq)
    dmstauneutsq_dvdsq = (1 / 4) * (gpr_sq + g2_sq)
    sigmauu_stau_sneut = (1 / 2) * loop_fac * logfunc2(mstauneutsq)\
        * dmstauneutsq_dvusq
    sigmadd_stau_sneut = (1 / 2) * loop_fac * logfunc2(mstauneutsq)\
        * dmstauneutsq_dvdsq
    sigmaud_stau_sneut = 0

    ##### Radiative corrections from 2nd generation sfermions #####
    # Scharm sector
    schm_denom = np.sqrt(np.power((4 * mQ2_sq_wk) - (4 * mU2_sq_wk)
                                    + ((vd_sq - vu_sq)
                                       * (g2_sq + (2 * gpr_sq))), 2)
                           + (64 * np.power(ac_wk, 2) * vu_sq)
                           + (64 * mu_wk_sq * vd_sq
                              * np.power(yc_wk, 2))
                           - (128 * ac_wk * mu_wk * yc_wk * vd * vu))
    dmsq_sc1_dvuvd = 8 * ac_wk * yc_wk * mu_wk / schm_denom
    dmsq_sc2_dvuvd = (-1) * dmsq_sc1_dvuvd
    sigmauu_scharm_1 = (3 * loop_fac) * logfunc2(m_scharm_1sq) \
        * (np.power(yc_wk, 2) - (gz_sq)
           - (((np.power(ac_wk, 2)) - (8 * gz_sq
                                       * ((1 / 4)
                                          - ((2 / 3) * sinsq_th_w))
                                       * (((mQ2_sq_wk - mU2_sq_wk) / 2)
                                          + (mz_q_sq * cos2b
                                             * ((1 / 4)
                                                - ((2 / 3) * sinsq_th_w))))))
              / (m_scharm_2sq - m_scharm_1sq)))
    sigmauu_scharm_2 = (3 * loop_fac) * logfunc2(m_scharm_2sq) \
        * (np.power(yc_wk, 2) - (gz_sq)
           + (((np.power(ac_wk, 2)) - (8 * gz_sq
                                       * ((1 / 4)
                                          - ((2 / 3) * sinsq_th_w))
                                       * (((mQ2_sq_wk - mU2_sq_wk) / 2)
                                          + (mz_q_sq * cos2b
                                             * ((1 / 4)
                                                - ((2 / 3) * sinsq_th_w))))))
              / (m_scharm_2sq - m_scharm_1sq)))
    sigmadd_scharm_1 = (3 * loop_fac) * logfunc2(m_scharm_1sq) \
        * ((gz_sq)
           - (((np.power(yc_wk, 2)
                * mu_wk_sq)
               + (8 * gz_sq
                  * ((1 / 4) - ((2 / 3) * sinsq_th_w))
                  * (((mQ2_sq_wk - mU2_sq_wk) / 2)
                     + (mz_q_sq * cos2b * ((1 / 4)
                                           - ((2 / 3) * sinsq_th_w))))))
         / (m_scharm_2sq - m_scharm_1sq)))
    sigmadd_scharm_2 = (3 * loop_fac) * logfunc2(m_scharm_2sq) \
        * ((gz_sq)
           + (((np.power(yc_wk, 2)
                * mu_wk_sq)
               + (8 * gz_sq
                  * ((1 / 4) - ((2 / 3) * sinsq_th_w))
                  * (((mQ2_sq_wk - mU2_sq_wk) / 2)
                     + (mz_q_sq * cos2b * ((1 / 4)
                                           - ((2 / 3) * sinsq_th_w))))))
         / (m_scharm_2sq - m_scharm_1sq)))
    sigmaud_scharm_1 = (3 * loop_fac) * logfunc2(m_scharm_1sq) \
        * dmsq_sc1_dvuvd
    sigmaud_scharm_2 = (3 * loop_fac) * logfunc2(m_scharm_2sq) \
        * dmsq_sc2_dvuvd
    # Sstrange sector
    sstr_denom = np.sqrt(np.power((4 * mD2_sq_wk) - (4 * mQ2_sq_wk)
                                    + ((vd_sq - vu_sq)
                                       * (g2_sq + (2 * gpr_sq))), 2)
                           + (64 * np.power(as_wk, 2) * vd_sq)
                           + (64 * mu_wk_sq * vu_sq
                              * np.power(ys_wk, 2))
                           - (128 * as_wk * mu_wk
                              * ys_wk * vd * vu))
    dmsq_ss1_dvuvd = 8 * as_wk * ys_wk * mu_wk / sstr_denom
    dmsq_ss2_dvuvd = (-1) * dmsq_ss1_dvuvd
    sigmauu_sstrange_1 = (3 * loop_fac) * logfunc2(m_sstrange_1sq) \
        * ((gz_sq)
           - (((np.power(ys_wk, 2)
                * mu_wk_sq)
               - (8 * gz_sq
                  * ((1 / 4) - ((1 / 3) * sinsq_th_w))
                  * (((mQ2_sq_wk - mD2_sq_wk) / 2)
                     - (mz_q_sq * cos2b * ((1 / 4)
                                           - ((1 / 3) * sinsq_th_w))))))
              / (m_sstrange_2sq - m_sstrange_1sq)))
    sigmauu_sstrange_2 = (3 * loop_fac) * logfunc2(m_sstrange_2sq) \
        * ((gz_sq)
           + (((np.power(ys_wk, 2)
                * mu_wk_sq)
               - (8 * gz_sq
                  * ((1 / 4) - ((1 / 3) * sinsq_th_w))
                  * (((mQ2_sq_wk - mD2_sq_wk) / 2)
                     - (mz_q_sq * cos2b * ((1 / 4)
                                           - ((1 / 3) * sinsq_th_w))))))
              / (m_sstrange_2sq - m_sstrange_1sq)))
    sigmadd_sstrange_1 = (3 * loop_fac) * logfunc2(m_sstrange_1sq) \
        * (np.power(ys_wk, 2) - (gz_sq)
           - (((np.power(as_wk, 2))
               - (8 * gz_sq
                  * ((1 / 4) - ((1 / 3) * sinsq_th_w))
                  * (((mQ2_sq_wk - mD2_sq_wk) / 2)
                     - (mz_q_sq * cos2b * ((1 / 4)
                                           - ((1 / 3) * sinsq_th_w))))))
              / (m_sstrange_2sq - m_sstrange_1sq)))
    sigmadd_sstrange_2 = (3 * loop_fac) * logfunc2(m_sstrange_2sq) \
        * (np.power(ys_wk, 2) - (gz_sq)
           + (((np.power(as_wk, 2))
               - (8 * gz_sq
                  * ((1 / 4) - ((1 / 3) * sinsq_th_w))
                  * (((mQ2_sq_wk - mD2_sq_wk) / 2)
                     - (mz_q_sq * cos2b * ((1 / 4)
                                           - ((1 / 3) * sinsq_th_w))))))
              / (m_sstrange_2sq - m_sstrange_1sq)))
    sigmaud_sstrange_1 = (3 * loop_fac) * logfunc2(m_sstrange_1sq) \
        * dmsq_ss1_dvuvd
    sigmaud_sstrange_2 = (3 * loop_fac) * logfunc2(m_sstrange_2sq) \
        * dmsq_ss2_dvuvd

    # Smu/smu sneutrino

    smu_denom = np.sqrt(np.power((4 * mE2_sq_wk) - (4 * mL2_sq_wk)
                                 + ((vd_sq - vu_sq)
                                    * (g2_sq + (2 * gpr_sq))), 2)
                        + (64 * np.power(amu_wk, 2) * vd_sq)
                        + (64 * mu_wk_sq * vu_sq
                           * np.power(ymu_wk, 2))
                        - (128 * amu_wk * mu_wk
                           * ymu_wk * vd * vu))
    dmsq_smu1_dvuvd = 8 * amu_wk * ymu_wk * mu_wk / smu_denom
    dmsq_smu2_dvuvd = (-1) * dmsq_smu1_dvuvd
    sigmauu_smu_1 = (loop_fac) * logfunc2(m_smu_1sq) \
        * ((gz_sq)
           - (((np.power(ymu_wk, 2)
                * mu_wk_sq)
               - (8 * gz_sq
                  * ((1 / 4) - (sinsq_th_w))
                  * (((mL2_sq_wk - mE2_sq_wk) / 2)
                     - (mz_q_sq * cos2b * ((1 / 4)
                                           - (sinsq_th_w))))))
              / (m_smu_2sq - m_smu_1sq)))
    sigmauu_smu_2 = (loop_fac) * logfunc2(m_smu_1sq) \
        * ((gz_sq)
           + (((np.power(ymu_wk, 2)
                * mu_wk_sq)
               - (8 * gz_sq
                  * ((1 / 4) - (sinsq_th_w))
                  * (((mL2_sq_wk - mE2_sq_wk) / 2)
                     - (mz_q_sq * cos2b * ((1 / 4)
                                           - (sinsq_th_w))))))
              / (m_smu_2sq - m_smu_1sq)))
    sigmadd_smu_1 = (loop_fac) * logfunc2(m_smu_1sq) \
        * (np.power(ymu_wk, 2) - (gz_sq)
           - (((np.power(amu_wk, 2))
               - (8 * gz_sq
                  * ((1 / 4) - (sinsq_th_w))
                  * (((mL2_sq_wk - mE2_sq_wk) / 2)
                     - (mz_q_sq * cos2b * ((1 / 4)
                                           - (sinsq_th_w))))))
              / (m_smu_2sq - m_smu_1sq)))
    sigmadd_smu_2 = (loop_fac) * logfunc2(m_smu_2sq) \
        * (np.power(ymu_wk, 2) - (gz_sq)
           + (((np.power(amu_wk, 2))
               - (8 * gz_sq
                  * ((1 / 4) - (sinsq_th_w))
                  * (((mL2_sq_wk - mE2_sq_wk) / 2)
                     - (mz_q_sq * cos2b * ((1 / 4)
                                           - (sinsq_th_w))))))
              / (m_smu_2sq - m_smu_1sq)))
    sigmaud_smu_1 = (loop_fac) * logfunc2(m_smu_1sq) \
        * dmsq_smu1_dvuvd
    sigmaud_smu_2 = (loop_fac) * logfunc2(m_smu_2sq) \
        * dmsq_smu2_dvuvd
    dmsmuneutsq_dvusq = ((-1) / 4) * (gpr_sq + g2_sq)
    dmsmuneutsq_dvdsq = (1 / 4) * (gpr_sq + g2_sq)
    sigmauu_smu_sneut = (1 / 2) * loop_fac * logfunc2(msmuneutsq)\
        * dmsmuneutsq_dvusq
    sigmadd_smu_sneut = (1 / 2) * loop_fac * logfunc2(msmuneutsq)\
        * dmsmuneutsq_dvdsq
    sigmaud_smu_sneut = 0

    ##### Radiative corrections from 1st generation sfermions #####
    # Sup sector

    sup_denom = np.sqrt(np.power((4 * mQ1_sq_wk) - (4 * mU1_sq_wk)
                                    + ((vd_sq - vu_sq)
                                       * (g2_sq + (2 * gpr_sq))), 2)
                        + (64 * np.power(au_wk, 2) * vu_sq)
                        + (64 * mu_wk_sq * vd_sq
                           * np.power(yu_wk, 2))
                        - (128 * au_wk * mu_wk * yu_wk * vd * vu))
    dmsq_su1_dvuvd = 8 * au_wk * yu_wk * mu_wk / sup_denom
    dmsq_su2_dvuvd = (-1) * dmsq_su1_dvuvd
    sigmauu_sup_1 = (3 * loop_fac) * logfunc2(m_sup_1sq) \
        * (np.power(yu_wk, 2) - (gz_sq)
           - (((np.power(au_wk, 2)) - (8 * gz_sq
                                       * ((1 / 4)
                                          - ((2 / 3) * sinsq_th_w))
                                       * (((mQ1_sq_wk - mU1_sq_wk) / 2)
                                          + (mz_q_sq * cos2b
                                             * ((1 / 4)
                                                - ((2 / 3) * sinsq_th_w))))))
              / (m_sup_2sq - m_sup_1sq)))
    sigmauu_sup_2 = (3 * loop_fac) * logfunc2(m_sup_2sq) \
        * (np.power(yu_wk, 2) - (gz_sq)
           + (((np.power(au_wk, 2)) - (8 * gz_sq
                                       * ((1 / 4)
                                          - ((2 / 3) * sinsq_th_w))
                                       * (((mQ1_sq_wk - mU1_sq_wk) / 2)
                                          + (mz_q_sq * cos2b
                                             * ((1 / 4)
                                                - ((2 / 3) * sinsq_th_w))))))
              / (m_sup_2sq - m_sup_1sq)))
    sigmadd_sup_1 = (3 * loop_fac) * logfunc2(m_sup_1sq) \
        * ((gz_sq)
           - (((np.power(yu_wk, 2)
                * mu_wk_sq)
               + (8 * gz_sq
                  * ((1 / 4) - ((2 / 3) * sinsq_th_w))
                  * (((mQ1_sq_wk - mU1_sq_wk) / 2)
                     + (mz_q_sq * cos2b * ((1 / 4)
                                           - ((2 / 3) * sinsq_th_w))))))
         / (m_sup_2sq - m_sup_1sq)))
    sigmadd_sup_2 = (3 * loop_fac) * logfunc2(m_sup_2sq) \
        * ((gz_sq)
           + (((np.power(yu_wk, 2)
                * mu_wk_sq)
               + (8 * gz_sq
                  * ((1 / 4) - ((2 / 3) * sinsq_th_w))
                  * (((mQ1_sq_wk - mU1_sq_wk) / 2)
                     + (mz_q_sq * cos2b * ((1 / 4)
                                           - ((2 / 3) * sinsq_th_w))))))
         / (m_sup_2sq - m_sup_1sq)))
    sigmaud_sup_1 = (3 * loop_fac) * logfunc2(m_sup_1sq) \
        * dmsq_su1_dvuvd
    sigmaud_sup_2 = (3 * loop_fac) * logfunc2(m_sup_2sq) \
        * dmsq_su2_dvuvd
    # Sdown sector
    sdwn_denom = np.sqrt(np.power((4 * mD1_sq_wk) - (4 * mQ1_sq_wk)
                                    + ((vd_sq - vu_sq)
                                       * (g2_sq + (2 * gpr_sq))), 2)
                           + (64 * np.power(ad_wk, 2) * vd_sq)
                           + (64 * mu_wk_sq * vu_sq
                              * np.power(yd_wk, 2))
                           - (128 * ad_wk * mu_wk
                              * yd_wk * vd * vu))
    dmsq_sd1_dvuvd = 8 * ad_wk * yd_wk * mu_wk / sdwn_denom
    dmsq_sd2_dvuvd = (-1) * dmsq_sd1_dvuvd
    sigmauu_sdown_1 = (3 * loop_fac) * logfunc2(m_sdown_1sq) \
        * ((gz_sq)
           - (((np.power(yd_wk, 2)
                * mu_wk_sq)
               - (8 * gz_sq
                  * ((1 / 4) - ((1 / 3) * sinsq_th_w))
                  * (((mQ1_sq_wk - mD1_sq_wk) / 2)
                     - (mz_q_sq * cos2b * ((1 / 4)
                                           - ((1 / 3) * sinsq_th_w))))))
              / (m_sdown_2sq - m_sdown_1sq)))
    sigmauu_sdown_2 = (3 * loop_fac) * logfunc2(m_sdown_2sq) \
        * ((gz_sq)
           + (((np.power(yd_wk, 2)
                * mu_wk_sq)
               - (8 * gz_sq
                  * ((1 / 4) - ((1 / 3) * sinsq_th_w))
                  * (((mQ1_sq_wk - mD1_sq_wk) / 2)
                     - (mz_q_sq * cos2b * ((1 / 4)
                                           - ((1 / 3) * sinsq_th_w))))))
              / (m_sdown_2sq - m_sdown_1sq)))
    sigmadd_sdown_1 = (3 * loop_fac) * logfunc2(m_sdown_1sq) \
        * (np.power(yd_wk, 2) - (gz_sq)
           - (((np.power(ad_wk, 2))
               - (8 * gz_sq
                  * ((1 / 4) - ((1 / 3) * sinsq_th_w))
                  * (((mQ1_sq_wk - mD1_sq_wk) / 2)
                     - (mz_q_sq * cos2b * ((1 / 4)
                                           - ((1 / 3) * sinsq_th_w))))))
              / (m_sdown_2sq - m_sdown_1sq)))
    sigmadd_sdown_2 = (3 * loop_fac) * logfunc2(m_sdown_2sq) \
        * (np.power(yd_wk, 2) - (gz_sq)
           + (((np.power(ad_wk, 2))
               - (8 * gz_sq
                  * ((1 / 4) - ((1 / 3) * sinsq_th_w))
                  * (((mQ1_sq_wk - mD1_sq_wk) / 2)
                     - (mz_q_sq * cos2b * ((1 / 4)
                                           - ((1 / 3) * sinsq_th_w))))))
              / (m_sdown_2sq - m_sdown_1sq)))
    sigmaud_sdown_1 = (3 * loop_fac) * logfunc2(m_sdown_1sq) \
        * dmsq_sd1_dvuvd
    sigmaud_sdown_2 = (3 * loop_fac) * logfunc2(m_sdown_2sq) \
        * dmsq_sd2_dvuvd
    # Selectron/selectron sneutrino
    s_e_denom = np.sqrt(np.power((4 * mE1_sq_wk) - (4 * mL1_sq_wk)
                                 + ((vd_sq - vu_sq)
                                    * (g2_sq + (2 * gpr_sq))), 2)
                        + (64 * np.power(ae_wk, 2) * vd_sq)
                        + (64 * mu_wk_sq * vu_sq
                           * np.power(ye_wk, 2))
                        - (128 * ae_wk * mu_wk
                           * ye_wk * vd * vu))
    dmsq_se1_dvuvd = 8 * ae_wk * ye_wk * mu_wk / s_e_denom
    dmsq_se2_dvuvd = (-1) * dmsq_se1_dvuvd
    sigmauu_se_1 = (loop_fac) * logfunc2(m_se_1sq) \
        * ((gz_sq)
           - (((np.power(ye_wk, 2)
                * mu_wk_sq)
               - (8 * gz_sq
                  * ((1 / 4) - (sinsq_th_w))
                  * (((mL1_sq_wk - mE1_sq_wk) / 2)
                     - (mz_q_sq * cos2b * ((1 / 4)
                                           - (sinsq_th_w))))))
              / (m_se_2sq - m_se_1sq)))
    sigmauu_se_2 = (loop_fac) * logfunc2(m_se_2sq) \
        * ((gz_sq)
           + (((np.power(ye_wk, 2)
                * mu_wk_sq)
               - (8 * gz_sq
                  * ((1 / 4) - (sinsq_th_w))
                  * (((mL1_sq_wk - mE1_sq_wk) / 2)
                     - (mz_q_sq * cos2b * ((1 / 4)
                                           - (sinsq_th_w))))))
              / (m_se_2sq - m_se_1sq)))
    sigmadd_se_1 = (loop_fac) * logfunc2(m_se_1sq) \
        * (np.power(ye_wk, 2) - (gz_sq)
           - (((np.power(ae_wk, 2))
               - (8 * gz_sq
                  * ((1 / 4) - (sinsq_th_w))
                  * (((mL1_sq_wk - mE1_sq_wk) / 2)
                     - (mz_q_sq * cos2b * ((1 / 4)
                                           - (sinsq_th_w))))))
              / (m_se_2sq - m_se_1sq)))
    sigmadd_se_2 = (loop_fac) * logfunc2(m_se_1sq) \
        * (np.power(ye_wk, 2) - (gz_sq)
           + (((np.power(ae_wk, 2))
               - (8 * gz_sq
                  * ((1 / 4) - (sinsq_th_w))
                  * (((mL1_sq_wk - mE1_sq_wk) / 2)
                     - (mz_q_sq * cos2b * ((1 / 4)
                                           - (sinsq_th_w))))))
              / (m_se_2sq - m_se_1sq)))
    sigmaud_se_1 = loop_fac * logfunc2(m_se_1sq) \
        * dmsq_se1_dvuvd
    sigmaud_se_2 = loop_fac * logfunc2(m_se_2sq) \
        * dmsq_se2_dvuvd
    dmseneutsq_dvusq = ((-1) / 4) * (gpr_sq + g2_sq)
    dmseneutsq_dvdsq = (1 / 4) * (gpr_sq + g2_sq)
    sigmauu_selec_sneut = (1 / 2) * loop_fac * logfunc2(mselecneutsq)\
        * dmseneutsq_dvusq
    sigmadd_selec_sneut = (1 / 2) * loop_fac * logfunc2(mselecneutsq)\
        * dmseneutsq_dvdsq
    sigmaud_selec_sneut = 0


    ##### Radiative corrections from neutralino sector #####
    def neutralino_denom(msnsq):
        """
        Return denominator for one-loop correction
            of neutralino according to method of Ibrahim
            and Nath in PhysRevD.66.015005 (2002).

        Parameters
        ----------
        msnsq : Float. 
            Neutralino squared mass used for evaluating results.

        """
        # Introduce coefficients of characteristic equation for eigenvals.
        # Char. eqn. is of the form x^4 + ax^3 + bx^2 + cx + d = 0
        char_a = (-1)\
            * (np.power(M1_wk, 2) + np.power(M2_wk, 2) 
               + ((g2_sq + gpr_sq) * v_sq) - (2 * mu_wk_sq))
        char_b = np.power(M1_wk, 2) * (np.power(M2_wk, 2)
                                       + (np.power(g2_wk, 2) * v_sq)
                                       - (2 * mu_wk_sq))\
            + np.power(mu_wk_sq, 2) - (mu_wk_sq * ((2 * np.power(M2_wk, 2))
                                                   + (g2_sq * v_sq)))\
            + (((np.power(g2_sq, 2) / 4) + (np.power(gpr_sq, 2) / 4))
               * np.power(v_sq, 2))\
            + ((gpr_sq / 2) * ((2 * np.power(M2_wk, 2) * v_sq)
                               + (g2_sq * np.power(v_sq, 2))
                               - (2 * v_sq * mu_wk_sq)))
        char_c = (1 / 4)\
            * (((-1) * np.power(gpr_sq, 2)
                * ((np.power(M2_wk, 2) * v_sq)))
               + (2 * ((4 * np.power(M1_wk * M2_wk, 2))
                       + (2 * np.power(gpr_wk * M2_wk, 2) * v_sq)
                       + (2 * np.power(gpr_sq, 2) * vu_sq * vd_sq))
                  * mu_wk_sq)
               - (4 * (np.power(M1_wk, 2) + np.power(M2_wk, 2))
                  * np.power(mu_wk_sq, 2))
               + (np.power(g2_sq, 2) * ((4 * mu_wk_sq * vu_sq * vd_sq)
                                        - (np.power(M1_wk * v_sq, 2))))
               + (2 * g2_sq
                  * ((2 * mu_wk_sq * v_sq * np.power(M1_wk, 2))
                     + (gpr_sq
                        * (((-1) * M1_wk * M2_wk * np.power(v_sq, 2))
                           + (4 * mu_wk_sq * vu_sq * vd_sq))))))

        myden = (4 * np.power(msnsq, 3)) + (3 * char_a
                                            * np.power(msnsq, 2))\
            + (2 * char_b * msnsq) + char_c
        return myden

    def neutralinouu_num(msnsq):
        """
        Return numerator for one-loop uu correction
            derivative term of neutralino.

        Parameters
        ----------
        msnsq : Float. 
            Neutralino squared mass used for evaluating results.

        """
        cubicterm = (-1) * (np.power(g2_wk, 2) + np.power(gpr_wk, 2))
        quadrterm = (1 / 4)\
            * ((np.power(g2_sq, 2) * ((2 * v_sq) + vd_sq))
               + (gpr_sq * ((4 * np.power(M2_wk, 2))
                            + (gpr_sq * ((2 * v_sq) + vd_sq))
                            - (4 * mu_wk_sq)))
               + (g2_sq * ((4 * np.power(M1_wk, 2))
                           + (gpr_sq * ((4 * v_sq) + (2 * vd_sq)))
                           - (4 * mu_wk_sq))))
        linterm = (1 / 2)\
            * ((np.power(gpr_sq, 2)
                * ((np.power(M1_wk, 2) * vd_sq)
                   - (np.power(M2_wk, 2) * v_sq)))
               + (np.power(g2_sq, 2)
                  * ((vd_sq * (np.power(M2_wk, 2) + mu_wk_sq))
                     - (v_sq * np.power(M1_wk, 2))))
               + (gpr_sq * mu_wk_sq * ((2 * np.power(M2_wk, 2))
                                       + (vd_sq * gpr_sq)))
               + (g2_sq
                  * ((gpr_sq * ((np.power(M1_wk - M2_wk, 2) * vd_sq)
                                - (2 * M1_wk * M2_wk * vu_sq)))
                     + (2 * mu_wk_sq * (np.power(M1_wk, 2) + (vd_sq
                                                              * gpr_sq)))
                     )))
        constterm = (1 / 8)\
            * ((np.power(g2_sq, 3) * np.power(M1_wk, 2)
                * (np.power(vd_sq, 2) + (vd_sq * vu_sq)))
               + (np.power(gpr_sq * M2_wk, 2)
                  * ((gpr_sq * (np.power(vd_sq, 2) + (vu_sq * vd_sq)))
                     - (2 * np.power(M1_wk, 2) * vd_sq)
                     - (4 * vd_sq * mu_wk_sq)))
               + (g2_sq * gpr_sq
                  * ((M1_wk * ((4 * gpr_sq * M2_wk
                                * (np.power(vd_sq, 2) + (vd_sq * vu_sq)))
                               - (M1_wk * ((4 * np.power(M2_wk, 2) * vd_sq)
                                           + (gpr_sq * (np.power(vd_sq, 2)
                                                        + (vu_sq * vd_sq)))
                                           ))))
                     + (4 * (np.power(M1_wk, 2) + np.power(M2_wk, 2)
                             - (4 * M1_wk * M2_wk))
                        * vd_sq * mu_wk_sq)))
               + (np.power(g2_sq, 2)
                  * ((4 * gpr_sq * M1_wk * M2_wk * (np.power(vd_sq, 2)
                                                    + (vu_sq * vd_sq)))
                     - (gpr_sq * np.power(M2_wk, 2) * (np.power(vd_sq, 2)
                                                       + (vu_sq * vd_sq)))
                     - (2 * np.power(M1_wk, 2) * vd_sq
                        * (np.power(M2_wk, 2) + (2 * mu_wk_sq))))))
        mynum = (cubicterm * np.power(msnsq, 3))\
            + (quadrterm * np.power(msnsq, 2))\
            + (linterm * msnsq) + constterm
        return mynum

    def neutralinodd_num(msnsq):
        """
        Return numerator for one-loop dd correction derivative term of
            neutralino.

        Parameters
        ----------
        msnsq : Float. 
            Neutralino squared mass used for evaluating results.

        """
        cubicterm = (-1) * (np.power(g2_wk, 2) + np.power(gpr_wk, 2))
        quadrterm = (1 / 4)\
            * ((np.power(g2_sq, 2) * ((2 * v_sq) + vu_sq))
               + (gpr_sq * ((4 * np.power(M2_wk, 2))
                            + (gpr_sq * ((2 * v_sq) + vu_sq))
                            - (4 * mu_wk_sq)))
               + (g2_sq * ((4 * np.power(M1_wk, 2))
                           + (gpr_sq * ((4 * v_sq) + (2 * vu_sq)))
                           - (4 * mu_wk_sq))))
        linterm = (1 / 2)\
            * ((np.power(gpr_sq, 2)
                * (((np.power(M1_wk, 2) + mu_wk_sq) * vu_sq)
                   - (np.power(M2_wk, 2) * v_sq)))
               + (np.power(g2_sq, 2)
                  * ((vu_sq * (np.power(M2_wk, 2) + mu_wk_sq))
                     - (v_sq * np.power(M1_wk, 2))))
               + (2 * gpr_sq * mu_wk_sq * np.power(M2_wk, 2))
               + (g2_sq
                  * ((gpr_sq * ((np.power(M1_wk - M2_wk, 2) * vu_sq)
                                - (2 * M1_wk * M2_wk * vd_sq)))
                     + (2 * mu_wk_sq * (np.power(M1_wk, 2) + (vu_sq
                                                              * gpr_sq)))
                     )))
        constterm = (1 / 8)\
            * ((np.power(g2_sq, 3) * np.power(M1_wk, 2)
                * (np.power(vu_sq, 2) + (vd_sq * vu_sq)))
               + (np.power(gpr_sq * M2_wk, 2)
                  * ((gpr_sq * (np.power(vu_sq, 2) + (vu_sq * vd_sq)))
                     - (2 * np.power(M1_wk, 2) * vu_sq)
                     - (4 * vu_sq * mu_wk_sq)))
               + (g2_sq * gpr_sq
                  * ((4 * (np.power(M1_wk, 2) - (4 * M1_wk * M2_wk)
                           + np.power(M2_wk, 2))
                      * vu_sq * mu_wk_sq)
                     - (gpr_sq * M1_wk * (M1_wk - (4 * M2_wk))
                        * (np.power(vu_sq, 2) + (vu_sq * vd_sq)))))
               + (np.power(g2_sq, 2)
                  * ((4 * gpr_sq * M1_wk * M2_wk * (np.power(vu_sq, 2)
                                                    + (vu_sq * vd_sq)))
                     - (gpr_sq * np.power(M2_wk, 2) * (np.power(vu_sq, 2)
                                                       + (vu_sq * vd_sq)))
                     - (2 * np.power(M1_wk, 2) * vu_sq
                        * (np.power(M2_wk, 2) + (2 * mu_wk_sq))))))
        mynum = (cubicterm * np.power(msnsq, 3))\
            + (quadrterm * np.power(msnsq, 2))\
            + (linterm * msnsq) + constterm
        return mynum

    def neutralinoud_num(msnsq):
        """
        Return numerator for one-loop ud correction derivative term of
            neutralino.

        Parameters
        ----------
        msnsq : Float. 
            Neutralino squared mass used for evaluating results.

        """
        cubicterm = 0
        quadrterm = (-1 / 2) * vd * vu * np.power(gpr_sq + g2_sq, 2)
        linterm = (-1)\
            * (vd * vu * (gpr_sq + g2_sq)
               * ((gpr_sq * (M1_wk - mu_wk) * (M1_wk + mu_wk))
                  + (g2_sq * (M2_wk - mu_wk) * (M2_wk + mu_wk))))
        constterm = (-1 / 4) * (gpr_sq + g2_sq) * vd * vu\
            * ((np.power(g2_sq * M1_wk, 2) * v_sq)
               + (gpr_sq * np.power(M2_wk, 2)
                  * ((4 * mu_wk_sq) - (2 * np.power(M1_wk, 2))
                     + (gpr_sq * v_sq)))
               - (g2_sq * ((gpr_sq * np.power(M2_wk, 2) * v_sq)
                           + (np.power(M1_wk, 2)
                              * ((2 * np.power(M2_wk, 2))
                                 + (v_sq * gpr_sq)
                                 - (4 * mu_wk_sq)))
                           - (4 * gpr_sq * M1_wk * M2_wk * v_sq))))
        mynum = (cubicterm * np.power(msnsq, 3))\
            + (quadrterm * np.power(msnsq, 2))\
            + (linterm * msnsq) + constterm
        return mynum

    def sigmauu_neutralino(msnsq):
        """
        Return one-loop correction Sigma_u^u(neutralino).

        Parameters
        ----------
        msnsq : Float.
            Neutralino squared mass.

        """
        sigma_uu_neutralino = ((-1) * loop_fac) \
            * ((neutralinouu_num(np.abs(msnsq))
                / neutralino_denom(np.abs(msnsq)))
               * logfunc2(np.abs(msnsq)))
        return sigma_uu_neutralino

    def sigmadd_neutralino(msnsq):
        """
        Return one-loop correction Sigma_d^d(neutralino).

        Parameters
        ----------
        msnsq : Float.
            Neutralino squared mass.

        """
        sigma_dd_neutralino = ((-1) * loop_fac) \
            * ((neutralinodd_num(np.abs(msnsq))
                / neutralino_denom(np.abs(msnsq)))
               * logfunc2(np.abs(msnsq)))
        return sigma_dd_neutralino

    def sigmaud_neutralino(msnsq):
        """
        Return one-loop correction Sigma_d^d(neutralino).

        Parameters
        ----------
        msnsq : Float.
            Neutralino squared mass.

        """
        sigma_ud_neutralino = ((-1) * loop_fac) \
            * ((neutralinoud_num(np.abs(msnsq))
                / neutralino_denom(np.abs(msnsq)))
               * logfunc2(np.abs(msnsq)))
        return sigma_ud_neutralino

    ##### Radiative corrections from chargino sector #####
    charginouu_num = np.power(M2_wk, 2) + (np.power(g2_wk, 2)
                                           * (vu_sq - vd_sq))\
        + np.abs(mu_wk_sq)
    charginodd_num = np.power(M2_wk, 2) - (np.power(g2_wk, 2)
                                           * (vu_sq - vd_sq))\
        + np.abs(mu_wk_sq)
    charginoud_num = 2 * gpr_sq * M2_wk * mu_wk
    chargino_den = msC2sq - msC1sq
    sigmauu_chargino1 = (-1) * ((g2_sq / 2) * loop_fac)\
        * (1 - (charginouu_num / chargino_den)) * logfunc2(msC1sq)
    sigmauu_chargino2 = (-1) * ((g2_sq / 2) * loop_fac)\
        * (1 + (charginouu_num / chargino_den)) * logfunc2(msC2sq)
    sigmadd_chargino1 = (-1) * ((g2_sq / 2) * loop_fac)\
        * (1 - (charginodd_num / chargino_den)) * logfunc2(msC1sq)
    sigmadd_chargino2 = (-1) * ((g2_sq / 2) * loop_fac)\
        * (1 + (charginodd_num / chargino_den)) * logfunc2(msC2sq)
    sigmaud_chargino1 = (-1) * loop_fac\
        * ((-1) * charginoud_num / chargino_den) * logfunc2(msC1sq)
    sigmaud_chargino2 = (-1) * loop_fac\
        * (charginoud_num / chargino_den) * logfunc2(msC2sq)

    ##### Radiative corrections from Higgs bosons sector #####

    higgsuu_num = mz_q_sq + (mA0sq * (1 + (4 * np.cos(2 * beta_wk))
                                      + (2 * np.power(np.cos(2 * beta_wk), 2))
                                      ))
    higgsdd_num = mz_q_sq + (mA0sq * (1 - (4 * np.cos(2 * beta_wk))
                                      + (2 * np.power(np.cos(2 * beta_wk), 2))
                                      ))
    higgsud_num = 4 * (g2_sq + gpr_sq) * mA0sq * np.cos(beta_wk)\
        * np.sin(beta_wk)
    higgs_den = mH0sq - mh0sq
    higgsud_den = np.sqrt(np.power(mA0sq - mz_q_sq, 2)
                          + (8 * mA0sq * cossqb * vu_sq
                             * (g2_sq + gpr_sq)))
    sigmauu_h0 = loop_fac * logfunc2(mh0sq) * (gz_sq)\
        * (1 - (higgsuu_num / higgs_den))
    sigmauu_heavy_h0 = loop_fac * logfunc2(mH0sq) * (gz_sq)\
        * (1 + (higgsuu_num / higgs_den))
    sigmadd_h0 = loop_fac * logfunc2(mh0sq) * (gz_sq)\
        * (1 - (higgsdd_num / higgs_den))
    sigmadd_heavy_h0 = loop_fac * logfunc2(mH0sq) * (gz_sq)\
        * (1 + (higgsdd_num / higgs_den))
    sigmaud_h0 = (loop_fac / 2.0)\
        * (higgsud_num / higgsud_den) * logfunc2(mh0sq)
    sigmaud_heavy_h0 = (loop_fac / 2.0)\
        * (higgsud_num / higgsud_den) * logfunc2(mH0sq)
    sigmauu_h_pm  = (np.power((g2_wk), 2) * loop_fac
                     / 2) * logfunc2(mH_pmsq)
    sigmadd_h_pm = sigmauu_h_pm
    sigmaud_h_pm = 0

    ##### Radiative corrections from weak vector bosons sector #####
    sigmauu_w_pm = (3 * np.power((g2_wk), 2) * loop_fac
                    / 2) * logfunc2(m_w_sq)
    sigmadd_w_pm = sigmauu_w_pm
    sigmaud_w_pm = 0
    sigmauu_z0 = (3 / 4) * loop_fac * (gpr_sq + g2_sq)\
        * logfunc2(mz_q_sq)
    sigmadd_z0 = sigmauu_z0
    sigmaud_z0 = 0

    ##### Radiative corrections from SM fermions sector #####
    sigmauu_top = (-6) * np.power(yt_wk, 2) * loop_fac\
        * logfunc2(mymtsq)
    sigmadd_top = 0
    sigmaud_top = 0
    sigmauu_bottom = 0
    sigmadd_bottom = (-6) * np.power(yb_wk, 2) * loop_fac\
        * logfunc2(mymbsq)
    sigmaud_bottom = 0
    sigmauu_tau = 0
    sigmadd_tau = (-2) * np.power(ytau_wk, 2) * loop_fac\
        * logfunc2(mymtausq)
    sigmaud_tau = 0
    sigmauu_charm = (-6) * np.power(yc_wk, 2) * loop_fac\
        * logfunc2(mymcsq)
    sigmadd_charm = 0
    sigmaud_charm = 0
    sigmauu_strange = 0
    sigmadd_strange = (-6) * np.power(ys_wk, 2) * loop_fac\
        * logfunc2(mymssq)
    sigmaud_strange = 0
    sigmauu_mu = 0
    sigmadd_mu = (-2) * np.power(ymu_wk, 2) * loop_fac\
        * logfunc2(mymmusq)
    sigmaud_mu = 0
    sigmauu_up = (-6) * np.power(yu_wk, 2) * loop_fac\
        * logfunc2(mymusq)
    sigmadd_up = 0
    sigmaud_up = 0
    sigmauu_down = 0
    sigmadd_down = (-6) * np.power(yd_wk, 2) * loop_fac\
        * logfunc2(mymdsq)
    sigmaud_down = 0
    sigmauu_elec = 0
    sigmadd_elec = (-2) * np.power(ye_wk, 2) * loop_fac\
        * logfunc2(mymesq)
    sigmaud_elec = 0

    ##### Radiative corrections from two-loop O(alpha_t alpha_s) sector #####
    # Corrections come from Dedes, Slavich paper, arXiv:hep-ph/0212132.
    # alpha_i = y_i^2 / (4 * pi)
    def sigmauu_2loop():
        def Deltafunc(x, y, z):
            mydelta = np.power(x, 2) + np.power(y, 2) + np.power(z, 2)\
                - (2 * ((x * y) + (x * z) + (y * z)))
            return mydelta

        def Phifunc(x, y, z):
            if(x / z < 1 and y / z < 1):
                myu = x / z
                myv = y / z
                mylambda = np.sqrt(np.abs(np.power((1 - myu - myv), 2)
                                          - (4 * myu * myv)))
                myxp = 0.5 * (1 + myu - myv - mylambda)
                myxm = 0.5 * (1 - myu + myv - mylambda)
                myphi = (1 / mylambda) * ((2 * np.log(np.abs(myxp))
                                           * np.log(np.abs(myxm)))
                                          - (np.log(np.abs(myu))
                                             * np.log(np.abs(myv)))
                                          - (2 * (spence(1 - myxp)
                                                  + spence(1 - myxm)))
                                          + (np.power(np.pi, 2) / 3))
            elif(x / z > 1 and y / z < 1):
                myu = z / x
                myv = y / x
                mylambda = np.sqrt(np.abs(np.power((1 - myu - myv), 2)
                                          - (4 * myu * myv)))
                myxp = 0.5 * (1 + myu - myv - mylambda)
                myxm = 0.5 * (1 - myu + myv - mylambda)
                myphi = (z / x) * (1 / mylambda)\
                    * ((2 * np.log(np.abs(myxp))
                        * np.log(np.abs(myxm)))
                       - (np.log(np.abs(myu))
                          * np.log(np.abs(myv)))
                       - (2 * (spence(1 - myxp)
                               + spence(1 - myxm)))
                       + (np.power(np.pi, 2) / 3))
            elif(x/z > 1 and y/ z > 1 and x > y):
                myu = z / x
                myv = y / x
                mylambda = np.sqrt(np.abs(np.power((1 - myu - myv), 2)
                                          - (4 * myu * myv)))
                myxp = 0.5 * (1 + myu - myv - mylambda)
                myxm = 0.5 * (1 - myu + myv - mylambda)
                myphi = (z / x) * (1 / mylambda)\
                    * ((2 * np.log(np.abs(myxp))
                        * np.log(np.abs(myxm)))
                       - (np.log(np.abs(myu))
                          * np.log(np.abs(myv)))
                       - (2 * (spence(1 - myxp)
                               + spence(1 - myxm)))
                       + (np.power(np.pi, 2) / 3))
            elif(x / z < 1 and y / z > 1):
                myu = z / y
                myv = x / y
                mylambda = np.sqrt(np.abs(np.power((1 - myu - myv), 2)
                                          - (4 * myu * myv)))
                myxp = 0.5 * (1 + myu - myv - mylambda)
                myxm = 0.5 * (1 - myu + myv - mylambda)
                myphi = (z / y) * (1 / mylambda)\
                    * ((2 * np.log(np.abs(myxp))
                        * np.log(np.abs(myxm)))
                       - (np.log(np.abs(myu))
                          * np.log(np.abs(myv)))
                       - (2 * (spence(1 - myxp)
                               + spence(1 - myxm)))
                       + (np.power(np.pi, 2) / 3))
            elif (x / z > 1 and y / z > 1 and y > x):
                myu = z / y
                myv = x / y
                mylambda = np.sqrt(np.abs(np.power((1 - myu - myv), 2)
                                          - (4 * myu * myv)))
                myxp = 0.5 * (1 + myu - myv - mylambda)
                myxm = 0.5 * (1 - myu + myv - mylambda)
                myphi = (z / y) * (1 / mylambda)\
                    * ((2 * np.log(np.abs(myxp))
                        * np.log(np.abs(myxm)))
                       - (np.log(np.abs(myu))
                          * np.log(np.abs(myv)))
                       - (2 * (spence(1 - myxp)
                               + spence(1 - myxm)))
                       + (np.power(np.pi, 2) / 3))
            return myphi

        mst1sq = m_stop_1sq
        mst2sq = m_stop_2sq
        s2theta = 2 * mymt * ((at_wk / yt_wk) + (mu_wk / np.tan(beta_wk)))\
            / (mst1sq - mst2sq)
        s2sqtheta = np.power(s2theta, 2)
        c2sqtheta = 1 - s2sqtheta
        mglsq = np.power(M3_wk, 2)
        myunits = np.power(g3_wk, 2) * 4 * loop_fac_sq
        Q_renorm_sq = np.power(myQ, 2)
        myF = myunits\
            * (((4 * M3_wk * mymt / s2theta) * (1 + (4 * c2sqtheta)))
               - (((2 * (mst1sq - mst2sq)) + (4 * M3_wk * mymt / s2theta))
                  * np.log(mglsq / Q_renorm_sq)
                  * np.log(mymtsq / Q_renorm_sq))
               - (2 * (4 - s2sqtheta) * (mst1sq - mst2sq))
               + ((((4 * mst1sq * mst2sq)
                    - s2sqtheta * np.power((mst1sq + mst2sq), 2))
                   / (mst1sq - mst2sq))
                  * (np.log(np.abs(mst1sq / Q_renorm_sq)))
                  * (np.log(mst2sq / Q_renorm_sq)))
                 + ((((4 * (mglsq + mymtsq + (2 * mst1sq)))
                      - (s2sqtheta * ((3 * mst1sq) + mst2sq))
                      - ((16 * c2sqtheta * M3_wk * mymt * mst1sq)
                         / (s2theta * (mst1sq - mst2sq)))
                      - (4 * s2theta * M3_wk * mymt))
                     * np.log(np.abs(mst1sq / Q_renorm_sq)))
                    + ((mst1sq / (mst1sq - mst2sq))
                       * ((s2sqtheta * (mst1sq + mst2sq))
                          - ((4 * mst1sq) - (2 * mst2sq)))
                       * np.power(np.log(np.abs(mst1sq / Q_renorm_sq)), 2))
                    + (2 * (mst1sq - mglsq - mymtsq
                            + (M3_wk * mymt * s2theta)
                            + ((2 * c2sqtheta * M3_wk * mymt * mst1sq)
                               / (s2theta * (mst1sq - mst2sq))))
                       * np.log(mglsq * mymtsq
                                / (np.power(Q_renorm_sq, 2)))
                       * np.log(np.abs(mst1sq / Q_renorm_sq)))
                    + (((4 * M3_wk * mymt * c2sqtheta * (mymtsq - mglsq))
                        / (s2theta * (mst1sq - mst2sq)))
                       * np.log(mymtsq / mglsq)
                       * np.log(np.abs(mst1sq / Q_renorm_sq)))
                    + (((((4 * mglsq * mymtsq)
                          + (2 * Deltafunc(mglsq, mymtsq, mst1sq))) / mst1sq)
                        - (((2 * M3_wk * mymt * s2theta) / mst1sq)
                           * (mglsq + mymtsq - mst1sq))
                        + ((4 * c2sqtheta * M3_wk * mymt
                            * Deltafunc(mglsq, mymtsq, mst1sq))
                           / (s2theta * mst1sq * (mst1sq - mst2sq))))
                       * Phifunc(mglsq, mymtsq, mst1sq)))
                 - ((((4 * (mglsq + mymtsq + (2 * mst2sq)))
                      - (s2sqtheta * ((3 * mst2sq) + mst1sq))
                      - ((16 * c2sqtheta * M3_wk * mymt * mst2sq)
                         / (((-1) * s2theta) * (mst2sq - mst1sq)))
                      - ((-4) * s2theta * M3_wk * mymt))
                     * np.log(mst2sq / Q_renorm_sq))
                    + ((mst2sq / (mst2sq - mst1sq))
                       * ((s2sqtheta * (mst2sq + mst1sq))
                          - ((4 * mst2sq) - (2 * mst1sq)))
                       * np.power(np.log(mst2sq / Q_renorm_sq), 2))
                    + (2 * (mst2sq - mglsq - mymtsq
                            - (M3_wk * mymt * s2theta)
                            + ((2 * c2sqtheta * M3_wk * mymt * mst2sq)
                               / (s2theta * (mst1sq - mst2sq))))
                       * np.log(mglsq * mymtsq
                                / (np.power(Q_renorm_sq, 2)))
                       * np.log(mst2sq / Q_renorm_sq))
                    + (((4 * M3_wk * mymt * c2sqtheta * (mymtsq - mglsq))
                        / (s2theta * (mst1sq - mst2sq)))
                       * np.log(mymtsq / mglsq)
                       * np.log(mst2sq / Q_renorm_sq))
                    + (((((4 * mglsq * mymtsq)
                          + (2 * Deltafunc(mglsq, mymtsq, mst2sq))) / mst2sq)
                        - ((((-2) * M3_wk * mymt * s2theta) / mst2sq)
                           * (mglsq + mymtsq - mst2sq))
                        + ((4 * c2sqtheta * M3_wk * mymt
                            * Deltafunc(mglsq, mymtsq, mst2sq))
                           / (s2theta * mst2sq * (mst1sq - mst2sq))))
                       * Phifunc(mglsq, mymtsq, mst2sq))))
        myG = myunits\
            * ((5 * M3_wk * s2theta * (mst1sq - mst2sq) / mymt)
               - (10 * (mst1sq + mst2sq - (2 * mymtsq)))
               - (4 * mglsq) + ((12 * mymtsq)
                                * (np.power(np.log(mymtsq / Q_renorm_sq), 2)
                                   - (2 * np.log(mymtsq / Q_renorm_sq))))
               + (((4 * mglsq) - ((M3_wk * s2theta / mymt)
                                  * (mst1sq - mst2sq)))
                  * np.log(mglsq / Q_renorm_sq) * np.log(mymtsq / Q_renorm_sq))
               + (s2sqtheta * (mst1sq + mst2sq)
                  * np.log(np.abs(mst1sq / Q_renorm_sq))
                  * np.log(mst2sq / Q_renorm_sq))
               + ((((4 * (mglsq + mymtsq + (2 * mst1sq)))
                    + (s2sqtheta * (mst1sq - mst2sq))
                    - ((4 * M3_wk * s2theta / mymt) * (mymtsq + mst1sq)))
                   * np.log(np.abs(mst1sq / Q_renorm_sq)))
                  + (((M3_wk * s2theta * ((5 * mymtsq) - mglsq + mst1sq)
                       / mymt)
                      - (2 * (mglsq + 2 * mymtsq)))
                     * np.log(mymtsq / Q_renorm_sq)
                     * np.log(np.abs(mst1sq / Q_renorm_sq)))
                  + (((M3_wk * s2theta * (mglsq - mymtsq + mst1sq) / mymt)
                      - (2 * mglsq))
                     * np.log(mglsq / Q_renorm_sq)
                     * np.log(np.abs(mst1sq / Q_renorm_sq)))
                  - ((2 + s2sqtheta) * mst1sq
                     * np.power(np.log(np.abs(mst1sq / Q_renorm_sq)), 2))
                  + (((2 * mglsq * (mglsq + mymtsq - mst1sq
                                    - (2 * M3_wk * mymt * s2theta)) / mst1sq)
                      + ((M3_wk * s2theta / (mymt * mst1sq))
                         * Deltafunc(mglsq, mymtsq, mst1sq)))
                     * Phifunc(mglsq, mymtsq, mst1sq)))
               + ((((4 * (mglsq + mymtsq + (2 * mst2sq)))
                    + (s2sqtheta * (mst2sq - mst1sq))
                    - (((-4) * M3_wk * s2theta / mymt) * (mymtsq + mst2sq)))
                   * np.log(mst2sq / Q_renorm_sq))
                  + ((((-1) * M3_wk * s2theta * ((5 * mymtsq) - mglsq + mst2sq)
                       / mymt)
                      - (2 * (mglsq + 2 * mymtsq)))
                     * np.log(mymtsq / Q_renorm_sq)
                     * np.log(mst2sq / Q_renorm_sq))
                  + ((((-1) * M3_wk * s2theta * (mglsq - mymtsq + mst2sq)
                       / mymt)
                      - (2 * mglsq))
                     * np.log(mglsq / Q_renorm_sq)
                     * np.log(mst2sq / Q_renorm_sq))
                  - ((2 + s2sqtheta) * mst2sq
                     * np.power(np.log(mst2sq / Q_renorm_sq), 2))
                  + (((2 * mglsq
                       * (mglsq + mymtsq - mst2sq
                          + (2 * M3_wk * mymt * s2theta)) / mst2sq)
                      + ((M3_wk * (-1) * s2theta / (mymt * mst2sq))
                         * Deltafunc(mglsq, mymtsq, mst2sq)))
                     * Phifunc(mglsq, mymtsq, mst2sq))))
        mysigmauu_2loop = ((mymt * (at_wk / yt_wk) * s2theta * myF)
                           + 2 * np.power(mymt, 2) * myG)\
            / (np.power((vHiggs_wk), 2) * sinsqb)
        return mysigmauu_2loop

    def sigmadd_2loop():
        def Deltafunc(x,y,z):
            mydelta = np.power(x, 2) + np.power(y, 2) + np.power(z, 2)\
                - (2 * ((x * y) + (x * z) + (y * z)))
            return mydelta

        def Phifunc(x, y, z):
            if(x / z < 1 and y / z < 1):
                myu = x / z
                myv = y / z
                mylambda = np.sqrt(np.abs(np.power((1 - myu - myv), 2)
                                          - (4 * myu * myv)))
                myxp = 0.5 * (1 + myu - myv - mylambda)
                myxm = 0.5 * (1 - myu + myv - mylambda)
                myphi = (1 / mylambda) * ((2 * np.log(np.abs(myxp))
                                           * np.log(np.abs(myxm)))
                                          - (np.log(np.abs(myu))
                                             * np.log(np.abs(myv)))
                                          - (2 * (spence(1 - myxp)
                                                  + spence(1 - myxm)))
                                          + (np.power(np.pi, 2) / 3))
            elif(x / z > 1 and y / z < 1):
                myu = z / x
                myv = y / x
                mylambda = np.sqrt(np.abs(np.power((1 - myu - myv), 2)
                                          - (4 * myu * myv)))
                myxp = 0.5 * (1 + myu - myv - mylambda)
                myxm = 0.5 * (1 - myu + myv - mylambda)
                myphi = (z / x) * (1 / mylambda)\
                    * ((2 * np.log(np.abs(myxp))
                        * np.log(np.abs(myxm)))
                       - (np.log(np.abs(myu))
                          * np.log(np.abs(myv)))
                       - (2 * (spence(1 - myxp)
                               + spence(1 - myxm)))
                       + (np.power(np.pi, 2) / 3))
            elif(x/z > 1 and y/ z > 1 and x > y):
                myu = z / x
                myv = y / x
                mylambda = np.sqrt(np.abs(np.power((1 - myu - myv), 2)
                                          - (4 * myu * myv)))
                myxp = 0.5 * (1 + myu - myv - mylambda)
                myxm = 0.5 * (1 - myu + myv - mylambda)
                myphi = (z / x) * (1 / mylambda)\
                    * ((2 * np.log(np.abs(myxp))
                        * np.log(np.abs(myxm)))
                       - (np.log(np.abs(myu))
                          * np.log(np.abs(myv)))
                       - (2 * (spence(1 - myxp)
                               + spence(1 - myxm)))
                       + (np.power(np.pi, 2) / 3))
            elif(x / z < 1 and y / z > 1):
                myu = z / y
                myv = x / y
                mylambda = np.sqrt(np.abs(np.power((1 - myu - myv), 2)
                                          - (4 * myu * myv)))
                myxp = 0.5 * (1 + myu - myv - mylambda)
                myxm = 0.5 * (1 - myu + myv - mylambda)
                myphi = (z / y) * (1 / mylambda)\
                    * ((2 * np.log(np.abs(myxp))
                        * np.log(np.abs(myxm)))
                       - (np.log(np.abs(myu))
                          * np.log(np.abs(myv)))
                       - (2 * (spence(1 - myxp)
                               + spence(1 - myxm)))
                       + (np.power(np.pi, 2) / 3))
            elif (x / z > 1 and y / z > 1 and y > x):
                myu = z / y
                myv = x / y
                mylambda = np.sqrt(np.abs(np.power((1 - myu - myv), 2)
                                          - (4 * myu * myv)))
                myxp = 0.5 * (1 + myu - myv - mylambda)
                myxm = 0.5 * (1 - myu + myv - mylambda)
                myphi = (z / y) * (1 / mylambda)\
                    * ((2 * np.log(np.abs(myxp))
                        * np.log(np.abs(myxm)))
                       - (np.log(np.abs(myu))
                          * np.log(np.abs(myv)))
                       - (2 * (spence(1 - myxp)
                               + spence(1 - myxm)))
                       + (np.power(np.pi, 2) / 3))
            return myphi

        mst1sq = m_stop_1sq
        mst2sq = m_stop_2sq
        Q_renorm_sq=np.power(myQ, 2)
        s2theta = (2 * mymt * ((at_wk / yt_wk)
                               + (mu_wk / np.tan(beta_wk))))\
            / (mst1sq - mst2sq)
        s2sqtheta = np.power(s2theta, 2)
        c2sqtheta = 1 - s2sqtheta
        mglsq = np.power(M3_wk, 2)
        myunits = np.power(g3_wk, 2) * 4\
            / np.power((16 * np.power(np.pi, 2)), 2)
        myF = myunits\
            * ((4 * M3_wk * mymt / s2theta) * (1 + 4 * c2sqtheta)
               - (((2 * (mst1sq - mst2sq))
                  + (4 * M3_wk * mymt / s2theta))
                  * np.log(mglsq / Q_renorm_sq)
                  * np.log(mymtsq / Q_renorm_sq))
               - (2 * (4 - s2sqtheta)
                  * (mst1sq - mst2sq))
               + ((((4 * mst1sq * mst2sq)
                    - s2sqtheta * np.power((mst1sq + mst2sq), 2))
                   / (mst1sq - mst2sq))
                  * (np.log(np.abs(mst1sq / Q_renorm_sq)))
                  * (np.log(mst2sq / Q_renorm_sq)))
               + ((((4 * (mglsq + mymtsq + (2 * mst1sq)))
                   - (s2sqtheta * ((3 * mst1sq) + mst2sq))
                   - ((16 * c2sqtheta * M3_wk * mymt * mst1sq)
                      / (s2theta * (mst1sq - mst2sq)))
                   - (4 * s2theta * M3_wk * mymt))
                   * np.log(np.abs(mst1sq / Q_renorm_sq)))
                  + ((mst1sq / (mst1sq - mst2sq))
                     * ((s2sqtheta * (mst1sq + mst2sq))
                        - ((4 * mst1sq) - (2 * mst2sq)))
                     * np.power(np.log(np.abs(mst1sq / Q_renorm_sq)), 2))
                  + (2 * (mst1sq - mglsq - mymtsq
                          + (M3_wk * mymt * s2theta)
                          + ((2 * c2sqtheta * M3_wk * mymt * mst1sq)
                             / (s2theta * (mst1sq - mst2sq))))
                     * np.log(mglsq * mymtsq
                              / (np.power(Q_renorm_sq, 2)))
                     * np.log(np.abs(mst1sq / Q_renorm_sq)))
                  + (((4 * M3_wk * mymt * c2sqtheta * (mymtsq - mglsq))
                      / (s2theta * (mst1sq - mst2sq)))
                     * np.log(mymtsq / mglsq)
                     * np.log(np.abs(mst1sq / Q_renorm_sq)))
                  + (((((4 * mglsq * mymtsq)
                        + (2 * Deltafunc(mglsq, mymtsq, mst1sq))) / mst1sq)
                      - (((2 * M3_wk * mymt * s2theta) / mst1sq)
                         * (mglsq + mymtsq - mst1sq))
                      + ((4 * c2sqtheta * M3_wk * mymt
                          * Deltafunc(mglsq, mymtsq, mst1sq))
                         / (s2theta * mst1sq * (mst1sq - mst2sq))))
                     * Phifunc(mglsq, mymtsq, mst1sq)))
               - ((((4 * (mglsq + mymtsq + (2 * mst2sq)))
                   - (s2sqtheta * ((3 * mst2sq) + mst1sq))
                   - ((16 * c2sqtheta * M3_wk * mymt * mst2sq)
                      / (((-1) * s2theta) * (mst2sq - mst1sq)))
                   - ((-4) * s2theta * M3_wk * mymt))
                   * np.log(mst2sq / Q_renorm_sq))
                  + ((mst2sq / (mst2sq - mst1sq))
                     * ((s2sqtheta * (mst2sq + mst1sq))
                        - ((4 * mst2sq) - (2 * mst1sq)))
                     * np.power(np.log(mst2sq / Q_renorm_sq), 2))
                  + (2 * (mst2sq - mglsq - mymtsq
                          - (M3_wk * mymt * s2theta)
                          + ((2 * c2sqtheta * M3_wk * mymt * mst2sq)
                             / (s2theta * (mst1sq - mst2sq))))
                     * np.log(mglsq * mymtsq
                              / (np.power(Q_renorm_sq, 2)))
                     * np.log(mst2sq / Q_renorm_sq))
                  + (((4 * M3_wk * mymt * c2sqtheta * (mymtsq - mglsq))
                      / (s2theta * (mst1sq - mst2sq)))
                     * np.log(mymtsq / mglsq)
                     * np.log(mst2sq / Q_renorm_sq))
                  + (((((4 * mglsq * mymtsq)
                        + (2 * Deltafunc(mglsq, mymtsq, mst2sq))) / mst2sq)
                      - ((((-2) * M3_wk * mymt * s2theta) / mst2sq)
                         * (mglsq + mymtsq - mst2sq))
                      + ((4 * c2sqtheta * M3_wk * mymt
                          * Deltafunc(mglsq, mymtsq, mst2sq))
                         / (s2theta * mst2sq * (mst1sq - mst2sq))))
                     * Phifunc(mglsq, mymtsq, mst2sq))))
        mysigmadd_2loop = (mymt * mu_wk * (1 / np.tan(beta_wk))
                           * s2theta * myF)\
            / (np.power((vHiggs_wk), 2) * cossqb)
        return mysigmadd_2loop

    ##### Total radiative corrections #####
    sigmauu_tot = sigmauu_stop_1 + sigmauu_stop_2 + sigmauu_sbot_1\
        + sigmauu_sbot_2 + sigmauu_stau_1 + sigmauu_stau_2\
        + sigmauu_stau_sneut + sigmauu_scharm_1 \
        + sigmauu_scharm_2 + sigmauu_sstrange_1 + sigmauu_sstrange_2\
        + sigmauu_smu_1 + sigmauu_smu_2 + sigmauu_smu_sneut + sigmauu_sup_1\
        + sigmauu_sup_2 + sigmauu_sdown_1 + sigmauu_sdown_2 + sigmauu_se_1\
        + sigmauu_se_2 + sigmauu_selec_sneut + sigmauu_neutralino(msN1sq)\
        + sigmauu_neutralino(msN2sq) + sigmauu_neutralino(msN3sq)\
        + sigmauu_neutralino(msN4sq) + sigmauu_chargino1\
        + sigmauu_chargino2\
        + sigmauu_h0 + sigmauu_heavy_h0 + sigmauu_h_pm + sigmauu_w_pm\
        + sigmauu_z0 + sigmauu_top + sigmauu_bottom + sigmauu_tau\
        + sigmauu_charm + sigmauu_strange + sigmauu_mu\
        + sigmauu_up + sigmauu_down + sigmauu_elec + sigmauu_2loop()
    #print("Sigma_u^u(total) = " + str(sigmauu_tot))

    sigmadd_tot = sigmadd_stop_1 + sigmadd_stop_2 + sigmadd_sbot_1\
        + sigmadd_sbot_2 + sigmadd_stau_1 + sigmadd_stau_2\
        + sigmadd_stau_sneut + sigmadd_scharm_1 \
        + sigmadd_scharm_2 + sigmadd_sstrange_1 + sigmadd_sstrange_2\
        + sigmadd_smu_1 + sigmadd_smu_2 + sigmadd_smu_sneut\
        + sigmadd_sup_1\
        + sigmadd_sup_2 + sigmadd_sdown_1 + sigmadd_sdown_2 + sigmadd_se_1\
        + sigmadd_se_2 + sigmadd_selec_sneut + sigmadd_neutralino(msN1sq)\
        + sigmadd_neutralino(msN2sq) + sigmadd_neutralino(msN3sq)\
        + sigmadd_neutralino(msN4sq) + sigmadd_chargino1\
        + sigmadd_chargino2\
        + sigmadd_h0 + sigmadd_heavy_h0 + sigmadd_h_pm + sigmadd_w_pm\
        + sigmadd_z0 + sigmadd_top + sigmadd_bottom + sigmadd_tau\
        + sigmadd_charm + sigmadd_strange + sigmadd_mu\
        + sigmadd_up + sigmadd_down + sigmadd_elec + sigmadd_2loop()
    #print("Sigma_d^d(total) = " + str(sigmadd_tot))

    sigmaud_tot = sigmaud_stop_1 + sigmaud_stop_2 + sigmaud_sbot_1\
        + sigmaud_sbot_2 + sigmaud_stau_1 + sigmaud_stau_2\
        + sigmaud_stau_sneut + sigmaud_scharm_1 \
        + sigmaud_scharm_2 + sigmaud_sstrange_1 + sigmaud_sstrange_2\
        + sigmaud_smu_1 + sigmaud_smu_2 + sigmaud_smu_sneut + sigmaud_sup_1\
        + sigmaud_sup_2 + sigmaud_sdown_1 + sigmaud_sdown_2 + sigmaud_se_1\
        + sigmaud_se_2 + sigmaud_selec_sneut + sigmaud_neutralino(msN1sq)\
        + sigmaud_neutralino(msN2sq) + sigmaud_neutralino(msN3sq)\
        + sigmaud_neutralino(msN4sq) + sigmaud_chargino1\
        + sigmaud_chargino2\
        + sigmaud_h0 + sigmaud_heavy_h0 + sigmaud_h_pm + sigmaud_w_pm\
        + sigmaud_z0 + sigmaud_top + sigmaud_bottom + sigmaud_tau\
        + sigmaud_charm + sigmaud_strange + sigmaud_mu\
        + sigmaud_up + sigmaud_down + sigmaud_elec# + sigmaud_2loop()
    #print("Sigma_u^d(total) = " + str(sigmaud_tot))
    # Return list of radiative corrections
    # (0: sigmauu_tot, 1: sigmadd_tot, 2: sigmaud_tot, 3: sigmauu_stop_1,
    #  4: sigmadd_stop_1, 5: sigmaud_stop_1, 6: sigmauu_stop_2,
    #  7: sigmadd_stop_2, 8: sigmaud_stop_2, 9: sigmauu_sbot_1,
    #  10: sigmadd_sbot_1, 11: sigmaud_sbot_1, 12: sigmauu_sbot_2,
    #  13: sigmadd_sbot_2, 14: sigmaud_sbot_2, 15: sigmauu_stau_1,
    #  16: sigmadd_stau_1, 17: sigmaud_stau_1, 18: sigmauu_stau_2,
    #  19: sigmadd_stau_2, 20: sigmaud_stau_2, 21: sigmauu_stau_sneut,
    #  22: sigmadd_stau_sneut, 23: sigmaud_stau_sneut, 24: sigmauu_scharm_1,
    #  25: sigmadd_scharm_1, 26: sigmaud_scharm_1, 27: sigmauu_scharm_2,
    #  28: sigmadd_scharm_2, 29: sigmaud_scharm_2, 30: sigmauu_sstrange_1,
    #  31: sigmadd_sstrange_1, 32: sigmaud_sstrange_1, 33: sigmauu_sstrange_2,
    #  34: sigmadd_sstrange_2, 35: sigmaud_sstrange_2, 36: sigmauu_smu_1,
    #  37: sigmadd_smu_1, 38: sigmaud_smu_1, 39: sigmauu_smu_2,
    #  40: sigmadd_smu_2, 41: sigmaud_smu_2, 42: sigmauu_smu_sneut,
    #  43: sigmadd_smu_sneut, 44: sigmaud_smu_sneut, 45: sigmauu_sup_1,
    #  46: sigmadd_sup_1, 47: sigmaud_sup_1, 48: sigmauu_sup_2,
    #  49: sigmadd_sup_2, 50: sigmaud_sup_2, 51: sigmauu_sdown_1,
    #  52: sigmadd_sdown_1, 53: sigmaud_sdown_1, 54: sigmauu_sdown_2,
    #  55: sigmadd_sdown_2, 56: sigmaud_sdown_2, 57: sigmauu_se_1,
    #  58: sigmadd_se_1, 59: sigmaud_se_1, 60: sigmauu_se_2, 61: sigmadd_se_2,
    #  62: sigmaud_se_2, 63: sigmauu_selec_sneut, 64: sigmadd_selec_sneut,
    #  65: sigmaud_selec_sneut, 66: sigmauu_neutralino(msN1sq),
    #  67: sigmadd_neutralino(msN1sq), 68: sigmaud_neutralino(msN1sq),
    #  69: sigmauu_neutralino(msN2sq), 70: sigmadd_neutralino(msN2sq),
    #  71: sigmaud_neutralino(msN2sq), 72: sigmauu_neutralino(msN3sq),
    #  73: sigmadd_neutralino(msN3sq), 74: sigmaud_neutralino(msN3sq),
    #  75: sigmauu_neutralino(msN4sq), 76: sigmadd_neutralino(msN4sq),
    #  77: sigmaud_neutralino(msN4sq), 78: sigmauu_chargino1,
    #  79: sigmadd_chargino1, 80: sigmaud_chargino1, 81: sigmauu_chargino2,
    #  82: sigmadd_chargino2, 83: sigmaud_chargino2, 84: sigmauu_h0,
    #  85: sigmadd_h0, 86: sigmaud_h0, 87: sigmauu_heavy_h0,
    #  88: sigmadd_heavy_h0, 89: sigmaud_heavy_h0, 90: sigmauu_h_pm,
    #  91: sigmadd_h_pm, 92: sigmaud_h_pm, 93: sigmauu_w_pm, 94: sigmadd_w_pm,
    #  95: sigmaud_w_pm, 96: sigmauu_z0, 97: sigmadd_z0, 98: sigmaud_z0,
    #  99: sigmauu_top, 100: sigmadd_top, 101: sigmaud_top,
    #  102: sigmauu_bottom, 103: sigmadd_bottom, 104: sigmaud_bottom,
    #  105: sigmauu_tau, 106: sigmadd_tau, 107: sigmaud_tau,
    #  108: sigmauu_charm, 109: sigmadd_charm, 110: sigmaud_charm,
    #  111: sigmauu_strange, 112: sigmadd_strange, 113: sigmaud_strange,
    #  114: sigmauu_mu, 115: sigmadd_mu, 116: sigmaud_mu, 117: sigmauu_up,
    #  118: sigmadd_up, 119: sigmaud_up, 120: sigmauu_down, 121: sigmadd_down,
    #  122: sigmaud_down, 123: sigmauu_elec, 124: sigmadd_elec,
    #  125: sigmaud_elec, 126: sigmauu_2loop(), 127: sigmadd_2loop())
    return [sigmauu_tot, sigmadd_tot, sigmaud_tot, sigmauu_stop_1,
            sigmadd_stop_1, sigmaud_stop_1, sigmauu_stop_2, sigmadd_stop_2,
            sigmaud_stop_2, sigmauu_sbot_1, sigmadd_sbot_1, sigmaud_sbot_1,
            sigmauu_sbot_2, sigmadd_sbot_2, sigmaud_sbot_2, sigmauu_stau_1,
            sigmadd_stau_1, sigmaud_stau_1, sigmauu_stau_2, sigmadd_stau_2,
            sigmaud_stau_2, sigmauu_stau_sneut, sigmadd_stau_sneut,
            sigmaud_stau_sneut, sigmauu_scharm_1, sigmadd_scharm_1,
            sigmaud_scharm_1, sigmauu_scharm_2, sigmadd_scharm_2,
            sigmaud_scharm_2, sigmauu_sstrange_1, sigmadd_sstrange_1,
            sigmaud_sstrange_1, sigmauu_sstrange_2, sigmadd_sstrange_2,
            sigmaud_sstrange_2, sigmauu_smu_1, sigmadd_smu_1, sigmaud_smu_1,
            sigmauu_smu_2, sigmadd_smu_2, sigmaud_smu_2, sigmauu_smu_sneut,
            sigmadd_smu_sneut, sigmaud_smu_sneut, sigmauu_sup_1, sigmadd_sup_1,
            sigmaud_sup_1, sigmauu_sup_2, sigmadd_sup_2, sigmaud_sup_2,
            sigmauu_sdown_1, sigmadd_sdown_1, sigmaud_sdown_1, sigmauu_sdown_2,
            sigmadd_sdown_2, sigmaud_sdown_2, sigmauu_se_1, sigmadd_se_1,
            sigmaud_se_1, sigmauu_se_2, sigmadd_se_2, sigmaud_se_2,
            sigmauu_selec_sneut, sigmadd_selec_sneut, sigmaud_selec_sneut,
            sigmauu_neutralino(msN1sq), sigmadd_neutralino(msN1sq),
            sigmaud_neutralino(msN1sq), sigmauu_neutralino(msN2sq),
            sigmadd_neutralino(msN2sq), sigmaud_neutralino(msN2sq),
            sigmauu_neutralino(msN3sq), sigmadd_neutralino(msN3sq),
            sigmaud_neutralino(msN3sq), sigmauu_neutralino(msN4sq),
            sigmadd_neutralino(msN4sq), sigmaud_neutralino(msN4sq),
            sigmauu_chargino1, sigmadd_chargino1, sigmaud_chargino1,
            sigmauu_chargino2, sigmadd_chargino2, sigmaud_chargino2,
            sigmauu_h0, sigmadd_h0, sigmaud_h0, sigmauu_heavy_h0,
            sigmadd_heavy_h0, sigmaud_heavy_h0, sigmauu_h_pm, sigmadd_h_pm,
            sigmaud_h_pm, sigmauu_w_pm, sigmadd_w_pm, sigmaud_w_pm,
            sigmauu_z0, sigmadd_z0, sigmaud_z0, sigmauu_top, sigmadd_top,
            sigmaud_top, sigmauu_bottom, sigmadd_bottom, sigmaud_bottom,
            sigmauu_tau, sigmadd_tau, sigmaud_tau, sigmauu_charm,
            sigmadd_charm, sigmaud_charm, sigmauu_strange, sigmadd_strange,
            sigmaud_strange, sigmauu_mu, sigmadd_mu, sigmaud_mu,
            sigmauu_up, sigmadd_up, sigmaud_up, sigmauu_down, sigmadd_down,
            sigmaud_down, sigmauu_elec, sigmadd_elec, sigmaud_elec,
            sigmauu_2loop(), sigmadd_2loop()]

def Delta_BG_calc(modselno, mymzsq, inputGUT_BCs):
    """
    Compute the fine-tuning measure Delta_BG for the selected model.

    Parameters
    ----------
    modselno : Int.
        Selected model number from model list.
    mymzsq : Float.
        Running mZ^2, evaluated at Q=2 TeV from original SLHA point.
    inputGUT_BCs : Array of floats.
        Original GUT-scale BCs from SLHA file.

    Returns
    -------
    Delta_BG : Float.
        Naturalness measure Delta_BG.

    """
    deriv_calc = 0
    if (modselno == 1):
        mym0 = inputGUT_BCs[27]
        hm0 = mym0 * 1e-4
        mymhf = inputGUT_BCs[3]
        hmhf = mymhf * 1e-4
        myA0 = inputGUT_BCs[16] / inputGUT_BCs[7]
        hA0 = myA0 * 1e-4
        mymu0 = inputGUT_BCs[6]
        hmu0 = mymu0 * 1e-4
        ##### Set up solutions for m_0 derivative #####
        # Boundary conditions first
        testBCs = inputGUT_BCs
        # Deviate m0 by small amount and square soft scalar masses for BCs
        testBCs[27] = np.power(inputGUT_BCs[27] + hm0, 2)

        deriv_array = np.array([0, 0, 0, 0])
        sens_params = np.sort(np.array([(np.abs((np.sqrt(mQ3sqGUT) / mymzsq)
                                                * deriv_calc), 'c_m_0'),
                                        (np.abs((M1GUT / mymzsq)
                                                * deriv_calc), 'c_m_1/2'),
                                        (np.abs(((atGUT / ytGUT) / mymzsq)
                                                * deriv_calc), 'c_A_0'),
                                        (np.abs((muGUT / mymzsq)
                                                * deriv_calc), 'c_mu')],
                                       dtype=[('BGContrib', float),
                                              ('BGlabel', 'U40')]),
                              order='BGContrib')
    elif (modselno == 2):
        sens_params = np.sort(np.array([(np.abs((np.sqrt(mQ3sqGUT) / mymzsq)
                                                * deriv_calc), 'c_m_0'),
                                        (np.abs((M1GUT / mymzsq)
                                                * deriv_calc), 'c_m_1/2'),
                                        (np.abs(((atGUT / ytGUT) / mymzsq)
                                                * deriv_calc), 'c_A_0'),
                                        (np.abs((muGUT / mymzsq)
                                                * deriv_calc), 'c_mu'),
                                        (np.abs((mHusqGUT / mymzsq)
                                                * deriv_calc), 'c_mHu^2')],
                                       dtype=[('BGContrib', float),
                                              ('BGlabel', 'U40')]),
                              order='BGContrib')
    elif (modselno == 3):
        sens_params = np.sort(np.array([(np.abs((np.sqrt(mQ3sqGUT) / mymzsq)
                                                * deriv_calc), 'c_m_0'),
                                        (np.abs((M1GUT / mymzsq)
                                                * deriv_calc), 'c_m_1/2'),
                                        (np.abs(((atGUT / ytGUT) / mymzsq)
                                                * deriv_calc), 'c_A_0'),
                                        (np.abs((muGUT / mymzsq)
                                                * deriv_calc), 'c_mu'),
                                        (np.abs((mHusqGUT / mymzsq)
                                                * deriv_calc), 'c_mHu^2'),
                                        (np.abs((mHdsqGUT / mymzsq)
                                                * deriv_calc), 'c_mHd^2')],
                                       dtype=[('BGContrib', float),
                                              ('BGlabel', 'U40')]),
                              order='BGContrib')
    elif (modselno == 4):
        sens_params = np.sort(np.array([(np.abs((np.sqrt(mQ3sqGUT) / mymzsq)
                                                * deriv_calc), 'c_m_0(3)'),
                                        (np.abs((np.sqrt(mQ2sqGUT) / mymzsq)
                                                * deriv_calc), 'c_m_0(1,2)'),
                                        (np.abs((M1GUT / mymzsq)
                                                * deriv_calc), 'c_m_1/2'),
                                        (np.abs(((atGUT / ytGUT) / mymzsq)
                                                * deriv_calc), 'c_A_0'),
                                        (np.abs((muGUT / mymzsq)
                                                * deriv_calc), 'c_mu'),
                                        (np.abs((mHusqGUT / mymzsq)
                                                * deriv_calc), 'c_mHu^2'),
                                        (np.abs((mHdsqGUT / mymzsq)
                                                * deriv_calc), 'c_mHd^2')],
                                       dtype=[('BGContrib', float),
                                              ('BGlabel', 'U40')]),
                              order='BGContrib')
    elif (modselno == 5):
        sens_params = np.sort(np.array([(np.abs((np.sqrt(mQ3sqGUT) / mymzsq)
                                                * deriv_calc), 'c_m_0(3)'),
                                        (np.abs((np.sqrt(mQ2sqGUT) / mymzsq)
                                                * deriv_calc), 'c_m_0(2)'),
                                        (np.abs((np.sqrt(mQ1sqGUT) / mymzsq)
                                                * deriv_calc), 'c_m_0(1)'),
                                        (np.abs((M1GUT / mymzsq)
                                                * deriv_calc), 'c_m_1/2'),
                                        (np.abs(((atGUT / ytGUT) / mymzsq)
                                                * deriv_calc), 'c_A_0'),
                                        (np.abs((muGUT / mymzsq)
                                                * deriv_calc), 'c_mu'),
                                        (np.abs((mHusqGUT / mymzsq)
                                                * deriv_calc), 'c_mHu^2'),
                                        (np.abs((mHdsqGUT / mymzsq)
                                                * deriv_calc), 'c_mHd^2')],
                                       dtype=[('BGContrib', float),
                                              ('BGlabel', 'U40')]),
                              order='BGContrib')
    elif (modselno == 6):
       sens_params = np.sort(np.array([(np.abs((np.sqrt(mQ3sqGUT) / mymzsq)
                                               * deriv_calc), 'c_m_Q3'),
                                       (np.abs((np.sqrt(mQ2sqGUT) / mymzsq)
                                               * deriv_calc), 'c_m_Q2'),
                                       (np.abs((np.sqrt(mQ1sqGUT) / mymzsq)
                                               * deriv_calc), 'c_m_Q1'),
                                       (np.abs((np.sqrt(mU3sqGUT) / mymzsq)
                                               * deriv_calc), 'c_m_U3'),
                                       (np.abs((np.sqrt(mU2sqGUT) / mymzsq)
                                               * deriv_calc), 'c_m_U2'),
                                       (np.abs((np.sqrt(mU1sqGUT) / mymzsq)
                                               * deriv_calc), 'c_m_U1'),
                                       (np.abs((np.sqrt(mD3sqGUT) / mymzsq)
                                               * deriv_calc), 'c_m_D3'),
                                       (np.abs((np.sqrt(mD2sqGUT) / mymzsq)
                                               * deriv_calc), 'c_m_D2'),
                                       (np.abs((np.sqrt(mD1sqGUT) / mymzsq)
                                               * deriv_calc), 'c_m_D1'),
                                       (np.abs((np.sqrt(mL3sqGUT) / mymzsq)
                                               * deriv_calc), 'c_m_L3'),
                                       (np.abs((np.sqrt(mL2sqGUT) / mymzsq)
                                               * deriv_calc), 'c_m_L2'),
                                       (np.abs((np.sqrt(mL1sqGUT) / mymzsq)
                                               * deriv_calc), 'c_m_L1'),
                                       (np.abs((np.sqrt(mE3sqGUT) / mymzsq)
                                               * deriv_calc), 'c_m_E3'),
                                       (np.abs((np.sqrt(mE2sqGUT) / mymzsq)
                                               * deriv_calc), 'c_m_E2'),
                                       (np.abs((np.sqrt(mE1sqGUT) / mymzsq)
                                               * deriv_calc), 'c_m_E1'),
                                       (np.abs((M1GUT / mymzsq)
                                               * deriv_calc), 'c_M_1'),
                                       (np.abs((M2GUT / mymzsq)
                                               * deriv_calc), 'c_M_2'),
                                       (np.abs((M3GUT / mymzsq)
                                               * deriv_calc), 'c_M_3'),
                                       (np.abs(((atGUT / ytGUT) / mymzsq)
                                               * deriv_calc), 'c_A_t'),
                                       (np.abs(((acGUT / ycGUT) / mymzsq)
                                               * deriv_calc), 'c_A_c'),
                                       (np.abs(((auGUT / yuGUT) / mymzsq)
                                               * deriv_calc), 'c_A_u'),
                                       (np.abs(((abGUT / ybGUT) / mymzsq)
                                               * deriv_calc), 'c_A_b'),
                                       (np.abs(((asGUT / ysGUT) / mymzsq)
                                               * deriv_calc), 'c_A_s'),
                                       (np.abs(((adGUT / ydGUT) / mymzsq)
                                               * deriv_calc), 'c_A_d'),
                                       (np.abs(((atauGUT / ytauGUT) / mymzsq)
                                               * deriv_calc), 'c_A_tau'),
                                       (np.abs(((amuGUT / ymuGUT) / mymzsq)
                                               * deriv_calc), 'c_A_mu'),
                                       (np.abs(((aeGUT / yeGUT) / mymzsq)
                                               * deriv_calc), 'c_A_e'),
                                       (np.abs((muGUT / mymzsq)
                                               * deriv_calc), 'c_mu'),
                                       (np.abs((mHusqGUT / mymzsq)
                                               * deriv_calc), 'c_mHu^2'),
                                       (np.abs((mHdsqGUT / mymzsq)
                                               * deriv_calc), 'c_mHd^2')],
                                      dtype=[('BGContrib', float),
                                             ('BGlabel', 'U40')]),
                             order='BGContrib')
    return sens_params

def Delta_HS_calc(mHdsq_Lambda, delta_mHdsq, mHusq_Lambda, delta_mHusq,
                  mu_Lambdasq, delta_musq, running_mz_sq, tanb_sq, sigmauutot,
                  sigmaddtot):
    """
    Compute the fine-tuning measure Delta_HS.

    Parameters
    ----------
    mHdsq_Lambda : Float.
        mHd^2(GUT).
    delta_mHdsq : Float.
        RGE running of mHd^2 down to 2 TeV.
    mHusq_Lambda : Float.
        mHu^2(GUT).
    delta_mHusq : Float.
        RGE running of mHu^2 down to 2 TeV.
    mu_Lambdasq : Float.
        mu^2(GUT).
    delta_musq : Float.
        RGE running of mu^2 down to 2 TeV.
    running_mz_sq : Float.
        Running mZ^2, evaluated at 2 TeV.
    tanb_sq : Float.
        tan^2(beta), evaluated at 2 TeV.
    sigmauutot : Float.
        Up-type radiative corrections evaluated at 2 TeV.
    sigmaddtot : Float.
        Down-type radiative corrections evaluated at 2 TeV.

    Returns
    -------
    Delta_HS : Float.
        Fine-tuning measure Delta_HS.

    """
    B_Hd = mHdsq_Lambda / (tanb_sq - 1)
    B_deltaHd = delta_mHdsq / (tanb_sq - 1)
    B_Hu = mHusq_Lambda * tanb_sq / (tanb_sq - 1)
    B_deltaHu = delta_mHusq * tanb_sq / (tanb_sq - 1)
    B_Sigmadd = sigmaddtot / (tanb_sq - 1)
    B_Sigmauu = sigmauutot * tanb_sq / (tanb_sq - 1)
    B_muLambdasq = mu_Lambdasq
    B_deltamusq = delta_musq
    Delta_HS = np.amax(np.array([np.abs(B_Hd), np.abs(B_deltaHd),
                                 np.abs(B_Hu), np.abs(B_deltaHu),
                                 np.abs(B_Sigmadd), np.abs(B_Sigmauu),
                                 np.abs(B_muLambdasq), np.abs(B_deltamusq)]))\
        / (running_mz_sq / 2)
    return Delta_HS

def Delta_EW_calc(myQ, vHiggs_wk, mu_wk, beta_wk, yt_wk, yc_wk, yu_wk, yb_wk,
                  ys_wk, yd_wk, ytau_wk, ymu_wk, ye_wk, g1_wk, g2_wk, g3_wk,
                  mQ3_sq_wk, mQ2_sq_wk, mQ1_sq_wk, mL3_sq_wk, mL2_sq_wk,
                  mL1_sq_wk, mU3_sq_wk, mU2_sq_wk, mU1_sq_wk, mD3_sq_wk,
                  mD2_sq_wk, mD1_sq_wk, mE3_sq_wk, mE2_sq_wk, mE1_sq_wk, M1_wk,
                  M2_wk, M3_wk, mHu_sq_wk, mHd_sq_wk, at_wk, ac_wk, au_wk,
                  ab_wk, as_wk, ad_wk, atau_wk, amu_wk, ae_wk):
    """
    Compute the fine-tuning measure Delta_EW.

    Parameters
    ----------
    myQ: Float.
        Renormalization scale for evaluation of radiative corrections.
    vHiggs_wk : Float.
        Weak-scale Higgs VEV.
    mu_wk : Float.
        Weak-scale Higgsino mass parameter mu.
    beta_wk : Float.
        Higgs mixing angle beta at the weak scale (from ratio of Higgs VEVs).
    yt_wk : Float.
        Weak-scale top Yukawa coupling.
    yc_wk : Float.
        Weak-scale charm Yukawa coupling.
    yu_wk : Float.
        Weak-scale up Yukawa coupling.
    yb_wk : Float.
        Weak-scale bottom Yukawa coupling.
    ys_wk : Float.
        Weak-scale strange Yukawa coupling.
    yd_wk : Float.
        Weak-scale down Yukawa coupling.
    ytau_wk : Float.
        Weak-scale tau Yukawa coupling.
    ymu_wk : Float.
        Weak-scale mu Yukawa coupling.
    ye_wk : Float.
        Weak-scale electron Yukawa coupling.
    g1_wk : Float.
        Weak-scale U(1) gauge coupling.
    g2_wk : Float.
        Weak-scale SU(2) gauge coupling.
    g3_wk : Float.
        Weak-scale SU(3) gauge coupling.
    mQ3_sq_wk : Float.
        Weak-scale 3rd gen left squark squared mass.
    mQ2_sq_wk : Float.
        Weak-scale 2nd gen left squark squared mass.
    mQ1_sq_wk : Float.
        Weak-scale 1st gen left squark squared mass.
    mL3_sq_wk : Float.
        Weak-scale 3rd gen left slepton squared mass.
    mL2_sq_wk : Float.
        Weak-scale 2nd gen left slepton squared mass.
    mL1_sq_wk : Float.
        Weak-scale 1st gen left slepton squared mass.
    mU3_sq_wk : Float.
        Weak-scale 3rd gen right up-type squark squared mass.
    mU2_sq_wk : Float.
        Weak-scale 2nd gen right up-type squark squared mass.
    mU1_sq_wk : Float.
        Weak-scale 1st gen right up-type squark squared mass.
    mD3_sq_wk : Float.
        Weak-scale 3rd gen right down-type squark squared mass.
    mD2_sq_wk : Float.
        Weak-scale 2nd gen right down-type squark squared mass.
    mD1_sq_wk : Float.
        Weak-scale 1st gen right down-type squark squared mass.
    mE3_sq_wk : Float.
        Weak-scale 3rd gen right slepton squared mass.
    mE2_sq_wk : Float.
        Weak-scale 2nd gen right slepton squared mass.
    mE1_sq_wk : Float.
        Weak-scale 1st gen right slepton squared mass.
    M1_wk : Float.
        Weak-scale bino mass parameter.
    M2_wk : Float.
        Weak-scale wino mass parameter.
    M3_wk : Float.
        Weak-scale gluino mass parameter.
    mHu_sq_wk : Float.
        Weak-scale up-type soft Higgs mass parameter.
    mHd_sq_wk : Float.
        Weak-scale down-type soft Higgs mass parameter.
    at_wk : Float.
        Weak-scale reduced top soft trilinear coupling.
    ac_wk : Float.
        Weak-scale reduced charm soft trilinear coupling.
    au_wk : Float.
        Weak-scale reduced up soft trilinear coupling.
    ab_wk : Float.
        Weak-scale reduced bottom soft trilinear coupling.
    as_wk : Float.
        Weak-scale reduced strange soft trilinear coupling.
    ad_wk : Float.
        Weak-scale reduced down soft trilinear coupling.
    atau_wk : Float.
        Weak-scale reduced tau soft trilinear coupling.
    amu_wk : Float.
        Weak-scale reduced mu soft trilinear coupling.
    ae_wk : Float.
        Weak-scale reduced electron soft trilinear coupling.
    Returns
    -------
    Delta_EW_contribs : Array of floats.
        Individual contributions to Delta_EW. Return an ordered-by-magnitude
        list of 42 Sigma_u^u corrections and 42 Sigma_d^d corrections to the
        Higgs minimization condition.
    """
    # First evaluate radiative corrections

    myradcorrs = my_radcorr_calc(myQ, vHiggs_wk, mu_wk, beta_wk, yt_wk, yc_wk,
                                 yu_wk, yb_wk, ys_wk, yd_wk, ytau_wk, ymu_wk,
                                 ye_wk, g1_wk, g2_wk, g3_wk, mQ3_sq_wk,
                                 mQ2_sq_wk, mQ1_sq_wk, mL3_sq_wk, mL2_sq_wk,
                                 mL1_sq_wk, mU3_sq_wk, mU2_sq_wk, mU1_sq_wk,
                                 mD3_sq_wk, mD2_sq_wk, mD1_sq_wk, mE3_sq_wk,
                                 mE2_sq_wk, mE1_sq_wk, M1_wk, M2_wk, M3_wk,
                                 mHu_sq_wk, mHd_sq_wk, at_wk, ac_wk, au_wk,
                                 ab_wk, as_wk, ad_wk, atau_wk, amu_wk, ae_wk)

    # DEW contribution computation: #

    def dew_funcu(inp):
        """
        Compute individual one-loop DEW contributions from Sigma_u^u.

        Parameters
        ----------
        inp : One-loop correction or Higgs to be inputted into the DEW eval.

        """
        mycontribuu = ((-1) * inp * (np.power(np.tan(beta_wk), 2)))\
            / ((np.power(np.tan(beta_wk), 2)) - 1)
        return mycontribuu

    def dew_funcd(inp):
        """
        Compute individual one-loop DEW contributions from Sigma_d^d.

        Parameters
        ----------
        inp : One-loop correction or Higgs to be inputted into the DEW eval.

        """
        mycontribdd = inp / ((np.power(np.tan(beta_wk), 2)) - 1)
        return mycontribdd
    
    running_mZ_sq = np.power(vHiggs_wk, 2) * ((np.power(g2_wk, 2)
                                               + ((3 / 5)
                                                  * np.power(g1_wk, 2))) / 2)
    cmu = (-1) * np.power(mu_wk, 2)
    cHu = dew_funcu(mHu_sq_wk)
    cHd = dew_funcd(mHd_sq_wk)
    contribs = np.array([cmu, cHu, cHd, dew_funcu(myradcorrs[3]), # 0: C_mu, 1: C_Hu, 2: C_Hd, 3: C_sigmauu_stop_1
                         dew_funcd(myradcorrs[4]), dew_funcu(myradcorrs[6]), # 4: C_sigmadd_stop_1, 5: C_sigmauu_stop_2
                         dew_funcd(myradcorrs[7]), dew_funcu(myradcorrs[9]), # 6: C_sigmadd_stop_2, 7: C_sigmauu_sbot_1
                         dew_funcd(myradcorrs[10]), dew_funcu(myradcorrs[12]), # 8: C_sigmadd_sbot_1, 9: C_sigmauu_sbot_2
                         dew_funcd(myradcorrs[13]), dew_funcu(myradcorrs[15]), # 10: C_sigmadd_sbot_2, 11: C_sigmauu_stau_1
                         dew_funcd(myradcorrs[16]), dew_funcu(myradcorrs[18]), # 12: C_sigmadd_stau_1, 13: C_sigmauu_stau_2
                         dew_funcd(myradcorrs[19]), dew_funcu(myradcorrs[21]), # 14: C_sigmadd_stau_2, 15: C_sigmauu_stau_sneut
                         dew_funcd(myradcorrs[22]), dew_funcu(myradcorrs[24]), # 16: C_sigmadd_stau_sneut, 17: C_sigmauu_scharm_1
                         dew_funcd(myradcorrs[25]), dew_funcu(myradcorrs[27]), # 18: C_sigmadd_scharm_1, 19: C_sigmauu_scharm_2
                         dew_funcd(myradcorrs[28]), dew_funcu(myradcorrs[30]), # 20: C_sigmadd_scharm_2, 21: C_sigmauu_sstrange_1
                         dew_funcd(myradcorrs[31]), dew_funcu(myradcorrs[33]), # 22: C_sigmadd_sstrange_1, 23: C_sigmauu_sstrange_2
                         dew_funcd(myradcorrs[34]), dew_funcu(myradcorrs[36]), # 24: C_sigmadd_sstrange_2, 25: C_sigmauu_smu_1
                         dew_funcd(myradcorrs[37]), dew_funcu(myradcorrs[39]), # 26: C_sigmadd_smu_1, 27: C_sigmauu_smu_2
                         dew_funcd(myradcorrs[40]), dew_funcu(myradcorrs[42]), # 28: C_sigmadd_smu_2, 29: C_sigmauu_smu_sneut
                         dew_funcd(myradcorrs[43]), dew_funcu(myradcorrs[45]), # 30: C_sigmadd_smu_sneut, 31: C_sigmauu_sup_1
                         dew_funcd(myradcorrs[46]), dew_funcu(myradcorrs[48]), # 32: C_sigmadd_sup_1, 33: C_sigmauu_sup_2
                         dew_funcd(myradcorrs[49]), dew_funcu(myradcorrs[51]), # 34: C_sigmadd_sup_2, 35: C_sigmauu_sdown_1
                         dew_funcd(myradcorrs[52]), dew_funcu(myradcorrs[54]), # 36: C_sigmadd_sdown_1, 37: C_sigmauu_sdown_2
                         dew_funcd(myradcorrs[55]), dew_funcu(myradcorrs[57]), # 38: C_sigmadd_sdown_2, 39: C_sigmauu_se_1
                         dew_funcd(myradcorrs[58]), dew_funcu(myradcorrs[60]), # 40: C_sigmadd_se_1, 41: C_sigmauu_se_2
                         dew_funcd(myradcorrs[61]), dew_funcu(myradcorrs[63]), # 42: C_sigmadd_se_2, 43: C_sigmauu_selec_sneut
                         dew_funcd(myradcorrs[64]), dew_funcu(myradcorrs[66]), # 44: C_sigmadd_selec_sneut, 45: C_sigmauu_neutralino1
                         dew_funcd(myradcorrs[67]), dew_funcu(myradcorrs[69]), # 46: C_sigmadd_neutralino1, 47: C_sigmauu_neutralino2
                         dew_funcd(myradcorrs[70]), dew_funcu(myradcorrs[72]), # 48: C_sigmadd_neutralino2, 49: C_sigmauu_neutralino3
                         dew_funcd(myradcorrs[73]), dew_funcu(myradcorrs[75]), # 50: C_sigmadd_neutralino3, 51: C_sigmauu_neutralino4
                         dew_funcd(myradcorrs[76]), dew_funcu(myradcorrs[78]), # 52: C_sigmadd_neutralino4, 53: C_sigmauu_chargino1
                         dew_funcd(myradcorrs[79]), dew_funcu(myradcorrs[81]), # 54: C_sigmadd_chargino1, 55: C_sigmauu_chargino2
                         dew_funcd(myradcorrs[82]), dew_funcu(myradcorrs[84]), # 56: C_sigmadd_chargino2, 57: C_sigmauu_h0
                         dew_funcd(myradcorrs[85]), dew_funcu(myradcorrs[87]), # 58: C_sigmadd_h0, 59: C_sigmauu_heavy_h0
                         dew_funcd(myradcorrs[88]), dew_funcu(myradcorrs[90]), # 60: C_sigmadd_heavy_h0, 61: C_sigmauu_h_pm
                         dew_funcd(myradcorrs[91]), dew_funcu(myradcorrs[93]), # 62: C_sigmadd_h_pm, 63: C_sigmauu_w_pm
                         dew_funcd(myradcorrs[94]), dew_funcu(myradcorrs[96]), # 64: C_sigmadd_w_pm, 65: C_sigmauu_z0
                         dew_funcd(myradcorrs[97]), # 66: C_sigmadd_Z0
                         dew_funcu(myradcorrs[99])
                         + dew_funcu(myradcorrs[102])
                         + dew_funcu(myradcorrs[105])
                         + dew_funcu(myradcorrs[108])
                         + dew_funcu(myradcorrs[111])
                         + dew_funcu(myradcorrs[114])
                         + dew_funcu(myradcorrs[117])
                         + dew_funcu(myradcorrs[120])
                         + dew_funcu(myradcorrs[123]), # 67: C_sigmauu_SM
                         dew_funcd(myradcorrs[100])
                         + dew_funcd(myradcorrs[103])
                         + dew_funcd(myradcorrs[106])
                         + dew_funcd(myradcorrs[109])
                         + dew_funcd(myradcorrs[112])
                         + dew_funcd(myradcorrs[115])
                         + dew_funcd(myradcorrs[118])
                         + dew_funcd(myradcorrs[121])
                         + dew_funcd(myradcorrs[124]), # 68: C_sigmadd_SM
                         dew_funcu(myradcorrs[126]), # 69: C_sigmauu_2loop
                         dew_funcd(myradcorrs[127])] # 70: C_sigmadd_2loop
                        ) / (running_mZ_sq / 2)
    #print(contribs)
    label_sort_array = np.sort(np.array([(contribs[0], np.abs(contribs[0]),
                                          'mu'),
                                         (contribs[1], np.abs(contribs[1]),
                                          'H_u'),
                                         (contribs[2], np.abs(contribs[2]),
                                          'H_d'),
                                         (contribs[3], np.abs(contribs[3]),
                                          'Sigma_u^u(stop_1)'),
                                         (contribs[4], np.abs(contribs[4]),
                                          'Sigma_d^d(stop_1)'),
                                         (contribs[5], np.abs(contribs[5]),
                                          'Sigma_u^u(stop_2)'),
                                         (contribs[6], np.abs(contribs[6]),
                                          'Sigma_d^d(stop_2)'),
                                         (contribs[7], np.abs(contribs[7]),
                                          'Sigma_u^u(sbot_1)'),
                                         (contribs[8], np.abs(contribs[8]),
                                          'Sigma_d^d(sbot_1)'),
                                         (contribs[9], np.abs(contribs[9]),
                                          'Sigma_u^u(sbot_2)'),
                                         (contribs[10], np.abs(contribs[10]),
                                          'Sigma_d^d(sbot_2)'),
                                         (contribs[11], np.abs(contribs[11]),
                                          'Sigma_u^u(stau_1)'),
                                         (contribs[12], np.abs(contribs[12]),
                                          'Sigma_d^d(stau_1)'),
                                         (contribs[13], np.abs(contribs[13]),
                                          'Sigma_u^u(stau_2)'),
                                         (contribs[14], np.abs(contribs[14]),
                                          'Sigma_d^d(stau_2)'),
                                         (contribs[15], np.abs(contribs[15]),
                                          'Sigma_u^u(stau sneutrino)'),
                                         (contribs[16], np.abs(contribs[16]),
                                          'Sigma_d^d(stau sneutrino)'),
                                         (contribs[17] + contribs[19]
                                          + contribs[21] + contribs[23],
                                          np.abs(contribs[17] + contribs[19]
                                                 + contribs[21]
                                                 + contribs[23]),
                                          'Sigma_u^u(sum 2nd gen. squarks)'),
                                         (contribs[18] + contribs[20]
                                          + contribs[22] + contribs[24],
                                          np.abs(contribs[18] + contribs[20]
                                                 + contribs[22]
                                                 + contribs[24]),
                                          'Sigma_d^d(sum 2nd gen. squarks)'),
                                         (contribs[25], np.abs(contribs[25]),
                                          'Sigma_u^u(smuon_1)'),
                                         (contribs[26], np.abs(contribs[26]),
                                          'Sigma_d^d(smuon_1)'),
                                         (contribs[27], np.abs(contribs[27]),
                                          'Sigma_u^u(smuon_2)'),
                                         (contribs[28], np.abs(contribs[28]),
                                          'Sigma_d^d(smuon_2)'),
                                         (contribs[29], np.abs(contribs[29]),
                                          'Sigma_u^u(smuon sneutrino)'),
                                         (contribs[30], np.abs(contribs[30]),
                                          'Sigma_d^d(smuon sneutrino)'),
                                         (contribs[31] + contribs[33]
                                          + contribs[35] + contribs[37],
                                          np.abs(contribs[31] + contribs[33]
                                                 + contribs[35]
                                                 + contribs[37]),
                                          'Sigma_u^u(sum 1st gen. squarks)'),
                                         (contribs[32] + contribs[34]
                                          + contribs[36] + contribs[38],
                                          np.abs(contribs[32] + contribs[34]
                                                 + contribs[36]
                                                 + contribs[38]),
                                          'Sigma_d^d(sum 1st gen. squarks)'),
                                         (contribs[39], np.abs(contribs[39]),
                                          'Sigma_u^u(selectron_1)'),
                                         (contribs[40], np.abs(contribs[40]),
                                          'Sigma_d^d(selectron_1)'),
                                         (contribs[41], np.abs(contribs[41]),
                                          'Sigma_u^u(selectron_2)'),
                                         (contribs[42], np.abs(contribs[42]),
                                          'Sigma_d^d(selectron_2)'),
                                         (contribs[43], np.abs(contribs[43]),
                                          'Sigma_u^u(selectron sneutrino)'),
                                         (contribs[44], np.abs(contribs[44]),
                                          'Sigma_d^d(selectron sneutrino)'),
                                         (contribs[45], np.abs(contribs[45]),
                                          'Sigma_u^u(neutralino_1)'),
                                         (contribs[46], np.abs(contribs[46]),
                                          'Sigma_d^d(neutralino_1)'),
                                         (contribs[47], np.abs(contribs[47]),
                                          'Sigma_u^u(neutralino_2)'),
                                         (contribs[48], np.abs(contribs[48]),
                                          'Sigma_d^d(neutralino_2)'),
                                         (contribs[49], np.abs(contribs[49]),
                                          'Sigma_u^u(neutralino_3)'),
                                         (contribs[50], np.abs(contribs[50]),
                                          'Sigma_d^d(neutralino_3)'),
                                         (contribs[51], np.abs(contribs[51]),
                                          'Sigma_u^u(neutralino_4)'),
                                         (contribs[52], np.abs(contribs[52]),
                                          'Sigma_d^d(neutralino_4)'),
                                         (contribs[53], np.abs(contribs[53]),
                                          'Sigma_u^u(chargino_1)'),
                                         (contribs[54], np.abs(contribs[54]),
                                          'Sigma_d^d(chargino_1)'),
                                         (contribs[55], np.abs(contribs[55]),
                                          'Sigma_u^u(chargino_2)'),
                                         (contribs[56], np.abs(contribs[56]),
                                          'Sigma_d^d(chargino_2)'),
                                         (contribs[57], np.abs(contribs[57]),
                                          'Sigma_u^u(h_0)'),
                                         (contribs[58], np.abs(contribs[58]),
                                          'Sigma_d^d(h_0)'),
                                         (contribs[59], np.abs(contribs[59]),
                                          'Sigma_u^u(H_0)'),
                                         (contribs[60], np.abs(contribs[60]),
                                          'Sigma_d^d(H_0)'),
                                         (contribs[61], np.abs(contribs[61]),
                                          'Sigma_u^u(H_+-)'),
                                         (contribs[62], np.abs(contribs[62]),
                                          'Sigma_d^d(H_+-)'),
                                         (contribs[63], np.abs(contribs[63]),
                                          'Sigma_u^u(W_+-)'),
                                         (contribs[64], np.abs(contribs[64]),
                                          'Sigma_d^d(W_+-)'),
                                         (contribs[65], np.abs(contribs[65]),
                                          'Sigma_u^u(Z_0)'),
                                         (contribs[66], np.abs(contribs[66]),
                                          'Sigma_d^d(Z_0)'),
                                         (contribs[67], np.abs(contribs[67]),
                                          'Sigma_u^u(SM fermions)'),
                                         (contribs[68], np.abs(contribs[68]),
                                          'Sigma_d^d(SM fermions)'),
                                         (contribs[69], np.abs(contribs[69]),
                                          'Sigma_u^u(O(alpha_s alpha_t))'),
                                         (contribs[70], np.abs(contribs[70]),
                                          'Sigma_d^d(O(alpha_s alpha_t))')],
                                        dtype=[('Contrib', float),
                                               ('AbsContrib', float),
                                               ('label', 'U40')]),
                               order='AbsContrib')
    reverse_sort_array = label_sort_array[::-1]
    return reverse_sort_array

if __name__ == "__main__":
    ##### Constants #####

    loop_fac = 1 / (16 * np.power(np.pi, 2))
    loop_fac_sq = np.power(loop_fac, 2)
    b_1l = [33/5, 1, -3]
    b_2l = [[199/25, 27/5, 88/5], [9/5, 25, 24], [11/5, 9, 14]]
    c_2l = [[26/5, 14/5, 18/5], [6, 6, 2], [4, 4, 0]]
    userContinue = True
    print("Welcome to DEW4SLHA, a program for computing the naturalness")
    print("measures Delta_EW, Delta_BG, and Delta_HS in the MSSM")
    print("from a SUSY Les Houches Accord (SLHA) file.")
    print("")
    print("To use this program, you may select a")
    print("MSSM SLHA file from your choice of spectrum generator (e.g.,")
    print("SoftSUSY, Isajet, SPheno, FlexibleSUSY, etc.). ")
    print("If multiple renormalization scales are present in the SLHA file,")
    print("then the first renormalization scale present in the SLHA file,")
    print("from top to bottom, is read in.")
    print("")
    print("If the read-in scale is not equal to 2 TeV, then full two-loop")
    print("RGEs, partial two-loop radiative corrections, and tree-level mass")
    print("relations are used to evolve your SLHA results to Q=2 TeV.")
    print("This is the scale at which Delta_EW, Delta_BG, and Delta_HS")
    print("will be evaluated.")
    print("")
    print("Supported models for the local RGE solver are MSSM models.")
    while userContinue:
        print("")
        print("For Delta_BG, the ``fundamental parameters'' vary from model")
        print("to model. For this reason, prior to entering the directory of")
        print("your SLHA file, please enter the model number below")
        print("corresponding to your SLHA file.")
        print("")
        print("Model numbers: ")
        print("1: CMSSM/mSUGRA")
        print("2: NUHM1")
        print("3: NUHM2")
        print("4: NUHM3")
        print("5: NUHM4")
        print("6: pMSSM-30 (pMSSM-19 + 6 diagonal, real "
              + "2nd & 1st gen. soft trilinears + non-universal 1st & 2nd gen."
              + "squark and slepton masses)")
        print("")
        modelCheck = True
        while modelCheck:
            try:
                modinp = int(input("From the list above, input the number of"
                                   + " the model your SLHA file corresponds"
                                   + " to: "))
                if (modinp not in [1, 2, 3, 4, 5, 6]):
                    print("Invalid model number selected, please try again.")
                    print("")
                    print("Model numbers: ")
                    print("1: CMSSM/mSUGRA")
                    print("2: NUHM1")
                    print("3: NUHM2")
                    print("4: NUHM3")
                    print("5: NUHM4")
                    print("6: pMSSM-25 (pMSSM-19 + 6 diagonal, real "
                          + "2nd & 1st gen. soft trilinears)")
                    print("")
                    modelCheck = True
                else:
                    modelCheck = False
            except(ValueError):
                print("Invalid model number selected, please try again.")
                print("")
                print("Model numbers: ")
                print("1: CMSSM/mSUGRA")
                print("2: NUHM1")
                print("3: NUHM2")
                print("4: NUHM3")
                print("5: NUHM4")
                print("6: pMSSM-25 (pMSSM-19 + 6 diagonal, real "
                      + "2nd & 1st gen. soft trilinears)")
                print("")
                modelCheck = True
        # SLHA input and definition of variables from SLHA file: #
        fileCheck = True
        while fileCheck:
            try:
                direc = input('Enter the full directory for your SLHA file: ')
                d = pyslha.read(direc)
                fileCheck = False
            except (FileNotFoundError):
                print("The input file cannot be found.\n")
                print("Please try checking your spelling and try again.\n")
                fileCheck = True
            except (IsADirectoryError):
                print("You have input a directory, not an SLHA file.\n")
                print("Please try again.\n")
        # Set up parameters for computations
        mZ = 91.1876 # This is the value in our universe, not for multiverse.
        [vHiggs, muQ, tanb, y_t] = [d.blocks['HMIX'][3] / np.sqrt(2),
                                    d.blocks['HMIX'][1], d.blocks['HMIX'][2],
                                    d.blocks['YU'][3, 3]]
        beta = np.arctan(tanb)
        [y_b, y_tau, g_2] = [d.blocks['YD'][3, 3], d.blocks['YE'][3, 3],
                              d.blocks['GAUGE'][2]]
        # See if SLHA has trilinears (a_i) in reduced form or not (A_i),
        # where a_i = y_i * A_i
        try:
            [a_t, a_b, a_tau] = [d.blocks['TU'][3, 3], d.blocks['TD'][3, 3],
                                 d.blocks['TE'][3, 3]]
        except(KeyError):
            [a_t, a_b, a_tau] = [d.blocks['AU'][3, 3] * y_t,
                                 d.blocks['AD'][3, 3] * y_b,
                                 d.blocks['AE'][3, 3] * y_tau]
        # Try to read in first two generations of Yukawas and soft trilinears
        # if present; if not, set approximate values based on 3rd gens,
        # coming from test BM points with universal trilinears at M_GUT.
        try:
            [y_c, y_u, y_s, y_d, y_mu, y_e] = [d.blocks['YU'][2, 2],
                                               d.blocks['YU'][1, 1],
                                               d.blocks['YD'][2, 2],
                                               d.blocks['YD'][1, 1],
                                               d.blocks['YE'][2, 2],
                                               d.blocks['YE'][1, 1]]
            [a_c, a_u, a_s, a_d, a_mu, a_e] = [d.blocks['TU'][2, 2],
                                               d.blocks['TU'][1, 1],
                                               d.blocks['TD'][2, 2],
                                               d.blocks['TD'][1, 1],
                                               d.blocks['TE'][2, 2],
                                               d.blocks['TE'][1, 1]]
        except(KeyError):
            try:
                [y_c, y_u, y_s, y_d, y_mu, y_e] = [d.blocks['YU'][2, 2],
                                                   d.blocks['YU'][1, 1],
                                                   d.blocks['YD'][2, 2],
                                                   d.blocks['YD'][1, 1],
                                                   d.blocks['YE'][2, 2],
                                                   d.blocks['YE'][1, 1]]
                [a_c, a_u, a_s, a_d, a_mu, a_e] = [d.blocks['AU'][2, 2] * y_c,
                                                   d.blocks['AU'][1, 1] * y_u,
                                                   d.blocks['AD'][2, 2] * y_s,
                                                   d.blocks['AD'][1, 1] * y_d,
                                                   d.blocks['AE'][2, 2] * y_mu,
                                                   d.blocks['AE'][1, 1] * y_e]
            except(KeyError):
                [y_c, y_u, y_s,
                 y_d, y_mu, y_e] = [0.0038707144115263633 * y_t,
                                    8.503914098792997e-06 * y_t,
                                    0.022511589541098956 * y_b,
                                    0.001028166075832906 * y_b,
                                    0.05797686773752101 * y_tau,
                                    0.0002803881828123263 * y_tau]
                [a_c, a_u, a_s,
                 a_d, a_mu, a_e] = [0.003058189551567951 * a_t,
                                    6.718559349422718e-06 * a_t,
                                    0.02075028366005483 * a_b,
                                    0.0009477258899500176 * a_b,
                                    0.07656170457694475 * a_tau,
                                    0.0003705675352847817 * a_tau]
        g_pr = d.blocks['GAUGE'][1]
        g_s = d.blocks['GAUGE'][3]
        # Different blocks exist for soft masses in different SLHA-producing
        # programs. Try to read in masses from the two most popular.
        try:
            [mQ3sq, mU3sq] = [d.blocks['MSQ2'][3, 3], d.blocks['MSU2'][3, 3]]
            [mD3sq, mL3sq, mE3sq] = [d.blocks['MSD2'][3, 3],
                                     d.blocks['MSL2'][3, 3],
                                     d.blocks['MSE2'][3, 3]]
            [mQ2sq, mU2sq] = [d.blocks['MSQ2'][2, 2], d.blocks['MSU2'][2, 2]]
            [mD2sq, mL2sq, mE2sq] = [d.blocks['MSD2'][2, 2],
                                     d.blocks['MSL2'][2, 2],
                                     d.blocks['MSE2'][2, 2]]
            [mQ1sq, mU1sq] = [d.blocks['MSQ2'][1, 1], d.blocks['MSU2'][1, 1]]
            [mD1sq, mL1sq, mE1sq] = [d.blocks['MSD2'][1, 1],
                                     d.blocks['MSL2'][1, 1],
                                     d.blocks['MSE2'][1, 1]]
        except(KeyError):
            [mQ3sq, mU3sq] = np.power([d.blocks['MSOFT'][43],
                                       d.blocks['MSOFT'][46]],2)
            [mD3sq, mL3sq, mE3sq] = np.power([d.blocks['MSOFT'][49],
                                              d.blocks['MSOFT'][33],
                                              d.blocks['MSOFT'][36]],2)
            [mQ2sq, mU2sq] = np.power([d.blocks['MSOFT'][42],
                                       d.blocks['MSOFT'][45]],2)
            [mD2sq, mL2sq, mE2sq] = np.power([d.blocks['MSOFT'][48],
                                              d.blocks['MSOFT'][32],
                                              d.blocks['MSOFT'][35]],2)
            [mQ1sq, mU1sq] = np.power([d.blocks['MSOFT'][41],
                                       d.blocks['MSOFT'][44]],2)
            [mD1sq, mL1sq, mE1sq] = np.power([d.blocks['MSOFT'][47],
                                              d.blocks['MSOFT'][31],
                                              d.blocks['MSOFT'][34]],2)
        my_M3 = d.blocks['MSOFT'][3]
        my_M2 = d.blocks['MSOFT'][2]
        my_M1 = d.blocks['MSOFT'][1]
        [mHusq, mHdsq] = [d.blocks['MSOFT'][22], d.blocks['MSOFT'][21]]
        # Read in SLHA scale from submitted file at which previous variables
        # are evaluated. 
        SLHA_scale = float(str(d.blocks['GAUGE'])
                           [str(d.blocks['GAUGE']).find('Q=')
                            +2:str(d.blocks['GAUGE']).find(')')])
        # Use 2-loop MSSM RGEs to evolve results to a renormalization scale
        # of Q = 2 TeV if the submitted SLHA file is not currently at that
        # scale. This is so evaluations of the naturalness measures are always
        # performed at a common scale of 2 TeV. 
        #######################################################################
        # The result is then run to a high scale of 10^17 GeV, and an
        # approximate GUT scale is chosen at the value where g1(Q) is closest
        # to g2(Q) over the scanned renormalization scales.
        #######################################################################
        # This running to the GUT scale is used in the evaluations of Delta_HS
        # and Delta_BG.
        radcorrs_from_SLHA = my_radcorr_calc(SLHA_scale, vHiggs, muQ, beta,
                                             y_t, y_c, y_u, y_b, y_s, y_d,
                                             y_tau, y_mu, y_e,
                                             np.sqrt(5 / 3) * g_pr, g_2, g_s,
                                             mQ3sq, mQ2sq, mQ1sq, mL3sq, mL2sq,
                                             mL1sq, mU3sq, mU2sq, mU1sq, mD3sq,
                                             mD2sq, mD1sq, mE3sq, mE2sq, mE1sq,
                                             my_M1, my_M2, my_M3, mHusq, mHdsq,
                                             a_t, a_c, a_u, a_b, a_s, a_d,
                                             a_tau, a_mu, a_e)
        # Compute partial 2-loop soft Higgs bilinear parameter b=B*mu at scale
        # in SLHA for RGE boundary condition.
        b_from_SLHA = ((mHusq + mHdsq + (2 * np.power(muQ, 2))
                        + radcorrs_from_SLHA[0] + radcorrs_from_SLHA[1])
                       * (np.sin(beta) * np.cos(beta))) + radcorrs_from_SLHA[2]
        mySLHABCs = [np.sqrt(5 / 3) * g_pr, g_2, g_s, my_M1, my_M2, my_M3,
                     muQ, y_t, y_c, y_u, y_b, y_s, y_d, y_tau, y_mu, y_e,
                     a_t, a_c, a_u, a_b, a_s, a_d, a_tau, a_mu, a_e,
                     mHusq, mHdsq, mQ1sq, mQ2sq, mQ3sq, mL1sq, mL2sq,
                     mL3sq, mU1sq, mU2sq, mU3sq, mD1sq, mD2sq, mD3sq,
                     mE1sq, mE2sq, mE3sq, b_from_SLHA, tanb]
        RGE_sols = my_RGE_solver(mySLHABCs, SLHA_scale, 2000.0)
        # Read in several parameters at weak and GUT scales
        myQGUT = RGE_sols[0]
        muQ = RGE_sols[8]
        muQ_GUT = RGE_sols[51]
        g1Q_GUT = RGE_sols[45]
        g2Q_GUT = RGE_sols[46]
        g3Q_GUT = RGE_sols[47]
        M1Q_GUT = RGE_sols[48]
        M2Q_GUT = RGE_sols[49]
        M3Q_GUT = RGE_sols[50]
        beta = np.arctan(RGE_sols[88])
        betaGUT = np.arctan(RGE_sols[89])
        tanb = RGE_sols[88]
        tanbQ_GUT = RGE_sols[89]
        y_t = RGE_sols[9]
        y_tQ_GUT = RGE_sols[52]
        y_c = RGE_sols[10]
        y_cQ_GUT = RGE_sols[53]
        y_u = RGE_sols[11]
        y_uQ_GUT = RGE_sols[54]
        y_b = RGE_sols[12]
        y_bQ_GUT = RGE_sols[55]
        y_s = RGE_sols[13]
        y_sQ_GUT = RGE_sols[56]
        y_d = RGE_sols[14]
        y_dQ_GUT = RGE_sols[57]
        y_tau = RGE_sols[15]
        y_tauQ_GUT = RGE_sols[58]
        y_mu = RGE_sols[16]
        y_muQ_GUT = RGE_sols[59]
        y_e = RGE_sols[17]
        y_eQ_GUT = RGE_sols[60]
        a_t = RGE_sols[18]
        a_tQ_GUT = RGE_sols[61]
        a_c = RGE_sols[19]
        a_cQ_GUT = RGE_sols[62]
        a_u = RGE_sols[20]
        a_uQ_GUT = RGE_sols[63]
        a_b = RGE_sols[21]
        a_bQ_GUT = RGE_sols[64]
        a_s = RGE_sols[22]
        a_sQ_GUT = RGE_sols[65]
        a_d = RGE_sols[23]
        a_dQ_GUT = RGE_sols[66]
        a_tau = RGE_sols[24]
        a_tauQ_GUT = RGE_sols[67]
        a_mu = RGE_sols[25]
        a_muQ_GUT = RGE_sols[68]
        a_e = RGE_sols[26]
        a_eQ_GUT = RGE_sols[69]
        mHusq = RGE_sols[27]
        mHusqQ_GUT = RGE_sols[70]
        mHdsq = RGE_sols[28]
        mHdsqQ_GUT = RGE_sols[71]
        mQ1sq = RGE_sols[29]
        m_uLQ_GUT = np.sqrt(RGE_sols[72])
        mQ2sq = RGE_sols[30]
        m_cLQ_GUT = np.sqrt(RGE_sols[73])
        mQ3sq = RGE_sols[31]
        m_tLQ_GUT = np.sqrt(RGE_sols[74])
        mL1sq = RGE_sols[32]
        m_eLQ_GUT = np.sqrt(RGE_sols[75])
        mL2sq = RGE_sols[33]
        m_muLQ_GUT = np.sqrt(RGE_sols[76])
        mL3sq = RGE_sols[34]
        m_tauLQ_GUT = np.sqrt(RGE_sols[77])
        mU1sq = RGE_sols[35]
        m_uRQ_GUT = np.sqrt(RGE_sols[78])
        mU2sq = RGE_sols[36]
        m_cRQ_GUT = np.sqrt(RGE_sols[79])
        mU3sq = RGE_sols[37]
        m_tRQ_GUT = np.sqrt(RGE_sols[80])
        mD1sq = RGE_sols[38]
        m_dRQ_GUT = np.sqrt(RGE_sols[81])
        mD2sq = RGE_sols[39]
        m_sRQ_GUT = np.sqrt(RGE_sols[82])
        mD3sq = RGE_sols[40]
        m_bRQ_GUT = np.sqrt(RGE_sols[83])
        mE1sq = RGE_sols[41]
        m_eRQ_GUT = np.sqrt(RGE_sols[84])
        mE2sq = RGE_sols[42]
        m_muRQ_GUT = np.sqrt(RGE_sols[85])
        mE3sq = RGE_sols[43]
        m_tauRQ_GUT = np.sqrt(RGE_sols[86])
        my_b_weak = RGE_sols[44]
        bQ_GUT = RGE_sols[87]
        tree_mzsq = (2 * (mHdsq - (mHusq * np.power(tanb, 2)))
                     / (np.power(tanb, 2) - 1)) - (2 * np.power(muQ, 2))
        Q_GUT_BCs = [g1Q_GUT, g2Q_GUT, g3Q_GUT, M1Q_GUT, M2Q_GUT, M3Q_GUT,
                     muQ_GUT, y_tQ_GUT, y_cQ_GUT, y_uQ_GUT, y_bQ_GUT, y_sQ_GUT,
                     y_dQ_GUT, y_tauQ_GUT, y_muQ_GUT, y_eQ_GUT, a_tQ_GUT,
                     a_cQ_GUT, a_uQ_GUT, a_bQ_GUT, a_sQ_GUT, a_dQ_GUT,
                     a_tauQ_GUT, a_muQ_GUT, a_eQ_GUT, mHusqQ_GUT, mHdsqQ_GUT,
                     m_uLQ_GUT, mcLQ_GUT, m_tLQ_GUT, m_eLQ_GUT, m_muLQ_GUT,
                     m_tauLQ_GUT, m_uRQ_GUT, m_cRQ_GUT, m_tRQ_GUT, m_dRQ_GUT,
                     m_sRQ_GUT, m_bRQ_GUT, m_eRQ_GUT, m_muRQ_GUT, m_tauRQ_GUT,
                     bQ_GUT, tanbQ_GUT]
        radcorrs_at_2TeV = my_radcorr_calc(2000, vHiggs, muQ, beta,
                                           y_t, y_c, y_u, y_b, y_s, y_d,
                                           y_tau, y_mu, y_e,
                                           np.sqrt(5 / 3) * g_pr, g_2, g_s,
                                           mQ3sq, mQ2sq, mQ1sq, mL3sq, mL2sq,
                                           mL1sq, mU3sq, mU2sq, mU1sq, mD3sq,
                                           mD2sq, mD1sq, mE3sq, mE2sq, mE1sq,
                                           my_M1, my_M2, my_M3, mHusq, mHdsq,
                                           a_t, a_c, a_u, a_b, a_s, a_d,
                                           a_tau, a_mu, a_e)
        dewlist = Delta_EW_calc(2000.0, vHiggs, muQ, beta, y_t,
                                y_c, y_u, y_b, y_s, y_d, y_tau, y_mu, y_e,
                                np.sqrt(5 / 3) * g_pr, g_2, g_s, mQ3sq, mQ2sq,
                                mQ1sq, mL3sq, mL2sq, mL1sq, mU3sq, mU2sq,
                                mU1sq, mD3sq, mD2sq, mD1sq, mE3sq, mE2sq,
                                mE1sq, my_M1, my_M2, my_M3, mHusq, mHdsq,
                                a_t, a_c, a_u, a_b, a_s, a_d, a_tau, a_mu, a_e)
        print('\nGiven the submitted SLHA file, your value for the electroweak'
              + ' naturalness measure, Delta_EW, is: '
              + str(dewlist[0][1]))
        print('\nThe ordered, signed contributions to Delta_EW are as follows '
              + '(decr. order): ')
        print('')
        for i in range(0, len(dewlist)):
            print(str(i + 1) + ': ' + str(dewlist[i][0]) + ', '
                  + str(dewlist[i][2]))

        myDelta_HS = Delta_HS_calc(RGE_sols[71],
                                   RGE_sols[28] - RGE_sols[71],
                                   RGE_sols[70],
                                   RGE_sols[27] - RGE_sols[70],
                                   np.power(RGE_sols[51], 2),
                                   np.power(RGE_sols[8], 2)
                                   - np.power(RGE_sols[51], 2),
                                   np.power(vHiggs, 2)
                                   * ((1 / 2)
                                      * (np.power(RGE_sols[3], 2)
                                         + (( 3 / 5)
                                            * np.power(RGE_sols[2], 2)))),
                                   np.power(RGE_sols[88], 2),
                                   radcorrs_at_2TeV[0],
                                   radcorrs_at_2TeV[1])
        print('\nYour value for the high-scale naturalness measure, Delta_HS,'
              + ' is: ' + str(myDelta_HS))
        # myDelta_BG = Delta_BG_calc(modinp, RGE_sols[51], RGE_sols[71],
        #                            RGE_sols[70], RGE_sols[48], RGE_sols[49],
        #                            RGE_sols[50], RGE_sols[74], RGE_sols[73],
        #                            RGE_sols[72], RGE_sols[80], RGE_sols[79],
        #                            RGE_sols[78], RGE_sols[83], RGE_sols[82],
        #                            RGE_sols[81], RGE_sols[77], RGE_sols[76],
        #                            RGE_sols[75], RGE_sols[86], RGE_sols[85],
        #                            RGE_sols[84], RGE_sols[61], RGE_sols[62],
        #                            RGE_sols[63], RGE_sols[64], RGE_sols[65],
        #                            RGE_sols[66], RGE_sols[67], RGE_sols[68],
        #                            RGE_sols[69], RGE_sols[70], RGE_sols[52],
        #                            RGE_sols[53], RGE_sols[54], RGE_sols[55],
        #                            RGE_sols[56], RGE_sols[57], RGE_sols[58],
        #                            RGE_sols[59], RGE_sols[60], tree_mzsq)
        # print('\nYour value for the Barbieri-Giudice naturalness measure,'
        #       + ' Delta_BG, is: ' + str(myDelta_BG[0][0]))
        # print('\nThe ordered contributions to Delta_BG are as follows '
        #       + '(decr. order): ')
        # print('')
        # for i in range(0, len(myDelta_BG)):
        #     print(str(i + 1) + ': ' + str(myDelta_BG[i][0]) + ', '
        #           + str(myDelta_BG[i][1]))
        checksavebool = True
        while checksavebool:
            checksave = input("\nWould you like to save these DEW results to a"
                              + " .txt file (will be saved to the"
                              + " directory " + str(os.getcwd())
                              + ")? Enter Y to save the result or"
                              + " N to continue: ")
            timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
            if checksave.lower() in ('y', 'yes'):
                filenamecheck = input('\nThe default file name is '
                                      + '"current_system_time_DEW_contrib_list'
                                      + '.txt'
                                      + '", e.g., '
                                      + timestr + '_DEW_contrib_list.txt.'
                                      + ' Would you like to keep this name or'
                                      + ' input your own file name?'
                                      +  ' Enter Y to keep the default file'
                                      + ' name'
                                      + ' or N to be able to input your own: ')
                if filenamecheck.lower() in ('y', 'yes'):
                    print('Given the submitted SLHA file, ' + str(direc) +
                          ', your value for the electroweak\n'
                          + 'naturalness measure, Delta_EW, is: '
                          + str(dewlist[0][1]),
                          file=open(timestr + "_DEW_contrib_list.txt", "w"))
                    print('\nThe ordered contributions to Delta_EW are as'
                          + ' follows (decr. order): ',
                          file=open(timestr + "_DEW_contrib_list.txt", "a"))
                    print('', file=open(timestr + "_DEW_contrib_list.txt",
                                        "a"))
                    for i in range(0, len(dewlist)):
                        print(str(i + 1) + ': ' + str(dewlist[i][0]) + ', '
                              + str(dewlist[i][2]),
                              file=open(timestr + "_DEW_contrib_list.txt",
                                        "a"))
                    print('\nThese results have been saved to the'
                          + ' directory ' + str(os.getcwd()) + ' as ' + timestr
                          + '_DEW_contrib_list.txt.\n')
                    checksavebool = False
                elif filenamecheck.lower() in ('n', 'no'):
                    newfilename = input('\nInput your desired filename with no'
                                        + ' whitespaces and without the .txt'
                                        + ' file '
                                        + 'extension (e.g. "my_SLHA_DEW_list"'
                                        + ' without the quotes): ')
                    print('Given the submitted SLHA file, ' + str(direc) +
                          ', your value for the electroweak\n'
                          + 'naturalness measure, Delta_EW, is: '
                          + str(dewlist[0][1]),
                          file=open(newfilename + ".txt", "w"))
                    print('\nThe ordered contributions to Delta_EW are as'
                          + ' follows (decr. order): ',
                          file=open(newfilename + ".txt", "a"))
                    print('', file=open(newfilename + ".txt", "a"))
                    for i in range(0, len(dewlist)):
                        print(str(i + 1) + ': ' + str(dewlist[i][0]) + ', '
                              + str(dewlist[i][2]),
                              file=open(newfilename + ".txt", "a"))
                    print('\nThese results have been saved to the'
                          + ' directory ' + str(os.getcwd())
                          + ' as ' + newfilename + '.txt.\n')
                    checksavebool = False
                else:
                    print("Invalid user input")
            else:
                print("\nOutput not saved.\n")
                checksavebool = False
        checkcontinue = input("Would you like to try again with a new SLHA "
                              + "file? Enter Y to try again or N to stop: ")
        if checkcontinue.lower() in ('y', 'yes'):
            userContinue = True
            print('')
        elif checkcontinue.lower() in ('n', 'no'):
            userContinue = False
        else:
            userContinue = True
            print("\nInvalid user input. Returning to SLHA directory input.\n")