#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 13:22:30 2023

Solve RGEs to weak scale and then to GUT scale to establish needed parameters.

@author: Dakotah Martinez
"""

import numpy as np
from scipy.integrate import solve_ivp
from constants import loop_fac, loop_fac_sq, b_1l, b_2l, c_2l
from math import ceil

def my_RGE_solver(BCs, inp_Q, target_Q_val):
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
        Lowest value for t parameter to run to in solution. Default is SLHA scale.

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
              (0: g1, 1: g2, 2: g3, 3: M1, 4: M2, 5: M3, 6: mu^2, 7: yt, 8: yc,
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
        musq_val = x[6]
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
        dg1_dt = (1) * ((loop_fac * dg1_dt_1l)
                        + (loop_fac_sq * dg1_dt_2l))

        dg2_dt = (1) * ((loop_fac * dg2_dt_1l)
                        + (loop_fac_sq * dg2_dt_2l))

        dg3_dt = (1) * ((loop_fac * dg3_dt_1l)
                        + (loop_fac_sq * dg3_dt_2l))

        dM1_dt = (2) * ((loop_fac * dM1_dt_1l)
                        + (loop_fac_sq * dM1_dt_2l))

        dM2_dt = (2) * ((loop_fac * dM2_dt_1l)
                        + (loop_fac_sq * dM2_dt_2l))

        dM3_dt = (2) * ((loop_fac * dM3_dt_1l)
                        + (loop_fac_sq * dM3_dt_2l))

        ##### Higgsino mass parameter mu #####
        # 1 loop part
        dmusq_dt_1l = (2*musq_val# Tr(3Yu^2 + 3Yd^2 + Ye^2)
                       * ((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                + np.power(yu_val, 2) + np.power(yb_val, 2)
                                + np.power(ys_val, 2) + np.power(yd_val, 2)))
                          + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                             + np.power(ye_val, 2))# end trace
                          - (3 * np.power(g2_val, 2))
                          - ((3 / 5) * np.power(g1_val, 2))))

        # 2 loop part
        dmusq_dt_2l = (2*musq_val# Tr(3Yu^4 + 3Yd^4 + (2Yu^2*Yd^2) + Ye^4)
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
        dmusq_dt = (1) * ((loop_fac * dmusq_dt_1l)
                          + (loop_fac_sq * dmusq_dt_2l))

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
        dyt_dt = (1) * ((loop_fac * dyt_dt_1l)
                            + (loop_fac_sq * dyt_dt_2l))

        dyc_dt = (1) * ((loop_fac * dyc_dt_1l)
                            + (loop_fac_sq * dyc_dt_2l))

        dyu_dt = (1) * ((loop_fac * dyu_dt_1l)
                            + (loop_fac_sq * dyu_dt_2l))

        dyb_dt = (1) * ((loop_fac * dyb_dt_1l)
                            + (loop_fac_sq * dyb_dt_2l))

        dys_dt = (1) * ((loop_fac * dys_dt_1l)
                            + (loop_fac_sq * dys_dt_2l))

        dyd_dt = (1) * ((loop_fac * dyd_dt_1l)
                            + (loop_fac_sq * dyd_dt_2l))

        dytau_dt = (1) * ((loop_fac * dytau_dt_1l)
                            + (loop_fac_sq * dytau_dt_2l))

        dymu_dt = (1) * ((loop_fac * dymu_dt_1l)
                            + (loop_fac_sq * dymu_dt_2l))

        dye_dt = (1) * ((loop_fac * dye_dt_1l)
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
        dat_dt = (1) * ((loop_fac * dat_dt_1l)
                        + (loop_fac_sq * dat_dt_2l))

        dac_dt = (1) * ((loop_fac * dac_dt_1l)
                        + (loop_fac_sq * dac_dt_2l))

        dau_dt = (1) * ((loop_fac * dau_dt_1l)
                        + (loop_fac_sq * dau_dt_2l))

        dab_dt = (1) * ((loop_fac * dab_dt_1l)
                        + (loop_fac_sq * dab_dt_2l))

        das_dt = (1) * ((loop_fac * das_dt_1l)
                        + (loop_fac_sq * das_dt_2l))

        dad_dt = (1) * ((loop_fac * dad_dt_1l)
                        + (loop_fac_sq * dad_dt_2l))

        datau_dt = (1) * ((loop_fac * datau_dt_1l)
                          + (loop_fac_sq * datau_dt_2l))

        damu_dt = (1) * ((loop_fac * damu_dt_1l)
                         + (loop_fac_sq * damu_dt_2l))

        dae_dt = (1) * ((loop_fac * dae_dt_1l)
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
                    + (np.sqrt(abs(musq_val))# Tr(6au*Yu + 6ad*Yd + 2ae*Ye)
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
                    + (np.sqrt(abs(musq_val)) * (((-12)# Tr(3au*Yu^3 + 3ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + ae*Ye^3)
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
        db_dt = (1) * ((loop_fac * db_dt_1l)
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
        dmHu_sq_dt = (1) * ((loop_fac * dmHu_sq_dt_1l)
                            + (loop_fac_sq * dmHu_sq_dt_2l))

        dmHd_sq_dt = (1) * ((loop_fac * dmHd_sq_dt_1l)
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
        dmQ3_sq_dt = (1) * ((loop_fac * dmQ3_sq_dt_1l)
                            + (loop_fac_sq * dmQ3_sq_dt_2l))

        dmQ2_sq_dt = (1) * ((loop_fac * dmQ2_sq_dt_1l)
                            + (loop_fac_sq * dmQ2_sq_dt_2l))

        dmQ1_sq_dt = (1) * ((loop_fac * dmQ1_sq_dt_1l)
                            + (loop_fac_sq * dmQ1_sq_dt_2l))

        dmL3_sq_dt = (1) * ((loop_fac * dmL3_sq_dt_1l)
                            + (loop_fac_sq * dmL3_sq_dt_2l))

        dmL2_sq_dt = (1) * ((loop_fac * dmL2_sq_dt_1l)
                            + (loop_fac_sq * dmL2_sq_dt_2l))

        dmL1_sq_dt = (1) * ((loop_fac * dmL1_sq_dt_1l)
                            + (loop_fac_sq * dmL1_sq_dt_2l))

        dmU3_sq_dt = (1) * ((loop_fac * dmU3_sq_dt_1l)
                            + (loop_fac_sq * dmU3_sq_dt_2l))

        dmU2_sq_dt = (1) * ((loop_fac * dmU2_sq_dt_1l)
                            + (loop_fac_sq * dmU2_sq_dt_2l))

        dmU1_sq_dt = (1) * ((loop_fac * dmU1_sq_dt_1l)
                            + (loop_fac_sq * dmU1_sq_dt_2l))

        dmD3_sq_dt = (1) * ((loop_fac * dmD3_sq_dt_1l)
                            + (loop_fac_sq * dmD3_sq_dt_2l))

        dmD2_sq_dt = (1) * ((loop_fac * dmD2_sq_dt_1l)
                            + (loop_fac_sq * dmD2_sq_dt_2l))

        dmD1_sq_dt = (1) * ((loop_fac * dmD1_sq_dt_1l)
                            + (loop_fac_sq * dmD1_sq_dt_2l))

        dmE3_sq_dt = (1) * ((loop_fac * dmE3_sq_dt_1l)
                            + (loop_fac_sq * dmE3_sq_dt_2l))

        dmE2_sq_dt = (1) * ((loop_fac * dmE2_sq_dt_1l)
                            + (loop_fac_sq * dmE2_sq_dt_2l))

        dmE1_sq_dt = (1) * ((loop_fac * dmE1_sq_dt_1l)
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
        dtanb_dt = (tanb_val) * ((loop_fac * dtanb_dt_1l)
                                 + (loop_fac_sq * dtanb_dt_2l))


        # Collect all for return
        dxdt = [dg1_dt, dg2_dt, dg3_dt, dM1_dt, dM2_dt, dM3_dt, dmusq_dt, dyt_dt,
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
        numpoints = ceil((np.log(target_Q_val / inp_Q + 0.1)) * 10000)
        t_vals = np.linspace(np.log(inp_Q), np.log(target_Q_val),
                             numpoints)
        t_vals[0] = np.log(inp_Q + 0.1)
        t_vals[-1] = np.log(target_Q_val)
        t_span = np.array([np.log(inp_Q + 0.1), np.log(target_Q_val)])

        # Now solve down to low scale
        sol = solve_ivp(my_odes, t_span, BCs, t_eval = t_vals,
                        dense_output=True, method='LSODA', atol=1e-9,
                        rtol=1e-9)
    else:
        numpoints = ceil((np.log((inp_Q + 0.1) / target_Q_val)) * 10000)
        t_vals = np.linspace(np.log(target_Q_val), np.log(inp_Q+0.1),
                             numpoints)
        t_vals[0] = np.log(target_Q_val)
        t_vals[-1] = np.log(inp_Q + 0.1)
        t_span = np.array([np.log(inp_Q + 0.1), np.log(target_Q_val)])

        # Now solve down to low scale
        sol = solve_ivp(my_odes, t_span, BCs, t_eval = t_vals[::-1],
                        dense_output=True, method='LSODA', atol=1e-9,
                        rtol=1e-9)
    myx1 = sol.y
    print("\nEstablished weak parameters.")
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

    # Now evolve results up to 5*10^17 GeV, find approximate unification scale,
    # where g1(Q) is closest to g2(Q),
    # and return results for calculation of naturalness measures.
    numpoints2 = ceil((np.log(5*1e17 / target_Q_val)) * 10000)
    t_vals2 = np.linspace(np.log(target_Q_val), np.log(5*1e17),
                          numpoints2)
    t_vals2[0] = np.log(target_Q_val)
    t_vals2[-1] = np.log(5*1e17)
    t_span2 = np.array([np.log(target_Q_val), np.log(5*1e17)])
    sol2 = solve_ivp(my_odes, t_span2, x1, t_eval = t_vals2,
                     dense_output=True, method='LSODA', atol=1e-9,
                     rtol=1e-9)
    x2 = sol2.y
    GUT_idx = np.where(np.abs(x2[0] - x2[1])
                       == np.amin(np.abs(x2[0]-x2[1])))[0][0]
    approx_GUT_scale = np.exp(t_vals2[GUT_idx])
    print("\nEstablished GUT-scale parameters.")
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
