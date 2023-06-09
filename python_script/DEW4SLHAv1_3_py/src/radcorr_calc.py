#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 13:40:49 2023

Calculate radiative corrections.

@author: Dakotah Martinez
"""

from mpmath import mp, mpf
from constants import loop_fac, loop_fac_sq
import numpy as np
from scipy.special import spence

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
    gpr_wk = g1_wk * mp.sqrt(3. / 5.)
    gpr_sq = mp.power(gpr_wk, 2)
    g2_sq = mp.power(g2_wk, 2)
    mu_wk_sq = mp.power(mu_wk, 2)

    ##### Fundamental equations: #####

    def logfunc(mass, Q_renorm_sq=mp.power(myQ, 2)):
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
        myf = mp.power(mass, 2) * (mp.log((mp.power(mass, 2))
                                          / (Q_renorm_sq)) - 1)
        return myf


    def logfunc2(masssq, Q_renorm_sq=mp.power(myQ, 2)):
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

        myf2 = masssq * (mp.log(mp.fabs((masssq) / (Q_renorm_sq))) - 1)
        return myf2

    sinsqb = mp.power(mp.sin(beta_wk), 2)
    cossqb = mp.power(mp.cos(beta_wk), 2)
    vu = vHiggs_wk * mp.sqrt(sinsqb)
    vd = vHiggs_wk * mp.sqrt(cossqb)
    vu_sq = mp.power(vu, 2)
    vd_sq = mp.power(vd, 2)
    v_sq = mp.power(vHiggs_wk, 2)
    tan_th_w = gpr_wk / g2_wk
    theta_w = mp.atan(tan_th_w)
    sinsq_th_w = mp.power(mp.sin(theta_w), 2)
    cos2b = mp.cos(2 * beta_wk)
    sin2b = mp.sin(2 * beta_wk)
    gz_sq = (mp.power(g2_wk, 2) + mp.power(gpr_wk, 2)) / 8

    ##### Mass relations: #####

    # W-boson tree-level running squared mass
    m_w_sq = (mp.power(g2_wk, 2) / 2) * v_sq

    # Z-boson tree-level running squared mass
    mz_q_sq = v_sq * ((mp.power(g2_wk, 2) + mp.power(gpr_wk, 2)) / 2)

    # Higgs psuedoscalar tree-level running squared mass
    mA0sq = 2 * mu_wk_sq + mHu_sq_wk + mHd_sq_wk

    # Top quark tree-level running mass
    mymt = yt_wk * vu
    mymtsq = mp.power(mymt, 2)

    # Bottom quark tree-level running mass
    mymb = yb_wk * vd
    mymbsq = mp.power(mymb, 2)

    # Tau tree-level running mass
    mymtau = ytau_wk * vd
    mymtausq = mp.power(mymtau, 2)

    # Charm quark tree-level running mass
    mymc = yc_wk * vu
    mymcsq = mp.power(mymc, 2)

    # Strange quark tree-level running mass
    myms = ys_wk * vd
    mymssq = mp.power(myms, 2)

    # Muon tree-level running mass
    mymmu = ymu_wk * vd
    mymmusq = mp.power(mymmu, 2)

    # Up quark tree-level running mass
    mymu = yu_wk * vu
    mymusq = mp.power(mymu, 2)

    # Down quark tree-level running mass
    mymd = yd_wk * vd
    mymdsq = mp.power(mymd, 2)

    # Electron tree-level running mass
    myme = ye_wk * vd
    mymesq = mp.power(myme, 2)

    # Sneutrino running masses
    mselecneutsq = mL1_sq_wk + (0.25 * (gpr_sq + g2_sq) * (vd_sq - vu_sq))
    msmuneutsq = mL2_sq_wk + (0.25 * (gpr_sq + g2_sq) * (vd_sq - vu_sq))
    mstauneutsq = mL3_sq_wk + (0.25 * (gpr_sq + g2_sq) * (vd_sq - vu_sq))

    # Tree-level charged Higgs floatt running squared mass.
    mH_pmsq = mA0sq + m_w_sq

    # Up-type squark mass eigenstate eigenvalues
    m_stop_1sq = (1 / 48)\
        * ((24 * (mQ3_sq_wk + mU3_sq_wk)) + (24 * v_sq * mp.power(yt_wk, 2))
           + (6 * v_sq * g2_sq * cos2b) - (20 * gpr_sq * v_sq * cos2b)
           - (24 * v_sq * mp.power(yt_wk, 2) * cos2b)
           - (6 * mp.sqrt((64 * v_sq * mp.power(yt_wk * mu_wk, 2) * cossqb)
                          + mp.power((4 * (mQ3_sq_wk - mU3_sq_wk))
                                     + (v_sq * (g2_sq + (2 * gpr_sq))
                                        * cos2b), 2)
                          + (64 * v_sq * sinsqb * mp.power(at_wk, 2))
                          - (64 * at_wk * v_sq * yt_wk * mu_wk * sin2b))))
    m_stop_2sq = (1 / 48)\
        * ((24 * (mQ3_sq_wk + mU3_sq_wk)) + (24 * v_sq * mp.power(yt_wk, 2))
           + (6 * v_sq * g2_sq * cos2b) - (20 * gpr_sq * v_sq * cos2b)
           - (24 * v_sq * mp.power(yt_wk, 2) * cos2b)
           + (6 * mp.sqrt((64 * v_sq * mp.power(yt_wk * mu_wk, 2) * cossqb)
                          + mp.power((4 * (mQ3_sq_wk - mU3_sq_wk))
                                     + (v_sq * (g2_sq + (2 * gpr_sq))
                                        * cos2b), 2)
                          + (64 * v_sq * sinsqb * mp.power(at_wk, 2))
                          - (64 * at_wk * v_sq * yt_wk * mu_wk * sin2b))))
    m_scharm_1sq = (1 / 48)\
        * ((24 * (mQ2_sq_wk + mU2_sq_wk)) + (24 * v_sq * mp.power(yc_wk, 2))
           + (6 * v_sq * g2_sq * cos2b) - (20 * gpr_sq * v_sq * cos2b)
           - (24 * v_sq * mp.power(yc_wk, 2) * cos2b)
           - (6 * mp.sqrt((64 * v_sq * mp.power(yc_wk * mu_wk, 2) * cossqb)
                          + mp.power((4 * (mQ2_sq_wk - mU2_sq_wk))
                                     + (v_sq * (g2_sq + (2 * gpr_sq))
                                        * cos2b), 2)
                          + (64 * v_sq * sinsqb * mp.power(ac_wk, 2))
                          - (64 * ac_wk * v_sq * yc_wk * mu_wk * sin2b))))
    m_scharm_2sq = (1 / 48)\
        * ((24 * (mQ2_sq_wk + mU2_sq_wk)) + (24 * v_sq * mp.power(yc_wk, 2))
           + (6 * v_sq * g2_sq * cos2b) - (20 * gpr_sq * v_sq * cos2b)
           - (24 * v_sq * mp.power(yc_wk, 2) * cos2b)
           + (6 * mp.sqrt((64 * v_sq * mp.power(yc_wk * mu_wk, 2) * cossqb)
                          + mp.power((4 * (mQ2_sq_wk - mU2_sq_wk))
                                     + (v_sq * (g2_sq + (2 * gpr_sq))
                                        * cos2b), 2)
                          + (64 * v_sq * sinsqb * mp.power(ac_wk, 2))
                          - (64 * ac_wk * v_sq * yc_wk * mu_wk * sin2b))))
    m_sup_1sq = (1 / 48)\
        * ((24 * (mQ1_sq_wk + mU1_sq_wk)) + (24 * v_sq * mp.power(yu_wk, 2))
           + (6 * v_sq * g2_sq * cos2b) - (20 * gpr_sq * v_sq * cos2b)
           - (24 * v_sq * mp.power(yu_wk, 2) * cos2b)
           - (6 * mp.sqrt((64 * v_sq * mp.power(yu_wk * mu_wk, 2) * cossqb)
                          + mp.power((4 * (mQ1_sq_wk - mU1_sq_wk))
                                     + (v_sq * (g2_sq + (2 * gpr_sq))
                                        * cos2b), 2)
                          + (64 * v_sq * sinsqb * mp.power(au_wk, 2))
                          - (64 * au_wk * v_sq * yu_wk * mu_wk * sin2b))))
    m_sup_2sq = (1 / 48)\
        * ((24 * (mQ1_sq_wk + mU1_sq_wk)) + (24 * v_sq * mp.power(yu_wk, 2))
           + (6 * v_sq * g2_sq * cos2b) - (20 * gpr_sq * v_sq * cos2b)
           - (24 * v_sq * mp.power(yu_wk, 2) * cos2b)
           + (6 * mp.sqrt((64 * v_sq * mp.power(yu_wk * mu_wk, 2) * cossqb)
                          + mp.power((4 * (mQ1_sq_wk - mU1_sq_wk))
                                     + (v_sq * (g2_sq + (2 * gpr_sq))
                                        * cos2b), 2)
                          + (64 * v_sq * sinsqb * mp.power(au_wk, 2))
                          - (64 * au_wk * v_sq * yu_wk * mu_wk * sin2b))))

    # Down-type squark mass eigenstate eigenvalues
    m_sbot_1sq = (1 / 48)\
        * ((24 * (mQ3_sq_wk + mD3_sq_wk)) + (24 * v_sq * mp.power(yb_wk, 2))
           - (6 * v_sq * g2_sq * cos2b) + (4 * gpr_sq * v_sq * cos2b)
           + (24 * v_sq * mp.power(yb_wk, 2) * cos2b)
           - (6 * mp.sqrt((64 * v_sq * mp.power(yb_wk * mu_wk, 2) * sinsqb)
                          + mp.power((4 * (mD3_sq_wk - mQ3_sq_wk))
                                     + (v_sq * (g2_sq + (2 * gpr_sq))
                                        * cos2b), 2)
                          + (64 * v_sq * cossqb * mp.power(ab_wk, 2))
                          - (64 * ab_wk * v_sq * yb_wk * mu_wk * sin2b))))
    m_sbot_2sq = (1 / 48)\
        * ((24 * (mQ3_sq_wk + mD3_sq_wk)) + (24 * v_sq * mp.power(yb_wk, 2))
           - (6 * v_sq * g2_sq * cos2b) + (4 * gpr_sq * v_sq * cos2b)
           + (24 * v_sq * mp.power(yb_wk, 2) * cos2b)
           + (6 * mp.sqrt((64 * v_sq * mp.power(yb_wk * mu_wk, 2) * sinsqb)
                          + mp.power((4 * (mD3_sq_wk - mQ3_sq_wk))
                                     + (v_sq * (g2_sq + (2 * gpr_sq))
                                        * cos2b), 2)
                          + (64 * v_sq * cossqb * mp.power(ab_wk, 2))
                          - (64 * ab_wk * v_sq * yb_wk * mu_wk * sin2b))))
    m_sstrange_1sq = (1 / 48)\
        * ((24 * (mQ2_sq_wk + mD2_sq_wk)) + (24 * v_sq * mp.power(ys_wk, 2))
           - (6 * v_sq * g2_sq * cos2b) + (4 * gpr_sq * v_sq * cos2b)
           + (24 * v_sq * mp.power(ys_wk, 2) * cos2b)
           - (6 * mp.sqrt((64 * v_sq * mp.power(ys_wk * mu_wk, 2) * sinsqb)
                          + mp.power((4 * (mD2_sq_wk - mQ2_sq_wk))
                                     + (v_sq * (g2_sq + (2 * gpr_sq))
                                        * cos2b), 2)
                          + (64 * v_sq * cossqb * mp.power(as_wk, 2))
                          - (64 * as_wk * v_sq * ys_wk * mu_wk * sin2b))))
    m_sstrange_2sq = (1 / 48)\
        * ((24 * (mQ2_sq_wk + mD2_sq_wk)) + (24 * v_sq * mp.power(ys_wk, 2))
           - (6 * v_sq * g2_sq * cos2b) + (4 * gpr_sq * v_sq * cos2b)
           + (24 * v_sq * mp.power(ys_wk, 2) * cos2b)
           + (6 * mp.sqrt((64 * v_sq * mp.power(ys_wk * mu_wk, 2) * sinsqb)
                          + mp.power((4 * (mD2_sq_wk - mQ2_sq_wk))
                                     + (v_sq * (g2_sq + (2 * gpr_sq))
                                        * cos2b), 2)
                          + (64 * v_sq * cossqb * mp.power(as_wk, 2))
                          - (64 * as_wk * v_sq * ys_wk * mu_wk * sin2b))))
    m_sdown_1sq = (1 / 48)\
        * ((24 * (mQ1_sq_wk + mD1_sq_wk)) + (24 * v_sq * mp.power(yd_wk, 2))
           - (6 * v_sq * g2_sq * cos2b) + (4 * gpr_sq * v_sq * cos2b)
           + (24 * v_sq * mp.power(yd_wk, 2) * cos2b)
           - (6 * mp.sqrt((64 * v_sq * mp.power(yd_wk * mu_wk, 2) * sinsqb)
                          + mp.power((4 * (mD1_sq_wk - mQ1_sq_wk))
                                     + (v_sq * (g2_sq + (2 * gpr_sq))
                                        * cos2b), 2)
                          + (64 * v_sq * cossqb * mp.power(ad_wk, 2))
                          - (64 * ad_wk * v_sq * yd_wk * mu_wk * sin2b))))
    m_sdown_2sq = (1 / 48)\
        * ((24 * (mQ1_sq_wk + mD1_sq_wk)) + (24 * v_sq * mp.power(yd_wk, 2))
           - (6 * v_sq * g2_sq * cos2b) + (4 * gpr_sq * v_sq * cos2b)
           + (24 * v_sq * mp.power(yd_wk, 2) * cos2b)
           + (6 * mp.sqrt((64 * v_sq * mp.power(yd_wk * mu_wk, 2) * sinsqb)
                          + mp.power((4 * (mD1_sq_wk - mQ1_sq_wk))
                                     + (v_sq * (g2_sq + (2 * gpr_sq))
                                        * cos2b), 2)
                          + (64 * v_sq * cossqb * mp.power(ad_wk, 2))
                          - (64 * ad_wk * v_sq * yd_wk * mu_wk * sin2b))))

    # Slepton mass eigenstate eigenvalues
    m_stau_1sq = (1 / 8)\
        * ((4 * (mE3_sq_wk + mL3_sq_wk)) + (4 * v_sq * mp.power(ytau_wk, 2))
           - (v_sq * g2_sq * cos2b) + (6 * gpr_sq * v_sq * cos2b)
           + (4 * v_sq * mp.power(ytau_wk, 2) * cos2b)
           - (mp.sqrt((64 * v_sq * mp.power(ytau_wk * mu_wk, 2) * sinsqb)
                      + mp.power((4 * (mE3_sq_wk - mL3_sq_wk))
                                 + (v_sq * (g2_sq + (2 * gpr_sq))
                                    * cos2b), 2)
                      + (64 * v_sq * cossqb * mp.power(atau_wk, 2))
                      - (64 * atau_wk * v_sq * ytau_wk * mu_wk * sin2b))))
    m_stau_2sq = (1 / 8)\
        * ((4 * (mE3_sq_wk + mL3_sq_wk)) + (4 * v_sq * mp.power(ytau_wk, 2))
           - (v_sq * g2_sq * cos2b) + (6 * gpr_sq * v_sq * cos2b)
           + (4 * v_sq * mp.power(ytau_wk, 2) * cos2b)
           + (mp.sqrt((64 * v_sq * mp.power(ytau_wk * mu_wk, 2) * sinsqb)
                      + mp.power((4 * (mE3_sq_wk - mL3_sq_wk))
                                 + (v_sq * (g2_sq + (2 * gpr_sq))
                                    * cos2b), 2)
                      + (64 * v_sq * cossqb * mp.power(atau_wk, 2))
                      - (64 * atau_wk * v_sq * ytau_wk * mu_wk * sin2b))))
    m_smu_1sq = (1 / 8)\
        * ((4 * (mE2_sq_wk + mL2_sq_wk)) + (4 * v_sq * mp.power(ymu_wk, 2))
           - (v_sq * g2_sq * cos2b) + (6 * gpr_sq * v_sq * cos2b)
           + (4 * v_sq * mp.power(ymu_wk, 2) * cos2b)
           - (mp.sqrt((64 * v_sq * mp.power(ymu_wk * mu_wk, 2) * sinsqb)
                      + mp.power((4 * (mE2_sq_wk - mL2_sq_wk))
                                 + (v_sq * (g2_sq + (2 * gpr_sq))
                                    * cos2b), 2)
                      + (64 * v_sq * cossqb * mp.power(amu_wk, 2))
                      - (64 * amu_wk * v_sq * ymu_wk * mu_wk * sin2b))))
    m_smu_2sq = (1 / 8)\
        * ((4 * (mE2_sq_wk + mL2_sq_wk)) + (4 * v_sq * mp.power(ymu_wk, 2))
           - (v_sq * g2_sq * cos2b) + (6 * gpr_sq * v_sq * cos2b)
           + (4 * v_sq * mp.power(ymu_wk, 2) * cos2b)
           + (mp.sqrt((64 * v_sq * mp.power(ymu_wk * mu_wk, 2) * sinsqb)
                      + mp.power((4 * (mE2_sq_wk - mL2_sq_wk))
                                 + (v_sq * (g2_sq + (2 * gpr_sq))
                                    * cos2b), 2)
                      + (64 * v_sq * cossqb * mp.power(amu_wk, 2))
                      - (64 * amu_wk * v_sq * ymu_wk * mu_wk * sin2b))))
    m_se_1sq = (1 / 8)\
        * ((4 * (mE1_sq_wk + mL1_sq_wk)) + (4 * v_sq * mp.power(ye_wk, 2))
           - (v_sq * g2_sq * cos2b) + (6 * gpr_sq * v_sq * cos2b)
           + (4 * v_sq * mp.power(ye_wk, 2) * cos2b)
           - (mp.sqrt((64 * v_sq * mp.power(ye_wk * mu_wk, 2) * sinsqb)
                      + mp.power((4 * (mE1_sq_wk - mL1_sq_wk))
                                 + (v_sq * (g2_sq + (2 * gpr_sq))
                                    * cos2b), 2)
                      + (64 * v_sq * cossqb * mp.power(ae_wk, 2))
                      - (64 * ae_wk * v_sq * ye_wk * mu_wk * sin2b))))
    m_se_2sq = (1 / 8)\
        * ((4 * (mE1_sq_wk + mL1_sq_wk)) + (4 * v_sq * mp.power(ye_wk, 2))
           - (v_sq * g2_sq * cos2b) + (6 * gpr_sq * v_sq * cos2b)
           + (4 * v_sq * mp.power(ye_wk, 2) * cos2b)
           + (mp.sqrt((64 * v_sq * mp.power(ye_wk * mu_wk, 2) * sinsqb)
                      + mp.power((4 * (mE1_sq_wk - mL1_sq_wk))
                                 + (v_sq * (g2_sq + (2 * gpr_sq))
                                    * cos2b), 2)
                      + (64 * v_sq * cossqb * mp.power(ae_wk, 2))
                      - (64 * ae_wk * v_sq * ye_wk * mu_wk * sin2b))))

    # Chargino mass eigenstate eigenvalues
    msC1sq = (0.5)\
        * ((g2_sq * v_sq) + mu_wk_sq + mp.power(M2_wk, 2)
           - mp.sqrt((mp.power(M2_wk + mu_wk, 2)
                      + (g2_sq * v_sq * (1 - sin2b)))
                     * (mp.power(M2_wk - mu_wk, 2)
                        + (g2_sq * v_sq * (1 + sin2b)))))
    msC2sq = (0.5)\
        * ((g2_sq * v_sq) + mu_wk_sq + mp.power(M2_wk, 2)
           + mp.sqrt((mp.power(M2_wk + mu_wk, 2)
                      + (g2_sq * v_sq * (1 - sin2b)))
                     * (mp.power(M2_wk - mu_wk, 2)
                        + (g2_sq * v_sq * (1 + sin2b)))))

    # Neutralino mass eigenstate eigenvalues
    neut_mass_mat = \
        mp.matrix([[M1_wk, 0, (-1) * gpr_wk * vd / mp.sqrt(2),
                    gpr_wk * vu / mp.sqrt(2)],
                   [0, M2_wk, g2_wk * vd / mp.sqrt(2),
                    (-1) * g2_wk * vu / mp.sqrt(2)],
                   [(-1) * gpr_wk * vd / mp.sqrt(2),
                    g2_wk * vd / mp.sqrt(2), 0, (-1) * mu_wk],
                   [gpr_wk * vu / mp.sqrt(2), (-1) * g2_wk * vu / mp.sqrt(2),
                    (-1) * mu_wk, 0]])
    my_neut_mass_eigvals, my_neut_mass_eigvecs = mp.eig(neut_mass_mat)
    sorted_mass_eigvals = sorted(my_neut_mass_eigvals, key=abs)
    mneutrsq = np.power(sorted_mass_eigvals, 2)
    msN1sq = mneutrsq[0]
    msN2sq = mneutrsq[1]
    msN3sq = mneutrsq[2]
    msN4sq = mneutrsq[3]
    # Neutral Higgs doublet mass eigenstate running squared masses
    mh0sq = (1 / 4)\
        * ((2 * mA0sq) + (2 * mz_q_sq)
           - mp.sqrt((4 * mp.power(mA0sq, 2)) + (mp.power(v_sq
                                                          * (g2_sq + gpr_sq),
                                                          2))
                     - (4 * (g2_sq + gpr_sq) * mA0sq * v_sq
                        * mp.cos(4 * beta_wk))))
    mH0sq = (1 / 4)\
        * ((2 * mA0sq) + (2 * mz_q_sq)
           + mp.sqrt((4 * mp.power(mA0sq, 2)) + (mp.power(v_sq
                                                          * (g2_sq + gpr_sq),
                                                          2))
                     - (4 * (g2_sq + gpr_sq) * mA0sq * v_sq
                        * mp.cos(4 * beta_wk))))

    ##### Radiative corrections in stop squark sector #####

    stop_denom = mp.sqrt((64 * v_sq * mp.power(yt_wk * mu_wk, 2) * cossqb)
                         + mp.power((4 * (mQ3_sq_wk - mU3_sq_wk))
                                    + (v_sq * (g2_sq + (2 * gpr_sq))
                                       * cos2b), 2)
                         + (64 * v_sq * sinsqb * mp.power(at_wk, 2))
                         - (64 * at_wk * v_sq * yt_wk * mu_wk * sin2b))
    stopuu_num = (4 * mp.power(at_wk, 2)) - ((1 / 2) * (g2_sq + (2 * gpr_sq))
                                             * (mQ3_sq_wk - mU3_sq_wk))\
        - ((1 / 8) * mp.power(g2_sq + (2 * gpr_sq), 2) * v_sq * cos2b)\
        - (4 * at_wk * yt_wk * mu_wk * 1 / mp.tan(beta_wk))
    stopdd_num = ((1 / 2) * (g2_sq + (2 * gpr_sq)) * (mQ3_sq_wk - mU3_sq_wk))\
        + (4 * mp.power(yt_wk * mu_wk, 2)) + (mp.power(g2_sq + (2 * gpr_sq), 2)
                                              * v_sq * cos2b / 8)\
        - (4 * at_wk * yt_wk * mu_wk * mp.tan(beta_wk))
    sigmauu_stop_1 = (3 * loop_fac) * logfunc2(m_stop_1sq) \
        * (mp.power(yt_wk, 2) - (g2_sq / 8) + (5 * gpr_sq / 12)
           - (stopuu_num / stop_denom))
    sigmauu_stop_2 = (3 * loop_fac) * logfunc2(m_stop_2sq) \
        * (mp.power(yt_wk, 2) - (g2_sq / 8) + (5 * gpr_sq / 12)
           + (stopuu_num / stop_denom))
    sigmadd_stop_1 = (3 * loop_fac) * logfunc2(m_stop_1sq) \
        * ((g2_sq / 8) - (5 * gpr_sq / 12)
           - (stopdd_num / stop_denom))
    sigmadd_stop_2 = (3 * loop_fac) * logfunc2(m_stop_2sq) \
        * ((g2_sq / 8) - (5 * gpr_sq / 12)
           + (stopdd_num / stop_denom))

    ##### Radiative corrections in sbottom squark sector #####
    sbot_denom = mp.sqrt((64 * v_sq * mp.power(yb_wk * mu_wk, 2) * sinsqb)
                         + mp.power((4 * (mD3_sq_wk - mQ3_sq_wk))
                                    + (v_sq * (g2_sq + (2 * gpr_sq))
                                       * cos2b), 2)
                         + (64 * v_sq * cossqb * mp.power(ab_wk, 2))
                         - (64 * ab_wk * v_sq * yb_wk * mu_wk * sin2b))
    sbotuu_num = ((mQ3_sq_wk - mD3_sq_wk) * (g2_sq + (2 * gpr_sq)) / 2)\
        + (4 * mp.power(yb_wk * mu_wk, 2)) - (4 * ab_wk * yb_wk * mu_wk
                                              / mp.tan(beta_wk))\
        - (v_sq * cos2b * mp.power(g2_sq + (2 * gpr_sq), 2) / 8)
    sbotdd_num = ((-4) * ab_wk * yb_wk * mu_wk * mp.tan(beta_wk))\
        + (4 * mp.power(ab_wk, 2)) + ((1 / 8) * mp.power(g2_sq + (2 * gpr_sq),
                                                         2) * v_sq * cos2b)\
        + ((g2_sq + (2 * gpr_sq)) * (mD3_sq_wk - mQ3_sq_wk) / 2)
    sigmauu_sbot_1 = (3 * loop_fac) * logfunc2(m_sbot_1sq) \
        * (((1 / 24) * ((3 * g2_sq) - (2 * gpr_sq)))
           - (sbotuu_num / sbot_denom))
    sigmauu_sbot_2 = (3 * loop_fac) * logfunc2(m_sbot_2sq) \
        * (((1 / 24) * ((3 * g2_sq) - (2 * gpr_sq)))
           + (sbotuu_num / sbot_denom))
    sigmadd_sbot_1 = (3 * loop_fac) * logfunc2(m_sbot_1sq) \
        * (((gpr_sq / 12) - (g2_sq / 8)) + (mp.power(yb_wk, 2))
           - (sbotdd_num / sbot_denom))
    sigmadd_sbot_2 = (3 * loop_fac) * logfunc2(m_sbot_2sq) \
        * (((gpr_sq / 12) - (g2_sq / 8)) + (mp.power(yb_wk, 2))
           + (sbotdd_num / sbot_denom))

    ##### Radiative corrections in stau slepton sector #####
    stau_denom = mp.sqrt((64 * v_sq * mp.power(ytau_wk * mu_wk, 2) * sinsqb)
                         + mp.power((4 * (mE3_sq_wk - mL3_sq_wk))
                                    + (v_sq * (g2_sq + (2 * gpr_sq))
                                       * cos2b), 2)
                         + (64 * v_sq * cossqb * mp.power(atau_wk, 2))
                         - (64 * atau_wk * v_sq * ytau_wk * mu_wk * sin2b))
    stauuu_num = (4 * mp.power(ytau_wk * mu_wk, 2)) - (4 * atau_wk
                                                       * ytau_wk * mu_wk
                                                       / mp.tan(beta_wk))\
        - ((mE3_sq_wk - mL3_sq_wk) * (g2_sq + (2 * gpr_sq)) / 2)\
        - ((1 / 8) * v_sq * cos2b * mp.power(g2_sq + (2 * gpr_sq), 2))
    staudd_num = (1 / 8)\
        * ((mp.power(g2_sq + (2 * gpr_sq), 2) * v_sq * cos2b)
           + (4 * ((8 * mp.power(atau_wk, 2))
                   + ((g2_sq + (2 * gpr_sq)) * (mE3_sq_wk - mL3_sq_wk))
                   - (8 * atau_wk * ytau_wk * mu_wk * mp.tan(beta_wk)))))
    sigmauu_stau_1 = (loop_fac) * logfunc2(m_stau_1sq) \
        * (((1 / 8) * (g2_sq - (6 * gpr_sq))) - (stauuu_num / stau_denom))
    sigmauu_stau_2 = (loop_fac) * logfunc2(m_stau_2sq) \
        * (((1 / 8) * (g2_sq - (6 * gpr_sq))) + (stauuu_num / stau_denom))
    sigmadd_stau_1 = (loop_fac) * logfunc2(m_stau_1sq) \
        * (mp.power(ytau_wk, 2) - (g2_sq / 8) + ((3 / 4) * gpr_sq)
           - (staudd_num / stau_denom))
    sigmadd_stau_2 = (loop_fac) * logfunc2(m_stau_1sq) \
        * (mp.power(ytau_wk, 2) - (g2_sq / 8) + ((3 / 4) * gpr_sq)
           + (staudd_num / stau_denom))
    # Tau sneutrino
    sigmauu_stau_sneut = (loop_fac / 8) * ((-1) * (g2_sq + gpr_sq))\
        * logfunc2(mstauneutsq)
    sigmadd_stau_sneut = (loop_fac / 8) * ((g2_sq + gpr_sq))\
        * logfunc2(mstauneutsq)

    ##### Radiative corrections from 2nd generation sfermions #####
    # Scharm sector
    schm_denom = mp.sqrt((64 * v_sq * mp.power(yc_wk * mu_wk, 2) * cossqb)
                         + mp.power((4 * (mQ2_sq_wk - mU2_sq_wk))
                                    + (v_sq * (g2_sq + (2 * gpr_sq))
                                       * cos2b), 2)
                         + (64 * v_sq * sinsqb * mp.power(ac_wk, 2))
                         - (64 * ac_wk * v_sq * yc_wk * mu_wk * sin2b))
    schmuu_num = (4 * mp.power(ac_wk, 2)) - ((1 / 2) * (g2_sq + (2 * gpr_sq))
                                             * (mQ2_sq_wk - mU2_sq_wk))\
        - ((1 / 8) * mp.power(g2_sq + (2 * gpr_sq), 2) * v_sq * cos2b)\
        - (4 * ac_wk * yc_wk * mu_wk * 1 / mp.tan(beta_wk))
    schmdd_num = ((1 / 2) * (g2_sq + (2 * gpr_sq)) * (mQ2_sq_wk - mU2_sq_wk))\
        + (4 * mp.power(yc_wk * mu_wk, 2)) + (mp.power(g2_sq + (2 * gpr_sq), 2)
                                              * v_sq * cos2b / 8)\
        - (4 * ac_wk * yc_wk * mu_wk * mp.tan(beta_wk))
    sigmauu_scharm_1 = (3 * loop_fac) * logfunc2(m_scharm_1sq) \
        * (mp.power(yc_wk, 2) - (g2_sq / 8) + (5 * gpr_sq / 12)
           - (schmuu_num / schm_denom))
    sigmauu_scharm_2 = (3 * loop_fac) * logfunc2(m_scharm_2sq) \
        * (mp.power(yc_wk, 2) - (g2_sq / 8) + (5 * gpr_sq / 12)
           + (schmuu_num / schm_denom))
    sigmadd_scharm_1 = (3 * loop_fac) * logfunc2(m_scharm_1sq) \
        * ((g2_sq / 8) - (5 * gpr_sq / 12)
           - (schmdd_num / schm_denom))
    sigmadd_scharm_2 = (3 * loop_fac) * logfunc2(m_scharm_2sq) \
        * ((g2_sq / 8) - (5 * gpr_sq / 12)
           + (schmdd_num / schm_denom))
    # Sstrange sector
    sstr_denom = mp.sqrt((64 * v_sq * mp.power(ys_wk * mu_wk, 2) * sinsqb)
                         + mp.power((4 * (mD2_sq_wk - mQ2_sq_wk))
                                    + (v_sq * (g2_sq + (2 * gpr_sq))
                                       * cos2b), 2)
                         + (64 * v_sq * cossqb * mp.power(as_wk, 2))
                         - (64 * as_wk * v_sq * ys_wk * mu_wk * sin2b))
    sstruu_num = ((mQ2_sq_wk - mD2_sq_wk) * (g2_sq + (2 * gpr_sq)) / 2)\
        + (4 * mp.power(ys_wk * mu_wk, 2)) - (4 * as_wk * ys_wk * mu_wk
                                              / mp.tan(beta_wk))\
        - (v_sq * cos2b * mp.power(g2_sq + (2 * gpr_sq), 2) / 8)
    sstrdd_num = ((-4) * as_wk * ys_wk * mu_wk * mp.tan(beta_wk))\
        + (4 * mp.power(as_wk, 2)) + ((1 / 8) * mp.power(g2_sq + (2 * gpr_sq),
                                                         2) * v_sq * cos2b)\
        + ((g2_sq + (2 * gpr_sq)) * (mD2_sq_wk - mQ2_sq_wk) / 2)
    sigmauu_sstrange_1 = (3 * loop_fac) * logfunc2(m_sstrange_1sq) \
        * (((1 / 24) * ((3 * g2_sq) - (2 * gpr_sq)))
           - (sstruu_num / sstr_denom))
    sigmauu_sstrange_2 = (3 * loop_fac) * logfunc2(m_sstrange_2sq) \
        * (((1 / 24) * ((3 * g2_sq) - (2 * gpr_sq)))
           + (sstruu_num / sstr_denom))
    sigmadd_sstrange_1 = (3 * loop_fac) * logfunc2(m_sstrange_1sq) \
        * (((gpr_sq / 12) - (g2_sq / 8)) + (mp.power(ys_wk, 2))
           - (sstrdd_num / sstr_denom))
    sigmadd_sstrange_2 = (3 * loop_fac) * logfunc2(m_sstrange_2sq) \
        * (((gpr_sq / 12) - (g2_sq / 8)) + (mp.power(ys_wk, 2))
           + (sstrdd_num / sstr_denom))

    # Smu/smu sneutrino

    smu_denom = mp.sqrt((64 * v_sq * mp.power(ymu_wk * mu_wk, 2) * sinsqb)
                        + mp.power((4 * (mE2_sq_wk - mL2_sq_wk))
                                   + (v_sq * (g2_sq + (2 * gpr_sq))
                                      * cos2b), 2)
                        + (64 * v_sq * cossqb * mp.power(amu_wk, 2))
                        - (64 * amu_wk * v_sq * ymu_wk * mu_wk * sin2b))
    smuuu_num = (4 * mp.power(ymu_wk * mu_wk, 2)) - (4 * amu_wk
                                                     * ymu_wk * mu_wk
                                                     / mp.tan(beta_wk))\
        - ((mE2_sq_wk - mL2_sq_wk) * (g2_sq + (2 * gpr_sq)) / 2)\
        - ((1 / 8) * v_sq * cos2b * mp.power(g2_sq + (2 * gpr_sq), 2))
    smudd_num = (1 / 8)\
        * ((mp.power(g2_sq + (2 * gpr_sq), 2) * v_sq * cos2b)
           + (4 * ((8 * mp.power(amu_wk, 2))
                   + ((g2_sq + (2 * gpr_sq)) * (mE2_sq_wk - mL2_sq_wk))
                   - (8 * amu_wk * ymu_wk * mu_wk * mp.tan(beta_wk)))))
    sigmauu_smu_1 = (loop_fac) * logfunc2(m_smu_1sq) \
        * (((1 / 8) * (g2_sq - (6 * gpr_sq))) - (smuuu_num / smu_denom))
    sigmauu_smu_2 = (loop_fac) * logfunc2(m_smu_2sq) \
        * (((1 / 8) * (g2_sq - (6 * gpr_sq))) + (smuuu_num / smu_denom))
    sigmadd_smu_1 = (loop_fac) * logfunc2(m_smu_1sq) \
        * (mp.power(ymu_wk, 2) - (g2_sq / 8) + ((3 / 4) * gpr_sq)
           - (smudd_num / smu_denom))
    sigmadd_smu_2 = (loop_fac) * logfunc2(m_smu_2sq) \
        * (mp.power(ymu_wk, 2) - (g2_sq / 8) + ((3 / 4) * gpr_sq)
           + (smudd_num / smu_denom))
    # Mu sneutrino
    sigmauu_smu_sneut = (loop_fac / 8) * ((-1) * (g2_sq + gpr_sq))\
        * logfunc2(msmuneutsq)
    sigmadd_smu_sneut = (loop_fac / 8) * ((g2_sq + gpr_sq))\
        * logfunc2(msmuneutsq)

    ##### Radiative corrections from 1st generation sfermions #####
    # Sup sector

    sup_denom = mp.sqrt((64 * v_sq * mp.power(yu_wk * mu_wk, 2) * cossqb)
                         + mp.power((4 * (mQ1_sq_wk - mU1_sq_wk))
                                    + (v_sq * (g2_sq + (2 * gpr_sq))
                                       * cos2b), 2)
                         + (64 * v_sq * sinsqb * mp.power(au_wk, 2))
                         - (64 * au_wk * v_sq * yu_wk * mu_wk * sin2b))
    supuu_num = (4 * mp.power(au_wk, 2)) - ((1 / 2) * (g2_sq + (2 * gpr_sq))
                                             * (mQ1_sq_wk - mU1_sq_wk))\
        - ((1 / 8) * mp.power(g2_sq + (2 * gpr_sq), 2) * v_sq * cos2b)\
        - (4 * au_wk * yu_wk * mu_wk * 1 / mp.tan(beta_wk))
    supdd_num = ((1 / 2) * (g2_sq + (2 * gpr_sq)) * (mQ1_sq_wk - mU1_sq_wk))\
        + (4 * mp.power(yu_wk * mu_wk, 2)) + (mp.power(g2_sq + (2 * gpr_sq), 2)
                                              * v_sq * cos2b / 8)\
        - (4 * au_wk * yu_wk * mu_wk * mp.tan(beta_wk))
    sigmauu_sup_1 = (3 * loop_fac) * logfunc2(m_sup_1sq) \
        * (mp.power(yu_wk, 2) - (g2_sq / 8) + (5 * gpr_sq / 12)
           - (supuu_num / sup_denom))
    sigmauu_sup_2 = (3 * loop_fac) * logfunc2(m_sup_2sq) \
        * (mp.power(yu_wk, 2) - (g2_sq / 8) + (5 * gpr_sq / 12)
           + (supuu_num / sup_denom))
    sigmadd_sup_1 = (3 * loop_fac) * logfunc2(m_sup_1sq) \
        * ((g2_sq / 8) - (5 * gpr_sq / 12)
           - (supdd_num / sup_denom))
    sigmadd_sup_2 = (3 * loop_fac) * logfunc2(m_sup_2sq) \
        * ((g2_sq / 8) - (5 * gpr_sq / 12)
           + (supdd_num / sup_denom))
    # Sdown sector
    sdwn_denom = mp.sqrt((64 * v_sq * mp.power(yd_wk * mu_wk, 2) * sinsqb)
                         + mp.power((4 * (mD1_sq_wk - mQ1_sq_wk))
                                    + (v_sq * (g2_sq + (2 * gpr_sq))
                                       * cos2b), 2)
                         + (64 * v_sq * cossqb * mp.power(ad_wk, 2))
                         - (64 * ad_wk * v_sq * yd_wk * mu_wk * sin2b))
    sdwnuu_num = ((mQ1_sq_wk - mD1_sq_wk) * (g2_sq + (2 * gpr_sq)) / 2)\
        + (4 * mp.power(yd_wk * mu_wk, 2)) - (4 * ad_wk * yd_wk * mu_wk
                                              / mp.tan(beta_wk))\
        - (v_sq * cos2b * mp.power(g2_sq + (2 * gpr_sq), 2) / 8)
    sdwndd_num = ((-4) * ad_wk * yd_wk * mu_wk * mp.tan(beta_wk))\
        + (4 * mp.power(ad_wk, 2)) + ((1 / 8) * mp.power(g2_sq + (2 * gpr_sq),
                                                         2) * v_sq * cos2b)\
        + ((g2_sq + (2 * gpr_sq)) * (mD1_sq_wk - mQ1_sq_wk) / 2)
    sigmauu_sdown_1 = (3 * loop_fac) * logfunc2(m_sdown_1sq) \
        * (((1 / 24) * ((3 * g2_sq) - (2 * gpr_sq)))
           - (sdwnuu_num / sdwn_denom))
    sigmauu_sdown_2 = (3 * loop_fac) * logfunc2(m_sdown_2sq) \
        * (((1 / 24) * ((3 * g2_sq) - (2 * gpr_sq)))
           + (sdwnuu_num / sdwn_denom))
    sigmadd_sdown_1 = (3 * loop_fac) * logfunc2(m_sdown_1sq) \
        * (((gpr_sq / 12) - (g2_sq / 8)) + (mp.power(yd_wk, 2))
           - (sdwndd_num / sdwn_denom))
    sigmadd_sdown_2 = (3 * loop_fac) * logfunc2(m_sdown_2sq) \
        * (((gpr_sq / 12) - (g2_sq / 8)) + (mp.power(yd_wk, 2))
           + (sdwndd_num / sdwn_denom))
    # Selectron/selectron sneutrino
    sel_denom = mp.sqrt((64 * v_sq * mp.power(ye_wk * mu_wk, 2) * sinsqb)
                         + mp.power((4 * (mE1_sq_wk - mL1_sq_wk))
                                    + (v_sq * (g2_sq + (2 * gpr_sq))
                                       * cos2b), 2)
                         + (64 * v_sq * cossqb * mp.power(ae_wk, 2))
                         - (64 * ae_wk * v_sq * ye_wk * mu_wk * sin2b))
    seluu_num = (4 * mp.power(ye_wk * mu_wk, 2)) - (4 * ae_wk
                                                    * ye_wk * mu_wk
                                                    / mp.tan(beta_wk))\
        - ((mE1_sq_wk - mL1_sq_wk) * (g2_sq + (2 * gpr_sq)) / 2)\
        - ((1 / 8) * v_sq * cos2b * mp.power(g2_sq + (2 * gpr_sq), 2))
    seldd_num = (1 / 8)\
        * ((mp.power(g2_sq + (2 * gpr_sq), 2) * v_sq * cos2b)
           + (4 * ((8 * mp.power(ae_wk, 2))
                   + ((g2_sq + (2 * gpr_sq)) * (mE1_sq_wk - mL1_sq_wk))
                   - (8 * ae_wk * ye_wk * mu_wk * mp.tan(beta_wk)))))
    sigmauu_se_1 = (loop_fac) * logfunc2(m_se_1sq) \
        * (((1 / 8) * (g2_sq - (6 * gpr_sq))) - (seluu_num / sel_denom))
    sigmauu_se_2 = (loop_fac) * logfunc2(m_se_2sq) \
        * (((1 / 8) * (g2_sq - (6 * gpr_sq))) + (seluu_num / sel_denom))
    sigmadd_se_1 = (loop_fac) * logfunc2(m_se_1sq) \
        * (mp.power(ye_wk, 2) - (g2_sq / 8) + ((3 / 4) * gpr_sq)
           - (seldd_num / sel_denom))
    sigmadd_se_2 = (loop_fac) * logfunc2(m_se_2sq) \
        * (mp.power(ye_wk, 2) - (g2_sq / 8) + ((3 / 4) * gpr_sq)
           + (seldd_num / sel_denom))
    # Electron sneutrino
    sigmauu_selec_sneut = (loop_fac / 8) * ((-1) * (g2_sq + gpr_sq))\
        * logfunc2(mselecneutsq)
    sigmadd_selec_sneut = (loop_fac / 8) * ((g2_sq + gpr_sq))\
        * logfunc2(mselecneutsq)


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
        msninp = mp.sqrt(abs(msnsq))
        # Introduce coefficients of characteristic equation for eigenvals.
        # Char. eqn. is of the form x^4 + ax^3 + bx^2 + cx + d = 0
        char_a = (-1) * (M1_wk + M2_wk)
        char_b = ((M1_wk * M2_wk) - (mp.power(mu_wk, 2))
                  - ((v_sq / 2) * (g2_sq + gpr_sq)))
        char_c = ((mp.power(mu_wk, 2) * (M1_wk + M2_wk))
                  - (mu_wk * vd * vu * (g2_sq + gpr_sq))
                  + ((v_sq / 2)
                     * ((g2_sq * M1_wk) + (gpr_sq * M2_wk))))
        myden = (4 * mp.power(msninp, 3)) + (3 * char_a
                                             * mp.power(msninp, 2))\
            + (2 * char_b * msninp) + char_c
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
        msninp = mp.sqrt(abs(msnsq))
        quadrterm = ((-1) / 2) * (gpr_sq + g2_sq)
        linterm = (1 / 2) * ((g2_sq * M1_wk) + (gpr_sq * M2_wk)
                             - (mu_wk * (1 / mp.tan(beta_wk))
                                * (g2_sq + gpr_sq)))
        constterm = (1 / 2) * (mu_wk / mp.tan(beta_wk))\
            * ((g2_sq * M1_wk) + (gpr_sq * M1_wk))
        mynum = (quadrterm * mp.power(msninp, 2))\
            + (linterm * msninp) + constterm
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
        msninp = mp.sqrt(abs(msnsq))
        quadrterm = ((-1) / 2) * (gpr_sq + g2_sq)
        linterm = (1 / 2) * ((g2_sq * M1_wk) + (gpr_sq * M2_wk)
                             - (mu_wk * (mp.tan(beta_wk))
                                * (g2_sq + gpr_sq)))
        constterm = (1 / 2) * (mu_wk * mp.tan(beta_wk))\
            * ((g2_sq * M1_wk) + (gpr_sq * M1_wk))
        mynum = (quadrterm * mp.power(msninp, 2))\
            + (linterm * msninp) + constterm
        return mynum

    def sigmauu_neutralino(msnsq):
        """
        Return one-loop correction Sigma_u^u(neutralino).

        Parameters
        ----------
        msnsq : Float.
            Neutralino squared mass.

        """
        sigma_uu_neutralino = (loop_fac * (-2) * mp.sqrt(msnsq)) \
            * ((neutralinouu_num(mp.fabs(msnsq))
                / neutralino_denom(mp.fabs(msnsq)))
               * logfunc2(mp.fabs(msnsq)))
        return sigma_uu_neutralino

    def sigmadd_neutralino(msnsq):
        """
        Return one-loop correction Sigma_d^d(neutralino).

        Parameters
        ----------
        msnsq : Float.
            Neutralino squared mass.

        """
        sigma_dd_neutralino = (loop_fac * (-2) * mp.sqrt(msnsq)) \
            * ((neutralinodd_num(mp.fabs(msnsq))
                / neutralino_denom(mp.fabs(msnsq)))
               * logfunc2(mp.fabs(msnsq)))
        return sigma_dd_neutralino

    #TODO: might need to address the fact that one of the neutrino masses might be negative? Could affect cubic and linear terms
    #in deriv. calc's
    ##### Radiative corrections from chargino sector #####
    charginouu_num = (1 / 2) * (mp.power(M2_wk, 2)
                                + mp.power(mu_wk, 2)
                                - (g2_sq * v_sq * cos2b)
                                + (2 * M2_wk * mu_wk / mp.tan(beta_wk)))
    charginodd_num = (1 / 2) * (mp.power(M2_wk, 2)
                                + mp.power(mu_wk, 2)
                                + (g2_sq * v_sq * cos2b)
                                + (2 * M2_wk * mu_wk * mp.tan(beta_wk)))
    chargino_den = mp.sqrt((mp.power(M2_wk + mu_wk, 2)
                            + (g2_sq * v_sq * (1 - sin2b)))
                           * (mp.power(M2_wk - mu_wk, 2)
                              + (g2_sq * v_sq * (1 + sin2b))))
    sigmauu_chargino1 = (-1) * ((g2_sq) * loop_fac / 2)\
        * ((1 / 2) - (charginouu_num / chargino_den)) * logfunc2(msC1sq)
    sigmauu_chargino2 = (-1) * ((g2_sq) * loop_fac / 2)\
        * ((1 / 2) + (charginouu_num / chargino_den)) * logfunc2(msC2sq)
    sigmadd_chargino1 = (-1) * ((g2_sq) * loop_fac / 2)\
        * ((1 / 2) - (charginodd_num / chargino_den)) * logfunc2(msC1sq)
    sigmadd_chargino2 = (-1) * ((g2_sq) * loop_fac / 2)\
        * ((1 / 2) + (charginodd_num / chargino_den)) * logfunc2(msC2sq)

    ##### Radiative corrections from Higgs bosons sector #####

    higgsuu_num = (v_sq * (g2_sq + gpr_sq)) + (2 * mA0sq
                                               * (2
                                                  + (4 * cos2b)
                                                  + mp.cos(4 * beta_wk)))
    higgsdd_num = (v_sq * (g2_sq + gpr_sq)) + (2 * mA0sq
                                               * (2
                                                  - (4 * cos2b)
                                                  + mp.cos(4 * beta_wk)))
    higgs_den = mp.sqrt((4 * mp.power(mA0sq, 2)) + (mp.power(v_sq
                                                             * (g2_sq
                                                                + gpr_sq),
                                                             2))
                        - (4 * (g2_sq + gpr_sq) * mA0sq * v_sq
                           * mp.cos(4 * beta_wk)))
    sigmauu_h0 = (loop_fac / 2) * logfunc2(mh0sq) * ((g2_sq + gpr_sq) / 4)\
        * (1 - (higgsuu_num / higgs_den))
    sigmauu_heavy_h0 = (loop_fac / 2) * logfunc2(mH0sq) * ((g2_sq + gpr_sq)
                                                           / 4)\
        * (1 + (higgsuu_num / higgs_den))
    sigmadd_h0 = (loop_fac / 2) * logfunc2(mh0sq) * ((g2_sq + gpr_sq) / 4)\
        * (1 - (higgsdd_num / higgs_den))
    sigmadd_heavy_h0 = (loop_fac / 2) * logfunc2(mH0sq) * ((g2_sq + gpr_sq)
                                                           / 4)\
        * (1 + (higgsdd_num / higgs_den))
    sigmauu_h_pm  = (g2_sq * loop_fac / 2) * logfunc2(mH_pmsq)
    sigmadd_h_pm = sigmauu_h_pm

    ##### Radiative corrections from weak vector bosons sector #####
    sigmauu_w_pm = (3 * g2_sq * loop_fac / 2) * logfunc2(m_w_sq)
    sigmadd_w_pm = sigmauu_w_pm
    sigmauu_z0 = (3 / 4) * loop_fac * (gpr_sq + g2_sq)\
        * logfunc2(mz_q_sq)
    sigmadd_z0 = sigmauu_z0

    ##### Radiative corrections from SM fermions sector #####
    sigmauu_top = (-6) * mp.power(yt_wk, 2) * loop_fac\
        * logfunc2(mymtsq)
    sigmadd_top = 0
    sigmauu_bottom = 0
    sigmadd_bottom = (-6) * mp.power(yb_wk, 2) * loop_fac\
        * logfunc2(mymbsq)
    sigmauu_tau = 0
    sigmadd_tau = (-2) * mp.power(ytau_wk, 2) * loop_fac\
        * logfunc2(mymtausq)
    sigmauu_charm = (-6) * mp.power(yc_wk, 2) * loop_fac\
        * logfunc2(mymcsq)
    sigmadd_charm = 0
    sigmauu_strange = 0
    sigmadd_strange = (-6) * mp.power(ys_wk, 2) * loop_fac\
        * logfunc2(mymssq)
    sigmauu_mu = 0
    sigmadd_mu = (-2) * mp.power(ymu_wk, 2) * loop_fac\
        * logfunc2(mymmusq)
    sigmauu_up = (-6) * mp.power(yu_wk, 2) * loop_fac\
        * logfunc2(mymusq)
    sigmadd_up = 0
    sigmauu_down = 0
    sigmadd_down = (-6) * mp.power(yd_wk, 2) * loop_fac\
        * logfunc2(mymdsq)
    sigmauu_elec = 0
    sigmadd_elec = (-2) * mp.power(ye_wk, 2) * loop_fac\
        * logfunc2(mymesq)

    ##### Radiative corrections from two-loop O(alpha_t alpha_s) sector #####
    # Corrections come from Dedes, Slavich paper, arXiv:hep-ph/0212132.
    # alpha_i = y_i^2 / (4 * pi)
    def sigmauu_2loop():
        def Deltafunc(x, y, z):
            mydelta = mp.power(x, 2) + mp.power(y, 2) + mp.power(z, 2)\
                - (2 * ((x * y) + (x * z) + (y * z)))
            return mydelta

        def Phifunc(x, y, z):
            if(x / z < 1 and y / z < 1):
                myu = x / z
                myv = y / z
                mylambda = mp.sqrt(mp.fabs(mp.power((1 - myu - myv), 2)
                                          - (4 * myu * myv)))
                myxp = 0.5 * (1 + myu - myv - mylambda)
                myxm = 0.5 * (1 - myu + myv - mylambda)
                myphi = (1 / mylambda) * ((2 * mp.log(mp.fabs(myxp))
                                           * mp.log(mp.fabs(myxm)))
                                          - (mp.log(mp.fabs(myu))
                                             * mp.log(mp.fabs(myv)))
                                          - (2 * (mpf(str(spence(1 - float(myxp))))
                                                  + mpf(str(spence(1 - float(myxm))))))
                                          + (mp.power(mp.pi, 2) / 3))
            elif(x / z > 1 and y / z < 1):
                myu = z / x
                myv = y / x
                mylambda = mp.sqrt(mp.fabs(mp.power((1 - myu - myv), 2)
                                          - (4 * myu * myv)))
                myxp = 0.5 * (1 + myu - myv - mylambda)
                myxm = 0.5 * (1 - myu + myv - mylambda)
                myphi = (z / x) * (1 / mylambda)\
                    * ((2 * mp.log(mp.fabs(myxp))
                        * mp.log(mp.fabs(myxm)))
                       - (mp.log(mp.fabs(myu))
                          * mp.log(mp.fabs(myv)))
                       - (2 * (mpf(str(spence(1 - float(myxp))))
                               + mpf(str(spence(1 - float(myxm))))))
                       + (mp.power(mp.pi, 2) / 3))
            elif(x/z > 1 and y/ z > 1 and x > y):
                myu = z / x
                myv = y / x
                mylambda = mp.sqrt(mp.fabs(mp.power((1 - myu - myv), 2)
                                          - (4 * myu * myv)))
                myxp = 0.5 * (1 + myu - myv - mylambda)
                myxm = 0.5 * (1 - myu + myv - mylambda)
                myphi = (z / x) * (1 / mylambda)\
                    * ((2 * mp.log(mp.fabs(myxp))
                        * mp.log(mp.fabs(myxm)))
                       - (mp.log(mp.fabs(myu))
                          * mp.log(mp.fabs(myv)))
                       - (2 * (mpf(str(spence(1 - float(myxp))))
                               + mpf(str(spence(1 - float(myxm))))))
                       + (mp.power(mp.pi, 2) / 3))
            elif(x / z < 1 and y / z > 1):
                myu = z / y
                myv = x / y
                mylambda = mp.sqrt(mp.fabs(mp.power((1 - myu - myv), 2)
                                          - (4 * myu * myv)))
                myxp = 0.5 * (1 + myu - myv - mylambda)
                myxm = 0.5 * (1 - myu + myv - mylambda)
                myphi = (z / y) * (1 / mylambda)\
                    * ((2 * mp.log(mp.fabs(myxp))
                        * mp.log(mp.fabs(myxm)))
                       - (mp.log(mp.fabs(myu))
                          * mp.log(mp.fabs(myv)))
                       - (2 * (mpf(str(spence(1 - float(myxp))))
                               + mpf(str(spence(1 - float(myxm))))))
                       + (mp.power(mp.pi, 2) / 3))
            elif (x / z > 1 and y / z > 1 and y > x):
                myu = z / y
                myv = x / y
                mylambda = mp.sqrt(mp.fabs(mp.power((1 - myu - myv), 2)
                                          - (4 * myu * myv)))
                myxp = 0.5 * (1 + myu - myv - mylambda)
                myxm = 0.5 * (1 - myu + myv - mylambda)
                myphi = (z / y) * (1 / mylambda)\
                    * ((2 * mp.log(mp.fabs(myxp))
                        * mp.log(mp.fabs(myxm)))
                       - (mp.log(mp.fabs(myu))
                          * mp.log(mp.fabs(myv)))
                       - (2 * (mpf(str(spence(1 - float(myxp))))
                               + mpf(str(spence(1 - float(myxm))))))
                       + (mp.power(mp.pi, 2) / 3))
            return myphi

        mst1sq = m_stop_1sq
        mst2sq = m_stop_2sq
        s2theta = 2 * mymt * ((at_wk / yt_wk) + (mu_wk / mp.tan(beta_wk)))\
            / (mst1sq - mst2sq)
        s2sqtheta = mp.power(s2theta, 2)
        c2sqtheta = 1 - s2sqtheta
        mglsq = mp.power(M3_wk, 2)
        myunits = mp.power(g3_wk, 2) * 4 * loop_fac_sq
        Q_renorm_sq = mp.power(myQ, 2)
        myF = myunits\
            * (((4 * M3_wk * mymt / s2theta) * (1 + (4 * c2sqtheta)))
               - (((2 * (mst1sq - mst2sq)) + (4 * M3_wk * mymt / s2theta))
                  * mp.log(mglsq / Q_renorm_sq)
                  * mp.log(mymtsq / Q_renorm_sq))
               - (2 * (4 - s2sqtheta) * (mst1sq - mst2sq))
               + ((((4 * mst1sq * mst2sq)
                    - s2sqtheta * mp.power((mst1sq + mst2sq), 2))
                   / (mst1sq - mst2sq))
                  * (mp.log(mp.fabs(mst1sq / Q_renorm_sq)))
                  * (mp.log(mst2sq / Q_renorm_sq)))
                 + ((((4 * (mglsq + mymtsq + (2 * mst1sq)))
                      - (s2sqtheta * ((3 * mst1sq) + mst2sq))
                      - ((16 * c2sqtheta * M3_wk * mymt * mst1sq)
                         / (s2theta * (mst1sq - mst2sq)))
                      - (4 * s2theta * M3_wk * mymt))
                     * mp.log(mp.fabs(mst1sq / Q_renorm_sq)))
                    + ((mst1sq / (mst1sq - mst2sq))
                       * ((s2sqtheta * (mst1sq + mst2sq))
                          - ((4 * mst1sq) - (2 * mst2sq)))
                       * mp.power(mp.log(mp.fabs(mst1sq / Q_renorm_sq)), 2))
                    + (2 * (mst1sq - mglsq - mymtsq
                            + (M3_wk * mymt * s2theta)
                            + ((2 * c2sqtheta * M3_wk * mymt * mst1sq)
                               / (s2theta * (mst1sq - mst2sq))))
                       * mp.log(mglsq * mymtsq
                                / (mp.power(Q_renorm_sq, 2)))
                       * mp.log(mp.fabs(mst1sq / Q_renorm_sq)))
                    + (((4 * M3_wk * mymt * c2sqtheta * (mymtsq - mglsq))
                        / (s2theta * (mst1sq - mst2sq)))
                       * mp.log(mymtsq / mglsq)
                       * mp.log(mp.fabs(mst1sq / Q_renorm_sq)))
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
                     * mp.log(mst2sq / Q_renorm_sq))
                    + ((mst2sq / (mst2sq - mst1sq))
                       * ((s2sqtheta * (mst2sq + mst1sq))
                          - ((4 * mst2sq) - (2 * mst1sq)))
                       * mp.power(mp.log(mst2sq / Q_renorm_sq), 2))
                    + (2 * (mst2sq - mglsq - mymtsq
                            - (M3_wk * mymt * s2theta)
                            + ((2 * c2sqtheta * M3_wk * mymt * mst2sq)
                               / (s2theta * (mst1sq - mst2sq))))
                       * mp.log(mglsq * mymtsq
                                / (mp.power(Q_renorm_sq, 2)))
                       * mp.log(mst2sq / Q_renorm_sq))
                    + (((4 * M3_wk * mymt * c2sqtheta * (mymtsq - mglsq))
                        / (s2theta * (mst1sq - mst2sq)))
                       * mp.log(mymtsq / mglsq)
                       * mp.log(mst2sq / Q_renorm_sq))
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
                                * (mp.power(mp.log(mymtsq / Q_renorm_sq), 2)
                                   - (2 * mp.log(mymtsq / Q_renorm_sq))))
               + (((4 * mglsq) - ((M3_wk * s2theta / mymt)
                                  * (mst1sq - mst2sq)))
                  * mp.log(mglsq / Q_renorm_sq) * mp.log(mymtsq / Q_renorm_sq))
               + (s2sqtheta * (mst1sq + mst2sq)
                  * mp.log(mp.fabs(mst1sq / Q_renorm_sq))
                  * mp.log(mst2sq / Q_renorm_sq))
               + ((((4 * (mglsq + mymtsq + (2 * mst1sq)))
                    + (s2sqtheta * (mst1sq - mst2sq))
                    - ((4 * M3_wk * s2theta / mymt) * (mymtsq + mst1sq)))
                   * mp.log(mp.fabs(mst1sq / Q_renorm_sq)))
                  + (((M3_wk * s2theta * ((5 * mymtsq) - mglsq + mst1sq)
                       / mymt)
                      - (2 * (mglsq + 2 * mymtsq)))
                     * mp.log(mymtsq / Q_renorm_sq)
                     * mp.log(mp.fabs(mst1sq / Q_renorm_sq)))
                  + (((M3_wk * s2theta * (mglsq - mymtsq + mst1sq) / mymt)
                      - (2 * mglsq))
                     * mp.log(mglsq / Q_renorm_sq)
                     * mp.log(mp.fabs(mst1sq / Q_renorm_sq)))
                  - ((2 + s2sqtheta) * mst1sq
                     * mp.power(mp.log(mp.fabs(mst1sq / Q_renorm_sq)), 2))
                  + (((2 * mglsq * (mglsq + mymtsq - mst1sq
                                    - (2 * M3_wk * mymt * s2theta)) / mst1sq)
                      + ((M3_wk * s2theta / (mymt * mst1sq))
                         * Deltafunc(mglsq, mymtsq, mst1sq)))
                     * Phifunc(mglsq, mymtsq, mst1sq)))
               + ((((4 * (mglsq + mymtsq + (2 * mst2sq)))
                    + (s2sqtheta * (mst2sq - mst1sq))
                    - (((-4) * M3_wk * s2theta / mymt) * (mymtsq + mst2sq)))
                   * mp.log(mst2sq / Q_renorm_sq))
                  + ((((-1) * M3_wk * s2theta * ((5 * mymtsq) - mglsq + mst2sq)
                       / mymt)
                      - (2 * (mglsq + 2 * mymtsq)))
                     * mp.log(mymtsq / Q_renorm_sq)
                     * mp.log(mst2sq / Q_renorm_sq))
                  + ((((-1) * M3_wk * s2theta * (mglsq - mymtsq + mst2sq)
                       / mymt)
                      - (2 * mglsq))
                     * mp.log(mglsq / Q_renorm_sq)
                     * mp.log(mst2sq / Q_renorm_sq))
                  - ((2 + s2sqtheta) * mst2sq
                     * mp.power(mp.log(mst2sq / Q_renorm_sq), 2))
                  + (((2 * mglsq
                       * (mglsq + mymtsq - mst2sq
                          + (2 * M3_wk * mymt * s2theta)) / mst2sq)
                      + ((M3_wk * (-1) * s2theta / (mymt * mst2sq))
                         * Deltafunc(mglsq, mymtsq, mst2sq)))
                     * Phifunc(mglsq, mymtsq, mst2sq))))
        mysigmauu_2loop = ((mymt * (at_wk / yt_wk) * s2theta * myF)
                           + 2 * mp.power(mymt, 2) * myG)\
            / (mp.power((vHiggs_wk), 2) * sinsqb)
        return mysigmauu_2loop

    def sigmadd_2loop():
        def Deltafunc(x,y,z):
            mydelta = mp.power(x, 2) + mp.power(y, 2) + mp.power(z, 2)\
                - (2 * ((x * y) + (x * z) + (y * z)))
            return mydelta

        def Phifunc(x, y, z):
            if(x / z < 1 and y / z < 1):
                myu = x / z
                myv = y / z
                mylambda = mp.sqrt(mp.fabs(mp.power((1 - myu - myv), 2)
                                          - (4 * myu * myv)))
                myxp = 0.5 * (1 + myu - myv - mylambda)
                myxm = 0.5 * (1 - myu + myv - mylambda)
                myphi = (1 / mylambda) * ((2 * mp.log(mp.fabs(myxp))
                                           * mp.log(mp.fabs(myxm)))
                                          - (mp.log(mp.fabs(myu))
                                             * mp.log(mp.fabs(myv)))
                                          - (2 * (mpf(str(spence(float(1 - myxp))))
                                                  + mpf(str(spence(float(1 - myxm))))))
                                          + (mp.power(mp.pi, 2) / 3))
            elif(x / z > 1 and y / z < 1):
                myu = z / x
                myv = y / x
                mylambda = mp.sqrt(mp.fabs(mp.power((1 - myu - myv), 2)
                                          - (4 * myu * myv)))
                myxp = 0.5 * (1 + myu - myv - mylambda)
                myxm = 0.5 * (1 - myu + myv - mylambda)
                myphi = (z / x) * (1 / mylambda)\
                    * ((2 * mp.log(mp.fabs(myxp))
                        * mp.log(mp.fabs(myxm)))
                       - (mp.log(mp.fabs(myu))
                          * mp.log(mp.fabs(myv)))
                       - (2 * (mpf(str(spence(float(1 - myxp))))
                               + mpf(str(spence(float(1 - myxm))))))
                       + (mp.power(mp.pi, 2) / 3))
            elif(x/z > 1 and y/ z > 1 and x > y):
                myu = z / x
                myv = y / x
                mylambda = mp.sqrt(mp.fabs(mp.power((1 - myu - myv), 2)
                                          - (4 * myu * myv)))
                myxp = 0.5 * (1 + myu - myv - mylambda)
                myxm = 0.5 * (1 - myu + myv - mylambda)
                myphi = (z / x) * (1 / mylambda)\
                    * ((2 * mp.log(mp.fabs(myxp))
                        * mp.log(mp.fabs(myxm)))
                       - (mp.log(mp.fabs(myu))
                          * mp.log(mp.fabs(myv)))
                       - (2 * (mpf(str(spence(float(1 - myxp))))
                               + mpf(str(spence(float(1 - myxm))))))
                       + (mp.power(mp.pi, 2) / 3))
            elif(x / z < 1 and y / z > 1):
                myu = z / y
                myv = x / y
                mylambda = mp.sqrt(mp.fabs(mp.power((1 - myu - myv), 2)
                                          - (4 * myu * myv)))
                myxp = 0.5 * (1 + myu - myv - mylambda)
                myxm = 0.5 * (1 - myu + myv - mylambda)
                myphi = (z / y) * (1 / mylambda)\
                    * ((2 * mp.log(mp.fabs(myxp))
                        * mp.log(mp.fabs(myxm)))
                       - (mp.log(mp.fabs(myu))
                          * mp.log(mp.fabs(myv)))
                       - (2 * (mpf(str(spence(float(1 - myxp))))
                               + mpf(str(spence(float(1 - myxm))))))
                       + (mp.power(mp.pi, 2) / 3))
            elif (x / z > 1 and y / z > 1 and y > x):
                myu = z / y
                myv = x / y
                mylambda = mp.sqrt(mp.fabs(mp.power((1 - myu - myv), 2)
                                          - (4 * myu * myv)))
                myxp = 0.5 * (1 + myu - myv - mylambda)
                myxm = 0.5 * (1 - myu + myv - mylambda)
                myphi = (z / y) * (1 / mylambda)\
                    * ((2 * mp.log(mp.fabs(myxp))
                        * mp.log(mp.fabs(myxm)))
                       - (mp.log(mp.fabs(myu))
                          * mp.log(mp.fabs(myv)))
                       - (2 * (mpf(str(spence(float(1 - myxp))))
                               + mpf(str(spence(float(1 - myxm))))))
                       + (mp.power(mp.pi, 2) / 3))
            return myphi

        mst1sq = m_stop_1sq
        mst2sq = m_stop_2sq
        Q_renorm_sq=mp.power(myQ, 2)
        s2theta = (2 * mymt * ((at_wk / yt_wk)
                               + (mu_wk / mp.tan(beta_wk))))\
            / (mst1sq - mst2sq)
        s2sqtheta = mp.power(s2theta, 2)
        c2sqtheta = 1 - s2sqtheta
        mglsq = mp.power(M3_wk, 2)
        myunits = mp.power(g3_wk, 2) * 4\
            / mp.power((16 * mp.power(mp.pi, 2)), 2)
        myF = myunits\
            * ((4 * M3_wk * mymt / s2theta) * (1 + 4 * c2sqtheta)
               - (((2 * (mst1sq - mst2sq))
                  + (4 * M3_wk * mymt / s2theta))
                  * mp.log(mglsq / Q_renorm_sq)
                  * mp.log(mymtsq / Q_renorm_sq))
               - (2 * (4 - s2sqtheta)
                  * (mst1sq - mst2sq))
               + ((((4 * mst1sq * mst2sq)
                    - s2sqtheta * mp.power((mst1sq + mst2sq), 2))
                   / (mst1sq - mst2sq))
                  * (mp.log(mp.fabs(mst1sq / Q_renorm_sq)))
                  * (mp.log(mst2sq / Q_renorm_sq)))
               + ((((4 * (mglsq + mymtsq + (2 * mst1sq)))
                   - (s2sqtheta * ((3 * mst1sq) + mst2sq))
                   - ((16 * c2sqtheta * M3_wk * mymt * mst1sq)
                      / (s2theta * (mst1sq - mst2sq)))
                   - (4 * s2theta * M3_wk * mymt))
                   * mp.log(mp.fabs(mst1sq / Q_renorm_sq)))
                  + ((mst1sq / (mst1sq - mst2sq))
                     * ((s2sqtheta * (mst1sq + mst2sq))
                        - ((4 * mst1sq) - (2 * mst2sq)))
                     * mp.power(mp.log(mp.fabs(mst1sq / Q_renorm_sq)), 2))
                  + (2 * (mst1sq - mglsq - mymtsq
                          + (M3_wk * mymt * s2theta)
                          + ((2 * c2sqtheta * M3_wk * mymt * mst1sq)
                             / (s2theta * (mst1sq - mst2sq))))
                     * mp.log(mglsq * mymtsq
                              / (mp.power(Q_renorm_sq, 2)))
                     * mp.log(mp.fabs(mst1sq / Q_renorm_sq)))
                  + (((4 * M3_wk * mymt * c2sqtheta * (mymtsq - mglsq))
                      / (s2theta * (mst1sq - mst2sq)))
                     * mp.log(mymtsq / mglsq)
                     * mp.log(mp.fabs(mst1sq / Q_renorm_sq)))
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
                   * mp.log(mst2sq / Q_renorm_sq))
                  + ((mst2sq / (mst2sq - mst1sq))
                     * ((s2sqtheta * (mst2sq + mst1sq))
                        - ((4 * mst2sq) - (2 * mst1sq)))
                     * mp.power(mp.log(mst2sq / Q_renorm_sq), 2))
                  + (2 * (mst2sq - mglsq - mymtsq
                          - (M3_wk * mymt * s2theta)
                          + ((2 * c2sqtheta * M3_wk * mymt * mst2sq)
                             / (s2theta * (mst1sq - mst2sq))))
                     * mp.log(mglsq * mymtsq
                              / (mp.power(Q_renorm_sq, 2)))
                     * mp.log(mst2sq / Q_renorm_sq))
                  + (((4 * M3_wk * mymt * c2sqtheta * (mymtsq - mglsq))
                      / (s2theta * (mst1sq - mst2sq)))
                     * mp.log(mymtsq / mglsq)
                     * mp.log(mst2sq / Q_renorm_sq))
                  + (((((4 * mglsq * mymtsq)
                        + (2 * Deltafunc(mglsq, mymtsq, mst2sq))) / mst2sq)
                      - ((((-2) * M3_wk * mymt * s2theta) / mst2sq)
                         * (mglsq + mymtsq - mst2sq))
                      + ((4 * c2sqtheta * M3_wk * mymt
                          * Deltafunc(mglsq, mymtsq, mst2sq))
                         / (s2theta * mst2sq * (mst1sq - mst2sq))))
                     * Phifunc(mglsq, mymtsq, mst2sq))))
        mysigmadd_2loop = (mymt * mu_wk * (1 / mp.tan(beta_wk))
                           * s2theta * myF)\
            / (mp.power((vHiggs_wk), 2) * cossqb)
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

    # Return list of radiative corrections
    # (0: sigmauu_tot, 1: sigmadd_tot, 2: sigmauu_stop_1,
    #  3: sigmadd_stop_1, 4: sigmauu_stop_2,
    #  5: sigmadd_stop_2, 6: sigmauu_sbot_1,
    #  7: sigmadd_sbot_1, 8: sigmauu_sbot_2,
    #  9: sigmadd_sbot_2, 10: sigmauu_stau_1,
    #  11: sigmadd_stau_1, 12: sigmauu_stau_2,
    #  13: sigmadd_stau_2, 14: sigmauu_stau_sneut,
    #  15: sigmadd_stau_sneut, 16: sigmauu_scharm_1,
    #  17: sigmadd_scharm_1, 18: sigmauu_scharm_2,
    #  19: sigmadd_scharm_2, 20: sigmauu_sstrange_1,
    #  21: sigmadd_sstrange_1, 22: sigmauu_sstrange_2,
    #  23: sigmadd_sstrange_2, 24: sigmauu_smu_1,
    #  25: sigmadd_smu_1, 26: sigmauu_smu_2,
    #  27: sigmadd_smu_2, 28: sigmauu_smu_sneut,
    #  29: sigmadd_smu_sneut, 30: sigmauu_sup_1,
    #  31: sigmadd_sup_1, 32: sigmauu_sup_2,
    #  33: sigmadd_sup_2, 34: sigmauu_sdown_1,
    #  35: sigmadd_sdown_1, 36: sigmauu_sdown_2,
    #  37: sigmadd_sdown_2, 38: sigmauu_se_1,
    #  39: sigmadd_se_1, 40: sigmauu_se_2, 41: sigmadd_se_2,
    #  42: sigmauu_selec_sneut, 43: sigmadd_selec_sneut,
    #  44: sigmauu_neutralino(msN1sq),
    #  45: sigmadd_neutralino(msN1sq),
    #  46: sigmauu_neutralino(msN2sq), 47: sigmadd_neutralino(msN2sq),
    #  48: sigmauu_neutralino(msN3sq),
    #  49: sigmadd_neutralino(msN3sq),
    #  50: sigmauu_neutralino(msN4sq), 51: sigmadd_neutralino(msN4sq),
    #  52: sigmauu_chargino1,
    #  53: sigmadd_chargino1, 54: sigmauu_chargino2,
    #  55: sigmadd_chargino2, 56: sigmauu_h0,
    #  57: sigmadd_h0, 58: sigmauu_heavy_h0,
    #  59: sigmadd_heavy_h0, 60: sigmauu_h_pm,
    #  61: sigmadd_h_pm, 62: sigmauu_w_pm, 63: sigmadd_w_pm,
    #  64: sigmauu_z0, 65: sigmadd_z0,
    #  66: sigmauu_top, 67: sigmadd_top,
    #  68: sigmauu_bottom, 69: sigmadd_bottom,
    #  70: sigmauu_tau, 71: sigmadd_tau,
    #  72: sigmauu_charm, 73: sigmadd_charm,
    #  74: sigmauu_strange, 75: sigmadd_strange,
    #  76: sigmauu_mu, 77: sigmadd_mu, 78: sigmauu_up,
    #  79: sigmadd_up, 80: sigmauu_down, 81: sigmadd_down,
    #  82: sigmauu_elec, 83: sigmadd_elec,
    #  84: sigmauu_2loop(), 85: sigmadd_2loop())
    return [sigmauu_tot, sigmadd_tot, sigmauu_stop_1,
            sigmadd_stop_1, sigmauu_stop_2, sigmadd_stop_2,
            sigmauu_sbot_1, sigmadd_sbot_1,
            sigmauu_sbot_2, sigmadd_sbot_2, sigmauu_stau_1,
            sigmadd_stau_1, sigmauu_stau_2, sigmadd_stau_2,
            sigmauu_stau_sneut, sigmadd_stau_sneut,
            sigmauu_scharm_1, sigmadd_scharm_1,
            sigmauu_scharm_2, sigmadd_scharm_2,
            sigmauu_sstrange_1, sigmadd_sstrange_1,
            sigmauu_sstrange_2, sigmadd_sstrange_2,
            sigmauu_smu_1, sigmadd_smu_1,
            sigmauu_smu_2, sigmadd_smu_2, sigmauu_smu_sneut,
            sigmadd_smu_sneut, sigmauu_sup_1, sigmadd_sup_1,
            sigmauu_sup_2, sigmadd_sup_2,
            sigmauu_sdown_1, sigmadd_sdown_1, sigmauu_sdown_2,
            sigmadd_sdown_2, sigmauu_se_1, sigmadd_se_1,
            sigmauu_se_2, sigmadd_se_2,
            sigmauu_selec_sneut, sigmadd_selec_sneut,
            sigmauu_neutralino(msN1sq), sigmadd_neutralino(msN1sq),
            sigmauu_neutralino(msN2sq),
            sigmadd_neutralino(msN2sq),
            sigmauu_neutralino(msN3sq), sigmadd_neutralino(msN3sq),
            sigmauu_neutralino(msN4sq),
            sigmadd_neutralino(msN4sq),
            sigmauu_chargino1, sigmadd_chargino1,
            sigmauu_chargino2, sigmadd_chargino2,
            sigmauu_h0, sigmadd_h0, sigmauu_heavy_h0,
            sigmadd_heavy_h0, sigmauu_h_pm, sigmadd_h_pm,
            sigmauu_w_pm, sigmadd_w_pm,
            sigmauu_z0, sigmadd_z0, sigmauu_top, sigmadd_top,
            sigmauu_bottom, sigmadd_bottom,
            sigmauu_tau, sigmadd_tau, sigmauu_charm,
            sigmadd_charm, sigmauu_strange, sigmadd_strange,
            sigmauu_mu, sigmadd_mu,
            sigmauu_up, sigmadd_up, sigmauu_down, sigmadd_down,
            sigmauu_elec, sigmadd_elec,
            sigmauu_2loop(), sigmadd_2loop()]
