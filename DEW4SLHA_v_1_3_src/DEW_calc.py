#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 13:35:06 2023

Delta_EW calculator.

@author: Dakotah Martinez
"""

from copy import deepcopy
from radcorr_calc import my_radcorr_calc
import numpy as np

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

    radcorrtempinps = deepcopy([myQ, vHiggs_wk, mu_wk, beta_wk, yt_wk, yc_wk,
                                yu_wk, yb_wk, ys_wk, yd_wk, ytau_wk, ymu_wk,
                                ye_wk, g1_wk, g2_wk, g3_wk, mQ3_sq_wk,
                                mQ2_sq_wk, mQ1_sq_wk, mL3_sq_wk, mL2_sq_wk,
                                mL1_sq_wk, mU3_sq_wk, mU2_sq_wk, mU1_sq_wk,
                                mD3_sq_wk, mD2_sq_wk, mD1_sq_wk, mE3_sq_wk,
                                mE2_sq_wk, mE1_sq_wk, M1_wk, M2_wk, M3_wk,
                                mHu_sq_wk, mHd_sq_wk, at_wk, ac_wk, au_wk,
                                ab_wk, as_wk, ad_wk, atau_wk, amu_wk, ae_wk])
    radcorrtempinps2 = deepcopy([myQ, vHiggs_wk, mu_wk, beta_wk, yt_wk, yc_wk,
                                 yu_wk, yb_wk, ys_wk, yd_wk, ytau_wk, ymu_wk,
                                 ye_wk, g1_wk, g2_wk, g3_wk, mQ3_sq_wk,
                                 mQ2_sq_wk, mQ1_sq_wk, mL3_sq_wk, mL2_sq_wk,
                                 mL1_sq_wk, mU3_sq_wk, mU2_sq_wk, mU1_sq_wk,
                                 mD3_sq_wk, mD2_sq_wk, mD1_sq_wk, mE3_sq_wk,
                                 mE2_sq_wk, mE1_sq_wk, M1_wk, M2_wk, M3_wk,
                                 mHu_sq_wk, mHd_sq_wk, at_wk, ac_wk, au_wk,
                                 ab_wk, as_wk, ad_wk, atau_wk, amu_wk, ae_wk])
    myradcorrs = my_radcorr_calc(radcorrtempinps[0],radcorrtempinps[1],
                                 radcorrtempinps[2],radcorrtempinps[3],
                                 radcorrtempinps[4],radcorrtempinps[5],
                                 radcorrtempinps[6],radcorrtempinps[7],
                                 radcorrtempinps[8],radcorrtempinps[9],
                                 radcorrtempinps[10],radcorrtempinps[11],
                                 radcorrtempinps[12],radcorrtempinps[13],
                                 radcorrtempinps[14],radcorrtempinps[15],
                                 radcorrtempinps[16],radcorrtempinps[17],
                                 radcorrtempinps[18],radcorrtempinps[19],
                                 radcorrtempinps[20],radcorrtempinps[21],
                                 radcorrtempinps[22],radcorrtempinps[23],
                                 radcorrtempinps[24],radcorrtempinps[25],
                                 radcorrtempinps[26],radcorrtempinps[27],
                                 radcorrtempinps[28],radcorrtempinps[29],
                                 radcorrtempinps[30],radcorrtempinps[31],
                                 radcorrtempinps[32],radcorrtempinps[33],
                                 radcorrtempinps[34],radcorrtempinps[35],
                                 radcorrtempinps[36],radcorrtempinps[37],
                                 radcorrtempinps[38],radcorrtempinps[39],
                                 radcorrtempinps[40],radcorrtempinps[41],
                                 radcorrtempinps[42],radcorrtempinps[43],
                                 radcorrtempinps[44])

    # DEW contribution computation: #

    def dew_funcu(inp):
        """
        Compute individual one-loop DEW contributions from Sigma_u^u.

        Parameters
        ----------
        inp : One-loop correction or Higgs to be inputted into the DEW eval.

        """
        mycontribuu = (inp / 2) * ((1 / np.cos(2 * beta_wk)) - 1)
        return mycontribuu

    def dew_funcd(inp):
        """
        Compute individual one-loop DEW contributions from Sigma_d^d.

        Parameters
        ----------
        inp : One-loop correction or Higgs to be inputted into the DEW eval.

        """
        mycontribdd = (inp / 2) * ((-1 / np.cos(2 * beta_wk)) - 1)
        return mycontribdd

    running_mZ_sq = np.power(vHiggs_wk, 2) * ((np.power(g2_wk, 2)
                                               + ((3 / 5)
                                                  * np.power(g1_wk, 2))) / 2)
    cmu = (-1) * np.power(mu_wk, 2)
    cHu = dew_funcu(mHu_sq_wk)
    cHd = dew_funcd(mHd_sq_wk)
    contribs = np.array([cmu, cHu, cHd, dew_funcu(myradcorrs[2]), # 0: C_mu, 1: C_Hu, 2: C_Hd, 3: C_sigmauu_stop_1
                         dew_funcd(myradcorrs[3]), dew_funcu(myradcorrs[4]), # 4: C_sigmadd_stop_1, 5: C_sigmauu_stop_2
                         dew_funcd(myradcorrs[5]), dew_funcu(myradcorrs[6]), # 6: C_sigmadd_stop_2, 7: C_sigmauu_sbot_1
                         dew_funcd(myradcorrs[7]), dew_funcu(myradcorrs[8]), # 8: C_sigmadd_sbot_1, 9: C_sigmauu_sbot_2
                         dew_funcd(myradcorrs[9]), dew_funcu(myradcorrs[10]), # 10: C_sigmadd_sbot_2, 11: C_sigmauu_stau_1
                         dew_funcd(myradcorrs[11]), dew_funcu(myradcorrs[12]), # 12: C_sigmadd_stau_1, 13: C_sigmauu_stau_2
                         dew_funcd(myradcorrs[13]), dew_funcu(myradcorrs[14]), # 14: C_sigmadd_stau_2, 15: C_sigmauu_stau_sneut
                         dew_funcd(myradcorrs[15]), dew_funcu(myradcorrs[16]), # 16: C_sigmadd_stau_sneut, 17: C_sigmauu_scharm_1
                         dew_funcd(myradcorrs[17]), dew_funcu(myradcorrs[18]), # 18: C_sigmadd_scharm_1, 19: C_sigmauu_scharm_2
                         dew_funcd(myradcorrs[19]), dew_funcu(myradcorrs[20]), # 20: C_sigmadd_scharm_2, 21: C_sigmauu_sstrange_1
                         dew_funcd(myradcorrs[21]), dew_funcu(myradcorrs[22]), # 22: C_sigmadd_sstrange_1, 23: C_sigmauu_sstrange_2
                         dew_funcd(myradcorrs[23]), dew_funcu(myradcorrs[24]), # 24: C_sigmadd_sstrange_2, 25: C_sigmauu_smu_1
                         dew_funcd(myradcorrs[25]), dew_funcu(myradcorrs[26]), # 26: C_sigmadd_smu_1, 27: C_sigmauu_smu_2
                         dew_funcd(myradcorrs[27]), dew_funcu(myradcorrs[28]), # 28: C_sigmadd_smu_2, 29: C_sigmauu_smu_sneut
                         dew_funcd(myradcorrs[29]), dew_funcu(myradcorrs[30]), # 30: C_sigmadd_smu_sneut, 31: C_sigmauu_sup_1
                         dew_funcd(myradcorrs[31]), dew_funcu(myradcorrs[32]), # 32: C_sigmadd_sup_1, 33: C_sigmauu_sup_2
                         dew_funcd(myradcorrs[33]), dew_funcu(myradcorrs[34]), # 34: C_sigmadd_sup_2, 35: C_sigmauu_sdown_1
                         dew_funcd(myradcorrs[35]), dew_funcu(myradcorrs[36]), # 36: C_sigmadd_sdown_1, 37: C_sigmauu_sdown_2
                         dew_funcd(myradcorrs[37]), dew_funcu(myradcorrs[38]), # 38: C_sigmadd_sdown_2, 39: C_sigmauu_se_1
                         dew_funcd(myradcorrs[39]), dew_funcu(myradcorrs[40]), # 40: C_sigmadd_se_1, 41: C_sigmauu_se_2
                         dew_funcd(myradcorrs[41]), dew_funcu(myradcorrs[42]), # 42: C_sigmadd_se_2, 43: C_sigmauu_selec_sneut
                         dew_funcd(myradcorrs[43]), dew_funcu(myradcorrs[44]), # 44: C_sigmadd_selec_sneut, 45: C_sigmauu_neutralino1
                         dew_funcd(myradcorrs[45]), dew_funcu(myradcorrs[46]), # 46: C_sigmadd_neutralino1, 47: C_sigmauu_neutralino2
                         dew_funcd(myradcorrs[47]), dew_funcu(myradcorrs[48]), # 48: C_sigmadd_neutralino2, 49: C_sigmauu_neutralino3
                         dew_funcd(myradcorrs[49]), dew_funcu(myradcorrs[50]), # 50: C_sigmadd_neutralino3, 51: C_sigmauu_neutralino4
                         dew_funcd(myradcorrs[51]), dew_funcu(myradcorrs[52]), # 52: C_sigmadd_neutralino4, 53: C_sigmauu_chargino1
                         dew_funcd(myradcorrs[53]), dew_funcu(myradcorrs[54]), # 54: C_sigmadd_chargino1, 55: C_sigmauu_chargino2
                         dew_funcd(myradcorrs[55]), dew_funcu(myradcorrs[56]), # 56: C_sigmadd_chargino2, 57: C_sigmauu_h0
                         dew_funcd(myradcorrs[57]), dew_funcu(myradcorrs[58]), # 58: C_sigmadd_h0, 59: C_sigmauu_heavy_h0
                         dew_funcd(myradcorrs[59]), dew_funcu(myradcorrs[60]), # 60: C_sigmadd_heavy_h0, 61: C_sigmauu_h_pm
                         dew_funcd(myradcorrs[61]), dew_funcu(myradcorrs[62]), # 62: C_sigmadd_h_pm, 63: C_sigmauu_w_pm
                         dew_funcd(myradcorrs[63]), dew_funcu(myradcorrs[64]), # 64: C_sigmadd_w_pm, 65: C_sigmauu_z0
                         dew_funcd(myradcorrs[65]), # 66: C_sigmadd_Z0
                         dew_funcu(myradcorrs[66])
                         + dew_funcu(myradcorrs[68])
                         + dew_funcu(myradcorrs[70])
                         + dew_funcu(myradcorrs[72])
                         + dew_funcu(myradcorrs[74])
                         + dew_funcu(myradcorrs[76])
                         + dew_funcu(myradcorrs[78])
                         + dew_funcu(myradcorrs[80])
                         + dew_funcu(myradcorrs[82]), # 67: C_sigmauu_SM
                         dew_funcd(myradcorrs[67])
                         + dew_funcd(myradcorrs[69])
                         + dew_funcd(myradcorrs[71])
                         + dew_funcd(myradcorrs[73])
                         + dew_funcd(myradcorrs[75])
                         + dew_funcd(myradcorrs[77])
                         + dew_funcd(myradcorrs[79])
                         + dew_funcd(myradcorrs[81])
                         + dew_funcd(myradcorrs[83]), # 68: C_sigmadd_SM
                         dew_funcu(myradcorrs[84]), # 69: C_sigmauu_2loop
                         dew_funcd(myradcorrs[85])] # 70: C_sigmadd_2loop
                        ) / ((91.1876 ** 2) / 2)

    label_sort_array = np.sort(np.array([(contribs[0], np.abs(contribs[0]),
                                          'Delta_EW(mu)'),
                                         (contribs[1], np.abs(contribs[1]),
                                          'Delta_EW(H_u)'),
                                         (contribs[2], np.abs(contribs[2]),
                                          'Delta_EW(H_d)'),
                                         (contribs[3], np.abs(contribs[3]),
                                          'Delta_EW(Sigma_u^u(stop_1))'),
                                         (contribs[4], np.abs(contribs[4]),
                                          'Delta_EW(Sigma_d^d(stop_1))'),
                                         (contribs[5], np.abs(contribs[5]),
                                          'Delta_EW(Sigma_u^u(stop_2))'),
                                         (contribs[6], np.abs(contribs[6]),
                                          'Delta_EW(Sigma_d^d(stop_2))'),
                                         (contribs[7], np.abs(contribs[7]),
                                          'Delta_EW(Sigma_u^u(sbot_1))'),
                                         (contribs[8], np.abs(contribs[8]),
                                          'Delta_EW(Sigma_d^d(sbot_1))'),
                                         (contribs[9], np.abs(contribs[9]),
                                          'Delta_EW(Sigma_u^u(sbot_2))'),
                                         (contribs[10], np.abs(contribs[10]),
                                          'Delta_EW(Sigma_d^d(sbot_2))'),
                                         (contribs[11], np.abs(contribs[11]),
                                          'Delta_EW(Sigma_u^u(stau_1))'),
                                         (contribs[12], np.abs(contribs[12]),
                                          'Delta_EW(Sigma_d^d(stau_1))'),
                                         (contribs[13], np.abs(contribs[13]),
                                          'Delta_EW(Sigma_u^u(stau_2))'),
                                         (contribs[14], np.abs(contribs[14]),
                                          'Delta_EW(Sigma_d^d(stau_2))'),
                                         (contribs[15], np.abs(contribs[15]),
                                          'Delta_EW(Sigma_u^u(tau sneutrino))'),
                                         (contribs[16], np.abs(contribs[16]),
                                          'Delta_EW(Sigma_d^d(tau sneutrino))'),
                                         (contribs[17] + contribs[19]
                                          + contribs[21] + contribs[23],
                                          np.abs(contribs[17] + contribs[19]
                                                 + contribs[21]
                                                 + contribs[23]),
                                          'Delta_EW(Sigma_u^u(sum 2nd gen. squarks))'),
                                         (contribs[18] + contribs[20]
                                          + contribs[22] + contribs[24],
                                          np.abs(contribs[18] + contribs[20]
                                                 + contribs[22]
                                                 + contribs[24]),
                                          'Delta_EW(Sigma_d^d(sum 2nd gen. squarks))'),
                                         (contribs[25], np.abs(contribs[25]),
                                          'Delta_EW(Sigma_u^u(smuon_1))'),
                                         (contribs[26], np.abs(contribs[26]),
                                          'Delta_EW(Sigma_d^d(smuon_1))'),
                                         (contribs[27], np.abs(contribs[27]),
                                          'Delta_EW(Sigma_u^u(smuon_2))'),
                                         (contribs[28], np.abs(contribs[28]),
                                          'Delta_EW(Sigma_d^d(smuon_2))'),
                                         (contribs[29], np.abs(contribs[29]),
                                          'Delta_EW(Sigma_u^u(muon sneutrino))'),
                                         (contribs[30], np.abs(contribs[30]),
                                          'Delta_EW(Sigma_d^d(muon sneutrino))'),
                                         (contribs[31] + contribs[33]
                                          + contribs[35] + contribs[37],
                                          np.abs(contribs[31] + contribs[33]
                                                 + contribs[35]
                                                 + contribs[37]),
                                          'Delta_EW(Sigma_u^u(sum 1st gen. squarks))'),
                                         (contribs[32] + contribs[34]
                                          + contribs[36] + contribs[38],
                                          np.abs(contribs[32] + contribs[34]
                                                 + contribs[36]
                                                 + contribs[38]),
                                          'Delta_EW(Sigma_d^d(sum 1st gen. squarks))'),
                                         (contribs[39], np.abs(contribs[39]),
                                          'Delta_EW(Sigma_u^u(selectron_1))'),
                                         (contribs[40], np.abs(contribs[40]),
                                          'Delta_EW(Sigma_d^d(selectron_1))'),
                                         (contribs[41], np.abs(contribs[41]),
                                          'Delta_EW(Sigma_u^u(selectron_2))'),
                                         (contribs[42], np.abs(contribs[42]),
                                          'Delta_EW(Sigma_d^d(selectron_2))'),
                                         (contribs[43], np.abs(contribs[43]),
                                          'Delta_EW(Sigma_u^u(electron sneutrino))'),
                                         (contribs[44], np.abs(contribs[44]),
                                          'Delta_EW(Sigma_d^d(electron sneutrino))'),
                                         (contribs[45], np.abs(contribs[45]),
                                          'Delta_EW(Sigma_u^u(neutralino_1))'),
                                         (contribs[46], np.abs(contribs[46]),
                                          'Delta_EW(Sigma_d^d(neutralino_1))'),
                                         (contribs[47], np.abs(contribs[47]),
                                          'Delta_EW(Sigma_u^u(neutralino_2))'),
                                         (contribs[48], np.abs(contribs[48]),
                                          'Delta_EW(Sigma_d^d(neutralino_2))'),
                                         (contribs[49], np.abs(contribs[49]),
                                          'Delta_EW(Sigma_u^u(neutralino_3))'),
                                         (contribs[50], np.abs(contribs[50]),
                                          'Delta_EW(Sigma_d^d(neutralino_3))'),
                                         (contribs[51], np.abs(contribs[51]),
                                          'Delta_EW(Sigma_u^u(neutralino_4))'),
                                         (contribs[52], np.abs(contribs[52]),
                                          'Delta_EW(Sigma_d^d(neutralino_4))'),
                                         (contribs[53], np.abs(contribs[53]),
                                          'Delta_EW(Sigma_u^u(chargino_1))'),
                                         (contribs[54], np.abs(contribs[54]),
                                          'Delta_EW(Sigma_d^d(chargino_1))'),
                                         (contribs[55], np.abs(contribs[55]),
                                          'Delta_EW(Sigma_u^u(chargino_2))'),
                                         (contribs[56], np.abs(contribs[56]),
                                          'Delta_EW(Sigma_d^d(chargino_2))'),
                                         (contribs[57], np.abs(contribs[57]),
                                          'Delta_EW(Sigma_u^u(h_0))'),
                                         (contribs[58], np.abs(contribs[58]),
                                          'Delta_EW(Sigma_d^d(h_0))'),
                                         (contribs[59], np.abs(contribs[59]),
                                          'Delta_EW(Sigma_u^u(H_0))'),
                                         (contribs[60], np.abs(contribs[60]),
                                          'Delta_EW(Sigma_d^d(H_0))'),
                                         (contribs[61], np.abs(contribs[61]),
                                          'Delta_EW(Sigma_u^u(H_+-))'),
                                         (contribs[62], np.abs(contribs[62]),
                                          'Delta_EW(Sigma_d^d(H_+-))'),
                                         (contribs[63], np.abs(contribs[63]),
                                          'Delta_EW(Sigma_u^u(W_+-))'),
                                         (contribs[64], np.abs(contribs[64]),
                                          'Delta_EW(Sigma_d^d(W_+-))'),
                                         (contribs[65], np.abs(contribs[65]),
                                          'Delta_EW(Sigma_u^u(Z_0))'),
                                         (contribs[66], np.abs(contribs[66]),
                                          'Delta_EW(Sigma_d^d(Z_0))'),
                                         (contribs[67], np.abs(contribs[67]),
                                          'Delta_EW(Sigma_u^u(SM fermions))'),
                                         (contribs[68], np.abs(contribs[68]),
                                          'Delta_EW(Sigma_d^d(SM fermions))'),
                                         (contribs[69], np.abs(contribs[69]),
                                          'Delta_EW(Sigma_u^u(O(alpha_s alpha_t)))'),
                                         (contribs[70], np.abs(contribs[70]),
                                          'Delta_EW(Sigma_d^d(O(alpha_s alpha_t)))')],
                                        dtype=[('Contrib', float),
                                               ('AbsContrib', float),
                                               ('label', 'U60')]),
                               order='AbsContrib')
    reverse_sort_array = label_sort_array[::-1]
    return reverse_sort_array
