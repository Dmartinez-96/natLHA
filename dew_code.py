# -*- coding: utf-8 -*-
"""Compute naturalness measure Delta_EW and top ten contributions to DEW."""

import numpy as np
import pyslha

#########################
# Mass relations:
#########################


def m_w_sq(g_coupling, v_higgs):
    """
    Return W boson squared mass.

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    v_higgs : Higgs VEV.

    Returns
    -------
    mymWsq : W boson squared mass.

    """
    my_mw_sq = (np.power(g_coupling, 2) / 2) * np.power(v_higgs, 2)
    return my_mw_sq


def mzsq(g_coupling, g_prime, v_higgs):
    """
    Return Z boson squared mass.

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    v_higgs : Higgs VEV.

    Returns
    -------
    mymZsq : Z boson squared mass.

    """
    my_mz_sq = ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2) * \
        np.power(v_higgs, 2)
    return my_mz_sq


def ma_0sq(mu_soft, mh_usq, mh_dsq):
    """
    Return A_0 squared mass.

    Parameters
    ----------
    mu_soft : SUSY Higgs mass parameter, mu.
    mHusq : Squared up-type Higgs mass.
    mHdsq : Squared down-type Higgs mass.

    Returns
    -------
    mymA0sq : A_0 squared mass.

    """
    my_ma0_sq = 2 * np.power(np.abs(mu_soft), 2) + mh_usq + mh_dsq
    return my_ma0_sq


#########################
# Fundamental equations
#########################


def logfunc(mass, q_renorm):
    """
    Return F = m^2 * (ln(m^2 / Q^2) - 1).

    Parameters
    ----------
    mass : Input mass.
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    myf : F = m^2 * (ln(m^2 / Q^2) - 1).

    """
    myf = np.power(mass, 2) * (np.log((np.power(mass, 2))
                                      / (np.power(q_renorm, 2))) - 1)
    return myf


def sinsqb(tanb):
    """
    Return sin^2(beta).

    Parameters
    ----------
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.

    Returns
    -------
    mysinsqb : Express sin^2(beta) in terms of tan(beta).

    """
    mysinsqb = (np.power(tanb, 2) / (1 + np.power(tanb, 2)))
    return mysinsqb


def cossqb(tanb):
    """
    Return cos^2(beta).

    Parameters
    ----------
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.

    Returns
    -------
    mycossqb : Express cos^2(beta) in terms of tan(beta).

    """
    mycossqb = 1 - (np.power(tanb, 2) / (1 + np.power(tanb, 2)))
    return mycossqb


def v_higgs_u(v_higgs, tanb):
    """
    Return up-type Higgs VEV.

    Parameters
    ----------
    v_higgs : Higgs VEV.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.

    Returns
    -------
    myvu : Up-type Higgs VEV.

    """
    myvu = v_higgs * np.sqrt(sinsqb(tanb))
    return myvu


def v_higgs_d(v_higgs, tanb):
    """
    Return down-type Higgs VEV.

    Parameters
    ----------
    v_higgs : Higgs VEV.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.

    Returns
    -------
    myvd : Down-type Higgs VEV.

    """
    myvd = v_higgs * np.sqrt(cossqb(tanb))
    return myvd


def tan_theta_w(g_coupling, g_prime):
    """
    Return tan(theta_W), the Weinberg angle.

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.

    Returns
    -------
    mytanthetaw : Ratio of coupling constants.

    """
    mytanthetaw = g_prime / g_coupling
    return mytanthetaw


def sin_squared_theta_w(g_coupling, g_prime):
    """
    Return sin^2(theta_W), the Weinberg angle.

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.

    Returns
    -------
    mysinsqthetaW : Gives sin^2(theta_W) in terms of tan(theta_W).

    """
    mysinsqthetaw = (np.power(tan_theta_w(g_coupling, g_prime), 2)
                     / (1 + np.power(tan_theta_w(g_coupling, g_prime), 2)))
    return mysinsqthetaw


def cos_squared_theta_w(g_coupling, g_prime):
    """
    Return cos^2(theta_W), the Weinberg angle.

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.

    Returns
    -------
    mycossqthetaW : Gives cos^2(theta_W) in terms of tan(theta_W).

    """
    mycossqthetaw = 1 - (np.power(tan_theta_w(g_coupling, g_prime), 2)
                         / (1 + np.power(tan_theta_w(g_coupling, g_prime), 2)))
    return mycossqthetaw


#########################
# Stop squarks:
#########################


def sigmauu_stop1(v_higgs, mu_soft, tanb, y_t, g_coupling, g_prime, m_stop_1,
                  m_stop_2, mtl, mtr, a_t, q_renorm):
    """
    Return one-loop correction Sigma_u^u(stop_1).

    Parameters
    ----------
    v_higgs : Higgs VEV.
    mu_soft : SUSY Higgs mass parameter, mu.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    y_t : Top Yukawa coupling.
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    m_stop_1 : Stop mass eigenstate mass 1.
    m_stop_2 : Stop mass eigenstate mass 2.
    mtl : Left gauge eigenstate stop mass.
    mtr : Right gauge eigenstate stop mass.
    a_t : Soft trilinear scalar top coupling.
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmauu_stop1 : One-loop correction Sigma_u^u(stop_1).

    """
    delta_ul = ((1 / 2) - (2 / 3) * sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(v_higgs_d(v_higgs, tanb), 2)
           - np.power(v_higgs_u(v_higgs, tanb), 2))
    delta_ur = (2 / 3) * sin_squared_theta_w(g_coupling, g_prime)\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(v_higgs_d(v_higgs, tanb), 2)
           - np.power(v_higgs_u(v_higgs, tanb), 2))
    stop_num = ((np.power(mtl, 2) - np.power(mtr, 2) + delta_ul - delta_ur)
                * ((-1 / 2) + ((4 / 3)
                               * sin_squared_theta_w(g_coupling, g_prime)))
                * (np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        + 2 * np.power(a_t, 2)
    sigmauu_stop1 = (3 / (32 * (np.power(np.pi, 2)))) *\
                    (2 * np.power(y_t, 2) + ((np.power(g_coupling, 2)
                                              + np.power(g_prime, 2))
                                             * (8 *
                                                sin_squared_theta_w(g_coupling,
                                                                    g_prime)
                                                - 3) / 12)
                     - (stop_num / (np.power(m_stop_2, 2)
                                    - np.power(m_stop_1, 2))))\
        * logfunc(m_stop_1, q_renorm)
    return sigmauu_stop1


def sigmadd_stop1(v_higgs, mu_soft, tanb, y_t, g_coupling, g_prime, m_stop_1,
                  m_stop_2, mtl, mtr, a_t, q_renorm):
    """
    Return one-loop correction Sigma_d^d(stop_1).

    Parameters
    ----------
    v_higgs : Higgs VEV.
    mu_soft : SUSY Higgs mass parameter, mu.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    y_t : Top Yukawa coupling.
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    m_stop_1 : Stop mass eigenstate mass 1.
    m_stop_2 : Stop mass eigenstate mass 2.
    mtl : Left gauge eigenstate stop mass.
    mtr : Right gauge eigenstate stop mass.
    a_t : Soft trilinear scalar top coupling.
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmadd_stop1 : One-loop correction Sigma_d^d(stop_1).

    """
    delta_ul = ((1 / 2) - (2 / 3) * sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(v_higgs_d(v_higgs, tanb), 2)
           - np.power(v_higgs_u(v_higgs, tanb), 2))
    delta_ur = (2 / 3) * sin_squared_theta_w(g, g_prime)\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(v_higgs_d(v_higgs, tanb), 2)
           - np.power(v_higgs_u(v_higgs, tanb), 2))
    stop_num = ((np.power(mtl, 2) - np.power(mtr, 2) + delta_ul - delta_ur)
                * ((1 / 2) + (4 / 3) * sin_squared_theta_w(g_coupling,
                                                           g_prime))
                * (np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        + 2 * np.power(y_t, 2) * np.power(mu_soft, 2)
    sigmadd_stop1 = (3 / (32 * (np.power(np.pi, 2))))\
        * (((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)
           - (stop_num / (np.power(m_stop_2, 2) - np.power(m_stop_1, 2))))\
        * logfunc(m_stop_1, q_renorm)
    return sigmadd_stop1


def sigmauu_stop2(v_higgs, mu_soft, tanb, y_t, g_coupling, g_prime, m_stop_1,
                  m_stop_2, mtl, mtr, a_t, q_renorm):
    """
    Return one-loop correction Sigma_u^u(stop_2).

    Parameters
    ----------
    v_higgs : Higgs VEV.
    mu_soft : SUSY Higgs mass parameter, mu.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    y_t : Top Yukawa coupling.
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    m_stop_1 : Stop mass eigenstate mass 1.
    m_stop_2 : Stop mass eigenstate mass 2.
    mtl : Left gauge eigenstate stop mass.
    mtr : Right gauge eigenstate stop mass.
    a_t : Soft trilinear scalar top coupling.
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmauu_stop2 : One-loop correction Sigma_u^u(stop_2).

    """
    delta_ul = ((1 / 2) - (2 / 3) * sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(v_higgs_d(v_higgs, tanb), 2)
           - np.power(v_higgs_u(v_higgs, tanb), 2))
    delta_ur = (2 / 3) * sin_squared_theta_w(g_coupling, g_prime)\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(v_higgs_d(v_higgs, tanb), 2)
           - np.power(v_higgs_u(v_higgs, tanb), 2))
    stop_num = ((np.power(mtl, 2) - np.power(mtr, 2) + delta_ul - delta_ur)
                * ((-1 / 2) + ((4 / 3) * sin_squared_theta_w(g_coupling,
                                                             g_prime)))
                * (np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        + 2 * np.power(a_t, 2)
    sigmauu_stop2 = (3 / (32 * (np.power(np.pi, 2))))\
        * (2 * np.power(y_t, 2) + ((np.power(g_coupling, 2)
                                    + np.power(g_prime, 2))
                                   * (8 * sin_squared_theta_w(g_coupling,
                                                              g_prime) - 3)
                                   / 12)
           + (stop_num / (np.power(m_stop_2, 2) - np.power(m_stop_1, 2))))\
        * logfunc(m_stop_2, q_renorm)
    return sigmauu_stop2


def sigmadd_stop2(v_higgs, mu_soft, tanb, y_t, g_coupling, g_prime, m_stop_1,
                  m_stop_2, mtl, mtr, a_t, q_renorm):
    """
    Return one-loop correction Sigma_d^d(stop_2).

    Parameters
    ----------
    v_higgs : Higgs VEV.
    mu_soft : SUSY Higgs mass parameter, mu.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    y_t : Top Yukawa coupling.
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    m_stop_1 : Stop mass eigenstate mass 1.
    m_stop_2 : Stop mass eigenstate mass 2.
    mtl : Left gauge eigenstate stop mass.
    mtr : Right gauge eigenstate stop mass.
    a_t : Soft trilinear scalar top coupling.
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmadd_stop2 : One-loop correction Sigma_d^d(stop_2).

    """
    delta_ul = ((1 / 2) - (2 / 3) * sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(v_higgs_d(v_higgs, tanb), 2)
           - np.power(v_higgs_u(v_higgs, tanb), 2))
    delta_ur = (2 / 3) * sin_squared_theta_w(g_coupling, g_prime)\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(v_higgs_d(v_higgs, tanb), 2)
           - np.power(v_higgs_u(v_higgs, tanb), 2))
    stop_num = ((np.power(mtl, 2) - np.power(mtr, 2) + delta_ul - delta_ur)
                * ((1 / 2) + (4 / 3) * sin_squared_theta_w(g_coupling,
                                                           g_prime))
                * (np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        + 2 * np.power(y_t, 2) * np.power(mu_soft, 2)
    sigmadd_stop2 = (3 / (32 * (np.power(np.pi, 2))))\
        * (((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)
           - (stop_num / (np.power(m_stop_2, 2) - np.power(m_stop_1, 2))))\
        * logfunc(m_stop_2, q_renorm)
    return sigmadd_stop2


#########################
# Sbottom squarks:
#########################


def sigmauu_sbottom1(v_higgs, mu_soft, tanb, y_b, g_coupling, g_prime,
                     m_sbot_1, m_sbot_2, mbl, mbr, a_b, q_renorm):
    """
    Return one-loop correction Sigma_u^u(sbottom_1).

    Parameters
    ----------
    v_higgs : Higgs VEV.
    mu_soft : SUSY Higgs mass parameter, mu.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    y_b : Bottom Yukawa coupling.
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    m_sbot_1 : Sbottom mass eigenstate mass 1.
    m_sbot_2 : Sbottom mass eigenstate mass 2.
    mbl : Left gauge eigenstate sbottom mass.
    mbr : Right gauge eigenstate sbottom mass.
    a_b : Soft trilinear scalar bottom coupling.
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmauu_sbot : One-loop correction Sigma_u^u(sbottom_1).

    """
    delta_dl = ((-1 / 2) + (1 / 3) * sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(v_higgs_d(v_higgs, tanb), 2)
           - np.power(v_higgs_u(v_higgs, tanb), 2))
    delta_dr = ((-1 / 3) * sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(v_higgs_d(v_higgs, tanb), 2)
           - np.power(v_higgs_u(v_higgs, tanb), 2))
    sbot_num = (np.power(mbr, 2) - np.power(mbl, 2) + delta_dr - delta_dl)\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * ((1 / 2) + (2 / 3) * sin_squared_theta_w(g_coupling, g_prime))\
        + 2 * np.power(y_b, 2) * np.power(mu_soft, 2)
    sigmauu_sbot = (3 / (32 * np.power(np.pi, 2)))\
        * (((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 4)
           - (sbot_num / (np.power(m_sbot_2, 2) - np.power(m_sbot_1, 2))))\
        * logfunc(m_sbot_1, q_renorm)
    return sigmauu_sbot


def sigmauu_sbottom2(v_higgs, mu_soft, tanb, y_b, g_coupling, g_prime,
                     m_sbot_1, m_sbot_2, mbl, mbr, a_b, q_renorm):
    """
    Return one-loop correction Sigma_u^u(sbottom_2).

    Parameters
    ----------
    v_higgs : Higgs VEV.
    mu_soft : SUSY Higgs mass parameter, mu.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    y_b : Bottom Yukawa coupling.
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    m_sbot_1 : Sbottom mass eigenstate mass 1.
    m_sbot_2 : Sbottom mass eigenstate mass 2.
    mbl : Left gauge eigenstate sbottom mass.
    mbr : Right gauge eigenstate sbottom mass.
    a_b : Soft trilinear scalar bottom coupling.
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmauu_sbot : One-loop correction Sigma_u^u(sbottom_2).

    """
    delta_dl = ((-1 / 2) + (1 / 3) * sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(v_higgs_d(v_higgs, tanb), 2)
           - np.power(v_higgs_u(v_higgs, tanb), 2))
    delta_dr = ((-1 / 3) * sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(v_higgs_d(v_higgs, tanb), 2)
           - np.power(v_higgs_u(v_higgs, tanb), 2))
    sbot_num = (np.power(mbr, 2) - np.power(mbl, 2) + delta_dr - delta_dl)\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * ((1 / 2) + (2 / 3) * sin_squared_theta_w(g_coupling, g_prime))\
        + 2 * np.power(y_b, 2) * np.power(mu_soft, 2)
    sigmauu_sbot = (3 / (32 * np.power(np.pi, 2)))\
        * (((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 4)
           + (sbot_num / (np.power(m_sbot_2, 2) - np.power(m_sbot_1, 2))))\
        * logfunc(m_sbot_2, q_renorm)
    return sigmauu_sbot


def sigmadd_sbottom1(v_higgs, mu_soft, tanb, y_b, g_coupling, g_prime,
                     m_sbot_1, m_sbot_2, mbl, mbr, a_b, q_renorm):
    """
    Return one-loop correction Sigma_d^d(sbottom_1).

    Parameters
    ----------
    v_higgs : Higgs VEV.
    mu_soft : SUSY Higgs mass parameter, mu.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    y_b : Bottom Yukawa coupling.
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    m_sbot_1 : Sbottom mass eigenstate mass 1.
    m_sbot_2 : Sbottom mass eigenstate mass 2.
    mbl : Left gauge eigenstate sbottom mass.
    mbr : Right gauge eigenstate sbottom mass.
    a_b : Soft trilinear scalar bottom coupling.
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmadd_sbot : One-loop correction Sigma_d^d(sbottom_1).

    """
    delta_dl = ((-1 / 2) + (1 / 3) * sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(v_higgs_d(v_higgs, tanb), 2)
           - np.power(v_higgs_u(v_higgs, tanb), 2))
    delta_dr = ((-1 / 3) * sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(v_higgs_d(v_higgs, tanb), 2)
           - np.power(v_higgs_u(v_higgs, tanb), 2))
    sbot_num = (np.power(mbr, 2) - np.power(mbl, 2) + delta_dr - delta_dl)\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * ((-1 / 2) - (2 / 3) * sin_squared_theta_w(g_coupling, g_prime))\
        + 2 * np.power(a_b, 2)
    sigmadd_sbot = (3 / (32 * np.power(np.pi, 2)))\
        * (((-1) * (np.power(g_coupling, 2) + np.power(g_prime, 2)) / 4)
           - (sbot_num / (np.power(m_sbot_2, 2) - np.power(m_sbot_1, 2))))\
        * logfunc(m_sbot_1, q_renorm)
    return sigmadd_sbot


def sigmadd_sbottom2(v_higgs, mu_soft, tanb, y_b, g_coupling, g_prime,
                     m_sbot_1, m_sbot_2, mbl, mbr, a_b, q_renorm):
    """
    Return one-loop correction Sigma_d^d(sbottom_2).

    Parameters
    ----------
    v_higgs : Higgs VEV.
    mu_soft : SUSY Higgs mass parameter, mu.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    y_b : Bottom Yukawa coupling.
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    m_sbot_1 : Sbottom mass eigenstate mass 1.
    m_sbot_2 : Sbottom mass eigenstate mass 2.
    mbl : Left gauge eigenstate sbottom mass.
    mbr : Right gauge eigenstate sbottom mass.
    a_b : Soft trilinear scalar bottom coupling.
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmadd_sbot : One-loop correction Sigma_d^d(sbottom_2).

    """
    delta_dl = ((-1 / 2) + (1 / 3) * sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(v_higgs_d(v_higgs, tanb), 2)
           - np.power(v_higgs_u(v_higgs, tanb), 2))
    delta_dr = ((-1 / 3) * sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(v_higgs_d(v_higgs, tanb), 2)
           - np.power(v_higgs_u(v_higgs, tanb), 2))
    sbot_num = (np.power(mbr, 2) - np.power(mbl, 2) + delta_dr - delta_dl)\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * ((-1 / 2) - (2 / 3) * sin_squared_theta_w(g_coupling, g_prime))\
        + 2 * np.power(a_b, 2)
    sigmadd_sbot = (3 / (32 * np.power(np.pi, 2)))\
        * (((-1) * (np.power(g_coupling, 2) + np.power(g_prime, 2)) / 4)
           + (sbot_num / (np.power(m_sbot_2, 2) - np.power(m_sbot_1, 2))))\
        * logfunc(m_sbot_2, q_renorm)
    return sigmadd_sbot


#########################
# Stau sleptons:
#########################


def sigmauu_stau1(v_higgs, mu_soft, tanb, y_tau, g_coupling, g_prime, m_stau_1,
                  m_stau_2, mtaul, mtaur, a_tau, q_renorm):
    """
    Return one-loop correction Sigma_u^u(stau_1).

    Parameters
    ----------
    v_higgs : Higgs VEV.
    mu_soft : SUSY Higgs mass parameter, mu.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    y_tau : Tau Yukawa coupling.
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    m_stau_1 : Stau mass eigenstate mass 1.
    m_stau_2 : Stau mass eigenstate mass 2.
    mtaul : Left gauge eigenstate stau mass.
    mtaur : Right gauge eigenstate stau mass.
    a_tau : Soft trilinear scalar tau coupling.
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmauu_stau : One-loop correction Sigma_u^u(stau_1).

    """
    delta_el = ((-1 / 2) + sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(v_higgs_d(v_higgs, tanb), 2)
           - np.power(v_higgs_u(v_higgs, tanb), 2))
    delta_er = ((-1) * sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(v_higgs_d(v_higgs, tanb), 2)
           - np.power(v_higgs_u(v_higgs, tanb), 2))
    stau_num = (np.power(mtaur, 2) - np.power(mtaul, 2) + delta_er - delta_el)\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * ((-1 / 2) + 2 * sin_squared_theta_w(g_coupling, g_prime))\
        + 2 * np.power(y_tau, 2) * np.power(mu_soft, 2)
    sigmauu_stau = (1 / (32 * np.power(np.pi, 2)))\
        * (((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 4)
           - (stau_num / (np.power(m_stau_2, 2) - np.power(m_stau_1, 2))))\
        * logfunc(m_stau_1, q_renorm)
    return sigmauu_stau


def sigmauu_stau2(v_higgs, mu_soft, tanb, y_tau, g_coupling, g_prime, m_stau_1,
                  m_stau_2, mtaul, mtaur, a_tau, q_renorm):
    """
    Return one-loop correction Sigma_u^u(stau_2).

    Parameters
    ----------
    v_higgs : Higgs VEV.
    mu_soft : SUSY Higgs mass parameter, mu.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    y_tau : Tau Yukawa coupling.
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    m_stau_1 : Stau mass eigenstate mass 1.
    m_stau_2 : Stau mass eigenstate mass 2.
    mtaul : Left gauge eigenstate stau mass.
    mtaur : Right gauge eigenstate stau mass.
    a_tau : Soft trilinear scalar tau coupling.
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmauu_stau : One-loop correction Sigma_u^u(stau_2).

    """
    delta_el = ((-1 / 2) + sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(v_higgs_d(v_higgs, tanb), 2)
           - np.power(v_higgs_u(v_higgs, tanb), 2))
    delta_er = ((-1) * sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(v_higgs_d(v_higgs, tanb), 2)
           - np.power(v_higgs_u(v_higgs, tanb), 2))
    stau_num = (np.power(mtaur, 2) - np.power(mtaul, 2) + delta_er - delta_el)\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * ((-1 / 2) + 2 * sin_squared_theta_w(g_coupling, g_prime))\
        + 2 * np.power(y_tau, 2) * np.power(mu_soft, 2)
    sigmauu_stau = (1 / (32 * np.power(np.pi, 2)))\
        * (((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 4)
           + (stau_num / (np.power(m_stau_2, 2) - np.power(m_stau_1, 2))))\
        * logfunc(m_stau_2, q_renorm)
    return sigmauu_stau


def sigmadd_stau1(v_higgs, mu_soft, tanb, y_tau, g_coupling, g_prime, m_stau_1,
                  m_stau_2, mtaul, mtaur, a_tau, q_renorm):
    """
    Return one-loop correction Sigma_d^d(stau_1).

    Parameters
    ----------
    v_higgs : Higgs VEV.
    mu_soft : SUSY Higgs mass parameter, mu.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    y_tau : Tau Yukawa coupling.
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    m_stau_1 : Stau mass eigenstate mass 1.
    m_stau_2 : Stau mass eigenstate mass 2.
    mtaul : Left gauge eigenstate stau mass.
    mtaur : Right gauge eigenstate stau mass.
    a_tau : Soft trilinear scalar tau coupling.
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmadd_stau : One-loop correction Sigma_d^d(stau_1).

    """
    delta_el = ((-1 / 2) + sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(v_higgs_d(v_higgs, tanb), 2)
           - np.power(v_higgs_u(v_higgs, tanb), 2))
    delta_er = ((-1) * sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(v_higgs_d(v_higgs, tanb), 2)
           - np.power(v_higgs_u(v_higgs, tanb), 2))
    stau_num = (np.power(mtaur, 2) - np.power(mtaul, 2) + delta_er - delta_el)\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * ((1 / 2) - (2 * sin_squared_theta_w(g_coupling, g_prime)))\
        + 2 * np.power(a_tau, 2)
    sigmadd_stau = (1 / (32 * np.power(np.pi, 2)))\
        * (((-1) * (np.power(g_coupling, 2) + np.power(g_prime, 2)) / 4)
           - (stau_num / (np.power(m_stau_2, 2) - np.power(m_stau_1, 2))))\
        * logfunc(m_stau_1, q_renorm)
    return sigmadd_stau


def sigmadd_stau2(v_higgs, mu_soft, tanb, y_tau, g_coupling, g_prime, m_stau_1,
                  m_stau_2, mtaul, mtaur, a_tau, q_renorm):
    """
    Return one-loop correction Sigma_d^d(stau_2).

    Parameters
    ----------
    v_higgs : Higgs VEV.
    mu_soft : SUSY Higgs mass parameter, mu.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    y_tau : Tau Yukawa coupling.
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    m_stau_1 : Stau mass eigenstate mass 1.
    m_stau_2 : Stau mass eigenstate mass 2.
    mtaul : Left gauge eigenstate stau mass.
    mtaur : Right gauge eigenstate stau mass.
    a_tau : Soft trilinear scalar tau coupling.
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmadd_stau : One-loop correction Sigma_d^d(stau_2).

    """
    delta_el = ((-1 / 2) + sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(v_higgs_d(v_higgs, tanb), 2)
           - np.power(v_higgs_u(v_higgs, tanb), 2))
    delta_er = ((-1) * sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(v_higgs_d(v_higgs, tanb), 2)
           - np.power(v_higgs_u(v_higgs, tanb), 2))
    stau_num = (np.power(mtaur, 2) - np.power(mtaul, 2) + delta_er - delta_el)\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * ((1 / 2) - (2 * sin_squared_theta_w(g_coupling, g_prime)))\
        + 2 * np.power(a_tau, 2)
    sigmadd_stau = (1 / (32 * np.power(np.pi, 2)))\
        * (((-1) * (np.power(g_coupling, 2) + np.power(g_prime, 2)) / 4)
           + (stau_num / (np.power(m_stau_2, 2) - np.power(m_stau_1, 2))))\
        * logfunc(m_stau_2, q_renorm)
    return sigmadd_stau


#########################
# Sfermions, 1st gen:
#########################


def sigmauu_sup_l(g_coupling, g_prime, msup_l, q_renorm):
    """
    Return one-loop correction Sigma_u^u(sup_L).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    msup_l : Soft SUSY breaking mass for scalar up quark (left).
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmauusup_l : One-loop correction Sigma_u^u(sup_L).

    """
    sigmauusup_l = ((-3) / (16 * np.power(np.pi, 2)))\
        * ((1 / 2) - (2 / 3) * sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * logfunc(msup_l, q_renorm)
    return sigmauusup_l


def sigmauu_sup_r(g_coupling, g_prime, msup_r, q_renorm):
    """
    Return one-loop correction Sigma_u^u(sup_R).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    msup_r : Soft SUSY breaking mass for scalar up quark (right).
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmauusup_r : One-loop correction Sigma_u^u(sup_R).

    """
    sigmauusup_r = ((-3) / (16 * np.power(np.pi, 2)))\
        * ((2 / 3) * sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * logfunc(msup_r, q_renorm)
    return sigmauusup_r


def sigmauu_sdown_l(g_coupling, g_prime, msdown_l, q_renorm):
    """
    Return one-loop correction Sigma_u^u(sdown_L).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    msdown_l : Soft SUSY breaking mass for scalar down quark (left).
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmauusdown_l : One-loop correction Sigma_u^u(sdown_L).

    """
    sigmauusdown_l = ((-3) / (16 * np.power(np.pi, 2)))\
        * (((-1) / 2) + (1 / 3) * sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * logfunc(msdown_l, q_renorm)
    return sigmauusdown_l


def sigmauu_sdown_r(g_coupling, g_prime, msdown_r, q_renorm):
    """
    Return one-loop correction Sigma_u^u(sdown_R).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    msdown_r : Soft SUSY breaking mass for scalar down quark (right).
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmauusdown_r : One-loop correction Sigma_u^u(sdown_R).

    """
    sigmauusdown_r = ((-3) / (16 * np.power(np.pi, 2)))\
        * (((-1) / 3) * sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * logfunc(msdown_r, q_renorm)
    return sigmauusdown_r


def sigmauu_selec_l(g_coupling, g_prime, mselec_l, q_renorm):
    """
    Return one-loop correction Sigma_u^u(selectron_L).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    mselec_l : Soft SUSY breaking mass for scalar electron (left).
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmauuselec_l : One-loop correction Sigma_u^u(selectron_L).

    """
    sigmauuselec_l = ((-1) / (16 * np.power(np.pi, 2)))\
        * (((-1) / 2) + sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * logfunc(mselec_l, q_renorm)
    return sigmauuselec_l


def sigmauu_selec_r(g_coupling, g_prime, mselec_r, q_renorm):
    """
    Return one-loop correction Sigma_u^u(selectron_R).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    mselec_r : Soft SUSY breaking mass for scalar electron (right).
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmauuselec_r : One-loop correction Sigma_u^u(selectron_R).

    """
    sigmauuselec_r = ((-1) / (16 * np.power(np.pi, 2)))\
        * ((-1) * sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * logfunc(mselec_r, q_renorm)
    return sigmauuselec_r


def sigmauu_selec_sneut(g_coupling, g_prime, mselec_sneut, q_renorm):
    """
    Return one-loop correction Sigma_u^u(selectron neutrino).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    mselec_sneut : Mass for scalar electron neutrino.
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmauuselec_sneut : One-loop correction Sigma_u^u(selectron neutrino).

    """
    sigmauuselec_sneut = ((-1) / (32 * np.power(np.pi, 2))) * (1 / 2)\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * logfunc(mselec_sneut, q_renorm)
    return sigmauuselec_sneut


def sigmadd_sup_l(g_coupling, g_prime, msup_l, q_renorm):
    """
    Return one-loop correction Sigma_d^d(sup_L).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    msup_l : Soft SUSY breaking mass for scalar up quark (left).
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmaddsup_l : One-loop correction Sigma_d^d(sup_L).

    """
    sigmaddsup_l = (3 / (16 * np.power(np.pi, 2)))\
        * ((1 / 2) - (2 / 3) * sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * logfunc(msup_l, q_renorm)
    return sigmaddsup_l


def sigmadd_sup_r(g_coupling, g_prime, msup_r, q_renorm):
    """
    Return one-loop correction Sigma_d^d(sup_R).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    msup_r : Soft SUSY breaking mass for scalar up quark (right).
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmaddsup_r : One-loop correction Sigma_d^d(sup_R).

    """
    sigmaddsup_r = (3 / (16 * np.power(np.pi, 2)))\
        * ((2 / 3) * sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * logfunc(msup_r, q_renorm)
    return sigmaddsup_r


def sigmadd_sdown_l(g_coupling, g_prime, msdown_l, q_renorm):
    """
    Return one-loop correction Sigma_d^d(sdown_L).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    msdown_l : Soft SUSY breaking mass for scalar down quark (left).
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmaddsdown_l : One-loop correction Sigma_d^d(sdown_L).

    """
    sigmaddsdown_l = (3 / (16 * np.power(np.pi, 2)))\
        * (((-1) / 2) + (1 / 3) * sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * logfunc(msdown_l, q_renorm)
    return sigmaddsdown_l


def sigmadd_sdown_r(g_coupling, g_prime, msdown_r, q_renorm):
    """
    Return one-loop correction Sigma_d^d(sdown_R).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    msdown_r : Soft SUSY breaking mass for scalar down quark (right).
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmaddsdown_r : One-loop correction Sigma_d^d(sdown_R).

    """
    sigmaddsdown_r = (3 / (16 * np.power(np.pi, 2)))\
        * (((-1) / 3) * sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * logfunc(msdown_r, q_renorm)
    return sigmaddsdown_r


def sigmadd_selec_l(g_coupling, g_prime, mselec_l, q_renorm):
    """
    Return one-loop correction Sigma_d^d(selectron_L).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    mselec_l : Soft SUSY breaking mass for scalar electron (left).
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmaddselec_l : One-loop correction Sigma_d^d(selectron_L).

    """
    sigmaddselec_l = (1 / (16 * np.power(np.pi, 2)))\
        * (((-1) / 2) + sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * logfunc(mselec_l, q_renorm)
    return sigmaddselec_l


def sigmadd_selec_r(g_coupling, g_prime, mselec_r, q_renorm):
    """
    Return one-loop correction Sigma_d^d(selectron_R).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    mselec_r : Soft SUSY breaking mass for scalar electron (right).
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmaddselec_r : One-loop correction Sigma_d^d(selectron_R).

    """
    sigmaddselec_r = (1 / (16 * np.power(np.pi, 2)))\
        * ((-1) * sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * logfunc(mselec_r, q_renorm)
    return sigmaddselec_r


def sigmadd_selec_sneut(g_coupling, g_prime, mselec_sneut, q_renorm):
    """
    Return one-loop correction Sigma_d^d(selectron neutrino).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    mselec_sneut : Mass for scalar electron neutrino.
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmaddselec_sneut : One-loop correction Sigma_d^d(selectron neutrino).

    """
    sigmauuselec_sneut = (1 / (32 * np.power(np.pi, 2))) * (1 / 2)\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * logfunc(mselec_sneut, q_renorm)
    return sigmauuselec_sneut


#########################
# Sfermions, 2nd gen:
#########################


def sigmauu_sstrange_l(g_coupling, g_prime, msstrange_l, q_renorm):
    """
    Return one-loop correction Sigma_u^u(sstrange_L).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    msstrange_l : Soft SUSY breaking mass for scalar strange quark (left).
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmauusstrange_l : One-loop correction Sigma_u^u(sstrange_L).

    """
    sigmauusstrange_l = ((-3) / (16 * np.power(np.pi, 2)))\
        * (((-1) / 2) + (1 / 3) * sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * logfunc(msstrange_l, q_renorm)
    return sigmauusstrange_l


def sigmauu_sstrange_r(g_coupling, g_prime, msstrange_r, q_renorm):
    """
    Return one-loop correction Sigma_u^u(sstrange_R).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    msstrange_r : Soft SUSY breaking mass for scalar strange quark (right).
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmauusstrange_r : One-loop correction Sigma_u^u(sstrange_R).

    """
    sigmauusstrange_r = ((-3) / (16 * np.power(np.pi, 2)))\
        * (((-1) / 3) * sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * logfunc(msstrange_r, q_renorm)
    return sigmauusstrange_r


def sigmauu_scharm_l(g_coupling, g_prime, mscharm_l, q_renorm):
    """
    Return one-loop correction Sigma_u^u(scharm_L).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    mscharm_l : Soft SUSY breaking mass for scalar charm quark (left).
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmauuscharm_l : One-loop correction Sigma_u^u(scharm_L).

    """
    sigmauuscharm_l = ((-3) / (16 * np.power(np.pi, 2)))\
        * ((1 / 2) - (2 / 3) * sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * logfunc(mscharm_l, q_renorm)
    return sigmauuscharm_l


def sigmauu_scharm_r(g_coupling, g_prime, mscharm_r, q_renorm):
    """
    Return one-loop correction Sigma_u^u(scharm_R).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    mscharm_r : Soft SUSY breaking mass for scalar charm quark (right).
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmauuscharm_r : One-loop correction Sigma_u^u(scharm_R).

    """
    sigmauuscharm_r = ((-3) / (16 * np.power(np.pi, 2)))\
        * ((2 / 3) * sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * logfunc(mscharm_r, q_renorm)
    return sigmauuscharm_r


def sigmauu_smu_l(g_coupling, g_prime, msmu_l, q_renorm):
    """
    Return one-loop correction Sigma_u^u(smu_L).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    msmu_l : Soft SUSY breaking mass for scalar muon (left).
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmauusmu_l : One-loop correction Sigma_u^u(smu_L).

    """
    sigmauusmu_l = ((-1) / (16 * np.power(np.pi, 2)))\
        * (((-1) / 2) + sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * logfunc(msmu_l, q_renorm)
    return sigmauusmu_l


def sigmauu_smu_r(g_coupling, g_prime, msmu_r, q_renorm):
    """
    Return one-loop correction Sigma_u^u(smu_R).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    msmu_r : Soft SUSY breaking mass for scalar muon (right).
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmauusmu_r : One-loop correction Sigma_u^u(smu_R).

    """
    sigmauusmu_r = ((-1) / (16 * np.power(np.pi, 2)))\
        * (((-1)) * sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * logfunc(msmu_r, q_renorm)
    return sigmauusmu_r


def sigmauu_smu_sneut(g_coupling, g_prime, msmu_sneut, q_renorm):
    """
    Return one-loop correction Sigma_u^u(smuon neutrino).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    msmuSneut : Mass for scalar muon neutrino.
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmauusmu_sneut : One-loop correction Sigma_u^u(smuon neutrino).

    """
    sigmauusmu_sneut = ((-1) / (32 * np.power(np.pi, 2))) * (1 / 2)\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * logfunc(msmu_sneut, q_renorm)
    return sigmauusmu_sneut


def sigmadd_sstrange_l(g_coupling, g_prime, msstrange_l, q_renorm):
    """
    Return one-loop correction Sigma_d^d(sstrange_L).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    msstrange_l : Soft SUSY breaking mass for scalar strange quark (left).
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmaddsstrange_l : One-loop correction Sigma_d^d(sstrange_L).

    """
    sigmaddsstrange_l = (3 / (16 * np.power(np.pi, 2)))\
        * (((-1) / 2) + (1 / 3) * sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * logfunc(msstrange_l, q_renorm)
    return sigmaddsstrange_l


def sigmadd_sstrange_r(g_coupling, g_prime, msstrange_r, q_renorm):
    """
    Return one-loop correction Sigma_d^d(sstrange_R).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    msstrange_r : Soft SUSY breaking mass for scalar strange quark (right).
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmaddsstrange_r : One-loop correction Sigma_d^d(sstrange_R).

    """
    sigmaddsstrange_r = (3 / (16 * np.power(np.pi, 2)))\
        * (((-1) / 3) * sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * logfunc(msstrange_r, q_renorm)
    return sigmaddsstrange_r


def sigmadd_scharm_l(g_coupling, g_prime, mscharm_l, q_renorm):
    """
    Return one-loop correction Sigma_d^d(scharm_L).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    mscharm_l : Soft SUSY breaking mass for scalar charm quark (left).
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmaddscharm_l : One-loop correction Sigma_d^d(scharm_L).

    """
    sigmaddscharm_l = (3 / (16 * np.power(np.pi, 2)))\
        * ((1 / 2) - (2 / 3) * sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * logfunc(mscharm_l, q_renorm)
    return sigmaddscharm_l


def sigmadd_scharm_r(g_coupling, g_prime, mscharm_r, q_renorm):
    """
    Return one-loop correction Sigma_d^d(scharm_R).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    mscharm_r : Soft SUSY breaking mass for scalar charm quark (right).
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmaddscharm_r : One-loop correction Sigma_d^d(scharm_R).

    """
    sigmaddscharm_r = (3 / (16 * np.power(np.pi, 2)))\
        * ((2 / 3) * sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * logfunc(mscharm_r, q_renorm)
    return sigmaddscharm_r


def sigmadd_smu_l(g_coupling, g_prime, msmu_l, q_renorm):
    """
    Return one-loop correction Sigma_d^d(smu_L).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    msmu_l : Soft SUSY breaking mass for scalar muon (left).
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmaddsmu_l : One-loop correction Sigma_d^d(smu_L).

    """
    sigmaddsmu_l = (1 / (16 * np.power(np.pi, 2)))\
        * (((-1) / 2) + sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * logfunc(msmu_l, q_renorm)
    return sigmaddsmu_l


def sigmadd_smu_r(g_coupling, g_prime, msmu_r, q_renorm):
    """
    Return one-loop correction Sigma_d^d(smu_R).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    msmu_r : Soft SUSY breaking mass for scalar muon (right).
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmaddsmu_r : One-loop correction Sigma_d^d(smu_R).

    """
    sigmaddsmu_r = (1 / (16 * np.power(np.pi, 2)))\
        * (((-1)) * sin_squared_theta_w(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * logfunc(msmu_r, q_renorm)
    return sigmaddsmu_r


def sigmadd_smu_sneut(g_coupling, g_prime, msmu_sneut, q_renorm):
    """
    Return one-loop correction Sigma_d^d(smuon neutrino).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    msmu_sneut : Mass for scalar muon neutrino.
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmaddsmu_sneut : One-loop correction Sigma_d^d(smuon neutrino).

    """
    sigmaddsmu_sneut = (1 / (32 * np.power(np.pi, 2))) * (1 / 2)\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * logfunc(msmu_sneut, q_renorm)
    return sigmaddsmu_sneut


#########################
# Neutralinos:
#########################
# Set up terms from characteristic polynomial for eigenvalues x of squared
# neutralino mass matrix, and use method by Ibrahim and Nath for derivatives
# of eigenvalues.
# x^4 + b(vu, vd) * x^3 + c(vu, vd) * x^2 + d(vu, vd) * x + e(vu, vd) = 0


def neutralinouu_deriv_num(m1_bino, m2_wino, mu_soft, g_coupling, g_prime,
                           v_higgs, tanb, msN):
    """
    Return numerator for one-loop uu correction derivative term of neutralino.

    Parameters
    ----------
    m1_bino : Bino mass parameter.
    m2_wino : Wino mass parameter.
    mu_soft : SUSY Higgs mass parameter, mu.
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    v_higgs : Higgs VEV.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    msN : Neutralino mass.

    Returns
    -------
    mynum : Numerator for one-loop correction derivative term of neutralino.

    """
    cubicterm = np.power(g_coupling, 2) + np.power(g_prime, 2)
    quadrterm = (((np.power(g_coupling, 2) * m2_wino * mu_soft)
                  + (np.power(g_prime, 2) * m1_bino * mu_soft)) / (tanb))\
        - ((np.power(g_coupling, 2) * np.power(m1_bino, 2))
           + (np.power(g_prime, 2) * np.power(m2_wino, 2))
           + ((np.power(g_coupling, 2) + np.power(g_prime, 2))
              * (np.power(mu_soft, 2)))
           + (np.power((np.power(g_coupling, 2) + np.power(g_prime, 2)), 2)
              / 2)
           * np.power(v_higgs, 2))
    linterm = (((-1) * mu_soft) * ((np.power(g_coupling, 2) * m2_wino
                                    * (np.power(m1_bino, 2)
                                       + np.power(mu_soft, 2)))
                                   + np.power(g_prime, 2) * m1_bino
                                   * (np.power(m2_wino, 2)
                                      + np.power(mu_soft, 2)))
               / tanb)\
        + ((np.power((np.power(g_coupling, 2) * m1_bino + np.power(g_prime, 2)
                      * m2_wino), 2) / 2)
           * np.power(v_higgs, 2))\
        + (np.power(mu_soft, 2) * ((np.power(g_coupling, 2)
                                    * np.power(m1_bino, 2))
                                   + (np.power(g_prime, 2)
                                      * np.power(m2_wino, 2))))\
        + (np.power((np.power(g_coupling, 2) + np.power(g_prime, 2)), 2)
           * np.power(v_higgs, 2) * np.power(mu_soft, 2) * cossqb(tanb))
    constterm = (m1_bino * m2_wino * ((np.power(g_coupling, 2) * m1_bino)
                                      + (np.power(g_prime, 2) * m2_wino))
                 * np.power(mu_soft, 3) * (1 / tanb))\
        - (np.power((np.power(g_coupling, 2) * m1_bino + np.power(g_prime, 2)
                     * m2_wino), 2)
           * np.power(v_higgs, 2) * np.power(mu_soft, 2) * cossqb(tanb))
    mynum = (cubicterm * np.power(msN, 6)) + (quadrterm * np.power(msN, 4))\
        + (linterm * np.power(msN, 2)) + constterm
    return mynum


def neutralinodd_deriv_num(m1_bino, m2_wino, mu_soft, g_coupling, g_prime,
                           v_higgs, tanb, msN):
    """
    Return numerator for one-loop dd correction derivative term of neutralino.

    Parameters
    ----------
    m1_bino : Bino mass parameter.
    m2_wino : Wino mass parameter.
    mu_soft : SUSY Higgs mass parameter, mu.
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    v_higgs : Higgs VEV.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    msN : Neutralino mass.

    Returns
    -------
    mynum : Numerator for one-loop correction derivative term of neutralino.

    """
    cubicterm = np.power(g_coupling, 2) + np.power(g_prime, 2)
    quadrterm = (((np.power(g_coupling, 2) * m2_wino * mu_soft)
                  + (np.power(g_prime, 2) * m1_bino * mu_soft)) * (tanb))\
        - ((np.power(g_coupling, 2) * np.power(m1_bino, 2))
           + (np.power(g_prime, 2) * np.power(m2_wino, 2))
           + ((np.power(g_coupling, 2) + np.power(g_prime, 2))
              * (np.power(mu_soft, 2)))
           + (np.power((np.power(g_coupling, 2) + np.power(g_prime, 2)), 2)
              / 2)
           * np.power(v_higgs, 2))
    linterm = (((-1) * mu_soft) * ((np.power(g_coupling, 2) * m2_wino
                                    * (np.power(m1_bino, 2)
                                       + np.power(mu_soft, 2)))
                                   + np.power(g_prime, 2) * m1_bino
                                   * (np.power(m2_wino, 2)
                                      + np.power(mu_soft, 2)))
               * tanb)\
        + ((np.power((np.power(g_coupling, 2) * m1_bino + np.power(g_prime, 2)
                      * m2_wino), 2) / 2)
           * np.power(v_higgs, 2))\
        + (np.power(mu_soft, 2) * ((np.power(g_coupling, 2)
                                    * np.power(m1_bino, 2))
           + np.power(g_prime, 2) * np.power(m2_wino, 2)))\
        + (np.power((np.power(g_coupling, 2) + np.power(g_prime, 2)), 2)
           * np.power(v_higgs, 2) * np.power(mu_soft, 2) * sinsqb(tanb))
    constterm = (m1_bino * m2_wino * (np.power(g_coupling, 2)
                                      * m1_bino + (np.power(g_prime, 2)
                                                   * m2_wino))
                 * np.power(mu_soft, 3) * tanb)\
        - (np.power((np.power(g_coupling, 2) * m1_bino + np.power(g_prime, 2)
                     * m2_wino), 2)
           * np.power(v_higgs, 2) * np.power(mu_soft, 2) * sinsqb(tanb))
    mynum = (cubicterm * np.power(msN, 6))\
        + (quadrterm * np.power(msN, 4))\
        + (linterm * np.power(msN, 2)) + constterm
    return mynum


def neutralino_deriv_denom(m1_bino, m2_wino, mu_soft, g_coupling, g_prime,
                           v_higgs, tanb, msN):
    """
    Return denominator for one-loop correction derivative term of neutralino.

    Parameters
    ----------
    m1_bino : Bino mass parameter.
    m2_wino : Wino mass parameter.
    mu_soft : SUSY Higgs mass parameter, mu.
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    v_higgs : Higgs VEV.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    msN : Neutralino mass.

    Returns
    -------
    mydenom : Denominator for 1-loop correction derivative term of neutralino.

    """
    quadrterm = -3 * ((np.power(m1_bino, 2)) + (np.power(m2_wino, 2))
                      + ((np.power(g_coupling, 2) + np.power(g_prime, 2))
                         * np.power(v_higgs, 2))
                      + (2 * np.power(mu_soft, 2)))
    linterm = (np.power(v_higgs, 4)
               * np.power((np.power(g_coupling, 2) + np.power(g_prime, 2)), 2)
               / 2)\
        + (np.power(v_higgs, 2)
           * (2 * ((np.power(g_coupling, 2) * np.power(m1_bino, 2))
                   + (np.power(g_prime, 2) * np.power(m2_wino, 2))
                   + ((np.power(g_coupling, 2)
                       + np.power(g_prime, 2)) * np.power(mu_soft, 2))
                   - (mu_soft
                      * (np.power(g_prime, 2) * m1_bino
                         + np.power(g_coupling, 2) * m2_wino)
                      * 2 * np.sqrt(sinsqb(tanb)) * np.sqrt(cossqb(tanb))))))\
        + (2 * ((np.power(m1_bino, 2) * np.power(m2_wino, 2))
                + (2 * (np.power(m1_bino, 2) + np.power(m2_wino, 2))
                   * np.power(mu_soft, 2))
                + (np.power(mu_soft, 4))))
    constterm = (np.power(v_higgs, 4) * (1 / 8)
                 * ((np.power((np.power(g_coupling, 2) + np.power(g_prime, 2)),
                              2)
                     * np.power(mu_soft, 2)
                     * (np.power(cossqb(tanb), 2)
                        - (6 * cossqb(tanb) * sinsqb(tanb))
                        + np.power(sinsqb(tanb), 2)))
                    - (2 * np.power((np.power(g_coupling, 2) * m1_bino
                                     + np.power(g_prime, 2) * m2_wino), 2))
                    - (np.power(mu_soft, 2)
                       * np.power((np.power(g_coupling, 2)
                                   + np.power(g_prime, 2)), 2))
                    ))\
        + (np.power(v_higgs, 2) * 2 * mu_soft
           * ((np.sqrt(cossqb(tanb)) * np.sqrt(sinsqb(tanb)))
              * (np.power(g_coupling, 2) * m2_wino * (np.power(m1_bino, 2)
                                                      + np.power(mu_soft, 2))
                 + (np.power(g_prime, 2) * m1_bino
                 * (np.power(m2_wino, 2) + np.power(mu_soft, 2))))))\
        - ((2 * np.power(m2_wino, 2) * np.power(m1_bino, 2)
            * np.power(mu_soft, 2))
           + (np.power(mu_soft, 4) * (np.power(m1_bino, 2)
                                      + np.power(m2_wino, 2))))
    mydenom = 4 * np.power(msN, 6)\
        + quadrterm * np.power(msN, 4)\
        + linterm * np.power(msN, 2)\
        + constterm
    return mydenom


def sigmauu_neutralino(m1_bino, m2_wino, mu_soft, g_coupling, g_prime, v_higgs,
                       tanb, msN, q_renorm):
    """
    Return one-loop correction Sigma_u^u(neutralino).

    Parameters
    ----------
    m1_bino : Bino mass parameter.
    m2_wino : Wino mass parameter.
    mu_soft : SUSY Higgs mass parameter, mu.
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    v_higgs : Higgs VEV.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    msN : Neutralino mass.
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmauu_neutralino : One-loop correction Sigma_u^u(neutralino).

    """
    sigmauu_neutralino = ((-1) / (16 * np.power(np.pi, 2))) \
        * (neutralinouu_deriv_num(m1_bino, m2_wino, mu_soft, g_coupling,
                                  g_prime, v_higgs, tanb, msN)
           / neutralino_deriv_denom(m1_bino, m2_wino, mu_soft, g_coupling,
                                    g_prime, v_higgs, tanb, msN))\
        * logfunc(msN, q_renorm)
    return sigmauu_neutralino


def sigmadd_neutralino(m1_bino, m2_wino, mu_soft, g_coupling, g_prime, v_higgs,
                       tanb, msN, q_renorm):
    """
    Return one-loop correction Sigma_d^d(neutralino).

    Parameters
    ----------
    m1_bino : Bino mass parameter.
    m2_wino : Wino mass parameter.
    mu_soft : SUSY Higgs mass parameter, mu.
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    v_higgs : Higgs VEV.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    msN : Neutralino mass.
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    Sigmadd_neutralino : One-loop correction Sigma_d^d(neutralino).

    """
    Sigmadd_neutralino = ((-1) / (16 * np.power(np.pi, 2))) \
        * (neutralinodd_deriv_num(m1_bino, m2_wino, mu_soft, g_coupling,
                                  g_prime, v_higgs, tanb, msN)
           / neutralino_deriv_denom(m1_bino, m2_wino, mu_soft, g_coupling,
                                    g_prime, v_higgs, tanb, msN))\
        * logfunc(msN, q_renorm)
    return Sigmadd_neutralino


#########################
# Charginos:
#########################


def sigmauu_chargino1(g_coupling, m2_wino, v_higgs, tanb, mu_soft, msC,
                      q_renorm):
    """
    Return one-loop correction Sigma_u^u(chargino_1).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    m2_wino : Wino mass parameter.
    v_higgs : Higgs VEV.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    mu_soft : SUSY Higgs mass parameter, mu.
    msC : Chargino mass.
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmauu_chargino1 : One-loop correction Sigma_u^u(chargino_1).

    """
    chargino_num = np.power(m2_wino, 2) + np.power(mu_soft, 2)\
        + (np.power(g_coupling, 2) * (np.power(v_higgs_u(v_higgs, tanb), 2)
                                      - np.power(v_higgs_d(v_higgs, tanb), 2)))
    chargino_den = np.sqrt(((np.power(g_coupling, 2)
                             * np.power((v_higgs_u(v_higgs, tanb)
                                         + v_higgs_d(v_higgs, tanb)), 2))
                            + np.power((m2_wino - mu_soft), 2))
                           * ((np.power(g_coupling, 2)
                               * np.power((v_higgs_d(v_higgs, tanb)
                                           - v_higgs_u(v_higgs, tanb)), 2))
                              + np.power((m2_wino + mu_soft), 2)))
    sigmauu_chargino1 = -1 * (np.power(g_coupling, 2) / (16 * np.power(np.pi,
                                                                       2)))\
        * (1 - (chargino_num / chargino_den)) * logfunc(msC, q_renorm)
    return sigmauu_chargino1


def sigmauu_chargino2(g_coupling, m2_wino, v_higgs, tanb, mu_soft, msC,
                      q_renorm):
    """
    Return one-loop correction Sigma_u^u(chargino_2).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    m2_wino : Wino mass parameter.
    v_higgs : Higgs VEV.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    mu_soft : SUSY Higgs mass parameter, mu.
    msC : Chargino mass.
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmauu_chargino2 : One-loop correction Sigma_u^u(chargino_2).

    """
    chargino_num = np.power(m2_wino, 2) + np.power(mu_soft, 2)\
        + (np.power(g_coupling, 2) * (np.power(v_higgs_u(v_higgs, tanb), 2)
                                      - np.power(v_higgs_d(v_higgs, tanb), 2)))
    chargino_den = np.sqrt(((np.power(g_coupling, 2)
                             * np.power((v_higgs_u(v_higgs, tanb)
                                         + v_higgs_d(v_higgs, tanb)), 2))
                            + np.power((m2_wino - mu_soft), 2))
                           * ((np.power(g_coupling, 2)
                               * np.power((v_higgs_d(v_higgs, tanb)
                                           - v_higgs_u(v_higgs, tanb)), 2))
                              + np.power((m2_wino + mu_soft), 2)))
    sigmauu_chargino2 = -1 * (np.power(g_coupling, 2) / (16 * np.power(np.pi,
                                                                       2)))\
        * (1 + (chargino_num / chargino_den)) * logfunc(msC, q_renorm)
    return sigmauu_chargino2


def sigmadd_chargino1(g_coupling, m2_wino, v_higgs, tanb, mu_soft, msC,
                      q_renorm):
    """
    Return one-loop correction Sigma_d^d(chargino_1).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    m2_wino : Wino mass parameter.
    v_higgs : Higgs VEV.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    mu_soft : SUSY Higgs mass parameter, mu.
    msC : Chargino mass.
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    Sigmadd_chargino1 : One-loop correction Sigma_d^d(chargino_1).

    """
    chargino_num = np.power(m2_wino, 2) + np.power(mu_soft, 2)\
        - (np.power(g_coupling, 2) * (np.power(v_higgs_u(v_higgs, tanb), 2)
                                      - np.power(v_higgs_d(v_higgs, tanb), 2)))
    chargino_den = np.sqrt(((np.power(g_coupling, 2)
                             * np.power((v_higgs_u(v_higgs, tanb)
                                         + v_higgs_d(v_higgs, tanb)), 2))
                            + np.power((m2_wino - mu_soft), 2))
                           * ((np.power(g_coupling, 2)
                               * np.power((v_higgs_d(v_higgs, tanb)
                                           - v_higgs_u(v_higgs, tanb)), 2))
                              + np.power((m2_wino + mu_soft), 2)))
    Sigmadd_chargino1 = -1 * (np.power(g_coupling, 2) / (16 * np.power(np.pi,
                                                                       2)))\
        * (1 - (chargino_num / chargino_den)) * logfunc(msC, q_renorm)
    return Sigmadd_chargino1


def sigmadd_chargino2(g_coupling, m2_wino, v_higgs, tanb, mu_soft, msC,
                      q_renorm):
    """
    Return one-loop correction Sigma_d^d(chargino_2).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    m2_wino : Wino mass parameter.
    v_higgs : Higgs VEV.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    mu_soft : SUSY Higgs mass parameter, mu.
    msC : Chargino mass.
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    Sigmadd_chargino2 : One-loop correction Sigma_d^d(chargino_2).

    """
    chargino_num = np.power(m2_wino, 2) + np.power(mu_soft, 2)\
        - (np.power(g_coupling, 2) * (np.power(v_higgs_u(v_higgs, tanb), 2)
                                      - np.power(v_higgs_d(v_higgs, tanb), 2)))
    chargino_den = np.sqrt(((np.power(g_coupling, 2)
                             * np.power((v_higgs_u(v_higgs, tanb)
                                         + v_higgs_d(v_higgs, tanb)), 2))
                            + np.power((m2_wino - mu_soft), 2))
                           * ((np.power(g_coupling, 2)
                               * np.power((v_higgs_d(v_higgs, tanb)
                                           - v_higgs_u(v_higgs, tanb)), 2))
                              + np.power((m2_wino + mu_soft), 2)))
    Sigmadd_chargino2 = -1 * (np.power(g_coupling, 2) / (16 * np.power(np.pi,
                                                                       2)))\
        * (1 + (chargino_num / chargino_den)) * logfunc(msC, q_renorm)
    return Sigmadd_chargino2


#########################
# Higgs bosons (sigmauu = sigmadd here):
#########################


def sigmauu_h0(g_coupling, g_prime, v_higgs, tanb, mHusq, mHdsq, mu_soft, mZ,
               mh0, q_renorm):
    """
    Return one-loop correction Sigma_u,d^u,d(h_0) (lighter neutral Higgs).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    v_higgs : Higgs VEV.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    mHusq : Squared up-type Higgs mass.
    mHdsq : Squared down-type Higgs mass.
    mu_soft : SUSY Higgs mass parameter, mu.
    mZ : Z boson mass.
    mh0 : Lighter neutral Higgs mass.
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmauu_h0 : One-loop correction Sigma_u,d^u,d(h_0) (lighter neutr. Higgs).

    """
    mynum = ((np.power(g_coupling, 2) + np.power(g_prime, 2))
             * np.power(v_higgs, 2))\
        - (2 * ma_0sq(mu_soft, mHusq, mHdsq) * (np.power(cossqb(tanb), 2)
                                                - (6 * cossqb(tanb)
                                                   * sinsqb(tanb))
                                                + np.power(sinsqb(tanb), 2)))
    myden = np.sqrt(np.power((ma_0sq(mu_soft, mHusq, mHdsq) - np.power(mZ, 2)),
                             2)
                    + (4 * np.power(mZ, 2) * ma_0sq(mu_soft, mHusq, mHdsq) * 4
                       * cossqb(tanb) * sinsqb(tanb)))
    sigmauu_h0 = (1 / (32 * np.power(np.pi, 2)))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 4)\
        * (1 - (mynum / myden)) * logfunc(mh0, q_renorm)
    return sigmauu_h0


def sigmauu_H0(g_coupling, g_prime, v_higgs, tanb, mHusq, mHdsq, mu_soft, mZ,
               mH0, q_renorm):
    """
    Return one-loop correction Sigma_u,d^u,d(H_0) (heavier neutral Higgs).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    v_higgs : Higgs VEV.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    mHusq : Squared up-type Higgs mass.
    mHdsq : Squared down-type Higgs mass.
    mu_soft : SUSY Higgs mass parameter, mu.
    mZ : Z boson mass.
    mH0 : Heavier neutral Higgs mass.
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmauu_H0 : One-loop correction Sigma_u,d^u,d(h_0) (heavier neut. Higgs).

    """
    mynum = ((np.power(g_coupling, 2) + np.power(g_prime, 2))
             * np.power(v_higgs, 2))\
        - (2 * ma_0sq(mu_soft, mHusq, mHdsq) * (np.power(cossqb(tanb), 2)
                                                - (6 * cossqb(tanb)
                                                   * sinsqb(tanb))
                                                + np.power(sinsqb(tanb), 2)))
    myden = np.sqrt(np.power((ma_0sq(mu_soft, mHusq, mHdsq) - np.power(mZ, 2)),
                             2)
                    + (4 * np.power(mZ, 2) * ma_0sq(mu_soft, mHusq, mHdsq)
                       * 4 * cossqb(tanb) * sinsqb(tanb)))
    sigmauu_H0 = (1/(32 * np.power(np.pi, 2)))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 4)\
        * (1 + (mynum / myden)) * logfunc(mH0, q_renorm)
    return sigmauu_H0


def sigmauu_H_pm(g_coupling, mH_pm, q_renorm):
    """
    Return one-loop correction Sigma_u,d^u,d(H_{+-}).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    mH_pm : Charged Higgs mass.
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmauu_H_pm : One-loop correction Sigma_u,d^u,d(H_{+-}).

    """
    sigmauu_H_pm = (np.power(g_coupling, 2) / (64 * np.power(np.pi, 2)))\
        * logfunc(mH_pm, q_renorm)
    return sigmauu_H_pm


#########################
# Weak bosons (sigmauu = sigmadd here):
#########################


def sigmauu_W_pm(g_coupling, v_higgs, q_renorm):
    """
    Return one-loop correction Sigma_u,d^u,d(W_{+-}).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    v_higgs : Higgs VEV.
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmauu_W_pm : One-loop correction Sigma_u,d^u,d(W_{+-}).

    """
    mymWsq = m_w_sq(g_coupling, v_higgs)
    sigmauu_W_pm = (3 * np.power(g_coupling, 2) / (32 * np.power(np.pi, 2)))\
        * logfunc(np.sqrt(mymWsq), q_renorm)
    return sigmauu_W_pm


def sigmauu_Z0(g_coupling, g_prime, v_higgs, q_renorm):
    """
    Return one-loop correction Sigma_u,d^u,d(Z_0).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    v_higgs : Higgs VEV.
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmauu_z0 : One-loop correction Sigma_u,d^u,d(Z_0).

    """
    mymzsq = mzsq(g_coupling, g_prime, v_higgs)
    sigmauu_z0 = (3 * (np.power(g_coupling, 2) + np.power(g_prime, 2))
                  / (64 * np.power(np.pi, 2)))\
        * logfunc(np.sqrt(mymzsq), q_renorm)
    return sigmauu_z0


#########################
# SM fermions (sigmadd_t = sigmauu_b = sigmauu_tau = 0):
#########################


def sigmauu_top(yt, v_higgs, tanb, q_renorm):
    """
    Return one-loop correction Sigma_u^u(top).

    Parameters
    ----------
    yt : Top Yukawa coupling.
    v_higgs : Higgs VEV.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    sigmauu_top : One-loop correction Sigma_u^u(top).

    """
    mymt = yt * v_higgs_u(v_higgs, tanb)
    sigmauu_top = ((-1) * np.power(yt, 2) / (16 * np.power(np.pi, 2)))\
        * logfunc(mymt, q_renorm)
    return sigmauu_top


def sigmadd_top():
    """Return one-loop correction Sigma_d^d(top) = 0."""
    return 0


def sigmauu_bottom():
    """Return one-loop correction Sigma_u^u(bottom) = 0."""
    return 0


def sigmadd_bottom(yb, v_higgs, tanb, q_renorm):
    """
    Return one-loop correction Sigma_d^d(bottom).

    Parameters
    ----------
    yb : Bottom Yukawa coupling.
    v_higgs : Higgs VEV.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    Sigmadd_bottom : One-loop correction Sigma_d^d(bottom).

    """
    mymb = yb * v_higgs_d(v_higgs, tanb)
    Sigmadd_bottom = (-1 * np.power(yb, 2) / (16 * np.power(np.pi, 2)))\
        * logfunc(mymb, q_renorm)
    return Sigmadd_bottom


def sigmauu_tau():
    """Return one-loop correction Sigma_u^u(tau) = 0."""
    return 0


def sigmadd_tau(ytau, v_higgs, tanb, q_renorm):
    """
    Return one-loop correction Sigma_d^d(tau).

    Parameters
    ----------
    ytau : Tau Yukawa coupling.
    v_higgs : Higgs VEV.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    Sigmadd_tau : One-loop correction Sigma_d^d(tau).

    """
    mymtau = ytau * v_higgs_d(v_higgs, tanb)
    Sigmadd_tau = (-1 * np.power(ytau, 2) / (16 * np.power(np.pi, 2)))\
        * logfunc(mymtau, q_renorm)
    return Sigmadd_tau


#########################
# DEW computation
#########################


def DEW_func(myinput, tanb):
    """
    Compute individual one-loop DEW contributions from Sigma_u,d^u,d.

    Parameters
    ----------
    myinput : One-loop correction to be inputted into the DEW function.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.

    Returns
    -------
    mycontrib : One-loop DEW contribution from Sigma_u,d^u,d(myinput).

    """
    mycontrib = np.absolute((np.absolute(myinput) / np.sqrt(1
                                                            - (4 * sinsqb(tanb)
                                                               * cossqb(tanb)))
                             ) - myinput) / 2
    return mycontrib


def DEW(v_higgs, mu_soft, tanb, y_t, y_b, y_tau, g_coupling, g_prime, m_stop_1,
        m_stop_2, m_sbot_1, m_sbot_2, m_stau_1, m_stau_2, mtl, mtr, mbl, mbr,
        mtaul, mtaur, msupl, msupr, msdownl, msdownr, mselecl, mselecr,
        mselec_sneut, msstrangel, msstranger, mscharml, mscharmr, msmul,
        msmur, msmu_sneut, msN1, msN2, msN3, msN4, msC1, msC2, mZ, mh0,
        mH0, mHusq, mHdsq, mH_pm, m1_bino, m2_wino, a_t, a_b, a_tau, q_renorm):
    """
    Return Delta_EW.

    Parameters
    ----------
    v_higgs : Higgs VEV.
    mu_soft : SUSY Higgs mass parameter, mu.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    y_t : Top Yukawa coupling.
    y_b : Bottom Yukawa coupling.
    y_tau : Tau Yukawa coupling.
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    m_stop_1 : Stop mass eigenstate mass 1.
    m_stop_2 : Stop mass eigenstate mass 2.
    m_sbot_1 : Sbottom mass eigenstate mass 1.
    m_sbot_2 : Sbottom mass eigenstate mass 2.
    m_stau_1 : Stau mass eigenstate mass 1.
    m_stau_2 : Stau mass eigenstate mass 2.
    mtl : Left gauge eigenstate stop mass.
    mtr : Right gauge eigenstate stop mass.
    mbl : Left gauge eigenstate sbottom mass.
    mbr : Right gauge eigenstate sbottom mass.
    mtaul : Left gauge eigenstate stau mass.
    mtaur : Right gauge eigenstate stau mass.
    msupl : Soft SUSY breaking mass for scalar up quark (left).
    msupr : Soft SUSY breaking mass for scalar up quark (right).
    msdownl : Soft SUSY breaking mass for scalar down quark (left).
    msdownr : Soft SUSY breaking mass for scalar down quark (right).
    mselecl : Soft SUSY breaking mass for scalar electron (left).
    mselecr : Soft SUSY breaking mass for scalar electron (right).
    mselec_sneut : Mass for scalar electron neutrino..
    msstrangel : Soft SUSY breaking mass for scalar strange quark (left).
    msstranger : Soft SUSY breaking mass for scalar strange quark (right).
    mscharml : Soft SUSY breaking mass for scalar charm quark (left).
    mscharmr : Soft SUSY breaking mass for scalar charm quark (right).
    msmul : Soft SUSY breaking mass for scalar muon (left).
    msmur : Soft SUSY breaking mass for scalar muon (right).
    msmu_sneut : Mass for scalar muon neutrino.
    msN1 : Neutralino mass 1.
    msN2 : Neutralino mass 2.
    msN3 : Neutralino mass 3.
    msN4 : Neutralino mass 4.
    msC1 : Chargino mass 1.
    msC2 : Chargino mass 2.
    mZ : Z boson mass.
    mh0 : Lighter neutral Higgs mass.
    mH0 : Heavier neutral Higgs mass.
    mHusq : Squared up-type Higgs mass..
    mHdsq : Squared down-type Higgs mass..
    mH_pm : Charged Higgs mass.
    m1_bino : Bino mass parameter.
    m2_wino : Wino mass parameter.
    a_t : Soft trilinear scalar top coupling.
    a_b : Soft trilinear scalar bottom coupling.
    a_tau : Soft trilinear scalar tau coupling.
    q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    mydew : Delta_EW based on all one-loop contributions, mu, and H_{u,d}.

    """
    cmu = np.absolute((-1) * np.power(mu_soft, 2))
    cHu = np.absolute((np.absolute((-1) * mHusq / np.sqrt(1
                                                          - (4 * sinsqb(tanb)
                                                             * cossqb(tanb)))))
                      - mHusq) / 2
    cHd = np.absolute((np.absolute(mHdsq / np.sqrt(1 - (4 * sinsqb(tanb)
                                                        * cossqb(tanb)))))
                      - mHdsq) / 2
    contribution_array = np.array([cmu, cHu, cHd,
                                   DEW_func(sigmadd_stop1(v_higgs, mu_soft,
                                                          tanb, y_t,
                                                          g_coupling, g_prime,
                                                          m_stop_1, m_stop_2,
                                                          mtl, mtr, a_t,
                                                          q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_stop2(v_higgs, mu_soft,
                                                          tanb, y_t,
                                                          g_coupling, g_prime,
                                                          m_stop_1, m_stop_2,
                                                          mtl, mtr, a_t,
                                                          q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_sbottom1(v_higgs, mu_soft,
                                                             tanb, y_b,
                                                             g_coupling,
                                                             g_prime, m_sbot_1,
                                                             m_sbot_2, mbl,
                                                             mbr, a_b,
                                                             q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_sbottom2(v_higgs,
                                                             mu_soft, tanb,
                                                             y_b, g_coupling,
                                                             g_prime,
                                                             m_sbot_1,
                                                             m_sbot_2,
                                                             mbl, mbr,
                                                             a_b,
                                                             q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_stau1(v_higgs, mu_soft,
                                                          tanb, y_tau,
                                                          g_coupling, g_prime,
                                                          m_stau_1, m_stau_2,
                                                          mtaul, mtaur, a_tau,
                                                          q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_stau2(v_higgs, mu_soft,
                                                          tanb, y_tau,
                                                          g_coupling,
                                                          g_prime, m_stau_1,
                                                          m_stau_2, mtaul,
                                                          mtaur, a_tau,
                                                          q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_sup_l(g_coupling, g_prime,
                                                          msupl, q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_sup_r(g_coupling, g_prime,
                                                          msupr, q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_sdown_l(g_coupling,
                                                            g_prime, msdownl,
                                                            q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_sdown_r(g_coupling,
                                                            g_prime, msdownr,
                                                            q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_selec_l(g_coupling,
                                                            g_prime, mselecl,
                                                            q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_selec_r(g_coupling,
                                                            g_prime, mselecr,
                                                            q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_selec_sneut(g_coupling,
                                                                g_prime,
                                                                mselec_sneut,
                                                                q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_sstrange_l(g_coupling,
                                                               g_prime,
                                                               msstrangel,
                                                               q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_sstrange_r(g_coupling,
                                                               g_prime,
                                                               msstranger,
                                                               q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_scharm_l(g_coupling,
                                                             g_prime, mscharml,
                                                             q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_scharm_r(g_coupling,
                                                             g_prime, mscharmr,
                                                             q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_smu_l(g_coupling, g_prime,
                                                          msmul, q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_smu_r(g_coupling, g_prime,
                                                          msmur, q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_smu_sneut(g_coupling,
                                                              g_prime,
                                                              msmu_sneut,
                                                              q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_neutralino(m1_bino,
                                                               m2_wino,
                                                               mu_soft,
                                                               g_coupling,
                                                               g_prime,
                                                               v_higgs, tanb,
                                                               msN1, q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_neutralino(m1_bino,
                                                               m2_wino,
                                                               mu_soft,
                                                               g_coupling,
                                                               g_prime,
                                                               v_higgs, tanb,
                                                               msN2, q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_neutralino(m1_bino,
                                                               m2_wino,
                                                               mu_soft,
                                                               g_coupling,
                                                               g_prime,
                                                               v_higgs, tanb,
                                                               msN3, q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_neutralino(m1_bino,
                                                               m2_wino,
                                                               mu_soft,
                                                               g_coupling,
                                                               g_prime,
                                                               v_higgs, tanb,
                                                               msN4, q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_chargino1(g_coupling,
                                                              m2_wino,
                                                              v_higgs, tanb,
                                                              mu_soft, msC1,
                                                              q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_chargino2(g_coupling,
                                                              m2_wino,
                                                              v_higgs, tanb,
                                                              mu_soft, msC2,
                                                              q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_h0(g_coupling, g_prime,
                                                       v_higgs, tanb, mHusq,
                                                       mHdsq, mu_soft, mZ, mh0,
                                                       q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_H0(g_coupling, g_prime,
                                                       v_higgs, tanb, mHusq,
                                                       mHdsq, mu_soft, mZ, mH0,
                                                       q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_H_pm(g_coupling, mH_pm,
                                                         q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_W_pm(g_coupling, v_higgs,
                                                         q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_Z0(g_coupling, g_prime,
                                                       v_higgs, q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_bottom(y_b, v_higgs, tanb,
                                                           q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_tau(y_tau, v_higgs, tanb,
                                                        q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_stop1(v_higgs, mu_soft,
                                                          tanb, y_t,
                                                          g_coupling,
                                                          g_prime, m_stop_1,
                                                          m_stop_2, mtl, mtr,
                                                          a_t, q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_stop2(v_higgs, mu_soft,
                                                          tanb, y_t,
                                                          g_coupling,
                                                          g_prime, m_stop_1,
                                                          m_stop_2, mtl, mtr,
                                                          a_t, q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_sbottom1(v_higgs, mu_soft,
                                                             tanb, y_b,
                                                             g_coupling,
                                                             g_prime, m_sbot_1,
                                                             m_sbot_2, mbl,
                                                             mbr, a_b,
                                                             q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_sbottom2(v_higgs, mu_soft,
                                                             tanb,
                                                             y_b, g_coupling,
                                                             g_prime, m_sbot_1,
                                                             m_sbot_2, mbl,
                                                             mbr, a_b,
                                                             q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_stau1(v_higgs, mu_soft,
                                                          tanb,
                                                          y_tau, g_coupling,
                                                          g_prime, m_stau_1,
                                                          m_stau_2, mtaul,
                                                          mtaur, a_tau,
                                                          q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_stau2(v_higgs, mu_soft,
                                                          tanb,
                                                          y_tau, g_coupling,
                                                          g_prime, m_stau_1,
                                                          m_stau_2, mtaul,
                                                          mtaur, a_tau,
                                                          q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_sup_l(g_coupling, g_prime,
                                                          msupl, q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_sup_r(g_coupling, g_prime,
                                                          msupr, q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_sdown_l(g_coupling,
                                                            g_prime, msdownl,
                                                            q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_sdown_r(g_coupling,
                                                            g_prime, msdownr,
                                                            q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_selec_l(g_coupling,
                                                            g_prime, mselecl,
                                                            q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_selec_r(g_coupling,
                                                            g_prime, mselecr,
                                                            q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_selec_sneut(g_coupling,
                                                                g_prime,
                                                                mselec_sneut,
                                                                q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_sstrange_l(g_coupling,
                                                               g_prime,
                                                               msstrangel,
                                                               q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_sstrange_r(g_coupling,
                                                               g_prime,
                                                               msstranger,
                                                               q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_scharm_l(g_coupling,
                                                             g_prime, mscharml,
                                                             q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_scharm_r(g_coupling,
                                                             g_prime, mscharmr,
                                                             q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_smu_l(g_coupling, g_prime,
                                                          msmul, q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_smu_r(g_coupling, g_prime,
                                                          msmur, q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_smu_sneut(g_coupling,
                                                              g_prime,
                                                              msmu_sneut,
                                                              q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_neutralino(m1_bino,
                                                               m2_wino,
                                                               mu_soft,
                                                               g_coupling,
                                                               g_prime,
                                                               v_higgs, tanb,
                                                               msN1, q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_neutralino(m1_bino,
                                                               m2_wino,
                                                               mu_soft,
                                                               g_coupling,
                                                               g_prime,
                                                               v_higgs, tanb,
                                                               msN2, q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_neutralino(m1_bino,
                                                               m2_wino,
                                                               mu_soft,
                                                               g_coupling,
                                                               g_prime,
                                                               v_higgs, tanb,
                                                               msN3, q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_neutralino(m1_bino,
                                                               m2_wino,
                                                               mu_soft,
                                                               g_coupling,
                                                               g_prime,
                                                               v_higgs, tanb,
                                                               msN4, q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_chargino1(g_coupling,
                                                              m2_wino, v_higgs,
                                                              tanb, mu_soft,
                                                              msC1, q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_chargino2(g_coupling,
                                                              m2_wino, v_higgs,
                                                              tanb, mu_soft,
                                                              msC2, q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_h0(g_coupling, g_prime,
                                                       v_higgs, tanb, mHusq,
                                                       mHdsq, mu_soft, mZ, mh0,
                                                       q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_H0(g_coupling, g_prime,
                                                       v_higgs, tanb, mHusq,
                                                       mHdsq, mu_soft, mZ, mH0,
                                                       q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_H_pm(g_coupling, mH_pm,
                                                         q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_W_pm(g_coupling, v_higgs,
                                                         q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_Z0(g_coupling, g_prime,
                                                       v_higgs, q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_top(y_t, v_higgs, tanb,
                                                        q_renorm),
                                            tanb)])
    mydew = (np.amax(contribution_array)) / (np.power(mZ, 2) / 2)
    return mydew


#########################
# SLHA input and definition of variables from SLHA file
#########################


direc = input('Enter the full directory for your SLHA file: ')
d = pyslha.read(direc)
vHiggs = d.blocks['HMIX'][3]  # Higgs VEV(Q) MSSM DRbar
mu = d.blocks['HMIX'][1]  # mu(Q) MSSM DRbar
tanb = d.blocks['HMIX'][2]  # tanb(Q) MSSM DRbar
y_t = d.blocks['YU'][3, 3]  # y_t(Q) MSSM DRbar
y_b = d.blocks['YD'][3, 3]  # y_b(Q) MSSM DRbar
y_tau = d.blocks['YE'][3, 3]  # y_tau(Q) MSSM DRbar
g_prime = d.blocks['GAUGE'][2]  # g'(Q) MSSM DRbar
g = d.blocks['GAUGE'][1]  # g(Q) MSSM DRbar
m_stop_1 = d.blocks['MASS'][1000006]  # m_stop_1
m_stop_2 = d.blocks['MASS'][2000006]  # m_stop_2
m_sbot_1 = d.blocks['MASS'][1000005]  # m_sbot_1
m_sbot_2 = d.blocks['MASS'][2000005]  # m_sbot_2
m_stau_1 = d.blocks['MASS'][1000015]  # m_stau_1
m_stau_2 = d.blocks['MASS'][2000015]  # m_stau_2
mtL = d.blocks['MSOFT'][43]  # m_~Q3_L(Q) MSSM DRbar
mtR = d.blocks['MSOFT'][46]  # m_stop_R(Q) MSSM DRbar
mbL = d.blocks['MSOFT'][43]  # m_~Q3_L(Q) MSSM DRbar
mbR = d.blocks['MSOFT'][49]  # m_sbot_R(Q) MSSM DRbar
mtauL = d.blocks['MSOFT'][33]  # m_stau_L(Q) MSSM DRbar
mtauR = d.blocks['MSOFT'][36]  # m_stau_R(Q) MSSM DRbar
msupL = d.blocks['MSOFT'][41]  # m_~Q1_L(Q) MSSM DRbar
msupR = d.blocks['MSOFT'][44]  # m_sup_R(Q) MSSM DRbar
msdownL = d.blocks['MSOFT'][41]  # m_~Q1_L(Q) MSSM DRbar
msdownR = d.blocks['MSOFT'][47]  # m_sdown_R(Q) MSSM DRbar
mselecL = d.blocks['MSOFT'][31]  # m_selec_L(Q) MSSM DRbar
mselecR = d.blocks['MSOFT'][34]  # m_selec_R(Q) MSSM DRbar
mselecSneut = d.blocks['MASS'][1000012]  # m_selecSneutrino_L
msstrangeL = d.blocks['MSOFT'][42]  # m_~Q2_L(Q) MSSM DRbar
msstrangeR = d.blocks['MSOFT'][48]  # m_sstrange_R(Q) MSSM DRbar
mscharmL = d.blocks['MSOFT'][42]  # m_~Q2_L(Q) MSSM DRbar
mscharmR = d.blocks['MSOFT'][45]  # m_scharm_R(Q) MSSM DRbar
msmuL = d.blocks['MSOFT'][32]  # m_smu_L(Q) MSSM DRbar
msmuR = d.blocks['MSOFT'][35]  # m_smu_R(Q) MSSM DRbar
msmuSneut = d.blocks['MASS'][1000014]  # m_smuSneutrino_L
msN1 = d.blocks['MASS'][1000022]  # m_Neutralino_1
msN2 = d.blocks['MASS'][1000023]  # m_Neutralino_2
msN3 = d.blocks['MASS'][1000025]  # m_Neutralino_3
msN4 = d.blocks['MASS'][1000035]  # m_Neutralino_4
msC1 = d.blocks['MASS'][1000024]  # m_Chargino_1
msC2 = d.blocks['MASS'][1000037]  # m_Chargino_2
mZ = d.blocks['SMINPUTS'][4]  # m_Z
mh0 = d.blocks['MASS'][25]  # m_h0
mH0 = d.blocks['MASS'][35]  # m_H0
mHusq = d.blocks['MSOFT'][22]  # m_Hu^2(Q) MSSM DRbar
mHdsq = d.blocks['MSOFT'][21]  # m_Hd^2(Q) MSSM DRbar
mH_pm = d.blocks['MASS'][37]  # m_H_+-
M1 = d.blocks['MSOFT'][1]  # M_1
M2 = d.blocks['MSOFT'][2]  # M_2
a_t = d.blocks['AU'][3, 3] * d.blocks['YU'][3, 3]  # a_t
a_b = d.blocks['AD'][3, 3] * d.blocks['YD'][3, 3]  # a_b
a_tau = d.blocks['AE'][3, 3] * d.blocks['YE'][3, 3]  # a_tau
Q_renorm = np.sqrt(m_stop_1 * m_stop_2)


halfmzsq = np.power(mZ, 2) / 2
cmu = np.absolute((-1) * np.power(mu, 2))
cHu = np.absolute((np.absolute((-1) * mHusq / np.sqrt(1
                                                      - (4 * sinsqb(tanb)
                                                         * cossqb(tanb)))))
                  - mHusq) / 2
cHd = np.absolute((np.absolute(mHdsq / np.sqrt(1 - (4 * sinsqb(tanb)
                                                    * cossqb(tanb)))))
                  - mHdsq) / 2
contribution_array = np.array([cmu, cHu, cHd,
                               DEW_func(sigmadd_stop1(vHiggs, mu,
                                                      tanb, y_t, g,
                                                      g_prime, m_stop_1,
                                                      m_stop_2, mtL, mtR,
                                                      a_t, Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_stop2(vHiggs, mu,
                                                      tanb, y_t, g,
                                                      g_prime, m_stop_1,
                                                      m_stop_2, mtL, mtR,
                                                      a_t, Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_sbottom1(vHiggs, mu,
                                                         tanb, y_b, g,
                                                         g_prime, m_sbot_1,
                                                         m_sbot_2, mbL,
                                                         mbR, a_b,
                                                         Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_sbottom2(vHiggs,
                                                         mu, tanb,
                                                         y_b, g,
                                                         g_prime,
                                                         m_sbot_1,
                                                         m_sbot_2,
                                                         mbL, mbR,
                                                         a_b,
                                                         Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_stau1(vHiggs, mu,
                                                      tanb, y_tau, g,
                                                      g_prime,
                                                      m_stau_1, m_stau_2,
                                                      mtauL, mtauR, a_tau,
                                                      Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_stau2(vHiggs, mu, tanb,
                                                      y_tau, g, g_prime,
                                                      m_stau_1, m_stau_2,
                                                      mtauL, mtauR, a_tau,
                                                      Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_sup_l(g, g_prime, msupL,
                                                      Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_sup_r(g, g_prime, msupR,
                                                      Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_sdown_l(g, g_prime,
                                                        msdownL,
                                                        Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_sdown_r(g, g_prime,
                                                        msdownR,
                                                        Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_selec_l(g, g_prime,
                                                        mselecL,
                                                        Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_selec_r(g, g_prime,
                                                        mselecR,
                                                        Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_selec_sneut(g, g_prime,
                                                            mselecSneut,
                                                            Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_sstrange_l(g, g_prime,
                                                           msstrangeL,
                                                           Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_sstrange_r(g, g_prime,
                                                           msstrangeR,
                                                           Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_scharm_l(g, g_prime,
                                                         mscharmL,
                                                         Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_scharm_r(g, g_prime,
                                                         mscharmR,
                                                         Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_smu_l(g, g_prime, msmuL,
                                                      Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_smu_r(g, g_prime, msmuR,
                                                      Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_smu_sneut(g, g_prime,
                                                          msmuSneut,
                                                          Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_neutralino(M1, M2, mu, g,
                                                           g_prime, vHiggs,
                                                           tanb, msN1,
                                                           Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_neutralino(M1, M2, mu, g,
                                                           g_prime, vHiggs,
                                                           tanb, msN2,
                                                           Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_neutralino(M1, M2, mu, g,
                                                           g_prime, vHiggs,
                                                           tanb, msN3,
                                                           Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_neutralino(M1, M2, mu, g,
                                                           g_prime, vHiggs,
                                                           tanb, msN4,
                                                           Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_chargino1(g, M2, vHiggs,
                                                          tanb, mu, msC1,
                                                          Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_chargino2(g, M2, vHiggs,
                                                          tanb, mu, msC2,
                                                          Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_h0(g, g_prime, vHiggs,
                                                   tanb, mHusq, mHdsq,
                                                   mu, mZ, mh0, Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_H0(g, g_prime, vHiggs,
                                                   tanb, mHusq, mHdsq,
                                                   mu, mZ, mH0, Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_H_pm(g, mH_pm, Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_W_pm(g, vHiggs, Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_Z0(g, g_prime, vHiggs,
                                                   Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_bottom(y_b, vHiggs, tanb,
                                                       Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_tau(y_tau, vHiggs, tanb,
                                                    Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_stop1(vHiggs, mu, tanb,
                                                      y_t, g, g_prime,
                                                      m_stop_1, m_stop_2,
                                                      mtL, mtR, a_t,
                                                      Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_stop2(vHiggs, mu, tanb,
                                                      y_t, g, g_prime,
                                                      m_stop_1, m_stop_2,
                                                      mtL, mtR, a_t,
                                                      Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_sbottom1(vHiggs, mu, tanb,
                                                         y_b, g, g_prime,
                                                         m_sbot_1,
                                                         m_sbot_2,
                                                         mbL, mbR, a_b,
                                                         Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_sbottom2(vHiggs, mu, tanb,
                                                         y_b, g, g_prime,
                                                         m_sbot_1,
                                                         m_sbot_2,
                                                         mbL, mbR, a_b,
                                                         Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_stau1(vHiggs, mu, tanb,
                                                      y_tau, g, g_prime,
                                                      m_stau_1, m_stau_2,
                                                      mtauL, mtauR,
                                                      a_tau, Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_stau2(vHiggs, mu, tanb,
                                                      y_tau, g, g_prime,
                                                      m_stau_1, m_stau_2,
                                                      mtauL, mtauR,
                                                      a_tau, Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_sup_l(g, g_prime, msupL,
                                                      Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_sup_r(g, g_prime, msupR,
                                                      Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_sdown_l(g, g_prime,
                                                        msdownL, Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_sdown_r(g, g_prime,
                                                        msdownR, Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_selec_l(g, g_prime,
                                                        mselecL, Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_selec_r(g, g_prime,
                                                        mselecR, Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_selec_sneut(g, g_prime,
                                                            mselecSneut,
                                                            Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_sstrange_l(g, g_prime,
                                                           msstrangeL,
                                                           Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_sstrange_r(g, g_prime,
                                                           msstrangeR,
                                                           Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_scharm_l(g, g_prime,
                                                         mscharmL,
                                                         Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_scharm_r(g, g_prime,
                                                         mscharmR,
                                                         Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_smu_l(g, g_prime, msmuL,
                                                      Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_smu_r(g, g_prime, msmuR,
                                                      Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_smu_sneut(g, g_prime,
                                                          msmuSneut,
                                                          Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_neutralino(M1, M2, mu, g,
                                                           g_prime, vHiggs,
                                                           tanb, msN1,
                                                           Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_neutralino(M1, M2, mu, g,
                                                           g_prime, vHiggs,
                                                           tanb, msN2,
                                                           Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_neutralino(M1, M2, mu, g,
                                                           g_prime, vHiggs,
                                                           tanb, msN3,
                                                           Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_neutralino(M1, M2, mu, g,
                                                           g_prime, vHiggs,
                                                           tanb, msN4,
                                                           Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_chargino1(g, M2, vHiggs,
                                                          tanb, mu, msC1,
                                                          Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_chargino2(g, M2, vHiggs,
                                                          tanb, mu, msC2,
                                                          Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_h0(g, g_prime, vHiggs,
                                                   tanb, mHusq, mHdsq,
                                                   mu, mZ, mh0, Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_H0(g, g_prime, vHiggs,
                                                   tanb, mHusq, mHdsq,
                                                   mu, mZ, mH0, Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_H_pm(g, mH_pm, Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_W_pm(g, vHiggs, Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_Z0(g, g_prime, vHiggs,
                                                   Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_top(y_t, vHiggs, tanb,
                                                    Q_renorm),
                                        tanb)])
labeled_contrib_array = np.array([(cmu / halfmzsq, 'mu'),
                                  (cHu / halfmzsq, 'H_u'),
                                  (cHd / halfmzsq, 'H_d'),
                                  (DEW_func(sigmadd_stop1(vHiggs, mu,
                                                          tanb, y_t, g,
                                                          g_prime,
                                                          m_stop_1,
                                                          m_stop_2, mtL,
                                                          mtR, a_t,
                                                          Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(stop_1)'),
                                  (DEW_func(sigmadd_stop2(vHiggs, mu,
                                                          tanb, y_t, g,
                                                          g_prime,
                                                          m_stop_1,
                                                          m_stop_2, mtL,
                                                          mtR,
                                                          a_t, Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(stop_2)'),
                                  (DEW_func(sigmadd_sbottom1(vHiggs, mu,
                                                             tanb, y_b, g,
                                                             g_prime, m_sbot_1,
                                                             m_sbot_2, mbL,
                                                             mbR, a_b,
                                                             Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(sbot_1)'),
                                  (DEW_func(sigmadd_sbottom2(vHiggs,
                                                             mu, tanb,
                                                             y_b, g,
                                                             g_prime,
                                                             m_sbot_1,
                                                             m_sbot_2,
                                                             mbL, mbR,
                                                             a_b,
                                                             Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(sbot_2)'),
                                  (DEW_func(sigmadd_stau1(vHiggs, mu,
                                                          tanb, y_tau, g,
                                                          g_prime,
                                                          m_stau_1, m_stau_2,
                                                          mtauL, mtauR, a_tau,
                                                          Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(stau_1)'),
                                  (DEW_func(sigmadd_stau2(vHiggs, mu, tanb,
                                                          y_tau, g, g_prime,
                                                          m_stau_1, m_stau_2,
                                                          mtauL, mtauR, a_tau,
                                                          Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(stau_2)'),
                                  (DEW_func(sigmadd_sup_l(g, g_prime, msupL,
                                                          Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(sup_L)'),
                                  (DEW_func(sigmadd_sup_r(g, g_prime, msupR,
                                                          Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(sup_R)'),
                                  (DEW_func(sigmadd_sdown_l(g, g_prime,
                                                            msdownL,
                                                            Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(sdown_L)'),
                                  (DEW_func(sigmadd_sdown_r(g, g_prime,
                                                            msdownR,
                                                            Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(sdown_R)'),
                                  (DEW_func(sigmadd_selec_l(g, g_prime,
                                                            mselecL,
                                                            Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(selec_L)'),
                                  (DEW_func(sigmadd_selec_r(g, g_prime,
                                                            mselecR,
                                                            Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(selec_R)'),
                                  (DEW_func(sigmadd_selec_sneut(g, g_prime,
                                                                mselecSneut,
                                                                Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(selec_neutr)'),
                                  (DEW_func(sigmadd_sstrange_l(g, g_prime,
                                                               msstrangeL,
                                                               Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(sstrange_L)'),
                                  (DEW_func(sigmadd_sstrange_r(g, g_prime,
                                                               msstrangeR,
                                                               Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(sstrange_R)'),
                                  (DEW_func(sigmadd_scharm_l(g, g_prime,
                                                             mscharmL,
                                                             Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(scharm_L)'),
                                  (DEW_func(sigmadd_scharm_r(g, g_prime,
                                                             mscharmR,
                                                             Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(scharm_R)'),
                                  (DEW_func(sigmadd_smu_l(g, g_prime, msmuL,
                                                          Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(smu_L)'),
                                  (DEW_func(sigmadd_smu_r(g, g_prime, msmuR,
                                                          Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(smu_R)'),
                                  (DEW_func(sigmadd_smu_sneut(g, g_prime,
                                                              msmuSneut,
                                                              Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(smu_neutr)'),
                                  (DEW_func(sigmadd_neutralino(M1, M2, mu, g,
                                                               g_prime, vHiggs,
                                                               tanb, msN1,
                                                               Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(neutralino_1)'),
                                  (DEW_func(sigmadd_neutralino(M1, M2, mu, g,
                                                               g_prime, vHiggs,
                                                               tanb, msN2,
                                                               Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(neutralino_2)'),
                                  (DEW_func(sigmadd_neutralino(M1, M2, mu, g,
                                                               g_prime, vHiggs,
                                                               tanb, msN3,
                                                               Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(neutralino_3)'),
                                  (DEW_func(sigmadd_neutralino(M1, M2, mu, g,
                                                               g_prime, vHiggs,
                                                               tanb, msN4,
                                                               Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(neutralino_4)'),
                                  (DEW_func(sigmadd_chargino1(g, M2, vHiggs,
                                                              tanb, mu, msC1,
                                                              Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(chargino_1)'),
                                  (DEW_func(sigmadd_chargino2(g, M2, vHiggs,
                                                              tanb, mu, msC2,
                                                              Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(chargino_2)'),
                                  (DEW_func(sigmauu_h0(g, g_prime, vHiggs,
                                                       tanb, mHusq, mHdsq,
                                                       mu, mZ, mh0, Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(h_0)'),
                                  (DEW_func(sigmauu_H0(g, g_prime, vHiggs,
                                                       tanb, mHusq, mHdsq,
                                                       mu, mZ, mH0, Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(H_0)'),
                                  (DEW_func(sigmauu_H_pm(g, mH_pm, Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(H_+-)'),
                                  (DEW_func(sigmauu_W_pm(g, vHiggs, Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(W_+-)'),
                                  (DEW_func(sigmauu_Z0(g, g_prime, vHiggs,
                                                       Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(Z_0)'),
                                  (DEW_func(sigmadd_bottom(y_b, vHiggs, tanb,
                                                           Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(bottom)'),
                                  (DEW_func(sigmadd_tau(y_tau, vHiggs, tanb,
                                                        Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(tau)'),
                                  (DEW_func(sigmauu_stop1(vHiggs, mu, tanb,
                                                          y_t, g, g_prime,
                                                          m_stop_1, m_stop_2,
                                                          mtL, mtR, a_t,
                                                          Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(stop_1)'),
                                  (DEW_func(sigmauu_stop2(vHiggs, mu, tanb,
                                                          y_t, g, g_prime,
                                                          m_stop_1, m_stop_2,
                                                          mtL, mtR, a_t,
                                                          Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(stop_2)'),
                                  (DEW_func(sigmauu_sbottom1(vHiggs, mu, tanb,
                                                             y_b, g, g_prime,
                                                             m_sbot_1,
                                                             m_sbot_2,
                                                             mbL, mbR, a_b,
                                                             Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(sbot_1)'),
                                  (DEW_func(sigmauu_sbottom2(vHiggs, mu, tanb,
                                                             y_b, g, g_prime,
                                                             m_sbot_1,
                                                             m_sbot_2,
                                                             mbL, mbR, a_b,
                                                             Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(sbot_2)'),
                                  (DEW_func(sigmauu_stau1(vHiggs, mu, tanb,
                                                          y_tau, g, g_prime,
                                                          m_stau_1, m_stau_2,
                                                          mtauL, mtauR,
                                                          a_tau, Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(stau_1)'),
                                  (DEW_func(sigmauu_stau2(vHiggs, mu, tanb,
                                                          y_tau, g, g_prime,
                                                          m_stau_1, m_stau_2,
                                                          mtauL, mtauR,
                                                          a_tau, Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(stau_2)'),
                                  (DEW_func(sigmauu_sup_l(g, g_prime, msupL,
                                                          Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(sup_L)'),
                                  (DEW_func(sigmauu_sup_r(g, g_prime, msupR,
                                                          Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(sup_R)'),
                                  (DEW_func(sigmauu_sdown_l(g, g_prime,
                                                            msdownL, Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(sdown_L)'),
                                  (DEW_func(sigmauu_sdown_r(g, g_prime,
                                                            msdownR, Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(sdown_R)'),
                                  (DEW_func(sigmauu_selec_l(g, g_prime,
                                                            mselecL, Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(selec_L)'),
                                  (DEW_func(sigmauu_selec_r(g, g_prime,
                                                            mselecR, Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(selec_R)'),
                                  (DEW_func(sigmauu_selec_sneut(g, g_prime,
                                                                mselecSneut,
                                                                Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(selec_neutr)'),
                                  (DEW_func(sigmauu_sstrange_l(g, g_prime,
                                                               msstrangeL,
                                                               Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(sstrange_L)'),
                                  (DEW_func(sigmauu_sstrange_r(g, g_prime,
                                                               msstrangeR,
                                                               Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(sstrange_R)'),
                                  (DEW_func(sigmauu_scharm_l(g, g_prime,
                                                             mscharmL,
                                                             Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(scharm_L)'),
                                  (DEW_func(sigmauu_scharm_r(g, g_prime,
                                                             mscharmR,
                                                             Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(scharm_R)'),
                                  (DEW_func(sigmauu_smu_l(g, g_prime, msmuL,
                                                          Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(smu_L)'),
                                  (DEW_func(sigmauu_smu_r(g, g_prime, msmuR,
                                                          Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(smu_R)'),
                                  (DEW_func(sigmauu_smu_sneut(g, g_prime,
                                                              msmuSneut,
                                                              Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(smu_neutr)'),
                                  (DEW_func(sigmauu_neutralino(M1, M2, mu, g,
                                                               g_prime, vHiggs,
                                                               tanb, msN1,
                                                               Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(neutralino_1)'),
                                  (DEW_func(sigmauu_neutralino(M1, M2, mu, g,
                                                               g_prime, vHiggs,
                                                               tanb, msN2,
                                                               Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(neutralino_2)'),
                                  (DEW_func(sigmauu_neutralino(M1, M2, mu, g,
                                                               g_prime, vHiggs,
                                                               tanb, msN3,
                                                               Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(neutralino_3)'),
                                  (DEW_func(sigmauu_neutralino(M1, M2, mu, g,
                                                               g_prime, vHiggs,
                                                               tanb, msN4,
                                                               Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(neutralino_4)'),
                                  (DEW_func(sigmauu_chargino1(g, M2, vHiggs,
                                                              tanb, mu, msC1,
                                                              Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(chargino_1)'),
                                  (DEW_func(sigmauu_chargino2(g, M2, vHiggs,
                                                              tanb, mu, msC2,
                                                              Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(chargino_2)'),
                                  (DEW_func(sigmauu_h0(g, g_prime, vHiggs,
                                                       tanb, mHusq, mHdsq,
                                                       mu, mZ, mh0, Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(h_0)'),
                                  (DEW_func(sigmauu_H0(g, g_prime, vHiggs,
                                                       tanb, mHusq, mHdsq,
                                                       mu, mZ, mH0, Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(H_0)'),
                                  (DEW_func(sigmauu_H_pm(g, mH_pm, Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(H_+-)'),
                                  (DEW_func(sigmauu_W_pm(g, vHiggs, Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(W_+-)'),
                                  (DEW_func(sigmauu_Z0(g, g_prime, vHiggs,
                                                       Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(Z_0)'),
                                  (DEW_func(sigmauu_top(y_t, vHiggs, tanb,
                                                        Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(top)')],
                                 dtype=[('Contribution', float),
                                        ('label', 'U30')])
sorted_array = np.sort(labeled_contrib_array, order='Contribution')
reverse_sorted_array = sorted_array[::-1]
mydew = (np.amax(contribution_array)) / (np.power(mZ, 2) / 2)
print('\nGiven the submitted SLHA file, your value for the electroweak'
      + ' naturalness measure, Delta_EW, is: ' + str(mydew))
print('\nThe top ten contributions to Delta_EW are as follows (decr. order): ')
print('')
print(reverse_sorted_array[0], ',')
print(reverse_sorted_array[1], ',')
print(reverse_sorted_array[2], ',')
print(reverse_sorted_array[3], ',')
print(reverse_sorted_array[4], ',')
print(reverse_sorted_array[5], ',')
print(reverse_sorted_array[6], ',')
print(reverse_sorted_array[7], ',')
print(reverse_sorted_array[8], ',')
print(reverse_sorted_array[9], ',')
