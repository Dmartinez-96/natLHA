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


def mZsq(g_coupling, g_prime, v_higgs):
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


def mA0sq(mu, mHusq, mHdsq):
    """
    Return A_0 squared mass.

    Parameters
    ----------
    mu : SUSY Higgs mass parameter, mu.
    mHusq : Squared up-type Higgs mass.
    mHdsq : Squared down-type Higgs mass.

    Returns
    -------
    mymA0sq : A_0 squared mass.

    """
    my_ma0_sq = 2 * np.power(np.abs(mu), 2) + mHusq + mHdsq
    return my_ma0_sq


def mHpmsq(mu, mHusq, mHdsq, g_coupling, v_higgs):
    """
    Return Higgs_{+-} squared mass.

    Parameters
    ----------
    mu : SUSY Higgs mass parameter, mu.
    mHusq : Squared up-type Higgs mass.
    mHdsq : Squared down-type Higgs mass.
    g_coupling : Electroweak coupling constant g.
    v_higgs : Higgs VEV.

    Returns
    -------
    mymHpmsq : Higgs_{+-} squared mass.

    """
    mymWsq = m_w_sq(g_coupling, v_higgs)
    mymHpmsq = mA0sq(mu, mHusq, mHdsq) + mymWsq
    return mymHpmsq


#########################
# Fundamental equations
#########################


def F(m, Q_renorm):
    """
    Return F = m^2 * (ln(m^2 / Q^2) - 1).

    Parameters
    ----------
    m : Input mass.
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    myF : F = m^2 * (ln(m^2 / Q^2) - 1).

    """
    myF = np.power(m, 2) * (np.log((np.power(m, 2))
                                   / (np.power(Q_renorm, 2))) - 1)
    return myF


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


def vu(v_higgs, tanb):
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


def vd(v_higgs, tanb):
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


def tan_theta_W(g_coupling, g_prime):
    """
    Return tan(theta_W), the Weinberg angle.

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.

    Returns
    -------
    mytanthetaW : Ratio of coupling constants.

    """
    mytanthetaW = g_prime / g_coupling
    return mytanthetaW


def sin_squared_theta_W(g_coupling, g_prime):
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
    mysinsqthetaW = (np.power(tan_theta_W(g_coupling, g_prime), 2)
                     / (1 + np.power(tan_theta_W(g_coupling, g_prime), 2)))
    return mysinsqthetaW


def cos_squared_theta_W(g_coupling, g_prime):
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
    mycossqthetaW = 1 - (np.power(tan_theta_W(g_coupling, g_prime), 2)
                         / (1 + np.power(tan_theta_W(g_coupling, g_prime), 2)))
    return mycossqthetaW


#########################
# Stop squarks:
#########################


def sigmauu_stop1(v_higgs, mu, tanb, y_t, g_coupling, g_prime, m_stop_1,
                  m_stop_2, mtL, mtR, a_t, Q_renorm):
    """
    Return one-loop correction Sigma_u^u(stop_1).

    Parameters
    ----------
    v_higgs : Higgs VEV.
    mu : SUSY Higgs mass parameter, mu.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    y_t : Top Yukawa coupling.
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    m_stop_1 : Stop mass eigenstate mass 1.
    m_stop_2 : Stop mass eigenstate mass 2.
    mtL : Left gauge eigenstate stop mass.
    mtR : Right gauge eigenstate stop mass.
    a_t : Soft trilinear scalar top coupling.
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    Sigmauu_stop1 : One-loop correction Sigma_u^u(stop_1).

    """
    delta_uL = ((1 / 2) - (2 / 3) * sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(v_higgs, tanb), 2) - np.power(vu(v_higgs, tanb), 2))
    delta_uR = (2 / 3) * sin_squared_theta_W(g_coupling, g_prime)\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(v_higgs, tanb), 2) - np.power(vu(v_higgs, tanb), 2))
    stop_num = ((np.power(mtL, 2) - np.power(mtR, 2) + delta_uL - delta_uR)
                * ((-1 / 2) + ((4 / 3)
                               * sin_squared_theta_W(g_coupling, g_prime)))
                * (np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        + 2 * np.power(a_t, 2)
    Sigmauu_stop1 = (3 / (32 * (np.power(np.pi, 2)))) *\
                    (2 * np.power(y_t, 2) + ((np.power(g_coupling, 2)
                                              + np.power(g_prime, 2))
                                             * (8 *
                                                sin_squared_theta_W(g_coupling,
                                                                    g_prime)
                                                - 3) / 12)
                     - (stop_num / (np.power(m_stop_2, 2)
                                    - np.power(m_stop_1, 2))))\
        * F(m_stop_1, Q_renorm)
    return Sigmauu_stop1


def sigmadd_stop1(v_higgs, mu, tanb, y_t, g_coupling, g_prime, m_stop_1,
                  m_stop_2, mtL, mtR, a_t, Q_renorm):
    """
    Return one-loop correction Sigma_d^d(stop_1).

    Parameters
    ----------
    v_higgs : Higgs VEV.
    mu : SUSY Higgs mass parameter, mu.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    y_t : Top Yukawa coupling.
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    m_stop_1 : Stop mass eigenstate mass 1.
    m_stop_2 : Stop mass eigenstate mass 2.
    mtL : Left gauge eigenstate stop mass.
    mtR : Right gauge eigenstate stop mass.
    a_t : Soft trilinear scalar top coupling.
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    Sigmadd_stop1 : One-loop correction Sigma_d^d(stop_1).

    """
    delta_uL = ((1 / 2) - (2 / 3) * sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(v_higgs, tanb), 2) - np.power(vu(v_higgs, tanb), 2))
    delta_uR = (2 / 3) * sin_squared_theta_W(g, g_prime)\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(v_higgs, tanb), 2) - np.power(vu(v_higgs, tanb), 2))
    stop_num = ((np.power(mtL, 2) - np.power(mtR, 2) + delta_uL - delta_uR)
                * ((1 / 2) + (4 / 3) * sin_squared_theta_W(g_coupling,
                                                           g_prime))
                * (np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        + 2 * np.power(y_t, 2) * np.power(mu, 2)
    Sigmadd_stop1 = (3 / (32 * (np.power(np.pi, 2))))\
        * (((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)
           - (stop_num / (np.power(m_stop_2, 2) - np.power(m_stop_1, 2))))\
        * F(m_stop_1, Q_renorm)
    return Sigmadd_stop1


def sigmauu_stop2(v_higgs, mu, tanb, y_t, g_coupling, g_prime, m_stop_1,
                  m_stop_2, mtL, mtR, a_t, Q_renorm):
    """
    Return one-loop correction Sigma_u^u(stop_2).

    Parameters
    ----------
    v_higgs : Higgs VEV.
    mu : SUSY Higgs mass parameter, mu.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    y_t : Top Yukawa coupling.
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    m_stop_1 : Stop mass eigenstate mass 1.
    m_stop_2 : Stop mass eigenstate mass 2.
    mtL : Left gauge eigenstate stop mass.
    mtR : Right gauge eigenstate stop mass.
    a_t : Soft trilinear scalar top coupling.
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    Sigmauu_stop2 : One-loop correction Sigma_u^u(stop_2).

    """
    delta_uL = ((1 / 2) - (2 / 3) * sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(v_higgs, tanb), 2) - np.power(vu(v_higgs, tanb), 2))
    delta_uR = (2 / 3) * sin_squared_theta_W(g_coupling, g_prime)\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(v_higgs, tanb), 2) - np.power(vu(v_higgs, tanb), 2))
    stop_num = ((np.power(mtL, 2) - np.power(mtR, 2) + delta_uL - delta_uR)
                * ((-1 / 2) + ((4 / 3) * sin_squared_theta_W(g_coupling,
                                                             g_prime)))
                * (np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        + 2 * np.power(a_t, 2)
    Sigmauu_stop2 = (3 / (32 * (np.power(np.pi, 2))))\
        * (2 * np.power(y_t, 2) + ((np.power(g_coupling, 2)
                                    + np.power(g_prime, 2))
                                   * (8 * sin_squared_theta_W(g_coupling,
                                                              g_prime) - 3)
                                   / 12)
           + (stop_num / (np.power(m_stop_2, 2) - np.power(m_stop_1, 2))))\
        * F(m_stop_2, Q_renorm)
    return Sigmauu_stop2


def sigmadd_stop2(v_higgs, mu, tanb, y_t, g_coupling, g_prime, m_stop_1,
                  m_stop_2, mtL, mtR, a_t, Q_renorm):
    """
    Return one-loop correction Sigma_d^d(stop_2).

    Parameters
    ----------
    v_higgs : Higgs VEV.
    mu : SUSY Higgs mass parameter, mu.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    y_t : Top Yukawa coupling.
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    m_stop_1 : Stop mass eigenstate mass 1.
    m_stop_2 : Stop mass eigenstate mass 2.
    mtL : Left gauge eigenstate stop mass.
    mtR : Right gauge eigenstate stop mass.
    a_t : Soft trilinear scalar top coupling.
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    Sigmadd_stop2 : One-loop correction Sigma_d^d(stop_2).

    """
    delta_uL = ((1 / 2) - (2 / 3) * sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(v_higgs, tanb), 2) - np.power(vu(v_higgs, tanb), 2))
    delta_uR = (2 / 3) * sin_squared_theta_W(g_coupling, g_prime)\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(v_higgs, tanb), 2) - np.power(vu(v_higgs, tanb), 2))
    stop_num = ((np.power(mtL, 2) - np.power(mtR, 2) + delta_uL - delta_uR)
                * ((1 / 2) + (4 / 3) * sin_squared_theta_W(g_coupling,
                                                           g_prime))
                * (np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        + 2 * np.power(y_t, 2) * np.power(mu, 2)
    Sigmadd_stop2 = (3 / (32 * (np.power(np.pi, 2))))\
        * (((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)
           - (stop_num / (np.power(m_stop_2, 2) - np.power(m_stop_1, 2))))\
        * F(m_stop_2, Q_renorm)
    return Sigmadd_stop2


#########################
# Sbottom squarks:
#########################


def sigmauu_sbottom1(v_higgs, mu, tanb, y_b, g_coupling, g_prime,
                     m_sbot_1, m_sbot_2, mbL, mbR, a_b, Q_renorm):
    """
    Return one-loop correction Sigma_u^u(sbottom_1).

    Parameters
    ----------
    v_higgs : Higgs VEV.
    mu : SUSY Higgs mass parameter, mu.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    y_b : Bottom Yukawa coupling.
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    m_sbot_1 : Sbottom mass eigenstate mass 1.
    m_sbot_2 : Sbottom mass eigenstate mass 2.
    mbL : Left gauge eigenstate sbottom mass.
    mbR : Right gauge eigenstate sbottom mass.
    a_b : Soft trilinear scalar bottom coupling.
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    Sigmauu_sbot : One-loop correction Sigma_u^u(sbottom_1).

    """
    delta_dL = ((-1 / 2) + (1 / 3) * sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(v_higgs, tanb), 2) - np.power(vu(v_higgs, tanb), 2))
    delta_dR = ((-1 / 3) * sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(v_higgs, tanb), 2) - np.power(vu(v_higgs, tanb), 2))
    sbot_num = (np.power(mbR, 2) - np.power(mbL, 2) + delta_dR - delta_dL)\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * ((1 / 2) + (2 / 3) * sin_squared_theta_W(g_coupling, g_prime))\
        + 2 * np.power(y_b, 2) * np.power(mu, 2)
    Sigmauu_sbot = (3 / (32 * np.power(np.pi, 2)))\
        * (((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 4)
           - (sbot_num / (np.power(m_sbot_2, 2) - np.power(m_sbot_1, 2))))\
        * F(m_sbot_1, Q_renorm)
    return Sigmauu_sbot


def sigmauu_sbottom2(v_higgs, mu, tanb, y_b, g_coupling, g_prime,
                     m_sbot_1, m_sbot_2, mbL, mbR, a_b, Q_renorm):
    """
    Return one-loop correction Sigma_u^u(sbottom_2).

    Parameters
    ----------
    v_higgs : Higgs VEV.
    mu : SUSY Higgs mass parameter, mu.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    y_b : Bottom Yukawa coupling.
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    m_sbot_1 : Sbottom mass eigenstate mass 1.
    m_sbot_2 : Sbottom mass eigenstate mass 2.
    mbL : Left gauge eigenstate sbottom mass.
    mbR : Right gauge eigenstate sbottom mass.
    a_b : Soft trilinear scalar bottom coupling.
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    Sigmauu_sbot : One-loop correction Sigma_u^u(sbottom_2).

    """
    delta_dL = ((-1 / 2) + (1 / 3) * sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(v_higgs, tanb), 2) - np.power(vu(v_higgs, tanb), 2))
    delta_dR = ((-1 / 3) * sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(v_higgs, tanb), 2) - np.power(vu(v_higgs, tanb), 2))
    sbot_num = (np.power(mbR, 2) - np.power(mbL, 2) + delta_dR - delta_dL)\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * ((1 / 2) + (2 / 3) * sin_squared_theta_W(g_coupling, g_prime))\
        + 2 * np.power(y_b, 2) * np.power(mu, 2)
    Sigmauu_sbot = (3 / (32 * np.power(np.pi, 2)))\
        * (((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 4)
           + (sbot_num / (np.power(m_sbot_2, 2) - np.power(m_sbot_1, 2))))\
        * F(m_sbot_2, Q_renorm)
    return Sigmauu_sbot


def sigmadd_sbottom1(v_higgs, mu, tanb, y_b, g_coupling, g_prime,
                     m_sbot_1, m_sbot_2, mbL, mbR, a_b, Q_renorm):
    """
    Return one-loop correction Sigma_d^d(sbottom_1).

    Parameters
    ----------
    v_higgs : Higgs VEV.
    mu : SUSY Higgs mass parameter, mu.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    y_b : Bottom Yukawa coupling.
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    m_sbot_1 : Sbottom mass eigenstate mass 1.
    m_sbot_2 : Sbottom mass eigenstate mass 2.
    mbL : Left gauge eigenstate sbottom mass.
    mbR : Right gauge eigenstate sbottom mass.
    a_b : Soft trilinear scalar bottom coupling.
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    Sigmadd_sbot : One-loop correction Sigma_d^d(sbottom_1).

    """
    delta_dL = ((-1 / 2) + (1 / 3) * sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(v_higgs, tanb), 2) - np.power(vu(v_higgs, tanb), 2))
    delta_dR = ((-1 / 3) * sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(v_higgs, tanb), 2) - np.power(vu(v_higgs, tanb), 2))
    sbot_num = (np.power(mbR, 2) - np.power(mbL, 2) + delta_dR - delta_dL)\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * ((-1 / 2) - (2 / 3) * sin_squared_theta_W(g_coupling, g_prime))\
        + 2 * np.power(a_b, 2)
    Sigmadd_sbot = (3 / (32 * np.power(np.pi, 2)))\
        * (((-1) * (np.power(g_coupling, 2) + np.power(g_prime, 2)) / 4)
           - (sbot_num / (np.power(m_sbot_2, 2) - np.power(m_sbot_1, 2))))\
        * F(m_sbot_1, Q_renorm)
    return Sigmadd_sbot


def sigmadd_sbottom2(v_higgs, mu, tanb, y_b, g_coupling, g_prime, m_sbot_1,
                     m_sbot_2, mbL, mbR, a_b, Q_renorm):
    """
    Return one-loop correction Sigma_d^d(sbottom_2).

    Parameters
    ----------
    v_higgs : Higgs VEV.
    mu : SUSY Higgs mass parameter, mu.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    y_b : Bottom Yukawa coupling.
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    m_sbot_1 : Sbottom mass eigenstate mass 1.
    m_sbot_2 : Sbottom mass eigenstate mass 2.
    mbL : Left gauge eigenstate sbottom mass.
    mbR : Right gauge eigenstate sbottom mass.
    a_b : Soft trilinear scalar bottom coupling.
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    Sigmadd_sbot : One-loop correction Sigma_d^d(sbottom_2).

    """
    delta_dL = ((-1 / 2) + (1 / 3) * sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(v_higgs, tanb), 2) - np.power(vu(v_higgs, tanb), 2))
    delta_dR = ((-1 / 3) * sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(v_higgs, tanb), 2) - np.power(vu(v_higgs, tanb), 2))
    sbot_num = (np.power(mbR, 2) - np.power(mbL, 2) + delta_dR - delta_dL)\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * ((-1 / 2) - (2 / 3) * sin_squared_theta_W(g_coupling, g_prime))\
        + 2 * np.power(a_b, 2)
    Sigmadd_sbot = (3 / (32 * np.power(np.pi, 2)))\
        * (((-1) * (np.power(g_coupling, 2) + np.power(g_prime, 2)) / 4)
           + (sbot_num / (np.power(m_sbot_2, 2) - np.power(m_sbot_1, 2))))\
        * F(m_sbot_2, Q_renorm)
    return Sigmadd_sbot


#########################
# Stau sleptons:
#########################


def sigmauu_stau1(v_higgs, mu, tanb, y_tau, g_coupling, g_prime, m_stau_1,
                  m_stau_2, mtauL, mtauR, a_tau, Q_renorm):
    """
    Return one-loop correction Sigma_u^u(stau_1).

    Parameters
    ----------
    v_higgs : Higgs VEV.
    mu : SUSY Higgs mass parameter, mu.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    y_tau : Tau Yukawa coupling.
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    m_stau_1 : Stau mass eigenstate mass 1.
    m_stau_2 : Stau mass eigenstate mass 2.
    mtauL : Left gauge eigenstate stau mass.
    mtauR : Right gauge eigenstate stau mass.
    a_tau : Soft trilinear scalar tau coupling.
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    Sigmauu_stau : One-loop correction Sigma_u^u(stau_1).

    """
    delta_eL = ((-1 / 2) + sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(v_higgs, tanb), 2) - np.power(vu(v_higgs, tanb), 2))
    delta_eR = ((-1) * sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(v_higgs, tanb), 2) - np.power(vu(v_higgs, tanb), 2))
    stau_num = (np.power(mtauR, 2) - np.power(mtauL, 2) + delta_eR - delta_eL)\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * ((-1 / 2) + 2 * sin_squared_theta_W(g_coupling, g_prime))\
        + 2 * np.power(y_tau, 2) * np.power(mu, 2)
    Sigmauu_stau = (1 / (32 * np.power(np.pi, 2)))\
        * (((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 4)
           - (stau_num / (np.power(m_stau_2, 2) - np.power(m_stau_1, 2))))\
        * F(m_stau_1, Q_renorm)
    return Sigmauu_stau


def sigmauu_stau2(v_higgs, mu, tanb, y_tau, g_coupling, g_prime, m_stau_1,
                  m_stau_2, mtauL, mtauR, a_tau, Q_renorm):
    """
    Return one-loop correction Sigma_u^u(stau_2).

    Parameters
    ----------
    v_higgs : Higgs VEV.
    mu : SUSY Higgs mass parameter, mu.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    y_tau : Tau Yukawa coupling.
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    m_stau_1 : Stau mass eigenstate mass 1.
    m_stau_2 : Stau mass eigenstate mass 2.
    mtauL : Left gauge eigenstate stau mass.
    mtauR : Right gauge eigenstate stau mass.
    a_tau : Soft trilinear scalar tau coupling.
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    Sigmauu_stau : One-loop correction Sigma_u^u(stau_2).

    """
    delta_eL = ((-1 / 2) + sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(v_higgs, tanb), 2) - np.power(vu(v_higgs, tanb), 2))
    delta_eR = ((-1) * sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(v_higgs, tanb), 2) - np.power(vu(v_higgs, tanb), 2))
    stau_num = (np.power(mtauR, 2) - np.power(mtauL, 2) + delta_eR - delta_eL)\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * ((-1 / 2) + 2 * sin_squared_theta_W(g_coupling, g_prime))\
        + 2 * np.power(y_tau, 2) * np.power(mu, 2)
    Sigmauu_stau = (1 / (32 * np.power(np.pi, 2)))\
        * (((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 4)
           + (stau_num / (np.power(m_stau_2, 2) - np.power(m_stau_1, 2))))\
        * F(m_stau_2, Q_renorm)
    return Sigmauu_stau


def sigmadd_stau1(v_higgs, mu, tanb, y_tau, g_coupling, g_prime, m_stau_1,
                  m_stau_2, mtauL, mtauR, a_tau, Q_renorm):
    """
    Return one-loop correction Sigma_d^d(stau_1).

    Parameters
    ----------
    v_higgs : Higgs VEV.
    mu : SUSY Higgs mass parameter, mu.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    y_tau : Tau Yukawa coupling.
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    m_stau_1 : Stau mass eigenstate mass 1.
    m_stau_2 : Stau mass eigenstate mass 2.
    mtauL : Left gauge eigenstate stau mass.
    mtauR : Right gauge eigenstate stau mass.
    a_tau : Soft trilinear scalar tau coupling.
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    Sigmadd_stau : One-loop correction Sigma_d^d(stau_1).

    """
    delta_eL = ((-1 / 2) + sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(v_higgs, tanb), 2) - np.power(vu(v_higgs, tanb), 2))
    delta_eR = ((-1) * sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(v_higgs, tanb), 2) - np.power(vu(v_higgs, tanb), 2))
    stau_num = (np.power(mtauR, 2) - np.power(mtauL, 2) + delta_eR - delta_eL)\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * ((1 / 2) - (2 * sin_squared_theta_W(g_coupling, g_prime)))\
        + 2 * np.power(a_tau, 2)
    Sigmadd_stau = (1 / (32 * np.power(np.pi, 2)))\
        * (((-1) * (np.power(g_coupling, 2) + np.power(g_prime, 2)) / 4)
           - (stau_num / (np.power(m_stau_2, 2) - np.power(m_stau_1, 2))))\
        * F(m_stau_1, Q_renorm)
    return Sigmadd_stau


def sigmadd_stau2(v_higgs, mu, tanb, y_tau, g_coupling, g_prime, m_stau_1,
                  m_stau_2, mtauL, mtauR, a_tau, Q_renorm):
    """
    Return one-loop correction Sigma_d^d(stau_2).

    Parameters
    ----------
    v_higgs : Higgs VEV.
    mu : SUSY Higgs mass parameter, mu.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    y_tau : Tau Yukawa coupling.
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    m_stau_1 : Stau mass eigenstate mass 1.
    m_stau_2 : Stau mass eigenstate mass 2.
    mtauL : Left gauge eigenstate stau mass.
    mtauR : Right gauge eigenstate stau mass.
    a_tau : Soft trilinear scalar tau coupling.
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    Sigmadd_stau : One-loop correction Sigma_d^d(stau_2).

    """
    delta_eL = ((-1 / 2) + sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(v_higgs, tanb), 2) - np.power(vu(v_higgs, tanb), 2))
    delta_eR = ((-1) * sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(v_higgs, tanb), 2) - np.power(vu(v_higgs, tanb), 2))
    stau_num = (np.power(mtauR, 2) - np.power(mtauL, 2) + delta_eR - delta_eL)\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * ((1 / 2) - (2 * sin_squared_theta_W(g_coupling, g_prime)))\
        + 2 * np.power(a_tau, 2)
    Sigmadd_stau = (1 / (32 * np.power(np.pi, 2)))\
        * (((-1) * (np.power(g_coupling, 2) + np.power(g_prime, 2)) / 4)
           + (stau_num / (np.power(m_stau_2, 2) - np.power(m_stau_1, 2))))\
        * F(m_stau_2, Q_renorm)
    return Sigmadd_stau


#########################
# Sfermions, 1st gen:
#########################


def sigmauu_sup_L(g_coupling, g_prime, msupL, Q_renorm):
    """
    Return one-loop correction Sigma_u^u(sup_L).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    msupL : Soft SUSY breaking mass for scalar up quark (left).
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    SigmauusupL : One-loop correction Sigma_u^u(sup_L).

    """
    SigmauusupL = ((-3) / (16 * np.power(np.pi, 2)))\
        * ((1 / 2) - (2 / 3) * sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * F(msupL, Q_renorm)
    return SigmauusupL


def sigmauu_sup_R(g_coupling, g_prime, msupR, Q_renorm):
    """
    Return one-loop correction Sigma_u^u(sup_R).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    msupR : Soft SUSY breaking mass for scalar up quark (right).
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    SigmauusupR : One-loop correction Sigma_u^u(sup_R).

    """
    SigmauusupR = ((-3) / (16 * np.power(np.pi, 2)))\
        * ((2 / 3) * sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * F(msupR, Q_renorm)
    return SigmauusupR


def sigmauu_sdown_L(g_coupling, g_prime, msdownL, Q_renorm):
    """
    Return one-loop correction Sigma_u^u(sdown_L).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    msdownL : Soft SUSY breaking mass for scalar down quark (left).
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    SigmauusdownL : One-loop correction Sigma_u^u(sdown_L).

    """
    SigmauusdownL = ((-3) / (16 * np.power(np.pi, 2)))\
        * (((-1) / 2) + (1 / 3) * sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * F(msdownL, Q_renorm)
    return SigmauusdownL


def sigmauu_sdown_R(g_coupling, g_prime, msdownR, Q_renorm):
    """
    Return one-loop correction Sigma_u^u(sdown_R).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    msdownR : Soft SUSY breaking mass for scalar down quark (right).
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    SigmauusdownR : One-loop correction Sigma_u^u(sdown_R).

    """
    SigmauusdownR = ((-3) / (16 * np.power(np.pi, 2)))\
        * (((-1) / 3) * sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * F(msdownR, Q_renorm)
    return SigmauusdownR


def sigmauu_selec_L(g_coupling, g_prime, mselecL, Q_renorm):
    """
    Return one-loop correction Sigma_u^u(selectron_L).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    mselecL : Soft SUSY breaking mass for scalar electron (left).
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    SigmauuselecL : One-loop correction Sigma_u^u(selectron_L).

    """
    SigmauuselecL = ((-1) / (16 * np.power(np.pi, 2)))\
        * (((-1) / 2) + sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * F(mselecL, Q_renorm)
    return SigmauuselecL


def sigmauu_selec_R(g_coupling, g_prime, mselecR, Q_renorm):
    """
    Return one-loop correction Sigma_u^u(selectron_R).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    mselecR : Soft SUSY breaking mass for scalar electron (right).
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    SigmauuselecR : One-loop correction Sigma_u^u(selectron_R).

    """
    SigmauuselecR = ((-1) / (16 * np.power(np.pi, 2)))\
        * ((-1) * sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * F(mselecR, Q_renorm)
    return SigmauuselecR


def sigmauu_selecSneut(g_coupling, g_prime, mselecSneut, Q_renorm):
    """
    Return one-loop correction Sigma_u^u(selectron neutrino).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    mselecSneut : Mass for scalar electron neutrino.
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    SigmauuselecSneut : One-loop correction Sigma_u^u(selectron neutrino).

    """
    SigmauuselecSneut = ((-1) / (32 * np.power(np.pi, 2))) * (1 / 2)\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * F(mselecSneut, Q_renorm)
    return SigmauuselecSneut


def sigmadd_sup_L(g_coupling, g_prime, msupL, Q_renorm):
    """
    Return one-loop correction Sigma_d^d(sup_L).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    msupL : Soft SUSY breaking mass for scalar up quark (left).
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    SigmaddsupL : One-loop correction Sigma_d^d(sup_L).

    """
    SigmaddsupL = (3 / (16 * np.power(np.pi, 2)))\
        * ((1 / 2) - (2 / 3) * sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * F(msupL, Q_renorm)
    return SigmaddsupL


def sigmadd_sup_R(g_coupling, g_prime, msupR, Q_renorm):
    """
    Return one-loop correction Sigma_d^d(sup_R).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    msupR : Soft SUSY breaking mass for scalar up quark (right).
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    SigmaddsupR : One-loop correction Sigma_d^d(sup_R).

    """
    SigmaddsupR = (3 / (16 * np.power(np.pi, 2)))\
        * ((2 / 3) * sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * F(msupR, Q_renorm)
    return SigmaddsupR


def sigmadd_sdown_L(g_coupling, g_prime, msdownL, Q_renorm):
    """
    Return one-loop correction Sigma_d^d(sdown_L).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    msdownL : Soft SUSY breaking mass for scalar down quark (left).
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    SigmaddsdownL : One-loop correction Sigma_d^d(sdown_L).

    """
    SigmaddsdownL = (3 / (16 * np.power(np.pi, 2)))\
        * (((-1) / 2) + (1 / 3) * sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * F(msdownL, Q_renorm)
    return SigmaddsdownL


def sigmadd_sdown_R(g_coupling, g_prime, msdownR, Q_renorm):
    """
    Return one-loop correction Sigma_d^d(sdown_R).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    msdownR : Soft SUSY breaking mass for scalar down quark (right).
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    SigmaddsdownR : One-loop correction Sigma_d^d(sdown_R).

    """
    SigmaddsdownR = (3 / (16 * np.power(np.pi, 2)))\
        * (((-1) / 3) * sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * F(msdownR, Q_renorm)
    return SigmaddsdownR


def sigmadd_selec_L(g_coupling, g_prime, mselecL, Q_renorm):
    """
    Return one-loop correction Sigma_d^d(selectron_L).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    mselecL : Soft SUSY breaking mass for scalar electron (left).
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    SigmaddselecL : One-loop correction Sigma_d^d(selectron_L).

    """
    SigmaddselecL = (1 / (16 * np.power(np.pi, 2)))\
        * (((-1) / 2) + sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * F(mselecL, Q_renorm)
    return SigmaddselecL


def sigmadd_selec_R(g_coupling, g_prime, mselecR, Q_renorm):
    """
    Return one-loop correction Sigma_d^d(selectron_R).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    mselecR : Soft SUSY breaking mass for scalar electron (right).
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    SigmaddselecR : One-loop correction Sigma_d^d(selectron_R).

    """
    SigmaddselecR = (1 / (16 * np.power(np.pi, 2)))\
        * ((-1) * sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * F(mselecR, Q_renorm)
    return SigmaddselecR


def sigmadd_selecSneut(g_coupling, g_prime, mselecSneut, Q_renorm):
    """
    Return one-loop correction Sigma_d^d(selectron neutrino).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    mselecSneut : Mass for scalar electron neutrino.
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    SigmaddselecSneut : One-loop correction Sigma_d^d(selectron neutrino).

    """
    SigmauuselecSneut = (1 / (32 * np.power(np.pi, 2))) * (1 / 2)\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * F(mselecSneut, Q_renorm)
    return SigmauuselecSneut


#########################
# Sfermions, 2nd gen:
#########################


def sigmauu_sstrange_L(g_coupling, g_prime, msstrangeL, Q_renorm):
    """
    Return one-loop correction Sigma_u^u(sstrange_L).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    msstrangeL : Soft SUSY breaking mass for scalar strange quark (left).
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    SigmauusstrangeL : One-loop correction Sigma_u^u(sstrange_L).

    """
    SigmauusstrangeL = ((-3) / (16 * np.power(np.pi, 2)))\
        * (((-1) / 2) + (1 / 3) * sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * F(msstrangeL, Q_renorm)
    return SigmauusstrangeL


def sigmauu_sstrange_R(g_coupling, g_prime, msstrangeR, Q_renorm):
    """
    Return one-loop correction Sigma_u^u(sstrange_R).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    msstrangeR : Soft SUSY breaking mass for scalar strange quark (right).
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    SigmauusstrangeR : One-loop correction Sigma_u^u(sstrange_R).

    """
    SigmauusstrangeR = ((-3) / (16 * np.power(np.pi, 2)))\
        * (((-1) / 3) * sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * F(msstrangeR, Q_renorm)
    return SigmauusstrangeR


def sigmauu_scharm_L(g_coupling, g_prime, mscharmL, Q_renorm):
    """
    Return one-loop correction Sigma_u^u(scharm_L).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    mscharmL : Soft SUSY breaking mass for scalar charm quark (left).
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    SigmauuscharmL : One-loop correction Sigma_u^u(scharm_L).

    """
    SigmauuscharmL = ((-3) / (16 * np.power(np.pi, 2)))\
        * ((1 / 2) - (2 / 3) * sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * F(mscharmL, Q_renorm)
    return SigmauuscharmL


def sigmauu_scharm_R(g_coupling, g_prime, mscharmR, Q_renorm):
    """
    Return one-loop correction Sigma_u^u(scharm_R).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    mscharmR : Soft SUSY breaking mass for scalar charm quark (right).
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    SigmauuscharmR : One-loop correction Sigma_u^u(scharm_R).

    """
    SigmauuscharmR = ((-3) / (16 * np.power(np.pi, 2)))\
        * ((2 / 3) * sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * F(mscharmR, Q_renorm)
    return SigmauuscharmR


def sigmauu_smu_L(g_coupling, g_prime, msmuL, Q_renorm):
    """
    Return one-loop correction Sigma_u^u(smu_L).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    msmuL : Soft SUSY breaking mass for scalar muon (left).
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    SigmauusmuL : One-loop correction Sigma_u^u(smu_L).

    """
    SigmauusmuL = ((-1) / (16 * np.power(np.pi, 2)))\
        * (((-1) / 2) + sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * F(msmuL, Q_renorm)
    return SigmauusmuL


def sigmauu_smu_R(g_coupling, g_prime, msmuR, Q_renorm):
    """
    Return one-loop correction Sigma_u^u(smu_R).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    msmuR : Soft SUSY breaking mass for scalar muon (right).
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    SigmauusmuR : One-loop correction Sigma_u^u(smu_R).

    """
    SigmauusmuR = ((-1) / (16 * np.power(np.pi, 2)))\
        * (((-1)) * sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * F(msmuR, Q_renorm)
    return SigmauusmuR


def sigmauu_smuSneut(g_coupling, g_prime, msmuSneut, Q_renorm):
    """
    Return one-loop correction Sigma_u^u(smuon neutrino).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    msmuSneut : Mass for scalar muon neutrino.
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    SigmauusmuSneut : One-loop correction Sigma_u^u(smuon neutrino).

    """
    SigmauusmuSneut = ((-1) / (32 * np.power(np.pi, 2))) * (1 / 2)\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * F(msmuSneut, Q_renorm)
    return SigmauusmuSneut


def sigmadd_sstrange_L(g_coupling, g_prime, msstrangeL, Q_renorm):
    """
    Return one-loop correction Sigma_d^d(sstrange_L).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    msstrangeL : Soft SUSY breaking mass for scalar strange quark (left).
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    SigmaddsstrangeL : One-loop correction Sigma_d^d(sstrange_L).

    """
    SigmaddsstrangeL = (3 / (16 * np.power(np.pi, 2)))\
        * (((-1) / 2) + (1 / 3) * sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * F(msstrangeL, Q_renorm)
    return SigmaddsstrangeL


def sigmadd_sstrange_R(g_coupling, g_prime, msstrangeR, Q_renorm):
    """
    Return one-loop correction Sigma_d^d(sstrange_R).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    msstrangeR : Soft SUSY breaking mass for scalar strange quark (right).
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    SigmaddsstrangeR : One-loop correction Sigma_d^d(sstrange_R).

    """
    SigmaddsstrangeR = (3 / (16 * np.power(np.pi, 2)))\
        * (((-1) / 3) * sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * F(msstrangeR, Q_renorm)
    return SigmaddsstrangeR


def sigmadd_scharm_L(g_coupling, g_prime, mscharmL, Q_renorm):
    """
    Return one-loop correction Sigma_d^d(scharm_L).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    mscharmL : Soft SUSY breaking mass for scalar charm quark (left).
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    SigmaddscharmL : One-loop correction Sigma_d^d(scharm_L).

    """
    SigmaddscharmL = (3 / (16 * np.power(np.pi, 2)))\
        * ((1 / 2) - (2 / 3) * sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * F(mscharmL, Q_renorm)
    return SigmaddscharmL


def sigmadd_scharm_R(g_coupling, g_prime, mscharmR, Q_renorm):
    """
    Return one-loop correction Sigma_d^d(scharm_R).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    mscharmR : Soft SUSY breaking mass for scalar charm quark (right).
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    SigmaddscharmR : One-loop correction Sigma_d^d(scharm_R).

    """
    SigmaddscharmR = (3 / (16 * np.power(np.pi, 2)))\
        * ((2 / 3) * sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * F(mscharmR, Q_renorm)
    return SigmaddscharmR


def sigmadd_smu_L(g_coupling, g_prime, msmuL, Q_renorm):
    """
    Return one-loop correction Sigma_d^d(smu_L).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    msmuL : Soft SUSY breaking mass for scalar muon (left).
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    SigmaddsmuL : One-loop correction Sigma_d^d(smu_L).

    """
    SigmaddsmuL = (1 / (16 * np.power(np.pi, 2)))\
        * (((-1) / 2) + sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * F(msmuL, Q_renorm)
    return SigmaddsmuL


def sigmadd_smu_R(g_coupling, g_prime, msmuR, Q_renorm):
    """
    Return one-loop correction Sigma_d^d(smu_R).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    msmuR : Soft SUSY breaking mass for scalar muon (right).
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    SigmaddsmuR : One-loop correction Sigma_d^d(smu_R).

    """
    SigmaddsmuR = (1 / (16 * np.power(np.pi, 2)))\
        * (((-1)) * sin_squared_theta_W(g_coupling, g_prime))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * F(msmuR, Q_renorm)
    return SigmaddsmuR


def sigmadd_smuSneut(g_coupling, g_prime, msmuSneut, Q_renorm):
    """
    Return one-loop correction Sigma_d^d(smuon neutrino).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    msmuSneut : Mass for scalar muon neutrino.
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    SigmaddsmuSneut : One-loop correction Sigma_d^d(smuon neutrino).

    """
    SigmaddsmuSneut = (1 / (32 * np.power(np.pi, 2))) * (1 / 2)\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 2)\
        * F(msmuSneut, Q_renorm)
    return SigmaddsmuSneut


#########################
# Neutralinos:
#########################
# Set up terms from characteristic polynomial for eigenvalues x of squared
# neutralino mass matrix, and use method by Ibrahim and Nath for derivatives
# of eigenvalues.
# x^4 + b(vu, vd) * x^3 + c(vu, vd) * x^2 + d(vu, vd) * x + e(vu, vd) = 0


def neutralinouu_deriv_num(M1, M2, mu, g_coupling, g_prime, v_higgs, tanb,
                           msN):
    """
    Return numerator for one-loop uu correction derivative term of neutralino.

    Parameters
    ----------
    M1 : Bino mass parameter.
    M2 : Wino mass parameter.
    mu : SUSY Higgs mass parameter, mu.
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
    quadrterm = (((np.power(g_coupling, 2) * M2 * mu)
                  + (np.power(g_prime, 2) * M1 * mu)) / (tanb))\
        - ((np.power(g_coupling, 2) * np.power(M1, 2)) + (np.power(g_prime, 2)
                                                          * np.power(M2, 2))
           + ((np.power(g_coupling, 2) + np.power(g_prime, 2)) * (np.power(mu,
                                                                           2)))
           + (np.power((np.power(g_coupling, 2) + np.power(g_prime, 2)), 2)
              / 2)
           * np.power(v_higgs, 2))
    linterm = (((-1) * mu) * ((np.power(g_coupling, 2) * M2
                               * (np.power(M1, 2) + np.power(mu, 2)))
                              + np.power(g_prime, 2) * M1
                              * (np.power(M2, 2) + np.power(mu, 2))) / tanb)\
        + ((np.power((np.power(g_coupling, 2) * M1 + np.power(g_prime, 2)
                      * M2), 2) / 2)
           * np.power(v_higgs, 2))\
        + (np.power(mu, 2) * ((np.power(g_coupling, 2) * np.power(M1, 2))
                              + np.power(g_prime, 2) * np.power(M2, 2)))\
        + (np.power((np.power(g_coupling, 2) + np.power(g_prime, 2)), 2)
           * np.power(v_higgs, 2) * np.power(mu, 2) * cossqb(tanb))
    constterm = (M1 * M2 * ((np.power(g_coupling, 2) * M1)
                            + (np.power(g_prime, 2) * M2))
                 * np.power(mu, 3) * (1 / tanb))\
        - (np.power((np.power(g_coupling, 2) * M1 + np.power(g_prime, 2)
                     * M2), 2)
           * np.power(v_higgs, 2) * np.power(mu, 2) * cossqb(tanb))
    mynum = (cubicterm * np.power(msN, 6)) + (quadrterm * np.power(msN, 4))\
        + (linterm * np.power(msN, 2)) + constterm
    return mynum


def neutralinodd_deriv_num(M1, M2, mu, g_coupling, g_prime, v_higgs, tanb,
                           msN):
    """
    Return numerator for one-loop dd correction derivative term of neutralino.

    Parameters
    ----------
    M1 : Bino mass parameter.
    M2 : Wino mass parameter.
    mu : SUSY Higgs mass parameter, mu.
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
    quadrterm = (((np.power(g_coupling, 2) * M2 * mu)
                  + (np.power(g_prime, 2) * M1 * mu)) * (tanb))\
        - ((np.power(g_coupling, 2) * np.power(M1, 2))
           + (np.power(g_prime, 2) * np.power(M2, 2))
           + ((np.power(g_coupling, 2) + np.power(g_prime, 2))
              * (np.power(mu, 2)))
           + (np.power((np.power(g_coupling, 2) + np.power(g_prime, 2)), 2)
              / 2)
           * np.power(v_higgs, 2))
    linterm = (((-1) * mu) * ((np.power(g_coupling, 2) * M2
                               * (np.power(M1, 2) + np.power(mu, 2)))
                              + np.power(g_prime, 2) * M1
                              * (np.power(M2, 2) + np.power(mu, 2))) * tanb)\
        + ((np.power((np.power(g_coupling, 2) * M1 + np.power(g_prime, 2)
                      * M2), 2) / 2)
           * np.power(v_higgs, 2))\
        + (np.power(mu, 2) * (np.power(g_coupling, 2) * np.power(M1, 2)
           + np.power(g_prime, 2) * np.power(M2, 2)))\
        + (np.power((np.power(g_coupling, 2) + np.power(g_prime, 2)), 2)
           * np.power(v_higgs, 2) * np.power(mu, 2) * sinsqb(tanb))
    constterm = (M1 * M2 * (np.power(g_coupling, 2) * M1 + np.power(g_prime, 2)
                            * M2)
                 * np.power(mu, 3) * tanb)\
        - (np.power((np.power(g_coupling, 2) * M1 + np.power(g_prime, 2)
                     * M2), 2)
           * np.power(v_higgs, 2) * np.power(mu, 2) * sinsqb(tanb))
    mynum = (cubicterm * np.power(msN, 6))\
        + (quadrterm * np.power(msN, 4))\
        + (linterm * np.power(msN, 2)) + constterm
    return mynum


def neutralino_deriv_denom(M1, M2, mu, g_coupling, g_prime, v_higgs, tanb,
                           msN):
    """
    Return denominator for one-loop correction derivative term of neutralino.

    Parameters
    ----------
    M1 : Bino mass parameter.
    M2 : Wino mass parameter.
    mu : SUSY Higgs mass parameter, mu.
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    v_higgs : Higgs VEV.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    msN : Neutralino mass.

    Returns
    -------
    mydenom : Denominator for 1-loop correction derivative term of neutralino.

    """
    quadrterm = -3 * ((np.power(M1, 2)) + (np.power(M2, 2))
                      + ((np.power(g_coupling, 2) + np.power(g_prime, 2))
                         * np.power(v_higgs, 2))
                      + (2 * np.power(mu, 2)))
    linterm = (np.power(v_higgs, 4)
               * np.power((np.power(g_coupling, 2) + np.power(g_prime, 2)), 2)
               / 2)\
        + (np.power(v_higgs, 2)
           * (2 * ((np.power(g_coupling, 2) * np.power(M1, 2))
                   + (np.power(g_prime, 2) * np.power(M2, 2))
                   + ((np.power(g_coupling, 2)
                       + np.power(g_prime, 2)) * np.power(mu, 2))
                   - (mu * (np.power(g_prime, 2) * M1 + np.power(g_coupling, 2)
                            * M2)
                      * 2 * np.sqrt(sinsqb(tanb)) * np.sqrt(cossqb(tanb))))))\
        + (2 * ((np.power(M1, 2) * np.power(M2, 2))
                + (2 * (np.power(M1, 2) + np.power(M2, 2)) * np.power(mu, 2))
                + (np.power(mu, 4))))
    constterm = (np.power(v_higgs, 4) * (1 / 8)
                 * ((np.power((np.power(g_coupling, 2) + np.power(g_prime, 2)),
                              2)
                     * np.power(mu, 2)
                     * (np.power(cossqb(tanb), 2)
                        - (6 * cossqb(tanb) * sinsqb(tanb))
                        + np.power(sinsqb(tanb), 2)))
                    - (2 * np.power((np.power(g_coupling, 2) * M1
                                     + np.power(g_prime, 2) * M2), 2))
                    - (np.power(mu, 2) * np.power((np.power(g_coupling, 2)
                                                   + np.power(g_prime, 2)), 2))
                    ))\
        + (np.power(v_higgs, 2) * 2 * mu
           * ((np.sqrt(cossqb(tanb)) * np.sqrt(sinsqb(tanb)))
              * (np.power(g_coupling, 2) * M2 * (np.power(M1, 2)
                                                 + np.power(mu, 2))
                 + (np.power(g_prime, 2) * M1
                 * (np.power(M2, 2) + np.power(mu, 2))))))\
        - ((2 * np.power(M2, 2) * np.power(M1, 2) * np.power(mu, 2))
           + (np.power(mu, 4) * (np.power(M1, 2) + np.power(M2, 2))))
    mydenom = 4 * np.power(msN, 6)\
        + quadrterm * np.power(msN, 4)\
        + linterm * np.power(msN, 2)\
        + constterm
    return mydenom


def sigmauu_neutralino(M1, M2, mu, g_coupling, g_prime, v_higgs, tanb, msN,
                       Q_renorm):
    """
    Return one-loop correction Sigma_u^u(neutralino).

    Parameters
    ----------
    M1 : Bino mass parameter.
    M2 : Wino mass parameter.
    mu : SUSY Higgs mass parameter, mu.
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    v_higgs : Higgs VEV.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    msN : Neutralino mass.
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    Sigmauu_neutralino : One-loop correction Sigma_u^u(neutralino).

    """
    Sigmauu_neutralino = ((-1) / (16 * np.power(np.pi, 2))) \
        * (neutralinouu_deriv_num(M1, M2, mu, g_coupling, g_prime,
                                  v_higgs, tanb, msN)
           / neutralino_deriv_denom(M1, M2, mu, g_coupling, g_prime,
                                    v_higgs, tanb, msN))\
        * F(msN, Q_renorm)
    return Sigmauu_neutralino


def sigmadd_neutralino(M1, M2, mu, g_coupling, g_prime, v_higgs, tanb, msN,
                       Q_renorm):
    """
    Return one-loop correction Sigma_d^d(neutralino).

    Parameters
    ----------
    M1 : Bino mass parameter.
    M2 : Wino mass parameter.
    mu : SUSY Higgs mass parameter, mu.
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    v_higgs : Higgs VEV.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    msN : Neutralino mass.
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    Sigmadd_neutralino : One-loop correction Sigma_d^d(neutralino).

    """
    Sigmadd_neutralino = ((-1) / (16 * np.power(np.pi, 2))) \
        * (neutralinodd_deriv_num(M1, M2, mu, g_coupling, g_prime,
                                  v_higgs, tanb, msN)
           / neutralino_deriv_denom(M1, M2, mu, g_coupling, g_prime,
                                    v_higgs, tanb, msN))\
        * F(msN, Q_renorm)
    return Sigmadd_neutralino


#########################
# Charginos:
#########################


def sigmauu_chargino1(g_coupling, M2, v_higgs, tanb, mu, msC, Q_renorm):
    """
    Return one-loop correction Sigma_u^u(chargino_1).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    M2 : Wino mass parameter.
    v_higgs : Higgs VEV.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    mu : SUSY Higgs mass parameter, mu.
    msC : Chargino mass.
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    Sigmauu_chargino1 : One-loop correction Sigma_u^u(chargino_1).

    """
    chargino_num = np.power(M2, 2) + np.power(mu, 2)\
        + (np.power(g_coupling, 2) * (np.power(vu(v_higgs, tanb), 2)
                                      - np.power(vd(v_higgs, tanb), 2)))
    chargino_den = np.sqrt(((np.power(g_coupling, 2)
                             * np.power((vu(v_higgs, tanb)
                                         + vd(v_higgs, tanb)), 2))
                            + np.power((M2 - mu), 2))
                           * ((np.power(g_coupling, 2)
                               * np.power((vd(v_higgs, tanb)
                                           - vu(v_higgs, tanb)), 2))
                              + np.power((M2 + mu), 2)))
    Sigmauu_chargino1 = -1 * (np.power(g_coupling, 2) / (16 * np.power(np.pi,
                                                                       2)))\
        * (1 - (chargino_num / chargino_den)) * F(msC, Q_renorm)
    return Sigmauu_chargino1


def sigmauu_chargino2(g_coupling, M2, v_higgs, tanb, mu, msC, Q_renorm):
    """
    Return one-loop correction Sigma_u^u(chargino_2).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    M2 : Wino mass parameter.
    v_higgs : Higgs VEV.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    mu : SUSY Higgs mass parameter, mu.
    msC : Chargino mass.
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    Sigmauu_chargino2 : One-loop correction Sigma_u^u(chargino_2).

    """
    chargino_num = np.power(M2, 2) + np.power(mu, 2)\
        + (np.power(g_coupling, 2) * (np.power(vu(v_higgs, tanb), 2)
                                      - np.power(vd(v_higgs, tanb), 2)))
    chargino_den = np.sqrt(((np.power(g_coupling, 2)
                             * np.power((vu(v_higgs, tanb)
                                         + vd(v_higgs, tanb)), 2))
                            + np.power((M2 - mu), 2))
                           * ((np.power(g_coupling, 2)
                               * np.power((vd(v_higgs, tanb)
                                           - vu(v_higgs, tanb)), 2))
                              + np.power((M2 + mu), 2)))
    Sigmauu_chargino2 = -1 * (np.power(g_coupling, 2) / (16 * np.power(np.pi,
                                                                       2)))\
        * (1 + (chargino_num / chargino_den)) * F(msC, Q_renorm)
    return Sigmauu_chargino2


def sigmadd_chargino1(g_coupling, M2, v_higgs, tanb, mu, msC, Q_renorm):
    """
    Return one-loop correction Sigma_d^d(chargino_1).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    M2 : Wino mass parameter.
    v_higgs : Higgs VEV.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    mu : SUSY Higgs mass parameter, mu.
    msC : Chargino mass.
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    Sigmadd_chargino1 : One-loop correction Sigma_d^d(chargino_1).

    """
    chargino_num = np.power(M2, 2) + np.power(mu, 2)\
        - (np.power(g_coupling, 2) * (np.power(vu(v_higgs, tanb), 2)
                                      - np.power(vd(v_higgs, tanb), 2)))
    chargino_den = np.sqrt(((np.power(g_coupling, 2)
                             * np.power((vu(v_higgs, tanb)
                                         + vd(v_higgs, tanb)), 2))
                            + np.power((M2 - mu), 2))
                           * ((np.power(g_coupling, 2)
                               * np.power((vd(v_higgs, tanb)
                                           - vu(v_higgs, tanb)), 2))
                              + np.power((M2 + mu), 2)))
    Sigmadd_chargino1 = -1 * (np.power(g_coupling, 2) / (16 * np.power(np.pi,
                                                                       2)))\
        * (1 - (chargino_num / chargino_den)) * F(msC, Q_renorm)
    return Sigmadd_chargino1


def sigmadd_chargino2(g_coupling, M2, v_higgs, tanb, mu, msC, Q_renorm):
    """
    Return one-loop correction Sigma_d^d(chargino_2).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    M2 : Wino mass parameter.
    v_higgs : Higgs VEV.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    mu : SUSY Higgs mass parameter, mu.
    msC : Chargino mass.
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    Sigmadd_chargino2 : One-loop correction Sigma_d^d(chargino_2).

    """
    chargino_num = np.power(M2, 2) + np.power(mu, 2)\
        - (np.power(g_coupling, 2) * (np.power(vu(v_higgs, tanb), 2)
                                      - np.power(vd(v_higgs, tanb), 2)))
    chargino_den = np.sqrt(((np.power(g_coupling, 2)
                             * np.power((vu(v_higgs, tanb)
                                         + vd(v_higgs, tanb)), 2))
                            + np.power((M2 - mu), 2))
                           * ((np.power(g_coupling, 2)
                               * np.power((vd(v_higgs, tanb)
                                           - vu(v_higgs, tanb)), 2))
                              + np.power((M2 + mu), 2)))
    Sigmadd_chargino2 = -1 * (np.power(g_coupling, 2) / (16 * np.power(np.pi,
                                                                       2)))\
        * (1 + (chargino_num / chargino_den)) * F(msC, Q_renorm)
    return Sigmadd_chargino2


#########################
# Higgs bosons (sigmauu = sigmadd here):
#########################


def sigmauu_h0(g_coupling, g_prime, v_higgs, tanb, mHusq, mHdsq, mu, mZ, mh0,
               Q_renorm):
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
    mu : SUSY Higgs mass parameter, mu.
    mZ : Z boson mass.
    mh0 : Lighter neutral Higgs mass.
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    Sigmauu_h0 : One-loop correction Sigma_u,d^u,d(h_0) (lighter neutr. Higgs).

    """
    mynum = ((np.power(g_coupling, 2) + np.power(g_prime, 2))
             * np.power(v_higgs, 2))\
        - (2 * mA0sq(mu, mHusq, mHdsq) * (np.power(cossqb(tanb), 2)
                                          - 6 * cossqb(tanb) * sinsqb(tanb)
                                          + np.power(sinsqb(tanb), 2)))
    myden = np.sqrt(np.power((mA0sq(mu, mHusq, mHdsq) - np.power(mZ, 2)), 2)
                    + (4 * np.power(mZ, 2) * mA0sq(mu, mHusq, mHdsq) * 4
                       * cossqb(tanb) * sinsqb(tanb)))
    Sigmauu_h0 = (1 / (32 * np.power(np.pi, 2)))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 4)\
        * (1 - (mynum / myden)) * F(mh0, Q_renorm)
    return Sigmauu_h0


def sigmauu_H0(g_coupling, g_prime, v_higgs, tanb, mHusq, mHdsq, mu, mZ, mH0,
               Q_renorm):
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
    mu : SUSY Higgs mass parameter, mu.
    mZ : Z boson mass.
    mH0 : Heavier neutral Higgs mass.
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    Sigmauu_H0 : One-loop correction Sigma_u,d^u,d(h_0) (heavier neut. Higgs).

    """
    mynum = ((np.power(g_coupling, 2) + np.power(g_prime, 2))
             * np.power(v_higgs, 2))\
        - (2 * mA0sq(mu, mHusq, mHdsq) * (np.power(cossqb(tanb), 2)
                                          - 6 * cossqb(tanb) * sinsqb(tanb)
                                          + np.power(sinsqb(tanb), 2)))
    myden = np.sqrt(np.power((mA0sq(mu, mHusq, mHdsq) - np.power(mZ, 2)), 2)
                    + (4 * np.power(mZ, 2) * mA0sq(mu, mHusq, mHdsq)
                       * 4 * cossqb(tanb) * sinsqb(tanb)))
    Sigmauu_H0 = (1/(32 * np.power(np.pi, 2)))\
        * ((np.power(g_coupling, 2) + np.power(g_prime, 2)) / 4)\
        * (1 + (mynum / myden)) * F(mH0, Q_renorm)
    return Sigmauu_H0


def sigmauu_H_pm(g_coupling, mH_pm, Q_renorm):
    """
    Return one-loop correction Sigma_u,d^u,d(H_{+-}).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    mH_pm : Charged Higgs mass.
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    Sigmauu_H_pm : One-loop correction Sigma_u,d^u,d(H_{+-}).

    """
    Sigmauu_H_pm = (np.power(g_coupling, 2) / (64 * np.power(np.pi, 2)))\
        * F(mH_pm, Q_renorm)
    return Sigmauu_H_pm


#########################
# Weak bosons (sigmauu = sigmadd here):
#########################


def sigmauu_W_pm(g_coupling, v_higgs, Q_renorm):
    """
    Return one-loop correction Sigma_u,d^u,d(W_{+-}).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    v_higgs : Higgs VEV.
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    Sigmauu_W_pm : One-loop correction Sigma_u,d^u,d(W_{+-}).

    """
    mymWsq = m_w_sq(g_coupling, v_higgs)
    Sigmauu_W_pm = (3 * np.power(g_coupling, 2) / (32 * np.power(np.pi, 2)))\
        * F(np.sqrt(mymWsq), Q_renorm)
    return Sigmauu_W_pm


def sigmauu_Z0(g_coupling, g_prime, v_higgs, Q_renorm):
    """
    Return one-loop correction Sigma_u,d^u,d(Z_0).

    Parameters
    ----------
    g_coupling : Electroweak coupling constant g.
    g_prime : Electroweak coupling constant g'.
    v_higgs : Higgs VEV.
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    Sigmauu_Z0 : One-loop correction Sigma_u,d^u,d(Z_0).

    """
    mymZsq = mZsq(g_coupling, g_prime, v_higgs)
    Sigmauu_W_pm = (3 * np.power(g_coupling, 2) / (64 * np.power(np.pi, 2)))\
        * F(np.sqrt(mymZsq), Q_renorm)
    return Sigmauu_W_pm


#########################
# SM fermions (sigmadd_t = sigmauu_b = sigmauu_tau = 0):
#########################


def sigmauu_top(yt, v_higgs, tanb, Q_renorm):
    """
    Return one-loop correction Sigma_u^u(top).

    Parameters
    ----------
    yt : Top Yukawa coupling.
    v_higgs : Higgs VEV.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    Sigmauu_top : One-loop correction Sigma_u^u(top).

    """
    mymt = yt * vu(v_higgs, tanb)
    Sigmauu_top = ((-1) * np.power(yt, 2) / (16 * np.power(np.pi, 2)))\
        * F(mymt, Q_renorm)
    return Sigmauu_top


def sigmadd_top():
    """Return one-loop correction Sigma_d^d(top) = 0."""
    return 0


def sigmauu_bottom():
    """Return one-loop correction Sigma_u^u(bottom) = 0."""
    return 0


def sigmadd_bottom(yb, v_higgs, tanb, Q_renorm):
    """
    Return one-loop correction Sigma_d^d(bottom).

    Parameters
    ----------
    yb : Bottom Yukawa coupling.
    v_higgs : Higgs VEV.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    Sigmadd_bottom : One-loop correction Sigma_d^d(bottom).

    """
    mymb = yb * vd(v_higgs, tanb)
    Sigmadd_bottom = (-1 * np.power(yb, 2) / (16 * np.power(np.pi, 2)))\
        * F(mymb, Q_renorm)
    return Sigmadd_bottom


def sigmauu_tau():
    """Return one-loop correction Sigma_u^u(tau) = 0."""
    return 0


def sigmadd_tau(ytau, v_higgs, tanb, Q_renorm):
    """
    Return one-loop correction Sigma_d^d(tau).

    Parameters
    ----------
    ytau : Tau Yukawa coupling.
    v_higgs : Higgs VEV.
    tanb : Ratio of Higgs VEVs, tan(beta) = v_u / v_d.
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    Sigmadd_tau : One-loop correction Sigma_d^d(tau).

    """
    mymtau = ytau * vd(v_higgs, tanb)
    Sigmadd_tau = (-1 * np.power(ytau, 2) / (16 * np.power(np.pi, 2)))\
        * F(mymtau, Q_renorm)
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


def DEW(v_higgs, mu, tanb, y_t, y_b, y_tau, g_coupling, g_prime, m_stop_1,
        m_stop_2, m_sbot_1, m_sbot_2, m_stau_1, m_stau_2, mtL, mtR, mbL, mbR,
        mtauL, mtauR, msupL, msupR, msdownL, msdownR, mselecL, mselecR,
        mselecSneut, msstrangeL, msstrangeR, mscharmL, mscharmR, msmuL,
        msmuR, msmuSneut, msN1, msN2, msN3, msN4, msC1, msC2, mZ, mh0,
        mH0, mHusq, mHdsq, mH_pm, M1, M2, a_t, a_b, a_tau, Q_renorm):
    """
    Return Delta_EW.

    Parameters
    ----------
    v_higgs : Higgs VEV.
    mu : SUSY Higgs mass parameter, mu.
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
    mtL : Left gauge eigenstate stop mass.
    mtR : Right gauge eigenstate stop mass.
    mbL : Left gauge eigenstate sbottom mass.
    mbR : Right gauge eigenstate sbottom mass.
    mtauL : Left gauge eigenstate stau mass.
    mtauR : Right gauge eigenstate stau mass.
    msupL : Soft SUSY breaking mass for scalar up quark (left).
    msupR : Soft SUSY breaking mass for scalar up quark (right).
    msdownL : Soft SUSY breaking mass for scalar down quark (left).
    msdownR : Soft SUSY breaking mass for scalar down quark (right).
    mselecL : Soft SUSY breaking mass for scalar electron (left).
    mselecR : Soft SUSY breaking mass for scalar electron (right).
    mselecSneut : Mass for scalar electron neutrino..
    msstrangeL : Soft SUSY breaking mass for scalar strange quark (left).
    msstrangeR : Soft SUSY breaking mass for scalar strange quark (right).
    mscharmL : Soft SUSY breaking mass for scalar charm quark (left).
    mscharmR : Soft SUSY breaking mass for scalar charm quark (right).
    msmuL : Soft SUSY breaking mass for scalar muon (left).
    msmuR : Soft SUSY breaking mass for scalar muon (right).
    msmuSneut : Mass for scalar muon neutrino.
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
    M1 : Bino mass parameter.
    M2 : Wino mass parameter.
    a_t : Soft trilinear scalar top coupling.
    a_b : Soft trilinear scalar bottom coupling.
    a_tau : Soft trilinear scalar tau coupling.
    Q_renorm : Renormalization scale (usually sqrt(m_stop_1 * m_stop_2)).

    Returns
    -------
    mydew : Delta_EW based on all one-loop contributions, mu, and H_{u,d}.

    """
    cmu = np.absolute((-1) * np.power(mu, 2))
    cHu = np.absolute((np.absolute((-1) * mHusq / np.sqrt(1
                                                          - (4 * sinsqb(tanb)
                                                             * cossqb(tanb)))))
                      - mHusq) / 2
    cHd = np.absolute((np.absolute(mHdsq / np.sqrt(1 - (4 * sinsqb(tanb)
                                                        * cossqb(tanb)))))
                      - mHdsq) / 2
    contribution_array = np.array([cmu, cHu, cHd,
                                   DEW_func(sigmadd_stop1(v_higgs, mu,
                                                          tanb, y_t,
                                                          g_coupling, g_prime,
                                                          m_stop_1, m_stop_2,
                                                          mtL, mtR, a_t,
                                                          Q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_stop2(v_higgs, mu,
                                                          tanb, y_t,
                                                          g_coupling, g_prime,
                                                          m_stop_1, m_stop_2,
                                                          mtL, mtR, a_t,
                                                          Q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_sbottom1(v_higgs, mu,
                                                             tanb, y_b,
                                                             g_coupling,
                                                             g_prime, m_sbot_1,
                                                             m_sbot_2, mbL,
                                                             mbR, a_b,
                                                             Q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_sbottom2(v_higgs,
                                                             mu, tanb,
                                                             y_b, g_coupling,
                                                             g_prime,
                                                             m_sbot_1,
                                                             m_sbot_2,
                                                             mbL, mbR,
                                                             a_b,
                                                             Q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_stau1(v_higgs, mu,
                                                          tanb, y_tau,
                                                          g_coupling, g_prime,
                                                          m_stau_1, m_stau_2,
                                                          mtauL, mtauR, a_tau,
                                                          Q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_stau2(v_higgs, mu, tanb,
                                                          y_tau, g_coupling,
                                                          g_prime, m_stau_1,
                                                          m_stau_2, mtauL,
                                                          mtauR, a_tau,
                                                          Q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_sup_L(g_coupling, g_prime,
                                                          msupL, Q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_sup_R(g_coupling, g_prime,
                                                          msupR, Q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_sdown_L(g_coupling,
                                                            g_prime, msdownL,
                                                            Q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_sdown_R(g_coupling,
                                                            g_prime, msdownR,
                                                            Q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_selec_L(g_coupling,
                                                            g_prime, mselecL,
                                                            Q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_selec_R(g_coupling,
                                                            g_prime, mselecR,
                                                            Q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_selecSneut(g_coupling,
                                                               g_prime,
                                                               mselecSneut,
                                                               Q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_sstrange_L(g_coupling,
                                                               g_prime,
                                                               msstrangeL,
                                                               Q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_sstrange_R(g_coupling,
                                                               g_prime,
                                                               msstrangeR,
                                                               Q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_scharm_L(g_coupling,
                                                             g_prime, mscharmL,
                                                             Q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_scharm_R(g_coupling,
                                                             g_prime, mscharmR,
                                                             Q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_smu_L(g_coupling, g_prime,
                                                          msmuL, Q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_smu_R(g_coupling, g_prime,
                                                          msmuR, Q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_smuSneut(g_coupling,
                                                             g_prime,
                                                             msmuSneut,
                                                             Q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_neutralino(M1, M2, mu,
                                                               g_coupling,
                                                               g_prime,
                                                               v_higgs, tanb,
                                                               msN1, Q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_neutralino(M1, M2, mu,
                                                               g_coupling,
                                                               g_prime,
                                                               v_higgs, tanb,
                                                               msN2, Q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_neutralino(M1, M2, mu,
                                                               g_coupling,
                                                               g_prime,
                                                               v_higgs, tanb,
                                                               msN3, Q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_neutralino(M1, M2, mu,
                                                               g_coupling,
                                                               g_prime,
                                                               v_higgs, tanb,
                                                               msN4, Q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_chargino1(g_coupling, M2,
                                                              v_higgs, tanb,
                                                              mu, msC1,
                                                              Q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_chargino2(g_coupling, M2,
                                                              v_higgs, tanb,
                                                              mu, msC2,
                                                              Q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_h0(g_coupling, g_prime,
                                                       v_higgs, tanb, mHusq,
                                                       mHdsq, mu, mZ, mh0,
                                                       Q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_H0(g_coupling, g_prime,
                                                       v_higgs, tanb, mHusq,
                                                       mHdsq, mu, mZ, mH0,
                                                       Q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_H_pm(g_coupling, mH_pm,
                                                         Q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_W_pm(g_coupling, v_higgs,
                                                         Q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_Z0(g_coupling, g_prime,
                                                       v_higgs, Q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_bottom(y_b, v_higgs, tanb,
                                                           Q_renorm),
                                            tanb),
                                   DEW_func(sigmadd_tau(y_tau, v_higgs, tanb,
                                                        Q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_stop1(v_higgs, mu, tanb,
                                                          y_t, g_coupling,
                                                          g_prime, m_stop_1,
                                                          m_stop_2, mtL, mtR,
                                                          a_t, Q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_stop2(v_higgs, mu, tanb,
                                                          y_t, g_coupling,
                                                          g_prime, m_stop_1,
                                                          m_stop_2, mtL, mtR,
                                                          a_t, Q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_sbottom1(v_higgs, mu, tanb,
                                                             y_b, g_coupling,
                                                             g_prime, m_sbot_1,
                                                             m_sbot_2, mbL,
                                                             mbR, a_b,
                                                             Q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_sbottom2(v_higgs, mu, tanb,
                                                             y_b, g_coupling,
                                                             g_prime, m_sbot_1,
                                                             m_sbot_2, mbL,
                                                             mbR, a_b,
                                                             Q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_stau1(v_higgs, mu, tanb,
                                                          y_tau, g_coupling,
                                                          g_prime, m_stau_1,
                                                          m_stau_2, mtauL,
                                                          mtauR, a_tau,
                                                          Q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_stau2(v_higgs, mu, tanb,
                                                          y_tau, g_coupling,
                                                          g_prime, m_stau_1,
                                                          m_stau_2, mtauL,
                                                          mtauR, a_tau,
                                                          Q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_sup_L(g_coupling, g_prime,
                                                          msupL, Q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_sup_R(g_coupling, g_prime,
                                                          msupR, Q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_sdown_L(g_coupling,
                                                            g_prime, msdownL,
                                                            Q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_sdown_R(g_coupling,
                                                            g_prime, msdownR,
                                                            Q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_selec_L(g_coupling,
                                                            g_prime, mselecL,
                                                            Q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_selec_R(g_coupling,
                                                            g_prime, mselecR,
                                                            Q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_selecSneut(g_coupling,
                                                               g_prime,
                                                               mselecSneut,
                                                               Q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_sstrange_L(g_coupling,
                                                               g_prime,
                                                               msstrangeL,
                                                               Q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_sstrange_R(g_coupling,
                                                               g_prime,
                                                               msstrangeR,
                                                               Q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_scharm_L(g_coupling,
                                                             g_prime, mscharmL,
                                                             Q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_scharm_R(g_coupling,
                                                             g_prime, mscharmR,
                                                             Q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_smu_L(g_coupling, g_prime,
                                                          msmuL, Q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_smu_R(g_coupling, g_prime,
                                                          msmuR, Q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_smuSneut(g_coupling,
                                                             g_prime,
                                                             msmuSneut,
                                                             Q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_neutralino(M1, M2, mu,
                                                               g_coupling,
                                                               g_prime,
                                                               v_higgs, tanb,
                                                               msN1, Q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_neutralino(M1, M2, mu,
                                                               g_coupling,
                                                               g_prime,
                                                               v_higgs, tanb,
                                                               msN2, Q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_neutralino(M1, M2, mu,
                                                               g_coupling,
                                                               g_prime,
                                                               v_higgs, tanb,
                                                               msN3, Q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_neutralino(M1, M2, mu,
                                                               g_coupling,
                                                               g_prime,
                                                               v_higgs, tanb,
                                                               msN4, Q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_chargino1(g_coupling,
                                                              M2, v_higgs,
                                                              tanb, mu, msC1,
                                                              Q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_chargino2(g_coupling,
                                                              M2, v_higgs,
                                                              tanb, mu, msC2,
                                                              Q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_h0(g_coupling, g_prime,
                                                       v_higgs, tanb, mHusq,
                                                       mHdsq, mu, mZ, mh0,
                                                       Q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_H0(g_coupling, g_prime,
                                                       v_higgs, tanb, mHusq,
                                                       mHdsq, mu, mZ, mH0,
                                                       Q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_H_pm(g_coupling, mH_pm,
                                                         Q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_W_pm(g_coupling, v_higgs,
                                                         Q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_Z0(g_coupling, g_prime,
                                                       v_higgs, Q_renorm),
                                            tanb),
                                   DEW_func(sigmauu_top(y_t, v_higgs, tanb,
                                                        Q_renorm),
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
                               DEW_func(sigmadd_sup_L(g, g_prime, msupL,
                                                      Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_sup_R(g, g_prime, msupR,
                                                      Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_sdown_L(g, g_prime,
                                                        msdownL,
                                                        Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_sdown_R(g, g_prime,
                                                        msdownR,
                                                        Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_selec_L(g, g_prime,
                                                        mselecL,
                                                        Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_selec_R(g, g_prime,
                                                        mselecR,
                                                        Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_selecSneut(g, g_prime,
                                                           mselecSneut,
                                                           Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_sstrange_L(g, g_prime,
                                                           msstrangeL,
                                                           Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_sstrange_R(g, g_prime,
                                                           msstrangeR,
                                                           Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_scharm_L(g, g_prime,
                                                         mscharmL,
                                                         Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_scharm_R(g, g_prime,
                                                         mscharmR,
                                                         Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_smu_L(g, g_prime, msmuL,
                                                      Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_smu_R(g, g_prime, msmuR,
                                                      Q_renorm),
                                        tanb),
                               DEW_func(sigmadd_smuSneut(g, g_prime,
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
                               DEW_func(sigmauu_sup_L(g, g_prime, msupL,
                                                      Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_sup_R(g, g_prime, msupR,
                                                      Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_sdown_L(g, g_prime,
                                                        msdownL, Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_sdown_R(g, g_prime,
                                                        msdownR, Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_selec_L(g, g_prime,
                                                        mselecL, Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_selec_R(g, g_prime,
                                                        mselecR, Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_selecSneut(g, g_prime,
                                                           mselecSneut,
                                                           Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_sstrange_L(g, g_prime,
                                                           msstrangeL,
                                                           Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_sstrange_R(g, g_prime,
                                                           msstrangeR,
                                                           Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_scharm_L(g, g_prime,
                                                         mscharmL,
                                                         Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_scharm_R(g, g_prime,
                                                         mscharmR,
                                                         Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_smu_L(g, g_prime, msmuL,
                                                      Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_smu_R(g, g_prime, msmuR,
                                                      Q_renorm),
                                        tanb),
                               DEW_func(sigmauu_smuSneut(g, g_prime,
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
                                  (DEW_func(sigmadd_sup_L(g, g_prime, msupL,
                                                          Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(sup_L)'),
                                  (DEW_func(sigmadd_sup_R(g, g_prime, msupR,
                                                          Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(sup_R)'),
                                  (DEW_func(sigmadd_sdown_L(g, g_prime,
                                                            msdownL,
                                                            Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(sdown_L)'),
                                  (DEW_func(sigmadd_sdown_R(g, g_prime,
                                                            msdownR,
                                                            Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(sdown_R)'),
                                  (DEW_func(sigmadd_selec_L(g, g_prime,
                                                            mselecL,
                                                            Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(selec_L)'),
                                  (DEW_func(sigmadd_selec_R(g, g_prime,
                                                            mselecR,
                                                            Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(selec_R)'),
                                  (DEW_func(sigmadd_selecSneut(g, g_prime,
                                                               mselecSneut,
                                                               Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(selec_neutr)'),
                                  (DEW_func(sigmadd_sstrange_L(g, g_prime,
                                                               msstrangeL,
                                                               Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(sstrange_L)'),
                                  (DEW_func(sigmadd_sstrange_R(g, g_prime,
                                                               msstrangeR,
                                                               Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(sstrange_R)'),
                                  (DEW_func(sigmadd_scharm_L(g, g_prime,
                                                             mscharmL,
                                                             Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(scharm_L)'),
                                  (DEW_func(sigmadd_scharm_R(g, g_prime,
                                                             mscharmR,
                                                             Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(scharm_R)'),
                                  (DEW_func(sigmadd_smu_L(g, g_prime, msmuL,
                                                          Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(smu_L)'),
                                  (DEW_func(sigmadd_smu_R(g, g_prime, msmuR,
                                                          Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_d^d(smu_R)'),
                                  (DEW_func(sigmadd_smuSneut(g, g_prime,
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
                                  (DEW_func(sigmauu_sup_L(g, g_prime, msupL,
                                                          Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(sup_L)'),
                                  (DEW_func(sigmauu_sup_R(g, g_prime, msupR,
                                                          Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(sup_R)'),
                                  (DEW_func(sigmauu_sdown_L(g, g_prime,
                                                            msdownL, Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(sdown_L)'),
                                  (DEW_func(sigmauu_sdown_R(g, g_prime,
                                                            msdownR, Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(sdown_R)'),
                                  (DEW_func(sigmauu_selec_L(g, g_prime,
                                                            mselecL, Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(selec_L)'),
                                  (DEW_func(sigmauu_selec_R(g, g_prime,
                                                            mselecR, Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(selec_R)'),
                                  (DEW_func(sigmauu_selecSneut(g, g_prime,
                                                               mselecSneut,
                                                               Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(selec_neutr)'),
                                  (DEW_func(sigmauu_sstrange_L(g, g_prime,
                                                               msstrangeL,
                                                               Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(sstrange_L)'),
                                  (DEW_func(sigmauu_sstrange_R(g, g_prime,
                                                               msstrangeR,
                                                               Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(sstrange_R)'),
                                  (DEW_func(sigmauu_scharm_L(g, g_prime,
                                                             mscharmL,
                                                             Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(scharm_L)'),
                                  (DEW_func(sigmauu_scharm_R(g, g_prime,
                                                             mscharmR,
                                                             Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(scharm_R)'),
                                  (DEW_func(sigmauu_smu_L(g, g_prime, msmuL,
                                                          Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(smu_L)'),
                                  (DEW_func(sigmauu_smu_R(g, g_prime, msmuR,
                                                          Q_renorm),
                                            tanb) / halfmzsq,
                                   'Sigma_u^u(smu_R)'),
                                  (DEW_func(sigmauu_smuSneut(g, g_prime,
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
