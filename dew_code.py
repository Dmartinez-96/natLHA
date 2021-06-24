# -*- coding: utf-8 -*-
"""Compute naturalness measure Delta_EW and top ten contributions to DEW."""

import numpy as np
import pyslha
# SLHA input and definition of variables from SLHA file: #

direc = input('Enter the full directory for your SLHA file: ')
d = pyslha.read(direc)
[vHiggs, muQ] = [d.blocks['HMIX'][3], d.blocks['HMIX'][1]]
[tanb, y_t] = [d.blocks['HMIX'][2], d.blocks['YU'][3, 3]]
[y_b, y_tau] = [d.blocks['YD'][3, 3], d.blocks['YE'][3, 3]]
[g_pr, g_EW] = [d.blocks['GAUGE'][2], d.blocks['GAUGE'][1]]
[m_stop_1, m_stop_2] = [d.blocks['MASS'][1000006], d.blocks['MASS'][2000006]]
[m_sbot_1, m_sbot_2] = [d.blocks['MASS'][1000005], d.blocks['MASS'][2000005]]
[m_stau_1, m_stau_2] = [d.blocks['MASS'][1000015], d.blocks['MASS'][2000015]]
[mtL, mtR] = [d.blocks['MSOFT'][43], d.blocks['MSOFT'][46]]
[mbL, mbR] = [d.blocks['MSOFT'][43], d.blocks['MSOFT'][49]]
[mtauL, mtauR] = [d.blocks['MSOFT'][33], d.blocks['MSOFT'][36]]
[msupL, msupR] = [d.blocks['MSOFT'][41], d.blocks['MSOFT'][44]]
[msdownL, msdownR] = [d.blocks['MSOFT'][41], d.blocks['MSOFT'][47]]
[mselecL, mselecR] = [d.blocks['MSOFT'][31], d.blocks['MSOFT'][34]]
[mselecneut, msmuneut] = [d.blocks['MASS'][1000012], d.blocks['MASS'][1000014]]
[msstrangeL, msstrangeR] = [d.blocks['MSOFT'][42], d.blocks['MSOFT'][48]]
[mscharmL, mscharmR] = [d.blocks['MSOFT'][42], d.blocks['MSOFT'][45]]
[msmuL, msmuR] = [d.blocks['MSOFT'][32], d.blocks['MSOFT'][35]]
[msN1, msN2] = [d.blocks['MASS'][1000022], d.blocks['MASS'][1000023]]
[msN3, msN4] = [d.blocks['MASS'][1000025], d.blocks['MASS'][1000035]]
[msC1, msC2] = [d.blocks['MASS'][1000024], d.blocks['MASS'][1000037]]
[mZ, mh0] = [d.blocks['SMINPUTS'][4], d.blocks['MASS'][25]]
[mH0, mHusq] = [d.blocks['MASS'][35], d.blocks['MSOFT'][22]]
[mHdsq, mH_pm] = [d.blocks['MSOFT'][21], d.blocks['MASS'][37]]
[M_1, M_2] = [d.blocks['MSOFT'][1], d.blocks['MSOFT'][2]]
[a_t, a_b] = [d.blocks['AU'][3, 3] * y_t, d.blocks['AD'][3, 3] * y_b]
a_tau = d.blocks['AE'][3, 3] * y_tau
[Q_renorm, halfmzsq] = [np.sqrt(m_stop_1 * m_stop_2), np.power(mZ, 2) / 2]


# Mass relations: #

def m_w_sq():
    """Return W boson squared mass."""
    my_mw_sq = (np.power(g_EW, 2) / 2) * np.power(vHiggs, 2)
    return my_mw_sq


def mzsq():
    """Return Z boson squared mass."""
    return ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2) * np.power(vHiggs, 2)


def ma_0sq():
    """Return A_0 squared mass."""
    my_ma0_sq = 2 * np.power(np.abs(muQ), 2) + mHusq + mHdsq
    return my_ma0_sq


# Fundamental equations: #

def logfunc(mass):
    """
    Return F = m^2 * (ln(m^2 / Q^2) - 1).

    Parameters
    ----------
    mass : Input mass.

    """
    myf = np.power(mass, 2) * (np.log((np.power(mass, 2))
                                      / (np.power(Q_renorm, 2))) - 1)
    return myf


def sinsqb():
    """Return sin^2(beta)."""
    mysinsqb = (np.power(tanb, 2) / (1 + np.power(tanb, 2)))
    return mysinsqb


def cossqb():
    """Return cos^2(beta)."""
    mycossqb = 1 - (np.power(tanb, 2) / (1 + np.power(tanb, 2)))
    return mycossqb


def v_higgs_u():
    """Return up-type Higgs VEV."""
    myvu = vHiggs * np.sqrt(sinsqb())
    return myvu


def v_higgs_d():
    """Return down-type Higgs VEV."""
    myvd = vHiggs * np.sqrt(cossqb())
    return myvd


def tan_theta_w():
    """Return tan(theta_W), the Weinberg angle."""
    mytanthetaw = g_pr / g_EW
    return mytanthetaw


def sinsq_theta_w():
    """Return sin^2(theta_W), the Weinberg angle."""
    mysinsqthetaw = (np.power(tan_theta_w(), 2)
                     / (1 + np.power(tan_theta_w(), 2)))
    return mysinsqthetaw


# Stop squarks: #

def sigmauu_stop1():
    """Return one-loop correction Sigma_u^u(stop_1)."""
    delta_ul = ((1 / 2) - (2 / 3) * sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)\
        * (np.power(v_higgs_d(), 2) - np.power(v_higgs_u(), 2))
    delta_ur = (2 / 3) * sinsq_theta_w()\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)\
        * (np.power(v_higgs_d(), 2) - np.power(v_higgs_u(), 2))
    stop_num = ((np.power(mtL, 2) - np.power(mtR, 2) + delta_ul - delta_ur)
                * ((-1 / 2) + ((4 / 3) * sinsq_theta_w()))
                * (np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)\
        + 2 * np.power(a_t, 2)
    sigmauu_stop_1 = (3 / (32 * (np.power(np.pi, 2))))\
        * (2 * np.power(y_t, 2) + ((np.power(g_EW, 2) + np.power(g_pr, 2))
                                   * (8 * sinsq_theta_w() - 3) / 12)
           - (stop_num / (np.power(m_stop_2, 2) - np.power(m_stop_1, 2))))\
        * logfunc(m_stop_1)
    return sigmauu_stop_1


def sigmadd_stop1():
    """Return one-loop correction Sigma_d^d(stop_1)."""
    delta_ul = ((1 / 2) - (2 / 3) * sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)\
        * (np.power(v_higgs_d(), 2) - np.power(v_higgs_u(), 2))
    delta_ur = (2 / 3) * sinsq_theta_w()\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)\
        * (np.power(v_higgs_d(), 2) - np.power(v_higgs_u(), 2))
    stop_num = ((np.power(mtL, 2) - np.power(mtR, 2) + delta_ul - delta_ur)
                * ((1 / 2) + (4 / 3) * sinsq_theta_w())
                * (np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)\
        + 2 * np.power((y_t * muQ), 2)
    sigmadd_stop_1 = (3 / (32 * (np.power(np.pi, 2))))\
        * (((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)
           - (stop_num / (np.power(m_stop_2, 2) - np.power(m_stop_1, 2))))\
        * logfunc(m_stop_1)
    return sigmadd_stop_1


def sigmauu_stop2():
    """Return one-loop correction Sigma_u^u(stop_2)."""
    delta_ul = ((1 / 2) - (2 / 3) * sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)\
        * (np.power(v_higgs_d(), 2) - np.power(v_higgs_u(), 2))
    delta_ur = (2 / 3) * sinsq_theta_w()\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)\
        * (np.power(v_higgs_d(), 2) - np.power(v_higgs_u(), 2))
    stop_num = ((np.power(mtL, 2) - np.power(mtR, 2) + delta_ul - delta_ur)
                * ((-1 / 2) + ((4 / 3) * sinsq_theta_w()))
                * (np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)\
        + 2 * np.power(a_t, 2)
    sigmauu_stop_2 = (3 / (32 * (np.power(np.pi, 2))))\
        * (2 * np.power(y_t, 2) + ((np.power(g_EW, 2) + np.power(g_pr, 2))
                                   * (8 * sinsq_theta_w() - 3) / 12)
           + (stop_num / (np.power(m_stop_2, 2) - np.power(m_stop_1, 2))))\
        * logfunc(m_stop_2)
    return sigmauu_stop_2


def sigmadd_stop2():
    """Return one-loop correction Sigma_d^d(stop_2)."""
    delta_ul = ((1 / 2) - (2 / 3) * sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)\
        * (np.power(v_higgs_d(), 2) - np.power(v_higgs_u(), 2))
    delta_ur = (2 / 3) * sinsq_theta_w()\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)\
        * (np.power(v_higgs_d(), 2) - np.power(v_higgs_u(), 2))
    stop_num = ((np.power(mtL, 2) - np.power(mtR, 2) + delta_ul - delta_ur)
                * ((1 / 2) + (4 / 3) * sinsq_theta_w())
                * (np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)\
        + 2 * np.power(y_t, 2) * np.power(muQ, 2)
    sigmadd_stop_2 = (3 / (32 * (np.power(np.pi, 2))))\
        * (((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)
           - (stop_num / (np.power(m_stop_2, 2) - np.power(m_stop_1, 2))))\
        * logfunc(m_stop_2)
    return sigmadd_stop_2


# Sbottom squarks: #

def sigmauu_sbottom1():
    """Return one-loop correction Sigma_u^u(sbottom_1)."""
    delta_dl = ((-1 / 2) + (1 / 3) * sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)\
        * (np.power(v_higgs_d(), 2) - np.power(v_higgs_u(), 2))
    delta_dr = ((-1 / 3) * sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)\
        * (np.power(v_higgs_d(), 2) - np.power(v_higgs_u(), 2))
    sbot_num = (np.power(mbR, 2) - np.power(mbL, 2) + delta_dr - delta_dl)\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)\
        * ((1 / 2) + (2 / 3) * sinsq_theta_w()) + 2 * np.power((y_b * muQ), 2)
    sigmauu_sbot = (3 / (32 * np.power(np.pi, 2)))\
        * (((np.power(g_EW, 2) + np.power(g_pr, 2)) / 4)
           - (sbot_num / (np.power(m_sbot_2, 2) - np.power(m_sbot_1, 2))))\
        * logfunc(m_sbot_1)
    return sigmauu_sbot


def sigmauu_sbottom2():
    """Return one-loop correction Sigma_u^u(sbottom_2)."""
    delta_dl = ((-1 / 2) + (1 / 3) * sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)\
        * (np.power(v_higgs_d(), 2) - np.power(v_higgs_u(), 2))
    delta_dr = ((-1 / 3) * sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)\
        * (np.power(v_higgs_d(), 2) - np.power(v_higgs_u(), 2))
    sbot_num = (np.power(mbR, 2) - np.power(mbL, 2) + delta_dr - delta_dl)\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)\
        * ((1 / 2) + (2 / 3) * sinsq_theta_w())\
        + 2 * np.power(y_b, 2) * np.power(muQ, 2)
    sigmauu_sbot = (3 / (32 * np.power(np.pi, 2)))\
        * (((np.power(g_EW, 2) + np.power(g_pr, 2)) / 4)
           + (sbot_num / (np.power(m_sbot_2, 2) - np.power(m_sbot_1, 2))))\
        * logfunc(m_sbot_2)
    return sigmauu_sbot


def sigmadd_sbottom1():
    """Return one-loop correction Sigma_d^d(sbottom_1)."""
    delta_dl = ((-1 / 2) + (1 / 3) * sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)\
        * (np.power(v_higgs_d(), 2) - np.power(v_higgs_u(), 2))
    delta_dr = ((-1 / 3) * sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)\
        * (np.power(v_higgs_d(), 2) - np.power(v_higgs_u(), 2))
    sbot_num = (np.power(mbR, 2) - np.power(mbL, 2) + delta_dr - delta_dl)\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)\
        * ((-1 / 2) - (2 / 3) * sinsq_theta_w()) + 2 * np.power(a_b, 2)
    sigmadd_sbot = (3 / (32 * np.power(np.pi, 2)))\
        * (((-1) * (np.power(g_EW, 2) + np.power(g_pr, 2)) / 4)
           - (sbot_num / (np.power(m_sbot_2, 2) - np.power(m_sbot_1, 2))))\
        * logfunc(m_sbot_1)
    return sigmadd_sbot


def sigmadd_sbottom2():
    """Return one-loop correction Sigma_d^d(sbottom_2)."""
    delta_dl = ((-1 / 2) + (1 / 3) * sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)\
        * (np.power(v_higgs_d(), 2) - np.power(v_higgs_u(), 2))
    delta_dr = ((-1 / 3) * sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)\
        * (np.power(v_higgs_d(), 2) - np.power(v_higgs_u(), 2))
    sbot_num = (np.power(mbR, 2) - np.power(mbL, 2) + delta_dr - delta_dl)\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)\
        * ((-1 / 2) - (2 / 3) * sinsq_theta_w()) + 2 * np.power(a_b, 2)
    sigmadd_sbot = (3 / (32 * np.power(np.pi, 2)))\
        * (((-1) * (np.power(g_EW, 2) + np.power(g_pr, 2)) / 4)
           + (sbot_num / (np.power(m_sbot_2, 2) - np.power(m_sbot_1, 2))))\
        * logfunc(m_sbot_2)
    return sigmadd_sbot


# Stau sleptons: #

def sigmauu_stau1():
    """Return one-loop correction Sigma_u^u(stau_1)."""
    delta_el = ((-1 / 2) + sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)\
        * (np.power(v_higgs_d(), 2) - np.power(v_higgs_u(), 2))
    delta_er = ((-1) * sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)\
        * (np.power(v_higgs_d(), 2) - np.power(v_higgs_u(), 2))
    stau_num = (np.power(mtauR, 2) - np.power(mtauL, 2) + delta_er - delta_el)\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)\
        * ((-1 / 2) + 2 * sinsq_theta_w()) + 2 * np.power((y_tau * muQ), 2)
    sigmauu_stau = (1 / (32 * np.power(np.pi, 2)))\
        * (((np.power(g_EW, 2) + np.power(g_pr, 2)) / 4)
           - (stau_num / (np.power(m_stau_2, 2) - np.power(m_stau_1, 2))))\
        * logfunc(m_stau_1)
    return sigmauu_stau


def sigmauu_stau2():
    """Return one-loop correction Sigma_u^u(stau_2)."""
    delta_el = ((-1 / 2) + sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)\
        * (np.power(v_higgs_d(), 2) - np.power(v_higgs_u(), 2))
    delta_er = ((-1) * sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)\
        * (np.power(v_higgs_d(), 2) - np.power(v_higgs_u(), 2))
    stau_num = (np.power(mtauR, 2) - np.power(mtauL, 2) + delta_er - delta_el)\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)\
        * ((-1 / 2) + 2 * sinsq_theta_w()) + 2 * np.power((y_tau * muQ), 2)
    sigmauu_stau = (1 / (32 * np.power(np.pi, 2)))\
        * (((np.power(g_EW, 2) + np.power(g_pr, 2)) / 4)
           + (stau_num / (np.power(m_stau_2, 2) - np.power(m_stau_1, 2))))\
        * logfunc(m_stau_2)
    return sigmauu_stau


def sigmadd_stau1():
    """Return one-loop correction Sigma_d^d(stau_1)."""
    delta_el = ((-1 / 2) + sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)\
        * (np.power(v_higgs_d(), 2) - np.power(v_higgs_u(), 2))
    delta_er = ((-1) * sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)\
        * (np.power(v_higgs_d(), 2) - np.power(v_higgs_u(), 2))
    stau_num = (np.power(mtauR, 2) - np.power(mtauL, 2) + delta_er - delta_el)\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)\
        * ((1 / 2) - (2 * sinsq_theta_w())) + 2 * np.power(a_tau, 2)
    sigmadd_stau = (1 / (32 * np.power(np.pi, 2)))\
        * (((-1) * (np.power(g_EW, 2) + np.power(g_pr, 2)) / 4)
           - (stau_num / (np.power(m_stau_2, 2) - np.power(m_stau_1, 2))))\
        * logfunc(m_stau_1)
    return sigmadd_stau


def sigmadd_stau2():
    """Return one-loop correction Sigma_d^d(stau_2)."""
    delta_el = ((-1 / 2) + sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)\
        * (np.power(v_higgs_d(), 2) - np.power(v_higgs_u(), 2))
    delta_er = ((-1) * sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)\
        * (np.power(v_higgs_d(), 2) - np.power(v_higgs_u(), 2))
    stau_num = (np.power(mtauR, 2) - np.power(mtauL, 2) + delta_er - delta_el)\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)\
        * ((1 / 2) - (2 * sinsq_theta_w())) + 2 * np.power(a_tau, 2)
    sigmadd_stau = (1 / (32 * np.power(np.pi, 2)))\
        * (((-1) * (np.power(g_EW, 2) + np.power(g_pr, 2)) / 4)
           + (stau_num / (np.power(m_stau_2, 2) - np.power(m_stau_1, 2))))\
        * logfunc(m_stau_2)
    return sigmadd_stau


# Sfermions, 1st gen: #

def sigmauu_sup_l():
    """Return one-loop correction Sigma_u^u(sup_L)."""
    sigmauusup_l = ((-3) / (16 * np.power(np.pi, 2)))\
        * ((1 / 2) - (2 / 3) * sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2) * logfunc(msupL)
    return sigmauusup_l


def sigmauu_sup_r():
    """Return one-loop correction Sigma_u^u(sup_R)."""
    sigmauusup_r = ((-3) / (16 * np.power(np.pi, 2)))\
        * ((2 / 3) * sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2) * logfunc(msupR)
    return sigmauusup_r


def sigmauu_sdown_l():
    """Return one-loop correction Sigma_u^u(sdown_L)."""
    sigmauusdown_l = ((-3) / (16 * np.power(np.pi, 2)))\
        * (((-1) / 2) + (1 / 3) * sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2) * logfunc(msdownL)
    return sigmauusdown_l


def sigmauu_sdown_r():
    """Return one-loop correction Sigma_u^u(sdown_R)."""
    sigmauusdown_r = ((-3) / (16 * np.power(np.pi, 2)))\
        * (((-1) / 3) * sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2) * logfunc(msdownR)
    return sigmauusdown_r


def sigmauu_selec_l():
    """Return one-loop correction Sigma_u^u(selectron_L)."""
    sigmauuselec_l = ((-1) / (16 * np.power(np.pi, 2)))\
        * (((-1) / 2) + sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2) * logfunc(mselecL)
    return sigmauuselec_l


def sigmauu_selec_r():
    """Return one-loop correction Sigma_u^u(selectron_R)."""
    sigmauuselec_r = ((-1) / (16 * np.power(np.pi, 2)))\
        * ((-1) * sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2) * logfunc(mselecR)
    return sigmauuselec_r


def sigmauu_sel_neut():
    """Return one-loop correction Sigma_u^u(selectron neutrino)."""
    sigmauuselec_sneut = ((-1) / (32 * np.power(np.pi, 2))) * (1 / 2)\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2) * logfunc(mselecneut)
    return sigmauuselec_sneut


def sigmadd_sup_l():
    """Return one-loop correction Sigma_d^d(sup_L)."""
    sigmaddsup_l = (3 / (16 * np.power(np.pi, 2)))\
        * ((1 / 2) - (2 / 3) * sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2) * logfunc(msupL)
    return sigmaddsup_l


def sigmadd_sup_r():
    """Return one-loop correction Sigma_d^d(sup_R)."""
    sigmaddsup_r = (3 / (16 * np.power(np.pi, 2)))\
        * ((2 / 3) * sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2) * logfunc(msupR)
    return sigmaddsup_r


def sigmadd_sdown_l():
    """Return one-loop correction Sigma_d^d(sdown_L)."""
    sigmaddsdown_l = (3 / (16 * np.power(np.pi, 2)))\
        * (((-1) / 2) + (1 / 3) * sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2) * logfunc(msdownL)
    return sigmaddsdown_l


def sigmadd_sdown_r():
    """Return one-loop correction Sigma_d^d(sdown_R)."""
    sigmaddsdown_r = (3 / (16 * np.power(np.pi, 2)))\
        * (((-1) / 3) * sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2) * logfunc(msdownR)
    return sigmaddsdown_r


def sigmadd_selec_l():
    """Return one-loop correction Sigma_d^d(selectron_L)."""
    sigmaddselec_l = (1 / (16 * np.power(np.pi, 2)))\
        * (((-1) / 2) + sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2) * logfunc(mselecL)
    return sigmaddselec_l


def sigmadd_selec_r():
    """Return one-loop correction Sigma_d^d(selectron_R)."""
    sigmaddselec_r = (1 / (16 * np.power(np.pi, 2)))\
        * ((-1) * sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2) * logfunc(mselecR)
    return sigmaddselec_r


def sigmadd_sel_neut():
    """Return one-loop correction Sigma_d^d(selectron neutrino)."""
    sigmaddselec_sneut = (1 / (32 * np.power(np.pi, 2))) * (1 / 2)\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2) * logfunc(mselecneut)
    return sigmaddselec_sneut


# Sfermions, 2nd gen: #

def sigmauu_sstrange_l():
    """Return one-loop correction Sigma_u^u(sstrange_L)."""
    sigmauusstrange_l = ((-3) / (16 * np.power(np.pi, 2)))\
        * (((-1) / 2) + (1 / 3) * sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2) * logfunc(msstrangeL)
    return sigmauusstrange_l


def sigmauu_sstrange_r():
    """Return one-loop correction Sigma_u^u(sstrange_R)."""
    sigmauusstrange_r = ((-3) / (16 * np.power(np.pi, 2)))\
        * (((-1) / 3) * sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2) * logfunc(msstrangeR)
    return sigmauusstrange_r


def sigmauu_scharm_l():
    """Return one-loop correction Sigma_u^u(scharm_L)."""
    sigmauuscharm_l = ((-3) / (16 * np.power(np.pi, 2)))\
        * ((1 / 2) - (2 / 3) * sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2) * logfunc(mscharmL)
    return sigmauuscharm_l


def sigmauu_scharm_r():
    """Return one-loop correction Sigma_u^u(scharm_R)."""
    sigmauuscharm_r = ((-3) / (16 * np.power(np.pi, 2)))\
        * ((2 / 3) * sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2) * logfunc(mscharmR)
    return sigmauuscharm_r


def sigmauu_smu_l():
    """Return one-loop correction Sigma_u^u(smu_L)."""
    sigmauusmu_l = ((-1) / (16 * np.power(np.pi, 2)))\
        * (((-1) / 2) + sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2) * logfunc(msmuL)
    return sigmauusmu_l


def sigmauu_smu_r():
    """Return one-loop correction Sigma_u^u(smu_R)."""
    sigmauusmu_r = ((-1) / (16 * np.power(np.pi, 2)))\
        * (((-1)) * sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2) * logfunc(msmuR)
    return sigmauusmu_r


def sigmauu_smu_sneut():
    """Return one-loop correction Sigma_u^u(smuon neutrino)."""
    sigmauusmu_sneut = ((-1) / (32 * np.power(np.pi, 2))) * (1 / 2)\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2) * logfunc(msmuneut)
    return sigmauusmu_sneut


def sigmadd_sstrange_l():
    """Return one-loop correction Sigma_d^d(sstrange_L)."""
    sigmaddsstrange_l = (3 / (16 * np.power(np.pi, 2)))\
        * (((-1) / 2) + (1 / 3) * sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2) * logfunc(msstrangeL)
    return sigmaddsstrange_l


def sigmadd_sstrange_r():
    """Return one-loop correction Sigma_d^d(sstrange_R)."""
    sigmaddsstrange_r = (3 / (16 * np.power(np.pi, 2)))\
        * (((-1) / 3) * sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2) * logfunc(msstrangeR)
    return sigmaddsstrange_r


def sigmadd_scharm_l():
    """Return one-loop correction Sigma_d^d(scharm_L)."""
    sigmaddscharm_l = (3 / (16 * np.power(np.pi, 2)))\
        * ((1 / 2) - (2 / 3) * sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2) * logfunc(mscharmL)
    return sigmaddscharm_l


def sigmadd_scharm_r():
    """Return one-loop correction Sigma_d^d(scharm_R)."""
    sigmaddscharm_r = (3 / (16 * np.power(np.pi, 2)))\
        * ((2 / 3) * sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2) * logfunc(mscharmR)
    return sigmaddscharm_r


def sigmadd_smu_l():
    """Return one-loop correction Sigma_d^d(smu_L)."""
    sigmaddsmu_l = (1 / (16 * np.power(np.pi, 2)))\
        * (((-1) / 2) + sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2) * logfunc(msmuL)
    return sigmaddsmu_l


def sigmadd_smu_r():
    """Return one-loop correction Sigma_d^d(smu_R)."""
    sigmaddsmu_r = (1 / (16 * np.power(np.pi, 2))) * ((-1) * sinsq_theta_w())\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2) * logfunc(msmuR)
    return sigmaddsmu_r


def sigmadd_smu_sneut():
    """Return one-loop correction Sigma_d^d(smuon neutrino)."""
    sigmaddsmu_sneut = (1 / (32 * np.power(np.pi, 2))) * (1 / 2)\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2) * logfunc(msmuneut)
    return sigmaddsmu_sneut


# Neutralinos: #
# Use method by Ibrahim and Nath for eigenvalue derivatives.

def neutralinouu_deriv_num(msn):
    """
    Return numerator for one-loop uu correction derivative term of neutralino.

    Parameters
    ----------
    msn : Neutralino mass.

    """
    cubicterm = np.power(g_EW, 2) + np.power(g_pr, 2)
    quadrterm = (((np.power(g_EW, 2) * M_2 * muQ)
                  + (np.power(g_pr, 2) * M_1 * muQ)) / (tanb))\
        - ((np.power((g_EW * M_1), 2)) + (np.power((g_pr * M_2), 2))
           + ((np.power(g_EW, 2) + np.power(g_pr, 2)) * (np.power(muQ, 2)))
           + (np.power((np.power(g_EW, 2) + np.power(g_pr, 2)), 2) / 2)
           * np.power(vHiggs, 2))
    linterm = (((-1) * muQ) * ((np.power(g_EW, 2) * M_2
                                * (np.power(M_1, 2) + np.power(muQ, 2)))
                               + np.power(g_pr, 2) * M_1
                               * (np.power(M_2, 2) + np.power(muQ, 2)))
               / tanb)\
        + ((np.power((np.power(g_EW, 2) * M_1 + np.power(g_pr, 2) * M_2), 2)
            / 2) * np.power(vHiggs, 2)) + (np.power(muQ, 2)
                                           * ((np.power((g_EW * M_1), 2))
                                              + (np.power((g_pr * M_2), 2))))\
        + (np.power((np.power(g_EW, 2) + np.power(g_pr, 2)), 2)
           * np.power((vHiggs * muQ), 2) * cossqb())
    constterm = (M_1 * M_2 * ((np.power(g_EW, 2) * M_1)
                              + (np.power(g_pr, 2) * M_2))
                 * np.power(muQ, 3) * (1 / tanb))\
        - (np.power((np.power(g_EW, 2) * M_1 + np.power(g_pr, 2) * M_2), 2)
           * np.power(vHiggs, 2) * np.power(muQ, 2) * cossqb())
    mynum = (cubicterm * np.power(msn, 6)) + (quadrterm * np.power(msn, 4))\
        + (linterm * np.power(msn, 2)) + constterm
    return mynum


def neutralinodd_deriv_num(msn):
    """
    Return numerator for one-loop dd correction derivative term of neutralino.

    Parameters
    ----------
    msn : Neutralino mass.

    """
    cubicterm = np.power(g_EW, 2) + np.power(g_pr, 2)
    quadrterm = (((np.power(g_EW, 2) * M_2 * muQ) + (np.power(g_pr, 2) * M_1
                                                     * muQ)) * (tanb))\
        - ((np.power((g_EW * M_1), 2)) + (np.power((g_pr * M_2), 2))
           + ((np.power(g_EW, 2) + np.power(g_pr, 2)) * (np.power(muQ, 2)))
           + (np.power((np.power(g_EW, 2) + np.power(g_pr, 2)), 2) / 2)
           * np.power(vHiggs, 2))
    linterm = (((-1) * muQ) * ((np.power(g_EW, 2) * M_2
                                * (np.power(M_1, 2) + np.power(muQ, 2)))
                               + np.power(g_pr, 2) * M_1
                               * (np.power(M_2, 2) + np.power(muQ, 2)))
               * tanb)\
        + ((np.power((np.power(g_EW, 2) * M_1 + np.power(g_pr, 2) * M_2), 2)
            / 2) * np.power(vHiggs, 2))\
        + (np.power(muQ, 2) * ((np.power((g_EW * M_1), 2))
           + np.power(g_pr, 2) * np.power(M_2, 2)))\
        + (np.power((np.power(g_EW, 2) + np.power(g_pr, 2)), 2)
           * np.power((vHiggs * muQ), 2) * sinsqb())
    constterm = (M_1 * M_2 * (np.power(g_EW, 2) * M_1 + (np.power(g_pr, 2)
                                                         * M_2))
                 * np.power(muQ, 3) * tanb)\
        - (np.power((np.power(g_EW, 2) * M_1 + np.power(g_pr, 2) * M_2), 2)
           * np.power((vHiggs * muQ), 2) * sinsqb())
    mynum = (cubicterm * np.power(msn, 6)) + (quadrterm * np.power(msn, 4))\
        + (linterm * np.power(msn, 2)) + constterm
    return mynum


def neutralino_deriv_denom(msn):
    """
    Return denominator for one-loop correction derivative term of neutralino.

    Parameters
    ----------
    msn : Neutralino mass.

    """
    quadrterm = -3 * ((np.power(M_1, 2)) + (np.power(M_2, 2))
                      + ((np.power(g_EW, 2) + np.power(g_pr, 2))
                         * np.power(vHiggs, 2)) + (2 * np.power(muQ, 2)))
    linterm = (np.power(vHiggs, 4)
               * np.power((np.power(g_EW, 2) + np.power(g_pr, 2)), 2) / 2)\
        + (np.power(vHiggs, 2)
           * (2 * ((np.power((g_EW * M_1), 2)) + (np.power((g_pr * M_2), 2))
                   + ((np.power(g_EW, 2) + np.power(g_pr, 2))
                      * np.power(muQ, 2))
                   - (muQ * (np.power(g_pr, 2) * M_1 + np.power(g_EW, 2) * M_2)
                      * 2 * np.sqrt(sinsqb()) * np.sqrt(cossqb())))))\
        + (2 * ((np.power((M_1 * M_2), 2))
                + (2 * (np.power((M_1 * muQ), 2) + np.power((M_2 * muQ), 2)))
                + (np.power(muQ, 4))))
    constterm = (np.power(vHiggs, 4) * (1 / 8)
                 * ((np.power((np.power(g_EW, 2) + np.power(g_pr, 2)), 2)
                     * np.power(muQ, 2) * (np.power(cossqb(), 2)
                                           - (6 * cossqb() * sinsqb())
                                           + np.power(sinsqb(), 2)))
                    - (2 * np.power((np.power(g_EW, 2) * M_1
                                     + np.power(g_pr, 2) * M_2), 2))
                    - (np.power(muQ, 2)
                       * np.power((np.power(g_EW, 2) + np.power(g_pr, 2)), 2))
                    ))\
        + (np.power(vHiggs, 2) * 2 * muQ
           * ((np.sqrt(cossqb()) * np.sqrt(sinsqb()))
              * (np.power(g_EW, 2) * M_2
                 * (np.power(M_1, 2) + np.power(muQ, 2))
                 + (np.power(g_pr, 2) * M_1
                 * (np.power(M_2, 2) + np.power(muQ, 2))))))\
        - ((2 * np.power((M_2 * M_1 * muQ), 2))
           + (np.power(muQ, 4) * (np.power(M_1, 2) + np.power(M_2, 2))))
    mydenom = 4 * np.power(msn, 6) + quadrterm * np.power(msn, 4)\
        + linterm * np.power(msn, 2) + constterm
    return mydenom


def sigmauu_neutralino(msn):
    """
    Return one-loop correction Sigma_u^u(neutralino).

    Parameters
    ----------
    msn : Neutralino mass.

    """
    sigma_uu_neutralino = ((-1) / (16 * np.power(np.pi, 2))) \
        * (neutralinouu_deriv_num(msn)
           / neutralino_deriv_denom(msn))\
        * logfunc(msn)
    return sigma_uu_neutralino


def sigmadd_neutralino(msn):
    """
    Return one-loop correction Sigma_d^d(neutralino).

    Parameters
    ----------
    msn : Neutralino mass.

    """
    sigma_dd_neutralino = ((-1) / (16 * np.power(np.pi, 2))) \
        * (neutralinodd_deriv_num(msn)
           / neutralino_deriv_denom(msn))\
        * logfunc(msn)
    return sigma_dd_neutralino


# Charginos: #

def sigmauu_chargino1():
    """Return one-loop correction Sigma_u^u(chargino_1)."""
    chargino_num = np.power(M_2, 2) + np.power(muQ, 2)\
        + (np.power(g_EW, 2) * (np.power(v_higgs_u(), 2)
                                - np.power(v_higgs_d(), 2)))
    chargino_den = np.sqrt(((np.power(g_EW, 2)
                             * np.power((v_higgs_u() + v_higgs_d()), 2))
                            + np.power((M_2 - muQ), 2))
                           * ((np.power(g_EW, 2) * np.power((v_higgs_d()
                                                            - v_higgs_u()), 2))
                              + np.power((M_2 + muQ), 2)))
    sigma_uu_chargino1 = -1 * (np.power(g_EW, 2) / (16 * np.power(np.pi, 2)))\
        * (1 - (chargino_num / chargino_den)) * logfunc(msC1)
    return sigma_uu_chargino1


def sigmauu_chargino2():
    """Return one-loop correction Sigma_u^u(chargino_2)."""
    chargino_num = np.power(M_2, 2) + np.power(muQ, 2)\
        + (np.power(g_EW, 2) * (np.power(v_higgs_u(), 2)
                                - np.power(v_higgs_d(), 2)))
    chargino_den = np.sqrt(((np.power(g_EW, 2)
                             * np.power((v_higgs_u() + v_higgs_d()), 2))
                            + np.power((M_2 - muQ), 2))
                           * ((np.power(g_EW, 2)
                               * np.power((v_higgs_d() - v_higgs_u()), 2))
                              + np.power((M_2 + muQ), 2)))
    sigma_uu_chargino2 = -1 * (np.power(g_EW, 2) / (16 * np.power(np.pi, 2)))\
        * (1 + (chargino_num / chargino_den)) * logfunc(msC2)
    return sigma_uu_chargino2


def sigmadd_chargino1():
    """Return one-loop correction Sigma_d^d(chargino_1)."""
    chargino_num = np.power(M_2, 2) + np.power(muQ, 2)\
        - (np.power(g_EW, 2) * (np.power(v_higgs_u(), 2)
                                - np.power(v_higgs_d(), 2)))
    chargino_den = np.sqrt(((np.power(g_EW, 2)
                             * np.power((v_higgs_u()
                                         + v_higgs_d()), 2))
                            + np.power((M_2 - muQ), 2))
                           * ((np.power(g_EW, 2)
                               * np.power((v_higgs_d() - v_higgs_u()), 2))
                              + np.power((M_2 + muQ), 2)))
    sigma_dd_chargino1 = -1 * (np.power(g_EW, 2) / (16 * np.power(np.pi, 2)))\
        * (1 - (chargino_num / chargino_den)) * logfunc(msC1)
    return sigma_dd_chargino1


def sigmadd_chargino2():
    """Return one-loop correction Sigma_d^d(chargino_2)."""
    chargino_num = np.power(M_2, 2) + np.power(muQ, 2)\
        - (np.power(g_EW, 2) * (np.power(v_higgs_u(), 2)
                                - np.power(v_higgs_d(), 2)))
    chargino_den = np.sqrt(((np.power(g_EW, 2)
                             * np.power((v_higgs_u() + v_higgs_d()), 2))
                            + np.power((M_2 - muQ), 2))
                           * ((np.power(g_EW, 2)
                               * np.power((v_higgs_d() - v_higgs_u()), 2))
                              + np.power((M_2 + muQ), 2)))
    sigma_dd_chargino2 = -1 * (np.power(g_EW, 2) / (16 * np.power(np.pi, 2)))\
        * (1 + (chargino_num / chargino_den)) * logfunc(msC2)
    return sigma_dd_chargino2


# Higgs bosons (sigmauu = sigmadd here): #

def sigmauu_h0():
    """Return one-loop correction Sigma_u,d^u,d(h_0) (lighter neutr. Higgs)."""
    mynum = ((np.power(g_EW, 2) + np.power(g_pr, 2)) * np.power(vHiggs, 2))\
        - (2 * ma_0sq() * (np.power(cossqb(), 2) - (6 * cossqb() * sinsqb())
                           + np.power(sinsqb(), 2)))
    myden = np.sqrt(np.power((ma_0sq() - np.power(mZ, 2)), 2)
                    + (4 * np.power(mZ, 2) * ma_0sq()
                       * 4 * cossqb() * sinsqb()))
    sigma_uu_h0 = (1 / (32 * np.power(np.pi, 2)))\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 4)\
        * (1 - (mynum / myden)) * logfunc(mh0)
    return sigma_uu_h0


def sigmauu_heavy_h0():
    """Return one-loop correction Sigma_u,d^u,d(H_0) (heavier neutr. Higgs)."""
    mynum = ((np.power(g_EW, 2) + np.power(g_pr, 2)) * np.power(vHiggs, 2))\
        - (2 * ma_0sq() * (np.power(cossqb(), 2) - (6 * cossqb() * sinsqb())
                           + np.power(sinsqb(), 2)))
    myden = np.sqrt(np.power((ma_0sq() - np.power(mZ, 2)), 2)
                    + (16 * np.power(mZ, 2) * ma_0sq() * cossqb() * sinsqb()))
    sigma_uu_heavy_h0 = (1/(128 * np.power(np.pi, 2)))\
        * ((np.power(g_EW, 2) + np.power(g_pr, 2))) * (1 + (mynum / myden))\
        * logfunc(mH0)
    return sigma_uu_heavy_h0


def sigmauu_h_pm():
    """Return one-loop correction Sigma_u,d^u,d(H_{+-})."""
    sigma_uu_h_pm = (np.power((g_EW / np.pi), 2) / (64)) * logfunc(mH_pm)
    return sigma_uu_h_pm


# Weak bosons (sigmauu = sigmadd here): #

def sigmauu_w_pm():
    """Return one-loop correction Sigma_u,d^u,d(W_{+-})."""
    sigma_uu_w_pm = (3 * np.power((g_EW / np.pi), 2) / (32))\
        * logfunc(np.sqrt(m_w_sq()))
    return sigma_uu_w_pm


def sigmauu_z0():
    """Return one-loop correction Sigma_u,d^u,d(Z_0)."""
    sigma_uu_z0 = (3 * (np.power(g_EW, 2) + np.power(g_pr, 2))
                   / (64 * np.power(np.pi, 2))) * logfunc(mZ)
    return sigma_uu_z0


# SM fermions (sigmadd_t = sigmauu_b = sigmauu_tau = 0): #

def sigmauu_top():
    """Return one-loop correction Sigma_u^u(top)."""
    mymt = y_t * v_higgs_u()
    sigma_uu_top = ((-1) * np.power((y_t / np.pi), 2) / (16)) * logfunc(mymt)
    return sigma_uu_top


def sigmadd_bottom():
    """Return one-loop correction Sigma_d^d(bottom)."""
    mymb = y_b * v_higgs_d()
    sigma_dd_bottom = (-1 * np.power((y_b / np.pi), 2) / (16)) * logfunc(mymb)
    return sigma_dd_bottom


def sigmadd_tau():
    """Return one-loop correction Sigma_d^d(tau)."""
    mymtau = y_tau * v_higgs_d()
    sigma_dd_tau = (-1 * np.power((y_tau / np.pi), 2) / (16)) * logfunc(mymtau)
    return sigma_dd_tau


# DEW computation: #

def dew_func(inp):
    """
    Compute individual one-loop DEW contributions from Sigma_u,d^u,d.

    Parameters
    ----------
    inp : One-loop correction or Higgs to be inputted into the DEW function.

    """
    mycontrib = np.abs((np.abs(inp) / np.sqrt(1 - (4 * sinsqb() * cossqb())))
                       - inp) / 2
    return mycontrib


[cmu, chu, chd] = [np.abs(np.power(muQ, 2)), dew_func(mHusq), dew_func(mHdsq)]
contribs = np.array([cmu, chu, chd, dew_func(sigmadd_stop1()),
                     dew_func(sigmadd_stop2()), dew_func(sigmadd_sbottom1()),
                     dew_func(sigmadd_sbottom2()), dew_func(sigmadd_stau1()),
                     dew_func(sigmadd_stau2()), dew_func(sigmadd_sup_l()),
                     dew_func(sigmadd_sup_r()), dew_func(sigmadd_sdown_l()),
                     dew_func(sigmadd_sdown_r()), dew_func(sigmadd_selec_l()),
                     dew_func(sigmadd_selec_r()), dew_func(sigmadd_sel_neut()),
                     dew_func(sigmadd_sstrange_l()),
                     dew_func(sigmadd_sstrange_r()),
                     dew_func(sigmadd_scharm_l()),
                     dew_func(sigmadd_scharm_r()), dew_func(sigmadd_smu_l()),
                     dew_func(sigmadd_smu_r()), dew_func(sigmadd_smu_sneut()),
                     dew_func(sigmadd_neutralino(msN1)),
                     dew_func(sigmadd_neutralino(msN2)),
                     dew_func(sigmadd_neutralino(msN3)),
                     dew_func(sigmadd_neutralino(msN4)),
                     dew_func(sigmadd_chargino1()),
                     dew_func(sigmadd_chargino2()),
                     dew_func(sigmauu_h0()), dew_func(sigmauu_heavy_h0()),
                     dew_func(sigmauu_h_pm()), dew_func(sigmauu_w_pm()),
                     dew_func(sigmauu_z0()), dew_func(sigmadd_bottom()),
                     dew_func(sigmadd_tau()), dew_func(sigmauu_stop1()),
                     dew_func(sigmauu_stop2()), dew_func(sigmauu_sbottom1()),
                     dew_func(sigmauu_sbottom2()), dew_func(sigmauu_stau1()),
                     dew_func(sigmauu_stau2()), dew_func(sigmauu_sup_l()),
                     dew_func(sigmauu_sup_r()), dew_func(sigmauu_sdown_l()),
                     dew_func(sigmauu_sdown_r()), dew_func(sigmauu_selec_l()),
                     dew_func(sigmauu_selec_r()), dew_func(sigmauu_sel_neut()),
                     dew_func(sigmauu_sstrange_l()),
                     dew_func(sigmauu_sstrange_r()),
                     dew_func(sigmauu_scharm_l()),
                     dew_func(sigmauu_scharm_r()), dew_func(sigmauu_smu_l()),
                     dew_func(sigmauu_smu_r()), dew_func(sigmauu_smu_sneut()),
                     dew_func(sigmauu_neutralino(msN1)),
                     dew_func(sigmauu_neutralino(msN2)),
                     dew_func(sigmauu_neutralino(msN3)),
                     dew_func(sigmauu_neutralino(msN4)),
                     dew_func(sigmauu_chargino1()),
                     dew_func(sigmauu_chargino2()), dew_func(sigmauu_h0()),
                     dew_func(sigmauu_heavy_h0()), dew_func(sigmauu_h_pm()),
                     dew_func(sigmauu_w_pm()), dew_func(sigmauu_z0()),
                     dew_func(sigmauu_top())]) / halfmzsq
label_sort_array = np.sort(np.array([(contribs[0], 'mu'), (contribs[1], 'H_u'),
                                     (contribs[2], 'H_d'),
                                     (contribs[3], 'Sigma_d^d(stop_1)'),
                                     (contribs[4], 'Sigma_d^d(stop_2)'),
                                     (contribs[5], 'Sigma_d^d(sbot_1)'),
                                     (contribs[6], 'Sigma_d^d(sbot_2)'),
                                     (contribs[7], 'Sigma_d^d(stau_1)'),
                                     (contribs[8], 'Sigma_d^d(stau_2)'),
                                     (contribs[9], 'Sigma_d^d(sup_L)'),
                                     (contribs[10], 'Sigma_d^d(sup_R)'),
                                     (contribs[11], 'Sigma_d^d(sdown_L)'),
                                     (contribs[12], 'Sigma_d^d(sdown_R)'),
                                     (contribs[13], 'Sigma_d^d(selec_L)'),
                                     (contribs[14], 'Sigma_d^d(selec_R)'),
                                     (contribs[15], 'Sigma_d^d(selec_neutr)'),
                                     (contribs[16], 'Sigma_d^d(sstrange_L)'),
                                     (contribs[17], 'Sigma_d^d(sstrange_R)'),
                                     (contribs[18], 'Sigma_d^d(scharm_L)'),
                                     (contribs[19], 'Sigma_d^d(scharm_R)'),
                                     (contribs[20], 'Sigma_d^d(smu_L)'),
                                     (contribs[21], 'Sigma_d^d(smu_R)'),
                                     (contribs[22], 'Sigma_d^d(smu_neutr)'),
                                     (contribs[23], 'Sigma_d^d(neutralino_1)'),
                                     (contribs[24], 'Sigma_d^d(neutralino_2)'),
                                     (contribs[25], 'Sigma_d^d(neutralino_3)'),
                                     (contribs[26], 'Sigma_d^d(neutralino_4)'),
                                     (contribs[27], 'Sigma_d^d(chargino_1)'),
                                     (contribs[28], 'Sigma_d^d(chargino_2)'),
                                     (contribs[29], 'Sigma_d^d(h_0)'),
                                     (contribs[30], 'Sigma_d^d(H_0)'),
                                     (contribs[31], 'Sigma_d^d(H_+-)'),
                                     (contribs[32], 'Sigma_d^d(W_+-)'),
                                     (contribs[33], 'Sigma_d^d(Z_0)'),
                                     (contribs[34], 'Sigma_d^d(bottom)'),
                                     (contribs[35], 'Sigma_d^d(tau)'),
                                     (contribs[36], 'Sigma_u^u(stop_1)'),
                                     (contribs[37], 'Sigma_u^u(stop_2)'),
                                     (contribs[38], 'Sigma_u^u(sbot_1)'),
                                     (contribs[39], 'Sigma_u^u(sbot_2)'),
                                     (contribs[40], 'Sigma_u^u(stau_1)'),
                                     (contribs[41], 'Sigma_u^u(stau_2)'),
                                     (contribs[42], 'Sigma_u^u(sup_L)'),
                                     (contribs[43], 'Sigma_u^u(sup_R)'),
                                     (contribs[44], 'Sigma_u^u(sdown_L)'),
                                     (contribs[45], 'Sigma_u^u(sdown_R)'),
                                     (contribs[46], 'Sigma_u^u(selec_L)'),
                                     (contribs[47], 'Sigma_u^u(selec_R)'),
                                     (contribs[48], 'Sigma_u^u(selec_neutr)'),
                                     (contribs[49], 'Sigma_u^u(sstrange_L)'),
                                     (contribs[50], 'Sigma_u^u(sstrange_R)'),
                                     (contribs[51], 'Sigma_u^u(scharm_L)'),
                                     (contribs[52], 'Sigma_u^u(scharm_R)'),
                                     (contribs[53], 'Sigma_u^u(smu_L)'),
                                     (contribs[54], 'Sigma_u^u(smu_R)'),
                                     (contribs[55], 'Sigma_u^u(smu_neutr)'),
                                     (contribs[56], 'Sigma_u^u(neutralino_1)'),
                                     (contribs[57], 'Sigma_u^u(neutralino_2)'),
                                     (contribs[58], 'Sigma_u^u(neutralino_3)'),
                                     (contribs[59], 'Sigma_u^u(neutralino_4)'),
                                     (contribs[60], 'Sigma_u^u(chargino_1)'),
                                     (contribs[61], 'Sigma_u^u(chargino_2)'),
                                     (contribs[62], 'Sigma_u^u(h_0)'),
                                     (contribs[63], 'Sigma_u^u(H_0)'),
                                     (contribs[64], 'Sigma_u^u(H_+-)'),
                                     (contribs[65], 'Sigma_u^u(W_+-)'),
                                     (contribs[66], 'Sigma_u^u(Z_0)'),
                                     (contribs[67], 'Sigma_u^u(top)')],
                                    dtype=[('Contrib', float),
                                           ('label', 'U30')]), order='Contrib')
reverse_sort_array = label_sort_array[::-1]
print('\nGiven the submitted SLHA file, your value for the electroweak'
      + ' naturalness measure, Delta_EW, is: ' + str(np.amax(contribs)))
print('\nThe top ten contributions to Delta_EW are as follows (decr. order): ')
print('')
print(str(reverse_sort_array[0]) + ',\n' + str(reverse_sort_array[1]) + ',')
print(str(reverse_sort_array[2]) + ',\n' + str(reverse_sort_array[3]) + ',')
print(str(reverse_sort_array[4]) + ',\n' + str(reverse_sort_array[5]) + ',')
print(str(reverse_sort_array[6]) + ',\n' + str(reverse_sort_array[7]) + ',')
print(str(reverse_sort_array[8]) + ',\n' + str(reverse_sort_array[9]))
