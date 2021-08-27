# -*- coding: utf-8 -*-
"""
Compute naturalness measure Delta_EW and contributions to DEW.

Author: Dakotah Martinez
"""

import numpy as np
import pyslha
import time
from pathlib import Path
import numba as nb


# Mass relations: #

@nb.njit(fastmath=True)
def m_w_sq():
    """Return W boson squared mass."""
    my_mw_sq = (np.power(g_EW, 2) / 2) * np.power(vHiggs, 2)
    return my_mw_sq


@nb.njit(fastmath=True)
def mz_q_sq():
    """Return m_Z(Q)^2."""
    mzqsq = np.power(vHiggs, 2) * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)
    return mzqsq


@nb.njit(fastmath=True)
def ma_0sq():
    """Return A_0 squared mass."""
    my_ma0_sq = 2 * np.power(np.abs(muQ), 2) + mHusq + mHdsq
    return my_ma0_sq


# Fundamental equations: #

@nb.njit(fastmath=True)
def logfunc(mass):
    """
    Return F = m^2 * (ln(m^2 / Q^2) - 1).

    Parameters
    ----------
    mass : Input mass.

    """
    myf = np.power(mass, 2) * (np.log((np.power(mass, 2)) / (Q_renorm_sq)) - 1)
    return myf


@nb.njit(fastmath=True)
def sinsqb():
    """Return sin^2(beta)."""
    mysinsqb = np.power(np.sin(beta), 2)
    return mysinsqb


@nb.njit(fastmath=True)
def cossqb():
    """Return cos^2(beta)."""
    mycossqb = np.power(np.cos(beta), 2)
    return mycossqb


@nb.njit(fastmath=True)
def v_higgs_u():
    """Return up-type Higgs VEV."""
    myvu = vHiggs * np.sin(beta)
    return myvu


@nb.njit(fastmath=True)
def v_higgs_d():
    """Return down-type Higgs VEV."""
    myvd = vHiggs * np.cos(beta)
    return myvd


@nb.njit(fastmath=True)
def tan_theta_w():
    """Return tan(theta_W), the Weinberg angle."""
    mytanthetaw = g_pr / g_EW
    return mytanthetaw


@nb.njit(fastmath=True)
def sinsq_theta_w():
    """Return sin^2(theta_W), the Weinberg angle."""
    thetaw = np.arctan(tan_theta_w())
    mysinsqthetaw = np.power(np.sin(thetaw), 2)
    return mysinsqthetaw


@nb.njit(fastmath=True)
def cos2b():
    """Return cos(2*beta)."""
    mycos2b = cossqb() - sinsqb()
    return mycos2b


@nb.njit(fastmath=True)
def gz_sq():
    """Return g_Z^2 = (g^2 + g'^2) / 8."""
    mygzsq = (np.power(g_EW, 2) + np.power(g_pr, 2)) / 8
    return mygzsq


# Stop squarks: #

@nb.njit(fastmath=True)
def sigmauu_stop1():
    """Return one-loop correction Sigma_u^u(stop_1)."""
    delta_stop = ((1 / 2) - (4 / 3) * sinsq_theta_w()) * 2\
        * (((np.power(mtL, 2) - np.power(mtR, 2)) / 2)
           + (mz_q_sq()) * cos2b() * ((1 / 4) - (2 / 3) * sinsq_theta_w()))
    stop_num = np.power(a_t, 2) - (2 * gz_sq() * delta_stop)
    sigmauu_stop_1 = (3 / (16 * (np.power(np.pi, 2)))) * logfunc(m_stop_1)\
        * (np.power(y_t, 2) - gz_sq()
           - (stop_num / (np.power(m_stop_2, 2) - np.power(m_stop_1, 2))))
    return sigmauu_stop_1


@nb.njit(fastmath=True)
def sigmadd_stop1():
    """Return one-loop correction Sigma_d^d(stop_1)."""
    delta_stop = ((1 / 2) - (4 / 3) * sinsq_theta_w()) * 2\
        * (((np.power(mtL, 2) - np.power(mtR, 2)) / 2)
           + (mz_q_sq()) * cos2b() * ((1 / 4) - (2 / 3) * sinsq_theta_w()))
    stop_num = np.power(y_t, 2) * np.power(muQ, 2)\
        + (2 * gz_sq() * delta_stop)
    sigmadd_stop_1 = (3 / (16 * np.power(np.pi, 2))) * logfunc(m_stop_1)\
        * (gz_sq() - (stop_num
                      / (np.power(m_stop_2, 2) - np.power(m_stop_1, 2))))
    return sigmadd_stop_1


@nb.njit(fastmath=True)
def sigmauu_stop2():
    """Return one-loop correction Sigma_u^u(stop_2)."""
    delta_stop = ((1 / 2) - (4 / 3) * sinsq_theta_w()) * 2\
        * (((np.power(mtL, 2) - np.power(mtR, 2)) / 2)
           + (mz_q_sq()) * cos2b() * ((1 / 4) - (2 / 3) * sinsq_theta_w()))
    stop_num = np.power(a_t, 2) - (2 * gz_sq() * delta_stop)
    sigmauu_stop_2 = (3 / (16 * (np.power(np.pi, 2)))) * logfunc(m_stop_2)\
        * (np.power(y_t, 2) - gz_sq()
           + (stop_num
              / (np.power(m_stop_2, 2) - np.power(m_stop_1, 2))))
    return sigmauu_stop_2


@nb.njit(fastmath=True)
def sigmadd_stop2():
    """Return one-loop correction Sigma_d^d(stop_2)."""
    delta_stop = ((1 / 2) - (4 / 3) * sinsq_theta_w()) * 2\
        * (((np.power(mtL, 2) - np.power(mtR, 2)) / 2)
           + (mz_q_sq()) * cos2b() * ((1 / 4) - (2 / 3) * sinsq_theta_w()))
    stop_num = np.power(y_t, 2) * np.power(muQ, 2)\
        + (2 * gz_sq() * delta_stop)
    sigmadd_stop_2 = (3 / (16 * np.power(np.pi, 2))) * logfunc(m_stop_2)\
        * (gz_sq() + (stop_num
                      / (np.power(m_stop_2, 2) - np.power(m_stop_1, 2))))
    return sigmadd_stop_2


# Sbottom squarks: #

@nb.njit(fastmath=True)
def sigmauu_sbottom1():
    """Return one-loop correction Sigma_u^u(sbottom_1)."""
    delta_sbot = ((1 / 2) - (2 / 3) * sinsq_theta_w()) * 2\
        * (((np.power(mbL, 2) - np.power(mbR, 2)) / 2)
           - (mz_q_sq()) * cos2b() * ((1 / 4) - (1 / 3) * sinsq_theta_w()))
    sbot_num = np.power(a_b, 2) - (2 * gz_sq() * delta_sbot)
    sigmauu_sbot = (3 / (16 * np.power(np.pi, 2))) * logfunc(m_sbot_1)\
        * (np.power(y_b, 2) - gz_sq()
           - (sbot_num / (np.power(m_sbot_2, 2) - np.power(m_sbot_1, 2))))
    return sigmauu_sbot


@nb.njit(fastmath=True)
def sigmauu_sbottom2():
    """Return one-loop correction Sigma_u^u(sbottom_2)."""
    delta_sbot = ((1 / 2) - (2 / 3) * sinsq_theta_w()) * 2\
        * (((np.power(mbL, 2) - np.power(mbR, 2)) / 2)
           - (mz_q_sq()) * cos2b() * ((1 / 4) - (1 / 3) * sinsq_theta_w()))
    sbot_num = np.power(a_b, 2) - (2 * gz_sq() * delta_sbot)
    sigmauu_sbot = (3 / (16 * np.power(np.pi, 2))) * logfunc(m_sbot_2)\
        * (np.power(y_b, 2) - gz_sq()
           + (sbot_num / (np.power(m_sbot_2, 2) - np.power(m_sbot_1, 2))))
    return sigmauu_sbot


@nb.njit(fastmath=True)
def sigmadd_sbottom1():
    """Return one-loop correction Sigma_d^d(sbottom_1)."""
    delta_sbot = ((1 / 2) - (2 / 3) * sinsq_theta_w()) * 2\
        * (((np.power(mbL, 2) - np.power(mbR, 2)) / 2)
           - (mz_q_sq()) * cos2b() * ((1 / 4) - (1 / 3) * sinsq_theta_w()))
    sbot_num = np.power(y_b, 2) * np.power(muQ, 2)\
        + (2 * gz_sq() * delta_sbot)
    sigmadd_sbot = (3 / (16 * np.power(np.pi, 2))) * logfunc(m_sbot_1)\
        * (gz_sq()
           - (sbot_num / (np.power(m_sbot_2, 2) - np.power(m_sbot_1, 2))))
    return sigmadd_sbot


@nb.njit(fastmath=True)
def sigmadd_sbottom2():
    """Return one-loop correction Sigma_d^d(sbottom_2)."""
    delta_sbot = ((1 / 2) - (2 / 3) * sinsq_theta_w()) * 2\
        * (((np.power(mbL, 2) - np.power(mbR, 2)) / 2)
           - (mz_q_sq()) * cos2b() * ((1 / 4) - (1 / 3) * sinsq_theta_w()))
    sbot_num = np.power(y_b, 2) * np.power(muQ, 2)\
        + (2 * gz_sq() * delta_sbot)
    sigmadd_sbot = (3 / (16 * np.power(np.pi, 2))) * logfunc(m_sbot_2)\
        * (gz_sq()
           + (sbot_num / (np.power(m_sbot_2, 2) - np.power(m_sbot_1, 2))))
    return sigmadd_sbot


# Stau sleptons: #

@nb.njit(fastmath=True)
def sigmauu_stau1():
    """Return one-loop correction Sigma_u^u(stau_1)."""
    delta_stau = ((1 / 2) - 2 * sinsq_theta_w()) * 2\
        * (((np.power(mtauL, 2) - np.power(mtauR, 2)) / 2)
           - (mz_q_sq()) * cos2b() * ((1 / 4) - sinsq_theta_w()))
    stau_num = np.power(a_tau, 2) - (2 * gz_sq() * delta_stau)
    sigmauu_stau = (1 / (16 * np.power(np.pi, 2))) * logfunc(m_stau_1)\
        * (np.power(y_tau, 2) - gz_sq()
           - (stau_num / (np.power(m_stau_2, 2) - np.power(m_stau_1, 2))))
    return sigmauu_stau


@nb.njit(fastmath=True)
def sigmauu_stau2():
    """Return one-loop correction Sigma_u^u(stau_2)."""
    delta_stau = ((1 / 2) - 2 * sinsq_theta_w()) * 2\
        * (((np.power(mtauL, 2) - np.power(mtauR, 2)) / 2)
           - (mz_q_sq()) * cos2b() * ((1 / 4) - sinsq_theta_w()))
    stau_num = np.power(a_tau, 2) - (2 * gz_sq() * delta_stau)
    sigmauu_stau = (1 / (16 * np.power(np.pi, 2))) * logfunc(m_stau_2)\
        * (np.power(y_tau, 2) - gz_sq()
           + (stau_num / (np.power(m_stau_2, 2) - np.power(m_stau_1, 2)))
           )
    return sigmauu_stau


@nb.njit(fastmath=True)
def sigmadd_stau1():
    """Return one-loop correction Sigma_d^d(stau_1)."""
    delta_stau = ((1 / 2) - 2 * sinsq_theta_w()) * 2\
        * (((np.power(mtauL, 2) - np.power(mtauR, 2)) / 2)
           - (mz_q_sq()) * cos2b() * ((1 / 4) - sinsq_theta_w()))
    stau_num = np.power(y_tau, 2) * np.power(muQ, 2)\
        + (2 * gz_sq() * delta_stau)
    sigmauu_stau = (1 / (16 * np.power(np.pi, 2))) * logfunc(m_stau_1)\
        * (gz_sq()
           - (stau_num / (np.power(m_stau_2, 2) - np.power(m_stau_1, 2))))
    return sigmauu_stau


@nb.njit(fastmath=True)
def sigmadd_stau2():
    """Return one-loop correction Sigma_d^d(stau_2)."""
    delta_stau = ((1 / 2) - 2 * sinsq_theta_w()) * 2\
        * (((np.power(mtauL, 2) - np.power(mtauR, 2)) / 2)
           - (mz_q_sq()) * cos2b() * ((1 / 4) - sinsq_theta_w()))
    stau_num = np.power(y_tau, 2) * np.power(muQ, 2)\
        + (2 * gz_sq() * delta_stau)
    sigmauu_stau = (1 / (16 * np.power(np.pi, 2))) * logfunc(m_stau_2)\
        * (gz_sq()
           + (stau_num / (np.power(m_stau_2, 2) - np.power(m_stau_1, 2)))
           )
    return sigmauu_stau


# Sfermions, 1st gen: #

@nb.njit(fastmath=True)
def sigmauu_sup_l():
    """Return one-loop correction Sigma_u^u(sup_L)."""
    sigmauusup_l = ((-3) / (4 * np.power(np.pi, 2)))\
        * ((1 / 2) - (2 / 3) * sinsq_theta_w()) * gz_sq()\
        * logfunc(msupL)
    return sigmauusup_l


@nb.njit(fastmath=True)
def sigmauu_sup_r():
    """Return one-loop correction Sigma_u^u(sup_R)."""
    sigmauusup_r = ((-3) / (4 * np.power(np.pi, 2)))\
        * ((2 / 3) * sinsq_theta_w()) * gz_sq() * logfunc(msupR)
    return sigmauusup_r


@nb.njit(fastmath=True)
def sigmauu_sdown_l():
    """Return one-loop correction Sigma_u^u(sdown_L)."""
    sigmauusdown_l = ((-3) / (4 * np.power(np.pi, 2))) * logfunc(msdownL)\
        * (((-1) / 2) + (1 / 3) * sinsq_theta_w()) * gz_sq()
    return sigmauusdown_l


@nb.njit(fastmath=True)
def sigmauu_sdown_r():
    """Return one-loop correction Sigma_u^u(sdown_R)."""
    sigmauusdown_r = ((-3) / (4 * np.power(np.pi, 2))) * logfunc(msdownR)\
        * (((-1) / 3) * sinsq_theta_w()) * gz_sq()
    return sigmauusdown_r


@nb.njit(fastmath=True)
def sigmauu_selec_l():
    """Return one-loop correction Sigma_u^u(selectron_L)."""
    sigmauuselec_l = ((-1) / (4 * np.power(np.pi, 2))) * logfunc(mselecL)\
        * (((-1) / 2) + sinsq_theta_w()) * gz_sq()
    return sigmauuselec_l


@nb.njit(fastmath=True)
def sigmauu_selec_r():
    """Return one-loop correction Sigma_u^u(selectron_R)."""
    sigmauuselec_r = ((-1) / (4 * np.power(np.pi, 2))) * logfunc(mselecR)\
        * ((-1) * sinsq_theta_w()) * gz_sq()
    return sigmauuselec_r


@nb.njit(fastmath=True)
def sigmauu_sel_neut():
    """Return one-loop correction Sigma_u^u(selectron neutrino)."""
    sigmauuselec_sneut = ((-1) / (4 * np.power(np.pi, 2))) * (1 / 2)\
        * logfunc(mselecneut) * gz_sq()
    return sigmauuselec_sneut


@nb.njit(fastmath=True)
def sigmadd_sup_l():
    """Return one-loop correction Sigma_d^d(sup_L)."""
    sigmaddsup_l = (3 / (4 * np.power(np.pi, 2)))\
        * ((1 / 2) - (2 / 3) * sinsq_theta_w()) * gz_sq()\
        * logfunc(msupL)
    return sigmaddsup_l


@nb.njit(fastmath=True)
def sigmadd_sup_r():
    """Return one-loop correction Sigma_d^d(sup_R)."""
    sigmaddsup_r = (3 / (4 * np.power(np.pi, 2)))\
        * ((2 / 3) * sinsq_theta_w()) * gz_sq() * logfunc(msupR)
    return sigmaddsup_r


@nb.njit(fastmath=True)
def sigmadd_sdown_l():
    """Return one-loop correction Sigma_d^d(sdown_L)."""
    sigmaddsdown_l = (3 / (4 * np.power(np.pi, 2))) * logfunc(msdownL)\
        * (((-1) / 2) + (1 / 3) * sinsq_theta_w()) * gz_sq()
    return sigmaddsdown_l


@nb.njit(fastmath=True)
def sigmadd_sdown_r():
    """Return one-loop correction Sigma_d^d(sdown_R)."""
    sigmaddsdown_r = (3 / (4 * np.power(np.pi, 2))) * logfunc(msdownR)\
        * (((-1) / 3) * sinsq_theta_w()) * gz_sq()
    return sigmaddsdown_r


@nb.njit(fastmath=True)
def sigmadd_selec_l():
    """Return one-loop correction Sigma_d^d(selectron_L)."""
    sigmaddselec_l = (1 / (4 * np.power(np.pi, 2))) * logfunc(mselecL)\
        * (((-1) / 2) + sinsq_theta_w()) * gz_sq()
    return sigmaddselec_l


@nb.njit(fastmath=True)
def sigmadd_selec_r():
    """Return one-loop correction Sigma_d^d(selectron_R)."""
    sigmaddselec_r = (1 / (4 * np.power(np.pi, 2))) * logfunc(mselecR)\
        * ((-1) * sinsq_theta_w()) * gz_sq()
    return sigmaddselec_r


@nb.njit(fastmath=True)
def sigmadd_sel_neut():
    """Return one-loop correction Sigma_d^d(selectron neutrino)."""
    sigmaddselec_sneut = (1 / (4 * np.power(np.pi, 2))) * (1 / 2)\
        * logfunc(mselecneut) * gz_sq()
    return sigmaddselec_sneut


# Sfermions, 2nd gen: #

@nb.njit(fastmath=True)
def sigmauu_sstrange_l():
    """Return one-loop correction Sigma_u^u(sstrange_L)."""
    sigmauusstrange_l = ((-3) / (4 * np.power(np.pi, 2))) * gz_sq()\
        * (((-1) / 2) + (1 / 3) * sinsq_theta_w()) * logfunc(msstrangeL)
    return sigmauusstrange_l


@nb.njit(fastmath=True)
def sigmauu_sstrange_r():
    """Return one-loop correction Sigma_u^u(sstrange_R)."""
    sigmauusstrange_r = ((-3) / (4 * np.power(np.pi, 2))) * gz_sq()\
        * (((-1) / 3) * sinsq_theta_w()) * logfunc(msstrangeR)
    return sigmauusstrange_r


@nb.njit(fastmath=True)
def sigmauu_scharm_l():
    """Return one-loop correction Sigma_u^u(scharm_L)."""
    sigmauuscharm_l = ((-3) / (4 * np.power(np.pi, 2))) * gz_sq()\
        * ((1 / 2) - (2 / 3) * sinsq_theta_w()) * logfunc(mscharmL)
    return sigmauuscharm_l


@nb.njit(fastmath=True)
def sigmauu_scharm_r():
    """Return one-loop correction Sigma_u^u(scharm_R)."""
    sigmauuscharm_r = ((-3) / (4 * np.power(np.pi, 2))) * gz_sq()\
        * ((2 / 3) * sinsq_theta_w()) * logfunc(mscharmR)
    return sigmauuscharm_r


@nb.njit(fastmath=True)
def sigmauu_smu_l():
    """Return one-loop correction Sigma_u^u(smu_L)."""
    sigmauusmu_l = ((-1) / (4 * np.power(np.pi, 2))) * gz_sq()\
        * (((-1) / 2) + sinsq_theta_w()) * logfunc(msmuL)
    return sigmauusmu_l


@nb.njit(fastmath=True)
def sigmauu_smu_r():
    """Return one-loop correction Sigma_u^u(smu_R)."""
    sigmauusmu_r = ((-1) / (4 * np.power(np.pi, 2))) * gz_sq()\
        * ((-1) * sinsq_theta_w()) * logfunc(msmuR)
    return sigmauusmu_r


@nb.njit(fastmath=True)
def sigmauu_smu_sneut():
    """Return one-loop correction Sigma_u^u(smuon neutrino)."""
    sigmauusmu_sneut = ((-1) / (4 * np.power(np.pi, 2))) * gz_sq()\
        * (1 / 2) * logfunc(msmuneut)
    return sigmauusmu_sneut


@nb.njit(fastmath=True)
def sigmadd_sstrange_l():
    """Return one-loop correction Sigma_d^d(sstrange_L)."""
    sigmaddsstrange_l = (3 / (4 * np.power(np.pi, 2))) * gz_sq()\
        * (((-1) / 2) + (1 / 3) * sinsq_theta_w()) * logfunc(msstrangeL)
    return sigmaddsstrange_l


@nb.njit(fastmath=True)
def sigmadd_sstrange_r():
    """Return one-loop correction Sigma_d^d(sstrange_R)."""
    sigmaddsstrange_r = (3 / (4 * np.power(np.pi, 2))) * gz_sq()\
        * (((-1) / 3) * sinsq_theta_w()) * logfunc(msstrangeR)
    return sigmaddsstrange_r


@nb.njit(fastmath=True)
def sigmadd_scharm_l():
    """Return one-loop correction Sigma_d^d(scharm_L)."""
    sigmaddscharm_l = (3 / (4 * np.power(np.pi, 2))) * gz_sq()\
        * ((1 / 2) - (2 / 3) * sinsq_theta_w()) * logfunc(mscharmL)
    return sigmaddscharm_l


@nb.njit(fastmath=True)
def sigmadd_scharm_r():
    """Return one-loop correction Sigma_d^d(scharm_R)."""
    sigmaddscharm_r = (3 / (4 * np.power(np.pi, 2))) * gz_sq()\
        * ((2 / 3) * sinsq_theta_w()) * logfunc(mscharmR)
    return sigmaddscharm_r


@nb.njit(fastmath=True)
def sigmadd_smu_l():
    """Return one-loop correction Sigma_d^d(smu_L)."""
    sigmaddsmu_l = (1 / (4 * np.power(np.pi, 2))) * gz_sq()\
        * (((-1) / 2) + sinsq_theta_w()) * logfunc(msmuL)
    return sigmaddsmu_l


@nb.njit(fastmath=True)
def sigmadd_smu_r():
    """Return one-loop correction Sigma_d^d(smu_R)."""
    sigmaddsmu_r = (1 / (4 * np.power(np.pi, 2))) * gz_sq()\
        * ((-1) * sinsq_theta_w()) * logfunc(msmuR)
    return sigmaddsmu_r


@nb.njit(fastmath=True)
def sigmadd_smu_sneut():
    """Return one-loop correction Sigma_d^d(smuon neutrino)."""
    sigmaddsmu_sneut = (1 / (4 * np.power(np.pi, 2))) * (1 / 2)\
        * gz_sq() * logfunc(msmuneut)
    return sigmaddsmu_sneut


# Neutralinos: #
# Use method by Ibrahim and Nath for eigenvalue derivatives.

@nb.njit(fastmath=True)
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


@nb.njit(fastmath=True)
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


@nb.njit(fastmath=True)
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


@nb.njit(fastmath=True)
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


@nb.njit(fastmath=True)
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

@nb.njit(fastmath=True)
def sigmauu_chargino1():
    """Return one-loop correction Sigma_u^u(chargino_1)."""
    chargino_num = ((-2) * m_w_sq() * cos2b()) + np.power(M_2, 2)\
        + np.power(muQ, 2)
    chargino_den = np.power(msC2, 2) - np.power(msC1, 2)
    sigma_uu_chargino1 = -1 * (np.power(g_EW, 2) / (16 * np.power(np.pi, 2)))\
        * (1 - (chargino_num / chargino_den)) * logfunc(msC1)
    return sigma_uu_chargino1


@nb.njit(fastmath=True)
def sigmauu_chargino2():
    """Return one-loop correction Sigma_u^u(chargino_2)."""
    chargino_num = ((-2) * m_w_sq() * cos2b()) + np.power(M_2, 2)\
        + np.power(muQ, 2)
    chargino_den = np.power(msC2, 2) - np.power(msC1, 2)
    sigma_uu_chargino2 = -1 * (np.power(g_EW, 2) / (16 * np.power(np.pi, 2)))\
        * (1 + (chargino_num / chargino_den)) * logfunc(msC2)
    return sigma_uu_chargino2


@nb.njit(fastmath=True)
def sigmadd_chargino1():
    """Return one-loop correction Sigma_d^d(chargino_1)."""
    chargino_num = (2 * m_w_sq() * cos2b()) + np.power(M_2, 2)\
        + np.power(muQ, 2)
    chargino_den = np.power(msC2, 2) - np.power(msC1, 2)
    sigma_dd_chargino1 = -1 * (np.power(g_EW, 2) / (16 * np.power(np.pi, 2)))\
        * (1 - (chargino_num / chargino_den)) * logfunc(msC1)
    return sigma_dd_chargino1


@nb.njit(fastmath=True)
def sigmadd_chargino2():
    """Return one-loop correction Sigma_d^d(chargino_2)."""
    chargino_num = (2 * m_w_sq() * cos2b()) + np.power(M_2, 2)\
        + np.power(muQ, 2)
    chargino_den = np.power(msC2, 2) - np.power(msC1, 2)
    sigma_dd_chargino2 = -1 * (np.power(g_EW, 2) / (16 * np.power(np.pi, 2)))\
        * (1 + (chargino_num / chargino_den)) * logfunc(msC2)
    return sigma_dd_chargino2


# Higgs bosons (sigmauu = sigmadd here): #

@nb.njit(fastmath=True)
def sigmauu_h0():
    """Return one-loop correction Sigma_u^u(h_0) (lighter neutral Higgs)."""
    mynum = mz_q_sq() + (mA0sq * (1 + (4 * cos2b())
                                  + (2 * np.power(cos2b(), 2))))
    myden = np.power(mH0, 2) - np.power(mh0, 2)
    sigma_uu_h0 = (gz_sq() / (16 * np.power(np.pi, 2)))\
        * (1 - (mynum / myden)) * logfunc(mh0)
    return sigma_uu_h0


@nb.njit(fastmath=True)
def sigmadd_h0():
    """Return one-loop correction Sigma_d^d(h_0) (lighter neutral Higgs)."""
    mynum = mz_q_sq() + (mA0sq * (1 - (4 * cos2b())
                                  + (2 * np.power(cos2b(), 2))))
    myden = np.power(mH0, 2) - np.power(mh0, 2)
    sigma_dd_h0 = (gz_sq() / (16 * np.power(np.pi, 2)))\
        * (1 - (mynum / myden)) * logfunc(mh0)
    return sigma_dd_h0


@nb.njit(fastmath=True)
def sigmauu_heavy_h0():
    """Return one-loop correction Sigma_u^u(H_0) (heavier neutr. Higgs)."""
    mynum = mz_q_sq() + (mA0sq * (1 + (4 * cos2b())
                                  + (2 * np.power(cos2b(), 2))))
    myden = np.power(mH0, 2) - np.power(mh0, 2)
    sigma_uu_heavy_h0 = (gz_sq() / (16 * np.power(np.pi, 2)))\
        * (1 + (mynum / myden)) * logfunc(mH0)
    return sigma_uu_heavy_h0


@nb.njit(fastmath=True)
def sigmadd_heavy_h0():
    """Return one-loop correction Sigma_d^d(H_0) (heavier neutr. Higgs)."""
    mynum = mz_q_sq() + (mA0sq * (1 - (4 * cos2b())
                                  + (2 * np.power(cos2b(), 2))))
    myden = np.power(mH0, 2) - np.power(mh0, 2)
    sigma_dd_heavy_h0 = (gz_sq() / (16 * np.power(np.pi, 2)))\
        * (1 + (mynum / myden)) * logfunc(mH0)
    return sigma_dd_heavy_h0


@nb.njit(fastmath=True)
def sigmauu_h_pm():
    """Return one-loop correction Sigma_u,d^u,d(H_{+-})."""
    sigma_uu_h_pm = (np.power((g_EW / np.pi), 2) / (32)) * logfunc(mH_pm)
    return sigma_uu_h_pm


# Weak bosons (sigmauu = sigmadd here): #

@nb.njit(fastmath=True)
def sigmauu_w_pm():
    """Return one-loop correction Sigma_u,d^u,d(W_{+-})."""
    sigma_uu_w_pm = (3 * np.power((g_EW / np.pi), 2) / (32))\
        * logfunc(np.sqrt(m_w_sq()))
    return sigma_uu_w_pm


@nb.njit(fastmath=True)
def sigmauu_z0():
    """Return one-loop correction Sigma_u,d^u,d(Z_0)."""
    sigma_uu_z0 = (3 * (np.power(g_EW, 2) + np.power(g_pr, 2))
                   / (64 * np.power(np.pi, 2))) * logfunc(np.sqrt(mz_q_sq()))
    return sigma_uu_z0


# SM fermions (sigmadd_t = sigmauu_b = sigmauu_tau = 0): #

@nb.njit(fastmath=True)
def sigmauu_top():
    """Return one-loop correction Sigma_u^u(top)."""
    mymt = y_t * v_higgs_u()
    sigma_uu_top = ((-1) * np.power((y_t / np.pi), 2) / (16)) * logfunc(mymt)
    return sigma_uu_top


@nb.njit(fastmath=True)
def sigmadd_bottom():
    """Return one-loop correction Sigma_d^d(bottom)."""
    mymb = y_b * v_higgs_d()
    sigma_dd_bottom = (-1 * np.power((y_b / np.pi), 2) / (16)) * logfunc(mymb)
    return sigma_dd_bottom


@nb.njit(fastmath=True)
def sigmadd_tau():
    """Return one-loop correction Sigma_d^d(tau)."""
    mymtau = y_tau * v_higgs_d()
    sigma_dd_tau = (-1 * np.power((y_tau / np.pi), 2) / (16)) * logfunc(mymtau)
    return sigma_dd_tau


# DEW contribution computation: #

@nb.njit(fastmath=True)
def dew_funcu(inp):
    """
    Compute individual one-loop DEW contributions from Sigma_u^u.

    Parameters
    ----------
    inp : One-loop correction or Higgs to be inputted into the DEW function.

    """
    mycontribuu = np.abs(((-1) * inp * (np.power(tanb, 2)))
                         / ((np.power(tanb, 2)) - 1))
    return mycontribuu


@nb.njit(fastmath=True)
def dew_funcd(inp):
    """
    Compute individual one-loop DEW contributions from Sigma_d^d.

    Parameters
    ----------
    inp : One-loop correction or Higgs to be inputted into the DEW function.

    """
    mycontribdd = np.abs((inp)
                         / ((np.power(tanb, 2)) - 1))
    return mycontribdd


if __name__ == "__main__":
    userContinue = True
    while userContinue:
        # SLHA input and definition of variables from SLHA file: #

        fileCheck = True
        while fileCheck:
            direc = input('Enter the full directory for your SLHA file: ')
            fileName = Path(direc)
            if fileName.exists():
                d = pyslha.read(direc)
                fileCheck = False
            else:
                print("The input file cannot be found.\n")
                print("Please try checking your spelling and try again.\n")
                fileCheck = True
        [vHiggs, muQ] = [d.blocks['HMIX'][3], d.blocks['HMIX'][1]]
        [tanb, y_t] = [d.blocks['HMIX'][2], d.blocks['YU'][3, 3]]
        beta = np.arctan(tanb)
        [y_b, y_tau] = [d.blocks['YD'][3, 3], d.blocks['YE'][3, 3]]
        [g_pr, g_EW] = [d.blocks['GAUGE'][2], d.blocks['GAUGE'][1]]
        [m_stop_1, m_stop_2] = [d.blocks['MASS'][1000006],
                                d.blocks['MASS'][2000006]]
        [m_sbot_1, m_sbot_2] = [d.blocks['MASS'][1000005],
                                d.blocks['MASS'][2000005]]
        [m_stau_1, m_stau_2] = [d.blocks['MASS'][1000015],
                                d.blocks['MASS'][2000015]]
        [mtL, mtR] = [d.blocks['MSOFT'][43], d.blocks['MSOFT'][46]]
        [mbL, mbR] = [d.blocks['MSOFT'][43], d.blocks['MSOFT'][49]]
        [mtauL, mtauR] = [d.blocks['MSOFT'][33], d.blocks['MSOFT'][36]]
        [msupL, msupR] = [d.blocks['MSOFT'][41], d.blocks['MSOFT'][44]]
        [msdownL, msdownR] = [d.blocks['MSOFT'][41], d.blocks['MSOFT'][47]]
        [mselecL, mselecR] = [d.blocks['MSOFT'][31], d.blocks['MSOFT'][34]]
        [mselecneut, msmuneut] = [d.blocks['MASS'][1000012],
                                  d.blocks['MASS'][1000014]]
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
        mA0sq = d.blocks['HMIX'][4]
        a_tau = d.blocks['AE'][3, 3] * y_tau
        [Q_renorm_sq, halfmzsq] = [m_stop_1 * m_stop_2, np.power(mZ, 2) / 2]
        [cmu, chu, chd] = [np.abs(np.power(muQ, 2)), dew_funcu(mHusq),
                           dew_funcd(mHdsq)]
        contribs = np.array([cmu, chu, chd, dew_funcd(sigmadd_stop1()),
                             dew_funcd(sigmadd_stop2()),
                             dew_funcd(sigmadd_sbottom1()),
                             dew_funcd(sigmadd_sbottom2()),
                             dew_funcd(sigmadd_stau1()),
                             dew_funcd(sigmadd_stau2()),
                             dew_funcd(sigmadd_sup_l() + sigmadd_sup_r()
                                       + sigmadd_sdown_l() + sigmadd_sdown_r()
                                       + sigmadd_selec_l() + sigmadd_selec_r()
                                       + sigmadd_sel_neut()),
                             dew_funcd(sigmadd_sstrange_l() + sigmadd_sstrange_r()
                                       + sigmadd_scharm_l() + sigmadd_scharm_r()
                                       + sigmadd_smu_l() + sigmadd_smu_r()
                                       + sigmadd_smu_sneut()),
                             dew_funcd(sigmadd_neutralino(msN1)),
                             dew_funcd(sigmadd_neutralino(msN2)),
                             dew_funcd(sigmadd_neutralino(msN3)),
                             dew_funcd(sigmadd_neutralino(msN4)),
                             dew_funcd(sigmadd_chargino1()),
                             dew_funcd(sigmadd_chargino2()),
                             dew_funcd(sigmadd_h0()),
                             dew_funcd(sigmadd_heavy_h0()),
                             dew_funcd(sigmauu_h_pm()), dew_funcd(sigmauu_w_pm()),
                             dew_funcd(sigmauu_z0()), dew_funcd(sigmadd_bottom()),
                             dew_funcd(sigmadd_tau()), dew_funcu(sigmauu_stop1()),
                             dew_funcu(sigmauu_stop2()),
                             dew_funcu(sigmauu_sbottom1()),
                             dew_funcu(sigmauu_sbottom2()),
                             dew_funcu(sigmauu_stau1()),
                             dew_funcu(sigmauu_stau2()),
                             dew_funcu(sigmauu_sup_l() + sigmauu_sup_r()
                                       + sigmauu_sdown_l() + sigmauu_sdown_r()
                                       + sigmauu_selec_l() + sigmauu_selec_r()
                                       + sigmauu_sel_neut()),
                             dew_funcu(sigmauu_sstrange_l() + sigmauu_sstrange_r()
                                       + sigmauu_scharm_l() + sigmauu_scharm_r()
                                       + sigmauu_smu_l() + sigmauu_smu_r()
                                       + sigmauu_smu_sneut()),
                             dew_funcu(sigmauu_neutralino(msN1)),
                             dew_funcu(sigmauu_neutralino(msN2)),
                             dew_funcu(sigmauu_neutralino(msN3)),
                             dew_funcu(sigmauu_neutralino(msN4)),
                             dew_funcu(sigmauu_chargino1()),
                             dew_funcu(sigmauu_chargino2()),
                             dew_funcu(sigmauu_h0()),
                             dew_funcu(sigmauu_heavy_h0()),
                             dew_funcu(sigmauu_h_pm()),
                             dew_funcu(sigmauu_w_pm()), dew_funcu(sigmauu_z0()),
                             dew_funcu(sigmauu_top()),
                             dew_funcu(sigmauu_h0()),
                             dew_funcu(sigmauu_heavy_h0())]) / halfmzsq
        label_sort_array = np.sort(np.array([(contribs[0], 'mu'),
                                             (contribs[1], 'H_u'),
                                             (contribs[2], 'H_d'),
                                             (contribs[3], 'Sigma_d^d(stop_1)'),
                                             (contribs[4], 'Sigma_d^d(stop_2)'),
                                             (contribs[5], 'Sigma_d^d(sbot_1)'),
                                             (contribs[6], 'Sigma_d^d(sbot_2)'),
                                             (contribs[7], 'Sigma_d^d(stau_1)'),
                                             (contribs[8], 'Sigma_d^d(stau_2)'),
                                             (contribs[9],
                                              'Sigma_d^d(1st gen. squarks)'),
                                             (contribs[10],
                                              'Sigma_d^d(2nd gen squarks)'),
                                             (contribs[11],
                                              'Sigma_d^d(neutralino_1)'),
                                             (contribs[12],
                                              'Sigma_d^d(neutralino_2)'),
                                             (contribs[13],
                                              'Sigma_d^d(neutralino_3)'),
                                             (contribs[14],
                                              'Sigma_d^d(neutralino_4)'),
                                             (contribs[15],
                                              'Sigma_d^d(chargino_1)'),
                                             (contribs[16],
                                              'Sigma_d^d(chargino_2)'),
                                             (contribs[17], 'Sigma_d^d(h_0)'),
                                             (contribs[18], 'Sigma_d^d(H_0)'),
                                             (contribs[19], 'Sigma_d,u^d,u(H_+-)'),
                                             (contribs[20], 'Sigma_d,u^d,u(W_+-)'),
                                             (contribs[21], 'Sigma_d,u^d,u(Z_0)'),
                                             (contribs[22], 'Sigma_d^d(bottom)'),
                                             (contribs[23], 'Sigma_d^d(tau)'),
                                             (contribs[24], 'Sigma_u^u(stop_1)'),
                                             (contribs[25], 'Sigma_u^u(stop_2)'),
                                             (contribs[26], 'Sigma_u^u(sbot_1)'),
                                             (contribs[27], 'Sigma_u^u(sbot_2)'),
                                             (contribs[28], 'Sigma_u^u(stau_1)'),
                                             (contribs[29], 'Sigma_u^u(stau_2)'),
                                             (contribs[30],
                                              'Sigma_u^u(sum 1st gen. squarks)'),
                                             (contribs[31],
                                              'Sigma_u^u(sum 2nd gen. squarks)'),
                                             (contribs[32],
                                              'Sigma_u^u(neutralino_1)'),
                                             (contribs[33],
                                              'Sigma_u^u(neutralino_2)'),
                                             (contribs[34],
                                              'Sigma_u^u(neutralino_3)'),
                                             (contribs[35],
                                              'Sigma_u^u(neutralino_4)'),
                                             (contribs[36],
                                              'Sigma_u^u(chargino_1)'),
                                             (contribs[37],
                                              'Sigma_u^u(chargino_2)'),
                                             (contribs[38], 'Sigma_u^u(h_0)'),
                                             (contribs[39], 'Sigma_u^u(H_0)'),
                                             (contribs[40], 'Sigma_u^u(H_+-)'),
                                             (contribs[41], 'Sigma_u^u(W_+-)'),
                                             (contribs[42], 'Sigma_u^u(Z_0)'),
                                             (contribs[43], 'Sigma_u^u(top)')],
                                            dtype=[('Contrib', float),
                                                   ('label', 'U30')]),
                                   order='Contrib')
        reverse_sort_array = label_sort_array[::-1]
        timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
        print('\nGiven the submitted SLHA file, your value for the electroweak'
              + ' naturalness measure, Delta_EW, is: ' + str(np.amax(contribs)))
        print('\nThe ordered contributions to Delta_EW are as follows ' +
              '(decr. order): ')
        print('')
        for i in range(0, len(reverse_sort_array)):
            print(str(i + 1) + ': ' + str(reverse_sort_array[i]))
        checksave = input("\nWould you like to save these results to a .txt file (will be saved to the current "
                              + "directory)? Enter Y to save the result or N to continue: ")
        if checksave in ('Y', 'y'):
            filenamecheck = input('\nThe default file name is "current_system_time_DEW_contrib_list.txt", e.g., '
                                  + timestr + '_DEW_contrib_list.txt.'
                                  + ' Would you like to keep this or input your own file name?'
                                  +  ' Enter Y to keep the default file name or N to be able to input your own: ')
            if filenamecheck.lower() in ('y', 'yes'):
                print('Given the submitted SLHA file, ' + str(direc) +
                      ', your value for the electroweak\n'
                      + 'naturalness measure, Delta_EW, is: ' + str(np.amax(contribs)),
                      file=open(timestr + "_DEW_contrib_list.txt", "w"))
                print('\nThe ordered contributions to Delta_EW are as follows ' +
                      '(decr. order): ',
                      file=open(timestr + "_DEW_contrib_list.txt", "a"))
                print('', file=open(timestr + "_DEW_contrib_list.txt", "a"))
                for i in range(0, len(reverse_sort_array)):
                    print(str(i + 1) + ': ' + str(reverse_sort_array[i]),
                          file=open(timestr + "_DEW_contrib_list.txt", "a"))
                print('\nThese results have been saved to the current directory as '
                      + timestr + '_DEW_contrib_list.txt.\n')
            else:
                newfilename = input('\nInput your desired filename with no whitespaces and without the .txt file '
                                    + 'extension (e.g. my_SLHA_DEW_list): ')
                print('Given the submitted SLHA file, ' + str(direc) +
                      ', your value for the electroweak\n'
                      + 'naturalness measure, Delta_EW, is: ' + str(np.amax(contribs)),
                      file=open(newfilename + "_DEW_contrib_list.txt", "w"))
                print('\nThe ordered contributions to Delta_EW are as follows ' +
                      '(decr. order): ',
                      file=open(newfilename + "_DEW_contrib_list.txt", "a"))
                print('', file=open(newfilename + "_DEW_contrib_list.txt", "a"))
                for i in range(0, len(reverse_sort_array)):
                    print(str(i + 1) + ': ' + str(reverse_sort_array[i]),
                          file=open(newfilename + "_DEW_contrib_list.txt", "a"))
                print('\nThese results have been saved to the current directory as '
                      + newfilename + '.txt.\n')
        else:
            print("\nOutput not saved or invalid user input.\n")
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
