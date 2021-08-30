# -*- coding: utf-8 -*-
"""
Compute naturalness measure Delta_EW and contributions to DEW.

Author: Dakotah Martinez
"""

import numpy as np
import pyslha
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


# DEW computation: #

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
    d = pyslha.read('test_out_2')
    vHiggs = d.blocks['HMIX'][3]
    muQ = d.blocks['HMIX'][1]
    tanb = d.blocks['HMIX'][2]
    beta = np.arctan(tanb)
    y_t = d.blocks['YU'][3, 3]
    y_b = d.blocks['YD'][3, 3]
    y_tau = d.blocks['YE'][3, 3]
    g_pr = d.blocks['GAUGE'][2]
    g_EW = d.blocks['GAUGE'][1]
    m_stop_1 = d.blocks['MASS'][1000006]
    m_stop_2 = d.blocks['MASS'][2000006]
    m_sbot_1 = d.blocks['MASS'][1000005]
    m_sbot_2 = d.blocks['MASS'][2000005]
    m_stau_1 = d.blocks['MASS'][1000015]
    m_stau_2 = d.blocks['MASS'][2000015]
    mtL = d.blocks['MSOFT'][43]
    mtR = d.blocks['MSOFT'][46]
    mbL = d.blocks['MSOFT'][43]
    mbR = d.blocks['MSOFT'][49]
    mtauL = d.blocks['MSOFT'][33]
    mtauR = d.blocks['MSOFT'][36]
    msupL = d.blocks['MSOFT'][41]
    msupR = d.blocks['MSOFT'][44]
    msdownL = d.blocks['MSOFT'][41]
    msdownR = d.blocks['MSOFT'][47]
    mselecL = d.blocks['MSOFT'][31]
    mselecR = d.blocks['MSOFT'][34]
    mselecneut = d.blocks['MASS'][1000012]
    msmuneut = d.blocks['MASS'][1000014]
    msstrangeL = d.blocks['MSOFT'][42]
    msstrangeR = d.blocks['MSOFT'][48]
    mscharmL = d.blocks['MSOFT'][42]
    mscharmR = d.blocks['MSOFT'][45]
    msmuL = d.blocks['MSOFT'][32]
    msmuR = d.blocks['MSOFT'][35]
    msN1 = d.blocks['MASS'][1000022]
    msN2 = d.blocks['MASS'][1000023]
    msN3 = d.blocks['MASS'][1000025]
    msN4 = d.blocks['MASS'][1000035]
    msC1 = d.blocks['MASS'][1000024]
    msC2 = d.blocks['MASS'][1000037]
    mZ = d.blocks['SMINPUTS'][4]
    mA0sq = d.blocks['HMIX'][4]
    mh0 = d.blocks['MASS'][25]
    mH0 = d.blocks['MASS'][35]
    mHusq = d.blocks['MSOFT'][22]
    mHdsq = d.blocks['MSOFT'][21]
    mH_pm = d.blocks['MASS'][37]
    mgl = d.blocks['MASS'][1000021]
    M_1 = d.blocks['MSOFT'][1]
    M_2 = d.blocks['MSOFT'][2]
    a_t = d.blocks['AU'][3, 3] * y_t
    a_b = d.blocks['AD'][3, 3] * y_b
    a_tau = d.blocks['AE'][3, 3] * y_tau
    Q_renorm_sq = m_stop_1 * m_stop_2
    halfmzsq = np.power(mZ, 2) / 2
    cmu = np.abs(np.power(muQ, 2))
    chu = dew_funcu(mHusq)
    chd = dew_funcd(mHdsq)
    contribs = np.array([cmu, chu, chd, dew_funcd(sigmadd_stop1()),
                         dew_funcd(sigmadd_stop2()),
                         dew_funcd(sigmadd_sbottom1()),
                         dew_funcd(sigmadd_sbottom2()),
                         dew_funcd(sigmadd_stau1()),
                         dew_funcd(sigmadd_stau2()),
                         dew_funcd(sigmadd_sup_l() + sigmadd_sup_r()
                                   + sigmadd_sdown_l()
                                   + sigmadd_sdown_r()
                                   + sigmadd_selec_l()
                                   + sigmadd_selec_r()
                                   + sigmadd_sel_neut()),
                         dew_funcd(sigmadd_sstrange_l()
                                   + sigmadd_sstrange_r()
                                   + sigmadd_scharm_l()
                                   + sigmadd_scharm_r()
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
                         dew_funcd(sigmauu_h_pm()),
                         dew_funcd(sigmauu_w_pm()),
                         dew_funcd(sigmauu_z0()),
                         dew_funcd(sigmadd_bottom()),
                         dew_funcd(sigmadd_tau()),
                         dew_funcu(sigmauu_stop1()),
                         dew_funcu(sigmauu_stop2()),
                         dew_funcu(sigmauu_sbottom1()),
                         dew_funcu(sigmauu_sbottom2()),
                         dew_funcu(sigmauu_stau1()),
                         dew_funcu(sigmauu_stau2()),
                         dew_funcu(sigmauu_sup_l() + sigmauu_sup_r()
                                   + sigmauu_sdown_l()
                                   + sigmauu_sdown_r()
                                   + sigmauu_selec_l()
                                   + sigmauu_selec_r()
                                   + sigmauu_sel_neut()),
                         dew_funcu(sigmauu_sstrange_l()
                                   + sigmauu_sstrange_r()
                                   + sigmauu_scharm_l()
                                   + sigmauu_scharm_r()
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
                         dew_funcu(sigmauu_w_pm()),
                         dew_funcu(sigmauu_z0()),
                         dew_funcu(sigmauu_top())]) / halfmzsq
    calculated_dew_array = np.amax(contribs)
    try: # Get rid of bad points
        test_if_bad = d.blocks['SPINFO'][4]
        calculated_dew_array = 1000
    except KeyError: # Keep good points!
        pass
    print(calculated_dew_array)
