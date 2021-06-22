# -*- coding: utf-8 -*-
import numpy as np
import pyslha

#########################
# Mass relations:
#########################


def mWsq(g, vHiggs):
    mymWsq = (np.power(g, 2) / 2) * np.power(vHiggs, 2)
    return mymWsq


def mZsq(g, g_prime, vHiggs):
    mymZsq = ((np.power(g, 2) + np.power(g_prime, 2)) / 2) * \
        np.power(vHiggs, 2)
    return mymZsq


def mA0sq(mu, mHusq, mHdsq):
    mymA0sq = 2 * np.power(np.abs(mu), 2) + mHusq + mHdsq
    return mymA0sq


def mHpmsq(mu, mHusq, mHdsq, g, vHiggs):
    mymWsq = mWsq(g, vHiggs)
    mymHpmsq = mA0sq(mu, mHusq, mHdsq) + mymWsq
    return mymHpmsq


#########################
# Fundamental equations
#########################


def F(m, Q_renorm):
    myF = np.power(m, 2) * (np.log((np.power(m, 2))
                                   / (np.power(Q_renorm, 2))) - 1)
    return myF


def sinsqb(tanb):  # sin^2(beta)
    mysinsqb = (np.power(tanb, 2) / (1 + np.power(tanb, 2)))
    return mysinsqb


def cossqb(tanb):  # cos^2(beta)
    mycossqb = 1 - (np.power(tanb, 2) / (1 + np.power(tanb, 2)))
    return mycossqb


def vu(vHiggs, tanb):  # up Higgs VEV
    myvu = vHiggs * np.sqrt(sinsqb(tanb))
    return myvu


def vd(vHiggs, tanb):  # down Higgs VEV
    myvd = vHiggs * np.sqrt(cossqb(tanb))
    return myvd


def tan_theta_W(g, g_prime):  # tan(theta_W)
    mytanthetaW = g_prime / g
    return mytanthetaW


def sin_squared_theta_W(g, g_prime):  # sin^2(theta_W)
    mysinsqthetaW = (np.power(tan_theta_W(g, g_prime), 2)
                     / (1 + np.power(tan_theta_W(g, g_prime), 2)))
    return mysinsqthetaW


def cos_squared_theta_W(g, g_prime):  # cos^2(theta_W)
    mycossqthetaW = 1 - (np.power(tan_theta_W(g, g_prime), 2)
                         / (1 + np.power(tan_theta_W(g, g_prime), 2)))
    return mycossqthetaW


#########################
# Stop squarks:
#########################


def sigmauu_stop1(vHiggs, mu, tanb, y_t, g, g_prime, m_stop_1, m_stop_2,
                  mtL, mtR, a_t, Q_renorm):
    delta_uL = ((1 / 2) - (2 / 3) * sin_squared_theta_W(g, g_prime)) \
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    delta_uR = (2 / 3) * sin_squared_theta_W(g, g_prime)\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    stop_num = ((np.power(mtL, 2) - np.power(mtR, 2) + delta_uL - delta_uR)
                * ((-1 / 2) + ((4 / 3) * sin_squared_theta_W(g, g_prime)))
                * (np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        + 2 * np.power(a_t, 2)
    Sigmauu_stop1 = (3 / (32 * (np.power(np.pi, 2)))) *\
                    (2 * np.power(y_t, 2) + ((np.power(g, 2)
                                              + np.power(g_prime, 2))
                                             * (8 *
                                                sin_squared_theta_W(g, g_prime)
                                                - 3) / 12)
                     - (stop_num / (np.power(m_stop_2, 2)
                                    - np.power(m_stop_1, 2))))\
        * F(m_stop_1, Q_renorm)
    return Sigmauu_stop1


def sigmadd_stop1(vHiggs, mu, tanb, y_t, g, g_prime, m_stop_1, m_stop_2, mtL,
                  mtR, a_t, Q_renorm):
    delta_uL = ((1 / 2) - (2 / 3) * sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    delta_uR = (2 / 3) * sin_squared_theta_W(g, g_prime)\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    stop_num = ((np.power(mtL, 2) - np.power(mtR, 2) + delta_uL - delta_uR)
                * ((1 / 2) + (4 / 3) * sin_squared_theta_W(g, g_prime))
                * (np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        + 2 * np.power(y_t, 2) * np.power(mu, 2)
    Sigmadd_stop1 = (3 / (32 * (np.power(np.pi, 2))))\
        * (((np.power(g, 2) + np.power(g_prime, 2)) / 2)
           - (stop_num / (np.power(m_stop_2, 2) - np.power(m_stop_1, 2))))\
        * F(m_stop_1, Q_renorm)
    return Sigmadd_stop1


def sigmauu_stop2(vHiggs, mu, tanb, y_t, g, g_prime, m_stop_1, m_stop_2, mtL,
                  mtR, a_t, Q_renorm):
    delta_uL = ((1 / 2) - (2 / 3) * sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    delta_uR = (2 / 3) * sin_squared_theta_W(g, g_prime)\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    stop_num = ((np.power(mtL, 2) - np.power(mtR, 2) + delta_uL - delta_uR)
                * ((-1 / 2) + ((4 / 3) * sin_squared_theta_W(g, g_prime)))
                * (np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        + 2 * np.power(a_t, 2)
    Sigmauu_stop2 = (3 / (32 * (np.power(np.pi, 2))))\
        * (2 * np.power(y_t, 2) + ((np.power(g, 2) + np.power(g_prime, 2))
                                   * (8 * sin_squared_theta_W(g, g_prime) - 3)
                                   / 12)
           + (stop_num / (np.power(m_stop_2, 2) - np.power(m_stop_1, 2))))\
        * F(m_stop_2, Q_renorm)
    return Sigmauu_stop2


def sigmadd_stop2(vHiggs, mu, tanb, y_t, g, g_prime, m_stop_1, m_stop_2, mtL,
                  mtR, a_t, Q_renorm):
    delta_uL = ((1 / 2) - (2 / 3) * sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    delta_uR = (2 / 3) * sin_squared_theta_W(g, g_prime)\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    stop_num = ((np.power(mtL, 2) - np.power(mtR, 2) + delta_uL - delta_uR)
                * ((1 / 2) + (4 / 3) * sin_squared_theta_W(g, g_prime))
                * (np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        + 2 * np.power(y_t, 2) * np.power(mu, 2)
    Sigmadd_stop = (3 / (32 * (np.power(np.pi, 2))))\
        * (((np.power(g, 2) + np.power(g_prime, 2)) / 2)
           - (stop_num / (np.power(m_stop_2, 2) - np.power(m_stop_1, 2))))\
        * F(m_stop_2, Q_renorm)
    return Sigmadd_stop


#########################
# Sbottom squarks:
#########################


def sigmauu_sbottom1(vHiggs, mu, tanb, y_b, g, g_prime, m_sbot_1, m_sbot_2,
                     mbL, mbR, a_b, Q_renorm):
    delta_dL = ((-1 / 2) + (1 / 3) * sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    delta_dR = ((-1 / 3) * sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    sbot_num = (np.power(mbR, 2) - np.power(mbL, 2) + delta_dR - delta_dL)\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * ((1 / 2) + (2 / 3) * sin_squared_theta_W(g, g_prime))\
        + 2 * np.power(y_b, 2) * np.power(mu, 2)
    Sigmauu_sbot = (3 / (32 * np.power(np.pi, 2)))\
        * (((np.power(g, 2) + np.power(g_prime, 2)) / 4)
           - (sbot_num / (np.power(m_sbot_2, 2) - np.power(m_sbot_1, 2))))\
        * F(m_sbot_1, Q_renorm)
    return Sigmauu_sbot


def sigmauu_sbottom2(vHiggs, mu, tanb, y_b, g, g_prime, m_sbot_1, m_sbot_2,
                     mbL, mbR, a_b, Q_renorm):
    delta_dL = ((-1 / 2) + (1 / 3) * sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    delta_dR = ((-1 / 3) * sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    sbot_num = (np.power(mbR, 2) - np.power(mbL, 2) + delta_dR - delta_dL)\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * ((1 / 2) + (2 / 3) * sin_squared_theta_W(g, g_prime))\
        + 2 * np.power(y_b, 2) * np.power(mu, 2)
    Sigmauu_sbot = (3 / (32 * np.power(np.pi, 2)))\
        * (((np.power(g, 2) + np.power(g_prime, 2)) / 4)
           + (sbot_num / (np.power(m_sbot_2, 2) - np.power(m_sbot_1, 2))))\
        * F(m_sbot_2, Q_renorm)
    return Sigmauu_sbot


def sigmadd_sbottom1(vHiggs, mu, tanb, y_b, g, g_prime, m_sbot_1, m_sbot_2,
                     mbL, mbR, a_b, Q_renorm):
    delta_dL = ((-1 / 2) + (1 / 3) * sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    delta_dR = ((-1 / 3) * sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    sbot_num = (np.power(mbR, 2) - np.power(mbL, 2) + delta_dR - delta_dL)\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * ((-1 / 2) - (2 / 3) * sin_squared_theta_W(g, g_prime))\
        + 2 * np.power(a_b, 2)
    Sigmadd_sbot = (3 / (32 * np.power(np.pi, 2)))\
        * (((-1) * (np.power(g, 2) + np.power(g_prime, 2)) / 4)
           - (sbot_num / (np.power(m_sbot_2, 2) - np.power(m_sbot_1, 2))))\
        * F(m_sbot_1, Q_renorm)
    return Sigmadd_sbot


def sigmadd_sbottom2(vHiggs, mu, tanb, y_b, g, g_prime, m_sbot_1, m_sbot_2,
                     mbL, mbR, a_b, Q_renorm):
    delta_dL = ((-1 / 2) + (1 / 3) * sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    delta_dR = ((-1 / 3) * sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    sbot_num = (np.power(mbR, 2) - np.power(mbL, 2) + delta_dR - delta_dL)\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * ((-1 / 2) - (2 / 3) * sin_squared_theta_W(g, g_prime))\
        + 2 * np.power(a_b, 2)
    Sigmadd_sbot = (3 / (32 * np.power(np.pi, 2)))\
        * (((-1) * (np.power(g, 2) + np.power(g_prime, 2)) / 4)
           + (sbot_num / (np.power(m_sbot_2, 2) - np.power(m_sbot_1, 2))))\
        * F(m_sbot_2, Q_renorm)
    return Sigmadd_sbot


#########################
# Stau sleptons:
#########################


def sigmauu_stau1(vHiggs, mu, tanb, y_tau, g, g_prime, m_stau_1, m_stau_2,
                  mtauL, mtauR, a_tau, Q_renorm):
    delta_eL = ((-1 / 2) + sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    delta_eR = ((-1) * sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    stau_num = (np.power(mtauR, 2) - np.power(mtauL, 2) + delta_eR - delta_eL)\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * ((-1 / 2) + 2 * sin_squared_theta_W(g, g_prime))\
        + 2 * np.power(y_tau, 2) * np.power(mu, 2)
    Sigmauu_stau = (1 / (32 * np.power(np.pi, 2)))\
        * (((np.power(g, 2) + np.power(g_prime, 2)) / 4)
           - (stau_num / (np.power(m_stau_2, 2) - np.power(m_stau_1, 2))))\
        * F(m_stau_1, Q_renorm)
    return Sigmauu_stau


def sigmauu_stau2(vHiggs, mu, tanb, y_tau, g, g_prime, m_stau_1, m_stau_2,
                  mtauL, mtauR, a_tau, Q_renorm):
    delta_eL = ((-1 / 2) + sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    delta_eR = ((-1) * sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    stau_num = (np.power(mtauR, 2) - np.power(mtauL, 2) + delta_eR - delta_eL)\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * ((-1 / 2) + 2 * sin_squared_theta_W(g, g_prime))\
        + 2 * np.power(y_tau, 2) * np.power(mu, 2)
    Sigmauu_stau = (1 / (32 * np.power(np.pi, 2)))\
        * (((np.power(g, 2) + np.power(g_prime, 2)) / 4)
           + (stau_num / (np.power(m_stau_2, 2) - np.power(m_stau_1, 2))))\
        * F(m_stau_2, Q_renorm)
    return Sigmauu_stau


def sigmadd_stau1(vHiggs, mu, tanb, y_tau, g, g_prime, m_stau_1, m_stau_2,
                  mtauL, mtauR, a_tau, Q_renorm):
    delta_eL = ((-1 / 2) + sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    delta_eR = ((-1) * sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    stau_num = (np.power(mtauR, 2) - np.power(mtauL, 2) + delta_eR - delta_eL)\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * ((1 / 2) - (2 * sin_squared_theta_W(g, g_prime)))\
        + 2 * np.power(a_tau, 2)
    Sigmadd_stau = (1 / (32 * np.power(np.pi, 2)))\
        * (((-1) * (np.power(g, 2) + np.power(g_prime, 2)) / 4)
           - (stau_num / (np.power(m_stau_2, 2) - np.power(m_stau_1, 2))))\
        * F(m_stau_1, Q_renorm)
    return Sigmadd_stau


def sigmadd_stau2(vHiggs, mu, tanb, y_tau, g, g_prime, m_stau_1, m_stau_2,
                  mtauL, mtauR, a_tau, Q_renorm):
    delta_eL = ((-1 / 2) + sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    delta_eR = ((-1) * sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    stau_num = (np.power(mtauR, 2) - np.power(mtauL, 2) + delta_eR - delta_eL)\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * ((1 / 2) - (2 * sin_squared_theta_W(g, g_prime)))\
        + 2 * np.power(a_tau, 2)
    Sigmadd_stau = (1 / (32 * np.power(np.pi, 2)))\
        * (((-1) * (np.power(g, 2) + np.power(g_prime, 2)) / 4)
           + (stau_num / (np.power(m_stau_2, 2) - np.power(m_stau_1, 2))))\
        * F(m_stau_2, Q_renorm)
    return Sigmadd_stau


#########################
# Sfermions, 1st gen:
#########################


def sigmauu_sup_L(g, g_prime, msupL, Q_renorm):
    SigmauusupL = ((-3) / (16 * np.power(np.pi, 2)))\
        * ((1 / 2) - (2 / 3) * sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2) * F(msupL, Q_renorm)
    return SigmauusupL


def sigmauu_sup_R(g, g_prime, msupR, Q_renorm):
    SigmauusupR = ((-3) / (16 * np.power(np.pi, 2)))\
        * ((2 / 3) * sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2) * F(msupR, Q_renorm)
    return SigmauusupR


def sigmauu_sdown_L(g, g_prime, msdownL, Q_renorm):
    SigmauusdownL = ((-3) / (16 * np.power(np.pi, 2)))\
        * (((-1) / 2) + (1 / 3) * sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2) * F(msdownL, Q_renorm)
    return SigmauusdownL


def sigmauu_sdown_R(g, g_prime, msdownR, Q_renorm):
    SigmauusdownR = ((-3) / (16 * np.power(np.pi, 2)))\
        * (((-1) / 3) * sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2) * F(msdownR, Q_renorm)
    return SigmauusdownR


def sigmauu_selec_L(g, g_prime, mselecL, Q_renorm):
    SigmauuselecL = ((-1) / (16 * np.power(np.pi, 2)))\
        * (((-1) / 2) + sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2) * F(mselecL, Q_renorm)
    return SigmauuselecL


def sigmauu_selec_R(g, g_prime, mselecR, Q_renorm):
    SigmauuselecR = ((-1) / (16 * np.power(np.pi, 2)))\
        * ((-1) * sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2) * F(mselecR, Q_renorm)
    return SigmauuselecR


def sigmauu_selecSneut(g, g_prime, mselecSneut, Q_renorm):
    SigmauuselecSneut = ((-1) / (32 * np.power(np.pi, 2))) * (1 / 2)\
                    * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
                    * F(mselecSneut, Q_renorm)
    return SigmauuselecSneut


def sigmadd_sup_L(g, g_prime, msupL, Q_renorm):
    SigmaddsupL = (3 / (16 * np.power(np.pi, 2)))\
        * ((1 / 2) - (2 / 3) * sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2) * F(msupL, Q_renorm)
    return SigmaddsupL


def sigmadd_sup_R(g, g_prime, msupR, Q_renorm):
    SigmaddsupR = (3 / (16 * np.power(np.pi, 2)))\
        * ((2 / 3) * sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2) * F(msupR, Q_renorm)
    return SigmaddsupR


def sigmadd_sdown_L(g, g_prime, msdownL, Q_renorm):
    SigmaddsdownL = (3 / (16 * np.power(np.pi, 2)))\
        * (((-1) / 2) + (1 / 3) * sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2) * F(msdownL, Q_renorm)
    return SigmaddsdownL


def sigmadd_sdown_R(g, g_prime, msdownR, Q_renorm):
    SigmaddsdownR = (3 / (16 * np.power(np.pi, 2)))\
        * (((-1) / 3) * sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2) * F(msdownR, Q_renorm)
    return SigmaddsdownR


def sigmadd_selec_L(g, g_prime, mselecL, Q_renorm):
    SigmaddselecL = (1 / (16 * np.power(np.pi, 2)))\
        * (((-1) / 2) + sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2) * F(mselecL, Q_renorm)
    return SigmaddselecL


def sigmadd_selec_R(g, g_prime, mselecR, Q_renorm):
    SigmaddselecR = (1 / (16 * np.power(np.pi, 2)))\
        * ((-1) * sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2) * F(mselecR, Q_renorm)
    return SigmaddselecR


def sigmadd_selecSneut(g, g_prime, mselecSneut, Q_renorm):
    SigmauuselecSneut = (1 / (32 * np.power(np.pi, 2))) * (1 / 2)\
                    * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
                    * F(mselecSneut, Q_renorm)
    return SigmauuselecSneut


#########################
# Sfermions, 2nd gen:
#########################


def sigmauu_sstrange_L(g, g_prime, msstrangeL, Q_renorm):
    SigmauusstrangeL = ((-3) / (16 * np.power(np.pi, 2)))\
        * (((-1) / 2) + (1 / 3) * sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * F(msstrangeL, Q_renorm)
    return SigmauusstrangeL


def sigmauu_sstrange_R(g, g_prime, msstrangeR, Q_renorm):
    SigmauusstrangeR = ((-3) / (16 * np.power(np.pi, 2)))\
        * (((-1) / 3) * sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * F(msstrangeR, Q_renorm)
    return SigmauusstrangeR


def sigmauu_scharm_L(g, g_prime, mscharmL, Q_renorm):
    SigmauuscharmL = ((-3) / (16 * np.power(np.pi, 2)))\
        * ((1 / 2) - (2 / 3) * sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * F(mscharmL, Q_renorm)
    return SigmauuscharmL


def sigmauu_scharm_R(g, g_prime, mscharmR, Q_renorm):
    SigmauuscharmR = ((-3) / (16 * np.power(np.pi, 2)))\
        * ((2 / 3) * sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * F(mscharmR, Q_renorm)
    return SigmauuscharmR


def sigmauu_smu_L(g, g_prime, msmuL, Q_renorm):
    SigmauusmuL = ((-1) / (16 * np.power(np.pi, 2)))\
        * (((-1) / 2) + sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * F(msmuL, Q_renorm)
    return SigmauusmuL


def sigmauu_smu_R(g, g_prime, msmuR, Q_renorm):
    SigmauusmuR = ((-1) / (16 * np.power(np.pi, 2)))\
        * (((-1)) * sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * F(msmuR, Q_renorm)
    return SigmauusmuR


def sigmauu_smuSneut(g, g_prime, msmuSneut, Q_renorm):
    SigmauusmuSneut = ((-1) / (32 * np.power(np.pi, 2))) * (1 / 2)\
                * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
                * F(msmuSneut, Q_renorm)
    return SigmauusmuSneut


def sigmadd_sstrange_L(g, g_prime, msstrangeL, Q_renorm):
    SigmaddsstrangeL = (3 / (16 * np.power(np.pi, 2)))\
        * (((-1) / 2) + (1 / 3) * sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * F(msstrangeL, Q_renorm)
    return SigmaddsstrangeL


def sigmadd_sstrange_R(g, g_prime, msstrangeR, Q_renorm):
    SigmaddsstrangeR = (3 / (16 * np.power(np.pi, 2)))\
        * (((-1) / 3) * sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * F(msstrangeR, Q_renorm)
    return SigmaddsstrangeR


def sigmadd_scharm_L(g, g_prime, mscharmL, Q_renorm):
    SigmaddscharmL = (3 / (16 * np.power(np.pi, 2)))\
        * ((1 / 2) - (2 / 3) * sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * F(mscharmL, Q_renorm)
    return SigmaddscharmL


def sigmadd_scharm_R(g, g_prime, mscharmR, Q_renorm):
    SigmaddscharmR = (3 / (16 * np.power(np.pi, 2)))\
        * ((2 / 3) * sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * F(mscharmR, Q_renorm)
    return SigmaddscharmR


def sigmadd_smu_L(g, g_prime, msmuL, Q_renorm):
    SigmaddsmuL = (1 / (16 * np.power(np.pi, 2)))\
        * (((-1) / 2) + sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * F(msmuL, Q_renorm)
    return SigmaddsmuL


def sigmadd_smu_R(g, g_prime, msmuR, Q_renorm):
    SigmaddsmuR = (1 / (16 * np.power(np.pi, 2)))\
        * (((-1)) * sin_squared_theta_W(g, g_prime))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
        * F(msmuR, Q_renorm)
    return SigmaddsmuR


def sigmadd_smuSneut(g, g_prime, msmuSneut, Q_renorm):
    SigmaddsmuSneut = (1 / (32 * np.power(np.pi, 2))) * (1 / 2)\
                * ((np.power(g, 2) + np.power(g_prime, 2)) / 2)\
                * F(msmuSneut, Q_renorm)
    return SigmaddsmuSneut


#########################
# Neutralinos:
#########################
# Set up terms from characteristic polynomial for eigenvalues x of squared
# neutralino mass matrix,
# x^4 + b(vu, vd) * x^3 + c(vu, vd) * x^2 + d(vu, vd) * x + e(vu, vd) = 0


def neutralinouu_deriv_num(M1, M2, mu, g, g_prime, vHiggs, tanb, msN):
    cubicterm = np.power(g, 2) + np.power(g_prime, 2)
    quadrterm = (((np.power(g, 2) * M2 * mu)
                  + (np.power(g_prime, 2) * M1 * mu)) / (tanb))\
        - ((np.power(g, 2) * np.power(M1, 2)) + (np.power(g_prime, 2)
                                                 * np.power(M2, 2))
           + ((np.power(g, 2) + np.power(g_prime, 2)) * (np.power(mu, 2)))
           + (np.power((np.power(g, 2) + np.power(g_prime, 2)), 2) / 2)
           * np.power(vHiggs, 2))
    linterm = (((-1) * mu) * ((np.power(g, 2) * M2
                               * (np.power(M1, 2) + np.power(mu, 2)))
                              + np.power(g_prime, 2) * M1
                              * (np.power(M2, 2) + np.power(mu, 2))) / tanb)\
        + ((np.power((np.power(g, 2) * M1 + np.power(g_prime, 2) * M2), 2) / 2)
           * np.power(vHiggs, 2))\
        + (np.power(mu, 2) * ((np.power(g, 2) * np.power(M1, 2))
                              + np.power(g_prime, 2) * np.power(M2, 2)))\
        + (np.power((np.power(g, 2) + np.power(g_prime, 2)), 2)
           * np.power(vHiggs, 2) * np.power(mu, 2) * cossqb(tanb))
    constterm = (M1 * M2 * ((np.power(g, 2) * M1) 
                            + (np.power(g_prime, 2) * M2))
                 * np.power(mu, 3) * (1 / tanb))\
        - (np.power((np.power(g, 2) * M1 + np.power(g_prime, 2) * M2), 2)
           * np.power(vHiggs, 2) * np.power(mu, 2) * cossqb(tanb))
    mynum = (cubicterm * np.power(msN, 6)) + (quadrterm * np.power(msN, 4))\
        + (linterm * np.power(msN, 2)) + constterm
    return mynum


def neutralinodd_deriv_num(M1, M2, mu, g, g_prime, vHiggs, tanb, msN):
    cubicterm = np.power(g, 2) + np.power(g_prime, 2)
    quadrterm = (((np.power(g, 2) * M2 * mu)
                  + (np.power(g_prime, 2) * M1 * mu)) * (tanb))\
        - ((np.power(g, 2) * np.power(M1, 2))
           + (np.power(g_prime, 2) * np.power(M2, 2))
           + ((np.power(g, 2) + np.power(g_prime, 2)) * (np.power(mu, 2)))
           + (np.power((np.power(g, 2) + np.power(g_prime, 2)), 2) / 2)
           * np.power(vHiggs, 2))
    linterm = (((-1) * mu) * ((np.power(g, 2) * M2
                               * (np.power(M1, 2) + np.power(mu, 2)))
                              + np.power(g_prime, 2) * M1
                              * (np.power(M2, 2) + np.power(mu, 2))) * tanb)\
        + ((np.power((np.power(g, 2) * M1 + np.power(g_prime, 2) * M2), 2) / 2)
           * np.power(vHiggs, 2))\
        + (np.power(mu, 2) * (np.power(g, 2) * np.power(M1, 2)
           + np.power(g_prime, 2) * np.power(M2, 2)))\
        + (np.power((np.power(g, 2) + np.power(g_prime, 2)), 2)
           * np.power(vHiggs, 2) * np.power(mu, 2) * sinsqb(tanb))
    constterm = (M1 * M2 * (np.power(g, 2) * M1 + np.power(g_prime, 2) * M2)
                 * np.power(mu, 3) * tanb)\
        - (np.power((np.power(g, 2) * M1 + np.power(g_prime, 2) * M2), 2)
           * np.power(vHiggs, 2) * np.power(mu, 2) * sinsqb(tanb))
    mynum = (cubicterm * np.power(msN, 6))\
        + (quadrterm * np.power(msN, 4))\
        + (linterm * np.power(msN, 2)) + constterm
    return mynum


def neutralino_deriv_denom(M1, M2, mu, g, g_prime, vHiggs, tanb, msN):
    quadrterm = -3 * ((np.power(M1, 2)) + (np.power(M2, 2))
                      + ((np.power(g, 2) + np.power(g_prime, 2))
                         * np.power(vHiggs, 2))
                      + (2 * np.power(mu, 2)))
    linterm = (np.power(vHiggs, 4)
               * np.power((np.power(g, 2) + np.power(g_prime, 2)), 2) / 2)\
        + (np.power(vHiggs, 2)
           * (2 * ((np.power(g, 2) * np.power(M1, 2))
                   + (np.power(g_prime, 2) * np.power(M2, 2))
                   + ((np.power(g, 2)
                       + np.power(g_prime, 2)) * np.power(mu, 2))
                   - (mu * (np.power(g_prime, 2) * M1 + np.power(g, 2) * M2)
                      * 2 * np.sqrt(sinsqb(tanb)) * np.sqrt(cossqb(tanb))))))\
        + (2 * ((np.power(M1, 2) * np.power(M2, 2))
                + (2 * (np.power(M1, 2) + np.power(M2, 2)) * np.power(mu, 2))
                + (np.power(mu, 4))))
    constterm = (np.power(vHiggs, 4) * (1 / 8)
                 * ((np.power((np.power(g, 2) + np.power(g_prime, 2)), 2)
                     * np.power(mu, 2)
                     * (np.power(cossqb(tanb), 2)
                        - (6 * cossqb(tanb) * sinsqb(tanb))
                        + np.power(sinsqb(tanb), 2)))
                    - (2 * np.power((np.power(g, 2) * M1
                                     + np.power(g_prime, 2) * M2), 2))
                    - (np.power(mu, 2) * np.power((np.power(g, 2)
                                                   + np.power(g_prime, 2)), 2))
                    ))\
        + (np.power(vHiggs, 2) * 2 * mu
           * ((np.sqrt(cossqb(tanb)) * np.sqrt(sinsqb(tanb)))
              * (np.power(g, 2) * M2 * (np.power(M1, 2) + np.power(mu, 2))
                 + (np.power(g_prime, 2) * M1
                 * (np.power(M2, 2) + np.power(mu, 2))))))\
        - ((2 * np.power(M2, 2) * np.power(M1, 2) * np.power(mu, 2))
           + (np.power(mu, 4) * (np.power(M1, 2) + np.power(M2, 2))))
    mydenom = 4 * np.power(msN, 6)\
        + quadrterm * np.power(msN, 4)\
        + linterm * np.power(msN, 2)\
        + constterm
    return mydenom


def sigmauu_neutralino(M1, M2, mu, g, g_prime, vHiggs, tanb, msN, Q_renorm):
    Sigmauu_neutralino = ((-1) / (16 * np.power(np.pi, 2))) \
                          * (neutralinouu_deriv_num(M1, M2, mu, g, g_prime,
                                                    vHiggs, tanb, msN)
                             / neutralino_deriv_denom(M1, M2, mu, g, g_prime,
                                                      vHiggs, tanb, msN))\
                          * F(msN, Q_renorm)
    return Sigmauu_neutralino


def sigmadd_neutralino(M1, M2, mu, g, g_prime, vHiggs, tanb, msN, Q_renorm):
    Sigmadd_neutralino = ((-1) / (16 * np.power(np.pi, 2))) \
                          * (neutralinodd_deriv_num(M1, M2, mu, g, g_prime,
                                                    vHiggs, tanb, msN)
                             / neutralino_deriv_denom(M1, M2, mu, g, g_prime,
                                                      vHiggs, tanb, msN))\
                          * F(msN, Q_renorm)
    return Sigmadd_neutralino


#########################
# Charginos:
#########################


def sigmauu_chargino1(g, M2, vHiggs, tanb, mu, msC, Q_renorm):
    chargino_num = np.power(M2, 2) + np.power(mu, 2)\
        + (np.power(g, 2) * (np.power(vu(vHiggs, tanb), 2)
                             - np.power(vd(vHiggs, tanb), 2)))
    chargino_den = np.sqrt(((np.power(g, 2)
                             * np.power((vu(vHiggs, tanb)
                                         + vd(vHiggs, tanb)), 2))
                            + np.power((M2 - mu), 2))
                           * ((np.power(g, 2)
                               * np.power((vd(vHiggs, tanb)
                                           - vu(vHiggs, tanb)), 2))
                              + np.power((M2 + mu), 2)))
    Sigmauu_chargino1 = -1 * (np.power(g, 2) / (16 * np.power(np.pi, 2)))\
        * (1 - (chargino_num / chargino_den)) * F(msC, Q_renorm)
    return Sigmauu_chargino1


def sigmauu_chargino2(g, M2, vHiggs, tanb, mu, msC, Q_renorm):
    chargino_num = np.power(M2, 2) + np.power(mu, 2)\
        + (np.power(g, 2) * (np.power(vu(vHiggs, tanb), 2)
                             - np.power(vd(vHiggs, tanb), 2)))
    chargino_den = np.sqrt(((np.power(g, 2)
                             * np.power((vu(vHiggs, tanb)
                                         + vd(vHiggs, tanb)), 2))
                            + np.power((M2 - mu), 2))
                           * ((np.power(g, 2)
                               * np.power((vd(vHiggs, tanb)
                                           - vu(vHiggs, tanb)), 2))
                              + np.power((M2 + mu), 2)))
    Sigmauu_chargino2 = -1 * (np.power(g, 2) / (16 * np.power(np.pi, 2)))\
        * (1 + (chargino_num / chargino_den)) * F(msC, Q_renorm)
    return Sigmauu_chargino2


def sigmadd_chargino1(g, M2, vHiggs, tanb, mu, msC, Q_renorm):
    chargino_num = np.power(M2, 2) + np.power(mu, 2)\
        - (np.power(g, 2) * (np.power(vu(vHiggs, tanb), 2)
                             - np.power(vd(vHiggs, tanb), 2)))
    chargino_den = np.sqrt(((np.power(g, 2)
                             * np.power((vu(vHiggs, tanb)
                                         + vd(vHiggs, tanb)), 2))
                            + np.power((M2 - mu), 2))
                           * ((np.power(g, 2)
                               * np.power((vd(vHiggs, tanb)
                                           - vu(vHiggs, tanb)), 2))
                              + np.power((M2 + mu), 2)))
    Sigmadd_chargino1 = -1 * (np.power(g, 2) / (16 * np.power(np.pi, 2)))\
        * (1 - (chargino_num / chargino_den)) * F(msC, Q_renorm)
    return Sigmadd_chargino1


def sigmadd_chargino2(g, M2, vHiggs, tanb, mu, msC, Q_renorm):
    chargino_num = np.power(M2, 2) + np.power(mu, 2)\
        - (np.power(g, 2) * (np.power(vu(vHiggs, tanb), 2)
                             - np.power(vd(vHiggs, tanb), 2)))
    chargino_den = np.sqrt(((np.power(g, 2)
                             * np.power((vu(vHiggs, tanb)
                                         + vd(vHiggs, tanb)), 2))
                            + np.power((M2 - mu), 2))
                           * ((np.power(g, 2)
                               * np.power((vd(vHiggs, tanb)
                                           - vu(vHiggs, tanb)), 2))
                              + np.power((M2 + mu), 2)))
    Sigmadd_chargino2 = -1 * (np.power(g, 2) / (16 * np.power(np.pi, 2)))\
        * (1 + (chargino_num / chargino_den)) * F(msC, Q_renorm)
    return Sigmadd_chargino2


#########################
# Higgs bosons (sigmauu = sigmadd here):
#########################


def sigmauu_h0(g, g_prime, vHiggs, tanb, mHusq, mHdsq, mu, mZ, mh0, Q_renorm):
    mynum = ((np.power(g, 2) + np.power(g_prime, 2))
             * np.power(vHiggs, 2))\
        - (2 * mA0sq(mu, mHusq, mHdsq) * (np.power(cossqb(tanb), 2)
                                          - 6 * cossqb(tanb) * sinsqb(tanb)
                                          + np.power(sinsqb(tanb), 2)))
    myden = np.sqrt(np.power((mA0sq(mu, mHusq, mHdsq) - np.power(mZ, 2)), 2)
                    + (4 * np.power(mZ, 2) * mA0sq(mu, mHusq, mHdsq) * 4
                       * cossqb(tanb) * sinsqb(tanb)))
    Sigmauu_h0 = (1 / (32 * np.power(np.pi, 2)))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 4)\
        * (1 - (mynum / myden)) * F(mh0, Q_renorm)
    return Sigmauu_h0


def sigmauu_H0(g, g_prime, vHiggs, tanb, mHusq, mHdsq, mu, mZ, mH0, Q_renorm):
    mynum = ((np.power(g, 2) + np.power(g_prime, 2)) * np.power(vHiggs, 2))\
        - (2 * mA0sq(mu, mHusq, mHdsq) * (np.power(cossqb(tanb), 2)
                                          - 6 * cossqb(tanb) * sinsqb(tanb)
                                          + np.power(sinsqb(tanb), 2)))
    myden = np.sqrt(np.power((mA0sq(mu, mHusq, mHdsq) - np.power(mZ, 2)), 2)
                    + (4 * np.power(mZ, 2) * mA0sq(mu, mHusq, mHdsq)
                       * 4 * cossqb(tanb) * sinsqb(tanb)))
    Sigmauu_H0 = (1/(32 * np.power(np.pi, 2)))\
        * ((np.power(g, 2) + np.power(g_prime, 2)) / 4)\
        * (1 + (mynum / myden)) * F(mH0, Q_renorm)
    return Sigmauu_H0


def sigmauu_H_pm(g, mH_pm, Q_renorm):
    Sigmauu_H_pm = (np.power(g, 2) / (64 * np.power(np.pi, 2)))\
        * F(mH_pm, Q_renorm)
    return Sigmauu_H_pm


#########################
# Weak bosons (sigmauu = sigmadd here):
#########################


def sigmauu_W_pm(g, vHiggs, Q_renorm):
    mymWsq = mWsq(g, vHiggs)
    Sigmauu_W_pm = (3 * np.power(g, 2) / (32 * np.power(np.pi, 2)))\
        * F(np.sqrt(mymWsq), Q_renorm)
    return Sigmauu_W_pm


def sigmauu_Z0(g, g_prime, vHiggs, Q_renorm):
    mymZsq = mZsq(g, g_prime, vHiggs)
    Sigmauu_W_pm = (3 * np.power(g, 2) / (64 * np.power(np.pi, 2)))\
        * F(np.sqrt(mymZsq), Q_renorm)
    return Sigmauu_W_pm


#########################
# SM fermions (sigmadd_t = sigmauu_b = sigmauu_tau = 0):
#########################


def sigmauu_top(yt, vHiggs, tanb, Q_renorm):
    mymt = yt * vu(vHiggs, tanb)
    Sigmauu_top = ((-1) * np.power(yt, 2) / (16 * np.power(np.pi, 2)))\
        * F(mymt, Q_renorm)
    return Sigmauu_top


def sigmadd_top(yt, vHiggs, tanb, Q_renorm):
    return 0


def sigmauu_bottom(yb, vHiggs, tanb, Q_renorm):
    return 0


def sigmadd_bottom(yb, vHiggs, tanb, Q_renorm):
    mymb = yb * vd(vHiggs, tanb)
    Sigmadd_bottom = (-1 * np.power(yb, 2) / (16 * np.power(np.pi, 2)))\
        * F(mymb, Q_renorm)
    return Sigmadd_bottom


def sigmauu_tau(ytau, vHiggs, tanb, Q_renorm):
    return 0


def sigmadd_tau(ytau, vHiggs, tanb, Q_renorm):
    mymtau = ytau * vd(vHiggs, tanb)
    Sigmadd_tau = (-1 * np.power(ytau, 2) / (16 * np.power(np.pi, 2)))\
        * F(mymtau, Q_renorm)
    return Sigmadd_tau


#########################
# Sigmauu computation
#########################


def sigmauu_net(vHiggs, mu, tanb, y_t, y_b, y_tau, g, g_prime, m_stop_1,
                m_stop_2, m_sbot_1, m_sbot_2, m_stau_1, m_stau_2, mtL, mtR,
                mbL, mbR, mtauL, mtauR, msupL, msupR, msdownL, msdownR,
                mselecL, mselecR, mselecSneut, msstrangeL, msstrangeR,
                mscharmL, mscharmR, msmuL, msmuR, msmuSneut, msN1, msN2, msN3,
                msN4, msC1, msC2, mZ, mh0, mH0, mHusq, mHdsq, mH_pm, M1, M2,
                a_t, a_b, a_tau, Q_renorm):
    Sigmauunet = sigmauu_stop1(vHiggs, mu, tanb, y_t, g, g_prime, m_stop_1,
                               m_stop_2, mtL, mtR, a_t, Q_renorm)\
            + sigmauu_stop2(vHiggs, mu, tanb, y_t, g, g_prime, m_stop_1,
                            m_stop_2, mtL, mtR, a_t, Q_renorm)\
            + sigmauu_sbottom1(vHiggs, mu, tanb, y_b, g, g_prime, m_sbot_1,
                               m_sbot_2, mbL, mbR, a_b, Q_renorm)\
            + sigmauu_sbottom2(vHiggs, mu, tanb, y_b, g, g_prime, m_sbot_1,
                               m_sbot_2, mbL, mbR, a_b, Q_renorm)\
            + sigmauu_stau1(vHiggs, mu, tanb, y_tau, g, g_prime, m_stau_1,
                            m_stau_2, mtauL, mtauR, a_tau, Q_renorm)\
            + sigmauu_stau2(vHiggs, mu, tanb, y_tau, g, g_prime, m_stau_1,
                            m_stau_2, mtauL, mtauR, a_tau, Q_renorm)\
            + sigmauu_sup_L(g, g_prime, msupL, Q_renorm)\
            + sigmauu_sup_R(g, g_prime, msupR, Q_renorm)\
            + sigmauu_sdown_L(g, g_prime, msdownL, Q_renorm)\
            + sigmauu_sdown_R(g, g_prime, msdownR, Q_renorm)\
            + sigmauu_selec_L(g, g_prime, mselecL, Q_renorm)\
            + sigmauu_selec_R(g, g_prime, mselecR, Q_renorm)\
            + sigmauu_selecSneut(g, g_prime, mselecSneut, Q_renorm)\
            + sigmauu_sstrange_L(g, g_prime, msstrangeL, Q_renorm)\
            + sigmauu_sstrange_R(g, g_prime, msstrangeR, Q_renorm)\
            + sigmauu_scharm_L(g, g_prime, mscharmL, Q_renorm)\
            + sigmauu_scharm_R(g, g_prime, mscharmR, Q_renorm)\
            + sigmauu_smu_L(g, g_prime, msmuL, Q_renorm)\
            + sigmauu_smu_R(g, g_prime, msmuR, Q_renorm)\
            + sigmauu_smuSneut(g, g_prime, msmuSneut, Q_renorm)\
            + sigmauu_neutralino(M1, M2, mu, g, g_prime, vHiggs, tanb, msN1,
                                 Q_renorm)\
            + sigmauu_neutralino(M1, M2, mu, g, g_prime, vHiggs, tanb, msN2,
                                 Q_renorm)\
            + sigmauu_neutralino(M1, M2, mu, g, g_prime, vHiggs, tanb, msN3,
                                 Q_renorm)\
            + sigmauu_neutralino(M1, M2, mu, g, g_prime, vHiggs, tanb, msN4,
                                 Q_renorm)\
            + sigmauu_chargino1(g, M2, vHiggs, tanb, mu, msC1, Q_renorm)\
            + sigmauu_chargino2(g, M2, vHiggs, tanb, mu, msC2, Q_renorm)\
            + sigmauu_h0(g, g_prime, vHiggs, tanb, mHusq, mHdsq, mu, mZ, mh0,
                         Q_renorm)\
            + sigmauu_H0(g, g_prime, vHiggs, tanb, mHusq, mHdsq, mu, mZ, mH0,
                         Q_renorm)\
            + sigmauu_H_pm(g, mH_pm, Q_renorm)\
            + sigmauu_W_pm(g, vHiggs, Q_renorm)\
            + sigmauu_Z0(g, g_prime, vHiggs, Q_renorm)\
            + sigmauu_top(y_t, vHiggs, tanb, Q_renorm)
    return Sigmauunet


#########################
# Sigmadd computation
#########################


def sigmadd_net(vHiggs, mu, tanb, y_t, y_b, y_tau, g, g_prime, m_stop_1,
                m_stop_2, m_sbot_1, m_sbot_2, m_stau_1, m_stau_2, mtL, mtR,
                mbL, mbR, mtauL, mtauR, msupL, msupR, msdownL, msdownR,
                mselecL, mselecR, mselecSneut, msstrangeL, msstrangeR,
                mscharmL, mscharmR, msmuL, msmuR, msmuSneut, msN1, msN2, msN3,
                msN4, msC1, msC2, mZ, mh0, mH0, mHusq, mHdsq, mH_pm, M1, M2,
                a_t, a_b, a_tau, Q_renorm):
    Sigmaddnet = sigmadd_stop1(vHiggs, mu, tanb, y_t, g, g_prime, m_stop_1,
                               m_stop_2, mtL, mtR, a_t, Q_renorm)\
            + sigmadd_stop2(vHiggs, mu, tanb, y_t, g, g_prime, m_stop_1,
                            m_stop_2, mtL, mtR, a_t, Q_renorm)\
            + sigmadd_sbottom1(vHiggs, mu, tanb, y_b, g, g_prime, m_sbot_1,
                               m_sbot_2, mbL, mbR, a_b, Q_renorm)\
            + sigmadd_sbottom2(vHiggs, mu, tanb, y_b, g, g_prime, m_sbot_1,
                               m_sbot_2, mbL, mbR, a_b, Q_renorm)\
            + sigmadd_stau1(vHiggs, mu, tanb, y_tau, g, g_prime, m_stau_1,
                            m_stau_2, mtauL, mtauR, a_tau, Q_renorm)\
            + sigmadd_stau2(vHiggs, mu, tanb, y_tau, g, g_prime, m_stau_1,
                            m_stau_2, mtauL, mtauR, a_tau, Q_renorm)\
            + sigmadd_sup_L(g, g_prime, msupL, Q_renorm)\
            + sigmadd_sup_R(g, g_prime, msupR, Q_renorm)\
            + sigmadd_sdown_L(g, g_prime, msdownL, Q_renorm)\
            + sigmadd_sdown_R(g, g_prime, msdownR, Q_renorm)\
            + sigmadd_selec_L(g, g_prime, mselecL, Q_renorm)\
            + sigmadd_selec_R(g, g_prime, mselecR, Q_renorm)\
            + sigmadd_selecSneut(g, g_prime, mselecSneut, Q_renorm)\
            + sigmadd_sstrange_L(g, g_prime, msstrangeL, Q_renorm)\
            + sigmadd_sstrange_R(g, g_prime, msstrangeR, Q_renorm)\
            + sigmadd_scharm_L(g, g_prime, mscharmL, Q_renorm)\
            + sigmadd_scharm_R(g, g_prime, mscharmR, Q_renorm)\
            + sigmadd_smu_L(g, g_prime, msmuL, Q_renorm)\
            + sigmadd_smu_R(g, g_prime, msmuR, Q_renorm)\
            + sigmadd_smuSneut(g, g_prime, msmuSneut, Q_renorm)\
            + sigmadd_neutralino(M1, M2, mu, g, g_prime, vHiggs, tanb, msN1,
                                 Q_renorm)\
            + sigmadd_neutralino(M1, M2, mu, g, g_prime, vHiggs, tanb, msN2,
                                 Q_renorm)\
            + sigmadd_neutralino(M1, M2, mu, g, g_prime, vHiggs, tanb, msN3,
                                 Q_renorm)\
            + sigmadd_neutralino(M1, M2, mu, g, g_prime, vHiggs, tanb, msN4,
                                 Q_renorm)\
            + sigmadd_chargino1(g, M2, vHiggs, tanb, mu, msC1, Q_renorm)\
            + sigmadd_chargino2(g, M2, vHiggs, tanb, mu, msC2, Q_renorm)\
            + sigmauu_h0(g, g_prime, vHiggs, tanb, mHusq, mHdsq, mu, mZ, mh0,
                         Q_renorm)\
            + sigmauu_H0(g, g_prime, vHiggs, tanb, mHusq, mHdsq, mu, mZ, mH0,
                         Q_renorm)\
            + sigmauu_H_pm(g, mH_pm, Q_renorm)\
            + sigmauu_W_pm(g, vHiggs, Q_renorm)\
            + sigmauu_Z0(g, g_prime, vHiggs, Q_renorm)\
            + sigmadd_bottom(y_b, vHiggs, tanb, Q_renorm)\
            + sigmadd_tau(y_tau, vHiggs, tanb, Q_renorm)
    return Sigmaddnet


#########################
# DEW computation
#########################


def Max_Sigmauu_contrib(vHiggs, mu, tanb, y_t, y_b, y_tau, g, g_prime,
                        m_stop_1, m_stop_2, m_sbot_1, m_sbot_2, m_stau_1,
                        m_stau_2, mtL, mtR, mbL, mbR, mtauL, mtauR, msupL,
                        msupR, msdownL, msdownR, mselecL, mselecR, mselecSneut,
                        msstrangeL, msstrangeR, mscharmL, mscharmR, msmuL,
                        msmuR, msmuSneut, msN1, msN2, msN3, msN4, msC1, msC2,
                        mZ, mh0, mH0, mHusq, mHdsq, mH_pm, M1, M2, a_t, a_b,
                        a_tau, Q_renorm):
    Sigmauuarray = np.array([sigmauu_stop1(vHiggs, mu, tanb, y_t, g, g_prime,
                                           m_stop_1, m_stop_2, mtL, mtR, a_t,
                                           Q_renorm),
                             sigmauu_stop2(vHiggs, mu, tanb, y_t, g, g_prime,
                                           m_stop_1, m_stop_2, mtL, mtR, a_t,
                                           Q_renorm),
                             sigmauu_sbottom1(vHiggs, mu, tanb, y_b, g,
                                              g_prime, m_sbot_1, m_sbot_2, mbL,
                                              mbR, a_b, Q_renorm),
                             sigmauu_sbottom2(vHiggs, mu, tanb, y_b, g,
                                              g_prime, m_sbot_1, m_sbot_2, mbL,
                                              mbR, a_b, Q_renorm),
                             sigmauu_stau1(vHiggs, mu, tanb, y_tau, g, g_prime,
                                           m_stau_1, m_stau_2, mtauL, mtauR,
                                           a_tau, Q_renorm),
                             sigmauu_stau2(vHiggs, mu, tanb, y_tau, g, g_prime,
                                           m_stau_1, m_stau_2, mtauL, mtauR,
                                           a_tau, Q_renorm),
                             sigmauu_sup_L(g, g_prime, msupL, Q_renorm),
                             sigmauu_sup_R(g, g_prime, msupR, Q_renorm),
                             sigmauu_sdown_L(g, g_prime, msdownL, Q_renorm),
                             sigmauu_sdown_R(g, g_prime, msdownR, Q_renorm),
                             sigmauu_selec_L(g, g_prime, mselecL, Q_renorm),
                             sigmauu_selec_R(g, g_prime, mselecR, Q_renorm),
                             sigmauu_selecSneut(g, g_prime, mselecSneut,
                                                Q_renorm),
                             sigmauu_sstrange_L(g, g_prime, msstrangeL,
                                                Q_renorm),
                             sigmauu_sstrange_R(g, g_prime, msstrangeR,
                                                Q_renorm),
                             sigmauu_scharm_L(g, g_prime, mscharmL, Q_renorm),
                             sigmauu_scharm_R(g, g_prime, mscharmR, Q_renorm),
                             sigmauu_smu_L(g, g_prime, msmuL, Q_renorm),
                             sigmauu_smu_R(g, g_prime, msmuR, Q_renorm),
                             sigmauu_smuSneut(g, g_prime, msmuSneut, Q_renorm),
                             sigmauu_neutralino(M1, M2, mu, g, g_prime, vHiggs,
                                                tanb, msN1, Q_renorm),
                             sigmauu_neutralino(M1, M2, mu, g, g_prime, vHiggs,
                                                tanb, msN2, Q_renorm),
                             sigmauu_neutralino(M1, M2, mu, g, g_prime, vHiggs,
                                                tanb, msN3, Q_renorm),
                             sigmauu_neutralino(M1, M2, mu, g, g_prime, vHiggs,
                                                tanb, msN4, Q_renorm),
                             sigmauu_chargino1(g, M2, vHiggs, tanb, mu, msC1,
                                               Q_renorm),
                             sigmauu_chargino2(g, M2, vHiggs, tanb, mu, msC2,
                                               Q_renorm),
                             sigmauu_h0(g, g_prime, vHiggs, tanb, mHusq, mHdsq,
                                        mu, mZ, mh0, Q_renorm),
                             sigmauu_H0(g, g_prime, vHiggs, tanb, mHusq, mHdsq,
                                        mu, mZ, mH0, Q_renorm),
                             sigmauu_H_pm(g, mH_pm, Q_renorm),
                             sigmauu_W_pm(g, vHiggs, Q_renorm),
                             sigmauu_Z0(g, g_prime, vHiggs, Q_renorm),
                             sigmauu_top(y_t, vHiggs, tanb, Q_renorm)])
    myuucontribarray = np.absolute((np.absolute((-1) * Sigmauuarray)
                                    / np.sqrt(1
                                              - (4 * sinsqb(tanb)
                                                 * cossqb(tanb))))
                                   - Sigmauuarray) / 2
    maxuucontrib = np.amax(myuucontribarray)
    return maxuucontrib / 2


def Max_Sigmadd_contrib(vHiggs, mu, tanb, y_t, y_b, y_tau, g, g_prime,
                        m_stop_1, m_stop_2, m_sbot_1, m_sbot_2, m_stau_1,
                        m_stau_2, mtL, mtR, mbL, mbR, mtauL, mtauR, msupL,
                        msupR, msdownL, msdownR, mselecL, mselecR, mselecSneut,
                        msstrangeL, msstrangeR, mscharmL, mscharmR, msmuL,
                        msmuR, msmuSneut, msN1, msN2, msN3, msN4, msC1, msC2,
                        mZ, mh0, mH0, mHusq, mHdsq, mH_pm, M1, M2, a_t, a_b,
                        a_tau, Q_renorm):
    Sigmaddarray = np.array([sigmadd_stop1(vHiggs, mu, tanb, y_t, g, g_prime,
                                           m_stop_1, m_stop_2, mtL, mtR, a_t,
                                           Q_renorm),
                             sigmadd_stop2(vHiggs, mu, tanb, y_t, g, g_prime,
                                           m_stop_1, m_stop_2, mtL, mtR, a_t,
                                           Q_renorm),
                             sigmadd_sbottom1(vHiggs, mu, tanb, y_b, g,
                                              g_prime, m_sbot_1, m_sbot_2, mbL,
                                              mbR, a_b, Q_renorm),
                             sigmadd_sbottom2(vHiggs, mu, tanb, y_b, g,
                                              g_prime, m_sbot_1, m_sbot_2, mbL,
                                              mbR, a_b, Q_renorm),
                             sigmadd_stau1(vHiggs, mu, tanb, y_tau, g, g_prime,
                                           m_stau_1, m_stau_2, mtauL, mtauR,
                                           a_tau, Q_renorm),
                             sigmadd_stau2(vHiggs, mu, tanb, y_tau, g, g_prime,
                                           m_stau_1, m_stau_2, mtauL, mtauR,
                                           a_tau, Q_renorm),
                             sigmadd_sup_L(g, g_prime, msupL, Q_renorm),
                             sigmadd_sup_R(g, g_prime, msupR, Q_renorm),
                             sigmadd_sdown_L(g, g_prime, msdownL, Q_renorm),
                             sigmadd_sdown_R(g, g_prime, msdownR, Q_renorm),
                             sigmadd_selec_L(g, g_prime, mselecL, Q_renorm),
                             sigmadd_selec_R(g, g_prime, mselecR, Q_renorm),
                             sigmadd_selecSneut(g, g_prime, mselecSneut,
                                                Q_renorm),
                             sigmadd_sstrange_L(g, g_prime, msstrangeL,
                                                Q_renorm),
                             sigmadd_sstrange_R(g, g_prime, msstrangeR,
                                                Q_renorm),
                             sigmadd_scharm_L(g, g_prime, mscharmL, Q_renorm),
                             sigmadd_scharm_R(g, g_prime, mscharmR, Q_renorm),
                             sigmadd_smu_L(g, g_prime, msmuL, Q_renorm),
                             sigmadd_smu_R(g, g_prime, msmuR, Q_renorm),
                             sigmadd_smuSneut(g, g_prime, msmuSneut, Q_renorm),
                             sigmadd_neutralino(M1, M2, mu, g, g_prime, vHiggs,
                                                tanb, msN1, Q_renorm),
                             sigmadd_neutralino(M1, M2, mu, g, g_prime, vHiggs,
                                                tanb, msN2, Q_renorm),
                             sigmadd_neutralino(M1, M2, mu, g, g_prime, vHiggs,
                                                tanb, msN3, Q_renorm),
                             sigmadd_neutralino(M1, M2, mu, g, g_prime, vHiggs,
                                                tanb, msN4, Q_renorm),
                             sigmadd_chargino1(g, M2, vHiggs, tanb, mu, msC1,
                                               Q_renorm),
                             sigmadd_chargino2(g, M2, vHiggs, tanb, mu, msC2,
                                               Q_renorm),
                             sigmauu_h0(g, g_prime, vHiggs, tanb, mHusq, mHdsq,
                                        mu, mZ, mh0, Q_renorm),
                             sigmauu_H0(g, g_prime, vHiggs, tanb, mHusq, mHdsq,
                                        mu, mZ, mH0, Q_renorm),
                             sigmauu_H_pm(g, mH_pm, Q_renorm),
                             sigmauu_W_pm(g, vHiggs, Q_renorm),
                             sigmauu_Z0(g, g_prime, vHiggs, Q_renorm),
                             sigmadd_bottom(y_b, vHiggs, tanb, Q_renorm),
                             sigmadd_tau(y_tau, vHiggs, tanb, Q_renorm)])
    myddcontribarray = np.absolute((np.absolute(Sigmaddarray)
                                    / np.sqrt(1
                                              - (4 * sinsqb(tanb)
                                                 * cossqb(tanb))))
                                   - Sigmaddarray) / 2
    maxddcontrib = np.amax(myddcontribarray)
    return maxddcontrib / 2


def DEW(vHiggs, mu, tanb, y_t, y_b, y_tau, g, g_prime, m_stop_1, m_stop_2,
        m_sbot_1, m_sbot_2, m_stau_1, m_stau_2, mtL, mtR, mbL, mbR,
        mtauL, mtauR, msupL, msupR, msdownL, msdownR, mselecL, mselecR,
        mselecSneut, msstrangeL, msstrangeR, mscharmL, mscharmR, msmuL,
        msmuR, msmuSneut, msN1, msN2, msN3, msN4, msC1, msC2, mZ, mh0,
        mH0, mHusq, mHdsq, mH_pm, M1, M2, a_t, a_b, a_tau, Q_renorm):
    cmu = np.absolute((-1) * np.power(mu, 2))
    cHu = np.absolute((np.absolute((-1) * mHusq / np.sqrt(1
                                                          - (4 * sinsqb(tanb)
                                                             * cossqb(tanb)))))
                      - mHusq) / 2
    cHd = np.absolute((np.absolute(mHdsq / np.sqrt(1 - (4 * sinsqb(tanb)
                                                        * cossqb(tanb)))))
                      - mHdsq) / 2
    contribution_array = np.array([cmu, cHu, cHd,
                                   Max_Sigmadd_contrib(vHiggs, mu, tanb, y_t,
                                                       y_b, y_tau, g, g_prime,
                                                       m_stop_1, m_stop_2,
                                                       m_sbot_1, m_sbot_2,
                                                       m_stau_1, m_stau_2, mtL,
                                                       mtR, mbL, mbR, mtauL,
                                                       mtauR, msupL, msupR,
                                                       msdownL, msdownR,
                                                       mselecL, mselecR,
                                                       mselecSneut,
                                                       msstrangeL, msstrangeR,
                                                       mscharmL, mscharmR,
                                                       msmuL, msmuR, msmuSneut,
                                                       msN1, msN2, msN3, msN4,
                                                       msC1, msC2, mZ, mh0,
                                                       mH0, mHusq, mHdsq,
                                                       mH_pm, M1, M2, a_t, a_b,
                                                       a_tau, Q_renorm),
                                   Max_Sigmauu_contrib(vHiggs, mu, tanb, y_t,
                                                       y_b, y_tau, g, g_prime,
                                                       m_stop_1, m_stop_2,
                                                       m_sbot_1, m_sbot_2,
                                                       m_stau_1, m_stau_2, mtL,
                                                       mtR, mbL, mbR, mtauL,
                                                       mtauR, msupL, msupR,
                                                       msdownL, msdownR,
                                                       mselecL, mselecR,
                                                       mselecSneut,
                                                       msstrangeL, msstrangeR,
                                                       mscharmL, mscharmR,
                                                       msmuL, msmuR, msmuSneut,
                                                       msN1, msN2, msN3, msN4,
                                                       msC1, msC2, mZ, mh0,
                                                       mH0, mHusq, mHdsq,
                                                       mH_pm, M1, M2, a_t, a_b,
                                                       a_tau, Q_renorm)])
    mydew = (np.amax(contribution_array)) / (np.power(mZ, 2) / 2)
    return mydew


#########################
# SLHA input
#########################


direc = input('Enter the full directory for your SLHA file: ')
d = pyslha.read(direc)
resultant_dew = DEW(d.blocks['HMIX'][3],  # Higgs VEV(Q) MSSM DRbar
                    d.blocks['HMIX'][1],  # mu(Q) MSSM DRbar
                    d.blocks['HMIX'][2],  # tanb(Q) MSSSM DRbar
                    d.blocks['YU'][3, 3],  # y_t(Q) MSSM DRbar
                    d.blocks['YD'][3, 3],  # y_b(Q) MSSM DRbar
                    d.blocks['YE'][3, 3],  # y_tau(Q) MSSM DRbar
                    d.blocks['GAUGE'][2],  # g'(Q) MSSM DRbar
                    d.blocks['GAUGE'][1],  # g(Q) MSSM DRbar
                    d.blocks['MASS'][1000006],  # m_stop_1
                    d.blocks['MASS'][2000006],  # m_stop_2
                    d.blocks['MASS'][1000005],  # m_sbot_1
                    d.blocks['MASS'][2000005],  # m_sbot_2
                    d.blocks['MASS'][1000015],  # m_stau_1
                    d.blocks['MASS'][2000015],  # m_stau_2
                    d.blocks['MSOFT'][43],  # m_~Q3_L(Q) MSSM DRbar
                    d.blocks['MSOFT'][46],  # m_stop_R(Q) MSSM DRbar
                    d.blocks['MSOFT'][43],  # m_~Q3_L(Q) MSSM DRbar
                    d.blocks['MSOFT'][49],  # m_sbot_R(Q) MSSM DRbar
                    d.blocks['MSOFT'][33],  # m_stau_L(Q) MSSM DRbar
                    d.blocks['MSOFT'][36],  # m_stau_R(Q) MSSM DRbar
                    d.blocks['MSOFT'][41],  # m_~Q1_L(Q) MSSM DRbar
                    d.blocks['MSOFT'][44],  # m_sup_R(Q) MSSM DRbar
                    d.blocks['MSOFT'][41],  # m_~Q1_L(Q) MSSM DRbar
                    d.blocks['MSOFT'][47],  # m_sdown_R(Q) MSSM DRbar
                    d.blocks['MSOFT'][31],  # m_selec_L(Q) MSSM DRbar
                    d.blocks['MSOFT'][34],  # m_selec_R(Q) MSSM DRbar
                    d.blocks['MASS'][1000012],  # m_selecSneutrino_L
                    d.blocks['MSOFT'][42],  # m_~Q2_L(Q) MSSM DRbar
                    d.blocks['MSOFT'][48],  # m_sstrange_R(Q) MSSM DRbar
                    d.blocks['MSOFT'][42],  # m_~Q2_L(Q) MSSM DRbar
                    d.blocks['MSOFT'][45],  # m_scharm_R(Q) MSSM DRbar
                    d.blocks['MSOFT'][32],  # m_smu_L(Q) MSSM DRbar
                    d.blocks['MSOFT'][35],  # m_smu_R(Q) MSSM DRbar
                    d.blocks['MASS'][1000014],  # m_smuSneutrino_L
                    d.blocks['MASS'][1000022],  # m_Neutralino_1
                    d.blocks['MASS'][1000023],  # m_Neutralino_2
                    d.blocks['MASS'][1000025],  # m_Neutralino_3
                    d.blocks['MASS'][1000035],  # m_Neutralino_4
                    d.blocks['MASS'][1000024],  # m_Chargino_1
                    d.blocks['MASS'][1000037],  # m_Chargino_2
                    d.blocks['SMINPUTS'][4],  # m_Z
                    d.blocks['MASS'][25],  # m_h0
                    d.blocks['MASS'][35],  # m_H0
                    d.blocks['MSOFT'][22],  # m_Hu^2(Q) MSSM DRbar
                    d.blocks['MSOFT'][21],  # m_Hd^2(Q) MSSM DRbar
                    d.blocks['MASS'][37],  # m_H_+-
                    d.blocks['MSOFT'][1],  # M_1
                    d.blocks['MSOFT'][2],  # M_2
                    d.blocks['AU'][3, 3] * d.blocks['YU'][3, 3],  # a_t
                    d.blocks['AD'][3, 3] * d.blocks['YD'][3, 3],  # a_b
                    d.blocks['AE'][3, 3] * d.blocks['YE'][3, 3],  # a_tau
                    np.sqrt(d.blocks['MASS'][1000006]
                            * d.blocks['MASS'][2000006]))
# Q_renorm = sqrt(m_stop_1 * m_stop_2) above

print('\nGiven the submitted SLHA file, your value for the electroweak'
      + ' naturalness measure, Delta_EW, is: ' + str(resultant_dew))
