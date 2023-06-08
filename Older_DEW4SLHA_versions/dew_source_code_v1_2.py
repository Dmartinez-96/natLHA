#!/usr/bin/python
"""
Compute naturalness measure Delta_EW and contributions to DEW.

Author: Dakotah Martinez
"""

import numpy as np
from scipy.special import spence
import pyslha
import time


if __name__ == "__main__":
    userContinue = True
    while userContinue:
        # Mass relations: #


        def m_w_sq():
            """Return W boson squared mass."""
            my_mw_sq = (np.power(g_EW, 2) / 2) * np.power(vHiggs, 2)
            return my_mw_sq



        def mz_q_sq():
            """Return m_Z(Q)^2."""
            mzqsq = np.power(vHiggs, 2) * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)
            return mzqsq



        def ma_0sq():
            """Return A_0 squared mass."""
            my_ma0_sq = 2 * np.power(np.abs(muQ), 2) + mHusq + mHdsq
            return my_ma0_sq


        # Fundamental equations: #


        def logfunc(mass):
            """
            Return F = m^2 * (ln(m^2 / Q^2) - 1), where input mass term is linear.

            Parameters
            ----------
            mass : Input mass.

            """
            myf = np.power(mass, 2) * (np.log((np.power(mass, 2)) / (Q_renorm_sq)) - 1)
            return myf


        def logfunc2(masssq):
            """
            Return F = m^2 * (ln(m^2 / Q^2) - 1), where input mass term is quadratic.

            Parameters
            ----------
            masssq : Input mass squared.

            """
            myf2 = masssq * (np.log((masssq) / (Q_renorm_sq)) - 1)
            return myf2


        def sinsqb():
            """Return sin^2(beta)."""
            mysinsqb = np.power(tanb, 2) / (1 + np.power(tanb, 2))
            return mysinsqb


        def cossqb():
            """Return cos^2(beta)."""
            mycossqb = 1 / (1 + np.power(tanb, 2))
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
            mysinsqthetaw = np.power(tan_theta_w(), 2) / (1 + np.power(tan_theta_w(), 2))
            return mysinsqthetaw


        def cos2b():
            """Return cos(2*beta)."""
            mycos2b = cossqb() - sinsqb()
            return mycos2b


        def sin2b():
            """Return sin(2*beta)."""
            mysin2b = 2.0 / ((1 / tanb) + tanb)
            return mysin2b


        def gz_sq():
            """Return g_Z^2 = (g^2 + g'^2) / 8."""
            mygzsq = (np.power(g_EW, 2) + np.power(g_pr, 2)) / 8
            return mygzsq


        # Stop squarks: #


        def sigmauu_stop1():
            """Return one-loop correction Sigma_u^u(stop_1)."""
            delta_stop = ((1 / 2) - (4 / 3) * sinsq_theta_w()) * 2 \
                         * (((np.power(mtL, 2) - np.power(mtR, 2)) / 2)
                            + (mz_q_sq()) * cos2b() * ((1 / 4) - (2 / 3) * sinsq_theta_w()))
            stop_num = np.power(a_t, 2) - (2 * gz_sq() * delta_stop)
            sigmauu_stop_1 = (3 / (16 * (np.power(np.pi, 2)))) * logfunc2(m_stop_1sq) \
                             * (np.power(y_t, 2) - gz_sq()
                                - (stop_num / (m_stop_2sq - m_stop_1sq)))
            return sigmauu_stop_1



        def sigmadd_stop1():
            """Return one-loop correction Sigma_d^d(stop_1)."""
            delta_stop = ((1 / 2) - (4 / 3) * sinsq_theta_w()) * 2 \
                         * (((np.power(mtL, 2) - np.power(mtR, 2)) / 2)
                            + (mz_q_sq()) * cos2b() * ((1 / 4) - (2 / 3) * sinsq_theta_w()))
            stop_num = np.power(y_t, 2) * np.power(muQ, 2) \
                       + (2 * gz_sq() * delta_stop)
            sigmadd_stop_1 = (3 / (16 * np.power(np.pi, 2))) * logfunc2(m_stop_1sq) \
                             * (gz_sq() - (stop_num
                                           / (m_stop_2sq - m_stop_1sq)))
            return sigmadd_stop_1



        def sigmauu_stop2():
            """Return one-loop correction Sigma_u^u(stop_2)."""
            delta_stop = ((1 / 2) - (4 / 3) * sinsq_theta_w()) * 2 \
                         * (((np.power(mtL, 2) - np.power(mtR, 2)) / 2)
                            + (mz_q_sq()) * cos2b() * ((1 / 4) - (2 / 3) * sinsq_theta_w()))
            stop_num = np.power(a_t, 2) - (2 * gz_sq() * delta_stop)
            sigmauu_stop_2 = (3 / (16 * (np.power(np.pi, 2)))) * logfunc2(m_stop_2sq) \
                             * (np.power(y_t, 2) - gz_sq()
                                + (stop_num
                                   / (m_stop_2sq - m_stop_1sq)))
            return sigmauu_stop_2



        def sigmadd_stop2():
            """Return one-loop correction Sigma_d^d(stop_2)."""
            delta_stop = ((1 / 2) - (4 / 3) * sinsq_theta_w()) * 2 \
                         * (((np.power(mtL, 2) - np.power(mtR, 2)) / 2)
                            + (mz_q_sq()) * cos2b() * ((1 / 4) - (2 / 3) * sinsq_theta_w()))
            stop_num = np.power(y_t, 2) * np.power(muQ, 2) \
                       + (2 * gz_sq() * delta_stop)
            sigmadd_stop_2 = (3 / (16 * np.power(np.pi, 2))) * logfunc2(m_stop_2sq) \
                             * (gz_sq() + (stop_num
                                           / (m_stop_2sq - m_stop_1sq)))
            return sigmadd_stop_2


        # Sbottom squarks: #


        def sigmauu_sbottom1():
            """Return one-loop correction Sigma_u^u(sbottom_1)."""
            delta_sbot = ((1 / 2) - (2 / 3) * sinsq_theta_w()) * 2 \
                         * (((np.power(mbL, 2) - np.power(mbR, 2)) / 2)
                            - (mz_q_sq()) * cos2b() * ((1 / 4) - (1 / 3) * sinsq_theta_w()))
            sbot_num = np.power(a_b, 2) - (2 * gz_sq() * delta_sbot)
            sigmauu_sbot = (3 / (16 * np.power(np.pi, 2))) * logfunc2(m_sbot_1sq) \
                           * (np.power(y_b, 2) - gz_sq()
                              - (sbot_num / (m_sbot_2sq - m_sbot_1sq)))
            return sigmauu_sbot



        def sigmauu_sbottom2():
            """Return one-loop correction Sigma_u^u(sbottom_2)."""
            delta_sbot = ((1 / 2) - (2 / 3) * sinsq_theta_w()) * 2 \
                         * (((np.power(mbL, 2) - np.power(mbR, 2)) / 2)
                            - (mz_q_sq()) * cos2b() * ((1 / 4) - (1 / 3) * sinsq_theta_w()))
            sbot_num = np.power(a_b, 2) - (2 * gz_sq() * delta_sbot)
            sigmauu_sbot = (3 / (16 * np.power(np.pi, 2))) * logfunc2(m_sbot_2sq) \
                           * (np.power(y_b, 2) - gz_sq()
                              + (sbot_num / (m_sbot_2sq - m_sbot_1sq)))
            return sigmauu_sbot



        def sigmadd_sbottom1():
            """Return one-loop correction Sigma_d^d(sbottom_1)."""
            delta_sbot = ((1 / 2) - (2 / 3) * sinsq_theta_w()) * 2 \
                         * (((np.power(mbL, 2) - np.power(mbR, 2)) / 2)
                            - (mz_q_sq()) * cos2b() * ((1 / 4) - (1 / 3) * sinsq_theta_w()))
            sbot_num = np.power(y_b, 2) * np.power(muQ, 2) \
                       + (2 * gz_sq() * delta_sbot)
            sigmadd_sbot = (3 / (16 * np.power(np.pi, 2))) * logfunc2(m_sbot_1sq) \
                           * (gz_sq()
                              - (sbot_num / (m_sbot_2sq - m_sbot_1sq)))
            return sigmadd_sbot



        def sigmadd_sbottom2():
            """Return one-loop correction Sigma_d^d(sbottom_2)."""
            delta_sbot = ((1 / 2) - (2 / 3) * sinsq_theta_w()) * 2 \
                         * (((np.power(mbL, 2) - np.power(mbR, 2)) / 2)
                            - (mz_q_sq()) * cos2b() * ((1 / 4) - (1 / 3) * sinsq_theta_w()))
            sbot_num = np.power(y_b, 2) * np.power(muQ, 2) \
                       + (2 * gz_sq() * delta_sbot)
            sigmadd_sbot = (3 / (16 * np.power(np.pi, 2))) * logfunc2(m_sbot_2sq) \
                           * (gz_sq()
                              + (sbot_num / (m_sbot_2sq - m_sbot_1sq)))
            return sigmadd_sbot


        # Stau sleptons: #


        def sigmauu_stau1():
            """Return one-loop correction Sigma_u^u(stau_1)."""
            delta_stau = ((1 / 2) - 2 * sinsq_theta_w()) * 2 \
                         * (((np.power(mtauL, 2) - np.power(mtauR, 2)) / 2)
                            - (mz_q_sq()) * cos2b() * ((1 / 4) - sinsq_theta_w()))
            stau_num = np.power(a_tau, 2) - (2 * gz_sq() * delta_stau)
            sigmauu_stau = (1 / (16 * np.power(np.pi, 2))) * logfunc2(m_stau_1sq) \
                           * (np.power(y_tau, 2) - gz_sq()
                              - (stau_num / (m_stau_2sq - m_stau_1sq)))
            return sigmauu_stau



        def sigmauu_stau2():
            """Return one-loop correction Sigma_u^u(stau_2)."""
            delta_stau = ((1 / 2) - 2 * sinsq_theta_w()) * 2 \
                         * (((np.power(mtauL, 2) - np.power(mtauR, 2)) / 2)
                            - (mz_q_sq()) * cos2b() * ((1 / 4) - sinsq_theta_w()))
            stau_num = np.power(a_tau, 2) - (2 * gz_sq() * delta_stau)
            sigmauu_stau = (1 / (16 * np.power(np.pi, 2))) * logfunc2(m_stau_2sq) \
                           * (np.power(y_tau, 2) - gz_sq()
                              + (stau_num / (m_stau_2sq - m_stau_1sq))
                              )
            return sigmauu_stau



        def sigmadd_stau1():
            """Return one-loop correction Sigma_d^d(stau_1)."""
            delta_stau = ((1 / 2) - 2 * sinsq_theta_w()) * 2 \
                         * (((np.power(mtauL, 2) - np.power(mtauR, 2)) / 2)
                            - (mz_q_sq()) * cos2b() * ((1 / 4) - sinsq_theta_w()))
            stau_num = np.power(y_tau, 2) * np.power(muQ, 2) \
                       + (2 * gz_sq() * delta_stau)
            sigmauu_stau = (1 / (16 * np.power(np.pi, 2))) * logfunc2(m_stau_1sq) \
                           * (gz_sq()
                              - (stau_num / (m_stau_2sq - m_stau_1sq)))
            return sigmauu_stau



        def sigmadd_stau2():
            """Return one-loop correction Sigma_d^d(stau_2)."""
            delta_stau = ((1 / 2) - 2 * sinsq_theta_w()) * 2 \
                         * (((np.power(mtauL, 2) - np.power(mtauR, 2)) / 2)
                            - (mz_q_sq()) * cos2b() * ((1 / 4) - sinsq_theta_w()))
            stau_num = np.power(y_tau, 2) * np.power(muQ, 2) \
                       + (2 * gz_sq() * delta_stau)
            sigmauu_stau = (1 / (16 * np.power(np.pi, 2))) * logfunc2(m_stau_2sq) \
                           * (gz_sq()
                              + (stau_num / (m_stau_2sq - m_stau_1sq)))
            return sigmauu_stau


        # Sfermions, 1st gen: #


        def sigmauu_sup_l():
            """Return one-loop correction Sigma_u^u(sup_L)."""
            sigmauusup_l = ((-3) / (4 * np.power(np.pi, 2))) \
                           * ((1 / 2) - (2 / 3) * sinsq_theta_w()) * gz_sq() \
                           * logfunc(msupL)
            return sigmauusup_l



        def sigmauu_sup_r():
            """Return one-loop correction Sigma_u^u(sup_R)."""
            sigmauusup_r = ((-3) / (4 * np.power(np.pi, 2))) \
                           * ((2 / 3) * sinsq_theta_w()) * gz_sq() * logfunc(msupR)
            return sigmauusup_r



        def sigmauu_sdown_l():
            """Return one-loop correction Sigma_u^u(sdown_L)."""
            sigmauusdown_l = ((-3) / (4 * np.power(np.pi, 2))) * logfunc(msdownL) \
                             * (((-1) / 2) + (1 / 3) * sinsq_theta_w()) * gz_sq()
            return sigmauusdown_l



        def sigmauu_sdown_r():
            """Return one-loop correction Sigma_u^u(sdown_R)."""
            sigmauusdown_r = ((-3) / (4 * np.power(np.pi, 2))) * logfunc(msdownR) \
                             * (((-1) / 3) * sinsq_theta_w()) * gz_sq()
            return sigmauusdown_r



        def sigmauu_selec_l():
            """Return one-loop correction Sigma_u^u(selectron_L)."""
            sigmauuselec_l = ((-1) / (4 * np.power(np.pi, 2))) * logfunc(mselecL) \
                             * (((-1) / 2) + sinsq_theta_w()) * gz_sq()
            return sigmauuselec_l



        def sigmauu_selec_r():
            """Return one-loop correction Sigma_u^u(selectron_R)."""
            sigmauuselec_r = ((-1) / (4 * np.power(np.pi, 2))) * logfunc(mselecR) \
                             * ((-1) * sinsq_theta_w()) * gz_sq()
            return sigmauuselec_r



        def sigmauu_sel_neut():
            """Return one-loop correction Sigma_u^u(selectron neutrino)."""
            sigmauuselec_sneut = ((-1) / (4 * np.power(np.pi, 2))) * (1 / 2) \
                                 * logfunc2(mselecneutsq) * gz_sq()
            return sigmauuselec_sneut



        def sigmadd_sup_l():
            """Return one-loop correction Sigma_d^d(sup_L)."""
            sigmaddsup_l = (3 / (4 * np.power(np.pi, 2))) \
                           * ((1 / 2) - (2 / 3) * sinsq_theta_w()) * gz_sq() \
                           * logfunc(msupL)
            return sigmaddsup_l



        def sigmadd_sup_r():
            """Return one-loop correction Sigma_d^d(sup_R)."""
            sigmaddsup_r = (3 / (4 * np.power(np.pi, 2))) \
                           * ((2 / 3) * sinsq_theta_w()) * gz_sq() * logfunc(msupR)
            return sigmaddsup_r



        def sigmadd_sdown_l():
            """Return one-loop correction Sigma_d^d(sdown_L)."""
            sigmaddsdown_l = (3 / (4 * np.power(np.pi, 2))) * logfunc(msdownL) \
                             * (((-1) / 2) + (1 / 3) * sinsq_theta_w()) * gz_sq()
            return sigmaddsdown_l



        def sigmadd_sdown_r():
            """Return one-loop correction Sigma_d^d(sdown_R)."""
            sigmaddsdown_r = (3 / (4 * np.power(np.pi, 2))) * logfunc(msdownR) \
                             * (((-1) / 3) * sinsq_theta_w()) * gz_sq()
            return sigmaddsdown_r



        def sigmadd_selec_l():
            """Return one-loop correction Sigma_d^d(selectron_L)."""
            sigmaddselec_l = (1 / (4 * np.power(np.pi, 2))) * logfunc(mselecL) \
                             * (((-1) / 2) + sinsq_theta_w()) * gz_sq()
            return sigmaddselec_l



        def sigmadd_selec_r():
            """Return one-loop correction Sigma_d^d(selectron_R)."""
            sigmaddselec_r = (1 / (4 * np.power(np.pi, 2))) * logfunc(mselecR) \
                             * ((-1) * sinsq_theta_w()) * gz_sq()
            return sigmaddselec_r



        def sigmadd_sel_neut():
            """Return one-loop correction Sigma_d^d(selectron neutrino)."""
            sigmaddselec_sneut = (1 / (4 * np.power(np.pi, 2))) * (1 / 2) \
                                 * logfunc2(mselecneutsq) * gz_sq()
            return sigmaddselec_sneut


        # Sfermions, 2nd gen: #


        def sigmauu_sstrange_l():
            """Return one-loop correction Sigma_u^u(sstrange_L)."""
            sigmauusstrange_l = ((-3) / (4 * np.power(np.pi, 2))) * gz_sq() \
                                * (((-1) / 2) + (1 / 3) * sinsq_theta_w()) * logfunc(msstrangeL)
            return sigmauusstrange_l



        def sigmauu_sstrange_r():
            """Return one-loop correction Sigma_u^u(sstrange_R)."""
            sigmauusstrange_r = ((-3) / (4 * np.power(np.pi, 2))) * gz_sq() \
                                * (((-1) / 3) * sinsq_theta_w()) * logfunc(msstrangeR)
            return sigmauusstrange_r



        def sigmauu_scharm_l():
            """Return one-loop correction Sigma_u^u(scharm_L)."""
            sigmauuscharm_l = ((-3) / (4 * np.power(np.pi, 2))) * gz_sq() \
                              * ((1 / 2) - (2 / 3) * sinsq_theta_w()) * logfunc(mscharmL)
            return sigmauuscharm_l



        def sigmauu_scharm_r():
            """Return one-loop correction Sigma_u^u(scharm_R)."""
            sigmauuscharm_r = ((-3) / (4 * np.power(np.pi, 2))) * gz_sq() \
                              * ((2 / 3) * sinsq_theta_w()) * logfunc(mscharmR)
            return sigmauuscharm_r



        def sigmauu_smu_l():
            """Return one-loop correction Sigma_u^u(smu_L)."""
            sigmauusmu_l = ((-1) / (4 * np.power(np.pi, 2))) * gz_sq() \
                           * (((-1) / 2) + sinsq_theta_w()) * logfunc(msmuL)
            return sigmauusmu_l



        def sigmauu_smu_r():
            """Return one-loop correction Sigma_u^u(smu_R)."""
            sigmauusmu_r = ((-1) / (4 * np.power(np.pi, 2))) * gz_sq() \
                           * ((-1) * sinsq_theta_w()) * logfunc(msmuR)
            return sigmauusmu_r



        def sigmauu_smu_sneut():
            """Return one-loop correction Sigma_u^u(smuon neutrino)."""
            sigmauusmu_sneut = ((-1) / (4 * np.power(np.pi, 2))) * gz_sq() \
                               * (1 / 2) * logfunc2(msmuneutsq)
            return sigmauusmu_sneut



        def sigmadd_sstrange_l():
            """Return one-loop correction Sigma_d^d(sstrange_L)."""
            sigmaddsstrange_l = (3 / (4 * np.power(np.pi, 2))) * gz_sq() \
                                * (((-1) / 2) + (1 / 3) * sinsq_theta_w()) * logfunc(msstrangeL)
            return sigmaddsstrange_l



        def sigmadd_sstrange_r():
            """Return one-loop correction Sigma_d^d(sstrange_R)."""
            sigmaddsstrange_r = (3 / (4 * np.power(np.pi, 2))) * gz_sq() \
                                * (((-1) / 3) * sinsq_theta_w()) * logfunc(msstrangeR)
            return sigmaddsstrange_r



        def sigmadd_scharm_l():
            """Return one-loop correction Sigma_d^d(scharm_L)."""
            sigmaddscharm_l = (3 / (4 * np.power(np.pi, 2))) * gz_sq() \
                              * ((1 / 2) - (2 / 3) * sinsq_theta_w()) * logfunc(mscharmL)
            return sigmaddscharm_l



        def sigmadd_scharm_r():
            """Return one-loop correction Sigma_d^d(scharm_R)."""
            sigmaddscharm_r = (3 / (4 * np.power(np.pi, 2))) * gz_sq() \
                              * ((2 / 3) * sinsq_theta_w()) * logfunc(mscharmR)
            return sigmaddscharm_r



        def sigmadd_smu_l():
            """Return one-loop correction Sigma_d^d(smu_L)."""
            sigmaddsmu_l = (1 / (4 * np.power(np.pi, 2))) * gz_sq() \
                           * (((-1) / 2) + sinsq_theta_w()) * logfunc(msmuL)
            return sigmaddsmu_l



        def sigmadd_smu_r():
            """Return one-loop correction Sigma_d^d(smu_R)."""
            sigmaddsmu_r = (1 / (4 * np.power(np.pi, 2))) * gz_sq() \
                           * ((-1) * sinsq_theta_w()) * logfunc(msmuR)
            return sigmaddsmu_r



        def sigmadd_smu_sneut():
            """Return one-loop correction Sigma_d^d(smuon neutrino)."""
            sigmaddsmu_sneut = (1 / (4 * np.power(np.pi, 2))) * (1 / 2) \
                               * gz_sq() * logfunc2(msmuneutsq)
            return sigmaddsmu_sneut


        # Third gen tau neutrino contributions: #
        def sigmauu_stau_sneut():
            """Return one-loop correction Sigma_u^u(stau neutrino)."""
            sigmauustau_sneut = (-1 / (4 * np.power(np.pi, 2))) * (1 / 2) * gz_sq() * logfunc2(mstauneutsq)
            return sigmauustau_sneut


        def sigmadd_stau_sneut():
            """Return one-loop correction Sigma_d^d(smuon neutrino)."""
            sigmaddstau_sneut = (1 / (4 * np.power(np.pi, 2))) * (1 / 2) * gz_sq() * logfunc2(mstauneutsq)
            return sigmaddstau_sneut


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
                          + (np.power(g_pr, 2) * M_1 * muQ)) / (tanb)) \
                        - ((np.power((g_EW * M_1), 2)) + (np.power((g_pr * M_2), 2))
                           + ((np.power(g_EW, 2) + np.power(g_pr, 2)) * (np.power(muQ, 2)))
                           + (np.power((np.power(g_EW, 2) + np.power(g_pr, 2)), 2) / 2)
                           * np.power(vHiggs, 2))
            linterm = (((-1) * muQ) * ((np.power(g_EW, 2) * M_2
                                        * (np.power(M_1, 2) + np.power(muQ, 2)))
                                       + np.power(g_pr, 2) * M_1
                                       * (np.power(M_2, 2) + np.power(muQ, 2)))
                       / tanb) \
                      + ((np.power((np.power(g_EW, 2) * M_1 + np.power(g_pr, 2) * M_2), 2)
                          / 2) * np.power(vHiggs, 2)) + (np.power(muQ, 2)
                                                         * ((np.power((g_EW * M_1), 2))
                                                            + (np.power((g_pr * M_2), 2)))) \
                      + (np.power((np.power(g_EW, 2) + np.power(g_pr, 2)), 2)
                         * np.power((vHiggs * muQ), 2) * cossqb())
            constterm = (M_1 * M_2 * ((np.power(g_EW, 2) * M_1)
                                      + (np.power(g_pr, 2) * M_2))
                         * np.power(muQ, 3) * (1 / tanb)) \
                        - (np.power((np.power(g_EW, 2) * M_1 + np.power(g_pr, 2) * M_2), 2)
                           * np.power(vHiggs, 2) * np.power(muQ, 2) * cossqb())
            mynum = (cubicterm * np.power(msn, 6)) + (quadrterm * np.power(msn, 4)) \
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
                                                             * muQ)) * (tanb)) \
                        - ((np.power((g_EW * M_1), 2)) + (np.power((g_pr * M_2), 2))
                           + ((np.power(g_EW, 2) + np.power(g_pr, 2)) * (np.power(muQ, 2)))
                           + (np.power((np.power(g_EW, 2) + np.power(g_pr, 2)), 2) / 2)
                           * np.power(vHiggs, 2))
            linterm = (((-1) * muQ) * ((np.power(g_EW, 2) * M_2
                                        * (np.power(M_1, 2) + np.power(muQ, 2)))
                                       + np.power(g_pr, 2) * M_1
                                       * (np.power(M_2, 2) + np.power(muQ, 2)))
                       * tanb) \
                      + ((np.power((np.power(g_EW, 2) * M_1 + np.power(g_pr, 2) * M_2), 2)
                          / 2) * np.power(vHiggs, 2)) \
                      + (np.power(muQ, 2) * ((np.power((g_EW * M_1), 2))
                                             + np.power(g_pr, 2) * np.power(M_2, 2))) \
                      + (np.power((np.power(g_EW, 2) + np.power(g_pr, 2)), 2)
                         * np.power((vHiggs * muQ), 2) * sinsqb())
            constterm = (M_1 * M_2 * (np.power(g_EW, 2) * M_1 + (np.power(g_pr, 2)
                                                                 * M_2))
                         * np.power(muQ, 3) * tanb) \
                        - (np.power((np.power(g_EW, 2) * M_1 + np.power(g_pr, 2) * M_2), 2)
                           * np.power((vHiggs * muQ), 2) * sinsqb())
            mynum = (cubicterm * np.power(msn, 6)) + (quadrterm * np.power(msn, 4)) \
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
                       * np.power((np.power(g_EW, 2) + np.power(g_pr, 2)), 2) / 2) \
                      + (np.power(vHiggs, 2)
                         * (2 * ((np.power((g_EW * M_1), 2)) + (np.power((g_pr * M_2), 2))
                                 + ((np.power(g_EW, 2) + np.power(g_pr, 2))
                                    * np.power(muQ, 2))
                                 - (muQ * (np.power(g_pr, 2) * M_1 + np.power(g_EW, 2) * M_2)
                                    * 2 * np.sqrt(sinsqb()) * np.sqrt(cossqb()))))) \
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
                            )) \
                        + (np.power(vHiggs, 2) * 2 * muQ
                           * ((np.sqrt(cossqb()) * np.sqrt(sinsqb()))
                              * (np.power(g_EW, 2) * M_2
                                 * (np.power(M_1, 2) + np.power(muQ, 2))
                                 + (np.power(g_pr, 2) * M_1
                                    * (np.power(M_2, 2) + np.power(muQ, 2)))))) \
                        - ((2 * np.power((M_2 * M_1 * muQ), 2))
                           + (np.power(muQ, 4) * (np.power(M_1, 2) + np.power(M_2, 2))))
            mydenom = 4 * np.power(msn, 6) + quadrterm * np.power(msn, 4) \
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
                                     / neutralino_deriv_denom(msn)) \
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
                                     / neutralino_deriv_denom(msn)) \
                                  * logfunc(msn)
            return sigma_dd_neutralino


        # Charginos: #


        def sigmauu_chargino1():
            """Return one-loop correction Sigma_u^u(chargino_1)."""
            chargino_num = ((-2) * m_w_sq() * cos2b()) + np.power(M_2, 2) \
                           + np.power(muQ, 2)
            chargino_den = msC2sq - msC1sq
            sigma_uu_chargino1 = -1 * (np.power(g_EW, 2) / (16 * np.power(np.pi, 2))) \
                                 * (1 - (chargino_num / chargino_den)) * logfunc2(msC1sq)
            return sigma_uu_chargino1



        def sigmauu_chargino2():
            """Return one-loop correction Sigma_u^u(chargino_2)."""
            chargino_num = ((-2) * m_w_sq() * cos2b()) + np.power(M_2, 2) \
                           + np.power(muQ, 2)
            chargino_den = msC2sq - msC1sq
            sigma_uu_chargino2 = -1 * (np.power(g_EW, 2) / (16 * np.power(np.pi, 2))) \
                                 * (1 + (chargino_num / chargino_den)) * logfunc2(msC2sq)
            return sigma_uu_chargino2



        def sigmadd_chargino1():
            """Return one-loop correction Sigma_d^d(chargino_1)."""
            chargino_num = (2 * m_w_sq() * cos2b()) + np.power(M_2, 2) \
                           + np.power(muQ, 2)
            chargino_den = msC2sq - msC1sq
            sigma_dd_chargino1 = -1 * (np.power(g_EW, 2) / (16 * np.power(np.pi, 2))) \
                                 * (1 - (chargino_num / chargino_den)) * logfunc2(msC1sq)
            return sigma_dd_chargino1



        def sigmadd_chargino2():
            """Return one-loop correction Sigma_d^d(chargino_2)."""
            chargino_num = (2 * m_w_sq() * cos2b()) + np.power(M_2, 2) \
                           + np.power(muQ, 2)
            chargino_den = msC2sq - msC1sq
            sigma_dd_chargino2 = -1 * (np.power(g_EW, 2) / (16 * np.power(np.pi, 2))) \
                                 * (1 + (chargino_num / chargino_den)) * logfunc2(msC2sq)
            return sigma_dd_chargino2


        # Higgs bosons (sigmauu = sigmadd here): #


        def sigmauu_h0():
            """Return one-loop correction Sigma_u^u(h_0) (lighter neutral Higgs)."""
            mynum = mz_q_sq() + (mA0sq * (1 + (4 * cos2b())
                                          + (2 * np.power(cos2b(), 2))))
            myden = mH0sq - mh0sq
            sigma_uu_h0 = (gz_sq() / (16 * np.power(np.pi, 2))) \
                          * (1 - (mynum / myden)) * logfunc2(mh0sq)
            return sigma_uu_h0



        def sigmadd_h0():
            """Return one-loop correction Sigma_d^d(h_0) (lighter neutral Higgs)."""
            mynum = mz_q_sq() + (mA0sq * (1 - (4 * cos2b())
                                          + (2 * np.power(cos2b(), 2))))
            myden = mH0sq - mh0sq
            sigma_dd_h0 = (gz_sq() / (16 * np.power(np.pi, 2))) \
                          * (1 - (mynum / myden)) * logfunc2(mh0sq)
            return sigma_dd_h0



        def sigmauu_heavy_h0():
            """Return one-loop correction Sigma_u^u(H_0) (heavier neutr. Higgs)."""
            mynum = mz_q_sq() + (mA0sq * (1 + (4 * cos2b()) + (2 * np.power(cos2b(), 2))))
            myden = mH0sq - mh0sq
            sigma_uu_heavy_h0 = (gz_sq() / (16 * np.power(np.pi, 2))) * (1 + (mynum / myden)) * logfunc2(mH0sq)
            return sigma_uu_heavy_h0



        def sigmadd_heavy_h0():
            """Return one-loop correction Sigma_d^d(H_0) (heavier neutr. Higgs)."""
            mynum = mz_q_sq() + (mA0sq * (1 - (4 * cos2b()) + (2 * np.power(cos2b(), 2))))
            myden = mH0sq - mh0sq
            sigma_dd_heavy_h0 = (gz_sq() / (16 * np.power(np.pi, 2))) * (1 + (mynum / myden)) * logfunc2(mH0sq)
            return sigma_dd_heavy_h0



        def sigmauu_h_pm():
            """Return one-loop correction Sigma_u,d^u,d(H_{+-})."""
            sigma_uu_h_pm = (np.power((g_EW / np.pi), 2) / (32)) * logfunc2(mH_pmsq)
            return sigma_uu_h_pm


        # Weak bosons (sigmauu = sigmadd here): #


        def sigmauu_w_pm():
            """Return one-loop correction Sigma_u,d^u,d(W_{+-})."""
            sigma_uu_w_pm = (3 * np.power((g_EW / np.pi), 2) / (32)) \
                            * logfunc(np.sqrt(m_w_sq()))
            return sigma_uu_w_pm



        def sigmauu_z0():
            """Return one-loop correction Sigma_u,d^u,d(Z_0)."""
            sigma_uu_z0 = (3 * (np.power(g_EW, 2) + np.power(g_pr, 2))
                           / (64 * np.power(np.pi, 2))) * logfunc(np.sqrt(mz_q_sq()))
            return sigma_uu_z0


        # SM fermions (sigmadd_t = sigmauu_b = sigmauu_tau = 0): #


        def sigmauu_top():
            """Return one-loop correction Sigma_u^u(top)."""
            sigma_uu_top = ((-1) * np.power((y_t / np.pi), 2) / (16)) * logfunc(mymt)
            return sigma_uu_top



        def sigmadd_bottom():
            """Return one-loop correction Sigma_d^d(bottom)."""
            sigma_dd_bottom = (-1 * np.power((y_b / np.pi), 2) / (16)) * logfunc(mymb)
            return sigma_dd_bottom



        def sigmadd_tau():
            """Return one-loop correction Sigma_d^d(tau)."""
            sigma_dd_tau = (-1 * np.power((y_tau / np.pi), 2) / (16)) * logfunc(mymtau)
            return sigma_dd_tau

        ##### Two loop calculations #####
        # Include two-loop O(alpha_t alpha_s) corrections from Dedes, Slavich
        # paper arxiv: hep-ph 0212132


        def sigmadd_2loop():
            def Deltafunc(x,y,z):
                mydelta = np.power(x, 2) + np.power(y, 2) + np.power(z, 2)\
                    - (2 * ((x * y) + (x * z) + (y * z)))
                return mydelta

            def Phifunc(x,y,z):
                if(x / z < 1 and y / z < 1):
                    myu = x / z
                    myv = y / z
                    mylambda = np.sqrt(np.power((1 - myu - myv), 2) - (4 * myu * myv))
                    myxp = 0.5 * (1 + myu - myv - mylambda)
                    myxm = 0.5 * (1 - myu + myv - mylambda)
                    myphi = (1 / mylambda) * ((2 * np.log(myxp) * np.log(myxm))
                                              - (np.log(myu) * np.log(myv))
                                              - (2 * (spence(1 - myxp) + spence(1 - myxm)))
                                              + (np.power(np.pi, 2) / 3))
                elif(x / z > 1 and y / z < 1):
                    myu = z / x
                    myv = y / x
                    mylambda = np.sqrt(np.power((1 - myu - myv), 2) - (4 * myu * myv))
                    myxp = 0.5 * (1 + myu - myv - mylambda)
                    myxm = 0.5 * (1 - myu + myv - mylambda)
                    myphi = (z / x) * (1 / mylambda) * ((2 * np.log(myxp) * np.log(myxm))
                                                        - (np.log(myu) * np.log(myv))
                                                        - (2 * (spence(1 - myxp) + spence(1 - myxm)))
                                                        + (np.power(np.pi, 2) / 3))
                elif(x/z > 1 and y/ z > 1 and x > y):
                    myu = z / x
                    myv = y / x
                    mylambda = np.sqrt(np.power((1 - myu - myv), 2) - (4 * myu * myv))
                    myxp = 0.5 * (1 + myu - myv - mylambda)
                    myxm = 0.5 * (1 - myu + myv - mylambda)
                    myphi = (z / x) * (1 / mylambda) * ((2 * np.log(myxp) * np.log(myxm))
                                                        - (np.log(myu) * np.log(myv))
                                                        - (2 * (spence(1 - myxp) + spence(1 - myxm)))
                                                        + (np.power(np.pi, 2) / 3))
                elif(x / z < 1 and y / z > 1):
                    myu = z / y
                    myv = x / y
                    mylambda = np.sqrt(np.power((1 - myu - myv), 2) - (4 * myu * myv))
                    myxp = 0.5 * (1 + myu - myv - mylambda)
                    myxm = 0.5 * (1 - myu + myv - mylambda)
                    myphi = (z / y) * (1 / mylambda) * ((2 * np.log(myxp) * np.log(myxm))
                                                        - (np.log(myu) * np.log(myv))
                                                        - (2 * (spence(1 - myxp) + spence(1 - myxm)))
                                                        + (np.power(np.pi, 2) / 3))
                elif (x / z > 1 and y / z > 1 and y>x):
                    myu = z / y
                    myv = x / y
                    mylambda = np.sqrt(np.power((1 - myu - myv), 2) - (4 * myu * myv))
                    myxp = 0.5 * (1 + myu - myv - mylambda)
                    myxm = 0.5 * (1 - myu + myv - mylambda)
                    myphi = (z / y) * (1 / mylambda) * ((2 * np.log(myxp) * np.log(myxm))
                                                        - (np.log(myu) * np.log(myv))
                                                        - (2 * (spence(1 - myxp) + spence(1 - myxm)))
                                                        + (np.power(np.pi, 2) / 3))
                return myphi

            mst1sq = m_stop_1sq
            mst2sq = m_stop_2sq
            s2theta = (2 * mymt * (A_t + (muQ / tanb)))\
                / (mst1sq-mst2sq)
            s2sqtheta = np.power(s2theta, 2)
            c2sqtheta = 1 - s2sqtheta
            mglsq = np.power(mgl, 2)
            myunits = np.power(g_s, 2) * 4\
                / np.power((16 * np.power(np.pi, 2)), 2)
            myF = myunits\
                * ((4 * mgl * mymt / s2theta) * (1 + 4 * c2sqtheta)
                   - (((2 * (mst1sq - mst2sq))
                      + (4 * mgl * mymt / s2theta))
                      * np.log(mglsq / Q_renorm_sq)
                      * np.log(mymtsq / Q_renorm_sq))
                   - (2 * (4 - s2sqtheta)
                      * (mst1sq - mst2sq))
                   + ((((4 * mst1sq * mst2sq)
                        - s2sqtheta * np.power((mst1sq + mst2sq), 2))
                       / (mst1sq - mst2sq)) * (np.log(mst1sq / Q_renorm_sq))
                      * (np.log(mst2sq / Q_renorm_sq)))
                   + ((((4 * (mglsq + mymtsq + (2 * mst1sq)))
                       - (s2sqtheta * ((3 * mst1sq) + mst2sq))
                       - ((16 * c2sqtheta * mgl * mymt * mst1sq)
                          / (s2theta * (mst1sq - mst2sq)))
                       - (4 * s2theta * mgl * mymt))
                       * np.log(mst1sq / Q_renorm_sq))
                      + ((mst1sq / (mst1sq - mst2sq))
                         * ((s2sqtheta * (mst1sq + mst2sq))
                            - ((4 * mst1sq) - (2 * mst2sq)))
                         * np.power(np.log(mst1sq / Q_renorm_sq), 2))
                      + (2 * (mst1sq - mglsq - mymtsq
                              + (mgl * mymt * s2theta)
                              + ((2 * c2sqtheta * mgl * mymt * mst1sq)
                                 / (s2theta * (mst1sq - mst2sq))))
                         * np.log(mglsq * mymtsq
                                  / (np.power(Q_renorm_sq, 2)))
                         * np.log(mst1sq / Q_renorm_sq))
                      + (((4 * mgl * mymt * c2sqtheta * (mymtsq - mglsq))
                          / (s2theta * (mst1sq - mst2sq)))
                         * np.log(mymtsq / mglsq)
                         * np.log(mst1sq / Q_renorm_sq))
                      + ((((4 * mglsq * mymtsq) + (2 * Deltafunc(mglsq, mymtsq, mst1sq))) / mst1sq)
                         - (((2 * mgl * mymt * s2theta) / mst1sq)
                            * (mglsq + mymtsq - mst1sq))
                         + ((4 * c2sqtheta * mgl * mymt * Deltafunc(mglsq, mymtsq, mst1sq))
                            / (s2theta * mst1sq * (mst1sq - mst2sq)))) * Phifunc(mglsq, mymtsq, mst1sq))
                   - ((((4 * (mglsq + mymtsq + (2 * mst2sq)))
                       - (s2sqtheta * ((3 * mst2sq) + mst1sq))
                       - ((16 * c2sqtheta * mgl * mymt * mst2sq)
                          / (((-1) * s2theta) * (mst2sq - mst1sq)))
                       - ((-4) * s2theta * mgl * mymt))
                       * np.log(mst2sq / Q_renorm_sq))
                      + ((mst2sq / (mst2sq - mst1sq))
                         * ((s2sqtheta * (mst2sq + mst1sq))
                            - ((4 * mst2sq) - (2 * mst1sq)))
                         * np.power(np.log(mst2sq / Q_renorm_sq), 2))
                      + (2 * (mst2sq - mglsq - mymtsq
                              - (mgl * mymt * s2theta)
                              + ((2 * c2sqtheta * mgl * mymt * mst2sq)
                                 / (s2theta * (mst1sq - mst2sq))))
                         * np.log(mglsq * mymtsq
                                  / (np.power(Q_renorm_sq, 2)))
                         * np.log(mst2sq / Q_renorm_sq))
                      + (((4 * mgl * mymt * c2sqtheta * (mymtsq - mglsq))
                          / (s2theta * (mst1sq - mst2sq)))
                         * np.log(mymtsq / mglsq)
                         * np.log(mst2sq / Q_renorm_sq))
                      + ((((4 * mglsq * mymtsq) + (2 * Deltafunc(mglsq, mymtsq, mst2sq))) / mst2sq)
                         - ((((-2) * mgl * mymt * s2theta) / mst2sq)
                            * (mglsq + mymtsq - mst2sq))
                         + ((4 * c2sqtheta * mgl * mymt * Deltafunc(mglsq, mymtsq, mst2sq))
                            / (s2theta * mst2sq * (mst1sq - mst2sq)))) * Phifunc(mglsq, mymtsq, mst2sq)))
            mysigmadd_2loop = (mymt * muQ * (1 / tanb) * s2theta * myF)\
                / (np.power((vHiggs), 2) * cossqb())
            return mysigmadd_2loop



        def sigmauu_2loop():
            def Deltafunc(x, y, z):
                mydelta = np.power(x, 2) + np.power(y, 2) + np.power(z, 2) \
                          - (2 * ((x * y) + (x * z) + (y * z)))
                return mydelta

            def Phifunc(x, y, z):
                if(x / z < 1 and y / z < 1):
                    myu = x / z
                    myv = y / z
                    mylambda = np.sqrt(np.power((1 - myu - myv), 2) - (4 * myu * myv))
                    myxp = 0.5 * (1 + myu - myv - mylambda)
                    myxm = 0.5 * (1 - myu + myv - mylambda)
                    myphi = (1 / mylambda) * ((2 * np.log(myxp) * np.log(myxm))
                                              - (np.log(myu) * np.log(myv))
                                              - (2 * (spence(1 - myxp) + spence(1 - myxm)))
                                              + (np.power(np.pi, 2) / 3))
                elif(x / z > 1 and y / z < 1):
                    myu = z / x
                    myv = y / x
                    mylambda = np.sqrt(np.power((1 - myu - myv), 2) - (4 * myu * myv))
                    myxp = 0.5 * (1 + myu - myv - mylambda)
                    myxm = 0.5 * (1 - myu + myv - mylambda)
                    myphi = (z / x) * (1 / mylambda) * ((2 * np.log(myxp) * np.log(myxm))
                                                        - (np.log(myu) * np.log(myv))
                                                        - (2 * (spence(1 - myxp) + spence(1 - myxm)))
                                                        + (np.power(np.pi, 2) / 3))
                elif(x/z > 1 and y/ z > 1 and x > y):
                    myu = z / x
                    myv = y / x
                    mylambda = np.sqrt(np.power((1 - myu - myv), 2) - (4 * myu * myv))
                    myxp = 0.5 * (1 + myu - myv - mylambda)
                    myxm = 0.5 * (1 - myu + myv - mylambda)
                    myphi = (z / x) * (1 / mylambda) * ((2 * np.log(myxp) * np.log(myxm))
                                                        - (np.log(myu) * np.log(myv))
                                                        - (2 * (spence(1 - myxp) + spence(1 - myxm)))
                                                        + (np.power(np.pi, 2) / 3))
                elif(x / z < 1 and y / z > 1):
                    myu = z / y
                    myv = x / y
                    mylambda = np.sqrt(np.power((1 - myu - myv), 2) - (4 * myu * myv))
                    myxp = 0.5 * (1 + myu - myv - mylambda)
                    myxm = 0.5 * (1 - myu + myv - mylambda)
                    myphi = (z / y) * (1 / mylambda) * ((2 * np.log(myxp) * np.log(myxm))
                                                        - (np.log(myu) * np.log(myv))
                                                        - (2 * (spence(1 - myxp) + spence(1 - myxm)))
                                                        + (np.power(np.pi, 2) / 3))
                elif (x / z > 1 and y / z > 1 and y > x):
                    myu = z / y
                    myv = x / y
                    mylambda = np.sqrt(np.power((1 - myu - myv), 2) - (4 * myu * myv))
                    myxp = 0.5 * (1 + myu - myv - mylambda)
                    myxm = 0.5 * (1 - myu + myv - mylambda)
                    myphi = (z / y) * (1 / mylambda) * ((2 * np.log(myxp) * np.log(myxm))
                                                        - (np.log(myu) * np.log(myv))
                                                        - (2 * (spence(1 - myxp) + spence(1 - myxm)))
                                                        + (np.power(np.pi, 2) / 3))
                return myphi

            mst1sq = m_stop_1sq
            mst2sq = m_stop_2sq
            s2theta = (2 * mymt * (A_t + (muQ / tanb)))\
                / (mst1sq - mst2sq)
            s2sqtheta = np.power(s2theta, 2)
            c2sqtheta = 1 - s2sqtheta
            mglsq = np.power(mgl, 2)
            myunits = np.power(g_s, 2) * 4 \
                      / np.power((16 * np.power(np.pi, 2)), 2)
            myF = myunits\
                  * ((4 * mgl * mymt / s2theta) * (1 + 4 * c2sqtheta)
                     - (((2 * (mst1sq - mst2sq))
                         + (4 * mgl * mymt / s2theta))
                        * np.log(mglsq / Q_renorm_sq)
                        * np.log(mymtsq / Q_renorm_sq))
                     - (2 * (4 - s2sqtheta)
                        * (mst1sq - mst2sq))
                     + ((((4 * mst1sq * mst2sq)
                          - s2sqtheta * np.power((mst1sq + mst2sq), 2))
                         / (mst1sq - mst2sq)) * (np.log(mst1sq / Q_renorm_sq))
                        * (np.log(mst2sq / Q_renorm_sq)))
                     + ((((4 * (mglsq + mymtsq + (2 * mst1sq)))
                          - (s2sqtheta * ((3 * mst1sq) + mst2sq))
                          - ((16 * c2sqtheta * mgl * mymt * mst1sq)
                             / (s2theta * (mst1sq - mst2sq)))
                          - (4 * s2theta * mgl * mymt))
                         * np.log(mst1sq / Q_renorm_sq))
                        + ((mst1sq / (mst1sq - mst2sq))
                           * ((s2sqtheta * (mst1sq + mst2sq))
                              - ((4 * mst1sq) - (2 * mst2sq)))
                           * np.power(np.log(mst1sq / Q_renorm_sq), 2))
                        + (2 * (mst1sq - mglsq - mymtsq
                                + (mgl * mymt * s2theta)
                                + ((2 * c2sqtheta * mgl * mymt * mst1sq)
                                   / (s2theta * (mst1sq - mst2sq))))
                           * np.log(mglsq * mymtsq
                                    / (np.power(Q_renorm_sq, 2)))
                           * np.log(mst1sq / Q_renorm_sq))
                        + (((4 * mgl * mymt * c2sqtheta * (mymtsq - mglsq))
                            / (s2theta * (mst1sq - mst2sq)))
                           * np.log(mymtsq / mglsq)
                           * np.log(mst1sq / Q_renorm_sq))
                        + ((((4 * mglsq * mymtsq) + (2 * Deltafunc(mglsq, mymtsq, mst1sq))) / mst1sq)
                           - (((2 * mgl * mymt * s2theta) / mst1sq)
                              * (mglsq + mymtsq - mst1sq))
                           + ((4 * c2sqtheta * mgl * mymt * Deltafunc(mglsq, mymtsq, mst1sq))
                              / (s2theta * mst1sq * (mst1sq - mst2sq)))) * Phifunc(mglsq, mymtsq, mst1sq))
                     - ((((4 * (mglsq + mymtsq + (2 * mst2sq)))
                          - (s2sqtheta * ((3 * mst2sq) + mst1sq))
                          - ((16 * c2sqtheta * mgl * mymt * mst2sq)
                             / (((-1) * s2theta) * (mst2sq - mst1sq)))
                          - ((-4) * s2theta * mgl * mymt))
                         * np.log(mst2sq / Q_renorm_sq))
                        + ((mst2sq / (mst2sq - mst1sq))
                           * ((s2sqtheta * (mst2sq + mst1sq))
                              - ((4 * mst2sq) - (2 * mst1sq)))
                           * np.power(np.log(mst2sq / Q_renorm_sq), 2))
                        + (2 * (mst2sq - mglsq - mymtsq
                                - (mgl * mymt * s2theta)
                                + ((2 * c2sqtheta * mgl * mymt * mst2sq)
                                   / (s2theta * (mst1sq - mst2sq))))
                           * np.log(mglsq * mymtsq
                                    / (np.power(Q_renorm_sq, 2)))
                           * np.log(mst2sq / Q_renorm_sq))
                        + (((4 * mgl * mymt * c2sqtheta * (mymtsq - mglsq))
                            / (s2theta * (mst1sq - mst2sq)))
                           * np.log(mymtsq / mglsq)
                           * np.log(mst2sq / Q_renorm_sq))
                        + ((((4 * mglsq * mymtsq) + (2 * Deltafunc(mglsq, mymtsq, mst2sq))) / mst2sq)
                           - ((((-2) * mgl * mymt * s2theta) / mst2sq)
                              * (mglsq + mymtsq - mst2sq))
                           + ((4 * c2sqtheta * mgl * mymt * Deltafunc(mglsq, mymtsq, mst2sq))
                              / (s2theta * mst2sq * (mst1sq - mst2sq)))) * Phifunc(mglsq, mymtsq, mst2sq)))
            myG = myunits\
                  * ((5 * mgl * s2theta * (mst1sq - mst2sq) / mymt)
                     - (10 * (mst1sq + mst2sq - (2 * mymtsq)))
                     - (4 * mglsq) + ((12 * mymtsq) * (np.power(np.log(mymtsq / Q_renorm_sq), 2)
                                                       - (2 * np.log(mymtsq / Q_renorm_sq))))
                     + (((4 * mglsq) - ((mgl * s2theta / mymt) * (mst1sq - mst2sq)))
                        * np.log(mglsq / Q_renorm_sq) * np.log(mymtsq / Q_renorm_sq))
                     + (s2sqtheta * (mst1sq + mst2sq) * np.log(mst1sq / Q_renorm_sq) * np.log(mst2sq / Q_renorm_sq))
                     + (((4 * (mglsq + mymtsq + (2 * mst1sq)))
                         + (s2sqtheta * (mst1sq - mst2sq))
                         - ((4 * mgl * s2theta / mymt) * (mymtsq + mst1sq))) * np.log(mst1sq / Q_renorm_sq)
                        + (((mgl * s2theta * ((5 * mymtsq) - mglsq + mst1sq) / mymt)
                            - (2 * (mglsq + 2 * mymtsq))) * np.log(mymtsq / Q_renorm_sq)
                           * np.log(mst1sq / Q_renorm_sq))
                        + (((mgl * s2theta * (mglsq - mymtsq + mst1sq) / mymt) - (2 * mglsq))
                           * np.log(mglsq / Q_renorm_sq) * np.log(mst1sq / Q_renorm_sq))
                        - ((2 + s2sqtheta) * mst1sq * np.power(np.log(mst1sq / Q_renorm_sq), 2))
                        + ((2 * mglsq * (mglsq + mymtsq - mst1sq - (2 * mgl * mymt * s2theta)) / mst1sq)
                           + ((mgl * s2theta / (mymt * mst1sq)) * Deltafunc(mglsq, mymtsq, mst1sq)))
                        * Phifunc(mglsq, mymtsq, mst1sq))
                     + (((4 * (mglsq + mymtsq + (2 * mst2sq)))
                         + (s2sqtheta * (mst2sq - mst1sq))
                         - (((-4) * mgl * s2theta / mymt) * (mymtsq + mst2sq))) * np.log(mst2sq / Q_renorm_sq)
                        + ((((-1) * mgl * s2theta * ((5 * mymtsq) - mglsq + mst2sq) / mymt)
                            - (2 * (mglsq + 2 * mymtsq))) * np.log(mymtsq / Q_renorm_sq)
                           * np.log(mst2sq / Q_renorm_sq))
                        + ((((-1) * mgl * s2theta * (mglsq - mymtsq + mst2sq) / mymt) - (2 * mglsq))
                           * np.log(mglsq / Q_renorm_sq) * np.log(mst2sq / Q_renorm_sq))
                        - ((2 + s2sqtheta) * mst2sq * np.power(np.log(mst2sq / Q_renorm_sq), 2))
                        + ((2 * mglsq * (mglsq + mymtsq - mst2sq + (2 * mgl * mymt * s2theta)) / mst2sq)
                           + ((mgl * (-1) * s2theta / (mymt * mst2sq)) * Deltafunc(mglsq, mymtsq, mst2sq)))
                        * Phifunc(mglsq, mymtsq, mst2sq)))
            mysigmauu_2loop = (mymt * A_t * s2theta * myF
                               + 2 * np.power(mymt, 2) * myG)\
                / (np.power((vHiggs), 2) * sinsqb())
            return mysigmauu_2loop


        # DEW contribution computation: #

        def dew_funcu(inp):
            """
            Compute individual one-loop DEW contributions from Sigma_u^u.

            Parameters
            ----------
            inp : One-loop correction or Higgs to be inputted into the DEW function.

            """
            mycontribuu = np.abs(((-1) * inp * (np.power(tanb, 2))) / ((np.power(tanb, 2)) - 1))
            return mycontribuu


        def dew_funcd(inp):
            """
            Compute individual one-loop DEW contributions from Sigma_d^d.

            Parameters
            ----------
            inp : One-loop correction or Higgs to be inputted into the DEW function.

            """
            mycontribdd = np.abs((inp) / ((np.power(tanb, 2)) - 1))
            return mycontribdd


        # SLHA input and definition of variables from SLHA file: #
        fileCheck = True
        while fileCheck:
            try:
                direc = input('Enter the full directory for your SLHA file: ')
                d = pyslha.read(direc)
                fileCheck = False
            except FileNotFoundError:
                print("The input file cannot be found.\n")
                print("Please try checking your spelling and try again.\n")
                fileCheck = True
        # Set up parameters for computations
        mZ = 91.1876 # This is the value in our universe, not a general value for multiverse scans.
        [vHiggs, muQ, tanb, y_t] = [d.blocks['HMIX'][3] / np.sqrt(2),
                                    d.blocks['HMIX'][1], d.blocks['HMIX'][2],
                                    d.blocks['YU'][3, 3]]
        beta = np.arctan(tanb)
        [y_b, y_tau, g_pr] = [d.blocks['YD'][3, 3], d.blocks['YE'][3, 3], d.blocks['GAUGE'][2]]
        g_EW = d.blocks['GAUGE'][1]
        g_s = d.blocks['GAUGE'][3]
        [mtL, mtR, mbL] = [d.blocks['MSOFT'][43], d.blocks['MSOFT'][46], d.blocks['MSOFT'][43]]
        thetastop = np.arccos(d.blocks['STOPMIX'][1, 1])
        [mbR, mtauL, mtauR] = [d.blocks['MSOFT'][49], d.blocks['MSOFT'][33], d.blocks['MSOFT'][36]]
        [mtLsq, mtRsq, mbLsq] = [np.power(mtL, 2), np.power(mtR, 2), np.power(mbL, 2)]
        [mbRsq, mtauLsq, mtauRsq] = [np.power(mbR, 2), np.power(mtauL, 2), np.power(mtauR, 2)]
        [msupL, msupR] = [d.blocks['MSOFT'][41], d.blocks['MSOFT'][44]]
        [msdownL, msdownR] = [d.blocks['MSOFT'][41], d.blocks['MSOFT'][47]]
        [mselecL, mselecR] = [d.blocks['MSOFT'][31], d.blocks['MSOFT'][34]]
        [msstrangeL, msstrangeR] = [d.blocks['MSOFT'][42], d.blocks['MSOFT'][48]]
        [mscharmL, mscharmR] = [d.blocks['MSOFT'][42], d.blocks['MSOFT'][45]]
        [msmuL, msmuR] = [d.blocks['MSOFT'][32], d.blocks['MSOFT'][35]]
        [mselecLsq, msmuLsq] = [np.power(mselecL, 2), np.power(msmuL, 2)]
        mgl = d.blocks['MSOFT'][3]
        [mHusq, mHdsq] = [d.blocks['MSOFT'][22], d.blocks['MSOFT'][21]]
        [M_1, M_2] = [d.blocks['MSOFT'][1], d.blocks['MSOFT'][2]]
        [A_t, A_b, A_tau] = [d.blocks['AU'][3, 3], d.blocks['AD'][3, 3], d.blocks['AE'][3, 3]]
        [a_t, a_b] = [d.blocks['AU'][3, 3] * y_t, d.blocks['AD'][3, 3] * y_b]
        [a_tau, mA0sq] = [d.blocks['AE'][3, 3] * y_tau, d.blocks['HMIX'][4]]

        mymt = y_t * v_higgs_u()
        mymtsq = np.power(mymt, 2)
        mymb = y_b * v_higgs_d()
        mymbsq = np.power(mymb, 2)
        mymtau = y_tau * v_higgs_d()
        mymtausq = np.power(mymtau, 2)

        mselecneutsq = mselecLsq + 0.5 * mz_q_sq() * cos2b()
        msmuneutsq = msmuLsq + 0.5 * mz_q_sq() * cos2b()
        mstauneutsq = mtauLsq + 0.5 * mz_q_sq() * cos2b()

        mH_pmsq = mA0sq + m_w_sq()

        DeltauL = (0.5 - ((2.0 / 3.0) * sinsq_theta_w())) * cos2b() * mz_q_sq()
        DeltauR = (((2.0 / 3.0) * sinsq_theta_w())) * cos2b() * mz_q_sq()
        DeltadL = (-0.5 + ((1.0 / 3.0) * sinsq_theta_w())) * cos2b() * mz_q_sq()
        DeltadR = (-1.0 / 3.0) * sinsq_theta_w() * cos2b() * mz_q_sq()
        DeltaeL = (-0.5 + sinsq_theta_w()) * cos2b() * mz_q_sq()
        DeltaeR = (-1.0 * sinsq_theta_w()) * cos2b() * mz_q_sq()

        m_stop_1sq = (0.5) * ((2 * mymtsq) + mtLsq + mtRsq + DeltauL + DeltauR - np.sqrt(np.power((mtLsq - mtRsq + DeltauL - DeltauR), 2) + (4 * np.power((a_t * v_higgs_u() - v_higgs_d() * y_t * muQ), 2))))
        m_stop_2sq = (0.5) * ((2 * mymtsq) + mtLsq + mtRsq + DeltauL + DeltauR + np.sqrt(np.power((mtLsq - mtRsq + DeltauL - DeltauR), 2) + (4 * np.power((a_t * v_higgs_u() - v_higgs_d() * y_t * muQ), 2))))

        m_sbot_1sq = (0.5) * ((2 * mymbsq) + mbLsq + mbRsq + DeltadL + DeltadR - np.sqrt(np.power((mbLsq - mbRsq + DeltadL - DeltadR), 2) + (4 * np.power((a_b * v_higgs_d() - v_higgs_u() * y_b * muQ), 2))))
        m_sbot_2sq = (0.5) * ((2 * mymbsq) + mbLsq + mbRsq + DeltadL + DeltadR + np.sqrt(np.power((mbLsq - mbRsq + DeltadL - DeltadR), 2) + (4 * np.power((a_b * v_higgs_d() - v_higgs_u() * y_b * muQ), 2))))

        m_stau_1sq = (0.5) * ((2 * mymtausq) + mtauLsq + mtauRsq + DeltaeL + DeltaeR - np.sqrt(np.power((mtauLsq - mtauRsq + DeltaeL - DeltaeR), 2) + (4 * np.power((a_tau * v_higgs_d() - v_higgs_u() * y_tau * muQ), 2))))
        m_stau_2sq = (0.5) * ((2 * mymtausq) + mtauLsq + mtauRsq + DeltaeL + DeltaeR + np.sqrt(np.power((mtauLsq - mtauRsq + DeltaeL - DeltaeR), 2) + (4 * np.power((a_tau * v_higgs_d() - v_higgs_u() * y_tau * muQ), 2))))

        msC1sq = (1.0 / 2.0) * ((np.abs(M_2) ** 2) + (np.abs(muQ) ** 2) + (2.0 * m_w_sq()) - np.sqrt(np.power(((np.abs(M_2) ** 2) + (np.abs(muQ) ** 2) + (2.0 * m_w_sq())), 2) - (4.0 * np.power(np.abs((muQ * M_2) - (m_w_sq() * sin2b())), 2))))
        msC2sq = (1.0 / 2.0) * ((np.abs(M_2) ** 2) + (np.abs(muQ) ** 2) + (2.0 * m_w_sq()) + np.sqrt(np.power(((np.abs(M_2) ** 2) + (np.abs(muQ) ** 2) + (2.0 * m_w_sq())), 2) - (4.0 * np.power(np.abs((muQ * M_2) - (m_w_sq() * sin2b())), 2))))

        neutralino_mass_matrix = np.array([[M_1, 0, -g_pr * v_higgs_d(), g_pr * v_higgs_u()],[0, M_2, g_EW * v_higgs_d(), -g_EW * v_higgs_u()],[-g_pr * v_higgs_d(), g_EW * v_higgs_d(), 0, -muQ],[g_pr * v_higgs_u(), -g_EW * v_higgs_u(), -muQ, 0]])
        my_neut_eigvals, my_neut_eigvecs = np.linalg.eig(neutralino_mass_matrix)
        sorted_eigvals = sorted(my_neut_eigvals, key=abs)
        msN1 = sorted_eigvals[0]
        msN2 = sorted_eigvals[1]
        msN3 = sorted_eigvals[2]
        msN4 = sorted_eigvals[3]


        mh0sq = (1.0 / 2.0) * (mA0sq + mz_q_sq() - np.sqrt(np.power((mA0sq - mz_q_sq()), 2) + (4.0 * mz_q_sq() * mA0sq * (np.power(sin2b(), 2)))))
        mH0sq = (1.0 / 2.0) * (mA0sq + mz_q_sq() + np.sqrt(np.power((mA0sq - mz_q_sq()), 2) + (4.0 * mz_q_sq() * mA0sq * (np.power(sin2b(), 2)))))
        [Q_renorm_sq, halfmzsq] = [np.sqrt(np.abs(m_stop_1sq)) * np.sqrt(np.abs(m_stop_2sq)), np.power(mZ, 2) / 2]
        [cmu, chu, chd] = [np.abs(np.power(muQ, 2)), dew_funcu(mHusq), dew_funcd(mHdsq)]
        # Check SLHA input scale values and if the max value exceeds > 2.0 * sqrt(mt1 * mt2) (pole masses)
        # This is because PySLHA (the program used to read in SLHA files, arXiv:1305.4194) automatically
        # selects the largest scale value printed in the SLHA file. By default, this is typically on the scale of
        # 1 to 2 times the geometric mean of the stop masses, but if a user changes these defaults, the radiative
        # corrections to mZ are no longer minimized and results may be significantly different. A warning is printed
        # if too large of a scale is detected.
        SLHA_scale = float(str(d.blocks['GAUGE'])[str(d.blocks['GAUGE']).find('Q=')+2:str(d.blocks['GAUGE']).find(')')])
        mean_stop_mass = np.sign(m_stop_1sq * m_stop_2sq) * np.sqrt(np.sqrt(np.abs(m_stop_1sq * m_stop_2sq)))
        # Simple check for possible tachyonic results
        if mean_stop_mass < 0:
            neg_mean_stop_mass_check = 1
        elif mean_stop_mass > 0:
            neg_mean_stop_mass_check = 0
        # Compare SLHA scale and mean stop mass
        if(SLHA_scale > (2.0 * np.abs(mean_stop_mass))):
            SLHA_scale_check = 1
        else:
            SLHA_scale_check = 0

        if neg_mean_stop_mass_check == 1:
            print('WARNING!')
            print('WARNING!')
            print('WARNING!')
            print('WARNING! Possible negative running of stop masses in file: mt1^2: ' + str(m_stop_1sq) + ', mt2^2: ' + str(m_stop_2sq))
            print('WARNING!')
            print('WARNING!')
            print('WARNING!')
        if SLHA_scale_check == 1:
            print('WARNING!')
            print('WARNING!')
            print('WARNING!')
            print('WARNING! The maximum grid scale in your SLHA file, Q='+str(SLHA_scale)+', is more than 2 times the magnitude of the mean of the stop pole masses, sqrt(abs(mt1*mt2))='+str(np.abs(mean_stop_mass)) + '.')
            print('Results may not be accurate because the radiative corrections may not be minimized at this scale.')
            print('WARNING!')
            print('WARNING!')
            print('WARNING!')


        # Calculate other DEW contributions and compare magnitudes
        contribs = np.array([cmu, chu, chd, dew_funcd(sigmadd_stop1()), dew_funcd(sigmadd_stop2()),
                             dew_funcd(sigmadd_sbottom1()), dew_funcd(sigmadd_sbottom2()), dew_funcd(sigmadd_stau1()),
                             dew_funcd(sigmadd_stau2()),
                             dew_funcd(sigmadd_sup_l() + sigmadd_sup_r() + sigmadd_sdown_l() + sigmadd_sdown_r()
                                       + sigmadd_selec_l() + sigmadd_selec_r() + sigmadd_sel_neut()),
                             dew_funcd(sigmadd_sstrange_l() + sigmadd_sstrange_r() + sigmadd_scharm_l()
                                       + sigmadd_scharm_r() + sigmadd_smu_l() + sigmadd_smu_r() + sigmadd_smu_sneut()),
                             dew_funcd(sigmadd_neutralino(msN1)), dew_funcd(sigmadd_neutralino(msN2)),
                             dew_funcd(sigmadd_neutralino(msN3)), dew_funcd(sigmadd_neutralino(msN4)),
                             dew_funcd(sigmadd_chargino1()), dew_funcd(sigmadd_chargino2()), dew_funcd(sigmadd_h0()),
                             dew_funcd(sigmadd_heavy_h0()), dew_funcd(sigmauu_h_pm()), dew_funcd(sigmauu_w_pm()),
                             dew_funcd(sigmauu_z0()), dew_funcd(sigmadd_bottom()), dew_funcd(sigmadd_tau()),
                             dew_funcu(sigmauu_stop1()), dew_funcu(sigmauu_stop2()), dew_funcu(sigmauu_sbottom1()),
                             dew_funcu(sigmauu_sbottom2()), dew_funcu(sigmauu_stau1()), dew_funcu(sigmauu_stau2()),
                             dew_funcu(sigmauu_sup_l() + sigmauu_sup_r() + sigmauu_sdown_l() + sigmauu_sdown_r()
                                       + sigmauu_selec_l() + sigmauu_selec_r() + sigmauu_sel_neut()),
                             dew_funcu(sigmauu_sstrange_l() + sigmauu_sstrange_r() + sigmauu_scharm_l()
                                       + sigmauu_scharm_r() + sigmauu_smu_l() + sigmauu_smu_r() + sigmauu_smu_sneut()),
                             dew_funcu(sigmauu_neutralino(msN1)), dew_funcu(sigmauu_neutralino(msN2)),
                             dew_funcu(sigmauu_neutralino(msN3)), dew_funcu(sigmauu_neutralino(msN4)),
                             dew_funcu(sigmauu_chargino1()), dew_funcu(sigmauu_chargino2()), dew_funcu(sigmauu_h0()),
                             dew_funcu(sigmauu_heavy_h0()), dew_funcu(sigmauu_h_pm()), dew_funcu(sigmauu_w_pm()),
                             dew_funcu(sigmauu_z0()), dew_funcu(sigmauu_top()),
                             dew_funcu(sigmauu_2loop()), dew_funcd(sigmadd_2loop()),
                             dew_funcu(sigmauu_stau_sneut()), dew_funcd(sigmadd_stau_sneut())])\
                   / halfmzsq
        label_sort_array = np.sort(np.array([(contribs[0], 'mu'), (contribs[1], 'H_u'), (contribs[2], 'H_d'),
                                             (contribs[3], 'Sigma_d^d(stop_1)'), (contribs[4], 'Sigma_d^d(stop_2)'),
                                             (contribs[5], 'Sigma_d^d(sbot_1)'), (contribs[6], 'Sigma_d^d(sbot_2)'),
                                             (contribs[7], 'Sigma_d^d(stau_1)'), (contribs[8], 'Sigma_d^d(stau_2)'),
                                             (contribs[9], 'Sigma_d^d(1st gen. squarks)'),
                                             (contribs[10], 'Sigma_d^d(2nd gen squarks)'),
                                             (contribs[11], 'Sigma_d^d(neutralino_1)'),
                                             (contribs[12], 'Sigma_d^d(neutralino_2)'),
                                             (contribs[13], 'Sigma_d^d(neutralino_3)'),
                                             (contribs[14], 'Sigma_d^d(neutralino_4)'),
                                             (contribs[15], 'Sigma_d^d(chargino_1)'),
                                             (contribs[16], 'Sigma_d^d(chargino_2)'),
                                             (contribs[17], 'Sigma_d^d(h_0)'), (contribs[18], 'Sigma_d^d(H_0)'),
                                             (contribs[19], 'Sigma_d,u^d,u(H_+-)'),
                                             (contribs[20], 'Sigma_d,u^d,u(W_+-)'),
                                             (contribs[21], 'Sigma_d,u^d,u(Z_0)'), (contribs[22], 'Sigma_d^d(bottom)'),
                                             (contribs[23], 'Sigma_d^d(tau)'), (contribs[24], 'Sigma_u^u(stop_1)'),
                                             (contribs[25], 'Sigma_u^u(stop_2)'), (contribs[26], 'Sigma_u^u(sbot_1)'),
                                             (contribs[27], 'Sigma_u^u(sbot_2)'), (contribs[28], 'Sigma_u^u(stau_1)'),
                                             (contribs[29], 'Sigma_u^u(stau_2)'),
                                             (contribs[30], 'Sigma_u^u(sum 1st gen. squarks)'),
                                             (contribs[31], 'Sigma_u^u(sum 2nd gen. squarks)'),
                                             (contribs[32], 'Sigma_u^u(neutralino_1)'),
                                             (contribs[33], 'Sigma_u^u(neutralino_2)'),
                                             (contribs[34], 'Sigma_u^u(neutralino_3)'),
                                             (contribs[35], 'Sigma_u^u(neutralino_4)'),
                                             (contribs[36], 'Sigma_u^u(chargino_1)'),
                                             (contribs[37], 'Sigma_u^u(chargino_2)'),
                                             (contribs[38], 'Sigma_u^u(h_0)'), (contribs[39], 'Sigma_u^u(H_0)'),
                                             (contribs[40], 'Sigma_u^u(H_+-)'), (contribs[41], 'Sigma_u^u(W_+-)'),
                                             (contribs[42], 'Sigma_u^u(Z_0)'), (contribs[43], 'Sigma_u^u(top)'),
                                             (contribs[44], 'Sigma_u^u(O(alpha_s alpha_t))'),
                                             (contribs[45], 'Sigma_d^d(O(alpha_s alpha_t))'),
                                             (contribs[46], 'Sigma_u^u(tau sneutrino)'),
                                             (contribs[47], 'Sigma_d^d(tau sneutrino)')],
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
        if neg_mean_stop_mass_check == 1:
            print('WARNING!')
            print('WARNING!')
            print('WARNING!')
            print('WARNING! Possible negative running of stop masses in file: mt1^2: ' + str(m_stop_1sq) + ', mt2^2: ' + str(m_stop_2sq))
            print('WARNING!')
            print('WARNING!')
            print('WARNING!')
        if SLHA_scale_check == 1:
            print('WARNING!')
            print('WARNING!')
            print('WARNING!')
            print('WARNING! The maximum grid scale in your SLHA file, Q='+str(SLHA_scale)+', is more than 2 times the magnitude of the mean of the stop pole masses, sqrt(abs(mt1*mt2))='+str(np.abs(mean_stop_mass)) + '.')
            print('Results may not be accurate because the radiative corrections may not be minimized at this scale.')
            print('WARNING!')
            print('WARNING!')
            print('WARNING!')
        checksave = input("\nWould you like to save these results to a .txt file (will be saved to the current "
                              + "directory)? Enter Y to save the result or N to continue: ")
        if checksave in ('Y', 'y'):
            filenamecheck = input('\nThe default file name is "current_system_time_DEW_contrib_list.txt", e.g., '
                                  + timestr + '_DEW_contrib_list.txt.'
                                  + ' Would you like to keep this or input your own file name?'
                                  +  ' Enter Y to keep the default file name or N to be able to input your own: ')
            if filenamecheck.lower() in ('y', 'yes'):
                if neg_mean_stop_mass_check == 1:
                    print('WARNING!')
                    print('WARNING!')
                    print('WARNING!')
                    print('WARNING! Possible negative running of stop masses in file: mt1^2: ' + str(m_stop_1sq) + ', mt2^2: ' + str(m_stop_2sq))
                    print('WARNING!')
                    print('WARNING!')
                    print('WARNING!')
                if SLHA_scale_check == 1:
                    print('WARNING!')
                    print('WARNING!')
                    print('WARNING!')
                    print('WARNING! The maximum grid scale in your SLHA file, Q='+str(SLHA_scale)+', is more than 2 times the magnitude of the mean of the stop pole masses, sqrt(abs(mt1*mt2))='+str(np.abs(mean_stop_mass)) + '.')
                    print('Results may not be accurate because the radiative corrections may not be minimized at this scale.')
                    print('WARNING!')
                    print('WARNING!')
                    print('WARNING!')
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
                      file=open(newfilename + ".txt", "w"))
                print('\nThe ordered contributions to Delta_EW are as follows ' +
                      '(decr. order): ',
                      file=open(newfilename + ".txt", "a"))
                print('', file=open(newfilename + ".txt", "a"))
                for i in range(0, len(reverse_sort_array)):
                    print(str(i + 1) + ': ' + str(reverse_sort_array[i]),
                          file=open(newfilename + ".txt", "a"))
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
        del vHiggs, muQ, tanb, y_t, beta, y_b, y_tau, g_pr, g_s, mgl, m_stop_1sq, m_stop_2sq, g_EW, m_sbot_1sq, m_sbot_2sq, m_stau_1sq, m_stau_2sq, mtL, mtR, mtLsq, mtRsq, mbLsq, mbRsq, mtauLsq, mtauRsq, mselecLsq, msmuLsq
        del mymt, mymtsq, mymb, mymbsq, mymtau, mymtausq, mstauneutsq
        del mbL, mbR, mtauL, mtauR, msupL, msupR, msdownL, msdownR, mselecL, mselecR, mselecneutsq, msmuneutsq, msstrangeL, msstrangeR, SLHA_scale_check, neg_mean_stop_mass_check, SLHA_scale, mean_stop_mass
        del mscharmL, mscharmR, msmuL, msmuR, msN1, msN2, msN3, msN4, msC1sq, msC2sq, mZ, mh0sq, mH0sq, mHusq, mHdsq, mH_pmsq, M_1, M_2, a_t, a_b
        del a_tau, A_t, A_b, A_tau, mA0sq, Q_renorm_sq, halfmzsq, cmu, chu, chd, contribs, label_sort_array, reverse_sort_array, d
