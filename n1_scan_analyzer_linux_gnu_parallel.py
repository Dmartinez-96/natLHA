"""
Created on Sun Sep 12 12:54:59 2021.

@author: Dakotah Martinez
"""

import numpy as np
import pyslha
import os
import subprocess
from subprocess import DEVNULL, STDOUT
import shutil
import warnings
from pyslha import ParseError
import random
warnings.simplefilter('ignore')
os.chdir('/mnt/c/Users/dakot/Documents/Research/DEW_code/softsusy-4.1.10')


def lin_dist_func(a, b):
    """
    Return linearly increasing probability distr'n function w/ hit-or-miss.

    Parameters
    ----------
    a : Lowest value of random variable.
    b : Highest value of random variable.

    Returns
    -------
    Array with linearly increasing probability distribution for soft term.
    """
    check_if_keep = True
    while check_if_keep:
        rand_val_1 = random.random() * (b - a) + a
        rand_val_2 = random.random() * (b - a) + a
        if abs(rand_val_2) > abs(rand_val_1):
            # Reject value and try again
            check_if_keep = True
        else:
            check_if_keep = False
    return rand_val_1


def unif_dist_func(a, b):
    """
    Return uniform probability dist.'n function on array data.

    Parameters
    ----------
    a : Lowest value of uniform random variable.
    b : Highest value of uniform random variable.

    Returns
    -------
    Array with uniform probability distribution for tanb.
    """
    unif_val = random.uniform(a, b)
    return unif_val


def output_writer():
    """Write softsusy output given input."""
    fileopen = open('scan_test_outputs/my_softsusy_output_' + str(filenumber),
                    'w')
    cmd = ['./softpoint.x leshouches < scan_test_inputs/my_softsusy_input_'
           + str(filenumber) + ' > scan_test_outputs/my_softsusy_output_'
           + str(filenumber)]
    proc = subprocess.Popen(cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
    proc.wait()
    fileopen.close()


def input_writer():
    """Generate softsusy input with random linear draw on soft terms."""
    m0_12_rand_val = lin_dist_func(100.0, 60000.0)
    # m0_12_rand_val = lin_dist_func(100.0, 55000.0)
    m0_3_rand_val = lin_dist_func(100.0, 20000.0)
    # m0_3_rand_val = lin_dist_func(100.0, np.min([17500.0, m0_12_rand_val]))
    mhf_rand_val = lin_dist_func(500.0, 10000.0)
    # mhf_rand_val = lin_dist_func(500.0, 8000.0)
    A0_rand_val = lin_dist_func(-50000.0, 0.0)
    # A0_rand_val = lin_dist_func(-2.85, 0.0) * m0_3_rand_val
    mA_rand_val = lin_dist_func(300.0, 10000.0)
    # mA_rand_val = lin_dist_func(300.0, 9000.0)
    tanb_rand_val = unif_dist_func(3, 60)
    # tanb_rand_val = unif_dist_func(3, 50)
    file = open('scan_test_inputs/my_softsusy_input_' + str(filenumber), "w")
    file.write('Block MODSEL\n' + '    1   1\n'
               + 'Block SMINPUTS\n' + '    1   1.279340000e+02\n'
               + '    2   1.166370000e-05\n' + '    3   1.172000000e-01\n'
               + '    4   9.118760000e+01\n' + '    5   4.250000000e+00\n'
               + '    6   1.732000000e+02\n' + '    7   1.777000000e+00\n'
               + 'Block MINPAR\n' + '    2   '
               + '{:.9e}'.format(mhf_rand_val)
               + '\n    3   ' + '{:.9e}'.format(tanb_rand_val)
               + '\n    4   1.000000000e+00\n'
               + '    5   ' + '{:.9e}'.format(A0_rand_val)
               + '\nBlock SOFTSUSY\n'
               + '    0   0.000000000e+00\n' + '    1   1.000000000e-03\n'
               + '    2   0.000000000e+00\n' + '    3   0.000000000e+00\n'
               + '    4   1.000000000e+00\n' + '    6   1.000000000e-04\n'
               + '    7   3.000000000e+00\n' + '   10   0.000000000e+00\n'
               + '   11   1.000000000e+19\n' + '   12   1.000000000e+00\n'
               + '   13   0.000000000e+00\n' + '   19   1.000000000e+00\n'
               + '   20   3.100000000e+01\n'
               + 'Block EXTPAR\n' + '   23   2.000000000e+02\n'
               + '   26   ' + '{:.9e}'.format(mA_rand_val) + '\n   31   '
               + '{:.9e}'.format(m0_12_rand_val) + '\n   32   '
               + '{:.9e}'.format(m0_12_rand_val) + '\n   33   '
               + '{:.9e}'.format(m0_3_rand_val) + '\n   34   '
               + '{:.9e}'.format(m0_12_rand_val) + '\n   35   '
               + '{:.9e}'.format(m0_12_rand_val) + '\n   36   '
               + '{:.9e}'.format(m0_3_rand_val) + '\n   41   '
               + '{:.9e}'.format(m0_12_rand_val) + '\n   42   '
               + '{:.9e}'.format(m0_12_rand_val) + '\n   43   '
               + '{:.9e}'.format(m0_3_rand_val) + '\n   44   '
               + '{:.9e}'.format(m0_12_rand_val) + '\n   45   '
               + '{:.9e}'.format(m0_12_rand_val) + '\n   46   '
               + '{:.9e}'.format(m0_3_rand_val) + '\n   47   '
               + '{:.9e}'.format(m0_12_rand_val) + '\n   48   '
               + '{:.9e}'.format(m0_12_rand_val) + '\n   49   '
               + '{:.9e}'.format(m0_3_rand_val))
    file.close()


def main():
    """
    Analyze softsusy output.

    Returns
    -------
    calculated_dew_array : MAX value of DEW from softsusy output.

    """
    try:
        d[0] = pyslha.read('scan_test_outputs/my_softsusy_output_'
                           + str(filenumber))
        vHiggs[0] = d[0].blocks['HMIX'][3]
        muQ[0] = d[0].blocks['HMIX'][1]
        tanb[0] = d[0].blocks['HMIX'][2]
        beta[0] = np.arctan(tanb[0])
        y_t[0] = d[0].blocks['YU'][3, 3]
        y_b[0] = d[0].blocks['YD'][3, 3]
        y_tau[0] = d[0].blocks['YE'][3, 3]
        g_pr[0] = d[0].blocks['GAUGE'][2]
        g_EW[0] = d[0].blocks['GAUGE'][1]
        m_stop_1[0] = d[0].blocks['MASS'][1000006]
        m_stop_2[0] = d[0].blocks['MASS'][2000006]
        m_sbot_1[0] = d[0].blocks['MASS'][1000005]
        m_sbot_2[0] = d[0].blocks['MASS'][2000005]
        m_stau_1[0] = d[0].blocks['MASS'][1000015]
        m_stau_2[0] = d[0].blocks['MASS'][2000015]
        mtL[0] = d[0].blocks['MSOFT'][43]
        mtR[0] = d[0].blocks['MSOFT'][46]
        mbL[0] = d[0].blocks['MSOFT'][43]
        mbR[0] = d[0].blocks['MSOFT'][49]
        mtauL[0] = d[0].blocks['MSOFT'][33]
        mtauR[0] = d[0].blocks['MSOFT'][36]
        msupL[0] = d[0].blocks['MSOFT'][41]
        msupR[0] = d[0].blocks['MSOFT'][44]
        msdownL[0] = d[0].blocks['MSOFT'][41]
        msdownR[0] = d[0].blocks['MSOFT'][47]
        mselecL[0] = d[0].blocks['MSOFT'][31]
        mselecR[0] = d[0].blocks['MSOFT'][34]
        mselecneut[0] = d[0].blocks['MASS'][1000012]
        msmuneut[0] = d[0].blocks['MASS'][1000014]
        msstrangeL[0] = d[0].blocks['MSOFT'][42]
        msstrangeR[0] = d[0].blocks['MSOFT'][48]
        mscharmL[0] = d[0].blocks['MSOFT'][42]
        mscharmR[0] = d[0].blocks['MSOFT'][45]
        msmuL[0] = d[0].blocks['MSOFT'][32]
        msmuR[0] = d[0].blocks['MSOFT'][35]
        msN1[0] = d[0].blocks['MASS'][1000022]
        msN2[0] = d[0].blocks['MASS'][1000023]
        msN3[0] = d[0].blocks['MASS'][1000025]
        msN4[0] = d[0].blocks['MASS'][1000035]
        msC1[0] = d[0].blocks['MASS'][1000024]
        msC2[0] = d[0].blocks['MASS'][1000037]
        mZ[0] = d[0].blocks['SMINPUTS'][4]
        mA0sq[0] = d[0].blocks['HMIX'][4]
        mh0[0] = d[0].blocks['MASS'][25]
        mH0[0] = d[0].blocks['MASS'][35]
        mHusq[0] = d[0].blocks['MSOFT'][22]
        mHdsq[0] = d[0].blocks['MSOFT'][21]
        mH_pm[0] = d[0].blocks['MASS'][37]
        mgl[0] = d[0].blocks['MASS'][1000021]
        M_1[0] = d[0].blocks['MSOFT'][1]
        M_2[0] = d[0].blocks['MSOFT'][2]
        a_t[0] = d[0].blocks['AU'][3, 3] * y_t[0]
        a_b[0] = d[0].blocks['AD'][3, 3] * y_b[0]
        a_tau[0] = d[0].blocks['AE'][3, 3] * y_tau[0]
        Q_renorm_sq[0] = m_stop_1[0] * m_stop_2[0]
        halfmzsq[0] = np.power(mZ[0], 2) / 2
        cmu[0] = np.abs(np.power(muQ[0], 2))
    except (KeyError, ParseError, AttributeError):
        calculated_dew_array[0] = 1000
    try:
        chu[0] = dew_funcu(mHusq[0])[0]
        chd[0] = dew_funcd(mHdsq[0])[0]
        contribs[0] = np.array([cmu[0], chu[0], chd[0],
                                dew_funcd(sigmadd_stop1())[0],
                                dew_funcd(sigmadd_stop2())[0],
                                dew_funcd(sigmadd_sbottom1())[0],
                                dew_funcd(sigmadd_sbottom2())[0],
                                dew_funcd(sigmadd_stau1())[0],
                                dew_funcd(sigmadd_stau2())[0],
                                dew_funcd(sigmadd_sup_l() + sigmadd_sup_r()
                                          + sigmadd_sdown_l()
                                          + sigmadd_sdown_r()
                                          + sigmadd_selec_l()
                                          + sigmadd_selec_r()
                                          + sigmadd_sel_neut())[0],
                                dew_funcd(sigmadd_sstrange_l()
                                          + sigmadd_sstrange_r()
                                          + sigmadd_scharm_l()
                                          + sigmadd_scharm_r()
                                          + sigmadd_smu_l() + sigmadd_smu_r()
                                          + sigmadd_smu_sneut())[0],
                                dew_funcd(sigmadd_neutralino(msN1[0]))[0],
                                dew_funcd(sigmadd_neutralino(msN2[0]))[0],
                                dew_funcd(sigmadd_neutralino(msN3[0]))[0],
                                dew_funcd(sigmadd_neutralino(msN4[0]))[0],
                                dew_funcd(sigmadd_chargino1())[0],
                                dew_funcd(sigmadd_chargino2())[0],
                                dew_funcd(sigmadd_h0())[0],
                                dew_funcd(sigmadd_heavy_h0())[0],
                                dew_funcd(sigmauu_h_pm())[0],
                                dew_funcd(sigmauu_w_pm())[0],
                                dew_funcd(sigmauu_z0())[0],
                                dew_funcd(sigmadd_bottom())[0],
                                dew_funcd(sigmadd_tau())[0],
                                dew_funcu(sigmauu_stop1())[0],
                                dew_funcu(sigmauu_stop2())[0],
                                dew_funcu(sigmauu_sbottom1())[0],
                                dew_funcu(sigmauu_sbottom2())[0],
                                dew_funcu(sigmauu_stau1())[0],
                                dew_funcu(sigmauu_stau2())[0],
                                dew_funcu(sigmauu_sup_l() + sigmauu_sup_r()
                                          + sigmauu_sdown_l()
                                          + sigmauu_sdown_r()
                                          + sigmauu_selec_l()
                                          + sigmauu_selec_r()
                                          + sigmauu_sel_neut())[0],
                                dew_funcu(sigmauu_sstrange_l()
                                          + sigmauu_sstrange_r()
                                          + sigmauu_scharm_l()
                                          + sigmauu_scharm_r()
                                          + sigmauu_smu_l() + sigmauu_smu_r()
                                          + sigmauu_smu_sneut())[0],
                                dew_funcu(sigmauu_neutralino(msN1[0]))[0],
                                dew_funcu(sigmauu_neutralino(msN2[0]))[0],
                                dew_funcu(sigmauu_neutralino(msN3[0]))[0],
                                dew_funcu(sigmauu_neutralino(msN4[0]))[0],
                                dew_funcu(sigmauu_chargino1())[0],
                                dew_funcu(sigmauu_chargino2())[0],
                                dew_funcu(sigmauu_h0())[0],
                                dew_funcu(sigmauu_heavy_h0())[0],
                                dew_funcu(sigmauu_h_pm())[0],
                                dew_funcu(sigmauu_w_pm())[0],
                                dew_funcu(sigmauu_z0())[0],
                                dew_funcu(sigmauu_top())[0]]) / halfmzsq[0]
        calculated_dew_array[0] = np.amax(contribs[0])
    except (KeyError, AttributeError, ParseError):
        calculated_dew_array[0] = 1000
    try:  # get rid of bad minima points and other problematic points
        test_if_bad = d[0].blocks['SPINFO'][4]
        calculated_dew_array[0] = 1000
    except (KeyError):  # keep valid points
        pass
    except:
        calculated_dew_array[0] = 1000
    return calculated_dew_array


def m_w_sq():
    """Return W boson squared mass."""
    my_mw_sq = (np.power(g_EW, 2) / 2) * np.power(vHiggs, 2)
    return my_mw_sq


def mz_q_sq():
    """Return m_Z(Q)^2."""
    mzqsq = np.power(vHiggs, 2) * ((np.power(g_EW, 2) + np.power(g_pr, 2)) / 2)
    return mzqsq


# Fundamental equations: #

def logfunc(mass):
    """
    Return F = m^2 * (ln(m^2 / Q^2) - 1).

    Parameters
    ----------
    mass : Input mass.
    """
    myf = np.power(mass, 2) * (np.log(np.abs((np.power(mass, 2))
                                             / (Q_renorm_sq))) - 1)
    return myf


def sinsqb():
    """Return sin^2(beta)."""
    mysinsqb = np.power(np.sin(beta), 2)
    return mysinsqb


def cossqb():
    """Return cos^2(beta)."""
    mycossqb = np.power(np.cos(beta), 2)
    return mycossqb


def v_higgs_u():
    """Return up-type Higgs VEV."""
    myvu = vHiggs * np.sin(beta)
    return myvu


def v_higgs_d():
    """Return down-type Higgs VEV."""
    myvd = vHiggs * np.cos(beta)
    return myvd


def tan_theta_w():
    """Return tan(theta_W), the Weinberg angle."""
    mytanthetaw = g_pr / g_EW
    return mytanthetaw


def sinsq_theta_w():
    """Return sin^2(theta_W), the Weinberg angle."""
    thetaw = np.arctan(tan_theta_w())
    mysinsqthetaw = np.power(np.sin(thetaw), 2)
    return mysinsqthetaw


def cos2b():
    """Return cos(2*beta)."""
    mycos2b = cossqb() - sinsqb()
    return mycos2b


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
    sigmauu_stop_1 = (3 / (16 * (np.power(np.pi, 2)))) * logfunc(m_stop_1) \
        * (np.power(y_t, 2) - gz_sq()
           - (stop_num / (np.power(m_stop_2, 2) - np.power(m_stop_1, 2))))
    return sigmauu_stop_1


def sigmadd_stop1():
    """Return one-loop correction Sigma_d^d(stop_1)."""
    delta_stop = ((1 / 2) - (4 / 3) * sinsq_theta_w()) * 2 \
        * (((np.power(mtL, 2) - np.power(mtR, 2)) / 2)
           + (mz_q_sq()) * cos2b() * ((1 / 4) - (2 / 3) * sinsq_theta_w()))
    stop_num = np.power(y_t, 2) * np.power(muQ, 2) \
        + (2 * gz_sq() * delta_stop)
    sigmadd_stop_1 = (3 / (16 * np.power(np.pi, 2))) * logfunc(m_stop_1) \
        * (gz_sq() - (stop_num
                      / (np.power(m_stop_2, 2) - np.power(m_stop_1, 2))))
    return sigmadd_stop_1


def sigmauu_stop2():
    """Return one-loop correction Sigma_u^u(stop_2)."""
    delta_stop = ((1 / 2) - (4 / 3) * sinsq_theta_w()) * 2 \
        * (((np.power(mtL, 2) - np.power(mtR, 2)) / 2)
           + (mz_q_sq()) * cos2b() * ((1 / 4) - (2 / 3) * sinsq_theta_w()))
    stop_num = np.power(a_t, 2) - (2 * gz_sq() * delta_stop)
    sigmauu_stop_2 = (3 / (16 * (np.power(np.pi, 2)))) * logfunc(m_stop_2) \
        * (np.power(y_t, 2) - gz_sq()
           + (stop_num / (np.power(m_stop_2, 2) - np.power(m_stop_1, 2))))
    return sigmauu_stop_2


def sigmadd_stop2():
    """Return one-loop correction Sigma_d^d(stop_2)."""
    delta_stop = ((1 / 2) - (4 / 3) * sinsq_theta_w()) * 2 \
        * (((np.power(mtL, 2) - np.power(mtR, 2)) / 2)
           + (mz_q_sq()) * cos2b() * ((1 / 4) - (2 / 3) * sinsq_theta_w()))
    stop_num = np.power(y_t, 2) * np.power(muQ, 2) \
        + (2 * gz_sq() * delta_stop)
    sigmadd_stop_2 = (3 / (16 * np.power(np.pi, 2))) * logfunc(m_stop_2) \
        * (gz_sq() + (stop_num
                      / (np.power(m_stop_2, 2) - np.power(m_stop_1, 2))))
    return sigmadd_stop_2

# Sbottom squarks: #


def sigmauu_sbottom1():
    """Return one-loop correction Sigma_u^u(sbottom_1)."""
    delta_sbot = ((1 / 2) - (2 / 3) * sinsq_theta_w()) * 2 \
        * (((np.power(mbL, 2) - np.power(mbR, 2)) / 2)
           - (mz_q_sq()) * cos2b() * ((1 / 4) - (1 / 3) * sinsq_theta_w()))
    sbot_num = np.power(a_b, 2) - (2 * gz_sq() * delta_sbot)
    sigmauu_sbot = (3 / (16 * np.power(np.pi, 2))) * logfunc(m_sbot_1) \
        * (np.power(y_b, 2) - gz_sq()
           - (sbot_num / (np.power(m_sbot_2, 2) - np.power(m_sbot_1, 2))))
    return sigmauu_sbot


def sigmauu_sbottom2():
    """Return one-loop correction Sigma_u^u(sbottom_2)."""
    delta_sbot = ((1 / 2) - (2 / 3) * sinsq_theta_w()) * 2 \
        * (((np.power(mbL, 2) - np.power(mbR, 2)) / 2)
           - (mz_q_sq()) * cos2b() * ((1 / 4) - (1 / 3) * sinsq_theta_w()))
    sbot_num = np.power(a_b, 2) - (2 * gz_sq() * delta_sbot)
    sigmauu_sbot = (3 / (16 * np.power(np.pi, 2))) * logfunc(m_sbot_2) \
        * (np.power(y_b, 2) - gz_sq()
           + (sbot_num / (np.power(m_sbot_2, 2) - np.power(m_sbot_1, 2))))
    return sigmauu_sbot


def sigmadd_sbottom1():
    """Return one-loop correction Sigma_d^d(sbottom_1)."""
    delta_sbot = ((1 / 2) - (2 / 3) * sinsq_theta_w()) * 2 \
        * (((np.power(mbL, 2) - np.power(mbR, 2)) / 2)
           - (mz_q_sq()) * cos2b() * ((1 / 4) - (1 / 3) * sinsq_theta_w()))
    sbot_num = np.power(y_b, 2) * np.power(muQ, 2) \
        + (2 * gz_sq() * delta_sbot)
    sigmadd_sbot = (3 / (16 * np.power(np.pi, 2))) * logfunc(m_sbot_1) \
        * (gz_sq()
           - (sbot_num / (np.power(m_sbot_2, 2) - np.power(m_sbot_1, 2))))
    return sigmadd_sbot


def sigmadd_sbottom2():
    """Return one-loop correction Sigma_d^d(sbottom_2)."""
    delta_sbot = ((1 / 2) - (2 / 3) * sinsq_theta_w()) * 2 \
        * (((np.power(mbL, 2) - np.power(mbR, 2)) / 2)
           - (mz_q_sq()) * cos2b() * ((1 / 4) - (1 / 3) * sinsq_theta_w()))
    sbot_num = np.power(y_b, 2) * np.power(muQ, 2) \
        + (2 * gz_sq() * delta_sbot)
    sigmadd_sbot = (3 / (16 * np.power(np.pi, 2))) * logfunc(m_sbot_2) \
        * (gz_sq()
           + (sbot_num / (np.power(m_sbot_2, 2) - np.power(m_sbot_1, 2))))
    return sigmadd_sbot


# Stau sleptons: #

def sigmauu_stau1():
    """Return one-loop correction Sigma_u^u(stau_1)."""
    delta_stau = ((1 / 2) - 2 * sinsq_theta_w()) * 2 \
        * (((np.power(mtauL, 2) - np.power(mtauR, 2)) / 2)
           - (mz_q_sq()) * cos2b() * ((1 / 4) - sinsq_theta_w()))
    stau_num = np.power(a_tau, 2) - (2 * gz_sq() * delta_stau)
    sigmauu_stau = (1 / (16 * np.power(np.pi, 2))) * logfunc(m_stau_1) \
        * (np.power(y_tau, 2) - gz_sq()
           - (stau_num / (np.power(m_stau_2, 2) - np.power(m_stau_1, 2))))
    return sigmauu_stau


def sigmauu_stau2():
    """Return one-loop correction Sigma_u^u(stau_2)."""
    delta_stau = ((1 / 2) - 2 * sinsq_theta_w()) * 2 \
        * (((np.power(mtauL, 2) - np.power(mtauR, 2)) / 2)
           - (mz_q_sq()) * cos2b() * ((1 / 4) - sinsq_theta_w()))
    stau_num = np.power(a_tau, 2) - (2 * gz_sq() * delta_stau)
    sigmauu_stau = (1 / (16 * np.power(np.pi, 2))) * logfunc(m_stau_2) \
        * (np.power(y_tau, 2) - gz_sq()
           + (stau_num / (np.power(m_stau_2, 2) - np.power(m_stau_1, 2))))
    return sigmauu_stau


def sigmadd_stau1():
    """Return one-loop correction Sigma_d^d(stau_1)."""
    delta_stau = ((1 / 2) - 2 * sinsq_theta_w()) * 2 \
        * (((np.power(mtauL, 2) - np.power(mtauR, 2)) / 2)
           - (mz_q_sq()) * cos2b() * ((1 / 4) - sinsq_theta_w()))
    stau_num = np.power(y_tau, 2) * np.power(muQ, 2) \
        + (2 * gz_sq() * delta_stau)
    sigmauu_stau = (1 / (16 * np.power(np.pi, 2))) * logfunc(m_stau_1) \
        * (gz_sq()
           - (stau_num / (np.power(m_stau_2, 2) - np.power(m_stau_1, 2))))
    return sigmauu_stau


def sigmadd_stau2():
    """Return one-loop correction Sigma_d^d(stau_2)."""
    delta_stau = ((1 / 2) - 2 * sinsq_theta_w()) * 2 \
        * (((np.power(mtauL, 2) - np.power(mtauR, 2)) / 2)
           - (mz_q_sq()) * cos2b() * ((1 / 4) - sinsq_theta_w()))
    stau_num = np.power(y_tau, 2) * np.power(muQ, 2) \
        + (2 * gz_sq() * delta_stau)
    sigmauu_stau = (1 / (16 * np.power(np.pi, 2))) * logfunc(m_stau_2) \
        * (gz_sq()
           + (stau_num / (np.power(m_stau_2, 2) - np.power(m_stau_1, 2))))
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
        * logfunc(mselecneut) * gz_sq()
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
        * logfunc(mselecneut) * gz_sq()
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
        * (1 / 2) * logfunc(msmuneut)
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
        * gz_sq() * logfunc(msmuneut)
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
    chargino_den = np.power(msC2, 2) - np.power(msC1, 2)
    sigma_uu_chargino1 = -1 * (np.power(g_EW, 2) / (16 * np.power(np.pi, 2))) \
        * (1 - (chargino_num / chargino_den)) * logfunc(msC1)
    return sigma_uu_chargino1


def sigmauu_chargino2():
    """Return one-loop correction Sigma_u^u(chargino_2)."""
    chargino_num = ((-2) * m_w_sq() * cos2b()) + np.power(M_2, 2) \
        + np.power(muQ, 2)
    chargino_den = np.power(msC2, 2) - np.power(msC1, 2)
    sigma_uu_chargino2 = -1 * (np.power(g_EW, 2) / (16 * np.power(np.pi, 2))) \
        * (1 + (chargino_num / chargino_den)) * logfunc(msC2)
    return sigma_uu_chargino2


def sigmadd_chargino1():
    """Return one-loop correction Sigma_d^d(chargino_1)."""
    chargino_num = (2 * m_w_sq() * cos2b()) + np.power(M_2, 2) \
        + np.power(muQ, 2)
    chargino_den = np.power(msC2, 2) - np.power(msC1, 2)
    sigma_dd_chargino1 = -1 * (np.power(g_EW, 2) / (16 * np.power(np.pi, 2))) \
        * (1 - (chargino_num / chargino_den)) * logfunc(msC1)
    return sigma_dd_chargino1


def sigmadd_chargino2():
    """Return one-loop correction Sigma_d^d(chargino_2)."""
    chargino_num = (2 * m_w_sq() * cos2b()) + np.power(M_2, 2) \
        + np.power(muQ, 2)
    chargino_den = np.power(msC2, 2) - np.power(msC1, 2)
    sigma_dd_chargino2 = -1 * (np.power(g_EW, 2) / (16 * np.power(np.pi, 2))) \
        * (1 + (chargino_num / chargino_den)) * logfunc(msC2)
    return sigma_dd_chargino2

# Higgs bosons (sigmauu = sigmadd here): #


def sigmauu_h0():
    """Return one-loop correction Sigma_u^u(h_0) (lighter neutral Higgs)."""
    mynum = mz_q_sq() + (mA0sq * (1 + (4 * cos2b())
                                  + (2 * np.power(cos2b(), 2))))
    myden = np.power(mH0, 2) - np.power(mh0, 2)
    sigma_uu_h0 = (gz_sq() / (16 * np.power(np.pi, 2))) \
        * (1 - (mynum / myden)) * logfunc(mh0)
    return sigma_uu_h0


def sigmadd_h0():
    """Return one-loop correction Sigma_d^d(h_0) (lighter neutral Higgs)."""
    mynum = mz_q_sq() + (mA0sq * (1 - (4 * cos2b())
                                  + (2 * np.power(cos2b(), 2))))
    myden = np.power(mH0, 2) - np.power(mh0, 2)
    sigma_dd_h0 = (gz_sq() / (16 * np.power(np.pi, 2))) \
        * (1 - (mynum / myden)) * logfunc(mh0)
    return sigma_dd_h0


def sigmauu_heavy_h0():
    """Return one-loop correction Sigma_u^u(H_0) (heavier neutr. Higgs)."""
    mynum = mz_q_sq() + (mA0sq * (1 + (4 * cos2b())
                                  + (2 * np.power(cos2b(), 2))))
    myden = np.power(mH0, 2) - np.power(mh0, 2)
    sigma_uu_heavy_h0 = (gz_sq() / (16 * np.power(np.pi, 2))) \
        * (1 + (mynum / myden)) * logfunc(mH0)
    return sigma_uu_heavy_h0


def sigmadd_heavy_h0():
    """Return one-loop correction Sigma_d^d(H_0) (heavier neutr. Higgs)."""
    mynum = mz_q_sq() + (mA0sq * (1 - (4 * cos2b())
                                  + (2 * np.power(cos2b(), 2))))
    myden = np.power(mH0, 2) - np.power(mh0, 2)
    sigma_dd_heavy_h0 = (gz_sq() / (16 * np.power(np.pi, 2))) \
        * (1 + (mynum / myden)) * logfunc(mH0)
    return sigma_dd_heavy_h0


def sigmauu_h_pm():
    """Return one-loop correction Sigma_u,d^u,d(H_{+-})."""
    sigma_uu_h_pm = (np.power((g_EW / np.pi), 2) / (32)) * logfunc(mH_pm)
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
    filenumber = np.random.randint(low=13, high=2**32 - 1, dtype=np.int64)
    d = [None] * 1
    vHiggs = np.empty(1)
    muQ = np.empty(1)
    tanb = np.empty(1)
    beta = np.empty(1)
    y_t = np.empty(1)
    y_b = np.empty(1)
    y_tau = np.empty(1)
    g_pr = np.empty(1)
    g_EW = np.empty(1)
    m_stop_1 = np.empty(1)
    m_stop_2 = np.empty(1)
    m_sbot_1 = np.empty(1)
    m_sbot_2 = np.empty(1)
    m_stau_1 = np.empty(1)
    m_stau_2 = np.empty(1)
    mtL = np.empty(1)
    mtR = np.empty(1)
    mbL = np.empty(1)
    mbR = np.empty(1)
    mtauL = np.empty(1)
    mtauR = np.empty(1)
    msupL = np.empty(1)
    msupR = np.empty(1)
    msdownL = np.empty(1)
    msdownR = np.empty(1)
    mselecL = np.empty(1)
    mselecR = np.empty(1)
    mselecneut = np.empty(1)
    msmuneut = np.empty(1)
    msstrangeL = np.empty(1)
    msstrangeR = np.empty(1)
    mscharmL = np.empty(1)
    mscharmR = np.empty(1)
    msmuL = np.empty(1)
    msmuR = np.empty(1)
    msN1 = np.empty(1)
    msN2 = np.empty(1)
    msN3 = np.empty(1)
    msN4 = np.empty(1)
    msC1 = np.empty(1)
    msC2 = np.empty(1)
    mZ = np.empty(1)
    mA0sq = np.empty(1)
    mh0 = np.empty(1)
    mH0 = np.empty(1)
    mHusq = np.empty(1)
    mHdsq = np.empty(1)
    mH_pm = np.empty(1)
    mgl = np.empty(1)
    M_1 = np.empty(1)
    M_2 = np.empty(1)
    a_t = np.empty(1)
    a_b = np.empty(1)
    a_tau = np.empty(1)
    Q_renorm_sq = np.empty(1)
    halfmzsq = np.empty(1)
    cmu = np.empty(1)
    chu = np.empty(1)
    chd = np.empty(1)
    calculated_dew_array = np.ones(1) * 1000
    contribs = np.empty((1, 44))
    d = [None] * 1
    vHiggs = np.empty(1, dtype=np.float64)
    muQ = np.empty(1, dtype=np.float64)
    tanb = np.empty(1, dtype=np.float64)
    beta = np.empty(1, dtype=np.float64)
    y_t = np.empty(1, dtype=np.float64)
    y_b = np.empty(1, dtype=np.float64)
    y_tau = np.empty(1, dtype=np.float64)
    g_pr = np.empty(1, dtype=np.float64)
    g_EW = np.empty(1, dtype=np.float64)
    m_stop_1 = np.empty(1, dtype=np.float64)
    m_stop_2 = np.empty(1, dtype=np.float64)
    m_sbot_1 = np.empty(1, dtype=np.float64)
    m_sbot_2 = np.empty(1, dtype=np.float64)
    m_stau_1 = np.empty(1, dtype=np.float64)
    m_stau_2 = np.empty(1, dtype=np.float64)
    mtL = np.empty(1, dtype=np.float64)
    mtR = np.empty(1, dtype=np.float64)
    mbL = np.empty(1, dtype=np.float64)
    mbR = np.empty(1, dtype=np.float64)
    mtauL = np.empty(1, dtype=np.float64)
    mtauR = np.empty(1, dtype=np.float64)
    msupL = np.empty(1, dtype=np.float64)
    msupR = np.empty(1, dtype=np.float64)
    msdownL = np.empty(1, dtype=np.float64)
    msdownR = np.empty(1, dtype=np.float64)
    mselecL = np.empty(1, dtype=np.float64)
    mselecR = np.empty(1, dtype=np.float64)
    mselecneut = np.empty(1, dtype=np.float64)
    msmuneut = np.empty(1, dtype=np.float64)
    msstrangeL = np.empty(1, dtype=np.float64)
    msstrangeR = np.empty(1, dtype=np.float64)
    mscharmL = np.empty(1, dtype=np.float64)
    mscharmR = np.empty(1, dtype=np.float64)
    msmuL = np.empty(1, dtype=np.float64)
    msmuR = np.empty(1, dtype=np.float64)
    msN1 = np.empty(1, dtype=np.float64)
    msN2 = np.empty(1, dtype=np.float64)
    msN3 = np.empty(1, dtype=np.float64)
    msN4 = np.empty(1, dtype=np.float64)
    msC1 = np.empty(1, dtype=np.float64)
    msC2 = np.empty(1, dtype=np.float64)
    mZ = np.empty(1, dtype=np.float64)
    mA0sq = np.empty(1, dtype=np.float64)
    mh0 = np.empty(1, dtype=np.float64)
    mH0 = np.empty(1, dtype=np.float64)
    mHusq = np.empty(1, dtype=np.float64)
    mHdsq = np.empty(1, dtype=np.float64)
    mH_pm = np.empty(1, dtype=np.float64)
    mgl = np.empty(1, dtype=np.float64)
    M_1 = np.empty(1, dtype=np.float64)
    M_2 = np.empty(1, dtype=np.float64)
    a_t = np.empty(1, dtype=np.float64)
    a_b = np.empty(1, dtype=np.float64)
    a_tau = np.empty(1, dtype=np.float64)
    Q_renorm_sq = np.empty(1, dtype=np.float64)
    halfmzsq = np.empty(1, dtype=np.float64)
    cmu = np.empty(1, dtype=np.float64)
    chu = np.empty(1, dtype=np.float64)
    chd = np.empty(1, dtype=np.float64)
    calculated_dew_array = np.ones(1, dtype=np.float64) * 1000
    contribs = np.empty((1, 44), dtype=np.float64)

    input_writer()
    output_writer()

    main()
    if calculated_dew_array[0] <= 30:
        shutil.copy('scan_test_outputs/my_softsusy_output_' + str(filenumber),
                    'valid_n1_scan_BM_points/' + str(filenumber))
    os.remove('scan_test_outputs/my_softsusy_output_' + str(filenumber))
    os.remove('scan_test_inputs/my_softsusy_input_' + str(filenumber))
    if calculated_dew_array[0] <= 50:
        print(str(calculated_dew_array[0]))
