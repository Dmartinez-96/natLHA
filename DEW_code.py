# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pyslha

filename = input("Enter name of input file: ")
inputFile = open(filename)

#########################
# Mass relations:
#########################

def mWsq(g, vHiggs):
    mymWsq = (np.power(g, 2) / 2) * np.power(vHiggs, 2)
    return mymWsq
    
def mZsq(g, g_prime, vHiggs):
    mymZsq = ((np.power(g, 2) + np.power(g_prime, 2)) / 2) * np.power(vHiggs, 2)
    return mymZsq

def mA0sq(mu, mHu, mHd):
    mymA0sq = 2 * np.power(np.abs(mu), 2) + np.power(mHu, 2) + np.power(mHd, 2)
    return mymA0sq

def mHpmsq(mu, mHu, mHd, g, vHiggs):
    mymWsq = mWsq(g, vHiggs)
    mymHpmsq = mA0sq(mu, mHu, mHd) + mymWsq
    return mymHpmsq
    
#########################
# Fundamental equations
#########################

def F(m, Q_renorm):
    myF = np.power(m, 2) * (np.log((np.power(m, 2)) / (np.power(Q_renorm, 2))) - 1)
    return myF

def sinsqb(tanb): #sin^2(beta)
    mysinsqb = (np.power(tanb, 2) / (1 + np.power(tanb, 2)))
    return mysinsqb

def cossqb(tanb): #cos^2(beta)
    mycossqb = 1 - (np.power(tanb, 2) / (1 + np.power(tanb, 2)))
    return mycossqb

def vu(vHiggs, tanb): # up Higgs VEV
    myvu = vHiggs * np.sqrt(sinsqb(tanb))
    return myvu

def vd(vHiggs, tanb): # down Higgs VEV
    myvd = vHiggs * np.sqrt(cossqb(tanb))
    return myvd

#########################
# Stop squarks:
#########################

def sigmauu_stop1(vHiggs, mu, tanb, y_t, g, g_prime, sinsq_theta_W, m_stop_1, m_stop_2, mtL, mtR, a_t, Q_renorm):
    delta_uL = ((1 / 2) - (2 / 3) * sinsq_theta_W) * ((np.power(g, 2) + np.power (g_prime, 2)) / 2) * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    delta_uR = (2 / 3) * sinsq_theta_W * ((np.power(g, 2) + np.power (g_prime, 2)) / 2) * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    stop_num = ((np.power(mtL, 2) - np.power(mtR, 2) + delta_uL - delta_uR) \
                * ((-1 / 2) + ((4 / 3) * sinsq_theta_W)) * (np.power(g, 2) + np.power(g_prime, 2)) / 2)\
                + 2 * np.power(a_t, 2) 
    Sigmauu_stop1  = (3 / (32 * (np.power(np.pi, 2)))) * (2 * np.power(y_t, 2) \
                   + ((np.power(g, 2) + np.power(g_prime, 2)) * (8 * sinsq_theta_W - 3) / 12)\
                   - (stop_num / (np.power(m_stop_2, 2) - np.power(m_stop_1, 2)))) * F(m_stop_1, Q_renorm)
    return Sigmauu_stop1
    
def sigmadd_stop1(vHiggs, mu, tanb, y_t, g, g_prime, sinsq_theta_W, m_stop_1, m_stop_2, mtL, mtR, a_t, Q_renorm):
    delta_uL = ((1 / 2) - (2 / 3) * sinsq_theta_W) * ((np.power(g, 2) + np.power (g_prime, 2)) / 2) * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    delta_uR = (2 / 3) * sinsq_theta_W * ((np.power(g, 2) + np.power (g_prime, 2)) / 2) * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    stop_num = ((np.power(mtL, 2) - np.power(mtR, 2) + delta_uL - delta_uR) \
                * ((1 / 2) + (4 / 3) * sinsq_theta_W) * (np.power(g, 2) + np.power(g_prime, 2)) / 2)\
                + 2 * np.power(y_t, 2) * np.power(mu, 2)
    Sigmadd_stop1 = (3 / (32 * (np.power(np.pi, 2)))) * (((np.power(g, 2) + np.power(g_prime, 2)) / 2) - (stop_num / (np.power(mt_stop_2, 2) - np.power(mt_stop_1, 2)))) * F(m_stop_1, Q_renorm)
    return Sigmadd_stop1

def sigmauu_stop2(vHiggs, mu, tanb, y_t, g, g_prime, theta_W, m_stop, mtL, mtR, a_t, Q_renorm):
    delta_uL = ((1 / 2) - (2 / 3) * sinsq_theta_W) * ((np.power(g, 2) + np.power (g_prime, 2)) / 2) * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    delta_uR = (2 / 3) * sinsq_theta_W * ((np.power(g, 2) + np.power (g_prime, 2)) / 2) * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    stop_num = ((np.power(mtL, 2) - np.power(mtR, 2) + delta_uL - delta_uR) \
                * ((-1 / 2) + ((4 / 3) * sinsq_theta_W)) * (np.power(g, 2) + np.power(g_prime, 2)) / 2)\
                + 2 * np.power(a_t, 2) 
    Sigmauu_stop2  = (3 / (32 * (np.power(np.pi, 2)))) * (2 * np.power(y_t, 2) \
                   + ((np.power(g, 2) + np.power(g_prime, 2)) * (8 * sinsq_theta_W - 3) / 12)\
                   + (stop_num / (np.power(m_stop_2, 2) - np.power(m_stop_1, 2)))) * F(m_stop_2, Q_renorm)
    return Sigmauu_stop2
    
def sigmadd_stop2(vHiggs, mu, tanb, y_t, g, g_prime, sinsq_theta_W, m_stop_1, m_stop_2, mtL, mtR, a_t, Q_renorm):
    delta_uL = ((1 / 2) - (2 / 3) * sinsq_theta_W) * ((np.power(g, 2) + np.power (g_prime, 2)) / 2) * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    delta_uR = (2 / 3) * sinsq_theta_W * ((np.power(g, 2) + np.power (g_prime, 2)) / 2) * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    stop_num = ((np.power(mtL, 2) - np.power(mtR, 2) + delta_uL - delta_uR) \
                * ((1 / 2) + (4 / 3) * sinsq_theta_W) * (np.power(g, 2) + np.power(g_prime, 2)) / 2)\
                + 2 * np.power(y_t, 2) * np.power(mu, 2)
    Sigmadd_stop = (3 / (32 * (np.power(np.pi, 2)))) * (((np.power(g, 2) + np.power(g_prime, 2)) / 2) - (stop_num / (np.power(mt_stop_2, 2) - np.power(mt_stop_1, 2)))) * F(m_stop_2, Q_renorm)
    return Sigmadd_stop

#########################
# Sbottom squarks:
#########################

def sigmauu_sbottom1(vHiggs, mu, tanb, y_b, g, g_prime, sinsq_theta_W, m_sbot_1, m_sbot_2, mbL, mbR, a_b, Q_renorm):
    delta_dL = ((-1 / 2) + (1 / 3) * sinsq_theta_W) * ((np.power(g, 2) + np.power (g_prime, 2)) / 2) * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    delta_dR = ((-1 / 3) * sinsq_theta_W) * ((np.power(g, 2) + np.power (g_prime, 2)) / 2) * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    sbot_num = (np.power(mbR, 2) - np.power(mbL, 2) + delta_dR - delta_dL) * ((np.power(g, 2) + np.power(g_prime, 2)) / 2) \
                * ((1 / 2) + (2 / 3) * sinsq_theta_W) \
                + 2 * np.power(y_b, 2) * np.power(mu, 2)
    Sigmauu_sbot = (3 / (32 * np.power(np.pi, 2))) * (((np.power(g, 2) + np.power(g_prime, 2)) / 4) - (sbot_num / (np.power(m_sbot_2, 2) - np.power(m_sbot_1, 2)))) * F(m_sbot_1, Q_renorm)
    return Sigmauu_sbot

def sigmauu_sbottom2(vHiggs, mu, tanb, y_b, g, g_prime, sinsq_theta_W, m_sbot_1, m_sbot_2, mbL, mbR, a_b, Q_renorm):
    delta_dL = ((-1 / 2) + (1 / 3) * sinsq_theta_W) * ((np.power(g, 2) + np.power (g_prime, 2)) / 2) * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    delta_dR = ((-1 / 3) * sinsq_theta_W) * ((np.power(g, 2) + np.power (g_prime, 2)) / 2) * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    sbot_num = (np.power(mbR, 2) - np.power(mbL, 2) + delta_dR - delta_dL) * ((np.power(g, 2) + np.power(g_prime, 2)) / 2) \
                * ((1 / 2) + (2 / 3) * sinsq_theta_W) \
                + 2 * np.power(y_b, 2) * np.power(mu, 2)
    Sigmauu_sbot = (3 / (32 * np.power(np.pi, 2))) * (((np.power(g, 2) + np.power(g_prime, 2)) / 4) + (sbot_num / (np.power(m_sbot_2, 2) - np.power(m_sbot_1, 2)))) * F(m_sbot_2, Q_renorm)
    return Sigmauu_sbot
    
def sigmadd_sbottom1(vHiggs, mu, tanb, y_b, g, g_prime, sinsq_theta_W, m_sbot_1, m_sbot_2, mbL, mbR, a_b, Q_renorm):
    delta_dL = ((-1 / 2) + (1 / 3) * sinsq_theta_W) * ((np.power(g, 2) + np.power (g_prime, 2)) / 2) * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    delta_dR = ((-1 / 3) * sinsq_theta_W) * ((np.power(g, 2) + np.power (g_prime, 2)) / 2) * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    sbot_num = (np.power(mbR, 2) - np.power(mbL, 2) + delta_dR - delta_dL) * ((np.power(g, 2) + np.power(g_prime, 2)) / 2) \
                * ((-1 / 2) - (2 / 3) * sinsq_theta_W) \
                + 2 * np.power(y_b, 2) * np.power(mu, 2)
    Sigmauu_sbot = (3 / (32 * np.power(np.pi, 2))) * (((-1) * (np.power(g, 2) + np.power(g_prime, 2)) / 4) - (sbot_num / (np.power(m_sbot_2, 2) - np.power(m_sbot_1, 2)))) * F(m_sbot_1, Q_renorm)
    return Sigmauu_sbot

def sigmadd_sbottom2(vHiggs, mu, tanb, y_b, g, g_prime, sinsq_theta_W, m_sbot_1, m_sbot_2, mbL, mbR, a_b, Q_renorm):
    delta_dL = ((-1 / 2) + (1 / 3) * sinsq_theta_W) * ((np.power(g, 2) + np.power (g_prime, 2)) / 2) * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    delta_dR = ((-1 / 3) * sinsq_theta_W) * ((np.power(g, 2) + np.power (g_prime, 2)) / 2) * (np.power(vd(vHiggs, tanb), 2) - np.power(vu(vHiggs, tanb), 2))
    sbot_num = (np.power(mbR, 2) - np.power(mbL, 2) + delta_dR - delta_dL) * ((np.power(g, 2) + np.power(g_prime, 2)) / 2) \
                * ((-1 / 2) - (2 / 3) * sinsq_theta_W) \
                + 2 * np.power(y_b, 2) * np.power(mu, 2)
    Sigmauu_sbot = (3 / (32 * np.power(np.pi, 2))) * (((-1) * (np.power(g, 2) + np.power(g_prime, 2)) / 4) + (sbot_num / (np.power(m_sbot_2, 2) - np.power(m_sbot_1, 2)))) * F(m_sbot_2, Q_renorm)
    return Sigmauu_sbot
    

#########################
# Stau sleptons:
#########################

def sigmauu_stau():
    
    
def sigmadd_stau():
    

#########################
# Sfermions, 1st gen:
#########################

def sigmauu_sup_L():
    

def sigmauu_sup_R():
    
    
def sigmauu_sdown_L():
    
    
def sigmauu_sdown_R():
    
    
def sigmauu_selec_L():
    
    
def sigmauu_selec_R():
    
    
def sigmauu_selecSneut():
    
    
def sigmadd_sup_L():
    

def sigmadd_sup_R():
    
    
def sigmadd_sdown_L():
    
    
def sigmadd_sdown_R():
    
    
def sigmadd_selec_L():
    
    
def sigmadd_selec_R():
    
    
def sigmadd_selecSneut():
    
    
#########################
# Sfermions, 2nd gen:
#########################

def sigmauu_sstrange_L():
    

def sigmauu_sstrange_R():
    
    
def sigmauu_scharm_L():
    
    
def sigmauu_scharm_R():
    
    
def sigmauu_smu_L():
    
    
def sigmauu_smu_R():
    
    
def sigmauu_smuSneut():
    
    
def sigmadd_sstrange_L():
    

def sigmadd_sstrange_R():
    
    
def sigmadd_scharm_L():
    
    
def sigmadd_scharm_R():
    
    
def sigmadd_smu_L():
    
    
def sigmadd_smu_R():
    
    
def sigmadd_smuSneut():
    
    
#########################
# Neutralinos:
#########################
# Set up terms from characteristic polynomial for eigenvalues x of squared neutralino mass matrix,
# x^4 + b(vu, vd) * x^3 + c(vu, vd) * x^2 + d(vu, vd) * x + e(vu, vd) = 0
def neutralinouu_deriv_num(M1, M2, mu, g, g_prime, vHiggs, tanb, msN):
    cubicterm = np.power(g, 2) + np.power(g_prime, 2)
    quadrterm = (((np.power(g, 2) * M2 * mu) + (np.power(g_prime, 2) * M1 * mu)) / (tanb)) \
                - ((np.power(g, 2) * np.power(M1, 2))\
                   + (np.power(g_prime, 2) * np.power(M2, 2))\
                   + ((np.power(g, 2) + np.power(g_prime, 2)) * (np.power(mu, 2)))\
                   + (np.power((np.power(g, 2) + np.power(g_prime, 2)), 2) / 2) * np.power(vHiggs, 2))
    linterm = (((-1) * mu) * ((np.power(g, 2) * M2 * (np.power(M1, 2) + np.power(mu, 2)))\
                  + np.power(g_prime, 2) * M1 * (np.power(M2, 2) + np.power(mu, 2))) / tanb)\
            + ((np.power((np.power(g, 2) * M1 + np.power(g_prime, 2) * M2), 2) / 2) * np.power(vHiggs, 2))\
            + (np.power(mu, 2) * (np.power(g, 2) * np.power(M1, 2) \
                  + np.power(g_prime, 2) * np.power(M2, 2)))\
            + (np.power((np.power(g, 2) + np.power(g_prime, 2)), 2) * np.power(vHiggs, 2) * np.power(mu, 2) *cossqb(tanb))\
    constterm = (M1 * M2 * (np.power(g, 2) * M1 + np.power(g_prime, 2) * M2) * np.power(mu, 3) / tanb) \
            - (np.power((np.power(g, 2) * M1 + np.power(g_prime, 2) * M2), 2) * np.power(vHiggs, 2) * np.power(mu, 2) * cossqb(tanb))
    mynum = (cubicterm * np.power(msN, 6)) + (quadrterm * np.power(msN, 4)) + (linterm * np.power(msN, 2)) + constterm
    return mynum

def neutralinodd_deriv_num(M1, M2, mu, g, g_prime, vHiggs, tanb, msN):
    cubicterm = np.power(g, 2) + np.power(g_prime, 2)
    quadrterm = (((np.power(g, 2) * M2 * mu) + (np.power(g_prime, 2) * M1 * mu)) * (tanb)) \
                - ((np.power(g, 2) * np.power(M1, 2))\
                   + (np.power(g_prime, 2) * np.power(M2, 2))\
                   + ((np.power(g, 2) + np.power(g_prime, 2)) * (np.power(mu, 2)))\
                   + (np.power((np.power(g, 2) + np.power(g_prime, 2)), 2) / 2) * np.power(vHiggs, 2))
    linterm = (((-1) * mu) * ((np.power(g, 2) * M2 * (np.power(M1, 2) + np.power(mu, 2)))\
                  + np.power(g_prime, 2) * M1 * (np.power(M2, 2) + np.power(mu, 2))) * tanb)\
            + ((np.power((np.power(g, 2) * M1 + np.power(g_prime, 2) * M2), 2) / 2) * np.power(vHiggs, 2))\
            + (np.power(mu, 2) * (np.power(g, 2) * np.power(M1, 2) \
                  + np.power(g_prime, 2) * np.power(M2, 2)))\
            + (np.power((np.power(g, 2) + np.power(g_prime, 2)), 2) * np.power(vHiggs, 2) * np.power(mu, 2) *sinsqb(tanb))\
    constterm = (M1 * M2 * (np.power(g, 2) * M1 + np.power(g_prime, 2) * M2) * np.power(mu, 3) * tanb) \
            - (np.power((np.power(g, 2) * M1 + np.power(g_prime, 2) * M2), 2) * np.power(vHiggs, 2) * np.power(mu, 2) * sinsqb(tanb))
    mynum = (cubicterm * np.power(msN, 6)) + (quadrterm * np.power(msN, 4)) + (linterm * np.power(msN, 2)) + constterm
    return mynum

def neutralino_deriv_denom(M1, M2, mu, g, g_prime, vHiggs, tanb, msN):
    quadrterm = -3 * ((np.power(M1, 2))\
              + (np.power(M2, 2))\
              + ((np.power(g, 2) + np.power(g_prime, 2)) * np.power(vHiggs, 2))\
              + (2 * np.power(mu, 2)))
    linterm = (np.power(vHiggs, 4) * np.power((np.power(g, 2) + np.power(g_prime, 2)), 2) / 2)\
            + (np.power(vHiggs, 2) * (2 * ((np.power(g, 2) * np.power(M1, 2))\
               + (np.power(g_prime, 2) * np.power(M2, 2)) \
               + ((np.power(g, 2) + np.power(g_prime, 2)) * np.power(mu, 2)) \
               - (mu * (np.power(g_prime, 2) * M1 + np.power(g, 2) * M2) * 2 \
                  * np.sqrt(sinsqb(tanb)) * np.sqrt(cossq(tanb)))))) \
            + (2 * ((np.power(M1, 2) * np.power(M2, 2)) \
                + (2 * (np.power(M1, 2) + np.power(M2, 2)) * np.power(mu, 2)) \
                + (np.power(mu, 4))))
    constterm = (np.power(vHiggs, 4) * (1 / 8) \
                     * ((np.power((np.power(g, 2) + np.power(g_prime, 2)), 2) * np.power(mu, 2) \
                       * (np.power(cossqb(tanb), 2) - (6 * cossqb(tanb) * sinsqb(tanb)) + np.power(sinsqb(tanb), 2))) \
                    - (2 * np.power((np.power(g, 2) * M1 + np.power(g_prime, 2) * M2), 2)) \
                    - (np.power(mu, 2) * np.power((np.power(g, 2) + np.power(g_prime, 2)), 2))))\
                + (np.power(vHiggs, 2) * 2 * mu \
                   * ((np.sqrt(cossqb(tanb)) * np.sqrt(sinsqb(tanb)))\
                      * (np.power(g, 2) * M2 * (np.power(M1, 2) + np.power(mu, 2)) + np.power(g_prime, 2) * M1 * (np.power(M2, 2) + np.power(mu, 2)))))
               - ((2 * np.power(M2, 2) * np.power(M1, 2) * np.power(mu, 2)) \
                  + (np.power(mu, 4) * (np.power(M1, 2) + np.power(M2, 2))))
    mydenom = 4 * np.power(msN, 6) + quadrterm * np.power(msN, 4) + linterm * np.power(msN, 2) + constterm
    return mydenom

def sigmauu_neutralino(M1, M2, mu, g, g_prime, vHiggs, tanb, msN, Q_renorm):
    Sigmauu_neutralino = ((-1) / (16 * np.power(np.pi, 2))) \
                          * (neutralinouu_deriv_num(M1, M2, mu, g, g_prime, vHiggs, tanb, msN) / neutralino_deriv_denom(M1, M2, mu, g, g_prime, vHiggs, tanb, msN))\
                          * F(msN, Q_renorm)
    return Sigmauu_neutralino
                          

def sigmadd_neutralino(M1, M2, mu, g, g_prime, vHiggs, tanb, msN, Q_renorm):
    Sigmadd_neutralino = ((-1) / (16 * np.power(np.pi, 2))) \
                          * (neutralinodd_deriv_num(M1, M2, mu, g, g_prime, vHiggs, tanb, msN) / neutralino_deriv_denom(M1, M2, mu, g, g_prime, vHiggs, tanb, msN))\
                          * F(msN, Q_renorm)
    return Sigmadd_neutralino
    
     
#########################
# Charginos:
#########################

def sigmauu_chargino1(g, M2, vHiggs, tanb, mu, msC, Q_renorm):
    chargino_num = np.power(M2, 2) + np.power(mu, 2) + (np.power(g, 2) * (np.power(vu(vHiggs, tanb), 2) - np.power(vd(vHiggs, tanb), 2)))
    chargino_den = np.sqrt(((np.power(g, 2) * np.power((vu(vHiggs, tanb) + vd(vHiggs, tanb)), 2)) + np.power((M2 - mu), 2)) * ((np.power(g, 2) * np.power((vd(vHiggs, tanb) - vu(vHiggs, tanb)), 2)) + np.power((M2 + mu), 2)))
    Sigmauu_chargino1 = -1 * (np.power(g, 2) / (16 * np.power(np.pi, 2))) *(1 - (chargino_num / chargino_den)) * F(msC, Q_renorm)
    return Sigmauu_chargino1

def sigmauu_chargino2(g, M2, vHiggs, tanb, mu, msC, Q_renorm):
    chargino_num = np.power(M2, 2) + np.power(mu, 2) + (np.power(g, 2) * (np.power(vu(vHiggs, tanb), 2) - np.power(vd(vHiggs, tanb), 2)))
    chargino_den = np.sqrt(((np.power(g, 2) * np.power((vu(vHiggs, tanb) + vd(vHiggs, tanb)), 2)) + np.power((M2 - mu), 2)) * ((np.power(g, 2) * np.power((vd(vHiggs, tanb) - vu(vHiggs, tanb)), 2)) + np.power((M2 + mu), 2)))
    Sigmauu_chargino2 = -1 * (np.power(g, 2) / (16 * np.power(np.pi, 2))) *(1 + (chargino_num / chargino_den)) * F(msC, Q_renorm)
    return Sigmauu_chargino2

def sigmadd_chargino1(g, M2, vHiggs, tanb, mu, msC, Q_renorm):
    chargino_num = np.power(M2, 2) + np.power(mu, 2) - (np.power(g, 2) * (np.power(vu(vHiggs, tanb), 2) - np.power(vd(vHiggs, tanb), 2)))
    chargino_den = np.sqrt(((np.power(g, 2) * np.power((vu(vHiggs, tanb) + vd(vHiggs, tanb)), 2)) + np.power((M2 - mu), 2)) * ((np.power(g, 2) * np.power((vd(vHiggs, tanb) - vu(vHiggs, tanb)), 2)) + np.power((M2 + mu), 2)))
    Sigmadd_chargino1 = -1 * (np.power(g, 2) / (16 * np.power(np.pi, 2))) *(1 - (chargino_num / chargino_den)) * F(msC, Q_renorm)
    return Sigmadd_chargino1

def sigmadd_chargino2(g, M2, vHiggs, tanb, mu, msC, Q_renorm):
    chargino_num = np.power(M2, 2) + np.power(mu, 2) - (np.power(g, 2) * (np.power(vu(vHiggs, tanb), 2) - np.power(vd(vHiggs, tanb), 2)))
    chargino_den = np.sqrt(((np.power(g, 2) * np.power((vu(vHiggs, tanb) + vd(vHiggs, tanb)), 2)) + np.power((M2 - mu), 2)) * ((np.power(g, 2) * np.power((vd(vHiggs, tanb) - vu(vHiggs, tanb)), 2)) + np.power((M2 + mu), 2)))
    Sigmadd_chargino2 = -1 * (np.power(g, 2) / (16 * np.power(np.pi, 2))) *(1 + (chargino_num / chargino_den)) * F(msC, Q_renorm)
    return Sigmadd_chargino2
    

#########################
# Higgs bosons (sigmauu = sigmadd here):
#########################

def sigmauu_h0(g, g_prime, vHiggs, tanb, mHu, mHd, mu, mZ, mh0, Q_renorm):
    mynum = ((np.power(g, 2) + np.power(g_prime, 2)) * np.power(vHiggs, 2)) - (2 * mA0sq(mu, mHu, mHd) * (np.power(cossqb(tanb), 2) - 6 * cossqb(tanb) * sinsqb(tanb) + nppower(sinsqb(tanb), 2))) 
    myden = np.sqrt(np.power((mA0sq(mu, mHu, mHd) - np.power(mZ, 2)), 2) + 4 * np.power(mZ, 2) * mA0sq(mu, mHu, mHd)* 4 * cossqb(tanb) * sinsqb(tanb))
    Sigmauu_h0 = (1/(32 * np.power(np.pi, 2))) * ((np.power(g, 2) + np.power(g_prime, 2)) / 4) * (1 - (mynum / myden)) * F(mh0, Q_renorm)
    return Sigmauu_h0
    
def sigmauu_H0(g, g_prime, vHiggs, tanb, mHu, mHd, mu, mZ, mh0, Q_renorm):
    mynum = ((np.power(g, 2) + np.power(g_prime, 2)) * np.power(vHiggs, 2)) - (2 * mA0sq(mu, mHu, mHd) * (np.power(cossqb(tanb), 2) - 6 * cossqb(tanb) * sinsqb(tanb) + nppower(sinsqb(tanb), 2))) 
    myden = np.sqrt(np.power((mA0sq(mu, mHu, mHd) - np.power(mZ, 2)), 2) + 4 * np.power(mZ, 2) * mA0sq(mu, mHu, mHd)* 4 * cossqb(tanb) * sinsqb(tanb))
    Sigmauu_H0 = (1/(32 * np.power(np.pi, 2))) * ((np.power(g, 2) + np.power(g_prime, 2)) / 4) * (1 + (mynum / myden)) * F(mh0, Q_renorm)
    return Sigmauu_H0
    
def sigmauu_H_pm(g, mH_pm, Q_renorm):
    Sigmauu_H_pm = (np.power(g, 2) / (64 * np.power(np.pi, 2))) * F(mH_pm, Q_renorm)
    return Sigmauu_H_pm

#########################
# Weak bosons (sigmauu = sigmadd here):
#########################

def sigmauu_W_pm(g, vHiggs, Q_renorm):
    mymWsq = mWsq(g, vHiggs)
    Sigmauu_W_pm = (3 * np.power(g, 2) / (32 * np.power(np.pi, 2))) * F(np.sqrt(mymWsq), Q_renorm)
    return Sigmauu_W_pm

def sigmauu_Z0(g, g_prime, vHiggs, Q_renorm):
    mymZsq = mZsq(g, g_prime, vHiggs)
    Sigmauu_W_pm = (3 * np.power(g, 2) / (64 * np.power(np.pi, 2))) * F(np.sqrt(mymZsq), Q_renorm)
    return Sigmauu_W_pm

#########################
# SM fermions (sigmadd_t = sigmauu_b = sigmauu_tau = 0):
#########################

def sigmauu_top(yt, vHiggs, tanb, Q_renorm):
    mymt = yt * vu(vHiggs, tanb)
    Sigmauu_top = (-1 * np.power(yt, 2) / (16 * np.power(np.pi, 2))) * F(mymt, Q_renorm)
    return Sigmauu_top

def sigmadd_top(yt, vHiggs, tanb, Q_renorm):
    return 0

def sigmauu_bottom(yb, vHiggs, tanb, Q_renorm):
    return 0

def sigmadd_bottom(yb, vHiggs, tanb, Q_renorm):
    mymb = yb * vd(vHiggs, tanb)
    Sigmadd_bottom = (-1 * np.power(yb, 2) / (16 * np.power(np.pi, 2))) * F(mymb, Q_renorm)
    return Sigmadd_bottom

def sigmauu_tau(ytau, vHiggs, tanb, Q_renorm):
    return 0

def sigmadd_tau(ytau, vHiggs, tanb, Q_renorm):
    mymtau = ytau * vd(vHiggs, tanb)
    Sigmadd_tau = (-1 * np.power(ytau, 2) / (16 * np.power(np.pi, 2))) * F(mymtau, Q_renorm)
    return Sigmadd_tau

#########################
# Sigmauu computation
#########################

def sigmauu_net():
    

#########################
# Sigmadd computation
#########################

def sigmadd_net():
    

#########################
# DEW computation
#########################

def DEW():
    
