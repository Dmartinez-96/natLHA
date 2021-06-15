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

def mHpmsq(msq_A0, g, vHiggs):
    mymWsq = mWsq(g, vHiggs)
    mymHpmsq = msq_A0 + mymWsq
    return mymHpmsq
    
#########################
# Fundamental log equation
#########################

def F(msq, Q_renormSq):
    myF = msq * (np.log(msq / Q_renormSq) - 1)
    return myF

#########################
# Stop squarks:
#########################

def sigmauu_stop(vHiggs, tanb, y_t, g, g_prime, theta_W, m_sqrd_stopGauge, m_stop_sq, a_t, Q_renormSq):
    Sigmauu_stop  = (1/(16 * (np.pi)**2))*F(m_stop_sq, Q_renormSq)
    return Sigmauu_stop
    
#########################
# Sbottom squarks:
#########################

def sigmauu_sbottom():
    
    
def sigmadd_sbottom():
    

#########################
# Stau sleptons:
#########################

def sigmauu_stau():
    
    
def sigmadd_stau():
    

#########################
# Sfermions:
#########################



#########################
# Neutralinos:
#########################



#########################
# Charginos:
#########################



#########################
# Higgs bosons:
#########################



#########################
# Weak bosons (sigmauu = sigmadd for both below):
#########################

def sigmauu_W_pm(g, vHiggs, Q_renormSq):
    mymWsq = mWsq(g, vHiggs)
    Sigmauu_W_pm = (3 * np.power(g, 2) / (32 * np.power(np.pi, 2))) * F(mymWsq, Q_renormSq)
    return Sigmauu_W_pm

def sigmauu_Z0(g, g_prime, vHiggs, Q_renormSq):
    mymZsq = mZsq(g, g_prime, vHiggs)
    Sigmauu_W_pm = (3 * np.power(g, 2) / (64 * np.power(np.pi, 2))) * F(mymZsq, Q_renormSq)
    return Sigmauu_W_pm

#########################
# SM fermions (sigmadd_t = sigmauu_b = sigmauu_tau = 0):
#########################

def sigmauu_top(yt, vHiggs, tanb, Q_renormSq):
    mymt_sq = np.power(yt, 2) * np.power(vHiggs, 2) * (np.power(tanb, 2)/(1 + np.power(tanb, 2)))
    Sigmauu_top = (-1 * np.power(yt, 2) / (16 * np.power(np.pi, 2))) * F(mymt_sq, Q_renormSq)
    return Sigmauu_top

def sigmadd_bottom(yb, vHiggs, tanb, Q_renormSq):
    mymb_sq = np.power(yb, 2) * np.power(vHiggs, 2) * (1 - (np.power(tanb, 2)/(1 + np.power(tanb, 2))))
    Sigmadd_bottom = (-1 * np.power(yb, 2) / (16 * np.power(np.pi, 2))) * F(mymb_sq, Q_renormSq)
    return Sigmadd_bottom

def sigmadd_tau(ytau, vHiggs, tanb, Q_renormSq):
    mymtau_sq = np.power(ytau, 2) * np.power(vHiggs, 2) * (1 - (np.power(tanb, 2)/(1 + np.power(tanb, 2))))
    Sigmadd_tau = (-1 * np.power(ytau, 2) / (16 * np.power(np.pi, 2))) * F(mymtau_sq, Q_renormSq)
    return Sigmadd_tau

#########################
# Sigmauu computation
#########################



#########################
# Sigmadd computation
#########################



#########################
# DEW computation
#########################

