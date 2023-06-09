#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 14:33:53 2023

Delta_HS calculator.

@author: Dakotah Martinez
"""

import numpy as np

def Delta_HS_calc(mHdsq_Lambda, delta_mHdsq, mHusq_Lambda, delta_mHusq,
                  mu_Lambdasq, delta_musq, running_mz_sq, tanb_sq, sigmauutot,
                  sigmaddtot):
    """
    Compute the fine-tuning measure Delta_HS.

    Parameters
    ----------
    mHdsq_Lambda : Float.
        mHd^2(GUT).
    delta_mHdsq : Float.
        RGE running of mHd^2 down to 2 TeV.
    mHusq_Lambda : Float.
        mHu^2(GUT).
    delta_mHusq : Float.
        RGE running of mHu^2 down to 2 TeV.
    mu_Lambdasq : Float.
        mu^2(GUT).
    delta_musq : Float.
        RGE running of mu^2 down to 2 TeV.
    running_mz_sq : Float.
        Running mZ^2, evaluated at 2 TeV.
    tanb_sq : Float.
        tan^2(beta), evaluated at 2 TeV.
    sigmauutot : Float.
        Up-type radiative corrections evaluated at 2 TeV.
    sigmaddtot : Float.
        Down-type radiative corrections evaluated at 2 TeV.

    Returns
    -------
    Delta_HS : Float.
        Fine-tuning measure Delta_HS.

    """
    B_Hd = mHdsq_Lambda / (tanb_sq - 1)
    B_deltaHd = delta_mHdsq / (tanb_sq - 1)
    B_Hu = mHusq_Lambda * tanb_sq / (tanb_sq - 1)
    B_deltaHu = delta_mHusq * tanb_sq / (tanb_sq - 1)
    B_Sigmadd = sigmaddtot / (tanb_sq - 1)
    B_Sigmauu = sigmauutot * tanb_sq / (tanb_sq - 1)
    B_muLambdasq = mu_Lambdasq
    B_deltamusq = delta_musq
    Delta_HS_contribs = np.sort(np.array([(np.abs(B_Hd) / (running_mz_sq / 2),
                                           'Delta_HS(mHd^2(GUT))'),
                                          (np.abs(B_deltaHd)
                                           / (running_mz_sq / 2),
                                           'Delta_HS(delta(mHd^2))'),
                                          (np.abs(B_Hu)/ (running_mz_sq / 2),
                                           'Delta_HS(mHu^2(GUT))'),
                                          (np.abs(B_deltaHu)
                                           / (running_mz_sq / 2),
                                           'Delta_HS(delta(mHu^2))'),
                                          (np.abs(B_Sigmadd)
                                           / (running_mz_sq / 2),
                                           'Delta_HS(Sigma_d^d)'),
                                          (np.abs(B_Sigmauu)
                                           / (running_mz_sq / 2),
                                           'Delta_HS(Sigma_u^u)'),
                                          (np.abs(B_muLambdasq)
                                           / (running_mz_sq / 2),
                                           'Delta_HS(mu(GUT))'),
                                          (np.abs(B_deltamusq)
                                           / (running_mz_sq / 2),
                                           'Delta_HS(delta(mu))')],
                                         dtype=[('HSContrib', float),
                                                ('HSlabel', 'U40')]),
                                order='HSContrib')
    return Delta_HS_contribs[::-1]
