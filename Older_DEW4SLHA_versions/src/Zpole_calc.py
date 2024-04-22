#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 14:04:43 2023

Various functions used in Z pole mass calculations.

@author: Dakotah Martinez
"""

from mpmath import mp, mpf

def A0func(inpmsq, inpQ):
    """
    Compute the Passarino-Veltman function A_0 for some input mass and
        renormalization scale.

    Parameters
    ----------
    inpmsq : Float.
        Squared mass input for evaluation.
    inpQ : Float.
        Renormalization scale for evaluation.

    Returns
    -------
    PVA0 : Float.
        Return result of Passarino-Veltman function A_0(m^2, Q).

    """
    PVA0 = (inpmsq * (1 - mp.log(inpmsq / mp.power(inpQ, 2))))
    return PVA0

def B0func(inpm1sq, inpm2sq, inpQ, varmZsq):
    """
    Compute the Passarino-Veltman function B_0 for some input masses, mZ pole,
        and renormalization scale.

    Parameters
    ----------
    inpm1sq : Float.
        Squared mass m_1^2 input for evaluation.
    inpm2sq : Float.
        Squared mass m_2^2 input for evaluation.
    inpQ : Float.
        Renormalization scale for evaluation.
    varmZsq : Float.
        The value of mZ^2(pole) that will be used in the DBG calculation.

    Returns
    -------
    PVB0 : Float.
        Return result of Passarino-Veltman function B_0(m_1^2, m_2^2).

    """
    def fb(x):
        return (mp.log(1 - x + 0j) - (x * mp.log(1- (1 / x + 0j)) - 1))

    s = varmZsq - inpm2sq + inpm1sq
    ssq = s ** 2
    xp = ((1 / (2 * varmZsq)) * (s + mp.sqrt(ssq - (4 * inpm1sq * varmZsq)
                                             + 0j)))
    xm = ((1 / (2 * varmZsq)) * (s - mp.sqrt(ssq - (4 * inpm1sq * varmZsq)
                                             + 0j)))
    PVB0 = ((-1) * mp.log(varmZsq / mp.power(inpQ, 2) + 0j))\
        - fb(xp) - fb(xm)
    return PVB0

def B1func(inpm1sq, inpm2sq, inpQ, varmZsq):
    """
    Compute the Passarino-Veltman function B_1 for some input masses, mZ pole,
        and renormalization scale.

    Parameters
    ----------
    inpm1sq : Float.
        Squared mass m_1^2 input for evaluation.
    inpm2sq : Float.
        Squared mass m_2^2 input for evaluation.
    inpQ : Float.
        Renormalization scale for evaluation.
    varmZsq : Float.
        The value of mZ^2(pole) that will be used in the DBG calculation.

    Returns
    -------
    PVB1 : Float.
        Return result of Passarino-Veltman function B_1(m_1^2, m_2^2).

    """
    PVB1 = (1 / (2 * varmZsq))\
        * (A0func(inpm2sq, inpQ) - A0func(inpm1sq, inpQ)
           + ((varmZsq + inpm1sq - inpm2sq) * B0func(inpm1sq, inpm2sq,
                                                     inpQ, varmZsq)))
    return PVB1

def B22func(inpm1sq, inpm2sq, inpQ, varmZsq):
    """
    Compute the Passarino-Veltman function B_22 for some input masses, mZ pole,
        and renormalization scale.

    Parameters
    ----------
    inpm1sq : Float.
        Squared mass m_1^2 input for evaluation.
    inpm2sq : Float.
        Squared mass m_2^2 input for evaluation.
    inpQ : Float.
        Renormalization scale for evaluation.
    varmZsq : Float.
        The value of mZ^2(pole) that will be used in the DBG calculation.

    Returns
    -------
    PVB22 : Float.
        Return result of Passarino-Veltman function B_22(m_1^2, m_2^2).

    """
    PVB22 = (1 / 6)\
        * (((1 / 2) * (A0func(inpm2sq, inpQ) + A0func(inpm1sq, inpQ)))
           + ((inpm1sq + inpm2sq - (varmZsq / 2)) * B0func(inpm1sq, inpm2sq,
                                                           inpQ, varmZsq))
           + (((inpm2sq - inpm1sq) / (2 * varmZsq))
              * (A0func(inpm2sq, inpQ) - A0func(inpm1sq, inpQ)
                 - ((inpm2sq - inpm1sq) * B0func(inpm1sq, inpm2sq, inpQ,
                                                 varmZsq))))
           + inpm1sq + inpm2sq - (varmZsq / 3))
    return PVB22

def Gfunc(inpm1sq, inpm2sq, inpQ, varmZsq):
    """
    Compute the Passarino-Veltman function G for some input masses, mZ pole,
        and renormalization scale.

    Parameters
    ----------
    inpm1sq : Float.
        Squared mass m_1^2 input for evaluation.
    inpm2sq : Float.
        Squared mass m_2^2 input for evaluation.
    inpQ : Float.
        Renormalization scale for evaluation.
    varmZsq : Float.
        The value of mZ^2(pole) that will be used in the DBG calculation.

    Returns
    -------
    PVG : Float.
        Return result of Passarino-Veltman function G(m_1^2, m_2^2).

    """
    PVG = (((varmZsq - inpm1sq - inpm2sq) * B0func(inpm1sq, inpm2sq,
                                                   inpQ, varmZsq))
           - A0func(inpm1sq, inpQ) - A0func(inpm2sq, inpQ))
    return PVG

def Hfunc(inpm1sq, inpm2sq, inpQ, varmZsq):
    """
    Compute the Passarino-Veltman function H for some input masses, mZ pole,
        and renormalization scale.

    Parameters
    ----------
    inpm1sq : Float.
        Squared mass m_1^2 input for evaluation.
    inpm2sq : Float.
        Squared mass m_2^2 input for evaluation.
    inpQ : Float.
        Renormalization scale for evaluation.
    varmZsq : Float.
        The value of mZ^2(pole) that will be used in the DBG calculation.

    Returns
    -------
    PVH : Float.
        Return result of Passarino-Veltman function H(m_1^2, m_2^2).

    """
    PVH = (4 * B22func(inpm1sq, inpm2sq, inpQ, varmZsq))\
        + Gfunc(inpm1sq, inpm2sq, inpQ, varmZsq)
    return PVH

def sB22func(inpm1sq, inpm2sq, inpQ, varmZsq):
    """
    Compute the Passarino-Veltman function ~B_22 for some input masses,
        mZ pole, and renormalization scale.

    Parameters
    ----------
    inpm1sq : Float.
        Squared mass m_1^2 input for evaluation.
    inpm2sq : Float.
        Squared mass m_2^2 input for evaluation.
    inpQ : Float.
        Renormalization scale for evaluation.
    varmZsq : Float.
        The value of mZ^2(pole) that will be used in the DBG calculation.

    Returns
    -------
    PVsB22 : Float.
        Return result of Passarino-Veltman function ~B_22(m_1^2, m_2^2).

    """
    PVsB22 = B22func(inpm1sq, inpm2sq, inpQ, varmZsq)\
        - ((1 / 4) * (A0func(inpm1sq, inpQ) + A0func(inpm2sq, inpQ)))
    return PVsB22

def mZpolecalc(varmZsq, runmZsq, SLHAscale, otherpars):
    """
    Compute the one-loop pole mass M_Z^2 for evaluation of derivatives in DBG.
        Many terms come from the appendix of arXiv:hep-ph/9606211.

    Parameters
    ----------
    varmZsq : Float.
        Value of pole mass M_Z^2 to be solved for by scipy.optimize.fsolve.
    runmZsq : Float.
        Value of running mass m_Z^2 evaluated from tree-level condition in DBG
            calculation.
    SLHAscale : Float.
        Renormalization scale at which self-energy is being computed.
    otherpars : Array of floats.
        Array of other values used in mZ pole calculation from RGEs.

    Returns
    -------
    myroot : Float.
        Numerical solution in current iteration for pole mass M_Z^2.

    """
    varmZsq = mpf(float(varmZsq))
    rungpr = mp.sqrt(3 / 5.) * otherpars[0]
    rung2 = otherpars[1]
    sinthW = rungpr / mp.sqrt((rung2 ** 2) + (rungpr ** 2))
    sinthWsq = sinthW ** 2
    sin2thW = mp.sin(2 * mp.asin(sinthW))
    costhW = rung2 / mp.sqrt((rung2 ** 2) + (rungpr ** 2))
    costhWsq = costhW ** 2
    cos2thW = mp.cos(2 * mp.acos(costhW))

    # Weak neutral-current couplings and electric charges of left sparticles
    g_uL = mpf('0.5') - ((sinthW ** 2) * mpf('2') / 3)
    g_uLsq = g_uL ** 2
    e_uL = mpf('2.0') / 3.0
    g_uR = (sinthW ** 2) * mpf('2') / 3
    g_uRsq = g_uR ** 2
    g_dL = mpf('-0.5') + ((sinthW ** 2) / 3)
    g_dLsq = g_dL ** 2
    e_dL = mpf('-1.0') / 3.0
    g_dR = ((-1) * (sinthW ** 2) / 3)
    g_dRsq = g_dR ** 2
    g_sneut = mpf('0.5')
    g_eL = mpf('-0.5') + ((sinthW ** 2))
    g_eLsq = g_eL ** 2
    e_eL = (-1)
    g_eR = (-1) * (sinthW ** 2)
    g_eRsq = g_eR ** 2

    # SM pole masses
    mym_t = mpf('173.2')
    mym_c = mpf('1.27')
    mym_u = mpf('0.0024')
    mym_b = mpf('4.18')
    mym_s = mpf('0.104')
    mym_d = mpf('0.00475')
    mym_tau = mpf('1.777')
    mym_mu = mpf('0.105658357')
    mym_e = mpf('0.000510998902')

    # Set up soft terms from RGEs
    my_mQ3sq = otherpars[29]
    my_mQ2sq = otherpars[28]
    my_mQ1sq = otherpars[27]
    my_mL3sq = otherpars[32]
    my_mL2sq = otherpars[31]
    my_mL1sq = otherpars[30]
    my_mU3sq = otherpars[35]
    my_mU2sq = otherpars[34]
    my_mU1sq = otherpars[33]
    my_mD3sq = otherpars[38]
    my_mD2sq = otherpars[37]
    my_mD1sq = otherpars[36]
    my_mE3sq = otherpars[41]
    my_mE2sq = otherpars[40]
    my_mE1sq = otherpars[39]
    my_At = otherpars[16] / otherpars[7]
    my_Ac = otherpars[17] / otherpars[8]
    my_Au = otherpars[18] / otherpars[9]
    my_Ab = otherpars[19] / otherpars[10]
    my_As = otherpars[20] / otherpars[11]
    my_Ad = otherpars[21] / otherpars[12]
    my_Atau = otherpars[22] / otherpars[13]
    my_Amu = otherpars[23] / otherpars[14]
    my_Ae = otherpars[24] / otherpars[15]
    my_mu = mpmath.sqrt(str(abs(otherpars[6])))

    # Terms involving Higgs mixing angles
    my_tan_beta = otherpars[43]
    testbeta = mp.atan(my_tan_beta)
    salphbet = mp.sin(myalpha - testbeta)
    calphbet = mp.cos(myalpha - testbeta)
    myc2b = mp.cos(2 * testbeta)

    # Neutralino mass matrix and diagonalizing unitary matrix
    my_neutmat = mp.matrix([[str(otherpars[3]), str(0),
                             str((-1) * mp.sqrt(varmZsq)
                                 * mp.cos(testbeta) * sinthW),
                             str(mp.sqrt(varmZsq)
                                 * mp.sin(testbeta) * sinthW)],
                            [str(0), str(otherpars[4]),
                             str(mp.sqrt(varmZsq)
                                 * mp.cos(testbeta) * costhW),
                             str((-1) * mp.sqrt(varmZsq)
                                 * mp.sin(testbeta) * costhW)],
                            [str((-1) * mp.sqrt(varmZsq)
                                 * mp.cos(testbeta) * sinthW),
                             str(mp.sqrt(varmZsq)
                                 * mp.cos(testbeta) * costhW), str(0),
                             str(my_mu)],
                            [str(mp.sqrt(varmZsq)
                                 * mp.sin(testbeta) * sinthW),
                             str((-1) * mp.sqrt(varmZsq)
                                 * mp.sin(testbeta) * costhW),
                             str(my_mu), str(0)]])
    neutE, neutQ = mp.eigh(my_neutmat)
    N11 = mp.conj(neutQ[0,0])
    N12 = mp.conj(neutQ[0,1])
    N13 = mp.conj(neutQ[0,2])
    N14 = mp.conj(neutQ[0,3])
    N21 = mp.conj(neutQ[1,0])
    N22 = mp.conj(neutQ[1,1])
    N23 = mp.conj(neutQ[1,2])
    N24 = mp.conj(neutQ[1,3])
    N31 = mp.conj(neutQ[2,0])
    N32 = mp.conj(neutQ[2,1])
    N33 = mp.conj(neutQ[2,2])
    N34 = mp.conj(neutQ[2,3])
    N41 = mp.conj(neutQ[3,0])
    N42 = mp.conj(neutQ[3,1])
    N43 = mp.conj(neutQ[3,2])
    N44 = mp.conj(neutQ[3,3])

    # Chargino mass matrix and diagonalizing unitary matrix
    my_chargmat = mp.matrix([[str(otherpars[4]),
                              str(mp.sqrt(2) * MWpole * mp.sin(testbeta))],
                             [str(mp.sqrt(2) * MWpole * mp.cos(testbeta)),
                              str((-1) * my_mu)]])
    my_chargmatconj = mp.matrix([[str(mp.conj(otherpars[4])),
                                  str(mp.conj(mp.sqrt(2)
                                              * MWpole * mp.sin(testbeta)))],
                                 [str(mp.conj(mp.sqrt(2)
                                              * MWpole * mp.cos(testbeta))),
                                  str(mp.conj((-1) * my_mu))]])
    my_chargmatT = mp.matrix([[str(otherpars[4]),
                               str(mp.sqrt(2) * MWpole * mp.cos(testbeta))],
                              [str(mp.sqrt(2) * MWpole * mp.sin(testbeta)),
                               str((-1) * my_mu)]])
    my_chargmatdag = mp.matrix([[str(mp.conj(otherpars[4])),
                                 str(mp.conj(mp.sqrt(2)
                                             * MWpole * mp.cos(testbeta)))],
                                [str(mp.conj(mp.sqrt(2)
                                             * MWpole * mp.sin(testbeta))),
                                 str(mp.conj((-1) * my_mu))]])
    Umatsource = my_chargmatconj * my_chargmatT
    Vmatsource = my_chargmatdag * my_chargmat
    chargE1, chargQ1 = mp.eigh(Umatsource)
    chargE2, chargQ2 = mp.eigh(Vmatsource)
    U11 = mp.conj(chargQ1[0,0])
    U12 = mp.conj(chargQ1[0,1])
    U21 = mp.conj(chargQ1[1,0])
    U22 = mp.conj(chargQ1[1,1])
    V11 = mp.conj(chargQ2[0,0])
    V12 = mp.conj(chargQ2[0,1])
    V21 = mp.conj(chargQ2[1,0])
    V22 = mp.conj(chargQ2[1,1])

    # Set up a- and b-type couplings for neutralinos/charginos
    # Neutralino-neutralino-Z couplings (all not listed below are zero)
    apsi03psi03Z = rung2 / (2 * costhW)
    apsi04psi04Z = (-1) * apsi03psi03Z
    bpsi03psi03Z = apsi04psi04Z
    bpsi04psi04Z = apsi03psi03Z
    # Chargino-chargino-Z couplings (all not listed below are zero)
    apsip1psip1Z = rung2 * costhW
    apsip2psip2Z = rung2 * cos2thW / (2 * costhW)
    bpsip1psip1Z = apsip1psip1Z
    bpsip2psip2Z = apsip2psip2Z
    # Rotate to mass eigenstate basis
    achi01chi01Z = (mp.conj(N13) * N13 * apsi03psi03Z)\
        + (mp.conj(N14) * N14 * apsi04psi04Z)
    bchi01chi01Z = ((-1) * N13 * mp.conj(N13) * apsi03psi03Z)\
        - (N14 * mp.conj(N14) * apsi04psi04Z)
    achi01chi02Z = (mp.conj(N13) * N23 * apsi03psi03Z)\
        + (mp.conj(N14) * N24 * apsi04psi04Z)
    bchi01chi02Z = ((-1) * N13 * mp.conj(N23) * apsi03psi03Z)\
        - (N14 * mp.conj(N24) * apsi04psi04Z)
    achi01chi03Z = (mp.conj(N13) * N33 * apsi03psi03Z)\
        + (mp.conj(N14) * N34 * apsi04psi04Z)
    bchi01chi03Z = ((-1) * N13 * mp.conj(N33) * apsi03psi03Z)\
        - (N14 * mp.conj(N34) * apsi04psi04Z)
    achi01chi04Z = (mp.conj(N13) * N43 * apsi03psi03Z)\
        + (mp.conj(N14) * N44 * apsi04psi04Z)
    bchi01chi04Z = ((-1) * N13 * mp.conj(N43) * apsi03psi03Z)\
        - (N14 * mp.conj(N44) * apsi04psi04Z)
    achi02chi01Z = (mp.conj(N23) * N13 * apsi03psi03Z)\
        + (mp.conj(N24) * N14 * apsi04psi04Z)
    bchi02chi01Z = ((-1) * N23 * mp.conj(N13) * apsi03psi03Z)\
        - (N24 * mp.conj(N14) * apsi04psi04Z)
    achi02chi02Z = (mp.conj(N23) * N23 * apsi03psi03Z)\
        + (mp.conj(N24) * N24 * apsi04psi04Z)
    bchi02chi02Z = ((-1) * N23 * mp.conj(N23) * apsi03psi03Z)\
        - (N24 * mp.conj(N24) * apsi04psi04Z)
    achi02chi03Z = (mp.conj(N23) * N33 * apsi03psi03Z)\
        + (mp.conj(N24) * N34 * apsi04psi04Z)
    bchi02chi03Z = ((-1) * N23 * mp.conj(N33) * apsi03psi03Z)\
        - (N24 * mp.conj(N34) * apsi04psi04Z)
    achi02chi04Z = (mp.conj(N23) * N43 * apsi03psi03Z)\
        + (mp.conj(N24) * N44 * apsi04psi04Z)
    bchi02chi04Z = ((-1) * N23 * mp.conj(N43) * apsi03psi03Z)\
        - (N24 * mp.conj(N44) * apsi04psi04Z)
    achi03chi01Z = (mp.conj(N33) * N13 * apsi03psi03Z)\
        + (mp.conj(N34) * N14 * apsi04psi04Z)
    bchi03chi01Z = ((-1) * N33 * mp.conj(N13) * apsi03psi03Z)\
        - (N34 * mp.conj(N14) * apsi04psi04Z)
    achi03chi02Z = (mp.conj(N33) * N23 * apsi03psi03Z)\
        + (mp.conj(N34) * N24 * apsi04psi04Z)
    bchi03chi02Z = ((-1) * N33 * mp.conj(N23) * apsi03psi03Z)\
        - (N34 * mp.conj(N24) * apsi04psi04Z)
    achi03chi03Z = (mp.conj(N33) * N33 * apsi03psi03Z)\
        + (mp.conj(N34) * N34 * apsi04psi04Z)
    bchi03chi03Z = ((-1) * N33 * mp.conj(N33) * apsi03psi03Z)\
        - (N34 * mp.conj(N34) * apsi04psi04Z)
    achi03chi04Z = (mp.conj(N33) * N43 * apsi03psi03Z)\
        + (mp.conj(N34) * N44 * apsi04psi04Z)
    bchi03chi04Z = ((-1) * N33 * mp.conj(N43) * apsi03psi03Z)\
        - (N34 * mp.conj(N44) * apsi04psi04Z)
    achi04chi01Z = (mp.conj(N43) * N13 * apsi03psi03Z)\
        + (mp.conj(N44) * N14 * apsi04psi04Z)
    bchi04chi01Z = ((-1) * N43 * mp.conj(N13) * apsi03psi03Z)\
        - (N44 * mp.conj(N14) * apsi04psi04Z)
    achi04chi02Z = (mp.conj(N43) * N23 * apsi03psi03Z)\
        + (mp.conj(N44) * N24 * apsi04psi04Z)
    bchi04chi02Z = ((-1) * N43 * mp.conj(N23) * apsi03psi03Z)\
        - (N44 * mp.conj(N24) * apsi04psi04Z)
    achi04chi03Z = (mp.conj(N43) * N33 * apsi03psi03Z)\
        + (mp.conj(N44) * N34 * apsi04psi04Z)
    bchi04chi03Z = ((-1) * N43 * mp.conj(N33) * apsi03psi03Z)\
        - (N44 * mp.conj(N34) * apsi04psi04Z)
    achi04chi04Z = (mp.conj(N43) * N43 * apsi03psi03Z)\
        + (mp.conj(N44) * N44 * apsi04psi04Z)
    bchi04chi04Z = ((-1) * N43 * mp.conj(N43) * apsi03psi03Z)\
        - (N44 * mp.conj(N44) * apsi04psi04Z)

    f011Z = mp.fsum([achi01chi01Z, bchi01chi01Z],
                    absolute=True, squared=True)
    f012Z = mp.fsum([achi01chi02Z, bchi01chi02Z],
                    absolute=True, squared=True)
    f013Z = mp.fsum([achi01chi03Z, bchi01chi03Z],
                    absolute=True, squared=True)
    f014Z = mp.fsum([achi01chi04Z, bchi01chi04Z],
                    absolute=True, squared=True)
    f021Z = mp.fsum([achi02chi01Z, bchi02chi01Z],
                    absolute=True, squared=True)
    f022Z = mp.fsum([achi02chi02Z, bchi02chi02Z],
                    absolute=True, squared=True)
    f023Z = mp.fsum([achi02chi03Z, bchi02chi03Z],
                    absolute=True, squared=True)
    f024Z = mp.fsum([achi02chi04Z, bchi02chi04Z],
                    absolute=True, squared=True)
    f031Z = mp.fsum([achi03chi01Z, bchi03chi01Z],
                    absolute=True, squared=True)
    f032Z = mp.fsum([achi03chi02Z, bchi03chi02Z],
                    absolute=True, squared=True)
    f033Z = mp.fsum([achi03chi03Z, bchi03chi03Z],
                    absolute=True, squared=True)
    f034Z = mp.fsum([achi03chi04Z, bchi03chi04Z],
                    absolute=True, squared=True)
    f041Z = mp.fsum([achi04chi01Z, bchi04chi01Z],
                    absolute=True, squared=True)
    f042Z = mp.fsum([achi04chi02Z, bchi04chi02Z],
                    absolute=True, squared=True)
    f043Z = mp.fsum([achi04chi03Z, bchi04chi03Z],
                    absolute=True, squared=True)
    f044Z = mp.fsum([achi04chi04Z, bchi04chi04Z],
                    absolute=True, squared=True)

    g011Z = 2 * mp.re(mp.conj(bchi01chi01Z) * achi01chi01Z)
    g012Z = 2 * mp.re(mp.conj(bchi01chi02Z) * achi01chi02Z)
    g013Z = 2 * mp.re(mp.conj(bchi01chi03Z) * achi01chi03Z)
    g014Z = 2 * mp.re(mp.conj(bchi01chi04Z) * achi01chi04Z)
    g021Z = 2 * mp.re(mp.conj(bchi02chi01Z) * achi02chi01Z)
    g022Z = 2 * mp.re(mp.conj(bchi02chi02Z) * achi02chi02Z)
    g023Z = 2 * mp.re(mp.conj(bchi02chi03Z) * achi02chi03Z)
    g024Z = 2 * mp.re(mp.conj(bchi02chi04Z) * achi02chi04Z)
    g031Z = 2 * mp.re(mp.conj(bchi03chi01Z) * achi03chi01Z)
    g032Z = 2 * mp.re(mp.conj(bchi03chi02Z) * achi03chi02Z)
    g033Z = 2 * mp.re(mp.conj(bchi03chi03Z) * achi03chi03Z)
    g034Z = 2 * mp.re(mp.conj(bchi03chi04Z) * achi03chi04Z)
    g041Z = 2 * mp.re(mp.conj(bchi04chi01Z) * achi04chi01Z)
    g042Z = 2 * mp.re(mp.conj(bchi04chi02Z) * achi04chi02Z)
    g043Z = 2 * mp.re(mp.conj(bchi04chi03Z) * achi04chi03Z)
    g044Z = 2 * mp.re(mp.conj(bchi04chi04Z) * achi04chi04Z)

    achip1chip1Z = (mp.conj(V11) * V11 * apsip1psip1Z)\
        + (mp.conj(V12) * V12 * apsip2psip2Z)
    bchip1chip1Z = (U11 * mp.conj(U11) * apsip1psip1Z)\
        + (U12 * mp.conj(U12) * apsip2psip2Z)
    achip1chip2Z = (mp.conj(V11) * V21 * apsip1psip1Z)\
        + (mp.conj(V12) * V22 * apsip2psip2Z)
    bchip1chip2Z = (U11 * mp.conj(U21) * apsip1psip1Z)\
        + (U12 * mp.conj(U22) * apsip2psip2Z)
    achip2chip1Z = (mp.conj(V21) * V11 * apsip1psip1Z)\
        + (mp.conj(V22) * V12 * apsip2psip2Z)
    bchip2chip1Z = (U21 * mp.conj(U11) * apsip1psip1Z)\
        + (U22 * mp.conj(U12) * apsip2psip2Z)
    achip2chip2Z = (mp.conj(V21) * V21 * apsip1psip1Z)\
        + (mp.conj(V22) * V22 * apsip2psip2Z)
    bchip2chip2Z = (U21 * mp.conj(U21) * apsip1psip1Z)\
        + (U22 * mp.conj(U22) * apsip2psip2Z)

    fp11Z = mp.fsum([achip1chip1Z, bchip1chip1Z],
                    absolute=True, squared=True)
    fp12Z = mp.fsum([achip1chip2Z, bchip1chip2Z],
                    absolute=True, squared=True)
    fp21Z = mp.fsum([achip2chip1Z, bchip2chip1Z],
                    absolute=True, squared=True)
    fp22Z = mp.fsum([achip2chip2Z, bchip2chip2Z],
                    absolute=True, squared=True)

    gp11Z = 2 * mp.re(mp.conj(bchip1chip1Z) * achip1chip1Z)
    gp12Z = 2 * mp.re(mp.conj(bchip1chip2Z) * achip1chip2Z)
    gp21Z = 2 * mp.re(mp.conj(bchip2chip1Z) * achip2chip1Z)
    gp22Z = 2 * mp.re(mp.conj(bchip2chip2Z) * achip2chip2Z)

    # Set up mixing angles
    theta_nu = mpf('0')
    theta_stop = 0.5 * mp.atan(2 * mym_t * (my_At + (my_mu / my_tan_beta))
                               / (my_mQ3sq - my_mU3sq
                                  + (varmZsq * myc2b
                                     * (0.5 - (2 * e_uL * sinthWsq)))))
    theta_sbot = 0.5 * mp.atan(2 * mym_b * (my_Ab + (my_mu * my_tan_beta))
                               / (my_mQ3sq - my_mD3sq
                                  - (varmZsq * myc2b
                                     * (0.5 + (2 * e_dL * sinthWsq)))))
    theta_stau = 0.5 * mp.atan(2 * mym_tau * (my_Atau + (my_mu * my_tan_beta))
                               / (my_mL3sq - my_mE3sq
                                  - (varmZsq * myc2b
                                     * (0.5 + (2 * e_eL * sinthWsq)))))
    theta_schm = 0.5 * mp.atan(2 * mym_c * (my_Ac + (my_mu / my_tan_beta))
                               / (my_mQ2sq - my_mU2sq
                                  + (varmZsq * myc2b
                                     * (0.5 - (2 * e_uL * sinthWsq)))))
    theta_sstr = 0.5 * mp.atan(2 * mym_s * (my_As + (my_mu * my_tan_beta))
                               / (my_mQ2sq - my_mD2sq
                                  - (varmZsq * myc2b
                                     * (0.5 + (2 * e_dL * sinthWsq)))))
    theta_smu = 0.5 * mp.atan(2 * mym_mu * (my_Amu + (my_mu * my_tan_beta))
                              / (my_mL2sq - my_mE2sq
                                 - (varmZsq * myc2b
                                    * (0.5 + (2 * e_eL * sinthWsq)))))
    theta_sup = 0.5 * mp.atan(2 * mym_u * (my_Au + (my_mu / my_tan_beta))
                              / (my_mQ1sq - my_mU1sq
                                 + (varmZsq * myc2b
                                    * (0.5 - (2 * e_uL * sinthWsq)))))
    theta_sdwn = 0.5 * mp.atan(2 * mym_d * (my_Ad + (my_mu * my_tan_beta))
                               / (my_mQ1sq - my_mD1sq
                                  - (varmZsq * myc2b
                                     * (0.5 + (2 * e_dL * sinthWsq)))))
    theta_se = 0.5 * mp.atan(2 * mym_e * (my_Ae + (my_mu * my_tan_beta))
                             / (my_mL1sq - my_mE1sq
                                - (varmZsq * myc2b
                                   * (0.5 + (2 * e_eL * sinthWsq)))))

    # Sfermion-sfermion-Z couplings
    vt11 = (g_uL * (mp.cos(theta_stop) ** 2))\
        - (g_uR * (mp.sin(theta_stop) ** 2))
    vt11sq = vt11 ** 2
    vc11 = (g_uL * (mp.cos(theta_schm) ** 2))\
        - (g_uR * (mp.sin(theta_schm) ** 2))
    vc11sq = vc11 ** 2
    vu11 = (g_uL * (mp.cos(theta_sup) ** 2))\
        - (g_uR * (mp.sin(theta_sup) ** 2))
    vu11sq = vu11 ** 2
    vb11 = (g_dL * (mp.cos(theta_sbot) ** 2))\
        - (g_dR * (mp.sin(theta_sbot) ** 2))
    vb11sq = vb11 ** 2
    vs11 = (g_dL * (mp.cos(theta_sstr) ** 2))\
        - (g_dR * (mp.sin(theta_sstr) ** 2))
    vs11sq = vs11 ** 2
    vd11 = (g_dL * (mp.cos(theta_sdwn) ** 2))\
        - (g_dR * (mp.sin(theta_sdwn) ** 2))
    vd11sq = vd11 ** 2
    vtau11 = (g_eL * (mp.cos(theta_stau) ** 2))\
        - (g_eR * (mp.sin(theta_stau) ** 2))
    vtau11sq = vtau11 ** 2
    vmu11 = (g_eL * (mp.cos(theta_smu) ** 2))\
        - (g_eR * (mp.sin(theta_smu) ** 2))
    vmu11sq = vmu11 ** 2
    ve11 = (g_eL * (mp.cos(theta_se) ** 2))\
        - (g_eR * (mp.sin(theta_se) ** 2))
    ve11sq = ve11 ** 2
    vsneut11 = g_sneut
    vsneut11sq = vsneut11 ** 2

    vt22 = (g_uR * (mp.cos(theta_stop) ** 2))\
        - (g_uL * (mp.sin(theta_stop) ** 2))
    vt22sq = vt22 ** 2
    vc22 = (g_uR * (mp.cos(theta_schm) ** 2))\
        - (g_uL * (mp.sin(theta_schm) ** 2))
    vc22sq = vc22 ** 2
    vu22 = (g_uR * (mp.cos(theta_sup) ** 2))\
        - (g_uL * (mp.sin(theta_sup) ** 2))
    vu22sq = vu22 ** 2
    vb22 = (g_dR * (mp.cos(theta_sbot) ** 2))\
        - (g_dL * (mp.sin(theta_sbot) ** 2))
    vb22sq = vb22 ** 2
    vs22 = (g_dR * (mp.cos(theta_sstr) ** 2))\
        - (g_dL * (mp.sin(theta_sstr) ** 2))
    vs22sq = vs22 ** 2
    vd22 = (g_dR * (mp.cos(theta_sdwn) ** 2))\
        - (g_dL * (mp.sin(theta_sdwn) ** 2))
    vd22sq = vd22 ** 2
    vtau22 = (g_eR * (mp.cos(theta_stau) ** 2))\
        - (g_eL * (mp.sin(theta_stau) ** 2))
    vtau22sq = vtau22 ** 2
    vmu22 = (g_eR * (mp.cos(theta_smu) ** 2))\
        - (g_eL * (mp.sin(theta_smu) ** 2))
    vmu22sq = vmu22 ** 2
    ve22 = (g_eR * (mp.cos(theta_se) ** 2))\
        - (g_eL * (mp.sin(theta_se) ** 2))
    ve22sq = ve22 ** 2
    vsneut22 = 0
    vsneut22sq = 0

    vt12 = (g_uL + g_uR) * mp.cos(theta_stop) * mp.sin(theta_stop)
    vt12sq = vt12 ** 2
    vc12 = (g_uL + g_uR) * mp.cos(theta_schm) * mp.sin(theta_schm)
    vc12sq = vc12 ** 2
    vu12 = (g_uL + g_uR) * mp.cos(theta_sup) * mp.sin(theta_sup)
    vu12sq = vu12 ** 2
    vb12 = (g_dL + g_dR) * mp.cos(theta_sbot) * mp.sin(theta_sbot)
    vb12sq = vb12 ** 2
    vs12 = (g_dL + g_dR) * mp.cos(theta_sstr) * mp.sin(theta_sstr)
    vs12sq = vs12 ** 2
    vd12 = (g_dL + g_dR) * mp.cos(theta_sdwn) * mp.sin(theta_sdwn)
    vd12sq = vd12 ** 2
    vtau12 = (g_eL + g_eR) * mp.cos(theta_stau) * mp.sin(theta_stau)
    vtau12sq = vtau12 ** 2
    vmu12 = (g_eL + g_eR) * mp.cos(theta_smu) * mp.sin(theta_smu)
    vmu12sq = vmu12 ** 2
    ve12 = (g_eL + g_eR) * mp.cos(theta_se) * mp.sin(theta_se)
    ve12sq = ve12 ** 2
    vsneut12 = 0
    vsneut12sq = 0

    def mZself_energy(varmZsq):
        leadingfac = (rung2 ** 2) / (16 * (mp.pi ** 2) * costhWsq)
        line1 = (-1) * (salphbet ** 2)\
            * (sB22func(MA0pole ** 2, MHpole ** 2, SLHAscale, varmZsq)
               + sB22func(varmZsq, Mhpole ** 2, SLHAscale, varmZsq)
               - (varmZsq * B0func(varmZsq, Mhpole ** 2, SLHAscale, varmZsq)))
        line2 = (-1) * (calphbet ** 2)\
            * (sB22func(varmZsq, MHpole ** 2, SLHAscale, varmZsq)
               + sB22func(MA0pole ** 2, Mhpole ** 2, SLHAscale, varmZsq)
               - (varmZsq * B0func(varmZsq, MHpole ** 2, SLHAscale, varmZsq)))
        line3 = (-2) * (costhWsq ** 2)\
            * (((2 * varmZsq) + (MWpole ** 2) - (varmZsq * (sinthWsq ** 2)
                                                 / costhWsq))
               * B0func(MWpole ** 2, MWpole ** 2, SLHAscale, varmZsq))
        line4 = ((-1) * ((8 * (costhWsq ** 2)) + (cos2thW ** 2))
                 * (sB22func(MWpole ** 2, MWpole ** 2, SLHAscale, varmZsq)))\
            - ((cos2thW ** 2) * sB22func(MHpmpole ** 2, MHpmpole ** 2,
                                         SLHAscale, varmZsq))
        line5 = ((-12) * ((vt11sq * sB22func(Mt1pole ** 2, Mt1pole ** 2,
                                             SLHAscale, varmZsq))
                          + (vt12sq * (sB22func(Mt1pole ** 2,
                                                Mt2pole ** 2,
                                                SLHAscale, varmZsq)
                                       + sB22func(Mt2pole **  2,
                                                  Mt1pole ** 2,
                                                  SLHAscale, varmZsq)))
                          + (vt22sq * sB22func(Mt2pole ** 2, Mt2pole ** 2,
                                               SLHAscale, varmZsq))
                          + (vc11sq * sB22func(Mc1pole ** 2, Mc1pole ** 2,
                                               SLHAscale, varmZsq))
                          + (vc12sq * (sB22func(Mc1pole ** 2,
                                                Mc2pole ** 2,
                                                SLHAscale, varmZsq)
                                       + sB22func(Mc2pole **  2,
                                                  Mc1pole ** 2,
                                                  SLHAscale, varmZsq)))
                          + (vc22sq * sB22func(Mc2pole ** 2, Mc2pole ** 2,
                                               SLHAscale, varmZsq))
                          + (vu11sq * sB22func(Mu1pole ** 2, Mu1pole ** 2,
                                               SLHAscale, varmZsq))
                          + (vu12sq * (sB22func(Mu1pole ** 2,
                                                Mu2pole ** 2,
                                                SLHAscale, varmZsq)
                                       + sB22func(Mu2pole **  2,
                                                  Mu1pole ** 2,
                                                  SLHAscale, varmZsq)))
                          + (vu22sq * sB22func(Mu2pole ** 2, Mu2pole ** 2,
                                               SLHAscale, varmZsq))
                          + (vb11sq * sB22func(Mb1pole ** 2, Mb1pole ** 2,
                                               SLHAscale, varmZsq))
                          + (vb12sq * (sB22func(Mb1pole ** 2,
                                                Mb2pole ** 2,
                                                SLHAscale, varmZsq)
                                       + sB22func(Mb2pole **  2,
                                                  Mb1pole ** 2,
                                                  SLHAscale, varmZsq)))
                          + (vb22sq * sB22func(Mb2pole ** 2, Mb2pole ** 2,
                                               SLHAscale, varmZsq))
                          + (vs11sq * sB22func(Ms1pole ** 2, Ms1pole ** 2,
                                               SLHAscale, varmZsq))
                          + (vs12sq * (sB22func(Ms1pole ** 2,
                                                Ms2pole ** 2,
                                                SLHAscale, varmZsq)
                                       + sB22func(Ms2pole **  2,
                                                  Ms1pole ** 2,
                                                  SLHAscale, varmZsq)))
                          + (vs22sq * sB22func(Ms2pole ** 2, Ms2pole ** 2,
                                               SLHAscale, varmZsq))
                          + (vd11sq * sB22func(Md1pole ** 2, Md1pole ** 2,
                                               SLHAscale, varmZsq))
                          + (vd12sq * (sB22func(Md1pole ** 2,
                                                Md2pole ** 2,
                                                SLHAscale, varmZsq)
                                       + sB22func(Md2pole **  2,
                                                  Md1pole ** 2,
                                                  SLHAscale, varmZsq)))
                          + (vd22sq * sB22func(Md2pole ** 2, Md2pole ** 2,
                                               SLHAscale, varmZsq)))
                 - (4 * ((vtau11sq * sB22func(Mtau1pole ** 2, Mtau1pole ** 2,
                                              SLHAscale, varmZsq))
                         + (vtau12sq * (sB22func(Mtau1pole ** 2,
                                                 Mtau2pole ** 2,
                                                 SLHAscale, varmZsq)
                                        + sB22func(Mtau2pole **  2,
                                                   Mtau1pole ** 2,
                                                   SLHAscale, varmZsq)))
                         + (vtau22sq * sB22func(Mtau2pole ** 2, Mtau2pole ** 2,
                                                SLHAscale, varmZsq))
                         + (vmu11sq * sB22func(Mmu1pole ** 2, Mmu1pole ** 2,
                                               SLHAscale, varmZsq))
                         + (vmu12sq * (sB22func(Mmu1pole ** 2,
                                                Mmu2pole ** 2,
                                                SLHAscale, varmZsq)
                                       + sB22func(Mmu2pole **  2,
                                                  Mmu1pole ** 2,
                                                  SLHAscale, varmZsq)))
                         + (vmu22sq * sB22func(Mmu2pole ** 2, Mmu2pole ** 2,
                                               SLHAscale, varmZsq))
                         + (ve11sq * sB22func(Me1pole ** 2, Me1pole ** 2,
                                              SLHAscale, varmZsq))
                         + (ve12sq * (sB22func(Me1pole ** 2,
                                               Me2pole ** 2,
                                               SLHAscale, varmZsq)
                                       + sB22func(Me2pole **  2,
                                                  Me1pole ** 2,
                                                  SLHAscale, varmZsq)))
                         + (ve22sq * sB22func(Me2pole ** 2, Me2pole ** 2,
                                              SLHAscale, varmZsq))
                         + (vsneut11sq * sB22func(MnutauLpole ** 2,
                                                  MnutauLpole ** 2,
                                                  SLHAscale, varmZsq))
                         + (vsneut11sq * sB22func(MnumuLpole ** 2,
                                                  MnumuLpole ** 2,
                                                  SLHAscale, varmZsq))
                         + (vsneut11sq * sB22func(MnueLpole ** 2,
                                                  MnueLpole ** 2,
                                                  SLHAscale, varmZsq)))))
        line6 = (3 * (((g_uLsq + g_uRsq) * (Hfunc(mym_t ** 2, mym_t ** 2,
                                                  SLHAscale, varmZsq)
                                            + Hfunc(mym_c ** 2, mym_c ** 2,
                                                    SLHAscale, varmZsq)
                                            + Hfunc(mym_u ** 2, mym_u ** 2,
                                                    SLHAscale, varmZsq)))
                      + ((g_dLsq + g_dRsq) * (Hfunc(mym_b ** 2, mym_b ** 2,
                                                    SLHAscale, varmZsq)
                                              + Hfunc(mym_s ** 2, mym_s ** 2,
                                                      SLHAscale, varmZsq)
                                              + Hfunc(mym_d ** 2, mym_d ** 2,
                                                      SLHAscale, varmZsq)))
                      - ((4 * g_uL * g_uR * (((mym_t ** 2)
                                              * B0func(mym_t ** 2, mym_t ** 2,
                                                       SLHAscale, varmZsq))
                                             + ((mym_c ** 2)
                                                * B0func(mym_c ** 2,
                                                         mym_c ** 2,
                                                         SLHAscale, varmZsq))
                                             + ((mym_u ** 2)
                                                * B0func(mym_u ** 2,
                                                         mym_u ** 2,
                                                         SLHAscale,
                                                         varmZsq))))
                         + (4 * g_dL * g_dR * (((mym_b ** 2)
                                                * B0func(mym_b ** 2,
                                                         mym_b ** 2,
                                                         SLHAscale, varmZsq))
                                               + ((mym_s ** 2)
                                                  * B0func(mym_s ** 2,
                                                           mym_s ** 2,
                                                           SLHAscale,
                                                           varmZsq))
                                               + ((mym_d ** 2)
                                                  * B0func(mym_d ** 2,
                                                           mym_d ** 2,
                                                           SLHAscale,
                                                           varmZsq)))))))\
            + (((g_eLsq + g_eRsq) * (Hfunc(mym_tau ** 2, mym_tau ** 2,
                                           SLHAscale, varmZsq)
                                     + Hfunc(mym_mu ** 2, mym_mu ** 2,
                                             SLHAscale, varmZsq)
                                     + Hfunc(mym_e ** 2, mym_e ** 2,
                                             SLHAscale, varmZsq)))
               - ((4 * g_eL * g_eR * (((mym_tau ** 2)
                                       * B0func(mym_tau ** 2, mym_tau ** 2,
                                                SLHAscale, varmZsq))
                                      + ((mym_mu ** 2)
                                         * B0func(mym_mu ** 2, mym_mu ** 2,
                                                  SLHAscale, varmZsq))
                                      + ((mym_e ** 2)
                                         * B0func(mym_e ** 2, mym_e ** 2,
                                                  SLHAscale, varmZsq))))))
        line7 = (costhWsq / (2 * (rung2 ** 2)))\
            * ((f011Z * Hfunc(Mneut1pole ** 2, Mneut1pole ** 2,
                              SLHAscale, varmZsq))
               + (f012Z * Hfunc(Mneut1pole ** 2, Mneut2pole ** 2,
                                SLHAscale, varmZsq))
               + (f013Z * Hfunc(Mneut1pole ** 2, Mneut3pole ** 2,
                                SLHAscale, varmZsq))
               + (f014Z * Hfunc(Mneut1pole ** 2, Mneut4pole ** 2,
                                SLHAscale, varmZsq))
               + (f021Z * Hfunc(Mneut2pole ** 2, Mneut1pole ** 2,
                                SLHAscale, varmZsq))
               + (f022Z * Hfunc(Mneut2pole ** 2, Mneut2pole ** 2,
                                SLHAscale, varmZsq))
               + (f023Z * Hfunc(Mneut2pole ** 2, Mneut3pole ** 2,
                                SLHAscale, varmZsq))
               + (f024Z * Hfunc(Mneut2pole ** 2, Mneut4pole ** 2,
                                SLHAscale, varmZsq))
               + (f031Z * Hfunc(Mneut3pole ** 2, Mneut1pole ** 2,
                                SLHAscale, varmZsq))
               + (f032Z * Hfunc(Mneut3pole ** 2, Mneut2pole ** 2,
                                SLHAscale, varmZsq))
               + (f033Z * Hfunc(Mneut3pole ** 2, Mneut3pole ** 2,
                                SLHAscale, varmZsq))
               + (f034Z * Hfunc(Mneut3pole ** 2, Mneut4pole ** 2,
                                SLHAscale, varmZsq))
               + (f041Z * Hfunc(Mneut4pole ** 2, Mneut1pole ** 2,
                                SLHAscale, varmZsq))
               + (f042Z * Hfunc(Mneut4pole ** 2, Mneut2pole ** 2,
                                SLHAscale, varmZsq))
               + (f043Z * Hfunc(Mneut4pole ** 2, Mneut3pole ** 2,
                                SLHAscale, varmZsq))
               + (f044Z * Hfunc(Mneut4pole ** 2, Mneut4pole ** 2,
                                SLHAscale, varmZsq))
               + (2 * g011Z * Mneut1pole * Mneut1pole
                  * B0func(Mneut1pole ** 2, Mneut1pole ** 2,
                           SLHAscale, varmZsq))
               + (2 * g012Z * Mneut1pole * Mneut2pole
                  * B0func(Mneut1pole ** 2, Mneut2pole ** 2,
                           SLHAscale, varmZsq))
               + (2 * g013Z * Mneut1pole * Mneut3pole
                  * B0func(Mneut1pole ** 2, Mneut3pole ** 2,
                           SLHAscale, varmZsq))
               + (2 * g014Z * Mneut1pole * Mneut4pole
                  * B0func(Mneut1pole ** 2, Mneut4pole ** 2,
                           SLHAscale, varmZsq))
               + (2 * g021Z * Mneut2pole * Mneut1pole
                  * B0func(Mneut2pole ** 2, Mneut1pole ** 2,
                           SLHAscale, varmZsq))
               + (2 * g022Z * Mneut2pole * Mneut2pole
                  * B0func(Mneut2pole ** 2, Mneut2pole ** 2,
                           SLHAscale, varmZsq))
               + (2 * g023Z * Mneut2pole * Mneut3pole
                  * B0func(Mneut2pole ** 2, Mneut3pole ** 2,
                           SLHAscale, varmZsq))
               + (2 * g024Z * Mneut2pole * Mneut4pole
                  * B0func(Mneut2pole ** 2, Mneut4pole ** 2,
                           SLHAscale, varmZsq))
               + (2 * g031Z * Mneut3pole * Mneut1pole
                  * B0func(Mneut3pole ** 2, Mneut1pole ** 2,
                           SLHAscale, varmZsq))
               + (2 * g032Z * Mneut3pole * Mneut2pole
                  * B0func(Mneut3pole ** 2, Mneut2pole ** 2,
                           SLHAscale, varmZsq))
               + (2 * g033Z * Mneut3pole * Mneut3pole
                  * B0func(Mneut3pole ** 2, Mneut3pole ** 2,
                           SLHAscale, varmZsq))
               + (2 * g034Z * Mneut3pole * Mneut4pole
                  * B0func(Mneut3pole ** 2, Mneut4pole ** 2,
                           SLHAscale, varmZsq))
               + (2 * g041Z * Mneut4pole * Mneut1pole
                  * B0func(Mneut4pole ** 2, Mneut1pole ** 2,
                           SLHAscale, varmZsq))
               + (2 * g042Z * Mneut4pole * Mneut2pole
                  * B0func(Mneut4pole ** 2, Mneut2pole ** 2,
                           SLHAscale, varmZsq))
               + (2 * g043Z * Mneut4pole * Mneut3pole
                  * B0func(Mneut4pole ** 2, Mneut3pole ** 2,
                           SLHAscale, varmZsq))
               + (2 * g044Z * Mneut4pole * Mneut4pole
                  * B0func(Mneut4pole ** 2, Mneut4pole ** 2,
                           SLHAscale, varmZsq)))
        line8 = (costhWsq / (rung2 ** 2))\
            * ((fp11Z * Hfunc(Mchar1pole ** 2, Mchar1pole ** 2,
                              SLHAscale, varmZsq))
               + (fp12Z * Hfunc(Mchar1pole ** 2, Mchar2pole ** 2,
                                SLHAscale, varmZsq))
               + (fp21Z * Hfunc(Mchar2pole ** 2, Mchar1pole ** 2,
                                SLHAscale, varmZsq))
               + (fp22Z * Hfunc(Mchar2pole ** 2, Mchar2pole ** 2,
                                SLHAscale, varmZsq))
               + ((gp11Z * B0func(Mchar1pole ** 2, Mchar1pole ** 2,
                                  SLHAscale, varmZsq)
                   * Mchar1pole * Mchar1pole)
                  + (gp12Z * B0func(Mchar1pole ** 2, Mchar2pole ** 2,
                                    SLHAscale, varmZsq)
                     * Mchar1pole * Mchar2pole)
                  + (gp21Z * B0func(Mchar2pole ** 2, Mchar1pole ** 2,
                                    SLHAscale, varmZsq)
                     * Mchar1pole * Mchar2pole)
                  + (gp22Z * B0func(Mchar2pole ** 2, Mchar2pole ** 2,
                                    SLHAscale, varmZsq))
                     * Mchar2pole * Mchar2pole) * 2)
        myPiTZZ = leadingfac * (line1 + line2 + line3 + line4 + line5 + line6
                                + line7 + line8)
        return myPiTZZ
    myroot = float(runmZsq - varmZsq - mp.re(mZself_energy(varmZsq)))
    return myroot
