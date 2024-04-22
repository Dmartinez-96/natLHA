#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 14:13:41 2023

Delta_BG calculator.

@author: Dakotah Martinez
"""

#import Zpole_calc
import GUT_to_weak_runner
from GUT_to_weak_runner import GUT_to_weak_runner as G2WR
from copy import deepcopy
from mpmath import mp, mpf
import math
from alive_progress import alive_bar
import numpy as np

def Delta_BG_calc(modselno, precselno, polecalccheck, mymzsq, GUT_SCALE, myweakscale, inptanbval,
                  dpinputGUT_BCs):
    """
    Compute the fine-tuning measure Delta_BG for the selected model.

    Parameters
    ----------
    modselno : Int.
        Selected model number from model list.
    precselno : Int.
        Selected precision setting from list.
    polecalccheck: Int.
        Determine whether or not to use mZ^2 pole calculation, according to
            user's choice.
    mymzsq : Float.
        Running mZ^2, evaluated at Q=2 TeV from original SLHA point.
    inputGUT_BCs : Array of floats.
        Original GUT-scale BCs from SLHA file.

    Returns
    -------
    Delta_BG : Float.
        Naturalness measure Delta_BG.

    """
    inputGUT_BCs = deepcopy(dpinputGUT_BCs)
    def mz_tree_calc(inpmHdsqweak, inpmHusqweak, inpmusqweak, inptanbsqweak):
        """
        Calculate tree-level mZ^2(weak).

        Parameters
        ----------
        mHdsqweak : Float.
            Weak-scale mHd^2(weak) for computing tree-level mZ^2.
        mHusqweak : Float.
            Weak-scale mHu^2(weak) for computing tree-level mZ^2.
        musqweak : Float.
            Weak-scale mu^2(weak) for computing tree-level mZ^2.
        tanbsqweak : Float.
            Weak-scale tan(beta)^2(weak) for computing tree-level mZ^2.

        Returns
        -------
        mzsq_tree_wk : Float.
            Value of m_{Z}^{2}(weak) computed from tree-level Higgs min. cond.

        """
        mHdsqweak = mpf(str(inpmHdsqweak))
        mHusqweak = mpf(str(inpmHusqweak))
        musqweak = mpf(str(inpmusqweak))
        tanbsqweak = mpf(str(inptanbsqweak))
        mzsq_tree_wk = 2 * ((mHdsqweak - (mHusqweak * tanbsqweak))
                            / (tanbsqweak - 1))\
            - (2 * musqweak)
        return mzsq_tree_wk

    def deriv_step_calc(RGE_scale_init_val, RGE_scale_final_val, tanb_to_use, polecalccheck, BCs_to_run):
        """
        Do one derivative step evaluation in difference quotient.

        Parameters
        ----------
        RGE_scale_init_val : Float.
            Value from which to evolve RGEs, usually GUT-scale.
        RGE_scale_final_val : Float.
            Value to which to evolve RGEs, usually weak scale.
        tanb_to_use : Float.
            Weak-scale value of tan(beta) to use.
        polecalccheck : Int.
            Determine whether to include pole mass corrections or not.
        BCs_to_run : Array of floats.
            RGE boundary conditions at GUT scale for current step.

        Returns
        -------
        Float.
            Return mZ^2 evaluated at current position in parameter space.

        """
        curr_weaksol = G2WR(deepcopy(BCs_to_run), RGE_scale_init_val, myweakscale)
        if polecalccheck==1:
            # Still need to add in pole mass evaluations for spectrum of MSSM particles,
            # so this is disabled right now
            mycurrmzsq = mz_tree_calc(curr_weaksol[26],
                                      curr_weaksol[25],
                                      curr_weaksol[6],
                                      mp.power(tanb_to_use, 2))
            #print(fsolve(mZpolecalc, 91.1876 ** 2, args=(mycurrmzsq,myweakscale,deepcopy(curr_weaksol)),
            #             full_output=True))
            #return fsolve(mZpolecalc, 91.1876 ** 2,
            #              args=(mycurrmzsq,myweakscale,deepcopy(curr_weaksol)))[0]
            return mz_tree_calc(curr_weaksol[26],
                                curr_weaksol[25],
                                curr_weaksol[6],
                                mp.power(tanb_to_use, 2))
        else:
            return mz_tree_calc(curr_weaksol[26],
                                curr_weaksol[25],
                                curr_weaksol[6],
                                mp.power(tanb_to_use, 2))
    # Define how numerical derivatives are approximated based on central differences & precision
    if (precselno == 1):
        def deriv_num_calc(curr_hval, mzsq_mmmmh,mzsq_mmmh,mzsq_mmh,mzsq_mh,
                           mzsq_ph,mzsq_pph,mzsq_ppph,mzsq_pppph):
            """
            Evaluate approximate 8-point derivative at current p-space point.

            Parameters
            ----------
            curr_hval : Float.
                Step size to be used in difference quotient.
            mzsq_mmmmh : Float.
                mZ^2 value shifted left by 4h.
            mzsq_mmmh : Float.
                mZ^2 value shifted left by 3h.
            mzsq_mmh : Float.
                mZ^2 value shifted left by 2h.
            mzsq_mh : Float.
                mZ^2 value shifted left by h.
            mzsq_ph : Float.
                mZ^2 value shifted right by h.
            mzsq_pph : Float.
                mZ^2 value shifted right by 2h.
            mzsq_ppph : Float.
                mZ^2 value shifted right by 3h.
            mzsq_pppph : Float.
                mZ^2 value shifted right by 4h.

            Returns
            -------
            approxderivval : Float.
                Return difference quotient approximated derivative.

            """
            approxderivval = (1 / mpf(str(curr_hval)))\
                * ((mpf(str(mzsq_mmmmh)) / 280)
                   - ((4 / 105) * mpf(str(mzsq_mmmh)))
                   + (mpf(str(mzsq_mmh)) / 5)
                   - ((4 / 5) * mpf(str(mzsq_mh)))
                   + ((4 / 5) * mpf(str(mzsq_ph)))
                   - (mpf(str(mzsq_pph)) / 5)
                   + ((4 / 105) * mpf(str(mzsq_ppph)))
                   - (mpf(str(mzsq_pppph)) / 280))
            return approxderivval
    elif (precselno == 2):
        def deriv_num_calc(curr_hval, mzsq_mmh,mzsq_mh,
                           mzsq_ph,mzsq_pph):
            """
            Evaluate approximate 4-point derivative at current p-space point.

            Parameters
            ----------
            curr_hval : Float.
                Step size to be used in difference quotient.
            mzsq_mmh : Float.
                mZ^2 value shifted left by 2h.
            mzsq_mh : Float.
                mZ^2 value shifted left by h.
            mzsq_ph : Float.
                mZ^2 value shifted right by h.
            mzsq_pph : Float.
                mZ^2 value shifted right by 2h.

            Returns
            -------
            approxderivval : Float.
                Return difference quotient approximated derivative.

            """
            approxderivval = (1 / mpf(str(curr_hval)))\
                * ((mpf(str(mzsq_mmh)) / 12)
                   - ((2 / 3) * mpf(str(mzsq_mh)))
                   + ((2 / 3) * mpf(str(mzsq_ph)))
                   - (mpf(str(mzsq_pph)) / 12))
            return approxderivval
    elif (precselno == 3):
        def deriv_num_calc(curr_hval, mzsq_mh,
                           mzsq_ph):
            """
            Evaluate approximate 2-point derivative at current p-space point.

            Parameters
            ----------
            curr_hval : Float.
                Step size to be used in difference quotient.
            mzsq_mh : Float.
                mZ^2 value shifted left by h.
            mzsq_ph : Float.
                mZ^2 value shifted right by h.

            Returns
            -------
            approxderivval : Float.
                Return difference quotient approximated derivative.

            """
            approxderivval = (1 / mpf(str(curr_hval)))\
                * ((((-1) / 2) * mpf(str(mzsq_mh)))
                   + ((1 / 2) * mpf(str(mzsq_ph))))
            return approxderivval

    if (modselno == 1):
        print("Computing derivatives...\n")
        print("NOTE: this computation can take a while\n")
        mym0 = mpf(str(inputGUT_BCs[27]))
        mymhf = mpf(str(inputGUT_BCs[3]))
        myA0 = mpf(str(inputGUT_BCs[16])) / mpf(str(inputGUT_BCs[7]))
        mymusq0 = mpf(str(inputGUT_BCs[6]))
        mymu0 = mp.sqrt(str(abs(inputGUT_BCs[6])))
        def deriv_step_calc_mu0(shift_amt):
            """
            Do the current mass derivative step for higgsino mass mu.

            Parameters
            ----------
            shift_amt : Float.
                How much to shift for current derivative step.

            Returns
            -------
            mytestmzsq : Float.
                Return current shifted value of mZ^2 with the
                    higgsino mass shifted by the specified amount.

            """
            testBCs = deepcopy(inputGUT_BCs[:])
            for i in np.arange(27,42):
                testBCs[i] = mp.power(testBCs[i], 2)
            testBCs[6] = np.sign(mymusq0) * mp.power(mymu0 + shift_amt, 2)
            mytestmzsq = deriv_step_calc(GUT_SCALE,myweakscale, inptanbval, polecalccheck, deepcopy(testBCs))
            del testBCs
            return mytestmzsq

        def deriv_step_calc_scalars(shift_amt):
            """
            Do the current mass derivative step for squarks or sleptons.

            Parameters
            ----------
            shift_amt : Float.
                How much to shift for current derivative step.

            Returns
            -------
            mytestmzsq : Float.
                Return current shifted value of mZ^2 with the corresponding
                    scalar masses shifted by the specified amount.

            """
            testBCs = deepcopy(inputGUT_BCs[:])
            for i in np.arange(25, 27):
                testBCs[i] = mp.power(mp.sqrt(str(abs(testBCs[i])))
                                      + shift_amt, 2)
            for i in np.arange(27,42):
                testBCs[i] = mp.power(testBCs[i] + shift_amt, 2)
            mytestmzsq = deriv_step_calc(GUT_SCALE,myweakscale, inptanbval, polecalccheck, deepcopy(testBCs))
            del testBCs
            return mytestmzsq

        def deriv_step_calc_trilin(shift_amt):
            """
            Do the current mass derivative step for soft trilinear couplings.

            Parameters
            ----------
            shift_amt : Float.
                How much to shift each soft trilinear for current derivative step.

            Returns
            -------
            mytestmzsq : Float.
                Return current shifted value of mZ^2 with the
                    soft trilinear couplings shifted by the specified amount.

            """
            testBCs = deepcopy(inputGUT_BCs[:])
            for i in np.arange(27,42):
                testBCs[i] = mp.power(testBCs[i], 2)
            for i in np.arange(16,25):
                testBCs[i] = (((testBCs[i] / testBCs[i-9]) + shift_amt)
                              * testBCs[i-9])
            mytestmzsq = deriv_step_calc(GUT_SCALE,myweakscale, inptanbval, polecalccheck, deepcopy(testBCs))
            del testBCs
            return mytestmzsq

        def deriv_step_calc_gaugino(shift_amt):
            """
            Do the current mass derivative step for soft gaugino masses.

            Parameters
            ----------
            shift_amt : Float.
                How much to shift for current derivative step.

            Returns
            -------
            mytestmzsq : Float.
                Return current shifted value of mZ^2 with the corresponding
                    soft gaugino mass shifted by the specified amount.

            """
            testBCs = deepcopy(inputGUT_BCs[:])
            for i in np.arange(27,42):
                testBCs[i] = mp.power(testBCs[i], 2)
            for i in np.arange(3,6):
                testBCs[i] = testBCs[i] + shift_amt
            mytestmzsq = deriv_step_calc(GUT_SCALE,myweakscale, inptanbval, polecalccheck, deepcopy(testBCs))
            del testBCs
            return mytestmzsq

        if (precselno == 1):
            derivcount = 0
            hm0 = mpf(str(abs(mp.power(math.ulp(mym0), 1/9))))
            hmhf = mpf(str(abs(mp.power(math.ulp(mymhf), 1/9))))
            hA0 = mpf(str(abs(mp.power(math.ulp(myA0), 1/9))))
            hmu0 = mpf(str(abs(mp.power(math.ulp(mymu0), 1/9))))

            with alive_bar(32, dual_line=True,
                           force_tty=True, title='Computations: ',
                           bar='brackets') as bar:
                bar.text = f'Derivatives: {derivcount}/4, please wait...'
                ##### Set up solutions for m_0 derivative #####
                print("Error estimate for m0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hm0, 8)))))
                mzsq_m0ph = deriv_step_calc_scalars(hm0)
                bar()
                # Two deviations to right
                mzsq_m0pph = deriv_step_calc_scalars(2 * hm0)
                bar()
                # Three deviations to right
                mzsq_m0ppph = deriv_step_calc_scalars(3 * hm0)
                bar()
                # Four deviations to right
                mzsq_m0pppph = deriv_step_calc_scalars(4 * hm0)
                bar()
                # Deviate m0 by small amount LEFT and square soft scalar masses for BCs
                mzsq_m0mh = deriv_step_calc_scalars((-1) * hm0)
                bar()
                # Two deviations to left
                mzsq_m0mmh = deriv_step_calc_scalars((-2) * hm0)
                bar()
                # Three deviations to left
                mzsq_m0mmmh = deriv_step_calc_scalars((-3) * hm0)
                bar()
                # Four deviations to left
                mzsq_m0mmmmh = deriv_step_calc_scalars((-4) * hm0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/4, please wait...'

                print("Error estimate for m_1/2 derivative: "
                      + str("{:.4e}".format(float(mp.power(hmhf, 8)))))
                ##### Set up solutions for m_1/2 derivative #####
                mzsq_mhfph = deriv_step_calc_gaugino(hmhf)
                bar()
                # Two deviations to right
                mzsq_mhfpph = deriv_step_calc_gaugino(2 * hmhf)
                bar()
                # Three deviations to right
                mzsq_mhfppph = deriv_step_calc_gaugino(3 * hmhf)
                bar()
                # Four deviations to right
                mzsq_mhfpppph = deriv_step_calc_gaugino(4 * hmhf)
                bar()
                # Boundary conditions first to left
                mzsq_mhfmh = deriv_step_calc_gaugino((-1) * hmhf)
                bar()
                # Two deviations to left
                mzsq_mhfmmh = deriv_step_calc_gaugino((-2) * hmhf)
                bar()
                # Three deviations to left
                mzsq_mhfmmmh = deriv_step_calc_gaugino((-3) * hmhf)
                bar()
                # Four deviations to left
                mzsq_mhfmmmmh = deriv_step_calc_gaugino((-4) * hmhf)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/4, please wait...'
                print("Error estimate for A0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hA0, 8)))))

                ##### Set up solutions for A_0 derivative #####
                mzsq_A0ph = deriv_step_calc_trilin(hA0)
                bar()
                # Two deviations to right
                mzsq_A0pph = deriv_step_calc_trilin(2 * hA0)
                bar()
                # Three deviations to right
                mzsq_A0ppph = deriv_step_calc_trilin(3 * hA0)
                bar()
                # Four deviations to right
                mzsq_A0pppph = deriv_step_calc_trilin(4 * hA0)
                bar()
                # Deviate A0 once to left
                mzsq_A0mh = deriv_step_calc_trilin((-1) * hA0)
                bar()
                # Two deviations to left
                mzsq_A0mmh = deriv_step_calc_trilin((-2) * hA0)
                bar()
                # Three deviations to left
                mzsq_A0mmmh = deriv_step_calc_trilin((-3) * hA0)
                bar()
                # Four deviations to left
                mzsq_A0mmmmh = deriv_step_calc_trilin((-4) * hA0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/4, please wait...'
                print("Error estimate for mu derivative: "
                      + str("{:.4e}".format(float(mp.power(hmu0, 8)))))

                ##### Set up solutions for mu derivative #####
                mzsq_mu0ph = deriv_step_calc_mu0(hmu0)
                bar()
                # Two deviations to right
                mzsq_mu0pph = deriv_step_calc_mu0(2 * hmu0)
                bar()
                # Three deviations to right
                mzsq_mu0ppph = deriv_step_calc_mu0(3 * hmu0)
                bar()
                # Four deviations to right
                mzsq_mu0pppph = deriv_step_calc_mu0(4 * hmu0)
                bar()
                # mu to left
                mzsq_mu0mh = deriv_step_calc_mu0((-1) * hmu0)
                bar()
                # Two deviations to left
                mzsq_mu0mmh = deriv_step_calc_mu0((-2) * hmu0)
                bar()
                # Three deviations to left
                mzsq_mu0mmmh = deriv_step_calc_mu0((-3) * hmu0)
                bar()
                # Four deviations to left
                mzsq_mu0mmmmh = deriv_step_calc_mu0((-4) * hmu0)
                bar()

                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/4, please wait...'
                print("Done!")

                # Construct derivative array
                deriv_array = np.array([deriv_num_calc(hm0,mzsq_m0mmmmh,mzsq_m0mmmh,
                                                       mzsq_m0mmh,mzsq_m0mh,mzsq_m0ph,
                                                       mzsq_m0pph,mzsq_m0ppph,mzsq_m0pppph),
                                        deriv_num_calc(hmhf,mzsq_mhfmmmmh,mzsq_mhfmmmh,
                                                       mzsq_mhfmmh,mzsq_mhfmh,mzsq_mhfph,
                                                       mzsq_mhfpph,mzsq_mhfppph,mzsq_mhfpppph),
                                        deriv_num_calc(hA0,mzsq_A0mmmmh,mzsq_A0mmmh,
                                                       mzsq_A0mmh,mzsq_A0mh,mzsq_A0ph,
                                                       mzsq_A0pph,mzsq_A0ppph,mzsq_A0pppph),
                                        deriv_num_calc(hmu0,mzsq_mu0mmmmh,mzsq_mu0mmmh,
                                                       mzsq_mu0mmh,mzsq_mu0mh,mzsq_mu0ph,
                                                       mzsq_mu0pph,mzsq_mu0ppph,mzsq_mu0pppph)])
                sens_params = np.sort(np.array([(mp.fabs((mym0
                                                         / mymzsq)#/ mymzsq)
                                                        * deriv_array[0]),
                                                 'Delta_BG(m_0)'),
                                                (mp.fabs((mymhf
                                                         / mymzsq)
                                                        * deriv_array[1]),
                                                 'Delta_BG(m_1/2)'),
                                                (mp.fabs((myA0
                                                         / mymzsq)
                                                        * deriv_array[2]),
                                                 'Delta_BG(A_0)'),
                                                (mp.fabs((mymu0
                                                         / mymzsq)
                                                        * deriv_array[3]),
                                                 'Delta_BG(mu)')],
                                               dtype=[('BGContrib', object),
                                                      ('BGlabel', 'U40')]),
                                      order='BGContrib')
        elif (precselno == 2):
            derivcount = 0
            hm0 = mpf(str(abs(mp.power(math.ulp(mym0), 1/5))))
            hmhf = mpf(str(abs(mp.power(math.ulp(mymhf), 1/5))))
            hA0 = mpf(str(abs(mp.power(math.ulp(myA0), 1/5))))
            hmu0 = mpf(str(abs(mp.power(math.ulp(mymu0), 1/5))))
            with alive_bar(16, dual_line=True, title='Computations: ') as bar:
                bar.text = f'Derivatives: {derivcount}/4, please wait...'
                ##### Set up solutions for m_0 derivative #####
                print("Error estimate for m0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hm0, 4)))))
                mzsq_m0ph = deriv_step_calc_scalars(hm0)
                bar()
                # Two deviations to right
                mzsq_m0pph = deriv_step_calc_scalars(2 * hm0)
                bar()
                # Deviate m0 by small amount LEFT and square soft scalar masses for BCs
                mzsq_m0mh = deriv_step_calc_scalars((-1) * hm0)
                bar()
                # Two deviations to left
                mzsq_m0mmh = deriv_step_calc_scalars((-2) * hm0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/4, please wait...'

                print("Error estimate for m_1/2 derivative: "
                      + str("{:.4e}".format(float(mp.power(hmhf, 4)))))
                ##### Set up solutions for m_1/2 derivative #####
                mzsq_mhfph = deriv_step_calc_gaugino(hmhf)
                bar()
                # Two deviations to right
                mzsq_mhfpph = deriv_step_calc_gaugino(2 * hmhf)
                bar()
                # Boundary conditions first to left
                mzsq_mhfmh = deriv_step_calc_gaugino((-1) * hmhf)
                bar()
                # Two deviations to left
                mzsq_mhfmmh = deriv_step_calc_gaugino((-2) * hmhf)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/4, please wait...'
                print("Error estimate for A0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hA0, 4)))))

                ##### Set up solutions for A_0 derivative #####
                mzsq_A0ph = deriv_step_calc_trilin(hA0)
                bar()
                # Two deviations to right
                mzsq_A0pph = deriv_step_calc_trilin(2 * hA0)
                bar()
                # Deviate A0 once to left
                mzsq_A0mh = deriv_step_calc_trilin((-1) * hA0)
                bar()
                # Two deviations to left
                mzsq_A0mmh = deriv_step_calc_trilin((-2) * hA0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/4, please wait...'
                print("Error estimate for mu derivative: "
                      + str("{:.4e}".format(float(mp.power(hmu0, 4)))))

                ##### Set up solutions for mu derivative #####
                mzsq_mu0ph = deriv_step_calc_mu0(hmu0)
                bar()
                # Two deviations to right
                mzsq_mu0pph = deriv_step_calc_mu0(2 * hmu0)
                bar()
                # mu to left
                mzsq_mu0mh = deriv_step_calc_mu0((-1) * hmu0)
                bar()
                # Two deviations to left
                mzsq_mu0mmh = deriv_step_calc_mu0((-2) * hmu0)
                bar()

                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/4, please wait...'
                print("Done!")

                # Construct derivative array
                deriv_array = np.array([deriv_num_calc(hm0,
                                                       mzsq_m0mmh,mzsq_m0mh,mzsq_m0ph,
                                                       mzsq_m0pph),
                                        deriv_num_calc(hmhf,
                                                       mzsq_mhfmmh,mzsq_mhfmh,mzsq_mhfph,
                                                       mzsq_mhfpph),
                                        deriv_num_calc(hA0,
                                                       mzsq_A0mmh,mzsq_A0mh,mzsq_A0ph,
                                                       mzsq_A0pph),
                                        deriv_num_calc(hmu0,
                                                       mzsq_mu0mmh,mzsq_mu0mh,mzsq_mu0ph,
                                                       mzsq_mu0pph)])
                sens_params = np.sort(np.array([(mp.fabs((mym0
                                                         / mymzsq)#/ mymzsq)
                                                        * deriv_array[0]),
                                                 'Delta_BG(m_0)'),
                                                (mp.fabs((mymhf
                                                         / mymzsq)
                                                        * deriv_array[1]),
                                                 'Delta_BG(m_1/2)'),
                                                (mp.fabs((myA0
                                                         / mymzsq)
                                                        * deriv_array[2]),
                                                 'Delta_BG(A_0)'),
                                                (mp.fabs((mymu0
                                                         / mymzsq)
                                                        * deriv_array[3]),
                                                 'Delta_BG(mu)')],
                                               dtype=[('BGContrib', object),
                                                      ('BGlabel', 'U40')]),
                                      order='BGContrib')
        elif (precselno == 3):
            derivcount = 0
            hm0 = mpf(str(abs(mp.power(math.ulp(mym0), 1/3))))
            hmhf = mpf(str(abs(mp.power(math.ulp(mymhf), 1/3))))
            hA0 = mpf(str(abs(mp.power(math.ulp(myA0), 1/3))))
            hmu0 = mpf(str(abs(mp.power(math.ulp(mymu0), 1/3))))
            with alive_bar(8, dual_line=True, title='Computations: ') as bar:
                bar.text = f'Derivatives: {derivcount}/4, please wait...'
                ##### Set up solutions for m_0 derivative #####
                print("Error estimate for m0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hm0, 2)))))
                mzsq_m0ph = deriv_step_calc_scalars(hm0)
                bar()
                # Deviate m0 by small amount LEFT and square soft scalar masses for BCs
                mzsq_m0mh = deriv_step_calc_scalars((-1) * hm0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/4, please wait...'

                print("Error estimate for m_1/2 derivative: "
                      + str("{:.4e}".format(float(mp.power(hmhf, 2)))))
                ##### Set up solutions for m_1/2 derivative #####
                mzsq_mhfph = deriv_step_calc_gaugino(hmhf)
                bar()
                # Boundary conditions first to left
                mzsq_mhfmh = deriv_step_calc_gaugino((-1) * hmhf)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/4, please wait...'
                print("Error estimate for A0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hA0, 2)))))

                ##### Set up solutions for A_0 derivative #####
                mzsq_A0ph = deriv_step_calc_trilin(hA0)
                bar()
                # Deviate A0 once to left
                mzsq_A0mh = deriv_step_calc_trilin((-1) * hA0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/4, please wait...'
                print("Error estimate for mu derivative: "
                      + str("{:.4e}".format(float(mp.power(hmu0, 2)))))

                ##### Set up solutions for mu derivative #####
                mzsq_mu0ph = deriv_step_calc_mu0(hmu0)
                bar()
                # mu to left
                mzsq_mu0mh = deriv_step_calc_mu0((-1) * hmu0)
                bar()

                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/4, please wait...'
                print("Done!")

                # Construct derivative array
                deriv_array = np.array([deriv_num_calc(hm0,mzsq_m0mh,mzsq_m0ph),
                                        deriv_num_calc(hmhf,mzsq_mhfmh,mzsq_mhfph),
                                        deriv_num_calc(hA0,mzsq_A0mh,mzsq_A0ph),
                                        deriv_num_calc(hmu0,mzsq_mu0mh,mzsq_mu0ph)])
                sens_params = np.sort(np.array([(mp.fabs((mym0
                                                         / mymzsq)#/ mymzsq)
                                                        * deriv_array[0]),
                                                 'Delta_BG(m_0)'),
                                                (mp.fabs((mymhf
                                                         / mymzsq)
                                                        * deriv_array[1]),
                                                 'Delta_BG(m_1/2)'),
                                                (mp.fabs((myA0
                                                         / mymzsq)
                                                        * deriv_array[2]),
                                                 'Delta_BG(A_0)'),
                                                (mp.fabs((mymu0
                                                         / mymzsq)
                                                        * deriv_array[3]),
                                                 'Delta_BG(mu)')],
                                               dtype=[('BGContrib', object),
                                                      ('BGlabel', 'U40')]),
                                      order='BGContrib')
    elif (modselno == 2):
        print("Computing derivatives...")
        print("NOTE: this computation can take a while")
        mym0 = mpf(str(inputGUT_BCs[27]))
        mymHud0 = mp.sqrt(str(max([abs(inputGUT_BCs[25]),
                                   abs(inputGUT_BCs[26])])))
        mymhf = mpf(str(inputGUT_BCs[3]))
        myA0 = mpf(str(inputGUT_BCs[16])) / mpf(str(inputGUT_BCs[7]))
        mymusq0 = mpf(str(inputGUT_BCs[6]))
        mymu0 = mp.sqrt(str(abs(inputGUT_BCs[6])))

        def deriv_step_calc_mu0(shift_amt):
            """
            Do the current mass derivative step for higgsino mass mu.

            Parameters
            ----------
            shift_amt : Float.
                How much to shift for current derivative step.

            Returns
            -------
            mytestmzsq : Float.
                Return current shifted value of mZ^2 with the
                    higgsino mass shifted by the specified amount.

            """
            testBCs = deepcopy(inputGUT_BCs[:])
            for i in np.arange(27,42):
                testBCs[i] = mp.power(testBCs[i], 2)
            testBCs[6] = np.sign(mymusq0) * mp.power(mymu0 + shift_amt, 2)
            mytestmzsq = deriv_step_calc(GUT_SCALE,myweakscale, inptanbval, polecalccheck, deepcopy(testBCs))
            del testBCs
            return mytestmzsq

        def deriv_step_calc_Higgs(shift_amt):
            """
            Do the current mass derivative step for Higgs masses.

            Parameters
            ----------
            shift_amt : Float.
                How much to shift for current derivative step.

            Returns
            -------
            mytestmzsq: Float.
                Return current shifted value of mZ^2 with the corresponding
                    Higgs masses shifted by the specified amount.

            """
            testBCs = deepcopy(inputGUT_BCs[:])
            for i in np.arange(25, 27):
                testBCs[i] = mp.power(mp.sqrt(str(abs(testBCs[i])))
                                      + shift_amt, 2)
            for i in np.arange(27, 42):
                testBCs[i] = mp.power(testBCs[i], 2)
            mytestmzsq = deriv_step_calc(GUT_SCALE,myweakscale, inptanbval, polecalccheck, deepcopy(testBCs))
            del testBCs
            return mytestmzsq

        def deriv_step_calc_sq_sl(shift_amt):
            """
            Do the current mass derivative step for squarks & sleptons.

            Parameters
            ----------
            shift_amt : Float.
                How much to shift for current derivative step.

            Returns
            -------
            mytestmzsq : Float.
                Return current shifted value of mZ^2 with the corresponding
                    squark and slepton masses shifted by the specified amount.

            """
            testBCs = deepcopy(inputGUT_BCs[:])
            for i in np.arange(27,42):
                testBCs[i] = mp.power(testBCs[i] + shift_amt, 2)
            mytestmzsq = deriv_step_calc(GUT_SCALE,myweakscale, inptanbval, polecalccheck, deepcopy(testBCs))
            del testBCs
            return mytestmzsq

        def deriv_step_calc_trilin(shift_amt):
            """
            Do the current mass derivative step for soft trilinear couplings.

            Parameters
            ----------
            shift_amt : Float.
                How much to shift each soft trilinear for current derivative step.

            Returns
            -------
            mytestmzsq : Float.
                Return current shifted value of mZ^2 with the
                    soft trilinear couplings shifted by the specified amount.

            """
            testBCs = deepcopy(inputGUT_BCs[:])
            for i in np.arange(27,42):
                testBCs[i] = mp.power(testBCs[i], 2)
            for i in np.arange(16,25):
                testBCs[i] = (((testBCs[i] / testBCs[i-9]) + shift_amt)
                              * testBCs[i-9])
            mytestmzsq = deriv_step_calc(GUT_SCALE,myweakscale, inptanbval, polecalccheck, deepcopy(testBCs))
            del testBCs
            return mytestmzsq

        def deriv_step_calc_gaugino(shift_amt):
            """
            Do the current mass derivative step for soft gaugino masses.

            Parameters
            ----------
            shift_amt : Float.
                How much to shift for current derivative step.

            Returns
            -------
            mytestmzsq : Float.
                Return current shifted value of mZ^2 with the corresponding
                    soft gaugino mass shifted by the specified amount.

            """
            testBCs = deepcopy(inputGUT_BCs[:])
            for i in np.arange(27,42):
                testBCs[i] = mp.power(testBCs[i], 2)
            for i in np.arange(3,6):
                testBCs[i] = testBCs[i] + shift_amt
            mytestmzsq = deriv_step_calc(GUT_SCALE,myweakscale, inptanbval, polecalccheck, deepcopy(testBCs))
            del testBCs
            return mytestmzsq

        if (precselno == 1):
            derivcount = 0
            hm0 = mpf(str(abs(mp.power(math.ulp(mym0), 1/9))))
            hmhf = mpf(str(abs(mp.power(math.ulp(mymhf), 1/9))))
            hA0 = mpf(str(abs(mp.power(math.ulp(myA0), 1/9))))
            hmu0 = mpf(str(abs(mp.power(math.ulp(mymu0), 1/9))))
            hmHud0 = mpf(str(abs(mp.power(math.ulp(mymHud0), 1/9))))
            with alive_bar(40, dual_line=True,
                           force_tty=True, title='Computations: ',
                           bar='brackets') as bar:
                bar.text = f'Derivatives: {derivcount}/5, please wait...'
                ##### Set up solutions for m_0 derivative #####
                print("Error estimate for m0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hm0, 8)))))
                mzsq_m0ph = deriv_step_calc_sq_sl(hm0)
                bar()
                # Two deviations to right
                mzsq_m0pph = deriv_step_calc_sq_sl(2 * hm0)
                bar()
                # Three deviations to right
                mzsq_m0ppph = deriv_step_calc_sq_sl(3 * hm0)
                bar()
                # Four deviations to right
                mzsq_m0pppph = deriv_step_calc_sq_sl(4 * hm0)
                bar()
                # Deviate m0 by small amount LEFT and square soft scalar masses for BCs
                mzsq_m0mh = deriv_step_calc_sq_sl((-1) * hm0)
                bar()
                # Two deviations to left
                mzsq_m0mmh = deriv_step_calc_sq_sl((-2) * hm0)
                bar()
                # Three deviations to left
                mzsq_m0mmmh = deriv_step_calc_sq_sl((-3) * hm0)
                bar()
                # Four deviations to left
                mzsq_m0mmmmh = deriv_step_calc_sq_sl((-4) * hm0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/5, please wait...'

                ##### Set up solutions for mHu,d^2 derivative #####
                print("Error estimate for mHu,d derivative: "
                      + str("{:.4e}".format(float(mp.power(hmHud0, 8)))))
                mzsq_mHud0ph = deriv_step_calc_Higgs(hmHud0)
                bar()
                # Two deviations to right
                mzsq_mHud0pph = deriv_step_calc_Higgs(2 * hmHud0)
                bar()
                # Three deviations to right
                mzsq_mHud0ppph = deriv_step_calc_Higgs(3 * hmHud0)
                bar()
                mzsq_mHud0pppph = deriv_step_calc_Higgs(4 * hmHud0)
                bar()
                # Deviate mHu,d by small amount left
                mzsq_mHud0mh = deriv_step_calc_Higgs((-1) * hmHud0)
                bar()
                # Two deviations to left
                mzsq_mHud0mmh = deriv_step_calc_Higgs((-2) * hmHud0)
                bar()
                # Three deviations to left
                mzsq_mHud0mmmh = deriv_step_calc_Higgs((-3) * hmHud0)
                bar()
                # Four deviations to left
                mzsq_mHud0mmmmh = deriv_step_calc_Higgs((-4) * hmHud0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/5, please wait...'

                ##### Set up solutions for m_1/2 derivative #####
                print("Error estimate for m_1/2 derivative: "
                      + str("{:.4e}".format(float(mp.power(hmhf, 8)))))
                mzsq_mhfph = deriv_step_calc_gaugino(hmhf)
                bar()
                # Two deviations to right
                mzsq_mhfpph = deriv_step_calc_gaugino(2 * hmhf)
                bar()
                # Three deviations to right
                mzsq_mhfppph = deriv_step_calc_gaugino(3 * hmhf)
                bar()
                # Four deviations to right
                mzsq_mhfpppph = deriv_step_calc_gaugino(4 * hmhf)
                bar()
                # Deviate m_1/2 by small amount left
                mzsq_mhfmh = deriv_step_calc_gaugino((-1) * hmhf)
                bar()
                # Two deviations to left
                mzsq_mhfmmh = deriv_step_calc_gaugino((-2) * hmhf)
                bar()
                # Three deviations to left
                mzsq_mhfmmmh = deriv_step_calc_gaugino((-3) * hmhf)
                bar()
                # Four deviations to left
                mzsq_mhfmmmmh = deriv_step_calc_gaugino((-4) * hmhf)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/5, please wait...'

                ##### Set up solutions for A_0 derivative #####
                print("Error estimate for A0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hA0, 8)))))
                mzsq_A0ph = deriv_step_calc_trilin(hA0)
                bar()
                # Two deviations to right
                mzsq_A0pph = deriv_step_calc_trilin(2 * hA0)
                bar()
                # Three deviations to right
                mzsq_A0ppph = deriv_step_calc_trilin(3 * hA0)
                bar()
                # Four deviations to right
                mzsq_A0pppph = deriv_step_calc_trilin(4 * hA0)
                bar()
                # Deviate A_0 by small amount left
                mzsq_A0mh = deriv_step_calc_trilin((-1) * hA0)
                bar()
                # Two deviations to left
                mzsq_A0mmh = deriv_step_calc_trilin((-2) * hA0)
                bar()
                # Three deviations to left
                mzsq_A0mmmh = deriv_step_calc_trilin((-3) * hA0)
                bar()
                # Four deviations to left
                mzsq_A0mmmmh = deriv_step_calc_trilin((-4) * hA0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/5, please wait...'

                ##### Set up solutions for mu derivative #####
                print("Error estimate for mu derivative: "
                      + str("{:.4e}".format(float(mp.power(hmu0, 8)))))
                mzsq_mu0ph = deriv_step_calc_mu0(hmu0)
                bar()
                # Two deviations to right
                mzsq_mu0pph = deriv_step_calc_mu0((2) * hmu0)
                bar()
                # Three deviations to right
                mzsq_mu0ppph = deriv_step_calc_mu0((3) * hmu0)
                bar()
                # Four deviations to right
                mzsq_mu0pppph = deriv_step_calc_mu0((4) * hmu0)
                bar()
                # Deviate mu_0 by small amount left
                mzsq_mu0mh = deriv_step_calc_mu0((-1) * hmu0)
                bar()
                # Two deviations to left
                mzsq_mu0mmh = deriv_step_calc_mu0((-2) * hmu0)
                bar()
                # Three deviations to left
                mzsq_mu0mmmh = deriv_step_calc_mu0((-3) * hmu0)
                bar()
                # Four deviations to left
                mzsq_mu0mmmmh = deriv_step_calc_mu0((-4) * hmu0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/5, please wait...'
                print("Done!")

                # Construct derivative array
                deriv_array = np.array([deriv_num_calc(hm0,mzsq_m0mmmmh,mzsq_m0mmmh,
                                                       mzsq_m0mmh,mzsq_m0mh,mzsq_m0ph,
                                                       mzsq_m0pph,mzsq_m0ppph,mzsq_m0pppph),
                                        deriv_num_calc(hmhf,mzsq_mhfmmmmh,mzsq_mhfmmmh,
                                                       mzsq_mhfmmh,mzsq_mhfmh,mzsq_mhfph,
                                                       mzsq_mhfpph,mzsq_mhfppph,mzsq_mhfpppph),
                                        deriv_num_calc(hA0,mzsq_A0mmmmh,mzsq_A0mmmh,
                                                       mzsq_A0mmh,mzsq_A0mh,mzsq_A0ph,
                                                       mzsq_A0pph,mzsq_A0ppph,mzsq_A0pppph),
                                        deriv_num_calc(hmu0,mzsq_mu0mmmmh,mzsq_mu0mmmh,
                                                       mzsq_mu0mmh,mzsq_mu0mh,mzsq_mu0ph,
                                                       mzsq_mu0pph,mzsq_mu0ppph,mzsq_mu0pppph),
                                        deriv_num_calc(hmHud0,mzsq_mHud0mmmmh,mzsq_mHud0mmmh,
                                                       mzsq_mHud0mmh,mzsq_mHud0mh,mzsq_mHud0ph,
                                                       mzsq_mHud0pph,mzsq_mHud0ppph,mzsq_mHud0pppph)])
                sens_params = np.sort(np.array([(mp.fabs((mym0
                                                         / mymzsq)
                                                        * deriv_array[0]),
                                                 'Delta_BG(m_0)'),
                                                (mp.fabs((mymhf
                                                         / mymzsq)
                                                        * deriv_array[1]),
                                                 'Delta_BG(m_1/2)'),
                                                (mp.fabs((myA0
                                                         / mymzsq)
                                                        * deriv_array[2]),
                                                 'Delta_BG(A_0)'),
                                                (mp.fabs((mymu0
                                                         / mymzsq)
                                                        * deriv_array[3]),
                                                 'Delta_BG(mu)'),
                                                (mp.fabs((mymHud0 / mymzsq)
                                                        * deriv_array[4]),
                                                 'Delta_BG(mHu,d)')],
                                               dtype=[('BGContrib', float),
                                                      ('BGlabel', 'U40')]),
                                      order='BGContrib')
        elif (precselno == 2):
            derivcount = 0
            hm0 = mpf(str(abs(mp.power(math.ulp(mym0), 1/5))))
            hmhf = mpf(str(abs(mp.power(math.ulp(mymhf), 1/5))))
            hA0 = mpf(str(abs(mp.power(math.ulp(myA0), 1/5))))
            hmu0 = mpf(str(abs(mp.power(math.ulp(mymu0), 1/5))))
            hmHud0 = mpf(str(abs(mp.power(math.ulp(mymHud0), 1/5))))
            with alive_bar(20, dual_line=True,
                           force_tty=True, title='Computations: ',
                           bar='brackets') as bar:
                bar.text = f'Derivatives: {derivcount}/5, please wait...'
                ##### Set up solutions for m_0 derivative #####
                print("Error estimate for m0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hm0, 4)))))
                mzsq_m0ph = deriv_step_calc_sq_sl(hm0)
                bar()
                # Two deviations to right
                mzsq_m0pph = deriv_step_calc_sq_sl(2 * hm0)
                bar()
                # Deviate m0 by small amount LEFT and square soft scalar masses for BCs
                mzsq_m0mh = deriv_step_calc_sq_sl((-1) * hm0)
                bar()
                # Two deviations to left
                mzsq_m0mmh = deriv_step_calc_sq_sl((-2) * hm0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/5, please wait...'

                ##### Set up solutions for mHu,d^2 derivative #####
                print("Error estimate for mHu,d derivative: "
                      + str("{:.4e}".format(float(mp.power(hmHud0, 4)))))
                mzsq_mHud0ph = deriv_step_calc_Higgs(hmHud0)
                bar()
                # Two deviations to right
                mzsq_mHud0pph = deriv_step_calc_Higgs(2 * hmHud0)
                bar()
                # Deviate mHu,d by small amount left
                mzsq_mHud0mh = deriv_step_calc_Higgs((-1) * hmHud0)
                bar()
                # Two deviations to left
                mzsq_mHud0mmh = deriv_step_calc_Higgs((-2) * hmHud0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/5, please wait...'

                ##### Set up solutions for m_1/2 derivative #####
                print("Error estimate for m_1/2 derivative: "
                      + str("{:.4e}".format(float(mp.power(hmhf, 4)))))
                mzsq_mhfph = deriv_step_calc_gaugino(hmhf)
                bar()
                # Two deviations to right
                mzsq_mhfpph = deriv_step_calc_gaugino(2 * hmhf)
                bar()
                # Deviate m_1/2 by small amount left
                mzsq_mhfmh = deriv_step_calc_gaugino((-1) * hmhf)
                bar()
                # Two deviations to left
                mzsq_mhfmmh = deriv_step_calc_gaugino((-2) * hmhf)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/5, please wait...'

                ##### Set up solutions for A_0 derivative #####
                print("Error estimate for A0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hA0, 4)))))
                mzsq_A0ph = deriv_step_calc_trilin(hA0)
                bar()
                # Two deviations to right
                mzsq_A0pph = deriv_step_calc_trilin(2 * hA0)
                bar()
                # Deviate A_0 by small amount left
                mzsq_A0mh = deriv_step_calc_trilin((-1) * hA0)
                bar()
                # Two deviations to left
                mzsq_A0mmh = deriv_step_calc_trilin((-2) * hA0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/5, please wait...'

                ##### Set up solutions for mu derivative #####
                print("Error estimate for mu derivative: "
                      + str("{:.4e}".format(float(mp.power(hmu0, 4)))))
                mzsq_mu0ph = deriv_step_calc_mu0(hmu0)
                bar()
                # Two deviations to right
                mzsq_mu0pph = deriv_step_calc_mu0((2) * hmu0)
                bar()
                # Deviate mu_0 by small amount left
                mzsq_mu0mh = deriv_step_calc_mu0((-1) * hmu0)
                bar()
                # Two deviations to left
                mzsq_mu0mmh = deriv_step_calc_mu0((-2) * hmu0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/5, please wait...'
                print("Done!")

                # Construct derivative array
                deriv_array = np.array([deriv_num_calc(hm0,
                                                       mzsq_m0mmh,mzsq_m0mh,mzsq_m0ph,
                                                       mzsq_m0pph),
                                        deriv_num_calc(hmhf,
                                                       mzsq_mhfmmh,mzsq_mhfmh,mzsq_mhfph,
                                                       mzsq_mhfpph),
                                        deriv_num_calc(hA0,
                                                       mzsq_A0mmh,mzsq_A0mh,mzsq_A0ph,
                                                       mzsq_A0pph),
                                        deriv_num_calc(hmu0,
                                                       mzsq_mu0mmh,mzsq_mu0mh,mzsq_mu0ph,
                                                       mzsq_mu0pph),
                                        deriv_num_calc(hmHud0,
                                                       mzsq_mHud0mmh,mzsq_mHud0mh,mzsq_mHud0ph,
                                                       mzsq_mHud0pph)])
                sens_params = np.sort(np.array([(mp.fabs((mym0
                                                         / mymzsq)
                                                        * deriv_array[0]),
                                                 'Delta_BG(m_0)'),
                                                (mp.fabs((mymhf
                                                         / mymzsq)
                                                        * deriv_array[1]),
                                                 'Delta_BG(m_1/2)'),
                                                (mp.fabs((myA0
                                                         / mymzsq)
                                                        * deriv_array[2]),
                                                 'Delta_BG(A_0)'),
                                                (mp.fabs((mymu0
                                                         / mymzsq)
                                                        * deriv_array[3]),
                                                 'Delta_BG(mu)'),
                                                (mp.fabs((mymHud0 / mymzsq)
                                                        * deriv_array[4]),
                                                 'Delta_BG(mHu,d)')],
                                               dtype=[('BGContrib', float),
                                                      ('BGlabel', 'U40')]),
                                      order='BGContrib')
        elif (precselno == 3):
            derivcount = 0
            hm0 = mpf(str(abs(mp.power(math.ulp(mym0), 1/3))))
            hmhf = mpf(str(abs(mp.power(math.ulp(mymhf), 1/3))))
            hA0 = mpf(str(abs(mp.power(math.ulp(myA0), 1/3))))
            hmu0 = mpf(str(abs(mp.power(math.ulp(mymu0), 1/3))))
            hmHud0 = mpf(str(abs(mp.power(math.ulp(mymHud0), 1/3))))
            with alive_bar(10, dual_line=True,
                           force_tty=True, title='Computations: ',
                           bar='brackets') as bar:
                bar.text = f'Derivatives: {derivcount}/5, please wait...'
                ##### Set up solutions for m_0 derivative #####
                print("Error estimate for m0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hm0, 4)))))
                mzsq_m0ph = deriv_step_calc_sq_sl(hm0)
                bar()
                # Deviate m0 by small amount LEFT and square soft scalar masses for BCs
                mzsq_m0mh = deriv_step_calc_sq_sl((-1) * hm0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/5, please wait...'

                ##### Set up solutions for mHu,d^2 derivative #####
                print("Error estimate for mHu,d derivative: "
                      + str("{:.4e}".format(float(mp.power(hmHud0, 4)))))
                mzsq_mHud0ph = deriv_step_calc_Higgs(hmHud0)
                bar()
                # Deviate mHu,d by small amount left
                mzsq_mHud0mh = deriv_step_calc_Higgs((-1) * hmHud0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/5, please wait...'

                ##### Set up solutions for m_1/2 derivative #####
                print("Error estimate for m_1/2 derivative: "
                      + str("{:.4e}".format(float(mp.power(hmhf, 4)))))
                mzsq_mhfph = deriv_step_calc_gaugino(hmhf)
                bar()
                # Deviate m_1/2 by small amount left
                mzsq_mhfmh = deriv_step_calc_gaugino((-1) * hmhf)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/5, please wait...'

                ##### Set up solutions for A_0 derivative #####
                print("Error estimate for A0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hA0, 4)))))
                mzsq_A0ph = deriv_step_calc_trilin(hA0)
                bar()
                # Deviate A_0 by small amount left
                mzsq_A0mh = deriv_step_calc_trilin((-1) * hA0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/5, please wait...'

                ##### Set up solutions for mu derivative #####
                print("Error estimate for mu derivative: "
                      + str("{:.4e}".format(float(mp.power(hmu0, 4)))))
                mzsq_mu0ph = deriv_step_calc_mu0(hmu0)
                bar()
                # Deviate mu_0 by small amount left
                mzsq_mu0mh = deriv_step_calc_mu0((-1) * hmu0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/5, please wait...'
                print("Done!")

                # Construct derivative array
                deriv_array = np.array([deriv_num_calc(hm0,mzsq_m0mh,mzsq_m0ph),
                                        deriv_num_calc(hmhf,mzsq_mhfmh,mzsq_mhfph),
                                        deriv_num_calc(hA0,mzsq_A0mh,mzsq_A0ph),
                                        deriv_num_calc(hmu0,mzsq_mu0mh,mzsq_mu0ph),
                                        deriv_num_calc(hmHud0,mzsq_mHud0mh,mzsq_mHud0ph)])
                sens_params = np.sort(np.array([(mp.fabs((mym0
                                                         / mymzsq)
                                                        * deriv_array[0]),
                                                 'Delta_BG(m_0)'),
                                                (mp.fabs((mymhf
                                                         / mymzsq)
                                                        * deriv_array[1]),
                                                 'Delta_BG(m_1/2)'),
                                                (mp.fabs((myA0
                                                         / mymzsq)
                                                        * deriv_array[2]),
                                                 'Delta_BG(A_0)'),
                                                (mp.fabs((mymu0
                                                         / mymzsq)
                                                        * deriv_array[3]),
                                                 'Delta_BG(mu)'),
                                                (mp.fabs((mymHud0 / mymzsq)
                                                        * deriv_array[4]),
                                                 'Delta_BG(mHu,d)')],
                                               dtype=[('BGContrib', float),
                                                      ('BGlabel', 'U40')]),
                                      order='BGContrib')
    elif (modselno == 3):
        print("Computing sensitivity coefficient derivatives...")
        print("NOTE: this computation can take a while")
        mym0 = mpf(str(inputGUT_BCs[27]))
        mymhf = mpf(str(inputGUT_BCs[3]))
        myA0 = mpf(str(inputGUT_BCs[16])) / mpf(str(inputGUT_BCs[7]))
        mymu0 = mp.sqrt(abs(inputGUT_BCs[6]))
        mymusq0 = inputGUT_BCs[6]
        mymHu0 = mp.sqrt(abs(inputGUT_BCs[25]))
        mymHd0 = mp.sqrt(abs(inputGUT_BCs[26]))
        def deriv_step_calc_mu0(shift_amt):
            """
            Do the current mass derivative step for higgsino mass mu.

            Parameters
            ----------
            shift_amt : Float.
                How much to shift for current derivative step.

            Returns
            -------
            mytestmzsq : Float.
                Return current shifted value of mZ^2 with the
                    higgsino mass shifted by the specified amount.

            """
            testBCs = deepcopy(inputGUT_BCs[:])
            for i in np.arange(27,42):
                testBCs[i] = mp.power(testBCs[i], 2)
            testBCs[6] = np.sign(mymusq0) * mp.power(mymu0 + shift_amt, 2)
            mytestmzsq = deriv_step_calc(GUT_SCALE,myweakscale, inptanbval, polecalccheck, deepcopy(testBCs))
            del testBCs
            return mytestmzsq

        def deriv_step_calc_Higgs(current_index, shift_amt):
            """
            Do the current mass derivative step for soft Higgs masses.

            Parameters
            ----------
            current_index : Int.
                Index of parameter to perform derivative with.
            shift_amt : Float.
                How much to shift for current derivative step.

            Returns
            -------
            mytestmzsq : Float.
                Return current shifted value of mZ^2 with the corresponding
                    soft Higgs mass shifted by the specified amount.

            """
            testBCs = deepcopy(inputGUT_BCs[:])
            for i in np.arange(27,42):
                testBCs[i] = mp.power(testBCs[i], 2)
            testBCs[current_index] = mp.power(mp.sqrt(str(abs(testBCs[current_index])))
                                              + shift_amt, 2)
            mytestmzsq = deriv_step_calc(GUT_SCALE,myweakscale, inptanbval, polecalccheck, deepcopy(testBCs))
            del testBCs
            return mytestmzsq

        def deriv_step_calc_sq_sl(shift_amt):
            """
            Do the current mass derivative step for squarks & sleptons.

            Parameters
            ----------
            shift_amt : Float.
                How much to shift for current derivative step.

            Returns
            -------
            mytestmzsq : Float.
                Return current shifted value of mZ^2 with the corresponding
                    squark and slepton masses shifted by the specified amount.

            """
            testBCs = deepcopy(inputGUT_BCs[:])
            for i in np.arange(27,42):
                testBCs[i] = mp.power(testBCs[i] + shift_amt, 2)
            mytestmzsq = deriv_step_calc(GUT_SCALE,myweakscale, inptanbval, polecalccheck, deepcopy(testBCs))
            del testBCs
            return mytestmzsq

        def deriv_step_calc_trilin(shift_amt):
            """
            Do the current mass derivative step for soft trilinear couplings.

            Parameters
            ----------
            shift_amt : Float.
                How much to shift each soft trilinear for current derivative step.

            Returns
            -------
            mytestmzsq : Float.
                Return current shifted value of mZ^2 with the
                    soft trilinear couplings shifted by the specified amount.

            """
            testBCs = deepcopy(inputGUT_BCs[:])
            for i in np.arange(27,42):
                testBCs[i] = mp.power(testBCs[i], 2)
            for i in np.arange(16,25):
                testBCs[i] = (((testBCs[i] / testBCs[i-9]) + shift_amt)
                              * testBCs[i-9])
            mytestmzsq = deriv_step_calc(GUT_SCALE,myweakscale, inptanbval, polecalccheck, deepcopy(testBCs))
            del testBCs
            return mytestmzsq

        def deriv_step_calc_gaugino(shift_amt):
            """
            Do the current mass derivative step for soft gaugino masses.

            Parameters
            ----------
            shift_amt : Float.
                How much to shift for current derivative step.

            Returns
            -------
            mytestmzsq : Float.
                Return current shifted value of mZ^2 with the corresponding
                    soft gaugino mass shifted by the specified amount.

            """
            testBCs = deepcopy(inputGUT_BCs[:])
            for i in np.arange(27,42):
                testBCs[i] = mp.power(testBCs[i], 2)
            for i in np.arange(3,6):
                testBCs[i] = testBCs[i] + shift_amt
            mytestmzsq = deriv_step_calc(GUT_SCALE,myweakscale, inptanbval, polecalccheck, deepcopy(testBCs))
            del testBCs
            return mytestmzsq

        if (precselno == 1):
            derivcount = 0
            hm0 = mpf(str(abs(mp.power(math.ulp(mym0), 1/9))))
            hmhf = mpf(str(abs(mp.power(math.ulp(mymhf), 1/9))))
            hA0 = mpf(str(abs(mp.power(math.ulp(myA0), 1/9))))
            hmu0 = mpf(str(abs(mp.power(math.ulp(mymu0), 1/9))))
            hmHu0 = mpf(str(abs(mp.power(math.ulp(mymHu0), 1/9))))
            hmHd0 = mpf(str(abs(mp.power(math.ulp(mymHd0), 1/9))))
            mzsq_mHph = np.zeros(2)
            mzsq_mHpph = np.zeros(2)
            mzsq_mHppph = np.zeros(2)
            mzsq_mHpppph = np.zeros(2)
            mzsq_mHmh = np.zeros(2)
            mzsq_mHmmh = np.zeros(2)
            mzsq_mHmmmh = np.zeros(2)
            mzsq_mHmmmmh = np.zeros(2)
            Higgs_hvals = [hmHu0, hmHd0]
            Higgs_labels = ['m_Hu0', 'm_Hd0']
            with alive_bar(48, dual_line=True,
                           force_tty=True, title='Computations: ',
                           bar='brackets') as bar:
                bar.text = f'Derivatives: {derivcount}/6, please wait...'
                print("Error estimate for m0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hm0, 8)))))
                ##### Set up solutions for m_0 derivative #####
                mzsq_m0ph = deriv_step_calc_sq_sl(hm0)
                bar()
                # Two deviations to right
                mzsq_m0pph = deriv_step_calc_sq_sl(2 * hm0)
                bar()
                # Three deviations to right
                mzsq_m0ppph = deriv_step_calc_sq_sl(3 * hm0)
                bar()
                # Four deviations to right
                mzsq_m0pppph = deriv_step_calc_sq_sl(4 * hm0)
                bar()
                # Deviate m0 by small amount LEFT and square soft scalar masses for BCs
                mzsq_m0mh = deriv_step_calc_sq_sl((-1) * hm0)
                bar()
                # Two deviations to left
                mzsq_m0mmh = deriv_step_calc_sq_sl((-2) * hm0)
                bar()
                # Three deviations to left
                mzsq_m0mmmh = deriv_step_calc_sq_sl((-3) * hm0)
                bar()
                # Four deviations to left
                mzsq_m0mmmmh = deriv_step_calc_sq_sl((-4) * hm0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/6, please wait...'

                ##### Set up solutions for mHu and mHd derivatives #####
                for j in np.arange(25, 27):
                    print("Error estimate for " + str(Higgs_labels[j-25]) + " derivative: "
                          + str("{:.4e}".format(float(mp.power(Higgs_hvals[j-25], 8)))))
                    # For loop to compute soft Higgs mass derivatives
                    # Right shifts
                    mzsq_mHph[j-25] = deriv_step_calc_Higgs(j, Higgs_hvals[j-25])
                    bar()
                    mzsq_mHpph[j-25] = deriv_step_calc_Higgs(j,
                                                             (2 * Higgs_hvals[j-25]))
                    bar()
                    mzsq_mHppph[j-25] = deriv_step_calc_Higgs(j,
                                                              (3 * Higgs_hvals[j-25]))
                    bar()
                    mzsq_mHpppph[j-25] = deriv_step_calc_Higgs(j,
                                                               (4 * Higgs_hvals[j-25]))
                    bar()
                    # Left shifts
                    mzsq_mHmh[j-25] = deriv_step_calc_Higgs(j, (-1) * Higgs_hvals[j-25])
                    bar()
                    mzsq_mHmmh[j-25] = deriv_step_calc_Higgs(j,
                                                             (-1) * (2 * Higgs_hvals[j-25]))
                    bar()
                    mzsq_mHmmmh[j-25] = deriv_step_calc_Higgs(j,
                                                              (-1) * (3 * Higgs_hvals[j-25]))
                    bar()
                    mzsq_mHmmmmh[j-25] = deriv_step_calc_Higgs(j,
                                                               (-1) * (4 * Higgs_hvals[j-25]))
                    bar()
                    derivcount += 1
                    bar.text = f'Derivatives: {derivcount}/6, please wait...'

                ##### Set up solutions for m_1/2 derivative #####
                print("Error estimate for m_1/2 derivative: "
                      + str("{:.4e}".format(float(mp.power(hmhf, 8)))))
                mzsq_mhfph = deriv_step_calc_gaugino(hmhf)
                bar()
                # Two deviations to right
                mzsq_mhfpph = deriv_step_calc_gaugino(2 * hmhf)
                bar()
                # Three deviations to right
                mzsq_mhfppph = deriv_step_calc_gaugino(3 * hmhf)
                bar()
                # Four deviations to right
                mzsq_mhfpppph = deriv_step_calc_gaugino(4 * hmhf)
                bar()
                # Boundary conditions first
                mzsq_mhfmh = deriv_step_calc_gaugino((-1) * hmhf)
                bar()
                # Two deviations to left
                mzsq_mhfmmh = deriv_step_calc_gaugino((-2) * hmhf)
                bar()
                # Three deviations to left
                mzsq_mhfmmmh = deriv_step_calc_gaugino((-3) * hmhf)
                bar()
                # Four deviations to left
                mzsq_mhfmmmmh = deriv_step_calc_gaugino((-4) * hmhf)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/6, please wait...'

                print("Error estimate for A0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hA0, 8)))))
                ##### Set up solutions for A_0 derivative #####
                mzsq_A0ph = deriv_step_calc_trilin(hA0)
                bar()
                # Two deviations to right
                mzsq_A0pph = deriv_step_calc_trilin(2 * hA0)
                bar()
                # Three deviations to right
                mzsq_A0ppph = deriv_step_calc_trilin(3 * hA0)
                bar()
                # Four deviations to right
                mzsq_A0pppph = deriv_step_calc_trilin(4 * hA0)
                bar()
                # Boundary conditions first
                mzsq_A0mh = deriv_step_calc_trilin((-1) * hA0)
                bar()
                # Two deviations to left
                mzsq_A0mmh = deriv_step_calc_trilin((-2) * hA0)
                bar()
                # Three deviations to left
                mzsq_A0mmmh = deriv_step_calc_trilin((-3) * hA0)
                bar()
                # Four deviations to left
                mzsq_A0mmmmh = deriv_step_calc_trilin((-4) * hA0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/6, please wait...'

                print("Error estimate for mu0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hmu0, 8)))))
                ##### Set up solutions for mu derivative #####
                mzsq_mu0ph = deriv_step_calc_mu0(hmu0)
                bar()
                # Two deviations to right
                mzsq_mu0pph = deriv_step_calc_mu0(2 * hmu0)
                bar()
                # Three deviations to right
                mzsq_mu0ppph = deriv_step_calc_mu0(3 * hmu0)
                bar()
                # Four deviations to right
                mzsq_mu0pppph = deriv_step_calc_mu0(4 * hmu0)
                bar()
                # Deviate mu_0 by small amount left
                mzsq_mu0mh = deriv_step_calc_mu0((-1) * hmu0)
                bar()
                # Two deviations to left
                mzsq_mu0mmh = deriv_step_calc_mu0((-2) * hmu0)
                bar()
                # Three deviations to left
                mzsq_mu0mmmh = deriv_step_calc_mu0((-3) * hmu0)
                bar()
                # Four deviations to left
                mzsq_mu0mmmmh = deriv_step_calc_mu0((-4) * hmu0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/6, please wait...'

                # Construct derivative array
                deriv_array = np.array([deriv_num_calc(hm0,mzsq_m0mmmmh,mzsq_m0mmmh,
                                                       mzsq_m0mmh,mzsq_m0mh,mzsq_m0ph,
                                                       mzsq_m0pph,mzsq_m0ppph,mzsq_m0pppph),
                                        deriv_num_calc(hmhf,mzsq_mhfmmmmh,mzsq_mhfmmmh,
                                                       mzsq_mhfmmh,mzsq_mhfmh,mzsq_mhfph,
                                                       mzsq_mhfpph,mzsq_mhfppph,mzsq_mhfpppph),
                                        deriv_num_calc(hA0,mzsq_A0mmmmh,mzsq_A0mmmh,
                                                       mzsq_A0mmh,mzsq_A0mh,mzsq_A0ph,
                                                       mzsq_A0pph,mzsq_A0ppph,mzsq_A0pppph),
                                        deriv_num_calc(hmu0,mzsq_mu0mmmmh,mzsq_mu0mmmh,
                                                       mzsq_mu0mmh,mzsq_mu0mh,mzsq_mu0ph,
                                                       mzsq_mu0pph,mzsq_mu0ppph,mzsq_mu0pppph),
                                        deriv_num_calc(Higgs_hvals[0],mzsq_mHmmmmh[0],mzsq_mHmmmh[0],
                                                       mzsq_mHmmh[0],mzsq_mHmh[0],mzsq_mHph[0],
                                                       mzsq_mHpph[0],mzsq_mHppph[0],mzsq_mHpppph[0]),
                                        deriv_num_calc(Higgs_hvals[1],mzsq_mHmmmmh[1],mzsq_mHmmmh[1],
                                                       mzsq_mHmmh[1],mzsq_mHmh[1],mzsq_mHph[1],
                                                       mzsq_mHpph[1],mzsq_mHppph[1],mzsq_mHpppph[1])])
                sens_params = np.sort(np.array([(mp.fabs((mym0
                                                         / mymzsq)
                                                        * deriv_array[0]),
                                                 'Delta_BG(m_0)'),
                                                (mp.fabs((mymhf
                                                         / mymzsq)
                                                        * deriv_array[1]),
                                                 'Delta_BG(m_1/2)'),
                                                (mp.fabs((myA0
                                                         / mymzsq)
                                                        * deriv_array[2]),
                                                 'Delta_BG(A_0)'),
                                                (mp.fabs((mymu0
                                                         / mymzsq)
                                                        * deriv_array[3]),
                                                 'Delta_BG(mu)'),
                                                (mp.fabs((mymHu0 / mymzsq)
                                                        * deriv_array[4]),
                                                 'Delta_BG(mHu)'),
                                                (mp.fabs((mymHd0 / mymzsq)
                                                        * deriv_array[5]),
                                                 'Delta_BG(mHd)')],
                                               dtype=[('BGContrib', float),
                                                      ('BGlabel', 'U40')]),
                                      order='BGContrib')
                print("Done!")
        elif (precselno == 2):
            derivcount = 0
            hm0 = mpf(str(abs(mp.power(math.ulp(mym0), 1/5))))
            hmhf = mpf(str(abs(mp.power(math.ulp(mymhf), 1/5))))
            hA0 = mpf(str(abs(mp.power(math.ulp(myA0), 1/5))))
            hmu0 = mpf(str(abs(mp.power(math.ulp(mymu0), 1/5))))
            hmHu0 = mpf(str(abs(mp.power(math.ulp(mymHu0), 1/5))))
            hmHd0 = mpf(str(abs(mp.power(math.ulp(mymHd0), 1/5))))
            mzsq_mHph = np.zeros(2)
            mzsq_mHpph = np.zeros(2)
            mzsq_mHmh = np.zeros(2)
            mzsq_mHmmh = np.zeros(2)
            Higgs_hvals = [hmHu0, hmHd0]
            Higgs_labels = ['m_Hu0', 'm_Hd0']
            with alive_bar(24, dual_line=True,
                           force_tty=True, title='Computations: ',
                           bar='brackets') as bar:
                bar.text = f'Derivatives: {derivcount}/6, please wait...'
                print("Error estimate for m0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hm0, 4)))))
                ##### Set up solutions for m_0 derivative #####
                mzsq_m0ph = deriv_step_calc_sq_sl(hm0)
                bar()
                # Two deviations to right
                mzsq_m0pph = deriv_step_calc_sq_sl(2 * hm0)
                bar()
                # Deviate m0 by small amount LEFT and square soft scalar masses for BCs
                mzsq_m0mh = deriv_step_calc_sq_sl((-1) * hm0)
                bar()
                # Two deviations to left
                mzsq_m0mmh = deriv_step_calc_sq_sl((-2) * hm0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/6, please wait...'

                ##### Set up solutions for mHu and mHd derivatives #####
                for j in np.arange(25, 27):
                    print("Error estimate for " + str(Higgs_labels[j-25]) + " derivative: "
                          + str("{:.4e}".format(float(mp.power(Higgs_hvals[j-25], 4)))))
                    # For loop to compute soft Higgs mass derivatives
                    # Right shifts
                    mzsq_mHph[j-25] = deriv_step_calc_Higgs(j, Higgs_hvals[j-25])
                    bar()
                    mzsq_mHpph[j-25] = deriv_step_calc_Higgs(j,
                                                             (2 * Higgs_hvals[j-25]))
                    bar()
                    # Left shifts
                    mzsq_mHmh[j-25] = deriv_step_calc_Higgs(j, (-1) * Higgs_hvals[j-25])
                    bar()
                    mzsq_mHmmh[j-25] = deriv_step_calc_Higgs(j,
                                                             (-1) * (2 * Higgs_hvals[j-25]))
                    bar()
                    derivcount += 1
                    bar.text = f'Derivatives: {derivcount}/6, please wait...'

                ##### Set up solutions for m_1/2 derivative #####
                print("Error estimate for m_1/2 derivative: "
                      + str("{:.4e}".format(float(mp.power(hmhf, 4)))))
                mzsq_mhfph = deriv_step_calc_gaugino(hmhf)
                bar()
                # Two deviations to right
                mzsq_mhfpph = deriv_step_calc_gaugino(2 * hmhf)
                bar()
                # Boundary conditions to left first
                mzsq_mhfmh = deriv_step_calc_gaugino((-1) * hmhf)
                bar()
                # Two deviations to left
                mzsq_mhfmmh = deriv_step_calc_gaugino((-2) * hmhf)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/6, please wait...'

                print("Error estimate for A0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hA0, 4)))))
                ##### Set up solutions for A_0 derivative #####
                mzsq_A0ph = deriv_step_calc_trilin(hA0)
                bar()
                # Two deviations to right
                mzsq_A0pph = deriv_step_calc_trilin(2 * hA0)
                bar()
                # Boundary conditions first
                mzsq_A0mh = deriv_step_calc_trilin((-1) * hA0)
                bar()
                # Two deviations to left
                mzsq_A0mmh = deriv_step_calc_trilin((-2) * hA0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/6, please wait...'

                print("Error estimate for mu0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hmu0, 4)))))
                ##### Set up solutions for mu derivative #####
                mzsq_mu0ph = deriv_step_calc_mu0(hmu0)
                bar()
                # Two deviations to right
                mzsq_mu0pph = deriv_step_calc_mu0(2 * hmu0)
                bar()
                # Deviate mu_0 by small amount left
                mzsq_mu0mh = deriv_step_calc_mu0((-1) * hmu0)
                bar()
                # Two deviations to left
                mzsq_mu0mmh = deriv_step_calc_mu0((-2) * hmu0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/6, please wait...'

                # Construct derivative array
                deriv_array = np.array([deriv_num_calc(hm0,
                                                       mzsq_m0mmh,mzsq_m0mh,mzsq_m0ph,
                                                       mzsq_m0pph),
                                        deriv_num_calc(hmhf,
                                                       mzsq_mhfmmh,mzsq_mhfmh,mzsq_mhfph,
                                                       mzsq_mhfpph),
                                        deriv_num_calc(hA0,
                                                       mzsq_A0mmh,mzsq_A0mh,mzsq_A0ph,
                                                       mzsq_A0pph),
                                        deriv_num_calc(hmu0,
                                                       mzsq_mu0mmh,mzsq_mu0mh,mzsq_mu0ph,
                                                       mzsq_mu0pph),
                                        deriv_num_calc(Higgs_hvals[0],
                                                       mzsq_mHmmh[0],mzsq_mHmh[0],mzsq_mHph[0],
                                                       mzsq_mHpph[0]),
                                        deriv_num_calc(Higgs_hvals[1],
                                                       mzsq_mHmmh[1],mzsq_mHmh[1],mzsq_mHph[1],
                                                       mzsq_mHpph[1])])
                sens_params = np.sort(np.array([(mp.fabs((mym0
                                                         / mymzsq)
                                                        * deriv_array[0]),
                                                 'Delta_BG(m_0)'),
                                                (mp.fabs((mymhf
                                                         / mymzsq)
                                                        * deriv_array[1]),
                                                 'Delta_BG(m_1/2)'),
                                                (mp.fabs((myA0
                                                         / mymzsq)
                                                        * deriv_array[2]),
                                                 'Delta_BG(A_0)'),
                                                (mp.fabs((mymu0
                                                         / mymzsq)
                                                        * deriv_array[3]),
                                                 'Delta_BG(mu)'),
                                                (mp.fabs((mymHu0 / mymzsq)
                                                        * deriv_array[4]),
                                                 'Delta_BG(mHu)'),
                                                (mp.fabs((mymHd0 / mymzsq)
                                                        * deriv_array[5]),
                                                 'Delta_BG(mHd)')],
                                               dtype=[('BGContrib', float),
                                                      ('BGlabel', 'U40')]),
                                      order='BGContrib')
                print("Done!")
        elif (precselno == 3):
            derivcount = 0
            hm0 = mpf(str(abs(mp.power(math.ulp(mym0), 1/3))))
            hmhf = mpf(str(abs(mp.power(math.ulp(mymhf), 1/3))))
            hA0 = mpf(str(abs(mp.power(math.ulp(myA0), 1/3))))
            hmu0 = mpf(str(abs(mp.power(math.ulp(mymu0), 1/3))))
            hmHu0 = mpf(str(abs(mp.power(math.ulp(mymHu0), 1/3))))
            hmHd0 = mpf(str(abs(mp.power(math.ulp(mymHd0), 1/3))))
            mzsq_mHph = np.zeros(2)
            mzsq_mHmh = np.zeros(2)
            Higgs_hvals = [hmHu0, hmHd0]
            Higgs_labels = ['m_Hu0', 'm_Hd0']
            with alive_bar(12, dual_line=True,
                           force_tty=True, title='Computations: ',
                           bar='brackets') as bar:
                bar.text = f'Derivatives: {derivcount}/6, please wait...'
                print("Error estimate for m0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hm0, 2)))))
                ##### Set up solutions for m_0 derivative #####
                mzsq_m0ph = deriv_step_calc_sq_sl(hm0)
                bar()
                # Deviate m0 by small amount LEFT and square soft scalar masses for BCs
                mzsq_m0mh = deriv_step_calc_sq_sl((-1) * hm0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/6, please wait...'

                ##### Set up solutions for mHu and mHd derivatives #####
                for j in np.arange(25, 27):
                    print("Error estimate for " + str(Higgs_labels[j-25]) + " derivative: "
                          + str("{:.4e}".format(float(mp.power(Higgs_hvals[j-25], 2)))))
                    # For loop to compute soft Higgs mass derivatives
                    # Right shifts
                    mzsq_mHph[j-25] = deriv_step_calc_Higgs(j, Higgs_hvals[j-25])
                    bar()
                    # Left shifts
                    mzsq_mHmh[j-25] = deriv_step_calc_Higgs(j, (-1) * Higgs_hvals[j-25])
                    bar()
                    derivcount += 1
                    bar.text = f'Derivatives: {derivcount}/6, please wait...'

                ##### Set up solutions for m_1/2 derivative #####
                print("Error estimate for m_1/2 derivative: "
                      + str("{:.4e}".format(float(mp.power(hmhf, 2)))))
                mzsq_mhfph = deriv_step_calc_gaugino(hmhf)
                bar()
                # Boundary conditions to left first
                mzsq_mhfmh = deriv_step_calc_gaugino((-1) * hmhf)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/6, please wait...'

                print("Error estimate for A0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hA0, 2)))))
                ##### Set up solutions for A_0 derivative #####
                mzsq_A0ph = deriv_step_calc_trilin(hA0)
                bar()
                # Boundary conditions first
                mzsq_A0mh = deriv_step_calc_trilin((-1) * hA0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/6, please wait...'

                print("Error estimate for mu0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hmu0, 2)))))
                ##### Set up solutions for mu derivative #####
                mzsq_mu0ph = deriv_step_calc_mu0(hmu0)
                bar()
                # Deviate mu_0 by small amount left
                mzsq_mu0mh = deriv_step_calc_mu0((-1) * hmu0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/6, please wait...'

                # Construct derivative array
                deriv_array = np.array([deriv_num_calc(hm0,mzsq_m0mh,mzsq_m0ph),
                                        deriv_num_calc(hmhf,mzsq_mhfmh,mzsq_mhfph),
                                        deriv_num_calc(hA0,mzsq_A0mh,mzsq_A0ph),
                                        deriv_num_calc(hmu0,mzsq_mu0mh,mzsq_mu0ph),
                                        deriv_num_calc(Higgs_hvals[0],mzsq_mHmh[0],mzsq_mHph[0]),
                                        deriv_num_calc(Higgs_hvals[1],mzsq_mHmh[1],mzsq_mHph[1])])
                sens_params = np.sort(np.array([(mp.fabs((mym0
                                                         / mymzsq)
                                                        * deriv_array[0]),
                                                 'Delta_BG(m_0)'),
                                                (mp.fabs((mymhf
                                                         / mymzsq)
                                                        * deriv_array[1]),
                                                 'Delta_BG(m_1/2)'),
                                                (mp.fabs((myA0
                                                         / mymzsq)
                                                        * deriv_array[2]),
                                                 'Delta_BG(A_0)'),
                                                (mp.fabs((mymu0
                                                         / mymzsq)
                                                        * deriv_array[3]),
                                                 'Delta_BG(mu)'),
                                                (mp.fabs((mymHu0 / mymzsq)
                                                        * deriv_array[4]),
                                                 'Delta_BG(mHu)'),
                                                (mp.fabs((mymHd0 / mymzsq)
                                                        * deriv_array[5]),
                                                 'Delta_BG(mHd)')],
                                               dtype=[('BGContrib', float),
                                                      ('BGlabel', 'U40')]),
                                      order='BGContrib')
                print("Done!")
    elif (modselno == 4):
        print("Computing sensitivity coefficient derivatives...")
        print("NOTE: this computation can take a while")
        mym012 = mpf(str(inputGUT_BCs[27]))
        mym03 = mpf(str(inputGUT_BCs[29]))
        mymhf = mpf(str(inputGUT_BCs[3]))
        myA0 = mpf(str(inputGUT_BCs[16])) / mpf(str(inputGUT_BCs[7]))
        mymu0 = mp.sqrt(abs(inputGUT_BCs[6]))
        mymusq0 = inputGUT_BCs[6]
        mymHu0 = mp.sqrt(abs(inputGUT_BCs[25]))
        mymHd0 = mp.sqrt(abs(inputGUT_BCs[26]))
        def deriv_step_calc_mu0(shift_amt):
            """
            Do the current mass derivative step for higgsino mass mu.

            Parameters
            ----------
            shift_amt : Float.
                How much to shift for current derivative step.

            Returns
            -------
            mytestmzsq : Float.
                Return current shifted value of mZ^2 with the
                    higgsino mass shifted by the specified amount.

            """
            testBCs = deepcopy(inputGUT_BCs[:])
            for i in np.arange(27,42):
                testBCs[i] = mp.power(testBCs[i], 2)
            testBCs[6] = np.sign(mymusq0) * mp.power(mymu0 + shift_amt, 2)
            mytestmzsq = deriv_step_calc(GUT_SCALE,myweakscale, inptanbval, polecalccheck, deepcopy(testBCs))
            del testBCs
            return mytestmzsq

        def deriv_step_calc_Higgs(current_index, shift_amt):
            """
            Do the current mass derivative step for soft Higgs masses.

            Parameters
            ----------
            current_index : Int.
                Index of parameter to perform derivative with.
            shift_amt : Float.
                How much to shift for current derivative step.

            Returns
            -------
            mytestmzsq : Float.
                Return current shifted value of mZ^2 with the corresponding
                    soft Higgs mass shifted by the specified amount.

            """
            testBCs = deepcopy(inputGUT_BCs[:])
            for i in np.arange(27,42):
                testBCs[i] = mp.power(testBCs[i], 2)
            testBCs[current_index] = mp.power(mp.sqrt(str(abs(testBCs[current_index])))
                                              + shift_amt, 2)
            mytestmzsq = deriv_step_calc(GUT_SCALE,myweakscale, inptanbval, polecalccheck, deepcopy(testBCs))
            del testBCs
            return mytestmzsq

        def deriv_step_calc_sq_sl12(shift_amt):
            """
            Do the current mass derivative step for squarks & sleptons of first 2 generations.

            Parameters
            ----------
            shift_amt : Float.
                How much to shift for current derivative step.

            Returns
            -------
            mytestmzsq : Float.
                Return current shifted value of mZ^2 with the corresponding
                    1st & 2nd gen. squark and slepton masses shifted by the specified amount.

            """
            testBCs = deepcopy(inputGUT_BCs[:])
            m012_indices = [27,28,30,31,33,34,36,37,39,40]
            for i in m012_indices:
                testBCs[i] = mp.power(testBCs[i] + shift_amt, 2)
            m03_indices = [29,32,35,38,41]
            for i in m03_indices:
                testBCs[i] = mp.power(testBCs[i], 2)
            mytestmzsq = deriv_step_calc(GUT_SCALE,myweakscale, inptanbval, polecalccheck, deepcopy(testBCs))
            del testBCs
            return mytestmzsq

        def deriv_step_calc_sq_sl3(shift_amt):
            """
            Do the current mass derivative step for squarks & sleptons of 3rd generation.

            Parameters
            ----------
            shift_amt : Float.
                How much to shift for current derivative step.

            Returns
            -------
            mytestmzsq : Float.
                Return current shifted value of mZ^2 with the corresponding
                    3rd gen. squark and slepton masses shifted by the specified amount.

            """
            testBCs = deepcopy(inputGUT_BCs[:])
            m012_indices = [27,28,30,31,33,34,36,37,39,40]
            for i in m012_indices:
                testBCs[i] = mp.power(testBCs[i], 2)
            m03_indices = [29,32,35,38,41]
            for i in m03_indices:
                testBCs[i] = mp.power(testBCs[i] + shift_amt, 2)
            mytestmzsq = deriv_step_calc(GUT_SCALE,myweakscale, inptanbval, polecalccheck, deepcopy(testBCs))
            del testBCs
            return mytestmzsq

        def deriv_step_calc_trilin(shift_amt):
            """
            Do the current mass derivative step for soft trilinear couplings.

            Parameters
            ----------
            shift_amt : Float.
                How much to shift each soft trilinear for current derivative step.

            Returns
            -------
            mytestmzsq : Float.
                Return current shifted value of mZ^2 with the
                    soft trilinear couplings shifted by the specified amount.

            """
            testBCs = deepcopy(inputGUT_BCs[:])
            for i in np.arange(27,42):
                testBCs[i] = mp.power(testBCs[i], 2)
            for i in np.arange(16,25):
                testBCs[i] = (((testBCs[i] / testBCs[i-9]) + shift_amt)
                              * testBCs[i-9])
            mytestmzsq = deriv_step_calc(GUT_SCALE,myweakscale, inptanbval, polecalccheck, deepcopy(testBCs))
            del testBCs
            return mytestmzsq

        def deriv_step_calc_gaugino(shift_amt):
            """
            Do the current mass derivative step for soft gaugino masses.

            Parameters
            ----------
            shift_amt : Float.
                How much to shift for current derivative step.

            Returns
            -------
            mytestmzsq : Float.
                Return current shifted value of mZ^2 with the corresponding
                    soft gaugino mass shifted by the specified amount.

            """
            testBCs = deepcopy(inputGUT_BCs[:])
            for i in np.arange(27,42):
                testBCs[i] = mp.power(testBCs[i], 2)
            for i in np.arange(3,6):
                testBCs[i] = testBCs[i] + shift_amt
            mytestmzsq = deriv_step_calc(GUT_SCALE,myweakscale, inptanbval, polecalccheck, deepcopy(testBCs))
            del testBCs
            return mytestmzsq

        if (precselno == 1):
            derivcount = 0
            hm012 = mpf(str(abs(mp.power(math.ulp(mym012), 1/9))))
            hm03 = mpf(str(abs(mp.power(math.ulp(mym03), 1/9))))
            hmhf = mpf(str(abs(mp.power(math.ulp(mymhf), 1/9))))
            hA0 = mpf(str(abs(mp.power(math.ulp(myA0), 1/9))))
            hmu0 = mpf(str(abs(mp.power(math.ulp(mymu0), 1/9))))
            hmHu0 = mpf(str(abs(mp.power(math.ulp(mymHu0), 1/9))))
            hmHd0 = mpf(str(abs(mp.power(math.ulp(mymHd0), 1/9))))
            mzsq_mHph = np.zeros(2)
            mzsq_mHpph = np.zeros(2)
            mzsq_mHppph = np.zeros(2)
            mzsq_mHpppph = np.zeros(2)
            mzsq_mHmh = np.zeros(2)
            mzsq_mHmmh = np.zeros(2)
            mzsq_mHmmmh = np.zeros(2)
            mzsq_mHmmmmh = np.zeros(2)
            Higgs_hvals = [hmHu0, hmHd0]
            Higgs_labels = ['m_Hu0', 'm_Hd0']
            with alive_bar(56, dual_line=True,
                           force_tty=True, title='Computations: ',
                           bar='brackets') as bar:
                bar.text = f'Derivatives: {derivcount}/7, please wait...'
                print("Error estimate for m0(1,2) derivative: "
                      + str("{:.4e}".format(float(mp.power(hm012, 8)))))
                ##### Set up solutions for m_0(1,2) derivative #####
                mzsq_m012ph = deriv_step_calc_sq_sl12(hm012)
                bar()
                # Two deviations to right
                mzsq_m012pph = deriv_step_calc_sq_sl12(2 * hm012)
                bar()
                # Three deviations to right
                mzsq_m012ppph = deriv_step_calc_sq_sl12(3 * hm012)
                bar()
                # Four deviations to right
                mzsq_m012pppph = deriv_step_calc_sq_sl12(4 * hm012)
                bar()
                # Deviate m0(1,2) by small amount LEFT and square soft scalar masses for BCs
                mzsq_m012mh = deriv_step_calc_sq_sl12((-1) * hm012)
                bar()
                # Two deviations to left
                mzsq_m012mmh = deriv_step_calc_sq_sl12((-2) * hm012)
                bar()
                # Three deviations to left
                mzsq_m012mmmh = deriv_step_calc_sq_sl12((-3) * hm012)
                bar()
                # Four deviations to left
                mzsq_m012mmmmh = deriv_step_calc_sq_sl12((-4) * hm012)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/7, please wait...'

                print("Error estimate for m0(3) derivative: "
                      + str("{:.4e}".format(float(mp.power(hm03, 8)))))
                ##### Set up solutions for m_0(3) derivative #####
                mzsq_m03ph = deriv_step_calc_sq_sl3(hm03)
                bar()
                # Two deviations to right
                mzsq_m03pph = deriv_step_calc_sq_sl3(2 * hm03)
                bar()
                # Three deviations to right
                mzsq_m03ppph = deriv_step_calc_sq_sl3(3 * hm03)
                bar()
                # Four deviations to right
                mzsq_m03pppph = deriv_step_calc_sq_sl3(4 * hm03)
                bar()
                # Deviate m0(3) by small amount LEFT and square soft scalar masses for BCs
                mzsq_m03mh = deriv_step_calc_sq_sl3((-1) * hm03)
                bar()
                # Two deviations to left
                mzsq_m03mmh = deriv_step_calc_sq_sl3((-2) * hm03)
                bar()
                # Three deviations to left
                mzsq_m03mmmh = deriv_step_calc_sq_sl3((-3) * hm03)
                bar()
                # Four deviations to left
                mzsq_m03mmmmh = deriv_step_calc_sq_sl3((-4) * hm03)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/7, please wait...'

                ##### Set up solutions for mHu and mHd derivatives #####
                for j in np.arange(25, 27):
                    print("Error estimate for " + str(Higgs_labels[j-25]) + " derivative: "
                          + str("{:.4e}".format(float(mp.power(Higgs_hvals[j-25], 8)))))
                    # For loop to compute soft Higgs mass derivatives
                    # Right shifts
                    mzsq_mHph[j-25] = deriv_step_calc_Higgs(j, Higgs_hvals[j-25])
                    bar()
                    mzsq_mHpph[j-25] = deriv_step_calc_Higgs(j,
                                                             (2 * Higgs_hvals[j-25]))
                    bar()
                    mzsq_mHppph[j-25] = deriv_step_calc_Higgs(j,
                                                              (3 * Higgs_hvals[j-25]))
                    bar()
                    mzsq_mHpppph[j-25] = deriv_step_calc_Higgs(j,
                                                               (4 * Higgs_hvals[j-25]))
                    bar()
                    # Left shifts
                    mzsq_mHmh[j-25] = deriv_step_calc_Higgs(j, (-1) * Higgs_hvals[j-25])
                    bar()
                    mzsq_mHmmh[j-25] = deriv_step_calc_Higgs(j,
                                                             (-1) * (2 * Higgs_hvals[j-25]))
                    bar()
                    mzsq_mHmmmh[j-25] = deriv_step_calc_Higgs(j,
                                                              (-1) * (3 * Higgs_hvals[j-25]))
                    bar()
                    mzsq_mHmmmmh[j-25] = deriv_step_calc_Higgs(j,
                                                               (-1) * (4 * Higgs_hvals[j-25]))
                    bar()
                    derivcount += 1
                    bar.text = f'Derivatives: {derivcount}/7, please wait...'

                print("Error estimate for m_1/2 derivative: "
                      + str("{:.4e}".format(float(mp.power(hmhf, 8)))))
                ##### Set up solutions for m_1/2 derivative #####
                mzsq_mhfph = deriv_step_calc_gaugino(hmhf)
                bar()
                # Two deviations to right
                mzsq_mhfpph = deriv_step_calc_gaugino(2 * hmhf)
                bar()
                # Three deviations to right
                mzsq_mhfppph = deriv_step_calc_gaugino(3 * hmhf)
                bar()
                # Four deviations to right
                mzsq_mhfpppph = deriv_step_calc_gaugino(4 * hmhf)
                bar()
                # Boundary conditions first
                mzsq_mhfmh = deriv_step_calc_gaugino((-1) * hmhf)
                bar()
                # Two deviations to left
                mzsq_mhfmmh = deriv_step_calc_gaugino((-2) * hmhf)
                bar()
                # Three deviations to left
                mzsq_mhfmmmh = deriv_step_calc_gaugino((-3) * hmhf)
                bar()
                # Four deviations to left
                mzsq_mhfmmmmh = deriv_step_calc_gaugino((-4) * hmhf)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/7, please wait...'

                print("Error estimate for A0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hA0, 8)))))
                ##### Set up solutions for A_0 derivative #####
                mzsq_A0ph = deriv_step_calc_trilin(hA0)
                bar()
                # Two deviations to right
                mzsq_A0pph = deriv_step_calc_trilin(2 * hA0)
                bar()
                # Three deviations to right
                mzsq_A0ppph = deriv_step_calc_trilin(3 * hA0)
                bar()
                # Four deviations to right
                mzsq_A0pppph = deriv_step_calc_trilin(4 * hA0)
                bar()
                # Boundary conditions first
                mzsq_A0mh = deriv_step_calc_trilin((-1) * hA0)
                bar()
                # Two deviations to left
                mzsq_A0mmh = deriv_step_calc_trilin((-2) * hA0)
                bar()
                # Three deviations to left
                mzsq_A0mmmh = deriv_step_calc_trilin((-3) * hA0)
                bar()
                # Four deviations to left
                mzsq_A0mmmmh = deriv_step_calc_trilin((-4) * hA0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/7, please wait...'

                print("Error estimate for mu0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hmu0, 8)))))
                ##### Set up solutions for mu derivative #####
                mzsq_mu0ph = deriv_step_calc_mu0(hmu0)
                bar()
                # Two deviations to right
                mzsq_mu0pph = deriv_step_calc_mu0(2 * hmu0)
                bar()
                # Three deviations to right
                mzsq_mu0ppph = deriv_step_calc_mu0(3 * hmu0)
                bar()
                # Four deviations to right
                mzsq_mu0pppph = deriv_step_calc_mu0(4 * hmu0)
                bar()
                # Deviate mu_0 by small amount left
                mzsq_mu0mh = deriv_step_calc_mu0((-1) * hmu0)
                bar()
                # Two deviations to left
                mzsq_mu0mmh = deriv_step_calc_mu0((-2) * hmu0)
                bar()
                # Three deviations to left
                mzsq_mu0mmmh = deriv_step_calc_mu0((-3) * hmu0)
                bar()
                # Four deviations to left
                mzsq_mu0mmmmh = deriv_step_calc_mu0((-4) * hmu0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/7, please wait...'

                # Construct derivative array
                deriv_array = np.array([deriv_num_calc(hm012,mzsq_m012mmmmh,mzsq_m012mmmh,
                                                       mzsq_m012mmh,mzsq_m012mh,mzsq_m012ph,
                                                       mzsq_m012pph,mzsq_m012ppph,mzsq_m012pppph),
                                        deriv_num_calc(hm03,mzsq_m03mmmmh,mzsq_m03mmmh,
                                                       mzsq_m03mmh,mzsq_m03mh,mzsq_m03ph,
                                                       mzsq_m03pph,mzsq_m03ppph,mzsq_m03pppph),
                                        deriv_num_calc(hmhf,mzsq_mhfmmmmh,mzsq_mhfmmmh,
                                                       mzsq_mhfmmh,mzsq_mhfmh,mzsq_mhfph,
                                                       mzsq_mhfpph,mzsq_mhfppph,mzsq_mhfpppph),
                                        deriv_num_calc(hA0,mzsq_A0mmmmh,mzsq_A0mmmh,
                                                       mzsq_A0mmh,mzsq_A0mh,mzsq_A0ph,
                                                       mzsq_A0pph,mzsq_A0ppph,mzsq_A0pppph),
                                        deriv_num_calc(hmu0,mzsq_mu0mmmmh,mzsq_mu0mmmh,
                                                       mzsq_mu0mmh,mzsq_mu0mh,mzsq_mu0ph,
                                                       mzsq_mu0pph,mzsq_mu0ppph,mzsq_mu0pppph),
                                        deriv_num_calc(Higgs_hvals[0],mzsq_mHmmmmh[0],mzsq_mHmmmh[0],
                                                       mzsq_mHmmh[0],mzsq_mHmh[0],mzsq_mHph[0],
                                                       mzsq_mHpph[0],mzsq_mHppph[0],mzsq_mHpppph[0]),
                                        deriv_num_calc(Higgs_hvals[1],mzsq_mHmmmmh[1],mzsq_mHmmmh[1],
                                                       mzsq_mHmmh[1],mzsq_mHmh[1],mzsq_mHph[1],
                                                       mzsq_mHpph[1],mzsq_mHppph[1],mzsq_mHpppph[1])])
                sens_params = np.sort(np.array([(mp.fabs((mym012
                                                         / mymzsq)
                                                        * deriv_array[0]),
                                                 'Delta_BG(m_0(1,2))'),
                                                (mp.fabs((mym03
                                                         / mymzsq)
                                                        * deriv_array[1]),
                                                 'Delta_BG(m_0(3))'),
                                                (mp.fabs((mymhf
                                                         / mymzsq)
                                                        * deriv_array[2]),
                                                 'Delta_BG(m_1/2)'),
                                                (mp.fabs((myA0
                                                         / mymzsq)
                                                        * deriv_array[3]),
                                                 'Delta_BG(A_0)'),
                                                (mp.fabs((mymu0
                                                         / mymzsq)
                                                        * deriv_array[4]),
                                                 'Delta_BG(mu)'),
                                                (mp.fabs((mymHu0 / mymzsq)
                                                        * deriv_array[5]),
                                                 'Delta_BG(mHu)'),
                                                (mp.fabs((mymHd0 / mymzsq)
                                                        * deriv_array[6]),
                                                 'Delta_BG(mHd)')],
                                               dtype=[('BGContrib', float),
                                                      ('BGlabel', 'U40')]),
                                      order='BGContrib')
                print("Done!")

        elif (precselno == 2):
            derivcount = 0
            hm012 = mpf(str(abs(mp.power(math.ulp(mym012), 1/5))))
            hm03 = mpf(str(abs(mp.power(math.ulp(mym03), 1/5))))
            hmhf = mpf(str(abs(mp.power(math.ulp(mymhf), 1/5))))
            hA0 = mpf(str(abs(mp.power(math.ulp(myA0), 1/5))))
            hmu0 = mpf(str(abs(mp.power(math.ulp(mymu0), 1/5))))
            hmHu0 = mpf(str(abs(mp.power(math.ulp(mymHu0), 1/5))))
            hmHd0 = mpf(str(abs(mp.power(math.ulp(mymHd0), 1/5))))
            mzsq_mHph = np.zeros(2)
            mzsq_mHpph = np.zeros(2)
            mzsq_mHmh = np.zeros(2)
            mzsq_mHmmh = np.zeros(2)
            Higgs_hvals = [hmHu0, hmHd0]
            Higgs_labels = ['m_Hu0', 'm_Hd0']
            with alive_bar(28, dual_line=True,
                           force_tty=True, title='Computations: ',
                           bar='brackets') as bar:
                bar.text = f'Derivatives: {derivcount}/7, please wait...'
                print("Error estimate for m0(1,2) derivative: "
                      + str("{:.4e}".format(float(mp.power(hm012, 4)))))
                ##### Set up solutions for m_0(1,2) derivative #####
                mzsq_m012ph = deriv_step_calc_sq_sl12(hm012)
                bar()
                # Two deviations to right
                mzsq_m012pph = deriv_step_calc_sq_sl12(2 * hm012)
                bar()
                # Deviate m0(1,2) by small amount LEFT and square soft scalar masses for BCs
                mzsq_m012mh = deriv_step_calc_sq_sl12((-1) * hm012)
                bar()
                # Two deviations to left
                mzsq_m012mmh = deriv_step_calc_sq_sl12((-2) * hm012)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/7, please wait...'

                print("Error estimate for m0(3) derivative: "
                      + str("{:.4e}".format(float(mp.power(hm03, 4)))))
                ##### Set up solutions for m_0(3) derivative #####
                mzsq_m03ph = deriv_step_calc_sq_sl3(hm03)
                bar()
                # Two deviations to right
                mzsq_m03pph = deriv_step_calc_sq_sl3(2 * hm03)
                bar()
                # Deviate m0(3) by small amount LEFT and square soft scalar masses for BCs
                mzsq_m03mh = deriv_step_calc_sq_sl3((-1) * hm03)
                bar()
                # Two deviations to left
                mzsq_m03mmh = deriv_step_calc_sq_sl3((-2) * hm03)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/7, please wait...'

                ##### Set up solutions for mHu and mHd derivatives #####
                for j in np.arange(25, 27):
                    print("Error estimate for " + str(Higgs_labels[j-25]) + " derivative: "
                          + str("{:.4e}".format(float(mp.power(Higgs_hvals[j-25], 4)))))
                    # For loop to compute soft Higgs mass derivatives
                    # Right shifts
                    mzsq_mHph[j-25] = deriv_step_calc_Higgs(j, Higgs_hvals[j-25])
                    bar()
                    mzsq_mHpph[j-25] = deriv_step_calc_Higgs(j,
                                                             (2 * Higgs_hvals[j-25]))
                    bar()
                    # Left shifts
                    mzsq_mHmh[j-25] = deriv_step_calc_Higgs(j, (-1) * Higgs_hvals[j-25])
                    bar()
                    mzsq_mHmmh[j-25] = deriv_step_calc_Higgs(j,
                                                             (-1) * (2 * Higgs_hvals[j-25]))
                    bar()
                    derivcount += 1
                    bar.text = f'Derivatives: {derivcount}/7, please wait...'

                print("Error estimate for m_1/2 derivative: "
                      + str("{:.4e}".format(float(mp.power(hmhf, 4)))))
                ##### Set up solutions for m_1/2 derivative #####
                mzsq_mhfph = deriv_step_calc_gaugino(hmhf)
                bar()
                # Two deviations to right
                mzsq_mhfpph = deriv_step_calc_gaugino(2 * hmhf)
                bar()
                # Boundary conditions first
                mzsq_mhfmh = deriv_step_calc_gaugino((-1) * hmhf)
                bar()
                # Two deviations to left
                mzsq_mhfmmh = deriv_step_calc_gaugino((-2) * hmhf)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/7, please wait...'

                print("Error estimate for A0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hA0, 4)))))
                ##### Set up solutions for A_0 derivative #####
                mzsq_A0ph = deriv_step_calc_trilin(hA0)
                bar()
                # Two deviations to right
                mzsq_A0pph = deriv_step_calc_trilin(2 * hA0)
                bar()
                # Boundary conditions first
                mzsq_A0mh = deriv_step_calc_trilin((-1) * hA0)
                bar()
                # Two deviations to left
                mzsq_A0mmh = deriv_step_calc_trilin((-2) * hA0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/7, please wait...'

                print("Error estimate for mu0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hmu0, 4)))))
                ##### Set up solutions for mu derivative #####
                mzsq_mu0ph = deriv_step_calc_mu0(hmu0)
                bar()
                # Two deviations to right
                mzsq_mu0pph = deriv_step_calc_mu0(2 * hmu0)
                bar()
                # Deviate mu_0 by small amount left
                mzsq_mu0mh = deriv_step_calc_mu0((-1) * hmu0)
                bar()
                # Two deviations to left
                mzsq_mu0mmh = deriv_step_calc_mu0((-2) * hmu0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/7, please wait...'

                # Construct derivative array
                deriv_array = np.array([deriv_num_calc(hm012,
                                                       mzsq_m012mmh,mzsq_m012mh,mzsq_m012ph,
                                                       mzsq_m012pph),
                                        deriv_num_calc(hm03,
                                                       mzsq_m03mmh,mzsq_m03mh,mzsq_m03ph,
                                                       mzsq_m03pph),
                                        deriv_num_calc(hmhf,
                                                       mzsq_mhfmmh,mzsq_mhfmh,mzsq_mhfph,
                                                       mzsq_mhfpph),
                                        deriv_num_calc(hA0,
                                                       mzsq_A0mmh,mzsq_A0mh,mzsq_A0ph,
                                                       mzsq_A0pph),
                                        deriv_num_calc(hmu0,
                                                       mzsq_mu0mmh,mzsq_mu0mh,mzsq_mu0ph,
                                                       mzsq_mu0pph),
                                        deriv_num_calc(Higgs_hvals[0],
                                                       mzsq_mHmmh[0],mzsq_mHmh[0],mzsq_mHph[0],
                                                       mzsq_mHpph[0]),
                                        deriv_num_calc(Higgs_hvals[1],
                                                       mzsq_mHmmh[1],mzsq_mHmh[1],mzsq_mHph[1],
                                                       mzsq_mHpph[1])])
                sens_params = np.sort(np.array([(mp.fabs((mym012
                                                         / mymzsq)
                                                        * deriv_array[0]),
                                                 'Delta_BG(m_0(1,2))'),
                                                (mp.fabs((mym03
                                                         / mymzsq)
                                                        * deriv_array[1]),
                                                 'Delta_BG(m_0(3))'),
                                                (mp.fabs((mymhf
                                                         / mymzsq)
                                                        * deriv_array[2]),
                                                 'Delta_BG(m_1/2)'),
                                                (mp.fabs((myA0
                                                         / mymzsq)
                                                        * deriv_array[3]),
                                                 'Delta_BG(A_0)'),
                                                (mp.fabs((mymu0
                                                         / mymzsq)
                                                        * deriv_array[4]),
                                                 'Delta_BG(mu)'),
                                                (mp.fabs((mymHu0 / mymzsq)
                                                        * deriv_array[5]),
                                                 'Delta_BG(mHu)'),
                                                (mp.fabs((mymHd0 / mymzsq)
                                                        * deriv_array[6]),
                                                 'Delta_BG(mHd)')],
                                               dtype=[('BGContrib', float),
                                                      ('BGlabel', 'U40')]),
                                      order='BGContrib')
                print("Done!")

        elif (precselno == 3):
            derivcount = 0
            hm012 = mpf(str(abs(mp.power(math.ulp(mym012), 1/3))))
            hm03 = mpf(str(abs(mp.power(math.ulp(mym03), 1/3))))
            hmhf = mpf(str(abs(mp.power(math.ulp(mymhf), 1/3))))
            hA0 = mpf(str(abs(mp.power(math.ulp(myA0), 1/3))))
            hmu0 = mpf(str(abs(mp.power(math.ulp(mymu0), 1/3))))
            hmHu0 = mpf(str(abs(mp.power(math.ulp(mymHu0), 1/3))))
            hmHd0 = mpf(str(abs(mp.power(math.ulp(mymHd0), 1/3))))
            mzsq_mHph = np.zeros(2)
            mzsq_mHmh = np.zeros(2)
            Higgs_hvals = [hmHu0, hmHd0]
            Higgs_labels = ['m_Hu0', 'm_Hd0']
            with alive_bar(14, dual_line=True,
                           force_tty=True, title='Computations: ',
                           bar='brackets') as bar:
                bar.text = f'Derivatives: {derivcount}/7, please wait...'
                print("Error estimate for m0(1,2) derivative: "
                      + str("{:.4e}".format(float(mp.power(hm012, 2)))))
                ##### Set up solutions for m_0(1,2) derivative #####
                mzsq_m012ph = deriv_step_calc_sq_sl12(hm012)
                bar()
                # Deviate m0(1,2) by small amount LEFT and square soft scalar masses for BCs
                mzsq_m012mh = deriv_step_calc_sq_sl12((-1) * hm012)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/7, please wait...'

                print("Error estimate for m0(3) derivative: "
                      + str("{:.4e}".format(float(mp.power(hm03, 2)))))
                ##### Set up solutions for m_0(3) derivative #####
                mzsq_m03ph = deriv_step_calc_sq_sl3(hm03)
                bar()
                # Deviate m0(3) by small amount LEFT and square soft scalar masses for BCs
                mzsq_m03mh = deriv_step_calc_sq_sl3((-1) * hm03)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/7, please wait...'

                ##### Set up solutions for mHu and mHd derivatives #####
                for j in np.arange(25, 27):
                    print("Error estimate for " + str(Higgs_labels[j-25]) + " derivative: "
                          + str("{:.4e}".format(float(mp.power(Higgs_hvals[j-25], 2)))))
                    # For loop to compute soft Higgs mass derivatives
                    # Right shifts
                    mzsq_mHph[j-25] = deriv_step_calc_Higgs(j, Higgs_hvals[j-25])
                    bar()
                    # Left shifts
                    mzsq_mHmh[j-25] = deriv_step_calc_Higgs(j, (-1) * Higgs_hvals[j-25])
                    bar()
                    derivcount += 1
                    bar.text = f'Derivatives: {derivcount}/7, please wait...'

                print("Error estimate for m_1/2 derivative: "
                      + str("{:.4e}".format(float(mp.power(hmhf, 2)))))
                ##### Set up solutions for m_1/2 derivative #####
                mzsq_mhfph = deriv_step_calc_gaugino(hmhf)
                bar()
                # Boundary conditions first
                mzsq_mhfmh = deriv_step_calc_gaugino((-1) * hmhf)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/7, please wait...'

                print("Error estimate for A0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hA0, 2)))))
                ##### Set up solutions for A_0 derivative #####
                mzsq_A0ph = deriv_step_calc_trilin(hA0)
                bar()
                # Boundary conditions first
                mzsq_A0mh = deriv_step_calc_trilin((-1) * hA0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/7, please wait...'

                print("Error estimate for mu0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hmu0, 2)))))
                ##### Set up solutions for mu derivative #####
                mzsq_mu0ph = deriv_step_calc_mu0(hmu0)
                bar()
                # Deviate mu_0 by small amount left
                mzsq_mu0mh = deriv_step_calc_mu0((-1) * hmu0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/7, please wait...'

                # Construct derivative array
                deriv_array = np.array([deriv_num_calc(hm012,mzsq_m012mh,mzsq_m012ph),
                                        deriv_num_calc(hm03,mzsq_m03mh,mzsq_m03ph),
                                        deriv_num_calc(hmhf,mzsq_mhfmh,mzsq_mhfph),
                                        deriv_num_calc(hA0,mzsq_A0mh,mzsq_A0ph),
                                        deriv_num_calc(hmu0,mzsq_mu0mh,mzsq_mu0ph),
                                        deriv_num_calc(Higgs_hvals[0],mzsq_mHmh[0],mzsq_mHph[0]),
                                        deriv_num_calc(Higgs_hvals[1],mzsq_mHmh[1],mzsq_mHph[1])])
                sens_params = np.sort(np.array([(mp.fabs((mym012
                                                         / mymzsq)
                                                        * deriv_array[0]),
                                                 'Delta_BG(m_0(1,2))'),
                                                (mp.fabs((mym03
                                                         / mymzsq)
                                                        * deriv_array[1]),
                                                 'Delta_BG(m_0(3))'),
                                                (mp.fabs((mymhf
                                                         / mymzsq)
                                                        * deriv_array[2]),
                                                 'Delta_BG(m_1/2)'),
                                                (mp.fabs((myA0
                                                         / mymzsq)
                                                        * deriv_array[3]),
                                                 'Delta_BG(A_0)'),
                                                (mp.fabs((mymu0
                                                         / mymzsq)
                                                        * deriv_array[4]),
                                                 'Delta_BG(mu)'),
                                                (mp.fabs((mymHu0 / mymzsq)
                                                        * deriv_array[5]),
                                                 'Delta_BG(mHu)'),
                                                (mp.fabs((mymHd0 / mymzsq)
                                                        * deriv_array[6]),
                                                 'Delta_BG(mHd)')],
                                               dtype=[('BGContrib', float),
                                                      ('BGlabel', 'U40')]),
                                      order='BGContrib')
                print("Done!")
    elif (modselno == 5):
        print("Computing sensitivity coefficient derivatives...")
        print("NOTE: this computation can take a while")
        mym01 = mpf(str(inputGUT_BCs[27]))
        mym02 = mpf(str(inputGUT_BCs[28]))
        mym03 = mpf(str(inputGUT_BCs[29]))
        mymhf = mpf(str(inputGUT_BCs[3]))
        myA0 = mpf(str(inputGUT_BCs[16])) / mpf(str(inputGUT_BCs[7]))
        mymu0 = mp.sqrt(abs(inputGUT_BCs[6]))
        mymusq0 = inputGUT_BCs[6]
        mymHu0 = mp.sqrt(abs(inputGUT_BCs[25]))
        mymHd0 = mp.sqrt(abs(inputGUT_BCs[26]))
        def deriv_step_calc_mu0(shift_amt):
            """
            Do the current mass derivative step for higgsino mass mu.

            Parameters
            ----------
            shift_amt : Float.
                How much to shift for current derivative step.

            Returns
            -------
            mytestmzsq : Float.
                Return current shifted value of mZ^2 with the
                    higgsino mass shifted by the specified amount.

            """
            testBCs = deepcopy(inputGUT_BCs[:])
            for i in np.arange(27,42):
                testBCs[i] = mp.power(testBCs[i], 2)
            testBCs[6] = np.sign(mymusq0) * mp.power(mymu0 + shift_amt, 2)
            mytestmzsq = deriv_step_calc(GUT_SCALE,myweakscale, inptanbval, polecalccheck, deepcopy(testBCs))
            del testBCs
            return mytestmzsq

        def deriv_step_calc_Higgs(current_index, shift_amt):
            """
            Do the current mass derivative step for soft Higgs masses.

            Parameters
            ----------
            current_index : Int.
                Index of parameter to perform derivative with.
            shift_amt : Float.
                How much to shift for current derivative step.

            Returns
            -------
            mytestmzsq : Float.
                Return current shifted value of mZ^2 with the corresponding
                    soft Higgs mass shifted by the specified amount.

            """
            testBCs = deepcopy(inputGUT_BCs[:])
            for i in np.arange(27,42):
                testBCs[i] = mp.power(testBCs[i], 2)
            testBCs[current_index] = mp.power(mp.sqrt(str(abs(testBCs[current_index])))
                                              + shift_amt, 2)
            mytestmzsq = deriv_step_calc(GUT_SCALE,myweakscale, inptanbval, polecalccheck, deepcopy(testBCs))
            del testBCs
            return mytestmzsq

        def deriv_step_calc_sq_sl1(shift_amt):
            """
            Do the current mass derivative step for squarks & sleptons of 1st generation.

            Parameters
            ----------
            shift_amt : Float.
                How much to shift for current derivative step.

            Returns
            -------
            mytestmzsq : Float.
                Return current shifted value of mZ^2 with the corresponding
                    1st gen. squark and slepton masses shifted by the specified amount.

            """
            testBCs = deepcopy(inputGUT_BCs[:])
            m01_indices = [27,30,33,36,39]
            for i in m01_indices:
                testBCs[i] = mp.power(testBCs[i] + shift_amt, 2)
            other_indices = [28,29,31,32,34,35,37,38,40,41]
            for i in other_indices:
                testBCs[i] = mp.power(testBCs[i], 2)
            mytestmzsq = deriv_step_calc(GUT_SCALE,myweakscale, inptanbval, polecalccheck, deepcopy(testBCs))
            del testBCs
            return mytestmzsq

        def deriv_step_calc_sq_sl2(shift_amt):
            """
            Do the current mass derivative step for squarks & sleptons of 2nd generation.

            Parameters
            ----------
            shift_amt : Float.
                How much to shift for current derivative step.

            Returns
            -------
            mytestmzsq : Float.
                Return current shifted value of mZ^2 with the corresponding
                    2nd gen. squark and slepton masses shifted by the specified amount.

            """
            testBCs = deepcopy(inputGUT_BCs[:])
            m01_indices = [28,31,34,37,40]
            for i in m01_indices:
                testBCs[i] = mp.power(testBCs[i] + shift_amt, 2)
            other_indices = [27,29,30,32,33,35,36,38,39,41]
            for i in other_indices:
                testBCs[i] = mp.power(testBCs[i], 2)
            mytestmzsq = deriv_step_calc(GUT_SCALE,myweakscale, inptanbval, polecalccheck, deepcopy(testBCs))
            del testBCs
            return mytestmzsq

        def deriv_step_calc_sq_sl3(shift_amt):
            """
            Do the current mass derivative step for squarks & sleptons of 3rd generation.

            Parameters
            ----------
            shift_amt : Float.
                How much to shift for current derivative step.

            Returns
            -------
            mytestmzsq : Float.
                Return current shifted value of mZ^2 with the corresponding
                    3rd gen. squark and slepton masses shifted by the specified amount.

            """
            testBCs = deepcopy(inputGUT_BCs[:])
            m012_indices = [27,28,30,31,33,34,36,37,39,40]
            for i in m012_indices:
                testBCs[i] = mp.power(testBCs[i], 2)
            m03_indices = [29,32,35,38,41]
            for i in m03_indices:
                testBCs[i] = mp.power(testBCs[i] + shift_amt, 2)
            mytestmzsq = deriv_step_calc(GUT_SCALE,myweakscale, inptanbval, polecalccheck, deepcopy(testBCs))
            del testBCs
            return mytestmzsq

        def deriv_step_calc_trilin(shift_amt):
            """
            Do the current mass derivative step for soft trilinear couplings.

            Parameters
            ----------
            shift_amt : Float.
                How much to shift each soft trilinear for current derivative step.

            Returns
            -------
            mytestmzsq : Float.
                Return current shifted value of mZ^2 with the
                    soft trilinear couplings shifted by the specified amount.

            """
            testBCs = deepcopy(inputGUT_BCs[:])
            for i in np.arange(27,42):
                testBCs[i] = mp.power(testBCs[i], 2)
            for i in np.arange(16,25):
                testBCs[i] = (((testBCs[i] / testBCs[i-9]) + shift_amt)
                              * testBCs[i-9])
            mytestmzsq = deriv_step_calc(GUT_SCALE,myweakscale, inptanbval, polecalccheck, deepcopy(testBCs))
            del testBCs
            return mytestmzsq

        def deriv_step_calc_gaugino(shift_amt):
            """
            Do the current mass derivative step for soft gaugino masses.

            Parameters
            ----------
            shift_amt : Float.
                How much to shift for current derivative step.

            Returns
            -------
            mytestmzsq : Float.
                Return current shifted value of mZ^2 with the corresponding
                    soft gaugino mass shifted by the specified amount.

            """
            testBCs = deepcopy(inputGUT_BCs[:])
            for i in np.arange(27,42):
                testBCs[i] = mp.power(testBCs[i], 2)
            for i in np.arange(3,6):
                testBCs[i] = testBCs[i] + shift_amt
            mytestmzsq = deriv_step_calc(GUT_SCALE,myweakscale, inptanbval, polecalccheck, deepcopy(testBCs))
            del testBCs
            return mytestmzsq

        if (precselno == 1):
            derivcount = 0
            hm01 = mpf(str(abs(mp.power(math.ulp(mym01), 1/9))))
            hm02 = mpf(str(abs(mp.power(math.ulp(mym02), 1/9))))
            hm03 = mpf(str(abs(mp.power(math.ulp(mym03), 1/9))))
            hmhf = mpf(str(abs(mp.power(math.ulp(mymhf), 1/9))))
            hA0 = mpf(str(abs(mp.power(math.ulp(myA0), 1/9))))
            hmu0 = mpf(str(abs(mp.power(math.ulp(mymu0), 1/9))))
            hmHu0 = mpf(str(abs(mp.power(math.ulp(mymHu0), 1/9))))
            hmHd0 = mpf(str(abs(mp.power(math.ulp(mymHd0), 1/9))))
            mzsq_mHph = np.zeros(2)
            mzsq_mHpph = np.zeros(2)
            mzsq_mHppph = np.zeros(2)
            mzsq_mHpppph = np.zeros(2)
            mzsq_mHmh = np.zeros(2)
            mzsq_mHmmh = np.zeros(2)
            mzsq_mHmmmh = np.zeros(2)
            mzsq_mHmmmmh = np.zeros(2)
            Higgs_hvals = [hmHu0, hmHd0]
            Higgs_labels = ['m_Hu0', 'm_Hd0']
            with alive_bar(64, dual_line=True,
                           force_tty=True, title='Computations: ',
                           bar='brackets') as bar:
                bar.text = f'Derivatives: {derivcount}/8, please wait...'
                print("Error estimate for m0(1) derivative: "
                      + str("{:.4e}".format(float(mp.power(hm01, 8)))))
                ##### Set up solutions for m_0(1) derivative #####
                mzsq_m01ph = deriv_step_calc_sq_sl1(hm01)
                bar()
                # Two deviations to right
                mzsq_m01pph = deriv_step_calc_sq_sl1(2 * hm01)
                bar()
                # Three deviations to right
                mzsq_m01ppph = deriv_step_calc_sq_sl1(3 * hm01)
                bar()
                # Four deviations to right
                mzsq_m01pppph = deriv_step_calc_sq_sl1(4 * hm01)
                bar()
                # Deviate m0(1) by small amount LEFT and square soft scalar masses for BCs
                mzsq_m01mh = deriv_step_calc_sq_sl1((-1) * hm01)
                bar()
                # Two deviations to left
                mzsq_m01mmh = deriv_step_calc_sq_sl1((-2) * hm01)
                bar()
                # Three deviations to left
                mzsq_m01mmmh = deriv_step_calc_sq_sl1((-3) * hm01)
                bar()
                # Four deviations to left
                mzsq_m01mmmmh = deriv_step_calc_sq_sl1((-4) * hm01)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/7, please wait...'

                print("Error estimate for m0(2) derivative: "
                      + str("{:.4e}".format(float(mp.power(hm02, 8)))))
                ##### Set up solutions for m_0(2) derivative #####
                mzsq_m02ph = deriv_step_calc_sq_sl2(hm02)
                bar()
                # Two deviations to right
                mzsq_m02pph = deriv_step_calc_sq_sl2(2 * hm02)
                bar()
                # Three deviations to right
                mzsq_m02ppph = deriv_step_calc_sq_sl2(3 * hm02)
                bar()
                # Four deviations to right
                mzsq_m02pppph = deriv_step_calc_sq_sl2(4 * hm02)
                bar()
                # Deviate m0(2) by small amount LEFT and square soft scalar masses for BCs
                mzsq_m02mh = deriv_step_calc_sq_sl2((-1) * hm02)
                bar()
                # Two deviations to left
                mzsq_m02mmh = deriv_step_calc_sq_sl2((-2) * hm02)
                bar()
                # Three deviations to left
                mzsq_m02mmmh = deriv_step_calc_sq_sl2((-3) * hm02)
                bar()
                # Four deviations to left
                mzsq_m02mmmmh = deriv_step_calc_sq_sl2((-4) * hm02)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/7, please wait...'

                print("Error estimate for m0(3) derivative: "
                      + str("{:.4e}".format(float(mp.power(hm03, 8)))))
                ##### Set up solutions for m_0(3) derivative #####
                mzsq_m03ph = deriv_step_calc_sq_sl3(hm03)
                bar()
                # Two deviations to right
                mzsq_m03pph = deriv_step_calc_sq_sl3(2 * hm03)
                bar()
                # Three deviations to right
                mzsq_m03ppph = deriv_step_calc_sq_sl3(3 * hm03)
                bar()
                # Four deviations to right
                mzsq_m03pppph = deriv_step_calc_sq_sl3(4 * hm03)
                bar()
                # Deviate m0(3) by small amount LEFT and square soft scalar masses for BCs
                mzsq_m03mh = deriv_step_calc_sq_sl3((-1) * hm03)
                bar()
                # Two deviations to left
                mzsq_m03mmh = deriv_step_calc_sq_sl3((-2) * hm03)
                bar()
                # Three deviations to left
                mzsq_m03mmmh = deriv_step_calc_sq_sl3((-3) * hm03)
                bar()
                # Four deviations to left
                mzsq_m03mmmmh = deriv_step_calc_sq_sl3((-4) * hm03)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/7, please wait...'

                ##### Set up solutions for mHu and mHd derivatives #####
                for j in np.arange(25, 27):
                    print("Error estimate for " + str(Higgs_labels[j-25]) + " derivative: "
                          + str("{:.4e}".format(float(mp.power(Higgs_hvals[j-25], 8)))))
                    # For loop to compute soft Higgs mass derivatives
                    # Right shifts
                    mzsq_mHph[j-25] = deriv_step_calc_Higgs(j, Higgs_hvals[j-25])
                    bar()
                    mzsq_mHpph[j-25] = deriv_step_calc_Higgs(j,
                                                             (2 * Higgs_hvals[j-25]))
                    bar()
                    mzsq_mHppph[j-25] = deriv_step_calc_Higgs(j,
                                                              (3 * Higgs_hvals[j-25]))
                    bar()
                    mzsq_mHpppph[j-25] = deriv_step_calc_Higgs(j,
                                                               (4 * Higgs_hvals[j-25]))
                    bar()
                    # Left shifts
                    mzsq_mHmh[j-25] = deriv_step_calc_Higgs(j, (-1) * Higgs_hvals[j-25])
                    bar()
                    mzsq_mHmmh[j-25] = deriv_step_calc_Higgs(j,
                                                             (-1) * (2 * Higgs_hvals[j-25]))
                    bar()
                    mzsq_mHmmmh[j-25] = deriv_step_calc_Higgs(j,
                                                              (-1) * (3 * Higgs_hvals[j-25]))
                    bar()
                    mzsq_mHmmmmh[j-25] = deriv_step_calc_Higgs(j,
                                                               (-1) * (4 * Higgs_hvals[j-25]))
                    bar()
                    derivcount += 1
                    bar.text = f'Derivatives: {derivcount}/7, please wait...'

                print("Error estimate for m_1/2 derivative: "
                      + str("{:.4e}".format(float(mp.power(hmhf, 8)))))
                ##### Set up solutions for m_1/2 derivative #####
                mzsq_mhfph = deriv_step_calc_gaugino(hmhf)
                bar()
                # Two deviations to right
                mzsq_mhfpph = deriv_step_calc_gaugino(2 * hmhf)
                bar()
                # Three deviations to right
                mzsq_mhfppph = deriv_step_calc_gaugino(3 * hmhf)
                bar()
                # Four deviations to right
                mzsq_mhfpppph = deriv_step_calc_gaugino(4 * hmhf)
                bar()
                # Boundary conditions first
                mzsq_mhfmh = deriv_step_calc_gaugino((-1) * hmhf)
                bar()
                # Two deviations to left
                mzsq_mhfmmh = deriv_step_calc_gaugino((-2) * hmhf)
                bar()
                # Three deviations to left
                mzsq_mhfmmmh = deriv_step_calc_gaugino((-3) * hmhf)
                bar()
                # Four deviations to left
                mzsq_mhfmmmmh = deriv_step_calc_gaugino((-4) * hmhf)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/7, please wait...'

                print("Error estimate for A0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hA0, 8)))))
                ##### Set up solutions for A_0 derivative #####
                mzsq_A0ph = deriv_step_calc_trilin(hA0)
                bar()
                # Two deviations to right
                mzsq_A0pph = deriv_step_calc_trilin(2 * hA0)
                bar()
                # Three deviations to right
                mzsq_A0ppph = deriv_step_calc_trilin(3 * hA0)
                bar()
                # Four deviations to right
                mzsq_A0pppph = deriv_step_calc_trilin(4 * hA0)
                bar()
                # Boundary conditions first
                mzsq_A0mh = deriv_step_calc_trilin((-1) * hA0)
                bar()
                # Two deviations to left
                mzsq_A0mmh = deriv_step_calc_trilin((-2) * hA0)
                bar()
                # Three deviations to left
                mzsq_A0mmmh = deriv_step_calc_trilin((-3) * hA0)
                bar()
                # Four deviations to left
                mzsq_A0mmmmh = deriv_step_calc_trilin((-4) * hA0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/7, please wait...'

                print("Error estimate for mu0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hmu0, 8)))))
                ##### Set up solutions for mu derivative #####
                mzsq_mu0ph = deriv_step_calc_mu0(hmu0)
                bar()
                # Two deviations to right
                mzsq_mu0pph = deriv_step_calc_mu0(2 * hmu0)
                bar()
                # Three deviations to right
                mzsq_mu0ppph = deriv_step_calc_mu0(3 * hmu0)
                bar()
                # Four deviations to right
                mzsq_mu0pppph = deriv_step_calc_mu0(4 * hmu0)
                bar()
                # Deviate mu_0 by small amount left
                mzsq_mu0mh = deriv_step_calc_mu0((-1) * hmu0)
                bar()
                # Two deviations to left
                mzsq_mu0mmh = deriv_step_calc_mu0((-2) * hmu0)
                bar()
                # Three deviations to left
                mzsq_mu0mmmh = deriv_step_calc_mu0((-3) * hmu0)
                bar()
                # Four deviations to left
                mzsq_mu0mmmmh = deriv_step_calc_mu0((-4) * hmu0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/7, please wait...'

                # Construct derivative array
                deriv_array = np.array([deriv_num_calc(hm01,mzsq_m01mmmmh,mzsq_m01mmmh,
                                                       mzsq_m01mmh,mzsq_m01mh,mzsq_m01ph,
                                                       mzsq_m01pph,mzsq_m01ppph,mzsq_m01pppph),
                                        deriv_num_calc(hm02,mzsq_m02mmmmh,mzsq_m02mmmh,
                                                       mzsq_m02mmh,mzsq_m02mh,mzsq_m02ph,
                                                       mzsq_m02pph,mzsq_m02ppph,mzsq_m02pppph),
                                        deriv_num_calc(hm03,mzsq_m03mmmmh,mzsq_m03mmmh,
                                                       mzsq_m03mmh,mzsq_m03mh,mzsq_m03ph,
                                                       mzsq_m03pph,mzsq_m03ppph,mzsq_m03pppph),
                                        deriv_num_calc(hmhf,mzsq_mhfmmmmh,mzsq_mhfmmmh,
                                                       mzsq_mhfmmh,mzsq_mhfmh,mzsq_mhfph,
                                                       mzsq_mhfpph,mzsq_mhfppph,mzsq_mhfpppph),
                                        deriv_num_calc(hA0,mzsq_A0mmmmh,mzsq_A0mmmh,
                                                       mzsq_A0mmh,mzsq_A0mh,mzsq_A0ph,
                                                       mzsq_A0pph,mzsq_A0ppph,mzsq_A0pppph),
                                        deriv_num_calc(hmu0,mzsq_mu0mmmmh,mzsq_mu0mmmh,
                                                       mzsq_mu0mmh,mzsq_mu0mh,mzsq_mu0ph,
                                                       mzsq_mu0pph,mzsq_mu0ppph,mzsq_mu0pppph),
                                        deriv_num_calc(Higgs_hvals[0],mzsq_mHmmmmh[0],mzsq_mHmmmh[0],
                                                       mzsq_mHmmh[0],mzsq_mHmh[0],mzsq_mHph[0],
                                                       mzsq_mHpph[0],mzsq_mHppph[0],mzsq_mHpppph[0]),
                                        deriv_num_calc(Higgs_hvals[1],mzsq_mHmmmmh[1],mzsq_mHmmmh[1],
                                                       mzsq_mHmmh[1],mzsq_mHmh[1],mzsq_mHph[1],
                                                       mzsq_mHpph[1],mzsq_mHppph[1],mzsq_mHpppph[1])])
                sens_params = np.sort(np.array([(mp.fabs((mym01
                                                         / mymzsq)
                                                        * deriv_array[0]),
                                                 'Delta_BG(m_0(1))'),
                                                (mp.fabs((mym02
                                                         / mymzsq)
                                                        * deriv_array[1]),
                                                 'Delta_BG(m_0(2))'),
                                                (mp.fabs((mym03
                                                         / mymzsq)
                                                        * deriv_array[2]),
                                                 'Delta_BG(m_0(3))'),
                                                (mp.fabs((mymhf
                                                         / mymzsq)
                                                        * deriv_array[3]),
                                                 'Delta_BG(m_1/2)'),
                                                (mp.fabs((myA0
                                                         / mymzsq)
                                                        * deriv_array[4]),
                                                 'Delta_BG(A_0)'),
                                                (mp.fabs((mymu0
                                                         / mymzsq)
                                                        * deriv_array[5]),
                                                 'Delta_BG(mu)'),
                                                (mp.fabs((mymHu0 / mymzsq)
                                                        * deriv_array[6]),
                                                 'Delta_BG(mHu)'),
                                                (mp.fabs((mymHd0 / mymzsq)
                                                        * deriv_array[7]),
                                                 'Delta_BG(mHd)')],
                                               dtype=[('BGContrib', float),
                                                      ('BGlabel', 'U40')]),
                                      order='BGContrib')
                print("Done!")

        elif (precselno == 2):
            derivcount = 0
            hm01 = mpf(str(abs(mp.power(math.ulp(mym01), 1/5))))
            hm02 = mpf(str(abs(mp.power(math.ulp(mym02), 1/5))))
            hm03 = mpf(str(abs(mp.power(math.ulp(mym03), 1/5))))
            hmhf = mpf(str(abs(mp.power(math.ulp(mymhf), 1/5))))
            hA0 = mpf(str(abs(mp.power(math.ulp(myA0), 1/5))))
            hmu0 = mpf(str(abs(mp.power(math.ulp(mymu0), 1/5))))
            hmHu0 = mpf(str(abs(mp.power(math.ulp(mymHu0), 1/5))))
            hmHd0 = mpf(str(abs(mp.power(math.ulp(mymHd0), 1/5))))
            mzsq_mHph = np.zeros(2)
            mzsq_mHpph = np.zeros(2)
            mzsq_mHmh = np.zeros(2)
            mzsq_mHmmh = np.zeros(2)
            Higgs_hvals = [hmHu0, hmHd0]
            Higgs_labels = ['m_Hu0', 'm_Hd0']
            with alive_bar(32, dual_line=True,
                           force_tty=True, title='Computations: ',
                           bar='brackets') as bar:
                bar.text = f'Derivatives: {derivcount}/8, please wait...'
                print("Error estimate for m0(1) derivative: "
                      + str("{:.4e}".format(float(mp.power(hm01, 4)))))
                ##### Set up solutions for m_0(1) derivative #####
                mzsq_m01ph = deriv_step_calc_sq_sl1(hm01)
                bar()
                # Two deviations to right
                mzsq_m01pph = deriv_step_calc_sq_sl1(2 * hm01)
                bar()
                # Deviate m0(1) by small amount LEFT and square soft scalar masses for BCs
                mzsq_m01mh = deriv_step_calc_sq_sl1((-1) * hm01)
                bar()
                # Two deviations to left
                mzsq_m01mmh = deriv_step_calc_sq_sl1((-2) * hm01)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/7, please wait...'

                print("Error estimate for m0(2) derivative: "
                      + str("{:.4e}".format(float(mp.power(hm02, 4)))))
                ##### Set up solutions for m_0(2) derivative #####
                mzsq_m02ph = deriv_step_calc_sq_sl2(hm02)
                bar()
                # Two deviations to right
                mzsq_m02pph = deriv_step_calc_sq_sl2(2 * hm02)
                bar()
                # Deviate m0(2) by small amount LEFT and square soft scalar masses for BCs
                mzsq_m02mh = deriv_step_calc_sq_sl2((-1) * hm02)
                bar()
                # Two deviations to left
                mzsq_m02mmh = deriv_step_calc_sq_sl2((-2) * hm02)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/7, please wait...'

                print("Error estimate for m0(3) derivative: "
                      + str("{:.4e}".format(float(mp.power(hm03, 4)))))
                ##### Set up solutions for m_0(3) derivative #####
                mzsq_m03ph = deriv_step_calc_sq_sl3(hm03)
                bar()
                # Two deviations to right
                mzsq_m03pph = deriv_step_calc_sq_sl3(2 * hm03)
                bar()
                # Deviate m0(3) by small amount LEFT and square soft scalar masses for BCs
                mzsq_m03mh = deriv_step_calc_sq_sl3((-1) * hm03)
                bar()
                # Two deviations to left
                mzsq_m03mmh = deriv_step_calc_sq_sl3((-2) * hm03)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/7, please wait...'

                ##### Set up solutions for mHu and mHd derivatives #####
                for j in np.arange(25, 27):
                    print("Error estimate for " + str(Higgs_labels[j-25]) + " derivative: "
                          + str("{:.4e}".format(float(mp.power(Higgs_hvals[j-25], 4)))))
                    # For loop to compute soft Higgs mass derivatives
                    # Right shifts
                    mzsq_mHph[j-25] = deriv_step_calc_Higgs(j, Higgs_hvals[j-25])
                    bar()
                    mzsq_mHpph[j-25] = deriv_step_calc_Higgs(j,
                                                             (2 * Higgs_hvals[j-25]))
                    bar()
                    # Left shifts
                    mzsq_mHmh[j-25] = deriv_step_calc_Higgs(j, (-1) * Higgs_hvals[j-25])
                    bar()
                    mzsq_mHmmh[j-25] = deriv_step_calc_Higgs(j,
                                                             (-1) * (2 * Higgs_hvals[j-25]))
                    bar()
                    derivcount += 1
                    bar.text = f'Derivatives: {derivcount}/7, please wait...'

                print("Error estimate for m_1/2 derivative: "
                      + str("{:.4e}".format(float(mp.power(hmhf, 4)))))
                ##### Set up solutions for m_1/2 derivative #####
                mzsq_mhfph = deriv_step_calc_gaugino(hmhf)
                bar()
                # Two deviations to right
                mzsq_mhfpph = deriv_step_calc_gaugino(2 * hmhf)
                bar()
                # Boundary conditions first
                mzsq_mhfmh = deriv_step_calc_gaugino((-1) * hmhf)
                bar()
                # Two deviations to left
                mzsq_mhfmmh = deriv_step_calc_gaugino((-2) * hmhf)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/7, please wait...'

                print("Error estimate for A0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hA0, 4)))))
                ##### Set up solutions for A_0 derivative #####
                mzsq_A0ph = deriv_step_calc_trilin(hA0)
                bar()
                # Two deviations to right
                mzsq_A0pph = deriv_step_calc_trilin(2 * hA0)
                bar()
                # Boundary conditions first
                mzsq_A0mh = deriv_step_calc_trilin((-1) * hA0)
                bar()
                # Two deviations to left
                mzsq_A0mmh = deriv_step_calc_trilin((-2) * hA0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/7, please wait...'

                print("Error estimate for mu0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hmu0, 4)))))
                ##### Set up solutions for mu derivative #####
                mzsq_mu0ph = deriv_step_calc_mu0(hmu0)
                bar()
                # Two deviations to right
                mzsq_mu0pph = deriv_step_calc_mu0(2 * hmu0)
                bar()
                # Deviate mu_0 by small amount left
                mzsq_mu0mh = deriv_step_calc_mu0((-1) * hmu0)
                bar()
                # Two deviations to left
                mzsq_mu0mmh = deriv_step_calc_mu0((-2) * hmu0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/7, please wait...'

                # Construct derivative array
                deriv_array = np.array([deriv_num_calc(hm01,
                                                       mzsq_m01mmh,mzsq_m01mh,mzsq_m01ph,
                                                       mzsq_m01pph),
                                        deriv_num_calc(hm02,
                                                       mzsq_m02mmh,mzsq_m02mh,mzsq_m02ph,
                                                       mzsq_m02pph),
                                        deriv_num_calc(hm03,
                                                       mzsq_m03mmh,mzsq_m03mh,mzsq_m03ph,
                                                       mzsq_m03pph),
                                        deriv_num_calc(hmhf,
                                                       mzsq_mhfmmh,mzsq_mhfmh,mzsq_mhfph,
                                                       mzsq_mhfpph),
                                        deriv_num_calc(hA0,
                                                       mzsq_A0mmh,mzsq_A0mh,mzsq_A0ph,
                                                       mzsq_A0pph),
                                        deriv_num_calc(hmu0,
                                                       mzsq_mu0mmh,mzsq_mu0mh,mzsq_mu0ph,
                                                       mzsq_mu0pph),
                                        deriv_num_calc(Higgs_hvals[0],
                                                       mzsq_mHmmh[0],mzsq_mHmh[0],mzsq_mHph[0],
                                                       mzsq_mHpph[0]),
                                        deriv_num_calc(Higgs_hvals[1],
                                                       mzsq_mHmmh[1],mzsq_mHmh[1],mzsq_mHph[1],
                                                       mzsq_mHpph[1])])
                sens_params = np.sort(np.array([(mp.fabs((mym01
                                                         / mymzsq)
                                                        * deriv_array[0]),
                                                 'Delta_BG(m_0(1))'),
                                                (mp.fabs((mym02
                                                         / mymzsq)
                                                        * deriv_array[1]),
                                                 'Delta_BG(m_0(2))'),
                                                (mp.fabs((mym03
                                                         / mymzsq)
                                                        * deriv_array[2]),
                                                 'Delta_BG(m_0(3))'),
                                                (mp.fabs((mymhf
                                                         / mymzsq)
                                                        * deriv_array[3]),
                                                 'Delta_BG(m_1/2)'),
                                                (mp.fabs((myA0
                                                         / mymzsq)
                                                        * deriv_array[4]),
                                                 'Delta_BG(A_0)'),
                                                (mp.fabs((mymu0
                                                         / mymzsq)
                                                        * deriv_array[5]),
                                                 'Delta_BG(mu)'),
                                                (mp.fabs((mymHu0 / mymzsq)
                                                        * deriv_array[6]),
                                                 'Delta_BG(mHu)'),
                                                (mp.fabs((mymHd0 / mymzsq)
                                                        * deriv_array[7]),
                                                 'Delta_BG(mHd)')],
                                               dtype=[('BGContrib', float),
                                                      ('BGlabel', 'U40')]),
                                      order='BGContrib')
                print("Done!")

        elif (precselno == 3):
            derivcount = 0
            hm01 = mpf(str(abs(mp.power(math.ulp(mym01), 1/3))))
            hm02 = mpf(str(abs(mp.power(math.ulp(mym02), 1/3))))
            hm03 = mpf(str(abs(mp.power(math.ulp(mym03), 1/3))))
            hmhf = mpf(str(abs(mp.power(math.ulp(mymhf), 1/3))))
            hA0 = mpf(str(abs(mp.power(math.ulp(myA0), 1/3))))
            hmu0 = mpf(str(abs(mp.power(math.ulp(mymu0), 1/3))))
            hmHu0 = mpf(str(abs(mp.power(math.ulp(mymHu0), 1/3))))
            hmHd0 = mpf(str(abs(mp.power(math.ulp(mymHd0), 1/3))))
            mzsq_mHph = np.zeros(2)
            mzsq_mHmh = np.zeros(2)
            Higgs_hvals = [hmHu0, hmHd0]
            Higgs_labels = ['m_Hu0', 'm_Hd0']
            with alive_bar(16, dual_line=True,
                           force_tty=True, title='Computations: ',
                           bar='brackets') as bar:
                bar.text = f'Derivatives: {derivcount}/8, please wait...'
                print("Error estimate for m0(1) derivative: "
                      + str("{:.4e}".format(float(mp.power(hm01, 2)))))
                ##### Set up solutions for m_0(1) derivative #####
                mzsq_m01ph = deriv_step_calc_sq_sl1(hm01)
                bar()
                # Deviate m0(1) by small amount LEFT and square soft scalar masses for BCs
                mzsq_m01mh = deriv_step_calc_sq_sl1((-1) * hm01)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/7, please wait...'

                print("Error estimate for m0(2) derivative: "
                      + str("{:.4e}".format(float(mp.power(hm02, 2)))))
                ##### Set up solutions for m_0(2) derivative #####
                mzsq_m02ph = deriv_step_calc_sq_sl2(hm02)
                bar()
                # Deviate m0(2) by small amount LEFT and square soft scalar masses for BCs
                mzsq_m02mh = deriv_step_calc_sq_sl2((-1) * hm02)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/7, please wait...'

                print("Error estimate for m0(3) derivative: "
                      + str("{:.4e}".format(float(mp.power(hm03, 2)))))
                ##### Set up solutions for m_0(3) derivative #####
                mzsq_m03ph = deriv_step_calc_sq_sl3(hm03)
                bar()
                # Deviate m0(3) by small amount LEFT and square soft scalar masses for BCs
                mzsq_m03mh = deriv_step_calc_sq_sl3((-1) * hm03)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/7, please wait...'

                ##### Set up solutions for mHu and mHd derivatives #####
                for j in np.arange(25, 27):
                    print("Error estimate for " + str(Higgs_labels[j-25]) + " derivative: "
                          + str("{:.4e}".format(float(mp.power(Higgs_hvals[j-25], 2)))))
                    # For loop to compute soft Higgs mass derivatives
                    # Right shifts
                    mzsq_mHph[j-25] = deriv_step_calc_Higgs(j, Higgs_hvals[j-25])
                    bar()
                    # Left shifts
                    mzsq_mHmh[j-25] = deriv_step_calc_Higgs(j, (-1) * Higgs_hvals[j-25])
                    bar()
                    derivcount += 1
                    bar.text = f'Derivatives: {derivcount}/7, please wait...'

                print("Error estimate for m_1/2 derivative: "
                      + str("{:.4e}".format(float(mp.power(hmhf, 2)))))
                ##### Set up solutions for m_1/2 derivative #####
                mzsq_mhfph = deriv_step_calc_gaugino(hmhf)
                bar()
                # Boundary conditions first
                mzsq_mhfmh = deriv_step_calc_gaugino((-1) * hmhf)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/7, please wait...'

                print("Error estimate for A0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hA0, 2)))))
                ##### Set up solutions for A_0 derivative #####
                mzsq_A0ph = deriv_step_calc_trilin(hA0)
                bar()
                # Boundary conditions first
                mzsq_A0mh = deriv_step_calc_trilin((-1) * hA0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/7, please wait...'

                print("Error estimate for mu0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hmu0, 2)))))
                ##### Set up solutions for mu derivative #####
                mzsq_mu0ph = deriv_step_calc_mu0(hmu0)
                bar()
                # Deviate mu_0 by small amount left
                mzsq_mu0mh = deriv_step_calc_mu0((-1) * hmu0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/7, please wait...'

                # Construct derivative array
                deriv_array = np.array([deriv_num_calc(hm01,mzsq_m01mh,mzsq_m01ph),
                                        deriv_num_calc(hm02,mzsq_m02mh,mzsq_m02ph),
                                        deriv_num_calc(hm03,mzsq_m03mh,mzsq_m03ph),
                                        deriv_num_calc(hmhf,mzsq_mhfmh,mzsq_mhfph),
                                        deriv_num_calc(hA0,mzsq_A0mh,mzsq_A0ph),
                                        deriv_num_calc(hmu0,mzsq_mu0mh,mzsq_mu0ph),
                                        deriv_num_calc(Higgs_hvals[0],mzsq_mHmh[0],mzsq_mHph[0]),
                                        deriv_num_calc(Higgs_hvals[1],mzsq_mHmh[1],mzsq_mHph[1])])
                sens_params = np.sort(np.array([(mp.fabs((mym01
                                                         / mymzsq)
                                                        * deriv_array[0]),
                                                 'Delta_BG(m_0(1))'),
                                                (mp.fabs((mym02
                                                         / mymzsq)
                                                        * deriv_array[1]),
                                                 'Delta_BG(m_0(2))'),
                                                (mp.fabs((mym03
                                                         / mymzsq)
                                                        * deriv_array[2]),
                                                 'Delta_BG(m_0(3))'),
                                                (mp.fabs((mymhf
                                                         / mymzsq)
                                                        * deriv_array[3]),
                                                 'Delta_BG(m_1/2)'),
                                                (mp.fabs((myA0
                                                         / mymzsq)
                                                        * deriv_array[4]),
                                                 'Delta_BG(A_0)'),
                                                (mp.fabs((mymu0
                                                         / mymzsq)
                                                        * deriv_array[5]),
                                                 'Delta_BG(mu)'),
                                                (mp.fabs((mymHu0 / mymzsq)
                                                        * deriv_array[6]),
                                                 'Delta_BG(mHu)'),
                                                (mp.fabs((mymHd0 / mymzsq)
                                                        * deriv_array[7]),
                                                 'Delta_BG(mHd)')],
                                               dtype=[('BGContrib', float),
                                                      ('BGlabel', 'U40')]),
                                      order='BGContrib')
                print("Done!")

    elif (modselno == 6):
        print("Computing sensitivity coefficient derivatives...")
        print("NOTE: this computation can take a while")
        mymqL12 = np.amax([inputGUT_BCs[27],inputGUT_BCs[28]])
        mymqL3 = inputGUT_BCs[29]
        mymtR = inputGUT_BCs[35]
        mymuR12 = np.amax([inputGUT_BCs[34],inputGUT_BCs[33]])
        mymbR = inputGUT_BCs[38]
        mymdR12 = np.amax([inputGUT_BCs[36],inputGUT_BCs[37]])
        mymtauL = inputGUT_BCs[32]
        mymeL12 = np.amax([inputGUT_BCs[30],inputGUT_BCs[31]])
        mymtauR = inputGUT_BCs[41]
        mymeR12 = np.amax([inputGUT_BCs[39],inputGUT_BCs[40]])
        myM1 = inputGUT_BCs[3]
        myM2 = inputGUT_BCs[4]
        myM3 = inputGUT_BCs[5]
        myAt = inputGUT_BCs[16]
        myAb = inputGUT_BCs[19]
        myAtau = inputGUT_BCs[22]
        mymusq0 = inputGUT_BCs[6]
        mymu0 = mp.sqrt(str(abs(mymusq0)))
        mymHusqGUT = inputGUT_BCs[25]
        mymHu0 = mp.sqrt(str(abs(mymHusqGUT)))
        mymHdsqGUT = inputGUT_BCs[26]
        mymHd0 = mp.sqrt(str(abs(mymHdsqGUT)))
        def deriv_step_calc_mu0(shift_amt):
            """
            Do the current mass derivative step for higgsino mass mu.

            Parameters
            ----------
            shift_amt : Float.
                How much to shift for current derivative step.

            Returns
            -------
            mytestmzsq : Float.
                Return current shifted value of mZ^2 with the
                    higgsino mass shifted by the specified amount.

            """
            testBCs = deepcopy(inputGUT_BCs[:])
            for i in np.arange(27,42):
                testBCs[i] = mp.power(testBCs[i], 2)
            testBCs[6] = np.sign(mymusq0) * mp.power(mymu0 + shift_amt, 2)
            mytestmzsq = deriv_step_calc(GUT_SCALE,myweakscale, inptanbval, polecalccheck, deepcopy(testBCs))
            del testBCs
            return mytestmzsq

        def deriv_step_calc_Higgs(current_index, shift_amt):
            """
            Do the current mass derivative step for soft Higgs masses.

            Parameters
            ----------
            current_index : Int.
                Index of parameter to perform derivative with.
            shift_amt : Float.
                How much to shift for current derivative step.

            Returns
            -------
            mytestmzsq : Float.
                Return current shifted value of mZ^2 with the corresponding
                    soft Higgs mass shifted by the specified amount.

            """
            testBCs = deepcopy(inputGUT_BCs[:])
            for i in np.arange(27,42):
                testBCs[i] = mp.power(testBCs[i], 2)
            testBCs[current_index] = mp.power(mp.sqrt(str(abs(testBCs[current_index])))
                                              + shift_amt, 2)
            mytestmzsq = deriv_step_calc(GUT_SCALE,myweakscale, inptanbval, polecalccheck, deepcopy(testBCs))
            del testBCs
            return mytestmzsq

        def deriv_step_calc_sq_sl12(current_index, shift_amt):
            """
            Do the current mass derivative step for selected squarks or sleptons in first 2 gens.

            Parameters
            ----------
            current_index: Int.
                Indices of scalar value being shifted in current derivative step.
            shift_amt : Float.
                How much to shift for current derivative step.

            Returns
            -------
            mytestmzsq : Float.
                Return current shifted value of mZ^2 with the corresponding
                    selected 1st & 2nd gen. squark or slepton masses shifted by the specified amount.

            """
            testBCs = deepcopy(inputGUT_BCs[:])
            for i in [current_index,current_index+1]:
                testBCs[i] = mp.power(testBCs[i] + shift_amt, 2)
            remaining_indices = np.delete(np.arange(27,42), np.where([i in [current_index,current_index+1] for i in np.arange(27,42)]))
            for i in remaining_indices:
                testBCs[i] = mp.power(testBCs[i], 2)
            mytestmzsq = deriv_step_calc(GUT_SCALE,myweakscale, inptanbval, polecalccheck, deepcopy(testBCs))
            del testBCs
            return mytestmzsq

        def deriv_step_calc_sq_sl3(current_index, shift_amt):
            """
            Do the current mass derivative step for selected squark or slepton in 3rd gen.

            Parameters
            ----------
            current_index: Int.
                Indices of scalar value being shifted in current derivative step.
            shift_amt : Float.
                How much to shift for current derivative step.

            Returns
            -------
            mytestmzsq : Float.
                Return current shifted value of mZ^2 with the corresponding
                    selected 3rd gen. squark or slepton mass shifted by the specified amount.

            """
            testBCs = deepcopy(inputGUT_BCs[:])
            testBCs[current_index] = mp.power(testBCs[current_index] + shift_amt, 2)
            remaining_indices = np.delete(np.arange(27,42), np.where(np.arange(27, 42)==current_index))
            for i in remaining_indices:
                testBCs[i] = mp.power(testBCs[i], 2)
            mytestmzsq = deriv_step_calc(GUT_SCALE,myweakscale, inptanbval, polecalccheck, deepcopy(testBCs))
            del testBCs
            return mytestmzsq

        def deriv_step_calc_trilin(current_index, shift_amt):
            """
            Do the current mass derivative step for soft trilinear couplings.

            Parameters
            ----------
            current_index : Int.
                Index of parameter to perform derivative with.
            shift_amt : Float.
                How much to shift for current derivative step.

            Returns
            -------
            mytestmzsq : Float.
                Return current shifted value of mZ^2 with the corresponding
                    soft trilinear coupling shifted by the specified amount.

            """
            testBCs = deepcopy(inputGUT_BCs[:])
            for i in np.arange(27,42):
                testBCs[i] = mp.power(testBCs[i], 2)
            testBCs[current_index] = testBCs[current_index] + shift_amt
            mytestmzsq = deriv_step_calc(GUT_SCALE,myweakscale, inptanbval, polecalccheck, deepcopy(testBCs))
            del testBCs
            return mytestmzsq

        def deriv_step_calc_gaugino(current_index, shift_amt):
            """
            Do the current mass derivative step for soft gaugino masses.

            Parameters
            ----------
            current_index : Int.
                Index of parameter to perform derivative with.
            shift_amt : Float.
                How much to shift for current derivative to step.

            Returns
            -------
            mytestmzsq : Float.
                Return current shifted value of mZ^2 with the corresponding
                    soft gaugino mass shifted by the specified amount.

            """
            testBCs = deepcopy(inputGUT_BCs[:])
            for i in np.arange(27,42):
                testBCs[i] = mp.power(testBCs[i], 2)
            testBCs[current_index] = testBCs[current_index] + shift_amt
            mytestmzsq = deriv_step_calc(GUT_SCALE,myweakscale, inptanbval, polecalccheck, deepcopy(testBCs))
            del testBCs
            return mytestmzsq

        if (precselno == 1):
            derivcount = 0
            mzsq_Aiph = np.zeros(3)
            mzsq_Aipph = np.zeros(3)
            mzsq_Aippph = np.zeros(3)
            mzsq_Aipppph = np.zeros(3)
            mzsq_Aimh = np.zeros(3)
            mzsq_Aimmh = np.zeros(3)
            mzsq_Aimmmh = np.zeros(3)
            mzsq_Aimmmmh = np.zeros(3)
            mzsq_Miph = np.zeros(3)
            mzsq_Mipph = np.zeros(3)
            mzsq_Mippph = np.zeros(3)
            mzsq_Mipppph = np.zeros(3)
            mzsq_Mimh = np.zeros(3)
            mzsq_Mimmh = np.zeros(3)
            mzsq_Mimmmh = np.zeros(3)
            mzsq_Mimmmmh = np.zeros(3)
            mzsq_mHph = np.zeros(2)
            mzsq_mHpph = np.zeros(2)
            mzsq_mHppph = np.zeros(2)
            mzsq_mHpppph = np.zeros(2)
            mzsq_mHmh = np.zeros(2)
            mzsq_mHmmh = np.zeros(2)
            mzsq_mHmmmh = np.zeros(2)
            mzsq_mHmmmmh = np.zeros(2)
            # mzsq_msqslpmh12 order: [m_qL(1,2), m_eL(1,2), m_uR(1,2),
            #                         m_dR(1,2), m_eR(1,2)]
            # mzsq_msqslpmh3 order: [m_qL(3), m_tauL, m_tR, m_bR, m_tauR]
            mzsq_msqsl12ph = np.zeros(5)
            mzsq_msqsl12pph = np.zeros(5)
            mzsq_msqsl12ppph = np.zeros(5)
            mzsq_msqsl12pppph = np.zeros(5)
            mzsq_msqsl12mh = np.zeros(5)
            mzsq_msqsl12mmh = np.zeros(5)
            mzsq_msqsl12mmmh = np.zeros(5)
            mzsq_msqsl12mmmmh = np.zeros(5)
            mzsq_msqsl3ph = np.zeros(5)
            mzsq_msqsl3pph = np.zeros(5)
            mzsq_msqsl3ppph = np.zeros(5)
            mzsq_msqsl3pppph = np.zeros(5)
            mzsq_msqsl3mh = np.zeros(5)
            mzsq_msqsl3mmh = np.zeros(5)
            mzsq_msqsl3mmmh = np.zeros(5)
            mzsq_msqsl3mmmmh = np.zeros(5)
            hmqL12 = mpf(str(abs(mp.power(math.ulp(mymqL12), 1/9))))
            hmqL3 = mpf(str(abs(mp.power(math.ulp(mymqL3), 1/9))))
            hmuR12 = mpf(str(abs(mp.power(math.ulp(mymuR12), 1/9))))
            hmtR = mpf(str(abs(mp.power(math.ulp(mymtR), 1/9))))
            hmdR12 = mpf(str(abs(mp.power(math.ulp(mymdR12), 1/9))))
            hmbR = mpf(str(abs(mp.power(math.ulp(mymbR), 1/9))))
            hmeL12 = mpf(str(abs(mp.power(math.ulp(mymeL12), 1/9))))
            hmtauL = mpf(str(abs(mp.power(math.ulp(mymtauL), 1/9))))
            hmeR12 = mpf(str(abs(mp.power(math.ulp(mymeR12), 1/9))))
            hmtauR = mpf(str(abs(mp.power(math.ulp(mymtauR), 1/9))))
            sq_sl12_hvals = [hmqL12, hmeL12, hmuR12, hmdR12, hmeR12]
            sq_sl12_labels = ['m_qL(1,2)','m_eL(1,2)','m_uR(1,2)','m_dR(1,2)',
                              'm_eR(1,2)']
            sq_sl3_hvals = [hmqL3, hmtauL, hmtR, hmbR, hmtauR]
            sq_sl3_labels = ['m_qL(3)','m_tauL','m_tR','m_bR',
                              'm_tauR']
            hM1 = mpf(str(abs(mp.power(math.ulp(myM1), 1/9))))
            hM2 = mpf(str(abs(mp.power(math.ulp(myM2), 1/9))))
            hM3 = mpf(str(abs(mp.power(math.ulp(myM3), 1/9))))
            Mi_hvals = [hM1, hM2, hM3]
            Mi_labels = ['M_1', 'M_2', 'M_3']
            hAt = mpf(str(abs(mp.power(math.ulp(myAt), 1/9))))
            hAb = mpf(str(abs(mp.power(math.ulp(myAb), 1/9))))
            hAtau = mpf(str(abs(mp.power(math.ulp(myAtau), 1/9))))
            Ai_hvals = [hAt, hAb, hAtau]
            Ai_labels = ['a_t','a_b','a_tau']
            hmu0 = mpf(str(abs(mp.power(math.ulp(mymu0), 1/9))))
            hmHu0 = mpf(str(abs(mp.power(math.ulp(mymHu0), 1/9))))
            hmHd0 = mpf(str(abs(mp.power(math.ulp(mymHd0), 1/9))))
            Higgs_hvals = [hmHu0, hmHd0]
            Higgs_labels = ['m_Hu0', 'm_Hd0']

            with alive_bar(152, dual_line=True,
                           force_tty=True, title='Computations: ',
                           bar='brackets') as bar:
                bar.text = f'Derivatives: {derivcount}/19, please wait...'
                for j in [27,30,33,36,39]:
                    print("Error estimate for " + str(sq_sl12_labels[int((j-27) / 3)]) + " derivative: "
                          + str("{:.4e}".format(float(mp.power(sq_sl12_hvals[int((j-27) / 3)], 8)))))
                    # For loop to compute squark and slepton mass derivatives of first two gens
                    # Right shifts
                    mzsq_msqsl12ph[int((j-27) / 3)] = deriv_step_calc_sq_sl12(j, sq_sl12_hvals[int((j-27) / 3)])
                    bar()
                    mzsq_msqsl12pph[int((j-27) / 3)] = deriv_step_calc_sq_sl12(j,
                                                                (2 * sq_sl12_hvals[int((j-27) / 3)]))
                    bar()
                    mzsq_msqsl12ppph[int((j-27) / 3)] = deriv_step_calc_sq_sl12(j,
                                                                 (3 * sq_sl12_hvals[int((j-27) / 3)]))
                    bar()
                    mzsq_msqsl12pppph[int((j-27) / 3)] = deriv_step_calc_sq_sl12(j,
                                                                  (4 * sq_sl12_hvals[int((j-27) / 3)]))
                    bar()
                    # Left shifts
                    mzsq_msqsl12mh[int((j-27) / 3)] = deriv_step_calc_sq_sl12(j, (-1) * sq_sl12_hvals[int((j-27) / 3)])
                    bar()
                    mzsq_msqsl12mmh[int((j-27) / 3)] = deriv_step_calc_sq_sl12(j,
                                                                (-1) * (2 * sq_sl12_hvals[int((j-27) / 3)]))
                    bar()
                    mzsq_msqsl12mmmh[int((j-27) / 3)] = deriv_step_calc_sq_sl12(j,
                                                                 (-1) * (3 * sq_sl12_hvals[int((j-27) / 3)]))
                    bar()
                    mzsq_msqsl12mmmmh[int((j-27) / 3)] = deriv_step_calc_sq_sl12(j,
                                                                  (-1) * (4 * sq_sl12_hvals[int((j-27) / 3)]))
                    bar()
                    derivcount += 1
                    bar.text = f'Derivatives: {derivcount}/19, please wait...'

                for j in [29,32,35,38,41]:
                    print("Error estimate for " + str(sq_sl3_labels[int((j-29) / 3)]) + " derivative: "
                          + str("{:.4e}".format(float(mp.power(sq_sl3_hvals[int((j-29) / 3)], 8)))))
                    # For loop to compute squark and slepton mass derivatives of first two gens
                    # Right shifts
                    mzsq_msqsl3ph[int((j-29) / 3)] = deriv_step_calc_sq_sl3(j, sq_sl3_hvals[int((j-29) / 3)])
                    bar()
                    mzsq_msqsl3pph[int((j-29) / 3)] = deriv_step_calc_sq_sl3(j,
                                                                (2 * sq_sl3_hvals[int((j-29) / 3)]))
                    bar()
                    mzsq_msqsl3ppph[int((j-29) / 3)] = deriv_step_calc_sq_sl3(j,
                                                                 (3 * sq_sl3_hvals[int((j-29) / 3)]))
                    bar()
                    mzsq_msqsl3pppph[int((j-29) / 3)] = deriv_step_calc_sq_sl3(j,
                                                                  (4 * sq_sl3_hvals[int((j-29) / 3)]))
                    bar()
                    # Left shifts
                    mzsq_msqsl3mh[int((j-29) / 3)] = deriv_step_calc_sq_sl3(j, (-1) * sq_sl3_hvals[int((j-29) / 3)])
                    bar()
                    mzsq_msqsl3mmh[int((j-29) / 3)] = deriv_step_calc_sq_sl3(j,
                                                                (-1) * (2 * sq_sl3_hvals[int((j-29) / 3)]))
                    bar()
                    mzsq_msqsl3mmmh[int((j-29) / 3)] = deriv_step_calc_sq_sl3(j,
                                                                 (-1) * (3 * sq_sl3_hvals[int((j-29) / 3)]))
                    bar()
                    mzsq_msqsl3mmmmh[int((j-29) / 3)] = deriv_step_calc_sq_sl3(j,
                                                                  (-1) * (4 * sq_sl3_hvals[int((j-29) / 3)]))
                    bar()
                    derivcount += 1
                    bar.text = f'Derivatives: {derivcount}/19, please wait...'

                for j in [16,19,22]:
                    print("Error estimate for " + str(Ai_labels[int((j-16) / 3)]) + " derivative: "
                          + str("{:.4e}".format(float(mp.power(Ai_hvals[int((j-16) / 3)], 8)))))
                    # For loop to compute soft trilinear coupling derivatives
                    # Right shifts
                    mzsq_Aiph[int((j-16) / 3)] = deriv_step_calc_trilin(j, Ai_hvals[int((j-16) / 3)])
                    bar()
                    mzsq_Aipph[int((j-16) / 3)] = deriv_step_calc_trilin(j,
                                                                     (2 * Ai_hvals[int((j-16) / 3)]))
                    bar()
                    mzsq_Aippph[int((j-16) / 3)] = deriv_step_calc_trilin(j,
                                                                      (3 * Ai_hvals[int((j-16) / 3)]))
                    bar()
                    mzsq_Aipppph[int((j-16) / 3)] = deriv_step_calc_trilin(j,
                                                                       (4 * Ai_hvals[int((j-16) / 3)]))
                    bar()
                    # Left shifts
                    mzsq_Aimh[int((j-16) / 3)] = deriv_step_calc_trilin(j, (-1) * Ai_hvals[int((j-16) / 3)])
                    bar()
                    mzsq_Aimmh[int((j-16) / 3)] = deriv_step_calc_trilin(j,
                                                                     (-1) * (2 * Ai_hvals[int((j-16) / 3)]))
                    bar()
                    mzsq_Aimmmh[int((j-16) / 3)] = deriv_step_calc_trilin(j,
                                                                      (-1) * (3 * Ai_hvals[int((j-16) / 3)]))
                    bar()
                    mzsq_Aimmmmh[int((j-16) / 3)] = deriv_step_calc_trilin(j,
                                                                       (-1) * (4 * Ai_hvals[int((j-16) / 3)]))
                    bar()
                    derivcount += 1
                    bar.text = f'Derivatives: {derivcount}/19, please wait...'

                for j in np.arange(3, 6):
                    print("Error estimate for " + str(Mi_labels[j-3]) + " derivative: "
                          + str("{:.4e}".format(float(mp.power(Mi_hvals[j-3], 8)))))
                    # For loop to compute soft gaugino mass derivatives
                    # Right shifts
                    mzsq_Miph[j-3] = deriv_step_calc_gaugino(j, Mi_hvals[j-3])
                    bar()
                    mzsq_Mipph[j-3] = deriv_step_calc_gaugino(j,
                                                              (2 * Mi_hvals[j-3]))
                    bar()
                    mzsq_Mippph[j-3] = deriv_step_calc_gaugino(j,
                                                               (3 * Mi_hvals[j-3]))
                    bar()
                    mzsq_Mipppph[j-3] = deriv_step_calc_gaugino(j,
                                                                (4 * Mi_hvals[j-3]))
                    bar()
                    # Left shifts
                    mzsq_Mimh[j-3] = deriv_step_calc_gaugino(j, (-1) * Mi_hvals[j-3])
                    bar()
                    mzsq_Mimmh[j-3] = deriv_step_calc_gaugino(j,
                                                              (-1) * (2 * Mi_hvals[j-3]))
                    bar()
                    mzsq_Mimmmh[j-3] = deriv_step_calc_gaugino(j,
                                                               (-1) * (3 * Mi_hvals[j-3]))
                    bar()
                    mzsq_Mimmmmh[j-3] = deriv_step_calc_gaugino(j,
                                                                (-1) * (4 * Mi_hvals[j-3]))
                    bar()
                    derivcount += 1
                    bar.text = f'Derivatives: {derivcount}/19, please wait...'

                for j in np.arange(25, 27):
                    print("Error estimate for " + str(Higgs_labels[j-25]) + " derivative: "
                          + str("{:.4e}".format(float(mp.power(Higgs_hvals[j-25], 8)))))
                    # For loop to compute soft Higgs mass derivatives
                    # Right shifts
                    mzsq_mHph[j-25] = deriv_step_calc_Higgs(j, Higgs_hvals[j-25])
                    bar()
                    mzsq_mHpph[j-25] = deriv_step_calc_Higgs(j,
                                                             (2 * Higgs_hvals[j-25]))
                    bar()
                    mzsq_mHppph[j-25] = deriv_step_calc_Higgs(j,
                                                              (3 * Higgs_hvals[j-25]))
                    bar()
                    mzsq_mHpppph[j-25] = deriv_step_calc_Higgs(j,
                                                               (4 * Higgs_hvals[j-25]))
                    bar()
                    # Left shifts
                    mzsq_mHmh[j-25] = deriv_step_calc_Higgs(j, (-1) * Higgs_hvals[j-25])
                    bar()
                    mzsq_mHmmh[j-25] = deriv_step_calc_Higgs(j,
                                                             (-1) * (2 * Higgs_hvals[j-25]))
                    bar()
                    mzsq_mHmmmh[j-25] = deriv_step_calc_Higgs(j,
                                                              (-1) * (3 * Higgs_hvals[j-25]))
                    bar()
                    mzsq_mHmmmmh[j-25] = deriv_step_calc_Higgs(j,
                                                               (-1) * (4 * Higgs_hvals[j-25]))
                    bar()
                    derivcount += 1
                    bar.text = f'Derivatives: {derivcount}/19, please wait...'

                print("Error estimate for mu_0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hmu0, 8)))))
                mzsq_mu0ph = deriv_step_calc_mu0(hmu0)
                bar()
                mzsq_mu0pph = deriv_step_calc_mu0(2*hmu0)
                bar()
                mzsq_mu0ppph = deriv_step_calc_mu0(3*hmu0)
                bar()
                mzsq_mu0pppph = deriv_step_calc_mu0(4*hmu0)
                bar()
                mzsq_mu0mh = deriv_step_calc_mu0((-1) * hmu0)
                bar()
                mzsq_mu0mmh = deriv_step_calc_mu0((-1) * 2*hmu0)
                bar()
                mzsq_mu0mmmh = deriv_step_calc_mu0((-1) * 3*hmu0)
                bar()
                mzsq_mu0mmmmh = deriv_step_calc_mu0((-1) * 4*hmu0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/19, please wait...'

                # Construct derivative array
                deriv_array = np.array([deriv_num_calc(hmqL12, mzsq_msqsl12mmmmh[0], mzsq_msqsl12mmmh[0],
                                                       mzsq_msqsl12mmh[0], mzsq_msqsl12mh[0], mzsq_msqsl12ph[0],
                                                       mzsq_msqsl12pph[0], mzsq_msqsl12ppph[0], mzsq_msqsl12pppph[0]), #end qL12: 0
                                        deriv_num_calc(hmeL12, mzsq_msqsl12mmmmh[1], mzsq_msqsl12mmmh[1],
                                                       mzsq_msqsl12mmh[1], mzsq_msqsl12mh[1], mzsq_msqsl12ph[1],
                                                       mzsq_msqsl12pph[1], mzsq_msqsl12ppph[1], mzsq_msqsl12pppph[1]), #end eL12: 1
                                        deriv_num_calc(hmuR12, mzsq_msqsl12mmmmh[2], mzsq_msqsl12mmmh[2],
                                                       mzsq_msqsl12mmh[2], mzsq_msqsl12mh[2], mzsq_msqsl12ph[2],
                                                       mzsq_msqsl12pph[2], mzsq_msqsl12ppph[2], mzsq_msqsl12pppph[2]), #end uR12: 2
                                        deriv_num_calc(hmdR12, mzsq_msqsl12mmmmh[3], mzsq_msqsl12mmmh[3],
                                                       mzsq_msqsl12mmh[3], mzsq_msqsl12mh[3], mzsq_msqsl12ph[3],
                                                       mzsq_msqsl12pph[3], mzsq_msqsl12ppph[3], mzsq_msqsl12pppph[3]), #end dR12: 3
                                        deriv_num_calc(hmeR12, mzsq_msqsl12mmmmh[4], mzsq_msqsl12mmmh[4],
                                                       mzsq_msqsl12mmh[4], mzsq_msqsl12mh[4], mzsq_msqsl12ph[4],
                                                       mzsq_msqsl12pph[4], mzsq_msqsl12ppph[4], mzsq_msqsl12pppph[4]), #end eR12: 4
                                        deriv_num_calc(hmqL3, mzsq_msqsl3mmmmh[0], mzsq_msqsl3mmmh[0],
                                                       mzsq_msqsl3mmh[0], mzsq_msqsl3mh[0], mzsq_msqsl3ph[0],
                                                       mzsq_msqsl3pph[0], mzsq_msqsl3ppph[0], mzsq_msqsl3pppph[0]), #end qL3: 5
                                        deriv_num_calc(hmtauL, mzsq_msqsl3mmmmh[1], mzsq_msqsl3mmmh[1],
                                                       mzsq_msqsl3mmh[1], mzsq_msqsl3mh[1], mzsq_msqsl3ph[1],
                                                       mzsq_msqsl3pph[1], mzsq_msqsl3ppph[1], mzsq_msqsl3pppph[1]), #end tauL: 6
                                        deriv_num_calc(hmtR, mzsq_msqsl3mmmmh[2], mzsq_msqsl3mmmh[2],
                                                       mzsq_msqsl3mmh[2], mzsq_msqsl3mh[2], mzsq_msqsl3ph[2],
                                                       mzsq_msqsl3pph[2], mzsq_msqsl3ppph[2], mzsq_msqsl3pppph[2]), #end tR: 7
                                        deriv_num_calc(hmbR, mzsq_msqsl3mmmmh[3], mzsq_msqsl3mmmh[3],
                                                       mzsq_msqsl3mmh[3], mzsq_msqsl3mh[3], mzsq_msqsl3ph[3],
                                                       mzsq_msqsl3pph[3], mzsq_msqsl3ppph[3], mzsq_msqsl3pppph[3]), #end bR: 8
                                        deriv_num_calc(hmtauR, mzsq_msqsl3mmmmh[4], mzsq_msqsl3mmmh[4],
                                                       mzsq_msqsl3mmh[4], mzsq_msqsl3mh[4], mzsq_msqsl3ph[4],
                                                       mzsq_msqsl3pph[4], mzsq_msqsl3ppph[4], mzsq_msqsl3pppph[4]), #end tauR: 9
                                        deriv_num_calc(hAt, mzsq_Aimmmmh[0], mzsq_Aimmmh[0],
                                                       mzsq_Aimmh[0], mzsq_Aimh[0], mzsq_Aiph[0],
                                                       mzsq_Aipph[0], mzsq_Aippph[0], mzsq_Aipppph[0]), #end At: 10
                                        deriv_num_calc(hAb, mzsq_Aimmmmh[1], mzsq_Aimmmh[1],
                                                       mzsq_Aimmh[1], mzsq_Aimh[1], mzsq_Aiph[1],
                                                       mzsq_Aipph[1], mzsq_Aippph[1], mzsq_Aipppph[1]), #end Ab: 11
                                        deriv_num_calc(hAtau, mzsq_Aimmmmh[2], mzsq_Aimmmh[2],
                                                       mzsq_Aimmh[2], mzsq_Aimh[2], mzsq_Aiph[2],
                                                       mzsq_Aipph[2], mzsq_Aippph[2], mzsq_Aipppph[2]), #end Atau: 12
                                        deriv_num_calc(hM1, mzsq_Mimmmmh[0], mzsq_Mimmmh[0],
                                                       mzsq_Mimmh[0], mzsq_Mimh[0], mzsq_Miph[0],
                                                       mzsq_Mipph[0], mzsq_Mippph[0], mzsq_Mipppph[0]), #end M1: 13
                                        deriv_num_calc(hM2, mzsq_Mimmmmh[1], mzsq_Mimmmh[1],
                                                       mzsq_Mimmh[1], mzsq_Mimh[1], mzsq_Miph[1],
                                                       mzsq_Mipph[1], mzsq_Mippph[1], mzsq_Mipppph[1]), #end M2: 14
                                        deriv_num_calc(hM3, mzsq_Mimmmmh[2], mzsq_Mimmmh[2],
                                                       mzsq_Mimmh[2], mzsq_Mimh[2], mzsq_Miph[2],
                                                       mzsq_Mipph[2], mzsq_Mippph[2], mzsq_Mipppph[2]), #end M3: 15
                                        deriv_num_calc(hmHu0, mzsq_mHmmmmh[0], mzsq_mHmmmh[0],
                                                       mzsq_mHmmh[0], mzsq_mHmh[0], mzsq_mHph[0],
                                                       mzsq_mHpph[0], mzsq_mHppph[0], mzsq_mHpppph[0]), #end mHu0: 16
                                        deriv_num_calc(hmHd0, mzsq_mHmmmmh[1], mzsq_mHmmmh[1],
                                                       mzsq_mHmmh[1], mzsq_mHmh[1], mzsq_mHph[1],
                                                       mzsq_mHpph[1], mzsq_mHppph[1], mzsq_mHpppph[1]), #end mHd0: 17
                                        deriv_num_calc(hmu0, mzsq_mu0mmmmh, mzsq_mu0mmmh,
                                                       mzsq_mu0mmh, mzsq_mu0mh, mzsq_mu0ph,
                                                       mzsq_mu0pph, mzsq_mu0ppph, mzsq_mu0pppph)]) #end mu0: 18
                sens_params = np.sort(np.array([(mp.fabs((mymqL12 / mymzsq)
                                                         * deriv_array[0]),
                                                 'Delta_BG(m_qL(1,2))'),
                                                (mp.fabs((mymeL12 / mymzsq)
                                                         * deriv_array[1]),
                                                 'Delta_BG(m_eL(1,2))'),
                                                (mp.fabs((mymuR12 / mymzsq)
                                                         * deriv_array[2]),
                                                 'Delta_BG(m_uR(1,2))'),
                                                (mp.fabs((mymdR12 / mymzsq)
                                                         * deriv_array[3]),
                                                 'Delta_BG(m_dR(1,2))'),
                                                (mp.fabs((mymeR12 / mymzsq)
                                                         * deriv_array[4]),
                                                 'Delta_BG(m_eR(1,2))'),
                                                (mp.fabs((mymqL3 / mymzsq)
                                                         * deriv_array[5]),
                                                 'Delta_BG(m_qL(3))'),
                                                (mp.fabs((mymtauL / mymzsq)
                                                         * deriv_array[6]),
                                                 'Delta_BG(m_tauL)'),
                                                (mp.fabs((mymtR / mymzsq)
                                                         * deriv_array[7]),
                                                 'Delta_BG(m_tR)'),
                                                (mp.fabs((mymbR / mymzsq)
                                                         * deriv_array[8]),
                                                 'Delta_BG(m_bR)'),
                                                (mp.fabs((mymtauR / mymzsq)
                                                         * deriv_array[9]),
                                                 'Delta_BG(m_tauR)'),
                                                (mp.fabs((myAt / mymzsq)
                                                         * deriv_array[10]),
                                                 'Delta_BG(a_t)'),
                                                (mp.fabs((myAb / mymzsq)
                                                         * deriv_array[11]),
                                                 'Delta_BG(a_b)'),
                                                (mp.fabs((myAtau / mymzsq)
                                                         * deriv_array[12]),
                                                 'Delta_BG(a_tau)'),
                                                (mp.fabs((myM1 / mymzsq)
                                                         * deriv_array[13]),
                                                 'Delta_BG(M_1)'),
                                                (mp.fabs((myM2 / mymzsq)
                                                         * deriv_array[14]),
                                                 'Delta_BG(M_2)'),
                                                (mp.fabs((myM3 / mymzsq)
                                                         * deriv_array[15]),
                                                 'Delta_BG(M_3)'),
                                                (mp.fabs((mymHu0 / mymzsq)
                                                         * deriv_array[16]),
                                                 'Delta_BG(m_Hu0)'),
                                                (mp.fabs((mymHd0 / mymzsq)
                                                         * deriv_array[17]),
                                                 'Delta_BG(m_Hd0)'),
                                                (mp.fabs((mymu0 / mymzsq)
                                                         * deriv_array[18]),
                                                 'Delta_BG(mu0)')],
                                               dtype=[('BGcontrib',float),
                                                      ('BGlabel','U40')]),
                                      order='BGcontrib')
                print("Done!")

        elif (precselno == 2):
            derivcount = 0
            mzsq_Aiph = np.zeros(3)
            mzsq_Aipph = np.zeros(3)
            mzsq_Aimh = np.zeros(3)
            mzsq_Aimmh = np.zeros(3)
            mzsq_Miph = np.zeros(3)
            mzsq_Mipph = np.zeros(3)
            mzsq_Mimh = np.zeros(3)
            mzsq_Mimmh = np.zeros(3)
            mzsq_mHph = np.zeros(2)
            mzsq_mHpph = np.zeros(2)
            mzsq_mHmh = np.zeros(2)
            mzsq_mHmmh = np.zeros(2)
            # mzsq_msqslpmh12 order: [m_qL(1,2), m_eL(1,2), m_uR(1,2),
            #                         m_dR(1,2), m_eR(1,2)]
            # mzsq_msqslpmh3 order: [m_qL(3), m_tauL, m_tR, m_bR, m_tauR]
            mzsq_msqsl12ph = np.zeros(5)
            mzsq_msqsl12pph = np.zeros(5)
            mzsq_msqsl12mh = np.zeros(5)
            mzsq_msqsl12mmh = np.zeros(5)
            mzsq_msqsl3ph = np.zeros(5)
            mzsq_msqsl3pph = np.zeros(5)
            mzsq_msqsl3mh = np.zeros(5)
            mzsq_msqsl3mmh = np.zeros(5)
            hmqL12 = mpf(str(abs(mp.power(math.ulp(mymqL12), 1/5))))
            hmqL3 = mpf(str(abs(mp.power(math.ulp(mymqL3), 1/5))))
            hmuR12 = mpf(str(abs(mp.power(math.ulp(mymuR12), 1/5))))
            hmtR = mpf(str(abs(mp.power(math.ulp(mymtR), 1/5))))
            hmdR12 = mpf(str(abs(mp.power(math.ulp(mymdR12), 1/5))))
            hmbR = mpf(str(abs(mp.power(math.ulp(mymbR), 1/5))))
            hmeL12 = mpf(str(abs(mp.power(math.ulp(mymeL12), 1/5))))
            hmtauL = mpf(str(abs(mp.power(math.ulp(mymtauL), 1/5))))
            hmeR12 = mpf(str(abs(mp.power(math.ulp(mymeR12), 1/5))))
            hmtauR = mpf(str(abs(mp.power(math.ulp(mymtauR), 1/5))))
            sq_sl12_hvals = [hmqL12, hmeL12, hmuR12, hmdR12, hmeR12]
            sq_sl12_labels = ['m_qL(1,2)','m_eL(1,2)','m_uR(1,2)','m_dR(1,2)',
                              'm_eR(1,2)']
            sq_sl3_hvals = [hmqL3, hmtauL, hmtR, hmbR, hmtauR]
            sq_sl3_labels = ['m_qL(3)','m_tauL','m_tR','m_bR',
                              'm_tauR']
            hM1 = mpf(str(abs(mp.power(math.ulp(myM1), 1/5))))
            hM2 = mpf(str(abs(mp.power(math.ulp(myM2), 1/5))))
            hM3 = mpf(str(abs(mp.power(math.ulp(myM3), 1/5))))
            Mi_hvals = [hM1, hM2, hM3]
            Mi_labels = ['M_1', 'M_2', 'M_3']
            hAt = mpf(str(abs(mp.power(math.ulp(myAt), 1/5))))
            hAb = mpf(str(abs(mp.power(math.ulp(myAb), 1/5))))
            hAtau = mpf(str(abs(mp.power(math.ulp(myAtau), 1/5))))
            Ai_hvals = [hAt, hAb, hAtau]
            Ai_labels = ['a_t','a_b','a_tau']
            hmu0 = mpf(str(abs(mp.power(math.ulp(mymu0), 1/5))))
            hmHu0 = mpf(str(abs(mp.power(math.ulp(mymHu0), 1/5))))
            hmHd0 = mpf(str(abs(mp.power(math.ulp(mymHd0), 1/5))))
            Higgs_hvals = [hmHu0, hmHd0]
            Higgs_labels = ['m_Hu0', 'm_Hd0']

            with alive_bar(76, dual_line=True,
                           force_tty=True, title='Computations: ',
                           bar='brackets') as bar:
                bar.text = f'Derivatives: {derivcount}/19, please wait...'
                for j in [27,30,33,36,39]:
                    print("Error estimate for " + str(sq_sl12_labels[int((j-27) / 3)]) + " derivative: "
                          + str("{:.4e}".format(float(mp.power(sq_sl12_hvals[int((j-27) / 3)], 4)))))
                    # For loop to compute squark and slepton mass derivatives of first two gens
                    # Right shifts
                    mzsq_msqsl12ph[int((j-27) / 3)] = deriv_step_calc_sq_sl12(j, sq_sl12_hvals[int((j-27) / 3)])
                    bar()
                    mzsq_msqsl12pph[int((j-27) / 3)] = deriv_step_calc_sq_sl12(j,
                                                                (2 * sq_sl12_hvals[int((j-27) / 3)]))
                    bar()
                    # Left shifts
                    mzsq_msqsl12mh[int((j-27) / 3)] = deriv_step_calc_sq_sl12(j, (-1) * sq_sl12_hvals[int((j-27) / 3)])
                    bar()
                    mzsq_msqsl12mmh[int((j-27) / 3)] = deriv_step_calc_sq_sl12(j,
                                                                (-1) * (2 * sq_sl12_hvals[int((j-27) / 3)]))
                    bar()
                    derivcount += 1
                    bar.text = f'Derivatives: {derivcount}/19, please wait...'

                for j in [29,32,35,38,41]:
                    print("Error estimate for " + str(sq_sl3_labels[int((j-29) / 3)]) + " derivative: "
                          + str("{:.4e}".format(float(mp.power(sq_sl3_hvals[int((j-29) / 3)], 4)))))
                    # For loop to compute squark and slepton mass derivatives of first two gens
                    # Right shifts
                    mzsq_msqsl3ph[int((j-29) / 3)] = deriv_step_calc_sq_sl3(j, sq_sl3_hvals[int((j-29) / 3)])
                    bar()
                    mzsq_msqsl3pph[int((j-29) / 3)] = deriv_step_calc_sq_sl3(j,
                                                                (2 * sq_sl3_hvals[int((j-29) / 3)]))
                    bar()
                    # Left shifts
                    mzsq_msqsl3mh[int((j-29) / 3)] = deriv_step_calc_sq_sl3(j, (-1) * sq_sl3_hvals[int((j-29) / 3)])
                    bar()
                    mzsq_msqsl3mmh[int((j-29) / 3)] = deriv_step_calc_sq_sl3(j,
                                                                         (-1) * (2 * sq_sl3_hvals[int((j-29) / 3)]))
                    bar()
                    derivcount += 1
                    bar.text = f'Derivatives: {derivcount}/19, please wait...'

                for j in [16,19,22]:
                    print("Error estimate for " + str(Ai_labels[int((j-16) / 3)]) + " derivative: "
                          + str("{:.4e}".format(float(mp.power(Ai_hvals[int((j-16) / 3)],4)))))
                    # For loop to compute soft trilinear coupling derivatives
                    # Right shifts
                    mzsq_Aiph[int((j-16) / 3)] = deriv_step_calc_trilin(j, Ai_hvals[int((j-16) / 3)])
                    bar()
                    mzsq_Aipph[int((j-16) / 3)] = deriv_step_calc_trilin(j,
                                                                     (2 * Ai_hvals[int((j-16) / 3)]))
                    bar()
                    # Left shifts
                    mzsq_Aimh[int((j-16) / 3)] = deriv_step_calc_trilin(j, (-1) * Ai_hvals[int((j-16) / 3)])
                    bar()
                    mzsq_Aimmh[int((j-16) / 3)] = deriv_step_calc_trilin(j,
                                                                     (-1) * (2 * Ai_hvals[int((j-16) / 3)]))
                    bar()
                    derivcount += 1
                    bar.text = f'Derivatives: {derivcount}/19, please wait...'

                for j in np.arange(3, 6):
                    print("Error estimate for " + str(Mi_labels[j-3]) + " derivative: "
                          + str("{:.4e}".format(float(mp.power(Mi_hvals[j-3], 4)))))
                    # For loop to compute soft gaugino mass derivatives
                    # Right shifts
                    mzsq_Miph[j-3] = deriv_step_calc_gaugino(j, Mi_hvals[j-3])
                    bar()
                    mzsq_Mipph[j-3] = deriv_step_calc_gaugino(j,
                                                              (2 * Mi_hvals[j-3]))
                    bar()
                    # Left shifts
                    mzsq_Mimh[j-3] = deriv_step_calc_gaugino(j, (-1) * Mi_hvals[j-3])
                    bar()
                    mzsq_Mimmh[j-3] = deriv_step_calc_gaugino(j,
                                                              (-1) * (2 * Mi_hvals[j-3]))
                    bar()
                    derivcount += 1
                    bar.text = f'Derivatives: {derivcount}/19, please wait...'

                for j in np.arange(25, 27):
                    print("Error estimate for " + str(Higgs_labels[j-25]) + " derivative: "
                          + str("{:.4e}".format(float(mp.power(Higgs_hvals[j-25], 4)))))
                    # For loop to compute soft Higgs mass derivatives
                    # Right shifts
                    mzsq_mHph[j-25] = deriv_step_calc_Higgs(j, Higgs_hvals[j-25])
                    bar()
                    mzsq_mHpph[j-25] = deriv_step_calc_Higgs(j,
                                                             (2 * Higgs_hvals[j-25]))
                    bar()
                    # Left shifts
                    mzsq_mHmh[j-25] = deriv_step_calc_Higgs(j, (-1) * Higgs_hvals[j-25])
                    bar()
                    mzsq_mHmmh[j-25] = deriv_step_calc_Higgs(j,
                                                             (-1) * (2 * Higgs_hvals[j-25]))
                    bar()
                    derivcount += 1
                    bar.text = f'Derivatives: {derivcount}/19, please wait...'

                print("Error estimate for mu_0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hmu0, 4)))))
                mzsq_mu0ph = deriv_step_calc_mu0(hmu0)
                bar()
                mzsq_mu0pph = deriv_step_calc_mu0(2*hmu0)
                bar()
                mzsq_mu0mh = deriv_step_calc_mu0((-1) * hmu0)
                bar()
                mzsq_mu0mmh = deriv_step_calc_mu0((-1) * 2*hmu0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/19, please wait...'

                # Construct derivative array
                deriv_array = np.array([deriv_num_calc(hmqL12,
                                                       mzsq_msqsl12mmh[0], mzsq_msqsl12mh[0], mzsq_msqsl12ph[0],
                                                       mzsq_msqsl12pph[0]), #end qL12: 0
                                        deriv_num_calc(hmeL12,
                                                       mzsq_msqsl12mmh[1], mzsq_msqsl12mh[1], mzsq_msqsl12ph[1],
                                                       mzsq_msqsl12pph[1]), #end eL12: 1
                                        deriv_num_calc(hmuR12,
                                                       mzsq_msqsl12mmh[2], mzsq_msqsl12mh[2], mzsq_msqsl12ph[2],
                                                       mzsq_msqsl12pph[2]), #end uR12: 2
                                        deriv_num_calc(hmdR12,
                                                       mzsq_msqsl12mmh[3], mzsq_msqsl12mh[3], mzsq_msqsl12ph[3],
                                                       mzsq_msqsl12pph[3]), #end dR12: 3
                                        deriv_num_calc(hmeR12,
                                                       mzsq_msqsl12mmh[4], mzsq_msqsl12mh[4], mzsq_msqsl12ph[4],
                                                       mzsq_msqsl12pph[4]), #end eR12: 4
                                        deriv_num_calc(hmqL3,
                                                       mzsq_msqsl3mmh[0], mzsq_msqsl3mh[0], mzsq_msqsl3ph[0],
                                                       mzsq_msqsl3pph[0]), #end qL3: 5
                                        deriv_num_calc(hmtauL,
                                                       mzsq_msqsl3mmh[1], mzsq_msqsl3mh[1], mzsq_msqsl3ph[1],
                                                       mzsq_msqsl3pph[1]), #end tauL: 6
                                        deriv_num_calc(hmtR,
                                                       mzsq_msqsl3mmh[2], mzsq_msqsl3mh[2], mzsq_msqsl3ph[2],
                                                       mzsq_msqsl3pph[2]), #end tR: 7
                                        deriv_num_calc(hmbR,
                                                       mzsq_msqsl3mmh[3], mzsq_msqsl3mh[3], mzsq_msqsl3ph[3],
                                                       mzsq_msqsl3pph[3]), #end bR: 8
                                        deriv_num_calc(hmtauR,
                                                       mzsq_msqsl3mmh[4], mzsq_msqsl3mh[4], mzsq_msqsl3ph[4],
                                                       mzsq_msqsl3pph[4]), #end tauR: 9
                                        deriv_num_calc(hAt,
                                                       mzsq_Aimmh[0], mzsq_Aimh[0], mzsq_Aiph[0],
                                                       mzsq_Aipph[0]), #end At: 10
                                        deriv_num_calc(hAb,
                                                       mzsq_Aimmh[1], mzsq_Aimh[1], mzsq_Aiph[1],
                                                       mzsq_Aipph[1]), #end Ab: 11
                                        deriv_num_calc(hAtau,
                                                       mzsq_Aimmh[2], mzsq_Aimh[2], mzsq_Aiph[2],
                                                       mzsq_Aipph[2]), #end Atau: 12
                                        deriv_num_calc(hM1,
                                                       mzsq_Mimmh[0], mzsq_Mimh[0], mzsq_Miph[0],
                                                       mzsq_Mipph[0]), #end M1: 13
                                        deriv_num_calc(hM2,
                                                       mzsq_Mimmh[1], mzsq_Mimh[1], mzsq_Miph[1],
                                                       mzsq_Mipph[1]), #end M2: 14
                                        deriv_num_calc(hM3,
                                                       mzsq_Mimmh[2], mzsq_Mimh[2], mzsq_Miph[2],
                                                       mzsq_Mipph[2]), #end M3: 15
                                        deriv_num_calc(hmHu0,
                                                       mzsq_mHmmh[0], mzsq_mHmh[0], mzsq_mHph[0],
                                                       mzsq_mHpph[0]), #end mHu0: 16
                                        deriv_num_calc(hmHd0,
                                                       mzsq_mHmmh[1], mzsq_mHmh[1], mzsq_mHph[1],
                                                       mzsq_mHpph[1]), #end mHd0: 17
                                        deriv_num_calc(hmu0,
                                                       mzsq_mu0mmh, mzsq_mu0mh, mzsq_mu0ph,
                                                       mzsq_mu0pph)]) #end mu0: 18
                sens_params = np.sort(np.array([(mp.fabs((mymqL12 / mymzsq)
                                                         * deriv_array[0]),
                                                 'Delta_BG(m_qL(1,2))'),
                                                (mp.fabs((mymeL12 / mymzsq)
                                                         * deriv_array[1]),
                                                 'Delta_BG(m_eL(1,2))'),
                                                (mp.fabs((mymuR12 / mymzsq)
                                                         * deriv_array[2]),
                                                 'Delta_BG(m_uR(1,2))'),
                                                (mp.fabs((mymdR12 / mymzsq)
                                                         * deriv_array[3]),
                                                 'Delta_BG(m_dR(1,2))'),
                                                (mp.fabs((mymeR12 / mymzsq)
                                                         * deriv_array[4]),
                                                 'Delta_BG(m_eR(1,2))'),
                                                (mp.fabs((mymqL3 / mymzsq)
                                                         * deriv_array[5]),
                                                 'Delta_BG(m_qL(3))'),
                                                (mp.fabs((mymtauL / mymzsq)
                                                         * deriv_array[6]),
                                                 'Delta_BG(m_tauL)'),
                                                (mp.fabs((mymtR / mymzsq)
                                                         * deriv_array[7]),
                                                 'Delta_BG(m_tR)'),
                                                (mp.fabs((mymbR / mymzsq)
                                                         * deriv_array[8]),
                                                 'Delta_BG(m_bR)'),
                                                (mp.fabs((mymtauR / mymzsq)
                                                         * deriv_array[9]),
                                                 'Delta_BG(m_tauR)'),
                                                (mp.fabs((myAt / mymzsq)
                                                         * deriv_array[10]),
                                                 'Delta_BG(a_t)'),
                                                (mp.fabs((myAb / mymzsq)
                                                         * deriv_array[11]),
                                                 'Delta_BG(a_b)'),
                                                (mp.fabs((myAtau / mymzsq)
                                                         * deriv_array[12]),
                                                 'Delta_BG(a_tau)'),
                                                (mp.fabs((myM1 / mymzsq)
                                                         * deriv_array[13]),
                                                 'Delta_BG(M_1)'),
                                                (mp.fabs((myM2 / mymzsq)
                                                         * deriv_array[14]),
                                                 'Delta_BG(M_2)'),
                                                (mp.fabs((myM3 / mymzsq)
                                                         * deriv_array[15]),
                                                 'Delta_BG(M_3)'),
                                                (mp.fabs((mymHu0 / mymzsq)
                                                         * deriv_array[16]),
                                                 'Delta_BG(m_Hu0)'),
                                                (mp.fabs((mymHd0 / mymzsq)
                                                         * deriv_array[17]),
                                                 'Delta_BG(m_Hd0)'),
                                                (mp.fabs((mymu0 / mymzsq)
                                                         * deriv_array[18]),
                                                 'Delta_BG(mu0)')],
                                               dtype=[('BGcontrib',float),
                                                      ('BGlabel','U40')]),
                                      order='BGcontrib')
                print("Done!")

        elif (precselno == 3):
            derivcount = 0
            mzsq_Aiph = np.zeros(3)
            mzsq_Aimh = np.zeros(3)
            mzsq_Miph = np.zeros(3)
            mzsq_Mimh = np.zeros(3)
            mzsq_mHph = np.zeros(2)
            mzsq_mHmh = np.zeros(2)
            # mzsq_msqslpmh12 order: [m_qL(1,2), m_eL(1,2), m_uR(1,2),
            #                         m_dR(1,2), m_eR(1,2)]
            # mzsq_msqslpmh3 order: [m_qL(3), m_tauL, m_tR, m_bR, m_tauR]
            mzsq_msqsl12ph = np.zeros(5)
            mzsq_msqsl12mh = np.zeros(5)
            mzsq_msqsl3ph = np.zeros(5)
            mzsq_msqsl3mh = np.zeros(5)
            hmqL12 = mpf(str(abs(mp.power(math.ulp(mymqL12), 1/3))))
            hmqL3 = mpf(str(abs(mp.power(math.ulp(mymqL3), 1/3))))
            hmuR12 = mpf(str(abs(mp.power(math.ulp(mymuR12), 1/3))))
            hmtR = mpf(str(abs(mp.power(math.ulp(mymtR), 1/3))))
            hmdR12 = mpf(str(abs(mp.power(math.ulp(mymdR12), 1/3))))
            hmbR = mpf(str(abs(mp.power(math.ulp(mymbR), 1/3))))
            hmeL12 = mpf(str(abs(mp.power(math.ulp(mymeL12), 1/3))))
            hmtauL = mpf(str(abs(mp.power(math.ulp(mymtauL), 1/3))))
            hmeR12 = mpf(str(abs(mp.power(math.ulp(mymeR12), 1/3))))
            hmtauR = mpf(str(abs(mp.power(math.ulp(mymtauR), 1/3))))
            sq_sl12_hvals = [hmqL12, hmeL12, hmuR12, hmdR12, hmeR12]
            sq_sl12_labels = ['m_qL(1,2)','m_eL(1,2)','m_uR(1,2)','m_dR(1,2)',
                              'm_eR(1,2)']
            sq_sl3_hvals = [hmqL3, hmtauL, hmtR, hmbR, hmtauR]
            sq_sl3_labels = ['m_qL(3)','m_tauL','m_tR','m_bR',
                              'm_tauR']
            hM1 = mpf(str(abs(mp.power(math.ulp(myM1), 1/3))))
            hM2 = mpf(str(abs(mp.power(math.ulp(myM2), 1/3))))
            hM3 = mpf(str(abs(mp.power(math.ulp(myM3), 1/3))))
            Mi_hvals = [hM1, hM2, hM3]
            Mi_labels = ['M_1', 'M_2', 'M_3']
            hAt = mpf(str(abs(mp.power(math.ulp(myAt), 1/3))))
            hAb = mpf(str(abs(mp.power(math.ulp(myAb), 1/3))))
            hAtau = mpf(str(abs(mp.power(math.ulp(myAtau), 1/3))))
            Ai_hvals = [hAt, hAb, hAtau]
            Ai_labels = ['a_t','a_b','a_tau']
            hmu0 = mpf(str(abs(mp.power(math.ulp(mymu0), 1/3))))
            hmHu0 = mpf(str(abs(mp.power(math.ulp(mymHu0), 1/3))))
            hmHd0 = mpf(str(abs(mp.power(math.ulp(mymHd0), 1/3))))
            Higgs_hvals = [hmHu0, hmHd0]
            Higgs_labels = ['m_Hu0', 'm_Hd0']

            with alive_bar(38, dual_line=True,
                           force_tty=True, title='Computations: ',
                           bar='brackets') as bar:
                bar.text = f'Derivatives: {derivcount}/19, please wait...'
                for j in [27,30,33,36,39]:
                    print("Error estimate for " + str(sq_sl12_labels[int((j-27) / 3)]) + " derivative: "
                          + str("{:.4e}".format(float(mp.power(sq_sl12_hvals[int((j-27) / 3)],2)))))
                    # For loop to compute squark and slepton mass derivatives of first two gens
                    # Right shifts
                    mzsq_msqsl12ph[int((j-27) / 3)] = deriv_step_calc_sq_sl12(j, sq_sl12_hvals[int((j-27) / 3)])
                    bar()
                    # Left shifts
                    mzsq_msqsl12mh[int((j-27) / 3)] = deriv_step_calc_sq_sl12(j, (-1) * sq_sl12_hvals[int((j-27) / 3)])
                    bar()
                    derivcount += 1
                    bar.text = f'Derivatives: {derivcount}/19, please wait...'

                for j in [29,32,35,38,41]:
                    print("Error estimate for " + str(sq_sl3_labels[int((j-29) / 3)]) + " derivative: "
                          + str("{:.4e}".format(float(mp.power(sq_sl3_hvals[int((j-29) / 3)], 2)))))
                    # For loop to compute squark and slepton mass derivatives of first two gens
                    # Right shifts
                    mzsq_msqsl3ph[int((j-29) / 3)] = deriv_step_calc_sq_sl3(j, sq_sl3_hvals[int((j-29) / 3)])
                    bar()
                    # Left shifts
                    mzsq_msqsl3mh[int((j-29) / 3)] = deriv_step_calc_sq_sl3(j, (-1) * sq_sl3_hvals[int((j-29) / 3)])
                    bar()
                    derivcount += 1
                    bar.text = f'Derivatives: {derivcount}/19, please wait...'

                for j in [16,19,22]:
                    print("Error estimate for " + str(Ai_labels[int((j-16) / 3)]) + " derivative: "
                          + str("{:.4e}".format(float(mp.power(Ai_hvals[int((j-16) / 3)], 2)))))
                    # For loop to compute soft trilinear coupling derivatives
                    # Right shifts
                    mzsq_Aiph[int((j-16) / 3)] = deriv_step_calc_trilin(j, Ai_hvals[int((j-16) / 3)])
                    bar()
                    # Left shifts
                    mzsq_Aimh[int((j-16) / 3)] = deriv_step_calc_trilin(j, (-1) * Ai_hvals[int((j-16) / 3)])
                    bar()
                    derivcount += 1
                    bar.text = f'Derivatives: {derivcount}/19, please wait...'

                for j in np.arange(3, 6):
                    print("Error estimate for " + str(Mi_labels[j-3]) + " derivative: "
                          + str("{:.4e}".format(float(mp.power(Mi_hvals[j-3], 2)))))
                    # For loop to compute soft gaugino mass derivatives
                    # Right shifts
                    mzsq_Miph[j-3] = deriv_step_calc_gaugino(j, Mi_hvals[j-3])
                    bar()
                    # Left shifts
                    mzsq_Mimh[j-3] = deriv_step_calc_gaugino(j, (-1) * Mi_hvals[j-3])
                    bar()
                    derivcount += 1
                    bar.text = f'Derivatives: {derivcount}/19, please wait...'

                for j in np.arange(25, 27):
                    print("Error estimate for " + str(Higgs_labels[j-25]) + " derivative: "
                          + str("{:.4e}".format(float(mp.power(Higgs_hvals[j-25], 2)))))
                    # For loop to compute soft Higgs mass derivatives
                    # Right shifts
                    mzsq_mHph[j-25] = deriv_step_calc_Higgs(j, Higgs_hvals[j-25])
                    bar()
                    # Left shifts
                    mzsq_mHmh[j-25] = deriv_step_calc_Higgs(j, (-1) * Higgs_hvals[j-25])
                    bar()
                    derivcount += 1
                    bar.text = f'Derivatives: {derivcount}/19, please wait...'

                print("Error estimate for mu_0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hmu0, 8)))))
                mzsq_mu0ph = deriv_step_calc_mu0(hmu0)
                bar()
                mzsq_mu0mh = deriv_step_calc_mu0((-1) * hmu0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/19, please wait...'

                # Construct derivative array
                deriv_array = np.array([deriv_num_calc(hmqL12, mzsq_msqsl12mh[0], mzsq_msqsl12ph[0]), #end qL12: 0
                                        deriv_num_calc(hmeL12, mzsq_msqsl12mh[1], mzsq_msqsl12ph[1]), #end eL12: 1
                                        deriv_num_calc(hmuR12, mzsq_msqsl12mh[2], mzsq_msqsl12ph[2]), #end uR12: 2
                                        deriv_num_calc(hmdR12, mzsq_msqsl12mh[3], mzsq_msqsl12ph[3]), #end dR12: 3
                                        deriv_num_calc(hmeR12,mzsq_msqsl12mh[4], mzsq_msqsl12ph[4]), #end eR12: 4
                                        deriv_num_calc(hmqL3,mzsq_msqsl3mh[0], mzsq_msqsl3ph[0]), #end qL3: 5
                                        deriv_num_calc(hmtauL, mzsq_msqsl3mh[1], mzsq_msqsl3ph[1]), #end tauL: 6
                                        deriv_num_calc(hmtR, mzsq_msqsl3mh[2], mzsq_msqsl3ph[2]), #end tR: 7
                                        deriv_num_calc(hmbR,mzsq_msqsl3mh[3], mzsq_msqsl3ph[3]), #end bR: 8
                                        deriv_num_calc(hmtauR, mzsq_msqsl3mh[4], mzsq_msqsl3ph[4]), #end tauR: 9
                                        deriv_num_calc(hAt,mzsq_Aimh[0], mzsq_Aiph[0]), #end At: 10
                                        deriv_num_calc(hAb,mzsq_Aimh[1], mzsq_Aiph[1]), #end Ab: 11
                                        deriv_num_calc(hAtau,mzsq_Aimh[2], mzsq_Aiph[2]), #end Atau: 12
                                        deriv_num_calc(hM1,mzsq_Mimh[0], mzsq_Miph[0]), #end M1: 13
                                        deriv_num_calc(hM2,mzsq_Mimh[1], mzsq_Miph[1]), #end M2: 14
                                        deriv_num_calc(hM3,mzsq_Mimh[2], mzsq_Miph[2]), #end M3: 15
                                        deriv_num_calc(hmHu0, mzsq_mHmh[0], mzsq_mHph[0]), #end mHu0: 16
                                        deriv_num_calc(hmHd0,mzsq_mHmh[1], mzsq_mHph[1]), #end mHd0: 17
                                        deriv_num_calc(hmu0,mzsq_mu0mh, mzsq_mu0ph)]) #end mu0: 18
                sens_params = np.sort(np.array([(mp.fabs((mymqL12 / mymzsq)
                                                         * deriv_array[0]),
                                                 'Delta_BG(m_qL(1,2))'),
                                                (mp.fabs((mymeL12 / mymzsq)
                                                         * deriv_array[1]),
                                                 'Delta_BG(m_eL(1,2))'),
                                                (mp.fabs((mymuR12 / mymzsq)
                                                         * deriv_array[2]),
                                                 'Delta_BG(m_uR(1,2))'),
                                                (mp.fabs((mymdR12 / mymzsq)
                                                         * deriv_array[3]),
                                                 'Delta_BG(m_dR(1,2))'),
                                                (mp.fabs((mymeR12 / mymzsq)
                                                         * deriv_array[4]),
                                                 'Delta_BG(m_eR(1,2))'),
                                                (mp.fabs((mymqL3 / mymzsq)
                                                         * deriv_array[5]),
                                                 'Delta_BG(m_qL(3))'),
                                                (mp.fabs((mymtauL / mymzsq)
                                                         * deriv_array[6]),
                                                 'Delta_BG(m_tauL)'),
                                                (mp.fabs((mymtR / mymzsq)
                                                         * deriv_array[7]),
                                                 'Delta_BG(m_tR)'),
                                                (mp.fabs((mymbR / mymzsq)
                                                         * deriv_array[8]),
                                                 'Delta_BG(m_bR)'),
                                                (mp.fabs((mymtauR / mymzsq)
                                                         * deriv_array[9]),
                                                 'Delta_BG(m_tauR)'),
                                                (mp.fabs((myAt / mymzsq)
                                                         * deriv_array[10]),
                                                 'Delta_BG(a_t)'),
                                                (mp.fabs((myAb / mymzsq)
                                                         * deriv_array[11]),
                                                 'Delta_BG(a_b)'),
                                                (mp.fabs((myAtau / mymzsq)
                                                         * deriv_array[12]),
                                                 'Delta_BG(a_tau)'),
                                                (mp.fabs((myM1 / mymzsq)
                                                         * deriv_array[13]),
                                                 'Delta_BG(M_1)'),
                                                (mp.fabs((myM2 / mymzsq)
                                                         * deriv_array[14]),
                                                 'Delta_BG(M_2)'),
                                                (mp.fabs((myM3 / mymzsq)
                                                         * deriv_array[15]),
                                                 'Delta_BG(M_3)'),
                                                (mp.fabs((mymHu0 / mymzsq)
                                                         * deriv_array[16]),
                                                 'Delta_BG(m_Hu0)'),
                                                (mp.fabs((mymHd0 / mymzsq)
                                                         * deriv_array[17]),
                                                 'Delta_BG(m_Hd0)'),
                                                (mp.fabs((mymu0 / mymzsq)
                                                         * deriv_array[18]),
                                                 'Delta_BG(mu0)')],
                                               dtype=[('BGcontrib',float),
                                                      ('BGlabel','U40')]),
                                      order='BGcontrib')
                print("Done!")

    elif (modselno == 7):
        print("Computing sensitivity coefficient derivatives...")
        print("NOTE: this computation can take a while")
        mymqL1 = inputGUT_BCs[27]
        mymqL2 = inputGUT_BCs[28]
        mymqL3 = inputGUT_BCs[29]
        mymtR = inputGUT_BCs[35]
        mymcR = inputGUT_BCs[34]
        mymuR = inputGUT_BCs[33]
        mymbR = inputGUT_BCs[38]
        mymsR = inputGUT_BCs[37]
        mymdR = inputGUT_BCs[36]
        mymtauL = inputGUT_BCs[32]
        mymmuL = inputGUT_BCs[31]
        mymeL = inputGUT_BCs[30]
        mymtauR = inputGUT_BCs[41]
        mymmuR = inputGUT_BCs[40]
        mymeR = inputGUT_BCs[39]
        myM1 = inputGUT_BCs[3]
        myM2 = inputGUT_BCs[4]
        myM3 = inputGUT_BCs[5]
        myAt = inputGUT_BCs[16]
        myAc = inputGUT_BCs[17]
        myAu = inputGUT_BCs[18]
        myAb = inputGUT_BCs[19]
        myAs = inputGUT_BCs[20]
        myAd = inputGUT_BCs[21]
        myAtau = inputGUT_BCs[22]
        myAmu = inputGUT_BCs[23]
        myAe = inputGUT_BCs[24]
        mymusq0 = inputGUT_BCs[6]
        mymu0 = mp.sqrt(str(abs(mymusq0)))
        mymHusqGUT = inputGUT_BCs[25]
        mymHu0 = mp.sqrt(str(abs(mymHusqGUT)))
        mymHdsqGUT = inputGUT_BCs[26]
        mymHd0 = mp.sqrt(str(abs(mymHdsqGUT)))
        def deriv_step_calc_mu0(shift_amt):
            """
            Do the current mass derivative step for higgsino mass mu.

            Parameters
            ----------
            shift_amt : Float.
                How much to shift for current derivative step.

            Returns
            -------
            mytestmzsq : Float.
                Return current shifted value of mZ^2 with the
                    higgsino mass shifted by the specified amount.

            """
            testBCs = deepcopy(inputGUT_BCs[:])
            for i in np.arange(27,42):
                testBCs[i] = mp.power(testBCs[i], 2)
            testBCs[6] = np.sign(mymusq0) * mp.power(mymu0 + shift_amt, 2)
            mytestmzsq = deriv_step_calc(GUT_SCALE,myweakscale, inptanbval, polecalccheck, deepcopy(testBCs))
            del testBCs
            return mytestmzsq

        def deriv_step_calc_Higgs(current_index, shift_amt):
            """
            Do the current mass derivative step for soft Higgs masses.

            Parameters
            ----------
            current_index : Int.
                Index of parameter to perform derivative with.
            shift_amt : Float.
                How much to shift for current derivative step.

            Returns
            -------
            mytestmzsq : Float.
                Return current shifted value of mZ^2 with the corresponding
                    soft Higgs mass shifted by the specified amount.

            """
            testBCs = deepcopy(inputGUT_BCs[:])
            for i in np.arange(27,42):
                testBCs[i] = mp.power(testBCs[i], 2)
            testBCs[current_index] = mp.power(mp.sqrt(str(abs(testBCs[current_index])))
                                              + shift_amt, 2)
            mytestmzsq = deriv_step_calc(GUT_SCALE,myweakscale, inptanbval, polecalccheck, deepcopy(testBCs))
            del testBCs
            return mytestmzsq

        def deriv_step_calc_sq_sl(current_index, shift_amt):
            """
            Do the current mass derivative step for squarks or sleptons.

            Parameters
            ----------
            current_index : Int.
                Index of parameter to perform derivative with.
            shift_amt : Float.
                How much to shift for current derivative step.

            Returns
            -------
            mytestmzsq : Float.
                Return current shifted value of mZ^2 with the corresponding
                    squark or slepton mass shifted by the specified amount.

            """
            testBCs = deepcopy(inputGUT_BCs[:])
            testBCs[current_index] = mp.power(testBCs[current_index] + shift_amt, 2)
            remaining_indices = np.delete(np.arange(27,42,dtype=np.int32),
                                          current_index-27)
            for i in remaining_indices:
                testBCs[i] = mp.power(testBCs[i], 2)
            mytestmzsq = deriv_step_calc(GUT_SCALE,myweakscale, inptanbval, polecalccheck, deepcopy(testBCs))
            del testBCs
            return mytestmzsq

        def deriv_step_calc_trilin(current_index, shift_amt):
            """
            Do the current mass derivative step for soft trilinear couplings.

            Parameters
            ----------
            current_index : Int.
                Index of parameter to perform derivative with.
            shift_amt : Float.
                How much to shift for current derivative step.

            Returns
            -------
            mytestmzsq : Float.
                Return current shifted value of mZ^2 with the corresponding
                    soft trilinear coupling shifted by the specified amount.

            """
            testBCs = deepcopy(inputGUT_BCs[:])
            for i in np.arange(27,42):
                testBCs[i] = mp.power(testBCs[i], 2)
            testBCs[current_index] = testBCs[current_index] + shift_amt
            mytestmzsq = deriv_step_calc(GUT_SCALE,myweakscale, inptanbval, polecalccheck, deepcopy(testBCs))
            del testBCs
            return mytestmzsq

        def deriv_step_calc_gaugino(current_index, shift_amt):
            """
            Do the current mass derivative step for soft gaugino masses.

            Parameters
            ----------
            current_index : Int.
                Index of parameter to perform derivative with.
            shift_amt : Float.
                How much to shift for current derivative to step.

            Returns
            -------
            mytestmzsq : Float.
                Return current shifted value of mZ^2 with the corresponding
                    soft gaugino mass shifted by the specified amount.

            """
            testBCs = deepcopy(inputGUT_BCs[:])
            for i in np.arange(27,42):
                testBCs[i] = mp.power(testBCs[i], 2)
            testBCs[current_index] = testBCs[current_index] + shift_amt
            mytestmzsq = deriv_step_calc(GUT_SCALE,myweakscale, inptanbval, polecalccheck, deepcopy(testBCs))
            del testBCs
            return mytestmzsq

        if (precselno == 1):
            derivcount = 0
            mzsq_msqslph = np.zeros(15)
            mzsq_msqslpph = np.zeros(15)
            mzsq_msqslppph = np.zeros(15)
            mzsq_msqslpppph = np.zeros(15)
            mzsq_msqslmh = np.zeros(15)
            mzsq_msqslmmh = np.zeros(15)
            mzsq_msqslmmmh = np.zeros(15)
            mzsq_msqslmmmmh = np.zeros(15)
            mzsq_Aiph = np.zeros(9)
            mzsq_Aipph = np.zeros(9)
            mzsq_Aippph = np.zeros(9)
            mzsq_Aipppph = np.zeros(9)
            mzsq_Aimh = np.zeros(9)
            mzsq_Aimmh = np.zeros(9)
            mzsq_Aimmmh = np.zeros(9)
            mzsq_Aimmmmh = np.zeros(9)
            mzsq_Miph = np.zeros(3)
            mzsq_Mipph = np.zeros(3)
            mzsq_Mippph = np.zeros(3)
            mzsq_Mipppph = np.zeros(3)
            mzsq_Mimh = np.zeros(3)
            mzsq_Mimmh = np.zeros(3)
            mzsq_Mimmmh = np.zeros(3)
            mzsq_Mimmmmh = np.zeros(3)
            mzsq_mHph = np.zeros(2)
            mzsq_mHpph = np.zeros(2)
            mzsq_mHppph = np.zeros(2)
            mzsq_mHpppph = np.zeros(2)
            mzsq_mHmh = np.zeros(2)
            mzsq_mHmmh = np.zeros(2)
            mzsq_mHmmmh = np.zeros(2)
            mzsq_mHmmmmh = np.zeros(2)
            hmqL1 = mpf(str(abs(mp.power(math.ulp(mymqL1), 1/9))))
            hmqL2 = mpf(str(abs(mp.power(math.ulp(mymqL2), 1/9))))
            hmqL3 = mpf(str(abs(mp.power(math.ulp(mymqL3), 1/9))))
            hmuR = mpf(str(abs(mp.power(math.ulp(mymuR), 1/9))))
            hmcR = mpf(str(abs(mp.power(math.ulp(mymcR), 1/9))))
            hmtR = mpf(str(abs(mp.power(math.ulp(mymtR), 1/9))))
            hmdR = mpf(str(abs(mp.power(math.ulp(mymdR), 1/9))))
            hmsR = mpf(str(abs(mp.power(math.ulp(mymsR), 1/9))))
            hmbR = mpf(str(abs(mp.power(math.ulp(mymbR), 1/9))))
            hmeL = mpf(str(abs(mp.power(math.ulp(mymeL), 1/9))))
            hmmuL = mpf(str(abs(mp.power(math.ulp(mymmuL), 1/9))))
            hmtauL = mpf(str(abs(mp.power(math.ulp(mymtauL), 1/9))))
            hmeR = mpf(str(abs(mp.power(math.ulp(mymeR), 1/9))))
            hmmuR = mpf(str(abs(mp.power(math.ulp(mymmuR), 1/9))))
            hmtauR = mpf(str(abs(mp.power(math.ulp(mymtauR), 1/9))))
            sq_sl_hvals = [hmqL1, hmqL2, hmqL3, hmeL, hmmuL, hmtauL,
                           hmuR, hmcR, hmtR, hmdR, hmsR, hmbR, hmeR, hmmuR,
                           hmtauR]
            sq_sl_labels = ['m_qL(1)','m_qL(2)','m_qL(3)','m_eL','m_muL',
                            'm_tauL','m_uR','m_cR','m_tR','m_dR',
                            'm_sR','m_bR','m_eR','m_muR','m_tauR']
            hM1 = mpf(str(abs(mp.power(math.ulp(myM1), 1/9))))
            hM2 = mpf(str(abs(mp.power(math.ulp(myM2), 1/9))))
            hM3 = mpf(str(abs(mp.power(math.ulp(myM3), 1/9))))
            Mi_hvals = [hM1, hM2, hM3]
            Mi_labels = ['M_1', 'M_2', 'M_3']
            hAt = mpf(str(abs(mp.power(math.ulp(myAt), 1/9))))
            hAc = mpf(str(abs(mp.power(math.ulp(myAc), 1/9))))
            hAu = mpf(str(abs(mp.power(math.ulp(myAu), 1/9))))
            hAb = mpf(str(abs(mp.power(math.ulp(myAb), 1/9))))
            hAs = mpf(str(abs(mp.power(math.ulp(myAs), 1/9))))
            hAd = mpf(str(abs(mp.power(math.ulp(myAd), 1/9))))
            hAtau = mpf(str(abs(mp.power(math.ulp(myAtau), 1/9))))
            hAmu = mpf(str(abs(mp.power(math.ulp(myAmu), 1/9))))
            hAe = mpf(str(abs(mp.power(math.ulp(myAe), 1/9))))
            Ai_hvals = [hAt, hAc, hAu, hAb, hAs, hAd, hAtau, hAmu, hAe]
            Ai_labels = ['a_t', 'a_c', 'a_u','a_b', 'a_s', 'a_d','a_tau', 'a_mu', 'a_e']
            hmu0 = mpf(str(abs(mp.power(math.ulp(mymu0), 1/9))))
            hmHu0 = mpf(str(abs(mp.power(math.ulp(mymHu0), 1/9))))
            hmHd0 = mpf(str(abs(mp.power(math.ulp(mymHd0), 1/9))))
            Higgs_hvals = [hmHu0, hmHd0]
            Higgs_labels = ['m_Hu0', 'm_Hd0']

            with alive_bar(240, dual_line=True,
                           force_tty=True, title='Computations: ',
                           bar='brackets') as bar:
                bar.text = f'Derivatives: {derivcount}/30, please wait...'
                for j in np.arange(27, 42):
                    print("Error estimate for " + str(sq_sl_labels[j-27]) + " derivative: "
                          + str("{:.4e}".format(float(mp.power(sq_sl_hvals[j-27], 8)))))
                    # For loop to compute squark and slepton mass derivatives
                    # Right shifts
                    mzsq_msqslph[j-27] = deriv_step_calc_sq_sl(j, sq_sl_hvals[j-27])
                    bar()
                    mzsq_msqslpph[j-27] = deriv_step_calc_sq_sl(j,
                                                                (2 * sq_sl_hvals[j-27]))
                    bar()
                    mzsq_msqslppph[j-27] = deriv_step_calc_sq_sl(j,
                                                                 (3 * sq_sl_hvals[j-27]))
                    bar()
                    mzsq_msqslpppph[j-27] = deriv_step_calc_sq_sl(j,
                                                                  (4 * sq_sl_hvals[j-27]))
                    bar()
                    # Left shifts
                    mzsq_msqslmh[j-27] = deriv_step_calc_sq_sl(j, (-1) * sq_sl_hvals[j-27])
                    bar()
                    mzsq_msqslmmh[j-27] = deriv_step_calc_sq_sl(j,
                                                                (-1) * (2 * sq_sl_hvals[j-27]))
                    bar()
                    mzsq_msqslmmmh[j-27] = deriv_step_calc_sq_sl(j,
                                                                 (-1) * (3 * sq_sl_hvals[j-27]))
                    bar()
                    mzsq_msqslmmmmh[j-27] = deriv_step_calc_sq_sl(j,
                                                                  (-1) * (4 * sq_sl_hvals[j-27]))
                    bar()
                    derivcount += 1
                    bar.text = f'Derivatives: {derivcount}/30, please wait...'

                for j in np.arange(16, 25):
                    print("Error estimate for " + str(Ai_labels[j-16]) + " derivative: "
                          + str("{:.4e}".format(float(mp.power(Ai_hvals[j-16], 8)))))
                    # For loop to compute soft trilinear coupling derivatives
                    # Right shifts
                    mzsq_Aiph[j-16] = deriv_step_calc_trilin(j, Ai_hvals[j-16])
                    bar()
                    mzsq_Aipph[j-16] = deriv_step_calc_trilin(j,
                                                              (2 * Ai_hvals[j-16]))
                    bar()
                    mzsq_Aippph[j-16] = deriv_step_calc_trilin(j,
                                                               (3 * Ai_hvals[j-16]))
                    bar()
                    mzsq_Aipppph[j-16] = deriv_step_calc_trilin(j,
                                                                (4 * Ai_hvals[j-16]))
                    bar()
                    # Left shifts
                    mzsq_Aimh[j-16] = deriv_step_calc_trilin(j, (-1) * Ai_hvals[j-16])
                    bar()
                    mzsq_Aimmh[j-16] = deriv_step_calc_trilin(j,
                                                              (-1) * (2 * Ai_hvals[j-16]))
                    bar()
                    mzsq_Aimmmh[j-16] = deriv_step_calc_trilin(j,
                                                               (-1) * (3 * Ai_hvals[j-16]))
                    bar()
                    mzsq_Aimmmmh[j-16] = deriv_step_calc_trilin(j,
                                                                (-1) * (4 * Ai_hvals[j-16]))
                    bar()
                    derivcount += 1
                    bar.text = f'Derivatives: {derivcount}/30, please wait...'

                for j in np.arange(3, 6):
                    print("Error estimate for " + str(Mi_labels[j-3]) + " derivative: "
                          + str("{:.4e}".format(float(mp.power(Mi_hvals[j-3], 8)))))
                    # For loop to compute soft gaugino mass derivatives
                    # Right shifts
                    mzsq_Miph[j-3] = deriv_step_calc_gaugino(j, Mi_hvals[j-3])
                    bar()
                    mzsq_Mipph[j-3] = deriv_step_calc_gaugino(j,
                                                              (2 * Mi_hvals[j-3]))
                    bar()
                    mzsq_Mippph[j-3] = deriv_step_calc_gaugino(j,
                                                               (3 * Mi_hvals[j-3]))
                    bar()
                    mzsq_Mipppph[j-3] = deriv_step_calc_gaugino(j,
                                                                (4 * Mi_hvals[j-3]))
                    bar()
                    # Left shifts
                    mzsq_Mimh[j-3] = deriv_step_calc_gaugino(j, (-1) * Mi_hvals[j-3])
                    bar()
                    mzsq_Mimmh[j-3] = deriv_step_calc_gaugino(j,
                                                              (-1) * (2 * Mi_hvals[j-3]))
                    bar()
                    mzsq_Mimmmh[j-3] = deriv_step_calc_gaugino(j,
                                                               (-1) * (3 * Mi_hvals[j-3]))
                    bar()
                    mzsq_Mimmmmh[j-3] = deriv_step_calc_gaugino(j,
                                                                (-1) * (4 * Mi_hvals[j-3]))
                    bar()
                    derivcount += 1
                    bar.text = f'Derivatives: {derivcount}/30, please wait...'

                for j in np.arange(25, 27):
                    print("Error estimate for " + str(Higgs_labels[j-25]) + " derivative: "
                          + str("{:.4e}".format(float(mp.power(Higgs_hvals[j-25], 8)))))
                    # For loop to compute soft Higgs mass derivatives
                    # Right shifts
                    mzsq_mHph[j-25] = deriv_step_calc_Higgs(j, Higgs_hvals[j-25])
                    bar()
                    mzsq_mHpph[j-25] = deriv_step_calc_Higgs(j,
                                                             (2 * Higgs_hvals[j-25]))
                    bar()
                    mzsq_mHppph[j-25] = deriv_step_calc_Higgs(j,
                                                              (3 * Higgs_hvals[j-25]))
                    bar()
                    mzsq_mHpppph[j-25] = deriv_step_calc_Higgs(j,
                                                               (4 * Higgs_hvals[j-25]))
                    bar()
                    # Left shifts
                    mzsq_mHmh[j-25] = deriv_step_calc_Higgs(j, (-1) * Higgs_hvals[j-25])
                    bar()
                    mzsq_mHmmh[j-25] = deriv_step_calc_Higgs(j,
                                                             (-1) * (2 * Higgs_hvals[j-25]))
                    bar()
                    mzsq_mHmmmh[j-25] = deriv_step_calc_Higgs(j,
                                                              (-1) * (3 * Higgs_hvals[j-25]))
                    bar()
                    mzsq_mHmmmmh[j-25] = deriv_step_calc_Higgs(j,
                                                               (-1) * (4 * Higgs_hvals[j-25]))
                    bar()
                    derivcount += 1
                    bar.text = f'Derivatives: {derivcount}/30, please wait...'

                print("Error estimate for mu_0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hmu0, 8)))))
                mzsq_mu0ph = deriv_step_calc_mu0(hmu0)
                bar()
                mzsq_mu0pph = deriv_step_calc_mu0(2*hmu0)
                bar()
                mzsq_mu0ppph = deriv_step_calc_mu0(3*hmu0)
                bar()
                mzsq_mu0pppph = deriv_step_calc_mu0(4*hmu0)
                bar()
                mzsq_mu0mh = deriv_step_calc_mu0((-1) * hmu0)
                bar()
                mzsq_mu0mmh = deriv_step_calc_mu0((-1) * 2*hmu0)
                bar()
                mzsq_mu0mmmh = deriv_step_calc_mu0((-1) * 3*hmu0)
                bar()
                mzsq_mu0mmmmh = deriv_step_calc_mu0((-1) * 4*hmu0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/30, please wait...'

                # Construct derivative array
                deriv_array = np.array([deriv_num_calc(hmqL1, mzsq_msqslmmmmh[0], mzsq_msqslmmmh[0],
                                                       mzsq_msqslmmh[0], mzsq_msqslmh[0], mzsq_msqslph[0],
                                                       mzsq_msqslpph[0], mzsq_msqslppph[0], mzsq_msqslpppph[0]), #end qL1: 0
                                        deriv_num_calc(hmqL2, mzsq_msqslmmmmh[1], mzsq_msqslmmmh[1],
                                                       mzsq_msqslmmh[1], mzsq_msqslmh[1], mzsq_msqslph[1],
                                                       mzsq_msqslpph[1], mzsq_msqslppph[1], mzsq_msqslpppph[1]), #end qL2: 1
                                        deriv_num_calc(hmqL3, mzsq_msqslmmmmh[2], mzsq_msqslmmmh[2],
                                                       mzsq_msqslmmh[2], mzsq_msqslmh[2], mzsq_msqslph[2],
                                                       mzsq_msqslpph[2], mzsq_msqslppph[2], mzsq_msqslpppph[2]), #end qL3: 2
                                        deriv_num_calc(hmeL, mzsq_msqslmmmmh[3], mzsq_msqslmmmh[3],
                                                       mzsq_msqslmmh[3], mzsq_msqslmh[3], mzsq_msqslph[3],
                                                       mzsq_msqslpph[3], mzsq_msqslppph[3], mzsq_msqslpppph[3]), #end eL: 3
                                        deriv_num_calc(hmmuL, mzsq_msqslmmmmh[4], mzsq_msqslmmmh[4],
                                                       mzsq_msqslmmh[4], mzsq_msqslmh[4], mzsq_msqslph[4],
                                                       mzsq_msqslpph[4], mzsq_msqslppph[4], mzsq_msqslpppph[4]), #end muL: 4
                                        deriv_num_calc(hmtauL, mzsq_msqslmmmmh[5], mzsq_msqslmmmh[5],
                                                       mzsq_msqslmmh[5], mzsq_msqslmh[5], mzsq_msqslph[5],
                                                       mzsq_msqslpph[5], mzsq_msqslppph[5], mzsq_msqslpppph[5]), #end tauL: 5
                                        deriv_num_calc(hmuR, mzsq_msqslmmmmh[6], mzsq_msqslmmmh[6],
                                                       mzsq_msqslmmh[6], mzsq_msqslmh[6], mzsq_msqslph[6],
                                                       mzsq_msqslpph[6], mzsq_msqslppph[6], mzsq_msqslpppph[6]), #end uR: 6
                                        deriv_num_calc(hmcR, mzsq_msqslmmmmh[7], mzsq_msqslmmmh[7],
                                                       mzsq_msqslmmh[7], mzsq_msqslmh[7], mzsq_msqslph[7],
                                                       mzsq_msqslpph[7], mzsq_msqslppph[7], mzsq_msqslpppph[7]), #end cR: 7
                                        deriv_num_calc(hmtR, mzsq_msqslmmmmh[8], mzsq_msqslmmmh[8],
                                                       mzsq_msqslmmh[8], mzsq_msqslmh[8], mzsq_msqslph[8],
                                                       mzsq_msqslpph[8], mzsq_msqslppph[8], mzsq_msqslpppph[8]), #end tR: 8
                                        deriv_num_calc(hmdR, mzsq_msqslmmmmh[9], mzsq_msqslmmmh[9],
                                                       mzsq_msqslmmh[9], mzsq_msqslmh[9], mzsq_msqslph[9],
                                                       mzsq_msqslpph[9], mzsq_msqslppph[9], mzsq_msqslpppph[9]), #end dR: 9
                                        deriv_num_calc(hmsR, mzsq_msqslmmmmh[10], mzsq_msqslmmmh[10],
                                                       mzsq_msqslmmh[10], mzsq_msqslmh[10], mzsq_msqslph[10],
                                                       mzsq_msqslpph[10], mzsq_msqslppph[10], mzsq_msqslpppph[10]), #end sR: 10
                                        deriv_num_calc(hmbR, mzsq_msqslmmmmh[11], mzsq_msqslmmmh[11],
                                                       mzsq_msqslmmh[11], mzsq_msqslmh[11], mzsq_msqslph[11],
                                                       mzsq_msqslpph[11], mzsq_msqslppph[11], mzsq_msqslpppph[11]), #end bR: 11
                                        deriv_num_calc(hmeR, mzsq_msqslmmmmh[12], mzsq_msqslmmmh[12],
                                                       mzsq_msqslmmh[12], mzsq_msqslmh[12], mzsq_msqslph[12],
                                                       mzsq_msqslpph[12], mzsq_msqslppph[12], mzsq_msqslpppph[12]), #end eR: 12
                                        deriv_num_calc(hmmuR, mzsq_msqslmmmmh[13], mzsq_msqslmmmh[13],
                                                       mzsq_msqslmmh[13], mzsq_msqslmh[13], mzsq_msqslph[13],
                                                       mzsq_msqslpph[13], mzsq_msqslppph[13], mzsq_msqslpppph[13]), #end muR: 13
                                        deriv_num_calc(hmtauR, mzsq_msqslmmmmh[14], mzsq_msqslmmmh[14],
                                                       mzsq_msqslmmh[14], mzsq_msqslmh[14], mzsq_msqslph[14],
                                                       mzsq_msqslpph[14], mzsq_msqslppph[14], mzsq_msqslpppph[0]), #end tauR: 14
                                        deriv_num_calc(hAt, mzsq_Aimmmmh[0], mzsq_Aimmmh[0],
                                                       mzsq_Aimmh[0], mzsq_Aimh[0], mzsq_Aiph[0],
                                                       mzsq_Aipph[0], mzsq_Aippph[0], mzsq_Aipppph[0]), #end At: 15
                                        deriv_num_calc(hAc, mzsq_Aimmmmh[1], mzsq_Aimmmh[1],
                                                       mzsq_Aimmh[1], mzsq_Aimh[1], mzsq_Aiph[1],
                                                       mzsq_Aipph[1], mzsq_Aippph[1], mzsq_Aipppph[1]), #end Ac: 16
                                        deriv_num_calc(hAu, mzsq_Aimmmmh[2], mzsq_Aimmmh[2],
                                                       mzsq_Aimmh[2], mzsq_Aimh[2], mzsq_Aiph[2],
                                                       mzsq_Aipph[2], mzsq_Aippph[2], mzsq_Aipppph[2]), #end Au: 17
                                        deriv_num_calc(hAb, mzsq_Aimmmmh[3], mzsq_Aimmmh[3],
                                                       mzsq_Aimmh[3], mzsq_Aimh[3], mzsq_Aiph[3],
                                                       mzsq_Aipph[3], mzsq_Aippph[3], mzsq_Aipppph[3]), #end Ab: 18
                                        deriv_num_calc(hAs, mzsq_Aimmmmh[4], mzsq_Aimmmh[4],
                                                       mzsq_Aimmh[4], mzsq_Aimh[4], mzsq_Aiph[4],
                                                       mzsq_Aipph[4], mzsq_Aippph[4], mzsq_Aipppph[4]), #end As: 19
                                        deriv_num_calc(hAd, mzsq_Aimmmmh[5], mzsq_Aimmmh[5],
                                                       mzsq_Aimmh[5], mzsq_Aimh[5], mzsq_Aiph[5],
                                                       mzsq_Aipph[5], mzsq_Aippph[5], mzsq_Aipppph[5]), #end Ad: 20
                                        deriv_num_calc(hAtau, mzsq_Aimmmmh[6], mzsq_Aimmmh[6],
                                                       mzsq_Aimmh[6], mzsq_Aimh[6], mzsq_Aiph[6],
                                                       mzsq_Aipph[6], mzsq_Aippph[6], mzsq_Aipppph[6]), #end Atau: 21
                                        deriv_num_calc(hAmu, mzsq_Aimmmmh[7], mzsq_Aimmmh[7],
                                                       mzsq_Aimmh[7], mzsq_Aimh[7], mzsq_Aiph[7],
                                                       mzsq_Aipph[7], mzsq_Aippph[7], mzsq_Aipppph[7]), #end Amu: 22
                                        deriv_num_calc(hAe, mzsq_Aimmmmh[8], mzsq_Aimmmh[8],
                                                       mzsq_Aimmh[8], mzsq_Aimh[8], mzsq_Aiph[8],
                                                       mzsq_Aipph[8], mzsq_Aippph[8], mzsq_Aipppph[8]), #end Ae: 23
                                        deriv_num_calc(hM1, mzsq_Mimmmmh[0], mzsq_Mimmmh[0],
                                                       mzsq_Mimmh[0], mzsq_Mimh[0], mzsq_Miph[0],
                                                       mzsq_Mipph[0], mzsq_Mippph[0], mzsq_Mipppph[0]), #end M1: 24
                                        deriv_num_calc(hM2, mzsq_Mimmmmh[1], mzsq_Mimmmh[1],
                                                       mzsq_Mimmh[1], mzsq_Mimh[1], mzsq_Miph[1],
                                                       mzsq_Mipph[1], mzsq_Mippph[1], mzsq_Mipppph[1]), #end M2: 25
                                        deriv_num_calc(hM3, mzsq_Mimmmmh[2], mzsq_Mimmmh[2],
                                                       mzsq_Mimmh[2], mzsq_Mimh[2], mzsq_Miph[2],
                                                       mzsq_Mipph[2], mzsq_Mippph[2], mzsq_Mipppph[2]), #end M3: 26
                                        deriv_num_calc(hmHu0, mzsq_mHmmmmh[0], mzsq_mHmmmh[0],
                                                       mzsq_mHmmh[0], mzsq_mHmh[0], mzsq_mHph[0],
                                                       mzsq_mHpph[0], mzsq_mHppph[0], mzsq_mHpppph[0]), #end mHu0: 27
                                        deriv_num_calc(hmHd0, mzsq_mHmmmmh[1], mzsq_mHmmmh[1],
                                                       mzsq_mHmmh[1], mzsq_mHmh[1], mzsq_mHph[1],
                                                       mzsq_mHpph[1], mzsq_mHppph[1], mzsq_mHpppph[1]), #end mHd0: 28
                                        deriv_num_calc(hmu0, mzsq_mu0mmmmh, mzsq_mu0mmmh,
                                                       mzsq_mu0mmh, mzsq_mu0mh, mzsq_mu0ph,
                                                       mzsq_mu0pph, mzsq_mu0ppph, mzsq_mu0pppph)]) #end mu0: 29
                sens_params = np.sort(np.array([(mp.fabs((mymqL1
                                                          / mymzsq)
                                                         * deriv_array[0]),
                                                 'Delta_BG(m_qL(1))'),
                                                (mp.fabs((mymqL2
                                                          / mymzsq)
                                                         * deriv_array[1]),
                                                 'Delta_BG(m_qL(2))'),
                                                (mp.fabs((mymqL3
                                                          / mymzsq)
                                                         * deriv_array[2]),
                                                 'Delta_BG(m_qL(3))'),
                                                (mp.fabs((mymeL
                                                          / mymzsq)
                                                         * deriv_array[3]),
                                                 'Delta_BG(m_eL)'),
                                                (mp.fabs((mymmuL
                                                          / mymzsq)
                                                         * deriv_array[4]),
                                                 'Delta_BG(m_muL)'),
                                                (mp.fabs((mymtauL
                                                          / mymzsq)
                                                         * deriv_array[5]),
                                                 'Delta_BG(m_tauL)'),
                                                (mp.fabs((mymuR
                                                          / mymzsq)
                                                         * deriv_array[6]),
                                                 'Delta_BG(m_uR)'),
                                                (mp.fabs((mymcR
                                                          / mymzsq)
                                                         * deriv_array[7]),
                                                 'Delta_BG(m_cR)'),
                                                (mp.fabs((mymtR
                                                          / mymzsq)
                                                         * deriv_array[8]),
                                                 'Delta_BG(m_tR)'),
                                                (mp.fabs((mymdR
                                                          / mymzsq)
                                                         * deriv_array[9]),
                                                 'Delta_BG(m_dR)'),
                                                (mp.fabs((mymsR
                                                          / mymzsq)
                                                         * deriv_array[10]),
                                                 'Delta_BG(m_sR)'),
                                                (mp.fabs((mymbR
                                                          / mymzsq)
                                                         * deriv_array[11]),
                                                 'Delta_BG(m_bR)'),
                                                (mp.fabs((mymeR
                                                          / mymzsq)
                                                         * deriv_array[12]),
                                                 'Delta_BG(m_eR)'),
                                                (mp.fabs((mymmuR
                                                          / mymzsq)
                                                         * deriv_array[13]),
                                                 'Delta_BG(m_muR)'),
                                                (mp.fabs((mymtauR
                                                          / mymzsq)
                                                         * deriv_array[14]),
                                                 'Delta_BG(m_tauR)'),
                                                (mp.fabs((myAt
                                                          / mymzsq)
                                                         * deriv_array[15]),
                                                 'Delta_BG(a_t)'),
                                                (mp.fabs((myAc
                                                          / mymzsq)
                                                         * deriv_array[16]),
                                                 'Delta_BG(a_c)'),
                                                (mp.fabs((myAu
                                                          / mymzsq)
                                                         * deriv_array[17]),
                                                 'Delta_BG(a_u)'),
                                                (mp.fabs((myAb
                                                          / mymzsq)
                                                         * deriv_array[18]),
                                                 'Delta_BG(a_b)'),
                                                (mp.fabs((myAs
                                                          / mymzsq)
                                                         * deriv_array[19]),
                                                 'Delta_BG(a_s)'),
                                                (mp.fabs((myAd
                                                          / mymzsq)
                                                         * deriv_array[20]),
                                                 'Delta_BG(a_d)'),
                                                (mp.fabs((myAtau
                                                          / mymzsq)
                                                         * deriv_array[21]),
                                                 'Delta_BG(a_tau)'),
                                                (mp.fabs((myAmu
                                                          / mymzsq)
                                                         * deriv_array[22]),
                                                 'Delta_BG(a_mu)'),
                                                (mp.fabs((myAe
                                                          / mymzsq)
                                                         * deriv_array[23]),
                                                 'Delta_BG(a_e)'),
                                                (mp.fabs((myM1
                                                          / mymzsq)
                                                         * deriv_array[24]),
                                                 'Delta_BG(M_1)'),
                                                (mp.fabs((myM2
                                                          / mymzsq)
                                                         * deriv_array[25]),
                                                 'Delta_BG(M_2)'),
                                                (mp.fabs((myM3
                                                          / mymzsq)
                                                         * deriv_array[26]),
                                                 'Delta_BG(M_3)'),
                                                (mp.fabs((mymHu0
                                                          / mymzsq)
                                                         * deriv_array[27]),
                                                 'Delta_BG(m_Hu0)'),
                                                (mp.fabs((mymHd0
                                                          / mymzsq)
                                                         * deriv_array[28]),
                                                 'Delta_BG(m_Hd0)'),
                                                (mp.fabs((mymu0
                                                          / mymzsq)
                                                         * deriv_array[29]),
                                                 'Delta_BG(mu0)')],
                                               dtype=[('BGcontrib',float),
                                                      ('BGlabel','U40')]),
                                      order='BGcontrib')
                print("Done!")

        elif (precselno == 2):
            derivcount = 0
            mzsq_msqslph = np.zeros(15)
            mzsq_msqslpph = np.zeros(15)
            mzsq_msqslmh = np.zeros(15)
            mzsq_msqslmmh = np.zeros(15)
            mzsq_Aiph = np.zeros(9)
            mzsq_Aipph = np.zeros(9)
            mzsq_Aimh = np.zeros(9)
            mzsq_Aimmh = np.zeros(9)
            mzsq_Miph = np.zeros(3)
            mzsq_Mipph = np.zeros(3)
            mzsq_Mimh = np.zeros(3)
            mzsq_Mimmh = np.zeros(3)
            mzsq_mHph = np.zeros(2)
            mzsq_mHpph = np.zeros(2)
            mzsq_mHmh = np.zeros(2)
            mzsq_mHmmh = np.zeros(2)
            hmqL1 = mpf(str(abs(mp.power(math.ulp(mymqL1), 1/5))))
            hmqL2 = mpf(str(abs(mp.power(math.ulp(mymqL2), 1/5))))
            hmqL3 = mpf(str(abs(mp.power(math.ulp(mymqL3), 1/5))))
            hmuR = mpf(str(abs(mp.power(math.ulp(mymuR), 1/5))))
            hmcR = mpf(str(abs(mp.power(math.ulp(mymcR), 1/5))))
            hmtR = mpf(str(abs(mp.power(math.ulp(mymtR), 1/5))))
            hmdR = mpf(str(abs(mp.power(math.ulp(mymdR), 1/5))))
            hmsR = mpf(str(abs(mp.power(math.ulp(mymsR), 1/5))))
            hmbR = mpf(str(abs(mp.power(math.ulp(mymbR), 1/5))))
            hmeL = mpf(str(abs(mp.power(math.ulp(mymeL), 1/5))))
            hmmuL = mpf(str(abs(mp.power(math.ulp(mymmuL), 1/5))))
            hmtauL = mpf(str(abs(mp.power(math.ulp(mymtauL), 1/5))))
            hmeR = mpf(str(abs(mp.power(math.ulp(mymeR), 1/5))))
            hmmuR = mpf(str(abs(mp.power(math.ulp(mymmuR), 1/5))))
            hmtauR = mpf(str(abs(mp.power(math.ulp(mymtauR), 1/5))))
            sq_sl_hvals = [hmqL1, hmqL2, hmqL3, hmeL, hmmuL, hmtauL,
                           hmuR, hmcR, hmtR, hmdR, hmsR, hmbR, hmeR, hmmuR,
                           hmtauR]
            sq_sl_labels = ['m_qL(1)','m_qL(2)','m_qL(3)','m_eL','m_muL',
                            'm_tauL','m_uR','m_cR','m_tR','m_dR',
                            'm_sR','m_bR','m_eR','m_muR','m_tauR']
            hM1 = mpf(str(abs(mp.power(math.ulp(myM1), 1/5))))
            hM2 = mpf(str(abs(mp.power(math.ulp(myM2), 1/5))))
            hM3 = mpf(str(abs(mp.power(math.ulp(myM3), 1/5))))
            Mi_hvals = [hM1, hM2, hM3]
            Mi_labels = ['M_1', 'M_2', 'M_3']
            hAt = mpf(str(abs(mp.power(math.ulp(myAt), 1/5))))
            hAc = mpf(str(abs(mp.power(math.ulp(myAc), 1/5))))
            hAu = mpf(str(abs(mp.power(math.ulp(myAu), 1/5))))
            hAb = mpf(str(abs(mp.power(math.ulp(myAb), 1/5))))
            hAs = mpf(str(abs(mp.power(math.ulp(myAs), 1/5))))
            hAd = mpf(str(abs(mp.power(math.ulp(myAd), 1/5))))
            hAtau = mpf(str(abs(mp.power(math.ulp(myAtau), 1/5))))
            hAmu = mpf(str(abs(mp.power(math.ulp(myAmu), 1/5))))
            hAe = mpf(str(abs(mp.power(math.ulp(myAe), 1/5))))
            Ai_hvals = [hAt, hAc, hAu, hAb, hAs, hAd, hAtau, hAmu, hAe]
            Ai_labels = ['a_t', 'a_c', 'a_u','a_b', 'a_s', 'a_d','a_tau', 'a_mu', 'a_e']
            hmu0 = mpf(str(abs(mp.power(math.ulp(mymu0), 1/5))))
            hmHu0 = mpf(str(abs(mp.power(math.ulp(mymHu0), 1/5))))
            hmHd0 = mpf(str(abs(mp.power(math.ulp(mymHd0), 1/5))))
            Higgs_hvals = [hmHu0, hmHd0]
            Higgs_labels = ['m_Hu0', 'm_Hd0']

            with alive_bar(120, dual_line=True,
                           force_tty=True, title='Computations: ',
                           bar='brackets') as bar:
                bar.text = f'Derivatives: {derivcount}/30, please wait...'
                for j in np.arange(27, 42):
                    print("Error estimate for " + str(sq_sl_labels[j-27]) + " derivative: "
                          + str("{:.4e}".format(float(mp.power(sq_sl_hvals[j-27], 4)))))
                    # For loop to compute squark and slepton mass derivatives
                    # Right shifts
                    mzsq_msqslph[j-27] = deriv_step_calc_sq_sl(j, sq_sl_hvals[j-27])
                    bar()
                    mzsq_msqslpph[j-27] = deriv_step_calc_sq_sl(j,
                                                                (2 * sq_sl_hvals[j-27]))
                    bar()
                    # Left shifts
                    mzsq_msqslmh[j-27] = deriv_step_calc_sq_sl(j, (-1) * sq_sl_hvals[j-27])
                    bar()
                    mzsq_msqslmmh[j-27] = deriv_step_calc_sq_sl(j,
                                                                (-1) * (2 * sq_sl_hvals[j-27]))
                    bar()
                    derivcount += 1
                    bar.text = f'Derivatives: {derivcount}/30, please wait...'

                for j in np.arange(16, 25):
                    print("Error estimate for " + str(Ai_labels[j-16]) + " derivative: "
                          + str("{:.4e}".format(float(mp.power(Ai_hvals[j-16], 4)))))
                    # For loop to compute soft trilinear coupling derivatives
                    # Right shifts
                    mzsq_Aiph[j-16] = deriv_step_calc_trilin(j, Ai_hvals[j-16])
                    bar()
                    mzsq_Aipph[j-16] = deriv_step_calc_trilin(j,
                                                              (2 * Ai_hvals[j-16]))
                    bar()
                    # Left shifts
                    mzsq_Aimh[j-16] = deriv_step_calc_trilin(j, (-1) * Ai_hvals[j-16])
                    bar()
                    mzsq_Aimmh[j-16] = deriv_step_calc_trilin(j,
                                                              (-1) * (2 * Ai_hvals[j-16]))
                    bar()
                    derivcount += 1
                    bar.text = f'Derivatives: {derivcount}/30, please wait...'

                for j in np.arange(3, 6):
                    print("Error estimate for " + str(Mi_labels[j-3]) + " derivative: "
                          + str("{:.4e}".format(float(mp.power(Mi_hvals[j-3], 4)))))
                    # For loop to compute soft gaugino mass derivatives
                    # Right shifts
                    mzsq_Miph[j-3] = deriv_step_calc_gaugino(j, Mi_hvals[j-3])
                    bar()
                    mzsq_Mipph[j-3] = deriv_step_calc_gaugino(j,
                                                              (2 * Mi_hvals[j-3]))
                    bar()
                    # Left shifts
                    mzsq_Mimh[j-3] = deriv_step_calc_gaugino(j, (-1) * Mi_hvals[j-3])
                    bar()
                    mzsq_Mimmh[j-3] = deriv_step_calc_gaugino(j,
                                                              (-1) * (2 * Mi_hvals[j-3]))
                    bar()
                    derivcount += 1
                    bar.text = f'Derivatives: {derivcount}/30, please wait...'

                for j in np.arange(25, 27):
                    print("Error estimate for " + str(Higgs_labels[j-25]) + " derivative: "
                          + str("{:.4e}".format(float(mp.power(Higgs_hvals[j-25], 4)))))
                    # For loop to compute soft Higgs mass derivatives
                    # Right shifts
                    mzsq_mHph[j-25] = deriv_step_calc_Higgs(j, Higgs_hvals[j-25])
                    bar()
                    mzsq_mHpph[j-25] = deriv_step_calc_Higgs(j,
                                                             (2 * Higgs_hvals[j-25]))
                    bar()
                    # Left shifts
                    mzsq_mHmh[j-25] = deriv_step_calc_Higgs(j, (-1) * Higgs_hvals[j-25])
                    bar()
                    mzsq_mHmmh[j-25] = deriv_step_calc_Higgs(j,
                                                             (-1) * (2 * Higgs_hvals[j-25]))
                    bar()
                    derivcount += 1
                    bar.text = f'Derivatives: {derivcount}/30, please wait...'

                print("Error estimate for mu_0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hmu0, 4)))))
                mzsq_mu0ph = deriv_step_calc_mu0(hmu0)
                bar()
                mzsq_mu0pph = deriv_step_calc_mu0(2*hmu0)
                bar()
                mzsq_mu0mh = deriv_step_calc_mu0((-1) * hmu0)
                bar()
                mzsq_mu0mmh = deriv_step_calc_mu0((-1) * 2*hmu0)
                bar()
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/30, please wait...'

                # Construct derivative array
                deriv_array = np.array([deriv_num_calc(hmqL1,
                                                       mzsq_msqslmmh[0], mzsq_msqslmh[0], mzsq_msqslph[0],
                                                       mzsq_msqslpph[0]), #end qL1: 0
                                        deriv_num_calc(hmqL2,
                                                       mzsq_msqslmmh[1], mzsq_msqslmh[1], mzsq_msqslph[1],
                                                       mzsq_msqslpph[1]), #end qL2: 1
                                        deriv_num_calc(hmqL3,
                                                       mzsq_msqslmmh[2], mzsq_msqslmh[2], mzsq_msqslph[2],
                                                       mzsq_msqslpph[2]), #end qL3: 2
                                        deriv_num_calc(hmeL,
                                                       mzsq_msqslmmh[3], mzsq_msqslmh[3], mzsq_msqslph[3],
                                                       mzsq_msqslpph[3]), #end eL: 3
                                        deriv_num_calc(hmmuL,
                                                       mzsq_msqslmmh[4], mzsq_msqslmh[4], mzsq_msqslph[4],
                                                       mzsq_msqslpph[4]), #end muL: 4
                                        deriv_num_calc(hmtauL,
                                                       mzsq_msqslmmh[5], mzsq_msqslmh[5], mzsq_msqslph[5],
                                                       mzsq_msqslpph[5]), #end tauL: 5
                                        deriv_num_calc(hmuR,
                                                       mzsq_msqslmmh[6], mzsq_msqslmh[6], mzsq_msqslph[6],
                                                       mzsq_msqslpph[6]), #end uR: 6
                                        deriv_num_calc(hmcR,
                                                       mzsq_msqslmmh[7], mzsq_msqslmh[7], mzsq_msqslph[7],
                                                       mzsq_msqslpph[7]), #end cR: 7
                                        deriv_num_calc(hmtR,
                                                       mzsq_msqslmmh[8], mzsq_msqslmh[8], mzsq_msqslph[8],
                                                       mzsq_msqslpph[8]), #end tR: 8
                                        deriv_num_calc(hmdR,
                                                       mzsq_msqslmmh[9], mzsq_msqslmh[9], mzsq_msqslph[9],
                                                       mzsq_msqslpph[9]), #end dR: 9
                                        deriv_num_calc(hmsR,
                                                       mzsq_msqslmmh[10], mzsq_msqslmh[10], mzsq_msqslph[10],
                                                       mzsq_msqslpph[10]), #end sR: 10
                                        deriv_num_calc(hmbR,
                                                       mzsq_msqslmmh[11], mzsq_msqslmh[11], mzsq_msqslph[11],
                                                       mzsq_msqslpph[11]), #end bR: 11
                                        deriv_num_calc(hmeR,
                                                       mzsq_msqslmmh[12], mzsq_msqslmh[12], mzsq_msqslph[12],
                                                       mzsq_msqslpph[12]), #end eR: 12
                                        deriv_num_calc(hmmuR,
                                                       mzsq_msqslmmh[13], mzsq_msqslmh[13], mzsq_msqslph[13],
                                                       mzsq_msqslpph[13]), #end muR: 13
                                        deriv_num_calc(hmtauR,
                                                       mzsq_msqslmmh[14], mzsq_msqslmh[14], mzsq_msqslph[14],
                                                       mzsq_msqslpph[14]), #end tauR: 14
                                        deriv_num_calc(hAt,
                                                       mzsq_Aimmh[0], mzsq_Aimh[0], mzsq_Aiph[0],
                                                       mzsq_Aipph[0]), #end At: 15
                                        deriv_num_calc(hAc,
                                                       mzsq_Aimmh[1], mzsq_Aimh[1], mzsq_Aiph[1],
                                                       mzsq_Aipph[1]), #end Ac: 16
                                        deriv_num_calc(hAu,
                                                       mzsq_Aimmh[2], mzsq_Aimh[2], mzsq_Aiph[2],
                                                       mzsq_Aipph[2]), #end Au: 17
                                        deriv_num_calc(hAb,
                                                       mzsq_Aimmh[3], mzsq_Aimh[3], mzsq_Aiph[3],
                                                       mzsq_Aipph[3]), #end Ab: 18
                                        deriv_num_calc(hAs,
                                                       mzsq_Aimmh[4], mzsq_Aimh[4], mzsq_Aiph[4],
                                                       mzsq_Aipph[4]), #end As: 19
                                        deriv_num_calc(hAd,
                                                       mzsq_Aimmh[5], mzsq_Aimh[5], mzsq_Aiph[5],
                                                       mzsq_Aipph[5]), #end Ad: 20
                                        deriv_num_calc(hAtau,
                                                       mzsq_Aimmh[6], mzsq_Aimh[6], mzsq_Aiph[6],
                                                       mzsq_Aipph[6]), #end Atau: 21
                                        deriv_num_calc(hAmu,
                                                       mzsq_Aimmh[7], mzsq_Aimh[7], mzsq_Aiph[7],
                                                       mzsq_Aipph[7]), #end Amu: 22
                                        deriv_num_calc(hAe,
                                                       mzsq_Aimmh[8], mzsq_Aimh[8], mzsq_Aiph[8],
                                                       mzsq_Aipph[8]), #end Ae: 23
                                        deriv_num_calc(hM1,
                                                       mzsq_Mimmh[0], mzsq_Mimh[0], mzsq_Miph[0],
                                                       mzsq_Mipph[0]), #end M1: 24
                                        deriv_num_calc(hM2,
                                                       mzsq_Mimmh[1], mzsq_Mimh[1], mzsq_Miph[1],
                                                       mzsq_Mipph[1]), #end M2: 25
                                        deriv_num_calc(hM3,
                                                       mzsq_Mimmh[2], mzsq_Mimh[2], mzsq_Miph[2],
                                                       mzsq_Mipph[2]), #end M3: 26
                                        deriv_num_calc(hmHu0,
                                                       mzsq_mHmmh[0], mzsq_mHmh[0], mzsq_mHph[0],
                                                       mzsq_mHpph[0]), #end mHu0: 27
                                        deriv_num_calc(hmHd0,
                                                       mzsq_mHmmh[1], mzsq_mHmh[1], mzsq_mHph[1],
                                                       mzsq_mHpph[1]), #end mHd0: 28
                                        deriv_num_calc(hmu0,
                                                       mzsq_mu0mmh, mzsq_mu0mh, mzsq_mu0ph,
                                                       mzsq_mu0pph)]) #end mu0: 29
                sens_params = np.sort(np.array([(mp.fabs((mymqL1
                                                          / mymzsq)
                                                         * deriv_array[0]),
                                                 'Delta_BG(m_qL(1))'),
                                                (mp.fabs((mymqL2
                                                          / mymzsq)
                                                         * deriv_array[1]),
                                                 'Delta_BG(m_qL(2))'),
                                                (mp.fabs((mymqL3
                                                          / mymzsq)
                                                         * deriv_array[2]),
                                                 'Delta_BG(m_qL(3))'),
                                                (mp.fabs((mymeL
                                                          / mymzsq)
                                                         * deriv_array[3]),
                                                 'Delta_BG(m_eL)'),
                                                (mp.fabs((mymmuL
                                                          / mymzsq)
                                                         * deriv_array[4]),
                                                 'Delta_BG(m_muL)'),
                                                (mp.fabs((mymtauL
                                                          / mymzsq)
                                                         * deriv_array[5]),
                                                 'Delta_BG(m_tauL)'),
                                                (mp.fabs((mymuR
                                                          / mymzsq)
                                                         * deriv_array[6]),
                                                 'Delta_BG(m_uR)'),
                                                (mp.fabs((mymcR
                                                          / mymzsq)
                                                         * deriv_array[7]),
                                                 'Delta_BG(m_cR)'),
                                                (mp.fabs((mymtR
                                                          / mymzsq)
                                                         * deriv_array[8]),
                                                 'Delta_BG(m_tR)'),
                                                (mp.fabs((mymdR
                                                          / mymzsq)
                                                         * deriv_array[9]),
                                                 'Delta_BG(m_dR)'),
                                                (mp.fabs((mymsR
                                                          / mymzsq)
                                                         * deriv_array[10]),
                                                 'Delta_BG(m_sR)'),
                                                (mp.fabs((mymbR
                                                          / mymzsq)
                                                         * deriv_array[11]),
                                                 'Delta_BG(m_bR)'),
                                                (mp.fabs((mymeR
                                                          / mymzsq)
                                                         * deriv_array[12]),
                                                 'Delta_BG(m_eR)'),
                                                (mp.fabs((mymmuR
                                                          / mymzsq)
                                                         * deriv_array[13]),
                                                 'Delta_BG(m_muR)'),
                                                (mp.fabs((mymtauR
                                                          / mymzsq)
                                                         * deriv_array[14]),
                                                 'Delta_BG(m_tauR)'),
                                                (mp.fabs((myAt
                                                          / mymzsq)
                                                         * deriv_array[15]),
                                                 'Delta_BG(a_t)'),
                                                (mp.fabs((myAc
                                                          / mymzsq)
                                                         * deriv_array[16]),
                                                 'Delta_BG(a_c)'),
                                                (mp.fabs((myAu
                                                          / mymzsq)
                                                         * deriv_array[17]),
                                                 'Delta_BG(a_u)'),
                                                (mp.fabs((myAb
                                                          / mymzsq)
                                                         * deriv_array[18]),
                                                 'Delta_BG(a_b)'),
                                                (mp.fabs((myAs
                                                          / mymzsq)
                                                         * deriv_array[19]),
                                                 'Delta_BG(a_s)'),
                                                (mp.fabs((myAd
                                                          / mymzsq)
                                                         * deriv_array[20]),
                                                 'Delta_BG(a_d)'),
                                                (mp.fabs((myAtau
                                                          / mymzsq)
                                                         * deriv_array[21]),
                                                 'Delta_BG(a_tau)'),
                                                (mp.fabs((myAmu
                                                          / mymzsq)
                                                         * deriv_array[22]),
                                                 'Delta_BG(a_mu)'),
                                                (mp.fabs((myAe
                                                          / mymzsq)
                                                         * deriv_array[23]),
                                                 'Delta_BG(a_e)'),
                                                (mp.fabs((myM1
                                                          / mymzsq)
                                                         * deriv_array[24]),
                                                 'Delta_BG(M_1)'),
                                                (mp.fabs((myM2
                                                          / mymzsq)
                                                         * deriv_array[25]),
                                                 'Delta_BG(M_2)'),
                                                (mp.fabs((myM3
                                                          / mymzsq)
                                                         * deriv_array[26]),
                                                 'Delta_BG(M_3)'),
                                                (mp.fabs((mymHu0
                                                          / mymzsq)
                                                         * deriv_array[27]),
                                                 'Delta_BG(m_Hu0)'),
                                                (mp.fabs((mymHd0
                                                          / mymzsq)
                                                         * deriv_array[28]),
                                                 'Delta_BG(m_Hd0)'),
                                                (mp.fabs((mymu0
                                                          / mymzsq)
                                                         * deriv_array[29]),
                                                 'Delta_BG(mu0)')],
                                               dtype=[('BGcontrib',float),
                                                      ('BGlabel','U40')]),
                                      order='BGcontrib')
                print("Done!")

        elif (precselno == 3):
            derivcount = 0
            mzsq_msqslph = np.zeros(15)
            mzsq_msqslpph = np.zeros(15)
            mzsq_msqslmh = np.zeros(15)
            mzsq_msqslmmh = np.zeros(15)
            mzsq_Aiph = np.zeros(9)
            mzsq_Aipph = np.zeros(9)
            mzsq_Aimh = np.zeros(9)
            mzsq_Aimmh = np.zeros(9)
            mzsq_Miph = np.zeros(3)
            mzsq_Mipph = np.zeros(3)
            mzsq_Mimh = np.zeros(3)
            mzsq_Mimmh = np.zeros(3)
            mzsq_mHph = np.zeros(2)
            mzsq_mHpph = np.zeros(2)
            mzsq_mHmh = np.zeros(2)
            mzsq_mHmmh = np.zeros(2)
            hmqL1 = mpf(str(abs(mp.power(math.ulp(mymqL1), 1/3))))
            hmqL2 = mpf(str(abs(mp.power(math.ulp(mymqL2), 1/3))))
            hmqL3 = mpf(str(abs(mp.power(math.ulp(mymqL3), 1/3))))
            hmuR = mpf(str(abs(mp.power(math.ulp(mymuR), 1/3))))
            hmcR = mpf(str(abs(mp.power(math.ulp(mymcR), 1/3))))
            hmtR = mpf(str(abs(mp.power(math.ulp(mymtR), 1/3))))
            hmdR = mpf(str(abs(mp.power(math.ulp(mymdR), 1/3))))
            hmsR = mpf(str(abs(mp.power(math.ulp(mymsR), 1/3))))
            hmbR = mpf(str(abs(mp.power(math.ulp(mymbR), 1/3))))
            hmeL = mpf(str(abs(mp.power(math.ulp(mymeL), 1/3))))
            hmmuL = mpf(str(abs(mp.power(math.ulp(mymmuL), 1/3))))
            hmtauL = mpf(str(abs(mp.power(math.ulp(mymtauL), 1/3))))
            hmeR = mpf(str(abs(mp.power(math.ulp(mymeR), 1/3))))
            hmmuR = mpf(str(abs(mp.power(math.ulp(mymmuR), 1/3))))
            hmtauR = mpf(str(abs(mp.power(math.ulp(mymtauR), 1/3))))
            sq_sl_hvals = [hmqL1, hmqL2, hmqL3, hmeL, hmmuL, hmtauL,
                           hmuR, hmcR, hmtR, hmdR, hmsR, hmbR, hmeR, hmmuR,
                           hmtauR]
            sq_sl_labels = ['m_qL(1)','m_qL(2)','m_qL(3)','m_eL','m_muL',
                            'm_tauL','m_uR','m_cR','m_tR','m_dR',
                            'm_sR','m_bR','m_eR','m_muR','m_tauR']
            hM1 = mpf(str(abs(mp.power(math.ulp(myM1), 1/3))))
            hM2 = mpf(str(abs(mp.power(math.ulp(myM2), 1/3))))
            hM3 = mpf(str(abs(mp.power(math.ulp(myM3), 1/3))))
            Mi_hvals = [hM1, hM2, hM3]
            Mi_labels = ['M_1', 'M_2', 'M_3']
            hAt = mpf(str(abs(mp.power(math.ulp(myAt), 1/3))))
            hAc = mpf(str(abs(mp.power(math.ulp(myAc), 1/3))))
            hAu = mpf(str(abs(mp.power(math.ulp(myAu), 1/3))))
            hAb = mpf(str(abs(mp.power(math.ulp(myAb), 1/3))))
            hAs = mpf(str(abs(mp.power(math.ulp(myAs), 1/3))))
            hAd = mpf(str(abs(mp.power(math.ulp(myAd), 1/3))))
            hAtau = mpf(str(abs(mp.power(math.ulp(myAtau), 1/3))))
            hAmu = mpf(str(abs(mp.power(math.ulp(myAmu), 1/3))))
            hAe = mpf(str(abs(mp.power(math.ulp(myAe), 1/3))))
            Ai_hvals = [hAt, hAc, hAu, hAb, hAs, hAd, hAtau, hAmu, hAe]
            Ai_labels = ['a_t', 'a_c', 'a_u','a_b', 'a_s', 'a_d','a_tau', 'a_mu', 'a_e']
            hmu0 = mpf(str(abs(mp.power(math.ulp(mymu0), 1/3))))
            hmHu0 = mpf(str(abs(mp.power(math.ulp(mymHu0), 1/3))))
            hmHd0 = mpf(str(abs(mp.power(math.ulp(mymHd0), 1/3))))
            Higgs_hvals = [hmHu0, hmHd0]
            Higgs_labels = ['m_Hu0', 'm_Hd0']

            with alive_bar(60, dual_line=True,
                           force_tty=True, title='Computations: ',
                           bar='brackets') as bar:
                bar.text = f'Derivatives: {derivcount}/30, please wait...'
                for j in np.arange(27, 42):
                    print("Error estimate for " + str(sq_sl_labels[j-27]) + " derivative: "
                          + str("{:.4e}".format(float(mp.power(sq_sl_hvals[j-27], 2)))))
                    # For loop to compute squark and slepton mass derivatives
                    # Right shifts
                    mzsq_msqslph[j-27] = deriv_step_calc_sq_sl(j, sq_sl_hvals[j-27])
                    bar()
                    mzsq_msqslpph[j-27] = 0
                    # Left shifts
                    mzsq_msqslmh[j-27] = deriv_step_calc_sq_sl(j, (-1) * sq_sl_hvals[j-27])
                    bar()
                    mzsq_msqslmmh[j-27] = 0
                    derivcount += 1
                    bar.text = f'Derivatives: {derivcount}/30, please wait...'

                for j in np.arange(16, 25):
                    print("Error estimate for " + str(Ai_labels[j-16]) + " derivative: "
                          + str("{:.4e}".format(float(mp.power(Ai_hvals[j-16],2)))))
                    # For loop to compute soft trilinear coupling derivatives
                    # Right shifts
                    mzsq_Aiph[j-16] = deriv_step_calc_trilin(j, Ai_hvals[j-16])
                    bar()
                    mzsq_Aipph[j-16] = 0
                    # Left shifts
                    mzsq_Aimh[j-16] = deriv_step_calc_trilin(j, (-1) * Ai_hvals[j-16])
                    bar()
                    mzsq_Aimmh[j-16] = 0
                    derivcount += 1
                    bar.text = f'Derivatives: {derivcount}/30, please wait...'

                for j in np.arange(3, 6):
                    print("Error estimate for " + str(Mi_labels[j-3]) + " derivative: "
                          + str("{:.4e}".format(float(mp.power(Mi_hvals[j-3], 2)))))
                    # For loop to compute soft gaugino mass derivatives
                    # Right shifts
                    mzsq_Miph[j-3] = deriv_step_calc_gaugino(j, Mi_hvals[j-3])
                    bar()
                    mzsq_Mipph[j-3] = 0
                    # Left shifts
                    mzsq_Mimh[j-3] = deriv_step_calc_gaugino(j, (-1) * Mi_hvals[j-3])
                    bar()
                    mzsq_Mimmh[j-3] = 0
                    derivcount += 1
                    bar.text = f'Derivatives: {derivcount}/30, please wait...'

                for j in np.arange(25, 27):
                    print("Error estimate for " + str(Higgs_labels[j-25]) + " derivative: "
                          + str("{:.4e}".format(float(mp.power(Higgs_hvals[j-25], 2)))))
                    # For loop to compute soft Higgs mass derivatives
                    # Right shifts
                    mzsq_mHph[j-25] = deriv_step_calc_Higgs(j, Higgs_hvals[j-25])
                    bar()
                    mzsq_mHpph[j-25] = 0
                    # Left shifts
                    mzsq_mHmh[j-25] = deriv_step_calc_Higgs(j, (-1) * Higgs_hvals[j-25])
                    bar()
                    mzsq_mHmmh[j-25] = 0
                    derivcount += 1
                    bar.text = f'Derivatives: {derivcount}/30, please wait...'

                print("Error estimate for mu_0 derivative: "
                      + str("{:.4e}".format(float(mp.power(hmu0, 2)))))
                mzsq_mu0ph = deriv_step_calc_mu0(hmu0)
                bar()
                mzsq_mu0pph = 0
                mzsq_mu0mh = deriv_step_calc_mu0((-1) * hmu0)
                bar()
                mzsq_mu0mmh = 0
                derivcount += 1
                bar.text = f'Derivatives: {derivcount}/30, please wait...'
                # Construct derivative array
                deriv_array = np.array([deriv_num_calc(hmqL1, mzsq_msqslmh[0], mzsq_msqslph[0]), #end qL1: 0
                                        deriv_num_calc(hmqL2, mzsq_msqslmh[1], mzsq_msqslph[1]), #end qL2: 1
                                        deriv_num_calc(hmqL3, mzsq_msqslmh[2], mzsq_msqslph[2]), #end qL3: 2
                                        deriv_num_calc(hmeL, mzsq_msqslmh[3], mzsq_msqslph[3]), #end eL: 3
                                        deriv_num_calc(hmmuL, mzsq_msqslmh[4], mzsq_msqslph[4]), #end muL: 4
                                        deriv_num_calc(hmtauL, mzsq_msqslmh[5], mzsq_msqslph[5]), #end tauL: 5
                                        deriv_num_calc(hmuR, mzsq_msqslmh[6], mzsq_msqslph[6]), #end uR: 6
                                        deriv_num_calc(hmcR, mzsq_msqslmh[7], mzsq_msqslph[7]), #end cR: 7
                                        deriv_num_calc(hmtR, mzsq_msqslmh[8], mzsq_msqslph[8]), #end tR: 8
                                        deriv_num_calc(hmdR, mzsq_msqslmh[9], mzsq_msqslph[9]), #end dR: 9
                                        deriv_num_calc(hmsR, mzsq_msqslmh[10], mzsq_msqslph[10]), #end sR: 10
                                        deriv_num_calc(hmbR, mzsq_msqslmh[11], mzsq_msqslph[11]), #end bR: 11
                                        deriv_num_calc(hmeR, mzsq_msqslmh[12], mzsq_msqslph[12]), #end eR: 12
                                        deriv_num_calc(hmmuR, mzsq_msqslmh[13], mzsq_msqslph[13]), #end muR: 13
                                        deriv_num_calc(hmtauR, mzsq_msqslmh[14], mzsq_msqslph[14]), #end tauR: 14
                                        deriv_num_calc(hAt, mzsq_Aimh[0], mzsq_Aiph[0]), #end At: 15
                                        deriv_num_calc(hAc, mzsq_Aimh[1], mzsq_Aiph[1]), #end Ac: 16
                                        deriv_num_calc(hAu, mzsq_Aimh[2], mzsq_Aiph[2]), #end Au: 17
                                        deriv_num_calc(hAb, mzsq_Aimh[3], mzsq_Aiph[3]), #end Ab: 18
                                        deriv_num_calc(hAs, mzsq_Aimh[4], mzsq_Aiph[4]), #end As: 19
                                        deriv_num_calc(hAd, mzsq_Aimh[5], mzsq_Aiph[5]), #end Ad: 20
                                        deriv_num_calc(hAtau, mzsq_Aimh[6], mzsq_Aiph[6]), #end Atau: 21
                                        deriv_num_calc(hAmu, mzsq_Aimh[7], mzsq_Aiph[7]), #end Amu: 22
                                        deriv_num_calc(hAe, mzsq_Aimh[8], mzsq_Aiph[8]), #end Ae: 23
                                        deriv_num_calc(hM1, mzsq_Mimh[0], mzsq_Miph[0]), #end M1: 24
                                        deriv_num_calc(hM2, mzsq_Mimh[1], mzsq_Miph[1]), #end M2: 25
                                        deriv_num_calc(hM3,mzsq_Mimh[2], mzsq_Miph[2]), #end M3: 26
                                        deriv_num_calc(hmHu0,mzsq_mHmh[0], mzsq_mHph[0]), #end mHu0: 27
                                        deriv_num_calc(hmHd0,mzsq_mHmh[1], mzsq_mHph[1]), #end mHd0: 28
                                        deriv_num_calc(hmu0, mzsq_mu0mh, mzsq_mu0ph)]) #end mu0: 29
                sens_params = np.sort(np.array([(mp.fabs((mymqL1
                                                          / mymzsq)
                                                         * deriv_array[0]),
                                                 'Delta_BG(m_qL(1))'),
                                                (mp.fabs((mymqL2
                                                          / mymzsq)
                                                         * deriv_array[1]),
                                                 'Delta_BG(m_qL(2))'),
                                                (mp.fabs((mymqL3
                                                          / mymzsq)
                                                         * deriv_array[2]),
                                                 'Delta_BG(m_qL(3))'),
                                                (mp.fabs((mymeL
                                                          / mymzsq)
                                                         * deriv_array[3]),
                                                 'Delta_BG(m_eL)'),
                                                (mp.fabs((mymmuL
                                                          / mymzsq)
                                                         * deriv_array[4]),
                                                 'Delta_BG(m_muL)'),
                                                (mp.fabs((mymtauL
                                                          / mymzsq)
                                                         * deriv_array[5]),
                                                 'Delta_BG(m_tauL)'),
                                                (mp.fabs((mymuR
                                                          / mymzsq)
                                                         * deriv_array[6]),
                                                 'Delta_BG(m_uR)'),
                                                (mp.fabs((mymcR
                                                          / mymzsq)
                                                         * deriv_array[7]),
                                                 'Delta_BG(m_cR)'),
                                                (mp.fabs((mymtR
                                                          / mymzsq)
                                                         * deriv_array[8]),
                                                 'Delta_BG(m_tR)'),
                                                (mp.fabs((mymdR
                                                          / mymzsq)
                                                         * deriv_array[9]),
                                                 'Delta_BG(m_dR)'),
                                                (mp.fabs((mymsR
                                                          / mymzsq)
                                                         * deriv_array[10]),
                                                 'Delta_BG(m_sR)'),
                                                (mp.fabs((mymbR
                                                          / mymzsq)
                                                         * deriv_array[11]),
                                                 'Delta_BG(m_bR)'),
                                                (mp.fabs((mymeR
                                                          / mymzsq)
                                                         * deriv_array[12]),
                                                 'Delta_BG(m_eR)'),
                                                (mp.fabs((mymmuR
                                                          / mymzsq)
                                                         * deriv_array[13]),
                                                 'Delta_BG(m_muR)'),
                                                (mp.fabs((mymtauR
                                                          / mymzsq)
                                                         * deriv_array[14]),
                                                 'Delta_BG(m_tauR)'),
                                                (mp.fabs((myAt
                                                          / mymzsq)
                                                         * deriv_array[15]),
                                                 'Delta_BG(a_t)'),
                                                (mp.fabs((myAc
                                                          / mymzsq)
                                                         * deriv_array[16]),
                                                 'Delta_BG(a_c)'),
                                                (mp.fabs((myAu
                                                          / mymzsq)
                                                         * deriv_array[17]),
                                                 'Delta_BG(a_u)'),
                                                (mp.fabs((myAb
                                                          / mymzsq)
                                                         * deriv_array[18]),
                                                 'Delta_BG(a_b)'),
                                                (mp.fabs((myAs
                                                          / mymzsq)
                                                         * deriv_array[19]),
                                                 'Delta_BG(a_s)'),
                                                (mp.fabs((myAd
                                                          / mymzsq)
                                                         * deriv_array[20]),
                                                 'Delta_BG(a_d)'),
                                                (mp.fabs((myAtau
                                                          / mymzsq)
                                                         * deriv_array[21]),
                                                 'Delta_BG(a_tau)'),
                                                (mp.fabs((myAmu
                                                          / mymzsq)
                                                         * deriv_array[22]),
                                                 'Delta_BG(a_mu)'),
                                                (mp.fabs((myAe
                                                          / mymzsq)
                                                         * deriv_array[23]),
                                                 'Delta_BG(a_e)'),
                                                (mp.fabs((myM1
                                                          / mymzsq)
                                                         * deriv_array[24]),
                                                 'Delta_BG(M_1)'),
                                                (mp.fabs((myM2
                                                          / mymzsq)
                                                         * deriv_array[25]),
                                                 'Delta_BG(M_2)'),
                                                (mp.fabs((myM3
                                                          / mymzsq)
                                                         * deriv_array[26]),
                                                 'Delta_BG(M_3)'),
                                                (mp.fabs((mymHu0
                                                          / mymzsq)
                                                         * deriv_array[27]),
                                                 'Delta_BG(m_Hu0)'),
                                                (mp.fabs((mymHd0
                                                          / mymzsq)
                                                         * deriv_array[28]),
                                                 'Delta_BG(m_Hd0)'),
                                                (mp.fabs((mymu0
                                                          / mymzsq)
                                                         * deriv_array[29]),
                                                 'Delta_BG(mu0)')],
                                               dtype=[('BGcontrib',float),
                                                      ('BGlabel','U40')]),
                                      order='BGcontrib')
                print("Done!")
    return sens_params[::-1]
