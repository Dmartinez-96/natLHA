#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 14:35:38 2023

Code for the terminal interface of the DEW4SLHA program.

@author: Dakotah Martinez
"""

from mpmath import mp, mpf, nstr
import pyslha
import os
from copy import deepcopy
import time
from DEW_calc import Delta_EW_calc
from DBG_calc import Delta_BG_calc
from DHS_calc import Delta_HS_calc
from radcorr_calc import my_radcorr_calc
from RGEsolver import my_RGE_solver as RGEsols

def main():
    """
    Main routine to allow user to interact with DEW4SLHA via the terminal.

    Returns
    -------
    None.

    """
    if (os.name == 'nt'):
        screen_clearer=os.system('cls')
    else:
        screen_clearer=os.system('clear')
    ##### Constants #####
    mp.dps = 50

    userContinue = True
    print("Welcome to DEW4SLHA, a program for computing the naturalness")
    print("measures Delta_EW, Delta_BG, and Delta_HS in the MSSM")
    print("from a SUSY Les Houches Accord (SLHA) file.")
    print("")
    print("To use this program, you may select a")
    print("MSSM SLHA file from your choice of spectrum generator (e.g.,")
    print("SoftSUSY, Isajet, SPheno, FlexibleSUSY, etc.). ")
    print("If multiple renormalization scales are present in the SLHA file,")
    print("then the first renormalization scale present in the SLHA file,")
    print("from top to bottom, is read in.")
    print("")
    print("Delta_EW, Delta_BG, and Delta_HS will be evaluated at the")
    print(" renormalization scale provided in the SLHA file.")
    print("")
    print("Supported models for the local solvers are MSSM EFT models for\n"
          + "Delta_EW and Delta_HS, but only the CMSSM, NUHM(1,2,3,4),\n"
          + "pMSSM-19, and pMSSM-30 for Delta_BG.\n\n")
    input("Press Enter to begin.")
    while userContinue:
        if (os.name == 'nt'):
            screen_clearer=os.system('cls')
        else:
            screen_clearer=os.system('clear')
        DEWprogcheck = True
        print("##############################################################")
        print("\nDEW4SLHA calculates the electroweak naturalness measure"
              + " Delta_EW by default.")
        print("")
        # Check if user wants to compute Delta_HS as well.
        checkcompDHS = True
        while checkcompDHS:
            print("\n####################################################\n")
            DHScheckinp = input("Would you like to also calculate the"
                                + " high-scale naturalness measure Delta_HS?"
                                + "\nEnter Y for yes or N for no: ")
            if (DHScheckinp.lower() in ('n', 'no')):
                DHScalc = False
                checkcompDHS = False
            elif (DHScheckinp.lower() in ('y', 'yes')):
                DHScalc = True
                checkcompDHS = False
            else:
                print("Invalid input, please try again.")
                print("")
                time.sleep(1)
                checkcompDHS = True

        print("\n####################################################\n")
        # Check if user wants to compute Delta_BG as well.
        checkcompDBG = True
        while checkcompDBG:
            print("\n####################################################\n")
            DBGcheckinp = input("Would you like to also calculate the"
                                + " Barbieri-Guidice naturalness measure "
                                + "Delta_BG?\nEnter Y for yes or N for no: ")
            if (DBGcheckinp.lower() in ('n', 'no')):
                DBGcalc = False
                checkcompDBG = False
            elif (DBGcheckinp.lower() in ('y', 'yes')):
                DBGcalc = True
                checkcompDBG = False
            else:
                print("Invalid input, please try again.")
                print("")
                time.sleep(1)
                checkcompDBG = True
        print("\n####################################################\n")
        if DBGcalc:
            print("\nFor Delta_BG, the ``fundamental parameters'' vary from "
                  + "model to model.\nFor this reason, prior to entering the"
                  + " directory of your SLHA file,\nplease enter the model"
                  + " number below corresponding to your SLHA file.")
            print("NOTE: this computation can take a while, especially"
                  + " for the pMSSM-19 and 30.")
            print("")
            print("Model numbers: ")
            print("1: CMSSM/mSUGRA")
            print("2: NUHM1")
            print("3: NUHM2")
            print("4: NUHM3")
            print("5: NUHM4")
            print("6: pMSSM-19")
            print("7: pMSSM-30 (pMSSM-19 + 11 diagonal,"
                  + " real, non-universal "
                  + "1st & 2nd gen. soft trilinears,"
                  + " squark and slepton"
                  + " masses)")
            print("")
            modelCheck = True
            while modelCheck:
                try:
                    modinp = int(input("From the list above, input the number "
                                       + "of the model your SLHA file"
                                       + " corresponds to: "))
                    if (modinp not in [1, 2, 3, 4, 5, 6, 7]):
                        print("Invalid model number selected, please try"
                              + " again.")
                        time.sleep(1)
                        print("\n################################"
                              + "####################\n")
                        print("Model numbers: ")
                        print("1: CMSSM/mSUGRA")
                        print("2: NUHM1")
                        print("3: NUHM2")
                        print("4: NUHM3")
                        print("5: NUHM4")
                        print("6: pMSSM-19")
                        print("7: pMSSM-30 (pMSSM-19 + 11 diagonal,"
                              + " real, non-universal "
                              + "1st & 2nd gen. soft trilinears,"
                              + " squark and slepton"
                              + " masses)")
                        print("")
                        modelCheck = True
                    else:
                        modelCheck = False
                except(ValueError):
                    print("Invalid model number selected, please try again.")
                    print("")
                    time.sleep(1)
                    print("\n################################"
                          + "####################\n")
                    print("Model numbers: ")
                    print("1: CMSSM/mSUGRA")
                    print("2: NUHM1")
                    print("3: NUHM2")
                    print("4: NUHM3")
                    print("5: NUHM4")
                    print("6: pMSSM-19")
                    print("7: pMSSM-30 (pMSSM-19 + 11 diagonal,"
                          + " real, non-universal "
                          + "1st & 2nd gen. soft trilinears,"
                          + " squark and slepton"
                          + " masses)")
                    print("")
                    modelCheck = True

            # Set up parameters for computations
            print("\n####################################################\n")
            print("Please select the level of precision you want for the"
                  + "  Delta_BG calculation.")
            print("Below are the options: ")
            print("1: High precision, slow calculation.")
            print("2: Medium precision, twice as fast as high precision mode.")
            print("3: Lowest precision, four times as fast as high precision"
                  + " mode.")
            print("")
            precisionCheck = True
            while precisionCheck:
                try:
                    precinp = int(input("From the list above, input the number "
                                        + "corresponding to the precision you "
                                        + "want: "))
                    if (precinp not in [1, 2, 3]):
                        print("Invalid Delta_BG precision setting selected, please"
                              + " try again.")
                        print("")
                        time.sleep(1)
                        print("\n################################"
                              + "####################\n")
                        print("Please select the level of precision you want for"
                              + " the Delta_BG calculation.")
                        print("Below are the options: ")
                        print("1: High precision, slow calculation.")
                        print("2: Medium precision, twice as fast as "
                              + "high precision mode.")
                        print("3: Lowest precision, four times as fast as"
                              + " high precision mode.")
                        print("")
                        precisionCheck = True
                    else:
                        precisionCheck = False
                except(ValueError):
                    print("Invalid Delta_BG precision setting selected, please"
                          + " try again.")
                    print("")
                    time.sleep(1)
                    print("\n################################"
                          + "####################\n")
                    print("Please select the level of precision you want for"
                          + " the Delta_BG calculation.")
                    print("Below are the options: ")
                    print("1: High precision, slow calculation.")
                    print("2: Medium precision, twice as fast as "
                          + "high precision mode.")
                    print("3: Lowest precision, four times as fast as"
                          + " high precision mode.")
                    print("")
                    precisionCheck=True

            # print("\n####################################################\n")
            # print("Would you like to include 1-loop self-energy corrections to"
            #       +" the mZ^2 evaluation?\n")
            poleinp = 1
            ##### TODO: Will enable again in a later version, with loop corrections to Delta_BG
            # polemZCheck = True
            # while polemZCheck:
            #     try:
            #         poleinp = int(input("Input 1 to include self-energy corrections,"
            #                             + " or 0 to ignore: "))
            #         if (poleinp not in [0,1]):
            #             print("Invalid mZ^2 pole mass setting selected, please"
            #                   + " try again.")
            #             print("")
            #             time.sleep(1)
            #             print("\n################################"
            #                   + "####################\n")
            #             print("Would you like to include 1-loop self-energy corrections to"
            #                   +" the mZ^2 evaluation?\n")
            #             polemZCheck = True
            #         else:
            #             polemZCheck = False
            #     except:
            #         print("Invalid mZ^2 pole mass setting selected, please"
            #               + " try again.")
            #         print("")
            #         time.sleep(1)
            #         print("\n################################"
            #               + "####################\n")
            #         print("Would you like to include 1-loop self-energy corrections to"
            #               +" the mZ^2 evaluation?\n")
            #         polemZCheck=True

        print("\n########## Configuration Complete ##########\n")
        time.sleep(1.5)
        if (os.name == 'nt'):
            screen_clearer=os.system('cls')
        else:
            screen_clearer=os.system('clear')
        # SLHA input and definition of variables from SLHA file: #
        fileCheck = True
        while fileCheck:
            try:
                direc = input('Enter the full directory for your'
                              + ' SLHA file: ')
                d = pyslha.read(direc)
                fileCheck = False
            except (FileNotFoundError):
                print("The input file cannot be found.\n")
                print("Please try checking your spelling and try again.\n")
                fileCheck = True
            except (IsADirectoryError):
                print("You have input a directory, not an SLHA file.\n")
                print("Please try again.\n")
            except (NotADirectoryError):
                print("The input file cannot be found.\n")
                print("Please try checking your spelling and try again,"
                      + " without putting a slash at the end of the file name.\n")
        time.sleep(0.5)
        if (os.name == 'nt'):
            screen_clearer=os.system('cls')
        else:
            screen_clearer=os.system('clear')
        print("Analyzing submitted SLHA.")
        mZ = 91.1876 # This is the value in our universe, not for multiverse.
        [vHiggs, muQ, tanb, y_t] = [mpf(str(d.blocks['HMIX'][3])) / float(mp.sqrt(2)),
                                    mpf(str(d.blocks['HMIX'][1])),
                                    mpf(str(d.blocks['HMIX'][2])),
                                    mpf(str(d.blocks['YU'][3, 3]))]
        beta = mp.atan(tanb)
        [y_b, y_tau, g_2] = [mpf(str(d.blocks['YD'][3, 3])),
                             mpf(str(d.blocks['YE'][3, 3])),
                             mpf(str(d.blocks['GAUGE'][2]))]
        # See if SLHA has trilinears (a_i) in reduced form or not (A_i),
        # where a_i = y_i * A_i
        try:
            [a_t, a_b, a_tau] = [mpf(str(d.blocks['TU'][3, 3])),
                                 mpf(str(d.blocks['TD'][3, 3])),
                                 mpf(str(d.blocks['TE'][3, 3]))]
        except(KeyError):
            [a_t, a_b, a_tau] = [mpf(str(d.blocks['AU'][3, 3])) * y_t,
                                 mpf(str(d.blocks['AD'][3, 3])) * y_b,
                                 mpf(str(d.blocks['AE'][3, 3])) * y_tau]
        # Try to read in first two generations of Yukawas and soft trilinears
        # if present; if not, set approximate values based on 3rd gens,
        # coming from test BM points with universal trilinears at M_GUT.
        try:
            [y_c, y_u, y_s, y_d, y_mu, y_e] = [mpf(str(d.blocks['YU'][2, 2])),
                                               mpf(str(d.blocks['YU'][1, 1])),
                                               mpf(str(d.blocks['YD'][2, 2])),
                                               mpf(str(d.blocks['YD'][1, 1])),
                                               mpf(str(d.blocks['YE'][2, 2])),
                                               mpf(str(d.blocks['YE'][1, 1]))]
            [a_c, a_u, a_s, a_d, a_mu, a_e] = [mpf(str(d.blocks['TU'][2, 2])),
                                               mpf(str(d.blocks['TU'][1, 1])),
                                               mpf(str(d.blocks['TD'][2, 2])),
                                               mpf(str(d.blocks['TD'][1, 1])),
                                               mpf(str(d.blocks['TE'][2, 2])),
                                               mpf(str(d.blocks['TE'][1, 1]))]
        except(KeyError):
            try:
                [y_c, y_u, y_s, y_d, y_mu, y_e] = [mpf(str(d.blocks['YU'][2,
                                                                          2])),
                                                   mpf(str(d.blocks['YU'][1,
                                                                          1])),
                                                   mpf(str(d.blocks['YD'][2,
                                                                          2])),
                                                   mpf(str(d.blocks['YD'][1,
                                                                          1])),
                                                   mpf(str(d.blocks['YE'][2,
                                                                          2])),
                                                   mpf(str(d.blocks['YE'][1,
                                                                          1]))]
                [a_c, a_u, a_s, a_d, a_mu, a_e] = [mpf(str(d.blocks['AU'][2,
                                                                          2]))
                                                   * y_c,
                                                   mpf(str(d.blocks['AU'][1,
                                                                          1]))
                                                   * y_u,
                                                   mpf(str(d.blocks['AD'][2,
                                                                          2]))
                                                   * y_s,
                                                   mpf(str(d.blocks['AD'][1,
                                                                          1]))
                                                   * y_d,
                                                   mpf(str(d.blocks['AE'][2,
                                                                          2]))
                                                   * y_mu,
                                                   mpf(str(d.blocks['AE'][1,
                                                                          1]))
                                                   * y_e]
            except(KeyError):
                try:
                    [y_c, y_u, y_s,
                     y_d, y_mu, y_e] = [mpf('0.003882759826930082') * y_t,
                                        mpf('7.779613278615955e-06') * y_t,
                                        mpf('0.0206648802754076') * y_b,
                                        mpf('0.0010117174290779725') * y_b,
                                        mpf('0.05792142442492775') * y_tau,
                                        mpf('0.0002801267571260388') * y_tau]
                    [a_c, a_u, a_s, a_d, a_mu, a_e] = [mpf(str(d.blocks['AU'][2,
                                                                              2]))
                                                       * y_c,
                                                       mpf(str(d.blocks['AU'][1,
                                                                              1]))
                                                       * y_u,
                                                       mpf(str(d.blocks['AD'][2,
                                                                              2]))
                                                       * y_s,
                                                       mpf(str(d.blocks['AD'][1,
                                                                              1]))
                                                       * y_d,
                                                       mpf(str(d.blocks['AE'][2,
                                                                              2]))
                                                       * y_mu,
                                                       mpf(str(d.blocks['AE'][1,
                                                                              1]))
                                                       * y_e]
                except(KeyError):
                    [y_c, y_u, y_s,
                     y_d, y_mu, y_e] = [mpf('0.003882759826930082') * y_t,
                                        mpf('7.779613278615955e-06') * y_t,
                                        mpf('0.0206648802754076') * y_b,
                                        mpf('0.0010117174290779725') * y_b,
                                        mpf('0.05792142442492775') * y_tau,
                                        mpf('0.0002801267571260388') * y_tau]
                    print("Can't find entries [2,2] and [1,1] of blocks AU,AD,AE or TU,TD,TE from SLHA.")
                    print("Approximating 1st and 2nd gen soft trilinears")
                    [a_c, a_u, a_s,
                     a_d, a_mu, a_e] = [mpf('0.004905858561422854') * a_t,
                                        mpf('9.829562752270226e-06') * a_t,
                                        mpf('0.021974714097596777') * a_b,
                                        mpf('0.0010758476898828158') * a_b,
                                        mpf('0.058219688597781954') * a_tau,
                                        mpf('0.0002815741158519892') * a_tau]
        g_pr = mpf(str(d.blocks['GAUGE'][1]))
        g_s = mpf(str(d.blocks['GAUGE'][3]))
        # Different blocks exist for soft masses in different SLHA-producing
        # programs. Try to read in masses from the two most popular.
        try:
            [mQ3sq, mU3sq] = [mpf(str(d.blocks['MSQ2'][3, 3])),
                              mpf(str(d.blocks['MSU2'][3, 3]))]
            [mD3sq, mL3sq, mE3sq] = [mpf(str(d.blocks['MSD2'][3, 3])),
                                     mpf(str(d.blocks['MSL2'][3, 3])),
                                     mpf(str(d.blocks['MSE2'][3, 3]))]
            [mQ2sq, mU2sq] = [mpf(str(d.blocks['MSQ2'][2, 2])),
                              mpf(str(d.blocks['MSU2'][2, 2]))]
            [mD2sq, mL2sq, mE2sq] = [mpf(str(d.blocks['MSD2'][2, 2])),
                                     mpf(str(d.blocks['MSL2'][2, 2])),
                                     mpf(str(d.blocks['MSE2'][2, 2]))]
            [mQ1sq, mU1sq] = [mpf(str(d.blocks['MSQ2'][1, 1])),
                              mpf(str(d.blocks['MSU2'][1, 1]))]
            [mD1sq, mL1sq, mE1sq] = [mpf(str(d.blocks['MSD2'][1, 1])),
                                     mpf(str(d.blocks['MSL2'][1, 1])),
                                     mpf(str(d.blocks['MSE2'][1, 1]))]
        except(KeyError):
            [mQ3sq, mU3sq] = [mp.power(d.blocks['MSOFT'][43], 2),
                              mp.power(d.blocks['MSOFT'][46], 2)]
            [mD3sq, mL3sq, mE3sq] = [mp.power(d.blocks['MSOFT'][49], 2),
                                     mp.power(d.blocks['MSOFT'][33], 2),
                                     mp.power(d.blocks['MSOFT'][36], 2)]
            [mQ2sq, mU2sq] = [mp.power(d.blocks['MSOFT'][42], 2),
                              mp.power(d.blocks['MSOFT'][45], 2)]
            [mD2sq, mL2sq, mE2sq] = [mp.power(d.blocks['MSOFT'][48], 2),
                                     mp.power(d.blocks['MSOFT'][32], 2),
                                     mp.power(d.blocks['MSOFT'][35], 2)]
            [mQ1sq, mU1sq] = [mp.power(d.blocks['MSOFT'][41], 2),
                              mp.power(d.blocks['MSOFT'][44], 2)]
            [mD1sq, mL1sq, mE1sq] = [mp.power(d.blocks['MSOFT'][47], 2),
                                     mp.power(d.blocks['MSOFT'][31], 2),
                                     mp.power(d.blocks['MSOFT'][34], 2)]
        my_M3 = mpf(str(d.blocks['MSOFT'][3]))
        my_M2 = mpf(str(d.blocks['MSOFT'][2]))
        my_M1 = mpf(str(d.blocks['MSOFT'][1]))
        [mHusq, mHdsq] = [mpf(str(d.blocks['MSOFT'][22])),
                          mpf(str(d.blocks['MSOFT'][21]))]
        # Read in SLHA scale from submitted file at which previous variables
        # are evaluated.
        SLHA_scale = float(str(d.blocks['GAUGE'])
                           [str(d.blocks['GAUGE']).find('Q=')
                            +2:str(d.blocks['GAUGE']).find(')')])
        print("SLHA parameters read in. RGE evolving to SLHA scale.")
        # If calculating Delta_BG, need pole masses for mZ^2 derivative.
        # Read these in.
        if DBGcalc:
            myalpha = mpf(str(d.blocks['ALPHA'][None]))
            MWpole = mpf(str(d.blocks['MASS'][24]))
            Mhpole = mpf(str(d.blocks['MASS'][25]))
            MHpole = mpf(str(d.blocks['MASS'][35]))
            MA0pole = mpf(str(d.blocks['MASS'][36]))
            MHpmpole = mpf(str(d.blocks['MASS'][37]))
            Mneut1pole = mpf(str(d.blocks['MASS'][1000022]))
            Mneut2pole = mpf(str(d.blocks['MASS'][1000023]))
            Mneut3pole = mpf(str(d.blocks['MASS'][1000025]))
            Mneut4pole = mpf(str(d.blocks['MASS'][1000035]))
            Mchar1pole = mpf(str(d.blocks['MASS'][1000024]))
            Mchar2pole = mpf(str(d.blocks['MASS'][1000037]))
            MdLpole = mpf(str(d.blocks['MASS'][1000001]))
            MuLpole = mpf(str(d.blocks['MASS'][1000002]))
            MsLpole = mpf(str(d.blocks['MASS'][1000003]))
            McLpole = mpf(str(d.blocks['MASS'][1000004]))
            Mb1pole = mpf(str(d.blocks['MASS'][1000005]))
            Mt1pole = mpf(str(d.blocks['MASS'][1000006]))
            MeLpole = mpf(str(d.blocks['MASS'][1000011]))
            MnueLpole = mpf(str(d.blocks['MASS'][1000012]))
            MmuLpole = mpf(str(d.blocks['MASS'][1000013]))
            MnumuLpole = mpf(str(d.blocks['MASS'][1000014]))
            Mtau1pole = mpf(str(d.blocks['MASS'][1000015]))
            MnutauLpole = mpf(str(d.blocks['MASS'][1000016]))
            MdRpole = mpf(str(d.blocks['MASS'][2000001]))
            MuRpole = mpf(str(d.blocks['MASS'][2000002]))
            MsRpole = mpf(str(d.blocks['MASS'][2000003]))
            McRpole = mpf(str(d.blocks['MASS'][2000004]))
            Mb2pole = mpf(str(d.blocks['MASS'][2000005]))
            Mt2pole = mpf(str(d.blocks['MASS'][2000006]))
            MeRpole = mpf(str(d.blocks['MASS'][2000011]))
            MmuRpole = mpf(str(d.blocks['MASS'][2000013]))
            Mtau2pole = mpf(str(d.blocks['MASS'][2000015]))
            Mc1pole = min([McLpole, McRpole])
            Mc2pole = max([McLpole, McRpole])
            Mu1pole = min([MuLpole, MuRpole])
            Mu2pole = max([MuLpole, MuRpole])
            Ms1pole = min([MsLpole, MsRpole])
            Ms2pole = max([MsLpole, MsRpole])
            Md1pole = min([MdLpole, MdRpole])
            Md2pole = max([MdLpole, MdRpole])
            Mmu1pole = min([MmuLpole, MmuRpole])
            Mmu2pole = max([MmuLpole, MmuRpole])
            Me1pole = min([MeLpole, MeRpole])
            Me2pole = max([MeLpole, MeRpole])
        # Use 2-loop MSSM RGEs to evolve results to a renormalization scale
        # of Q = 2 TeV if the submitted SLHA file is not currently at that
        # scale. This is so evaluations of the naturalness measures are always
        # performed at a common scale of 2 TeV.
        #######################################################################
        # The result is then run to a high scale of 10^17 GeV, and an
        # approximate GUT scale is chosen at the value where g1(Q) is closest
        # to g2(Q) over the scanned renormalization scales.
        #######################################################################
        # This running to the GUT scale is used in the evaluations of Delta_HS
        # and Delta_BG.
        # Compute tree-level soft Higgs bilinear parameter b=B*mu at scale
        # in SLHA for RGE boundary condition.
        b_from_SLHA = ((mHusq + mHdsq + (2 * mp.power(muQ, 2)))#+ radcorrs_from_SLHA[0] + radcorrs_from_SLHA[1])
                       * (mp.sin(beta) * mp.cos(beta)))
        mySLHABCs = deepcopy([mp.sqrt(5 / 3) * g_pr, g_2, g_s, my_M1, my_M2, my_M3,
                              muQ ** 2, y_t, y_c, y_u, y_b, y_s, y_d, y_tau, y_mu, y_e,
                              a_t, a_c, a_u, a_b, a_s, a_d, a_tau, a_mu, a_e,
                              mHusq, mHdsq, mQ1sq, mQ2sq, mQ3sq, mL1sq, mL2sq,
                              mL3sq, mU1sq, mU2sq, mU3sq, mD1sq, mD2sq, mD3sq,
                              mE1sq, mE2sq, mE3sq, b_from_SLHA, tanb])
        RGE_sols = RGEsols(mySLHABCs, SLHA_scale, SLHA_scale)

        print("\nRGEs solved.")
        time.sleep(1)
        # Read in several parameters at weak and GUT scales
        myQGUT = RGE_sols[0]
        muQsq = RGE_sols[8]
        muQ_GUTsq = RGE_sols[51]
        g1Q_GUT = RGE_sols[45]
        g2Q_GUT = RGE_sols[46]
        g3Q_GUT = RGE_sols[47]
        # Set GUT-scale gaugino masses
        try:
            M1Q_GUT = mpf(str(d.blocks['MINPAR'][2]))
            M2Q_GUT = mpf(str(d.blocks['MINPAR'][2]))
            M3Q_GUT = mpf(str(d.blocks['MINPAR'][2]))
        except:
            try:
                M1Q_GUT = mpf(str(d.blocks['EXTPAR'][1]))
                M2Q_GUT = mpf(str(d.blocks['EXTPAR'][2]))
                M3Q_GUT = mpf(str(d.blocks['EXTPAR'][3]))
            except:
                M1Q_GUT = RGE_sols[48]
                M2Q_GUT = RGE_sols[49]
                M3Q_GUT = RGE_sols[50]
        beta = float(mp.atan(RGE_sols[88]))
        betaGUT = float(mp.atan(RGE_sols[89]))
        tanbQ_GUT = RGE_sols[89]
        y_t = RGE_sols[9]
        y_tQ_GUT = RGE_sols[52]
        y_c = RGE_sols[10]
        y_cQ_GUT = RGE_sols[53]
        y_u = RGE_sols[11]
        y_uQ_GUT = RGE_sols[54]
        y_b = RGE_sols[12]
        y_bQ_GUT = RGE_sols[55]
        y_s = RGE_sols[13]
        y_sQ_GUT = RGE_sols[56]
        y_d = RGE_sols[14]
        y_dQ_GUT = RGE_sols[57]
        y_tau = RGE_sols[15]
        y_tauQ_GUT = RGE_sols[58]
        y_mu = RGE_sols[16]
        y_muQ_GUT = RGE_sols[59]
        y_e = RGE_sols[17]
        y_eQ_GUT = RGE_sols[60]

        a_t = RGE_sols[18]
        a_tQ_GUT = RGE_sols[61]
        a_c = RGE_sols[19]
        a_cQ_GUT = RGE_sols[62]
        a_u = RGE_sols[20]
        a_uQ_GUT = RGE_sols[63]
        a_b = RGE_sols[21]
        a_bQ_GUT = RGE_sols[64]
        a_s = RGE_sols[22]
        a_sQ_GUT = RGE_sols[65]
        a_d = RGE_sols[23]
        a_dQ_GUT = RGE_sols[66]
        a_tau = RGE_sols[24]
        a_tauQ_GUT = RGE_sols[67]
        a_mu = RGE_sols[25]
        a_muQ_GUT = RGE_sols[68]
        a_e = RGE_sols[26]
        a_eQ_GUT = RGE_sols[69]
        mHusq = RGE_sols[27]
        mHusqQ_GUT = RGE_sols[70]
        mHdsq = RGE_sols[28]
        mHdsqQ_GUT = RGE_sols[71]
        # Set unified scalar masses from SLHA file (if present)
        try:
            mQ1sq = RGE_sols[29]
            m_uLQ_GUT = d.blocks['MINPAR'][1]
            mQ2sq = RGE_sols[30]
            m_cLQ_GUT = d.blocks['MINPAR'][1]
            mQ3sq = RGE_sols[31]
            m_tLQ_GUT = d.blocks['MINPAR'][1]
            mL1sq = RGE_sols[32]
            m_eLQ_GUT = d.blocks['MINPAR'][1]
            mL2sq = RGE_sols[33]
            m_muLQ_GUT = d.blocks['MINPAR'][1]
            mL3sq = RGE_sols[34]
            m_tauLQ_GUT = d.blocks['MINPAR'][1]
            mU1sq = RGE_sols[35]
            m_uRQ_GUT = d.blocks['MINPAR'][1]
            mU2sq = RGE_sols[36]
            m_cRQ_GUT = d.blocks['MINPAR'][1]
            mU3sq = RGE_sols[37]
            m_tRQ_GUT = d.blocks['MINPAR'][1]
            mD1sq = RGE_sols[38]
            m_dRQ_GUT = d.blocks['MINPAR'][1]
            mD2sq = RGE_sols[39]
            m_sRQ_GUT = d.blocks['MINPAR'][1]
            mD3sq = RGE_sols[40]
            m_bRQ_GUT = d.blocks['MINPAR'][1]
            mE1sq = RGE_sols[41]
            m_eRQ_GUT = d.blocks['MINPAR'][1]
            mE2sq = RGE_sols[42]
            m_muRQ_GUT = d.blocks['MINPAR'][1]
            mE3sq = RGE_sols[43]
            m_tauRQ_GUT = d.blocks['MINPAR'][1]
        except(KeyError):
            try:
                mQ1sq = RGE_sols[29]
                m_uLQ_GUT = d.blocks['EXTPAR'][41]
                mQ2sq = RGE_sols[30]
                m_cLQ_GUT = d.blocks['EXTPAR'][42]
                mQ3sq = RGE_sols[31]
                m_tLQ_GUT = d.blocks['EXTPAR'][43]
                mL1sq = RGE_sols[32]
                m_eLQ_GUT = d.blocks['EXTPAR'][31]
                mL2sq = RGE_sols[33]
                m_muLQ_GUT = d.blocks['EXTPAR'][32]
                mL3sq = RGE_sols[34]
                m_tauLQ_GUT = d.blocks['EXTPAR'][33]
                mU1sq = RGE_sols[35]
                m_uRQ_GUT = d.blocks['EXTPAR'][44]
                mU2sq = RGE_sols[36]
                m_cRQ_GUT = d.blocks['EXTPAR'][45]
                mU3sq = RGE_sols[37]
                m_tRQ_GUT = d.blocks['EXTPAR'][46]
                mD1sq = RGE_sols[38]
                m_dRQ_GUT = d.blocks['EXTPAR'][47]
                mD2sq = RGE_sols[39]
                m_sRQ_GUT = d.blocks['EXTPAR'][48]
                mD3sq = RGE_sols[40]
                m_bRQ_GUT = d.blocks['EXTPAR'][49]
                mE1sq = RGE_sols[41]
                m_eRQ_GUT = d.blocks['EXTPAR'][34]
                mE2sq = RGE_sols[42]
                m_muRQ_GUT = d.blocks['EXTPAR'][35]
                mE3sq = RGE_sols[43]
                m_tauRQ_GUT = d.blocks['EXTPAR'][36]
            except(KeyError):
                mQ1sq = RGE_sols[29]
                m_uLQ_GUT = mp.sqrt(RGE_sols[72])
                mQ2sq = RGE_sols[30]
                m_cLQ_GUT = mp.sqrt(RGE_sols[73])
                mQ3sq = RGE_sols[31]
                m_tLQ_GUT = mp.sqrt(RGE_sols[74])
                mL1sq = RGE_sols[32]
                m_eLQ_GUT = mp.sqrt(RGE_sols[75])
                mL2sq = RGE_sols[33]
                m_muLQ_GUT = mp.sqrt(RGE_sols[76])
                mL3sq = RGE_sols[34]
                m_tauLQ_GUT = mp.sqrt(RGE_sols[77])
                mU1sq = RGE_sols[35]
                m_uRQ_GUT = mp.sqrt(RGE_sols[78])
                mU2sq = RGE_sols[36]
                m_cRQ_GUT = mp.sqrt(RGE_sols[79])
                mU3sq = RGE_sols[37]
                m_tRQ_GUT = mp.sqrt(RGE_sols[80])
                mD1sq = RGE_sols[38]
                m_dRQ_GUT = mp.sqrt(RGE_sols[81])
                mD2sq = RGE_sols[39]
                m_sRQ_GUT = mp.sqrt(RGE_sols[82])
                mD3sq = RGE_sols[40]
                m_bRQ_GUT = mp.sqrt(RGE_sols[83])
                mE1sq = RGE_sols[41]
                m_eRQ_GUT = mp.sqrt(RGE_sols[84])
                mE2sq = RGE_sols[42]
                m_muRQ_GUT = mp.sqrt(RGE_sols[85])
                mE3sq = RGE_sols[43]
                m_tauRQ_GUT = mp.sqrt(RGE_sols[86])
        del d
        my_b_weak = b_from_SLHA
        bQ_GUT = RGE_sols[87]
        Q_GUT_BCs = deepcopy([g1Q_GUT, g2Q_GUT, g3Q_GUT, M1Q_GUT, M2Q_GUT, M3Q_GUT,
                              muQ_GUTsq, y_tQ_GUT, y_cQ_GUT, y_uQ_GUT, y_bQ_GUT, y_sQ_GUT,
                              y_dQ_GUT, y_tauQ_GUT, y_muQ_GUT, y_eQ_GUT, a_tQ_GUT,
                              a_cQ_GUT, a_uQ_GUT, a_bQ_GUT, a_sQ_GUT, a_dQ_GUT,
                              a_tauQ_GUT, a_muQ_GUT, a_eQ_GUT, mHusqQ_GUT, mHdsqQ_GUT,
                              m_uLQ_GUT, m_cLQ_GUT, m_tLQ_GUT, m_eLQ_GUT, m_muLQ_GUT,
                              m_tauLQ_GUT, m_uRQ_GUT, m_cRQ_GUT, m_tRQ_GUT, m_dRQ_GUT,
                              m_sRQ_GUT, m_bRQ_GUT, m_eRQ_GUT, m_muRQ_GUT, m_tauRQ_GUT,
                              bQ_GUT, tanbQ_GUT])
        print("\n########## Computing Delta_EW... ##########\n")
        print("\nSolving loop-corrected minimization conditions.\n")
        time.sleep(1)
        radcorrinps = deepcopy([SLHA_scale, vHiggs, muQ, beta,
                                y_t, y_c, y_u, y_b, y_s, y_d,
                                y_tau, y_mu, y_e,
                                float(mp.sqrt(5 / 3)) * g_pr, g_2, g_s,
                                mQ3sq, mQ2sq, mQ1sq, mL3sq, mL2sq,
                                mL1sq, mU3sq, mU2sq, mU1sq, mD3sq,
                                mD2sq, mD1sq, mE3sq, mE2sq, mE1sq,
                                my_M1, my_M2, my_M3, mHusq, mHdsq,
                                a_t, a_c, a_u, a_b, a_s, a_d,
                                a_tau, a_mu, a_e])
        radcorrinps2 = deepcopy([SLHA_scale, vHiggs, muQ, beta,
                                 y_t, y_c, y_u, y_b, y_s, y_d,
                                 y_tau, y_mu, y_e,
                                 float(mp.sqrt(5 / 3)) * g_pr, g_2, g_s,
                                 mQ3sq, mQ2sq, mQ1sq, mL3sq, mL2sq,
                                 mL1sq, mU3sq, mU2sq, mU1sq, mD3sq,
                                 mD2sq, mD1sq, mE3sq, mE2sq, mE1sq,
                                 my_M1, my_M2, my_M3, mHusq, mHdsq,
                                 a_t, a_c, a_u, a_b, a_s, a_d,
                                 a_tau, a_mu, a_e])
        radcorrs_at_2TeV = my_radcorr_calc(radcorrinps[0], radcorrinps[1], radcorrinps[2], radcorrinps[3],
                                           radcorrinps[4],radcorrinps[5], radcorrinps[6], radcorrinps[7], radcorrinps[8], radcorrinps[9],
                                           radcorrinps[10], radcorrinps[11],radcorrinps[12],
                                           radcorrinps[13], radcorrinps[14], radcorrinps[15],
                                           radcorrinps[16],radcorrinps[17], radcorrinps[18],radcorrinps[19],radcorrinps[20],
                                           radcorrinps[21],radcorrinps[22],radcorrinps[23],radcorrinps[24],radcorrinps[25],
                                           radcorrinps[26],radcorrinps[27],radcorrinps[28],radcorrinps[29],radcorrinps[30],
                                           radcorrinps[31],radcorrinps[32],radcorrinps[33],radcorrinps[34],radcorrinps[35],
                                           radcorrinps[36],radcorrinps[37],radcorrinps[38],radcorrinps[39],radcorrinps[40],radcorrinps[41],
                                           radcorrinps[42],radcorrinps[43],radcorrinps[44])
        dewlist = Delta_EW_calc(radcorrinps2[0], radcorrinps2[1], radcorrinps2[2], radcorrinps2[3],
                                radcorrinps2[4],radcorrinps2[5], radcorrinps2[6], radcorrinps2[7], radcorrinps2[8], radcorrinps2[9],
                                radcorrinps2[10], radcorrinps2[11],radcorrinps2[12],
                                radcorrinps2[13], radcorrinps2[14], radcorrinps2[15],
                                radcorrinps2[16],radcorrinps2[17], radcorrinps2[18],radcorrinps2[19],radcorrinps2[20],
                                radcorrinps2[21],radcorrinps2[22],radcorrinps2[23],radcorrinps2[24],radcorrinps2[25],
                                radcorrinps2[26],radcorrinps2[27],radcorrinps2[28],radcorrinps2[29],radcorrinps2[30],
                                radcorrinps2[31],radcorrinps2[32],radcorrinps2[33],radcorrinps2[34],radcorrinps2[35],
                                radcorrinps2[36],radcorrinps2[37],radcorrinps2[38],radcorrinps2[39],radcorrinps2[40],radcorrinps2[41],
                                radcorrinps2[42],radcorrinps2[43],radcorrinps2[44])
        DEWprogcheck = False
        print("\n########## Delta_EW Results ##########\n")
        time.sleep(2)
        print('Given the submitted SLHA file, your value for the electroweak'
              + ' naturalness measure, Delta_EW, is: '
              + nstr(mpf(str(dewlist[0][1])), 8))
        time.sleep(0.25)
        print('\nThe ordered, signed contributions to Delta_EW are as follows '
              + '(decr. order): ')
        print('')
        for i in range(0, len(dewlist)):
            print(str(i + 1) + ': ' + nstr(mpf(str(dewlist[i][0])), 8) + ', '
                  + str(dewlist[i][2]))
            time.sleep(1/len(dewlist))
        input("\n##### Press Enter to continue... #####")
        # Perform Delta_HS calculation if user requested it
        if DHScalc:
            DHSinplist = deepcopy([RGE_sols[71],
                                   RGE_sols[28] - RGE_sols[71],
                                   RGE_sols[70],
                                   RGE_sols[27] - RGE_sols[70],
                                   RGE_sols[51],
                                   muQsq - muQ_GUTsq - (2 * mp.sqrt(abs(muQ_GUTsq))
                                                        * (mp.sqrt(abs(muQ_GUTsq))
                                                           - mp.sqrt(abs(muQsq)))),
                                   91.1876 ** 2,
                                   mp.power(RGE_sols[88], 2),
                                   radcorrs_at_2TeV[0],
                                   radcorrs_at_2TeV[1]])
            print ("\n########## Computing Delta_HS... ##########\n")
            myDelta_HS = Delta_HS_calc(DHSinplist[0],
                                       DHSinplist[1],
                                       DHSinplist[2],
                                       DHSinplist[3],
                                       DHSinplist[4],
                                       DHSinplist[5],
                                       DHSinplist[6],
                                       DHSinplist[7],
                                       DHSinplist[8],
                                       DHSinplist[9])
            print("Done.")
            time.sleep(1)
            print ("\n########## Delta_HS Results ##########\n")
            time.sleep(1)
            print('\nYour value for the high-scale naturalness measure,'
                  + ' Delta_HS, is: ' + nstr(mpf(str(myDelta_HS[0][0])), 8))
            time.sleep(0.25)
            print('\nThe ordered contributions to Delta_HS are as follows '
                  + '(decr. order): ')
            print('')
            for i in range(0, len(myDelta_HS)):
                print(str(i + 1) + ': ' + nstr(mpf(str(myDelta_HS[i][0])), 8) + ', '
                      + str(myDelta_HS[i][1]))
                time.sleep(1 / len(myDelta_HS))
            input("\n##### Press Enter to continue... #####")
        # Perform Delta_BG calculation if user requested it
        if DBGcalc:
            print ("\n########## Calculating Delta_BG... ##########\n")
            DBGinplist = deepcopy([modinp, precinp, poleinp, 91.1876 ** 2, myQGUT, SLHA_scale,tanb])
            myDelta_BG = Delta_BG_calc(DBGinplist[0], DBGinplist[1], DBGinplist[2],DBGinplist[3],DBGinplist[4],
                                       DBGinplist[5],DBGinplist[6],
                                       deepcopy(Q_GUT_BCs))
            print ("\n########## Delta_BG Results ##########\n")
            time.sleep(1)
            print('Your value for the Barbieri-Giudice naturalness measure,'
                  + ' Delta_BG, is: ' + nstr(mpf(str(myDelta_BG[0][0])), 8))
            time.sleep(0.25)
            print('\nThe ordered contributions to Delta_BG are as follows '
                  + '(decr. order): ')
            print('')
            for i in range(0, len(myDelta_BG)):
                print(str(i + 1) + ': ' + nstr(mpf(str(myDelta_BG[i][0])), 8) + ', '
                      + str(myDelta_BG[i][1]))
                time.sleep(1 / len(myDelta_BG))
            input("\n##### Press Enter to continue... #####")
        ##### Save Delta_EW results? #####
        checksavebool = True
        while checksavebool:
            checksave = input("\nWould you like to save these DEW results to a"
                              + " .txt file " + "(will be saved to the"
                              + " directory \n" + str(os.getcwd())
                              + "/DEW4SLHA_results/DEW)?\n" + "Enter Y to save the result or"
                              + " N to continue: ")
            timestr = time.strftime("%Y-%m-%d_%H-%M-%S")

            if checksave.lower() in ('y', 'yes'):
                # Check if DEW4SLHA_results folder exists -- if not, make it
                CHECK_resFOLDER = os.path.isdir('DEW4SLHA_results')
                if not CHECK_resFOLDER:
                    os.makedirs('DEW4SLHA_results')
                CHECK_DEWFOLDER = os.path.isdir('DEW4SLHA_results/DEW')
                if not CHECK_DEWFOLDER:
                    os.makedirs('DEW4SLHA_results/DEW')
                filenamecheck = input('\nThe default file name is '
                                      + '"current_system_time_DEW_contrib_list'
                                      + '.txt'
                                      + '", e.g., '
                                      + timestr + '_DEW_contrib_list.txt.\n'
                                      + 'Would you like to keep this name or'
                                      + ' input your own file name?\n'
                                      +  'Enter Y to keep the default file'
                                      + ' name'
                                      + ' or N to be able to input your own: ')
                if filenamecheck.lower() in ('y', 'yes'):
                    print('Given the submitted SLHA file, ' + str(direc) +
                          ', your value for the electroweak\n'
                          + 'naturalness measure, Delta_EW, is: '
                          + nstr(mpf(str(dewlist[0][1])),8),
                          file=open("DEW4SLHA_results/DEW/"
                                    + timestr + "_DEW_contrib_list.txt", "w"))
                    print('\nThe ordered contributions to Delta_EW are as'
                          + ' follows (decr. order): ',
                          file=open("DEW4SLHA_results/DEW/"
                                    + timestr + "_DEW_contrib_list.txt", "a"))
                    print('', file=open("DEW4SLHA_results/DEW/"
                                        + timestr + "_DEW_contrib_list.txt",
                                        "a"))
                    for i in range(0, len(dewlist)):
                        print(str(i + 1) + ': ' + nstr(mpf(str(dewlist[i][0])),8) + ', '
                              + str(dewlist[i][2]),
                              file=open("DEW4SLHA_results/DEW/"
                                        + timestr + "_DEW_contrib_list.txt",
                                        "a"))
                    print('\nThese results have been saved to the'
                          + ' directory \n' + str(os.getcwd())
                          + '/DEW4SLHA_results/DEW as ' + timestr
                          + '_DEW_contrib_list.txt.\n')
                    checksavebool = False
                    input("##### Press Enter to continue... #####\n")
                elif filenamecheck.lower() in ('n', 'no'):
                    newfilename = input('\nInput your desired filename with no'
                                        + ' whitespaces and \n'
                                        + 'without the .txt'
                                        + ' file '
                                        + 'extension (e.g. "my_SLHA_DEW_list"'
                                        + ' without the quotes): ')
                    print('Given the submitted SLHA file, ' + str(direc) +
                          ', your value for the electroweak\n'
                          + 'naturalness measure, Delta_EW, is: '
                          + str(dewlist[0][1]),
                          file=open("DEW4SLHA_results/DEW/"
                                    + newfilename + ".txt", "w"))
                    print('\nThe ordered contributions to Delta_EW are as'
                          + ' follows (decr. order): ',
                          file=open("DEW4SLHA_results/DEW/"
                                    + newfilename + ".txt",
                                    "a"))
                    print('', file=open("DEW4SLHA_results/DEW/"
                                        + newfilename + ".txt",
                                        "a"))
                    for i in range(0, len(dewlist)):
                        print(str(i + 1) + ': ' + nstr(mpf(str(dewlist[i][0])),8) + ', '
                              + str(dewlist[i][2]),
                              file=open("DEW4SLHA_results/DEW/"
                                        + newfilename + ".txt",
                                        "a"))
                    print('\nThese results have been saved to the'
                          + ' directory \n' + str(os.getcwd())
                          + '/DEW4SLHA_results/DEW as ' + newfilename + '.txt.\n')
                    checksavebool = False
                    input("##### Press Enter to continue... #####\n")
                else:
                    print("Invalid user input.")
                    time.sleep(1)
            else:
                print("\nOutput not saved.\n")
                checksavebool = False
                input("##### Press Enter to continue... #####\n")
        ##### Save Delta_HS results? #####
        if DHScalc:
            checksaveboolHS = True
            while checksaveboolHS:
                checksave = input("Would you like to save these Delta_HS"
                                  + " results"
                                  + " to a .txt file"
                                  + " (will be saved to the"
                                  + " directory \n" + str(os.getcwd())
                                  + "/DEW4SLHA_results/DHS)?\n"
                                  + "Enter Y to save the result or"
                                  + " N to continue: ")
                timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
                if checksave.lower() in ('y', 'yes'):
                    # Check if DEW4SLHA_results folder exists -- if not, make it
                    CHECK_resFOLDER = os.path.isdir('DEW4SLHA_results')
                    if not CHECK_resFOLDER:
                        os.makedirs('DEW4SLHA_results')
                    CHECK_DHSFOLDER = os.path.isdir('DEW4SLHA_results/DHS')
                    if not CHECK_DHSFOLDER:
                        os.makedirs('DEW4SLHA_results/DHS')
                    filenamecheck = input('\nThe default file name is '
                                          + '"current_system_time_DHS_contrib_'
                                          + 'list.txt'
                                          + '", e.g., '
                                          + timestr + '_DHS_contrib_list.txt.\n'
                                          + 'Would you like to keep this'
                                          + ' name or'
                                          + ' input your own file name?\n'
                                          +  'Enter Y to keep the'
                                          + ' default file'
                                          + ' name'
                                          + ' or N to be able to input your'
                                          + ' own: ')
                    if filenamecheck.lower() in ('y', 'yes'):
                        print('Given the submitted SLHA file, ' + str(direc) +
                              ', your value for the high-scale\n'
                              + 'naturalness measure, Delta_HS, is: '
                              + str(myDelta_HS[0][0]),
                              file=open("DEW4SLHA_results/DHS/"
                                        +timestr+"_DHS_contrib_list.txt", "w"))
                        print('\nThe ordered contributions to Delta_HS are as'
                              + ' follows (decr. order): ',
                              file=open("DEW4SLHA_results/DHS/"
                                        +timestr+"_DHS_contrib_list.txt", "a"))
                        print('', file=open("DEW4SLHA_results/DHS/"
                                            +timestr+"_DHS_contrib_list.txt",
                                            "a"))
                        for i in range(0, len(myDelta_HS)):
                            print(str(i + 1) + ': ' + nstr(mpf(str(myDelta_HS[i][0])),8)
                                  + ', ' + str(myDelta_HS[i][1]),
                                  file=open("DEW4SLHA_results/DHS/"
                                            +timestr+"_DHS_contrib_list.txt",
                                            "a"))
                        print('\nThese results have been saved to the'
                              + ' directory ' + str(os.getcwd())
                              + '/DEW4SLHA_results/DHS as ' + timestr
                              + '_DHS_contrib_list.txt.\n\n')
                        checksaveboolHS = False
                        input("##### Press Enter to continue... #####")
                    elif filenamecheck.lower() in ('n', 'no'):
                        newfilename = input('Input your desired filename with'
                                            + ' no whitespaces and without the'
                                            + ' .txt file '
                                            + 'extension (e.g. "my_SLHA'
                                            + '_DHS_list"'
                                            + ' without the quotes): ')
                        print('Given the submitted SLHA file, ' + str(direc) +
                              ', your value for the high-scale\n'
                              + 'naturalness measure, Delta_HS, is: '
                              + nstr(mpf(str(myDelta_HS[0][0])),8),
                              file=open("DEW4SLHA_results/DHS/"
                                        + newfilename + ".txt", "w"))
                        print('\nThe ordered contributions to Delta_HS are as'
                              + ' follows (decr. order): ',
                              file=open("DEW4SLHA_results/DHS/"
                                        + newfilename + ".txt", "a"))
                        print('', file=open("DEW4SLHA_results/DHS/"
                                            + newfilename + ".txt", "a"))
                        for i in range(0, len(myDelta_HS)):
                            print(str(i + 1)+': '+nstr(mpf(str(myDelta_HS[i][0])),8)+', '
                                  + str(myDelta_HS[i][1]),
                                  file=open("DEW4SLHA_results/DHS/"
                                            + newfilename + ".txt", "a"))
                        print('\nThese results have been saved to the'
                              + ' directory ' + str(os.getcwd())
                              + '/DEW4SLHA_results/DHS as '+ newfilename +'.txt.\n')
                        checksaveboolHS = False
                        input("##### Press Enter to continue... #####")
                    else:
                        print("Invalid user input.")
                        time.sleep(1)
                else:
                    print("\nOutput not saved.\n")
                    checksaveboolHS = False
                    input("##### Press Enter to continue... #####")
        ##### Save Delta_BG results? #####
        if DBGcalc:
            checksaveboolBG = True
            while checksaveboolBG:
                checksave = input("\nWould you like to save these Delta_BG "
                                  + "results to a .txt file (will be saved"
                                  + " to the directory \n" + str(os.getcwd())
                                  + "/DEW4SLHA_results/DBG)?\n"
                                  + "Enter Y to save the result or"
                                  + " N to continue: ")
                timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
                if checksave.lower() in ('y', 'yes'):
                    # Check if DEW4SLHA_results folder exists -- if not, make it
                    CHECK_resFOLDER = os.path.isdir('DEW4SLHA_results')
                    if not CHECK_resFOLDER:
                        os.makedirs('DEW4SLHA_results')
                    CHECK_DBGFOLDER = os.path.isdir('DEW4SLHA_results/DBG')
                    if not CHECK_DBGFOLDER:
                        os.makedirs('DEW4SLHA_results/DBG')
                    filenamecheck = input('\nThe default file name is '
                                          + '"current_system_time_DBG_'
                                          + 'contrib_list.txt", e.g., '
                                          + timestr + '_DBG_contrib_list.txt.\n'
                                          + 'Would you like to keep this name'
                                          + ' or input your own file name?\n'
                                          +  'Enter Y to keep the'
                                          + ' default file name'
                                          + ' or N to be able to input your'
                                          + ' own: ')
                    if filenamecheck.lower() in ('y', 'yes'):
                        print('Given the submitted SLHA file, ' + str(direc) +
                              ', your value for the Barbieri-Giudice\n'
                              + 'naturalness measure, Delta_BG, is: '
                              + nstr(mpf(str(myDelta_BG[0][0])),8),
                              file=open("DEW4SLHA_results/DBG/"
                                        + timestr + "_DBG_contrib_list.txt",
                                        "w"))
                        print('\nThe ordered contributions to Delta_BG are as'
                              + ' follows (decr. order): ',
                              file=open("DEW4SLHA_results/DBG/"
                                        + timestr + "_DBG_contrib_list.txt",
                                        "a"))
                        print('', file=open("DEW4SLHA_results/DBG/"
                                            + timestr + "_DBG_contrib_list.txt",
                                            "a"))
                        for i in range(0, len(myDelta_BG)):
                            print(str(i + 1) + ': ' + nstr(mpf(str(myDelta_BG[i][0])),8)
                                  + ', '
                                  + str(myDelta_BG[i][1]),
                                  file=open("DEW4SLHA_results/DBG/"
                                            + timestr + "_DBG_contrib_list.txt",
                                            "a"))
                        print('\nThese results have been saved to the'
                              + ' directory \n' + str(os.getcwd())
                              + '/DEW4SLHA_results/DBG as ' + timestr
                              + '_DBG_contrib_list.txt.\n')
                        checksaveboolBG = False
                    elif filenamecheck.lower() in ('n', 'no'):
                        newfilename = input('\nInput your desired filename'
                                            + ' with no'
                                            + ' whitespaces and without the '
                                            + '.txt file '
                                            + 'extension (e.g. "my_SLHA_DBG'
                                            + '_list"'
                                            + ' without the quotes): ')
                        print('Given the submitted SLHA file, ' + str(direc) +
                              ', your value for the Barbieri-Giudice\n'
                              + 'naturalness measure, Delta_BG, is: '
                              + str(myDelta_BG[0][0]),
                              file=open("DEW4SLHA_results/DBG/"
                                        + newfilename + ".txt", "w"))
                        print('\nThe ordered contributions to Delta_BG are as'
                              + ' follows (decr. order): ',
                              file=open("DEW4SLHA_results/DBG/"
                                        + newfilename + ".txt", "a"))
                        print('', file=open("DEW4SLHA_results/DBG/"
                                            + newfilename + ".txt", "a"))
                        for i in range(0, len(myDelta_BG)):
                            print(str(i + 1) + ': ' + nstr(mpf(str(myDelta_BG[i][0])),8)
                                  + ', ' + str(myDelta_BG[i][1]),
                                  file=open("DEW4SLHA_results/DBG/"
                                            + newfilename + ".txt", "a"))
                        print('\nThese results have been saved to the'
                              + ' directory ' + str(os.getcwd())
                              + '/DEW4SLHA_results/DBG as ' + newfilename
                              + '.txt.\n')
                        checksaveboolBG = False
                        input("##### Press Enter to continue... #####\n")
                    else:
                        print("Invalid user input.")
                        time.sleep(1)
                else:
                    print("\nOutput not saved.\n")
                    checksaveboolBG = False
                    input("##### Press Enter to continue... #####\n")
        ##### Try again? #####
        checkcontinue = input("\nWould you like to try again with a new SLHA "
                              + "file? Enter Y to try again or N to stop: ")
        if checkcontinue.lower() in ('y', 'yes'):
            userContinue = True
            print('\nReturning to configuration screen.\n')
            time.sleep(1)
        elif checkcontinue.lower() in ('n', 'no'):
            userContinue = False
            print('\nThank you for using DEW4SLHA.\n')
        else:
            userContinue = True
            print("\nInvalid user input. Returning to model selection"
                  + " screen.\n")
            time.sleep(1)
