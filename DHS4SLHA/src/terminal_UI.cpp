#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <thread>
#include <chrono>
#include <iomanip>
#include <limits>
#include <regex>
#include <cstdlib>
#include "mZ_numsolver.hpp"
#include "terminal_UI.hpp"
#include "MSSM_RGE_solver.hpp"
#include "MSSM_RGE_solver_with_stopfinder.hpp"
#include "DEW_calc.hpp"
#include "DBG_calc.hpp"
#include "DHS_calc.hpp"
#include "DSN_calc.hpp"
#include "radcorr_calc.hpp"
#include "slhaea.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;
using namespace SLHAea;

void clearScreen() {
    #ifdef _WIN32
    system("cls");
    #else
    system("clear");
    #endif
}

std::vector<double> beta_g1g2(const double& g1val, const double& g2val, const double& g3val,
                              const double& ytval, const double& ycval, const double& yuval,
                              const double& ybval, const double& ysval, const double& ydval,
                              const double& ytauval, const double& ymuval, const double& yeval) {
    const double loop_fac = 1.0 / (16.0 * pow(M_PIl, 2.0));
    const double loop_fac_sq = pow(loop_fac, 2.0);
    const std::vector<double> b_1 = { 33.0 / 5.0, 1.0, -3.0 };

    const std::vector<std::vector<double>> b_2 = {
        {199.0 / 25.0, 27.0 / 5.0, 88.0 / 5.0},
        {9.0 / 5.0, 25.0, 24.0},
        {11.0 / 5.0, 9.0, 14.0}
    };

    const std::vector<std::vector<double>> c_2 = {
        {26.0 / 5.0, 14.0 / 5.0, 18.0 / 5.0},
        {6.0, 6.0, 2.0},
        {4.0, 4.0, 0.0}
    };
    double dg1_dt_1 = b_1[0] * pow(g1val, 3.0);
    double dg2_dt_1 = b_1[1] * pow(g2val, 3.0);
    double dg1_dt_2 = (pow(g1val, 3.0)
        * ((b_2[0][0] * pow(g1val, 2.0))
            + (b_2[0][1] * pow(g2val, 2.0))
            + (b_2[0][2] * pow(g3val, 2.0))// Tr(Yu^2)
            - (c_2[0][0] * (pow(ytval, 2.0)
                + pow(ycval, 2.0)
                + pow(yuval, 2.0)))// end trace, begin Tr(Yd^2)
            - (c_2[0][1] * (pow(ybval, 2.0)
                + pow(ysval, 2.0)
                + pow(ydval, 2.0)))// end trace, begin Tr(Ye^2)
            - (c_2[0][2] * (pow(ytauval, 2.0)
                + pow(ymuval, 2.0)
                + pow(yeval, 2.0)))));// end trace

    double dg2_dt_2 = (pow(g2val, 3.0)
        * ((b_2[1][0] * pow(g1val, 2.0))
            + (b_2[1][1] * pow(g2val, 2.0))
            + (b_2[1][2] * pow(g3val, 2.0))// Tr(Yu^2)
            - (c_2[1][0] * (pow(ytval, 2.0)
                + pow(ycval, 2.0)
                + pow(yuval, 2.0)))// end trace, begin Tr(Yd^2)
            - (c_2[1][1] * (pow(ybval, 2.0)
                + pow(ysval, 2.0)
                + pow(ydval, 2.0)))// end trace, begin Tr(Ye^2)
            - (c_2[1][2] * (pow(ytauval, 2.0)
                + pow(ymuval, 2.0)
                + pow(yeval, 2.0)))));// end trace
    double dg1_dt = (1.0) * ((loop_fac * dg1_dt_1)
        + (loop_fac_sq * dg1_dt_2));

    double dg2_dt = (1.0) * ((loop_fac * dg2_dt_1)
        + (loop_fac_sq * dg2_dt_2));
    std::vector<double> g1g2_derivs = { dg1_dt, dg2_dt };
    return g1g2_derivs;
}

double getRenormalizationScale(const Coll& slha, const string& blockName) {
    double scale = 2000.0; // Default scale value if not found

    if (slha.find(blockName) != slha.end()) {
        for (const auto& line : slha.at(blockName)) {
            // Convert the line to a string for regex search
            string lineStr = to_string(line);
            smatch match;
            // Regex to find 'Q=' followed by a number (the scale)
            regex scaleRegex("Q= ([\\d\\.eE\\-\\+]+)");

            if (regex_search(lineStr, match, scaleRegex) && match.size() > 1) {
                // Convert the first captured group to a double
                scale = stod(match.str(1));
                break; // Assuming we only need the first occurrence
            }
        }
    }

    return scale;
}

void terminalUI() {
    std::cout << fixed << setprecision(9);
    bool userContinue = true;
    std::cout << "Welcome to DEW4SLHA, a program for computing the naturalness\n"
         << "measures Delta_EW, Delta_BG, and Delta_HS in the MSSM\n"
         << "from a SUSY Les Houches Accord (SLHA) file.\n\n"
         << "To use this program, you may select a\n"
         << "MSSM SLHA file from your choice of spectrum generator (e.g.,\n"
         << "SoftSUSY, Isajet, SPheno, FlexibleSUSY, etc.).\n"
         << "If multiple renormalization scales are present in the SLHA file,\n"
         << "then the first renormalization scale present in the SLHA file,\n"
         << "from top to bottom, is read in.\n\n"
         << "Delta_EW, Delta_BG, and Delta_HS will be evaluated at the\n"
         << " renormalization scale given by the geometric mean of the stop masses\n"
         << " as provided in the SLHA file to minimize logarithmic contributions.\n\n"
         << "Supported models for the local solvers are MSSM EFT models for\n"
         << "Delta_EW and Delta_HS, but only the CMSSM, NUHM(1,2,3,4),\n"
         << "pMSSM-19, and pMSSM-30 for Delta_BG.\n\n"
         << "Press Enter to begin." << endl;
    string input;
    getline(cin, input); // User reads intro and presses enter

    while (userContinue) {
        clearScreen();
        bool DEWprogcheck = true;
        /******************************************************************
         ********************* CALCULATION SELECTION **********************
        ******************************************************************/
        std::cout << "##############################################################\n";
        std::cout << "DEW4SLHA calculates the electroweak naturalness measure\n";
        std::cout << "Delta_EW by default.\n\n";

        // Check if user wants to compute Delta_HS as well

        bool checkcompDHS = true;
        bool DHScalc = false;
        while (checkcompDHS) {
            std::cout << "##############################################################\n";
            std::cout << "Would you like to also calculate the high-scale naturalness measure Delta_HS?\n";
            std::cout << "Enter Y for yes or N for no: ";
            string dhsCheckInp;
            getline(cin, dhsCheckInp);

            // Convert to lowercase to normalize
            transform(dhsCheckInp.begin(), dhsCheckInp.end(), dhsCheckInp.begin(),
                      [](unsigned char c) { return tolower(c); });
            if (dhsCheckInp == "n" || dhsCheckInp == "no") {
                DHScalc = false;
                checkcompDHS = false;
            } else if (dhsCheckInp == "y" || dhsCheckInp == "yes") {
                DHScalc = true;
                checkcompDHS = false;
            } else {
                std::cout << "Invalid input, please try again.\n\n";
                // Sleep for 1 second
                this_thread::sleep_for(chrono::seconds(1));
            }
        }

        // Check if user wants to compute Delta_BG as well

        bool checkcompDBG = true;
        bool DBGcalc = false;
        while (checkcompDBG) {
            std::cout << "\n##############################################################\n";
            std::cout << "Would you like to also calculate the Barbieri-Giudice naturalness measure Delta_BG?\n";
            std::cout << "Enter Y for yes or N for no: ";
            string dbgCheckInp;
            getline(cin, dbgCheckInp);

            // Convert to lowercase to normalize
            transform(dbgCheckInp.begin(), dbgCheckInp.end(), dbgCheckInp.begin(),
                      [](unsigned char c) { return tolower(c); });
            if (dbgCheckInp == "n" || dbgCheckInp == "no") {
                DBGcalc = false;
                checkcompDBG = false;
            } else if (dbgCheckInp == "y" || dbgCheckInp == "yes") {
                DBGcalc = true;
                checkcompDBG = false;
            } else {
                std::cout << "Invalid input, please try again.\n\n";
                // Sleep for 1 second
                this_thread::sleep_for(chrono::seconds(1));
            }
        }
        
        // Check if user wants to compute Delta_SN as well

        bool checkcompDSN = true;
        bool DSNcalc = false;
        while (checkcompDSN) {
            std::cout << "\n##############################################################\n";
            std::cout << "Would you like to also calculate the stringy naturalness measure Delta_SN?\n";
            std::cout << "Enter Y for yes or N for no: ";
            string dsnCheckInp;
            getline(cin, dsnCheckInp);

            // Convert to lowercase to normalize
            transform(dsnCheckInp.begin(), dsnCheckInp.end(), dsnCheckInp.begin(),
                      [](unsigned char c) { return tolower(c); });
            if (dsnCheckInp == "n" || dsnCheckInp == "no") {
                DSNcalc = false;
                checkcompDSN = false;
            } else if (dsnCheckInp == "y" || dsnCheckInp == "yes") {
                DSNcalc = true;
                checkcompDSN = false;
            } else {
                std::cout << "Invalid input, please try again.\n\n";
                // Sleep for 1 second
                this_thread::sleep_for(chrono::seconds(1));
            }
        }

        /******************************************************************
         ********************* PRECISION SELECTION ************************
        ******************************************************************/

        bool checkPrec = true;
        int printPrec = 9;
        while (checkPrec) {
            std::cout << "\n##############################################################\n";
            std::cout << "To what precision, 10^(-n), do you want your results printed?" << endl << "The default value is n=9.\n";
            std::cout << "Valid values for n are integers between 1 and 12, though higher precision (e.g., n=12) may lose accuracy due to double floating-point precision." << endl;
            std::cout << "Input the number of decimals, n, to which you want the results printed: ";
            string precCheckInp;
            getline(cin, precCheckInp);
            stringstream ss(precCheckInp);
            int n;

            if (ss >> n && !(ss >> precCheckInp)) {
                if (n >= 1 && n <= 12) {
                    printPrec = n;
                    checkPrec = false; // Input is valid, exit the loop
                    cout << "Precision level set to " << printPrec << " decimal places.\n";
                } else {
                    cerr << "Error: Please input an integer between 1 and 12.\n";
                    this_thread::sleep_for(chrono::seconds(1));
                }
            
            } else {
                cerr << "Error: Invalid input. Please input an integer between 1 and 12.\n";
            }
        }
        std::cout << fixed << setprecision(printPrec);
        
        /******************************************************************
         ********************* DBG MODEL SELECTION ************************
        ******************************************************************/
        std::cout << "\n##############################################################\n";
        int modinp = 0;
        int precinp = 0;
        if (DBGcalc) {
            std::cout << "For Delta_BG, the ``fundamental parameters'' vary from model to model.\n"
                << "For this reason, prior to entering the directory of your SLHA file, please\n"
                << "enter the model number below corresponding to your SLHA file.\n\n"
                << "Model numbers: \n"
                << "1: CMSSM/mSUGRA\n"
                << "2: NUHM1\n"
                << "3: NUHM2\n"
                << "4: NUHM3\n"
                << "5: NUHM4\n"
                << "6: pMSSM-19\n\n";
            while (true) {
                std::cout << "From the list above, input the number of the model your SLHA file corresponds to: "; 
                if ((!(cin >> modinp)) || (modinp < 1 || modinp > 6)) {
                    std::cout << "Invalid model number selected, please try again.\n\n";
                    this_thread::sleep_for(chrono::seconds(1));
                } else {
                    break;
                }
            }
            std::cout << "\n####################################################\n"
                    << "Please select the level of precision you want for the Delta_BG calculation.\n"
                    << "Below are the options: \n"
                    << "1: High precision, slowest calculation.\n"
                    << "2: Medium precision, twice as fast as high precision mode.\n"
                    << "3: Lowest precision, four times as fast as high precision mode.\n\n";

            while (true) {
                std::cout << "From the list above, input the number corresponding to the precision you want: ";
                if (!(cin >> precinp) || (precinp < 1 || precinp > 3)) {
                    std::cout << "Invalid Delta_BG precision setting selected, please try again.\n\n";
                    
                    this_thread::sleep_for(chrono::seconds(1));
                } else {
                    break; 
                }
            }
        }
        
        /******************************************************************
         ********************** DSN Configuration *************************
        ******************************************************************/
       
        int DSNcalcSelect = 0;
        int nF_input = 0;
        int nD_input = 0;
        if (DSNcalc) {
            std::cout << "\n####################################################\n"
                    << "Please select the level of precision you want for the Delta_SN calculation.\n"
                    << "Below are the options: \n"
                    << "1: Full DSN P_mu + soft terms hypervolume density measure\n"
                    << "2: P_mu (normalized width of ABDS window in mu parameter)\n\n";
            while (true) {
                std::cout << "From the list above, input the number corresponding to the precision you want: ";
                if (!(cin >> DSNcalcSelect) || (DSNcalcSelect < 1 || DSNcalcSelect > 3)) {
                    std::cout << "Invalid Delta_SN precision setting selected, please try again.\n\n";
                    
                    this_thread::sleep_for(chrono::seconds(1));
                    std::cout << "1: Full DSN P_mu + soft terms hypervolume density measure\n"
                        << "2: P_mu (normalized width of ABDS window in mu parameter)\n\n";
                    
                } else {
                    break; 
                }
            }            
            std::cout << "\n####################################################\n";
            while (true) {
                std::cout << "Please input the number of F-type SUSY breaking fields as an integer: ";
                if (!(cin >> nF_input) || (nF_input < 0) || (isnan(nF_input))) {
                    std::cout << "Invalid number of F-type fields input, please try again.\n\n";
                    
                    this_thread::sleep_for(chrono::seconds(1));
                } else {
                    break; 
                }
            }                
            std::cout << "\n####################################################\n";
            while (true) {
                std::cout << "Please input the number of D-type SUSY breaking fields as an integer: ";
                if (!(cin >> nD_input) || (nD_input < 0) || (isnan(nD_input))) {
                    std::cout << "Invalid number of D-type fields input, please try again.\n\n";
                    
                    this_thread::sleep_for(chrono::seconds(1));
                } else {
                    break; 
                }
            }       
        }

        std::cout << "\n########## Configuration Complete ##########\n";
        this_thread::sleep_for(chrono::milliseconds(1500));
        clearScreen();
        cin.clear();
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
            
        /******************************************************************
         ************************ SLHA READ-IN ****************************
        ******************************************************************/
       
        bool fileCheck = true;
        string direc;
        while (fileCheck) {
            std::cout << "Enter the full directory for your SLHA file: ";
            getline(cin, direc);
            
            // Attempt to open the file to check if it exists
            ifstream testFile(direc);
            if (testFile.good()) {
                fileCheck = false;
                testFile.close(); 
            } else {
                std::cout << "The input file cannot be found.\n"
                    << "Please try checking your spelling and try again.\n";
            }
            
        }
        this_thread::sleep_for(chrono::milliseconds(500));
        clearScreen();
        ifstream ifs(direc);
        Coll input(ifs);
        std::cout << "Analyzing submitted SLHA.\n";
        double mZ = 91.1876;

        auto getDoubleVecValue = [&](const string& block, int i, double defaultValue = 0.0) -> double {
            try {
                return to<double>(input.at(block).at(to_string(i)).at(1));
            } catch (const exception& e) {
                return defaultValue;
            }
        };

        auto getDoubleMatValue = [&](const string& block, int i, int j, double defaultValue = 0.0) -> double {
            try {
                return to<double>(input.at(block).at(i, j).at(2));
            } catch (const exception& e) {
                return defaultValue;
            }
        };
        // Higgs sector variables
        double vHiggs = getDoubleVecValue("HMIX", 3);
        double tanb = getDoubleVecValue("HMIX", 2);
        double beta = atan(tanb);
        double muQ = getDoubleVecValue("HMIX", 1);
        // Yukawas (2nd and 1st gens approximated if not present)
        double y_t = getDoubleMatValue("YU",3,3);
        double y_c = getDoubleMatValue("YU",2,2);
        if (y_c == 0.0) {
            y_c = 0.003882759826930082 * y_t;
        }
        double y_u = getDoubleMatValue("YU",1,1);
        if (y_u == 0.0) {
            y_u = 7.779613278615955e-6 * y_t;
        }
        double y_b = getDoubleMatValue("YD",3,3);
        double y_s = getDoubleMatValue("YD",2,2);
        if (y_s == 0.0) {
            y_s = 0.0206648802754076 * y_b;
        }
        double y_d = getDoubleMatValue("YD",1,1);
        if (y_d == 0.0) {
            y_d = 0.0010117174290779725 * y_b;
        }
        double y_tau = getDoubleMatValue("YE",3,3);
        double y_mu = getDoubleMatValue("YE",2,2);
        if (y_mu == 0.0) {
            y_mu = 0.05792142442492775 * y_tau;
        }
        double y_e = getDoubleMatValue("YE",1,1);
        if (y_e == 0.0) {
            y_e = 0.0002801267571260388 * y_tau;
        }
        // Gauge couplings
        double g_pr = getDoubleVecValue("GAUGE", 1);
        double g_2 = getDoubleVecValue("GAUGE", 2);
        double g_s = getDoubleVecValue("GAUGE", 3);
        // Soft trilinear couplings
        // Check for which soft trilinear block is present
        // softTrilinIdentif: 0 = "TU,TD,TE", 1 = "AU, AD, AE"
        int softTrilinIdentif = 0;
        string softTrilinUBlock, softTrilinDBlock, softTrilinEBlock;
        double test_at = getDoubleMatValue("TU", 3, 3);
        double test_ab = getDoubleMatValue("TD", 3, 3);
        double test_atau = getDoubleMatValue("TE", 3, 3);
        double a_t, a_c, a_u, a_b, a_s, a_d, a_tau, a_mu, a_e;
        if ((test_at == 0.0) && (test_ab == 0.0) && (test_atau == 0.0)) {
            softTrilinIdentif = 1;
        }
        if (softTrilinIdentif == 0) {
            softTrilinUBlock = "TU";
            softTrilinDBlock = "TD";
            softTrilinEBlock = "TE";
            a_t = 1.0;
            a_c = 1.0;
            a_u = 1.0;
            a_b = 1.0;
            a_s = 1.0;
            a_d = 1.0;
            a_tau = 1.0;
            a_mu = 1.0;
            a_e = 1.0;
        } else {
            softTrilinUBlock = "AU";
            softTrilinDBlock = "AD";
            softTrilinEBlock = "AE";
            a_t = y_t;
            a_c = y_c;
            a_u = y_u;
            a_b = y_b;
            a_s = y_s;
            a_d = y_d;
            a_tau = y_tau;
            a_mu = y_mu;
            a_e = y_e;
        }
        a_t *= getDoubleMatValue(softTrilinUBlock, 3, 3);
        a_c *= getDoubleMatValue(softTrilinUBlock, 2, 2);
        a_u *= getDoubleMatValue(softTrilinUBlock, 1, 1);
        a_b *= getDoubleMatValue(softTrilinDBlock, 3, 3);
        a_s *= getDoubleMatValue(softTrilinDBlock, 2, 2);
        a_d *= getDoubleMatValue(softTrilinDBlock, 1, 1);
        a_tau *= getDoubleMatValue(softTrilinEBlock, 3, 3);
        a_mu *= getDoubleMatValue(softTrilinEBlock, 2, 2);
        a_e *= getDoubleMatValue(softTrilinEBlock, 1, 1);
        // Gaugino masses
        double my_M1, my_M2, my_M3;
        my_M1 = getDoubleVecValue("MSOFT", 1);
        my_M2 = getDoubleVecValue("MSOFT", 2);
        my_M3 = getDoubleVecValue("MSOFT", 3);
        // Soft Higgs masses
        double mHusq, mHdsq;
        mHusq = getDoubleVecValue("MSOFT", 22);
        mHdsq = getDoubleVecValue("MSOFT", 21);
        // Soft scalar masses
        // Check for which soft mass block(s) is (are) present
        // softMassIdentif: 0 = "MSQ2,MSU2,MSD2,MSL2,MSE2", 1 = "MSOFT"x5
        double test_mQ3sq = getDoubleMatValue("MSQ2", 3, 3);
        double test_mU3sq = getDoubleMatValue("MSU2", 3, 3);
        double test_mE3sq = getDoubleMatValue("MSE2", 3, 3);
        double mQ3sq, mQ2sq, mQ1sq;
        double mL3sq, mL2sq, mL1sq;
        double mU3sq, mU2sq, mU1sq;
        double mD3sq, mD2sq, mD1sq;
        double mE3sq, mE2sq, mE1sq;
        if ((test_mQ3sq == 0.0) && (test_mU3sq == 0.0) && (test_mE3sq == 0.0)) {
            mQ3sq = pow(getDoubleVecValue("MSOFT", 43), 2.0);
            mQ2sq = pow(getDoubleVecValue("MSOFT", 42), 2.0);
            mQ1sq = pow(getDoubleVecValue("MSOFT", 41), 2.0);
            mL3sq = pow(getDoubleVecValue("MSOFT", 33), 2.0);
            mL2sq = pow(getDoubleVecValue("MSOFT", 32), 2.0);
            mL1sq = pow(getDoubleVecValue("MSOFT", 31), 2.0);
            mU3sq = pow(getDoubleVecValue("MSOFT", 46), 2.0);
            mU2sq = pow(getDoubleVecValue("MSOFT", 45), 2.0);
            mU1sq = pow(getDoubleVecValue("MSOFT", 44), 2.0);
            mD3sq = pow(getDoubleVecValue("MSOFT", 49), 2.0);
            mD2sq = pow(getDoubleVecValue("MSOFT", 48), 2.0);
            mD1sq = pow(getDoubleVecValue("MSOFT", 47), 2.0);
            mE3sq = pow(getDoubleVecValue("MSOFT", 36), 2.0);
            mE2sq = pow(getDoubleVecValue("MSOFT", 35), 2.0);
            mE1sq = pow(getDoubleVecValue("MSOFT", 34), 2.0);
        } else {
            mQ3sq = getDoubleMatValue("MSQ2", 3, 3);
            mQ2sq = getDoubleMatValue("MSQ2", 2, 2);
            mQ1sq = getDoubleMatValue("MSQ2", 1, 1);
            mL3sq = getDoubleMatValue("MSL2", 3, 3);
            mL2sq = getDoubleMatValue("MSL2", 2, 2);
            mL1sq = getDoubleMatValue("MSL2", 1, 1);
            mU3sq = getDoubleMatValue("MSU2", 3, 3);
            mU2sq = getDoubleMatValue("MSU2", 2, 2);
            mU1sq = getDoubleMatValue("MSU2", 1, 1);
            mD3sq = getDoubleMatValue("MSD2", 3, 3);
            mD2sq = getDoubleMatValue("MSD2", 2, 2);
            mD1sq = getDoubleMatValue("MSD2", 1, 1);
            mE3sq = getDoubleMatValue("MSE2", 3, 3);
            mE2sq = getDoubleMatValue("MSE2", 2, 2);
            mE1sq = getDoubleMatValue("MSE2", 1, 1);
        }
        double SLHA_scale = getRenormalizationScale(input, "GAUGE");
        std::cout << "Q(SLHA) = " << SLHA_scale << endl;
        std::cout << "SLHA parameters read in." << endl;
        /* Use 2-loop MSSM RGEs to evolve results to a renormalization scale of 
           Q = sqrt(mst1 * mst2) if the submitted SLHA file is not currently at that scale.
           This is so evaluations of the naturalness measures are always performed
           at a scale that somewhat minimizes logs and to avoid badly organized SLHA files.
           ///////////////////////////////////////////////////////////////////////
           The result is then run to a high scale of 3*10^16 GeV, and an approximate GUT
           scale is chosen at the value where g1(Q) is closest to g2(Q) over the scanned
           renormalization scales. This is done by iterating and adjusting GUT thresholds to 
           account for log corrections at that scale. 
           ///////////////////////////////////////////////////////////////////////
           This running to the GUT scale is used in the evaluations of Delta_HS and Delta_BG.
           Compute loop-level soft Higgs bilinear parameter b=B*mu at SUSY scale for RGE BC
           after. 
        */        

        /******************************************************************
         ***************** ESTABLISH WEAK-SCALE VALUES ********************
         ******************************************************************/

        vector<double> mySLHABCs;
        mySLHABCs = {sqrt(5.0 / 3.0) * g_pr, g_2, g_s, my_M1, my_M2, my_M3,
                     muQ, y_t, y_c, y_u, y_b, y_s, y_d, y_tau, y_mu, y_e,
                     a_t, a_c, a_u, a_b, a_s, a_d, a_tau, a_mu, a_e,
                     mHusq, mHdsq, mQ1sq, mQ2sq, mQ3sq, mL1sq, mL2sq,
                     mL3sq, mU1sq, mU2sq, mU3sq, mD1sq, mD2sq, mD3sq,
                     mE1sq, mE2sq, mE3sq, 0.0, tanb};
        vector<double> dummyrun = solveODEs(mySLHABCs, log(SLHA_scale), log(1.0e12), copysign(1.0e-6, (log(1.0e12 / SLHA_scale))));
        // SUSY scale equal to Q = sqrt(mt1(Q) * mt2(Q))
        double tempT_target = log(250.0); 
        vector<RGEStruct> SUSYscale_struct = solveODEstoMSUSY(dummyrun, log(1.0e12), -1.0e-6, tempT_target, 91.1876 * 91.1876);

        double SLHAQSUSY = exp(SUSYscale_struct[0].SUSYscale_eval);
        std::cout << "Q(SUSY) = " << SLHAQSUSY << endl;
        vector<double> first_SUSY_BCs = solveODEs(mySLHABCs, log(SLHA_scale), log(SLHAQSUSY), copysign(1.0e-6, (SLHAQSUSY - SLHA_scale)));
        vector<double> first_radcorrs = radcorr_calc(first_SUSY_BCs, SLHAQSUSY, 91.1876 * 91.1876);
        tanb = first_SUSY_BCs[43];
        mHdsq = first_SUSY_BCs[26];
        mHusq = first_SUSY_BCs[25];
        muQ = first_SUSY_BCs[6];
        // Converge a value of mu that gives mZ=91.1876 GeV
        double lsqtol = 1.0e-8;
        double curr_iter_lsq = 100.0;
        double muQsq = muQ * muQ;
        double newmuQsq = muQsq;
        while (curr_iter_lsq > lsqtol) {
            newmuQsq = ((mHdsq + first_radcorrs[1] - ((mHusq + first_radcorrs[0]) * pow(tanb, 2.0))) / (pow(tanb, 2.0) - 1.0)) - (91.1876 * 91.1876 / 2.0);
            first_SUSY_BCs[6] = copysign(sqrt(abs(newmuQsq)), muQ);
            first_radcorrs = radcorr_calc(first_SUSY_BCs, SLHAQSUSY, 91.1876 * 91.1876);
            curr_iter_lsq = pow((muQsq) - (newmuQsq), 2.0);
            muQsq = newmuQsq;
        }
        double currentmZ2 = ((2.0 * ((mHdsq + first_radcorrs[1] - ((mHusq + first_radcorrs[0]) * pow(tanb, 2.0))) / (pow(tanb, 2.0) - 1.0)))
                             - (2.0 * muQsq));
        cout << "first mZ: " << sqrt(currentmZ2) << endl;
        double getmZ2_value = getmZ2(first_SUSY_BCs, SLHAQSUSY, 91.1876 * 91.1876);
        // cout << "mZ value from routine = " << copysign(sqrt(abs(getmZ2_value)), getmZ2_value) << endl;

        // Now we calculate the value of b=B*mu coming from this SLHA point. 
        double BmuSLHA = sin(2.0 * beta) * (mHusq + first_radcorrs[0] + mHdsq + first_radcorrs[1] + (2.0 * muQsq)) / 2.0;
        first_SUSY_BCs[42] = BmuSLHA;
        std::cout << "Weak scale parameters established." << endl;
        this_thread::sleep_for(chrono::seconds(1));
        // std::cout << "First weak scale BCs: " << endl;
        // for (double value : first_SUSY_BCs) {
        //     cout << value << endl;
        // }

        /******************************************************************
         ******************** ESTABLISH GUT VALUES ************************
         ******************************************************************/
        
        // Get GUT scale now
        curr_iter_lsq = 100.0;
        vector<double> first_GUT_BCs = solveODEs(first_SUSY_BCs, log(SLHAQSUSY), log(3.0e16), 1.0e-6);
        vector<double> currbetag1g2GUT = beta_g1g2(first_GUT_BCs[0], first_GUT_BCs[1], first_GUT_BCs[2], first_GUT_BCs[7], first_GUT_BCs[8], first_GUT_BCs[9],
                                                   first_GUT_BCs[10], first_GUT_BCs[11], first_GUT_BCs[12], first_GUT_BCs[13], first_GUT_BCs[14], first_GUT_BCs[15]);
        double curr_iter_QGUT = log(3.0e16 * exp((first_GUT_BCs[1] - first_GUT_BCs[0]) / (currbetag1g2GUT[0] - currbetag1g2GUT[1])));
        double new_QGUT = curr_iter_QGUT;
        while (curr_iter_lsq > lsqtol) {
            first_GUT_BCs = solveODEs(first_SUSY_BCs, log(SLHAQSUSY), curr_iter_QGUT, 1.0e-6);
            new_QGUT = log(exp(curr_iter_QGUT) * exp((first_GUT_BCs[1] - first_GUT_BCs[0]) / (currbetag1g2GUT[0] - currbetag1g2GUT[1])));
            curr_iter_lsq = pow((1.0 - (new_QGUT / curr_iter_QGUT)), 2.0);
            curr_iter_QGUT = new_QGUT;
        }
        std::cout << "GUT scale = " << curr_iter_QGUT << endl;
        std::cout << "GUT BCs: " << endl;
        for (double value : first_GUT_BCs) {
            std::cout << value << endl;
        }

        /******************************************************************
         ********************* COMPUTE DEW VALUES *************************
         ******************************************************************/

        std::cout << "\n########## Computing Delta_EW... ##########\n" << endl;
        vector<LabeledValue> dewlist = DEW_calc(first_SUSY_BCs, SLHAQSUSY);
        std::cout << "\n########## Delta_EW Results ##########\n";
        this_thread::sleep_for(chrono::milliseconds(1500));
        std::cout << "Given the submitted SLHA file, your value for the electroweak naturalness measure"
             << ", Delta_EW, is: " << dewlist[0].value;
        this_thread::sleep_for(chrono::milliseconds(250));
        std::cout << "\nThe ordered, signed contributions to Delta_EW are as follows (decr. order):\n";
        for (size_t i = 0; i < dewlist.size(); ++i) {
            std::cout << (i + 1) << ": " << dewlist[i].value << ", " << dewlist[i].label << endl;
            this_thread::sleep_for(chrono::milliseconds(static_cast<int>(1000 / dewlist.size())));
        }
        
        string continueinput;
        std::cout << "\n##### Press Enter to continue... #####";
        getline(cin, continueinput); // User presses enter to continue.

        /******************************************************************
         ********************* COMPUTE DHS VALUES *************************
         ******************************************************************/

        // Perform Delta_HS calculation if user requested it
        if (DHScalc) {
            std::cout << "\n########## Computing Delta_HS... ##########\n" << endl;
            vector<LabeledValueHS> dhslist = DHS_calc(first_GUT_BCs[26], first_SUSY_BCs[26] - first_GUT_BCs[26],
                                                      first_GUT_BCs[25], first_SUSY_BCs[25] - first_GUT_BCs[25],
                                                      pow(first_GUT_BCs[6], 2.0),
                                                      pow(first_SUSY_BCs[6], 2.0) - pow(first_GUT_BCs[6], 2.0),
                                                      91.1876 * 91.1876, first_SUSY_BCs[43] * first_SUSY_BCs[43], first_radcorrs[0], first_radcorrs[1]);

            this_thread::sleep_for(chrono::seconds(1));
            std::cout << "\n########## Delta_HS Results ##########\n";
            this_thread::sleep_for(chrono::seconds(1));
            std::cout << "Your value for the high-scale naturalness measure, Delta_HS, is: "
                 << dhslist[0].value;
            this_thread::sleep_for(chrono::milliseconds(250));
            std::cout << "\nThe ordered, signed contributions to Delta_HS are as follows (decr. order):\n";
            for (size_t i = 0; i < dhslist.size(); ++i) {
                std::cout << (i + 1) << ": " << dhslist[i].value << ", " << dhslist[i].label << endl;
                this_thread::sleep_for(chrono::milliseconds(static_cast<int>(1000 / dhslist.size())));
            }
        
            string continueinputHS;
            std::cout << "\n##### Press Enter to continue... #####";
            getline(cin, continueinputHS); // User presses enter to continue.
        }

        /******************************************************************
         ********************* COMPUTE DBG VALUES *************************
         ******************************************************************/

        if (DBGcalc) {
            double logQSUSY = log(SLHAQSUSY);
            std::cout << "\n########## Computing Delta_BG... ##########\n" << endl;
            std::cout << "(This can take a while...)\n";
            vector<LabeledValueBG> myDBGlist = DBG_calc(modinp, precinp, curr_iter_QGUT,
                                                        logQSUSY, tanb, first_GUT_BCs, currentmZ2);
            this_thread::sleep_for(chrono::seconds(1));
            std::cout << "\n########## Delta_BG Results ##########\n";
            this_thread::sleep_for(chrono::seconds(1));
            std::cout << "Your value for the Barbieri-Giudice naturalness measure, Delta_BG, is: "
                 << myDBGlist[0].value;
            this_thread::sleep_for(chrono::milliseconds(250));
            std::cout << "\nThe ordered, signed contributions to Delta_BG are as follows (decr. order):\n";
            for (size_t i = 0; i < myDBGlist.size(); ++i) {
                std::cout << (i + 1) << ": " << myDBGlist[i].value << ", " << myDBGlist[i].label << endl;
                this_thread::sleep_for(chrono::milliseconds(static_cast<int>(1000 / myDBGlist.size())));
            }
        
            string continueinputBG;
            std::cout << "\n##### Press Enter to continue... #####";
            getline(cin, continueinputBG); // User presses enter to continue.
        
        }
     
        /******************************************************************
         ********************* COMPUTE DSN VALUES *************************
         ******************************************************************/

        if (DSNcalc) {
            double logQSUSY = log(SLHAQSUSY);
            std::vector<DSNLabeledValue> myDSNlist = DSN_calc(DSNcalcSelect, first_SUSY_BCs, getmZ2_value, logQSUSY, curr_iter_QGUT, nF_input, nD_input);
            double totalDSN = 0.0;
            for (const auto& item : myDSNlist) {
                totalDSN += item.value;
            }
            std::cout << "\n########## Delta_SN Results ##########\n";
            std::cout << "Your value for the stringy naturalness measure, Delta_SN, is: "
                 << totalDSN;
            this_thread::sleep_for(chrono::milliseconds(250));
            std::cout << "\nThe ordered contributions to Delta_SN are as follows (decr. order):\n";
            for (size_t i = 0; i < myDSNlist.size(); ++i) {
                std::cout << (i + 1) << ": " << myDSNlist[i].value << ", " << myDSNlist[i].label << endl;
                this_thread::sleep_for(chrono::milliseconds(static_cast<int>(1000 / myDSNlist.size())));
            }
        
            string continueinputSN;
            std::cout << "\n##### Press Enter to continue... #####";
            getline(cin, continueinputSN); // User presses enter to continue.
        }
        
        // Try again?
        string checkcontinue;
        std::cout << "Would you like to try again with a new SLHA file? Enter Y to try again or N to stop: ";
        getline(cin, checkcontinue);
        std::transform(checkcontinue.begin(), checkcontinue.end(), checkcontinue.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        if (checkcontinue == "y" || checkcontinue == "yes") {
            userContinue = true;
            std::cout << "\nReturning to configuration screen.\n";
            std::this_thread::sleep_for(std::chrono::seconds(1));
        } else if (checkcontinue == "n" || checkcontinue == "no") {
            userContinue = false;
            std::cout << "\nThank you for using natLHA.\n";
            break; 
        } else {
            userContinue = true;
            std::cout << "\nInvalid user input. Returning to configuration screen.\n";
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        if (!userContinue) {
            break;
        }
    }
}

int main() {
    terminalUI();
    return 0;
}