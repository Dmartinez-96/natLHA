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
#include <filesystem>
#include <ctime>
#include <boost/multiprecision/mpfr.hpp>
#include "mZ_numsolver.hpp"
#include "terminal_UI.hpp"
#include "MSSM_RGE_solver.hpp"
#include "MSSM_RGE_solver_with_stopfinder.hpp"
#include "DEW_calc.hpp"
#include "radcorr_calc.hpp"
#include "slhaea.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//using namespace std;
using namespace SLHAea;
using namespace boost::multiprecision;
typedef number<mpfr_float_backend<50>> high_prec_float;  // 50 decimal digits of precision
namespace fs = std::filesystem;

std::string getCurrentTimeFormatted() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm now_tm = *std::localtime(&now_time);
    char buffer[20];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d_%H-%M-%S", &now_tm);
    return buffer;
}

void saveDEWResults(const std::vector<LabeledValue>& dewlist, const std::string& directory, const std::string& filename, const int& printprec) {
    std::ofstream outFile(directory + "/" + filename, std::ios::out);
    outFile << "Given the submitted SLHA file, " << directory << ", your value for the electroweak\n"
            << "naturalness measure, Delta_EW, is: " << std::fixed << std::setprecision(printprec) << dewlist[0].value << std::endl;
    outFile << "\nThe ordered contributions to Delta_EW are as follows (decr. order): \n\n";
    for (const auto& item : dewlist) {
        outFile << item.label << ": " << std::fixed << std::setprecision(printprec) << item.value << std::endl;
    }
    outFile.close();
    std::cout << "\nThese results have been saved to the directory \n" << directory << " as " << filename << ".\n";
}

void clearScreen() {
    #ifdef _WIN32
    system("cls");
    #else
    system("clear");
    #endif
}

std::vector<high_prec_float> beta_g1g2(const high_prec_float& g1val, const high_prec_float& g2val, const high_prec_float& g3val,
                              const high_prec_float& ytval, const high_prec_float& ycval, const high_prec_float& yuval,
                              const high_prec_float& ybval, const high_prec_float& ysval, const high_prec_float& ydval,
                              const high_prec_float& ytauval, const high_prec_float& ymuval, const high_prec_float& yeval) {
    const high_prec_float loop_fac = 1.0 / (16.0 * pow(M_PIl, 2.0));
    const high_prec_float loop_fac_sq = pow(loop_fac, 2.0);
    const std::vector<high_prec_float> b_1 = { 33.0 / 5.0, 1.0, -3.0 };

    const std::vector<std::vector<high_prec_float>> b_2 = {
        {199.0 / 25.0, 27.0 / 5.0, 88.0 / 5.0},
        {9.0 / 5.0, 25.0, 24.0},
        {11.0 / 5.0, 9.0, 14.0}
    };

    const std::vector<std::vector<high_prec_float>> c_2 = {
        {26.0 / 5.0, 14.0 / 5.0, 18.0 / 5.0},
        {6.0, 6.0, 2.0},
        {4.0, 4.0, 0.0}
    };
    high_prec_float dg1_dt_1 = b_1[0] * pow(g1val, 3.0);
    high_prec_float dg2_dt_1 = b_1[1] * pow(g2val, 3.0);
    high_prec_float dg1_dt_2 = (pow(g1val, 3.0)
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

    high_prec_float dg2_dt_2 = (pow(g2val, 3.0)
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
    high_prec_float dg1_dt = (1.0) * ((loop_fac * dg1_dt_1)
        + (loop_fac_sq * dg1_dt_2));

    high_prec_float dg2_dt = (1.0) * ((loop_fac * dg2_dt_1)
        + (loop_fac_sq * dg2_dt_2));
    std::vector<high_prec_float> g1g2_derivs = { dg1_dt, dg2_dt };
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
    std::cout << "Welcome to DEW4SLHA, a program for computing the naturalness measure Delta_EW (electroweak naturalness) in the MSSM\n"
         << "from a SUSY Les Houches Accord (SLHA) file.\n\n"
         << "To use this program, you may select a\n"
         << "MSSM SLHA file from your choice of spectrum generator (e.g.,\n"
         << "SoftSUSY, Isajet, SPheno, FlexibleSUSY, etc.).\n"
         << "If multiple renormalization scales are present in the SLHA file,\n"
         << "then the first renormalization scale present in the SLHA file,\n"
         << "from top to bottom, is read in.\n\n"
         << "Delta_EW will be evaluated at the\n"
         << "renormalization scale given by the geometric mean of the stop masses\n"
         << "as calculated at tree level from the SLHA file to minimize logarithmic contributions.\n\n"
         << "Supported models for the local solvers are MSSM EFT models.\n\n"
         << "Press Enter to begin." << endl;
    string input;
    getline(cin, input); // User reads intro and presses enter

    while (userContinue) {
        clearScreen();

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
        
        std::cout << "\n########## Configuration Complete ##########\n";
        this_thread::sleep_for(chrono::milliseconds(1500));
        clearScreen();
            
        /******************************************************************
         ************************ SLHA READ-IN ****************************
        ******************************************************************/
       
        bool fileCheck = true;
        string direc;
        while (fileCheck) {
            std::cout << "Enter the full directory for your SLHA file: ";
            getline(cin, direc);

            fs::path filePath(direc);

            // Check if the path exists and is a file
            if (fs::exists(filePath) && fs::is_regular_file(filePath)) {
                std::ifstream testFile(direc);
                if (testFile.good()) {
                    fileCheck = false;
                    testFile.close();
                } else {
                    std::cout << "The input file cannot be opened.\n"
                            << "Please check your permissions and try again.\n";
                }
            } else if (fs::exists(filePath) && fs::is_directory(filePath)) {
                std::cout << "The path you entered is a directory, not a file.\n"
                        << "Please enter a valid file path.\n";
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
        high_prec_float mZ = 91.1876;

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
        high_prec_float vHiggs = high_prec_float(getDoubleVecValue("HMIX", 3));
        high_prec_float tanb = high_prec_float(getDoubleVecValue("HMIX", 2));
        high_prec_float beta = atan(tanb);
        high_prec_float muQ = high_prec_float(getDoubleVecValue("HMIX", 1));
        // Yukawas (2nd and 1st gens approximated if not present)
        high_prec_float y_t = high_prec_float(getDoubleMatValue("YU",3,3));
        high_prec_float y_c = high_prec_float(getDoubleMatValue("YU",2,2));
        if (y_c == 0.0) {
            y_c = high_prec_float(0.003882759826930082) * y_t;
        }
        high_prec_float y_u = high_prec_float(getDoubleMatValue("YU",1,1));
        if (y_u == 0.0) {
            y_u = high_prec_float(7.779613278615955e-6) * y_t;
        }
        high_prec_float y_b = high_prec_float(getDoubleMatValue("YD",3,3));
        high_prec_float y_s = high_prec_float(getDoubleMatValue("YD",2,2));
        if (y_s == 0.0) {
            y_s = high_prec_float(0.0206648802754076) * y_b;
        }
        high_prec_float y_d = high_prec_float(getDoubleMatValue("YD",1,1));
        if (y_d == 0.0) {
            y_d = high_prec_float(0.0010117174290779725) * y_b;
        }
        high_prec_float y_tau = high_prec_float(getDoubleMatValue("YE",3,3));
        high_prec_float y_mu = high_prec_float(getDoubleMatValue("YE",2,2));
        if (y_mu == 0.0) {
            y_mu = high_prec_float(0.05792142442492775) * y_tau;
        }
        high_prec_float y_e = high_prec_float(getDoubleMatValue("YE",1,1));
        if (y_e == 0.0) {
            y_e = high_prec_float(0.0002801267571260388) * y_tau;
        }
        // Gauge couplings
        high_prec_float g_pr = high_prec_float(getDoubleVecValue("GAUGE", 1));
        high_prec_float g_2 = high_prec_float(getDoubleVecValue("GAUGE", 2));
        high_prec_float g_s = high_prec_float(getDoubleVecValue("GAUGE", 3));
        // Soft trilinear couplings
        // Check for which soft trilinear block is present
        // softTrilinIdentif: 0 = "TU,TD,TE", 1 = "AU, AD, AE"
        int softTrilinIdentif = 0;
        string softTrilinUBlock, softTrilinDBlock, softTrilinEBlock;
        high_prec_float test_at = high_prec_float(getDoubleMatValue("TU", 3, 3));
        high_prec_float test_ab = high_prec_float(getDoubleMatValue("TD", 3, 3));
        high_prec_float test_atau = high_prec_float(getDoubleMatValue("TE", 3, 3));
        high_prec_float a_t, a_c, a_u, a_b, a_s, a_d, a_tau, a_mu, a_e;
        if ((test_at == 0.0) && (test_ab == 0.0) && (test_atau == 0.0)) {
            softTrilinIdentif = 1;
        }
        if (softTrilinIdentif == 0) {
            softTrilinUBlock = "TU";
            softTrilinDBlock = "TD";
            softTrilinEBlock = "TE";
            a_t = high_prec_float(1.0);
            a_c = high_prec_float(1.0);
            a_u = high_prec_float(1.0);
            a_b = high_prec_float(1.0);
            a_s = high_prec_float(1.0);
            a_d = high_prec_float(1.0);
            a_tau = high_prec_float(1.0);
            a_mu = high_prec_float(1.0);
            a_e = high_prec_float(1.0);
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
        a_t *= high_prec_float(getDoubleMatValue(softTrilinUBlock, 3, 3));
        a_c *= high_prec_float(getDoubleMatValue(softTrilinUBlock, 2, 2));
        a_u *= high_prec_float(getDoubleMatValue(softTrilinUBlock, 1, 1));
        a_b *= high_prec_float(getDoubleMatValue(softTrilinDBlock, 3, 3));
        a_s *= high_prec_float(getDoubleMatValue(softTrilinDBlock, 2, 2));
        a_d *= high_prec_float(getDoubleMatValue(softTrilinDBlock, 1, 1));
        a_tau *= high_prec_float(getDoubleMatValue(softTrilinEBlock, 3, 3));
        a_mu *= high_prec_float(getDoubleMatValue(softTrilinEBlock, 2, 2));
        a_e *= high_prec_float(getDoubleMatValue(softTrilinEBlock, 1, 1));
        // Gaugino masses
        high_prec_float my_M1, my_M2, my_M3;
        my_M1 = high_prec_float(getDoubleVecValue("MSOFT", 1));
        my_M2 = high_prec_float(getDoubleVecValue("MSOFT", 2));
        my_M3 = high_prec_float(getDoubleVecValue("MSOFT", 3));
        // Soft Higgs masses
        high_prec_float mHusq, mHdsq;
        mHusq = high_prec_float(getDoubleVecValue("MSOFT", 22));
        mHdsq = high_prec_float(getDoubleVecValue("MSOFT", 21));
        // Soft scalar masses
        // Check for which soft mass block(s) is (are) present
        // softMassIdentif: 0 = "MSQ2,MSU2,MSD2,MSL2,MSE2", 1 = "MSOFT"x5
        high_prec_float test_mQ3sq = high_prec_float(getDoubleMatValue("MSQ2", 3, 3));
        high_prec_float test_mU3sq = high_prec_float(getDoubleMatValue("MSU2", 3, 3));
        high_prec_float test_mE3sq = high_prec_float(getDoubleMatValue("MSE2", 3, 3));
        high_prec_float mQ3sq, mQ2sq, mQ1sq;
        high_prec_float mL3sq, mL2sq, mL1sq;
        high_prec_float mU3sq, mU2sq, mU1sq;
        high_prec_float mD3sq, mD2sq, mD1sq;
        high_prec_float mE3sq, mE2sq, mE1sq;
        if ((test_mQ3sq == 0.0) && (test_mU3sq == 0.0) && (test_mE3sq == 0.0)) {
            mQ3sq = pow(high_prec_float(getDoubleVecValue("MSOFT", 43)), 2.0);
            mQ2sq = pow(high_prec_float(getDoubleVecValue("MSOFT", 42)), 2.0);
            mQ1sq = pow(high_prec_float(getDoubleVecValue("MSOFT", 41)), 2.0);
            mL3sq = pow(high_prec_float(getDoubleVecValue("MSOFT", 33)), 2.0);
            mL2sq = pow(high_prec_float(getDoubleVecValue("MSOFT", 32)), 2.0);
            mL1sq = pow(high_prec_float(getDoubleVecValue("MSOFT", 31)), 2.0);
            mU3sq = pow(high_prec_float(getDoubleVecValue("MSOFT", 46)), 2.0);
            mU2sq = pow(high_prec_float(getDoubleVecValue("MSOFT", 45)), 2.0);
            mU1sq = pow(high_prec_float(getDoubleVecValue("MSOFT", 44)), 2.0);
            mD3sq = pow(high_prec_float(getDoubleVecValue("MSOFT", 49)), 2.0);
            mD2sq = pow(high_prec_float(getDoubleVecValue("MSOFT", 48)), 2.0);
            mD1sq = pow(high_prec_float(getDoubleVecValue("MSOFT", 47)), 2.0);
            mE3sq = pow(high_prec_float(getDoubleVecValue("MSOFT", 36)), 2.0);
            mE2sq = pow(high_prec_float(getDoubleVecValue("MSOFT", 35)), 2.0);
            mE1sq = pow(high_prec_float(getDoubleVecValue("MSOFT", 34)), 2.0);
        } else {
            mQ3sq = high_prec_float(getDoubleMatValue("MSQ2", 3, 3));
            mQ2sq = high_prec_float(getDoubleMatValue("MSQ2", 2, 2));
            mQ1sq = high_prec_float(getDoubleMatValue("MSQ2", 1, 1));
            mL3sq = high_prec_float(getDoubleMatValue("MSL2", 3, 3));
            mL2sq = high_prec_float(getDoubleMatValue("MSL2", 2, 2));
            mL1sq = high_prec_float(getDoubleMatValue("MSL2", 1, 1));
            mU3sq = high_prec_float(getDoubleMatValue("MSU2", 3, 3));
            mU2sq = high_prec_float(getDoubleMatValue("MSU2", 2, 2));
            mU1sq = high_prec_float(getDoubleMatValue("MSU2", 1, 1));
            mD3sq = high_prec_float(getDoubleMatValue("MSD2", 3, 3));
            mD2sq = high_prec_float(getDoubleMatValue("MSD2", 2, 2));
            mD1sq = high_prec_float(getDoubleMatValue("MSD2", 1, 1));
            mE3sq = high_prec_float(getDoubleMatValue("MSE2", 3, 3));
            mE2sq = high_prec_float(getDoubleMatValue("MSE2", 2, 2));
            mE1sq = high_prec_float(getDoubleMatValue("MSE2", 1, 1));
        }
        double SLHA_scale_dbl = getRenormalizationScale(input, "GAUGE");
        high_prec_float SLHA_scale = high_prec_float(SLHA_scale_dbl);
        // std::cout << "Q(SLHA) = " << SLHA_scale << endl;
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

        std::vector<high_prec_float> mySLHABCs;
        mySLHABCs = {sqrt(5.0 / 3.0) * g_pr, g_2, g_s, my_M1, my_M2, my_M3,
                     muQ, y_t, y_c, y_u, y_b, y_s, y_d, y_tau, y_mu, y_e,
                     a_t, a_c, a_u, a_b, a_s, a_d, a_tau, a_mu, a_e,
                     mHusq, mHdsq, mQ1sq, mQ2sq, mQ3sq, mL1sq, mL2sq,
                     mL3sq, mU1sq, mU2sq, mU3sq, mD1sq, mD2sq, mD3sq,
                     mE1sq, mE2sq, mE3sq, 0.0, tanb};
        std::vector<double> mySLHABCs_dbl;
        // std::cout << "SLHA BCs: " << std::endl;
        for (const auto& value : mySLHABCs) {
            mySLHABCs_dbl.push_back(double(value));
            // std::cout << value << std::endl;
        }
        std::vector<double> dummyrun = solveODEs(mySLHABCs_dbl, log(SLHA_scale_dbl), log(1.0e12), std::copysign(1.0e-6, (log(1.0e12 / SLHA_scale_dbl))));
        // SUSY scale equal to Q = sqrt(mt1(Q) * mt2(Q))
        double tempT_target = log(250.0); 
        std::vector<RGEStruct> SUSYscale_struct = solveODEstoMSUSY(dummyrun, log(1.0e12), -1.0e-6, tempT_target, 91.1876 * 91.1876);

        high_prec_float SLHAQSUSY = exp(high_prec_float(SUSYscale_struct[0].SUSYscale_eval));
        double SLHAQSUSY_dbl = double(SLHAQSUSY);
        std::vector<double> first_SUSY_BCs_dbl = solveODEs(mySLHABCs_dbl, log(SLHA_scale_dbl), log(SLHAQSUSY_dbl), copysign(1.0e-6, (SLHAQSUSY_dbl - SLHA_scale_dbl)));
        std::vector<high_prec_float> first_SUSY_BCs;
        for (const auto& value : first_SUSY_BCs_dbl) {
            first_SUSY_BCs.push_back(high_prec_float(value));
        }
        // std::cout << "SUSY BCs: " << std::endl;
        // for (const auto& value : first_SUSY_BCs) {
        //     std::cout << value << endl;
        // }
        std::vector<high_prec_float> first_radcorrs = radcorr_calc(first_SUSY_BCs, SLHAQSUSY, high_prec_float(91.1876 * 91.1876));
        // std::cout << "Sigma_u = " << first_radcorrs[0] << std::endl << "Sigma_d = " << first_radcorrs[1] << std::endl;
        tanb = first_SUSY_BCs[43];
        mHdsq = first_SUSY_BCs[26];
        mHusq = first_SUSY_BCs[25];
        muQ = first_SUSY_BCs[6];
        // Converge a value of mu that gives mZ=91.1876 GeV
        high_prec_float lsqtol = high_prec_float(1.0) / high_prec_float(1000000000000.0);
        high_prec_float curr_iter_lsq = high_prec_float(100.0);
        high_prec_float muQsq = muQ * muQ;
        high_prec_float newmuQsq = muQsq;
        while (curr_iter_lsq > lsqtol) {
            newmuQsq = ((mHdsq + first_radcorrs[1] - ((mHusq + first_radcorrs[0]) * pow(tanb, 2.0))) / (pow(tanb, 2.0) - 1.0)) - (91.1876 * 91.1876 / 2.0);
            first_SUSY_BCs[6] = copysign(sqrt(abs(newmuQsq)), muQ);
            first_radcorrs = radcorr_calc(first_SUSY_BCs, SLHAQSUSY, high_prec_float(91.1876 * 91.1876));
            curr_iter_lsq = pow((muQsq) - (newmuQsq), 2.0);
            muQsq = newmuQsq;
        }
        high_prec_float currentmZ2 = ((2.0 * ((mHdsq + first_radcorrs[1] - ((mHusq + first_radcorrs[0]) * pow(tanb, 2.0))) / (pow(tanb, 2.0) - 1.0)))
                             - (2.0 * muQsq));
        // std::cout << "Sigma_u(tuned) = " << first_radcorrs[0] << std::endl << "Sigma_d(tuned) = " << first_radcorrs[1] << std::endl;
        // std::cout << "mZ(tuned) = " << copysign(sqrt(abs(currentmZ2)), currentmZ2) << std::endl;
        
        high_prec_float getmZ2_value = getmZ2(first_SUSY_BCs, SLHAQSUSY, 91.1876 * 91.1876);
        // std::cout << "getmZ(tuned) = " << copysign(sqrt(abs(getmZ2_value)), getmZ2_value) << std::endl;
        // Now we calculate the value of b=B*mu coming from this SLHA point. 
        high_prec_float BmuSLHA = sin(2.0 * beta) * (mHusq + first_radcorrs[0] + mHdsq + first_radcorrs[1] + (2.0 * muQsq)) / 2.0;
        first_SUSY_BCs[42] = BmuSLHA;
        // std::cout << "Bmu = " << BmuSLHA << std::endl;
        std::cout << "Weak scale parameters established." << endl;
        this_thread::sleep_for(chrono::seconds(1));

        /******************************************************************
         ******************** ESTABLISH GUT VALUES ************************
         ******************************************************************/
        
        // Get GUT scale now
        curr_iter_lsq = high_prec_float(100.0);
        std::vector<double> first_GUT_BCs_dbl = solveODEs(first_SUSY_BCs_dbl, log(SLHAQSUSY_dbl), log(3.0e16), 1.0e-6);
        std::vector<high_prec_float> first_GUT_BCs;
        for (const auto& value : first_GUT_BCs_dbl) {
            first_GUT_BCs.push_back(high_prec_float(value));
        }
        std::vector<high_prec_float> currbetag1g2GUT = beta_g1g2(first_GUT_BCs[0], first_GUT_BCs[1], first_GUT_BCs[2], first_GUT_BCs[7], first_GUT_BCs[8], first_GUT_BCs[9],
                                                   first_GUT_BCs[10], first_GUT_BCs[11], first_GUT_BCs[12], first_GUT_BCs[13], first_GUT_BCs[14], first_GUT_BCs[15]);
        high_prec_float curr_iter_QGUT = log(high_prec_float(3.0e16) * exp((first_GUT_BCs[1] - first_GUT_BCs[0]) / (currbetag1g2GUT[0] - currbetag1g2GUT[1])));
        high_prec_float new_QGUT = curr_iter_QGUT;
        double curr_iter_QGUT_dbl = double(curr_iter_QGUT);
        double new_QGUT_dbl = double(new_QGUT_dbl);
        while (curr_iter_lsq > lsqtol) {
            first_GUT_BCs_dbl = solveODEs(first_SUSY_BCs_dbl, log(SLHAQSUSY_dbl), curr_iter_QGUT_dbl, 1.0e-6);
            for (int GUTidx = 0; GUTidx < 44; ++GUTidx) {
                first_GUT_BCs[GUTidx] = high_prec_float(first_GUT_BCs_dbl[GUTidx]);
            }
            new_QGUT = log(exp(curr_iter_QGUT) * exp((first_GUT_BCs[1] - first_GUT_BCs[0]) / (currbetag1g2GUT[0] - currbetag1g2GUT[1])));
            curr_iter_lsq = pow((high_prec_float(1.0) - (new_QGUT / curr_iter_QGUT)), high_prec_float(2.0));
            curr_iter_QGUT = new_QGUT;
            curr_iter_QGUT_dbl = double(curr_iter_QGUT);
        }
        // std::cout << "Tuned GUT scale = " << exp(curr_iter_QGUT) << std::endl;
        // std::cout << "GUT parameters:" << std::endl;
        for (const auto& value : first_GUT_BCs) {
            std::cout << value << std::endl;
        }
        /******************************************************************
         ********************* COMPUTE DEW VALUES *************************
         ******************************************************************/

        std::cout << "\n########## Computing Delta_EW... ##########\n" << endl;
        std::vector<LabeledValue> dewlist = DEW_calc(first_SUSY_BCs, SLHAQSUSY);
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
        
        // Save DEW results?
        bool checkSaveBool = true;
        string saveinput;
        while (checkSaveBool) {
            std::cout << "\nWould you like to save these DEW results to a .txt file (will be saved to the directory \n" << fs::current_path().string() << "/DEW4SLHA_results/DEW)?\nEnter Y to save the result or N to continue: ";
            std::getline(std::cin, saveinput);

            std::string timeStr = getCurrentTimeFormatted();

            if (saveinput == "Y" || saveinput == "y" || saveinput == "Yes" || saveinput == "yes") {
                std::string path = "DEW4SLHA_results/DEW";
                if (!fs::exists("DEW4SLHA_results")) fs::create_directory("DEW4SLHA_results");
                if (!fs::exists(path)) fs::create_directory(path);

                std::cout << "\nThe default file name is 'current_system_time_DEW_contrib_list.txt', e.g., " << timeStr << "_DEW_contrib_list.txt.\nWould you like to keep this default file name or input your own?\nEnter Y to keep the default file name or N to input your own: ";
                std::getline(std::cin, saveinput);

                if (saveinput == "Y" || saveinput == "y" || saveinput == "Yes" || saveinput == "yes") {
                    saveDEWResults(dewlist, path, timeStr + "_DEW_contrib_list.txt", printPrec);
                    checkSaveBool = false;
                    std::cout << "##### Press Enter to continue... #####\n";
                    std::cin.get();
                } else if (saveinput == "N" || saveinput == "n" || saveinput == "No" || saveinput == "no") {
                    std::cout << "\nInput your desired filename with no whitespaces and without the .txt extension (e.g. 'my_SLHA_DEW_list' without the quotes): ";
                    std::string newFileName;
                    std::getline(std::cin, newFileName);
                    saveDEWResults(dewlist, path, newFileName + ".txt", printPrec);
                    checkSaveBool = false;
                    std::cout << "##### Press Enter to continue... #####\n";
                    std::cin.get();
                } else {
                    std::cout << "Invalid user input.\n";
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                }
            } else {
                std::cout << "\nOutput not saved.\n";
                checkSaveBool = false;
                std::cout << "##### Press Enter to continue... #####\n";
                std::cin.get();
            }
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
            std::cout << "\nThank you for using DEW4SLHA.\n";
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