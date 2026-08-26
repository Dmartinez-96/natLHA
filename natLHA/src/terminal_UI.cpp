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
#include "natlha_api.hpp"
#include "shared_helpers.hpp"
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

using namespace SLHAea;
using namespace boost::multiprecision;
typedef number<mpfr_float_backend<50>> high_prec_float;  // 50 decimal digits of precision
namespace fs = std::filesystem;

/// Read one line from stdin, treating end-of-input as fatal.
///
/// USE THIS INSTEAD OF getline(cin, ...) FOR EVERY PROMPT. Each prompt in this file sits in a
/// loop that re-prompts on unrecognised input. At end of input, getline returns immediately
/// without extracting anything and without modifying the target string, so the loop's
/// condition is unchanged and it spins forever printing its own error message.
///
/// Measured on the file-path prompt by feeding it a short script and letting stdin run out:
/// 158713 copies of the "cannot be found" message filled a 20 MB output cap in 1.606 s, a
/// rate near 12.5 MB/s or 0.7 GB per minute. Unbounded, that fills a disk.
///
/// End of input is reached by ANY piped or redirected stdin, so for this program it is an
/// ordinary occurrence rather than an exotic failure. There is no useful recovery, since no
/// answer to the prompt can ever arrive, so this exits rather than returning a sentinel that
/// every one of the callers would have to remember to check.
static std::string promptLine() {
    std::string line;
    if (!std::getline(std::cin, line)) {
        std::cerr << "\nnatLHA: reached end of input while waiting for a response.\n"
                  << "Interactive prompts need one line of input each. For non-interactive\n"
                  << "or batch use, run natlha-cli instead.\n";
        std::exit(3);
    }
    return line;
}

/// Prompt until an integer in [lo, hi] is entered, treating end-of-input as fatal.
///
/// Reads a whole LINE via promptLine() and parses it, rather than using `cin >> n`. Two
/// reasons. First, end of input: a failed `cin >> n` leaves the variable untouched, so the
/// `while (true)` retry loops this replaces never terminate once input runs out -- the same
/// defect promptLine() exists to stop, which it does by exiting on a failed getline.
/// Second, `cin >> n` leaves the trailing newline in the buffer, so it cannot be freely mixed
/// with line-based reads; going line-based everywhere removes that ordering hazard entirely.
///
/// Rejects trailing characters, so "2junk" is an error rather than silently parsing as 2.
///
/// NOT to be confused with the `std::cin.get()` calls further down: those sit under "Press
/// Enter to continue" prompts and are deliberate pauses, not leftover-newline cleanup. They
/// are outside any retry loop, so at end of input they return immediately and a scripted run
/// simply does not pause -- no hang, and nothing for this helper to replace.
static int promptInt(const std::string & prompt, int lo, int hi, const std::string & onError) {
    while (true) {
        std::cout << prompt;
        const std::string raw = promptLine();
        try {
            std::size_t pos = 0;
            const int value = std::stoi(raw, &pos);
            if (pos == raw.size() && value >= lo && value <= hi) {
                return value;
            }
        } catch (const std::exception &) {
            // Fall through to the shared error message below.
        }
        std::cout << onError << "\n\n";
        this_thread::sleep_for(chrono::seconds(1));
    }
}

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

void saveDHSResults(const std::vector<LabeledValueHS>& dhslist, const std::string& directory, const std::string& filename, const int& printprec) {
    std::ofstream outFile(directory + "/" + filename, std::ios::out);
    outFile << "Given the submitted SLHA file, " << directory << ", your value for the high-scale\n"
            << "naturalness measure, Delta_HS, is: " << std::fixed << std::setprecision(printprec) << dhslist[0].value << std::endl;
    outFile << "\nThe ordered contributions to Delta_HS are as follows (decr. order): \n\n";
    for (const auto& item : dhslist) {
        outFile << item.label << ": " << std::fixed << std::setprecision(printprec) << item.value << std::endl;
    }
    outFile.close();
    std::cout << "\nThese results have been saved to the directory \n" << directory << " as " << filename << ".\n";
}

void saveDBGResults(const std::vector<LabeledValueBG>& dbglist, const std::string& directory, const std::string& filename, const int& printprec) {
    std::ofstream outFile(directory + "/" + filename, std::ios::out);
    outFile << "Given the submitted SLHA file, " << directory << ", your value for the Barbieri-Giudice\n"
            << "naturalness measure, Delta_BG, is: " << std::fixed << std::setprecision(printprec) << dbglist[0].value << std::endl;
    outFile << "\nThe ordered contributions to Delta_BG are as follows (decr. order): \n\n";
    for (const auto& item : dbglist) {
        outFile << item.label << ": " << std::fixed << std::setprecision(printprec) << item.value << std::endl;
    }
    outFile.close();
    std::cout << "\nThese results have been saved to the directory \n" << directory << " as " << filename << ".\n";
}

void saveDSNResults(const std::vector<DSNLabeledValue>& dsnlist, const high_prec_float& totalNvac, const std::string& directory, const std::string& filename, const int& printprec) {
    std::ofstream outFile(directory + "/" + filename, std::ios::out);
    outFile << "Given the submitted SLHA file, " << directory << ", your value for the stringy\n"
            << "naturalness measure, Delta_SN ~ 1 / N_vac, is: " << std::fixed << std::setprecision(printprec) << high_prec_float(1.0) / totalNvac << std::endl;
    outFile << "\nThe ordered contributions to N_vac are as follows (decr. order): \n\n";
    for (const auto& item : dsnlist) {
        if (item.value < 1.0e-3) {
            outFile << item.label << ": " << std::fixed << std::setprecision(printprec) << std::scientific << item.value << std::endl;
        } else {
            outFile << item.label << ": " << std::fixed << std::setprecision(printprec) << item.value << std::endl;
        }
    }
    outFile.close();
    std::cout << "\nThese results have been saved to the directory \n" << directory << " as " << filename << ".\n";
}

void savedeltaSNResults(const std::vector<DSNLabeledValue>& dsnlist, const high_prec_float& totalNvac, const std::string& directory, const std::string& filename, const int& printprec) {
    std::ofstream outFile(directory + "/" + filename, std::ios::out);
    outFile << "Given the submitted SLHA file, " << directory << ", your value for the differential stringy\n"
            << "naturalness measure, delta_SN = log10(1 / dN_vac), is: " << std::fixed << std::setprecision(printprec) << log10(high_prec_float(1.0) / totalNvac) << std::endl;
    outFile << "\nThe ordered contributions to dN_vac are as follows (decr. order): \n\n";
    for (const auto& item : dsnlist) {
        if (item.value < 1.0e-3) {
            outFile << item.label << ": " << std::fixed << std::setprecision(printprec) << std::scientific << item.value << std::endl;
        } else {
            outFile << item.label << ": " << std::fixed << std::setprecision(printprec) << item.value << std::endl;
        }
    }
    outFile.close();
    std::cout << "\nThese results have been saved to the directory \n" << directory << " as " << filename << ".\n";
}

void clearScreen() {
    std::cout << "\x1b[2J\x1b[H" << std::flush;
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
    std::cout << "Welcome to natLHA, a computational software suite for computing the naturalness\n"
         << "measures Delta_EW (electroweak naturalness), Delta_BG (Barbieri-Giudice naturalness),\n"
         << "Delta_HS (high-scale naturalness), and Delta_SN (stringy naturalness) in the MSSM\n"
         << "from a SUSY Les Houches Accord (SLHA) file.\n\n"
         << "To use this program, you may select a\n"
         << "MSSM SLHA file from your choice of spectrum generator (e.g.,\n"
         << "SoftSUSY, Isajet, SPheno, FlexibleSUSY, etc.).\n"
         << "If multiple renormalization scales are present in the SLHA file,\n"
         << "then the first renormalization scale present in the SLHA file,\n"
         << "from top to bottom, is read in.\n\n"
         << "All naturalness measures will be evaluated at the\n"
         << "renormalization scale given by the geometric mean of the stop masses\n"
         << "as calculated at tree level from the SLHA file to minimize logarithmic contributions.\n\n"
         << "Supported models for the local solvers are MSSM EFT models for\n"
         << "Delta_EW, Delta_SN, and Delta_HS, but only the CMSSM, NUHM(1,2,3,4),\n"
         << "and pMSSM-30 plus mu for Delta_BG.\n\n"
         << "Press Enter to begin." << endl;
    string input;
    input = promptLine(); // User reads intro and presses enter

    while (userContinue) {
        clearScreen();
        bool DEWprogcheck = true;
        /******************************************************************
         ********************* CALCULATION SELECTION **********************
        ******************************************************************/
        std::cout << "##############################################################\n";
        std::cout << "natLHA calculates the electroweak naturalness measure\n";
        std::cout << "Delta_EW by default.\n\n";

        // Check if user wants to compute Delta_HS as well

        bool checkcompDHS = true;
        bool DHScalc = false;
        while (checkcompDHS) {
            std::cout << "##############################################################\n";
            std::cout << "Would you like to also calculate the high-scale naturalness measure Delta_HS?\n";
            std::cout << "Enter Y for yes or N for no: ";
            string dhsCheckInp;
            dhsCheckInp = promptLine();

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
            dbgCheckInp = promptLine();

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
            dsnCheckInp = promptLine();

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
            precCheckInp = promptLine();
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
                << "6: pMSSM-30 plus mu (31 independent directions)\n\n";
            modinp = promptInt("From the list above, input the number of the model your SLHA file corresponds to: ",
                               1, 6, "Invalid model number selected, please try again.");
            std::cout << "\n####################################################\n"
                    << "Please select the level of precision you want for the Delta_BG calculation.\n"
                    << "Below are the options: \n"
                    << "1: Fixed 8-point diagnostic stencil.\n"
                    << "2: Fixed 4-point diagnostic stencil.\n"
                    << "3: Adaptive 2-point production mode (default).\n\n";

            precinp = promptInt("From the list above, input the number corresponding to the precision you want: ",
                                1, 3, "Invalid Delta_BG precision setting selected, please try again.");
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
                    << "1: Full DSN P_mu + soft terms integrated density measure\n"
                    << "2: P_mu (integrated ABDS density measure in mu parameter alone)\n"
                    << "3: Differential ABDS density at current BM point.\n\n";
            DSNcalcSelect = promptInt("From the list above, input the number corresponding to the precision you want: ",
                                      1, 3,
                                      "Invalid Delta_SN mode selected, please try again.\n\n"
                                      "1: Full DSN P_mu + soft terms integrated density measure\n"
                                      "2: P_mu (integrated ABDS density measure in mu parameter alone)\n"
                                      "3: Differential ABDS density at current BM point.");
            std::cout << "\n####################################################\n";
            if ((DSNcalcSelect == 1) || (DSNcalcSelect == 3)) {
                // The isnan() guard the previous loop carried could never fire: nF_input is an
                // int, and isnan of an integral value is always false. promptInt covers the
                // case it was reaching for -- std::stoi throws on a non-numeric entry and the
                // catch re-prompts -- and its lo bound enforces non-negativity.
                nF_input = promptInt("Please input the number of F-type SUSY breaking fields as an integer: ",
                                     0, std::numeric_limits<int>::max(),
                                     "Invalid number of F-type fields input, please try again.");
                std::cout << "\n####################################################\n";
                nD_input = promptInt("Please input the number of D-type SUSY breaking fields as an integer: ",
                                     0, std::numeric_limits<int>::max(),
                                     "Invalid number of D-type fields input, please try again.");
            }       
        }

        std::cout << "\n########## Configuration Complete ##########\n";
        this_thread::sleep_for(chrono::milliseconds(1500));
        clearScreen();
        // No stdin flush here. This used to be cin.clear() plus
        // cin.ignore(max, '\n'), which discarded the newline that `cin >> n` leaves behind.
        // Every prompt above now reads whole lines, so nothing partial is ever pending and
        // that ignore() consumed the NEXT REAL LINE instead.
        //
        // Isolated rather than assumed: with the flush still in place, a scripted run whose
        // SLHA path came directly after the precision answer reported "The input file cannot
        // be found", while the same script with ONE extra blank line inserted ahead of the
        // path -- giving the ignore() something harmless to eat -- accepted the file and
        // completed normally. That is the flush consuming exactly one line.
            
        /******************************************************************
         ************************ SLHA READ-IN ****************************
        ******************************************************************/
       
        bool fileCheck = true;
        string direc;
        while (fileCheck) {
            std::cout << "Enter the full directory for your SLHA file: ";
            direc = promptLine();

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
        // THE PIPELINE LIVES IN natlha::evaluate() NOW, not here.
        //
        // What used to be inlined at this point -- parse the SLHA, run to
        // exactly one positive-stop sign-changing or exact Q_SUSY root at the declared
        // maximum log(Q) scan spacing, jointly converge it with the EWSB mu solve,
        // fill b = B*mu, and iterate to the g1 = g2 scale -- is one function, and
        // src/main_cli.cpp calls that same function
        // for its non-interactive modes. One implementation, so a fix to the pipeline reaches
        // both front ends instead of only the one it was made in.
        //
        // What remains here is what is genuinely interactive: collecting the choices above,
        // and reporting and saving the results below.
        natlha::Config apiCfg;
        apiCfg.slhaPath = direc;
        // The measures are left OFF here on purpose, so this call performs the shared setup
        // only. The reporting code below invokes DEW_calc, DHS_calc, DBG_calc and DSN_calc
        // itself, because it interleaves each result with its own prompts and save handling.
        // Asking evaluate() for them as well would run every calculator twice per point --
        // which for DBG_calc means paying its finite-difference stencil twice over, the most
        // expensive thing natLHA does.
        apiCfg.computeDEW = false;
        apiCfg.computeDHS = false;
        apiCfg.computeDBG = false;
        apiCfg.computeDSN = false;
        // Every compute* flag above is false because this front end calls the calculators
        // itself. It does still read apiResult.mZ2FromSolver and hand it to its own DSN_calc,
        // and evaluate() skips that solve unless something asks for it, so ask explicitly.
        // Without this the interactive delta_SN would be fed a zero.
        apiCfg.wantMZ2FromSolver = true;
        apiCfg.bgModelIndex = modinp;
        apiCfg.bgPrecision = precinp;
        apiCfg.snMode = DSNcalcSelect;
        apiCfg.snNF = nF_input;
        apiCfg.snND = nD_input;

        std::cout << "Analyzing submitted SLHA.\n";
        const natlha::Result apiResult = natlha::evaluate(apiCfg);
        if (!apiResult.ok) {
            std::cout << "This SLHA file could not be evaluated: " << apiResult.error << "\n"
                      << "Returning to the configuration screen.\n";
            this_thread::sleep_for(chrono::seconds(2));
            continue;
        }
        std::cout << "SLHA parameters read in." << endl;
        std::cout << "Weak scale parameters established." << endl;
        this_thread::sleep_for(chrono::seconds(1));

        // Names kept exactly as they were, so the reporting and saving code below reads the
        // same quantities it always did and needed no changes.
        std::vector<high_prec_float> first_SUSY_BCs = apiResult.weakBCs;
        std::vector<high_prec_float> first_GUT_BCs = apiResult.gutBCs;
        std::vector<high_prec_float> first_radcorrs = apiResult.radCorrs;
        high_prec_float SLHAQSUSY = apiResult.qSusy;
        high_prec_float curr_iter_QGUT = apiResult.logQGut;
        high_prec_float currentmZ2 = apiResult.mZ2;
        high_prec_float solverMZ2Value = apiResult.mZ2FromSolver;
        high_prec_float tanb = first_SUSY_BCs[43];

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
        
        // Save DEW results?
        bool checkSaveBool = true;
        string saveinput;
        while (checkSaveBool) {
            std::cout << "\nWould you like to save these DEW results to a .txt file (will be saved to the directory \n" << fs::current_path().string() << "/natLHA_results/DEW)?\nEnter Y to save the result or N to continue: ";
            saveinput = promptLine();

            std::string timeStr = getCurrentTimeFormatted();

            if (saveinput == "Y" || saveinput == "y" || saveinput == "Yes" || saveinput == "yes") {
                std::string path = "natLHA_results/DEW";
                if (!fs::exists("natLHA_results")) fs::create_directory("natLHA_results");
                if (!fs::exists(path)) fs::create_directory(path);

                std::cout << "\nThe default file name is 'current_system_time_DEW_contrib_list.txt', e.g., " << timeStr << "_DEW_contrib_list.txt.\nWould you like to keep this default file name or input your own?\nEnter Y to keep the default file name or N to input your own: ";
                saveinput = promptLine();

                if (saveinput == "Y" || saveinput == "y" || saveinput == "Yes" || saveinput == "yes") {
                    saveDEWResults(dewlist, path, timeStr + "_DEW_contrib_list.txt", printPrec);
                    checkSaveBool = false;
                    std::cout << "##### Press Enter to continue... #####\n";
                    std::cin.get();
                } else if (saveinput == "N" || saveinput == "n" || saveinput == "No" || saveinput == "no") {
                    std::cout << "\nInput your desired filename with no whitespaces and without the .txt extension (e.g. 'my_SLHA_DEW_list' without the quotes): ";
                    std::string newFileName;
                    newFileName = promptLine();
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
                                                      91.1876 * 91.1876, first_SUSY_BCs[43] * first_SUSY_BCs[43], first_radcorrs[0], first_radcorrs[1]);  // stale-ok: this literal is DHS_calc's running_mZ_sq parameter (DHS_calc.hpp:18), an argument to the call

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

            bool checkDHSSaveBool = true;
            string saveDHSinput;
            while (checkDHSSaveBool) {
                std::cout << "\nWould you like to save these DHS results to a .txt file (will be saved to the directory \n" << fs::current_path().string() << "/natLHA_results/DHS)?\nEnter Y to save the result or N to continue: ";
                saveDHSinput = promptLine();

                std::string DHStimeStr = getCurrentTimeFormatted();

                if (saveDHSinput == "Y" || saveDHSinput == "y" || saveDHSinput == "Yes" || saveDHSinput == "yes") {
                    std::string DHSpath = "natLHA_results/DHS";
                    if (!fs::exists("natLHA_results")) fs::create_directory("natLHA_results");
                    if (!fs::exists(DHSpath)) fs::create_directory(DHSpath);

                    std::cout << "\nThe default file name is 'current_system_time_DHS_contrib_list.txt', e.g., " << DHStimeStr << "_DHS_contrib_list.txt.\nWould you like to keep this default file name or input your own?\nEnter Y to keep the default file name or N to input your own: ";
                    saveDHSinput = promptLine();

                    if (saveDHSinput == "Y" || saveDHSinput == "y" || saveDHSinput == "Yes" || saveDHSinput == "yes") {
                        saveDHSResults(dhslist, DHSpath, DHStimeStr + "_DHS_contrib_list.txt", printPrec);
                        checkDHSSaveBool = false;
                        std::cout << "##### Press Enter to continue... #####\n";
                        std::cin.get();
                    } else if (saveDHSinput == "N" || saveDHSinput == "n" || saveDHSinput == "No" || saveDHSinput == "no") {
                        std::cout << "\nInput your desired filename with no whitespaces and without the .txt extension (e.g. 'my_SLHA_DHS_list' without the quotes): ";
                        std::string newDHSFileName;
                        newDHSFileName = promptLine();
                        saveDHSResults(dhslist, DHSpath, newDHSFileName + ".txt", printPrec);
                        checkDHSSaveBool = false;
                        std::cout << "##### Press Enter to continue... #####\n";
                        std::cin.get();
                    } else {
                        std::cout << "Invalid user input.\n";
                        std::this_thread::sleep_for(std::chrono::seconds(1));
                    }
                } else {
                    std::cout << "\nOutput not saved.\n";
                    checkDHSSaveBool = false;
                    std::cout << "##### Press Enter to continue... #####\n";
                    std::cin.get();
                }
            }
        }

        /******************************************************************
         ********************* COMPUTE DBG VALUES *************************
         ******************************************************************/

        if (DBGcalc) {
            high_prec_float logQSUSY = log(SLHAQSUSY);
            std::cout << "\n########## Computing Delta_BG... ##########\n" << endl;
            std::cout << "(This can take a while...)\n";
            const BGResult bgResult = DBG_calc(modinp, precinp, curr_iter_QGUT,
                                               logQSUSY, tanb, first_GUT_BCs, currentmZ2);
            if (!bgResult.ok) {
                std::cout << "Delta_BG failed: " << bgResult.failure << "\n"
                          << "Returning to the configuration screen.\n";
                continue;
            }
            const vector<LabeledValueBG>& myDBGlist = bgResult.contributions;
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

            bool checkDBGSaveBool = true;
            string saveDBGinput;
            while (checkDBGSaveBool) {
                std::cout << "\nWould you like to save these DBG results to a .txt file (will be saved to the directory \n" << fs::current_path().string() << "/natLHA_results/DBG)?\nEnter Y to save the result or N to continue: ";
                saveDBGinput = promptLine();

                std::string DBGtimeStr = getCurrentTimeFormatted();

                if (saveDBGinput == "Y" || saveDBGinput == "y" || saveDBGinput == "Yes" || saveDBGinput == "yes") {
                    std::string DBGpath = "natLHA_results/DBG";
                    if (!fs::exists("natLHA_results")) fs::create_directory("natLHA_results");
                    if (!fs::exists(DBGpath)) fs::create_directory(DBGpath);

                    std::cout << "\nThe default file name is 'current_system_time_DBG_contrib_list.txt', e.g., " << DBGtimeStr << "_DBG_contrib_list.txt.\nWould you like to keep this default file name or input your own?\nEnter Y to keep the default file name or N to input your own: ";
                    saveDBGinput = promptLine();

                    if (saveDBGinput == "Y" || saveDBGinput == "y" || saveDBGinput == "Yes" || saveDBGinput == "yes") {
                        saveDBGResults(myDBGlist, DBGpath, DBGtimeStr + "_DBG_contrib_list.txt", printPrec);
                        checkDBGSaveBool = false;
                        std::cout << "##### Press Enter to continue... #####\n";
                        std::cin.get();
                    } else if (saveDBGinput == "N" || saveDBGinput == "n" || saveDBGinput == "No" || saveDBGinput == "no") {
                        std::cout << "\nInput your desired filename with no whitespaces and without the .txt extension (e.g. 'my_SLHA_DBG_list' without the quotes): ";
                        std::string newDBGFileName;
                        newDBGFileName = promptLine();
                        saveDBGResults(myDBGlist, DBGpath, newDBGFileName + ".txt", printPrec);
                        checkDBGSaveBool = false;
                        std::cout << "##### Press Enter to continue... #####\n";
                        std::cin.get();
                    } else {
                        std::cout << "Invalid user input.\n";
                        std::this_thread::sleep_for(std::chrono::seconds(1));
                    }
                } else {
                    std::cout << "\nOutput not saved.\n";
                    checkDBGSaveBool = false;
                    std::cout << "##### Press Enter to continue... #####\n";
                    std::cin.get();
                }
            }
        }
     
        /******************************************************************
         ********************* COMPUTE DSN VALUES *************************
         ******************************************************************/

        if (DSNcalc) {
            high_prec_float logQSUSY = log(SLHAQSUSY);
            std::vector<DSNLabeledValue> myDSNlist = DSN_calc(DSNcalcSelect, first_SUSY_BCs, solverMZ2Value, logQSUSY, curr_iter_QGUT, nF_input, nD_input);
            high_prec_float totalN = 0.0;
            for (const auto& item : myDSNlist) {
                totalN += item.value;
            }
            if (myDSNlist.empty() || totalN <= 0.0 || isnan(totalN) || isinf(totalN)) {
                std::cerr << "Error: DSN_calc returned an invalid vacuum density.\n";
                continue;
            }
            // Mode 3 reports log10(1 / dN_vac) per dissertation Eq. 5.21; modes 1/2 report
            // the plain reciprocal 1 / N_vac. Kept mode-conditional so the console headline
            // matches what savedeltaSNResults writes for the same run.
            high_prec_float totalDSN = (DSNcalcSelect == 3)
                                           ? log10(high_prec_float(1.0) / totalN)
                                           : high_prec_float(1.0) / totalN;
            if ((DSNcalcSelect == 1) || (DSNcalcSelect == 2)) {
                std::cout << "\n########## Delta_SN Results ##########\n";
                std::cout << "Your value for the stringy naturalness measure, Delta_SN ~ 1 / N_vac, is: "
                        << totalDSN;
                this_thread::sleep_for(chrono::milliseconds(250));
                std::cout << "\nThe ordered contributions to N_vac are as follows (decr. order):\n";
                for (size_t i = 0; i < myDSNlist.size(); ++i) {
                    if (myDSNlist[i].value < 1.0e-3) {
                        std::cout << (i + 1) << ": " << std::scientific << myDSNlist[i].value << ", " << myDSNlist[i].label << endl;
                        this_thread::sleep_for(chrono::milliseconds(static_cast<int>(1000 / myDSNlist.size())));
                    } else {
                        std::cout << (i + 1) << ": " << myDSNlist[i].value << ", " << myDSNlist[i].label << endl;
                        this_thread::sleep_for(chrono::milliseconds(static_cast<int>(1000 / myDSNlist.size())));
                    }
                }

                bool checkDSNSaveBool = true;
                string saveDSNinput;
                while (checkDSNSaveBool) {
                    std::cout << "\nWould you like to save these DSN results to a .txt file (will be saved to the directory \n" << fs::current_path().string() << "/DSN4SLHA_results/DSN)?\nEnter Y to save the result or N to continue: ";
                    saveDSNinput = promptLine();

                    std::string DSNtimeStr = getCurrentTimeFormatted();

                    if (saveDSNinput == "Y" || saveDSNinput == "y" || saveDSNinput == "Yes" || saveDSNinput == "yes") {
                        std::string DSNpath = "DSN4SLHA_results/DSN";
                        if (!fs::exists("DSN4SLHA_results")) fs::create_directory("DSN4SLHA_results");
                        if (!fs::exists(DSNpath)) fs::create_directory(DSNpath);

                        std::cout << "\nThe default file name is 'current_system_time_DSN_contrib_list.txt', e.g., " << DSNtimeStr << "_DSN_contrib_list.txt.\nWould you like to keep this default file name or input your own?\nEnter Y to keep the default file name or N to input your own: ";
                        saveDSNinput = promptLine();

                        if (saveDSNinput == "Y" || saveDSNinput == "y" || saveDSNinput == "Yes" || saveDSNinput == "yes") {
                            saveDSNResults(myDSNlist, totalN, DSNpath, DSNtimeStr + "_DSN_contrib_list.txt", printPrec);
                            checkDSNSaveBool = false;
                            std::cout << "##### Press Enter to continue... #####\n";
                            std::cin.get();
                        } else if (saveDSNinput == "N" || saveDSNinput == "n" || saveDSNinput == "No" || saveDSNinput == "no") {
                            std::cout << "\nInput your desired filename with no whitespaces and without the .txt extension (e.g. 'my_SLHA_DSN_list' without the quotes): ";
                            std::string newDSNFileName;
                            newDSNFileName = promptLine();
                            saveDSNResults(myDSNlist, totalN, DSNpath, newDSNFileName + ".txt", printPrec);
                            checkDSNSaveBool = false;
                            std::cout << "##### Press Enter to continue... #####\n";
                            std::cin.get();
                        } else {
                            std::cout << "Invalid user input.\n";
                            std::this_thread::sleep_for(std::chrono::seconds(1));
                        }
                    } else {
                        std::cout << "\nOutput not saved.\n";
                        checkDSNSaveBool = false;
                        std::cout << "##### Press Enter to continue... #####\n";
                        std::cin.get();
                    }
                }
            } else if ((DSNcalcSelect == 3)) {
                std::cout << "\n########## delta_SN Results ##########\n";
                std::cout << "Your value for the differential stringy naturalness measure, delta_SN = log10(1 / dN_vac), is: "
                    << totalDSN;
                this_thread::sleep_for(chrono::milliseconds(250));
                std::cout << "\nThe ordered contributions to dN_vac are as follows (decr. order):\n";
                for (size_t i = 0; i < myDSNlist.size(); ++i) {
                    if (myDSNlist[i].value < 1.0e-3) {
                        std::cout << (i + 1) << ": " << std::scientific << myDSNlist[i].value << ", " << myDSNlist[i].label << endl;
                        this_thread::sleep_for(chrono::milliseconds(static_cast<int>(1000 / myDSNlist.size())));
                    } else {
                        std::cout << (i + 1) << ": " << myDSNlist[i].value << ", " << myDSNlist[i].label << endl;
                        this_thread::sleep_for(chrono::milliseconds(static_cast<int>(1000 / myDSNlist.size())));
                    }
                }

                bool checkDSNSaveBool = true;
                string saveDSNinput;
                while (checkDSNSaveBool) {
                    std::cout << "\nWould you like to save these DSN results to a .txt file (will be saved to the directory \n" << fs::current_path().string() << "/DSN4SLHA_results/DSN)?\nEnter Y to save the result or N to continue: ";
                    saveDSNinput = promptLine();

                    std::string DSNtimeStr = getCurrentTimeFormatted();

                    if (saveDSNinput == "Y" || saveDSNinput == "y" || saveDSNinput == "Yes" || saveDSNinput == "yes") {
                        std::string DSNpath = "DSN4SLHA_results/DSN";
                        if (!fs::exists("DSN4SLHA_results")) fs::create_directory("DSN4SLHA_results");
                        if (!fs::exists(DSNpath)) fs::create_directory(DSNpath);

                        std::cout << "\nThe default file name is 'current_system_time_deltaSN_contrib_list.txt', e.g., " << DSNtimeStr << "_deltaSN_contrib_list.txt.\nWould you like to keep this default file name or input your own?\nEnter Y to keep the default file name or N to input your own: ";
                        saveDSNinput = promptLine();

                        if (saveDSNinput == "Y" || saveDSNinput == "y" || saveDSNinput == "Yes" || saveDSNinput == "yes") {
                            savedeltaSNResults(myDSNlist, totalN, DSNpath, DSNtimeStr + "_deltaSN_contrib_list.txt", printPrec);
                            checkDSNSaveBool = false;
                            std::cout << "##### Press Enter to continue... #####\n";
                            std::cin.get();
                        } else if (saveDSNinput == "N" || saveDSNinput == "n" || saveDSNinput == "No" || saveDSNinput == "no") {
                            std::cout << "\nInput your desired filename with no whitespaces and without the .txt extension (e.g. 'my_SLHA_deltaSN_list' without the quotes): ";
                            std::string newDSNFileName;
                            newDSNFileName = promptLine();
                            savedeltaSNResults(myDSNlist, totalN, DSNpath, newDSNFileName + ".txt", printPrec);
                            checkDSNSaveBool = false;
                            std::cout << "##### Press Enter to continue... #####\n";
                            std::cin.get();
                        } else {
                            std::cout << "Invalid user input.\n";
                            std::this_thread::sleep_for(std::chrono::seconds(1));
                        }
                    } else {
                        std::cout << "\nOutput not saved.\n";
                        checkDSNSaveBool = false;
                        std::cout << "##### Press Enter to continue... #####\n";
                        std::cin.get();
                    }
                }
            }
        }

        
        // Try again?
        string checkcontinue;
        std::cout << "Would you like to try again with a new SLHA file? Enter Y to try again or N to stop: ";
        checkcontinue = promptLine();
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
