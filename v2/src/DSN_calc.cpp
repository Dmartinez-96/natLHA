#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <string>
#include <thread>
#include <chrono>
#include <boost/math/special_functions/next.hpp>
#include "DSN_calc.hpp"
#include "MSSM_RGE_solver.hpp"
#include "MSSM_RGE_solver_with_stopfinder.hpp"
#include "MSSM_RGE_solver_with_U3Q3finder.hpp"
#include "mZ_numsolver.hpp"
#include "radcorr_calc.hpp"
#include "tree_mass_calc.hpp"
#include "EWSB_loop.hpp"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

double signum(double x) {
    if (x < 0) {
        return -1.0;
    } else if (x > 0) {
        return 1.0;
    } else {
        return 0.0;
    }
}

double soft_prob_calc(double x, int nPower) {
    return ((0.5 * x / (static_cast<double>(nPower) + 1.0))
            * ((signum(x) * (pow(x, nPower) - pow((-1.0) * x, nPower))) + pow(x, nPower) + pow((-1.0) * x, nPower)));
}

bool EWSB_Check(vector<double>& weak_boundary_conditions, vector<double>& radiat_correcs) {
    bool checkifEWSB = true;

    if (abs(2.0 * weak_boundary_conditions[42]) > abs((2.0 * pow(weak_boundary_conditions[6], 2.0)) + weak_boundary_conditions[25] + radiat_correcs[0] + weak_boundary_conditions[26] + radiat_correcs[1])) {
        std::cout << "Scalar pot'l UFB at loop-level." << endl;
        checkifEWSB = false;
    }
    return checkifEWSB;
}

bool CCB_Check(vector<double>& weak_boundary_conditions) {
    bool checkifNoCCB = true;
    for (int i = 27; i < 42; ++i) {
        if (weak_boundary_conditions[i] < 0) {
            std::cout << "CCB minima" << endl;
            checkifNoCCB = false;
        }
    }
    return checkifNoCCB;
}

double first_derivative_calc(double hStep, double pm2h, double pmh, double pph, double pp2h) {
    return ((pm2h / 12.0) - (2.0 * pmh / 3.0) + (2.0 * pph / 3.0) - (pp2h / 12.0)) / hStep;
}

double second_derivative_calc(double hStep, double pStart, double pm2h, double pmh, double pph, double pp2h) {
    return (((-1.0) * pm2h / 12.0) + (4.0 * pmh / 3.0) - (5.0 * pStart / 2.0) + (4.0 * pph / 3.0) - (pp2h / 12.0)) / (hStep * hStep);
}

double mixed_second_derivative_calc(double pStep, double tStep, double fm2pm2t, double fm2pmt, double fm2ppt, double fm2pp2t,
                                    double fmpm2t, double fmpmt, double fmppt, double fmpp2t, double fppm2t, double fppmt, double fpppt,
                                    double fppp2t, double fp2pm2t, double fp2pmt, double fp2ppt, double fp2pp2t) {
    return ((1.0 / (32.0 * pStep * tStep))
            * ((4.0 * fpppt) - fp2ppt - fppp2t + (2.0 * fp2pp2t) - (4.0 * fmppt) + fm2ppt + fmpp2t - (2.0 * fm2pp2t)
               - (4.0 * fppmt) + fp2pmt + fppm2t - (2.0 * fp2pm2t) + (4.0 * fmpmt) - fm2pmt - fmpm2t + (2.0 * fm2pm2t)));
}

double calculate_approx_mZ2(vector<double> weak_solutions, double explogQSUSY, double mZ2Value) {
    vector<double> calculateRadCorrs = radcorr_calc(weak_solutions, explogQSUSY, mZ2Value);
    return 2.0 * ((((weak_solutions[26] + calculateRadCorrs[1] - ((weak_solutions[25] + calculateRadCorrs[0]) * weak_solutions[43] * weak_solutions[43]))) / ((weak_solutions[43] * weak_solutions[43]) - 1.0)) - (weak_solutions[6] * weak_solutions[6]));
}

double calculate_approx_tanb(vector<double> weak_solutions, double explogQSUSY, double mZ2Value) {
    vector<double> calculateRadCorrs = radcorr_calc(weak_solutions, explogQSUSY, mZ2Value);
    return tan(0.5 * (M_PI - asin(abs(2.0 * weak_solutions[42] / (weak_solutions[25] + weak_solutions[26] + calculateRadCorrs[0] + calculateRadCorrs[1] + (2.0 * weak_solutions[6] * weak_solutions[6]))))));
}

vector<double> single_var_deriv_approxes(vector<double>& original_GUT_conditions, double& fixed_mZ2_val, int idx_to_shift, double& logQSUSYval, double& logQGUTval) {
    double p_orig, h_p, p_plus, p_minus, p_plusplus, p_minusminus;
    if (idx_to_shift == 42) {
        p_orig = original_GUT_conditions[idx_to_shift] / original_GUT_conditions[6];
        h_p = max(pow(10.25 * (boost::math::float_next(abs(p_orig)) - abs(p_orig)), (1.0 / 5.0)), 1.0e-6);//max(pow(10.25 * (boost::math::float_next(abs(p_orig)) - abs(p_orig)), 0.2), 1.0e-6);
        p_plus = p_orig + h_p;
        p_minus = p_orig - h_p;
        p_plusplus = p_plus + h_p;
        p_minusminus = p_minus - h_p;
    }
    else {
        p_orig = original_GUT_conditions[idx_to_shift];
        h_p = max(pow(10.25 * (boost::math::float_next(abs(p_orig)) - abs(p_orig)), (1.0 / 5.0)), 1.0e-6);//max(pow(10.25 * (boost::math::float_next(abs(p_orig)) - abs(p_orig)), 0.2), 1.0e-6);
        p_plus = p_orig + h_p;
        p_minus = p_orig - h_p;
        p_plusplus = p_plus + h_p;
        p_minusminus = p_minus - h_p;
    }

    vector<double> newmZ2GUTs_plus = original_GUT_conditions;
    vector<double> newmZ2GUTs_plusplus = original_GUT_conditions;
    vector<double> newtanbGUTs_plus = original_GUT_conditions;
    vector<double> newtanbGUTs_plusplus = original_GUT_conditions;
    vector<double> newmZ2GUTs_minus = original_GUT_conditions;
    vector<double> newmZ2GUTs_minusminus = original_GUT_conditions;
    vector<double> newtanbGUTs_minus = original_GUT_conditions;
    vector<double> newtanbGUTs_minusminus = original_GUT_conditions;
    vector<double> newtanb_plus_p_plus_GUTS = original_GUT_conditions;
    vector<double> newtanb_plusplus_p_plus_GUTS = original_GUT_conditions;
    vector<double> newtanb_plus_p_plusplus_GUTS = original_GUT_conditions;
    vector<double> newtanb_plusplus_p_plusplus_GUTS = original_GUT_conditions;
    vector<double> newtanb_plus_p_minus_GUTS = original_GUT_conditions;
    vector<double> newtanb_plusplus_p_minus_GUTS = original_GUT_conditions;
    vector<double> newtanb_plus_p_minusminus_GUTS = original_GUT_conditions;
    vector<double> newtanb_plusplus_p_minusminus_GUTS = original_GUT_conditions;
    vector<double> newtanb_minus_p_plus_GUTS = original_GUT_conditions;
    vector<double> newtanb_minusminus_p_plus_GUTS = original_GUT_conditions;
    vector<double> newtanb_minus_p_plusplus_GUTS = original_GUT_conditions;
    vector<double> newtanb_minusminus_p_plusplus_GUTS = original_GUT_conditions;
    vector<double> newtanb_minus_p_minus_GUTS = original_GUT_conditions;
    vector<double> newtanb_minusminus_p_minus_GUTS = original_GUT_conditions;
    vector<double> newtanb_minus_p_minusminus_GUTS = original_GUT_conditions;
    vector<double> newtanb_minusminus_p_minusminus_GUTS = original_GUT_conditions;

    double tanb_orig = original_GUT_conditions[43];
    double h_tanb = pow(3.0 * (boost::math::float_next(abs(tanb_orig)) - abs(tanb_orig)), (1.0 / 3.0));//pow(10.25 * (boost::math::float_next(abs(tanb_orig)) - abs(tanb_orig)), 0.2);
    
    newtanbGUTs_plus[43] = tanb_orig + h_tanb;
    newtanbGUTs_plusplus[43] = tanb_orig + (2.0 * h_tanb);
    newtanb_plus_p_plus_GUTS[43] = tanb_orig + h_tanb;
    newtanb_plus_p_plusplus_GUTS[43] = tanb_orig + h_tanb;
    newtanb_plusplus_p_plus_GUTS[43] = tanb_orig + (2.0 * h_tanb);
    newtanb_plusplus_p_plusplus_GUTS[43] = tanb_orig + (2.0 * h_tanb);
    newtanb_plus_p_minus_GUTS[43] = tanb_orig + h_tanb;
    newtanb_plus_p_minusminus_GUTS[43] = tanb_orig + h_tanb;
    newtanb_plusplus_p_minus_GUTS[43] = tanb_orig + (2.0 * h_tanb);
    newtanb_plusplus_p_minusminus_GUTS[43] = tanb_orig + (2.0 * h_tanb);
    newtanbGUTs_minus[43] = tanb_orig - h_tanb;
    newtanbGUTs_minusminus[43] = tanb_orig - (2.0 * h_tanb);
    newtanb_minus_p_plus_GUTS[43] = tanb_orig - h_tanb;
    newtanb_minus_p_plusplus_GUTS[43] = tanb_orig - h_tanb;
    newtanb_minusminus_p_plus_GUTS[43] = tanb_orig - (2.0 * h_tanb);
    newtanb_minusminus_p_plusplus_GUTS[43] = tanb_orig - (2.0 * h_tanb);
    newtanb_minus_p_minus_GUTS[43] = tanb_orig - h_tanb;
    newtanb_minus_p_minusminus_GUTS[43] = tanb_orig - h_tanb;
    newtanb_minusminus_p_minus_GUTS[43] = tanb_orig - (2.0 * h_tanb);
    newtanb_minusminus_p_minusminus_GUTS[43] = tanb_orig - (2.0 * h_tanb);

    // Adjust Yukawas at Q=mt=173.2 GeV for shifted tanb points
    vector<double> weaksols_original = solveODEs(original_GUT_conditions, logQGUTval, logQSUSYval, -1.0e-6);
    double wk_tanb = weaksols_original[43];
    vector<double> mZsols_original = solveODEs(original_GUT_conditions, logQGUTval, log(173.2), -1.0e-6);
    vector<double> mZsolstanb_plus = solveODEs(newtanbGUTs_plus, logQGUTval, log(173.2), -1.0e-6);
    vector<double> mZsolstanb_minus = solveODEs(newtanbGUTs_minus, logQGUTval, log(173.2), -1.0e-6);
    vector<double> mZsolstanb_plusplus = solveODEs(newtanbGUTs_plusplus, logQGUTval, log(173.2), -1.0e-6);
    vector<double> mZsolstanb_minusminus = solveODEs(newtanbGUTs_minusminus, logQGUTval, log(173.2), -1.0e-6);
    for (int UpYukawaIndex = 7; UpYukawaIndex < 10; ++UpYukawaIndex) {
        mZsolstanb_plus[UpYukawaIndex] *= sin(atan(mZsols_original[43])) / sin(atan(mZsolstanb_plus[43]));
        mZsolstanb_plusplus[UpYukawaIndex] *= sin(atan(mZsols_original[43])) / sin(atan(mZsolstanb_plusplus[43]));
        mZsolstanb_minus[UpYukawaIndex] *= sin(atan(mZsols_original[43])) / sin(atan(mZsolstanb_minus[43]));
        mZsolstanb_minusminus[UpYukawaIndex] *= sin(atan(mZsols_original[43])) / sin(atan(mZsolstanb_minusminus[43]));
    }
    for (int DownYukawaIndex = 10; DownYukawaIndex < 16; ++DownYukawaIndex) {
        mZsolstanb_plus[DownYukawaIndex] *= cos(atan(mZsols_original[43])) / cos(atan(mZsolstanb_plus[43]));
        mZsolstanb_plusplus[DownYukawaIndex] *= cos(atan(mZsols_original[43])) / cos(atan(mZsolstanb_plusplus[43]));
        mZsolstanb_minus[DownYukawaIndex] *= cos(atan(mZsols_original[43])) / cos(atan(mZsolstanb_minus[43]));
        mZsolstanb_minusminus[DownYukawaIndex] *= cos(atan(mZsols_original[43])) / cos(atan(mZsolstanb_minusminus[43]));
    }
    vector<double> newGUTyuks_tanb_plus = solveODEs(mZsolstanb_plus, log(173.2), logQGUTval, 1.0e-6);
    vector<double> newGUTyuks_tanb_plusplus = solveODEs(mZsolstanb_plusplus, log(173.2), logQGUTval, 1.0e-6);
    vector<double> newGUTyuks_tanb_minus = solveODEs(mZsolstanb_minus, log(173.2), logQGUTval, 1.0e-6);
    vector<double> newGUTyuks_tanb_minusminus = solveODEs(mZsolstanb_minusminus, log(173.2), logQGUTval, 1.0e-6);
    for (int YukawaIndex = 7; YukawaIndex < 16; ++YukawaIndex) {
        newtanbGUTs_plus[YukawaIndex] = newGUTyuks_tanb_plus[YukawaIndex];
        newtanbGUTs_plusplus[YukawaIndex] = newGUTyuks_tanb_plusplus[YukawaIndex];
        newtanb_plus_p_plus_GUTS[YukawaIndex] = newGUTyuks_tanb_plus[YukawaIndex];
        newtanb_plus_p_plusplus_GUTS[YukawaIndex] = newGUTyuks_tanb_plus[YukawaIndex];
        newtanb_plusplus_p_plus_GUTS[YukawaIndex] = newGUTyuks_tanb_plusplus[YukawaIndex];
        newtanb_plusplus_p_plusplus_GUTS[YukawaIndex] = newGUTyuks_tanb_plusplus[YukawaIndex];
        newtanb_plus_p_minus_GUTS[YukawaIndex] = newGUTyuks_tanb_plus[YukawaIndex];
        newtanb_plus_p_minusminus_GUTS[YukawaIndex] = newGUTyuks_tanb_plus[YukawaIndex];
        newtanb_plusplus_p_minus_GUTS[YukawaIndex] = newGUTyuks_tanb_plusplus[YukawaIndex];
        newtanb_plusplus_p_minusminus_GUTS[YukawaIndex] = newGUTyuks_tanb_plusplus[YukawaIndex];
        newtanbGUTs_minus[YukawaIndex] = newGUTyuks_tanb_minus[YukawaIndex];
        newtanbGUTs_minusminus[YukawaIndex] = newGUTyuks_tanb_minusminus[YukawaIndex];
        newtanb_minus_p_plus_GUTS[YukawaIndex] = newGUTyuks_tanb_minus[YukawaIndex];
        newtanb_minus_p_plusplus_GUTS[YukawaIndex] = newGUTyuks_tanb_minus[YukawaIndex];
        newtanb_minusminus_p_plus_GUTS[YukawaIndex] = newGUTyuks_tanb_minusminus[YukawaIndex];
        newtanb_minusminus_p_plusplus_GUTS[YukawaIndex] = newGUTyuks_tanb_minusminus[YukawaIndex];
        newtanb_minus_p_minus_GUTS[YukawaIndex] = newGUTyuks_tanb_minus[YukawaIndex];
        newtanb_minus_p_minusminus_GUTS[YukawaIndex] = newGUTyuks_tanb_minus[YukawaIndex];
        newtanb_minusminus_p_minus_GUTS[YukawaIndex] = newGUTyuks_tanb_minusminus[YukawaIndex];
        newtanb_minusminus_p_minusminus_GUTS[YukawaIndex] = newGUTyuks_tanb_minusminus[YukawaIndex];
    }

    vector<double> weaksolstanb_plus = solveODEs(newtanbGUTs_plus, logQGUTval, logQSUSYval, copysign(1.0e-6, logQSUSYval - logQGUTval));
    vector<double> weaksolstanb_minus = solveODEs(newtanbGUTs_minus, logQGUTval, logQSUSYval, copysign(1.0e-6, logQSUSYval - logQGUTval));
    vector<double> weaksolstanb_plusplus = solveODEs(newtanbGUTs_plusplus, logQGUTval, logQSUSYval, copysign(1.0e-6, logQSUSYval - logQGUTval));
    vector<double> weaksolstanb_minusminus = solveODEs(newtanbGUTs_minusminus, logQGUTval, logQSUSYval, copysign(1.0e-6, logQSUSYval - logQGUTval));
    
    double mZ2_tanb_plus = calculate_approx_mZ2(weaksolstanb_plus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_tanb_minus = calculate_approx_mZ2(weaksolstanb_minus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_tanb_plusplus = calculate_approx_mZ2(weaksolstanb_plusplus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_tanb_minusminus = calculate_approx_mZ2(weaksolstanb_minusminus, exp(logQSUSYval), fixed_mZ2_val);

    if (idx_to_shift == 6) {
        newmZ2GUTs_plus[42] = original_GUT_conditions[42] * p_plus / original_GUT_conditions[6];
        newmZ2GUTs_plusplus[42] = original_GUT_conditions[42] * p_plusplus / original_GUT_conditions[6];
        newtanb_plus_p_plus_GUTS[42] = original_GUT_conditions[42] * p_plus / original_GUT_conditions[6];
        newtanb_plusplus_p_plus_GUTS[42] = original_GUT_conditions[42] * p_plus / original_GUT_conditions[6];
        newtanb_plus_p_plusplus_GUTS[42] = original_GUT_conditions[42] * p_plusplus / original_GUT_conditions[6];
        newtanb_plusplus_p_plusplus_GUTS[42] = original_GUT_conditions[42] * p_plusplus / original_GUT_conditions[6];
        newtanb_minus_p_plus_GUTS[42] = original_GUT_conditions[42] * p_plus / original_GUT_conditions[6];
        newtanb_minusminus_p_plus_GUTS[42] = original_GUT_conditions[42] * p_plus / original_GUT_conditions[6];
        newtanb_minus_p_plusplus_GUTS[42] = original_GUT_conditions[42] * p_plusplus / original_GUT_conditions[6];
        newtanb_minusminus_p_plusplus_GUTS[42] = original_GUT_conditions[42] * p_plusplus / original_GUT_conditions[6];
        newmZ2GUTs_minus[42] = original_GUT_conditions[42] * p_minus / original_GUT_conditions[6];
        newmZ2GUTs_minusminus[42] = original_GUT_conditions[42] * p_minusminus / original_GUT_conditions[6];
        newtanb_plus_p_minus_GUTS[42] = original_GUT_conditions[42] * p_minus / original_GUT_conditions[6];
        newtanb_plusplus_p_minus_GUTS[42] = original_GUT_conditions[42] * p_minus / original_GUT_conditions[6];
        newtanb_plusplus_p_minusminus_GUTS[42] = original_GUT_conditions[42] * p_minusminus / original_GUT_conditions[6];
        newtanb_plus_p_minusminus_GUTS[42] = original_GUT_conditions[42] * p_minusminus / original_GUT_conditions[6];
        newtanb_minus_p_minus_GUTS[42] = original_GUT_conditions[42] * p_minus / original_GUT_conditions[6];
        newtanb_minusminus_p_minus_GUTS[42] = original_GUT_conditions[42] * p_minus / original_GUT_conditions[6];
        newtanb_minusminus_p_minusminus_GUTS[42] = original_GUT_conditions[42] * p_minusminus / original_GUT_conditions[6];
        newtanb_minus_p_minusminus_GUTS[42] = original_GUT_conditions[42] * p_minusminus / original_GUT_conditions[6];
        newmZ2GUTs_plus[6] = p_plus;
        newmZ2GUTs_plusplus[6] = p_plusplus;
        newtanb_plus_p_plus_GUTS[6] = p_plus;
        newtanb_plusplus_p_plus_GUTS[6] = p_plus;
        newtanb_plusplus_p_plusplus_GUTS[6] = p_plusplus;
        newtanb_plus_p_plusplus_GUTS[6] = p_plusplus;
        newtanb_minus_p_plus_GUTS[6] = p_plus;
        newtanb_minusminus_p_plus_GUTS[6] = p_plus;
        newtanb_minusminus_p_plusplus_GUTS[6] = p_plusplus;
        newtanb_minus_p_plusplus_GUTS[6] = p_plusplus;
        newmZ2GUTs_minus[6] = p_minus;
        newmZ2GUTs_minusminus[6] = p_minusminus;
        newtanb_plus_p_minus_GUTS[6] = p_minus;
        newtanb_plusplus_p_minus_GUTS[6] = p_minus;
        newtanb_plusplus_p_minusminus_GUTS[6] = p_minusminus;
        newtanb_plus_p_minusminus_GUTS[6] = p_minusminus;
        newtanb_minus_p_minus_GUTS[6] = p_minus;
        newtanb_minusminus_p_minus_GUTS[6] = p_minus;
        newtanb_minusminus_p_minusminus_GUTS[6] = p_minusminus;
        newtanb_minus_p_minusminus_GUTS[6] = p_minusminus;
    } else if (idx_to_shift == 42) {
        newmZ2GUTs_plus[42] = original_GUT_conditions[6] * p_plus;
        newmZ2GUTs_plusplus[42] = original_GUT_conditions[6] * p_plusplus;
        newtanb_plus_p_plus_GUTS[42] = original_GUT_conditions[6] * p_plus;
        newtanb_plusplus_p_plus_GUTS[42] = original_GUT_conditions[6] * p_plus;
        newtanb_plus_p_plusplus_GUTS[42] = original_GUT_conditions[6] * p_plusplus;
        newtanb_plusplus_p_plusplus_GUTS[42] = original_GUT_conditions[6] * p_plusplus;
        newtanb_minus_p_plus_GUTS[42] = original_GUT_conditions[6] * p_plus;
        newtanb_minusminus_p_plus_GUTS[42] = original_GUT_conditions[6] * p_plus;
        newtanb_minus_p_plusplus_GUTS[42] = original_GUT_conditions[6] * p_plusplus;
        newtanb_minusminus_p_plusplus_GUTS[42] = original_GUT_conditions[6] * p_plusplus;
        newmZ2GUTs_minus[42] = original_GUT_conditions[6] * p_minus;
        newmZ2GUTs_minusminus[42] = original_GUT_conditions[6] * p_minusminus;
        newtanb_plus_p_minus_GUTS[42] = original_GUT_conditions[6] * p_minus;
        newtanb_plusplus_p_minus_GUTS[42] = original_GUT_conditions[6] * p_minus;
        newtanb_plus_p_minusminus_GUTS[42] = original_GUT_conditions[6] * p_minusminus;
        newtanb_plusplus_p_minusminus_GUTS[42] = original_GUT_conditions[6] * p_minusminus;
        newtanb_minus_p_minus_GUTS[42] = original_GUT_conditions[6] * p_minus;
        newtanb_minusminus_p_minus_GUTS[42] = original_GUT_conditions[6] * p_minus;
        newtanb_minus_p_minusminus_GUTS[42] = original_GUT_conditions[6] * p_minusminus;
        newtanb_minusminus_p_minusminus_GUTS[42] = original_GUT_conditions[6] * p_minusminus;
    } else {
        newmZ2GUTs_plus[idx_to_shift] = p_plus;
        newmZ2GUTs_plusplus[idx_to_shift] = p_plusplus;
        newtanb_plus_p_plus_GUTS[idx_to_shift] = p_plus;
        newtanb_plusplus_p_plus_GUTS[idx_to_shift] = p_plus;
        newtanb_plus_p_plusplus_GUTS[idx_to_shift] = p_plusplus;
        newtanb_plusplus_p_plusplus_GUTS[idx_to_shift] = p_plusplus;
        newtanb_minus_p_plus_GUTS[idx_to_shift] = p_plus;
        newtanb_minusminus_p_plus_GUTS[idx_to_shift] = p_plus;
        newtanb_minusminus_p_plusplus_GUTS[idx_to_shift] = p_plusplus;
        newtanb_minus_p_plusplus_GUTS[idx_to_shift] = p_plusplus;
        newmZ2GUTs_minus[idx_to_shift] = p_minus;
        newmZ2GUTs_minusminus[idx_to_shift] = p_minusminus;
        newtanb_plus_p_minus_GUTS[idx_to_shift] = p_minus;
        newtanb_plusplus_p_minus_GUTS[idx_to_shift] = p_minus;
        newtanb_plus_p_minusminus_GUTS[idx_to_shift] = p_minusminus;
        newtanb_plusplus_p_minusminus_GUTS[idx_to_shift] = p_minusminus;
        newtanb_minus_p_minus_GUTS[idx_to_shift] = p_minus;
        newtanb_minusminus_p_minus_GUTS[idx_to_shift] = p_minus;
        newtanb_minus_p_minusminus_GUTS[idx_to_shift] = p_minusminus;
        newtanb_minusminus_p_minusminus_GUTS[idx_to_shift] = p_minusminus;
    }

    vector<double> weaksolsp_plus = solveODEs(newmZ2GUTs_plus, logQGUTval, logQSUSYval, copysign(1.0e-6, logQSUSYval - logQGUTval));
    vector<double> weaksolsp_plusplus = solveODEs(newmZ2GUTs_plusplus, logQGUTval, logQSUSYval, copysign(1.0e-6, logQSUSYval - logQGUTval));
    vector<double> weaksolstanb_plus_p_plus = solveODEs(newtanb_plus_p_plus_GUTS, logQGUTval, logQSUSYval, copysign(1.0e-6, logQSUSYval - logQGUTval));
    vector<double> weaksolstanb_plusplus_p_plus = solveODEs(newtanb_plusplus_p_plus_GUTS, logQGUTval, logQSUSYval, copysign(1.0e-6, logQSUSYval - logQGUTval));
    vector<double> weaksolstanb_plus_p_plusplus = solveODEs(newtanb_plus_p_plusplus_GUTS, logQGUTval, logQSUSYval, copysign(1.0e-6, logQSUSYval - logQGUTval));
    vector<double> weaksolstanb_plusplus_p_plusplus = solveODEs(newtanb_plusplus_p_plusplus_GUTS, logQGUTval, logQSUSYval, copysign(1.0e-6, logQSUSYval - logQGUTval));
    vector<double> weaksolstanb_minus_p_plus = solveODEs(newtanb_minus_p_plus_GUTS, logQGUTval, logQSUSYval, copysign(1.0e-6, logQSUSYval - logQGUTval));
    vector<double> weaksolstanb_minusminus_p_plus = solveODEs(newtanb_minusminus_p_plus_GUTS, logQGUTval, logQSUSYval, copysign(1.0e-6, logQSUSYval - logQGUTval));
    vector<double> weaksolstanb_minus_p_plusplus = solveODEs(newtanb_minus_p_plusplus_GUTS, logQGUTval, logQSUSYval, copysign(1.0e-6, logQSUSYval - logQGUTval));
    vector<double> weaksolstanb_minusminus_p_plusplus = solveODEs(newtanb_minusminus_p_plusplus_GUTS, logQGUTval, logQSUSYval, copysign(1.0e-6, logQSUSYval - logQGUTval));
    vector<double> weaksolsp_minus = solveODEs(newmZ2GUTs_minus, logQGUTval, logQSUSYval, copysign(1.0e-6, logQSUSYval - logQGUTval));
    vector<double> weaksolsp_minusminus = solveODEs(newmZ2GUTs_minusminus, logQGUTval, logQSUSYval, copysign(1.0e-6, logQSUSYval - logQGUTval));
    vector<double> weaksolstanb_plus_p_minus = solveODEs(newtanb_plus_p_minus_GUTS, logQGUTval, logQSUSYval, copysign(1.0e-6, logQSUSYval - logQGUTval));
    vector<double> weaksolstanb_plusplus_p_minus = solveODEs(newtanb_plusplus_p_minus_GUTS, logQGUTval, logQSUSYval, copysign(1.0e-6, logQSUSYval - logQGUTval));
    vector<double> weaksolstanb_plus_p_minusminus = solveODEs(newtanb_plus_p_minusminus_GUTS, logQGUTval, logQSUSYval, copysign(1.0e-6, logQSUSYval - logQGUTval));
    vector<double> weaksolstanb_plusplus_p_minusminus = solveODEs(newtanb_plusplus_p_minusminus_GUTS, logQGUTval, logQSUSYval, copysign(1.0e-6, logQSUSYval - logQGUTval));
    vector<double> weaksolstanb_minus_p_minus = solveODEs(newtanb_minus_p_minus_GUTS, logQGUTval, logQSUSYval, copysign(1.0e-6, logQSUSYval - logQGUTval));
    vector<double> weaksolstanb_minusminus_p_minus = solveODEs(newtanb_minusminus_p_minus_GUTS, logQGUTval, logQSUSYval, copysign(1.0e-6, logQSUSYval - logQGUTval));
    vector<double> weaksolstanb_minus_p_minusminus = solveODEs(newtanb_minus_p_minusminus_GUTS, logQGUTval, logQSUSYval, copysign(1.0e-6, logQSUSYval - logQGUTval));
    vector<double> weaksolstanb_minusminus_p_minusminus = solveODEs(newtanb_minusminus_p_minusminus_GUTS, logQGUTval, logQSUSYval, copysign(1.0e-6, logQSUSYval - logQGUTval));
        
    double mZ2_p_plus = calculate_approx_mZ2(weaksolsp_plus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_p_plusplus = calculate_approx_mZ2(weaksolsp_plusplus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_p_plus_tanb_plus = calculate_approx_mZ2(weaksolstanb_plus_p_plus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_p_plusplus_tanb_plus = calculate_approx_mZ2(weaksolstanb_plus_p_plusplus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_p_plus_tanb_plusplus = calculate_approx_mZ2(weaksolstanb_plusplus_p_plus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_p_plusplus_tanb_plusplus = calculate_approx_mZ2(weaksolstanb_plusplus_p_plusplus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_p_plus_tanb_minus = calculate_approx_mZ2(weaksolstanb_minus_p_plus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_p_plusplus_tanb_minus = calculate_approx_mZ2(weaksolstanb_minus_p_plusplus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_p_plus_tanb_minusminus = calculate_approx_mZ2(weaksolstanb_minusminus_p_plus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_p_plusplus_tanb_minusminus = calculate_approx_mZ2(weaksolstanb_minusminus_p_plusplus, exp(logQSUSYval), fixed_mZ2_val);
    
    double mZ2_p_minus = calculate_approx_mZ2(weaksolsp_minus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_p_minusminus = calculate_approx_mZ2(weaksolsp_minusminus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_p_minus_tanb_plus = calculate_approx_mZ2(weaksolstanb_plus_p_minus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_p_minus_tanb_plusplus = calculate_approx_mZ2(weaksolstanb_plusplus_p_minus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_p_minusminus_tanb_plus = calculate_approx_mZ2(weaksolstanb_plus_p_minusminus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_p_minusminus_tanb_plusplus = calculate_approx_mZ2(weaksolstanb_plusplus_p_minusminus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_p_minus_tanb_minus = calculate_approx_mZ2(weaksolstanb_minus_p_minus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_p_minusminus_tanb_minus = calculate_approx_mZ2(weaksolstanb_minus_p_minusminus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_p_minus_tanb_minusminus = calculate_approx_mZ2(weaksolstanb_minusminus_p_minus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_p_minusminus_tanb_minusminus = calculate_approx_mZ2(weaksolstanb_minusminus_p_minusminus, exp(logQSUSYval), fixed_mZ2_val);
    
    double tanb_p_plus = calculate_approx_tanb(weaksolsp_plus, exp(logQSUSYval), fixed_mZ2_val);
    double tanb_p_plusplus = calculate_approx_tanb(weaksolsp_plusplus, exp(logQSUSYval), fixed_mZ2_val);
    
    double tanb_p_minus = calculate_approx_tanb(weaksolsp_minus, exp(logQSUSYval), fixed_mZ2_val);
    double tanb_p_minusminus = calculate_approx_tanb(weaksolsp_minusminus, exp(logQSUSYval), fixed_mZ2_val);
    
    /* Order of derivatives:
        (0: dt/dp,
         1: d^2t/dp^2,
         2: dm/dt, 
         3: dm/dp,
         4: d^2m/dt^2,
         5: d^2m/dtdp,
         6: d^2m/dp^2)
    */
    vector<double> evaluated_derivs = {first_derivative_calc(h_p, tanb_p_minusminus, tanb_p_minus, tanb_p_plus, tanb_p_plusplus),
                                       second_derivative_calc(h_p, wk_tanb, tanb_p_minusminus, tanb_p_minus, tanb_p_plus, tanb_p_plusplus),
                                       first_derivative_calc(h_tanb, mZ2_tanb_minusminus, mZ2_tanb_minus, mZ2_tanb_plus, mZ2_tanb_plusplus),
                                       first_derivative_calc(h_p, mZ2_p_minusminus, mZ2_p_minus, mZ2_p_plus, mZ2_p_plusplus),
                                       second_derivative_calc(h_tanb, fixed_mZ2_val, mZ2_tanb_minusminus, mZ2_tanb_minus, mZ2_tanb_plus, mZ2_tanb_plusplus),
                                       mixed_second_derivative_calc(h_p, h_tanb, mZ2_p_minusminus_tanb_minusminus, mZ2_p_minusminus_tanb_minus, mZ2_p_minusminus_tanb_plus, mZ2_p_minusminus_tanb_plusplus,
                                                                    mZ2_p_minus_tanb_minusminus, mZ2_p_minus_tanb_minus, mZ2_p_minus_tanb_plus, mZ2_p_minus_tanb_plusplus, mZ2_p_plus_tanb_minusminus, mZ2_p_plus_tanb_minus,
                                                                    mZ2_p_plus_tanb_plus, mZ2_p_plus_tanb_plusplus, mZ2_p_plusplus_tanb_minusminus, mZ2_p_plusplus_tanb_minus, mZ2_p_plusplus_tanb_plus, mZ2_p_plusplus_tanb_plusplus),
                                       second_derivative_calc(h_p, fixed_mZ2_val, mZ2_p_minusminus, mZ2_p_minus, mZ2_p_plus, mZ2_p_plusplus)};
    return evaluated_derivs;
}

vector<double> get_F_G_vals(vector<double>& GUT_BCs, double& curr_mZ2, double& curr_logQSUSY, double& curr_logQGUT, int derivIndex) {
    //vector<double> derivatives = single_var_deriv_approxes(GUT_BCs, curr_mZ2, derivIndex, curr_logQSUSY, curr_logQGUT);
            
    vector<double> weaksolution = solveODEs(GUT_BCs, curr_logQGUT, curr_logQSUSY, copysign(1.0e-6, (curr_logQSUSY - curr_logQGUT)));
    vector<double> RadiatCorrs = radcorr_calc(weaksolution, exp(curr_logQSUSY), curr_mZ2);
    
    double myF = (1.0 / 2.0) - (((weaksolution[26] + RadiatCorrs[1] - ((weaksolution[25] + RadiatCorrs[0]) * pow(weaksolution[43], 2.0))) / (curr_mZ2 * (pow(weaksolution[43], 2.0) - 1.0))) - (weaksolution[6] * weaksolution[6] / curr_mZ2));
    double myG = weaksolution[43] - tan(0.5 * (M_PI - asin(2.0 * weaksolution[42] / (weaksolution[25] + RadiatCorrs[0] + weaksolution[26] + RadiatCorrs[1] + (2.0 * pow(weaksolution[6], 2.0))))));
    return {myF, myG};
}

vector<double> DSN_B_windows(vector<double>& GUT_boundary_conditions, double& current_mZ2, double& current_logQSUSY, double& current_logQGUT) {
    double t_target = log(500.0);
    vector<double> BnewGUTs_plus = GUT_boundary_conditions;
    vector<double> BnewGUTs_minus = GUT_boundary_conditions;
    double BcurrentlogQGUT = current_logQGUT;
    double BcurrentlogQSUSY = current_logQSUSY;
    double BnewlogQGUT = current_logQGUT;
    double BnewlogQSUSY = current_logQSUSY;
    double Bnew_mZ2plus = current_mZ2;
    double Bnew_mZ2minus = current_mZ2;

    double Bplus = BnewGUTs_plus[42] / BnewGUTs_plus[6];
    double newBplus = Bplus;

    double Bminus = BnewGUTs_minus[42] / BnewGUTs_minus[6];
    double newBminus = Bminus;

    bool BminusNoCCB = true;
    bool BminusEWSB = true;
    bool BplusNoCCB = true;
    bool BplusEWSB = true;
    double Bcurr_tanbGUT = GUT_boundary_conditions[43];
    double Bnew_tanbGUT = Bcurr_tanbGUT;
    double Bcurr_mZ2plus = current_mZ2;
    double Bcurr_mZ2minus = current_mZ2;
    double muGUT_original = GUT_boundary_conditions[6];
    // First, compute width of ABDS window
    double lambdaB = 0.5;
    double B_least_Sq_Tol = 1.0e-2;
    double prev_fB = std::numeric_limits<double>::max();
    double curr_lsq_eval = std::numeric_limits<double>::max();
    
    while ((BminusEWSB) && (BminusNoCCB) && ((Bnew_mZ2minus > (45.5938 * 45.5938)) && (Bnew_mZ2minus < (364.7504 * 364.7504)))) {
        lambdaB = 0.5;
        B_least_Sq_Tol = 1.0e-2;
        curr_lsq_eval = std::numeric_limits<double>::max();
        Bnew_mZ2minus *= 0.99;
        std::cout << "New mZ = " << sqrt(Bnew_mZ2minus) << endl;
        std::cout << "New B = " << BnewGUTs_minus[42] / BnewGUTs_minus[6] << endl;
        std::cout << "New tanb = " << BnewGUTs_minus[43] << endl;
        int numStepsDone = 0;
        vector<double> checkweaksols = solveODEs(BnewGUTs_minus, BcurrentlogQGUT, BcurrentlogQSUSY, copysign(1.0e-6, (BcurrentlogQSUSY - BcurrentlogQGUT)));
        vector<double> checkRadCorrs = radcorr_calc(checkweaksols, exp(BnewlogQSUSY), Bnew_mZ2minus);
        BminusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
        if (BminusEWSB == true) {
            BminusEWSB = Hessian_check(checkweaksols, exp(BnewlogQSUSY));
        }
        // if (BminusEWSB == true) {
        //     BminusEWSB = BFB_check(checkweaksols);
        // }
        BminusNoCCB = CCB_Check(checkweaksols);
        if (!(BminusEWSB) || !(BminusNoCCB)) {
            break;
        }
        vector<double> tree_level_masses = TreeMassCalculator(checkweaksols, exp(BcurrentlogQSUSY), Bnew_mZ2minus);
        for (double value : tree_level_masses) {
            if (value < 0) {
                BminusNoCCB = false;
            }
        }
        if (!(BminusNoCCB)) {
            break;
        }                
        vector<double> BoldGUTs_minus = BnewGUTs_minus;
        while ((numStepsDone < 100) && (curr_lsq_eval > B_least_Sq_Tol)) {
            vector<double> current_derivatives = single_var_deriv_approxes(BnewGUTs_minus, Bnew_mZ2minus, 42, BnewlogQSUSY, BnewlogQGUT);
            
            vector<double> weaksol_minus = solveODEs(BnewGUTs_minus, BcurrentlogQGUT, BcurrentlogQSUSY, copysign(1.0e-6, (BcurrentlogQSUSY - BcurrentlogQGUT)));
            vector<double> RadCorrsMinus = radcorr_calc(weaksol_minus, exp(BnewlogQSUSY), Bnew_mZ2minus);
            
            double FB = (1.0 / 2.0) - (((weaksol_minus[26] + RadCorrsMinus[1] - ((weaksol_minus[25] + RadCorrsMinus[0]) * pow(weaksol_minus[43], 2.0))) / (Bnew_mZ2minus * (pow(weaksol_minus[43], 2.0) - 1.0))) - (weaksol_minus[6] * weaksol_minus[6] / Bnew_mZ2minus));
            double GB = weaksol_minus[43] - tan(0.5 * (M_PI - asin(2.0 * weaksol_minus[42] / (weaksol_minus[25] + RadCorrsMinus[0] + weaksol_minus[26] + RadCorrsMinus[1] + (2.0 * pow(weaksol_minus[6], 2.0))))));

            double DeltaTanbNum = ((current_derivatives[0] * GB) - (current_derivatives[2] * FB));
            double DeltaDenom = ((current_derivatives[0] * current_derivatives[3]) - (current_derivatives[1] * current_derivatives[2]));
            double DeltaTanb = DeltaTanbNum / DeltaDenom;

            double DeltaBNum = ((current_derivatives[3] * FB) - (current_derivatives[1] * GB));
            double DeltaB = DeltaBNum / DeltaDenom;

            curr_lsq_eval = pow(DeltaB, 2.0) + pow(DeltaTanb, 2.0);
            BnewGUTs_minus[43] = BnewGUTs_minus[43] - DeltaTanb;
            BnewGUTs_minus[42] = ((BnewGUTs_minus[42] / muGUT_original) - DeltaB) * muGUT_original;
            if ((isnan(BnewGUTs_minus[42]) || (isnan(BnewGUTs_minus[43])))) {
                BminusEWSB = false;
                break;
            }
            numStepsDone++;
        }
        if (!BminusEWSB) {
            // std::cout << "Failed to converge" << endl;
            BnewGUTs_minus[6] = BoldGUTs_minus[6];
            BnewGUTs_minus[42] = BoldGUTs_minus[42];
            BnewGUTs_minus[43] = BoldGUTs_minus[43];
            break;
        }
        if ((BnewGUTs_minus[43] < 3.0) || (BnewGUTs_minus[43] > 60.0)) {
            BminusEWSB = false;
        }
    }
    std::cout << "B(ABDS, minus) = " << BnewGUTs_minus[42] / BnewGUTs_minus[6] << endl;
    double B_GUT_minus = BnewGUTs_minus[42] / BnewGUTs_minus[6];
    if (abs(B_GUT_minus - (GUT_boundary_conditions[42] / GUT_boundary_conditions[6])) < 1.0e-9) {
        B_GUT_minus = 0.9999 * B_GUT_minus;
    }
    Bcurr_tanbGUT = GUT_boundary_conditions[43];
    Bnew_tanbGUT = Bcurr_tanbGUT;
    BcurrentlogQGUT = current_logQGUT;
    BcurrentlogQSUSY = current_logQSUSY;

    while ((BplusEWSB) && (BplusNoCCB) && ((Bnew_mZ2plus > (45.5938 * 45.5938)) && (Bnew_mZ2plus < (364.7504 * 364.7504)))) {
        lambdaB = 0.5;
        B_least_Sq_Tol = 1.0e-2;
        curr_lsq_eval = std::numeric_limits<double>::max();
        Bnew_mZ2plus *= 1.01;
        std::cout << "New mZ = " << sqrt(Bnew_mZ2plus) << endl;
        std::cout << "New B = " << BnewGUTs_plus[42] / BnewGUTs_plus[6] << endl;
        std::cout << "New tanb = " << BnewGUTs_plus[43] << endl;
        int numStepsDone = 0;
        vector<double> checkweaksols = solveODEs(BnewGUTs_plus, BcurrentlogQGUT, BcurrentlogQSUSY, copysign(1.0e-6, (BcurrentlogQSUSY - BcurrentlogQGUT)));
        vector<double> checkRadCorrs = radcorr_calc(checkweaksols, exp(BnewlogQSUSY), Bnew_mZ2plus);
        BplusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
        if (BplusEWSB == true) {
            BplusEWSB = Hessian_check(checkweaksols, exp(BnewlogQSUSY));
        }
        // if (BplusEWSB == true) {
        //     BplusEWSB = BFB_check(checkweaksols);
        // }
        BplusNoCCB = CCB_Check(checkweaksols);
        if (!(BplusEWSB) || !(BplusNoCCB)) {
            break;
        }
        vector<double> tree_level_masses = TreeMassCalculator(checkweaksols, exp(BcurrentlogQSUSY), Bnew_mZ2plus);
        for (double value : tree_level_masses) {
            if (value < 0) {
                BplusNoCCB = false;
            }
        }
        if (!(BplusNoCCB)) {
            break;
        }             
        vector<double> BoldGUTs_plus = BnewGUTs_plus;   
        while ((numStepsDone < 100) && (curr_lsq_eval > B_least_Sq_Tol)) {
            vector<double> current_derivatives = single_var_deriv_approxes(BnewGUTs_plus, Bnew_mZ2plus, 42, BnewlogQSUSY, BnewlogQGUT);
            
            vector<double> weaksol_plus = solveODEs(BnewGUTs_plus, BcurrentlogQGUT, BcurrentlogQSUSY, copysign(1.0e-6, (BcurrentlogQSUSY - BcurrentlogQGUT)));
            vector<double> RadCorrsPlus = radcorr_calc(weaksol_plus, exp(BnewlogQSUSY), Bnew_mZ2plus);
            
            double FB = (1.0 / 2.0) - (((weaksol_plus[26] + RadCorrsPlus[1] - ((weaksol_plus[25] + RadCorrsPlus[0]) * pow(weaksol_plus[43], 2.0))) / (Bnew_mZ2plus * (pow(weaksol_plus[43], 2.0) - 1.0))) - (weaksol_plus[6] * weaksol_plus[6] / Bnew_mZ2plus));
            double GB = weaksol_plus[43] - tan(0.5 * (M_PI - asin(2.0 * weaksol_plus[42] / (weaksol_plus[25] + RadCorrsPlus[0] + weaksol_plus[26] + RadCorrsPlus[1] + (2.0 * pow(weaksol_plus[6], 2.0))))));

            double DeltaTanbNum = ((current_derivatives[0] * GB) - (current_derivatives[2] * FB));
            double DeltaDenom = ((current_derivatives[0] * current_derivatives[3]) - (current_derivatives[1] * current_derivatives[2]));
            double DeltaTanb = DeltaTanbNum / DeltaDenom;

            double DeltaBNum = ((current_derivatives[3] * FB) - (current_derivatives[1] * GB));
            double DeltaB = DeltaBNum / DeltaDenom;

            curr_lsq_eval = pow(DeltaB, 2.0) + pow(DeltaTanb, 2.0);
            BnewGUTs_plus[43] = BnewGUTs_plus[43] - DeltaTanb;
            BnewGUTs_plus[42] = ((BnewGUTs_plus[42] / muGUT_original) - DeltaB) * muGUT_original;
            if ((isnan(BnewGUTs_plus[42]) || (isnan(BnewGUTs_plus[43])))) {
                BplusEWSB = false;
                break;
            }
            numStepsDone++;
        }
        if (!BplusEWSB) {
            // std::cout << "Failed to converge" << endl;
            BnewGUTs_plus[6] = BoldGUTs_plus[6];
            BnewGUTs_plus[42] = BoldGUTs_plus[42];
            BnewGUTs_plus[43] = BoldGUTs_plus[43];
            break;
        }
        if ((BnewGUTs_plus[43] < 3.0) || (BnewGUTs_plus[43] > 60.0)) {
            BplusEWSB = false;
        }
    }
    std::cout << "B(ABDS, plus) = " << BnewGUTs_plus[42] / BnewGUTs_plus[6] << endl;
    double B_GUT_plus = BnewGUTs_plus[42] / BnewGUTs_plus[6];
    if (abs(B_GUT_plus - (GUT_boundary_conditions[42] / GUT_boundary_conditions[6])) < 1.0e-9) {
        B_GUT_plus = 1.0001 * B_GUT_plus;
    }
    Bcurr_tanbGUT = BnewGUTs_minus[43];
    Bnew_tanbGUT = Bcurr_tanbGUT;
    BcurrentlogQGUT = current_logQGUT;
    BcurrentlogQSUSY = current_logQSUSY;

    std::cout << "ABDS window established for B variation." << endl;
    
    bool ABDSminuscheck = (BminusEWSB && BminusNoCCB); 
    bool ABDSpluscheck = (BplusEWSB && BplusNoCCB);
    double B_TOTAL_GUT_minus, B_TOTAL_GUT_plus;
    if (!(ABDSminuscheck) && !(ABDSpluscheck)) {
        if (abs(B_GUT_minus) <= abs(B_GUT_plus)) {
            B_TOTAL_GUT_minus = pow(10.0, -0.5) * B_GUT_minus;
            B_TOTAL_GUT_plus = pow(10.0, 0.5) * B_GUT_plus;
        } else {
            B_TOTAL_GUT_minus = pow(10.0, 0.5) * B_GUT_minus;
            B_TOTAL_GUT_plus = pow(10.0, -0.5) * B_GUT_plus;
        }

        std::cout << "General window established for B variation." << endl;

        return {B_GUT_minus, B_GUT_plus, B_TOTAL_GUT_minus, B_TOTAL_GUT_plus};
    }

    while ((BminusEWSB) && (BminusNoCCB) && ((Bnew_mZ2minus > (5.0)))) {
        lambdaB = 0.5;
        B_least_Sq_Tol = 1.0e-2;
        curr_lsq_eval = std::numeric_limits<double>::max();
        Bnew_mZ2minus *= 0.99;
        // std::cout << "New mZ = " << sqrt(Bnew_mZ2minus) << endl;
        // std::cout << "New B = " << BnewGUTs_minus[42] / BnewGUTs_minus[6] << endl;
        int numStepsDone = 0;
        vector<double> checkweaksols = solveODEs(BnewGUTs_minus, BcurrentlogQGUT, BcurrentlogQSUSY, copysign(1.0e-6, (BcurrentlogQSUSY - BcurrentlogQGUT)));
        vector<double> checkRadCorrs = radcorr_calc(checkweaksols, exp(BnewlogQSUSY), Bnew_mZ2minus);
        BminusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
        if (BminusEWSB == true) {
            BminusEWSB = Hessian_check(checkweaksols, exp(BnewlogQSUSY));
        }
        // if (BminusEWSB == true) {
        //     BminusEWSB = BFB_check(checkweaksols);
        // }
        BminusNoCCB = CCB_Check(checkweaksols);
        if (!(BminusEWSB) || !(BminusNoCCB)) {
            break;
        }
        vector<double> tree_level_masses = TreeMassCalculator(checkweaksols, exp(BcurrentlogQSUSY), Bnew_mZ2minus);
        for (double value : tree_level_masses) {
            if (value < 0) {
                BminusNoCCB = false;
            }
        }
        if (!(BminusNoCCB)) {
            break;
        }                
        vector<double> BoldGUTs_minus = BnewGUTs_minus;
        while ((numStepsDone < 100) && (curr_lsq_eval > B_least_Sq_Tol)) {
            vector<double> current_derivatives = single_var_deriv_approxes(BnewGUTs_minus, Bnew_mZ2minus, 42, BnewlogQSUSY, BnewlogQGUT);
            
            vector<double> weaksol_minus = solveODEs(BnewGUTs_minus, BcurrentlogQGUT, BcurrentlogQSUSY, copysign(1.0e-6, (BcurrentlogQSUSY - BcurrentlogQGUT)));
            vector<double> RadCorrsMinus = radcorr_calc(weaksol_minus, exp(BnewlogQSUSY), Bnew_mZ2minus);
            
            double FB = (1.0 / 2.0) - (((weaksol_minus[26] + RadCorrsMinus[1] - ((weaksol_minus[25] + RadCorrsMinus[0]) * pow(weaksol_minus[43], 2.0))) / (Bnew_mZ2minus * (pow(weaksol_minus[43], 2.0) - 1.0))) - (weaksol_minus[6] * weaksol_minus[6] / Bnew_mZ2minus));
            double GB = weaksol_minus[43] - tan(0.5 * (M_PI - asin(2.0 * weaksol_minus[42] / (weaksol_minus[25] + RadCorrsMinus[0] + weaksol_minus[26] + RadCorrsMinus[1] + (2.0 * pow(weaksol_minus[6], 2.0))))));

            double DeltaTanbNum = ((current_derivatives[0] * GB) - (current_derivatives[2] * FB));
            double DeltaDenom = ((current_derivatives[0] * current_derivatives[3]) - (current_derivatives[1] * current_derivatives[2]));
            double DeltaTanb = DeltaTanbNum / DeltaDenom;

            double DeltaBNum = ((current_derivatives[3] * FB) - (current_derivatives[1] * GB));
            double DeltaB = DeltaBNum / DeltaDenom;

            curr_lsq_eval = pow(DeltaB, 2.0) + pow(DeltaTanb, 2.0);
            BnewGUTs_minus[43] = BnewGUTs_minus[43] - DeltaTanb;
            BnewGUTs_minus[42] = ((BnewGUTs_minus[42] / muGUT_original) - DeltaB) * muGUT_original;
            if ((isnan(BnewGUTs_minus[42]) || (isnan(BnewGUTs_minus[43])))) {
                BminusEWSB = false;
                break;
            }
            numStepsDone++;
        }
        if (!BminusEWSB) {
            // std::cout << "Failed to converge" << endl;
            BnewGUTs_minus[6] = BoldGUTs_minus[6];
            BnewGUTs_minus[42] = BoldGUTs_minus[42];
            BnewGUTs_minus[43] = BoldGUTs_minus[43];
            break;
        }
        if ((BnewGUTs_minus[43] < 3.0) || (BnewGUTs_minus[43] > 60.0)) {
            BminusEWSB = false;
        }
    }
    std::cout << "B(total, minus) = " << BnewGUTs_minus[42] / BnewGUTs_minus[6] << endl;
    B_TOTAL_GUT_minus = BnewGUTs_minus[42] / BnewGUTs_minus[6];

    while ((BplusEWSB) && (BplusNoCCB) && ((Bnew_mZ2plus > (5.0)))) {
        lambdaB = 0.5;
        B_least_Sq_Tol = 1.0e-2;
        curr_lsq_eval = std::numeric_limits<double>::max();
        Bnew_mZ2plus *= 1.01;
        //std::cout << "New mZ = " << sqrt(Bnew_mZ2plus) << endl;
        int numStepsDone = 0;
        vector<double> checkweaksols = solveODEs(BnewGUTs_plus, BcurrentlogQGUT, BcurrentlogQSUSY, copysign(1.0e-6, (BcurrentlogQSUSY - BcurrentlogQGUT)));
        vector<double> checkRadCorrs = radcorr_calc(checkweaksols, exp(BnewlogQSUSY), Bnew_mZ2plus);
        BplusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
        if (BplusEWSB == true) {
            BplusEWSB = Hessian_check(checkweaksols, exp(BnewlogQSUSY));
        }
        // if (BplusEWSB == true) {
        //     BplusEWSB = BFB_check(checkweaksols);
        // }
        BplusNoCCB = CCB_Check(checkweaksols);
        if (!(BplusEWSB) || !(BplusNoCCB)) {
            break;
        }
        vector<double> tree_level_masses = TreeMassCalculator(checkweaksols, exp(BcurrentlogQSUSY), Bnew_mZ2plus);
        for (double value : tree_level_masses) {
            if (value < 0) {
                BplusNoCCB = false;
            }
        }
        if (!(BplusNoCCB)) {
            break;
        }             
        vector<double> BoldGUTs_plus = BnewGUTs_plus;   
        while ((numStepsDone < 100) && (curr_lsq_eval > B_least_Sq_Tol)) {
            vector<double> current_derivatives = single_var_deriv_approxes(BnewGUTs_plus, Bnew_mZ2plus, 42, BnewlogQSUSY, BnewlogQGUT);
            
            vector<double> weaksol_plus = solveODEs(BnewGUTs_plus, BcurrentlogQGUT, BcurrentlogQSUSY, copysign(1.0e-6, (BcurrentlogQSUSY - BcurrentlogQGUT)));
            vector<double> RadCorrsPlus = radcorr_calc(weaksol_plus, exp(BnewlogQSUSY), Bnew_mZ2plus);
            
            double FB = (1.0 / 2.0) - (((weaksol_plus[26] + RadCorrsPlus[1] - ((weaksol_plus[25] + RadCorrsPlus[0]) * pow(weaksol_plus[43], 2.0))) / (Bnew_mZ2plus * (pow(weaksol_plus[43], 2.0) - 1.0))) - (weaksol_plus[6] * weaksol_plus[6] / Bnew_mZ2plus));
            double GB = weaksol_plus[43] - tan(0.5 * (M_PI - asin(2.0 * weaksol_plus[42] / (weaksol_plus[25] + RadCorrsPlus[0] + weaksol_plus[26] + RadCorrsPlus[1] + (2.0 * pow(weaksol_plus[6], 2.0))))));

            double DeltaTanbNum = ((current_derivatives[0] * GB) - (current_derivatives[2] * FB));
            double DeltaDenom = ((current_derivatives[0] * current_derivatives[3]) - (current_derivatives[1] * current_derivatives[2]));
            double DeltaTanb = DeltaTanbNum / DeltaDenom;

            double DeltaBNum = ((current_derivatives[3] * FB) - (current_derivatives[1] * GB));
            double DeltaB = DeltaBNum / DeltaDenom;

            curr_lsq_eval = pow(DeltaB, 2.0) + pow(DeltaTanb, 2.0);
            BnewGUTs_plus[43] = BnewGUTs_plus[43] - DeltaTanb;
            BnewGUTs_plus[42] = ((BnewGUTs_plus[42] / muGUT_original) - DeltaB) * muGUT_original;
            if ((isnan(BnewGUTs_plus[42]) || (isnan(BnewGUTs_plus[43])))) {
                BplusEWSB = false;
                break;
            }
            numStepsDone++;
        }
        if (!BplusEWSB) {
            // std::cout << "Failed to converge" << endl;
            BnewGUTs_plus[6] = BoldGUTs_plus[6];
            BnewGUTs_plus[42] = BoldGUTs_plus[42];
            BnewGUTs_plus[43] = BoldGUTs_plus[43];
            break;
        }
        if ((BnewGUTs_plus[43] < 3.0) || (BnewGUTs_plus[43] > 60.0)) {
            BplusEWSB = false;
        }
    }
    std::cout << "B(total, plus) = " << BnewGUTs_plus[42] / BnewGUTs_plus[6] << endl;
    B_TOTAL_GUT_plus = BnewGUTs_plus[42] / BnewGUTs_plus[6];

    if ((abs(B_TOTAL_GUT_minus - B_GUT_minus) < 1.0e-12) && (abs(B_TOTAL_GUT_plus - B_GUT_plus) < 1.0e-12)) {
        if (abs(B_GUT_minus) <= abs(B_GUT_plus)) {
            B_TOTAL_GUT_minus = pow(10.0, -0.5) * B_GUT_minus;
            B_TOTAL_GUT_plus = pow(10.0, 0.5) * B_GUT_plus;
        } else {
            B_TOTAL_GUT_minus = pow(10.0, 0.5) * B_GUT_minus;
            B_TOTAL_GUT_plus = pow(10.0, -0.5) * B_GUT_plus;
        }
        std::cout << "General window established for B variation." << endl;

        return {B_GUT_minus, B_GUT_plus, B_TOTAL_GUT_minus, B_TOTAL_GUT_plus};
    }
    std::cout << "General window established for B variation." << endl;

    return {B_GUT_minus, B_GUT_plus, B_TOTAL_GUT_minus, B_TOTAL_GUT_plus};
}

vector<double> DSN_specific_windows(vector<double>& GUT_boundary_conditions, double& current_mZ2, double& current_logQSUSY, double& current_logQGUT, int SpecificIndex) {
    double t_target = log(500.0);
    vector<double> pinewGUTs_plus = GUT_boundary_conditions;
    vector<double> pinewGUTs_minus = GUT_boundary_conditions;
    double picurrentlogQGUT = current_logQGUT;
    double picurrentlogQSUSY = current_logQSUSY;
    double pinewlogQGUT = current_logQGUT;
    double pinewlogQSUSY = current_logQSUSY;
    double pinew_mZ2plus = current_mZ2;
    double pinew_mZ2minus = current_mZ2;
    bool piminusNoCCB = true;
    bool piminusEWSB = true;
    bool piplusNoCCB = true;
    bool piplusEWSB = true;
    string paramName;
    if (SpecificIndex == 3) {
        paramName = "M1";
    } else if (SpecificIndex == 4) {
        paramName = "M2";
    } else if (SpecificIndex == 5) {
        paramName = "M3";
    } else if (SpecificIndex == 16) {
        paramName = "a_t";
    } else if (SpecificIndex == 17) {
        paramName = "a_c";
    } else if (SpecificIndex == 18) {
        paramName = "a_u";
    } else if (SpecificIndex == 19) {
        paramName = "a_b";
    } else if (SpecificIndex == 20) {
        paramName = "a_s";
    } else if (SpecificIndex == 21) {
        paramName = "a_d";
    } else if (SpecificIndex == 22) {
        paramName = "a_tau";
    } else if (SpecificIndex == 23) {
        paramName = "a_mu";
    } else if (SpecificIndex == 24) {
        paramName = "a_e";
    } else if (SpecificIndex == 25) {
        paramName = "mHu^2";
    } else if (SpecificIndex == 26) {
        paramName = "mHd^2";
    } else if (SpecificIndex == 27) {
        paramName = "mQ1^2";
    } else if (SpecificIndex == 28) {
        paramName = "mQ2^2";
    } else if (SpecificIndex == 29) {
        paramName = "mQ3^2";
    } else if (SpecificIndex == 30) {
        paramName = "mL1^2";
    } else if (SpecificIndex == 31) {
        paramName = "mL2^2";
    } else if (SpecificIndex == 32) {
        paramName = "mL3^2";
    } else if (SpecificIndex == 33) {
        paramName = "mU1^2";
    } else if (SpecificIndex == 34) {
        paramName = "mU2^2";
    } else if (SpecificIndex == 35) {
        paramName = "mU3^2";
    } else if (SpecificIndex == 36) {
        paramName = "mD1^2";
    } else if (SpecificIndex == 37) {
        paramName = "mD2^2";
    } else if (SpecificIndex == 38) {
        paramName = "mD3^2";
    } else if (SpecificIndex == 39) {
        paramName = "mE1^2";
    } else if (SpecificIndex == 40) {
        paramName = "mE2^2";
    } else {
        paramName = "mE3^2";
    }

    double piplus = pinewGUTs_plus[SpecificIndex];
    double newpiplus = piplus;
    double tanbplus = pinewGUTs_plus[43];
    double newtanbplus = tanbplus;

    double piminus = pinewGUTs_minus[SpecificIndex];
    double newpiminus = piminus;
    double tanbminus = pinewGUTs_minus[43];
    double newtanbminus = tanbminus;

    // First compute width of ABDS window
    double lambdapi = 0.5;
    double pi_least_Sq_Tol = 1.0e-2;
    double prev_fpi = std::numeric_limits<double>::max();
    double curr_lsq_eval = std::numeric_limits<double>::max();
    while ((piminusEWSB) && (piminusNoCCB) && ((pinew_mZ2minus > (45.5938 * 45.5938)) && (pinew_mZ2minus < (364.7504 * 364.7504)))) {
        lambdapi = 0.5;
        pi_least_Sq_Tol = 1.0e-2;
        curr_lsq_eval = std::numeric_limits<double>::max();
        prev_fpi = std::numeric_limits<double>::max();
        pinew_mZ2minus *= 0.99;
        int numStepsDone = 0;
        std::cout << "New mZ = " << sqrt(pinew_mZ2minus) << endl;
        std::cout << "New " << paramName << "(GUT) = " << pinewGUTs_minus[SpecificIndex] << endl;
        std::cout << "New tanb = " << pinewGUTs_minus[43] << endl;
        // std::cout << "---------------------------------------" << endl;
        vector<double> checkweaksols = solveODEs(pinewGUTs_minus, picurrentlogQGUT, picurrentlogQSUSY, copysign(1.0e-6, (picurrentlogQSUSY - picurrentlogQGUT)));
        vector<double> checkRadCorrs = radcorr_calc(checkweaksols, exp(pinewlogQSUSY), pinew_mZ2minus);
        piminusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
        // Checking loop-level EWSB
        if (piminusEWSB == true) {
            piminusEWSB = Hessian_check(checkweaksols, exp(pinewlogQSUSY));
        }
        // if (piminusEWSB == true) {
        //     piminusEWSB = BFB_check(checkweaksols);
        // }
        piminusNoCCB = CCB_Check(checkweaksols);
        if (!(piminusEWSB) || !(piminusNoCCB)) {
            break;
        } 
        vector<double> tree_level_masses = TreeMassCalculator(checkweaksols, exp(picurrentlogQSUSY), pinew_mZ2minus);
        for (double value : tree_level_masses) {
            if (value < 0) {
                piminusNoCCB = false;
            }
        }
        if (!(piminusNoCCB)) {
            break;
        } 
        vector<double> oldSolutions = pinewGUTs_minus;
        while ((numStepsDone < 100) && (curr_lsq_eval > pi_least_Sq_Tol)) {
            vector<double> current_derivatives = single_var_deriv_approxes(pinewGUTs_minus, pinew_mZ2minus, SpecificIndex, pinewlogQSUSY, pinewlogQGUT);
            
            vector<double> weaksol_minus = solveODEs(pinewGUTs_minus, picurrentlogQGUT, picurrentlogQSUSY, copysign(1.0e-6, (picurrentlogQSUSY - picurrentlogQGUT)));
            vector<double> RadCorrsMinus = radcorr_calc(weaksol_minus, exp(pinewlogQSUSY), pinew_mZ2minus);
            
            double FMi = (1.0 / 2.0) - (((weaksol_minus[26] + RadCorrsMinus[1] - ((weaksol_minus[25] + RadCorrsMinus[0]) * pow(weaksol_minus[43], 2.0))) / (pinew_mZ2minus * (pow(weaksol_minus[43], 2.0) - 1.0))) - (weaksol_minus[6] * weaksol_minus[6] / pinew_mZ2minus));
            double GMi = weaksol_minus[43] - tan(0.5 * (M_PI - asin(2.0 * weaksol_minus[42] / (weaksol_minus[25] + RadCorrsMinus[0] + weaksol_minus[26] + RadCorrsMinus[1] + (2.0 * pow(weaksol_minus[6], 2.0))))));

            double DeltaTanbNum = ((current_derivatives[0] * GMi) - (current_derivatives[2] * FMi));
            double DeltaDenom = ((current_derivatives[0] * current_derivatives[3]) - (current_derivatives[1] * current_derivatives[2]));
            double DeltaTanb = DeltaTanbNum / DeltaDenom;

            double DeltaPiNum = ((current_derivatives[3] * FMi) - (current_derivatives[1] * GMi));
            double DeltaPi = DeltaPiNum / DeltaDenom;
            curr_lsq_eval = pow(DeltaPi, 2.0) + pow(DeltaTanb, 2.0);
            pinewGUTs_minus[43] = pinewGUTs_minus[43] - DeltaTanb;
            pinewGUTs_minus[SpecificIndex] = pinewGUTs_minus[SpecificIndex] - DeltaPi;
            newpiminus = pinewGUTs_minus[SpecificIndex];
            if ((isnan(pinewGUTs_minus[SpecificIndex])) || (isnan(pinewGUTs_minus[43]))) {
                piminusEWSB = false;
                break;
            }
            numStepsDone++;
        }
        if (!piminusEWSB) {
            // std::cout << "Failed to converge" << endl;
            pinewGUTs_minus[SpecificIndex] = oldSolutions[SpecificIndex];
            pinewGUTs_minus[43] = oldSolutions[43];
            break;
        }
        vector<double> check_valid_solutions = get_F_G_vals(pinewGUTs_minus, pinew_mZ2minus, current_logQSUSY, current_logQGUT, SpecificIndex);
        if ((abs(check_valid_solutions[0]) > 1.0e-2) || (abs(check_valid_solutions[1]) > 1.0e-2)) {
            // std::cout << "Failed to converge" << endl;
            piminusEWSB = false;
            pinewGUTs_minus[SpecificIndex] = oldSolutions[SpecificIndex];
            pinewGUTs_minus[43] = oldSolutions[43];
        }
        if ((pinewGUTs_minus[43] < 3.0) || (pinewGUTs_minus[43] > 60.0)) {
            piminusEWSB = false;
        }
    }
    std::cout << paramName << "(ABDS, minus) = " << pinewGUTs_minus[SpecificIndex] << endl; 
    double pi_GUT_minus = pinewGUTs_minus[SpecificIndex];
    if (abs(pi_GUT_minus - GUT_boundary_conditions[SpecificIndex]) < 1.0e-9) {
        pi_GUT_minus = 0.9999 * pi_GUT_minus;
    }
    lambdapi = 0.5;
    pi_least_Sq_Tol = 1.0e-2;
    prev_fpi = std::numeric_limits<double>::max();
    while ((piplusEWSB) && (piplusNoCCB) && ((pinew_mZ2plus > (45.5938 * 45.5938)) && (pinew_mZ2plus < (364.7504 * 364.7504)))) {
        lambdapi = 0.5;
        pi_least_Sq_Tol = 1.0e-2;
        curr_lsq_eval = std::numeric_limits<double>::max();
        prev_fpi = std::numeric_limits<double>::max();
        pinew_mZ2plus *= 1.01;
        int numStepsDone = 0;
        std::cout << "New mZ = " << sqrt(pinew_mZ2plus) << endl;
        std::cout << "New " << paramName << "(GUT) = " << pinewGUTs_plus[SpecificIndex] << endl;
        std::cout << "New tanb = " << pinewGUTs_plus[43] << endl;
        // std::cout << "---------------------------------------" << endl;
        vector<double> checkweaksols = solveODEs(pinewGUTs_plus, picurrentlogQGUT, picurrentlogQSUSY, copysign(1.0e-6, (picurrentlogQSUSY - picurrentlogQGUT)));
        vector<double> checkRadCorrs = radcorr_calc(checkweaksols, exp(pinewlogQSUSY), pinew_mZ2plus);
        piplusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
        // Checking loop-level EWSB
        if (piplusEWSB == true) {
            piplusEWSB = Hessian_check(checkweaksols, exp(pinewlogQSUSY));
        }
        // if (piplusEWSB == true) {
        //     piplusEWSB = BFB_check(checkweaksols);
        // }
        piplusNoCCB = CCB_Check(checkweaksols);
        if (!(piplusEWSB) || !(piplusNoCCB)) {
            break;
        } 
        vector<double> tree_level_masses = TreeMassCalculator(checkweaksols, exp(picurrentlogQSUSY), pinew_mZ2plus);
        for (double value : tree_level_masses) {
            if (value < 0) {
                piplusNoCCB = false;
            }
        }
        if (!(piplusNoCCB)) {
            break;
        } 
                // Finish out convergence with Newton-Raphson now that we have a good enough initial guess
        vector<double> oldSolutions = pinewGUTs_plus;
        while ((numStepsDone < 100) && (curr_lsq_eval > pi_least_Sq_Tol)) {
            //std::cout << numStepsDone << " steps done" << endl;
            vector<double> current_derivatives = single_var_deriv_approxes(pinewGUTs_plus, pinew_mZ2plus, SpecificIndex, pinewlogQSUSY, pinewlogQGUT);

            vector<double> weaksol_plus = solveODEs(pinewGUTs_plus, picurrentlogQGUT, picurrentlogQSUSY, copysign(1.0e-6, (picurrentlogQSUSY - picurrentlogQGUT)));
            vector<double> RadCorrsplus = radcorr_calc(weaksol_plus, exp(pinewlogQSUSY), pinew_mZ2plus);
            
            double FMi = (1.0 / 2.0) - (((weaksol_plus[26] + RadCorrsplus[1] - ((weaksol_plus[25] + RadCorrsplus[0]) * pow(weaksol_plus[43], 2.0))) / (pinew_mZ2plus * (pow(weaksol_plus[43], 2.0) - 1.0))) - (weaksol_plus[6] * weaksol_plus[6] / pinew_mZ2plus));
            double GMi = weaksol_plus[43] - tan(0.5 * (M_PI - asin(2.0 * weaksol_plus[42] / (weaksol_plus[25] + RadCorrsplus[0] + weaksol_plus[26] + RadCorrsplus[1] + (2.0 * pow(weaksol_plus[6], 2.0))))));

            double DeltaTanbNum = ((current_derivatives[0] * GMi) - (current_derivatives[2] * FMi));
            double DeltaDenom = ((current_derivatives[0] * current_derivatives[3]) - (current_derivatives[1] * current_derivatives[2]));
            double DeltaTanb = DeltaTanbNum / DeltaDenom;

            double DeltaPiNum = ((current_derivatives[3] * FMi) - (current_derivatives[1] * GMi));
            double DeltaPi = DeltaPiNum / DeltaDenom;

            curr_lsq_eval = pow(DeltaPi, 2.0) + pow(DeltaTanb, 2.0);
            pinewGUTs_plus[43] = pinewGUTs_plus[43] - DeltaTanb;
            pinewGUTs_plus[SpecificIndex] = pinewGUTs_plus[SpecificIndex] - DeltaPi;
                        
            newpiplus = pinewGUTs_plus[SpecificIndex];
            if ((isnan(pinewGUTs_plus[SpecificIndex])) || (isnan(pinewGUTs_plus[43]))) {
                piplusEWSB = false;
                break;
            }
            numStepsDone++;
        }
        if (!piplusEWSB) {
            // std::cout << "Failed to converge" << endl;
            pinewGUTs_plus[SpecificIndex] = oldSolutions[SpecificIndex];
            pinewGUTs_plus[43] = oldSolutions[43];
            break;
        }  
        if ((pinewGUTs_plus[43] < 3.0) || (pinewGUTs_plus[43] > 60.0)) {
            piplusEWSB = false;
        }
        vector<double> check_valid_solutions = get_F_G_vals(pinewGUTs_plus, pinew_mZ2plus, current_logQSUSY, current_logQGUT, SpecificIndex);
        if ((abs(check_valid_solutions[0]) > 1.0e-2) || (abs(check_valid_solutions[1]) > 1.0e-2) || (isnan(check_valid_solutions[0])) || (isnan(check_valid_solutions[1]))) {
            // std::cout << "Failed to converge" << endl;
            piplusEWSB = false;
            pinewGUTs_plus[SpecificIndex] = oldSolutions[SpecificIndex];
            pinewGUTs_plus[43] = oldSolutions[43];
        }         
    }
    std::cout << paramName << "(ABDS, plus) = " << pinewGUTs_plus[SpecificIndex] << endl; 
    double pi_GUT_plus = pinewGUTs_plus[SpecificIndex];
    if (abs(pi_GUT_plus - GUT_boundary_conditions[SpecificIndex]) < 1.0e-9) {
        pi_GUT_plus = 1.0001 * pi_GUT_plus;
    }

    std::cout << "ABDS window established for " << paramName << " variation." << endl;

    bool ABDSminuscheck = (piminusEWSB && piminusNoCCB); 
    bool ABDSpluscheck = (piplusEWSB && piplusNoCCB);
    double pi_TOTAL_GUT_minus, pi_TOTAL_GUT_plus;
    if (!(ABDSminuscheck) && !(ABDSpluscheck)) {
        if (abs(pi_GUT_minus) <= abs(pi_GUT_plus)) {
            pi_TOTAL_GUT_minus = pow(10.0, -0.5) * pi_TOTAL_GUT_minus;
            pi_TOTAL_GUT_plus = pow(10.0, 0.5) * pi_TOTAL_GUT_plus;
        } else {
            pi_TOTAL_GUT_minus = pow(10.0, 0.5) * pi_TOTAL_GUT_minus;
            pi_TOTAL_GUT_plus = pow(10.0, -0.5) * pi_TOTAL_GUT_plus;
        }

        std::cout << "General window established for " << paramName << " variation." << endl;

        return {pi_GUT_minus, pi_GUT_plus, pi_TOTAL_GUT_minus, pi_TOTAL_GUT_plus};
    }

    while ((piminusEWSB) && (piminusNoCCB) && ((pinew_mZ2minus > (5.0)))) {
        lambdapi = 0.5;
        pi_least_Sq_Tol = 1.0e-2;
        curr_lsq_eval = std::numeric_limits<double>::max();
        prev_fpi = std::numeric_limits<double>::max();
        pinew_mZ2minus *= 0.99;
        int numStepsDone = 0;
        // std::cout << "New mZ = " << sqrt(pinew_mZ2minus) << endl;
        // std::cout << "New " << paramName << "(GUT) = " << pinewGUTs_minus[SpecificIndex] << endl;
        // std::cout << "---------------------------------------" << endl;
        vector<double> checkweaksols = solveODEs(pinewGUTs_minus, picurrentlogQGUT, picurrentlogQSUSY, copysign(1.0e-6, (picurrentlogQSUSY - picurrentlogQGUT)));
        vector<double> checkRadCorrs = radcorr_calc(checkweaksols, exp(pinewlogQSUSY), pinew_mZ2minus);
        piminusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
        // Checking loop-level EWSB
        if (piminusEWSB == true) {
            piminusEWSB = Hessian_check(checkweaksols, exp(pinewlogQSUSY));
        }
        piminusNoCCB = CCB_Check(checkweaksols);
        if (!(piminusEWSB) || !(piminusNoCCB)) {
            break;
        } 
        vector<double> tree_level_masses = TreeMassCalculator(checkweaksols, exp(picurrentlogQSUSY), pinew_mZ2minus);
        for (double value : tree_level_masses) {
            if (value < 0) {
                piminusNoCCB = false;
            }
        }
        if (!(piminusNoCCB)) {
            break;
        } 
        vector<double> oldSolutions = pinewGUTs_minus;
        while ((numStepsDone < 100) && (curr_lsq_eval > pi_least_Sq_Tol)) {
            vector<double> current_derivatives = single_var_deriv_approxes(pinewGUTs_minus, pinew_mZ2minus, SpecificIndex, pinewlogQSUSY, pinewlogQGUT);
            
            vector<double> weaksol_minus = solveODEs(pinewGUTs_minus, picurrentlogQGUT, picurrentlogQSUSY, copysign(1.0e-6, (picurrentlogQSUSY - picurrentlogQGUT)));
            vector<double> RadCorrsMinus = radcorr_calc(weaksol_minus, exp(pinewlogQSUSY), pinew_mZ2minus);
            
            double FMi = (1.0 / 2.0) - (((weaksol_minus[26] + RadCorrsMinus[1] - ((weaksol_minus[25] + RadCorrsMinus[0]) * pow(weaksol_minus[43], 2.0))) / (pinew_mZ2minus * (pow(weaksol_minus[43], 2.0) - 1.0))) - (weaksol_minus[6] * weaksol_minus[6] / pinew_mZ2minus));
            double GMi = weaksol_minus[43] - tan(0.5 * (M_PI - asin(2.0 * weaksol_minus[42] / (weaksol_minus[25] + RadCorrsMinus[0] + weaksol_minus[26] + RadCorrsMinus[1] + (2.0 * pow(weaksol_minus[6], 2.0))))));

            double DeltaTanbNum = ((current_derivatives[0] * GMi) - (current_derivatives[2] * FMi));
            double DeltaDenom = ((current_derivatives[0] * current_derivatives[3]) - (current_derivatives[1] * current_derivatives[2]));
            double DeltaTanb = DeltaTanbNum / DeltaDenom;

            double DeltaPiNum = ((current_derivatives[3] * FMi) - (current_derivatives[1] * GMi));
            double DeltaPi = DeltaPiNum / DeltaDenom;

            curr_lsq_eval = pow(DeltaPi, 2.0) + pow(DeltaTanb, 2.0);
            pinewGUTs_minus[43] = pinewGUTs_minus[43] - DeltaTanb;
            pinewGUTs_minus[SpecificIndex] = pinewGUTs_minus[SpecificIndex] - DeltaPi;
            newpiminus = pinewGUTs_minus[SpecificIndex];
            if ((isnan(pinewGUTs_minus[SpecificIndex])) || (isnan(pinewGUTs_minus[43]))) {
                piminusEWSB = false;
                break;
            }
            numStepsDone++;
        }
        if (!piminusEWSB) {
            // std::cout << "Failed to converge" << endl;
            pinewGUTs_minus[SpecificIndex] = oldSolutions[SpecificIndex];
            pinewGUTs_minus[43] = oldSolutions[43];
            break;
        }
        vector<double> check_valid_solutions = get_F_G_vals(pinewGUTs_minus, pinew_mZ2minus, current_logQSUSY, current_logQGUT, SpecificIndex);
        if ((abs(check_valid_solutions[0]) > 1.0e-2) || (abs(check_valid_solutions[1]) > 1.0e-2)) {
            // std::cout << "Failed to converge" << endl;
            piminusEWSB = false;
            pinewGUTs_minus[SpecificIndex] = oldSolutions[SpecificIndex];
            pinewGUTs_minus[43] = oldSolutions[43];
        }
        if ((pinewGUTs_minus[43] < 3.0) || (pinewGUTs_minus[43] > 60.0)) {
            piminusEWSB = false;
        }
    }
    std::cout << paramName << "(total, minus) = " << pinewGUTs_minus[SpecificIndex] << endl; 
    pi_TOTAL_GUT_minus = pinewGUTs_minus[SpecificIndex];

    while ((piplusEWSB) && (piplusNoCCB) && ((pinew_mZ2plus > (5.0)))) {
        lambdapi = 0.5;
        pi_least_Sq_Tol = 1.0e-2;
        curr_lsq_eval = std::numeric_limits<double>::max();
        prev_fpi = std::numeric_limits<double>::max();
        pinew_mZ2plus *= 1.01;
        int numStepsDone = 0;
        // std::cout << "New mZ = " << sqrt(pinew_mZ2plus) << endl;
        // std::cout << "New " << paramName << "(GUT) = " << pinewGUTs_plus[SpecificIndex] << endl;
        // std::cout << "---------------------------------------" << endl;
        vector<double> checkweaksols = solveODEs(pinewGUTs_plus, picurrentlogQGUT, picurrentlogQSUSY, copysign(1.0e-6, (picurrentlogQSUSY - picurrentlogQGUT)));
        vector<double> checkRadCorrs = radcorr_calc(checkweaksols, exp(pinewlogQSUSY), pinew_mZ2plus);
        piplusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
        // Checking loop-level EWSB
        if (piplusEWSB == true) {
            piplusEWSB = Hessian_check(checkweaksols, exp(pinewlogQSUSY));
        }
        // if (piplusEWSB == true) {
        //     piplusEWSB = BFB_check(checkweaksols);
        // }
        piplusNoCCB = CCB_Check(checkweaksols);
        if (!(piplusEWSB) || !(piplusNoCCB)) {
            break;
        } 
        vector<double> tree_level_masses = TreeMassCalculator(checkweaksols, exp(picurrentlogQSUSY), pinew_mZ2plus);
        for (double value : tree_level_masses) {
            if (value < 0) {
                piplusNoCCB = false;
            }
        }
        if (!(piplusNoCCB)) {
            break;
        } 
                // Finish out convergence with Newton-Raphson now that we have a good enough initial guess
        vector<double> oldSolutions = pinewGUTs_plus;
        while ((numStepsDone < 100) && (curr_lsq_eval > pi_least_Sq_Tol)) {
            //std::cout << numStepsDone << " steps done" << endl;
            vector<double> current_derivatives = single_var_deriv_approxes(pinewGUTs_plus, pinew_mZ2plus, SpecificIndex, pinewlogQSUSY, pinewlogQGUT);

            vector<double> weaksol_plus = solveODEs(pinewGUTs_plus, picurrentlogQGUT, picurrentlogQSUSY, copysign(1.0e-6, (picurrentlogQSUSY - picurrentlogQGUT)));
            vector<double> RadCorrsplus = radcorr_calc(weaksol_plus, exp(pinewlogQSUSY), pinew_mZ2plus);
            
            double FMi = (1.0 / 2.0) - (((weaksol_plus[26] + RadCorrsplus[1] - ((weaksol_plus[25] + RadCorrsplus[0]) * pow(weaksol_plus[43], 2.0))) / (pinew_mZ2plus * (pow(weaksol_plus[43], 2.0) - 1.0))) - (weaksol_plus[6] * weaksol_plus[6] / pinew_mZ2plus));
            double GMi = weaksol_plus[43] - tan(0.5 * (M_PI - asin(2.0 * weaksol_plus[42] / (weaksol_plus[25] + RadCorrsplus[0] + weaksol_plus[26] + RadCorrsplus[1] + (2.0 * pow(weaksol_plus[6], 2.0))))));

            double DeltaTanbNum = ((current_derivatives[0] * GMi) - (current_derivatives[2] * FMi));
            double DeltaDenom = ((current_derivatives[0] * current_derivatives[3]) - (current_derivatives[1] * current_derivatives[2]));
            double DeltaTanb = DeltaTanbNum / DeltaDenom;

            double DeltaPiNum = ((current_derivatives[3] * FMi) - (current_derivatives[1] * GMi));
            double DeltaPi = DeltaPiNum / DeltaDenom;

            curr_lsq_eval = pow(DeltaPi, 2.0) + pow(DeltaTanb, 2.0);
            pinewGUTs_plus[43] = pinewGUTs_plus[43] - DeltaTanb;
            pinewGUTs_plus[SpecificIndex] = pinewGUTs_plus[SpecificIndex] - DeltaPi;
                        
            newpiplus = pinewGUTs_plus[SpecificIndex];
            if ((isnan(pinewGUTs_plus[SpecificIndex])) || (isnan(pinewGUTs_plus[43]))) {
                piplusEWSB = false;
                break;
            }
            numStepsDone++;
        }
        if (!piplusEWSB) {
            // std::cout << "Failed to converge" << endl;
            pinewGUTs_plus[SpecificIndex] = oldSolutions[SpecificIndex];
            pinewGUTs_plus[43] = oldSolutions[43];
            break;
        }  
        if ((pinewGUTs_plus[43] < 3.0) || (pinewGUTs_plus[43] > 60.0)) {
            piplusEWSB = false;
        }
        vector<double> check_valid_solutions = get_F_G_vals(pinewGUTs_plus, pinew_mZ2plus, current_logQSUSY, current_logQGUT, SpecificIndex);
        if ((abs(check_valid_solutions[0]) > 1.0e-2) || (abs(check_valid_solutions[1]) > 1.0e-2) || (isnan(check_valid_solutions[0])) || (isnan(check_valid_solutions[1]))) {
            // std::cout << "Failed to converge" << endl;
            piplusEWSB = false;
            pinewGUTs_plus[SpecificIndex] = oldSolutions[SpecificIndex];
            pinewGUTs_plus[43] = oldSolutions[43];
        }         
    }
    std::cout << paramName << "(total, plus) = " << pinewGUTs_plus[SpecificIndex] << endl; 
    pi_TOTAL_GUT_plus = pinewGUTs_plus[SpecificIndex];

    if ((abs(pi_TOTAL_GUT_minus - pi_GUT_minus) < 1.0e-12) && (abs(pi_TOTAL_GUT_plus - pi_GUT_plus) < 1.0e-12)) {
        if (abs(pi_GUT_minus) <= abs(pi_GUT_plus)) {
            pi_TOTAL_GUT_minus = pow(10.0, -0.5) * pi_GUT_minus;
            pi_TOTAL_GUT_plus = pow(10.0, 0.5) * pi_GUT_plus;
        } else {
            pi_TOTAL_GUT_minus = pow(10.0, 0.5) * pi_GUT_minus;
            pi_TOTAL_GUT_plus = pow(10.0, -0.5) * pi_GUT_plus;
        }

        std::cout << "General window established for " << paramName << " variation." << endl;

        return {pi_GUT_minus, pi_GUT_plus, pi_TOTAL_GUT_minus, pi_TOTAL_GUT_plus};
    }

    std::cout << "General window established for " << paramName << " variation." << endl;

    return {pi_GUT_minus, pi_GUT_plus, pi_TOTAL_GUT_minus, pi_TOTAL_GUT_plus};
}

vector<double> DSN_mu_windows(vector<double>& GUT_boundary_conditions, double& current_mZ2, double& current_logQSUSY, double& current_logQGUT) {
    double t_target = log(500.0);
    vector<double> munewGUTs_plus = GUT_boundary_conditions;
    vector<double> munewGUTs_minus = GUT_boundary_conditions;
    double mucurrentlogQGUT = current_logQGUT;
    double mucurrentlogQSUSY = current_logQSUSY;
    double munewlogQGUT = current_logQGUT;
    double munewlogQSUSY = current_logQSUSY;
    double munew_mZ2plus = current_mZ2;
    double munew_mZ2minus = current_mZ2;
    bool muminusNoCCB = true;
    bool muminusEWSB = true;
    bool muplusNoCCB = true;
    bool muplusEWSB = true;

    double muplus = munewGUTs_plus[6];
    double newmuplus = muplus;
    double tanbplus = munewGUTs_plus[43];
    double newtanbplus = tanbplus;

    double muminus = munewGUTs_minus[6];
    double newmuminus = muminus;
    double tanbminus = munewGUTs_minus[43];
    double newtanbminus = tanbminus;
    double BGUT_original = GUT_boundary_conditions[42] / GUT_boundary_conditions[6];

    // First compute width of ABDS window
    double lambdaMu = 0.5;
    double Mu_least_Sq_Tol = 1.0e-2;
    double prev_fmu = std::numeric_limits<double>::max();
    double curr_lsq_eval = std::numeric_limits<double>::max();
    vector<double> current_derivatives = single_var_deriv_approxes(munewGUTs_minus, munew_mZ2minus, 6, munewlogQSUSY, munewlogQGUT);
    for (double deriv_value : current_derivatives) {
        //std::cout << "Derivative: " << deriv_value << endl;
        if (isnan(deriv_value) || isinf(deriv_value)) {
            muminusEWSB = false;
        }
    }
    double mustep, bigmustep;
    mustep = (boost::math::float_next(abs(munewGUTs_minus[6])) - abs(munewGUTs_minus[6])) * abs(munewGUTs_minus[6]);
        
    double mZ2shift_minus = ((((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (-1.0 * mustep))
                           + (0.5 * mustep * mustep * ((current_derivatives[1] * current_derivatives[2])
                                                       + (current_derivatives[0] * current_derivatives[0] * current_derivatives[4])
                                                       + (2.0 * current_derivatives[0] * current_derivatives[5]) + current_derivatives[6])));
    
    std::cout << "mZ^2 minus shift: " << mZ2shift_minus << endl;
    if (abs(mZ2shift_minus) > 1.0) {
        mZ2shift_minus = (((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (-1.0 * mustep));
    }
    double tanbshift_minus = (current_derivatives[0] * (-1.0) * mustep) + (0.5 * current_derivatives[1] * mustep * mustep);
    std::cout << "tanb minus shift: " << tanbshift_minus << endl;
    if (abs(tanbshift_minus) > 1.0) {
        tanbshift_minus = (current_derivatives[0] * (-1.0) * mustep);
    }
    bool too_sensitive_flag = false;
    double mu_GUT_minus, mu_GUT_plus;
    if ((abs(mZ2shift_minus) > 1.0) || (abs(tanbshift_minus) > 1.0)) {
        std::cout << "Sensitivity too high, approximating solution" << endl;
        too_sensitive_flag = true;
        mu_GUT_minus = munewGUTs_minus[6] - mustep;
    } 
    // Mu convergence becomes bad when mu is small (i.e. < 10 GeV), so cutoff at abs(mu) = 10 GeV
    while ((!too_sensitive_flag) && (muminusEWSB) && (muminusNoCCB) && (abs(munewGUTs_minus[6]) > 10.0) && ((munew_mZ2minus > (45.5938 * 45.5938)) && (munew_mZ2minus < (364.7504 * 364.7504)))) {
        int numStepsDone = 0;
        bigmustep = mustep * ((2.0 * sqrt(munew_mZ2minus)) + 1.0) / abs(mZ2shift_minus);
        std::cout << "New mZ = " << sqrt(munew_mZ2minus) << endl;
        // std::cout << "New mu = " << munewGUTs_minus[6] << endl;
        // std::cout << "New tanb = " << munewGUTs_minus[43] << endl;
        vector<double> checkweaksols = solveODEs(munewGUTs_minus, mucurrentlogQGUT, mucurrentlogQSUSY, copysign(1.0e-6, (mucurrentlogQSUSY - mucurrentlogQGUT)));
        vector<double> checkRadCorrs = radcorr_calc(checkweaksols, exp(munewlogQSUSY), munew_mZ2minus);
        muminusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
        // Checking loop-level EWSB
        if (muminusEWSB == true) {
            muminusEWSB = Hessian_check(checkweaksols, exp(munewlogQSUSY));
        }
        muminusNoCCB = CCB_Check(checkweaksols);
        if (!(muminusEWSB) || !(muminusNoCCB)) {
            break;
        } 
        if (!(muminusNoCCB)) {
            break;
        } 
        // std::cout << "Mu step size = " << mustep << endl;
        vector<double> muoldGUTs_minus = munewGUTs_minus;
        munewGUTs_minus[6] -= bigmustep;
        munewGUTs_minus[42] = (munewGUTs_minus[42] / muoldGUTs_minus[6]) * munewGUTs_minus[6];
        
        
        if (!(muminusEWSB)) {
            munewGUTs_minus[6] = muoldGUTs_minus[6];
            break;
        }
        munew_mZ2minus += copysign((2.0 * sqrt(munew_mZ2minus)) + 1.0, mZ2shift_minus);
        munewGUTs_minus[43] += (tanbshift_minus * ((2.0 * sqrt(munew_mZ2minus)) + 1.0) / abs(mZ2shift_minus));
        // Now adjust Yukawas for next iteration.
        if ((munewGUTs_minus[43] < 3.0) || (munewGUTs_minus[43] > 60.0)) {
            // std::cout << "Yukawas non-perturbative" << endl;
            muminusEWSB = false;
        } else {
            vector<double> oldmZsols = solveODEs(muoldGUTs_minus, current_logQGUT, log(173.2), -1.0e-6);
            vector<double> newmZsols = solveODEs(munewGUTs_minus, current_logQGUT, log(173.2), -1.0e-6);
            for (int YukIndx = 7; YukIndx < 16; ++YukIndx) {
                if ((YukIndx >=7) && (YukIndx < 10)) {
                    newmZsols[YukIndx] *= sin(atan(oldmZsols[43])) / sin(atan(newmZsols[43]));
                } else {
                    newmZsols[YukIndx] *= cos(atan(oldmZsols[43])) / cos(atan(newmZsols[43]));
                }
            }
            vector<double> adjustedYukGUTs = solveODEs(newmZsols, log(173.2), current_logQGUT, 1.0e-6);
            for (int YukIndex = 7; YukIndex < 16; ++YukIndex) {
                munewGUTs_minus[YukIndex] = adjustedYukGUTs[YukIndex];
            }
        }        

        if (!muminusEWSB) {
            // std::cout << "EWSB issue in convergence loop" << endl;
            munewGUTs_minus[6] = muoldGUTs_minus[6];
            munewGUTs_minus[42] = muoldGUTs_minus[42];
            munewGUTs_minus[43] = muoldGUTs_minus[43];
            break;
        }
        mustep = (boost::math::float_next(abs(munewGUTs_minus[6])) - abs(munewGUTs_minus[6])) * abs(munewGUTs_minus[6]);
        current_derivatives = single_var_deriv_approxes(munewGUTs_minus, munew_mZ2minus, 6, munewlogQSUSY, munewlogQGUT);
    
        mZ2shift_minus = ((((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (-1.0 * mustep))
                           + (0.5 * mustep * mustep * ((current_derivatives[1] * current_derivatives[2])
                                                       + (current_derivatives[0] * current_derivatives[0] * current_derivatives[4])
                                                       + (2.0 * current_derivatives[0] * current_derivatives[5]) + current_derivatives[6])));
        if (abs(mZ2shift_minus) > 1.0) {
            mZ2shift_minus = (((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (-1.0 * mustep));
        }
        tanbshift_minus = (current_derivatives[0] * (-1.0) * mustep) + (0.5 * current_derivatives[1] * mustep * mustep);
        if (abs(tanbshift_minus) > 1.0) {
            tanbshift_minus = (current_derivatives[0] * (-1.0) * mustep);
        }
        if ((abs(mZ2shift_minus) > 1.0) || (abs(tanbshift_minus) > 1.0)) {
            std::cout << "Sensitivity too high, approximating solution" << endl;
            too_sensitive_flag = true;
            //mu_GUT_minus = munewGUTs_minus[6] - mustep;
        } 
    }
    std::cout << "mu(ABDS, minus) = " << munewGUTs_minus[6] << endl; 
    if (!too_sensitive_flag) {
        mu_GUT_minus = munewGUTs_minus[6];
    }

    double mZ2shift_plus = ((((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (mustep))
                           + (0.5 * mustep * mustep * ((current_derivatives[1] * current_derivatives[2])
                                                       + (current_derivatives[0] * current_derivatives[0] * current_derivatives[4])
                                                       + (2.0 * current_derivatives[0] * current_derivatives[5]) + current_derivatives[6])));
    double tanbshift_plus = (current_derivatives[0] * mustep) + (0.5 * current_derivatives[1] * mustep * mustep);

    std::cout << "mZ^2 plus shift: " << mZ2shift_plus << endl;
    if (abs(mZ2shift_plus) > 1.0) {
        mZ2shift_plus = (((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (mustep));
    }
    std::cout << "tanb plus shift: " << tanbshift_plus << endl;
    if (abs(tanbshift_plus) > 1.0) {
        tanbshift_plus = (current_derivatives[0] * mustep);
    }
    if ((abs(mZ2shift_plus) > 1.0) || (abs(tanbshift_plus) > 1.0)) {
        std::cout << "Sensitivity too high, approximating solution" << endl;
        too_sensitive_flag = true;
        mu_GUT_plus = munewGUTs_plus[6] + mustep;
    }
    // Mu convergence becomes bad when mu is small (i.e. < 10 GeV), so cutoff at abs(mu) = 10 GeV
    while ((!too_sensitive_flag) && (muplusEWSB) && (muplusNoCCB) && (abs(munewGUTs_plus[6]) > 10.0) && ((munew_mZ2plus > (45.5938 * 45.5938)) && (munew_mZ2plus < (364.7504 * 364.7504)))) {
        int numStepsDone = 0;
        bigmustep = mustep * ((2.0 * sqrt(munew_mZ2plus)) + 1.0) / abs(mZ2shift_plus);
        std::cout << "New mZ = " << sqrt(munew_mZ2plus) << endl;
        // std::cout << "New mu = " << munewGUTs_plus[6] << endl;
        // std::cout << "New tanb = " << munewGUTs_plus[43] << endl;
        vector<double> checkweaksols = solveODEs(munewGUTs_plus, mucurrentlogQGUT, mucurrentlogQSUSY, copysign(1.0e-6, (mucurrentlogQSUSY - mucurrentlogQGUT)));
        vector<double> checkRadCorrs = radcorr_calc(checkweaksols, exp(munewlogQSUSY), munew_mZ2plus);
        muplusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
        // Checking loop-level EWSB
        if (muplusEWSB == true) {
            muplusEWSB = Hessian_check(checkweaksols, exp(munewlogQSUSY));
        }
        muplusNoCCB = CCB_Check(checkweaksols);
        if (!(muplusEWSB) || !(muplusNoCCB)) {
            break;
        } 
        if (!(muplusNoCCB)) {
            break;
        } 
        // std::cout << "Mu step size = " << mustep << endl;
        vector<double> muoldGUTs_plus = munewGUTs_plus;
        munewGUTs_plus[6] += bigmustep;
        munewGUTs_plus[42] = (munewGUTs_plus[42] / muoldGUTs_plus[6]) * munewGUTs_plus[6];
        
        
        if (!(muplusEWSB)) {
            munewGUTs_plus[6] = muoldGUTs_plus[6];
            break;
        }
        munew_mZ2plus += copysign((2.0 * sqrt(munew_mZ2plus)) + 1.0, mZ2shift_plus);
        munewGUTs_plus[43] += (tanbshift_plus * ((2.0 * sqrt(munew_mZ2plus)) + 1.0) / abs(mZ2shift_plus));
        // Now adjust Yukawas for next iteration.
        if ((munewGUTs_plus[43] < 3.0) || (munewGUTs_plus[43] > 60.0)) {
            // std::cout << "Yukawas non-perturbative" << endl;
            muplusEWSB = false;
        } else {
            vector<double> oldmZsols = solveODEs(muoldGUTs_plus, current_logQGUT, log(173.2), -1.0e-6);
            vector<double> newmZsols = solveODEs(munewGUTs_plus, current_logQGUT, log(173.2), -1.0e-6);
            for (int YukIndx = 7; YukIndx < 16; ++YukIndx) {
                if ((YukIndx >=7) && (YukIndx < 10)) {
                    newmZsols[YukIndx] *= sin(atan(oldmZsols[43])) / sin(atan(newmZsols[43]));
                } else {
                    newmZsols[YukIndx] *= cos(atan(oldmZsols[43])) / cos(atan(newmZsols[43]));
                }
            }
            vector<double> adjustedYukGUTs = solveODEs(newmZsols, log(173.2), current_logQGUT, 1.0e-6);
            for (int YukIndex = 7; YukIndex < 16; ++YukIndex) {
                munewGUTs_plus[YukIndex] = adjustedYukGUTs[YukIndex];
            }
        }        

        if (!muplusEWSB) {
            // std::cout << "EWSB issue in convergence loop" << endl;
            munewGUTs_plus[6] = muoldGUTs_plus[6];
            munewGUTs_plus[42] = muoldGUTs_plus[42];
            munewGUTs_plus[43] = muoldGUTs_plus[43];
            break;
        }
        mustep = (boost::math::float_next(abs(munewGUTs_plus[6])) - abs(munewGUTs_plus[6])) * abs(munewGUTs_plus[6]);
        current_derivatives = single_var_deriv_approxes(munewGUTs_plus, munew_mZ2plus, 6, munewlogQSUSY, munewlogQGUT);
    
        mZ2shift_plus = ((((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (mustep))
                           + (0.5 * mustep * mustep * ((current_derivatives[1] * current_derivatives[2])
                                                       + (current_derivatives[0] * current_derivatives[0] * current_derivatives[4])
                                                       + (2.0 * current_derivatives[0] * current_derivatives[5]) + current_derivatives[6])));
        if (abs(mZ2shift_plus) > 1.0) {
            mZ2shift_plus = (((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (mustep));
        }
        tanbshift_plus = (current_derivatives[0] * mustep) + (0.5 * current_derivatives[1] * mustep * mustep);
        if (abs(tanbshift_plus) > 1.0) {
            tanbshift_plus = (current_derivatives[0] * mustep);
        }
        if ((abs(mZ2shift_plus) > 1.0) || (abs(tanbshift_plus) > 1.0)) {
            std::cout << "Sensitivity too high, approximating solution" << endl;
            too_sensitive_flag = true;
            //mu_GUT_plus = munewGUTs_plus[6] + mustep;
        } 
    }
    std::cout << "mu(ABDS, plus) = " << munewGUTs_plus[6] << endl; 
    if (!too_sensitive_flag) {
        mu_GUT_plus = munewGUTs_plus[6];
    }

    std::cout << "ABDS window established for mu variation." << endl;

    bool ABDSminuscheck = (muminusEWSB && muminusNoCCB); 
    bool ABDSpluscheck = (muplusEWSB && muplusNoCCB);
    bool total_ABDScheck = (ABDSminuscheck && ABDSpluscheck);
    double mu_TOTAL_GUT_minus, mu_TOTAL_GUT_plus;
    if ((!(total_ABDScheck)) || too_sensitive_flag) {
        if (abs(mu_GUT_minus) <= abs(mu_GUT_plus)) {
            mu_TOTAL_GUT_minus = 10.0;//0.000001 * mu_GUT_minus;//pow(10.0, -0.5) * mu_GUT_minus;
            mu_TOTAL_GUT_plus = 1.0e16;//1000000.0 * mu_GUT_plus;//pow(10.0, 0.5) * mu_GUT_plus;
        } else {
            mu_TOTAL_GUT_minus = 1.0e16;//1000000.0 * mu_GUT_minus;//pow(10.0, 0.5) * mu_GUT_minus;
            mu_TOTAL_GUT_plus = 10.0;//0.000001 * mu_GUT_plus;//pow(10.0, -0.5) * mu_GUT_plus;
        }

        std::cout << "General window established for mu variation." << endl;

        return {mu_GUT_minus, mu_GUT_plus, mu_TOTAL_GUT_minus, mu_TOTAL_GUT_plus};
    }

    while ((muminusEWSB) && (muminusNoCCB) && (abs(munewGUTs_minus[6]) > 10.0) && ((munew_mZ2minus > (5.0)))) {
        int numStepsDone = 0;
        bigmustep = mustep * ((2.0 * sqrt(munew_mZ2minus)) + 1.0) / abs(mZ2shift_minus);
        std::cout << "New mZ = " << sqrt(munew_mZ2minus) << endl;
        // std::cout << "New mu = " << munewGUTs_minus[6] << endl;
        // std::cout << "New tanb = " << munewGUTs_minus[43] << endl;
        vector<double> checkweaksols = solveODEs(munewGUTs_minus, mucurrentlogQGUT, mucurrentlogQSUSY, copysign(1.0e-6, (mucurrentlogQSUSY - mucurrentlogQGUT)));
        vector<double> checkRadCorrs = radcorr_calc(checkweaksols, exp(munewlogQSUSY), munew_mZ2minus);
        muminusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
        // Checking loop-level EWSB
        if (muminusEWSB == true) {
            muminusEWSB = Hessian_check(checkweaksols, exp(munewlogQSUSY));
        }
        muminusNoCCB = CCB_Check(checkweaksols);
        if (!(muminusEWSB) || !(muminusNoCCB)) {
            break;
        } 
        if (!(muminusNoCCB)) {
            break;
        } 
        // std::cout << "Mu step size = " << mustep << endl;
        vector<double> muoldGUTs_minus = munewGUTs_minus;
        munewGUTs_minus[6] -= bigmustep;
        munewGUTs_minus[42] = (munewGUTs_minus[42] / muoldGUTs_minus[6]) * munewGUTs_minus[6];
        
        
        if (!(muminusEWSB)) {
            munewGUTs_minus[6] = muoldGUTs_minus[6];
            break;
        }
        munew_mZ2minus += copysign((2.0 * sqrt(munew_mZ2minus)) + 1.0, mZ2shift_minus);
        munewGUTs_minus[43] += (tanbshift_minus * ((2.0 * sqrt(munew_mZ2minus)) + 1.0) / abs(mZ2shift_minus));
        // Now adjust Yukawas for next iteration.
        if ((munewGUTs_minus[43] < 3.0) || (munewGUTs_minus[43] > 60.0)) {
            // std::cout << "Yukawas non-perturbative" << endl;
            muminusEWSB = false;
        } else {
            vector<double> oldmZsols = solveODEs(muoldGUTs_minus, current_logQGUT, log(173.2), -1.0e-6);
            vector<double> newmZsols = solveODEs(munewGUTs_minus, current_logQGUT, log(173.2), -1.0e-6);
            for (int YukIndx = 7; YukIndx < 16; ++YukIndx) {
                if ((YukIndx >=7) && (YukIndx < 10)) {
                    newmZsols[YukIndx] *= sin(atan(oldmZsols[43])) / sin(atan(newmZsols[43]));
                } else {
                    newmZsols[YukIndx] *= cos(atan(oldmZsols[43])) / cos(atan(newmZsols[43]));
                }
            }
            vector<double> adjustedYukGUTs = solveODEs(newmZsols, log(173.2), current_logQGUT, 1.0e-6);
            for (int YukIndex = 7; YukIndex < 16; ++YukIndex) {
                munewGUTs_minus[YukIndex] = adjustedYukGUTs[YukIndex];
            }
        }        

        if (!muminusEWSB) {
            // std::cout << "EWSB issue in convergence loop" << endl;
            munewGUTs_minus[6] = muoldGUTs_minus[6];
            munewGUTs_minus[42] = muoldGUTs_minus[42];
            munewGUTs_minus[43] = muoldGUTs_minus[43];
            break;
        }
        mustep = (boost::math::float_next(abs(munewGUTs_minus[6])) - abs(munewGUTs_minus[6])) * abs(munewGUTs_minus[6]);
        current_derivatives = single_var_deriv_approxes(munewGUTs_minus, munew_mZ2minus, 6, munewlogQSUSY, munewlogQGUT);
    
        mZ2shift_minus = ((((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (-1.0 * mustep))
                           + (0.5 * mustep * mustep * ((current_derivatives[1] * current_derivatives[2])
                                                       + (current_derivatives[0] * current_derivatives[0] * current_derivatives[4])
                                                       + (2.0 * current_derivatives[0] * current_derivatives[5]) + current_derivatives[6])));
        if (abs(mZ2shift_minus) > 1.0) {
            mZ2shift_minus = (((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (-1.0 * mustep));
        }
        tanbshift_minus = (current_derivatives[0] * (-1.0) * mustep) + (0.5 * current_derivatives[1] * mustep * mustep);
        if (abs(tanbshift_minus) > 1.0) {
            tanbshift_minus = (current_derivatives[0] * (-1.0) * mustep);
        }
        if ((abs(mZ2shift_minus) > 1.0) || (abs(tanbshift_minus) > 1.0)) {
            std::cout << "Sensitivity too high, approximating solution" << endl;
            too_sensitive_flag = true;
            //mu_GUT_minus = munewGUTs_minus[6] - mustep;
        } 
    }
    std::cout << "mu(total, minus) = " << munewGUTs_minus[6] << endl; 
    mu_TOTAL_GUT_minus = munewGUTs_minus[6];

    while ((muplusEWSB) && (muplusNoCCB) && (abs(munewGUTs_plus[6]) > 10.0) && ((munew_mZ2plus > (5.0)))) {
        int numStepsDone = 0;
        bigmustep = mustep * ((2.0 * sqrt(munew_mZ2plus)) + 1.0) / abs(mZ2shift_plus);
        std::cout << "New mZ = " << sqrt(munew_mZ2plus) << endl;
        // std::cout << "New mu = " << munewGUTs_plus[6] << endl;
        // std::cout << "New tanb = " << munewGUTs_plus[43] << endl;
        vector<double> checkweaksols = solveODEs(munewGUTs_plus, mucurrentlogQGUT, mucurrentlogQSUSY, copysign(1.0e-6, (mucurrentlogQSUSY - mucurrentlogQGUT)));
        vector<double> checkRadCorrs = radcorr_calc(checkweaksols, exp(munewlogQSUSY), munew_mZ2plus);
        muplusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
        // Checking loop-level EWSB
        if (muplusEWSB == true) {
            muplusEWSB = Hessian_check(checkweaksols, exp(munewlogQSUSY));
        }
        muplusNoCCB = CCB_Check(checkweaksols);
        if (!(muplusEWSB) || !(muplusNoCCB)) {
            break;
        } 
        if (!(muplusNoCCB)) {
            break;
        } 
        // std::cout << "Mu step size = " << mustep << endl;
        vector<double> muoldGUTs_plus = munewGUTs_plus;
        munewGUTs_plus[6] += bigmustep;
        munewGUTs_plus[42] = (munewGUTs_plus[42] / muoldGUTs_plus[6]) * munewGUTs_plus[6];
        
        
        if (!(muplusEWSB)) {
            munewGUTs_plus[6] = muoldGUTs_plus[6];
            break;
        }
        munew_mZ2plus += copysign((2.0 * sqrt(munew_mZ2plus)) + 1.0, mZ2shift_plus);
        munewGUTs_plus[43] += (tanbshift_plus * ((2.0 * sqrt(munew_mZ2plus)) + 1.0) / abs(mZ2shift_plus));
        // Now adjust Yukawas for next iteration.
        if ((munewGUTs_plus[43] < 3.0) || (munewGUTs_plus[43] > 60.0)) {
            // std::cout << "Yukawas non-perturbative" << endl;
            muplusEWSB = false;
        } else {
            vector<double> oldmZsols = solveODEs(muoldGUTs_plus, current_logQGUT, log(173.2), -1.0e-6);
            vector<double> newmZsols = solveODEs(munewGUTs_plus, current_logQGUT, log(173.2), -1.0e-6);
            for (int YukIndx = 7; YukIndx < 16; ++YukIndx) {
                if ((YukIndx >=7) && (YukIndx < 10)) {
                    newmZsols[YukIndx] *= sin(atan(oldmZsols[43])) / sin(atan(newmZsols[43]));
                } else {
                    newmZsols[YukIndx] *= cos(atan(oldmZsols[43])) / cos(atan(newmZsols[43]));
                }
            }
            vector<double> adjustedYukGUTs = solveODEs(newmZsols, log(173.2), current_logQGUT, 1.0e-6);
            for (int YukIndex = 7; YukIndex < 16; ++YukIndex) {
                munewGUTs_plus[YukIndex] = adjustedYukGUTs[YukIndex];
            }
        }        

        if (!muplusEWSB) {
            // std::cout << "EWSB issue in convergence loop" << endl;
            munewGUTs_plus[6] = muoldGUTs_plus[6];
            munewGUTs_plus[42] = muoldGUTs_plus[42];
            munewGUTs_plus[43] = muoldGUTs_plus[43];
            break;
        }
        mustep = (boost::math::float_next(abs(munewGUTs_plus[6])) - abs(munewGUTs_plus[6])) * abs(munewGUTs_plus[6]);
        current_derivatives = single_var_deriv_approxes(munewGUTs_plus, munew_mZ2plus, 6, munewlogQSUSY, munewlogQGUT);
    
        mZ2shift_plus = ((((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (mustep))
                           + (0.5 * mustep * mustep * ((current_derivatives[1] * current_derivatives[2])
                                                       + (current_derivatives[0] * current_derivatives[0] * current_derivatives[4])
                                                       + (2.0 * current_derivatives[0] * current_derivatives[5]) + current_derivatives[6])));
        if (abs(mZ2shift_plus) > 1.0) {
            mZ2shift_plus = (((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (mustep));
        }
        tanbshift_plus = (current_derivatives[0] * mustep) + (0.5 * current_derivatives[1] * mustep * mustep);
        if (abs(tanbshift_plus) > 1.0) {
            tanbshift_plus = (current_derivatives[0] * mustep);
        }
        if ((abs(mZ2shift_plus) > 1.0) || (abs(tanbshift_plus) > 1.0)) {
            std::cout << "Sensitivity too high, approximating solution" << endl;
            too_sensitive_flag = true;
            //mu_GUT_plus = munewGUTs_plus[6] + mustep;
        } 
    }
    std::cout << "mu(total, plus) = " << munewGUTs_plus[6] << endl; 
    mu_TOTAL_GUT_plus = munewGUTs_plus[6];

    if ((abs(mu_TOTAL_GUT_minus - mu_GUT_minus) < 1.0e-12) && (abs(mu_TOTAL_GUT_plus - mu_GUT_plus) < 1.0e-12)) {
        if (abs(mu_GUT_minus) <= abs(mu_GUT_plus)) {
            mu_TOTAL_GUT_minus = 10.0;//0.000001 * mu_GUT_minus;//pow(10.0, -0.5) * mu_GUT_minus;
            mu_TOTAL_GUT_plus = 1.0e16;//1000000.0 * mu_GUT_plus;//pow(10.0, 0.5) * mu_GUT_plus;
        } else {
            mu_TOTAL_GUT_minus = 1.0e16;//1000000.0 * mu_GUT_minus;//pow(10.0, 0.5) * mu_GUT_minus;
            mu_TOTAL_GUT_plus = 10.0;//0.000001 * mu_GUT_plus;//pow(10.0, -0.5) * mu_GUT_plus;
        }

        std::cout << "General window established for mu variation." << endl;

        return {mu_GUT_minus, mu_GUT_plus, mu_TOTAL_GUT_minus, mu_TOTAL_GUT_plus};
    }

    std::cout << "General window established for mu variation." << endl;

    return {mu_GUT_minus, mu_GUT_plus, mu_TOTAL_GUT_minus, mu_TOTAL_GUT_plus};
}

double DSN_calc(int precselno, std::vector<double> GUT_boundary_conditions,
                double& current_mZ2, double& current_logQSUSY,
                double& current_logQGUT, int& nF, int& nD) {
    double DSN, DSN_soft_num, DSN_soft_denom, DSN_higgsino, newterm;
    DSN = 0.0;
    double t_target = log(500.0);
    std::cout << "This may take a while...\n\nProgress:\n-----------------------------------------------\n" << endl;
    if ((precselno == 1) || (precselno == 2)) {
        // Compute mu windows around original point
        vector<double> muinitGUTBCs = GUT_boundary_conditions;
        vector<double> muwindows = DSN_mu_windows(muinitGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT);
        DSN_higgsino = abs(log10(abs(muwindows[1] / muwindows[0])));
        DSN_higgsino /= abs(muwindows[1] - muwindows[0]);
        newterm = DSN_higgsino;
        // Total normalization
        DSN_higgsino = abs(log10(abs(muwindows[3] / muwindows[2])));
        DSN_higgsino /= abs(muwindows[3] - muwindows[2]);
        if ((abs(DSN_higgsino - newterm) < numeric_limits<double>::epsilon()) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan(DSN_higgsino)) || (DSN_higgsino == 0.0) || isinf(DSN_higgsino)) {
            newterm = abs(log10(1.0 + (numeric_limits<double>::epsilon() * abs(GUT_boundary_conditions[6]))))\
                / abs(numeric_limits<double>::epsilon() * abs(GUT_boundary_conditions[6]));
            DSN_higgsino = 1.0 / abs(((pow(10.0, 0.5) - pow(10.0, -0.5))) * abs(GUT_boundary_conditions[6]));
        }
        DSN += abs(log10(abs(DSN_higgsino)) - log10(abs(newterm)));
        std::cout << "DSN after higgsino = " << DSN << endl;
        
        // Now do same thing with mHu^2(GUT)
        vector<double> mHu2initGUTBCs = GUT_boundary_conditions;
        vector<double> mHu2windows = DSN_specific_windows(mHu2initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 25);
        DSN_soft_denom = abs(copysign(sqrt(abs(mHu2windows[1])), mHu2windows[1]) - copysign(sqrt(abs(mHu2windows[0])), mHu2windows[0]));
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mHu2windows[1])), mHu2windows[1]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mHu2windows[0])), mHu2windows[0]), (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mHu2windows[3])), mHu2windows[3]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mHu2windows[2])), mHu2windows[2]), (2.0 * nF) + (1.0 * nD) - 1.0);
        DSN_soft_denom = abs(copysign(sqrt(abs(mHu2windows[3])), mHu2windows[3]) - copysign(sqrt(abs(mHu2windows[2])), mHu2windows[2]));
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(copysign(sqrt(abs(boost::math::float_next(GUT_boundary_conditions[25]))), boost::math::float_next(GUT_boundary_conditions[25])), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(copysign(sqrt(abs(boost::math::float_prior(GUT_boundary_conditions[25]))), boost::math::float_prior(GUT_boundary_conditions[25])), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(copysign(sqrt(abs(GUT_boundary_conditions[25])), GUT_boundary_conditions[25])) - boost::math::float_prior(copysign(sqrt(abs(GUT_boundary_conditions[25])), GUT_boundary_conditions[25]))));
            DSN_soft_num = soft_prob_calc(copysign(pow(10.0, 0.5) * sqrt(abs(GUT_boundary_conditions[25])), GUT_boundary_conditions[25]), (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(copysign(pow(10.0, -0.5) * sqrt(abs(GUT_boundary_conditions[25])), GUT_boundary_conditions[25]), (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with mHd^2(GUT)
        vector<double> mHd2initGUTBCs = GUT_boundary_conditions;
        vector<double> mHd2windows = DSN_specific_windows(mHd2initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 26);
        DSN_soft_denom = abs(copysign(sqrt(abs(mHd2windows[1])), mHd2windows[1]) - copysign(sqrt(abs(mHd2windows[0])), mHd2windows[0]));
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mHd2windows[1])), mHd2windows[1]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mHd2windows[0])), mHd2windows[0]), (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mHd2windows[3])), mHd2windows[3]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mHd2windows[2])), mHd2windows[2]), (2.0 * nF) + (1.0 * nD) - 1.0);
        DSN_soft_denom = abs(copysign(sqrt(abs(mHd2windows[3])), mHd2windows[3]) - copysign(sqrt(abs(mHd2windows[2])), mHd2windows[2]));
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(copysign(sqrt(abs(boost::math::float_next(GUT_boundary_conditions[26]))), boost::math::float_next(GUT_boundary_conditions[26])), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(copysign(sqrt(abs(boost::math::float_prior(GUT_boundary_conditions[26]))), boost::math::float_prior(GUT_boundary_conditions[26])), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(copysign(sqrt(abs(GUT_boundary_conditions[26])), GUT_boundary_conditions[26])) - boost::math::float_prior(copysign(sqrt(abs(GUT_boundary_conditions[26])), GUT_boundary_conditions[26]))));
            DSN_soft_num = soft_prob_calc(copysign(pow(10.0, 0.5) * sqrt(abs(GUT_boundary_conditions[26])), GUT_boundary_conditions[26]), (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(copysign(pow(10.0, -0.5) * sqrt(abs(GUT_boundary_conditions[26])), GUT_boundary_conditions[26]), (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with M1
        vector<double> M1initGUTBCs = GUT_boundary_conditions;
        vector<double> M1windows = DSN_specific_windows(M1initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 3);
        DSN_soft_denom = abs(M1windows[1] - M1windows[0]);
        DSN_soft_num = soft_prob_calc(M1windows[1], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(M1windows[0], (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_denom = abs(M1windows[3] - M1windows[2]);
        DSN_soft_num = soft_prob_calc(M1windows[3], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(M1windows[2], (2.0 * nF) + (1.0 * nD) - 1.0);
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(boost::math::float_next(GUT_boundary_conditions[3]), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(boost::math::float_prior(GUT_boundary_conditions[3]), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(GUT_boundary_conditions[3]) - boost::math::float_prior(GUT_boundary_conditions[3])));
            DSN_soft_num = soft_prob_calc(pow(10.0, 0.5) * GUT_boundary_conditions[3], (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(pow(10.0, -0.5) * GUT_boundary_conditions[3], (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with M2
        vector<double> M2initGUTBCs = GUT_boundary_conditions;
        vector<double> M2windows = DSN_specific_windows(M2initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 4);
        DSN_soft_denom = abs(M2windows[1] - M2windows[0]);
        DSN_soft_num = soft_prob_calc(M2windows[1], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(M2windows[0], (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_denom = abs(M2windows[3] - M2windows[2]);
        DSN_soft_num = soft_prob_calc(M2windows[3], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(M2windows[2], (2.0 * nF) + (1.0 * nD) - 1.0);
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(boost::math::float_next(GUT_boundary_conditions[4]), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(boost::math::float_prior(GUT_boundary_conditions[4]), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(GUT_boundary_conditions[4]) - boost::math::float_prior(GUT_boundary_conditions[4])));
            DSN_soft_num = soft_prob_calc(pow(10.0, 0.5) * GUT_boundary_conditions[4], (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(pow(10.0, -0.5) * GUT_boundary_conditions[4], (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with M3
        vector<double> M3initGUTBCs = GUT_boundary_conditions;
        vector<double> M3windows = DSN_specific_windows(M3initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 5);
        DSN_soft_denom = abs(M3windows[1] - M3windows[0]);
        DSN_soft_num = soft_prob_calc(M3windows[1], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(M3windows[0], (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_denom = abs(M3windows[3] - M3windows[2]);
        DSN_soft_num = soft_prob_calc(M3windows[3], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(M3windows[2], (2.0 * nF) + (1.0 * nD) - 1.0);
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(boost::math::float_next(GUT_boundary_conditions[5]), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(boost::math::float_prior(GUT_boundary_conditions[5]), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(GUT_boundary_conditions[5]) - boost::math::float_prior(GUT_boundary_conditions[5])));
            DSN_soft_num = soft_prob_calc(pow(10.0, 0.5) * GUT_boundary_conditions[5], (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(pow(10.0, -0.5) * GUT_boundary_conditions[5], (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with mQ3
        vector<double> MQ3initGUTBCs = GUT_boundary_conditions;
        vector<double> MQ3windows = DSN_specific_windows(MQ3initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 29);
        DSN_soft_denom = abs(copysign(sqrt(abs(MQ3windows[1])), MQ3windows[1])  - copysign(sqrt(abs(MQ3windows[0])), MQ3windows[0]));
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(MQ3windows[1])), MQ3windows[1]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(MQ3windows[0])), MQ3windows[0]), (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(MQ3windows[3])), MQ3windows[3]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(MQ3windows[2])), MQ3windows[2]), (2.0 * nF) + (1.0 * nD) - 1.0);
        DSN_soft_denom = abs(copysign(sqrt(abs(MQ3windows[3])), MQ3windows[3]) - copysign(sqrt(abs(MQ3windows[2])), MQ3windows[2]));
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(copysign(sqrt(abs(boost::math::float_next(GUT_boundary_conditions[29]))), boost::math::float_next(GUT_boundary_conditions[29])), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(copysign(sqrt(abs(boost::math::float_prior(GUT_boundary_conditions[29]))), boost::math::float_prior(GUT_boundary_conditions[29])), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(copysign(sqrt(abs(GUT_boundary_conditions[29])), GUT_boundary_conditions[29])) - boost::math::float_prior(copysign(sqrt(abs(GUT_boundary_conditions[29])), GUT_boundary_conditions[29]))));
            DSN_soft_num = soft_prob_calc(copysign(pow(10.0, 0.5) * sqrt(abs(GUT_boundary_conditions[29])), GUT_boundary_conditions[29]), (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(copysign(pow(10.0, -0.5) * sqrt(abs(GUT_boundary_conditions[29])), GUT_boundary_conditions[29]), (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with mQ2
        vector<double> MQ2initGUTBCs = GUT_boundary_conditions;
        vector<double> MQ2windows = DSN_specific_windows(MQ2initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 28);
        DSN_soft_denom = abs(copysign(sqrt(abs(MQ2windows[1])), MQ2windows[1])  - copysign(sqrt(abs(MQ2windows[0])), MQ2windows[0]));
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(MQ2windows[1])), MQ2windows[1]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(MQ2windows[0])), MQ2windows[0]), (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(MQ2windows[3])), MQ2windows[3]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(MQ2windows[2])), MQ2windows[2]), (2.0 * nF) + (1.0 * nD) - 1.0);
        DSN_soft_denom = abs(copysign(sqrt(abs(MQ2windows[3])), MQ2windows[3]) - copysign(sqrt(abs(MQ2windows[2])), MQ2windows[2]));
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(copysign(sqrt(abs(boost::math::float_next(GUT_boundary_conditions[28]))), boost::math::float_next(GUT_boundary_conditions[28])), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(copysign(sqrt(abs(boost::math::float_prior(GUT_boundary_conditions[28]))), boost::math::float_prior(GUT_boundary_conditions[28])), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(copysign(sqrt(abs(GUT_boundary_conditions[28])), GUT_boundary_conditions[28])) - boost::math::float_prior(copysign(sqrt(abs(GUT_boundary_conditions[28])), GUT_boundary_conditions[28]))));
            DSN_soft_num = soft_prob_calc(copysign(pow(10.0, 0.5) * sqrt(abs(GUT_boundary_conditions[28])), GUT_boundary_conditions[28]), (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(copysign(pow(10.0, -0.5) * sqrt(abs(GUT_boundary_conditions[28])), GUT_boundary_conditions[28]), (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with mQ1
        vector<double> MQ1initGUTBCs = GUT_boundary_conditions;
        vector<double> MQ1windows = DSN_specific_windows(MQ1initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 27);
        DSN_soft_denom = abs(copysign(sqrt(abs(MQ1windows[1])), MQ1windows[1])  - copysign(sqrt(abs(MQ1windows[0])), MQ1windows[0]));
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(MQ1windows[1])), MQ1windows[1]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(MQ1windows[0])), MQ1windows[0]), (2.0 * nF) + (1.0 * nD) - 1.0);
        // std::cout << DSN_soft_num << endl;
        // std::cout << DSN_soft_denom << endl;
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(MQ1windows[3])), MQ1windows[3]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(MQ1windows[2])), MQ1windows[2]), (2.0 * nF) + (1.0 * nD) - 1.0);
        DSN_soft_denom = abs(copysign(sqrt(abs(MQ1windows[3])), MQ1windows[3]) - copysign(sqrt(abs(MQ1windows[2])), MQ1windows[2]));
        // std::cout << DSN_soft_num / DSN_soft_denom << endl;
        // std::cout << newterm << endl;
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(copysign(sqrt(abs(boost::math::float_next(GUT_boundary_conditions[27]))), boost::math::float_next(GUT_boundary_conditions[27])), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(copysign(sqrt(abs(boost::math::float_prior(GUT_boundary_conditions[27]))), boost::math::float_prior(GUT_boundary_conditions[27])), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(copysign(sqrt(abs(GUT_boundary_conditions[27])), GUT_boundary_conditions[27])) - boost::math::float_prior(copysign(sqrt(abs(GUT_boundary_conditions[27])), GUT_boundary_conditions[27]))));
            DSN_soft_num = soft_prob_calc(copysign(pow(10.0, 0.5) * sqrt(abs(GUT_boundary_conditions[27])), GUT_boundary_conditions[27]), (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(copysign(pow(10.0, -0.5) * sqrt(abs(GUT_boundary_conditions[27])), GUT_boundary_conditions[27]), (2.0 * nF) + (1.0 * nD) - 1.0);
            DSN_soft_denom = abs((pow(10.0, 0.5) * sqrt(abs(GUT_boundary_conditions[27]))) - (pow(10.0, -0.5) * sqrt(abs(GUT_boundary_conditions[27]))));
            // std::cout << DSN_soft_num / DSN_soft_denom << endl;
            // std::cout << newterm << endl;
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with mL3
        vector<double> mL3initGUTBCs = GUT_boundary_conditions;
        vector<double> mL3windows = DSN_specific_windows(mL3initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 32);
        DSN_soft_denom = abs(copysign(sqrt(abs(mL3windows[1])), mL3windows[1])  - copysign(sqrt(abs(mL3windows[0])), mL3windows[0]));
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mL3windows[1])), mL3windows[1]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mL3windows[0])), mL3windows[0]), (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mL3windows[3])), mL3windows[3]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mL3windows[2])), mL3windows[2]), (2.0 * nF) + (1.0 * nD) - 1.0);
        DSN_soft_denom = abs(copysign(sqrt(abs(mL3windows[3])), mL3windows[3]) - copysign(sqrt(abs(mL3windows[2])), mL3windows[2]));
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(copysign(sqrt(abs(boost::math::float_next(GUT_boundary_conditions[32]))), boost::math::float_next(GUT_boundary_conditions[32])), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(copysign(sqrt(abs(boost::math::float_prior(GUT_boundary_conditions[32]))), boost::math::float_prior(GUT_boundary_conditions[32])), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(copysign(sqrt(abs(GUT_boundary_conditions[32])), GUT_boundary_conditions[32])) - boost::math::float_prior(copysign(sqrt(abs(GUT_boundary_conditions[32])), GUT_boundary_conditions[32]))));
            DSN_soft_num = soft_prob_calc(copysign(pow(10.0, 0.5) * sqrt(abs(GUT_boundary_conditions[32])), GUT_boundary_conditions[32]), (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(copysign(pow(10.0, -0.5) * sqrt(abs(GUT_boundary_conditions[32])), GUT_boundary_conditions[32]), (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with mL2
        vector<double> mL2initGUTBCs = GUT_boundary_conditions;
        vector<double> mL2windows = DSN_specific_windows(mL2initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 31);
        DSN_soft_denom = abs(copysign(sqrt(abs(mL2windows[1])), mL2windows[1])  - copysign(sqrt(abs(mL2windows[0])), mL2windows[0]));
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mL2windows[1])), mL2windows[1]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mL2windows[0])), mL2windows[0]), (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mL2windows[3])), mL2windows[3]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mL2windows[2])), mL2windows[2]), (2.0 * nF) + (1.0 * nD) - 1.0);
        DSN_soft_denom = abs(copysign(sqrt(abs(mL2windows[3])), mL2windows[3]) - copysign(sqrt(abs(mL2windows[2])), mL2windows[2]));
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(copysign(sqrt(abs(boost::math::float_next(GUT_boundary_conditions[31]))), boost::math::float_next(GUT_boundary_conditions[31])), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(copysign(sqrt(abs(boost::math::float_prior(GUT_boundary_conditions[31]))), boost::math::float_prior(GUT_boundary_conditions[31])), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(copysign(sqrt(abs(GUT_boundary_conditions[31])), GUT_boundary_conditions[31])) - boost::math::float_prior(copysign(sqrt(abs(GUT_boundary_conditions[31])), GUT_boundary_conditions[31]))));
            DSN_soft_num = soft_prob_calc(copysign(pow(10.0, 0.5) * sqrt(abs(GUT_boundary_conditions[31])), GUT_boundary_conditions[31]), (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(copysign(pow(10.0, -0.5) * sqrt(abs(GUT_boundary_conditions[31])), GUT_boundary_conditions[31]), (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with mL1
        vector<double> mL1initGUTBCs = GUT_boundary_conditions;
        vector<double> mL1windows = DSN_specific_windows(mL1initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 30);
        DSN_soft_denom = abs(copysign(sqrt(abs(mL1windows[1])), mL1windows[1])  - copysign(sqrt(abs(mL1windows[0])), mL1windows[0]));
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mL1windows[1])), mL1windows[1]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mL1windows[0])), mL1windows[0]), (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mL1windows[3])), mL1windows[3]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mL1windows[2])), mL1windows[2]), (2.0 * nF) + (1.0 * nD) - 1.0);
        DSN_soft_denom = abs(copysign(sqrt(abs(mL1windows[3])), mL1windows[3]) - copysign(sqrt(abs(mL1windows[2])), mL1windows[2]));
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(copysign(sqrt(abs(boost::math::float_next(GUT_boundary_conditions[30]))), boost::math::float_next(GUT_boundary_conditions[30])), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(copysign(sqrt(abs(boost::math::float_prior(GUT_boundary_conditions[30]))), boost::math::float_prior(GUT_boundary_conditions[30])), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(copysign(sqrt(abs(GUT_boundary_conditions[30])), GUT_boundary_conditions[30])) - boost::math::float_prior(copysign(sqrt(abs(GUT_boundary_conditions[30])), GUT_boundary_conditions[30]))));
            DSN_soft_num = soft_prob_calc(copysign(pow(10.0, 0.5) * sqrt(abs(GUT_boundary_conditions[30])), GUT_boundary_conditions[30]), (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(copysign(pow(10.0, -0.5) * sqrt(abs(GUT_boundary_conditions[30])), GUT_boundary_conditions[30]), (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with mU3
        vector<double> mU3initGUTBCs = GUT_boundary_conditions;
        vector<double> mU3windows = DSN_specific_windows(mU3initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 35);
        DSN_soft_denom = abs(copysign(sqrt(abs(mU3windows[1])), mU3windows[1])  - copysign(sqrt(abs(mU3windows[0])), mU3windows[0]));
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mU3windows[1])), mU3windows[1]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mU3windows[0])), mU3windows[0]), (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mU3windows[3])), mU3windows[3]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mU3windows[2])), mU3windows[2]), (2.0 * nF) + (1.0 * nD) - 1.0);
        DSN_soft_denom = abs(copysign(sqrt(abs(mU3windows[3])), mU3windows[3]) - copysign(sqrt(abs(mU3windows[2])), mU3windows[2]));
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(copysign(sqrt(abs(boost::math::float_next(GUT_boundary_conditions[35]))), boost::math::float_next(GUT_boundary_conditions[35])), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(copysign(sqrt(abs(boost::math::float_prior(GUT_boundary_conditions[35]))), boost::math::float_prior(GUT_boundary_conditions[35])), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(copysign(sqrt(abs(GUT_boundary_conditions[35])), GUT_boundary_conditions[35])) - boost::math::float_prior(copysign(sqrt(abs(GUT_boundary_conditions[35])), GUT_boundary_conditions[35]))));
            DSN_soft_num = soft_prob_calc(copysign(pow(10.0, 0.5) * sqrt(abs(GUT_boundary_conditions[35])), GUT_boundary_conditions[35]), (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(copysign(pow(10.0, -0.5) * sqrt(abs(GUT_boundary_conditions[35])), GUT_boundary_conditions[35]), (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with mU2
        vector<double> mU2initGUTBCs = GUT_boundary_conditions;
        vector<double> mU2windows = DSN_specific_windows(mU2initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 34);
        DSN_soft_denom = abs(copysign(sqrt(abs(mU2windows[1])), mU2windows[1])  - copysign(sqrt(abs(mU2windows[0])), mU2windows[0]));
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mU2windows[1])), mU2windows[1]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mU2windows[0])), mU2windows[0]), (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mU2windows[3])), mU2windows[3]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mU2windows[2])), mU2windows[2]), (2.0 * nF) + (1.0 * nD) - 1.0);
        DSN_soft_denom = abs(copysign(sqrt(abs(mU2windows[3])), mU2windows[3]) - copysign(sqrt(abs(mU2windows[2])), mU2windows[2]));
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(copysign(sqrt(abs(boost::math::float_next(GUT_boundary_conditions[34]))), boost::math::float_next(GUT_boundary_conditions[34])), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(copysign(sqrt(abs(boost::math::float_prior(GUT_boundary_conditions[34]))), boost::math::float_prior(GUT_boundary_conditions[34])), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(copysign(sqrt(abs(GUT_boundary_conditions[34])), GUT_boundary_conditions[34])) - boost::math::float_prior(copysign(sqrt(abs(GUT_boundary_conditions[34])), GUT_boundary_conditions[34]))));
            DSN_soft_num = soft_prob_calc(copysign(pow(10.0, 0.5) * sqrt(abs(GUT_boundary_conditions[34])), GUT_boundary_conditions[34]), (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(copysign(pow(10.0, -0.5) * sqrt(abs(GUT_boundary_conditions[34])), GUT_boundary_conditions[34]), (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with mU1
        vector<double> mU1initGUTBCs = GUT_boundary_conditions;
        vector<double> mU1windows = DSN_specific_windows(mU1initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 33);
        DSN_soft_denom = abs(copysign(sqrt(abs(mU1windows[1])), mU1windows[1])  - copysign(sqrt(abs(mU1windows[0])), mU1windows[0]));
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mU1windows[1])), mU1windows[1]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mU1windows[0])), mU1windows[0]), (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mU1windows[3])), mU1windows[3]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mU1windows[2])), mU1windows[2]), (2.0 * nF) + (1.0 * nD) - 1.0);
        DSN_soft_denom = abs(copysign(sqrt(abs(mU1windows[3])), mU1windows[3]) - copysign(sqrt(abs(mU1windows[2])), mU1windows[2]));
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(copysign(sqrt(abs(boost::math::float_next(GUT_boundary_conditions[33]))), boost::math::float_next(GUT_boundary_conditions[33])), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(copysign(sqrt(abs(boost::math::float_prior(GUT_boundary_conditions[33]))), boost::math::float_prior(GUT_boundary_conditions[33])), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(copysign(sqrt(abs(GUT_boundary_conditions[33])), GUT_boundary_conditions[33])) - boost::math::float_prior(copysign(sqrt(abs(GUT_boundary_conditions[33])), GUT_boundary_conditions[33]))));
            DSN_soft_num = soft_prob_calc(copysign(pow(10.0, 0.5) * sqrt(abs(GUT_boundary_conditions[33])), GUT_boundary_conditions[33]), (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(copysign(pow(10.0, -0.5) * sqrt(abs(GUT_boundary_conditions[33])), GUT_boundary_conditions[33]), (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with mD3
        vector<double> mD3initGUTBCs = GUT_boundary_conditions;
        vector<double> mD3windows = DSN_specific_windows(mD3initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 38);
        DSN_soft_denom = abs(copysign(sqrt(abs(mD3windows[1])), mD3windows[1])  - copysign(sqrt(abs(mD3windows[0])), mD3windows[0]));
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mD3windows[1])), mD3windows[1]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mD3windows[0])), mD3windows[0]), (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mD3windows[3])), mD3windows[3]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mD3windows[2])), mD3windows[2]), (2.0 * nF) + (1.0 * nD) - 1.0);
        DSN_soft_denom = abs(copysign(sqrt(abs(mD3windows[3])), mD3windows[3]) - copysign(sqrt(abs(mD3windows[2])), mD3windows[2]));
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(copysign(sqrt(abs(boost::math::float_next(GUT_boundary_conditions[38]))), boost::math::float_next(GUT_boundary_conditions[38])), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(copysign(sqrt(abs(boost::math::float_prior(GUT_boundary_conditions[38]))), boost::math::float_prior(GUT_boundary_conditions[38])), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(copysign(sqrt(abs(GUT_boundary_conditions[38])), GUT_boundary_conditions[38])) - boost::math::float_prior(copysign(sqrt(abs(GUT_boundary_conditions[38])), GUT_boundary_conditions[38]))));
            DSN_soft_num = soft_prob_calc(copysign(pow(10.0, 0.5) * sqrt(abs(GUT_boundary_conditions[38])), GUT_boundary_conditions[38]), (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(copysign(pow(10.0, -0.5) * sqrt(abs(GUT_boundary_conditions[38])), GUT_boundary_conditions[38]), (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with mD2
        vector<double> mD2initGUTBCs = GUT_boundary_conditions;
        vector<double> mD2windows = DSN_specific_windows(mD2initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 37);
        DSN_soft_denom = abs(copysign(sqrt(abs(mD2windows[1])), mD2windows[1])  - copysign(sqrt(abs(mD2windows[0])), mD2windows[0]));
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mD2windows[1])), mD2windows[1]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mD2windows[0])), mD2windows[0]), (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mD2windows[3])), mD2windows[3]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mD2windows[2])), mD2windows[2]), (2.0 * nF) + (1.0 * nD) - 1.0);
        DSN_soft_denom = abs(copysign(sqrt(abs(mD2windows[3])), mD2windows[3]) - copysign(sqrt(abs(mD2windows[2])), mD2windows[2]));
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(copysign(sqrt(abs(boost::math::float_next(GUT_boundary_conditions[37]))), boost::math::float_next(GUT_boundary_conditions[37])), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(copysign(sqrt(abs(boost::math::float_prior(GUT_boundary_conditions[37]))), boost::math::float_prior(GUT_boundary_conditions[37])), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(copysign(sqrt(abs(GUT_boundary_conditions[37])), GUT_boundary_conditions[37])) - boost::math::float_prior(copysign(sqrt(abs(GUT_boundary_conditions[37])), GUT_boundary_conditions[37]))));
            DSN_soft_num = soft_prob_calc(copysign(pow(10.0, 0.5) * sqrt(abs(GUT_boundary_conditions[37])), GUT_boundary_conditions[37]), (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(copysign(pow(10.0, -0.5) * sqrt(abs(GUT_boundary_conditions[37])), GUT_boundary_conditions[37]), (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with mD1
        vector<double> mD1initGUTBCs = GUT_boundary_conditions;
        vector<double> mD1windows = DSN_specific_windows(mD1initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 36);
        DSN_soft_denom = abs(copysign(sqrt(abs(mD1windows[1])), mD1windows[1])  - copysign(sqrt(abs(mD1windows[0])), mD1windows[0]));
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mD1windows[1])), mD1windows[1]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mD1windows[0])), mD1windows[0]), (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mD1windows[3])), mD1windows[3]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mD1windows[2])), mD1windows[2]), (2.0 * nF) + (1.0 * nD) - 1.0);
        DSN_soft_denom = abs(copysign(sqrt(abs(mD1windows[3])), mD1windows[3]) - copysign(sqrt(abs(mD1windows[2])), mD1windows[2]));
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(copysign(sqrt(abs(boost::math::float_next(GUT_boundary_conditions[36]))), boost::math::float_next(GUT_boundary_conditions[36])), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(copysign(sqrt(abs(boost::math::float_prior(GUT_boundary_conditions[36]))), boost::math::float_prior(GUT_boundary_conditions[36])), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(copysign(sqrt(abs(GUT_boundary_conditions[36])), GUT_boundary_conditions[36])) - boost::math::float_prior(copysign(sqrt(abs(GUT_boundary_conditions[36])), GUT_boundary_conditions[36]))));
            DSN_soft_num = soft_prob_calc(copysign(pow(10.0, 0.5) * sqrt(abs(GUT_boundary_conditions[36])), GUT_boundary_conditions[36]), (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(copysign(pow(10.0, -0.5) * sqrt(abs(GUT_boundary_conditions[36])), GUT_boundary_conditions[36]), (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with mE3
        vector<double> mE3initGUTBCs = GUT_boundary_conditions;
        vector<double> mE3windows = DSN_specific_windows(mE3initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 41);
        DSN_soft_denom = abs(copysign(sqrt(abs(mE3windows[1])), mE3windows[1])  - copysign(sqrt(abs(mE3windows[0])), mE3windows[0]));
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mE3windows[1])), mE3windows[1]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mE3windows[0])), mE3windows[0]), (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mE3windows[3])), mE3windows[3]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mE3windows[2])), mE3windows[2]), (2.0 * nF) + (1.0 * nD) - 1.0);
        DSN_soft_denom = abs(copysign(sqrt(abs(mE3windows[3])), mE3windows[3]) - copysign(sqrt(abs(mE3windows[2])), mE3windows[2]));
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(copysign(sqrt(abs(boost::math::float_next(GUT_boundary_conditions[41]))), boost::math::float_next(GUT_boundary_conditions[41])), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(copysign(sqrt(abs(boost::math::float_prior(GUT_boundary_conditions[41]))), boost::math::float_prior(GUT_boundary_conditions[41])), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(copysign(sqrt(abs(GUT_boundary_conditions[41])), GUT_boundary_conditions[41])) - boost::math::float_prior(copysign(sqrt(abs(GUT_boundary_conditions[41])), GUT_boundary_conditions[41]))));
            DSN_soft_num = soft_prob_calc(copysign(pow(10.0, 0.5) * sqrt(abs(GUT_boundary_conditions[41])), GUT_boundary_conditions[41]), (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(copysign(pow(10.0, -0.5) * sqrt(abs(GUT_boundary_conditions[41])), GUT_boundary_conditions[41]), (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with mE2
        vector<double> mE2initGUTBCs = GUT_boundary_conditions;
        vector<double> mE2windows = DSN_specific_windows(mE2initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 40);
        DSN_soft_denom = abs(copysign(sqrt(abs(mE2windows[1])), mE2windows[1])  - copysign(sqrt(abs(mE2windows[0])), mE2windows[0]));
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mE2windows[1])), mE2windows[1]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mE2windows[0])), mE2windows[0]), (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mE2windows[3])), mE2windows[3]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mE2windows[2])), mE2windows[2]), (2.0 * nF) + (1.0 * nD) - 1.0);
        DSN_soft_denom = abs(copysign(sqrt(abs(mE2windows[3])), mE2windows[3]) - copysign(sqrt(abs(mE2windows[2])), mE2windows[2]));
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(copysign(sqrt(abs(boost::math::float_next(GUT_boundary_conditions[40]))), boost::math::float_next(GUT_boundary_conditions[40])), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(copysign(sqrt(abs(boost::math::float_prior(GUT_boundary_conditions[40]))), boost::math::float_prior(GUT_boundary_conditions[40])), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(copysign(sqrt(abs(GUT_boundary_conditions[40])), GUT_boundary_conditions[40])) - boost::math::float_prior(copysign(sqrt(abs(GUT_boundary_conditions[40])), GUT_boundary_conditions[40]))));
            DSN_soft_num = soft_prob_calc(copysign(pow(10.0, 0.5) * sqrt(abs(GUT_boundary_conditions[40])), GUT_boundary_conditions[40]), (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(copysign(pow(10.0, -0.5) * sqrt(abs(GUT_boundary_conditions[40])), GUT_boundary_conditions[40]), (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with mE1
        vector<double> mE1initGUTBCs = GUT_boundary_conditions;
        vector<double> mE1windows = DSN_specific_windows(mE1initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 39);
        DSN_soft_denom = abs(copysign(sqrt(abs(mE1windows[1])), mE1windows[1])  - copysign(sqrt(abs(mE1windows[0])), mE1windows[0]));
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mE1windows[1])), mE1windows[1]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mE1windows[0])), mE1windows[0]), (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mE1windows[3])), mE1windows[3]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mE1windows[2])), mE1windows[2]), (2.0 * nF) + (1.0 * nD) - 1.0);
        DSN_soft_denom = abs(copysign(sqrt(abs(mE1windows[3])), mE1windows[3]) - copysign(sqrt(abs(mE1windows[2])), mE1windows[2]));
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(copysign(sqrt(abs(boost::math::float_next(GUT_boundary_conditions[39]))), boost::math::float_next(GUT_boundary_conditions[39])), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(copysign(sqrt(abs(boost::math::float_prior(GUT_boundary_conditions[39]))), boost::math::float_prior(GUT_boundary_conditions[39])), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(copysign(sqrt(abs(GUT_boundary_conditions[39])), GUT_boundary_conditions[39])) - boost::math::float_prior(copysign(sqrt(abs(GUT_boundary_conditions[39])), GUT_boundary_conditions[39]))));
            DSN_soft_num = soft_prob_calc(copysign(pow(10.0, 0.5) * sqrt(abs(GUT_boundary_conditions[39])), GUT_boundary_conditions[39]), (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(copysign(pow(10.0, -0.5) * sqrt(abs(GUT_boundary_conditions[39])), GUT_boundary_conditions[39]), (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with at
        vector<double> atinitGUTBCs = GUT_boundary_conditions;
        vector<double> atwindows = DSN_specific_windows(atinitGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 16);
        DSN_soft_denom = abs(atwindows[1] - atwindows[0]);
        DSN_soft_num = soft_prob_calc(atwindows[1], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(atwindows[0], (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_denom = abs(atwindows[3] - atwindows[2]);
        DSN_soft_num = soft_prob_calc(atwindows[3], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(atwindows[2], (2.0 * nF) + (1.0 * nD) - 1.0);
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(boost::math::float_next(GUT_boundary_conditions[16]), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(boost::math::float_prior(GUT_boundary_conditions[16]), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(GUT_boundary_conditions[16]) - boost::math::float_prior(GUT_boundary_conditions[16])));
            DSN_soft_num = soft_prob_calc(pow(10.0, 0.5) * GUT_boundary_conditions[16], (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(pow(10.0, -0.5) * GUT_boundary_conditions[16], (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with ac
        vector<double> acinitGUTBCs = GUT_boundary_conditions;
        vector<double> acwindows = DSN_specific_windows(acinitGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 17);
        DSN_soft_denom = abs(acwindows[1] - acwindows[0]);
        DSN_soft_num = soft_prob_calc(acwindows[1], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(acwindows[0], (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_denom = abs(acwindows[3] - acwindows[2]);
        DSN_soft_num = soft_prob_calc(acwindows[3], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(acwindows[2], (2.0 * nF) + (1.0 * nD) - 1.0);
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(boost::math::float_next(GUT_boundary_conditions[17]), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(boost::math::float_prior(GUT_boundary_conditions[17]), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(GUT_boundary_conditions[17]) - boost::math::float_prior(GUT_boundary_conditions[17])));
            DSN_soft_num = soft_prob_calc(pow(10.0, 0.5) * GUT_boundary_conditions[17], (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(pow(10.0, -0.5) * GUT_boundary_conditions[17], (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with au    
        vector<double> auinitGUTBCs = GUT_boundary_conditions;
        vector<double> auwindows = DSN_specific_windows(auinitGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 18);
        DSN_soft_denom = abs(auwindows[1] - auwindows[0]);
        DSN_soft_num = soft_prob_calc(auwindows[1], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(auwindows[0], (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_denom = abs(auwindows[3] - auwindows[2]);
        DSN_soft_num = soft_prob_calc(auwindows[3], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(auwindows[2], (2.0 * nF) + (1.0 * nD) - 1.0);
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(boost::math::float_next(GUT_boundary_conditions[18]), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(boost::math::float_prior(GUT_boundary_conditions[18]), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(GUT_boundary_conditions[18]) - boost::math::float_prior(GUT_boundary_conditions[18])));
            DSN_soft_num = soft_prob_calc(pow(10.0, 0.5) * GUT_boundary_conditions[18], (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(pow(10.0, -0.5) * GUT_boundary_conditions[18], (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with ab
        vector<double> abinitGUTBCs = GUT_boundary_conditions;
        vector<double> abwindows = DSN_specific_windows(abinitGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 19);
        DSN_soft_denom = abs(abwindows[1] - abwindows[0]);
        DSN_soft_num = soft_prob_calc(abwindows[1], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(abwindows[0], (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_denom = abs(abwindows[3] - abwindows[2]);
        DSN_soft_num = soft_prob_calc(abwindows[3], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(abwindows[2], (2.0 * nF) + (1.0 * nD) - 1.0);
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(boost::math::float_next(GUT_boundary_conditions[19]), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(boost::math::float_prior(GUT_boundary_conditions[19]), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(GUT_boundary_conditions[19]) - boost::math::float_prior(GUT_boundary_conditions[19])));
            DSN_soft_num = soft_prob_calc(pow(10.0, 0.5) * GUT_boundary_conditions[19], (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(pow(10.0, -0.5) * GUT_boundary_conditions[19], (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with as
        vector<double> asinitGUTBCs = GUT_boundary_conditions;
        vector<double> aswindows = DSN_specific_windows(asinitGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 20);
        DSN_soft_denom = abs(aswindows[1] - aswindows[0]);
        DSN_soft_num = soft_prob_calc(aswindows[1], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(aswindows[0], (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_denom = abs(aswindows[3] - aswindows[2]);
        DSN_soft_num = soft_prob_calc(aswindows[3], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(aswindows[2], (2.0 * nF) + (1.0 * nD) - 1.0);
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(boost::math::float_next(GUT_boundary_conditions[20]), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(boost::math::float_prior(GUT_boundary_conditions[20]), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(GUT_boundary_conditions[20]) - boost::math::float_prior(GUT_boundary_conditions[20])));
            DSN_soft_num = soft_prob_calc(pow(10.0, 0.5) * GUT_boundary_conditions[20], (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(pow(10.0, -0.5) * GUT_boundary_conditions[20], (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with ad    
        vector<double> adinitGUTBCs = GUT_boundary_conditions;
        vector<double> adwindows = DSN_specific_windows(adinitGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 21);
        DSN_soft_denom = abs(adwindows[1] - adwindows[0]);
        DSN_soft_num = soft_prob_calc(adwindows[1], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(adwindows[0], (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_denom = abs(adwindows[3] - adwindows[2]);
        DSN_soft_num = soft_prob_calc(adwindows[3], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(adwindows[2], (2.0 * nF) + (1.0 * nD) - 1.0);
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(boost::math::float_next(GUT_boundary_conditions[21]), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(boost::math::float_prior(GUT_boundary_conditions[21]), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(GUT_boundary_conditions[21]) - boost::math::float_prior(GUT_boundary_conditions[21])));
            DSN_soft_num = soft_prob_calc(pow(10.0, 0.5) * GUT_boundary_conditions[21], (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(pow(10.0, -0.5) * GUT_boundary_conditions[21], (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with atau
        vector<double> atauinitGUTBCs = GUT_boundary_conditions;
        vector<double> atauwindows = DSN_specific_windows(atauinitGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 22);
        DSN_soft_denom = abs(atauwindows[1] - atauwindows[0]);
        DSN_soft_num = soft_prob_calc(atauwindows[1], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(atauwindows[0], (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_denom = abs(atauwindows[3] - atauwindows[2]);
        DSN_soft_num = soft_prob_calc(atauwindows[3], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(atauwindows[2], (2.0 * nF) + (1.0 * nD) - 1.0);
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(boost::math::float_next(GUT_boundary_conditions[22]), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(boost::math::float_prior(GUT_boundary_conditions[22]), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(GUT_boundary_conditions[22]) - boost::math::float_prior(GUT_boundary_conditions[22])));
            DSN_soft_num = soft_prob_calc(pow(10.0, 0.5) * GUT_boundary_conditions[22], (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(pow(10.0, -0.5) * GUT_boundary_conditions[22], (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;
        
        // Now do same thing with amu
        vector<double> amuinitGUTBCs = GUT_boundary_conditions;
        vector<double> amuwindows = DSN_specific_windows(amuinitGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 23);
        DSN_soft_denom = abs(amuwindows[1] - amuwindows[0]);
        DSN_soft_num = soft_prob_calc(amuwindows[1], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(amuwindows[0], (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_denom = abs(amuwindows[3] - amuwindows[2]);
        DSN_soft_num = soft_prob_calc(amuwindows[3], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(amuwindows[2], (2.0 * nF) + (1.0 * nD) - 1.0);
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(boost::math::float_next(GUT_boundary_conditions[23]), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(boost::math::float_prior(GUT_boundary_conditions[23]), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(GUT_boundary_conditions[23]) - boost::math::float_prior(GUT_boundary_conditions[23])));
            DSN_soft_num = soft_prob_calc(pow(10.0, 0.5) * GUT_boundary_conditions[23], (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(pow(10.0, -0.5) * GUT_boundary_conditions[23], (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with ae    
        vector<double> aeinitGUTBCs = GUT_boundary_conditions;
        vector<double> aewindows = DSN_specific_windows(aeinitGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 24);
        DSN_soft_denom = abs(aewindows[1] - aewindows[0]);
        DSN_soft_num = soft_prob_calc(aewindows[1], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(aewindows[0], (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_denom = abs(aewindows[3] - aewindows[2]);
        DSN_soft_num = soft_prob_calc(aewindows[3], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(aewindows[2], (2.0 * nF) + (1.0 * nD) - 1.0);
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(boost::math::float_next(GUT_boundary_conditions[24]), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(boost::math::float_prior(GUT_boundary_conditions[24]), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(GUT_boundary_conditions[24]) - boost::math::float_prior(GUT_boundary_conditions[24])));
            DSN_soft_num = soft_prob_calc(pow(10.0, 0.5) * GUT_boundary_conditions[24], (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(pow(10.0, -0.5) * GUT_boundary_conditions[24], (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with B = b/mu;
        vector<double> BinitGUTBCs = GUT_boundary_conditions;
        vector<double> Bwindows = DSN_B_windows(BinitGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT);
        DSN_soft_denom = abs(Bwindows[1] - Bwindows[0]);
        DSN_soft_num = soft_prob_calc(Bwindows[1], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(Bwindows[0], (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_denom = abs(Bwindows[3] - Bwindows[2]);
        DSN_soft_num = soft_prob_calc(Bwindows[3], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(Bwindows[2], (2.0 * nF) + (1.0 * nD) - 1.0);
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(boost::math::float_next(GUT_boundary_conditions[42] / GUT_boundary_conditions[6]), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(boost::math::float_prior(GUT_boundary_conditions[42] / GUT_boundary_conditions[6]), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(GUT_boundary_conditions[42] / GUT_boundary_conditions[6]) - boost::math::float_prior(GUT_boundary_conditions[42] / GUT_boundary_conditions[6])));
            DSN_soft_num = soft_prob_calc(pow(10.0, 0.5) * GUT_boundary_conditions[42] / GUT_boundary_conditions[6], (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(pow(10.0, -0.5) * GUT_boundary_conditions[42] / GUT_boundary_conditions[6], (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;
    } else {
        // Compute mu windows around original point
        vector<double> muinitGUTBCs = GUT_boundary_conditions;
        vector<double> muwindows = DSN_mu_windows(muinitGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT);
        DSN_higgsino = abs(log10(abs(muwindows[1] / muwindows[0])));
        DSN_higgsino /= abs(muwindows[1] - muwindows[0]);
        newterm = DSN_higgsino;
        // Total normalization
        DSN_higgsino = abs(log10(abs(muwindows[3] / muwindows[2])));
        DSN_higgsino /= abs(muwindows[3] - muwindows[2]);
        if ((abs(DSN_higgsino - newterm) < numeric_limits<double>::epsilon()) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan(DSN_higgsino)) || (DSN_higgsino == 0.0) || isinf(DSN_higgsino)) {
            newterm = abs(log10(1.0 + (numeric_limits<double>::epsilon() * abs(GUT_boundary_conditions[6]))))\
                / abs(numeric_limits<double>::epsilon() * abs(GUT_boundary_conditions[6]));
            DSN_higgsino = 1.0 / abs(((pow(10.0, 0.5) - pow(10.0, -0.5))) * abs(GUT_boundary_conditions[6]));
        }
        DSN += abs(log10(abs(DSN_higgsino)) - log10(abs(newterm)));
        std::cout << "DSN after higgsino = " << DSN << endl;
    }    

    return DSN;
}
