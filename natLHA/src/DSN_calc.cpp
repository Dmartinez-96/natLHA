#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <string>
#include <thread>
#include <chrono>
#include <stdexcept>
#include <boost/math/special_functions/next.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include "DSN_calc.hpp"
#include "MSSM_RGE_solver.hpp"
#include "MSSM_RGE_solver_with_stopfinder.hpp"
#include "mZ_numsolver.hpp"
#include "radcorr_calc.hpp"
#include "tree_mass_calc.hpp"
#include "EWSB_loop.hpp"
#include "other_derivs.hpp"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace boost::multiprecision;
typedef number<mpfr_float_backend<50>> high_prec_float;

bool DSNabsValCompare(const DSNLabeledValue& a, const DSNLabeledValue& b) {
    return abs(a.value) < abs(b.value);
}

std::vector<DSNLabeledValue> sortAndReturnDSN(const std::vector<DSNLabeledValue>& concatenatedList) {
    std::vector<DSNLabeledValue> sortedList = concatenatedList;
    std::sort(sortedList.begin(), sortedList.end(), DSNabsValCompare);
    std::reverse(sortedList.begin(), sortedList.end());
    return sortedList;
}

high_prec_float signum(high_prec_float x) {
    if (x < 0) {
        return -1.0;
    } else if (x > 0) {
        return 1.0;
    } else {
        return 0.0;
    }
}

high_prec_float signed_square(high_prec_float x2, high_prec_float h) {
    high_prec_float signedsqrt = copysign(sqrt(abs(x2)), x2);
    high_prec_float shiftedvalue = signedsqrt + h;
    high_prec_float signedsquare;
    if ((signum(signedsqrt) != signum(shiftedvalue))) {
        signedsquare = (-1.0) * copysign(shiftedvalue * shiftedvalue, x2);
    } else {
        signedsquare = copysign(shiftedvalue * shiftedvalue, x2);
    }
    return signedsquare;
}

high_prec_float soft_prob_calc(high_prec_float x, int nPower) {
    return ((0.5 * x / (static_cast<high_prec_float>(nPower) + 1.0))
            * ((signum(x) * (pow(x, nPower) - pow((-1.0) * x, nPower))) + pow(x, nPower) + pow((-1.0) * x, nPower)));
}

bool EWSB_Check(vector<high_prec_float>& weak_boundary_conditions, vector<high_prec_float>& radiat_correcs) {
    bool checkifEWSB = true;

    if (abs(2.0 * weak_boundary_conditions[42]) > abs((2.0 * pow(weak_boundary_conditions[6], 2.0)) + weak_boundary_conditions[25] + radiat_correcs[0] + weak_boundary_conditions[26] + radiat_correcs[1])) {
        checkifEWSB = false;
    }
    return checkifEWSB;
}

bool CCB_Check(vector<high_prec_float>& weak_boundary_conditions) {
    bool checkifNoCCB = true;
    for (int i = 3; i < 6; ++i) {
        if (weak_boundary_conditions[i] < 0) {
            checkifNoCCB = false;
        }
    }
    for (int i = 27; i < 42; ++i) {
        if (weak_boundary_conditions[i] < 0) {
            checkifNoCCB = false;
        }
    }
    return checkifNoCCB;
}

std::vector<high_prec_float> modifyNewtonInitGuesses(high_prec_float dm, high_prec_float fmpt, high_prec_float gmpt, high_prec_float fmp, high_prec_float gmp, high_prec_float fmt, high_prec_float gmt, high_prec_float fpt, high_prec_float gpt, high_prec_float ft, high_prec_float fp, high_prec_float gt, high_prec_float gp, high_prec_float fm, high_prec_float gm) {
    high_prec_float adp, bdp, cdp, adt, bdt;
    adp = (dm * dm * ((fmpt * gmp) - (fmp * gmpt))) + (fpt * gp) - (fp * gpt) + (dm * ((fpt * gmp) - (fp * gmpt) + (fmpt * gp) - (fmp * gpt)));
    bdp = (dm * dm * ((fmpt * gm) + (fmt * gmp) - (fm * gmpt) - (fmp * gmt))) + (ft * gp) - (fp * gt) + (dm * ((fpt * gm) + (ft * gmp) - (fp * gmt) + (fmt * gp) - (fm * gpt) - (fmp * gt)));
    cdp = dm * ((dm * fmt * gm) + (ft * gm) - (dm * fm * gmt) - (fm * gt));
    adt = (dm * dm * ((fmt * gmpt) - (fmpt * gmt))) + (ft * gpt) - (fpt * gt) + (dm * ((ft * gmpt) - (fpt * gmt) + (fmt * gpt) - (fmpt * gt)));
    bdt = (dm * dm * ((fmt * gmp) - (fmpt * gm) + (fm * gmpt) - (fmp * gmt))) + (ft * gp) - (fp * gt) + (dm * ((ft * gmp) - (fpt * gm) - (fp * gmt) + (fmt * gp) + (fm * gpt) - (fmp * gt)));
    high_prec_float retDP_plus, retDT_plus, retDP_minus, retDT_minus;
    retDP_plus = (high_prec_float(1.0) / (high_prec_float(2.0) * adp)) * ((high_prec_float(-1.0) * bdp) + sqrt(abs((bdp * bdp) - (high_prec_float(4.0) * adp * cdp))));
    retDT_plus = (high_prec_float(1.0) / (high_prec_float(2.0) * adt)) * ((high_prec_float(-1.0) * bdt) + sqrt(abs((bdp * bdp) - (high_prec_float(4.0) * adp * cdp))));
    retDP_minus = (high_prec_float(1.0) / (high_prec_float(2.0) * adp)) * ((high_prec_float(-1.0) * bdp) - sqrt(abs((bdp * bdp) - (high_prec_float(4.0) * adp * cdp))));
    retDT_minus = (high_prec_float(1.0) / (high_prec_float(2.0) * adt)) * ((high_prec_float(-1.0) * bdt) - sqrt(abs((bdp * bdp) - (high_prec_float(4.0) * adp * cdp))));
    std::vector<high_prec_float> retvec = {retDP_plus, retDT_plus, retDP_minus, retDT_minus};
    return retvec;
}

high_prec_float first_derivative_calc(high_prec_float hStep, high_prec_float pm2h, high_prec_float pmh, high_prec_float pph, high_prec_float pp2h) {
    return ((pm2h / 12.0) - (2.0 * pmh / 3.0) + (2.0 * pph / 3.0) - (pp2h / 12.0)) / hStep;
}

high_prec_float second_derivative_calc(high_prec_float hStep, high_prec_float pStart, high_prec_float pm2h, high_prec_float pmh, high_prec_float pph, high_prec_float pp2h) {
    return (((-1.0) * pm2h / 12.0) + (4.0 * pmh / 3.0) - (5.0 * pStart / 2.0) + (4.0 * pph / 3.0) - (pp2h / 12.0)) / (hStep * hStep);
}

high_prec_float mixed_second_derivative_calc(high_prec_float pStep, high_prec_float tStep, high_prec_float fm2pm2t, high_prec_float fm2pmt, high_prec_float fm2ppt, high_prec_float fm2pp2t,
                                    high_prec_float fmpm2t, high_prec_float fmpmt, high_prec_float fmppt, high_prec_float fmpp2t, high_prec_float fppm2t, high_prec_float fppmt, high_prec_float fpppt,
                                    high_prec_float fppp2t, high_prec_float fp2pm2t, high_prec_float fp2pmt, high_prec_float fp2ppt, high_prec_float fp2pp2t) {
    return ((high_prec_float(1.0) / (32.0 * pStep * tStep))
            * ((4.0 * fpppt) - fp2ppt - fppp2t + (2.0 * fp2pp2t) - (4.0 * fmppt) + fm2ppt + fmpp2t - (2.0 * fm2pp2t)
               - (4.0 * fppmt) + fp2pmt + fppm2t - (2.0 * fp2pm2t) + (4.0 * fmpmt) - fm2pmt - fmpm2t + (2.0 * fm2pm2t)));
}

high_prec_float calculate_approx_mZ2(vector<high_prec_float> weak_solutions, high_prec_float explogQSUSY, high_prec_float mZ2Value) {
    vector<high_prec_float> calculateRadCorrs = radcorr_calc(weak_solutions, explogQSUSY, mZ2Value);
    return high_prec_float(1.0) - ((high_prec_float(2.0) / mZ2Value) * ((((weak_solutions[26] + calculateRadCorrs[1] - ((weak_solutions[25] + calculateRadCorrs[0]) * weak_solutions[43] * weak_solutions[43]))) / ((weak_solutions[43] * weak_solutions[43]) - 1.0)) - (weak_solutions[6] * weak_solutions[6])));
}

high_prec_float calculate_approx_tanb(vector<high_prec_float> weak_solutions, high_prec_float explogQSUSY, high_prec_float mZ2Value) {
    vector<high_prec_float> calculateRadCorrs = radcorr_calc(weak_solutions, explogQSUSY, mZ2Value);
    return high_prec_float(1.0) - tan(0.5 * (M_PI - asin(abs(2.0 * weak_solutions[42] / (weak_solutions[25] + weak_solutions[26] + calculateRadCorrs[0] + calculateRadCorrs[1] + (2.0 * weak_solutions[6] * weak_solutions[6])))))) / weak_solutions[43];
}

vector<high_prec_float> single_var_deriv_approxes(vector<high_prec_float>& original_weak_conditions, high_prec_float& fixed_mZ2_val, int idx_to_shift, high_prec_float& logQSUSYval) {
    high_prec_float p_orig, h_p, p_plus, p_minus, p_plusplus, p_minusminus;
    if (idx_to_shift == 42) {
        p_orig = original_weak_conditions[idx_to_shift] / original_weak_conditions[6];
        h_p = min(high_prec_float(0.95), pow(high_prec_float(10.25) * (boost::math::float_next(abs(p_orig)) - abs(p_orig)), high_prec_float(high_prec_float(1.0) / high_prec_float(5.0))));
        p_plus = p_orig + h_p;
        p_minus = p_orig - h_p;
        p_plusplus = p_plus + h_p;
        p_minusminus = p_minus - h_p;
    } else if ((idx_to_shift >= 16) && (idx_to_shift <= 24)) {
        p_orig = original_weak_conditions[idx_to_shift] / original_weak_conditions[idx_to_shift-9];
        h_p = min(high_prec_float(0.95), pow(high_prec_float(10.25) * (boost::math::float_next(abs(p_orig)) - abs(p_orig)), high_prec_float(high_prec_float(1.0) / high_prec_float(5.0))));
        p_plus = (p_orig + h_p) * original_weak_conditions[idx_to_shift-9];
        p_minus = (p_orig - h_p) * original_weak_conditions[idx_to_shift-9];
        p_plusplus = (p_plus + h_p) * original_weak_conditions[idx_to_shift-9];
        p_minusminus = (p_minus - h_p) * original_weak_conditions[idx_to_shift-9];
    }
    else {
        p_orig = original_weak_conditions[idx_to_shift];
        h_p = min(high_prec_float(0.95), pow(high_prec_float(10.25) * (boost::math::float_next(abs(p_orig)) - abs(p_orig)), high_prec_float(high_prec_float(1.0) / high_prec_float(5.0))));
        p_plus = p_orig + h_p;
        p_minus = p_orig - h_p;
        p_plusplus = p_plus + h_p;
        p_minusminus = p_minus - h_p;
    }

    vector<high_prec_float> newmZ2weak_plus = original_weak_conditions;
    vector<high_prec_float> newmZ2weak_plusplus = original_weak_conditions;
    vector<high_prec_float> newtanbweak_plus = original_weak_conditions;
    vector<high_prec_float> newtanbweak_plusplus = original_weak_conditions;
    vector<high_prec_float> newmZ2weak_minus = original_weak_conditions;
    vector<high_prec_float> newmZ2weak_minusminus = original_weak_conditions;
    vector<high_prec_float> newtanbweak_minus = original_weak_conditions;
    vector<high_prec_float> newtanbweak_minusminus = original_weak_conditions;
    
    high_prec_float tanb_orig = original_weak_conditions[43];
    high_prec_float h_tanb = pow(high_prec_float(10.25) * (boost::math::float_next(abs(tanb_orig)) - abs(tanb_orig)), (high_prec_float(1.0) / high_prec_float(5.0)));
    
    newtanbweak_plus[43] = tanb_orig + h_tanb;
    newtanbweak_plusplus[43] = tanb_orig + (2.0 * h_tanb);
    newtanbweak_minus[43] = tanb_orig - h_tanb;
    newtanbweak_minusminus[43] = tanb_orig - (2.0 * h_tanb);
 
    // Adjust Yukawas at Q=mt=173.2 GeV for shifted tanb points
    high_prec_float wk_tanb = original_weak_conditions[43];
    vector<high_prec_float> weaksols_original = original_weak_conditions;
    vector<high_prec_float> weaksolstanb_plus = newtanbweak_plus;
    vector<high_prec_float> weaksolstanb_minus = newtanbweak_minus;
    vector<high_prec_float> weaksolstanb_plusplus = newtanbweak_plusplus;
    vector<high_prec_float> weaksolstanb_minusminus = newtanbweak_minusminus;
    for (int UpYukawaIndex = 7; UpYukawaIndex < 10; ++UpYukawaIndex) {
        weaksolstanb_plus[UpYukawaIndex] *= sin(atan(weaksols_original[43])) / sin(atan(weaksolstanb_plus[43]));
        weaksolstanb_plus[UpYukawaIndex+9] *= weaksolstanb_plus[UpYukawaIndex] / weaksols_original[UpYukawaIndex];
        weaksolstanb_plusplus[UpYukawaIndex] *= sin(atan(weaksols_original[43])) / sin(atan(weaksolstanb_plusplus[43]));
        weaksolstanb_plusplus[UpYukawaIndex+9] *= weaksolstanb_plusplus[UpYukawaIndex] / weaksols_original[UpYukawaIndex];
        weaksolstanb_minus[UpYukawaIndex] *= sin(atan(weaksols_original[43])) / sin(atan(weaksolstanb_minus[43]));
        weaksolstanb_minus[UpYukawaIndex+9] *= weaksolstanb_minus[UpYukawaIndex] / weaksols_original[UpYukawaIndex];
        weaksolstanb_minusminus[UpYukawaIndex] *= sin(atan(weaksols_original[43])) / sin(atan(weaksolstanb_minusminus[43]));
        weaksolstanb_minusminus[UpYukawaIndex+9] *= weaksolstanb_minusminus[UpYukawaIndex] / weaksols_original[UpYukawaIndex];
    }
    for (int DownYukawaIndex = 10; DownYukawaIndex < 16; ++DownYukawaIndex) {
        weaksolstanb_plus[DownYukawaIndex] *= cos(atan(weaksols_original[43])) / cos(atan(weaksolstanb_plus[43]));
        weaksolstanb_plus[DownYukawaIndex+9] *= weaksolstanb_plus[DownYukawaIndex] / weaksols_original[DownYukawaIndex];
        weaksolstanb_plusplus[DownYukawaIndex] *= cos(atan(weaksols_original[43])) / cos(atan(weaksolstanb_plusplus[43]));
        weaksolstanb_plusplus[DownYukawaIndex+9] *= weaksolstanb_plusplus[DownYukawaIndex] / weaksols_original[DownYukawaIndex];
        weaksolstanb_minus[DownYukawaIndex] *= cos(atan(weaksols_original[43])) / cos(atan(weaksolstanb_minus[43]));
        weaksolstanb_minus[DownYukawaIndex+9] *= weaksolstanb_minus[DownYukawaIndex] / weaksols_original[DownYukawaIndex];
        weaksolstanb_minusminus[DownYukawaIndex] *= cos(atan(weaksols_original[43])) / cos(atan(weaksolstanb_minusminus[43]));
        weaksolstanb_minusminus[DownYukawaIndex+9] *= weaksolstanb_minusminus[DownYukawaIndex] / weaksols_original[DownYukawaIndex];
    }
    for (int YukawaIndex = 7; YukawaIndex < 16; ++YukawaIndex) {
        newtanbweak_plus[YukawaIndex] = weaksolstanb_plus[YukawaIndex];
        newtanbweak_plus[YukawaIndex+9] = weaksolstanb_plus[YukawaIndex+9];
        newtanbweak_plusplus[YukawaIndex] = weaksolstanb_plusplus[YukawaIndex];
        newtanbweak_plusplus[YukawaIndex+9] = weaksolstanb_plusplus[YukawaIndex+9];
        newtanbweak_minus[YukawaIndex] = weaksolstanb_minus[YukawaIndex];
        newtanbweak_minus[YukawaIndex+9] = weaksolstanb_minus[YukawaIndex+9];
        newtanbweak_minusminus[YukawaIndex] = weaksolstanb_minusminus[YukawaIndex];
        newtanbweak_minusminus[YukawaIndex+9] = weaksolstanb_minusminus[YukawaIndex+9];
    }
    high_prec_float mZ2_tanb_plus = calculate_approx_mZ2(weaksolstanb_plus, exp(logQSUSYval), fixed_mZ2_val);
    high_prec_float mZ2_tanb_minus = calculate_approx_mZ2(weaksolstanb_minus, exp(logQSUSYval), fixed_mZ2_val);
    high_prec_float mZ2_tanb_plusplus = calculate_approx_mZ2(weaksolstanb_plusplus, exp(logQSUSYval), fixed_mZ2_val);
    high_prec_float mZ2_tanb_minusminus = calculate_approx_mZ2(weaksolstanb_minusminus, exp(logQSUSYval), fixed_mZ2_val);

    if (idx_to_shift == 6) {
        newmZ2weak_plus[42] = original_weak_conditions[42] * p_plus / original_weak_conditions[6];
        newmZ2weak_plusplus[42] = original_weak_conditions[42] * p_plusplus / original_weak_conditions[6];
        newmZ2weak_minus[42] = original_weak_conditions[42] * p_minus / original_weak_conditions[6];
        newmZ2weak_minusminus[42] = original_weak_conditions[42] * p_minusminus / original_weak_conditions[6];
        newmZ2weak_plus[6] = p_plus;
        newmZ2weak_plusplus[6] = p_plusplus;
        newmZ2weak_minus[6] = p_minus;
        newmZ2weak_minusminus[6] = p_minusminus;
    } else if (idx_to_shift == 42) {
        newmZ2weak_plus[42] = original_weak_conditions[6] * p_plus;
        newmZ2weak_plusplus[42] = original_weak_conditions[6] * p_plusplus;
        newmZ2weak_minus[42] = original_weak_conditions[6] * p_minus;
        newmZ2weak_minusminus[42] = original_weak_conditions[6] * p_minusminus;
    } else {
        newmZ2weak_plus[idx_to_shift] = p_plus;
        newmZ2weak_plusplus[idx_to_shift] = p_plusplus;
        newmZ2weak_minus[idx_to_shift] = p_minus;
        newmZ2weak_minusminus[idx_to_shift] = p_minusminus;
    }

    vector<high_prec_float> weaksolsp_plus = newmZ2weak_plus;
    vector<high_prec_float> weaksolsp_plusplus = newmZ2weak_plusplus;
    vector<high_prec_float> weaksolsp_minus = newmZ2weak_minus;
    vector<high_prec_float> weaksolsp_minusminus = newmZ2weak_minusminus;
        
    high_prec_float mZ2_original_pt = calculate_approx_mZ2(original_weak_conditions, exp(logQSUSYval), fixed_mZ2_val);

    high_prec_float mZ2_p_plus = calculate_approx_mZ2(weaksolsp_plus, exp(logQSUSYval), fixed_mZ2_val);
    high_prec_float mZ2_p_plusplus = calculate_approx_mZ2(weaksolsp_plusplus, exp(logQSUSYval), fixed_mZ2_val);
    
    high_prec_float mZ2_p_minus = calculate_approx_mZ2(weaksolsp_minus, exp(logQSUSYval), fixed_mZ2_val);
    high_prec_float mZ2_p_minusminus = calculate_approx_mZ2(weaksolsp_minusminus, exp(logQSUSYval), fixed_mZ2_val);
    
    high_prec_float tanb_original_pt = calculate_approx_tanb(original_weak_conditions, exp(logQSUSYval), fixed_mZ2_val);
    
    high_prec_float tanb_p_plus = calculate_approx_tanb(weaksolsp_plus, exp(logQSUSYval), fixed_mZ2_val);
    high_prec_float tanb_p_plusplus = calculate_approx_tanb(weaksolsp_plusplus, exp(logQSUSYval), fixed_mZ2_val);
    
    high_prec_float tanb_p_minus = calculate_approx_tanb(weaksolsp_minus, exp(logQSUSYval), fixed_mZ2_val);
    high_prec_float tanb_p_minusminus = calculate_approx_tanb(weaksolsp_minusminus, exp(logQSUSYval), fixed_mZ2_val);

    high_prec_float tanb_t_plus = calculate_approx_tanb(weaksolstanb_plus, exp(logQSUSYval), fixed_mZ2_val);
    high_prec_float tanb_t_plusplus = calculate_approx_tanb(weaksolstanb_plusplus, exp(logQSUSYval), fixed_mZ2_val);
    high_prec_float tanb_t_minus = calculate_approx_tanb(weaksolstanb_minus, exp(logQSUSYval), fixed_mZ2_val);
    high_prec_float tanb_t_minusminus = calculate_approx_tanb(weaksolstanb_minusminus, exp(logQSUSYval), fixed_mZ2_val);
    
    /* Order of derivatives:
        (0: d(tanb eq)/dp,
         1: d(tanb eq)/dt
         2: dm/dt, 
         3: dm/dp,
         4: d^2m/dt^2,
         5: d^2m/dp^2,
         6: d^2(tanb eq)/dt^2,
         7: d^2(tanb eq)/dp^2)
    */
    vector<high_prec_float> evaluated_derivs = {first_derivative_calc(h_p, tanb_p_minusminus, tanb_p_minus, tanb_p_plus, tanb_p_plusplus),
                                       first_derivative_calc(h_tanb, tanb_t_minusminus, tanb_t_minus, tanb_t_plus, tanb_t_plusplus),//second_derivative_calc(h_p, wk_tanb, tanb_p_minusminus, tanb_p_minus, tanb_p_plus, tanb_p_plusplus),
                                       first_derivative_calc(h_tanb, mZ2_tanb_minusminus, mZ2_tanb_minus, mZ2_tanb_plus, mZ2_tanb_plusplus),
                                       first_derivative_calc(h_p, mZ2_p_minusminus, mZ2_p_minus, mZ2_p_plus, mZ2_p_plusplus)};//,
                                    //    second_derivative_calc(h_tanb, mZ2_original_pt, mZ2_tanb_minusminus, mZ2_tanb_minus, mZ2_tanb_plus, mZ2_tanb_plusplus),
                                    //    second_derivative_calc(h_p, mZ2_original_pt, mZ2_p_minusminus, mZ2_p_minus, mZ2_p_plus, mZ2_p_plusplus),
                                    //    second_derivative_calc(h_tanb, tanb_original_pt, tanb_t_minusminus, tanb_t_minus, tanb_t_plus, tanb_t_plusplus),
                                    //    second_derivative_calc(h_p, tanb_original_pt, tanb_p_minusminus, tanb_p_minus, tanb_p_plus, tanb_p_plusplus)};//,
                                    //    second_derivative_calc(h_tanb, fixed_mZ2_val, mZ2_tanb_minusminus, mZ2_tanb_minus, mZ2_tanb_plus, mZ2_tanb_plusplus),
                                    //    mixed_second_derivative_calc(h_p, h_tanb, mZ2_p_minusminus_tanb_minusminus, mZ2_p_minusminus_tanb_minus, mZ2_p_minusminus_tanb_plus, mZ2_p_minusminus_tanb_plusplus,
                                    //                                 mZ2_p_minus_tanb_minusminus, mZ2_p_minus_tanb_minus, mZ2_p_minus_tanb_plus, mZ2_p_minus_tanb_plusplus, mZ2_p_plus_tanb_minusminus, mZ2_p_plus_tanb_minus,
                                    //                                 mZ2_p_plus_tanb_plus, mZ2_p_plus_tanb_plusplus, mZ2_p_plusplus_tanb_minusminus, mZ2_p_plusplus_tanb_minus, mZ2_p_plusplus_tanb_plus, mZ2_p_plusplus_tanb_plusplus),
                                    //    second_derivative_calc(h_p, fixed_mZ2_val, mZ2_p_minusminus, mZ2_p_minus, mZ2_p_plus, mZ2_p_plusplus)};
    return evaluated_derivs;
}

vector<high_prec_float> DSN_B_windows(vector<high_prec_float> Wk_boundary_conditions, high_prec_float& current_mZ2, high_prec_float& current_logQSUSY) {
    vector<high_prec_float> Bnewweaks_plus = Wk_boundary_conditions;
    vector<high_prec_float> Bnewweaks_minus = Wk_boundary_conditions;
    high_prec_float BcurrentlogQSUSY = current_logQSUSY;
    high_prec_float BnewlogQSUSY = current_logQSUSY;
    high_prec_float Bnew_mZ2plus = current_mZ2;
    high_prec_float Bnew_mZ2minus = current_mZ2;
    bool BminusNoCCB = true;
    bool BminusEWSB = true;
    bool BplusNoCCB = true;
    bool BplusEWSB = true;

    high_prec_float Bplus = Bnewweaks_plus[42] / Wk_boundary_conditions[6];
    high_prec_float newBplus = Bplus;
    high_prec_float tanbplus = Bnewweaks_plus[43];
    high_prec_float newtanbplus = tanbplus;

    high_prec_float Bminus = Bnewweaks_minus[42] / Wk_boundary_conditions[6];
    high_prec_float newBminus = Bminus;
    high_prec_float tanbminus = Bnewweaks_minus[43];
    high_prec_float newtanbminus = tanbminus;
    high_prec_float muGUT_original = Wk_boundary_conditions[6];

    // First compute width of ABDS window
    high_prec_float lambdaB = 0.5;
    high_prec_float B_least_Sq_tol = 1.0e-8;
    high_prec_float prev_fB = std::numeric_limits<high_prec_float>::max();
    high_prec_float curr_lsq_eval = std::numeric_limits<high_prec_float>::max();
    vector<high_prec_float> current_derivatives = single_var_deriv_approxes(Bnewweaks_minus, Bnew_mZ2minus, 42, BnewlogQSUSY);
    for (high_prec_float deriv_value : current_derivatives) {
        if (isnan(deriv_value) || isinf(deriv_value)) {
            BminusEWSB = false;
        }
    }
    int max_iter = 100;
    high_prec_float tol = 1.0e-8;
    while ((BminusEWSB) && (BminusNoCCB) && (abs(Bnewweaks_minus[6]) > 25.0) && ((Bnew_mZ2minus > (45.5938 * 45.5938)) && (Bnew_mZ2minus < (364.7504 * 364.7504)))) {
        vector<high_prec_float> checkweaksols = Bnewweaks_minus;
        vector<high_prec_float> checkRadCorrs = radcorr_calc(checkweaksols, exp(BnewlogQSUSY), Bnew_mZ2minus);
        BminusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
        // Checking loop-level EWSB
        if (BminusEWSB == true) {
            BminusEWSB = Hessian_check(checkweaksols, exp(BnewlogQSUSY));
        }
        BminusNoCCB = CCB_Check(checkweaksols);
        if (!(BminusEWSB) || !(BminusNoCCB)) {
            break;
        } 

        Bnew_mZ2minus = pow((sqrt(Bnew_mZ2minus) - 0.05), 2.0);
        std::cout << "New mZ- = " << sqrt(Bnew_mZ2minus) << endl;
        std::cout << "New B- = " << Bnewweaks_minus[42] / Bnewweaks_minus[6] << endl;
        std::cout << "New tanb- = " << Bnewweaks_minus[43] << endl;
        //high_prec_float Bbefore = Bnewweaks_minus[6];
        try {
            for (int i = 0; i < max_iter; ++i) {
                vector<high_prec_float> current_derivatives = single_var_deriv_approxes(Bnewweaks_minus, Bnew_mZ2minus, 42, BnewlogQSUSY);
                
                high_prec_float det = (current_derivatives[3] * current_derivatives[1]) - (current_derivatives[2] * current_derivatives[0]);
                high_prec_float Feval = calculate_approx_mZ2(Bnewweaks_minus, exp(current_logQSUSY), Bnew_mZ2minus);
                high_prec_float Geval = calculate_approx_tanb(Bnewweaks_minus, exp(current_logQSUSY), Bnew_mZ2minus);
                high_prec_float dB = ((current_derivatives[1] * Feval) - (current_derivatives[2] * Geval)) / det;
                high_prec_float dtanb = ((current_derivatives[3] * Geval) - (current_derivatives[0] * Feval)) / det;

                vector<high_prec_float> oldweaks = Bnewweaks_minus;
                high_prec_float oldB = oldweaks[42] / oldweaks[6];
                high_prec_float newB = oldB - dB;
                
                Bnewweaks_minus[42] *= newB / oldB;
                Bnewweaks_minus[43] -= dtanb;
                for (int YukIndx = 7; YukIndx < 16; ++YukIndx) {
                    if ((YukIndx >=7) && (YukIndx < 10)) {
                        Bnewweaks_minus[YukIndx] *= sin(atan(oldweaks[43])) / sin(atan(Bnewweaks_minus[43]));
                        Bnewweaks_minus[YukIndx+9] *= Bnewweaks_minus[YukIndx] / oldweaks[YukIndx];
                    } else {
                        Bnewweaks_minus[YukIndx] *= cos(atan(oldweaks[43])) / cos(atan(Bnewweaks_minus[43]));
                        Bnewweaks_minus[YukIndx+9] *= Bnewweaks_minus[YukIndx] / oldweaks[YukIndx];
                    }
                }
                // std::cout << "current L2: " << sqrt((dB * dB) + (dtanb * dtanb)) << endl;
                if (sqrt((Feval * Feval) + (Geval * Geval)) < tol) {
                    break;
                }
                if ((i == 99) && (sqrt((Feval * Feval) + (Geval * Geval)) > tol)) {
                    BminusEWSB = false;
                    Bnewweaks_minus[42] = checkweaksols[42];
                }
            } 
        } catch (...) {
            BminusEWSB = false;
        }
        vector<high_prec_float> oldweak = Bnewweaks_minus;
        if ((Bnewweaks_minus[43] < 3.0) || (Bnewweaks_minus[43] > 60.0)) {
            BminusEWSB = false;
        } else {
            for (int YukIndx = 7; YukIndx < 16; ++YukIndx) {
                if ((YukIndx >=7) && (YukIndx < 10)) {
                    Bnewweaks_minus[YukIndx] *= sin(atan(oldweak[43])) / sin(atan(Bnewweaks_minus[43]));
                    Bnewweaks_minus[YukIndx+9] *= Bnewweaks_minus[YukIndx] / oldweak[YukIndx];
                } else {
                    Bnewweaks_minus[YukIndx] *= cos(atan(oldweak[43])) / cos(atan(Bnewweaks_minus[43]));
                    Bnewweaks_minus[YukIndx+9] *= Bnewweaks_minus[YukIndx] / oldweak[YukIndx];
                }
            }
        }
    }
    
    high_prec_float B_weak_minus = Bnewweaks_minus[42] / Bnewweaks_minus[6];
    
    while ((BplusEWSB) && (BplusNoCCB) && (abs(Bnewweaks_plus[6]) > 25.0) && ((Bnew_mZ2plus > (45.5938 * 45.5938)) && (Bnew_mZ2plus < (364.7504 * 364.7504)))) {
        vector<high_prec_float> checkweaksols = Bnewweaks_plus;
        vector<high_prec_float> checkRadCorrs = radcorr_calc(checkweaksols, exp(BnewlogQSUSY), Bnew_mZ2plus);
        BplusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
        // Checking loop-level EWSB
        if (BplusEWSB == true) {
            BplusEWSB = Hessian_check(checkweaksols, exp(BnewlogQSUSY));
        }
        BplusNoCCB = CCB_Check(checkweaksols);
        if (!(BplusEWSB) || !(BplusNoCCB)) {
            break;
        } 

        Bnew_mZ2plus = pow((sqrt(Bnew_mZ2plus) + 0.05), 2.0);
        std::cout << "New mZ+ = " << sqrt(Bnew_mZ2plus) << endl;
        std::cout << "New B+ = " << Bnewweaks_plus[42] / Bnewweaks_plus[6] << endl;
        std::cout << "New tanb+ = " << Bnewweaks_plus[43] << endl;
        //high_prec_float Bbefore = Bnewweaks_plus[6];
        try {
            for (int i = 0; i < max_iter; ++i) {
                vector<high_prec_float> current_derivatives = single_var_deriv_approxes(Bnewweaks_plus, Bnew_mZ2plus, 42, BnewlogQSUSY);
                
                high_prec_float det = (current_derivatives[3] * current_derivatives[1]) - (current_derivatives[2] * current_derivatives[0]);
                high_prec_float Feval = calculate_approx_mZ2(Bnewweaks_plus, exp(current_logQSUSY), Bnew_mZ2plus);
                high_prec_float Geval = calculate_approx_tanb(Bnewweaks_plus, exp(current_logQSUSY), Bnew_mZ2plus);
                high_prec_float dB = ((current_derivatives[1] * Feval) - (current_derivatives[2] * Geval)) / det;
                high_prec_float dtanb = ((current_derivatives[3] * Geval) - (current_derivatives[0] * Feval)) / det;

                vector<high_prec_float> oldweaks = Bnewweaks_plus;
                high_prec_float oldB = oldweaks[42] / oldweaks[6];
                high_prec_float newB = oldB - dB;
                
                Bnewweaks_plus[42] *= newB / oldB;
                Bnewweaks_plus[43] -= dtanb;
                for (int YukIndx = 7; YukIndx < 16; ++YukIndx) {
                    if ((YukIndx >=7) && (YukIndx < 10)) {
                        Bnewweaks_plus[YukIndx] *= sin(atan(oldweaks[43])) / sin(atan(Bnewweaks_plus[43]));
                        Bnewweaks_plus[YukIndx+9] *= Bnewweaks_plus[YukIndx] / oldweaks[YukIndx];
                    } else {
                        Bnewweaks_plus[YukIndx] *= cos(atan(oldweaks[43])) / cos(atan(Bnewweaks_plus[43]));
                        Bnewweaks_plus[YukIndx+9] *= Bnewweaks_plus[YukIndx] / oldweaks[YukIndx];
                    }
                }
                // std::cout << "current L2: " << sqrt((dB * dB) + (dtanb * dtanb)) << endl;
                if (sqrt((Feval * Feval) + (Geval * Geval)) < tol) {
                    break;
                }
                if ((i == 99) && (sqrt((Feval * Feval) + (Geval * Geval)) > tol)) {
                    BplusEWSB = false;
                    Bnewweaks_plus[42] = checkweaksols[42];
                }
            } 
        } catch (...) {
            BplusEWSB = false;
        }
        vector<high_prec_float> oldweak = Bnewweaks_plus;
        if ((Bnewweaks_plus[43] < 3.0) || (Bnewweaks_plus[43] > 60.0)) {
            BplusEWSB = false;
        } else {
            for (int YukIndx = 7; YukIndx < 16; ++YukIndx) {
                if ((YukIndx >=7) && (YukIndx < 10)) {
                    Bnewweaks_plus[YukIndx] *= sin(atan(oldweak[43])) / sin(atan(Bnewweaks_plus[43]));
                    Bnewweaks_plus[YukIndx+9] *= Bnewweaks_plus[YukIndx] / oldweak[YukIndx];
                } else {
                    Bnewweaks_plus[YukIndx] *= cos(atan(oldweak[43])) / cos(atan(Bnewweaks_plus[43]));
                    Bnewweaks_plus[YukIndx+9] *= Bnewweaks_plus[YukIndx] / oldweak[YukIndx];
                }
            }
        }
    }
    
    high_prec_float B_weak_plus = Bnewweaks_plus[42] / Wk_boundary_conditions[6];
    
    std::cout << "ABDS window established for B variation." << endl;

    return {B_weak_minus, B_weak_plus};//, B_TOTAL_weak_minus, B_TOTAL_weak_plus};
}

vector<high_prec_float> DSN_specific_windows(vector<high_prec_float>& Wk_boundary_conditions, high_prec_float& current_mZ2, high_prec_float& current_logQSUSY, int SpecificIndex) {
    high_prec_float t_target = log(500.0);
    vector<high_prec_float> pinewweaks_plus = Wk_boundary_conditions;
    vector<high_prec_float> pinewweaks_minus = Wk_boundary_conditions;
    high_prec_float picurrentlogQSUSY = current_logQSUSY;
    high_prec_float pinewlogQSUSY = current_logQSUSY;
    high_prec_float pinew_mZ2plus = current_mZ2;
    high_prec_float pinew_mZ2minus = current_mZ2;
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

    high_prec_float piplus = pinewweaks_plus[SpecificIndex];
    high_prec_float newpiplus = piplus;
    high_prec_float tanbplus = pinewweaks_plus[43];
    high_prec_float newtanbplus = tanbplus;

    high_prec_float piminus = pinewweaks_minus[SpecificIndex];
    high_prec_float newpiminus = piminus;
    high_prec_float tanbminus = pinewweaks_minus[43];
    high_prec_float newtanbminus = tanbminus;
    // First compute width of ABDS window
    vector<high_prec_float> current_derivatives = single_var_deriv_approxes(pinewweaks_minus, pinew_mZ2minus, SpecificIndex, pinewlogQSUSY);
    for (high_prec_float deriv_value : current_derivatives) {
        if (isnan(deriv_value) || isinf(deriv_value)) {
            piminusEWSB = false;
        }
    }
    int max_iter = 100;
    high_prec_float tol = 1.0e-8;
    bool numstabminus = true;
    bool numstabplus = true;
    while ((piminusEWSB) && (piminusNoCCB) && (numstabminus) && (abs(pinewweaks_minus[6]) > 25.0) && ((pinew_mZ2minus > (45.5938 * 45.5938)) && (pinew_mZ2minus < (364.7504 * 364.7504)))) {
        vector<high_prec_float> checkweaksols = pinewweaks_minus;
        vector<high_prec_float> checkRadCorrs = radcorr_calc(checkweaksols, exp(pinewlogQSUSY), pinew_mZ2minus);
        piminusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
        // Checking loop-level EWSB
        if (piminusEWSB == true) {
            piminusEWSB = Hessian_check(checkweaksols, exp(pinewlogQSUSY));
        }
        piminusNoCCB = CCB_Check(checkweaksols);
        if (!(piminusEWSB) || !(piminusNoCCB)) {
            break;
        } 

        pinew_mZ2minus = pow((sqrt(pinew_mZ2minus) - (high_prec_float(1.0) / high_prec_float(100.0))), 2.0);
        std::cout << "New mZ- = " << sqrt(pinew_mZ2minus) << endl;
        std::cout << "New " << paramName << "- = " << pinewweaks_minus[SpecificIndex] << endl;
        std::cout << "New tanb- = " << pinewweaks_minus[43] << endl;
        try {
            for (int i = 0; i < max_iter; ++i) {
                vector<high_prec_float> current_derivatives = single_var_deriv_approxes(pinewweaks_minus, pinew_mZ2minus, SpecificIndex, pinewlogQSUSY);
                
                high_prec_float det = (current_derivatives[3] * current_derivatives[1]) - (current_derivatives[2] * current_derivatives[0]);
                high_prec_float Feval = calculate_approx_mZ2(pinewweaks_minus, exp(current_logQSUSY), pinew_mZ2minus);
                high_prec_float Geval = calculate_approx_tanb(pinewweaks_minus, exp(current_logQSUSY), pinew_mZ2minus);
                high_prec_float dpi = ((current_derivatives[1] * Feval) - (current_derivatives[2] * Geval)) / det;
                high_prec_float dtanb = ((current_derivatives[3] * Geval) - (current_derivatives[0] * Feval)) / det;

                vector<high_prec_float> oldweaks = pinewweaks_minus;
                
                if ((SpecificIndex >= 16) && (SpecificIndex <= 24)) {
                    pinewweaks_minus[SpecificIndex] = ((pinewweaks_minus[SpecificIndex] / pinewweaks_minus[SpecificIndex-9]) - dpi) * pinewweaks_minus[SpecificIndex-9];
                } else {
                    pinewweaks_minus[SpecificIndex] -= dpi;
                }
                // pinewweaks_minus[SpecificIndex] -= dpi;
                pinewweaks_minus[43] -= dtanb;
                for (int YukIndx = 7; YukIndx < 16; ++YukIndx) {
                    if ((YukIndx >=7) && (YukIndx < 10)) {
                        pinewweaks_minus[YukIndx] *= sin(atan(oldweaks[43])) / sin(atan(pinewweaks_minus[43]));
                        pinewweaks_minus[YukIndx+9] *= pinewweaks_minus[YukIndx] / oldweaks[YukIndx];
                    } else {
                        pinewweaks_minus[YukIndx] *= cos(atan(oldweaks[43])) / cos(atan(pinewweaks_minus[43]));
                        pinewweaks_minus[YukIndx+9] *= pinewweaks_minus[YukIndx] / oldweaks[YukIndx];
                    }
                }
                // std::cout << "current L2: " << sqrt((Feval * Feval) + (Geval * Geval)) << endl;
                if (sqrt((Feval * Feval) + (Geval * Geval)) < tol) {
                    break;
                }
                if ((i == 99) && (sqrt((Feval * Feval) + (Geval * Geval)) > tol)) {
                    numstabminus = false;
                    pinewweaks_minus = checkweaksols;
                    high_prec_float oldmZ2val = pow((sqrt(pinew_mZ2minus) + (high_prec_float(1.0) / high_prec_float(100.0))), 2.0);
    
                    // Will try to overcome numerical instability by linearizing the system once and obtaining new initial guesses
                    // high_prec_float prev_F = calculate_approx_mZ2(pinewweaks_minus, exp(pinewlogQSUSY), pow((sqrt(pinew_mZ2minus) + (high_prec_float(1.0) / high_prec_float(100.0))), 2.0));
                    // high_prec_float prev_G = calculate_approx_tanb(pinewweaks_minus, exp(pinewlogQSUSY), pow((sqrt(pinew_mZ2minus) + (high_prec_float(1.0) / high_prec_float(100.0))), 2.0));

                    // Collect derivatives needed for shifts
                    /* derivs_first_order:
                        (0: df/dm
                         1: dg/dm
                         2: df/dp
                         3: dg/dp
                         4: df/dt
                         5: dg/dt)
                    */
                    std::vector<high_prec_float> pt_first_derivs = single_var_deriv_approxes(pinewweaks_minus, oldmZ2val, SpecificIndex, pinewlogQSUSY);
                    std::vector<high_prec_float> m_first_derivs = mZ2_derivatives(pinewweaks_minus, oldmZ2val, pinewlogQSUSY);
                    std::vector<high_prec_float> derivs_first_order = {m_first_derivs[0], m_first_derivs[1], pt_first_derivs[3], pt_first_derivs[0], pt_first_derivs[2], pt_first_derivs[1]};
                    /* derivs_2_and_3:
                    (0: d^2f/dmdp
                     1: d^2f/dmdt
                     2: d^2f/dpdt
                     3: d^3f/dmdpdt
                     4: d^2g/dmdp
                     5: d^2g/dmdt
                     6: d^2g/dpdt
                     7: d^3g/dmdpdt)
                    */
                    std::vector<high_prec_float> derivs_2_and_3 = crossDerivs(pinewweaks_minus, oldmZ2val, SpecificIndex, pinewlogQSUSY);
                    std::vector<high_prec_float> initGuessesShifts = modifyNewtonInitGuesses(high_prec_float(1.0) / high_prec_float(100.0),
                                                                                            derivs_2_and_3[3], derivs_2_and_3[7], derivs_2_and_3[0], derivs_2_and_3[4], derivs_2_and_3[1], derivs_2_and_3[5],
                                                                                            derivs_2_and_3[2], derivs_2_and_3[6], derivs_first_order[4], derivs_first_order[2], derivs_first_order[5], derivs_first_order[3],
                                                                                            derivs_first_order[0], derivs_first_order[1]);
                    std::vector<high_prec_float> pinewweaks_minus2 = pinewweaks_minus;
                    if ((SpecificIndex >= 16) && (SpecificIndex <= 24)) {
                        pinewweaks_minus2[SpecificIndex] = ((pinewweaks_minus2[SpecificIndex] / pinewweaks_minus2[SpecificIndex-9]) + (high_prec_float(10.0) * initGuessesShifts[0])) * pinewweaks_minus2[SpecificIndex-9];
                    } else {
                        pinewweaks_minus2[SpecificIndex] += high_prec_float(10.0) * initGuessesShifts[0];
                    }
                    pinewweaks_minus2[43] += initGuessesShifts[1];
                    std::vector<high_prec_float> RadCorrs2 = radcorr_calc(pinewweaks_minus2, exp(pinewlogQSUSY), oldmZ2val);
                    // Try convergence again, but check EWSB/CCB first
                    if ((pinewweaks_minus2[43] >= high_prec_float(3.0)) && (pinewweaks_minus2[43] <= high_prec_float(60.0)) && EWSB_Check(pinewweaks_minus2, RadCorrs2) && Hessian_check(pinewweaks_minus2, exp(pinewlogQSUSY)) && CCB_Check(pinewweaks_minus2)) {
                        for (int i2 = 0; i2 < max_iter; ++i2) {
                            vector<high_prec_float> current_derivatives2 = single_var_deriv_approxes(pinewweaks_minus2, pinew_mZ2minus, SpecificIndex, pinewlogQSUSY);
                            
                            high_prec_float det2 = (current_derivatives2[3] * current_derivatives2[1]) - (current_derivatives2[2] * current_derivatives2[0]);
                            high_prec_float Feval2 = calculate_approx_mZ2(pinewweaks_minus2, exp(current_logQSUSY), pinew_mZ2minus);
                            high_prec_float Geval2 = calculate_approx_tanb(pinewweaks_minus2, exp(current_logQSUSY), pinew_mZ2minus);
                            high_prec_float dpi2 = ((current_derivatives2[1] * Feval2) - (current_derivatives2[2] * Geval2)) / det2;
                            high_prec_float dtanb2 = ((current_derivatives2[3] * Geval2) - (current_derivatives2[0] * Feval2)) / det2;

                            vector<high_prec_float> oldweaks2 = pinewweaks_minus2;
                            
                            if ((SpecificIndex >= 16) && (SpecificIndex <= 24)) {
                                pinewweaks_minus2[SpecificIndex] = ((pinewweaks_minus2[SpecificIndex] / pinewweaks_minus2[SpecificIndex-9]) - dpi2) * pinewweaks_minus2[SpecificIndex-9];
                            } else {
                                pinewweaks_minus2[SpecificIndex] -= dpi2;
                            }
                            // pinewweaks_minus2[SpecificIndex] -= dpi2;
                            pinewweaks_minus2[43] -= dtanb2;
                            for (int YukIndx = 7; YukIndx < 16; ++YukIndx) {
                                if ((YukIndx >=7) && (YukIndx < 10)) {
                                    pinewweaks_minus2[YukIndx] *= sin(atan(oldweaks2[43])) / sin(atan(pinewweaks_minus2[43]));
                                    pinewweaks_minus2[YukIndx+9] *= pinewweaks_minus2[YukIndx] / oldweaks2[YukIndx];
                                } else {
                                    pinewweaks_minus2[YukIndx] *= cos(atan(oldweaks2[43])) / cos(atan(pinewweaks_minus2[43]));
                                    pinewweaks_minus2[YukIndx+9] *= pinewweaks_minus2[YukIndx] / oldweaks2[YukIndx];
                                }
                            }
                            // std::cout << "current L2: " << sqrt((Feval2 * Feval2) + (Geval2 * Geval2)) << endl;
                            if (sqrt((Feval2 * Feval2) + (Geval2 * Geval2)) < tol) {
                                numstabminus = true;
                                break;
                            }
                            if ((i2 == 99) && (sqrt((Feval2 * Feval2) + (Geval2 * Geval2)) > tol)) {
                                numstabminus = false;
                            }
                        }
                    } else {
                        numstabminus = false;
                    }

                    // Check if converged
                    if (numstabminus) {
                        pinewweaks_minus = pinewweaks_minus2;
                    } else {
                        // Try convergence with other solution for dp, dt as last resort
                        std::vector<high_prec_float> pinewweaks_minus3 = pinewweaks_minus;
                        if ((SpecificIndex >= 16) && (SpecificIndex <= 24)) {
                            pinewweaks_minus3[SpecificIndex] = ((pinewweaks_minus3[SpecificIndex] / pinewweaks_minus3[SpecificIndex-9]) + high_prec_float(10.0) * initGuessesShifts[2]) * pinewweaks_minus3[SpecificIndex-9];
                        } else {
                            pinewweaks_minus3[SpecificIndex] += high_prec_float(10.0) * initGuessesShifts[2];
                        }
                        pinewweaks_minus3[43] += initGuessesShifts[3];
                        std::vector<high_prec_float> RadCorrs3 = radcorr_calc(pinewweaks_minus3, exp(pinewlogQSUSY), oldmZ2val);
                        // Try convergence again, but check EWSB/CCB first
                        if ((pinewweaks_minus3[43] >= high_prec_float(3.0)) && (pinewweaks_minus3[43] <= high_prec_float(60.0)) && EWSB_Check(pinewweaks_minus3, RadCorrs3) && Hessian_check(pinewweaks_minus3, exp(pinewlogQSUSY)) && CCB_Check(pinewweaks_minus3)) {
                            for (int i3 = 0; i3 < max_iter; ++i3) {
                                vector<high_prec_float> current_derivatives3 = single_var_deriv_approxes(pinewweaks_minus3, pinew_mZ2minus, SpecificIndex, pinewlogQSUSY);
                                
                                high_prec_float det = (current_derivatives3[3] * current_derivatives3[1]) - (current_derivatives3[2] * current_derivatives3[0]);
                                high_prec_float Feval3 = calculate_approx_mZ2(pinewweaks_minus3, exp(current_logQSUSY), pinew_mZ2minus);
                                high_prec_float Geval3 = calculate_approx_tanb(pinewweaks_minus3, exp(current_logQSUSY), pinew_mZ2minus);
                                high_prec_float dpi3 = ((current_derivatives3[1] * Feval3) - (current_derivatives3[2] * Geval3)) / det;
                                high_prec_float dtanb3 = ((current_derivatives3[3] * Geval3) - (current_derivatives3[0] * Feval3)) / det;

                                vector<high_prec_float> oldweaks3 = pinewweaks_minus3;
                                
                                if ((SpecificIndex >= 16) && (SpecificIndex <= 24)) {
                                    pinewweaks_minus3[SpecificIndex] = ((pinewweaks_minus3[SpecificIndex] / pinewweaks_minus3[SpecificIndex-9]) - dpi3) * pinewweaks_minus3[SpecificIndex-9];
                                } else {
                                    pinewweaks_minus3[SpecificIndex] -= dpi3;
                                }
                                // pinewweaks_minus3[SpecificIndex] -= dpi3;
                                pinewweaks_minus3[43] -= dtanb3;
                                for (int YukIndx = 7; YukIndx < 16; ++YukIndx) {
                                    if ((YukIndx >=7) && (YukIndx < 10)) {
                                        pinewweaks_minus3[YukIndx] *= sin(atan(oldweaks3[43])) / sin(atan(pinewweaks_minus3[43]));
                                        pinewweaks_minus3[YukIndx+9] *= pinewweaks_minus3[YukIndx] / oldweaks3[YukIndx];
                                    } else {
                                        pinewweaks_minus3[YukIndx] *= cos(atan(oldweaks3[43])) / cos(atan(pinewweaks_minus3[43]));
                                        pinewweaks_minus3[YukIndx+9] *= pinewweaks_minus3[YukIndx] / oldweaks3[YukIndx];
                                    }
                                }
                                // std::cout << "current L2: " << sqrt((Feval3 * Feval3) + (Geval3 * Geval3)) << endl;
                                if (sqrt((Feval3 * Feval3) + (Geval3 * Geval3)) < tol) {
                                    numstabminus = true;
                                    break;
                                }
                                if ((i3 == 99) && (sqrt((Feval3 * Feval3) + (Geval3 * Geval3)) > tol)) {
                                    numstabminus = false;
                                }
                            }
                        } else {
                            numstabminus = false;
                        }
                        if (numstabminus) {
                            pinewweaks_minus = pinewweaks_minus3;
                        }
                    }
                }
            } 
        } catch (...) {
            piminusEWSB = false;
        }
        vector<high_prec_float> oldweak = pinewweaks_minus;
        if ((pinewweaks_minus[43] < 3.0) || (pinewweaks_minus[43] > 60.0)) {
            piminusEWSB = false;
        }
    }

    high_prec_float pi_weak_minus = pinewweaks_minus[SpecificIndex];

    while ((piplusEWSB) && (piplusNoCCB) && (numstabplus) && (abs(pinewweaks_plus[6]) > 25.0) && ((pinew_mZ2plus > (45.5938 * 45.5938)) && (pinew_mZ2plus < (364.7504 * 364.7504)))) {
        vector<high_prec_float> checkweaksols = pinewweaks_plus;
        vector<high_prec_float> checkRadCorrs = radcorr_calc(checkweaksols, exp(pinewlogQSUSY), pinew_mZ2plus);
        piplusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
        // Checking loop-level EWSB
        if (piplusEWSB == true) {
            piplusEWSB = Hessian_check(checkweaksols, exp(pinewlogQSUSY));
        }
        piplusNoCCB = CCB_Check(checkweaksols);
        if (!(piplusEWSB) || !(piplusNoCCB)) {
            break;
        } 

        pinew_mZ2plus = pow((sqrt(pinew_mZ2plus) + (high_prec_float(1.0) / high_prec_float(100.0))), 2.0);
        std::cout << "New mZ+ = " << sqrt(pinew_mZ2plus) << endl;
        std::cout << "New " << paramName << "+ = " << pinewweaks_plus[SpecificIndex] << endl;
        std::cout << "New tanb+ = " << pinewweaks_plus[43] << endl;
        try {
            for (int i = 0; i < max_iter; ++i) {
                vector<high_prec_float> current_derivatives = single_var_deriv_approxes(pinewweaks_plus, pinew_mZ2plus, SpecificIndex, pinewlogQSUSY);
                
                high_prec_float det = (current_derivatives[3] * current_derivatives[1]) - (current_derivatives[2] * current_derivatives[0]);
                high_prec_float Feval = calculate_approx_mZ2(pinewweaks_plus, exp(current_logQSUSY), pinew_mZ2plus);
                high_prec_float Geval = calculate_approx_tanb(pinewweaks_plus, exp(current_logQSUSY), pinew_mZ2plus);
                high_prec_float dpi = ((current_derivatives[1] * Feval) - (current_derivatives[2] * Geval)) / det;
                high_prec_float dtanb = ((current_derivatives[3] * Geval) - (current_derivatives[0] * Feval)) / det;

                vector<high_prec_float> oldweaks = pinewweaks_plus;
                
                if ((SpecificIndex >= 16) && (SpecificIndex <= 24)) {
                    pinewweaks_plus[SpecificIndex] = ((pinewweaks_plus[SpecificIndex] / pinewweaks_plus[SpecificIndex-9]) - dpi) * pinewweaks_plus[SpecificIndex-9];
                } else {
                    pinewweaks_plus[SpecificIndex] -= dpi;
                }
                // pinewweaks_plus[SpecificIndex] -= dpi;
                pinewweaks_plus[43] -= dtanb;
                for (int YukIndx = 7; YukIndx < 16; ++YukIndx) {
                    if ((YukIndx >=7) && (YukIndx < 10)) {
                        pinewweaks_plus[YukIndx] *= sin(atan(oldweaks[43])) / sin(atan(pinewweaks_plus[43]));
                        pinewweaks_plus[YukIndx+9] *= pinewweaks_plus[YukIndx] / oldweaks[YukIndx];
                    } else {
                        pinewweaks_plus[YukIndx] *= cos(atan(oldweaks[43])) / cos(atan(pinewweaks_plus[43]));
                        pinewweaks_plus[YukIndx+9] *= pinewweaks_plus[YukIndx] / oldweaks[YukIndx];
                    }
                }
                // std::cout << "current L2: " << sqrt((Feval * Feval) + (Geval * Geval)) << endl;
                if (sqrt((Feval * Feval) + (Geval * Geval)) < tol) {
                    break;
                }
                if ((i == 99) && (sqrt((Feval * Feval) + (Geval * Geval)) > tol)) {
                    numstabplus = false;
                    pinewweaks_plus = checkweaksols;
                    high_prec_float oldmZ2val = pow((sqrt(pinew_mZ2plus) - (high_prec_float(1.0) / high_prec_float(100.0))), 2.0);
    
                    // Will try to overcome numerical instability by linearizing the system once and obtaining new initial guesses
                    // high_prec_float prev_F = calculate_approx_mZ2(pinewweaks_plus, exp(pinewlogQSUSY), pow((sqrt(pinew_mZ2plus) + (high_prec_float(1.0) / high_prec_float(100.0))), 2.0));
                    // high_prec_float prev_G = calculate_approx_tanb(pinewweaks_plus, exp(pinewlogQSUSY), pow((sqrt(pinew_mZ2plus) + (high_prec_float(1.0) / high_prec_float(100.0))), 2.0));

                    // Collect derivatives needed for shifts
                    /* derivs_first_order:
                        (0: df/dm
                         1: dg/dm
                         2: df/dp
                         3: dg/dp
                         4: df/dt
                         5: dg/dt)
                    */
                    std::vector<high_prec_float> pt_first_derivs = single_var_deriv_approxes(pinewweaks_plus, oldmZ2val, SpecificIndex, pinewlogQSUSY);
                    std::vector<high_prec_float> m_first_derivs = mZ2_derivatives(pinewweaks_plus, oldmZ2val, pinewlogQSUSY);
                    std::vector<high_prec_float> derivs_first_order = {m_first_derivs[0], m_first_derivs[1], pt_first_derivs[3], pt_first_derivs[0], pt_first_derivs[2], pt_first_derivs[1]};
                    /* derivs_2_and_3:
                    (0: d^2f/dmdp
                     1: d^2f/dmdt
                     2: d^2f/dpdt
                     3: d^3f/dmdpdt
                     4: d^2g/dmdp
                     5: d^2g/dmdt
                     6: d^2g/dpdt
                     7: d^3g/dmdpdt)
                    */
                    std::vector<high_prec_float> derivs_2_and_3 = crossDerivs(pinewweaks_plus, oldmZ2val, SpecificIndex, pinewlogQSUSY);
                    std::vector<high_prec_float> initGuessesShifts = modifyNewtonInitGuesses(high_prec_float(1.0) / high_prec_float(100.0),
                                                                                            derivs_2_and_3[3], derivs_2_and_3[7], derivs_2_and_3[0], derivs_2_and_3[4], derivs_2_and_3[1], derivs_2_and_3[5],
                                                                                            derivs_2_and_3[2], derivs_2_and_3[6], derivs_first_order[4], derivs_first_order[2], derivs_first_order[5], derivs_first_order[3],
                                                                                            derivs_first_order[0], derivs_first_order[1]);
                    std::vector<high_prec_float> pinewweaks_plus2 = pinewweaks_plus;
                    if ((SpecificIndex >= 16) && (SpecificIndex <= 24)) {
                        pinewweaks_plus2[SpecificIndex] = ((pinewweaks_plus2[SpecificIndex] / pinewweaks_plus2[SpecificIndex-9]) + (high_prec_float(10.0) * initGuessesShifts[0])) * pinewweaks_plus2[SpecificIndex-9];
                    } else {
                        pinewweaks_plus2[SpecificIndex] += high_prec_float(10.0) * initGuessesShifts[0];
                    }
                    pinewweaks_plus2[43] += initGuessesShifts[1];
                    std::vector<high_prec_float> RadCorrs2 = radcorr_calc(pinewweaks_plus2, exp(pinewlogQSUSY), oldmZ2val);
                    // Try convergence again, but check EWSB/CCB first
                    if ((pinewweaks_plus2[43] >= high_prec_float(3.0)) && (pinewweaks_plus2[43] <= high_prec_float(60.0)) && EWSB_Check(pinewweaks_plus2, RadCorrs2) && Hessian_check(pinewweaks_plus2, exp(pinewlogQSUSY)) && CCB_Check(pinewweaks_plus2)) {
                        for (int i2 = 0; i2 < max_iter; ++i2) {
                            vector<high_prec_float> current_derivatives2 = single_var_deriv_approxes(pinewweaks_plus2, pinew_mZ2plus, SpecificIndex, pinewlogQSUSY);
                            
                            high_prec_float det2 = (current_derivatives2[3] * current_derivatives2[1]) - (current_derivatives2[2] * current_derivatives2[0]);
                            high_prec_float Feval2 = calculate_approx_mZ2(pinewweaks_plus2, exp(current_logQSUSY), pinew_mZ2plus);
                            high_prec_float Geval2 = calculate_approx_tanb(pinewweaks_plus2, exp(current_logQSUSY), pinew_mZ2plus);
                            high_prec_float dpi2 = ((current_derivatives2[1] * Feval2) - (current_derivatives2[2] * Geval2)) / det2;
                            high_prec_float dtanb2 = ((current_derivatives2[3] * Geval2) - (current_derivatives2[0] * Feval2)) / det2;

                            vector<high_prec_float> oldweaks2 = pinewweaks_plus2;
                            
                            if ((SpecificIndex >= 16) && (SpecificIndex <= 24)) {
                                pinewweaks_plus2[SpecificIndex] = ((pinewweaks_plus2[SpecificIndex] / pinewweaks_plus2[SpecificIndex-9]) - dpi2) * pinewweaks_plus2[SpecificIndex-9];
                            } else {
                                pinewweaks_plus2[SpecificIndex] -= dpi2;
                            }
                            // pinewweaks_plus2[SpecificIndex] -= dpi2;
                            pinewweaks_plus2[43] -= dtanb2;
                            for (int YukIndx = 7; YukIndx < 16; ++YukIndx) {
                                if ((YukIndx >=7) && (YukIndx < 10)) {
                                    pinewweaks_plus2[YukIndx] *= sin(atan(oldweaks2[43])) / sin(atan(pinewweaks_plus2[43]));
                                    pinewweaks_plus2[YukIndx+9] *= pinewweaks_plus2[YukIndx] / oldweaks2[YukIndx];
                                } else {
                                    pinewweaks_plus2[YukIndx] *= cos(atan(oldweaks2[43])) / cos(atan(pinewweaks_plus2[43]));
                                    pinewweaks_plus2[YukIndx+9] *= pinewweaks_plus2[YukIndx] / oldweaks2[YukIndx];
                                }
                            }
                            // std::cout << "current L2: " << sqrt((Feval2 * Feval2) + (Geval2 * Geval2)) << endl;
                            if (sqrt((Feval2 * Feval2) + (Geval2 * Geval2)) < tol) {
                                numstabplus = true;
                                break;
                            }
                            if ((i2 == 99) && (sqrt((Feval2 * Feval2) + (Geval2 * Geval2)) > tol)) {
                                numstabplus = false;
                            }
                        }
                    } else {
                        numstabplus = false;
                    }

                    // Check if converged
                    if (numstabplus) {
                        pinewweaks_plus = pinewweaks_plus2;
                    } else {
                        // Try convergence with other solution for dp, dt as last resort
                        std::vector<high_prec_float> pinewweaks_plus3 = pinewweaks_plus;
                        if ((SpecificIndex >= 16) && (SpecificIndex <= 24)) {
                            pinewweaks_plus3[SpecificIndex] = ((pinewweaks_plus3[SpecificIndex] / pinewweaks_plus3[SpecificIndex-9]) + (high_prec_float(10.0) * initGuessesShifts[2])) * pinewweaks_plus3[SpecificIndex-9];
                        } else {
                            pinewweaks_plus3[SpecificIndex] += high_prec_float(10.0) * initGuessesShifts[2];
                        }
                        pinewweaks_plus3[43] += initGuessesShifts[3];
                        std::vector<high_prec_float> RadCorrs3 = radcorr_calc(pinewweaks_plus3, exp(pinewlogQSUSY), oldmZ2val);
                        // Try convergence again, but check EWSB/CCB first
                        if ((pinewweaks_plus3[43] >= high_prec_float(3.0)) && (pinewweaks_plus3[43] <= high_prec_float(60.0)) && EWSB_Check(pinewweaks_plus3, RadCorrs3) && Hessian_check(pinewweaks_plus3, exp(pinewlogQSUSY)) && CCB_Check(pinewweaks_plus3)) {
                            for (int i3 = 0; i3 < max_iter; ++i3) {
                                vector<high_prec_float> current_derivatives3 = single_var_deriv_approxes(pinewweaks_plus3, pinew_mZ2plus, SpecificIndex, pinewlogQSUSY);
                                
                                high_prec_float det = (current_derivatives3[3] * current_derivatives3[1]) - (current_derivatives3[2] * current_derivatives3[0]);
                                high_prec_float Feval3 = calculate_approx_mZ2(pinewweaks_plus3, exp(current_logQSUSY), pinew_mZ2plus);
                                high_prec_float Geval3 = calculate_approx_tanb(pinewweaks_plus3, exp(current_logQSUSY), pinew_mZ2plus);
                                high_prec_float dpi3 = ((current_derivatives3[1] * Feval3) - (current_derivatives3[2] * Geval3)) / det;
                                high_prec_float dtanb3 = ((current_derivatives3[3] * Geval3) - (current_derivatives3[0] * Feval3)) / det;

                                vector<high_prec_float> oldweaks3 = pinewweaks_plus3;
                                
                                if ((SpecificIndex >= 16) && (SpecificIndex <= 24)) {
                                    pinewweaks_plus3[SpecificIndex] = ((pinewweaks_plus3[SpecificIndex] / pinewweaks_plus3[SpecificIndex-9]) - dpi3) * pinewweaks_plus3[SpecificIndex-9];
                                } else {
                                    pinewweaks_plus3[SpecificIndex] -= dpi3;
                                }
                                // pinewweaks_plus3[SpecificIndex] -= dpi3;
                                pinewweaks_plus3[43] -= dtanb3;
                                for (int YukIndx = 7; YukIndx < 16; ++YukIndx) {
                                    if ((YukIndx >=7) && (YukIndx < 10)) {
                                        pinewweaks_plus3[YukIndx] *= sin(atan(oldweaks3[43])) / sin(atan(pinewweaks_plus3[43]));
                                        pinewweaks_plus3[YukIndx+9] *= pinewweaks_plus3[YukIndx] / oldweaks3[YukIndx];
                                    } else {
                                        pinewweaks_plus3[YukIndx] *= cos(atan(oldweaks3[43])) / cos(atan(pinewweaks_plus3[43]));
                                        pinewweaks_plus3[YukIndx+9] *= pinewweaks_plus3[YukIndx] / oldweaks3[YukIndx];
                                    }
                                }
                                // std::cout << "current L2: " << sqrt((Feval3 * Feval3) + (Geval3 * Geval3)) << endl;
                                if (sqrt((Feval3 * Feval3) + (Geval3 * Geval3)) < tol) {
                                    numstabplus = true;
                                    break;
                                }
                                if ((i3 == 99) && (sqrt((Feval3 * Feval3) + (Geval3 * Geval3)) > tol)) {
                                    numstabplus = false;
                                }
                            }
                        } else {
                            numstabplus = false;
                        }
                        if (numstabplus) {
                            pinewweaks_plus = pinewweaks_plus3;
                        }
                    }
                }
            } 
        } catch (...) {
            piplusEWSB = false;
        }
        vector<high_prec_float> oldweak = pinewweaks_plus;
        if ((pinewweaks_plus[43] < 3.0) || (pinewweaks_plus[43] > 60.0)) {
            piplusEWSB = false;
        } else {
            for (int YukIndx = 7; YukIndx < 16; ++YukIndx) {
                if ((YukIndx >=7) && (YukIndx < 10)) {
                    pinewweaks_plus[YukIndx] *= sin(atan(oldweak[43])) / sin(atan(pinewweaks_plus[43]));
                    pinewweaks_plus[YukIndx+9] *= pinewweaks_plus[YukIndx] / oldweak[YukIndx];
                } else {
                    pinewweaks_plus[YukIndx] *= cos(atan(oldweak[43])) / cos(atan(pinewweaks_plus[43]));
                    pinewweaks_plus[YukIndx+9] *= pinewweaks_plus[YukIndx] / oldweak[YukIndx];
                }
            }
        }
    }
    
    high_prec_float pi_weak_plus = pinewweaks_plus[SpecificIndex];
    
    std::cout << "ABDS window established for " << paramName << " variation." << endl;

    return {pi_weak_minus, pi_weak_plus};//, pi_TOTAL_weak_minus, pi_TOTAL_weak_plus};
}

vector<high_prec_float> DSN_mu_windows(vector<high_prec_float>& Wk_boundary_conditions, high_prec_float& current_mZ2, high_prec_float& current_logQSUSY) {
    high_prec_float t_target = log(500.0);
    vector<high_prec_float> munewweaks_plus = Wk_boundary_conditions;
    vector<high_prec_float> munewweaks_minus = Wk_boundary_conditions;
    high_prec_float mucurrentlogQSUSY = current_logQSUSY;
    high_prec_float munewlogQSUSY = current_logQSUSY;
    high_prec_float munew_mZ2plus = current_mZ2;
    high_prec_float munew_mZ2minus = current_mZ2;
    bool muminusNoCCB = true;
    bool muminusEWSB = true;
    bool muplusNoCCB = true;
    bool muplusEWSB = true;

    high_prec_float muplus = munewweaks_plus[6];
    high_prec_float newmuplus = muplus;
    high_prec_float tanbplus = munewweaks_plus[43];
    high_prec_float newtanbplus = tanbplus;

    high_prec_float muminus = munewweaks_minus[6];
    high_prec_float newmuminus = muminus;
    high_prec_float tanbminus = munewweaks_minus[43];
    high_prec_float newtanbminus = tanbminus;

    // First compute width of ABDS window
    high_prec_float lambdaMu = 0.5;
    high_prec_float Mu_least_Sq_tol = 1.0e-8;
    high_prec_float prev_fmu = std::numeric_limits<high_prec_float>::max();
    high_prec_float curr_lsq_eval = std::numeric_limits<high_prec_float>::max();
    vector<high_prec_float> current_derivatives = single_var_deriv_approxes(munewweaks_minus, munew_mZ2minus, 6, munewlogQSUSY);
    for (high_prec_float deriv_value : current_derivatives) {
        std::cout << deriv_value << endl;
        if (isnan(deriv_value) || isinf(deriv_value)) {
            muminusEWSB = false;
        }
    }
    int max_iter = 100;
    high_prec_float tol = 1.0e-8;
    while ((muminusEWSB) && (muminusNoCCB) && (abs(munewweaks_minus[6]) > 25.0) && ((munew_mZ2minus > (45.5938 * 45.5938)) && (munew_mZ2minus < (364.7504 * 364.7504)))) {
        vector<high_prec_float> checkweaksols = munewweaks_minus;
        vector<high_prec_float> checkRadCorrs = radcorr_calc(checkweaksols, exp(munewlogQSUSY), munew_mZ2minus);
        muminusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
        // Checking loop-level EWSB
        if (muminusEWSB == true) {
            muminusEWSB = Hessian_check(checkweaksols, exp(munewlogQSUSY));
        }
        muminusNoCCB = CCB_Check(checkweaksols);
        if (!(muminusEWSB) || !(muminusNoCCB)) {
            break;
        } 

        munew_mZ2minus = pow((sqrt(munew_mZ2minus) - 0.1), 2.0);
        std::cout << "New mZ- = " << sqrt(munew_mZ2minus) << endl;
        std::cout << "New mu- = " << munewweaks_minus[6] << endl;
        std::cout << "New tanb- = " << munewweaks_minus[43] << endl;
        high_prec_float mubefore = munewweaks_minus[6];
        try {
            for (int i = 0; i < max_iter; ++i) {
                vector<high_prec_float> current_derivatives = single_var_deriv_approxes(munewweaks_minus, munew_mZ2minus, 6, munewlogQSUSY);
                
                high_prec_float det = (current_derivatives[3] * current_derivatives[1]) - (current_derivatives[2] * current_derivatives[0]);
                high_prec_float Feval = calculate_approx_mZ2(munewweaks_minus, exp(current_logQSUSY), munew_mZ2minus);
                high_prec_float Geval = calculate_approx_tanb(munewweaks_minus, exp(current_logQSUSY), munew_mZ2minus);
                // high_prec_float alphaterm = (current_derivatives[2] * Geval) - (current_derivatives[1] * Feval);
                // high_prec_float betaterm = (current_derivatives[0] * Feval) - (current_derivatives[3] * Geval);
                // high_prec_float gammaterm = (current_derivatives[7] * current_derivatives[3]) - (current_derivatives[5] * current_derivatives[0]);
                // high_prec_float epsilonterm = (current_derivatives[5] * current_derivatives[1]) - (current_derivatives[7] * current_derivatives[2]);
                // high_prec_float lambdaterm = (current_derivatives[4] * current_derivatives[1]) - (current_derivatives[6] * current_derivatives[2]);
                // high_prec_float omegaterm = (current_derivatives[3] * current_derivatives[6]) - (current_derivatives[0] * current_derivatives[4]);
                high_prec_float dmu = ((current_derivatives[1] * Feval) - (current_derivatives[2] * Geval)) / det;
                high_prec_float dtanb = ((current_derivatives[3] * Geval) - (current_derivatives[0] * Feval)) / det;
                // high_prec_float dmu = (high_prec_float(2.0) * det * alphaterm * alphaterm) / ((alphaterm * alphaterm * epsilonterm) + (betaterm * betaterm * lambdaterm) + (high_prec_float(2.0) * det * det * alphaterm));
                // high_prec_float dtanb = (high_prec_float(2.0) * det * betaterm * betaterm) / ((alphaterm * alphaterm * gammaterm) + (betaterm * betaterm * omegaterm) + (high_prec_float(2.0) * det * det * betaterm));
                vector<high_prec_float> oldweaks = munewweaks_minus;
                munewweaks_minus[6] -= dmu;
                // munewweaks_minus[6] += dmu;
                // std::cout << "dmu = " << dmu << std::endl;
                munewweaks_minus[42] *= munewweaks_minus[6] / oldweaks[6];
                munewweaks_minus[43] -= dtanb;
                // munewweaks_minus[43] += dtanb;
                // std::cout << "dtanb = " << dtanb << std::endl;
                for (int YukIndx = 7; YukIndx < 16; ++YukIndx) {
                    if ((YukIndx >=7) && (YukIndx < 10)) {
                        munewweaks_minus[YukIndx] *= sin(atan(oldweaks[43])) / sin(atan(munewweaks_minus[43]));
                        munewweaks_minus[YukIndx+9] *= munewweaks_minus[YukIndx] / oldweaks[YukIndx];
                    } else {
                        munewweaks_minus[YukIndx] *= cos(atan(oldweaks[43])) / cos(atan(munewweaks_minus[43]));
                        munewweaks_minus[YukIndx+9] *= munewweaks_minus[YukIndx] / oldweaks[YukIndx];
                    }
                }
                // std::cout << "current L2: " << sqrt((Feval * Feval) + (Geval * Geval)) << endl;
                if (sqrt((Feval * Feval) + (Geval * Geval)) < tol) {
                    break;
                }
                if ((i == 99) && (sqrt((Feval * Feval) + (Geval * Geval)) > tol)) {
                    muminusEWSB = false;
                    munewweaks_minus[6] = checkweaksols[6];
                }
            } 
        } catch (...) {
            muminusEWSB = false;
        }
        vector<high_prec_float> oldweak = munewweaks_minus;
        if (signum(mubefore) != signum(munewweaks_minus[6])) {
            muminusEWSB = false;
            munewweaks_minus[6] = mubefore;
        }
        if ((munewweaks_minus[43] < 3.0) || (munewweaks_minus[43] > 60.0)) {
            muminusEWSB = false;
        } else {
            for (int YukIndx = 7; YukIndx < 16; ++YukIndx) {
                if ((YukIndx >=7) && (YukIndx < 10)) {
                    munewweaks_minus[YukIndx] *= sin(atan(oldweak[43])) / sin(atan(munewweaks_minus[43]));
                    munewweaks_minus[YukIndx+9] *= munewweaks_minus[YukIndx] / oldweak[YukIndx];
                } else {
                    munewweaks_minus[YukIndx] *= cos(atan(oldweak[43])) / cos(atan(munewweaks_minus[43]));
                    munewweaks_minus[YukIndx+9] *= munewweaks_minus[YukIndx] / oldweak[YukIndx];
                }
            }
        }
    }
    
    high_prec_float mu_weak_minus = munewweaks_minus[6];
    
    while ((muplusEWSB) && (muplusNoCCB) && (abs(munewweaks_plus[6]) > 25.0) && ((munew_mZ2plus > (45.5938 * 45.5938)) && (munew_mZ2plus < (364.7504 * 364.7504)))) {
        vector<high_prec_float> checkweaksols = munewweaks_plus;
        vector<high_prec_float> checkRadCorrs = radcorr_calc(checkweaksols, exp(munewlogQSUSY), munew_mZ2plus);
        muplusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
        // Checking loop-level EWSB
        if (muplusEWSB == true) {
            muplusEWSB = Hessian_check(checkweaksols, exp(munewlogQSUSY));
        }
        muplusNoCCB = CCB_Check(checkweaksols);
        if (!(muplusEWSB) || !(muplusNoCCB)) {
            break;
        } 

        munew_mZ2plus = pow((sqrt(munew_mZ2plus) + 0.1), 2.0);
        std::cout << "New mZ+ = " << sqrt(munew_mZ2plus) << endl;
        std::cout << "New mu+ = " << munewweaks_plus[6] << endl;
        std::cout << "New tanb+ = " << munewweaks_plus[43] << endl;
        high_prec_float mubefore = munewweaks_plus[6];
        try {
            for (int i = 0; i < max_iter; ++i) {
                vector<high_prec_float> current_derivatives = single_var_deriv_approxes(munewweaks_plus, munew_mZ2plus, 6, munewlogQSUSY);
                
                high_prec_float det = (current_derivatives[3] * current_derivatives[1]) - (current_derivatives[2] * current_derivatives[0]);
                high_prec_float Feval = calculate_approx_mZ2(munewweaks_plus, exp(current_logQSUSY), munew_mZ2plus);
                high_prec_float Geval = calculate_approx_tanb(munewweaks_plus, exp(current_logQSUSY), munew_mZ2plus);
                // high_prec_float alphaterm = (current_derivatives[2] * Geval) - (current_derivatives[1] * Feval);
                // high_prec_float betaterm = (current_derivatives[0] * Feval) - (current_derivatives[3] * Geval);
                // high_prec_float gammaterm = (current_derivatives[7] * current_derivatives[3]) - (current_derivatives[5] * current_derivatives[0]);
                // high_prec_float epsilonterm = (current_derivatives[5] * current_derivatives[1]) - (current_derivatives[7] * current_derivatives[2]);
                // high_prec_float lambdaterm = (current_derivatives[4] * current_derivatives[1]) - (current_derivatives[6] * current_derivatives[2]);
                // high_prec_float omegaterm = (current_derivatives[3] * current_derivatives[6]) - (current_derivatives[0] * current_derivatives[4]);
                high_prec_float dmu = ((current_derivatives[1] * Feval) - (current_derivatives[2] * Geval)) / det;
                high_prec_float dtanb = ((current_derivatives[3] * Geval) - (current_derivatives[0] * Feval)) / det;
                // high_prec_float dmu = (high_prec_float(2.0) * det * alphaterm * alphaterm) / ((alphaterm * alphaterm * epsilonterm) + (betaterm * betaterm * lambdaterm) + (high_prec_float(2.0) * det * det * alphaterm));
                // high_prec_float dtanb = (high_prec_float(2.0) * det * betaterm * betaterm) / ((alphaterm * alphaterm * gammaterm) + (betaterm * betaterm * omegaterm) + (high_prec_float(2.0) * det * det * betaterm));
                vector<high_prec_float> oldweaks = munewweaks_plus;
                munewweaks_plus[6] -= dmu;
                // munewweaks_plus[6] += dmu;
                // std::cout << "dmu = " << dmu << std::endl;
                munewweaks_plus[42] *= munewweaks_plus[6] / oldweaks[6];
                munewweaks_plus[43] -= dtanb;
                // munewweaks_plus[43] += dtanb;
                // std::cout << "dtanb = " << dtanb << std::endl;
                for (int YukIndx = 7; YukIndx < 16; ++YukIndx) {
                    if ((YukIndx >=7) && (YukIndx < 10)) {
                        munewweaks_plus[YukIndx] *= sin(atan(oldweaks[43])) / sin(atan(munewweaks_plus[43]));
                        munewweaks_plus[YukIndx+9] *= munewweaks_plus[YukIndx] / oldweaks[YukIndx];
                    } else {
                        munewweaks_plus[YukIndx] *= cos(atan(oldweaks[43])) / cos(atan(munewweaks_plus[43]));
                        munewweaks_plus[YukIndx+9] *= munewweaks_plus[YukIndx] / oldweaks[YukIndx];
                    }
                }
                // std::cout << "current L2: " << sqrt((Feval * Feval) + (Geval * Geval)) << endl;
                if (sqrt((Feval * Feval) + (Geval * Geval)) < tol) {
                    break;
                }
                if ((i == 99) && (sqrt((Feval * Feval) + (Geval * Geval)) > tol)) {
                    muplusEWSB = false;
                    munewweaks_plus[6] = checkweaksols[6];
                }
            } 
        } catch (...) {
            muplusEWSB = false;
        }
        vector<high_prec_float> oldweak = munewweaks_plus;
        if (signum(mubefore) != signum(munewweaks_plus[6])) {
            muplusEWSB = false;
            munewweaks_plus[6] = mubefore;
        }
        if ((munewweaks_plus[43] < 3.0) || (munewweaks_plus[43] > 60.0)) {
            muplusEWSB = false;
        } else {
            for (int YukIndx = 7; YukIndx < 16; ++YukIndx) {
                if ((YukIndx >=7) && (YukIndx < 10)) {
                    munewweaks_plus[YukIndx] *= sin(atan(oldweak[43])) / sin(atan(munewweaks_plus[43]));
                    munewweaks_plus[YukIndx+9] *= munewweaks_plus[YukIndx] / oldweak[YukIndx];
                } else {
                    munewweaks_plus[YukIndx] *= cos(atan(oldweak[43])) / cos(atan(munewweaks_plus[43]));
                    munewweaks_plus[YukIndx+9] *= munewweaks_plus[YukIndx] / oldweak[YukIndx];
                }
            }
        }
    }
    
    high_prec_float mu_weak_plus = munewweaks_plus[6];
    
    std::cout << "ABDS window established for mu variation." << endl;

    std::cout << "mu(weak, -) = " << mu_weak_minus << endl << "mu(weak, +) = " << mu_weak_plus << endl;
    return {mu_weak_minus, mu_weak_plus};//, mu_TOTAL_weak_minus, mu_TOTAL_weak_plus};
}

high_prec_float Nsoft_term_calc(high_prec_float nPower, std::vector<high_prec_float> softvec) {
    high_prec_float Ncontrib = high_prec_float(0.0);
    high_prec_float MPlanck = (high_prec_float(1.0) / sqrt(high_prec_float(8.0) * high_prec_float(M_PI))) * high_prec_float(1.22089e19);
    high_prec_float prefactor = high_prec_float(1.0) / ((nPower + high_prec_float(1.0)) * pow(MPlanck, (nPower + high_prec_float(1.0))));
    for (const auto& value : softvec) {
        Ncontrib += pow(value, high_prec_float(2.0));
    }
    Ncontrib = pow(Ncontrib, (nPower + high_prec_float(1.0)));
    Ncontrib *= prefactor;
    return Ncontrib;
}

std::vector<DSNLabeledValue> DSN_calc(int precselno, std::vector<high_prec_float> Wk_boundary_conditions,
                                      high_prec_float& current_mZ2, high_prec_float& current_logQSUSY,
                                      high_prec_float& current_logQGUT, int& nF, int& nD) {
    high_prec_float Nmu, NmHu, NmHd, NM1, NM2, NM3, NmQ1, NmQ2, NmQ3, NmL1, NmL2, NmL3, NmU1, NmU2, NmU3, NmD1, NmD2, NmD3, NmE1, NmE2, NmE3, Nat, Nac, Nau, Nab, Nas, Nad, Natau, Namu, Nae, NB;
    vector<DSNLabeledValue> DSNlabeledlist, unsortedDSNlabeledlist;
    std::cout << "This may take a while...\n\nProgress:\n-----------------------------------------------\n" << endl;
    if ((precselno == 1)) {
        vector<high_prec_float> minussofts, plussofts;
        // Compute mu windows around original point
        vector<high_prec_float> muinitwkBCs = Wk_boundary_conditions;
        vector<high_prec_float> muwindows = DSN_mu_windows(muinitwkBCs, current_mZ2, current_logQSUSY);
        std::cout << "muwindows: [" << muwindows[0] << ", " << muwindows[1] << "]" << endl;
        Nmu = abs(log10(abs(muwindows[1] / muwindows[0])));

        // Now do same thing with mHu^2(GUT)
        vector<high_prec_float> mHu2initwkBCs = Wk_boundary_conditions;
        vector<high_prec_float> mHu2windows;
        if (abs(boost::math::float_next(Wk_boundary_conditions[25]) - Wk_boundary_conditions[25]) >= 1.0) {
            mHu2windows = {pow(Wk_boundary_conditions[25], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[25], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } else {
            mHu2windows = DSN_specific_windows(mHu2initwkBCs, current_mZ2, current_logQSUSY, 25);
        }
        for (const auto& value : mHu2windows) {
            if (isnan(value)) {
                mHu2windows = {pow(Wk_boundary_conditions[25], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[25], high_prec_float((2.0 * nF) + (1.0 * nD)))};
            }
        }
        if (abs(mHu2windows[0] - mHu2windows[1]) < (high_prec_float(1.0e-20))) {
            mHu2windows = {pow(Wk_boundary_conditions[25], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[25], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } 
        minussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                      Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                      sqrt(abs(mHu2windows[0])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                      sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                      sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        plussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                     Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                     sqrt(abs(mHu2windows[1])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                     sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                     sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        if (signum(mHu2windows[0]) != signum(mHu2windows[1])) {
            vector<high_prec_float> tempsofts = minussofts;
            tempsofts[12] = high_prec_float(0.0);
            NmHu = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), tempsofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)))\
                + abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), tempsofts)));
        } else {
            NmHu = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)));
        }

        // Now do same thing with mHd^2(GUT)
        vector<high_prec_float> mHd2initwkBCs = Wk_boundary_conditions;
        vector<high_prec_float> mHd2windows;
        if (abs(boost::math::float_next(Wk_boundary_conditions[26]) - Wk_boundary_conditions[26]) >= 1.0) {
            mHd2windows = {pow(Wk_boundary_conditions[26], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[26], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } else {
            mHd2windows = DSN_specific_windows(mHd2initwkBCs, current_mZ2, current_logQSUSY, 26);
        }
        for (const auto& value : mHd2windows) {
            if (isnan(value)) {
                mHd2windows = {pow(Wk_boundary_conditions[26], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[26], high_prec_float((2.0 * nF) + (1.0 * nD)))};
            }
        }
        if (abs(mHu2windows[0] - mHu2windows[1]) < (high_prec_float(1.0e-20))) {
            mHd2windows = {pow(Wk_boundary_conditions[26], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[26], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } 
        minussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                      Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                      sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(mHd2windows[0])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                      sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                      sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        plussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                     Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                     sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(mHd2windows[1])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                     sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                     sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        if (signum(mHd2windows[0]) != signum(mHd2windows[1])) {
            vector<high_prec_float> tempsofts = minussofts;
            tempsofts[13] = high_prec_float(0.0);
            NmHd = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), tempsofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)))\
                + abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), tempsofts)));
        } else {
            NmHd = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)));
        }

        // Now do same thing with M1
        vector<high_prec_float> M1initwkBCs = Wk_boundary_conditions;
        vector<high_prec_float> M1windows;
        if (abs(boost::math::float_next(Wk_boundary_conditions[3]) - Wk_boundary_conditions[3]) >= 1.0) {
            M1windows = {pow(Wk_boundary_conditions[3], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[3], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } else {
            M1windows = DSN_specific_windows(M1initwkBCs, current_mZ2, current_logQSUSY, 3);
        }
        for (const auto& value : M1windows) {
            if (isnan(value)) {
                M1windows = {pow(Wk_boundary_conditions[3], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[3], high_prec_float((2.0 * nF) + (1.0 * nD)))};
            }
        }
        if (abs(M1windows[0] - M1windows[1]) < (high_prec_float(1.0e-20))) {
            M1windows = {pow(Wk_boundary_conditions[3], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[3], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } 
        minussofts = {M1windows[0], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                      Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                      sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                      sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                      sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        plussofts = {M1windows[1], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                     Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                     sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                     sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                     sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        NM1 = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)));

        // Now do same thing with M2
        vector<high_prec_float> M2initwkBCs = Wk_boundary_conditions;
        vector<high_prec_float> M2windows;
        if (abs(boost::math::float_next(Wk_boundary_conditions[4]) - Wk_boundary_conditions[4]) >= 1.0) {
            M2windows = {pow(Wk_boundary_conditions[4], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[4], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } else {
            M2windows = DSN_specific_windows(M2initwkBCs, current_mZ2, current_logQSUSY, 4);
        }
        for (const auto& value : M2windows) {
            if (isnan(value)) {
                M2windows = {pow(Wk_boundary_conditions[4], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[4], high_prec_float((2.0 * nF) + (1.0 * nD)))};
            }
        }
        if (abs(M2windows[0] - M2windows[1]) < (high_prec_float(1.0e-20))) {
            M2windows = {pow(Wk_boundary_conditions[4], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[4], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } 
        minussofts = {Wk_boundary_conditions[3], M2windows[0], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                      Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                      sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                      sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                      sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        plussofts = {Wk_boundary_conditions[3], M2windows[1], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                     Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                     sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                     sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                     sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        NM2 = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)));

        // Now do same thing with M3
        vector<high_prec_float> M3initwkBCs = Wk_boundary_conditions;
        vector<high_prec_float> M3windows;
        if (abs(boost::math::float_next(Wk_boundary_conditions[5]) - Wk_boundary_conditions[5]) >= 1.0) {
            M3windows = {pow(Wk_boundary_conditions[5], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[5], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } else {
            M3windows = DSN_specific_windows(M3initwkBCs, current_mZ2, current_logQSUSY, 5);
        }
        for (const auto& value : M3windows) {
            if (isnan(value)) {
                M3windows = {pow(Wk_boundary_conditions[5], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[5], high_prec_float((2.0 * nF) + (1.0 * nD)))};
            }
        }
        if (abs(M3windows[0] - M3windows[1]) < (high_prec_float(1.0e-20))) {
            M3windows = {pow(Wk_boundary_conditions[5], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[5], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } 
        minussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], M3windows[0], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                      Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                      sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                      sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                      sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        plussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], M3windows[1], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                     Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                     sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                     sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                     sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        NM3 = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)));

        // Now do same thing with mQ3
        vector<high_prec_float> MQ3initwkBCs = Wk_boundary_conditions;
        vector<high_prec_float> MQ3windows;
        if (abs(boost::math::float_next(Wk_boundary_conditions[29]) - Wk_boundary_conditions[29]) >= 1.0) {
            MQ3windows = {pow(Wk_boundary_conditions[29], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[29], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } else {
            MQ3windows = DSN_specific_windows(MQ3initwkBCs, current_mZ2, current_logQSUSY, 29);
        }
        for (const auto& value : MQ3windows) {
            if (isnan(value)) {
                MQ3windows = {pow(Wk_boundary_conditions[29], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[29], high_prec_float((2.0 * nF) + (1.0 * nD)))};
            }
        }
        if (abs(MQ3windows[0] - MQ3windows[1]) < (high_prec_float(1.0e-20))) {
            MQ3windows = {pow(Wk_boundary_conditions[29], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[29], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } 
        minussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                      Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                      sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(MQ3windows[0])), sqrt(abs(Wk_boundary_conditions[30])),
                      sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                      sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        plussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                     Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                     sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(MQ3windows[1])), sqrt(abs(Wk_boundary_conditions[30])),
                     sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                     sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        NmQ3 = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)));

        // Now do same thing with mQ2
        vector<high_prec_float> MQ2initwkBCs = Wk_boundary_conditions;
        vector<high_prec_float> MQ2windows;
        if (abs(boost::math::float_next(Wk_boundary_conditions[28]) - Wk_boundary_conditions[28]) >= 1.0) {
            MQ2windows = {pow(Wk_boundary_conditions[28], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[28], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } else {
            MQ2windows = DSN_specific_windows(MQ2initwkBCs, current_mZ2, current_logQSUSY, 28);
        }
        for (const auto& value : MQ2windows) {
            if (isnan(value)) {
                MQ2windows = {pow(Wk_boundary_conditions[28], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[28], high_prec_float((2.0 * nF) + (1.0 * nD)))};
            }
        }
        if (abs(MQ2windows[0] - MQ2windows[1]) < (high_prec_float(1.0e-20))) {
            MQ2windows = {pow(Wk_boundary_conditions[28], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[28], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } 
        minussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                      Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                      sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(MQ2windows[0])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                      sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                      sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        plussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                     Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                     sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(MQ2windows[1])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                     sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                     sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        NmQ2 = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)));

        // Now do same thing with mQ1
        vector<high_prec_float> MQ1initwkBCs = Wk_boundary_conditions;
        vector<high_prec_float> MQ1windows;
        if (abs(boost::math::float_next(Wk_boundary_conditions[27]) - Wk_boundary_conditions[27]) >= 1.0) {
            MQ1windows = {pow(Wk_boundary_conditions[27], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[27], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } else {
            MQ1windows = DSN_specific_windows(MQ1initwkBCs, current_mZ2, current_logQSUSY, 27);
        }
        for (const auto& value : MQ1windows) {
            if (isnan(value)) {
                MQ1windows = {pow(Wk_boundary_conditions[27], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[27], high_prec_float((2.0 * nF) + (1.0 * nD)))};
            }
        }
        if (abs(MQ1windows[0] - MQ1windows[1]) < (high_prec_float(1.0e-20))) {
            MQ1windows = {pow(Wk_boundary_conditions[27], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[27], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } 
        minussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                      Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                      sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(MQ1windows[0])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                      sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                      sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        plussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                     Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                     sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(MQ1windows[1])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                     sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                     sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        NmQ1 = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)));

        // Now do same thing with mL3
        vector<high_prec_float> mL3initwkBCs = Wk_boundary_conditions;
        vector<high_prec_float> mL3windows;
        if (abs(boost::math::float_next(Wk_boundary_conditions[32]) - Wk_boundary_conditions[32]) >= 1.0) {
            mL3windows = {pow(Wk_boundary_conditions[32], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[32], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } else {
            mL3windows = DSN_specific_windows(mL3initwkBCs, current_mZ2, current_logQSUSY, 32);
        }
        for (const auto& value : mL3windows) {
            if (isnan(value)) {
                mL3windows = {pow(Wk_boundary_conditions[32], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[32], high_prec_float((2.0 * nF) + (1.0 * nD)))};
            }
        }
        if (abs(mL3windows[0] - mL3windows[1]) < (high_prec_float(1.0e-20))) {
            mL3windows = {pow(Wk_boundary_conditions[32], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[32], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } 
        minussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                      Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                      sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                      sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(mL3windows[0])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                      sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        plussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                     Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                     sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                     sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(mL3windows[1])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                     sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        NmL3 = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)));
        
        // Now do same thing with mL2
        vector<high_prec_float> mL2initwkBCs = Wk_boundary_conditions;
        vector<high_prec_float> mL2windows;
        if (abs(boost::math::float_next(Wk_boundary_conditions[31]) - Wk_boundary_conditions[31]) >= 1.0) {
            mL2windows = {pow(Wk_boundary_conditions[31], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[31], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } else {
            mL2windows = DSN_specific_windows(mL2initwkBCs, current_mZ2, current_logQSUSY, 31);
        }
        for (const auto& value : mL2windows) {
            if (isnan(value)) {
                mL2windows = {pow(Wk_boundary_conditions[31], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[31], high_prec_float((2.0 * nF) + (1.0 * nD)))};
            }
        }
        if (abs(mL2windows[0] - mL2windows[1]) < (high_prec_float(1.0e-20))) {
            mL2windows = {pow(Wk_boundary_conditions[31], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[31], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } 
        minussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                      Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                      sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                      sqrt(abs(mL2windows[0])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                      sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        plussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                     Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                     sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                     sqrt(abs(mL2windows[1])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                     sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        NmL2 = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)));
        
        // Now do same thing with mL1
        vector<high_prec_float> mL1initwkBCs = Wk_boundary_conditions;
        vector<high_prec_float> mL1windows;
        if (abs(boost::math::float_next(Wk_boundary_conditions[30]) - Wk_boundary_conditions[30]) >= 1.0) {
            mL1windows = {pow(Wk_boundary_conditions[30], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[30], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } else {
            mL1windows = DSN_specific_windows(mL1initwkBCs, current_mZ2, current_logQSUSY, 30);
        }
        for (const auto& value : mL1windows) {
            if (isnan(value)) {
                mL1windows = {pow(Wk_boundary_conditions[30], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[30], high_prec_float((2.0 * nF) + (1.0 * nD)))};
            }
        }
        if (abs(mL1windows[0] - mL1windows[1]) < (high_prec_float(1.0e-20))) {
            mL1windows = {pow(Wk_boundary_conditions[30], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[30], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } 
        minussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                      Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                      sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(mL1windows[0])),
                      sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                      sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        plussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                     Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                     sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(mL1windows[1])),
                     sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                     sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        NmL1 = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)));
        
        // Now do same thing with mU3
        vector<high_prec_float> mU3initwkBCs = Wk_boundary_conditions;
        vector<high_prec_float> mU3windows;
        if (abs(boost::math::float_next(Wk_boundary_conditions[35]) - Wk_boundary_conditions[35]) >= 1.0) {
            mU3windows = {pow(Wk_boundary_conditions[35], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[35], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } else {
            mU3windows = DSN_specific_windows(mU3initwkBCs, current_mZ2, current_logQSUSY, 35);
        }
        for (const auto& value : mU3windows) {
            if (isnan(value)) {
                mU3windows = {pow(Wk_boundary_conditions[35], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[35], high_prec_float((2.0 * nF) + (1.0 * nD)))};
            }
        }
        if (abs(mU3windows[0] - mU3windows[1]) < (high_prec_float(1.0e-20))) {
            mU3windows = {pow(Wk_boundary_conditions[35], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[35], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } 
        minussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                      Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                      sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                      sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(mU3windows[0])), sqrt(abs(Wk_boundary_conditions[36])),
                      sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        plussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                     Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                     sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                     sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(mU3windows[1])), sqrt(abs(Wk_boundary_conditions[36])),
                     sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        NmU3 = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)));
        
        // Now do same thing with mU2
        vector<high_prec_float> mU2initwkBCs = Wk_boundary_conditions;
        vector<high_prec_float> mU2windows;
        if (abs(boost::math::float_next(Wk_boundary_conditions[34]) - Wk_boundary_conditions[34]) >= 1.0) {
            mU2windows = {pow(Wk_boundary_conditions[34], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[34], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } else {
            mU2windows = DSN_specific_windows(mU2initwkBCs, current_mZ2, current_logQSUSY, 34);
        }
        for (const auto& value : mU2windows) {
            if (isnan(value)) {
                mU2windows = {pow(Wk_boundary_conditions[34], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[34], high_prec_float((2.0 * nF) + (1.0 * nD)))};
            }
        }
        if (abs(mU2windows[0] - mU2windows[1]) < (high_prec_float(1.0e-20))) {
            mU2windows = {pow(Wk_boundary_conditions[34], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[34], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } 
        minussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                      Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                      sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                      sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(mU2windows[0])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                      sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        plussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                     Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                     sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                     sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(mU2windows[1])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                     sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        NmU2 = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)));
        
        // Now do same thing with mU1
        vector<high_prec_float> mU1initwkBCs = Wk_boundary_conditions;
        vector<high_prec_float> mU1windows;
        if (abs(boost::math::float_next(Wk_boundary_conditions[33]) - Wk_boundary_conditions[33]) >= 1.0) {
            mU1windows = {pow(Wk_boundary_conditions[33], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[33], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } else {
            mU1windows = DSN_specific_windows(mU1initwkBCs, current_mZ2, current_logQSUSY, 33);
        }
        for (const auto& value : mU1windows) {
            if (isnan(value)) {
                mU1windows = {pow(Wk_boundary_conditions[33], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[33], high_prec_float((2.0 * nF) + (1.0 * nD)))};
            }
        }
        if (abs(mU1windows[0] - mU1windows[1]) < (high_prec_float(1.0e-20))) {
            mU1windows = {pow(Wk_boundary_conditions[33], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[33], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } 
        minussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                      Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                      sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                      sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(mU1windows[0])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                      sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        plussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                     Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                     sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                     sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(mU1windows[1])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                     sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        NmU1 = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)));
        
        // Now do same thing with mD3
        vector<high_prec_float> mD3initwkBCs = Wk_boundary_conditions;
        vector<high_prec_float> mD3windows;
        if (abs(boost::math::float_next(Wk_boundary_conditions[38]) - Wk_boundary_conditions[38]) >= 1.0) {
            mD3windows = {pow(Wk_boundary_conditions[38], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[38], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } else {
            mD3windows = DSN_specific_windows(mD3initwkBCs, current_mZ2, current_logQSUSY, 38);
        }
        for (const auto& value : mD3windows) {
            if (isnan(value)) {
                mD3windows = {pow(Wk_boundary_conditions[38], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[38], high_prec_float((2.0 * nF) + (1.0 * nD)))};
            }
        }
        if (abs(mD3windows[0] - mD3windows[1]) < (high_prec_float(1.0e-20))) {
            mD3windows = {pow(Wk_boundary_conditions[38], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[38], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } 
        minussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                      Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                      sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                      sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                      sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(mD3windows[0])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        plussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                     Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                     sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                     sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                     sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(mD3windows[1])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        NmD3 = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)));
        
        // Now do same thing with mD2
        vector<high_prec_float> mD2initwkBCs = Wk_boundary_conditions;
        vector<high_prec_float> mD2windows;
        if (abs(boost::math::float_next(Wk_boundary_conditions[37]) - Wk_boundary_conditions[37]) >= 1.0) {
            mD2windows = {pow(Wk_boundary_conditions[37], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[37], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } else {
            mD2windows = DSN_specific_windows(mD2initwkBCs, current_mZ2, current_logQSUSY, 37);
        }
        for (const auto& value : mD2windows) {
            if (isnan(value)) {
                mD2windows = {pow(Wk_boundary_conditions[37], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[37], high_prec_float((2.0 * nF) + (1.0 * nD)))};
            }
        }
        if (abs(mD2windows[0] - mD2windows[1]) < (high_prec_float(1.0e-20))) {
            mD2windows = {pow(Wk_boundary_conditions[37], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[37], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } 
        minussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                      Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                      sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                      sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                      sqrt(abs(mD2windows[0])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        plussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                     Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                     sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                     sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                     sqrt(abs(mD2windows[1])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        NmD2 = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)));
        
        // Now do same thing with mD1
        vector<high_prec_float> mD1initwkBCs = Wk_boundary_conditions;
        vector<high_prec_float> mD1windows;
        if (abs(boost::math::float_next(Wk_boundary_conditions[36]) - Wk_boundary_conditions[36]) >= 1.0) {
            mD1windows = {pow(Wk_boundary_conditions[36], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[36], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } else {
            mD1windows = DSN_specific_windows(mD1initwkBCs, current_mZ2, current_logQSUSY, 36);
        }
        for (const auto& value : mD1windows) {
            if (isnan(value)) {
                mD1windows = {pow(Wk_boundary_conditions[36], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[36], high_prec_float((2.0 * nF) + (1.0 * nD)))};
            }
        }
        if (abs(mD1windows[0] - mD1windows[1]) < (high_prec_float(1.0e-20))) {
            mD1windows = {pow(Wk_boundary_conditions[36], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[36], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } 
        minussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                      Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                      sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                      sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(mD1windows[0])),
                      sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        plussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                     Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                     sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                     sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(mD1windows[1])),
                     sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        NmD1 = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)));
        
        // Now do same thing with mE3
        vector<high_prec_float> mE3initwkBCs = Wk_boundary_conditions;
        vector<high_prec_float> mE3windows;
        if (abs(boost::math::float_next(Wk_boundary_conditions[41]) - Wk_boundary_conditions[41]) >= 1.0) {
            mE3windows = {pow(Wk_boundary_conditions[41], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[41], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } else {
            mE3windows = DSN_specific_windows(mE3initwkBCs, current_mZ2, current_logQSUSY, 41);
        }
        for (const auto& value : mE3windows) {
            if (isnan(value)) {
                mE3windows = {pow(Wk_boundary_conditions[41], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[41], high_prec_float((2.0 * nF) + (1.0 * nD)))};
            }
        }
        if (abs(mE3windows[0] - mE3windows[1]) < (high_prec_float(1.0e-20))) {
            mE3windows = {pow(Wk_boundary_conditions[41], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[41], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } 
        minussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                      Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                      sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                      sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                      sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(mE3windows[0])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        plussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                     Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                     sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                     sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                     sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(mE3windows[1])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        NmE3 = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)));
        
        // Now do same thing with mE2
        vector<high_prec_float> mE2initwkBCs = Wk_boundary_conditions;
        vector<high_prec_float> mE2windows;
        if (abs(boost::math::float_next(Wk_boundary_conditions[40]) - Wk_boundary_conditions[40]) >= 1.0) {
            mE2windows = {pow(Wk_boundary_conditions[40], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[40], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } else {
            mE2windows = DSN_specific_windows(mE2initwkBCs, current_mZ2, current_logQSUSY, 40);
        }
        for (const auto& value : mE2windows) {
            if (isnan(value)) {
                mE2windows = {pow(Wk_boundary_conditions[40], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[40], high_prec_float((2.0 * nF) + (1.0 * nD)))};
            }
        }
        if (abs(mE2windows[0] - mE2windows[1]) < (high_prec_float(1.0e-20))) {
            mE2windows = {pow(Wk_boundary_conditions[40], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[40], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } 
        minussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                      Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                      sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                      sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                      sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(mE2windows[0])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        plussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                     Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                     sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                     sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                     sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(mE2windows[1])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        NmE2 = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)));
        
        // Now do same thing with mE1
        vector<high_prec_float> mE1initwkBCs = Wk_boundary_conditions;
        vector<high_prec_float> mE1windows;
        if (abs(boost::math::float_next(Wk_boundary_conditions[39]) - Wk_boundary_conditions[39]) >= 1.0) {
            mE1windows = {pow(Wk_boundary_conditions[39], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[39], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } else {
            mE1windows = DSN_specific_windows(mE1initwkBCs, current_mZ2, current_logQSUSY, 39);
        }
        for (const auto& value : mE1windows) {
            if (isnan(value)) {
                mE1windows = {pow(Wk_boundary_conditions[39], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[39], high_prec_float((2.0 * nF) + (1.0 * nD)))};
            }
        }
        if (abs(mE1windows[0] - mE1windows[1]) < (high_prec_float(1.0e-20))) {
            mE1windows = {pow(Wk_boundary_conditions[39], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[39], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } 
        minussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                      Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                      sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                      sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                      sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(mE1windows[0])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        plussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                     Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                     sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                     sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                     sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(mE1windows[1])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        NmE1 = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)));
        
        // Now do same thing with at
        vector<high_prec_float> atinitwkBCs = Wk_boundary_conditions;
        vector<high_prec_float> atwindows;
        if (abs(boost::math::float_next(Wk_boundary_conditions[16]) - Wk_boundary_conditions[16]) >= 1.0) {
            atwindows = {pow(Wk_boundary_conditions[16], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[16], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } else {
            atwindows = DSN_specific_windows(atinitwkBCs, current_mZ2, current_logQSUSY, 16);
        }
        for (const auto& value : atwindows) {
            if (isnan(value)) {
                atwindows = {pow(Wk_boundary_conditions[16], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[16], high_prec_float((2.0 * nF) + (1.0 * nD)))};
            }
        }
        if (abs(atwindows[0] - atwindows[1]) < (high_prec_float(1.0e-20))) {
            atwindows = {pow(Wk_boundary_conditions[16], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[16], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } 
        minussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], atwindows[0], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                      Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                      sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                      sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                      sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        plussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], atwindows[1], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                     Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                     sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                     sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                     sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        if (signum(atwindows[0]) != signum(atwindows[1])) {
            vector<high_prec_float> tempsofts = minussofts;
            tempsofts[16] = high_prec_float(0.0);
            Nat = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), tempsofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)))\
                + abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), tempsofts)));
        } else {
            Nat = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)));
        }
        
        // Now do same thing with ac
        vector<high_prec_float> acinitwkBCs = Wk_boundary_conditions;
        vector<high_prec_float> acwindows;
        if (abs(boost::math::float_next(Wk_boundary_conditions[17]) - Wk_boundary_conditions[17]) >= 1.0) {
            acwindows = {pow(Wk_boundary_conditions[17], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[17], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } else {
            acwindows = DSN_specific_windows(acinitwkBCs, current_mZ2, current_logQSUSY, 17);
        }
        for (const auto& value : acwindows) {
            if (isnan(value)) {
                acwindows = {pow(Wk_boundary_conditions[17], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[17], high_prec_float((2.0 * nF) + (1.0 * nD)))};
            }
        }
        if (abs(acwindows[0] - acwindows[1]) < (high_prec_float(1.0e-20))) {
            acwindows = {pow(Wk_boundary_conditions[17], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[17], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } 
        minussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], acwindows[0], Wk_boundary_conditions[18],
                      Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                      sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                      sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                      sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        plussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], acwindows[1], Wk_boundary_conditions[18],
                     Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                     sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                     sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                     sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        if (signum(acwindows[0]) != signum(acwindows[1])) {
            vector<high_prec_float> tempsofts = minussofts;
            tempsofts[17] = high_prec_float(0.0);
            Nac = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), tempsofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)))\
                + abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), tempsofts)));
        } else {
            Nac = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)));
        }
        
        // Now do same thing with au    
        vector<high_prec_float> auinitwkBCs = Wk_boundary_conditions;
        vector<high_prec_float> auwindows;
        if (abs(boost::math::float_next(Wk_boundary_conditions[18]) - Wk_boundary_conditions[18]) >= 1.0) {
            auwindows = {pow(Wk_boundary_conditions[18], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[18], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } else {
            auwindows = DSN_specific_windows(auinitwkBCs, current_mZ2, current_logQSUSY, 18);
        }
        for (const auto& value : auwindows) {
            if (isnan(value)) {
                auwindows = {pow(Wk_boundary_conditions[18], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[18], high_prec_float((2.0 * nF) + (1.0 * nD)))};
            }
        }
        if (abs(auwindows[0] - auwindows[1]) < (high_prec_float(1.0e-20))) {
            auwindows = {pow(Wk_boundary_conditions[18], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[18], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } 
        minussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], auwindows[0],
                      Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                      sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                      sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                      sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        plussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], auwindows[1],
                     Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                     sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                     sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                     sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        if (signum(auwindows[0]) != signum(auwindows[1])) {
            vector<high_prec_float> tempsofts = minussofts;
            tempsofts[18] = high_prec_float(0.0);
            Nau = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), tempsofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)))\
                + abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), tempsofts)));
        } else {
            Nau = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)));
        }
        
        // Now do same thing with ab
        vector<high_prec_float> abinitwkBCs = Wk_boundary_conditions;
        vector<high_prec_float> abwindows;
        if (abs(boost::math::float_next(Wk_boundary_conditions[19]) - Wk_boundary_conditions[19]) >= 1.0) {
            abwindows = {pow(Wk_boundary_conditions[19], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[19], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } else {
            abwindows = DSN_specific_windows(abinitwkBCs, current_mZ2, current_logQSUSY, 19);
        }
        for (const auto& value : abwindows) {
            if (isnan(value)) {
                abwindows = {pow(Wk_boundary_conditions[19], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[19], high_prec_float((2.0 * nF) + (1.0 * nD)))};
            }
        }
        if (abs(abwindows[0] - abwindows[1]) < (high_prec_float(1.0e-20))) {
            abwindows = {pow(Wk_boundary_conditions[19], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[19], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } 
        minussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                      abwindows[0], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                      sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                      sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                      sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        plussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                     abwindows[1], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                     sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                     sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                     sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        if (signum(abwindows[0]) != signum(abwindows[1])) {
            vector<high_prec_float> tempsofts = minussofts;
            tempsofts[19] = high_prec_float(0.0);
            Nab = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), tempsofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)))\
                + abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), tempsofts)));
        } else {
            Nab = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)));
        }
        
        // Now do same thing with as
        vector<high_prec_float> asinitwkBCs = Wk_boundary_conditions;
        vector<high_prec_float> aswindows;
        if (abs(boost::math::float_next(Wk_boundary_conditions[20]) - Wk_boundary_conditions[20]) >= 1.0) {
            aswindows = {pow(Wk_boundary_conditions[20], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[20], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } else {
            aswindows = DSN_specific_windows(asinitwkBCs, current_mZ2, current_logQSUSY, 20);
        }
        for (const auto& value : aswindows) {
            if (isnan(value)) {
                aswindows = {pow(Wk_boundary_conditions[20], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[20], high_prec_float((2.0 * nF) + (1.0 * nD)))};
            }
        }
        if (abs(aswindows[0] - aswindows[1]) < (high_prec_float(1.0e-20))) {
            aswindows = {pow(Wk_boundary_conditions[20], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[20], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } 
        minussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                      Wk_boundary_conditions[19], aswindows[0], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                      sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                      sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                      sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        plussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                     Wk_boundary_conditions[19], aswindows[1], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                     sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                     sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                     sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        if (signum(aswindows[0]) != signum(aswindows[1])) {
            vector<high_prec_float> tempsofts = minussofts;
            tempsofts[20] = high_prec_float(0.0);
            Nas = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), tempsofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)))\
                + abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), tempsofts)));
        } else {
            Nas = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)));
        }
        
        // Now do same thing with ad    
        vector<high_prec_float> adinitwkBCs = Wk_boundary_conditions;
        vector<high_prec_float> adwindows;
        if (abs(boost::math::float_next(Wk_boundary_conditions[21]) - Wk_boundary_conditions[21]) >= 1.0) {
            adwindows = {pow(Wk_boundary_conditions[21], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[21], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } else {
            adwindows = DSN_specific_windows(adinitwkBCs, current_mZ2, current_logQSUSY, 21);
        }
        for (const auto& value : adwindows) {
            if (isnan(value)) {
                adwindows = {pow(Wk_boundary_conditions[21], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[21], high_prec_float((2.0 * nF) + (1.0 * nD)))};
            }
        }
        if (abs(adwindows[0] - adwindows[1]) < (high_prec_float(1.0e-20))) {
            adwindows = {pow(Wk_boundary_conditions[21], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[21], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } 
        minussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                      Wk_boundary_conditions[19], Wk_boundary_conditions[20], adwindows[0], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                      sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                      sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                      sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        plussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                     Wk_boundary_conditions[19], Wk_boundary_conditions[20], adwindows[1], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                     sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                     sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                     sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        if (signum(adwindows[0]) != signum(adwindows[1])) {
            vector<high_prec_float> tempsofts = minussofts;
            tempsofts[21] = high_prec_float(0.0);
            Nad = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), tempsofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)))\
                + abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), tempsofts)));
        } else {
            Nad = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)));
        }
        
        // Now do same thing with atau
        vector<high_prec_float> atauinitwkBCs = Wk_boundary_conditions;
        vector<high_prec_float> atauwindows;
        if (abs(boost::math::float_next(Wk_boundary_conditions[22]) - Wk_boundary_conditions[22]) >= 1.0) {
            atauwindows = {pow(Wk_boundary_conditions[22], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[22], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } else {
            atauwindows = DSN_specific_windows(atauinitwkBCs, current_mZ2, current_logQSUSY, 22);
        }
        for (const auto& value : atauwindows) {
            if (isnan(value)) {
                atauwindows = {pow(Wk_boundary_conditions[22], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[22], high_prec_float((2.0 * nF) + (1.0 * nD)))};
            }
        }
        if (abs(atauwindows[0] - atauwindows[1]) < (high_prec_float(1.0e-20))) {
            atauwindows = {pow(Wk_boundary_conditions[22], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[22], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } 
        minussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                      Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], atauwindows[0], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                      sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                      sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                      sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        plussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                     Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], atauwindows[1], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                     sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                     sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                     sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        if (signum(atauwindows[0]) != signum(atauwindows[1])) {
            vector<high_prec_float> tempsofts = minussofts;
            tempsofts[22] = high_prec_float(0.0);
            Natau = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), tempsofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)))\
                + abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), tempsofts)));
        } else {
            Natau = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)));
        }
        
        // Now do same thing with amu
        vector<high_prec_float> amuinitwkBCs = Wk_boundary_conditions;
        vector<high_prec_float> amuwindows;
        if (abs(boost::math::float_next(Wk_boundary_conditions[23]) - Wk_boundary_conditions[23]) >= 1.0) {
            amuwindows = {pow(Wk_boundary_conditions[23], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[23], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } else {
            amuwindows = DSN_specific_windows(amuinitwkBCs, current_mZ2, current_logQSUSY, 23);
        }
        for (const auto& value : amuwindows) {
            if (isnan(value)) {
                amuwindows = {pow(Wk_boundary_conditions[23], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[23], high_prec_float((2.0 * nF) + (1.0 * nD)))};
            }
        }
        if (abs(amuwindows[0] - amuwindows[1]) < (high_prec_float(1.0e-20))) {
            amuwindows = {pow(Wk_boundary_conditions[23], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[23], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } 
        minussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                      Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], amuwindows[0], Wk_boundary_conditions[24],
                      sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                      sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                      sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        plussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                     Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], amuwindows[1], Wk_boundary_conditions[24],
                     sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                     sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                     sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        if (signum(amuwindows[0]) != signum(amuwindows[1])) {
            vector<high_prec_float> tempsofts = minussofts;
            tempsofts[23] = high_prec_float(0.0);
            Namu = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), tempsofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)))\
                + abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), tempsofts)));
        } else {
            Namu = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)));
        }
        
        // Now do same thing with ae    
        vector<high_prec_float> aeinitwkBCs = Wk_boundary_conditions;
        vector<high_prec_float> aewindows;
        if (abs(boost::math::float_next(Wk_boundary_conditions[24]) - Wk_boundary_conditions[24]) >= 1.0) {
            aewindows = {pow(Wk_boundary_conditions[24], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[24], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } else {
            aewindows = DSN_specific_windows(aeinitwkBCs, current_mZ2, current_logQSUSY, 24);
        }
        for (const auto& value : aewindows) {
            if (isnan(value)) {
                aewindows = {pow(Wk_boundary_conditions[24], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[24], high_prec_float((2.0 * nF) + (1.0 * nD)))};
            }
        }
        if (abs(aewindows[0] - aewindows[1]) < (high_prec_float(1.0e-20))) {
            aewindows = {pow(Wk_boundary_conditions[24], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[24], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } 
        minussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                      Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], aewindows[0],
                      sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                      sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                      sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        plussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                     Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], aewindows[1],
                     sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                     sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                     sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Wk_boundary_conditions[42] / Wk_boundary_conditions[6]};
        if (signum(aewindows[0]) != signum(aewindows[1])) {
            vector<high_prec_float> tempsofts = minussofts;
            tempsofts[24] = high_prec_float(0.0);
            Nae = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), tempsofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)))\
                + abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), tempsofts)));
        } else {
            Nae = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)));
        }
        
        // Now do same thing with B = b/mu;
        vector<high_prec_float> BinitwkBCs = Wk_boundary_conditions;
        vector<high_prec_float> Bwindows;
        if (abs(boost::math::float_next(Wk_boundary_conditions[42] / Wk_boundary_conditions[6]) - Wk_boundary_conditions[42] / Wk_boundary_conditions[6]) >= 1.0) {
            Bwindows = {pow(Wk_boundary_conditions[42] / Wk_boundary_conditions[6], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[42] / Wk_boundary_conditions[6], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } else {
            Bwindows = DSN_B_windows(BinitwkBCs, current_mZ2, current_logQSUSY);
        }
        for (const auto& value : Bwindows) {
            if (isnan(value)) {
                Bwindows = {pow(Wk_boundary_conditions[42] / Wk_boundary_conditions[6], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[42] / Wk_boundary_conditions[6], high_prec_float((2.0 * nF) + (1.0 * nD)))};
            }
        }
        if (abs(Bwindows[0] - Bwindows[1]) < (high_prec_float(1.0e-20))) {
            Bwindows = {pow(Wk_boundary_conditions[42] / Wk_boundary_conditions[6], high_prec_float(-1.0) * (high_prec_float((2.0 * nF) + (1.0 * nD)))), pow(Wk_boundary_conditions[42] / Wk_boundary_conditions[6], high_prec_float((2.0 * nF) + (1.0 * nD)))};
        } 
        minussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                      Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                      sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                      sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                      sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Bwindows[0]};
        plussofts = {Wk_boundary_conditions[3], Wk_boundary_conditions[4], Wk_boundary_conditions[5], Wk_boundary_conditions[16], Wk_boundary_conditions[17], Wk_boundary_conditions[18],
                     Wk_boundary_conditions[19], Wk_boundary_conditions[20], Wk_boundary_conditions[21], Wk_boundary_conditions[22], Wk_boundary_conditions[23], Wk_boundary_conditions[24],
                     sqrt(abs(Wk_boundary_conditions[25])), sqrt(abs(Wk_boundary_conditions[26])), sqrt(abs(Wk_boundary_conditions[27])), sqrt(abs(Wk_boundary_conditions[28])), sqrt(abs(Wk_boundary_conditions[29])), sqrt(abs(Wk_boundary_conditions[30])),
                     sqrt(abs(Wk_boundary_conditions[31])), sqrt(abs(Wk_boundary_conditions[32])), sqrt(abs(Wk_boundary_conditions[33])), sqrt(abs(Wk_boundary_conditions[34])), sqrt(abs(Wk_boundary_conditions[35])), sqrt(abs(Wk_boundary_conditions[36])),
                     sqrt(abs(Wk_boundary_conditions[37])), sqrt(abs(Wk_boundary_conditions[38])), sqrt(abs(Wk_boundary_conditions[39])), sqrt(abs(Wk_boundary_conditions[40])), sqrt(abs(Wk_boundary_conditions[41])), Bwindows[1]};
        if (signum(Bwindows[0]) != signum(Bwindows[1])) {
            vector<high_prec_float> tempsofts = minussofts;
            tempsofts[42] = high_prec_float(0.0);
            NB = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), tempsofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)))\
                + abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), tempsofts)));
        } else {
            NB = abs((Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), plussofts) - Nsoft_term_calc(high_prec_float((2.0 * nF) + (1.0 * nD) - 1.0), minussofts)));
        }
        
        // Create return list
        unsortedDSNlabeledlist = {{Nmu, "mu"},
                                  {NmHu, "mHu"},
                                  {NmHd, "mHd"},
                                  {NM1, "M1"},
                                  {NM2, "M2"},
                                  {NM3, "M3"},
                                  {NmQ3, "mQ3"},
                                  {NmQ2, "mQ2"},
                                  {NmQ1, "mQ1"},
                                  {NmL3, "mL3"},
                                  {NmL2, "mL2"},
                                  {NmL1, "mL1"},
                                  {NmU3, "mU3"},
                                  {NmU2, "mU2"},
                                  {NmU1, "mU1"},
                                  {NmD3, "mD3"},
                                  {NmD2, "mD2"},
                                  {NmD1, "mD1"},
                                  {NmE3, "mE3"},
                                  {NmE2, "mE2"},
                                  {NmE1, "mE1"},
                                  {Nat, "a_t"},
                                  {Nac, "a_c"},
                                  {Nau, "a_u"},
                                  {Nab, "a_b"},
                                  {Nas, "a_s"},
                                  {Nad, "a_d"},
                                  {Natau, "a_tau"},
                                  {Namu, "a_mu"},
                                  {Nae, "a_e"},
                                  {NB, "B"}};
        DSNlabeledlist = sortAndReturnDSN(unsortedDSNlabeledlist);
    } else {
        // Compute mu windows around original point
        vector<high_prec_float> muinitwkBCs = Wk_boundary_conditions;
        vector<high_prec_float> muwindows = DSN_mu_windows(muinitwkBCs, current_mZ2, current_logQSUSY);
        std::cout << "muwindows: [" << muwindows[0] << ", " << muwindows[1] << "]" << endl;
        Nmu = abs(log10(abs(muwindows[1] / muwindows[0])));

        // Create return list
        DSNlabeledlist = {{Nmu, "mu"}};
    }    

    return DSNlabeledlist;
}
