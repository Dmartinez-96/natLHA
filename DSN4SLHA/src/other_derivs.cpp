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
#include "radcorr_calc.hpp"
#include "other_derivs.hpp"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace boost::multiprecision;
typedef number<mpfr_float_backend<50>> high_prec_float;

high_prec_float first_derivative(high_prec_float hStep, high_prec_float pm4h, high_prec_float pm3h, high_prec_float pm2h, high_prec_float pmh, high_prec_float pph, high_prec_float pp2h, high_prec_float pp3h, high_prec_float pp4h) {
    return (high_prec_float(1.0) / hStep) * ((pm4h / high_prec_float(280.0)) - (high_prec_float(4.0) * pm3h / high_prec_float(105.0)) + (pm2h / high_prec_float(5.0)) - (high_prec_float(4.0) * pmh / high_prec_float(5.0))
                                 + (high_prec_float(4.0) * pph / high_prec_float(5.0)) - (pp2h / high_prec_float(5.0)) + (high_prec_float(4.0) * pp3h / high_prec_float(105.0)) - (pp4h / high_prec_float(280.0)));
}

high_prec_float second_derivative(high_prec_float hStep, high_prec_float pStart, high_prec_float pm4h, high_prec_float pm3h, high_prec_float pm2h, high_prec_float pmh, high_prec_float pph, high_prec_float pp2h, high_prec_float pp3h, high_prec_float pp4h) {
    return (high_prec_float(1.0) / (hStep * hStep)) * ((high_prec_float(-1.0) * pm4h / high_prec_float(560.0)) + (high_prec_float(8.0) * pm3h / high_prec_float(315.0)) - (pm2h / high_prec_float(5.0)) + (high_prec_float(8.0) * pmh / high_prec_float(5.0)) - (high_prec_float(205.0) * pStart / high_prec_float(72.0))
                                           + (high_prec_float(8.0) * pph / high_prec_float(5.0)) - (pp2h / high_prec_float(5.0)) + (high_prec_float(8.0) * pp3h / high_prec_float(315.0)) - (pp4h / high_prec_float(560.0)));
}

high_prec_float third_derivative(high_prec_float hStep, high_prec_float pm4h, high_prec_float pm3h, high_prec_float pm2h, high_prec_float pmh, high_prec_float pph, high_prec_float pp2h, high_prec_float pp3h, high_prec_float pp4h) {
    return (high_prec_float(1.0) / (hStep * hStep * hStep)) * ((high_prec_float(-7.0) * pm4h / high_prec_float(240.0)) + (high_prec_float(3.0) * pm3h / high_prec_float(10.0)) - (high_prec_float(169.0) * pm2h / high_prec_float(120.0) / high_prec_float(5.0)) + (high_prec_float(61.0) * pmh / high_prec_float(30.0))
                                                   - (high_prec_float(61.0) * pph / high_prec_float(30.0)) + (high_prec_float(169.0) * pp2h / high_prec_float(120.0)) - (high_prec_float(3.0) * pp3h / high_prec_float(10.0)) + (high_prec_float(7.0) * pp4h / high_prec_float(240.0)));
}

high_prec_float calculate_mZ2_eq(std::vector<high_prec_float> weak_solutions, high_prec_float explogQSUSY, high_prec_float mZ2Value) {
    std::vector<high_prec_float> calculateRadCorrs = radcorr_calc(weak_solutions, explogQSUSY, mZ2Value);
    return high_prec_float(mZ2Value) - ((high_prec_float(2.0)) * ((((weak_solutions[26] + calculateRadCorrs[1] - ((weak_solutions[25] + calculateRadCorrs[0]) * weak_solutions[43] * weak_solutions[43]))) / ((weak_solutions[43] * weak_solutions[43]) - 1.0)) - (weak_solutions[6] * weak_solutions[6])));
}

high_prec_float calculate_tanb_eq(std::vector<high_prec_float> weak_solutions, high_prec_float explogQSUSY, high_prec_float mZ2Value) {
    std::vector<high_prec_float> calculateRadCorrs = radcorr_calc(weak_solutions, explogQSUSY, mZ2Value);
    return weak_solutions[43] - tan(high_prec_float(0.5) * (high_prec_float(M_PI) - asin(abs(2.0 * weak_solutions[42] / (weak_solutions[25] + weak_solutions[26] + calculateRadCorrs[0] + calculateRadCorrs[1] + (2.0 * weak_solutions[6] * weak_solutions[6]))))));
}

std::vector<high_prec_float> mZ2_derivatives(std::vector<high_prec_float>& original_weak_conditions, high_prec_float& orig_mZ2_val, high_prec_float& logQSUSYval) {
    high_prec_float h_m = pow(((high_prec_float(2625.0) / high_prec_float(16.0)) * (boost::math::float_next(sqrt(orig_mZ2_val)) - sqrt(orig_mZ2_val))), (high_prec_float(1.0) / high_prec_float(9.0)));
    high_prec_float m_p = sqrt(orig_mZ2_val) + h_m;
    high_prec_float m_pp = m_p + h_m;
    high_prec_float m_ppp = m_pp + h_m;
    high_prec_float m_pppp = m_ppp + h_m;
    high_prec_float m_m = sqrt(orig_mZ2_val) - h_m;
    high_prec_float m_mm = m_m - h_m;
    high_prec_float m_mmm = m_mm - h_m;
    high_prec_float m_mmmm = m_mmm - h_m;

    high_prec_float m2_p = pow(m_p, high_prec_float(2.0));
    high_prec_float m2_p2 = pow(m_pp, high_prec_float(2.0));
    high_prec_float m2_p3 = pow(m_ppp, high_prec_float(2.0));
    high_prec_float m2_p4 = pow(m_pppp, high_prec_float(2.0));
    high_prec_float m2_m = pow(m_m, high_prec_float(2.0));
    high_prec_float m2_m2 = pow(m_mm, high_prec_float(2.0));
    high_prec_float m2_m3 = pow(m_mmm, high_prec_float(2.0));
    high_prec_float m2_m4 = pow(m_mmmm, high_prec_float(2.0));

    // high_prec_float f_orig = calculate_mZ2_eq(original_weak_conditions, exp(logQSUSYval), orig_mZ2_val);

    high_prec_float f_mp = calculate_mZ2_eq(original_weak_conditions, exp(logQSUSYval), m2_p);
    high_prec_float f_mp2 = calculate_mZ2_eq(original_weak_conditions, exp(logQSUSYval), m2_p2);
    high_prec_float f_mp3 = calculate_mZ2_eq(original_weak_conditions, exp(logQSUSYval), m2_p3);
    high_prec_float f_mp4 = calculate_mZ2_eq(original_weak_conditions, exp(logQSUSYval), m2_p4);
    high_prec_float f_mm = calculate_mZ2_eq(original_weak_conditions, exp(logQSUSYval), m2_m);
    high_prec_float f_mm2 = calculate_mZ2_eq(original_weak_conditions, exp(logQSUSYval), m2_m2);
    high_prec_float f_mm3 = calculate_mZ2_eq(original_weak_conditions, exp(logQSUSYval), m2_m3);
    high_prec_float f_mm4 = calculate_mZ2_eq(original_weak_conditions, exp(logQSUSYval), m2_m4);
    
    // high_prec_float g_orig = calculate_tanb_eq(original_weak_conditions, exp(logQSUSYval), orig_mZ2_val);

    high_prec_float g_mp = calculate_tanb_eq(original_weak_conditions, exp(logQSUSYval), m2_p);
    high_prec_float g_mp2 = calculate_tanb_eq(original_weak_conditions, exp(logQSUSYval), m2_p2);
    high_prec_float g_mp3 = calculate_tanb_eq(original_weak_conditions, exp(logQSUSYval), m2_p3);
    high_prec_float g_mp4 = calculate_tanb_eq(original_weak_conditions, exp(logQSUSYval), m2_p4);
    high_prec_float g_mm = calculate_tanb_eq(original_weak_conditions, exp(logQSUSYval), m2_m);
    high_prec_float g_mm2 = calculate_tanb_eq(original_weak_conditions, exp(logQSUSYval), m2_m2);
    high_prec_float g_mm3 = calculate_tanb_eq(original_weak_conditions, exp(logQSUSYval), m2_m3);
    high_prec_float g_mm4 = calculate_tanb_eq(original_weak_conditions, exp(logQSUSYval), m2_m4);
    
    /*
    Returned derivative order:
    (0: df/dm
     1: dg/dm)
    */

    std::vector<high_prec_float> derivs_with_mZ = {first_derivative(h_m, f_mm4, f_mm3, f_mm2, f_mm, f_mp, f_mp2, f_mp3, f_mp4),
                                                   first_derivative(h_m, g_mm4, g_mm3, g_mm2, g_mm, g_mp, g_mp2, g_mp3, g_mp4)};//,
                                    //    second_derivative(h_m, f_orig, f_mm4, f_mm3, f_mm2, f_mm, f_mp, f_mp2, f_mp3, f_mp4)};//,
                                    //    third_derivative(h_m, f_mm4, f_mm3, f_mm2, f_mm, f_mp, f_mp2, f_mp3, f_mp4),
                                    //    first_derivative(h_m, g_mm4, g_mm3, g_mm2, g_mm, g_mp, g_mp2, g_mp3, g_mp4),
                                    //    second_derivative(h_m, g_orig, g_mm4, g_mm3, g_mm2, g_mm, g_mp, g_mp2, g_mp3, g_mp4),
                                    //    third_derivative(h_m, g_mm4, g_mm3, g_mm2, g_mm, g_mp, g_mp2, g_mp3, g_mp4)};
    return derivs_with_mZ;
}

high_prec_float second_cross_derivative(high_prec_float hx, high_prec_float hy, high_prec_float f_xp_yp, high_prec_float f_xp_ym, high_prec_float f_xm_yp, high_prec_float f_xm_ym) {
    return ((high_prec_float(1.0) / (high_prec_float(4.0) * hx * hy))
            * (f_xp_yp + f_xm_ym - f_xp_ym - f_xm_yp));
}

high_prec_float third_cross_derivative(high_prec_float hx, high_prec_float hy, high_prec_float hz, high_prec_float f_xp_yp_zp, high_prec_float f_xm_ym_zm, high_prec_float f_xp_ym_zp, high_prec_float f_xm_yp_zp, high_prec_float f_xm_ym_zp,
                                       high_prec_float f_xp_yp_zm, high_prec_float f_xm_yp_zm, high_prec_float f_xp_ym_zm, high_prec_float f_xm_zm, high_prec_float f_xm_zp, high_prec_float f_xp_zm, high_prec_float f_xp_zp,
                                       high_prec_float f_xm_ym, high_prec_float f_xm_yp, high_prec_float f_xp_ym, high_prec_float f_xp_yp) {
    return ((high_prec_float(1.0) / (high_prec_float(8.0) * hx * hy * hz))
            * (f_xp_yp_zp - f_xp_yp_zm - f_xp_ym_zp - f_xm_yp_zp - f_xm_ym_zm + f_xm_ym_zp + f_xm_yp_zm + f_xp_ym_zm
               + (high_prec_float(0.5) * (f_xm_ym + f_xp_yp + f_xp_ym + f_xm_yp - f_xm_zm - f_xm_zp - f_xp_zm - f_xp_zp))));
}

std::vector<high_prec_float> crossDerivs(std::vector<high_prec_float>& original_weak_conditions, high_prec_float& orig_mZ2_val, int idx_to_shift, high_prec_float& logQSUSYval) {
    high_prec_float h_m = pow(((high_prec_float(3.0)) * (boost::math::float_next(sqrt(orig_mZ2_val)) - sqrt(orig_mZ2_val))), (high_prec_float(1.0) / high_prec_float(3.0)));
    high_prec_float m_p = sqrt(orig_mZ2_val) + h_m;
    high_prec_float m_m = sqrt(orig_mZ2_val) - h_m;    
    high_prec_float m2_p = pow(m_p, high_prec_float(2.0));
    high_prec_float m2_m = pow(m_m, high_prec_float(2.0));
    
    high_prec_float p_orig, h_p, p_plus, p_minus;
    if (idx_to_shift == 42) {
        p_orig = original_weak_conditions[idx_to_shift] / original_weak_conditions[6];
        h_p = min(high_prec_float(0.95), pow(high_prec_float(3.0) * (boost::math::float_next(abs(p_orig)) - abs(p_orig)), high_prec_float(high_prec_float(1.0) / high_prec_float(3.0))));
        p_plus = p_orig + h_p;
        p_minus = p_orig - h_p;
    } else if ((idx_to_shift >= 16) && (idx_to_shift <= 24)) {
        p_orig = original_weak_conditions[idx_to_shift] / original_weak_conditions[idx_to_shift-9];
        h_p = min(high_prec_float(0.95), pow(high_prec_float(3.0) * (boost::math::float_next(abs(p_orig)) - abs(p_orig)), high_prec_float(high_prec_float(1.0) / high_prec_float(3.0))));
        p_plus = (p_orig + h_p) * original_weak_conditions[idx_to_shift-9];
        p_minus = (p_orig - h_p) * original_weak_conditions[idx_to_shift-9];
    }
    else {
        p_orig = original_weak_conditions[idx_to_shift];
        h_p = min(high_prec_float(0.95), pow(high_prec_float(3.0) * (boost::math::float_next(abs(p_orig)) - abs(p_orig)), high_prec_float(high_prec_float(1.0) / high_prec_float(3.0))));
        p_plus = p_orig + h_p;
        p_minus = p_orig - h_p;
    }
    std::vector<high_prec_float> weaksols_pp = original_weak_conditions;
    std::vector<high_prec_float> newtanbweak_plus = original_weak_conditions;
    std::vector<high_prec_float> weaksols_pm = original_weak_conditions;
    std::vector<high_prec_float> newtanbweak_minus = original_weak_conditions;

    if (idx_to_shift == 6) {
        weaksols_pp[42] = original_weak_conditions[42] * p_plus / original_weak_conditions[6];
        weaksols_pm[42] = original_weak_conditions[42] * p_minus / original_weak_conditions[6];
        weaksols_pp[6] = p_plus;
        weaksols_pm[6] = p_minus;
    } else if (idx_to_shift == 42) {
        weaksols_pp[42] = original_weak_conditions[6] * p_plus;
        weaksols_pm[42] = original_weak_conditions[6] * p_minus;
    } else {
        weaksols_pp[idx_to_shift] = p_plus;
        weaksols_pm[idx_to_shift] = p_minus;
    }

    high_prec_float tanb_orig = original_weak_conditions[43];
    high_prec_float h_tanb = pow(high_prec_float(3.0) * (boost::math::float_next(abs(tanb_orig)) - abs(tanb_orig)), (high_prec_float(1.0) / 5.0));
    
    newtanbweak_plus[43] = tanb_orig + h_tanb;
    newtanbweak_minus[43] = tanb_orig - h_tanb;
        
    std::vector<high_prec_float> weaksols_original = original_weak_conditions;
    std::vector<high_prec_float> weaksols_tp = newtanbweak_plus;
    std::vector<high_prec_float> weaksols_pp_tp = weaksols_pp;
    weaksols_pp_tp[43] = newtanbweak_plus[43];
    std::vector<high_prec_float> weaksols_pm_tp = weaksols_pm;
    weaksols_pm_tp[43] = newtanbweak_plus[43];
    std::vector<high_prec_float> weaksols_tm = newtanbweak_minus;
    std::vector<high_prec_float> weaksols_pp_tm = weaksols_pp;
    weaksols_pp_tm[43] = newtanbweak_minus[43];
    std::vector<high_prec_float> weaksols_pm_tm = weaksols_pm;
    weaksols_pm_tm[43] = newtanbweak_minus[43];

    for (int UpYukawaIndex = 7; UpYukawaIndex < 10; ++UpYukawaIndex) {
        weaksols_tp[UpYukawaIndex] *= sin(atan(weaksols_original[43])) / sin(atan(weaksols_tp[43]));
        weaksols_tp[UpYukawaIndex+9] *= weaksols_tp[UpYukawaIndex] / weaksols_original[UpYukawaIndex];
        weaksols_pp_tp[UpYukawaIndex] *= sin(atan(weaksols_original[43])) / sin(atan(weaksols_tp[43]));
        weaksols_pp_tp[UpYukawaIndex+9] *= weaksols_tp[UpYukawaIndex] / weaksols_original[UpYukawaIndex];
        weaksols_pm_tp[UpYukawaIndex] *= sin(atan(weaksols_original[43])) / sin(atan(weaksols_tp[43]));
        weaksols_pm_tp[UpYukawaIndex+9] *= weaksols_tp[UpYukawaIndex] / weaksols_original[UpYukawaIndex];
        weaksols_tm[UpYukawaIndex] *= sin(atan(weaksols_original[43])) / sin(atan(weaksols_tm[43]));
        weaksols_tm[UpYukawaIndex+9] *= weaksols_tm[UpYukawaIndex] / weaksols_original[UpYukawaIndex];
    }
    for (int DownYukawaIndex = 10; DownYukawaIndex < 16; ++DownYukawaIndex) {
        weaksols_tp[DownYukawaIndex] *= sin(atan(weaksols_original[43])) / sin(atan(weaksols_tp[43]));
        weaksols_tp[DownYukawaIndex+9] *= weaksols_tp[DownYukawaIndex] / weaksols_original[DownYukawaIndex];
        weaksols_pp_tp[DownYukawaIndex] *= sin(atan(weaksols_original[43])) / sin(atan(weaksols_tp[43]));
        weaksols_pp_tp[DownYukawaIndex+9] *= weaksols_tp[DownYukawaIndex] / weaksols_original[DownYukawaIndex];
        weaksols_pm_tp[DownYukawaIndex] *= sin(atan(weaksols_original[43])) / sin(atan(weaksols_tp[43]));
        weaksols_pm_tp[DownYukawaIndex+9] *= weaksols_tp[DownYukawaIndex] / weaksols_original[DownYukawaIndex];
        weaksols_tm[DownYukawaIndex] *= sin(atan(weaksols_original[43])) / sin(atan(weaksols_tm[43]));
        weaksols_tm[DownYukawaIndex+9] *= weaksols_tm[DownYukawaIndex] / weaksols_original[DownYukawaIndex];
    }

    high_prec_float f_mm_pm_tm, f_mp_pp_tp, f_mm_pp_tp, f_mp_pm_tp, f_mp_pp_tm, f_mm_pm_tp, f_mm_pp_tm, f_mp_pm_tm, f_mm_tm, f_mm_tp, f_mm_pm, f_mm_pp, f_mp_tm, f_mp_tp, f_mp_pm, f_mp_pp, f_pm_tm, f_pm_tp, f_pp_tm, f_pp_tp;
    high_prec_float g_mm_pm_tm, g_mp_pp_tp, g_mm_pp_tp, g_mp_pm_tp, g_mp_pp_tm, g_mm_pm_tp, g_mm_pp_tm, g_mp_pm_tm, g_mm_tm, g_mm_tp, g_mm_pm, g_mm_pp, g_mp_tm, g_mp_tp, g_mp_pm, g_mp_pp, g_pm_tm, g_pm_tp, g_pp_tm, g_pp_tp;
    
    f_mm_pm_tm = calculate_mZ2_eq(weaksols_pm_tm, exp(logQSUSYval), m2_m);
    f_mp_pp_tp = calculate_mZ2_eq(weaksols_pp_tp, exp(logQSUSYval), m2_p);
    f_mm_pp_tp = calculate_mZ2_eq(weaksols_pp_tp, exp(logQSUSYval), m2_m);
    f_mp_pm_tp = calculate_mZ2_eq(weaksols_pm_tp, exp(logQSUSYval), m2_p);
    f_mp_pp_tm = calculate_mZ2_eq(weaksols_pp_tm, exp(logQSUSYval), m2_p);
    f_mm_pm_tp = calculate_mZ2_eq(weaksols_pm_tp, exp(logQSUSYval), m2_m);
    f_mm_pp_tm = calculate_mZ2_eq(weaksols_pp_tm, exp(logQSUSYval), m2_m);
    f_mp_pm_tm = calculate_mZ2_eq(weaksols_pm_tm, exp(logQSUSYval), m2_p);
    f_mm_tm = calculate_mZ2_eq(weaksols_tm, exp(logQSUSYval), m2_m);
    f_mm_tp = calculate_mZ2_eq(weaksols_tp, exp(logQSUSYval), m2_m);
    f_mm_pm = calculate_mZ2_eq(weaksols_pm, exp(logQSUSYval), m2_m);
    f_mm_pp = calculate_mZ2_eq(weaksols_pp, exp(logQSUSYval), m2_m);
    f_mp_tm = calculate_mZ2_eq(weaksols_tm, exp(logQSUSYval), m2_p);
    f_mp_tp = calculate_mZ2_eq(weaksols_tp, exp(logQSUSYval), m2_p);
    f_mp_pm = calculate_mZ2_eq(weaksols_pm, exp(logQSUSYval), m2_p);
    f_mp_pp = calculate_mZ2_eq(weaksols_pp, exp(logQSUSYval), m2_p);
    f_pm_tm = calculate_mZ2_eq(weaksols_pm_tm, exp(logQSUSYval), orig_mZ2_val);
    f_pm_tp = calculate_mZ2_eq(weaksols_pm_tp, exp(logQSUSYval), orig_mZ2_val);
    f_pp_tm = calculate_mZ2_eq(weaksols_pp_tm, exp(logQSUSYval), orig_mZ2_val);
    f_pp_tp = calculate_mZ2_eq(weaksols_pp_tp, exp(logQSUSYval), orig_mZ2_val);
    
    g_mm_pm_tm = calculate_tanb_eq(weaksols_pm_tm, exp(logQSUSYval), m2_m);
    g_mp_pp_tp = calculate_tanb_eq(weaksols_pp_tp, exp(logQSUSYval), m2_p);
    g_mm_pp_tp = calculate_tanb_eq(weaksols_pp_tp, exp(logQSUSYval), m2_m);
    g_mp_pm_tp = calculate_tanb_eq(weaksols_pm_tp, exp(logQSUSYval), m2_p);
    g_mp_pp_tm = calculate_tanb_eq(weaksols_pp_tm, exp(logQSUSYval), m2_p);
    g_mm_pm_tp = calculate_tanb_eq(weaksols_pm_tp, exp(logQSUSYval), m2_m);
    g_mm_pp_tm = calculate_tanb_eq(weaksols_pp_tm, exp(logQSUSYval), m2_m);
    g_mp_pm_tm = calculate_tanb_eq(weaksols_pm_tm, exp(logQSUSYval), m2_p);
    g_mm_tm = calculate_tanb_eq(weaksols_tm, exp(logQSUSYval), m2_m);
    g_mm_tp = calculate_tanb_eq(weaksols_tp, exp(logQSUSYval), m2_m);
    g_mm_pm = calculate_tanb_eq(weaksols_pm, exp(logQSUSYval), m2_m);
    g_mm_pp = calculate_tanb_eq(weaksols_pp, exp(logQSUSYval), m2_m);
    g_mp_tm = calculate_tanb_eq(weaksols_tm, exp(logQSUSYval), m2_p);
    g_mp_tp = calculate_tanb_eq(weaksols_tp, exp(logQSUSYval), m2_p);
    g_mp_pm = calculate_tanb_eq(weaksols_pm, exp(logQSUSYval), m2_p);
    g_mp_pp = calculate_tanb_eq(weaksols_pp, exp(logQSUSYval), m2_p);
    g_pm_tm = calculate_tanb_eq(weaksols_pm_tm, exp(logQSUSYval), orig_mZ2_val);
    g_pm_tp = calculate_tanb_eq(weaksols_pm_tp, exp(logQSUSYval), orig_mZ2_val);
    g_pp_tm = calculate_tanb_eq(weaksols_pp_tm, exp(logQSUSYval), orig_mZ2_val);
    g_pp_tp = calculate_tanb_eq(weaksols_pp_tp, exp(logQSUSYval), orig_mZ2_val);
    /*
    Return order for cross derivatives (M = mZ^2 minimization condition, T = tan(beta) minimization condition):
    (0: d^2M/dmdp
     1: d^2M/dmdt
     2: d^2M/dpdt
     3: d^3M/dmdpdt
     4: d^2T/dmdp
     5: d^2T/dmdt
     6: d^2T/dpdt
     7: d^3T/dmdpdt)
    */
    std::vector<high_prec_float> CrossDerivatives = {second_cross_derivative(h_m, h_p, f_mp_pp, f_mp_pm, f_mm_pp, f_mm_pm),
                                                     second_cross_derivative(h_m, h_tanb, f_mp_tp, f_mp_tm, f_mm_tp, f_mm_tm),
                                                     second_cross_derivative(h_p, h_tanb, f_pp_tp, f_pp_tm, f_pm_tp, f_pm_tm),
                                                     third_cross_derivative(h_p, h_m, h_tanb, f_mp_pp_tp, f_mm_pm_tm, f_mp_pm_tp, f_mm_pp_tp, f_mm_pm_tp, f_mp_pp_tm, f_mm_pp_tm, f_mp_pm_tm, f_mm_tm, f_mm_tp, f_mp_tm, f_mp_tp, f_mm_pm, f_mm_pp, f_mp_pm, f_mp_pp),
                                                     second_cross_derivative(h_m, h_p, g_mp_pp, g_mp_pm, g_mm_pp, g_mm_pm),
                                                     second_cross_derivative(h_m, h_tanb, g_mp_tp, g_mp_tm, g_mm_tp, g_mm_tm),
                                                     second_cross_derivative(h_p, h_tanb, g_pp_tp, g_pp_tm, g_pm_tp, g_pm_tm),
                                                     third_cross_derivative(h_p, h_m, h_tanb, g_mp_pp_tp, g_mm_pm_tm, g_mp_pm_tp, g_mm_pp_tp, g_mm_pm_tp, g_mp_pp_tm, g_mm_pp_tm, g_mp_pm_tm, g_mm_tm, g_mm_tp, g_mp_tm, g_mp_tp, g_mm_pm, g_mm_pp, g_mp_pm, g_mp_pp)};
    return CrossDerivatives;
}