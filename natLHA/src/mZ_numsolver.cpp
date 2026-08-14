#include <iostream>
#include <cmath>
#include <vector>
#include <boost/math/special_functions/next.hpp>
#include <boost/math/tools/roots.hpp>
#include <thread>
#include <chrono>
#include <limits>
#include "radcorr_calc.hpp"
#include "mZ_numsolver.hpp"
#include <boost/multiprecision/mpfr.hpp>

using namespace std;
using namespace boost::multiprecision;

// Define high precision floating point type with 50 decimal digits of precision
typedef number<mpfr_float_backend<50>> high_prec_float;

/// The EWSB residual whose root is m_Z^2:
///     f(m) = m - [ 2 * (mHd^2 + Sigma_d - (mHu^2 + Sigma_u) tan^2(beta)) / (tan^2(beta) - 1)
///                  - 2 * mu^2 ]
/// with Sigma_u and Sigma_d re-evaluated at m. Slot numbering follows the 44-entry weak-scale
/// state: 6 = mu, 25 = mHu^2, 26 = mHd^2, 43 = tanb.
///
/// ONE radcorr_calc per evaluation, and that call is the entire cost of a solver step, so the
/// count of residual evaluations is the quantity to minimise.
static high_prec_float mZ2Residual(const vector<high_prec_float>& bcs,
                                   high_prec_float qSusy, high_prec_float mZ2) {
    const vector<high_prec_float> rc = radcorr_calc(bcs, qSusy, mZ2);
    const high_prec_float tanbSq = pow(bcs[43], high_prec_float(2.0));
    const high_prec_float rhs =
        (high_prec_float(2.0) * ((bcs[26] + rc[1] - ((bcs[25] + rc[0]) * tanbSq))
                                 / (tanbSq - high_prec_float(1.0))))
        - (high_prec_float(2.0) * pow(bcs[6], high_prec_float(2.0)));
    return mZ2 - rhs;
}

/// Damped fixed-point solve for m_Z^2.
///
/// CONVERGENCE IS JUDGED ON THE RESIDUAL, scaled by the magnitude being solved for, and this
/// is the load-bearing choice. A step-size test cannot do the job: a damped iteration whose
/// damping collapses produces vanishing steps while |f| is still large, and a caller reading
/// only the returned value cannot tell that apart from a solved root. DBG_calc.cpp:65 calls
/// this function inside `deriv_mZ_step_calc`, the per-stencil-point evaluation whose values
/// are differenced to form Delta_BG, so a silently unsolved point corrupts a derivative rather
/// than merely costing time.
///
/// THE ITERATION. f(m) = m - R(m) where R depends on m only through the tadpoles Sigma_u and
/// Sigma_d, so f'(m) sits near 1 and the Newton step is close to -f(x). Taking that step
/// directly needs ONE residual evaluation, where forming a numerical derivative would need
/// three. The damping factor guards the cases where R varies enough for the undamped step to
/// overshoot: a step that fails to reduce |f| is retried from the same point with lambda
/// halved, and successive successes let lambda grow back toward 1.
///
/// `converged` is set only when the residual test is actually met, and remains false when the
/// step budget or the damping floor is reached first. Exhausting either returns the best
/// iterate reached, which is why the flag rather than the value is what tells a caller
/// whether to trust it.
high_prec_float getmZ2(const vector<high_prec_float>& input_weakscaleBCs, high_prec_float input_QSUSY, high_prec_float guess, bool* converged) {
    if (converged != nullptr) *converged = false;

    const high_prec_float scale = max(abs(guess), high_prec_float(1.0));
    const high_prec_float residTol = high_prec_float(1.0e-20) * scale;
    const high_prec_float lambdaFloor = high_prec_float(1.0e-8);
    const int kMaxSteps = 200;

    high_prec_float x = guess;
    high_prec_float fx = mZ2Residual(input_weakscaleBCs, input_QSUSY, x);
    high_prec_float lambda = high_prec_float(1.0);

    for (int step = 0; step < kMaxSteps; ++step) {
        if (abs(fx) <= residTol) {
            if (converged != nullptr) *converged = true;
            return x;
        }
        const high_prec_float xNew = x - (lambda * fx);
        const high_prec_float fNew = mZ2Residual(input_weakscaleBCs, input_QSUSY, xNew);
        if (abs(fNew) < abs(fx)) {
            x = xNew;
            fx = fNew;
            lambda = min(high_prec_float(1.0), lambda * high_prec_float(1.25));
        } else {
            lambda *= high_prec_float(0.5);
            if (lambda < lambdaFloor) break;
        }
    }

    if (abs(fx) <= residTol && converged != nullptr) *converged = true;
    return x;
}

// double gettanb(const vector<double>& input_weakscaleBCs, double input_QSUSY, double mZ2fixed, double guess) {
//     double current_tanb = guess;
//     double prev_f_x = std::numeric_limits<double>::max();
//     //double h = 1.0e-3;//boost::math::float_next(current_mZ2) - current_mZ2; // Small step for derivative approximation
//     vector<double> RadCorrs, RadCorrs_plush, RadCorrs_minush;
//     int number_of_steps_done = 0;
//     double lambda = 0.5; // Damping factor to address oscillations
//     double least_Sq_Tol = 1.0e-4;
//     while (number_of_steps_done < 25000) {
//         //cout << "mZ^2 currently = " << current_mZ2 << endl;
//         RadCorrs = radcorr_calc(input_weakscaleBCs, input_QSUSY, mZ2fixed);
//         double f_x = current_tanb - tan(0.5 * (M_PI - asin((high_prec_float(2.0) * input_weakscaleBCs[42] / (input_weakscaleBCs[25] + input_weakscaleBCs[26] + (high_prec_float(2.0) * pow(input_weakscaleBCs[6], high_prec_float(2.0))) + RadCorrs[0] + RadCorrs[1])))));
//         double h = std::cbrt(3.0 * (boost::math::float_next(abs(current_tanb)) - abs(current_tanb)));//std::cbrt(std::numeric_limits<double>::epsilon()) * std::max(1.0, std::abs(current_mZ2)); // Step size for numerical derivative
//         //cout << "Derivative step size: " << h << endl;
//         // Approximate derivative (f'(x)) with respect to mZ2
//         double tanb_plus_h = current_tanb + h;//abs(pow(3.0 * (boost::math::float_next(current_mZ2) - current_mZ2), 1.0 / 3.0));
//         RadCorrs_plush = radcorr_calc(input_weakscaleBCs, input_QSUSY, mZ2fixed);
//         double f_x_plus_h = tanb_plus_h - tan(0.5 * (M_PI - asin((high_prec_float(2.0) * input_weakscaleBCs[42] / (input_weakscaleBCs[25] + input_weakscaleBCs[26] + (high_prec_float(2.0) * pow(input_weakscaleBCs[6], high_prec_float(2.0))) + RadCorrs_plush[0] + RadCorrs_plush[1])))));
//         double tanb_minus_h = current_tanb - h;//abs(pow(3.0 * (current_mZ2 - boost::math::float_prior(current_mZ2)), 1.0 / 3.0));
//         RadCorrs_minush = radcorr_calc(input_weakscaleBCs, input_QSUSY, mZ2fixed);
//         double f_x_minus_h = tanb_minus_h - tan(0.5 * (M_PI - asin((high_prec_float(2.0) * input_weakscaleBCs[42] / (input_weakscaleBCs[25] + input_weakscaleBCs[26] + (high_prec_float(2.0) * pow(input_weakscaleBCs[6], high_prec_float(2.0))) + RadCorrs_minush[0] + RadCorrs_minush[1])))));
        
//         double df_dx = (f_x_plus_h - f_x_minus_h) / (high_prec_float(2.0) * h);

//         // Check for division by zero or extremely small derivative
//         if (fabs(df_dx) < least_Sq_Tol) {
//             //cerr << "Derivative is too small, stopping iteration." << endl;
//             break;
//         }
//         //cout << "Current f_x: " << f_x << endl;
//         //cout << "Current mZ^2: " << current_mZ2 << endl;
//         // Newton's update step
//         double deltaX = lambda * f_x / df_dx;

//         // Adjust lambda based on the behavior
//         if (fabs(f_x) >= fabs(prev_f_x)) { // No progress or oscillation
//             lambda *= 0.8; // Reduce step size
//         } else if (lambda < 1.0) { // Smooth convergence
//             lambda += 0.1; // Try increasing lambda cautiously
//             lambda = min(lambda, 1.0); // Limit lambda to 1.0
//         }

//         // Update for next iteration
//         prev_f_x = f_x;
//         current_tanb -= deltaX;
//         // if (number_of_steps_done % 100 == 0) {
//         //     cout << "Current mZ^2: " << current_mZ2 << endl;
//         // }

//         // Check for convergence
//         if (fabs(deltaX) < least_Sq_Tol) {
//             //cout << "Converged in " << number_of_steps_done + 1 << " iterations." << endl;
//             break;
//         }
//         number_of_steps_done++;
//     }

//     // if (number_of_steps_done == 100000) {
//     //     //cerr << "Ran out of iteration attempts to converge mZ^2, results may be inaccurate" << endl;
//     // }
//     return current_tanb;
// }
