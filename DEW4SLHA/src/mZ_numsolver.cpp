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

//Newton's method
high_prec_float getmZ2(const vector<high_prec_float>& input_weakscaleBCs, high_prec_float input_QSUSY, high_prec_float guess) {
    high_prec_float current_mZ2 = guess;
    high_prec_float prev_f_x = numeric_limits<high_prec_float>::max();
    vector<high_prec_float> RadCorrs, RadCorrs_plush, RadCorrs_minush;
    int number_of_steps_done = 0;
    high_prec_float lambda = high_prec_float(0.5); // Damping factor to address oscillations
    high_prec_float least_Sq_Tol = high_prec_float(1.0e-12);
    while (number_of_steps_done < 25000) {
        RadCorrs = radcorr_calc(input_weakscaleBCs, input_QSUSY, current_mZ2);
        high_prec_float f_x = current_mZ2 - ((high_prec_float(2.0) * ((input_weakscaleBCs[26] + RadCorrs[1] - ((input_weakscaleBCs[25] + RadCorrs[0]) * pow(input_weakscaleBCs[43], high_prec_float(2.0)))) / (pow(input_weakscaleBCs[43], high_prec_float(2.0)) - high_prec_float(1.0)))) - (high_prec_float(2.0) * pow(input_weakscaleBCs[6], high_prec_float(2.0))));
        high_prec_float h = cbrt(3.0 * (nextafter(current_mZ2, current_mZ2 + 1) - current_mZ2));
        high_prec_float mZ2_plus_h = current_mZ2 + h;
        RadCorrs_plush = radcorr_calc(input_weakscaleBCs, input_QSUSY, mZ2_plus_h);
        high_prec_float f_x_plus_h = mZ2_plus_h - ((high_prec_float(2.0) * ((input_weakscaleBCs[26] + RadCorrs_plush[1] - ((input_weakscaleBCs[25] + RadCorrs_plush[0]) * pow(input_weakscaleBCs[43], high_prec_float(2.0)))) / (pow(input_weakscaleBCs[43], high_prec_float(2.0)) - high_prec_float(1.0)))) - (high_prec_float(2.0) * pow(input_weakscaleBCs[6], high_prec_float(2.0))));
        high_prec_float mZ2_minus_h = current_mZ2 - h;
        RadCorrs_minush = radcorr_calc(input_weakscaleBCs, input_QSUSY, mZ2_minus_h);
        high_prec_float f_x_minus_h = mZ2_minus_h - ((high_prec_float(2.0) * ((input_weakscaleBCs[26] + RadCorrs_minush[1] - ((input_weakscaleBCs[25] + RadCorrs_minush[0]) * pow(input_weakscaleBCs[43], high_prec_float(2.0)))) / (pow(input_weakscaleBCs[43], high_prec_float(2.0)) - high_prec_float(1.0)))) - (high_prec_float(2.0) * pow(input_weakscaleBCs[6], high_prec_float(2.0))));
        
        high_prec_float df_dx = (f_x_plus_h - f_x_minus_h) / (high_prec_float(2.0) * h);

        // Newton's update step
        high_prec_float deltaX = lambda * f_x / df_dx;

        // Adjust lambda based on the behavior
        if (fabs(f_x) >= fabs(prev_f_x)) {
            lambda *= high_prec_float(0.8); // Reduce step size
        } else if (lambda < high_prec_float(1.0)) {
            lambda += high_prec_float(0.1); // Try increasing lambda cautiously
            lambda = min(lambda, high_prec_float(1.0)); // Limit lambda to 1.0
        }

        // Update for next iteration
        prev_f_x = f_x;
        current_mZ2 -= deltaX;

        // Check for convergence
        if (fabs(deltaX) < least_Sq_Tol) {
            break;
        }
        number_of_steps_done++;
    }

    return current_mZ2;
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
