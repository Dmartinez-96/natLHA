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

using namespace std;

//Newton's method
double getmZ2(const vector<double>& input_weakscaleBCs, double input_QSUSY, double guess) {
//     const double least_Sq_Tol = 1.0e-6;
//     boost::uintmax_t max_iter = 1000;
    
//     pair<double, double> result = boost::math::tools::bracket_and_solve_root(
//         Func(input_weakscaleBCs, input_QSUSY), lowerbnd, upperbnd, 
//         true, [](double min, double max) { return (max - min) / 1000; }, max_iter);
//     double root = (result.first + result.second) / 2;

//     cout << "mZ^2 found at: " << root << " after " << max_iter << " iterations" << endl;
//     return root;
// }
    // cout << "Checking weak BCs: " << endl;
    // for (double value : input_weakscaleBCs) {
    //     cout << value << endl;
    // }
    double current_mZ2 = guess;
    double prev_f_x = std::numeric_limits<double>::max();
    //double h = 1.0e-3;//boost::math::float_next(current_mZ2) - current_mZ2; // Small step for derivative approximation
    vector<double> RadCorrs, RadCorrs_plush, RadCorrs_minush;
    int number_of_steps_done = 0;
    double lambda = 0.5; // Damping factor to address oscillations
    double least_Sq_Tol = 1.0e-4;
    while (number_of_steps_done < 25000) {
        //cout << "mZ^2 currently = " << current_mZ2 << endl;
        RadCorrs = radcorr_calc(input_weakscaleBCs, input_QSUSY, current_mZ2);
        double f_x = current_mZ2 - ((2.0 * ((input_weakscaleBCs[26] + RadCorrs[1] - ((input_weakscaleBCs[25] + RadCorrs[0]) * pow(input_weakscaleBCs[43], 2.0))) / (pow(input_weakscaleBCs[43], 2.0) - 1.0))) - (2.0 * pow(input_weakscaleBCs[6], 2.0)));
        double h = std::cbrt(3.0 * (boost::math::float_next(current_mZ2) - current_mZ2));//std::cbrt(std::numeric_limits<double>::epsilon()) * std::max(1.0, std::abs(current_mZ2)); // Step size for numerical derivative
        //cout << "Derivative step size: " << h << endl;
        // Approximate derivative (f'(x)) with respect to mZ2
        double mZ2_plus_h = current_mZ2 + h;//abs(pow(3.0 * (boost::math::float_next(current_mZ2) - current_mZ2), 1.0 / 3.0));
        RadCorrs_plush = radcorr_calc(input_weakscaleBCs, input_QSUSY, mZ2_plus_h);
        double f_x_plus_h = mZ2_plus_h - ((2.0 * ((input_weakscaleBCs[26] + RadCorrs_plush[1] - ((input_weakscaleBCs[25] + RadCorrs_plush[0]) * pow(input_weakscaleBCs[43], 2.0))) / (pow(input_weakscaleBCs[43], 2.0) - 1.0))) - (2.0 * pow(input_weakscaleBCs[6], 2.0)));
        double mZ2_minus_h = current_mZ2 - h;//abs(pow(3.0 * (current_mZ2 - boost::math::float_prior(current_mZ2)), 1.0 / 3.0));
        RadCorrs_minush = radcorr_calc(input_weakscaleBCs, input_QSUSY, mZ2_minus_h);
        double f_x_minus_h = mZ2_minus_h - ((2.0 * ((input_weakscaleBCs[26] + RadCorrs_minush[1] - ((input_weakscaleBCs[25] + RadCorrs_minush[0]) * pow(input_weakscaleBCs[43], 2.0))) / (pow(input_weakscaleBCs[43], 2.0) - 1.0))) - (2.0 * pow(input_weakscaleBCs[6], 2.0)));
        
        double df_dx = (f_x_plus_h - f_x_minus_h) / (2.0 * h);

        // Check for division by zero or extremely small derivative
        if (fabs(df_dx) < least_Sq_Tol) {
            //cerr << "Derivative is too small, stopping iteration." << endl;
            break;
        }
        //cout << "Current f_x: " << f_x << endl;
        //cout << "Current mZ^2: " << current_mZ2 << endl;
        // Newton's update step
        double deltaX = lambda * f_x / df_dx;

        // Adjust lambda based on the behavior
        if (fabs(f_x) >= fabs(prev_f_x)) { // No progress or oscillation
            lambda *= 0.8; // Reduce step size
        } else if (lambda < 1.0) { // Smooth convergence
            lambda += 0.1; // Try increasing lambda cautiously
            lambda = min(lambda, 1.0); // Limit lambda to 1.0
        }

        // Update for next iteration
        prev_f_x = f_x;
        current_mZ2 -= deltaX;
        // if (number_of_steps_done % 100 == 0) {
        //     cout << "Current mZ^2: " << current_mZ2 << endl;
        // }

        // Check for convergence
        if (fabs(deltaX) < least_Sq_Tol) {
            //cout << "Converged in " << number_of_steps_done + 1 << " iterations." << endl;
            break;
        }
        number_of_steps_done++;
    }

    // if (number_of_steps_done == 100000) {
    //     //cerr << "Ran out of iteration attempts to converge mZ^2, results may be inaccurate" << endl;
    // }
    return current_mZ2;
}

double gettanb(const vector<double>& input_weakscaleBCs, double input_QSUSY, double mZ2fixed, double guess) {
    double current_tanb = guess;
    double prev_f_x = std::numeric_limits<double>::max();
    //double h = 1.0e-3;//boost::math::float_next(current_mZ2) - current_mZ2; // Small step for derivative approximation
    vector<double> RadCorrs, RadCorrs_plush, RadCorrs_minush;
    int number_of_steps_done = 0;
    double lambda = 0.5; // Damping factor to address oscillations
    double least_Sq_Tol = 1.0e-4;
    while (number_of_steps_done < 25000) {
        //cout << "mZ^2 currently = " << current_mZ2 << endl;
        RadCorrs = radcorr_calc(input_weakscaleBCs, input_QSUSY, mZ2fixed);
        double f_x = current_tanb - tan(0.5 * (M_PI - asin((2.0 * input_weakscaleBCs[42] / (input_weakscaleBCs[25] + input_weakscaleBCs[26] + (2.0 * pow(input_weakscaleBCs[6], 2.0)) + RadCorrs[0] + RadCorrs[1])))));
        double h = std::cbrt(3.0 * (boost::math::float_next(abs(current_tanb)) - abs(current_tanb)));//std::cbrt(std::numeric_limits<double>::epsilon()) * std::max(1.0, std::abs(current_mZ2)); // Step size for numerical derivative
        //cout << "Derivative step size: " << h << endl;
        // Approximate derivative (f'(x)) with respect to mZ2
        double tanb_plus_h = current_tanb + h;//abs(pow(3.0 * (boost::math::float_next(current_mZ2) - current_mZ2), 1.0 / 3.0));
        RadCorrs_plush = radcorr_calc(input_weakscaleBCs, input_QSUSY, mZ2fixed);
        double f_x_plus_h = tanb_plus_h - tan(0.5 * (M_PI - asin((2.0 * input_weakscaleBCs[42] / (input_weakscaleBCs[25] + input_weakscaleBCs[26] + (2.0 * pow(input_weakscaleBCs[6], 2.0)) + RadCorrs_plush[0] + RadCorrs_plush[1])))));
        double tanb_minus_h = current_tanb - h;//abs(pow(3.0 * (current_mZ2 - boost::math::float_prior(current_mZ2)), 1.0 / 3.0));
        RadCorrs_minush = radcorr_calc(input_weakscaleBCs, input_QSUSY, mZ2fixed);
        double f_x_minus_h = tanb_minus_h - tan(0.5 * (M_PI - asin((2.0 * input_weakscaleBCs[42] / (input_weakscaleBCs[25] + input_weakscaleBCs[26] + (2.0 * pow(input_weakscaleBCs[6], 2.0)) + RadCorrs_minush[0] + RadCorrs_minush[1])))));
        
        double df_dx = (f_x_plus_h - f_x_minus_h) / (2.0 * h);

        // Check for division by zero or extremely small derivative
        if (fabs(df_dx) < least_Sq_Tol) {
            //cerr << "Derivative is too small, stopping iteration." << endl;
            break;
        }
        //cout << "Current f_x: " << f_x << endl;
        //cout << "Current mZ^2: " << current_mZ2 << endl;
        // Newton's update step
        double deltaX = lambda * f_x / df_dx;

        // Adjust lambda based on the behavior
        if (fabs(f_x) >= fabs(prev_f_x)) { // No progress or oscillation
            lambda *= 0.8; // Reduce step size
        } else if (lambda < 1.0) { // Smooth convergence
            lambda += 0.1; // Try increasing lambda cautiously
            lambda = min(lambda, 1.0); // Limit lambda to 1.0
        }

        // Update for next iteration
        prev_f_x = f_x;
        current_tanb -= deltaX;
        // if (number_of_steps_done % 100 == 0) {
        //     cout << "Current mZ^2: " << current_mZ2 << endl;
        // }

        // Check for convergence
        if (fabs(deltaX) < least_Sq_Tol) {
            //cout << "Converged in " << number_of_steps_done + 1 << " iterations." << endl;
            break;
        }
        number_of_steps_done++;
    }

    // if (number_of_steps_done == 100000) {
    //     //cerr << "Ran out of iteration attempts to converge mZ^2, results may be inaccurate" << endl;
    // }
    return current_tanb;
}

// Simple iterator

// double RHS_Func(const double& mHuSquared, const double& mHdSquared, const double& tanBeta, const double& muSquared, const double& SigmaUU, const double& SigmaDD) {
//     return ((2.0 * (mHdSquared + SigmaDD - ((mHuSquared + SigmaUU) * tanBeta * tanBeta))) / ((tanBeta * tanBeta) - 1.0)) - (2.0 * muSquared);
// }

// double getmZ2(const vector<double>& input_weakscaleBCs, double input_QSUSY, double guess) {
//     double LSqTol = 1.0e-9;
//     double Current_LSq = 100.0;
//     double muS = input_weakscaleBCs[6] * input_weakscaleBCs[6];
//     double mHuS = input_weakscaleBCs[25];
//     double mHdS = input_weakscaleBCs[26];
//     double TanBeta = input_weakscaleBCs[43];
//     int NumIter = 0;
//     // Damping parameter
//     // double lambda = 0.5;
    
//     double Current_mZ2 = guess;
//     vector<double> RadCorrs = radcorr_calc(input_weakscaleBCs, input_QSUSY, Current_mZ2);
//     double New_mZ2 = Current_mZ2;
//     // Simple iterator with averaging and damping to facilitate convergence
//     // Error metric is determined via least squares
//     while ((Current_LSq > LSqTol) && (NumIter < 1000)) {
//         RadCorrs = radcorr_calc(input_weakscaleBCs, input_QSUSY, Current_mZ2);
//         New_mZ2 = RHS_Func(mHuS, mHdS, TanBeta, muS, RadCorrs[0], RadCorrs[1]);
//         Current_LSq = pow((1.0 - (Current_mZ2 / New_mZ2)), 2.0);
//         Current_mZ2 = New_mZ2;
//         //cout << "Current mZ^2: " << Current_mZ2 << endl;
//         NumIter++;
//         //cout << "Least squares: " << Current_LSq << endl;
//         //cout << "------------------------" << endl;
//     }
//     //cout << "Current mZ^2 = " << Current_mZ2 << " in " << NumIter << " iterations." << endl;
//     return Current_mZ2;
// }