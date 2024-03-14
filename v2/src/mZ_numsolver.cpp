#include <iostream>
#include <cmath>
#include <vector>
#include <boost/math/special_functions/next.hpp>
#include "radcorr_calc.hpp"
#include "mZ_numsolver.hpp"

using namespace std;

double getmZ2(vector<double> input_weakscaleBCs, double input_QSUSY, double mZ2GuessValue) {
    double least_Sq_Tol = 1.0e-6;
    double current_iter_lsq = 100.0;
    double mu2 = input_weakscaleBCs[6] * input_weakscaleBCs[6];
    double mHd2 = input_weakscaleBCs[26];
    double mHu2 = input_weakscaleBCs[25];
    double tan_b = input_weakscaleBCs[43];
    double current_mZ2 = mZ2GuessValue;
    int number_of_steps_done = 0;
    vector<double> RadCorrs = radcorr_calc(input_weakscaleBCs, input_QSUSY, current_mZ2);
    current_mZ2 = ((2.0 * ((mHd2 + RadCorrs[1] - ((mHu2 + RadCorrs[0]) * pow(tan_b, 2.0))) / (pow(tan_b, 2.0) - 1.0)))
                   - (2.0 * mu2));
    double new_mZ2 = current_mZ2;
    double approxDerivStep = pow(3.0 * boost::math::float_next(abs(new_mZ2) - new_mZ2), (1.0 / 3.0));
    while ((current_iter_lsq > least_Sq_Tol) && (number_of_steps_done < 100000)) {
        new_mZ2 = ((2.0 * ((mHd2 + RadCorrs[1] - ((mHu2 + RadCorrs[0]) * pow(tan_b, 2.0))) / (pow(tan_b, 2.0) - 1.0)))
                   - (2.0 * mu2));
        approxDerivStep = pow(3.0 * boost::math::float_next(abs(new_mZ2) - new_mZ2), (1.0 / 3.0));
        RadCorrs = radcorr_calc(input_weakscaleBCs, input_QSUSY, new_mZ2);
        current_iter_lsq = pow((1.0 - (current_mZ2 / new_mZ2)), 2.0);
        current_mZ2 = new_mZ2;
        number_of_steps_done++;
    }
    if (number_of_steps_done == 100000) {
        cout << "Ran out of iteration attempts to converge mZ^2" << endl;
    }
    return current_mZ2;
}