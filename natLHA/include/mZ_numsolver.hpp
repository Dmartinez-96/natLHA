// MZ_NUMSOLVER_HPP

#ifndef MZ_NUMSOLVER_HPP
#define MZ_NUMSOLVER_HPP

#include <vector>

using namespace std;

double getmZ2(const vector<double>& input_weakscaleBCs, double input_QSUSY, double guess);//double lowerbnd, double upperbnd);

double gettanb(const vector<double>& input_weakscaleBCs, double input_QSUSY, double mZ2fixed, double guess);


#endif