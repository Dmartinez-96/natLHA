// MZ_NUMSOLVER_HPP

#ifndef MZ_NUMSOLVER_HPP
#define MZ_NUMSOLVER_HPP

#include <vector>
#include <boost/multiprecision/mpfr.hpp>

using namespace std;
using namespace boost::multiprecision;

typedef number<mpfr_float_backend<50>> high_prec_float;

/// Solve the EWSB relation for m_Z^2, starting from `guess`.
///
/// `converged`, when non-null, receives whether the solve met its tolerance. Read it wherever
/// the result feeds a derivative. Passing null ignores the distinction.
high_prec_float getmZ2(const vector<high_prec_float>& input_weakscaleBCs, high_prec_float input_QSUSY, high_prec_float guess, bool* converged = nullptr);//double lowerbnd, double upperbnd);

//double gettanb(const vector<double>& input_weakscaleBCs, double input_QSUSY, double mZ2fixed, double guess);


#endif