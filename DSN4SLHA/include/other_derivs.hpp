// OTHER_DERIVS_HPP

#ifndef OTHER_DERIVS_HPP
#define OTHER_DERIVS_HPP

#include <vector>
#include <boost/multiprecision/mpfr.hpp>
using namespace boost::multiprecision;
typedef number<mpfr_float_backend<50>> high_prec_float;

std::vector<high_prec_float> mZ2_derivatives(std::vector<high_prec_float>& original_weak_conditions, high_prec_float& orig_mZ2_val, high_prec_float& logQSUSYval);
std::vector<high_prec_float> crossDerivs(std::vector<high_prec_float>& original_weak_conditions, high_prec_float& orig_mZ2_val, int idx_to_shift, high_prec_float& logQSUSYval);

#endif