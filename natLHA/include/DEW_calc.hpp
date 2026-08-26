// DEW_CALC_HPP

#ifndef DEW_CALC_HPP
#define DEW_CALC_HPP

#include <vector>
#include <string>
#include <boost/multiprecision/mpfr.hpp>
using namespace boost::multiprecision;
typedef number<mpfr_float_backend<50>> high_prec_float;  // 50 decimal digits of precision


struct LabeledValue {
    high_prec_float value;
    std::string label;
};

std::vector<LabeledValue> DEW_calc(std::vector<high_prec_float> weak_boundary_conditions, high_prec_float myQ);

#endif
