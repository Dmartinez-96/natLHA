// DSN_CALC_HPP

#ifndef DSN_CALC_HPP
#define DSN_CALC_HPP

#include <vector>
#include <string>
#include <boost/multiprecision/mpfr.hpp>
using namespace boost::multiprecision;
typedef number<mpfr_float_backend<50>> high_prec_float;

struct DSNLabeledValue {
    high_prec_float value;
    std::string label;
};

std::vector<DSNLabeledValue> DSN_calc(int precselno, std::vector<high_prec_float> Wk_boundary_conditions,
                                      high_prec_float& current_mZ2, high_prec_float& current_logQSUSY,
                                      high_prec_float& current_logQGUT, int& nF, int& nD);

#endif