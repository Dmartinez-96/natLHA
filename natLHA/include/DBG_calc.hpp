// DBG_CALC_HPP

#ifndef DBG_CALC_HPP
#define DBG_CALC_HPP

#include <vector>
#include <string>
#include <boost/multiprecision/mpfr.hpp>
using namespace boost::multiprecision;
typedef number<mpfr_float_backend<50>> high_prec_float;  // 50 decimal digits of precision

struct LabeledValueBG {
    high_prec_float value;
    std::string label;
};

std::vector<LabeledValueBG> DBG_calc(int& modselno, int& precselno,
                                     high_prec_float GUT_SCALE, high_prec_float myweakscale, high_prec_float inptanbval,
                                     std::vector<high_prec_float> GUT_boundary_conditions, high_prec_float originalmZ2value);

#endif