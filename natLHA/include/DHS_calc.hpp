// DHS_CALC_HPP

#ifndef DHS_CALC_HPP
#define DHS_CALC_HPP

#include <vector>
#include <string>
#include <boost/multiprecision/mpfr.hpp>
using namespace boost::multiprecision;
typedef number<mpfr_float_backend<50>> high_prec_float;

struct LabeledValueHS {
    high_prec_float value;
    std::string label;
};

std::vector<LabeledValueHS> DHS_calc(high_prec_float& mHdsq_Lambda, high_prec_float delta_mHdsq, high_prec_float& mHusq_Lambda, high_prec_float delta_mHusq,
                                     high_prec_float mu_Lambdasq, high_prec_float delta_musq, high_prec_float running_mZ_sq, high_prec_float tanb_sq, high_prec_float& sigmauu_tot,
                                     high_prec_float& sigmadd_tot);

#endif