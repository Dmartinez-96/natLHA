// RADCORR_CALC_HPP
#ifndef RADCORR_CALC_HPP
#define RADCORR_CALC_HPP

#include <complex>
#include <stdexcept>
#include <string>
#include <vector>
#include <boost/multiprecision/mpfr.hpp>
using namespace boost::multiprecision;
typedef number<mpfr_float_backend<50>> high_prec_float;  // 50 decimal digits of precision

struct NamedRadiativeCorrection {
    high_prec_float value;
    std::string label;
};

/// An invalid numerical state invalidates the calculation that consumed it.  The stage and the
/// complete list of invalid terms are carried to the non-throwing API boundary; no caller may
/// silently omit a failed contribution and continue with a partial sum.
class NumericalFailure : public std::runtime_error {
public:
    NumericalFailure(std::string stage, std::vector<std::string> invalidTerms);

    const std::string stage;
    const std::vector<std::string> invalidTerms;
};

void requireFiniteRadiativeCorrections(
    const std::string& stage,
    const std::vector<NamedRadiativeCorrection>& terms);

high_prec_float sumFiniteRadiativeCorrections(
    const std::string& stage,
    const std::vector<NamedRadiativeCorrection>& terms);

namespace radcorr_detail {

std::complex<high_prec_float> checkedDilogarithm(
    const std::complex<high_prec_float>& input);
high_prec_float checkedLogFunctionFromSquaredMass(
    const high_prec_float& massSquared,
    const high_prec_float& renormalizationScaleSquared);
high_prec_float checkedSignedMZ2ContinuationLog(
    const high_prec_float& signedMZSquared,
    const high_prec_float& renormalizationScaleSquared);
void requireNegligiblePhiImaginaryPart(
    const std::complex<high_prec_float>& value,
    const high_prec_float& tolerance);
high_prec_float checkedPhiFunction(
    const high_prec_float& x,
    const high_prec_float& y,
    const high_prec_float& z);

}  // namespace radcorr_detail

std::vector<high_prec_float> radcorr_calc(std::vector<high_prec_float> weak_boundary_conditions, high_prec_float myQ, high_prec_float mymZsq);

#endif
