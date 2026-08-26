#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <boost/math/constants/constants.hpp>

#include "radcorr_calc.hpp"

namespace {

bool expect(bool condition, const std::string& message) {
    if (condition) return true;
    std::cerr << message << "\n";
    return false;
}

}  // namespace

int main() {
    bool ok = true;
    const std::vector<NamedRadiativeCorrection> finite = {
        {high_prec_float(2), "a"}, {high_prec_float(-0.5), "b"}};
    ok &= expect(sumFiniteRadiativeCorrections("finite", finite) == high_prec_float(1.5),
                 "finite terms did not retain their full sum");

    const high_prec_float nan = std::numeric_limits<high_prec_float>::quiet_NaN();
    try {
        sumFiniteRadiativeCorrections(
            "nan-stage", {{high_prec_float(1), "good"}, {nan, "bad-nan"}});
        ok &= expect(false, "NaN radiative correction was silently accepted");
    } catch (const NumericalFailure& failure) {
        ok &= expect(failure.stage == "nan-stage"
                         && failure.invalidTerms == std::vector<std::string>{"bad-nan"},
                     "NaN failure lost its stage or term identity");
    }

    const high_prec_float infinity = std::numeric_limits<high_prec_float>::infinity();
    try {
        requireFiniteRadiativeCorrections(
            "inf-stage", {{infinity, "bad-inf"}, {-infinity, "bad-negative-inf"}});
        ok &= expect(false, "infinite radiative corrections were silently accepted");
    } catch (const NumericalFailure& failure) {
        ok &= expect(failure.stage == "inf-stage" && failure.invalidTerms.size() == 2,
                     "infinite-term failure did not retain every invalid term");
    }

    try {
        radcorr_detail::checkedDilogarithm({nan, high_prec_float(0)});
        ok &= expect(false, "the checked dilogarithm silently mapped a NaN input to a finite result");
    } catch (const NumericalFailure& failure) {
        ok &= expect(failure.stage == "radcorr_calc/dilog input"
                         && failure.invalidTerms == std::vector<std::string>{"real"},
                     "checked-dilogarithm input failure lost its stage or term identity");
    }

    const auto finiteDilogarithm = radcorr_detail::checkedDilogarithm(
        {high_prec_float("0.5"), high_prec_float(0)});
    const high_prec_float pi = boost::math::constants::pi<high_prec_float>();
    const high_prec_float logTwo = log(high_prec_float(2));
    const high_prec_float halfReference = pi * pi / 12 - logTwo * logTwo / 2;
    ok &= expect(abs(real(finiteDilogarithm) - halfReference)
                         < high_prec_float("1e-35")
                     && abs(imag(finiteDilogarithm)) < high_prec_float("1e-35"),
                 "checked dilogarithm rejected or changed a finite reference value");

    const auto hugeDilogarithm = radcorr_detail::checkedDilogarithm(
        {high_prec_float("1e10000"), high_prec_float(0)});
    ok &= expect((boost::math::isfinite)(real(hugeDilogarithm))
                     && (boost::math::isfinite)(imag(hugeDilogarithm))
                     && imag(hugeDilogarithm) < 0,
                 "finite high-precision dilogarithm input was narrowed through double");

    const std::complex<high_prec_float> smoothPoint(
        high_prec_float("0.5"), high_prec_float("0.2"));
    const high_prec_float smoothStep("1e-20");
    const auto smoothValue = radcorr_detail::checkedDilogarithm(smoothPoint);
    const auto shiftedSmoothValue = radcorr_detail::checkedDilogarithm(
        smoothPoint + std::complex<high_prec_float>(smoothStep, 0));
    const auto expectedSmoothDifference =
        -log(std::complex<high_prec_float>(1) - smoothPoint)
        / smoothPoint * smoothStep;
    ok &= expect(abs((shiftedSmoothValue - smoothValue) - expectedSmoothDifference)
                     < high_prec_float("1e-35"),
                 "dilogarithm lost sub-double precision or violated its local derivative");

    const auto invertedDilogarithm = radcorr_detail::checkedDilogarithm(
        {high_prec_float("1.3"), high_prec_float("0.4")});
    ok &= expect(abs(real(invertedDilogarithm)
                         - high_prec_float("1.50467403177290149"))
                         < high_prec_float("1e-14")
                     && abs(imag(invertedDilogarithm)
                         - high_prec_float("1.23402999093257471"))
                         < high_prec_float("1e-14"),
                 "complex dilogarithm inversion branch changed its reference value");

    const auto reflectedDilogarithm = radcorr_detail::checkedDilogarithm(
        {high_prec_float("0.99"), high_prec_float("0.01")});
    ok &= expect(abs(real(reflectedDilogarithm)
                         - high_prec_float("1.58441816263516522"))
                         < high_prec_float("1e-14")
                     && abs(imag(reflectedDilogarithm)
                         - high_prec_float("0.0452114364222384532"))
                         < high_prec_float("1e-14"),
                 "complex dilogarithm reflection branch changed its reference value");

    const std::complex<high_prec_float> unitCirclePoint(
        high_prec_float("0.5"), sqrt(high_prec_float(3)) / 2);
    const auto unitCircleDilogarithm =
        radcorr_detail::checkedDilogarithm(unitCirclePoint);
    ok &= expect(abs(real(unitCircleDilogarithm) - pi * pi / 36)
                         < high_prec_float("1e-35")
                     && abs(imag(unitCircleDilogarithm)
                         - high_prec_float("1.01494160640965363"))
                         < high_prec_float("1e-14"),
                 "complex dilogarithm quadrature band changed its unit-circle value");

    const high_prec_float cutProbe("1e-30");
    const auto exactCut = radcorr_detail::checkedDilogarithm(
        {high_prec_float(2), high_prec_float(0)});
    const auto aboveCut = radcorr_detail::checkedDilogarithm(
        {high_prec_float(2), cutProbe});
    const auto belowCut = radcorr_detail::checkedDilogarithm(
        {high_prec_float(2), -cutProbe});
    ok &= expect(imag(exactCut) < 0 && imag(aboveCut) > 0 && imag(belowCut) < 0
                     && abs(exactCut - belowCut) < high_prec_float("1e-25")
                     && abs(real(aboveCut) - real(belowCut))
                         < high_prec_float("1e-25")
                     && abs(imag(aboveCut) + imag(belowCut))
                         < high_prec_float("1e-25"),
                 "real-cut dilogarithm did not match the exact-real lower-lip convention");

    try {
        radcorr_detail::checkedPhiFunction(high_prec_float(1), nan, high_prec_float(2));
        ok &= expect(false, "the checked phi function silently mapped a NaN input to a finite result");
    } catch (const NumericalFailure& failure) {
        ok &= expect(failure.stage == "radcorr_calc/Phifunc input"
                         && failure.invalidTerms == std::vector<std::string>{"y"},
                     "checked-phi input failure lost its stage or term identity");
    }

    const high_prec_float equalPhi = radcorr_detail::checkedPhiFunction(
        high_prec_float(1), high_prec_float(1), high_prec_float(1));
    const high_prec_float equalPhiReference = high_prec_float(4)
        * imag(unitCircleDilogarithm) / sqrt(high_prec_float(3));
    ok &= expect(abs(equalPhi - equalPhiReference) < high_prec_float("1e-30"),
                 "equal-magnitude phi lost its high-precision Clausen relation");
    const high_prec_float nearEqualPhi = radcorr_detail::checkedPhiFunction(
        high_prec_float(1), high_prec_float(1), high_prec_float("1.0000000001"));
    ok &= expect(abs(equalPhi - nearEqualPhi) < high_prec_float("1e-8"),
                 "equal-magnitude phi branch is discontinuous with its nearby branch");

    const high_prec_float phiXY = radcorr_detail::checkedPhiFunction(
        high_prec_float(7), high_prec_float(3), high_prec_float(2));
    const high_prec_float phiYX = radcorr_detail::checkedPhiFunction(
        high_prec_float(3), high_prec_float(7), high_prec_float(2));
    ok &= expect(abs(phiXY - phiYX) < high_prec_float("1e-30"),
                 "phi violated x/y exchange symmetry");
    const high_prec_float phiScaled = radcorr_detail::checkedPhiFunction(
        high_prec_float(2), high_prec_float(3), high_prec_float(7));
    ok &= expect(abs(high_prec_float(7) * phiXY - high_prec_float(2) * phiScaled)
                     < high_prec_float("1e-29"),
                 "phi violated its scale/permutation identity");

    try {
        radcorr_detail::checkedPhiFunction(
            high_prec_float(4), high_prec_float(1), high_prec_float(1));
        ok &= expect(false, "zero-lambda phi threshold was silently accepted");
    } catch (const NumericalFailure& failure) {
        ok &= expect(failure.stage == "radcorr_calc/Phifunc branch"
                         && failure.invalidTerms == std::vector<std::string>{"zero lambda"},
                     "zero-lambda failure lost its stage or term identity");
    }

    ok &= expect(radcorr_detail::checkedLogFunctionFromSquaredMass(
                     high_prec_float(4), high_prec_float(4))
                     == high_prec_float(-4),
                 "finite squared-mass logarithm changed its reference value");
    try {
        radcorr_detail::checkedLogFunctionFromSquaredMass(
            high_prec_float(-4), high_prec_float(4));
        ok &= expect(false, "tachyonic mass square was hidden by an absolute value");
    } catch (const NumericalFailure& failure) {
        ok &= expect(failure.stage == "radcorr_calc/log input"
                         && failure.invalidTerms
                             == std::vector<std::string>{"non-positive mass squared"},
                     "tachyonic-log failure lost its stage or term identity");
    }

    ok &= expect(radcorr_detail::checkedSignedMZ2ContinuationLog(
                     high_prec_float(-4), high_prec_float(4))
                     == high_prec_float(4),
                 "signed mZ2 continuation log did not preserve the signed coordinate");
    try {
        radcorr_detail::checkedSignedMZ2ContinuationLog(
            high_prec_float(0), high_prec_float(4));
        ok &= expect(false, "signed mZ2 continuation log accepted its physical boundary");
    } catch (const NumericalFailure& failure) {
        ok &= expect(failure.stage == "radcorr_calc/signed mZ2 continuation log input"
                         && failure.invalidTerms
                             == std::vector<std::string>{"zero signed mZ squared"},
                     "signed-mZ2 boundary failure lost its stage or term identity");
    }

    radcorr_detail::requireNegligiblePhiImaginaryPart(
        {high_prec_float(2), high_prec_float("1e-12")}, high_prec_float("1e-10"));
    try {
        radcorr_detail::requireNegligiblePhiImaginaryPart(
            {high_prec_float(2), high_prec_float("1e-4")}, high_prec_float("1e-10"));
        ok &= expect(false, "large finite phi imaginary residue was silently discarded");
    } catch (const NumericalFailure& failure) {
        ok &= expect(failure.stage == "radcorr_calc/Phifunc imaginary gate"
                         && failure.invalidTerms
                             == std::vector<std::string>{"imaginary residue"},
                     "phi-imaginary failure lost its stage or term identity");
    }

    return ok ? 0 : 1;
}
