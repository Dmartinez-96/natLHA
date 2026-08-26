#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <limits>
#include <sstream>
#include <utility>
#include <boost/math/constants/constants.hpp>
#include <boost/math/quadrature/tanh_sinh.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/multiprecision/eigen.hpp>
#include <eigen3/Eigen/Dense>
#include "radcorr_calc.hpp"

using namespace boost::multiprecision;
using namespace Eigen;
typedef number<mpfr_float_backend<50>> high_prec_float;  // 50 decimal digits of precision

namespace {

#ifdef M_PI
#undef M_PI
#endif
// Keep the legacy formula spelling below while ensuring every loop factor is evaluated
// with the same 50-decimal-digit precision as the surrounding expressions.
const high_prec_float M_PI = boost::math::constants::pi<high_prec_float>();
const high_prec_float GPR_NORMALIZATION =
    sqrt(high_prec_float(3) / high_prec_float(5));
const high_prec_float SQRT_TWO = sqrt(high_prec_float(2));

std::string numericalFailureMessage(const std::string& stage,
                                    const std::vector<std::string>& invalidTerms) {
    std::ostringstream out;
    out << "invalid numerical state at " << stage << ": ";
    for (std::size_t i = 0; i < invalidTerms.size(); ++i) {
        if (i != 0) out << ", ";
        out << invalidTerms[i];
    }
    return out.str();
}

std::vector<NamedRadiativeCorrection> indexedTerms(
        const std::string& prefix,
        const std::vector<high_prec_float>& values) {
    std::vector<NamedRadiativeCorrection> terms;
    terms.reserve(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        terms.push_back({values[i], prefix + "[" + std::to_string(i) + "]"});
    }
    return terms;
}

}  // namespace

NumericalFailure::NumericalFailure(std::string failureStage,
                                   std::vector<std::string> failureTerms)
    : std::runtime_error(numericalFailureMessage(failureStage, failureTerms)),
      stage(std::move(failureStage)),
      invalidTerms(std::move(failureTerms)) {}

void requireFiniteRadiativeCorrections(
        const std::string& stage,
        const std::vector<NamedRadiativeCorrection>& terms) {
    std::vector<std::string> invalid;
    for (const auto& term : terms) {
        if (!(boost::math::isfinite)(term.value)) invalid.push_back(term.label);
    }
    if (!invalid.empty()) throw NumericalFailure(stage, invalid);
}

high_prec_float sumFiniteRadiativeCorrections(
        const std::string& stage,
        const std::vector<NamedRadiativeCorrection>& terms) {
    requireFiniteRadiativeCorrections(stage, terms);
    high_prec_float total = 0;
    for (const auto& term : terms) total += term.value;
    if (!(boost::math::isfinite)(total)) {
        throw NumericalFailure(stage, {"aggregate"});
    }
    return total;
}


namespace radcorr_detail {

namespace {

struct DilogarithmEvaluation {
    std::complex<high_prec_float> value;
    high_prec_float absoluteError;
};

DilogarithmEvaluation evaluateDilogarithm(
        const std::complex<high_prec_float>& input) {
    using Complex = std::complex<high_prec_float>;
    requireFiniteRadiativeCorrections(
        "radcorr_calc/dilog input",
        {{real(input), "real"}, {imag(input), "imaginary"}});
    const high_prec_float pi = boost::math::constants::pi<high_prec_float>();
    const high_prec_float radius = abs(input);
    const high_prec_float tolerance("1e-40");

    auto integrateUnitDisk = [&](const Complex& argument) {
        if (argument == Complex(0)) return DilogarithmEvaluation{Complex(0), 0};
        if (argument == Complex(1)) {
            return DilogarithmEvaluation{Complex(pi * pi / 6), 0};
        }

        auto seriesUnitDisk = [&](const Complex& seriesArgument) {
            const high_prec_float seriesRadius = abs(seriesArgument);
            const high_prec_float seriesGate("1e-42");
            const std::size_t maximumTerms = 100000;
            Complex power = seriesArgument;
            Complex sum(0);
            high_prec_float remainder = 0;
            const high_prec_float geometricDenominator = 1 - seriesRadius;

            if (seriesRadius <= high_prec_float("0.25")) {
                // DLMF 25.12.1, with the geometric majorant used as an explicit
                // absolute remainder bound.
                sum = power;
                for (std::size_t n = 1; n < maximumTerms; ++n) {
                    const high_prec_float nextIndex(n + 1);
                    power *= seriesArgument;
                    const high_prec_float denominator = nextIndex * nextIndex;
                    sum += power / denominator;
                    const high_prec_float tailDenominator =
                        (nextIndex + 1) * (nextIndex + 1)
                        * geometricDenominator;
                    remainder = abs(power * seriesArgument)
                        / tailDenominator;
                    if (remainder <= seriesGate) {
                        return DilogarithmEvaluation{sum, remainder};
                    }
                }
            } else {
                // Splitting one elementary term from the defining series gives
                // Li2(z) = 1 + (1-z)log(1-z)/z
                //          + sum z^k/[k^2(k+1)].
                for (std::size_t n = 1; n < maximumTerms; ++n) {
                    const high_prec_float index(n);
                    const high_prec_float denominator =
                        index * index * (index + 1);
                    sum += power / denominator;
                    power *= seriesArgument;
                    const high_prec_float nextIndex(n + 1);
                    const high_prec_float tailDenominator =
                        nextIndex * nextIndex * (nextIndex + 1)
                        * geometricDenominator;
                    remainder = abs(power)
                        / tailDenominator;
                    if (remainder <= seriesGate) {
                        const Complex value = Complex(1)
                            + (Complex(1) - seriesArgument)
                                * log(Complex(1) - seriesArgument) / seriesArgument
                            + sum;
                        return DilogarithmEvaluation{value, remainder};
                    }
                }
            }
            throw NumericalFailure(
                "radcorr_calc/dilog series", {"term budget exhausted"});
        };

        const high_prec_float argumentRadius = abs(argument);
        if (argumentRadius < high_prec_float("0.98")) {
            return seriesUnitDisk(argument);
        }
        const Complex reflected = Complex(1) - argument;
        if (real(argument) > high_prec_float("0.5")
                && abs(reflected) < high_prec_float("0.98")) {
            const DilogarithmEvaluation reflectedValue = seriesUnitDisk(reflected);
            const Complex value = Complex(pi * pi / 6) - reflectedValue.value
                - log(argument) * log(reflected);
            return DilogarithmEvaluation{value, reflectedValue.absoluteError};
        }

        // Near the unit circle, use DLMF 25.12.2 on the straight path t = s*z.
        // Arguments passed here lie in the closed unit disk, so the path does not
        // cross the principal-branch cut [1, infinity).
        boost::math::quadrature::tanh_sinh<high_prec_float> integrator(15);
        auto integrand = [&](const high_prec_float& s) {
                if (s == 0) return argument;
                return -log(Complex(1) - argument * s) / s;
        };
        high_prec_float realError = 0;
        high_prec_float imaginaryError = 0;
        const high_prec_float realValue = integrator.integrate(
            [&](const high_prec_float& s) { return real(integrand(s)); },
            high_prec_float(0), high_prec_float(1), tolerance,
            &realError);
        const high_prec_float imaginaryValue = integrator.integrate(
            [&](const high_prec_float& s) { return imag(integrand(s)); },
            high_prec_float(0), high_prec_float(1), tolerance,
            &imaginaryError);
        const Complex value(realValue, imaginaryValue);
        const high_prec_float error = sqrt(
            realError * realError + imaginaryError * imaginaryError);
        requireFiniteRadiativeCorrections(
            "radcorr_calc/dilog quadrature",
            {{real(value), "real"}, {imag(value), "imaginary"},
             {error, "error estimate"}});
        const high_prec_float errorGate = high_prec_float("1e-35")
            * max(abs(value), high_prec_float(1));
        if (error < 0 || error > errorGate) {
            throw NumericalFailure(
                "radcorr_calc/dilog quadrature", {"error estimate exceeds gate"});
        }
        return DilogarithmEvaluation{value, error};
    };

    DilogarithmEvaluation result;
    if (radius <= 1) {
        result = integrateUnitDisk(input);
    } else if (imag(input) == 0 && real(input) > 1) {
        // An exactly real argument on the principal-branch cut selects the lower-lip
        // limit, whose imaginary part is -pi*log(x). Inputs with either nonzero imaginary
        // sign take the general inversion branch below and retain that side of the cut.
        const DilogarithmEvaluation inverse =
            integrateUnitDisk(Complex(1 / real(input), 0));
        const high_prec_float logarithm = log(real(input));
        result.value = Complex(
            pi * pi / 3 - real(inverse.value) - logarithm * logarithm / 2,
            -pi * logarithm);
        result.absoluteError = inverse.absoluteError;
    } else {
        // DLMF 25.12.4 reduces the principal branch to the open unit disk.  The
        // explicit phase matches the side of the branch cut selected by the input.
        const high_prec_float radiusSquared = norm(input);
        const Complex inverse = conj(input) / radiusSquared;
        const DilogarithmEvaluation inverseResult = integrateUnitDisk(inverse);
        const high_prec_float theta = atan2(imag(input), real(input));
        const high_prec_float thetaSign = theta < 0 ? -1 : 1;
        const Complex logMinusInput(
            log(radius), thetaSign * (abs(theta) - pi));
        result.value = -inverseResult.value - Complex(pi * pi / 6)
            - logMinusInput * logMinusInput * high_prec_float("0.5");
        result.absoluteError = inverseResult.absoluteError;
    }

    requireFiniteRadiativeCorrections(
        "radcorr_calc/dilog output",
        {{real(result.value), "real"}, {imag(result.value), "imaginary"},
         {result.absoluteError, "absolute error"}});
    return result;
}

}  // namespace

std::complex<high_prec_float> checkedDilogarithm(
        const std::complex<high_prec_float>& input) {
    return evaluateDilogarithm(input).value;
}

high_prec_float checkedLogFunctionFromSquaredMass(
        const high_prec_float& massSquared,
        const high_prec_float& renormalizationScaleSquared) {
    requireFiniteRadiativeCorrections(
        "radcorr_calc/log input",
        {{massSquared, "mass squared"},
         {renormalizationScaleSquared, "renormalization scale squared"}});
    std::vector<std::string> invalidDomain;
    if (massSquared <= 0) invalidDomain.push_back("non-positive mass squared");
    if (renormalizationScaleSquared <= 0) {
        invalidDomain.push_back("non-positive renormalization scale squared");
    }
    if (!invalidDomain.empty()) {
        throw NumericalFailure("radcorr_calc/log input", invalidDomain);
    }
    const high_prec_float result = massSquared
        * (log(massSquared / renormalizationScaleSquared) - 1);
    requireFiniteRadiativeCorrections(
        "radcorr_calc/log output", {{result, "value"}});
    return result;
}

high_prec_float checkedSignedMZ2ContinuationLog(
        const high_prec_float& signedMZSquared,
        const high_prec_float& renormalizationScaleSquared) {
    requireFiniteRadiativeCorrections(
        "radcorr_calc/signed mZ2 continuation log input",
        {{signedMZSquared, "signed mZ squared"},
         {renormalizationScaleSquared, "renormalization scale squared"}});
    std::vector<std::string> invalidDomain;
    if (signedMZSquared == 0) invalidDomain.push_back("zero signed mZ squared");
    if (renormalizationScaleSquared <= 0) {
        invalidDomain.push_back("non-positive renormalization scale squared");
    }
    if (!invalidDomain.empty()) {
        throw NumericalFailure(
            "radcorr_calc/signed mZ2 continuation log input", invalidDomain);
    }
    const high_prec_float result = signedMZSquared
        * (log(abs(signedMZSquared) / renormalizationScaleSquared) - 1);
    requireFiniteRadiativeCorrections(
        "radcorr_calc/signed mZ2 continuation log output", {{result, "value"}});
    return result;
}

high_prec_float logfunc2(
        const high_prec_float& massSquared,
        const high_prec_float& renormalizationScaleSquared) {
    return checkedLogFunctionFromSquaredMass(
        massSquared, renormalizationScaleSquared);
}

void requireNegligiblePhiImaginaryPart(
        const std::complex<high_prec_float>& value,
        const high_prec_float& tolerance) {
    requireFiniteRadiativeCorrections(
        "radcorr_calc/Phifunc imaginary gate",
        {{real(value), "real"}, {imag(value), "imaginary"}, {tolerance, "tolerance"}});
    if (tolerance < 0 || abs(imag(value)) > tolerance) {
        throw NumericalFailure(
            "radcorr_calc/Phifunc imaginary gate", {"imaginary residue"});
    }
}

////////// Radiative corrections from neutralino sector //////////
high_prec_float neutralino_denom(const high_prec_float& msninp, const high_prec_float& M1val, const high_prec_float& M2val, const high_prec_float& muval, const high_prec_float& g2sqval,
                        const high_prec_float& gprsqval, const high_prec_float& vsqval, const high_prec_float& vuval, const high_prec_float& vdval, const high_prec_float& betaval) {
    /*
    Return denominator for one-loop correction
        of neutralino according to method of Ibrahim
        and Nath in PhysRevD.66.015005 (2002).

    Parameters
    ----------
    msninp : high_prec_float float.
        Neutralino un-squared mass used for evaluating results.
    //TODO: Finish this documentation

    Returns
    -------
    myden : high_prec_float float.
        Return denominator of neutralino radiative corrections.
    */
    
    // Introduce coefficients of characteristic equation for eigenvals.
    // Char. eqn. is of the form x^4 + ax^3 + bx^2 + cx + d = 0
    high_prec_float char_a = (-1.0) * (M1val + M2val);
    high_prec_float char_b = ((M1val * M2val) - (pow(muval, 2.0))
              - ((vsqval / 2.0) * (g2sqval + gprsqval)));
    high_prec_float char_c = ((pow(muval, 2.0) * (M1val + M2val))
              - (muval * vdval * vuval * (g2sqval + gprsqval))
              + ((vsqval / 2.0)
                 * ((g2sqval * M1val) + (gprsqval * M2val))));
    high_prec_float myden = (4.0 * pow(msninp, 3.0)) + (3.0 * char_a
                                         * pow(msninp, 2.0))\
        + (2.0 * char_b * msninp) + char_c;
    return myden;
}

high_prec_float neutralinouu_num(const high_prec_float& msninp, const high_prec_float& M1val, const high_prec_float& M2val, const high_prec_float& muval, const high_prec_float& g2sqval,
                        const high_prec_float& gprsqval, const high_prec_float& vsqval, const high_prec_float& vuval, const high_prec_float& vdval, const high_prec_float& betaval) {
    /*
    Return numerator for one-loop uu correction
        derivative term of neutralino.

    Parameters
    ----------
    msninp : Float.
        Neutralino un-squared mass used for evaluating results.

    */
    high_prec_float quadrterm = ((-1.0) * vuval) * (gprsqval + g2sqval);
    high_prec_float linterm = (((g2sqval * M1val) + (gprsqval * M2val)) * vuval)\
        - (muval * vdval * (g2sqval + gprsqval));
    high_prec_float constterm = muval * vdval * ((g2sqval * M1val) + (gprsqval * M2val));
    high_prec_float mynum = (quadrterm * pow(msninp, 2.0))\
        + (linterm * msninp) + constterm;
    return mynum;
}

high_prec_float neutralinodd_num(const high_prec_float& msninp, const high_prec_float& M1val, const high_prec_float& M2val, const high_prec_float& muval, const high_prec_float& g2sqval,
                        const high_prec_float& gprsqval, const high_prec_float& vsqval, const high_prec_float& vuval, const high_prec_float& vdval, const high_prec_float& betaval) {
    /*
    Return numerator for one-loop dd correction derivative term of
        neutralino.

    Parameters
    ----------
    msninp : Float.
        Neutralino squared mass used for evaluating results.

    */
    high_prec_float quadrterm = ((-1.0) * vdval) * (gprsqval + g2sqval);
    high_prec_float linterm = (((g2sqval * M1val) + (gprsqval * M2val)) * vdval)\
        - (muval * vuval * (g2sqval + gprsqval));
    high_prec_float constterm = muval * vuval * ((g2sqval * M1val) + (gprsqval * M2val));
    high_prec_float mynum = (quadrterm * pow(msninp, 2.0))\
        + (linterm * msninp) + constterm;
    return mynum;
}

high_prec_float sigmauu_neutralino(const high_prec_float& msninp, const high_prec_float& M1val, const high_prec_float& M2val, const high_prec_float& muval, const high_prec_float& g2sqval,
                          const high_prec_float& gprsqval, const high_prec_float& vsqval, const high_prec_float& vuval, const high_prec_float& vdval, const high_prec_float& betaval, const high_prec_float& myQval) {
    /*
    Return one-loop correction Sigma_u^u(neutralino).

    Parameters
    ----------
    msninp : Float.
        Neutralino un-squared mass.

    */
    high_prec_float sigma_uu_neutralino = ((1.0 / (16.0 * (pow(M_PI, 2.0)))) * msninp / vuval) \
        * ((neutralinouu_num(msninp, M1val, M2val, muval, g2sqval, gprsqval, vsqval, vuval, vdval, betaval)
            / neutralino_denom(msninp, M1val, M2val, muval, g2sqval, gprsqval, vsqval, vuval, vdval, betaval))
           * logfunc2((msninp * msninp), pow(myQval, 2.0)));
    return sigma_uu_neutralino;
}

high_prec_float sigmadd_neutralino(const high_prec_float& msninp, const high_prec_float& M1val, const high_prec_float& M2val, const high_prec_float& muval, const high_prec_float& g2sqval,
                          const high_prec_float& gprsqval, const high_prec_float& vsqval, const high_prec_float& vuval, const high_prec_float& vdval, const high_prec_float& betaval, const high_prec_float& myQval) {
    /*
    Return one-loop correction Sigma_d^d(neutralino).

    Parameters
    ----------
    msninp : Float.
        Neutralino un-squared mass.

    */
    high_prec_float sigma_dd_neutralino = ((1.0 / (16.0 * (pow(M_PI, 2.0)))) * msninp / vdval) \
        * ((neutralinodd_num(msninp, M1val, M2val, muval, g2sqval, gprsqval, vsqval, vuval, vdval, betaval)
            / neutralino_denom(msninp, M1val, M2val, muval, g2sqval, gprsqval, vsqval, vuval, vdval, betaval))
           * logfunc2((msninp * msninp), pow(myQval, 2.0)));
    return sigma_dd_neutralino;
}
////////// Radiative corrections from two-loop O(alpha_t alpha_s) sector //////////
// Corrections come from Dedes, Slavich paper, arXiv:hep-ph/0212132.
// alpha_i = y_i^2 / (4.0 * pi)

high_prec_float Deltafunc(const high_prec_float& x, const high_prec_float& y, const high_prec_float& z) {
    /*
    DOCFUNC HERE
    */
    high_prec_float mydelta = pow(x, 2.0) + pow(y, 2.0) + pow(z, 2.0)\
        - (2.0 * ((x * y) + (x * z) + (y * z)));
    return mydelta;
}

high_prec_float checkedPhiFunction(const high_prec_float& x, const high_prec_float& y, const high_prec_float& z) {
    requireFiniteRadiativeCorrections(
        "radcorr_calc/Phifunc input", {{x, "x"}, {y, "y"}, {z, "z"}});
    if (x <= 0 || y <= 0 || z <= 0) {
        throw NumericalFailure("radcorr_calc/Phifunc input", {"non-positive mass square"});
    }

    using Complex = std::complex<high_prec_float>;
    const high_prec_float pi = boost::math::constants::pi<high_prec_float>();
    high_prec_float first;
    high_prec_float second;
    high_prec_float scale;
    high_prec_float prefactor;
    // The normalization follows the Phi symmetries in Appendix A of
    // Dedes and Slavich, arXiv:hep-ph/0212132.  Non-strict comparisons make
    // equal-magnitude branches deterministic instead of returning zero.
    if (abs(z) >= abs(x) && abs(z) >= abs(y)) {
        first = x;
        second = y;
        scale = z;
        prefactor = 1;
    } else if (abs(x) >= abs(y)) {
        first = z;
        second = y;
        scale = x;
        prefactor = z / x;
    } else {
        first = z;
        second = x;
        scale = y;
        prefactor = z / y;
    }

    const Complex myu(high_prec_float(first / scale));
    const Complex myv(high_prec_float(second / scale));
    const Complex mylambda = sqrt(
        pow(Complex(1) - myu - myv, 2) - Complex(4) * myu * myv);
    if (mylambda == Complex(0)) {
        throw NumericalFailure("radcorr_calc/Phifunc branch", {"zero lambda"});
    }
    const Complex myxp = Complex(0.5) * (Complex(1) + myu - myv - mylambda);
    const Complex myxm = Complex(0.5) * (Complex(1) - myu + myv - mylambda);
    const DilogarithmEvaluation dilogPlus = evaluateDilogarithm(myxp);
    const DilogarithmEvaluation dilogMinus = evaluateDilogarithm(myxm);
    const Complex phiMultiplier = Complex(prefactor) / mylambda;
    const Complex myphi = phiMultiplier
        * (Complex(2) * log(myxp) * log(myxm)
           - log(myu) * log(myv)
           - Complex(2) * (dilogPlus.value + dilogMinus.value)
           + Complex(pi * pi / 3));
    requireFiniteRadiativeCorrections(
        "radcorr_calc/Phifunc output",
        {{real(myphi), "real"}, {imag(myphi), "imaginary"}});
    const high_prec_float dilogError = abs(phiMultiplier) * 2
        * (dilogPlus.absoluteError + dilogMinus.absoluteError);
    const high_prec_float realMagnitude = abs(real(myphi));
    const high_prec_float roundoffAllowance = 64
        * std::numeric_limits<high_prec_float>::epsilon()
        * std::max(realMagnitude, high_prec_float(1));
    requireNegligiblePhiImaginaryPart(
        myphi, dilogError + roundoffAllowance);
    return high_prec_float(real(myphi));
}

high_prec_float sigmauu_2loop(const high_prec_float& myQ, const high_prec_float& mu_wk, const high_prec_float& beta_wk, const high_prec_float& yt_wk, const high_prec_float& yc_wk, const high_prec_float& yu_wk, const high_prec_float& yb_wk, const high_prec_float& ys_wk,
                     const high_prec_float& yd_wk, const high_prec_float& ytau_wk, const high_prec_float& ymu_wk, const high_prec_float& ye_wk, const high_prec_float& g1_wk, const high_prec_float& g2_wk, const high_prec_float& g3_wk, const high_prec_float& mQ3_sq_wk,
                     const high_prec_float& mQ2_sq_wk, const high_prec_float& mQ1_sq_wk, const high_prec_float& mL3_sq_wk, const high_prec_float& mL2_sq_wk, const high_prec_float& mL1_sq_wk,
                     const high_prec_float& mU3_sq_wk, const high_prec_float& mU2_sq_wk, const high_prec_float& mU1_sq_wk, const high_prec_float& mD3_sq_wk, const high_prec_float& mD2_sq_wk, const high_prec_float& mD1_sq_wk,
                     const high_prec_float& mE3_sq_wk, const high_prec_float& mE2_sq_wk, const high_prec_float& mE1_sq_wk, const high_prec_float& M1_wk, const high_prec_float& M2_wk, const high_prec_float& M3_wk, const high_prec_float& mHu_sq_wk,
                     const high_prec_float& mHd_sq_wk, const high_prec_float& at_wk, const high_prec_float& ac_wk, const high_prec_float& au_wk, const high_prec_float& ab_wk, const high_prec_float& as_wk, const high_prec_float& ad_wk, const high_prec_float& atau_wk,
                     const high_prec_float& amu_wk, const high_prec_float& ae_wk, const high_prec_float& m_stop_1sq, const high_prec_float& m_stop_2sq, const high_prec_float& mymt, const high_prec_float& vHiggs_wk) {
    high_prec_float s2theta = 2.0 * mymt * ((at_wk / yt_wk) - (mu_wk / tan(beta_wk)))\
        / (m_stop_1sq - m_stop_2sq);
    high_prec_float s2sqtheta = pow(s2theta, 2.0);
    high_prec_float c2sqtheta = 1.0 - s2sqtheta;
    high_prec_float mglsq = pow((M3_wk), 2.0);
    high_prec_float myunits = pow(g3_wk, 2.0) * 4.0 * pow((1.0 / (16.0 * (pow(M_PI, 2.0)))), 2.0);
    high_prec_float Q_renorm_sq = pow(myQ, 2.0);
    high_prec_float myF = myunits\
        * (((4.0 * (M3_wk) * mymt / s2theta) * (1.0 + (4.0 * c2sqtheta)))
           - (((2.0 * (m_stop_1sq - m_stop_2sq)) + (4.0 * (M3_wk) * mymt / s2theta))
              * log(mglsq / Q_renorm_sq)
              * log(pow(mymt, 2.0) / Q_renorm_sq))
           - (2.0 * (4.0 - s2sqtheta) * (m_stop_1sq - m_stop_2sq))
           + ((((4.0 * m_stop_1sq * m_stop_2sq)
                - s2sqtheta * pow((m_stop_1sq + m_stop_2sq), 2.0))
               / (m_stop_1sq - m_stop_2sq))
              * (log((m_stop_1sq / Q_renorm_sq)))
              * (log(m_stop_2sq / Q_renorm_sq)))
             + ((((4.0 * (mglsq + pow(mymt, 2.0) + (2.0 * m_stop_1sq)))
                  - (s2sqtheta * ((3.0 * m_stop_1sq) + m_stop_2sq))
                  - ((16.0 * c2sqtheta * (M3_wk) * mymt * m_stop_1sq)
                     / (s2theta * (m_stop_1sq - m_stop_2sq)))
                  - (4.0 * s2theta * (M3_wk) * mymt))
                 * log((m_stop_1sq / Q_renorm_sq)))
                + ((m_stop_1sq / (m_stop_1sq - m_stop_2sq))
                   * ((s2sqtheta * (m_stop_1sq + m_stop_2sq))
                      - ((4.0 * m_stop_1sq) - (2.0 * m_stop_2sq)))
                   * pow(log((m_stop_1sq / Q_renorm_sq)), 2.0))
                + (2.0 * (m_stop_1sq - mglsq - pow(mymt, 2.0)
                        + ((M3_wk) * mymt * s2theta)
                        + ((2.0 * c2sqtheta * (M3_wk) * mymt * m_stop_1sq)
                           / (s2theta * (m_stop_1sq - m_stop_2sq))))
                   * log(mglsq * pow(mymt, 2.0)
                            / (pow(Q_renorm_sq, 2.0)))
                   * log((m_stop_1sq / Q_renorm_sq)))
                + (((4.0 * (M3_wk) * mymt * c2sqtheta * (pow(mymt, 2.0) - mglsq))
                    / (s2theta * (m_stop_1sq - m_stop_2sq)))
                   * log(pow(mymt, 2.0) / mglsq)
                   * log((m_stop_1sq / Q_renorm_sq)))
                + (((((4.0 * mglsq * pow(mymt, 2.0))
                      + (2.0 * Deltafunc(mglsq, pow(mymt, 2.0), m_stop_1sq))) / m_stop_1sq)
                    - (((2.0 * (M3_wk) * mymt * s2theta) / m_stop_1sq)
                       * (mglsq + pow(mymt, 2.0) - m_stop_1sq))
                    + ((4.0 * c2sqtheta * (M3_wk) * mymt
                        * Deltafunc(mglsq, pow(mymt, 2.0), m_stop_1sq))
                       / (s2theta * m_stop_1sq * (m_stop_1sq - m_stop_2sq))))
                   * checkedPhiFunction(mglsq, pow(mymt, 2.0), m_stop_1sq)))
             - ((((4.0 * (mglsq + pow(mymt, 2.0) + (2.0 * m_stop_2sq)))
                  - (s2sqtheta * ((3.0 * m_stop_2sq) + m_stop_1sq))
                  - ((16.0 * c2sqtheta * (M3_wk) * mymt * m_stop_2sq)
                     / (((-1.0) * s2theta) * (m_stop_2sq - m_stop_1sq)))
                  - ((-4.0) * s2theta * (M3_wk) * mymt))
                 * log(m_stop_2sq / Q_renorm_sq))
                + ((m_stop_2sq / (m_stop_2sq - m_stop_1sq))
                   * ((s2sqtheta * (m_stop_2sq + m_stop_1sq))
                      - ((4.0 * m_stop_2sq) - (2.0 * m_stop_1sq)))
                   * pow(log(m_stop_2sq / Q_renorm_sq), 2.0))
                + (2.0 * (m_stop_2sq - mglsq - pow(mymt, 2.0)
                        - ((M3_wk) * mymt * s2theta)
                        + ((2.0 * c2sqtheta * (M3_wk) * mymt * m_stop_2sq)
                           / (s2theta * (m_stop_1sq - m_stop_2sq))))
                   * log(mglsq * pow(mymt, 2.0)
                            / (pow(Q_renorm_sq, 2.0)))
                   * log(m_stop_2sq / Q_renorm_sq))
                + (((4.0 * (M3_wk) * mymt * c2sqtheta * (pow(mymt, 2.0) - mglsq))
                    / (s2theta * (m_stop_1sq - m_stop_2sq)))
                   * log(pow(mymt, 2.0) / mglsq)
                   * log(m_stop_2sq / Q_renorm_sq))
                + (((((4.0 * mglsq * pow(mymt, 2.0))
                      + (2.0 * Deltafunc(mglsq, pow(mymt, 2.0), m_stop_2sq))) / m_stop_2sq)
                    - ((((-2.0) * (M3_wk) * mymt * s2theta) / m_stop_2sq)
                       * (mglsq + pow(mymt, 2.0) - m_stop_2sq))
                    + ((4.0 * c2sqtheta * (M3_wk) * mymt
                        * Deltafunc(mglsq, pow(mymt, 2.0), m_stop_2sq))
                       / (s2theta * m_stop_2sq * (m_stop_1sq - m_stop_2sq))))
                   * checkedPhiFunction(mglsq, pow(mymt, 2.0), m_stop_2sq))));
    high_prec_float myG = myunits\
        * ((5.0 * (M3_wk) * s2theta * (m_stop_1sq - m_stop_2sq) / mymt)
           - (10.0 * (m_stop_1sq + m_stop_2sq - (2.0 * pow(mymt, 2.0))))
           - (4.0 * mglsq) + ((12.0 * pow(mymt, 2.0))
                            * (pow(log(pow(mymt, 2.0) / Q_renorm_sq), 2.0)
                               - (2.0 * log(pow(mymt, 2.0) / Q_renorm_sq))))
           + (((4.0 * mglsq) - (((M3_wk) * s2theta / mymt)
                              * (m_stop_1sq - m_stop_2sq)))
              * log(mglsq / Q_renorm_sq) * log(pow(mymt, 2.0) / Q_renorm_sq))
           + (s2sqtheta * (m_stop_1sq + m_stop_2sq)
              * log((m_stop_1sq / Q_renorm_sq))
              * log(m_stop_2sq / Q_renorm_sq))
           + ((((4.0 * (mglsq + pow(mymt, 2.0) + (2.0 * m_stop_1sq)))
                + (s2sqtheta * (m_stop_1sq - m_stop_2sq))
                - ((4.0 * (M3_wk) * s2theta / mymt) * (pow(mymt, 2.0) + m_stop_1sq)))
               * log((m_stop_1sq / Q_renorm_sq)))
              + ((((M3_wk) * s2theta * ((5.0 * pow(mymt, 2.0)) - mglsq + m_stop_1sq)
                   / mymt)
                  - (2.0 * (mglsq + 2.0 * pow(mymt, 2.0))))
                 * log(pow(mymt, 2.0) / Q_renorm_sq)
                 * log((m_stop_1sq / Q_renorm_sq)))
              + ((((M3_wk) * s2theta * (mglsq - pow(mymt, 2.0) + m_stop_1sq) / mymt)
                  - (2.0 * mglsq))
                 * log(mglsq / Q_renorm_sq)
                 * log((m_stop_1sq / Q_renorm_sq)))
              - ((2.0 + s2sqtheta) * m_stop_1sq
                 * pow(log((m_stop_1sq / Q_renorm_sq)), 2.0))
              + (((2.0 * mglsq * (mglsq + pow(mymt, 2.0) - m_stop_1sq
                                - (2.0 * (M3_wk) * mymt * s2theta)) / m_stop_1sq)
                  + (((M3_wk) * s2theta / (mymt * m_stop_1sq))
                     * Deltafunc(mglsq, pow(mymt, 2.0), m_stop_1sq)))
                 * checkedPhiFunction(mglsq, pow(mymt, 2.0), m_stop_1sq)))
           + ((((4.0 * (mglsq + pow(mymt, 2.0) + (2.0 * m_stop_2sq)))
                + (s2sqtheta * (m_stop_2sq - m_stop_1sq))
                - (((-4.0) * (M3_wk) * s2theta / mymt) * (pow(mymt, 2.0) + m_stop_2sq)))
               * log(m_stop_2sq / Q_renorm_sq))
              + ((((-1.0) * (M3_wk) * s2theta * ((5.0 * pow(mymt, 2.0)) - mglsq + m_stop_2sq)
                   / mymt)
                  - (2.0 * (mglsq + 2.0 * pow(mymt, 2.0))))
                 * log(pow(mymt, 2.0) / Q_renorm_sq)
                 * log(m_stop_2sq / Q_renorm_sq))
              + ((((-1.0) * (M3_wk) * s2theta * (mglsq - pow(mymt, 2.0) + m_stop_2sq)
                   / mymt)
                  - (2.0 * mglsq))
                 * log(mglsq / Q_renorm_sq)
                 * log(m_stop_2sq / Q_renorm_sq))
              - ((2.0 + s2sqtheta) * m_stop_2sq
                 * pow(log(m_stop_2sq / Q_renorm_sq), 2.0))
              + (((2.0 * mglsq
                   * (mglsq + pow(mymt, 2.0) - m_stop_2sq
                      + (2.0 * (M3_wk) * mymt * s2theta)) / m_stop_2sq)
                  + (((M3_wk) * (-1.0) * s2theta / (mymt * m_stop_2sq))
                     * Deltafunc(mglsq, pow(mymt, 2.0), m_stop_2sq)))
                 * checkedPhiFunction(mglsq, pow(mymt, 2.0), m_stop_2sq))));
    high_prec_float sinsqb = pow(sin(beta_wk), 2.0);
    high_prec_float mysigmauu_2loop = ((mymt * (at_wk / yt_wk) * s2theta * myF)
                       + 2.0 * pow(mymt, 2.0) * myG)\
        / (pow((vHiggs_wk), 2.0) * sinsqb);
    return real(mysigmauu_2loop);
}

high_prec_float sigmadd_2loop(const high_prec_float& myQ, const high_prec_float& mu_wk, const high_prec_float& beta_wk, const high_prec_float& yt_wk, const high_prec_float& yc_wk, const high_prec_float& yu_wk, const high_prec_float& yb_wk, const high_prec_float& ys_wk,
                     const high_prec_float& yd_wk, const high_prec_float& ytau_wk, const high_prec_float& ymu_wk, const high_prec_float& ye_wk, const high_prec_float& g1_wk, const high_prec_float& g2_wk, const high_prec_float& g3_wk, const high_prec_float& mQ3_sq_wk,
                     const high_prec_float& mQ2_sq_wk, const high_prec_float& mQ1_sq_wk, const high_prec_float& mL3_sq_wk, const high_prec_float& mL2_sq_wk, const high_prec_float& mL1_sq_wk,
                     const high_prec_float& mU3_sq_wk, const high_prec_float& mU2_sq_wk, const high_prec_float& mU1_sq_wk, const high_prec_float& mD3_sq_wk, const high_prec_float& mD2_sq_wk, const high_prec_float& mD1_sq_wk,
                     const high_prec_float& mE3_sq_wk, const high_prec_float& mE2_sq_wk, const high_prec_float& mE1_sq_wk, const high_prec_float& M1_wk, const high_prec_float& M2_wk, const high_prec_float& M3_wk, const high_prec_float& mHu_sq_wk,
                     const high_prec_float& mHd_sq_wk, const high_prec_float& at_wk, const high_prec_float& ac_wk, const high_prec_float& au_wk, const high_prec_float& ab_wk, const high_prec_float& as_wk, const high_prec_float& ad_wk, const high_prec_float& atau_wk,
                     const high_prec_float& amu_wk, const high_prec_float& ae_wk, const high_prec_float& m_stop_1sq, const high_prec_float& m_stop_2sq, const high_prec_float& mymt, const high_prec_float& vHiggs_wk) {
    high_prec_float Q_renorm_sq = pow(myQ, 2.0);
    high_prec_float s2theta = (2.0 * mymt * ((at_wk / yt_wk)
                           - (mu_wk / tan(beta_wk))))\
        / (m_stop_1sq - m_stop_2sq);
    high_prec_float s2sqtheta = pow(s2theta, 2.0);
    high_prec_float c2sqtheta = 1.0 - s2sqtheta;
    high_prec_float mglsq = pow(M3_wk, 2.0);
    high_prec_float myunits = pow(g3_wk, 2.0) * 4\
        / pow((16.0 * pow(M_PI, 2.0)), 2.0);
    high_prec_float myF = myunits\
        * ((4.0 * (M3_wk) * mymt / s2theta) * (1.0 + 4.0 * c2sqtheta)
           - (((2.0 * (m_stop_1sq - m_stop_2sq))
              + (4.0 * (M3_wk) * mymt / s2theta))
              * log(mglsq / Q_renorm_sq)
              * log(pow(mymt, 2.0) / Q_renorm_sq))
           - (2.0 * (4.0 - s2sqtheta)
              * (m_stop_1sq - m_stop_2sq))
           + ((((4.0 * m_stop_1sq * m_stop_2sq)
                - s2sqtheta * pow((m_stop_1sq + m_stop_2sq), 2.0))
               / (m_stop_1sq - m_stop_2sq))
              * (log((m_stop_1sq / Q_renorm_sq)))
              * (log(m_stop_2sq / Q_renorm_sq)))
           + ((((4.0 * (mglsq + pow(mymt, 2.0) + (2.0 * m_stop_1sq)))
               - (s2sqtheta * ((3.0 * m_stop_1sq) + m_stop_2sq))
               - ((16.0 * c2sqtheta * (M3_wk) * mymt * m_stop_1sq)
                  / (s2theta * (m_stop_1sq - m_stop_2sq)))
               - (4.0 * s2theta * (M3_wk) * mymt))
               * log((m_stop_1sq / Q_renorm_sq)))
              + ((m_stop_1sq / (m_stop_1sq - m_stop_2sq))
                 * ((s2sqtheta * (m_stop_1sq + m_stop_2sq))
                    - ((4.0 * m_stop_1sq) - (2.0 * m_stop_2sq)))
                 * pow(log((m_stop_1sq / Q_renorm_sq)), 2.0))
              + (2.0 * (m_stop_1sq - mglsq - pow(mymt, 2.0)
                      + ((M3_wk) * mymt * s2theta)
                      + ((2.0 * c2sqtheta * (M3_wk) * mymt * m_stop_1sq)
                         / (s2theta * (m_stop_1sq - m_stop_2sq))))
                 * log(mglsq * pow(mymt, 2.0)
                          / (pow(Q_renorm_sq, 2.0)))
                 * log((m_stop_1sq / Q_renorm_sq)))
              + (((4.0 * (M3_wk) * mymt * c2sqtheta * (pow(mymt, 2.0) - mglsq))
                  / (s2theta * (m_stop_1sq - m_stop_2sq)))
                 * log(pow(mymt, 2.0) / mglsq)
                 * log((m_stop_1sq / Q_renorm_sq)))
              + (((((4.0 * mglsq * pow(mymt, 2.0))
                    + (2.0 * Deltafunc(mglsq, pow(mymt, 2.0), m_stop_1sq))) / m_stop_1sq)
                  - (((2.0 * (M3_wk) * mymt * s2theta) / m_stop_1sq)
                     * (mglsq + pow(mymt, 2.0) - m_stop_1sq))
                  + ((4.0 * c2sqtheta * (M3_wk) * mymt
                      * Deltafunc(mglsq, pow(mymt, 2.0), m_stop_1sq))
                     / (s2theta * m_stop_1sq * (m_stop_1sq - m_stop_2sq))))
                 * checkedPhiFunction(mglsq, pow(mymt, 2.0), m_stop_1sq)))
           - ((((4.0 * (mglsq + pow(mymt, 2.0) + (2.0 * m_stop_2sq)))
               - (s2sqtheta * ((3.0 * m_stop_2sq) + m_stop_1sq))
               - ((16.0 * c2sqtheta * (M3_wk) * mymt * m_stop_2sq)
                  / (((-1.0) * s2theta) * (m_stop_2sq - m_stop_1sq)))
               - ((-4.0) * s2theta * (M3_wk) * mymt))
               * log(m_stop_2sq / Q_renorm_sq))
              + ((m_stop_2sq / (m_stop_2sq - m_stop_1sq))
                 * ((s2sqtheta * (m_stop_2sq + m_stop_1sq))
                    - ((4.0 * m_stop_2sq) - (2.0 * m_stop_1sq)))
                 * pow(log(m_stop_2sq / Q_renorm_sq), 2.0))
              + (2.0 * (m_stop_2sq - mglsq - pow(mymt, 2.0)
                      - ((M3_wk) * mymt * s2theta)
                      + ((2.0 * c2sqtheta * (M3_wk) * mymt * m_stop_2sq)
                         / (s2theta * (m_stop_1sq - m_stop_2sq))))
                 * log(mglsq * pow(mymt, 2.0)
                          / (pow(Q_renorm_sq, 2.0)))
                 * log(m_stop_2sq / Q_renorm_sq))
              + (((4.0 * (M3_wk) * mymt * c2sqtheta * (pow(mymt, 2.0) - mglsq))
                  / (s2theta * (m_stop_1sq - m_stop_2sq)))
                 * log(pow(mymt, 2.0) / mglsq)
                 * log(m_stop_2sq / Q_renorm_sq))
              + (((((4.0 * mglsq * pow(mymt, 2.0))
                    + (2.0 * Deltafunc(mglsq, pow(mymt, 2.0), m_stop_2sq))) / m_stop_2sq)
                  - ((((-2.0) * (M3_wk) * mymt * s2theta) / m_stop_2sq)
                     * (mglsq + pow(mymt, 2.0) - m_stop_2sq))
                  + ((4.0 * c2sqtheta * (M3_wk) * mymt
                      * Deltafunc(mglsq, pow(mymt, 2.0), m_stop_2sq))
                     / (s2theta * m_stop_2sq * (m_stop_1sq - m_stop_2sq))))
                 * checkedPhiFunction(mglsq, pow(mymt, 2.0), m_stop_2sq))));
    high_prec_float cossqb = (pow(cos(beta_wk), 2.0));
    high_prec_float mysigmadd_2loop = (mymt * (-1.0 * mu_wk) * (1.0 / tan(beta_wk))
                       * s2theta * myF)\
        / (pow((vHiggs_wk), 2.0) * cossqb);
    return real(mysigmadd_2loop);
}

high_prec_float dew_funcu(const high_prec_float& inp, const high_prec_float& tangentbeta) {
    /*
    Compute individual one-loop DEW contributions from Sigma_u^u.

    Parameters
    ----------
    inp : One-loop correction or Higgs to be inputted into the DEW eval.

    */
    high_prec_float mycontribuu = ((-1.0) * inp * pow(tangentbeta, 2.0) / (pow(tangentbeta, 2.0) - 1.0));
    return mycontribuu;
}

high_prec_float dew_funcd(const high_prec_float& inp, const high_prec_float& tangentbeta) {
    /*
    Compute individual one-loop DEW contributions from Sigma_d^d.

    Parameters
    ----------
    inp : One-loop correction or Higgs to be inputted into the DEW eval.

    */
    high_prec_float mycontribdd = (inp / (pow(tangentbeta, 2.0) - 1.0));
    return mycontribdd;
}

}  // namespace radcorr_detail

std::vector<high_prec_float> radcorr_calc(std::vector<high_prec_float> weak_boundary_conditions, high_prec_float myQ, high_prec_float mymZsq) {
    /*
    DOCSTRING HERE
    */
    using namespace radcorr_detail;
    requireFiniteRadiativeCorrections(
        "radcorr_calc/mZ2 continuation input", {{mymZsq, "signed mZ squared"}});
    if (mymZsq == 0) {
        throw NumericalFailure(
            "radcorr_calc/mZ2 continuation input", {"zero signed mZ squared"});
    }
    // The bounded root solver searches a signed mathematical mZ^2 domain.  The Higgs vev
    // uses its magnitude while neutral-Higgs and Z-sector expressions retain the sign;
    // invalid continuation points still fail through the checked terms below.
    const high_prec_float mymZ = sqrt(abs(mymZsq));
    // Gauge couplings
    const high_prec_float g1_wk = weak_boundary_conditions[0];
    const high_prec_float g2_wk = weak_boundary_conditions[1];
    const high_prec_float g3_wk = weak_boundary_conditions[2];
    // Higgs parameters
    const high_prec_float beta_wk = atan(weak_boundary_conditions[43]);
    const high_prec_float mu_wk = weak_boundary_conditions[6];
    const high_prec_float mu_wk_sq = pow(mu_wk, 2.0);
    // Yukawas
    const high_prec_float yt_wk = weak_boundary_conditions[7];
    const high_prec_float yc_wk = weak_boundary_conditions[8];
    const high_prec_float yu_wk = weak_boundary_conditions[9];
    const high_prec_float yb_wk = weak_boundary_conditions[10];
    const high_prec_float ys_wk = weak_boundary_conditions[11];
    const high_prec_float yd_wk = weak_boundary_conditions[12];
    const high_prec_float ytau_wk = weak_boundary_conditions[13];
    const high_prec_float ymu_wk = weak_boundary_conditions[14];
    const high_prec_float ye_wk = weak_boundary_conditions[15];
    // Soft trilinears
    const high_prec_float at_wk = weak_boundary_conditions[16];
    const high_prec_float ac_wk = weak_boundary_conditions[17];
    const high_prec_float au_wk = weak_boundary_conditions[18];
    const high_prec_float ab_wk = weak_boundary_conditions[19];
    const high_prec_float as_wk = weak_boundary_conditions[20];
    const high_prec_float ad_wk = weak_boundary_conditions[21];
    const high_prec_float atau_wk = weak_boundary_conditions[22];
    const high_prec_float amu_wk = weak_boundary_conditions[23];
    const high_prec_float ae_wk = weak_boundary_conditions[24];
    // Gaugino masses
    const high_prec_float M1_wk = weak_boundary_conditions[3];
    const high_prec_float M2_wk = weak_boundary_conditions[4];
    const high_prec_float M3_wk = weak_boundary_conditions[5];
    // Soft mass dim. 2 terms
    const high_prec_float mHu_sq_wk = weak_boundary_conditions[25];
    const high_prec_float mHd_sq_wk = weak_boundary_conditions[26];
    const high_prec_float mQ1_sq_wk = weak_boundary_conditions[27];
    const high_prec_float mQ2_sq_wk = weak_boundary_conditions[28];
    const high_prec_float mQ3_sq_wk = weak_boundary_conditions[29];
    const high_prec_float mL1_sq_wk = weak_boundary_conditions[30];
    const high_prec_float mL2_sq_wk = weak_boundary_conditions[31];
    const high_prec_float mL3_sq_wk = weak_boundary_conditions[32];
    const high_prec_float mU1_sq_wk = weak_boundary_conditions[33];
    const high_prec_float mU2_sq_wk = weak_boundary_conditions[34];
    const high_prec_float mU3_sq_wk = weak_boundary_conditions[35];
    const high_prec_float mD1_sq_wk = weak_boundary_conditions[36];
    const high_prec_float mD2_sq_wk = weak_boundary_conditions[37];
    const high_prec_float mD3_sq_wk = weak_boundary_conditions[38];
    const high_prec_float mE1_sq_wk = weak_boundary_conditions[39];
    const high_prec_float mE2_sq_wk = weak_boundary_conditions[40];
    const high_prec_float mE3_sq_wk = weak_boundary_conditions[41];
    const high_prec_float b_wk = weak_boundary_conditions[42];
    high_prec_float gpr_wk = g1_wk * GPR_NORMALIZATION;
    // // cout << "gpr_wk: " << gpr_wk << endl;
    high_prec_float gpr_sq = pow(gpr_wk, 2.0);
    // // cout << "gpr_sq: " << gpr_sq << endl;
    high_prec_float g2_sq = pow(g2_wk, 2.0);
    // // cout << "g2_sq: " << g2_sq << endl;
    // // cout << "mu_wk_sq: " << mu_wk_sq << endl;
    high_prec_float vHiggs_wk = mymZ * sqrt(2.0 / (gpr_sq + g2_sq));
    high_prec_float sinsqb = pow(sin(beta_wk), 2.0);
    // // cout << "sinsqb: " << sinsqb << endl;
    high_prec_float cossqb = pow(cos(beta_wk), 2.0);
    // // cout << "cossqb: " << cossqb << endl;
    high_prec_float vu = vHiggs_wk * sqrt(sinsqb);
    // // cout << "vu: " << vu << endl;
    high_prec_float vd = vHiggs_wk * sqrt(cossqb);
    // // cout << "vd: " << vd << endl;
    high_prec_float vu_sq = pow(vu, 2.0);
    // // cout << "vu_sq: " << vu_sq << endl;
    high_prec_float vd_sq = pow(vd, 2.0);
    // // cout << "vd_sq: " << vd_sq << endl;
    high_prec_float v_sq = pow(vHiggs_wk, 2.0);
    // // cout << "v_sq: " << v_sq << endl;
    high_prec_float tan_th_w = gpr_wk / g2_wk;
    // // cout << "tan_th_w: " << tan_th_w << endl;
    high_prec_float theta_w = atan(tan_th_w);
    // // cout << "theta_w: " << theta_w << endl;
    high_prec_float sinsq_th_w = pow(sin(theta_w), 2.0);
    // // cout << "sinsq_th_w: " << sinsq_th_w << endl;
    high_prec_float cos2b = cos(2.0 * beta_wk);
    // // cout << "cos2b: " << cos2b << endl;
    high_prec_float sin2b = sin(2.0 * beta_wk);
    // // cout << "sin2b: " << sin2b << endl;
    high_prec_float gz_sq = (pow(g2_wk, 2.0) + pow(gpr_wk, 2.0)) / 8.0;
    // // cout << "gz_sq: " << gz_sq << endl;

    ////////// Mass relations: //////////

    // W-boson tree-level running squared mass
    const high_prec_float m_w_sq = (pow(g2_wk, 2.0) / 2.0) * v_sq;

    // Z-boson tree-level running squared mass
    const high_prec_float mz_q_sq = mymZsq;// v_sq* ((pow(g2_wk, 2.0) + pow(gpr_wk, 2.0)) / 2.0);

    // Higgs psuedoscalar tree-level running squared mass
    const high_prec_float mA0sq = 2.0 * mu_wk_sq + mHu_sq_wk + mHd_sq_wk;

    // Top quark tree-level running mass
    const high_prec_float mymt = yt_wk * vu;
    const high_prec_float mymtsq = pow(mymt, 2.0);

    // Bottom quark tree-level running mass
    const high_prec_float mymb = yb_wk * vd;
    const high_prec_float mymbsq = pow(mymb, 2.0);

    // Tau tree-level running mass
    const high_prec_float mymtau = ytau_wk * vd;
    const high_prec_float mymtausq = pow(mymtau, 2.0);

    // Charm quark tree-level running mass
    const high_prec_float mymc = yc_wk * vu;
    const high_prec_float mymcsq = pow(mymc, 2.0);

    // Strange quark tree-level running mass
    const high_prec_float myms = ys_wk * vd;
    const high_prec_float mymssq = pow(myms, 2.0);

    // Muon tree-level running mass
    const high_prec_float mymmu = ymu_wk * vd;
    const high_prec_float mymmusq = pow(mymmu, 2.0);

    // Up quark tree-level running mass
    const high_prec_float mymu = yu_wk * vu;
    const high_prec_float mymusq = pow(mymu, 2.0);

    // Down quark tree-level running mass
    const high_prec_float mymd = yd_wk * vd;
    const high_prec_float mymdsq = pow(mymd, 2.0);

    // Electron tree-level running mass
    const high_prec_float myme = ye_wk * vd;
    const high_prec_float mymesq = pow(myme, 2.0);

    // Sneutrino running masses
    const high_prec_float mselecneutsq = mL1_sq_wk + (0.25 * (gpr_sq + g2_sq) * (vd_sq - vu_sq));
    const high_prec_float msmuneutsq = mL2_sq_wk + (0.25 * (gpr_sq + g2_sq) * (vd_sq - vu_sq));
    const high_prec_float mstauneutsq = mL3_sq_wk + (0.25 * (gpr_sq + g2_sq) * (vd_sq - vu_sq));

    // Tree-level charged Higgs running squared mass.
    const high_prec_float mH_pmsq = mA0sq + m_w_sq;

    // Electroweak D-term contributions to the squark and slepton squared masses.
    //
    // From S. P. Martin's SUSY primer, eq. (defDeltaphi) at primerv7.tex:10638-10643:
    //     Delta_phi = (1/2) * (T3_phi * g^2 - Y_phi * g'^2) * (vd^2 - vu^2)
    // where T3, Y and Q belong to the LEFT-HANDED chiral supermultiplet containing phi, in the
    // convention Q = T3 + Y. Written over (vu^2 - vd^2), as below, every coefficient flips sign.
    //
    // The hypercharges that follow from Q = T3 + Y, each checked against the electric charges
    // they must reproduce: the quark doublet has Y = 1/6 (Q_u = 1/2 + 1/6 = 2/3,
    // Q_d = -1/2 + 1/6 = -1/3), u-bar has Y = -2/3, d-bar has Y = 1/3, the lepton doublet has
    // Y = -1/2 (Q_nu = 0, Q_e = -1), and e-bar has Y = 1. So:
    //     suL   T3 = +1/2, Y =  1/6   ->  (vu^2 - vd^2) * ( g'^2/12 - g^2/4 )
    //     suR   T3 =    0, Y = -2/3   -> -(vu^2 - vd^2) * ( g'^2/3 )
    //     sdL   T3 = -1/2, Y =  1/6   ->  (vu^2 - vd^2) * ( g'^2/12 + g^2/4 )
    //     sdR   T3 =    0, Y =  1/3   ->  (vu^2 - vd^2) * ( g'^2/6 )
    //     seL   T3 = -1/2, Y = -1/2   ->  (vu^2 - vd^2) * ( g^2/4 - g'^2/4 )
    //     seR   T3 =    0, Y =  1     ->  (vu^2 - vd^2) * ( g'^2/2 )
    //
    // `gpr` is the UNNORMALIZED g', formed a few lines above as g1 * sqrt(3/5), so these are
    // the primer's own g' and no further rescaling applies.
    high_prec_float Delta_suL = (pow(vu, 2.0) - pow(vd, 2.0)) * ((gpr_sq / 12.0) - (g2_sq / 4.0));
    high_prec_float Delta_suR = (-1.0) * (pow(vu, 2.0) - pow(vd, 2.0)) * (gpr_sq / 3.0);
    high_prec_float Delta_sdL = (pow(vu, 2.0) - pow(vd, 2.0)) * ((gpr_sq / 12.0) + (g2_sq / 4.0));
    high_prec_float Delta_sdR = (pow(vu, 2.0) - pow(vd, 2.0)) * (gpr_sq / 6.0);
    high_prec_float Delta_seL = (pow(vu, 2.0) - pow(vd, 2.0)) * ((g2_sq / 4.0) - (gpr_sq / 4.0));
    high_prec_float Delta_seR = (pow(vu, 2.0) - pow(vd, 2.0)) * (gpr_sq / 2.0);

    // Up-type squark mass eigenstate eigenvalues
    high_prec_float m_stop_1sq = (0.5)\
        * (mQ3_sq_wk + mU3_sq_wk + (2.0 * mymtsq) + Delta_suL + Delta_suR
           - sqrt(pow((mQ3_sq_wk + Delta_suL - mU3_sq_wk - Delta_suR), 2.0)
                  + (4.0 * pow(((at_wk * vu) - (mu_wk * yt_wk * vd)), 2.0))));
    high_prec_float m_stop_2sq = (0.5)\
        * (mQ3_sq_wk + mU3_sq_wk + (2.0 * mymtsq) + Delta_suL + Delta_suR
           + sqrt(pow((mQ3_sq_wk + Delta_suL - mU3_sq_wk - Delta_suR), 2.0)
                  + (4.0 * pow(((at_wk * vu) - (mu_wk * yt_wk * vd)), 2.0))));
    high_prec_float m_scharm_1sq = (0.5)\
        * (mQ2_sq_wk + mU2_sq_wk + (2.0 * mymcsq) + Delta_suL + Delta_suR
           - sqrt(pow((mQ2_sq_wk + Delta_suL - mU2_sq_wk - Delta_suR), 2.0)
                  + (4.0 * pow(((ac_wk * vu) - (mu_wk * yc_wk * vd)), 2.0))));
    high_prec_float m_scharm_2sq = (0.5)\
        * (mQ2_sq_wk + mU2_sq_wk + (2.0 * mymcsq) + Delta_suL + Delta_suR
           + sqrt(pow((mQ2_sq_wk + Delta_suL - mU2_sq_wk - Delta_suR), 2.0)
                  + (4.0 * pow(((ac_wk * vu) - (mu_wk * yc_wk * vd)), 2.0))));
    high_prec_float m_sup_1sq = (0.5)\
        * (mQ1_sq_wk + mU1_sq_wk + (2.0 * mymusq) + Delta_suL + Delta_suR
           - sqrt(pow((mQ1_sq_wk + Delta_suL - mU1_sq_wk - Delta_suR), 2.0)
                  + (4.0 * pow(((au_wk * vu) - (mu_wk * yu_wk * vd)), 2.0))));
    high_prec_float m_sup_2sq = (0.5)\
        * (mQ1_sq_wk + mU1_sq_wk + (2.0 * mymusq) + Delta_suL + Delta_suR
           + sqrt(pow((mQ1_sq_wk + Delta_suL - mU1_sq_wk - Delta_suR), 2.0)
                  + (4.0 * pow(((au_wk * vu) - (mu_wk * yu_wk * vd)), 2.0))));

    // Down-type squark mass eigenstate eigenvalues
    high_prec_float m_sbot_1sq = (0.5)\
        * (mQ3_sq_wk + mD3_sq_wk + (2.0 * mymbsq) + Delta_sdL + Delta_sdR
           - sqrt(pow((mQ3_sq_wk + Delta_sdL - mD3_sq_wk - Delta_sdR), 2.0)
                  + (4.0 * pow(((ab_wk * vd) - (mu_wk * yb_wk * vu)), 2.0))));
    high_prec_float m_sbot_2sq = (0.5)\
        * (mQ3_sq_wk + mD3_sq_wk + (2.0 * mymbsq) + Delta_sdL + Delta_sdR
           + sqrt(pow((mQ3_sq_wk + Delta_sdL - mD3_sq_wk - Delta_sdR), 2.0)
                  + (4.0 * pow(((ab_wk * vd) - (mu_wk * yb_wk * vu)), 2.0))));
    high_prec_float m_sstrange_1sq = (0.5)\
        * (mQ2_sq_wk + mD2_sq_wk + (2.0 * mymssq) + Delta_sdL + Delta_sdR
           - sqrt(pow((mQ2_sq_wk + Delta_sdL - mD2_sq_wk - Delta_sdR), 2.0)
                  + (4.0 * pow(((as_wk * vd) - (mu_wk * ys_wk * vu)), 2.0))));
    high_prec_float m_sstrange_2sq = (0.5)\
        * (mQ2_sq_wk + mD2_sq_wk + (2.0 * mymssq) + Delta_sdL + Delta_sdR
           + sqrt(pow((mQ2_sq_wk + Delta_sdL - mD2_sq_wk - Delta_sdR), 2.0)
                  + (4.0 * pow(((as_wk * vd) - (mu_wk * ys_wk * vu)), 2.0))));
    high_prec_float m_sdown_1sq = (0.5)\
        * (mQ1_sq_wk + mD1_sq_wk + (2.0 * mymdsq) + Delta_sdL + Delta_sdR
           - sqrt(pow((mQ1_sq_wk + Delta_sdL - mD1_sq_wk - Delta_sdR), 2.0)
                  + (4.0 * pow(((ad_wk * vd) - (mu_wk * yd_wk * vu)), 2.0))));
    high_prec_float m_sdown_2sq = (0.5)\
        * (mQ1_sq_wk + mD1_sq_wk + (2.0 * mymdsq) + Delta_sdL + Delta_sdR
           + sqrt(pow((mQ1_sq_wk + Delta_sdL - mD1_sq_wk - Delta_sdR), 2.0)
                  + (4.0 * pow(((ad_wk * vd) - (mu_wk * yd_wk * vu)), 2.0))));

    // Slepton mass eigenstate eigenvalues
    high_prec_float m_stau_1sq = (0.5)\
        * (mL3_sq_wk + mE3_sq_wk + (2.0 * mymtausq) + Delta_seL + Delta_seR
           - sqrt(pow((mL3_sq_wk + Delta_seL - mE3_sq_wk - Delta_seR), 2.0)
                  + (4.0 * pow(((atau_wk * vd) - (mu_wk * ytau_wk * vu)), 2.0))));
    high_prec_float m_stau_2sq = (0.5)\
        * (mL3_sq_wk + mE3_sq_wk + (2.0 * mymtausq) + Delta_seL + Delta_seR
           + sqrt(pow((mL3_sq_wk + Delta_seL - mE3_sq_wk - Delta_seR), 2.0)
                  + (4.0 * pow(((atau_wk * vd) - (mu_wk * ytau_wk * vu)), 2.0))));
    high_prec_float m_smu_1sq = (0.5)\
        * (mL2_sq_wk + mE2_sq_wk + (2.0 * mymmusq) + Delta_seL + Delta_seR
           - sqrt(pow((mL2_sq_wk + Delta_seL - mE2_sq_wk - Delta_seR), 2.0)
                  + (4.0 * pow(((amu_wk * vd) - (mu_wk * ymu_wk * vu)), 2.0))));
    high_prec_float m_smu_2sq = (0.5)\
        * (mL2_sq_wk + mE2_sq_wk + (2.0 * mymmusq) + Delta_seL + Delta_seR
           + sqrt(pow((mL2_sq_wk + Delta_seL - mE2_sq_wk - Delta_seR), 2.0)
                  + (4.0 * pow(((amu_wk * vd) - (mu_wk * ymu_wk * vu)), 2.0))));
    high_prec_float m_se_1sq = (0.5)\
        * (mL1_sq_wk + mE1_sq_wk + (2.0 * mymesq) + Delta_seL + Delta_seR
           - sqrt(pow((mL1_sq_wk + Delta_seL - mE1_sq_wk - Delta_seR), 2.0)
                  + (4.0 * pow(((ae_wk * vd) - (mu_wk * ye_wk * vu)), 2.0))));
    high_prec_float m_se_2sq = (0.5)\
        * (mL1_sq_wk + mE1_sq_wk + (2.0 * mymesq) + Delta_seL + Delta_seR
           + sqrt(pow((mL1_sq_wk + Delta_seL - mE1_sq_wk - Delta_seR), 2.0)
                  + (4.0 * pow(((ae_wk * vd) - (mu_wk * ye_wk * vu)), 2.0))));

    // Chargino mass eigenstate eigenvalues
    high_prec_float msC1sq = (0.5)\
        * (pow(M2_wk, 2.0) + mu_wk_sq + (2.0 * m_w_sq)
           - sqrt(pow(pow(M2_wk, 2.0) + mu_wk_sq
                      + (2.0 * m_w_sq), 2.0)
                  - (4.0 * pow((mu_wk * M2_wk)
                               - (m_w_sq * sin2b), 2.0))));
    high_prec_float msC2sq = (0.5)\
        * (pow(M2_wk, 2.0) + mu_wk_sq + (2.0 * m_w_sq)
           + sqrt(pow(pow(M2_wk, 2.0) + mu_wk_sq
                      + (2.0 * m_w_sq), 2.0)
                  - (4.0 * pow((mu_wk * M2_wk)
                               - (m_w_sq * sin2b), 2.0))));

    // Neutralino mass eigenstate eigenvalues
    Eigen::Matrix<high_prec_float, 4, 4> neut_mass_mat(4, 4);
    neut_mass_mat << high_prec_float(M1_wk), 0.0, (-1.0) * gpr_wk * vd / SQRT_TWO, gpr_wk * vu / SQRT_TWO,
                    0.0, high_prec_float(M2_wk), g2_wk * vd / SQRT_TWO, (-1.0) * g2_wk * vu / SQRT_TWO,
                    (-1.0) * gpr_wk * vd / SQRT_TWO, g2_wk * vd / SQRT_TWO, 0.0, (-1.0) * mu_wk,
                    gpr_wk * vu / SQRT_TWO, (-1.0) * g2_wk * vu / SQRT_TWO, (-1.0) * mu_wk, 0.0;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<high_prec_float, 4, 4>> solver(
        neut_mass_mat, Eigen::EigenvaluesOnly);
    if (solver.info() != Eigen::Success) {
        throw NumericalFailure(
            "radcorr_calc/neutralino eigensolver", {"no convergence"});
    }
    const Eigen::Matrix<high_prec_float, 4, 1> my_neut_mass_eigvals =
        solver.eigenvalues();
    Eigen::Matrix<high_prec_float, 4, 1> mneutrsq = my_neut_mass_eigvals.array().square();

    // Sort eigenvalues using Eigen's built-in functions
    std::sort(mneutrsq.data(), mneutrsq.data() + mneutrsq.size());

    std::vector<high_prec_float> eigval_vector(my_neut_mass_eigvals.data(), my_neut_mass_eigvals.data() + my_neut_mass_eigvals.size());
    // Sort eigenvalues using Eigen's built-in functions
    std::sort(eigval_vector.begin(), eigval_vector.end(), [](const high_prec_float& a, const high_prec_float& b) {
        return abs(a) < abs(b);
    });

    high_prec_float msN1 = eigval_vector[0];
    high_prec_float msN2 = eigval_vector[1];
    high_prec_float msN3 = eigval_vector[2];
    high_prec_float msN4 = eigval_vector[3];
    //cout << "msN1 = " << msN1 << "\nmsN2 = " << msN2 <<  "\nmsN3 = " << msN3 <<  "\nmsN4 = " << msN4 << endl;

    high_prec_float msN1sq = mneutrsq[0];
    high_prec_float msN2sq = mneutrsq[1];
    high_prec_float msN3sq = mneutrsq[2];
    high_prec_float msN4sq = mneutrsq[3];
    
    // Neutral Higgs high_prec_floatt mass eigenstate running squared masses
    high_prec_float mh0sq = (0.5)\
        * ((mA0sq) + (mz_q_sq)
           - sqrt(pow(mA0sq - mz_q_sq, 2.0) + (4.0 * mz_q_sq * mA0sq * pow(sin(2.0 * beta_wk), 2.0))));
    high_prec_float mH0sq = (0.5)\
        * ((mA0sq) + (mz_q_sq)
           + sqrt(pow(mA0sq - mz_q_sq, 2.0) + (4.0 * mz_q_sq * mA0sq * pow(sin(2.0 * beta_wk), 2.0))));

    ////////// Radiative corrections in stop squark sector //////////

    const high_prec_float stop_denom = m_stop_2sq - m_stop_1sq;
    const high_prec_float stopuu_num = (pow(at_wk, 2.0)) - (at_wk * yt_wk * mu_wk / (tan(beta_wk)))\
        - ((1.0 / 24.0) * ((3.0 * g2_sq) - (10.0 * gpr_sq)) * (mQ3_sq_wk - mU3_sq_wk))\
        - ((1.0 / 288.0) * (pow((3.0 * g2_sq) - (10.0 * gpr_sq), 2.0) * v_sq * cos2b));
    const high_prec_float stopdd_num = (yt_wk * mu_wk) * ((yt_wk * mu_wk)
                                                 - at_wk * tan(beta_wk))\
        + ((1.0 / 24.0) * ((3.0 * g2_sq) - (10.0 * gpr_sq)) * (mQ3_sq_wk - mU3_sq_wk))\
        + ((1.0 / 288.0) * (pow((3.0 * g2_sq) - (10.0 * gpr_sq), 2.0) * v_sq * cos2b));
    //std::cout << "stopuu_num = " << stopuu_num << "\t" << "stopdd_num = " << stopdd_num;
    const high_prec_float sigmauu_stop_1 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_stop_1sq, pow(myQ, 2.0)) \
        * (pow(yt_wk, 2.0) - ((g2_sq + (2.0 * gpr_sq)) / 8.0)
           - (stopuu_num / stop_denom));
    const high_prec_float sigmauu_stop_2 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_stop_2sq, pow(myQ, 2.0)) \
        * (pow(yt_wk, 2.0) - ((g2_sq + (2.0 * gpr_sq)) / 8.0)
           + (stopuu_num / stop_denom));
    //std::cout << std::endl << std::endl << "Sigma_u(stop_1): " << sigmauu_stop_1 << "\t" << "Sigma_u(stop_2): " << sigmauu_stop_2;
    const high_prec_float sigmadd_stop_1 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_stop_1sq, pow(myQ, 2.0)) \
        * (((g2_sq + (2.0 * gpr_sq)) / 8.0) - (stopdd_num / stop_denom));
    const high_prec_float sigmadd_stop_2 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_stop_2sq, pow(myQ, 2.0)) \
        * (((g2_sq + (2.0 * gpr_sq)) / 8.0) + (stopdd_num / stop_denom));

    ////////// Radiative corrections in sbottom squark sector //////////

    const high_prec_float sbot_denom = m_sbot_2sq - m_sbot_1sq;
    const high_prec_float sbotuu_num = (yb_wk * mu_wk) * ((yb_wk * mu_wk)
                                                 - ab_wk / tan(beta_wk))\
        + ((1.0 / 24.0) * ((3.0 * g2_sq) - (2.0 * gpr_sq)) * (mQ3_sq_wk - mD3_sq_wk))\
        - ((1.0 / 288.0) * (pow((3.0 * g2_sq) - (2.0 * gpr_sq), 2.0) * v_sq * cos2b));
    const high_prec_float sbotdd_num = (pow(ab_wk, 2.0)) - (ab_wk * yb_wk * mu_wk * (tan(beta_wk)))\
        - ((1.0 / 24.0) * ((3.0 * g2_sq) - (2.0 * gpr_sq)) * (mQ3_sq_wk - mD3_sq_wk))\
        + ((1.0 / 288.0) * (pow((3.0 * g2_sq) - (2.0 * gpr_sq), 2.0) * v_sq * cos2b));
    const high_prec_float sigmauu_sbot_1 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_sbot_1sq, pow(myQ, 2.0)) \
        * (((g2_sq + (2.0 * gpr_sq)) / 8.0) - (sbotuu_num / sbot_denom));
    const high_prec_float sigmauu_sbot_2 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_sbot_2sq, pow(myQ, 2.0)) \
        * (((g2_sq + (2.0 * gpr_sq)) / 8.0) + (sbotuu_num / sbot_denom));
    const high_prec_float sigmadd_sbot_1 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_sbot_1sq, pow(myQ, 2.0)) \
        * ((pow(yb_wk, 2.0)) - ((g2_sq + (2.0 * gpr_sq)) / 8.0)
        - (sbotdd_num / sbot_denom));
    const high_prec_float sigmadd_sbot_2 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_sbot_2sq, pow(myQ, 2.0)) \
        * ((pow(yb_wk, 2.0)) - ((g2_sq + (2.0 * gpr_sq)) / 8.0)
        + (sbotdd_num / sbot_denom));

    ////////// Radiative corrections in stau slepton sector //////////
        
    const high_prec_float stau_denom = m_stau_2sq - m_stau_1sq;
    const high_prec_float stauuu_num = (ytau_wk * mu_wk) * ((ytau_wk * mu_wk)
                                                   - atau_wk / tan(beta_wk))\
        + ((1.0 / 8.0) * ((g2_sq) - (6.0 * gpr_sq)) * (mL3_sq_wk - mE3_sq_wk))\
        - ((1.0 / 32.0) * (pow((g2_sq) - (6.0 * gpr_sq), 2.0) * v_sq * cos2b));
    const high_prec_float staudd_num = pow(atau_wk, 2.0) - (atau_wk * ytau_wk * mu_wk * tan(beta_wk))\
        - ((1.0 / 8.0) * ((g2_sq) - (6.0 * gpr_sq)) * (mL3_sq_wk - mE3_sq_wk))\
        + ((1.0 / 32.0) * (pow((g2_sq) - (6.0 * gpr_sq), 2.0) * v_sq * cos2b));
    const high_prec_float sigmauu_stau_1 = ((1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_stau_1sq, pow(myQ, 2.0)) \
        * (((g2_sq + (2.0 * gpr_sq)) / 8.0) - (stauuu_num / stau_denom));
    const high_prec_float sigmauu_stau_2 = ((1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_stau_2sq, pow(myQ, 2.0)) \
        * (((g2_sq + (2.0 * gpr_sq)) / 8.0) + (stauuu_num / stau_denom));
    const high_prec_float sigmadd_stau_1 = ((1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_stau_1sq, pow(myQ, 2.0)) \
        * (pow(ytau_wk, 2.0) - ((g2_sq + (2.0 * gpr_sq)) / 8.0)
        - (staudd_num / stau_denom));
    const high_prec_float sigmadd_stau_2 = ((1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_stau_2sq, pow(myQ, 2.0)) \
        * (pow(ytau_wk, 2.0) - ((g2_sq + (2.0 * gpr_sq)) / 8.0)
        + (staudd_num / stau_denom));
            
    // Tau sneutrino
        
    const high_prec_float sigmauu_stau_sneut = ((1.0 / (16.0 * (pow(M_PI, 2.0))))/ 8.0) * ((-1.0) * (g2_sq + gpr_sq))\
        * logfunc2(mstauneutsq, pow(myQ, 2.0));
    const high_prec_float sigmadd_stau_sneut = ((1.0 / (16.0 * (pow(M_PI, 2.0))))/ 8.0) * ((g2_sq + gpr_sq))\
        * logfunc2(mstauneutsq, pow(myQ, 2.0));

    ////////// Radiative corrections from 2nd generation sfermions //////////
    // Scharm sector
        
    const high_prec_float schm_denom = m_scharm_2sq - m_scharm_1sq;
    const high_prec_float schmuu_num = (pow(ac_wk, 2.0)) - (ac_wk * yc_wk * mu_wk / (tan(beta_wk)))\
        - ((1.0 / 24.0) * ((3.0 * g2_sq) - (10.0 * gpr_sq)) * (mQ2_sq_wk - mU2_sq_wk))\
        - ((1.0 / 288.0) * (pow((3.0 * g2_sq) - (10.0 * gpr_sq), 2.0) * v_sq * cos2b));
    const high_prec_float schmdd_num = (yc_wk * mu_wk) * ((yc_wk * mu_wk)
                                                 - ac_wk * tan(beta_wk))\
        + ((1.0 / 24.0) * ((3.0 * g2_sq) - (10.0 * gpr_sq)) * (mQ2_sq_wk - mU2_sq_wk))\
        + ((1.0 / 288.0) * (pow((3.0 * g2_sq) - (10.0 * gpr_sq), 2.0) * v_sq * cos2b));
    const high_prec_float sigmauu_scharm_1 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_scharm_1sq, pow(myQ, 2.0)) \
        * (pow(yc_wk, 2.0) - ((g2_sq + (2.0 * gpr_sq)) / 8.0)
           - (schmuu_num / schm_denom));
    const high_prec_float sigmauu_scharm_2 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_scharm_2sq, pow(myQ, 2.0)) \
        * (pow(yc_wk, 2.0) - ((g2_sq + (2.0 * gpr_sq)) / 8.0)
        + (schmuu_num / schm_denom));
    const high_prec_float sigmadd_scharm_1 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_scharm_1sq, pow(myQ, 2.0)) \
        * (((g2_sq + (2.0 * gpr_sq)) / 8.0) - (schmdd_num / schm_denom));
    const high_prec_float sigmadd_scharm_2 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_scharm_2sq, pow(myQ, 2.0)) \
        * (((g2_sq + (2.0 * gpr_sq)) / 8.0) + (schmdd_num / schm_denom));

    // Sstrange sector

    const high_prec_float sstr_denom = m_sstrange_2sq - m_sstrange_1sq;
    const high_prec_float sstruu_num = (ys_wk * mu_wk) * ((ys_wk * mu_wk)
                                                 - as_wk / tan(beta_wk))\
        + ((1.0 / 24.0) * ((3.0 * g2_sq) - (2.0 * gpr_sq)) * (mQ2_sq_wk - mD2_sq_wk))\
        - ((1.0 / 288.0) * (pow((3.0 * g2_sq) - (2.0 * gpr_sq), 2.0) * v_sq * cos2b));
    const high_prec_float sstrdd_num = pow(as_wk, 2.0) - (as_wk * ys_wk * mu_wk * tan(beta_wk))\
        - ((1.0 / 24.0) * ((3.0 * g2_sq) - (2.0 * gpr_sq)) * (mQ2_sq_wk - mD2_sq_wk))\
        + ((1.0 / 288.0) * (pow((3.0 * g2_sq) - (2.0 * gpr_sq), 2.0) * v_sq * cos2b));
    const high_prec_float sigmauu_sstrange_1 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_sstrange_1sq, pow(myQ, 2.0)) \
        * (((g2_sq + (2.0 * gpr_sq)) / 8.0) - (sstruu_num / sstr_denom));
    const high_prec_float sigmauu_sstrange_2 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_sstrange_2sq, pow(myQ, 2.0)) \
        * (((g2_sq + (2.0 * gpr_sq)) / 8.0) + (sstruu_num / sstr_denom));
    const high_prec_float sigmadd_sstrange_1 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_sstrange_1sq, pow(myQ, 2.0)) \
        * ((pow(ys_wk, 2.0)) - ((g2_sq + (2.0 * gpr_sq)) / 8.0)
        - (sstrdd_num / sstr_denom));
    const high_prec_float sigmadd_sstrange_2 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_sstrange_2sq, pow(myQ, 2.0)) \
        * ((pow(ys_wk, 2.0)) - ((g2_sq + (2.0 * gpr_sq)) / 8.0)
        + (sstrdd_num / sstr_denom));

    // Smu/smu sneutrino

    const high_prec_float smu_denom = m_smu_2sq - m_smu_1sq;
    const high_prec_float smuuu_num = (ymu_wk * mu_wk) * ((ymu_wk * mu_wk)
                                                 - amu_wk / tan(beta_wk))\
        + ((1.0 / 8.0) * ((g2_sq) - (6.0 * gpr_sq)) * (mL2_sq_wk - mE2_sq_wk))\
        - ((1.0 / 32.0) * (pow((g2_sq) - (6.0 * gpr_sq), 2.0) * v_sq * cos2b));
    const high_prec_float smudd_num = pow(amu_wk, 2.0) - (amu_wk * ymu_wk * mu_wk * tan(beta_wk))\
        - ((1.0 / 8.0) * ((g2_sq) - (6.0 * gpr_sq)) * (mL2_sq_wk - mE2_sq_wk))\
        + ((1.0 / 32.0) * (pow((g2_sq) - (6.0 * gpr_sq), 2.0) * v_sq * cos2b));
    const high_prec_float sigmauu_smu_1 = ((1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_smu_1sq, pow(myQ, 2.0)) \
        * (((g2_sq + (2.0 * gpr_sq)) / 8.0) - (smuuu_num / smu_denom));
    const high_prec_float sigmauu_smu_2 = ((1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_smu_2sq, pow(myQ, 2.0)) \
        * (((g2_sq + (2.0 * gpr_sq)) / 8.0) + (smuuu_num / smu_denom));
    const high_prec_float sigmadd_smu_1 = ((1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_smu_1sq, pow(myQ, 2.0)) \
        * (pow(ymu_wk, 2.0) - ((g2_sq + (2.0 * gpr_sq)) / 8.0)
        - (smudd_num / smu_denom));
    const high_prec_float sigmadd_smu_2 = ((1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_smu_2sq, pow(myQ, 2.0)) \
        * (pow(ymu_wk, 2.0) - ((g2_sq + (2.0 * gpr_sq)) / 8.0)
        + (smudd_num / smu_denom));

    // Mu sneutrino
    const high_prec_float sigmauu_smu_sneut = ((1.0 / (16.0 * (pow(M_PI, 2.0))))/ 8.0) * ((-1.0) * (g2_sq + gpr_sq))\
        * logfunc2(msmuneutsq, pow(myQ, 2.0));
    const high_prec_float sigmadd_smu_sneut = ((1.0 / (16.0 * (pow(M_PI, 2.0))))/ 8.0) * ((g2_sq + gpr_sq))\
        * logfunc2(msmuneutsq, pow(myQ, 2.0));

    ////////// Radiative corrections from 1st generation sfermions //////////
    // Sup sector

    const high_prec_float sup_denom = m_sup_2sq - m_sup_1sq;
    const high_prec_float supuu_num = (pow(au_wk, 2.0)) - (au_wk * yu_wk * mu_wk / (tan(beta_wk)))\
        - ((1.0 / 24.0) * ((3.0 * g2_sq) - (10.0 * gpr_sq)) * (mQ1_sq_wk - mU1_sq_wk))\
        - ((1.0 / 288.0) * (pow((3.0 * g2_sq) - (10.0 * gpr_sq), 2.0) * v_sq * cos2b));
    const high_prec_float supdd_num = (yu_wk * mu_wk) * ((yu_wk * mu_wk)
                                                - au_wk * tan(beta_wk))\
        + ((1.0 / 24.0) * ((3.0 * g2_sq) - (10.0 * gpr_sq)) * (mQ1_sq_wk - mU1_sq_wk))\
        + ((1.0 / 288.0) * (pow((3.0 * g2_sq) - (10.0 * gpr_sq), 2.0) * v_sq * cos2b));
    const high_prec_float sigmauu_sup_1 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_sup_1sq, pow(myQ, 2.0)) \
        * (pow(yu_wk, 2.0) - ((g2_sq + (2.0 * gpr_sq)) / 8.0)
           - (supuu_num / sup_denom));
    const high_prec_float sigmauu_sup_2 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_sup_2sq, pow(myQ, 2.0)) \
        * (pow(yu_wk, 2.0) - ((g2_sq + (2.0 * gpr_sq)) / 8.0)
           + (supuu_num / sup_denom));
    const high_prec_float sigmadd_sup_1 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_sup_1sq, pow(myQ, 2.0)) \
        * (((g2_sq + (2.0 * gpr_sq)) / 8.0) - (supdd_num / sup_denom));
    const high_prec_float sigmadd_sup_2 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_sup_2sq, pow(myQ, 2.0)) \
        * (((g2_sq + (2.0 * gpr_sq)) / 8.0) + (supdd_num / sup_denom));

    // Sdown sector

    const high_prec_float sdwn_denom = m_sdown_2sq - m_sdown_1sq;
    const high_prec_float sdwnuu_num = (yd_wk * mu_wk) * ((yd_wk * mu_wk)
                                            - ad_wk / tan(beta_wk))\
        + ((1.0 / 24.0) * ((3.0 * g2_sq) - (2.0 * gpr_sq)) * (mQ1_sq_wk - mD1_sq_wk))\
        - ((1.0 / 288.0) * (pow((3.0 * g2_sq) - (2.0 * gpr_sq), 2.0) * v_sq * cos2b));
    const high_prec_float sdwndd_num = pow(ad_wk, 2.0) - (ad_wk * yd_wk * mu_wk * tan(beta_wk))\
        - ((1.0 / 24.0) * ((3.0 * g2_sq) - (2.0 * gpr_sq)) * (mQ1_sq_wk - mD1_sq_wk))\
        + ((1.0 / 288.0) * (pow((3.0 * g2_sq) - (2.0 * gpr_sq), 2.0) * v_sq * cos2b));
    high_prec_float sigmauu_sdown_1 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_sdown_1sq, pow(myQ, 2.0)) \
        * (((g2_sq + (2.0 * gpr_sq)) / 8.0) - (sdwnuu_num / sdwn_denom));
    high_prec_float sigmauu_sdown_2 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_sdown_2sq, pow(myQ, 2.0)) \
        * (((g2_sq + (2.0 * gpr_sq)) / 8.0) + (sdwnuu_num / sdwn_denom));
    high_prec_float sigmadd_sdown_1 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_sdown_1sq, pow(myQ, 2.0)) \
        * ((pow(yd_wk, 2.0)) - ((g2_sq + (2.0 * gpr_sq)) / 8.0)
           - (sdwndd_num / sdwn_denom));
    high_prec_float sigmadd_sdown_2 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_sdown_2sq, pow(myQ, 2.0)) \
        * ((pow(yd_wk, 2.0)) - ((g2_sq + (2.0 * gpr_sq)) / 8.0)
           + (sdwndd_num / sdwn_denom));

    // Selectron/selectron sneutrino

    high_prec_float sel_denom = m_se_2sq - m_se_1sq;
    high_prec_float seluu_num = (ye_wk * mu_wk) * ((ye_wk * mu_wk)
                                          - ae_wk / tan(beta_wk))\
        + ((1.0 / 8.0) * ((g2_sq) - (6.0 * gpr_sq)) * (mL1_sq_wk - mE1_sq_wk))\
        - ((1.0 / 32.0) * (pow((g2_sq) - (6.0 * gpr_sq), 2.0) * v_sq * cos2b));
    high_prec_float seldd_num = pow(ae_wk, 2.0) - (ae_wk * ye_wk * mu_wk * tan(beta_wk))\
        - ((1.0 / 8.0) * ((g2_sq) - (6.0 * gpr_sq)) * (mL1_sq_wk - mE1_sq_wk))\
        + ((1.0 / 32.0) * (pow((g2_sq) - (6.0 * gpr_sq), 2.0) * v_sq * cos2b));
    high_prec_float sigmauu_se_1 = ((1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_se_1sq, pow(myQ, 2.0)) \
        * (((g2_sq + (2.0 * gpr_sq)) / 8.0) - (seluu_num / sel_denom));
    high_prec_float sigmauu_se_2 = ((1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_se_2sq, pow(myQ, 2.0)) \
        * (((g2_sq + (2.0 * gpr_sq)) / 8.0) + (seluu_num / sel_denom));
    high_prec_float sigmadd_se_1 = ((1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_se_1sq, pow(myQ, 2.0)) \
        * (pow(ye_wk, 2.0) - ((g2_sq + (2.0 * gpr_sq)) / 8.0)
        - (seldd_num / sel_denom));
    high_prec_float sigmadd_se_2 = ((1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_se_2sq, pow(myQ, 2.0)) \
        * (pow(ye_wk, 2.0) - ((g2_sq + (2.0 * gpr_sq)) / 8.0)
        + (seldd_num / sel_denom));

    // Electron sneutrino

    high_prec_float sigmauu_selec_sneut = ((1.0 / (16.0 * (pow(M_PI, 2.0))))/ 8.0) * ((-1.0) * (g2_sq + gpr_sq))\
        * logfunc2(mselecneutsq, pow(myQ, 2.0));
    high_prec_float sigmadd_selec_sneut = ((1.0 / (16.0 * (pow(M_PI, 2.0))))/ 8.0) * ((g2_sq + gpr_sq))\
        * logfunc2(mselecneutsq, pow(myQ, 2.0));

    ////////// Radiative corrections from chargino sector //////////
    high_prec_float charginouu_num = ((2.0 * M2_wk * mu_wk / tan(beta_wk))
                             + (pow(M2_wk, 2.0) + mu_wk_sq
                             - (g2_sq * v_sq * cos2b))) * (g2_sq / 2.0);
    high_prec_float charginodd_num = ((2.0 * M2_wk * mu_wk * tan(beta_wk))
                             + (pow(M2_wk, 2.0) + mu_wk_sq
                             + (g2_sq * v_sq * cos2b))) * (g2_sq / 2.0);
    high_prec_float chargino_den = msC2sq - msC1sq;
    high_prec_float sigmauu_chargino1 = ((-1.0) / (8.0 * (pow(M_PI, 2.0))))\
        * ((g2_sq / 2.0) - (charginouu_num / chargino_den)) * logfunc2(msC1sq, pow(myQ, 2.0));
    high_prec_float sigmauu_chargino2 = ((-1.0) / (8.0 * (pow(M_PI, 2.0))))\
        * ((g2_sq / 2.0) + (charginouu_num / chargino_den)) * logfunc2(msC2sq, pow(myQ, 2.0));
    high_prec_float sigmadd_chargino1 = ((-1.0) / (8.0 * (pow(M_PI, 2.0))))\
        * ((g2_sq / 2.0) - (charginodd_num / chargino_den)) * logfunc2(msC1sq, pow(myQ, 2.0));
    high_prec_float sigmadd_chargino2 = ((-1.0) / (8.0 * (pow(M_PI, 2.0))))\
        * ((g2_sq / 2.0) + (charginodd_num / chargino_den)) * logfunc2(msC2sq, pow(myQ, 2.0));

    ////////// Radiative corrections from Higgs bosons sector //////////
    high_prec_float higgsuu_num = (mz_q_sq + (mA0sq * (2.0 + (4.0 * cos2b) + cos(4.0 * beta_wk))))\
        * ((g2_sq + gpr_sq) / 4.0);
    high_prec_float higgsdd_num = (mz_q_sq + (mA0sq * (2.0 - (4.0 * cos2b) + cos(4.0 * beta_wk))))\
        * ((g2_sq + gpr_sq) / 4.0);
    high_prec_float higgs_den = (mH0sq - mh0sq);
    high_prec_float sigmauu_h0 = ((1.0 / (32.0 * (pow(M_PI, 2.0))))) * logfunc2(mh0sq, pow(myQ, 2.0))\
        * (((g2_sq + gpr_sq) / 4.0) - (higgsuu_num / higgs_den));
    high_prec_float sigmauu_heavy_h0 = ((1.0 / (32.0 * (pow(M_PI, 2.0))))) * logfunc2(mH0sq, pow(myQ, 2.0))\
        * (((g2_sq + gpr_sq) / 4.0) + (higgsuu_num / higgs_den));
    high_prec_float sigmadd_h0 = ((1.0 / (32.0 * (pow(M_PI, 2.0))))) * logfunc2(mh0sq, pow(myQ, 2.0))\
        * (((g2_sq + gpr_sq) / 4.0) - (higgsdd_num / higgs_den));
    high_prec_float sigmadd_heavy_h0 = ((1.0 / (32.0 * (pow(M_PI, 2.0))))) * logfunc2(mH0sq, pow(myQ, 2.0))\
        * (((g2_sq + gpr_sq) / 4.0) + (higgsdd_num / higgs_den));
    high_prec_float sigmauu_h_pm  = (g2_sq * (1.0 / (16.0 * (pow(M_PI, 2.0)))) / 2.0) * logfunc2(mH_pmsq, pow(myQ, 2.0));
    high_prec_float sigmadd_h_pm = sigmauu_h_pm;

    ////////// Radiative corrections from weak vector bosons sector //////////
    high_prec_float sigmauu_w_pm = (3.0 * g2_sq * (1.0 / (16.0 * (pow(M_PI, 2.0)))) / 2.0) * logfunc2(m_w_sq, pow(myQ, 2.0));
    high_prec_float sigmadd_w_pm = sigmauu_w_pm;
    high_prec_float sigmauu_z0 = (3.0 / 4.0) * (1.0 / (16.0 * (pow(M_PI, 2.0)))) * (gpr_sq + g2_sq)\
        * checkedSignedMZ2ContinuationLog(mz_q_sq, pow(myQ, 2.0));
    high_prec_float sigmadd_z0 = sigmauu_z0;

    ////////// Radiative corrections from SM fermions sector //////////
    high_prec_float sigmauu_top = (-6.0) * pow(yt_wk, 2.0) * (1.0 / (16.0 * (pow(M_PI, 2.0))))\
        * logfunc2(mymtsq, pow(myQ, 2.0));
    high_prec_float sigmadd_top = 0.0;
    high_prec_float sigmauu_bottom = 0.0;
    high_prec_float sigmadd_bottom = (-6.0) * pow(yb_wk, 2.0) * (1.0 / (16.0 * (pow(M_PI, 2.0))))\
        * logfunc2(mymbsq, pow(myQ, 2.0));
    high_prec_float sigmauu_tau = 0.0;
    high_prec_float sigmadd_tau = (-2.0) * pow(ytau_wk, 2.0) * (1.0 / (16.0 * (pow(M_PI, 2.0))))\
        * logfunc2(mymtausq, pow(myQ, 2.0));
    high_prec_float sigmauu_charm = (-6.0) * pow(yc_wk, 2.0) * (1.0 / (16.0 * (pow(M_PI, 2.0))))\
        * logfunc2(mymcsq, pow(myQ, 2.0));
    high_prec_float sigmadd_charm = 0.0;
    high_prec_float sigmauu_strange = 0.0;
    high_prec_float sigmadd_strange = (-6.0) * pow(ys_wk, 2.0) * (1.0 / (16.0 * (pow(M_PI, 2.0))))\
        * logfunc2(mymssq, pow(myQ, 2.0));
    high_prec_float sigmauu_mu = 0.0;
    high_prec_float sigmadd_mu = (-2.0) * pow(ymu_wk, 2.0) * (1.0 / (16.0 * (pow(M_PI, 2.0))))\
        * logfunc2(mymmusq, pow(myQ, 2.0));
    high_prec_float sigmauu_up = (-6.0) * pow(yu_wk, 2.0) * (1.0 / (16.0 * (pow(M_PI, 2.0))))\
        * logfunc2(mymusq, pow(myQ, 2.0));
    high_prec_float sigmadd_up = 0.0;
    high_prec_float sigmauu_down = 0.0;
    high_prec_float sigmadd_down = (-6.0) * pow(yd_wk, 2.0) * (1.0 / (16.0 * (pow(M_PI, 2.0))))\
        * logfunc2(mymdsq, pow(myQ, 2.0));
    high_prec_float sigmauu_elec = 0.0;
    high_prec_float sigmadd_elec = (-2.0) * pow(ye_wk, 2.0) * (1.0 / (16.0 * (pow(M_PI, 2.0))))\
        * logfunc2(mymesq, pow(myQ, 2.0));

    high_prec_float sigmadd2l = sigmadd_2loop(myQ, mu_wk, beta_wk, yt_wk, yc_wk, yu_wk, yb_wk, ys_wk,
                                        yd_wk, ytau_wk, ymu_wk, ye_wk, g1_wk, g2_wk, g3_wk, mQ3_sq_wk,
                                        mQ2_sq_wk, mQ1_sq_wk, mL3_sq_wk, mL2_sq_wk, mL1_sq_wk,
                                        mU3_sq_wk, mU2_sq_wk, mU1_sq_wk, mD3_sq_wk, mD2_sq_wk, mD1_sq_wk,
                                        mE3_sq_wk, mE2_sq_wk, mE1_sq_wk, M1_wk, M2_wk, M3_wk, mHu_sq_wk,
                                        mHd_sq_wk, at_wk, ac_wk, au_wk, ab_wk, as_wk, ad_wk, atau_wk,
                                        amu_wk, ae_wk, m_stop_1sq, m_stop_2sq, mymt, vHiggs_wk);
    high_prec_float sigmauu2l = sigmauu_2loop(myQ, mu_wk, beta_wk, yt_wk, yc_wk, yu_wk, yb_wk, ys_wk,
                                        yd_wk, ytau_wk, ymu_wk, ye_wk, g1_wk, g2_wk, g3_wk, mQ3_sq_wk,
                                        mQ2_sq_wk, mQ1_sq_wk, mL3_sq_wk, mL2_sq_wk, mL1_sq_wk,
                                        mU3_sq_wk, mU2_sq_wk, mU1_sq_wk, mD3_sq_wk, mD2_sq_wk, mD1_sq_wk,
                                        mE3_sq_wk, mE2_sq_wk, mE1_sq_wk, M1_wk, M2_wk, M3_wk, mHu_sq_wk,
                                        mHd_sq_wk, at_wk, ac_wk, au_wk, ab_wk, as_wk, ad_wk, atau_wk,
                                        amu_wk, ae_wk, m_stop_1sq, m_stop_2sq, mymt, vHiggs_wk);

    high_prec_float sigmauuZ1 = sigmauu_neutralino(msN1, M1_wk, M2_wk, mu_wk, g2_sq, gpr_sq, v_sq, vu, vd, beta_wk, myQ);
    high_prec_float sigmauuZ2 = sigmauu_neutralino(msN2, M1_wk, M2_wk, mu_wk, g2_sq, gpr_sq, v_sq, vu, vd, beta_wk, myQ);
    high_prec_float sigmauuZ3 = sigmauu_neutralino(msN3, M1_wk, M2_wk, mu_wk, g2_sq, gpr_sq, v_sq, vu, vd, beta_wk, myQ);
    high_prec_float sigmauuZ4 = sigmauu_neutralino(msN4, M1_wk, M2_wk, mu_wk, g2_sq, gpr_sq, v_sq, vu, vd, beta_wk, myQ);
    high_prec_float sigmaddZ1 = sigmadd_neutralino(msN1, M1_wk, M2_wk, mu_wk, g2_sq, gpr_sq, v_sq, vu, vd, beta_wk, myQ);
    high_prec_float sigmaddZ2 = sigmadd_neutralino(msN2, M1_wk, M2_wk, mu_wk, g2_sq, gpr_sq, v_sq, vu, vd, beta_wk, myQ);
    high_prec_float sigmaddZ3 = sigmadd_neutralino(msN3, M1_wk, M2_wk, mu_wk, g2_sq, gpr_sq, v_sq, vu, vd, beta_wk, myQ);
    high_prec_float sigmaddZ4 = sigmadd_neutralino(msN4, M1_wk, M2_wk, mu_wk, g2_sq, gpr_sq, v_sq, vu, vd, beta_wk, myQ);                     
    ////////// Total radiative corrections //////////
    std::vector<high_prec_float> list_of_myuus = {sigmauu_stop_1, sigmauu_stop_2, sigmauu_sbot_1,
                                            sigmauu_sbot_2, sigmauu_stau_1, sigmauu_stau_2,
                                            sigmauu_stau_sneut, sigmauu_scharm_1+sigmauu_scharm_2+sigmauu_sstrange_1+sigmauu_sstrange_2,
                                            sigmauu_smu_1, sigmauu_smu_2, sigmauu_smu_sneut, 
                                            sigmauu_sup_1+sigmauu_sup_2+sigmauu_sdown_1+sigmauu_sdown_2, sigmauu_se_1,
                                            sigmauu_se_2, sigmauu_selec_sneut,
                                            sigmauuZ1, sigmauuZ2, sigmauuZ3, sigmauuZ4,
                                            sigmauu_chargino1,
                                            sigmauu_chargino2,
                                            sigmauu_h0, sigmauu_heavy_h0, sigmauu_h_pm, sigmauu_w_pm,
                                            sigmauu_z0, sigmauu_top + sigmauu_bottom + sigmauu_tau\
                                            + sigmauu_charm + sigmauu_strange + sigmauu_mu\
                                            + sigmauu_up + sigmauu_down + sigmauu_elec,
                                            sigmauu2l};
    // cout << "Sigma_u values: " << endl;
    // for (high_prec_float value : list_of_myuus) {
    //     cout << value << endl;
    // }
    std::vector<high_prec_float> list_of_mydds = {sigmadd_stop_1, sigmadd_stop_2, sigmadd_sbot_1,
                                            sigmadd_sbot_2, sigmadd_stau_1, sigmadd_stau_2,
                                            sigmadd_stau_sneut, sigmadd_scharm_1+sigmadd_scharm_2+sigmadd_sstrange_1+sigmadd_sstrange_2,
                                            sigmadd_smu_1, sigmadd_smu_2, sigmadd_smu_sneut, 
                                            sigmadd_sup_1+sigmadd_sup_2+sigmadd_sdown_1+sigmadd_sdown_2, sigmadd_se_1,
                                            sigmadd_se_2, sigmadd_selec_sneut,
                                            sigmaddZ1, sigmaddZ2, sigmaddZ3, sigmaddZ4, sigmadd_chargino1, sigmadd_chargino2,
                                            sigmadd_h0, sigmadd_heavy_h0, sigmadd_h_pm, sigmadd_w_pm,
                                            sigmadd_z0, sigmadd_top + sigmadd_bottom + sigmadd_tau\
                                            + sigmadd_charm + sigmadd_strange + sigmadd_mu\
                                            + sigmadd_up + sigmadd_down + sigmadd_elec,
                                            sigmadd2l};
    // cout << "Sigma_d values: " << endl;
    // for (high_prec_float value : list_of_mydds) {
    //     cout << value << endl;
    // }
    const high_prec_float sigmauu_tot = sumFiniteRadiativeCorrections(
        "radcorr_calc/Sigma_u", indexedTerms("Sigma_u", list_of_myuus));
    const high_prec_float sigmadd_tot = sumFiniteRadiativeCorrections(
        "radcorr_calc/Sigma_d", indexedTerms("Sigma_d", list_of_mydds));
    std::vector<high_prec_float> listofres = {sigmauu_tot, sigmadd_tot};
    return listofres;
    
}
