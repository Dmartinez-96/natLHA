#include <algorithm>
#include <array>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

#include "EWSB_loop.hpp"

namespace {

using Terms = std::array<high_prec_float, 3>;

high_prec_float loopLog(const high_prec_float& massSquared,
                        const high_prec_float& scaleSquared) {
    return log(abs(massSquared) / scaleSquared) - 1;
}

Terms add(const Terms& left, const Terms& right) {
    return {left[0] + right[0], left[1] + right[1], left[2] + right[2]};
}

Terms upSquarkReference(const high_prec_float& mQSquared,
                        const high_prec_float& mUSquared,
                        const high_prec_float& yukawa,
                        const high_prec_float& trilinear,
                        const high_prec_float& mu,
                        const high_prec_float& g2Squared,
                        const high_prec_float& gPrimeSquared,
                        const high_prec_float& scaleSquared) {
    const high_prec_float pi = boost::math::constants::pi<high_prec_float>();
    const high_prec_float splitting = abs(mQSquared - mUSquared);
    const high_prec_float light = (mQSquared + mUSquared - splitting) / 2;
    const high_prec_float heavy = (mQSquared + mUSquared + splitting) / 2;
    const high_prec_float prefactor = 1 / (64 * pi * pi * splitting);
    const high_prec_float gaugeDifference =
        (3 * g2Squared - 10 * gPrimeSquared) * (mQSquared - mUSquared);
    const high_prec_float diagonalGauge = g2Squared + 2 * gPrimeSquared;
    const high_prec_float mixedGauge = diagonalGauge - 8 * yukawa * yukawa;

    const high_prec_float uuLight = light * prefactor * loopLog(light, scaleSquared)
        * (gaugeDifference - 24 * trilinear * trilinear
           - 3 * mixedGauge * splitting);
    const high_prec_float uuHeavy = -heavy * prefactor * loopLog(heavy, scaleSquared)
        * (gaugeDifference - 24 * trilinear * trilinear
           + 3 * mixedGauge * splitting);
    const high_prec_float ddLight = -light * prefactor * loopLog(light, scaleSquared)
        * (gaugeDifference + 24 * yukawa * yukawa * mu * mu
           - 3 * diagonalGauge * splitting);
    const high_prec_float ddHeavy = heavy * prefactor * loopLog(heavy, scaleSquared)
        * (gaugeDifference + 24 * yukawa * yukawa * mu * mu
           + 3 * diagonalGauge * splitting);
    const high_prec_float udLight = 3 * trilinear * yukawa * mu * light
        * loopLog(light, scaleSquared) / (8 * pi * pi * splitting);
    const high_prec_float udHeavy = -3 * trilinear * yukawa * mu * heavy
        * loopLog(heavy, scaleSquared) / (8 * pi * pi * splitting);
    return {uuLight + uuHeavy, ddLight + ddHeavy, udLight + udHeavy};
}

Terms downSquarkReference(const high_prec_float& mQSquared,
                          const high_prec_float& mDSquared,
                          const high_prec_float& yukawa,
                          const high_prec_float& trilinear,
                          const high_prec_float& mu,
                          const high_prec_float& g2Squared,
                          const high_prec_float& gPrimeSquared,
                          const high_prec_float& scaleSquared) {
    const high_prec_float pi = boost::math::constants::pi<high_prec_float>();
    const high_prec_float splitting = abs(mQSquared - mDSquared);
    const high_prec_float light = (mQSquared + mDSquared - splitting) / 2;
    const high_prec_float heavy = (mQSquared + mDSquared + splitting) / 2;
    const high_prec_float prefactor = 1 / (64 * pi * pi * splitting);
    const high_prec_float gaugeDifference =
        (3 * g2Squared - 2 * gPrimeSquared) * (mQSquared - mDSquared);
    const high_prec_float diagonalGauge = g2Squared + 2 * gPrimeSquared;
    const high_prec_float mixedGauge = diagonalGauge - 8 * yukawa * yukawa;

    const high_prec_float uuLight = -light * prefactor * loopLog(light, scaleSquared)
        * (gaugeDifference + 24 * yukawa * yukawa * mu * mu
           - 3 * diagonalGauge * splitting);
    const high_prec_float uuHeavy = heavy * prefactor * loopLog(heavy, scaleSquared)
        * (gaugeDifference + 24 * yukawa * yukawa * mu * mu
           + 3 * diagonalGauge * splitting);
    const high_prec_float ddLight = light * prefactor * loopLog(light, scaleSquared)
        * (gaugeDifference - 24 * trilinear * trilinear
           - 3 * mixedGauge * splitting);
    const high_prec_float ddHeavy = -heavy * prefactor * loopLog(heavy, scaleSquared)
        * (gaugeDifference - 24 * trilinear * trilinear
           + 3 * mixedGauge * splitting);
    const high_prec_float udLight = 3 * trilinear * yukawa * mu * light
        * loopLog(light, scaleSquared) / (8 * pi * pi * splitting);
    const high_prec_float udHeavy = -3 * trilinear * yukawa * mu * heavy
        * loopLog(heavy, scaleSquared) / (8 * pi * pi * splitting);
    return {uuLight + uuHeavy, ddLight + ddHeavy, udLight + udHeavy};
}

Terms squarkReference(const std::vector<high_prec_float>& state,
                      const high_prec_float& scale) {
    const high_prec_float g2Squared = state[1] * state[1];
    const high_prec_float gPrimeSquared = state[0] * state[0] * high_prec_float(3) / 5;
    const high_prec_float scaleSquared = scale * scale;
    Terms result = {0, 0, 0};
    const int qIndices[] = {29, 28, 27};
    const int upMassIndices[] = {35, 34, 33};
    const int upYukawaIndices[] = {7, 8, 9};
    const int upTrilinearIndices[] = {16, 17, 18};
    const int downMassIndices[] = {38, 37, 36};
    const int downYukawaIndices[] = {10, 11, 12};
    const int downTrilinearIndices[] = {19, 20, 21};
    for (int generation = 0; generation < 3; ++generation) {
        result = add(result, upSquarkReference(
            state[qIndices[generation]], state[upMassIndices[generation]],
            state[upYukawaIndices[generation]], state[upTrilinearIndices[generation]],
            state[6], g2Squared, gPrimeSquared, scaleSquared));
        result = add(result, downSquarkReference(
            state[qIndices[generation]], state[downMassIndices[generation]],
            state[downYukawaIndices[generation]], state[downTrilinearIndices[generation]],
            state[6], g2Squared, gPrimeSquared, scaleSquared));
    }
    return result;
}

std::vector<high_prec_float> stateA() {
    std::vector<high_prec_float> state(44, 0);
    state[0] = high_prec_float("0.47");
    state[1] = high_prec_float("0.64");
    state[2] = high_prec_float("1.05");
    state[3] = 310;
    state[4] = 620;
    state[5] = 1700;
    state[6] = 470;
    const char* yukawas[] = {"0.82", "0.17", "0.031", "0.41", "0.083",
                              "0.019", "0.12", "0.026", "0.007"};
    const char* trilinears[] = {"1300", "-540", "230", "-960", "410",
                                "-175", "380", "-145", "62"};
    for (int i = 0; i < 9; ++i) {
        state[7 + i] = high_prec_float(yukawas[i]);
        state[16 + i] = high_prec_float(trilinears[i]);
    }
    const char* masses[] = {
        "240000", "390000", "1210000", "840000", "530000", "960000",
        "690000", "1480000", "460000", "1330000", "570000", "1180000",
        "430000", "1090000", "350000", "870000", "290000"};
    for (int i = 0; i < 17; ++i) state[25 + i] = high_prec_float(masses[i]);
    state[42] = 180000;
    state[43] = 11;
    return state;
}

bool closeEnough(const high_prec_float& actual, const high_prec_float& expected) {
    const high_prec_float expectedMagnitude = abs(expected);
    const high_prec_float scale =
        expectedMagnitude > 1 ? expectedMagnitude : high_prec_float(1);
    return abs(actual - expected) <= high_prec_float("1e-30") * scale;
}

template <typename Callable>
bool throwsEWSBNumericalFailure(Callable&& callable) {
    try {
        callable();
    } catch (const EWSBNumericalFailure&) {
        return true;
    }
    return false;
}

}  // namespace

int main() {
    const high_prec_float scale = 2100;
    const std::vector<high_prec_float> first = stateA();
    std::vector<high_prec_float> second = first;
    second[7] = high_prec_float("0.73");
    second[8] = high_prec_float("0.11");
    second[9] = high_prec_float("0.044");
    second[10] = high_prec_float("0.35");
    second[11] = high_prec_float("0.061");
    second[12] = high_prec_float("0.012");
    second[16] = 1180;
    second[17] = -470;
    second[18] = 315;
    second[19] = -820;
    second[20] = 360;
    second[21] = -215;
    second[27] = 760000;
    second[28] = 1020000;
    second[29] = 1290000;
    second[33] = 510000;
    second[34] = 910000;
    second[35] = 1510000;
    second[36] = 390000;
    second[37] = 790000;
    second[38] = 1410000;

    const auto productionFirst = ewsb_detail::loopHessianTerms(first, scale);
    const auto productionSecond = ewsb_detail::loopHessianTerms(second, scale);
    if (productionFirst.size() != 3 || productionSecond.size() != 3) {
        std::cerr << "production Hessian helper returned the wrong number of terms\n";
        return 1;
    }
    const Terms referenceFirst = squarkReference(first, scale);
    const Terms referenceSecond = squarkReference(second, scale);
    const char* names[] = {"Sigma_uu^(2)", "Sigma_dd^(2)", "Sigma_ud^(2)"};
    for (int component = 0; component < 3; ++component) {
        const high_prec_float actual =
            productionFirst[component] - productionSecond[component];
        const high_prec_float expected =
            referenceFirst[component] - referenceSecond[component];
        if (!(boost::math::isfinite)(actual) || !(boost::math::isfinite)(expected)
                || abs(expected) < high_prec_float("1e-20")
                || !closeEnough(actual, expected)) {
            std::cerr << std::setprecision(50)
                      << names[component] << " squark Hessian mismatch\n"
                      << "actual   " << actual << "\n"
                      << "expected " << expected << "\n"
                      << "difference " << actual - expected << "\n";
            return 1;
        }
    }
    std::vector<high_prec_float> degenerate = first;
    degenerate[27] = degenerate[33];
    if (!throwsEWSBNumericalFailure([&] {
            ewsb_detail::loopHessianTerms(degenerate, scale);
        })) {
        std::cerr << "production Hessian helper did not diagnose exact degeneracy\n";
        return 1;
    }
    if (!throwsEWSBNumericalFailure([&] { Hessian_check(degenerate, scale); })) {
        std::cerr << "Hessian_check did not diagnose exact degeneracy\n";
        return 1;
    }
    if (!throwsEWSBNumericalFailure([&] {
            Hessian_check(std::vector<high_prec_float>(43, 0), scale);
        }) || !throwsEWSBNumericalFailure([&] { Hessian_check(first, 0); })) {
        std::cerr << "Hessian_check did not diagnose an invalid input domain\n";
        return 1;
    }
    if (!throwsEWSBNumericalFailure([&] {
            BFB_check(std::vector<high_prec_float>(43, 0));
        })) {
        std::cerr << "BFB_check did not diagnose a short weak-scale state\n";
        return 1;
    }
    return 0;
}
