#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <array>
#include <string>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include "DBG_calc.hpp"
#include "MSSM_RGE_solver.hpp"
#include "mZ_numsolver.hpp"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace boost::multiprecision;
typedef number<mpfr_float_backend<50>> high_prec_float;

namespace {

bool finiteBG(const high_prec_float& value) {
    return (boost::math::isfinite)(value);
}

MZ2SolveResult shiftedMZ2Solve(high_prec_float RGE_scale_init_val,
                               high_prec_float RGE_scale_final_val,
                               const vector<high_prec_float>& BCs_to_run,
                               high_prec_float continuationSeed) {
    vector<double> BCs_to_run_dbl;
    BCs_to_run_dbl.reserve(BCs_to_run.size());
    for (const auto& value : BCs_to_run) {
        BCs_to_run_dbl.push_back(double(value));
    }
    double RGE_scale_init_val_dbl = double(RGE_scale_init_val);
    double RGE_scale_final_val_dbl = double(RGE_scale_final_val);
    vector<double> currentweaksol_dbl = solveODEs(BCs_to_run_dbl, RGE_scale_init_val_dbl, RGE_scale_final_val_dbl, copysign(1.0e-6, (RGE_scale_final_val_dbl - RGE_scale_init_val_dbl)));
    vector<high_prec_float> currentweaksol;
    for (const auto& value : currentweaksol_dbl) {
        currentweaksol.push_back(high_prec_float(value));
    }
    high_prec_float QSUSY_for_calc = exp(RGE_scale_final_val);
    return solveMZ2(currentweaksol, QSUSY_for_calc, continuationSeed);
}

bool distinctDoubleStates(const std::vector<high_prec_float>& unshifted,
                          const std::vector<high_prec_float>& negative,
                          const std::vector<high_prec_float>& positive,
                          std::string& failure) {
    if (unshifted.size() != negative.size() || unshifted.size() != positive.size()) {
        failure = "shifted boundary-state size changed";
        return false;
    }
    bool negativeDiffers = false;
    bool positiveDiffers = false;
    bool pairDiffers = false;
    for (std::size_t i = 0; i < unshifted.size(); ++i) {
        const double central = double(unshifted[i]);
        const double minus = double(negative[i]);
        const double plus = double(positive[i]);
        if (!std::isfinite(central) || !std::isfinite(minus) || !std::isfinite(plus)) {
            failure = "a boundary state is non-finite after conversion to double";
            return false;
        }
        negativeDiffers = negativeDiffers || minus != central;
        positiveDiffers = positiveDiffers || plus != central;
        pairDiffers = pairDiffers || minus != plus;
    }
    if (!negativeDiffers || !positiveDiffers || !pairDiffers) {
        failure = "plus, minus, and unshifted boundary states are not distinct in double";
        return false;
    }
    return true;
}

bool validNode(const BGNodeDiagnostic& node) {
    return node.boundaryStatesDistinct && node.failure.empty() && node.root.ok
        && finiteBG(node.root.value) && finiteBG(node.root.lower)
        && finiteBG(node.root.upper) && node.root.lower > 0
        && node.root.upper >= node.root.lower;
}

high_prec_float deriv_num_calc(int precselno, high_prec_float curr_hval,
                               const vector<high_prec_float>& mzsq_values) {
    high_prec_float approxderivval = 0.0;
    if (precselno == 1) {
        // 8-point derivative calculation
        approxderivval = (1.0 / curr_hval) *
            ((mzsq_values[0] / 280.0) - (4.0 / 105.0) * mzsq_values[1] +
            mzsq_values[2] / 5.0 - (4.0 / 5.0) * mzsq_values[3] +
            (4.0 / 5.0) * mzsq_values[4] - mzsq_values[5] / 5.0 +
            (4.0 / 105.0) * mzsq_values[6] - mzsq_values[7] / 280.0);
    } else if (precselno == 2) {
        // 4-point derivative calculation
        approxderivval = (1.0 / curr_hval) *
            ((mzsq_values[0] / 12.0) - (2.0 / 3.0) * mzsq_values[1] +
            (2.0 / 3.0) * mzsq_values[2] - mzsq_values[3] / 12.0);
    } else {
        // 2-point derivative branch
        approxderivval = (1.0 / curr_hval) *
            ((-0.5) * mzsq_values[0] + 0.5 * mzsq_values[1]);
    }

    return approxderivval;
}

high_prec_float fixedStencilUncertainty(
        int precision, const high_prec_float& h, const high_prec_float& prefactor,
        const std::vector<std::pair<BGNodeDiagnostic, BGNodeDiagnostic>>& pairs) {
    const std::array<high_prec_float, 4> weights8 = {
        high_prec_float(4) / 5, high_prec_float(1) / 5,
        high_prec_float(4) / 105, high_prec_float(1) / 280};
    const std::array<high_prec_float, 2> weights4 = {
        high_prec_float(2) / 3, high_prec_float(1) / 12};
    high_prec_float weightedWidth = 0;
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        const high_prec_float weight = precision == 1 ? weights8[i] : weights4[i];
        const high_prec_float minusWidth = pairs[i].first.root.upper - pairs[i].first.root.lower;
        const high_prec_float plusWidth = pairs[i].second.root.upper - pairs[i].second.root.lower;
        weightedWidth += weight * (minusWidth + plusWidth) / 2;
    }
    return abs(prefactor) * weightedWidth / h;
}

}  // namespace

namespace dbg_detail {

high_prec_float doubleDomainStep(const high_prec_float& coordinate) {
    const double coordinateDouble = double(coordinate);
    if (!std::isfinite(coordinateDouble)) {
        throw std::runtime_error("Delta_BG coordinate is not finite in the production double domain");
    }
    const double reference = std::max(std::abs(coordinateDouble), 1.0);
    const double next = std::nextafter(reference, std::numeric_limits<double>::infinity());
    const double ulp = next - reference;
    if (!std::isfinite(ulp) || !(ulp > 0)) {
        throw std::runtime_error("Delta_BG coordinate has no finite positive double ULP");
    }
    return cbrt(high_prec_float(3) * high_prec_float(ulp));
}

bool usesAdaptiveTwoPoint(int precision) {
    return precision == 3;
}

std::vector<LabeledValueBG> orderContributions(
        const std::vector<LabeledValueBG>& contributions) {
    std::vector<LabeledValueBG> ordered = contributions;
    std::sort(ordered.begin(), ordered.end(), [](const LabeledValueBG& left,
                                                  const LabeledValueBG& right) {
        const high_prec_float leftMagnitude = abs(left.value);
        const high_prec_float rightMagnitude = abs(right.value);
        if (leftMagnitude != rightMagnitude) return leftMagnitude > rightMagnitude;
        return left.ordinal < right.ordinal;
    });
    return ordered;
}

BGHeadlineDiagnostic makeHeadlineDiagnostic(
        const std::vector<LabeledValueBG>& orderedContributions) {
    BGHeadlineDiagnostic diagnostic;
    if (orderedContributions.empty()) return diagnostic;
    const LabeledValueBG& top = orderedContributions[0];
    diagnostic.topLabel = top.label;
    diagnostic.topValue = top.value;
    diagnostic.topRootUncertainty = top.rootUncertainty;
    for (const auto& contribution : orderedContributions) {
        if (abs(contribution.value) == abs(top.value)) {
            diagnostic.tiedDirectionOrdinals.push_back(contribution.ordinal);
        }
    }
    if (orderedContributions.size() < 2) return diagnostic;
    const LabeledValueBG& second = orderedContributions[1];
    diagnostic.secondLabel = second.label;
    diagnostic.secondValue = second.value;
    diagnostic.secondRootUncertainty = second.rootUncertainty;
    diagnostic.headlineMagnitudeGap = abs(top.value) - abs(second.value);
    const bool signsDiffer = (top.value < 0) != (second.value < 0);
    diagnostic.headlineSignFragileRootUncertainty = signsDiffer
        && diagnostic.headlineMagnitudeGap
            <= top.rootUncertainty + second.rootUncertainty;
    return diagnostic;
}

// ==========================================================================================
// GENERAL N-DIRECTION MACHINERY
// ==========================================================================================

/// How a direction's shift is applied to the GUT-scale boundary conditions.
///
/// The forms are not interchangeable, and which one a slot takes is set by what that slot
/// stores. Sfermion and Higgs soft slots hold SQUARED masses while the direction is a
/// dimension-1 magnitude, so the shift is applied to the square root and squared back with the
/// original sign restored. Trilinear slots hold the soft term a_ij while the direction is the
/// ratio A = a/y, so the shift is applied to the ratio and multiplied back by the Yukawa, which
/// for slot i sits at i-9. Gaugino masses and mu are already dimension-1 and shift directly.
///
/// The bilinear form exists because slot 42 holds b = B*mu while the direction is B, so the
/// shift applies to b/mu and multiplies back by mu at slot 6. That denominator is NOT i-9,
/// which is why it cannot reuse the trilinear form.
/// Finite-difference step for one direction at one precision setting.
///
/// Precision 1/2 retain their fixed 8-/4-point diagnostic stencils. Precision 3 uses the
/// adaptive two-point start returned by doubleDomainStep instead.
static high_prec_float fixedDiagnosticStep(const high_prec_float& value, int precision) {
    const double coordinate = double(value);
    if (!std::isfinite(coordinate)) {
        throw std::runtime_error("Delta_BG coordinate is not finite in the production double domain");
    }
    const double reference = std::max(std::abs(coordinate), 1.0);
    const double ulp = std::nextafter(reference, std::numeric_limits<double>::infinity())
        - reference;
    if (!std::isfinite(ulp) || !(ulp > 0)) {
        throw std::runtime_error("Delta_BG coordinate has no finite positive double ULP");
    }
    if (precision == 1) {
        return pow((high_prec_float(2625) / 16) * high_prec_float(ulp),
                   high_prec_float(1) / 9);
    }
    return pow((high_prec_float(45) / 4) * high_prec_float(ulp),
               high_prec_float(1) / 5);
}

bool applyShift(const BGDirection& direction, const high_prec_float& shift,
                const std::vector<high_prec_float>& input,
                std::vector<high_prec_float>& shifted, std::string& failure) {
    shifted = input;
    for (int i : direction.shiftIndices) {
        if (i < 0 || static_cast<std::size_t>(i) >= shifted.size()) {
            failure = "direction contains an out-of-range boundary-condition index";
            return false;
        }
        if (direction.kind == BGShiftKind::Plain) {
            shifted[i] += shift;
        } else if (direction.kind == BGShiftKind::Scalar) {
            const high_prec_float magnitude = sqrt(abs(shifted[i]));
            shifted[i] = copysign(pow(magnitude + shift, high_prec_float(2)), shifted[i]);
        } else {
            const int denominatorIndex = direction.kind == BGShiftKind::Bilinear ? 6 : i - 9;
            if (denominatorIndex < 0
                    || static_cast<std::size_t>(denominatorIndex) >= shifted.size()
                    || shifted[denominatorIndex] == 0) {
                failure = "direction has a zero or invalid ratio denominator";
                return false;
            }
            shifted[i] = ((shifted[i] / shifted[denominatorIndex]) + shift)
                * shifted[denominatorIndex];
        }
        if (!finiteBG(shifted[i])) {
            failure = "shift produced a non-finite boundary condition";
            return false;
        }
    }
    return true;
}

/// The directions of one model, in the order they are reported.
///
/// The ordinal recorded here is the deterministic tie-break used when two contributions have
/// exactly equal magnitude.
///
/// Every scalar direction uses the positive magnitude p=sqrt(|m^2|) both as its shift
/// coordinate and as the prefactor. The signed contribution comes from dmZ^2/dp, not from
/// silently orienting p with sign(m^2).
std::vector<BGDirection> buildDirections(
        int modselno, const std::vector<high_prec_float>& G) {
    auto idxRange = [](int lo, int hiExclusive) {
        std::vector<int> v;
        for (int i = lo; i < hiExclusive; ++i) v.push_back(i);
        return v;
    };
    auto maxByAbs = [&G](const std::vector<int>& idx) {
        high_prec_float best = G[idx[0]];
        for (int i : idx) if (abs(G[i]) > abs(best)) best = G[i];
        return best;
    };
    auto plainRoot = [](high_prec_float v) { return sqrt(abs(v)); };

    // Largest gaugino mass by VALUE, not by magnitude: the gaugino candidate comparison is a
    // plain max in every model.
    high_prec_float gauginoVal = G[3];
    for (int i = 3; i < 6; ++i) if (G[i] > gauginoVal) gauginoVal = G[i];

    // Largest trilinear RATIO A = a/y by magnitude, over all nine generations.
    high_prec_float trilinVal = G[16] / G[7];
    for (int i = 16; i < 25; ++i) {
        const high_prec_float r = G[i] / G[i - 9];
        if (abs(r) > abs(trilinVal)) trilinVal = r;
    }

    // All 15 sfermion soft slots, 27 through 41 inclusive. Slot 32 is mL3^2, the
    // third-generation left-handed slepton; in a model with a single universal scalar mass it
    // is unified with the other soft scalars at the GUT scale, so it belongs in the list that
    // picks the universal magnitude. The shift over these directions already covered slot 32
    // via its index range, so only the magnitude had been ignoring it.
    const std::vector<int> universalScalars =
        {27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41};
    const std::vector<int> gen12Scalars = {27, 28, 30, 31, 33, 34, 36, 37, 39, 40};
    const std::vector<int> gen1Scalars  = {27, 30, 33, 36, 39};
    const std::vector<int> gen2Scalars  = {28, 31, 34, 37, 40};
    const std::vector<int> gen3Scalars  = {29, 32, 35, 38, 41};

    const BGDirection gauginoDir = {
        "Delta_BG(m_1/2)", BGShiftKind::Plain, {3, 4, 5}, gauginoVal};
    const BGDirection trilinDir = {
        "Delta_BG(A_0)", BGShiftKind::Trilinear, idxRange(16, 25), trilinVal};
    const BGDirection muDir = {"Delta_BG(mu_0)", BGShiftKind::Plain, {6}, G[6]};
    const BGDirection mHuDir = {
        "Delta_BG(mHu)", BGShiftKind::Scalar, {25}, plainRoot(G[25])};
    const BGDirection mHdDir = {
        "Delta_BG(mHd)", BGShiftKind::Scalar, {26}, plainRoot(G[26])};

    std::vector<BGDirection> dirs;
    if (modselno == 1) {
        std::vector<int> cands = universalScalars;
        cands.push_back(25);
        cands.push_back(26);
        dirs.push_back({"Delta_BG(m_0)", BGShiftKind::Scalar, idxRange(25, 42),
                        plainRoot(maxByAbs(cands))});
        dirs.push_back(gauginoDir);
        dirs.push_back(trilinDir);
        dirs.push_back(muDir);
    } else if (modselno == 2) {
        dirs.push_back({"Delta_BG(mHu,d)", BGShiftKind::Scalar, {25, 26},
                        plainRoot(maxByAbs({25, 26}))});
        dirs.push_back({"Delta_BG(m_0)", BGShiftKind::Scalar, idxRange(27, 42),
                        plainRoot(maxByAbs(universalScalars))});
        dirs.push_back(gauginoDir);
        dirs.push_back(trilinDir);
        dirs.push_back(muDir);
    } else if (modselno == 3) {
        dirs.push_back(mHuDir);
        dirs.push_back(mHdDir);
        dirs.push_back({"Delta_BG(m_0)", BGShiftKind::Scalar, idxRange(27, 42),
                        plainRoot(maxByAbs(universalScalars))});
        dirs.push_back(gauginoDir);
        dirs.push_back(trilinDir);
        dirs.push_back(muDir);
    } else if (modselno == 4) {
        dirs.push_back(mHuDir);
        dirs.push_back(mHdDir);
        dirs.push_back({"Delta_BG(m_0(1,2))", BGShiftKind::Scalar, gen12Scalars,
                        plainRoot(maxByAbs(gen12Scalars))});
        dirs.push_back({"Delta_BG(m_0(3))", BGShiftKind::Scalar, gen3Scalars,
                        plainRoot(maxByAbs(gen3Scalars))});
        dirs.push_back(gauginoDir);
        dirs.push_back(trilinDir);
        dirs.push_back(muDir);
    } else if (modselno == 5) {
        dirs.push_back(mHuDir);
        dirs.push_back(mHdDir);
        dirs.push_back({"Delta_BG(m_0(1))", BGShiftKind::Scalar, gen1Scalars,
                        plainRoot(maxByAbs(gen1Scalars))});
        dirs.push_back({"Delta_BG(m_0(2))", BGShiftKind::Scalar, gen2Scalars,
                        plainRoot(maxByAbs(gen2Scalars))});
        dirs.push_back({"Delta_BG(m_0(3))", BGShiftKind::Scalar, gen3Scalars,
                        plainRoot(maxByAbs(gen3Scalars))});
        dirs.push_back(gauginoDir);
        dirs.push_back(trilinDir);
        dirs.push_back(muDir);
    } else {
        // pMSSM-30 plus mu: 31 independent directions, which is the whole point of this branch.
        // Nothing is collapsed by a max here -- each slot is its own direction, so the 15
        // sfermion soft masses, the 9 trilinears and the 3 gauginos are separate, and the Higgs
        // bilinear b joins as the 31st alongside mu.
        //   3 gaugino + 1 mu + 9 trilinear + 2 Higgs soft + 15 sfermion + 1 bilinear = 31.
        static const char * sfermionNames[15] = {
            "mQ1", "mQ2", "mQ3", "mL1", "mL2", "mL3", "mU1", "mU2", "mU3",
            "mD1", "mD2", "mD3", "mE1", "mE2", "mE3"
        };
        static const char * trilinearNames[9] = {
            "A_t", "A_c", "A_u", "A_b", "A_s", "A_d", "A_tau", "A_mu", "A_e"
        };
        static const char * gauginoNames[3] = {"M_1", "M_2", "M_3"};

        dirs.push_back(mHuDir);
        dirs.push_back(mHdDir);
        for (int k = 0; k < 15; ++k) {
            const int i = 27 + k;
            dirs.push_back({std::string("Delta_BG(") + sfermionNames[k] + ")",
                            BGShiftKind::Scalar, {i}, plainRoot(G[i])});
        }
        for (int k = 0; k < 3; ++k) {
            const int i = 3 + k;
            dirs.push_back({std::string("Delta_BG(") + gauginoNames[k] + ")",
                            BGShiftKind::Plain, {i}, G[i]});
        }
        for (int k = 0; k < 9; ++k) {
            const int i = 16 + k;
            dirs.push_back({std::string("Delta_BG(") + trilinearNames[k] + ")",
                            BGShiftKind::Trilinear, {i}, G[i] / G[i - 9]});
        }
        dirs.push_back(muDir);
        dirs.push_back({"Delta_BG(B)", BGShiftKind::Bilinear, {42}, G[42] / G[6]});
    }
    for (std::size_t i = 0; i < dirs.size(); ++i) dirs[i].ordinal = i;
    return dirs;
}

BGDirectionDiagnostic adaptiveTwoPointDirection(
        const BGDirection& direction, const high_prec_float& mZSquared,
        const BGNodePairEvaluator& evaluatePair) {
    BGDirectionDiagnostic diagnostic;
    diagnostic.ordinal = direction.ordinal;
    diagnostic.label = direction.label;
    if (!finiteBG(direction.value) || !finiteBG(mZSquared) || !(mZSquared > 0)) {
        diagnostic.failure = "the direction coordinate or mZ squared denominator is invalid";
        return diagnostic;
    }

    high_prec_float h0 = 0;
    try {
        h0 = doubleDomainStep(direction.value);
    } catch (const std::exception& error) {
        diagnostic.failure = error.what();
        return diagnostic;
    }

    std::vector<std::pair<BGNodeDiagnostic, BGNodeDiagnostic>> cache;
    auto getPair = [&](unsigned level)
            -> const std::pair<BGNodeDiagnostic, BGNodeDiagnostic>& {
        while (cache.size() <= level) {
            const unsigned nextLevel = static_cast<unsigned>(cache.size());
            const high_prec_float magnitude = ldexp(h0, static_cast<int>(nextLevel));
            try {
                auto pair = evaluatePair(magnitude, nextLevel);
                pair.first.shift = -magnitude;
                pair.first.level = nextLevel;
                pair.second.shift = magnitude;
                pair.second.level = nextLevel;
                cache.push_back(std::move(pair));
            } catch (const std::exception& error) {
                BGNodeDiagnostic negative;
                negative.shift = -magnitude;
                negative.level = nextLevel;
                negative.failure = std::string("node evaluation threw: ") + error.what();
                BGNodeDiagnostic positive = negative;
                positive.shift = magnitude;
                cache.push_back({std::move(negative), std::move(positive)});
            } catch (...) {
                BGNodeDiagnostic negative;
                negative.shift = -magnitude;
                negative.level = nextLevel;
                negative.failure = "node evaluation threw an unknown exception";
                BGNodeDiagnostic positive = negative;
                positive.shift = magnitude;
                cache.push_back({std::move(negative), std::move(positive)});
            }
        }
        return cache[level];
    };

    const high_prec_float prefactor = direction.value / mZSquared;
    for (unsigned firstLevel = 0; ; ++firstLevel) {
        const high_prec_float h = ldexp(h0, static_cast<int>(firstLevel));
        if (!(high_prec_float(4) * h < high_prec_float(1))) break;

        BGWindowDiagnostic window;
        window.firstLevel = firstLevel;
        window.h = h;
        getPair(firstLevel + 2);
        const auto& pairH = cache[firstLevel];
        const auto& pair2H = cache[firstLevel + 1];
        const auto& pair4H = cache[firstLevel + 2];
        const std::array<const std::pair<BGNodeDiagnostic, BGNodeDiagnostic>*, 3> pairs = {
            &pairH, &pair2H, &pair4H};

        bool nodesValid = true;
        for (const auto* pair : pairs) {
            nodesValid = nodesValid && validNode(pair->first) && validNode(pair->second);
        }
        if (!nodesValid) {
            window.failure = "a required plus/minus state or root is invalid";
            diagnostic.windows.push_back(std::move(window));
            continue;
        }

        const std::array<high_prec_float, 3> steps = {
            h, high_prec_float(2) * h, high_prec_float(4) * h};
        bool finiteWindow = true;
        for (std::size_t i = 0; i < pairs.size(); ++i) {
            const auto& pair = *pairs[i];
            const high_prec_float contribution = prefactor
                * (pair.second.root.value - pair.first.root.value)
                / (high_prec_float(2) * steps[i]);
            const high_prec_float uncertainty = abs(prefactor)
                * ((pair.second.root.upper - pair.second.root.lower)
                   + (pair.first.root.upper - pair.first.root.lower))
                / (high_prec_float(4) * steps[i]);
            window.contributions.push_back(contribution);
            window.rootUncertainties.push_back(uncertainty);
            finiteWindow = finiteWindow && finiteBG(contribution) && finiteBG(uncertainty)
                && uncertainty >= 0;
        }
        if (!finiteWindow) {
            window.failure = "a contribution or propagated root uncertainty is non-finite";
            diagnostic.windows.push_back(std::move(window));
            continue;
        }

        high_prec_float minimum = window.contributions[0];
        high_prec_float maximum = window.contributions[0];
        high_prec_float maximumMagnitude = abs(window.contributions[0]);
        for (const auto& contribution : window.contributions) {
            minimum = min(minimum, contribution);
            maximum = max(maximum, contribution);
            maximumMagnitude = max(maximumMagnitude, abs(contribution));
        }
        window.contributionSpan = maximum - minimum;
        window.agreementTolerance = max(high_prec_float(1),
            high_prec_float("0.005") * maximumMagnitude);
        bool uncertaintyAccepted = true;
        for (const auto& uncertainty : window.rootUncertainties) {
            uncertaintyAccepted = uncertaintyAccepted
                && uncertainty <= high_prec_float("0.01") * window.agreementTolerance;
        }
        if (!uncertaintyAccepted) {
            window.failure = "propagated root uncertainty exceeds 1% of the agreement tolerance";
            diagnostic.windows.push_back(std::move(window));
            continue;
        }
        if (window.contributionSpan > window.agreementTolerance) {
            window.failure = "C(h), C(2h), and C(4h) do not meet the agreement tolerance";
            diagnostic.windows.push_back(std::move(window));
            continue;
        }

        window.accepted = true;
        diagnostic.contribution = window.contributions[0];
        diagnostic.rootUncertainty = window.rootUncertainties[0];
        diagnostic.acceptedH = h;
        diagnostic.accepted = true;
        diagnostic.windows.push_back(std::move(window));
        break;
    }

    for (const auto& pair : cache) {
        diagnostic.nodes.push_back(pair.first);
        diagnostic.nodes.push_back(pair.second);
    }
    if (!diagnostic.accepted) {
        if (diagnostic.windows.empty()) {
            diagnostic.failure = "the initial adaptive window already reaches the unit shift bound";
        } else {
            diagnostic.failure = "no outward adaptive two-point window met the root and agreement gates";
            if (!diagnostic.windows.back().failure.empty()) {
                diagnostic.failure += "; last window: " + diagnostic.windows.back().failure;
            }
            std::size_t invalidNodes = 0;
            const BGNodeDiagnostic* firstInvalid = nullptr;
            for (const auto& node : diagnostic.nodes) {
                if (!validNode(node)) {
                    ++invalidNodes;
                    if (firstInvalid == nullptr) firstInvalid = &node;
                }
            }
            if (firstInvalid != nullptr) {
                std::ostringstream details;
                details << "; invalid nodes=" << invalidNodes
                        << "; first invalid shift=" << firstInvalid->shift << ": ";
                if (!firstInvalid->failure.empty()) {
                    details << firstInvalid->failure;
                } else {
                    details << describeMZ2Failure(firstInvalid->root);
                }
                diagnostic.failure += details.str();
            }
        }
    }
    return diagnostic;
}

}  // namespace dbg_detail

namespace {

std::pair<BGNodeDiagnostic, BGNodeDiagnostic> evaluateProductionPair(
        const dbg_detail::BGDirection& direction, const high_prec_float& magnitude,
        unsigned level, const std::vector<high_prec_float>& unshifted,
        const high_prec_float& initialScale, const high_prec_float& finalScale,
        const high_prec_float& continuationSeed) {
    BGNodeDiagnostic negative;
    negative.shift = -magnitude;
    negative.level = level;
    BGNodeDiagnostic positive;
    positive.shift = magnitude;
    positive.level = level;
    std::vector<high_prec_float> negativeState;
    std::vector<high_prec_float> positiveState;
    std::string negativeFailure;
    std::string positiveFailure;
    const bool negativeShifted = dbg_detail::applyShift(
        direction, -magnitude, unshifted, negativeState, negativeFailure);
    const bool positiveShifted = dbg_detail::applyShift(
        direction, magnitude, unshifted, positiveState, positiveFailure);
    if (!negativeShifted || !positiveShifted) {
        negative.failure = negativeShifted ? positiveFailure : negativeFailure;
        positive.failure = positiveShifted ? negativeFailure : positiveFailure;
        return {negative, positive};
    }
    std::string distinctFailure;
    const bool distinct = distinctDoubleStates(
        unshifted, negativeState, positiveState, distinctFailure);
    negative.boundaryStatesDistinct = distinct;
    positive.boundaryStatesDistinct = distinct;
    if (!distinct) {
        negative.failure = distinctFailure;
        positive.failure = distinctFailure;
        return {negative, positive};
    }
    try {
        negative.root = shiftedMZ2Solve(
            initialScale, finalScale, negativeState, continuationSeed);
        if (!negative.root.ok) negative.failure = describeMZ2Failure(negative.root);
    } catch (const std::exception& error) {
        negative.failure = std::string("negative node threw: ") + error.what();
    } catch (...) {
        negative.failure = "negative node threw an unknown exception";
    }
    try {
        positive.root = shiftedMZ2Solve(
            initialScale, finalScale, positiveState, continuationSeed);
        if (!positive.root.ok) positive.failure = describeMZ2Failure(positive.root);
    } catch (const std::exception& error) {
        positive.failure = std::string("positive node threw: ") + error.what();
    } catch (...) {
        positive.failure = "positive node threw an unknown exception";
    }
    return {negative, positive};
}

BGDirectionDiagnostic fixedDiagnosticDirection(
        const dbg_detail::BGDirection& direction, int precision,
        const high_prec_float& mZSquared,
        const dbg_detail::BGNodePairEvaluator& evaluatePair) {
    BGDirectionDiagnostic diagnostic;
    diagnostic.ordinal = direction.ordinal;
    diagnostic.label = direction.label;
    high_prec_float h = 0;
    try {
        h = dbg_detail::fixedDiagnosticStep(direction.value, precision);
    } catch (const std::exception& error) {
        diagnostic.failure = error.what();
        return diagnostic;
    }
    const unsigned pairCount = precision == 1 ? 4 : 2;
    std::vector<std::pair<BGNodeDiagnostic, BGNodeDiagnostic>> pairs;
    pairs.reserve(pairCount);
    for (unsigned i = 0; i < pairCount; ++i) {
        const high_prec_float magnitude = high_prec_float(i + 1) * h;
        auto pair = evaluatePair(magnitude, i);
        pair.first.shift = -magnitude;
        pair.first.level = i;
        pair.second.shift = magnitude;
        pair.second.level = i;
        diagnostic.nodes.push_back(pair.first);
        diagnostic.nodes.push_back(pair.second);
        if (!validNode(pair.first) || !validNode(pair.second)) {
            diagnostic.failure = "a fixed-stencil plus/minus state or root is invalid";
            return diagnostic;
        }
        pairs.push_back(std::move(pair));
    }

    std::vector<high_prec_float> mZSquaredValues;
    mZSquaredValues.reserve(pairCount * 2);
    for (unsigned i = pairCount; i > 0; --i) {
        mZSquaredValues.push_back(pairs[i - 1].first.root.value);
    }
    for (unsigned i = 0; i < pairCount; ++i) {
        mZSquaredValues.push_back(pairs[i].second.root.value);
    }
    const high_prec_float prefactor = direction.value / mZSquared;
    diagnostic.contribution = prefactor * deriv_num_calc(precision, h, mZSquaredValues);
    diagnostic.rootUncertainty = fixedStencilUncertainty(
        precision, h, prefactor, pairs);
    if (!finiteBG(diagnostic.contribution) || !finiteBG(diagnostic.rootUncertainty)) {
        diagnostic.failure = "fixed-stencil contribution or root uncertainty is non-finite";
        return diagnostic;
    }
    diagnostic.accepted = true;
    diagnostic.acceptedH = h;
    BGWindowDiagnostic window;
    window.h = h;
    window.contributions = {diagnostic.contribution};
    window.rootUncertainties = {diagnostic.rootUncertainty};
    window.accepted = true;
    diagnostic.windows.push_back(std::move(window));
    return diagnostic;
}

}  // namespace

BGResult DBG_calc(int& modselno, int& precselno,
                  high_prec_float GUT_SCALE, high_prec_float myweakscale,
                  high_prec_float inptanbval,
                  std::vector<high_prec_float> GUT_boundary_conditions,
                  high_prec_float originalmZ2value) {
    (void)inptanbval;
    BGResult result;
    const high_prec_float physicalMZSquared = high_prec_float("91.1876")
        * high_prec_float("91.1876");
    if (modselno < 1 || modselno > 6) {
        result.failure = "Delta_BG model index must be in [1, 6]";
        return result;
    }
    if (precselno < 1 || precselno > 3) {
        result.failure = "Delta_BG precision must be 1, 2, or 3";
        return result;
    }
    if (GUT_boundary_conditions.size() < 44) {
        result.failure = "Delta_BG requires all 44 GUT-scale boundary conditions";
        return result;
    }
    if (!finiteBG(GUT_SCALE) || !finiteBG(myweakscale)
            || !finiteBG(originalmZ2value) || !(originalmZ2value > 0)) {
        result.failure = "Delta_BG received an invalid scale or continuation seed";
        return result;
    }

    const std::vector<dbg_detail::BGDirection> directions =
        dbg_detail::buildDirections(modselno, GUT_boundary_conditions);
    std::vector<LabeledValueBG> contributions;
    contributions.reserve(directions.size());
    for (const auto& direction : directions) {
        if (!finiteBG(direction.value)) {
            result.failure = "Delta_BG direction has a non-finite coordinate: "
                + direction.label;
            return result;
        }
        const dbg_detail::BGNodePairEvaluator evaluatePair =
            [&](const high_prec_float& magnitude, unsigned level) {
                return evaluateProductionPair(
                    direction, magnitude, level, GUT_boundary_conditions,
                    GUT_SCALE, myweakscale, originalmZ2value);
            };
        BGDirectionDiagnostic diagnostic = dbg_detail::usesAdaptiveTwoPoint(precselno)
            ? dbg_detail::adaptiveTwoPointDirection(
                direction, physicalMZSquared, evaluatePair)
            : fixedDiagnosticDirection(
                direction, precselno, physicalMZSquared, evaluatePair);
        result.directions.push_back(diagnostic);
        if (!diagnostic.accepted) {
            result.failure = "Delta_BG direction failed: " + direction.label
                + ": " + diagnostic.failure;
            return result;
        }
        contributions.push_back({diagnostic.contribution, direction.label,
                                 direction.ordinal, diagnostic.rootUncertainty});
    }
    result.contributions = dbg_detail::orderContributions(contributions);
    if (result.contributions.empty()) {
        result.failure = "Delta_BG produced no contributions";
        return result;
    }
    result.headline = dbg_detail::makeHeadlineDiagnostic(result.contributions);
    result.ok = true;
    return result;
}
