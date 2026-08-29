#ifndef NATLHA_MSSM_QSUSY_HELPERS_INL
#define NATLHA_MSSM_QSUSY_HELPERS_INL

#include <cmath>
#include <cstddef>

#ifndef NATLHA_QSUSY_HD
#define NATLHA_QSUSY_HD
#define NATLHA_QSUSY_HD_DEFINED_LOCALLY
#endif

namespace natlha::qsusy_numeric {

// Shared search contract. These are the established host values from
// MSSM_RGE_solver_with_stopfinder.cpp, centralized so CPU and device paths cannot drift.
inline constexpr double kMZ = 91.1876;
inline constexpr double kSearchLowerScale = 500.0;
inline constexpr double kRootUpperScale = 1.0e11;

struct StopPoint {
    bool numericallyValid = true;
    bool physical = false;
    double stop1Squared = 0.0;
    double stop2Squared = 0.0;
    double logResidual = 0.0;
};

struct ScanState {
    bool inInvalidDomain = false;
    bool inNonFiniteDomain = false;
    unsigned long long invalidBoundaries = 0;
    unsigned long long nonFiniteBoundaries = 0;
};

enum ScanEventBits : unsigned {
    NoEvent = 0,
    ExactHigh = 1u << 0,
    ExactLow = 1u << 1,
    SignBracket = 1u << 2
};

enum class ScanStepStatus : int {
    Success = 0,
    InvalidInput = 1,
    NonAdvancing = 2
};

template <typename Real>
NATLHA_QSUSY_HD inline double scalarToDouble(Real value) {
    return static_cast<double>(value);
}

NATLHA_QSUSY_HD inline bool finiteDouble(double value) {
    constexpr double maximumFinite = 1.7976931348623157e308;
    return value == value
        && value <= maximumFinite
        && value >= -maximumFinite;
}

template <typename Real>
NATLHA_QSUSY_HD StopPoint evaluateStopPoint(
        const Real* state,
        std::size_t stateSize,
        double logScale,
        double mZ) {
    StopPoint point;
    if (state == nullptr || stateSize != 44 || !finiteDouble(logScale)) {
        point.numericallyValid = false;
        return point;
    }
    for (std::size_t i = 0; i < stateSize; ++i) {
        if (!finiteDouble(scalarToDouble(state[i]))) {
            point.numericallyValid = false;
            return point;
        }
    }

    const double state0 = scalarToDouble(state[0]);
    const double state1 = scalarToDouble(state[1]);
    const double gPrime = ::sqrt(3.0 / 5.0) * state0;
    const double g2 = state1;
    const double vevDenominator = (3.0 * state0 * state0 / 5.0) + g2 * g2;
    const double higgsVev = ::sqrt(2.0 / vevDenominator) * mZ;
    const double beta = ::atan(scalarToDouble(state[43]));
    const double vu = higgsVev * ::sqrt(::pow(::sin(beta), 2.0));
    const double vd = higgsVev * ::sqrt(::pow(::cos(beta), 2.0));
    const double mt = scalarToDouble(state[7]) * vu;
    const double vevDifference = vu * vu - vd * vd;
    const double deltaL =
        vevDifference * ((gPrime * gPrime / 12.0) - (g2 * g2 / 4.0));
    const double deltaR = -vevDifference * gPrime * gPrime / 3.0;
    const double mixing = scalarToDouble(state[16]) * vu
        - scalarToDouble(state[6]) * scalarToDouble(state[7]) * vd;
    const double mLL = scalarToDouble(state[29]) + mt * mt + deltaL;
    const double mRR = scalarToDouble(state[35]) + mt * mt + deltaR;
    const double discriminant =
        (mLL - mRR) * (mLL - mRR) + 4.0 * mixing * mixing;
    const double splitting = ::sqrt(discriminant);

    point.stop1Squared = 0.5 * (mLL + mRR - splitting);
    point.stop2Squared = 0.5 * (mLL + mRR + splitting);
    if (!finiteDouble(point.stop1Squared) || !finiteDouble(point.stop2Squared)) {
        point.numericallyValid = false;
        return point;
    }
    if (point.stop1Squared <= 0.0 || point.stop2Squared <= 0.0) return point;

    point.physical = true;
    point.logResidual = logScale
        - 0.25 * (::log(point.stop1Squared) + ::log(point.stop2Squared));
    if (!finiteDouble(point.logResidual)) {
        point.numericallyValid = false;
        point.physical = false;
    }
    return point;
}

NATLHA_QSUSY_HD inline bool incrementBoundary(unsigned long long& count) {
    if (count == ~0ULL) return false;
    ++count;
    return true;
}

NATLHA_QSUSY_HD inline bool classifySegment(
        const StopPoint& highPoint,
        const StopPoint& lowPoint,
        ScanState& scanState,
        unsigned& events) {
    events = NoEvent;
    const auto recordDomain = [&](const StopPoint& point) {
        if (!point.numericallyValid) {
            if (!scanState.inNonFiniteDomain
                    && !incrementBoundary(scanState.nonFiniteBoundaries)) {
                return false;
            }
            scanState.inNonFiniteDomain = true;
        } else {
            scanState.inNonFiniteDomain = false;
        }
        if (!point.numericallyValid || !point.physical) {
            if (!scanState.inInvalidDomain
                    && !incrementBoundary(scanState.invalidBoundaries)) {
                return false;
            }
            scanState.inInvalidDomain = true;
        } else {
            scanState.inInvalidDomain = false;
        }
        return true;
    };
    if (!recordDomain(highPoint) || !recordDomain(lowPoint)) return false;

    const bool highValid = highPoint.numericallyValid && highPoint.physical;
    const bool lowValid = lowPoint.numericallyValid && lowPoint.physical;
    if (highValid && highPoint.logResidual == 0.0) events |= ExactHigh;
    if (lowValid && lowPoint.logResidual == 0.0) events |= ExactLow;
    if (highValid && lowValid && highPoint.logResidual != 0.0
            && lowPoint.logResidual != 0.0
            && (highPoint.logResidual < 0.0) != (lowPoint.logResidual < 0.0)) {
        events |= SignBracket;
    }
    return true;
}

NATLHA_QSUSY_HD inline ScanStepStatus nextScanLogScale(
        double currentLogScale,
        double lowerLogScale,
        double maxDeltaLogQ,
        double& nextLogScale) {
    if (!finiteDouble(currentLogScale) || !finiteDouble(lowerLogScale)
            || !finiteDouble(maxDeltaLogQ) || maxDeltaLogQ <= 0.0
            || currentLogScale <= lowerLogScale) {
        return ScanStepStatus::InvalidInput;
    }
    const double remaining = currentLogScale - lowerLogScale;
    nextLogScale = remaining <= maxDeltaLogQ
        ? lowerLogScale
        : ::nextafter(currentLogScale - maxDeltaLogQ, currentLogScale);
    const double observed = currentLogScale - nextLogScale;
    if (!finiteDouble(remaining) || !finiteDouble(nextLogScale)
            || !finiteDouble(observed) || nextLogScale < lowerLogScale
            || nextLogScale >= currentLogScale || observed > maxDeltaLogQ) {
        return ScanStepStatus::NonAdvancing;
    }
    return ScanStepStatus::Success;
}

}  // namespace natlha::qsusy_numeric

#ifdef NATLHA_QSUSY_HD_DEFINED_LOCALLY
#undef NATLHA_QSUSY_HD_DEFINED_LOCALLY
#undef NATLHA_QSUSY_HD
#endif

#endif  // NATLHA_MSSM_QSUSY_HELPERS_INL
