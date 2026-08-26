#ifndef SC_MSSM_RGE_SOLVER_WITH_STOPFINDER_HPP
#define SC_MSSM_RGE_SOLVER_WITH_STOPFINDER_HPP

#include <cstddef>
#include <string>
#include <vector>

#include "radcorr_calc.hpp"

class QSusyRootSearchFailure : public NumericalFailure {
public:
    QSusyRootSearchFailure(
        std::size_t roots,
        std::size_t invalidBoundaries,
        std::size_t nonFiniteBoundaries);

    const std::size_t rootsFound;
    const std::size_t invalidBoundaries;
    const std::size_t nonFiniteBoundaries;
};

struct StopScalePoint {
    bool numericallyValid = true;
    bool physical = false;
    double stop1Squared = 0.0;
    double stop2Squared = 0.0;
    double logResidual = 0.0;
};

struct QSusyResult {
    std::vector<double> stateAtRoot;
    double logScale = 0.0;
    double scale = 0.0;
    double residual = 0.0;
    double stop1Squared = 0.0;
    double stop2Squared = 0.0;
    std::size_t acceptedSteps = 0;
    double declaredMaxDeltaLogQ = 0.0;
    std::size_t scanSegments = 0;
    double maxObservedDeltaLogQ = 0.0;
    std::size_t rootsFound = 0;
    std::size_t invalidBoundaries = 0;
    std::size_t refinementEvaluations = 0;
    std::string diagnostic;
};

StopScalePoint evaluateStopScalePoint(const std::vector<double>& state, double logScale);

QSusyResult findQSusy(const std::vector<double>& highScaleState,
                      double highLogScale,
                      double timeStep,
                      double maxDeltaLogQ);

namespace qsusy_detail {

enum class ScanEventKind { exactHigh, exactLow, signBracket };

struct ScanEvent {
    ScanEventKind kind;
    double highLogScale;
    double lowLogScale;
};

struct ScanState {
    bool inInvalidDomain = false;
    bool inNonFiniteDomain = false;
    std::size_t invalidBoundaries = 0;
    std::size_t nonFiniteBoundaries = 0;
};

struct RootCandidate {
    double logScale = 0.0;
    std::vector<double> state;
    StopScalePoint point;
};

std::vector<ScanEvent> classifySegment(
    double highLogScale,
    const StopScalePoint& highPoint,
    double lowLogScale,
    const StopScalePoint& lowPoint,
    ScanState& scanState);

bool sameRootScale(double first, double second);

void addRoot(std::vector<RootCandidate>& roots, RootCandidate candidate);

double nextScanLogScale(
    double currentLogScale,
    double lowerLogScale,
    double maxDeltaLogQ);

void recordIsolatedNumericalBoundary(ScanState& scanState);

void requireUniqueRootCount(
    std::size_t roots,
    std::size_t invalidBoundaries,
    std::size_t nonFiniteBoundaries);

}  // namespace qsusy_detail

#endif  // SC_MSSM_RGE_SOLVER_WITH_STOPFINDER_HPP
