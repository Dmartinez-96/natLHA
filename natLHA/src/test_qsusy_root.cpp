#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "MSSM_RGE_solver.hpp"
#include "MSSM_RGE_solver_with_stopfinder.hpp"
#include "radcorr_calc.hpp"

namespace {

bool expect(bool condition, const std::string& message) {
    if (condition) return true;
    std::cerr << message << "\n";
    return false;
}

std::vector<double> simpleState() {
    std::vector<double> state(44, 0.0);
    state[1] = 1.0;
    state[29] = 1.0e6;
    state[35] = 4.0e6;
    state[43] = 1.0;
    return state;
}

std::vector<double> integrationState() {
    std::vector<double> state(44, 0.0);
    state[0] = 0.70;
    state[1] = 0.70;
    state[2] = 0.70;
    state[3] = 500.0;
    state[4] = 500.0;
    state[5] = 500.0;
    state[6] = 500.0;
    state[7] = 0.50;
    state[8] = 0.003;
    state[9] = 0.00001;
    state[10] = 0.10;
    state[11] = 0.002;
    state[12] = 0.0001;
    state[13] = 0.10;
    state[14] = 0.006;
    state[15] = 0.00003;
    for (std::size_t i = 25; i <= 41; ++i) state[i] = 4.0e6;
    state[42] = 1.0e6;
    state[43] = 10.0;
    return state;
}

StopScalePoint point(double residual) {
    StopScalePoint result;
    result.physical = true;
    result.stop1Squared = 1.0e6;
    result.stop2Squared = 4.0e6;
    result.logResidual = residual;
    return result;
}

}  // namespace

int main() {
    bool ok = true;

    const ODETolerances& tolerances = odeTolerances();
    ok &= expect(tolerances.absolute > 0.0 && tolerances.relative > 0.0,
                 "shared ODE tolerances were not finite positive values");

    const std::vector<double> exact = simpleState();
    const double exactLogScale = 0.25 * std::log(4.0e12);
    const StopScalePoint exactPoint = evaluateStopScalePoint(exact, exactLogScale);
    ok &= expect(exactPoint.physical,
                 "finite positive diagonal stop matrix was rejected");
    ok &= expect(std::abs(exactPoint.stop1Squared - 1.0e6) < 1.0e-8
                     && std::abs(exactPoint.stop2Squared - 4.0e6) < 1.0e-8,
                 "stop eigenvalues changed the diagonal reference matrix");
    ok &= expect(std::abs(exactPoint.logResidual) < 1.0e-14,
                 "stop residual changed its exact logarithmic reference");

    std::vector<double> tachyon = exact;
    tachyon[29] = -1.0e6;
    const StopScalePoint tachyonPoint = evaluateStopScalePoint(tachyon, exactLogScale);
    ok &= expect(!tachyonPoint.physical && tachyonPoint.stop1Squared < 0.0,
                 "nonpositive running stop mass-square was accepted as physical");

    std::vector<double> nonfinite = exact;
    nonfinite[6] = std::numeric_limits<double>::quiet_NaN();
    try {
        evaluateStopScalePoint(nonfinite, exactLogScale);
        ok &= expect(false, "non-finite stop-matrix state was silently accepted");
    } catch (const NumericalFailure& failure) {
        ok &= expect(failure.stage == "Q_SUSY stop matrix input",
                     "non-finite stop-matrix failure lost its stage");
    }

    std::vector<double> overflowingMatrix = exact;
    overflowingMatrix[29] = std::numeric_limits<double>::max() / 2.0;
    overflowingMatrix[35] = -std::numeric_limits<double>::max() / 2.0;
    const StopScalePoint overflowingPoint =
        evaluateStopScalePoint(overflowingMatrix, exactLogScale);
    ok &= expect(!overflowingPoint.numericallyValid
                     && !overflowingPoint.physical,
                 "non-finite derived stop eigenvalue did not become a numerical boundary");

    try {
        const QSusyResult root = findQSusy(
            integrationState(), std::log(1.0e12), -1.0e-6, 0.05);
        ok &= expect(root.rootsFound == 1 && root.stateAtRoot.size() == 44,
                     "production root search did not return one complete state");
        ok &= expect(root.scale >= 500.0 && root.scale < 1.0e11,
                     "production root escaped the bounded search window");
        ok &= expect(root.stop1Squared > 0.0 && root.stop2Squared > 0.0,
                     "production root retained a nonpositive stop mass-square");
        ok &= expect(std::abs(root.residual)
                         <= std::max(tolerances.absolute, tolerances.relative),
                     "production root failed its logarithmic residual check");
        ok &= expect(root.declaredMaxDeltaLogQ == 0.05
                         && root.scanSegments > root.acceptedSteps
                         && root.maxObservedDeltaLogQ <= 0.05,
                     "production root search did not enforce its declared scan spacing");
    } catch (const std::exception& error) {
        std::cerr << "production root integration failed: " << error.what() << "\n";
        ok = false;
    }

    qsusy_detail::ScanState scanState;
    const StopScalePoint invalidPoint;
    const std::vector<qsusy_detail::ScanEvent> enterInvalid =
        qsusy_detail::classifySegment(
            10.0, point(1.0), 9.0, invalidPoint, scanState);
    const std::vector<qsusy_detail::ScanEvent> leaveInvalid =
        qsusy_detail::classifySegment(
            9.0, invalidPoint, 8.0, point(-1.0), scanState);
    const std::vector<qsusy_detail::ScanEvent> recoveredBracket =
        qsusy_detail::classifySegment(
            8.0, point(-1.0), 7.0, point(1.0), scanState);
    ok &= expect(enterInvalid.empty() && leaveInvalid.empty()
                     && scanState.invalidBoundaries == 1
                     && scanState.nonFiniteBoundaries == 0
                     && recoveredBracket.size() == 1
                     && recoveredBracket[0].kind
                            == qsusy_detail::ScanEventKind::signBracket,
                 "invalid stop domain did not reset continuity and recover a later bracket");

    StopScalePoint numericalBoundary;
    numericalBoundary.numericallyValid = false;
    qsusy_detail::ScanState boundaryState;
    const auto enterNumericalBoundary = qsusy_detail::classifySegment(
        10.0, point(1.0), 9.0, numericalBoundary, boundaryState);
    const auto leaveNumericalBoundary = qsusy_detail::classifySegment(
        9.0, numericalBoundary, 8.0, point(-1.0), boundaryState);
    const auto bracketAfterNumericalBoundary = qsusy_detail::classifySegment(
        8.0, point(-1.0), 7.0, point(1.0), boundaryState);
    ok &= expect(enterNumericalBoundary.empty()
                     && leaveNumericalBoundary.empty()
                     && boundaryState.invalidBoundaries == 1
                     && boundaryState.nonFiniteBoundaries == 1
                     && bracketAfterNumericalBoundary.size() == 1,
                 "numerical boundary stopped the scan or corrupted later root counting");

    StopScalePoint inconsistentBoundary = point(0.0);
    inconsistentBoundary.numericallyValid = false;
    qsusy_detail::ScanState inconsistentState;
    const auto inconsistentEvents = qsusy_detail::classifySegment(
        10.0, inconsistentBoundary, 9.0, point(-1.0), inconsistentState);
    ok &= expect(inconsistentEvents.empty()
                     && inconsistentState.invalidBoundaries == 1
                     && inconsistentState.nonFiniteBoundaries == 1,
                 "contradictory numerical and physical flags produced a root event");
    try {
        qsusy_detail::requireUniqueRootCount(
            1, boundaryState.invalidBoundaries,
            boundaryState.nonFiniteBoundaries);
        ok &= expect(false,
                     "one root plus a non-finite numerical boundary was accepted");
    } catch (const QSusyRootSearchFailure& failure) {
        ok &= expect(failure.stage == "Q_SUSY root search"
                         && failure.rootsFound == 1
                         && failure.invalidBoundaries == 1
                         && failure.nonFiniteBoundaries == 1
                         && failure.invalidTerms.size() == 1
                         && failure.invalidTerms[0]
                                == "roots=1, invalid_boundaries=1, "
                                   "nonfinite_boundaries=1",
                     "non-finite numerical boundary lost its fail-closed diagnostic");
    }

    qsusy_detail::ScanState refinementBoundaryState;
    qsusy_detail::recordIsolatedNumericalBoundary(refinementBoundaryState);
    ok &= expect(refinementBoundaryState.invalidBoundaries == 1
                     && refinementBoundaryState.nonFiniteBoundaries == 1
                     && !refinementBoundaryState.inInvalidDomain
                     && !refinementBoundaryState.inNonFiniteDomain,
                 "isolated refinement boundary was not recorded independently");

    const std::vector<qsusy_detail::ScanEvent> exactLow =
        qsusy_detail::classifySegment(
            7.0, point(1.0), 6.0, point(0.0), scanState);
    const std::vector<qsusy_detail::ScanEvent> exactHigh =
        qsusy_detail::classifySegment(
            6.0, point(0.0), 5.0, point(-1.0), scanState);
    ok &= expect(exactLow.size() == 1 && exactHigh.size() == 1
                     && exactLow[0].kind == qsusy_detail::ScanEventKind::exactLow
                     && exactHigh[0].kind == qsusy_detail::ScanEventKind::exactHigh
                     && qsusy_detail::sameRootScale(
                            exactLow[0].lowLogScale, exactHigh[0].highLogScale),
                 "shared exact-zero endpoint was not available for root deduplication");
    std::vector<qsusy_detail::RootCandidate> sharedEndpointRoots;
    qsusy_detail::addRoot(
        sharedEndpointRoots,
        {exactLow[0].lowLogScale, {}, point(0.0)});
    qsusy_detail::addRoot(
        sharedEndpointRoots,
        {exactHigh[0].highLogScale, {}, point(0.0)});
    ok &= expect(sharedEndpointRoots.size() == 1,
                 "production root accumulator did not deduplicate a shared exact node");
    qsusy_detail::addRoot(
        sharedEndpointRoots,
        {std::nextafter(exactLow[0].lowLogScale,
                        std::numeric_limits<double>::infinity()),
         {}, point(0.0)});
    ok &= expect(sharedEndpointRoots.size() == 2,
                 "production root accumulator merged distinct adjacent scales");

    qsusy_detail::ScanState noRootState;
    const std::vector<qsusy_detail::ScanEvent> noRoot =
        qsusy_detail::classifySegment(
            3.0, point(2.0), 2.0, point(1.0), noRootState);
    ok &= expect(noRoot.empty(), "same-sign physical segment produced a root candidate");

    double current = 27.63;
    const double lower = 27.20;
    std::size_t constructedSegments = 0;
    while (current > lower) {
        const double next = qsusy_detail::nextScanLogScale(current, lower, 0.1);
        ok &= expect(current - next <= 0.1 && next >= lower && next < current,
                     "high-log-Q scan-node construction exceeded its declared spacing");
        current = next;
        ++constructedSegments;
    }
    ok &= expect(current == lower && constructedSegments == 5,
                 "clipped final scan node did not reach the exact lower endpoint");

    try {
        qsusy_detail::nextScanLogScale(27.0, 26.0, 0.0);
        ok &= expect(false, "zero maximum scan spacing was silently accepted");
    } catch (const NumericalFailure& failure) {
        ok &= expect(failure.stage == "Q_SUSY scan spacing",
                     "invalid scan spacing lost its distinct failure stage");
    }
    try {
        qsusy_detail::nextScanLogScale(
            27.0, 26.0, std::numeric_limits<double>::denorm_min());
        ok &= expect(false, "non-advancing maximum scan spacing was silently accepted");
    } catch (const NumericalFailure& failure) {
        ok &= expect(failure.stage == "Q_SUSY scan spacing",
                     "non-advancing scan spacing lost its distinct failure stage");
    }

    qsusy_detail::ScanState twoRootState;
    const auto firstCrossing = qsusy_detail::classifySegment(
        3.0, point(1.0), 2.0, point(-1.0), twoRootState);
    const auto secondCrossing = qsusy_detail::classifySegment(
        2.0, point(-1.0), 1.0, point(1.0), twoRootState);
    std::vector<qsusy_detail::RootCandidate> separatedRoots;
    for (const auto& event : firstCrossing) {
        qsusy_detail::addRoot(
            separatedRoots,
            {event.lowLogScale
                 + 0.5 * (event.highLogScale - event.lowLogScale),
             {}, point(0.0)});
    }
    for (const auto& event : secondCrossing) {
        qsusy_detail::addRoot(
            separatedRoots,
            {event.lowLogScale
                 + 0.5 * (event.highLogScale - event.lowLogScale),
             {}, point(0.0)});
    }
    ok &= expect(separatedRoots.size() == 2,
                 "production root accumulator merged separated sign changes");
    try {
        qsusy_detail::requireUniqueRootCount(
            separatedRoots.size(),
            twoRootState.invalidBoundaries,
            twoRootState.nonFiniteBoundaries);
        ok &= expect(false, "two separated sign changes were accepted as unique");
    } catch (const NumericalFailure& failure) {
        ok &= expect(failure.invalidTerms.size() == 1
                         && failure.invalidTerms[0].find("roots=2") != std::string::npos,
                     "two separated sign changes lost their root-count diagnostic");
    }

    try {
        qsusy_detail::requireUniqueRootCount(2, 1, 0);
        ok &= expect(false, "two Q_SUSY roots were accepted as unique");
    } catch (const QSusyRootSearchFailure& failure) {
        ok &= expect(failure.stage == "Q_SUSY root search"
                         && failure.rootsFound == 2
                         && failure.invalidBoundaries == 1
                         && failure.nonFiniteBoundaries == 0
                         && failure.invalidTerms.size() == 1
                         && failure.invalidTerms[0]
                                == "roots=2, invalid_boundaries=1, "
                                   "nonfinite_boundaries=0",
                     "multiple-root rejection lost its count diagnostic");
    }

    std::vector<double> invalidDomain = integrationState();
    invalidDomain[29] = -4.0e6;
    invalidDomain[35] = -4.0e6;
    try {
        findQSusy(invalidDomain, std::log(1.0e12), -1.0e-6, 0.05);
        ok &= expect(false, "all-invalid stop domain produced a Q_SUSY root");
    } catch (const NumericalFailure& failure) {
        ok &= expect(failure.stage == "Q_SUSY root search",
                     "all-invalid stop-domain failure lost the root-search stage");
    }

    return ok ? 0 : 1;
}
