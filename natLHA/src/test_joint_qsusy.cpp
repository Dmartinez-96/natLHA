#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "MSSM_RGE_solver.hpp"
#include "natlha_api.hpp"
#include "natlha_api_detail.hpp"
#include "radcorr_calc.hpp"

namespace {

bool expect(bool condition, const std::string& message) {
    if (condition) return true;
    std::cerr << message << "\n";
    return false;
}

QSusyResult rootAt(double logScale, double stateMarker) {
    QSusyResult root;
    root.stateAtRoot.assign(44, 0.0);
    root.stateAtRoot[0] = stateMarker;
    root.logScale = logScale;
    root.scale = std::exp(logScale);
    root.residual = 0.0;
    root.stop1Squared = 1.0e6;
    root.stop2Squared = 4.0e6;
    root.acceptedSteps = 10;
    root.rootsFound = 1;
    root.declaredMaxDeltaLogQ = 0.05;
    root.scanSegments = 100;
    root.maxObservedDeltaLogQ = 0.04;
    root.diagnostic = "one positive-stop root";
    return root;
}

struct EvolveCall {
    double from;
    double to;
    double step;
    double marker;
};

natlha::detail::EWSBTuneResult tunedState(double marker, long iterations) {
    natlha::detail::EWSBTuneResult tuned;
    tuned.state.assign(44, high_prec_float(0));
    tuned.doubleState.assign(44, 0.0);
    tuned.state[0] = high_prec_float(marker);
    tuned.doubleState[0] = marker;
    tuned.state[6] = high_prec_float(marker);
    tuned.doubleState[6] = marker;
    tuned.state[42] = high_prec_float(marker + 1000.0);
    tuned.doubleState[42] = marker + 1000.0;
    tuned.radCorrs.assign(2, high_prec_float(0));
    tuned.iterations = iterations;
    return tuned;
}

StopScalePoint physicalPoint(double residual) {
    StopScalePoint point;
    point.physical = true;
    point.stop1Squared = 1.0e6;
    point.stop2Squared = 4.0e6;
    point.logResidual = residual;
    return point;
}

bool contains(const std::string& value, const std::string& fragment) {
    return value.find(fragment) != std::string::npos;
}

}  // namespace

int main(int argc, char** argv) {
    bool ok = true;
    const ODETolerances& odeTolerance = odeTolerances();
    const double tolerance = std::max(
        odeTolerance.absolute, odeTolerance.relative);
    std::vector<double> immutableHighState(44, 0.0);
    for (std::size_t i = 0; i < immutableHighState.size(); ++i) {
        immutableHighState[i] = 7000.0 + static_cast<double>(i);
    }

    std::vector<natlha::QSusySearchDiagnostic> acceptedSearches;
    QSusyResult acceptedRoot;
    acceptedRoot.logScale = 8.25;
    acceptedRoot.rootsFound = 1;
    acceptedRoot.invalidBoundaries = 2;
    const QSusyResult returnedRoot = natlha::detail::runAuditedQSusySearch(
        acceptedSearches, [&] { return acceptedRoot; });
    ok &= expect(returnedRoot.logScale == acceptedRoot.logScale
                     && acceptedSearches.size() == 1
                     && acceptedSearches[0].ordinal == 1
                     && acceptedSearches[0].scanComplete
                     && acceptedSearches[0].accepted
                     && acceptedSearches[0].logScale == acceptedRoot.logScale
                     && acceptedSearches[0].rootsFound == 1
                     && acceptedSearches[0].invalidBoundaries == 2,
                 "successful root-search telemetry did not preserve its result");

    std::vector<natlha::QSusySearchDiagnostic> rejectedSearches;
    try {
        natlha::detail::runAuditedQSusySearch(rejectedSearches, []() -> QSusyResult {
            throw QSusyRootSearchFailure(2, 3, 1);
        });
        ok &= expect(false, "completed root-search rejection was swallowed");
    } catch (const QSusyRootSearchFailure& failure) {
        ok &= expect(failure.rootsFound == 2
                         && failure.invalidBoundaries == 3
                         && failure.nonFiniteBoundaries == 1
                         && rejectedSearches.size() == 1
                         && rejectedSearches[0].ordinal == 1
                         && rejectedSearches[0].scanComplete
                         && !rejectedSearches[0].accepted
                         && rejectedSearches[0].rootsFound == 2
                         && rejectedSearches[0].invalidBoundaries == 3
                         && rejectedSearches[0].nonFiniteBoundaries == 1,
                     "completed rejection lost its typed root-search telemetry");
    }

    std::vector<natlha::QSusySearchDiagnostic> unstructuredFailures;
    try {
        natlha::detail::runAuditedQSusySearch(unstructuredFailures, []() -> QSusyResult {
            throw std::runtime_error("injected unstructured search failure");
        });
        ok &= expect(false, "unstructured root-search failure was swallowed");
    } catch (const std::runtime_error& failure) {
        ok &= expect(std::string(failure.what()) == "injected unstructured search failure"
                         && unstructuredFailures.size() == 1
                         && unstructuredFailures[0].ordinal == 1
                         && !unstructuredFailures[0].scanComplete
                         && !unstructuredFailures[0].accepted
                         && unstructuredFailures[0].rootsFound == 0
                         && unstructuredFailures[0].invalidBoundaries == 0
                         && unstructuredFailures[0].nonFiniteBoundaries == 0,
                     "unstructured failure lost its unknown-progress telemetry or exception");
    }

    std::vector<EvolveCall> evolves;
    int rootCalls = 0;
    int tuneCalls = 0;
    int stopCalls = 0;
    natlha::detail::JointQSusyOperations operations;
    operations.immutableHighState = immutableHighState;
    operations.evolve = [&](const std::vector<double>& state, double from,
                            double to, double step) {
        evolves.push_back({from, to, step, state.at(0)});
        std::vector<double> transported(44, -1.0);
        transported[6] = state.at(6) + 10.0;
        transported[42] = state.at(42) + 20.0;
        return transported;
    };
    operations.findRoot = [&](const std::vector<double>& state, double high,
                              double step, double maxDeltaLogQ) {
        ++rootCalls;
        bool immutableCoordinatesPreserved = true;
        for (std::size_t i = 0; i < state.size(); ++i) {
            if (i != 6 && i != 42 && state[i] != immutableHighState[i]) {
                immutableCoordinatesPreserved = false;
            }
        }
        const double expectedMu = rootCalls == 1 ? 111.0 : 112.0;
        const double expectedB = rootCalls == 1 ? 1121.0 : 1122.0;
        ok &= expect(immutableCoordinatesPreserved
                         && state.at(6) == expectedMu && state.at(42) == expectedB
                         && high == 20.0 && step < 0.0 && maxDeltaLogQ == 0.05,
                     "next root search did not receive immutable 42 plus transported mu and b");
        return rootAt(9.0, 90.0);
    };
    operations.tuneMu = [&](const std::vector<double>& state, const high_prec_float& q) {
        ++tuneCalls;
        const double expectedQ = std::exp(tuneCalls == 1 ? 8.0 : 9.0);
        const double expectedMarker = tuneCalls == 1 ? 80.0 : 90.0;
        ok &= expect(state.at(0) == expectedMarker
                         && std::abs(static_cast<double>(q) - expectedQ) < 1.0e-6,
                     "mu tuning did not receive the current accepted root state and scale");
        return tunedState(tuneCalls == 1 ? 101.0 : 102.0, tuneCalls);
    };
    operations.evaluateStop = [&](const std::vector<double>& state, double logScale) {
        ++stopCalls;
        const double expectedMarker = stopCalls == 1 ? 101.0 : 102.0;
        ok &= expect(state.at(0) == expectedMarker,
                     "stop residual was evaluated before the mu-tuned state arrived");
        ok &= expect(logScale == (stopCalls == 1 ? 8.0 : 9.0),
                     "stop residual was evaluated at the wrong root scale");
        return physicalPoint(stopCalls == 1 ? 100.0 * tolerance : 0.5 * tolerance);
    };

    std::vector<double> initialState(44, 0.0);
    initialState[0] = 7.0;
    try {
        natlha::detail::solveJointQSusyMu(
            rootAt(8.0, 80.0), 20.0, 1.0e-6, 0.05,
            std::numeric_limits<long>::max(), operations);
        ok &= expect(false, "an overflow-capable joint iteration bound was accepted");
    } catch (const NumericalFailure& failure) {
        ok &= expect(failure.stage == "joint Q_SUSY/mu solve input"
                         && contains(failure.what(), "invalid orchestration input")
                         && rootCalls == 0 && tuneCalls == 0 && stopCalls == 0,
                     "overflow-capable joint iteration bound crossed the input boundary");
    }
    natlha::detail::JointQSusyOperations shortImmutableOperations = operations;
    shortImmutableOperations.immutableHighState.pop_back();
    try {
        natlha::detail::solveJointQSusyMu(
            rootAt(8.0, 80.0), 20.0, 1.0e-6, 0.05, 2,
            shortImmutableOperations);
        ok &= expect(false, "a short immutable high-search state was accepted");
    } catch (const NumericalFailure& failure) {
        ok &= expect(contains(failure.what(), "immutable high-search state")
                         && rootCalls == 0 && tuneCalls == 0 && stopCalls == 0,
                     "short immutable high-search state crossed the input boundary");
    }
    natlha::detail::JointQSusyOperations nonfiniteImmutableOperations = operations;
    nonfiniteImmutableOperations.immutableHighState[17] =
        std::numeric_limits<double>::quiet_NaN();
    try {
        natlha::detail::solveJointQSusyMu(
            rootAt(8.0, 80.0), 20.0, 1.0e-6, 0.05, 2,
            nonfiniteImmutableOperations);
        ok &= expect(false, "a non-finite immutable high-search state was accepted");
    } catch (const NumericalFailure& failure) {
        ok &= expect(contains(failure.what(), "immutable high-search state")
                         && rootCalls == 0 && tuneCalls == 0 && stopCalls == 0,
                     "non-finite immutable high-search state crossed the input boundary");
    }
    const natlha::detail::JointQSusySolution synthetic =
        natlha::detail::solveJointQSusyMu(
            rootAt(8.0, 80.0), 20.0, 1.0e-6, 0.05, 10, operations);
    ok &= expect(rootCalls == 2 && tuneCalls == 3 && stopCalls == 3,
                 "joint gates did not require the expected three completed iterations");
    ok &= expect(evolves.size() == 2
                     && evolves[0].from == 8.0 && evolves[0].to == 20.0
                     && evolves[0].marker == 101.0 && evolves[0].step > 0.0
                     && evolves[1].from == 9.0 && evolves[1].to == 20.0
                     && evolves[1].marker == 102.0 && evolves[1].step > 0.0,
                 "joint solve reconstructed a root state or skipped the full search window");
    ok &= expect(synthetic.diagnostics.size() == 3
                     && std::abs(static_cast<double>(synthetic.diagnostics[0].residual))
                            > tolerance
                     && std::abs(static_cast<double>(synthetic.diagnostics[1].residual))
                            <= tolerance
                     && std::abs(static_cast<double>(synthetic.diagnostics[2].residual))
                            <= tolerance
                     && synthetic.diagnostics[0].mu == high_prec_float(101)
                     && synthetic.diagnostics[1].mu == high_prec_float(102)
                     && synthetic.diagnostics[2].mu == high_prec_float(102)
                     && synthetic.ewsbIterations == 6,
                 "joint diagnostics did not preserve the post-retune convergence history");

    int firstResidualEvolves = 0;
    int firstResidualRoots = 0;
    int firstResidualTunes = 0;
    natlha::detail::JointQSusyOperations firstResidualOperations;
    firstResidualOperations.immutableHighState = immutableHighState;
    firstResidualOperations.evolve = [&] (
        const std::vector<double>& state, double, double, double) {
        ++firstResidualEvolves;
        return state;
    };
    firstResidualOperations.findRoot = [&] (
        const std::vector<double>&, double, double, double) {
        ++firstResidualRoots;
        return rootAt(8.0, 80.0);
    };
    firstResidualOperations.tuneMu = [&] (
        const std::vector<double>&, const high_prec_float&) {
        ++firstResidualTunes;
        return tunedState(1.0, 1);
    };
    firstResidualOperations.evaluateStop = [&] (
        const std::vector<double>&, double) {
        return physicalPoint(0.5 * tolerance);
    };
    const natlha::detail::JointQSusySolution firstResidual =
        natlha::detail::solveJointQSusyMu(
            rootAt(8.0, 80.0), 20.0, 1.0e-6, 0.05, 3,
            firstResidualOperations);
    ok &= expect(firstResidual.diagnostics.size() == 2
                     && firstResidualEvolves == 1
                     && firstResidualRoots == 1
                     && firstResidualTunes == 2,
                 "a small first residual bypassed consecutive-state confirmation");

    int movingQRoots = 0;
    natlha::detail::JointQSusyOperations movingQOperations = firstResidualOperations;
    movingQOperations.findRoot = [&] (
        const std::vector<double>&, double, double, double) {
        ++movingQRoots;
        return rootAt(9.0, 90.0);
    };
    const natlha::detail::JointQSusySolution movingQ =
        natlha::detail::solveJointQSusyMu(
            rootAt(8.0, 80.0), 20.0, 1.0e-6, 0.05, 3,
            movingQOperations);
    ok &= expect(movingQ.diagnostics.size() == 3 && movingQRoots == 2,
                 "stable mu and residual silently bypassed the consecutive log-Q gate");

    int movingMuTunes = 0;
    natlha::detail::JointQSusyOperations movingMuOperations = firstResidualOperations;
    movingMuOperations.tuneMu = [&] (
        const std::vector<double>&, const high_prec_float&) {
        ++movingMuTunes;
        return tunedState(movingMuTunes == 1 ? 1.0 : 2.0, 1);
    };
    const natlha::detail::JointQSusySolution movingMu =
        natlha::detail::solveJointQSusyMu(
            rootAt(8.0, 80.0), 20.0, 1.0e-6, 0.05, 3,
            movingMuOperations);
    ok &= expect(movingMu.diagnostics.size() == 3 && movingMuTunes == 3,
                 "stable log-Q and residual silently bypassed the consecutive mu gate");

    int slowMuTunes = 0;
    natlha::detail::JointQSusyOperations slowMuOperations = firstResidualOperations;
    slowMuOperations.tuneMu = [&] (
        const std::vector<double>&, const high_prec_float&) {
        ++slowMuTunes;
        return tunedState(slowMuTunes < 4 ? slowMuTunes : 3.0, 1);
    };
    const natlha::detail::JointQSusySolution slowMu =
        natlha::detail::solveJointQSusyMu(
            rootAt(8.0, 80.0), 20.0, 1.0e-6, 0.05, 4,
            slowMuOperations);
    ok &= expect(slowMu.diagnostics.size() == 4 && slowMuTunes == 4,
                 "stable log-Q was misclassified as a two-cycle before mu convergence");

    int relativeMuTunes = 0;
    natlha::detail::JointQSusyOperations relativeMuOperations = firstResidualOperations;
    relativeMuOperations.tuneMu = [&] (
        const std::vector<double>&, const high_prec_float&) {
        ++relativeMuTunes;
        return tunedState(
            relativeMuTunes == 1 ? 1.0e12 : 1.0e12 + 0.5, 1);
    };
    const natlha::detail::JointQSusySolution relativeMu =
        natlha::detail::solveJointQSusyMu(
            rootAt(8.0, 80.0), 20.0, 1.0e-6, 0.05, 2,
            relativeMuOperations);
    ok &= expect(relativeMu.diagnostics.size() == 2 && relativeMuTunes == 2,
                 "the relative mu tolerance was not scaled by consecutive magnitudes");

    int absoluteMuTunes = 0;
    natlha::detail::JointQSusyOperations absoluteMuOperations = firstResidualOperations;
    absoluteMuOperations.tuneMu = [&] (
        const std::vector<double>&, const high_prec_float&) {
        ++absoluteMuTunes;
        return tunedState(
            absoluteMuTunes == 1 ? 0.0 : 0.5 * odeTolerance.absolute, 1);
    };
    const natlha::detail::JointQSusySolution absoluteMu =
        natlha::detail::solveJointQSusyMu(
            rootAt(8.0, 80.0), 20.0, 1.0e-6, 0.05, 2,
            absoluteMuOperations);
    ok &= expect(absoluteMu.diagnostics.size() == 2 && absoluteMuTunes == 2,
                 "the absolute mu-tolerance floor was not applied near zero");

    natlha::detail::JointQSusyOperations residualBoundaryOperations =
        firstResidualOperations;
    residualBoundaryOperations.evaluateStop = [tolerance] (
        const std::vector<double>&, double) {
        return physicalPoint(tolerance);
    };
    const natlha::detail::JointQSusySolution residualAtBoundary =
        natlha::detail::solveJointQSusyMu(
            rootAt(8.0, 80.0), 20.0, 1.0e-6, 0.05, 2,
            residualBoundaryOperations);
    ok &= expect(residualAtBoundary.diagnostics.size() == 2,
                 "a stop residual equal to its tolerance was rejected");

    natlha::detail::JointQSusyOperations residualAboveBoundaryOperations =
        residualBoundaryOperations;
    residualAboveBoundaryOperations.evaluateStop = [tolerance] (
        const std::vector<double>&, double) {
        return physicalPoint(std::nextafter(
            tolerance, std::numeric_limits<double>::infinity()));
    };
    try {
        natlha::detail::solveJointQSusyMu(
            rootAt(8.0, 80.0), 20.0, 1.0e-6, 0.05, 2,
            residualAboveBoundaryOperations);
        ok &= expect(false, "a stop residual just above its tolerance was accepted");
    } catch (const natlha::detail::JointQSusyConvergenceFailure& failure) {
        ok &= expect(contains(failure.what(), "2-iteration limit exhausted"),
                     "the just-over stop residual lost its bounded failure");
    }

    natlha::detail::JointQSusyOperations qBoundaryOperations =
        firstResidualOperations;
    qBoundaryOperations.findRoot = [tolerance] (
        const std::vector<double>&, double, double, double) {
        return rootAt(tolerance, 80.0);
    };
    const natlha::detail::JointQSusySolution qAtBoundary =
        natlha::detail::solveJointQSusyMu(
            rootAt(0.0, 80.0), 20.0, 1.0e-6, 0.05, 2,
            qBoundaryOperations);
    ok &= expect(qAtBoundary.diagnostics.size() == 2,
                 "a consecutive log-Q difference equal to its tolerance was rejected");

    const double qJustAboveTolerance = std::nextafter(
        tolerance, std::numeric_limits<double>::infinity());
    natlha::detail::JointQSusyOperations qAboveBoundaryOperations =
        firstResidualOperations;
    qAboveBoundaryOperations.findRoot = [qJustAboveTolerance] (
        const std::vector<double>&, double, double, double) {
        return rootAt(qJustAboveTolerance, 80.0);
    };
    try {
        natlha::detail::solveJointQSusyMu(
            rootAt(0.0, 80.0), 20.0, 1.0e-6, 0.05, 2,
            qAboveBoundaryOperations);
        ok &= expect(false, "a consecutive log-Q difference just above tolerance was accepted");
    } catch (const natlha::detail::JointQSusyConvergenceFailure& failure) {
        ok &= expect(contains(failure.what(), "2-iteration limit exhausted"),
                     "the just-over consecutive log-Q difference lost its bounded failure");
    }

    int muBoundaryTunes = 0;
    natlha::detail::JointQSusyOperations muBoundaryOperations =
        firstResidualOperations;
    muBoundaryOperations.tuneMu = [&] (
        const std::vector<double>&, const high_prec_float&) {
        ++muBoundaryTunes;
        return tunedState(muBoundaryTunes == 1 ? 0.0 : tolerance, 1);
    };
    const natlha::detail::JointQSusySolution muAtBoundary =
        natlha::detail::solveJointQSusyMu(
            rootAt(8.0, 80.0), 20.0, 1.0e-6, 0.05, 2,
            muBoundaryOperations);
    ok &= expect(muAtBoundary.diagnostics.size() == 2 && muBoundaryTunes == 2,
                 "a consecutive mu difference equal to its tolerance was rejected");

    int muAboveBoundaryTunes = 0;
    const double muJustAboveTolerance = std::nextafter(
        tolerance, std::numeric_limits<double>::infinity());
    natlha::detail::JointQSusyOperations muAboveBoundaryOperations =
        firstResidualOperations;
    muAboveBoundaryOperations.tuneMu = [&] (
        const std::vector<double>&, const high_prec_float&) {
        ++muAboveBoundaryTunes;
        return tunedState(
            muAboveBoundaryTunes == 1 ? 0.0 : muJustAboveTolerance, 1);
    };
    try {
        natlha::detail::solveJointQSusyMu(
            rootAt(8.0, 80.0), 20.0, 1.0e-6, 0.05, 2,
            muAboveBoundaryOperations);
        ok &= expect(false, "a consecutive mu difference just above tolerance was accepted");
    } catch (const natlha::detail::JointQSusyConvergenceFailure& failure) {
        ok &= expect(contains(failure.what(), "2-iteration limit exhausted")
                         && muAboveBoundaryTunes == 2,
                     "the just-over consecutive mu difference lost its bounded failure");
    }

    int cycleRootCalls = 0;
    natlha::detail::JointQSusyOperations cycleOperations = operations;
    cycleOperations.findRoot = [&](const std::vector<double>&, double, double, double) {
        ++cycleRootCalls;
        const double logScale = cycleRootCalls == 1 ? 9.0 : 8.0;
        return rootAt(logScale, 10.0 * logScale);
    };
    cycleOperations.tuneMu = [&](const std::vector<double>&, const high_prec_float&) {
        return tunedState(1.0, 1);
    };
    cycleOperations.evaluateStop = [&](const std::vector<double>&, double) {
        return physicalPoint(0.5 * tolerance);
    };
    try {
        natlha::detail::solveJointQSusyMu(
            rootAt(8.0, 80.0), 20.0, 1.0e-6, 0.05, 10,
            cycleOperations);
        ok &= expect(false, "Q_SUSY two-cycle was silently accepted");
    } catch (const natlha::detail::JointQSusyConvergenceFailure& failure) {
        ok &= expect(contains(failure.what(), "repeated at lag two"),
                     "Q_SUSY two-cycle lost its distinct failure message");
        ok &= expect(failure.diagnostics.size() == 3
                         && failure.ewsbIterations == 3,
                     "Q_SUSY two-cycle lost its completed diagnostic prefix");
        if (failure.diagnostics.size() == 3) {
            for (std::size_t i = 0; i < failure.diagnostics.size(); ++i) {
                const auto& diagnostic = failure.diagnostics[i];
                const double expectedLogQ = i == 1 ? 9.0 : 8.0;
                ok &= expect(diagnostic.iteration == static_cast<long>(i + 1)
                                 && std::abs(std::log(static_cast<double>(
                                        diagnostic.qSusy)) - expectedLogQ) < 1.0e-12
                                 && std::abs(static_cast<double>(diagnostic.residual)
                                             - 0.5 * tolerance) < tolerance
                                 && diagnostic.mu == high_prec_float(1)
                                 && diagnostic.stop1Squared == 1.0e6
                                 && diagnostic.stop2Squared == 4.0e6,
                             "Q_SUSY two-cycle carried an incomplete iteration");
            }
        }
    }

    natlha::detail::JointQSusyOperations capOperations = cycleOperations;
    capOperations.findRoot = [&](const std::vector<double>&, double, double, double) {
        return rootAt(9.0, 90.0);
    };
    try {
        natlha::detail::solveJointQSusyMu(
            rootAt(8.0, 80.0), 20.0, 1.0e-6, 0.05, 1,
            capOperations);
        ok &= expect(false, "one-iteration Q_SUSY cap was silently accepted");
    } catch (const natlha::detail::JointQSusyConvergenceFailure& failure) {
        ok &= expect(contains(failure.what(), "1-iteration limit exhausted")
                         && failure.diagnostics.size() == 1
                         && failure.ewsbIterations == 1,
                     "one-iteration cap lost its distinct completed diagnostic");
    }
    try {
        natlha::detail::solveJointQSusyMu(
            rootAt(8.0, 80.0), 20.0, 1.0e-6, 0.05, 2,
            capOperations);
        ok &= expect(false, "Q_SUSY outer iteration cap was silently exceeded");
    } catch (const natlha::detail::JointQSusyConvergenceFailure& failure) {
        ok &= expect(contains(failure.what(), "2-iteration limit exhausted"),
                     "Q_SUSY outer cap lost its distinct failure message");
        ok &= expect(failure.diagnostics.size() == 2
                         && failure.ewsbIterations == 2,
                     "Q_SUSY outer cap lost its completed diagnostic prefix");
        if (failure.diagnostics.size() == 2) {
            ok &= expect(failure.diagnostics[0].iteration == 1
                             && failure.diagnostics[1].iteration == 2
                             && std::abs(std::log(static_cast<double>(
                                    failure.diagnostics[0].qSusy)) - 8.0) < 1.0e-12
                             && std::abs(std::log(static_cast<double>(
                                    failure.diagnostics[1].qSusy)) - 9.0) < 1.0e-12,
                         "Q_SUSY outer cap carried the wrong iteration prefix");
        }
    }

    int invalidRootEvolves = 0;
    int invalidRootTunes = 0;
    natlha::detail::JointQSusyOperations invalidRootOperations = operations;
    invalidRootOperations.evolve = [&](
        const std::vector<double>& state, double, double, double) {
        ++invalidRootEvolves;
        return state;
    };
    invalidRootOperations.tuneMu = [&](
        const std::vector<double>&, const high_prec_float&) {
        ++invalidRootTunes;
        return tunedState(1.0, 1);
    };

    const auto rejectsInvalidRootMetadata = [&] (
        QSusyResult invalidRoot, const std::string& description) {
        try {
            natlha::detail::solveJointQSusyMu(
                std::move(invalidRoot), 20.0, 1.0e-6, 0.05, 2,
                invalidRootOperations);
            ok &= expect(false, description + " reached mu tuning");
        } catch (const NumericalFailure& failure) {
            ok &= expect(failure.stage == "joint Q_SUSY/mu solve"
                             && contains(failure.what(), "invalid root returned"),
                         description + " lost its fail-closed diagnostic");
        }
    };

    QSusyResult nonuniqueRoot = rootAt(8.0, 80.0);
    nonuniqueRoot.rootsFound = 2;
    rejectsInvalidRootMetadata(nonuniqueRoot, "non-unique accepted root metadata");

    QSusyResult mismatchedSpacingRoot = rootAt(8.0, 80.0);
    mismatchedSpacingRoot.declaredMaxDeltaLogQ = 0.1;
    rejectsInvalidRootMetadata(mismatchedSpacingRoot, "mismatched root scan spacing");

    QSusyResult exceededSpacingRoot = rootAt(8.0, 80.0);
    exceededSpacingRoot.maxObservedDeltaLogQ = 0.051;
    rejectsInvalidRootMetadata(exceededSpacingRoot, "exceeded observed root scan spacing");

    QSusyResult nonconvergedRoot = rootAt(8.0, 80.0);
    nonconvergedRoot.residual = std::nextafter(
        tolerance, std::numeric_limits<double>::infinity());
    rejectsInvalidRootMetadata(nonconvergedRoot, "unconverged accepted root residual");

    QSusyResult nonfiniteStopRoot = rootAt(8.0, 80.0);
    nonfiniteStopRoot.stop1Squared = std::numeric_limits<double>::quiet_NaN();
    rejectsInvalidRootMetadata(nonfiniteStopRoot, "non-finite accepted root stop mass-square");

    QSusyResult emptyScanRoot = rootAt(8.0, 80.0);
    emptyScanRoot.acceptedSteps = 0;
    rejectsInvalidRootMetadata(emptyScanRoot, "zero-step accepted root metadata");

    QSusyResult emptyDiagnosticRoot = rootAt(8.0, 80.0);
    emptyDiagnosticRoot.diagnostic.clear();
    rejectsInvalidRootMetadata(emptyDiagnosticRoot, "empty accepted root diagnostic");

    QSusyResult excessiveRefinementRoot = rootAt(8.0, 80.0);
    excessiveRefinementRoot.refinementEvaluations =
        excessiveRefinementRoot.scanSegments
            * 2 * static_cast<std::size_t>(std::numeric_limits<double>::digits) + 1;
    rejectsInvalidRootMetadata(
        excessiveRefinementRoot, "impossible root refinement-evaluation count");

    QSusyResult overflowingRefinementBoundRoot = rootAt(8.0, 80.0);
    overflowingRefinementBoundRoot.scanSegments =
        std::numeric_limits<std::size_t>::max()
            / (2 * static_cast<std::size_t>(std::numeric_limits<double>::digits)) + 1;
    rejectsInvalidRootMetadata(
        overflowingRefinementBoundRoot, "overflowing root refinement-count bound");

    QSusyResult shortInitialRoot = rootAt(8.0, 80.0);
    shortInitialRoot.stateAtRoot.pop_back();
    try {
        natlha::detail::solveJointQSusyMu(
            std::move(shortInitialRoot), 20.0, 1.0e-6, 0.05, 2,
            invalidRootOperations);
        ok &= expect(false, "short initial root state reached mu tuning");
    } catch (const NumericalFailure& failure) {
        ok &= expect(failure.stage == "joint Q_SUSY/mu solve"
                         && contains(failure.what(), "invalid root state"),
                     "short initial root state lost its fail-closed diagnostic");
    }
    QSusyResult nonfiniteInitialRoot = rootAt(8.0, 80.0);
    nonfiniteInitialRoot.stateAtRoot[17] =
        std::numeric_limits<double>::quiet_NaN();
    try {
        natlha::detail::solveJointQSusyMu(
            std::move(nonfiniteInitialRoot), 20.0, 1.0e-6, 0.05, 2,
            invalidRootOperations);
        ok &= expect(false, "non-finite initial root state reached mu tuning");
    } catch (const NumericalFailure& failure) {
        ok &= expect(failure.stage == "joint Q_SUSY/mu solve"
                         && contains(failure.what(), "invalid root state"),
                     "non-finite initial root state lost its fail-closed diagnostic");
    }
    ok &= expect(invalidRootEvolves == 0 && invalidRootTunes == 0,
                 "invalid initial root state reached an orchestration operation");

    int invalidTunedStopCalls = 0;
    const auto rejectsInvalidTunedState = [&] (
        natlha::detail::EWSBTuneResult injected, const std::string& description) {
        natlha::detail::JointQSusyOperations invalidTunedOperations =
            firstResidualOperations;
        invalidTunedOperations.tuneMu = [injected] (
            const std::vector<double>&, const high_prec_float&) {
            return injected;
        };
        invalidTunedOperations.evaluateStop = [&] (
            const std::vector<double>&, double) {
            ++invalidTunedStopCalls;
            return physicalPoint(0.5 * tolerance);
        };
        try {
            natlha::detail::solveJointQSusyMu(
                rootAt(8.0, 80.0), 20.0, 1.0e-6, 0.05, 2,
                invalidTunedOperations);
            ok &= expect(false, description + " reached stop-scale evaluation");
        } catch (const NumericalFailure& failure) {
            ok &= expect(failure.stage == "joint Q_SUSY/mu solve"
                             && contains(failure.what(), "invalid tuned state"),
                         description + " lost its fail-closed diagnostic");
        }
    };

    natlha::detail::EWSBTuneResult shortTunedState = tunedState(1.0, 1);
    shortTunedState.state.pop_back();
    rejectsInvalidTunedState(shortTunedState, "short high-precision tuned state");

    natlha::detail::EWSBTuneResult shortTunedDoubleState = tunedState(1.0, 1);
    shortTunedDoubleState.doubleState.pop_back();
    rejectsInvalidTunedState(shortTunedDoubleState, "short double tuned state");

    natlha::detail::EWSBTuneResult nonfiniteTunedState = tunedState(1.0, 1);
    nonfiniteTunedState.state[17] =
        std::numeric_limits<high_prec_float>::quiet_NaN();
    rejectsInvalidTunedState(nonfiniteTunedState, "non-finite high-precision tuned state");

    natlha::detail::EWSBTuneResult nonfiniteTunedDoubleState = tunedState(1.0, 1);
    nonfiniteTunedDoubleState.doubleState[17] =
        std::numeric_limits<double>::infinity();
    rejectsInvalidTunedState(nonfiniteTunedDoubleState, "non-finite double tuned state");

    natlha::detail::EWSBTuneResult inconsistentTunedState = tunedState(1.0, 1);
    inconsistentTunedState.doubleState[17] = 1.0;
    rejectsInvalidTunedState(inconsistentTunedState, "inconsistent tuned state representations");

    natlha::detail::EWSBTuneResult shortTunedRadCorrs = tunedState(1.0, 1);
    shortTunedRadCorrs.radCorrs.pop_back();
    rejectsInvalidTunedState(shortTunedRadCorrs, "short tuned radiative corrections");

    natlha::detail::EWSBTuneResult nonfiniteTunedRadCorrs = tunedState(1.0, 1);
    nonfiniteTunedRadCorrs.radCorrs[1] =
        std::numeric_limits<high_prec_float>::infinity();
    rejectsInvalidTunedState(nonfiniteTunedRadCorrs, "non-finite tuned radiative corrections");

    natlha::detail::EWSBTuneResult nonfiniteTunedRelation = tunedState(1.0, 1);
    nonfiniteTunedRelation.relationMZ2 =
        std::numeric_limits<high_prec_float>::quiet_NaN();
    rejectsInvalidTunedState(nonfiniteTunedRelation, "non-finite tuned mZ relation");

    natlha::detail::EWSBTuneResult negativeTunedResidual = tunedState(1.0, 1);
    negativeTunedResidual.squaredDifference = high_prec_float(-1);
    rejectsInvalidTunedState(negativeTunedResidual, "negative tuned squared residual");

    natlha::detail::EWSBTuneResult negativeTunedIterations = tunedState(1.0, -1);
    rejectsInvalidTunedState(negativeTunedIterations, "negative tuned iteration count");

    natlha::detail::EWSBTuneResult excessiveTunedIterations = tunedState(1.0, 101);
    rejectsInvalidTunedState(excessiveTunedIterations, "excessive tuned iteration count");

    ok &= expect(invalidTunedStopCalls == 0,
                 "invalid tuned callback output reached stop-scale evaluation");

    int endpointIterationTunes = 0;
    natlha::detail::JointQSusyOperations endpointIterationOperations =
        firstResidualOperations;
    endpointIterationOperations.tuneMu = [&] (
        const std::vector<double>&, const high_prec_float&) {
        ++endpointIterationTunes;
        return tunedState(1.0, endpointIterationTunes == 1 ? 0 : 100);
    };
    const natlha::detail::JointQSusySolution endpointIterations =
        natlha::detail::solveJointQSusyMu(
            rootAt(8.0, 80.0), 20.0, 1.0e-6, 0.05, 2,
            endpointIterationOperations);
    ok &= expect(endpointIterations.diagnostics.size() == 2
                     && endpointIterationTunes == 2
                     && endpointIterations.ewsbIterations == 100,
                 "inclusive tuned iteration endpoints [0,100] were rejected or miscounted");

    int subsequentRootEvolves = 0;
    int subsequentRootSearches = 0;
    int subsequentRootTunes = 0;
    natlha::detail::JointQSusyOperations subsequentRootOperations = operations;
    subsequentRootOperations.evolve = [&](
        const std::vector<double>& state, double, double, double) {
        ++subsequentRootEvolves;
        return state;
    };
    subsequentRootOperations.findRoot = [&](
        const std::vector<double>&, double, double, double) {
        ++subsequentRootSearches;
        QSusyResult invalidRoot = rootAt(9.0, 90.0);
        invalidRoot.stateAtRoot.clear();
        return invalidRoot;
    };
    subsequentRootOperations.tuneMu = [&](
        const std::vector<double>&, const high_prec_float&) {
        ++subsequentRootTunes;
        return tunedState(1.0, 1);
    };
    subsequentRootOperations.evaluateStop = [&](
        const std::vector<double>&, double) {
        return physicalPoint(100.0 * tolerance);
    };
    try {
        natlha::detail::solveJointQSusyMu(
            rootAt(8.0, 80.0), 20.0, 1.0e-6, 0.05, 3,
            subsequentRootOperations);
        ok &= expect(false, "invalid subsequent root state reached mu tuning");
    } catch (const NumericalFailure& failure) {
        ok &= expect(failure.stage == "joint Q_SUSY/mu solve"
                         && contains(failure.what(), "invalid root state"),
                     "invalid subsequent root state lost its fail-closed diagnostic");
    }
    ok &= expect(subsequentRootEvolves == 1
                     && subsequentRootSearches == 1
                     && subsequentRootTunes == 1,
                 "invalid subsequent root state crossed the wrong orchestration boundary");

    int invalidTransportRootSearches = 0;
    const auto rejectsInvalidTransport = [&] (
        std::vector<double> injected, const std::string& description) {
        natlha::detail::JointQSusyOperations invalidTransportOperations =
            firstResidualOperations;
        invalidTransportOperations.evolve = [injected] (
            const std::vector<double>&, double, double, double) {
            return injected;
        };
        invalidTransportOperations.findRoot = [&] (
            const std::vector<double>&, double, double, double) {
            ++invalidTransportRootSearches;
            return rootAt(8.0, 80.0);
        };
        try {
            natlha::detail::solveJointQSusyMu(
                rootAt(8.0, 80.0), 20.0, 1.0e-6, 0.05, 2,
                invalidTransportOperations);
            ok &= expect(false, description + " reached the next root search");
        } catch (const NumericalFailure& failure) {
            ok &= expect(contains(failure.what(), "transported high-search state"),
                         description + " lost its fail-closed diagnostic");
        }
    };
    std::vector<double> shortTransport(43, 0.0);
    rejectsInvalidTransport(shortTransport, "short transported high-search state");
    std::vector<double> nonfiniteTransport(44, 0.0);
    nonfiniteTransport[17] = std::numeric_limits<double>::infinity();
    rejectsInvalidTransport(
        nonfiniteTransport, "non-finite transported high-search state");
    ok &= expect(invalidTransportRootSearches == 0,
                 "invalid transported high-search state reached the next root search");

    natlha::detail::JointQSusyOperations invalidOperations = operations;
    invalidOperations.tuneMu = [&](const std::vector<double>&, const high_prec_float&) {
        return tunedState(1.0, 1);
    };
    invalidOperations.evaluateStop = [&](const std::vector<double>&, double) {
        return StopScalePoint{};
    };
    try {
        natlha::detail::solveJointQSusyMu(
            rootAt(8.0, 80.0), 20.0, 1.0e-6, 0.05, 2,
            invalidOperations);
        ok &= expect(false, "nonpositive retuned stop state was silently accepted");
    } catch (const std::runtime_error& error) {
        ok &= expect(contains(error.what(), "nonpositive stop mass-square"),
                     "nonpositive retuned stop state lost its distinct diagnostic");
    }

    const auto rejectsInvalidRetunedPoint = [&] (
        StopScalePoint injected, const std::string& description) {
        natlha::detail::JointQSusyOperations invalidPointOperations = operations;
        invalidPointOperations.tuneMu = [&] (
            const std::vector<double>&, const high_prec_float&) {
            return tunedState(1.0, 1);
        };
        invalidPointOperations.evaluateStop = [injected] (
            const std::vector<double>&, double) {
            return injected;
        };
        try {
            natlha::detail::solveJointQSusyMu(
                rootAt(8.0, 80.0), 20.0, 1.0e-6, 0.05, 2,
                invalidPointOperations);
            ok &= expect(false, description + " was silently accepted");
        } catch (const NumericalFailure& failure) {
            ok &= expect(failure.stage == "joint Q_SUSY/mu solve"
                             && contains(failure.what(), "invalid retuned stop point"),
                         description + " lost its fail-closed diagnostic");
        }
    };

    StopScalePoint invalidNumericalPoint = physicalPoint(0.5 * tolerance);
    invalidNumericalPoint.numericallyValid = false;
    rejectsInvalidRetunedPoint(invalidNumericalPoint, "numerically invalid retuned stop point");

    StopScalePoint nonfiniteStopMass = physicalPoint(0.5 * tolerance);
    nonfiniteStopMass.stop1Squared = std::numeric_limits<double>::quiet_NaN();
    rejectsInvalidRetunedPoint(nonfiniteStopMass, "non-finite retuned stop mass-square");

    StopScalePoint nonfiniteStopResidual = physicalPoint(0.5 * tolerance);
    nonfiniteStopResidual.logResidual = std::numeric_limits<double>::infinity();
    rejectsInvalidRetunedPoint(nonfiniteStopResidual, "non-finite retuned stop residual");

    natlha::detail::JointQSusyOperations inconsistentPhysicalOperations = operations;
    inconsistentPhysicalOperations.tuneMu = [&] (
        const std::vector<double>&, const high_prec_float&) {
        return tunedState(1.0, 1);
    };
    inconsistentPhysicalOperations.evaluateStop = [&] (
        const std::vector<double>&, double) {
        StopScalePoint point = physicalPoint(0.5 * tolerance);
        point.stop2Squared = 0.0;
        return point;
    };
    try {
        natlha::detail::solveJointQSusyMu(
            rootAt(8.0, 80.0), 20.0, 1.0e-6, 0.05, 2,
            inconsistentPhysicalOperations);
        ok &= expect(false, "physical=true with a zero stop mass-square was accepted");
    } catch (const std::runtime_error& error) {
        ok &= expect(contains(error.what(), "nonpositive stop mass-square"),
                     "inconsistent physical stop point lost its distinct diagnostic");
    }

    int gutEvolves = 0;
    natlha::detail::GutScaleOperations gutOperations;
    gutOperations.evolve = [&](const std::vector<double>&, double, double, double) {
        std::vector<double> state(44, 0.0);
        if (gutEvolves++ == 0) {
            state[0] = 0.0;
            state[1] = 1.0;
        } else {
            state[0] = 1.0;
            state[1] = 1.0;
        }
        return state;
    };
    gutOperations.gaugeBetas = [](const std::vector<high_prec_float>&) {
        return std::vector<high_prec_float>{high_prec_float(2), high_prec_float(1)};
    };
    const natlha::detail::GutScaleSolution gut = natlha::detail::solveGutScale(
        initialState, 7.0, 20.0, 1.0e-6, 5, gutOperations);
    ok &= expect(gut.iterations == 2 && gutEvolves == 2
                     && gut.logScale == 21 && gut.state.size() == 44
                     && gut.state[0] == 1 && gut.state[1] == 1,
                 "GUT-scale solve did not converge on the state at its reported scale");

    natlha::detail::GutScaleOperations singularGut = gutOperations;
    singularGut.gaugeBetas = [](const std::vector<high_prec_float>&) {
        return std::vector<high_prec_float>{high_prec_float(1), high_prec_float(1)};
    };
    try {
        natlha::detail::solveGutScale(
            initialState, 7.0, 20.0, 1.0e-6, 5, singularGut);
        ok &= expect(false, "zero GUT beta-function difference was silently accepted");
    } catch (const NumericalFailure& failure) {
        ok &= expect(contains(failure.what(), "zero beta_g1-beta_g2"),
                     "singular GUT update lost its distinct diagnostic");
    }

    natlha::detail::GutScaleOperations nonfiniteGut = gutOperations;
    nonfiniteGut.gaugeBetas = [](const std::vector<high_prec_float>&) {
        return std::vector<high_prec_float>{
            std::numeric_limits<high_prec_float>::quiet_NaN(), high_prec_float(1)};
    };
    try {
        natlha::detail::solveGutScale(
            initialState, 7.0, 20.0, 1.0e-6, 5, nonfiniteGut);
        ok &= expect(false, "non-finite GUT beta function was silently accepted");
    } catch (const NumericalFailure& failure) {
        ok &= expect(contains(failure.what(), "beta_g1"),
                     "non-finite GUT beta function lost its diagnostic");
    }

    int capEvolves = 0;
    natlha::detail::GutScaleOperations cappedGut;
    cappedGut.evolve = [&](const std::vector<double>&, double, double, double) {
        ++capEvolves;
        std::vector<double> state(44, 0.0);
        state[1] = 1.0;
        return state;
    };
    cappedGut.gaugeBetas = gutOperations.gaugeBetas;
    try {
        natlha::detail::solveGutScale(
            initialState, 7.0, 20.0, 1.0e-6, 2, cappedGut);
        ok &= expect(false, "GUT-scale iteration cap was silently exceeded");
    } catch (const std::runtime_error& error) {
        ok &= expect(capEvolves == 2
                         && contains(error.what(), "2-iteration limit exhausted"),
                     "GUT-scale cap lost its bound or distinct diagnostic");
    }

    if (argc != 2) {
        std::cerr << "expected one SLHA integration-fixture path\n";
        return 2;
    }
    natlha::Config config;
    config.slhaPath = argv[1];
    config.computeDEW = true;
    config.computeDHS = true;
    config.qSusyMaxDeltaLogQ = 0.05;
    const natlha::Result integrated = natlha::evaluate(config);
    ok &= expect(integrated.ok, "real joint evaluate failed: " + integrated.error);
    if (integrated.ok) {
        ok &= expect(integrated.qSusyDiagnostics.size() >= 2
                         && std::abs(static_cast<double>(
                                integrated.qSusyDiagnostics.front().residual)) > tolerance
                         && std::abs(static_cast<double>(
                                integrated.qSusyDiagnostics.back().residual)) <= tolerance,
                     "real evaluate accepted its pre-retune root or missed final convergence");
        if (integrated.qSusyDiagnostics.size() >= 2) {
            const auto& previousDiagnostic =
                integrated.qSusyDiagnostics[integrated.qSusyDiagnostics.size() - 2];
            const auto& finalDiagnostic = integrated.qSusyDiagnostics.back();
            const high_prec_float fixtureMuTolerance = std::max(
                high_prec_float(odeTolerance.absolute),
                high_prec_float(odeTolerance.relative)
                    * std::max(abs(finalDiagnostic.mu), abs(previousDiagnostic.mu)));
            ok &= expect(std::abs(
                             std::log(static_cast<double>(finalDiagnostic.qSusy))
                             - std::log(static_cast<double>(previousDiagnostic.qSusy)))
                                 <= tolerance,
                         "real evaluate returned before consecutive log-Q convergence");
            ok &= expect(abs(finalDiagnostic.mu - previousDiagnostic.mu)
                                 <= fixtureMuTolerance,
                         "real evaluate returned before consecutive mu convergence");
        }
        for (const auto& diagnostic : integrated.qSusyDiagnostics) {
            ok &= expect(diagnostic.declaredMaxDeltaLogQ == 0.05
                             && diagnostic.scanSegments > 0
                             && diagnostic.maxObservedDeltaLogQ <= 0.05,
                         "non-default scan spacing did not reach a joint iteration");
        }
        ok &= expect(integrated.qSusySearchDiagnostics.size()
                            == integrated.qSusyDiagnostics.size(),
                     "root-search telemetry did not reach the public API once per search");
        if (integrated.qSusySearchDiagnostics.size()
                == integrated.qSusyDiagnostics.size()) {
            for (std::size_t i = 0; i < integrated.qSusySearchDiagnostics.size(); ++i) {
                const auto& search = integrated.qSusySearchDiagnostics[i];
                const auto& iteration = integrated.qSusyDiagnostics[i];
                ok &= expect(search.ordinal == i + 1
                                 && search.scanComplete && search.accepted
                                 && search.rootsFound == 1
                                 && std::abs(search.logScale
                                             - std::log(static_cast<double>(iteration.qSusy)))
                                        <= tolerance,
                             "public root-search telemetry diverged from its joint iteration");
            }
        }
        ok &= expect(std::abs(static_cast<double>(
                                integrated.qSusyDiagnostics.front().mu)
                              - 11628.3602) > 1.0,
                     "integration fixture no longer exercises a material mu retune");
        ok &= expect(integrated.qSusyIters
                            == static_cast<long>(integrated.qSusyDiagnostics.size())
                         && integrated.qSusyStop1Squared > 0.0
                         && integrated.qSusyStop2Squared > 0.0,
                     "real evaluate lost its joint-iteration or positive-stop diagnostics");
        ok &= expect(integrated.weakBCs.size() == 44
                         && integrated.gutBCs.size() == 44
                         && integrated.radCorrs.size() == 2
                         && integrated.gutIters > 0
                         && integrated.haveDEW && integrated.haveDHS,
                     "jointly tuned state did not reach the GUT and requested-label consumers");
    }

    return ok ? 0 : 1;
}
