#include "MSSM_RGE_solver_with_stopfinder.hpp"

#include "MSSM_RGE_solver.hpp"
#include "radcorr_calc.hpp"

#include <boost/math/tools/roots.hpp>
#include <boost/numeric/odeint.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

using State = std::vector<double>;
using Stepper = boost::numeric::odeint::runge_kutta_dopri5<State>;

constexpr std::size_t kStateSize = 44;
constexpr double kMZ = 91.1876;
constexpr double kSearchLowerScale = 500.0;
constexpr double kRootUpperScale = 1.0e11;

bool traceEnabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("NATLHA_ODE_TRACE");
        return value != nullptr && *value != '\0';
    }();
    return enabled;
}

void requireFiniteState(const State& state, const std::string& stage) {
    std::vector<std::string> invalid;
    if (state.size() != kStateSize) {
        invalid.push_back("state size=" + std::to_string(state.size()));
    } else {
        for (std::size_t i = 0; i < state.size(); ++i) {
            if (!std::isfinite(state[i])) {
                invalid.push_back("state[" + std::to_string(i) + "]");
            }
        }
    }
    if (!invalid.empty()) throw NumericalFailure(stage, invalid);
}

class UnusableNumericalBoundary {};

void incrementBoundaryCount(std::size_t& count, const std::string& name) {
    if (count == std::numeric_limits<std::size_t>::max()) {
        throw NumericalFailure("Q_SUSY root search", {name + " counter overflow"});
    }
    ++count;
}

std::string rootCountDiagnostic(
    std::size_t roots,
    std::size_t invalidBoundaries,
    std::size_t nonFiniteBoundaries) {
    return "roots=" + std::to_string(roots)
           + ", invalid_boundaries=" + std::to_string(invalidBoundaries)
           + ", nonfinite_boundaries=" + std::to_string(nonFiniteBoundaries);
}

}  // namespace

QSusyRootSearchFailure::QSusyRootSearchFailure(
    std::size_t roots,
    std::size_t invalidBoundaryCount,
    std::size_t nonFiniteBoundaryCount)
    : NumericalFailure(
          "Q_SUSY root search",
          {rootCountDiagnostic(
              roots, invalidBoundaryCount, nonFiniteBoundaryCount)}),
      rootsFound(roots),
      invalidBoundaries(invalidBoundaryCount),
      nonFiniteBoundaries(nonFiniteBoundaryCount) {}

bool qsusy_detail::sameRootScale(double first, double second) {
    return first == second;
}

void qsusy_detail::addRoot(
    std::vector<RootCandidate>& roots,
    RootCandidate candidate) {
    for (const RootCandidate& root : roots) {
        if (sameRootScale(root.logScale, candidate.logScale)) return;
    }
    roots.push_back(std::move(candidate));
}

double qsusy_detail::nextScanLogScale(
    double currentLogScale,
    double lowerLogScale,
    double maxDeltaLogQ) {
    if (!std::isfinite(currentLogScale) || !std::isfinite(lowerLogScale)
            || !std::isfinite(maxDeltaLogQ) || maxDeltaLogQ <= 0.0
            || currentLogScale <= lowerLogScale) {
        throw NumericalFailure(
            "Q_SUSY scan spacing", {"invalid scan bounds or maximum spacing"});
    }

    const double remaining = currentLogScale - lowerLogScale;
    const double nextLogScale = remaining <= maxDeltaLogQ
        ? lowerLogScale
        : std::nextafter(currentLogScale - maxDeltaLogQ, currentLogScale);
    const double observed = currentLogScale - nextLogScale;
    if (!std::isfinite(remaining) || !std::isfinite(nextLogScale)
            || !std::isfinite(observed) || nextLogScale < lowerLogScale
            || nextLogScale >= currentLogScale || observed > maxDeltaLogQ) {
        throw NumericalFailure(
            "Q_SUSY scan spacing",
            {"current=" + std::to_string(currentLogScale),
             "next=" + std::to_string(nextLogScale),
             "lower=" + std::to_string(lowerLogScale),
             "observed=" + std::to_string(observed),
             "maximum=" + std::to_string(maxDeltaLogQ)});
    }
    return nextLogScale;
}

void qsusy_detail::requireUniqueRootCount(
    std::size_t roots,
    std::size_t invalidBoundaries,
    std::size_t nonFiniteBoundaries) {
    if (roots != 1 || nonFiniteBoundaries != 0) {
        throw QSusyRootSearchFailure(
            roots, invalidBoundaries, nonFiniteBoundaries);
    }
}

void qsusy_detail::recordIsolatedNumericalBoundary(ScanState& scanState) {
    incrementBoundaryCount(scanState.invalidBoundaries, "invalid-boundary");
    incrementBoundaryCount(scanState.nonFiniteBoundaries, "nonfinite-boundary");
}

std::vector<qsusy_detail::ScanEvent> qsusy_detail::classifySegment(
    double highLogScale,
    const StopScalePoint& highPoint,
    double lowLogScale,
    const StopScalePoint& lowPoint,
    ScanState& scanState) {
    const auto recordDomain = [&](const StopScalePoint& point) {
        if (!point.numericallyValid) {
            if (!scanState.inNonFiniteDomain) {
                incrementBoundaryCount(
                    scanState.nonFiniteBoundaries, "nonfinite-boundary");
            }
            scanState.inNonFiniteDomain = true;
        } else {
            scanState.inNonFiniteDomain = false;
        }
        if (!point.numericallyValid || !point.physical) {
            if (!scanState.inInvalidDomain) {
                incrementBoundaryCount(
                    scanState.invalidBoundaries, "invalid-boundary");
            }
            scanState.inInvalidDomain = true;
        } else {
            scanState.inInvalidDomain = false;
        }
    };
    recordDomain(highPoint);
    recordDomain(lowPoint);

    const auto validRootPoint = [](const StopScalePoint& point) {
        return point.numericallyValid && point.physical;
    };
    std::vector<ScanEvent> events;
    if (validRootPoint(highPoint) && highPoint.logResidual == 0.0) {
        events.push_back({ScanEventKind::exactHigh, highLogScale, highLogScale});
    }
    if (validRootPoint(lowPoint) && lowPoint.logResidual == 0.0) {
        events.push_back({ScanEventKind::exactLow, lowLogScale, lowLogScale});
    }
    if (validRootPoint(highPoint) && validRootPoint(lowPoint)
            && highPoint.logResidual != 0.0 && lowPoint.logResidual != 0.0
            && std::signbit(highPoint.logResidual)
                   != std::signbit(lowPoint.logResidual)) {
        events.push_back({ScanEventKind::signBracket, highLogScale, lowLogScale});
    }
    return events;
}

StopScalePoint evaluateStopScalePoint(const std::vector<double>& state, double logScale) {
    requireFiniteState(state, "Q_SUSY stop matrix input");
    if (!std::isfinite(logScale)) {
        throw NumericalFailure("Q_SUSY stop matrix input", {"log scale"});
    }

    const double gPrime = std::sqrt(3.0 / 5.0) * state[0];
    const double g2 = state[1];
    const double vevDenominator = (3.0 * state[0] * state[0] / 5.0) + g2 * g2;
    const double higgsVev = std::sqrt(2.0 / vevDenominator) * kMZ;
    const double beta = std::atan(state[43]);
    const double vu = higgsVev * std::sqrt(std::pow(std::sin(beta), 2.0));
    const double vd = higgsVev * std::sqrt(std::pow(std::cos(beta), 2.0));
    const double mt = state[7] * vu;
    const double vevDifference = vu * vu - vd * vd;
    const double deltaL = vevDifference * ((gPrime * gPrime / 12.0) - (g2 * g2 / 4.0));
    const double deltaR = -vevDifference * gPrime * gPrime / 3.0;
    const double mixing = state[16] * vu - state[6] * state[7] * vd;
    const double mLL = state[29] + mt * mt + deltaL;
    const double mRR = state[35] + mt * mt + deltaR;
    const double discriminant = (mLL - mRR) * (mLL - mRR) + 4.0 * mixing * mixing;
    const double splitting = std::sqrt(discriminant);

    StopScalePoint point;
    point.stop1Squared = 0.5 * (mLL + mRR - splitting);
    point.stop2Squared = 0.5 * (mLL + mRR + splitting);
    if (!std::isfinite(point.stop1Squared) || !std::isfinite(point.stop2Squared)) {
        point.numericallyValid = false;
        return point;
    }
    if (point.stop1Squared <= 0.0 || point.stop2Squared <= 0.0) return point;

    point.physical = true;
    point.logResidual = logScale
                        - 0.25 * (std::log(point.stop1Squared)
                                  + std::log(point.stop2Squared));
    if (!std::isfinite(point.logResidual)) {
        point.numericallyValid = false;
        point.physical = false;
    }
    return point;
}

QSusyResult findQSusy(const std::vector<double>& highScaleState,
                      double highLogScale,
                      double timeStep,
                      double maxDeltaLogQ) {
    requireFiniteState(highScaleState, "Q_SUSY root search input");
    const double lowerLogScale = std::log(kSearchLowerScale);
    if (!std::isfinite(highLogScale) || highLogScale <= lowerLogScale) {
        throw NumericalFailure("Q_SUSY root search input", {"high log scale"});
    }
    if (!std::isfinite(timeStep) || timeStep == 0.0) {
        throw NumericalFailure("Q_SUSY root search input", {"time step"});
    }
    if (!std::isfinite(maxDeltaLogQ) || maxDeltaLogQ <= 0.0) {
        throw NumericalFailure(
            "Q_SUSY root search input", {"maximum delta log Q"});
    }

    const ODETolerances& tolerances = odeTolerances();
    auto dense = boost::numeric::odeint::make_dense_output(
        tolerances.absolute, tolerances.relative, Stepper());
    dense.initialize(highScaleState, highLogScale, -std::abs(timeStep));

    const double scanUpper = std::min(
        highLogScale,
        std::nextafter(std::log(kRootUpperScale), lowerLogScale));
    std::vector<qsusy_detail::RootCandidate> roots;
    std::size_t acceptedSteps = 0;
    std::size_t scanSegments = 0;
    double maxObservedDeltaLogQ = 0.0;
    std::size_t refinementEvaluations = 0;
    qsusy_detail::ScanState scanState;
    bool havePrevious = false;
    double previousLogScale = 0.0;
    State previousState;
    StopScalePoint previousPoint;

    const auto started = std::chrono::steady_clock::now();
    try {
        while (dense.current_time() > lowerLogScale) {
            if (dense.current_time() + dense.current_time_step() < lowerLogScale) {
                dense.initialize(dense.current_state(), dense.current_time(),
                                 lowerLogScale - dense.current_time());
            }
            const std::pair<double, double> interval = dense.do_step(MSSMRGESolver);
            ++acceptedSteps;

            const double segmentHigh = std::min(interval.first, scanUpper);
            const double segmentLow = std::max(interval.second, lowerLogScale);
            if (segmentHigh <= segmentLow) continue;

            State scanHighState(kStateSize);
            StopScalePoint scanHighPoint;
            if (havePrevious && previousLogScale == segmentHigh) {
                scanHighState = previousState;
                scanHighPoint = previousPoint;
            } else {
                dense.calc_state(segmentHigh, scanHighState);
                scanHighPoint = evaluateStopScalePoint(scanHighState, segmentHigh);
            }

            double scanHigh = segmentHigh;
            while (scanHigh > segmentLow) {
                const double scanLow = qsusy_detail::nextScanLogScale(
                    scanHigh, segmentLow, maxDeltaLogQ);
                const double observedDeltaLogQ = scanHigh - scanLow;
                if (scanSegments == std::numeric_limits<std::size_t>::max()) {
                    throw NumericalFailure(
                        "Q_SUSY scan spacing", {"scan-segment counter overflow"});
                }
                ++scanSegments;
                maxObservedDeltaLogQ = std::max(
                    maxObservedDeltaLogQ, observedDeltaLogQ);

                State scanLowState(kStateSize);
                dense.calc_state(scanLow, scanLowState);
                const StopScalePoint scanLowPoint =
                    evaluateStopScalePoint(scanLowState, scanLow);

                const std::vector<qsusy_detail::ScanEvent> events =
                    qsusy_detail::classifySegment(
                        scanHigh, scanHighPoint, scanLow, scanLowPoint, scanState);
                for (const qsusy_detail::ScanEvent& event : events) {
                    if (event.kind == qsusy_detail::ScanEventKind::exactHigh) {
                        qsusy_detail::addRoot(
                            roots, {scanHigh, scanHighState, scanHighPoint});
                        continue;
                    }
                    if (event.kind == qsusy_detail::ScanEventKind::exactLow) {
                        qsusy_detail::addRoot(
                            roots, {scanLow, scanLowState, scanLowPoint});
                        continue;
                    }

                    try {
                        auto residual = [&](double logScale) {
                            State state(kStateSize);
                            dense.calc_state(logScale, state);
                            const StopScalePoint point =
                                evaluateStopScalePoint(state, logScale);
                            if (!point.numericallyValid) {
                                throw UnusableNumericalBoundary{};
                            }
                            if (!point.physical) {
                                throw NumericalFailure(
                                    "Q_SUSY root search",
                                    {"nonpositive stop inside valid bracket"});
                            }
                            return point.logResidual;
                        };

                        boost::math::tools::eps_tolerance<double> tolerance;
                        boost::uintmax_t evaluations =
                            2 * static_cast<boost::uintmax_t>(
                                std::numeric_limits<double>::digits);
                        const std::pair<double, double> refined =
                            boost::math::tools::toms748_solve(
                                residual, scanLow, scanHigh,
                                scanLowPoint.logResidual,
                                scanHighPoint.logResidual, tolerance, evaluations);
                        refinementEvaluations += static_cast<std::size_t>(evaluations);
                        if (!tolerance(refined.first, refined.second)) {
                            throw NumericalFailure(
                                "Q_SUSY root search",
                                {"TOMS 748 evaluation budget exhausted"});
                        }
                        const double lowerResidual = residual(refined.first);
                        const double upperResidual = residual(refined.second);
                        if (lowerResidual != 0.0 && upperResidual != 0.0
                                && std::signbit(lowerResidual)
                                       == std::signbit(upperResidual)) {
                            throw NumericalFailure(
                                "Q_SUSY root search",
                                {"refined interval lost its sign bracket"});
                        }

                        const double rootLogScale =
                            refined.first
                            + 0.5 * (refined.second - refined.first);
                        State rootState(kStateSize);
                        dense.calc_state(rootLogScale, rootState);
                        const StopScalePoint rootPoint =
                            evaluateStopScalePoint(rootState, rootLogScale);
                        if (!rootPoint.numericallyValid) {
                            throw UnusableNumericalBoundary{};
                        }
                        if (!rootPoint.physical) {
                            throw NumericalFailure(
                                "Q_SUSY root search",
                                {"refined root has nonpositive stop"});
                        }
                        if (std::abs(rootPoint.logResidual)
                                > std::max(
                                    tolerances.absolute, tolerances.relative)) {
                            throw NumericalFailure(
                                "Q_SUSY root search",
                                {"refined root failed residual gate"});
                        }
                        qsusy_detail::addRoot(
                            roots,
                            {rootLogScale, std::move(rootState), rootPoint});
                    } catch (const UnusableNumericalBoundary&) {
                        qsusy_detail::recordIsolatedNumericalBoundary(scanState);
                    }
                }

                scanHigh = scanLow;
                scanHighState = std::move(scanLowState);
                scanHighPoint = scanLowPoint;
            }

            previousLogScale = segmentLow;
            previousState = std::move(scanHighState);
            previousPoint = scanHighPoint;
            havePrevious = true;
        }
    } catch (const NumericalFailure&) {
        throw;
    } catch (const std::exception& error) {
        throw NumericalFailure("Q_SUSY root search", {error.what()});
    }

    qsusy_detail::requireUniqueRootCount(
        roots.size(), scanState.invalidBoundaries,
        scanState.nonFiniteBoundaries);

    QSusyResult result;
    result.stateAtRoot = std::move(roots.front().state);
    result.logScale = roots.front().logScale;
    result.scale = std::exp(result.logScale);
    result.residual = roots.front().point.logResidual;
    result.stop1Squared = roots.front().point.stop1Squared;
    result.stop2Squared = roots.front().point.stop2Squared;
    result.acceptedSteps = acceptedSteps;
    result.declaredMaxDeltaLogQ = maxDeltaLogQ;
    result.scanSegments = scanSegments;
    result.maxObservedDeltaLogQ = maxObservedDeltaLogQ;
    result.rootsFound = roots.size();
    result.invalidBoundaries = scanState.invalidBoundaries;
    result.refinementEvaluations = refinementEvaluations;
    result.diagnostic =
        "one positive-stop sign-changing or exact root at the declared scan spacing";

    if (!std::isfinite(result.scale) || result.scale <= 0.0) {
        throw NumericalFailure("Q_SUSY root search", {"nonpositive or non-finite scale"});
    }
    if (traceEnabled()) {
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        std::cerr << "# stopfinder_trace seconds " << elapsed
                  << "  t_from " << highLogScale << "  t_to " << lowerLogScale
                  << "  t_root " << result.logScale
                  << "  residual " << result.residual
                  << "  accepted_steps " << result.acceptedSteps
                  << "  max_dlogQ " << result.declaredMaxDeltaLogQ
                  << "  scan_segments " << result.scanSegments
                  << "  max_observed_dlogQ " << result.maxObservedDeltaLogQ
                  << "  roots " << result.rootsFound
                  << "  invalid_boundaries " << result.invalidBoundaries
                  << "  refinement_evaluations " << result.refinementEvaluations
                  << "  eps_abs " << tolerances.absolute
                  << "  eps_rel " << tolerances.relative << "\n";
    }
    return result;
}
