// Non-interactive implementation of the natLHA pipeline. See include/natlha_api.hpp for the
// contract this satisfies.
//
// The sequence mirrors what the interactive program does, in this order:
//   1. read the SLHA and assemble the 44-entry weak-scale state at the file's own scale
//   2. run to Q = 1e12, then locate exactly one positive-stop sign-changing or exact
//      Q_SUSY root at the declared maximum log(Q) scan spacing
//   3. re-solve EWSB for mu on the state evaluated at the accepted Q_SUSY root
//   4. transport only the solved mu and derived b back to an otherwise immutable high-search
//      state, then repeat the full-window root search and mu solve until the post-retune stop
//      residual, consecutive log(Q_SUSY), and consecutive mu satisfy their ODE-tolerance gates
//   5. iterate to the scale where g1 = g2 to get the GUT-scale state
//   6. evaluate whichever measures were requested
//
// Steps 1-5 are shared setup for every measure, which is the whole reason this
// function exists: a batch caller pays for them once per point regardless of how many labels
// it asks for.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/multiprecision/mpfr.hpp>

#include "natlha_api.hpp"
#include "natlha_api_detail.hpp"
#include "MSSM_RGE_solver.hpp"
#include "MSSM_RGE_solver_with_stopfinder.hpp"
#include "mZ_numsolver.hpp"
#include "radcorr_calc.hpp"
#include "shared_helpers.hpp"
#include "slhaea.h"

namespace natlha {

void detail::failLabelRow(Result& result, std::string error) {
    result.ok = false;
    result.error = std::move(error);
    result.haveDEW = false;
    result.haveDHS = false;
    result.haveDBG = false;
    result.haveDSN = false;
    result.deltaEW = 0;
    result.deltaHS = 0;
    result.deltaBG = 0;
    result.deltaSN = 0;
    result.snTotalNvac = 0;
    result.dewContributions.clear();
    result.dhsContributions.clear();
    result.dbgContributions.clear();
    result.dsnContributions.clear();
}

namespace {

/// m_Z used throughout, matching the value the interactive path hard-codes.
const double kMZ = 91.1876;
const double kQSusySearchHigh = 1.0e12;
const long kMaxEWSBIterations = 100;
const long kMaxQSusyIterations = 100;
const long kMaxGutIterations = 100;
constexpr std::size_t kStateSize = 44;
constexpr std::size_t kMuIndex = 6;
constexpr std::size_t kBIndex = 42;

void requireFiniteJointState(
    const std::vector<double>& state,
    const std::string& stage) {
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

std::vector<double> toDoubleState(const std::vector<high_prec_float>& state,
                                  const std::string& stage) {
    std::vector<double> converted;
    converted.reserve(state.size());
    for (std::size_t i = 0; i < state.size(); ++i) {
        const double value = static_cast<double>(state[i]);
        if (!std::isfinite(value)) {
            throw NumericalFailure(stage, {"state[" + std::to_string(i) + "]"});
        }
        converted.push_back(value);
    }
    return converted;
}

detail::EWSBTuneResult tuneEWSBMu(const std::vector<double>& inputState,
                                  const high_prec_float& qSusy) {
    if (inputState.size() != 44) {
        throw NumericalFailure(
            "EWSB mu solve input", {"state size=" + std::to_string(inputState.size())});
    }
    if (!(boost::math::isfinite)(qSusy) || qSusy <= 0) {
        throw NumericalFailure("EWSB mu solve input", {"Q_SUSY"});
    }

    detail::EWSBTuneResult result;
    result.state.reserve(inputState.size());
    for (const double value : inputState) {
        if (!std::isfinite(value)) {
            throw NumericalFailure("EWSB mu solve input", {"non-finite state"});
        }
        result.state.push_back(high_prec_float(value));
    }

    const high_prec_float tanBeta = result.state[43];
    const high_prec_float tanBetaSquared = tanBeta * tanBeta;
    const high_prec_float denominator = tanBetaSquared - 1;
    const high_prec_float mHdSquared = result.state[26];
    const high_prec_float mHuSquared = result.state[25];
    const bool negativeMu = result.state[6] < 0;
    high_prec_float muSquared = result.state[6] * result.state[6];
    const high_prec_float convergenceSquared =
        high_prec_float(1) / high_prec_float(1000000000000.0);
    result.squaredDifference = high_prec_float(100);
    result.radCorrs = radcorr_calc(result.state, qSusy, high_prec_float(kMZ * kMZ));

    while (result.squaredDifference > convergenceSquared
           && result.iterations < kMaxEWSBIterations) {
        ++result.iterations;
        const high_prec_float newMuSquared =
            ((mHdSquared + result.radCorrs[1]
              - (mHuSquared + result.radCorrs[0]) * tanBetaSquared)
             / denominator) - high_prec_float(kMZ * kMZ / 2.0);
        if (!(boost::math::isfinite)(newMuSquared)) {
            throw NumericalFailure("EWSB mu solve", {"non-finite mu^2"});
        }
        if (newMuSquared < 0) {
            throw std::runtime_error(
                "EWSB mu solve: negative mu^2; no real mu solution");
        }

        const high_prec_float muMagnitude = sqrt(newMuSquared);
        result.state[6] = negativeMu ? -muMagnitude : muMagnitude;
        result.radCorrs = radcorr_calc(
            result.state, qSusy, high_prec_float(kMZ * kMZ));
        result.squaredDifference = pow(muSquared - newMuSquared, 2);
        if (!(boost::math::isfinite)(result.squaredDifference)) {
            throw NumericalFailure("EWSB mu solve", {"non-finite convergence residual"});
        }
        muSquared = newMuSquared;
    }
    if (result.squaredDifference > convergenceSquared) {
        throw std::runtime_error("EWSB mu solve: 100-iteration limit exhausted");
    }

    result.relationMZ2 =
        2 * ((mHdSquared + result.radCorrs[1]
              - (mHuSquared + result.radCorrs[0]) * tanBetaSquared)
             / denominator) - 2 * muSquared;
    if (!(boost::math::isfinite)(result.relationMZ2)) {
        throw NumericalFailure("EWSB mu solve", {"non-finite relation mZ^2"});
    }

    const high_prec_float beta = atan(tanBeta);
    result.state[42] = sin(2 * beta)
                       * (mHuSquared + result.radCorrs[0]
                          + mHdSquared + result.radCorrs[1] + 2 * muSquared) / 2;
    if (!(boost::math::isfinite)(result.state[42])) {
        throw NumericalFailure("EWSB mu solve", {"non-finite b=B*mu"});
    }
    result.doubleState = toDoubleState(result.state, "EWSB mu solve output");
    return result;
}

void requireValidJointRoot(
    const QSusyResult& root,
    double highLogScale,
    double maxDeltaLogQ) {
    const ODETolerances& tolerances = odeTolerances();
    const double rootTolerance = std::max(
        tolerances.absolute, tolerances.relative);
    const std::size_t maxRefinementEvaluationsPerSegment =
        2 * static_cast<std::size_t>(std::numeric_limits<double>::digits);
    const bool refinementCountValid =
        root.scanSegments <= std::numeric_limits<std::size_t>::max()
                                 / maxRefinementEvaluationsPerSegment
        && root.refinementEvaluations
               <= root.scanSegments * maxRefinementEvaluationsPerSegment;
    if (!std::isfinite(root.logScale)
            || !std::isfinite(root.scale) || root.scale <= 0.0
            || root.logScale >= highLogScale
            || std::abs(std::log(root.scale) - root.logScale) > rootTolerance
            || !std::isfinite(root.residual) || std::abs(root.residual) > rootTolerance
            || !std::isfinite(root.stop1Squared) || root.stop1Squared <= 0.0
            || !std::isfinite(root.stop2Squared) || root.stop2Squared <= 0.0
            || root.acceptedSteps == 0 || root.scanSegments == 0
            || !std::isfinite(root.declaredMaxDeltaLogQ)
            || root.declaredMaxDeltaLogQ != maxDeltaLogQ
            || !std::isfinite(root.maxObservedDeltaLogQ)
            || root.maxObservedDeltaLogQ <= 0.0
            || root.maxObservedDeltaLogQ > root.declaredMaxDeltaLogQ
            || root.rootsFound != 1 || !refinementCountValid
            || root.diagnostic.empty()) {
        throw NumericalFailure(
            "joint Q_SUSY/mu solve",
            {"invalid root returned by root search"});
    }
    if (root.stateAtRoot.size() != 44
            || !std::all_of(
                root.stateAtRoot.begin(), root.stateAtRoot.end(),
                [](double value) { return std::isfinite(value); })) {
        throw NumericalFailure(
            "joint Q_SUSY/mu solve", {"invalid root state returned by root search"});
    }
}

void requireValidTunedState(const detail::EWSBTuneResult& tuned) {
    bool stateValid = tuned.state.size() == 44 && tuned.doubleState.size() == 44;
    if (stateValid) {
        for (std::size_t i = 0; i < tuned.state.size(); ++i) {
            const double converted = static_cast<double>(tuned.state[i]);
            if (!(boost::math::isfinite)(tuned.state[i])
                    || !std::isfinite(converted)
                    || !std::isfinite(tuned.doubleState[i])
                    || converted != tuned.doubleState[i]) {
                stateValid = false;
                break;
            }
        }
    }
    const bool radCorrsValid = tuned.radCorrs.size() == 2
        && std::all_of(
            tuned.radCorrs.begin(), tuned.radCorrs.end(),
            [](const high_prec_float& value) {
                return (boost::math::isfinite)(value);
            });
    if (!stateValid || !radCorrsValid
            || !(boost::math::isfinite)(tuned.relationMZ2)
            || !(boost::math::isfinite)(tuned.squaredDifference)
            || tuned.squaredDifference < 0
            || tuned.iterations < 0 || tuned.iterations > kMaxEWSBIterations) {
        throw NumericalFailure(
            "joint Q_SUSY/mu solve",
            {"invalid tuned state returned by EWSB solve"});
    }
}

void requireValidRetunedPoint(const StopScalePoint& point) {
    if (!point.numericallyValid
            || !std::isfinite(point.stop1Squared)
            || !std::isfinite(point.stop2Squared)
            || !std::isfinite(point.logResidual)) {
        throw NumericalFailure(
            "joint Q_SUSY/mu solve",
            {"invalid retuned stop point returned by stop evaluation"});
    }
    if (!point.physical || point.stop1Squared <= 0.0 || point.stop2Squared <= 0.0) {
        throw std::runtime_error(
            "joint Q_SUSY/mu solve: retuned state has a nonpositive stop mass-square");
    }
}

bool jointResidualConverged(const StopScalePoint& point) {
    const ODETolerances& tolerances = odeTolerances();
    return std::abs(point.logResidual)
           <= std::max(tolerances.absolute, tolerances.relative);
}

QSusyIterationDiagnostic makeQSusyDiagnostic(
    long iteration,
    const QSusyResult& root,
    const StopScalePoint& retunedPoint,
    const detail::EWSBTuneResult& tuned) {
    QSusyIterationDiagnostic diagnostic;
    diagnostic.iteration = iteration;
    diagnostic.qSusy = root.scale;
    diagnostic.residual = retunedPoint.logResidual;
    diagnostic.mu = tuned.state[6];
    diagnostic.stop1Squared = retunedPoint.stop1Squared;
    diagnostic.stop2Squared = retunedPoint.stop2Squared;
    diagnostic.acceptedSteps = root.acceptedSteps;
    diagnostic.declaredMaxDeltaLogQ = root.declaredMaxDeltaLogQ;
    diagnostic.scanSegments = root.scanSegments;
    diagnostic.maxObservedDeltaLogQ = root.maxObservedDeltaLogQ;
    diagnostic.rootsFound = root.rootsFound;
    diagnostic.invalidBoundaries = root.invalidBoundaries;
    diagnostic.refinementEvaluations = root.refinementEvaluations;
    return diagnostic;
}

std::vector<high_prec_float> toHighPrecisionState(
    const std::vector<double>& state,
    const std::string& stage) {
    if (state.size() != 44) {
        throw NumericalFailure(
            stage, {"state size=" + std::to_string(state.size())});
    }
    std::vector<high_prec_float> converted;
    converted.reserve(state.size());
    for (std::size_t i = 0; i < state.size(); ++i) {
        if (!std::isfinite(state[i])) {
            throw NumericalFailure(stage, {"state[" + std::to_string(i) + "]"});
        }
        converted.push_back(high_prec_float(state[i]));
    }
    return converted;
}

void requireFiniteGutValue(
    const high_prec_float& value,
    const std::string& name) {
    if (!(boost::math::isfinite)(value)) {
        throw NumericalFailure("GUT-scale solve", {name});
    }
}

}  // namespace

QSusyResult detail::runAuditedQSusySearch(
    std::vector<QSusySearchDiagnostic>& diagnostics,
    const std::function<QSusyResult()>& search) {
    try {
        QSusyResult root = search();
        QSusySearchDiagnostic diagnostic;
        diagnostic.ordinal = diagnostics.size() + 1;
        diagnostic.scanComplete = true;
        diagnostic.accepted = true;
        diagnostic.logScale = root.logScale;
        diagnostic.rootsFound = root.rootsFound;
        diagnostic.invalidBoundaries = root.invalidBoundaries;
        diagnostics.push_back(diagnostic);
        return root;
    } catch (const QSusyRootSearchFailure& failure) {
        QSusySearchDiagnostic diagnostic;
        diagnostic.ordinal = diagnostics.size() + 1;
        diagnostic.scanComplete = true;
        diagnostic.rootsFound = failure.rootsFound;
        diagnostic.invalidBoundaries = failure.invalidBoundaries;
        diagnostic.nonFiniteBoundaries = failure.nonFiniteBoundaries;
        diagnostics.push_back(diagnostic);
        throw;
    } catch (...) {
        QSusySearchDiagnostic diagnostic;
        diagnostic.ordinal = diagnostics.size() + 1;
        diagnostics.push_back(diagnostic);
        throw;
    }
}

detail::JointQSusyConvergenceFailure::JointQSusyConvergenceFailure(
    std::string message,
    std::vector<QSusyIterationDiagnostic> completedDiagnostics,
    long completedEWSBIterations)
    : std::runtime_error(std::move(message)),
      diagnostics(std::move(completedDiagnostics)),
      ewsbIterations(completedEWSBIterations) {}

detail::JointQSusySolution detail::solveJointQSusyMu(
    QSusyResult initialRoot,
    double highLogScale,
    double timeStep,
    double maxDeltaLogQ,
    long maxIterations,
    const JointQSusyOperations& operations) {
    if (!std::isfinite(highLogScale)
            || !std::isfinite(timeStep) || timeStep == 0.0 || maxIterations <= 0
            || maxIterations
                   > std::numeric_limits<long>::max() / kMaxEWSBIterations
            || !std::isfinite(maxDeltaLogQ) || maxDeltaLogQ <= 0.0
            || !operations.evolve || !operations.findRoot
            || !operations.tuneMu || !operations.evaluateStop) {
        throw NumericalFailure("joint Q_SUSY/mu solve input", {"invalid orchestration input"});
    }
    requireFiniteJointState(
        operations.immutableHighState,
        "joint Q_SUSY/mu solve immutable high-search state");

    JointQSusySolution solution;
    QSusyResult root = std::move(initialRoot);
    std::vector<double> qHistory;
    std::vector<high_prec_float> muHistory;

    for (long attempt = 0; attempt < maxIterations; ++attempt) {
        const double qLog = root.logScale;
        requireValidJointRoot(root, highLogScale, maxDeltaLogQ);
        EWSBTuneResult tuned = operations.tuneMu(
            root.stateAtRoot, high_prec_float(root.scale));
        requireValidTunedState(tuned);
        solution.ewsbIterations += tuned.iterations;
        const StopScalePoint retunedPoint = operations.evaluateStop(tuned.doubleState, qLog);
        requireValidRetunedPoint(retunedPoint);

        solution.diagnostics.push_back(makeQSusyDiagnostic(
            attempt + 1, root, retunedPoint, tuned));
        qHistory.push_back(qLog);
        muHistory.push_back(tuned.state[6]);

        bool qConverged = false;
        bool muConverged = false;
        if (qHistory.size() >= 2) {
            const ODETolerances& tolerances = odeTolerances();
            qConverged = std::abs(qHistory.back() - qHistory[qHistory.size() - 2])
                         <= std::max(tolerances.absolute, tolerances.relative);

            const high_prec_float& currentMu = muHistory.back();
            const high_prec_float& previousMu = muHistory[muHistory.size() - 2];
            const high_prec_float muTolerance = std::max(
                high_prec_float(tolerances.absolute),
                high_prec_float(tolerances.relative)
                    * std::max(abs(currentMu), abs(previousMu)));
            muConverged = abs(currentMu - previousMu) <= muTolerance;
        }
        if (jointResidualConverged(retunedPoint) && qConverged && muConverged) {
            solution.root = std::move(root);
            solution.tuned = std::move(tuned);
            solution.retunedPoint = retunedPoint;
            return solution;
        }

        if (!qConverged && qHistory.size() >= 3) {
            const double priorCycleLog = qHistory[qHistory.size() - 3];
            const ODETolerances& tolerances = odeTolerances();
            if (std::abs(qLog - priorCycleLog)
                    <= std::max(tolerances.absolute, tolerances.relative)) {
                throw JointQSusyConvergenceFailure(
                    "joint Q_SUSY/mu solve: log(Q_SUSY) repeated at lag two "
                    "before joint convergence",
                    std::move(solution.diagnostics), solution.ewsbIterations);
            }
        }
        if (attempt + 1 == maxIterations) break;

        const std::vector<double> transportedHighState = operations.evolve(
            tuned.doubleState, qLog, highLogScale, std::abs(timeStep));
        requireFiniteJointState(
            transportedHighState,
            "joint Q_SUSY/mu solve transported high-search state");
        std::vector<double> nextHighState = operations.immutableHighState;
        nextHighState[kMuIndex] = transportedHighState[kMuIndex];
        nextHighState[kBIndex] = transportedHighState[kBIndex];
        root = operations.findRoot(
            nextHighState, highLogScale, -std::abs(timeStep), maxDeltaLogQ);
    }

    throw JointQSusyConvergenceFailure(
        "joint Q_SUSY/mu solve: " + std::to_string(maxIterations)
            + "-iteration limit exhausted",
        std::move(solution.diagnostics), solution.ewsbIterations);
}

detail::GutScaleSolution detail::solveGutScale(
    const std::vector<double>& weakState,
    double weakLogScale,
    double initialHighLogScale,
    double timeStep,
    long maxIterations,
    const GutScaleOperations& operations) {
    if (weakState.size() != 44 || !std::isfinite(weakLogScale)
            || !std::isfinite(initialHighLogScale)
            || !std::isfinite(timeStep) || timeStep == 0.0
            || maxIterations < 2 || !operations.evolve || !operations.gaugeBetas) {
        throw NumericalFailure("GUT-scale solve input", {"invalid orchestration input"});
    }

    const high_prec_float toleranceSquared =
        high_prec_float(1.0) / high_prec_float(1000000000000.0);
    GutScaleSolution solution;
    std::vector<double> running = operations.evolve(
        weakState, weakLogScale, initialHighLogScale,
        std::copysign(std::abs(timeStep), initialHighLogScale - weakLogScale));
    solution.state = toHighPrecisionState(running, "GUT-scale initial evolution");
    solution.iterations = 1;

    const auto nextTrial = [&](const std::vector<high_prec_float>& state,
                               const high_prec_float& currentLogScale) {
        const std::vector<high_prec_float> betas = operations.gaugeBetas(state);
        if (betas.size() < 2) {
            throw NumericalFailure("GUT-scale solve", {"gauge beta count"});
        }
        requireFiniteGutValue(betas[0], "beta_g1");
        requireFiniteGutValue(betas[1], "beta_g2");
        const high_prec_float denominator = betas[0] - betas[1];
        if (denominator == 0) {
            throw NumericalFailure("GUT-scale solve", {"zero beta_g1-beta_g2"});
        }
        const high_prec_float next =
            currentLogScale + (state[1] - state[0]) / denominator;
        requireFiniteGutValue(next, "trial log scale");
        return next;
    };

    high_prec_float currentLogScale = nextTrial(
        solution.state, high_prec_float(initialHighLogScale));
    requireFiniteGutValue(currentLogScale, "initial trial log scale");

    while (solution.iterations < maxIterations) {
        const double currentLogScaleDouble = static_cast<double>(currentLogScale);
        if (!std::isfinite(currentLogScaleDouble) || currentLogScale == 0) {
            throw NumericalFailure("GUT-scale solve", {"current trial log scale"});
        }
        running = operations.evolve(
            weakState, weakLogScale, currentLogScaleDouble,
            std::copysign(std::abs(timeStep), currentLogScaleDouble - weakLogScale));
        solution.state = toHighPrecisionState(running, "GUT-scale trial evolution");
        ++solution.iterations;

        const high_prec_float nextLogScale =
            nextTrial(solution.state, currentLogScale);
        const high_prec_float squaredDifference = pow(
            high_prec_float(1.0) - nextLogScale / currentLogScale,
            high_prec_float(2.0));
        requireFiniteGutValue(squaredDifference, "convergence residual");
        if (squaredDifference <= toleranceSquared) {
            solution.logScale = currentLogScale;
            return solution;
        }
        currentLogScale = nextLogScale;
    }

    throw std::runtime_error(
        "GUT-scale solve: " + std::to_string(maxIterations)
        + "-iteration limit exhausted");
}

void detail::requireFiniteDBGContributions(
    const std::vector<LabeledValueBG>& contributions) {
    std::vector<std::string> invalid;
    for (const auto& contribution : contributions) {
        if (!(boost::math::isfinite)(contribution.value)) {
            invalid.push_back(contribution.label);
        }
    }
    if (!invalid.empty()) {
        throw NumericalFailure("DBG_calc contributions", invalid);
    }
}

Result evaluate(const Config & cfg) {
    Result out;
    try {
        if (cfg.computeDSN && cfg.snMode != 3) {
            throw std::invalid_argument(
                "non-interactive delta_SN supports only mode 3; "
                "capital Delta_SN continuation is deferred");
        }
        if (!std::isfinite(cfg.qSusyMaxDeltaLogQ)
                || cfg.qSusyMaxDeltaLogQ <= 0.0) {
            throw std::invalid_argument(
                "Q_SUSY maximum delta log Q must be finite and positive");
        }
        std::ifstream ifs(cfg.slhaPath);
        if (!ifs.good()) {
            out.error = "cannot open SLHA file: " + cfg.slhaPath;
            return out;
        }
        SLHAea::Coll input(ifs);

        // Both accessors swallow a missing entry and hand back `defaultValue`, which is what
        // makes the block-convention fallbacks below work: an absent block reads as zero.
        auto vecVal = [&](const std::string & block, int i, double defaultValue = 0.0) -> double {
            try {
                return SLHAea::to<double>(input.at(block).at(std::to_string(i)).at(1));
            } catch (const std::exception &) {
                return defaultValue;
            }
        };
        auto matVal = [&](const std::string & block, int i, int j, double defaultValue = 0.0) -> double {
            try {
                return SLHAea::to<double>(input.at(block).at(i, j).at(2));
            } catch (const std::exception &) {
                return defaultValue;
            }
        };

        // ---- Higgs sector -------------------------------------------------------------
        high_prec_float tanb = high_prec_float(vecVal("HMIX", 2));
        high_prec_float muQ = high_prec_float(vecVal("HMIX", 1));

        // ---- Yukawas ------------------------------------------------------------------
        // Generations 1 and 2 are only estimated from the third when the SLHA omits them.
        // A file carrying full 3x3 YU/YD/YE never reaches these fallbacks, which is why a
        // generator should be asked to emit them rather than relying on these ratios.
        high_prec_float y_t = high_prec_float(matVal("YU", 3, 3));
        high_prec_float y_c = high_prec_float(matVal("YU", 2, 2));
        if (y_c == 0.0) y_c = high_prec_float(0.003882759826930082) * y_t;
        high_prec_float y_u = high_prec_float(matVal("YU", 1, 1));
        if (y_u == 0.0) y_u = high_prec_float(7.779613278615955e-6) * y_t;
        high_prec_float y_b = high_prec_float(matVal("YD", 3, 3));
        high_prec_float y_s = high_prec_float(matVal("YD", 2, 2));
        if (y_s == 0.0) y_s = high_prec_float(0.0206648802754076) * y_b;
        high_prec_float y_d = high_prec_float(matVal("YD", 1, 1));
        if (y_d == 0.0) y_d = high_prec_float(0.0010117174290779725) * y_b;
        high_prec_float y_tau = high_prec_float(matVal("YE", 3, 3));
        high_prec_float y_mu = high_prec_float(matVal("YE", 2, 2));
        if (y_mu == 0.0) y_mu = high_prec_float(0.05792142442492775) * y_tau;
        high_prec_float y_e = high_prec_float(matVal("YE", 1, 1));
        if (y_e == 0.0) y_e = high_prec_float(0.0002801267571260388) * y_tau;

        // ---- Gauge couplings ----------------------------------------------------------
        high_prec_float g_pr = high_prec_float(vecVal("GAUGE", 1));
        high_prec_float g_2 = high_prec_float(vecVal("GAUGE", 2));
        high_prec_float g_s = high_prec_float(vecVal("GAUGE", 3));

        // ---- Trilinears ---------------------------------------------------------------
        // Two conventions in the wild. TU/TD/TE hold the soft term a_ij directly, so the
        // multiplier starts at 1. AU/AD/AE hold the RATIO A_ij, so the multiplier starts at
        // the corresponding Yukawa and the product is again a_ij. Which one is present is
        // decided by whether the third-generation TU/TD/TE entries are all absent.
        high_prec_float a_t, a_c, a_u, a_b, a_s, a_d, a_tau, a_mu, a_e;
        const bool haveT = !(matVal("TU", 3, 3) == 0.0 && matVal("TD", 3, 3) == 0.0 &&
                             matVal("TE", 3, 3) == 0.0);
        std::string uBlock, dBlock, eBlock;
        if (haveT) {
            uBlock = "TU"; dBlock = "TD"; eBlock = "TE";
            a_t = a_c = a_u = a_b = a_s = a_d = a_tau = a_mu = a_e = high_prec_float(1.0);
        } else {
            uBlock = "AU"; dBlock = "AD"; eBlock = "AE";
            a_t = y_t; a_c = y_c; a_u = y_u;
            a_b = y_b; a_s = y_s; a_d = y_d;
            a_tau = y_tau; a_mu = y_mu; a_e = y_e;
        }
        a_t   *= high_prec_float(matVal(uBlock, 3, 3));
        a_c   *= high_prec_float(matVal(uBlock, 2, 2));
        a_u   *= high_prec_float(matVal(uBlock, 1, 1));
        a_b   *= high_prec_float(matVal(dBlock, 3, 3));
        a_s   *= high_prec_float(matVal(dBlock, 2, 2));
        a_d   *= high_prec_float(matVal(dBlock, 1, 1));
        a_tau *= high_prec_float(matVal(eBlock, 3, 3));
        a_mu  *= high_prec_float(matVal(eBlock, 2, 2));
        a_e   *= high_prec_float(matVal(eBlock, 1, 1));

        // ---- Gauginos and soft Higgs masses -------------------------------------------
        high_prec_float my_M1 = high_prec_float(vecVal("MSOFT", 1));
        high_prec_float my_M2 = high_prec_float(vecVal("MSOFT", 2));
        high_prec_float my_M3 = high_prec_float(vecVal("MSOFT", 3));
        high_prec_float mHusq = high_prec_float(vecVal("MSOFT", 22));
        high_prec_float mHdsq = high_prec_float(vecVal("MSOFT", 21));

        // ---- Sfermion soft masses -----------------------------------------------------
        // SLHA2 gives squared masses in MSQ2/MSU2/MSD2/MSL2/MSE2; SLHA1 gives MASSES in
        // MSOFT 41-49 and 31-36, which must therefore be squared. SOFTSUSY's default output
        // is the latter, so this fallback is the common path, not an edge case.
        high_prec_float mQ1sq, mQ2sq, mQ3sq, mL1sq, mL2sq, mL3sq;
        high_prec_float mU1sq, mU2sq, mU3sq, mD1sq, mD2sq, mD3sq, mE1sq, mE2sq, mE3sq;
        const bool haveSlha2Soft = !(matVal("MSQ2", 3, 3) == 0.0 && matVal("MSU2", 3, 3) == 0.0 &&
                                     matVal("MSE2", 3, 3) == 0.0);
        if (!haveSlha2Soft) {
            mQ3sq = pow(high_prec_float(vecVal("MSOFT", 43)), 2.0);
            mQ2sq = pow(high_prec_float(vecVal("MSOFT", 42)), 2.0);
            mQ1sq = pow(high_prec_float(vecVal("MSOFT", 41)), 2.0);
            mL3sq = pow(high_prec_float(vecVal("MSOFT", 33)), 2.0);
            mL2sq = pow(high_prec_float(vecVal("MSOFT", 32)), 2.0);
            mL1sq = pow(high_prec_float(vecVal("MSOFT", 31)), 2.0);
            mU3sq = pow(high_prec_float(vecVal("MSOFT", 46)), 2.0);
            mU2sq = pow(high_prec_float(vecVal("MSOFT", 45)), 2.0);
            mU1sq = pow(high_prec_float(vecVal("MSOFT", 44)), 2.0);
            mD3sq = pow(high_prec_float(vecVal("MSOFT", 49)), 2.0);
            mD2sq = pow(high_prec_float(vecVal("MSOFT", 48)), 2.0);
            mD1sq = pow(high_prec_float(vecVal("MSOFT", 47)), 2.0);
            mE3sq = pow(high_prec_float(vecVal("MSOFT", 36)), 2.0);
            mE2sq = pow(high_prec_float(vecVal("MSOFT", 35)), 2.0);
            mE1sq = pow(high_prec_float(vecVal("MSOFT", 34)), 2.0);
        } else {
            mQ3sq = high_prec_float(matVal("MSQ2", 3, 3));
            mQ2sq = high_prec_float(matVal("MSQ2", 2, 2));
            mQ1sq = high_prec_float(matVal("MSQ2", 1, 1));
            mL3sq = high_prec_float(matVal("MSL2", 3, 3));
            mL2sq = high_prec_float(matVal("MSL2", 2, 2));
            mL1sq = high_prec_float(matVal("MSL2", 1, 1));
            mU3sq = high_prec_float(matVal("MSU2", 3, 3));
            mU2sq = high_prec_float(matVal("MSU2", 2, 2));
            mU1sq = high_prec_float(matVal("MSU2", 1, 1));
            mD3sq = high_prec_float(matVal("MSD2", 3, 3));
            mD2sq = high_prec_float(matVal("MSD2", 2, 2));
            mD1sq = high_prec_float(matVal("MSD2", 1, 1));
            mE3sq = high_prec_float(matVal("MSE2", 3, 3));
            mE2sq = high_prec_float(matVal("MSE2", 2, 2));
            mE1sq = high_prec_float(matVal("MSE2", 1, 1));
        }

        const double slhaScale = getRenormalizationScale(input, "GAUGE");

        // ---- Assemble the 44-entry state ----------------------------------------------
        // Slot order is load-bearing and shared with every calculator: g1 is GUT-normalised
        // here by the sqrt(5/3), because the GAUGE block carries the unnormalised g'.
        std::vector<high_prec_float> slhaBCs = {
            sqrt(5.0 / 3.0) * g_pr, g_2, g_s, my_M1, my_M2, my_M3,
            muQ, y_t, y_c, y_u, y_b, y_s, y_d, y_tau, y_mu, y_e,
            a_t, a_c, a_u, a_b, a_s, a_d, a_tau, a_mu, a_e,
            mHusq, mHdsq, mQ1sq, mQ2sq, mQ3sq, mL1sq, mL2sq,
            mL3sq, mU1sq, mU2sq, mU3sq, mD1sq, mD2sq, mD3sq,
            mE1sq, mE2sq, mE3sq, 0.0, tanb};
        std::vector<double> slhaBCsDbl;
        slhaBCsDbl.reserve(slhaBCs.size());
        for (const auto & v : slhaBCs) slhaBCsDbl.push_back(double(v));

        // ---- Locate Q_SUSY ------------------------------------------------------------
        // Run well above the SUSY scale first, then search downward for the unique root of
        // log(Q) - (log(m_stop1^2) + log(m_stop2^2))/4. Both running stop mass-squares must
        // remain finite and strictly positive at an accepted root.
        std::vector<double> upRun = solveODEs(slhaBCsDbl, log(slhaScale), log(1.0e12),
                                              std::copysign(1.0e-6, log(1.0e12 / slhaScale)));
        const auto auditedFindQSusy = [&](const std::vector<double>& state,
                                          double highLogScale,
                                          double timeStep,
                                          double maxDeltaLogQ) {
            return detail::runAuditedQSusySearch(
                out.qSusySearchDiagnostics,
                [&] {
                    return findQSusy(
                        state, highLogScale, timeStep, maxDeltaLogQ);
                });
        };
        QSusyResult susyRoot = auditedFindQSusy(
            upRun, log(kQSusySearchHigh), -1.0e-6, cfg.qSusyMaxDeltaLogQ);
        detail::JointQSusyOperations jointOperations;
        jointOperations.immutableHighState = upRun;
        jointOperations.evolve = solveODEs;
        jointOperations.findRoot = auditedFindQSusy;
        jointOperations.tuneMu = tuneEWSBMu;
        jointOperations.evaluateStop = evaluateStopScalePoint;
        detail::JointQSusySolution joint;
        try {
            joint = detail::solveJointQSusyMu(
                std::move(susyRoot), log(kQSusySearchHigh), 1.0e-6,
                cfg.qSusyMaxDeltaLogQ,
                kMaxQSusyIterations,
                jointOperations);
        } catch (const detail::JointQSusyConvergenceFailure& failure) {
            out.ewsbIters = failure.ewsbIterations;
            out.qSusyIters = static_cast<long>(failure.diagnostics.size());
            out.qSusyDiagnostics = failure.diagnostics;
            throw;
        }
        out.ewsbIters = joint.ewsbIterations;
        out.qSusyIters = static_cast<long>(joint.diagnostics.size());
        out.qSusyDiagnostics = joint.diagnostics;

        const high_prec_float qSusy = high_prec_float(joint.root.scale);
        const double qSusyDbl = joint.root.scale;
        out.qSusy = qSusy;
        out.qSusyResidual = joint.retunedPoint.logResidual;
        out.qSusyStop1Squared = joint.retunedPoint.stop1Squared;
        out.qSusyStop2Squared = joint.retunedPoint.stop2Squared;
        std::vector<high_prec_float> weakBCs = joint.tuned.state;
        std::vector<double> weakDbl = joint.tuned.doubleState;
        std::vector<high_prec_float> radCorrs = joint.tuned.radCorrs;
        const high_prec_float currentMZ2 = joint.tuned.relationMZ2;
        tanb = weakBCs[43];
        // Solve for m_Z^2 only when something downstream reads it. delta_SN is the one measure
        // that consumes it, and `wantMZ2FromSolver` covers callers that read the field without
        // asking for a measure. With all four measures requested this condition is true, so
        // the skip matters to a Delta_EW-only pass rather than to a full production run.
        //
        // Wall time printed when NATLHA_ODE_TRACE is set, to stderr, so stdout is unchanged.
        static const bool apiTrace = [] {
            const char * e = std::getenv("NATLHA_ODE_TRACE");
            return e != nullptr && *e != '\0';
        }();
        high_prec_float solverMZ2Value = 0;
        if (cfg.computeDSN || cfg.wantMZ2FromSolver) {
            const auto tMZ0 = std::chrono::steady_clock::now();
            const MZ2SolveResult mz2Solve = solveMZ2(weakBCs, qSusy, kMZ * kMZ);
            out.haveMZ2FromSolver = true;
            out.mZ2SolverConverged = mz2Solve.ok;
            if (!mz2Solve.ok) {
                out.error = "mZ2 solve failed: " + describeMZ2Failure(mz2Solve);
                return out;
            }
            solverMZ2Value = mz2Solve.value;
            if (apiTrace) {
                std::cerr << "# api_trace solveMZ2_seconds "
                          << std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - tMZ0).count()
                          << "  converged 1\n";
            }
        }

        out.mZ2 = currentMZ2;
        out.mZ2FromSolver = solverMZ2Value;
        out.weakBCs = weakBCs;
        out.radCorrs = radCorrs;

        // ---- Iterate to the g1 = g2 scale ---------------------------------------------
        detail::GutScaleOperations gutOperations;
        gutOperations.evolve = solveODEs;
        gutOperations.gaugeBetas = [](const std::vector<high_prec_float>& state) {
            return beta_g1g2(
                state[0], state[1], state[2], state[7], state[8], state[9],
                state[10], state[11], state[12], state[13], state[14], state[15]);
        };
        detail::GutScaleSolution gut = detail::solveGutScale(
            weakDbl, log(qSusyDbl), log(3.0e16), 1.0e-6,
            kMaxGutIterations, gutOperations);
        out.gutIters = gut.iterations;
        out.logQGut = gut.logScale;
        std::vector<high_prec_float> gutBCs = std::move(gut.state);
        out.gutBCs = gutBCs;
        const high_prec_float currIterQGut = gut.logScale;

        // ---- The measures -------------------------------------------------------------
        // DEW, DHS, and DBG expose their signed headlines as element [0]. Lowercase
        // delta_SN instead sums its complete contribution list below.
        const high_prec_float logQSusy = log(qSusy);

        if (cfg.computeDEW) {
            out.dewContributions = DEW_calc(weakBCs, qSusy);
            if (out.dewContributions.empty()) {
                detail::failLabelRow(out, "DEW_calc returned no contributions");
                return out;
            }
            for (const auto& contribution : out.dewContributions) {
                if (!(boost::math::isfinite)(contribution.value)) {
                    detail::failLabelRow(
                        out, "DEW_calc returned a non-finite contribution: "
                            + contribution.label);
                    return out;
                }
            }
            out.deltaEW = out.dewContributions[0].value;
            out.haveDEW = true;
        }
        if (cfg.computeDHS) {
            out.dhsContributions = DHS_calc(
                gutBCs[26], weakBCs[26] - gutBCs[26],
                gutBCs[25], weakBCs[25] - gutBCs[25],
                pow(gutBCs[6], 2.0), pow(weakBCs[6], 2.0) - pow(gutBCs[6], 2.0),
                kMZ * kMZ, weakBCs[43] * weakBCs[43], radCorrs[0], radCorrs[1]);
            if (out.dhsContributions.empty()) {
                detail::failLabelRow(out, "DHS_calc returned no contributions");
                return out;
            }
            for (const auto& contribution : out.dhsContributions) {
                if (!(boost::math::isfinite)(contribution.value)) {
                    detail::failLabelRow(
                        out, "DHS_calc returned a non-finite contribution: "
                            + contribution.label);
                    return out;
                }
            }
            out.deltaHS = out.dhsContributions[0].value;
            out.haveDHS = true;
        }
        if (cfg.computeDBG) {
            int modsel = cfg.bgModelIndex;
            int precsel = cfg.bgPrecision;
            const BGResult bg = DBG_calc(modsel, precsel, currIterQGut, logQSusy,
                                         tanb, gutBCs, currentMZ2);
            out.dbgDiagnostics = bg.directions;
            out.dbgHeadline = bg.headline;
            if (!bg.ok) {
                detail::failLabelRow(
                    out, bg.failure.empty()
                        ? "DBG_calc failed without a diagnostic" : bg.failure);
                return out;
            }
            out.dbgContributions = bg.contributions;
            if (out.dbgContributions.empty()) {
                detail::failLabelRow(out, "DBG_calc returned no contributions");
                return out;
            }
            detail::requireFiniteDBGContributions(out.dbgContributions);
            out.deltaBG = out.dbgContributions[0].value;
            out.haveDBG = true;
        }
        if (cfg.computeDSN) {
            high_prec_float mz2ForSn = solverMZ2Value;
            high_prec_float logQSusyForSn = logQSusy;
            high_prec_float logQGutForSn = currIterQGut;
            int nF = cfg.snNF;
            int nD = cfg.snND;
            out.dsnContributions = DSN_calc(cfg.snMode, weakBCs, mz2ForSn, logQSusyForSn,
                                            logQGutForSn, nF, nD);
            out.snTotalNvac = 0.0;
            for (const auto & item : out.dsnContributions) out.snTotalNvac += item.value;
            if (out.dsnContributions.empty() || out.snTotalNvac <= 0.0
                    || isnan(out.snTotalNvac) || isinf(out.snTotalNvac)) {
                detail::failLabelRow(out, "DSN_calc returned an invalid vacuum density");
                return out;
            }
            out.deltaSN = log10(high_prec_float(1.0) / out.snTotalNvac);
            if (isnan(out.deltaSN) || isinf(out.deltaSN)) {
                detail::failLabelRow(
                    out, "DSN_calc returned a non-finite naturalness measure");
                return out;
            }
            out.haveDSN = true;
        }

        out.ok = true;
        return out;
    } catch (const std::exception & e) {
        detail::failLabelRow(out, std::string("exception: ") + e.what());
        return out;
    } catch (...) {
        detail::failLabelRow(out, "unknown exception in natlha::evaluate");
        return out;
    }
}

}  // namespace natlha
