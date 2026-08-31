#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "MSSM_RGE_solver.hpp"
#include "MSSM_QSUSY_helpers.inl"
#include "natlha_api.hpp"
#include "natlha_cuda_backend.hpp"

namespace {

bool expect(bool condition, const std::string& message) {
    if (condition) return true;
    std::cerr << message << "\n";
    return false;
}

bool closeEnough(double gpu, double cpu) {
    const double scale = std::max({1.0, std::abs(gpu), std::abs(cpu)});
    return std::isfinite(gpu) && std::abs(gpu - cpu) <= 2.0e-8 * scale;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: test-cuda-ode FIXTURE.slha\n";
        return 2;
    }

    bool ok = true;
    const natlha::CudaDeviceInfo device = natlha::queryCudaDevice(0);
    ok &= expect(device.compiled && device.available && !device.name.empty()
                     && device.computeCapabilityMajor > 0
                     && device.multiprocessorCount > 0
                     && device.totalMemoryBytes > 0,
                 "CUDA device discovery did not return an executable device");
    if (!ok) return 1;

    natlha::PointExecutionDiagnostic fp64Result;
    natlha::detail::recordCudaPointResult(fp64Result, false);
    ok &= expect(fp64Result.executed
                     && fp64Result.candidateTier == natlha::ExecutionTier::CudaFp64
                     && fp64Result.finalTier == natlha::ExecutionTier::CudaFp64,
                 "FP64 result did not establish execution provenance");
    natlha::detail::recordCudaPointEscape(fp64Result, true);
    ok &= expect(fp64Result.executed
                     && fp64Result.candidateTier == natlha::ExecutionTier::CudaFp64
                     && fp64Result.finalTier == natlha::ExecutionTier::None,
                 "double-double escape erased the prior FP64 candidate");

    natlha::PointExecutionDiagnostic fp64ThenDoubleDouble;
    natlha::detail::recordCudaPointResult(fp64ThenDoubleDouble, false);
    natlha::detail::recordCudaPointResult(fp64ThenDoubleDouble, true);
    ok &= expect(fp64ThenDoubleDouble.executed
                     && fp64ThenDoubleDouble.candidateTier
                            == natlha::ExecutionTier::CudaFp64
                     && fp64ThenDoubleDouble.finalTier
                            == natlha::ExecutionTier::CudaFp64,
                 "double-double retry replaced first-candidate provenance before acceptance");

    natlha::PointExecutionDiagnostic fp64Escape;
    natlha::detail::recordCudaPointEscape(fp64Escape, false);
    ok &= expect(!fp64Escape.executed
                     && fp64Escape.candidateTier == natlha::ExecutionTier::None
                     && fp64Escape.finalTier == natlha::ExecutionTier::None,
                 "FP64 escape claimed a CUDA candidate result");
    natlha::detail::recordCudaPointResult(fp64Escape, true);
    ok &= expect(fp64Escape.executed
                     && fp64Escape.candidateTier
                            == natlha::ExecutionTier::CudaDoubleDouble
                     && fp64Escape.finalTier == natlha::ExecutionTier::None,
                 "double-double result did not recover after an FP64 escape");

    const auto reason = [](natlha::AdjudicationReason value) {
        return natlha::adjudicationReason(value);
    };
    natlha::Result failedBoundary;
    const natlha::AdjudicationReasons failedReasons =
        natlha::detail::cudaBranchBoundaryReasons(failedBoundary);
    ok &= expect(
        failedReasons
            == (reason(natlha::AdjudicationReason::BranchBoundary)
                | reason(natlha::AdjudicationReason::FailedCandidateBoundary)),
        "failed CUDA candidate lost its structured branch-boundary reason");

    natlha::Result headlineBoundary;
    headlineBoundary.ok = true;
    headlineBoundary.haveDBG = true;
    headlineBoundary.dbgHeadline.headlineSignFragileRootUncertainty = true;
    headlineBoundary.dbgHeadline.headlineMagnitudeGap = 10;
    ok &= expect(
        natlha::detail::cudaBranchBoundaryReasons(headlineBoundary)
            == (reason(natlha::AdjudicationReason::BranchBoundary)
                | reason(natlha::AdjudicationReason::HeadlineBoundary)),
        "headline fragility was not distinguished from other branch boundaries");

    natlha::Result orderingBoundary;
    orderingBoundary.ok = true;
    orderingBoundary.haveDBG = true;
    orderingBoundary.dbgHeadline.headlineMagnitudeGap = 10;
    orderingBoundary.dbgContributions = {
        {high_prec_float("2.0"), "first", 0, 0},
        {high_prec_float("1.9999"), "second", 1, 0}};
    ok &= expect(
        natlha::detail::cudaBranchBoundaryReasons(orderingBoundary)
            == (reason(natlha::AdjudicationReason::BranchBoundary)
                | reason(natlha::AdjudicationReason::ContributionOrderBoundary)
                | reason(natlha::AdjudicationReason::HeadlineOrderBoundary)),
        "top-two ordering ambiguity was not distinguished from other boundaries");

    natlha::Result lowerOrderingBoundary;
    lowerOrderingBoundary.ok = true;
    lowerOrderingBoundary.haveDBG = true;
    lowerOrderingBoundary.dbgHeadline.headlineMagnitudeGap = 10;
    lowerOrderingBoundary.dbgContributions = {
        {high_prec_float("10.0"), "first", 0, 0},
        {high_prec_float("2.0"), "second", 1, 0},
        {high_prec_float("1.9999"), "third", 2, 0}};
    ok &= expect(
        natlha::detail::cudaBranchBoundaryReasons(lowerOrderingBoundary)
            == (reason(natlha::AdjudicationReason::BranchBoundary)
                | reason(natlha::AdjudicationReason::ContributionOrderBoundary)
                | reason(
                    natlha::AdjudicationReason::LowerContributionOrderBoundary)),
        "lower-ranked ordering ambiguity was not isolated from top-two ambiguity");
    const natlha::AdjudicationReasons lowerOnlyReasons =
        natlha::detail::cudaBranchBoundaryReasons(lowerOrderingBoundary);
    ok &= expect(
        natlha::detail::cudaBatchRowAcceptsBranchReasons(lowerOnlyReasons)
            && natlha::detail::cudaBatchRowAcceptsBranchReasons(0)
            && !natlha::detail::cudaBatchRowAcceptsBranchReasons(
                natlha::detail::cudaBranchBoundaryReasons(orderingBoundary))
            && !natlha::detail::cudaBatchRowAcceptsBranchReasons(
                lowerOnlyReasons
                | reason(natlha::AdjudicationReason::RootBoundary)),
        "batch-row branch gate did not isolate the exact lower-order boundary");
    const natlha::AdjudicationReasons lowerContributionTierReasons =
        lowerOnlyReasons
        | reason(natlha::AdjudicationReason::TierDisagreement)
        | reason(natlha::AdjudicationReason::ContributionTierDisagreement);
    ok &= expect(
        natlha::detail::cudaBatchRowAcceptsRelaxedDiagnosticReasons(
            lowerOnlyReasons)
            && natlha::detail::cudaBatchRowAcceptsRelaxedDiagnosticReasons(
                lowerContributionTierReasons)
            && !natlha::detail::cudaBatchRowAcceptsRelaxedDiagnosticReasons(
                lowerContributionTierReasons
                | reason(natlha::AdjudicationReason::EmittedFieldTierDisagreement)),
        "batch-row relaxed diagnostic gate admitted a non-lower-order reason set");

    natlha::Result adaptiveBoundary;
    adaptiveBoundary.ok = true;
    adaptiveBoundary.dbgDiagnostics.emplace_back();
    ok &= expect(
        natlha::detail::cudaBranchBoundaryReasons(adaptiveBoundary)
            == (reason(natlha::AdjudicationReason::BranchBoundary)
                | reason(natlha::AdjudicationReason::AdaptiveWindowBoundary)),
        "adaptive-window rejection was not distinguished from other branch boundaries");

    natlha::Result comparisonReference;
    comparisonReference.ok = true;
    comparisonReference.snTotalNvac = high_prec_float("1e-6");
    natlha::QSusySearchDiagnostic acceptedSearch;
    acceptedSearch.scanComplete = true;
    acceptedSearch.accepted = true;
    acceptedSearch.logScale = 8.0;
    acceptedSearch.rootsFound = 1;
    comparisonReference.qSusySearchDiagnostics.push_back(acceptedSearch);
    natlha::Result changedNvac = comparisonReference;
    changedNvac.snTotalNvac = high_prec_float("2e-6");
    ok &= expect(
        natlha::detail::cudaResultMismatch(changedNvac, comparisonReference)
            == "dN_vac differs",
        "CUDA result comparison omitted the emitted dN_vac field");
    natlha::Result changedAudit = comparisonReference;
    changedAudit.qSusySearchDiagnostics.back().rootsFound = 2;
    ok &= expect(
        natlha::detail::cudaResultMismatch(changedAudit, comparisonReference)
            == "Q_SUSY audit summary differs",
        "CUDA result comparison omitted the emitted Q_SUSY audit fields");

    natlha::Result contributionReference = comparisonReference;
    contributionReference.dbgContributions = {
        {high_prec_float("10.0"), "first", 0, 0},
        {high_prec_float("2.0"), "second", 1, 0}};
    natlha::Result changedContribution = contributionReference;
    changedContribution.dbgContributions[1].label = "changed";
    ok &= expect(
        natlha::detail::cudaResultMismatch(
            changedContribution, contributionReference)
            == "Delta_BG label/ordinal differs at index 1 (candidate 'changed' ordinal 1, CPU 'second' ordinal 1)",
        "CUDA result comparison lost full contribution-order coverage");

    contributionReference.haveDBG = true;
    contributionReference.deltaBG = high_prec_float("10.0");
    contributionReference.dbgHeadline.topLabel = "first";
    contributionReference.dbgHeadline.tiedDirectionOrdinals = {0};
    natlha::Result changedLowerContribution = contributionReference;
    changedLowerContribution.dbgContributions[1].label = "changed-lower";
    ok &= expect(
        natlha::detail::cudaBatchRowMismatch(
            changedLowerContribution, contributionReference).empty()
            && !natlha::detail::cudaResultMismatch(
                changedLowerContribution, contributionReference).empty(),
        "batch-row comparison did not exclude only lower contribution detail");
    natlha::Result changedHeadlineOrdinal = contributionReference;
    changedHeadlineOrdinal.dbgContributions[0].ordinal = 9;
    ok &= expect(
        natlha::detail::cudaBatchRowMismatch(
            changedHeadlineOrdinal, contributionReference)
            == "Delta_BG headline label/ordinal differs",
        "batch-row comparison omitted the selected Delta_BG ordinal");
    natlha::Result changedTieSet = contributionReference;
    changedTieSet.dbgHeadline.tiedDirectionOrdinals = {0, 1};
    ok &= expect(
        natlha::detail::cudaBatchRowMismatch(changedTieSet, contributionReference)
            == "Delta_BG exact-tie ordinal set differs",
        "batch-row comparison omitted the Delta_BG exact-tie set");
    if (!ok) return 1;

    natlha::Config config;
    config.slhaPath = argv[1];
    config.computeDEW = false;
    const natlha::Result fixture = natlha::evaluate(config);
    ok &= expect(fixture.ok && fixture.weakBCs.size() == natlha::detail::kRgeStateSize,
                 "CPU fixture setup returned an unexpected weak-scale state size");
    if (!ok) return 1;

    std::vector<double> initial;
    initial.reserve(natlha::detail::kRgeStateSize);
    for (const auto& value : fixture.weakBCs) initial.push_back(static_cast<double>(value));

    std::vector<natlha::detail::CudaQSusyHelperInput> helperInputs(4);
    for (auto& input : helperInputs) {
        input.state = initial;
        input.logScale = std::log(static_cast<double>(fixture.qSusy));
        input.currentLogScale = 27.63;
        input.lowerLogScale = 27.20;
        input.maxDeltaLogQ = 0.1;
        input.highPoint.numericallyValid = true;
        input.highPoint.physical = true;
        input.highPoint.logResidual = 1.0;
        input.lowPoint.numericallyValid = true;
        input.lowPoint.physical = true;
        input.lowPoint.logResidual = -1.0;
    }
    helperInputs[1].highPoint.logResidual = 0.0;
    helperInputs[1].lowPoint.logResidual = 1.0;
    helperInputs[2].highPoint.numericallyValid = false;
    helperInputs[2].highPoint.physical = false;
    helperInputs[2].lowPoint.logResidual = 1.0;
    helperInputs[3].maxDeltaLogQ = 0.0;
    helperInputs[3].state[0] = std::numeric_limits<double>::quiet_NaN();

    const std::vector<natlha::detail::CudaQSusyHelperResult> helperGpu =
        natlha::detail::evaluateCudaQSusyHelpers(helperInputs, 0);
    ok &= expect(helperGpu.size() == helperInputs.size(),
                 "CUDA Q_SUSY helper results lost input alignment");
    if (!ok) return 1;
    for (std::size_t point = 0; point < helperInputs.size(); ++point) {
        const auto& input = helperInputs[point];
        const natlha::qsusy_numeric::StopPoint expectedStop =
            natlha::qsusy_numeric::evaluateStopPoint(
                input.state.data(), input.state.size(), input.logScale, 91.1876);
        double expectedNext = 0.0;
        const auto expectedStep = natlha::qsusy_numeric::nextScanLogScale(
            input.currentLogScale, input.lowerLogScale, input.maxDeltaLogQ,
            expectedNext);
        const auto toCore = [](const StopScalePoint& source) {
            natlha::qsusy_numeric::StopPoint target;
            target.numericallyValid = source.numericallyValid;
            target.physical = source.physical;
            target.stop1Squared = source.stop1Squared;
            target.stop2Squared = source.stop2Squared;
            target.logResidual = source.logResidual;
            return target;
        };
        natlha::qsusy_numeric::ScanState expectedScan;
        expectedScan.inInvalidDomain = input.inInvalidDomain;
        expectedScan.inNonFiniteDomain = input.inNonFiniteDomain;
        expectedScan.invalidBoundaries = input.invalidBoundaries;
        expectedScan.nonFiniteBoundaries = input.nonFiniteBoundaries;
        unsigned expectedEvents = 0;
        const bool expectedClassification = natlha::qsusy_numeric::classifySegment(
            toCore(input.highPoint), toCore(input.lowPoint), expectedScan,
            expectedEvents);
        const auto& candidate = helperGpu[point];
        ok &= expect(candidate.evaluatedPoint.numericallyValid
                         == expectedStop.numericallyValid
                         && candidate.evaluatedPoint.physical == expectedStop.physical
                         && closeEnough(candidate.evaluatedPoint.stop1Squared,
                                        expectedStop.stop1Squared)
                         && closeEnough(candidate.evaluatedPoint.stop2Squared,
                                        expectedStop.stop2Squared)
                         && closeEnough(candidate.evaluatedPoint.logResidual,
                                        expectedStop.logResidual),
                     "CUDA/CPU shared Q_SUSY stop helper mismatch at point "
                         + std::to_string(point));
        ok &= expect(candidate.scanStepStatus == static_cast<int>(expectedStep)
                         && (expectedStep
                                 != natlha::qsusy_numeric::ScanStepStatus::Success
                             || candidate.nextLogScale == expectedNext),
                     "CUDA/CPU shared Q_SUSY scan-step mismatch at point "
                         + std::to_string(point));
        ok &= expect(candidate.classificationOk == expectedClassification
                         && candidate.scanEvents == expectedEvents
                         && candidate.inInvalidDomain == expectedScan.inInvalidDomain
                         && candidate.inNonFiniteDomain == expectedScan.inNonFiniteDomain
                         && candidate.invalidBoundaries == expectedScan.invalidBoundaries
                         && candidate.nonFiniteBoundaries
                                == expectedScan.nonFiniteBoundaries,
                     "CUDA/CPU shared Q_SUSY classification mismatch at point "
                         + std::to_string(point));
    }

    const double start = std::log(static_cast<double>(fixture.qSusy));
    const double qSusyHigh = std::log(1.0e12);
    constexpr std::size_t kDistinctTrajectories = 17;
    std::vector<std::vector<double>> distinctInitialStates;
    std::vector<std::vector<double>> qSusyHighStates;
    std::vector<QSusyResult> qSusyCpuReferences;
    distinctInitialStates.reserve(kDistinctTrajectories);
    qSusyHighStates.reserve(kDistinctTrajectories);
    qSusyCpuReferences.reserve(kDistinctTrajectories);
    bool sawCpuRefinedBracket = false;
    for (std::size_t variant = 0; variant < kDistinctTrajectories; ++variant) {
        std::vector<double> state = initial;
        state[3] *= 1.0 + 2.0e-4 * static_cast<double>(variant);
        distinctInitialStates.push_back(state);
        qSusyHighStates.push_back(
            solveODEs(state, start, qSusyHigh, 1.0e-6));
        qSusyCpuReferences.push_back(
            findQSusyCpu(qSusyHighStates.back(), qSusyHigh, -1.0e-6, 0.1));
        const double cpuBracketWidth = qSusyCpuReferences.back().refinedBracketWidth;
        ok &= expect(std::isfinite(cpuBracketWidth) && cpuBracketWidth >= 0.0,
                     "CPU Q_SUSY reference returned an invalid refined bracket width at "
                         "variant " + std::to_string(variant));
        sawCpuRefinedBracket = sawCpuRefinedBracket || cpuBracketWidth > 0.0;
    }
    ok &= expect(sawCpuRefinedBracket,
                 "CPU Q_SUSY references never reported a refined sign bracket");
    constexpr std::size_t kQSusyPoints = 33;
    natlha::detail::CudaQSusyBatch qSusyBatch;
    qSusyBatch.highLogScales.assign(kQSusyPoints, qSusyHigh);
    qSusyBatch.initialSteps.assign(kQSusyPoints, -1.0e-6);
    qSusyBatch.maxDeltaLogQ.assign(kQSusyPoints, 0.1);
    for (std::size_t point = 0; point < kQSusyPoints; ++point) {
        const std::size_t variant = point % kDistinctTrajectories;
        qSusyBatch.states.insert(
            qSusyBatch.states.end(), qSusyHighStates[variant].begin(),
            qSusyHighStates[variant].end());
    }
    const natlha::detail::CudaQSusyBatchResult qSusyGpu =
        natlha::detail::solveCudaQSusyBatchFp64(qSusyBatch, 0, 7);
    ok &= expect(qSusyGpu.statuses.size() == kQSusyPoints
                     && qSusyGpu.logScales.size() == kQSusyPoints
                     && qSusyGpu.residuals.size() == kQSusyPoints
                     && qSusyGpu.stop1Squared.size() == kQSusyPoints
                     && qSusyGpu.stop2Squared.size() == kQSusyPoints
                     && qSusyGpu.refinedBracketWidths.size() == kQSusyPoints
                     && qSusyGpu.scanSegments.size() == kQSusyPoints
                     && qSusyGpu.maxObservedDeltaLogQ.size() == kQSusyPoints
                     && qSusyGpu.rootsFound.size() == kQSusyPoints
                     && qSusyGpu.nonFiniteBoundaries.size() == kQSusyPoints
                     && qSusyGpu.statesAtRoot.size()
                            == kQSusyPoints * natlha::detail::kRgeStateSize,
                 "CUDA Q_SUSY result arrays lost batch alignment");
    if (!ok) return 1;
    bool sawRefinedBracket = false;
    for (std::size_t point = 0; point < kQSusyPoints; ++point) {
        const std::size_t variant = point % kDistinctTrajectories;
        const QSusyResult& qSusyCpu = qSusyCpuReferences[variant];
        ok &= expect(qSusyGpu.statuses[point]
                         == natlha::detail::CudaQSusyStatus::Success
                         && qSusyGpu.rootsFound[point] == 1
                         && qSusyGpu.nonFiniteBoundaries[point] == 0
                         && qSusyGpu.scanSegments[point] > 0
                         && qSusyGpu.maxObservedDeltaLogQ[point] > 0.0
                         && qSusyGpu.maxObservedDeltaLogQ[point] <= 0.1
                         && std::isfinite(qSusyGpu.refinedBracketWidths[point])
                         && qSusyGpu.refinedBracketWidths[point] >= 0.0
                         && std::abs(qSusyGpu.residuals[point]) <= 1.0e-12,
                     "CUDA Q_SUSY candidate did not satisfy its semantic gates at point "
                         + std::to_string(point) + ", status "
                         + std::to_string(static_cast<int>(qSusyGpu.statuses[point])));
        ok &= expect(closeEnough(qSusyGpu.logScales[point], qSusyCpu.logScale)
                         && closeEnough(qSusyGpu.stop1Squared[point],
                                        qSusyCpu.stop1Squared)
                         && closeEnough(qSusyGpu.stop2Squared[point],
                                        qSusyCpu.stop2Squared),
                     "CUDA/CPU Q_SUSY root metadata mismatch at point "
                         + std::to_string(point));
        sawRefinedBracket = sawRefinedBracket
            || qSusyGpu.refinedBracketWidths[point] > 0.0;
        for (std::size_t component = 0;
             component < natlha::detail::kRgeStateSize; ++component) {
            ok &= expect(closeEnough(
                             qSusyGpu.statesAtRoot[
                                 point * natlha::detail::kRgeStateSize + component],
                             qSusyCpu.stateAtRoot[component]),
                         "CUDA/CPU Q_SUSY root-state mismatch at point "
                             + std::to_string(point) + ", component "
                             + std::to_string(component));
        }
    }
    ok &= expect(!closeEnough(
                     qSusyGpu.statesAtRoot[3],
                     qSusyGpu.statesAtRoot[natlha::detail::kRgeStateSize + 3]),
                 "heterogeneous CUDA Q_SUSY rows lost distinct outputs");
    ok &= expect(sawRefinedBracket,
                 "CUDA Q_SUSY population never reported a refined root bracket");
    natlha::detail::CudaQSusyBatch qSusyDoubleDoubleBatch;
    constexpr std::size_t kQSusyDoubleDoublePoints = 3;
    qSusyDoubleDoubleBatch.highLogScales.assign(
        kQSusyDoubleDoublePoints, qSusyHigh);
    qSusyDoubleDoubleBatch.initialSteps.assign(
        kQSusyDoubleDoublePoints, -1.0e-6);
    qSusyDoubleDoubleBatch.maxDeltaLogQ.assign(kQSusyDoubleDoublePoints, 0.1);
    for (std::size_t point = 0; point < kQSusyDoubleDoublePoints; ++point) {
        qSusyDoubleDoubleBatch.states.insert(
            qSusyDoubleDoubleBatch.states.end(),
            qSusyHighStates[point].begin(), qSusyHighStates[point].end());
    }
    const natlha::detail::CudaQSusyBatchResult qSusyDoubleDouble =
        natlha::detail::solveCudaQSusyBatchDoubleDouble(
            qSusyDoubleDoubleBatch, 0, 2);
    ok &= expect(qSusyDoubleDouble.statuses.size() == kQSusyDoubleDoublePoints
                     && qSusyDoubleDouble.rootsFound.size()
                            == kQSusyDoubleDoublePoints
                     && qSusyDoubleDouble.logScales.size()
                            == kQSusyDoubleDoublePoints
                     && qSusyDoubleDouble.refinedBracketWidths.size()
                            == kQSusyDoubleDoublePoints,
                 "CUDA double-double Q_SUSY arrays lost batch alignment");
    if (!ok) return 1;
    for (std::size_t point = 0; point < kQSusyDoubleDoublePoints; ++point) {
        ok &= expect(qSusyDoubleDouble.statuses[point]
                         == natlha::detail::CudaQSusyStatus::Success
                         && qSusyDoubleDouble.rootsFound[point] == 1
                         && std::isfinite(
                             qSusyDoubleDouble.refinedBracketWidths[point])
                         && closeEnough(qSusyDoubleDouble.logScales[point],
                                        qSusyCpuReferences[point].logScale),
                     "CUDA double-double Q_SUSY candidate disagreed with CPU at point "
                         + std::to_string(point));
    }

    natlha::detail::CudaQSusyBatch invalidQSusy;
    invalidQSusy.states = qSusyHighStates.front();
    invalidQSusy.highLogScales = {
        std::numeric_limits<double>::quiet_NaN()};
    invalidQSusy.initialSteps = {-1.0e-6};
    invalidQSusy.maxDeltaLogQ = {0.1};
    const natlha::detail::CudaQSusyBatchResult rejectedQSusy =
        natlha::detail::solveCudaQSusyBatchFp64(invalidQSusy, 0, 1);
    ok &= expect(rejectedQSusy.statuses.size() == 1
                     && rejectedQSusy.statuses[0]
                            == natlha::detail::CudaQSusyStatus::NonFiniteInput,
                 "CUDA Q_SUSY accepted a non-finite high scale");
    const natlha::AdjudicationReasons nonFiniteRootReasons =
        natlha::detail::cudaQSusyAdjudicationReasons(
            natlha::detail::CudaQSusyStatus::NonFiniteBoundary);
    for (const natlha::detail::CudaQSusyStatus status : {
             natlha::detail::CudaQSusyStatus::NonFiniteInput,
             natlha::detail::CudaQSusyStatus::NonFiniteState,
             natlha::detail::CudaQSusyStatus::StepLimit,
             natlha::detail::CudaQSusyStatus::StepUnderflow,
             natlha::detail::CudaQSusyStatus::ScanSpacing,
             natlha::detail::CudaQSusyStatus::BoundaryCounterOverflow,
             natlha::detail::CudaQSusyStatus::NonPhysicalBracket,
             natlha::detail::CudaQSusyStatus::RefinementFailure,
             natlha::detail::CudaQSusyStatus::ResidualFailure,
             natlha::detail::CudaQSusyStatus::NonUniqueRoot,
             natlha::detail::CudaQSusyStatus::NonFiniteBoundary}) {
        ok &= expect(
            natlha::hasAdjudicationReason(
                natlha::detail::cudaQSusyAdjudicationReasons(status),
                natlha::AdjudicationReason::RootBoundary),
            "CUDA Q_SUSY failure status lost its root-boundary reason");
    }
    ok &= expect(
        natlha::hasAdjudicationReason(
            nonFiniteRootReasons, natlha::AdjudicationReason::RootBoundary)
            && natlha::hasAdjudicationReason(
                nonFiniteRootReasons, natlha::AdjudicationReason::NonFiniteState)
            && natlha::hasAdjudicationReason(
                natlha::detail::cudaQSusyAdjudicationReasons(
                    natlha::detail::CudaQSusyStatus::StepLimit),
                natlha::AdjudicationReason::OdeStepLimit)
            && natlha::hasAdjudicationReason(
                natlha::detail::cudaQSusyAdjudicationReasons(
                    natlha::detail::CudaQSusyStatus::ResidualFailure),
                natlha::AdjudicationReason::ErrorEstimate)
            && natlha::hasAdjudicationReason(
                natlha::detail::cudaQSusyAdjudicationReasons(
                    natlha::detail::CudaQSusyStatus::StepUnderflow),
                natlha::AdjudicationReason::ErrorEstimate)
            && natlha::detail::cudaQSusyAdjudicationReasons(
                   natlha::detail::CudaQSusyStatus::Success) == 0,
        "CUDA Q_SUSY status-to-adjudication mapping lost a failure class");

    natlha::detail::CudaQSusyBatch malformedQSusy;
    malformedQSusy.states = qSusyHighStates.front();
    malformedQSusy.highLogScales = {qSusyHigh};
    malformedQSusy.initialSteps = {};
    malformedQSusy.maxDeltaLogQ = {0.1};
    bool malformedQSusyRejected = false;
    try {
        (void)natlha::detail::solveCudaQSusyBatchFp64(malformedQSusy, 0, 1);
    } catch (const std::invalid_argument&) {
        malformedQSusyRejected = true;
    }
    ok &= expect(malformedQSusyRejected,
                 "CUDA Q_SUSY accepted mismatched batch dimensions");

    const double end = static_cast<double>(fixture.logQGut);
    std::vector<std::vector<double>> cpuReferences;
    cpuReferences.reserve(kDistinctTrajectories);
    for (std::size_t variant = 0; variant < kDistinctTrajectories; ++variant) {
        cpuReferences.push_back(solveODEs(
            distinctInitialStates[variant], start,
            end + 1.0e-3 * static_cast<double>(variant), 1.0e-6));
    }

    constexpr std::size_t kPoints = 257;
    natlha::detail::CudaOdeBatch batch;
    batch.states.reserve(kPoints * natlha::detail::kRgeStateSize);
    batch.startTimes.assign(kPoints, start);
    batch.endTimes.reserve(kPoints);
    batch.initialSteps.assign(kPoints, 1.0e-6);
    for (std::size_t point = 0; point < kPoints; ++point) {
        const std::size_t variant = point % kDistinctTrajectories;
        batch.endTimes.push_back(
            end + 1.0e-3 * static_cast<double>(variant));
        batch.states.insert(
            batch.states.end(), distinctInitialStates[variant].begin(),
            distinctInitialStates[variant].end());
    }

    const natlha::detail::CudaOdeBatchResult gpu =
        natlha::detail::solveCudaOdeBatchFp64(batch, 0, 17);
    ok &= expect(gpu.states.size() == batch.states.size()
                     && gpu.statuses.size() == kPoints
                     && gpu.acceptedSteps.size() == kPoints
                     && gpu.rejectedSteps.size() == kPoints,
                 "CUDA ODE result arrays lost batch alignment");
    if (!ok) return 1;
    for (std::size_t point = 0; point < kPoints; ++point) {
        const std::size_t variant = point % kDistinctTrajectories;
        ok &= expect(gpu.statuses[point] == natlha::detail::CudaOdeStatus::Success
                         && gpu.acceptedSteps[point] > 0,
                     "CUDA ODE trajectory did not complete with accepted steps");
        for (std::size_t component = 0;
             component < natlha::detail::kRgeStateSize;
             ++component) {
            const double candidate =
                gpu.states[point * natlha::detail::kRgeStateSize + component];
            ok &= expect(closeEnough(candidate, cpuReferences[variant][component]),
                         "CUDA/CPU RGE endpoint mismatch at point "
                             + std::to_string(point) + ", component "
                             + std::to_string(component) + ": gpu="
                             + std::to_string(candidate) + ", cpu="
                             + std::to_string(cpuReferences[variant][component]));
        }
    }
    ok &= expect(!closeEnough(gpu.states[3], gpu.states[
                         natlha::detail::kRgeStateSize + 3]),
                 "heterogeneous CUDA ODE rows lost distinct outputs");

    natlha::detail::CudaOdeBatch doubleDoubleBatch;
    constexpr std::size_t kDoubleDoublePoints = 5;
    doubleDoubleBatch.startTimes.assign(kDoubleDoublePoints, start);
    doubleDoubleBatch.endTimes.reserve(kDoubleDoublePoints);
    doubleDoubleBatch.initialSteps.assign(kDoubleDoublePoints, 1.0e-6);
    for (std::size_t point = 0; point < kDoubleDoublePoints; ++point) {
        doubleDoubleBatch.endTimes.push_back(
            end + 1.0e-3 * static_cast<double>(point));
        doubleDoubleBatch.states.insert(
            doubleDoubleBatch.states.end(), distinctInitialStates[point].begin(),
            distinctInitialStates[point].end());
    }
    const natlha::detail::CudaOdeBatchResult doubleDouble =
        natlha::detail::solveCudaOdeBatchDoubleDouble(doubleDoubleBatch, 0, 3);
    ok &= expect(doubleDouble.statuses.size() == kDoubleDoublePoints
                     && doubleDouble.acceptedSteps.size() == kDoubleDoublePoints
                     && doubleDouble.states.size()
                            == kDoubleDoublePoints * natlha::detail::kRgeStateSize,
                 "CUDA double-double ODE arrays lost batch alignment");
    if (!ok) return 1;
    for (std::size_t point = 0; point < kDoubleDoublePoints; ++point) {
        ok &= expect(
            doubleDouble.statuses[point] == natlha::detail::CudaOdeStatus::Success
                && doubleDouble.acceptedSteps[point] > 0,
            "CUDA double-double ODE trajectory did not complete");
        for (std::size_t component = 0;
             component < natlha::detail::kRgeStateSize;
             ++component) {
            ok &= expect(closeEnough(
                             doubleDouble.states[
                                 point * natlha::detail::kRgeStateSize + component],
                             cpuReferences[point][component]),
                         "CUDA double-double/CPU endpoint mismatch at point "
                             + std::to_string(point) + ", component "
                             + std::to_string(component));
        }
    }

    natlha::detail::CudaOdeBatch invalid;
    invalid.states = initial;
    invalid.states[0] = std::numeric_limits<double>::quiet_NaN();
    invalid.startTimes = {start};
    invalid.endTimes = {end};
    invalid.initialSteps = {1.0e-6};
    const natlha::detail::CudaOdeBatchResult rejected =
        natlha::detail::solveCudaOdeBatchFp64(invalid, 0, 1);
    ok &= expect(rejected.statuses.size() == 1
                     && rejected.statuses[0]
                            == natlha::detail::CudaOdeStatus::NonFiniteInput,
                 "CUDA ODE accepted a non-finite input state");

    natlha::detail::CudaOdeBatch malformedOde;
    malformedOde.states = initial;
    malformedOde.startTimes = {start};
    malformedOde.endTimes = {};
    malformedOde.initialSteps = {1.0e-6};
    bool malformedOdeRejected = false;
    try {
        (void)natlha::detail::solveCudaOdeBatchFp64(malformedOde, 0, 1);
    } catch (const std::invalid_argument&) {
        malformedOdeRejected = true;
    }
    ok &= expect(malformedOdeRejected,
                 "CUDA ODE accepted mismatched batch dimensions");

    natlha::BatchOptions auditedOptions;
    auditedOptions.backend = natlha::Backend::Cuda;
    auditedOptions.cudaDevice = 0;
    auditedOptions.cudaWorkers = 3;
    auditedOptions.backendAudit = true;
    std::vector<natlha::Config> auditedConfigs(8, config);
    for (std::size_t point = 0; point < auditedConfigs.size(); ++point) {
        auditedConfigs[point].qSusyMaxDeltaLogQ =
            0.04 + 0.01 * static_cast<double>(point);
    }
    const natlha::BatchRun audited =
        natlha::evaluateBatch(auditedConfigs, auditedOptions);
    ok &= expect(audited.results.size() == auditedConfigs.size()
                     && audited.diagnostics.size() == auditedConfigs.size()
                     && audited.summary.points == auditedConfigs.size()
                     && audited.summary.succeeded == auditedConfigs.size()
                     && audited.summary.failed == 0
                     && audited.summary.cudaFp64LaunchLimit > 0
                     && audited.summary.cudaFp64LaunchLimit
                            <= auditedConfigs.size()
                     && audited.summary.maximumCudaLaunchSize > 0
                     && audited.summary.maximumCudaLaunchSize
                            <= audited.summary.cudaFp64LaunchLimit
                     && audited.summary.rgeProfile.requests > 0
                     && audited.summary.rgeProfile.launches > 0
                     && audited.summary.rgeProfile.kernelAndSyncSeconds > 0.0
                     && audited.summary.qSusyProfile.requests > 0
                     && audited.summary.qSusyProfile.launches > 0
                     && audited.summary.qSusyProfile.kernelAndSyncSeconds > 0.0,
                 "audited CUDA label batch lost aligned successful rows");
    if (!ok) return 1;
    bool sawPublicRefinedBracket = false;
    for (std::size_t point = 0; point < audited.results.size(); ++point) {
        const natlha::Result& result = audited.results[point];
        ok &= expect(result.ok
                         && !result.qSusyDiagnostics.empty()
                         && result.qSusyDiagnostics.back()
                                .declaredMaxDeltaLogQ
                                == auditedConfigs[point].qSusyMaxDeltaLogQ
                         && std::isfinite(result.qSusyRootBracketWidth)
                         && result.qSusyRootBracketWidth >= 0.0
                         && result.qSusyRootBracketWidth
                                == result.qSusyDiagnostics.back().refinedBracketWidth
                         && audited.diagnostics[point].executed
                         && audited.diagnostics[point].selectedBackend
                                == natlha::Backend::Cuda
                         && audited.diagnostics[point].auditCompared
                         && audited.diagnostics[point].auditMatched
                         && !audited.diagnostics[point].cpuAdjudicated,
                     "audited CUDA label candidate disagreed with CPU at point "
                         + std::to_string(point) + ": "
                         + audited.diagnostics[point].detail);
        sawPublicRefinedBracket = sawPublicRefinedBracket
            || result.qSusyRootBracketWidth > 0.0;
    }
    ok &= expect(sawPublicRefinedBracket,
                 "public CUDA batch results never exposed a refined root bracket");

    return ok ? 0 : 1;
}
