#ifndef NATLHA_CUDA_BACKEND_HPP
#define NATLHA_CUDA_BACKEND_HPP

#include <cstddef>
#include <string>
#include <vector>

#include "MSSM_RGE_solver_with_stopfinder.hpp"
#include "natlha_api.hpp"

namespace natlha::detail {

constexpr std::size_t kRgeStateSize = 44;

enum class CudaOdeStatus : int {
    Success = 0,
    NonFiniteInput = 1,
    NonFiniteState = 2,
    StepLimit = 3,
    StepUnderflow = 4
};

enum class CudaQSusyStatus : int {
    Success = 0,
    NonFiniteInput = 1,
    NonFiniteState = 2,
    StepLimit = 3,
    StepUnderflow = 4,
    ScanSpacing = 5,
    BoundaryCounterOverflow = 6,
    NonPhysicalBracket = 7,
    RefinementFailure = 8,
    ResidualFailure = 9,
    NonUniqueRoot = 10,
    NonFiniteBoundary = 11
};

enum class CudaResultContract {
    Full,
    BatchRow
};

constexpr AdjudicationReasons cudaQSusyAdjudicationReasons(
        CudaQSusyStatus status) {
    const auto root = adjudicationReason(AdjudicationReason::RootBoundary);
    switch (status) {
        case CudaQSusyStatus::NonFiniteInput:
        case CudaQSusyStatus::NonFiniteState:
        case CudaQSusyStatus::NonFiniteBoundary:
            return root | adjudicationReason(AdjudicationReason::NonFiniteState);
        case CudaQSusyStatus::StepLimit:
            return root | adjudicationReason(AdjudicationReason::OdeStepLimit);
        case CudaQSusyStatus::StepUnderflow:
        case CudaQSusyStatus::RefinementFailure:
        case CudaQSusyStatus::ResidualFailure:
            return root | adjudicationReason(AdjudicationReason::ErrorEstimate);
        case CudaQSusyStatus::ScanSpacing:
        case CudaQSusyStatus::BoundaryCounterOverflow:
        case CudaQSusyStatus::NonPhysicalBracket:
        case CudaQSusyStatus::NonUniqueRoot:
            return root;
        case CudaQSusyStatus::Success:
            return adjudicationReason(AdjudicationReason::None);
    }
    return adjudicationReason(AdjudicationReason::InfrastructureFailure);
}

inline void recordCudaPointResult(
        PointExecutionDiagnostic& diagnostic, bool doubleDouble) {
    diagnostic.executed = true;
    if (diagnostic.candidateTier == ExecutionTier::None) {
        diagnostic.candidateTier = doubleDouble
            ? ExecutionTier::CudaDoubleDouble : ExecutionTier::CudaFp64;
    }
    if (!doubleDouble) diagnostic.finalTier = ExecutionTier::CudaFp64;
}

inline void recordCudaPointEscape(
        PointExecutionDiagnostic& diagnostic, bool doubleDouble) {
    diagnostic.finalTier = ExecutionTier::None;
    if (!doubleDouble) {
        diagnostic.candidateTier = ExecutionTier::None;
        diagnostic.executed = false;
    }
}

/// Structured host-side CUDA acceptance diagnostics. These are exposed from the detail
/// namespace so contract tests execute the same predicates used by production adjudication.
AdjudicationReasons cudaBranchBoundaryReasons(const Result& candidate);
std::string cudaResultMismatch(const Result& candidate, const Result& reference);
std::string cudaBatchRowMismatch(const Result& candidate, const Result& reference);
bool cudaBatchRowAcceptsBranchReasons(AdjudicationReasons reasons);
bool cudaBatchRowAcceptsRelaxedDiagnosticReasons(AdjudicationReasons reasons);

struct CudaOdeBatch {
    /// Point-major 44-component input states.
    std::vector<double> states;
    std::vector<double> startTimes;
    std::vector<double> endTimes;
    std::vector<double> initialSteps;
};

struct CudaOdeBatchResult {
    /// Point-major 44-component final states, aligned with the input trajectories.
    std::vector<double> states;
    std::vector<CudaOdeStatus> statuses;
    std::vector<unsigned long long> acceptedSteps;
    std::vector<unsigned long long> rejectedSteps;
    CudaStageProfile profile;
};

struct CudaQSusyBatch {
    std::vector<double> states;
    std::vector<double> highLogScales;
    std::vector<double> initialSteps;
    std::vector<double> maxDeltaLogQ;
};

struct CudaQSusyBatchResult {
    std::vector<double> statesAtRoot;
    std::vector<double> logScales;
    std::vector<double> residuals;
    std::vector<double> stop1Squared;
    std::vector<double> stop2Squared;
    std::vector<double> refinedBracketWidths;
    std::vector<CudaQSusyStatus> statuses;
    std::vector<unsigned long long> acceptedSteps;
    std::vector<unsigned long long> rejectedSteps;
    std::vector<unsigned long long> scanSegments;
    std::vector<double> maxObservedDeltaLogQ;
    std::vector<unsigned long long> rootsFound;
    std::vector<unsigned long long> invalidBoundaries;
    std::vector<unsigned long long> nonFiniteBoundaries;
    std::vector<unsigned long long> refinementEvaluations;
    CudaStageProfile profile;
};

/// Fixed inputs used to prove that CPU and device instantiations of the shared Q_SUSY
/// scalar helpers agree. This is an internal validation surface, not the production root
/// batch contract.
struct CudaQSusyHelperInput {
    std::vector<double> state;
    double logScale = 0.0;
    double currentLogScale = 0.0;
    double lowerLogScale = 0.0;
    double maxDeltaLogQ = 0.0;
    StopScalePoint highPoint;
    StopScalePoint lowPoint;
    bool inInvalidDomain = false;
    bool inNonFiniteDomain = false;
    unsigned long long invalidBoundaries = 0;
    unsigned long long nonFiniteBoundaries = 0;
};

struct CudaQSusyHelperResult {
    StopScalePoint evaluatedPoint;
    double nextLogScale = 0.0;
    int scanStepStatus = 0;
    unsigned scanEvents = 0;
    bool classificationOk = false;
    bool inInvalidDomain = false;
    bool inNonFiniteDomain = false;
    unsigned long long invalidBoundaries = 0;
    unsigned long long nonFiniteBoundaries = 0;
};

CudaDeviceInfo queryCudaDevice(int device);

CudaOdeBatchResult solveCudaOdeBatchFp64(
    const CudaOdeBatch& batch,
    int device,
    std::size_t requestedBatchSize = 0);

CudaOdeBatchResult solveCudaOdeBatchDoubleDouble(
    const CudaOdeBatch& batch,
    int device,
    std::size_t requestedBatchSize = 0);

CudaQSusyBatchResult solveCudaQSusyBatchFp64(
    const CudaQSusyBatch& batch,
    int device,
    std::size_t requestedBatchSize = 0);

CudaQSusyBatchResult solveCudaQSusyBatchDoubleDouble(
    const CudaQSusyBatch& batch,
    int device,
    std::size_t requestedBatchSize = 0);

std::vector<CudaQSusyHelperResult> evaluateCudaQSusyHelpers(
    const std::vector<CudaQSusyHelperInput>& inputs,
    int device);

BatchRun evaluateCudaBatch(
    const std::vector<Config>& configs,
    const BatchOptions& options,
    const CudaDeviceInfo& device,
    CudaResultContract contract);

}  // namespace natlha::detail

#endif  // NATLHA_CUDA_BACKEND_HPP
