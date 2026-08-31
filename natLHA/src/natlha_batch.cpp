// Ordered batch dispatch and execution provenance for the public natLHA API.

#include "natlha_api.hpp"

#include <exception>
#include <string>
#include <utility>
#include <vector>

#ifdef NATLHA_HAS_CUDA
#include "natlha_cuda_backend.hpp"
#endif

namespace natlha {
namespace {

enum class BatchContract {
    Full,
    Row
};

BatchRun evaluateCpuBatch(
        const std::vector<Config>& configs,
        Backend requestedBackend,
        const std::string& selectionDetail,
        AdjudicationReasons selectionReasons) {
    BatchRun run;
    run.results.reserve(configs.size());
    run.diagnostics.reserve(configs.size());
    run.summary.points = configs.size();

    for (const Config& config : configs) {
        Result result = evaluate(config);
        if (result.ok) {
            ++run.summary.succeeded;
        } else {
            ++run.summary.failed;
        }
        run.results.push_back(std::move(result));

        PointExecutionDiagnostic diagnostic;
        diagnostic.requestedBackend = requestedBackend;
        diagnostic.selectedBackend = Backend::Cpu;
        diagnostic.candidateTier = ExecutionTier::CpuMpfr;
        diagnostic.finalTier = ExecutionTier::CpuMpfr;
        diagnostic.executed = true;
        diagnostic.adjudicationReasons = selectionReasons;
        diagnostic.detail = selectionDetail;
        run.diagnostics.push_back(std::move(diagnostic));
    }
    return run;
}

BatchRun unavailableCudaBatch(
        const std::vector<Config>& configs,
        Backend requestedBackend,
        const CudaDeviceInfo& device) {
    BatchRun run;
    run.results.reserve(configs.size());
    run.diagnostics.reserve(configs.size());
    run.summary.points = configs.size();
    run.summary.failed = configs.size();

    const std::string error = device.diagnostic.empty()
        ? "CUDA backend is unavailable" : device.diagnostic;
    for (std::size_t i = 0; i < configs.size(); ++i) {
        Result result;
        result.error = error;
        run.results.push_back(std::move(result));

        PointExecutionDiagnostic diagnostic;
        diagnostic.requestedBackend = requestedBackend;
        diagnostic.selectedBackend = Backend::Cuda;
        diagnostic.candidateTier = ExecutionTier::None;
        diagnostic.finalTier = ExecutionTier::None;
        diagnostic.adjudicationReasons =
            adjudicationReason(AdjudicationReason::BackendUnavailable);
        diagnostic.detail = error;
        run.diagnostics.push_back(std::move(diagnostic));
    }
    return run;
}

BatchRowResult projectBatchRow(const Result& result) {
    BatchRowResult row;
    row.ok = result.ok;
    row.error = result.error;
    row.haveDEW = result.haveDEW;
    row.haveDHS = result.haveDHS;
    row.haveDBG = result.haveDBG;
    row.haveDSN = result.haveDSN;
    row.deltaEW = result.deltaEW;
    row.deltaHS = result.deltaHS;
    row.deltaBG = result.deltaBG;
    row.deltaSN = result.deltaSN;
    row.snTotalNvac = result.snTotalNvac;
    row.qSusy = result.qSusy;
    row.logQGut = result.logQGut;
    row.mZ2 = result.mZ2;
    row.qSusyAudit = summarizeQSusyAudit(result);
    if (result.haveDBG && !result.dbgContributions.empty()) {
        row.dbgIdentity.available = true;
        row.dbgIdentity.label = result.dbgContributions.front().label;
        row.dbgIdentity.ordinal = result.dbgContributions.front().ordinal;
        row.dbgIdentity.tiedDirectionOrdinals =
            result.dbgHeadline.tiedDirectionOrdinals;
    }
    return row;
}

BatchRowRun projectBatchRows(BatchRun run) {
    BatchRowRun rows;
    rows.results.reserve(run.results.size());
    for (const Result& result : run.results) {
        rows.results.push_back(projectBatchRow(result));
    }
    rows.diagnostics = std::move(run.diagnostics);
    rows.summary = std::move(run.summary);
    return rows;
}

BatchRun evaluateBatchContract(
        const std::vector<Config>& configs,
        const BatchOptions& options,
        BatchContract contract) {
    if (options.backend == Backend::Cpu) {
        return evaluateCpuBatch(configs, options.backend, "CPU backend requested", 0);
    }

    const CudaDeviceInfo device = queryCudaDevice(options.cudaDevice);
    if (!device.available) {
        if (options.backend == Backend::Auto) {
            return evaluateCpuBatch(
                configs, options.backend,
                "CUDA unavailable; auto selected the CPU backend: " + device.diagnostic,
                adjudicationReason(AdjudicationReason::BackendUnavailable));
        }
        return unavailableCudaBatch(configs, options.backend, device);
    }

#ifdef NATLHA_HAS_CUDA
    return detail::evaluateCudaBatch(
        configs, options, device,
        contract == BatchContract::Full
            ? detail::CudaResultContract::Full
            : detail::CudaResultContract::BatchRow);
#else
    // `device.available` is false in a CPU-only build. Keep a fail-closed return here so the
    // compiler and static analysers do not have to infer that relationship across functions.
    (void) contract;
    return unavailableCudaBatch(configs, options.backend, device);
#endif
}

}  // namespace

const char* backendName(Backend backend) {
    switch (backend) {
        case Backend::Cpu: return "cpu";
        case Backend::Cuda: return "cuda";
        case Backend::Auto: return "auto";
    }
    return "unknown";
}

const char* executionTierName(ExecutionTier tier) {
    switch (tier) {
        case ExecutionTier::None: return "none";
        case ExecutionTier::CpuMpfr: return "cpu-mpfr";
        case ExecutionTier::CudaFp64: return "cuda-fp64";
        case ExecutionTier::CudaDoubleDouble: return "cuda-double-double";
    }
    return "unknown";
}

CudaDeviceInfo queryCudaDevice(int device) {
#ifdef NATLHA_HAS_CUDA
    return detail::queryCudaDevice(device);
#else
    CudaDeviceInfo info;
    info.device = device;
    info.diagnostic =
        "CUDA backend was not compiled; configure with -DNATLHA_ENABLE_CUDA=ON "
        "-DNATLHA_STATIC_LINK=OFF";
    return info;
#endif
}

BatchRun evaluateBatch(
        const std::vector<Config>& configs,
        const BatchOptions& options) {
    return evaluateBatchContract(configs, options, BatchContract::Full);
}

BatchRun evaluateBatch(
        const Config& commonConfig,
        const std::vector<std::string>& slhaPaths,
        const BatchOptions& options) {
    std::vector<Config> configs;
    configs.reserve(slhaPaths.size());
    for (const std::string& path : slhaPaths) {
        Config config = commonConfig;
        config.slhaPath = path;
        configs.push_back(std::move(config));
    }
    return evaluateBatch(configs, options);
}

BatchRowRun evaluateBatchRows(
        const std::vector<Config>& configs,
        const BatchOptions& options) {
    return projectBatchRows(
        evaluateBatchContract(configs, options, BatchContract::Row));
}

BatchRowRun evaluateBatchRows(
        const Config& commonConfig,
        const std::vector<std::string>& slhaPaths,
        const BatchOptions& options) {
    std::vector<Config> configs;
    configs.reserve(slhaPaths.size());
    for (const std::string& path : slhaPaths) {
        Config config = commonConfig;
        config.slhaPath = path;
        configs.push_back(std::move(config));
    }
    return evaluateBatchRows(configs, options);
}

}  // namespace natlha
