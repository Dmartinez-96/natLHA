#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "natlha_api.hpp"

namespace {

bool expect(bool condition, const std::string& message) {
    if (condition) return true;
    std::cerr << message << "\n";
    return false;
}

bool sameAuditSummary(const natlha::QSusyAuditSummary& left,
                      const natlha::QSusyAuditSummary& right) {
    return left.allAccepted == right.allAccepted
        && left.allCountsKnown == right.allCountsKnown
        && left.searches == right.searches
        && left.haveLastRootCount == right.haveLastRootCount
        && left.lastRootsFound == right.lastRootsFound
        && left.haveAcceptedLogScale == right.haveAcceptedLogScale
        && left.acceptedLogScale == right.acceptedLogScale;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: test-batch-api FIXTURE.slha\n";
        return 2;
    }

    bool ok = true;
#ifdef NATLHA_HAS_CUDA
    std::cout << "batch-api mode=cuda\n";
#else
    std::cout << "batch-api mode=cpu-only\n";
#endif
    natlha::Config valid;
    valid.slhaPath = argv[1];
    natlha::Config invalid = valid;
    invalid.slhaPath = std::string(argv[1]) + ".missing";

    natlha::BatchOptions cpuOptions;
    const natlha::BatchRun cpu = natlha::evaluateBatch({valid, invalid}, cpuOptions);
    const bool haveCpuAlignment =
        cpu.results.size() == 2 && cpu.diagnostics.size() == 2;
    ok &= expect(haveCpuAlignment,
                 "CPU batch did not preserve one output slot per input");
    if (haveCpuAlignment) {
        ok &= expect(cpu.results[0].ok && !cpu.results[1].ok,
                     "CPU batch changed input ordering or failure alignment");
        for (const auto& diagnostic : cpu.diagnostics) {
            ok &= expect(diagnostic.requestedBackend == natlha::Backend::Cpu
                             && diagnostic.selectedBackend == natlha::Backend::Cpu
                             && diagnostic.candidateTier == natlha::ExecutionTier::CpuMpfr
                             && diagnostic.finalTier == natlha::ExecutionTier::CpuMpfr
                             && diagnostic.executed,
                         "CPU execution provenance named a non-CPU backend or tier");
        }
    }
    ok &= expect(cpu.summary.points == 2 && cpu.summary.succeeded == 1
                     && cpu.summary.failed == 1
                     && cpu.summary.rgeProfile.requests == 0
                     && cpu.summary.qSusyProfile.requests == 0,
                 "CPU batch summary did not count aligned success and failure rows");

    const natlha::BatchRowRun cpuRows =
        natlha::evaluateBatchRows({valid, invalid}, cpuOptions);
    if (haveCpuAlignment) {
        ok &= expect(cpuRows.results.size() == 2 && cpuRows.diagnostics.size() == 2
                         && cpuRows.results[0].ok && !cpuRows.results[1].ok
                         && cpuRows.results[1].error == cpu.results[1].error
                         && cpuRows.results[0].qSusy == cpu.results[0].qSusy
                         && cpuRows.results[0].deltaEW == cpu.results[0].deltaEW
                         && sameAuditSummary(
                             cpuRows.results[0].qSusyAudit,
                             natlha::summarizeQSusyAudit(cpu.results[0])),
                     "CPU batch-row projection changed alignment or emitted values");
    }

    natlha::Config dbgValid = valid;
    dbgValid.computeDEW = false;
    dbgValid.computeDBG = true;
    const natlha::BatchRun dbgFull =
        natlha::evaluateBatch({dbgValid}, cpuOptions);
    const natlha::BatchRowRun dbgRows =
        natlha::evaluateBatchRows({dbgValid}, cpuOptions);
    const bool haveDbgProjection = dbgFull.results.size() == 1
        && dbgRows.results.size() == 1
        && dbgFull.results[0].ok
        && dbgRows.results[0].ok
        && dbgFull.results[0].haveDBG
        && !dbgFull.results[0].dbgContributions.empty();
    ok &= expect(haveDbgProjection,
                 "CPU Delta_BG batch-row projection fixture did not produce a label");
    if (haveDbgProjection) {
        const natlha::Result& full = dbgFull.results[0];
        const natlha::BatchRowResult& row = dbgRows.results[0];
        ok &= expect(row.ok == full.ok
                         && row.error == full.error
                         && row.haveDEW == full.haveDEW
                         && row.haveDHS == full.haveDHS
                         && row.haveDBG == full.haveDBG
                         && row.haveDSN == full.haveDSN
                         && row.deltaEW == full.deltaEW
                         && row.deltaHS == full.deltaHS
                         && row.deltaBG == full.deltaBG
                         && row.deltaSN == full.deltaSN
                         && row.snTotalNvac == full.snTotalNvac
                         && row.qSusy == full.qSusy
                         && row.logQGut == full.logQGut
                         && row.mZ2 == full.mZ2
                         && sameAuditSummary(
                             row.qSusyAudit, natlha::summarizeQSusyAudit(full)),
                     "CPU Delta_BG batch-row projection changed an emitted field");
        ok &= expect(full.dbgHeadline.topLabel
                            == full.dbgContributions.front().label
                         && full.dbgHeadline.topValue
                                == full.dbgContributions.front().value
                         && row.dbgIdentity.available
                         && row.dbgIdentity.label
                                == full.dbgHeadline.topLabel
                         && row.dbgIdentity.ordinal
                                == full.dbgContributions.front().ordinal
                         && row.dbgIdentity.tiedDirectionOrdinals
                                == full.dbgHeadline.tiedDirectionOrdinals,
                     "CPU batch-row projection changed the Delta_BG headline identity");
    }

    const natlha::BatchRun convenience = natlha::evaluateBatch(
        valid, std::vector<std::string>{valid.slhaPath, invalid.slhaPath}, cpuOptions);
    ok &= expect(convenience.results.size() == 2 && convenience.results[0].ok
                     && !convenience.results[1].ok,
                 "path convenience overload changed ordering or path substitution");
    const natlha::BatchRowRun rowConvenience = natlha::evaluateBatchRows(
        valid, std::vector<std::string>{valid.slhaPath, invalid.slhaPath}, cpuOptions);
    ok &= expect(rowConvenience.results.size() == 2
                     && rowConvenience.results[0].ok
                     && !rowConvenience.results[1].ok,
                 "batch-row path convenience overload changed input alignment");

    natlha::BatchOptions unavailableCuda;
    unavailableCuda.backend = natlha::Backend::Cuda;
    unavailableCuda.cudaDevice = std::numeric_limits<int>::max();
    const natlha::CudaDeviceInfo invalidDevice =
        natlha::queryCudaDevice(unavailableCuda.cudaDevice);
#ifdef NATLHA_HAS_CUDA
    ok &= expect(invalidDevice.compiled,
                 "CUDA-enabled build reported that its backend was not compiled");
#else
    ok &= expect(!invalidDevice.compiled,
                 "CPU-only build reported that the CUDA backend was compiled");
    ok &= expect(invalidDevice.diagnostic.find("-DNATLHA_ENABLE_CUDA=ON")
                         != std::string::npos
                     && invalidDevice.diagnostic.find("-DNATLHA_STATIC_LINK=OFF")
                         != std::string::npos,
                 "CPU-only CUDA diagnostic omitted a required configuration flag");
#endif
    ok &= expect(!invalidDevice.available && !invalidDevice.diagnostic.empty(),
                 "out-of-range or uncompiled CUDA device was reported as available");
    const natlha::BatchRun refused = natlha::evaluateBatch({valid}, unavailableCuda);
    ok &= expect(refused.results.size() == 1 && refused.diagnostics.size() == 1
                     && !refused.results[0].ok
                     && refused.summary.failed == 1
                     && natlha::hasAdjudicationReason(
                         refused.diagnostics[0].adjudicationReasons,
                         natlha::AdjudicationReason::BackendUnavailable)
                     && !refused.diagnostics[0].executed
                     && refused.diagnostics[0].candidateTier
                            == natlha::ExecutionTier::None
                     && refused.diagnostics[0].finalTier == natlha::ExecutionTier::None,
                 "explicit unavailable CUDA request did not fail closed");

    unavailableCuda.backend = natlha::Backend::Auto;
    const natlha::BatchRun automatic = natlha::evaluateBatch({valid}, unavailableCuda);
    ok &= expect(automatic.results.size() == 1 && automatic.diagnostics.size() == 1
                     && automatic.results[0].ok
                     && automatic.diagnostics[0].selectedBackend == natlha::Backend::Cpu
                     && natlha::hasAdjudicationReason(
                         automatic.diagnostics[0].adjudicationReasons,
                         natlha::AdjudicationReason::BackendUnavailable),
                 "auto backend did not record its unavailable-CUDA CPU selection");

    const natlha::BatchRun empty = natlha::evaluateBatch({}, cpuOptions);
    ok &= expect(empty.results.empty() && empty.diagnostics.empty()
                     && empty.summary.points == 0,
                 "empty batch did not return an empty aligned result");
    ok &= expect(std::string(natlha::backendName(natlha::Backend::Cuda)) == "cuda"
                     && std::string(natlha::executionTierName(
                         natlha::ExecutionTier::None)) == "none"
                     && std::string(natlha::executionTierName(
                         natlha::ExecutionTier::CudaDoubleDouble)) == "cuda-double-double",
                 "stable backend/tier names changed");

    return ok ? 0 : 1;
}
