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
    ok &= expect(cpu.results.size() == 2 && cpu.diagnostics.size() == 2,
                 "CPU batch did not preserve one output slot per input");
    ok &= expect(cpu.results[0].ok && !cpu.results[1].ok,
                 "CPU batch changed input ordering or failure alignment");
    ok &= expect(cpu.summary.points == 2 && cpu.summary.succeeded == 1
                     && cpu.summary.failed == 1
                     && cpu.summary.rgeProfile.requests == 0
                     && cpu.summary.qSusyProfile.requests == 0,
                 "CPU batch summary did not count aligned success and failure rows");
    for (const auto& diagnostic : cpu.diagnostics) {
        ok &= expect(diagnostic.requestedBackend == natlha::Backend::Cpu
                         && diagnostic.selectedBackend == natlha::Backend::Cpu
                         && diagnostic.candidateTier == natlha::ExecutionTier::CpuMpfr
                         && diagnostic.finalTier == natlha::ExecutionTier::CpuMpfr
                         && diagnostic.executed,
                     "CPU execution provenance named a non-CPU backend or tier");
    }

    const natlha::BatchRun convenience = natlha::evaluateBatch(
        valid, std::vector<std::string>{valid.slhaPath, invalid.slhaPath}, cpuOptions);
    ok &= expect(convenience.results.size() == 2 && convenience.results[0].ok
                     && !convenience.results[1].ok,
                 "path convenience overload changed ordering or path substitution");

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
    ok &= expect(refused.results.size() == 1 && !refused.results[0].ok
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
    ok &= expect(automatic.results.size() == 1 && automatic.results[0].ok
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
