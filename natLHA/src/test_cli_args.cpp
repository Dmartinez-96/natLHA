#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "natlha_cli_args.hpp"
#include "natlha_cli_output.hpp"

namespace {

bool expect(bool condition, const std::string& message) {
    if (condition) return true;
    std::cerr << message << "\n";
    return false;
}

natlha_cli::ParseStatus parse(std::vector<std::string> arguments,
                              natlha_cli::Options& options,
                              std::string& errorText) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (auto& argument : arguments) argv.push_back(argument.data());
    std::ostringstream errors;
    const natlha_cli::ParseStatus status = natlha_cli::parseArgs(
        static_cast<int>(argv.size()), argv.data(), options, errors);
    errorText = errors.str();
    return status;
}

}  // namespace

int main() {
    bool ok = true;
    std::string error;

    natlha_cli::Options omitted;
    ok &= expect(parse({"natlha-cli", "--slha", "point.slha", "--dbg"},
                       omitted, error) == natlha_cli::ParseStatus::Ok
                     && omitted.config.bgPrecision == 3
                     && omitted.config.snMode == 3
                     && omitted.config.qSusyMaxDeltaLogQ == 0.1,
                 "omitted --bg-precision did not select adaptive mode 3: " + error);

    for (int precision = 1; precision <= 3; ++precision) {
        natlha_cli::Options explicitMode;
        const std::string value = std::to_string(precision);
        ok &= expect(parse({"natlha-cli", "--slha", "point.slha", "--dbg",
                            "--bg-precision", value},
                           explicitMode, error) == natlha_cli::ParseStatus::Ok
                         && explicitMode.config.bgPrecision == precision,
                     "explicit --bg-precision " + value + " did not reach Config: " + error);
    }

    natlha_cli::Options invalid;
    ok &= expect(parse({"natlha-cli", "--slha", "point.slha", "--bg-precision", "4"},
                       invalid, error) == natlha_cli::ParseStatus::Error
                     && error.find("must be in [1, 3]") != std::string::npos,
                 "out-of-range --bg-precision was not rejected with its range");

    natlha_cli::Options deferredSN;
    ok &= expect(parse({"natlha-cli", "--slha", "point.slha", "--dsn",
                        "--sn-mode", "2", "--sn-nf", "1", "--sn-nd", "1"},
                       deferredSN, error) == natlha_cli::ParseStatus::Error
                     && error.find("continuation modes 1 and 2 are deferred")
                            != std::string::npos,
                 "deferred capital Delta_SN continuation remained CLI-reachable: " + error);

    natlha_cli::Options allOptions;
    ok &= expect(parse({"natlha-cli", "--batch", "points.txt", "--out", "labels.tsv",
                        "--dhs", "--dbg", "--dsn", "--bg-model", "6",
                        "--bg-precision", "2", "--sn-mode", "3", "--sn-nf", "10",
                        "--sn-nd", "5", "--qsusy-max-dlogq", "0.05",
                        "--qsusy-audit", "--backend", "cuda", "--cuda-device", "2",
                        "--cuda-batch-size", "4096", "--cuda-workers", "512",
                        "--backend-audit",
                        "--digits", "30"},
                       allOptions, error) == natlha_cli::ParseStatus::Ok
                     && error.empty()
                     && allOptions.batchPath == "points.txt"
                     && allOptions.outputPath == "labels.tsv"
                     && allOptions.config.computeDHS && allOptions.config.computeDBG
                     && allOptions.config.computeDSN
                     && allOptions.config.bgModelIndex == 6
                     && allOptions.config.bgPrecision == 2
                     && allOptions.config.snMode == 3
                     && allOptions.config.snNF == 10 && allOptions.config.snND == 5
                     && allOptions.config.qSusyMaxDeltaLogQ == 0.05
                     && allOptions.qSusyAudit
                     && allOptions.batchOptions.backend == natlha::Backend::Cuda
                     && allOptions.batchOptions.cudaDevice == 2
                     && allOptions.batchOptions.cudaBatchSize == 4096
                     && allOptions.batchOptions.cudaWorkers == 512
                     && allOptions.batchOptions.backendAudit
                     && allOptions.digits == 30,
                 "the refactored parser changed an existing accepted option: " + error);

    natlha_cli::Options singleAudit;
    ok &= expect(parse({"natlha-cli", "--slha", "point.slha", "--qsusy-audit"},
                       singleAudit, error) == natlha_cli::ParseStatus::Error
                     && error.find("requires --batch") != std::string::npos,
                 "single-point --qsusy-audit was accepted without a batch: " + error);

    for (const std::string backend : {"cuda", "auto"}) {
        natlha_cli::Options singleGpu;
        ok &= expect(parse({"natlha-cli", "--slha", "point.slha",
                            "--backend", backend},
                           singleGpu, error) == natlha_cli::ParseStatus::Error
                         && error.find("require --batch") != std::string::npos,
                     "single-point " + backend + " backend was accepted: " + error);
    }
    natlha_cli::Options unknownBackend;
    ok &= expect(parse({"natlha-cli", "--batch", "points.txt",
                        "--backend", "gpu"},
                       unknownBackend, error) == natlha_cli::ParseStatus::Error
                     && error.find("cpu, cuda, or auto") != std::string::npos,
                 "unknown backend spelling was accepted: " + error);
    natlha_cli::Options cpuCudaControl;
    ok &= expect(parse({"natlha-cli", "--batch", "points.txt",
                        "--cuda-device", "0"},
                       cpuCudaControl, error) == natlha_cli::ParseStatus::Error
                     && error.find("require --backend cuda") != std::string::npos,
                 "CUDA device control was accepted by the CPU backend: " + error);
    natlha_cli::Options cpuAudit;
    ok &= expect(parse({"natlha-cli", "--batch", "points.txt", "--backend-audit"},
                       cpuAudit, error) == natlha_cli::ParseStatus::Error
                     && error.find("requires --batch with") != std::string::npos,
                 "backend audit was accepted without a CUDA-capable selection: " + error);
    natlha_cli::Options tooManyCudaWorkers;
    ok &= expect(parse({"natlha-cli", "--batch", "points.txt", "--backend", "cuda",
                        "--cuda-workers", "4097"},
                       tooManyCudaWorkers, error) == natlha_cli::ParseStatus::Error
                     && error.find("must be in [0, 4096]") != std::string::npos,
                 "excessive CUDA worker count was accepted: " + error);

    natlha_cli::Options autoBoundaryControls;
    ok &= expect(parse({"natlha-cli", "--batch", "points.txt", "--backend", "auto",
                        "--cuda-device", "0", "--cuda-batch-size", "0",
                        "--cuda-workers", "4096", "--backend-audit"},
                       autoBoundaryControls, error) == natlha_cli::ParseStatus::Ok
                     && error.empty()
                     && autoBoundaryControls.batchOptions.backend == natlha::Backend::Auto
                     && autoBoundaryControls.batchOptions.cudaDevice == 0
                     && autoBoundaryControls.batchOptions.cudaBatchSize == 0
                     && autoBoundaryControls.batchOptions.cudaWorkers == 4096
                     && autoBoundaryControls.batchOptions.backendAudit,
                 "valid automatic-backend boundary controls were rejected: " + error);

    natlha_cli::Options missingBackendValue;
    ok &= expect(parse({"natlha-cli", "--batch", "points.txt", "--backend"},
                       missingBackendValue, error) == natlha_cli::ParseStatus::Error
                     && error.find("needs a value") != std::string::npos,
                 "missing backend value was not rejected: " + error);

    natlha_cli::Options negativeCudaDevice;
    ok &= expect(parse({"natlha-cli", "--batch", "points.txt", "--backend", "cuda",
                        "--cuda-device", "-1"},
                       negativeCudaDevice, error) == natlha_cli::ParseStatus::Error
                     && error.find("must be in [0,") != std::string::npos,
                 "negative CUDA device ordinal was accepted: " + error);

    for (const std::string option : {"--cuda-batch-size", "--cuda-workers"}) {
        natlha_cli::Options negativeUnsignedControl;
        ok &= expect(parse({"natlha-cli", "--batch", "points.txt", "--backend", "cuda",
                            option, "-1"},
                           negativeUnsignedControl, error) == natlha_cli::ParseStatus::Error
                         && error.find("not an unsigned integer") != std::string::npos,
                     "negative " + option + " value was accepted: " + error);
    }

    for (const std::string value : {"0", "-0.1", "nan", "inf"}) {
        natlha_cli::Options badSpacing;
        ok &= expect(parse({"natlha-cli", "--slha", "point.slha",
                            "--qsusy-max-dlogq", value},
                           badSpacing, error) == natlha_cli::ParseStatus::Error
                         && error.find("finite and positive") != std::string::npos,
                     "invalid --qsusy-max-dlogq " + value + " was accepted: " + error);
    }
    natlha_cli::Options trailingSpacing;
    ok &= expect(parse({"natlha-cli", "--slha", "point.slha",
                        "--qsusy-max-dlogq", "0.05x"},
                       trailingSpacing, error) == natlha_cli::ParseStatus::Error
                     && error.find("trailing characters") != std::string::npos,
                 "trailing --qsusy-max-dlogq characters were accepted: " + error);
    natlha_cli::Options missingSpacing;
    ok &= expect(parse({"natlha-cli", "--slha", "point.slha",
                        "--qsusy-max-dlogq"},
                       missingSpacing, error) == natlha_cli::ParseStatus::Error
                     && error.find("needs a value") != std::string::npos,
                 "missing --qsusy-max-dlogq value was accepted: " + error);

    natlha::Result noSearches;
    const natlha_cli::QSusyAuditSummary emptyAudit =
        natlha_cli::summarizeQSusyAudit(noSearches);
    ok &= expect(!emptyAudit.allAccepted && !emptyAudit.allCountsKnown
                     && emptyAudit.searches == 0 && !emptyAudit.haveLastRootCount
                     && !emptyAudit.haveAcceptedLogScale,
                 "empty root-search history did not retain unknown-count sentinels");

    natlha::Result acceptedSearches;
    natlha::QSusySearchDiagnostic firstAccepted;
    firstAccepted.scanComplete = true;
    firstAccepted.accepted = true;
    firstAccepted.rootsFound = 1;
    firstAccepted.logScale = 8.0;
    acceptedSearches.qSusySearchDiagnostics.push_back(firstAccepted);
    natlha::QSusySearchDiagnostic lastAccepted = firstAccepted;
    lastAccepted.logScale = 9.0;
    acceptedSearches.qSusySearchDiagnostics.push_back(lastAccepted);
    const natlha_cli::QSusyAuditSummary acceptedAudit =
        natlha_cli::summarizeQSusyAudit(acceptedSearches);
    ok &= expect(acceptedAudit.allAccepted && acceptedAudit.allCountsKnown
                     && acceptedAudit.searches == 2 && acceptedAudit.haveLastRootCount
                     && acceptedAudit.lastRootsFound == 1
                     && acceptedAudit.haveAcceptedLogScale
                     && acceptedAudit.acceptedLogScale == 9.0,
                 "accepted root-search history lost its final measured values");

    natlha::Result structuredRejection = acceptedSearches;
    natlha::QSusySearchDiagnostic rejected;
    rejected.scanComplete = true;
    rejected.rootsFound = 2;
    structuredRejection.qSusySearchDiagnostics.push_back(rejected);
    const natlha_cli::QSusyAuditSummary rejectedAudit =
        natlha_cli::summarizeQSusyAudit(structuredRejection);
    ok &= expect(!rejectedAudit.allAccepted && rejectedAudit.allCountsKnown
                     && rejectedAudit.searches == 3 && rejectedAudit.haveLastRootCount
                     && rejectedAudit.lastRootsFound == 2
                     && !rejectedAudit.haveAcceptedLogScale,
                 "structured root rejection lost its measured root count");

    natlha::Result unstructuredFailure = acceptedSearches;
    unstructuredFailure.qSusySearchDiagnostics.emplace_back();
    const natlha_cli::QSusyAuditSummary unknownAudit =
        natlha_cli::summarizeQSusyAudit(unstructuredFailure);
    ok &= expect(!unknownAudit.allAccepted && !unknownAudit.allCountsKnown
                     && unknownAudit.searches == 3 && !unknownAudit.haveLastRootCount
                     && !unknownAudit.haveAcceptedLogScale,
                 "unstructured root-search failure was reported as a measured zero");

    return ok ? 0 : 1;
}
