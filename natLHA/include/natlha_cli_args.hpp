#ifndef NATLHA_CLI_ARGS_HPP
#define NATLHA_CLI_ARGS_HPP

#include <cstdint>
#include <cmath>
#include <exception>
#include <limits>
#include <ostream>
#include <string>

#include "natlha_api.hpp"

namespace natlha_cli {

enum class ParseStatus {
    Ok,
    Help,
    Error
};

struct Options {
    natlha::Config config;
    natlha::BatchOptions batchOptions;
    std::string singlePath;
    std::string batchPath;
    std::string outputPath;
    std::string backendProvenancePath;
    bool backendProvenanceSet = false;
    int digits = 12;
    bool randomSN = false;
    bool fixedNF = false;
    bool fixedND = false;
    bool qSusyAudit = false;
    bool cudaDeviceSet = false;
    bool cudaBatchSizeSet = false;
    bool cudaWorkersSet = false;
    uint64_t snSeed = 0;
};

inline bool parseBackend(int argc, char** argv, int& index, natlha::Backend& destination,
                         const char* option, std::ostream& error) {
    if (index + 1 >= argc) {
        error << "error: " << option << " needs a value\n";
        return false;
    }
    const std::string raw = argv[++index];
    if (raw == "cpu") destination = natlha::Backend::Cpu;
    else if (raw == "cuda") destination = natlha::Backend::Cuda;
    else if (raw == "auto") destination = natlha::Backend::Auto;
    else {
        error << "error: " << option << " must be cpu, cuda, or auto; got "
              << raw << "\n";
        return false;
    }
    return true;
}

inline bool parseInt(int argc, char** argv, int& index, int& destination,
                     const char* option, int minimum, int maximum,
                     std::ostream& error) {
    if (index + 1 >= argc) {
        error << "error: " << option << " needs a value\n";
        return false;
    }
    const std::string raw = argv[++index];
    int value = 0;
    try {
        std::size_t position = 0;
        value = std::stoi(raw, &position);
        if (position != raw.size()) {
            error << "error: " << option << " value has trailing characters: "
                  << raw << "\n";
            return false;
        }
    } catch (const std::exception&) {
        error << "error: " << option << " value is not an integer: " << raw << "\n";
        return false;
    }
    if (value < minimum || value > maximum) {
        error << "error: " << option << " must be in [" << minimum << ", "
              << maximum << "], got " << value << "\n";
        return false;
    }
    destination = value;
    return true;
}

inline bool parseUint64(int argc, char** argv, int& index, uint64_t& destination,
                        const char* option, std::ostream& error) {
    if (index + 1 >= argc) {
        error << "error: " << option << " needs a value\n";
        return false;
    }
    const std::string raw = argv[++index];
    if (raw.empty() || raw[0] == '-') {
        error << "error: " << option << " value is not an unsigned integer: "
              << raw << "\n";
        return false;
    }
    try {
        std::size_t position = 0;
        const unsigned long long value = std::stoull(raw, &position);
        if (position != raw.size()) {
            error << "error: " << option << " value has trailing characters: "
                  << raw << "\n";
            return false;
        }
        destination = static_cast<uint64_t>(value);
        return true;
    } catch (const std::exception&) {
        error << "error: " << option << " value is not an unsigned integer: "
              << raw << "\n";
        return false;
    }
}

inline bool parsePositiveDouble(int argc, char** argv, int& index, double& destination,
                                const char* option, std::ostream& error) {
    if (index + 1 >= argc) {
        error << "error: " << option << " needs a value\n";
        return false;
    }
    const std::string raw = argv[++index];
    double value = 0.0;
    try {
        std::size_t position = 0;
        value = std::stod(raw, &position);
        if (position != raw.size()) {
            error << "error: " << option << " value has trailing characters: "
                  << raw << "\n";
            return false;
        }
    } catch (const std::exception&) {
        error << "error: " << option << " value is not a number: " << raw << "\n";
        return false;
    }
    if (!std::isfinite(value) || value <= 0.0) {
        error << "error: " << option << " must be finite and positive, got "
              << raw << "\n";
        return false;
    }
    destination = value;
    return true;
}

inline ParseStatus parseArgs(int argc, char** argv, Options& options,
                             std::ostream& error) {
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--slha" && i + 1 < argc) {
            options.singlePath = argv[++i];
        } else if (argument == "--batch" && i + 1 < argc) {
            options.batchPath = argv[++i];
        } else if (argument == "--out" && i + 1 < argc) {
            options.outputPath = argv[++i];
        } else if (argument == "--backend-provenance-out") {
            if (i + 1 >= argc) {
                error << "error: --backend-provenance-out needs a value\n";
                return ParseStatus::Error;
            }
            options.backendProvenanceSet = true;
            options.backendProvenancePath = argv[++i];
        } else if (argument == "--dhs") {
            options.config.computeDHS = true;
        } else if (argument == "--dbg") {
            options.config.computeDBG = true;
        } else if (argument == "--dsn") {
            options.config.computeDSN = true;
        } else if (argument == "--bg-model") {
            if (!parseInt(argc, argv, i, options.config.bgModelIndex,
                          argument.c_str(), 1, 6, error)) return ParseStatus::Error;
        } else if (argument == "--bg-precision") {
            if (!parseInt(argc, argv, i, options.config.bgPrecision,
                          argument.c_str(), 1, 3, error)) return ParseStatus::Error;
        } else if (argument == "--sn-mode") {
            if (!parseInt(argc, argv, i, options.config.snMode,
                          argument.c_str(), 1, 3, error)) return ParseStatus::Error;
        } else if (argument == "--sn-nf") {
            if (!parseInt(argc, argv, i, options.config.snNF,
                          argument.c_str(), 0, 1000000, error)) return ParseStatus::Error;
            options.fixedNF = true;
        } else if (argument == "--sn-nd") {
            if (!parseInt(argc, argv, i, options.config.snND,
                          argument.c_str(), 0, 1000000, error)) return ParseStatus::Error;
            options.fixedND = true;
        } else if (argument == "--sn-random-seed") {
            if (!parseUint64(argc, argv, i, options.snSeed,
                             argument.c_str(), error)) return ParseStatus::Error;
            options.randomSN = true;
        } else if (argument == "--qsusy-max-dlogq") {
            if (!parsePositiveDouble(
                    argc, argv, i, options.config.qSusyMaxDeltaLogQ,
                    argument.c_str(), error)) return ParseStatus::Error;
        } else if (argument == "--qsusy-audit") {
            options.qSusyAudit = true;
        } else if (argument == "--backend") {
            if (!parseBackend(argc, argv, i, options.batchOptions.backend,
                              argument.c_str(), error)) return ParseStatus::Error;
        } else if (argument == "--cuda-device") {
            if (!parseInt(argc, argv, i, options.batchOptions.cudaDevice,
                          argument.c_str(), 0, std::numeric_limits<int>::max(), error)) {
                return ParseStatus::Error;
            }
            options.cudaDeviceSet = true;
        } else if (argument == "--cuda-batch-size") {
            uint64_t value = 0;
            if (!parseUint64(argc, argv, i, value, argument.c_str(), error)) {
                return ParseStatus::Error;
            }
            if (value > std::numeric_limits<std::size_t>::max()) {
                error << "error: --cuda-batch-size exceeds this platform's size range\n";
                return ParseStatus::Error;
            }
            options.batchOptions.cudaBatchSize = static_cast<std::size_t>(value);
            options.cudaBatchSizeSet = true;
        } else if (argument == "--cuda-workers") {
            uint64_t value = 0;
            if (!parseUint64(argc, argv, i, value, argument.c_str(), error)) {
                return ParseStatus::Error;
            }
            constexpr uint64_t maximumWorkers = 4096;
            if (value > maximumWorkers) {
                error << "error: --cuda-workers must be in [0, "
                      << maximumWorkers << "], got " << value << "\n";
                return ParseStatus::Error;
            }
            options.batchOptions.cudaWorkers = static_cast<std::size_t>(value);
            options.cudaWorkersSet = true;
        } else if (argument == "--backend-audit") {
            options.batchOptions.backendAudit = true;
        } else if (argument == "--digits") {
            if (!parseInt(argc, argv, i, options.digits,
                          argument.c_str(), 1, 50, error)) return ParseStatus::Error;
        } else if (argument == "-h" || argument == "--help") {
            return ParseStatus::Help;
        } else {
            error << "error: unrecognised argument: " << argument << "\n";
            return ParseStatus::Error;
        }
    }

    if (options.singlePath.empty() == options.batchPath.empty()) {
        error << "error: give exactly one of --slha or --batch\n";
        return ParseStatus::Error;
    }
    if (options.randomSN && !options.config.computeDSN) {
        error << "error: --sn-random-seed requires --dsn\n";
        return ParseStatus::Error;
    }
    if (options.randomSN && (options.fixedNF || options.fixedND)) {
        error << "error: --sn-random-seed cannot be combined with --sn-nf or --sn-nd\n";
        return ParseStatus::Error;
    }
    if (options.config.computeDSN && !options.randomSN
            && (!options.fixedNF || !options.fixedND)) {
        error << "error: fixed-mode --dsn requires both --sn-nf and --sn-nd\n";
        return ParseStatus::Error;
    }
    if (options.config.snMode != 3) {
        error << "error: --sn-mode exposes only mode 3; capital Delta_SN "
                 "continuation modes 1 and 2 are deferred\n";
        return ParseStatus::Error;
    }
    if (options.qSusyAudit && options.batchPath.empty()) {
        error << "error: --qsusy-audit requires --batch\n";
        return ParseStatus::Error;
    }
    if (options.batchOptions.backend != natlha::Backend::Cpu
            && options.batchPath.empty()) {
        error << "error: CUDA and auto backends require --batch\n";
        return ParseStatus::Error;
    }
    if ((options.cudaDeviceSet || options.cudaBatchSizeSet || options.cudaWorkersSet)
            && options.batchOptions.backend == natlha::Backend::Cpu) {
        error << "error: --cuda-device, --cuda-batch-size, and --cuda-workers require "
                 "--backend cuda or --backend auto\n";
        return ParseStatus::Error;
    }
    if (options.batchOptions.backendAudit
            && (options.batchPath.empty()
                || options.batchOptions.backend == natlha::Backend::Cpu)) {
        error << "error: --backend-audit requires --batch with "
                 "--backend cuda or --backend auto\n";
        return ParseStatus::Error;
    }
    if (options.backendProvenanceSet && options.backendProvenancePath.empty()) {
        error << "error: --backend-provenance-out path must not be empty\n";
        return ParseStatus::Error;
    }
    if (options.backendProvenanceSet && (options.batchPath.empty()
            || options.batchOptions.backend == natlha::Backend::Cpu)) {
        error << "error: --backend-provenance-out requires --batch with "
                 "--backend cuda or --backend auto\n";
        return ParseStatus::Error;
    }
    return ParseStatus::Ok;
}

}  // namespace natlha_cli

#endif
