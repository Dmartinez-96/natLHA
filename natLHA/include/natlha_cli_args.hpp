#ifndef NATLHA_CLI_ARGS_HPP
#define NATLHA_CLI_ARGS_HPP

#include <cstdint>
#include <cmath>
#include <exception>
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
    std::string singlePath;
    std::string batchPath;
    std::string outputPath;
    int digits = 12;
    bool randomSN = false;
    bool fixedNF = false;
    bool fixedND = false;
    bool qSusyAudit = false;
    uint64_t snSeed = 0;
};

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
    return ParseStatus::Ok;
}

}  // namespace natlha_cli

#endif
