// Non-interactive command-line front end over natlha::evaluate.
//
// Two modes, because the two use cases have opposite requirements:
//
//   single point   natlha-cli --slha FILE [--dhs] [--dbg] [--dsn] ...
//                  Prints a labelled, human-readable report including the full contribution
//                  breakdown. For debugging one point and for reproducing a benchmark.
//
//   batch          natlha-cli --batch LIST [--dhs] ...
//                  LIST is a file of SLHA paths, one per line, blank lines and lines
//                  starting with '#' ignored. Emits one whitespace-separated row per point
//                  to stdout by default, or to FILE with --out, with a '#'-commented header.
//                  Process startup and static initialisation are paid ONCE for the whole list
//                  rather than per point, which is the point of the mode.
//
// Rows for points that fail are still emitted, with ok=0, so a caller can align input and
// output line by line instead of guessing which point vanished. Every label field is zero on
// an ok=0 row: evaluate() invalidates the complete requested label row if any later measure
// fails. In --sn-random-seed mode, a filename that has no numeric draw-index stem is rejected
// before evaluate() and its sn_nF/sn_nD fields are written as 0/0 sentinels. Diagnostics go
// to stderr so the output stream stays parseable.
//
// Exit status: 0 if every requested point succeeded, 1 on usage error, 2 if any point failed.

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <streambuf>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "natlha_api.hpp"
#include "natlha_cli_args.hpp"
#include "natlha_cli_output.hpp"

namespace {

void usage() {
    std::cerr <<
        "usage: natlha-cli --slha FILE  [measure flags] [options]\n"
        "       natlha-cli --batch LIST [measure flags] [options]\n"
        "\n"
        "measure flags (Delta_EW is always computed):\n"
        "  --dhs                also compute Delta_HS\n"
        "  --dbg                also compute Delta_BG\n"
        "  --dsn                also compute delta_SN\n"
        "\n"
        "options:\n"
        "  --out FILE           write report or batch rows to FILE (default stdout)\n"
        "  --bg-model N         DBG_calc model index, 1-6; 6=pMSSM-30+mu (default 1)\n"
        "  --bg-precision N     1=8-point diagnostic, 2=4-point diagnostic,\n"
        "                       3=adaptive 2-point production (default 3)\n"
        "  --sn-mode 3          lowercase differential mode    (default 3)\n"
        "                       capital continuation modes 1/2 are deferred\n"
        "  --sn-nf N            fixed F-term count; required with --dsn unless random\n"
        "  --sn-nd N            fixed D-term count; required with --dsn unless random\n"
        "  --sn-random-seed N   choose one reproducible uniform (nF,nD) pair per draw,\n"
        "                       with nF in 1-10 and nD in 1-5; requires --dsn\n"
        "  --qsusy-max-dlogq H maximum Q_SUSY scan spacing in log(Q)\n"
        "                       (provisional audit candidate default 0.1)\n"
        "  --qsusy-audit       append structured Q_SUSY freeze-audit columns\n"
        "                       to batch rows without changing default output\n"
        "  --digits N           printed significant digits     (default 12)\n";
}

class CoutRedirect {
public:
    explicit CoutRedirect(const std::string & path) : requested_(!path.empty()) {
        if (!requested_) return;
        output_.open(path);
        if (output_.good()) original_ = std::cout.rdbuf(output_.rdbuf());
    }

    ~CoutRedirect() {
        if (original_ != nullptr) {
            std::cout.flush();
            std::cout.rdbuf(original_);
        }
    }

    CoutRedirect(const CoutRedirect &) = delete;
    CoutRedirect & operator=(const CoutRedirect &) = delete;

    bool good() const { return !requested_ || output_.good(); }

private:
    bool requested_ = false;
    std::ofstream output_;
    std::streambuf * original_ = nullptr;
};

bool sameExistingFile(const std::string & inputPath, const std::string & outputPath) {
    if (outputPath.empty()) return false;
    std::error_code error;
    const bool equivalent = std::filesystem::equivalent(inputPath, outputPath, error);
    return !error && equivalent;
}

uint64_t splitmix64(uint64_t & state) {
    state += UINT64_C(0x9E3779B97F4A7C15);
    uint64_t z = state;
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

bool drawIndex(const std::string & path, uint64_t & draw) {
    const std::string stem = std::filesystem::path(path).stem().string();
    if (stem.empty()) return false;
    for (const char c : stem) {
        if (c < '0' || c > '9') return false;
    }
    try {
        std::size_t pos = 0;
        draw = std::stoull(stem, &pos);
        return pos == stem.size();
    } catch (const std::exception &) {
        return false;
    }
}

void randomSNPair(uint64_t seed, uint64_t draw, int & nF, int & nD) {
    uint64_t state = seed ^ (draw * UINT64_C(0xD1B54A32D192ED03));
    const uint64_t bound = 50;
    const uint64_t threshold = (uint64_t(0) - bound) % bound;
    uint64_t sample = 0;
    do {
        sample = splitmix64(state);
    } while (sample < threshold);
    const int pair = static_cast<int>(sample % bound);
    nF = 1 + pair / 5;
    nD = 1 + pair % 5;
}

void printRow(const std::string & path, const natlha::Result & r, const natlha::Config & cfg,
              int digits, bool randomSN, bool qSusyAudit) {
    std::cout << std::setprecision(digits) << std::scientific
              << (r.ok ? 1 : 0) << " " << r.deltaEW;
    if (cfg.computeDHS) std::cout << " " << r.deltaHS;
    if (cfg.computeDBG) std::cout << " " << r.deltaBG;
    if (cfg.computeDSN) {
        std::cout << " " << r.deltaSN << " " << r.snTotalNvac;
        if (randomSN) std::cout << " " << cfg.snNF << " " << cfg.snND;
    }
    std::cout << " " << r.qSusy << " " << r.logQGut << " " << r.mZ2;
    if (qSusyAudit) {
        const natlha_cli::QSusyAuditSummary audit =
            natlha_cli::summarizeQSusyAudit(r);
        std::cout << " " << (audit.allAccepted ? 1 : 0);
        if (audit.haveLastRootCount) {
            std::cout << " " << audit.lastRootsFound;
        } else {
            std::cout << " -1";
        }
        std::cout << " " << (audit.allCountsKnown ? 1 : 0)
                  << " " << audit.searches;
        if (audit.haveAcceptedLogScale) {
            std::cout << " " << audit.acceptedLogScale;
        } else {
            std::cout << " " << 0.0;
        }
    }
    std::cout << " " << path << "\n";
}

void printHeader(const natlha::Config & cfg, bool randomSN, uint64_t snSeed,
                 bool qSusyAudit) {
    if (randomSN) std::cout << "# sn_random_seed " << snSeed << "\n";
    std::cout << "# ok Delta_EW";
    if (cfg.computeDHS) std::cout << " Delta_HS";
    if (cfg.computeDBG) std::cout << " Delta_BG";
    if (cfg.computeDSN) {
        std::cout << " delta_SN dN_vac";
        if (randomSN) std::cout << " sn_nF sn_nD";
    }
    std::cout << " Q_SUSY logQ_GUT mZ2";
    if (qSusyAudit) {
        std::cout << " Q_SUSY_search_ok Q_SUSY_roots Q_SUSY_scan_complete"
                     " Q_SUSY_searches Q_SUSY_search_logQ";
    }
    std::cout << " slha_path\n";
}

void printReport(const natlha::Result & r, const natlha::Config & cfg, int digits,
                 bool randomSN, uint64_t snSeed) {
    std::cout << std::setprecision(digits);
    std::cout << "Q_SUSY max dlogQ " << cfg.qSusyMaxDeltaLogQ << "\n";
    if (!r.ok) {
        std::cout << "FAILED: " << r.error << "\n";
        for (const auto& diagnostic : r.qSusyDiagnostics) {
            std::cout << "Q_SUSY iteration " << diagnostic.iteration
                      << "  Q " << diagnostic.qSusy
                      << "  residual " << diagnostic.residual
                      << "  mu " << diagnostic.mu
                      << "  stop1_sq " << diagnostic.stop1Squared
                      << "  stop2_sq " << diagnostic.stop2Squared
                      << "  ODE_steps " << diagnostic.acceptedSteps
                      << "  max_dlogQ " << diagnostic.declaredMaxDeltaLogQ
                      << "  scan_segments " << diagnostic.scanSegments
                      << "  max_observed_dlogQ "
                      << diagnostic.maxObservedDeltaLogQ
                      << "  roots " << diagnostic.rootsFound
                      << "  invalid_boundaries " << diagnostic.invalidBoundaries
                      << "  root_evaluations " << diagnostic.refinementEvaluations << "\n";
        }
        return;
    }
    std::cout << "Q_SUSY    " << r.qSusy << "\n"
              << "logQ_GUT  " << r.logQGut << "\n"
              << "mZ2       " << r.mZ2 << "   (should be 91.1876^2 = 8315.17839376)\n"
              << "mu(Q_SUSY) " << r.weakBCs[6]
              << "   <- re-solved from EWSB, NOT the SLHA's mu\n"
              << "b = B*mu  " << r.weakBCs[42] << "\n"
              << "Sigma_u   " << r.radCorrs[0] << "\n"
              << "Sigma_d   " << r.radCorrs[1] << "\n"
              << "iters     q_susy " << r.qSusyIters
              << "   ewsb " << r.ewsbIters << "   gut " << r.gutIters << "\n";
    for (const auto& diagnostic : r.qSusyDiagnostics) {
        std::cout << "Q_SUSY iteration " << diagnostic.iteration
                  << "  Q " << diagnostic.qSusy
                  << "  residual " << diagnostic.residual
                  << "  mu " << diagnostic.mu
                  << "  stop1_sq " << diagnostic.stop1Squared
                  << "  stop2_sq " << diagnostic.stop2Squared
                  << "  ODE_steps " << diagnostic.acceptedSteps
                  << "  max_dlogQ " << diagnostic.declaredMaxDeltaLogQ
                  << "  scan_segments " << diagnostic.scanSegments
                  << "  max_observed_dlogQ " << diagnostic.maxObservedDeltaLogQ
                  << "  roots " << diagnostic.rootsFound
                  << "  invalid_boundaries " << diagnostic.invalidBoundaries
                  << "  root_evaluations " << diagnostic.refinementEvaluations << "\n";
    }
    if (r.haveDEW) {
        std::cout << "\nDelta_EW  " << r.deltaEW << "\n";
        for (std::size_t i = 0; i < r.dewContributions.size(); ++i) {
            std::cout << "  " << (i + 1) << ": " << r.dewContributions[i].value
                      << ", " << r.dewContributions[i].label << "\n";
        }
    }
    if (r.haveDHS) {
        std::cout << "\nDelta_HS  " << r.deltaHS << "\n";
        for (const auto & c : r.dhsContributions)
            std::cout << "  " << c.value << ", " << c.label << "\n";
    }
    if (r.haveDBG) {
        std::cout << "\nDelta_BG  " << r.deltaBG << "\n";
        for (const auto & c : r.dbgContributions)
            std::cout << "  " << c.value << ", " << c.label << "\n";
    }
    if (r.haveDSN) {
        std::cout << "\ndelta_SN  " << r.deltaSN
                  << "   dN_vac " << r.snTotalNvac << "\n"
                  << "nF/nD     " << cfg.snNF << "/" << cfg.snND;
        if (randomSN) std::cout << "   random seed " << snSeed;
        std::cout << "\n";
        for (const auto & c : r.dsnContributions)
            std::cout << "  " << c.value << ", " << c.label << "\n";
    }
}

}  // namespace

int main(int argc, char ** argv) {
    natlha_cli::Options options;
    const natlha_cli::ParseStatus parseStatus =
        natlha_cli::parseArgs(argc, argv, options, std::cerr);
    if (parseStatus == natlha_cli::ParseStatus::Help) {
        usage();
        return 0;
    }
    if (parseStatus == natlha_cli::ParseStatus::Error) {
        usage();
        return 1;
    }
    natlha::Config& cfg = options.config;
    const std::string& singlePath = options.singlePath;
    const std::string& batchPath = options.batchPath;
    const std::string& outputPath = options.outputPath;
    const int digits = options.digits;
    const bool randomSN = options.randomSN;
    const bool qSusyAudit = options.qSusyAudit;
    const uint64_t snSeed = options.snSeed;

    const std::string & inputPath = singlePath.empty() ? batchPath : singlePath;
    if (sameExistingFile(inputPath, outputPath)) {
        std::cerr << "error: --out FILE resolves to the input file\n";
        return 1;
    }

    std::ifstream list;
    std::vector<std::string> batchEntries;
    if (!batchPath.empty()) {
        list.open(batchPath);
        if (!list.good()) {
            std::cerr << "error: cannot open batch list: " << batchPath << "\n";
            return 1;
        }
        std::string line;
        while (std::getline(list, line)) {
            if (line.empty() || line[0] == '#') continue;
            if (sameExistingFile(line, outputPath)) {
                std::cerr << "error: --out FILE resolves to a spectrum in the batch list: "
                          << line << "\n";
                return 1;
            }
            batchEntries.push_back(std::move(line));
        }
        if (list.bad()) {
            std::cerr << "error: failed while reading batch list: " << batchPath << "\n";
            return 1;
        }
    }

    CoutRedirect output(outputPath);
    if (!output.good()) {
        std::cerr << "error: cannot open output file: " << outputPath << "\n";
        return 1;
    }

    if (!singlePath.empty()) {
        if (randomSN) {
            uint64_t draw = 0;
            if (!drawIndex(singlePath, draw)) {
                std::cerr << "error: SLHA filename stem is not a global draw index: "
                          << singlePath << "\n";
                return 1;
            }
            randomSNPair(snSeed, draw, cfg.snNF, cfg.snND);
        }
        cfg.slhaPath = singlePath;
        const natlha::Result r = natlha::evaluate(cfg);
        printReport(r, cfg, digits, randomSN, snSeed);
        return r.ok ? 0 : 2;
    }

    printHeader(cfg, randomSN, snSeed, qSusyAudit);
    std::cerr << std::setprecision(17)
              << "# q_susy_max_dlogq " << cfg.qSusyMaxDeltaLogQ << "\n";
    long done = 0, failed = 0;
    for (const std::string& line : batchEntries) {
        natlha::Result r;
        if (randomSN) {
            uint64_t draw = 0;
            if (!drawIndex(line, draw)) {
                cfg.snNF = 0;
                cfg.snND = 0;
                r.error = "SLHA filename stem is not a global draw index";
            } else {
                randomSNPair(snSeed, draw, cfg.snNF, cfg.snND);
                cfg.slhaPath = line;
                r = natlha::evaluate(cfg);
            }
        } else {
            cfg.slhaPath = line;
            r = natlha::evaluate(cfg);
        }
        printRow(line, r, cfg, digits, randomSN, qSusyAudit);
        std::cout.flush();
        ++done;
        if (!r.ok) {
            ++failed;
            std::cerr << "point failed: " << line << ": " << r.error << "\n";
        }
    }
    std::cerr << "# points " << done << ", failed " << failed << "\n";
    return failed ? 2 : 0;
}
