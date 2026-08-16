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
// output line by line instead of guessing which point vanished. Do NOT read any other column
// on an ok=0 row. evaluate() computes requested measures in sequence and sets ok only after
// all of them, so an exception can leave a partially populated result. In --sn-random-seed
// mode, a filename that has no numeric draw-index stem is rejected before evaluate() and its
// sn_nF/sn_nD fields are written as 0/0 sentinels. Diagnostics go to stderr so the output
// stream stays parseable.
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
#include <vector>

#include "natlha_api.hpp"

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
        "  --bg-model N         DBG_calc model index, 1-6      (default 1)\n"
        "  --bg-precision N     DBG_calc stencil, 1-3; LOWER IS MORE EXPENSIVE (default 1)\n"
        "  --sn-mode N          DSN_calc mode, 1-3             (default 1)\n"
        "  --sn-nf N            fixed F-term count; required with --dsn unless random\n"
        "  --sn-nd N            fixed D-term count; required with --dsn unless random\n"
        "  --sn-random-seed N   choose one reproducible uniform (nF,nD) pair per draw,\n"
        "                       with nF in 1-10 and nD in 1-5; requires --dsn\n"
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
              int digits, bool randomSN) {
    std::cout << std::setprecision(digits) << std::scientific
              << (r.ok ? 1 : 0) << " " << r.deltaEW;
    if (cfg.computeDHS) std::cout << " " << r.deltaHS;
    if (cfg.computeDBG) std::cout << " " << r.deltaBG;
    if (cfg.computeDSN) {
        std::cout << " " << r.deltaSN << " " << r.snTotalNvac;
        if (randomSN) std::cout << " " << cfg.snNF << " " << cfg.snND;
    }
    std::cout << " " << r.qSusy << " " << r.logQGut << " " << r.mZ2 << " " << path << "\n";
}

void printHeader(const natlha::Config & cfg, bool randomSN, uint64_t snSeed) {
    if (randomSN) std::cout << "# sn_random_seed " << snSeed << "\n";
    std::cout << "# ok Delta_EW";
    if (cfg.computeDHS) std::cout << " Delta_HS";
    if (cfg.computeDBG) std::cout << " Delta_BG";
    if (cfg.computeDSN) {
        std::cout << (cfg.snMode == 3 ? " delta_SN dN_vac" : " Delta_SN N_vac");
        if (randomSN) std::cout << " sn_nF sn_nD";
    }
    std::cout << " Q_SUSY logQ_GUT mZ2 slha_path\n";
}

void printReport(const natlha::Result & r, const natlha::Config & cfg, int digits,
                 bool randomSN, uint64_t snSeed) {
    std::cout << std::setprecision(digits);
    if (!r.ok) {
        std::cout << "FAILED: " << r.error << "\n";
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
              << "iters     ewsb " << r.ewsbIters << "   gut " << r.gutIters << "\n";
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
        std::cout << (cfg.snMode == 3 ? "\ndelta_SN  " : "\nDelta_SN  ") << r.deltaSN
                  << (cfg.snMode == 3 ? "   dN_vac " : "   N_vac ") << r.snTotalNvac << "\n"
                  << "nF/nD     " << cfg.snNF << "/" << cfg.snND;
        if (randomSN) std::cout << "   random seed " << snSeed;
        std::cout << "\n";
        for (const auto & c : r.dsnContributions)
            std::cout << "  " << c.value << ", " << c.label << "\n";
    }
}

bool intArg(int argc, char ** argv, int & i, int & dest, const char * what, int lo, int hi) {
    if (i + 1 >= argc) {
        std::cerr << "error: " << what << " needs a value\n";
        return false;
    }
    const std::string raw = argv[++i];
    int value = 0;
    try {
        std::size_t pos = 0;
        value = std::stoi(raw, &pos);
        if (pos != raw.size()) {
            std::cerr << "error: " << what << " value has trailing characters: " << raw << "\n";
            return false;
        }
    } catch (const std::exception &) {
        std::cerr << "error: " << what << " value is not an integer: " << raw << "\n";
        return false;
    }
    if (value < lo || value > hi) {
        std::cerr << "error: " << what << " must be in [" << lo << ", " << hi
                  << "], got " << value << "\n";
        return false;
    }
    dest = value;
    return true;
}

bool uint64Arg(int argc, char ** argv, int & i, uint64_t & dest, const char * what) {
    if (i + 1 >= argc) {
        std::cerr << "error: " << what << " needs a value\n";
        return false;
    }
    const std::string raw = argv[++i];
    if (raw.empty() || raw[0] == '-') {
        std::cerr << "error: " << what << " value is not an unsigned integer: " << raw << "\n";
        return false;
    }
    try {
        std::size_t pos = 0;
        const unsigned long long value = std::stoull(raw, &pos);
        if (pos != raw.size()) {
            std::cerr << "error: " << what << " value has trailing characters: " << raw << "\n";
            return false;
        }
        dest = static_cast<uint64_t>(value);
        return true;
    } catch (const std::exception &) {
        std::cerr << "error: " << what << " value is not an unsigned integer: " << raw << "\n";
        return false;
    }
}

}  // namespace

int main(int argc, char ** argv) {
    natlha::Config cfg;
    std::string singlePath, batchPath, outputPath;
    int digits = 12;
    bool randomSN = false, fixedNF = false, fixedND = false;
    uint64_t snSeed = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--slha" && i + 1 < argc)        singlePath = argv[++i];
        else if (a == "--batch" && i + 1 < argc)  batchPath = argv[++i];
        else if (a == "--out" && i + 1 < argc)    outputPath = argv[++i];
        else if (a == "--dhs")                    cfg.computeDHS = true;
        else if (a == "--dbg")                    cfg.computeDBG = true;
        else if (a == "--dsn")                    cfg.computeDSN = true;
        else if (a == "--bg-model") { if (!intArg(argc, argv, i, cfg.bgModelIndex, a.c_str(), 1, 6)) return 1; }
        else if (a == "--bg-precision") { if (!intArg(argc, argv, i, cfg.bgPrecision, a.c_str(), 1, 3)) return 1; }
        else if (a == "--sn-mode") { if (!intArg(argc, argv, i, cfg.snMode, a.c_str(), 1, 3)) return 1; }
        else if (a == "--sn-nf") { if (!intArg(argc, argv, i, cfg.snNF, a.c_str(), 0, 1000000)) return 1; fixedNF = true; }
        else if (a == "--sn-nd") { if (!intArg(argc, argv, i, cfg.snND, a.c_str(), 0, 1000000)) return 1; fixedND = true; }
        else if (a == "--sn-random-seed") { if (!uint64Arg(argc, argv, i, snSeed, a.c_str())) return 1; randomSN = true; }
        else if (a == "--digits") { if (!intArg(argc, argv, i, digits, a.c_str(), 1, 50)) return 1; }
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else {
            std::cerr << "error: unrecognised argument: " << a << "\n";
            usage();
            return 1;
        }
    }

    if (singlePath.empty() == batchPath.empty()) {
        std::cerr << "error: give exactly one of --slha or --batch\n";
        usage();
        return 1;
    }
    if (randomSN && !cfg.computeDSN) {
        std::cerr << "error: --sn-random-seed requires --dsn\n";
        return 1;
    }
    if (randomSN && (fixedNF || fixedND)) {
        std::cerr << "error: --sn-random-seed cannot be combined with --sn-nf or --sn-nd\n";
        return 1;
    }
    if (cfg.computeDSN && !randomSN && (!fixedNF || !fixedND)) {
        std::cerr << "error: fixed-mode --dsn requires both --sn-nf and --sn-nd\n";
        return 1;
    }

    const std::string & inputPath = singlePath.empty() ? batchPath : singlePath;
    if (sameExistingFile(inputPath, outputPath)) {
        std::cerr << "error: --out FILE resolves to the input file\n";
        return 1;
    }

    std::ifstream list;
    if (!batchPath.empty()) {
        list.open(batchPath);
        if (!list.good()) {
            std::cerr << "error: cannot open batch list: " << batchPath << "\n";
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

    printHeader(cfg, randomSN, snSeed);
    std::string line;
    long done = 0, failed = 0;
    while (std::getline(list, line)) {
        if (line.empty() || line[0] == '#') continue;
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
        printRow(line, r, cfg, digits, randomSN);
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
