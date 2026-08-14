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
//                  to stdout with a '#'-commented header. Process startup and static
//                  initialisation are paid ONCE for the whole list rather than per point,
//                  which is the point of the mode.
//
// Rows for points that fail are still emitted, with ok=0, so a caller can align input and
// output line by line instead of guessing which point vanished. Do NOT read the measure
// columns on an ok=0 row. evaluate() sets `ok` last (natlha_api.cpp:342) and fills the
// measures before it, Delta_EW at line 302 and Delta_BG at 323, so a throw between them
// leaves a failed row carrying a real Delta_EW beside an untouched Delta_BG, with nothing to
// distinguish "never reached" from "computed as zero". The ok column is the only guard.
// Diagnostics go to stderr so stdout stays parseable.
//
// Exit status: 0 if every requested point succeeded, 1 on usage error, 2 if any point failed.

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
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
        "  --bg-model N         DBG_calc model index, 1-6      (default 1)\n"
        "  --bg-precision N     DBG_calc stencil, 1-3; LOWER IS MORE EXPENSIVE (default 1)\n"
        "  --sn-mode N          DSN_calc mode, 1-4             (default 1)\n"
        "  --sn-nf N            F-term count for delta_SN      (default 0)\n"
        "  --sn-nd N            D-term count for delta_SN      (default 0)\n"
        "  --digits N           printed significant digits     (default 12)\n";
}

/// One row per point, in a fixed column order so the header describes every row.
void printRow(const std::string & path, const natlha::Result & r, const natlha::Config & cfg,
              int digits) {
    std::cout << std::setprecision(digits) << std::scientific
              << (r.ok ? 1 : 0) << " " << r.deltaEW;
    if (cfg.computeDHS) std::cout << " " << r.deltaHS;
    if (cfg.computeDBG) std::cout << " " << r.deltaBG;
    if (cfg.computeDSN) std::cout << " " << r.deltaSN << " " << r.snTotalNvac;
    std::cout << " " << r.qSusy << " " << r.logQGut << " " << r.mZ2 << " " << path << "\n";
}

void printHeader(const natlha::Config & cfg) {
    std::cout << "# ok Delta_EW";
    if (cfg.computeDHS) std::cout << " Delta_HS";
    if (cfg.computeDBG) std::cout << " Delta_BG";
    if (cfg.computeDSN) std::cout << " delta_SN totalNvac";
    std::cout << " Q_SUSY logQ_GUT mZ2 slha_path\n";
}

/// Full report for one point, including the breakdown that makes a surprising total
/// diagnosable: the total alone does not say which sector produced it.
void printReport(const natlha::Result & r, const natlha::Config & cfg, int digits) {
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
        std::cout << "\ndelta_SN  " << r.deltaSN << "   total N_vac " << r.snTotalNvac << "\n";
        for (const auto & c : r.dsnContributions)
            std::cout << "  " << c.value << ", " << c.label << "\n";
    }
}

/// Reads an integer argument, reporting a missing, unparseable or out-of-range value rather
/// than silently defaulting, because a silently ignored flag is worse than an error.
///
/// `lo`/`hi` bound the accepted range inclusively. Validating HERE rather than trusting the
/// calculators matters: an out-of-range DBG precision does not error downstream, it falls
/// through deriv_num_calc's trailing `else` and quietly computes with the cheapest 2-point
/// stencil, so the run would look successful while answering a different question.
///
/// std::stoi alone is not enough: it stops at the first non-digit, so "2junk" parses as 2 and
/// a typo would be silently accepted. `pos` must therefore reach the end of the string.
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

}  // namespace

int main(int argc, char ** argv) {
    natlha::Config cfg;
    std::string singlePath, batchPath;
    int digits = 12;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--slha" && i + 1 < argc)        singlePath = argv[++i];
        else if (a == "--batch" && i + 1 < argc)  batchPath = argv[++i];
        else if (a == "--dhs")                    cfg.computeDHS = true;
        else if (a == "--dbg")                    cfg.computeDBG = true;
        else if (a == "--dsn")                    cfg.computeDSN = true;
        // Ranges match the interactive menus' own bounds checks: terminal_UI.cpp:366 rejects
        // modinp outside 1-6, :382 rejects precinp outside 1-3, :408 rejects DSNcalcSelect
        // outside 1-4, and :424 / :435 reject negative nF / nD. `digits` is capped at 50
        // because high_prec_float is `number<mpfr_float_backend<50>>`, so asking for more
        // prints digits the type does not carry.
        else if (a == "--bg-model") { if (!intArg(argc, argv, i, cfg.bgModelIndex, a.c_str(), 1, 6)) return 1; }
        else if (a == "--bg-precision") { if (!intArg(argc, argv, i, cfg.bgPrecision, a.c_str(), 1, 3)) return 1; }
        else if (a == "--sn-mode") { if (!intArg(argc, argv, i, cfg.snMode, a.c_str(), 1, 4)) return 1; }
        else if (a == "--sn-nf") { if (!intArg(argc, argv, i, cfg.snNF, a.c_str(), 0, 1000000)) return 1; }
        else if (a == "--sn-nd") { if (!intArg(argc, argv, i, cfg.snND, a.c_str(), 0, 1000000)) return 1; }
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

    if (!singlePath.empty()) {
        cfg.slhaPath = singlePath;
        const natlha::Result r = natlha::evaluate(cfg);
        printReport(r, cfg, digits);
        return r.ok ? 0 : 2;
    }

    std::ifstream list(batchPath);
    if (!list.good()) {
        std::cerr << "error: cannot open batch list: " << batchPath << "\n";
        return 1;
    }
    printHeader(cfg);
    std::string line;
    long done = 0, failed = 0;
    while (std::getline(list, line)) {
        if (line.empty() || line[0] == '#') continue;
        cfg.slhaPath = line;
        const natlha::Result r = natlha::evaluate(cfg);
        printRow(line, r, cfg, digits);
        // Flush per row so a long run is streamable: a consumer reading the pipe sees each
        // point as it lands instead of waiting for a full buffer. This is about latency, not
        // durability -- it hands the bytes to the OS, and says nothing about what reaches
        // disk if the process is killed.
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
