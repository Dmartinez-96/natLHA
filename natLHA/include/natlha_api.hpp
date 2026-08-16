// NATLHA_API_HPP
//
// Non-interactive entry point to natLHA's naturalness calculators.
//
// One call takes an SLHA file plus a choice of which measures to compute, and returns the
// results as data. Nothing is read from stdin and, unless `verbose` is set, nothing is
// written to stdout, which is what makes it usable from a batch driver or another program.
//
// The four measures are all declared here from the outset, including the ones not wired up
// yet, so that adding one later does not change this interface. Which measures a given call
// actually computes is decided by the `compute*` flags, and each result carries a `have*`
// flag so a caller can tell "not requested" from "requested and came out zero".

#ifndef NATLHA_API_HPP
#define NATLHA_API_HPP

#include <string>
#include <vector>

#include "DEW_calc.hpp"   // LabeledValue,      high_prec_float
#include "DHS_calc.hpp"   // LabeledValueHS
#include "DBG_calc.hpp"   // LabeledValueBG
#include "DSN_calc.hpp"   // DSNLabeledValue

namespace natlha {

/// What to compute, and how.
struct Config {
    /// Path to an SLHA spectrum file. Both conventions are accepted: the SLHA2 matrix
    /// blocks MSQ2/MSU2/MSD2/MSL2/MSE2, and the SLHA1 style where the sfermion masses live
    /// in MSOFT 41-49 and 31-36. The reader falls back to the latter when the former are
    /// absent, so SOFTSUSY's default output works unmodified.
    std::string slhaPath;

    bool computeDEW = true;
    bool computeDHS = false;
    bool computeDBG = false;
    bool computeDSN = false;

    /// Ask for `Result::mZ2FromSolver` even when no measure needs it.
    ///
    /// The solve behind that field is skipped unless something wants it, so a caller that
    /// reads it without computing delta_SN must set this. The interactive front end is exactly
    /// such a caller: it leaves every compute* flag false and runs the calculators itself, but
    /// still reads mZ2FromSolver to pass into its own DSN_calc.
    bool wantMZ2FromSolver = false;

    /// DBG_calc's `modselno`, the model whose parameters the derivatives are taken with
    /// respect to. Valid range 1-6, matching the interactive menu.
    int bgModelIndex = 1;
    /// DBG_calc's `precselno`, 1-3. NOTE the ordering is INVERTED relative to what the name
    /// suggests: lower is MORE expensive. `deriv_num_calc` (DBG_calc.cpp:50-68) branches on
    /// it and natLHA's own comments there name each stencil:
    ///     precselno == 1  ->  "8-point derivative calculation",  reads mzsq_values[0..7]
    ///     precselno == 2  ->  "4-point derivative calculation",  reads mzsq_values[0..3]
    ///     anything else   ->  "2-point derivative calculation (default)", reads [0..1]
    /// The stencil width IS the cost, since each mzsq_values entry is one m_Z^2 evaluation,
    /// so precselno = 1 costs four times the trailing case per direction.
    int bgPrecision = 1;

    /// DSN_calc's mode, 1-3. Mode 3 is the lowercase differential delta_SN defined by
    /// dissertation Eq. 5.21. Capital Delta_SN via numerical continuation is deferred and
    /// is not reachable through this struct.
    int snMode = 1;
    /// Numbers of F-term and D-term contributions for the delta_SN calculation.
    int snNF = 0;
    int snND = 0;

    /// When false, this call prints NOTHING to stdout. Required for batch use, where a
    /// per-point progress dump would swamp the output and where the caller is parsing
    /// stdout. Set it true to get the running commentary a human watching a single point
    /// wants.
    bool verbose = false;
};

/// Everything the pipeline established, not just the headline numbers.
///
/// The intermediate quantities are returned deliberately: a caller assembling a
/// mixed-renormalization-scale feature set needs qSusy and qGut, and a caller checking the
/// pipeline against a generator needs the tuned weak-scale state to compare mu against.
struct Result {
    /// False if the SLHA could not be read or the pipeline threw; `error` says why. Check
    /// this before reading anything else.
    bool ok = false;
    std::string error;

    /// Q_SUSY = sqrt(mst1 * mst2), located by running the spectrum from its own scale.
    high_prec_float qSusy = 0;
    /// The scale where g1 = g2, found by iteration. This is a LOG scale, matching what
    /// DBG_calc and DSN_calc expect.
    high_prec_float logQGut = 0;
    /// m_Z^2 computed from the EWSB relation after the mu re-solve, whose target is 91.1876^2.
    high_prec_float mZ2 = 0;
    /// m_Z^2 as returned by getmZ2(), a separate evaluation rather than a copy of the above.
    /// Both are exposed because the calculators do not take the same one: DBG_calc receives
    /// the relation value and DSN_calc receives this solver value, matching how the
    /// interactive path passes `currentmZ2` and `getmZ2_value` respectively.
    high_prec_float mZ2FromSolver = 0;

    /// The 44-entry running state at Q_SUSY, AFTER the mu convergence and after b = B*mu is
    /// filled in. The slot numbering follows the initializer in terminal_UI.cpp:666-671,
    /// which is 0-based: positions 0-5 hold sqrt(5/3)*g', g_2, g_s, M1, M2, M3, so
    ///   index  6 = mu    -- the mu loop writes the tuned value there (terminal_UI.cpp:704)
    ///   index 42 = b     -- b = B*mu, assigned at terminal_UI.cpp:718
    ///   index 43 = tanb  -- read back at terminal_UI.cpp:693
    /// The mu at index 6 need not equal the mu the generator wrote into HMIX, because it is
    /// re-derived from the EWSB condition rather than trusted. How far it moves is a
    /// property of the point and of the two codes' loop orders, and ONE case has been
    /// measured: on the arXiv:2111.03096 Table-1 benchmark against a SOFTSUSY 4.1.23
    /// spectrum, an input mu = 200.024 came back near 245 GeV, about 22.6 percent higher.
    /// That single point says nothing about the typical size of the shift.
    std::vector<high_prec_float> weakBCs;
    /// The 44-entry state at the converged GUT scale, same slot numbering.
    std::vector<high_prec_float> gutBCs;
    /// Sigma_u and Sigma_d, the one-loop tadpole corrections at Q_SUSY, in that order.
    std::vector<high_prec_float> radCorrs;

    /// Whether `mZ2FromSolver` was computed at all, and whether its solve converged.
    ///
    /// `haveMZ2FromSolver` is false when the solve was skipped because nothing asked for it,
    /// in which case `mZ2FromSolver` keeps its zero initialiser and means nothing. Check it
    /// before reading that field: a skipped solve and a solve that genuinely returned zero
    /// are otherwise the same value.
    ///
    /// `mZ2SolverConverged` reports whether the solver met its residual tolerance. It is
    /// meaningful only when `haveMZ2FromSolver` is true.
    bool haveMZ2FromSolver = false;
    bool mZ2SolverConverged = false;

    /// Iterations taken by the two fixed-point loops in `evaluate`, reported so that a slow
    /// point can be attributed to one of them instead of guessed at.
    ///
    /// Both loops run to a tolerance with NO iteration cap, so either can in principle spin
    /// for a long time on an awkward point. They are not equally expensive per pass:
    /// `gutIters` counts passes that each call `solveODEs` over the full run from Q_SUSY to
    /// the trial GUT scale, while `ewsbIters` counts passes that each call `radcorr_calc` at
    /// a single scale.
    ///
    /// These count iterations, not seconds. A large count is evidence about WHERE time went
    /// only together with a timing measurement, since neither per-pass cost is recorded here.
    long ewsbIters = 0;
    long gutIters = 0;

    bool haveDEW = false, haveDHS = false, haveDBG = false, haveDSN = false;

    /// The Delta_EW, Delta_HS and Delta_BG headlines retain the sign of the contribution
    /// largest in ABSOLUTE value. Their calculators sort with an
    /// `abs(a.value) < abs(b.value)` comparator and then reverse, so element [0] is the
    /// largest by magnitude. The stringy-naturalness headline is mode-dependent: modes 1/2
    /// return the reciprocal of the summed N_vac contributions, and mode 3 returns
    /// log10(1 / dN_vac) per dissertation Eq. 5.21.
    high_prec_float deltaEW = 0, deltaHS = 0, deltaBG = 0, deltaSN = 0;

    /// Every contribution with its label, ordered as the underlying calculator returns it.
    /// Kept because the breakdown identifies which sector produced a surprising headline.
    std::vector<LabeledValue> dewContributions;
    std::vector<LabeledValueHS> dhsContributions;
    std::vector<LabeledValueBG> dbgContributions;
    std::vector<DSNLabeledValue> dsnContributions;
    /// Summed vacuum density (N_vac, or differential dN_vac in mode 3).
    high_prec_float snTotalNvac = 0;
};

/// Run the pipeline: read the SLHA, locate Q_SUSY, re-solve EWSB for mu, iterate to the
/// g1 = g2 scale, then evaluate whichever measures were requested.
///
/// Never throws: failures are reported through Result::ok and Result::error.
Result evaluate(const Config & cfg);

}  // namespace natlha

#endif  // NATLHA_API_HPP
