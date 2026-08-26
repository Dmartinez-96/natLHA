// NATLHA_API_HPP
//
// Non-interactive entry point to natLHA's naturalness calculators.
//
// One call takes an SLHA file plus a choice of which measures to compute, and returns the
// results as data. Nothing is read from stdin or written to stdout, which is what makes it
// usable from a batch driver or another program.
//
// All four measures are available through this interface. Which measures a given call
// computes is decided by the `compute*` flags, and each result carries a `have*` flag so a
// caller can tell "not requested" from "requested and came out zero".

#ifndef NATLHA_API_HPP
#define NATLHA_API_HPP

#include <cstddef>
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
    /// DBG_calc's mode: 1 is a fixed 8-point diagnostic, 2 is a fixed 4-point diagnostic,
    /// and 3 is the adaptive two-point production mode. Adaptive work is data-dependent;
    /// mode 3 is the omitted-flag default.
    int bgPrecision = 3;

    /// The non-interactive API exposes only mode 3: lowercase differential delta_SN from
    /// dissertation Eq. 5.21. Modes 1 and 2 are legacy interactive continuation paths and
    /// are rejected here while capital Delta_SN remains deferred.
    int snMode = 3;
    /// Numbers of F-term and D-term contributions for the delta_SN calculation.
    int snNF = 0;
    int snND = 0;

    /// Retained for source compatibility. `evaluate` is always silent; front ends own any
    /// human-readable reporting.
    bool verbose = false;

    /// Maximum spacing between adjacent Q_SUSY residual-classification nodes in log(Q).
    /// 0.1 is the provisional first audit candidate; it is not a frozen production value
    /// until the complete development population agrees between h and h/2.
    double qSusyMaxDeltaLogQ = 0.1;
};

struct QSusyIterationDiagnostic {
    long iteration = 0;
    high_prec_float qSusy = 0;
    high_prec_float residual = 0;
    high_prec_float mu = 0;
    double stop1Squared = 0.0;
    double stop2Squared = 0.0;
    std::size_t acceptedSteps = 0;
    double declaredMaxDeltaLogQ = 0.0;
    std::size_t scanSegments = 0;
    double maxObservedDeltaLogQ = 0.0;
    std::size_t rootsFound = 0;
    std::size_t invalidBoundaries = 0;
    std::size_t refinementEvaluations = 0;
};

struct QSusySearchDiagnostic {
    /// One entry is appended for every bounded root-search attempt. `scanComplete` is true
    /// when the wrapped search returned normally or reported measured counts through the
    /// typed root-search failure; otherwise search progress is unknown and the count fields
    /// retain their zero initialisers. `accepted` means the wrapped search returned normally.
    /// Production `findQSusy` returns only after a complete scan finds exactly one
    /// positive-stop root and no non-finite numerical boundary, so an accepted diagnostic's
    /// `nonFiniteBoundaries == 0` is guaranteed by that success gate rather than copied from
    /// a returned counter; typed rejections preserve the measured non-finite count.
    std::size_t ordinal = 0;
    bool scanComplete = false;
    bool accepted = false;
    double logScale = 0.0;
    std::size_t rootsFound = 0;
    std::size_t invalidBoundaries = 0;
    std::size_t nonFiniteBoundaries = 0;
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

    /// Q_SUSY is the one positive-stop sign-changing or exact sampled root of
    /// log(Q) = (log(m_stop1^2) + log(m_stop2^2)) / 4 on the bounded dense-output
    /// trajectory at the declared maximum log(Q) scan spacing. This operational result
    /// does not claim detection of tangent roots between classification nodes.
    high_prec_float qSusy = 0;
    /// The scale where g1 = g2, found by iteration. This is a LOG scale, matching what
    /// DBG_calc and DSN_calc expect.
    high_prec_float logQGut = 0;
    /// m_Z^2 computed from the EWSB relation after the mu re-solve, whose target is 91.1876^2.
    high_prec_float mZ2 = 0;
    /// m_Z^2 as returned by solveMZ2(), a separate evaluation rather than a copy of the above.
    /// Both are exposed because the calculators do not take the same one: DBG_calc receives
    /// the relation value and DSN_calc receives this solver value, matching how the
    /// interactive path passes `currentmZ2` and the structured solver value respectively.
    high_prec_float mZ2FromSolver = 0;

    std::vector<QSusyIterationDiagnostic> qSusyDiagnostics;
    /// Root-search-level telemetry retained even when a later EWSB, GUT, or label stage
    /// invalidates the row. This is the structured source used by the opt-in CLI freeze
    /// audit; it is separate from the post-mu iteration diagnostics above.
    std::vector<QSusySearchDiagnostic> qSusySearchDiagnostics;
    high_prec_float qSusyResidual = 0;
    double qSusyStop1Squared = 0.0;
    double qSusyStop2Squared = 0.0;

    /// The 44-entry running state at the jointly converged Q_SUSY, after the mu solve and
    /// after b = B*mu is filled in. The slot numbering is 0-based: positions 0-5 hold
    /// sqrt(5/3)*g', g_2, g_s, M1, M2, M3, while index 6 is mu, index 42 is b, and index 43
    /// is the running tan(beta).
    /// The mu at index 6 need not equal the value in the input HMIX block because natLHA
    /// re-derives it from its EWSB condition.
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

    /// Iterations taken by the joint Q_SUSY/mu solve and the GUT-scale solve, reported so
    /// that a slow point can be attributed to one of them instead of guessed at.
    ///
    /// `qSusyIters` counts outer joint iterations and `ewsbIters` is the cumulative number
    /// of inner mu iterations across them; both loops fail closed at 100 iterations.
    /// `gutIters` counts passes that each call `solveODEs` over the full run from Q_SUSY to
    /// a trial GUT scale. The GUT loop fails closed on non-finite updates and at 100 passes.
    ///
    /// These count iterations, not seconds. A large count is evidence about WHERE time went
    /// only together with a timing measurement, since neither per-pass cost is recorded here.
    long ewsbIters = 0;
    long qSusyIters = 0;
    long gutIters = 0;

    bool haveDEW = false, haveDHS = false, haveDBG = false, haveDSN = false;

    /// The Delta_EW, Delta_HS and Delta_BG headlines retain the sign of the contribution
    /// largest in absolute value. For Delta_BG, exact equal-magnitude ties select the lowest
    /// fixed direction ordinal. This signed-dominant natLHA convention deliberately differs
    /// from the conventional non-negative max_i |C_i| definition. The lowercase
    /// stringy-naturalness headline is log10(1 / dN_vac) per dissertation Eq. 5.21.
    high_prec_float deltaEW = 0, deltaHS = 0, deltaBG = 0, deltaSN = 0;

    /// Every contribution with its label, ordered as the underlying calculator returns it.
    /// Kept because the breakdown identifies which sector produced a surprising headline.
    std::vector<LabeledValue> dewContributions;
    std::vector<LabeledValueHS> dhsContributions;
    std::vector<LabeledValueBG> dbgContributions;
    /// Node, adaptive-window, propagated-root-width, tie, and headline-stability diagnostics
    /// for Delta_BG. On a failed requested Delta_BG row, the completed prefix remains here for
    /// diagnosis, but `haveDBG` is false and `dbgContributions` is empty.
    std::vector<BGDirectionDiagnostic> dbgDiagnostics;
    BGHeadlineDiagnostic dbgHeadline;
    std::vector<DSNLabeledValue> dsnContributions;
    /// Summed differential vacuum density dN_vac.
    high_prec_float snTotalNvac = 0;
};

namespace detail {

/// Invalidate every label in a requested multi-label row while retaining setup state and
/// Delta_BG diagnostics. Production failure paths and the contract test share this function.
void failLabelRow(Result& result, std::string error);

}  // namespace detail

/// Run the pipeline: read the SLHA, locate Q_SUSY, re-solve EWSB for mu, iterate to the
/// g1 = g2 scale, then evaluate whichever measures were requested.
///
/// Never throws: failures are reported through Result::ok and Result::error.
Result evaluate(const Config & cfg);

}  // namespace natlha

#endif  // NATLHA_API_HPP
