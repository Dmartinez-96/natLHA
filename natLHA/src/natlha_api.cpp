// Non-interactive implementation of the natLHA pipeline. See include/natlha_api.hpp for the
// contract this satisfies.
//
// The sequence mirrors what the interactive program does, in this order:
//   1. read the SLHA and assemble the 44-entry weak-scale state at the file's own scale
//   2. run to Q = 1e12, then locate Q_SUSY = sqrt(mst1 * mst2) with the stop-finder
//   3. run the state from the SLHA scale to Q_SUSY
//   4. re-solve EWSB there: iterate mu until m_Z = 91.1876, then fill b = B*mu
//   5. iterate to the scale where g1 = g2 to get the GUT-scale state
//   6. evaluate whichever measures were requested
//
// Steps 1-5 are pure setup and are shared by every measure, which is the whole reason this
// function exists: a batch caller pays for them once per point regardless of how many labels
// it asks for.

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <boost/multiprecision/mpfr.hpp>

#include "natlha_api.hpp"
#include "MSSM_RGE_solver.hpp"
#include "MSSM_RGE_solver_with_stopfinder.hpp"
#include "mZ_numsolver.hpp"
#include "radcorr_calc.hpp"
#include "shared_helpers.hpp"
#include "slhaea.h"

namespace natlha {

namespace {

/// m_Z used throughout, matching the value the interactive path hard-codes.
const double kMZ = 91.1876;

}  // namespace

Result evaluate(const Config & cfg) {
    Result out;
    try {
        std::ifstream ifs(cfg.slhaPath);
        if (!ifs.good()) {
            out.error = "cannot open SLHA file: " + cfg.slhaPath;
            return out;
        }
        SLHAea::Coll input(ifs);

        // Both accessors swallow a missing entry and hand back `defaultValue`, which is what
        // makes the block-convention fallbacks below work: an absent block reads as zero.
        auto vecVal = [&](const std::string & block, int i, double defaultValue = 0.0) -> double {
            try {
                return SLHAea::to<double>(input.at(block).at(std::to_string(i)).at(1));
            } catch (const std::exception &) {
                return defaultValue;
            }
        };
        auto matVal = [&](const std::string & block, int i, int j, double defaultValue = 0.0) -> double {
            try {
                return SLHAea::to<double>(input.at(block).at(i, j).at(2));
            } catch (const std::exception &) {
                return defaultValue;
            }
        };

        // ---- Higgs sector -------------------------------------------------------------
        high_prec_float tanb = high_prec_float(vecVal("HMIX", 2));
        high_prec_float beta = atan(tanb);
        high_prec_float muQ = high_prec_float(vecVal("HMIX", 1));

        // ---- Yukawas ------------------------------------------------------------------
        // Generations 1 and 2 are only estimated from the third when the SLHA omits them.
        // A file carrying full 3x3 YU/YD/YE never reaches these fallbacks, which is why a
        // generator should be asked to emit them rather than relying on these ratios.
        high_prec_float y_t = high_prec_float(matVal("YU", 3, 3));
        high_prec_float y_c = high_prec_float(matVal("YU", 2, 2));
        if (y_c == 0.0) y_c = high_prec_float(0.003882759826930082) * y_t;
        high_prec_float y_u = high_prec_float(matVal("YU", 1, 1));
        if (y_u == 0.0) y_u = high_prec_float(7.779613278615955e-6) * y_t;
        high_prec_float y_b = high_prec_float(matVal("YD", 3, 3));
        high_prec_float y_s = high_prec_float(matVal("YD", 2, 2));
        if (y_s == 0.0) y_s = high_prec_float(0.0206648802754076) * y_b;
        high_prec_float y_d = high_prec_float(matVal("YD", 1, 1));
        if (y_d == 0.0) y_d = high_prec_float(0.0010117174290779725) * y_b;
        high_prec_float y_tau = high_prec_float(matVal("YE", 3, 3));
        high_prec_float y_mu = high_prec_float(matVal("YE", 2, 2));
        if (y_mu == 0.0) y_mu = high_prec_float(0.05792142442492775) * y_tau;
        high_prec_float y_e = high_prec_float(matVal("YE", 1, 1));
        if (y_e == 0.0) y_e = high_prec_float(0.0002801267571260388) * y_tau;

        // ---- Gauge couplings ----------------------------------------------------------
        high_prec_float g_pr = high_prec_float(vecVal("GAUGE", 1));
        high_prec_float g_2 = high_prec_float(vecVal("GAUGE", 2));
        high_prec_float g_s = high_prec_float(vecVal("GAUGE", 3));

        // ---- Trilinears ---------------------------------------------------------------
        // Two conventions in the wild. TU/TD/TE hold the soft term a_ij directly, so the
        // multiplier starts at 1. AU/AD/AE hold the RATIO A_ij, so the multiplier starts at
        // the corresponding Yukawa and the product is again a_ij. Which one is present is
        // decided by whether the third-generation TU/TD/TE entries are all absent.
        high_prec_float a_t, a_c, a_u, a_b, a_s, a_d, a_tau, a_mu, a_e;
        const bool haveT = !(matVal("TU", 3, 3) == 0.0 && matVal("TD", 3, 3) == 0.0 &&
                             matVal("TE", 3, 3) == 0.0);
        std::string uBlock, dBlock, eBlock;
        if (haveT) {
            uBlock = "TU"; dBlock = "TD"; eBlock = "TE";
            a_t = a_c = a_u = a_b = a_s = a_d = a_tau = a_mu = a_e = high_prec_float(1.0);
        } else {
            uBlock = "AU"; dBlock = "AD"; eBlock = "AE";
            a_t = y_t; a_c = y_c; a_u = y_u;
            a_b = y_b; a_s = y_s; a_d = y_d;
            a_tau = y_tau; a_mu = y_mu; a_e = y_e;
        }
        a_t   *= high_prec_float(matVal(uBlock, 3, 3));
        a_c   *= high_prec_float(matVal(uBlock, 2, 2));
        a_u   *= high_prec_float(matVal(uBlock, 1, 1));
        a_b   *= high_prec_float(matVal(dBlock, 3, 3));
        a_s   *= high_prec_float(matVal(dBlock, 2, 2));
        a_d   *= high_prec_float(matVal(dBlock, 1, 1));
        a_tau *= high_prec_float(matVal(eBlock, 3, 3));
        a_mu  *= high_prec_float(matVal(eBlock, 2, 2));
        a_e   *= high_prec_float(matVal(eBlock, 1, 1));

        // ---- Gauginos and soft Higgs masses -------------------------------------------
        high_prec_float my_M1 = high_prec_float(vecVal("MSOFT", 1));
        high_prec_float my_M2 = high_prec_float(vecVal("MSOFT", 2));
        high_prec_float my_M3 = high_prec_float(vecVal("MSOFT", 3));
        high_prec_float mHusq = high_prec_float(vecVal("MSOFT", 22));
        high_prec_float mHdsq = high_prec_float(vecVal("MSOFT", 21));

        // ---- Sfermion soft masses -----------------------------------------------------
        // SLHA2 gives squared masses in MSQ2/MSU2/MSD2/MSL2/MSE2; SLHA1 gives MASSES in
        // MSOFT 41-49 and 31-36, which must therefore be squared. SOFTSUSY's default output
        // is the latter, so this fallback is the common path, not an edge case.
        high_prec_float mQ1sq, mQ2sq, mQ3sq, mL1sq, mL2sq, mL3sq;
        high_prec_float mU1sq, mU2sq, mU3sq, mD1sq, mD2sq, mD3sq, mE1sq, mE2sq, mE3sq;
        const bool haveSlha2Soft = !(matVal("MSQ2", 3, 3) == 0.0 && matVal("MSU2", 3, 3) == 0.0 &&
                                     matVal("MSE2", 3, 3) == 0.0);
        if (!haveSlha2Soft) {
            mQ3sq = pow(high_prec_float(vecVal("MSOFT", 43)), 2.0);
            mQ2sq = pow(high_prec_float(vecVal("MSOFT", 42)), 2.0);
            mQ1sq = pow(high_prec_float(vecVal("MSOFT", 41)), 2.0);
            mL3sq = pow(high_prec_float(vecVal("MSOFT", 33)), 2.0);
            mL2sq = pow(high_prec_float(vecVal("MSOFT", 32)), 2.0);
            mL1sq = pow(high_prec_float(vecVal("MSOFT", 31)), 2.0);
            mU3sq = pow(high_prec_float(vecVal("MSOFT", 46)), 2.0);
            mU2sq = pow(high_prec_float(vecVal("MSOFT", 45)), 2.0);
            mU1sq = pow(high_prec_float(vecVal("MSOFT", 44)), 2.0);
            mD3sq = pow(high_prec_float(vecVal("MSOFT", 49)), 2.0);
            mD2sq = pow(high_prec_float(vecVal("MSOFT", 48)), 2.0);
            mD1sq = pow(high_prec_float(vecVal("MSOFT", 47)), 2.0);
            mE3sq = pow(high_prec_float(vecVal("MSOFT", 36)), 2.0);
            mE2sq = pow(high_prec_float(vecVal("MSOFT", 35)), 2.0);
            mE1sq = pow(high_prec_float(vecVal("MSOFT", 34)), 2.0);
        } else {
            mQ3sq = high_prec_float(matVal("MSQ2", 3, 3));
            mQ2sq = high_prec_float(matVal("MSQ2", 2, 2));
            mQ1sq = high_prec_float(matVal("MSQ2", 1, 1));
            mL3sq = high_prec_float(matVal("MSL2", 3, 3));
            mL2sq = high_prec_float(matVal("MSL2", 2, 2));
            mL1sq = high_prec_float(matVal("MSL2", 1, 1));
            mU3sq = high_prec_float(matVal("MSU2", 3, 3));
            mU2sq = high_prec_float(matVal("MSU2", 2, 2));
            mU1sq = high_prec_float(matVal("MSU2", 1, 1));
            mD3sq = high_prec_float(matVal("MSD2", 3, 3));
            mD2sq = high_prec_float(matVal("MSD2", 2, 2));
            mD1sq = high_prec_float(matVal("MSD2", 1, 1));
            mE3sq = high_prec_float(matVal("MSE2", 3, 3));
            mE2sq = high_prec_float(matVal("MSE2", 2, 2));
            mE1sq = high_prec_float(matVal("MSE2", 1, 1));
        }

        const double slhaScale = getRenormalizationScale(input, "GAUGE");

        // ---- Assemble the 44-entry state ----------------------------------------------
        // Slot order is load-bearing and shared with every calculator: g1 is GUT-normalised
        // here by the sqrt(5/3), because the GAUGE block carries the unnormalised g'.
        std::vector<high_prec_float> slhaBCs = {
            sqrt(5.0 / 3.0) * g_pr, g_2, g_s, my_M1, my_M2, my_M3,
            muQ, y_t, y_c, y_u, y_b, y_s, y_d, y_tau, y_mu, y_e,
            a_t, a_c, a_u, a_b, a_s, a_d, a_tau, a_mu, a_e,
            mHusq, mHdsq, mQ1sq, mQ2sq, mQ3sq, mL1sq, mL2sq,
            mL3sq, mU1sq, mU2sq, mU3sq, mD1sq, mD2sq, mD3sq,
            mE1sq, mE2sq, mE3sq, 0.0, tanb};
        std::vector<double> slhaBCsDbl;
        slhaBCsDbl.reserve(slhaBCs.size());
        for (const auto & v : slhaBCs) slhaBCsDbl.push_back(double(v));

        // ---- Locate Q_SUSY ------------------------------------------------------------
        // Run well above the SUSY scale first, then walk back down with the stop-finder so
        // Q_SUSY = sqrt(mst1 * mst2) is found from running masses rather than assumed. This
        // is what makes the measures independent of the scale the input file happened to be
        // written at.
        std::vector<double> upRun = solveODEs(slhaBCsDbl, log(slhaScale), log(1.0e12),
                                              std::copysign(1.0e-6, log(1.0e12 / slhaScale)));
        double tTarget = log(250.0);
        std::vector<RGEStruct> susyScale = solveODEstoMSUSY(upRun, log(1.0e12), -1.0e-6,
                                                            tTarget, kMZ * kMZ);
        const high_prec_float qSusy = exp(high_prec_float(susyScale[0].SUSYscale_eval));
        const double qSusyDbl = double(qSusy);
        out.qSusy = qSusy;

        std::vector<double> weakDbl = solveODEs(slhaBCsDbl, log(slhaScale), log(qSusyDbl),
                                                copysign(1.0e-6, qSusyDbl - slhaScale));
        std::vector<high_prec_float> weakBCs;
        weakBCs.reserve(weakDbl.size());
        for (const auto & v : weakDbl) weakBCs.push_back(high_prec_float(v));

        std::vector<high_prec_float> radCorrs =
            radcorr_calc(weakBCs, qSusy, high_prec_float(kMZ * kMZ));

        // ---- Re-solve EWSB for mu -----------------------------------------------------
        // mu is an OUTPUT here, not the value the generator wrote: it is iterated until the
        // EWSB condition reproduces m_Z, with Sigma_u and Sigma_d recomputed at each step
        // because they themselves depend on mu. tanb and the soft Higgs masses are read from
        // the run-down state and held fixed; only mu and the tadpoles move.
        tanb = weakBCs[43];
        mHdsq = weakBCs[26];
        mHusq = weakBCs[25];
        muQ = weakBCs[6];
        const high_prec_float lsqtol = high_prec_float(1.0) / high_prec_float(1000000000000.0);
        high_prec_float currIterLsq = high_prec_float(100.0);
        high_prec_float muQsq = muQ * muQ;
        high_prec_float newMuQsq = muQsq;
        while (currIterLsq > lsqtol) {
            ++out.ewsbIters;
            newMuQsq = ((mHdsq + radCorrs[1] - ((mHusq + radCorrs[0]) * pow(tanb, 2.0)))
                        / (pow(tanb, 2.0) - 1.0)) - (kMZ * kMZ / 2.0);
            weakBCs[6] = copysign(sqrt(abs(newMuQsq)), muQ);
            radCorrs = radcorr_calc(weakBCs, qSusy, high_prec_float(kMZ * kMZ));
            currIterLsq = pow(muQsq - newMuQsq, 2.0);
            muQsq = newMuQsq;
        }
        const high_prec_float currentMZ2 =
            (2.0 * ((mHdsq + radCorrs[1] - ((mHusq + radCorrs[0]) * pow(tanb, 2.0)))
                    / (pow(tanb, 2.0) - 1.0))) - (2.0 * muQsq);
        // Solve for m_Z^2 only when something downstream reads it. delta_SN is the one measure
        // that consumes it, and `wantMZ2FromSolver` covers callers that read the field without
        // asking for a measure. With all four measures requested this condition is true, so
        // the skip matters to a Delta_EW-only pass rather than to a full production run.
        //
        // Wall time printed when NATLHA_ODE_TRACE is set, to stderr, so stdout is unchanged.
        static const bool apiTrace = [] {
            const char * e = std::getenv("NATLHA_ODE_TRACE");
            return e != nullptr && *e != '\0';
        }();
        high_prec_float getmZ2Value = 0;
        if (cfg.computeDSN || cfg.wantMZ2FromSolver) {
            const auto tMZ0 = std::chrono::steady_clock::now();
            bool mz2Converged = false;
            getmZ2Value = getmZ2(weakBCs, qSusy, kMZ * kMZ, &mz2Converged);
            out.haveMZ2FromSolver = true;
            out.mZ2SolverConverged = mz2Converged;
            if (apiTrace) {
                std::cerr << "# api_trace getmZ2_seconds "
                          << std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - tMZ0).count()
                          << "  converged " << (mz2Converged ? 1 : 0) << "\n";
            }
        }

        // b = B*mu at the SUSY scale, from the tree relation with the loop-corrected soft
        // masses substituted. Slot 42 holds b itself, not B; consumers divide by mu.
        weakBCs[42] = sin(2.0 * beta) *
                      (mHusq + radCorrs[0] + mHdsq + radCorrs[1] + (2.0 * muQsq)) / 2.0;

        out.mZ2 = currentMZ2;
        out.mZ2FromSolver = getmZ2Value;
        out.weakBCs = weakBCs;
        out.radCorrs = radCorrs;

        // ---- Iterate to the g1 = g2 scale ---------------------------------------------
        // REFRESH the double-precision copy from the TUNED weak-scale state first. weakDbl
        // was produced by the run down to Q_SUSY, BEFORE the EWSB re-solve, so it still holds
        // the untuned mu in slot 6 and a zero in slot 42 where b belongs. Evolving from it
        // carries boundary conditions upward that no longer describe the weak-scale point just
        // solved for, and the GUT-scale consumers below inherit that: DHS_calc reads gutBCs
        // [26]/[25]/[6], DBG_calc takes both gutBCs and currIterQGut, and DSN_calc takes
        // currIterQGut. DEW_calc is unaffected -- its only arguments are weakBCs and qSusy.
        for (std::size_t i = 0; i < weakDbl.size() && i < weakBCs.size(); ++i) {
            weakDbl[i] = double(weakBCs[i]);
        }

        currIterLsq = high_prec_float(100.0);
        std::vector<double> gutDbl = solveODEs(weakDbl, log(qSusyDbl), log(3.0e16), 1.0e-6);
        std::vector<high_prec_float> gutBCs;
        gutBCs.reserve(gutDbl.size());
        for (const auto & v : gutDbl) gutBCs.push_back(high_prec_float(v));

        std::vector<high_prec_float> betaG1G2 =
            beta_g1g2(gutBCs[0], gutBCs[1], gutBCs[2], gutBCs[7], gutBCs[8], gutBCs[9],
                      gutBCs[10], gutBCs[11], gutBCs[12], gutBCs[13], gutBCs[14], gutBCs[15]);
        high_prec_float currIterQGut =
            log(high_prec_float(3.0e16) * exp((gutBCs[1] - gutBCs[0]) / (betaG1G2[0] - betaG1G2[1])));
        double currIterQGutDbl = double(currIterQGut);
        while (currIterLsq > lsqtol) {
            ++out.gutIters;
            gutDbl = solveODEs(weakDbl, log(qSusyDbl), currIterQGutDbl, 1.0e-6);
            for (std::size_t i = 0; i < gutBCs.size() && i < gutDbl.size(); ++i) {
                gutBCs[i] = high_prec_float(gutDbl[i]);
            }
            // Re-evaluated at the current trial scale: the extrapolation divides by
            // (beta_g1 - beta_g2), and those slopes move with the scale because the gauge
            // couplings and Yukawas feeding them do.
            betaG1G2 = beta_g1g2(gutBCs[0], gutBCs[1], gutBCs[2], gutBCs[7], gutBCs[8], gutBCs[9],
                                 gutBCs[10], gutBCs[11], gutBCs[12], gutBCs[13], gutBCs[14],
                                 gutBCs[15]);
            const high_prec_float newQGut =
                log(exp(currIterQGut) * exp((gutBCs[1] - gutBCs[0]) / (betaG1G2[0] - betaG1G2[1])));
            currIterLsq = pow(high_prec_float(1.0) - (newQGut / currIterQGut), high_prec_float(2.0));
            currIterQGut = newQGut;
            currIterQGutDbl = double(currIterQGut);
        }
        out.logQGut = currIterQGut;
        out.gutBCs = gutBCs;

        // ---- The measures -------------------------------------------------------------
        // Element [0] of each list is the largest contribution by absolute value; see the
        // header for why that holds for all four calculators.
        const high_prec_float logQSusy = log(qSusy);

        if (cfg.computeDEW) {
            out.dewContributions = DEW_calc(weakBCs, qSusy);
            if (!out.dewContributions.empty()) {
                out.deltaEW = out.dewContributions[0].value;
                out.haveDEW = true;
            }
        }
        if (cfg.computeDHS) {
            out.dhsContributions = DHS_calc(
                gutBCs[26], weakBCs[26] - gutBCs[26],
                gutBCs[25], weakBCs[25] - gutBCs[25],
                pow(gutBCs[6], 2.0), pow(weakBCs[6], 2.0) - pow(gutBCs[6], 2.0),
                kMZ * kMZ, weakBCs[43] * weakBCs[43], radCorrs[0], radCorrs[1]);
            if (!out.dhsContributions.empty()) {
                out.deltaHS = out.dhsContributions[0].value;
                out.haveDHS = true;
            }
        }
        if (cfg.computeDBG) {
            int modsel = cfg.bgModelIndex;
            int precsel = cfg.bgPrecision;
            out.dbgContributions = DBG_calc(modsel, precsel, currIterQGut, logQSusy,
                                            tanb, gutBCs, currentMZ2);
            if (!out.dbgContributions.empty()) {
                out.deltaBG = out.dbgContributions[0].value;
                out.haveDBG = true;
            }
        }
        if (cfg.computeDSN) {
            high_prec_float mz2ForSn = getmZ2Value;
            high_prec_float logQSusyForSn = logQSusy;
            high_prec_float logQGutForSn = currIterQGut;
            int nF = cfg.snNF;
            int nD = cfg.snND;
            out.dsnContributions = DSN_calc(cfg.snMode, weakBCs, mz2ForSn, logQSusyForSn,
                                            logQGutForSn, nF, nD);
            out.snTotalNvac = 0.0;
            for (const auto & item : out.dsnContributions) out.snTotalNvac += item.value;
            if (out.dsnContributions.empty() || out.snTotalNvac <= 0.0
                    || isnan(out.snTotalNvac) || isinf(out.snTotalNvac)) {
                out.error = "DSN_calc returned an invalid vacuum density";
                return out;
            }
            out.deltaSN = cfg.snMode == 3
                            ? log10(high_prec_float(1.0) / out.snTotalNvac)
                            : high_prec_float(1.0) / out.snTotalNvac;
            if (isnan(out.deltaSN) || isinf(out.deltaSN)) {
                out.error = "DSN_calc returned a non-finite naturalness measure";
                return out;
            }
            out.haveDSN = true;
        }

        out.ok = true;
        return out;
    } catch (const std::exception & e) {
        out.ok = false;
        out.error = std::string("exception: ") + e.what();
        return out;
    } catch (...) {
        out.ok = false;
        out.error = "unknown exception in natlha::evaluate";
        return out;
    }
}

}  // namespace natlha
