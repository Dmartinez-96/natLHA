#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <limits>
#include <boost/math/special_functions/next.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include "DBG_calc.hpp"
#include "MSSM_RGE_solver.hpp"
#include "mZ_numsolver.hpp"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace boost::multiprecision;
typedef number<mpfr_float_backend<50>> high_prec_float;

/// x advanced by one DOUBLE-precision ULP, so that `dblNext(x) - x` is exactly that ULP.
///
/// This is the right ULP for sizing the finite-difference steps in this file, and
/// boost::math::float_next is not. float_next advances by one ULP of `high_prec_float`, which is
/// mpfr_float_backend<50> -- roughly 1e-50 relative. The steps below are sized as
/// h = (C * ulp)^(1/n), but the function being differentiated is evaluated in DOUBLE:
/// deriv_mZ_step_calc converts its boundary conditions to vector<double> and hands them to
/// solveODEs. Estimating the MPFR ULP as 1e-50 relative, at a reference scale of 5000 that
/// choice gives
///     8-point  1.263e-05  relative 2.53e-09   (fine, far above double resolution)
///     4-point  8.913e-10  relative 1.78e-13   (only ~800 double ULPs, at the noise floor)
///     2-point  5.313e-16  relative 1.06e-19   (BELOW double epsilon, 2.22e-16)
/// For the 2-point rule x + h == x exactly in double, making f(+h) and f(-h) bit-identical and
/// the central difference identically zero. Measured on the arXiv:2111.03096 benchmark with
/// MPFR-sized steps: Delta_BG = -5253.40158787 at precision 1, -5234.93148081 at precision 2,
/// and exactly 0 at precision 3, where all six of its contributions were 0. The agreement
/// between precisions 1 and 2 to 0.35 percent was luck near the noise floor, not accuracy.
///
/// Sizing from the double ULP gives relative steps of 1.66e-05, 1.32e-06 and 2.99e-08 for the
/// 8-, 4- and 2-point rules at that same scale -- all far above double epsilon. The reference is
/// max(|x|, 1.0) rather than |x| so a parameter passing through zero cannot collapse the step to
/// a denormal.
static high_prec_float dblNext(const high_prec_float & x) {
    const double xd = double(x);
    const double ref = std::max(std::abs(xd), 1.0);
    const double ulp = std::nextafter(ref, std::numeric_limits<double>::infinity()) - ref;
    return x + high_prec_float(ulp);
}

high_prec_float deriv_mZ_step_calc(high_prec_float RGE_scale_init_val, high_prec_float RGE_scale_final_val, vector<high_prec_float> BCs_to_run) {
    vector<double> BCs_to_run_dbl;
    for (const auto& value : BCs_to_run) {
        BCs_to_run_dbl.push_back(double(value));
    }
    double RGE_scale_init_val_dbl = double(RGE_scale_init_val);
    double RGE_scale_final_val_dbl = double(RGE_scale_final_val);
    vector<double> currentweaksol_dbl = solveODEs(BCs_to_run_dbl, RGE_scale_init_val_dbl, RGE_scale_final_val_dbl, copysign(1.0e-6, (RGE_scale_final_val_dbl - RGE_scale_init_val_dbl)));
    vector<high_prec_float> currentweaksol;
    for (const auto& value : currentweaksol_dbl) {
        currentweaksol.push_back(high_prec_float(value));
    }
    high_prec_float QSUSY_for_calc = exp(RGE_scale_final_val);
    high_prec_float mZ2_calc = getmZ2(currentweaksol, QSUSY_for_calc, high_prec_float(91.1876 * 91.1876));
    return mZ2_calc;
}

bool absValCompareBG(const LabeledValueBG& a, const LabeledValueBG& b) {
    return abs(a.value) < abs(b.value);
}

vector<LabeledValueBG> sortAndReturnBG(const vector<LabeledValueBG>& DBGList) {
    vector<LabeledValueBG> sortedList = DBGList;
    sort(sortedList.begin(), sortedList.end(), absValCompareBG);
    reverse(sortedList.begin(), sortedList.end());
    return sortedList;
}

high_prec_float deriv_num_calc(int precselno, high_prec_float curr_hval, vector<high_prec_float> mzsq_values) {
    high_prec_float approxderivval = 0.0;
    if (precselno == 1) {
        // 8-point derivative calculation
        approxderivval = (1.0 / curr_hval) *
            ((mzsq_values[0] / 280.0) - (4.0 / 105.0) * mzsq_values[1] +
            mzsq_values[2] / 5.0 - (4.0 / 5.0) * mzsq_values[3] +
            (4.0 / 5.0) * mzsq_values[4] - mzsq_values[5] / 5.0 +
            (4.0 / 105.0) * mzsq_values[6] - mzsq_values[7] / 280.0);
    } else if (precselno == 2) {
        // 4-point derivative calculation
        approxderivval = (1.0 / curr_hval) *
            ((mzsq_values[0] / 12.0) - (2.0 / 3.0) * mzsq_values[1] +
            (2.0 / 3.0) * mzsq_values[2] - mzsq_values[3] / 12.0);
    } else {
        // 2-point derivative calculation (default)
        approxderivval = (1.0 / curr_hval) *
            ((-0.5) * mzsq_values[0] + 0.5 * mzsq_values[1]);
    }

    return approxderivval;
}

// ==========================================================================================
// GENERAL N-DIRECTION MACHINERY
// ==========================================================================================

/// How a direction's shift is applied to the GUT-scale boundary conditions.
///
/// The forms are not interchangeable, and which one a slot takes is set by what that slot
/// stores. Sfermion and Higgs soft slots hold SQUARED masses while the direction is a
/// dimension-1 magnitude, so the shift is applied to the square root and squared back with the
/// original sign restored. Trilinear slots hold the soft term a_ij while the direction is the
/// ratio A = a/y, so the shift is applied to the ratio and multiplied back by the Yukawa, which
/// for slot i sits at i-9. Gaugino masses and mu are already dimension-1 and shift directly.
///
/// The bilinear form exists because slot 42 holds b = B*mu while the direction is B, so the
/// shift applies to b/mu and multiplies back by mu at slot 6. That denominator is NOT i-9,
/// which is why it cannot reuse the trilinear form.
enum BGShiftKind {
    kBGShiftPlain = 0,      ///< BCs[i] += h
    kBGShiftScalar = 1,     ///< BCs[i] = copysign((sqrt(|BCs[i]|) + h)^2, BCs[i])
    kBGShiftTrilinear = 2,  ///< BCs[i] = ((BCs[i] / BCs[i-9]) + h) * BCs[i-9]
    kBGShiftBilinear = 3    ///< BCs[i] = ((BCs[i] / BCs[6]) + h) * BCs[6]
};

/// One Barbieri-Giudice direction.
///
/// `shiftIndices` and `value` are deliberately independent. The set of slots a direction moves
/// is not always the set its magnitude is drawn from: a universal m_0 shifts every soft scalar
/// slot while its magnitude comes from a max over a candidate list, and those two lists differ
/// in the existing models. Keeping them separate lets the general loop stay agnostic while each
/// model supplies whatever value convention that model uses.
struct BGDirection {
    std::string label;             ///< reported as-is in the returned LabeledValueBG
    BGShiftKind kind;
    std::vector<int> shiftIndices;
    high_prec_float value;         ///< dimension-1 magnitude: sets the step size and prefactor
};

/// Finite-difference step for one direction at one precision setting.
///
/// The constant and the root are fixed by the stencil order, and pair with the coefficients in
/// deriv_num_calc: (2625/16, 9) for the 8-point form, (45/4, 5) for the 4-point, (3, 3) for the
/// 2-point default. The scale comes from dblNext, for the reason given on that function.
static high_prec_float bgStepSize(high_prec_float value, int precselno) {
    const high_prec_float delta = dblNext(value) - value;
    if (precselno == 1) return pow((high_prec_float(2625.0) / high_prec_float(16.0)) * delta,
                                   high_prec_float(1.0) / high_prec_float(9.0));
    if (precselno == 2) return pow((high_prec_float(45.0) / high_prec_float(4.0)) * delta,
                                   high_prec_float(1.0) / high_prec_float(5.0));
    return pow(high_prec_float(3.0) * delta, high_prec_float(1.0) / high_prec_float(3.0));
}

/// Stencil node offsets in units of the step, ordered to match deriv_num_calc's coefficients.
///
/// Every stencil here is central and omits the zero node, which is why no unshifted evaluation
/// appears. LOWER precselno IS MORE EXPENSIVE: 1 costs eight solves per direction, 2 costs
/// four, and the default costs two.
static std::vector<high_prec_float> bgStencilNodes(int precselno) {
    if (precselno == 1) return {-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0};
    if (precselno == 2) return {-2.0, -1.0, 1.0, 2.0};
    return {-1.0, 1.0};
}

/// Apply one direction's shift to a COPY of the boundary conditions and return the resulting
/// m_Z^2 after running down to the weak scale.
///
/// Takes `inputGUT_BCs` by value: every stencil node must start from the unshifted point, so
/// shifting a shared vector would accumulate the offsets across nodes.
static high_prec_float bgShiftedMZ2(const BGDirection& dir, high_prec_float shift,
                                    std::vector<high_prec_float> inputGUT_BCs,
                                    high_prec_float initialScale, high_prec_float finalScale) {
    for (int i : dir.shiftIndices) {
        if (dir.kind == kBGShiftPlain) {
            inputGUT_BCs[i] += shift;
        } else if (dir.kind == kBGShiftScalar) {
            inputGUT_BCs[i] = copysign(pow(sqrt(abs(inputGUT_BCs[i])) + shift, high_prec_float(2.0)),
                                       inputGUT_BCs[i]);
        } else {
            // Trilinear and bilinear differ only in which slot holds the denominator: the
            // matching Yukawa at i-9 for a_ij, and mu at slot 6 for b.
            const int denom = (dir.kind == kBGShiftBilinear) ? 6 : (i - 9);
            inputGUT_BCs[i] = ((inputGUT_BCs[i] / inputGUT_BCs[denom]) + shift) * inputGUT_BCs[denom];
        }
    }
    return deriv_mZ_step_calc(initialScale, finalScale, inputGUT_BCs);
}

/// The directions of one model, in the order they are reported.
///
/// Order does not affect the returned Delta_BG: DBG_calc sorts by absolute value before
/// returning, so element [0] is the largest contribution regardless of how this list is built.
///
/// ONE `value` SERVES BOTH THE PREFACTOR AND THE STEP SIZE, and that is only sound because the
/// step is sign-independent. bgStepSize takes its scale from dblNext(value) - value, and
/// dblNext advances by ulp(max(|x|, 1.0)), so the difference depends on |value| alone. The
/// prefactor, by contrast, is sign-sensitive, so `value` carries the SIGNED magnitude wherever
/// the model's prefactor is signed.
///
/// EACH MODEL'S PREFACTOR CONVENTION IS REPRODUCED AS IT STANDS, including where the models
/// disagree with each other, because Delta_BG is defined with respect to them:
///   - The SIGNED root is used by mHu and mHd wherever they appear as separate directions, by
///     model 1's universal m_0, and by model 2's combined mHu,d. Everything else scalar takes
///     the UNSIGNED root: the universal m_0 of models 2 and 3, the per-generation m_0 groups of
///     models 4 and 5, and each per-slot sfermion direction of pMSSM-30. A signed prefactor is
///     what lets a contribution come out negative, which is why Delta_BG(mHu) is negative on
///     the benchmark.
/// That sign convention is preserved as it stands rather than unified, because Delta_BG for
/// those models is defined with respect to it.
static std::vector<BGDirection> bgDirections(int modselno,
                                             const std::vector<high_prec_float>& G) {
    auto idxRange = [](int lo, int hiExclusive) {
        std::vector<int> v;
        for (int i = lo; i < hiExclusive; ++i) v.push_back(i);
        return v;
    };
    auto maxByAbs = [&G](const std::vector<int>& idx) {
        high_prec_float best = G[idx[0]];
        for (int i : idx) if (abs(G[i]) > abs(best)) best = G[i];
        return best;
    };
    auto signedRoot = [](high_prec_float v) { return copysign(sqrt(abs(v)), v); };
    auto plainRoot = [](high_prec_float v) { return sqrt(abs(v)); };

    // Largest gaugino mass by VALUE, not by magnitude: the gaugino candidate comparison is a
    // plain max in every model.
    high_prec_float gauginoVal = G[3];
    for (int i = 3; i < 6; ++i) if (G[i] > gauginoVal) gauginoVal = G[i];

    // Largest trilinear RATIO A = a/y by magnitude, over all nine generations.
    high_prec_float trilinVal = G[16] / G[7];
    for (int i = 16; i < 25; ++i) {
        const high_prec_float r = G[i] / G[i - 9];
        if (abs(r) > abs(trilinVal)) trilinVal = r;
    }

    // All 15 sfermion soft slots, 27 through 41 inclusive. Slot 32 is mL3^2, the
    // third-generation left-handed slepton; in a model with a single universal scalar mass it
    // is unified with the other soft scalars at the GUT scale, so it belongs in the list that
    // picks the universal magnitude. The shift over these directions already covered slot 32
    // via its index range, so only the magnitude had been ignoring it.
    const std::vector<int> universalScalars =
        {27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41};
    const std::vector<int> gen12Scalars = {27, 28, 30, 31, 33, 34, 36, 37, 39, 40};
    const std::vector<int> gen1Scalars  = {27, 30, 33, 36, 39};
    const std::vector<int> gen2Scalars  = {28, 31, 34, 37, 40};
    const std::vector<int> gen3Scalars  = {29, 32, 35, 38, 41};

    const BGDirection gauginoDir = {"Delta_BG(m_1/2)", kBGShiftPlain, {3, 4, 5}, gauginoVal};
    const BGDirection trilinDir  = {"Delta_BG(A_0)", kBGShiftTrilinear, idxRange(16, 25), trilinVal};
    const BGDirection muDir      = {"Delta_BG(mu_0)", kBGShiftPlain, {6}, G[6]};
    const BGDirection mHuDir     = {"Delta_BG(mHu)", kBGShiftScalar, {25}, signedRoot(G[25])};
    const BGDirection mHdDir     = {"Delta_BG(mHd)", kBGShiftScalar, {26}, signedRoot(G[26])};

    std::vector<BGDirection> dirs;
    if (modselno == 1) {
        std::vector<int> cands = universalScalars;
        cands.push_back(25);
        cands.push_back(26);
        dirs.push_back({"Delta_BG(m_0)", kBGShiftScalar, idxRange(25, 42),
                        signedRoot(maxByAbs(cands))});
        dirs.push_back(gauginoDir);
        dirs.push_back(trilinDir);
        dirs.push_back(muDir);
    } else if (modselno == 2) {
        dirs.push_back({"Delta_BG(mHu,d)", kBGShiftScalar, {25, 26},
                        signedRoot(maxByAbs({25, 26}))});
        dirs.push_back({"Delta_BG(m_0)", kBGShiftScalar, idxRange(27, 42),
                        plainRoot(maxByAbs(universalScalars))});
        dirs.push_back(gauginoDir);
        dirs.push_back(trilinDir);
        dirs.push_back(muDir);
    } else if (modselno == 3) {
        dirs.push_back(mHuDir);
        dirs.push_back(mHdDir);
        dirs.push_back({"Delta_BG(m_0)", kBGShiftScalar, idxRange(27, 42),
                        plainRoot(maxByAbs(universalScalars))});
        dirs.push_back(gauginoDir);
        dirs.push_back(trilinDir);
        dirs.push_back(muDir);
    } else if (modselno == 4) {
        dirs.push_back(mHuDir);
        dirs.push_back(mHdDir);
        dirs.push_back({"Delta_BG(m_0(1,2))", kBGShiftScalar, gen12Scalars,
                        plainRoot(maxByAbs(gen12Scalars))});
        dirs.push_back({"Delta_BG(m_0(3))", kBGShiftScalar, gen3Scalars,
                        plainRoot(maxByAbs(gen3Scalars))});
        dirs.push_back(gauginoDir);
        dirs.push_back(trilinDir);
        dirs.push_back(muDir);
    } else if (modselno == 5) {
        dirs.push_back(mHuDir);
        dirs.push_back(mHdDir);
        dirs.push_back({"Delta_BG(m_0(1))", kBGShiftScalar, gen1Scalars,
                        plainRoot(maxByAbs(gen1Scalars))});
        dirs.push_back({"Delta_BG(m_0(2))", kBGShiftScalar, gen2Scalars,
                        plainRoot(maxByAbs(gen2Scalars))});
        dirs.push_back({"Delta_BG(m_0(3))", kBGShiftScalar, gen3Scalars,
                        plainRoot(maxByAbs(gen3Scalars))});
        dirs.push_back(gauginoDir);
        dirs.push_back(trilinDir);
        dirs.push_back(muDir);
    } else {
        // pMSSM-30 plus mu: 31 independent directions, which is the whole point of this branch.
        // Nothing is collapsed by a max here -- each slot is its own direction, so the 15
        // sfermion soft masses, the 9 trilinears and the 3 gauginos are separate, and the Higgs
        // bilinear b joins as the 31st alongside mu.
        //   3 gaugino + 1 mu + 9 trilinear + 2 Higgs soft + 15 sfermion + 1 bilinear = 31.
        static const char * sfermionNames[15] = {
            "mQ1", "mQ2", "mQ3", "mL1", "mL2", "mL3", "mU1", "mU2", "mU3",
            "mD1", "mD2", "mD3", "mE1", "mE2", "mE3"
        };
        static const char * trilinearNames[9] = {
            "A_t", "A_c", "A_u", "A_b", "A_s", "A_d", "A_tau", "A_mu", "A_e"
        };
        static const char * gauginoNames[3] = {"M_1", "M_2", "M_3"};

        dirs.push_back(mHuDir);
        dirs.push_back(mHdDir);
        for (int k = 0; k < 15; ++k) {
            const int i = 27 + k;
            dirs.push_back({std::string("Delta_BG(") + sfermionNames[k] + ")",
                            kBGShiftScalar, {i}, plainRoot(G[i])});
        }
        for (int k = 0; k < 3; ++k) {
            const int i = 3 + k;
            dirs.push_back({std::string("Delta_BG(") + gauginoNames[k] + ")",
                            kBGShiftPlain, {i}, G[i]});
        }
        for (int k = 0; k < 9; ++k) {
            const int i = 16 + k;
            dirs.push_back({std::string("Delta_BG(") + trilinearNames[k] + ")",
                            kBGShiftTrilinear, {i}, G[i] / G[i - 9]});
        }
        dirs.push_back(muDir);
        dirs.push_back({"Delta_BG(B)", kBGShiftBilinear, {42}, G[42] / G[6]});
    }
    return dirs;
}

std::vector<LabeledValueBG> DBG_calc(int& modselno, int& precselno,
                                high_prec_float GUT_SCALE, high_prec_float myweakscale, high_prec_float inptanbval,
                                std::vector<high_prec_float> GUT_boundary_conditions, high_prec_float originalmZ2value) {
    // GUT_SCALE and myweakscale should be log(Q)
    vector<LabeledValueBG> dbglist;
    high_prec_float mymZ_squared = 91.1876 * 91.1876;

    // One loop over the model's directions, whatever the model. The direction list carries the
    // shift kind, the slots to move and the magnitude, so nothing here depends on which model
    // is selected or on how many directions it has -- which is what lets pMSSM-30 plus mu run
    // its directions through the same code that runs CMSSM's.
    //
    // COST, so it is not a surprise: solves = directions * stencil nodes, and every solve is
    // one solveODEs from the GUT scale to the weak scale plus one getmZ2. bgStencilNodes
    // returns 8, 4 and 2 nodes for precselno 1, 2 and anything else respectively, so a LOWER
    // precselno is more expensive.
    const std::vector<BGDirection> directions = bgDirections(modselno, GUT_boundary_conditions);
    const std::vector<high_prec_float> stencilNodes = bgStencilNodes(precselno);
    for (std::size_t d = 0; d < directions.size(); ++d) {
        const BGDirection& dir = directions[d];
        const high_prec_float h = bgStepSize(dir.value, precselno);
        std::vector<high_prec_float> mZ2Values;
        mZ2Values.reserve(stencilNodes.size());
        for (std::size_t n = 0; n < stencilNodes.size(); ++n) {
            mZ2Values.push_back(bgShiftedMZ2(dir, stencilNodes[n] * h, GUT_boundary_conditions,
                                             GUT_SCALE, myweakscale));
        }
        dbglist.push_back({(dir.value / mymZ_squared)
                               * deriv_num_calc(precselno, h, mZ2Values),
                           dir.label});
    }
    return sortAndReturnBG(dbglist);
}
