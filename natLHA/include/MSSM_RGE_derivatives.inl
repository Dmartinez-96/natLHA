#ifndef NATLHA_MSSM_RGE_DERIVATIVES_INL
#define NATLHA_MSSM_RGE_DERIVATIVES_INL

#include "constants.hpp"
#include <cmath>
#include <type_traits>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef __CUDACC__
#define NATLHA_RGE_HD __host__ __device__
#else
#define NATLHA_RGE_HD
#endif

template <typename Real>
NATLHA_RGE_HD inline Real rgePow(const Real& value, double exponent) {
    using std::pow;
    static_assert(
        std::is_same<decltype(pow(value, exponent)), Real>::value,
        "the underlying pow overload must preserve the RGE scalar type");
    return pow(value, exponent);
}

template <typename InputState, typename OutputState>
NATLHA_RGE_HD inline void MSSMRGEDerivatives(
        const InputState& x, OutputState& dxdt) {
    using Real = decltype(x[0] + x[0]);
    const Real rgeLoopFactor = Real(1) / (Real(16) * rgePow(Real(M_PI), 2.0));
    const Real rgeLoopFactorSquared = rgePow(rgeLoopFactor, 2.0);
    // Extract values from the input vector x
    Real g1_val = x[0];
    Real g2_val = x[1];
    Real g3_val = x[2];
    Real M1_val = x[3];
    Real M2_val = x[4];
    Real M3_val = x[5];
    Real mu_val = x[6];
    Real yt_val = x[7];
    Real yc_val = x[8];
    Real yu_val = x[9];
    Real yb_val = x[10];
    Real ys_val = x[11];
    Real yd_val = x[12];
    Real ytau_val = x[13];
    Real ymu_val = x[14];
    Real ye_val = x[15];
    Real at_val = x[16];
    Real ac_val = x[17];
    Real au_val = x[18];
    Real ab_val = x[19];
    Real as_val = x[20];
    Real ad_val = x[21];
    Real atau_val = x[22];
    Real amu_val = x[23];
    Real ae_val = x[24];
    Real mHu_sq_val = x[25];
    Real mHd_sq_val = x[26];
    Real mQ1_sq_val = x[27];
    Real mQ2_sq_val = x[28];
    Real mQ3_sq_val = x[29];
    Real mL1_sq_val = x[30];
    Real mL2_sq_val = x[31];
    Real mL3_sq_val = x[32];
    Real mU1_sq_val = x[33];
    Real mU2_sq_val = x[34];
    Real mU3_sq_val = x[35];
    Real mD1_sq_val = x[36];
    Real mD2_sq_val = x[37];
    Real mD3_sq_val = x[38];
    Real mE1_sq_val = x[39];
    Real mE2_sq_val = x[40];
    Real mE3_sq_val = x[41];
    Real b_val = x[42];
    Real tanb_val = x[43];

    // Gauge coupling and gaugino mass beta functions
    /////////////////////////////////////////////////
    // 1-loop
    Real dg1_dt_1l = b_1l[0] * rgePow(g1_val, 3.0);
    Real dg2_dt_1l = b_1l[1] * rgePow(g2_val, 3.0);
    Real dg3_dt_1l = b_1l[2] * rgePow(g3_val, 3.0);
    Real dM1_dt_1l = b_1l[0] * 2.0 * rgePow(g1_val, 2.0) * M1_val;
    Real dM2_dt_1l = b_1l[1] * 2.0 * rgePow(g2_val, 2.0) * M2_val;
    Real dM3_dt_1l = b_1l[2] * 2.0 * rgePow(g3_val, 2.0) * M3_val;

    // 2-loop
    Real dg1_dt_2l = (rgePow(g1_val, 3.0)
                        * ((b_2l[0][0] * rgePow(g1_val, 2.0))
                            + (b_2l[0][1] * rgePow(g2_val, 2.0))
                            + (b_2l[0][2] * rgePow(g3_val, 2.0)) // Tr(Yu^2)
                            - (c_2l[0][0] * (rgePow(yt_val, 2.0)
                                                + rgePow(yc_val, 2.0)
                                                + rgePow(yu_val, 2.0))) // end trace, begin Tr(Yd^2)
                            - (c_2l[0][1] * (rgePow(yb_val, 2.0)
                                                + rgePow(ys_val, 2.0)
                                                + rgePow(yd_val, 2.0))) // end trace, begin Tr(Ye^2)
                            - (c_2l[0][2] * (rgePow(ytau_val, 2.0)
                                                + rgePow(ymu_val, 2.0)
                                                + rgePow(ye_val, 2.0))))); // end trace
    Real dg2_dt_2l = (rgePow(g2_val, 3.0)
                        * ((b_2l[1][0] * rgePow(g1_val, 2.0))
                            + (b_2l[1][1] * rgePow(g2_val, 2.0))
                            + (b_2l[1][2] * rgePow(g3_val, 2.0)) // Tr(Yu^2)
                            - (c_2l[1][0] * (rgePow(yt_val, 2.0)
                                                + rgePow(yc_val, 2.0)
                                                + rgePow(yu_val, 2.0))) // end trace, begin Tr(Yd^2)
                            - (c_2l[1][1] * (rgePow(yb_val, 2.0)
                                                + rgePow(ys_val, 2.0)
                                                + rgePow(yd_val, 2.0))) // end trace, begin Tr(Ye^2)
                            - (c_2l[1][2] * (rgePow(ytau_val, 2.0)
                                                + rgePow(ymu_val, 2.0)
                                                + rgePow(ye_val, 2.0))))); // end trace;
    Real dg3_dt_2l = (rgePow(g3_val, 3.0)
                        * ((b_2l[2][0] * rgePow(g1_val, 2.0))
                            + (b_2l[2][1] * rgePow(g2_val, 2.0))
                            + (b_2l[2][2] * rgePow(g3_val, 2.0)) // Tr(Yu^2)
                            - (c_2l[2][0] * (rgePow(yt_val, 2.0)
                                                + rgePow(yc_val, 2.0)
                                                + rgePow(yu_val, 2.0))) // end trace, begin Tr(Yd^2)
                            - (c_2l[2][1] * (rgePow(yb_val, 2.0)
                                                + rgePow(ys_val, 2.0)
                                                + rgePow(yd_val, 2.0))) // end trace, begin Tr(Ye^2)
                            - (c_2l[2][2] * (rgePow(ytau_val, 2.0)
                                                + rgePow(ymu_val, 2.0)
                                                + rgePow(ye_val, 2.0))))); // end trace
;
    Real dM1_dt_2l = (2.0 * rgePow(g1_val, 2.0)
                * (((b_2l[0][0] * rgePow(g1_val, 2.0) * (M1_val + M1_val))
                        + (b_2l[0][1] * rgePow(g2_val, 2.0)
                        * (M1_val + M2_val))
                        + (b_2l[0][2] * rgePow(g3_val, 2.0)
                        * (M1_val + M3_val))) // Tr(Yu*au)
                    + ((c_2l[0][0] * (((yt_val * at_val)
                                        + (yc_val * ac_val)
                                        + (yu_val * au_val)) // end trace, begin Tr(Yu^2)
                                    - (M1_val * (rgePow(yt_val, 2.0)
                                                    + rgePow(yc_val, 2.0)
                                                    + rgePow(yu_val, 2.0))) // end trace
                                    ))) // Tr(Yd*ad)
                    + ((c_2l[0][1] * (((yb_val * ab_val)
                                        + (ys_val * as_val)
                                        + (yd_val * ad_val)) // end trace, begin Tr(Yd^2)
                                    - (M1_val * (rgePow(yb_val, 2.0)
                                                    + rgePow(ys_val, 2.0)
                                                    + rgePow(yd_val, 2.0))) // end trace
                                    ))) // Tr(Ye*ae)
                    + ((c_2l[0][2] * (((ytau_val * atau_val)
                                        + (ymu_val * amu_val)
                                        + (ye_val * ae_val)) // end trace, begin Tr(Ye^2)
                                    - (M1_val * (rgePow(ytau_val, 2.0)
                                                    + rgePow(ymu_val, 2.0)
                                                    + rgePow(ye_val, 2.0)))
                                    ))))); // end trace
    Real dM2_dt_2l = (2.0 * rgePow(g2_val, 2.0)
                * (((b_2l[1][0] * rgePow(g1_val, 2.0) * (M2_val + M1_val))
                        + (b_2l[1][1] * rgePow(g2_val, 2.0)
                        * (M2_val + M2_val))
                        + (b_2l[1][2] * rgePow(g3_val, 2.0)
                        * (M2_val + M3_val))) // Tr(Yu*au)
                    + ((c_2l[1][0] * (((yt_val * at_val)
                                        + (yc_val * ac_val)
                                        + (yu_val * au_val)) // end trace, begin Tr(Yu^2)
                                    - (M2_val * (rgePow(yt_val, 2.0)
                                                    + rgePow(yc_val, 2.0)
                                                    + rgePow(yu_val, 2.0))) // end trace
                                    ))) // Tr(Yd*ad)
                    + ((c_2l[1][1] * (((yb_val * ab_val)
                                        + (ys_val * as_val)
                                        + (yd_val * ad_val)) // end trace, begin Tr(Yd^2)
                                    - (M2_val * (rgePow(yb_val, 2.0)
                                                    + rgePow(ys_val, 2.0)
                                                    + rgePow(yd_val, 2.0))) // end trace
                                    ))) // Tr(Ye*ae)
                    + ((c_2l[1][2] * (((ytau_val * atau_val)
                                        + (ymu_val * amu_val)
                                        + (ye_val * ae_val)) // end trace, begin Tr(Ye^2)
                                    - (M2_val * (rgePow(ytau_val, 2.0)
                                                    + rgePow(ymu_val, 2.0)
                                                    + rgePow(ye_val, 2.0))) // end trace
                                    )))));
    Real dM3_dt_2l = (2.0 * rgePow(g3_val, 2.0)
                * (((b_2l[2][0] * rgePow(g1_val, 2.0) * (M3_val + M1_val))
                        + (b_2l[2][1] * rgePow(g2_val, 2.0)
                        * (M3_val + M2_val))
                        + (b_2l[2][2] * rgePow(g3_val, 2.0)
                        * (M3_val + M3_val))) // Tr(Yu*au)
                    + ((c_2l[2][0] * (((yt_val * at_val)
                                        + (yc_val * ac_val)
                                        + (yu_val * au_val)) // end trace, begin Tr(Yu^2)
                                    - (M3_val * (rgePow(yt_val, 2.0)
                                                    + rgePow(yc_val, 2.0)
                                                    + rgePow(yu_val, 2.0))) // end trace
                                    ))) // Tr(Yd*ad)
                    + ((c_2l[2][1] * (((yb_val * ab_val)
                                        + (ys_val * as_val)
                                        + (yd_val * ad_val)) // end trace, begin Tr(Yd^2)
                                    - (M3_val * (rgePow(yb_val, 2.0)
                                                    + rgePow(ys_val, 2.0)
                                                    + rgePow(yd_val, 2.0))) // end trace
                                    ))) // Tr(Ye*ae)
                    + ((c_2l[2][2] * (((ytau_val * atau_val)
                                        + (ymu_val * amu_val)
                                        + (ye_val * ae_val)) // end trace, begin Tr(Ye^2)
                                    - (M3_val * (rgePow(ytau_val, 2.0)
                                                    + rgePow(ymu_val, 2.0)
                                                    + rgePow(ye_val, 2.0))) // end trace
                                    )))));


    // Calculate total gauge coupling and gaugino mass beta functions
    Real dg1_dt = rgeLoopFactor * dg1_dt_1l + rgeLoopFactorSquared * dg1_dt_2l;
    Real dg2_dt = rgeLoopFactor * dg2_dt_1l + rgeLoopFactorSquared * dg2_dt_2l;
    Real dg3_dt = rgeLoopFactor * dg3_dt_1l + rgeLoopFactorSquared * dg3_dt_2l;
    Real dM1_dt = rgeLoopFactor * dM1_dt_1l + rgeLoopFactorSquared * dM1_dt_2l;
    Real dM2_dt = rgeLoopFactor * dM2_dt_1l + rgeLoopFactorSquared * dM2_dt_2l;
    Real dM3_dt = rgeLoopFactor * dM3_dt_1l + rgeLoopFactorSquared * dM3_dt_2l;
    
    // Higgsino mass parameter mu
    ////////////////////////////////////////////////////////////////////
    // 1-loop
    Real dmu_dt_1l = (mu_val // Tr(3Yu^2 + 3Yd^2 + Ye^2)
                * ((3.0 * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                        + rgePow(yu_val, 2.0) + rgePow(yb_val, 2.0)
                        + rgePow(ys_val, 2.0) + rgePow(yd_val, 2.0)))
                    + (rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                        + rgePow(ye_val, 2.0)) // end trace
                    - (3.0 * rgePow(g2_val, 2.0))
                    - ((3.0 / 5.0) * rgePow(g1_val, 2.0))));

    // 2-loop
    Real dmu_dt_2l = (mu_val // Tr(3Yu^4 + 3Yd^4 + (2Yu^2*Yd^2) + Ye^4)
                * ((-3.0 * ((3.0 * (rgePow(yt_val, 4.0) + rgePow(yc_val, 4.0)
                                    + rgePow(yu_val, 4.0)
                                    + rgePow(yb_val, 4.0)
                                    + rgePow(ys_val, 4.0)
                                    + rgePow(yd_val, 4.0)))
                            + (2.0 * ((rgePow(yt_val, 2.0)
                                    * rgePow(yb_val, 2.0))
                                    + (rgePow(yc_val, 2.0)
                                    * rgePow(ys_val, 2.0))
                                    + (rgePow(yu_val, 2.0)
                                    * rgePow(yd_val, 2.0))))
                            + (rgePow(ytau_val, 4.0) + rgePow(ymu_val, 4.0)
                                + rgePow(ye_val, 4.0)))) // end trace
                    + (((16.0 * rgePow(g3_val, 2.0))
                        + (4.0 * rgePow(g1_val, 2.0) / 5.0)) // Tr(Yu^2)
                        * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                        + rgePow(yu_val, 2.0))) // end trace
                    + (((16.0 * rgePow(g3_val, 2.0))
                        - (2.0 * rgePow(g1_val, 2.0) / 5.0)) // Tr(Yd^2)
                        * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                        + rgePow(yd_val, 2.0))) // end trace
                    + ((6.0 / 5.0) * rgePow(g1_val, 2.0) // Tr(Ye^2)
                        * (rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                        + rgePow(ye_val, 2.0))) // end trace
                    + ((15.0 / 2.0) * rgePow(g2_val, 4.0))
                    + ((9.0 / 5.0) * rgePow(g1_val, 2.0)
                        * rgePow(g2_val, 2.0))
                    + ((207.0 / 50.0) * rgePow(g1_val, 4.0))));

    // Calculate total gauge coupling and gaugino mass beta functions
    Real dmu_dt = rgeLoopFactor * dmu_dt_1l + rgeLoopFactorSquared * dmu_dt_2l;

    // Yukawa couplings for all 3 generations, assumed diagonalized
    //////////////////////////////////////////////////////////////////////////
    // 1-loop
    Real dyt_dt_1l = (yt_val // Tr(3Yu^2)
                * ((3.0 * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                        + rgePow(yu_val, 2.0))) // end trace
                    + (3.0 * (rgePow(yt_val, 2.0)))
                    + rgePow(yb_val, 2.0)
                    - ((16.0 / 3.0) * rgePow(g3_val, 2.0))
                    - (3.0 * rgePow(g2_val, 2.0))
                    - ((13.0 / 15.0) * rgePow(g1_val, 2.0))));
    Real dyc_dt_1l = (yc_val // Tr(3Yu^2)
                * ((3.0 * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                        + rgePow(yu_val, 2.0))) // end trace
                    + (3.0 * (rgePow(yc_val, 2.0)))
                    + rgePow(ys_val, 2.0)
                    - ((16.0 / 3.0) * rgePow(g3_val, 2.0))
                    - (3.0 * rgePow(g2_val, 2.0))
                    - ((13.0 / 15.0) * rgePow(g1_val, 2.0))));
    Real dyu_dt_1l = (yu_val // Tr(3Yu^2)
                * ((3.0 * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                        + rgePow(yu_val, 2.0))) // end trace
                    + (3.0 * (rgePow(yu_val, 2.0)))
                    + rgePow(yd_val, 2.0)
                    - ((16.0 / 3.0) * rgePow(g3_val, 2.0))
                    - (3.0 * rgePow(g2_val, 2.0))
                    - ((13.0 / 15.0) * rgePow(g1_val, 2.0))));
    
    Real dyb_dt_1l = (yb_val // Tr(3Yd^2 + Ye^2)
                * ((3.0 * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                        + rgePow(yd_val, 2.0)))
                    + (rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                        + rgePow(ye_val, 2.0)) // end trace
                    + (3.0 * (rgePow(yb_val, 2.0))) + rgePow(yt_val, 2.0)
                    - ((16.0 / 3.0) * rgePow(g3_val, 2.0))
                    - (3.0 * rgePow(g2_val, 2.0))
                    - ((7.0 / 15.0) * rgePow(g1_val, 2.0))));
    
    Real dys_dt_1l = (ys_val // Tr(3Yd^2 + Ye^2)
                * ((3.0 * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                        + rgePow(yd_val, 2.0)))
                    + (rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                        + rgePow(ye_val, 2.0)) // end trace
                    + (3.0 * (rgePow(ys_val, 2.0))) + rgePow(yc_val, 2.0)
                    - ((16.0 / 3.0) * rgePow(g3_val, 2.0))
                    - (3.0 * rgePow(g2_val, 2.0))
                    - ((7.0 / 15.0) * rgePow(g1_val, 2.0))));
    
    Real dyd_dt_1l = (yd_val // Tr(3Yd^2 + Ye^2)
                * ((3.0 * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                        + rgePow(yd_val, 2.0)))
                    + (rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                        + rgePow(ye_val, 2.0)) // end trace
                    + (3.0 * (rgePow(yd_val, 2.0))) + rgePow(yu_val, 2.0)
                    - ((16.0 / 3.0) * rgePow(g3_val, 2.0))
                    - (3.0 * rgePow(g2_val, 2.0))
                    - ((7.0 / 15.0) * rgePow(g1_val, 2.0))));

    Real dytau_dt_1l = (ytau_val // Tr(3Yd^2 + Ye^2)
                    * ((3.0 * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                            + rgePow(yd_val, 2.0)))
                        + (rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                        + rgePow(ye_val, 2.0)) // end trace
                        + (3.0 * (rgePow(ytau_val, 2.0)))
                        - (3.0 * rgePow(g2_val, 2.0))
                        - ((9.0 / 5.0) * rgePow(g1_val, 2.0))));

    Real dymu_dt_1l = (ymu_val // Tr(3Yd^2 + Ye^2)
                    * ((3.0 * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                            + rgePow(yd_val, 2.0)))
                        + (rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                        + rgePow(ye_val, 2.0)) // end trace
                        + (3.0 * (rgePow(ymu_val, 2.0)))
                        - (3.0 * rgePow(g2_val, 2.0))
                        - ((9.0 / 5.0) * rgePow(g1_val, 2.0))));

    Real dye_dt_1l = (ye_val // Tr(3Yd^2 + Ye^2)
                * ((3.0 * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                        + rgePow(yd_val, 2.0)))
                    + (rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                        + rgePow(ye_val, 2.0)) // end trace
                    + (3.0 * (rgePow(ye_val, 2.0)))
                    - (3.0 * rgePow(g2_val, 2.0))
                    - ((9.0 / 5.0) * rgePow(g1_val, 2.0))));

    // 2-loop
    Real dyt_dt_2l = (yt_val  // Tr(3Yu^4 + (Yu^2*Yd^2))
                * (((-3.0) * ((3.0 * (rgePow(yt_val, 4.0)
                                    + rgePow(yc_val, 4.0)
                                    + rgePow(yu_val, 4.0)))
                            + (rgePow(yt_val, 2.0) * rgePow(yb_val, 2.0))
                            + (rgePow(yc_val, 2.0) * rgePow(ys_val, 2.0))
                            + (rgePow(yu_val, 2.0) * rgePow(yd_val, 2.0)))
                        ) // end trace
                    - (rgePow(yb_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (rgePow(yb_val, 2.0)
                                + rgePow(ys_val, 2.0)
                                + rgePow(yd_val, 2.0)))
                        + (rgePow(ytau_val, 2.0)
                            + rgePow(ymu_val, 2.0)
                            + rgePow(ye_val, 2.0)))) // end trace
                    - (9.0 * rgePow(yt_val, 2.0) // Tr(Yu^2)
                        * (rgePow(yt_val, 2.0)
                        + rgePow(yc_val, 2.0)
                        + rgePow(yu_val, 2.0))) // end trace
                    - (4.0 * rgePow(yt_val, 4.0) )
                    - (2.0 * rgePow(yb_val, 4.0) )
                    - (2.0 * rgePow(yb_val, 2.0) * rgePow(yt_val, 2.0))
                    + (((16.0 *  rgePow(g3_val, 2.0))
                        + ((4.0 / 5.0) * rgePow(g1_val, 2.0))) // Tr(Yu^2)
                        * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                        + rgePow(yu_val, 2.0))) // end trace
                    + (((6.0 * rgePow(g2_val, 2.0))
                        + ((2.0 / 5.0) * rgePow(g1_val, 2.0)))
                        * rgePow(yt_val, 2.0))
                    + ((2.0 / 5.0) * rgePow(g1_val, 2.0) * rgePow(yb_val, 2.0))
                    - ((16.0 / 9.0) * rgePow(g3_val, 4.0) )
                    + (8.0 * rgePow(g3_val, 2.0) * rgePow(g2_val, 2.0))
                    + ((136.0 / 45.0) * rgePow(g3_val, 2.0)
                        * rgePow(g1_val, 2.0))
                    + ((15.0 / 2.0) * rgePow(g2_val, 4.0) )
                    + (rgePow(g2_val, 2.0) * rgePow(g1_val, 2.0))
                    + ((2743.0 / 450.0) * rgePow(g1_val, 4.0) )));

    Real dyc_dt_2l = (yc_val  // Tr(3Yu^4 + (Yu^2*Yd^2))
                * (((-3.0) * ((3.0 * (rgePow(yt_val, 4.0) 
                                    + rgePow(yc_val, 4.0) 
                                    + rgePow(yu_val, 4.0) ))
                            + (rgePow(yt_val, 2.0) * rgePow(yb_val, 2.0))
                            + (rgePow(yc_val, 2.0) * rgePow(ys_val, 2.0))
                            + (rgePow(yu_val, 2.0) * rgePow(yd_val, 2.0)))
                        ) //end trace
                    - (rgePow(ys_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (rgePow(yb_val, 2.0)
                                + rgePow(ys_val, 2.0)
                                + rgePow(yd_val, 2.0)))
                        + (rgePow(ytau_val, 2.0)
                            + rgePow(ymu_val, 2.0)
                            + rgePow(ye_val, 2.0)))) // end trace
                    - (9.0 * rgePow(yc_val, 2.0) // Tr(Yu^2)
                        * (rgePow(yt_val, 2.0)
                        + rgePow(yc_val, 2.0)
                        + rgePow(yu_val, 2.0))) // end trace
                    - (4.0 * rgePow(yc_val, 4.0) )
                    - (2.0 * rgePow(ys_val, 4.0) )
                    - (2.0 * rgePow(ys_val, 2.0)
                        * rgePow(yc_val, 2.0))
                    + (((16.0 *  rgePow(g3_val, 2.0))
                        + ((4.0 / 5.0) * rgePow(g1_val, 2.0))) // Tr(Yu^2)
                        * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                        + rgePow(yu_val, 2.0))) // end trace
                    + (((6.0 * rgePow(g2_val, 2.0))
                        + ((2.0 / 5.0) * rgePow(g1_val, 2.0)))
                        * rgePow(yc_val, 2.0))
                    + ((2.0 / 5.0) * rgePow(g1_val, 2.0) * rgePow(ys_val, 2.0))
                    - ((16.0 / 9.0) * rgePow(g3_val, 4.0) )
                    + (8.0 * rgePow(g3_val, 2.0) * rgePow(g2_val, 2.0))
                    + ((136.0 / 45.0) * rgePow(g3_val, 2.0)
                        * rgePow(g1_val, 2.0))
                    + ((15.0 / 2.0) * rgePow(g2_val, 4.0) )
                    + (rgePow(g2_val, 2.0) * rgePow(g1_val, 2.0))
                    + ((2743.0 / 450.0) * rgePow(g1_val, 4.0) )));

    Real dyu_dt_2l = (yu_val // Tr(3Yu^4 + (Yu^2*Yd^2))
                * (((-3.0) * ((3.0 * (rgePow(yt_val, 4.0) 
                                    + rgePow(yc_val, 4.0) 
                                    + rgePow(yu_val, 4.0) ))
                            + (rgePow(yt_val, 2.0) * rgePow(yb_val, 2.0))
                            + (rgePow(yc_val, 2.0) * rgePow(ys_val, 2.0))
                            + (rgePow(yu_val, 2.0) * rgePow(yd_val, 2.0)))
                        ) // end trace
                    - (rgePow(yd_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (rgePow(yb_val, 2.0)
                                + rgePow(ys_val, 2.0)
                                + rgePow(yd_val, 2.0)))
                        + (rgePow(ytau_val, 2.0)
                            + rgePow(ymu_val, 2.0)
                            + rgePow(ye_val, 2.0))))
                    - (9.0 * rgePow(yu_val, 2.0) // Tr(Yu^2)
                        * (rgePow(yt_val, 2.0)
                        + rgePow(yc_val, 2.0)
                        + rgePow(yu_val, 2.0)))
                    - (4.0 * rgePow(yu_val, 4.0) )
                    - (2.0 * rgePow(yd_val, 4.0) )
                    - (2.0 * rgePow(yd_val, 2.0) * rgePow(yu_val, 2.0))
                    + (((16.0 *  rgePow(g3_val, 2.0))
                        + ((4.0 / 5.0) * rgePow(g1_val, 2.0))) // Tr(Yu^2)
                        * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                        + rgePow(yu_val, 2.0))) // end trace
                    + (((6.0 * rgePow(g2_val, 2.0))
                        + ((2.0 / 5.0) * rgePow(g1_val, 2.0)))
                        * rgePow(yu_val, 2.0))
                    + ((2.0 / 5.0) * rgePow(g1_val, 2.0) * rgePow(yd_val, 2.0))
                    - ((16.0 / 9.0) * rgePow(g3_val, 4.0) )
                    + (8.0 * rgePow(g3_val, 2.0) * rgePow(g2_val, 2.0))
                    + ((136.0 / 45.0) * rgePow(g3_val, 2.0)
                        * rgePow(g1_val, 2.0))
                    + ((15.0 / 2.0) * rgePow(g2_val, 4.0) )
                    + (rgePow(g2_val, 2.0) * rgePow(g1_val, 2.0))
                    + ((2743.0 / 450.0) * rgePow(g1_val, 4.0) )));

    Real dyb_dt_2l = (yb_val // Tr(3Yd^4 + (Yu^2*Yd^2) + Ye^4)
                * (((-3.0) * ((3.0 * (rgePow(yb_val, 4.0) 
                                    + rgePow(ys_val, 4.0) 
                                    + rgePow(yd_val, 4.0) ))
                            + (rgePow(yt_val, 2.0) * rgePow(yb_val, 2.0))
                            + (rgePow(yc_val, 2.0) * rgePow(ys_val, 2.0))
                            + (rgePow(yu_val, 2.0) * rgePow(yd_val, 2.0))
                            + rgePow(ytau_val, 4.0)  + rgePow(ymu_val, 4.0) 
                            + rgePow(ye_val, 4.0) )) // end trace
                    - (3.0 * rgePow(yt_val, 2.0) // Tr(Yu^2)
                        * (rgePow(yt_val, 2.0)
                        + rgePow(yc_val, 2.0)
                        + rgePow(yu_val, 2.0))) // end trace
                    - (3.0 * rgePow(yb_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (rgePow(yb_val, 2.0)
                                + rgePow(ys_val, 2.0)
                                + rgePow(yd_val, 2.0)))
                        + rgePow(ytau_val, 2.0)
                        + rgePow(ymu_val, 2.0)
                        + rgePow(ye_val, 2.0))) // end trace
                    - (4.0 * rgePow(yb_val, 4.0) )
                    - (2.0 * rgePow(yt_val, 4.0) )
                    - (2.0 * rgePow(yt_val, 2.0) * rgePow(yb_val, 2.0))
                    + (((16.0 *  rgePow(g3_val, 2.0))
                        - ((2.0 / 5.0) * rgePow(g1_val, 2.0))) // Tr(Yd^2)
                        * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                        + rgePow(yd_val, 2.0))) // end trace
                    + ((6.0 / 5.0) * rgePow(g1_val, 2.0) // Tr(Ye^2)
                        * (rgePow(ytau_val, 2.0)
                        + rgePow(ymu_val, 2.0)
                        + rgePow(ye_val, 2.0))) // end trace
                    + ((4.0 / 5.0) * rgePow(g1_val, 2.0)
                        * rgePow(yt_val, 2.0))
                    + (rgePow(yb_val, 2.0)
                        * ((6.0 * rgePow(g2_val, 2.0))
                        + ((4.0 / 5.0) * rgePow(g1_val, 2.0))))
                    - ((16.0 / 9.0) * rgePow(g3_val, 4.0) )
                    + (8.0 * rgePow(g3_val, 2.0) * rgePow(g2_val, 2.0))
                    + ((8.0 / 9.0) * rgePow(g3_val, 2.0) * rgePow(g1_val, 2.0))
                    + ((15.0 / 2.0) * rgePow(g2_val, 4.0) )
                    + (rgePow(g2_val, 2.0) * rgePow(g1_val, 2.0))
                    + ((287.0 / 90.0) * rgePow(g1_val, 4.0) )));

    Real dys_dt_2l = (ys_val // Tr(3Yd^4 + (Yu^2*Yd^2) + Ye^4)
                * (((-3.0) * ((3.0 * (rgePow(yb_val, 4.0) 
                                    + rgePow(ys_val, 4.0) 
                                    + rgePow(yd_val, 4.0) ))
                            + (rgePow(yt_val, 2.0) * rgePow(yb_val, 2.0))
                            + (rgePow(yc_val, 2.0) * rgePow(ys_val, 2.0))
                            + (rgePow(yu_val, 2.0) * rgePow(yd_val, 2.0))
                            + rgePow(ytau_val, 4.0) 
                            + rgePow(ymu_val, 4.0) 
                            + rgePow(ye_val, 4.0) )) // end trace
                    - (3.0 * rgePow(yc_val, 2.0) // Tr(Yu^2)
                        * (rgePow(yt_val, 2.0)
                        + rgePow(yc_val, 2.0)
                        + rgePow(yu_val, 2.0))) // end trace
                    - (3.0 * rgePow(ys_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (rgePow(yb_val, 2.0)
                                + rgePow(ys_val, 2.0)
                                + rgePow(yd_val, 2.0)))
                        + rgePow(ytau_val, 2.0)
                        + rgePow(ymu_val, 2.0)
                        + rgePow(ye_val, 2.0))) // end trace
                    - (4.0 * rgePow(ys_val, 4.0) )
                    - (2.0 * rgePow(yc_val, 4.0) )
                    - (2.0 * rgePow(yc_val, 2.0) * rgePow(ys_val, 2.0))
                    + (((16.0 *  rgePow(g3_val, 2.0))
                        - ((2.0 / 5.0) * rgePow(g1_val, 2.0))) // Tr(Yd^2)
                        * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                        + rgePow(yd_val, 2.0))) // end trace
                    + ((6.0 / 5.0) * rgePow(g1_val, 2.0) // Tr(Ye^2)
                        * (rgePow(ytau_val, 2.0)
                        + rgePow(ymu_val, 2.0)
                        + rgePow(ye_val, 2.0))) // end trace
                    + ((4.0 / 5.0) * rgePow(g1_val, 2.0) * rgePow(yc_val, 2.0))
                    + (rgePow(ys_val, 2.0)
                        * ((6.0 * rgePow(g2_val, 2.0))
                        + ((4.0 / 5.0) * rgePow(g1_val, 2.0))))
                    - ((16.0 / 9.0) * rgePow(g3_val, 4.0) )
                    + (8.0 * rgePow(g3_val, 2.0) * rgePow(g2_val, 2.0))
                    + ((8.0 / 9.0) * rgePow(g3_val, 2.0) * rgePow(g1_val, 2.0))
                    + ((15.0 / 2.0) * rgePow(g2_val, 4.0) )
                    + (rgePow(g2_val, 2.0) * rgePow(g1_val, 2.0))
                    + ((287.0 / 90.0) * rgePow(g1_val, 4.0) )));

    Real dyd_dt_2l = (yd_val // Tr(3Yd^4 + (Yu^2*Yd^2) + Ye^4)
                * (((-3.0) * ((3.0 * (rgePow(yb_val, 4.0) 
                                    + rgePow(ys_val, 4.0) 
                                    + rgePow(yd_val, 4.0) ))
                            + (rgePow(yt_val, 2.0) * rgePow(yb_val, 2.0))
                            + (rgePow(yc_val, 2.0) * rgePow(ys_val, 2.0))
                            + (rgePow(yu_val, 2.0) * rgePow(yd_val, 2.0))
                            + rgePow(ytau_val, 4.0)  + rgePow(ymu_val, 4.0) 
                            + rgePow(ye_val, 4.0) )) // end trace
                    - (3.0 * rgePow(yu_val, 2.0) // Tr(Yu^2)
                        * (rgePow(yt_val, 2.0)
                        + rgePow(yc_val, 2.0)
                        + rgePow(yu_val, 2.0))) // end trace
                    - (3.0 * rgePow(yd_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (rgePow(yb_val, 2.0)
                                + rgePow(ys_val, 2.0)
                                + rgePow(yd_val, 2.0)))
                        + rgePow(ytau_val, 2.0)
                        + rgePow(ymu_val, 2.0)
                        + rgePow(ye_val, 2.0))) // end trace
                    - (4.0 * rgePow(yd_val, 4.0) )
                    - (2.0 * rgePow(yu_val, 4.0) )
                    - (2.0 * rgePow(yd_val, 2.0) * rgePow(yu_val, 2.0))
                    + (((16.0 *  rgePow(g3_val, 2.0))
                        - ((2.0 / 5.0) * rgePow(g1_val, 2.0))) // Tr(Yd^2)
                        * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                        + rgePow(yd_val, 2.0))) // end trace
                    + ((6.0 / 5.0) * rgePow(g1_val, 2.0) // Tr(Ye^2)
                        * (rgePow(ytau_val, 2.0)
                        + rgePow(ymu_val, 2.0)
                        + rgePow(ye_val, 2.0))) // end trace
                    + ((4.0 / 5.0) * rgePow(g1_val, 2.0) * rgePow(yu_val, 2.0))
                    + (rgePow(yd_val, 2.0)
                        * ((6.0 * rgePow(g2_val, 2.0))
                        + ((4.0 / 5.0) * rgePow(g1_val, 2.0))))
                    - ((16.0 / 9.0) * rgePow(g3_val, 4.0) )
                    + (8.0 * rgePow(g3_val, 2.0) * rgePow(g2_val, 2.0))
                    + ((8.0 / 9.0) * rgePow(g3_val, 2.0) * rgePow(g1_val, 2.0))
                    + ((15.0 / 2.0) * rgePow(g2_val, 4.0) )
                    + (rgePow(g2_val, 2.0) * rgePow(g1_val, 2.0))
                    + ((287.0 / 90.0) * rgePow(g1_val, 4.0) )));

    Real dytau_dt_2l = (ytau_val // Tr(3Yd^4 + (Yu^2*Yd^2) + Ye^4)
                    * (((-3.0) * ((3.0 * (rgePow(yb_val, 4.0) 
                                    + rgePow(ys_val, 4.0) 
                                    + rgePow(yd_val, 4.0) ))
                                + (rgePow(yt_val, 2.0)
                                    * rgePow(yb_val, 2.0))
                                + (rgePow(yc_val, 2.0)
                                    * rgePow(ys_val, 2.0))
                                + (rgePow(yu_val, 2.0)
                                    * rgePow(yd_val, 2.0))
                                + rgePow(ytau_val, 4.0) 
                                + rgePow(ymu_val, 4.0) 
                                + rgePow(ye_val, 4.0) )) // end trace
                        - (3.0 * rgePow(ytau_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (rgePow(yb_val, 2.0)
                                    + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + rgePow(ytau_val, 2.0)
                            + rgePow(ymu_val, 2.0)
                            + rgePow(ye_val, 2.0))) // end trace
                        - (4.0 * rgePow(ytau_val, 4.0) )
                        + (((16.0 *  rgePow(g3_val, 2.0))
                        - ((2.0 / 5.0) * rgePow(g1_val, 2.0))) // Tr(Yd^2)
                        * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                            + rgePow(yd_val, 2.0))) // end trace
                        + ((6.0 / 5.0) * rgePow(g1_val, 2.0) // Tr(Ye^2)
                        * (rgePow(ytau_val, 2.0)
                            + rgePow(ymu_val, 2.0)
                            + rgePow(ye_val, 2.0))) // end trace
                        + (6.0 * rgePow(g2_val, 2.0) * rgePow(ytau_val, 2.0))
                        + ((15.0 / 2.0) * rgePow(g2_val, 4.0) )
                        + ((9.0 / 5.0) * rgePow(g2_val, 2.0)
                        * rgePow(g1_val, 2.0))
                        + ((27.0 / 2.0) * rgePow(g1_val, 4.0) )));

    Real dymu_dt_2l = (ymu_val // Tr(3Yd^4 + (Yu^2*Yd^2) + Ye^4)
                    * (((-3.0) * ((3.0 * (rgePow(yb_val, 4.0) 
                                    + rgePow(ys_val, 4.0) 
                                    + rgePow(yd_val, 4.0) ))
                                + (rgePow(yt_val, 2.0)
                                    * rgePow(yb_val, 2.0))
                                + (rgePow(yc_val, 2.0)
                                    * rgePow(ys_val, 2.0))
                                + (rgePow(yu_val, 2.0)
                                    * rgePow(yd_val, 2.0))
                                + rgePow(ytau_val, 4.0) 
                                + rgePow(ymu_val, 4.0) 
                                + rgePow(ye_val, 4.0) )) // end trace
                        - (3.0 * rgePow(ymu_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (rgePow(yb_val, 2.0)
                                    + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + rgePow(ytau_val, 2.0)
                            + rgePow(ymu_val, 2.0)
                            + rgePow(ye_val, 2.0))) // end trace
                        - (4.0 * rgePow(ymu_val, 4.0) )
                        + (((16.0 *  rgePow(g3_val, 2.0))
                        - ((2.0 / 5.0) * rgePow(g1_val, 2.0))) // Tr(Yd^2)
                        * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                            + rgePow(yd_val, 2.0))) // end trace
                        + ((6.0 / 5.0) * rgePow(g1_val, 2.0) // Tr(Ye^2)
                        * (rgePow(ytau_val, 2.0)
                            + rgePow(ymu_val, 2.0)
                            + rgePow(ye_val, 2.0))) // end trace
                        + (6.0 * rgePow(g2_val, 2.0) * rgePow(ymu_val, 2.0))
                        + ((15.0 / 2.0) * rgePow(g2_val, 4.0) )
                        + ((9.0 / 5.0) * rgePow(g2_val, 2.0) * rgePow(g1_val, 2.0))
                        + ((27.0 / 2.0) * rgePow(g1_val, 4.0) )));

    Real dye_dt_2l = (ye_val // Tr(3Yd^4 + (Yu^2*Yd^2) + Ye^4)
                * (((-3.0) * ((3.0 * (rgePow(yb_val, 4.0) 
                                    + rgePow(ys_val, 4.0) 
                                    + rgePow(yd_val, 4.0) ))
                            + (rgePow(yt_val, 2.0)
                                * rgePow(yb_val, 2.0))
                            + (rgePow(yc_val, 2.0)
                                * rgePow(ys_val, 2.0))
                            + (rgePow(yu_val, 2.0)
                                * rgePow(yd_val, 2.0))
                            + rgePow(ytau_val, 4.0) 
                            + rgePow(ymu_val, 4.0) 
                            + rgePow(ye_val, 4.0) )) // end trace
                    - (3.0 * rgePow(ye_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (rgePow(yb_val, 2.0)
                                + rgePow(ys_val, 2.0)
                                + rgePow(yd_val, 2.0)))
                        + rgePow(ytau_val, 2.0)
                        + rgePow(ymu_val, 2.0)
                        + rgePow(ye_val, 2.0))) // end trace
                    - (4.0 * rgePow(ye_val, 4.0) )
                    + (((16.0 *  rgePow(g3_val, 2.0))
                        - ((2.0 / 5.0) * rgePow(g1_val, 2.0))) // Tr(Yd^2)
                        * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                        + rgePow(yd_val, 2.0))) // end trace
                    + ((6.0 / 5.0) * rgePow(g1_val, 2.0) // Tr(Ye^2)
                        * (rgePow(ytau_val, 2.0)
                        + rgePow(ymu_val, 2.0)
                        + rgePow(ye_val, 2.0)))
                    + (6.0 * rgePow(g2_val, 2.0) * rgePow(ye_val, 2.0))
                    + ((15.0 / 2.0) * rgePow(g2_val, 4.0) )
                    + ((9.0 / 5.0) * rgePow(g2_val, 2.0) * rgePow(g1_val, 2.0))
                    + ((27.0 / 2.0) * rgePow(g1_val, 4.0) )));

    // Total Yukawa coupling beta functions
    Real dyt_dt = ((rgeLoopFactor * dyt_dt_1l) + (rgeLoopFactorSquared * dyt_dt_2l));

    Real dyc_dt = ((rgeLoopFactor * dyc_dt_1l) + (rgeLoopFactorSquared * dyc_dt_2l));

    Real dyu_dt = ((rgeLoopFactor * dyu_dt_1l) + (rgeLoopFactorSquared * dyu_dt_2l));

    Real dyb_dt = ((rgeLoopFactor * dyb_dt_1l) + (rgeLoopFactorSquared * dyb_dt_2l));

    Real dys_dt = ((rgeLoopFactor * dys_dt_1l) + (rgeLoopFactorSquared * dys_dt_2l));

    Real dyd_dt = ((rgeLoopFactor * dyd_dt_1l) + (rgeLoopFactorSquared * dyd_dt_2l));

    Real dytau_dt = ((rgeLoopFactor * dytau_dt_1l) + (rgeLoopFactorSquared * dytau_dt_2l));

    Real dymu_dt = ((rgeLoopFactor * dymu_dt_1l) + (rgeLoopFactorSquared * dymu_dt_2l));

    Real dye_dt = ((rgeLoopFactor * dye_dt_1l) + (rgeLoopFactorSquared * dye_dt_2l));

    // Soft trilinear couplings, assumed diagonalized
    /////////////////////////////////////////////////////////////////////
    // 1-loop
    Real dat_dt_1l = ((at_val // Tr(Yu^2)
                    * ((3.0 * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                            + rgePow(yu_val, 2.0))) // end trace
                        + (5.0 * rgePow(yt_val, 2.0)) + rgePow(yb_val, 2.0)
                        - ((16.0 / 3.0) * rgePow(g3_val, 2.0))
                        - (3.0 * rgePow(g2_val, 2.0))
                        - ((13.0 / 15.0) * rgePow(g1_val, 2.0))))
                + (yt_val // Tr(au*Yu)
                    * ((6.0 * ((at_val * yt_val) + (ac_val * yc_val)
                            + (au_val * yu_val))) // end trace
                        + (4.0 * yt_val * at_val)
                        + (2.0 * yb_val * ab_val)
                        + ((32.0 / 3.0) * rgePow(g3_val, 2.0) * M3_val)
                        + (6.0 * rgePow(g2_val, 2.0) * M2_val)
                        + ((26.0 / 15.0) * rgePow(g1_val, 2.0) * M1_val))));

    Real dac_dt_1l = ((ac_val // Tr(Yu^2)
                    * ((3.0 * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                            + rgePow(yu_val, 2.0))) // end trace
                        + (5.0 * rgePow(yc_val, 2.0)) + rgePow(ys_val, 2.0)
                        - ((16.0 / 3.0) * rgePow(g3_val, 2.0))
                        - (3.0 * rgePow(g2_val, 2.0))
                        - ((13.0 / 15.0) * rgePow(g1_val, 2.0))))
                + (yc_val // Tr(au*Yu)
                    * ((6.0 * ((at_val * yt_val) + (ac_val * yc_val)
                            + (au_val * yu_val))) // end trace
                        + (4.0 * yc_val * ac_val)
                        + (2.0 * ys_val * as_val)
                        + ((32.0 / 3.0) * rgePow(g3_val, 2.0) * M3_val)
                        + (6.0 * rgePow(g2_val, 2.0) * M2_val)
                        + ((26.0 / 15.0) * rgePow(g1_val, 2.0) * M1_val))));

    Real dau_dt_1l = ((au_val // Tr(Yu^2)
                    * ((3.0 * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                            + rgePow(yu_val, 2.0))) // end trace
                        + (5.0 * rgePow(yu_val, 2.0)) + rgePow(yd_val, 2.0)
                        - ((16.0 / 3.0) * rgePow(g3_val, 2.0))
                        - (3.0 * rgePow(g2_val, 2.0))
                        - ((13.0 / 15.0) * rgePow(g1_val, 2.0))))
                + (yu_val // Tr(au*Yu)
                    * ((6.0 * ((at_val * yt_val) + (ac_val * yc_val)
                            + (au_val * yu_val))) // end trace
                        + (4.0 * yu_val * au_val)
                        + (2.0 * yd_val * ad_val)
                        + ((32.0 / 3.0) * rgePow(g3_val, 2.0) * M3_val)
                        + (6.0 * rgePow(g2_val, 2.0) * M2_val)
                        + ((26.0 / 15.0) * rgePow(g1_val, 2.0) * M1_val))));

    Real dab_dt_1l = ((ab_val // Tr(3Yd^2 + Ye^2)
                    * (((3.0 * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                            + rgePow(yd_val, 2.0)))
                        + (rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                        + rgePow(ye_val, 2.0))) // end trace
                        + (5.0 * rgePow(yb_val, 2.0)) + rgePow(yt_val, 2.0)
                        - ((16.0 / 3.0) * rgePow(g3_val, 2.0))
                        - (3.0 * rgePow(g2_val, 2.0))
                        - ((7.0 / 15.0) * rgePow(g1_val, 2.0))))
                + (yb_val // Tr(6ad*Yd + 2ae*Ye)
                    * ((6.0 * ((ab_val * yb_val) + (as_val * ys_val)
                            + (ad_val * yd_val)))
                        + (2.0 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                + (ae_val * ye_val))) // end trace
                        + (4.0 * yb_val * ab_val) + (2.0 * yt_val * at_val)
                        + ((32.0 / 3.0) * rgePow(g3_val, 2.0) * M3_val)
                        + (6.0 * rgePow(g2_val, 2.0) * M2_val)
                        + ((14.0 / 15.0) * rgePow(g1_val, 2.0) * M1_val))));

    Real das_dt_1l = ((as_val // Tr(3Yd^2 + Ye^2)
                    * (((3.0 * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                            + rgePow(yd_val, 2.0)))
                        + (rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                        + rgePow(ye_val, 2.0))) // end trace
                        + (5.0 * rgePow(ys_val, 2.0)) + rgePow(yc_val, 2.0)
                        - ((16.0 / 3.0) * rgePow(g3_val, 2.0))
                        - (3.0 * rgePow(g2_val, 2.0))
                        - ((7.0 / 15.0) * rgePow(g1_val, 2.0))))
                + (ys_val // Tr(6ad*Yd + 2ae*Ye)
                    * ((6.0 * ((ab_val * yb_val) + (as_val * ys_val)
                            + (ad_val * yd_val)))
                        + (2.0 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                + (ae_val * ye_val)))
                        + (4.0 * ys_val * as_val) + (2.0 * yc_val * ac_val)
                        + ((32.0 / 3.0) * rgePow(g3_val, 2.0) * M3_val)
                        + (6.0 * rgePow(g2_val, 2.0) * M2_val)
                        + ((14.0 / 15.0) * rgePow(g1_val, 2.0) * M1_val))));

    Real dad_dt_1l = ((ad_val // Tr(3Yd^2 + Ye^2)
                    * (((3.0 * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                            + rgePow(yd_val, 2.0)))
                        + (rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                        + rgePow(ye_val, 2.0))) // end trace
                        + (5.0 * rgePow(yd_val, 2.0)) + rgePow(yu_val, 2.0)
                        - ((16.0 / 3.0) * rgePow(g3_val, 2.0))
                        - (3.0 * rgePow(g2_val, 2.0))
                        - ((7.0 / 15.0) * rgePow(g1_val, 2.0))))
                + (yd_val // Tr(6ad*Yd + 2ae*Ye)
                    * ((6.0 * ((ab_val * yb_val) + (as_val * ys_val)
                            + (ad_val * yd_val)))
                        + (2.0 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                + (ae_val * ye_val))) // end trace
                        + (4.0 * yd_val * ad_val) + (2.0 * yu_val * au_val)
                        + ((32.0 / 3.0) * rgePow(g3_val, 2.0) * M3_val)
                        + (6.0 * rgePow(g2_val, 2.0) * M2_val)
                        + ((14.0 / 15.0) * rgePow(g1_val, 2.0) * M1_val))));

    Real datau_dt_1l = ((atau_val // Tr(3Yd^2 + Ye^2)
                    * (((3.0 * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                                + rgePow(yd_val, 2.0)))
                        + (rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                            + rgePow(ye_val, 2.0))) // end trace
                        + (5.0 * rgePow(ytau_val, 2.0))
                        - (3.0 * rgePow(g2_val, 2.0))
                        - ((9.0 / 5.0) * rgePow(g1_val, 2.0))))
                    + (ytau_val // Tr(6ad*Yd + 2ae*Ye)
                        * ((6.0 * ((ab_val * yb_val) + (as_val * ys_val)
                                + (ad_val * yd_val)))
                        + (2.0 * ((atau_val * ytau_val)
                                    + (amu_val * ymu_val)
                                    + (ae_val * ye_val))) // end trace
                        + (4.0 * ytau_val * atau_val)
                        + (6.0 * rgePow(g2_val, 2.0) * M2_val)
                        + ((18.0 / 5.0) * rgePow(g1_val, 2.0) * M1_val))));

    Real damu_dt_1l = ((amu_val // Tr(3Yd^2 + Ye^2)
                    * (((3.0 * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                            + rgePow(yd_val, 2.0)))
                        + (rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                        + rgePow(ye_val, 2.0))) // end trace
                        + (5.0 * rgePow(ymu_val, 2.0))
                        - (3.0 * rgePow(g2_val, 2.0))
                        - ((9.0 / 5.0) * rgePow(g1_val, 2.0))))
                    + (ymu_val // Tr(6ad*Yd + 2ae*Ye)
                        * ((6.0 * ((ab_val * yb_val) + (as_val * ys_val)
                                + (ad_val * yd_val)))
                        + (2.0 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                + (ae_val * ye_val))) // end trace
                        + (4.0 * ymu_val * amu_val)
                        + (6.0 * rgePow(g2_val, 2.0) * M2_val)
                        + ((18.0 / 5.0) * rgePow(g1_val, 2.0) * M1_val))));

    Real dae_dt_1l = ((ae_val // Tr(3Yd^2 + Ye^2)
                    * (((3.0 * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                            + rgePow(yd_val, 2.0)))
                        + (rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                        + rgePow(ye_val, 2.0))) // end trace
                        + (5.0 * rgePow(ye_val, 2.0))
                        - (3.0 * rgePow(g2_val, 2.0))
                        - ((9.0 / 5.0) * rgePow(g1_val, 2.0))))
                + (ye_val // Tr(6ad*Yd + 2ae*Ye)
                    * ((6.0 * ((ab_val * yb_val) + (as_val * ys_val)
                            + (ad_val * yd_val)))
                        + (2.0 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                + (ae_val * ye_val))) // end trace
                        + (4.0 * ye_val * ae_val)
                        + (6.0 * rgePow(g2_val, 2.0) * M2_val)
                        + ((18.0 / 5.0) * rgePow(g1_val, 2.0) * M1_val))));

    // 2-loop
    Real dat_dt_2l = ((at_val // Tr(3Yu^4 + (Yu^2*Yd^2))
                    * (((-3.0) * ((3.0 * (rgePow(yt_val, 4.0)
                                    + rgePow(yc_val, 4.0)
                                    + rgePow(yu_val, 4.0)))
                                + ((rgePow(yt_val, 2.0) * rgePow(yb_val, 2.0))
                                    + (rgePow(yc_val, 2.0)
                                    * rgePow(ys_val, 2.0))
                                    + (rgePow(yu_val, 2.0)
                                    * rgePow(yd_val, 2.0))))) // end trace
                        - (rgePow(yb_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (rgePow(yb_val, 2.0)
                                    + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + (rgePow(ytau_val, 2.0)
                                + rgePow(ymu_val, 2.0)
                                + rgePow(ye_val, 2.0)))) // end trace
                        - (15.0 * rgePow(yt_val, 2.0) // Tr(Yu^2)
                        * (rgePow(yt_val, 2.0)
                            + rgePow(yc_val, 2.0)
                            + rgePow(yu_val, 2.0))) // end trace
                        - (6.0 * rgePow(yt_val, 4.0))
                        - (2.0 * rgePow(yb_val, 4.0))
                        - (4.0 * rgePow(yb_val, 2.0) * rgePow(yt_val, 2.0))
                        + (((16.0 * rgePow(g3_val, 2.0))
                        + ((4.0 / 5.0) * rgePow(g1_val, 2.0))) // Tr(Yu^2)
                        * (rgePow(yt_val, 2.0)
                            + rgePow(yc_val, 2.0)
                            + rgePow(yu_val, 2.0))) // end trace
                        + (12.0 * rgePow(g2_val, 2.0)
                        * rgePow(yt_val, 2.0))
                        + ((2.0 / 5.0) * rgePow(g1_val, 2.0)
                        * rgePow(yb_val, 2.0))
                        - ((16.0 / 9.0) * rgePow(g3_val, 4.0))
                        + (8.0 * rgePow(g3_val, 2.0)
                        * rgePow(g2_val, 2.0))
                        + ((136.0 / 45.0) * rgePow(g3_val, 2.0)
                        * rgePow(g1_val, 2.0))
                        + ((15.0 / 2.0) * rgePow(g2_val, 4.0))
                        + (rgePow(g2_val, 2.0) * rgePow(g1_val, 2.0))
                        + ((2743.0 / 450.0) * rgePow(g1_val, 4.0))))
                + (yt_val // Tr(6au*Yu^3 + au*Yd^2*Yu + ad*Yu^2*Yd)
                    * (((-6.0) * ((6.0 * ((at_val * rgePow(yt_val, 3.0))
                                    + (ac_val * rgePow(yc_val, 3.0))
                                    + (au_val * rgePow(yu_val, 3.0))))
                                + (at_val * rgePow(yb_val, 2.0) * yt_val)
                                + (ac_val * rgePow(ys_val, 2.0) * yc_val)
                                + (au_val * rgePow(yd_val, 2.0) * yu_val)
                                + (ab_val * rgePow(yt_val, 2.0) * yb_val)
                                + (as_val * rgePow(yc_val, 2.0) * ys_val)
                                + (ad_val * rgePow(yu_val, 2.0) * yd_val))) // end trace
                        - (18.0 * rgePow(yt_val, 2.0) // Tr(au*Yu)
                        * ((at_val * yt_val)
                            + (ac_val * yc_val)
                            + (au_val * yu_val))) // end trace
                        - (rgePow(yb_val, 2.0) // Tr(6ad*Yd + 2ae*Ye)
                        * ((6.0 * ((ab_val * yb_val) + (as_val * ys_val)
                                    + (ad_val * yd_val)))
                            + (2.0 * ((atau_val * ytau_val)
                                    + (amu_val * ymu_val)
                                    + (ae_val * ye_val))))) // end trace
                        - (12.0 * yt_val * at_val // Tr(Yu^2)
                        * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                            + rgePow(yu_val, 2.0))) // end trace
                        - (yb_val * ab_val // Tr(6Yd^2 + 2Ye^2)
                        * ((6.0 * (rgePow(yb_val, 2.0)
                                    + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + (2.0 * (rgePow(ytau_val, 2.0)
                                    + rgePow(ymu_val, 2.0)
                                    + rgePow(ye_val, 2.0))))) // end trace
                        - (14.0 * rgePow(yt_val, 3.0) * at_val)
                        - (8.0 * rgePow(yb_val, 3.0) * ab_val)
                        - (2.0 * rgePow(yb_val, 2.0) * yt_val * at_val)
                        - (4.0 * yb_val * ab_val * rgePow(yt_val, 2.0))
                        + (((32.0 * rgePow(g3_val, 2.0))
                            + ((8.0 / 5.0) * rgePow(g1_val, 2.0))) // Tr(au*Yu)
                        * ((at_val * yt_val) + (ac_val * yc_val)
                            + (au_val * yu_val))) // end trace
                        + (((6.0 * rgePow(g2_val, 2.0))
                            + ((6.0 / 5.0) * rgePow(g1_val, 2.0)))
                        * yt_val * at_val)
                        + ((4.0 / 5.0) * rgePow(g1_val, 2.0) * yb_val * ab_val)
                        - (((32.0 * rgePow(g3_val, 2.0) * M3_val)
                            + ((8.0 / 5.0) * rgePow(g1_val, 2.0) * M1_val)) // Tr(Yu^2)
                        * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                            + rgePow(yu_val, 2.0))) // end trace
                        - (((12.0 * rgePow(g2_val, 2.0) * M2_val)
                            + ((4.0 / 5.0) * rgePow(g1_val, 2.0) * M1_val))
                        * rgePow(yt_val, 2.0))
                        - ((4.0 / 5.0) * rgePow(g1_val, 2.0) * M1_val
                        * rgePow(yb_val, 2.0))
                        + ((64.0 / 9.0) * rgePow(g3_val, 4.0) * M3_val)
                        - (16.0 * rgePow(g3_val, 2.0) * rgePow(g2_val, 2.0)
                        * (M3_val + M2_val))
                        - ((272.0 / 45.0) * rgePow(g3_val, 2.0)
                        * rgePow(g1_val, 2.0)
                        * (M3_val + M1_val))
                        - (30.0 * rgePow(g2_val, 4.0) * M2_val)
                        - (2.0 * rgePow(g2_val, 2.0) * rgePow(g1_val, 2.0)
                        * (M2_val + M1_val))
                        - ((5486.0 / 225.0) * rgePow(g1_val, 4.0) * M1_val))));

    Real dac_dt_2l = ((ac_val // Tr(3Yu^4 + (Yu^2*Yd^2))
                    * (((-3.0) * ((3.0 * (rgePow(yt_val, 4.0)
                                    + rgePow(yc_val, 4.0)
                                    + rgePow(yu_val, 4.0)))
                                + ((rgePow(yt_val, 2.0) * rgePow(yb_val, 2.0))
                                    + (rgePow(yc_val, 2.0)
                                    * rgePow(ys_val, 2.0))
                                    + (rgePow(yu_val, 2.0)
                                    * rgePow(yd_val, 2.0))))) // end trace
                        - (rgePow(ys_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (rgePow(yb_val, 2.0)
                                    + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + (rgePow(ytau_val, 2.0)
                                + rgePow(ymu_val, 2.0)
                                + rgePow(ye_val, 2.0)))) // end trace
                        - (15.0 * rgePow(yc_val, 2.0) // Tr(Yu^2)
                        * (rgePow(yt_val, 2.0)
                            + rgePow(yc_val, 2.0)
                            + rgePow(yu_val, 2.0))) // end trace
                        - (6.0 * rgePow(yc_val, 4.0))
                        - (2.0 * rgePow(ys_val, 4.0))
                        - (4.0 * rgePow(ys_val, 2.0) * rgePow(yc_val, 2.0))
                        + (((16.0 * rgePow(g3_val, 2.0))
                        + ((4.0 / 5.0) * rgePow(g1_val, 2.0))) // Tr(Yu^2)
                        * (rgePow(yt_val, 2.0)
                            + rgePow(yc_val, 2.0)
                            + rgePow(yu_val, 2.0))) // end trace
                        + (12.0 * rgePow(g2_val, 2.0)
                        * rgePow(yc_val, 2.0))
                        + ((2.0 / 5.0) * rgePow(g1_val, 2.0)
                        * rgePow(ys_val, 2.0))
                        - ((16.0 / 9.0) * rgePow(g3_val, 4.0))
                        + (8.0 * rgePow(g3_val, 2.0)
                        * rgePow(g2_val, 2.0))
                        + ((136.0 / 45.0) * rgePow(g3_val, 2.0)
                        * rgePow(g1_val, 2.0))
                        + ((15.0 / 2.0) * rgePow(g2_val, 4.0))
                        + (rgePow(g2_val, 2.0) * rgePow(g1_val, 2.0))
                        + ((2743.0 / 450.0) * rgePow(g1_val, 4.0))))
                + (yc_val // Tr(6au*Yu^3 + au*Yd^2*Yu + ad*Yu^2*Yd)
                    * (((-6.0) * ((6.0 * ((at_val * rgePow(yt_val, 3.0))
                                    + (ac_val * rgePow(yc_val, 3.0))
                                    + (au_val * rgePow(yu_val, 3.0))))
                                + (at_val * rgePow(yb_val, 2.0) * yt_val)
                                + (ac_val * rgePow(ys_val, 2.0) * yc_val)
                                + (au_val * rgePow(yd_val, 2.0) * yu_val)
                                + (ab_val * rgePow(yt_val, 2.0) * yb_val)
                                + (as_val * rgePow(yc_val, 2.0) * ys_val)
                                + (ad_val * rgePow(yu_val, 2.0) * yd_val))) // end trace
                        - (18.0 * rgePow(yc_val, 2.0) // Tr(au*Yu)
                        * ((at_val * yt_val)
                            + (ac_val * yc_val)
                            + (au_val * yu_val))) // end trace
                        - (rgePow(ys_val, 2.0) // Tr(6ad*Yd + 2ae*Ye)
                        * ((6.0 * ((ab_val * yb_val) + (as_val * ys_val)
                                    + (ad_val * yd_val)))
                            + (2.0 * ((atau_val * ytau_val)
                                    + (amu_val * ymu_val)
                                    + (ae_val * ye_val))))) // end trace
                        - (12.0 * yc_val * ac_val // Tr(Yu^2)
                        * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                            + rgePow(yu_val, 2.0))) // end trace
                        - (ys_val * as_val // Tr(6Yd^2 + 2Ye^2)
                        * ((6.0 * (rgePow(yb_val, 2.0)
                                    + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + (2.0 * (rgePow(ytau_val, 2.0)
                                    + rgePow(ymu_val, 2.0)
                                    + rgePow(ye_val, 2.0))))) // end trace
                        - (14.0 * rgePow(yc_val, 3.0) * ac_val)
                        - (8.0 * rgePow(ys_val, 3.0) * as_val)
                        - (2.0 * rgePow(ys_val, 2.0) * yc_val * ac_val)
                        - (4.0 * ys_val * as_val * rgePow(yc_val, 2.0))
                        + (((32.0 * rgePow(g3_val, 2.0))
                            + ((8.0 / 5.0) * rgePow(g1_val, 2.0))) // Tr(au*Yu)
                        * ((at_val * yt_val) + (ac_val * yc_val)
                            + (au_val * yu_val))) // end trace
                        + (((6.0 * rgePow(g2_val, 2.0))
                            + ((6.0 / 5.0) * rgePow(g1_val, 2.0)))
                        * yc_val * ac_val)
                        + ((4.0 / 5.0) * rgePow(g1_val, 2.0) * ys_val * as_val)
                        - (((32.0 * rgePow(g3_val, 2.0) * M3_val)
                            + ((8.0 / 5.0) * rgePow(g1_val, 2.0) * M1_val)) // Tr(Yu^2)
                        * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                            + rgePow(yu_val, 2.0))) // end trace
                        - (((12.0 * rgePow(g2_val, 2.0) * M2_val)
                            + ((4.0 / 5.0) * rgePow(g1_val, 2.0) * M1_val))
                        * rgePow(yc_val, 2.0))
                        - ((4.0 / 5.0) * rgePow(g1_val, 2.0) * M1_val
                        * rgePow(ys_val, 2.0))
                        + ((64.0 / 9.0) * rgePow(g3_val, 4.0) * M3_val)
                        - (16.0 * rgePow(g3_val, 2.0) * rgePow(g2_val, 2.0)
                        * (M3_val + M2_val))
                        - ((272.0 / 45.0) * rgePow(g3_val, 2.0)
                        * rgePow(g1_val, 2.0)
                        * (M3_val + M1_val))
                        - (30.0 * rgePow(g2_val, 4.0) * M2_val)
                        - (2.0 * rgePow(g2_val, 2.0) * rgePow(g1_val, 2.0)
                        * (M2_val + M1_val))
                        - ((5486.0 / 225.0) * rgePow(g1_val, 4.0) * M1_val))));

    Real dau_dt_2l = ((au_val // Tr(3Yu^4 + (Yu^2*Yd^2))
                    * (((-3.0) * ((3.0 * (rgePow(yt_val, 4.0)
                                    + rgePow(yc_val, 4.0)
                                    + rgePow(yu_val, 4.0)))
                                + ((rgePow(yt_val, 2.0) * rgePow(yb_val, 2.0))
                                    + (rgePow(yc_val, 2.0)
                                    * rgePow(ys_val, 2.0))
                                    + (rgePow(yu_val, 2.0)
                                    * rgePow(yd_val, 2.0))))) // end trace
                        - (rgePow(yd_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (rgePow(yb_val, 2.0)
                                    + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + (rgePow(ytau_val, 2.0)
                                + rgePow(ymu_val, 2.0)
                                + rgePow(ye_val, 2.0)))) // end trace
                        - (15.0 * rgePow(yu_val, 2.0) // Tr(Yu^2)
                        * (rgePow(yt_val, 2.0)
                            + rgePow(yc_val, 2.0)
                            + rgePow(yu_val, 2.0))) // end trace
                        - (6.0 * rgePow(yu_val, 4.0))
                        - (2.0 * rgePow(yd_val, 4.0))
                        - (4.0 * rgePow(yd_val, 2.0) * rgePow(yu_val, 2.0))
                        + (((16.0 * rgePow(g3_val, 2.0))
                        + ((4.0 / 5.0) * rgePow(g1_val, 2.0))) // Tr(Yu^2)
                        * (rgePow(yt_val, 2.0)
                            + rgePow(yc_val, 2.0)
                            + rgePow(yu_val, 2.0))) // end trace
                        + (12.0 * rgePow(g2_val, 2.0)
                        * rgePow(yu_val, 2.0))
                        + ((2.0 / 5.0) * rgePow(g1_val, 2.0)
                        * rgePow(yd_val, 2.0))
                        - ((16.0 / 9.0) * rgePow(g3_val, 4.0))
                        + (8.0 * rgePow(g3_val, 2.0)
                        * rgePow(g2_val, 2.0))
                        + ((136.0 / 45.0) * rgePow(g3_val, 2.0)
                        * rgePow(g1_val, 2.0))
                        + ((15.0 / 2.0) * rgePow(g2_val, 4.0))
                        + (rgePow(g2_val, 2.0) * rgePow(g1_val, 2.0))
                        + ((2743.0 / 450.0) * rgePow(g1_val, 4.0))))
                + (yu_val // Tr(6au*Yu^3 + au*Yd^2*Yu + ad*Yu^2*Yd)
                    * (((-6.0) * ((6.0 * ((at_val * rgePow(yt_val, 3.0))
                                    + (ac_val * rgePow(yc_val, 3.0))
                                    + (au_val * rgePow(yu_val, 3.0))))
                                + (at_val * rgePow(yb_val, 2.0) * yt_val)
                                + (ac_val * rgePow(ys_val, 2.0) * yc_val)
                                + (au_val * rgePow(yd_val, 2.0) * yu_val)
                                + (ab_val * rgePow(yt_val, 2.0) * yb_val)
                                + (as_val * rgePow(yc_val, 2.0) * ys_val)
                                + (ad_val * rgePow(yu_val, 2.0) * yd_val))) // end trace
                        - (18.0 * rgePow(yu_val, 2.0) // Tr(au*Yu)
                        * ((at_val * yt_val)
                            + (ac_val * yc_val)
                            + (au_val * yu_val))) // end trace
                        - (rgePow(yd_val, 2.0) // Tr(6ad*Yd + 2ae*Ye)
                        * ((6.0 * ((ab_val * yb_val) + (as_val * ys_val)
                                    + (ad_val * yd_val)))
                            + (2.0 * ((atau_val * ytau_val)
                                    + (amu_val * ymu_val)
                                    + (ae_val * ye_val))))) // end trace
                        - (12.0 * yu_val * au_val // Tr(Yu^2)
                        * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                            + rgePow(yu_val, 2.0))) // end trace
                        - (yd_val * ad_val // Tr(6Yd^2 + 2Ye^2)
                        * ((6.0 * (rgePow(yb_val, 2.0)
                                    + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + (2.0 * (rgePow(ytau_val, 2.0)
                                    + rgePow(ymu_val, 2.0)
                                    + rgePow(ye_val, 2.0))))) // end trace
                        - (14.0 * rgePow(yu_val, 3.0) * au_val)
                        - (8.0 * rgePow(yd_val, 3.0) * ad_val)
                        - (2.0 * rgePow(yd_val, 2.0) * yu_val * au_val)
                        - (4.0 * yd_val * ad_val * rgePow(yu_val, 2.0))
                        + (((32.0 * rgePow(g3_val, 2.0))
                            + ((8.0 / 5.0) * rgePow(g1_val, 2.0))) // Tr(au*Yu)
                        * ((at_val * yt_val) + (ac_val * yc_val)
                            + (au_val * yu_val))) // end trace
                        + (((6.0 * rgePow(g2_val, 2.0))
                            + ((6.0 / 5.0) * rgePow(g1_val, 2.0)))
                        * yu_val * au_val)
                        + ((4.0 / 5.0) * rgePow(g1_val, 2.0) * yd_val * ad_val)
                        - (((32.0 * rgePow(g3_val, 2.0) * M3_val)
                            + ((8.0 / 5.0) * rgePow(g1_val, 2.0) * M1_val)) // Tr(Yu^2)
                        * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                            + rgePow(yu_val, 2.0))) // end trace
                        - (((12.0 * rgePow(g2_val, 2.0) * M2_val)
                            + ((4.0 / 5.0) * rgePow(g1_val, 2.0) * M1_val))
                        * rgePow(yu_val, 2.0))
                        - ((4.0 / 5.0) * rgePow(g1_val, 2.0) * M1_val
                        * rgePow(yd_val, 2.0))
                        + ((64.0 / 9.0) * rgePow(g3_val, 4.0) * M3_val)
                        - (16.0 * rgePow(g3_val, 2.0) * rgePow(g2_val, 2.0)
                        * (M3_val + M2_val))
                        - ((272.0 / 45.0) * rgePow(g3_val, 2.0)
                        * rgePow(g1_val, 2.0)
                        * (M3_val + M1_val))
                        - (30.0 * rgePow(g2_val, 4.0) * M2_val)
                        - (2.0 * rgePow(g2_val, 2.0) * rgePow(g1_val, 2.0)
                        * (M2_val + M1_val))
                        - ((5486.0 / 225.0) * rgePow(g1_val, 4.0) * M1_val))));

    Real dab_dt_2l = ((ab_val // Tr(3Yd^4 + Yu^2*Yd^2 + Ye^4)
                    * (((-3.0) * ((3.0 * (rgePow(yb_val, 4.0)
                                    + rgePow(ys_val, 4.0)
                                    + rgePow(yd_val, 4.0)))
                                + ((rgePow(yt_val, 2.0) * rgePow(yb_val, 2.0))
                                    + (rgePow(yc_val, 2.0)
                                    * rgePow(ys_val, 2.0))
                                    + (rgePow(yu_val, 2.0)
                                    * rgePow(yd_val, 2.0)))
                                + rgePow(ytau_val, 4.0)
                                + rgePow(ymu_val, 4.0)
                                + rgePow(ye_val, 4.0))) // end trace
                        - (3.0 * rgePow(yt_val, 2.0) // Tr(Yu^2)
                        * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                            + rgePow(yu_val, 2.0))) // end trace
                        - (5 * rgePow(yb_val, 2.0) // Tr(3Yd^2+Ye^2)
                        * ((3.0 * (rgePow(yb_val, 2.0)
                                    + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + (rgePow(ytau_val, 2.0)
                                + rgePow(ymu_val, 2.0)
                                + rgePow(ye_val, 2.0)))) // end trace
                        - (6.0 * rgePow(yb_val, 4.0))
                        - (2.0 * rgePow(yt_val, 4.0))
                        - (4.0 * rgePow(yb_val, 2.0) * rgePow(yt_val, 2.0))
                        + (((16.0 * rgePow(g3_val, 2.0))
                        - ((2.0 / 5.0) * rgePow(g1_val, 2.0))) // Tr(Yd^2)
                        * (rgePow(yb_val, 2.0)
                            + rgePow(ys_val, 2.0)
                            + rgePow(yd_val, 2.0))) // end trace
                        + ((6.0 / 5.0) * rgePow(g1_val, 2.0) // Tr(Ye^2)
                        * (rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                            + rgePow(ye_val, 2.0))) // end trace
                        + ((4.0 / 5.0) * rgePow(g1_val, 2.0)
                        * rgePow(yt_val, 2.0))
                        + (((12.0 * rgePow(g2_val, 2.0))
                        + ((6.0 / 5.0) * rgePow(g1_val, 2.0)))
                        * rgePow(yb_val, 2.0))
                        - ((16.0 / 9.0) * rgePow(g3_val, 4.0))
                        + (8.0 * rgePow(g3_val, 2.0)
                        * rgePow(g2_val, 2.0))
                        + ((8.0 / 9.0) * rgePow(g3_val, 2.0)
                        * rgePow(g1_val, 2.0))
                        + ((15.0 / 2.0) * rgePow(g2_val, 4.0))
                        + (rgePow(g2_val, 2.0) * rgePow(g1_val, 2.0))
                        + ((287.0 / 90.0) * rgePow(g1_val, 4.0))))
                + (yb_val // Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
                    * (((-6.0) * ((6.0 * ((ab_val * rgePow(yb_val, 3.0))
                                    + (as_val * rgePow(ys_val, 3.0))
                                    + (ad_val * rgePow(yd_val, 3.0))))
                                + (at_val * rgePow(yb_val, 2.0) * yt_val)
                                + (ac_val * rgePow(ys_val, 2.0) * yc_val)
                                + (au_val * rgePow(yd_val, 2.0) * yu_val)
                                + (ab_val * rgePow(yt_val, 2.0) * yb_val)
                                + (as_val * rgePow(yc_val, 2.0) * ys_val)
                                + (ad_val * rgePow(yu_val, 2.0) * yd_val)
                                + (2.0 * ((atau_val * rgePow(ytau_val, 3.0))
                                        + (amu_val * rgePow(ymu_val, 3.0))
                                        + (ae_val * rgePow(ye_val, 3.0)))
                                    ))) // end trace
                        - (6.0 * rgePow(yt_val, 2.0) // Tr(au*Yu)
                        * ((at_val * yt_val)
                            + (ac_val * yc_val)
                            + (au_val * yu_val))) // end trace
                        - (6.0 * rgePow(yb_val, 2.0) // Tr(3ad*Yd + ae*Ye)
                        * ((3.0 * ((ab_val * yb_val) + (as_val * ys_val)
                                    + (ad_val * yd_val)))
                            + ((atau_val * ytau_val) + (amu_val * ymu_val)
                                + (ae_val * ye_val)))) // end trace
                        - (6.0 * yt_val * at_val // Tr(Yu^2)
                        * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                            + rgePow(yu_val, 2.0))) // end trace
                        - (4.0 * yb_val * ab_val // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (rgePow(yb_val, 2.0)
                                    + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + ((rgePow(ytau_val, 2.0)
                                    + rgePow(ymu_val, 2.0)
                                    + rgePow(ye_val, 2.0))))) // end trace
                        - (14.0 * rgePow(yb_val, 3.0) * ab_val)
                        - (8.0 * rgePow(yt_val, 3.0) * at_val)
                        - (4.0 * rgePow(yb_val, 2.0) * yt_val * at_val)
                        - (2.0 * yb_val * ab_val * rgePow(yt_val, 2.0))
                        + (((32.0 * rgePow(g3_val, 2.0))
                            - ((4.0 / 5.0) * rgePow(g1_val, 2.0))) // Tr(ad*Yd)
                        * ((ab_val * yb_val) + (as_val * ys_val)
                            + (ad_val * yd_val))) // end trace
                        + ((12.0 / 5.0) * rgePow(g1_val, 2.0) // Tr(ae*Ye)
                        * ((atau_val * ytau_val) + (amu_val * ymu_val)
                            + (ae_val * ye_val))) // end trace
                        + ((8.0 / 5.0) * rgePow(g1_val, 2.0) * yt_val * at_val)
                        + (((6.0 * rgePow(g2_val, 2.0))
                            + ((6.0 / 5.0) * rgePow(g1_val, 2.0)))
                        * yb_val * ab_val)
                        - (((32.0 * rgePow(g3_val, 2.0) * M3_val)
                            - ((4.0 / 5.0) * rgePow(g1_val, 2.0) * M1_val)) // Tr(Yd^2)
                        * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                            + rgePow(yd_val, 2.0))) // end trace
                        - ((12.0 / 5.0) * rgePow(g1_val, 2.0) * M1_val // Tr(Ye^2)
                        * (rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                            + rgePow(ye_val, 2.0))) // end trace
                        - (((12.0 * rgePow(g2_val, 2.0) * M2_val)
                            + ((8.0 / 5.0) * rgePow(g1_val, 2.0) * M1_val))
                        * rgePow(yb_val, 2.0))
                        - ((8.0 / 5.0) * rgePow(g1_val, 2.0) * M1_val
                        * rgePow(yt_val, 2.0))
                        + ((64.0 / 9.0) * rgePow(g3_val, 4.0) * M3_val)
                        - (16.0 * rgePow(g3_val, 2.0) * rgePow(g2_val, 2.0)
                        * (M3_val + M2_val))
                        - ((16.0 / 9.0) * rgePow(g3_val, 2.0)
                        * rgePow(g1_val, 2.0)
                        * (M3_val + M1_val))
                        - (30.0 * rgePow(g2_val, 4.0) * M2_val)
                        - (2.0 * rgePow(g2_val, 2.0) * rgePow(g1_val, 2.0)
                        * (M2_val + M1_val))
                        - ((574.0 / 45.0) * rgePow(g1_val, 4.0) * M1_val))));

    Real das_dt_2l = ((as_val // Tr(3Yd^4 + Yu^2*Yd^2 + Ye^4)
                    * (((-3.0) * ((3.0 * (rgePow(yb_val, 4.0)
                                    + rgePow(ys_val, 4.0)
                                    + rgePow(yd_val, 4.0)))
                                + ((rgePow(yt_val, 2.0) * rgePow(yb_val, 2.0))
                                    + (rgePow(yc_val, 2.0)
                                    * rgePow(ys_val, 2.0))
                                    + (rgePow(yu_val, 2.0)
                                    * rgePow(yd_val, 2.0)))
                                + rgePow(ytau_val, 4.0)
                                + rgePow(ymu_val, 4.0)
                                + rgePow(ye_val, 4.0))) // end trace
                        - (3.0 * rgePow(yc_val, 2.0) // Tr(Yu^2)
                        * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                            + rgePow(yu_val, 2.0))) // end trace
                        - (5 * rgePow(ys_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (rgePow(yb_val, 2.0)
                                    + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + (rgePow(ytau_val, 2.0)
                                + rgePow(ymu_val, 2.0)
                                + rgePow(ye_val, 2.0)))) // end trace
                        - (6.0 * rgePow(ys_val, 4.0))
                        - (2.0 * rgePow(yc_val, 4.0))
                        - (4.0 * rgePow(ys_val, 2.0) * rgePow(yc_val, 2.0))
                        + (((16.0 * rgePow(g3_val, 2.0))
                        - ((2.0 / 5.0) * rgePow(g1_val, 2.0))) // Tr(Yd^2)
                        * (rgePow(yb_val, 2.0)
                            + rgePow(ys_val, 2.0)
                            + rgePow(yd_val, 2.0))) // end trace
                        + ((6.0 / 5.0) * rgePow(g1_val, 2.0) // Tr(Ye^2)
                        * (rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                            + rgePow(ye_val, 2.0))) // end trace
                        + ((4.0 / 5.0) * rgePow(g1_val, 2.0)
                        * rgePow(yc_val, 2.0))
                        + (((12.0 * rgePow(g2_val, 2.0))
                        + ((6.0 / 5.0) * rgePow(g1_val, 2.0)))
                        * rgePow(ys_val, 2.0))
                        - ((16.0 / 9.0) * rgePow(g3_val, 4.0))
                        + (8.0 * rgePow(g3_val, 2.0)
                        * rgePow(g2_val, 2.0))
                        + ((8.0 / 9.0) * rgePow(g3_val, 2.0)
                        * rgePow(g1_val, 2.0))
                        + ((15.0 / 2.0) * rgePow(g2_val, 4.0))
                        + (rgePow(g2_val, 2.0) * rgePow(g1_val, 2.0))
                        + ((287.0 / 90.0) * rgePow(g1_val, 4.0))))
                + (ys_val // Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
                    * (((-6.0) * ((6.0 * ((ab_val * rgePow(yb_val, 3.0))
                                    + (as_val * rgePow(ys_val, 3.0))
                                    + (ad_val * rgePow(yd_val, 3.0))))
                                + (at_val * rgePow(yb_val, 2.0) * yt_val)
                                + (ac_val * rgePow(ys_val, 2.0) * yc_val)
                                + (au_val * rgePow(yd_val, 2.0) * yu_val)
                                + (ab_val * rgePow(yt_val, 2.0) * yb_val)
                                + (as_val * rgePow(yc_val, 2.0) * ys_val)
                                + (ad_val * rgePow(yu_val, 2.0) * yd_val)
                                + (2.0 * ((atau_val * rgePow(ytau_val, 3.0))
                                        + (amu_val * rgePow(ymu_val, 3.0))
                                        + (ae_val * rgePow(ye_val, 3.0)))
                                    ))) // end trace
                        - (6.0 * rgePow(yc_val, 2.0) // Tr(au*Yu)
                        * ((at_val * yt_val)
                            + (ac_val * yc_val)
                            + (au_val * yu_val))) // end trace
                        - (6.0 * rgePow(ys_val, 2.0) // Tr(3ad*Yd + ae*Ye)
                        * ((3.0 * ((ab_val * yb_val) + (as_val * ys_val)
                                    + (ad_val * yd_val)))
                            + ((atau_val * ytau_val) + (amu_val * ymu_val)
                                + (ae_val * ye_val)))) // end trace
                        - (6.0 * yc_val * ac_val // Tr(Yu^2)
                        * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                            + rgePow(yu_val, 2.0))) // end trace
                        - (4.0 * ys_val * as_val // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (rgePow(yb_val, 2.0)
                                    + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + ((rgePow(ytau_val, 2.0)
                                    + rgePow(ymu_val, 2.0)
                                    + rgePow(ye_val, 2.0))))) // end trace
                        - (14.0 * rgePow(ys_val, 3.0) * as_val)
                        - (8.0 * rgePow(yc_val, 3.0) * ac_val)
                        - (4.0 * rgePow(ys_val, 2.0) * yc_val * ac_val)
                        - (2.0 * ys_val * as_val * rgePow(yc_val, 2.0))
                        + (((32.0 * rgePow(g3_val, 2.0))
                            - ((4.0 / 5.0) * rgePow(g1_val, 2.0))) // Tr(ad*Yd)
                        * ((ab_val * yb_val) + (as_val * ys_val)
                            + (ad_val * yd_val))) // end trace
                        + ((12.0 / 5.0) * rgePow(g1_val, 2.0) // Tr(ae*Ye)
                        * ((atau_val * ytau_val) + (amu_val * ymu_val)
                            + (ae_val * ye_val))) // end trace
                        + ((8.0 / 5.0) * rgePow(g1_val, 2.0) * yc_val * ac_val)
                        + (((6.0 * rgePow(g2_val, 2.0))
                            + ((6.0 / 5.0) * rgePow(g1_val, 2.0)))
                        * ys_val * as_val)
                        - (((32.0 * rgePow(g3_val, 2.0) * M3_val)
                            - ((4.0 / 5.0) * rgePow(g1_val, 2.0) * M1_val)) // Tr(Yd^2)
                        * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                            + rgePow(yd_val, 2.0))) // end trace
                        - ((12.0 / 5.0) * rgePow(g1_val, 2.0) * M1_val // Tr(Ye^2)
                        * (rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                            + rgePow(ye_val, 2.0))) // end trace
                        - (((12.0 * rgePow(g2_val, 2.0) * M2_val)
                            + ((8.0 / 5.0) * rgePow(g1_val, 2.0) * M1_val))
                        * rgePow(ys_val, 2.0))
                        - ((8.0 / 5.0) * rgePow(g1_val, 2.0) * M1_val
                        * rgePow(yc_val, 2.0))
                        + ((64.0 / 9.0) * rgePow(g3_val, 4.0) * M3_val)
                        - (16.0 * rgePow(g3_val, 2.0) * rgePow(g2_val, 2.0)
                        * (M3_val + M2_val))
                        - ((16.0 / 9.0) * rgePow(g3_val, 2.0)
                        * rgePow(g1_val, 2.0)
                        * (M3_val + M1_val))
                        - (30.0 * rgePow(g2_val, 4.0) * M2_val)
                        - (2.0 * rgePow(g2_val, 2.0) * rgePow(g1_val, 2.0)
                        * (M2_val + M1_val))
                        - ((574.0 / 45.0) * rgePow(g1_val, 4.0) * M1_val))));

    Real dad_dt_2l = ((ad_val // Tr(3Yd^4 + Yu^2*Yd^2 + Ye^4)
                    * (((-3.0) * ((3.0 * (rgePow(yb_val, 4.0)
                                    + rgePow(ys_val, 4.0)
                                    + rgePow(yd_val, 4.0)))
                                + ((rgePow(yt_val, 2.0) * rgePow(yb_val, 2.0))
                                    + (rgePow(yc_val, 2.0)
                                    * rgePow(ys_val, 2.0))
                                    + (rgePow(yu_val, 2.0)
                                    * rgePow(yd_val, 2.0)))
                                + rgePow(ytau_val, 4.0)
                                + rgePow(ymu_val, 4.0)
                                + rgePow(ye_val, 4.0))) // end trace
                        - (3.0 * rgePow(yu_val, 2.0) // Tr(Yu^2)
                        * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                            + rgePow(yu_val, 2.0))) // end trace
                        - (5 * rgePow(yd_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (rgePow(yb_val, 2.0)
                                    + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + (rgePow(ytau_val, 2.0)
                                + rgePow(ymu_val, 2.0)
                                + rgePow(ye_val, 2.0)))) // end trace
                        - (6.0 * rgePow(yd_val, 4.0))
                        - (2.0 * rgePow(yu_val, 4.0))
                        - (4.0 * rgePow(yd_val, 2.0) * rgePow(yu_val, 2.0))
                        + (((16.0 * rgePow(g3_val, 2.0))
                        - ((2.0 / 5.0) * rgePow(g1_val, 2.0))) // Tr(Yd^2)
                        * (rgePow(yb_val, 2.0)
                            + rgePow(ys_val, 2.0)
                            + rgePow(yd_val, 2.0))) // end trace
                        + ((6.0 / 5.0) * rgePow(g1_val, 2.0) // Tr(Ye^2)
                        * (rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                            + rgePow(ye_val, 2.0))) // end trace
                        + ((4.0 / 5.0) * rgePow(g1_val, 2.0)
                        * rgePow(yu_val, 2.0))
                        + (((12.0 * rgePow(g2_val, 2.0))
                        + ((6.0 / 5.0) * rgePow(g1_val, 2.0)))
                        * rgePow(yd_val, 2.0))
                        - ((16.0 / 9.0) * rgePow(g3_val, 4.0))
                        + (8.0 * rgePow(g3_val, 2.0)
                        * rgePow(g2_val, 2.0))
                        + ((8.0 / 9.0) * rgePow(g3_val, 2.0)
                        * rgePow(g1_val, 2.0))
                        + ((15.0 / 2.0) * rgePow(g2_val, 4.0))
                        + (rgePow(g2_val, 2.0) * rgePow(g1_val, 2.0))
                        + ((287.0 / 90.0) * rgePow(g1_val, 4.0))))
                + (yd_val // Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
                    * (((-6.0) * ((6.0 * ((ab_val * rgePow(yb_val, 3.0))
                                    + (as_val * rgePow(ys_val, 3.0))
                                    + (ad_val * rgePow(yd_val, 3.0))))
                                + (at_val * rgePow(yb_val, 2.0) * yt_val)
                                + (ac_val * rgePow(ys_val, 2.0) * yc_val)
                                + (au_val * rgePow(yd_val, 2.0) * yu_val)
                                + (ab_val * rgePow(yt_val, 2.0) * yb_val)
                                + (as_val * rgePow(yc_val, 2.0) * ys_val)
                                + (ad_val * rgePow(yu_val, 2.0) * yd_val)
                                + (2.0 * ((atau_val * rgePow(ytau_val, 3.0))
                                        + (amu_val * rgePow(ymu_val, 3.0))
                                        + (ae_val * rgePow(ye_val, 3.0)))
                                    ))) // end trace
                        - (6.0 * rgePow(yu_val, 2.0) // Tr(au*Yu)
                        * ((at_val * yt_val)
                            + (ac_val * yc_val)
                            + (au_val * yu_val))) // end trace
                        - (6.0 * rgePow(yd_val, 2.0) // Tr(3ad*Yd + ae*Ye)
                        * ((3.0 * ((ab_val * yb_val) + (as_val * ys_val)
                                    + (ad_val * yd_val)))
                            + ((atau_val * ytau_val) + (amu_val * ymu_val)
                                + (ae_val * ye_val)))) // end trace
                        - (6.0 * yu_val * au_val // Tr(Yu^2)
                        * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                            + rgePow(yu_val, 2.0))) // end trace
                        - (4.0 * yd_val * ad_val // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (rgePow(yb_val, 2.0)
                                    + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + ((rgePow(ytau_val, 2.0)
                                    + rgePow(ymu_val, 2.0)
                                    + rgePow(ye_val, 2.0))))) // end trace
                        - (14.0 * rgePow(yd_val, 3.0) * ad_val)
                        - (8.0 * rgePow(yu_val, 3.0) * au_val)
                        - (4.0 * rgePow(yd_val, 2.0) * yu_val * au_val)
                        - (2.0 * yd_val * ad_val * rgePow(yu_val, 2.0))
                        + (((32.0 * rgePow(g3_val, 2.0))
                            - ((4.0 / 5.0) * rgePow(g1_val, 2.0))) // Tr(ad*Yd)
                        * ((ab_val * yb_val) + (as_val * ys_val)
                            + (ad_val * yd_val))) // end trace
                        + ((12.0 / 5.0) * rgePow(g1_val, 2.0) // Tr(ae*Ye)
                        * ((atau_val * ytau_val) + (amu_val * ymu_val)
                            + (ae_val * ye_val))) // end trace
                        + ((8.0 / 5.0) * rgePow(g1_val, 2.0) * yu_val * au_val)
                        + (((6.0 * rgePow(g2_val, 2.0))
                            + ((6.0 / 5.0) * rgePow(g1_val, 2.0)))
                        * yd_val * ad_val)
                        - (((32.0 * rgePow(g3_val, 2.0) * M3_val)
                            - ((4.0 / 5.0) * rgePow(g1_val, 2.0) * M1_val)) // Tr(Yd^2)
                        * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                            + rgePow(yd_val, 2.0))) // end trace
                        - ((12.0 / 5.0) * rgePow(g1_val, 2.0) * M1_val // Tr(Ye^2)
                        * (rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                            + rgePow(ye_val, 2.0))) // end trace
                        - (((12.0 * rgePow(g2_val, 2.0) * M2_val)
                            + ((8.0 / 5.0) * rgePow(g1_val, 2.0) * M1_val))
                        * rgePow(yd_val, 2.0))
                        - ((8.0 / 5.0) * rgePow(g1_val, 2.0) * M1_val
                        * rgePow(yu_val, 2.0))
                        + ((64.0 / 9.0) * rgePow(g3_val, 4.0) * M3_val)
                        - (16.0 * rgePow(g3_val, 2.0) * rgePow(g2_val, 2.0)
                        * (M3_val + M2_val))
                        - ((16.0 / 9.0) * rgePow(g3_val, 2.0)
                        * rgePow(g1_val, 2.0)
                        * (M3_val + M1_val))
                        - (30.0 * rgePow(g2_val, 4.0) * M2_val)
                        - (2.0 * rgePow(g2_val, 2.0) * rgePow(g1_val, 2.0)
                        * (M2_val + M1_val))
                        - ((574.0 / 45.0) * rgePow(g1_val, 4.0) * M1_val))));

    Real datau_dt_2l = ((atau_val // Tr(3Yd^4 + Yu^2*Yd^2 + Ye^4)
                    * (((-3.0) * ((3.0 * (rgePow(yb_val, 4.0)
                                    + rgePow(ys_val, 4.0)
                                    + rgePow(yd_val, 4.0)))
                                + ((rgePow(yt_val, 2.0)
                                    * rgePow(yb_val, 2.0))
                                    + (rgePow(yc_val, 2.0)
                                    * rgePow(ys_val, 2.0))
                                    + (rgePow(yu_val, 2.0)
                                    * rgePow(yd_val, 2.0)))
                                + rgePow(ytau_val, 4.0)
                                + rgePow(ymu_val, 4.0)
                                + rgePow(ye_val, 4.0))) // end trace
                        - (5 * rgePow(ytau_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (rgePow(yb_val, 2.0)
                                    + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + (rgePow(ytau_val, 2.0)
                                + rgePow(ymu_val, 2.0)
                                + rgePow(ye_val, 2.0)))) // end trace
                        - (6.0 * rgePow(ytau_val, 4.0))
                        + (((16.0 * rgePow(g3_val, 2.0))
                            - ((2.0 / 5.0) * rgePow(g1_val, 2.0))) // Tr(Yd^2)
                        * (rgePow(yb_val, 2.0)
                            + rgePow(ys_val, 2.0)
                            + rgePow(yd_val, 2.0))) // end trace
                        + ((6.0 / 5.0) * rgePow(g1_val, 2.0) // Tr(Ye^2)
                        * (rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                            + rgePow(ye_val, 2.0))) // end trace
                        + (((12.0 * rgePow(g2_val, 2.0))
                            - ((6.0 / 5.0) * rgePow(g1_val, 2.0)))
                        * rgePow(ytau_val, 2.0))
                        + ((15.0 / 2.0) * rgePow(g2_val, 4.0))
                        + ((9.0 / 5.0) * rgePow(g2_val, 2.0)
                        * rgePow(g1_val, 2.0))
                        + ((27.0 / 2.0) * rgePow(g1_val, 4.0))))
                    + (ytau_val // Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
                        * (((-6.0) * ((6.0 * ((ab_val * rgePow(yb_val, 3.0))
                                        + (as_val * rgePow(ys_val, 3.0))
                                        + (ad_val * rgePow(yd_val, 3.0))))
                                    + (at_val * rgePow(yb_val, 2.0) * yt_val)
                                    + (ac_val * rgePow(ys_val, 2.0) * yc_val)
                                    + (au_val * rgePow(yd_val, 2.0) * yu_val)
                                    + (ab_val * rgePow(yt_val, 2.0) * yb_val)
                                    + (as_val * rgePow(yc_val, 2.0) * ys_val)
                                    + (ad_val * rgePow(yu_val, 2.0) * yd_val)
                                    + (2.0 * ((atau_val
                                            * rgePow(ytau_val, 3.0))
                                            + (amu_val
                                                * rgePow(ymu_val, 3.0))
                                            + (ae_val
                                                * rgePow(ye_val, 3.0)))
                                    ))) // end trace
                        - (4.0 * ytau_val * atau_val // Tr(3Yd^2 + Ye^2)
                            * ((3.0 * (rgePow(yb_val, 2.0)
                                    + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                                + ((rgePow(ytau_val, 2.0)
                                    + rgePow(ymu_val, 2.0)
                                    + rgePow(ye_val, 2.0))))) // end trace
                        - (6.0 * rgePow(ytau_val, 2.0) // Tr(3ad*Yd + ae*Ye)
                            * ((3.0 * ((ab_val * yb_val) + (as_val * ys_val)
                                    + (ad_val * yd_val)))
                                + (atau_val * ytau_val)
                                + (amu_val * ymu_val)
                                + (ae_val * ye_val))) // end trace
                        - (14.0 * rgePow(ytau_val, 3.0) * atau_val)
                        + (((32.0 * rgePow(g3_val, 2.0))
                            - ((4.0 / 5.0) * rgePow(g1_val, 2.0))) // Tr(ad*Yd)
                            * ((ab_val * yb_val) + (as_val * ys_val)
                                + (ad_val * yd_val))) // end trace
                        + ((12.0 / 5.0) * rgePow(g1_val, 2.0) // Tr(ae*Ye)
                            * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                + (ae_val * ye_val))) // end trace
                        + (((6.0 * rgePow(g2_val, 2.0))
                            + ((6.0 / 5.0) * rgePow(g1_val, 2.0)))
                            * ytau_val * atau_val)
                        - (((32.0 * rgePow(g3_val, 2.0) * M3_val)
                            - ((4.0 / 5.0) * rgePow(g1_val, 2.0) * M1_val)) // Tr(Yd^2)
                            * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                                + rgePow(yd_val, 2.0))) // end trace
                        - ((12.0 / 5.0) * rgePow(g1_val, 2.0) * M1_val // Tr(Ye^2)
                            * (rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                                + rgePow(ye_val, 2.0))) // end trace
                        - (12.0 * rgePow(g2_val, 2.0) * M2_val
                            * rgePow(ytau_val, 2.0))
                        - (30.0 * rgePow(g2_val, 4.0) * M2_val)
                        - ((18.0 / 5.0) * rgePow(g2_val, 2.0)
                            * rgePow(g1_val, 2.0)
                            * (M1_val + M2_val))
                        - (54.0 * rgePow(g1_val, 4.0) * M1_val))));

    Real damu_dt_2l = ((amu_val // Tr(3Yd^4 + Yu^2*Yd^2 + Ye^4)
                    * (((-3.0) * ((3.0 * (rgePow(yb_val, 4.0)
                                    + rgePow(ys_val, 4.0)
                                    + rgePow(yd_val, 4.0)))
                                + ((rgePow(yt_val, 2.0)
                                    * rgePow(yb_val, 2.0))
                                    + (rgePow(yc_val, 2.0)
                                    * rgePow(ys_val, 2.0))
                                    + (rgePow(yu_val, 2.0)
                                    * rgePow(yd_val, 2.0)))
                                + rgePow(ytau_val, 4.0)
                                + rgePow(ymu_val, 4.0)
                                + rgePow(ye_val, 4.0))) // end trace
                        - (5 * rgePow(ymu_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (rgePow(yb_val, 2.0)
                                    + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + (rgePow(ytau_val, 2.0)
                                + rgePow(ymu_val, 2.0)
                                + rgePow(ye_val, 2.0)))) // end trace
                        - (6.0 * rgePow(ymu_val, 4.0))
                        + (((16.0 * rgePow(g3_val, 2.0))
                        - ((2.0 / 5.0) * rgePow(g1_val, 2.0))) // Tr(Yd^2)
                        * (rgePow(yb_val, 2.0)
                            + rgePow(ys_val, 2.0)
                            + rgePow(yd_val, 2.0))) // end trace
                        + ((6.0 / 5.0) * rgePow(g1_val, 2.0) // Tr(Ye^2)
                        * (rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                            + rgePow(ye_val, 2.0))) // end trace
                        + (((12.0 * rgePow(g2_val, 2.0))
                        - ((6.0 / 5.0) * rgePow(g1_val, 2.0)))
                        * rgePow(ymu_val, 2.0))
                        + ((15.0 / 2.0) * rgePow(g2_val, 4.0))
                        + ((9.0 / 5.0) * rgePow(g2_val, 2.0)
                        * rgePow(g1_val, 2.0))
                        + ((27.0 / 2.0) * rgePow(g1_val, 4.0))))
                    + (ymu_val // Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
                        * (((-6.0) * ((6.0 * ((ab_val * rgePow(yb_val, 3.0))
                                        + (as_val * rgePow(ys_val, 3.0))
                                        + (ad_val * rgePow(yd_val, 3.0))))
                                    + (at_val * rgePow(yb_val, 2.0) * yt_val)
                                    + (ac_val * rgePow(ys_val, 2.0) * yc_val)
                                    + (au_val * rgePow(yd_val, 2.0) * yu_val)
                                    + (ab_val * rgePow(yt_val, 2.0) * yb_val)
                                    + (as_val * rgePow(yc_val, 2.0) * ys_val)
                                    + (ad_val * rgePow(yu_val, 2.0) * yd_val)
                                    + (2.0 * ((atau_val * rgePow(ytau_val, 3.0))
                                        + (amu_val * rgePow(ymu_val, 3.0))
                                        + (ae_val * rgePow(ye_val, 3.0)))))) // end trace
                        - (4.0 * ymu_val * amu_val // Tr(3Yd^2 + Ye^2)
                            * ((3.0 * (rgePow(yb_val, 2.0)
                                    + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                                + ((rgePow(ytau_val, 2.0)
                                    + rgePow(ymu_val, 2.0)
                                    + rgePow(ye_val, 2.0))))) // end trace
                        - (6.0 * rgePow(ymu_val, 2.0) // Tr(3ad*Yd + ae*Ye)
                            * ((3.0 * ((ab_val * yb_val) + (as_val * ys_val)
                                    + (ad_val * yd_val)))
                                + (atau_val * ytau_val) + (amu_val * ymu_val)
                                + (ae_val * ye_val))) // end trace
                        - (14.0 * rgePow(ymu_val, 3.0) * amu_val)
                        + (((32.0 * rgePow(g3_val, 2.0))
                            - ((4.0 / 5.0) * rgePow(g1_val, 2.0))) // Tr(ad*Yd)
                            * ((ab_val * yb_val) + (as_val * ys_val)
                                + (ad_val * yd_val))) // end trace
                        + ((12.0 / 5.0) * rgePow(g1_val, 2.0) // Tr(ae*Ye)
                            * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                + (ae_val * ye_val))) // end trace
                        + (((6.0 * rgePow(g2_val, 2.0))
                            + ((6.0 / 5.0) * rgePow(g1_val, 2.0)))
                            * ymu_val * amu_val)
                        - (((32.0 * rgePow(g3_val, 2.0) * M3_val)
                            - ((4.0 / 5.0) * rgePow(g1_val, 2.0) * M1_val)) // Tr(Yd^2)
                            * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                                + rgePow(yd_val, 2.0))) // end trace
                        - ((12.0 / 5.0) * rgePow(g1_val, 2.0) * M1_val // Tr(Ye^2)
                            * (rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                                + rgePow(ye_val, 2.0))) // end trace
                        - (12.0 * rgePow(g2_val, 2.0) * M2_val
                            * rgePow(ymu_val, 2.0))
                        - (30.0 * rgePow(g2_val, 4.0) * M2_val)
                        - ((18.0 / 5.0) * rgePow(g2_val, 2.0) * rgePow(g1_val, 2.0)
                            * (M1_val + M2_val))
                        - (54.0 * rgePow(g1_val, 4.0) * M1_val))));

    Real dae_dt_2l = ((ae_val // Tr(3Yd^4 + Yu^2*Yd^2 + Ye^4)
                    * (((-3.0) * ((3.0 * (rgePow(yb_val, 4.0)
                                    + rgePow(ys_val, 4.0)
                                    + rgePow(yd_val, 4.0)))
                                + ((rgePow(yt_val, 2.0)
                                    * rgePow(yb_val, 2.0))
                                    + (rgePow(yc_val, 2.0)
                                    * rgePow(ys_val, 2.0))
                                    + (rgePow(yu_val, 2.0)
                                    * rgePow(yd_val, 2.0)))
                                + rgePow(ytau_val, 4.0)
                                + rgePow(ymu_val, 4.0)
                                + rgePow(ye_val, 4.0))) // end trace
                        - (5 * rgePow(ye_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (rgePow(yb_val, 2.0)
                                    + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + (rgePow(ytau_val, 2.0)
                                + rgePow(ymu_val, 2.0)
                                + rgePow(ye_val, 2.0)))) // end trace
                        - (6.0 * rgePow(ye_val, 4.0))
                        + (((16.0 * rgePow(g3_val, 2.0))
                        - ((2.0 / 5.0) * rgePow(g1_val, 2.0))) // Tr(Yd^2)
                        * (rgePow(yb_val, 2.0)
                            + rgePow(ys_val, 2.0)
                            + rgePow(yd_val, 2.0))) // end trace
                        + ((6.0 / 5.0) * rgePow(g1_val, 2.0) // Tr(Ye^2)
                        * (rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                            + rgePow(ye_val, 2.0))) // end trace
                        + (((12.0 * rgePow(g2_val, 2.0))
                        - ((6.0 / 5.0) * rgePow(g1_val, 2.0)))
                        * rgePow(ye_val, 2.0))
                        + ((15.0 / 2.0) * rgePow(g2_val, 4.0))
                        + ((9.0 / 5.0) * rgePow(g2_val, 2.0)
                        * rgePow(g1_val, 2.0))
                        + ((27.0 / 2.0) * rgePow(g1_val, 4.0))))
                + (ye_val // Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
                    * (((-6.0) * ((6.0 * ((ab_val * rgePow(yb_val, 3.0))
                                    + (as_val * rgePow(ys_val, 3.0))
                                    + (ad_val * rgePow(yd_val, 3.0))))
                                + (at_val * rgePow(yb_val, 2.0) * yt_val)
                                + (ac_val * rgePow(ys_val, 2.0) * yc_val)
                                + (au_val * rgePow(yd_val, 2.0) * yu_val)
                                + (ab_val * rgePow(yt_val, 2.0) * yb_val)
                                + (as_val * rgePow(yc_val, 2.0) * ys_val)
                                + (ad_val * rgePow(yu_val, 2.0) * yd_val)
                                + (2.0 * ((atau_val * rgePow(ytau_val, 3.0))
                                        + (amu_val * rgePow(ymu_val, 3.0))
                                        + (ae_val * rgePow(ye_val, 3.0)))))) // end trace
                        - (4.0 * ye_val * ae_val // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (rgePow(yb_val, 2.0)
                                    + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + ((rgePow(ytau_val, 2.0)
                                    + rgePow(ymu_val, 2.0)
                                    + rgePow(ye_val, 2.0))))) // end trace
                        - (6.0 * rgePow(ye_val, 2.0) // Tr(3ad*Yd + ae*Ye)
                        * ((3.0 * ((ab_val * yb_val) + (as_val * ys_val)
                                    + (ad_val * yd_val)))
                            + (atau_val * ytau_val) + (amu_val * ymu_val)
                            + (ae_val * ye_val))) // end trace
                        - (14.0 * rgePow(ye_val, 3.0) * ae_val)
                        + (((32.0 * rgePow(g3_val, 2.0))
                            - ((4.0 / 5.0) * rgePow(g1_val, 2.0))) // Tr(ad*Yd)
                        * ((ab_val * yb_val) + (as_val * ys_val)
                            + (ad_val * yd_val))) // end trace
                        + ((12.0 / 5.0) * rgePow(g1_val, 2.0) // Tr(ae*Ye)
                        * ((atau_val * ytau_val) + (amu_val * ymu_val)
                            + (ae_val * ye_val))) // end trace
                        + (((6.0 * rgePow(g2_val, 2.0))
                            + ((6.0 / 5.0) * rgePow(g1_val, 2.0)))
                        * ye_val * ae_val)
                        - (((32.0 * rgePow(g3_val, 2.0) * M3_val)
                            - ((4.0 / 5.0) * rgePow(g1_val, 2.0) * M1_val)) // Tr(Yd^2)
                        * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                            + rgePow(yd_val, 2.0))) // end trace
                        - ((12.0 / 5.0) * rgePow(g1_val, 2.0) * M1_val // Tr(Ye^2)
                        * (rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                            + rgePow(ye_val, 2.0))) // end trace
                        - (12.0 * rgePow(g2_val, 2.0) * M2_val
                        * rgePow(ye_val, 2.0))
                        - (30.0 * rgePow(g2_val, 4.0) * M2_val)
                        - ((18.0 / 5.0) * rgePow(g2_val, 2.0) * rgePow(g1_val, 2.0)
                        * (M1_val + M2_val))
                        - (54.0 * rgePow(g1_val, 4.0) * M1_val))));

    // Total soft trilinear coupling beta functions
    Real dat_dt = ((rgeLoopFactor * dat_dt_1l) + (rgeLoopFactorSquared * dat_dt_2l));

    Real dac_dt = ((rgeLoopFactor * dac_dt_1l) + (rgeLoopFactorSquared * dac_dt_2l));

    Real dau_dt = ((rgeLoopFactor * dau_dt_1l) + (rgeLoopFactorSquared * dau_dt_2l));

    Real dab_dt = ((rgeLoopFactor * dab_dt_1l) + (rgeLoopFactorSquared * dab_dt_2l));

    Real das_dt = ((rgeLoopFactor * das_dt_1l) + (rgeLoopFactorSquared * das_dt_2l));

    Real dad_dt = ((rgeLoopFactor * dad_dt_1l) + (rgeLoopFactorSquared * dad_dt_2l));

    Real datau_dt = ((rgeLoopFactor * datau_dt_1l) + (rgeLoopFactorSquared * datau_dt_2l));

    Real damu_dt = ((rgeLoopFactor * damu_dt_1l) + (rgeLoopFactorSquared * damu_dt_2l));

    Real dae_dt = ((rgeLoopFactor * dae_dt_1l) + (rgeLoopFactorSquared * dae_dt_2l));

    // Soft bilinear coupling b=B*mu
    /////////////////////////////////////////////////////////////////////////////////
    // 1-loop
    Real db_dt_1l = ((b_val // Tr(3Yu^2 + 3Yd^2 + Ye^2)
                * (((3.0 * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                            + rgePow(yu_val, 2.0) + rgePow(yb_val, 2.0)
                            + rgePow(ys_val, 2.0) + rgePow(yd_val, 2.0)))
                        + rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                        + rgePow(ye_val, 2.0)) // end trace
                    - (3.0 * rgePow(g2_val, 2.0))
                    - ((3.0 / 5.0) * rgePow(g1_val, 2.0))))
                + (mu_val // Tr(6au*Yu + 6ad*Yd + 2ae*Ye)
                    * (((6.0 * ((at_val * yt_val) + (ac_val * yc_val)
                            + (au_val * yu_val) + (ab_val * yb_val)
                            + (as_val * ys_val) + (ad_val * yd_val)))
                        + (2.0 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                + (ae_val * ye_val))))
                        + (6.0 * rgePow(g2_val, 2.0) * M2_val)
                        + ((6.0 / 5.0) * rgePow(g1_val, 2.0) * M1_val))));

    // 2-loop
    Real db_dt_2l = ((b_val // Tr(3Yu^4 + 3Yd^4 + 2Yu^2*Yd^2 + Ye^4)
                * (((-3.0) * ((3.0 *  (rgePow(yt_val, 4.0) + rgePow(yc_val, 4.0)
                                    + rgePow(yu_val, 4.0)
                                    + rgePow(yb_val, 4.0)
                                    + rgePow(ys_val, 4.0)
                                    + rgePow(yd_val, 4.0)))
                            + (2.0 * ((rgePow(yt_val, 2.0)
                                    * rgePow(yb_val, 2.0))
                                    + (rgePow(yc_val, 2.0)
                                        * rgePow(ys_val, 2.0))
                                    + (rgePow(yu_val, 2.0)
                                        * rgePow(yd_val, 2.0))))
                            + rgePow(ytau_val, 4.0) + rgePow(ymu_val, 4.0)
                            + rgePow(ye_val, 4.0))) // end trace
                    + (((16.0 * rgePow(g3_val, 2.0))
                        + ((4.0 / 5.0) * rgePow(g1_val, 2.0))) // Tr(Yu^2)
                        * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                        + rgePow(yu_val, 2.0))) // end trace
                    + (((16.0 * rgePow(g3_val, 2.0))
                        - ((2.0 / 5.0) * rgePow(g1_val, 2.0))) // Tr(Yd^2)
                        * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                        + rgePow(yd_val, 2.0))) // end trace
                    + ((6.0 / 5.0) * rgePow(g1_val, 2.0) // Tr(Ye^2)
                        * (rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                        + rgePow(ye_val, 2.0))) // end trace
                    + ((15.0 / 2.0) * rgePow(g2_val, 4.0))
                    + ((9.0 / 5.0) * rgePow(g1_val, 2.0) * rgePow(g2_val, 2.0))
                    + ((207.0 / 50.0) * rgePow(g1_val, 4.0))))
                + (mu_val * (((-12.0) // Tr(3au*Yu^3 + 3ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + ae*Ye^3)
                        * ((3.0 *  ((at_val * rgePow(yt_val, 3.0))
                                + (ac_val * rgePow(yc_val, 3.0))
                                + (au_val * rgePow(yu_val, 3.0))
                                + (ab_val * rgePow(yb_val, 3.0))
                                + (as_val * rgePow(ys_val, 3.0))
                                + (ad_val * rgePow(yd_val, 3.0))))
                        + ((at_val * rgePow(yb_val, 2.0) * yt_val)
                            + (ac_val * rgePow(ys_val, 2.0) * yc_val)
                            + (au_val * rgePow(yd_val, 2.0) * yu_val))
                        + ((ab_val * rgePow(yt_val, 2.0) * yb_val)
                            + (as_val * rgePow(yc_val, 2.0) * ys_val)
                            + (ad_val * rgePow(yu_val, 2.0) * yd_val))
                        + ((atau_val * rgePow(ytau_val, 3.0))
                            + (amu_val * rgePow(ymu_val, 3.0))
                            + (ae_val * rgePow(ye_val, 3.0))))) // end trace
                    + (((32.0 * rgePow(g3_val, 2.0))
                        + ((8.0 / 5.0) * rgePow(g1_val, 2.0))) // Tr(au*Yu)
                        * ((at_val * yt_val) + (ac_val * yc_val)
                            + (au_val * yu_val))) // end trace
                        + (((32.0 * rgePow(g3_val, 2.0))
                        - ((4.0 / 5.0) * rgePow(g1_val, 2.0))) // Tr(ad*Yd)
                        * ((ab_val * yb_val) + (as_val * ys_val)
                            + (ad_val * yd_val))) // end trace
                        + ((12.0 / 5.0) * rgePow(g1_val, 2.0) // Tr(ae*Ye)
                        * ((atau_val * ytau_val) + (amu_val * ymu_val)
                            + (ae_val * ye_val))) // end trace
                        - (((32.0 * rgePow(g3_val, 2.0) * M3_val)
                        + ((8.0 / 5.0) * rgePow(g1_val, 2.0) * M1_val)) // Tr(Yu^2)
                        * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                            + rgePow(yu_val, 2.0)))
                        - (((32.0 * rgePow(g3_val, 2.0) * M3_val) // end trace
                        - ((4.0 / 5.0) * rgePow(g1_val, 2.0) * M1_val)) // Tr(Yd^2)
                        * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                            + rgePow(yd_val, 2.0))) // end trace
                        - ((12.0 / 5.0) * rgePow(g1_val, 2.0) * M1_val // Tr(Ye^2)
                        * (rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                            + rgePow(ye_val, 2.0))) // end trace
                        - (30.0 * rgePow(g2_val, 4.0) * M2_val)
                        - ((18.0 / 5.0) * rgePow(g1_val, 2.0)
                        * rgePow(g2_val, 2.0)
                        * (M1_val + M2_val))
                        - ((414.0 / 25.0) * rgePow(g1_val, 4.0) * M1_val))));

    // Total b beta function
    Real db_dt = ((rgeLoopFactor * db_dt_1l) + (rgeLoopFactorSquared * db_dt_2l));

    // Scalar squared masses
    /////////////////////////////////////////////////////////////////////////////////
    // Introduce S, S', and sigma terms
    Real S_val = (mHu_sq_val - mHd_sq_val + mQ3_sq_val + mQ2_sq_val + mQ1_sq_val
            - mL3_sq_val - mL2_sq_val - mL1_sq_val
            - (2.0 * (mU3_sq_val + mU2_sq_val + mU1_sq_val))
            + mD3_sq_val + mD2_sq_val + mD1_sq_val
            + mE3_sq_val + mE2_sq_val + mE1_sq_val);
    
    Real Spr_val = ((( (-1.0) * ((((3.0 * mHu_sq_val) + mQ3_sq_val)
                        * rgePow(yt_val, 2.0))
                        + (((3.0 * mHu_sq_val) + mQ2_sq_val)
                            * rgePow(yc_val, 2.0))
                        + (((3.0 * mHu_sq_val) + mQ1_sq_val)
                            * rgePow(yu_val, 2.0))))
                + (4.0 * rgePow(yt_val, 2.0) * mU3_sq_val)
                + (4.0 * rgePow(yc_val, 2.0) * mU2_sq_val)
                + (4.0 * rgePow(yu_val, 2.0) * mU1_sq_val)
                + ((((3.0 * mHd_sq_val) - mQ3_sq_val) * rgePow(yb_val, 2.0))
                    + (((3.0 * mHd_sq_val) - mQ2_sq_val)
                        * rgePow(ys_val, 2.0))
                    + (((3.0 * mHd_sq_val) - mQ1_sq_val)
                        * rgePow(yd_val, 2.0)))
                - (2.0 * ((mD3_sq_val * rgePow(yb_val, 2.0))
                        + (mD2_sq_val * rgePow(ys_val, 2.0))
                        + (mD1_sq_val * rgePow(yd_val, 2.0))))
                + (((mHd_sq_val + mL3_sq_val) * rgePow(ytau_val, 2.0))
                    + ((mHd_sq_val + mL2_sq_val) * rgePow(ymu_val, 2.0))
                    + ((mHd_sq_val + mL1_sq_val) * rgePow(ye_val, 2.0)))
                - (2.0 * ((rgePow(ytau_val, 2.0) * mE3_sq_val)
                        + (rgePow(ymu_val, 2.0) * mE2_sq_val)
                        + (rgePow(ye_val, 2.0) * mE1_sq_val)))) // end trace
                + ((((3.0 / 2.0) * rgePow(g2_val, 2.0))
                    + ((3.0 / 10.0) * rgePow(g1_val, 2.0)))
                    * (mHu_sq_val - mHd_sq_val // Tr(mL^2)
                        - (mL3_sq_val + mL2_sq_val + mL1_sq_val))) // end trace
                + ((((8.0 / 3.0) * rgePow(g3_val, 2.0))
                    + ((3.0 / 2.0) * rgePow(g2_val, 2.0))
                    + ((1.0 / 30.0) * rgePow(g1_val, 2.0))) // Tr(mQ^2)
                    * (mQ3_sq_val + mQ2_sq_val + mQ1_sq_val)) // end trace
                - ((((16.0 / 3.0) * rgePow(g3_val, 2.0))
                    + ((16.0 / 15.0) * rgePow(g1_val, 2.0))) // Tr (mU^2)
                    * (mU3_sq_val + mU2_sq_val + mU1_sq_val)) // end trace
                + ((((8.0 / 3.0) * rgePow(g3_val, 2.0))
                    + ((2.0 / 15.0) * rgePow(g1_val, 2.0))) // Tr(mD^2)
                    * (mD3_sq_val + mD2_sq_val + mD1_sq_val)) // end trace
                + ((6.0 / 5.0) * rgePow(g1_val, 2.0) // Tr(mE^2)
                    * (mE3_sq_val + mE2_sq_val + mE1_sq_val))); // end trace

    Real sigma1 = ((1.0 / 5.0) * rgePow(g1_val, 2.0)
            * ((3.0 * (mHu_sq_val + mHd_sq_val)) // Tr(mQ^2 + 3mL^2 + 8mU^2 + 2mD^2 + 6mE^2)
                + mQ3_sq_val + mQ2_sq_val + mQ1_sq_val
                + (3.0 * (mL3_sq_val + mL2_sq_val + mL1_sq_val))
                + (8.0 * (mU3_sq_val + mU2_sq_val + mU1_sq_val))
                + (2.0 * (mD3_sq_val + mD2_sq_val + mD1_sq_val))
                + (6.0 * (mE3_sq_val + mE2_sq_val + mE1_sq_val)))); // end trace

    Real sigma2 = (rgePow(g2_val, 2.0)
            * (mHu_sq_val + mHd_sq_val // Tr(3mQ^2 + mL^2)
                + (3.0 * (mQ3_sq_val + mQ2_sq_val + mQ1_sq_val))
                + mL3_sq_val + mL2_sq_val + mL1_sq_val)); // end trace

    Real sigma3 = (rgePow(g3_val, 2.0) // Tr(2mQ^2 + mU^2 + mD^2)
            * ((2.0 * (mQ3_sq_val + mQ2_sq_val + mQ1_sq_val))
                + mU3_sq_val + mU2_sq_val + mU1_sq_val
                + mD3_sq_val + mD2_sq_val + mD1_sq_val)); // end trace
    
    // 1-loop parts of masses
    Real dmHu_sq_dt_1l = ((6.0 // Tr((mHu^2 + mQ^2) * Yu^2 + Yu^2.0 * mU^2 + au^2)
                        * (((mHu_sq_val + mQ3_sq_val) * rgePow(yt_val, 2.0))
                        + ((mHu_sq_val + mQ2_sq_val)
                            * rgePow(yc_val, 2.0))
                        + ((mHu_sq_val + mQ1_sq_val)
                            * rgePow(yu_val, 2.0))
                        + (mU3_sq_val * rgePow(yt_val, 2.0))
                        + (mU2_sq_val * rgePow(yc_val, 2.0))
                        + (mU1_sq_val * rgePow(yu_val, 2.0))
                        + rgePow(at_val, 2.0) + rgePow(ac_val, 2.0)
                        + rgePow(au_val, 2.0))) // end trace
                        - (6.0 * rgePow(g2_val, 2.0) * rgePow(M2_val, 2.0))
                        - ((6.0 / 5.0) * rgePow(g1_val, 2.0)
                        * rgePow(M1_val, 2.0))
                        + ((3.0 / 5.0) * rgePow(g1_val, 2.0) * S_val));

    Real dmHd_sq_dt_1l = ((6.0 * (((mHd_sq_val + mQ3_sq_val)
                            * rgePow(yb_val, 2.0))
                            + ((mHd_sq_val + mQ2_sq_val)
                                * rgePow(ys_val, 2.0))
                            + ((mHd_sq_val + mQ1_sq_val)
                                * rgePow(yd_val, 2.0)))
                        + (6.0 * ((mD3_sq_val * rgePow(yb_val, 2.0))
                                + (mD2_sq_val * rgePow(ys_val, 2.0))
                                + (mD1_sq_val * rgePow(yd_val, 2.0))))
                        + (2.0 * (((mHd_sq_val + mL3_sq_val)
                                * rgePow(ytau_val, 2.0))
                                + ((mHd_sq_val + mL2_sq_val)
                                    * rgePow(ymu_val, 2.0))
                                + ((mHd_sq_val + mL1_sq_val)
                                    * rgePow(ye_val, 2.0))))
                        + (2.0 * ((mE3_sq_val * rgePow(ytau_val, 2.0))
                                + (mE2_sq_val * rgePow(ymu_val, 2.0))
                                + (mE1_sq_val * rgePow(ye_val, 2.0))))
                        + (6.0 * (rgePow(ab_val, 2.0) + rgePow(as_val, 2.0)
                                + rgePow(ad_val, 2.0)))
                        + (2.0 * (rgePow(atau_val, 2.0) + rgePow(amu_val, 2.0)
                                + rgePow(ae_val, 2.0)))) // end trace
                        - (6.0 * rgePow(g2_val, 2.0) * rgePow(M2_val, 2.0))
                        - ((6.0 / 5.0) * rgePow(g1_val, 2.0)
                        * rgePow(M1_val, 2.0))
                        - ((3.0 / 5.0) * rgePow(g1_val, 2.0) * S_val));

    Real dmQ3_sq_dt_1l = (((mQ3_sq_val + (2.0 * mHu_sq_val))
                        * rgePow(yt_val, 2.0))
                        + ((mQ3_sq_val + (2.0 * mHd_sq_val))
                        * rgePow(yb_val, 2.0))
                        + ((rgePow(yt_val, 2.0) + rgePow(yb_val, 2.0))
                        * mQ3_sq_val)
                        + (2.0 * rgePow(yt_val, 2.0) * mU3_sq_val)
                        + (2.0 * rgePow(yb_val, 2.0) * mD3_sq_val)
                        + (2.0 * rgePow(at_val, 2.0))
                        + (2.0 * rgePow(ab_val, 2.0))
                        - ((32.0 / 3.0) * rgePow(g3_val, 2.0)
                        * rgePow(M3_val, 2.0))
                        - (6.0 * rgePow(g2_val, 2.0) * rgePow(M2_val, 2.0))
                        - ((2.0 / 15.0) * rgePow(g1_val, 2.0)
                        * rgePow(M1_val, 2.0))
                        + ((1.0 / 5.0) * rgePow(g1_val, 2.0) * S_val));

    Real dmQ2_sq_dt_1l = (((mQ2_sq_val + (2.0 * mHu_sq_val))
                        * rgePow(yc_val, 2.0))
                        + ((mQ2_sq_val + (2.0 * mHd_sq_val))
                        * rgePow(ys_val, 2.0))
                        + ((rgePow(yc_val, 2.0) + rgePow(ys_val, 2.0))
                        * mQ2_sq_val)
                        + (2.0 * rgePow(yc_val, 2.0) * mU2_sq_val)
                        + (2.0 * rgePow(ys_val, 2.0) * mD2_sq_val)
                        + (2.0 * rgePow(ac_val, 2.0))
                        + (2.0 * rgePow(as_val, 2.0))
                        - ((32.0 / 3.0) * rgePow(g3_val, 2.0)
                        * rgePow(M3_val, 2.0))
                        - (6.0 * rgePow(g2_val, 2.0) * rgePow(M2_val, 2.0))
                        - ((2.0 / 15.0) * rgePow(g1_val, 2.0)
                        * rgePow(M1_val, 2.0))
                        + ((1.0 / 5.0) * rgePow(g1_val, 2.0) * S_val));

    Real dmQ1_sq_dt_1l = (((mQ1_sq_val + (2.0 * mHu_sq_val))
                        * rgePow(yu_val, 2.0))
                        + ((mQ1_sq_val + (2.0 * mHd_sq_val))
                        * rgePow(yd_val, 2.0))
                        + ((rgePow(yu_val, 2.0)
                        + rgePow(yd_val, 2.0)) * mQ1_sq_val)
                        + (2.0 * rgePow(yu_val, 2.0) * mU1_sq_val)
                        + (2.0 * rgePow(yd_val, 2.0) * mD1_sq_val)
                        + (2.0 * rgePow(au_val, 2.0))
                        + (2.0 * rgePow(ad_val, 2.0))
                        - ((32.0 / 3.0) * rgePow(g3_val, 2.0)
                        * rgePow(M3_val, 2.0))
                        - (6.0 * rgePow(g2_val, 2.0) * rgePow(M2_val, 2.0))
                        - ((2.0 / 15.0) * rgePow(g1_val, 2.0)
                        * rgePow(M1_val, 2.0))
                        + ((1.0 / 5.0) * rgePow(g1_val, 2.0) * S_val));

    // Left leptons
    Real dmL3_sq_dt_1l = (((mL3_sq_val + (2.0 * mHd_sq_val))
                        * rgePow(ytau_val, 2.0))
                        + (2.0 * rgePow(ytau_val, 2.0) * mE3_sq_val)
                        + (rgePow(ytau_val, 2.0) * mL3_sq_val)
                        + (2.0 * rgePow(atau_val, 2.0))
                        - (6.0 * rgePow(g2_val, 2.0) * rgePow(M2_val, 2.0))
                        - ((6.0 / 5.0) * rgePow(g1_val, 2.0)
                        * rgePow(M1_val, 2.0))
                        - ((3.0 / 5.0) * rgePow(g1_val, 2.0) * S_val));

    Real dmL2_sq_dt_1l = (((mL2_sq_val + (2.0 * mHd_sq_val))
                        * rgePow(ymu_val, 2.0))
                        + (2.0 * rgePow(ymu_val, 2.0) * mE2_sq_val)
                        + (rgePow(ymu_val, 2.0) * mL2_sq_val)
                        + (2.0 * rgePow(amu_val, 2.0))
                        - (6.0 * rgePow(g2_val, 2.0) * rgePow(M2_val, 2.0))
                        - ((6.0 / 5.0) * rgePow(g1_val, 2.0)
                        * rgePow(M1_val, 2.0))
                        - ((3.0 / 5.0) * rgePow(g1_val, 2.0) * S_val));

    Real dmL1_sq_dt_1l = (((mL1_sq_val + (2.0 * mHd_sq_val))
                        * rgePow(ye_val, 2.0))
                        + (2.0 * rgePow(ye_val, 2.0) * mE1_sq_val)
                        + (rgePow(ye_val, 2.0) * mL1_sq_val)
                        + (2.0 * rgePow(ae_val, 2.0))
                        - (6.0 * rgePow(g2_val, 2.0) * rgePow(M2_val, 2.0))
                        - ((6.0 / 5.0) * rgePow(g1_val, 2.0)
                        * rgePow(M1_val, 2.0))
                        - ((3.0 / 5.0) * rgePow(g1_val, 2.0) * S_val));

    // Right up-type squarks
    Real dmU3_sq_dt_1l = ((2.0 * (mU3_sq_val + (2.0 * mHu_sq_val))
                        * rgePow(yt_val, 2.0))
                        + (4.0 * rgePow(yt_val, 2.0) * mQ3_sq_val)
                        + (2.0 * rgePow(yt_val, 2.0) * mU3_sq_val)
                        + (4.0 * rgePow(at_val, 2.0))
                        - ((32.0 / 3.0) * rgePow(g3_val, 2.0)
                        * rgePow(M3_val, 2.0))
                        - ((32.0 / 15.0) * rgePow(g1_val, 2.0)
                        * rgePow(M1_val, 2.0))
                        - ((4.0 / 5.0) * rgePow(g1_val, 2.0) * S_val));

    Real dmU2_sq_dt_1l = ((2.0 * (mU2_sq_val + (2.0 * mHu_sq_val))
                        * rgePow(yc_val, 2.0))
                        + (4.0 * rgePow(yc_val, 2.0) * mQ2_sq_val)
                        + (2.0 * rgePow(yc_val, 2.0) * mU2_sq_val)
                        + (4.0 * rgePow(ac_val, 2.0))
                        - ((32.0 / 3.0) * rgePow(g3_val, 2.0)
                        * rgePow(M3_val, 2.0))
                        - ((32.0 / 15.0) * rgePow(g1_val, 2.0)
                        * rgePow(M1_val, 2.0))
                        - ((4.0 / 5.0) * rgePow(g1_val, 2.0) * S_val));

    Real dmU1_sq_dt_1l = ((2.0 * (mU1_sq_val + (2.0 * mHu_sq_val))
                        * rgePow(yu_val, 2.0))
                        + (4.0 * rgePow(yu_val, 2.0) * mQ1_sq_val)
                        + (2.0 * rgePow(yu_val, 2.0) * mU1_sq_val)
                        + (4.0 * rgePow(au_val, 2.0))
                        - ((32.0 / 3.0) * rgePow(g3_val, 2.0)
                        * rgePow(M3_val, 2.0))
                        - ((32.0 / 15.0) * rgePow(g1_val, 2.0)
                        * rgePow(M1_val, 2.0))
                        - ((4.0 / 5.0) * rgePow(g1_val, 2.0) * S_val));

    // Right down-type squarks
    Real dmD3_sq_dt_1l = ((2.0 * (mD3_sq_val + (2.0 * mHd_sq_val))
                        * rgePow(yb_val, 2.0))
                        + (4.0 * rgePow(yb_val, 2.0) * mQ3_sq_val)
                        + (2.0 * rgePow(yb_val, 2.0) * mD3_sq_val)
                        + (4.0 * rgePow(ab_val, 2.0))
                        - ((32.0 / 3.0) * rgePow(g3_val, 2.0)
                        * rgePow(M3_val, 2.0))
                        - ((8.0 / 15.0) * rgePow(g1_val, 2.0)
                        * rgePow(M1_val, 2.0))
                        + ((2.0 / 5.0) * rgePow(g1_val, 2.0) * S_val));

    Real dmD2_sq_dt_1l = ((2.0 * (mD2_sq_val + (2.0 * mHd_sq_val))
                        * rgePow(ys_val, 2.0))
                        + (4.0 * rgePow(ys_val, 2.0) * mQ2_sq_val)
                        + (2.0 * rgePow(ys_val, 2.0) * mD2_sq_val)
                        + (4.0 * rgePow(as_val, 2.0))
                        - ((32.0 / 3.0) * rgePow(g3_val, 2.0)
                        * rgePow(M3_val, 2.0))
                        - ((8.0 / 15.0) * rgePow(g1_val, 2.0)
                        * rgePow(M1_val, 2.0))
                        + ((2.0 / 5.0) * rgePow(g1_val, 2.0) * S_val));

    Real dmD1_sq_dt_1l = ((2.0 * (mD1_sq_val + (2.0 * mHd_sq_val))
                        * rgePow(yd_val, 2.0))
                        + (4.0 * rgePow(yd_val, 2.0) * mQ1_sq_val)
                        + (2.0 * rgePow(yd_val, 2.0) * mD1_sq_val)
                        + (4.0 * rgePow(ad_val, 2.0))
                        - ((32.0 / 3.0) * rgePow(g3_val, 2.0)
                        * rgePow(M3_val, 2.0))
                        - ((8.0 / 15.0) * rgePow(g1_val, 2.0)
                        * rgePow(M1_val, 2.0))
                        + ((2.0 / 5.0) * rgePow(g1_val, 2.0) * S_val));

    // Right leptons
    Real dmE3_sq_dt_1l = ((2.0 * (mE3_sq_val + (2.0 * mHd_sq_val))
                        * rgePow(ytau_val, 2.0))
                        + (4.0 * rgePow(ytau_val, 2.0) * mL3_sq_val)
                        + (2.0 * rgePow(ytau_val, 2.0) * mE3_sq_val)
                        + (4.0 * rgePow(atau_val, 2.0))
                        - ((24.0 / 5.0) * rgePow(g1_val, 2.0)
                        * rgePow(M1_val, 2.0))
                        + ((6.0 / 5.0) * rgePow(g1_val, 2.0) * S_val));

    Real dmE2_sq_dt_1l = ((2.0 * (mE2_sq_val + (2.0 * mHd_sq_val))
                        * rgePow(ymu_val, 2.0))
                        + (4.0 * rgePow(ymu_val, 2.0) * mL2_sq_val)
                        + (2.0 * rgePow(ymu_val, 2.0) * mE2_sq_val)
                        + (4.0 * rgePow(amu_val, 2.0))
                        - ((24.0 / 5.0) * rgePow(g1_val, 2.0)
                        * rgePow(M1_val, 2.0))
                        + ((6.0 / 5.0) * rgePow(g1_val, 2.0) * S_val));

    Real dmE1_sq_dt_1l = ((2.0 * (mE1_sq_val + (2.0 * mHd_sq_val))
                        * rgePow(ye_val, 2.0))
                        + (4.0 * rgePow(ye_val, 2.0) * mL1_sq_val)
                        + (2.0 * rgePow(ye_val, 2.0) * mE1_sq_val)
                        + (4.0 * rgePow(ae_val, 2.0))
                        - ((24.0 / 5.0) * rgePow(g1_val, 2.0)
                        * rgePow(M1_val, 2.0))
                        + ((6.0 / 5.0) * rgePow(g1_val, 2.0) * S_val));

    // 2-loop parts of masses
    Real dmHu_sq_dt_2l = (((-6.0)  // Tr(6(mHu^2 + mQ^2)*Yu^4 + 6Yu^4 * mU^2 + (mHu^2 + mHd^2 + mQ^2) * Yu^2.0 * Yd^2 + Yu^2.0 * Yd^2.0 * mU^2 + Yu^2.0 * Yd^2.0 * mQ^2 + Yu^2.0 * Yd^2.0 * mD^2 + 12au^2.0 * Yu^2 + ad^2.0 * Yu^2 + Yd^2.0 * au^2 + 2ad * Yd * Yu * au)
                        * ((6.0 * (((mHu_sq_val + mQ3_sq_val)
                                * rgePow(yt_val, 4.0))
                                + ((mHu_sq_val + mQ2_sq_val)
                                    * rgePow(yc_val, 4.0))
                                + ((mHu_sq_val + mQ1_sq_val)
                                    * rgePow(yu_val, 4.0))))
                        + (6.0 * ((mU3_sq_val * rgePow(yt_val, 4.0))
                                    + (mU2_sq_val * rgePow(yc_val, 4.0))
                                    + (mU1_sq_val * rgePow(yu_val, 4.0))))
                        + ((mHu_sq_val + mHd_sq_val + mQ3_sq_val)
                            * rgePow(yt_val, 2.0) * rgePow(yb_val, 2.0))
                        + ((mHu_sq_val + mHd_sq_val + mQ2_sq_val)
                            * rgePow(yc_val, 2.0) * rgePow(ys_val, 2.0))
                        + ((mHu_sq_val + mHd_sq_val + mQ1_sq_val)
                            * rgePow(yu_val, 2.0) * rgePow(yd_val, 2.0))
                        + ((mU3_sq_val + mQ3_sq_val + mD3_sq_val)
                            * rgePow(yt_val, 2.0) * rgePow(yb_val, 2.0))
                        + ((mU2_sq_val + mQ2_sq_val + mD2_sq_val)
                            * rgePow(yc_val, 2.0) * rgePow(ys_val, 2.0))
                        + ((mU1_sq_val + mQ1_sq_val + mD1_sq_val)
                            * rgePow(yu_val, 2.0) * rgePow(yd_val, 2.0))
                        + (12.0 * ((rgePow(at_val, 2.0)
                                    * rgePow(yt_val, 2.0))
                                    + (rgePow(ac_val, 2.0)
                                    * rgePow(yc_val, 2.0))
                                    + (rgePow(au_val, 2.0)
                                    * rgePow(yu_val, 2.0))))
                        + (rgePow(ab_val, 2.0) * rgePow(yt_val, 2.0))
                        + (rgePow(as_val, 2.0) * rgePow(yc_val, 2.0))
                        + (rgePow(ad_val, 2.0) * rgePow(yu_val, 2.0))
                        + (rgePow(yb_val, 2.0) * rgePow(at_val, 2.0))
                        + (rgePow(ys_val, 2.0) * rgePow(ac_val, 2.0))
                        + (rgePow(yd_val, 2.0) * rgePow(au_val, 2.0))
                        + (2.0 * ((yb_val * ab_val * at_val * yt_val)
                                    + (ys_val * as_val * ac_val * yc_val)
                                    + (yd_val * ad_val * au_val * yu_val))))) // end trace
                        + (((32.0 * rgePow(g3_val, 2.0))
                        + ((8.0 / 5.0) * rgePow(g1_val, 2.0)))  // Tr((mHu^2 + mQ^2 + mU^2) * Yu^2 + au^2)
                        * (((mHu_sq_val + mQ3_sq_val + mU3_sq_val)
                            * rgePow(yt_val, 2.0))
                            + ((mHu_sq_val + mQ2_sq_val + mU2_sq_val)
                                * rgePow(yc_val, 2.0))
                            + ((mHu_sq_val + mQ1_sq_val + mU1_sq_val)
                                * rgePow(yu_val, 2.0))
                            + rgePow(at_val, 2.0) + rgePow(ac_val, 2.0)
                            + rgePow(au_val, 2.0))) // end trace
                        + (32.0 * rgePow(g3_val, 2.0)
                        * ((2.0 * rgePow(M3_val, 2.0) // Tr(Yu^2)
                            * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                                + rgePow(yu_val, 2.0))) // end trace
                            - (2.0 * M3_val // Tr(Yu*au)
                                * ((yt_val * at_val) + (yc_val * ac_val)
                                    + (yu_val * au_val))))) // end trace
                        + ((8.0 / 5.0) * rgePow(g1_val, 2.0)
                        * ((2.0 * rgePow(M1_val, 2.0) // Tr(Yu^2)
                            * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                                + rgePow(yu_val, 2.0))) // end trace
                            - (2.0 * M1_val // Tr(Yu*au)
                                * ((yt_val * at_val) + (yc_val * ac_val)
                                    + (yu_val * au_val))))) // end trace
                        + ((6.0 / 5.0) * rgePow(g1_val, 2.0) * Spr_val)
                        + (33.0 * rgePow(g2_val, 4.0) * rgePow(M2_val, 2.0))
                        + ((18.0 / 5.0) * rgePow(g2_val, 2.0)
                        * rgePow(g1_val, 2.0)
                        * (rgePow(M2_val, 2.0) + rgePow(M1_val, 2.0)
                            + (M1_val * M2_val)))
                        + ((621.0 / 25.0) * rgePow(g1_val, 4.0)
                        * rgePow(M1_val, 2.0))
                        + (3.0 * rgePow(g2_val, 2.0) * sigma2)
                        + ((3.0 / 5.0) * rgePow(g1_val, 2.0) * sigma1));

    Real dmHd_sq_dt_2l = (((-6.0)  // Tr(6(mHd^2 + mQ^2)*Yd^4 + 6Yd^4 * mD^2 + (mHu^2 + mHd^2 + mQ^2) * Yu^2.0 * Yd^2 + Yu^2.0 * Yd^2.0 * mU^2 + Yu^2.0 * Yd^2.0 * mQ^2 + Yu^2.0 * Yd^2.0 * mD^2 + 2(mHd^2 + mL^2) * Ye^4 + 2Ye^4 * mE^2 + 12ad^2.0 * Yd^2 + ad^2.0 * Yu^2 + Yd^2.0 * au^2 + 2ad * Yd * Yu * au + 4ae^2.0 * Ye^2)
                        * ((6.0 * (((mHd_sq_val + mQ3_sq_val)
                                * rgePow(yb_val, 4.0))
                                + ((mHd_sq_val + mQ2_sq_val)
                                    * rgePow(ys_val, 4.0))
                                + ((mHd_sq_val + mQ1_sq_val)
                                    * rgePow(yd_val, 4.0))))
                        + (6.0 * ((mD3_sq_val * rgePow(yb_val, 4.0))
                                    + (mD2_sq_val * rgePow(ys_val, 4.0))
                                    + (mD1_sq_val * rgePow(yd_val, 4.0))))
                        + ((mHu_sq_val + mHd_sq_val + mQ3_sq_val)
                            * rgePow(yt_val, 2.0) * rgePow(yb_val, 2.0))
                        + ((mHu_sq_val + mHd_sq_val + mQ2_sq_val)
                            * rgePow(yc_val, 2.0) * rgePow(ys_val, 2.0))
                        + ((mHu_sq_val + mHd_sq_val + mQ1_sq_val)
                            * rgePow(yu_val, 2.0) * rgePow(yd_val, 2.0))
                        + ((mU3_sq_val + mQ3_sq_val + mD3_sq_val)
                            * rgePow(yt_val, 2.0) * rgePow(yb_val, 2.0))
                        + ((mU2_sq_val + mQ2_sq_val + mD2_sq_val)
                            * rgePow(yc_val, 2.0) * rgePow(ys_val, 2.0))
                        + ((mU1_sq_val + mQ1_sq_val + mD1_sq_val)
                            * rgePow(yu_val, 2.0) * rgePow(yd_val, 2.0))
                        + (2.0 * (((mHd_sq_val + mL3_sq_val + mE3_sq_val)
                                    * rgePow(ytau_val, 4.0))
                                    + ((mHd_sq_val + mL2_sq_val + mE2_sq_val)
                                    * rgePow(ymu_val, 4.0))
                                    + ((mHd_sq_val + mL1_sq_val + mE1_sq_val)
                                    * rgePow(ye_val, 4.0))))
                        + (12.0 * ((rgePow(ab_val, 2.0)
                                    * rgePow(yb_val, 2.0))
                                    + (rgePow(as_val, 2.0)
                                    * rgePow(ys_val, 2.0))
                                    + (rgePow(ad_val, 2.0)
                                    * rgePow(yd_val, 2.0))))
                        + (rgePow(ab_val, 2.0) * rgePow(yt_val, 2.0))
                        + (rgePow(as_val, 2.0) * rgePow(yc_val, 2.0))
                        + (rgePow(ad_val, 2.0) * rgePow(yu_val, 2.0))
                        + (rgePow(yb_val, 2.0) * rgePow(at_val, 2.0))
                        + (rgePow(ys_val, 2.0) * rgePow(ac_val, 2.0))
                        + (rgePow(yd_val, 2.0) * rgePow(au_val, 2.0))
                        + (2.0 * ((yb_val * ab_val * at_val * yt_val)
                                    + (ys_val * as_val * ac_val * yc_val)
                                    + (yd_val * ad_val * au_val * yu_val)
                                    + (2.0 * ((rgePow(atau_val, 2.0)
                                            * rgePow(ytau_val, 2.0))
                                        + (rgePow(amu_val, 2.0)
                                            * rgePow(ymu_val, 2.0))
                                        + (rgePow(ae_val, 2.0)
                                            * rgePow(ye_val, 2.0)))))))) // end trace
                        + (((32.0 * rgePow(g3_val, 2.0))
                        - ((4.0 / 5.0) * rgePow(g1_val, 2.0)))  // Tr((mHd^2 + mQ^2 + mD^2) * Yd^2 + ad^2)
                        * (((mHu_sq_val + mQ3_sq_val + mD3_sq_val)
                            * rgePow(yb_val, 2.0))
                            + ((mHu_sq_val + mQ2_sq_val + mD2_sq_val)
                                * rgePow(ys_val, 2.0))
                            + ((mHu_sq_val + mQ1_sq_val + mD1_sq_val)
                                * rgePow(yd_val, 2.0))
                            + rgePow(ab_val, 2.0) + rgePow(as_val, 2.0)
                            + rgePow(ad_val, 2.0))) // end trace
                        + (32.0 * rgePow(g3_val, 2.0)
                        * ((2.0 * rgePow(M3_val, 2.0) // Tr(Yd^2)
                            * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                                + rgePow(yd_val, 2.0))) // end trace
                            - (2.0 * M3_val  // Tr(Yd*ad)
                                * ((yb_val * ab_val) + (ys_val * as_val)
                                    + (yd_val * ad_val))))) // end trace
                        - ((4.0 / 5.0) * rgePow(g1_val, 2.0)
                        * ((2.0 * rgePow(M1_val, 2.0) // Tr(Yd^2)
                            * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                                + rgePow(yd_val, 2.0))) // end trace
                            - (2.0 * M1_val  // Tr(Yd*ad)
                                * ((yb_val * ab_val) + (ys_val * as_val)
                                    + (yd_val * ad_val))))) // end trace
                        + ((12.0 / 5.0) * rgePow(g1_val, 2.0)
                        * (( // Tr((mHd^2 + mL^2 + mE^2) * Ye^2 + ae^2)
                            ((mHd_sq_val + mL3_sq_val + mE3_sq_val)
                            * rgePow(ytau_val, 2.0))
                            + ((mHd_sq_val + mL2_sq_val + mE2_sq_val)
                                * rgePow(ymu_val, 2.0))
                            + ((mHd_sq_val + mL1_sq_val + mE1_sq_val)
                                * rgePow(ye_val, 2.0))
                            + rgePow(atau_val, 2.0) + rgePow(amu_val, 2.0)
                            + rgePow(ae_val, 2.0)) // end trace
                            + (2.0 * rgePow(M1_val, 2.0) // Tr(Ye^2)
                                * (rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                                    + rgePow(ye_val, 2.0))) // end trace
                            - (2.0 * M1_val  // Tr(ae * Ye)
                                * ((atau_val * ytau_val)
                                    + (amu_val * ymu_val)
                                    + (ae_val * ye_val))) // end trace
                            ))
                        - ((6.0 / 5.0) * rgePow(g1_val, 2.0) * Spr_val)
                        + (33.0 * rgePow(g2_val, 4.0) * rgePow(M2_val, 2.0))
                        + ((18.0 / 5.0) * rgePow(g2_val, 2.0)
                        * rgePow(g1_val, 2.0)
                        * (rgePow(M2_val, 2.0) + rgePow(M1_val, 2.0)
                            + (M1_val * M2_val)))
                        + ((621.0 / 25.0) * rgePow(g1_val, 4.0)
                        * rgePow(M1_val, 2.0))
                        + (3.0 * rgePow(g2_val, 2.0) * sigma2)
                        + ((3.0 / 5.0) * rgePow(g1_val, 2.0) * sigma1));

        // Left squarks
    Real dmQ3_sq_dt_2l = (((-8.0)* (mQ3_sq_val + mHu_sq_val + mU3_sq_val)
                        * rgePow(yt_val, 4.0))
                        - (8.0 * (mQ3_sq_val + mHd_sq_val + mD3_sq_val)
                        * rgePow(yb_val, 4.0))
                        - (rgePow(yt_val, 2.0)
                        * ((2.0 * mQ3_sq_val) + (2.0 * mU3_sq_val)
                            + (4.0 * mHu_sq_val)) // Tr(3Yu^2)
                        * 3.0 * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                                + rgePow(yu_val, 2.0))) // end trace
                        - (rgePow(yb_val, 2.0)
                        * ((2.0 * mQ3_sq_val) + (2.0 * mD3_sq_val)
                            + (4.0 * mHd_sq_val)) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                            + rgePow(ye_val, 2.0))) // end trace
                        - (6.0 * rgePow(yt_val, 2.0) // Tr((mQ^2 + mU^2)*Yu^2)
                        * (((mQ3_sq_val + mU3_sq_val)
                            * rgePow(yt_val, 2.0))
                            + ((mQ2_sq_val + mU2_sq_val)
                                * rgePow(yc_val, 2.0))
                            + ((mQ1_sq_val + mU1_sq_val)
                                * rgePow(yu_val, 2.0)))) // end trace
                        - (rgePow(yb_val, 2.0) // Tr(6(mQ^2 + mD^2)*Yd^2 + 2(mL^2 + mE^2)*Ye^2)
                        * ((6.0 * (((mQ3_sq_val + mD3_sq_val)
                                    * rgePow(yb_val, 2.0))
                                    + ((mQ2_sq_val + mD2_sq_val)
                                    * rgePow(ys_val, 2.0))
                                    + ((mQ1_sq_val + mD1_sq_val)
                                    * rgePow(yd_val, 2.0))))
                            + (2.0 * (((mL3_sq_val + mE3_sq_val)
                                    * rgePow(ytau_val, 2.0))
                                    + ((mL2_sq_val + mE2_sq_val)
                                    * rgePow(ymu_val, 2.0))
                                    + ((mL1_sq_val + mE1_sq_val)
                                    * rgePow(ye_val, 2.0)))) // end trace
                            ))
                        - (16.0 * rgePow(yt_val, 2.0) * rgePow(at_val, 2.0))
                        - (16.0 * rgePow(yb_val, 2.0) * rgePow(ab_val, 2.0))
                        - (rgePow(at_val, 2.0) // Tr(6Yu^2)
                        * 6.0 * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                                + rgePow(yu_val, 2.0))) // end trace
                        - (rgePow(yt_val, 2.0) // Tr(6au^2)
                        * 6.0 * (rgePow(at_val, 2.0) + rgePow(ac_val, 2.0)
                                + rgePow(au_val, 2.0))) // end trace
                        - (at_val * yt_val // Tr(12Yu*au)
                        * 12.0 * ((yt_val * at_val) + (yc_val * ac_val)
                                + (yu_val * au_val))) // end trace
                        - (rgePow(ab_val, 2.0) // Tr(6Yd^2 + 2Ye^2)
                        * ((6.0 * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + (2.0 * (rgePow(ytau_val, 2.0)
                                    + rgePow(ymu_val, 2.0)
                                    + rgePow(ye_val, 2.0))))) // end trace
                        - (rgePow(yb_val, 2.0) // Tr(6ad^2 + 2ae^2)
                        * ((6.0 * (rgePow(ab_val, 2.0) + rgePow(as_val, 2.0)
                                    + rgePow(ad_val, 2.0)))
                            + (2.0 * (rgePow(atau_val, 2.0)
                                    + rgePow(amu_val, 2.0)
                                    + rgePow(ae_val, 2.0))))) // end trace
                        - (2.0 * ab_val * yb_val // Tr(6Yd*ad + 2Ye*ae)
                        * ((6.0 * ((yb_val * ab_val) + (ys_val * as_val)
                                    + (yd_val * ad_val)))
                            + (2.0 * ((ytau_val * atau_val)
                                    + (ymu_val * amu_val)
                                    + (ye_val * ae_val))))) // end trace
                        + ((2.0 / 5.0) * rgePow(g1_val, 2.0)
                        * ((4.0 * (mQ3_sq_val + mHu_sq_val + mU3_sq_val)
                            * rgePow(yt_val, 2.0))
                            + (4.0 * rgePow(at_val, 2.0))
                            - (8.0 * M1_val * at_val * yt_val)
                            + (8.0 * rgePow(M1_val, 2.0) * rgePow(yt_val, 2.0))
                            + (2.0 * (mQ3_sq_val + mHd_sq_val + mD3_sq_val)
                                * rgePow(yb_val, 2.0))
                            + (2.0 * rgePow(ab_val, 2.0))
                            - (4.0 * M1_val * ab_val * yb_val)
                            + (4.0 * rgePow(M1_val, 2.0)
                                * rgePow(yb_val, 2.0))))
                        + ((2.0 / 5.0) * rgePow(g1_val, 2.0) * Spr_val)
                        - ((128.0 / 3.0) * rgePow(g3_val, 4.0)
                        * rgePow(M3_val, 2.0))
                        + (32.0 * rgePow(g3_val, 2.0) * rgePow(g2_val, 2.0)
                        * (rgePow(M3_val, 2.0) + rgePow(M2_val, 2.0)
                            + (M2_val * M3_val)))
                        + ((32.0 / 45.0) * rgePow(g3_val, 2.0)
                        * rgePow(g1_val, 2.0)
                        * (rgePow(M3_val, 2.0) + rgePow(M1_val, 2.0)
                            + (M3_val * M1_val)))
                        + (33.0 * rgePow(g2_val, 4.0) * rgePow(M2_val, 2.0))
                        + ((2.0 / 5.0) * rgePow(g2_val, 2.0) * rgePow(g1_val, 2.0)
                        * (rgePow(M1_val, 2.0) + rgePow(M2_val, 2.0)
                            + (M1_val * M2_val)))
                        + ((199.0 / 75.0) * rgePow(g1_val, 4.0) * rgePow(M1_val, 2.0))
                        + ((16.0 / 3.0) * rgePow(g3_val, 2.0) * sigma3)
                        + (3.0 * rgePow(g2_val, 2.0) * sigma2)
                        + ((1.0 / 15.0) * rgePow(g1_val, 2.0) * sigma1));

    Real dmQ2_sq_dt_2l = (((-8.0)* (mQ2_sq_val + mHu_sq_val + mU2_sq_val)
                        * rgePow(yc_val, 4.0))
                        - (8.0 * (mQ2_sq_val + mHd_sq_val + mD2_sq_val)
                        * rgePow(ys_val, 4.0))
                        - (rgePow(yc_val, 2.0)
                        * ((2.0 * mQ2_sq_val) + (2.0 * mU2_sq_val)
                            + (4.0 * mHu_sq_val)) // Tr(3Yu^2)
                        * 3.0 * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                                + rgePow(yu_val, 2.0))) // end trace
                        - (rgePow(ys_val, 2.0)
                        * ((2.0 * mQ2_sq_val) + (2.0 * mD2_sq_val)
                            + (4.0 * mHd_sq_val)) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                            + rgePow(ye_val, 2.0))) // end trace
                        - (6.0 * rgePow(yc_val, 2.0) // Tr((mQ^2 + mU^2)*Yu^2)
                        * (((mQ3_sq_val + mU3_sq_val)
                            * rgePow(yt_val, 2.0))
                            + ((mQ2_sq_val + mU2_sq_val)
                                * rgePow(yc_val, 2.0))
                            + ((mQ1_sq_val + mU1_sq_val)
                                * rgePow(yu_val, 2.0)))) // end trace
                        - (rgePow(ys_val, 2.0) // Tr(6(mQ^2 + mD^2)*Yd^2 + 2(mL^2 + mE^2)*Ye^2)
                        * ((6.0 * (((mQ3_sq_val + mD3_sq_val)
                                    * rgePow(yb_val, 2.0))
                                    + ((mQ2_sq_val + mD2_sq_val)
                                    * rgePow(ys_val, 2.0))
                                    + ((mQ1_sq_val + mD1_sq_val)
                                    * rgePow(yd_val, 2.0))))
                            + (2.0 * (((mL3_sq_val + mE3_sq_val)
                                    * rgePow(ytau_val, 2.0))
                                    + ((mL2_sq_val + mE2_sq_val)
                                    * rgePow(ymu_val, 2.0))
                                    + ((mL1_sq_val + mE1_sq_val)
                                    * rgePow(ye_val, 2.0)))) // end trace
                            ))
                        - (16.0 * rgePow(yc_val, 2.0) * rgePow(ac_val, 2.0))
                        - (16.0 * rgePow(ys_val, 2.0) * rgePow(as_val, 2.0))
                        - (rgePow(ac_val, 2.0) // Tr(6Yu^2)
                        * 6.0 * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                                + rgePow(yu_val, 2.0))) // end trace
                        - (rgePow(yc_val, 2.0) // Tr(6au^2)
                        * 6.0 * (rgePow(at_val, 2.0) + rgePow(ac_val, 2.0)
                                + rgePow(au_val, 2.0))) // end trace
                        - (ac_val * yc_val // Tr(12Yu*au)
                        * 12.0 * ((yt_val * at_val) + (yc_val * ac_val)
                                + (yu_val * au_val))) // end trace
                        - (rgePow(as_val, 2.0) // Tr(6Yd^2 + 2Ye^2)
                        * ((6.0 * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + (2.0 * (rgePow(ytau_val, 2.0)
                                    + rgePow(ymu_val, 2.0)
                                    + rgePow(ye_val, 2.0))))) // end trace
                        - (rgePow(ys_val, 2.0) // Tr(6ad^2 + 2ae^2)
                        * ((6.0 * (rgePow(ab_val, 2.0) + rgePow(as_val, 2.0)
                                    + rgePow(ad_val, 2.0)))
                            + (2.0 * (rgePow(atau_val, 2.0)
                                    + rgePow(amu_val, 2.0)
                                    + rgePow(ae_val, 2.0))))) // end trace
                        - (2.0 * as_val * ys_val // Tr(6Yd*ad + 2Ye*ae)
                        * ((6.0 * ((yb_val * ab_val) + (ys_val * as_val)
                                    + (yd_val * ad_val)))
                            + (2.0 * ((ytau_val * atau_val)
                                    + (ymu_val * amu_val)
                                    + (ye_val * ae_val))))) // end trace
                        + ((2.0 / 5.0) * rgePow(g1_val, 2.0)
                        * ((4.0 * (mQ2_sq_val + mHu_sq_val + mU2_sq_val)
                            * rgePow(yc_val, 2.0))
                            + (4.0 * rgePow(ac_val, 2.0))
                            - (8.0 * M1_val * ac_val * yc_val)
                            + (8.0 * rgePow(M1_val, 2.0) * rgePow(yc_val, 2.0))
                            + (2.0 * (mQ2_sq_val + mHd_sq_val + mD2_sq_val)
                                * rgePow(ys_val, 2.0))
                            + (2.0 * rgePow(as_val, 2.0))
                            - (4.0 * M1_val * as_val * ys_val)
                            + (4.0 * rgePow(M1_val, 2.0)
                                * rgePow(ys_val, 2.0))))
                        + ((2.0 / 5.0) * rgePow(g1_val, 2.0) * Spr_val)
                        - ((128.0 / 3.0) * rgePow(g3_val, 4.0)
                        * rgePow(M3_val, 2.0))
                        + (32.0 * rgePow(g3_val, 2.0) * rgePow(g2_val, 2.0)
                        * (rgePow(M3_val, 2.0) + rgePow(M2_val, 2.0)
                            + (M2_val * M3_val)))
                        + ((32.0 / 45.0) * rgePow(g3_val, 2.0)
                        * rgePow(g1_val, 2.0)
                        * (rgePow(M3_val, 2.0) + rgePow(M1_val, 2.0)
                            + (M3_val * M1_val)))
                        + (33.0 * rgePow(g2_val, 4.0) * rgePow(M2_val, 2.0))
                        + ((2.0 / 5.0) * rgePow(g2_val, 2.0) * rgePow(g1_val, 2.0)
                        * (rgePow(M1_val, 2.0) + rgePow(M2_val, 2.0)
                            + (M1_val * M2_val)))
                        + ((199.0 / 75.0) * rgePow(g1_val, 4.0) * rgePow(M1_val, 2.0))
                        + ((16.0 / 3.0) * rgePow(g3_val, 2.0) * sigma3)
                        + (3.0 * rgePow(g2_val, 2.0) * sigma2)
                        + ((1.0 / 15.0) * rgePow(g1_val, 2.0) * sigma1));

    Real dmQ1_sq_dt_2l = (((-8.0)* (mQ1_sq_val + mHu_sq_val + mU1_sq_val)
                        * rgePow(yu_val, 4.0))
                        - (8.0 * (mQ1_sq_val + mHd_sq_val + mD1_sq_val)
                        * rgePow(yd_val, 4.0))
                        - (rgePow(yu_val, 2.0)
                        * ((2.0 * mQ1_sq_val) + (2.0 * mU1_sq_val)
                            + (4.0 * mHu_sq_val)) // Tr(3Yu^2)
                        * 3.0 * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                                + rgePow(yu_val, 2.0))) // end trace
                        - (rgePow(yd_val, 2.0)
                        * ((2.0 * mQ1_sq_val) + (2.0 * mD1_sq_val)
                            + (4.0 * mHd_sq_val)) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                            + rgePow(ye_val, 2.0))) // end trace
                        - (6.0 * rgePow(yu_val, 2.0) // Tr((mQ^2 + mU^2)*Yu^2)
                        * (((mQ3_sq_val + mU3_sq_val)
                            * rgePow(yt_val, 2.0))
                            + ((mQ2_sq_val + mU2_sq_val)
                                * rgePow(yc_val, 2.0))
                            + ((mQ1_sq_val + mU1_sq_val)
                                * rgePow(yu_val, 2.0)))) // end trace
                        - (rgePow(yd_val, 2.0) // Tr(6(mQ^2 + mD^2)*Yd^2 + 2(mL^2 + mE^2)*Ye^2)
                        * ((6.0 * (((mQ3_sq_val + mD3_sq_val)
                                    * rgePow(yb_val, 2.0))
                                    + ((mQ2_sq_val + mD2_sq_val)
                                    * rgePow(ys_val, 2.0))
                                    + ((mQ1_sq_val + mD1_sq_val)
                                    * rgePow(yd_val, 2.0))))
                            + (2.0 * (((mL3_sq_val + mE3_sq_val)
                                    * rgePow(ytau_val, 2.0))
                                    + ((mL2_sq_val + mE2_sq_val)
                                    * rgePow(ymu_val, 2.0))
                                    + ((mL1_sq_val + mE1_sq_val)
                                    * rgePow(ye_val, 2.0)))) // end trace
                            ))
                        - (16.0 * rgePow(yu_val, 2.0) * rgePow(au_val, 2.0))
                        - (16.0 * rgePow(yd_val, 2.0) * rgePow(ad_val, 2.0))
                        - (rgePow(au_val, 2.0) // Tr(6Yu^2)
                        * 6.0 * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                                + rgePow(yu_val, 2.0))) // end trace
                        - (rgePow(yu_val, 2.0) // Tr(6au^2)
                        * 6.0 * (rgePow(at_val, 2.0) + rgePow(ac_val, 2.0)
                                + rgePow(au_val, 2.0))) // end trace
                        - (au_val * yu_val // Tr(12Yu*au)
                        * 12.0 * ((yt_val * at_val) + (yc_val * ac_val)
                                + (yu_val * au_val))) // end trace
                        - (rgePow(ad_val, 2.0) // Tr(6Yd^2 + 2Ye^2)
                        * ((6.0 * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + (2.0 * (rgePow(ytau_val, 2.0)
                                    + rgePow(ymu_val, 2.0)
                                    + rgePow(ye_val, 2.0))))) // end trace
                        - (rgePow(yd_val, 2.0) // Tr(6ad^2 + 2ae^2)
                        * ((6.0 * (rgePow(ab_val, 2.0) + rgePow(as_val, 2.0)
                                    + rgePow(ad_val, 2.0)))
                            + (2.0 * (rgePow(atau_val, 2.0)
                                    + rgePow(amu_val, 2.0)
                                    + rgePow(ae_val, 2.0))))) // end trace
                        - (2.0 * ad_val * yd_val // Tr(6Yd*ad + 2Ye*ae)
                        * ((6.0 * ((yb_val * ab_val) + (ys_val * as_val)
                                    + (yd_val * ad_val)))
                            + (2.0 * ((ytau_val * atau_val)
                                    + (ymu_val * amu_val)
                                    + (ye_val * ae_val))))) // end trace
                        + ((2.0 / 5.0) * rgePow(g1_val, 2.0)
                        * ((4.0 * (mQ1_sq_val + mHu_sq_val + mU1_sq_val)
                            * rgePow(yu_val, 2.0))
                            + (4.0 * rgePow(au_val, 2.0))
                            - (8.0 * M1_val * au_val * yu_val)
                            + (8.0 * rgePow(M1_val, 2.0) * rgePow(yu_val, 2.0))
                            + (2.0 * (mQ1_sq_val + mHd_sq_val + mD1_sq_val)
                                * rgePow(yd_val, 2.0))
                            + (2.0 * rgePow(ad_val, 2.0))
                            - (4.0 * M1_val * ad_val * yd_val)
                            + (4.0 * rgePow(M1_val, 2.0)
                                * rgePow(yd_val, 2.0))))
                        + ((2.0 / 5.0) * rgePow(g1_val, 2.0) * Spr_val)
                        - ((128.0 / 3.0) * rgePow(g3_val, 4.0)
                        * rgePow(M3_val, 2.0))
                        + (32.0 * rgePow(g3_val, 2.0) * rgePow(g2_val, 2.0)
                        * (rgePow(M3_val, 2.0) + rgePow(M2_val, 2.0)
                            + (M2_val * M3_val)))
                        + ((32.0 / 45.0) * rgePow(g3_val, 2.0)
                        * rgePow(g1_val, 2.0)
                        * (rgePow(M3_val, 2.0) + rgePow(M1_val, 2.0)
                            + (M3_val * M1_val)))
                        + (33.0 * rgePow(g2_val, 4.0) * rgePow(M2_val, 2.0))
                        + ((2.0 / 5.0) * rgePow(g2_val, 2.0) * rgePow(g1_val, 2.0)
                        * (rgePow(M1_val, 2.0) + rgePow(M2_val, 2.0)
                            + (M1_val * M2_val)))
                        + ((199.0 / 75.0) * rgePow(g1_val, 4.0) * rgePow(M1_val, 2.0))
                        + ((16.0 / 3.0) * rgePow(g3_val, 2.0) * sigma3)
                        + (3.0 * rgePow(g2_val, 2.0) * sigma2)
                        + ((1.0 / 15.0) * rgePow(g1_val, 2.0) * sigma1));

        // Left leptons
    Real dmL3_sq_dt_2l = (((-8.0)* (mL3_sq_val + mHd_sq_val + mE3_sq_val)
                        * rgePow(ytau_val, 4.0))
                        - (rgePow(ytau_val, 2.0)
                        * ((2.0 * mL3_sq_val) + (2.0 * mE3_sq_val)
                            + (4.0 * mHd_sq_val)) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                            + rgePow(ye_val, 2.0))) // end trace
                        - (rgePow(ytau_val, 2.0) // Tr(6(mQ^2 + mD^2)*Yd^2 + 2(mL^2 + mE^2)*Ye^2)
                        * ((6.0 * (((mQ3_sq_val + mD3_sq_val)
                                    * rgePow(yb_val, 2.0))
                                    + ((mQ2_sq_val + mD2_sq_val)
                                    * rgePow(ys_val, 2.0))
                                    + ((mQ1_sq_val + mD1_sq_val)
                                    * rgePow(yd_val, 2.0))))
                            + (2.0 * (((mL3_sq_val + mE3_sq_val)
                                    * rgePow(ytau_val, 2.0))
                                    + ((mL2_sq_val + mE2_sq_val)
                                    * rgePow(ymu_val, 2.0))
                                    + ((mL1_sq_val + mE1_sq_val)
                                    * rgePow(ye_val, 2.0)))) // end trace
                            ))
                        - (16.0 * rgePow(ytau_val, 2.0) * rgePow(atau_val, 2.0))
                        - (rgePow(atau_val, 2.0) // Tr(6Yd^2 + 2Ye^2)
                        * ((6.0 * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + (2.0 * (rgePow(ytau_val, 2.0)
                                    + rgePow(ymu_val, 2.0)
                                    + rgePow(ye_val, 2.0))))) // end trace
                        - (rgePow(ytau_val, 2.0) // Tr(6ad^2 + 2ae^2)
                        * ((6.0 * (rgePow(ab_val, 2.0) + rgePow(as_val, 2.0)
                                    + rgePow(ad_val, 2.0)))
                            + (2.0 * (rgePow(atau_val, 2.0)
                                    + rgePow(amu_val, 2.0)
                                    + rgePow(ae_val, 2.0))))) // end trace
                        - (2.0 * atau_val * ytau_val // Tr(6Yd*ad + 2Ye*ae)
                        * ((6.0 * ((yb_val * ab_val) + (ys_val * as_val)
                                    + (yd_val * ad_val)))
                            + (2.0 * ((ytau_val * atau_val)
                                    + (ymu_val * amu_val)
                                    + (ye_val * ae_val))))) // end trace
                        + ((6.0 / 5.0) * rgePow(g1_val, 2.0)
                        * ((2.0 * (mL3_sq_val + mHd_sq_val + mE3_sq_val)
                            * rgePow(ytau_val, 2.0))
                            + (2.0 * rgePow(atau_val, 2.0))
                            - (4.0 * M1_val * atau_val
                                * ytau_val)
                            + (4.0 * rgePow(M1_val, 2.0)
                                * rgePow(ytau_val, 2.0))))
                        - ((6.0 / 5.0) * rgePow(g1_val, 2.0) * Spr_val)
                        + (33.0 * rgePow(g2_val, 4.0) * rgePow(M2_val, 2.0))
                        + ((18.0 / 5.0) * rgePow(g2_val, 2.0)
                        * rgePow(g1_val, 2.0)
                        * (rgePow(M1_val, 2.0) + rgePow(M2_val, 2.0)
                            + (M1_val * M2_val)))
                        + ((621.0 / 25.0) * rgePow(g1_val, 4.0)
                        * rgePow(M1_val, 2.0))
                        + (3.0 * rgePow(g2_val, 2.0) * sigma2)
                        + (3.0 / 5.0 * rgePow(g1_val, 2.0) * sigma1));

    Real dmL2_sq_dt_2l = (((-8.0)* (mL2_sq_val + mHd_sq_val + mE2_sq_val)
                        * rgePow(ymu_val, 4.0))
                        - (rgePow(ymu_val, 2.0)
                        * ((2.0 * mL2_sq_val) + (2.0 * mE2_sq_val)
                            + (4.0 * mHd_sq_val)) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                            + rgePow(ye_val, 2.0))) // end trace
                        - (rgePow(ymu_val, 2.0) // Tr(6(mQ^2 + mD^2)*Yd^2 + 2(mL^2 + mE^2)*Ye^2)
                        * ((6.0 * (((mQ3_sq_val + mD3_sq_val)
                                    * rgePow(yb_val, 2.0))
                                    + ((mQ2_sq_val + mD2_sq_val)
                                    * rgePow(ys_val, 2.0))
                                    + ((mQ1_sq_val + mD1_sq_val)
                                    * rgePow(yd_val, 2.0))))
                            + (2.0 * (((mL3_sq_val + mE3_sq_val)
                                    * rgePow(ytau_val, 2.0))
                                    + ((mL2_sq_val + mE2_sq_val)
                                    * rgePow(ymu_val, 2.0))
                                    + ((mL1_sq_val + mE1_sq_val)
                                    * rgePow(ye_val, 2.0)))) // end trace
                            ))
                        - (16.0 * rgePow(ymu_val, 2.0) * rgePow(amu_val, 2.0))
                        - (rgePow(amu_val, 2.0) // Tr(6Yd^2 + 2Ye^2)
                        * ((6.0 * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + (2.0 * (rgePow(ytau_val, 2.0)
                                    + rgePow(ymu_val, 2.0)
                                    + rgePow(ye_val, 2.0))))) // end trace
                        - (rgePow(ymu_val, 2.0) // Tr(6ad^2 + 2ae^2)
                        * ((6.0 * (rgePow(ab_val, 2.0) + rgePow(as_val, 2.0)
                                    + rgePow(ad_val, 2.0)))
                            + (2.0 * (rgePow(atau_val, 2.0)
                                    + rgePow(amu_val, 2.0)
                                    + rgePow(ae_val, 2.0))))) // end trace
                        - (2.0 * amu_val * ymu_val // Tr(6Yd*ad + 2Ye*ae)
                        * ((6.0 * ((yb_val * ab_val) + (ys_val * as_val)
                                    + (yd_val * ad_val)))
                            + (2.0 * ((ytau_val * atau_val)
                                    + (ymu_val * amu_val)
                                    + (ye_val * ae_val))))) // end trace
                        + ((6.0 / 5.0) * rgePow(g1_val, 2.0)
                        * ((2.0 * (mL2_sq_val + mHd_sq_val + mE2_sq_val)
                            * rgePow(ymu_val, 2.0))
                            + (2.0 * rgePow(amu_val, 2.0))
                            - (4.0 * M1_val * amu_val
                                * ymu_val)
                            + (4.0 * rgePow(M1_val, 2.0)
                                * rgePow(ymu_val, 2.0))))
                        - ((6.0 / 5.0) * rgePow(g1_val, 2.0) * Spr_val)
                        + (33.0 * rgePow(g2_val, 4.0) * rgePow(M2_val, 2.0))
                        + ((18.0 / 5.0) * rgePow(g2_val, 2.0)
                        * rgePow(g1_val, 2.0)
                        * (rgePow(M1_val, 2.0) + rgePow(M2_val, 2.0)
                            + (M1_val * M2_val)))
                        + ((621.0 / 25.0) * rgePow(g1_val, 4.0)
                        * rgePow(M1_val, 2.0))
                        + (3.0 * rgePow(g2_val, 2.0) * sigma2)
                        + (3.0 / 5.0 * rgePow(g1_val, 2.0) * sigma1));

    Real dmL1_sq_dt_2l = (((-8.0)* (mL1_sq_val + mHd_sq_val + mE1_sq_val)
                        * rgePow(ye_val, 4.0))
                        - (rgePow(ye_val, 2.0)
                        * ((2.0 * mL1_sq_val) + (2.0 * mE1_sq_val)
                            + (4.0 * mHd_sq_val)) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                            + rgePow(ye_val, 2.0))) // end trace
                        - (rgePow(ye_val, 2.0) // Tr(6(mQ^2 + mD^2)*Yd^2 + 2(mL^2 + mE^2)*Ye^2)
                        * ((6.0 * (((mQ3_sq_val + mD3_sq_val)
                                    * rgePow(yb_val, 2.0))
                                    + ((mQ2_sq_val + mD2_sq_val)
                                    * rgePow(ys_val, 2.0))
                                    + ((mQ1_sq_val + mD1_sq_val)
                                    * rgePow(yd_val, 2.0))))
                            + (2.0 * (((mL3_sq_val + mE3_sq_val)
                                    * rgePow(ytau_val, 2.0))
                                    + ((mL2_sq_val + mE2_sq_val)
                                    * rgePow(ymu_val, 2.0))
                                    + ((mL1_sq_val + mE1_sq_val)
                                    * rgePow(ye_val, 2.0)))) // end trace
                            ))
                        - (16.0 * rgePow(ye_val, 2.0) * rgePow(ae_val, 2.0))
                        - (rgePow(ae_val, 2.0) // Tr(6Yd^2 + 2Ye^2)
                        * ((6.0 * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + (2.0 * (rgePow(ytau_val, 2.0)
                                    + rgePow(ymu_val, 2.0)
                                    + rgePow(ye_val, 2.0))))) // end trace
                        - (rgePow(ye_val, 2.0) // Tr(6ad^2 + 2ae^2)
                        * ((6.0 * (rgePow(ab_val, 2.0) + rgePow(as_val, 2.0)
                                    + rgePow(ad_val, 2.0)))
                            + (2.0 * (rgePow(atau_val, 2.0)
                                    + rgePow(amu_val, 2.0)
                                    + rgePow(ae_val, 2.0))))) // end trace
                        - (2.0 * ae_val * ye_val // Tr(6Yd*ad + 2Ye*ae)
                        * ((6.0 * ((yb_val * ab_val) + (ys_val * as_val)
                                    + (yd_val * ad_val)))
                            + (2.0 * ((ytau_val * atau_val)
                                    + (ymu_val * amu_val)
                                    + (ye_val * ae_val))))) // end trace
                        + ((6.0 / 5.0) * rgePow(g1_val, 2.0)
                        * ((2.0 * (mL1_sq_val + mHd_sq_val + mE1_sq_val)
                            * rgePow(ye_val, 2.0))
                            + (2.0 * rgePow(ae_val, 2.0))
                            - (4.0 * M1_val * ae_val
                                * ye_val)
                            + (4.0 * rgePow(M1_val, 2.0)
                                * rgePow(ye_val, 2.0))))
                        - ((6.0 / 5.0) * rgePow(g1_val, 2.0) * Spr_val)
                        + (33.0 * rgePow(g2_val, 4.0) * rgePow(M2_val, 2.0))
                        + ((18.0 / 5.0) * rgePow(g2_val, 2.0)
                        * rgePow(g1_val, 2.0)
                        * (rgePow(M1_val, 2.0) + rgePow(M2_val, 2.0)
                            + (M1_val * M2_val)))
                        + ((621.0 / 25.0) * rgePow(g1_val, 4.0)
                        * rgePow(M1_val, 2.0))
                        + (3.0 * rgePow(g2_val, 2.0) * sigma2)
                        + (3.0 / 5.0 * rgePow(g1_val, 2.0) * sigma1));

        // Right up-type squarks
    Real dmU3_sq_dt_2l = (((-8.0)* (mQ3_sq_val + mHu_sq_val + mU3_sq_val)
                        * rgePow(yt_val, 4.0))
                        - (4.0 * (mU3_sq_val + mHu_sq_val + mHd_sq_val
                            + (2.0 * mQ3_sq_val) + mD3_sq_val)
                        * rgePow(yb_val, 2.0) * rgePow(yt_val, 2.0))
                        - (rgePow(yt_val, 2.0)
                        * ((2.0 * mQ3_sq_val) + (2.0 * mU3_sq_val)
                            + (4.0 * mHu_sq_val)) // Tr(6Yu^2)
                        * 6.0 * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                                + rgePow(yu_val, 2.0))) // end trace
                        - (12.0 * rgePow(yt_val, 2.0) // Tr((mQ^2 + mU^2)*Yu^2)
                        * (((mQ3_sq_val + mU3_sq_val)
                            * rgePow(yt_val, 2.0))
                            + ((mQ2_sq_val + mU2_sq_val)
                                * rgePow(yc_val, 2.0))
                            + ((mQ1_sq_val + mU1_sq_val)
                                * rgePow(yu_val, 2.0)))) // end trace
                        - (16.0 * rgePow(yt_val, 2.0) * rgePow(at_val, 2.0))
                        - (16.0 * at_val * ab_val * yb_val * yt_val)
                        - (12.0 * ((rgePow(at_val, 2.0) // Tr(Yu^2)
                                * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                                    + rgePow(yu_val, 2.0))) // end trace
                                + (rgePow(yt_val, 2.0)  // Tr(au^2)
                                    * (rgePow(at_val, 2.0)
                                    + rgePow(ac_val, 2.0)
                                    + rgePow(au_val, 2.0))) // end trace
                                + (at_val * yt_val * 2 // Tr(Yu*au)
                                    * ((yt_val * at_val) + (yc_val * ac_val)
                                    + (yu_val * au_val))))) // end trace
                        + (((6.0 * rgePow(g2_val, 2.0))
                        - ((2.0 / 5.0) * rgePow(g1_val, 2.0)))
                        * ((2.0 * (mQ3_sq_val + mHu_sq_val + mU3_sq_val)
                            * rgePow(yt_val, 2.0))
                            + (2.0 * rgePow(at_val, 2.0))))
                        + (12.0 * rgePow(g2_val, 2.0)
                        * 2.0 * ((rgePow(M2_val, 2.0) * rgePow(yt_val, 2.0))
                                - (M2_val * at_val * yt_val)))
                        - ((4.0 / 5.0) * rgePow(g1_val, 2.0)
                        * 2.0 * ((rgePow(M1_val, 2.0) * rgePow(yt_val, 2.0))
                                - (M1_val * at_val * yt_val)))
                        - ((8.0 / 5.0) * rgePow(g1_val, 2.0) * Spr_val)
                        - ((128.0 / 3.0) * rgePow(g3_val, 4.0)
                        * rgePow(M3_val, 2.0))
                        + ((512.0 / 45.0) * rgePow(g3_val, 2.0)
                        * rgePow(g1_val, 2.0)
                        * (rgePow(M3_val, 2.0) + rgePow(M1_val, 2.0)
                            + (M3_val * M1_val)))
                        + ((3424.0 / 75.0) * rgePow(g1_val, 4.0)
                        * rgePow(M1_val, 2.0))
                        + ((16.0 / 3.0) * rgePow(g3_val, 2.0) * sigma3)
                        + ((16.0 / 15.0) * rgePow(g1_val, 2.0) * sigma1));

    Real dmU2_sq_dt_2l = (((-8.0)* (mQ2_sq_val + mHu_sq_val + mU2_sq_val)
                        * rgePow(yc_val, 4.0))
                        - (4.0 * (mU2_sq_val + mHu_sq_val + mHd_sq_val
                            + (2.0 * mQ2_sq_val)
                            + mD2_sq_val)
                        * rgePow(ys_val, 2.0) * rgePow(yc_val, 2.0))
                        - (rgePow(yc_val, 2.0)
                        * ((2.0 * mQ2_sq_val) + (2.0 * mU2_sq_val)
                            + (4.0 * mHu_sq_val)) // Tr(6Yu^2)
                        * 6.0 * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                                + rgePow(yu_val, 2.0))) // end trace
                        - (12.0 * rgePow(yc_val, 2.0) // Tr((mQ^2 + mU^2)*Yu^2)
                        * (((mQ3_sq_val + mU3_sq_val)
                            * rgePow(yt_val, 2.0))
                            + ((mQ2_sq_val + mU2_sq_val)
                                * rgePow(yc_val, 2.0))
                            + ((mQ1_sq_val + mU1_sq_val)
                                * rgePow(yu_val, 2.0)))) // end trace
                        - (16.0 * rgePow(yc_val, 2.0) * rgePow(ac_val, 2.0))
                        - (16.0 * ac_val * as_val * ys_val * yc_val)
                        - (12.0 * ((rgePow(ac_val, 2.0) // Tr(Yu^2)
                                * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                                    + rgePow(yu_val, 2.0))) // end trace
                                + (rgePow(yc_val, 2.0)  // Tr(au^2)
                                    * (rgePow(at_val, 2.0)
                                    + rgePow(ac_val, 2.0)
                                    + rgePow(au_val, 2.0))) // end trace
                                + (ac_val * yc_val * 2 // Tr(Yu*au)
                                    * ((yt_val * at_val) + (yc_val * ac_val)
                                    + (yu_val * au_val))))) // end trace
                        + (((6.0 * rgePow(g2_val, 2.0))
                        - ((2.0 / 5.0) * rgePow(g1_val, 2.0)))
                        * ((2.0 * (mQ2_sq_val + mHu_sq_val + mU2_sq_val)
                            * rgePow(yc_val, 2.0))
                            + (2.0 * rgePow(ac_val, 2.0))))
                        + (12.0 * rgePow(g2_val, 2.0)
                        * 2.0 * ((rgePow(M2_val, 2.0) * rgePow(yc_val, 2.0))
                                - (M2_val * ac_val * yc_val)))
                        - ((4.0 / 5.0) * rgePow(g1_val, 2.0)
                        * 2.0 * ((rgePow(M1_val, 2.0) * rgePow(yc_val, 2.0))
                                - (M1_val * ac_val * yc_val)))
                        - ((8.0 / 5.0) * rgePow(g1_val, 2.0) * Spr_val)
                        - ((128.0 / 3.0) * rgePow(g3_val, 4.0)
                        * rgePow(M3_val, 2.0))
                        + ((512.0 / 45.0) * rgePow(g3_val, 2.0)
                        * rgePow(g1_val, 2.0)
                        * (rgePow(M3_val, 2.0) + rgePow(M1_val, 2.0)
                            + (M3_val * M1_val)))
                        + ((3424.0 / 75.0) * rgePow(g1_val, 4.0)
                        * rgePow(M1_val, 2.0))
                        + ((16.0 / 3.0) * rgePow(g3_val, 2.0) * sigma3)
                        + ((16.0 / 15.0) * rgePow(g1_val, 2.0) * sigma1));

    Real dmU1_sq_dt_2l = (((-8.0)* (mQ1_sq_val + mHu_sq_val + mU1_sq_val)
                        * rgePow(yu_val, 4.0))
                        - (4.0 * (mU1_sq_val + mHu_sq_val + mHd_sq_val
                            + (2.0 * mQ1_sq_val)
                            + mD1_sq_val)
                        * rgePow(yd_val, 2.0) * rgePow(yu_val, 2.0))
                        - (rgePow(yu_val, 2.0)
                        * ((2.0 * mQ1_sq_val) + (2.0 * mU1_sq_val)
                            + (4.0 * mHu_sq_val)) // Tr(6Yu^2)
                        * 6.0 * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                                + rgePow(yu_val, 2.0))) // end trace
                        - (12.0 * rgePow(yu_val, 2.0) // Tr((mQ^2 + mU^2)*Yu^2)
                        * (((mQ3_sq_val + mU3_sq_val)
                            * rgePow(yt_val, 2.0))
                            + ((mQ2_sq_val + mU2_sq_val)
                                * rgePow(yc_val, 2.0))
                            + ((mQ1_sq_val + mU1_sq_val)
                                * rgePow(yu_val, 2.0)))) // end trace
                        - (16.0 * rgePow(yu_val, 2.0) * rgePow(au_val, 2.0))
                        - (16.0 * au_val * ad_val * yd_val * yu_val)
                        - (12.0 * ((rgePow(au_val, 2.0) // Tr(Yu^2)
                                * (rgePow(yt_val, 2.0) + rgePow(yc_val, 2.0)
                                    + rgePow(yu_val, 2.0))) // end trace
                                + (rgePow(yu_val, 2.0)  // Tr(au^2)
                                    * (rgePow(at_val, 2.0)
                                    + rgePow(ac_val, 2.0)
                                    + rgePow(au_val, 2.0))) // end trace
                                + (au_val * yu_val * 2 // Tr(Yu*au)
                                    * ((yt_val * at_val) + (yc_val * ac_val)
                                    + (yu_val * au_val))))) // end trace
                        + (((6.0 * rgePow(g2_val, 2.0))
                        - ((2.0 / 5.0) * rgePow(g1_val, 2.0)))
                        * ((2.0 * (mQ1_sq_val + mHu_sq_val + mU1_sq_val)
                            * rgePow(yu_val, 2.0))
                            + (2.0 * rgePow(au_val, 2.0))))
                        + (12.0 * rgePow(g2_val, 2.0)
                        * 2.0 * ((rgePow(M2_val, 2.0) * rgePow(yu_val, 2.0))
                                - (M2_val * au_val * yu_val)))
                        - ((4.0 / 5.0) * rgePow(g1_val, 2.0)
                        * 2.0 * ((rgePow(M1_val, 2.0) * rgePow(yu_val, 2.0))
                                - (M1_val * au_val * yu_val)))
                        - ((8.0 / 5.0) * rgePow(g1_val, 2.0) * Spr_val)
                        - ((128.0 / 3.0) * rgePow(g3_val, 4.0)
                        * rgePow(M3_val, 2.0))
                        + ((512.0 / 45.0) * rgePow(g3_val, 2.0)
                        * rgePow(g1_val, 2.0)
                        * (rgePow(M3_val, 2.0) + rgePow(M1_val, 2.0)
                            + (M3_val * M1_val)))
                        + ((3424.0 / 75.0) * rgePow(g1_val, 4.0)
                        * rgePow(M1_val, 2.0))
                        + ((16.0 / 3.0) * rgePow(g3_val, 2.0) * sigma3)
                        + ((16.0 / 15.0) * rgePow(g1_val, 2.0) * sigma1));

        // Right down-type squarks
    Real dmD3_sq_dt_2l = (((-8.0)* (mQ3_sq_val + mHd_sq_val + mD3_sq_val)
                        * rgePow(yb_val, 4.0))
                        - (4.0 * (mU3_sq_val + mHu_sq_val + mHd_sq_val
                            + (2.0 * mQ3_sq_val)
                            + mD3_sq_val) * rgePow(yb_val, 2.0)
                        * rgePow(yt_val, 2.0))
                        - (rgePow(yb_val, 2.0)
                        * (2.0 * (mD3_sq_val + mQ3_sq_val
                                + (2.0 * mHd_sq_val))) // Tr(6Yd^2 + 2Ye^2)
                        * ((6.0 * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + (2.0 * (rgePow(ytau_val, 2.0)
                                    + rgePow(ymu_val, 2.0)
                                    + rgePow(ye_val, 2.0))))) // end trace
                        - (4.0 * rgePow(yb_val, 2.0)  // Tr(3(mQ^2 + mD^2) * Yd^2 + (mL^2 + mE^2) * Ye^2)
                        * ((3.0 * (((mQ3_sq_val + mD3_sq_val)
                                    * rgePow(yb_val, 2.0))
                                    + ((mQ2_sq_val + mD2_sq_val)
                                    * rgePow(ys_val, 2.0))
                                    + ((mQ1_sq_val + mD1_sq_val)
                                    * rgePow(yd_val, 2.0))))
                            + (((mL3_sq_val + mE3_sq_val)
                                * rgePow(ytau_val, 2.0))
                                + ((mL2_sq_val + mE2_sq_val)
                                    * rgePow(ymu_val, 2.0))
                                + ((mL1_sq_val + mE1_sq_val)
                                    * rgePow(ye_val, 2.0))) // end trace
                            ))
                        - (16.0 * rgePow(yb_val, 2.0) * rgePow(ab_val, 2.0))
                        - (16.0 * at_val * ab_val * yb_val * yt_val)
                        - (4.0 * rgePow(ab_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                            + rgePow(ye_val, 2.0))) // end trace
                        - (4.0 * rgePow(yb_val, 2.0)  // Tr(3ad^2 + ae^2)
                        * ((3.0 * (rgePow(ab_val, 2.0) + rgePow(as_val, 2.0)
                                    + rgePow(ad_val, 2.0)))
                            + rgePow(atau_val, 2.0) + rgePow(amu_val, 2.0)
                            + rgePow(ae_val, 2.0))) // end trace
                        - (8.0 * ab_val * yb_val  // Tr(3Yd * ad + Ye * ae)
                        * ((3.0 * ((yb_val * ab_val) + (ys_val * as_val)
                                    + (yd_val * ad_val)))
                            + (ytau_val * atau_val) + (ymu_val * amu_val)
                            + (ye_val * ae_val))) // end trace
                        + (((6.0 * rgePow(g2_val, 2.0))
                        + ((2.0 / 5.0) * rgePow(g1_val, 2.0)))
                        * ((2.0 * (mQ3_sq_val + mHd_sq_val + mD3_sq_val)
                            * rgePow(yb_val, 2.0))
                            + (2.0 * rgePow(ab_val, 2.0))))
                        + (12.0 * rgePow(g2_val, 2.0)
                        * 2.0 * ((rgePow(M2_val, 2.0) * rgePow(yb_val, 2.0))
                                - (M2_val * ab_val * yb_val)))
                        + ((4.0 / 5.0) * rgePow(g1_val, 2.0)
                        * 2.0 * ((rgePow(M1_val, 2.0) * rgePow(yb_val, 2.0))
                                - (M1_val * ab_val * yb_val)))
                        + ((4.0 / 5.0) * rgePow(g1_val, 2.0) * Spr_val)
                        - ((128.0 / 3.0) * rgePow(g3_val, 4.0)
                        * rgePow(M3_val, 2.0))
                        + ((128.0 / 45.0) * rgePow(g3_val, 2.0)
                        * rgePow(g1_val, 2.0)
                        * (rgePow(M3_val, 2.0) + rgePow(M1_val, 2.0)
                            + (M3_val * M1_val)))
                        + ((808.0 / 75.0) * rgePow(g1_val, 4.0)
                        * rgePow(M1_val, 2.0))
                        + ((16.0 / 3.0) * rgePow(g3_val, 2.0) * sigma3)
                        + ((4.0 / 15.0) * rgePow(g1_val, 2.0) * sigma1));

    Real dmD2_sq_dt_2l = (((-8.0)* (mQ2_sq_val + mHd_sq_val + mD2_sq_val)
                        * rgePow(ys_val, 4.0))
                        - (4.0 * (mU2_sq_val + mHu_sq_val + mHd_sq_val
                            + (2.0 * mQ2_sq_val)
                            + mD2_sq_val) * rgePow(ys_val, 2.0)
                        * rgePow(yc_val, 2.0))
                        - (rgePow(ys_val, 2.0)
                        * (2.0 * (mD2_sq_val + mQ2_sq_val
                                + (2.0 * mHd_sq_val))) // Tr(6Yd^2 + 2Ye^2)
                        * ((6.0 * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + (2.0 * (rgePow(ytau_val, 2.0)
                                    + rgePow(ymu_val, 2.0)
                                    + rgePow(ye_val, 2.0))))) // end trace
                        - (4.0 * rgePow(ys_val, 2.0)  // Tr(3(mQ^2 + mD^2) * Yd^2 + (mL^2 + mE^2) * Ye^2)
                        * ((3.0 * (((mQ3_sq_val + mD3_sq_val)
                                    * rgePow(yb_val, 2.0))
                                    + ((mQ2_sq_val + mD2_sq_val)
                                    * rgePow(ys_val, 2.0))
                                    + ((mQ1_sq_val + mD1_sq_val)
                                    * rgePow(yd_val, 2.0))))
                            + (((mL3_sq_val + mE3_sq_val)
                                * rgePow(ytau_val, 2.0))
                                + ((mL2_sq_val + mE2_sq_val)
                                    * rgePow(ymu_val, 2.0))
                                + ((mL1_sq_val + mE1_sq_val)
                                    * rgePow(ye_val, 2.0))) // end trace
                            ))
                        - (16.0 * rgePow(ys_val, 2.0) * rgePow(as_val, 2.0))
                        - (16.0 * ac_val * as_val * ys_val * yc_val)
                        - (4.0 * rgePow(as_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                            + rgePow(ye_val, 2.0))) // end trace
                        - (4.0 * rgePow(ys_val, 2.0)  // Tr(3ad^2 + ae^2)
                        * ((3.0 * (rgePow(ab_val, 2.0) + rgePow(as_val, 2.0)
                                    + rgePow(ad_val, 2.0)))
                            + rgePow(atau_val, 2.0) + rgePow(amu_val, 2.0)
                            + rgePow(ae_val, 2.0))) // end trace
                        - (8.0 * as_val * ys_val  // Tr(3Yd * ad + Ye * ae)
                        * ((3.0 * ((yb_val * ab_val) + (ys_val * as_val)
                                    + (yd_val * ad_val)))
                            + (ytau_val * atau_val) + (ymu_val * amu_val)
                            + (ye_val * ae_val))) // end trace
                        + (((6.0 * rgePow(g2_val, 2.0))
                        + ((2.0 / 5.0) * rgePow(g1_val, 2.0)))
                        * ((2.0 * (mQ2_sq_val + mHd_sq_val + mD2_sq_val)
                            * rgePow(ys_val, 2.0))
                            + (2.0 * rgePow(as_val, 2.0))))
                        + (12.0 * rgePow(g2_val, 2.0)
                        * 2.0 * ((rgePow(M2_val, 2.0) * rgePow(ys_val, 2.0))
                                - (M2_val * as_val * ys_val)))
                        + ((4.0 / 5.0) * rgePow(g1_val, 2.0)
                        * 2.0 * ((rgePow(M1_val, 2.0) * rgePow(ys_val, 2.0))
                                - (M1_val * as_val * ys_val)))
                        + ((4.0 / 5.0) * rgePow(g1_val, 2.0) * Spr_val)
                        - ((128.0 / 3.0) * rgePow(g3_val, 4.0)
                        * rgePow(M3_val, 2.0))
                        + ((128.0 / 45.0) * rgePow(g3_val, 2.0)
                        * rgePow(g1_val, 2.0)
                        * (rgePow(M3_val, 2.0) + rgePow(M1_val, 2.0)
                            + (M3_val * M1_val)))
                        + ((808.0 / 75.0) * rgePow(g1_val, 4.0)
                        * rgePow(M1_val, 2.0))
                        + ((16.0 / 3.0) * rgePow(g3_val, 2.0) * sigma3)
                        + ((4.0 / 15.0) * rgePow(g1_val, 2.0) * sigma1));

    Real dmD1_sq_dt_2l = (((-8.0)* (mQ1_sq_val + mHd_sq_val + mD1_sq_val)
                        * rgePow(yd_val, 4.0))
                        - (4.0 * (mU1_sq_val + mHu_sq_val + mHd_sq_val
                            + (2.0 * mQ1_sq_val)
                            + mD1_sq_val) * rgePow(yd_val, 2.0)
                        * rgePow(yu_val, 2.0))
                        - (rgePow(yd_val, 2.0)
                        * (2.0 * (mD1_sq_val + mQ1_sq_val
                                + (2.0 * mHd_sq_val))) // Tr(6Yd^2 + 2Ye^2)
                        * ((6.0 * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + (2.0 * (rgePow(ytau_val, 2.0)
                                    + rgePow(ymu_val, 2.0)
                                    + rgePow(ye_val, 2.0))))) // end trace
                        - (4.0 * rgePow(yd_val, 2.0)  // Tr(3(mQ^2 + mD^2) * Yd^2 + (mL^2 + mE^2) * Ye^2)
                        * ((3.0 * (((mQ3_sq_val + mD3_sq_val)
                                    * rgePow(yb_val, 2.0))
                                    + ((mQ2_sq_val + mD2_sq_val)
                                    * rgePow(ys_val, 2.0))
                                    + ((mQ1_sq_val + mD1_sq_val)
                                    * rgePow(yd_val, 2.0))))
                            + (((mL3_sq_val + mE3_sq_val)
                                * rgePow(ytau_val, 2.0))
                                + ((mL2_sq_val + mE2_sq_val)
                                    * rgePow(ymu_val, 2.0))
                                + ((mL1_sq_val + mE1_sq_val)
                                    * rgePow(ye_val, 2.0))) // end trace
                            ))
                        - (16.0 * rgePow(yd_val, 2.0) * rgePow(ad_val, 2.0))
                        - (16.0 * au_val * ad_val * yd_val * yu_val)
                        - (4.0 * rgePow(ad_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                            + rgePow(ye_val, 2.0))) // end trace
                        - (4.0 * rgePow(yd_val, 2.0)  // Tr(3ad^2 + ae^2)
                        * ((3.0 * (rgePow(ab_val, 2.0) + rgePow(as_val, 2.0)
                                    + rgePow(ad_val, 2.0)))
                            + rgePow(atau_val, 2.0) + rgePow(amu_val, 2.0)
                            + rgePow(ae_val, 2.0))) // end trace
                        - (8.0 * ad_val * yd_val  // Tr(3Yd * ad + Ye * ae)
                        * ((3.0 * ((yb_val * ab_val) + (ys_val * as_val)
                                    + (yd_val * ad_val)))
                            + (ytau_val * atau_val) + (ymu_val * amu_val)
                            + (ye_val * ae_val))) // end trace
                        + (((6.0 * rgePow(g2_val, 2.0))
                        + ((2.0 / 5.0) * rgePow(g1_val, 2.0)))
                        * ((2.0 * (mQ1_sq_val + mHd_sq_val + mD1_sq_val)
                            * rgePow(yd_val, 2.0))
                            + (2.0 * rgePow(ad_val, 2.0))))
                        + (12.0 * rgePow(g2_val, 2.0)
                        * 2.0 * ((rgePow(M2_val, 2.0) * rgePow(yd_val, 2.0))
                                - (M2_val * ad_val * yd_val)))
                        + ((4.0 / 5.0) * rgePow(g1_val, 2.0)
                        * 2.0 * ((rgePow(M1_val, 2.0) * rgePow(yd_val, 2.0))
                                - (M1_val * ad_val * yd_val)))
                        + ((4.0 / 5.0) * rgePow(g1_val, 2.0) * Spr_val)
                        - ((128.0 / 3.0) * rgePow(g3_val, 4.0)
                        * rgePow(M3_val, 2.0))
                        + ((128.0 / 45.0) * rgePow(g3_val, 2.0)
                        * rgePow(g1_val, 2.0)
                        * (rgePow(M3_val, 2.0) + rgePow(M1_val, 2.0)
                            + (M3_val * M1_val)))
                        + ((808.0 / 75.0) * rgePow(g1_val, 4.0)
                        * rgePow(M1_val, 2.0))
                        + ((16.0 / 3.0) * rgePow(g3_val, 2.0) * sigma3)
                        + ((4.0 / 15.0) * rgePow(g1_val, 2.0) * sigma1));

        // Right leptons
    Real dmE3_sq_dt_2l = (((-8.0)* (mL3_sq_val + mHd_sq_val + mE3_sq_val)
                        * rgePow(ytau_val, 4.0))
        - (rgePow(ytau_val, 2.0)
            * ((2.0 * mL3_sq_val) + (2.0 * mE3_sq_val)
            + (4.0 * mHd_sq_val)) // Tr(6Yd^2 + 2Ye^2)
            * ((6.0 * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                    + rgePow(yd_val, 2.0)))
            + (2.0 * (rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                        + rgePow(ye_val, 2.0))))) // end trace
        - (4.0 * rgePow(ytau_val, 2.0)  // Tr(3(mQ^2 + mD^2) * Yd^2 + (mL^2 + mE^2) * Ye^2)
            * ((3.0 * (((mQ3_sq_val + mD3_sq_val) * rgePow(yb_val, 2.0))
                    + ((mQ2_sq_val + mD2_sq_val) * rgePow(ys_val, 2.0))
                    + ((mQ1_sq_val + mD1_sq_val) * rgePow(yd_val, 2.0))))
            + ((((mL3_sq_val + mE3_sq_val) * rgePow(ytau_val, 2.0))
                    + ((mL2_sq_val + mE2_sq_val) * rgePow(ymu_val, 2.0))
                    + ((mL1_sq_val + mE1_sq_val) * rgePow(ye_val, 2.0)))) // end trace
            ))
        - (16.0 * rgePow(ytau_val, 2.0) * rgePow(atau_val, 2.0))
        - (4.0 * rgePow(atau_val, 2.0) // Tr(3Yd^2 + Ye^2)
            * ((3.0 * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                    + rgePow(yd_val, 2.0)))
            + ((rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                    + rgePow(ye_val, 2.0))))) // end trace
        - (4.0 * rgePow(ytau_val, 2.0)  // Tr(3ad^2 + ae^2)
            * ((3.0 * (rgePow(ab_val, 2.0) + rgePow(as_val, 2.0)
                    + rgePow(ad_val, 2.0)))
            + ((rgePow(atau_val, 2.0) + rgePow(amu_val, 2.0)
                    + rgePow(ae_val, 2.0))))) // end trace
        - (8.0 * atau_val * ytau_val  // Tr(3Yd * ad + Ye * ae)
            * ((3.0 * ((yb_val * ab_val) + (ys_val * as_val)
                        + (yd_val * ad_val)))
            + (((ytau_val * atau_val) + (ymu_val * amu_val)
                    + (ye_val * ae_val))))) // end trace
        + (((6.0 * rgePow(g2_val, 2.0)) - (6.0 / 5.0) * rgePow(g1_val, 2.0))
            * ((2.0 * (mL3_sq_val + mHd_sq_val + mE3_sq_val)
                * rgePow(ytau_val, 2.0))
            + (2.0 * rgePow(atau_val, 2.0))))
        + (12.0 * rgePow(g2_val, 2.0) * 2
            * ((rgePow(M2_val, 2.0) * rgePow(ytau_val, 2.0))
            - (M2_val * atau_val * ytau_val)))
        - ((12.0 / 5.0) * rgePow(g1_val, 2.0) * 2
            * ((rgePow(M1_val, 2.0) * rgePow(ytau_val, 2.0))
            - (M1_val * atau_val * ytau_val)))
        + ((12.0 / 5.0) * rgePow(g1_val, 2.0) * Spr_val)
        + ((2808.0 / 25.0) * rgePow(g1_val, 4.0) * rgePow(M1_val, 2.0))
        + ((12.0 / 5.0) * rgePow(g1_val, 2.0) * sigma1));

    Real dmE2_sq_dt_2l = (((-8.0)* (mL2_sq_val + mHd_sq_val + mE2_sq_val)
                        * rgePow(ymu_val, 4.0))
                        - (rgePow(ymu_val, 2.0)
                        * ((2.0 * mL2_sq_val) + (2.0 * mE2_sq_val)
                            + (4.0 * mHd_sq_val)) // Tr(6Yd^2 + 2Ye^2)
                        * ((6.0 * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + (2.0 * (rgePow(ytau_val, 2.0)
                                    + rgePow(ymu_val, 2.0)
                                    + rgePow(ye_val, 2.0))))) // end trace
                        - (4.0 * rgePow(ymu_val, 2.0)  // Tr(3(mQ^2 + mD^2) * Yd^2 + (mL^2 + mE^2) * Ye^2)
                        * ((3.0 * (((mQ3_sq_val + mD3_sq_val)
                                    * rgePow(yb_val, 2.0))
                                    + ((mQ2_sq_val + mD2_sq_val)
                                    * rgePow(ys_val, 2.0))
                                    + ((mQ1_sq_val + mD1_sq_val)
                                    * rgePow(yd_val, 2.0))))
                            + ((((mL3_sq_val + mE3_sq_val)
                                * rgePow(ytau_val, 2.0))
                                + ((mL2_sq_val + mE2_sq_val)
                                    * rgePow(ymu_val, 2.0))
                                + ((mL1_sq_val + mE1_sq_val)
                                    * rgePow(ye_val, 2.0)))) // end trace
                            ))
                        - (16.0 * rgePow(ymu_val, 2.0) * rgePow(amu_val, 2.0))
                        - (4.0 * rgePow(amu_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + ((rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                                + rgePow(ye_val, 2.0))))) // end trace
                        - (4.0 * rgePow(ymu_val, 2.0)  // Tr(3ad^2 + ae^2)
                        * ((3.0 * (rgePow(ab_val, 2.0) + rgePow(as_val, 2.0)
                                    + rgePow(ad_val, 2.0)))
                            + ((rgePow(atau_val, 2.0) + rgePow(amu_val, 2.0)
                                + rgePow(ae_val, 2.0))))) // end trace
                        - (8.0 * amu_val * ymu_val  // Tr(3Yd * ad + Ye * ae)
                        * ((3.0 * ((yb_val * ab_val) + (ys_val * as_val)
                                    + (yd_val * ad_val)))
                            + (((ytau_val * atau_val) + (ymu_val * amu_val)
                                + (ye_val * ae_val))))) // end trace
                        + (((6.0 * rgePow(g2_val, 2.0))
                        - (6.0 / 5.0) * rgePow(g1_val, 2.0))
                        * ((2.0 * (mL2_sq_val + mHd_sq_val + mE2_sq_val)
                            * rgePow(ymu_val, 2.0))
                            + (2.0 * rgePow(amu_val, 2.0))))
                        + (12.0 * rgePow(g2_val, 2.0) * 2
                        * ((rgePow(M2_val, 2.0) * rgePow(ymu_val, 2.0))
                            - (M2_val * amu_val * ymu_val)))
                        - ((12.0 / 5.0) * rgePow(g1_val, 2.0) * 2
                        * ((rgePow(M1_val, 2.0) * rgePow(ymu_val, 2.0))
                            - (M1_val * amu_val * ymu_val)))
                        + ((12.0 / 5.0) * rgePow(g1_val, 2.0) * Spr_val)
                        + ((2808.0 / 25.0) * rgePow(g1_val, 4.0)
                        * rgePow(M1_val, 2.0))
                        + ((12.0 / 5.0) * rgePow(g1_val, 2.0) * sigma1));

    Real dmE1_sq_dt_2l = (((-8.0)* (mL1_sq_val + mHd_sq_val + mE1_sq_val)
                        * rgePow(ye_val, 4.0))
                        - (rgePow(ye_val, 2.0)
                        * ((2.0 * mL1_sq_val) + (2.0 * mE1_sq_val)
                            + (4.0 * mHd_sq_val)) // Tr(6Yd^2 + 2Ye^2)
                        * ((6.0 * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + (2.0 * (rgePow(ytau_val, 2.0)
                                    + rgePow(ymu_val, 2.0)
                                    + rgePow(ye_val, 2.0))))) // end trace
                        - (4.0 * rgePow(ye_val, 2.0)  // Tr(3(mQ^2 + mD^2) * Yd^2 + (mL^2 + mE^2) * Ye^2)
                        * ((3.0 * (((mQ3_sq_val + mD3_sq_val)
                                    * rgePow(yb_val, 2.0))
                                    + ((mQ2_sq_val + mD2_sq_val)
                                    * rgePow(ys_val, 2.0))
                                    + ((mQ1_sq_val + mD1_sq_val)
                                    * rgePow(yd_val, 2.0))))
                            + ((((mL3_sq_val + mE3_sq_val)
                                * rgePow(ytau_val, 2.0))
                                + ((mL2_sq_val + mE2_sq_val)
                                    * rgePow(ymu_val, 2.0))
                                + ((mL1_sq_val + mE1_sq_val)
                                    * rgePow(ye_val, 2.0)))) // end trace
                            ))
                        - (16.0 * rgePow(ye_val, 2.0) * rgePow(ae_val, 2.0))
                        - (4.0 * rgePow(ae_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (rgePow(yb_val, 2.0) + rgePow(ys_val, 2.0)
                                    + rgePow(yd_val, 2.0)))
                            + ((rgePow(ytau_val, 2.0) + rgePow(ymu_val, 2.0)
                                + rgePow(ye_val, 2.0))))) // end trace
                        - (4.0 * rgePow(ye_val, 2.0)  // Tr(3ad^2 + ae^2)
                        * ((3.0 * (rgePow(ab_val, 2.0) + rgePow(as_val, 2.0)
                                    + rgePow(ad_val, 2.0)))
                            + ((rgePow(atau_val, 2.0) + rgePow(amu_val, 2.0)
                                + rgePow(ae_val, 2.0))))) // end trace
                        - (8.0 * ae_val * ye_val  // Tr(3Yd * ad + Ye * ae)
                        * ((3.0 * ((yb_val * ab_val) + (ys_val * as_val)
                                    + (yd_val * ad_val)))
                            + (((ytau_val * atau_val) + (ymu_val * amu_val)
                                + (ye_val * ae_val))))) // end trace
                        + (((6.0 * rgePow(g2_val, 2.0))
                        - (6.0 / 5.0) * rgePow(g1_val, 2.0))
                        * ((2.0 * (mL1_sq_val + mHd_sq_val + mE1_sq_val)
                            * rgePow(ye_val, 2.0))
                            + (2.0 * rgePow(ae_val, 2.0))))
                        + (12.0 * rgePow(g2_val, 2.0) * 2
                        * ((rgePow(M2_val, 2.0) * rgePow(ye_val, 2.0))
                            - (M2_val * ae_val * ye_val)))
                        - ((12.0 / 5.0) * rgePow(g1_val, 2.0) * 2
                        * ((rgePow(M1_val, 2.0) * rgePow(ye_val, 2.0))
                            - (M1_val * ae_val * ye_val)))
                        + ((12.0 / 5.0) * rgePow(g1_val, 2.0) * Spr_val)
                        + ((2808.0 / 25.0) * rgePow(g1_val, 4.0)
                        * rgePow(M1_val, 2.0))
                        + ((12.0 / 5.0) * rgePow(g1_val, 2.0) * sigma1));

    // Total scalar squared mass beta functions
    Real dmHu_sq_dt = ((rgeLoopFactor * dmHu_sq_dt_1l) + (rgeLoopFactorSquared * dmHu_sq_dt_2l));
    Real dmHd_sq_dt = ((rgeLoopFactor * dmHd_sq_dt_1l) + (rgeLoopFactorSquared * dmHd_sq_dt_2l));
    Real dmQ3_sq_dt = ((rgeLoopFactor * dmQ3_sq_dt_1l) + (rgeLoopFactorSquared * dmQ3_sq_dt_2l));
    Real dmQ2_sq_dt = ((rgeLoopFactor * dmQ2_sq_dt_1l) + (rgeLoopFactorSquared * dmQ2_sq_dt_2l));
    Real dmQ1_sq_dt = ((rgeLoopFactor * dmQ1_sq_dt_1l) + (rgeLoopFactorSquared * dmQ1_sq_dt_2l));
    Real dmL3_sq_dt = ((rgeLoopFactor * dmL3_sq_dt_1l) + (rgeLoopFactorSquared * dmL3_sq_dt_2l));
    Real dmL2_sq_dt = ((rgeLoopFactor * dmL2_sq_dt_1l) + (rgeLoopFactorSquared * dmL2_sq_dt_2l));
    Real dmL1_sq_dt = ((rgeLoopFactor * dmL1_sq_dt_1l) + (rgeLoopFactorSquared * dmL1_sq_dt_2l));
    Real dmU3_sq_dt = ((rgeLoopFactor * dmU3_sq_dt_1l) + (rgeLoopFactorSquared * dmU3_sq_dt_2l));
    Real dmU2_sq_dt = ((rgeLoopFactor * dmU2_sq_dt_1l) + (rgeLoopFactorSquared * dmU2_sq_dt_2l));
    Real dmU1_sq_dt = ((rgeLoopFactor * dmU1_sq_dt_1l) + (rgeLoopFactorSquared * dmU1_sq_dt_2l));
    Real dmD3_sq_dt = ((rgeLoopFactor * dmD3_sq_dt_1l) + (rgeLoopFactorSquared * dmD3_sq_dt_2l));
    Real dmD2_sq_dt = ((rgeLoopFactor * dmD2_sq_dt_1l) + (rgeLoopFactorSquared * dmD2_sq_dt_2l));
    Real dmD1_sq_dt = ((rgeLoopFactor * dmD1_sq_dt_1l) + (rgeLoopFactorSquared * dmD1_sq_dt_2l));
    Real dmE3_sq_dt = ((rgeLoopFactor * dmE3_sq_dt_1l) + (rgeLoopFactorSquared * dmE3_sq_dt_2l));
    Real dmE2_sq_dt = ((rgeLoopFactor * dmE2_sq_dt_1l) + (rgeLoopFactorSquared * dmE2_sq_dt_2l));
    Real dmE1_sq_dt = ((rgeLoopFactor * dmE1_sq_dt_1l) + (rgeLoopFactorSquared * dmE1_sq_dt_2l));

    // tanb beta function at one-loop order here
    Real dtanb_dt = 3.0 * rgeLoopFactor * tanb_val
        * (rgePow(yb_val, 2.0) - rgePow(yt_val, 2.0));

    dxdt[0] = dg1_dt;
    dxdt[1] = dg2_dt;
    dxdt[2] = dg3_dt;
    dxdt[3] = dM1_dt;
    dxdt[4] = dM2_dt;
    dxdt[5] = dM3_dt;
    dxdt[6] = dmu_dt;
    dxdt[7] = dyt_dt;
    dxdt[8] = dyc_dt;
    dxdt[9] = dyu_dt;
    dxdt[10] = dyb_dt;
    dxdt[11] = dys_dt;
    dxdt[12] = dyd_dt;
    dxdt[13] = dytau_dt;
    dxdt[14] = dymu_dt;
    dxdt[15] = dye_dt;
    dxdt[16] = dat_dt;
    dxdt[17] = dac_dt;
    dxdt[18] = dau_dt;
    dxdt[19] = dab_dt;
    dxdt[20] = das_dt;
    dxdt[21] = dad_dt;
    dxdt[22] = datau_dt;
    dxdt[23] = damu_dt;
    dxdt[24] = dae_dt;
    dxdt[25] = dmHu_sq_dt;
    dxdt[26] = dmHd_sq_dt;
    dxdt[27] = dmQ1_sq_dt;
    dxdt[28] = dmQ2_sq_dt;
    dxdt[29] = dmQ3_sq_dt;
    dxdt[30] = dmL1_sq_dt;
    dxdt[31] = dmL2_sq_dt;
    dxdt[32] = dmL3_sq_dt;
    dxdt[33] = dmU1_sq_dt;
    dxdt[34] = dmU2_sq_dt;
    dxdt[35] = dmU3_sq_dt;
    dxdt[36] = dmD1_sq_dt;
    dxdt[37] = dmD2_sq_dt;
    dxdt[38] = dmD3_sq_dt;
    dxdt[39] = dmE1_sq_dt;
    dxdt[40] = dmE2_sq_dt;
    dxdt[41] = dmE3_sq_dt;
    dxdt[42] = db_dt;
    dxdt[43] = dtanb_dt;
}


#undef NATLHA_RGE_HD

#endif  // NATLHA_MSSM_RGE_DERIVATIVES_INL
