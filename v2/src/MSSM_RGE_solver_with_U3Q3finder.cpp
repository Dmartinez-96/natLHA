#include "MSSM_RGE_solver.hpp"
#include "MSSM_RGE_solver_with_U3Q3finder.hpp"
#include "constants.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <array>
#include <limits>
#include <algorithm>
#include <utility>
#include <boost/numeric/odeint/stepper/runge_kutta_dopri5.hpp>
#include <boost/numeric/odeint/stepper/generation.hpp>
#include <boost/numeric/odeint/iterator/adaptive_iterator.hpp>
#include <boost/numeric/odeint.hpp>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void MSSM_approx_RGESolver(const std::vector<double>& x, std::vector<double>& dxdt, const double t ) {
    // Extract values from the input vector x
    double g1_val = x[0];
    double g2_val = x[1];
    double g3_val = x[2];
    double M1_val = x[3];
    double M2_val = x[4];
    double M3_val = x[5];
    double mu_val = x[6];
    double yt_val = x[7];
    double yc_val = x[8];
    double yu_val = x[9];
    double yb_val = x[10];
    double ys_val = x[11];
    double yd_val = x[12];
    double ytau_val = x[13];
    double ymu_val = x[14];
    double ye_val = x[15];
    double at_val = x[16];
    double ac_val = x[17];
    double au_val = x[18];
    double ab_val = x[19];
    double as_val = x[20];
    double ad_val = x[21];
    double atau_val = x[22];
    double amu_val = x[23];
    double ae_val = x[24];
    double mHu_sq_val = x[25];
    double mHd_sq_val = x[26];
    double mQ1_sq_val = x[27];
    double mQ2_sq_val = x[28];
    double mQ3_sq_val = x[29];
    double mL1_sq_val = x[30];
    double mL2_sq_val = x[31];
    double mL3_sq_val = x[32];
    double mU1_sq_val = x[33];
    double mU2_sq_val = x[34];
    double mU3_sq_val = x[35];
    double mD1_sq_val = x[36];
    double mD2_sq_val = x[37];
    double mD3_sq_val = x[38];
    double mE1_sq_val = x[39];
    double mE2_sq_val = x[40];
    double mE3_sq_val = x[41];
    double b_val = x[42];
    double tanb_val = x[43];

    // Gauge coupling and gaugino mass beta functions
    /////////////////////////////////////////////////
    // 1-loop
    double dg1_dt_1l = b_1l[0] * std::pow(g1_val, 3.0);
    double dg2_dt_1l = b_1l[1] * std::pow(g2_val, 3.0);
    double dg3_dt_1l = b_1l[2] * std::pow(g3_val, 3.0);
    double dM1_dt_1l = b_1l[0] * 2.0 * std::pow(g1_val, 2.0) * M1_val;
    double dM2_dt_1l = b_1l[1] * 2.0 * std::pow(g2_val, 2.0) * M2_val;
    double dM3_dt_1l = b_1l[2] * 2.0 * std::pow(g3_val, 2.0) * M3_val;

    // 2-loop
    double dg1_dt_2l = (std::pow(g1_val, 3.0)
                        * ((b_2l[0][0] * std::pow(g1_val, 2.0))
                            + (b_2l[0][1] * std::pow(g2_val, 2.0))
                            + (b_2l[0][2] * std::pow(g3_val, 2.0)) // Tr(Yu^2)
                            - (c_2l[0][0] * (std::pow(yt_val, 2.0)
                                                + std::pow(yc_val, 2.0)
                                                + std::pow(yu_val, 2.0))) // end trace, begin Tr(Yd^2)
                            - (c_2l[0][1] * (std::pow(yb_val, 2.0)
                                                + std::pow(ys_val, 2.0)
                                                + std::pow(yd_val, 2.0))) // end trace, begin Tr(Ye^2)
                            - (c_2l[0][2] * (std::pow(ytau_val, 2.0)
                                                + std::pow(ymu_val, 2.0)
                                                + std::pow(ye_val, 2.0))))); // end trace
    double dg2_dt_2l = (std::pow(g2_val, 3.0)
                        * ((b_2l[1][0] * std::pow(g1_val, 2.0))
                            + (b_2l[1][1] * std::pow(g2_val, 2.0))
                            + (b_2l[1][2] * std::pow(g3_val, 2.0)) // Tr(Yu^2)
                            - (c_2l[1][0] * (std::pow(yt_val, 2.0)
                                                + std::pow(yc_val, 2.0)
                                                + std::pow(yu_val, 2.0))) // end trace, begin Tr(Yd^2)
                            - (c_2l[1][1] * (std::pow(yb_val, 2.0)
                                                + std::pow(ys_val, 2.0)
                                                + std::pow(yd_val, 2.0))) // end trace, begin Tr(Ye^2)
                            - (c_2l[1][2] * (std::pow(ytau_val, 2.0)
                                                + std::pow(ymu_val, 2.0)
                                                + std::pow(ye_val, 2.0))))); // end trace;
    double dg3_dt_2l = (std::pow(g3_val, 3.0)
                        * ((b_2l[2][0] * std::pow(g1_val, 2.0))
                            + (b_2l[2][1] * std::pow(g2_val, 2.0))
                            + (b_2l[2][2] * std::pow(g3_val, 2.0)) // Tr(Yu^2)
                            - (c_2l[2][0] * (std::pow(yt_val, 2.0)
                                                + std::pow(yc_val, 2.0)
                                                + std::pow(yu_val, 2.0))) // end trace, begin Tr(Yd^2)
                            - (c_2l[2][1] * (std::pow(yb_val, 2.0)
                                                + std::pow(ys_val, 2.0)
                                                + std::pow(yd_val, 2.0))) // end trace, begin Tr(Ye^2)
                            - (c_2l[2][2] * (std::pow(ytau_val, 2.0)
                                                + std::pow(ymu_val, 2.0)
                                                + std::pow(ye_val, 2.0))))); // end trace
;
    double dM1_dt_2l = (2.0 * std::pow(g1_val, 2.0)
                * (((b_2l[0][0] * std::pow(g1_val, 2.0) * (M1_val + M1_val))
                        + (b_2l[0][1] * std::pow(g2_val, 2.0)
                        * (M1_val + M2_val))
                        + (b_2l[0][2] * std::pow(g3_val, 2.0)
                        * (M1_val + M3_val))) // Tr(Yu*au)
                    + ((c_2l[0][0] * (((yt_val * at_val)
                                        + (yc_val * ac_val)
                                        + (yu_val * au_val)) // end trace, begin Tr(Yu^2)
                                    - (M1_val * (std::pow(yt_val, 2.0)
                                                    + std::pow(yc_val, 2.0)
                                                    + std::pow(yu_val, 2.0))) // end trace
                                    ))) // Tr(Yd*ad)
                    + ((c_2l[0][1] * (((yb_val * ab_val)
                                        + (ys_val * as_val)
                                        + (yd_val * ad_val)) // end trace, begin Tr(Yd^2)
                                    - (M1_val * (std::pow(yb_val, 2.0)
                                                    + std::pow(ys_val, 2.0)
                                                    + std::pow(yd_val, 2.0))) // end trace
                                    ))) // Tr(Ye*ae)
                    + ((c_2l[0][2] * (((ytau_val * atau_val)
                                        + (ymu_val * amu_val)
                                        + (ye_val * ae_val)) // end trace, begin Tr(Ye^2)
                                    - (M1_val * (std::pow(ytau_val, 2.0)
                                                    + std::pow(ymu_val, 2.0)
                                                    + std::pow(ye_val, 2.0)))
                                    ))))); // end trace
    double dM2_dt_2l = (2.0 * std::pow(g2_val, 2.0)
                * (((b_2l[1][0] * std::pow(g1_val, 2.0) * (M2_val + M1_val))
                        + (b_2l[1][1] * std::pow(g2_val, 2.0)
                        * (M2_val + M2_val))
                        + (b_2l[1][2] * std::pow(g3_val, 2.0)
                        * (M2_val + M3_val))) // Tr(Yu*au)
                    + ((c_2l[1][0] * (((yt_val * at_val)
                                        + (yc_val * ac_val)
                                        + (yu_val * au_val)) // end trace, begin Tr(Yu^2)
                                    - (M2_val * (std::pow(yt_val, 2.0)
                                                    + std::pow(yc_val, 2.0)
                                                    + std::pow(yu_val, 2.0))) // end trace
                                    ))) // Tr(Yd*ad)
                    + ((c_2l[1][1] * (((yb_val * ab_val)
                                        + (ys_val * as_val)
                                        + (yd_val * ad_val)) // end trace, begin Tr(Yd^2)
                                    - (M2_val * (std::pow(yb_val, 2.0)
                                                    + std::pow(ys_val, 2.0)
                                                    + std::pow(yd_val, 2.0))) // end trace
                                    ))) // Tr(Ye*ae)
                    + ((c_2l[1][2] * (((ytau_val * atau_val)
                                        + (ymu_val * amu_val)
                                        + (ye_val * ae_val)) // end trace, begin Tr(Ye^2)
                                    - (M2_val * (std::pow(ytau_val, 2.0)
                                                    + std::pow(ymu_val, 2.0)
                                                    + std::pow(ye_val, 2.0))) // end trace
                                    )))));
    double dM3_dt_2l = (2.0 * std::pow(g3_val, 2.0)
                * (((b_2l[2][0] * std::pow(g1_val, 2.0) * (M3_val + M1_val))
                        + (b_2l[2][1] * std::pow(g2_val, 2.0)
                        * (M3_val + M2_val))
                        + (b_2l[2][2] * std::pow(g3_val, 2.0)
                        * (M3_val + M3_val))) // Tr(Yu*au)
                    + ((c_2l[2][0] * (((yt_val * at_val)
                                        + (yc_val * ac_val)
                                        + (yu_val * au_val)) // end trace, begin Tr(Yu^2)
                                    - (M3_val * (std::pow(yt_val, 2.0)
                                                    + std::pow(yc_val, 2.0)
                                                    + std::pow(yu_val, 2.0))) // end trace
                                    ))) // Tr(Yd*ad)
                    + ((c_2l[2][1] * (((yb_val * ab_val)
                                        + (ys_val * as_val)
                                        + (yd_val * ad_val)) // end trace, begin Tr(Yd^2)
                                    - (M3_val * (std::pow(yb_val, 2.0)
                                                    + std::pow(ys_val, 2.0)
                                                    + std::pow(yd_val, 2.0))) // end trace
                                    ))) // Tr(Ye*ae)
                    + ((c_2l[2][2] * (((ytau_val * atau_val)
                                        + (ymu_val * amu_val)
                                        + (ye_val * ae_val)) // end trace, begin Tr(Ye^2)
                                    - (M3_val * (std::pow(ytau_val, 2.0)
                                                    + std::pow(ymu_val, 2.0)
                                                    + std::pow(ye_val, 2.0))) // end trace
                                    )))));


    // Calculate total gauge coupling and gaugino mass beta functions
    double dg1_dt = loop_fac * dg1_dt_1l + loop_fac_sq * dg1_dt_2l;
    double dg2_dt = loop_fac * dg2_dt_1l + loop_fac_sq * dg2_dt_2l;
    double dg3_dt = loop_fac * dg3_dt_1l + loop_fac_sq * dg3_dt_2l;
    double dM1_dt = loop_fac * dM1_dt_1l + loop_fac_sq * dM1_dt_2l;
    double dM2_dt = loop_fac * dM2_dt_1l + loop_fac_sq * dM2_dt_2l;
    double dM3_dt = loop_fac * dM3_dt_1l + loop_fac_sq * dM3_dt_2l;
    
    // Higgsino mass parameter mu
    ////////////////////////////////////////////////////////////////////
    // 1-loop
    double dmu_dt_1l = (mu_val // Tr(3Yu^2 + 3Yd^2 + Ye^2)
                * ((3.0 * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                        + std::pow(yu_val, 2.0) + std::pow(yb_val, 2.0)
                        + std::pow(ys_val, 2.0) + std::pow(yd_val, 2.0)))
                    + (std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                        + std::pow(ye_val, 2.0)) // end trace
                    - (3.0 * std::pow(g2_val, 2.0))
                    - ((3.0 / 5.0) * std::pow(g1_val, 2.0))));

    // 2-loop
    double dmu_dt_2l = (mu_val // Tr(3Yu^4 + 3Yd^4 + (2Yu^2*Yd^2) + Ye^4)
                * ((-3.0 * ((3.0 * (std::pow(yt_val, 4.0) + std::pow(yc_val, 4.0)
                                    + std::pow(yu_val, 4.0)
                                    + std::pow(yb_val, 4.0)
                                    + std::pow(ys_val, 4.0)
                                    + std::pow(yd_val, 4.0)))
                            + (2.0 * ((std::pow(yt_val, 2.0)
                                    * std::pow(yb_val, 2.0))
                                    + (std::pow(yc_val, 2.0)
                                    * std::pow(ys_val, 2.0))
                                    + (std::pow(yu_val, 2.0)
                                    * std::pow(yd_val, 2.0))))
                            + (std::pow(ytau_val, 4.0) + std::pow(ymu_val, 4.0)
                                + std::pow(ye_val, 4.0)))) // end trace
                    + (((16.0 * std::pow(g3_val, 2.0))
                        + (4.0 * std::pow(g1_val, 2.0) / 5.0)) // Tr(Yu^2)
                        * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                        + std::pow(yu_val, 2.0))) // end trace
                    + (((16.0 * std::pow(g3_val, 2.0))
                        - (2.0 * std::pow(g1_val, 2.0) / 5.0)) // Tr(Yd^2)
                        * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                        + std::pow(yd_val, 2.0))) // end trace
                    + ((6.0 / 5.0) * std::pow(g1_val, 2.0) // Tr(Ye^2)
                        * (std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                        + std::pow(ye_val, 2.0))) // end trace
                    + ((15.0 / 2.0) * std::pow(g2_val, 4.0))
                    + ((9.0 / 5.0) * std::pow(g1_val, 2.0)
                        * std::pow(g2_val, 2.0))
                    + ((207.0 / 50.0) * std::pow(g1_val, 4.0))));

    // Calculate total gauge coupling and gaugino mass beta functions
    double dmu_dt = loop_fac * dmu_dt_1l + loop_fac_sq * dmu_dt_2l;

    // Yukawa couplings for all 3 generations, assumed diagonalized
    //////////////////////////////////////////////////////////////////////////
    // 1-loop
    double dyt_dt_1l = (yt_val // Tr(3Yu^2)
                * ((3.0 * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                        + std::pow(yu_val, 2.0))) // end trace
                    + (3.0 * (std::pow(yt_val, 2.0)))
                    + std::pow(yb_val, 2.0)
                    - ((16.0 / 3.0) * std::pow(g3_val, 2.0))
                    - (3.0 * std::pow(g2_val, 2.0))
                    - ((13.0 / 15.0) * std::pow(g1_val, 2.0))));
    double dyc_dt_1l = (yc_val // Tr(3Yu^2)
                * ((3.0 * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                        + std::pow(yu_val, 2.0))) // end trace
                    + (3.0 * (std::pow(yc_val, 2.0)))
                    + std::pow(ys_val, 2.0)
                    - ((16.0 / 3.0) * std::pow(g3_val, 2.0))
                    - (3.0 * std::pow(g2_val, 2.0))
                    - ((13.0 / 15.0) * std::pow(g1_val, 2.0))));
    double dyu_dt_1l = (yu_val // Tr(3Yu^2)
                * ((3.0 * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                        + std::pow(yu_val, 2.0))) // end trace
                    + (3.0 * (std::pow(yu_val, 2.0)))
                    + std::pow(yd_val, 2.0)
                    - ((16.0 / 3.0) * std::pow(g3_val, 2.0))
                    - (3.0 * std::pow(g2_val, 2.0))
                    - ((13.0 / 15.0) * std::pow(g1_val, 2.0))));
    
    double dyb_dt_1l = (yb_val // Tr(3Yd^2 + Ye^2)
                * ((3.0 * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                        + std::pow(yd_val, 2.0)))
                    + (std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                        + std::pow(ye_val, 2.0)) // end trace
                    + (3.0 * (std::pow(yb_val, 2.0))) + std::pow(yt_val, 2.0)
                    - ((16.0 / 3.0) * std::pow(g3_val, 2.0))
                    - (3.0 * std::pow(g2_val, 2.0))
                    - ((7.0 / 15.0) * std::pow(g1_val, 2.0))));
    
    double dys_dt_1l = (ys_val // Tr(3Yd^2 + Ye^2)
                * ((3.0 * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                        + std::pow(yd_val, 2.0)))
                    + (std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                        + std::pow(ye_val, 2.0)) // end trace
                    + (3.0 * (std::pow(ys_val, 2.0))) + std::pow(yc_val, 2.0)
                    - ((16.0 / 3.0) * std::pow(g3_val, 2.0))
                    - (3.0 * std::pow(g2_val, 2.0))
                    - ((7.0 / 15.0) * std::pow(g1_val, 2.0))));
    
    double dyd_dt_1l = (yd_val // Tr(3Yd^2 + Ye^2)
                * ((3.0 * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                        + std::pow(yd_val, 2.0)))
                    + (std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                        + std::pow(ye_val, 2.0)) // end trace
                    + (3.0 * (std::pow(yd_val, 2.0))) + std::pow(yu_val, 2.0)
                    - ((16.0 / 3.0) * std::pow(g3_val, 2.0))
                    - (3.0 * std::pow(g2_val, 2.0))
                    - ((7.0 / 15.0) * std::pow(g1_val, 2.0))));

    double dytau_dt_1l = (ytau_val // Tr(3Yd^2 + Ye^2)
                    * ((3.0 * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                            + std::pow(yd_val, 2.0)))
                        + (std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                        + std::pow(ye_val, 2.0)) // end trace
                        + (3.0 * (std::pow(ytau_val, 2.0)))
                        - (3.0 * std::pow(g2_val, 2.0))
                        - ((9.0 / 5.0) * std::pow(g1_val, 2.0))));

    double dymu_dt_1l = (ymu_val // Tr(3Yd^2 + Ye^2)
                    * ((3.0 * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                            + std::pow(yd_val, 2.0)))
                        + (std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                        + std::pow(ye_val, 2.0)) // end trace
                        + (3.0 * (std::pow(ymu_val, 2.0)))
                        - (3.0 * std::pow(g2_val, 2.0))
                        - ((9.0 / 5.0) * std::pow(g1_val, 2.0))));

    double dye_dt_1l = (ye_val // Tr(3Yd^2 + Ye^2)
                * ((3.0 * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                        + std::pow(yd_val, 2.0)))
                    + (std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                        + std::pow(ye_val, 2.0)) // end trace
                    + (3.0 * (std::pow(ye_val, 2.0)))
                    - (3.0 * std::pow(g2_val, 2.0))
                    - ((9.0 / 5.0) * std::pow(g1_val, 2.0))));

    // 2-loop
    double dyt_dt_2l = (yt_val  // Tr(3Yu^4 + (Yu^2*Yd^2))
                * (((-3.0) * ((3.0 * (std::pow(yt_val, 4.0)
                                    + std::pow(yc_val, 4.0)
                                    + std::pow(yu_val, 4.0)))
                            + (std::pow(yt_val, 2.0) * std::pow(yb_val, 2.0))
                            + (std::pow(yc_val, 2.0) * std::pow(ys_val, 2.0))
                            + (std::pow(yu_val, 2.0) * std::pow(yd_val, 2.0)))
                        ) // end trace
                    - (std::pow(yb_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (std::pow(yb_val, 2.0)
                                + std::pow(ys_val, 2.0)
                                + std::pow(yd_val, 2.0)))
                        + (std::pow(ytau_val, 2.0)
                            + std::pow(ymu_val, 2.0)
                            + std::pow(ye_val, 2.0)))) // end trace
                    - (9.0 * std::pow(yt_val, 2.0) // Tr(Yu^2)
                        * (std::pow(yt_val, 2.0)
                        + std::pow(yc_val, 2.0)
                        + std::pow(yu_val, 2.0))) // end trace
                    - (4.0 * std::pow(yt_val, 4.0) )
                    - (2.0 * std::pow(yb_val, 4.0) )
                    - (2.0 * std::pow(yb_val, 2.0) * std::pow(yt_val, 2.0))
                    + (((16.0 *  std::pow(g3_val, 2.0))
                        + ((4.0 / 5.0) * std::pow(g1_val, 2.0))) // Tr(Yu^2)
                        * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                        + std::pow(yu_val, 2.0))) // end trace
                    + (((6.0 * std::pow(g2_val, 2.0))
                        + ((2.0 / 5.0) * std::pow(g1_val, 2.0)))
                        * std::pow(yt_val, 2.0))
                    + ((2.0 / 5.0) * std::pow(g1_val, 2.0) * std::pow(yb_val, 2.0))
                    - ((16.0 / 9.0) * std::pow(g3_val, 4.0) )
                    + (8.0 * std::pow(g3_val, 2.0) * std::pow(g2_val, 2.0))
                    + ((136.0 / 45.0) * std::pow(g3_val, 2.0)
                        * std::pow(g1_val, 2.0))
                    + ((15.0 / 2.0) * std::pow(g2_val, 4.0) )
                    + (std::pow(g2_val, 2.0) * std::pow(g1_val, 2.0))
                    + ((2743.0 / 450.0) * std::pow(g1_val, 4.0) )));

    double dyc_dt_2l = (yc_val  // Tr(3Yu^4 + (Yu^2*Yd^2))
                * (((-3.0) * ((3.0 * (std::pow(yt_val, 4.0) 
                                    + std::pow(yc_val, 4.0) 
                                    + std::pow(yu_val, 4.0) ))
                            + (std::pow(yt_val, 2.0) * std::pow(yb_val, 2.0))
                            + (std::pow(yc_val, 2.0) * std::pow(ys_val, 2.0))
                            + (std::pow(yu_val, 2.0) * std::pow(yd_val, 2.0)))
                        ) //end trace
                    - (std::pow(ys_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (std::pow(yb_val, 2.0)
                                + std::pow(ys_val, 2.0)
                                + std::pow(yd_val, 2.0)))
                        + (std::pow(ytau_val, 2.0)
                            + std::pow(ymu_val, 2.0)
                            + std::pow(ye_val, 2.0)))) // end trace
                    - (9.0 * std::pow(yc_val, 2.0) // Tr(Yu^2)
                        * (std::pow(yt_val, 2.0)
                        + std::pow(yc_val, 2.0)
                        + std::pow(yu_val, 2.0))) // end trace
                    - (4.0 * std::pow(yc_val, 4.0) )
                    - (2.0 * std::pow(ys_val, 4.0) )
                    - (2.0 * std::pow(ys_val, 2.0)
                        * std::pow(yc_val, 2.0))
                    + (((16.0 *  std::pow(g3_val, 2.0))
                        + ((4.0 / 5.0) * std::pow(g1_val, 2.0))) // Tr(Yu^2)
                        * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                        + std::pow(yu_val, 2.0))) // end trace
                    + (((6.0 * std::pow(g2_val, 2.0))
                        + ((2.0 / 5.0) * std::pow(g1_val, 2.0)))
                        * std::pow(yc_val, 2.0))
                    + ((2.0 / 5.0) * std::pow(g1_val, 2.0) * std::pow(ys_val, 2.0))
                    - ((16.0 / 9.0) * std::pow(g3_val, 4.0) )
                    + (8.0 * std::pow(g3_val, 2.0) * std::pow(g2_val, 2.0))
                    + ((136.0 / 45.0) * std::pow(g3_val, 2.0)
                        * std::pow(g1_val, 2.0))
                    + ((15.0 / 2.0) * std::pow(g2_val, 4.0) )
                    + (std::pow(g2_val, 2.0) * std::pow(g1_val, 2.0))
                    + ((2743.0 / 450.0) * std::pow(g1_val, 4.0) )));

    double dyu_dt_2l = (yu_val // Tr(3Yu^4 + (Yu^2*Yd^2))
                * (((-3.0) * ((3.0 * (std::pow(yt_val, 4.0) 
                                    + std::pow(yc_val, 4.0) 
                                    + std::pow(yu_val, 4.0) ))
                            + (std::pow(yt_val, 2.0) * std::pow(yb_val, 2.0))
                            + (std::pow(yc_val, 2.0) * std::pow(ys_val, 2.0))
                            + (std::pow(yu_val, 2.0) * std::pow(yd_val, 2.0)))
                        ) // end trace
                    - (std::pow(yd_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (std::pow(yb_val, 2.0)
                                + std::pow(ys_val, 2.0)
                                + std::pow(yd_val, 2.0)))
                        + (std::pow(ytau_val, 2.0)
                            + std::pow(ymu_val, 2.0)
                            + std::pow(ye_val, 2.0))))
                    - (9.0 * std::pow(yu_val, 2.0) // Tr(Yu^2)
                        * (std::pow(yt_val, 2.0)
                        + std::pow(yc_val, 2.0)
                        + std::pow(yu_val, 2.0)))
                    - (4.0 * std::pow(yu_val, 4.0) )
                    - (2.0 * std::pow(yd_val, 4.0) )
                    - (2.0 * std::pow(yd_val, 2.0) * std::pow(yu_val, 2.0))
                    + (((16.0 *  std::pow(g3_val, 2.0))
                        + ((4.0 / 5.0) * std::pow(g1_val, 2.0))) // Tr(Yu^2)
                        * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                        + std::pow(yu_val, 2.0))) // end trace
                    + (((6.0 * std::pow(g2_val, 2.0))
                        + ((2.0 / 5.0) * std::pow(g1_val, 2.0)))
                        * std::pow(yu_val, 2.0))
                    + ((2.0 / 5.0) * std::pow(g1_val, 2.0) * std::pow(yd_val, 2.0))
                    - ((16.0 / 9.0) * std::pow(g3_val, 4.0) )
                    + (8.0 * std::pow(g3_val, 2.0) * std::pow(g2_val, 2.0))
                    + ((136.0 / 45.0) * std::pow(g3_val, 2.0)
                        * std::pow(g1_val, 2.0))
                    + ((15.0 / 2.0) * std::pow(g2_val, 4.0) )
                    + (std::pow(g2_val, 2.0) * std::pow(g1_val, 2.0))
                    + ((2743.0 / 450.0) * std::pow(g1_val, 4.0) )));

    double dyb_dt_2l = (yb_val // Tr(3Yd^4 + (Yu^2*Yd^2) + Ye^4)
                * (((-3.0) * ((3.0 * (std::pow(yb_val, 4.0) 
                                    + std::pow(ys_val, 4.0) 
                                    + std::pow(yd_val, 4.0) ))
                            + (std::pow(yt_val, 2.0) * std::pow(yb_val, 2.0))
                            + (std::pow(yc_val, 2.0) * std::pow(ys_val, 2.0))
                            + (std::pow(yu_val, 2.0) * std::pow(yd_val, 2.0))
                            + std::pow(ytau_val, 4.0)  + std::pow(ymu_val, 4.0) 
                            + std::pow(ye_val, 4.0) )) // end trace
                    - (3.0 * std::pow(yt_val, 2.0) // Tr(Yu^2)
                        * (std::pow(yt_val, 2.0)
                        + std::pow(yc_val, 2.0)
                        + std::pow(yu_val, 2.0))) // end trace
                    - (3.0 * std::pow(yb_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (std::pow(yb_val, 2.0)
                                + std::pow(ys_val, 2.0)
                                + std::pow(yd_val, 2.0)))
                        + std::pow(ytau_val, 2.0)
                        + std::pow(ymu_val, 2.0)
                        + std::pow(ye_val, 2.0))) // end trace
                    - (4.0 * std::pow(yb_val, 4.0) )
                    - (2.0 * std::pow(yt_val, 4.0) )
                    - (2.0 * std::pow(yt_val, 2.0) * std::pow(yb_val, 2.0))
                    + (((16.0 *  std::pow(g3_val, 2.0))
                        - ((2.0 / 5.0) * std::pow(g1_val, 2.0))) // Tr(Yd^2)
                        * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                        + std::pow(yd_val, 2.0))) // end trace
                    + ((6.0 / 5.0) * std::pow(g1_val, 2.0) // Tr(Ye^2)
                        * (std::pow(ytau_val, 2.0)
                        + std::pow(ymu_val, 2.0)
                        + std::pow(ye_val, 2.0))) // end trace
                    + ((4.0 / 5.0) * std::pow(g1_val, 2.0)
                        * std::pow(yt_val, 2.0))
                    + (std::pow(yb_val, 2.0)
                        * ((6.0 * std::pow(g2_val, 2.0))
                        + ((4.0 / 5.0) * std::pow(g1_val, 2.0))))
                    - ((16.0 / 9.0) * std::pow(g3_val, 4.0) )
                    + (8.0 * std::pow(g3_val, 2.0) * std::pow(g2_val, 2.0))
                    + ((8.0 / 9.0) * std::pow(g3_val, 2.0) * std::pow(g1_val, 2.0))
                    + ((15.0 / 2.0) * std::pow(g2_val, 4.0) )
                    + (std::pow(g2_val, 2.0) * std::pow(g1_val, 2.0))
                    + ((287.0 / 90.0) * std::pow(g1_val, 4.0) )));

    double dys_dt_2l = (ys_val // Tr(3Yd^4 + (Yu^2*Yd^2) + Ye^4)
                * (((-3.0) * ((3.0 * (std::pow(yb_val, 4.0) 
                                    + std::pow(ys_val, 4.0) 
                                    + std::pow(yd_val, 4.0) ))
                            + (std::pow(yt_val, 2.0) * std::pow(yb_val, 2.0))
                            + (std::pow(yc_val, 2.0) * std::pow(ys_val, 2.0))
                            + (std::pow(yu_val, 2.0) * std::pow(yd_val, 2.0))
                            + std::pow(ytau_val, 4.0) 
                            + std::pow(ymu_val, 4.0) 
                            + std::pow(ye_val, 4.0) )) // end trace
                    - (3.0 * std::pow(yc_val, 2.0) // Tr(Yu^2)
                        * (std::pow(yt_val, 2.0)
                        + std::pow(yc_val, 2.0)
                        + std::pow(yu_val, 2.0))) // end trace
                    - (3.0 * std::pow(ys_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (std::pow(yb_val, 2.0)
                                + std::pow(ys_val, 2.0)
                                + std::pow(yd_val, 2.0)))
                        + std::pow(ytau_val, 2.0)
                        + std::pow(ymu_val, 2.0)
                        + std::pow(ye_val, 2.0))) // end trace
                    - (4.0 * std::pow(ys_val, 4.0) )
                    - (2.0 * std::pow(yc_val, 4.0) )
                    - (2.0 * std::pow(yc_val, 2.0) * std::pow(ys_val, 2.0))
                    + (((16.0 *  std::pow(g3_val, 2.0))
                        - ((2.0 / 5.0) * std::pow(g1_val, 2.0))) // Tr(Yd^2)
                        * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                        + std::pow(yd_val, 2.0))) // end trace
                    + ((6.0 / 5.0) * std::pow(g1_val, 2.0) // Tr(Ye^2)
                        * (std::pow(ytau_val, 2.0)
                        + std::pow(ymu_val, 2.0)
                        + std::pow(ye_val, 2.0))) // end trace
                    + ((4.0 / 5.0) * std::pow(g1_val, 2.0) * std::pow(yc_val, 2.0))
                    + (std::pow(ys_val, 2.0)
                        * ((6.0 * std::pow(g2_val, 2.0))
                        + ((4.0 / 5.0) * std::pow(g1_val, 2.0))))
                    - ((16.0 / 9.0) * std::pow(g3_val, 4.0) )
                    + (8.0 * std::pow(g3_val, 2.0) * std::pow(g2_val, 2.0))
                    + ((8.0 / 9.0) * std::pow(g3_val, 2.0) * std::pow(g1_val, 2.0))
                    + ((15.0 / 2.0) * std::pow(g2_val, 4.0) )
                    + (std::pow(g2_val, 2.0) * std::pow(g1_val, 2.0))
                    + ((287.0 / 90.0) * std::pow(g1_val, 4.0) )));

    double dyd_dt_2l = (yd_val // Tr(3Yd^4 + (Yu^2*Yd^2) + Ye^4)
                * (((-3.0) * ((3.0 * (std::pow(yb_val, 4.0) 
                                    + std::pow(ys_val, 4.0) 
                                    + std::pow(yd_val, 4.0) ))
                            + (std::pow(yt_val, 2.0) * std::pow(yb_val, 2.0))
                            + (std::pow(yc_val, 2.0) * std::pow(ys_val, 2.0))
                            + (std::pow(yu_val, 2.0) * std::pow(yd_val, 2.0))
                            + std::pow(ytau_val, 4.0)  + std::pow(ymu_val, 4.0) 
                            + std::pow(ye_val, 4.0) )) // end trace
                    - (3.0 * std::pow(yu_val, 2.0) // Tr(Yu^2)
                        * (std::pow(yt_val, 2.0)
                        + std::pow(yc_val, 2.0)
                        + std::pow(yu_val, 2.0))) // end trace
                    - (3.0 * std::pow(yd_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (std::pow(yb_val, 2.0)
                                + std::pow(ys_val, 2.0)
                                + std::pow(yd_val, 2.0)))
                        + std::pow(ytau_val, 2.0)
                        + std::pow(ymu_val, 2.0)
                        + std::pow(ye_val, 2.0))) // end trace
                    - (4.0 * std::pow(yd_val, 4.0) )
                    - (2.0 * std::pow(yu_val, 4.0) )
                    - (2.0 * std::pow(yd_val, 2.0) * std::pow(yu_val, 2.0))
                    + (((16.0 *  std::pow(g3_val, 2.0))
                        - ((2.0 / 5.0) * std::pow(g1_val, 2.0))) // Tr(Yd^2)
                        * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                        + std::pow(yd_val, 2.0))) // end trace
                    + ((6.0 / 5.0) * std::pow(g1_val, 2.0) // Tr(Ye^2)
                        * (std::pow(ytau_val, 2.0)
                        + std::pow(ymu_val, 2.0)
                        + std::pow(ye_val, 2.0))) // end trace
                    + ((4.0 / 5.0) * std::pow(g1_val, 2.0) * std::pow(yu_val, 2.0))
                    + (std::pow(yd_val, 2.0)
                        * ((6.0 * std::pow(g2_val, 2.0))
                        + ((4.0 / 5.0) * std::pow(g1_val, 2.0))))
                    - ((16.0 / 9.0) * std::pow(g3_val, 4.0) )
                    + (8.0 * std::pow(g3_val, 2.0) * std::pow(g2_val, 2.0))
                    + ((8.0 / 9.0) * std::pow(g3_val, 2.0) * std::pow(g1_val, 2.0))
                    + ((15.0 / 2.0) * std::pow(g2_val, 4.0) )
                    + (std::pow(g2_val, 2.0) * std::pow(g1_val, 2.0))
                    + ((287.0 / 90.0) * std::pow(g1_val, 4.0) )));

    double dytau_dt_2l = (ytau_val // Tr(3Yd^4 + (Yu^2*Yd^2) + Ye^4)
                    * (((-3.0) * ((3.0 * (std::pow(yb_val, 4.0) 
                                    + std::pow(ys_val, 4.0) 
                                    + std::pow(yd_val, 4.0) ))
                                + (std::pow(yt_val, 2.0)
                                    * std::pow(yb_val, 2.0))
                                + (std::pow(yc_val, 2.0)
                                    * std::pow(ys_val, 2.0))
                                + (std::pow(yu_val, 2.0)
                                    * std::pow(yd_val, 2.0))
                                + std::pow(ytau_val, 4.0) 
                                + std::pow(ymu_val, 4.0) 
                                + std::pow(ye_val, 4.0) )) // end trace
                        - (3.0 * std::pow(ytau_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (std::pow(yb_val, 2.0)
                                    + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + std::pow(ytau_val, 2.0)
                            + std::pow(ymu_val, 2.0)
                            + std::pow(ye_val, 2.0))) // end trace
                        - (4.0 * std::pow(ytau_val, 4.0) )
                        + (((16.0 *  std::pow(g3_val, 2.0))
                        - ((2.0 / 5.0) * std::pow(g1_val, 2.0))) // Tr(Yd^2)
                        * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                            + std::pow(yd_val, 2.0))) // end trace
                        + ((6.0 / 5.0) * std::pow(g1_val, 2.0) // Tr(Ye^2)
                        * (std::pow(ytau_val, 2.0)
                            + std::pow(ymu_val, 2.0)
                            + std::pow(ye_val, 2.0))) // end trace
                        + (6.0 * std::pow(g2_val, 2.0) * std::pow(ytau_val, 2.0))
                        + ((15.0 / 2.0) * std::pow(g2_val, 4.0) )
                        + ((9.0 / 5.0) * std::pow(g2_val, 2.0)
                        * std::pow(g1_val, 2.0))
                        + ((27.0 / 2.0) * std::pow(g1_val, 4.0) )));

    double dymu_dt_2l = (ymu_val // Tr(3Yd^4 + (Yu^2*Yd^2) + Ye^4)
                    * (((-3.0) * ((3.0 * (std::pow(yb_val, 4.0) 
                                    + std::pow(ys_val, 4.0) 
                                    + std::pow(yd_val, 4.0) ))
                                + (std::pow(yt_val, 2.0)
                                    * std::pow(yb_val, 2.0))
                                + (std::pow(yc_val, 2.0)
                                    * std::pow(ys_val, 2.0))
                                + (std::pow(yu_val, 2.0)
                                    * std::pow(yd_val, 2.0))
                                + std::pow(ytau_val, 4.0) 
                                + std::pow(ymu_val, 4.0) 
                                + std::pow(ye_val, 4.0) )) // end trace
                        - (3.0 * std::pow(ymu_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (std::pow(yb_val, 2.0)
                                    + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + std::pow(ytau_val, 2.0)
                            + std::pow(ymu_val, 2.0)
                            + std::pow(ye_val, 2.0))) // end trace
                        - (4.0 * std::pow(ymu_val, 4.0) )
                        + (((16.0 *  std::pow(g3_val, 2.0))
                        - ((2.0 / 5.0) * std::pow(g1_val, 2.0))) // Tr(Yd^2)
                        * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                            + std::pow(yd_val, 2.0))) // end trace
                        + ((6.0 / 5.0) * std::pow(g1_val, 2.0) // Tr(Ye^2)
                        * (std::pow(ytau_val, 2.0)
                            + std::pow(ymu_val, 2.0)
                            + std::pow(ye_val, 2.0))) // end trace
                        + (6.0 * std::pow(g2_val, 2.0) * std::pow(ymu_val, 2.0))
                        + ((15.0 / 2.0) * std::pow(g2_val, 4.0) )
                        + ((9.0 / 5.0) * std::pow(g2_val, 2.0) * std::pow(g1_val, 2.0))
                        + ((27.0 / 2.0) * std::pow(g1_val, 4.0) )));

    double dye_dt_2l = (ye_val // Tr(3Yd^4 + (Yu^2*Yd^2) + Ye^4)
                * (((-3.0) * ((3.0 * (std::pow(yb_val, 4.0) 
                                    + std::pow(ys_val, 4.0) 
                                    + std::pow(yd_val, 4.0) ))
                            + (std::pow(yt_val, 2.0)
                                * std::pow(yb_val, 2.0))
                            + (std::pow(yc_val, 2.0)
                                * std::pow(ys_val, 2.0))
                            + (std::pow(yu_val, 2.0)
                                * std::pow(yd_val, 2.0))
                            + std::pow(ytau_val, 4.0) 
                            + std::pow(ymu_val, 4.0) 
                            + std::pow(ye_val, 4.0) )) // end trace
                    - (3.0 * std::pow(ye_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (std::pow(yb_val, 2.0)
                                + std::pow(ys_val, 2.0)
                                + std::pow(yd_val, 2.0)))
                        + std::pow(ytau_val, 2.0)
                        + std::pow(ymu_val, 2.0)
                        + std::pow(ye_val, 2.0))) // end trace
                    - (4.0 * std::pow(ye_val, 4.0) )
                    + (((16.0 *  std::pow(g3_val, 2.0))
                        - ((2.0 / 5.0) * std::pow(g1_val, 2.0))) // Tr(Yd^2)
                        * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                        + std::pow(yd_val, 2.0))) // end trace
                    + ((6.0 / 5.0) * std::pow(g1_val, 2.0) // Tr(Ye^2)
                        * (std::pow(ytau_val, 2.0)
                        + std::pow(ymu_val, 2.0)
                        + std::pow(ye_val, 2.0)))
                    + (6.0 * std::pow(g2_val, 2.0) * std::pow(ye_val, 2.0))
                    + ((15.0 / 2.0) * std::pow(g2_val, 4.0) )
                    + ((9.0 / 5.0) * std::pow(g2_val, 2.0) * std::pow(g1_val, 2.0))
                    + ((27.0 / 2.0) * std::pow(g1_val, 4.0) )));

    // Total Yukawa coupling beta functions
    double dyt_dt = ((loop_fac * dyt_dt_1l) + (loop_fac_sq * dyt_dt_2l));

    double dyc_dt = ((loop_fac * dyc_dt_1l) + (loop_fac_sq * dyc_dt_2l));

    double dyu_dt = ((loop_fac * dyu_dt_1l) + (loop_fac_sq * dyu_dt_2l));

    double dyb_dt = ((loop_fac * dyb_dt_1l) + (loop_fac_sq * dyb_dt_2l));

    double dys_dt = ((loop_fac * dys_dt_1l) + (loop_fac_sq * dys_dt_2l));

    double dyd_dt = ((loop_fac * dyd_dt_1l) + (loop_fac_sq * dyd_dt_2l));

    double dytau_dt = ((loop_fac * dytau_dt_1l) + (loop_fac_sq * dytau_dt_2l));

    double dymu_dt = ((loop_fac * dymu_dt_1l) + (loop_fac_sq * dymu_dt_2l));

    double dye_dt = ((loop_fac * dye_dt_1l) + (loop_fac_sq * dye_dt_2l));

    // Soft trilinear couplings, assumed diagonalized
    /////////////////////////////////////////////////////////////////////
    // 1-loop
    double dat_dt_1l = ((at_val // Tr(Yu^2)
                    * ((3.0 * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                            + std::pow(yu_val, 2.0))) // end trace
                        + (5.0 * std::pow(yt_val, 2.0)) + std::pow(yb_val, 2.0)
                        - ((16.0 / 3.0) * std::pow(g3_val, 2.0))
                        - (3.0 * std::pow(g2_val, 2.0))
                        - ((13.0 / 15.0) * std::pow(g1_val, 2.0))))
                + (yt_val // Tr(au*Yu)
                    * ((6.0 * ((at_val * yt_val) + (ac_val * yc_val)
                            + (au_val * yu_val))) // end trace
                        + (4.0 * yt_val * at_val)
                        + (2.0 * yb_val * ab_val)
                        + ((32.0 / 3.0) * std::pow(g3_val, 2.0) * M3_val)
                        + (6.0 * std::pow(g2_val, 2.0) * M2_val)
                        + ((26.0 / 15.0) * std::pow(g1_val, 2.0) * M1_val))));

    double dac_dt_1l = ((ac_val // Tr(Yu^2)
                    * ((3.0 * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                            + std::pow(yu_val, 2.0))) // end trace
                        + (5.0 * std::pow(yc_val, 2.0)) + std::pow(ys_val, 2.0)
                        - ((16.0 / 3.0) * std::pow(g3_val, 2.0))
                        - (3.0 * std::pow(g2_val, 2.0))
                        - ((13.0 / 15.0) * std::pow(g1_val, 2.0))))
                + (yc_val // Tr(au*Yu)
                    * ((6.0 * ((at_val * yt_val) + (ac_val * yc_val)
                            + (au_val * yu_val))) // end trace
                        + (4.0 * yc_val * ac_val)
                        + (2.0 * ys_val * as_val)
                        + ((32.0 / 3.0) * std::pow(g3_val, 2.0) * M3_val)
                        + (6.0 * std::pow(g2_val, 2.0) * M2_val)
                        + ((26.0 / 15.0) * std::pow(g1_val, 2.0) * M1_val))));

    double dau_dt_1l = ((au_val // Tr(Yu^2)
                    * ((3.0 * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                            + std::pow(yu_val, 2.0))) // end trace
                        + (5.0 * std::pow(yu_val, 2.0)) + std::pow(yd_val, 2.0)
                        - ((16.0 / 3.0) * std::pow(g3_val, 2.0))
                        - (3.0 * std::pow(g2_val, 2.0))
                        - ((13.0 / 15.0) * std::pow(g1_val, 2.0))))
                + (yu_val // Tr(au*Yu)
                    * ((6.0 * ((at_val * yt_val) + (ac_val * yc_val)
                            + (au_val * yu_val))) // end trace
                        + (4.0 * yu_val * au_val)
                        + (2.0 * yd_val * ad_val)
                        + ((32.0 / 3.0) * std::pow(g3_val, 2.0) * M3_val)
                        + (6.0 * std::pow(g2_val, 2.0) * M2_val)
                        + ((26.0 / 15.0) * std::pow(g1_val, 2.0) * M1_val))));

    double dab_dt_1l = ((ab_val // Tr(3Yd^2 + Ye^2)
                    * (((3.0 * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                            + std::pow(yd_val, 2.0)))
                        + (std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                        + std::pow(ye_val, 2.0))) // end trace
                        + (5.0 * std::pow(yb_val, 2.0)) + std::pow(yt_val, 2.0)
                        - ((16.0 / 3.0) * std::pow(g3_val, 2.0))
                        - (3.0 * std::pow(g2_val, 2.0))
                        - ((7.0 / 15.0) * std::pow(g1_val, 2.0))))
                + (yb_val // Tr(6ad*Yd + 2ae*Ye)
                    * ((6.0 * ((ab_val * yb_val) + (as_val * ys_val)
                            + (ad_val * yd_val)))
                        + (2.0 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                + (ae_val * ye_val))) // end trace
                        + (4.0 * yb_val * ab_val) + (2.0 * yt_val * at_val)
                        + ((32.0 / 3.0) * std::pow(g3_val, 2.0) * M3_val)
                        + (6.0 * std::pow(g2_val, 2.0) * M2_val)
                        + ((14.0 / 15.0) * std::pow(g1_val, 2.0) * M1_val))));

    double das_dt_1l = ((as_val // Tr(3Yd^2 + Ye^2)
                    * (((3.0 * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                            + std::pow(yd_val, 2.0)))
                        + (std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                        + std::pow(ye_val, 2.0))) // end trace
                        + (5.0 * std::pow(ys_val, 2.0)) + std::pow(yc_val, 2.0)
                        - ((16.0 / 3.0) * std::pow(g3_val, 2.0))
                        - (3.0 * std::pow(g2_val, 2.0))
                        - ((7.0 / 15.0) * std::pow(g1_val, 2.0))))
                + (ys_val // Tr(6ad*Yd + 2ae*Ye)
                    * ((6.0 * ((ab_val * yb_val) + (as_val * ys_val)
                            + (ad_val * yd_val)))
                        + (2.0 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                + (ae_val * ye_val)))
                        + (4.0 * ys_val * as_val) + (2.0 * yc_val * ac_val)
                        + ((32.0 / 3.0) * std::pow(g3_val, 2.0) * M3_val)
                        + (6.0 * std::pow(g2_val, 2.0) * M2_val)
                        + ((14.0 / 15.0) * std::pow(g1_val, 2.0) * M1_val))));

    double dad_dt_1l = ((ad_val // Tr(3Yd^2 + Ye^2)
                    * (((3.0 * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                            + std::pow(yd_val, 2.0)))
                        + (std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                        + std::pow(ye_val, 2.0))) // end trace
                        + (5.0 * std::pow(yd_val, 2.0)) + std::pow(yu_val, 2.0)
                        - ((16.0 / 3.0) * std::pow(g3_val, 2.0))
                        - (3.0 * std::pow(g2_val, 2.0))
                        - ((7.0 / 15.0) * std::pow(g1_val, 2.0))))
                + (yd_val // Tr(6ad*Yd + 2ae*Ye)
                    * ((6.0 * ((ab_val * yb_val) + (as_val * ys_val)
                            + (ad_val * yd_val)))
                        + (2.0 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                + (ae_val * ye_val))) // end trace
                        + (4.0 * yd_val * ad_val) + (2.0 * yu_val * au_val)
                        + ((32.0 / 3.0) * std::pow(g3_val, 2.0) * M3_val)
                        + (6.0 * std::pow(g2_val, 2.0) * M2_val)
                        + ((14.0 / 15.0) * std::pow(g1_val, 2.0) * M1_val))));

    double datau_dt_1l = ((atau_val // Tr(3Yd^2 + Ye^2)
                    * (((3.0 * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                                + std::pow(yd_val, 2.0)))
                        + (std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                            + std::pow(ye_val, 2.0))) // end trace
                        + (5.0 * std::pow(ytau_val, 2.0))
                        - (3.0 * std::pow(g2_val, 2.0))
                        - ((9.0 / 5.0) * std::pow(g1_val, 2.0))))
                    + (ytau_val // Tr(6ad*Yd + 2ae*Ye)
                        * ((6.0 * ((ab_val * yb_val) + (as_val * ys_val)
                                + (ad_val * yd_val)))
                        + (2.0 * ((atau_val * ytau_val)
                                    + (amu_val * ymu_val)
                                    + (ae_val * ye_val))) // end trace
                        + (4.0 * ytau_val * atau_val)
                        + (6.0 * std::pow(g2_val, 2.0) * M2_val)
                        + ((18.0 / 5.0) * std::pow(g1_val, 2.0) * M1_val))));

    double damu_dt_1l = ((amu_val // Tr(3Yd^2 + Ye^2)
                    * (((3.0 * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                            + std::pow(yd_val, 2.0)))
                        + (std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                        + std::pow(ye_val, 2.0))) // end trace
                        + (5.0 * std::pow(ymu_val, 2.0))
                        - (3.0 * std::pow(g2_val, 2.0))
                        - ((9.0 / 5.0) * std::pow(g1_val, 2.0))))
                    + (ymu_val // Tr(6ad*Yd + 2ae*Ye)
                        * ((6.0 * ((ab_val * yb_val) + (as_val * ys_val)
                                + (ad_val * yd_val)))
                        + (2.0 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                + (ae_val * ye_val))) // end trace
                        + (4.0 * ymu_val * amu_val)
                        + (6.0 * std::pow(g2_val, 2.0) * M2_val)
                        + ((18.0 / 5.0) * std::pow(g1_val, 2.0) * M1_val))));

    double dae_dt_1l = ((ae_val // Tr(3Yd^2 + Ye^2)
                    * (((3.0 * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                            + std::pow(yd_val, 2.0)))
                        + (std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                        + std::pow(ye_val, 2.0))) // end trace
                        + (5.0 * std::pow(ye_val, 2.0))
                        - (3.0 * std::pow(g2_val, 2.0))
                        - ((9.0 / 5.0) * std::pow(g1_val, 2.0))))
                + (ye_val // Tr(6ad*Yd + 2ae*Ye)
                    * ((6.0 * ((ab_val * yb_val) + (as_val * ys_val)
                            + (ad_val * yd_val)))
                        + (2.0 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                + (ae_val * ye_val))) // end trace
                        + (4.0 * ye_val * ae_val)
                        + (6.0 * std::pow(g2_val, 2.0) * M2_val)
                        + ((18.0 / 5.0) * std::pow(g1_val, 2.0) * M1_val))));

    // 2-loop
    double dat_dt_2l = ((at_val // Tr(3Yu^4 + (Yu^2*Yd^2))
                    * (((-3.0) * ((3.0 * (std::pow(yt_val, 4.0)
                                    + std::pow(yc_val, 4.0)
                                    + std::pow(yu_val, 4.0)))
                                + ((std::pow(yt_val, 2.0) * std::pow(yb_val, 2.0))
                                    + (std::pow(yc_val, 2.0)
                                    * std::pow(ys_val, 2.0))
                                    + (std::pow(yu_val, 2.0)
                                    * std::pow(yd_val, 2.0))))) // end trace
                        - (std::pow(yb_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (std::pow(yb_val, 2.0)
                                    + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + (std::pow(ytau_val, 2.0)
                                + std::pow(ymu_val, 2.0)
                                + std::pow(ye_val, 2.0)))) // end trace
                        - (15.0 * std::pow(yt_val, 2.0) // Tr(Yu^2)
                        * (std::pow(yt_val, 2.0)
                            + std::pow(yc_val, 2.0)
                            + std::pow(yu_val, 2.0))) // end trace
                        - (6.0 * std::pow(yt_val, 4.0))
                        - (2.0 * std::pow(yb_val, 4.0))
                        - (4.0 * std::pow(yb_val, 2.0) * std::pow(yt_val, 2.0))
                        + (((16.0 * std::pow(g3_val, 2.0))
                        + ((4.0 / 5.0) * std::pow(g1_val, 2.0))) // Tr(Yu^2)
                        * (std::pow(yt_val, 2.0)
                            + std::pow(yc_val, 2.0)
                            + std::pow(yu_val, 2.0))) // end trace
                        + (12.0 * std::pow(g2_val, 2.0)
                        * std::pow(yt_val, 2.0))
                        + ((2.0 / 5.0) * std::pow(g1_val, 2.0)
                        * std::pow(yb_val, 2.0))
                        - ((16.0 / 9.0) * std::pow(g3_val, 4.0))
                        + (8.0 * std::pow(g3_val, 2.0)
                        * std::pow(g2_val, 2.0))
                        + ((136.0 / 45.0) * std::pow(g3_val, 2.0)
                        * std::pow(g1_val, 2.0))
                        + ((15.0 / 2.0) * std::pow(g2_val, 4.0))
                        + (std::pow(g2_val, 2.0) * std::pow(g1_val, 2.0))
                        + ((2743.0 / 450.0) * std::pow(g1_val, 4.0))))
                + (yt_val // Tr(6au*Yu^3 + au*Yd^2*Yu + ad*Yu^2*Yd)
                    * (((-6.0) * ((6.0 * ((at_val * std::pow(yt_val, 3.0))
                                    + (ac_val * std::pow(yc_val, 3.0))
                                    + (au_val * std::pow(yu_val, 3.0))))
                                + (at_val * std::pow(yb_val, 2.0) * yt_val)
                                + (ac_val * std::pow(ys_val, 2.0) * yc_val)
                                + (au_val * std::pow(yd_val, 2.0) * yu_val)
                                + (ab_val * std::pow(yt_val, 2.0) * yb_val)
                                + (as_val * std::pow(yc_val, 2.0) * ys_val)
                                + (ad_val * std::pow(yu_val, 2.0) * yd_val))) // end trace
                        - (18.0 * std::pow(yt_val, 2.0) // Tr(au*Yu)
                        * ((at_val * yt_val)
                            + (ac_val * yc_val)
                            + (au_val * yu_val))) // end trace
                        - (std::pow(yb_val, 2.0) // Tr(6ad*Yd + 2ae*Ye)
                        * ((6.0 * ((ab_val * yb_val) + (as_val * ys_val)
                                    + (ad_val * yd_val)))
                            + (2.0 * ((atau_val * ytau_val)
                                    + (amu_val * ymu_val)
                                    + (ae_val * ye_val))))) // end trace
                        - (12.0 * yt_val * at_val // Tr(Yu^2)
                        * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                            + std::pow(yu_val, 2.0))) // end trace
                        - (yb_val * ab_val // Tr(6Yd^2 + 2Ye^2)
                        * ((6.0 * (std::pow(yb_val, 2.0)
                                    + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + (2.0 * (std::pow(ytau_val, 2.0)
                                    + std::pow(ymu_val, 2.0)
                                    + std::pow(ye_val, 2.0))))) // end trace
                        - (14.0 * std::pow(yt_val, 3.0) * at_val)
                        - (8.0 * std::pow(yb_val, 3.0) * ab_val)
                        - (2.0 * std::pow(yb_val, 2.0) * yt_val * at_val)
                        - (4.0 * yb_val * ab_val * std::pow(yt_val, 2.0))
                        + (((32.0 * std::pow(g3_val, 2.0))
                            + ((8.0 / 5.0) * std::pow(g1_val, 2.0))) // Tr(au*Yu)
                        * ((at_val * yt_val) + (ac_val * yc_val)
                            + (au_val * yu_val))) // end trace
                        + (((6.0 * std::pow(g2_val, 2.0))
                            + ((6.0 / 5.0) * std::pow(g1_val, 2.0)))
                        * yt_val * at_val)
                        + ((4.0 / 5.0) * std::pow(g1_val, 2.0) * yb_val * ab_val)
                        - (((32.0 * std::pow(g3_val, 2.0) * M3_val)
                            + ((8.0 / 5.0) * std::pow(g1_val, 2.0) * M1_val)) // Tr(Yu^2)
                        * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                            + std::pow(yu_val, 2.0))) // end trace
                        - (((12.0 * std::pow(g2_val, 2.0) * M2_val)
                            + ((4.0 / 5.0) * std::pow(g1_val, 2.0) * M1_val))
                        * std::pow(yt_val, 2.0))
                        - ((4.0 / 5.0) * std::pow(g1_val, 2.0) * M1_val
                        * std::pow(yb_val, 2.0))
                        + ((64.0 / 9.0) * std::pow(g3_val, 4.0) * M3_val)
                        - (16.0 * std::pow(g3_val, 2.0) * std::pow(g2_val, 2.0)
                        * (M3_val + M2_val))
                        - ((272.0 / 45.0) * std::pow(g3_val, 2.0)
                        * std::pow(g1_val, 2.0)
                        * (M3_val + M1_val))
                        - (30.0 * std::pow(g2_val, 4.0) * M2_val)
                        - (2.0 * std::pow(g2_val, 2.0) * std::pow(g1_val, 2.0)
                        * (M2_val + M1_val))
                        - ((5486.0 / 225.0) * std::pow(g1_val, 4.0) * M1_val))));

    double dac_dt_2l = ((ac_val // Tr(3Yu^4 + (Yu^2*Yd^2))
                    * (((-3.0) * ((3.0 * (std::pow(yt_val, 4.0)
                                    + std::pow(yc_val, 4.0)
                                    + std::pow(yu_val, 4.0)))
                                + ((std::pow(yt_val, 2.0) * std::pow(yb_val, 2.0))
                                    + (std::pow(yc_val, 2.0)
                                    * std::pow(ys_val, 2.0))
                                    + (std::pow(yu_val, 2.0)
                                    * std::pow(yd_val, 2.0))))) // end trace
                        - (std::pow(ys_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (std::pow(yb_val, 2.0)
                                    + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + (std::pow(ytau_val, 2.0)
                                + std::pow(ymu_val, 2.0)
                                + std::pow(ye_val, 2.0)))) // end trace
                        - (15.0 * std::pow(yc_val, 2.0) // Tr(Yu^2)
                        * (std::pow(yt_val, 2.0)
                            + std::pow(yc_val, 2.0)
                            + std::pow(yu_val, 2.0))) // end trace
                        - (6.0 * std::pow(yc_val, 4.0))
                        - (2.0 * std::pow(ys_val, 4.0))
                        - (4.0 * std::pow(ys_val, 2.0) * std::pow(yc_val, 2.0))
                        + (((16.0 * std::pow(g3_val, 2.0))
                        + ((4.0 / 5.0) * std::pow(g1_val, 2.0))) // Tr(Yu^2)
                        * (std::pow(yt_val, 2.0)
                            + std::pow(yc_val, 2.0)
                            + std::pow(yu_val, 2.0))) // end trace
                        + (12.0 * std::pow(g2_val, 2.0)
                        * std::pow(yc_val, 2.0))
                        + ((2.0 / 5.0) * std::pow(g1_val, 2.0)
                        * std::pow(ys_val, 2.0))
                        - ((16.0 / 9.0) * std::pow(g3_val, 4.0))
                        + (8.0 * std::pow(g3_val, 2.0)
                        * std::pow(g2_val, 2.0))
                        + ((136.0 / 45.0) * std::pow(g3_val, 2.0)
                        * std::pow(g1_val, 2.0))
                        + ((15.0 / 2.0) * std::pow(g2_val, 4.0))
                        + (std::pow(g2_val, 2.0) * std::pow(g1_val, 2.0))
                        + ((2743.0 / 450.0) * std::pow(g1_val, 4.0))))
                + (yc_val // Tr(6au*Yu^3 + au*Yd^2*Yu + ad*Yu^2*Yd)
                    * (((-6.0) * ((6.0 * ((at_val * std::pow(yt_val, 3.0))
                                    + (ac_val * std::pow(yc_val, 3.0))
                                    + (au_val * std::pow(yu_val, 3.0))))
                                + (at_val * std::pow(yb_val, 2.0) * yt_val)
                                + (ac_val * std::pow(ys_val, 2.0) * yc_val)
                                + (au_val * std::pow(yd_val, 2.0) * yu_val)
                                + (ab_val * std::pow(yt_val, 2.0) * yb_val)
                                + (as_val * std::pow(yc_val, 2.0) * ys_val)
                                + (ad_val * std::pow(yu_val, 2.0) * yd_val))) // end trace
                        - (18.0 * std::pow(yc_val, 2.0) // Tr(au*Yu)
                        * ((at_val * yt_val)
                            + (ac_val * yc_val)
                            + (au_val * yu_val))) // end trace
                        - (std::pow(ys_val, 2.0) // Tr(6ad*Yd + 2ae*Ye)
                        * ((6.0 * ((ab_val * yb_val) + (as_val * ys_val)
                                    + (ad_val * yd_val)))
                            + (2.0 * ((atau_val * ytau_val)
                                    + (amu_val * ymu_val)
                                    + (ae_val * ye_val))))) // end trace
                        - (12.0 * yc_val * ac_val // Tr(Yu^2)
                        * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                            + std::pow(yu_val, 2.0))) // end trace
                        - (ys_val * as_val // Tr(6Yd^2 + 2Ye^2)
                        * ((6.0 * (std::pow(yb_val, 2.0)
                                    + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + (2.0 * (std::pow(ytau_val, 2.0)
                                    + std::pow(ymu_val, 2.0)
                                    + std::pow(ye_val, 2.0))))) // end trace
                        - (14.0 * std::pow(yc_val, 3.0) * ac_val)
                        - (8.0 * std::pow(ys_val, 3.0) * as_val)
                        - (2.0 * std::pow(ys_val, 2.0) * yc_val * ac_val)
                        - (4.0 * ys_val * as_val * std::pow(yc_val, 2.0))
                        + (((32.0 * std::pow(g3_val, 2.0))
                            + ((8.0 / 5.0) * std::pow(g1_val, 2.0))) // Tr(au*Yu)
                        * ((at_val * yt_val) + (ac_val * yc_val)
                            + (au_val * yu_val))) // end trace
                        + (((6.0 * std::pow(g2_val, 2.0))
                            + ((6.0 / 5.0) * std::pow(g1_val, 2.0)))
                        * yc_val * ac_val)
                        + ((4.0 / 5.0) * std::pow(g1_val, 2.0) * ys_val * as_val)
                        - (((32.0 * std::pow(g3_val, 2.0) * M3_val)
                            + ((8.0 / 5.0) * std::pow(g1_val, 2.0) * M1_val)) // Tr(Yu^2)
                        * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                            + std::pow(yu_val, 2.0))) // end trace
                        - (((12.0 * std::pow(g2_val, 2.0) * M2_val)
                            + ((4.0 / 5.0) * std::pow(g1_val, 2.0) * M1_val))
                        * std::pow(yc_val, 2.0))
                        - ((4.0 / 5.0) * std::pow(g1_val, 2.0) * M1_val
                        * std::pow(ys_val, 2.0))
                        + ((64.0 / 9.0) * std::pow(g3_val, 4.0) * M3_val)
                        - (16.0 * std::pow(g3_val, 2.0) * std::pow(g2_val, 2.0)
                        * (M3_val + M2_val))
                        - ((272.0 / 45.0) * std::pow(g3_val, 2.0)
                        * std::pow(g1_val, 2.0)
                        * (M3_val + M1_val))
                        - (30.0 * std::pow(g2_val, 4.0) * M2_val)
                        - (2.0 * std::pow(g2_val, 2.0) * std::pow(g1_val, 2.0)
                        * (M2_val + M1_val))
                        - ((5486.0 / 225.0) * std::pow(g1_val, 4.0) * M1_val))));

    double dau_dt_2l = ((au_val // Tr(3Yu^4 + (Yu^2*Yd^2))
                    * (((-3.0) * ((3.0 * (std::pow(yt_val, 4.0)
                                    + std::pow(yc_val, 4.0)
                                    + std::pow(yu_val, 4.0)))
                                + ((std::pow(yt_val, 2.0) * std::pow(yb_val, 2.0))
                                    + (std::pow(yc_val, 2.0)
                                    * std::pow(ys_val, 2.0))
                                    + (std::pow(yu_val, 2.0)
                                    * std::pow(yd_val, 2.0))))) // end trace
                        - (std::pow(yd_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (std::pow(yb_val, 2.0)
                                    + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + (std::pow(ytau_val, 2.0)
                                + std::pow(ymu_val, 2.0)
                                + std::pow(ye_val, 2.0)))) // end trace
                        - (15.0 * std::pow(yu_val, 2.0) // Tr(Yu^2)
                        * (std::pow(yt_val, 2.0)
                            + std::pow(yc_val, 2.0)
                            + std::pow(yu_val, 2.0))) // end trace
                        - (6.0 * std::pow(yu_val, 4.0))
                        - (2.0 * std::pow(yd_val, 4.0))
                        - (4.0 * std::pow(yd_val, 2.0) * std::pow(yu_val, 2.0))
                        + (((16.0 * std::pow(g3_val, 2.0))
                        + ((4.0 / 5.0) * std::pow(g1_val, 2.0))) // Tr(Yu^2)
                        * (std::pow(yt_val, 2.0)
                            + std::pow(yc_val, 2.0)
                            + std::pow(yu_val, 2.0))) // end trace
                        + (12.0 * std::pow(g2_val, 2.0)
                        * std::pow(yu_val, 2.0))
                        + ((2.0 / 5.0) * std::pow(g1_val, 2.0)
                        * std::pow(yd_val, 2.0))
                        - ((16.0 / 9.0) * std::pow(g3_val, 4.0))
                        + (8.0 * std::pow(g3_val, 2.0)
                        * std::pow(g2_val, 2.0))
                        + ((136.0 / 45.0) * std::pow(g3_val, 2.0)
                        * std::pow(g1_val, 2.0))
                        + ((15.0 / 2.0) * std::pow(g2_val, 4.0))
                        + (std::pow(g2_val, 2.0) * std::pow(g1_val, 2.0))
                        + ((2743.0 / 450.0) * std::pow(g1_val, 4.0))))
                + (yu_val // Tr(6au*Yu^3 + au*Yd^2*Yu + ad*Yu^2*Yd)
                    * (((-6.0) * ((6.0 * ((at_val * std::pow(yt_val, 3.0))
                                    + (ac_val * std::pow(yc_val, 3.0))
                                    + (au_val * std::pow(yu_val, 3.0))))
                                + (at_val * std::pow(yb_val, 2.0) * yt_val)
                                + (ac_val * std::pow(ys_val, 2.0) * yc_val)
                                + (au_val * std::pow(yd_val, 2.0) * yu_val)
                                + (ab_val * std::pow(yt_val, 2.0) * yb_val)
                                + (as_val * std::pow(yc_val, 2.0) * ys_val)
                                + (ad_val * std::pow(yu_val, 2.0) * yd_val))) // end trace
                        - (18.0 * std::pow(yu_val, 2.0) // Tr(au*Yu)
                        * ((at_val * yt_val)
                            + (ac_val * yc_val)
                            + (au_val * yu_val))) // end trace
                        - (std::pow(yd_val, 2.0) // Tr(6ad*Yd + 2ae*Ye)
                        * ((6.0 * ((ab_val * yb_val) + (as_val * ys_val)
                                    + (ad_val * yd_val)))
                            + (2.0 * ((atau_val * ytau_val)
                                    + (amu_val * ymu_val)
                                    + (ae_val * ye_val))))) // end trace
                        - (12.0 * yu_val * au_val // Tr(Yu^2)
                        * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                            + std::pow(yu_val, 2.0))) // end trace
                        - (yd_val * ad_val // Tr(6Yd^2 + 2Ye^2)
                        * ((6.0 * (std::pow(yb_val, 2.0)
                                    + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + (2.0 * (std::pow(ytau_val, 2.0)
                                    + std::pow(ymu_val, 2.0)
                                    + std::pow(ye_val, 2.0))))) // end trace
                        - (14.0 * std::pow(yu_val, 3.0) * au_val)
                        - (8.0 * std::pow(yd_val, 3.0) * ad_val)
                        - (2.0 * std::pow(yd_val, 2.0) * yu_val * au_val)
                        - (4.0 * yd_val * ad_val * std::pow(yu_val, 2.0))
                        + (((32.0 * std::pow(g3_val, 2.0))
                            + ((8.0 / 5.0) * std::pow(g1_val, 2.0))) // Tr(au*Yu)
                        * ((at_val * yt_val) + (ac_val * yc_val)
                            + (au_val * yu_val))) // end trace
                        + (((6.0 * std::pow(g2_val, 2.0))
                            + ((6.0 / 5.0) * std::pow(g1_val, 2.0)))
                        * yu_val * au_val)
                        + ((4.0 / 5.0) * std::pow(g1_val, 2.0) * yd_val * ad_val)
                        - (((32.0 * std::pow(g3_val, 2.0) * M3_val)
                            + ((8.0 / 5.0) * std::pow(g1_val, 2.0) * M1_val)) // Tr(Yu^2)
                        * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                            + std::pow(yu_val, 2.0))) // end trace
                        - (((12.0 * std::pow(g2_val, 2.0) * M2_val)
                            + ((4.0 / 5.0) * std::pow(g1_val, 2.0) * M1_val))
                        * std::pow(yu_val, 2.0))
                        - ((4.0 / 5.0) * std::pow(g1_val, 2.0) * M1_val
                        * std::pow(yd_val, 2.0))
                        + ((64.0 / 9.0) * std::pow(g3_val, 4.0) * M3_val)
                        - (16.0 * std::pow(g3_val, 2.0) * std::pow(g2_val, 2.0)
                        * (M3_val + M2_val))
                        - ((272.0 / 45.0) * std::pow(g3_val, 2.0)
                        * std::pow(g1_val, 2.0)
                        * (M3_val + M1_val))
                        - (30.0 * std::pow(g2_val, 4.0) * M2_val)
                        - (2.0 * std::pow(g2_val, 2.0) * std::pow(g1_val, 2.0)
                        * (M2_val + M1_val))
                        - ((5486.0 / 225.0) * std::pow(g1_val, 4.0) * M1_val))));

    double dab_dt_2l = ((ab_val // Tr(3Yd^4 + Yu^2*Yd^2 + Ye^4)
                    * (((-3.0) * ((3.0 * (std::pow(yb_val, 4.0)
                                    + std::pow(ys_val, 4.0)
                                    + std::pow(yd_val, 4.0)))
                                + ((std::pow(yt_val, 2.0) * std::pow(yb_val, 2.0))
                                    + (std::pow(yc_val, 2.0)
                                    * std::pow(ys_val, 2.0))
                                    + (std::pow(yu_val, 2.0)
                                    * std::pow(yd_val, 2.0)))
                                + std::pow(ytau_val, 4.0)
                                + std::pow(ymu_val, 4.0)
                                + std::pow(ye_val, 4.0))) // end trace
                        - (3.0 * std::pow(yt_val, 2.0) // Tr(Yu^2)
                        * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                            + std::pow(yu_val, 2.0))) // end trace
                        - (5 * std::pow(yb_val, 2.0) // Tr(3Yd^2+Ye^2)
                        * ((3.0 * (std::pow(yb_val, 2.0)
                                    + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + (std::pow(ytau_val, 2.0)
                                + std::pow(ymu_val, 2.0)
                                + std::pow(ye_val, 2.0)))) // end trace
                        - (6.0 * std::pow(yb_val, 4.0))
                        - (2.0 * std::pow(yt_val, 4.0))
                        - (4.0 * std::pow(yb_val, 2.0) * std::pow(yt_val, 2.0))
                        + (((16.0 * std::pow(g3_val, 2.0))
                        - ((2.0 / 5.0) * std::pow(g1_val, 2.0))) // Tr(Yd^2)
                        * (std::pow(yb_val, 2.0)
                            + std::pow(ys_val, 2.0)
                            + std::pow(yd_val, 2.0))) // end trace
                        + ((6.0 / 5.0) * std::pow(g1_val, 2.0) // Tr(Ye^2)
                        * (std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                            + std::pow(ye_val, 2.0))) // end trace
                        + ((4.0 / 5.0) * std::pow(g1_val, 2.0)
                        * std::pow(yt_val, 2.0))
                        + (((12.0 * std::pow(g2_val, 2.0))
                        + ((6.0 / 5.0) * std::pow(g1_val, 2.0)))
                        * std::pow(yb_val, 2.0))
                        - ((16.0 / 9.0) * std::pow(g3_val, 4.0))
                        + (8.0 * std::pow(g3_val, 2.0)
                        * std::pow(g2_val, 2.0))
                        + ((8.0 / 9.0) * std::pow(g3_val, 2.0)
                        * std::pow(g1_val, 2.0))
                        + ((15.0 / 2.0) * std::pow(g2_val, 4.0))
                        + (std::pow(g2_val, 2.0) * std::pow(g1_val, 2.0))
                        + ((287.0 / 90.0) * std::pow(g1_val, 4.0))))
                + (yb_val // Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
                    * (((-6.0) * ((6.0 * ((ab_val * std::pow(yb_val, 3.0))
                                    + (as_val * std::pow(ys_val, 3.0))
                                    + (ad_val * std::pow(yd_val, 3.0))))
                                + (at_val * std::pow(yb_val, 2.0) * yt_val)
                                + (ac_val * std::pow(ys_val, 2.0) * yc_val)
                                + (au_val * std::pow(yd_val, 2.0) * yu_val)
                                + (ab_val * std::pow(yt_val, 2.0) * yb_val)
                                + (as_val * std::pow(yc_val, 2.0) * ys_val)
                                + (ad_val * std::pow(yu_val, 2.0) * yd_val)
                                + (2.0 * ((atau_val * std::pow(ytau_val, 3.0))
                                        + (amu_val * std::pow(ymu_val, 3.0))
                                        + (ae_val * std::pow(ye_val, 3.0)))
                                    ))) // end trace
                        - (6.0 * std::pow(yt_val, 2.0) // Tr(au*Yu)
                        * ((at_val * yt_val)
                            + (ac_val * yc_val)
                            + (au_val * yu_val))) // end trace
                        - (6.0 * std::pow(yb_val, 2.0) // Tr(3ad*Yd + ae*Ye)
                        * ((3.0 * ((ab_val * yb_val) + (as_val * ys_val)
                                    + (ad_val * yd_val)))
                            + ((atau_val * ytau_val) + (amu_val * ymu_val)
                                + (ae_val * ye_val)))) // end trace
                        - (6.0 * yt_val * at_val // Tr(Yu^2)
                        * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                            + std::pow(yu_val, 2.0))) // end trace
                        - (4.0 * yb_val * ab_val // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (std::pow(yb_val, 2.0)
                                    + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + ((std::pow(ytau_val, 2.0)
                                    + std::pow(ymu_val, 2.0)
                                    + std::pow(ye_val, 2.0))))) // end trace
                        - (14.0 * std::pow(yb_val, 3.0) * ab_val)
                        - (8.0 * std::pow(yt_val, 3.0) * at_val)
                        - (4.0 * std::pow(yb_val, 2.0) * yt_val * at_val)
                        - (2.0 * yb_val * ab_val * std::pow(yt_val, 2.0))
                        + (((32.0 * std::pow(g3_val, 2.0))
                            - ((4.0 / 5.0) * std::pow(g1_val, 2.0))) // Tr(ad*Yd)
                        * ((ab_val * yb_val) + (as_val * ys_val)
                            + (ad_val * yd_val))) // end trace
                        + ((12.0 / 5.0) * std::pow(g1_val, 2.0) // Tr(ae*Ye)
                        * ((atau_val * ytau_val) + (amu_val * ymu_val)
                            + (ae_val * ye_val))) // end trace
                        + ((8.0 / 5.0) * std::pow(g1_val, 2.0) * yt_val * at_val)
                        + (((6.0 * std::pow(g2_val, 2.0))
                            + ((6.0 / 5.0) * std::pow(g1_val, 2.0)))
                        * yb_val * ab_val)
                        - (((32.0 * std::pow(g3_val, 2.0) * M3_val)
                            - ((4.0 / 5.0) * std::pow(g1_val, 2.0) * M1_val)) // Tr(Yd^2)
                        * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                            + std::pow(yd_val, 2.0))) // end trace
                        - ((12.0 / 5.0) * std::pow(g1_val, 2.0) * M1_val // Tr(Ye^2)
                        * (std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                            + std::pow(ye_val, 2.0))) // end trace
                        - (((12.0 * std::pow(g2_val, 2.0) * M2_val)
                            + ((8.0 / 5.0) * std::pow(g1_val, 2.0) * M1_val))
                        * std::pow(yb_val, 2.0))
                        - ((8.0 / 5.0) * std::pow(g1_val, 2.0) * M1_val
                        * std::pow(yt_val, 2.0))
                        + ((64.0 / 9.0) * std::pow(g3_val, 4.0) * M3_val)
                        - (16.0 * std::pow(g3_val, 2.0) * std::pow(g2_val, 2.0)
                        * (M3_val + M2_val))
                        - ((16.0 / 9.0) * std::pow(g3_val, 2.0)
                        * std::pow(g1_val, 2.0)
                        * (M3_val + M1_val))
                        - (30.0 * std::pow(g2_val, 4.0) * M2_val)
                        - (2.0 * std::pow(g2_val, 2.0) * std::pow(g1_val, 2.0)
                        * (M2_val + M1_val))
                        - ((574.0 / 45.0) * std::pow(g1_val, 4.0) * M1_val))));

    double das_dt_2l = ((as_val // Tr(3Yd^4 + Yu^2*Yd^2 + Ye^4)
                    * (((-3.0) * ((3.0 * (std::pow(yb_val, 4.0)
                                    + std::pow(ys_val, 4.0)
                                    + std::pow(yd_val, 4.0)))
                                + ((std::pow(yt_val, 2.0) * std::pow(yb_val, 2.0))
                                    + (std::pow(yc_val, 2.0)
                                    * std::pow(ys_val, 2.0))
                                    + (std::pow(yu_val, 2.0)
                                    * std::pow(yd_val, 2.0)))
                                + std::pow(ytau_val, 4.0)
                                + std::pow(ymu_val, 4.0)
                                + std::pow(ye_val, 4.0))) // end trace
                        - (3.0 * std::pow(yc_val, 2.0) // Tr(Yu^2)
                        * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                            + std::pow(yu_val, 2.0))) // end trace
                        - (5 * std::pow(ys_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (std::pow(yb_val, 2.0)
                                    + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + (std::pow(ytau_val, 2.0)
                                + std::pow(ymu_val, 2.0)
                                + std::pow(ye_val, 2.0)))) // end trace
                        - (6.0 * std::pow(ys_val, 4.0))
                        - (2.0 * std::pow(yc_val, 4.0))
                        - (4.0 * std::pow(ys_val, 2.0) * std::pow(yc_val, 2.0))
                        + (((16.0 * std::pow(g3_val, 2.0))
                        - ((2.0 / 5.0) * std::pow(g1_val, 2.0))) // Tr(Yd^2)
                        * (std::pow(yb_val, 2.0)
                            + std::pow(ys_val, 2.0)
                            + std::pow(yd_val, 2.0))) // end trace
                        + ((6.0 / 5.0) * std::pow(g1_val, 2.0) // Tr(Ye^2)
                        * (std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                            + std::pow(ye_val, 2.0))) // end trace
                        + ((4.0 / 5.0) * std::pow(g1_val, 2.0)
                        * std::pow(yc_val, 2.0))
                        + (((12.0 * std::pow(g2_val, 2.0))
                        + ((6.0 / 5.0) * std::pow(g1_val, 2.0)))
                        * std::pow(ys_val, 2.0))
                        - ((16.0 / 9.0) * std::pow(g3_val, 4.0))
                        + (8.0 * std::pow(g3_val, 2.0)
                        * std::pow(g2_val, 2.0))
                        + ((8.0 / 9.0) * std::pow(g3_val, 2.0)
                        * std::pow(g1_val, 2.0))
                        + ((15.0 / 2.0) * std::pow(g2_val, 4.0))
                        + (std::pow(g2_val, 2.0) * std::pow(g1_val, 2.0))
                        + ((287.0 / 90.0) * std::pow(g1_val, 4.0))))
                + (ys_val // Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
                    * (((-6.0) * ((6.0 * ((ab_val * std::pow(yb_val, 3.0))
                                    + (as_val * std::pow(ys_val, 3.0))
                                    + (ad_val * std::pow(yd_val, 3.0))))
                                + (at_val * std::pow(yb_val, 2.0) * yt_val)
                                + (ac_val * std::pow(ys_val, 2.0) * yc_val)
                                + (au_val * std::pow(yd_val, 2.0) * yu_val)
                                + (ab_val * std::pow(yt_val, 2.0) * yb_val)
                                + (as_val * std::pow(yc_val, 2.0) * ys_val)
                                + (ad_val * std::pow(yu_val, 2.0) * yd_val)
                                + (2.0 * ((atau_val * std::pow(ytau_val, 3.0))
                                        + (amu_val * std::pow(ymu_val, 3.0))
                                        + (ae_val * std::pow(ye_val, 3.0)))
                                    ))) // end trace
                        - (6.0 * std::pow(yc_val, 2.0) // Tr(au*Yu)
                        * ((at_val * yt_val)
                            + (ac_val * yc_val)
                            + (au_val * yu_val))) // end trace
                        - (6.0 * std::pow(ys_val, 2.0) // Tr(3ad*Yd + ae*Ye)
                        * ((3.0 * ((ab_val * yb_val) + (as_val * ys_val)
                                    + (ad_val * yd_val)))
                            + ((atau_val * ytau_val) + (amu_val * ymu_val)
                                + (ae_val * ye_val)))) // end trace
                        - (6.0 * yc_val * ac_val // Tr(Yu^2)
                        * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                            + std::pow(yu_val, 2.0))) // end trace
                        - (4.0 * ys_val * as_val // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (std::pow(yb_val, 2.0)
                                    + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + ((std::pow(ytau_val, 2.0)
                                    + std::pow(ymu_val, 2.0)
                                    + std::pow(ye_val, 2.0))))) // end trace
                        - (14.0 * std::pow(ys_val, 3.0) * as_val)
                        - (8.0 * std::pow(yc_val, 3.0) * ac_val)
                        - (4.0 * std::pow(ys_val, 2.0) * yc_val * ac_val)
                        - (2.0 * ys_val * as_val * std::pow(yc_val, 2.0))
                        + (((32.0 * std::pow(g3_val, 2.0))
                            - ((4.0 / 5.0) * std::pow(g1_val, 2.0))) // Tr(ad*Yd)
                        * ((ab_val * yb_val) + (as_val * ys_val)
                            + (ad_val * yd_val))) // end trace
                        + ((12.0 / 5.0) * std::pow(g1_val, 2.0) // Tr(ae*Ye)
                        * ((atau_val * ytau_val) + (amu_val * ymu_val)
                            + (ae_val * ye_val))) // end trace
                        + ((8.0 / 5.0) * std::pow(g1_val, 2.0) * yc_val * ac_val)
                        + (((6.0 * std::pow(g2_val, 2.0))
                            + ((6.0 / 5.0) * std::pow(g1_val, 2.0)))
                        * ys_val * as_val)
                        - (((32.0 * std::pow(g3_val, 2.0) * M3_val)
                            - ((4.0 / 5.0) * std::pow(g1_val, 2.0) * M1_val)) // Tr(Yd^2)
                        * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                            + std::pow(yd_val, 2.0))) // end trace
                        - ((12.0 / 5.0) * std::pow(g1_val, 2.0) * M1_val // Tr(Ye^2)
                        * (std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                            + std::pow(ye_val, 2.0))) // end trace
                        - (((12.0 * std::pow(g2_val, 2.0) * M2_val)
                            + ((8.0 / 5.0) * std::pow(g1_val, 2.0) * M1_val))
                        * std::pow(ys_val, 2.0))
                        - ((8.0 / 5.0) * std::pow(g1_val, 2.0) * M1_val
                        * std::pow(yc_val, 2.0))
                        + ((64.0 / 9.0) * std::pow(g3_val, 4.0) * M3_val)
                        - (16.0 * std::pow(g3_val, 2.0) * std::pow(g2_val, 2.0)
                        * (M3_val + M2_val))
                        - ((16.0 / 9.0) * std::pow(g3_val, 2.0)
                        * std::pow(g1_val, 2.0)
                        * (M3_val + M1_val))
                        - (30.0 * std::pow(g2_val, 4.0) * M2_val)
                        - (2.0 * std::pow(g2_val, 2.0) * std::pow(g1_val, 2.0)
                        * (M2_val + M1_val))
                        - ((574.0 / 45.0) * std::pow(g1_val, 4.0) * M1_val))));

    double dad_dt_2l = ((ad_val // Tr(3Yd^4 + Yu^2*Yd^2 + Ye^4)
                    * (((-3.0) * ((3.0 * (std::pow(yb_val, 4.0)
                                    + std::pow(ys_val, 4.0)
                                    + std::pow(yd_val, 4.0)))
                                + ((std::pow(yt_val, 2.0) * std::pow(yb_val, 2.0))
                                    + (std::pow(yc_val, 2.0)
                                    * std::pow(ys_val, 2.0))
                                    + (std::pow(yu_val, 2.0)
                                    * std::pow(yd_val, 2.0)))
                                + std::pow(ytau_val, 4.0)
                                + std::pow(ymu_val, 4.0)
                                + std::pow(ye_val, 4.0))) // end trace
                        - (3.0 * std::pow(yu_val, 2.0) // Tr(Yu^2)
                        * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                            + std::pow(yu_val, 2.0))) // end trace
                        - (5 * std::pow(yd_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (std::pow(yb_val, 2.0)
                                    + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + (std::pow(ytau_val, 2.0)
                                + std::pow(ymu_val, 2.0)
                                + std::pow(ye_val, 2.0)))) // end trace
                        - (6.0 * std::pow(yd_val, 4.0))
                        - (2.0 * std::pow(yu_val, 4.0))
                        - (4.0 * std::pow(yd_val, 2.0) * std::pow(yu_val, 2.0))
                        + (((16.0 * std::pow(g3_val, 2.0))
                        - ((2.0 / 5.0) * std::pow(g1_val, 2.0))) // Tr(Yd^2)
                        * (std::pow(yb_val, 2.0)
                            + std::pow(ys_val, 2.0)
                            + std::pow(yd_val, 2.0))) // end trace
                        + ((6.0 / 5.0) * std::pow(g1_val, 2.0) // Tr(Ye^2)
                        * (std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                            + std::pow(ye_val, 2.0))) // end trace
                        + ((4.0 / 5.0) * std::pow(g1_val, 2.0)
                        * std::pow(yu_val, 2.0))
                        + (((12.0 * std::pow(g2_val, 2.0))
                        + ((6.0 / 5.0) * std::pow(g1_val, 2.0)))
                        * std::pow(yd_val, 2.0))
                        - ((16.0 / 9.0) * std::pow(g3_val, 4.0))
                        + (8.0 * std::pow(g3_val, 2.0)
                        * std::pow(g2_val, 2.0))
                        + ((8.0 / 9.0) * std::pow(g3_val, 2.0)
                        * std::pow(g1_val, 2.0))
                        + ((15.0 / 2.0) * std::pow(g2_val, 4.0))
                        + (std::pow(g2_val, 2.0) * std::pow(g1_val, 2.0))
                        + ((287.0 / 90.0) * std::pow(g1_val, 4.0))))
                + (yd_val // Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
                    * (((-6.0) * ((6.0 * ((ab_val * std::pow(yb_val, 3.0))
                                    + (as_val * std::pow(ys_val, 3.0))
                                    + (ad_val * std::pow(yd_val, 3.0))))
                                + (at_val * std::pow(yb_val, 2.0) * yt_val)
                                + (ac_val * std::pow(ys_val, 2.0) * yc_val)
                                + (au_val * std::pow(yd_val, 2.0) * yu_val)
                                + (ab_val * std::pow(yt_val, 2.0) * yb_val)
                                + (as_val * std::pow(yc_val, 2.0) * ys_val)
                                + (ad_val * std::pow(yu_val, 2.0) * yd_val)
                                + (2.0 * ((atau_val * std::pow(ytau_val, 3.0))
                                        + (amu_val * std::pow(ymu_val, 3.0))
                                        + (ae_val * std::pow(ye_val, 3.0)))
                                    ))) // end trace
                        - (6.0 * std::pow(yu_val, 2.0) // Tr(au*Yu)
                        * ((at_val * yt_val)
                            + (ac_val * yc_val)
                            + (au_val * yu_val))) // end trace
                        - (6.0 * std::pow(yd_val, 2.0) // Tr(3ad*Yd + ae*Ye)
                        * ((3.0 * ((ab_val * yb_val) + (as_val * ys_val)
                                    + (ad_val * yd_val)))
                            + ((atau_val * ytau_val) + (amu_val * ymu_val)
                                + (ae_val * ye_val)))) // end trace
                        - (6.0 * yu_val * au_val // Tr(Yu^2)
                        * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                            + std::pow(yu_val, 2.0))) // end trace
                        - (4.0 * yd_val * ad_val // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (std::pow(yb_val, 2.0)
                                    + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + ((std::pow(ytau_val, 2.0)
                                    + std::pow(ymu_val, 2.0)
                                    + std::pow(ye_val, 2.0))))) // end trace
                        - (14.0 * std::pow(yd_val, 3.0) * ad_val)
                        - (8.0 * std::pow(yu_val, 3.0) * au_val)
                        - (4.0 * std::pow(yd_val, 2.0) * yu_val * au_val)
                        - (2.0 * yd_val * ad_val * std::pow(yu_val, 2.0))
                        + (((32.0 * std::pow(g3_val, 2.0))
                            - ((4.0 / 5.0) * std::pow(g1_val, 2.0))) // Tr(ad*Yd)
                        * ((ab_val * yb_val) + (as_val * ys_val)
                            + (ad_val * yd_val))) // end trace
                        + ((12.0 / 5.0) * std::pow(g1_val, 2.0) // Tr(ae*Ye)
                        * ((atau_val * ytau_val) + (amu_val * ymu_val)
                            + (ae_val * ye_val))) // end trace
                        + ((8.0 / 5.0) * std::pow(g1_val, 2.0) * yu_val * au_val)
                        + (((6.0 * std::pow(g2_val, 2.0))
                            + ((6.0 / 5.0) * std::pow(g1_val, 2.0)))
                        * yd_val * ad_val)
                        - (((32.0 * std::pow(g3_val, 2.0) * M3_val)
                            - ((4.0 / 5.0) * std::pow(g1_val, 2.0) * M1_val)) // Tr(Yd^2)
                        * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                            + std::pow(yd_val, 2.0))) // end trace
                        - ((12.0 / 5.0) * std::pow(g1_val, 2.0) * M1_val // Tr(Ye^2)
                        * (std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                            + std::pow(ye_val, 2.0))) // end trace
                        - (((12.0 * std::pow(g2_val, 2.0) * M2_val)
                            + ((8.0 / 5.0) * std::pow(g1_val, 2.0) * M1_val))
                        * std::pow(yd_val, 2.0))
                        - ((8.0 / 5.0) * std::pow(g1_val, 2.0) * M1_val
                        * std::pow(yu_val, 2.0))
                        + ((64.0 / 9.0) * std::pow(g3_val, 4.0) * M3_val)
                        - (16.0 * std::pow(g3_val, 2.0) * std::pow(g2_val, 2.0)
                        * (M3_val + M2_val))
                        - ((16.0 / 9.0) * std::pow(g3_val, 2.0)
                        * std::pow(g1_val, 2.0)
                        * (M3_val + M1_val))
                        - (30.0 * std::pow(g2_val, 4.0) * M2_val)
                        - (2.0 * std::pow(g2_val, 2.0) * std::pow(g1_val, 2.0)
                        * (M2_val + M1_val))
                        - ((574.0 / 45.0) * std::pow(g1_val, 4.0) * M1_val))));

    double datau_dt_2l = ((atau_val // Tr(3Yd^4 + Yu^2*Yd^2 + Ye^4)
                    * (((-3.0) * ((3.0 * (std::pow(yb_val, 4.0)
                                    + std::pow(ys_val, 4.0)
                                    + std::pow(yd_val, 4.0)))
                                + ((std::pow(yt_val, 2.0)
                                    * std::pow(yb_val, 2.0))
                                    + (std::pow(yc_val, 2.0)
                                    * std::pow(ys_val, 2.0))
                                    + (std::pow(yu_val, 2.0)
                                    * std::pow(yd_val, 2.0)))
                                + std::pow(ytau_val, 4.0)
                                + std::pow(ymu_val, 4.0)
                                + std::pow(ye_val, 4.0))) // end trace
                        - (5 * std::pow(ytau_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (std::pow(yb_val, 2.0)
                                    + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + (std::pow(ytau_val, 2.0)
                                + std::pow(ymu_val, 2.0)
                                + std::pow(ye_val, 2.0)))) // end trace
                        - (6.0 * std::pow(ytau_val, 4.0))
                        + (((16.0 * std::pow(g3_val, 2.0))
                            - ((2.0 / 5.0) * std::pow(g1_val, 2.0))) // Tr(Yd^2)
                        * (std::pow(yb_val, 2.0)
                            + std::pow(ys_val, 2.0)
                            + std::pow(yd_val, 2.0))) // end trace
                        + ((6.0 / 5.0) * std::pow(g1_val, 2.0) // Tr(Ye^2)
                        * (std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                            + std::pow(ye_val, 2.0))) // end trace
                        + (((12.0 * std::pow(g2_val, 2.0))
                            - ((6.0 / 5.0) * std::pow(g1_val, 2.0)))
                        * std::pow(ytau_val, 2.0))
                        + ((15.0 / 2.0) * std::pow(g2_val, 4.0))
                        + ((9.0 / 5.0) * std::pow(g2_val, 2.0)
                        * std::pow(g1_val, 2.0))
                        + ((27.0 / 2.0) * std::pow(g1_val, 4.0))))
                    + (ytau_val // Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
                        * (((-6.0) * ((6.0 * ((ab_val * std::pow(yb_val, 3.0))
                                        + (as_val * std::pow(ys_val, 3.0))
                                        + (ad_val * std::pow(yd_val, 3.0))))
                                    + (at_val * std::pow(yb_val, 2.0) * yt_val)
                                    + (ac_val * std::pow(ys_val, 2.0) * yc_val)
                                    + (au_val * std::pow(yd_val, 2.0) * yu_val)
                                    + (ab_val * std::pow(yt_val, 2.0) * yb_val)
                                    + (as_val * std::pow(yc_val, 2.0) * ys_val)
                                    + (ad_val * std::pow(yu_val, 2.0) * yd_val)
                                    + (2.0 * ((atau_val
                                            * std::pow(ytau_val, 3.0))
                                            + (amu_val
                                                * std::pow(ymu_val, 3.0))
                                            + (ae_val
                                                * std::pow(ye_val, 3.0)))
                                    ))) // end trace
                        - (4.0 * ytau_val * atau_val // Tr(3Yd^2 + Ye^2)
                            * ((3.0 * (std::pow(yb_val, 2.0)
                                    + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                                + ((std::pow(ytau_val, 2.0)
                                    + std::pow(ymu_val, 2.0)
                                    + std::pow(ye_val, 2.0))))) // end trace
                        - (6.0 * std::pow(ytau_val, 2.0) // Tr(3ad*Yd + ae*Ye)
                            * ((3.0 * ((ab_val * yb_val) + (as_val * ys_val)
                                    + (ad_val * yd_val)))
                                + (atau_val * ytau_val)
                                + (amu_val * ymu_val)
                                + (ae_val * ye_val))) // end trace
                        - (14.0 * std::pow(ytau_val, 3.0) * atau_val)
                        + (((32.0 * std::pow(g3_val, 2.0))
                            - ((4.0 / 5.0) * std::pow(g1_val, 2.0))) // Tr(ad*Yd)
                            * ((ab_val * yb_val) + (as_val * ys_val)
                                + (ad_val * yd_val))) // end trace
                        + ((12.0 / 5.0) * std::pow(g1_val, 2.0) // Tr(ae*Ye)
                            * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                + (ae_val * ye_val))) // end trace
                        + (((6.0 * std::pow(g2_val, 2.0))
                            + ((6.0 / 5.0) * std::pow(g1_val, 2.0)))
                            * ytau_val * atau_val)
                        - (((32.0 * std::pow(g3_val, 2.0) * M3_val)
                            - ((4.0 / 5.0) * std::pow(g1_val, 2.0) * M1_val)) // Tr(Yd^2)
                            * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                                + std::pow(yd_val, 2.0))) // end trace
                        - ((12.0 / 5.0) * std::pow(g1_val, 2.0) * M1_val // Tr(Ye^2)
                            * (std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                                + std::pow(ye_val, 2.0))) // end trace
                        - (12.0 * std::pow(g2_val, 2.0) * M2_val
                            * std::pow(ytau_val, 2.0))
                        - (30.0 * std::pow(g2_val, 4.0) * M2_val)
                        - ((18.0 / 5.0) * std::pow(g2_val, 2.0)
                            * std::pow(g1_val, 2.0)
                            * (M1_val + M2_val))
                        - (54.0 * std::pow(g1_val, 4.0) * M1_val))));

    double damu_dt_2l = ((amu_val // Tr(3Yd^4 + Yu^2*Yd^2 + Ye^4)
                    * (((-3.0) * ((3.0 * (std::pow(yb_val, 4.0)
                                    + std::pow(ys_val, 4.0)
                                    + std::pow(yd_val, 4.0)))
                                + ((std::pow(yt_val, 2.0)
                                    * std::pow(yb_val, 2.0))
                                    + (std::pow(yc_val, 2.0)
                                    * std::pow(ys_val, 2.0))
                                    + (std::pow(yu_val, 2.0)
                                    * std::pow(yd_val, 2.0)))
                                + std::pow(ytau_val, 4.0)
                                + std::pow(ymu_val, 4.0)
                                + std::pow(ye_val, 4.0))) // end trace
                        - (5 * std::pow(ymu_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (std::pow(yb_val, 2.0)
                                    + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + (std::pow(ytau_val, 2.0)
                                + std::pow(ymu_val, 2.0)
                                + std::pow(ye_val, 2.0)))) // end trace
                        - (6.0 * std::pow(ymu_val, 4.0))
                        + (((16.0 * std::pow(g3_val, 2.0))
                        - ((2.0 / 5.0) * std::pow(g1_val, 2.0))) // Tr(Yd^2)
                        * (std::pow(yb_val, 2.0)
                            + std::pow(ys_val, 2.0)
                            + std::pow(yd_val, 2.0))) // end trace
                        + ((6.0 / 5.0) * std::pow(g1_val, 2.0) // Tr(Ye^2)
                        * (std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                            + std::pow(ye_val, 2.0))) // end trace
                        + (((12.0 * std::pow(g2_val, 2.0))
                        - ((6.0 / 5.0) * std::pow(g1_val, 2.0)))
                        * std::pow(ymu_val, 2.0))
                        + ((15.0 / 2.0) * std::pow(g2_val, 4.0))
                        + ((9.0 / 5.0) * std::pow(g2_val, 2.0)
                        * std::pow(g1_val, 2.0))
                        + ((27.0 / 2.0) * std::pow(g1_val, 4.0))))
                    + (ymu_val // Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
                        * (((-6.0) * ((6.0 * ((ab_val * std::pow(yb_val, 3.0))
                                        + (as_val * std::pow(ys_val, 3.0))
                                        + (ad_val * std::pow(yd_val, 3.0))))
                                    + (at_val * std::pow(yb_val, 2.0) * yt_val)
                                    + (ac_val * std::pow(ys_val, 2.0) * yc_val)
                                    + (au_val * std::pow(yd_val, 2.0) * yu_val)
                                    + (ab_val * std::pow(yt_val, 2.0) * yb_val)
                                    + (as_val * std::pow(yc_val, 2.0) * ys_val)
                                    + (ad_val * std::pow(yu_val, 2.0) * yd_val)
                                    + (2.0 * ((atau_val * std::pow(ytau_val, 3.0))
                                        + (amu_val * std::pow(ymu_val, 3.0))
                                        + (ae_val * std::pow(ye_val, 3.0)))))) // end trace
                        - (4.0 * ymu_val * amu_val // Tr(3Yd^2 + Ye^2)
                            * ((3.0 * (std::pow(yb_val, 2.0)
                                    + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                                + ((std::pow(ytau_val, 2.0)
                                    + std::pow(ymu_val, 2.0)
                                    + std::pow(ye_val, 2.0))))) // end trace
                        - (6.0 * std::pow(ymu_val, 2.0) // Tr(3ad*Yd + ae*Ye)
                            * ((3.0 * ((ab_val * yb_val) + (as_val * ys_val)
                                    + (ad_val * yd_val)))
                                + (atau_val * ytau_val) + (amu_val * ymu_val)
                                + (ae_val * ye_val))) // end trace
                        - (14.0 * std::pow(ymu_val, 3.0) * amu_val)
                        + (((32.0 * std::pow(g3_val, 2.0))
                            - ((4.0 / 5.0) * std::pow(g1_val, 2.0))) // Tr(ad*Yd)
                            * ((ab_val * yb_val) + (as_val * ys_val)
                                + (ad_val * yd_val))) // end trace
                        + ((12.0 / 5.0) * std::pow(g1_val, 2.0) // Tr(ae*Ye)
                            * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                + (ae_val * ye_val))) // end trace
                        + (((6.0 * std::pow(g2_val, 2.0))
                            + ((6.0 / 5.0) * std::pow(g1_val, 2.0)))
                            * ymu_val * amu_val)
                        - (((32.0 * std::pow(g3_val, 2.0) * M3_val)
                            - ((4.0 / 5.0) * std::pow(g1_val, 2.0) * M1_val)) // Tr(Yd^2)
                            * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                                + std::pow(yd_val, 2.0))) // end trace
                        - ((12.0 / 5.0) * std::pow(g1_val, 2.0) * M1_val // Tr(Ye^2)
                            * (std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                                + std::pow(ye_val, 2.0))) // end trace
                        - (12.0 * std::pow(g2_val, 2.0) * M2_val
                            * std::pow(ymu_val, 2.0))
                        - (30.0 * std::pow(g2_val, 4.0) * M2_val)
                        - ((18.0 / 5.0) * std::pow(g2_val, 2.0) * std::pow(g1_val, 2.0)
                            * (M1_val + M2_val))
                        - (54.0 * std::pow(g1_val, 4.0) * M1_val))));

    double dae_dt_2l = ((ae_val // Tr(3Yd^4 + Yu^2*Yd^2 + Ye^4)
                    * (((-3.0) * ((3.0 * (std::pow(yb_val, 4.0)
                                    + std::pow(ys_val, 4.0)
                                    + std::pow(yd_val, 4.0)))
                                + ((std::pow(yt_val, 2.0)
                                    * std::pow(yb_val, 2.0))
                                    + (std::pow(yc_val, 2.0)
                                    * std::pow(ys_val, 2.0))
                                    + (std::pow(yu_val, 2.0)
                                    * std::pow(yd_val, 2.0)))
                                + std::pow(ytau_val, 4.0)
                                + std::pow(ymu_val, 4.0)
                                + std::pow(ye_val, 4.0))) // end trace
                        - (5 * std::pow(ye_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (std::pow(yb_val, 2.0)
                                    + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + (std::pow(ytau_val, 2.0)
                                + std::pow(ymu_val, 2.0)
                                + std::pow(ye_val, 2.0)))) // end trace
                        - (6.0 * std::pow(ye_val, 4.0))
                        + (((16.0 * std::pow(g3_val, 2.0))
                        - ((2.0 / 5.0) * std::pow(g1_val, 2.0))) // Tr(Yd^2)
                        * (std::pow(yb_val, 2.0)
                            + std::pow(ys_val, 2.0)
                            + std::pow(yd_val, 2.0))) // end trace
                        + ((6.0 / 5.0) * std::pow(g1_val, 2.0) // Tr(Ye^2)
                        * (std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                            + std::pow(ye_val, 2.0))) // end trace
                        + (((12.0 * std::pow(g2_val, 2.0))
                        - ((6.0 / 5.0) * std::pow(g1_val, 2.0)))
                        * std::pow(ye_val, 2.0))
                        + ((15.0 / 2.0) * std::pow(g2_val, 4.0))
                        + ((9.0 / 5.0) * std::pow(g2_val, 2.0)
                        * std::pow(g1_val, 2.0))
                        + ((27.0 / 2.0) * std::pow(g1_val, 4.0))))
                + (ye_val // Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
                    * (((-6.0) * ((6.0 * ((ab_val * std::pow(yb_val, 3.0))
                                    + (as_val * std::pow(ys_val, 3.0))
                                    + (ad_val * std::pow(yd_val, 3.0))))
                                + (at_val * std::pow(yb_val, 2.0) * yt_val)
                                + (ac_val * std::pow(ys_val, 2.0) * yc_val)
                                + (au_val * std::pow(yd_val, 2.0) * yu_val)
                                + (ab_val * std::pow(yt_val, 2.0) * yb_val)
                                + (as_val * std::pow(yc_val, 2.0) * ys_val)
                                + (ad_val * std::pow(yu_val, 2.0) * yd_val)
                                + (2.0 * ((atau_val * std::pow(ytau_val, 3.0))
                                        + (amu_val * std::pow(ymu_val, 3.0))
                                        + (ae_val * std::pow(ye_val, 3.0)))))) // end trace
                        - (4.0 * ye_val * ae_val // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (std::pow(yb_val, 2.0)
                                    + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + ((std::pow(ytau_val, 2.0)
                                    + std::pow(ymu_val, 2.0)
                                    + std::pow(ye_val, 2.0))))) // end trace
                        - (6.0 * std::pow(ye_val, 2.0) // Tr(3ad*Yd + ae*Ye)
                        * ((3.0 * ((ab_val * yb_val) + (as_val * ys_val)
                                    + (ad_val * yd_val)))
                            + (atau_val * ytau_val) + (amu_val * ymu_val)
                            + (ae_val * ye_val))) // end trace
                        - (14.0 * std::pow(ye_val, 3.0) * ae_val)
                        + (((32.0 * std::pow(g3_val, 2.0))
                            - ((4.0 / 5.0) * std::pow(g1_val, 2.0))) // Tr(ad*Yd)
                        * ((ab_val * yb_val) + (as_val * ys_val)
                            + (ad_val * yd_val))) // end trace
                        + ((12.0 / 5.0) * std::pow(g1_val, 2.0) // Tr(ae*Ye)
                        * ((atau_val * ytau_val) + (amu_val * ymu_val)
                            + (ae_val * ye_val))) // end trace
                        + (((6.0 * std::pow(g2_val, 2.0))
                            + ((6.0 / 5.0) * std::pow(g1_val, 2.0)))
                        * ye_val * ae_val)
                        - (((32.0 * std::pow(g3_val, 2.0) * M3_val)
                            - ((4.0 / 5.0) * std::pow(g1_val, 2.0) * M1_val)) // Tr(Yd^2)
                        * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                            + std::pow(yd_val, 2.0))) // end trace
                        - ((12.0 / 5.0) * std::pow(g1_val, 2.0) * M1_val // Tr(Ye^2)
                        * (std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                            + std::pow(ye_val, 2.0))) // end trace
                        - (12.0 * std::pow(g2_val, 2.0) * M2_val
                        * std::pow(ye_val, 2.0))
                        - (30.0 * std::pow(g2_val, 4.0) * M2_val)
                        - ((18.0 / 5.0) * std::pow(g2_val, 2.0) * std::pow(g1_val, 2.0)
                        * (M1_val + M2_val))
                        - (54.0 * std::pow(g1_val, 4.0) * M1_val))));

    // Total soft trilinear coupling beta functions
    double dat_dt = ((loop_fac * dat_dt_1l) + (loop_fac_sq * dat_dt_2l));

    double dac_dt = ((loop_fac * dac_dt_1l) + (loop_fac_sq * dac_dt_2l));

    double dau_dt = ((loop_fac * dau_dt_1l) + (loop_fac_sq * dau_dt_2l));

    double dab_dt = ((loop_fac * dab_dt_1l) + (loop_fac_sq * dab_dt_2l));

    double das_dt = ((loop_fac * das_dt_1l) + (loop_fac_sq * das_dt_2l));

    double dad_dt = ((loop_fac * dad_dt_1l) + (loop_fac_sq * dad_dt_2l));

    double datau_dt = ((loop_fac * datau_dt_1l) + (loop_fac_sq * datau_dt_2l));

    double damu_dt = ((loop_fac * damu_dt_1l) + (loop_fac_sq * damu_dt_2l));

    double dae_dt = ((loop_fac * dae_dt_1l) + (loop_fac_sq * dae_dt_2l));

    // Soft bilinear coupling b=B*mu
    /////////////////////////////////////////////////////////////////////////////////
    // 1-loop
    double db_dt_1l = ((b_val // Tr(3Yu^2 + 3Yd^2 + Ye^2)
                * (((3.0 * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                            + std::pow(yu_val, 2.0) + std::pow(yb_val, 2.0)
                            + std::pow(ys_val, 2.0) + std::pow(yd_val, 2.0)))
                        + std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                        + std::pow(ye_val, 2.0)) // end trace
                    - (3.0 * std::pow(g2_val, 2.0))
                    - ((3.0 / 5.0) * std::pow(g1_val, 2.0))))
                + (mu_val // Tr(6au*Yu + 6ad*Yd + 2ae*Ye)
                    * (((6.0 * ((at_val * yt_val) + (ac_val * yc_val)
                            + (au_val * yu_val) + (ab_val * yb_val)
                            + (as_val * ys_val) + (ad_val * yd_val)))
                        + (2.0 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                + (ae_val * ye_val))))
                        + (6.0 * std::pow(g2_val, 2.0) * M2_val)
                        + ((6.0 / 5.0) * std::pow(g1_val, 2.0) * M1_val))));

    // 2-loop
    double db_dt_2l = ((b_val // Tr(3Yu^4 + 3Yd^4 + 2Yu^2*Yd^2 + Ye^4)
                * (((-3.0) * ((3.0 *  (std::pow(yt_val, 4.0) + std::pow(yc_val, 4.0)
                                    + std::pow(yu_val, 4.0)
                                    + std::pow(yb_val, 4.0)
                                    + std::pow(ys_val, 4.0)
                                    + std::pow(yd_val, 4.0)))
                            + (2.0 * ((std::pow(yt_val, 2.0)
                                    * std::pow(yb_val, 2.0))
                                    + (std::pow(yc_val, 2.0)
                                        * std::pow(ys_val, 2.0))
                                    + (std::pow(yu_val, 2.0)
                                        * std::pow(yd_val, 2.0))))
                            + std::pow(ytau_val, 4.0) + std::pow(ymu_val, 4.0)
                            + std::pow(ye_val, 4.0))) // end trace
                    + (((16.0 * std::pow(g3_val, 2.0))
                        + ((4.0 / 5.0) * std::pow(g1_val, 2.0))) // Tr(Yu^2)
                        * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                        + std::pow(yu_val, 2.0))) // end trace
                    + (((16.0 * std::pow(g3_val, 2.0))
                        - ((2.0 / 5.0) * std::pow(g1_val, 2.0))) // Tr(Yd^2)
                        * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                        + std::pow(yd_val, 2.0))) // end trace
                    + ((6.0 / 5.0) * std::pow(g1_val, 2.0) // Tr(Ye^2)
                        * (std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                        + std::pow(ye_val, 2.0))) // end trace
                    + ((15.0 / 2.0) * std::pow(g2_val, 4.0))
                    + ((9.0 / 5.0) * std::pow(g1_val, 2.0) * std::pow(g2_val, 2.0))
                    + ((207.0 / 50.0) * std::pow(g1_val, 4.0))))
                + (mu_val * (((-12.0) // Tr(3au*Yu^3 + 3ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + ae*Ye^3)
                        * ((3.0 *  ((at_val * std::pow(yt_val, 3.0))
                                + (ac_val * std::pow(yc_val, 3.0))
                                + (au_val * std::pow(yu_val, 3.0))
                                + (ab_val * std::pow(yb_val, 3.0))
                                + (as_val * std::pow(ys_val, 3.0))
                                + (ad_val * std::pow(yd_val, 3.0))))
                        + ((at_val * std::pow(yb_val, 2.0) * yt_val)
                            + (ac_val * std::pow(ys_val, 2.0) * yc_val)
                            + (au_val * std::pow(yd_val, 2.0) * yu_val))
                        + ((ab_val * std::pow(yt_val, 2.0) * yb_val)
                            + (as_val * std::pow(yc_val, 2.0) * ys_val)
                            + (ad_val * std::pow(yu_val, 2.0) * yd_val))
                        + ((atau_val * std::pow(ytau_val, 3.0))
                            + (amu_val * std::pow(ymu_val, 3.0))
                            + (ae_val * std::pow(ye_val, 3.0))))) // end trace
                    + (((32.0 * std::pow(g3_val, 2.0))
                        + ((8.0 / 5.0) * std::pow(g1_val, 2.0))) // Tr(au*Yu)
                        * ((at_val * yt_val) + (ac_val * yc_val)
                            + (au_val * yu_val))) // end trace
                        + (((32.0 * std::pow(g3_val, 2.0))
                        - ((4.0 / 5.0) * std::pow(g1_val, 2.0))) // Tr(ad*Yd)
                        * ((ab_val * yb_val) + (as_val * ys_val)
                            + (ad_val * yd_val))) // end trace
                        + ((12.0 / 5.0) * std::pow(g1_val, 2.0) // Tr(ae*Ye)
                        * ((atau_val * ytau_val) + (amu_val * ymu_val)
                            + (ae_val * ye_val))) // end trace
                        - (((32.0 * std::pow(g3_val, 2.0) * M3_val)
                        + ((8.0 / 5.0) * std::pow(g1_val, 2.0) * M1_val)) // Tr(Yu^2)
                        * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                            + std::pow(yu_val, 2.0)))
                        - (((32.0 * std::pow(g3_val, 2.0) * M3_val) // end trace
                        - ((4.0 / 5.0) * std::pow(g1_val, 2.0) * M1_val)) // Tr(Yd^2)
                        * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                            + std::pow(yd_val, 2.0))) // end trace
                        - ((12.0 / 5.0) * std::pow(g1_val, 2.0) * M1_val // Tr(Ye^2)
                        * (std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                            + std::pow(ye_val, 2.0))) // end trace
                        - (30.0 * std::pow(g2_val, 4.0) * M2_val)
                        - ((18.0 / 5.0) * std::pow(g1_val, 2.0)
                        * std::pow(g2_val, 2.0)
                        * (M1_val + M2_val))
                        - ((414.0 / 25.0) * std::pow(g1_val, 4.0) * M1_val))));

    // Total b beta function
    double db_dt = ((loop_fac * db_dt_1l) + (loop_fac_sq * db_dt_2l));

    // Scalar squared masses
    /////////////////////////////////////////////////////////////////////////////////
    // Introduce S, S', and sigma terms
    double S_val = (mHu_sq_val - mHd_sq_val + mQ3_sq_val + mQ2_sq_val + mQ1_sq_val
            - mL3_sq_val - mL2_sq_val - mL1_sq_val
            - (2.0 * (mU3_sq_val + mU2_sq_val + mU1_sq_val))
            + mD3_sq_val + mD2_sq_val + mD1_sq_val
            + mE3_sq_val + mE2_sq_val + mE1_sq_val);
    
    double Spr_val = ((( (-1.0) * ((((3.0 * mHu_sq_val) + mQ3_sq_val)
                        * std::pow(yt_val, 2.0))
                        + (((3.0 * mHu_sq_val) + mQ2_sq_val)
                            * std::pow(yc_val, 2.0))
                        + (((3.0 * mHu_sq_val) + mQ1_sq_val)
                            * std::pow(yu_val, 2.0))))
                + (4.0 * std::pow(yt_val, 2.0) * mU3_sq_val)
                + (4.0 * std::pow(yc_val, 2.0) * mU2_sq_val)
                + (4.0 * std::pow(yu_val, 2.0) * mU1_sq_val)
                + ((((3.0 * mHd_sq_val) - mQ3_sq_val) * std::pow(yb_val, 2.0))
                    + (((3.0 * mHd_sq_val) - mQ2_sq_val)
                        * std::pow(ys_val, 2.0))
                    + (((3.0 * mHd_sq_val) - mQ1_sq_val)
                        * std::pow(yd_val, 2.0)))
                - (2.0 * ((mD3_sq_val * std::pow(yb_val, 2.0))
                        + (mD2_sq_val * std::pow(ys_val, 2.0))
                        + (mD1_sq_val * std::pow(yd_val, 2.0))))
                + (((mHd_sq_val + mL3_sq_val) * std::pow(ytau_val, 2.0))
                    + ((mHd_sq_val + mL2_sq_val) * std::pow(ymu_val, 2.0))
                    + ((mHd_sq_val + mL1_sq_val) * std::pow(ye_val, 2.0)))
                - (2.0 * ((std::pow(ytau_val, 2.0) * mE3_sq_val)
                        + (std::pow(ymu_val, 2.0) * mE2_sq_val)
                        + (std::pow(ye_val, 2.0) * mE1_sq_val)))) // end trace
                + ((((3.0 / 2.0) * std::pow(g2_val, 2.0))
                    + ((3.0 / 10.0) * std::pow(g1_val, 2.0)))
                    * (mHu_sq_val - mHd_sq_val // Tr(mL^2)
                        - (mL3_sq_val + mL2_sq_val + mL1_sq_val))) // end trace
                + ((((8.0 / 3.0) * std::pow(g3_val, 2.0))
                    + ((3.0 / 2.0) * std::pow(g2_val, 2.0))
                    + ((1.0 / 30.0) * std::pow(g1_val, 2.0))) // Tr(mQ^2)
                    * (mQ3_sq_val + mQ2_sq_val + mQ1_sq_val)) // end trace
                - ((((16.0 / 3.0) * std::pow(g3_val, 2.0))
                    + ((16.0 / 15.0) * std::pow(g1_val, 2.0))) // Tr (mU^2)
                    * (mU3_sq_val + mU2_sq_val + mU1_sq_val)) // end trace
                + ((((8.0 / 3.0) * std::pow(g3_val, 2.0))
                    + ((2.0 / 15.0) * std::pow(g1_val, 2.0))) // Tr(mD^2)
                    * (mD3_sq_val + mD2_sq_val + mD1_sq_val)) // end trace
                + ((6.0 / 5.0) * std::pow(g1_val, 2.0) // Tr(mE^2)
                    * (mE3_sq_val + mE2_sq_val + mE1_sq_val))); // end trace

    double sigma1 = ((1.0 / 5.0) * std::pow(g1_val, 2.0)
            * ((3.0 * (mHu_sq_val + mHd_sq_val)) // Tr(mQ^2 + 3mL^2 + 8mU^2 + 2mD^2 + 6mE^2)
                + mQ3_sq_val + mQ2_sq_val + mQ1_sq_val
                + (3.0 * (mL3_sq_val + mL2_sq_val + mL1_sq_val))
                + (8.0 * (mU3_sq_val + mU2_sq_val + mU1_sq_val))
                + (2.0 * (mD3_sq_val + mD2_sq_val + mD1_sq_val))
                + (6.0 * (mE3_sq_val + mE2_sq_val + mE1_sq_val)))); // end trace

    double sigma2 = (std::pow(g2_val, 2.0)
            * (mHu_sq_val + mHd_sq_val // Tr(3mQ^2 + mL^2)
                + (3.0 * (mQ3_sq_val + mQ2_sq_val + mQ1_sq_val))
                + mL3_sq_val + mL2_sq_val + mL1_sq_val)); // end trace

    double sigma3 = (std::pow(g3_val, 2.0) // Tr(2mQ^2 + mU^2 + mD^2)
            * ((2.0 * (mQ3_sq_val + mQ2_sq_val + mQ1_sq_val))
                + mU3_sq_val + mU2_sq_val + mU1_sq_val
                + mD3_sq_val + mD2_sq_val + mD1_sq_val)); // end trace
    
    // 1-loop parts of masses
    double dmHu_sq_dt_1l = ((6.0 // Tr((mHu^2 + mQ^2) * Yu^2 + Yu^2.0 * mU^2 + au^2)
                        * (((mHu_sq_val + mQ3_sq_val) * std::pow(yt_val, 2.0))
                        + ((mHu_sq_val + mQ2_sq_val)
                            * std::pow(yc_val, 2.0))
                        + ((mHu_sq_val + mQ1_sq_val)
                            * std::pow(yu_val, 2.0))
                        + (mU3_sq_val * std::pow(yt_val, 2.0))
                        + (mU2_sq_val * std::pow(yc_val, 2.0))
                        + (mU1_sq_val * std::pow(yu_val, 2.0))
                        + std::pow(at_val, 2.0) + std::pow(ac_val, 2.0)
                        + std::pow(au_val, 2.0))) // end trace
                        - (6.0 * std::pow(g2_val, 2.0) * std::pow(M2_val, 2.0))
                        - ((6.0 / 5.0) * std::pow(g1_val, 2.0)
                        * std::pow(M1_val, 2.0))
                        + ((3.0 / 5.0) * std::pow(g1_val, 2.0) * S_val));

    double dmHd_sq_dt_1l = ((6.0 * (((mHd_sq_val + mQ3_sq_val)
                            * std::pow(yb_val, 2.0))
                            + ((mHd_sq_val + mQ2_sq_val)
                                * std::pow(ys_val, 2.0))
                            + ((mHd_sq_val + mQ1_sq_val)
                                * std::pow(yd_val, 2.0)))
                        + (6.0 * ((mD3_sq_val * std::pow(yb_val, 2.0))
                                + (mD2_sq_val * std::pow(ys_val, 2.0))
                                + (mD1_sq_val * std::pow(yd_val, 2.0))))
                        + (2.0 * (((mHd_sq_val + mL3_sq_val)
                                * std::pow(ytau_val, 2.0))
                                + ((mHd_sq_val + mL2_sq_val)
                                    * std::pow(ymu_val, 2.0))
                                + ((mHd_sq_val + mL1_sq_val)
                                    * std::pow(ye_val, 2.0))))
                        + (2.0 * ((mE3_sq_val * std::pow(ytau_val, 2.0))
                                + (mE2_sq_val * std::pow(ymu_val, 2.0))
                                + (mE1_sq_val * std::pow(ye_val, 2.0))))
                        + (6.0 * (std::pow(ab_val, 2.0) + std::pow(as_val, 2.0)
                                + std::pow(ad_val, 2.0)))
                        + (2.0 * (std::pow(atau_val, 2.0) + std::pow(amu_val, 2.0)
                                + std::pow(ae_val, 2.0)))) // end trace
                        - (6.0 * std::pow(g2_val, 2.0) * std::pow(M2_val, 2.0))
                        - ((6.0 / 5.0) * std::pow(g1_val, 2.0)
                        * std::pow(M1_val, 2.0))
                        - ((3.0 / 5.0) * std::pow(g1_val, 2.0) * S_val));

    double dmQ3_sq_dt_1l = (((mQ3_sq_val + (2.0 * mHu_sq_val))
                        * std::pow(yt_val, 2.0))
                        + ((mQ3_sq_val + (2.0 * mHd_sq_val))
                        * std::pow(yb_val, 2.0))
                        + ((std::pow(yt_val, 2.0) + std::pow(yb_val, 2.0))
                        * mQ3_sq_val)
                        + (2.0 * std::pow(yt_val, 2.0) * mU3_sq_val)
                        + (2.0 * std::pow(yb_val, 2.0) * mD3_sq_val)
                        + (2.0 * std::pow(at_val, 2.0))
                        + (2.0 * std::pow(ab_val, 2.0))
                        - ((32.0 / 3.0) * std::pow(g3_val, 2.0)
                        * std::pow(M3_val, 2.0))
                        - (6.0 * std::pow(g2_val, 2.0) * std::pow(M2_val, 2.0))
                        - ((2.0 / 15.0) * std::pow(g1_val, 2.0)
                        * std::pow(M1_val, 2.0))
                        + ((1.0 / 5.0) * std::pow(g1_val, 2.0) * S_val));

    double dmQ2_sq_dt_1l = (((mQ2_sq_val + (2.0 * mHu_sq_val))
                        * std::pow(yc_val, 2.0))
                        + ((mQ2_sq_val + (2.0 * mHd_sq_val))
                        * std::pow(ys_val, 2.0))
                        + ((std::pow(yc_val, 2.0) + std::pow(ys_val, 2.0))
                        * mQ2_sq_val)
                        + (2.0 * std::pow(yc_val, 2.0) * mU2_sq_val)
                        + (2.0 * std::pow(ys_val, 2.0) * mD2_sq_val)
                        + (2.0 * std::pow(ac_val, 2.0))
                        + (2.0 * std::pow(as_val, 2.0))
                        - ((32.0 / 3.0) * std::pow(g3_val, 2.0)
                        * std::pow(M3_val, 2.0))
                        - (6.0 * std::pow(g2_val, 2.0) * std::pow(M2_val, 2.0))
                        - ((2.0 / 15.0) * std::pow(g1_val, 2.0)
                        * std::pow(M1_val, 2.0))
                        + ((1.0 / 5.0) * std::pow(g1_val, 2.0) * S_val));

    double dmQ1_sq_dt_1l = (((mQ1_sq_val + (2.0 * mHu_sq_val))
                        * std::pow(yu_val, 2.0))
                        + ((mQ1_sq_val + (2.0 * mHd_sq_val))
                        * std::pow(yd_val, 2.0))
                        + ((std::pow(yu_val, 2.0)
                        + std::pow(yd_val, 2.0)) * mQ1_sq_val)
                        + (2.0 * std::pow(yu_val, 2.0) * mU1_sq_val)
                        + (2.0 * std::pow(yd_val, 2.0) * mD1_sq_val)
                        + (2.0 * std::pow(au_val, 2.0))
                        + (2.0 * std::pow(ad_val, 2.0))
                        - ((32.0 / 3.0) * std::pow(g3_val, 2.0)
                        * std::pow(M3_val, 2.0))
                        - (6.0 * std::pow(g2_val, 2.0) * std::pow(M2_val, 2.0))
                        - ((2.0 / 15.0) * std::pow(g1_val, 2.0)
                        * std::pow(M1_val, 2.0))
                        + ((1.0 / 5.0) * std::pow(g1_val, 2.0) * S_val));

    // Left leptons
    double dmL3_sq_dt_1l = (((mL3_sq_val + (2.0 * mHd_sq_val))
                        * std::pow(ytau_val, 2.0))
                        + (2.0 * std::pow(ytau_val, 2.0) * mE3_sq_val)
                        + (std::pow(ytau_val, 2.0) * mL3_sq_val)
                        + (2.0 * std::pow(atau_val, 2.0))
                        - (6.0 * std::pow(g2_val, 2.0) * std::pow(M2_val, 2.0))
                        - ((6.0 / 5.0) * std::pow(g1_val, 2.0)
                        * std::pow(M1_val, 2.0))
                        - ((3.0 / 5.0) * std::pow(g1_val, 2.0) * S_val));

    double dmL2_sq_dt_1l = (((mL2_sq_val + (2.0 * mHd_sq_val))
                        * std::pow(ymu_val, 2.0))
                        + (2.0 * std::pow(ymu_val, 2.0) * mE2_sq_val)
                        + (std::pow(ymu_val, 2.0) * mL2_sq_val)
                        + (2.0 * std::pow(amu_val, 2.0))
                        - (6.0 * std::pow(g2_val, 2.0) * std::pow(M2_val, 2.0))
                        - ((6.0 / 5.0) * std::pow(g1_val, 2.0)
                        * std::pow(M1_val, 2.0))
                        - ((3.0 / 5.0) * std::pow(g1_val, 2.0) * S_val));

    double dmL1_sq_dt_1l = (((mL1_sq_val + (2.0 * mHd_sq_val))
                        * std::pow(ye_val, 2.0))
                        + (2.0 * std::pow(ye_val, 2.0) * mE1_sq_val)
                        + (std::pow(ye_val, 2.0) * mL1_sq_val)
                        + (2.0 * std::pow(ae_val, 2.0))
                        - (6.0 * std::pow(g2_val, 2.0) * std::pow(M2_val, 2.0))
                        - ((6.0 / 5.0) * std::pow(g1_val, 2.0)
                        * std::pow(M1_val, 2.0))
                        - ((3.0 / 5.0) * std::pow(g1_val, 2.0) * S_val));

    // Right up-type squarks
    double dmU3_sq_dt_1l = ((2.0 * (mU3_sq_val + (2.0 * mHu_sq_val))
                        * std::pow(yt_val, 2.0))
                        + (4.0 * std::pow(yt_val, 2.0) * mQ3_sq_val)
                        + (2.0 * std::pow(yt_val, 2.0) * mU3_sq_val)
                        + (4.0 * std::pow(at_val, 2.0))
                        - ((32.0 / 3.0) * std::pow(g3_val, 2.0)
                        * std::pow(M3_val, 2.0))
                        - ((32.0 / 15.0) * std::pow(g1_val, 2.0)
                        * std::pow(M1_val, 2.0))
                        - ((4.0 / 5.0) * std::pow(g1_val, 2.0) * S_val));

    double dmU2_sq_dt_1l = ((2.0 * (mU2_sq_val + (2.0 * mHu_sq_val))
                        * std::pow(yc_val, 2.0))
                        + (4.0 * std::pow(yc_val, 2.0) * mQ2_sq_val)
                        + (2.0 * std::pow(yc_val, 2.0) * mU2_sq_val)
                        + (4.0 * std::pow(ac_val, 2.0))
                        - ((32.0 / 3.0) * std::pow(g3_val, 2.0)
                        * std::pow(M3_val, 2.0))
                        - ((32.0 / 15.0) * std::pow(g1_val, 2.0)
                        * std::pow(M1_val, 2.0))
                        - ((4.0 / 5.0) * std::pow(g1_val, 2.0) * S_val));

    double dmU1_sq_dt_1l = ((2.0 * (mU1_sq_val + (2.0 * mHu_sq_val))
                        * std::pow(yu_val, 2.0))
                        + (4.0 * std::pow(yu_val, 2.0) * mQ1_sq_val)
                        + (2.0 * std::pow(yu_val, 2.0) * mU1_sq_val)
                        + (4.0 * std::pow(au_val, 2.0))
                        - ((32.0 / 3.0) * std::pow(g3_val, 2.0)
                        * std::pow(M3_val, 2.0))
                        - ((32.0 / 15.0) * std::pow(g1_val, 2.0)
                        * std::pow(M1_val, 2.0))
                        - ((4.0 / 5.0) * std::pow(g1_val, 2.0) * S_val));

    // Right down-type squarks
    double dmD3_sq_dt_1l = ((2.0 * (mD3_sq_val + (2.0 * mHd_sq_val))
                        * std::pow(yb_val, 2.0))
                        + (4.0 * std::pow(yb_val, 2.0) * mQ3_sq_val)
                        + (2.0 * std::pow(yb_val, 2.0) * mD3_sq_val)
                        + (4.0 * std::pow(ab_val, 2.0))
                        - ((32.0 / 3.0) * std::pow(g3_val, 2.0)
                        * std::pow(M3_val, 2.0))
                        - ((8.0 / 15.0) * std::pow(g1_val, 2.0)
                        * std::pow(M1_val, 2.0))
                        + ((2.0 / 5.0) * std::pow(g1_val, 2.0) * S_val));

    double dmD2_sq_dt_1l = ((2.0 * (mD2_sq_val + (2.0 * mHd_sq_val))
                        * std::pow(ys_val, 2.0))
                        + (4.0 * std::pow(ys_val, 2.0) * mQ2_sq_val)
                        + (2.0 * std::pow(ys_val, 2.0) * mD2_sq_val)
                        + (4.0 * std::pow(as_val, 2.0))
                        - ((32.0 / 3.0) * std::pow(g3_val, 2.0)
                        * std::pow(M3_val, 2.0))
                        - ((8.0 / 15.0) * std::pow(g1_val, 2.0)
                        * std::pow(M1_val, 2.0))
                        + ((2.0 / 5.0) * std::pow(g1_val, 2.0) * S_val));

    double dmD1_sq_dt_1l = ((2.0 * (mD1_sq_val + (2.0 * mHd_sq_val))
                        * std::pow(yd_val, 2.0))
                        + (4.0 * std::pow(yd_val, 2.0) * mQ1_sq_val)
                        + (2.0 * std::pow(yd_val, 2.0) * mD1_sq_val)
                        + (4.0 * std::pow(ad_val, 2.0))
                        - ((32.0 / 3.0) * std::pow(g3_val, 2.0)
                        * std::pow(M3_val, 2.0))
                        - ((8.0 / 15.0) * std::pow(g1_val, 2.0)
                        * std::pow(M1_val, 2.0))
                        + ((2.0 / 5.0) * std::pow(g1_val, 2.0) * S_val));

    // Right leptons
    double dmE3_sq_dt_1l = ((2.0 * (mE3_sq_val + (2.0 * mHd_sq_val))
                        * std::pow(ytau_val, 2.0))
                        + (4.0 * std::pow(ytau_val, 2.0) * mL3_sq_val)
                        + (2.0 * std::pow(ytau_val, 2.0) * mE3_sq_val)
                        + (4.0 * std::pow(atau_val, 2.0))
                        - ((24.0 / 5.0) * std::pow(g1_val, 2.0)
                        * std::pow(M1_val, 2.0))
                        + ((6.0 / 5.0) * std::pow(g1_val, 2.0) * S_val));

    double dmE2_sq_dt_1l = ((2.0 * (mE2_sq_val + (2.0 * mHd_sq_val))
                        * std::pow(ymu_val, 2.0))
                        + (4.0 * std::pow(ymu_val, 2.0) * mL2_sq_val)
                        + (2.0 * std::pow(ymu_val, 2.0) * mE2_sq_val)
                        + (4.0 * std::pow(amu_val, 2.0))
                        - ((24.0 / 5.0) * std::pow(g1_val, 2.0)
                        * std::pow(M1_val, 2.0))
                        + ((6.0 / 5.0) * std::pow(g1_val, 2.0) * S_val));

    double dmE1_sq_dt_1l = ((2.0 * (mE1_sq_val + (2.0 * mHd_sq_val))
                        * std::pow(ye_val, 2.0))
                        + (4.0 * std::pow(ye_val, 2.0) * mL1_sq_val)
                        + (2.0 * std::pow(ye_val, 2.0) * mE1_sq_val)
                        + (4.0 * std::pow(ae_val, 2.0))
                        - ((24.0 / 5.0) * std::pow(g1_val, 2.0)
                        * std::pow(M1_val, 2.0))
                        + ((6.0 / 5.0) * std::pow(g1_val, 2.0) * S_val));

    // 2-loop parts of masses
    double dmHu_sq_dt_2l = (((-6.0)  // Tr(6(mHu^2 + mQ^2)*Yu^4 + 6Yu^4 * mU^2 + (mHu^2 + mHd^2 + mQ^2) * Yu^2.0 * Yd^2 + Yu^2.0 * Yd^2.0 * mU^2 + Yu^2.0 * Yd^2.0 * mQ^2 + Yu^2.0 * Yd^2.0 * mD^2 + 12au^2.0 * Yu^2 + ad^2.0 * Yu^2 + Yd^2.0 * au^2 + 2ad * Yd * Yu * au)
                        * ((6.0 * (((mHu_sq_val + mQ3_sq_val)
                                * std::pow(yt_val, 4.0))
                                + ((mHu_sq_val + mQ2_sq_val)
                                    * std::pow(yc_val, 4.0))
                                + ((mHu_sq_val + mQ1_sq_val)
                                    * std::pow(yu_val, 4.0))))
                        + (6.0 * ((mU3_sq_val * std::pow(yt_val, 4.0))
                                    + (mU2_sq_val * std::pow(yc_val, 4.0))
                                    + (mU1_sq_val * std::pow(yu_val, 4.0))))
                        + ((mHu_sq_val + mHd_sq_val + mQ3_sq_val)
                            * std::pow(yt_val, 2.0) * std::pow(yb_val, 2.0))
                        + ((mHu_sq_val + mHd_sq_val + mQ2_sq_val)
                            * std::pow(yc_val, 2.0) * std::pow(ys_val, 2.0))
                        + ((mHu_sq_val + mHd_sq_val + mQ1_sq_val)
                            * std::pow(yu_val, 2.0) * std::pow(yd_val, 2.0))
                        + ((mU3_sq_val + mQ3_sq_val + mD3_sq_val)
                            * std::pow(yt_val, 2.0) * std::pow(yb_val, 2.0))
                        + ((mU2_sq_val + mQ2_sq_val + mD2_sq_val)
                            * std::pow(yc_val, 2.0) * std::pow(ys_val, 2.0))
                        + ((mU1_sq_val + mQ1_sq_val + mD1_sq_val)
                            * std::pow(yu_val, 2.0) * std::pow(yd_val, 2.0))
                        + (12.0 * ((std::pow(at_val, 2.0)
                                    * std::pow(yt_val, 2.0))
                                    + (std::pow(ac_val, 2.0)
                                    * std::pow(yc_val, 2.0))
                                    + (std::pow(au_val, 2.0)
                                    * std::pow(yu_val, 2.0))))
                        + (std::pow(ab_val, 2.0) * std::pow(yt_val, 2.0))
                        + (std::pow(as_val, 2.0) * std::pow(yc_val, 2.0))
                        + (std::pow(ad_val, 2.0) * std::pow(yu_val, 2.0))
                        + (std::pow(yb_val, 2.0) * std::pow(at_val, 2.0))
                        + (std::pow(ys_val, 2.0) * std::pow(ac_val, 2.0))
                        + (std::pow(yd_val, 2.0) * std::pow(au_val, 2.0))
                        + (2.0 * ((yb_val * ab_val * at_val * yt_val)
                                    + (ys_val * as_val * ac_val * yc_val)
                                    + (yd_val * ad_val * au_val * yu_val))))) // end trace
                        + (((32.0 * std::pow(g3_val, 2.0))
                        + ((8.0 / 5.0) * std::pow(g1_val, 2.0)))  // Tr((mHu^2 + mQ^2 + mU^2) * Yu^2 + au^2)
                        * (((mHu_sq_val + mQ3_sq_val + mU3_sq_val)
                            * std::pow(yt_val, 2.0))
                            + ((mHu_sq_val + mQ2_sq_val + mU2_sq_val)
                                * std::pow(yc_val, 2.0))
                            + ((mHu_sq_val + mQ1_sq_val + mU1_sq_val)
                                * std::pow(yu_val, 2.0))
                            + std::pow(at_val, 2.0) + std::pow(ac_val, 2.0)
                            + std::pow(au_val, 2.0))) // end trace
                        + (32.0 * std::pow(g3_val, 2.0)
                        * ((2.0 * std::pow(M3_val, 2.0) // Tr(Yu^2)
                            * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                                + std::pow(yu_val, 2.0))) // end trace
                            - (2.0 * M3_val // Tr(Yu*au)
                                * ((yt_val * at_val) + (yc_val * ac_val)
                                    + (yu_val * au_val))))) // end trace
                        + ((8.0 / 5.0) * std::pow(g1_val, 2.0)
                        * ((2.0 * std::pow(M1_val, 2.0) // Tr(Yu^2)
                            * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                                + std::pow(yu_val, 2.0))) // end trace
                            - (2.0 * M1_val // Tr(Yu*au)
                                * ((yt_val * at_val) + (yc_val * ac_val)
                                    + (yu_val * au_val))))) // end trace
                        + ((6.0 / 5.0) * std::pow(g1_val, 2.0) * Spr_val)
                        + (33.0 * std::pow(g2_val, 4.0) * std::pow(M2_val, 2.0))
                        + ((18.0 / 5.0) * std::pow(g2_val, 2.0)
                        * std::pow(g1_val, 2.0)
                        * (std::pow(M2_val, 2.0) + std::pow(M1_val, 2.0)
                            + (M1_val * M2_val)))
                        + ((621.0 / 25.0) * std::pow(g1_val, 4.0)
                        * std::pow(M1_val, 2.0))
                        + (3.0 * std::pow(g2_val, 2.0) * sigma2)
                        + ((3.0 / 5.0) * std::pow(g1_val, 2.0) * sigma1));

    double dmHd_sq_dt_2l = (((-6.0)  // Tr(6(mHd^2 + mQ^2)*Yd^4 + 6Yd^4 * mD^2 + (mHu^2 + mHd^2 + mQ^2) * Yu^2.0 * Yd^2 + Yu^2.0 * Yd^2.0 * mU^2 + Yu^2.0 * Yd^2.0 * mQ^2 + Yu^2.0 * Yd^2.0 * mD^2 + 2(mHd^2 + mL^2) * Ye^4 + 2Ye^4 * mE^2 + 12ad^2.0 * Yd^2 + ad^2.0 * Yu^2 + Yd^2.0 * au^2 + 2ad * Yd * Yu * au + 4ae^2.0 * Ye^2)
                        * ((6.0 * (((mHd_sq_val + mQ3_sq_val)
                                * std::pow(yb_val, 4.0))
                                + ((mHd_sq_val + mQ2_sq_val)
                                    * std::pow(ys_val, 4.0))
                                + ((mHd_sq_val + mQ1_sq_val)
                                    * std::pow(yd_val, 4.0))))
                        + (6.0 * ((mD3_sq_val * std::pow(yb_val, 4.0))
                                    + (mD2_sq_val * std::pow(ys_val, 4.0))
                                    + (mD1_sq_val * std::pow(yd_val, 4.0))))
                        + ((mHu_sq_val + mHd_sq_val + mQ3_sq_val)
                            * std::pow(yt_val, 2.0) * std::pow(yb_val, 2.0))
                        + ((mHu_sq_val + mHd_sq_val + mQ2_sq_val)
                            * std::pow(yc_val, 2.0) * std::pow(ys_val, 2.0))
                        + ((mHu_sq_val + mHd_sq_val + mQ1_sq_val)
                            * std::pow(yu_val, 2.0) * std::pow(yd_val, 2.0))
                        + ((mU3_sq_val + mQ3_sq_val + mD3_sq_val)
                            * std::pow(yt_val, 2.0) * std::pow(yb_val, 2.0))
                        + ((mU2_sq_val + mQ2_sq_val + mD2_sq_val)
                            * std::pow(yc_val, 2.0) * std::pow(ys_val, 2.0))
                        + ((mU1_sq_val + mQ1_sq_val + mD1_sq_val)
                            * std::pow(yu_val, 2.0) * std::pow(yd_val, 2.0))
                        + (2.0 * (((mHd_sq_val + mL3_sq_val + mE3_sq_val)
                                    * std::pow(ytau_val, 4.0))
                                    + ((mHd_sq_val + mL2_sq_val + mE2_sq_val)
                                    * std::pow(ymu_val, 4.0))
                                    + ((mHd_sq_val + mL1_sq_val + mE1_sq_val)
                                    * std::pow(ye_val, 4.0))))
                        + (12.0 * ((std::pow(ab_val, 2.0)
                                    * std::pow(yb_val, 2.0))
                                    + (std::pow(as_val, 2.0)
                                    * std::pow(ys_val, 2.0))
                                    + (std::pow(ad_val, 2.0)
                                    * std::pow(yd_val, 2.0))))
                        + (std::pow(ab_val, 2.0) * std::pow(yt_val, 2.0))
                        + (std::pow(as_val, 2.0) * std::pow(yc_val, 2.0))
                        + (std::pow(ad_val, 2.0) * std::pow(yu_val, 2.0))
                        + (std::pow(yb_val, 2.0) * std::pow(at_val, 2.0))
                        + (std::pow(ys_val, 2.0) * std::pow(ac_val, 2.0))
                        + (std::pow(yd_val, 2.0) * std::pow(au_val, 2.0))
                        + (2.0 * ((yb_val * ab_val * at_val * yt_val)
                                    + (ys_val * as_val * ac_val * yc_val)
                                    + (yd_val * ad_val * au_val * yu_val)
                                    + (2.0 * ((std::pow(atau_val, 2.0)
                                            * std::pow(ytau_val, 2.0))
                                        + (std::pow(amu_val, 2.0)
                                            * std::pow(ymu_val, 2.0))
                                        + (std::pow(ae_val, 2.0)
                                            * std::pow(ye_val, 2.0)))))))) // end trace
                        + (((32.0 * std::pow(g3_val, 2.0))
                        - ((4.0 / 5.0) * std::pow(g1_val, 2.0)))  // Tr((mHd^2 + mQ^2 + mD^2) * Yd^2 + ad^2)
                        * (((mHu_sq_val + mQ3_sq_val + mD3_sq_val)
                            * std::pow(yb_val, 2.0))
                            + ((mHu_sq_val + mQ2_sq_val + mD2_sq_val)
                                * std::pow(ys_val, 2.0))
                            + ((mHu_sq_val + mQ1_sq_val + mD1_sq_val)
                                * std::pow(yd_val, 2.0))
                            + std::pow(ab_val, 2.0) + std::pow(as_val, 2.0)
                            + std::pow(ad_val, 2.0))) // end trace
                        + (32.0 * std::pow(g3_val, 2.0)
                        * ((2.0 * std::pow(M3_val, 2.0) // Tr(Yd^2)
                            * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                                + std::pow(yd_val, 2.0))) // end trace
                            - (2.0 * M3_val  // Tr(Yd*ad)
                                * ((yb_val * ab_val) + (ys_val * as_val)
                                    + (yd_val * ad_val))))) // end trace
                        - ((4.0 / 5.0) * std::pow(g1_val, 2.0)
                        * ((2.0 * std::pow(M1_val, 2.0) // Tr(Yd^2)
                            * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                                + std::pow(yd_val, 2.0))) // end trace
                            - (2.0 * M1_val  // Tr(Yd*ad)
                                * ((yb_val * ab_val) + (ys_val * as_val)
                                    + (yd_val * ad_val))))) // end trace
                        + ((12.0 / 5.0) * std::pow(g1_val, 2.0)
                        * (( // Tr((mHd^2 + mL^2 + mE^2) * Ye^2 + ae^2)
                            ((mHd_sq_val + mL3_sq_val + mE3_sq_val)
                            * std::pow(ytau_val, 2.0))
                            + ((mHd_sq_val + mL2_sq_val + mE2_sq_val)
                                * std::pow(ymu_val, 2.0))
                            + ((mHd_sq_val + mL1_sq_val + mE1_sq_val)
                                * std::pow(ye_val, 2.0))
                            + std::pow(atau_val, 2.0) + std::pow(amu_val, 2.0)
                            + std::pow(ae_val, 2.0)) // end trace
                            + (2.0 * std::pow(M1_val, 2.0) // Tr(Ye^2)
                                * (std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                                    + std::pow(ye_val, 2.0))) // end trace
                            - (2.0 * M1_val  // Tr(ae * Ye)
                                * ((atau_val * ytau_val)
                                    + (amu_val * ymu_val)
                                    + (ae_val * ye_val))) // end trace
                            ))
                        - ((6.0 / 5.0) * std::pow(g1_val, 2.0) * Spr_val)
                        + (33.0 * std::pow(g2_val, 4.0) * std::pow(M2_val, 2.0))
                        + ((18.0 / 5.0) * std::pow(g2_val, 2.0)
                        * std::pow(g1_val, 2.0)
                        * (std::pow(M2_val, 2.0) + std::pow(M1_val, 2.0)
                            + (M1_val * M2_val)))
                        + ((621.0 / 25.0) * std::pow(g1_val, 4.0)
                        * std::pow(M1_val, 2.0))
                        + (3.0 * std::pow(g2_val, 2.0) * sigma2)
                        + ((3.0 / 5.0) * std::pow(g1_val, 2.0) * sigma1));

        // Left squarks
    double dmQ3_sq_dt_2l = (((-8.0)* (mQ3_sq_val + mHu_sq_val + mU3_sq_val)
                        * std::pow(yt_val, 4.0))
                        - (8.0 * (mQ3_sq_val + mHd_sq_val + mD3_sq_val)
                        * std::pow(yb_val, 4.0))
                        - (std::pow(yt_val, 2.0)
                        * ((2.0 * mQ3_sq_val) + (2.0 * mU3_sq_val)
                            + (4.0 * mHu_sq_val)) // Tr(3Yu^2)
                        * 3.0 * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                                + std::pow(yu_val, 2.0))) // end trace
                        - (std::pow(yb_val, 2.0)
                        * ((2.0 * mQ3_sq_val) + (2.0 * mD3_sq_val)
                            + (4.0 * mHd_sq_val)) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                            + std::pow(ye_val, 2.0))) // end trace
                        - (6.0 * std::pow(yt_val, 2.0) // Tr((mQ^2 + mU^2)*Yu^2)
                        * (((mQ3_sq_val + mU3_sq_val)
                            * std::pow(yt_val, 2.0))
                            + ((mQ2_sq_val + mU2_sq_val)
                                * std::pow(yc_val, 2.0))
                            + ((mQ1_sq_val + mU1_sq_val)
                                * std::pow(yu_val, 2.0)))) // end trace
                        - (std::pow(yb_val, 2.0) // Tr(6(mQ^2 + mD^2)*Yd^2 + 2(mL^2 + mE^2)*Ye^2)
                        * ((6.0 * (((mQ3_sq_val + mD3_sq_val)
                                    * std::pow(yb_val, 2.0))
                                    + ((mQ2_sq_val + mD2_sq_val)
                                    * std::pow(ys_val, 2.0))
                                    + ((mQ1_sq_val + mD1_sq_val)
                                    * std::pow(yd_val, 2.0))))
                            + (2.0 * (((mL3_sq_val + mE3_sq_val)
                                    * std::pow(ytau_val, 2.0))
                                    + ((mL2_sq_val + mE2_sq_val)
                                    * std::pow(ymu_val, 2.0))
                                    + ((mL1_sq_val + mE1_sq_val)
                                    * std::pow(ye_val, 2.0)))) // end trace
                            ))
                        - (16.0 * std::pow(yt_val, 2.0) * std::pow(at_val, 2.0))
                        - (16.0 * std::pow(yb_val, 2.0) * std::pow(ab_val, 2.0))
                        - (std::pow(at_val, 2.0) // Tr(6Yu^2)
                        * 6.0 * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                                + std::pow(yu_val, 2.0))) // end trace
                        - (std::pow(yt_val, 2.0) // Tr(6au^2)
                        * 6.0 * (std::pow(at_val, 2.0) + std::pow(ac_val, 2.0)
                                + std::pow(au_val, 2.0))) // end trace
                        - (at_val * yt_val // Tr(12Yu*au)
                        * 12.0 * ((yt_val * at_val) + (yc_val * ac_val)
                                + (yu_val * au_val))) // end trace
                        - (std::pow(ab_val, 2.0) // Tr(6Yd^2 + 2Ye^2)
                        * ((6.0 * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + (2.0 * (std::pow(ytau_val, 2.0)
                                    + std::pow(ymu_val, 2.0)
                                    + std::pow(ye_val, 2.0))))) // end trace
                        - (std::pow(yb_val, 2.0) // Tr(6ad^2 + 2ae^2)
                        * ((6.0 * (std::pow(ab_val, 2.0) + std::pow(as_val, 2.0)
                                    + std::pow(ad_val, 2.0)))
                            + (2.0 * (std::pow(atau_val, 2.0)
                                    + std::pow(amu_val, 2.0)
                                    + std::pow(ae_val, 2.0))))) // end trace
                        - (2.0 * ab_val * yb_val // Tr(6Yd*ad + 2Ye*ae)
                        * ((6.0 * ((yb_val * ab_val) + (ys_val * as_val)
                                    + (yd_val * ad_val)))
                            + (2.0 * ((ytau_val * atau_val)
                                    + (ymu_val * amu_val)
                                    + (ye_val * ae_val))))) // end trace
                        + ((2.0 / 5.0) * std::pow(g1_val, 2.0)
                        * ((4.0 * (mQ3_sq_val + mHu_sq_val + mU3_sq_val)
                            * std::pow(yt_val, 2.0))
                            + (4.0 * std::pow(at_val, 2.0))
                            - (8.0 * M1_val * at_val * yt_val)
                            + (8.0 * std::pow(M1_val, 2.0) * std::pow(yt_val, 2.0))
                            + (2.0 * (mQ3_sq_val + mHd_sq_val + mD3_sq_val)
                                * std::pow(yb_val, 2.0))
                            + (2.0 * std::pow(ab_val, 2.0))
                            - (4.0 * M1_val * ab_val * yb_val)
                            + (4.0 * std::pow(M1_val, 2.0)
                                * std::pow(yb_val, 2.0))))
                        + ((2.0 / 5.0) * std::pow(g1_val, 2.0) * Spr_val)
                        - ((128.0 / 3.0) * std::pow(g3_val, 4.0)
                        * std::pow(M3_val, 2.0))
                        + (32.0 * std::pow(g3_val, 2.0) * std::pow(g2_val, 2.0)
                        * (std::pow(M3_val, 2.0) + std::pow(M2_val, 2.0)
                            + (M2_val * M3_val)))
                        + ((32 / 45) * std::pow(g3_val, 2.0)
                        * std::pow(g1_val, 2.0)
                        * (std::pow(M3_val, 2.0) + std::pow(M1_val, 2.0)
                            + (M3_val * M1_val)))
                        + (33.0 * std::pow(g2_val, 4.0) * std::pow(M2_val, 2.0))
                        + ((2.0 / 5.0) * std::pow(g2_val, 2.0) * std::pow(g1_val, 2.0)
                        * (std::pow(M1_val, 2.0) + std::pow(M2_val, 2.0)
                            + (M1_val * M2_val)))
                        + ((199 / 75) * std::pow(g1_val, 4.0) * std::pow(M1_val, 2.0))
                        + ((16.0 / 3.0) * std::pow(g3_val, 2.0) * sigma3)
                        + (3.0 * std::pow(g2_val, 2.0) * sigma2)
                        + ((1 / 15) * std::pow(g1_val, 2.0) * sigma1));

    double dmQ2_sq_dt_2l = (((-8.0)* (mQ2_sq_val + mHu_sq_val + mU2_sq_val)
                        * std::pow(yc_val, 4.0))
                        - (8.0 * (mQ2_sq_val + mHd_sq_val + mD2_sq_val)
                        * std::pow(ys_val, 4.0))
                        - (std::pow(yc_val, 2.0)
                        * ((2.0 * mQ2_sq_val) + (2.0 * mU2_sq_val)
                            + (4.0 * mHu_sq_val)) // Tr(3Yu^2)
                        * 3.0 * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                                + std::pow(yu_val, 2.0))) // end trace
                        - (std::pow(ys_val, 2.0)
                        * ((2.0 * mQ2_sq_val) + (2.0 * mD2_sq_val)
                            + (4.0 * mHd_sq_val)) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                            + std::pow(ye_val, 2.0))) // end trace
                        - (6.0 * std::pow(yc_val, 2.0) // Tr((mQ^2 + mU^2)*Yu^2)
                        * (((mQ3_sq_val + mU3_sq_val)
                            * std::pow(yt_val, 2.0))
                            + ((mQ2_sq_val + mU2_sq_val)
                                * std::pow(yc_val, 2.0))
                            + ((mQ1_sq_val + mU1_sq_val)
                                * std::pow(yu_val, 2.0)))) // end trace
                        - (std::pow(ys_val, 2.0) // Tr(6(mQ^2 + mD^2)*Yd^2 + 2(mL^2 + mE^2)*Ye^2)
                        * ((6.0 * (((mQ3_sq_val + mD3_sq_val)
                                    * std::pow(yb_val, 2.0))
                                    + ((mQ2_sq_val + mD2_sq_val)
                                    * std::pow(ys_val, 2.0))
                                    + ((mQ1_sq_val + mD1_sq_val)
                                    * std::pow(yd_val, 2.0))))
                            + (2.0 * (((mL3_sq_val + mE3_sq_val)
                                    * std::pow(ytau_val, 2.0))
                                    + ((mL2_sq_val + mE2_sq_val)
                                    * std::pow(ymu_val, 2.0))
                                    + ((mL1_sq_val + mE1_sq_val)
                                    * std::pow(ye_val, 2.0)))) // end trace
                            ))
                        - (16.0 * std::pow(yc_val, 2.0) * std::pow(ac_val, 2.0))
                        - (16.0 * std::pow(ys_val, 2.0) * std::pow(as_val, 2.0))
                        - (std::pow(ac_val, 2.0) // Tr(6Yu^2)
                        * 6.0 * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                                + std::pow(yu_val, 2.0))) // end trace
                        - (std::pow(yc_val, 2.0) // Tr(6au^2)
                        * 6.0 * (std::pow(at_val, 2.0) + std::pow(ac_val, 2.0)
                                + std::pow(au_val, 2.0))) // end trace
                        - (ac_val * yc_val // Tr(12Yu*au)
                        * 12.0 * ((yt_val * at_val) + (yc_val * ac_val)
                                + (yu_val * au_val))) // end trace
                        - (std::pow(as_val, 2.0) // Tr(6Yd^2 + 2Ye^2)
                        * ((6.0 * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + (2.0 * (std::pow(ytau_val, 2.0)
                                    + std::pow(ymu_val, 2.0)
                                    + std::pow(ye_val, 2.0))))) // end trace
                        - (std::pow(ys_val, 2.0) // Tr(6ad^2 + 2ae^2)
                        * ((6.0 * (std::pow(ab_val, 2.0) + std::pow(as_val, 2.0)
                                    + std::pow(ad_val, 2.0)))
                            + (2.0 * (std::pow(atau_val, 2.0)
                                    + std::pow(amu_val, 2.0)
                                    + std::pow(ae_val, 2.0))))) // end trace
                        - (2.0 * as_val * ys_val // Tr(6Yd*ad + 2Ye*ae)
                        * ((6.0 * ((yb_val * ab_val) + (ys_val * as_val)
                                    + (yd_val * ad_val)))
                            + (2.0 * ((ytau_val * atau_val)
                                    + (ymu_val * amu_val)
                                    + (ye_val * ae_val))))) // end trace
                        + ((2.0 / 5.0) * std::pow(g1_val, 2.0)
                        * ((4.0 * (mQ2_sq_val + mHu_sq_val + mU2_sq_val)
                            * std::pow(yc_val, 2.0))
                            + (4.0 * std::pow(ac_val, 2.0))
                            - (8.0 * M1_val * ac_val * yc_val)
                            + (8.0 * std::pow(M1_val, 2.0) * std::pow(yc_val, 2.0))
                            + (2.0 * (mQ2_sq_val + mHd_sq_val + mD2_sq_val)
                                * std::pow(ys_val, 2.0))
                            + (2.0 * std::pow(as_val, 2.0))
                            - (4.0 * M1_val * as_val * ys_val)
                            + (4.0 * std::pow(M1_val, 2.0)
                                * std::pow(ys_val, 2.0))))
                        + ((2.0 / 5.0) * std::pow(g1_val, 2.0) * Spr_val)
                        - ((128.0 / 3.0) * std::pow(g3_val, 4.0)
                        * std::pow(M3_val, 2.0))
                        + (32.0 * std::pow(g3_val, 2.0) * std::pow(g2_val, 2.0)
                        * (std::pow(M3_val, 2.0) + std::pow(M2_val, 2.0)
                            + (M2_val * M3_val)))
                        + ((32 / 45) * std::pow(g3_val, 2.0)
                        * std::pow(g1_val, 2.0)
                        * (std::pow(M3_val, 2.0) + std::pow(M1_val, 2.0)
                            + (M3_val * M1_val)))
                        + (33.0 * std::pow(g2_val, 4.0) * std::pow(M2_val, 2.0))
                        + ((2.0 / 5.0) * std::pow(g2_val, 2.0) * std::pow(g1_val, 2.0)
                        * (std::pow(M1_val, 2.0) + std::pow(M2_val, 2.0)
                            + (M1_val * M2_val)))
                        + ((199 / 75) * std::pow(g1_val, 4.0) * std::pow(M1_val, 2.0))
                        + ((16.0 / 3.0) * std::pow(g3_val, 2.0) * sigma3)
                        + (3.0 * std::pow(g2_val, 2.0) * sigma2)
                        + ((1 / 15) * std::pow(g1_val, 2.0) * sigma1));

    double dmQ1_sq_dt_2l = (((-8.0)* (mQ1_sq_val + mHu_sq_val + mU1_sq_val)
                        * std::pow(yu_val, 4.0))
                        - (8.0 * (mQ1_sq_val + mHd_sq_val + mD1_sq_val)
                        * std::pow(yd_val, 4.0))
                        - (std::pow(yu_val, 2.0)
                        * ((2.0 * mQ1_sq_val) + (2.0 * mU1_sq_val)
                            + (4.0 * mHu_sq_val)) // Tr(3Yu^2)
                        * 3.0 * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                                + std::pow(yu_val, 2.0))) // end trace
                        - (std::pow(yd_val, 2.0)
                        * ((2.0 * mQ1_sq_val) + (2.0 * mD1_sq_val)
                            + (4.0 * mHd_sq_val)) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                            + std::pow(ye_val, 2.0))) // end trace
                        - (6.0 * std::pow(yu_val, 2.0) // Tr((mQ^2 + mU^2)*Yu^2)
                        * (((mQ3_sq_val + mU3_sq_val)
                            * std::pow(yt_val, 2.0))
                            + ((mQ2_sq_val + mU2_sq_val)
                                * std::pow(yc_val, 2.0))
                            + ((mQ1_sq_val + mU1_sq_val)
                                * std::pow(yu_val, 2.0)))) // end trace
                        - (std::pow(yd_val, 2.0) // Tr(6(mQ^2 + mD^2)*Yd^2 + 2(mL^2 + mE^2)*Ye^2)
                        * ((6.0 * (((mQ3_sq_val + mD3_sq_val)
                                    * std::pow(yb_val, 2.0))
                                    + ((mQ2_sq_val + mD2_sq_val)
                                    * std::pow(ys_val, 2.0))
                                    + ((mQ1_sq_val + mD1_sq_val)
                                    * std::pow(yd_val, 2.0))))
                            + (2.0 * (((mL3_sq_val + mE3_sq_val)
                                    * std::pow(ytau_val, 2.0))
                                    + ((mL2_sq_val + mE2_sq_val)
                                    * std::pow(ymu_val, 2.0))
                                    + ((mL1_sq_val + mE1_sq_val)
                                    * std::pow(ye_val, 2.0)))) // end trace
                            ))
                        - (16.0 * std::pow(yu_val, 2.0) * std::pow(au_val, 2.0))
                        - (16.0 * std::pow(yd_val, 2.0) * std::pow(ad_val, 2.0))
                        - (std::pow(au_val, 2.0) // Tr(6Yu^2)
                        * 6.0 * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                                + std::pow(yu_val, 2.0))) // end trace
                        - (std::pow(yu_val, 2.0) // Tr(6au^2)
                        * 6.0 * (std::pow(at_val, 2.0) + std::pow(ac_val, 2.0)
                                + std::pow(au_val, 2.0))) // end trace
                        - (au_val * yu_val // Tr(12Yu*au)
                        * 12.0 * ((yt_val * at_val) + (yc_val * ac_val)
                                + (yu_val * au_val))) // end trace
                        - (std::pow(ad_val, 2.0) // Tr(6Yd^2 + 2Ye^2)
                        * ((6.0 * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + (2.0 * (std::pow(ytau_val, 2.0)
                                    + std::pow(ymu_val, 2.0)
                                    + std::pow(ye_val, 2.0))))) // end trace
                        - (std::pow(yd_val, 2.0) // Tr(6ad^2 + 2ae^2)
                        * ((6.0 * (std::pow(ab_val, 2.0) + std::pow(as_val, 2.0)
                                    + std::pow(ad_val, 2.0)))
                            + (2.0 * (std::pow(atau_val, 2.0)
                                    + std::pow(amu_val, 2.0)
                                    + std::pow(ae_val, 2.0))))) // end trace
                        - (2.0 * ad_val * yd_val // Tr(6Yd*ad + 2Ye*ae)
                        * ((6.0 * ((yb_val * ab_val) + (ys_val * as_val)
                                    + (yd_val * ad_val)))
                            + (2.0 * ((ytau_val * atau_val)
                                    + (ymu_val * amu_val)
                                    + (ye_val * ae_val))))) // end trace
                        + ((2.0 / 5.0) * std::pow(g1_val, 2.0)
                        * ((4.0 * (mQ1_sq_val + mHu_sq_val + mU1_sq_val)
                            * std::pow(yu_val, 2.0))
                            + (4.0 * std::pow(au_val, 2.0))
                            - (8.0 * M1_val * au_val * yu_val)
                            + (8.0 * std::pow(M1_val, 2.0) * std::pow(yu_val, 2.0))
                            + (2.0 * (mQ1_sq_val + mHd_sq_val + mD1_sq_val)
                                * std::pow(yd_val, 2.0))
                            + (2.0 * std::pow(ad_val, 2.0))
                            - (4.0 * M1_val * ad_val * yd_val)
                            + (4.0 * std::pow(M1_val, 2.0)
                                * std::pow(yd_val, 2.0))))
                        + ((2.0 / 5.0) * std::pow(g1_val, 2.0) * Spr_val)
                        - ((128.0 / 3.0) * std::pow(g3_val, 4.0)
                        * std::pow(M3_val, 2.0))
                        + (32.0 * std::pow(g3_val, 2.0) * std::pow(g2_val, 2.0)
                        * (std::pow(M3_val, 2.0) + std::pow(M2_val, 2.0)
                            + (M2_val * M3_val)))
                        + ((32 / 45) * std::pow(g3_val, 2.0)
                        * std::pow(g1_val, 2.0)
                        * (std::pow(M3_val, 2.0) + std::pow(M1_val, 2.0)
                            + (M3_val * M1_val)))
                        + (33.0 * std::pow(g2_val, 4.0) * std::pow(M2_val, 2.0))
                        + ((2.0 / 5.0) * std::pow(g2_val, 2.0) * std::pow(g1_val, 2.0)
                        * (std::pow(M1_val, 2.0) + std::pow(M2_val, 2.0)
                            + (M1_val * M2_val)))
                        + ((199 / 75) * std::pow(g1_val, 4.0) * std::pow(M1_val, 2.0))
                        + ((16.0 / 3.0) * std::pow(g3_val, 2.0) * sigma3)
                        + (3.0 * std::pow(g2_val, 2.0) * sigma2)
                        + ((1 / 15) * std::pow(g1_val, 2.0) * sigma1));

        // Left leptons
    double dmL3_sq_dt_2l = (((-8.0)* (mL3_sq_val + mHd_sq_val + mE3_sq_val)
                        * std::pow(ytau_val, 4.0))
                        - (std::pow(ytau_val, 2.0)
                        * ((2.0 * mL3_sq_val) + (2.0 * mE3_sq_val)
                            + (4.0 * mHd_sq_val)) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                            + std::pow(ye_val, 2.0))) // end trace
                        - (std::pow(ytau_val, 2.0) // Tr(6(mQ^2 + mD^2)*Yd^2 + 2(mL^2 + mE^2)*Ye^2)
                        * ((6.0 * (((mQ3_sq_val + mD3_sq_val)
                                    * std::pow(yb_val, 2.0))
                                    + ((mQ2_sq_val + mD2_sq_val)
                                    * std::pow(ys_val, 2.0))
                                    + ((mQ1_sq_val + mD1_sq_val)
                                    * std::pow(yd_val, 2.0))))
                            + (2.0 * (((mL3_sq_val + mE3_sq_val)
                                    * std::pow(ytau_val, 2.0))
                                    + ((mL2_sq_val + mE2_sq_val)
                                    * std::pow(ymu_val, 2.0))
                                    + ((mL1_sq_val + mE1_sq_val)
                                    * std::pow(ye_val, 2.0)))) // end trace
                            ))
                        - (16.0 * std::pow(ytau_val, 2.0) * std::pow(atau_val, 2.0))
                        - (std::pow(atau_val, 2.0) // Tr(6Yd^2 + 2Ye^2)
                        * ((6.0 * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + (2.0 * (std::pow(ytau_val, 2.0)
                                    + std::pow(ymu_val, 2.0)
                                    + std::pow(ye_val, 2.0))))) // end trace
                        - (std::pow(ytau_val, 2.0) // Tr(6ad^2 + 2ae^2)
                        * ((6.0 * (std::pow(ab_val, 2.0) + std::pow(as_val, 2.0)
                                    + std::pow(ad_val, 2.0)))
                            + (2.0 * (std::pow(atau_val, 2.0)
                                    + std::pow(amu_val, 2.0)
                                    + std::pow(ae_val, 2.0))))) // end trace
                        - (2.0 * atau_val * ytau_val // Tr(6Yd*ad + 2Ye*ae)
                        * ((6.0 * ((yb_val * ab_val) + (ys_val * as_val)
                                    + (yd_val * ad_val)))
                            + (2.0 * ((ytau_val * atau_val)
                                    + (ymu_val * amu_val)
                                    + (ye_val * ae_val))))) // end trace
                        + ((6.0 / 5.0) * std::pow(g1_val, 2.0)
                        * ((2.0 * (mL3_sq_val + mHd_sq_val + mE3_sq_val)
                            * std::pow(ytau_val, 2.0))
                            + (2.0 * std::pow(atau_val, 2.0))
                            - (4.0 * M1_val * atau_val
                                * ytau_val)
                            + (4.0 * std::pow(M1_val, 2.0)
                                * std::pow(ytau_val, 2.0))))
                        - ((6.0 / 5.0) * std::pow(g1_val, 2.0) * Spr_val)
                        + (33.0 * std::pow(g2_val, 4.0) * std::pow(M2_val, 2.0))
                        + ((18.0 / 5.0) * std::pow(g2_val, 2.0)
                        * std::pow(g1_val, 2.0)
                        * (std::pow(M1_val, 2.0) + std::pow(M2_val, 2.0)
                            + (M1_val * M2_val)))
                        + ((621.0 / 25.0) * std::pow(g1_val, 4.0)
                        * std::pow(M1_val, 2.0))
                        + (3.0 * std::pow(g2_val, 2.0) * sigma2)
                        + (3.0 / 5.0 * std::pow(g1_val, 2.0) * sigma1));

    double dmL2_sq_dt_2l = (((-8.0)* (mL2_sq_val + mHd_sq_val + mE2_sq_val)
                        * std::pow(ymu_val, 4.0))
                        - (std::pow(ymu_val, 2.0)
                        * ((2.0 * mL2_sq_val) + (2.0 * mE2_sq_val)
                            + (4.0 * mHd_sq_val)) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                            + std::pow(ye_val, 2.0))) // end trace
                        - (std::pow(ymu_val, 2.0) // Tr(6(mQ^2 + mD^2)*Yd^2 + 2(mL^2 + mE^2)*Ye^2)
                        * ((6.0 * (((mQ3_sq_val + mD3_sq_val)
                                    * std::pow(yb_val, 2.0))
                                    + ((mQ2_sq_val + mD2_sq_val)
                                    * std::pow(ys_val, 2.0))
                                    + ((mQ1_sq_val + mD1_sq_val)
                                    * std::pow(yd_val, 2.0))))
                            + (2.0 * (((mL3_sq_val + mE3_sq_val)
                                    * std::pow(ytau_val, 2.0))
                                    + ((mL2_sq_val + mE2_sq_val)
                                    * std::pow(ymu_val, 2.0))
                                    + ((mL1_sq_val + mE1_sq_val)
                                    * std::pow(ye_val, 2.0)))) // end trace
                            ))
                        - (16.0 * std::pow(ymu_val, 2.0) * std::pow(amu_val, 2.0))
                        - (std::pow(amu_val, 2.0) // Tr(6Yd^2 + 2Ye^2)
                        * ((6.0 * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + (2.0 * (std::pow(ytau_val, 2.0)
                                    + std::pow(ymu_val, 2.0)
                                    + std::pow(ye_val, 2.0))))) // end trace
                        - (std::pow(ymu_val, 2.0) // Tr(6ad^2 + 2ae^2)
                        * ((6.0 * (std::pow(ab_val, 2.0) + std::pow(as_val, 2.0)
                                    + std::pow(ad_val, 2.0)))
                            + (2.0 * (std::pow(atau_val, 2.0)
                                    + std::pow(amu_val, 2.0)
                                    + std::pow(ae_val, 2.0))))) // end trace
                        - (2.0 * amu_val * ymu_val // Tr(6Yd*ad + 2Ye*ae)
                        * ((6.0 * ((yb_val * ab_val) + (ys_val * as_val)
                                    + (yd_val * ad_val)))
                            + (2.0 * ((ytau_val * atau_val)
                                    + (ymu_val * amu_val)
                                    + (ye_val * ae_val))))) // end trace
                        + ((6.0 / 5.0) * std::pow(g1_val, 2.0)
                        * ((2.0 * (mL2_sq_val + mHd_sq_val + mE2_sq_val)
                            * std::pow(ymu_val, 2.0))
                            + (2.0 * std::pow(amu_val, 2.0))
                            - (4.0 * M1_val * amu_val
                                * ymu_val)
                            + (4.0 * std::pow(M1_val, 2.0)
                                * std::pow(ymu_val, 2.0))))
                        - ((6.0 / 5.0) * std::pow(g1_val, 2.0) * Spr_val)
                        + (33.0 * std::pow(g2_val, 4.0) * std::pow(M2_val, 2.0))
                        + ((18.0 / 5.0) * std::pow(g2_val, 2.0)
                        * std::pow(g1_val, 2.0)
                        * (std::pow(M1_val, 2.0) + std::pow(M2_val, 2.0)
                            + (M1_val * M2_val)))
                        + ((621.0 / 25.0) * std::pow(g1_val, 4.0)
                        * std::pow(M1_val, 2.0))
                        + (3.0 * std::pow(g2_val, 2.0) * sigma2)
                        + (3.0 / 5.0 * std::pow(g1_val, 2.0) * sigma1));

    double dmL1_sq_dt_2l = (((-8.0)* (mL1_sq_val + mHd_sq_val + mE1_sq_val)
                        * std::pow(ye_val, 4.0))
                        - (std::pow(ye_val, 2.0)
                        * ((2.0 * mL1_sq_val) + (2.0 * mE1_sq_val)
                            + (4.0 * mHd_sq_val)) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                            + std::pow(ye_val, 2.0))) // end trace
                        - (std::pow(ye_val, 2.0) // Tr(6(mQ^2 + mD^2)*Yd^2 + 2(mL^2 + mE^2)*Ye^2)
                        * ((6.0 * (((mQ3_sq_val + mD3_sq_val)
                                    * std::pow(yb_val, 2.0))
                                    + ((mQ2_sq_val + mD2_sq_val)
                                    * std::pow(ys_val, 2.0))
                                    + ((mQ1_sq_val + mD1_sq_val)
                                    * std::pow(yd_val, 2.0))))
                            + (2.0 * (((mL3_sq_val + mE3_sq_val)
                                    * std::pow(ytau_val, 2.0))
                                    + ((mL2_sq_val + mE2_sq_val)
                                    * std::pow(ymu_val, 2.0))
                                    + ((mL1_sq_val + mE1_sq_val)
                                    * std::pow(ye_val, 2.0)))) // end trace
                            ))
                        - (16.0 * std::pow(ye_val, 2.0) * std::pow(ae_val, 2.0))
                        - (std::pow(ae_val, 2.0) // Tr(6Yd^2 + 2Ye^2)
                        * ((6.0 * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + (2.0 * (std::pow(ytau_val, 2.0)
                                    + std::pow(ymu_val, 2.0)
                                    + std::pow(ye_val, 2.0))))) // end trace
                        - (std::pow(ye_val, 2.0) // Tr(6ad^2 + 2ae^2)
                        * ((6.0 * (std::pow(ab_val, 2.0) + std::pow(as_val, 2.0)
                                    + std::pow(ad_val, 2.0)))
                            + (2.0 * (std::pow(atau_val, 2.0)
                                    + std::pow(amu_val, 2.0)
                                    + std::pow(ae_val, 2.0))))) // end trace
                        - (2.0 * ae_val * ye_val // Tr(6Yd*ad + 2Ye*ae)
                        * ((6.0 * ((yb_val * ab_val) + (ys_val * as_val)
                                    + (yd_val * ad_val)))
                            + (2.0 * ((ytau_val * atau_val)
                                    + (ymu_val * amu_val)
                                    + (ye_val * ae_val))))) // end trace
                        + ((6.0 / 5.0) * std::pow(g1_val, 2.0)
                        * ((2.0 * (mL1_sq_val + mHd_sq_val + mE1_sq_val)
                            * std::pow(ye_val, 2.0))
                            + (2.0 * std::pow(ae_val, 2.0))
                            - (4.0 * M1_val * ae_val
                                * ye_val)
                            + (4.0 * std::pow(M1_val, 2.0)
                                * std::pow(ye_val, 2.0))))
                        - ((6.0 / 5.0) * std::pow(g1_val, 2.0) * Spr_val)
                        + (33.0 * std::pow(g2_val, 4.0) * std::pow(M2_val, 2.0))
                        + ((18.0 / 5.0) * std::pow(g2_val, 2.0)
                        * std::pow(g1_val, 2.0)
                        * (std::pow(M1_val, 2.0) + std::pow(M2_val, 2.0)
                            + (M1_val * M2_val)))
                        + ((621.0 / 25.0) * std::pow(g1_val, 4.0)
                        * std::pow(M1_val, 2.0))
                        + (3.0 * std::pow(g2_val, 2.0) * sigma2)
                        + (3.0 / 5.0 * std::pow(g1_val, 2.0) * sigma1));

        // Right up-type squarks
    double dmU3_sq_dt_2l = (((-8.0)* (mQ3_sq_val + mHu_sq_val + mU3_sq_val)
                        * std::pow(yt_val, 4.0))
                        - (4.0 * (mU3_sq_val + mHu_sq_val + mHd_sq_val
                            + (2.0 * mQ3_sq_val) + mD3_sq_val)
                        * std::pow(yb_val, 2.0) * std::pow(yt_val, 2.0))
                        - (std::pow(yt_val, 2.0)
                        * ((2.0 * mQ3_sq_val) + (2.0 * mU3_sq_val)
                            + (4.0 * mHu_sq_val)) // Tr(6Yu^2)
                        * 6.0 * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                                + std::pow(yu_val, 2.0))) // end trace
                        - (12.0 * std::pow(yt_val, 2.0) // Tr((mQ^2 + mU^2)*Yu^2)
                        * (((mQ3_sq_val + mU3_sq_val)
                            * std::pow(yt_val, 2.0))
                            + ((mQ2_sq_val + mU2_sq_val)
                                * std::pow(yc_val, 2.0))
                            + ((mQ1_sq_val + mU1_sq_val)
                                * std::pow(yu_val, 2.0)))) // end trace
                        - (16.0 * std::pow(yt_val, 2.0) * std::pow(at_val, 2.0))
                        - (16.0 * at_val * ab_val * yb_val * yt_val)
                        - (12.0 * ((std::pow(at_val, 2.0) // Tr(Yu^2)
                                * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                                    + std::pow(yu_val, 2.0))) // end trace
                                + (std::pow(yt_val, 2.0)  // Tr(au^2)
                                    * (std::pow(at_val, 2.0)
                                    + std::pow(ac_val, 2.0)
                                    + std::pow(au_val, 2.0))) // end trace
                                + (at_val * yt_val * 2 // Tr(Yu*au)
                                    * ((yt_val * at_val) + (yc_val * ac_val)
                                    + (yu_val * au_val))))) // end trace
                        + (((6.0 * std::pow(g2_val, 2.0))
                        - ((2.0 / 5.0) * std::pow(g1_val, 2.0)))
                        * ((2.0 * (mQ3_sq_val + mHu_sq_val + mU3_sq_val)
                            * std::pow(yt_val, 2.0))
                            + (2.0 * std::pow(at_val, 2.0))))
                        + (12.0 * std::pow(g2_val, 2.0)
                        * 2.0 * ((std::pow(M2_val, 2.0) * std::pow(yt_val, 2.0))
                                - (M2_val * at_val * yt_val)))
                        - ((4.0 / 5.0) * std::pow(g1_val, 2.0)
                        * 2.0 * ((std::pow(M1_val, 2.0) * std::pow(yt_val, 2.0))
                                - (M1_val * at_val * yt_val)))
                        - ((8.0 / 5.0) * std::pow(g1_val, 2.0) * Spr_val)
                        - ((128.0 / 3.0) * std::pow(g3_val, 4.0)
                        * std::pow(M3_val, 2.0))
                        + ((512.0 / 45.0) * std::pow(g3_val, 2.0)
                        * std::pow(g1_val, 2.0)
                        * (std::pow(M3_val, 2.0) + std::pow(M1_val, 2.0)
                            + (M3_val * M1_val)))
                        + ((3424.0 / 75.0) * std::pow(g1_val, 4.0)
                        * std::pow(M1_val, 2.0))
                        + ((16.0 / 3.0) * std::pow(g3_val, 2.0) * sigma3)
                        + ((16.0 / 15.0) * std::pow(g1_val, 2.0) * sigma1));

    double dmU2_sq_dt_2l = (((-8.0)* (mQ2_sq_val + mHu_sq_val + mU2_sq_val)
                        * std::pow(yc_val, 4.0))
                        - (4.0 * (mU2_sq_val + mHu_sq_val + mHd_sq_val
                            + (2.0 * mQ2_sq_val)
                            + mD2_sq_val)
                        * std::pow(ys_val, 2.0) * std::pow(yc_val, 2.0))
                        - (std::pow(yc_val, 2.0)
                        * ((2.0 * mQ2_sq_val) + (2.0 * mU2_sq_val)
                            + (4.0 * mHu_sq_val)) // Tr(6Yu^2)
                        * 6.0 * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                                + std::pow(yu_val, 2.0))) // end trace
                        - (12.0 * std::pow(yc_val, 2.0) // Tr((mQ^2 + mU^2)*Yu^2)
                        * (((mQ3_sq_val + mU3_sq_val)
                            * std::pow(yt_val, 2.0))
                            + ((mQ2_sq_val + mU2_sq_val)
                                * std::pow(yc_val, 2.0))
                            + ((mQ1_sq_val + mU1_sq_val)
                                * std::pow(yu_val, 2.0)))) // end trace
                        - (16.0 * std::pow(yc_val, 2.0) * std::pow(ac_val, 2.0))
                        - (16.0 * ac_val * as_val * ys_val * yc_val)
                        - (12.0 * ((std::pow(ac_val, 2.0) // Tr(Yu^2)
                                * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                                    + std::pow(yu_val, 2.0))) // end trace
                                + (std::pow(yc_val, 2.0)  // Tr(au^2)
                                    * (std::pow(at_val, 2.0)
                                    + std::pow(ac_val, 2.0)
                                    + std::pow(au_val, 2.0))) // end trace
                                + (ac_val * yc_val * 2 // Tr(Yu*au)
                                    * ((yt_val * at_val) + (yc_val * ac_val)
                                    + (yu_val * au_val))))) // end trace
                        + (((6.0 * std::pow(g2_val, 2.0))
                        - ((2.0 / 5.0) * std::pow(g1_val, 2.0)))
                        * ((2.0 * (mQ2_sq_val + mHu_sq_val + mU2_sq_val)
                            * std::pow(yc_val, 2.0))
                            + (2.0 * std::pow(ac_val, 2.0))))
                        + (12.0 * std::pow(g2_val, 2.0)
                        * 2.0 * ((std::pow(M2_val, 2.0) * std::pow(yc_val, 2.0))
                                - (M2_val * ac_val * yc_val)))
                        - ((4.0 / 5.0) * std::pow(g1_val, 2.0)
                        * 2.0 * ((std::pow(M1_val, 2.0) * std::pow(yc_val, 2.0))
                                - (M1_val * ac_val * yc_val)))
                        - ((8.0 / 5.0) * std::pow(g1_val, 2.0) * Spr_val)
                        - ((128.0 / 3.0) * std::pow(g3_val, 4.0)
                        * std::pow(M3_val, 2.0))
                        + ((512.0 / 45.0) * std::pow(g3_val, 2.0)
                        * std::pow(g1_val, 2.0)
                        * (std::pow(M3_val, 2.0) + std::pow(M1_val, 2.0)
                            + (M3_val * M1_val)))
                        + ((3424.0 / 75.0) * std::pow(g1_val, 4.0)
                        * std::pow(M1_val, 2.0))
                        + ((16.0 / 3.0) * std::pow(g3_val, 2.0) * sigma3)
                        + ((16.0 / 15.0) * std::pow(g1_val, 2.0) * sigma1));

    double dmU1_sq_dt_2l = (((-8.0)* (mQ1_sq_val + mHu_sq_val + mU1_sq_val)
                        * std::pow(yu_val, 4.0))
                        - (4.0 * (mU1_sq_val + mHu_sq_val + mHd_sq_val
                            + (2.0 * mQ1_sq_val)
                            + mD1_sq_val)
                        * std::pow(yd_val, 2.0) * std::pow(yu_val, 2.0))
                        - (std::pow(yu_val, 2.0)
                        * ((2.0 * mQ1_sq_val) + (2.0 * mU1_sq_val)
                            + (4.0 * mHu_sq_val)) // Tr(6Yu^2)
                        * 6.0 * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                                + std::pow(yu_val, 2.0))) // end trace
                        - (12.0 * std::pow(yu_val, 2.0) // Tr((mQ^2 + mU^2)*Yu^2)
                        * (((mQ3_sq_val + mU3_sq_val)
                            * std::pow(yt_val, 2.0))
                            + ((mQ2_sq_val + mU2_sq_val)
                                * std::pow(yc_val, 2.0))
                            + ((mQ1_sq_val + mU1_sq_val)
                                * std::pow(yu_val, 2.0)))) // end trace
                        - (16.0 * std::pow(yu_val, 2.0) * std::pow(au_val, 2.0))
                        - (16.0 * au_val * ad_val * yd_val * yu_val)
                        - (12.0 * ((std::pow(au_val, 2.0) // Tr(Yu^2)
                                * (std::pow(yt_val, 2.0) + std::pow(yc_val, 2.0)
                                    + std::pow(yu_val, 2.0))) // end trace
                                + (std::pow(yu_val, 2.0)  // Tr(au^2)
                                    * (std::pow(at_val, 2.0)
                                    + std::pow(ac_val, 2.0)
                                    + std::pow(au_val, 2.0))) // end trace
                                + (au_val * yu_val * 2 // Tr(Yu*au)
                                    * ((yt_val * at_val) + (yc_val * ac_val)
                                    + (yu_val * au_val))))) // end trace
                        + (((6.0 * std::pow(g2_val, 2.0))
                        - ((2.0 / 5.0) * std::pow(g1_val, 2.0)))
                        * ((2.0 * (mQ1_sq_val + mHu_sq_val + mU1_sq_val)
                            * std::pow(yu_val, 2.0))
                            + (2.0 * std::pow(au_val, 2.0))))
                        + (12.0 * std::pow(g2_val, 2.0)
                        * 2.0 * ((std::pow(M2_val, 2.0) * std::pow(yu_val, 2.0))
                                - (M2_val * au_val * yu_val)))
                        - ((4.0 / 5.0) * std::pow(g1_val, 2.0)
                        * 2.0 * ((std::pow(M1_val, 2.0) * std::pow(yu_val, 2.0))
                                - (M1_val * au_val * yu_val)))
                        - ((8.0 / 5.0) * std::pow(g1_val, 2.0) * Spr_val)
                        - ((128.0 / 3.0) * std::pow(g3_val, 4.0)
                        * std::pow(M3_val, 2.0))
                        + ((512.0 / 45.0) * std::pow(g3_val, 2.0)
                        * std::pow(g1_val, 2.0)
                        * (std::pow(M3_val, 2.0) + std::pow(M1_val, 2.0)
                            + (M3_val * M1_val)))
                        + ((3424.0 / 75.0) * std::pow(g1_val, 4.0)
                        * std::pow(M1_val, 2.0))
                        + ((16.0 / 3.0) * std::pow(g3_val, 2.0) * sigma3)
                        + ((16.0 / 15.0) * std::pow(g1_val, 2.0) * sigma1));

        // Right down-type squarks
    double dmD3_sq_dt_2l = (((-8.0)* (mQ3_sq_val + mHd_sq_val + mD3_sq_val)
                        * std::pow(yb_val, 4.0))
                        - (4.0 * (mU3_sq_val + mHu_sq_val + mHd_sq_val
                            + (2.0 * mQ3_sq_val)
                            + mD3_sq_val) * std::pow(yb_val, 2.0)
                        * std::pow(yt_val, 2.0))
                        - (std::pow(yb_val, 2.0)
                        * (2.0 * (mD3_sq_val + mQ3_sq_val
                                + (2.0 * mHd_sq_val))) // Tr(6Yd^2 + 2Ye^2)
                        * ((6.0 * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + (2.0 * (std::pow(ytau_val, 2.0)
                                    + std::pow(ymu_val, 2.0)
                                    + std::pow(ye_val, 2.0))))) // end trace
                        - (4.0 * std::pow(yb_val, 2.0)  // Tr(3(mQ^2 + mD^2) * Yd^2 + (mL^2 + mE^2) * Ye^2)
                        * ((3.0 * (((mQ3_sq_val + mD3_sq_val)
                                    * std::pow(yb_val, 2.0))
                                    + ((mQ2_sq_val + mD2_sq_val)
                                    * std::pow(ys_val, 2.0))
                                    + ((mQ1_sq_val + mD1_sq_val)
                                    * std::pow(yd_val, 2.0))))
                            + (((mL3_sq_val + mE3_sq_val)
                                * std::pow(ytau_val, 2.0))
                                + ((mL2_sq_val + mE2_sq_val)
                                    * std::pow(ymu_val, 2.0))
                                + ((mL1_sq_val + mE1_sq_val)
                                    * std::pow(ye_val, 2.0))) // end trace
                            ))
                        - (16.0 * std::pow(yb_val, 2.0) * std::pow(ab_val, 2.0))
                        - (16.0 * at_val * ab_val * yb_val * yt_val)
                        - (4.0 * std::pow(ab_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                            + std::pow(ye_val, 2.0))) // end trace
                        - (4.0 * std::pow(yb_val, 2.0)  // Tr(3ad^2 + ae^2)
                        * ((3.0 * (std::pow(ab_val, 2.0) + std::pow(as_val, 2.0)
                                    + std::pow(ad_val, 2.0)))
                            + std::pow(atau_val, 2.0) + std::pow(amu_val, 2.0)
                            + std::pow(ae_val, 2.0))) // end trace
                        - (8.0 * ab_val * yb_val  // Tr(3Yd * ad + Ye * ae)
                        * ((3.0 * ((yb_val * ab_val) + (ys_val * as_val)
                                    + (yd_val * ad_val)))
                            + (ytau_val * atau_val) + (ymu_val * amu_val)
                            + (ye_val * ae_val))) // end trace
                        + (((6.0 * std::pow(g2_val, 2.0))
                        + ((2.0 / 5.0) * std::pow(g1_val, 2.0)))
                        * ((2.0 * (mQ3_sq_val + mHd_sq_val + mD3_sq_val)
                            * std::pow(yb_val, 2.0))
                            + (2.0 * std::pow(ab_val, 2.0))))
                        + (12.0 * std::pow(g2_val, 2.0)
                        * 2.0 * ((std::pow(M2_val, 2.0) * std::pow(yb_val, 2.0))
                                - (M2_val * ab_val * yb_val)))
                        + ((4.0 / 5.0) * std::pow(g1_val, 2.0)
                        * 2.0 * ((std::pow(M1_val, 2.0) * std::pow(yb_val, 2.0))
                                - (M1_val * ab_val * yb_val)))
                        + ((4.0 / 5.0) * std::pow(g1_val, 2.0) * Spr_val)
                        - ((128.0 / 3.0) * std::pow(g3_val, 4.0)
                        * std::pow(M3_val, 2.0))
                        + ((128.0 / 45.0) * std::pow(g3_val, 2.0)
                        * std::pow(g1_val, 2.0)
                        * (std::pow(M3_val, 2.0) + std::pow(M1_val, 2.0)
                            + (M3_val * M1_val)))
                        + ((808.0 / 75.0) * std::pow(g1_val, 4.0)
                        * std::pow(M1_val, 2.0))
                        + ((16.0 / 3.0) * std::pow(g3_val, 2.0) * sigma3)
                        + ((4.0 / 15.0) * std::pow(g1_val, 2.0) * sigma1));

    double dmD2_sq_dt_2l = (((-8.0)* (mQ2_sq_val + mHd_sq_val + mD2_sq_val)
                        * std::pow(ys_val, 4.0))
                        - (4.0 * (mU2_sq_val + mHu_sq_val + mHd_sq_val
                            + (2.0 * mQ2_sq_val)
                            + mD2_sq_val) * std::pow(ys_val, 2.0)
                        * std::pow(yc_val, 2.0))
                        - (std::pow(ys_val, 2.0)
                        * (2.0 * (mD2_sq_val + mQ2_sq_val
                                + (2.0 * mHd_sq_val))) // Tr(6Yd^2 + 2Ye^2)
                        * ((6.0 * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + (2.0 * (std::pow(ytau_val, 2.0)
                                    + std::pow(ymu_val, 2.0)
                                    + std::pow(ye_val, 2.0))))) // end trace
                        - (4.0 * std::pow(ys_val, 2.0)  // Tr(3(mQ^2 + mD^2) * Yd^2 + (mL^2 + mE^2) * Ye^2)
                        * ((3.0 * (((mQ3_sq_val + mD3_sq_val)
                                    * std::pow(yb_val, 2.0))
                                    + ((mQ2_sq_val + mD2_sq_val)
                                    * std::pow(ys_val, 2.0))
                                    + ((mQ1_sq_val + mD1_sq_val)
                                    * std::pow(yd_val, 2.0))))
                            + (((mL3_sq_val + mE3_sq_val)
                                * std::pow(ytau_val, 2.0))
                                + ((mL2_sq_val + mE2_sq_val)
                                    * std::pow(ymu_val, 2.0))
                                + ((mL1_sq_val + mE1_sq_val)
                                    * std::pow(ye_val, 2.0))) // end trace
                            ))
                        - (16.0 * std::pow(ys_val, 2.0) * std::pow(as_val, 2.0))
                        - (16.0 * ac_val * as_val * ys_val * yc_val)
                        - (4.0 * std::pow(as_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                            + std::pow(ye_val, 2.0))) // end trace
                        - (4.0 * std::pow(ys_val, 2.0)  // Tr(3ad^2 + ae^2)
                        * ((3.0 * (std::pow(ab_val, 2.0) + std::pow(as_val, 2.0)
                                    + std::pow(ad_val, 2.0)))
                            + std::pow(atau_val, 2.0) + std::pow(amu_val, 2.0)
                            + std::pow(ae_val, 2.0))) // end trace
                        - (8.0 * as_val * ys_val  // Tr(3Yd * ad + Ye * ae)
                        * ((3.0 * ((yb_val * ab_val) + (ys_val * as_val)
                                    + (yd_val * ad_val)))
                            + (ytau_val * atau_val) + (ymu_val * amu_val)
                            + (ye_val * ae_val))) // end trace
                        + (((6.0 * std::pow(g2_val, 2.0))
                        + ((2.0 / 5.0) * std::pow(g1_val, 2.0)))
                        * ((2.0 * (mQ2_sq_val + mHd_sq_val + mD2_sq_val)
                            * std::pow(ys_val, 2.0))
                            + (2.0 * std::pow(as_val, 2.0))))
                        + (12.0 * std::pow(g2_val, 2.0)
                        * 2.0 * ((std::pow(M2_val, 2.0) * std::pow(ys_val, 2.0))
                                - (M2_val * as_val * ys_val)))
                        + ((4.0 / 5.0) * std::pow(g1_val, 2.0)
                        * 2.0 * ((std::pow(M1_val, 2.0) * std::pow(ys_val, 2.0))
                                - (M1_val * as_val * ys_val)))
                        + ((4.0 / 5.0) * std::pow(g1_val, 2.0) * Spr_val)
                        - ((128.0 / 3.0) * std::pow(g3_val, 4.0)
                        * std::pow(M3_val, 2.0))
                        + ((128.0 / 45.0) * std::pow(g3_val, 2.0)
                        * std::pow(g1_val, 2.0)
                        * (std::pow(M3_val, 2.0) + std::pow(M1_val, 2.0)
                            + (M3_val * M1_val)))
                        + ((808.0 / 75.0) * std::pow(g1_val, 4.0)
                        * std::pow(M1_val, 2.0))
                        + ((16.0 / 3.0) * std::pow(g3_val, 2.0) * sigma3)
                        + ((4.0 / 15.0) * std::pow(g1_val, 2.0) * sigma1));

    double dmD1_sq_dt_2l = (((-8.0)* (mQ1_sq_val + mHd_sq_val + mD1_sq_val)
                        * std::pow(yd_val, 4.0))
                        - (4.0 * (mU1_sq_val + mHu_sq_val + mHd_sq_val
                            + (2.0 * mQ1_sq_val)
                            + mD1_sq_val) * std::pow(yd_val, 2.0)
                        * std::pow(yu_val, 2.0))
                        - (std::pow(yd_val, 2.0)
                        * (2.0 * (mD1_sq_val + mQ1_sq_val
                                + (2.0 * mHd_sq_val))) // Tr(6Yd^2 + 2Ye^2)
                        * ((6.0 * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + (2.0 * (std::pow(ytau_val, 2.0)
                                    + std::pow(ymu_val, 2.0)
                                    + std::pow(ye_val, 2.0))))) // end trace
                        - (4.0 * std::pow(yd_val, 2.0)  // Tr(3(mQ^2 + mD^2) * Yd^2 + (mL^2 + mE^2) * Ye^2)
                        * ((3.0 * (((mQ3_sq_val + mD3_sq_val)
                                    * std::pow(yb_val, 2.0))
                                    + ((mQ2_sq_val + mD2_sq_val)
                                    * std::pow(ys_val, 2.0))
                                    + ((mQ1_sq_val + mD1_sq_val)
                                    * std::pow(yd_val, 2.0))))
                            + (((mL3_sq_val + mE3_sq_val)
                                * std::pow(ytau_val, 2.0))
                                + ((mL2_sq_val + mE2_sq_val)
                                    * std::pow(ymu_val, 2.0))
                                + ((mL1_sq_val + mE1_sq_val)
                                    * std::pow(ye_val, 2.0))) // end trace
                            ))
                        - (16.0 * std::pow(yd_val, 2.0) * std::pow(ad_val, 2.0))
                        - (16.0 * au_val * ad_val * yd_val * yu_val)
                        - (4.0 * std::pow(ad_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                            + std::pow(ye_val, 2.0))) // end trace
                        - (4.0 * std::pow(yd_val, 2.0)  // Tr(3ad^2 + ae^2)
                        * ((3.0 * (std::pow(ab_val, 2.0) + std::pow(as_val, 2.0)
                                    + std::pow(ad_val, 2.0)))
                            + std::pow(atau_val, 2.0) + std::pow(amu_val, 2.0)
                            + std::pow(ae_val, 2.0))) // end trace
                        - (8.0 * ad_val * yd_val  // Tr(3Yd * ad + Ye * ae)
                        * ((3.0 * ((yb_val * ab_val) + (ys_val * as_val)
                                    + (yd_val * ad_val)))
                            + (ytau_val * atau_val) + (ymu_val * amu_val)
                            + (ye_val * ae_val))) // end trace
                        + (((6.0 * std::pow(g2_val, 2.0))
                        + ((2.0 / 5.0) * std::pow(g1_val, 2.0)))
                        * ((2.0 * (mQ1_sq_val + mHd_sq_val + mD1_sq_val)
                            * std::pow(yd_val, 2.0))
                            + (2.0 * std::pow(ad_val, 2.0))))
                        + (12.0 * std::pow(g2_val, 2.0)
                        * 2.0 * ((std::pow(M2_val, 2.0) * std::pow(yd_val, 2.0))
                                - (M2_val * ad_val * yd_val)))
                        + ((4.0 / 5.0) * std::pow(g1_val, 2.0)
                        * 2.0 * ((std::pow(M1_val, 2.0) * std::pow(yd_val, 2.0))
                                - (M1_val * ad_val * yd_val)))
                        + ((4.0 / 5.0) * std::pow(g1_val, 2.0) * Spr_val)
                        - ((128.0 / 3.0) * std::pow(g3_val, 4.0)
                        * std::pow(M3_val, 2.0))
                        + ((128.0 / 45.0) * std::pow(g3_val, 2.0)
                        * std::pow(g1_val, 2.0)
                        * (std::pow(M3_val, 2.0) + std::pow(M1_val, 2.0)
                            + (M3_val * M1_val)))
                        + ((808.0 / 75.0) * std::pow(g1_val, 4.0)
                        * std::pow(M1_val, 2.0))
                        + ((16.0 / 3.0) * std::pow(g3_val, 2.0) * sigma3)
                        + ((4.0 / 15.0) * std::pow(g1_val, 2.0) * sigma1));

        // Right leptons
    double dmE3_sq_dt_2l = (((-8.0)* (mL3_sq_val + mHd_sq_val + mE3_sq_val)
                        * std::pow(ytau_val, 4.0))
        - (std::pow(ytau_val, 2.0)
            * ((2.0 * mL3_sq_val) + (2.0 * mE3_sq_val)
            + (4.0 * mHd_sq_val)) // Tr(6Yd^2 + 2Ye^2)
            * ((6.0 * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                    + std::pow(yd_val, 2.0)))
            + (2.0 * (std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                        + std::pow(ye_val, 2.0))))) // end trace
        - (4.0 * std::pow(ytau_val, 2.0)  // Tr(3(mQ^2 + mD^2) * Yd^2 + (mL^2 + mE^2) * Ye^2)
            * ((3.0 * (((mQ3_sq_val + mD3_sq_val) * std::pow(yb_val, 2.0))
                    + ((mQ2_sq_val + mD2_sq_val) * std::pow(ys_val, 2.0))
                    + ((mQ1_sq_val + mD1_sq_val) * std::pow(yd_val, 2.0))))
            + ((((mL3_sq_val + mE3_sq_val) * std::pow(ytau_val, 2.0))
                    + ((mL2_sq_val + mE2_sq_val) * std::pow(ymu_val, 2.0))
                    + ((mL1_sq_val + mE1_sq_val) * std::pow(ye_val, 2.0)))) // end trace
            ))
        - (16.0 * std::pow(ytau_val, 2.0) * std::pow(atau_val, 2.0))
        - (4.0 * std::pow(atau_val, 2.0) // Tr(3Yd^2 + Ye^2)
            * ((3.0 * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                    + std::pow(yd_val, 2.0)))
            + ((std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                    + std::pow(ye_val, 2.0))))) // end trace
        - (4.0 * std::pow(ytau_val, 2.0)  // Tr(3ad^2 + ae^2)
            * ((3.0 * (std::pow(ab_val, 2.0) + std::pow(as_val, 2.0)
                    + std::pow(ad_val, 2.0)))
            + ((std::pow(atau_val, 2.0) + std::pow(amu_val, 2.0)
                    + std::pow(ae_val, 2.0))))) // end trace
        - (8.0 * atau_val * ytau_val  // Tr(3Yd * ad + Ye * ae)
            * ((3.0 * ((yb_val * ab_val) + (ys_val * as_val)
                        + (yd_val * ad_val)))
            + (((ytau_val * atau_val) + (ymu_val * amu_val)
                    + (ye_val * ae_val))))) // end trace
        + (((6.0 * std::pow(g2_val, 2.0)) - (6.0 / 5.0) * std::pow(g1_val, 2.0))
            * ((2.0 * (mL3_sq_val + mHd_sq_val + mE3_sq_val)
                * std::pow(ytau_val, 2.0))
            + (2.0 * std::pow(atau_val, 2.0))))
        + (12.0 * std::pow(g2_val, 2.0) * 2
            * ((std::pow(M2_val, 2.0) * std::pow(ytau_val, 2.0))
            - (M2_val * atau_val * ytau_val)))
        - ((12.0 / 5.0) * std::pow(g1_val, 2.0) * 2
            * ((std::pow(M1_val, 2.0) * std::pow(ytau_val, 2.0))
            - (M1_val * atau_val * ytau_val)))
        + ((12.0 / 5.0) * std::pow(g1_val, 2.0) * Spr_val)
        + ((2808.0 / 25.0) * std::pow(g1_val, 4.0) * std::pow(M1_val, 2.0))
        + ((12.0 / 5.0) * std::pow(g1_val, 2.0) * sigma1));

    double dmE2_sq_dt_2l = (((-8.0)* (mL2_sq_val + mHd_sq_val + mE2_sq_val)
                        * std::pow(ymu_val, 4.0))
                        - (std::pow(ymu_val, 2.0)
                        * ((2.0 * mL2_sq_val) + (2.0 * mE2_sq_val)
                            + (4.0 * mHd_sq_val)) // Tr(6Yd^2 + 2Ye^2)
                        * ((6.0 * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + (2.0 * (std::pow(ytau_val, 2.0)
                                    + std::pow(ymu_val, 2.0)
                                    + std::pow(ye_val, 2.0))))) // end trace
                        - (4.0 * std::pow(ymu_val, 2.0)  // Tr(3(mQ^2 + mD^2) * Yd^2 + (mL^2 + mE^2) * Ye^2)
                        * ((3.0 * (((mQ3_sq_val + mD3_sq_val)
                                    * std::pow(yb_val, 2.0))
                                    + ((mQ2_sq_val + mD2_sq_val)
                                    * std::pow(ys_val, 2.0))
                                    + ((mQ1_sq_val + mD1_sq_val)
                                    * std::pow(yd_val, 2.0))))
                            + ((((mL3_sq_val + mE3_sq_val)
                                * std::pow(ytau_val, 2.0))
                                + ((mL2_sq_val + mE2_sq_val)
                                    * std::pow(ymu_val, 2.0))
                                + ((mL1_sq_val + mE1_sq_val)
                                    * std::pow(ye_val, 2.0)))) // end trace
                            ))
                        - (16.0 * std::pow(ymu_val, 2.0) * std::pow(amu_val, 2.0))
                        - (4.0 * std::pow(amu_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + ((std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                                + std::pow(ye_val, 2.0))))) // end trace
                        - (4.0 * std::pow(ymu_val, 2.0)  // Tr(3ad^2 + ae^2)
                        * ((3.0 * (std::pow(ab_val, 2.0) + std::pow(as_val, 2.0)
                                    + std::pow(ad_val, 2.0)))
                            + ((std::pow(atau_val, 2.0) + std::pow(amu_val, 2.0)
                                + std::pow(ae_val, 2.0))))) // end trace
                        - (8.0 * amu_val * ymu_val  // Tr(3Yd * ad + Ye * ae)
                        * ((3.0 * ((yb_val * ab_val) + (ys_val * as_val)
                                    + (yd_val * ad_val)))
                            + (((ytau_val * atau_val) + (ymu_val * amu_val)
                                + (ye_val * ae_val))))) // end trace
                        + (((6.0 * std::pow(g2_val, 2.0))
                        - (6.0 / 5.0) * std::pow(g1_val, 2.0))
                        * ((2.0 * (mL2_sq_val + mHd_sq_val + mE2_sq_val)
                            * std::pow(ymu_val, 2.0))
                            + (2.0 * std::pow(amu_val, 2.0))))
                        + (12.0 * std::pow(g2_val, 2.0) * 2
                        * ((std::pow(M2_val, 2.0) * std::pow(ymu_val, 2.0))
                            - (M2_val * amu_val * ymu_val)))
                        - ((12.0 / 5.0) * std::pow(g1_val, 2.0) * 2
                        * ((std::pow(M1_val, 2.0) * std::pow(ymu_val, 2.0))
                            - (M1_val * amu_val * ymu_val)))
                        + ((12.0 / 5.0) * std::pow(g1_val, 2.0) * Spr_val)
                        + ((2808.0 / 25.0) * std::pow(g1_val, 4.0)
                        * std::pow(M1_val, 2.0))
                        + ((12.0 / 5.0) * std::pow(g1_val, 2.0) * sigma1));

    double dmE1_sq_dt_2l = (((-8.0)* (mL1_sq_val + mHd_sq_val + mE1_sq_val)
                        * std::pow(ye_val, 4.0))
                        - (std::pow(ye_val, 2.0)
                        * ((2.0 * mL1_sq_val) + (2.0 * mE1_sq_val)
                            + (4.0 * mHd_sq_val)) // Tr(6Yd^2 + 2Ye^2)
                        * ((6.0 * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + (2.0 * (std::pow(ytau_val, 2.0)
                                    + std::pow(ymu_val, 2.0)
                                    + std::pow(ye_val, 2.0))))) // end trace
                        - (4.0 * std::pow(ye_val, 2.0)  // Tr(3(mQ^2 + mD^2) * Yd^2 + (mL^2 + mE^2) * Ye^2)
                        * ((3.0 * (((mQ3_sq_val + mD3_sq_val)
                                    * std::pow(yb_val, 2.0))
                                    + ((mQ2_sq_val + mD2_sq_val)
                                    * std::pow(ys_val, 2.0))
                                    + ((mQ1_sq_val + mD1_sq_val)
                                    * std::pow(yd_val, 2.0))))
                            + ((((mL3_sq_val + mE3_sq_val)
                                * std::pow(ytau_val, 2.0))
                                + ((mL2_sq_val + mE2_sq_val)
                                    * std::pow(ymu_val, 2.0))
                                + ((mL1_sq_val + mE1_sq_val)
                                    * std::pow(ye_val, 2.0)))) // end trace
                            ))
                        - (16.0 * std::pow(ye_val, 2.0) * std::pow(ae_val, 2.0))
                        - (4.0 * std::pow(ae_val, 2.0) // Tr(3Yd^2 + Ye^2)
                        * ((3.0 * (std::pow(yb_val, 2.0) + std::pow(ys_val, 2.0)
                                    + std::pow(yd_val, 2.0)))
                            + ((std::pow(ytau_val, 2.0) + std::pow(ymu_val, 2.0)
                                + std::pow(ye_val, 2.0))))) // end trace
                        - (4.0 * std::pow(ye_val, 2.0)  // Tr(3ad^2 + ae^2)
                        * ((3.0 * (std::pow(ab_val, 2.0) + std::pow(as_val, 2.0)
                                    + std::pow(ad_val, 2.0)))
                            + ((std::pow(atau_val, 2.0) + std::pow(amu_val, 2.0)
                                + std::pow(ae_val, 2.0))))) // end trace
                        - (8.0 * ae_val * ye_val  // Tr(3Yd * ad + Ye * ae)
                        * ((3.0 * ((yb_val * ab_val) + (ys_val * as_val)
                                    + (yd_val * ad_val)))
                            + (((ytau_val * atau_val) + (ymu_val * amu_val)
                                + (ye_val * ae_val))))) // end trace
                        + (((6.0 * std::pow(g2_val, 2.0))
                        - (6.0 / 5.0) * std::pow(g1_val, 2.0))
                        * ((2.0 * (mL1_sq_val + mHd_sq_val + mE1_sq_val)
                            * std::pow(ye_val, 2.0))
                            + (2.0 * std::pow(ae_val, 2.0))))
                        + (12.0 * std::pow(g2_val, 2.0) * 2
                        * ((std::pow(M2_val, 2.0) * std::pow(ye_val, 2.0))
                            - (M2_val * ae_val * ye_val)))
                        - ((12.0 / 5.0) * std::pow(g1_val, 2.0) * 2
                        * ((std::pow(M1_val, 2.0) * std::pow(ye_val, 2.0))
                            - (M1_val * ae_val * ye_val)))
                        + ((12.0 / 5.0) * std::pow(g1_val, 2.0) * Spr_val)
                        + ((2808.0 / 25.0) * std::pow(g1_val, 4.0)
                        * std::pow(M1_val, 2.0))
                        + ((12.0 / 5.0) * std::pow(g1_val, 2.0) * sigma1));

    // Total scalar squared mass beta functions
    double dmHu_sq_dt = ((loop_fac * dmHu_sq_dt_1l) + (loop_fac_sq * dmHu_sq_dt_2l));
    double dmHd_sq_dt = ((loop_fac * dmHd_sq_dt_1l) + (loop_fac_sq * dmHd_sq_dt_2l));
    double dmQ3_sq_dt = ((loop_fac * dmQ3_sq_dt_1l) + (loop_fac_sq * dmQ3_sq_dt_2l));
    double dmQ2_sq_dt = ((loop_fac * dmQ2_sq_dt_1l) + (loop_fac_sq * dmQ2_sq_dt_2l));
    double dmQ1_sq_dt = ((loop_fac * dmQ1_sq_dt_1l) + (loop_fac_sq * dmQ1_sq_dt_2l));
    double dmL3_sq_dt = ((loop_fac * dmL3_sq_dt_1l) + (loop_fac_sq * dmL3_sq_dt_2l));
    double dmL2_sq_dt = ((loop_fac * dmL2_sq_dt_1l) + (loop_fac_sq * dmL2_sq_dt_2l));
    double dmL1_sq_dt = ((loop_fac * dmL1_sq_dt_1l) + (loop_fac_sq * dmL1_sq_dt_2l));
    double dmU3_sq_dt = ((loop_fac * dmU3_sq_dt_1l) + (loop_fac_sq * dmU3_sq_dt_2l));
    double dmU2_sq_dt = ((loop_fac * dmU2_sq_dt_1l) + (loop_fac_sq * dmU2_sq_dt_2l));
    double dmU1_sq_dt = ((loop_fac * dmU1_sq_dt_1l) + (loop_fac_sq * dmU1_sq_dt_2l));
    double dmD3_sq_dt = ((loop_fac * dmD3_sq_dt_1l) + (loop_fac_sq * dmD3_sq_dt_2l));
    double dmD2_sq_dt = ((loop_fac * dmD2_sq_dt_1l) + (loop_fac_sq * dmD2_sq_dt_2l));
    double dmD1_sq_dt = ((loop_fac * dmD1_sq_dt_1l) + (loop_fac_sq * dmD1_sq_dt_2l));
    double dmE3_sq_dt = ((loop_fac * dmE3_sq_dt_1l) + (loop_fac_sq * dmE3_sq_dt_2l));
    double dmE2_sq_dt = ((loop_fac * dmE2_sq_dt_1l) + (loop_fac_sq * dmE2_sq_dt_2l));
    double dmE1_sq_dt = ((loop_fac * dmE1_sq_dt_1l) + (loop_fac_sq * dmE1_sq_dt_2l));

    // tanb beta function at one-loop level
    double dtanb_dt = 3.0 * loop_fac * tanb_val * (pow(yb_val, 2.0) - pow(yt_val, 2.0));

    // std::vector<double> dxdt = {dg1_dt, dg2_dt, dg3_dt, dM1_dt, dM2_dt, dM3_dt, dmu_dt, dyt_dt,
    //                             dyc_dt, dyu_dt, dyb_dt, dys_dt, dyd_dt, dytau_dt, dymu_dt,
    //                             dye_dt, dat_dt, dac_dt, dau_dt, dab_dt, das_dt, dad_dt,
    //                             datau_dt, damu_dt, dae_dt, dmHu_sq_dt + (2.0 * mu_val * dmu_dt),
    //                             dmHd_sq_dt + (2.0 * mu_val * dmu_dt), dmQ1_sq_dt, dmQ2_sq_dt, dmQ3_sq_dt,
    //                             dmL1_sq_dt, dmL2_sq_dt, dmL3_sq_dt, dmU1_sq_dt, dmU2_sq_dt,
    //                             dmU3_sq_dt, dmD1_sq_dt, dmD2_sq_dt, dmD3_sq_dt, dmE1_sq_dt,
    //                             dmE2_sq_dt, dmE3_sq_dt, db_dt, dtanb_dt};
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

typedef boost::numeric::odeint::runge_kutta_dopri5<std::vector<double>> stepper_type;

struct MyObserver2 {
    double& t_target;
    double& min_difference;
    bool& condition_met;

    MyObserver2(double& t_target, double& min_difference, bool& condition_met) : t_target(t_target),
        min_difference(min_difference), condition_met(condition_met) {}

    void operator()(const std::vector<double>& x, const double t) const {
        double current_Q = std::exp(t);
        double current_difference = std::abs(current_Q - std::pow(std::abs(x[29] * x[35]), 0.25));

        if (current_difference > min_difference) {
            return; // No need to check further
        }

        if ((current_difference < min_difference) && (current_Q < 1.0e11)) {
            condition_met = true;
            t_target = t;
            min_difference = current_difference;
        }
    }
};

std::vector<RGEStruct2> solveODEstoapproxMSUSY(std::vector<double> initialConditions, double startTime, double timeStep, double& t_target) {
    using state_type = std::vector<double>;
    state_type x = initialConditions;
    double endTime = std::log(500.0);
    double min_difference = std::numeric_limits<double>::infinity();
    bool condition_met = false;

    MyObserver2 myObserver(t_target, min_difference, condition_met);

    // Integrate
    boost::numeric::odeint::integrate_adaptive(
        boost::numeric::odeint::make_controlled(1.0E-12, 1.0E-12, stepper_type()),
        MSSM_approx_RGESolver, x, startTime, endTime, timeStep, myObserver
    );
    // Return final solution
    std::vector<RGEStruct2> solstruct = { {x, t_target} };
    return solstruct;

}