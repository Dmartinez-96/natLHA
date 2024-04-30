#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <eigen3/Eigen/Dense>
#include "EWSB_loop.hpp"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

high_prec_float mylogSq(high_prec_float& x, high_prec_float& Qsq) {
    return (log(abs(x) / Qsq) - 1.0);
}

high_prec_float mylogLin(high_prec_float& x, high_prec_float& Qsq) {
    return (log(x * x / Qsq) - 1.0);
}

high_prec_float neut_UU_DD_2deriv(high_prec_float& sLambda, high_prec_float& g2Sq, high_prec_float& gpSq, high_prec_float& M1, high_prec_float& M2, high_prec_float& muvalue) {
    high_prec_float myNum = ((g2Sq + gpSq) * sLambda * sLambda) - (sLambda * ((M1 * g2Sq) + (M2 * gpSq)));
    high_prec_float myDenom = (4.0 * pow(sLambda, 3.0)) - (3.0 * (M1 + M2) * sLambda * sLambda)\
        + (2.0 * sLambda * ((M1 * M2) - (muvalue * muvalue))) - (M1 * M2 * muvalue * muvalue);
    return (myNum / myDenom);
}

high_prec_float neut_UD_2deriv(high_prec_float& sLambda, high_prec_float& g2Sq, high_prec_float& gpSq, high_prec_float& M1, high_prec_float& M2, high_prec_float& muvalue) {
    high_prec_float myNum = ((g2Sq + gpSq) * muvalue * sLambda) - (muvalue * ((M1 * g2Sq) + (M2 * gpSq)));
    high_prec_float myDenom = (4.0 * pow(sLambda, 3.0)) - (3.0 * (M1 + M2) * sLambda * sLambda)\
        + (2.0 * sLambda * ((M1 * M2) - (muvalue * muvalue))) - (M1 * M2 * muvalue * muvalue);
    return (myNum / myDenom);
}

bool Hessian_check(vector<high_prec_float> weak_boundary_conditions, high_prec_float myQ) {
    high_prec_float Sigmauu2, Sigmadd2, Sigmaud2;
    //high_prec_float mymZ = sqrt(abs(mymZsq));
    high_prec_float g1_wk = weak_boundary_conditions[0];
    high_prec_float g2_wk = weak_boundary_conditions[1];
    high_prec_float g3_wk = weak_boundary_conditions[2];
    // Higgs parameters
    high_prec_float beta_wk = atan(weak_boundary_conditions[43]);
    high_prec_float mu_wk = weak_boundary_conditions[6];
    high_prec_float mu_wk_sq = pow(mu_wk, 2.0);
    // Yukawas
    high_prec_float yt_wk = weak_boundary_conditions[7];
    high_prec_float yc_wk = weak_boundary_conditions[8];
    high_prec_float yu_wk = weak_boundary_conditions[9];
    high_prec_float yb_wk = weak_boundary_conditions[10];
    high_prec_float ys_wk = weak_boundary_conditions[11];
    high_prec_float yd_wk = weak_boundary_conditions[12];
    high_prec_float ytau_wk = weak_boundary_conditions[13];
    high_prec_float ymu_wk = weak_boundary_conditions[14];
    high_prec_float ye_wk = weak_boundary_conditions[15];
    // Soft trilinears
    high_prec_float at_wk = weak_boundary_conditions[16];
    high_prec_float ac_wk = weak_boundary_conditions[17];
    high_prec_float au_wk = weak_boundary_conditions[18];
    high_prec_float ab_wk = weak_boundary_conditions[19];
    high_prec_float as_wk = weak_boundary_conditions[20];
    high_prec_float ad_wk = weak_boundary_conditions[21];
    high_prec_float atau_wk = weak_boundary_conditions[22];
    high_prec_float amu_wk = weak_boundary_conditions[23];
    high_prec_float ae_wk = weak_boundary_conditions[24];
    // Gaugino masses
    high_prec_float M1_wk = weak_boundary_conditions[3];
    high_prec_float M2_wk = weak_boundary_conditions[4];
    high_prec_float M3_wk = weak_boundary_conditions[5];
    // Soft mass dim. 2 terms
    high_prec_float mHu_sq_wk = weak_boundary_conditions[25];
    high_prec_float mHd_sq_wk = weak_boundary_conditions[26];
    high_prec_float mQ1_sq_wk = weak_boundary_conditions[27];
    high_prec_float mQ2_sq_wk = weak_boundary_conditions[28];
    high_prec_float mQ3_sq_wk = weak_boundary_conditions[29];
    high_prec_float mL1_sq_wk = weak_boundary_conditions[30];
    high_prec_float mL2_sq_wk = weak_boundary_conditions[31];
    high_prec_float mL3_sq_wk = weak_boundary_conditions[32];
    high_prec_float mU1_sq_wk = weak_boundary_conditions[33];
    high_prec_float mU2_sq_wk = weak_boundary_conditions[34];
    high_prec_float mU3_sq_wk = weak_boundary_conditions[35];
    high_prec_float mD1_sq_wk = weak_boundary_conditions[36];
    high_prec_float mD2_sq_wk = weak_boundary_conditions[37];
    high_prec_float mD3_sq_wk = weak_boundary_conditions[38];
    high_prec_float mE1_sq_wk = weak_boundary_conditions[39];
    high_prec_float mE2_sq_wk = weak_boundary_conditions[40];
    high_prec_float mE3_sq_wk = weak_boundary_conditions[41];
    high_prec_float b_wk = weak_boundary_conditions[42];
    high_prec_float gpr_wk = g1_wk * sqrt(3.0 / 5.0);
    // // cout << "gpr_wk: " << gpr_wk << endl;
    high_prec_float gpr_sq = pow(gpr_wk, 2.0);
    // // cout << "gpr_sq: " << gpr_sq << endl;
    high_prec_float g2_sq = pow(g2_wk, 2.0);
    // // cout << "g2_sq: " << g2_sq << endl;
    // // cout << "mu_wk_sq: " << mu_wk_sq << endl;
    //high_prec_float vHiggs_wk = mymZ * sqrt(2.0 / (gpr_sq + g2_sq));
    high_prec_float sinsqb = pow(sin(beta_wk), 2.0);
    // // cout << "sinsqb: " << sinsqb << endl;
    high_prec_float cossqb = pow(cos(beta_wk), 2.0);
    // // cout << "cossqb: " << cossqb << endl;
    //high_prec_float vu = vHiggs_wk * sqrt(sinsqb);
    // // cout << "vu: " << vu << endl;
    //high_prec_float vd = vHiggs_wk * sqrt(cossqb);
    // // cout << "vd: " << vd << endl;
    //high_prec_float vu_sq = pow(vu, 2.0);
    // // cout << "vu_sq: " << vu_sq << endl;
    //high_prec_float vd_sq = pow(vd, 2.0);
    // // cout << "vd_sq: " << vd_sq << endl;
    //high_prec_float v_sq = pow(vHiggs_wk, 2.0);
    // // cout << "v_sq: " << v_sq << endl;
    high_prec_float tan_th_w = gpr_wk / g2_wk;
    // // cout << "tan_th_w: " << tan_th_w << endl;
    high_prec_float theta_w = atan(tan_th_w);
    // // cout << "theta_w: " << theta_w << endl;
    high_prec_float sinsq_th_w = pow(sin(theta_w), 2.0);
    // // cout << "sinsq_th_w: " << sinsq_th_w << endl;
    high_prec_float cos2b = cos(2.0 * beta_wk);
    // // cout << "cos2b: " << cos2b << endl;
    high_prec_float sin2b = sin(2.0 * beta_wk);
    // // cout << "sin2b: " << sin2b << endl;
    high_prec_float gz_sq = (pow(g2_wk, 2.0) + pow(gpr_wk, 2.0)) / 8.0;
    // // cout << "gz_sq: " << gz_sq << endl;

    ////////// Mass relations: //////////

    // W-boson tree-level running squared mass
    //high_prec_float m_w_sq = (pow(g2_wk, 2.0) / 2.0) * v_sq;

    // Z-boson tree-level running squared mass
    //high_prec_float mz_q_sq = mymZsq;// v_sq* ((pow(g2_wk, 2.0) + pow(gpr_wk, 2.0)) / 2.0);

    // Higgs psuedoscalar tree-level running squared mass
    high_prec_float mA0sq = 2.0 * mu_wk_sq + mHu_sq_wk + mHd_sq_wk;

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                    BEGIN HESSIAN TERMS                                                        //
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // SM particles fermions and gauge bosons do not contribute anything to loop EWSB conditions.
    // Lightest neutral Higgs doesn't contribute either
    high_prec_float Q2 = myQ * myQ;

    // Higgs sector
    high_prec_float Sigmauu2_H0 = ((g2_sq + gpr_sq) / (8.0 * M_PI * M_PI))\
        * (mA0sq * pow(cossqb, 2.0) * ((2.0 * cos2b) - 1.0) * mylogSq(mA0sq, Q2));
    high_prec_float Sigmadd2_H0 = ((-1.0) * (g2_sq + gpr_sq) / (8.0 * M_PI * M_PI))\
        * (mA0sq * pow(sinsqb, 2.0) * ((2.0 * cos2b) + 1.0) * mylogSq(mA0sq, Q2));
    high_prec_float Sigmaud2_H0 = ((g2_sq + gpr_sq) / (16.0 * M_PI * M_PI))\
        * (mA0sq * pow(sin2b, 3.0) * mylogSq(mA0sq, Q2));

    high_prec_float Sigmauu2_Hpm = (g2_sq / (16.0 * M_PI * M_PI)) * mA0sq * mylogSq(mA0sq, Q2);
    high_prec_float Sigmadd2_Hpm = Sigmauu2_Hpm;

    // Neutralino sector
    high_prec_float mN1 = mu_wk;
    high_prec_float mN2 = (-1.0) * mu_wk;
    high_prec_float mN3 = M1_wk;
    high_prec_float mN4 = M2_wk;
    high_prec_float Sigmauu2_N1 = ((-1.0) / (8.0 * M_PI * M_PI)) * pow(mN1, 3.0)\
        * mylogLin(mN1, Q2) * neut_UU_DD_2deriv(mN1, g2_sq, gpr_sq, M1_wk, M2_wk, mu_wk);
    high_prec_float Sigmadd2_N1 = Sigmauu2_N1;
    high_prec_float Sigmaud2_N1 = ((-1.0) / (8.0 * M_PI * M_PI)) * pow(mN1, 3.0)\
        * mylogLin(mN1, Q2) * neut_UD_2deriv(mN1, g2_sq, gpr_sq, M1_wk, M2_wk, mu_wk);
    
    high_prec_float Sigmauu2_N2 = ((-1.0) / (8.0 * M_PI * M_PI)) * pow(mN2, 3.0)\
        * mylogLin(mN2, Q2) * neut_UU_DD_2deriv(mN2, g2_sq, gpr_sq, M1_wk, M2_wk, mu_wk);
    high_prec_float Sigmadd2_N2 = Sigmauu2_N2;
    high_prec_float Sigmaud2_N2 = ((-1.0) / (8.0 * M_PI * M_PI)) * pow(mN2, 3.0)\
        * mylogLin(mN2, Q2) * neut_UD_2deriv(mN2, g2_sq, gpr_sq, M1_wk, M2_wk, mu_wk);
    
    high_prec_float Sigmauu2_N3 = ((-1.0) / (8.0 * M_PI * M_PI)) * pow(mN3, 3.0)\
        * mylogLin(mN3, Q2) * neut_UU_DD_2deriv(mN3, g2_sq, gpr_sq, M1_wk, M2_wk, mu_wk);
    high_prec_float Sigmadd2_N3 = Sigmauu2_N3;
    high_prec_float Sigmaud2_N3 = ((-1.0) / (8.0 * M_PI * M_PI)) * pow(mN3, 3.0)\
        * mylogLin(mN3, Q2) * neut_UD_2deriv(mN3, g2_sq, gpr_sq, M1_wk, M2_wk, mu_wk);
    
    high_prec_float Sigmauu2_N4 = ((-1.0) / (8.0 * M_PI * M_PI)) * pow(mN4, 3.0)\
        * mylogLin(mN4, Q2) * neut_UU_DD_2deriv(mN4, g2_sq, gpr_sq, M1_wk, M2_wk, mu_wk);
    high_prec_float Sigmadd2_N4 = Sigmauu2_N4;
    high_prec_float Sigmaud2_N4 = ((-1.0) / (8.0 * M_PI * M_PI)) * pow(mN4, 3.0)\
        * mylogLin(mN4, Q2) * neut_UD_2deriv(mN4, g2_sq, gpr_sq, M1_wk, M2_wk, mu_wk);

    // Chargino sector
    high_prec_float mCharg1Sq = 0.5 * ((M2_wk * M2_wk) + (mu_wk * mu_wk) - abs((M2_wk * M2_wk) - (mu_wk * mu_wk)));
    high_prec_float mCharg2Sq = 0.5 * ((M2_wk * M2_wk) + (mu_wk * mu_wk) + abs((M2_wk * M2_wk) - (mu_wk * mu_wk)));
    high_prec_float Sigmauu2_C1 = ((-1.0) * g2_sq / (4.0 * M_PI * M_PI)) * mCharg1Sq * mCharg1Sq * mylogSq(mCharg1Sq, Q2) / abs((M2_wk * M2_wk) - (mu_wk * mu_wk));
    high_prec_float Sigmauu2_C2 = (g2_sq / (4.0 * M_PI * M_PI)) * mCharg2Sq * mCharg2Sq * mylogSq(mCharg2Sq, Q2) / abs((M2_wk * M2_wk) - (mu_wk * mu_wk));
    high_prec_float Sigmadd2_C1 = Sigmauu2_C1;
    high_prec_float Sigmadd2_C2 = Sigmauu2_C2;
    high_prec_float Sigmaud2_C1 = (g2_sq * M2_wk * mu_wk / (8.0 * M_PI * M_PI)) * mCharg1Sq * mylogSq(mCharg1Sq, Q2) / abs((M2_wk * M2_wk) - (mu_wk * mu_wk));
    high_prec_float Sigmaud2_C2 = ((-1.0) * g2_sq * M2_wk * mu_wk / (8.0 * M_PI * M_PI)) * mCharg2Sq * mylogSq(mCharg2Sq, Q2) / abs((M2_wk * M2_wk) - (mu_wk * mu_wk));

    // Squark sector
    high_prec_float m_stop1_sq = 0.5 * (mQ3_sq_wk + mU3_sq_wk - abs(mQ3_sq_wk - mU3_sq_wk));
    high_prec_float m_stop2_sq = 0.5 * (mQ3_sq_wk + mU3_sq_wk + abs(mQ3_sq_wk - mU3_sq_wk));
    high_prec_float m_scharm1_sq = 0.5 * (mQ2_sq_wk + mU2_sq_wk - abs(mQ2_sq_wk - mU2_sq_wk));
    high_prec_float m_scharm2_sq = 0.5 * (mQ2_sq_wk + mU2_sq_wk + abs(mQ2_sq_wk - mU2_sq_wk));
    high_prec_float m_sup1_sq = 0.5 * (mQ1_sq_wk + mU1_sq_wk - abs(mQ1_sq_wk - mU1_sq_wk));
    high_prec_float m_sup2_sq = 0.5 * (mQ1_sq_wk + mU1_sq_wk + abs(mQ1_sq_wk - mU1_sq_wk));
    high_prec_float m_sbot1_sq = 0.5 * (mQ3_sq_wk + mD3_sq_wk - abs(mQ3_sq_wk - mD3_sq_wk));
    high_prec_float m_sbot2_sq = 0.5 * (mQ3_sq_wk + mD3_sq_wk + abs(mQ3_sq_wk - mD3_sq_wk));
    high_prec_float m_sstrange1_sq = 0.5 * (mQ2_sq_wk + mD2_sq_wk - abs(mQ2_sq_wk - mD2_sq_wk));
    high_prec_float m_sstrange2_sq = 0.5 * (mQ2_sq_wk + mD2_sq_wk + abs(mQ2_sq_wk - mD2_sq_wk));
    high_prec_float m_sdown1_sq = 0.5 * (mQ1_sq_wk + mD1_sq_wk - abs(mQ1_sq_wk - mD1_sq_wk));
    high_prec_float m_sdown2_sq = 0.5 * (mQ1_sq_wk + mD1_sq_wk + abs(mQ1_sq_wk - mD1_sq_wk));

    high_prec_float Sigmauu2_stop1 = (m_stop1_sq / (64.0 * M_PI * M_PI * abs(mQ3_sq_wk - mU3_sq_wk))) * mylogSq(m_stop1_sq, Q2)\
        * ((((3.0 * g2_sq) - (10.0 * gpr_sq)) * (mQ3_sq_wk - mU3_sq_wk)) - (24.0 * at_wk * at_wk) - (3.0 * (g2_sq + (2.0 * gpr_sq) - (8.0 * yt_wk * yt_wk)) * (abs(mQ3_sq_wk - mU3_sq_wk))));
    high_prec_float Sigmauu2_stop2 = ((-1.0) * m_stop2_sq / (64.0 * M_PI * M_PI * abs(mQ3_sq_wk - mU3_sq_wk))) * mylogSq(m_stop2_sq, Q2)\
        * ((((3.0 * g2_sq) - (10.0 * gpr_sq)) * (mQ3_sq_wk - mU3_sq_wk)) - (24.0 * at_wk * at_wk) + (3.0 * (g2_sq + (2.0 * gpr_sq) - (8.0 * yt_wk * yt_wk)) * (abs(mQ3_sq_wk - mU3_sq_wk))));
    high_prec_float Sigmauu2_scharm1 = (m_scharm1_sq / (64.0 * M_PI * M_PI * abs(mQ2_sq_wk - mU2_sq_wk))) * mylogSq(m_scharm1_sq, Q2)\
        * ((((3.0 * g2_sq) - (10.0 * gpr_sq)) * (mQ2_sq_wk - mU2_sq_wk)) - (24.0 * ac_wk * ac_wk) - (3.0 * (g2_sq + (2.0 * gpr_sq) - (8.0 * yc_wk * yc_wk)) * (abs(mQ2_sq_wk - mU2_sq_wk))));
    high_prec_float Sigmauu2_scharm2 = ((-3.0) * m_scharm2_sq / (64.0 * M_PI * M_PI * abs(mQ2_sq_wk - mU2_sq_wk))) * mylogSq(m_scharm2_sq, Q2)\
        * ((((3.0 * g2_sq) - (10.0 * gpr_sq)) * (mQ2_sq_wk - mU2_sq_wk)) - (24.0 * ac_wk * ac_wk) + (3.0 * (g2_sq + (2.0 * gpr_sq) - (8.0 * yc_wk * yc_wk)) * (abs(mQ2_sq_wk - mU2_sq_wk))));
    high_prec_float Sigmauu2_sup1 = (m_sup1_sq / (64.0 * M_PI * M_PI * abs(mQ1_sq_wk - mU1_sq_wk))) * mylogSq(m_sup1_sq, Q2)\
        * ((((3.0 * g2_sq) - (10.0 * gpr_sq)) * (mQ1_sq_wk - mU1_sq_wk)) + (24.0 * au_wk * au_wk) - (3.0 * (g2_sq + (2.0 * gpr_sq) - (8.0 * yu_wk * yu_wk)) * (abs(mQ1_sq_wk - mU1_sq_wk))));
    high_prec_float Sigmauu2_sup2 = ((-1.0) * m_sup2_sq / (64.0 * M_PI * M_PI * abs(mQ1_sq_wk - mU1_sq_wk))) * mylogSq(m_sup2_sq, Q2)\
        * ((((3.0 * g2_sq) - (10.0 * gpr_sq)) * (mQ1_sq_wk - mU1_sq_wk)) + (24.0 * au_wk * au_wk) + (3.0 * (g2_sq + (2.0 * gpr_sq) - (8.0 * yu_wk * yu_wk)) * (abs(mQ1_sq_wk - mU1_sq_wk))));

    high_prec_float Sigmauu2_sbot1 = ((-1.0) * m_sbot1_sq / (64.0 * M_PI * M_PI * abs(mQ3_sq_wk - mD3_sq_wk))) * mylogSq(m_sbot1_sq, Q2)\
        * ((((3.0 * g2_sq) - (2.0 * gpr_sq)) * (mQ3_sq_wk - mD3_sq_wk)) + (24.0 * yb_wk * yb_wk * mu_wk_sq) - (3.0 * (g2_sq + (2.0 * gpr_sq)) * (abs(mQ3_sq_wk - mD3_sq_wk))));
    high_prec_float Sigmauu2_sbot2 = (m_sbot2_sq / (64.0 * M_PI * M_PI * abs(mQ3_sq_wk - mD3_sq_wk))) * mylogSq(m_sbot2_sq, Q2)\
        * ((((3.0 * g2_sq) - (2.0 * gpr_sq)) * (mQ3_sq_wk - mD3_sq_wk)) + (24.0 * yb_wk * yb_wk * mu_wk_sq) + (3.0 * (g2_sq + (2.0 * gpr_sq)) * (abs(mQ3_sq_wk - mD3_sq_wk))));
    high_prec_float Sigmauu2_sstrange1 = ((-1.0) * m_sstrange1_sq / (64.0 * M_PI * M_PI * abs(mQ2_sq_wk - mD2_sq_wk))) * mylogSq(m_sstrange1_sq, Q2)\
        * ((((3.0 * g2_sq) - (2.0 * gpr_sq)) * (mQ2_sq_wk - mD2_sq_wk)) + (24.0 * ys_wk * ys_wk * mu_wk_sq) - (3.0 * (g2_sq + (2.0 * gpr_sq)) * (abs(mQ2_sq_wk - mD2_sq_wk))));
    high_prec_float Sigmauu2_sstrange2 = (m_sstrange2_sq / (64.0 * M_PI * M_PI * abs(mQ2_sq_wk - mD2_sq_wk))) * mylogSq(m_sstrange2_sq, Q2)\
        * ((((3.0 * g2_sq) - (2.0 * gpr_sq)) * (mQ2_sq_wk - mD2_sq_wk)) + (24.0 * ys_wk * ys_wk * mu_wk_sq) + (3.0 * (g2_sq + (2.0 * gpr_sq)) * (abs(mQ2_sq_wk - mD2_sq_wk))));
    high_prec_float Sigmauu2_sdown1 = ((-1.0) * m_sdown1_sq / (64.0 * M_PI * M_PI * abs(mQ1_sq_wk - mD1_sq_wk))) * mylogSq(m_sdown1_sq, Q2)\
        * ((((3.0 * g2_sq) - (2.0 * gpr_sq)) * (mQ1_sq_wk - mD1_sq_wk)) + (24.0 * yd_wk * yd_wk * mu_wk_sq) - (3.0 * (g2_sq + (2.0 * gpr_sq)) * (abs(mQ1_sq_wk - mD1_sq_wk))));
    high_prec_float Sigmauu2_sdown2 = (m_sdown2_sq / (64.0 * M_PI * M_PI * abs(mQ1_sq_wk - mD1_sq_wk))) * mylogSq(m_sdown2_sq, Q2)\
        * ((((3.0 * g2_sq) - (2.0 * gpr_sq)) * (mQ1_sq_wk - mD1_sq_wk)) + (24.0 * yd_wk * yd_wk * mu_wk_sq) + (3.0 * (g2_sq + (2.0 * gpr_sq)) * (abs(mQ1_sq_wk - mD1_sq_wk))));

    high_prec_float Sigmadd2_stop1 = ((-1.0) * m_stop1_sq / (64.0 * M_PI * M_PI * abs(mQ3_sq_wk - mU3_sq_wk))) * mylogSq(m_stop1_sq, Q2)\
        * ((((3.0 * g2_sq) - (10.0 * gpr_sq)) * (mQ3_sq_wk - mU3_sq_wk)) + (24.0 * yt_wk * yt_wk * mu_wk_sq) - (3.0 * (g2_sq + (2.0 * gpr_sq)) * (abs(mQ3_sq_wk - mU3_sq_wk))));
    high_prec_float Sigmadd2_stop2 = (m_stop2_sq / (64.0 * M_PI * M_PI * abs(mQ3_sq_wk - mU3_sq_wk))) * mylogSq(m_stop2_sq, Q2)\
        * ((((3.0 * g2_sq) - (10.0 * gpr_sq)) * (mQ3_sq_wk - mU3_sq_wk)) + (24.0 * yt_wk * yt_wk * mu_wk_sq) + (3.0 * (g2_sq + (2.0 * gpr_sq)) * (abs(mQ3_sq_wk - mU3_sq_wk))));
    high_prec_float Sigmadd2_scharm1 = ((-1.0) * m_scharm1_sq / (64.0 * M_PI * M_PI * abs(mQ2_sq_wk - mU2_sq_wk))) * mylogSq(m_scharm1_sq, Q2)\
        * ((((3.0 * g2_sq) - (10.0 * gpr_sq)) * (mQ2_sq_wk - mU2_sq_wk)) + (24.0 * yc_wk * yc_wk * mu_wk_sq) - (3.0 * (g2_sq + (2.0 * gpr_sq)) * (abs(mQ2_sq_wk - mU2_sq_wk))));
    high_prec_float Sigmadd2_scharm2 = (m_scharm2_sq / (64.0 * M_PI * M_PI * abs(mQ2_sq_wk - mU2_sq_wk))) * mylogSq(m_scharm2_sq, Q2)\
        * ((((3.0 * g2_sq) - (10.0 * gpr_sq)) * (mQ2_sq_wk - mU2_sq_wk)) + (24.0 * yc_wk * yc_wk * mu_wk_sq) + (3.0 * (g2_sq + (2.0 * gpr_sq)) * (abs(mQ2_sq_wk - mU2_sq_wk))));
    high_prec_float Sigmadd2_sup1 = ((-1.0) * m_sup1_sq / (64.0 * M_PI * M_PI * abs(mQ1_sq_wk - mU1_sq_wk))) * mylogSq(m_sup1_sq, Q2)\
        * ((((3.0 * g2_sq) - (10.0 * gpr_sq)) * (mQ1_sq_wk - mU1_sq_wk)) + (24.0 * yu_wk * yu_wk * mu_wk_sq) - (3.0 * (g2_sq + (2.0 * gpr_sq)) * (abs(mQ1_sq_wk - mU1_sq_wk))));
    high_prec_float Sigmadd2_sup2 = (m_sup2_sq / (64.0 * M_PI * M_PI * abs(mQ1_sq_wk - mU1_sq_wk))) * mylogSq(m_sup2_sq, Q2)\
        * ((((3.0 * g2_sq) - (10.0 * gpr_sq)) * (mQ1_sq_wk - mU1_sq_wk)) + (24.0 * yu_wk * yu_wk * mu_wk_sq) + (3.0 * (g2_sq + (2.0 * gpr_sq)) * (abs(mQ1_sq_wk - mU1_sq_wk))));
        
    high_prec_float Sigmadd2_sbot1 = (m_sbot1_sq / (64.0 * M_PI * M_PI * abs(mQ3_sq_wk - mD3_sq_wk))) * mylogSq(m_sbot1_sq, Q2)\
        * ((((3.0 * g2_sq) - (2.0 * gpr_sq)) * (mQ3_sq_wk - mD3_sq_wk)) - (24.0 * ab_wk * ab_wk) - (3.0 * (g2_sq + (2.0 * gpr_sq) - (8.0 * yb_wk * yb_wk)) * (abs(mQ3_sq_wk - mD3_sq_wk))));
    high_prec_float Sigmadd2_sbot2 = (m_sbot2_sq / (64.0 * M_PI * M_PI * abs(mQ3_sq_wk - mD3_sq_wk))) * mylogSq(m_sbot2_sq, Q2)\
        * ((((3.0 * g2_sq) - (2.0 * gpr_sq)) * (mQ3_sq_wk - mD3_sq_wk)) - (24.0 * ab_wk * ab_wk) + (3.0 * (g2_sq + (2.0 * gpr_sq) - (8.0 * yb_wk * yb_wk)) * (abs(mQ3_sq_wk - mD3_sq_wk))));
    high_prec_float Sigmadd2_sstrange1 = (m_sstrange1_sq / (64.0 * M_PI * M_PI * abs(mQ2_sq_wk - mD2_sq_wk))) * mylogSq(m_sstrange1_sq, Q2)\
        * ((((3.0 * g2_sq) - (2.0 * gpr_sq)) * (mQ2_sq_wk - mD2_sq_wk)) - (24.0 * as_wk * as_wk) - (3.0 * (g2_sq + (2.0 * gpr_sq) - (8.0 * yb_wk * yb_wk)) * (abs(mQ2_sq_wk - mD2_sq_wk))));
    high_prec_float Sigmadd2_sstrange2 = (m_sstrange2_sq / (64.0 * M_PI * M_PI * abs(mQ2_sq_wk - mD2_sq_wk))) * mylogSq(m_sstrange2_sq, Q2)\
        * ((((3.0 * g2_sq) - (2.0 * gpr_sq)) * (mQ2_sq_wk - mD2_sq_wk)) - (24.0 * as_wk * as_wk) + (3.0 * (g2_sq + (2.0 * gpr_sq) - (8.0 * yb_wk * yb_wk)) * (abs(mQ2_sq_wk - mD2_sq_wk))));
    high_prec_float Sigmadd2_sdown1 = (m_sdown1_sq / (64.0 * M_PI * M_PI * abs(mQ1_sq_wk - mD1_sq_wk))) * mylogSq(m_sdown1_sq, Q2)\
        * ((((3.0 * g2_sq) - (2.0 * gpr_sq)) * (mQ1_sq_wk - mD1_sq_wk)) - (24.0 * ad_wk * ad_wk) - (3.0 * (g2_sq + (2.0 * gpr_sq) - (8.0 * yb_wk * yb_wk)) * (abs(mQ1_sq_wk - mD1_sq_wk))));
    high_prec_float Sigmadd2_sdown2 = (m_sdown2_sq / (64.0 * M_PI * M_PI * abs(mQ1_sq_wk - mD1_sq_wk))) * mylogSq(m_sdown2_sq, Q2)\
        * ((((3.0 * g2_sq) - (2.0 * gpr_sq)) * (mQ1_sq_wk - mD1_sq_wk)) - (24.0 * ad_wk * ad_wk) + (3.0 * (g2_sq + (2.0 * gpr_sq) - (8.0 * yb_wk * yb_wk)) * (abs(mQ1_sq_wk - mD1_sq_wk))));

    high_prec_float Sigmaud2_stop1 = ((3.0) * at_wk * yt_wk * mu_wk * m_stop1_sq / (8.0 * M_PI * M_PI * abs(mQ3_sq_wk - mU3_sq_wk))) * mylogSq(m_stop1_sq, Q2);
    high_prec_float Sigmaud2_stop2 = ((-3.0) * at_wk * yt_wk * mu_wk * m_stop2_sq / (8.0 * M_PI * M_PI * abs(mQ3_sq_wk - mU3_sq_wk))) * mylogSq(m_stop2_sq, Q2);
    high_prec_float Sigmaud2_scharm1 = ((3.0) * ac_wk * yc_wk * mu_wk * m_scharm1_sq / (8.0 * M_PI * M_PI * abs(mQ2_sq_wk - mU2_sq_wk))) * mylogSq(m_scharm1_sq, Q2);
    high_prec_float Sigmaud2_scharm2 = ((-3.0) * ac_wk * yc_wk * mu_wk * m_scharm2_sq / (8.0 * M_PI * M_PI * abs(mQ2_sq_wk - mU2_sq_wk))) * mylogSq(m_scharm2_sq, Q2);
    high_prec_float Sigmaud2_sup1 = ((3.0) * au_wk * yu_wk * mu_wk * m_sup1_sq / (8.0 * M_PI * M_PI * abs(mQ1_sq_wk - mU1_sq_wk))) * mylogSq(m_sup1_sq, Q2);
    high_prec_float Sigmaud2_sup2 = ((-3.0) * au_wk * yu_wk * mu_wk * m_sup2_sq / (8.0 * M_PI * M_PI * abs(mQ1_sq_wk - mU1_sq_wk))) * mylogSq(m_sup2_sq, Q2);

    high_prec_float Sigmaud2_sbot1 = ((3.0) * ab_wk * yb_wk * mu_wk * m_sbot1_sq / (8.0 * M_PI * M_PI * abs(mQ3_sq_wk - mD3_sq_wk))) * mylogSq(m_sbot1_sq, Q2);
    high_prec_float Sigmaud2_sbot2 = ((-3.0) * ab_wk * yb_wk * mu_wk * m_sbot2_sq / (8.0 * M_PI * M_PI * abs(mQ3_sq_wk - mD3_sq_wk))) * mylogSq(m_sbot2_sq, Q2);
    high_prec_float Sigmaud2_sstrange1 = ((3.0) * as_wk * ys_wk * mu_wk * m_sstrange1_sq / (8.0 * M_PI * M_PI * abs(mQ2_sq_wk - mD2_sq_wk))) * mylogSq(m_sstrange1_sq, Q2);
    high_prec_float Sigmaud2_sstrange2 = ((-3.0) * as_wk * ys_wk * mu_wk * m_sstrange2_sq / (8.0 * M_PI * M_PI * abs(mQ2_sq_wk - mD2_sq_wk))) * mylogSq(m_sstrange2_sq, Q2);
    high_prec_float Sigmaud2_sdown1 = ((3.0) * ad_wk * yd_wk * mu_wk * m_sdown1_sq / (8.0 * M_PI * M_PI * abs(mQ1_sq_wk - mD1_sq_wk))) * mylogSq(m_sdown1_sq, Q2);
    high_prec_float Sigmaud2_sdown2 = ((-3.0) * ad_wk * yd_wk * mu_wk * m_sdown2_sq / (8.0 * M_PI * M_PI * abs(mQ1_sq_wk - mD1_sq_wk))) * mylogSq(m_sdown2_sq, Q2);

    // Slepton sector
    high_prec_float m_stau1_sq = 0.5 * (mL3_sq_wk + mE3_sq_wk - abs(mL3_sq_wk - mE3_sq_wk));
    high_prec_float m_stau2_sq = 0.5 * (mL3_sq_wk + mE3_sq_wk + abs(mL3_sq_wk - mE3_sq_wk));
    high_prec_float m_smu1_sq = 0.5 * (mL2_sq_wk + mE2_sq_wk - abs(mL2_sq_wk - mE2_sq_wk));
    high_prec_float m_smu2_sq = 0.5 * (mL2_sq_wk + mE2_sq_wk + abs(mL2_sq_wk - mE2_sq_wk));
    high_prec_float m_se1_sq = 0.5 * (mL1_sq_wk + mE1_sq_wk - abs(mL1_sq_wk - mE1_sq_wk));
    high_prec_float m_se2_sq = 0.5 * (mL1_sq_wk + mE1_sq_wk + abs(mL1_sq_wk - mE1_sq_wk));
    
    high_prec_float Sigmauu2_stau1 = ((-1.0) * m_stau1_sq / (64.0 * M_PI * M_PI * abs(mL3_sq_wk - mE3_sq_wk))) * mylogSq(m_stau1_sq, Q2)\
        * ((((g2_sq) - (6.0 * gpr_sq)) * (mL3_sq_wk - mE3_sq_wk)) + (8.0 * ytau_wk * ytau_wk * mu_wk_sq) - ((g2_sq + (2.0 * gpr_sq)) * (abs(mL3_sq_wk - mE3_sq_wk))));
    high_prec_float Sigmauu2_stau2 = (m_stau2_sq / (64.0 * M_PI * M_PI * abs(mL3_sq_wk - mE3_sq_wk))) * mylogSq(m_stau2_sq, Q2)\
        * ((((g2_sq) - (6.0 * gpr_sq)) * (mL3_sq_wk - mE3_sq_wk)) + (8.0 * ytau_wk * ytau_wk * mu_wk_sq) + ((g2_sq + (2.0 * gpr_sq)) * (abs(mL3_sq_wk - mE3_sq_wk))));
    high_prec_float Sigmauu2_smu1 = ((-1.0) * m_smu1_sq / (64.0 * M_PI * M_PI * abs(mL2_sq_wk - mE2_sq_wk))) * mylogSq(m_smu1_sq, Q2)\
        * ((((g2_sq) - (6.0 * gpr_sq)) * (mL2_sq_wk - mE2_sq_wk)) + (8.0 * ymu_wk * ymu_wk * mu_wk_sq) - ((g2_sq + (2.0 * gpr_sq)) * (abs(mL2_sq_wk - mE2_sq_wk))));
    high_prec_float Sigmauu2_smu2 = (m_smu2_sq / (64.0 * M_PI * M_PI * abs(mL2_sq_wk - mE2_sq_wk))) * mylogSq(m_smu2_sq, Q2)\
        * ((((g2_sq) - (6.0 * gpr_sq)) * (mL2_sq_wk - mE2_sq_wk)) + (8.0 * ymu_wk * ymu_wk * mu_wk_sq) + ((g2_sq + (2.0 * gpr_sq)) * (abs(mL2_sq_wk - mE2_sq_wk))));
    high_prec_float Sigmauu2_se1 = ((-1.0) * m_se1_sq / (64.0 * M_PI * M_PI * abs(mL1_sq_wk - mE1_sq_wk))) * mylogSq(m_se1_sq, Q2)\
        * ((((g2_sq) - (6.0 * gpr_sq)) * (mL1_sq_wk - mE1_sq_wk)) + (8.0 * ye_wk * ye_wk * mu_wk_sq) - ((g2_sq + (2.0 * gpr_sq)) * (abs(mL1_sq_wk - mE1_sq_wk))));
    high_prec_float Sigmauu2_se2 = (m_se2_sq / (64.0 * M_PI * M_PI * abs(mL1_sq_wk - mE1_sq_wk))) * mylogSq(m_se2_sq, Q2)\
        * ((((g2_sq) - (6.0 * gpr_sq)) * (mL1_sq_wk - mE1_sq_wk)) + (8.0 * ye_wk * ye_wk * mu_wk_sq) + ((g2_sq + (2.0 * gpr_sq)) * (abs(mL1_sq_wk - mE1_sq_wk))));
    high_prec_float Sigmauu2_tau_sneut = ((-1.0) * (g2_sq + gpr_sq) / (64.0 * M_PI * M_PI)) * mL3_sq_wk * mylogSq(mL3_sq_wk, Q2);
    high_prec_float Sigmauu2_mu_sneut = ((-1.0) * (g2_sq + gpr_sq) / (64.0 * M_PI * M_PI)) * mL2_sq_wk * mylogSq(mL2_sq_wk, Q2);
    high_prec_float Sigmauu2_e_sneut = ((-1.0) * (g2_sq + gpr_sq) / (64.0 * M_PI * M_PI)) * mL1_sq_wk * mylogSq(mL1_sq_wk, Q2);

    high_prec_float Sigmadd2_stau1 = (m_stau1_sq / (64.0 * M_PI * M_PI * abs(mL3_sq_wk - mE3_sq_wk))) * mylogSq(m_stau1_sq, Q2)\
        * ((((g2_sq) - (6.0 * gpr_sq)) * (mL3_sq_wk - mE3_sq_wk)) - (8.0 * atau_wk * atau_wk) - ((g2_sq + (2.0 * gpr_sq) - (8.0 * ytau_wk * ytau_wk)) * (abs(mL3_sq_wk - mE3_sq_wk))));
    high_prec_float Sigmadd2_stau2 = ((-1.0) * m_stau2_sq / (64.0 * M_PI * M_PI * abs(mL3_sq_wk - mE3_sq_wk))) * mylogSq(m_stau2_sq, Q2)\
        * ((((g2_sq) - (6.0 * gpr_sq)) * (mL3_sq_wk - mE3_sq_wk)) - (8.0 * atau_wk * atau_wk) + ((g2_sq + (2.0 * gpr_sq) - (8.0 * ytau_wk * ytau_wk)) * (abs(mL3_sq_wk - mE3_sq_wk))));
    high_prec_float Sigmadd2_smu1 = (m_smu1_sq / (64.0 * M_PI * M_PI * abs(mL2_sq_wk - mE2_sq_wk))) * mylogSq(m_smu1_sq, Q2)\
        * ((((g2_sq) - (6.0 * gpr_sq)) * (mL2_sq_wk - mE2_sq_wk)) - (8.0 * amu_wk * amu_wk) - ((g2_sq + (2.0 * gpr_sq) - (8.0 * ymu_wk * ymu_wk)) * (abs(mL2_sq_wk - mE2_sq_wk))));
    high_prec_float Sigmadd2_smu2 = ((-1.0) * m_smu2_sq / (64.0 * M_PI * M_PI * abs(mL2_sq_wk - mE2_sq_wk))) * mylogSq(m_smu2_sq, Q2)\
        * ((((g2_sq) - (6.0 * gpr_sq)) * (mL2_sq_wk - mE2_sq_wk)) - (8.0 * amu_wk * amu_wk) + ((g2_sq + (2.0 * gpr_sq) - (8.0 * ymu_wk * ymu_wk)) * (abs(mL2_sq_wk - mE2_sq_wk))));
    high_prec_float Sigmadd2_se1 = (m_se1_sq / (64.0 * M_PI * M_PI * abs(mL1_sq_wk - mE1_sq_wk))) * mylogSq(m_se1_sq, Q2)\
        * ((((g2_sq) - (6.0 * gpr_sq)) * (mL1_sq_wk - mE1_sq_wk)) - (8.0 * ae_wk * ae_wk) - ((g2_sq + (2.0 * gpr_sq) - (8.0 * ye_wk * ye_wk)) * (abs(mL1_sq_wk - mE1_sq_wk))));
    high_prec_float Sigmadd2_se2 = ((-1.0) * m_se2_sq / (64.0 * M_PI * M_PI * abs(mL1_sq_wk - mE1_sq_wk))) * mylogSq(m_se2_sq, Q2)\
        * ((((g2_sq) - (6.0 * gpr_sq)) * (mL1_sq_wk - mE1_sq_wk)) - (8.0 * ae_wk * ae_wk) + ((g2_sq + (2.0 * gpr_sq) - (8.0 * ye_wk * ye_wk)) * (abs(mL1_sq_wk - mE1_sq_wk))));
    high_prec_float Sigmadd2_tau_sneut = ((g2_sq + gpr_sq) / (64.0 * M_PI * M_PI)) * mL3_sq_wk * mylogSq(mL3_sq_wk, Q2);
    high_prec_float Sigmadd2_mu_sneut = ((g2_sq + gpr_sq) / (64.0 * M_PI * M_PI)) * mL2_sq_wk * mylogSq(mL2_sq_wk, Q2);
    high_prec_float Sigmadd2_e_sneut = ((g2_sq + gpr_sq) / (64.0 * M_PI * M_PI)) * mL1_sq_wk * mylogSq(mL1_sq_wk, Q2);
    
    high_prec_float Sigmaud2_stau1 = (atau_wk * ytau_wk * mu_wk * m_stau1_sq / (8.0 * M_PI * M_PI * abs(mL3_sq_wk - mE3_sq_wk))) * mylogSq(m_stau1_sq, Q2);
    high_prec_float Sigmaud2_stau2 = ((-1.0) * atau_wk * ytau_wk * mu_wk * m_stau2_sq / (8.0 * M_PI * M_PI * abs(mL3_sq_wk - mE3_sq_wk))) * mylogSq(m_stau2_sq, Q2);
    high_prec_float Sigmaud2_smu1 = (amu_wk * ymu_wk * mu_wk * m_smu1_sq / (8.0 * M_PI * M_PI * abs(mL2_sq_wk - mE2_sq_wk))) * mylogSq(m_smu1_sq, Q2);
    high_prec_float Sigmaud2_smu2 = ((-1.0) * amu_wk * ymu_wk * mu_wk * m_smu2_sq / (8.0 * M_PI * M_PI * abs(mL2_sq_wk - mE2_sq_wk))) * mylogSq(m_smu2_sq, Q2);
    high_prec_float Sigmaud2_se1 = (ae_wk * ye_wk * mu_wk * m_se1_sq / (8.0 * M_PI * M_PI * abs(mL1_sq_wk - mE1_sq_wk))) * mylogSq(m_se1_sq, Q2);
    high_prec_float Sigmaud2_se2 = ((-1.0) * ae_wk * ye_wk * mu_wk * m_se2_sq / (8.0 * M_PI * M_PI * abs(mL1_sq_wk - mE1_sq_wk))) * mylogSq(m_se2_sq, Q2);

    /* Vector order is:
     (0: Sigma_uu^(2),
      1: Sigma_dd^(2),
      2: Sigma_ud^(2))
    */
    Sigmauu2 = 0.0;
    Sigmadd2 = 0.0;
    Sigmaud2 = 0.0;
    Sigmauu2 += Sigmauu2_H0 + Sigmauu2_Hpm + Sigmauu2_N1 + Sigmauu2_N2 + Sigmauu2_N3 + Sigmauu2_N4\
        + Sigmauu2_C1 + Sigmauu2_C2 + Sigmauu2_stop1 + Sigmauu2_stop2 + Sigmauu2_scharm1 + Sigmauu2_scharm2\
        + Sigmauu2_sup1 + Sigmauu2_sup2 + Sigmauu2_sbot1 + Sigmauu2_sbot2 + Sigmauu2_sstrange1 + Sigmauu2_sstrange2\
        + Sigmauu2_stau1 + Sigmauu2_stau2 + Sigmauu2_smu1 + Sigmauu2_smu2 + Sigmauu2_se1 + Sigmauu2_se2 + Sigmauu2_tau_sneut\
        + Sigmauu2_mu_sneut + Sigmauu2_e_sneut;
    Sigmadd2 += Sigmadd2_H0 + Sigmadd2_Hpm + Sigmadd2_N1 + Sigmadd2_N2 + Sigmadd2_N3 + Sigmadd2_N4\
        + Sigmadd2_C1 + Sigmadd2_C2 + Sigmadd2_stop1 + Sigmadd2_stop2 + Sigmadd2_scharm1 + Sigmadd2_scharm2\
        + Sigmadd2_sup1 + Sigmadd2_sup2 + Sigmadd2_sbot1 + Sigmadd2_sbot2 + Sigmadd2_sstrange1 + Sigmadd2_sstrange2\
        + Sigmadd2_stau1 + Sigmadd2_stau2 + Sigmadd2_smu1 + Sigmadd2_smu2 + Sigmadd2_se1 + Sigmadd2_se2 + Sigmadd2_tau_sneut\
        + Sigmadd2_mu_sneut + Sigmadd2_e_sneut;
    Sigmaud2 += Sigmaud2_H0 + Sigmaud2_N1 + Sigmaud2_N2 + Sigmaud2_N3 + Sigmaud2_N4\
        + Sigmaud2_C1 + Sigmaud2_C2 + Sigmaud2_stop1 + Sigmaud2_stop2 + Sigmaud2_scharm1 + Sigmaud2_scharm2\
        + Sigmaud2_sup1 + Sigmaud2_sup2 + Sigmaud2_sbot1 + Sigmaud2_sbot2 + Sigmaud2_sstrange1 + Sigmaud2_sstrange2\
        + Sigmaud2_stau1 + Sigmaud2_stau2 + Sigmaud2_smu1 + Sigmaud2_smu2 + Sigmaud2_se1 + Sigmaud2_se2;
    vector<high_prec_float> HessianTerms = {Sigmauu2, Sigmadd2, Sigmaud2};
    bool OriginCheck = true;
    high_prec_float bval = weak_boundary_conditions[42];
    if ((pow((((-1.0) * bval) + Sigmaud2), 2.0) < ((mHu_sq_wk + mu_wk_sq + Sigmauu2) * (mHd_sq_wk + mu_wk_sq + Sigmadd2))) && ((mHu_sq_wk + mHd_sq_wk + (2.0 * mu_wk_sq) + Sigmauu2 + Sigmadd2) > 0)) {
            OriginCheck = false;
    }
    if (OriginCheck == false) {
        // std::cout << "Origin failed to destabilize at loop-level" << endl;
    }
    return OriginCheck;
}

bool BFB_check(vector<high_prec_float> weak_boundary_conditions) {
    high_prec_float Yt, Yc, Yu, Yb, Ys, Yd, Ytau, Ymu, Ye, Gp, G2, G3, Cos2Beta, LHS, RHS;
    Cos2Beta = cos(2.0 * atan(weak_boundary_conditions[43]));
    Yt = weak_boundary_conditions[7];
    Yc = weak_boundary_conditions[8];
    Yu = weak_boundary_conditions[9];
    Yb = weak_boundary_conditions[10];
    Ys = weak_boundary_conditions[11];
    Yd = weak_boundary_conditions[12];
    Ytau = weak_boundary_conditions[13];
    Ymu = weak_boundary_conditions[14];
    Ye = weak_boundary_conditions[15];
    Gp = sqrt(0.6) * weak_boundary_conditions[0];
    G2 = weak_boundary_conditions[1];
    G3 = weak_boundary_conditions[2];
    LHS = 8.0 * ((G2 * G2) + (2.0 * Gp * Gp))\
        * ((3.0 * (((Yb * Yb) - (Yt * Yt))
                   + ((Ys * Ys) - (Yc * Yc))
                   + ((Yd * Yd) - (Yu * Yu))))
           + ((Ytau * Ytau) + (Ymu * Ymu) + (Ye * Ye)));
    RHS = (13.0 * pow(G2, 4.0)) + (299.0 * pow(Gp, 4.0)) - (18.0 * G2 * G2 * Gp * Gp)\
        - (8.0 * ((2.0 * Gp * Gp) + (G2 * G2))
           * (3.0 * (((Yb * Yb) + (Yt * Yt))
                     + ((Ys * Ys) + (Yc * Yc))
                     + ((Yd * Yd) + (Yu * Yu)))));
    RHS *= Cos2Beta;
    if (LHS < RHS) {
        std::cout << "Potential not BFB at loop-level" << endl;
    }
    return (LHS > RHS);
}
