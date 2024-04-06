#include <iostream>
#include <cmath>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "tree_mass_calc.hpp"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

// Function to compute F = m^2.0 * (ln(m^2 / Q^2) - 1) for linear mass term
double logfunc(double mass, double Q_renorm_sq) {
    return pow(mass, 2.0) * (log(pow(mass, 2.0) / Q_renorm_sq) - 1);
}

// Function to compute F = m^2.0 * (ln(m^2 / Q^2) - 1) for quadratic mass term
double logfunc2(double masssq, double Q_renorm_sq) {
    return masssq * (log(masssq / Q_renorm_sq) - 1);
}

double PVB1(double extmom, double mass2, double Qval) {
    double my_M = max(pow(extmom, 2.0), mass2);
    double my_x = pow((extmom / mass2),2.0);
    double condexpr = 0.0;
    if (my_x > 1.0) {
        condexpr = 0.5 * log(my_x);
    }
    return (((-0.5) * log((my_M) / pow(Qval, 2.0))
                + 1.0 - ((1.0 / (2.0 * my_x))
                        * (1.0 + (pow((my_x - 1.0), 2.0)
                                * log(abs(my_x - 1.0))
                                / (my_x))))
                + condexpr));
}

// Function to compute tree-level mass spectrum
vector<double> TreeMassCalculator(std::vector<double> weak_boundary_conditions, double myQ, double mymZsq) {
    const double mymZ = copysign(sqrt(abs(mymZsq)), mymZsq);
    const double g1_wk = weak_boundary_conditions[0];
    const double g2_wk = weak_boundary_conditions[1];
    const double g3_wk = weak_boundary_conditions[2];
    // Higgs parameters
    const double beta_wk = atan(weak_boundary_conditions[43]);
    const double mu_wk = weak_boundary_conditions[6];
    const double mu_wk_sq = pow(mu_wk, 2.0);
    // Yukawas
    const double yt_wk = weak_boundary_conditions[7];
    const double yc_wk = weak_boundary_conditions[8];
    const double yu_wk = weak_boundary_conditions[9];
    const double yb_wk = weak_boundary_conditions[10];
    const double ys_wk = weak_boundary_conditions[11];
    const double yd_wk = weak_boundary_conditions[12];
    const double ytau_wk = weak_boundary_conditions[13];
    const double ymu_wk = weak_boundary_conditions[14];
    const double ye_wk = weak_boundary_conditions[15];
    // Soft trilinears
    const double at_wk = weak_boundary_conditions[16];
    const double ac_wk = weak_boundary_conditions[17];
    const double au_wk = weak_boundary_conditions[18];
    const double ab_wk = weak_boundary_conditions[19];
    const double as_wk = weak_boundary_conditions[20];
    const double ad_wk = weak_boundary_conditions[21];
    const double atau_wk = weak_boundary_conditions[22];
    const double amu_wk = weak_boundary_conditions[23];
    const double ae_wk = weak_boundary_conditions[24];
    // Gaugino masses
    const double M1_wk = weak_boundary_conditions[3];
    const double M2_wk = weak_boundary_conditions[4];
    const double M3_wk = weak_boundary_conditions[5];
    // Soft mass dim. 2 terms
    const double mHu_sq_wk = weak_boundary_conditions[25];
    const double mHd_sq_wk = weak_boundary_conditions[26];
    const double mQ1_sq_wk = weak_boundary_conditions[27];
    const double mQ2_sq_wk = weak_boundary_conditions[28];
    const double mQ3_sq_wk = weak_boundary_conditions[29];
    const double mL1_sq_wk = weak_boundary_conditions[30];
    const double mL2_sq_wk = weak_boundary_conditions[31];
    const double mL3_sq_wk = weak_boundary_conditions[32];
    const double mU1_sq_wk = weak_boundary_conditions[33];
    const double mU2_sq_wk = weak_boundary_conditions[34];
    const double mU3_sq_wk = weak_boundary_conditions[35];
    const double mD1_sq_wk = weak_boundary_conditions[36];
    const double mD2_sq_wk = weak_boundary_conditions[37];
    const double mD3_sq_wk = weak_boundary_conditions[38];
    const double mE1_sq_wk = weak_boundary_conditions[39];
    const double mE2_sq_wk = weak_boundary_conditions[40];
    const double mE3_sq_wk = weak_boundary_conditions[41];
    const double b_wk = weak_boundary_conditions[42];
    double gpr_wk = g1_wk * sqrt(3.0 / 5.0);
    // // cout << "gpr_wk: " << gpr_wk << endl;
    double gpr_sq = pow(gpr_wk, 2.0);
    // // cout << "gpr_sq: " << gpr_sq << endl;
    double g2_sq = pow(g2_wk, 2.0);
    // // cout << "g2_sq: " << g2_sq << endl;
    // // cout << "mu_wk_sq: " << mu_wk_sq << endl;
    double vHiggs_wk = mymZ * sqrt(2.0 / (gpr_sq + g2_sq));
    double sinsqb = pow(sin(beta_wk), 2.0);
    // // cout << "sinsqb: " << sinsqb << endl;
    double cossqb = pow(cos(beta_wk), 2.0);
    // // cout << "cossqb: " << cossqb << endl;
    double vu = vHiggs_wk * sqrt(sinsqb);
    // // cout << "vu: " << vu << endl;
    double vd = vHiggs_wk * sqrt(cossqb);
    // // cout << "vd: " << vd << endl;
    double vu_sq = pow(vu, 2.0);
    // // cout << "vu_sq: " << vu_sq << endl;
    double vd_sq = pow(vd, 2.0);
    // // cout << "vd_sq: " << vd_sq << endl;
    double v_sq = pow(vHiggs_wk, 2.0);
    // // cout << "v_sq: " << v_sq << endl;
    double tan_th_w = gpr_wk / g2_wk;
    // // cout << "tan_th_w: " << tan_th_w << endl;
    double theta_w = atan(tan_th_w);
    // // cout << "theta_w: " << theta_w << endl;
    double sinsq_th_w = pow(sin(theta_w), 2.0);
    // // cout << "sinsq_th_w: " << sinsq_th_w << endl;
    double cos2b = cos(2.0 * beta_wk);
    // // cout << "cos2b: " << cos2b << endl;
    double sin2b = sin(2.0 * beta_wk);
    // // cout << "sin2b: " << sin2b << endl;
    double gz_sq = (pow(g2_wk, 2.0) + pow(gpr_wk, 2.0)) / 8.0;
    // // cout << "gz_sq: " << gz_sq << endl;

    ////////// Mass relations: //////////

    // W-boson tree-level running squared mass
    const double m_w_sq = (pow(g2_wk, 2.0) / 2.0) * v_sq;

    // Z-boson tree-level running squared mass
    const double mz_q_sq = pow(mymZ, 2.0);// v_sq* ((pow(g2_wk, 2.0) + pow(gpr_wk, 2.0)) / 2.0);

    // Higgs psuedoscalar tree-level running squared mass
    const double mA0sq = 2.0 * mu_wk_sq + mHu_sq_wk + mHd_sq_wk;

    // Top quark tree-level running mass
    const double mymt = yt_wk * vu;
    const double mymtsq = pow(mymt, 2.0);

    // Bottom quark tree-level running mass
    const double mymb = yb_wk * vd;
    const double mymbsq = pow(mymb, 2.0);

    // Tau tree-level running mass
    const double mymtau = ytau_wk * vd;
    const double mymtausq = pow(mymtau, 2.0);

    // Charm quark tree-level running mass
    const double mymc = yc_wk * vu;
    const double mymcsq = pow(mymc, 2.0);

    // Strange quark tree-level running mass
    const double myms = ys_wk * vd;
    const double mymssq = pow(myms, 2.0);

    // Muon tree-level running mass
    const double mymmu = ymu_wk * vd;
    const double mymmusq = pow(mymmu, 2.0);

    // Up quark tree-level running mass
    const double mymu = yu_wk * vu;
    const double mymusq = pow(mymu, 2.0);

    // Down quark tree-level running mass
    const double mymd = yd_wk * vd;
    const double mymdsq = pow(mymd, 2.0);

    // Electron tree-level running mass
    const double myme = ye_wk * vd;
    const double mymesq = pow(myme, 2.0);

    // Sneutrino running masses
    const double mselecneutsq = mL1_sq_wk + (0.25 * (gpr_sq + g2_sq) * (vd_sq - vu_sq));
    const double msmuneutsq = mL2_sq_wk + (0.25 * (gpr_sq + g2_sq) * (vd_sq - vu_sq));
    const double mstauneutsq = mL3_sq_wk + (0.25 * (gpr_sq + g2_sq) * (vd_sq - vu_sq));

    // Tree-level charged Higgs running squared mass.
    const double mH_pmsq = mA0sq + m_w_sq;

    // Set up hyperfine splitting contributions to squark/slepton masses
    double Delta_suL = (pow(vu, 2.0) - pow(vd, 2.0)) * ((gpr_sq / 6.0) - (g2_sq / 4.0));
    double Delta_suR = (-1.0) * (pow(vu, 2.0) - pow(vd, 2.0)) * ((4.0 * gpr_sq / 3.0));
    double Delta_sdL = (pow(vu, 2.0) - pow(vd, 2.0)) * ((gpr_sq / 6.0) + (g2_sq / 4.0));
    double Delta_sdR = (pow(vu, 2.0) - pow(vd, 2.0)) * ((gpr_sq / 3.0));
    double Delta_seL = (pow(vu, 2.0) - pow(vd, 2.0)) * ((g2_sq / 4.0) - (gpr_sq / 2.0));
    double Delta_seR = (pow(vu, 2.0) - pow(vd, 2.0)) * gpr_sq;

    // Up-type squark mass eigenstate eigenvalues
    double m_stop_1sq = (0.5)\
        * (mQ3_sq_wk + mU3_sq_wk + (2.0 * mymtsq) + Delta_suL + Delta_suR
           - sqrt(pow((mQ3_sq_wk + Delta_suL - mU3_sq_wk - Delta_suR), 2.0)
                  + (4.0 * pow(((at_wk * vu) - (mu_wk * yt_wk * vd)), 2.0))));
    double m_stop_2sq = (0.5)\
        * (mQ3_sq_wk + mU3_sq_wk + (2.0 * mymtsq) + Delta_suL + Delta_suR
           + sqrt(pow((mQ3_sq_wk + Delta_suL - mU3_sq_wk - Delta_suR), 2.0)
                  + (4.0 * pow(((at_wk * vu) - (mu_wk * yt_wk * vd)), 2.0))));
    double m_scharm_1sq = (0.5)\
        * (mQ2_sq_wk + mU2_sq_wk + (2.0 * mymcsq) + Delta_suL + Delta_suR
           - sqrt(pow((mQ2_sq_wk + Delta_suL - mU2_sq_wk - Delta_suR), 2.0)
                  + (4.0 * pow(((ac_wk * vu) - (mu_wk * yc_wk * vd)), 2.0))));
    double m_scharm_2sq = (0.5)\
        * (mQ2_sq_wk + mU2_sq_wk + (2.0 * mymcsq) + Delta_suL + Delta_suR
           + sqrt(pow((mQ2_sq_wk + Delta_suL - mU2_sq_wk - Delta_suR), 2.0)
                  + (4.0 * pow(((ac_wk * vu) - (mu_wk * yc_wk * vd)), 2.0))));
    double m_sup_1sq = (0.5)\
        * (mQ1_sq_wk + mU1_sq_wk + (2.0 * mymusq) + Delta_suL + Delta_suR
           - sqrt(pow((mQ1_sq_wk + Delta_suL - mU1_sq_wk - Delta_suR), 2.0)
                  + (4.0 * pow(((au_wk * vu) - (mu_wk * yu_wk * vd)), 2.0))));
    double m_sup_2sq = (0.5)\
        * (mQ1_sq_wk + mU1_sq_wk + (2.0 * mymusq) + Delta_suL + Delta_suR
           + sqrt(pow((mQ1_sq_wk + Delta_suL - mU1_sq_wk - Delta_suR), 2.0)
                  + (4.0 * pow(((au_wk * vu) - (mu_wk * yu_wk * vd)), 2.0))));

    // Down-type squark mass eigenstate eigenvalues
    double m_sbot_1sq = (0.5)\
        * (mQ3_sq_wk + mD3_sq_wk + (2.0 * mymbsq) + Delta_sdL + Delta_sdR
           - sqrt(pow((mQ3_sq_wk + Delta_sdL - mD3_sq_wk - Delta_sdR), 2.0)
                  + (4.0 * pow(((ab_wk * vd) - (mu_wk * yb_wk * vu)), 2.0))));
    double m_sbot_2sq = (0.5)\
        * (mQ3_sq_wk + mD3_sq_wk + (2.0 * mymbsq) + Delta_sdL + Delta_sdR
           + sqrt(pow((mQ3_sq_wk + Delta_sdL - mD3_sq_wk - Delta_sdR), 2.0)
                  + (4.0 * pow(((ab_wk * vd) - (mu_wk * yb_wk * vu)), 2.0))));
    double m_sstrange_1sq = (0.5)\
        * (mQ2_sq_wk + mD2_sq_wk + (2.0 * mymssq) + Delta_sdL + Delta_sdR
           - sqrt(pow((mQ2_sq_wk + Delta_sdL - mD2_sq_wk - Delta_sdR), 2.0)
                  + (4.0 * pow(((as_wk * vd) - (mu_wk * ys_wk * vu)), 2.0))));
    double m_sstrange_2sq = (0.5)\
        * (mQ2_sq_wk + mD2_sq_wk + (2.0 * mymssq) + Delta_sdL + Delta_sdR
           + sqrt(pow((mQ2_sq_wk + Delta_sdL - mD2_sq_wk - Delta_sdR), 2.0)
                  + (4.0 * pow(((as_wk * vd) - (mu_wk * ys_wk * vu)), 2.0))));
    double m_sdown_1sq = (0.5)\
        * (mQ1_sq_wk + mD1_sq_wk + (2.0 * mymdsq) + Delta_sdL + Delta_sdR
           - sqrt(pow((mQ1_sq_wk + Delta_sdL - mD1_sq_wk - Delta_sdR), 2.0)
                  + (4.0 * pow(((ad_wk * vd) - (mu_wk * yd_wk * vu)), 2.0))));
    double m_sdown_2sq = (0.5)\
        * (mQ1_sq_wk + mD1_sq_wk + (2.0 * mymdsq) + Delta_sdL + Delta_sdR
           + sqrt(pow((mQ1_sq_wk + Delta_sdL - mD1_sq_wk - Delta_sdR), 2.0)
                  + (4.0 * pow(((ad_wk * vd) - (mu_wk * yd_wk * vu)), 2.0))));

    // Slepton mass eigenstate eigenvalues
    double m_stau_1sq = (0.5)\
        * (mL3_sq_wk + mE3_sq_wk + (2.0 * mymtausq) + Delta_seL + Delta_seR
           - sqrt(pow((mL3_sq_wk + Delta_seL - mE3_sq_wk - Delta_seR), 2.0)
                  + (4.0 * pow(((atau_wk * vd) - (mu_wk * ytau_wk * vu)), 2.0))));
    double m_stau_2sq = (0.5)\
        * (mL3_sq_wk + mE3_sq_wk + (2.0 * mymtausq) + Delta_seL + Delta_seR
           + sqrt(pow((mL3_sq_wk + Delta_seL - mE3_sq_wk - Delta_seR), 2.0)
                  + (4.0 * pow(((atau_wk * vd) - (mu_wk * ytau_wk * vu)), 2.0))));
    double m_smu_1sq = (0.5)\
        * (mL2_sq_wk + mE2_sq_wk + (2.0 * mymmusq) + Delta_seL + Delta_seR
           - sqrt(pow((mL2_sq_wk + Delta_seL - mE2_sq_wk - Delta_seR), 2.0)
                  + (4.0 * pow(((amu_wk * vd) - (mu_wk * ymu_wk * vu)), 2.0))));
    double m_smu_2sq = (0.5)\
        * (mL2_sq_wk + mE2_sq_wk + (2.0 * mymmusq) + Delta_seL + Delta_seR
           + sqrt(pow((mL2_sq_wk + Delta_seL - mE2_sq_wk - Delta_seR), 2.0)
                  + (4.0 * pow(((amu_wk * vd) - (mu_wk * ymu_wk * vu)), 2.0))));
    double m_se_1sq = (0.5)\
        * (mL1_sq_wk + mE1_sq_wk + (2.0 * mymesq) + Delta_seL + Delta_seR
           - sqrt(pow((mL1_sq_wk + Delta_seL - mE1_sq_wk - Delta_seR), 2.0)
                  + (4.0 * pow(((ae_wk * vd) - (mu_wk * ye_wk * vu)), 2.0))));
    double m_se_2sq = (0.5)\
        * (mL1_sq_wk + mE1_sq_wk + (2.0 * mymesq) + Delta_seL + Delta_seR
           + sqrt(pow((mL1_sq_wk + Delta_seL - mE1_sq_wk - Delta_seR), 2.0)
                  + (4.0 * pow(((ae_wk * vd) - (mu_wk * ye_wk * vu)), 2.0))));

    // Chargino mass eigenstate eigenvalues
    double msC1sq = (0.5)\
        * (pow(M2_wk, 2.0) + mu_wk_sq + (2.0 * m_w_sq)
           - sqrt(pow(pow(M2_wk, 2.0) + mu_wk_sq
                      + (2.0 * m_w_sq), 2.0)
                  - (4.0 * pow((mu_wk * M2_wk)
                               - (m_w_sq * sin2b), 2.0))));
    double msC2sq = (0.5)\
        * (pow(M2_wk, 2.0) + mu_wk_sq + (2.0 * m_w_sq)
           + sqrt(pow(pow(M2_wk, 2.0) + mu_wk_sq
                      + (2.0 * m_w_sq), 2.0)
                  - (4.0 * pow((mu_wk * M2_wk)
                               - (m_w_sq * sin2b), 2.0))));

    // Neutralino mass eigenstate eigenvalues
    Eigen::Matrix<double, 4, 4> neut_mass_mat(4, 4);
    neut_mass_mat << M1_wk, 0.0, (-1.0) * gpr_wk * vd / sqrt(2.0), gpr_wk * vu / sqrt(2.0),
                0.0, M2_wk, g2_wk * vd / sqrt(2.0), (-1.0) * g2_wk * vu / sqrt(2.0),
                (-1.0) * gpr_wk * vd / sqrt(2.0), g2_wk * vd / sqrt(2.0), 0.0, (-1.0) * mu_wk,
                gpr_wk * vu / sqrt(2.0), (-1.0) * g2_wk * vu / sqrt(2.0), (-1.0) * mu_wk, 0.0;
    Eigen::EigenSolver<Eigen::Matrix<double, 4, 4>> solver(neut_mass_mat);
    Eigen::Matrix<double, 4, 1> my_neut_mass_eigvals = solver.eigenvalues().real();
    Eigen::Matrix<double, 4, 4> my_neut_mass_eigvecs = solver.eigenvectors().real();
    Eigen::Matrix<double, 4, 1> mneutrsq = my_neut_mass_eigvals.array().square();
    sort(mneutrsq.data(), mneutrsq.data() + mneutrsq.size());

    vector<double> eigval_vector(my_neut_mass_eigvals.data(), my_neut_mass_eigvals.data() + my_neut_mass_eigvals.size());
    sort(eigval_vector.begin(), eigval_vector.end(), [](double a, double b) {
        return abs(a) < abs(b);
    });

    double msN1 = eigval_vector[0];
    double msN2 = eigval_vector[1];
    double msN3 = eigval_vector[2];
    double msN4 = eigval_vector[3];
    //cout << "msN1 = " << msN1 << "\nmsN2 = " << msN2 <<  "\nmsN3 = " << msN3 <<  "\nmsN4 = " << msN4 << endl;

    double msN1sq = mneutrsq[0];
    double msN2sq = mneutrsq[1];
    double msN3sq = mneutrsq[2];
    double msN4sq = mneutrsq[3];
    
    // Neutral Higgs doublet mass eigenstate running squared masses
    double mh0sq = (0.5)\
        * ((mA0sq) + (mz_q_sq)
           - sqrt(pow(mA0sq - mz_q_sq, 2.0) + (4.0 * mz_q_sq * mA0sq * pow(sin(2.0 * beta_wk), 2.0))));
    double mH0sq = (0.5)\
        * ((mA0sq) + (mz_q_sq)
           + sqrt(pow(mA0sq - mz_q_sq, 2.0) + (4.0 * mz_q_sq * mA0sq * pow(sin(2.0 * beta_wk), 2.0))));

    double Deltamgl_gluon_gluino = (((3.0 * pow(g3_wk, 2.0)) / (16.0 * pow(M_PI, 2.0)))
                             * (5.0 + (3.0 * log(pow((myQ / M3_wk) , 2.0)))));
    
    double Deltamgl_quark_squark = ((-3.0) * pow(g3_wk, 2.0) / (4.0 * pow(M_PI, 2.0))) * PVB1(M3_wk, mQ1_sq_wk, myQ);
    double Deltamgl_SUSY = Deltamgl_gluon_gluino + Deltamgl_quark_squark;
    double m_gluino = M3_wk * (1.0 - Deltamgl_SUSY);

    /* Output order:
     {0: mu^2, 1: mst1^2, 2: mst2^2, 3: msc1^2, 4: msc2^2, 5: msu1^2, 6: msu2^2, 7: msb1^2, 8: msb2^2,
      9: mss1^2, 10: mss2^2, 11: msd1^2, 12: msd2^2, 13: mstau1^2, 14: mstau2^2, 15: msmu1^2, 16: msmu2^2,
      17: mse1^2, 18: mse2^2, 19: m_Chargino_1, 20: m_chargino_2, 21: m_neutralino_1, 22: m_neutralino_2,
      23: m_neutralino_3, 24: m_neutralino_4, 25: m_gluino, 26: mA0^2, 27: mH0^2, 28: mHpm^2,
      29: m_snu_tau^2, 30: m_snu_mu^2, 31: m_snu_e^2}
    */
    vector<double> masses = {mu_wk * mu_wk, m_stop_1sq, m_stop_2sq, m_scharm_1sq, m_scharm_2sq,
                             m_sup_1sq, m_sup_2sq, m_sbot_1sq, m_sbot_2sq, m_sstrange_1sq,
                             m_sstrange_2sq, m_sdown_1sq, m_sdown_2sq, m_stau_1sq,
                             m_stau_2sq, m_smu_1sq, m_smu_2sq, m_se_1sq, m_se_2sq, copysign(sqrt(msC1sq), msC1sq),
                             copysign(sqrt(msC2sq), msC2sq), copysign(sqrt(msN1sq), msN1sq), copysign(sqrt(msN2sq), msN2sq),
                             copysign(sqrt(msN3sq), msN3sq), copysign(sqrt(msN4sq), msN4sq), m_gluino, mA0sq, mH0sq, mH_pmsq,
                             mstauneutsq, msmuneutsq, mselecneutsq};

    return masses;
}