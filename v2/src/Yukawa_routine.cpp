#include <vector>
#include <cmath>
#include <iostream>
#include "Yukawa_routine.hpp"
#include "Gauge_routine.hpp"
#include <gsl/gsl_roots.h>
#include <gsl/gsl_errno.h>

using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double B0_PV_0mom(double m1, double m2) {
	double B0_M = max(m1, m2);
	double B0_m = min(m1, m2);
	return (((-1.0) * log(pow((B0_M / 91.1876), 2.0))) + 1.0 + ((pow(B0_m, 2.0) / (pow(B0_m, 2.0) - pow(B0_M, 2.0))) * (log(pow((B0_M / B0_m), 2.0)))));
}

double B1_PV_0mom(double m1, double m2) {
	double B1_M = max(m1, m2);
	double B1_m = min(m1, m2);
	double B1_x = pow((m2 / m1), 2.0);

	double condterm;
	
	if (B1_x > 1.0) {
		condterm = log(B1_x);
	}
	else {
		condterm = 0.0;
	}
	return (0.5 * (((-1.0) * log(pow((B1_M / 91.1876), 2.0)))
				   + 0.5 + (1.0 / (1.0 - B1_x)) + (log(B1_x) / (pow((1.0 - B1_x), 2.0))) - condterm));
}

double topMass_Solver(const double& tan_beta, const double& m_gluino, const double& m_stop_1, const double& m_stop_2,
					  const double& g2_mZ, const double& g3_mZ, const double& theta_stop) {
	// XtTanb_SUSY = (at/yt)*tanb-mu might need to be checked
	// Needed parameters
	double mtpole = 173.2;
	double g2_2 = pow(g2_mZ, 2.0);
	double g3_2 = pow(g3_mZ, 2.0);
	double zeta3 = 1.2020569031595942;

	// Get mt corrections from SM and SUSY
	double Deltamt_QCD_1l = (g3_2 / (12.0 * pow(M_PI, 2.0))) * (5.0 + (3.0 * log(pow((91.1876 / 173.2), 2.0))));
	double Deltamt_QCD_2l = ((pow(g3_2, 2.0) / (4608.0 * pow(M_PI, 4.0)))
		* ((396.0 * pow(log(pow((173.2 / 91.1876), 2.0)), 2.0))
			- (1476.0 * log(pow((173.2 / 91.1876), 2.0))) - (48.0 * zeta3) + 2011.0 + ((16.0 * pow(M_PI, 2.0)) * (1.0 + log(4.0))))) - (pow(Deltamt_QCD_1l, 2.0));
	double Deltamt_stop_gl = (((-1.0) * g3_2 / (12.0 * pow(M_PI, 2.0)))) * (B1_PV_0mom(m_gluino, m_stop_1) + B1_PV_0mom(m_gluino, m_stop_2)
		- (sin(2.0 * theta_stop) * (m_gluino / 173.2) * (B0_PV_0mom(m_gluino, m_stop_1) - B0_PV_0mom(m_gluino, m_stop_2))));
	// Calculate running mt and return
	double mt_solution = mtpole / (1.0 + Deltamt_QCD_1l + Deltamt_QCD_2l + Deltamt_stop_gl);
	return mt_solution;

}

struct topYukSolParams {
	double tan_beta;
	double m_gluino;
	double m_stop_1;
	double m_stop_2;
	double g2_mZ;
	double g3_mZ;
	double theta_stop;
};

double num_top_Solver(const double& tan_beta, const double& m_gluino, const double& m_stop_1, const double& m_stop_2,
					  const double& g2_mZ, const double& g3_mZ, const double& theta_stop) {
	// XtTanb_SUSY = (at/yt)*tanb-mu might need to be checked
	// Needed parameters
	double mtpole = 173.2;
	double g2_2 = pow(g2_mZ, 2.0);
	double g3_2 = pow(g3_mZ, 2.0);
	double zeta3 = 1.2020569031595942;

	// Get mt corrections from SM and SUSY
	double Deltamt_QCD_1l = (g3_2 / (12.0 * pow(M_PI, 2.0))) * (5.0 + (3.0 * log(pow((91.1876 / 173.2), 2.0))));
	double Deltamt_QCD_2l = ((pow(g3_2, 2.0) / (4608.0 * pow(M_PI, 4.0)))
		* ((396.0 * pow(log(pow((173.2 / 91.1876), 2.0)), 2.0))
			- (1476.0 * log(pow((173.2 / 91.1876), 2.0))) - (48.0 * zeta3) + 2011.0 + ((16.0 * pow(M_PI, 2.0)) * (1.0 + log(4.0))))) - (pow(Deltamt_QCD_1l, 2.0));
	double Deltamt_stop_gl = (((-1.0) * g3_2 / (12.0 * pow(M_PI, 2.0)))) * (B1_PV_0mom(m_gluino, m_stop_1) + B1_PV_0mom(m_gluino, m_stop_2)
		- (sin(2.0 * theta_stop) * (m_gluino / 173.2) * (B0_PV_0mom(m_gluino, m_stop_1) - B0_PV_0mom(m_gluino, m_stop_2))));
	// Calculate running mt and return
	double mt_solution = mtpole / (1.0 + Deltamt_QCD_1l + Deltamt_QCD_2l + Deltamt_stop_gl);
	return mt_solution;

}

struct botYukSolParams {
	double tan_beta;
	double m_gluino;
	double m_stop_1;
	double m_stop_2;
	double m_sbot_1;
	double m_sbot_2;
	double m_charg_1;
	double m_charg_2;
	double g2_mZ;
	double g3_mZ;
	double theta_stop;
	double theta_sbot;
};

double botMass_Solver(const double& tan_beta, const double& m_gluino, const double& m_stop_1, const double& m_stop_2, const double& m_sbot_1, const double& m_sbot_2, const double& m_chargino_1, const double& m_chargino_2,
				 	  const double& g2_mZ, const double& g3_mZ, const double& theta_stop, const double& theta_sbot, const double& yt, const double& atmZ, const double& mu) {
	double g2_2 = pow(g2_mZ, 2.0);
	double g3_2 = pow(g3_mZ, 2.0);
	double mb_DRbar_SM_mZ = 2.83;
	
	// Now get mb corrections from SM and SUSY
	double Deltamb_sbot_gl = (((-1.0) * g3_2 / (12.0 * pow(M_PI, 2.0)))) * (B1_PV_0mom(m_gluino, m_sbot_1) + B1_PV_0mom(m_gluino, m_sbot_2)
																			 - (sin(2.0 * theta_sbot) * (m_gluino / 173.2) * (B0_PV_0mom(m_gluino, m_sbot_1) - B0_PV_0mom(m_gluino, m_sbot_2))));
	double Deltamb_sbot_chargino = (((g2_2 / (16.0 * pow(M_PI, 2.0)))
									 * ((m_chargino_1 * m_chargino_2 * tan_beta / (pow(m_chargino_2, 2.0) - pow(m_chargino_1, 2.0)))
										* ((pow(cos(theta_stop), 2.0) * (B0_PV_0mom(m_chargino_1, m_stop_1) - B0_PV_0mom(m_chargino_2, m_stop_1)))
										   + (pow(sin(theta_stop), 2.0) * (B0_PV_0mom(m_chargino_1, m_stop_2) - B0_PV_0mom(m_chargino_2, m_stop_2)))))));
	Deltamb_sbot_chargino += ((yt / (16.0 * pow(M_PI, 2.0))) * m_chargino_2 * (((atmZ * tan_beta) + (yt * mu)) / (pow(m_stop_1, 2.0) - pow(m_stop_2, 2.0)))
							  * (B0_PV_0mom(m_chargino_2, m_stop_1) - B0_PV_0mom(m_chargino_2, m_stop_2)));
	double mb_solution = mb_DRbar_SM_mZ * (1.0 - Deltamb_sbot_gl - Deltamb_sbot_chargino);
	return mb_solution;
}

struct tauYukSolParams {
	double tan_beta;
	double m_chargino_1;
	double m_chargino_2;
	double m_tau_sneutrino;
	double g2_mZ;
	double g3_mZ;
};

double tauMass_Solver(const double& tan_beta, const double& g1_mZ, const double& g2_mZ, const double& g3_mZ, const double& m_chargino_1, const double& m_chargino_2, const double& m_tau_sneutrino) {
	double g1_2 = pow(g1_mZ, 2.0);
	double g2_2 = pow(g2_mZ, 2.0);
	double g3_2 = pow(g3_mZ, 2.0);
	double mtau_MSbar_SM_mZ = 1.7463;
	double DRbar_conv = (3.0 / (128.0 * pow(M_PI, 2.0))) * (g1_2 - g2_2);
	double Deltamtau = (g2_2 / (16.0 * pow(M_PI, 2.0))) * (m_chargino_1 * m_chargino_2 * tan_beta / (pow(m_chargino_2, 2.0) - pow(m_chargino_1, 2.0)))\
		* (B0_PV_0mom(m_chargino_1, m_tau_sneutrino) - B0_PV_0mom(m_chargino_2, m_tau_sneutrino));
	double mtau_solution = mtau_MSbar_SM_mZ * (1.0 + Deltamtau);
	return mtau_solution;
}

std::vector<double> get_init_yukawas(const double& tan_beta, const double& g1_mZ, const double& g2_mZ, const double& g3_mZ) {
	const double mtpole = 173.2;
	double gpr_2 = 3.0 * pow(g1_mZ, 2.0) / 5.0;
	double g2_2 = pow(g2_mZ, 2.0);
	double g3_2 = pow(g3_mZ, 2.0);
	double vHiggs_mZ = sqrt(2.0) * 91.1876 / sqrt(gpr_2 + g2_2);
	double beta_mZ = atan(tan_beta);
	double vu_mZ = vHiggs_mZ * sin(beta_mZ);
	double vd_mZ = vHiggs_mZ * cos(beta_mZ);
	double zeta3 = 1.2020569031595942;

	// Get mt corrections from SM
	double Deltamt_QCD_1l = (g3_2 / (12.0 * pow(M_PI, 2.0))) * (5.0 + (3.0 * log(pow((91.1876 / 173.2), 2.0))));
	double Deltamt_QCD_2l = ((pow(g3_2, 2.0) / (4608.0 * pow(M_PI, 4.0)))
		* ((396.0 * pow(log(pow((173.2 / 91.1876), 2.0)), 2.0))
			- (1476.0 * log(pow((173.2 / 91.1876), 2.0))) - (48.0 * zeta3) + 2011.0 + ((16.0 * pow(M_PI, 2.0)) * (1.0 + log(4.0))))) - (pow(Deltamt_QCD_1l, 2.0));
	// Calculate running mt and return
	double mt_solution = mtpole / (1.0 + Deltamt_QCD_1l + Deltamt_QCD_2l);
	double mb_DRbar_SM_mZ = 2.83;
	double mtau_MSbar_SM_mZ = 1.7463;

	const double mymc = 0.619;
	const double mymu = 1.27E-3;
	const double myms = 0.055;
	const double mymd = 2.9e-3;
	const double mymmu = 0.1027181359;
	const double myme = 4.86570161e-4;


	double yt_eval = mt_solution / vu_mZ;
	double yc_eval = mymc / vu_mZ;
	double yu_eval = mymu / vu_mZ;
	double yb_eval = mb_DRbar_SM_mZ / vd_mZ;
	double ys_eval = myms / vd_mZ;
	double yd_eval = mymd / vd_mZ;
	double ytau_eval = mtau_MSbar_SM_mZ / vd_mZ;
	double ymu_eval = mymmu / vd_mZ;
	double ye_eval = myme / vd_mZ;
	vector<double> inityuks = { yt_eval, yc_eval, yu_eval, yb_eval, ys_eval, yd_eval, ytau_eval, ymu_eval, ye_eval };
	return inityuks;
}

std::vector<double> get_yukawa_couplings(const double& tan_beta, const double& m_gluino, const double& m_stop_1, const double& m_stop_2, const double& m_sbot_1, const double& m_sbot_2,
										 const double& m_tau_sneutrino, const double& m_chargino_1, const double& m_chargino_2, const double& g1_mZ, const double& g2_mZ,
										 const double& g3_mZ, const double& theta_stop, const double& theta_sbot, const double& yt, const double& atmZ, const double& mu) {
	double currmt = topMass_Solver(tan_beta, m_gluino, m_stop_1, m_stop_2, g2_mZ, g3_mZ, theta_stop);
	double currmb = botMass_Solver(tan_beta, m_gluino, m_stop_1, m_stop_2, m_sbot_1, m_sbot_2, m_chargino_1, m_chargino_2, g2_mZ, g3_mZ, theta_stop, theta_sbot, yt, atmZ, mu);
	double currmtau = tauMass_Solver(tan_beta, g1_mZ, g2_mZ, g3_mZ, m_chargino_1, m_chargino_2, m_tau_sneutrino);
	const double mymc = 0.619;
	const double mymu = 1.27E-3;
	const double myms = 0.055;
	const double mymd = 2.9e-3;
	const double mymmu = 0.1027181359;
	const double myme = 4.86570161e-4;
	double gpr_2 = 3.0 * pow(g1_mZ, 2.0) / 5.0;
	double g2_2 = pow(g2_mZ, 2.0);
	double g3_2 = pow(g3_mZ, 2.0);
	double vHiggs_mZ = sqrt(2.0) * 91.1876 / sqrt(gpr_2 + g2_2);
	double beta_mZ = atan(tan_beta);
	double vu_mZ = vHiggs_mZ * sin(beta_mZ);
	double vd_mZ = vHiggs_mZ * cos(beta_mZ);

	double yt_eval = currmt / vu_mZ;
	double yc_eval = mymc / vu_mZ;
	double yu_eval = mymu / vu_mZ;
	double yb_eval = currmb / vd_mZ;
	double ys_eval = myms / vd_mZ;
	double yd_eval = mymd / vd_mZ;
	double ytau_eval = currmtau / vd_mZ;
	double ymu_eval = mymmu / vd_mZ;
	double ye_eval = myme / vd_mZ;
	return {yt_eval, yc_eval, yu_eval, yb_eval, ys_eval, yd_eval, ytau_eval, ymu_eval, ye_eval};
}