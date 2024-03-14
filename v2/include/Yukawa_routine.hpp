// YUKAWA_ROUTINE_HPP

#ifndef YUKAWA_ROUTINE_HPP
#define YUKAWA_ROUTINE_HPP

#include <vector>

double B0_PV_0mom(double m1, double m2);
double B1_PV_0mom(double m1, double m2);

double topMass_Solver(const double& tan_beta, const double& m_gluino, const double& m_stop_1, const double& m_stop_2,
					  const double& g2_mZ, const double& g3_mZ, const double& theta_stop);
double botMass_Solver(const double& tan_beta, const double& m_gluino, const double& m_stop_1, const double& m_stop_2, const double& m_sbot_1, const double& m_sbot_2, const double& m_chargino_1, const double& m_chargino_2,
				 	  const double& g2_mZ, const double& g3_mZ, const double& theta_stop, const double& theta_sbot, const double& yt, const double& atmZ, const double& mu);
double tauMass_Solver(const double& tan_beta, const double& g1_mZ, const double& g2_mZ, const double& g3_mZ, const double& m_chargino_1, const double& m_chargino_2, const double& m_tau_sneutrino);

std::vector<double> get_yukawa_couplings(const double& tan_beta, const double& m_gluino, const double& m_stop_1, const double& m_stop_2, const double& m_sbot_1, const double& m_sbot_2,
										 const double& m_tau_sneutrino, const double& m_chargino_1, const double& m_chargino_2, const double& g1_mZ, const double& g2_mZ,
										 const double& g3_mZ, const double& theta_stop, const double& theta_sbot, const double& yt, const double& atmZ, const double& mu);

std::vector<double> get_init_yukawas(const double& tan_beta, const double& g1_mZ, const double& g2_mZ, const double& g3_mZ);

#endif