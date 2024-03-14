#include <vector>
#include <cmath>
#include <iostream>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_errno.h>
#include "Gauge_routine.hpp"
#include "mixing.hpp"

using namespace std;

std::vector<double> get_gauge_couplings(double& m_gluino, double& m_Hpm, double& m_stop_1, double& m_stop_2, double& m_scharm_1, double& m_scharm_2,
										double& m_sup_1, double& m_sup_2, double& m_sbottom_1, double& m_sbottom_2, double& m_sstrange_1, double& m_sstrange_2,
										double& m_sdown_1, double& m_sdown_2, double& m_stau_1, double& m_stau_2, double& m_smu_1, double& m_smu_2,
										double& m_selectron_1, double& m_selectron_2, double& m_chargino_1, double& m_chargino_2) {
	
	double alphaem_MSbar_MZ = 1.0 / 137.036;
	double Deltaalphaem = ((1.0 / 3.0) + (7.0 * log(80.404 / 91.1876)) - ((16.0 / 9.0) * log(173.2 / 91.1876))
						   - ((1.0 / 3.0) * log(m_Hpm / 91.1876)) - ((4.0 / 9.0) * (log(m_stop_1 / 91.1876) + log(m_stop_2 / 91.1876) + log(m_scharm_1 / 91.1876)
																					+ log(m_scharm_2 / 91.1876) + log(m_sup_1 / 91.1876) + log(m_sup_2 / 91.1876)))
						   - ((1.0 / 9.0) * (log(m_sbottom_1 / 91.1876) + log(m_sbottom_2 / 91.1876) + log(m_sstrange_1 / 91.1876) + log(m_sstrange_2 / 91.1876)
						 	 				 + log(m_sdown_1 / 91.1876) + log(m_sdown_2 / 91.1876)))
						   - ((1.0 / 3.0) * (log(m_stau_1 / 91.1876) + log(m_stau_2 / 91.1876) + log(m_smu_1 / 91.1876) + log(m_smu_2 / 91.1876) + log(m_selectron_1 / 91.1876)
											 + log(m_selectron_2 / 91.1876)))
						   - ((4.0 / 3.0) * (log(m_chargino_1 / 91.1876) + log(m_chargino_2 / 91.1876))));
	double alphaem_DRbar_MZ;
	
	alphaem_DRbar_MZ = alphaem_MSbar_MZ / (1.0 - 0.0682 - ((alphaem_MSbar_MZ / (2.0 * M_PIl)) * Deltaalphaem));
	
	double alphas_MSbar_MZ = 0.1185;
	double Deltaalphas = (0.5 - (2.0 * log(173.2 / 91.1876) / 3.0) - (2.0 * log(m_gluino / 91.1876))
							- ((1.0 / 6.0) * (log(m_stop_1 / 91.1876) + log(m_stop_2 / 91.1876) + log(m_scharm_1 / 91.1876)
								+ log(m_scharm_2 / 91.1876) + log(m_sup_1 / 91.1876) + log(m_sup_2 / 91.1876)
								+ log(m_sbottom_1 / 91.1876) + log(m_sbottom_2 / 91.1876) + log(m_sstrange_1 / 91.1876) + log(m_sstrange_2 / 91.1876)
								+ log(m_sdown_1 / 91.1876) + log(m_sdown_2 / 91.1876))));
	double alphas_DRbar_MZ;
	if (Deltaalphas > 0) {
		alphas_DRbar_MZ = (M_PIl + (sqrt(M_PIl) * sqrt(M_PIl - (2.0 * alphas_MSbar_MZ * Deltaalphas)))) / Deltaalphas;
	}
	else {
		alphas_DRbar_MZ = (M_PIl - (sqrt(M_PIl) * sqrt(M_PIl - (2.0 * alphas_MSbar_MZ * Deltaalphas)))) / Deltaalphas;
	}
	std::vector<double> retalphs = { alphaem_DRbar_MZ, alphas_DRbar_MZ };
	return retalphs;
}

std::vector<double> first_run_gauge_couplings() {
	double alphaem_MSbar_MZ = (1.0 / 137.036);
	double alphas_MSbar_MZ = 0.1185;
	double thetaW_OS = asin(sqrt(1.0 - pow((80.404 / 91.1876), 2.0)));
	double evaluated_g1_MZ = sqrt(20.0 * M_PIl * alphaem_MSbar_MZ / 3.0) / cos(thetaW_OS);
	double evaluated_g2_MZ = sqrt(4.0 * M_PIl * alphaem_MSbar_MZ) / sin(thetaW_OS);
	double evaluated_g3_MZ = sqrt(4.0 * M_PIl * alphas_MSbar_MZ);
	std::vector<double> returngauges = { evaluated_g1_MZ, evaluated_g2_MZ, evaluated_g3_MZ };
	return returngauges;
}