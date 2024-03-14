// GAUGE_ROUTINE_HPP

#ifndef GAUGE_ROUTINE_HPP
#define GAUGE_ROUTINE_HPP

#include <vector>

struct GCSolParams {
	std::vector<double> pole_higgs_masses;
	std::vector<double> pole_squark_slep_masses;
	std::vector<double> neutralino_masses;
	std::vector<double> chargino_masses;
	std::vector<double> SUSYscale_soft_trilins;
	std::vector<double> yukawas;
	std::vector<double> running_gauge_couplings;
	std::vector<double> running_squark_slep_masses_sq;
	std::vector<double> running_gaugino_masses;
	double tempalphaem;
	double tempalphas;
	double vHiggs;
	double mu;
	double tanbMZ;
	double tanbSUSY;
};

std::vector<double> get_gauge_couplings(double& m_gluino, double& m_Hpm, double& m_stop_1, double& m_stop_2, double& m_scharm_1, double& m_scharm_2,
										double& m_sup_1, double& m_sup_2, double& m_sbottom_1, double& m_sbottom_2, double& m_sstrange_1, double& m_sstrange_2,
										double& m_sdown_1, double& m_sdown_2, double& m_stau_1, double& m_stau_2, double& m_smu_1, double& m_smu_2,
										double& m_selectron_1, double& m_selectron_2, double& m_chargino_1, double& m_chargino_2);
std::vector<double> first_run_gauge_couplings();


#endif