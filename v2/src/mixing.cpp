#include <iostream>
#include <cmath>
#include <vector>
#include <complex>
#include <numeric>
#include <eigen3/Eigen/Dense>
#include <algorithm>
#include <iostream>
#include "mixing.hpp"
#include "Pass_Velt.hpp"

using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Sfermion angles, evaluated at SUSY scale
std::vector<double> ferm_angle_calc(const double& vHiggs, const double& mu, const double& yt, const double& yc, const double& yu, const double& yb, const double& ys, const double& yd, const double& ytau, const double& ymu, const double& ye,
									const double& at, const double& ac, const double& au, const double& ab, const double& as, const double& ad, const double& atau, const double& amu, const double& ae,
									const double& tanb, const double& gpr, const double& g2, const double& mQ1_2, const double& mQ2_2, const double& mQ3_2,
									const double& mL1_2, const double& mL2_2, const double& mL3_2, const double& mU1_2, const double& mU2_2, const double& mU3_2, const double& mD1_2, const double& mD2_2, const double& mD3_2,
									const double& mE1_2, const double& mE2_2, const double& mE3_2) {
	const double vu = vHiggs * sin(atan(tanb));
	const double vd = vHiggs * cos(atan(tanb));
	const double theta_weinb = atan(gpr / g2);
	std::vector<double> ret_angles;
	const double tan2th_t = ((2.0 * vu * (at + (mu * yt / tanb))) / (mQ3_2 - mU3_2 + ((0.5 - ((4.0 / 3.0) * pow(sin(theta_weinb), 2.0))) * (pow(91.1876, 2.0) * cos(2.0 * atan(tanb))))));
	const double tan2th_c = ((2.0 * vu * (ac + (mu * yc / tanb))) / (mQ2_2 - mU2_2 + ((0.5 - ((4.0 / 3.0) * pow(sin(theta_weinb), 2.0))) * (pow(91.1876, 2.0) * cos(2.0 * atan(tanb))))));
	const double tan2th_u = ((2.0 * vu * (au + (mu * yu / tanb))) / (mQ1_2 - mU1_2 + ((0.5 - ((4.0 / 3.0) * pow(sin(theta_weinb), 2.0))) * (pow(91.1876, 2.0) * cos(2.0 * atan(tanb))))));
	const double theta_2t = ((tan2th_t >= 0) ? atan(tan2th_t) : atan(tan2th_t) + M_PI);
	const double theta_2c = ((tan2th_c >= 0) ? atan(tan2th_c) : atan(tan2th_c) + M_PI);
	const double theta_2u = ((tan2th_u >= 0) ? atan(tan2th_u) : atan(tan2th_u) + M_PI);
	const double theta_t = theta_2t / 2.0;
	const double theta_c = theta_2c / 2.0;
	const double theta_u = theta_2u / 2.0;

	const double tan2th_b = ((2.0 * vd * (ab + (mu * yb * tanb))) / (mQ3_2 - mD3_2 + (((-0.5) + ((2.0 / 3.0) * pow(sin(theta_weinb), 2.0))) * (pow(91.1876, 2.0) * cos(2.0 * atan(tanb))))));
	const double tan2th_s = ((2.0 * vd * (as + (mu * ys * tanb))) / (mQ2_2 - mD2_2 + (((-0.5) + ((2.0 / 3.0) * pow(sin(theta_weinb), 2.0))) * (pow(91.1876, 2.0) * cos(2.0 * atan(tanb))))));
	const double tan2th_d = ((2.0 * vd * (ad + (mu * yd * tanb))) / (mQ1_2 - mD1_2 + (((-0.5) + ((2.0 / 3.0) * pow(sin(theta_weinb), 2.0))) * (pow(91.1876, 2.0) * cos(2.0 * atan(tanb))))));
	const double theta_2b = ((tan2th_b >= 0) ? atan(tan2th_b) : atan(tan2th_b) + M_PI);
	const double theta_2s = ((tan2th_s >= 0) ? atan(tan2th_s) : atan(tan2th_s) + M_PI);
	const double theta_2d = ((tan2th_d >= 0) ? atan(tan2th_d) : atan(tan2th_d) + M_PI);
	const double theta_b = theta_2b / 2.0;
	const double theta_s = theta_2s / 2.0;
	const double theta_d = theta_2d / 2.0;

	const double tan2th_tau = ((2.0 * vd * (atau + (mu * ytau * tanb))) / (mL3_2 - mE3_2 + (((-0.5) + (2.0 * pow(sin(theta_weinb), 2.0))) * (pow(91.1876, 2.0) * cos(2.0 * atan(tanb))))));
	const double tan2th_mu = ((2.0 * vd * (amu + (mu * ymu * tanb))) / (mL2_2 - mE2_2 + (((-0.5) + (2.0 * pow(sin(theta_weinb), 2.0))) * (pow(91.1876, 2.0) * cos(2.0 * atan(tanb))))));
	const double tan2th_e = ((2.0 * vd * (ae + (mu * ye * tanb))) / (mL1_2 - mE1_2 + (((-0.5) + (2.0 * pow(sin(theta_weinb), 2.0))) * (pow(91.1876, 2.0) * cos(2.0 * atan(tanb))))));
	const double theta_2tau = ((tan2th_tau >= 0) ? atan(tan2th_tau) : atan(tan2th_tau) + M_PI);
	const double theta_2mu = ((tan2th_mu >= 0) ? atan(tan2th_mu) : atan(tan2th_mu) + M_PI);
	const double theta_2e = ((tan2th_e >= 0) ? atan(tan2th_e) : atan(tan2th_e) + M_PI);
	const double theta_tau = 0.5 * theta_2tau;
	const double theta_mu = 0.5 * theta_2mu;
	const double theta_e = 0.5 * theta_2e;

	ret_angles = { theta_t, theta_c, theta_u, theta_b, theta_s, theta_d, theta_tau, theta_mu, theta_e };
	return ret_angles;
}

// Higgs mixing angle, evaluated at SUSY scale
double alpha_angle_calc(const double& tanb, const double& mA0_2) {
	double alpha_eval = 0.5 * atan(((mA0_2 + pow(91.1876, 2.0)) / (mA0_2 - pow(91.1876, 2.0))) * tan(2.0 * atan(tanb)));
	return alpha_eval;
}

// Neutralino mixing matrix, evaluated at SUSY scale
std::vector<std::vector<complex<double>>> N_neutralino_calc(const double& vHiggs, const double& mu, const double& tanb, const double& M1, const double& M2, const double& g2, const double& gp) {
	Eigen::MatrixXcd neutMassMat(4, 4);
	double betaval = atan(tanb);
	const double loc_thW = atan(gp / g2);
	double sW = sqrt(1.0 - pow((80.404 / 91.1876), 2.0));
	double cW = 80.404 / 91.1876; 
	double mZ = 91.1876;
	neutMassMat << M1, 0.0, (-1.0) * gp * cos(betaval) * vHiggs / sqrt(2.0), gp * sin(betaval) * vHiggs / sqrt(2.0),
					0.0, M2, g2 * cos(betaval) * vHiggs / sqrt(2.0), (-1.0) * g2 * sin(betaval) * vHiggs / sqrt(2.0),
					(-1.0) * gp * cos(betaval) * vHiggs / sqrt(2.0), g2 * vHiggs * cos(betaval) / sqrt(2.0), 0.0, mu,
					gp * sin(betaval) * vHiggs / sqrt(2.0), (-1.0) * g2 * vHiggs * sin(betaval) / sqrt(2.0), mu, 0.0;
	Eigen::ComplexEigenSolver<Eigen::MatrixXcd> solver(neutMassMat);
	Eigen::VectorXcd eigenvalues = solver.eigenvalues();
	Eigen::MatrixXcd eigenvectors = solver.eigenvectors();
    std::vector<int> indices(eigenvalues.size());
	std::iota(indices.begin(), indices.end(), 0);

	std::sort(indices.begin(), indices.end(), [&](int i, int j) {
		return abs(eigenvalues(i)) < abs(eigenvalues(j));
		});

	Eigen::MatrixXcd sortedEigenvectors(eigenvectors.rows(), eigenvectors.cols());
	for (size_t i = 0; i < indices.size(); ++i) {
		sortedEigenvectors.col(i) = eigenvectors.col(indices[i]);
	}
	std::vector<std::vector<complex<double>>> ret_neut_mix_mat(sortedEigenvectors.rows());

	Eigen::MatrixXcd invSortedEVs = sortedEigenvectors.inverse();
	for (int i = 0; i < sortedEigenvectors.rows(); ++i) {
		ret_neut_mix_mat[i].resize(sortedEigenvectors.cols());
		for (int j = 0; j < sortedEigenvectors.cols(); ++j) {
			ret_neut_mix_mat[i][j] = invSortedEVs(i, j);
		}
	}
	return ret_neut_mix_mat;
}

static std::vector<std::vector<std::vector<double>>> UV_chargino_calc(const double& vHiggs, const double& mu, const double& tanb, const double& M2, const double& g2) {
	Eigen::MatrixXd chargmat(2, 2);
	chargmat << M2, g2 * vHiggs * sin(atan(tanb)),
				g2 * vHiggs * cos(atan(tanb)), (-1.0) * mu;
	// Singular value decomposition
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(chargmat, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::MatrixXd U = svd.matrixU().inverse().normalized();
	Eigen::MatrixXd V = svd.matrixV().transpose().normalized();
	std::vector<std::vector<double>> matU = { { U(0, 0), U(0, 1) }, { U(1, 0), U(1, 1) } };
	std::vector<std::vector<double>> matV = { { V(0, 0), V(0, 1) }, { V(1, 0), V(1, 1) } };
	return { matU, matV };
}

// Chargino mixing matrices, evaluated at SUSY scale
std::vector<std::vector<double>> U_chargino_calc(const double& vHiggs, const double& mu, const double& tanb, const double& M2, const double& g2) {
	double betaval = atan(tanb);
	Eigen::MatrixXcd XXdag(2, 2);
	XXdag << pow(M2, 2.0) + (pow(g2 * vHiggs * sin(betaval), 2.0)), g2 * vHiggs * ((M2 * cos(betaval)) + (mu * sin(betaval))),
			 g2 * vHiggs * ((M2 * cos(betaval)) + (mu * sin(betaval))), pow(mu, 2.0) + (pow(g2 * vHiggs * cos(betaval), 2.0));
	Eigen::ComplexEigenSolver<Eigen::MatrixXcd> solver(XXdag);
	Eigen::VectorXcd eigenvalues = solver.eigenvalues().transpose();
	Eigen::MatrixXcd eigenvectors = solver.eigenvectors().normalized();
	std::vector<int> indices(eigenvalues.size());
	std::iota(indices.begin(), indices.end(), 0);

	std::sort(indices.begin(), indices.end(), [&](int i, int j) {
		return abs(eigenvalues(i)) < abs(eigenvalues(j));
		});

	Eigen::MatrixXcd sortedEigenvectors(eigenvectors.rows(), eigenvectors.cols());
	for (size_t i = 0; i < indices.size(); ++i) {
		sortedEigenvectors.col(i) = eigenvectors.col(indices[i]).normalized();
	}

	Eigen::MatrixXcd diagMatrix = sortedEigenvectors.inverse() * XXdag * sortedEigenvectors;

	std::vector<std::vector<double>> ret_Umat(sortedEigenvectors.rows());

	Eigen::MatrixXcd invSortedEVs = sortedEigenvectors.inverse();
	for (int i = 0; i < sortedEigenvectors.rows(); ++i) {
		ret_Umat[i].resize(sortedEigenvectors.cols());
		for (int j = 0; j < sortedEigenvectors.cols(); ++j) {
			ret_Umat[i][j] = real(invSortedEVs(i, j));
        }
	}
	return ret_Umat;
}

std::vector<std::vector<double>> V_chargino_calc(const double& vHiggs, const double& mu, const double& tanb, const double& M2, const double& g2) {
	double betaval = atan(tanb);
	Eigen::MatrixXcd XdagX(2, 2);
	XdagX << pow(M2, 2.0) + (pow(g2 * vHiggs * cos(betaval), 2.0)), g2 * vHiggs * ((M2 * sin(betaval)) + (mu * cos(betaval))),
			 g2 * vHiggs * ((M2 * sin(betaval)) + (mu * cos(betaval))), pow(mu, 2.0) + (pow(g2 * vHiggs * sin(betaval), 2.0));
	Eigen::ComplexEigenSolver<Eigen::MatrixXcd> solver(XdagX);
	Eigen::VectorXcd eigenvalues = solver.eigenvalues().transpose();
	Eigen::MatrixXcd eigenvectors = solver.eigenvectors().normalized();
	std::vector<int> indices(eigenvalues.size());
	std::iota(indices.begin(), indices.end(), 0);

	std::sort(indices.begin(), indices.end(), [&](int i, int j) {
		return abs(eigenvalues(i)) < abs(eigenvalues(j));
		});

	Eigen::MatrixXcd sortedEigenvectors(eigenvectors.rows(), eigenvectors.cols());
	for (size_t i = 0; i < indices.size(); ++i) {
		sortedEigenvectors.col(i) = eigenvectors.col(indices[i]).normalized();
	}

	Eigen::MatrixXcd diagMatrix = sortedEigenvectors.inverse() * XdagX * sortedEigenvectors;

	std::vector<std::vector<double>> ret_Vmat(sortedEigenvectors.rows());

	Eigen::MatrixXcd invSortedEVs = sortedEigenvectors.inverse();
	for (int i = 0; i < sortedEigenvectors.rows(); ++i) {
		ret_Vmat[i].resize(sortedEigenvectors.cols());
		for (int j = 0; j < sortedEigenvectors.cols(); ++j) {
			ret_Vmat[i][j] = real(invSortedEVs(i, j));
			//cout << "Vmix[" << i << "][" << j << "] = " << ret_Vmat[i][j] << endl;
		}
	}
	return ret_Vmat;
}
