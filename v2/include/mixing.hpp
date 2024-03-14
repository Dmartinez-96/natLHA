// MIXING_HPP

#ifndef MIXING_HPP
#define MIXING_HPP

#include <vector>
#include <complex>

using namespace std;

// Sfermion angles, evaluated at SUSY scale
std::vector<double> ferm_angle_calc(const double& vHiggs, const double& mu, const double& yt, const double& yc, const double& yu, const double& yb, const double& ys, const double& yd, const double& ytau, const double& ymu, const double& ye,
									const double& at, const double& ac, const double& au, const double& ab, const double& as, const double& ad, const double& atau, const double& amu, const double& ae,
									const double& tanb, const double& gpr, const double& g2, const double& mQ1_2, const double& mQ2_2, const double& mQ3_2,
									const double& mL1_2, const double& mL2_2, const double& mL3_2, const double& mU1_2, const double& mU2_2, const double& mU3_2, const double& mD1_2, const double& mD2_2, const double& mD3_2,
									const double& mE1_2, const double& mE2_2, const double& mE3_2);
// Higgs mixing angle, evaluated at SUSY scale
double alpha_angle_calc(const double& tanb, const double& mA0_2);
// Neutralino mixing matrix, evaluated at SUSY scale
std::vector<std::vector<complex<double>>> N_neutralino_calc(const double& vHiggs, const double& mu, const double& tanb, const double& M1, const double& M2, const double& g2, const double& gp);// ,
															//const double& mQ1sq, const double& mL1sq, const double& mAsq);
// Chargino mixing matrices, evaluated at SUSY scale
std::vector<std::vector<double>> U_chargino_calc(const double& vHiggs, const double& mu, const double& tanb, const double& M2, const double& g2);
std::vector<std::vector<double>> V_chargino_calc(const double& vHiggs, const double& mu, const double& tanb, const double& M2, const double& g2);

#endif