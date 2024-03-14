// PASS_VELT_HPP

#ifndef PASS_VELT_HPP
#define PASS_VELT_HPP

#include <cmath>
#include <vector>
#include <complex>

using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

complex<double> PV_A0(const double& msq);
complex<double> PV_B0_genp(const double& psq, const double& m1sq, const double& m2sq);
complex<double> PV_B1_genp(const double& psq, const double& m1sq, const double& m2sq);
complex<double> PV_B22_genp(const double& psq, const double& m1sq, const double& m2sq);
complex<double> PV_F_genp(const double& psq, const double& m1sq, const double& m2sq);
complex<double> PV_G_genp(const double& psq, const double& m1sq, const double& m2sq);
complex<double> PV_H_genp(const double& psq, const double& m1sq, const double& m2sq);
complex<double> PV_sB22_genp(const double& psq, const double& m1sq, const double& m2sq);

complex<double> PV_A0_genQ(const double& msq, const double& genQ);
complex<double> PV_B0_genp_genQ(const double& psq, const double& m1sq, const double& m2sq, const double& genQ);
complex<double> PV_B1_genp_genQ(const double& psq, const double& m1sq, const double& m2sq, const double& genQ);
complex<double> PV_B22_genp_genQ(const double& psq, const double& m1sq, const double& m2sq, const double& genQ);
complex<double> PV_F_genp_genQ(const double& psq, const double& m1sq, const double& m2sq, const double& genQ);
complex<double> PV_G_genp_genQ(const double& psq, const double& m1sq, const double& m2sq, const double& genQ);
complex<double> PV_H_genp_genQ(const double& psq, const double& m1sq, const double& m2sq, const double& genQ);
complex<double> PV_sB22_genp_genQ(const double& psq, const double& m1sq, const double& m2sq, const double& genQ);

#endif