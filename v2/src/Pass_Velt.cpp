#include <cmath>
#include <vector>
#include <complex>
#include <string>
#include <iostream>
#include <cstdlib>
#include "Pass_Velt.hpp"

using namespace std;

// Passarino-Veltman functions with generalized external momentum
complex<double> PV_A0(const double& msq) {
	complex<double> PVA0;
	if (msq == 0.0) {
		PVA0 = 0.0;
	}
	else {
		PVA0 = msq * (1.0 - log(msq / pow(91.1876, 2.0)));
	}
	return PVA0;
}

complex<double> PV_A0_genQ(const double& msq, const double& genQ) {
	complex<double> PVA0;
	if (msq == 0.0) {
		PVA0 = 0.0;
	}
	else {
		PVA0 = msq * (1.0 - log(msq / pow(genQ, 2.0)));
	}
	return PVA0;
}

complex<double> PV_B0_genp(const double& psq, const double& m1sq, const double& m2sq) {
	const complex<double> s_term = psq - m2sq + m1sq;
	complex<double> PVB0;
	if (psq == 0.0 && m1sq == 0.0 && m2sq == 0.0) {
		PVB0 = 0.0;
		return PVB0;
	}
	double m1sqp, m2sqp;
	if (m1sq == 0.0) {
		m1sqp = 1.0e-10;
	}
	else {
		m1sqp = m1sq;
	}
	if (m2sq == 0.0) {
		m2sqp = 1.0e-10;
	}
	else {
		m2sqp = m2sq;
	}
	
	if (psq == 0.0) {
		double Mval = max(m1sq, m2sq);
		double mval = min(m1sq, m2sq) + 1.0e-10;
		//double xval = m2sq / m1sq;
		// might need to account for how neutralinos can be negative mass
		PVB0 = 1.0 + log(pow(91.1876, 2.0) / Mval) + ((mval / (mval - Mval)) * log(Mval / mval));
	}
	else {
		const complex<double> radical = pow(s_term, 2.0) - ((4.0 * psq * m1sqp));
		const complex<double> xp = (1.0 / (2.0 * psq)) * (s_term + sqrt(radical));
		const complex<double> xm = (1.0 / (2.0 * psq)) * (s_term - sqrt(radical));
		const complex<double> logterm_xm1 = 1.0 - xm;
		const complex<double> logterm_xm2 = 1.0 - (1.0 / xm);
		const complex<double> logterm_xp1 = 1.0 - xp;
		const complex<double> logterm_xp2 = 1.0 - (1.0 / xp);
		const complex<double> fbxp = log(logterm_xp1) - (xp * log(logterm_xp2)) - 1.0;
		const complex<double> fbxm = log(logterm_xm1) - (xm * log(logterm_xm2)) - 1.0;
		PVB0 = ((-1.0) * (log(psq * pow((1.0 / 91.1876), 2.0)) + fbxp + fbxm));
	}
	return PVB0;
}

complex<double> PV_B0_genp_genQ(const double& psq, const double& m1sq, const double& m2sq, const double& genQ) {
	const complex<double> s_term = psq - m2sq + m1sq;
	complex<double> PVB0;
	if (psq == 0.0 && m1sq == 0.0 && m2sq == 0.0) {
		PVB0 = 0.0;
		return PVB0;
	}
	double m1sqp, m2sqp;
	if (m1sq == 0.0) {
		m1sqp = 1.0e-10;
	}
	else {
		m1sqp = m1sq;
	}
	if (m2sq == 0.0) {
		m2sqp = 1.0e-10;
	}
	else {
		m2sqp = m2sq;
	}

	if (psq == 0.0) {
		double Mval = max(m1sq, m2sq);
		double mval = min(m1sq, m2sq) + 1.0e-10;
		//double xval = m2sq / m1sq;
		// might need to account for how neutralinos can be negative mass
		PVB0 = 1.0 + log(pow(genQ, 2.0) / Mval) + ((mval / (mval - Mval)) * log(Mval / mval));
	}
	else {
		const complex<double> radical = pow(s_term, 2.0) - ((4.0 * psq * m1sqp));
		const complex<double> xp = (1.0 / (2.0 * psq)) * (s_term + sqrt(radical));
		const complex<double> xm = (1.0 / (2.0 * psq)) * (s_term - sqrt(radical));
		const complex<double> logterm_xm1 = 1.0 - xm;
		const complex<double> logterm_xm2 = 1.0 - (1.0 / xm);
		const complex<double> logterm_xp1 = 1.0 - xp;
		const complex<double> logterm_xp2 = 1.0 - (1.0 / xp);
		const complex<double> fbxp = log(logterm_xp1) - (xp * log(logterm_xp2)) - 1.0;
		const complex<double> fbxm = log(logterm_xm1) - (xm * log(logterm_xm2)) - 1.0;
		PVB0 = ((-1.0) * (log(psq * pow((1.0 / genQ), 2.0)) + fbxp + fbxm));
	}
	return PVB0;
}

complex<double> PV_B1_genp(const double& psq, const double& m1sq, const double& m2sq) {
	complex<double> PVB1;
	if (psq == 0.0 && m1sq == 0.0 && m2sq == 0.0) {
		PVB1 = 0.0;
		return PVB1;
	}
	
	double m1sqp, m2sqp;
	if (m1sq == 0.0) {
		m1sqp = 1.0e-3;
	}
	else {
		m1sqp = m1sq;
	}
	if (m2sq == 0.0) {
		m2sqp = 1.0e-3;
	}
	else {
		m2sqp = m2sq;
	}

	if (psq == 0.0) {
		PVB1 = 0.5 * (1.0 + log(pow(91.1876, 2.0) / (m2sq + 1.0e-3)) + (pow((m1sq / (m1sq - m2sq + 1.0e-6)), 2.0) * log((m2sqp + 1.0e-3) / (m1sqp + 1.0e-3))) + (0.5 * ((m1sq + m2sq) / (m1sq - m2sq + 1.0e-6))));
	}
	else {
		PVB1 = (1.0 / (2.0 * psq)) * (PV_A0(m2sq) - PV_A0(m1sq) + ((psq + m1sq - m2sq) * PV_B0_genp(psq, m1sq, m2sq)));
	}
	return PVB1;
}

complex<double> PV_B1_genp_genQ(const double& psq, const double& m1sq, const double& m2sq, const double& genQ) {
	complex<double> PVB1;
	if (psq == 0.0 && m1sq == 0.0 && m2sq == 0.0) {
		PVB1 = 0.0;
		return PVB1;
	}

	double m1sqp, m2sqp;
	if (m1sq == 0.0) {
		m1sqp = 1.0e-3;
	}
	else {
		m1sqp = m1sq;
	}
	if (m2sq == 0.0) {
		m2sqp = 1.0e-3;
	}
	else {
		m2sqp = m2sq;
	}

	if (psq == 0.0) {
		PVB1 = 0.5 * (1.0 + log(pow(91.1876, 2.0) / (m2sq + 1.0e-3)) + (pow((m1sq / (m1sq - m2sq + 1.0e-6)), 2.0) * log((m2sqp + 1.0e-3) / (m1sqp + 1.0e-3))) + (0.5 * ((m1sq + m2sq) / (m1sq - m2sq + 1.0e-6))));
	}
	else {
		PVB1 = (1.0 / (2.0 * psq)) * (PV_A0_genQ(m2sq, genQ) - PV_A0_genQ(m1sq, genQ) + ((psq + m1sq - m2sq) * PV_B0_genp_genQ(psq, m1sq, m2sq, genQ)));
	}
    return PVB1;
}

complex<double> PV_B22_genp(const double& psq, const double& m1sq, const double& m2sq) {
	double psqp;
	complex<double> PVB22;
	if (psq == 0.0 && m1sq == 0.0 && m2sq == 0.0) {
		PVB22 = 0.0;
		return PVB22;
	}

	PVB22 = (1.0 / 6.0)\
		* ((0.5 * (PV_A0(m1sq) + PV_A0(m2sq))) + ((m1sq + m2sq - (0.5 * psq)) * PV_B0_genp(psq, m1sq, m2sq))
		   + ((0.5 * (m2sq - m1sq) / (psq + 1.0e-3)) * (PV_A0(m2sq) - PV_A0(m1sq) - ((m2sq - m1sq) * PV_B0_genp(psq, m1sq, m2sq))))
		   + m1sq + m2sq - (psq / 3.0));
	return PVB22;
}

complex<double> PV_B22_genp_genQ(const double& psq, const double& m1sq, const double& m2sq, const double& genQ) {
	double psqp;
	complex<double> PVB22;
	if (psq == 0.0 && m1sq == 0.0 && m2sq == 0.0) {
		PVB22 = 0.0;
		return PVB22;
	}

	PVB22 = (1.0 / 6.0)\
		* ((0.5 * (PV_A0_genQ(m1sq, genQ) + PV_A0_genQ(m2sq, genQ))) + ((m1sq + m2sq - (0.5 * psq)) * PV_B0_genp_genQ(psq, m1sq, m2sq, genQ))
			+ ((0.5 * (m2sq - m1sq) / (psq + 1.0e-3)) * (PV_A0_genQ(m2sq, genQ) - PV_A0_genQ(m1sq, genQ) - ((m2sq - m1sq) * PV_B0_genp_genQ(psq, m1sq, m2sq, genQ))))
			+ m1sq + m2sq - (psq / 3.0));
	return PVB22;
}

complex<double> PV_F_genp(const double& psq, const double& m1sq, const double& m2sq) {
	const complex<double> PVF = PV_A0(m1sq) - (2.0 * PV_A0(m2sq)) - (((2.0 * psq) + (2.0 * m1sq) - m2sq) * PV_B0_genp(psq, m1sq, m2sq));
	return PVF;
}

complex<double> PV_F_genp_genQ(const double& psq, const double& m1sq, const double& m2sq, const double& genQ) {
	const complex<double> PVF = PV_A0_genQ(m1sq, genQ) - (2.0 * PV_A0_genQ(m2sq, genQ)) - (((2.0 * psq) + (2.0 * m1sq) - m2sq) * PV_B0_genp_genQ(psq, m1sq, m2sq, genQ));
	return PVF;
}

complex<double> PV_G_genp(const double& psq, const double& m1sq, const double& m2sq) {
	const complex<double> PVG = ((psq - m1sq - m2sq) * PV_B0_genp(psq, m1sq, m2sq)) - PV_A0(m1sq) - PV_A0(m2sq);
	return PVG;
}

complex<double> PV_G_genp_genQ(const double& psq, const double& m1sq, const double& m2sq, const double& genQ) {
	const complex<double> PVG = ((psq - m1sq - m2sq) * PV_B0_genp_genQ(psq, m1sq, m2sq, genQ)) - PV_A0_genQ(m1sq, genQ) - PV_A0_genQ(m2sq, genQ);
	return PVG;
}

complex<double> PV_H_genp(const double& psq, const double& m1sq, const double& m2sq) {
	const complex<double> PVH = (4.0 * PV_B22_genp(psq, m1sq, m2sq)) + PV_G_genp(psq, m1sq, m2sq);
	return PVH;
}

complex<double> PV_H_genp_genQ(const double& psq, const double& m1sq, const double& m2sq, const double& genQ) {
	const complex<double> PVH = (4.0 * PV_B22_genp_genQ(psq, m1sq, m2sq, genQ)) + PV_G_genp_genQ(psq, m1sq, m2sq, genQ);
	return PVH;
}

complex<double> PV_sB22_genp(const double& psq, const double& m1sq, const double& m2sq) {
	const complex<double> PVsB22 = PV_B22_genp(psq, m1sq, m2sq) - (0.25 * PV_A0(m1sq)) - (0.25 * PV_A0(m2sq));
	return PVsB22;
}

complex<double> PV_sB22_genp_genQ(const double& psq, const double& m1sq, const double& m2sq, const double& genQ) {
	const complex<double> PVsB22 = PV_B22_genp_genQ(psq, m1sq, m2sq, genQ) - (0.25 * PV_A0_genQ(m1sq, genQ)) - (0.25 * PV_A0_genQ(m2sq, genQ));
	return PVsB22;
}