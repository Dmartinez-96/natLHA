// MSSM_RGE_SOLVER_HPP

#ifndef MSSM_RGE_SOLVER_HPP
#define MSSM_RGE_SOLVER_HPP

#include <vector>

struct ODETolerances {
    double absolute;
    double relative;
};

const ODETolerances& odeTolerances();

void MSSMRGESolver(const std::vector<double>& x, std::vector<double>& dxdt, const double t);

std::vector<double> solveODEs(std::vector<double> initialConditions, double startTime, double endTime, double timeStep);

#endif
