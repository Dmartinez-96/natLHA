// MSSM_U3Q3_RGE_SOLVER_HPP

#ifndef MSSM_U3Q3_RGE_SOLVER_HPP
#define MSSM_U3Q3_RGE_SOLVER_HPP

#include <vector>
#include <cmath>

struct RGEStruct2 {
    std::vector<double> RGEsolvec;
    double SUSYscale_eval;

    RGEStruct2(const std::vector<double>& RGEsolvec, double SUSYscale_eval) : RGEsolvec(RGEsolvec), SUSYscale_eval(SUSYscale_eval) {}
};

void MSSM_approx_RGESolver(const std::vector<double>& x, std::vector<double>& dxdt, const double t);    

std::vector<RGEStruct2> solveODEstoapproxMSUSY(std::vector<double> initialConditions, double startTime, double timeStep, double& t_target);

#endif 