// MSSM_STOP_RGE_SOLVER_HPP

#ifndef SC_MSSM_RGE_SOLVER_WITH_STOPFINDER_HPP
#define SC_MSSM_RGE_SOLVER_WITH_STOPFINDER_HPP

#include <vector>

struct RGEStruct {
    std::vector<double> RGEsolvec;
    double SUSYscale_eval;

    RGEStruct(const std::vector<double>& RGEsolvec, double SUSYscale_eval) : RGEsolvec(RGEsolvec), SUSYscale_eval(SUSYscale_eval) {}
};
void MSSM_stopscale_RGESolver(const std::vector<double>& x, std::vector<double>& dxdt, const double t);
struct MyObserver;

std::vector<RGEStruct> solveODEstoMSUSY(std::vector<double> initialConditions, double startTime, double timeStep, double& t_target, double value_of_mZ2);

#endif