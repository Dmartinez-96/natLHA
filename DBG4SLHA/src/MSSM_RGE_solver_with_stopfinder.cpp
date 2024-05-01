#include "MSSM_RGE_solver.hpp"
#include "MSSM_RGE_solver_with_stopfinder.hpp"
#include "constants.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <boost/numeric/odeint.hpp>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef boost::numeric::odeint::runge_kutta_dopri5<std::vector<double>> stepper_type;

struct MyObserver{
    double& t_target;
    double& min_difference;
    bool& condition_met;
    double intermmz_q_sq;

    MyObserver(double& t_target, double& min_difference, bool& condition_met, double intermmz_q_sq) : t_target(t_target),
        min_difference(min_difference), condition_met(condition_met), intermmz_q_sq(intermmz_q_sq) {}

    void operator()(const std::vector<double>& x, const double t) const {
        double current_Q = std::exp(t);
        double intermmQ3_sq_arr = (x[29]);
        double intermmU3_sq_arr = (x[35]);
        double intermvHiggs_wk = std::sqrt((2.0) / ((3.0 * std::pow(x[0], 2.0) / 5.0) + (std::pow(x[1], 2.0)))) * 91.1876;
        double intermbeta_wk = std::atan(x[43]);
        double intermsinsqb = std::pow(std::sin(intermbeta_wk), 2.0);
        double intermcossqb = std::pow(std::cos(intermbeta_wk), 2.0);
        double intermvu = intermvHiggs_wk * std::sqrt(intermsinsqb);
        double intermvd = intermvHiggs_wk * std::sqrt(intermcossqb);
        double intermvu_sq = std::pow(intermvu, 2.0);
        double intermvd_sq = std::pow(intermvd, 2.0);
        double intermv_sq = std::pow(intermvHiggs_wk, 2.0);
        double intermtan_th_w = std::sqrt(3.0 / 5.0) * x[0] / x[1];
        double intermtheta_w = std::atan(intermtan_th_w);
        double intermsinsq_th_w = std::pow(std::sin(intermtheta_w), 2.0);
        double intermgpr_wk = std::sqrt(3.0 / 5.0) * x[0];
        double intermg2_wk = x[1];
        double intermgz_sq = (std::pow(intermg2_wk, 2.0) + std::pow(intermgpr_wk, 2.0)) / 8.0;
        double intermyt_wk = x[7];
        double intermmymt = intermyt_wk * intermvu;
        double intermmymtsq = std::pow(intermmymt, 2.0);
        //double intermmz_q_sq = std::pow(91.1876, 2.0);
        double intermat_wk = x[16];
        double intermmuweakBC = x[6];
        double intermcos2b = std::cos(2.0 * intermbeta_wk);
        double intermsin2b = std::sin(2.0 * intermbeta_wk);
        double intermDelta_suL = (pow(intermvu, 2.0) - pow(intermvd, 2.0)) * ((intermgpr_wk * intermgpr_wk / 6.0) - (intermg2_wk * intermg2_wk / 4.0));
        double intermDelta_suR = (-1.0) * (pow(intermvu, 2.0) - pow(intermvd, 2.0)) * ((4.0 * intermgpr_wk * intermgpr_wk / 3.0));
        double intermm_stop_1sq = (0.5)\
            * (intermmQ3_sq_arr + intermmU3_sq_arr + (2.0 * intermmymtsq) + intermDelta_suL + intermDelta_suR
               - sqrt(pow((intermmQ3_sq_arr + intermDelta_suL - intermmU3_sq_arr - intermDelta_suR), 2.0)
                      + (4.0 * pow(((intermat_wk * intermvu) - (intermmuweakBC * intermyt_wk * intermvd)), 2.0))));
        double intermm_stop_2sq = (0.5)\
            * (intermmQ3_sq_arr + intermmU3_sq_arr + (2.0 * intermmymtsq) + intermDelta_suL + intermDelta_suR
               + sqrt(pow((intermmQ3_sq_arr + intermDelta_suL - intermmU3_sq_arr - intermDelta_suR), 2.0)
                      + (4.0 * pow(((intermat_wk * intermvu) - (intermmuweakBC * intermyt_wk * intermvd)), 2.0))));

        double current_difference = std::abs(current_Q - std::pow(std::abs(intermm_stop_1sq
                                                                           * intermm_stop_2sq), 0.25));

        if (current_difference > min_difference) {
            return; // No need to check further
        }

        if ((current_difference < min_difference) && (current_Q < 1.0e11)) {
            condition_met = true;
            t_target = t;
            min_difference = current_difference;
        }
    }
};

std::vector<RGEStruct> solveODEstoMSUSY(std::vector<double> initialConditions, double startTime, double timeStep, double& t_target, double value_of_mZ2) {
    using state_type = std::vector<double>;
    state_type x = initialConditions;
    double endTime = std::log(500.0);
    double min_difference = std::numeric_limits<double>::infinity();
    bool condition_met = false;

    MyObserver myObserver(t_target, min_difference, condition_met, value_of_mZ2);

    // Integrate
    boost::numeric::odeint::integrate_adaptive(
        make_controlled(1.0E-12, 1.0E-12, stepper_type()),
        MSSMRGESolver, x, startTime, std::log(500.0), timeStep, myObserver
    );
    
    std::vector<RGEStruct> solstruct = { {x, t_target} };
    return solstruct;
}