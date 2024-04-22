// DSN_CALC_HPP

#ifndef DSN_CALC_HPP
#define DSN_CALC_HPP

#include <vector>
#include <string>

struct DSNLabeledValue {
    double value;
    std::string label;
};

std::vector<DSNLabeledValue> DSN_calc(int precselno, std::vector<double> Wk_boundary_conditions,
                                      double& current_mZ2, double& current_logQSUSY,
                                      double& current_logQGUT, int& nF, int& nD);

#endif