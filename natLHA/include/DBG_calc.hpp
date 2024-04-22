// DBG_CALC_HPP

#ifndef DBG_CALC_HPP
#define DBG_CALC_HPP

#include <vector>
#include <string>

struct LabeledValueBG {
    double value;
    std::string label;
};

std::vector<LabeledValueBG> DBG_calc(int& modselno, int& precselno,
                                     double& GUT_SCALE, double& myweakscale, double& inptanbval,
                                     std::vector<double>& GUT_boundary_conditions, double& originalmZ2value);

#endif