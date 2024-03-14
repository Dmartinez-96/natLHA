// DSN_CALC_HPP

#ifndef DSN_CALC_HPP
#define DSN_CALC_HPP

#include <vector>

struct LabeledValueSN;

std::vector<LabeledValueSN> DSN_calc(int modselno, double precselno, double& mymZsq,
                                     double& GUT_SCALE, double& myweakscale, double& inptanbval,
                                     std::vector<double>& GUT_boundary_conditions);

#endif