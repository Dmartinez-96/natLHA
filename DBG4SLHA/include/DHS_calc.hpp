// DHS_CALC_HPP

#ifndef DHS_CALC_HPP
#define DHS_CALC_HPP

#include <vector>

struct LabeledValueHS {
    double value;
    std::string label;
};

std::vector<LabeledValueHS> DHS_calc(double& mHdsq_Lambda, double delta_mHdsq, double& mHusq_Lambda, double delta_mHusq,
                                     double mu_Lambdasq, double delta_musq, double running_mZ_sq, double tanb_sq, double& sigmauu_tot,
                                     double& sigmadd_tot);

#endif