// CONSTANTS_HPP

#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include <vector>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Define important constants
const double loop_fac = 1.0 / (16.0 * std::pow(M_PI, 2.0));
const double loop_fac_sq = std::pow(loop_fac, 2.0);

// Define constant arrays
const std::vector<double> b_1l = { 33.0 / 5.0, 1.0, -3.0 };

const std::vector<std::vector<double>> b_2l = {
    {199.0 / 25.0, 27.0 / 5.0, 88.0 / 5.0},
    {9.0 / 5.0, 25.0, 24.0},
    {11.0 / 5.0, 9.0, 14.0}
};

const std::vector<std::vector<double>> c_2l = {
    {26.0 / 5.0, 14.0 / 5.0, 18.0 / 5.0},
    {6.0, 6.0, 2.0},
    {4.0, 4.0, 0.0}
};

#endif
