#include <iostream>
#include <string>
#include <cstdlib>
#include <cmath>
#include <array>
#include <memory>
#include <algorithm>

using namespace std;

double get_tanb(double& mHusq, double& mHdsq, double& mu, double& Bmu, double& gp, double& g2, double& Sigmauutotal, double& Sigmaddtotal) {
    double newEvalTanb = (1.0 / (2.0 * Bmu))\
        * (mHusq + mHdsq + (2.0 * mu * mu)
           + sqrt(pow((mHusq + mHdsq + (2.0 * mu * mu)), 2.0) - (4.0 * pow(Bmu, 2.0))));
    return newEvalTanb;
}