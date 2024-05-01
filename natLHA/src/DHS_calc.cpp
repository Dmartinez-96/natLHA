#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include "DHS_calc.hpp"
#include <boost/multiprecision/mpfr.hpp>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace boost::multiprecision;
typedef number<mpfr_float_backend<50>> high_prec_float;

bool absValCompareHS(const LabeledValueHS& a, const LabeledValueHS& b) {
    return abs(a.value) < abs(b.value);
}

std::vector<LabeledValueHS> sortAndReturnHS(const std::vector<LabeledValueHS>& DHSList) {
    std::vector<LabeledValueHS> sortedList = DHSList;
    sort(sortedList.begin(), sortedList.end(), absValCompareHS);
    reverse(sortedList.begin(), sortedList.end());
    return sortedList;
}

std::vector<LabeledValueHS> DHS_calc(high_prec_float& mHdsq_Lambda, high_prec_float delta_mHdsq, high_prec_float& mHusq_Lambda, high_prec_float delta_mHusq,
                                high_prec_float mu_Lambdasq, high_prec_float delta_musq, high_prec_float running_mZ_sq, high_prec_float tanb_sq, high_prec_float& sigmauu_tot,
                                high_prec_float& sigmadd_tot) {
    high_prec_float B_Hd = mHdsq_Lambda / (tanb_sq - 1.0);
    high_prec_float B_Hu = mHusq_Lambda * tanb_sq / (tanb_sq - 1.0);
    high_prec_float B_Sigmadd = sigmadd_tot / (tanb_sq - 1.0);
    high_prec_float B_Sigmauu = sigmauu_tot * tanb_sq / (tanb_sq - 1.0);
    high_prec_float B_muLambdasq = mu_Lambdasq;
    
    high_prec_float B_deltaHd = delta_mHdsq / (tanb_sq - 1.0);
    high_prec_float B_deltaHu = delta_mHusq * tanb_sq / (tanb_sq - 1.0);
    high_prec_float B_deltamusq = delta_musq;

    std::vector<LabeledValueHS> Delta_HS_contribs = {{B_Hd / (pow(high_prec_float(911876.0) / high_prec_float(10000.0), high_prec_float(2.0)) / high_prec_float(2.0)), "Delta_HS(mHd^2(GUT))"},
                                                {B_Hu / (pow(high_prec_float(911876.0) / high_prec_float(10000.0), high_prec_float(2.0)) / high_prec_float(2.0)), "Delta_HS(mHu^2(GUT))"},
                                                {B_muLambdasq / (pow(high_prec_float(911876.0) / high_prec_float(10000.0), high_prec_float(2.0)) / high_prec_float(2.0)), "Delta_HS(mu^2(GUT))"},
                                                {B_deltaHd / (pow(high_prec_float(911876.0) / high_prec_float(10000.0), high_prec_float(2.0)) / high_prec_float(2.0)), "Delta_HS(delta(mHd^2))"},
                                                {B_deltaHu / (pow(high_prec_float(911876.0) / high_prec_float(10000.0), high_prec_float(2.0)) / high_prec_float(2.0)), "Delta_HS(delta(mHu^2))"},
                                                {B_deltamusq / (pow(high_prec_float(911876.0) / high_prec_float(10000.0), high_prec_float(2.0)) / high_prec_float(2.0)), "Delta_HS(delta(mu^2))"},
                                                {B_Sigmadd / (pow(high_prec_float(911876.0) / high_prec_float(10000.0), high_prec_float(2.0)) / high_prec_float(2.0)), "Delta_HS(Sigma_d^d)"},
                                                {B_Sigmauu / (pow(high_prec_float(911876.0) / high_prec_float(10000.0), high_prec_float(2.0)) / high_prec_float(2.0)), "Delta_HS(Sigma_u^u)"}};
    std::vector<LabeledValueHS> sortedList = sortAndReturnHS(Delta_HS_contribs);
    return sortedList;
}