#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include "DHS_calc.hpp"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

bool absValCompareHS(const LabeledValueHS& a, const LabeledValueHS& b) {
    return abs(a.value) < abs(b.value);
}

vector<LabeledValueHS> sortAndReturnHS(const vector<LabeledValueHS>& DHSList) {
    vector<LabeledValueHS> sortedList = DHSList;
    sort(sortedList.begin(), sortedList.end(), absValCompareHS);
    reverse(sortedList.begin(), sortedList.end());
    return sortedList;
}

vector<LabeledValueHS> DHS_calc(double& mHdsq_Lambda, double delta_mHdsq, double& mHusq_Lambda, double delta_mHusq,
                                double mu_Lambdasq, double delta_musq, double running_mZ_sq, double tanb_sq, double& sigmauu_tot,
                                double& sigmadd_tot) {
    double B_Hd = mHdsq_Lambda / (tanb_sq - 1.0);
    double B_Hu = mHusq_Lambda * tanb_sq / (tanb_sq - 1.0);
    double B_Sigmadd = sigmadd_tot / (tanb_sq - 1.0);
    double B_Sigmauu = sigmauu_tot * tanb_sq / (tanb_sq - 1.0);
    double B_muLambdasq = mu_Lambdasq;
    
    double B_deltaHd = delta_mHdsq / (tanb_sq - 1.0);
    double B_deltaHu = delta_mHusq * tanb_sq / (tanb_sq - 1.0);
    double B_deltamusq = delta_musq;

    vector<LabeledValueHS> Delta_HS_contribs = {{B_Hd / (pow(91.1876, 2.0) / 2.0), "Delta_HS(mHd^2(GUT))"},
                                                {B_Hu / (pow(91.1876, 2.0) / 2.0), "Delta_HS(mHu^2(GUT))"},
                                                {B_muLambdasq / (pow(91.1876, 2.0) / 2.0), "Delta_HS(mu^2(GUT))"},
                                                {B_deltaHd / (pow(91.1876, 2.0) / 2.0), "Delta_HS(delta(mHd^2))"},
                                                {B_deltaHu / (pow(91.1876, 2.0) / 2.0), "Delta_HS(delta(mHu^2))"},
                                                {B_deltamusq / (pow(91.1876, 2.0) / 2.0), "Delta_HS(delta(mu^2))"},
                                                {B_Sigmadd / (pow(91.1876, 2.0) / 2.0), "Delta_HS(Sigma_d^d)"},
                                                {B_Sigmauu / (pow(91.1876, 2.0) / 2.0), "Delta_HS(Sigma_u^u)"}};
    vector<LabeledValueHS> sortedList = sortAndReturnHS(Delta_HS_contribs);
    return sortedList;
}