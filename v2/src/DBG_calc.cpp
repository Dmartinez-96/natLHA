#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <boost/math/special_functions/next.hpp>
#include "DBG_calc.hpp"
#include "MSSM_RGE_solver.hpp"
#include "mZ_numsolver.hpp"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

double deriv_mZ_step_calc(double RGE_scale_init_val, double RGE_scale_final_val, vector<double> BCs_to_run) {
    vector<double> currentweaksol = solveODEs(BCs_to_run, RGE_scale_init_val, RGE_scale_final_val, copysign(1.0e-4, (RGE_scale_final_val - RGE_scale_init_val)));
    double QSUSY_for_calc = exp(RGE_scale_final_val);
    // cout << "Weak-scale values on current iteration: " << endl;
    // for (double value : currentweaksol) {
    //     cout << value << endl;
    // }
    double mZ2_calc = getmZ2(currentweaksol, QSUSY_for_calc, 91.1876 * 91.1876);
    return mZ2_calc;
}

bool absValCompareBG(const LabeledValueBG& a, const LabeledValueBG& b) {
    return abs(a.value) < abs(b.value);
}

vector<LabeledValueBG> sortAndReturnBG(const vector<LabeledValueBG>& DBGList) {
    vector<LabeledValueBG> sortedList = DBGList;
    sort(sortedList.begin(), sortedList.end(), absValCompareBG);
    reverse(sortedList.begin(), sortedList.end());
    return sortedList;
}

double deriv_num_calc(int precselno, double curr_hval, vector<double> mzsq_values) {
    double approxderivval = 0.0;
    if (precselno == 1) {
        // 8-point derivative calculation
        approxderivval = (1.0 / curr_hval) * 
            ((mzsq_values[0] / 280.0) - (4.0 / 105.0) * mzsq_values[1] + 
            mzsq_values[2] / 5.0 - (4.0 / 5.0) * mzsq_values[3] + 
            (4.0 / 5.0) * mzsq_values[4] - mzsq_values[5] / 5.0 + 
            (4.0 / 105.0) * mzsq_values[6] - mzsq_values[7] / 280.0);
    } else if (precselno == 2) {
        // 4-point derivative calculation
        approxderivval = (1.0 / curr_hval) * 
            ((mzsq_values[0] / 12.0) - (2.0 / 3.0) * mzsq_values[1] + 
            (2.0 / 3.0) * mzsq_values[2] - mzsq_values[3] / 12.0);
    } else {
        // 2-point derivative calculation (default)
        approxderivval = (1.0 / curr_hval) * 
            ((-0.5) * mzsq_values[0] + 0.5 * mzsq_values[1]);
    }

    return approxderivval;
}

double deriv_step_calc_mu0(double& shift_amt, vector<double> inputGUT_BCs, double& initialScale, double& finalScale) {
    inputGUT_BCs[6] += shift_amt;
    double testShiftedMZ2 = deriv_mZ_step_calc(initialScale, finalScale, inputGUT_BCs);
    return testShiftedMZ2;
}

double deriv_step_calc_customRange(double& shift_amt, vector<double> inputGUT_BCs, double& initialScale, double& finalScale, int startingIndex, int endingIndex) {
    for (int i = startingIndex; i < endingIndex; ++i) {
        inputGUT_BCs[i] = copysign(pow((sqrt(abs(inputGUT_BCs[i])) + shift_amt), 2.0), inputGUT_BCs[i]);
    }
    double testShiftedMZ2 = deriv_mZ_step_calc(initialScale, finalScale, inputGUT_BCs);
    return testShiftedMZ2;
}

double deriv_step_calc_genScalarIndices(double& shift_amt, vector<double> inputGUT_BCs, double& initialScale, double& finalScale, vector<int> IndexValues) {
    for (int i : IndexValues) {
        inputGUT_BCs[i] = copysign(pow((sqrt(abs(inputGUT_BCs[i])) + shift_amt), 2.0), inputGUT_BCs[i]);
    }
    double testShiftedMZ2 = deriv_mZ_step_calc(initialScale, finalScale, inputGUT_BCs);
    return testShiftedMZ2;
}

double deriv_step_calc_scalars(double& shift_amt, vector<double> inputGUT_BCs, double& initialScale, double& finalScale) {
    for (int i = 25; i < 42; ++i) {
        inputGUT_BCs[i] = copysign(pow((sqrt(abs(inputGUT_BCs[i])) + shift_amt), 2.0), inputGUT_BCs[i]);
    }
    double testShiftedMZ2 = deriv_mZ_step_calc(initialScale, finalScale, inputGUT_BCs);
    return testShiftedMZ2;
}

double deriv_step_calc_trilin(double& shift_amt, vector<double> inputGUT_BCs, double& initialScale, double& finalScale) {
    for (int i = 16; i < 25; ++i) {
        inputGUT_BCs[i] = (((inputGUT_BCs[i] / inputGUT_BCs[i-9]) + shift_amt) * inputGUT_BCs[i-9]);
    }
    double testShiftedMZ2 = deriv_mZ_step_calc(initialScale, finalScale, inputGUT_BCs);
    return testShiftedMZ2;
}

double deriv_step_calc_gentrilinRange(double& shift_amt, vector<double> inputGUT_BCs, double& initialScale, double& finalScale, int startingIndex, int endingIndex) {
    for (int i = startingIndex; i < endingIndex; ++i) {
        inputGUT_BCs[i] = (((inputGUT_BCs[i] / inputGUT_BCs[i-9]) + shift_amt) * inputGUT_BCs[i-9]);
    }
    double testShiftedMZ2 = deriv_mZ_step_calc(initialScale, finalScale, inputGUT_BCs);
    return testShiftedMZ2;
}

double deriv_step_calc_gaugino(double& shift_amt, vector<double> inputGUT_BCs, double& initialScale, double& finalScale) {
    for (int i = 3; i < 6; ++i) {
        inputGUT_BCs[i] += shift_amt;
    }
    double testShiftedMZ2 = deriv_mZ_step_calc(initialScale, finalScale, inputGUT_BCs);
    return testShiftedMZ2;
}

double gen_deriv_step_calc(double& shift_amt, vector<double> inputGUT_BCs, int index_to_shift, double& initialScale, double& finalScale) {
    inputGUT_BCs[index_to_shift] += shift_amt;
    double testShiftedMZ2 = deriv_mZ_step_calc(initialScale, finalScale, inputGUT_BCs);
    return testShiftedMZ2;    
}

double gen_deriv_step_scalar_calc(double& shift_amt, vector<double> inputGUT_BCs, int index_to_shift, double& initialScale, double& finalScale) {
    inputGUT_BCs[index_to_shift] = copysign(pow(sqrt(abs(inputGUT_BCs[index_to_shift])) + shift_amt, 2.0), inputGUT_BCs[index_to_shift]);
    double testShiftedMZ2 = deriv_mZ_step_calc(initialScale, finalScale, inputGUT_BCs);
    return testShiftedMZ2;    
}

vector<double> stepsize_generator(int& modselno, int& precselno, vector<double>& inputGUT_BCs) {
    vector<double> stepsizes;
    if (modselno == 1) {
        vector<double> m0candidates = {inputGUT_BCs[27], inputGUT_BCs[28], inputGUT_BCs[29], inputGUT_BCs[30],
                                       inputGUT_BCs[31], inputGUT_BCs[33], inputGUT_BCs[34], inputGUT_BCs[35],
                                       inputGUT_BCs[36], inputGUT_BCs[37], inputGUT_BCs[38], inputGUT_BCs[39],
                                       inputGUT_BCs[40], inputGUT_BCs[41], inputGUT_BCs[25], inputGUT_BCs[26]};
        auto maxm0It = max_element(m0candidates.begin(), m0candidates.end(),
                                   [](double a, double b) { return abs(a) < abs(b); });
        double maxm0Val = sqrt(abs(*maxm0It));
        vector<double> mhfcandidates = {inputGUT_BCs[3], inputGUT_BCs[4], inputGUT_BCs[5]};
        auto maxmhfIt = max_element(mhfcandidates.begin(), mhfcandidates.end());
        double maxGauginoMass = *maxmhfIt;
        vector<double> A0candidates;
        for (int i = 16; i < 25; ++i) {
                A0candidates.push_back(inputGUT_BCs[i] / inputGUT_BCs[i-9]);
        }
        auto maxA0It = max_element(A0candidates.begin(), A0candidates.end(),
                                   [](double a, double b) { return abs(a) < abs(b); });
        double maxTrilin = abs(*maxA0It);
        double Absmu0value = abs(inputGUT_BCs[6]);
        if (precselno == 1) {
            double hm0 = pow(((2625.0 / 16.0) * (boost::math::float_next(maxm0Val) - maxm0Val)), (1.0 / 9.0));

            double hmhf = pow(((2625.0 / 16.0) * (boost::math::float_next(maxGauginoMass) - maxGauginoMass)), (1.0 / 9.0));

            double hA0 = pow(((2625.0 / 16.0) * (boost::math::float_next(maxTrilin) - maxTrilin)), (1.0 / 9.0));
            
            double hmu0 = pow(((2625.0 / 16.0) * (boost::math::float_next(Absmu0value) - Absmu0value)), (1.0 / 9.0));
            
            stepsizes = {hm0, hmhf, hA0, hmu0};
        } else if (precselno == 2) {
            double hm0 = pow(((45.0 / 4.0) * (boost::math::float_next(maxm0Val) - maxm0Val)), (1.0 / 5.0));

            double hmhf = pow(((45.0 / 4.0) * (boost::math::float_next(maxGauginoMass) - maxGauginoMass)), (1.0 / 5.0));

            double hA0 = pow(((45.0 / 4.0) * (boost::math::float_next(maxTrilin) - maxTrilin)), (1.0 / 5.0));
            
            double hmu0 = pow(((45.0 / 4.0) * (boost::math::float_next(Absmu0value) - Absmu0value)), (1.0 / 5.0));
            
            stepsizes = {hm0, hmhf, hA0, hmu0};
        } else {
            double hm0 = pow(((3.0) * (boost::math::float_next(maxm0Val) - maxm0Val)), (1.0 / 3.0));

            double hmhf = pow(((3.0) * (boost::math::float_next(maxGauginoMass) - maxGauginoMass)), (1.0 / 3.0));

            double hA0 = pow(((3.0) * (boost::math::float_next(maxTrilin) - maxTrilin)), (1.0 / 3.0));
            
            double hmu0 = pow(((3.0) * (boost::math::float_next(Absmu0value) - Absmu0value)), (1.0 / 3.0));
            
            stepsizes = {hm0, hmhf, hA0, hmu0};
        }
        // cout << "Step sizes: " << endl;
        // for (double value : stepsizes) {
        //     cout << value << endl;
        // }
    } 
    //////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////
    else if (modselno == 2) {
        auto maxmHudIt = max_element(inputGUT_BCs.begin() + 25, inputGUT_BCs.begin() + 27,
                                     [](double a, double b) { return abs(a) < abs(b); });
        double maxHiggsMass = sqrt(abs(*maxmHudIt));
        vector<double> m0candidates = {inputGUT_BCs[27], inputGUT_BCs[28], inputGUT_BCs[29], inputGUT_BCs[30],
                                       inputGUT_BCs[31], inputGUT_BCs[33], inputGUT_BCs[34], inputGUT_BCs[35],
                                       inputGUT_BCs[36], inputGUT_BCs[37], inputGUT_BCs[38], inputGUT_BCs[39],
                                       inputGUT_BCs[40], inputGUT_BCs[41]};
        auto maxm0It = max_element(m0candidates.begin(), m0candidates.end(),
                                   [](double a, double b) { return abs(a) < abs(b); });
        double maxm0Val = sqrt(abs(*maxm0It));
        vector<double> mhfcandidates = {inputGUT_BCs[3], inputGUT_BCs[4], inputGUT_BCs[5]};
        auto maxmhfIt = max_element(mhfcandidates.begin(), mhfcandidates.end());
        double maxGauginoMass = *maxmhfIt;
        vector<double> A0candidates;
        for (int i = 16; i < 25; ++i) {
                A0candidates.push_back(inputGUT_BCs[i] / inputGUT_BCs[i-9]);
        }
        auto maxA0It = max_element(A0candidates.begin(), A0candidates.end(),
                                   [](double a, double b) { return abs(a) < abs(b); });
        double maxTrilin = abs(*maxA0It);
        double Absmu0value = abs(inputGUT_BCs[6]);
        if (precselno == 1) {
            double hmHud0 = pow(((2625.0 / 16.0) * (boost::math::float_next(maxHiggsMass) - maxHiggsMass)), (1.0 / 9.0));

            double hm0 = pow(((2625.0 / 16.0) * (boost::math::float_next(maxm0Val) - maxm0Val)), (1.0 / 9.0));

            double hmhf = pow(((2625.0 / 16.0) * (boost::math::float_next(maxGauginoMass) - maxGauginoMass)), (1.0 / 9.0));

            double hA0 = pow(((2625.0 / 16.0) * (boost::math::float_next(maxTrilin) - maxTrilin)), (1.0 / 9.0));
            
            double hmu0 = pow(((2625.0 / 16.0) * (boost::math::float_next(Absmu0value) - Absmu0value)), (1.0 / 9.0));
            
            stepsizes = {hmHud0, hm0, hmhf, hA0, hmu0};
        } else if (precselno == 2) {
            double hmHud0 = pow(((45.0 / 4.0) * (boost::math::float_next(maxHiggsMass) - maxHiggsMass)), (1.0 / 5.0));

            double hm0 = pow(((45.0 / 4.0) * (boost::math::float_next(maxm0Val) - maxm0Val)), (1.0 / 5.0));

            double hmhf = pow(((45.0 / 4.0) * (boost::math::float_next(maxGauginoMass) - maxGauginoMass)), (1.0 / 5.0));

            double hA0 = pow(((45.0 / 4.0) * (boost::math::float_next(maxTrilin) - maxTrilin)), (1.0 / 5.0));
            
            double hmu0 = pow(((45.0 / 4.0) * (boost::math::float_next(Absmu0value) - Absmu0value)), (1.0 / 5.0));
            
            stepsizes = {hmHud0, hm0, hmhf, hA0, hmu0};
        } else {
            double hmHud0 = pow(((3.0) * (boost::math::float_next(maxHiggsMass) - maxHiggsMass)), (1.0 / 3.0));

            double hm0 = pow(((3.0) * (boost::math::float_next(maxm0Val) - maxm0Val)), (1.0 / 3.0));

            double hmhf = pow(((3.0) * (boost::math::float_next(maxGauginoMass) - maxGauginoMass)), (1.0 / 3.0));

            double hA0 = pow(((3.0) * (boost::math::float_next(maxTrilin) - maxTrilin)), (1.0 / 3.0));
            
            double hmu0 = pow(((3.0) * (boost::math::float_next(Absmu0value) - Absmu0value)), (1.0 / 3.0));
            
            stepsizes = {hmHud0, hm0, hmhf, hA0, hmu0};
        }
    }
    //////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////
    else if (modselno == 3) {
        double mHu0Value = sqrt(abs(inputGUT_BCs[25]));
        double mHd0Value = sqrt(abs(inputGUT_BCs[26]));
        vector<double> m0candidates = {inputGUT_BCs[27], inputGUT_BCs[28], inputGUT_BCs[29], inputGUT_BCs[30],
                                       inputGUT_BCs[31], inputGUT_BCs[33], inputGUT_BCs[34], inputGUT_BCs[35],
                                       inputGUT_BCs[36], inputGUT_BCs[37], inputGUT_BCs[38], inputGUT_BCs[39],
                                       inputGUT_BCs[40], inputGUT_BCs[41]};
        auto maxm0It = max_element(m0candidates.begin(), m0candidates.end(),
                                   [](double a, double b) { return abs(a) < abs(b); });
        double maxm0Val = sqrt(abs(*maxm0It));
        vector<double> mhfcandidates = {inputGUT_BCs[3], inputGUT_BCs[4], inputGUT_BCs[5]};
        auto maxmhfIt = max_element(mhfcandidates.begin(), mhfcandidates.end());
        double maxGauginoMass = *maxmhfIt;
        vector<double> A0candidates;
        for (int i = 16; i < 25; ++i) {
                A0candidates.push_back(inputGUT_BCs[i] / inputGUT_BCs[i-9]);
        }
        auto maxA0It = max_element(A0candidates.begin(), A0candidates.end(),
                                   [](double a, double b) { return abs(a) < abs(b); });
        double maxTrilin = abs(*maxA0It);
        double Absmu0value = abs(inputGUT_BCs[6]);
        if (precselno == 1) {
            double hmHu0 = pow(((2625.0 / 16.0) * (boost::math::float_next(mHu0Value) - mHu0Value)), (1.0 / 9.0));
            double hmHd0 = pow(((2625.0 / 16.0) * (boost::math::float_next(mHd0Value) - mHd0Value)), (1.0 / 9.0));

            double hm0 = pow(((2625.0 / 16.0) * (boost::math::float_next(maxm0Val) - maxm0Val)), (1.0 / 9.0));

            double hmhf = pow(((2625.0 / 16.0) * (boost::math::float_next(maxGauginoMass) - maxGauginoMass)), (1.0 / 9.0));

            double hA0 = pow(((2625.0 / 16.0) * (boost::math::float_next(maxTrilin) - maxTrilin)), (1.0 / 9.0));
            
            double hmu0 = pow(((2625.0 / 16.0) * (boost::math::float_next(Absmu0value) - Absmu0value)), (1.0 / 9.0));
            
            stepsizes = {hmHu0, hmHd0, hm0, hmhf, hA0, hmu0};
        } else if (precselno == 2) {
            double hmHu0 = pow(((45.0 / 4.0) * (boost::math::float_next(mHu0Value) - mHu0Value)), (1.0 / 5.0));
            double hmHd0 = pow(((45.0 / 4.0) * (boost::math::float_next(mHd0Value) - mHd0Value)), (1.0 / 5.0));

            double hm0 = pow(((45.0 / 4.0) * (boost::math::float_next(maxm0Val) - maxm0Val)), (1.0 / 5.0));

            double hmhf = pow(((45.0 / 4.0) * (boost::math::float_next(maxGauginoMass) - maxGauginoMass)), (1.0 / 5.0));

            double hA0 = pow(((45.0 / 4.0) * (boost::math::float_next(maxTrilin) - maxTrilin)), (1.0 / 5.0));
            
            double hmu0 = pow(((45.0 / 4.0) * (boost::math::float_next(Absmu0value) - Absmu0value)), (1.0 / 5.0));
            
            stepsizes = {hmHu0, hmHd0, hm0, hmhf, hA0, hmu0};
        } else {
            double hmHu0 = pow(((3.0) * (boost::math::float_next(mHu0Value) - mHu0Value)), (1.0 / 3.0));
            double hmHd0 = pow(((3.0) * (boost::math::float_next(mHd0Value) - mHd0Value)), (1.0 / 3.0));

            double hm0 = pow(((3.0) * (boost::math::float_next(maxm0Val) - maxm0Val)), (1.0 / 3.0));

            double hmhf = pow(((3.0) * (boost::math::float_next(maxGauginoMass) - maxGauginoMass)), (1.0 / 3.0));

            double hA0 = pow(((3.0) * (boost::math::float_next(maxTrilin) - maxTrilin)), (1.0 / 3.0));
            
            double hmu0 = pow(((3.0) * (boost::math::float_next(Absmu0value) - Absmu0value)), (1.0 / 3.0));
            
            stepsizes = {hmHu0, hmHd0, hm0, hmhf, hA0, hmu0};
        }
    }
    //////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////
    else if (modselno == 4) {
        double mHu0Value = sqrt(abs(inputGUT_BCs[25]));
        double mHd0Value = sqrt(abs(inputGUT_BCs[26]));
        vector<double> m012candidates = {inputGUT_BCs[27], inputGUT_BCs[28], inputGUT_BCs[30],
                                         inputGUT_BCs[31], inputGUT_BCs[33], inputGUT_BCs[34],
                                         inputGUT_BCs[36], inputGUT_BCs[37], inputGUT_BCs[39],
                                         inputGUT_BCs[40]};
        auto maxm012It = max_element(m012candidates.begin(), m012candidates.end(),
                                     [](double a, double b) { return abs(a) < abs(b); });
        double maxm012Val = sqrt(abs(*maxm012It));
        vector<double> m03candidates = {inputGUT_BCs[29], inputGUT_BCs[32], inputGUT_BCs[35], inputGUT_BCs[38],
                                        inputGUT_BCs[41]};
        auto maxm03It = max_element(m03candidates.begin(), m03candidates.end(),
                                     [](double a, double b) { return abs(a) < abs(b); });
        double maxm03Val = sqrt(abs(*maxm03It));
        vector<double> mhfcandidates = {inputGUT_BCs[3], inputGUT_BCs[4], inputGUT_BCs[5]};
        auto maxmhfIt = max_element(mhfcandidates.begin(), mhfcandidates.end());
        double maxGauginoMass = *maxmhfIt;
        vector<double> A0candidates;
        for (int i = 16; i < 25; ++i) {
                A0candidates.push_back(inputGUT_BCs[i] / inputGUT_BCs[i-9]);
        }
        auto maxA0It = max_element(A0candidates.begin(), A0candidates.end(),
                                   [](double a, double b) { return abs(a) < abs(b); });
        double maxTrilin = abs(*maxA0It);
        double Absmu0value = abs(inputGUT_BCs[6]);
        if (precselno == 1) {
            double hmHu0 = pow(((2625.0 / 16.0) * (boost::math::float_next(mHu0Value) - mHu0Value)), (1.0 / 9.0));
            double hmHd0 = pow(((2625.0 / 16.0) * (boost::math::float_next(mHd0Value) - mHd0Value)), (1.0 / 9.0));

            double hm012 = pow(((2625.0 / 16.0) * (boost::math::float_next(maxm012Val) - maxm012Val)), (1.0 / 9.0));
            double hm03 = pow(((2625.0 / 16.0) * (boost::math::float_next(maxm03Val) - maxm03Val)), (1.0 / 9.0));

            double hmhf = pow(((2625.0 / 16.0) * (boost::math::float_next(maxGauginoMass) - maxGauginoMass)), (1.0 / 9.0));

            double hA0 = pow(((2625.0 / 16.0) * (boost::math::float_next(maxTrilin) - maxTrilin)), (1.0 / 9.0));
            double hmu0 = pow(((2625.0 / 16.0) * (boost::math::float_next(Absmu0value) - Absmu0value)), (1.0 / 9.0));
            
            stepsizes = {hmHu0, hmHd0, hm012, hm03, hmhf, hA0, hmu0};
        } else if (precselno == 2) {
            double hmHu0 = pow(((45.0 / 4.0) * (boost::math::float_next(mHu0Value) - mHu0Value)), (1.0 / 5.0));
            double hmHd0 = pow(((45.0 / 4.0) * (boost::math::float_next(mHd0Value) - mHd0Value)), (1.0 / 5.0));

            double hm012 = pow(((45.0 / 4.0) * (boost::math::float_next(maxm012Val) - maxm012Val)), (1.0 / 5.0));
            double hm03 = pow(((45.0 / 4.0) * (boost::math::float_next(maxm03Val) - maxm03Val)), (1.0 / 5.0));

            double hmhf = pow(((45.0 / 4.0) * (boost::math::float_next(maxGauginoMass) - maxGauginoMass)), (1.0 / 5.0));

            double hA0 = pow(((45.0 / 4.0) * (boost::math::float_next(maxTrilin) - maxTrilin)), (1.0 / 5.0));
            double hmu0 = pow(((45.0 / 4.0) * (boost::math::float_next(Absmu0value) - Absmu0value)), (1.0 / 5.0));
            
            stepsizes = {hmHu0, hmHd0, hm012, hm03, hmhf, hA0, hmu0};
        } else {
            double hmHu0 = pow(((3.0) * (boost::math::float_next(mHu0Value) - mHu0Value)), (1.0 / 3.0));
            double hmHd0 = pow(((3.0) * (boost::math::float_next(mHd0Value) - mHd0Value)), (1.0 / 3.0));

            double hm012 = pow(((3.0) * (boost::math::float_next(maxm012Val) - maxm012Val)), (1.0 / 3.0));
            double hm03 = pow(((3.0) * (boost::math::float_next(maxm03Val) - maxm03Val)), (1.0 / 3.0));

            double hmhf = pow(((3.0) * (boost::math::float_next(maxGauginoMass) - maxGauginoMass)), (1.0 / 3.0));

            double hA0 = pow(((3.0) * (boost::math::float_next(maxTrilin) - maxTrilin)), (1.0 / 3.0));
            double hmu0 = pow(((3.0) * (boost::math::float_next(Absmu0value) - Absmu0value)), (1.0 / 3.0));
            
            stepsizes = {hmHu0, hmHd0, hm012, hm03, hmhf, hA0, hmu0};
        }
    }
    //////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////
    else if (modselno == 5) {
        double mHu0Value = sqrt(abs(inputGUT_BCs[25]));
        double mHd0Value = sqrt(abs(inputGUT_BCs[26]));
        vector<double> m01candidates = {inputGUT_BCs[27], inputGUT_BCs[30],
                                        inputGUT_BCs[33], inputGUT_BCs[36],
                                        inputGUT_BCs[39]};
        vector<double> m02candidates = {inputGUT_BCs[28], inputGUT_BCs[31],
                                        inputGUT_BCs[34], inputGUT_BCs[37],
                                        inputGUT_BCs[40]};
        auto maxm01It = max_element(m01candidates.begin(), m01candidates.end(),
                                     [](double a, double b) { return abs(a) < abs(b); });
        double maxm01Val = sqrt(abs(*maxm01It));
        auto maxm02It = max_element(m02candidates.begin(), m02candidates.end(),
                                     [](double a, double b) { return abs(a) < abs(b); });
        double maxm02Val = sqrt(abs(*maxm02It));
        vector<double> m03candidates = {inputGUT_BCs[29], inputGUT_BCs[32], inputGUT_BCs[35], inputGUT_BCs[38],
                                        inputGUT_BCs[41]};
        auto maxm03It = max_element(m03candidates.begin(), m03candidates.end(),
                                     [](double a, double b) { return abs(a) < abs(b); });
        double maxm03Val = sqrt(abs(*maxm03It));
        vector<double> mhfcandidates = {inputGUT_BCs[3], inputGUT_BCs[4], inputGUT_BCs[5]};
        auto maxmhfIt = max_element(mhfcandidates.begin(), mhfcandidates.end());
        double maxGauginoMass = *maxmhfIt;
        vector<double> A0candidates;
        for (int i = 16; i < 25; ++i) {
                A0candidates.push_back(inputGUT_BCs[i] / inputGUT_BCs[i-9]);
        }
        auto maxA0It = max_element(A0candidates.begin(), A0candidates.end(),
                                   [](double a, double b) { return abs(a) < abs(b); });
        double maxTrilin = abs(*maxA0It);
        double Absmu0value = abs(inputGUT_BCs[6]);
        if (precselno == 1) {
            double hmHu0 = pow(((2625.0 / 16.0) * (boost::math::float_next(mHu0Value) - mHu0Value)), (1.0 / 9.0));
            double hmHd0 = pow(((2625.0 / 16.0) * (boost::math::float_next(mHd0Value) - mHd0Value)), (1.0 / 9.0));

            double hm01 = pow(((2625.0 / 16.0) * (boost::math::float_next(maxm01Val) - maxm01Val)), (1.0 / 9.0));
            double hm02 = pow(((2625.0 / 16.0) * (boost::math::float_next(maxm02Val) - maxm02Val)), (1.0 / 9.0));
            double hm03 = pow(((2625.0 / 16.0) * (boost::math::float_next(maxm03Val) - maxm03Val)), (1.0 / 9.0));

            double hmhf = pow(((2625.0 / 16.0) * (boost::math::float_next(maxGauginoMass) - maxGauginoMass)), (1.0 / 9.0));

            double hA0 = pow(((2625.0 / 16.0) * (boost::math::float_next(maxTrilin) - maxTrilin)), (1.0 / 9.0));
            double hmu0 = pow(((2625.0 / 16.0) * (boost::math::float_next(Absmu0value) - Absmu0value)), (1.0 / 9.0));
            
            stepsizes = {hmHu0, hmHd0, hm01, hm02, hm03, hmhf, hA0, hmu0};
        } else if (precselno == 2) {
            double hmHu0 = pow(((45.0 / 4.0) * (boost::math::float_next(mHu0Value) - mHu0Value)), (1.0 / 5.0));
            double hmHd0 = pow(((45.0 / 4.0) * (boost::math::float_next(mHd0Value) - mHd0Value)), (1.0 / 5.0));

            double hm01 = pow(((45.0 / 4.0) * (boost::math::float_next(maxm01Val) - maxm01Val)), (1.0 / 5.0));
            double hm02 = pow(((45.0 / 4.0) * (boost::math::float_next(maxm02Val) - maxm02Val)), (1.0 / 5.0));
            double hm03 = pow(((45.0 / 4.0) * (boost::math::float_next(maxm03Val) - maxm03Val)), (1.0 / 5.0));

            double hmhf = pow(((45.0 / 4.0) * (boost::math::float_next(maxGauginoMass) - maxGauginoMass)), (1.0 / 5.0));

            double hA0 = pow(((45.0 / 4.0) * (boost::math::float_next(maxTrilin) - maxTrilin)), (1.0 / 5.0));
            double hmu0 = pow(((45.0 / 4.0) * (boost::math::float_next(Absmu0value) - Absmu0value)), (1.0 / 5.0));
            
            stepsizes = {hmHu0, hmHd0, hm01, hm02, hm03, hmhf, hA0, hmu0};
        } else {
            double hmHu0 = pow(((3.0) * (boost::math::float_next(mHu0Value) - mHu0Value)), (1.0 / 3.0));
            double hmHd0 = pow(((3.0) * (boost::math::float_next(mHd0Value) - mHd0Value)), (1.0 / 3.0));

            double hm01 = pow(((3.0) * (boost::math::float_next(maxm01Val) - maxm01Val)), (1.0 / 3.0));
            double hm02 = pow(((3.0) * (boost::math::float_next(maxm02Val) - maxm02Val)), (1.0 / 3.0));
            double hm03 = pow(((3.0) * (boost::math::float_next(maxm03Val) - maxm03Val)), (1.0 / 3.0));

            double hmhf = pow(((3.0) * (boost::math::float_next(maxGauginoMass) - maxGauginoMass)), (1.0 / 3.0));

            double hA0 = pow(((3.0) * (boost::math::float_next(maxTrilin) - maxTrilin)), (1.0 / 3.0));
            double hmu0 = pow(((3.0) * (boost::math::float_next(Absmu0value) - Absmu0value)), (1.0 / 3.0));
            
            stepsizes = {hmHu0, hmHd0, hm01, hm02, hm03, hmhf, hA0, hmu0};
        }
    }
    //////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////
    else {
        double mHu0Value = sqrt(abs(inputGUT_BCs[25]));
        double mHd0Value = sqrt(abs(inputGUT_BCs[26]));
        double maxmqL12Val = sqrt(abs(max(inputGUT_BCs[27], inputGUT_BCs[28])));
        double mqL3Val = sqrt(abs(inputGUT_BCs[29]));
        double maxmuR12Val = sqrt(abs(max(inputGUT_BCs[33], inputGUT_BCs[34])));
        double muR3Val = sqrt(abs(inputGUT_BCs[35]));
        double maxmdR12Val = sqrt(abs(max(inputGUT_BCs[36], inputGUT_BCs[37])));
        double mdR3Val = sqrt(abs(inputGUT_BCs[38]));
        double maxmeL12Val = sqrt(abs(max(inputGUT_BCs[30], inputGUT_BCs[31])));
        double meL3Val = sqrt(abs(inputGUT_BCs[32]));
        double maxmeR12Val = sqrt(abs(max(inputGUT_BCs[39], inputGUT_BCs[40])));
        double meR3Val = sqrt(abs(inputGUT_BCs[41]));
        double M1GUTVal = inputGUT_BCs[3];
        double M2GUTVal = inputGUT_BCs[4];
        double M3GUTVal = inputGUT_BCs[5];
        vector<double> Au0candidates = {inputGUT_BCs[16] / inputGUT_BCs[7],
                                        inputGUT_BCs[17] / inputGUT_BCs[8],
                                        inputGUT_BCs[18] / inputGUT_BCs[9]};
        auto maxAu0It = max_element(Au0candidates.begin(), Au0candidates.end(),
                                    [](double a, double b) { return abs(a) < abs(b); });
        double maxUpTrilin = abs(*maxAu0It);
        vector<double> Ad0candidates = {inputGUT_BCs[19] / inputGUT_BCs[10],
                                        inputGUT_BCs[20] / inputGUT_BCs[11],
                                        inputGUT_BCs[21] / inputGUT_BCs[12]};
        auto maxAd0It = max_element(Ad0candidates.begin(), Ad0candidates.end(),
                                    [](double a, double b) { return abs(a) < abs(b); });
        double maxDownTrilin = abs(*maxAd0It);
        vector<double> Ae0candidates = {inputGUT_BCs[22] / inputGUT_BCs[13],
                                        inputGUT_BCs[23] / inputGUT_BCs[14],
                                        inputGUT_BCs[24] / inputGUT_BCs[15]};
        auto maxAe0It = max_element(Ae0candidates.begin(), Ae0candidates.end(),
                                    [](double a, double b) { return abs(a) < abs(b); });
        double maxLeptTrilin = abs(*maxAe0It);
        double Absmu0value = abs(inputGUT_BCs[6]);
        if (precselno == 1) {
            double hmHu0 = pow(((2625.0 / 16.0) * (boost::math::float_next(mHu0Value) - mHu0Value)), (1.0 / 9.0));
            double hmHd0 = pow(((2625.0 / 16.0) * (boost::math::float_next(mHd0Value) - mHd0Value)), (1.0 / 9.0));

            double hmqL12 = pow(((2625.0 / 16.0) * (boost::math::float_next(maxmqL12Val) - maxmqL12Val)), (1.0 / 9.0));
            double hmqL3 = pow(((2625.0 / 16.0) * (boost::math::float_next(mqL3Val) - mqL3Val)), (1.0 / 9.0));
            double hmuR12 = pow(((2625.0 / 16.0) * (boost::math::float_next(maxmuR12Val) - maxmuR12Val)), (1.0 / 9.0));
            double hmuR3 = pow(((2625.0 / 16.0) * (boost::math::float_next(muR3Val) - muR3Val)), (1.0 / 9.0));
            double hmdR12 = pow(((2625.0 / 16.0) * (boost::math::float_next(maxmdR12Val) - maxmdR12Val)), (1.0 / 9.0));
            double hmdR3 = pow(((2625.0 / 16.0) * (boost::math::float_next(mdR3Val) - mdR3Val)), (1.0 / 9.0));
            double hmeL12 = pow(((2625.0 / 16.0) * (boost::math::float_next(maxmeL12Val) - maxmeL12Val)), (1.0 / 9.0));
            double hmeL3 = pow(((2625.0 / 16.0) * (boost::math::float_next(meL3Val) - meL3Val)), (1.0 / 9.0));
            double hmeR12 = pow(((2625.0 / 16.0) * (boost::math::float_next(maxmeR12Val) - maxmeR12Val)), (1.0 / 9.0));
            double hmeR3 = pow(((2625.0 / 16.0) * (boost::math::float_next(meR3Val) - meR3Val)), (1.0 / 9.0));

            double hM1 = pow(((2625.0 / 16.0) * (boost::math::float_next(M1GUTVal) - M1GUTVal)), (1.0 / 9.0));
            double hM2 = pow(((2625.0 / 16.0) * (boost::math::float_next(M2GUTVal) - M2GUTVal)), (1.0 / 9.0));
            double hM3 = pow(((2625.0 / 16.0) * (boost::math::float_next(M3GUTVal) - M3GUTVal)), (1.0 / 9.0));

            double hAu0 = pow(((2625.0 / 16.0) * (boost::math::float_next(maxUpTrilin) - maxUpTrilin)), (1.0 / 9.0));
            double hAd0 = pow(((2625.0 / 16.0) * (boost::math::float_next(maxDownTrilin) - maxDownTrilin)), (1.0 / 9.0));
            double hAe0 = pow(((2625.0 / 16.0) * (boost::math::float_next(maxLeptTrilin) - maxLeptTrilin)), (1.0 / 9.0));
            
            double hmu0 = pow(((2625.0 / 16.0) * (boost::math::float_next(Absmu0value) - Absmu0value)), (1.0 / 9.0));
            
            stepsizes = {hmHu0, hmHd0, hmqL12, hmqL3, hmuR12, hmuR3, hmdR12, hmdR3, hmeL12, hmeL3, hmeR12, hmeR3,
                         hM1, hM2, hM3, hAu0, hAd0, hAe0, hmu0};
        } else if (precselno == 2) {
            double hmHu0 = pow(((45.0 / 4.0) * (boost::math::float_next(mHu0Value) - mHu0Value)), (1.0 / 5.0));
            double hmHd0 = pow(((45.0 / 4.0) * (boost::math::float_next(mHd0Value) - mHd0Value)), (1.0 / 5.0));

            double hmqL12 = pow(((45.0 / 4.0) * (boost::math::float_next(maxmqL12Val) - maxmqL12Val)), (1.0 / 5.0));
            double hmqL3 = pow(((45.0 / 4.0) * (boost::math::float_next(mqL3Val) - mqL3Val)), (1.0 / 5.0));
            double hmuR12 = pow(((45.0 / 4.0) * (boost::math::float_next(maxmuR12Val) - maxmuR12Val)), (1.0 / 5.0));
            double hmuR3 = pow(((45.0 / 4.0) * (boost::math::float_next(muR3Val) - muR3Val)), (1.0 / 5.0));
            double hmdR12 = pow(((45.0 / 4.0) * (boost::math::float_next(maxmdR12Val) - maxmdR12Val)), (1.0 / 5.0));
            double hmdR3 = pow(((45.0 / 4.0) * (boost::math::float_next(mdR3Val) - mdR3Val)), (1.0 / 5.0));
            double hmeL12 = pow(((45.0 / 4.0) * (boost::math::float_next(maxmeL12Val) - maxmeL12Val)), (1.0 / 5.0));
            double hmeL3 = pow(((45.0 / 4.0) * (boost::math::float_next(meL3Val) - meL3Val)), (1.0 / 5.0));
            double hmeR12 = pow(((45.0 / 4.0) * (boost::math::float_next(maxmeR12Val) - maxmeR12Val)), (1.0 / 5.0));
            double hmeR3 = pow(((45.0 / 4.0) * (boost::math::float_next(meR3Val) - meR3Val)), (1.0 / 5.0));

            double hM1 = pow(((45.0 / 4.0) * (boost::math::float_next(M1GUTVal) - M1GUTVal)), (1.0 / 5.0));
            double hM2 = pow(((45.0 / 4.0) * (boost::math::float_next(M2GUTVal) - M2GUTVal)), (1.0 / 5.0));
            double hM3 = pow(((45.0 / 4.0) * (boost::math::float_next(M3GUTVal) - M3GUTVal)), (1.0 / 5.0));

            double hAu0 = pow(((45.0 / 4.0) * (boost::math::float_next(maxUpTrilin) - maxUpTrilin)), (1.0 / 5.0));
            double hAd0 = pow(((45.0 / 4.0) * (boost::math::float_next(maxDownTrilin) - maxDownTrilin)), (1.0 / 5.0));
            double hAe0 = pow(((45.0 / 4.0) * (boost::math::float_next(maxLeptTrilin) - maxLeptTrilin)), (1.0 / 5.0));
            
            double hmu0 = pow(((45.0 / 4.0) * (boost::math::float_next(Absmu0value) - Absmu0value)), (1.0 / 5.0));
            
            stepsizes = {hmHu0, hmHd0, hmqL12, hmqL3, hmuR12, hmuR3, hmdR12, hmdR3, hmeL12, hmeL3, hmeR12, hmeR3,
                         hM1, hM2, hM3, hAu0, hAd0, hAe0, hmu0};
        } else {
            double hmHu0 = pow(((3.0) * (boost::math::float_next(mHu0Value) - mHu0Value)), (1.0 / 3.0));
            double hmHd0 = pow(((3.0) * (boost::math::float_next(mHd0Value) - mHd0Value)), (1.0 / 3.0));

            double hmqL12 = pow(((3.0) * (boost::math::float_next(maxmqL12Val) - maxmqL12Val)), (1.0 / 3.0));
            double hmqL3 = pow(((3.0) * (boost::math::float_next(mqL3Val) - mqL3Val)), (1.0 / 3.0));
            double hmuR12 = pow(((3.0) * (boost::math::float_next(maxmuR12Val) - maxmuR12Val)), (1.0 / 3.0));
            double hmuR3 = pow(((3.0) * (boost::math::float_next(muR3Val) - muR3Val)), (1.0 / 3.0));
            double hmdR12 = pow(((3.0) * (boost::math::float_next(maxmdR12Val) - maxmdR12Val)), (1.0 / 3.0));
            double hmdR3 = pow(((3.0) * (boost::math::float_next(mdR3Val) - mdR3Val)), (1.0 / 3.0));
            double hmeL12 = pow(((3.0) * (boost::math::float_next(maxmeL12Val) - maxmeL12Val)), (1.0 / 3.0));
            double hmeL3 = pow(((3.0) * (boost::math::float_next(meL3Val) - meL3Val)), (1.0 / 3.0));
            double hmeR12 = pow(((3.0) * (boost::math::float_next(maxmeR12Val) - maxmeR12Val)), (1.0 / 3.0));
            double hmeR3 = pow(((3.0) * (boost::math::float_next(meR3Val) - meR3Val)), (1.0 / 3.0));

            double hM1 = pow(((3.0) * (boost::math::float_next(M1GUTVal) - M1GUTVal)), (1.0 / 3.0));
            double hM2 = pow(((3.0) * (boost::math::float_next(M2GUTVal) - M2GUTVal)), (1.0 / 3.0));
            double hM3 = pow(((3.0) * (boost::math::float_next(M3GUTVal) - M3GUTVal)), (1.0 / 3.0));

            double hAu0 = pow(((3.0) * (boost::math::float_next(maxUpTrilin) - maxUpTrilin)), (1.0 / 3.0));
            double hAd0 = pow(((3.0) * (boost::math::float_next(maxDownTrilin) - maxDownTrilin)), (1.0 / 3.0));
            double hAe0 = pow(((3.0) * (boost::math::float_next(maxLeptTrilin) - maxLeptTrilin)), (1.0 / 3.0));
            
            double hmu0 = pow(((3.0) * (boost::math::float_next(Absmu0value) - Absmu0value)), (1.0 / 3.0));
            
            stepsizes = {hmHu0, hmHd0, hmqL12, hmqL3, hmuR12, hmuR3, hmdR12, hmdR3, hmeL12, hmeL3, hmeR12, hmeR3,
                         hM1, hM2, hM3, hAu0, hAd0, hAe0, hmu0};
        }
    }
    return stepsizes;
}

vector<LabeledValueBG> DBG_calc(int& modselno, int& precselno,
                                double& GUT_SCALE, double& myweakscale, double& inptanbval,
                                std::vector<double>& GUT_boundary_conditions, double& originalmZ2value) {
    // GUT_SCALE and myweakscale should be log(Q)
    vector<LabeledValueBG> dbglist;
    double mymZ_squared = 91.1876 * 91.1876;
    vector<double> derivative_stepsizes = stepsize_generator(modselno, precselno, GUT_boundary_conditions);
    if (modselno == 1) {
        vector<double> mZ_m0_var_vals, mZ_mhf_var_vals, mZ_A0_var_vals, mZ_mu0_var_vals;
        
        vector<double> m0candidates = {GUT_boundary_conditions[27], GUT_boundary_conditions[28], GUT_boundary_conditions[29], GUT_boundary_conditions[30],
                                       GUT_boundary_conditions[31], GUT_boundary_conditions[33], GUT_boundary_conditions[34], GUT_boundary_conditions[35],
                                       GUT_boundary_conditions[36], GUT_boundary_conditions[37], GUT_boundary_conditions[38], GUT_boundary_conditions[39],
                                       GUT_boundary_conditions[40], GUT_boundary_conditions[41], GUT_boundary_conditions[25], GUT_boundary_conditions[26]};
        auto maxm0It = max_element(m0candidates.begin(), m0candidates.end(),
                                   [](double a, double b) { return abs(a) < abs(b); });
        double maxm0Val = copysign(sqrt(abs(*maxm0It)), *maxm0It);
        vector<double> mhfcandidates = {GUT_boundary_conditions[3], GUT_boundary_conditions[4], GUT_boundary_conditions[5]};
        auto maxmhfIt = max_element(mhfcandidates.begin(), mhfcandidates.end());
        double maxGauginoMass = *maxmhfIt;
        vector<double> A0candidates;
        for (int i = 16; i < 25; ++i) {
                A0candidates.push_back(GUT_boundary_conditions[i] / GUT_boundary_conditions[i-9]);
        }
        auto maxA0It = max_element(A0candidates.begin(), A0candidates.end(),
                                   [](double a, double b) { return abs(a) < abs(b); });
        double maxTrilin = *maxA0It;
        double mu0value = GUT_boundary_conditions[6];
        if (precselno == 1) {
            vector<double> m0steps = {(-4.0) * derivative_stepsizes[0], (-3.0) * derivative_stepsizes[0], (-2.0) * derivative_stepsizes[0], (-1.0) * derivative_stepsizes[0],
                                      derivative_stepsizes[0], 2.0 * derivative_stepsizes[0], 3.0 * derivative_stepsizes[0], 4.0 * derivative_stepsizes[0]};
            vector<double> mhfsteps = {(-4.0) * derivative_stepsizes[1], (-3.0) * derivative_stepsizes[1], (-2.0) * derivative_stepsizes[1], (-1.0) * derivative_stepsizes[1],
                                       derivative_stepsizes[1], 2.0 * derivative_stepsizes[1], 3.0 * derivative_stepsizes[1], 4.0 * derivative_stepsizes[1]};
            vector<double> A0steps = {(-4.0) * derivative_stepsizes[2], (-3.0) * derivative_stepsizes[2], (-2.0) * derivative_stepsizes[2], (-1.0) * derivative_stepsizes[2],
                                      derivative_stepsizes[2], 2.0 * derivative_stepsizes[2], 3.0 * derivative_stepsizes[2], 4.0 * derivative_stepsizes[2]};
            vector<double> mu0steps = {(-4.0) * derivative_stepsizes[3], (-3.0) * derivative_stepsizes[3], (-2.0) * derivative_stepsizes[3], (-1.0) * derivative_stepsizes[3],
                                       derivative_stepsizes[3], 2.0 * derivative_stepsizes[3], 3.0 * derivative_stepsizes[3], 4.0 * derivative_stepsizes[3]};
            // cout << "m0steps: " << endl;
            // for (double value : m0steps) {
            //     cout << value << endl;
            // }
            for (int j = 0; j < m0steps.size(); ++j) {
                double step_Resultm0 = deriv_step_calc_scalars(m0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                cout << "Step mZ result with m0 variation: " << endl << step_Resultm0 << endl;
                mZ_m0_var_vals.push_back(step_Resultm0);
                double step_Resultmhf = deriv_step_calc_gaugino(mhfsteps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_mhf_var_vals.push_back(step_Resultmhf);
                double step_ResultA0 = deriv_step_calc_trilin(A0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_A0_var_vals.push_back(step_ResultA0);
                double step_Resultmu0 = deriv_step_calc_mu0(mu0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_mu0_var_vals.push_back(step_Resultmu0);
            }

            cout << "mZ due to m0 variation: " << endl;
            for (double value : mZ_m0_var_vals) {
                cout << value << endl;
            }

            cout << "DBG(m0) = " << (maxm0Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[0], mZ_m0_var_vals) << endl;
            dbglist.push_back({(maxm0Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[0], mZ_m0_var_vals), "Delta_BG(m_0)"});
            dbglist.push_back({(maxGauginoMass / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[1], mZ_mhf_var_vals), "Delta_BG(m_1/2)"});
            dbglist.push_back({(maxTrilin / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[2], mZ_A0_var_vals), "Delta_BG(A_0)"});
            dbglist.push_back({(mu0value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[3], mZ_mu0_var_vals), "Delta_BG(mu_0)"});
        }
        //////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////
        else if (precselno == 2) {
            vector<double> m0steps = {(-2.0) * derivative_stepsizes[0], (-1.0) * derivative_stepsizes[0], derivative_stepsizes[0], 2.0 * derivative_stepsizes[0]};
            vector<double> mhfsteps = {(-2.0) * derivative_stepsizes[1], (-1.0) * derivative_stepsizes[1], derivative_stepsizes[1], 2.0 * derivative_stepsizes[1]};
            vector<double> A0steps = {(-2.0) * derivative_stepsizes[2], (-1.0) * derivative_stepsizes[2], derivative_stepsizes[2], 2.0 * derivative_stepsizes[2]};
            vector<double> mu0steps = {(-2.0) * derivative_stepsizes[3], (-1.0) * derivative_stepsizes[3], derivative_stepsizes[3], 2.0 * derivative_stepsizes[3]};

            for (int j = 0; j < m0steps.size(); ++j) {
                double step_Resultm0 = deriv_step_calc_scalars(m0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_m0_var_vals.push_back(step_Resultm0);
                double step_Resultmhf = deriv_step_calc_gaugino(mhfsteps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_mhf_var_vals.push_back(step_Resultmhf);
                double step_ResultA0 = deriv_step_calc_trilin(A0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_A0_var_vals.push_back(step_ResultA0);
                double step_Resultmu0 = deriv_step_calc_mu0(mu0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_mu0_var_vals.push_back(step_Resultmu0);
            }

            dbglist.push_back({(maxm0Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[0], mZ_m0_var_vals), "Delta_BG(m_0)"});
            dbglist.push_back({(maxGauginoMass / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[1], mZ_mhf_var_vals), "Delta_BG(m_1/2)"});
            dbglist.push_back({(maxTrilin / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[2], mZ_A0_var_vals), "Delta_BG(A_0)"});
            dbglist.push_back({(mu0value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[3], mZ_mu0_var_vals), "Delta_BG(mu_0)"});
        }
        //////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////
        else {
            vector<double> m0steps = {(-1.0) * derivative_stepsizes[0], derivative_stepsizes[0]};
            vector<double> mhfsteps = {(-1.0) * derivative_stepsizes[1], derivative_stepsizes[1]};
            vector<double> A0steps = {(-1.0) * derivative_stepsizes[2], derivative_stepsizes[2]};
            vector<double> mu0steps = {(-1.0) * derivative_stepsizes[3], derivative_stepsizes[3]};
        
            for (int j = 0; j < m0steps.size(); ++j) {
                double step_Resultm0 = deriv_step_calc_scalars(m0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_m0_var_vals.push_back(step_Resultm0);
                double step_Resultmhf = deriv_step_calc_gaugino(mhfsteps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_mhf_var_vals.push_back(step_Resultmhf);
                double step_ResultA0 = deriv_step_calc_trilin(A0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_A0_var_vals.push_back(step_ResultA0);
                double step_Resultmu0 = deriv_step_calc_mu0(mu0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_mu0_var_vals.push_back(step_Resultmu0);
            }

            dbglist.push_back({(maxm0Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[0], mZ_m0_var_vals), "Delta_BG(m_0)"});
            dbglist.push_back({(maxGauginoMass / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[1], mZ_mhf_var_vals), "Delta_BG(m_1/2)"});
            dbglist.push_back({(maxTrilin / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[2], mZ_A0_var_vals), "Delta_BG(A_0)"});
            dbglist.push_back({(mu0value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[3], mZ_mu0_var_vals), "Delta_BG(mu_0)"});
        }
    }
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    else if (modselno == 2) {
        vector<double> mZ_m0_var_vals, mZ_mhf_var_vals, mZ_A0_var_vals, mZ_mu0_var_vals;
        vector<double> mZ_mHud_var_vals;
        auto maxmHudIt = max_element(GUT_boundary_conditions.begin() + 25, GUT_boundary_conditions.begin() + 27,
                                     [](double a, double b) { return abs(a) < abs(b); });
        double maxHiggsMass = copysign(sqrt(abs(*maxmHudIt)),*maxmHudIt);
        vector<double> m0candidates = {GUT_boundary_conditions[27], GUT_boundary_conditions[28], GUT_boundary_conditions[29], GUT_boundary_conditions[30],
                                       GUT_boundary_conditions[31], GUT_boundary_conditions[33], GUT_boundary_conditions[34], GUT_boundary_conditions[35],
                                       GUT_boundary_conditions[36], GUT_boundary_conditions[37], GUT_boundary_conditions[38], GUT_boundary_conditions[39],
                                       GUT_boundary_conditions[40], GUT_boundary_conditions[41]};
        auto maxm0It = max_element(m0candidates.begin(), m0candidates.end(),
                                   [](double a, double b) { return abs(a) < abs(b); });
        double maxm0Val = sqrt(abs(*maxm0It));
        vector<double> mhfcandidates = {GUT_boundary_conditions[3], GUT_boundary_conditions[4], GUT_boundary_conditions[5]};
        auto maxmhfIt = max_element(mhfcandidates.begin(), mhfcandidates.end());
        double maxGauginoMass = *maxmhfIt;
        vector<double> A0candidates;
        for (int i = 16; i < 25; ++i) {
                A0candidates.push_back(GUT_boundary_conditions[i] / GUT_boundary_conditions[i-9]);
        }
        auto maxA0It = max_element(A0candidates.begin(), A0candidates.end(),
                                   [](double a, double b) { return abs(a) < abs(b); });
        double maxTrilin = *maxA0It;
        double mu0value = GUT_boundary_conditions[6];
        if (precselno == 1) {
            vector<double> mHud0steps = {(-4.0) * derivative_stepsizes[0], (-3.0) * derivative_stepsizes[0], (-2.0) * derivative_stepsizes[0], (-1.0) * derivative_stepsizes[0],
                                         derivative_stepsizes[0], 2.0 * derivative_stepsizes[0], 3.0 * derivative_stepsizes[0], 4.0 * derivative_stepsizes[0]};
            vector<double> m0steps = {(-4.0) * derivative_stepsizes[1], (-3.0) * derivative_stepsizes[1], (-2.0) * derivative_stepsizes[1], (-1.0) * derivative_stepsizes[1],
                                      derivative_stepsizes[1], 2.0 * derivative_stepsizes[1], 3.0 * derivative_stepsizes[1], 4.0 * derivative_stepsizes[1]};
            vector<double> mhfsteps = {(-4.0) * derivative_stepsizes[2], (-3.0) * derivative_stepsizes[2], (-2.0) * derivative_stepsizes[2], (-1.0) * derivative_stepsizes[2],
                                       derivative_stepsizes[2], 2.0 * derivative_stepsizes[2], 3.0 * derivative_stepsizes[2], 4.0 * derivative_stepsizes[2]};
            vector<double> A0steps = {(-4.0) * derivative_stepsizes[3], (-3.0) * derivative_stepsizes[3], (-2.0) * derivative_stepsizes[3], (-1.0) * derivative_stepsizes[3],
                                      derivative_stepsizes[3], 2.0 * derivative_stepsizes[3], 3.0 * derivative_stepsizes[3], 4.0 * derivative_stepsizes[3]};
            vector<double> mu0steps = {(-4.0) * derivative_stepsizes[4], (-3.0) * derivative_stepsizes[4], (-2.0) * derivative_stepsizes[4], (-1.0) * derivative_stepsizes[4],
                                      derivative_stepsizes[4], 2.0 * derivative_stepsizes[4], 3.0 * derivative_stepsizes[4], 4.0 * derivative_stepsizes[4]};

            for (int j = 0; j < m0steps.size(); ++j) {
                double step_Resultm0 = deriv_step_calc_customRange(m0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, 27, 42);
                mZ_m0_var_vals.push_back(step_Resultm0);
                double step_Resultmhf = deriv_step_calc_gaugino(mhfsteps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_mhf_var_vals.push_back(step_Resultmhf);
                double step_ResultA0 = deriv_step_calc_trilin(A0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_A0_var_vals.push_back(step_ResultA0);
                double step_Resultmu0 = deriv_step_calc_mu0(mu0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_mu0_var_vals.push_back(step_Resultmu0);
                double step_ResultmHud0 = deriv_step_calc_customRange(mHud0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, 25, 27);
                mZ_mHud_var_vals.push_back(step_ResultmHud0);
            }

            dbglist.push_back({(maxm0Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[1], mZ_m0_var_vals), "Delta_BG(m_0)"});
            dbglist.push_back({(maxGauginoMass / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[2], mZ_mhf_var_vals), "Delta_BG(m_1/2)"});
            dbglist.push_back({(maxTrilin / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[3], mZ_A0_var_vals), "Delta_BG(A_0)"});
            dbglist.push_back({(mu0value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[4], mZ_mu0_var_vals), "Delta_BG(mu_0)"});
            dbglist.push_back({(maxHiggsMass / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[0], mZ_mu0_var_vals), "Delta_BG(mHu,d)"});
        }
        //////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////
        else if (precselno == 2) {
            vector<double> mHud0steps = {(-2.0) * derivative_stepsizes[0], (-1.0) * derivative_stepsizes[0],
                                         derivative_stepsizes[0], 2.0 * derivative_stepsizes[0]};
            vector<double> m0steps = {(-2.0) * derivative_stepsizes[1], (-1.0) * derivative_stepsizes[1],
                                      derivative_stepsizes[1], 2.0 * derivative_stepsizes[1]};
            vector<double> mhfsteps = {(-2.0) * derivative_stepsizes[2], (-1.0) * derivative_stepsizes[2],
                                       derivative_stepsizes[2], 2.0 * derivative_stepsizes[2]};
            vector<double> A0steps = {(-2.0) * derivative_stepsizes[3], (-1.0) * derivative_stepsizes[3],
                                      derivative_stepsizes[3], 2.0 * derivative_stepsizes[3]};
            vector<double> mu0steps = {(-2.0) * derivative_stepsizes[4], (-1.0) * derivative_stepsizes[4],
                                       derivative_stepsizes[4], 2.0 * derivative_stepsizes[4]};

            for (int j = 0; j < m0steps.size(); ++j) {
                double step_Resultm0 = deriv_step_calc_customRange(m0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, 27, 42);
                mZ_m0_var_vals.push_back(step_Resultm0);
                double step_Resultmhf = deriv_step_calc_gaugino(mhfsteps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_mhf_var_vals.push_back(step_Resultmhf);
                double step_ResultA0 = deriv_step_calc_trilin(A0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_A0_var_vals.push_back(step_ResultA0);
                double step_Resultmu0 = deriv_step_calc_mu0(mu0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_mu0_var_vals.push_back(step_Resultmu0);
                double step_ResultmHud0 = deriv_step_calc_customRange(mHud0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, 25, 27);
                mZ_mHud_var_vals.push_back(step_ResultmHud0);
            }

            dbglist.push_back({(maxm0Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[1], mZ_m0_var_vals), "Delta_BG(m_0)"});
            dbglist.push_back({(maxGauginoMass / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[2], mZ_mhf_var_vals), "Delta_BG(m_1/2)"});
            dbglist.push_back({(maxTrilin / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[3], mZ_A0_var_vals), "Delta_BG(A_0)"});
            dbglist.push_back({(mu0value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[4], mZ_mu0_var_vals), "Delta_BG(mu_0)"});
            dbglist.push_back({(maxHiggsMass / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[0], mZ_mu0_var_vals), "Delta_BG(mHu,d)"});
        }
        //////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////
        else {
            vector<double> mHud0steps = {(-1.0) * derivative_stepsizes[0],
                                         derivative_stepsizes[0]};
            vector<double> m0steps = {(-1.0) * derivative_stepsizes[1],
                                      derivative_stepsizes[1]};
            vector<double> mhfsteps = {(-1.0) * derivative_stepsizes[2],
                                       derivative_stepsizes[2]};
            vector<double> A0steps = {(-1.0) * derivative_stepsizes[3],
                                      derivative_stepsizes[3]};
            vector<double> mu0steps = {(-1.0) * derivative_stepsizes[4],
                                       derivative_stepsizes[4]};

            for (int j = 0; j < m0steps.size(); ++j) {
                double step_Resultm0 = deriv_step_calc_customRange(m0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, 27, 42);
                mZ_m0_var_vals.push_back(step_Resultm0);
                double step_Resultmhf = deriv_step_calc_gaugino(mhfsteps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_mhf_var_vals.push_back(step_Resultmhf);
                double step_ResultA0 = deriv_step_calc_trilin(A0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_A0_var_vals.push_back(step_ResultA0);
                double step_Resultmu0 = deriv_step_calc_mu0(mu0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_mu0_var_vals.push_back(step_Resultmu0);
                double step_ResultmHud0 = deriv_step_calc_customRange(mHud0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, 25, 27);
                mZ_mHud_var_vals.push_back(step_ResultmHud0);
            }

            dbglist.push_back({(maxm0Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[1], mZ_m0_var_vals), "Delta_BG(m_0)"});
            dbglist.push_back({(maxGauginoMass / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[2], mZ_mhf_var_vals), "Delta_BG(m_1/2)"});
            dbglist.push_back({(maxTrilin / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[3], mZ_A0_var_vals), "Delta_BG(A_0)"});
            dbglist.push_back({(mu0value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[4], mZ_mu0_var_vals), "Delta_BG(mu_0)"});
            dbglist.push_back({(maxHiggsMass / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[0], mZ_mu0_var_vals), "Delta_BG(mHu,d)"});
        }
    }
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    else if (modselno == 3) {
        vector<double> mZ_m0_var_vals, mZ_mhf_var_vals, mZ_A0_var_vals, mZ_mu0_var_vals;
        vector<double> mZ_mHu_var_vals, mZ_mHd_var_vals;
        double mHu0Value = copysign(sqrt(abs(GUT_boundary_conditions[25])), GUT_boundary_conditions[25]);
        double mHd0Value = copysign(sqrt(abs(GUT_boundary_conditions[26])), GUT_boundary_conditions[26]);
        vector<double> m0candidates = {GUT_boundary_conditions[27], GUT_boundary_conditions[28], GUT_boundary_conditions[29], GUT_boundary_conditions[30],
                                       GUT_boundary_conditions[31], GUT_boundary_conditions[33], GUT_boundary_conditions[34], GUT_boundary_conditions[35],
                                       GUT_boundary_conditions[36], GUT_boundary_conditions[37], GUT_boundary_conditions[38], GUT_boundary_conditions[39],
                                       GUT_boundary_conditions[40], GUT_boundary_conditions[41]};
        auto maxm0It = max_element(m0candidates.begin(), m0candidates.end(),
                                   [](double a, double b) { return abs(a) < abs(b); });
        double maxm0Val = sqrt(abs(*maxm0It));
        vector<double> mhfcandidates = {GUT_boundary_conditions[3], GUT_boundary_conditions[4], GUT_boundary_conditions[5]};
        auto maxmhfIt = max_element(mhfcandidates.begin(), mhfcandidates.end());
        double maxGauginoMass = *maxmhfIt;
        vector<double> A0candidates;
        for (int i = 16; i < 25; ++i) {
                A0candidates.push_back(GUT_boundary_conditions[i] / GUT_boundary_conditions[i-9]);
        }
        auto maxA0It = max_element(A0candidates.begin(), A0candidates.end(),
                                   [](double a, double b) { return abs(a) < abs(b); });
        double maxTrilin = *maxA0It;
        double mu0value = GUT_boundary_conditions[6];
        if (precselno == 1) {
            vector<double> mHu0steps = {(-4.0) * derivative_stepsizes[0], (-3.0) * derivative_stepsizes[0], (-2.0) * derivative_stepsizes[0], (-1.0) * derivative_stepsizes[0],
                                        derivative_stepsizes[0], 2.0 * derivative_stepsizes[0], 3.0 * derivative_stepsizes[0], 4.0 * derivative_stepsizes[0]};
            vector<double> mHd0steps = {(-4.0) * derivative_stepsizes[1], (-3.0) * derivative_stepsizes[1], (-2.0) * derivative_stepsizes[1], (-1.0) * derivative_stepsizes[1],
                                         derivative_stepsizes[1], 2.0 * derivative_stepsizes[1], 3.0 * derivative_stepsizes[1], 4.0 * derivative_stepsizes[1]};
            vector<double> m0steps = {(-4.0) * derivative_stepsizes[2], (-3.0) * derivative_stepsizes[2], (-2.0) * derivative_stepsizes[2], (-1.0) * derivative_stepsizes[2],
                                      derivative_stepsizes[2], 2.0 * derivative_stepsizes[2], 3.0 * derivative_stepsizes[2], 4.0 * derivative_stepsizes[2]};
            vector<double> mhfsteps = {(-4.0) * derivative_stepsizes[3], (-3.0) * derivative_stepsizes[3], (-2.0) * derivative_stepsizes[3], (-1.0) * derivative_stepsizes[3],
                                       derivative_stepsizes[3], 2.0 * derivative_stepsizes[3], 3.0 * derivative_stepsizes[3], 4.0 * derivative_stepsizes[3]};
            vector<double> A0steps = {(-4.0) * derivative_stepsizes[4], (-3.0) * derivative_stepsizes[4], (-2.0) * derivative_stepsizes[4], (-1.0) * derivative_stepsizes[4],
                                      derivative_stepsizes[4], 2.0 * derivative_stepsizes[4], 3.0 * derivative_stepsizes[4], 4.0 * derivative_stepsizes[4]};
            vector<double> mu0steps = {(-4.0) * derivative_stepsizes[5], (-3.0) * derivative_stepsizes[5], (-2.0) * derivative_stepsizes[5], (-1.0) * derivative_stepsizes[5],
                                       derivative_stepsizes[5], 2.0 * derivative_stepsizes[5], 3.0 * derivative_stepsizes[5], 4.0 * derivative_stepsizes[5]};
            
            for (int j = 0; j < m0steps.size(); ++j) {
                double step_Resultm0 = deriv_step_calc_customRange(m0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, 27, 42);
                mZ_m0_var_vals.push_back(step_Resultm0);
                double step_Resultmhf = deriv_step_calc_gaugino(mhfsteps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_mhf_var_vals.push_back(step_Resultmhf);
                double step_ResultA0 = deriv_step_calc_trilin(A0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_A0_var_vals.push_back(step_ResultA0);
                double step_Resultmu0 = deriv_step_calc_mu0(mu0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_mu0_var_vals.push_back(step_Resultmu0);
                double step_ResultmHu0 = gen_deriv_step_scalar_calc(mHu0steps[j], GUT_boundary_conditions, 25, GUT_SCALE, myweakscale);
                mZ_mHu_var_vals.push_back(step_ResultmHu0);
                double step_ResultmHd0 = gen_deriv_step_scalar_calc(mHd0steps[j], GUT_boundary_conditions, 26, GUT_SCALE, myweakscale);
                mZ_mHd_var_vals.push_back(step_ResultmHd0);
            }            

            dbglist.push_back({(maxm0Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[2], mZ_m0_var_vals), "Delta_BG(m_0)"});
            dbglist.push_back({(maxGauginoMass / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[3], mZ_mhf_var_vals), "Delta_BG(m_1/2)"});
            dbglist.push_back({(maxTrilin / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[4], mZ_A0_var_vals), "Delta_BG(A_0)"});
            dbglist.push_back({(mu0value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[5], mZ_mu0_var_vals), "Delta_BG(mu_0)"});
            dbglist.push_back({(mHu0Value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[0], mZ_mHu_var_vals), "Delta_BG(mHu)"});
            dbglist.push_back({(mHd0Value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[1], mZ_mHd_var_vals), "Delta_BG(mHd)"});
        }
        //////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////
        else if (precselno == 2) {
            vector<double> mHu0steps = {(-2.0) * derivative_stepsizes[0], (-1.0) * derivative_stepsizes[0],
                                        derivative_stepsizes[0], 2.0 * derivative_stepsizes[0]};
            vector<double> mHd0steps = {(-2.0) * derivative_stepsizes[1], (-1.0) * derivative_stepsizes[1],
                                        derivative_stepsizes[1], 2.0 * derivative_stepsizes[1]};
            vector<double> m0steps = {(-2.0) * derivative_stepsizes[2], (-1.0) * derivative_stepsizes[2],
                                      derivative_stepsizes[2], 2.0 * derivative_stepsizes[2]};
            vector<double> mhfsteps = {(-2.0) * derivative_stepsizes[3], (-1.0) * derivative_stepsizes[3],
                                       derivative_stepsizes[3], 2.0 * derivative_stepsizes[3]};
            vector<double> A0steps = {(-2.0) * derivative_stepsizes[4], (-1.0) * derivative_stepsizes[4],
                                      derivative_stepsizes[4], 2.0 * derivative_stepsizes[4]};
            vector<double> mu0steps = {(-2.0) * derivative_stepsizes[5], (-1.0) * derivative_stepsizes[5],
                                       derivative_stepsizes[5], 2.0 * derivative_stepsizes[5]};
            
            for (int j = 0; j < m0steps.size(); ++j) {
                double step_Resultm0 = deriv_step_calc_customRange(m0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, 27, 42);
                mZ_m0_var_vals.push_back(step_Resultm0);
                double step_Resultmhf = deriv_step_calc_gaugino(mhfsteps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_mhf_var_vals.push_back(step_Resultmhf);
                double step_ResultA0 = deriv_step_calc_trilin(A0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_A0_var_vals.push_back(step_ResultA0);
                double step_Resultmu0 = deriv_step_calc_mu0(mu0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_mu0_var_vals.push_back(step_Resultmu0);
                double step_ResultmHu0 = gen_deriv_step_scalar_calc(mHu0steps[j], GUT_boundary_conditions, 25, GUT_SCALE, myweakscale);
                mZ_mHu_var_vals.push_back(step_ResultmHu0);
                double step_ResultmHd0 = gen_deriv_step_scalar_calc(mHd0steps[j], GUT_boundary_conditions, 26, GUT_SCALE, myweakscale);
                mZ_mHd_var_vals.push_back(step_ResultmHd0);
            }            

            dbglist.push_back({(maxm0Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[2], mZ_m0_var_vals), "Delta_BG(m_0)"});
            dbglist.push_back({(maxGauginoMass / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[3], mZ_mhf_var_vals), "Delta_BG(m_1/2)"});
            dbglist.push_back({(maxTrilin / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[4], mZ_A0_var_vals), "Delta_BG(A_0)"});
            dbglist.push_back({(mu0value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[5], mZ_mu0_var_vals), "Delta_BG(mu_0)"});
            dbglist.push_back({(mHu0Value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[0], mZ_mHu_var_vals), "Delta_BG(mHu)"});
            dbglist.push_back({(mHd0Value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[1], mZ_mHd_var_vals), "Delta_BG(mHd)"});
        }
        //////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////
        else {
            vector<double> mHu0steps = {(-1.0) * derivative_stepsizes[0],
                                        derivative_stepsizes[0]};
            vector<double> mHd0steps = {(-1.0) * derivative_stepsizes[1],
                                        derivative_stepsizes[1]};
            vector<double> m0steps = {(-1.0) * derivative_stepsizes[2],
                                      derivative_stepsizes[2]};
            vector<double> mhfsteps = {(-1.0) * derivative_stepsizes[3],
                                       derivative_stepsizes[3]};
            vector<double> A0steps = {(-1.0) * derivative_stepsizes[4],
                                      derivative_stepsizes[4]};
            vector<double> mu0steps = {(-1.0) * derivative_stepsizes[5],
                                       derivative_stepsizes[5]};
            
            for (int j = 0; j < m0steps.size(); ++j) {
                double step_Resultm0 = deriv_step_calc_customRange(m0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, 27, 42);
                mZ_m0_var_vals.push_back(step_Resultm0);
                double step_Resultmhf = deriv_step_calc_gaugino(mhfsteps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_mhf_var_vals.push_back(step_Resultmhf);
                double step_ResultA0 = deriv_step_calc_trilin(A0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_A0_var_vals.push_back(step_ResultA0);
                double step_Resultmu0 = deriv_step_calc_mu0(mu0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_mu0_var_vals.push_back(step_Resultmu0);
                double step_ResultmHu0 = gen_deriv_step_scalar_calc(mHu0steps[j], GUT_boundary_conditions, 25, GUT_SCALE, myweakscale);
                mZ_mHu_var_vals.push_back(step_ResultmHu0);
                double step_ResultmHd0 = gen_deriv_step_scalar_calc(mHd0steps[j], GUT_boundary_conditions, 26, GUT_SCALE, myweakscale);
                mZ_mHd_var_vals.push_back(step_ResultmHd0);
            }            

            dbglist.push_back({(maxm0Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[2], mZ_m0_var_vals), "Delta_BG(m_0)"});
            dbglist.push_back({(maxGauginoMass / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[3], mZ_mhf_var_vals), "Delta_BG(m_1/2)"});
            dbglist.push_back({(maxTrilin / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[4], mZ_A0_var_vals), "Delta_BG(A_0)"});
            dbglist.push_back({(mu0value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[5], mZ_mu0_var_vals), "Delta_BG(mu_0)"});
            dbglist.push_back({(mHu0Value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[0], mZ_mHu_var_vals), "Delta_BG(mHu)"});
            dbglist.push_back({(mHd0Value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[1], mZ_mHd_var_vals), "Delta_BG(mHd)"});
        
        }
    }
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    else if (modselno == 4) {
        vector<double> mZ_m012_var_vals, mZ_m03_var_vals, mZ_mhf_var_vals, mZ_A0_var_vals, mZ_mu0_var_vals;
        vector<double> mZ_mHu_var_vals, mZ_mHd_var_vals;
        double mHu0Value = copysign(sqrt(abs(GUT_boundary_conditions[25])), GUT_boundary_conditions[25]);
        double mHd0Value = copysign(sqrt(abs(GUT_boundary_conditions[26])), GUT_boundary_conditions[26]);
        vector<double> m012candidates = {GUT_boundary_conditions[27], GUT_boundary_conditions[28], GUT_boundary_conditions[30],
                                         GUT_boundary_conditions[31], GUT_boundary_conditions[33], GUT_boundary_conditions[34],
                                         GUT_boundary_conditions[36], GUT_boundary_conditions[37], GUT_boundary_conditions[39],
                                         GUT_boundary_conditions[40]};
        auto maxm012It = max_element(m012candidates.begin(), m012candidates.end(),
                                     [](double a, double b) { return abs(a) < abs(b); });
        double maxm012Val = sqrt(abs(*maxm012It));
        vector<double> m03candidates = {GUT_boundary_conditions[29], GUT_boundary_conditions[32], GUT_boundary_conditions[35], GUT_boundary_conditions[38],
                                        GUT_boundary_conditions[41]};
        auto maxm03It = max_element(m03candidates.begin(), m03candidates.end(),
                                     [](double a, double b) { return abs(a) < abs(b); });
        double maxm03Val = sqrt(abs(*maxm03It));
        vector<double> mhfcandidates = {GUT_boundary_conditions[3], GUT_boundary_conditions[4], GUT_boundary_conditions[5]};
        auto maxmhfIt = max_element(mhfcandidates.begin(), mhfcandidates.end());
        double maxGauginoMass = *maxmhfIt;
        vector<double> A0candidates;
        for (int i = 16; i < 25; ++i) {
                A0candidates.push_back(GUT_boundary_conditions[i] / GUT_boundary_conditions[i-9]);
        }
        auto maxA0It = max_element(A0candidates.begin(), A0candidates.end(),
                                   [](double a, double b) { return abs(a) < abs(b); });
        double maxTrilin = *maxA0It;
        double mu0value = GUT_boundary_conditions[6];
        if (precselno == 1) {
            vector<double> mHu0steps = {(-4.0) * derivative_stepsizes[0], (-3.0) * derivative_stepsizes[0], (-2.0) * derivative_stepsizes[0], (-1.0) * derivative_stepsizes[0],
                                        derivative_stepsizes[0], 2.0 * derivative_stepsizes[0], 3.0 * derivative_stepsizes[0], 4.0 * derivative_stepsizes[0]};
            vector<double> mHd0steps = {(-4.0) * derivative_stepsizes[1], (-3.0) * derivative_stepsizes[1], (-2.0) * derivative_stepsizes[1], (-1.0) * derivative_stepsizes[1],
                                         derivative_stepsizes[1], 2.0 * derivative_stepsizes[1], 3.0 * derivative_stepsizes[1], 4.0 * derivative_stepsizes[1]};
            vector<double> m012steps = {(-4.0) * derivative_stepsizes[2], (-3.0) * derivative_stepsizes[2], (-2.0) * derivative_stepsizes[2], (-1.0) * derivative_stepsizes[2],
                                        derivative_stepsizes[2], 2.0 * derivative_stepsizes[2], 3.0 * derivative_stepsizes[2], 4.0 * derivative_stepsizes[2]};
            vector<double> m03steps = {(-4.0) * derivative_stepsizes[3], (-3.0) * derivative_stepsizes[3], (-2.0) * derivative_stepsizes[3], (-1.0) * derivative_stepsizes[3],
                                       derivative_stepsizes[3], 2.0 * derivative_stepsizes[3], 3.0 * derivative_stepsizes[3], 4.0 * derivative_stepsizes[3]};
            vector<double> mhfsteps = {(-4.0) * derivative_stepsizes[4], (-3.0) * derivative_stepsizes[4], (-2.0) * derivative_stepsizes[4], (-1.0) * derivative_stepsizes[4],
                                       derivative_stepsizes[4], 2.0 * derivative_stepsizes[4], 3.0 * derivative_stepsizes[4], 4.0 * derivative_stepsizes[4]};
            vector<double> A0steps = {(-4.0) * derivative_stepsizes[5], (-3.0) * derivative_stepsizes[5], (-2.0) * derivative_stepsizes[5], (-1.0) * derivative_stepsizes[5],
                                      derivative_stepsizes[5], 2.0 * derivative_stepsizes[5], 3.0 * derivative_stepsizes[5], 4.0 * derivative_stepsizes[5]};
            vector<double> mu0steps = {(-4.0) * derivative_stepsizes[6], (-3.0) * derivative_stepsizes[6], (-2.0) * derivative_stepsizes[6], (-1.0) * derivative_stepsizes[6],
                                       derivative_stepsizes[6], 2.0 * derivative_stepsizes[6], 3.0 * derivative_stepsizes[6], 4.0 * derivative_stepsizes[6]};
            
            for (int j = 0; j < m03steps.size(); ++j) {
                double step_Resultm012 = deriv_step_calc_genScalarIndices(m012steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, {27, 28, 30, 31, 33, 34, 36, 37, 39, 40});
                mZ_m012_var_vals.push_back(step_Resultm012);
                double step_Resultm03 = deriv_step_calc_genScalarIndices(m03steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, {29, 32, 35, 38, 41});
                mZ_m03_var_vals.push_back(step_Resultm03);
                double step_Resultmhf = deriv_step_calc_gaugino(mhfsteps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_mhf_var_vals.push_back(step_Resultmhf);
                double step_ResultA0 = deriv_step_calc_trilin(A0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_A0_var_vals.push_back(step_ResultA0);
                double step_Resultmu0 = deriv_step_calc_mu0(mu0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_mu0_var_vals.push_back(step_Resultmu0);
                double step_ResultmHu0 = gen_deriv_step_scalar_calc(mHu0steps[j], GUT_boundary_conditions, 25, GUT_SCALE, myweakscale);
                mZ_mHu_var_vals.push_back(step_ResultmHu0);
                double step_ResultmHd0 = gen_deriv_step_scalar_calc(mHd0steps[j], GUT_boundary_conditions, 26, GUT_SCALE, myweakscale);
                mZ_mHd_var_vals.push_back(step_ResultmHd0);
            }            

            dbglist.push_back({(maxm012Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[2], mZ_m012_var_vals), "Delta_BG(m_0(1,2))"});
            dbglist.push_back({(maxm03Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[3], mZ_m03_var_vals), "Delta_BG(m_0(3))"});
            dbglist.push_back({(maxGauginoMass / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[4], mZ_mhf_var_vals), "Delta_BG(m_1/2)"});
            dbglist.push_back({(maxTrilin / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[5], mZ_A0_var_vals), "Delta_BG(A_0)"});
            dbglist.push_back({(mu0value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[6], mZ_mu0_var_vals), "Delta_BG(mu_0)"});
            dbglist.push_back({(mHu0Value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[0], mZ_mHu_var_vals), "Delta_BG(mHu)"});
            dbglist.push_back({(mHd0Value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[1], mZ_mHd_var_vals), "Delta_BG(mHd)"});
        }
        //////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////
        else if (precselno == 2) {
            vector<double> mHu0steps = {(-2.0) * derivative_stepsizes[0], (-1.0) * derivative_stepsizes[0],
                                        derivative_stepsizes[0], 2.0 * derivative_stepsizes[0]};
            vector<double> mHd0steps = {(-2.0) * derivative_stepsizes[1], (-1.0) * derivative_stepsizes[1],
                                         derivative_stepsizes[1], 2.0 * derivative_stepsizes[1]};
            vector<double> m012steps = {(-2.0) * derivative_stepsizes[2], (-1.0) * derivative_stepsizes[2],
                                        derivative_stepsizes[2], 2.0 * derivative_stepsizes[2]};
            vector<double> m03steps = {(-2.0) * derivative_stepsizes[3], (-1.0) * derivative_stepsizes[3],
                                       derivative_stepsizes[3], 2.0 * derivative_stepsizes[3]};
            vector<double> mhfsteps = {(-2.0) * derivative_stepsizes[4], (-1.0) * derivative_stepsizes[4],
                                       derivative_stepsizes[4], 2.0 * derivative_stepsizes[4]};
            vector<double> A0steps = {(-2.0) * derivative_stepsizes[5], (-1.0) * derivative_stepsizes[5],
                                      derivative_stepsizes[5], 2.0 * derivative_stepsizes[5]};
            vector<double> mu0steps = {(-2.0) * derivative_stepsizes[6], (-1.0) * derivative_stepsizes[6],
                                       derivative_stepsizes[6], 2.0 * derivative_stepsizes[6]};
            
            for (int j = 0; j < m03steps.size(); ++j) {
                double step_Resultm012 = deriv_step_calc_genScalarIndices(m012steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, {27, 28, 30, 31, 33, 34, 36, 37, 39, 40});
                mZ_m012_var_vals.push_back(step_Resultm012);
                double step_Resultm03 = deriv_step_calc_genScalarIndices(m03steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, {29, 32, 35, 38, 41});
                mZ_m03_var_vals.push_back(step_Resultm03);
                double step_Resultmhf = deriv_step_calc_gaugino(mhfsteps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_mhf_var_vals.push_back(step_Resultmhf);
                double step_ResultA0 = deriv_step_calc_trilin(A0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_A0_var_vals.push_back(step_ResultA0);
                double step_Resultmu0 = deriv_step_calc_mu0(mu0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_mu0_var_vals.push_back(step_Resultmu0);
                double step_ResultmHu0 = gen_deriv_step_scalar_calc(mHu0steps[j], GUT_boundary_conditions, 25, GUT_SCALE, myweakscale);
                mZ_mHu_var_vals.push_back(step_ResultmHu0);
                double step_ResultmHd0 = gen_deriv_step_scalar_calc(mHd0steps[j], GUT_boundary_conditions, 26, GUT_SCALE, myweakscale);
                mZ_mHd_var_vals.push_back(step_ResultmHd0);
            }            

            dbglist.push_back({(maxm012Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[2], mZ_m012_var_vals), "Delta_BG(m_0(1,2))"});
            dbglist.push_back({(maxm03Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[3], mZ_m03_var_vals), "Delta_BG(m_0(3))"});
            dbglist.push_back({(maxGauginoMass / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[4], mZ_mhf_var_vals), "Delta_BG(m_1/2)"});
            dbglist.push_back({(maxTrilin / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[5], mZ_A0_var_vals), "Delta_BG(A_0)"});
            dbglist.push_back({(mu0value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[6], mZ_mu0_var_vals), "Delta_BG(mu_0)"});
            dbglist.push_back({(mHu0Value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[0], mZ_mHu_var_vals), "Delta_BG(mHu)"});
            dbglist.push_back({(mHd0Value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[1], mZ_mHd_var_vals), "Delta_BG(mHd)"});
        }
        //////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////
        else {
            vector<double> mHu0steps = {(-1.0) * derivative_stepsizes[0],
                                        derivative_stepsizes[0]};
            vector<double> mHd0steps = {(-1.0) * derivative_stepsizes[1],
                                         derivative_stepsizes[1]};
            vector<double> m012steps = {(-1.0) * derivative_stepsizes[2],
                                        derivative_stepsizes[2]};
            vector<double> m03steps = {(-1.0) * derivative_stepsizes[3],
                                       derivative_stepsizes[3]};
            vector<double> mhfsteps = {(-1.0) * derivative_stepsizes[4],
                                       derivative_stepsizes[4]};
            vector<double> A0steps = {(-1.0) * derivative_stepsizes[5],
                                      derivative_stepsizes[5]};
            vector<double> mu0steps = {(-1.0) * derivative_stepsizes[6],
                                       derivative_stepsizes[6]};
            
            for (int j = 0; j < m03steps.size(); ++j) {
                double step_Resultm012 = deriv_step_calc_genScalarIndices(m012steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, {27, 28, 30, 31, 33, 34, 36, 37, 39, 40});
                mZ_m012_var_vals.push_back(step_Resultm012);
                double step_Resultm03 = deriv_step_calc_genScalarIndices(m03steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, {29, 32, 35, 38, 41});
                mZ_m03_var_vals.push_back(step_Resultm03);
                double step_Resultmhf = deriv_step_calc_gaugino(mhfsteps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_mhf_var_vals.push_back(step_Resultmhf);
                double step_ResultA0 = deriv_step_calc_trilin(A0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_A0_var_vals.push_back(step_ResultA0);
                double step_Resultmu0 = deriv_step_calc_mu0(mu0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_mu0_var_vals.push_back(step_Resultmu0);
                double step_ResultmHu0 = gen_deriv_step_scalar_calc(mHu0steps[j], GUT_boundary_conditions, 25, GUT_SCALE, myweakscale);
                mZ_mHu_var_vals.push_back(step_ResultmHu0);
                double step_ResultmHd0 = gen_deriv_step_scalar_calc(mHd0steps[j], GUT_boundary_conditions, 26, GUT_SCALE, myweakscale);
                mZ_mHd_var_vals.push_back(step_ResultmHd0);
            }            

            dbglist.push_back({(maxm012Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[2], mZ_m012_var_vals), "Delta_BG(m_0(1,2))"});
            dbglist.push_back({(maxm03Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[3], mZ_m03_var_vals), "Delta_BG(m_0(3))"});
            dbglist.push_back({(maxGauginoMass / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[4], mZ_mhf_var_vals), "Delta_BG(m_1/2)"});
            dbglist.push_back({(maxTrilin / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[5], mZ_A0_var_vals), "Delta_BG(A_0)"});
            dbglist.push_back({(mu0value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[6], mZ_mu0_var_vals), "Delta_BG(mu_0)"});
            dbglist.push_back({(mHu0Value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[0], mZ_mHu_var_vals), "Delta_BG(mHu)"});
            dbglist.push_back({(mHd0Value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[1], mZ_mHd_var_vals), "Delta_BG(mHd)"});
        }
    }
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    else if (modselno == 5) {
        vector<double> mZ_m01_var_vals, mZ_m02_var_vals, mZ_m03_var_vals, mZ_mhf_var_vals, mZ_A0_var_vals, mZ_mu0_var_vals;
        vector<double> mZ_mHu_var_vals, mZ_mHd_var_vals;
        double mHu0Value = copysign(sqrt(abs(GUT_boundary_conditions[25])), GUT_boundary_conditions[25]);
        double mHd0Value = copysign(sqrt(abs(GUT_boundary_conditions[26])), GUT_boundary_conditions[26]);
        vector<double> m01candidates = {GUT_boundary_conditions[27], GUT_boundary_conditions[30],
                                        GUT_boundary_conditions[33], GUT_boundary_conditions[36],
                                        GUT_boundary_conditions[39]};
        vector<double> m02candidates = {GUT_boundary_conditions[28], GUT_boundary_conditions[31],
                                        GUT_boundary_conditions[34], GUT_boundary_conditions[37],
                                        GUT_boundary_conditions[40]};
        auto maxm01It = max_element(m01candidates.begin(), m01candidates.end(),
                                     [](double a, double b) { return abs(a) < abs(b); });
        double maxm01Val = sqrt(abs(*maxm01It));
        auto maxm02It = max_element(m02candidates.begin(), m02candidates.end(),
                                     [](double a, double b) { return abs(a) < abs(b); });
        double maxm02Val = sqrt(abs(*maxm02It));
        vector<double> m03candidates = {GUT_boundary_conditions[29], GUT_boundary_conditions[32], GUT_boundary_conditions[35], GUT_boundary_conditions[38],
                                        GUT_boundary_conditions[41]};
        auto maxm03It = max_element(m03candidates.begin(), m03candidates.end(),
                                     [](double a, double b) { return abs(a) < abs(b); });
        double maxm03Val = sqrt(abs(*maxm03It));
        vector<double> mhfcandidates = {GUT_boundary_conditions[3], GUT_boundary_conditions[4], GUT_boundary_conditions[5]};
        auto maxmhfIt = max_element(mhfcandidates.begin(), mhfcandidates.end());
        double maxGauginoMass = *maxmhfIt;
        vector<double> A0candidates;
        for (int i = 16; i < 25; ++i) {
                A0candidates.push_back(GUT_boundary_conditions[i] / GUT_boundary_conditions[i-9]);
        }
        auto maxA0It = max_element(A0candidates.begin(), A0candidates.end(),
                                   [](double a, double b) { return abs(a) < abs(b); });
        double maxTrilin = *maxA0It;
        double mu0value = GUT_boundary_conditions[6];
        if (precselno == 1) {
            vector<double> mHu0steps = {(-4.0) * derivative_stepsizes[0], (-3.0) * derivative_stepsizes[0], (-2.0) * derivative_stepsizes[0], (-1.0) * derivative_stepsizes[0],
                                        derivative_stepsizes[0], 2.0 * derivative_stepsizes[0], 3.0 * derivative_stepsizes[0], 4.0 * derivative_stepsizes[0]};
            vector<double> mHd0steps = {(-4.0) * derivative_stepsizes[1], (-3.0) * derivative_stepsizes[1], (-2.0) * derivative_stepsizes[1], (-1.0) * derivative_stepsizes[1],
                                         derivative_stepsizes[1], 2.0 * derivative_stepsizes[1], 3.0 * derivative_stepsizes[1], 4.0 * derivative_stepsizes[1]};
            vector<double> m01steps = {(-4.0) * derivative_stepsizes[2], (-3.0) * derivative_stepsizes[2], (-2.0) * derivative_stepsizes[2], (-1.0) * derivative_stepsizes[2],
                                        derivative_stepsizes[2], 2.0 * derivative_stepsizes[2], 3.0 * derivative_stepsizes[2], 4.0 * derivative_stepsizes[2]};
            vector<double> m02steps = {(-4.0) * derivative_stepsizes[3], (-3.0) * derivative_stepsizes[3], (-2.0) * derivative_stepsizes[3], (-1.0) * derivative_stepsizes[3],
                                       derivative_stepsizes[3], 2.0 * derivative_stepsizes[3], 3.0 * derivative_stepsizes[3], 4.0 * derivative_stepsizes[3]};
            vector<double> m03steps = {(-4.0) * derivative_stepsizes[4], (-3.0) * derivative_stepsizes[4], (-2.0) * derivative_stepsizes[4], (-1.0) * derivative_stepsizes[4],
                                       derivative_stepsizes[4], 2.0 * derivative_stepsizes[4], 3.0 * derivative_stepsizes[4], 4.0 * derivative_stepsizes[4]};
            vector<double> mhfsteps = {(-4.0) * derivative_stepsizes[5], (-3.0) * derivative_stepsizes[5], (-2.0) * derivative_stepsizes[5], (-1.0) * derivative_stepsizes[5],
                                       derivative_stepsizes[5], 2.0 * derivative_stepsizes[5], 3.0 * derivative_stepsizes[5], 4.0 * derivative_stepsizes[5]};
            vector<double> A0steps = {(-4.0) * derivative_stepsizes[6], (-3.0) * derivative_stepsizes[6], (-2.0) * derivative_stepsizes[6], (-1.0) * derivative_stepsizes[6],
                                      derivative_stepsizes[6], 2.0 * derivative_stepsizes[6], 3.0 * derivative_stepsizes[6], 4.0 * derivative_stepsizes[6]};
            vector<double> mu0steps = {(-4.0) * derivative_stepsizes[7], (-3.0) * derivative_stepsizes[7], (-2.0) * derivative_stepsizes[7], (-1.0) * derivative_stepsizes[7],
                                       derivative_stepsizes[7], 2.0 * derivative_stepsizes[7], 3.0 * derivative_stepsizes[7], 4.0 * derivative_stepsizes[7]};
            
            for (int j = 0; j < m03steps.size(); ++j) {
                double step_Resultm01 = deriv_step_calc_genScalarIndices(m01steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, {27, 30, 33, 36, 39});
                mZ_m01_var_vals.push_back(step_Resultm01);
                double step_Resultm02 = deriv_step_calc_genScalarIndices(m02steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, {28, 31, 34, 37, 40});
                mZ_m02_var_vals.push_back(step_Resultm02);
                double step_Resultm03 = deriv_step_calc_genScalarIndices(m03steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, {29, 32, 35, 38, 41});
                mZ_m03_var_vals.push_back(step_Resultm03);
                double step_Resultmhf = deriv_step_calc_gaugino(mhfsteps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_mhf_var_vals.push_back(step_Resultmhf);
                double step_ResultA0 = deriv_step_calc_trilin(A0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_A0_var_vals.push_back(step_ResultA0);
                double step_Resultmu0 = deriv_step_calc_mu0(mu0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_mu0_var_vals.push_back(step_Resultmu0);
                double step_ResultmHu0 = gen_deriv_step_scalar_calc(mHu0steps[j], GUT_boundary_conditions, 25, GUT_SCALE, myweakscale);
                mZ_mHu_var_vals.push_back(step_ResultmHu0);
                double step_ResultmHd0 = gen_deriv_step_scalar_calc(mHd0steps[j], GUT_boundary_conditions, 26, GUT_SCALE, myweakscale);
                mZ_mHd_var_vals.push_back(step_ResultmHd0);
            }            

            dbglist.push_back({(maxm01Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[2], mZ_m01_var_vals), "Delta_BG(m_0(1))"});
            dbglist.push_back({(maxm02Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[3], mZ_m02_var_vals), "Delta_BG(m_0(2))"});
            dbglist.push_back({(maxm03Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[4], mZ_m03_var_vals), "Delta_BG(m_0(3))"});
            dbglist.push_back({(maxGauginoMass / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[5], mZ_mhf_var_vals), "Delta_BG(m_1/2)"});
            dbglist.push_back({(maxTrilin / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[6], mZ_A0_var_vals), "Delta_BG(A_0)"});
            dbglist.push_back({(mu0value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[7], mZ_mu0_var_vals), "Delta_BG(mu_0)"});
            dbglist.push_back({(mHu0Value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[0], mZ_mHu_var_vals), "Delta_BG(mHu)"});
            dbglist.push_back({(mHd0Value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[1], mZ_mHd_var_vals), "Delta_BG(mHd)"});
        }
        //////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////
        else if (precselno == 2) {
            vector<double> mHu0steps = {(-2.0) * derivative_stepsizes[0], (-1.0) * derivative_stepsizes[0],
                                        derivative_stepsizes[0], 2.0 * derivative_stepsizes[0]};
            vector<double> mHd0steps = {(-2.0) * derivative_stepsizes[1], (-1.0) * derivative_stepsizes[1],
                                         derivative_stepsizes[1], 2.0 * derivative_stepsizes[1]};
            vector<double> m01steps = {(-2.0) * derivative_stepsizes[2], (-1.0) * derivative_stepsizes[2],
                                        derivative_stepsizes[2], 2.0 * derivative_stepsizes[2]};
            vector<double> m02steps = {(-2.0) * derivative_stepsizes[3], (-1.0) * derivative_stepsizes[3],
                                       derivative_stepsizes[3], 2.0 * derivative_stepsizes[3]};
            vector<double> m03steps = {(-2.0) * derivative_stepsizes[4], (-1.0) * derivative_stepsizes[4],
                                       derivative_stepsizes[4], 2.0 * derivative_stepsizes[4]};
            vector<double> mhfsteps = {(-2.0) * derivative_stepsizes[5], (-1.0) * derivative_stepsizes[5],
                                       derivative_stepsizes[5], 2.0 * derivative_stepsizes[5]};
            vector<double> A0steps = {(-2.0) * derivative_stepsizes[6], (-1.0) * derivative_stepsizes[6],
                                      derivative_stepsizes[6], 2.0 * derivative_stepsizes[6]};
            vector<double> mu0steps = {(-2.0) * derivative_stepsizes[7], (-1.0) * derivative_stepsizes[7],
                                       derivative_stepsizes[7], 2.0 * derivative_stepsizes[7]};
            
            for (int j = 0; j < m03steps.size(); ++j) {
                double step_Resultm01 = deriv_step_calc_genScalarIndices(m01steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, {27, 30, 33, 36, 39});
                mZ_m01_var_vals.push_back(step_Resultm01);
                double step_Resultm02 = deriv_step_calc_genScalarIndices(m02steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, {28, 31, 34, 37, 40});
                mZ_m02_var_vals.push_back(step_Resultm02);
                double step_Resultm03 = deriv_step_calc_genScalarIndices(m03steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, {29, 32, 35, 38, 41});
                mZ_m03_var_vals.push_back(step_Resultm03);
                double step_Resultmhf = deriv_step_calc_gaugino(mhfsteps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_mhf_var_vals.push_back(step_Resultmhf);
                double step_ResultA0 = deriv_step_calc_trilin(A0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_A0_var_vals.push_back(step_ResultA0);
                double step_Resultmu0 = deriv_step_calc_mu0(mu0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_mu0_var_vals.push_back(step_Resultmu0);
                double step_ResultmHu0 = gen_deriv_step_scalar_calc(mHu0steps[j], GUT_boundary_conditions, 25, GUT_SCALE, myweakscale);
                mZ_mHu_var_vals.push_back(step_ResultmHu0);
                double step_ResultmHd0 = gen_deriv_step_scalar_calc(mHd0steps[j], GUT_boundary_conditions, 26, GUT_SCALE, myweakscale);
                mZ_mHd_var_vals.push_back(step_ResultmHd0);
            }            

            dbglist.push_back({(maxm01Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[2], mZ_m01_var_vals), "Delta_BG(m_0(1))"});
            dbglist.push_back({(maxm02Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[3], mZ_m02_var_vals), "Delta_BG(m_0(2))"});
            dbglist.push_back({(maxm03Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[4], mZ_m03_var_vals), "Delta_BG(m_0(3))"});
            dbglist.push_back({(maxGauginoMass / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[5], mZ_mhf_var_vals), "Delta_BG(m_1/2)"});
            dbglist.push_back({(maxTrilin / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[6], mZ_A0_var_vals), "Delta_BG(A_0)"});
            dbglist.push_back({(mu0value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[7], mZ_mu0_var_vals), "Delta_BG(mu_0)"});
            dbglist.push_back({(mHu0Value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[0], mZ_mHu_var_vals), "Delta_BG(mHu)"});
            dbglist.push_back({(mHd0Value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[1], mZ_mHd_var_vals), "Delta_BG(mHd)"});
        }
        //////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////
        else {
            vector<double> mHu0steps = {(-1.0) * derivative_stepsizes[0],
                                        derivative_stepsizes[0]};
            vector<double> mHd0steps = {(-1.0) * derivative_stepsizes[1],
                                         derivative_stepsizes[1]};
            vector<double> m01steps = {(-1.0) * derivative_stepsizes[2],
                                        derivative_stepsizes[2]};
            vector<double> m02steps = {(-1.0) * derivative_stepsizes[3],
                                       derivative_stepsizes[3]};
            vector<double> m03steps = {(-1.0) * derivative_stepsizes[4],
                                       derivative_stepsizes[4]};
            vector<double> mhfsteps = {(-1.0) * derivative_stepsizes[5],
                                       derivative_stepsizes[5]};
            vector<double> A0steps = {(-1.0) * derivative_stepsizes[6],
                                      derivative_stepsizes[6]};
            vector<double> mu0steps = {(-1.0) * derivative_stepsizes[7],
                                       derivative_stepsizes[7]};
            
            for (int j = 0; j < m03steps.size(); ++j) {
                double step_Resultm01 = deriv_step_calc_genScalarIndices(m01steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, {27, 30, 33, 36, 39});
                mZ_m01_var_vals.push_back(step_Resultm01);
                double step_Resultm02 = deriv_step_calc_genScalarIndices(m02steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, {28, 31, 34, 37, 40});
                mZ_m02_var_vals.push_back(step_Resultm02);
                double step_Resultm03 = deriv_step_calc_genScalarIndices(m03steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, {29, 32, 35, 38, 41});
                mZ_m03_var_vals.push_back(step_Resultm03);
                double step_Resultmhf = deriv_step_calc_gaugino(mhfsteps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_mhf_var_vals.push_back(step_Resultmhf);
                double step_ResultA0 = deriv_step_calc_trilin(A0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_A0_var_vals.push_back(step_ResultA0);
                double step_Resultmu0 = deriv_step_calc_mu0(mu0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_mu0_var_vals.push_back(step_Resultmu0);
                double step_ResultmHu0 = gen_deriv_step_scalar_calc(mHu0steps[j], GUT_boundary_conditions, 25, GUT_SCALE, myweakscale);
                mZ_mHu_var_vals.push_back(step_ResultmHu0);
                double step_ResultmHd0 = gen_deriv_step_scalar_calc(mHd0steps[j], GUT_boundary_conditions, 26, GUT_SCALE, myweakscale);
                mZ_mHd_var_vals.push_back(step_ResultmHd0);
            }            

            dbglist.push_back({(maxm01Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[2], mZ_m01_var_vals), "Delta_BG(m_0(1))"});
            dbglist.push_back({(maxm02Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[3], mZ_m02_var_vals), "Delta_BG(m_0(2))"});
            dbglist.push_back({(maxm03Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[4], mZ_m03_var_vals), "Delta_BG(m_0(3))"});
            dbglist.push_back({(maxGauginoMass / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[5], mZ_mhf_var_vals), "Delta_BG(m_1/2)"});
            dbglist.push_back({(maxTrilin / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[6], mZ_A0_var_vals), "Delta_BG(A_0)"});
            dbglist.push_back({(mu0value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[7], mZ_mu0_var_vals), "Delta_BG(mu_0)"});
            dbglist.push_back({(mHu0Value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[0], mZ_mHu_var_vals), "Delta_BG(mHu)"});
            dbglist.push_back({(mHd0Value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[1], mZ_mHd_var_vals), "Delta_BG(mHd)"});
        }
    }
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    else {
        vector<double> mZ_mqL12_var_vals, mZ_mqL3_var_vals, mZ_muR12_var_vals, mZ_muR3_var_vals;
        vector<double> mZ_mdR12_var_vals, mZ_mdR3_var_vals, mZ_meL12_var_vals, mZ_meL3_var_vals;
        vector<double> mZ_meR12_var_vals, mZ_meR3_var_vals, mZ_M1_var_vals, mZ_M2_var_vals, mZ_M3_var_vals, mZ_mu0_var_vals;
        vector<double> mZ_Au0_var_vals, mZ_Ad0_var_vals, mZ_Ae0_var_vals;
        vector<double> mZ_mHu_var_vals, mZ_mHd_var_vals;
        double mHu0Value = copysign(sqrt(abs(GUT_boundary_conditions[25])), GUT_boundary_conditions[25]);
        double mHd0Value = copysign(sqrt(abs(GUT_boundary_conditions[26])), GUT_boundary_conditions[26]);
        double maxmqL12Val = sqrt(abs(max(GUT_boundary_conditions[27], GUT_boundary_conditions[28])));
        double mqL3Val = sqrt(abs(GUT_boundary_conditions[29]));
        double maxmuR12Val = sqrt(abs(max(GUT_boundary_conditions[33], GUT_boundary_conditions[34])));
        double muR3Val = sqrt(abs(GUT_boundary_conditions[35]));
        double maxmdR12Val = sqrt(abs(max(GUT_boundary_conditions[36], GUT_boundary_conditions[37])));
        double mdR3Val = sqrt(abs(GUT_boundary_conditions[38]));
        double maxmeL12Val = sqrt(abs(max(GUT_boundary_conditions[30], GUT_boundary_conditions[31])));
        double meL3Val = sqrt(abs(GUT_boundary_conditions[32]));
        double maxmeR12Val = sqrt(abs(max(GUT_boundary_conditions[39], GUT_boundary_conditions[40])));
        double meR3Val = sqrt(abs(GUT_boundary_conditions[41]));
        double M1GUTVal = GUT_boundary_conditions[3];
        double M2GUTVal = GUT_boundary_conditions[4];
        double M3GUTVal = GUT_boundary_conditions[5];
        vector<double> Au0candidates = {GUT_boundary_conditions[16] / GUT_boundary_conditions[7],
                                        GUT_boundary_conditions[17] / GUT_boundary_conditions[8],
                                        GUT_boundary_conditions[18] / GUT_boundary_conditions[9]};
        auto maxAu0It = max_element(Au0candidates.begin(), Au0candidates.end(),
                                    [](double a, double b) { return abs(a) < abs(b); });
        double maxUpTrilin = *maxAu0It;
        vector<double> Ad0candidates = {GUT_boundary_conditions[19] / GUT_boundary_conditions[10],
                                        GUT_boundary_conditions[20] / GUT_boundary_conditions[11],
                                        GUT_boundary_conditions[21] / GUT_boundary_conditions[12]};
        auto maxAd0It = max_element(Ad0candidates.begin(), Ad0candidates.end(),
                                    [](double a, double b) { return abs(a) < abs(b); });
        double maxDownTrilin = *maxAd0It;
        vector<double> Ae0candidates = {GUT_boundary_conditions[22] / GUT_boundary_conditions[13],
                                        GUT_boundary_conditions[23] / GUT_boundary_conditions[14],
                                        GUT_boundary_conditions[24] / GUT_boundary_conditions[15]};
        auto maxAe0It = max_element(Ae0candidates.begin(), Ae0candidates.end(),
                                    [](double a, double b) { return abs(a) < abs(b); });
        double maxLeptTrilin = *maxAe0It;
        double mu0value = GUT_boundary_conditions[6];
        if (precselno == 1) {
            vector<double> mHu0steps = {(-4.0) * derivative_stepsizes[0], (-3.0) * derivative_stepsizes[0], (-2.0) * derivative_stepsizes[0], (-1.0) * derivative_stepsizes[0],
                                        derivative_stepsizes[0], 2.0 * derivative_stepsizes[0], 3.0 * derivative_stepsizes[0], 4.0 * derivative_stepsizes[0]};
            vector<double> mHd0steps = {(-4.0) * derivative_stepsizes[1], (-3.0) * derivative_stepsizes[1], (-2.0) * derivative_stepsizes[1], (-1.0) * derivative_stepsizes[1],
                                         derivative_stepsizes[1], 2.0 * derivative_stepsizes[1], 3.0 * derivative_stepsizes[1], 4.0 * derivative_stepsizes[1]};
            vector<double> mqL12steps = {(-4.0) * derivative_stepsizes[2], (-3.0) * derivative_stepsizes[2], (-2.0) * derivative_stepsizes[2], (-1.0) * derivative_stepsizes[2],
                                         derivative_stepsizes[2], 2.0 * derivative_stepsizes[2], 3.0 * derivative_stepsizes[2], 4.0 * derivative_stepsizes[2]};
            vector<double> mqL3steps = {(-4.0) * derivative_stepsizes[3], (-3.0) * derivative_stepsizes[3], (-2.0) * derivative_stepsizes[3], (-1.0) * derivative_stepsizes[3],
                                        derivative_stepsizes[3], 2.0 * derivative_stepsizes[3], 3.0 * derivative_stepsizes[3], 4.0 * derivative_stepsizes[3]};
            vector<double> muR12steps = {(-4.0) * derivative_stepsizes[4], (-3.0) * derivative_stepsizes[4], (-2.0) * derivative_stepsizes[4], (-1.0) * derivative_stepsizes[4],
                                         derivative_stepsizes[4], 2.0 * derivative_stepsizes[4], 3.0 * derivative_stepsizes[4], 4.0 * derivative_stepsizes[4]};
            vector<double> muR3steps = {(-4.0) * derivative_stepsizes[5], (-3.0) * derivative_stepsizes[5], (-2.0) * derivative_stepsizes[5], (-1.0) * derivative_stepsizes[5],
                                        derivative_stepsizes[5], 2.0 * derivative_stepsizes[5], 3.0 * derivative_stepsizes[5], 4.0 * derivative_stepsizes[5]};
            vector<double> mdR12steps = {(-4.0) * derivative_stepsizes[6], (-3.0) * derivative_stepsizes[6], (-2.0) * derivative_stepsizes[6], (-1.0) * derivative_stepsizes[6],
                                         derivative_stepsizes[6], 2.0 * derivative_stepsizes[6], 3.0 * derivative_stepsizes[6], 4.0 * derivative_stepsizes[6]};
            vector<double> mdR3steps = {(-4.0) * derivative_stepsizes[7], (-3.0) * derivative_stepsizes[7], (-2.0) * derivative_stepsizes[7], (-1.0) * derivative_stepsizes[7],
                                        derivative_stepsizes[7], 2.0 * derivative_stepsizes[7], 3.0 * derivative_stepsizes[7], 4.0 * derivative_stepsizes[7]};
            vector<double> meL12steps = {(-4.0) * derivative_stepsizes[8], (-3.0) * derivative_stepsizes[8], (-2.0) * derivative_stepsizes[8], (-1.0) * derivative_stepsizes[8],
                                         derivative_stepsizes[8], 2.0 * derivative_stepsizes[8], 3.0 * derivative_stepsizes[8], 4.0 * derivative_stepsizes[8]};
            vector<double> meL3steps = {(-4.0) * derivative_stepsizes[9], (-3.0) * derivative_stepsizes[9], (-2.0) * derivative_stepsizes[9], (-1.0) * derivative_stepsizes[9],
                                        derivative_stepsizes[9], 2.0 * derivative_stepsizes[9], 3.0 * derivative_stepsizes[9], 4.0 * derivative_stepsizes[9]};
            vector<double> meR12steps = {(-4.0) * derivative_stepsizes[10], (-3.0) * derivative_stepsizes[10], (-2.0) * derivative_stepsizes[10], (-1.0) * derivative_stepsizes[10],
                                         derivative_stepsizes[10], 2.0 * derivative_stepsizes[10], 3.0 * derivative_stepsizes[10], 4.0 * derivative_stepsizes[10]};
            vector<double> meR3steps = {(-4.0) * derivative_stepsizes[11], (-3.0) * derivative_stepsizes[11], (-2.0) * derivative_stepsizes[11], (-1.0) * derivative_stepsizes[11],
                                        derivative_stepsizes[11], 2.0 * derivative_stepsizes[11], 3.0 * derivative_stepsizes[11], 4.0 * derivative_stepsizes[11]};
            vector<double> M1steps = {(-4.0) * derivative_stepsizes[12], (-3.0) * derivative_stepsizes[12], (-2.0) * derivative_stepsizes[12], (-1.0) * derivative_stepsizes[12],
                                      derivative_stepsizes[12], 2.0 * derivative_stepsizes[12], 3.0 * derivative_stepsizes[12], 4.0 * derivative_stepsizes[12]};
            vector<double> M2steps = {(-4.0) * derivative_stepsizes[13], (-3.0) * derivative_stepsizes[13], (-2.0) * derivative_stepsizes[13], (-1.0) * derivative_stepsizes[13],
                                      derivative_stepsizes[13], 2.0 * derivative_stepsizes[13], 3.0 * derivative_stepsizes[13], 4.0 * derivative_stepsizes[13]};
            vector<double> M3steps = {(-4.0) * derivative_stepsizes[14], (-3.0) * derivative_stepsizes[14], (-2.0) * derivative_stepsizes[14], (-1.0) * derivative_stepsizes[14],
                                      derivative_stepsizes[14], 2.0 * derivative_stepsizes[14], 3.0 * derivative_stepsizes[14], 4.0 * derivative_stepsizes[14]};
            vector<double> Au0steps = {(-4.0) * derivative_stepsizes[15], (-3.0) * derivative_stepsizes[15], (-2.0) * derivative_stepsizes[15], (-1.0) * derivative_stepsizes[15],
                                       derivative_stepsizes[15], 2.0 * derivative_stepsizes[15], 3.0 * derivative_stepsizes[15], 4.0 * derivative_stepsizes[15]};
            vector<double> Ad0steps = {(-4.0) * derivative_stepsizes[16], (-3.0) * derivative_stepsizes[16], (-2.0) * derivative_stepsizes[16], (-1.0) * derivative_stepsizes[16],
                                       derivative_stepsizes[16], 2.0 * derivative_stepsizes[16], 3.0 * derivative_stepsizes[16], 4.0 * derivative_stepsizes[16]};
            vector<double> Ae0steps = {(-4.0) * derivative_stepsizes[17], (-3.0) * derivative_stepsizes[17], (-2.0) * derivative_stepsizes[17], (-1.0) * derivative_stepsizes[17],
                                       derivative_stepsizes[17], 2.0 * derivative_stepsizes[17], 3.0 * derivative_stepsizes[17], 4.0 * derivative_stepsizes[17]};
            vector<double> mu0steps = {(-4.0) * derivative_stepsizes[18], (-3.0) * derivative_stepsizes[18], (-2.0) * derivative_stepsizes[18], (-1.0) * derivative_stepsizes[18],
                                       derivative_stepsizes[18], 2.0 * derivative_stepsizes[18], 3.0 * derivative_stepsizes[18], 4.0 * derivative_stepsizes[18]};
            
            for (int j = 0; j < mHu0steps.size(); ++j) {
                double step_ResultmqL12 = deriv_step_calc_genScalarIndices(mqL12steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, {27, 28});
                mZ_mqL12_var_vals.push_back(step_ResultmqL12);
                double step_ResultmuR12 = deriv_step_calc_genScalarIndices(muR12steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, {33, 34});
                mZ_muR12_var_vals.push_back(step_ResultmuR12);
                double step_ResultmdR12 = deriv_step_calc_genScalarIndices(mdR12steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, {36, 37});
                mZ_mdR12_var_vals.push_back(step_ResultmdR12);
                double step_ResultmeR12 = deriv_step_calc_genScalarIndices(meR12steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, {39, 40});
                mZ_meR12_var_vals.push_back(step_ResultmeR12);
                double step_ResultmqL3 = gen_deriv_step_scalar_calc(mqL3steps[j], GUT_boundary_conditions, 29, GUT_SCALE, myweakscale);
                mZ_mqL3_var_vals.push_back(step_ResultmqL3);
                double step_ResultmeL3 = gen_deriv_step_scalar_calc(meL3steps[j], GUT_boundary_conditions, 32, GUT_SCALE, myweakscale);
                mZ_meL3_var_vals.push_back(step_ResultmeL3);
                double step_ResultmuR3 = gen_deriv_step_scalar_calc(muR3steps[j], GUT_boundary_conditions, 35, GUT_SCALE, myweakscale);
                mZ_muR3_var_vals.push_back(step_ResultmuR3);
                double step_ResultmdR3 = gen_deriv_step_scalar_calc(mdR3steps[j], GUT_boundary_conditions, 38, GUT_SCALE, myweakscale);
                mZ_mdR3_var_vals.push_back(step_ResultmdR3);
                double step_ResultmeR3 = gen_deriv_step_scalar_calc(meR3steps[j], GUT_boundary_conditions, 41, GUT_SCALE, myweakscale);
                mZ_meR3_var_vals.push_back(step_ResultmeR3);
                double step_ResultM1 = gen_deriv_step_calc(M1steps[j], GUT_boundary_conditions, 3, GUT_SCALE, myweakscale);
                mZ_M1_var_vals.push_back(step_ResultM1);
                double step_ResultM2 = gen_deriv_step_calc(M2steps[j], GUT_boundary_conditions, 4, GUT_SCALE, myweakscale);
                mZ_M2_var_vals.push_back(step_ResultM2);
                double step_ResultM3 = gen_deriv_step_calc(M3steps[j], GUT_boundary_conditions, 5, GUT_SCALE, myweakscale);
                mZ_M3_var_vals.push_back(step_ResultM3);
                double step_ResultAu0 = deriv_step_calc_gentrilinRange(Au0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, 16, 19);
                mZ_Au0_var_vals.push_back(step_ResultAu0);
                double step_ResultAd0 = deriv_step_calc_gentrilinRange(Ad0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, 19, 22);
                mZ_Ad0_var_vals.push_back(step_ResultAd0);
                double step_ResultAe0 = deriv_step_calc_gentrilinRange(Ae0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, 22, 25);
                mZ_Ae0_var_vals.push_back(step_ResultAe0);
                double step_Resultmu0 = deriv_step_calc_mu0(mu0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_mu0_var_vals.push_back(step_Resultmu0);
                double step_ResultmHu0 = gen_deriv_step_scalar_calc(mHu0steps[j], GUT_boundary_conditions, 25, GUT_SCALE, myweakscale);
                mZ_mHu_var_vals.push_back(step_ResultmHu0);
                double step_ResultmHd0 = gen_deriv_step_scalar_calc(mHd0steps[j], GUT_boundary_conditions, 26, GUT_SCALE, myweakscale);
                mZ_mHd_var_vals.push_back(step_ResultmHd0);
            }            

            dbglist.push_back({(maxmqL12Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[2], mZ_mqL12_var_vals), "Delta_BG(m_qL(1,2))"});
            dbglist.push_back({(mqL3Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[3], mZ_mqL3_var_vals), "Delta_BG(m_qL(3))"});
            dbglist.push_back({(maxmuR12Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[4], mZ_muR12_var_vals), "Delta_BG(m_uR(1,2))"});
            dbglist.push_back({(muR3Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[5], mZ_muR3_var_vals), "Delta_BG(m_uR(3))"});
            dbglist.push_back({(maxmdR12Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[6], mZ_mdR12_var_vals), "Delta_BG(m_dR(1,2))"});
            dbglist.push_back({(mdR3Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[7], mZ_mdR3_var_vals), "Delta_BG(m_dR(3))"});
            dbglist.push_back({(maxmeL12Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[8], mZ_meL12_var_vals), "Delta_BG(m_eL(1,2))"});
            dbglist.push_back({(meL3Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[9], mZ_meL3_var_vals), "Delta_BG(m_eL(3))"});
            dbglist.push_back({(maxmeR12Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[10], mZ_meR12_var_vals), "Delta_BG(m_eR(1,2))"});
            dbglist.push_back({(meR3Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[11], mZ_meR3_var_vals), "Delta_BG(m_eR(3))"});
            dbglist.push_back({(M1GUTVal / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[12], mZ_M1_var_vals), "Delta_BG(M_1)"});
            dbglist.push_back({(M2GUTVal / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[13], mZ_M2_var_vals), "Delta_BG(M_2)"});
            dbglist.push_back({(M3GUTVal / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[14], mZ_M3_var_vals), "Delta_BG(M_3)"});
            dbglist.push_back({(maxUpTrilin / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[15], mZ_Au0_var_vals), "Delta_BG(A_t,A_c,A_u)"});
            dbglist.push_back({(maxDownTrilin / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[16], mZ_Ad0_var_vals), "Delta_BG(A_b,A_s,A_d)"});
            dbglist.push_back({(maxLeptTrilin / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[17], mZ_Ae0_var_vals), "Delta_BG(A_tau,A_mu,A_e)"});
            dbglist.push_back({(mu0value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[18], mZ_mu0_var_vals), "Delta_BG(mu_0)"});
            dbglist.push_back({(mHu0Value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[0], mZ_mHu_var_vals), "Delta_BG(mHu)"});
            dbglist.push_back({(mHd0Value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[1], mZ_mHd_var_vals), "Delta_BG(mHd)"});
        }
        //////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////
        else if (precselno == 2) {
            vector<double> mHu0steps = {(-2.0) * derivative_stepsizes[0], (-1.0) * derivative_stepsizes[0],
                                        derivative_stepsizes[0], 2.0 * derivative_stepsizes[0]};
            vector<double> mHd0steps = {(-2.0) * derivative_stepsizes[1], (-1.0) * derivative_stepsizes[1],
                                         derivative_stepsizes[1], 2.0 * derivative_stepsizes[1]};
            vector<double> mqL12steps = {(-2.0) * derivative_stepsizes[2], (-1.0) * derivative_stepsizes[2],
                                         derivative_stepsizes[2], 2.0 * derivative_stepsizes[2]};
            vector<double> mqL3steps = {(-2.0) * derivative_stepsizes[3], (-1.0) * derivative_stepsizes[3],
                                        derivative_stepsizes[3], 2.0 * derivative_stepsizes[3]};
            vector<double> muR12steps = {(-2.0) * derivative_stepsizes[4], (-1.0) * derivative_stepsizes[4],
                                         derivative_stepsizes[4], 2.0 * derivative_stepsizes[4]};
            vector<double> muR3steps = {(-2.0) * derivative_stepsizes[5], (-1.0) * derivative_stepsizes[5],
                                        derivative_stepsizes[5], 2.0 * derivative_stepsizes[5]};
            vector<double> mdR12steps = {(-2.0) * derivative_stepsizes[6], (-1.0) * derivative_stepsizes[6],
                                         derivative_stepsizes[6], 2.0 * derivative_stepsizes[6]};
            vector<double> mdR3steps = {(-2.0) * derivative_stepsizes[7], (-1.0) * derivative_stepsizes[7],
                                        derivative_stepsizes[7], 2.0 * derivative_stepsizes[7]};
            vector<double> meL12steps = {(-2.0) * derivative_stepsizes[8], (-1.0) * derivative_stepsizes[8],
                                         derivative_stepsizes[8], 2.0 * derivative_stepsizes[8]};
            vector<double> meL3steps = {(-2.0) * derivative_stepsizes[9], (-1.0) * derivative_stepsizes[9],
                                        derivative_stepsizes[9], 2.0 * derivative_stepsizes[9]};
            vector<double> meR12steps = {(-2.0) * derivative_stepsizes[10], (-1.0) * derivative_stepsizes[10],
                                         derivative_stepsizes[10], 2.0 * derivative_stepsizes[10]};
            vector<double> meR3steps = {(-2.0) * derivative_stepsizes[11], (-1.0) * derivative_stepsizes[11],
                                        derivative_stepsizes[11], 2.0 * derivative_stepsizes[11]};
            vector<double> M1steps = {(-2.0) * derivative_stepsizes[12], (-1.0) * derivative_stepsizes[12],
                                      derivative_stepsizes[12], 2.0 * derivative_stepsizes[12]};
            vector<double> M2steps = {(-2.0) * derivative_stepsizes[13], (-1.0) * derivative_stepsizes[13],
                                      derivative_stepsizes[13], 2.0 * derivative_stepsizes[13]};
            vector<double> M3steps = {(-2.0) * derivative_stepsizes[14], (-1.0) * derivative_stepsizes[14],
                                      derivative_stepsizes[14], 2.0 * derivative_stepsizes[14]};
            vector<double> Au0steps = {(-2.0) * derivative_stepsizes[15], (-1.0) * derivative_stepsizes[15],
                                       derivative_stepsizes[15], 2.0 * derivative_stepsizes[15]};
            vector<double> Ad0steps = {(-2.0) * derivative_stepsizes[16], (-1.0) * derivative_stepsizes[16],
                                       derivative_stepsizes[16], 2.0 * derivative_stepsizes[16]};
            vector<double> Ae0steps = {(-2.0) * derivative_stepsizes[17], (-1.0) * derivative_stepsizes[17],
                                       derivative_stepsizes[17], 2.0 * derivative_stepsizes[17]};
            vector<double> mu0steps = {(-2.0) * derivative_stepsizes[18], (-1.0) * derivative_stepsizes[18],
                                       derivative_stepsizes[18], 2.0 * derivative_stepsizes[18]};
            
            for (int j = 0; j < mHu0steps.size(); ++j) {
                double step_ResultmqL12 = deriv_step_calc_genScalarIndices(mqL12steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, {27, 28});
                mZ_mqL12_var_vals.push_back(step_ResultmqL12);
                double step_ResultmuR12 = deriv_step_calc_genScalarIndices(muR12steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, {33, 34});
                mZ_muR12_var_vals.push_back(step_ResultmuR12);
                double step_ResultmdR12 = deriv_step_calc_genScalarIndices(mdR12steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, {36, 37});
                mZ_mdR12_var_vals.push_back(step_ResultmdR12);
                double step_ResultmeR12 = deriv_step_calc_genScalarIndices(meR12steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, {39, 40});
                mZ_meR12_var_vals.push_back(step_ResultmeR12);
                double step_ResultmqL3 = gen_deriv_step_scalar_calc(mqL3steps[j], GUT_boundary_conditions, 29, GUT_SCALE, myweakscale);
                mZ_mqL3_var_vals.push_back(step_ResultmqL3);
                double step_ResultmeL3 = gen_deriv_step_scalar_calc(meL3steps[j], GUT_boundary_conditions, 32, GUT_SCALE, myweakscale);
                mZ_meL3_var_vals.push_back(step_ResultmeL3);
                double step_ResultmuR3 = gen_deriv_step_scalar_calc(muR3steps[j], GUT_boundary_conditions, 35, GUT_SCALE, myweakscale);
                mZ_muR3_var_vals.push_back(step_ResultmuR3);
                double step_ResultmdR3 = gen_deriv_step_scalar_calc(mdR3steps[j], GUT_boundary_conditions, 38, GUT_SCALE, myweakscale);
                mZ_mdR3_var_vals.push_back(step_ResultmdR3);
                double step_ResultmeR3 = gen_deriv_step_scalar_calc(meR3steps[j], GUT_boundary_conditions, 41, GUT_SCALE, myweakscale);
                mZ_meR3_var_vals.push_back(step_ResultmeR3);
                double step_ResultM1 = gen_deriv_step_calc(M1steps[j], GUT_boundary_conditions, 3, GUT_SCALE, myweakscale);
                mZ_M1_var_vals.push_back(step_ResultM1);
                double step_ResultM2 = gen_deriv_step_calc(M2steps[j], GUT_boundary_conditions, 4, GUT_SCALE, myweakscale);
                mZ_M2_var_vals.push_back(step_ResultM2);
                double step_ResultM3 = gen_deriv_step_calc(M3steps[j], GUT_boundary_conditions, 5, GUT_SCALE, myweakscale);
                mZ_M3_var_vals.push_back(step_ResultM3);
                double step_ResultAu0 = deriv_step_calc_gentrilinRange(Au0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, 16, 19);
                mZ_Au0_var_vals.push_back(step_ResultAu0);
                double step_ResultAd0 = deriv_step_calc_gentrilinRange(Ad0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, 19, 22);
                mZ_Ad0_var_vals.push_back(step_ResultAd0);
                double step_ResultAe0 = deriv_step_calc_gentrilinRange(Ae0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, 22, 25);
                mZ_Ae0_var_vals.push_back(step_ResultAe0);
                double step_Resultmu0 = deriv_step_calc_mu0(mu0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_mu0_var_vals.push_back(step_Resultmu0);
                double step_ResultmHu0 = gen_deriv_step_scalar_calc(mHu0steps[j], GUT_boundary_conditions, 25, GUT_SCALE, myweakscale);
                mZ_mHu_var_vals.push_back(step_ResultmHu0);
                double step_ResultmHd0 = gen_deriv_step_scalar_calc(mHd0steps[j], GUT_boundary_conditions, 26, GUT_SCALE, myweakscale);
                mZ_mHd_var_vals.push_back(step_ResultmHd0);
            }            

            dbglist.push_back({(maxmqL12Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[2], mZ_mqL12_var_vals), "Delta_BG(m_qL(1,2))"});
            dbglist.push_back({(mqL3Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[3], mZ_mqL3_var_vals), "Delta_BG(m_qL(3))"});
            dbglist.push_back({(maxmuR12Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[4], mZ_muR12_var_vals), "Delta_BG(m_uR(1,2))"});
            dbglist.push_back({(muR3Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[5], mZ_muR3_var_vals), "Delta_BG(m_uR(3))"});
            dbglist.push_back({(maxmdR12Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[6], mZ_mdR12_var_vals), "Delta_BG(m_dR(1,2))"});
            dbglist.push_back({(mdR3Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[7], mZ_mdR3_var_vals), "Delta_BG(m_dR(3))"});
            dbglist.push_back({(maxmeL12Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[8], mZ_meL12_var_vals), "Delta_BG(m_eL(1,2))"});
            dbglist.push_back({(meL3Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[9], mZ_meL3_var_vals), "Delta_BG(m_eL(3))"});
            dbglist.push_back({(maxmeR12Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[10], mZ_meR12_var_vals), "Delta_BG(m_eR(1,2))"});
            dbglist.push_back({(meR3Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[11], mZ_meR3_var_vals), "Delta_BG(m_eR(3))"});
            dbglist.push_back({(M1GUTVal / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[12], mZ_M1_var_vals), "Delta_BG(M_1)"});
            dbglist.push_back({(M2GUTVal / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[13], mZ_M2_var_vals), "Delta_BG(M_2)"});
            dbglist.push_back({(M3GUTVal / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[14], mZ_M3_var_vals), "Delta_BG(M_3)"});
            dbglist.push_back({(maxUpTrilin / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[15], mZ_Au0_var_vals), "Delta_BG(A_t,A_c,A_u)"});
            dbglist.push_back({(maxDownTrilin / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[16], mZ_Ad0_var_vals), "Delta_BG(A_b,A_s,A_d)"});
            dbglist.push_back({(maxLeptTrilin / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[17], mZ_Ae0_var_vals), "Delta_BG(A_tau,A_mu,A_e)"});
            dbglist.push_back({(mu0value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[18], mZ_mu0_var_vals), "Delta_BG(mu_0)"});
            dbglist.push_back({(mHu0Value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[0], mZ_mHu_var_vals), "Delta_BG(mHu)"});
            dbglist.push_back({(mHd0Value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[1], mZ_mHd_var_vals), "Delta_BG(mHd)"});
        }
        //////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////
        else {
            vector<double> mHu0steps = {(-1.0) * derivative_stepsizes[0],
                                        derivative_stepsizes[0]};
            vector<double> mHd0steps = {(-1.0) * derivative_stepsizes[1],
                                         derivative_stepsizes[1]};
            vector<double> mqL12steps = {(-1.0) * derivative_stepsizes[2],
                                         derivative_stepsizes[2]};
            vector<double> mqL3steps = {(-1.0) * derivative_stepsizes[3],
                                        derivative_stepsizes[3]};
            vector<double> muR12steps = {(-1.0) * derivative_stepsizes[4],
                                         derivative_stepsizes[4]};
            vector<double> muR3steps = {(-1.0) * derivative_stepsizes[5],
                                        derivative_stepsizes[5]};
            vector<double> mdR12steps = {(-1.0) * derivative_stepsizes[6],
                                         derivative_stepsizes[6]};
            vector<double> mdR3steps = {(-1.0) * derivative_stepsizes[7],
                                        derivative_stepsizes[7]};
            vector<double> meL12steps = {(-1.0) * derivative_stepsizes[8],
                                         derivative_stepsizes[8]};
            vector<double> meL3steps = {(-1.0) * derivative_stepsizes[9],
                                        derivative_stepsizes[9]};
            vector<double> meR12steps = {(-1.0) * derivative_stepsizes[10],
                                         derivative_stepsizes[10]};
            vector<double> meR3steps = {(-1.0) * derivative_stepsizes[11],
                                        derivative_stepsizes[11]};
            vector<double> M1steps = {(-1.0) * derivative_stepsizes[12],
                                      derivative_stepsizes[12]};
            vector<double> M2steps = {(-1.0) * derivative_stepsizes[13],
                                      derivative_stepsizes[13]};
            vector<double> M3steps = {(-1.0) * derivative_stepsizes[14],
                                      derivative_stepsizes[14]};
            vector<double> Au0steps = {(-1.0) * derivative_stepsizes[15],
                                       derivative_stepsizes[15]};
            vector<double> Ad0steps = {(-1.0) * derivative_stepsizes[16],
                                       derivative_stepsizes[16]};
            vector<double> Ae0steps = {(-1.0) * derivative_stepsizes[17],
                                       derivative_stepsizes[17]};
            vector<double> mu0steps = {(-1.0) * derivative_stepsizes[18],
                                       derivative_stepsizes[18]};
            
            for (int j = 0; j < mHu0steps.size(); ++j) {
                double step_ResultmqL12 = deriv_step_calc_genScalarIndices(mqL12steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, {27, 28});
                mZ_mqL12_var_vals.push_back(step_ResultmqL12);
                double step_ResultmuR12 = deriv_step_calc_genScalarIndices(muR12steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, {33, 34});
                mZ_muR12_var_vals.push_back(step_ResultmuR12);
                double step_ResultmdR12 = deriv_step_calc_genScalarIndices(mdR12steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, {36, 37});
                mZ_mdR12_var_vals.push_back(step_ResultmdR12);
                double step_ResultmeR12 = deriv_step_calc_genScalarIndices(meR12steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, {39, 40});
                mZ_meR12_var_vals.push_back(step_ResultmeR12);
                double step_ResultmqL3 = gen_deriv_step_scalar_calc(mqL3steps[j], GUT_boundary_conditions, 29, GUT_SCALE, myweakscale);
                mZ_mqL3_var_vals.push_back(step_ResultmqL3);
                double step_ResultmeL3 = gen_deriv_step_scalar_calc(meL3steps[j], GUT_boundary_conditions, 32, GUT_SCALE, myweakscale);
                mZ_meL3_var_vals.push_back(step_ResultmeL3);
                double step_ResultmuR3 = gen_deriv_step_scalar_calc(muR3steps[j], GUT_boundary_conditions, 35, GUT_SCALE, myweakscale);
                mZ_muR3_var_vals.push_back(step_ResultmuR3);
                double step_ResultmdR3 = gen_deriv_step_scalar_calc(mdR3steps[j], GUT_boundary_conditions, 38, GUT_SCALE, myweakscale);
                mZ_mdR3_var_vals.push_back(step_ResultmdR3);
                double step_ResultmeR3 = gen_deriv_step_scalar_calc(meR3steps[j], GUT_boundary_conditions, 41, GUT_SCALE, myweakscale);
                mZ_meR3_var_vals.push_back(step_ResultmeR3);
                double step_ResultM1 = gen_deriv_step_calc(M1steps[j], GUT_boundary_conditions, 3, GUT_SCALE, myweakscale);
                mZ_M1_var_vals.push_back(step_ResultM1);
                double step_ResultM2 = gen_deriv_step_calc(M2steps[j], GUT_boundary_conditions, 4, GUT_SCALE, myweakscale);
                mZ_M2_var_vals.push_back(step_ResultM2);
                double step_ResultM3 = gen_deriv_step_calc(M3steps[j], GUT_boundary_conditions, 5, GUT_SCALE, myweakscale);
                mZ_M3_var_vals.push_back(step_ResultM3);
                double step_ResultAu0 = deriv_step_calc_gentrilinRange(Au0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, 16, 19);
                mZ_Au0_var_vals.push_back(step_ResultAu0);
                double step_ResultAd0 = deriv_step_calc_gentrilinRange(Ad0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, 19, 22);
                mZ_Ad0_var_vals.push_back(step_ResultAd0);
                double step_ResultAe0 = deriv_step_calc_gentrilinRange(Ae0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale, 22, 25);
                mZ_Ae0_var_vals.push_back(step_ResultAe0);
                double step_Resultmu0 = deriv_step_calc_mu0(mu0steps[j], GUT_boundary_conditions, GUT_SCALE, myweakscale);
                mZ_mu0_var_vals.push_back(step_Resultmu0);
                double step_ResultmHu0 = gen_deriv_step_scalar_calc(mHu0steps[j], GUT_boundary_conditions, 25, GUT_SCALE, myweakscale);
                mZ_mHu_var_vals.push_back(step_ResultmHu0);
                double step_ResultmHd0 = gen_deriv_step_scalar_calc(mHd0steps[j], GUT_boundary_conditions, 26, GUT_SCALE, myweakscale);
                mZ_mHd_var_vals.push_back(step_ResultmHd0);
            }            

            dbglist.push_back({(maxmqL12Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[2], mZ_mqL12_var_vals), "Delta_BG(m_qL(1,2))"});
            dbglist.push_back({(mqL3Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[3], mZ_mqL3_var_vals), "Delta_BG(m_qL(3))"});
            dbglist.push_back({(maxmuR12Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[4], mZ_muR12_var_vals), "Delta_BG(m_uR(1,2))"});
            dbglist.push_back({(muR3Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[5], mZ_muR3_var_vals), "Delta_BG(m_uR(3))"});
            dbglist.push_back({(maxmdR12Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[6], mZ_mdR12_var_vals), "Delta_BG(m_dR(1,2))"});
            dbglist.push_back({(mdR3Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[7], mZ_mdR3_var_vals), "Delta_BG(m_dR(3))"});
            dbglist.push_back({(maxmeL12Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[8], mZ_meL12_var_vals), "Delta_BG(m_eL(1,2))"});
            dbglist.push_back({(meL3Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[9], mZ_meL3_var_vals), "Delta_BG(m_eL(3))"});
            dbglist.push_back({(maxmeR12Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[10], mZ_meR12_var_vals), "Delta_BG(m_eR(1,2))"});
            dbglist.push_back({(meR3Val / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[11], mZ_meR3_var_vals), "Delta_BG(m_eR(3))"});
            dbglist.push_back({(M1GUTVal / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[12], mZ_M1_var_vals), "Delta_BG(M_1)"});
            dbglist.push_back({(M2GUTVal / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[13], mZ_M2_var_vals), "Delta_BG(M_2)"});
            dbglist.push_back({(M3GUTVal / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[14], mZ_M3_var_vals), "Delta_BG(M_3)"});
            dbglist.push_back({(maxUpTrilin / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[15], mZ_Au0_var_vals), "Delta_BG(A_t,A_c,A_u)"});
            dbglist.push_back({(maxDownTrilin / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[16], mZ_Ad0_var_vals), "Delta_BG(A_b,A_s,A_d)"});
            dbglist.push_back({(maxLeptTrilin / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[17], mZ_Ae0_var_vals), "Delta_BG(A_tau,A_mu,A_e)"});
            dbglist.push_back({(mu0value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[18], mZ_mu0_var_vals), "Delta_BG(mu_0)"});
            dbglist.push_back({(mHu0Value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[0], mZ_mHu_var_vals), "Delta_BG(mHu)"});
            dbglist.push_back({(mHd0Value / mymZ_squared) * deriv_num_calc(precselno, derivative_stepsizes[1], mZ_mHd_var_vals), "Delta_BG(mHd)"});
        }
    }
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    vector<LabeledValueBG> sortedList = sortAndReturnBG(dbglist);
    return sortedList;
}
