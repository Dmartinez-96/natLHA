// DEW_CALC_HPP

#ifndef DEW_CALC_HPP
#define DEW_CALC_HPP

#include <vector>

//Functions

struct LabeledValue {
    double value;
    std::string label;
};

inline std::vector<LabeledValue> sortAndReturn(const std::vector<LabeledValue>& concatenatedList);

inline bool absValCompare(const LabeledValue& a, const LabeledValue& b);

inline double spence(const double& spenceinp);

inline double logfunc(const double& mass, const double& Q_renorm_sq);

inline double logfunc2(const double& masssq, const double& Q_renorm_sq);

inline double neutralino_denom(const double& msnsq, const double& M1val, const double& M2val,
                               const double& muval, const double& g2sqval, const double& gprsqval,
                               const double& vsqval, const double& vuval, const double& vdval, const double& betaval);

inline double neutralinouu_num(const double& msnq, const double& M1val, const double& M2val,
                               const double& muval, const double& g2sqval, const double& gprsqval,
                               const double& vsqval, const double& vuval, const double& vdval, const double& betaval);

inline double neutralinodd_num(const double& msnq, const double& M1val, const double& M2val,
                               const double& muval, const double& g2sqval, const double& gprsqval,
                               const double& vsqval, const double& vuval, const double& vdval, const double& betaval);

inline double sigmauu_neutralino(const double& msnq, const double& M1val, const double& M2val,
                                 const double& muval, const double& g2sqval, const double& gprsqval,
                                 const double& vsqval, const double& vuval, const double& vdval, const double& betaval,
                                 const double& myQval);

inline double sigmadd_neutralino(const double& msnq, const double& M1val, const double& M2val,
                                 const double& muval, const double& g2sqval, const double& gprsqval,
                                 const double& vsqval, const double& vuval, const double& vdval, const double& betaval,
                                 const double& myQval);

inline double Deltafunc(const double& x, const double& y, const double& z);

inline double Phifunc(const double& x, const double& y, const double& z);

inline double sigmauu_2loop(const double& myQ, const double& mu_wk, const double& beta_wk, const double& yt_wk, const double& yc_wk, const double& yu_wk, const double& yb_wk, const double& ys_wk,
                            const double& yd_wk, const double& ytau_wk, const double& ymu_wk, const double& ye_wk, const double& g1_wk, const double& g2_wk, const double& g3_wk, const double& mQ3_sq_wk,
                            const double& mQ2_sq_wk, const double& mQ1_sq_wk, const double& mL3_sq_wk, const double& mL2_sq_wk, const double& mL1_sq_wk,
                            const double& mU3_sq_wk, const double& mU2_sq_wk, const double& mU1_sq_wk, const double& mD3_sq_wk, const double& mD2_sq_wk, const double& mD1_sq_wk,
                            const double& mE3_sq_wk, const double& mE2_sq_wk, const double& mE1_sq_wk, const double& M1_wk, const double& M2_wk, const double& M3_wk, const double& mHu_sq_wk,
                            const double& mHd_sq_wk, const double& at_wk, const double& ac_wk, const double& au_wk, const double& ab_wk, const double& as_wk, const double& ad_wk, const double& atau_wk,
                            const double& amu_wk, const double& ae_wk, const double& m_stop_1sq, const double& m_stop_2sq, const double& mymt, const double& vHiggs_wk);

inline double sigmadd_2loop(const double& myQ, const double& mu_wk, const double& beta_wk, const double& yt_wk, const double& yc_wk, const double& yu_wk, const double& yb_wk, const double& ys_wk,
                            const double& yd_wk, const double& ytau_wk, const double& ymu_wk, const double& ye_wk, const double& g1_wk, const double& g2_wk, const double& g3_wk, const double& mQ3_sq_wk,
                            const double& mQ2_sq_wk, const double& mQ1_sq_wk, const double& mL3_sq_wk, const double& mL2_sq_wk, const double& mL1_sq_wk,
                            const double& mU3_sq_wk, const double& mU2_sq_wk, const double& mU1_sq_wk, const double& mD3_sq_wk, const double& mD2_sq_wk, const double& mD1_sq_wk,
                            const double& mE3_sq_wk, const double& mE2_sq_wk, const double& mE1_sq_wk, const double& M1_wk, const double& M2_wk, const double& M3_wk, const double& mHu_sq_wk,
                            const double& mHd_sq_wk, const double& at_wk, const double& ac_wk, const double& au_wk, const double& ab_wk, const double& as_wk, const double& ad_wk, const double& atau_wk,
                            const double& amu_wk, const double& ae_wk, const double& m_stop_1sq, const double& m_stop_2sq, const double& mymt, const double& vHiggs_wk);

inline double dew_funcu(const double& inp, const double& beta, const double& Hrad);

inline double dew_funcd(const double& inp, const double& beta, const double& Hrad);


std::vector<LabeledValue> DEW_calc(std::vector<double> weak_boundary_conditions, double myQ);

#endif