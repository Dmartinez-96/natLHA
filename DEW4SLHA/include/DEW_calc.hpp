// DEW_CALC_HPP

#ifndef DEW_CALC_HPP
#define DEW_CALC_HPP

#include <vector>
#include <string>
#include <boost/multiprecision/mpfr.hpp>
using namespace boost::multiprecision;
typedef number<mpfr_float_backend<50>> high_prec_float;  // 50 decimal digits of precision


//Functions

struct LabeledValue {
    high_prec_float value;
    std::string label;
};

inline std::vector<LabeledValue> sortAndReturn(const std::vector<LabeledValue>& concatenatedList);

inline bool absValCompare(const LabeledValue& a, const LabeledValue& b);

inline high_prec_float spence(const high_prec_float& spenceinp);

inline high_prec_float logfunc(const high_prec_float& mass, const high_prec_float& Q_renorm_sq);

inline high_prec_float logfunc2(const high_prec_float& masssq, const high_prec_float& Q_renorm_sq);

inline high_prec_float neutralino_denom(const high_prec_float& msnsq, const high_prec_float& M1val, const high_prec_float& M2val,
                               const high_prec_float& muval, const high_prec_float& g2sqval, const high_prec_float& gprsqval,
                               const high_prec_float& vsqval, const high_prec_float& vuval, const high_prec_float& vdval, const high_prec_float& betaval);

inline high_prec_float neutralinouu_num(const high_prec_float& msnq, const high_prec_float& M1val, const high_prec_float& M2val,
                               const high_prec_float& muval, const high_prec_float& g2sqval, const high_prec_float& gprsqval,
                               const high_prec_float& vsqval, const high_prec_float& vuval, const high_prec_float& vdval, const high_prec_float& betaval);

inline high_prec_float neutralinodd_num(const high_prec_float& msnq, const high_prec_float& M1val, const high_prec_float& M2val,
                               const high_prec_float& muval, const high_prec_float& g2sqval, const high_prec_float& gprsqval,
                               const high_prec_float& vsqval, const high_prec_float& vuval, const high_prec_float& vdval, const high_prec_float& betaval);

inline high_prec_float sigmauu_neutralino(const high_prec_float& msnq, const high_prec_float& M1val, const high_prec_float& M2val,
                                 const high_prec_float& muval, const high_prec_float& g2sqval, const high_prec_float& gprsqval,
                                 const high_prec_float& vsqval, const high_prec_float& vuval, const high_prec_float& vdval, const high_prec_float& betaval,
                                 const high_prec_float& myQval);

inline high_prec_float sigmadd_neutralino(const high_prec_float& msnq, const high_prec_float& M1val, const high_prec_float& M2val,
                                 const high_prec_float& muval, const high_prec_float& g2sqval, const high_prec_float& gprsqval,
                                 const high_prec_float& vsqval, const high_prec_float& vuval, const high_prec_float& vdval, const high_prec_float& betaval,
                                 const high_prec_float& myQval);

inline high_prec_float Deltafunc(const high_prec_float& x, const high_prec_float& y, const high_prec_float& z);

inline high_prec_float Phifunc(const high_prec_float& x, const high_prec_float& y, const high_prec_float& z);

inline high_prec_float sigmauu_2loop(const high_prec_float& myQ, const high_prec_float& mu_wk, const high_prec_float& beta_wk, const high_prec_float& yt_wk, const high_prec_float& yc_wk, const high_prec_float& yu_wk, const high_prec_float& yb_wk, const high_prec_float& ys_wk,
                            const high_prec_float& yd_wk, const high_prec_float& ytau_wk, const high_prec_float& ymu_wk, const high_prec_float& ye_wk, const high_prec_float& g1_wk, const high_prec_float& g2_wk, const high_prec_float& g3_wk, const high_prec_float& mQ3_sq_wk,
                            const high_prec_float& mQ2_sq_wk, const high_prec_float& mQ1_sq_wk, const high_prec_float& mL3_sq_wk, const high_prec_float& mL2_sq_wk, const high_prec_float& mL1_sq_wk,
                            const high_prec_float& mU3_sq_wk, const high_prec_float& mU2_sq_wk, const high_prec_float& mU1_sq_wk, const high_prec_float& mD3_sq_wk, const high_prec_float& mD2_sq_wk, const high_prec_float& mD1_sq_wk,
                            const high_prec_float& mE3_sq_wk, const high_prec_float& mE2_sq_wk, const high_prec_float& mE1_sq_wk, const high_prec_float& M1_wk, const high_prec_float& M2_wk, const high_prec_float& M3_wk, const high_prec_float& mHu_sq_wk,
                            const high_prec_float& mHd_sq_wk, const high_prec_float& at_wk, const high_prec_float& ac_wk, const high_prec_float& au_wk, const high_prec_float& ab_wk, const high_prec_float& as_wk, const high_prec_float& ad_wk, const high_prec_float& atau_wk,
                            const high_prec_float& amu_wk, const high_prec_float& ae_wk, const high_prec_float& m_stop_1sq, const high_prec_float& m_stop_2sq, const high_prec_float& mymt, const high_prec_float& vHiggs_wk);

inline high_prec_float sigmadd_2loop(const high_prec_float& myQ, const high_prec_float& mu_wk, const high_prec_float& beta_wk, const high_prec_float& yt_wk, const high_prec_float& yc_wk, const high_prec_float& yu_wk, const high_prec_float& yb_wk, const high_prec_float& ys_wk,
                            const high_prec_float& yd_wk, const high_prec_float& ytau_wk, const high_prec_float& ymu_wk, const high_prec_float& ye_wk, const high_prec_float& g1_wk, const high_prec_float& g2_wk, const high_prec_float& g3_wk, const high_prec_float& mQ3_sq_wk,
                            const high_prec_float& mQ2_sq_wk, const high_prec_float& mQ1_sq_wk, const high_prec_float& mL3_sq_wk, const high_prec_float& mL2_sq_wk, const high_prec_float& mL1_sq_wk,
                            const high_prec_float& mU3_sq_wk, const high_prec_float& mU2_sq_wk, const high_prec_float& mU1_sq_wk, const high_prec_float& mD3_sq_wk, const high_prec_float& mD2_sq_wk, const high_prec_float& mD1_sq_wk,
                            const high_prec_float& mE3_sq_wk, const high_prec_float& mE2_sq_wk, const high_prec_float& mE1_sq_wk, const high_prec_float& M1_wk, const high_prec_float& M2_wk, const high_prec_float& M3_wk, const high_prec_float& mHu_sq_wk,
                            const high_prec_float& mHd_sq_wk, const high_prec_float& at_wk, const high_prec_float& ac_wk, const high_prec_float& au_wk, const high_prec_float& ab_wk, const high_prec_float& as_wk, const high_prec_float& ad_wk, const high_prec_float& atau_wk,
                            const high_prec_float& amu_wk, const high_prec_float& ae_wk, const high_prec_float& m_stop_1sq, const high_prec_float& m_stop_2sq, const high_prec_float& mymt, const high_prec_float& vHiggs_wk);

inline high_prec_float dew_funcu(const high_prec_float& inp, const high_prec_float& tangentbeta);

inline high_prec_float dew_funcd(const high_prec_float& inp, const high_prec_float& tangentbeta);


std::vector<LabeledValue> DEW_calc(std::vector<high_prec_float> weak_boundary_conditions, high_prec_float myQ);

#endif