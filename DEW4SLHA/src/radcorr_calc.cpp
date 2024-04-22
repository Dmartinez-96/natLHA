#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <gsl/gsl_sf_dilog.h>
#include <eigen3/Eigen/Dense>
#include "radcorr_calc.hpp"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

double spence(const double& spenceinp) {
    /*
    Return spence's function, or dilogarithm(spencinp).

    Parameters
    ----------
    spenceinp : double float.
        Input value to evaluate dilogarithm of.
    
    Returns
    -------
    myspenceval : double float.
        Return dilogarithm of spencinp.
    */
    double myspenceval = gsl_sf_dilog(spenceinp);
    return myspenceval;
}

double logfunc(const double& mass, const double& Q_renorm_sq) {
    /*
    Return F = m^2 * (ln(m^2 / Q^2) - 1.0), where input mass term is linear.

    Parameters
    ----------
    mass : double float.
        Input mass to be evaluated.
    Q_renorm_sq : double float.
        Squared renormalization scale, read in from supplied SLHA file.

    Returns
    -------
    myf : double float.
        Return F = m^2 * (ln(m^2 / Q^2) - 1.0),
        where input mass term is linear.

    */
    double myf = pow(mass, 2.0) * (log((pow(mass, 2.0))
                                      / (Q_renorm_sq)) - 1.0);
    return myf;
}

double logfunc2(const double& masssq, const double& Q_renorm_sq) {
    /*
    Return F = m^2 * (ln(m^2 / Q^2) - 1.0), where input mass term is
    quadratic.

    Parameters
    ----------
    mass : double float.
        Input mass to be evaluated.
    Q_renorm_sq : double float.
        Squared renormalization scale, read in from supplied SLHA file.

    Returns
    -------
    myf2 : double float.
        Return F = m^2 * (ln(m^2 / Q^2) - 1.0),
        where input mass term is quadratic.

    */
    double myf2 = masssq * (log((abs(masssq) / (Q_renorm_sq))) - 1.0);
    return myf2;
}

////////// Radiative corrections from neutralino sector //////////
double neutralino_denom(const double& msninp, const double& M1val, const double& M2val, const double& muval, const double& g2sqval,
                        const double& gprsqval, const double& vsqval, const double& vuval, const double& vdval, const double& betaval) {
    /*
    Return denominator for one-loop correction
        of neutralino according to method of Ibrahim
        and Nath in PhysRevD.66.015005 (2002).

    Parameters
    ----------
    msninp : double float.
        Neutralino un-squared mass used for evaluating results.
    //TODO: Finish this documentation

    Returns
    -------
    myden : double float.
        Return denominator of neutralino radiative corrections.
    */
    
    // Introduce coefficients of characteristic equation for eigenvals.
    // Char. eqn. is of the form x^4 + ax^3 + bx^2 + cx + d = 0
    double char_a = (-1.0) * (M1val + M2val);
    double char_b = ((M1val * M2val) - (pow(muval, 2.0))
              - ((vsqval / 2.0) * (g2sqval + gprsqval)));
    double char_c = ((pow(muval, 2.0) * (M1val + M2val))
              - (muval * vdval * vuval * (g2sqval + gprsqval))
              + ((vsqval / 2.0)
                 * ((g2sqval * M1val) + (gprsqval * M2val))));
    double myden = (4.0 * pow(msninp, 3.0)) + (3.0 * char_a
                                         * pow(msninp, 2.0))\
        + (2.0 * char_b * msninp) + char_c;
    return myden;
}

double neutralinouu_num(const double& msninp, const double& M1val, const double& M2val, const double& muval, const double& g2sqval,
                        const double& gprsqval, const double& vsqval, const double& vuval, const double& vdval, const double& betaval) {
    /*
    Return numerator for one-loop uu correction
        derivative term of neutralino.

    Parameters
    ----------
    msninp : Float.
        Neutralino un-squared mass used for evaluating results.

    */
    double quadrterm = ((-1.0) * vuval) * (gprsqval + g2sqval);
    double linterm = (((g2sqval * M1val) + (gprsqval * M2val)) * vuval)\
        - (muval * vdval * (g2sqval + gprsqval));
    double constterm = muval * vdval * ((g2sqval * M1val) + (gprsqval * M2val));
    double mynum = (quadrterm * pow(msninp, 2.0))\
        + (linterm * msninp) + constterm;
    return mynum;
}

double neutralinodd_num(const double& msninp, const double& M1val, const double& M2val, const double& muval, const double& g2sqval,
                        const double& gprsqval, const double& vsqval, const double& vuval, const double& vdval, const double& betaval) {
    /*
    Return numerator for one-loop dd correction derivative term of
        neutralino.

    Parameters
    ----------
    msninp : Float.
        Neutralino squared mass used for evaluating results.

    */
    double quadrterm = ((-1.0) * vdval) * (gprsqval + g2sqval);
    double linterm = (((g2sqval * M1val) + (gprsqval * M2val)) * vdval)\
        - (muval * vuval * (g2sqval + gprsqval));
    double constterm = muval * vuval * ((g2sqval * M1val) + (gprsqval * M2val));
    double mynum = (quadrterm * pow(msninp, 2.0))\
        + (linterm * msninp) + constterm;
    return mynum;
}

double sigmauu_neutralino(const double& msninp, const double& M1val, const double& M2val, const double& muval, const double& g2sqval,
                          const double& gprsqval, const double& vsqval, const double& vuval, const double& vdval, const double& betaval, const double& myQval) {
    /*
    Return one-loop correction Sigma_u^u(neutralino).

    Parameters
    ----------
    msninp : Float.
        Neutralino un-squared mass.

    */
    double sigma_uu_neutralino = ((1.0 / (16.0 * (pow(M_PI, 2.0)))) * msninp / vuval) \
        * ((neutralinouu_num(msninp, M1val, M2val, muval, g2sqval, gprsqval, vsqval, vuval, vdval, betaval)
            / neutralino_denom(msninp, M1val, M2val, muval, g2sqval, gprsqval, vsqval, vuval, vdval, betaval))
           * logfunc2((msninp * msninp), pow(myQval, 2.0)));
    return sigma_uu_neutralino;
}

double sigmadd_neutralino(const double& msninp, const double& M1val, const double& M2val, const double& muval, const double& g2sqval,
                          const double& gprsqval, const double& vsqval, const double& vuval, const double& vdval, const double& betaval, const double& myQval) {
    /*
    Return one-loop correction Sigma_d^d(neutralino).

    Parameters
    ----------
    msninp : Float.
        Neutralino un-squared mass.

    */
    double sigma_dd_neutralino = ((1.0 / (16.0 * (pow(M_PI, 2.0)))) * msninp / vdval) \
        * ((neutralinodd_num(msninp, M1val, M2val, muval, g2sqval, gprsqval, vsqval, vuval, vdval, betaval)
            / neutralino_denom(msninp, M1val, M2val, muval, g2sqval, gprsqval, vsqval, vuval, vdval, betaval))
           * logfunc2((msninp * msninp), pow(myQval, 2.0)));
    return sigma_dd_neutralino;
}
////////// Radiative corrections from two-loop O(alpha_t alpha_s) sector //////////
// Corrections come from Dedes, Slavich paper, arXiv:hep-ph/0212132.
// alpha_i = y_i^2 / (4.0 * pi)

double Deltafunc(const double& x, const double& y, const double& z) {
    /*
    DOCFUNC HERE
    */
    double mydelta = pow(x, 2.0) + pow(y, 2.0) + pow(z, 2.0)\
        - (2.0 * ((x * y) + (x * z) + (y * z)));
    return mydelta;
}

double Phifunc(const double& x, const double& y, const double& z) {
    /*
    DOCFUNC HERE
    */
    complex<double> myu, myv, mylambda, myxp, myxm, myphi;
    if((abs(x) < abs(z)) && (abs(y) < abs(z))) {
        myu = x / z;
        myv = y / z;
        mylambda = sqrt(pow((1.0 - myu - myv), 2.0)
                                   - (4.0 * myu * myv));
        myxp = 0.5 * (1.0 + myu - myv - mylambda);
        myxm = 0.5 * (1.0 - myu + myv - mylambda);
        myphi = (1.0 / mylambda) * ((2.0 * log(myxp)
                                 * log(myxm))
                                    - (log(myu)
                                       * log(myv))
                                    - (2.0 * (spence(real(myxp))
                                       + spence(real(myxm)))) + static_cast<complex<double>>(pow(M_PI, 2.0) / 3.0));
    }
    else if((abs(x) > abs(z)) && (abs(y) < abs(z))) {
        myu = z / x;
        myv = y / x;
        mylambda = sqrt(pow((1.0 - myu - myv), 2.0)
                                  - (4.0 * myu * myv));
        myxp = 0.5 * (1.0 + myu - myv - mylambda);
        myxm = 0.5 * (1.0 - myu + myv - mylambda);
        myphi = (z / x) * (1.0 / mylambda)\
            * ((2.0 * log(myxp)
                * log(myxm))
               - (log(myu)
                  * log(myv))
               - (2.0 * (spence(real(myxp))
                       + spence(real(myxm))))
               + static_cast<complex<double>>(pow(M_PI, 2.0) / 3.0));
    }
    else if((abs(x) > abs(z)) && (abs(y) > abs(z)) && (abs(x) > abs(y))) {
        myu = z / x;
        myv = y / x;
        mylambda = sqrt(pow((1.0 - myu - myv), 2.0)
                                  - (4.0 * myu * myv));
        myxp = 0.5 * (1.0 + myu - myv - mylambda);
        myxm = 0.5 * (1.0 - myu + myv - mylambda);
        myphi = (z / x) * (1.0 / mylambda)\
            * ((2.0 * log(myxp)
                * log(myxm))
               - (log(myu)
                  * log(myv))
               - (2.0 * (spence(real(myxp))
                       + spence(real(myxm))))
               + static_cast<complex<double>>(pow(M_PI, 2.0) / 3.0));
    }
    else if((abs(x) < abs(z)) && (abs(y) > abs(z))) {
        myu = z / y;
        myv = x / y;
        mylambda = sqrt(pow((1.0 - myu - myv), 2.0)
                                  - (4.0 * myu * myv));
        myxp = 0.5 * (1.0 + myu - myv - mylambda);
        myxm = 0.5 * (1.0 - myu + myv - mylambda);
        myphi = (z / y) * (1.0 / mylambda)\
            * ((2.0 * log(myxp)
                * log(myxm))
               - (log(myu)
                  * log(myv))
               - (2.0 * (spence(real(myxp))
                       + spence(real(myxm))))
               + static_cast<complex<double>>(pow(M_PI, 2.0) / 3.0));
    }
    else if ((abs(x) > abs(z)) && (abs(y) > abs(z)) && (abs(y) > abs(x))) {
        myu = z / y;
        myv = x / y;
        mylambda = sqrt(pow((1.0 - myu - myv), 2.0)
                                  - (4.0 * myu * myv));
        myxp = 0.5 * (1.0 + myu - myv - mylambda);
        myxm = 0.5 * (1.0 - myu + myv - mylambda);
        myphi = (z / y) * (1.0 / mylambda)\
            * ((2.0 * log(myxp)
                * log(myxm))
               - (log(myu)
                  * log(myv))
               - (2.0 * (spence(real(myxp))
                       + spence(real(myxm))))
               + static_cast<complex<double>>(pow(M_PI, 2.0) / 3.0));
    }
    else {
        myphi = 0.0;
    }
    return static_cast<double>(real(myphi));
}

double sigmauu_2loop(const double& myQ, const double& mu_wk, const double& beta_wk, const double& yt_wk, const double& yc_wk, const double& yu_wk, const double& yb_wk, const double& ys_wk,
                     const double& yd_wk, const double& ytau_wk, const double& ymu_wk, const double& ye_wk, const double& g1_wk, const double& g2_wk, const double& g3_wk, const double& mQ3_sq_wk,
                     const double& mQ2_sq_wk, const double& mQ1_sq_wk, const double& mL3_sq_wk, const double& mL2_sq_wk, const double& mL1_sq_wk,
                     const double& mU3_sq_wk, const double& mU2_sq_wk, const double& mU1_sq_wk, const double& mD3_sq_wk, const double& mD2_sq_wk, const double& mD1_sq_wk,
                     const double& mE3_sq_wk, const double& mE2_sq_wk, const double& mE1_sq_wk, const double& M1_wk, const double& M2_wk, const double& M3_wk, const double& mHu_sq_wk,
                     const double& mHd_sq_wk, const double& at_wk, const double& ac_wk, const double& au_wk, const double& ab_wk, const double& as_wk, const double& ad_wk, const double& atau_wk,
                     const double& amu_wk, const double& ae_wk, const double& m_stop_1sq, const double& m_stop_2sq, const double& mymt, const double& vHiggs_wk) {
    double s2theta = 2.0 * mymt * ((at_wk / yt_wk) - (mu_wk / tan(beta_wk)))\
        / (m_stop_1sq - m_stop_2sq);
    double s2sqtheta = pow(s2theta, 2.0);
    double c2sqtheta = 1.0 - s2sqtheta;
    double mglsq = pow((M3_wk), 2.0);
    double myunits = pow(g3_wk, 2.0) * 4.0 * pow((1.0 / (16.0 * (pow(M_PI, 2.0)))), 2.0);
    double Q_renorm_sq = pow(myQ, 2.0);
    double myF = myunits\
        * (((4.0 * (M3_wk) * mymt / s2theta) * (1.0 + (4.0 * c2sqtheta)))
           - (((2.0 * (m_stop_1sq - m_stop_2sq)) + (4.0 * (M3_wk) * mymt / s2theta))
              * log(mglsq / Q_renorm_sq)
              * log(pow(mymt, 2.0) / Q_renorm_sq))
           - (2.0 * (4.0 - s2sqtheta) * (m_stop_1sq - m_stop_2sq))
           + ((((4.0 * m_stop_1sq * m_stop_2sq)
                - s2sqtheta * pow((m_stop_1sq + m_stop_2sq), 2.0))
               / (m_stop_1sq - m_stop_2sq))
              * (log((m_stop_1sq / Q_renorm_sq)))
              * (log(m_stop_2sq / Q_renorm_sq)))
             + ((((4.0 * (mglsq + pow(mymt, 2.0) + (2.0 * m_stop_1sq)))
                  - (s2sqtheta * ((3.0 * m_stop_1sq) + m_stop_2sq))
                  - ((16.0 * c2sqtheta * (M3_wk) * mymt * m_stop_1sq)
                     / (s2theta * (m_stop_1sq - m_stop_2sq)))
                  - (4.0 * s2theta * (M3_wk) * mymt))
                 * log((m_stop_1sq / Q_renorm_sq)))
                + ((m_stop_1sq / (m_stop_1sq - m_stop_2sq))
                   * ((s2sqtheta * (m_stop_1sq + m_stop_2sq))
                      - ((4.0 * m_stop_1sq) - (2.0 * m_stop_2sq)))
                   * pow(log((m_stop_1sq / Q_renorm_sq)), 2.0))
                + (2.0 * (m_stop_1sq - mglsq - pow(mymt, 2.0)
                        + ((M3_wk) * mymt * s2theta)
                        + ((2.0 * c2sqtheta * (M3_wk) * mymt * m_stop_1sq)
                           / (s2theta * (m_stop_1sq - m_stop_2sq))))
                   * log(mglsq * pow(mymt, 2.0)
                            / (pow(Q_renorm_sq, 2.0)))
                   * log((m_stop_1sq / Q_renorm_sq)))
                + (((4.0 * (M3_wk) * mymt * c2sqtheta * (pow(mymt, 2.0) - mglsq))
                    / (s2theta * (m_stop_1sq - m_stop_2sq)))
                   * log(pow(mymt, 2.0) / mglsq)
                   * log((m_stop_1sq / Q_renorm_sq)))
                + (((((4.0 * mglsq * pow(mymt, 2.0))
                      + (2.0 * Deltafunc(mglsq, pow(mymt, 2.0), m_stop_1sq))) / m_stop_1sq)
                    - (((2.0 * (M3_wk) * mymt * s2theta) / m_stop_1sq)
                       * (mglsq + pow(mymt, 2.0) - m_stop_1sq))
                    + ((4.0 * c2sqtheta * (M3_wk) * mymt
                        * Deltafunc(mglsq, pow(mymt, 2.0), m_stop_1sq))
                       / (s2theta * m_stop_1sq * (m_stop_1sq - m_stop_2sq))))
                   * Phifunc(mglsq, pow(mymt, 2.0), m_stop_1sq)))
             - ((((4.0 * (mglsq + pow(mymt, 2.0) + (2.0 * m_stop_2sq)))
                  - (s2sqtheta * ((3.0 * m_stop_2sq) + m_stop_1sq))
                  - ((16.0 * c2sqtheta * (M3_wk) * mymt * m_stop_2sq)
                     / (((-1.0) * s2theta) * (m_stop_2sq - m_stop_1sq)))
                  - ((-4.0) * s2theta * (M3_wk) * mymt))
                 * log(m_stop_2sq / Q_renorm_sq))
                + ((m_stop_2sq / (m_stop_2sq - m_stop_1sq))
                   * ((s2sqtheta * (m_stop_2sq + m_stop_1sq))
                      - ((4.0 * m_stop_2sq) - (2.0 * m_stop_1sq)))
                   * pow(log(m_stop_2sq / Q_renorm_sq), 2.0))
                + (2.0 * (m_stop_2sq - mglsq - pow(mymt, 2.0)
                        - ((M3_wk) * mymt * s2theta)
                        + ((2.0 * c2sqtheta * (M3_wk) * mymt * m_stop_2sq)
                           / (s2theta * (m_stop_1sq - m_stop_2sq))))
                   * log(mglsq * pow(mymt, 2.0)
                            / (pow(Q_renorm_sq, 2.0)))
                   * log(m_stop_2sq / Q_renorm_sq))
                + (((4.0 * (M3_wk) * mymt * c2sqtheta * (pow(mymt, 2.0) - mglsq))
                    / (s2theta * (m_stop_1sq - m_stop_2sq)))
                   * log(pow(mymt, 2.0) / mglsq)
                   * log(m_stop_2sq / Q_renorm_sq))
                + (((((4.0 * mglsq * pow(mymt, 2.0))
                      + (2.0 * Deltafunc(mglsq, pow(mymt, 2.0), m_stop_2sq))) / m_stop_2sq)
                    - ((((-2.0) * (M3_wk) * mymt * s2theta) / m_stop_2sq)
                       * (mglsq + pow(mymt, 2.0) - m_stop_2sq))
                    + ((4.0 * c2sqtheta * (M3_wk) * mymt
                        * Deltafunc(mglsq, pow(mymt, 2.0), m_stop_2sq))
                       / (s2theta * m_stop_2sq * (m_stop_1sq - m_stop_2sq))))
                   * Phifunc(mglsq, pow(mymt, 2.0), m_stop_2sq))));
    double myG = myunits\
        * ((5.0 * (M3_wk) * s2theta * (m_stop_1sq - m_stop_2sq) / mymt)
           - (10.0 * (m_stop_1sq + m_stop_2sq - (2.0 * pow(mymt, 2.0))))
           - (4.0 * mglsq) + ((12.0 * pow(mymt, 2.0))
                            * (pow(log(pow(mymt, 2.0) / Q_renorm_sq), 2.0)
                               - (2.0 * log(pow(mymt, 2.0) / Q_renorm_sq))))
           + (((4.0 * mglsq) - (((M3_wk) * s2theta / mymt)
                              * (m_stop_1sq - m_stop_2sq)))
              * log(mglsq / Q_renorm_sq) * log(pow(mymt, 2.0) / Q_renorm_sq))
           + (s2sqtheta * (m_stop_1sq + m_stop_2sq)
              * log((m_stop_1sq / Q_renorm_sq))
              * log(m_stop_2sq / Q_renorm_sq))
           + ((((4.0 * (mglsq + pow(mymt, 2.0) + (2.0 * m_stop_1sq)))
                + (s2sqtheta * (m_stop_1sq - m_stop_2sq))
                - ((4.0 * (M3_wk) * s2theta / mymt) * (pow(mymt, 2.0) + m_stop_1sq)))
               * log((m_stop_1sq / Q_renorm_sq)))
              + ((((M3_wk) * s2theta * ((5.0 * pow(mymt, 2.0)) - mglsq + m_stop_1sq)
                   / mymt)
                  - (2.0 * (mglsq + 2.0 * pow(mymt, 2.0))))
                 * log(pow(mymt, 2.0) / Q_renorm_sq)
                 * log((m_stop_1sq / Q_renorm_sq)))
              + ((((M3_wk) * s2theta * (mglsq - pow(mymt, 2.0) + m_stop_1sq) / mymt)
                  - (2.0 * mglsq))
                 * log(mglsq / Q_renorm_sq)
                 * log((m_stop_1sq / Q_renorm_sq)))
              - ((2.0 + s2sqtheta) * m_stop_1sq
                 * pow(log((m_stop_1sq / Q_renorm_sq)), 2.0))
              + (((2.0 * mglsq * (mglsq + pow(mymt, 2.0) - m_stop_1sq
                                - (2.0 * (M3_wk) * mymt * s2theta)) / m_stop_1sq)
                  + (((M3_wk) * s2theta / (mymt * m_stop_1sq))
                     * Deltafunc(mglsq, pow(mymt, 2.0), m_stop_1sq)))
                 * Phifunc(mglsq, pow(mymt, 2.0), m_stop_1sq)))
           + ((((4.0 * (mglsq + pow(mymt, 2.0) + (2.0 * m_stop_2sq)))
                + (s2sqtheta * (m_stop_2sq - m_stop_1sq))
                - (((-4.0) * (M3_wk) * s2theta / mymt) * (pow(mymt, 2.0) + m_stop_2sq)))
               * log(m_stop_2sq / Q_renorm_sq))
              + ((((-1.0) * (M3_wk) * s2theta * ((5.0 * pow(mymt, 2.0)) - mglsq + m_stop_2sq)
                   / mymt)
                  - (2.0 * (mglsq + 2.0 * pow(mymt, 2.0))))
                 * log(pow(mymt, 2.0) / Q_renorm_sq)
                 * log(m_stop_2sq / Q_renorm_sq))
              + ((((-1.0) * (M3_wk) * s2theta * (mglsq - pow(mymt, 2.0) + m_stop_2sq)
                   / mymt)
                  - (2.0 * mglsq))
                 * log(mglsq / Q_renorm_sq)
                 * log(m_stop_2sq / Q_renorm_sq))
              - ((2.0 + s2sqtheta) * m_stop_2sq
                 * pow(log(m_stop_2sq / Q_renorm_sq), 2.0))
              + (((2.0 * mglsq
                   * (mglsq + pow(mymt, 2.0) - m_stop_2sq
                      + (2.0 * (M3_wk) * mymt * s2theta)) / m_stop_2sq)
                  + (((M3_wk) * (-1.0) * s2theta / (mymt * m_stop_2sq))
                     * Deltafunc(mglsq, pow(mymt, 2.0), m_stop_2sq)))
                 * Phifunc(mglsq, pow(mymt, 2.0), m_stop_2sq))));
    double sinsqb = pow(sin(beta_wk), 2.0);
    double mysigmauu_2loop = ((mymt * (at_wk / yt_wk) * s2theta * myF)
                       + 2.0 * pow(mymt, 2.0) * myG)\
        / (pow((vHiggs_wk), 2.0) * sinsqb);
    return real(mysigmauu_2loop);
}

double sigmadd_2loop(const double& myQ, const double& mu_wk, const double& beta_wk, const double& yt_wk, const double& yc_wk, const double& yu_wk, const double& yb_wk, const double& ys_wk,
                     const double& yd_wk, const double& ytau_wk, const double& ymu_wk, const double& ye_wk, const double& g1_wk, const double& g2_wk, const double& g3_wk, const double& mQ3_sq_wk,
                     const double& mQ2_sq_wk, const double& mQ1_sq_wk, const double& mL3_sq_wk, const double& mL2_sq_wk, const double& mL1_sq_wk,
                     const double& mU3_sq_wk, const double& mU2_sq_wk, const double& mU1_sq_wk, const double& mD3_sq_wk, const double& mD2_sq_wk, const double& mD1_sq_wk,
                     const double& mE3_sq_wk, const double& mE2_sq_wk, const double& mE1_sq_wk, const double& M1_wk, const double& M2_wk, const double& M3_wk, const double& mHu_sq_wk,
                     const double& mHd_sq_wk, const double& at_wk, const double& ac_wk, const double& au_wk, const double& ab_wk, const double& as_wk, const double& ad_wk, const double& atau_wk,
                     const double& amu_wk, const double& ae_wk, const double& m_stop_1sq, const double& m_stop_2sq, const double& mymt, const double& vHiggs_wk) {
    double Q_renorm_sq = pow(myQ, 2.0);
    double s2theta = (2.0 * mymt * ((at_wk / yt_wk)
                           - (mu_wk / tan(beta_wk))))\
        / (m_stop_1sq - m_stop_2sq);
    double s2sqtheta = pow(s2theta, 2.0);
    double c2sqtheta = 1.0 - s2sqtheta;
    double mglsq = pow(M3_wk, 2.0);
    double myunits = pow(g3_wk, 2.0) * 4\
        / pow((16.0 * pow(M_PI, 2.0)), 2.0);
    double myF = myunits\
        * ((4.0 * (M3_wk) * mymt / s2theta) * (1.0 + 4.0 * c2sqtheta)
           - (((2.0 * (m_stop_1sq - m_stop_2sq))
              + (4.0 * (M3_wk) * mymt / s2theta))
              * log(mglsq / Q_renorm_sq)
              * log(pow(mymt, 2.0) / Q_renorm_sq))
           - (2.0 * (4.0 - s2sqtheta)
              * (m_stop_1sq - m_stop_2sq))
           + ((((4.0 * m_stop_1sq * m_stop_2sq)
                - s2sqtheta * pow((m_stop_1sq + m_stop_2sq), 2.0))
               / (m_stop_1sq - m_stop_2sq))
              * (log((m_stop_1sq / Q_renorm_sq)))
              * (log(m_stop_2sq / Q_renorm_sq)))
           + ((((4.0 * (mglsq + pow(mymt, 2.0) + (2.0 * m_stop_1sq)))
               - (s2sqtheta * ((3.0 * m_stop_1sq) + m_stop_2sq))
               - ((16.0 * c2sqtheta * (M3_wk) * mymt * m_stop_1sq)
                  / (s2theta * (m_stop_1sq - m_stop_2sq)))
               - (4.0 * s2theta * (M3_wk) * mymt))
               * log((m_stop_1sq / Q_renorm_sq)))
              + ((m_stop_1sq / (m_stop_1sq - m_stop_2sq))
                 * ((s2sqtheta * (m_stop_1sq + m_stop_2sq))
                    - ((4.0 * m_stop_1sq) - (2.0 * m_stop_2sq)))
                 * pow(log((m_stop_1sq / Q_renorm_sq)), 2.0))
              + (2.0 * (m_stop_1sq - mglsq - pow(mymt, 2.0)
                      + ((M3_wk) * mymt * s2theta)
                      + ((2.0 * c2sqtheta * (M3_wk) * mymt * m_stop_1sq)
                         / (s2theta * (m_stop_1sq - m_stop_2sq))))
                 * log(mglsq * pow(mymt, 2.0)
                          / (pow(Q_renorm_sq, 2.0)))
                 * log((m_stop_1sq / Q_renorm_sq)))
              + (((4.0 * (M3_wk) * mymt * c2sqtheta * (pow(mymt, 2.0) - mglsq))
                  / (s2theta * (m_stop_1sq - m_stop_2sq)))
                 * log(pow(mymt, 2.0) / mglsq)
                 * log((m_stop_1sq / Q_renorm_sq)))
              + (((((4.0 * mglsq * pow(mymt, 2.0))
                    + (2.0 * Deltafunc(mglsq, pow(mymt, 2.0), m_stop_1sq))) / m_stop_1sq)
                  - (((2.0 * (M3_wk) * mymt * s2theta) / m_stop_1sq)
                     * (mglsq + pow(mymt, 2.0) - m_stop_1sq))
                  + ((4.0 * c2sqtheta * (M3_wk) * mymt
                      * Deltafunc(mglsq, pow(mymt, 2.0), m_stop_1sq))
                     / (s2theta * m_stop_1sq * (m_stop_1sq - m_stop_2sq))))
                 * Phifunc(mglsq, pow(mymt, 2.0), m_stop_1sq)))
           - ((((4.0 * (mglsq + pow(mymt, 2.0) + (2.0 * m_stop_2sq)))
               - (s2sqtheta * ((3.0 * m_stop_2sq) + m_stop_1sq))
               - ((16.0 * c2sqtheta * (M3_wk) * mymt * m_stop_2sq)
                  / (((-1.0) * s2theta) * (m_stop_2sq - m_stop_1sq)))
               - ((-4.0) * s2theta * (M3_wk) * mymt))
               * log(m_stop_2sq / Q_renorm_sq))
              + ((m_stop_2sq / (m_stop_2sq - m_stop_1sq))
                 * ((s2sqtheta * (m_stop_2sq + m_stop_1sq))
                    - ((4.0 * m_stop_2sq) - (2.0 * m_stop_1sq)))
                 * pow(log(m_stop_2sq / Q_renorm_sq), 2.0))
              + (2.0 * (m_stop_2sq - mglsq - pow(mymt, 2.0)
                      - ((M3_wk) * mymt * s2theta)
                      + ((2.0 * c2sqtheta * (M3_wk) * mymt * m_stop_2sq)
                         / (s2theta * (m_stop_1sq - m_stop_2sq))))
                 * log(mglsq * pow(mymt, 2.0)
                          / (pow(Q_renorm_sq, 2.0)))
                 * log(m_stop_2sq / Q_renorm_sq))
              + (((4.0 * (M3_wk) * mymt * c2sqtheta * (pow(mymt, 2.0) - mglsq))
                  / (s2theta * (m_stop_1sq - m_stop_2sq)))
                 * log(pow(mymt, 2.0) / mglsq)
                 * log(m_stop_2sq / Q_renorm_sq))
              + (((((4.0 * mglsq * pow(mymt, 2.0))
                    + (2.0 * Deltafunc(mglsq, pow(mymt, 2.0), m_stop_2sq))) / m_stop_2sq)
                  - ((((-2.0) * (M3_wk) * mymt * s2theta) / m_stop_2sq)
                     * (mglsq + pow(mymt, 2.0) - m_stop_2sq))
                  + ((4.0 * c2sqtheta * (M3_wk) * mymt
                      * Deltafunc(mglsq, pow(mymt, 2.0), m_stop_2sq))
                     / (s2theta * m_stop_2sq * (m_stop_1sq - m_stop_2sq))))
                 * Phifunc(mglsq, pow(mymt, 2.0), m_stop_2sq))));
    double cossqb = (pow(cos(beta_wk), 2.0));
    double mysigmadd_2loop = (mymt * (-1.0 * mu_wk) * (1.0 / tan(beta_wk))
                       * s2theta * myF)\
        / (pow((vHiggs_wk), 2.0) * cossqb);
    return real(mysigmadd_2loop);
}

double dew_funcu(const double& inp, const double& tangentbeta) {
    /*
    Compute individual one-loop DEW contributions from Sigma_u^u.

    Parameters
    ----------
    inp : One-loop correction or Higgs to be inputted into the DEW eval.

    */
    double mycontribuu = ((-1.0) * inp * pow(tangentbeta, 2.0) / (pow(tangentbeta, 2.0) - 1.0));
    return mycontribuu;
}

double dew_funcd(const double& inp, const double& tangentbeta) {
    /*
    Compute individual one-loop DEW contributions from Sigma_d^d.

    Parameters
    ----------
    inp : One-loop correction or Higgs to be inputted into the DEW eval.

    */
    double mycontribdd = (inp / (pow(tangentbeta, 2.0) - 1.0));
    return mycontribdd;
}

std::vector<double> radcorr_calc(std::vector<double> weak_boundary_conditions, double myQ, double mymZsq) {
    /*
    DOCSTRING HERE
    */
    // Gauge couplings
    if (mymZsq < 0) {
        //cout << "Warning! mZ^2 < 0 in radcorr_calc, using abs(mZ^2)" << endl;
    }
    const double mymZ = sqrt(abs(mymZsq));
    const double g1_wk = weak_boundary_conditions[0];
    const double g2_wk = weak_boundary_conditions[1];
    const double g3_wk = weak_boundary_conditions[2];
    // Higgs parameters
    const double beta_wk = atan(weak_boundary_conditions[43]);
    const double mu_wk = weak_boundary_conditions[6];
    const double mu_wk_sq = pow(mu_wk, 2.0);
    // Yukawas
    const double yt_wk = weak_boundary_conditions[7];
    const double yc_wk = weak_boundary_conditions[8];
    const double yu_wk = weak_boundary_conditions[9];
    const double yb_wk = weak_boundary_conditions[10];
    const double ys_wk = weak_boundary_conditions[11];
    const double yd_wk = weak_boundary_conditions[12];
    const double ytau_wk = weak_boundary_conditions[13];
    const double ymu_wk = weak_boundary_conditions[14];
    const double ye_wk = weak_boundary_conditions[15];
    // Soft trilinears
    const double at_wk = weak_boundary_conditions[16];
    const double ac_wk = weak_boundary_conditions[17];
    const double au_wk = weak_boundary_conditions[18];
    const double ab_wk = weak_boundary_conditions[19];
    const double as_wk = weak_boundary_conditions[20];
    const double ad_wk = weak_boundary_conditions[21];
    const double atau_wk = weak_boundary_conditions[22];
    const double amu_wk = weak_boundary_conditions[23];
    const double ae_wk = weak_boundary_conditions[24];
    // Gaugino masses
    const double M1_wk = weak_boundary_conditions[3];
    const double M2_wk = weak_boundary_conditions[4];
    const double M3_wk = weak_boundary_conditions[5];
    // Soft mass dim. 2 terms
    const double mHu_sq_wk = weak_boundary_conditions[25];
    const double mHd_sq_wk = weak_boundary_conditions[26];
    const double mQ1_sq_wk = weak_boundary_conditions[27];
    const double mQ2_sq_wk = weak_boundary_conditions[28];
    const double mQ3_sq_wk = weak_boundary_conditions[29];
    const double mL1_sq_wk = weak_boundary_conditions[30];
    const double mL2_sq_wk = weak_boundary_conditions[31];
    const double mL3_sq_wk = weak_boundary_conditions[32];
    const double mU1_sq_wk = weak_boundary_conditions[33];
    const double mU2_sq_wk = weak_boundary_conditions[34];
    const double mU3_sq_wk = weak_boundary_conditions[35];
    const double mD1_sq_wk = weak_boundary_conditions[36];
    const double mD2_sq_wk = weak_boundary_conditions[37];
    const double mD3_sq_wk = weak_boundary_conditions[38];
    const double mE1_sq_wk = weak_boundary_conditions[39];
    const double mE2_sq_wk = weak_boundary_conditions[40];
    const double mE3_sq_wk = weak_boundary_conditions[41];
    const double b_wk = weak_boundary_conditions[42];
    double gpr_wk = g1_wk * sqrt(3.0 / 5.0);
    // // cout << "gpr_wk: " << gpr_wk << endl;
    double gpr_sq = pow(gpr_wk, 2.0);
    // // cout << "gpr_sq: " << gpr_sq << endl;
    double g2_sq = pow(g2_wk, 2.0);
    // // cout << "g2_sq: " << g2_sq << endl;
    // // cout << "mu_wk_sq: " << mu_wk_sq << endl;
    double vHiggs_wk = mymZ * sqrt(2.0 / (gpr_sq + g2_sq));
    double sinsqb = pow(sin(beta_wk), 2.0);
    // // cout << "sinsqb: " << sinsqb << endl;
    double cossqb = pow(cos(beta_wk), 2.0);
    // // cout << "cossqb: " << cossqb << endl;
    double vu = vHiggs_wk * sqrt(sinsqb);
    // // cout << "vu: " << vu << endl;
    double vd = vHiggs_wk * sqrt(cossqb);
    // // cout << "vd: " << vd << endl;
    double vu_sq = pow(vu, 2.0);
    // // cout << "vu_sq: " << vu_sq << endl;
    double vd_sq = pow(vd, 2.0);
    // // cout << "vd_sq: " << vd_sq << endl;
    double v_sq = pow(vHiggs_wk, 2.0);
    // // cout << "v_sq: " << v_sq << endl;
    double tan_th_w = gpr_wk / g2_wk;
    // // cout << "tan_th_w: " << tan_th_w << endl;
    double theta_w = atan(tan_th_w);
    // // cout << "theta_w: " << theta_w << endl;
    double sinsq_th_w = pow(sin(theta_w), 2.0);
    // // cout << "sinsq_th_w: " << sinsq_th_w << endl;
    double cos2b = cos(2.0 * beta_wk);
    // // cout << "cos2b: " << cos2b << endl;
    double sin2b = sin(2.0 * beta_wk);
    // // cout << "sin2b: " << sin2b << endl;
    double gz_sq = (pow(g2_wk, 2.0) + pow(gpr_wk, 2.0)) / 8.0;
    // // cout << "gz_sq: " << gz_sq << endl;

    ////////// Mass relations: //////////

    // W-boson tree-level running squared mass
    const double m_w_sq = (pow(g2_wk, 2.0) / 2.0) * v_sq;

    // Z-boson tree-level running squared mass
    const double mz_q_sq = mymZsq;// v_sq* ((pow(g2_wk, 2.0) + pow(gpr_wk, 2.0)) / 2.0);

    // Higgs psuedoscalar tree-level running squared mass
    const double mA0sq = 2.0 * mu_wk_sq + mHu_sq_wk + mHd_sq_wk;

    // Top quark tree-level running mass
    const double mymt = yt_wk * vu;
    const double mymtsq = pow(mymt, 2.0);

    // Bottom quark tree-level running mass
    const double mymb = yb_wk * vd;
    const double mymbsq = pow(mymb, 2.0);

    // Tau tree-level running mass
    const double mymtau = ytau_wk * vd;
    const double mymtausq = pow(mymtau, 2.0);

    // Charm quark tree-level running mass
    const double mymc = yc_wk * vu;
    const double mymcsq = pow(mymc, 2.0);

    // Strange quark tree-level running mass
    const double myms = ys_wk * vd;
    const double mymssq = pow(myms, 2.0);

    // Muon tree-level running mass
    const double mymmu = ymu_wk * vd;
    const double mymmusq = pow(mymmu, 2.0);

    // Up quark tree-level running mass
    const double mymu = yu_wk * vu;
    const double mymusq = pow(mymu, 2.0);

    // Down quark tree-level running mass
    const double mymd = yd_wk * vd;
    const double mymdsq = pow(mymd, 2.0);

    // Electron tree-level running mass
    const double myme = ye_wk * vd;
    const double mymesq = pow(myme, 2.0);

    // Sneutrino running masses
    const double mselecneutsq = mL1_sq_wk + (0.25 * (gpr_sq + g2_sq) * (vd_sq - vu_sq));
    const double msmuneutsq = mL2_sq_wk + (0.25 * (gpr_sq + g2_sq) * (vd_sq - vu_sq));
    const double mstauneutsq = mL3_sq_wk + (0.25 * (gpr_sq + g2_sq) * (vd_sq - vu_sq));

    // Tree-level charged Higgs running squared mass.
    const double mH_pmsq = mA0sq + m_w_sq;

    // Set up hyperfine splitting contributions to squark/slepton masses
    double Delta_suL = (pow(vu, 2.0) - pow(vd, 2.0)) * ((gpr_sq / 6.0) - (g2_sq / 4.0));
    double Delta_suR = (-1.0) * (pow(vu, 2.0) - pow(vd, 2.0)) * ((4.0 * gpr_sq / 3.0));
    double Delta_sdL = (pow(vu, 2.0) - pow(vd, 2.0)) * ((gpr_sq / 6.0) + (g2_sq / 4.0));
    double Delta_sdR = (pow(vu, 2.0) - pow(vd, 2.0)) * ((gpr_sq / 3.0));
    double Delta_seL = (pow(vu, 2.0) - pow(vd, 2.0)) * ((g2_sq / 4.0) - (gpr_sq / 2.0));
    double Delta_seR = (pow(vu, 2.0) - pow(vd, 2.0)) * gpr_sq;

    // Up-type squark mass eigenstate eigenvalues
    double m_stop_1sq = (0.5)\
        * (mQ3_sq_wk + mU3_sq_wk + (2.0 * mymtsq) + Delta_suL + Delta_suR
           - sqrt(pow((mQ3_sq_wk + Delta_suL - mU3_sq_wk - Delta_suR), 2.0)
                  + (4.0 * pow(((at_wk * vu) - (mu_wk * yt_wk * vd)), 2.0))));
    double m_stop_2sq = (0.5)\
        * (mQ3_sq_wk + mU3_sq_wk + (2.0 * mymtsq) + Delta_suL + Delta_suR
           + sqrt(pow((mQ3_sq_wk + Delta_suL - mU3_sq_wk - Delta_suR), 2.0)
                  + (4.0 * pow(((at_wk * vu) - (mu_wk * yt_wk * vd)), 2.0))));
    double m_scharm_1sq = (0.5)\
        * (mQ2_sq_wk + mU2_sq_wk + (2.0 * mymcsq) + Delta_suL + Delta_suR
           - sqrt(pow((mQ2_sq_wk + Delta_suL - mU2_sq_wk - Delta_suR), 2.0)
                  + (4.0 * pow(((ac_wk * vu) - (mu_wk * yc_wk * vd)), 2.0))));
    double m_scharm_2sq = (0.5)\
        * (mQ2_sq_wk + mU2_sq_wk + (2.0 * mymcsq) + Delta_suL + Delta_suR
           + sqrt(pow((mQ2_sq_wk + Delta_suL - mU2_sq_wk - Delta_suR), 2.0)
                  + (4.0 * pow(((ac_wk * vu) - (mu_wk * yc_wk * vd)), 2.0))));
    double m_sup_1sq = (0.5)\
        * (mQ1_sq_wk + mU1_sq_wk + (2.0 * mymusq) + Delta_suL + Delta_suR
           - sqrt(pow((mQ1_sq_wk + Delta_suL - mU1_sq_wk - Delta_suR), 2.0)
                  + (4.0 * pow(((au_wk * vu) - (mu_wk * yu_wk * vd)), 2.0))));
    double m_sup_2sq = (0.5)\
        * (mQ1_sq_wk + mU1_sq_wk + (2.0 * mymusq) + Delta_suL + Delta_suR
           + sqrt(pow((mQ1_sq_wk + Delta_suL - mU1_sq_wk - Delta_suR), 2.0)
                  + (4.0 * pow(((au_wk * vu) - (mu_wk * yu_wk * vd)), 2.0))));

    // Down-type squark mass eigenstate eigenvalues
    double m_sbot_1sq = (0.5)\
        * (mQ3_sq_wk + mD3_sq_wk + (2.0 * mymbsq) + Delta_sdL + Delta_sdR
           - sqrt(pow((mQ3_sq_wk + Delta_sdL - mD3_sq_wk - Delta_sdR), 2.0)
                  + (4.0 * pow(((ab_wk * vd) - (mu_wk * yb_wk * vu)), 2.0))));
    double m_sbot_2sq = (0.5)\
        * (mQ3_sq_wk + mD3_sq_wk + (2.0 * mymbsq) + Delta_sdL + Delta_sdR
           + sqrt(pow((mQ3_sq_wk + Delta_sdL - mD3_sq_wk - Delta_sdR), 2.0)
                  + (4.0 * pow(((ab_wk * vd) - (mu_wk * yb_wk * vu)), 2.0))));
    double m_sstrange_1sq = (0.5)\
        * (mQ2_sq_wk + mD2_sq_wk + (2.0 * mymssq) + Delta_sdL + Delta_sdR
           - sqrt(pow((mQ2_sq_wk + Delta_sdL - mD2_sq_wk - Delta_sdR), 2.0)
                  + (4.0 * pow(((as_wk * vd) - (mu_wk * ys_wk * vu)), 2.0))));
    double m_sstrange_2sq = (0.5)\
        * (mQ2_sq_wk + mD2_sq_wk + (2.0 * mymssq) + Delta_sdL + Delta_sdR
           + sqrt(pow((mQ2_sq_wk + Delta_sdL - mD2_sq_wk - Delta_sdR), 2.0)
                  + (4.0 * pow(((as_wk * vd) - (mu_wk * ys_wk * vu)), 2.0))));
    double m_sdown_1sq = (0.5)\
        * (mQ1_sq_wk + mD1_sq_wk + (2.0 * mymdsq) + Delta_sdL + Delta_sdR
           - sqrt(pow((mQ1_sq_wk + Delta_sdL - mD1_sq_wk - Delta_sdR), 2.0)
                  + (4.0 * pow(((ad_wk * vd) - (mu_wk * yd_wk * vu)), 2.0))));
    double m_sdown_2sq = (0.5)\
        * (mQ1_sq_wk + mD1_sq_wk + (2.0 * mymdsq) + Delta_sdL + Delta_sdR
           + sqrt(pow((mQ1_sq_wk + Delta_sdL - mD1_sq_wk - Delta_sdR), 2.0)
                  + (4.0 * pow(((ad_wk * vd) - (mu_wk * yd_wk * vu)), 2.0))));

    // Slepton mass eigenstate eigenvalues
    double m_stau_1sq = (0.5)\
        * (mL3_sq_wk + mE3_sq_wk + (2.0 * mymtausq) + Delta_seL + Delta_seR
           - sqrt(pow((mL3_sq_wk + Delta_seL - mE3_sq_wk - Delta_seR), 2.0)
                  + (4.0 * pow(((atau_wk * vd) - (mu_wk * ytau_wk * vu)), 2.0))));
    double m_stau_2sq = (0.5)\
        * (mL3_sq_wk + mE3_sq_wk + (2.0 * mymtausq) + Delta_seL + Delta_seR
           + sqrt(pow((mL3_sq_wk + Delta_seL - mE3_sq_wk - Delta_seR), 2.0)
                  + (4.0 * pow(((atau_wk * vd) - (mu_wk * ytau_wk * vu)), 2.0))));
    double m_smu_1sq = (0.5)\
        * (mL2_sq_wk + mE2_sq_wk + (2.0 * mymmusq) + Delta_seL + Delta_seR
           - sqrt(pow((mL2_sq_wk + Delta_seL - mE2_sq_wk - Delta_seR), 2.0)
                  + (4.0 * pow(((amu_wk * vd) - (mu_wk * ymu_wk * vu)), 2.0))));
    double m_smu_2sq = (0.5)\
        * (mL2_sq_wk + mE2_sq_wk + (2.0 * mymmusq) + Delta_seL + Delta_seR
           + sqrt(pow((mL2_sq_wk + Delta_seL - mE2_sq_wk - Delta_seR), 2.0)
                  + (4.0 * pow(((amu_wk * vd) - (mu_wk * ymu_wk * vu)), 2.0))));
    double m_se_1sq = (0.5)\
        * (mL1_sq_wk + mE1_sq_wk + (2.0 * mymesq) + Delta_seL + Delta_seR
           - sqrt(pow((mL1_sq_wk + Delta_seL - mE1_sq_wk - Delta_seR), 2.0)
                  + (4.0 * pow(((ae_wk * vd) - (mu_wk * ye_wk * vu)), 2.0))));
    double m_se_2sq = (0.5)\
        * (mL1_sq_wk + mE1_sq_wk + (2.0 * mymesq) + Delta_seL + Delta_seR
           + sqrt(pow((mL1_sq_wk + Delta_seL - mE1_sq_wk - Delta_seR), 2.0)
                  + (4.0 * pow(((ae_wk * vd) - (mu_wk * ye_wk * vu)), 2.0))));

    // Chargino mass eigenstate eigenvalues
    double msC1sq = (0.5)\
        * (pow(M2_wk, 2.0) + mu_wk_sq + (2.0 * m_w_sq)
           - sqrt(pow(pow(M2_wk, 2.0) + mu_wk_sq
                      + (2.0 * m_w_sq), 2.0)
                  - (4.0 * pow((mu_wk * M2_wk)
                               - (m_w_sq * sin2b), 2.0))));
    double msC2sq = (0.5)\
        * (pow(M2_wk, 2.0) + mu_wk_sq + (2.0 * m_w_sq)
           + sqrt(pow(pow(M2_wk, 2.0) + mu_wk_sq
                      + (2.0 * m_w_sq), 2.0)
                  - (4.0 * pow((mu_wk * M2_wk)
                               - (m_w_sq * sin2b), 2.0))));

    // Neutralino mass eigenstate eigenvalues
    Eigen::Matrix<double, 4, 4> neut_mass_mat(4, 4);
    neut_mass_mat << M1_wk, 0.0, (-1.0) * gpr_wk * vd / sqrt(2.0), gpr_wk * vu / sqrt(2.0),
                0.0, M2_wk, g2_wk * vd / sqrt(2.0), (-1.0) * g2_wk * vu / sqrt(2.0),
                (-1.0) * gpr_wk * vd / sqrt(2.0), g2_wk * vd / sqrt(2.0), 0.0, (-1.0) * mu_wk,
                gpr_wk * vu / sqrt(2.0), (-1.0) * g2_wk * vu / sqrt(2.0), (-1.0) * mu_wk, 0.0;
    Eigen::EigenSolver<Eigen::Matrix<double, 4, 4>> solver(neut_mass_mat);
    Eigen::Matrix<double, 4, 1> my_neut_mass_eigvals = solver.eigenvalues().real();
    Eigen::Matrix<double, 4, 4> my_neut_mass_eigvecs = solver.eigenvectors().real();
    Eigen::Matrix<double, 4, 1> mneutrsq = my_neut_mass_eigvals.array().square();
    sort(mneutrsq.data(), mneutrsq.data() + mneutrsq.size());

    vector<double> eigval_vector(my_neut_mass_eigvals.data(), my_neut_mass_eigvals.data() + my_neut_mass_eigvals.size());
    sort(eigval_vector.begin(), eigval_vector.end(), [](double a, double b) {
        return abs(a) < abs(b);
    });

    double msN1 = eigval_vector[0];
    double msN2 = eigval_vector[1];
    double msN3 = eigval_vector[2];
    double msN4 = eigval_vector[3];
    //cout << "msN1 = " << msN1 << "\nmsN2 = " << msN2 <<  "\nmsN3 = " << msN3 <<  "\nmsN4 = " << msN4 << endl;

    double msN1sq = mneutrsq[0];
    double msN2sq = mneutrsq[1];
    double msN3sq = mneutrsq[2];
    double msN4sq = mneutrsq[3];
    
    // Neutral Higgs doublet mass eigenstate running squared masses
    double mh0sq = (0.5)\
        * ((mA0sq) + (mz_q_sq)
           - sqrt(pow(mA0sq - mz_q_sq, 2.0) + (4.0 * mz_q_sq * mA0sq * pow(sin(2.0 * beta_wk), 2.0))));
    double mH0sq = (0.5)\
        * ((mA0sq) + (mz_q_sq)
           + sqrt(pow(mA0sq - mz_q_sq, 2.0) + (4.0 * mz_q_sq * mA0sq * pow(sin(2.0 * beta_wk), 2.0))));

    ////////// Radiative corrections in stop squark sector //////////

    const double stop_denom = m_stop_2sq - m_stop_1sq;
    const double stopuu_num = (pow(at_wk, 2.0)) - (at_wk * yt_wk * mu_wk / (tan(beta_wk)))\
        - ((1.0 / 24.0) * ((3.0 * g2_sq) - (10.0 * gpr_sq)) * (mQ3_sq_wk - mU3_sq_wk))\
        - ((1.0 / 288.0) * (pow((3.0 * g2_sq) - (10.0 * gpr_sq), 2.0) * v_sq * cos2b));
    const double stopdd_num = (yt_wk * mu_wk) * ((yt_wk * mu_wk)
                                                 - at_wk * tan(beta_wk))\
        + ((1.0 / 24.0) * ((3.0 * g2_sq) - (10.0 * gpr_sq)) * (mQ3_sq_wk - mU3_sq_wk))\
        + ((1.0 / 288.0) * (pow((3.0 * g2_sq) - (10.0 * gpr_sq), 2.0) * v_sq * cos2b));
    //std::cout << "stopuu_num = " << stopuu_num << "\t" << "stopdd_num = " << stopdd_num;
    const double sigmauu_stop_1 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_stop_1sq, pow(myQ, 2.0)) \
        * (pow(yt_wk, 2.0) - ((g2_sq + (2.0 * gpr_sq)) / 8.0)
           - (stopuu_num / stop_denom));
    const double sigmauu_stop_2 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_stop_2sq, pow(myQ, 2.0)) \
        * (pow(yt_wk, 2.0) - ((g2_sq + (2.0 * gpr_sq)) / 8.0)
           + (stopuu_num / stop_denom));
    //std::cout << std::endl << std::endl << "Sigma_u(stop_1): " << sigmauu_stop_1 << "\t" << "Sigma_u(stop_2): " << sigmauu_stop_2;
    const double sigmadd_stop_1 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_stop_1sq, pow(myQ, 2.0)) \
        * (((g2_sq + (2.0 * gpr_sq)) / 8.0) - (stopdd_num / stop_denom));
    const double sigmadd_stop_2 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_stop_2sq, pow(myQ, 2.0)) \
        * (((g2_sq + (2.0 * gpr_sq)) / 8.0) + (stopdd_num / stop_denom));

    ////////// Radiative corrections in sbottom squark sector //////////

    const double sbot_denom = m_sbot_2sq - m_sbot_1sq;
    const double sbotuu_num = (yb_wk * mu_wk) * ((yb_wk * mu_wk)
                                                 - ab_wk / tan(beta_wk))\
        + ((1.0 / 24.0) * ((3.0 * g2_sq) - (2.0 * gpr_sq)) * (mQ3_sq_wk - mD3_sq_wk))\
        - ((1.0 / 288.0) * (pow((3.0 * g2_sq) - (2.0 * gpr_sq), 2.0) * v_sq * cos2b));
    const double sbotdd_num = (pow(ab_wk, 2.0)) - (ab_wk * yb_wk * mu_wk * (tan(beta_wk)))\
        - ((1.0 / 24.0) * ((3.0 * g2_sq) - (2.0 * gpr_sq)) * (mQ3_sq_wk - mD3_sq_wk))\
        + ((1.0 / 288.0) * (pow((3.0 * g2_sq) - (2.0 * gpr_sq), 2.0) * v_sq * cos2b));
    const double sigmauu_sbot_1 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_sbot_1sq, pow(myQ, 2.0)) \
        * (((g2_sq + (2.0 * gpr_sq)) / 8.0) - (sbotuu_num / sbot_denom));
    const double sigmauu_sbot_2 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_sbot_2sq, pow(myQ, 2.0)) \
        * (((g2_sq + (2.0 * gpr_sq)) / 8.0) + (sbotuu_num / sbot_denom));
    const double sigmadd_sbot_1 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_sbot_1sq, pow(myQ, 2.0)) \
        * ((pow(yb_wk, 2.0)) - ((g2_sq + (2.0 * gpr_sq)) / 8.0)
        - (sbotdd_num / sbot_denom));
    const double sigmadd_sbot_2 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_sbot_2sq, pow(myQ, 2.0)) \
        * ((pow(yb_wk, 2.0)) - ((g2_sq + (2.0 * gpr_sq)) / 8.0)
        + (sbotdd_num / sbot_denom));

    ////////// Radiative corrections in stau slepton sector //////////
        
    const double stau_denom = m_stau_2sq - m_stau_1sq;
    const double stauuu_num = (ytau_wk * mu_wk) * ((ytau_wk * mu_wk)
                                                   - atau_wk / tan(beta_wk))\
        + ((1.0 / 8.0) * ((g2_sq) - (6.0 * gpr_sq)) * (mL3_sq_wk - mE3_sq_wk))\
        - ((1.0 / 32.0) * (pow((g2_sq) - (6.0 * gpr_sq), 2.0) * v_sq * cos2b));
    const double staudd_num = pow(atau_wk, 2.0) - (atau_wk * ytau_wk * mu_wk * tan(beta_wk))\
        - ((1.0 / 8.0) * ((g2_sq) - (6.0 * gpr_sq)) * (mL3_sq_wk - mE3_sq_wk))\
        + ((1.0 / 32.0) * (pow((g2_sq) - (6.0 * gpr_sq), 2.0) * v_sq * cos2b));
    const double sigmauu_stau_1 = ((1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_stau_1sq, pow(myQ, 2.0)) \
        * (((g2_sq + (2.0 * gpr_sq)) / 8.0) - (stauuu_num / stau_denom));
    const double sigmauu_stau_2 = ((1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_stau_2sq, pow(myQ, 2.0)) \
        * (((g2_sq + (2.0 * gpr_sq)) / 8.0) + (stauuu_num / stau_denom));
    const double sigmadd_stau_1 = ((1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_stau_1sq, pow(myQ, 2.0)) \
        * (pow(ytau_wk, 2.0) - ((g2_sq + (2.0 * gpr_sq)) / 8.0)
        - (staudd_num / stau_denom));
    const double sigmadd_stau_2 = ((1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_stau_2sq, pow(myQ, 2.0)) \
        * (pow(ytau_wk, 2.0) - ((g2_sq + (2.0 * gpr_sq)) / 8.0)
        + (staudd_num / stau_denom));
            
    // Tau sneutrino
        
    const double sigmauu_stau_sneut = ((1.0 / (16.0 * (pow(M_PI, 2.0))))/ 8.0) * ((-1.0) * (g2_sq + gpr_sq))\
        * logfunc2(mstauneutsq, pow(myQ, 2.0));
    const double sigmadd_stau_sneut = ((1.0 / (16.0 * (pow(M_PI, 2.0))))/ 8.0) * ((g2_sq + gpr_sq))\
        * logfunc2(mstauneutsq, pow(myQ, 2.0));

    ////////// Radiative corrections from 2nd generation sfermions //////////
    // Scharm sector
        
    const double schm_denom = m_scharm_2sq - m_scharm_1sq;
    const double schmuu_num = (pow(ac_wk, 2.0)) - (ac_wk * yc_wk * mu_wk / (tan(beta_wk)))\
        - ((1.0 / 24.0) * ((3.0 * g2_sq) - (10.0 * gpr_sq)) * (mQ2_sq_wk - mU2_sq_wk))\
        - ((1.0 / 288.0) * (pow((3.0 * g2_sq) - (10.0 * gpr_sq), 2.0) * v_sq * cos2b));
    const double schmdd_num = (yc_wk * mu_wk) * ((yc_wk * mu_wk)
                                                 - ac_wk * tan(beta_wk))\
        + ((1.0 / 24.0) * ((3.0 * g2_sq) - (10.0 * gpr_sq)) * (mQ2_sq_wk - mU2_sq_wk))\
        + ((1.0 / 288.0) * (pow((3.0 * g2_sq) - (10.0 * gpr_sq), 2.0) * v_sq * cos2b));
    const double sigmauu_scharm_1 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_scharm_1sq, pow(myQ, 2.0)) \
        * (pow(yc_wk, 2.0) - ((g2_sq + (2.0 * gpr_sq)) / 8.0)
           - (schmuu_num / schm_denom));
    const double sigmauu_scharm_2 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_scharm_2sq, pow(myQ, 2.0)) \
        * (pow(yc_wk, 2.0) - ((g2_sq + (2.0 * gpr_sq)) / 8.0)
        + (schmuu_num / schm_denom));
    const double sigmadd_scharm_1 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_scharm_1sq, pow(myQ, 2.0)) \
        * (((g2_sq + (2.0 * gpr_sq)) / 8.0) - (schmdd_num / schm_denom));
    const double sigmadd_scharm_2 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_scharm_2sq, pow(myQ, 2.0)) \
        * (((g2_sq + (2.0 * gpr_sq)) / 8.0) + (schmdd_num / schm_denom));

    // Sstrange sector

    const double sstr_denom = m_sstrange_2sq - m_sstrange_1sq;
    const double sstruu_num = (ys_wk * mu_wk) * ((ys_wk * mu_wk)
                                                 - as_wk / tan(beta_wk))\
        + ((1.0 / 24.0) * ((3.0 * g2_sq) - (2.0 * gpr_sq)) * (mQ2_sq_wk - mD2_sq_wk))\
        - ((1.0 / 288.0) * (pow((3.0 * g2_sq) - (2.0 * gpr_sq), 2.0) * v_sq * cos2b));
    const double sstrdd_num = pow(as_wk, 2.0) - (as_wk * ys_wk * mu_wk * tan(beta_wk))\
        - ((1.0 / 24.0) * ((3.0 * g2_sq) - (2.0 * gpr_sq)) * (mQ2_sq_wk - mD2_sq_wk))\
        + ((1.0 / 288.0) * (pow((3.0 * g2_sq) - (2.0 * gpr_sq), 2.0) * v_sq * cos2b));
    const double sigmauu_sstrange_1 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_sstrange_1sq, pow(myQ, 2.0)) \
        * (((g2_sq + (2.0 * gpr_sq)) / 8.0) - (sstruu_num / sstr_denom));
    const double sigmauu_sstrange_2 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_sstrange_2sq, pow(myQ, 2.0)) \
        * (((g2_sq + (2.0 * gpr_sq)) / 8.0) + (sstruu_num / sstr_denom));
    const double sigmadd_sstrange_1 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_sstrange_1sq, pow(myQ, 2.0)) \
        * ((pow(ys_wk, 2.0)) - ((g2_sq + (2.0 * gpr_sq)) / 8.0)
        - (sstrdd_num / sstr_denom));
    const double sigmadd_sstrange_2 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_sstrange_2sq, pow(myQ, 2.0)) \
        * ((pow(ys_wk, 2.0)) - ((g2_sq + (2.0 * gpr_sq)) / 8.0)
        + (sstrdd_num / sstr_denom));

    // Smu/smu sneutrino

    const double smu_denom = m_smu_2sq - m_smu_1sq;
    const double smuuu_num = (ymu_wk * mu_wk) * ((ymu_wk * mu_wk)
                                                 - amu_wk / tan(beta_wk))\
        + ((1.0 / 8.0) * ((g2_sq) - (6.0 * gpr_sq)) * (mL2_sq_wk - mE2_sq_wk))\
        - ((1.0 / 32.0) * (pow((g2_sq) - (6.0 * gpr_sq), 2.0) * v_sq * cos2b));
    const double smudd_num = pow(amu_wk, 2.0) - (amu_wk * ymu_wk * mu_wk * tan(beta_wk))\
        - ((1.0 / 8.0) * ((g2_sq) - (6.0 * gpr_sq)) * (mL2_sq_wk - mE2_sq_wk))\
        + ((1.0 / 32.0) * (pow((g2_sq) - (6.0 * gpr_sq), 2.0) * v_sq * cos2b));
    const double sigmauu_smu_1 = ((1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_smu_1sq, pow(myQ, 2.0)) \
        * (((g2_sq + (2.0 * gpr_sq)) / 8.0) - (smuuu_num / smu_denom));
    const double sigmauu_smu_2 = ((1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_smu_2sq, pow(myQ, 2.0)) \
        * (((g2_sq + (2.0 * gpr_sq)) / 8.0) + (smuuu_num / smu_denom));
    const double sigmadd_smu_1 = ((1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_smu_1sq, pow(myQ, 2.0)) \
        * (pow(ymu_wk, 2.0) - ((g2_sq + (2.0 * gpr_sq)) / 8.0)
        - (smudd_num / smu_denom));
    const double sigmadd_smu_2 = ((1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_smu_2sq, pow(myQ, 2.0)) \
        * (pow(ymu_wk, 2.0) - ((g2_sq + (2.0 * gpr_sq)) / 8.0)
        + (smudd_num / smu_denom));

    // Mu sneutrino
    const double sigmauu_smu_sneut = ((1.0 / (16.0 * (pow(M_PI, 2.0))))/ 8.0) * ((-1.0) * (g2_sq + gpr_sq))\
        * logfunc2(msmuneutsq, pow(myQ, 2.0));
    const double sigmadd_smu_sneut = ((1.0 / (16.0 * (pow(M_PI, 2.0))))/ 8.0) * ((g2_sq + gpr_sq))\
        * logfunc2(msmuneutsq, pow(myQ, 2.0));

    ////////// Radiative corrections from 1st generation sfermions //////////
    // Sup sector

    const double sup_denom = m_sup_2sq - m_sup_1sq;
    const double supuu_num = (pow(au_wk, 2.0)) - (au_wk * yu_wk * mu_wk / (tan(beta_wk)))\
        - ((1.0 / 24.0) * ((3.0 * g2_sq) - (10.0 * gpr_sq)) * (mQ1_sq_wk - mU1_sq_wk))\
        - ((1.0 / 288.0) * (pow((3.0 * g2_sq) - (10.0 * gpr_sq), 2.0) * v_sq * cos2b));
    const double supdd_num = (yu_wk * mu_wk) * ((yu_wk * mu_wk)
                                                - au_wk * tan(beta_wk))\
        + ((1.0 / 24.0) * ((3.0 * g2_sq) - (10.0 * gpr_sq)) * (mQ1_sq_wk - mU1_sq_wk))\
        + ((1.0 / 288.0) * (pow((3.0 * g2_sq) - (10.0 * gpr_sq), 2.0) * v_sq * cos2b));
    const double sigmauu_sup_1 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_sup_1sq, pow(myQ, 2.0)) \
        * (pow(yu_wk, 2.0) - ((g2_sq + (2.0 * gpr_sq)) / 8.0)
           - (supuu_num / sup_denom));
    const double sigmauu_sup_2 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_sup_2sq, pow(myQ, 2.0)) \
        * (pow(yu_wk, 2.0) - ((g2_sq + (2.0 * gpr_sq)) / 8.0)
           + (supuu_num / sup_denom));
    const double sigmadd_sup_1 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_sup_1sq, pow(myQ, 2.0)) \
        * (((g2_sq + (2.0 * gpr_sq)) / 8.0) - (supdd_num / sup_denom));
    const double sigmadd_sup_2 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_sup_2sq, pow(myQ, 2.0)) \
        * (((g2_sq + (2.0 * gpr_sq)) / 8.0) + (supdd_num / sup_denom));

    // Sdown sector

    const double sdwn_denom = m_sdown_2sq - m_sdown_1sq;
    const double sdwnuu_num = (yd_wk * mu_wk) * ((yd_wk * mu_wk)
                                            - ad_wk / tan(beta_wk))\
        + ((1.0 / 24.0) * ((3.0 * g2_sq) - (2.0 * gpr_sq)) * (mQ1_sq_wk - mD1_sq_wk))\
        - ((1.0 / 288.0) * (pow((3.0 * g2_sq) - (2.0 * gpr_sq), 2.0) * v_sq * cos2b));
    const double sdwndd_num = pow(ad_wk, 2.0) - (ad_wk * yd_wk * mu_wk * tan(beta_wk))\
        - ((1.0 / 24.0) * ((3.0 * g2_sq) - (2.0 * gpr_sq)) * (mQ1_sq_wk - mD1_sq_wk))\
        + ((1.0 / 288.0) * (pow((3.0 * g2_sq) - (2.0 * gpr_sq), 2.0) * v_sq * cos2b));
    double sigmauu_sdown_1 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_sdown_1sq, pow(myQ, 2.0)) \
        * (((g2_sq + (2.0 * gpr_sq)) / 8.0) - (sdwnuu_num / sdwn_denom));
    double sigmauu_sdown_2 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_sdown_2sq, pow(myQ, 2.0)) \
        * (((g2_sq + (2.0 * gpr_sq)) / 8.0) + (sdwnuu_num / sdwn_denom));
    double sigmadd_sdown_1 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_sdown_1sq, pow(myQ, 2.0)) \
        * ((pow(yd_wk, 2.0)) - ((g2_sq + (2.0 * gpr_sq)) / 8.0)
           - (sdwndd_num / sdwn_denom));
    double sigmadd_sdown_2 = (3.0 * (1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_sdown_2sq, pow(myQ, 2.0)) \
        * ((pow(yd_wk, 2.0)) - ((g2_sq + (2.0 * gpr_sq)) / 8.0)
           + (sdwndd_num / sdwn_denom));

    // Selectron/selectron sneutrino

    double sel_denom = m_se_2sq - m_se_1sq;
    double seluu_num = (ye_wk * mu_wk) * ((ye_wk * mu_wk)
                                          - ae_wk / tan(beta_wk))\
        + ((1.0 / 8.0) * ((g2_sq) - (6.0 * gpr_sq)) * (mL1_sq_wk - mE1_sq_wk))\
        - ((1.0 / 32.0) * (pow((g2_sq) - (6.0 * gpr_sq), 2.0) * v_sq * cos2b));
    double seldd_num = pow(ae_wk, 2.0) - (ae_wk * ye_wk * mu_wk * tan(beta_wk))\
        - ((1.0 / 8.0) * ((g2_sq) - (6.0 * gpr_sq)) * (mL1_sq_wk - mE1_sq_wk))\
        + ((1.0 / 32.0) * (pow((g2_sq) - (6.0 * gpr_sq), 2.0) * v_sq * cos2b));
    double sigmauu_se_1 = ((1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_se_1sq, pow(myQ, 2.0)) \
        * (((g2_sq + (2.0 * gpr_sq)) / 8.0) - (seluu_num / sel_denom));
    double sigmauu_se_2 = ((1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_se_2sq, pow(myQ, 2.0)) \
        * (((g2_sq + (2.0 * gpr_sq)) / 8.0) + (seluu_num / sel_denom));
    double sigmadd_se_1 = ((1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_se_1sq, pow(myQ, 2.0)) \
        * (pow(ye_wk, 2.0) - ((g2_sq + (2.0 * gpr_sq)) / 8.0)
        - (seldd_num / sel_denom));
    double sigmadd_se_2 = ((1.0 / (16.0 * (pow(M_PI, 2.0))))) * logfunc2(m_se_2sq, pow(myQ, 2.0)) \
        * (pow(ye_wk, 2.0) - ((g2_sq + (2.0 * gpr_sq)) / 8.0)
        + (seldd_num / sel_denom));

    // Electron sneutrino

    double sigmauu_selec_sneut = ((1.0 / (16.0 * (pow(M_PI, 2.0))))/ 8.0) * ((-1.0) * (g2_sq + gpr_sq))\
        * logfunc2(mselecneutsq, pow(myQ, 2.0));
    double sigmadd_selec_sneut = ((1.0 / (16.0 * (pow(M_PI, 2.0))))/ 8.0) * ((g2_sq + gpr_sq))\
        * logfunc2(mselecneutsq, pow(myQ, 2.0));

    ////////// Radiative corrections from chargino sector //////////
    double charginouu_num = ((2.0 * M2_wk * mu_wk / tan(beta_wk))
                             + (pow(M2_wk, 2.0) + mu_wk_sq
                             - (g2_sq * v_sq * cos2b))) * (g2_sq / 2.0);
    double charginodd_num = ((2.0 * M2_wk * mu_wk * tan(beta_wk))
                             + (pow(M2_wk, 2.0) + mu_wk_sq
                             + (g2_sq * v_sq * cos2b))) * (g2_sq / 2.0);
    double chargino_den = msC2sq - msC1sq;
    double sigmauu_chargino1 = ((-1.0) / (8.0 * (pow(M_PI, 2.0))))\
        * ((g2_sq / 2.0) - (charginouu_num / chargino_den)) * logfunc2(msC1sq, pow(myQ, 2.0));
    double sigmauu_chargino2 = ((-1.0) / (8.0 * (pow(M_PI, 2.0))))\
        * ((g2_sq / 2.0) + (charginouu_num / chargino_den)) * logfunc2(msC2sq, pow(myQ, 2.0));
    double sigmadd_chargino1 = ((-1.0) / (8.0 * (pow(M_PI, 2.0))))\
        * ((g2_sq / 2.0) - (charginodd_num / chargino_den)) * logfunc2(msC1sq, pow(myQ, 2.0));
    double sigmadd_chargino2 = ((-1.0) / (8.0 * (pow(M_PI, 2.0))))\
        * ((g2_sq / 2.0) + (charginodd_num / chargino_den)) * logfunc2(msC2sq, pow(myQ, 2.0));

    ////////// Radiative corrections from Higgs bosons sector //////////
    double higgsuu_num = (mz_q_sq + (mA0sq * (2.0 + (4.0 * cos2b) + cos(4.0 * beta_wk))))\
        * ((g2_sq + gpr_sq) / 4.0);
    double higgsdd_num = (mz_q_sq + (mA0sq * (2.0 - (4.0 * cos2b) + cos(4.0 * beta_wk))))\
        * ((g2_sq + gpr_sq) / 4.0);
    double higgs_den = (mH0sq - mh0sq);
    double sigmauu_h0 = ((1.0 / (32.0 * (pow(M_PI, 2.0))))) * logfunc2(mh0sq, pow(myQ, 2.0))\
        * (((g2_sq + gpr_sq) / 4.0) - (higgsuu_num / higgs_den));
    double sigmauu_heavy_h0 = ((1.0 / (32.0 * (pow(M_PI, 2.0))))) * logfunc2(mH0sq, pow(myQ, 2.0))\
        * (((g2_sq + gpr_sq) / 4.0) + (higgsuu_num / higgs_den));
    double sigmadd_h0 = ((1.0 / (32.0 * (pow(M_PI, 2.0))))) * logfunc2(mh0sq, pow(myQ, 2.0))\
        * (((g2_sq + gpr_sq) / 4.0) - (higgsdd_num / higgs_den));
    double sigmadd_heavy_h0 = ((1.0 / (32.0 * (pow(M_PI, 2.0))))) * logfunc2(mH0sq, pow(myQ, 2.0))\
        * (((g2_sq + gpr_sq) / 4.0) + (higgsdd_num / higgs_den));
    double sigmauu_h_pm  = (g2_sq * (1.0 / (16.0 * (pow(M_PI, 2.0)))) / 2.0) * logfunc2(mH_pmsq, pow(myQ, 2.0));
    double sigmadd_h_pm = sigmauu_h_pm;

    ////////// Radiative corrections from weak vector bosons sector //////////
    double sigmauu_w_pm = (3.0 * g2_sq * (1.0 / (16.0 * (pow(M_PI, 2.0)))) / 2.0) * logfunc2(m_w_sq, pow(myQ, 2.0));
    double sigmadd_w_pm = sigmauu_w_pm;
    double sigmauu_z0 = (3.0 / 4.0) * (1.0 / (16.0 * (pow(M_PI, 2.0)))) * (gpr_sq + g2_sq)\
        * logfunc2(mz_q_sq, pow(myQ, 2.0));
    double sigmadd_z0 = sigmauu_z0;

    ////////// Radiative corrections from SM fermions sector //////////
    double sigmauu_top = (-6.0) * pow(yt_wk, 2.0) * (1.0 / (16.0 * (pow(M_PI, 2.0))))\
        * logfunc2(mymtsq, pow(myQ, 2.0));
    double sigmadd_top = 0.0;
    double sigmauu_bottom = 0.0;
    double sigmadd_bottom = (-6.0) * pow(yb_wk, 2.0) * (1.0 / (16.0 * (pow(M_PI, 2.0))))\
        * logfunc2(mymbsq, pow(myQ, 2.0));
    double sigmauu_tau = 0.0;
    double sigmadd_tau = (-2.0) * pow(ytau_wk, 2.0) * (1.0 / (16.0 * (pow(M_PI, 2.0))))\
        * logfunc2(mymtausq, pow(myQ, 2.0));
    double sigmauu_charm = (-6.0) * pow(yc_wk, 2.0) * (1.0 / (16.0 * (pow(M_PI, 2.0))))\
        * logfunc2(mymcsq, pow(myQ, 2.0));
    double sigmadd_charm = 0.0;
    double sigmauu_strange = 0.0;
    double sigmadd_strange = (-6.0) * pow(ys_wk, 2.0) * (1.0 / (16.0 * (pow(M_PI, 2.0))))\
        * logfunc2(mymssq, pow(myQ, 2.0));
    double sigmauu_mu = 0.0;
    double sigmadd_mu = (-2.0) * pow(ymu_wk, 2.0) * (1.0 / (16.0 * (pow(M_PI, 2.0))))\
        * logfunc2(mymmusq, pow(myQ, 2.0));
    double sigmauu_up = (-6.0) * pow(yu_wk, 2.0) * (1.0 / (16.0 * (pow(M_PI, 2.0))))\
        * logfunc2(mymusq, pow(myQ, 2.0));
    double sigmadd_up = 0.0;
    double sigmauu_down = 0.0;
    double sigmadd_down = (-6.0) * pow(yd_wk, 2.0) * (1.0 / (16.0 * (pow(M_PI, 2.0))))\
        * logfunc2(mymdsq, pow(myQ, 2.0));
    double sigmauu_elec = 0.0;
    double sigmadd_elec = (-2.0) * pow(ye_wk, 2.0) * (1.0 / (16.0 * (pow(M_PI, 2.0))))\
        * logfunc2(mymesq, pow(myQ, 2.0));

    double sigmadd2l = sigmadd_2loop(myQ, mu_wk, beta_wk, yt_wk, yc_wk, yu_wk, yb_wk, ys_wk,
                                        yd_wk, ytau_wk, ymu_wk, ye_wk, g1_wk, g2_wk, g3_wk, mQ3_sq_wk,
                                        mQ2_sq_wk, mQ1_sq_wk, mL3_sq_wk, mL2_sq_wk, mL1_sq_wk,
                                        mU3_sq_wk, mU2_sq_wk, mU1_sq_wk, mD3_sq_wk, mD2_sq_wk, mD1_sq_wk,
                                        mE3_sq_wk, mE2_sq_wk, mE1_sq_wk, M1_wk, M2_wk, M3_wk, mHu_sq_wk,
                                        mHd_sq_wk, at_wk, ac_wk, au_wk, ab_wk, as_wk, ad_wk, atau_wk,
                                        amu_wk, ae_wk, m_stop_1sq, m_stop_2sq, mymt, vHiggs_wk);
    double sigmauu2l = sigmauu_2loop(myQ, mu_wk, beta_wk, yt_wk, yc_wk, yu_wk, yb_wk, ys_wk,
                                        yd_wk, ytau_wk, ymu_wk, ye_wk, g1_wk, g2_wk, g3_wk, mQ3_sq_wk,
                                        mQ2_sq_wk, mQ1_sq_wk, mL3_sq_wk, mL2_sq_wk, mL1_sq_wk,
                                        mU3_sq_wk, mU2_sq_wk, mU1_sq_wk, mD3_sq_wk, mD2_sq_wk, mD1_sq_wk,
                                        mE3_sq_wk, mE2_sq_wk, mE1_sq_wk, M1_wk, M2_wk, M3_wk, mHu_sq_wk,
                                        mHd_sq_wk, at_wk, ac_wk, au_wk, ab_wk, as_wk, ad_wk, atau_wk,
                                        amu_wk, ae_wk, m_stop_1sq, m_stop_2sq, mymt, vHiggs_wk);

    double sigmauuZ1 = sigmauu_neutralino(msN1, M1_wk, M2_wk, mu_wk, g2_sq, gpr_sq, v_sq, vu, vd, beta_wk, myQ);
    double sigmauuZ2 = sigmauu_neutralino(msN2, M1_wk, M2_wk, mu_wk, g2_sq, gpr_sq, v_sq, vu, vd, beta_wk, myQ);
    double sigmauuZ3 = sigmauu_neutralino(msN3, M1_wk, M2_wk, mu_wk, g2_sq, gpr_sq, v_sq, vu, vd, beta_wk, myQ);
    double sigmauuZ4 = sigmauu_neutralino(msN4, M1_wk, M2_wk, mu_wk, g2_sq, gpr_sq, v_sq, vu, vd, beta_wk, myQ);
    double sigmaddZ1 = sigmadd_neutralino(msN1, M1_wk, M2_wk, mu_wk, g2_sq, gpr_sq, v_sq, vu, vd, beta_wk, myQ);
    double sigmaddZ2 = sigmadd_neutralino(msN2, M1_wk, M2_wk, mu_wk, g2_sq, gpr_sq, v_sq, vu, vd, beta_wk, myQ);
    double sigmaddZ3 = sigmadd_neutralino(msN3, M1_wk, M2_wk, mu_wk, g2_sq, gpr_sq, v_sq, vu, vd, beta_wk, myQ);
    double sigmaddZ4 = sigmadd_neutralino(msN4, M1_wk, M2_wk, mu_wk, g2_sq, gpr_sq, v_sq, vu, vd, beta_wk, myQ);                     
    ////////// Total radiative corrections //////////
    // Include successful radiative corrections
    double sigmauu_tot = 0.0;
    double sigmadd_tot = 0.0;
    std::vector<double> list_of_myuus = {sigmauu_stop_1, sigmauu_stop_2, sigmauu_sbot_1,
                                            sigmauu_sbot_2, sigmauu_stau_1, sigmauu_stau_2,
                                            sigmauu_stau_sneut, sigmauu_scharm_1+sigmauu_scharm_2+sigmauu_sstrange_1+sigmauu_sstrange_2,
                                            sigmauu_smu_1, sigmauu_smu_2, sigmauu_smu_sneut, 
                                            sigmauu_sup_1+sigmauu_sup_2+sigmauu_sdown_1+sigmauu_sdown_2, sigmauu_se_1,
                                            sigmauu_se_2, sigmauu_selec_sneut,
                                            sigmauuZ1, sigmauuZ2, sigmauuZ3, sigmauuZ4,
                                            sigmauu_chargino1,
                                            sigmauu_chargino2,
                                            sigmauu_h0, sigmauu_heavy_h0, sigmauu_h_pm, sigmauu_w_pm,
                                            sigmauu_z0, sigmauu_top + sigmauu_bottom + sigmauu_tau\
                                            + sigmauu_charm + sigmauu_strange + sigmauu_mu\
                                            + sigmauu_up + sigmauu_down + sigmauu_elec,
                                            sigmauu2l};
    // cout << "Sigma_u values: " << endl;
    // for (double value : list_of_myuus) {
    //     cout << value << endl;
    // }
    std::vector<double> list_of_mydds = {sigmadd_stop_1, sigmadd_stop_2, sigmadd_sbot_1,
                                            sigmadd_sbot_2, sigmadd_stau_1, sigmadd_stau_2,
                                            sigmadd_stau_sneut, sigmadd_scharm_1+sigmadd_scharm_2+sigmadd_sstrange_1+sigmadd_sstrange_2,
                                            sigmadd_smu_1, sigmadd_smu_2, sigmadd_smu_sneut, 
                                            sigmadd_sup_1+sigmadd_sup_2+sigmadd_sdown_1+sigmadd_sdown_2, sigmadd_se_1,
                                            sigmadd_se_2, sigmadd_selec_sneut,
                                            sigmaddZ1, sigmaddZ2, sigmaddZ3, sigmaddZ4, sigmadd_chargino1, sigmadd_chargino2,
                                            sigmadd_h0, sigmadd_heavy_h0, sigmadd_h_pm, sigmadd_w_pm,
                                            sigmadd_z0, sigmadd_top + sigmadd_bottom + sigmadd_tau\
                                            + sigmadd_charm + sigmadd_strange + sigmadd_mu\
                                            + sigmadd_up + sigmadd_down + sigmadd_elec,
                                            sigmadd2l};
    // cout << "Sigma_d values: " << endl;
    // for (double value : list_of_mydds) {
    //     cout << value << endl;
    // }
    //std::cout << "Sigma_d contribs:" << std::endl;
    for (const auto& value : list_of_mydds) {
        // Check if value is not NaN
        if (!std::isnan(value)) {
            //std::cout << value << std::endl;
            sigmadd_tot += value;
        }
    }
    //std::cout << "Sigma_u contribs:" << std::endl;
    for (const auto& value : list_of_myuus) {
        // Check if value is not NaN
        if (!std::isnan(value)) {
            //std::cout << value << std::endl;
            sigmauu_tot += value;
        }
    }
    std::vector<double> listofres = {sigmauu_tot, sigmadd_tot};
    return listofres;
    
}