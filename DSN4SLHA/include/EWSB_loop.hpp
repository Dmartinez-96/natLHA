// EWSB_LOOP_HPP

#ifndef EWSB_LOOP_HPP
#define EWSB_LOOP_HPP

#include <vector>
#include <boost/multiprecision/mpfr.hpp>
using namespace boost::multiprecision;
typedef number<mpfr_float_backend<50>> high_prec_float;

using namespace std;

bool Hessian_check(vector<high_prec_float> weak_boundary_conditions, high_prec_float myQ);

bool BFB_check(vector<high_prec_float> weak_boundary_conditions);

#endif