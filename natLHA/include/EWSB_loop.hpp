// EWSB_LOOP_HPP

#ifndef EWSB_LOOP_HPP
#define EWSB_LOOP_HPP

#include <stdexcept>
#include <vector>
#include <boost/multiprecision/mpfr.hpp>
using namespace boost::multiprecision;
typedef number<mpfr_float_backend<50>> high_prec_float;

using namespace std;

class EWSBNumericalFailure : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

namespace ewsb_detail {

std::vector<high_prec_float> loopHessianTerms(
    const std::vector<high_prec_float>& weak_boundary_conditions,
    high_prec_float myQ);

}  // namespace ewsb_detail

bool Hessian_check(vector<high_prec_float> weak_boundary_conditions, high_prec_float myQ);

bool BFB_check(vector<high_prec_float> weak_boundary_conditions);

#endif
