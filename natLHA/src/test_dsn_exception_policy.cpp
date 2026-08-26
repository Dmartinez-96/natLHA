#include <exception>
#include <iostream>
#include <stdexcept>

#include "DSN_calc.hpp"
#include "EWSB_loop.hpp"
#include "radcorr_calc.hpp"

namespace {

template <typename ExpectedException>
bool propagates(const std::exception_ptr& failure) {
    try {
        dsn_detail::propagateRequiredNumericalFailure(failure);
    } catch (const ExpectedException&) {
        return true;
    } catch (...) {
    }
    return false;
}

bool returnsNormally(const std::exception_ptr& failure) {
    try {
        dsn_detail::propagateRequiredNumericalFailure(failure);
    } catch (...) {
        return false;
    }
    return true;
}

}  // namespace

int main() {
    const auto radiativeFailure = std::make_exception_ptr(
        NumericalFailure("delta_SN test", {"non-finite correction"}));
    const auto ewsbFailure = std::make_exception_ptr(
        EWSBNumericalFailure("non-finite Hessian"));
    const auto recoverableFailure = std::make_exception_ptr(
        std::runtime_error("legacy recoverable iteration failure"));

    bool ok = true;
    if (!propagates<NumericalFailure>(radiativeFailure)) {
        std::cerr << "delta_SN swallowed a radiative numerical failure\n";
        ok = false;
    }
    if (!propagates<EWSBNumericalFailure>(ewsbFailure)) {
        std::cerr << "delta_SN swallowed an EWSB numerical failure\n";
        ok = false;
    }
    if (!returnsNormally(recoverableFailure)) {
        std::cerr << "delta_SN changed the legacy recoverable-exception policy\n";
        ok = false;
    }
    if (!returnsNormally(nullptr)) {
        std::cerr << "delta_SN exception policy rejected an empty exception pointer\n";
        ok = false;
    }
    return ok ? 0 : 1;
}
