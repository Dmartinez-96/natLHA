// SHARED_HELPERS_HPP
//
// Declarations for the helpers that are DEFINED in src/terminal_UI.cpp but called from more
// than one translation unit. The definition site and every caller must include this header,
// so that one declaration is the single source of truth and the compiler checks each side
// against it.
//
// A hand-copied declaration is a hazard rather than merely untidy, because the two ways it
// can drift fail differently. A wrong PARAMETER type is loud: parameter types are encoded in
// the mangled name, so the link fails with an undefined reference. A wrong RETURN type alone
// is silent: the return type is not mangled, so the program links and runs and the caller
// simply reads the wrong value. Including one declaration everywhere turns both into compile
// errors.
//
// Only the declarations live here; the definitions stay in terminal_UI.cpp.

#ifndef SHARED_HELPERS_HPP
#define SHARED_HELPERS_HPP

#include <string>
#include <vector>

#include <boost/multiprecision/mpfr.hpp>

#include "slhaea.h"

using namespace boost::multiprecision;

typedef number<mpfr_float_backend<50>> high_prec_float;

/// Beta functions for the electroweak gauge couplings, returned as {dg1/dt, dg2/dt}.
///
/// Two-loop, read from the body of `beta_g1g2` in terminal_UI.cpp: each returned slope is
/// `loop_fac * <one-loop term> + loop_fac_sq * <two-loop term>`, with one-loop coefficients
/// `b_1 = { 33.0 / 5.0, 1.0, -3.0 }`.
///
/// g1val is expected in the GUT normalization sqrt(5/3) * g', which is what the existing
/// caller supplies: natlha_api.cpp builds state entry 0 as `sqrt(5.0 / 3.0) * g_pr`, its
/// comment noting that the SLHA GAUGE block carries the unnormalised g'. This declaration
/// does not enforce that -- passing g' directly would compile and return wrong slopes.
std::vector<high_prec_float> beta_g1g2(const high_prec_float & g1val, const high_prec_float & g2val,
                                       const high_prec_float & g3val, const high_prec_float & ytval,
                                       const high_prec_float & ycval, const high_prec_float & yuval,
                                       const high_prec_float & ybval, const high_prec_float & ysval,
                                       const high_prec_float & ydval, const high_prec_float & ytauval,
                                       const high_prec_float & ymuval, const high_prec_float & yeval);

/// Renormalization scale Q of one SLHA block, parsed from a "Q= <value>" field on any of the
/// block's lines; the first match wins.
///
/// Returns 2000.0 when the block is absent or carries no Q= field. That fallback is silent,
/// so a caller that needs to know whether the scale was really present has to check the block
/// itself -- a returned 2000.0 is indistinguishable from a file that genuinely says Q= 2000.
double getRenormalizationScale(const SLHAea::Coll & slha, const std::string & blockName);

#endif
