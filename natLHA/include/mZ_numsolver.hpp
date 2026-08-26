// MZ_NUMSOLVER_HPP

#ifndef MZ_NUMSOLVER_HPP
#define MZ_NUMSOLVER_HPP

#include <functional>
#include <string>
#include <vector>
#include <boost/multiprecision/mpfr.hpp>

using namespace std;
using namespace boost::multiprecision;

typedef number<mpfr_float_backend<50>> high_prec_float;

enum class MZ2FailureCode {
    None,
    InvalidSeedEvaluation,
    BracketNotFound,
    DomainBoundaryUnresolved,
    RefinementBudgetExhausted,
    RootResidualTooLarge,
    AmbiguousBranch,
    PhysicalBoundaryUnresolved,
    NegativeRootNoEWSB,
    NegativeContinuedBranchWithPositiveAlternative
};

struct MZ2ResidualEvaluation {
    bool valid = true;
    high_prec_float value = 0;
    std::string stage;
    std::vector<std::string> invalidTerms;
};

struct MZ2RootCandidate {
    high_prec_float lower = 0;
    high_prec_float upper = 0;
    high_prec_float value = 0;
    high_prec_float residual = 0;
};

struct MZ2RejectedCandidate {
    MZ2RootCandidate candidate;
    MZ2FailureCode failure = MZ2FailureCode::None;
};

struct MZ2DomainBoundary {
    high_prec_float point = 0;
    std::string stage;
    std::vector<std::string> invalidTerms;
};

struct MZ2SolveOptions {
    unsigned expansionSteps = 12;
    unsigned refinementSteps = 128;
};

struct MZ2SolveResult {
    bool ok = false;
    high_prec_float value = 0;
    high_prec_float residual = 0;
    high_prec_float lower = 0;
    high_prec_float upper = 0;
    unsigned evaluations = 0;
    MZ2FailureCode failure = MZ2FailureCode::None;
    std::string diagnostic;
    std::vector<MZ2RootCandidate> candidates;
    std::vector<MZ2RejectedCandidate> rejectedCandidates;
    std::vector<MZ2DomainBoundary> invalidBoundaries;
};

using MZ2ResidualFunction =
    std::function<MZ2ResidualEvaluation(const high_prec_float&)>;

/// Search a bounded, two-sided deterministic slice and select the unique root branch whose
/// refined interval is provably nearest to `continuationSeed`.
MZ2SolveResult solveMZ2Residual(const MZ2ResidualFunction& residual,
                                high_prec_float continuationSeed,
                                const MZ2SolveOptions& options = MZ2SolveOptions());

/// Solve the loop-corrected EWSB relation for m_Z^2.  Failures are data, never an unconverged
/// numeric value: callers must check `ok` before consuming `value`.
MZ2SolveResult solveMZ2(const vector<high_prec_float>& input_weakscaleBCs,
                        high_prec_float input_QSUSY,
                        high_prec_float continuationSeed,
                        const MZ2SolveOptions& options = MZ2SolveOptions());

std::string describeMZ2Failure(const MZ2SolveResult& result);

//double gettanb(const vector<double>& input_weakscaleBCs, double input_QSUSY, double mZ2fixed, double guess);


#endif
