#include <iostream>
#include <limits>
#include <string>

#include "mZ_numsolver.hpp"

namespace {

MZ2ResidualEvaluation valid(const high_prec_float& value) {
    return {true, value, {}, {}};
}

bool expect(bool condition, const std::string& message) {
    if (condition) return true;
    std::cerr << message << "\n";
    return false;
}

}  // namespace

int main() {
    bool ok = true;

    const MZ2SolveResult simple = solveMZ2Residual(
        [](const high_prec_float& x) { return valid(x - 4); }, 5);
    ok &= expect(simple.ok && abs(simple.value - 4) < high_prec_float("1e-18"),
                 "simple positive root was not recovered");
    ok &= expect(abs(simple.residual) <= high_prec_float("5e-20"),
                 "simple root missed the scaled residual gate");

    const MZ2SolveResult continuous = solveMZ2Residual(
        [](const high_prec_float& x) { return valid((x - 2) * (x - 20)); }, 3);
    ok &= expect(continuous.ok && abs(continuous.value - 2) < high_prec_float("1e-18"),
                 "nearest continuous root branch was not selected: "
                     + describeMZ2Failure(continuous));
    ok &= expect(continuous.candidates.size() == 2,
                 "bounded search did not retain both analytic root candidates: "
                     + std::to_string(continuous.candidates.size()));

    const MZ2SolveResult ambiguous = solveMZ2Residual(
        [](const high_prec_float& x) { return valid((x + 2) * (x - 2)); }, 0);
    ok &= expect(!ambiguous.ok && ambiguous.failure == MZ2FailureCode::AmbiguousBranch,
                 "equal-distance branches were not rejected as ambiguous: "
                     + describeMZ2Failure(ambiguous));

    const MZ2SolveResult negativeOnly = solveMZ2Residual(
        [](const high_prec_float& x) { return valid(x + 2); }, -1);
    ok &= expect(!negativeOnly.ok
                     && negativeOnly.failure == MZ2FailureCode::NegativeRootNoEWSB,
                 "negative-only bounded search was not classified as no EWSB");

    const MZ2SolveResult negativeWithPositive = solveMZ2Residual(
        [](const high_prec_float& x) { return valid((x + 2) * (x - 10)); }, -1);
    ok &= expect(!negativeWithPositive.ok
                     && negativeWithPositive.failure
                         == MZ2FailureCode::NegativeContinuedBranchWithPositiveAlternative,
                 "negative continued branch with a positive alternative was misclassified: "
                     + describeMZ2Failure(negativeWithPositive));

    const MZ2SolveResult boundary = solveMZ2Residual(
        [](const high_prec_float& x) {
            if (x > 5) {
                return MZ2ResidualEvaluation{false, 0, "analytic boundary", {"pole"}};
            }
            return valid(high_prec_float(-1));
        }, 0);
    ok &= expect(!boundary.ok
                     && boundary.failure == MZ2FailureCode::DomainBoundaryUnresolved
                     && !boundary.invalidBoundaries.empty(),
                 "non-finite exploratory boundary was not recorded and propagated");

    const MZ2SolveResult invalidSeed = solveMZ2Residual(
        [](const high_prec_float&) {
            return MZ2ResidualEvaluation{false, 0, "seed", {"NaN"}};
        }, 1);
    ok &= expect(!invalidSeed.ok
                     && invalidSeed.failure == MZ2FailureCode::InvalidSeedEvaluation,
                 "invalid continuation seed did not fail closed");

    const MZ2SolveResult physicalBoundary = solveMZ2Residual(
        [](const high_prec_float& x) { return valid(x); }, 1);
    ok &= expect(!physicalBoundary.ok
                     && physicalBoundary.failure
                         == MZ2FailureCode::PhysicalBoundaryUnresolved,
                 "a root at mZ2 = 0 was not rejected as a physical boundary");

    const high_prec_float infinity = std::numeric_limits<high_prec_float>::infinity();
    const MZ2SolveResult nonFiniteExploration = solveMZ2Residual(
        [infinity](const high_prec_float& x) {
            if (x > 5) return valid(infinity);
            return valid(x - 4);
        }, 5);
    ok &= expect(nonFiniteExploration.ok
                     && abs(nonFiniteExploration.value - 4) < high_prec_float("1e-18")
                     && !nonFiniteExploration.invalidBoundaries.empty(),
                 "a non-finite exploratory point was not recorded while valid regions continued");

    const MZ2SolveResult noBracket = solveMZ2Residual(
        [](const high_prec_float&) { return valid(high_prec_float(1)); }, 0);
    ok &= expect(!noBracket.ok
                     && noBracket.failure == MZ2FailureCode::BracketNotFound,
                 "a finite bounded search without a sign change was misclassified");

    const MZ2SolveResult residualTooLarge = solveMZ2Residual(
        [](const high_prec_float& x) {
            return valid(x < 4 ? high_prec_float(-1) : high_prec_float(1));
        }, 5);
    ok &= expect(!residualTooLarge.ok
                     && residualTooLarge.failure == MZ2FailureCode::RootResidualTooLarge
                     && residualTooLarge.rejectedCandidates.size() == 1
                     && residualTooLarge.rejectedCandidates.front().failure
                         == MZ2FailureCode::RootResidualTooLarge
                     && abs(residualTooLarge.rejectedCandidates.front().candidate.residual)
                         == 1,
                 "a discontinuous sign change was accepted without satisfying the residual gate");

    const MZ2SolveResult competingPhysicalBoundary = solveMZ2Residual(
        [](const high_prec_float& x) { return valid(x * (x + 2)); },
        high_prec_float("-1.5"));
    ok &= expect(!competingPhysicalBoundary.ok
                     && competingPhysicalBoundary.failure
                         == MZ2FailureCode::PhysicalBoundaryUnresolved,
                 "a zero-straddling competing branch allowed a false negative-only diagnosis");

    const MZ2SolveResult positiveAndPhysicalAlternatives = solveMZ2Residual(
        [](const high_prec_float& x) { return valid(x * (x + 2) * (x - 10)); },
        high_prec_float("-1.5"));
    ok &= expect(!positiveAndPhysicalAlternatives.ok
                     && positiveAndPhysicalAlternatives.failure
                         == MZ2FailureCode::NegativeContinuedBranchWithPositiveAlternative,
                 "a zero-straddling competitor masked a proven positive alternative");

    MZ2SolveOptions shortRefinement;
    shortRefinement.refinementSteps = 3;
    const MZ2SolveResult exhausted = solveMZ2Residual(
        [](const high_prec_float& x) { return valid(exp(x) - 2); }, 0,
        shortRefinement);
    ok &= expect(!exhausted.ok
                     && exhausted.failure == MZ2FailureCode::RefinementBudgetExhausted,
                 "refinement budget exhaustion was not distinguished");

    const MZ2SolveResult partial = solveMZ2Residual(
        [](const high_prec_float& x) { return valid((x - 2) * (x - 20)); }, 3,
        shortRefinement);
    ok &= expect(!partial.ok
                     && partial.failure == MZ2FailureCode::RefinementBudgetExhausted,
                 "a resolved root incorrectly masked another unresolved candidate bracket");

    return ok ? 0 : 1;
}
