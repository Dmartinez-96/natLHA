#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <sstream>
#include <vector>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/math/tools/toms748_solve.hpp>
#include "radcorr_calc.hpp"
#include "mZ_numsolver.hpp"
#include <boost/multiprecision/mpfr.hpp>

using namespace std;
using namespace boost::multiprecision;

// Define high precision floating point type with 50 decimal digits of precision
typedef number<mpfr_float_backend<50>> high_prec_float;

namespace {

struct Sample {
    high_prec_float point;
    MZ2ResidualEvaluation evaluation;
};

struct Bracket {
    Sample lower;
    Sample upper;
};

struct InvalidResidualDuringRefinement {};

bool finite(const high_prec_float& value) {
    return (boost::math::isfinite)(value);
}

high_prec_float acceptedWidth(const high_prec_float& midpoint) {
    return max(high_prec_float("1e-8"),
               high_prec_float("1e-12") * max(abs(midpoint), high_prec_float(1)));
}

high_prec_float acceptedResidual(const high_prec_float& continuationSeed) {
    return high_prec_float("1e-20")
        * max(abs(continuationSeed), high_prec_float(1));
}

high_prec_float intervalMinDistance(const MZ2RootCandidate& candidate,
                                    const high_prec_float& seed) {
    if (seed < candidate.lower) return candidate.lower - seed;
    if (seed > candidate.upper) return seed - candidate.upper;
    return 0;
}

high_prec_float intervalMaxDistance(const MZ2RootCandidate& candidate,
                                    const high_prec_float& seed) {
    return max(abs(candidate.lower - seed), abs(candidate.upper - seed));
}

const char* failureName(MZ2FailureCode failure) {
    switch (failure) {
        case MZ2FailureCode::None: return "none";
        case MZ2FailureCode::InvalidSeedEvaluation: return "invalid_seed_evaluation";
        case MZ2FailureCode::BracketNotFound: return "bracket_not_found";
        case MZ2FailureCode::DomainBoundaryUnresolved: return "domain_boundary_unresolved";
        case MZ2FailureCode::RefinementBudgetExhausted: return "refinement_budget_exhausted";
        case MZ2FailureCode::RootResidualTooLarge: return "root_residual_too_large";
        case MZ2FailureCode::AmbiguousBranch: return "ambiguous_branch";
        case MZ2FailureCode::PhysicalBoundaryUnresolved: return "physical_boundary_unresolved";
        case MZ2FailureCode::NegativeRootNoEWSB: return "negative_root_no_ewsb";
        case MZ2FailureCode::NegativeContinuedBranchWithPositiveAlternative:
            return "negative_continued_branch_with_positive_alternative";
    }
    return "unknown";
}

}  // namespace

MZ2SolveResult solveMZ2Residual(const MZ2ResidualFunction& residual,
                                high_prec_float continuationSeed,
                                const MZ2SolveOptions& options) {
    MZ2SolveResult result;
    auto evaluate = [&](const high_prec_float& point) {
        ++result.evaluations;
        MZ2ResidualEvaluation evaluation;
        try {
            evaluation = residual(point);
        } catch (const NumericalFailure& failure) {
            evaluation.valid = false;
            evaluation.stage = failure.stage;
            evaluation.invalidTerms = failure.invalidTerms;
        }
        if (evaluation.valid && !finite(evaluation.value)) {
            evaluation.valid = false;
            evaluation.stage = "mZ2 residual";
            evaluation.invalidTerms = {"non-finite residual"};
        }
        if (!evaluation.valid) {
            result.invalidBoundaries.push_back(
                {point, evaluation.stage, evaluation.invalidTerms});
        }
        return Sample{point, evaluation};
    };

    const Sample seed = evaluate(continuationSeed);
    if (!seed.evaluation.valid) {
        result.failure = MZ2FailureCode::InvalidSeedEvaluation;
        result.diagnostic = "the continuation seed has an invalid radiative-correction evaluation";
        return result;
    }

    std::vector<Bracket> brackets;
    auto addBracket = [&](Sample a, Sample b) {
        if (b.point < a.point) std::swap(a, b);
        for (const auto& existing : brackets) {
            if (existing.lower.point == a.point && existing.upper.point == b.point) return;
        }
        brackets.push_back({a, b});
    };
    if (seed.evaluation.value == 0) addBracket(seed, seed);

    std::optional<Sample> previousLeft = seed;
    std::optional<Sample> previousRight = seed;
    high_prec_float radius = max(high_prec_float(1), abs(seed.evaluation.value));
    radius = max(radius, abs(continuationSeed) / high_prec_float(100));
    for (unsigned step = 0; step < options.expansionSteps; ++step) {
        const Sample left = evaluate(continuationSeed - radius);
        if (left.evaluation.valid) {
            if (left.evaluation.value == 0) addBracket(left, left);
            if (previousLeft && previousLeft->evaluation.valid
                    && left.evaluation.value != 0
                    && previousLeft->evaluation.value != 0
                    && ((left.evaluation.value < 0) != (previousLeft->evaluation.value < 0))) {
                addBracket(left, *previousLeft);
            }
            previousLeft = left;
        } else {
            previousLeft.reset();
        }

        const Sample right = evaluate(continuationSeed + radius);
        if (right.evaluation.valid) {
            if (right.evaluation.value == 0) addBracket(right, right);
            if (previousRight && previousRight->evaluation.valid
                    && right.evaluation.value != 0
                    && previousRight->evaluation.value != 0
                    && ((right.evaluation.value < 0) != (previousRight->evaluation.value < 0))) {
                addBracket(*previousRight, right);
            }
            previousRight = right;
        } else {
            previousRight.reset();
        }
        radius *= 2;
    }

    bool refinementBudgetFailed = false;
    bool refinementDomainFailed = false;
    bool refinementResidualFailed = false;
    for (const auto& bracket : brackets) {
        if (bracket.lower.point == bracket.upper.point) {
            result.candidates.push_back({bracket.lower.point, bracket.upper.point,
                                         bracket.lower.point, 0});
            continue;
        }
        boost::uintmax_t iterations = options.refinementSteps;
        try {
            auto callback = [&](const high_prec_float& point) {
                const Sample sample = evaluate(point);
                if (!sample.evaluation.valid) throw InvalidResidualDuringRefinement{};
                return sample.evaluation.value;
            };
            auto tolerance = [&](const high_prec_float& lower, const high_prec_float& upper) {
                const high_prec_float midpoint = (lower + upper) / 2;
                const high_prec_float refinementTarget =
                    acceptedResidual(continuationSeed) / high_prec_float(100);
                return abs(upper - lower) <= min(acceptedWidth(midpoint), refinementTarget);
            };
            const auto refined = boost::math::tools::toms748_solve(
                callback, bracket.lower.point, bracket.upper.point,
                bracket.lower.evaluation.value, bracket.upper.evaluation.value,
                tolerance, iterations);
            const high_prec_float lower = refined.first;
            const high_prec_float upper = refined.second;
            const high_prec_float midpoint = (lower + upper) / 2;
            const high_prec_float requiredWidth = min(
                acceptedWidth(midpoint),
                acceptedResidual(continuationSeed) / high_prec_float(100));
            if (abs(upper - lower) > requiredWidth) {
                refinementBudgetFailed = true;
                result.rejectedCandidates.push_back(
                    {{lower, upper, midpoint, 0},
                     MZ2FailureCode::RefinementBudgetExhausted});
                continue;
            }
            const Sample midpointSample = evaluate(midpoint);
            if (!midpointSample.evaluation.valid) {
                refinementDomainFailed = true;
                continue;
            }
            if (abs(midpointSample.evaluation.value)
                    > acceptedResidual(continuationSeed)) {
                refinementResidualFailed = true;
                result.rejectedCandidates.push_back(
                    {{lower, upper, midpoint, midpointSample.evaluation.value},
                     MZ2FailureCode::RootResidualTooLarge});
                continue;
            }
            result.candidates.push_back(
                {lower, upper, midpoint, midpointSample.evaluation.value});
        } catch (const InvalidResidualDuringRefinement&) {
            refinementDomainFailed = true;
        }
    }

    if (refinementBudgetFailed) {
        result.failure = MZ2FailureCode::RefinementBudgetExhausted;
        result.diagnostic = "TOMS748 did not meet the root-width gate within its budget";
        return result;
    }
    if (refinementDomainFailed) {
        result.failure = MZ2FailureCode::DomainBoundaryUnresolved;
        result.diagnostic = "an invalid radiative-correction evaluation interrupted root refinement";
        return result;
    }
    if (refinementResidualFailed) {
        result.failure = MZ2FailureCode::RootResidualTooLarge;
        result.diagnostic = "a refined candidate did not meet the scaled residual gate";
        return result;
    }
    if (result.candidates.empty()) {
        if (!result.invalidBoundaries.empty()) {
            result.failure = MZ2FailureCode::DomainBoundaryUnresolved;
            result.diagnostic = "invalid radiative-correction evaluations leave the bounded root search unresolved";
        } else {
            result.failure = MZ2FailureCode::BracketNotFound;
            result.diagnostic = "the bounded two-sided search found no root bracket";
        }
        return result;
    }

    std::optional<std::size_t> selected;
    for (std::size_t i = 0; i < result.candidates.size(); ++i) {
        bool strictlyNearest = true;
        const high_prec_float maximum = intervalMaxDistance(result.candidates[i], continuationSeed);
        for (std::size_t j = 0; j < result.candidates.size(); ++j) {
            if (i == j) continue;
            if (!(maximum < intervalMinDistance(result.candidates[j], continuationSeed))) {
                strictlyNearest = false;
                break;
            }
        }
        if (strictlyNearest) {
            if (selected) {
                selected.reset();
                break;
            }
            selected = i;
        }
    }
    if (!selected) {
        result.failure = MZ2FailureCode::AmbiguousBranch;
        result.diagnostic = "no candidate root interval is provably nearest to the continuation seed";
        return result;
    }

    const MZ2RootCandidate& root = result.candidates[*selected];
    result.value = root.value;
    result.residual = root.residual;
    result.lower = root.lower;
    result.upper = root.upper;
    if (root.lower <= 0 && root.upper >= 0) {
        result.failure = MZ2FailureCode::PhysicalBoundaryUnresolved;
        result.diagnostic = "the continued root interval intersects mZ^2 = 0";
        return result;
    }
    if (root.upper < 0) {
        bool positiveAlternative = false;
        bool physicalBoundaryAlternative = false;
        for (const auto& candidate : result.candidates) {
            if (candidate.lower > 0) positiveAlternative = true;
            if (candidate.lower <= 0 && candidate.upper >= 0) {
                physicalBoundaryAlternative = true;
            }
        }
        if (positiveAlternative) {
            result.failure = MZ2FailureCode::NegativeContinuedBranchWithPositiveAlternative;
            result.diagnostic = "the continued branch is negative while a positive root branch exists";
        } else if (physicalBoundaryAlternative) {
            result.failure = MZ2FailureCode::PhysicalBoundaryUnresolved;
            result.diagnostic = "a competing root interval intersects mZ^2 = 0";
        } else if (!result.invalidBoundaries.empty()) {
            result.failure = MZ2FailureCode::DomainBoundaryUnresolved;
            result.diagnostic = "a non-finite boundary prevents proving that no positive root exists";
        } else {
            result.failure = MZ2FailureCode::NegativeRootNoEWSB;
            result.diagnostic = "the bounded search found roots only at negative mZ^2";
        }
        return result;
    }

    result.ok = true;
    result.failure = MZ2FailureCode::None;
    result.diagnostic.clear();
    return result;
}

MZ2SolveResult solveMZ2(const vector<high_prec_float>& input_weakscaleBCs,
                        high_prec_float input_QSUSY,
                        high_prec_float continuationSeed,
                        const MZ2SolveOptions& options) {
    const MZ2ResidualFunction residual =
        [input_weakscaleBCs, input_QSUSY](const high_prec_float& mZ2) {
            const vector<high_prec_float> rc =
                radcorr_calc(input_weakscaleBCs, input_QSUSY, mZ2);
            const high_prec_float tanbSq =
                pow(input_weakscaleBCs[43], high_prec_float(2));
            const high_prec_float rhs =
                high_prec_float(2)
                    * ((input_weakscaleBCs[26] + rc[1]
                        - ((input_weakscaleBCs[25] + rc[0]) * tanbSq))
                       / (tanbSq - high_prec_float(1)))
                - high_prec_float(2) * pow(input_weakscaleBCs[6], high_prec_float(2));
            return MZ2ResidualEvaluation{true, mZ2 - rhs, {}, {}};
        };
    return solveMZ2Residual(residual, continuationSeed, options);
}

std::string describeMZ2Failure(const MZ2SolveResult& result) {
    std::ostringstream out;
    out << failureName(result.failure);
    if (!result.diagnostic.empty()) out << ": " << result.diagnostic;
    out << " (evaluations=" << result.evaluations
        << ", candidates=" << result.candidates.size()
        << ", rejected_candidates=" << result.rejectedCandidates.size()
        << ", invalid_boundaries=" << result.invalidBoundaries.size();
    if (!result.rejectedCandidates.empty()) {
        const MZ2RootCandidate& rejected = result.rejectedCandidates.front().candidate;
        out << ", first_rejected_interval=[" << rejected.lower << ", "
            << rejected.upper << "]"
            << ", first_rejected_residual=" << rejected.residual;
    }
    out << ")";
    return out.str();
}

// double gettanb(const vector<double>& input_weakscaleBCs, double input_QSUSY, double mZ2fixed, double guess) {
//     double current_tanb = guess;
//     double prev_f_x = std::numeric_limits<double>::max();
//     //double h = 1.0e-3;//boost::math::float_next(current_mZ2) - current_mZ2; // Small step for derivative approximation
//     vector<double> RadCorrs, RadCorrs_plush, RadCorrs_minush;
//     int number_of_steps_done = 0;
//     double lambda = 0.5; // Damping factor to address oscillations
//     double least_Sq_Tol = 1.0e-4;
//     while (number_of_steps_done < 25000) {
//         //cout << "mZ^2 currently = " << current_mZ2 << endl;
//         RadCorrs = radcorr_calc(input_weakscaleBCs, input_QSUSY, mZ2fixed);
//         double f_x = current_tanb - tan(0.5 * (M_PI - asin((high_prec_float(2.0) * input_weakscaleBCs[42] / (input_weakscaleBCs[25] + input_weakscaleBCs[26] + (high_prec_float(2.0) * pow(input_weakscaleBCs[6], high_prec_float(2.0))) + RadCorrs[0] + RadCorrs[1])))));
//         double h = std::cbrt(3.0 * (boost::math::float_next(abs(current_tanb)) - abs(current_tanb)));//std::cbrt(std::numeric_limits<double>::epsilon()) * std::max(1.0, std::abs(current_mZ2)); // Step size for numerical derivative
//         //cout << "Derivative step size: " << h << endl;
//         // Approximate derivative (f'(x)) with respect to mZ2
//         double tanb_plus_h = current_tanb + h;//abs(pow(3.0 * (boost::math::float_next(current_mZ2) - current_mZ2), 1.0 / 3.0));
//         RadCorrs_plush = radcorr_calc(input_weakscaleBCs, input_QSUSY, mZ2fixed);
//         double f_x_plus_h = tanb_plus_h - tan(0.5 * (M_PI - asin((high_prec_float(2.0) * input_weakscaleBCs[42] / (input_weakscaleBCs[25] + input_weakscaleBCs[26] + (high_prec_float(2.0) * pow(input_weakscaleBCs[6], high_prec_float(2.0))) + RadCorrs_plush[0] + RadCorrs_plush[1])))));
//         double tanb_minus_h = current_tanb - h;//abs(pow(3.0 * (current_mZ2 - boost::math::float_prior(current_mZ2)), 1.0 / 3.0));
//         RadCorrs_minush = radcorr_calc(input_weakscaleBCs, input_QSUSY, mZ2fixed);
//         double f_x_minus_h = tanb_minus_h - tan(0.5 * (M_PI - asin((high_prec_float(2.0) * input_weakscaleBCs[42] / (input_weakscaleBCs[25] + input_weakscaleBCs[26] + (high_prec_float(2.0) * pow(input_weakscaleBCs[6], high_prec_float(2.0))) + RadCorrs_minush[0] + RadCorrs_minush[1])))));
        
//         double df_dx = (f_x_plus_h - f_x_minus_h) / (high_prec_float(2.0) * h);

//         // Check for division by zero or extremely small derivative
//         if (fabs(df_dx) < least_Sq_Tol) {
//             //cerr << "Derivative is too small, stopping iteration." << endl;
//             break;
//         }
//         //cout << "Current f_x: " << f_x << endl;
//         //cout << "Current mZ^2: " << current_mZ2 << endl;
//         // Newton's update step
//         double deltaX = lambda * f_x / df_dx;

//         // Adjust lambda based on the behavior
//         if (fabs(f_x) >= fabs(prev_f_x)) { // No progress or oscillation
//             lambda *= 0.8; // Reduce step size
//         } else if (lambda < 1.0) { // Smooth convergence
//             lambda += 0.1; // Try increasing lambda cautiously
//             lambda = min(lambda, 1.0); // Limit lambda to 1.0
//         }

//         // Update for next iteration
//         prev_f_x = f_x;
//         current_tanb -= deltaX;
//         // if (number_of_steps_done % 100 == 0) {
//         //     cout << "Current mZ^2: " << current_mZ2 << endl;
//         // }

//         // Check for convergence
//         if (fabs(deltaX) < least_Sq_Tol) {
//             //cout << "Converged in " << number_of_steps_done + 1 << " iterations." << endl;
//             break;
//         }
//         number_of_steps_done++;
//     }

//     // if (number_of_steps_done == 100000) {
//     //     //cerr << "Ran out of iteration attempts to converge mZ^2, results may be inaccurate" << endl;
//     // }
//     return current_tanb;
// }
