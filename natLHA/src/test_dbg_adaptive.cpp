#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "DBG_calc.hpp"

namespace {

bool expect(bool condition, const std::string& message) {
    if (condition) return true;
    std::cerr << message << "\n";
    return false;
}

BGNodeDiagnostic solvedNode(const high_prec_float& shift,
                            const high_prec_float& value,
                            const high_prec_float& width = 0) {
    BGNodeDiagnostic node;
    node.shift = shift;
    node.boundaryStatesDistinct = true;
    node.root.ok = true;
    node.root.value = value;
    node.root.lower = value - width / 2;
    node.root.upper = value + width / 2;
    return node;
}

dbg_detail::BGNodePairEvaluator analyticPair(
        const std::function<high_prec_float(const high_prec_float&)>& function,
        const high_prec_float& width = 0) {
    return [function, width](const high_prec_float& magnitude, unsigned level) {
        BGNodeDiagnostic negative = solvedNode(-magnitude, function(-magnitude), width);
        negative.level = level;
        BGNodeDiagnostic positive = solvedNode(magnitude, function(magnitude), width);
        positive.level = level;
        return std::make_pair(negative, positive);
    };
}

}  // namespace

int main() {
    bool ok = true;
    const high_prec_float denominator = 10;
    dbg_detail::BGDirection direction{
        "linear", dbg_detail::BGShiftKind::Plain, {3}, high_prec_float(2), 0};
    ok &= expect(!dbg_detail::usesAdaptiveTwoPoint(1)
                     && !dbg_detail::usesAdaptiveTwoPoint(2)
                     && dbg_detail::usesAdaptiveTwoPoint(3),
                 "production precision dispatch no longer selects adaptive mode only for 3");

    const BGDirectionDiagnostic linear = dbg_detail::adaptiveTwoPointDirection(
        direction, denominator,
        analyticPair([](const high_prec_float& shift) {
            return high_prec_float(1000) + high_prec_float(5) * shift;
        }));
    ok &= expect(linear.accepted && linear.windows.size() == 1
                     && linear.nodes.size() == 6
                     && abs(linear.contribution - high_prec_float(1))
                         < high_prec_float("1e-35"),
                 "adaptive two-point did not accept the exact linear sensitivity");
    ok &= expect(linear.windows[0].contributions.size() == 3
                     && linear.windows[0].contributionSpan < high_prec_float("1e-35"),
                 "accepted linear window did not retain all three agreeing estimates");

    const BGDirectionDiagnostic quadratic = dbg_detail::adaptiveTwoPointDirection(
        direction, denominator,
        analyticPair([](const high_prec_float& shift) {
            return high_prec_float(1000) + high_prec_float(5) * shift
                + high_prec_float(17) * shift * shift;
        }));
    ok &= expect(quadratic.accepted
                     && abs(quadratic.contribution - high_prec_float(1))
                         < high_prec_float("1e-30"),
                 "central two-point windows did not cancel an analytic quadratic term");

    const high_prec_float h0 = dbg_detail::doubleDomainStep(direction.value);
    const high_prec_float prefactor = direction.value / denominator;
    const high_prec_float absoluteBoundaryCubic =
        high_prec_float("0.999") / (prefactor * high_prec_float(15) * h0 * h0);
    const BGDirectionDiagnostic absoluteBoundary = dbg_detail::adaptiveTwoPointDirection(
        direction, denominator,
        analyticPair([absoluteBoundaryCubic](const high_prec_float& shift) {
            return high_prec_float("1e6")
                + absoluteBoundaryCubic * shift * shift * shift;
        }));
    const BGDirectionDiagnostic aboveAbsoluteBoundary = dbg_detail::adaptiveTwoPointDirection(
        direction, denominator,
        analyticPair([absoluteBoundaryCubic](const high_prec_float& shift) {
            return high_prec_float("1e6")
                + (high_prec_float("1.001") / high_prec_float("0.999"))
                    * absoluteBoundaryCubic
                    * shift * shift * shift;
        }));
    ok &= expect(absoluteBoundary.accepted && !aboveAbsoluteBoundary.accepted,
                 "absolute one-naturalness-unit agreement gate did not distinguish its two sides");

    const high_prec_float relativeSlope = high_prec_float(5000);
    const high_prec_float relativeBoundaryCubic =
        high_prec_float(5) / (prefactor * high_prec_float(15) * h0 * h0);
    const BGDirectionDiagnostic relativeBoundary = dbg_detail::adaptiveTwoPointDirection(
        direction, denominator,
        analyticPair([relativeSlope, relativeBoundaryCubic](const high_prec_float& shift) {
            return high_prec_float("1e6") + relativeSlope * shift
                + relativeBoundaryCubic * shift * shift * shift;
        }));
    ok &= expect(relativeBoundary.accepted
                     && relativeBoundary.windows[0].agreementTolerance
                         > high_prec_float(1),
                 "relative 0.5% agreement branch was not exercised or accepted");

    const BGDirectionDiagnostic exactZero = dbg_detail::adaptiveTwoPointDirection(
        direction, denominator,
        analyticPair([](const high_prec_float&) { return high_prec_float(1000); }));
    ok &= expect(exactZero.accepted && exactZero.contribution == 0,
                 "an exact zero failed the adaptive state/root/agreement gates");

    const BGDirectionDiagnostic noisyCubic = dbg_detail::adaptiveTwoPointDirection(
        direction, denominator,
        analyticPair([](const high_prec_float& shift) {
            return high_prec_float("1e12") + high_prec_float("1e12") * shift * shift * shift;
        }));
    ok &= expect(!noisyCubic.accepted && noisyCubic.windows.size() == 15
                     && noisyCubic.nodes.size() == 34,
                 "outward adaptive search did not honor its 15-window/34-node bound");

    const BGDirectionDiagnostic uncertainZero = dbg_detail::adaptiveTwoPointDirection(
        direction, denominator,
        analyticPair([](const high_prec_float&) { return high_prec_float("1e9"); },
                     high_prec_float("1e6")));
    ok &= expect(!uncertainZero.accepted && !uncertainZero.windows.empty()
                     && uncertainZero.windows[0].failure.find("root uncertainty")
                         != std::string::npos,
                 "an apparent zero bypassed the separate root-uncertainty gate");

    const auto invalidFirstPair = [](const high_prec_float& magnitude, unsigned level) {
        if (level == 0) {
            BGNodeDiagnostic negative;
            negative.shift = -magnitude;
            negative.failure = "analytic boundary";
            BGNodeDiagnostic positive = negative;
            positive.shift = magnitude;
            return std::make_pair(negative, positive);
        }
        return analyticPair([](const high_prec_float& shift) {
            return high_prec_float(1000) + high_prec_float(5) * shift;
        })(magnitude, level);
    };
    const BGDirectionDiagnostic outwardRecovery = dbg_detail::adaptiveTwoPointDirection(
        direction, denominator, invalidFirstPair);
    ok &= expect(outwardRecovery.accepted && outwardRecovery.windows.size() == 2
                     && outwardRecovery.windows[0].accepted == false
                     && outwardRecovery.windows[1].accepted
                     && outwardRecovery.nodes.size() == 8,
                 "an invalid inner level prevented a valid outward window or defeated caching");

    const high_prec_float infinity = std::numeric_limits<high_prec_float>::infinity();
    const BGDirectionDiagnostic nonFinite = dbg_detail::adaptiveTwoPointDirection(
        direction, denominator,
        analyticPair([infinity](const high_prec_float&) { return infinity; }));
    ok &= expect(!nonFinite.accepted,
                 "non-finite adaptive roots were accepted");

    std::vector<high_prec_float> boundary(44, high_prec_float(1));
    boundary[3] = 5;
    boundary[7] = 2;
    boundary[16] = 10;
    boundary[6] = 4;
    boundary[42] = 20;
    boundary[25] = -9;
    std::vector<high_prec_float> shifted;
    std::string failure;
    ok &= expect(dbg_detail::applyShift(
                     {"plain", dbg_detail::BGShiftKind::Plain, {3}, 5, 0},
                     2, boundary, shifted, failure)
                     && shifted[3] == 7,
                 "plain BG shift changed its coordinate transform");
    ok &= expect(dbg_detail::applyShift(
                     {"scalar", dbg_detail::BGShiftKind::Scalar, {25}, 3, 0},
                     1, boundary, shifted, failure)
                     && shifted[25] == -16,
                 "negative scalar mass did not shift through positive sqrt(|m2|)");
    ok &= expect(dbg_detail::applyShift(
                     {"trilinear", dbg_detail::BGShiftKind::Trilinear, {16}, 5, 0},
                     1, boundary, shifted, failure)
                     && shifted[16] == 12,
                 "trilinear ratio shift changed its coordinate transform");
    ok &= expect(dbg_detail::applyShift(
                     {"bilinear", dbg_detail::BGShiftKind::Bilinear, {42}, 5, 0},
                     1, boundary, shifted, failure)
                     && shifted[42] == 24,
                 "bilinear ratio shift changed its coordinate transform");

    for (std::size_t i = 0; i < boundary.size(); ++i) {
        if (boundary[i] == 0) boundary[i] = 1;
    }
    boundary[25] = -9;
    boundary[26] = -16;
    const std::vector<dbg_detail::BGDirection> model6 =
        dbg_detail::buildDirections(6, boundary);
    ok &= expect(model6.size() == 31,
                 "model 6 did not expose all 31 pMSSM-30-plus-mu directions");
    ok &= expect(model6[0].label == "Delta_BG(mHu)" && model6[0].value == 3
                     && model6[1].label == "Delta_BG(mHd)" && model6[1].value == 4,
                 "negative Higgs squared masses retained signed-root BG prefactors");
    for (std::size_t i = 0; i < model6.size(); ++i) {
        ok &= expect(model6[i].ordinal == i,
                     "model direction ordinals are not fixed by construction order");
    }

    const std::vector<LabeledValueBG> tied = dbg_detail::orderContributions({
        {high_prec_float(5), "later-positive", 2, high_prec_float("0.1")},
        {high_prec_float(-5), "first-negative", 0, high_prec_float("0.1")},
        {high_prec_float(4), "third", 1, high_prec_float("0.1")}});
    const BGHeadlineDiagnostic tiedHeadline = dbg_detail::makeHeadlineDiagnostic(tied);
    ok &= expect(tied[0].label == "first-negative" && tied[0].value == -5
                     && tiedHeadline.tiedDirectionOrdinals
                         == std::vector<std::size_t>({0, 2})
                     && tiedHeadline.headlineSignFragileRootUncertainty,
                 "equal magnitudes did not select the lowest ordinal signed headline");

    const BGHeadlineDiagnostic stableOpposite = dbg_detail::makeHeadlineDiagnostic(
        dbg_detail::orderContributions({
            {high_prec_float(-5), "top", 0, high_prec_float("0.1")},
            {high_prec_float("4.7"), "second", 1, high_prec_float("0.1")}}));
    const BGHeadlineDiagnostic fragileOpposite = dbg_detail::makeHeadlineDiagnostic(
        dbg_detail::orderContributions({
            {high_prec_float(-5), "top", 0, high_prec_float("0.1")},
            {high_prec_float("4.8"), "second", 1, high_prec_float("0.1")}}));
    ok &= expect(!stableOpposite.headlineSignFragileRootUncertainty
                     && fragileOpposite.headlineSignFragileRootUncertainty,
                 "headline sign-fragility flag did not respect the uncertainty-sum boundary");

    return ok ? 0 : 1;
}
