// DBG_CALC_HPP

#ifndef DBG_CALC_HPP
#define DBG_CALC_HPP

#include <cstddef>
#include <functional>
#include <utility>
#include <vector>
#include <string>

#include "mZ_numsolver.hpp"

struct LabeledValueBG {
    high_prec_float value;
    std::string label;
    std::size_t ordinal = 0;
    high_prec_float rootUncertainty = 0;
};

struct BGNodeDiagnostic {
    high_prec_float shift = 0;
    unsigned level = 0;
    bool boundaryStatesDistinct = false;
    MZ2SolveResult root;
    std::string failure;
};

struct BGWindowDiagnostic {
    unsigned firstLevel = 0;
    high_prec_float h = 0;
    std::vector<high_prec_float> contributions;
    std::vector<high_prec_float> rootUncertainties;
    high_prec_float agreementTolerance = 0;
    high_prec_float contributionSpan = 0;
    bool accepted = false;
    std::string failure;
};

struct BGDirectionDiagnostic {
    std::size_t ordinal = 0;
    std::string label;
    std::vector<BGNodeDiagnostic> nodes;
    std::vector<BGWindowDiagnostic> windows;
    bool accepted = false;
    high_prec_float contribution = 0;
    high_prec_float rootUncertainty = 0;
    high_prec_float acceptedH = 0;
    std::string failure;
};

struct BGHeadlineDiagnostic {
    std::string topLabel;
    high_prec_float topValue = 0;
    high_prec_float topRootUncertainty = 0;
    std::string secondLabel;
    high_prec_float secondValue = 0;
    high_prec_float secondRootUncertainty = 0;
    high_prec_float headlineMagnitudeGap = 0;
    bool headlineSignFragileRootUncertainty = false;
    std::vector<std::size_t> tiedDirectionOrdinals;
};

struct BGResult {
    bool ok = false;
    std::vector<LabeledValueBG> contributions;
    std::vector<BGDirectionDiagnostic> directions;
    BGHeadlineDiagnostic headline;
    std::string failure;
};

namespace dbg_detail {

enum class BGShiftKind {
    Plain,
    Scalar,
    Trilinear,
    Bilinear
};

struct BGDirection {
    std::string label;
    BGShiftKind kind = BGShiftKind::Plain;
    std::vector<int> shiftIndices;
    high_prec_float value = 0;
    std::size_t ordinal = 0;
};

using BGNodePairEvaluator = std::function<std::pair<BGNodeDiagnostic, BGNodeDiagnostic>(
    const high_prec_float&, unsigned)>;

high_prec_float doubleDomainStep(const high_prec_float& coordinate);

bool usesAdaptiveTwoPoint(int precision);

std::vector<BGDirection> buildDirections(
    int modelIndex, const std::vector<high_prec_float>& gutBoundaryConditions);

bool applyShift(const BGDirection& direction, const high_prec_float& shift,
                const std::vector<high_prec_float>& input,
                std::vector<high_prec_float>& shifted, std::string& failure);

BGDirectionDiagnostic adaptiveTwoPointDirection(
    const BGDirection& direction, const high_prec_float& mZSquared,
    const BGNodePairEvaluator& evaluatePair);

std::vector<LabeledValueBG> orderContributions(
    const std::vector<LabeledValueBG>& contributions);

BGHeadlineDiagnostic makeHeadlineDiagnostic(
    const std::vector<LabeledValueBG>& orderedContributions);

}  // namespace dbg_detail

BGResult DBG_calc(int& modselno, int& precselno,
                  high_prec_float GUT_SCALE, high_prec_float myweakscale,
                  high_prec_float inptanbval,
                  std::vector<high_prec_float> GUT_boundary_conditions,
                  high_prec_float originalmZ2value);

#endif
