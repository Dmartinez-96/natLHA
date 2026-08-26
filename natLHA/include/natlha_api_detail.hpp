#ifndef NATLHA_API_DETAIL_HPP
#define NATLHA_API_DETAIL_HPP

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "MSSM_RGE_solver_with_stopfinder.hpp"
#include "natlha_api.hpp"

namespace natlha::detail {

struct EWSBTuneResult {
    std::vector<high_prec_float> state;
    std::vector<double> doubleState;
    std::vector<high_prec_float> radCorrs;
    high_prec_float relationMZ2 = 0;
    high_prec_float squaredDifference = 0;
    long iterations = 0;
};

struct JointQSusyOperations {
    std::vector<double> immutableHighState;
    std::function<std::vector<double>(const std::vector<double>&, double, double, double)>
        evolve;
    std::function<QSusyResult(const std::vector<double>&, double, double, double)> findRoot;
    std::function<EWSBTuneResult(const std::vector<double>&, const high_prec_float&)> tuneMu;
    std::function<StopScalePoint(const std::vector<double>&, double)> evaluateStop;
};

struct JointQSusySolution {
    QSusyResult root;
    EWSBTuneResult tuned;
    StopScalePoint retunedPoint;
    std::vector<QSusyIterationDiagnostic> diagnostics;
    long ewsbIterations = 0;
};

struct JointQSusyConvergenceFailure : std::runtime_error {
    JointQSusyConvergenceFailure(
        std::string message,
        std::vector<QSusyIterationDiagnostic> completedDiagnostics,
        long completedEWSBIterations);

    std::vector<QSusyIterationDiagnostic> diagnostics;
    long ewsbIterations = 0;
};

QSusyResult runAuditedQSusySearch(
    std::vector<QSusySearchDiagnostic>& diagnostics,
    const std::function<QSusyResult()>& search);

struct GutScaleOperations {
    std::function<std::vector<double>(const std::vector<double>&, double, double, double)>
        evolve;
    std::function<std::vector<high_prec_float>(const std::vector<high_prec_float>&)>
        gaugeBetas;
};

struct GutScaleSolution {
    std::vector<high_prec_float> state;
    high_prec_float logScale = 0;
    long iterations = 0;
};

JointQSusySolution solveJointQSusyMu(
    QSusyResult initialRoot,
    double highLogScale,
    double timeStep,
    double maxDeltaLogQ,
    long maxIterations,
    const JointQSusyOperations& operations);

GutScaleSolution solveGutScale(
    const std::vector<double>& weakState,
    double weakLogScale,
    double initialHighLogScale,
    double timeStep,
    long maxIterations,
    const GutScaleOperations& operations);

void requireFiniteDBGContributions(
    const std::vector<LabeledValueBG>& contributions);

}  // namespace natlha::detail

#endif  // NATLHA_API_DETAIL_HPP
