#ifndef NATLHA_CLI_OUTPUT_HPP
#define NATLHA_CLI_OUTPUT_HPP

#include <cstddef>

#include "natlha_api.hpp"

namespace natlha_cli {

struct QSusyAuditSummary {
    bool allAccepted = false;
    bool allCountsKnown = false;
    std::size_t searches = 0;
    bool haveLastRootCount = false;
    std::size_t lastRootsFound = 0;
    bool haveAcceptedLogScale = false;
    double acceptedLogScale = 0.0;
};

inline QSusyAuditSummary summarizeQSusyAudit(const natlha::Result& result) {
    QSusyAuditSummary summary;
    summary.searches = result.qSusySearchDiagnostics.size();
    if (result.qSusySearchDiagnostics.empty()) return summary;

    summary.allAccepted = true;
    summary.allCountsKnown = true;
    for (const auto& diagnostic : result.qSusySearchDiagnostics) {
        summary.allAccepted = summary.allAccepted && diagnostic.accepted;
        summary.allCountsKnown = summary.allCountsKnown && diagnostic.scanComplete;
    }

    const auto& last = result.qSusySearchDiagnostics.back();
    if (last.scanComplete) {
        summary.haveLastRootCount = true;
        summary.lastRootsFound = last.rootsFound;
    }
    if (last.accepted) {
        summary.haveAcceptedLogScale = true;
        summary.acceptedLogScale = last.logScale;
    }
    return summary;
}

}  // namespace natlha_cli

#endif
