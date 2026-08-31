#ifndef NATLHA_CLI_OUTPUT_HPP
#define NATLHA_CLI_OUTPUT_HPP

#include "natlha_api.hpp"

namespace natlha_cli {

using QSusyAuditSummary = natlha::QSusyAuditSummary;

inline QSusyAuditSummary summarizeQSusyAudit(const natlha::Result& result) {
    return natlha::summarizeQSusyAudit(result);
}

inline QSusyAuditSummary summarizeQSusyAudit(
        const natlha::BatchRowResult& result) {
    return natlha::summarizeQSusyAudit(result);
}

}  // namespace natlha_cli

#endif
