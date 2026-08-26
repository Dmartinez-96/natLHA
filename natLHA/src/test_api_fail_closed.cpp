#include <iostream>
#include <limits>
#include <string>

#include "natlha_api.hpp"
#include "natlha_api_detail.hpp"
#include "radcorr_calc.hpp"

namespace {

bool expect(bool condition, const std::string& message) {
    if (condition) return true;
    std::cerr << message << "\n";
    return false;
}

}  // namespace

int main() {
    natlha::Result result;
    result.ok = true;
    result.haveDEW = true;
    result.haveDHS = true;
    result.haveDBG = true;
    result.haveDSN = true;
    result.deltaEW = 1;
    result.deltaHS = 2;
    result.deltaBG = 3;
    result.deltaSN = 4;
    result.snTotalNvac = 5;
    result.dewContributions.resize(1);
    result.dhsContributions.resize(1);
    result.dbgContributions.resize(1);
    result.dsnContributions.resize(1);
    result.dbgDiagnostics.resize(1);
    result.qSusySearchDiagnostics.resize(1);

    natlha::detail::failLabelRow(result, "late requested-label failure");

    bool ok = true;
    ok &= expect(!result.ok && result.error == "late requested-label failure",
                 "label-row failure did not retain its status and diagnostic");
    ok &= expect(!result.haveDEW && !result.haveDHS
                     && !result.haveDBG && !result.haveDSN,
                 "label-row failure retained a successful have flag");
    ok &= expect(result.deltaEW == 0 && result.deltaHS == 0
                     && result.deltaBG == 0 && result.deltaSN == 0
                     && result.snTotalNvac == 0,
                 "label-row failure retained a headline or vacuum density");
    ok &= expect(result.dewContributions.empty()
                     && result.dhsContributions.empty()
                     && result.dbgContributions.empty()
                     && result.dsnContributions.empty(),
                 "label-row failure retained a partial contribution list");
    ok &= expect(result.dbgDiagnostics.size() == 1,
                 "label-row failure discarded Delta_BG diagnostics");
    ok &= expect(result.qSusySearchDiagnostics.size() == 1,
                 "label-row failure discarded Q_SUSY search diagnostics");

    natlha::Config deferredSN;
    deferredSN.slhaPath = "this-path-must-not-be-opened.slha";
    deferredSN.computeDSN = true;
    deferredSN.snMode = 2;
    const natlha::Result deferredResult = natlha::evaluate(deferredSN);
    ok &= expect(!deferredResult.ok
                     && deferredResult.error.find("capital Delta_SN continuation is deferred")
                            != std::string::npos,
                 "non-interactive API still entered a deferred Delta_SN continuation mode");

    std::vector<LabeledValueBG> invalidBG = {
        {std::numeric_limits<high_prec_float>::quiet_NaN(), "non-finite direction", 0, 0}};
    try {
        natlha::detail::requireFiniteDBGContributions(invalidBG);
        ok &= expect(false, "non-finite Delta_BG contribution was silently accepted");
    } catch (const NumericalFailure& failure) {
        ok &= expect(failure.stage == "DBG_calc contributions"
                         && failure.invalidTerms.size() == 1
                         && failure.invalidTerms[0] == "non-finite direction",
                     "non-finite Delta_BG contribution lost its diagnostic");
    }
    return ok ? 0 : 1;
}
