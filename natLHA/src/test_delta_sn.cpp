#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "DSN_calc.hpp"

namespace {

bool closeEnough(const high_prec_float& actual, const high_prec_float& expected) {
    const high_prec_float expectedAbs = abs(expected);
    const high_prec_float scale = expectedAbs > 1.0 ? expectedAbs : high_prec_float(1.0);
    return abs(actual - expected) <= high_prec_float("1e-40") * scale;
}

std::vector<high_prec_float> knownState() {
    std::vector<high_prec_float> state(43, 0.0);
    state[3] = 1.0;
    state[4] = -2.0;
    state[5] = 3.0;
    for (int i = 16; i <= 24; ++i) {
        const int magnitude = i - 12;
        state[i] = (i % 2 == 0) ? high_prec_float(magnitude) : high_prec_float(-magnitude);
    }
    for (int i = 25; i <= 41; ++i) {
        const int magnitude = i - 12;
        const high_prec_float squared = high_prec_float(magnitude * magnitude);
        state[i] = (i % 2 == 0) ? squared : -squared;
    }
    state[6] = 5.0;
    state[42] = -150.0;
    return state;
}

std::map<std::string, high_prec_float> byLabel(
        const std::vector<DSNLabeledValue>& contributions) {
    std::map<std::string, high_prec_float> result;
    for (const auto& item : contributions) result[item.label] = item.value;
    return result;
}

}  // namespace

int main() {
    std::vector<high_prec_float> state = knownState();
    high_prec_float mZ2 = 0.0;
    high_prec_float logQSusy = 0.0;
    high_prec_float logQGut = 0.0;
    int nF = 2;
    int nD = 3;

    const std::vector<DSNLabeledValue> contributions =
        DSN_calc(3, state, mZ2, logQSusy, logQGut, nF, nD);
    if (contributions.size() != 30) {
        std::cerr << "expected 30 contributions, got " << contributions.size() << "\n";
        return 1;
    }

    static const char* labels[] = {
        "M1", "M2", "M3", "a_t", "a_c", "a_u", "a_b", "a_s", "a_d",
        "a_tau", "a_mu", "a_e", "mHu", "mHd", "mQ1", "mQ2", "mQ3",
        "mL1", "mL2", "mL3", "mU1", "mU2", "mU3", "mD1", "mD2",
        "mD3", "mE1", "mE2", "mE3", "B"
    };
    // Fixed independently with Python Decimal at 80 digits from dissertation Eq. 5.21 for
    // |p_j| = 1..30, |mu| = 5, nF = 2 and nD = 3. These values do not reuse the
    // production helper's algebra.
    const high_prec_float expectedTotal(
        "0.059804486032578940201139064244253814611031718831980314022318225878578082841877292");
    const high_prec_float expectedM1(
        "0.00012861179791952460258309476181559960131404670716554906241358758253457652224059633");
    const high_prec_float expectedB(
        "0.0038583539375857380774928428544679880394214012149664718724076274760372956672178899");

    const auto actual = byLabel(contributions);
    if (actual.size() != 30 || !closeEnough(actual.at("M1"), expectedM1)
            || !closeEnough(actual.at("B"), expectedB)) {
        std::cerr << "pinned Eq. 5.21 contributions do not match\n";
        return 1;
    }
    high_prec_float actualTotal = 0.0;
    for (const auto& item : contributions) actualTotal += item.value;
    if (!closeEnough(actualTotal, expectedTotal)) {
        std::cerr << "dN_vac sum does not match pinned Eq. 5.21 oracle\n";
        return 1;
    }
    for (int i = 0; i < 30; ++i) {
        if (actual.find(labels[i]) == actual.end()) {
            std::cerr << "missing contribution label " << labels[i] << "\n";
            return 1;
        }
    }

    std::vector<high_prec_float> signFlipped = state;
    for (int i = 3; i <= 5; ++i) signFlipped[i] *= -1.0;
    for (int i = 16; i <= 42; ++i) signFlipped[i] *= -1.0;
    const auto flipped = byLabel(DSN_calc(3, signFlipped, mZ2, logQSusy, logQGut, nF, nD));
    if (flipped.size() != actual.size()) {
        std::cerr << "sign-flipped state changed contribution count\n";
        return 1;
    }
    for (const auto& item : actual) {
        if (!closeEnough(flipped.at(item.first), item.second)) {
            std::cerr << "sign invariance failed for " << item.first << "\n";
            return 1;
        }
    }

    std::vector<high_prec_float> zeroMu = state;
    zeroMu[6] = 0.0;
    if (!DSN_calc(3, zeroMu, mZ2, logQSusy, logQGut, nF, nD).empty()) {
        std::cerr << "mu=0 should produce no mode-3 density\n";
        return 1;
    }

    std::vector<high_prec_float> zeroSoft(43, 0.0);
    zeroSoft[6] = 5.0;
    if (!DSN_calc(3, zeroSoft, mZ2, logQSusy, logQGut, nF, nD).empty()) {
        std::cerr << "all-soft-zero state should produce no mode-3 density\n";
        return 1;
    }

    std::vector<high_prec_float> tooShort(42, 0.0);
    tooShort[6] = 5.0;
    if (!DSN_calc(3, tooShort, mZ2, logQSusy, logQGut, nF, nD).empty()) {
        std::cerr << "short state should produce no mode-3 density\n";
        return 1;
    }

    bool rejectedMode4 = false;
    try {
        DSN_calc(4, state, mZ2, logQSusy, logQGut, nF, nD);
    } catch (const std::invalid_argument&) {
        rejectedMode4 = true;
    }
    if (!rejectedMode4) {
        std::cerr << "mode 4 was not rejected\n";
        return 1;
    }

    return 0;
}
