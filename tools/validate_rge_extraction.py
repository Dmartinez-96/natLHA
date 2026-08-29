#!/usr/bin/env python3
"""Verify the generic RGE equation body against natLHA's pre-extraction source."""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys


DEFAULT_BASE = "223600d005eb73759a2bb062ec8ff16768bf81e2"
BASE_PATH = "natLHA/src/MSSM_RGE_solver.cpp"
BASE_CONSTANTS_PATH = "natLHA/include/constants.hpp"
GENERIC_PATH = "natLHA/include/MSSM_RGE_derivatives.inl"
START_MARKER = "// Extract values from the input vector x"
END_MARKER = "dxdt[43] = dtanb_dt;"


def command_output(*arguments: str) -> str:
    return subprocess.check_output(arguments, text=True)


def equation_body(source: str, identity: str) -> str:
    start = source.find(START_MARKER)
    if start < 0:
        raise ValueError(f"{identity}: missing start marker {START_MARKER!r}")
    end = source.find(END_MARKER, start)
    if end < 0:
        raise ValueError(f"{identity}: missing end marker {END_MARKER!r}")
    return source[start : end + len(END_MARKER)]


def without_comments_or_whitespace(source: str) -> str:
    source = re.sub(r"//[^\n]*", "", source)
    return re.sub(r"\s+", "", source)


def normalize_base(source: str) -> str:
    source = re.sub(r"\b(?:std::)?pow\b", "RGE_POW", source)
    return without_comments_or_whitespace(source)


def normalize_generic(source: str) -> str:
    source = re.sub(r"\bReal\b", "double", source)
    source = re.sub(r"\brgePow\b", "RGE_POW", source)
    source = source.replace("rgeLoopFactorSquared", "loop_fac_sq")
    source = source.replace("rgeLoopFactor", "loop_fac")
    return without_comments_or_whitespace(source)


def loop_factor_definitions(source: str, generic: bool, identity: str) -> str:
    if generic:
        pattern = (
            r"const\s+Real\s+rgeLoopFactor\s*=.*?;\s*"
            r"const\s+Real\s+rgeLoopFactorSquared\s*=.*?;"
        )
    else:
        pattern = (
            r"const\s+double\s+loop_fac\s*=.*?;\s*"
            r"const\s+double\s+loop_fac_sq\s*=.*?;"
        )
    match = re.search(pattern, source, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"{identity}: missing loop-factor definitions")
    return match.group(0)


def normalize_loop_factors(source: str) -> str:
    source = source.replace("rgeLoopFactorSquared", "loop_fac_sq")
    source = source.replace("rgeLoopFactor", "loop_fac")
    source = re.sub(r"\brgePow\b|\b(?:std::)?pow\b", "RGE_POW", source)
    source = re.sub(r"\bReal\(([+-]?\d+)\)", r"\1.0", source)
    source = re.sub(r"\bReal\(M_PI\)", "M_PI", source)
    source = re.sub(r"\bReal\b", "double", source)
    return without_comments_or_whitespace(source)


def power_inventory(source: str) -> tuple[int, list[str], bool, bool]:
    calls = len(re.findall(r"\brgePow\s*\(", source))
    matches = re.findall(r"rgePow\([^,\n]+,\s*([0-9.]+)\)", source)
    exponents = sorted(set(matches))
    return (
        calls,
        exponents,
        len(matches) == calls,
        set(exponents) == {"2.0", "3.0", "4.0"},
    )


def validate_instrument() -> None:
    base = """
        const double loop_fac = 1.0 / (16.0 * std::pow(M_PI, 2.0));
        const double loop_fac_sq = std::pow(loop_fac, 2.0);
    """
    generic = """
        const Real rgeLoopFactor =
            Real(1) / (Real(16) * rgePow(Real(M_PI), 2.0));
        const Real rgeLoopFactorSquared = rgePow(rgeLoopFactor, 2.0);
    """
    normalized_base = normalize_loop_factors(base)
    if normalize_loop_factors(generic) != normalized_base:
        raise RuntimeError("loop-factor instrument rejected its equivalent control")
    if normalize_loop_factors(generic.replace("Real(16)", "Real(8)")) == normalized_base:
        raise RuntimeError("loop-factor instrument accepted a changed denominator")
    bare_integers = generic.replace("Real(1)", "1").replace("Real(16)", "16")
    if normalize_loop_factors(bare_integers) == normalized_base:
        raise RuntimeError("loop-factor instrument accepted bare-integer drift")

    valid_powers = "rgePow(x, 2.0); rgePow(x, 3.0); rgePow(x, 4.0);"
    if power_inventory(valid_powers) != (
        3, ["2.0", "3.0", "4.0"], True, True
    ):
        raise RuntimeError("power inventory rejected its valid control")
    if power_inventory(valid_powers.replace("4.0", "exponent"))[2]:
        raise RuntimeError("power inventory accepted an unparsed exponent")
    if power_inventory(valid_powers.replace("4.0", "5.0"))[3]:
        raise RuntimeError("power inventory accepted an unexpected exponent")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        default=DEFAULT_BASE,
        help="git commit containing the authoritative pre-extraction CPU body",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    validate_instrument()
    repository_hint = pathlib.Path(__file__).resolve().parents[1]
    repository = pathlib.Path(
        command_output(
            "git", "-C", str(repository_hint), "rev-parse", "--show-toplevel"
        ).strip()
    )
    base_source = command_output(
        "git", "-C", str(repository), "show", f"{arguments.base}:{BASE_PATH}"
    )
    base_constants = command_output(
        "git",
        "-C",
        str(repository),
        "show",
        f"{arguments.base}:{BASE_CONSTANTS_PATH}",
    )
    generic_source = (repository / GENERIC_PATH).read_text(encoding="utf-8")

    base_body = equation_body(base_source, f"{arguments.base}:{BASE_PATH}")
    generic_body = equation_body(generic_source, GENERIC_PATH)
    normalized_base = normalize_base(base_body)
    normalized_generic = normalize_generic(generic_body)
    normalized_base_loop_factors = normalize_loop_factors(
        loop_factor_definitions(
            base_constants, False, f"{arguments.base}:{BASE_CONSTANTS_PATH}"
        )
    )
    normalized_generic_loop_factors = normalize_loop_factors(
        loop_factor_definitions(generic_source, True, GENERIC_PATH)
    )

    base_power_calls = len(re.findall(r"\b(?:std::)?pow\s*\(", base_body))
    (
        generic_power_calls,
        generic_exponents,
        exponent_coverage_complete,
        exponents_expected,
    ) = power_inventory(generic_body)
    body_equal = normalized_base == normalized_generic
    loop_factors_equal = (
        normalized_base_loop_factors == normalized_generic_loop_factors
    )
    passed = (
        body_equal
        and loop_factors_equal
        and exponent_coverage_complete
        and exponents_expected
    )

    print(f"base_commit={arguments.base}")
    print("instrument_self_test_passed=true")
    print(f"normalized_base_bytes={len(normalized_base.encode('utf-8'))}")
    print(f"normalized_generic_bytes={len(normalized_generic.encode('utf-8'))}")
    print(f"base_power_calls={base_power_calls}")
    print(f"generic_power_calls={generic_power_calls}")
    print(
        "generic_power_exponents_matched="
        f"{generic_power_calls if exponent_coverage_complete else 'incomplete'}"
    )
    print(f"generic_power_exponents={','.join(generic_exponents)}")
    print(f"body_normalized_equal={'true' if body_equal else 'false'}")
    print(f"loop_factors_equal={'true' if loop_factors_equal else 'false'}")
    print(
        "power_exponent_coverage_complete="
        f"{'true' if exponent_coverage_complete else 'false'}"
    )
    print(f"power_exponents_expected={'true' if exponents_expected else 'false'}")
    print(f"validation_passed={'true' if passed else 'false'}")
    if not body_equal:
        mismatch = next(
            (
                index
                for index, values in enumerate(zip(normalized_base, normalized_generic))
                if values[0] != values[1]
            ),
            min(len(normalized_base), len(normalized_generic)),
        )
        print(f"first_mismatch_offset={mismatch}", file=sys.stderr)
    if not loop_factors_equal:
        print("loop-factor definitions differ", file=sys.stderr)
    if not exponent_coverage_complete:
        print("not every generic power call supplied a matched exponent", file=sys.stderr)
    if not exponents_expected:
        print("generic power exponent set differs from 2.0,3.0,4.0", file=sys.stderr)
    if not passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
