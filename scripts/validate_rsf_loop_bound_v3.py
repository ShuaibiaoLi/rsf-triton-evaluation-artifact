#!/usr/bin/env python3
"""CPU-only semantic gate for the registered C3 fixed-replay loop-bound test.

This checks only the nine frozen SRC-03 cases. It is not a GPU correctness or
performance result, and a 22-iteration bound must not be generalized to other
RSF inputs without deriving and checking their actual iteration counts.
"""

from __future__ import annotations

import csv
import importlib.util
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RSF = ROOT / "BW1000_test/rsf-triton-gate"
INPUT = RSF / "test-data/generated/src03-aging-rsf-newton-fp64-input-v1.tsv"
REFERENCE = RSF / "aging_rsf_fp64_reference.py"
SIZES = (1, 9, 65, 257, 4099, 65537)
EXPECTED_COUNTS = (8, 22, 6, 4, 4, 6, 6, 1, 6)


def load_reference():
    spec = importlib.util.spec_from_file_location("aging_rsf_fp64_reference_v3", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def replay(row: dict[str, str], count: int, converged: bool, loop_limit: int):
    get = lambda name: float(row[name])
    tau, sigma, state = (get(name) for name in (
        "shear_traction", "effective_normal_stress", "state_reference"))
    a, b, f0, v0, length, eta, previous = (get(name) for name in (
        "direct_effect_a", "evolution_effect_b", "reference_friction",
        "reference_velocity", "state_evolution_distance", "shear_impedance",
        "previous_velocity"))
    velocity = max(previous, 1.0e-45)
    coefficient = 0.5 / v0 * math.exp((f0 + b * math.log(v0 * state / length)) / a)
    inv_eta = 1.0 / eta
    for iteration in range(loop_limit):
        argument = velocity * coefficient
        square_root = math.sqrt(argument * argument + 1.0)
        friction = a * math.log(argument + square_root)
        derivative = a * coefficient / square_root
        residual = (tau - sigma * friction) * inv_eta - velocity
        active = iteration < count
        last = iteration + 1 == count
        if active and (not last or not converged):
            derivative_g = -(sigma * inv_eta) * derivative - 1.0
            velocity = max(velocity - residual / derivative_g, 1.0e-45)
    argument = velocity * coefficient
    friction = a * math.log(argument + math.sqrt(argument * argument + 1.0))
    residual = (tau - sigma * friction) * inv_eta - velocity
    return velocity, friction, residual, count, int(converged)


def main() -> None:
    reference = load_reference()
    with INPUT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 9:
        raise SystemExit(f"FAIL expected nine frozen cases, found {len(rows)}")

    cases = []
    for index, row in enumerate(rows):
        case = reference.CASES[index]
        if row["case_id"] != case.case_id:
            raise SystemExit(f"FAIL case order at {index}")
        for field in reference.fields(case):
            if field.name == "case_id":
                continue
            from_tsv = int(row[field.name]) if field.name in (
                "state_updates", "max_newton_iterations") else float(row[field.name])
            if from_tsv != getattr(case, field.name):
                raise SystemExit(f"FAIL TSV/reference mismatch: {case.case_id}.{field.name}")
        velocity, friction, residual, count, converged = reference.solve_velocity(
            case.state_reference, case.previous_velocity, case)
        if count != EXPECTED_COUNTS[index]:
            raise SystemExit(f"FAIL registered iteration count for {case.case_id}: {count}")
        truth = (velocity, friction, residual, count, int(converged))
        for bound in (60, 22):
            got = replay(row, count, converged, bound)
            if max(abs(a - b) for a, b in zip(got[:3], truth[:3])) > 1.0e-10 or got[3:] != truth[3:]:
                raise SystemExit(f"FAIL {case.case_id} at bound={bound}: {got} != {truth}")
        cases.append((row, count, converged))

    # A too-short loop is deliberately invalid for the 22-step case.
    row, count, converged = cases[1]
    if max(abs(a - b) for a, b in zip(replay(row, count, converged, 8)[:3],
                                      replay(row, count, converged, 60)[:3])) <= 1.0e-10:
        raise SystemExit("FAIL insufficient-bound negative control did not diverge")

    for size in SIZES:
        for offset in range(size):
            row, count, converged = cases[offset % len(cases)]
            left = replay(row, count, converged, 60)
            right = replay(row, count, converged, 22)
            if max(abs(a - b) for a, b in zip(left[:3], right[:3])) > 1.0e-10 or left[3:] != right[3:]:
                raise SystemExit(f"FAIL size={size}, offset={offset}")
        print(f"PASS N={size}: 60 and 22 preserve five output fields")
    print("PASS registered count histogram: 1, 4, 6, 8, 22; bound=8 negative control differs")


if __name__ == "__main__":
    main()
