#!/usr/bin/env python3
"""Independent FP64 reference for the Paper One Aging-RSF feasibility gate.

This is deliberately a scalar, standard-library-only implementation.  It is
not a performance baseline and it does not import SeisSol or Triton.  Python's
``float`` is used as the host IEEE-754 binary64 reference type.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable


ALMOST_ZERO_FP64 = 1.0e-45
DEFAULT_NEWTON_TOLERANCE = 1.0e-8
DEFAULT_MAX_NEWTON_ITERATIONS = 60
DEFAULT_STATE_UPDATES = 2


@dataclass(frozen=True)
class AgingRsfInput:
    case_id: str
    shear_traction: float
    effective_normal_stress: float
    state_reference: float
    direct_effect_a: float
    evolution_effect_b: float
    reference_friction: float
    reference_velocity: float
    state_evolution_distance: float
    shear_impedance: float
    previous_velocity: float
    time_step: float
    state_updates: int = DEFAULT_STATE_UPDATES
    newton_tolerance: float = DEFAULT_NEWTON_TOLERANCE
    max_newton_iterations: int = DEFAULT_MAX_NEWTON_ITERATIONS


@dataclass(frozen=True)
class AgingRsfOutput:
    case_id: str
    velocity: float
    friction: float
    state: float
    last_newton_residual_g: float
    post_state_update_residual_g: float
    total_newton_iterations: int
    max_inner_iterations: int
    converged: int
    finite: int


CASES = (
    AgingRsfInput("steady_reference", 8.0e6, 1.0e7, 0.20, 0.010, 0.015, 0.60,
                  1.0e-6, 0.020, 4.0e6, 1.0e-6, 1.0e-3),
    AgingRsfInput("near_zero_previous_velocity", 5.8e6, 1.0e7, 2.0e4, 0.012,
                  0.018, 0.60, 1.0e-6, 0.020, 4.0e6, 0.0, 1.0e-3),
    AgingRsfInput("low_traction", 1.0e5, 1.0e7, 2.0e4, 0.012, 0.018, 0.60,
                  1.0e-6, 0.020, 4.0e6, 1.0e-12, 1.0e-3),
    AgingRsfInput("high_velocity", 1.2e7, 1.0e7, 0.02, 0.008, 0.012, 0.58,
                  1.0e-6, 0.010, 2.0e6, 1.0, 2.0e-3),
    AgingRsfInput("weak_normal_stress", 2.0e4, 1.0e3, 2.0e4, 0.010, 0.015,
                  0.60, 1.0e-6, 0.020, 4.0e6, 1.0e-6, 1.0e-3),
    AgingRsfInput("small_direct_effect", 7.0e6, 1.0e7, 0.20, 0.004, 0.018,
                  0.60, 1.0e-6, 0.010, 3.0e6, 1.0e-5, 5.0e-4),
    AgingRsfInput("long_time_step", 8.0e6, 1.0e7, 0.20, 0.010, 0.015, 0.60,
                  1.0e-6, 0.020, 4.0e6, 1.0e-4, 0.25),
    AgingRsfInput("forced_nonconvergence", 8.0e6, 1.0e7, 0.20, 0.010, 0.015,
                  0.60, 1.0e-6, 0.020, 4.0e6, 1.0e-45, 1.0e-3,
                  max_newton_iterations=1),
    AgingRsfInput("small_length_scale", 6.5e6, 9.0e6, 0.05, 0.009, 0.014,
                  0.59, 1.0e-6, 1.0e-4, 3.5e6, 1.0e-3, 2.0e-4),
)


def _require_finite_positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")


def validate_input(case: AgingRsfInput) -> None:
    for name in ("effective_normal_stress", "state_reference", "direct_effect_a",
                 "reference_velocity", "state_evolution_distance", "shear_impedance",
                 "newton_tolerance"):
        _require_finite_positive(name, float(getattr(case, name)))
    if not math.isfinite(case.shear_traction) or case.shear_traction < 0.0:
        raise ValueError("shear_traction must be finite and >= 0")
    if not math.isfinite(case.evolution_effect_b) or case.evolution_effect_b < 0.0:
        raise ValueError("evolution_effect_b must be finite and >= 0")
    if not math.isfinite(case.reference_friction):
        raise ValueError("reference_friction must be finite")
    if not math.isfinite(case.previous_velocity) or case.previous_velocity < 0.0:
        raise ValueError("previous_velocity must be finite and >= 0")
    if not math.isfinite(case.time_step) or case.time_step < 0.0:
        raise ValueError("time_step must be finite and >= 0")
    if case.state_updates <= 0 or case.max_newton_iterations <= 0:
        raise ValueError("iteration counts are outside the supported domain")


def friction_and_derivative(velocity: float, state: float, case: AgingRsfInput) -> tuple[float, float]:
    """Return Aging-RSF friction coefficient and d(mu)/dV in binary64."""
    velocity = max(ALMOST_ZERO_FP64, velocity)
    _require_finite_positive("state", state)
    log_state = math.log(case.reference_velocity * state / case.state_evolution_distance)
    exponent = (case.reference_friction + case.evolution_effect_b * log_state) / case.direct_effect_a
    coefficient = 0.5 / case.reference_velocity * math.exp(exponent)
    argument = velocity * coefficient
    friction = case.direct_effect_a * math.asinh(argument)
    derivative = case.direct_effect_a * coefficient / math.sqrt(argument * argument + 1.0)
    return friction, derivative


def update_aging_state(state_reference: float, velocity: float, time_step: float,
                       state_evolution_distance: float) -> float:
    """Analytic Aging-law update using expm1 to avoid small-x cancellation."""
    velocity = max(ALMOST_ZERO_FP64, velocity)
    x = velocity * time_step / state_evolution_distance
    decay = math.exp(-x)
    source = time_step if x == 0.0 else time_step * (-math.expm1(-x)) / x
    return state_reference * decay + source


def solve_velocity(state: float, initial_velocity: float,
                   case: AgingRsfInput) -> tuple[float, float, float, int, bool]:
    """SeisSol-aligned Newton solve at a fixed state.

    g(V) = (tau - sigma*mu(V, psi))/eta - V.  Convergence uses the
    absolute |g| tolerance from SeisSol, not the project's older traction-
    scaled residual convention.
    """
    velocity = max(ALMOST_ZERO_FP64, initial_velocity)
    for iteration in range(1, case.max_newton_iterations + 1):
        friction, derivative = friction_and_derivative(velocity, state, case)
        residual_g = ((case.shear_traction - case.effective_normal_stress * friction)
                      / case.shear_impedance - velocity)
        if abs(residual_g) < case.newton_tolerance:
            return velocity, friction, residual_g, iteration, True
        derivative_g = -(case.effective_normal_stress / case.shear_impedance) * derivative - 1.0
        velocity = max(ALMOST_ZERO_FP64, velocity - residual_g / derivative_g)
    friction, _ = friction_and_derivative(velocity, state, case)
    residual_g = ((case.shear_traction - case.effective_normal_stress * friction)
                  / case.shear_impedance - velocity)
    return velocity, friction, residual_g, case.max_newton_iterations, False


def coupled_aging_step(case: AgingRsfInput) -> AgingRsfOutput:
    """Execute the isolated SeisSol-style state/Newton coupling for one step."""
    validate_input(case)
    velocity = max(ALMOST_ZERO_FP64, case.previous_velocity)
    mean_velocity = velocity
    state = case.state_reference
    converged = True
    total_iterations = 0
    max_inner_iterations = 0
    last_newton_residual_g = math.nan

    for _ in range(case.state_updates):
        state = update_aging_state(case.state_reference, mean_velocity, case.time_step,
                                   case.state_evolution_distance)
        old_velocity = velocity
        velocity, _, last_newton_residual_g, iterations, inner_converged = solve_velocity(
            state, velocity, case)
        total_iterations += iterations
        max_inner_iterations = max(max_inner_iterations, iterations)
        converged = converged and inner_converged
        mean_velocity = 0.5 * (old_velocity + abs(velocity))

    state = update_aging_state(case.state_reference, mean_velocity, case.time_step,
                               case.state_evolution_distance)
    friction, _ = friction_and_derivative(velocity, state, case)
    post_state_update_residual_g = (
        (case.shear_traction - case.effective_normal_stress * friction)
        / case.shear_impedance - velocity)
    finite = all(math.isfinite(x) for x in
                 (velocity, friction, state, last_newton_residual_g,
                  post_state_update_residual_g))
    return AgingRsfOutput(case.case_id, velocity, friction, state,
                          last_newton_residual_g, post_state_update_residual_g,
                          total_iterations, max_inner_iterations,
                          int(converged), int(finite))


def write_dataset(output_dir: Path, cases: Iterable[AgingRsfInput] = CASES) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "src03-aging-rsf-newton-fp64-input-v1.tsv"
    golden_path = output_dir / "src03-aging-rsf-newton-fp64-golden-v1.tsv"
    case_list = tuple(cases)
    outputs = tuple(coupled_aging_step(case) for case in case_list)
    with input_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[field.name for field in fields(AgingRsfInput)],
                                delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for case in case_list:
            writer.writerow(case.__dict__)
    with golden_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[field.name for field in fields(AgingRsfOutput)],
                                delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for result in outputs:
            writer.writerow(result.__dict__)


def self_test() -> None:
    outputs = tuple(coupled_aging_step(case) for case in CASES)
    if not all(result.finite for result in outputs):
        raise AssertionError("all reference cases must remain finite")
    forced = next(result for result in outputs if result.case_id == "forced_nonconvergence")
    if forced.converged != 0:
        raise AssertionError("the forced non-convergence case unexpectedly converged")
    ordinary = [result for result in outputs if result.case_id != "forced_nonconvergence"]
    if not all(result.converged for result in ordinary):
        raise AssertionError("an ordinary reference case failed to converge")
    if not all(abs(result.last_newton_residual_g) < DEFAULT_NEWTON_TOLERANCE
               for result in ordinary):
        raise AssertionError("a converged case violates the SeisSol residual criterion")
    nominal = CASES[0]
    velocity = 0.1
    friction, derivative = friction_and_derivative(velocity, nominal.state_reference, nominal)
    step = 1.0e-7
    friction_plus, _ = friction_and_derivative(velocity + step, nominal.state_reference, nominal)
    friction_minus, _ = friction_and_derivative(velocity - step, nominal.state_reference, nominal)
    finite_difference = (friction_plus - friction_minus) / (2.0 * step)
    if not math.isclose(derivative, finite_difference, rel_tol=1.0e-8, abs_tol=1.0e-12):
        raise AssertionError("analytic friction derivative failed finite-difference check")
    tiny_velocity_state = update_aging_state(2.0, ALMOST_ZERO_FP64, 0.25, 0.02)
    if not math.isclose(tiny_velocity_state, 2.25, rel_tol=0.0, abs_tol=1.0e-15):
        raise AssertionError("Aging update does not approach the V->0 limit")
    try:
        friction_and_derivative(1.0, -1.0, CASES[0])
    except ValueError:
        pass
    else:
        raise AssertionError("invalid state was silently accepted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-dataset", type=Path,
                        help="write deterministic TSV input and golden files")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test or args.write_dataset is None:
        self_test()
    if args.write_dataset is not None:
        write_dataset(args.write_dataset)
    for case in CASES:
        result = coupled_aging_step(case)
        print(f"{result.case_id}: V={result.velocity:.17e} mu={result.friction:.17e} "
              f"psi={result.state:.17e} g_newton={result.last_newton_residual_g:.17e} "
              f"g_post={result.post_state_update_residual_g:.17e} "
              f"iters={result.total_newton_iterations} converged={result.converged}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
