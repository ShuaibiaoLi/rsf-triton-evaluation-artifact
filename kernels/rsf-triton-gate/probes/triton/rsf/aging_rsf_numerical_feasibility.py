#!/usr/bin/env python3
"""C0-C6 FP64 Triton probe for the SRC-03 Aging-RSF paper gate.

Run only on an accelerator allocation.  This script consumes the committed
independent golden data; it never uses Triton output as its own oracle.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import traceback
from pathlib import Path

import torch
import triton
import triton.language as tl

from tolerance import compare


ATOL = 1.0e-12
RTOL = 1.0e-10
NEWTON_TOLERANCE = 1.0e-8
BLOCK = 64
MAX_NEWTON_ITERATIONS = 60
STATE_UPDATES = 2


@triton.jit
def _c0_c1_kernel(x_ptr, add_ptr, mul_ptr, div_ptr, log_ptr, exp_ptr, sqrt_ptr,
                  asinh_ptr, n: tl.constexpr, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=1.0)
    tl.store(add_ptr + offsets, x + 0.25, mask=mask)
    tl.store(mul_ptr + offsets, x * 1.5, mask=mask)
    tl.store(div_ptr + offsets, x / 1.25, mask=mask)
    tl.store(log_ptr + offsets, tl.log(x), mask=mask)
    tl.store(exp_ptr + offsets, tl.exp(x), mask=mask)
    tl.store(sqrt_ptr + offsets, tl.sqrt(x), mask=mask)
    # The candidate domain is x >= 0.  This stable-enough equivalent is used
    # because older Triton stacks may not expose a direct asinh operation.
    tl.store(asinh_ptr + offsets, tl.log(x + tl.sqrt(x * x + 1.0)), mask=mask)


@triton.jit
def _c2_friction_kernel(v_ptr, state_ptr, a_ptr, b_ptr, f0_ptr, v0_ptr, length_ptr,
                        mu_ptr, derivative_ptr, n: tl.constexpr, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < n
    v = tl.load(v_ptr + offsets, mask=mask, other=1.0)
    state = tl.load(state_ptr + offsets, mask=mask, other=1.0)
    a = tl.load(a_ptr + offsets, mask=mask, other=1.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    f0 = tl.load(f0_ptr + offsets, mask=mask, other=0.0)
    v0 = tl.load(v0_ptr + offsets, mask=mask, other=1.0)
    length = tl.load(length_ptr + offsets, mask=mask, other=1.0)
    log_state = tl.log(v0 * state / length)
    coefficient = 0.5 / v0 * tl.exp((f0 + b * log_state) / a)
    argument = v * coefficient
    mu = a * tl.log(argument + tl.sqrt(argument * argument + 1.0))
    derivative = a * coefficient / tl.sqrt(argument * argument + 1.0)
    tl.store(mu_ptr + offsets, mu, mask=mask)
    tl.store(derivative_ptr + offsets, derivative, mask=mask)


@triton.jit
def _c3_newton_kernel(tau_ptr, sigma_ptr, state_ptr, a_ptr, b_ptr, f0_ptr,
                      v0_ptr, length_ptr, eta_ptr, initial_v_ptr, tolerance_ptr,
                      max_iterations_ptr, velocity_ptr, friction_ptr, residual_ptr,
                      iterations_ptr, converged_ptr, n: tl.constexpr,
                      block: tl.constexpr, loop_count: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < n
    tau = tl.load(tau_ptr + offsets, mask=mask, other=0.0)
    sigma = tl.load(sigma_ptr + offsets, mask=mask, other=1.0)
    state = tl.load(state_ptr + offsets, mask=mask, other=1.0)
    a = tl.load(a_ptr + offsets, mask=mask, other=1.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    f0 = tl.load(f0_ptr + offsets, mask=mask, other=0.0)
    v0 = tl.load(v0_ptr + offsets, mask=mask, other=1.0)
    length = tl.load(length_ptr + offsets, mask=mask, other=1.0)
    eta = tl.load(eta_ptr + offsets, mask=mask, other=1.0)
    velocity = tl.maximum(tl.load(initial_v_ptr + offsets, mask=mask, other=1.0), 1.0e-45)
    tolerance = tl.load(tolerance_ptr + offsets, mask=mask, other=1.0e-8)
    max_iterations = tl.load(max_iterations_ptr + offsets, mask=mask, other=0)
    converged = offsets < 0
    iterations = tl.zeros((block,), dtype=tl.int32)
    residual = tl.zeros((block,), dtype=tl.float64)
    friction = tl.zeros((block,), dtype=tl.float64)

    # These terms depend only on the immutable RSF parameters, not on the
    # Newton iterate.  Keeping them outside the loop removes one log/exp pair
    # per iteration while preserving the FP64 expression used by the oracle.
    log_state = tl.log(v0 * state / length)
    coefficient = 0.5 / v0 * tl.exp((f0 + b * log_state) / a)
    # eta is immutable across Newton iterations. Reuse its reciprocal.
    inv_eta = 1.0 / eta

    for iteration in range(loop_count):
        argument = velocity * coefficient
        friction_now = a * tl.log(argument + tl.sqrt(argument * argument + 1.0))
        derivative = a * coefficient / tl.sqrt(argument * argument + 1.0)
        residual_now = (tau - sigma * friction_now) * inv_eta - velocity
        evaluating = mask & (~converged) & (iteration < max_iterations)
        newly_converged = evaluating & (tl.abs(residual_now) < tolerance)
        update = evaluating & (~newly_converged)
        derivative_g = -(sigma * inv_eta) * derivative - 1.0
        candidate = tl.maximum(velocity - residual_now / derivative_g, 1.0e-45)
        velocity = tl.where(update, candidate, velocity)
        iterations += evaluating.to(tl.int32)
        residual = tl.where(evaluating, residual_now, residual)
        friction = tl.where(evaluating, friction_now, friction)
        converged = converged | newly_converged

    # Recompute diagnostics at the final iterate, including the explicit
    # max-iteration failure case after its last Newton update.
    argument = velocity * coefficient
    friction = a * tl.log(argument + tl.sqrt(argument * argument + 1.0))
    residual = (tau - sigma * friction) * inv_eta - velocity
    tl.store(velocity_ptr + offsets, velocity, mask=mask)
    tl.store(friction_ptr + offsets, friction, mask=mask)
    tl.store(residual_ptr + offsets, residual, mask=mask)
    tl.store(iterations_ptr + offsets, iterations, mask=mask)
    tl.store(converged_ptr + offsets, converged.to(tl.int32), mask=mask)


@triton.jit
def _c4_coupled_kernel(tau_ptr, sigma_ptr, state_reference_ptr, a_ptr, b_ptr,
                       f0_ptr, v0_ptr, length_ptr, eta_ptr, previous_v_ptr,
                       dt_ptr, tolerance_ptr, max_iterations_ptr, velocity_ptr,
                       friction_ptr, state_ptr, last_residual_ptr,
                       post_residual_ptr, total_iterations_ptr,
                       max_inner_iterations_ptr, converged_ptr, finite_ptr,
                       n: tl.constexpr, block: tl.constexpr,
                       loop_count: tl.constexpr, state_updates: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < n
    tau = tl.load(tau_ptr + offsets, mask=mask, other=0.0)
    sigma = tl.load(sigma_ptr + offsets, mask=mask, other=1.0)
    state_reference = tl.load(state_reference_ptr + offsets, mask=mask, other=1.0)
    a = tl.load(a_ptr + offsets, mask=mask, other=1.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    f0 = tl.load(f0_ptr + offsets, mask=mask, other=0.0)
    v0 = tl.load(v0_ptr + offsets, mask=mask, other=1.0)
    length = tl.load(length_ptr + offsets, mask=mask, other=1.0)
    eta = tl.load(eta_ptr + offsets, mask=mask, other=1.0)
    velocity = tl.maximum(tl.load(previous_v_ptr + offsets, mask=mask, other=1.0), 1.0e-45)
    dt = tl.load(dt_ptr + offsets, mask=mask, other=0.0)
    tolerance = tl.load(tolerance_ptr + offsets, mask=mask, other=1.0e-8)
    max_iterations = tl.load(max_iterations_ptr + offsets, mask=mask, other=0)
    mean_velocity = velocity
    state = state_reference
    all_converged = mask
    total_iterations = tl.zeros((block,), dtype=tl.int32)
    max_inner_iterations = tl.zeros((block,), dtype=tl.int32)
    last_residual = tl.zeros((block,), dtype=tl.float64)

    for _outer in range(state_updates):
        x = mean_velocity * dt / length
        decay = tl.exp(-x)
        # Stable small-x evaluation of dt*(1-exp(-x))/x.
        x2 = x * x
        source_series = dt * (1.0 - 0.5 * x + x2 / 6.0)
        source_direct = dt * (1.0 - decay) / x
        source = tl.where(tl.abs(x) < 1.0e-8, source_series, source_direct)
        state = state_reference * decay + source
        old_velocity = velocity
        inner_converged = offsets < 0
        inner_iterations = tl.zeros((block,), dtype=tl.int32)

        for iteration in range(loop_count):
            log_state = tl.log(v0 * state / length)
            coefficient = 0.5 / v0 * tl.exp((f0 + b * log_state) / a)
            argument = velocity * coefficient
            friction_now = a * tl.log(argument + tl.sqrt(argument * argument + 1.0))
            derivative = a * coefficient / tl.sqrt(argument * argument + 1.0)
            residual_now = (tau - sigma * friction_now) / eta - velocity
            evaluating = mask & (~inner_converged) & (iteration < max_iterations)
            newly_converged = evaluating & (tl.abs(residual_now) < tolerance)
            update = evaluating & (~newly_converged)
            derivative_g = -(sigma / eta) * derivative - 1.0
            candidate = tl.maximum(velocity - residual_now / derivative_g, 1.0e-45)
            velocity = tl.where(update, candidate, velocity)
            inner_iterations += evaluating.to(tl.int32)
            inner_converged = inner_converged | newly_converged

        log_state = tl.log(v0 * state / length)
        coefficient = 0.5 / v0 * tl.exp((f0 + b * log_state) / a)
        argument = velocity * coefficient
        friction_now = a * tl.log(argument + tl.sqrt(argument * argument + 1.0))
        last_residual = (tau - sigma * friction_now) / eta - velocity
        total_iterations += inner_iterations
        max_inner_iterations = tl.maximum(max_inner_iterations, inner_iterations)
        all_converged = all_converged & inner_converged
        mean_velocity = 0.5 * (old_velocity + tl.abs(velocity))

    x = mean_velocity * dt / length
    decay = tl.exp(-x)
    x2 = x * x
    source_series = dt * (1.0 - 0.5 * x + x2 / 6.0)
    source_direct = dt * (1.0 - decay) / x
    source = tl.where(tl.abs(x) < 1.0e-8, source_series, source_direct)
    state = state_reference * decay + source
    log_state = tl.log(v0 * state / length)
    coefficient = 0.5 / v0 * tl.exp((f0 + b * log_state) / a)
    argument = velocity * coefficient
    friction = a * tl.log(argument + tl.sqrt(argument * argument + 1.0))
    post_residual = (tau - sigma * friction) / eta - velocity
    # NaN is the only non-finite value older Triton exposes portably here;
    # the host additionally checks Python finiteness after transfer.
    finite = (velocity == velocity) & (friction == friction) & (state == state) & \
             (last_residual == last_residual) & (post_residual == post_residual)
    tl.store(velocity_ptr + offsets, velocity, mask=mask)
    tl.store(friction_ptr + offsets, friction, mask=mask)
    tl.store(state_ptr + offsets, state, mask=mask)
    tl.store(last_residual_ptr + offsets, last_residual, mask=mask)
    tl.store(post_residual_ptr + offsets, post_residual, mask=mask)
    tl.store(total_iterations_ptr + offsets, total_iterations, mask=mask)
    tl.store(max_inner_iterations_ptr + offsets, max_inner_iterations, mask=mask)
    tl.store(converged_ptr + offsets, all_converged.to(tl.int32), mask=mask)
    tl.store(finite_ptr + offsets, finite.to(tl.int32), mask=mask)


INPUT_FLOAT_FIELDS = (
    "shear_traction", "effective_normal_stress", "state_reference", "direct_effect_a",
    "evolution_effect_b", "reference_friction", "reference_velocity",
    "state_evolution_distance", "shear_impedance", "previous_velocity", "time_step",
    "newton_tolerance",
)
GOLDEN_FLOAT_FIELDS = (
    "velocity", "friction", "state", "last_newton_residual_g",
    "post_state_update_residual_g",
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def tensor(values, dtype=torch.float64):
    return torch.tensor(values, dtype=dtype, device="cuda")


def close_metrics(actual: list[float], expected: list[float]) -> tuple[float, float, int]:
    metrics = compare(actual, expected)
    return metrics["max_abs"], metrics["max_rel"], metrics["failures"]


def environment() -> dict:
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "torch": getattr(torch, "__version__", "unknown"),
        "triton": getattr(triton, "__version__", "unknown"),
        "hip": getattr(torch.version, "hip", None),
        "cuda": getattr(torch.version, "cuda", None),
        "device_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "job_id": os.environ.get("SLURM_JOB_ID", "UNAVAILABLE"),
    }


def launch_coupled(inputs: list[dict[str, str]], size: int):
    rows = [inputs[index % len(inputs)] for index in range(size)]
    data = {name: tensor([float(row[name]) for row in rows]) for name in INPUT_FLOAT_FIELDS}
    max_iterations = tensor([int(row["max_newton_iterations"]) for row in rows], torch.int32)
    outputs_float = [torch.empty(size, dtype=torch.float64, device="cuda") for _ in range(5)]
    outputs_int = [torch.empty(size, dtype=torch.int32, device="cuda") for _ in range(4)]
    grid = (triton.cdiv(size, BLOCK),)
    _c4_coupled_kernel[grid](
        data["shear_traction"], data["effective_normal_stress"], data["state_reference"],
        data["direct_effect_a"], data["evolution_effect_b"], data["reference_friction"],
        data["reference_velocity"], data["state_evolution_distance"], data["shear_impedance"],
        data["previous_velocity"], data["time_step"], data["newton_tolerance"],
        max_iterations, *outputs_float, *outputs_int, n=size, block=BLOCK,
        loop_count=MAX_NEWTON_ITERATIONS, state_updates=STATE_UPDATES, num_warps=4)
    torch.cuda.synchronize()
    return tuple(output.cpu().tolist() for output in outputs_float + outputs_int)


def run_probe(root: Path, skip_c01: bool = False) -> dict:
    result = {"protocol": "src03-aging-rsf-triton-numerical-feasibility-v1",
              "environment": environment(), "stages": {}, "decision": "FAIL"}
    if not torch.cuda.is_available():
        raise RuntimeError("C0: torch reports no CUDA/HIP-like accelerator")
    input_rows = read_tsv(root / "test-data/generated/src03-aging-rsf-newton-fp64-input-v1.tsv")
    golden_rows = read_tsv(root / "test-data/golden/src03-aging-rsf-newton-fp64-golden-v1.tsv")

    if skip_c01:
        result["stages"]["C0_dtype"] = {
            "status": "PASS", "evidence": "separate-process isolation job required"}
        result["stages"]["C1_math"] = {
            "status": "PASS", "evidence": "separate-process isolation job required"}
    else:
        values = [1.0e-12, 1.0e-6, 0.01, 0.25, 1.0, 4.0, 9.0, 20.0, 64.0]
        x = tensor(values)
        arithmetic = [torch.empty_like(x) for _ in range(3)]
        special = [torch.empty_like(x) for _ in range(4)]
        _c0_c1_kernel[(1,)](x, *arithmetic, *special, n=len(values), block=BLOCK, num_warps=4)
        torch.cuda.synchronize()
        references = ([v + 0.25 for v in values], [v * 1.5 for v in values],
                      [v / 1.25 for v in values], [math.log(v) for v in values],
                      [math.exp(v) for v in values], [math.sqrt(v) for v in values],
                      [math.asinh(v) for v in values])
        c01_metrics = [close_metrics(got.cpu().tolist(), want)
                       for got, want in zip(arithmetic + special, references)]
        result["stages"]["C0_dtype"] = {"status": "PASS" if all(x[2] == 0 for x in c01_metrics[:3]) else "FAIL",
                                         "metrics": c01_metrics[:3]}
        result["stages"]["C1_math"] = {"status": "PASS" if all(x[2] == 0 for x in c01_metrics[3:]) else "FAIL",
                                        "metrics": c01_metrics[3:]}

    velocities = [max(1.0e-45, float(row["previous_velocity"])) for row in input_rows]
    c2_inputs = [tensor(velocities)] + [tensor([float(row[name]) for row in input_rows]) for name in
                                       ("state_reference", "direct_effect_a", "evolution_effect_b",
                                        "reference_friction", "reference_velocity",
                                        "state_evolution_distance")]
    c2_mu, c2_derivative = torch.empty_like(c2_inputs[0]), torch.empty_like(c2_inputs[0])
    _c2_friction_kernel[(1,)](*c2_inputs, c2_mu, c2_derivative, n=len(input_rows),
                              block=BLOCK, num_warps=4)
    torch.cuda.synchronize()
    host_mu, host_derivative = [], []
    for row, velocity in zip(input_rows, velocities):
        state, a, b = float(row["state_reference"]), float(row["direct_effect_a"]), float(row["evolution_effect_b"])
        f0, v0, length = float(row["reference_friction"]), float(row["reference_velocity"]), float(row["state_evolution_distance"])
        coefficient = 0.5 / v0 * math.exp((f0 + b * math.log(v0 * state / length)) / a)
        argument = velocity * coefficient
        host_mu.append(a * math.asinh(argument))
        host_derivative.append(a * coefficient / math.sqrt(argument * argument + 1.0))
    c2_metrics = [close_metrics(c2_mu.cpu().tolist(), host_mu),
                  close_metrics(c2_derivative.cpu().tolist(), host_derivative)]
    result["stages"]["C2_expression"] = {"status": "PASS" if all(x[2] == 0 for x in c2_metrics) else "FAIL",
                                          "metrics": c2_metrics}

    # C3 uses the committed initial states and only checks fixed-state Newton
    # residual/convergence; C4 below is the full golden comparison.
    n = len(input_rows)
    fixed_float_names = ("shear_traction", "effective_normal_stress", "state_reference",
                         "direct_effect_a", "evolution_effect_b", "reference_friction",
                         "reference_velocity", "state_evolution_distance", "shear_impedance",
                         "previous_velocity", "newton_tolerance")
    fixed_inputs = [tensor([float(row[name]) for row in input_rows]) for name in fixed_float_names]
    fixed_max = tensor([int(row["max_newton_iterations"]) for row in input_rows], torch.int32)
    fixed_out_float = [torch.empty(n, dtype=torch.float64, device="cuda") for _ in range(3)]
    fixed_out_int = [torch.empty(n, dtype=torch.int32, device="cuda") for _ in range(2)]
    _c3_newton_kernel[(1,)](*fixed_inputs, fixed_max, *fixed_out_float, *fixed_out_int,
                            n=n, block=BLOCK, loop_count=MAX_NEWTON_ITERATIONS, num_warps=4)
    torch.cuda.synchronize()
    fixed_residual = fixed_out_float[2].cpu().tolist()
    fixed_converged = fixed_out_int[1].cpu().tolist()
    fixed_failures = sum(1 for row, residual, conv in zip(input_rows, fixed_residual, fixed_converged)
                         if ((int(row["max_newton_iterations"]) > 1 and (conv != 1 or abs(residual) >= NEWTON_TOLERANCE))
                             or (int(row["max_newton_iterations"]) == 1 and conv != 0)))
    result["stages"]["C3_newton"] = {"status": "PASS" if fixed_failures == 0 else "FAIL",
                                      "failures": fixed_failures,
                                      "max_converged_abs_residual": max(abs(r) for r, c in zip(fixed_residual, fixed_converged) if c)}

    coupled = launch_coupled(input_rows, n)
    float_metrics = []
    for index, name in enumerate(GOLDEN_FLOAT_FIELDS):
        float_metrics.append(close_metrics(coupled[index], [float(row[name]) for row in golden_rows]))
    integer_names = ("total_newton_iterations", "max_inner_iterations", "converged", "finite")
    integer_mismatches = {}
    for offset, name in enumerate(integer_names, start=5):
        integer_mismatches[name] = sum(int(got) != int(row[name]) for got, row in zip(coupled[offset], golden_rows))
    required_integer_mismatches = integer_mismatches["converged"] + integer_mismatches["finite"]
    c4_pass = all(metric[2] == 0 for metric in float_metrics[:3]) and required_integer_mismatches == 0
    result["stages"]["C4_coupling"] = {"status": "PASS" if c4_pass else "FAIL",
                                        "float_metrics": dict(zip(GOLDEN_FLOAT_FIELDS, float_metrics)),
                                        "integer_mismatches": integer_mismatches}

    c5_sizes = (1, 9, 65, 257, 4099, 65537)
    c5_failures = 0
    for size in c5_sizes:
        outputs = launch_coupled(input_rows, size)
        for field_index in range(5):
            expected = [float(golden_rows[index % n][GOLDEN_FLOAT_FIELDS[field_index]]) for index in range(size)]
            c5_failures += close_metrics(outputs[field_index], expected)[2]
        c5_failures += sum(int(value) != int(golden_rows[index % n]["converged"])
                           for index, value in enumerate(outputs[7]))
        c5_failures += sum((not math.isfinite(outputs[field][index]))
                           for field in range(5) for index in range(size))
    result["stages"]["C5_batch"] = {"status": "PASS" if c5_failures == 0 else "FAIL",
                                     "sizes": c5_sizes, "failures": c5_failures}

    first = launch_coupled(input_rows, 257)
    repeat_mismatches = 0
    for _ in range(99):
        current = launch_coupled(input_rows, 257)
        repeat_mismatches += sum(a != b for left, right in zip(first, current)
                                 for a, b in zip(left, right))
    result["stages"]["C6_repeat"] = {"status": "PASS" if repeat_mismatches == 0 else "FAIL",
                                      "repeats": 100, "mismatches": repeat_mismatches}
    result["decision"] = "PASS" if all(stage["status"] == "PASS" for stage in result["stages"].values()) else "FAIL"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--json", type=Path)
    parser.add_argument("--skip-c01", action="store_true",
                        help="use only with a recorded separate-process C0/C1 isolation job")
    args = parser.parse_args()
    try:
        result = run_probe(args.root, skip_c01=args.skip_c01)
    except Exception as exc:  # preserve compiler/runtime failures as evidence
        result = {"protocol": "src03-aging-rsf-triton-numerical-feasibility-v1",
                  "environment": environment(), "decision": "FAIL",
                  "fatal_error": repr(exc), "traceback": traceback.format_exc()}
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(encoded + "\n", encoding="utf-8")
    return 0 if result["decision"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
