#!/usr/bin/env python3
"""Triton half of the frozen C3 fixed-actual-iteration reversible ablation."""

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


FLOAT_NAMES = (
    "shear_traction", "effective_normal_stress", "state_reference", "direct_effect_a",
    "evolution_effect_b", "reference_friction", "reference_velocity",
    "state_evolution_distance", "shear_impedance", "previous_velocity", "newton_tolerance",
)


@triton.jit
def fixed_replay_kernel(tau_ptr, sigma_ptr, state_ptr, a_ptr, b_ptr, f0_ptr,
                        v0_ptr, length_ptr, eta_ptr, initial_v_ptr, count_ptr,
                        converged_target_ptr, velocity_ptr, friction_ptr, residual_ptr,
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
    count = tl.load(count_ptr + offsets, mask=mask, other=0)
    target_converged = tl.load(converged_target_ptr + offsets, mask=mask, other=0) != 0
    log_state = tl.log(v0 * state / length)
    coefficient = 0.5 / v0 * tl.exp((f0 + b * log_state) / a)
    inv_eta = 1.0 / eta
    for iteration in range(loop_count):
        argument = velocity * coefficient
        friction = a * tl.log(argument + tl.sqrt(argument * argument + 1.0))
        derivative = a * coefficient / tl.sqrt(argument * argument + 1.0)
        residual = (tau - sigma * friction) * inv_eta - velocity
        evaluating = mask & (iteration < count)
        is_last = iteration + 1 == count
        update = evaluating & ((~is_last) | (~target_converged))
        derivative_g = -(sigma * inv_eta) * derivative - 1.0
        candidate = tl.maximum(velocity - residual / derivative_g, 1.0e-45)
        velocity = tl.where(update, candidate, velocity)
    argument = velocity * coefficient
    friction = a * tl.log(argument + tl.sqrt(argument * argument + 1.0))
    residual = (tau - sigma * friction) * inv_eta - velocity
    tl.store(velocity_ptr + offsets, velocity, mask=mask)
    tl.store(friction_ptr + offsets, friction, mask=mask)
    tl.store(residual_ptr + offsets, residual, mask=mask)
    tl.store(iterations_ptr + offsets, count, mask=mask)
    tl.store(converged_ptr + offsets, target_converged.to(tl.int32), mask=mask)


def load_module(path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("rsf_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stats(values):
    values = sorted(values)
    def quantile(q):
        position = (len(values) - 1) * q
        lo, hi = int(position), min(int(position) + 1, len(values) - 1)
        return values[lo] * (hi - position) + values[hi] * (position - lo)
    return {"median_us": statistics.median(values), "p5_us": quantile(0.05),
            "p95_us": quantile(0.95), "samples_us": values}


def timed(fn, warmup, repeat):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeat):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); fn(); stop.record(); stop.synchronize()
        values.append(start.elapsed_time(stop) * 1.0e3)
    return stats(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rsf-root", type=Path, required=True)
    parser.add_argument("--n", type=int, default=65537)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probe = load_module(args.rsf_root / "probes/triton/rsf/aging_rsf_numerical_feasibility.py")
    rows = probe.read_tsv(args.rsf_root / "test-data/generated/src03-aging-rsf-newton-fp64-input-v1.tsv")
    selected = [rows[i % len(rows)] for i in range(args.n)]
    floats = {name: probe.tensor([float(row[name]) for row in selected]) for name in FLOAT_NAMES}
    maximum = probe.tensor([int(row["max_newton_iterations"]) for row in selected], torch.int32)
    actual_base = (8, 22, 6, 4, 4, 6, 6, 1, 6)
    converged_base = (1, 1, 1, 1, 1, 1, 1, 0, 1)
    actual = probe.tensor([actual_base[i % 9] for i in range(args.n)], torch.int32)
    target = probe.tensor([converged_base[i % 9] for i in range(args.n)], torch.int32)
    outputs = [torch.empty(args.n, dtype=torch.float64, device="cuda") for _ in range(3)]
    outputs += [torch.empty(args.n, dtype=torch.int32, device="cuda") for _ in range(2)]

    def original():
        probe._c3_newton_kernel[(triton.cdiv(args.n, 64),)](
            *(floats[name] for name in FLOAT_NAMES), maximum, *outputs,
            n=args.n, block=64, loop_count=60, num_warps=1)

    def replay():
        fixed_replay_kernel[(triton.cdiv(args.n, 64),)](
            *(floats[name] for name in FLOAT_NAMES[:-1]), actual, target, *outputs,
            n=args.n, block=64, loop_count=60, num_warps=1)

    original(); torch.cuda.synchronize()
    reference = [x.clone() for x in outputs]
    replay(); torch.cuda.synchronize()
    differences = [float(torch.max(torch.abs(a.to(torch.float64) - b.to(torch.float64))).cpu())
                   for a, b in zip(reference, outputs)]
    if any(value > 1.0e-10 for value in differences[:3]) or any(value != 0 for value in differences[3:]):
        raise RuntimeError(f"fixed replay mismatch: {differences}")

    original_pre = timed(original, args.warmup, args.repeat)
    fixed = timed(replay, args.warmup, args.repeat)
    original_restore = timed(original, args.warmup, args.repeat)
    restore_ratio = original_restore["median_us"] / original_pre["median_us"]
    result = {
        "schema_version": "rsf-c3-fixed-replay-triton/2.0", "status": "PASS",
        "backend": "triton", "n": args.n, "warmup": args.warmup, "repeat": args.repeat,
        "actual_iteration_histogram": {str(k): actual_base.count(k) for k in sorted(set(actual_base))},
        "max_abs_differences": differences,
        "variants": {"original_pre": original_pre, "fixed_iteration_replay": fixed,
                     "original_restore": original_restore},
        "restore_over_pre": restore_ratio,
        "reversibility_pass": 0.9 <= restore_ratio <= 1.1,
    }
    if not result["reversibility_pass"]:
        raise RuntimeError(f"reversibility criterion failed: {restore_ratio}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "restore_over_pre": restore_ratio,
                      "fixed_over_original": fixed["median_us"] / original_pre["median_us"]}))


if __name__ == "__main__":
    main()
