#!/usr/bin/env python3
"""Bounded-loop intervention for the frozen nine-case C3 fixed replay."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import torch
import triton


ACTUAL = (8, 22, 6, 4, 4, 6, 6, 1, 6)
CONVERGED = (1, 1, 1, 1, 1, 1, 1, 0, 1)
FLOAT_NAMES = (
    "shear_traction", "effective_normal_stress", "state_reference", "direct_effect_a",
    "evolution_effect_b", "reference_friction", "reference_velocity",
    "state_evolution_distance", "shear_impedance", "previous_velocity",
    "newton_tolerance",
)


def load(path: Path, name: str):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def measure(fn, warmup: int, repeat: int) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop) * 1000.0)
    ordered = sorted(samples)
    def quantile(q):
        index = (len(ordered) - 1) * q
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)
    return {"median_us": statistics.median(samples), "p5_us": quantile(0.05),
            "p95_us": quantile(0.95), "samples_us": samples}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rsf-root", type=Path, required=True)
    parser.add_argument("--source-v2", type=Path, required=True)
    parser.add_argument("--n", type=int, default=65537)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.n < 1 or args.warmup < 0 or args.repeat < 2:
        raise SystemExit("invalid n/warmup/repeat")
    probe = load(args.rsf_root / "probes/triton/rsf/aging_rsf_numerical_feasibility.py", "rsf_probe_v3")
    previous = load(args.source_v2, "fixed_replay_v2")
    independent = load(args.rsf_root / "aging_rsf_fp64_reference.py", "rsf_reference_v3")
    rows = probe.read_tsv(args.rsf_root / "test-data/generated/src03-aging-rsf-newton-fp64-input-v1.tsv")
    if len(rows) != 9:
        raise RuntimeError("expected nine frozen cases")
    inputs = {name: probe.tensor([float(rows[i % 9][name]) for i in range(args.n)])
              for name in FLOAT_NAMES}
    maximum = probe.tensor([int(rows[i % 9]["max_newton_iterations"]) for i in range(args.n)], torch.int32)
    actual = probe.tensor([ACTUAL[i % 9] for i in range(args.n)], torch.int32)
    converged = probe.tensor([CONVERGED[i % 9] for i in range(args.n)], torch.int32)
    outputs = [torch.empty(args.n, dtype=torch.float64, device="cuda") for _ in range(3)]
    outputs += [torch.empty(args.n, dtype=torch.int32, device="cuda") for _ in range(2)]
    grid = (triton.cdiv(args.n, 64),)

    def original():
        probe._c3_newton_kernel[grid](
            *(inputs[name] for name in FLOAT_NAMES), maximum, *outputs,
            n=args.n, block=64, loop_count=60, num_warps=1)

    def fixed(bound: int):
        previous.fixed_replay_kernel[grid](
            *(inputs[name] for name in FLOAT_NAMES[:-1]), actual, converged, *outputs,
            n=args.n, block=64, loop_count=bound, num_warps=1)

    original()
    torch.cuda.synchronize()
    gold = [tensor.clone() for tensor in outputs]
    max_differences = {}
    for bound in (60, 22):
        fixed(bound)
        torch.cuda.synchronize()
        differences = [float(torch.max(torch.abs(a.to(torch.float64) - b.to(torch.float64))).cpu())
                       for a, b in zip(gold, outputs)]
        if any(value > 1.0e-10 for value in differences[:3]) or any(value != 0 for value in differences[3:]):
            raise RuntimeError(f"bound={bound} GPU output mismatch: {differences}")
        max_differences[str(bound)] = differences

    # Check each distinct input against an independently written scalar FP64 reference.
    for index in range(min(args.n, 9)):
        case = independent.CASES[index]
        velocity, friction, residual, iterations, ok = independent.solve_velocity(
            case.state_reference, case.previous_velocity, case)
        expected = (velocity, friction, residual, iterations, int(ok))
        for field, target in zip(outputs, expected):
            value = field[index].item()
            if abs(value - target) > (0.0 if isinstance(target, int) else 1.0e-10):
                raise RuntimeError(f"scalar reference mismatch case={case.case_id}: {value} != {target}")

    if args.smoke:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result = {"schema_version": "rsf-c3-loop-bound-triton/3.0", "status": "PASS",
                  "mode": "smoke", "n": args.n, "max_abs_differences": max_differences}
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"status": "PASS", "mode": "smoke", "n": args.n}))
        return

    pre = measure(lambda: fixed(60), args.warmup, args.repeat)
    intervention = measure(lambda: fixed(22), args.warmup, args.repeat)
    restore = measure(lambda: fixed(60), args.warmup, args.repeat)
    restore_ratio = restore["median_us"] / pre["median_us"]
    result = {"schema_version": "rsf-c3-loop-bound-triton/3.0", "status": "PASS",
              "backend": "triton", "n": args.n, "warmup": args.warmup,
              "repeat": args.repeat, "actual_iterations": list(ACTUAL),
              "max_abs_differences": max_differences,
              "variants": {"pre60": pre, "bound22": intervention, "restore60": restore},
              "restore_over_pre": restore_ratio,
              "reversibility_pass": 0.9 <= restore_ratio <= 1.1}
    if not result["reversibility_pass"]:
        raise RuntimeError(f"failed reversibility gate: {restore_ratio}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "n": args.n, "restore_over_pre": restore_ratio,
                      "bound22_over_pre60": intervention["median_us"] / pre["median_us"]}))


if __name__ == "__main__":
    main()
