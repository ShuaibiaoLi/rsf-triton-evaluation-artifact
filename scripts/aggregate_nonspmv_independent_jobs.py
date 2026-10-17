#!/usr/bin/env python3
"""Aggregate paired non-SpMV runs at the independent Slurm-job level."""

import argparse
import json
import math
import random
import re
from pathlib import Path


SUMMARY = re.compile(
    r"^summary case=(\S+) boundary=steady_batch_per_call batch=128 "
    r"median_us=([0-9.eE+-]+) p5_us=([0-9.eE+-]+) p95_us=([0-9.eE+-]+)$",
    re.MULTILINE,
)
CORRECT = re.compile(r"^correctness case=(\S+).* status=(\S+)$", re.MULTILINE)


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] if lo == hi else values[lo] * (hi - position) + values[hi] * (position - lo)


def bootstrap_ci(values, seed=20261015, draws=10000):
    rng = random.Random(seed)
    medians = []
    for _ in range(draws):
        sample = [values[rng.randrange(len(values))] for _ in values]
        medians.append(percentile(sample, 0.5))
    return [percentile(medians, 0.025), percentile(medians, 0.975)]


def parse_native(path):
    text = path.read_text()
    if "_PASS" not in text:
        raise ValueError(f"terminal PASS missing: {path}")
    if any(status != "PASS" for _, status in CORRECT.findall(text)):
        raise ValueError(f"correctness failure: {path}")
    return {case: float(median) for case, median, _, _ in SUMMARY.findall(text)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directories = sorted(p for p in args.root.iterdir() if p.is_dir())
    if len(directories) != 5:
        raise ValueError(f"expected five independent task directories, found {len(directories)}")
    selected = ("interior_stencil_large", "row_reduction", "rsf_c3")
    jobs = []
    for directory in directories:
        task_id = int(directory.name.rsplit("_", 1)[1])
        generic = parse_native(directory / "hip-generic.txt")
        rsf = parse_native(directory / "hip-rsf.txt")
        hip = generic | rsf
        triton_document = json.loads((directory / "triton.json").read_text())
        triton_cases = {row["name"]: row for row in triton_document["cases"]}
        if any(triton_cases[name]["correctness"]["status"] != "PASS" for name in selected):
            raise ValueError(f"Triton correctness failure: {directory}")
        rows = []
        for name in selected:
            triton = triton_cases[name]["steady_batch_per_call"]["128"]["median_us"]
            rows.append(
                {
                    "case": name,
                    "hip_median_us": hip[name],
                    "triton_median_us": triton,
                    "triton_over_hip": triton / hip[name],
                }
            )
        jobs.append(
            {
                "array_task_id": task_id,
                "order": (directory / "order.txt").read_text().strip(),
                "source_directory": str(directory),
                "rows": rows,
            }
        )
    if sorted(job["array_task_id"] for job in jobs) != list(range(5)):
        raise ValueError("array task IDs are incomplete or duplicated")

    aggregate = []
    for name in selected:
        values = [next(row for row in job["rows"] if row["case"] == name)["triton_over_hip"] for job in jobs]
        aggregate.append(
            {
                "case": name,
                "independent_jobs": len(values),
                "job_ratios": values,
                "median_ratio": percentile(values, 0.5),
                "p5_ratio": percentile(values, 0.05),
                "p95_ratio": percentile(values, 0.95),
                "bootstrap_median_ci95": bootstrap_ci(values),
                "direction_stable": all(value > 1 for value in values) or all(value < 1 for value in values),
            }
        )
    output = {
        "schema_version": "nonspmv-independent-jobs/1.0",
        "status": "PASS",
        "experimental_unit": "one independent Slurm array task; within-task batch=128 per-call median",
        "kernel_calls_are_not_independent_replicates": True,
        "jobs": sorted(jobs, key=lambda row: row["array_task_id"]),
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
