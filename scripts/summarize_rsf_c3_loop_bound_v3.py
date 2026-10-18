#!/usr/bin/env python3
"""Aggregate five independent BW1000 loop-bound jobs without pseudo-replication."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/raw/bw1000"
VARIANTS = ("pre60", "bound22", "restore60")


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    point = (len(ordered) - 1) * q
    lower = int(point)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] * (upper - point) + ordered[upper] * (point - lower)


def native_summary(path: Path) -> dict[str, float]:
    raw = path.read_text()
    for name in VARIANTS[:2]:
        if not re.search(rf"correctness variant={name} n=65537 .*failures=0 status=PASS", raw):
            raise ValueError(f"native correctness gate missing: {path}, {name}")
    if "RSF_C3_LOOP_BOUND_NATIVE_V3_PASS" not in raw or not re.search(
            r"reversibility restore_over_pre=[0-9.]+ status=PASS", raw):
        raise ValueError(f"native reversibility gate missing: {path}")
    found = {}
    for name, value in re.findall(r"summary variant=(\w+) median_us=([0-9.]+)", raw):
        found[name] = float(value)
    if set(found) != set(VARIANTS):
        raise ValueError(f"native variants incomplete: {path}: {found}")
    if any(len(re.findall(rf"sample variant={name} index=", raw)) != 30 for name in VARIANTS):
        raise ValueError(f"native sample count not 30: {path}")
    return found


def triton_summary(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    if data.get("status") != "PASS" or data.get("n") != 65537 or not data.get("reversibility_pass"):
        raise ValueError(f"Triton gate missing: {path}")
    variants = data["variants"]
    if set(variants) != set(VARIANTS):
        raise ValueError(f"Triton variants incomplete: {path}")
    if any(len(variants[name]["samples_us"]) != 30 for name in VARIANTS):
        raise ValueError(f"Triton sample count not 30: {path}")
    for differences in data["max_abs_differences"].values():
        if any(value > 1.0e-10 for value in differences[:3]) or any(value != 0 for value in differences[3:]):
            raise ValueError(f"Triton correctness mismatch: {path}")
    return {name: variants[name]["median_us"] for name in VARIANTS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--array-job", type=int, required=True)
    parser.add_argument("--job-ids", required=True,
                        help="five comma-separated Slurm task job IDs in task-index order")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    job_ids = [int(value) for value in args.job_ids.split(",")]
    if len(job_ids) != 5 or len(set(job_ids)) != 5:
        raise SystemExit("--job-ids must name five distinct task jobs")
    candidates = [RAW / f"revision-v2-rsf-c3-loop-bound-formal-{job_id}"
                  for job_id in job_ids]
    tasks = {}
    source_manifest = None
    for expected_task_id, folder in enumerate(candidates):
        environment = (folder / "environment.txt").read_text()
        match = re.search(r"^slurm_array_task_id=(\d+)$", environment, re.M)
        if (not match or int(match.group(1)) != expected_task_id or
                f"slurm_job_id={job_ids[expected_task_id]}" not in environment):
            raise ValueError(f"job/task identity mismatch: {folder}")
        task_id = int(match.group(1))
        if task_id in tasks:
            raise ValueError(f"duplicate task id: {task_id}")
        sources = (folder / "source.sha256").read_text()
        if source_manifest is None:
            source_manifest = sources
        elif sources != source_manifest:
            raise ValueError(f"source hash mismatch: {folder}")
        hip = native_summary(folder / "native-65537.txt")
        triton = triton_summary(folder / "triton-65537.json")
        tasks[task_id] = {
            "task_id": task_id, "raw_directory": str(folder.relative_to(ROOT)),
            "order": "triton_first" if task_id % 2 else "hip_first",
            "hip_median_us": hip, "triton_median_us": triton,
            "hip_bound22_over_pre60": hip["bound22"] / hip["pre60"],
            "triton_bound22_over_pre60": triton["bound22"] / triton["pre60"],
            "triton_over_hip_pre60": triton["pre60"] / hip["pre60"],
            "triton_over_hip_bound22": triton["bound22"] / hip["bound22"],
        }
    if set(tasks) != set(range(5)):
        raise SystemExit(f"expected five complete tasks 0-4; found {sorted(tasks)}")
    rows = [tasks[index] for index in range(5)]
    summary = {}
    for key in ("hip_bound22_over_pre60", "triton_bound22_over_pre60",
                "triton_over_hip_pre60", "triton_over_hip_bound22"):
        values = [row[key] for row in rows]
        summary[key] = {"median": statistics.median(values),
                        "p5": quantile(values, 0.05), "p95": quantile(values, 0.95)}
    result = {"schema_version": "rsf-c3-loop-bound-independent-jobs/3.0",
              "platform": "BW1000", "array_job_id": args.array_job,
              "experimental_unit": "one Slurm array task",
              "status": "PASS_DESCRIPTIVE", "jobs": rows, "summary": summary,
              "source_sha256_manifest": source_manifest.splitlines()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
