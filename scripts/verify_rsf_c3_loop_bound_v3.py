#!/usr/bin/env python3
"""Read-only audit of the frozen BW1000 C3 loop-bound v3 evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/bw1000/revision_v2/ablation"
VARIANTS = ("pre60", "bound22", "restore60")
METRICS = (
    "hip_bound22_over_pre60",
    "triton_bound22_over_pre60",
    "triton_over_hip_pre60",
    "triton_over_hip_bound22",
)
SOURCES = {
    "rsf_c3_fixed_replay_native_v2.cpp": "BW1000_test/generic_microbench/rsf_c3_fixed_replay_native_v2.cpp",
    "rsf_c3_fixed_replay_triton_v2.py": "BW1000_test/generic_microbench/rsf_c3_fixed_replay_triton_v2.py",
    "rsf_c3_loop_bound_native_v3.cpp": "BW1000_test/generic_microbench/rsf_c3_loop_bound_native_v3.cpp",
    "rsf_c3_loop_bound_triton_v3.py": "BW1000_test/generic_microbench/rsf_c3_loop_bound_triton_v3.py",
    "aging_rsf_fp64_reference.py": "BW1000_test/rsf-triton-gate/aging_rsf_fp64_reference.py",
    "src03-aging-rsf-newton-fp64-input-v1.tsv": "BW1000_test/rsf-triton-gate/test-data/generated/src03-aging-rsf-newton-fp64-input-v1.tsv",
}


def artifact_path(relative: str) -> Path:
    """Resolve the author-workspace and curated-release layouts explicitly."""
    direct = ROOT / relative
    if direct.exists():
        return direct
    if relative.startswith("results/raw/"):
        return ROOT / relative.replace("results/raw/", "raw/", 1)
    if relative.startswith("BW1000_test/"):
        return ROOT / relative.replace("BW1000_test/", "kernels/", 1)
    return direct


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(left: float, right: float, label: str, tolerance: float = 1e-9) -> None:
    require(math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance),
            f"{label}: {left} != {right}")


def quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def hashes(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        basename = Path(name).name
        require(basename not in result, f"duplicate source basename: {path}: {basename}")
        result[basename] = digest
    return result


def audit_native(path: Path) -> dict[str, float]:
    content = path.read_text()
    require("RSF_C3_LOOP_BOUND_NATIVE_V3_PASS" in content, f"native pass marker missing: {path}")
    for variant in VARIANTS[:2]:
        require(re.search(rf"^correctness variant={variant} n=65537 .*failures=0 status=PASS$",
                          content, re.M) is not None, f"native correctness missing: {path}: {variant}")
    medians = {}
    for variant in VARIANTS:
        summary = re.findall(rf"^summary variant={variant} median_us=([0-9.]+)", content, re.M)
        require(len(summary) == 1, f"native summary missing/duplicate: {path}: {variant}")
        samples = re.findall(rf"^sample variant={variant} index=(\d+) us=([0-9.]+)$",
                             content, re.M)
        require(len(samples) == 30 and {int(i) for i, _ in samples} == set(range(30)),
                f"native samples incomplete: {path}: {variant}")
        median = statistics.median(float(value) for _, value in samples)
        close(median, float(summary[0]), f"native median {path}: {variant}")
        medians[variant] = median
    recovery = medians["restore60"] / medians["pre60"]
    require(.9 <= recovery <= 1.1, f"native recovery failed: {path}: {recovery}")
    match = re.search(r"^reversibility restore_over_pre=([0-9.]+) status=PASS$", content, re.M)
    require(match is not None, f"native recovery marker missing: {path}")
    close(recovery, float(match.group(1)), f"native recovery {path}")
    return medians


def audit_triton(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    require(data["status"] == "PASS" and data["n"] == 65537 and data["repeat"] == 30,
            f"Triton formal contract failed: {path}")
    require(data["actual_iterations"] == [8, 22, 6, 4, 4, 6, 6, 1, 6],
            f"unexpected iteration oracle: {path}")
    for differences in data["max_abs_differences"].values():
        require(len(differences) == 5 and all(value <= 1e-10 for value in differences[:3])
                and differences[3:] == [0.0, 0.0], f"Triton correctness failed: {path}")
    medians = {}
    for variant in VARIANTS:
        record = data["variants"][variant]
        samples = record["samples_us"]
        require(len(samples) == 30 and all(value > 0 for value in samples),
                f"Triton samples incomplete: {path}: {variant}")
        median = statistics.median(samples)
        close(median, record["median_us"], f"Triton median {path}: {variant}")
        medians[variant] = median
    recovery = medians["restore60"] / medians["pre60"]
    require(data["reversibility_pass"] and .9 <= recovery <= 1.1,
            f"Triton recovery failed: {path}: {recovery}")
    close(recovery, data["restore_over_pre"], f"Triton recovery {path}")
    return medians


def audit_pmc(data: dict, source_hashes: dict[str, str]) -> None:
    require(data["status"] == "PASS_DIRECTIONAL_PROFILE" and
            data["profile_time_used_for_speedup"] is False,
            "PMC status or measurement boundary failed")
    folder = artifact_path(data["raw_directory"])
    profile_hashes = hashes(folder / "source.sha256")
    for basename in SOURCES:
        require(profile_hashes.get(basename) == source_hashes[basename],
                f"PMC/formal source hash mismatch: {basename}")
    require("RSF_C3_LOOP_BOUND_PMC_V3_PASS" in
            (folder / f"rsf-c3-bound-pmc-v3-{data['job_id']}.out").read_text(),
            "PMC pass marker missing")
    for backend, stem in (("hip", "native"), ("triton", "triton")):
        with (folder / f"{stem}-pmc.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if backend == "hip":
            pre = [row for row in rows if "fixed_replay_kernel" in row["KernelName"]]
            bounded = [row for row in rows if "bounded_replay22_kernel" in row["KernelName"]]
        else:
            fixed = [row for row in rows if row["KernelName"] == "fixed_replay_kernel"]
            pre, bounded = fixed[:1], fixed[1:]
        require(len(pre) == len(bounded) == 1,
                f"PMC dispatch identification failed: {backend}")
        for variant, row in (("pre60", pre[0]), ("bound22", bounded[0])):
            stored = data["backends"][backend][variant]
            for field, column in (("dispatch_index", "Index"), ("kernel_name", "KernelName"),
                                  ("grid", "grd"), ("workgroup", "wgr")):
                value = row[column] if field == "kernel_name" else int(row[column])
                require(stored[field] == value, f"PMC {backend} {variant} {field} mismatch")
            for field in ("SQ_INSTS_VALU", "SQ_INSTS_VMEM_RD", "SQ_INSTS_VMEM_WR",
                          "arch_vgpr", "sgpr", "scr"):
                require(stored[field] == int(row[field]),
                        f"PMC {backend} {variant} {field} mismatch")
        pre_count = int(pre[0]["SQ_INSTS_VALU"])
        bounded_count = int(bounded[0]["SQ_INSTS_VALU"])
        close(bounded_count / pre_count,
              data["backends"][backend]["valu_bound22_over_pre60"], f"PMC ratio {backend}")


def main() -> None:
    aggregate = json.loads((RESULTS / "rsf-c3-loop-bound-independent-jobs-v3.json").read_text())
    require(aggregate["platform"] == "BW1000" and aggregate["status"] == "PASS_DESCRIPTIVE",
            "aggregate status/platform mismatch")
    jobs = aggregate["jobs"]
    require(len(jobs) == 5 and {job["task_id"] for job in jobs} == set(range(5)),
            "expected five distinct task units")
    source_hashes = None
    for job in jobs:
        task = job["task_id"]
        folder = artifact_path(job["raw_directory"])
        job_id = int(folder.name.rsplit("-", 1)[1])
        environment = (folder / "environment.txt").read_text()
        require(re.search(rf"^slurm_job_id={job_id}$", environment, re.M) and
                re.search(rf"^slurm_array_task_id={task}$", environment, re.M) and
                re.search(r"^mode=formal$", environment, re.M),
                f"formal job identity mismatch: {folder}")
        require(f"RSF_C3_LOOP_BOUND_V3_FORMAL_PASS" in
                (folder / f"rsf-c3-bound-v3-{job_id}.out").read_text(),
                f"formal pass marker missing: {folder}")
        current_hashes = hashes(folder / "source.sha256")
        if source_hashes is None:
            source_hashes = current_hashes
        require(current_hashes == source_hashes, f"formal source hash drift: {folder}")
        hip = audit_native(folder / "native-65537.txt")
        triton = audit_triton(folder / "triton-65537.json")
        for backend, actual in (("hip", hip), ("triton", triton)):
            for variant in VARIANTS:
                close(actual[variant], job[f"{backend}_median_us"][variant],
                      f"aggregate {backend} task={task} {variant}")
        ratios = {
            METRICS[0]: hip["bound22"] / hip["pre60"],
            METRICS[1]: triton["bound22"] / triton["pre60"],
            METRICS[2]: triton["pre60"] / hip["pre60"],
            METRICS[3]: triton["bound22"] / hip["bound22"],
        }
        for name, value in ratios.items():
            close(value, job[name], f"aggregate task={task} {name}")
        require(ratios[METRICS[3]] < ratios[METRICS[2]],
                f"ratio did not shrink in task {task}")
        require(job["order"] == ("hip_first" if task % 2 == 0 else "triton_first"),
                f"alternating order mismatch in task {task}")
    require(source_hashes is not None, "source hashes missing")
    for basename, relative in SOURCES.items():
        digest = hashlib.sha256(artifact_path(relative).read_bytes()).hexdigest()
        require(source_hashes.get(basename) == digest, f"local source hash mismatch: {relative}")
    for name in METRICS:
        values = [job[name] for job in jobs]
        stored = aggregate["summary"][name]
        for field, value in (("median", statistics.median(values)),
                             ("p5", quantile(values, .05)), ("p95", quantile(values, .95))):
            close(value, stored[field], f"aggregate summary {name} {field}")
    pmc = json.loads((RESULTS / "rsf-c3-loop-bound-pmc-v3.json").read_text())
    audit_pmc(pmc, source_hashes)
    print("PASS: five BW1000 task units, raw medians, restoration, source hashes, "
          "summary ratios, and independent PMC match")


if __name__ == "__main__":
    main()
