#!/usr/bin/env python3
"""Recompute published task-level descriptive statistics from frozen formal JSON."""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
TOL = 1e-12


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    i = (len(values) - 1) * q / 100.0
    lo, hi = math.floor(i), math.ceil(i)
    return values[lo] if lo == hi else values[lo] + (values[hi] - values[lo]) * (i - lo)


def check(entry: dict, values: list[float], field_names: tuple[str, str, str]) -> dict:
    computed = {field_names[0]: percentile(values, 50), field_names[1]: percentile(values, 5), field_names[2]: percentile(values, 95)}
    stored = {key: entry[key] for key in field_names}
    deltas = {key: computed[key] - stored[key] for key in field_names}
    return {"key": {key: entry[key] for key in ("case", "matrix", "metric") if key in entry},
            "n": len(values), "stored": stored, "recomputed": computed, "max_abs_delta": max(abs(x) for x in deltas.values()),
            "pass": all(abs(x) <= TOL for x in deltas.values())}


def main() -> None:
    nonspmv = json.loads((ROOT / "results/bw1000/revision_v1/formal/nonspmv-independent-jobs-v1.json").read_text())
    spmv = json.loads((ROOT / "results/bw1000/revision_v1/formal/spmv-four-cell-independent-jobs-v1.json").read_text())
    rows = []
    for item in nonspmv["aggregate"]:
        rows.append(check(item, item["job_ratios"], ("median_ratio", "p5_ratio", "p95_ratio")))
    for item in spmv["aggregate"]:
        rows.append(check(item, item["job_values"], ("median", "p5", "p95")))
    output = {"schema_version": "paper-task-stat-reproduction/1.0", "method": "linear percentile, identical to NumPy default percentile convention", "tolerance": TOL, "checks": rows, "status": "PASS" if all(row["pass"] for row in rows) else "FAIL"}
    (HERE / "reproducibility_check_v1.json").write_text(json.dumps(output, indent=2) + "\n")
    print(f"{output['status']}: {len(rows)} task-level aggregates recomputed")


if __name__ == "__main__":
    main()
