#!/usr/bin/env python3
"""Freeze revision-v1 SpMV cell configurations from a complete tuning ledger."""

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/protocol/revision_v1_experiment_protocol.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometric_mean(values):
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric mean requires finite positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--source-sha256", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if ledger.get("status") != "PASS" or ledger.get("candidate_count") != 96:
        raise ValueError("refusing to freeze: tuning ledger is not a complete 96-candidate PASS")
    records = ledger["records"]
    if any(record["correctness"] != "PASS" or record["median_us"] is None for record in records):
        raise ValueError("refusing to freeze: an eligible record failed correctness or timing")
    tuning = set(protocol["splits"]["tuning"])
    observed = {record["matrix"] for record in records}
    if observed != tuning:
        raise ValueError(f"tuning set mismatch: expected {sorted(tuning)}, observed {sorted(observed)}")

    normalized = {}
    for matrix in sorted(tuning):
        for backend in ("hip", "triton"):
            subset = [record for record in records if record["matrix"] == matrix and record["backend"] == backend]
            best = min(record["median_us"] for record in subset)
            for record in subset:
                key = (backend, record["mapping"], record["block_size"], record["num_warps_or_wave_groups"])
                normalized.setdefault(key, []).append(record["median_us"] / best)

    ranked = []
    for key, slowdowns in normalized.items():
        backend, mapping, block, groups = key
        if len(slowdowns) != len(tuning):
            raise ValueError(f"configuration does not cover every tuning matrix: {key}")
        ranked.append({
            "backend": backend, "mapping": mapping, "block_size": block,
            "num_warps_or_wave_groups": groups,
            "normalized_geomean": geometric_mean(slowdowns),
            "per_matrix_normalized_slowdown": dict(zip(sorted(tuning), slowdowns)),
        })
    ranked.sort(key=lambda item: (item["backend"], item["mapping"], item["normalized_geomean"],
                                  item["block_size"], item["num_warps_or_wave_groups"]))
    selected_cells = []
    for backend in ("hip", "triton"):
        mappings = sorted({item["mapping"] for item in ranked if item["backend"] == backend})
        for mapping in mappings:
            eligible = [item for item in ranked if item["backend"] == backend and item["mapping"] == mapping]
            selected_cells.append(min(eligible, key=lambda item: (item["normalized_geomean"], item["block_size"], item["num_warps_or_wave_groups"])))
    default_by_backend = {}
    for backend in ("hip", "triton"):
        eligible = [item for item in selected_cells if item["backend"] == backend]
        default_by_backend[backend] = min(eligible, key=lambda item: (item["normalized_geomean"], item["mapping"]))

    frozen = {
        "schema_version": "spmv-frozen-rule/1.0",
        "status": "FROZEN",
        "freeze_time_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(PROTOCOL),
        "tuning_ledger": {"path": str(args.ledger), "sha256": sha256(args.ledger)},
        "source_sha256_manifest": {"path": str(args.source_sha256), "sha256": sha256(args.source_sha256)},
        "selection_metric": "geometric mean of per-matrix latency normalized by the best same-backend candidate",
        "tie_break": ["normalized_geomean", "block_size", "num_warps_or_wave_groups"],
        "selected_per_cell": selected_cells,
        "constant_default_per_backend": default_by_backend,
        "rule_scope": "configuration freeze and constant default only",
        "structural_rule_status": "NOT_ESTABLISHED_TWO_TUNING_MATRICES",
        "holdout_policy": "revision holdout remains locked until this artifact exists; do not modify after holdout execution"
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "FROZEN", "output": str(args.output),
                      "selected_per_cell": selected_cells,
                      "constant_default_per_backend": default_by_backend}, indent=2))


if __name__ == "__main__":
    main()
