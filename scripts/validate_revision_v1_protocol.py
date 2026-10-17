#!/usr/bin/env python3
"""Validate the frozen major-revision experiment contracts without GPU access."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "docs/protocol/revision_v1_experiment_protocol.json",
    )
    parser.add_argument(
        "--correctness",
        type=Path,
        default=ROOT / "docs/protocol/correctness_contract_v1.json",
    )
    args = parser.parse_args()

    protocol = load(args.protocol)
    correctness = load(args.correctness)
    splits = protocol["splits"]
    tuning = set(splits["tuning"])
    holdout = set(splits["revision_holdout"])
    formal = set(splits["formal_representatives"])
    require(not tuning & holdout, "tuning and revision holdout overlap")
    require(not formal & holdout, "formal representatives and revision holdout overlap")
    require(protocol["timing"]["independent_jobs_minimum"] >= 5,
            "independent job minimum must be at least five")
    require(protocol["timing"]["steady_calls_per_job"] >= 20,
            "each independent job must contain at least 20 steady calls")
    mappings = protocol["spmv"]["mappings"]
    require(len(mappings["hip"]) == 2 and len(mappings["triton"]) == 2,
            "SpMV fairness requires two mappings per backend")
    candidates = protocol["spmv"]["candidate_space"]
    enumerated = len(candidates["block_size"]) * len(candidates["num_warps_or_wave_groups"])
    require(enumerated == candidates["maximum_candidates_per_mapping"],
            "declared SpMV candidate budget does not match candidate grid")
    require(len(protocol["formal_cases"]) == 6, "six formal paired cases are required")
    require(len(protocol["timing_boundaries"]) == 7, "seven timing boundaries are required")
    require(protocol["rsf_ablations"][0] == "original" and
            protocol["rsf_ablations"][-1] == "restored_original",
            "RSF ablation must begin and end with the original path")
    require("aging_rsf_c3" in correctness["operators"], "RSF correctness contract missing")
    require("csr_spmv" in correctness["operators"], "SpMV correctness contract missing")

    matrix_paths = {
        "bcsstk17": ROOT / "data/suitesparse/bcsstk17/bcsstk17.mtx",
        "poisson3Da": ROOT / "data/suitesparse/poisson3Da/poisson3Da.mtx",
        "dw8192": ROOT / "data/suitesparse/expanded/dw8192.tar.gz",
        "apache2": ROOT / "data/suitesparse/expanded/apache2.tar.gz",
        "orani678": ROOT / "data/suitesparse/expanded/orani678.tar.gz",
        "sme3Da": ROOT / "data/suitesparse/expanded/sme3Da.tar.gz",
    }
    for matrix in tuning | holdout | formal:
        require(matrix in matrix_paths, f"no registered path for matrix {matrix}")
        require(matrix_paths[matrix].is_file(), f"missing matrix asset: {matrix_paths[matrix]}")

    print(json.dumps({
        "status": "PASS",
        "protocol": str(args.protocol.relative_to(ROOT)),
        "correctness": str(args.correctness.relative_to(ROOT)),
        "tuning": sorted(tuning),
        "formal_representatives": sorted(formal),
        "revision_holdout": sorted(holdout),
        "candidate_count_per_mapping": enumerated,
    }, indent=2))


if __name__ == "__main__":
    main()
