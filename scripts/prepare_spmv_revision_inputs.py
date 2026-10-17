#!/usr/bin/env python3
"""Build deterministic CSR structure manifests for the revision-v1 SpMV sets.

The default command deliberately excludes the locked revision holdout.  Supplying
--include-holdout requires a frozen-rule artifact so the holdout cannot silently
enter rule formation.
"""

import argparse
import hashlib
import io
import json
import math
import statistics
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/protocol/revision_v1_experiment_protocol.json"

MATRICES = {
    "bcsstk17": ROOT / "data/suitesparse/bcsstk17/bcsstk17.mtx",
    "poisson3Da": ROOT / "data/suitesparse/poisson3Da/poisson3Da.mtx",
    "dw8192": ROOT / "data/suitesparse/expanded/dw8192.tar.gz",
    "apache2": ROOT / "data/suitesparse/expanded/apache2.tar.gz",
    "orani678": ROOT / "data/suitesparse/expanded/orani678.tar.gz",
    "sme3Da": ROOT / "data/suitesparse/expanded/sme3Da.tar.gz",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def open_matrix(path):
    if path.name.endswith(".tar.gz"):
        archive = tarfile.open(path, "r:gz")
        members = [m for m in archive.getmembers() if m.isfile() and m.name.endswith(".mtx")]
        dataset = path.name.removesuffix(".tar.gz")
        primary = [m for m in members if Path(m.name).name == f"{dataset}.mtx"]
        if len(primary) != 1:
            archive.close()
            raise ValueError(f"{path}: expected one primary {dataset}.mtx member, got {len(primary)}")
        raw = archive.extractfile(primary[0])
        if raw is None:
            archive.close()
            raise ValueError(f"{path}: cannot read {primary[0].name}")
        text = io.TextIOWrapper(raw, encoding="utf-8", errors="strict")
        return archive, text, primary[0].name
    text = path.open(encoding="utf-8")
    return None, text, path.name


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def matrix_features(name, path):
    archive, handle, member = open_matrix(path)
    try:
        header = handle.readline().strip().lower()
        if not header.startswith("%%matrixmarket matrix coordinate"):
            raise ValueError(f"{path}: unsupported MatrixMarket header")
        symmetric = "symmetric" in header or "hermitian" in header
        dimensions = None
        for line in handle:
            if line.strip() and not line.startswith("%"):
                dimensions = tuple(map(int, line.split()[:3]))
                break
        if dimensions is None:
            raise ValueError(f"{path}: dimensions missing")
        rows, cols, stored_nnz = dimensions
        lengths = [0] * rows
        parsed = 0
        expanded = 0
        for line in handle:
            if not line.strip() or line.startswith("%"):
                continue
            row, col = map(int, line.split()[:2])
            if not 1 <= row <= rows or not 1 <= col <= cols:
                raise ValueError(f"{path}: coordinate out of range")
            lengths[row - 1] += 1
            parsed += 1
            expanded += 1
            if symmetric and row != col:
                lengths[col - 1] += 1
                expanded += 1
        if parsed != stored_nnz:
            raise ValueError(f"{path}: declared {stored_nnz} entries, parsed {parsed}")
    finally:
        handle.close()
        if archive is not None:
            archive.close()

    maximum = max(lengths, default=0)
    kmax = 1 if maximum == 0 else 1 << (maximum - 1).bit_length()
    padded_items = rows * kmax
    mean = statistics.fmean(lengths) if lengths else 0.0
    cv = statistics.pstdev(lengths) / mean if mean else 0.0
    return {
        "name": name,
        "asset": str(path.relative_to(ROOT)),
        "asset_sha256": sha256(path),
        "archive_member": member,
        "rows": rows,
        "cols": cols,
        "stored_nnz": stored_nnz,
        "nnz": expanded,
        "symmetric_expansion": symmetric,
        "mean_row_nnz": mean,
        "p95_row_nnz": percentile(lengths, 0.95),
        "p99_row_nnz": percentile(lengths, 0.99),
        "max_row_nnz": maximum,
        "row_nnz_cv": cv,
        "kmax": kmax,
        "padded_items": padded_items,
        "padding_amplification": padded_items / expanded if expanded else None,
        "effective_lane_ratio": expanded / padded_items if padded_items else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-holdout", action="store_true")
    parser.add_argument("--frozen-rule", type=Path)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    splits = protocol["splits"]
    selected = [*splits["tuning"], *splits["formal_representatives"]]
    if args.include_holdout:
        if args.frozen_rule is None or not args.frozen_rule.is_file():
            parser.error("--include-holdout requires an existing --frozen-rule artifact")
        selected.extend(splits["revision_holdout"])
    records = [matrix_features(name, MATRICES[name]) for name in selected]
    output = {
        "schema_version": "spmv-structure-manifest/1.0",
        "protocol_sha256": sha256(PROTOCOL),
        "holdout_included": args.include_holdout,
        "frozen_rule": str(args.frozen_rule) if args.frozen_rule else None,
        "matrices": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output),
                      "matrices": selected, "holdout_included": args.include_holdout}, indent=2))


if __name__ == "__main__":
    main()
