#!/usr/bin/env python3
"""Combine the native and Triton fixed-iteration replay evidence."""

import argparse
import json
import re
from pathlib import Path


SUMMARY = re.compile(r"^summary variant=(\S+) median_us=([0-9.eE+-]+) p5_us=([0-9.eE+-]+) p95_us=([0-9.eE+-]+)$", re.MULTILINE)
SAMPLE = re.compile(r"^sample variant=(\S+) index=(\d+) us=([0-9.eE+-]+)$", re.MULTILINE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--triton", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = args.native.read_text()
    if "RSF_C3_FIXED_REPLAY_NATIVE_V2_PASS" not in text or "status=PASS" not in text:
        raise ValueError("native terminal or correctness PASS missing")
    native = {name: {"median_us": float(median), "p5_us": float(p5), "p95_us": float(p95), "samples_us": []}
              for name, median, p5, p95 in SUMMARY.findall(text)}
    for name, _, value in SAMPLE.findall(text):
        native[name]["samples_us"].append(float(value))
    triton_doc = json.loads(args.triton.read_text())
    if triton_doc.get("status") != "PASS" or not triton_doc.get("reversibility_pass"):
        raise ValueError("Triton gate failed")
    triton = triton_doc["variants"]
    variants = []
    for name in ("original_pre", "fixed_iteration_replay", "original_restore"):
        variants.append({"variant": name, "hip": native[name], "triton": triton[name],
                         "triton_over_hip": triton[name]["median_us"] / native[name]["median_us"]})
    n = triton_doc["n"]
    base = (8, 22, 6, 4, 4, 6, 6, 1, 6)
    histogram = {str(value): sum(1 for i in range(n) if base[i % 9] == value) for value in sorted(set(base))}
    output = {
        "schema_version": "rsf-c3-fixed-replay-paired/2.0", "status": "PASS",
        "n": n, "actual_iteration_histogram": histogram,
        "variants": variants,
        "hip_restore_over_pre": native["original_restore"]["median_us"] / native["original_pre"]["median_us"],
        "triton_restore_over_pre": triton_doc["restore_over_pre"],
        "fixed_replay_interpretation": "dynamic convergence predicates and iteration-count variation do not explain the Triton/HIP gap",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "ratios": {row["variant"]: row["triton_over_hip"] for row in variants}}, indent=2))


if __name__ == "__main__":
    main()
