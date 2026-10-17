#!/usr/bin/env python3
import csv, hashlib, json, statistics
from pathlib import Path

root = Path(__file__).resolve().parent
non = json.loads((root / "results/bw1000/revision_v1/formal/nonspmv-independent-jobs-v1.json").read_text())
spmv = json.loads((root / "results/bw1000/revision_v1/formal/spmv-four-cell-independent-jobs-v1.json").read_text())

def close(name, got, expected, tol=5e-4):
    if abs(got - expected) > tol:
        raise SystemExit(f"FAIL {name}: {got} != {expected}")
    print(f"PASS {name}: {got:.6f}")

for case, expected in [("interior_stencil_large", 1.0020894474), ("row_reduction", 0.9962882953), ("rsf_c3", 1.9835988294)]:
    rows = [next(r for r in j["rows"] if r["case"] == case)["triton_over_hip"] for j in non["jobs"]]
    close(case, statistics.median(rows), expected)

rows = [next(c for c in j["comparisons"] if c["matrix"] == "orani678") for j in spmv["jobs"]]
close("orani678 simple row-level", statistics.median(r["row_level_triton_over_hip"] for r in rows), 0.2716660735)
close("orani678 registered best/best", statistics.median(r["best_triton_over_best_hip"] for r in rows), 1.9580515116)

dongfang = json.loads((root / "results/dongfang/revision_v1/formal/spmv-four-cell-formal-local-replay.json").read_text())
df_orani = next(r for r in dongfang["comparisons"] if r["matrix"] == "orani678")
close("Dongfang orani678 simple row-level", df_orani["row_level_triton_over_hip"], 0.3107578764)
close("Dongfang orani678 registered best/best", df_orani["best_triton_over_best_hip"], 4.6619427125)

abl = json.loads((root / "results/bw1000/revision_v1/ablation/rsf-c3-function-path-paired-v2.json").read_text())
variants = {v["variant"]: v for v in abl["variants"]}
close("RSF fixed original ratio", variants["fixed_original_pre"]["triton_over_hip"], 3.8928572484)
close("RSF lookup-all ratio", variants["lookup_all_transcendentals"]["triton_over_hip"], 5.6384979471)
if abl["gap_shrinks_lookup_all"] or abl["gap_shrinks_arithmetic_placeholder"]:
    raise SystemExit("FAIL RSF negative causal conclusion")
print("PASS RSF reversible negative causal conclusion")

for backend, filename in [("Triton", "triton-kernel-timeline-v2.json"), ("HIP", "hip-generic-kernel-timeline-v2.json")]:
    timeline = json.loads((root / "results/bw1000/revision_v1/timing" / filename).read_text())
    empty = next(k for k in timeline["kernels"] if "empty" in k["kernel"])
    close(f"{backend} empty-kernel profiler duration us", empty["median_duration_ns"] / 1000.0, 0.64, 1e-9)

with (root / "docs/paper/rewrite_20261016/source_data/figure6_tpv101_application_context.csv").open() as handle:
    tpv = list(csv.DictReader(handle))
close("TPV101 Dynamic Rupture compute-kernel fraction percent", float(tpv[2]["rsf_fraction_percent"]), 33.4, 1e-9)
close("CPU RSF-path proxy fraction percent", float(tpv[0]["rsf_fraction_percent"]), 47.6, 1e-9)
close("BW1000 RSF-path proxy fraction percent", float(tpv[1]["rsf_fraction_percent"]), 27.0, 1e-9)

fig_manifest = json.loads((root / "figure_source_manifest.json").read_text())
figures = [entry["figure"] for entry in fig_manifest["entries"]]
if figures != list(range(1, 8)) or len({entry["source_data"] for entry in fig_manifest["entries"]}) != 7:
    raise SystemExit("FAIL unique Figure 1-7 source mapping")
for entry in fig_manifest["entries"]:
    for key in ("source_data", "png", "svg", "pdf"):
        if not (root / entry[key]).is_file():
            raise SystemExit(f"FAIL missing figure artifact: {entry[key]}")
print("PASS unique Figure 1-7 source mapping")

for line in (root / "SHA256SUMS").read_text().splitlines():
    expected, rel = line.split("  ", 1)
    got = hashlib.sha256((root / rel).read_bytes()).hexdigest()
    if got != expected:
        raise SystemExit(f"FAIL checksum: {rel}")
print("PASS all checksums")
