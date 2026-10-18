#!/usr/bin/env python3
"""Extract named 60/22 kernels from the separate BW1000 PMC job."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COUNTERS = ("SQ_INSTS_VALU", "SQ_INSTS_VMEM_RD", "SQ_INSTS_VMEM_WR")
RESOURCES = ("arch_vgpr", "sgpr", "scr")


def load_csv(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def select(rows: list[dict], backend: str):
    if backend == "hip":
        pre = [row for row in rows if "fixed_replay_kernel" in row["KernelName"]]
        bounded = [row for row in rows if "bounded_replay22_kernel" in row["KernelName"]]
        if len(pre) != 1 or len(bounded) != 1:
            raise ValueError(f"HIP kernel identification failed: {len(pre)}, {len(bounded)}")
        return pre[0], bounded[0]
    fixed = [row for row in rows if row["KernelName"] == "fixed_replay_kernel"]
    if len(fixed) != 2 or int(fixed[0]["Index"]) >= int(fixed[1]["Index"]):
        raise ValueError(f"Triton 60/22 dispatch identification failed: {len(fixed)}")
    # The profiled --smoke path launches 60 first, then 22, with no extra fixed kernel.
    return fixed[0], fixed[1]


def summarize(row: dict):
    out = {"dispatch_index": int(row["Index"]), "kernel_name": row["KernelName"],
           "grid": int(row["grd"]), "workgroup": int(row["wgr"])}
    out.update({key: int(row[key]) for key in COUNTERS + RESOURCES})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=int, required=True)
    parser.add_argument("--formal-job", type=int, required=True,
                        help="one formal task job ID for frozen-source comparison")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folder = ROOT / f"results/raw/bw1000/revision-v2-rsf-c3-loop-bound-pmc-{args.job}"
    formal = ROOT / f"results/raw/bw1000/revision-v2-rsf-c3-loop-bound-formal-{args.formal_job}"
    if "RSF_C3_LOOP_BOUND_PMC_V3_PASS" not in (folder / f"rsf-c3-bound-pmc-v3-{args.job}.out").read_text():
        raise ValueError("PMC job completion marker missing")
    source_hashes = {}
    for path in (folder / "source.sha256", formal / "source.sha256"):
        entries = {}
        for line in path.read_text().splitlines():
            sha, name = line.split("  ", 1)
            entries[Path(name).name] = sha
        source_hashes[path.name + ":" + path.parent.name] = entries
    manifests = list(source_hashes.values())
    for name in ("rsf_c3_fixed_replay_native_v2.cpp", "rsf_c3_fixed_replay_triton_v2.py",
                 "rsf_c3_loop_bound_native_v3.cpp", "rsf_c3_loop_bound_triton_v3.py",
                 "aging_rsf_fp64_reference.py", "src03-aging-rsf-newton-fp64-input-v1.tsv"):
        if manifests[0].get(name) != manifests[1].get(name):
            raise ValueError(f"PMC/formal source mismatch: {name}")

    backends = {}
    for backend, stem in (("hip", "native"), ("triton", "triton")):
        pre_row, bounded_row = select(load_csv(folder / f"{stem}-pmc.csv"), backend)
        pre, bounded = summarize(pre_row), summarize(bounded_row)
        if pre["grid"] != bounded["grid"] or pre["workgroup"] != bounded["workgroup"]:
            raise ValueError(f"mapping changed within {backend}")
        backends[backend] = {"pre60": pre, "bound22": bounded,
                             "valu_bound22_over_pre60": bounded["SQ_INSTS_VALU"] / pre["SQ_INSTS_VALU"]}
    result = {"schema_version": "rsf-c3-loop-bound-pmc/3.0", "platform": "BW1000",
              "job_id": args.job, "status": "PASS_DIRECTIONAL_PROFILE",
              "profile_time_used_for_speedup": False,
              "source_hashes_match_formal": True,
              "raw_directory": str(folder.relative_to(ROOT)), "backends": backends}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: {"valu_pre60": v["pre60"]["SQ_INSTS_VALU"],
                          "valu_bound22": v["bound22"]["SQ_INSTS_VALU"],
                          "bound22_over_pre60": v["valu_bound22_over_pre60"]}
                      for k, v in backends.items()}, indent=2))


if __name__ == "__main__":
    main()
