#!/usr/bin/env python3
"""Aggregate independent-job SpMV ratios without treating kernel calls as replicates."""

import argparse
import json
import math
import random
import re
from pathlib import Path


RUN_KEY = re.compile(r"paired-(\d+_\d+)$")


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered)-1)*fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] if low == high else ordered[low]*(high-position)+ordered[high]*(position-low)


def bootstrap_median_ci(values, seed=20261015, draws=10000):
    generator=random.Random(seed); size=len(values); medians=[]
    for _ in range(draws):
        sample=[values[generator.randrange(size)] for _ in range(size)]
        medians.append(percentile(sample,0.5))
    return [percentile(medians,0.025),percentile(medians,0.975)]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--summary", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args=parser.parse_args()
    if len(args.summary)<5: raise ValueError("at least five independent job summaries are required")
    jobs=[]
    for path in args.summary:
        result=json.loads(path.read_text())
        if result.get("status")!="PASS": raise ValueError(f"non-PASS job summary: {path}")
        match=RUN_KEY.search(path.parent.name)
        if not match: raise ValueError(f"cannot derive job key from {path}")
        jobs.append({"job_key":match.group(1),"path":str(path),"comparisons":result["comparisons"]})
    if len({job["job_key"] for job in jobs})!=len(jobs): raise ValueError("duplicate independent job key")
    matrices=sorted({row["matrix"] for job in jobs for row in job["comparisons"]})
    metrics=["hip_cooperative_over_thread","triton_block_over_row","row_level_triton_over_hip",
             "cooperative_triton_over_hip","best_triton_over_best_hip"]
    aggregate=[]
    for matrix in matrices:
        for metric in metrics:
            values=[]
            for job in jobs:
                row=next(item for item in job["comparisons"] if item["matrix"]==matrix)
                values.append(row[metric])
            aggregate.append({"matrix":matrix,"metric":metric,"independent_jobs":len(values),
                              "job_values":values,"median":percentile(values,0.5),
                              "p5":percentile(values,0.05),"p95":percentile(values,0.95),
                              "bootstrap_median_ci95":bootstrap_median_ci(values)})
    output={"schema_version":"spmv-four-cell-independent-jobs/1.0","status":"PASS",
            "experimental_unit":"one independent Slurm array task; within-task median device-event latency",
            "kernel_calls_are_not_counted_as_independent_replicates":True,"jobs":jobs,"aggregate":aggregate}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps({"status":"PASS","independent_jobs":len(jobs),"aggregate":aggregate},indent=2))


if __name__=="__main__": main()
