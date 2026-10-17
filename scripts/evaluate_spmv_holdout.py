#!/usr/bin/env python3
"""Evaluate a frozen constant-default SpMV rule once on the revision holdout."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--rule",type=Path,required=True)
    parser.add_argument("--ledger",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); rule=json.loads(args.rule.read_text()); ledger=json.loads(args.ledger.read_text())
    if rule.get("status")!="FROZEN": raise ValueError("rule is not frozen")
    if ledger.get("status")!="PASS" or ledger.get("candidate_count")!=48: raise ValueError("holdout ledger must be a complete 48-candidate PASS")
    matrices={record["matrix"] for record in ledger["records"]}
    if matrices!={"sme3Da"}: raise ValueError(f"unexpected holdout set: {matrices}")
    evaluations=[]
    for backend in ("hip","triton"):
        records=[record for record in ledger["records"] if record["backend"]==backend]
        default=rule["constant_default_per_backend"][backend]
        predicted=next(record for record in records if record["mapping"]==default["mapping"] and
                       record["block_size"]==default["block_size"] and
                       record["num_warps_or_wave_groups"]==default["num_warps_or_wave_groups"])
        best=min(records,key=lambda item:(item["median_us"],item["mapping"],item["block_size"],item["num_warps_or_wave_groups"]))
        evaluations.append({"backend":backend,"predicted":{k:predicted[k] for k in ("mapping","block_size","num_warps_or_wave_groups","median_us")},
                            "oracle_best":{k:best[k] for k in ("mapping","block_size","num_warps_or_wave_groups","median_us")},
                            "top1_hit":all(predicted[k]==best[k] for k in ("mapping","block_size","num_warps_or_wave_groups")),
                            "regret":predicted["median_us"]/best["median_us"]-1.0})
    output={"schema_version":"spmv-holdout-evaluation/1.0","status":"PASS","holdout":"sme3Da",
            "rule_sha256":sha256(args.rule),"ledger_sha256":sha256(args.ledger),"evaluations":evaluations,
            "rule_modified_after_holdout":False,
            "scope":"revision holdout; not pristine because the matrix appeared in prior exploratory work"}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps(output,indent=2))


if __name__=="__main__":main()
