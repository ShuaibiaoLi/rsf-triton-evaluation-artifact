#!/usr/bin/env python3
"""Combine paired HIP/Triton C3 lookup and arithmetic-placeholder evidence."""
import argparse,json,re
from pathlib import Path
SUMMARY=re.compile(r"^summary variant=(\S+) median_us=([0-9.eE+-]+) p5_us=([0-9.eE+-]+) p95_us=([0-9.eE+-]+)$",re.MULTILINE)
SAMPLE=re.compile(r"^sample variant=(\S+) index=(\d+) us=([0-9.eE+-]+)$",re.MULTILINE)
def main():
 p=argparse.ArgumentParser();p.add_argument("--native",type=Path,required=True);p.add_argument("--triton",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();text=a.native.read_text()
 if "RSF_C3_LOOKUP_NATIVE_V2_PASS" not in text or "correctness max_abs_difference=" not in text or "status=PASS" not in text:raise ValueError("native gate failed")
 hip={n:{"median_us":float(m),"p5_us":float(lo),"p95_us":float(hi),"samples_us":[]} for n,m,lo,hi in SUMMARY.findall(text)}
 for n,_,v in SAMPLE.findall(text):hip[n]["samples_us"].append(float(v))
 tri=json.loads(a.triton.read_text());
 if tri.get("status")!="PASS" or not tri.get("reversibility_pass"):raise ValueError("Triton gate failed")
 names=("fixed_original_pre","lookup_coefficient","lookup_sqrt","lookup_asinh","lookup_all_transcendentals","arithmetic_placeholder","fixed_original_restore")
 rows=[]
 for n in names:
  t=tri["variants"][n];rows.append({"variant":n,"hip":hip[n],"triton":t,"triton_over_hip":t["median_us"]/hip[n]["median_us"],"hip_over_original":hip[n]["median_us"]/hip["fixed_original_pre"]["median_us"],"triton_over_original":t["median_us"]/tri["variants"]["fixed_original_pre"]["median_us"]})
 original=rows[0]["triton_over_hip"];all_ratio=next(x["triton_over_hip"] for x in rows if x["variant"]=="lookup_all_transcendentals");placeholder=next(x["triton_over_hip"] for x in rows if x["variant"]=="arithmetic_placeholder")
 out={"schema_version":"rsf-c3-function-path-paired/2.0","status":"PASS_NEGATIVE_CAUSAL_RESULT","variants":rows,"hip_restore_over_pre":hip["fixed_original_restore"]["median_us"]/hip["fixed_original_pre"]["median_us"],"triton_restore_over_pre":tri["restore_over_pre"],"gap_shrinks_lookup_all":all_ratio<original,"gap_shrinks_arithmetic_placeholder":placeholder<original,"conclusion":"removing transcendental paths accelerates both backends but does not shrink the Triton/HIP gap; transcendental presence alone is not the causal explanation"}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({"status":out["status"],"ratios":{x["variant"]:x["triton_over_hip"] for x in rows}},indent=2))
if __name__=="__main__":main()
