#!/usr/bin/env python3
"""Combine paired HIP/Triton log/exp coefficient-isolation evidence."""
import argparse,json,re
from pathlib import Path
R=re.compile(r"^summary variant=(\S+) median_us=([0-9.eE+-]+) p5_us=([0-9.eE+-]+) p95_us=([0-9.eE+-]+)$",re.MULTILINE)
S=re.compile(r"^sample variant=(\S+) index=(\d+) us=([0-9.eE+-]+)$",re.MULTILINE)
def main():
 p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--triton',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();text=a.native.read_text()
 if 'RSF_COEFFICIENT_ISOLATION_NATIVE_V1_PASS' not in text or 'status=PASS' not in text:raise ValueError('native gate failed')
 hip={n:{'median_us':float(m),'p5_us':float(lo),'p95_us':float(hi),'samples_us':[]} for n,m,lo,hi in R.findall(text)}
 for n,_,v in S.findall(text):hip[n]['samples_us'].append(float(v))
 tri=json.loads(a.triton.read_text());
 if tri.get('status')!='PASS' or not tri.get('reversibility_pass'):raise ValueError('Triton gate failed')
 names=('original_pre','lookup_log','lookup_exp_retain_log','original_restore');rows=[]
 for n in names:
  t=tri['variants'][n];rows.append({'variant':n,'hip':hip[n],'triton':t,'triton_over_hip':t['median_us']/hip[n]['median_us']})
 out={'schema_version':'rsf-coefficient-isolation-paired/1.0','status':'PASS','scope':'invariant coefficient subexpression; not additive with full C3 latency','variants':rows,'hip_restore_over_pre':hip['original_restore']['median_us']/hip['original_pre']['median_us'],'triton_restore_over_pre':tri['restore_over_pre']}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':'PASS','ratios':{x['variant']:x['triton_over_hip'] for x in rows}},indent=2))
if __name__=='__main__':main()
