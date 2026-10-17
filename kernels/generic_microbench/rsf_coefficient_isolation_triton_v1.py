#!/usr/bin/env python3
"""Separate log and exp interventions in the invariant RSF coefficient path."""
import argparse,json,math,statistics
from pathlib import Path
import torch,triton,triton.language as tl

@triton.jit
def coefficient_kernel(state,a,b,f0,v0,length,log_table,exp_table,out,dummy,n:tl.constexpr,block:tl.constexpr,mode:tl.constexpr):
 o=tl.program_id(0)*block+tl.arange(0,block);m=o<n
 ps=tl.load(state+o,mask=m,other=1.);aa=tl.load(a+o,mask=m,other=1.);bb=tl.load(b+o,mask=m,other=0.);ff=tl.load(f0+o,mask=m,other=0.);vv=tl.load(v0+o,mask=m,other=1.);ll=tl.load(length+o,mask=m,other=1.)
 if mode==1:ls=tl.load(log_table+o,mask=m,other=0.)
 else:ls=tl.log(vv*ps/ll)
 exponent=(ff+bb*ls)/aa
 if mode==2:ev=tl.load(exp_table+o,mask=m,other=1.)
 else:ev=tl.exp(exponent)
 tl.store(out+o,.5/vv*ev,mask=m);tl.store(dummy+o,ls,mask=m)

def q(x,p):
 x=sorted(x);z=(len(x)-1)*p;l=int(z);h=min(l+1,len(x)-1);return x[l]*(h-z)+x[h]*(z-l)
def timing(fn,warm,repeat,batch):
 for _ in range(warm):fn()
 torch.cuda.synchronize();v=[]
 for _ in range(repeat):
  a,b=torch.cuda.Event(True),torch.cuda.Event(True);a.record()
  for _ in range(batch):fn()
  b.record();b.synchronize();v.append(a.elapsed_time(b)*1000./batch)
 return {'median_us':statistics.median(v),'p5_us':q(v,.05),'p95_us':q(v,.95),'samples_us':v}
def main():
 p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=65537);p.add_argument('--warmup',type=int,default=10);p.add_argument('--repeat',type=int,default=30);p.add_argument('--batch',type=int,default=128);p.add_argument('--output',type=Path,required=True);x=p.parse_args()
 base=((.2,.01,.015,.6,1e-6,.02),(2e4,.012,.018,.6,1e-6,.02),(2e4,.012,.018,.6,1e-6,.02),(.02,.008,.012,.58,1e-6,.01),(2e4,.01,.015,.6,1e-6,.02),(.2,.004,.018,.6,1e-6,.01),(.2,.01,.015,.6,1e-6,.02),(.2,.01,.015,.6,1e-6,.02),(.05,.009,.014,.59,1e-6,1e-4))
 cols=list(zip(*(base[i%9] for i in range(x.n))));t=[torch.tensor(v,dtype=torch.float64,device='cuda') for v in cols];logs=[math.log(r[4]*r[0]/r[5]) for r in (base[i%9] for i in range(x.n))];exps=[math.exp((r[3]+r[2]*math.log(r[4]*r[0]/r[5]))/r[1]) for r in (base[i%9] for i in range(x.n))];lt=torch.tensor(logs,dtype=torch.float64,device='cuda');et=torch.tensor(exps,dtype=torch.float64,device='cuda');out=torch.empty(x.n,dtype=torch.float64,device='cuda');dummy=torch.empty_like(out)
 def launch(mode):coefficient_kernel[(triton.cdiv(x.n,64),)](*t,lt,et,out,dummy,n=x.n,block=64,mode=mode,num_warps=1)
 variants={};errors={}
 for name,mode in (('original_pre',0),('lookup_log',1),('lookup_exp_retain_log',2),('original_restore',0)):
  launch(mode);torch.cuda.synchronize();eo=max(float(torch.max(torch.abs(out-torch.tensor([.5/base[i%9][4]*exps[i] for i in range(x.n)],dtype=torch.float64,device='cuda'))).cpu()),float(torch.max(torch.abs(dummy-lt)).cpu()));errors[name]=eo
  if eo>1e-10:raise RuntimeError((name,eo))
  variants[name]=timing(lambda m=mode:launch(m),x.warmup,x.repeat,x.batch)
 restore=variants['original_restore']['median_us']/variants['original_pre']['median_us'];result={'schema_version':'rsf-coefficient-isolation-triton/1.0','status':'PASS','n':x.n,'warmup':x.warmup,'repeat':x.repeat,'batch':x.batch,'errors':errors,'variants':variants,'restore_over_pre':restore,'reversibility_pass':.9<=restore<=1.1}
 if not result['reversibility_pass']:raise RuntimeError(restore)
 x.output.parent.mkdir(parents=True,exist_ok=True);x.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'PASS','medians':{k:v['median_us'] for k,v in variants.items()},'restore':restore}))
if __name__=='__main__':main()
