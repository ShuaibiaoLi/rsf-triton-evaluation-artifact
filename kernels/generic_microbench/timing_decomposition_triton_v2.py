#!/usr/bin/env python3
"""Triton timing-boundary decomposition for the six revision-v1 cases."""

import argparse
import importlib.util
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

from spmv_four_cell_triton_v1 import csr_block_rows, csr_row_program, parse_mtx


@triton.jit
def empty_kernel(DUMMY: tl.constexpr):
    tl.static_assert(DUMMY == 1)


@triton.jit
def stencil7(x, y, n: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = index < n
    left = tl.maximum(index - 1, 0)
    right = tl.minimum(index + 1, n - 1)
    value = 0.5 * tl.load(x + index, mask=active, other=0.0)
    value += 0.25 * (tl.load(x + left, mask=active, other=0.0) + tl.load(x + right, mask=active, other=0.0))
    tl.store(y + index, value, mask=active)


@triton.jit
def row_reduce256(x, y, rows: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, 256)
    values = tl.load(x + row * 256 + lane, mask=row < rows, other=0.0)
    tl.store(y + row, tl.sum(values, axis=0), mask=row < rows)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] if low == high else ordered[low] * (high-position) + ordered[high] * (position-low)


def sample_host(fn, repeat):
    values=[]
    for _ in range(repeat):
        torch.cuda.synchronize(); start=time.perf_counter(); fn(); torch.cuda.synchronize()
        values.append((time.perf_counter()-start)*1e6)
    return values


def sample_event(fn, repeat, batch=1):
    values=[]
    for _ in range(repeat):
        begin=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(batch): fn()
        end.record(); end.synchronize(); values.append(begin.elapsed_time(end)*1e3/batch)
    return values


def stats(values):
    return {"median_us":statistics.median(values),"p5_us":percentile(values,.05),
            "p95_us":percentile(values,.95),"samples_us":values}


def graph_measure(fn, repeat):
    try:
        graph=torch.cuda.CUDAGraph()
        capture=torch.cuda.Stream()
        capture.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture):
            fn()
        torch.cuda.current_stream().wait_stream(capture); torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=capture): fn()
        graph.replay(); torch.cuda.synchronize()
        return {"status":"PASS",**stats(sample_event(graph.replay,repeat))}
    except Exception as error:
        return {"status":"UNAVAILABLE","error_type":type(error).__name__,"error":str(error)[:500]}


def audit_case(name, fn, repeat, batches, correctness):
    torch.cuda.synchronize(); start=time.perf_counter(); fn(); torch.cuda.synchronize()
    first=(time.perf_counter()-start)*1e6
    for _ in range(10): fn()
    torch.cuda.synchronize()
    host=sample_host(fn,repeat); event=sample_event(fn,repeat)
    return {"name":name,"correctness":correctness,"first_call_including_jit_us":first,
            "post_jit_host_submit_to_sync":stats(host),"single_device_event":stats(event),
            "steady_batch_per_call":{str(batch):stats(sample_event(fn,repeat,batch)) for batch in batches},
            "graph_replay":graph_measure(fn,repeat),
            "profiler_kernel_start_end":{"status":"NOT_COLLECTED_SEPARATE_PROFILER_JOB_REQUIRED"}}


def setup_stencil(n):
    host=torch.arange(n,dtype=torch.float64)/10000.0
    expected=.5*host+.25*(torch.cat((host[:1],host[:-1]))+torch.cat((host[1:],host[-1:])))
    x=host.cuda(); y=torch.empty_like(x)
    fn=lambda:stencil7[(triton.cdiv(n,256),)](x,y,n=n,BLOCK=256,num_warps=4)
    fn_for_check=lambda:fn()
    fn_for_check();torch.cuda.synchronize();error=float(torch.max(torch.abs(y.cpu()-expected)))
    # Correctness launch above intentionally uses a different specialization cache directory
    # only when called by the caller after first-call timing; return deferred check metadata.
    return fn,{"max_abs_error":error,"tolerance":1e-12+1e-10*max(1.,float(torch.max(torch.abs(expected))))}


def setup_stencil_deferred(n):
    host=torch.arange(n,dtype=torch.float64)/10000.0
    expected=.5*host+.25*(torch.cat((host[:1],host[:-1]))+torch.cat((host[1:],host[-1:])))
    x=host.cuda(); y=torch.empty_like(x)
    fn=lambda:stencil7[(triton.cdiv(n,256),)](x,y,n=n,BLOCK=256,num_warps=4)
    def check():
        error=float(torch.max(torch.abs(y.cpu()-expected))); tolerance=1e-12+1e-10*max(1.,float(torch.max(torch.abs(expected))))
        return {"max_abs_error":error,"tolerance":tolerance,"status":"PASS" if error<=tolerance else "FAIL"}
    return fn,check


def setup_reduction(rows):
    host=torch.arange(rows*256,dtype=torch.float64).reshape(rows,256)/100000.0
    expected=torch.sum(host,dim=1); x=host.cuda(); y=torch.empty(rows,dtype=torch.float64,device="cuda")
    fn=lambda:row_reduce256[(rows,)](x,y,rows=rows,num_warps=4)
    def check():
        error=float(torch.max(torch.abs(y.cpu()-expected))); tolerance=1e-12+1e-10*max(1.,float(torch.max(torch.abs(expected))))
        return {"max_abs_error":error,"tolerance":tolerance,"status":"PASS" if error<=tolerance else "FAIL"}
    return fn,check


def setup_spmv(path, rule):
    rows,cols,rowptr,columns,values=parse_mtx(path); lengths=[rowptr[i+1]-rowptr[i] for i in range(rows)]
    kmax=1 << (max(lengths)-1).bit_length(); config=rule["constant_default_per_backend"]["triton"]
    host_x=[math.sin(.001*i)+math.cos(.0003*i) for i in range(cols)]
    reference=[sum(values[p]*host_x[columns[p]] for p in range(rowptr[row],rowptr[row+1])) for row in range(rows)]
    rp=torch.tensor(rowptr,dtype=torch.int32,device="cuda");ci=torch.tensor(columns,dtype=torch.int32,device="cuda")
    vv=torch.tensor(values,dtype=torch.float64,device="cuda");x=torch.tensor(host_x,dtype=torch.float64,device="cuda");y=torch.empty(rows,dtype=torch.float64,device="cuda")
    block=config["block_size"];warps=config["num_warps_or_wave_groups"]
    requested=block//64; rpp=min(requested,max(1,2048//kmax))
    if config["mapping"]=="block_row": fn=lambda:csr_block_rows[(triton.cdiv(rows,rpp),)](rp,ci,vv,x,y,rows=rows,KMAX=kmax,ROWS_PER_PROGRAM=rpp,num_warps=warps)
    else: fn=lambda:csr_row_program[(rows,)](rp,ci,vv,x,y,rows=rows,KMAX=kmax,CHUNK=block,num_warps=warps)
    def check():
        observed=y.cpu().tolist();error=max(abs(a-b) for a,b in zip(observed,reference));scale=max(1.,max(map(abs,reference)));tolerance=1e-12+1e-10*scale
        return {"max_abs_error":error,"tolerance":tolerance,"status":"PASS" if error<=tolerance else "FAIL","mapping":config["mapping"]}
    return fn,check


def setup_rsf(rsf_root,size):
    probe_dir=rsf_root/"probes/triton/rsf"
    if str(probe_dir) not in sys.path: sys.path.insert(0,str(probe_dir))
    bench=load_module(rsf_root/"probes/triton/rsf/benchmark_c3_native_compare_v1.py","rsf_bench")
    probe=bench.load_probe(rsf_root/"probes/triton/rsf/aging_rsf_numerical_feasibility.py")
    rows=probe.read_tsv(rsf_root/"test-data/generated/src03-aging-rsf-newton-fp64-input-v1.tsv")
    floats,maximum=bench.inputs_for(probe,rows,size);outputs=[torch.empty(size,dtype=torch.float64,device="cuda") for _ in range(3)]+[torch.empty(size,dtype=torch.int32,device="cuda") for _ in range(2)]
    fn=lambda:bench.triton_c3(probe,floats,maximum,outputs,block=64,num_warps=1)
    def check():
        native=bench.native_c3(probe,floats,maximum);difference=bench.max_abs(native,outputs)
        passed=all(value<=1e-10 for value in difference[:3]) and all(value==0 for value in difference[3:])
        return {"max_abs_differences":difference,"status":"PASS" if passed else "FAIL"}
    return fn,check


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True);parser.add_argument("--rsf-root",type=Path,required=True)
    parser.add_argument("--rule",type=Path,required=True);parser.add_argument("--json",type=Path,required=True);parser.add_argument("--repeat",type=int,default=20);parser.add_argument("--batches",nargs="+",type=int,default=[1,8,32,128]);args=parser.parse_args()
    rule=json.loads(args.rule.read_text()); cases={"empty":(lambda:empty_kernel[(1,)](DUMMY=1),lambda:{"status":"PASS"})}
    cases["stencil_small"]=setup_stencil_deferred(257);cases["interior_stencil_large"]=setup_stencil_deferred(4194304);cases["row_reduction"]=setup_reduction(65537)
    cases["spmv_short_row"]=setup_spmv(args.root/"data/suitesparse/expanded/dw8192/dw8192.mtx",rule)
    cases["spmv_long_tail"]=setup_spmv(args.root/"data/suitesparse/expanded/orani678/orani678.mtx",rule);cases["rsf_c3"]=setup_rsf(args.rsf_root,65537)
    output={"schema_version":"timing-decomposition-triton/2.0","backend":"triton","device":torch.cuda.get_device_name(0),"torch":torch.__version__,"triton":triton.__version__,"repeat":args.repeat,"batches":args.batches,"cases":[]}
    for name,(fn,check) in cases.items():
        row=audit_case(name,fn,args.repeat,args.batches,{"status":"PENDING_AFTER_FIRST_LAUNCH"});row["correctness"]=check();output["cases"].append(row)
        if row["correctness"]["status"]!="PASS": raise RuntimeError(f"correctness failed: {name}")
    args.json.parent.mkdir(parents=True,exist_ok=True);args.json.write_text(json.dumps(output,indent=2)+"\n");print(json.dumps({"status":"PASS","cases":[row["name"] for row in output["cases"]]},indent=2))


if __name__=="__main__":main()
