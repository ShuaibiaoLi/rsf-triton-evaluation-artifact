#!/usr/bin/env python3
"""Frozen-trajectory function-path lookup ablations for Triton RSF C3."""

import argparse
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


FLOAT_NAMES = (
    "shear_traction", "effective_normal_stress", "state_reference", "direct_effect_a",
    "evolution_effect_b", "reference_friction", "reference_velocity",
    "state_evolution_distance", "shear_impedance", "previous_velocity", "newton_tolerance",
)
STEPS = 61


@triton.jit
def lookup_kernel(tau_ptr, sigma_ptr, state_ptr, a_ptr, b_ptr, f0_ptr,
                  v0_ptr, length_ptr, eta_ptr, initial_v_ptr, count_ptr,
                  converged_target_ptr, coefficient_table_ptr, sqrt_table_ptr,
                  asinh_table_ptr, velocity_ptr, friction_ptr, residual_ptr,
                  iterations_ptr, converged_ptr, n: tl.constexpr,
                  block: tl.constexpr, loop_count: tl.constexpr,
                  mode: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < n
    tau = tl.load(tau_ptr + offsets, mask=mask, other=0.0)
    sigma = tl.load(sigma_ptr + offsets, mask=mask, other=1.0)
    state = tl.load(state_ptr + offsets, mask=mask, other=1.0)
    a = tl.load(a_ptr + offsets, mask=mask, other=1.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    f0 = tl.load(f0_ptr + offsets, mask=mask, other=0.0)
    v0 = tl.load(v0_ptr + offsets, mask=mask, other=1.0)
    length = tl.load(length_ptr + offsets, mask=mask, other=1.0)
    eta = tl.load(eta_ptr + offsets, mask=mask, other=1.0)
    velocity = tl.maximum(tl.load(initial_v_ptr + offsets, mask=mask, other=1.0), 1.0e-45)
    count = tl.load(count_ptr + offsets, mask=mask, other=0)
    target_converged = tl.load(converged_target_ptr + offsets, mask=mask, other=0) != 0
    base_index = offsets % 9
    if mode == 5:
        coefficient_seed = f0 + b * state
        coefficient = 1.0 + coefficient_seed / (1.0 + tl.abs(coefficient_seed))
    elif mode == 1 or mode == 4:
        coefficient = tl.load(coefficient_table_ptr + base_index, mask=mask, other=1.0)
    else:
        coefficient = 0.5 / v0 * tl.exp((f0 + b * tl.log(v0 * state / length)) / a)
    inv_eta = 1.0 / eta
    for iteration in range(loop_count):
        argument = velocity * coefficient
        table_index = base_index * 61 + iteration
        if mode == 5:
            root = 1.0 + tl.abs(argument)
        elif mode == 2 or mode == 4:
            root = tl.load(sqrt_table_ptr + table_index, mask=mask, other=1.0)
        else:
            root = tl.sqrt(argument * argument + 1.0)
        if mode == 5:
            asinh_value = argument / (1.0 + tl.abs(argument))
        elif mode == 3 or mode == 4:
            asinh_value = tl.load(asinh_table_ptr + table_index, mask=mask, other=0.0)
        else:
            asinh_value = tl.log(argument + root)
        friction = a * asinh_value
        derivative = a * coefficient / root
        residual = (tau - sigma * friction) * inv_eta - velocity
        evaluating = mask & (iteration < count)
        is_last = iteration + 1 == count
        update = evaluating & ((~is_last) | (~target_converged))
        derivative_g = -(sigma * inv_eta) * derivative - 1.0
        candidate = tl.maximum(velocity - residual / derivative_g, 1.0e-45)
        velocity = tl.where(update, candidate, velocity)
    argument = velocity * coefficient
    final_index = base_index * 61 + count
    if mode == 5:
        root = 1.0 + tl.abs(argument)
    elif mode == 2 or mode == 4:
        root = tl.load(sqrt_table_ptr + final_index, mask=mask, other=1.0)
    else:
        root = tl.sqrt(argument * argument + 1.0)
    if mode == 5:
        asinh_value = argument / (1.0 + tl.abs(argument))
    elif mode == 3 or mode == 4:
        asinh_value = tl.load(asinh_table_ptr + final_index, mask=mask, other=0.0)
    else:
        asinh_value = tl.log(argument + root)
    friction = a * asinh_value
    residual = (tau - sigma * friction) * inv_eta - velocity
    tl.store(velocity_ptr + offsets, velocity, mask=mask)
    tl.store(friction_ptr + offsets, friction, mask=mask)
    tl.store(residual_ptr + offsets, residual, mask=mask)
    tl.store(iterations_ptr + offsets, count, mask=mask)
    tl.store(converged_ptr + offsets, target_converged.to(tl.int32), mask=mask)


def load_probe(path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("rsf_probe_lookup", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def trajectory_tables(rows):
    counts = (8, 22, 6, 4, 4, 6, 6, 1, 6)
    flags = (1, 1, 1, 1, 1, 1, 1, 0, 1)
    coefficients, roots, asinhs = [], [], []
    for index, row in enumerate(rows):
        tau=float(row["shear_traction"]);sigma=float(row["effective_normal_stress"]);state=float(row["state_reference"])
        a=float(row["direct_effect_a"]);b=float(row["evolution_effect_b"]);f0=float(row["reference_friction"])
        v0=float(row["reference_velocity"]);length=float(row["state_evolution_distance"]);eta=float(row["shear_impedance"])
        velocity=max(float(row["previous_velocity"]),1e-45)
        coefficient=.5/v0*math.exp((f0+b*math.log(v0*state/length))/a);coefficients.append(coefficient)
        row_roots=[];row_asinhs=[]
        for step in range(STEPS):
            argument=velocity*coefficient;root=math.sqrt(argument*argument+1.0);asinh=math.asinh(argument)
            row_roots.append(root);row_asinhs.append(asinh)
            if step < counts[index] and (step+1 < counts[index] or not flags[index]):
                friction=a*asinh;derivative=a*coefficient/root;residual=(tau-sigma*friction)/eta-velocity
                velocity=max(1e-45,velocity-residual/(-(sigma/eta)*derivative-1.0))
        roots.extend(row_roots);asinhs.extend(row_asinhs)
    return counts,flags,coefficients,roots,asinhs


def placeholder_reference(rows):
    counts=(8,22,6,4,4,6,6,1,6);flags=(1,1,1,1,1,1,1,0,1);result=[]
    for index,row in enumerate(rows):
        tau=float(row["shear_traction"]);sigma=float(row["effective_normal_stress"]);state=float(row["state_reference"]);a=float(row["direct_effect_a"]);b=float(row["evolution_effect_b"]);f0=float(row["reference_friction"]);eta=float(row["shear_impedance"]);z=max(float(row["previous_velocity"]),1e-45)
        seed=f0+b*state;coefficient=1.0+seed/(1.0+abs(seed))
        for step in range(60):
            q=z*coefficient;root=1.0+abs(q);av=q/(1.0+abs(q));friction=a*av;derivative=a*coefficient/root;residual=(tau-sigma*friction)/eta-z
            if step<counts[index] and (step+1<counts[index] or not flags[index]):z=max(1e-45,z-residual/(-(sigma/eta)*derivative-1.0))
        q=z*coefficient;av=q/(1.0+abs(q));friction=a*av;residual=(tau-sigma*friction)/eta-z;result.append((z,friction,residual,counts[index],flags[index]))
    return result


def summarize(values):
    ordered=sorted(values)
    def q(p):
        x=(len(ordered)-1)*p;lo=int(x);hi=min(lo+1,len(ordered)-1);return ordered[lo]*(hi-x)+ordered[hi]*(x-lo)
    return {"median_us":statistics.median(ordered),"p5_us":q(.05),"p95_us":q(.95),"samples_us":values}


def timed(fn,warmup,repeat):
    for _ in range(warmup):fn()
    torch.cuda.synchronize();values=[]
    for _ in range(repeat):
        start,stop=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record();fn();stop.record();stop.synchronize();values.append(start.elapsed_time(stop)*1e3)
    return summarize(values)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--rsf-root",type=Path,required=True);parser.add_argument("--n",type=int,default=65537);parser.add_argument("--warmup",type=int,default=10);parser.add_argument("--repeat",type=int,default=30);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    probe=load_probe(args.rsf_root/"probes/triton/rsf/aging_rsf_numerical_feasibility.py");rows=probe.read_tsv(args.rsf_root/"test-data/generated/src03-aging-rsf-newton-fp64-input-v1.tsv")
    selected=[rows[i%9] for i in range(args.n)];floats={name:probe.tensor([float(row[name]) for row in selected]) for name in FLOAT_NAMES}
    counts,flags,coefficients,roots,asinhs=trajectory_tables(rows);count=probe.tensor([counts[i%9] for i in range(args.n)],torch.int32);flag=probe.tensor([flags[i%9] for i in range(args.n)],torch.int32)
    coefficient_table=probe.tensor(coefficients);sqrt_table=probe.tensor(roots);asinh_table=probe.tensor(asinhs)
    outputs=[torch.empty(args.n,dtype=torch.float64,device="cuda") for _ in range(3)]+[torch.empty(args.n,dtype=torch.int32,device="cuda") for _ in range(2)]
    def launch(mode):
        lookup_kernel[(triton.cdiv(args.n,64),)](*(floats[name] for name in FLOAT_NAMES[:-1]),count,flag,coefficient_table,sqrt_table,asinh_table,*outputs,n=args.n,block=64,loop_count=60,mode=mode,num_warps=1)
    launch(0);torch.cuda.synchronize();reference=[x.clone() for x in outputs]
    placeholder=placeholder_reference(rows);placeholder_expected=[probe.tensor([placeholder[i%9][field] for i in range(args.n)],torch.int32 if field>=3 else torch.float64) for field in range(5)]
    names=(("fixed_original_pre",0),("lookup_coefficient",1),("lookup_sqrt",2),("lookup_asinh",3),("lookup_all_transcendentals",4),("arithmetic_placeholder",5),("fixed_original_restore",0));variants={};differences={}
    for name,mode in names:
        launch(mode);torch.cuda.synchronize();expected=placeholder_expected if mode==5 else reference;diff=[float(torch.max(torch.abs(a.to(torch.float64)-b.to(torch.float64))).cpu()) for a,b in zip(expected,outputs)];differences[name]=diff
        if any(x>1e-10 for x in diff[:3]) or any(x!=0 for x in diff[3:]):raise RuntimeError(f"{name} mismatch: {diff}")
        variants[name]=timed(lambda m=mode:launch(m),args.warmup,args.repeat)
    restore=variants["fixed_original_restore"]["median_us"]/variants["fixed_original_pre"]["median_us"]
    result={"schema_version":"rsf-c3-lookup-triton/2.0","status":"PASS","backend":"triton","n":args.n,"warmup":args.warmup,"repeat":args.repeat,"differences":differences,"variants":variants,"restore_over_pre":restore,"reversibility_pass":.9<=restore<=1.1}
    if not result["reversibility_pass"]:raise RuntimeError(f"reversibility failed: {restore}")
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({"status":"PASS","restore_over_pre":restore,"medians":{k:v["median_us"] for k,v in variants.items()}}))


if __name__=="__main__":main()
