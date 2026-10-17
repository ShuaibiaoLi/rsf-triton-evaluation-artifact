#!/usr/bin/env python3
"""Generate six revision-v1 manuscript figures and their source-data CSV files."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
SRC = HERE / "source_data"
SKILL = Path("/root/.codex/skills/nature-figure/scripts")
sys.path.insert(0, str(SKILL))
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    def require_matplotlib_panel_alignment(*args, **kwargs):
        """Public-artifact fallback; figure rendering does not require the audit helper."""
        return {"verdict": "NOT_RUN", "reason": "optional alignment helper unavailable"}

BLUE="#0072B2"; ORANGE="#E69F00"; GREEN="#009E73"; RED="#D55E00"
GREY="#777777"; DARK="#333333"; MID="#666666"; LIGHT="#CCCCCC"; PALE="#F7F7F7"; WHITE="#FFFFFF"

mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["DejaVu Sans","Arial","sans-serif"],
 "font.size":7.2,"axes.labelsize":7.2,"axes.titlesize":7.4,"xtick.labelsize":7.2,"ytick.labelsize":7.2,
 "legend.fontsize":7.2,"svg.fonttype":"none","pdf.fonttype":42,"axes.spines.top":False,
 "axes.spines.right":False,"axes.linewidth":.7,"legend.frameon":False,"figure.facecolor":"white",
 "axes.facecolor":"white","savefig.facecolor":"white"})

def J(p): return json.loads((ROOT/p).read_text())
def write_csv(name, rows, fields):
    SRC.mkdir(parents=True,exist_ok=True)
    with (SRC/name).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def label(ax,s): ax.text(-.10,1.03,s,transform=ax.transAxes,fontweight="bold",fontsize=8,va="bottom")
def box(ax,x,y,w,h,title,body="",ec=LIGHT,fc=WHITE,ls="-",lw=1.1):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.012,rounding_size=.018",fc=fc,ec=ec,lw=lw,ls=ls))
    ax.text(x+w/2,y+h*.65,title,ha="center",va="center",fontweight="bold",fontsize=7.2,color=DARK)
    if body: ax.text(x+w/2,y+h*.29,body,ha="center",va="center",fontsize=7.2,color=MID,linespacing=1.15)
def arrow(ax,a,b,color=DARK,ls="-"):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=8,lw=.9,color=color,ls=ls))
def schematic_ax(ax): ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")

def save(fig,stem,row_groups=None,column_groups=None,exemptions=()):
    OUT.mkdir(parents=True,exist_ok=True); fig.canvas.draw()
    axes=[a for a in fig.axes if a.get_gid()]
    if len(axes)>1:
        require_matplotlib_panel_alignment(fig=fig,json_out=str(OUT/f"{stem}.alignment.json"),
          overlay_svg=str(OUT/f"{stem}.alignment.svg"),tolerance_pt=1.5,gutter_tolerance_pt=1.5,
          row_groups=row_groups,column_groups=column_groups,exemptions=exemptions,strict=True)
    else:
        (OUT/f"{stem}.alignment.json").write_text(json.dumps({"verdict":"NOT APPLICABLE","reason":"single plot-area"},indent=2))
    fig.savefig(OUT/f"{stem}.svg")
    fig.savefig(OUT/f"{stem}.pdf")
    fig.savefig(OUT/f"{stem}.png",dpi=600)
    fig.savefig(OUT/f"{stem}.tiff",dpi=600)
    plt.close(fig)

def fig1_mapping_reversal():
    bw=J(Path("results/bw1000/revision_v1/formal/spmv-four-cell-independent-jobs-v1.json"))
    def agg(metric): return next(x for x in bw["aggregate"] if x["matrix"]=="orani678" and x["metric"]==metric)
    d=J(Path("results/dongfang/revision_v1/formal/spmv-four-cell-formal-local-replay.json"))
    dc=next(x for x in d["comparisons"] if x["matrix"]=="orani678")
    rows=[{"platform":"BW1000","comparison":"simple row-level","ratio":agg("row_level_triton_over_hip")["median"],"evidence":"A independent jobs"},
          {"platform":"BW1000","comparison":"best/best registered candidates","ratio":agg("best_triton_over_best_hip")["median"],"evidence":"n=5 independent Slurm tasks; median"},
          {"platform":"Dongfang","comparison":"simple row-level","ratio":dc["row_level_triton_over_hip"],"evidence":"A formal replay"},
          {"platform":"Dongfang","comparison":"best/best registered candidates","ratio":dc["best_triton_over_best_hip"],"evidence":"one frozen formal replay"}]
    write_csv("figure1_mapping_reversal.csv",rows,list(rows[0]))
    fig=plt.figure(figsize=(7.2,2.75)); gs=fig.add_gridspec(1,3,width_ratios=[1.7,1.25,1.05],left=.11,right=.98,bottom=.19,top=.85,wspace=.38)
    ax=fig.add_subplot(gs[0]); ax.set_gid("a"); label(ax,"a")
    ax.axvline(1,color=DARK,ls="--",lw=.8)
    for y,p,m in [(1,"BW1000","o"),(0,"Dongfang","s")]:
        vals=[r["ratio"] for r in rows if r["platform"]==p]
        ax.plot(vals,[y,y],color=LIGHT,lw=2); ax.scatter(vals[0],y,color=BLUE,marker=m,s=30,zorder=3)
        ax.scatter(vals[1],y,facecolor=WHITE,edgecolor=RED,marker=m,s=38,lw=1.2,zorder=3)
        off=9 if y else -14
        ax.annotate(f"{vals[0]:.3f}×",(vals[0],y),xytext=(0,off),textcoords="offset points",ha="center",fontsize=7.2,color=BLUE)
        xoff=-8 if (p=="Dongfang") else 0
        ax.annotate(f"{vals[1]:.3f}×",(vals[1],y),xytext=(xoff,off),textcoords="offset points",ha="center",fontsize=7.2,color=RED)
    assert all(r["ratio"] > 0 for r in rows)
    ax.set_xscale("log"); ax.set_xlim(.18,6.2); ax.set_ylim(-.25,1.25); ax.set_yticks([0,1],["Dongfang","BW1000"]); ax.set_xlabel("Triton / HIP latency (log scale)")
    ax.tick_params(axis="y",length=0); ax.spines["left"].set_visible(False)
    ax.text(.02,1.07,"● simple row-level",transform=ax.transAxes,color=BLUE,ha="left"); ax.text(.54,1.07,"○ registered best/best",transform=ax.transAxes,color=RED,ha="left")
    ax=fig.add_subplot(gs[1]); ax.set_gid("b"); label(ax,"b"); schematic_ax(ax)
    for i,(t,c) in enumerate([("HIP\none-thread",GREY),("HIP\ncooperative",GREY),("Triton\nrow prog.",BLUE),("Triton\nblock row",BLUE)]):
        x=.03+(i%2)*.51; y=.56-(i//2)*.45; box(ax,x,y,.41,.34,t,"Equal budget",ec=c,fc=PALE)
    ax=fig.add_subplot(gs[2]); ax.set_gid("c"); label(ax,"c"); schematic_ax(ax)
    lengths=[.12,.18,.20,.25,.82,.15,.58,.17]
    for i,v in enumerate(lengths): ax.add_patch(Rectangle((.08,.82-i*.09),v,.045,fc=BLUE,alpha=.75))
    ax.text(.52,.91,"Heavy-tailed row lengths",ha="center",fontweight="bold")
    ax.text(.5,.08,"Structure changes\nmapping gain,\nnot language by itself",ha="center",va="center",fontsize=7.2,color=MID)
    save(fig,"figure1_mapping_reversal",row_groups=[["a","b","c"]],
         exemptions=[{"panels":["a","b","c"],"checks":["panel-width"],"reason":"hero claim-reversal panel intentionally wider than explanatory schematic panels"}])

def fig2_framework():
    rows=[{"loop":"mapping fairness","sequence":"simple observation > four-cell tuning > independent replay","result":"bounded backend statement"},
          {"loop":"timing boundary","sequence":"first call > host > event > batch > graph > profiler","result":"steady-state statement"},
          {"loop":"causal intervention","sequence":"hypothesis > fixed replay > removal > restoration","result":"negative causal result"},
          {"loop":"migration gates","sequence":"kernel > interop > production","result":"production pending"}]
    write_csv("figure2_framework_evidence.csv",rows,list(rows[0]))
    fig,ax=plt.subplots(figsize=(7.2,3.05)); schematic_ax(ax); ax.set_gid("a"); label(ax,"a")
    loops=[("MAPPING FAIRNESS","Simple mapping","Four-cell equal budget","Independent replay","Bounded backend",BLUE),
           ("TIMING BOUNDARY","Single event","Batch + graph","Kernel duration","Steady-state claim",ORANGE),
           ("REVERSIBLE CAUSAL TEST","Registered hypothesis","Fixed replay + lookup","Restore original","Negative result",GREEN)]
    for j,(head,a,b,c,d,col) in enumerate(loops):
        y=.75-j*.25
        ax.add_patch(Rectangle((.02,y-.025),.20,.14,fc=col,alpha=.13,ec="none"))
        ax.text(.12,y+.045,head,fontweight="bold",color=col,ha="center",va="center",fontsize=5.4)
        ax.add_patch(Rectangle((.25,y-.025),.72,.14,fc=PALE,ec="none"))
        ax.text(.61,y+.045,f"{a}  →  {b}  →  {c}  →  {d}",ha="center",va="center",fontsize=5.8,color=DARK)
    ax.add_patch(Rectangle((.02,.04),.96,.16,fc=PALE,ec="none"))
    ax.text(.17,.12,"Kernel correctness\nPASS",ha="center",va="center",color=GREEN,fontweight="bold")
    ax.text(.49,.12,"Minimal interop\nPASS",ha="center",va="center",color=GREEN,fontweight="bold")
    ax.text(.80,.12,"Production application\nNOT TESTED",ha="center",va="center",color=ORANGE,fontweight="bold")
    ax.text(.33,.12,"→",ha="center",va="center",color=DARK); ax.text(.65,.12,"⇢",ha="center",va="center",color=DARK)
    save(fig,"figure2_evidence_framework")

def fig3_timing():
    ratios={"Empty":[4.46,5.72,1.52,1.00],"Large stencil":[1.64,1.00,1.00,1.10],"Reduction":[1.15,1.00,1.00,1.09],"RSF-C3":[3.80,1.99,1.81,3.24]}
    layers=["Single event","Batch=128","Graph replay","Kernel duration"]
    rows=[]
    for case,v in ratios.items():
        rows += [{"case":case,"timing_layer":l,"triton_over_hip":x,"source":"timing_decomposition_revision_v2"} for l,x in zip(layers,v)]
    ind=J(Path("results/bw1000/revision_v1/formal/nonspmv-independent-jobs-v1.json"))
    for r in ind["aggregate"]:
        rows.append({"case":r["case"],"timing_layer":"independent jobs median","triton_over_hip":r["median_ratio"],"source":"nonspmv-independent-jobs-v1.json"})
    write_csv("figure4_timing_decomposition.csv",rows,list(rows[0]))
    fig,axs=plt.subplots(1,3,figsize=(7.2,2.7)); fig.subplots_adjust(left=.08,right=.965,bottom=.23,top=.83,wspace=.46)
    colors=[GREY,BLUE,GREEN,RED]; marks=["o","s","^","D"]
    for ax,g in zip(axs,"abc"): ax.set_gid(g); label(ax,g)
    x=np.arange(4)
    for (case,v),c,m in zip(ratios.items(),colors,marks): axs[0].plot(x,v,marker=m,color=c,lw=1.1,ms=4)
    axs[0].axhline(1,color=DARK,ls="--",lw=.8); axs[0].set_xticks(x,["Single\nevent","Batch\n128","Graph","Kernel\nduration"]); axs[0].set_ylabel("Triton / HIP latency"); axs[0].text(.01,1.10,"● Empty  ■ Stencil  ▲ Reduction  ◆ RSF",transform=axs[0].transAxes,fontsize=6.2)
    sel=["Large stencil","Reduction","RSF-C3"]
    for i,k in enumerate(sel): axs[1].scatter(np.arange(3)+i*.08,[ratios[k][1],ratios[k][2],ratios[k][3]],color=colors[["Large stencil","Reduction","RSF-C3"].index(k)+1],s=23,label=k)
    axs[1].axhline(1,color=DARK,ls="--",lw=.8); axs[1].set_xticks(range(3),["Batch","Graph","Kernel"]); axs[1].set_ylabel("Steady-state ratio")
    amap={r["case"]:r for r in ind["aggregate"]}; keys=["interior_stencil_large","row_reduction","rsf_c3"]
    med=np.array([amap[k]["median_ratio"] for k in keys]); lo=np.array([amap[k]["p5_ratio"] for k in keys]); hi=np.array([amap[k]["p95_ratio"] for k in keys])
    axs[2].errorbar(range(3),med,yerr=[med-lo,hi-med],fmt="o",color=BLUE,capsize=2.5); axs[2].axhline(1,color=DARK,ls="--",lw=.8)
    axs[2].set_xticks(range(3),["Stencil","Reduction","RSF"]); axs[2].set_ylabel("Triton / HIP ratio")
    axs[2].set_title("n=5 Slurm tasks; median and descriptive P5–P95",fontsize=6.2)
    save(fig,"figure4_timing_decomposition",row_groups=[["a","b","c"]])

def fig4_spmv():
    bw=J(Path("results/bw1000/revision_v1/formal/spmv-four-cell-independent-jobs-v1.json")); df=J(Path("results/dongfang/revision_v1/formal/spmv-four-cell-formal-local-replay.json"))
    metrics=["hip_cooperative_over_thread","triton_block_over_row","row_level_triton_over_hip","best_triton_over_best_hip"]
    mats=["apache2","dw8192","orani678"]
    rows=[]
    for p in ["BW1000"]:
        for m in mats:
            for q in metrics:
                a=next(x for x in bw["aggregate"] if x["matrix"]==m and x["metric"]==q); rows.append({"platform":p,"matrix":m,"metric":q,"ratio":a["median"]})
    for m in mats:
        c=next(x for x in df["comparisons"] if x["matrix"]==m)
        for q in metrics: rows.append({"platform":"Dongfang","matrix":m,"metric":q,"ratio":c[q]})
    for p,path in [("BW1000","results/bw1000/revision_v1/holdout/sme3Da-evaluation-v1.json"),("Dongfang","results/dongfang/revision_v1/holdout/sme3Da-evaluation-v1.json")]:
        h=J(Path(path));
        for e in h["evaluations"]: rows.append({"platform":p,"matrix":"sme3Da","metric":e["backend"]+"_holdout_regret","ratio":e["regret"]})
    write_csv("figure3_spmv_fair_mapping.csv",rows,list(rows[0]))
    fig,axs=plt.subplots(1,3,figsize=(7.2,2.75)); fig.subplots_adjust(left=.09,right=.97,bottom=.22,top=.83,wspace=.50)
    for ax,g in zip(axs,"abc"): ax.set_gid(g); label(ax,g)
    heat=np.array([[next(r["ratio"] for r in rows if r["platform"]=="BW1000" and r["matrix"]==m and r["metric"]==q) for q in metrics] for m in mats])
    # All latency ratios are strictly positive before log10 encoding.
    assert np.all(heat > 0), "Latency ratios must be positive before log10 encoding"
    im=axs[0].imshow(np.log10(heat),cmap="coolwarm",aspect="auto",vmin=-1.1,vmax=1.1)
    axs[0].set_xticks(range(4),["HIP\ncoop.","Triton\nblock","Simple\npair","Best\npair"]); axs[0].set_yticks(range(3),mats)
    axs[0].tick_params(axis="x", labelsize=5.8, pad=1)
    for i in range(3):
        for j in range(4): axs[0].text(j,i,f"{heat[i,j]:.2f}",ha="center",va="center",fontsize=7.2,color=DARK)
    axs[0].set_title("BW1000 mapping ratios")
    vals={p:[next(r["ratio"] for r in rows if r["platform"]==p and r["matrix"]==m and r["metric"]=="best_triton_over_best_hip") for m in mats] for p in ["BW1000","Dongfang"]}
    y=np.arange(3)
    for i,m in enumerate(mats): axs[1].plot([vals["BW1000"][i],vals["Dongfang"][i]],[i,i],color=LIGHT,lw=1.5)
    axs[1].scatter(vals["BW1000"],y,color=BLUE,s=25); axs[1].scatter(vals["Dongfang"],y,facecolor=WHITE,edgecolor=ORANGE,marker="s",s=29)
    # All latency ratios are strictly positive before log-scale plotting.
    assert all(v > 0 for p in vals.values() for v in p), "Latency ratios must be positive before log-scale plotting"
    axs[1].axvline(1,color=DARK,ls="--",lw=.8); axs[1].set_xscale("log"); axs[1].set_yticks(y,mats); axs[1].set_xlabel("Best Triton / best HIP"); axs[1].text(.01,1.08,"● BW1000   □ Dongfang",transform=axs[1].transAxes,fontsize=6.5)
    hold=[]
    for p in ["BW1000","Dongfang"]:
        hold.append([100*next(r["ratio"] for r in rows if r["platform"]==p and r["metric"]==b+"_holdout_regret") for b in ["hip","triton"]])
    xx=np.arange(2); w=.34
    axs[2].bar(xx-w/2,hold[0],w,color=BLUE); axs[2].bar(xx+w/2,hold[1],w,color=ORANGE)
    axs[2].set_xticks(xx,["HIP","Triton"]); axs[2].set_ylabel("Frozen-rule regret (%)"); axs[2].set_title("sme3Da revision holdout")
    save(fig,"figure3_spmv_fair_mapping",row_groups=[["a","b","c"]])

def fig5_rsf():
    bw=J(Path("results/bw1000/revision_v1/ablation/rsf-c3-function-path-paired-v2.json")); pmc=J(Path("results/bw1000/revision_v1/ablation/rsf-c3-lookup-pmc-v2.json")); df=J(Path("results/dongfang/revision_v1/ablation/rsf-c3-function-path-paired-v2.json"))
    rows=[]
    for p,d in [("BW1000",bw),("Dongfang",df)]:
        for v in d["variants"]: rows.append({"platform":p,"variant":v["variant"],"hip_us":v["hip"]["median_us"],"triton_us":v["triton"]["median_us"],"ratio":v["triton_over_hip"],"valu_ratio":""})
    for v in pmc["variants"]: rows.append({"platform":"BW1000 PMC","variant":v["variant"],"hip_us":"","triton_us":"","ratio":"","valu_ratio":v["triton_over_hip_valu"]})
    write_csv("figure5_rsf_reversible_ablation.csv",rows,list(rows[0]))
    fig,axs=plt.subplots(1,3,figsize=(7.2,2.85)); fig.subplots_adjust(left=.08,right=.97,bottom=.22,top=.83,wspace=.48)
    for ax,g in zip(axs,"abc"): ax.set_gid(g); label(ax,g)
    variants=["fixed_original_pre","lookup_coefficient","lookup_sqrt","lookup_asinh","lookup_all_transcendentals","arithmetic_placeholder","fixed_original_restore"]
    disp=["Orig","Coef","Sqrt","Asinh","All","Arith","Rest"]
    B={v["variant"]:v for v in bw["variants"]}; x=np.arange(len(variants)); hp=np.array([B[v]["hip"]["median_us"] for v in variants]); tr=np.array([B[v]["triton"]["median_us"] for v in variants])
    axs[0].plot(x,hp,"s-",color=GREY); axs[0].plot(x,tr,"o-",color=BLUE); axs[0].set_xticks(x,disp); axs[0].set_ylabel("Median latency (μs)"); axs[0].text(.02,1.08,"■ HIP   ● Triton",transform=axs[0].transAxes,fontsize=6.5)
    ratio=np.array([B[v]["triton_over_hip"] for v in variants]); axs[1].plot(x,ratio,"o-",color=RED); axs[1].axhline(ratio[0],color=GREY,ls="--",lw=.8,label="Original gap")
    axs[1].set_xticks(x,disp); axs[1].set_ylabel("Triton / HIP latency")
    P={v["variant"]:v["triton_over_hip_valu"] for v in pmc["variants"]}; pv=["fixed_original","lookup_coefficient","lookup_sqrt","lookup_asinh","lookup_all_transcendentals","arithmetic_placeholder"]
    axs[2].bar(range(len(pv)),[P[v] for v in pv],color=[BLUE,BLUE,BLUE,ORANGE,RED,RED]); axs[2].set_xticks(range(len(pv)),["Orig","Coef","Sqrt","Asinh","All","Arith"]); axs[2].set_ylabel("Triton / HIP VALU instructions"); axs[2].axhline(1,color=DARK,ls="--",lw=.8)
    save(fig,"figure5_rsf_reversible_ablation",row_groups=[["a","b","c"]])

def fig7_migration():
    rows=[
      {"platform":"BW1000","gate":"RSF correctness","status":"PASS","detail":"C0-C6"},{"platform":"BW1000","gate":"minimal interop","status":"PASS","detail":"pointer/stream/padded layout"},{"platform":"BW1000","gate":"native TPV101 application","status":"PASS","detail":"official HIP baseline"},{"platform":"BW1000","gate":"Triton production application","status":"NOT_TESTED","detail":"real LTS/MPI"},
      {"platform":"Dongfang","gate":"RSF correctness","status":"PASS","detail":"C2-C3; decomposed C4-C6"},{"platform":"Dongfang","gate":"fused C4","status":"FAIL_COMPILE","detail":"old Triton/MLIR assertion"},{"platform":"Dongfang","gate":"minimal interop","status":"PASS","detail":"max abs error 1.6965e-16"},{"platform":"Dongfang","gate":"production application","status":"NOT_TESTED","detail":"real LTS/MPI"}]
    write_csv("figure7_seissol_migration_gates.csv",rows,list(rows[0]))
    fig,ax=plt.subplots(figsize=(7.2,3.05)); schematic_ax(ax); ax.set_gid("a"); label(ax,"a")
    box(ax,.03,.55,.25,.28,"1  KERNEL CORRECTNESS","BW1000: C0–C6 PASS\nDongfang: C2/C3 PASS",ec=BLUE,fc="#EAF3F8")
    box(ax,.38,.55,.25,.28,"2  MINIMAL INTEROP","Pointers + non-default stream\n3 faces; 9 / 16 padded points",ec=GREEN,fc="#E8F5EF")
    box(ax,.73,.55,.24,.28,"3  TRITON APP","Native TPV101: PASS\nTriton LTS/MPI: NOT TESTED\nTriton speedup: NOT TESTED",ec=ORANGE,fc=PALE,ls="--")
    arrow(ax,(.28,.69),(.38,.69),BLUE); arrow(ax,(.63,.69),(.73,.69),DARK,ls="--")
    box(ax,.06,.16,.28,.20,"Dongfang fused C4","compile assertion / FAIL",ec=RED,fc="#FFF1EC")
    box(ax,.40,.16,.28,.20,"Four-kernel decomposition","C4 + six C5 scales + C6 PASS",ec=GREEN,fc="#E8F5EF")
    arrow(ax,(.34,.26),(.40,.26),GREEN)
    ax.text(.20,.10,"Compatibility boundary, not numerical failure",ha="center",fontsize=7.2,color=RED)
    box(ax,.73,.16,.24,.20,"CLAIM BOUNDARY","LAYOUT_BRIDGE_PASS\nFULL_APPLICATION_PENDING",ec=ORANGE,fc=WHITE,ls="--")
    ax.text(.5,.94,"Correctness, interoperability, and production replacement are independent gates",ha="center",fontweight="bold",fontsize=7.2)
    save(fig,"figure7_seissol_migration_gates")

def fig6_tpv101_context():
    rows=[
      {"platform":"CPU","configuration":"FL=3 Aging RSF","dynamic_rupture_s":1.248650,"rsf_delta_s":0.594436,"rsf_fraction_percent":47.6},
      {"platform":"BW1000 gfx936 HIP","configuration":"FL=3 Aging RSF","dynamic_rupture_s":0.292790,"rsf_delta_s":0.079184,"rsf_fraction_percent":27.0},
      {"platform":"BW1000 gfx936 HIP","configuration":"TPV101 kernel breakdown","dynamic_rupture_s":7.181980,"rsf_delta_s":"","rsf_fraction_percent":33.4}
    ]
    write_csv("figure6_tpv101_application_context.csv",rows,list(rows[0]))
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.65)); fig.subplots_adjust(left=.10,right=.97,bottom=.22,top=.82,wspace=.38)
    for ax,g in zip(axs,"ab"): ax.set_gid(g); label(ax,g)
    platforms=["CPU\n(1 thread)","BW1000\nHIP"]
    total=np.array([1.248650,.292790]); delta=np.array([.594436,.079184]); base=total-delta
    x=np.arange(2)
    axs[0].bar(x,base,color=LIGHT,label="FL=0 control")
    axs[0].bar(x,delta,bottom=base,color=ORANGE,label="FL=3 − FL=0")
    axs[0].set_xticks(x,platforms); axs[0].set_ylabel("Dynamic Rupture time (s)")
    axs[0].text(.02,1.06,"Grey: matched FL=0 control; orange: RSF-path proxy",transform=axs[0].transAxes,fontsize=6.2)
    for i,v in enumerate([47.6,27.0]): axs[0].text(i,total[i]+total.max()*.035,f"{v:.1f}%",ha="center",color=ORANGE,fontweight="bold")
    vals=np.array([7.13288,7.17273,7.18198]); names=["Local\nintegration","Neighbor\nintegration","Dynamic\nrupture"]
    axs[1].bar(range(3),vals,color=[BLUE,BLUE,ORANGE])
    axs[1].set_xticks(range(3),names); axs[1].set_ylabel("TPV101 compute-kernel time (s)")
    axs[1].text(.5,1.06,"BW1000 official HIP TPV101; Dynamic Rupture = 33.4%",transform=axs[1].transAxes,ha="center",fontsize=6.2)
    save(fig,"figure6_tpv101_application_context",row_groups=[["a","b"]])

def main():
    OUT.mkdir(parents=True,exist_ok=True); SRC.mkdir(parents=True,exist_ok=True)
    fig1_mapping_reversal(); fig2_framework(); fig3_timing(); fig4_spmv(); fig5_rsf(); fig6_tpv101_context(); fig7_migration()
if __name__=="__main__": main()
