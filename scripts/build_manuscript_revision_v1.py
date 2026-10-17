#!/usr/bin/env python3
"""Build the authorized round-1 academic-paper revision artifacts.

The prose and operation list are intentionally explicit.  The official
academic-paper revision tools still validate the authority sidecar and apply
the resulting patch; this helper only removes hand-transcription risk.
"""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "revisions" / "roadmap_v1"
OUT = REV / "round1_authorized"
BASE = REV / "manuscript_cn_full_v1.anchored.md"
MANIFEST = REV / "manuscript_cn_full_v1.block-manifest.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jcs(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


REPLACEMENTS = {
    "B0001": "# 国产 DCU 平台栈上的科学计算 Triton Kernel：结构敏感性能、机制干预与 SeisSol 迁移边界\n",
    "B0003": """高层 GPU kernel 语言在非主流加速器上的性能取决于程序映射、输入结构和软件栈边界。本文以 BW1000/DCU 为主平台、东方 gfx906 DCU 为第二平台，研究 FP64 stencil、行归约、稀疏矩阵向量乘（SpMV）和 SeisSol Aging rate-and-state friction（RSF）Newton 迭代。实验采用独立 CPU reference、对称候选预算、独立 Slurm 作业、冻结 holdout、七层计时分解、性能计数器和可逆机制干预。BW1000 上五个独立作业显示，stencil 与行归约的 Triton/HIP 配对比值中位数分别为 1.002 和 0.996：前者区间跨1，后者方向一致但差异小于0.5%；RSF C3 为 1.984。SpMV 四格实验进一步表明，映射选择能够改变结论：`orani678` 的 row-level Triton/HIP 比值约为 0.272，但候选集合内 best Triton/best HIP 为 1.958。冻结规则在 `sme3Da` 上未命中 top-1，BW1000 的 HIP/Triton regret 分别为 17.81%/2.80%，东方为 0.19%/12.63%，因此当前规则只作为探索性常数默认。逐 dispatch 时间线将设备执行与调用路径分开；RSF 固定迭代和函数替换表明，移除超越函数并未缩小 Triton/HIP 差距，否定了“超越函数存在本身”这一单因果解释。两平台软件栈未对齐，本文不计算正式性能可移植性指标，而将贡献限定为国产 DCU 平台栈上的结构敏感案例研究、机制边界和 SeisSol 分级迁移门禁。\n""",
    "B0007": "**关键词：** Triton；国产 DCU；FP64；稀疏矩阵向量乘；机制干预；SeisSol\n",
    "B0009": "## Scientific-Computing Triton Kernels on Domestic DCU Software Stacks: Structure-Sensitive Performance, Mechanism Interventions, and SeisSol Migration Boundaries\n",
    "B0011": """High-level GPU kernel performance on non-mainstream accelerators depends on program mapping, input structure, and software-stack boundaries. This study evaluates FP64 stencil, row reduction, sparse matrix-vector multiplication (SpMV), and SeisSol Aging rate-and-state friction (RSF) Newton kernels on a primary BW1000/DCU stack and a second Dongfang gfx906 DCU stack. The protocol combines independent CPU references, symmetric tuning budgets, independent Slurm jobs, a frozen holdout, seven timing boundaries, performance counters, and reversible interventions. Across five independent BW1000 jobs, the median paired Triton/HIP ratios are 1.002 for stencil, 0.996 for row reduction, and 1.984 for RSF C3: the stencil interval crosses one, whereas the reduction direction is consistent but differs by less than 0.5%. The four-cell SpMV experiment shows that mapping can reverse an apparent backend result: on `orani678`, the row-level ratio is approximately 0.272, whereas the candidate-set best-Triton/best-HIP ratio is 1.958. Frozen defaults miss the holdout top configuration on both stacks, with HIP/Triton regrets of 17.81%/2.80% on BW1000 and 0.19%/12.63% on Dongfang, so the rule remains exploratory. Dispatch timelines separate device execution from the broader call path. Fixed-iteration and function-replacement interventions further show that removing transcendental functions does not reduce the RSF gap, rejecting transcendental presence alone as the causal explanation. Because the two software stacks are not aligned, the paper does not compute a formal performance-portability metric. Its contribution is instead a bounded case study of structure-sensitive behavior, mechanism evidence, and staged SeisSol migration gates on domestic DCU software stacks.\n""",
    "B0014": "**Keywords:** Triton; domestic DCU; FP64; sparse matrix-vector multiplication; mechanism intervention; SeisSol\n",
    "B0018": """近期研究已覆盖 Triton 在 HPC 算子和多厂商 stencil 上的比较，并区分了 kernel correctness 与 replacement quality[16–18]。因此，本文不把“跨厂商运行”或“正确但较慢”本身作为新颖性，而聚焦三个尚需系统证据的边界：国产 DCU 平台栈上的 FP64 非规则结构、能够排除替代解释的映射与函数路径干预，以及从独立 RSF kernel 到 SeisSol 生产调用点的分级迁移门禁。\n""",
    "B0021": """- **RQ1：** 在同卡、同输入和对称候选预算下，后端与 program 映射分别如何影响代表性 FP64 算子的性能？
- **RQ2：** 核心胜负在独立作业层是否稳定，冻结配置在 revision holdout 上产生多大 regret？
- **RQ3：** 调用路径分解、PMC 与可逆消融能够支持或否定哪些机制解释？
- **RQ4：** 双平台软件栈中的结构趋势可以迁移到何种范围，从独立 kernel 到 SeisSol 生产路径还缺少哪些门禁？\n""",
    "B0023": """1. 建立 HIP one-thread-row、HIP cooperative-row、Triton row-program 和 Triton block-row 的四格 SpMV 对照，在相同候选预算下分离映射效应与登记后端效应。
2. 以独立 Slurm 作业为统计单位复测核心算子，并在规则冻结后报告 holdout 的预测配置、oracle 配置和 regret，而不把同进程 kernel calls 当作独立重复。
3. 将首次调用、host submit-to-sync、device event、稳态 batch、graph replay、profiler start/end 和空 kernel 分开，并用 RSF 固定迭代与函数路径替换检验候选机制。
4. 在 BW1000 主平台形成正式证据链，在东方第二软件栈复测共同子集；跨平台只比较结构趋势，不计算未满足可比性前提的性能可移植性指标。
5. 区分 kernel correctness、候选集合内性能、interop 与 production integration；真实 `LTSRateAndState`、MPI/LTS 和完整时间推进仍为 `NOT_TESTED`。\n""",
    "B0031": """Pennycook 等将性能可移植性定义为明确平台集合上的相对效率[3]。本文虽然在两种 DCU 平台栈上执行共同实验，但 Triton、HIP、PyTorch 和 native 编译器版本并未对齐，且没有计算相应聚合指标。因此，本文不将实验解释为正式 performance portability 测量，而将其定位为双平台软件栈案例研究：每个平台内部以同卡 HIP 为参考，平台之间只检验结构趋势和迁移边界。\n""",
    "B0032": """这一定位同时限定了本文的外推范围。“国产 DCU”描述受测平台栈，不代表国产异构加速器总体；同向趋势表示现象在两个登记环境中复现，不表示已经控制硬件与软件栈差异。正式性能可移植性评价需要预先定义平台集合、可比的相对效率基线和聚合指标，并在对齐或充分控制的软件环境中重新测量。\n""",
    "B0055": """| 算子 | HIP native 配置或候选 | Triton 配置或候选 | 索引/布局 |
|---|---|---|---|
| stencil7 | 256 threads/block | block={256,512}；`num_warps={1,2,4,8}` | FP64 连续数组 |
| row_reduce256 | 256 threads/block | B=256；`num_warps={1,2,4,8}` | 每行256个FP64元素 |
| SpMV | one-thread-row 与 cooperative-row；block={64,128,256}；groups={1,2,4,8} | row-program 与 block-row；block={64,128,256}；warps={1,2,4,8} | FP64 CSR/SoA；32位列索引 |
| RSF/Newton | 256 threads/block；同一冻结语义 | block=64；`num_warps={1,4}` | FP64 SoA；相同迭代与函数替换路径 |\n""",
    "B0056": """所有候选先通过独立 CPU FP64 reference。SpMV 在每个平台分别使用两个 tuning 矩阵选择配置：每个后端、每种 mapping 均有 12 个候选，选择指标为中位 device-event 延迟，tie-break 预先冻结。正式矩阵和 `sme3Da` revision holdout 不参与配置形成；holdout 运行后不回改规则。\n""",
    "B0057": """集中式 correctness contract 登记数学语义、边界条件、dtype、索引类型、布局、输入域、随机种子、绝对/相对容差以及 NaN、Inf 和不收敛政策。每个算子覆盖极小、非整除、边界、典型和正式规模。RSF 额外检查 `g(V)` 与 `g'(V)`、`V_floor` 邻域、极端参数、停止阈值和预期不收敛输入；达到迭代上限必须显式返回失败标志。正式性能只接受 correctness `PASS` 的候选。\n""",
    "B0058": """SpMV 数据分为 correctness、tuning、formal representative 和 revision holdout 四层。`bcsstk17` 与 `poisson3Da` 仅用于 tuning；`dw8192`、`apache2` 与 `orani678` 用于正式四格比较；`sme3Da` 在规则冻结后一次性运行完整候选空间。每个矩阵保存 rows、nnz、mean/P95/P99/max row length、CV、KMAX、padding amplification、有效 lane 比例及输入 SHA256。`sme3Da` 曾用于更早探索，因此是 revision holdout，而非 pristine holdout。\n""",
    "B0063": """SpMV 的公平基线由四个 cell 构成：HIP one-thread-per-row、HIP cooperative-row、Triton row-program 和 Triton block-row。四格使用相同 CSR 输入、FP64 语义、32 位索引、输出和候选数量。该设计允许分别比较同一后端内的映射变化，以及论文登记的 row-level 与 cooperative-level 跨后端实现；后两者不是指令级完全相同算法，结论仍限定为登记实现。\n""",
    "B0064": """配置选择与正式评估严格分离。每个平台独立调优，不复用另一平台的最优参数；完整候选账本保留 correctness 过滤、逐候选样本、最终选择、tie-break、冻结时间和源码 SHA256。只有冻结候选集合中的最佳配置称为“candidate-set best”，不外推为厂商库或全局最优 baseline。\n""",
    "B0067": """计时采用七个互补边界：首次调用（含 JIT）、JIT 后 host submit-to-sync、单次 device event、连续 batch 的稳态单次值、graph replay、profiler kernel start/end，以及同协议空 kernel。device event 表示受测调用路径中的设备可见区间；profiler start/end 才用于近似纯 dispatch 执行时间。profiling 与正式 latency 作业分离，profiler 时间不进入 speedup。\n""",
    "B0068": """六个代表点均完成七层分解。空 kernel 的 profiler 中位数在 HIP 与 Triton 上均为 0.64 μs；小 stencil 为 2.24/2.56 μs，大 stencil 为 57.12/62.88 μs，row reduction 为 221.92/242.08 μs，短行 SpMV 为 4.72/8.32 μs，长尾 SpMV 为 31.44/28.64 μs，RSF C3 为 56.32/182.72 μs（HIP/Triton）。这些结果将小算子的 event 差异定位到调用与事件路径，同时保留大算子和 RSF 的设备执行差距。\n""",
    "B0073": """核心性能推断以独立 Slurm 任务为实验单位。每个任务内部先预热，再以 30 次或 128 次稳态调用形成任务内中位数；五个任务按奇偶编号交替 HIP-first 与 Triton-first。正文报告五个任务的配对比值中位数及 bootstrap 95% 区间，不把任务内 kernel calls 作为独立重复。单作业扫描和 profiler 结果仅用于结构或机制分析。\n""",
    "B0074": """BW1000 承担正式统计推断。东方承担冻结协议后的共同子集复测、holdout 和旧软件栈边界，不用单作业结果建立总体置信区间。跨平台比较只判断结构趋势是否同向，并分别披露版本差异；绝对延迟和倍率不跨平台直接排序。\n""",
    "B0085": """五个独立 BW1000 任务在相同输入、交替后端顺序和独立正确性门禁下复测代表性 stencil、row reduction、三类 SpMV 矩阵和 RSF C3。表3使用任务内中位数作为一个实验单位。\n""",
    "B0086": "**表3 BW1000 独立作业层核心结果**\n",
    "B0087": """| 案例 | Triton/HIP 配对比值中位数 | 95%区间或范围解释 | 结论 |
|---|---:|---|---|
| interior stencil | 1.002 | 区间跨1 | 未观察到稳定胜负 |
| row reduction | 0.996 | 极小差异 | 工程上近似持平 |
| RSF C3 | 1.984 | 1.947–2.046 | Triton稳定较慢 |
| SpMV `dw8192`，candidate-set best | 10.589 | 9.590–10.676 | HIP较快 |
| SpMV `apache2`，candidate-set best | 3.635 | 3.565–3.640 | HIP较快 |
| SpMV `orani678`，candidate-set best | 1.958 | 1.792–1.988 | HIP较快 |\n""",
    "B0088": """Stencil 与 reduction 的结果修正了早期同进程调用级别的胜负叙述：两者在当前正式协议下应写为持平或差异不确定。RSF C3 的独立作业区间未跨1，保留 Triton 较慢的结论。SpMV 的正式解释必须结合四格映射，不能沿用单一 row-level 对照。\n""",
    "B0090": """独立作业复测中，代表性 interior stencil 的 Triton/HIP 配对比值中位数为 1.002，置信区间跨越1。因此，扩大规模和移除边界逻辑后，两种登记实现表现接近；现有证据不支持宣称任一后端稳定获胜。七层计时分解显示大 stencil 的 profiler kernel 中位数为 HIP 57.12 μs、Triton 62.88 μs，约9%的设备执行差异与 event 层近似持平并存。\n""",
    "B0096": """行归约的五任务配对比值中位数为 0.996，差异不足1%。结合 profiler kernel 中位数 HIP 221.92 μs、Triton 242.08 μs，可见不同计时边界给出的细小排序并不完全相同。本文因此将该案例定义为“当前协议下近似持平”，不报告稳定后端优势。\n""",
    "B0098": """四格实验改变了早期 SpMV 结论的解释。五个独立任务中，HIP cooperative-row / one-thread-row 的比值中位数在 `dw8192`、`apache2`、`orani678` 上分别为 0.941、0.557、0.097；Triton block-row / row-program 分别为 0.916、0.269、0.710。映射收益随矩阵结构变化，且 cooperative mapping 对长尾 `orani678` 的 HIP baseline 改善最大。\n""",
    "B0099": """表4同时报告 row-level、cooperative-level 与 candidate-set best 比较。比值均为 Triton/HIP；小于1表示登记 Triton 实现更快。\n""",
    "B0100": "**表4 SpMV 四格公平比较的独立作业结果**\n",
    "B0101": """| 矩阵 | row-level Triton/HIP | cooperative-level Triton/HIP | candidate-set best Triton/HIP |
|---|---:|---:|---:|
| `dw8192` | 10.755 | 10.589 | 10.589 |
| `apache2` | 7.525 | 3.635 | 3.635 |
| `orani678` | 约0.272 | 约2.0 | 1.958 |\n""",
    "B0102": """`orani678` 的 row-level 比值保留了“看似 Triton 更快”的反转，但同一输入加入 cooperative HIP 后，candidate-set best 比值变为 1.958。该结果证明早期胜负混合了后端与映射效应；它不证明两种后端在所有等价算法下的固有排序。冻结规则在 BW1000 `sme3Da` holdout 上，HIP 与 Triton 均未命中 top-1，regret 分别为17.81%和2.80%。规则未回改，因此只能作为探索性常数默认。\n""",
    "B0108": """RSF C3 的因果实验依次执行原始路径、固定实际迭代次数回放、分别替换 `log`、`exp`、`sqrt`、`asinh`、全部函数查表、普通算术占位和恢复原函数，并在两后端使用相同输入与 correctness gate。固定迭代回放的 Triton/HIP 比值约为3.6–3.9，说明动态迭代次数不是主要差距来源。该干预协议与表3的正式独立作业协议不同，倍率只用于同一消融内部比较。\n""",
    "B0109": """函数路径替换得到负面因果结果。原始路径的 Triton/HIP 比值为3.893；替换 coefficient、`sqrt`、`asinh`、全部超越函数和普通算术占位后，比值分别为3.897、4.021、4.929、5.639和5.544，恢复路径为4.020。移除函数虽然使两端绝对延迟下降，却没有缩小跨后端差距，反而通常扩大。\n""",
    "B0110": "**表5 RSF C3 可逆消融与机制判断**\n",
    "B0111": """| 干预 | Triton/HIP | 对预登记假设的判断 |
|---|---:|---|
| 原始路径 | 3.893 | 基准 |
| 固定迭代回放 | 约3.6–3.9 | 动态迭代不是主要原因 |
| 替换 coefficient | 3.897 | 不支持单独归因于 coefficient 函数路径 |
| 替换 `sqrt` | 4.021 | 差距未缩小 |
| 替换 `asinh` | 4.929 | 差距扩大 |
| 替换全部超越函数 | 5.639 | 否定“函数存在本身”这一单因果解释 |
| 普通算术占位 | 5.544 | 差距仍未缩小 |
| 恢复原路径 | 4.020 | 恢复检查通过 |\n""",
    "B0112": """PMC 中各变体的 Triton/HIP VALU 比值约为2.21–2.64，说明整体表达式 lowering、控制流、指令调度或代码生成仍是候选解释，但现有计数不能在这些机制之间唯一归因。单独隔离 coefficient 中的 `log` 与 `exp` 同样未改变 Triton 约70 μs的延迟。正文因此撤回“transcendental lowering 是主要瓶颈”的强因果表述，只保留已被干预排除的解释和后续待检验机制。\n""",
    "B0123": """冻结协议在东方复测 RSF 共同消融及 SpMV tuning、正式矩阵和 holdout。东方独立选择的四格配置不同于 BW1000，全部96个 tuning 候选、12个正式记录和48个 holdout 候选均通过独立 FP64 reference。三个正式矩阵的 candidate-set best Triton/HIP 比值为 `dw8192` 8.721、`apache2` 1.952、`orani678` 4.662；`orani678` 的 row-level 比值仍为0.311，加入 cooperative HIP 后反转消失。`sme3Da` holdout 的 HIP/Triton regret 为0.19%/12.63%，两端均未命中 top-1且规则未回改。东方 RSF 函数替换同样显示移除超越函数后差距不缩小。第二平台因软件版本不同只支持趋势复现：映射会改变 SpMV 胜负，两个 tuning 矩阵形成的常数默认不具备普适样本外效度，RSF 差距也不能由超越函数存在本身解释。\n""",
    "B0157": """表9将修订后的证据映射到研究问题。BW1000 提供独立作业层统计、完整 timing 与机制干预；东方提供冻结协议的共同子集复测。\n""",
    "B0159": """| 研究问题 | 主要证据 | 回答与边界 |
|---|---|---|
| RQ1 后端与映射 | SpMV四格、相同候选预算 | 映射可改变胜负；结论限于登记实现和候选集合 |
| RQ2 稳定性与holdout | 五个独立作业；两平台`sme3Da`评估 | stencil/reduction近似持平，RSF稳定较慢；常数规则未普适命中top-1 |
| RQ3 timing与机制 | 七层计时、PMC、固定迭代和函数替换 | 小算子event包含调用路径成本；RSF差距不能归因于动态迭代或超越函数存在本身 |
| RQ4 迁移范围 | 东方共同子集；SeisSol kernel/interop/production门禁 | 两平台支持有限结构趋势；production integration仍为`NOT_TESTED` |\n""",
    "B0174": "综合结果形成的是有适用边界的工程判断，而不是已验证的普适后端选择器。\n",
    "B0175": """1. **Stencil 与行归约：** 当前正式协议下近似持平，选择应结合开发、部署与调用路径成本。
2. **SpMV：** 先保证 mapping 对称，再比较后端；row-level 反转不能替代 cooperative baseline。
3. **配置规则：** 两矩阵形成的常数默认只能作为探索起点；holdout regret 必须随规则报告。
4. **RSF/Newton：** 当前 Triton 实现稳定慢于 HIP；已排除动态迭代次数和超越函数存在本身这两个简单解释。
5. **SeisSol：** kernel、interop 与 production gate 分别验收；生产路径未闭合前不作应用加速主张。\n""",
    "B0176": """这些判断来自明确输入、候选空间和软件栈，不能外推为语言级胜负。后续若建立结构选择器，需要扩大形成集并使用真正未观察的 pristine holdout；若提出正式性能可移植性主张，还需对齐平台集合、相对效率基线和聚合指标。\n""",
    "B0180": """映射干预解释了早期 SpMV 反转中的主要混淆因素；RSF 可逆消融则否定了两个直观但过强的机制解释。两类结果说明，PMC 用于提出候选机制，因果结论仍需依赖改变目标机制并恢复路径的受控干预。\n""",
    "B0181": "### 6.3 双平台软件栈案例研究的范围\n",
    "B0182": """本文没有计算正式性能可移植性指标。两平台的软件版本和 profiler 口径未完全对齐，东方承担的实验范围也小于 BW1000。平台内部的 Triton/HIP 比值回答登记软件栈中的相对表现；平台之间只能检查结构趋势是否复现，不能把幅值差异单独归因于硬件。\n""",
    "B0183": """两个平台共同支持三项有限结论：SpMV mapping 会改变胜负；小 tuning set 形成的常数默认不能保证 holdout top-1；RSF 中删除超越函数不会自动缩小 Triton/HIP 差距。这些是受测平台栈上的复现实证，不是国产加速器总体规律。\n""",
    "B0187": """**平台范围。** 正式统计、完整 timing 和机制干预来自 BW1000；东方只复测冻结共同子集。两平台软件栈不一致，本文不计算性能可移植性指标，也不作硬件单因素归因。\n""",
    "B0188": """**统计范围。** BW1000 核心结果有五个独立 Slurm 任务；东方正式矩阵和 RSF 共同子集主要是单作业复测，只用于趋势检查。任务内 calls、单作业扫描和 profiler dispatch 均不作为独立实验单位。\n""",
    "B0189": """**实现与输入范围。** SpMV 比较覆盖四种登记 mapping 和有限候选空间，不等价于厂商稀疏库或所有算法。`sme3Da` 只与本轮 tuning 隔离，曾出现在早期探索中，因此不是 pristine holdout。\n""",
    "B0190": """**规则外部效度。** 两个平台的冻结默认均未同时命中 HIP 与 Triton top-1；当前数据不足以建立结构阈值选择器。正文只报告预测、oracle 与 regret，不把规则称为已泛化。\n""",
    "B0191": """**机制范围。** Timing 与干预排除了部分替代解释，但未在整体代码生成、控制流、指令调度和编译器 lowering 之间建立唯一因果链。PMC 字段仅在口径一致时比较，profiler 时间不进入正式 speedup。\n""",
    "B0194": """**应用接入。** 嵌入式 bridge 已通过设备指针、stream 和布局门禁，但真实 SeisSol `LTSRateAndState`、MPI/LTS 与完整时间推进仍为 `NOT_TESTED`；本文不报告完整应用 Triton/HIP speedup。\n""",
    "B0199": """本文将中心主张从“性能可移植性”收紧为国产 DCU 双平台软件栈案例研究。对称 SpMV 四格实验表明，映射不对称可以制造后端反转；五个独立作业将 stencil 和行归约修正为近似持平，并确认 RSF C3 的 Triton 实现稳定较慢。两平台冻结 holdout 均未使常数默认全面命中 top-1，因此选择规则保持探索性。\n""",
    "B0200": """七层计时分解区分了调用路径与设备执行，RSF 固定迭代及函数替换进一步表明，动态迭代次数和超越函数存在本身都不是当前差距的充分解释。该负面结果缩小了后续机制搜索范围，但不授权将剩余差距唯一归因于编译器或运行时。\n""",
    "B0201": """本文最可靠的结论是：Triton 在国产 DCU 软件栈上能够通过严格正确性门禁，但性能评价必须同时控制 program mapping、候选预算、实验单位和计时边界。对于 SeisSol，独立 kernel 和最小 interop 已验证，production integration 尚未验证。未来工作应使用更大的规则形成集、pristine holdout、对齐软件栈和真实 `LTSRateAndState` 路径；在这些条件满足前，不提出正式性能可移植性或完整应用替换主张。\n""",
    "B0203": """本轮修订的正式结构化证据位于 `results/bw1000/revision_v1/` 与 `results/dongfang/revision_v1/`；相应方法与结论报告位于 `docs/reports/spmv_four_cell_revision_v1.md`、`nonspmv_independent_jobs_revision_v1.md`、`timing_decomposition_revision_v2.md`、`rsf_c3_causal_ablation_revision_v2.md` 和 `dongfang_spmv_revision_v1.md`。\n""",
    "B0204": """原始 stdout、stderr、候选账本、环境记录、源码 SHA256、profiler 时间线和失败运行保存在 `results/raw/`。正文不再列举作业流水；复现包通过协议文件、结构化汇总和原始目录之间的路径关联恢复完整 provenance。受限平台软件不随论文重新分发。\n""",
}


DELETIONS = {
    "B0004", "B0005", "B0006", "B0008", "B0012", "B0013",
    "B0069", "B0070", "B0071", "B0103", "B0104", "B0105", "B0106",
    "B0113", "B0114", "B0115", "B0116", "B0196", "B0197",
    *{f"B{i:04d}" for i in range(124, 157)},
}


BLOCK_ITEMS = {
    "B0001": ["REV-R4"], "B0003": ["REV-R1", "REV-R3", "REV-R4", "REV-S3"],
    "B0004": ["REV-R1", "REV-R2"], "B0005": ["REV-R1", "REV-R2"],
    "B0006": ["REV-R1", "REV-R2", "REV-R6"], "B0007": ["REV-R1", "REV-R4"],
    "B0008": ["REV-R1", "REV-R2", "REV-R7"], "B0009": ["REV-R4"],
    "B0011": ["REV-R1", "REV-R3", "REV-R4", "REV-S3"],
    "B0012": ["REV-R1", "REV-R2"], "B0013": ["REV-R1", "REV-R2", "REV-R6"],
    "B0014": ["REV-R1", "REV-R4"], "B0018": ["REV-R3"],
    "B0021": ["REV-R3", "REV-R4"], "B0023": ["REV-R3", "REV-R4", "REV-R6", "REV-R7", "REV-R8", "REV-S3"],
    "B0031": ["REV-R4"], "B0032": ["REV-R4"],
    "B0055": ["REV-R6"], "B0056": ["REV-R6", "REV-R8"], "B0057": ["REV-S1"],
    "B0058": ["REV-R6", "REV-R8", "REV-S1"], "B0063": ["REV-R6"], "B0064": ["REV-R6", "REV-R8"],
    "B0067": ["REV-R7"], "B0068": ["REV-R2", "REV-R7"], "B0069": ["REV-R2", "REV-R7"],
    "B0070": ["REV-R2", "REV-R7"], "B0071": ["REV-R2", "REV-R7"],
    "B0073": ["REV-R7"], "B0074": ["REV-R4", "REV-R7"],
    "B0085": ["REV-R2", "REV-R7"], "B0086": ["REV-R2", "REV-R7"],
    "B0087": ["REV-R2", "REV-R6", "REV-R7"], "B0088": ["REV-R2", "REV-R6", "REV-R7"],
    "B0090": ["REV-R2", "REV-R7"], "B0096": ["REV-R2", "REV-R7"],
    "B0098": ["REV-R2", "REV-R6", "REV-R7"], "B0099": ["REV-R2", "REV-R6"],
    "B0100": ["REV-R2", "REV-R6"], "B0101": ["REV-R2", "REV-R6", "REV-R7"],
    "B0102": ["REV-R2", "REV-R6", "REV-R8"],
    "B0103": ["REV-R2", "REV-R6"], "B0104": ["REV-R2", "REV-R6"],
    "B0105": ["REV-R2", "REV-R6"], "B0106": ["REV-R2", "REV-R6"],
    "B0108": ["REV-R2", "REV-S3"], "B0109": ["REV-R2", "REV-S3"],
    "B0110": ["REV-R2", "REV-S3"], "B0111": ["REV-R2", "REV-S3"],
    "B0112": ["REV-R2", "REV-S3"], "B0113": ["REV-R2", "REV-S3"],
    "B0114": ["REV-R2", "REV-S3"], "B0115": ["REV-R2", "REV-S3"], "B0116": ["REV-R2", "REV-S3"],
    "B0123": ["REV-R2", "REV-R4", "REV-R6", "REV-R8", "REV-S3"],
    "B0157": ["REV-R2", "REV-R4"], "B0159": ["REV-R2", "REV-R4", "REV-R6", "REV-R7", "REV-R8", "REV-S3"],
    "B0174": ["REV-R8"], "B0175": ["REV-R6", "REV-R7", "REV-R8", "REV-S3"],
    "B0176": ["REV-R4", "REV-R8"], "B0180": ["REV-R6", "REV-S3"],
    "B0181": ["REV-R4"], "B0182": ["REV-R4"], "B0183": ["REV-R4", "REV-R6", "REV-R8", "REV-S3"],
    "B0187": ["REV-R4", "REV-R7"], "B0188": ["REV-R2", "REV-R7"],
    "B0189": ["REV-R6", "REV-R8"], "B0190": ["REV-R6", "REV-R8"],
    "B0191": ["REV-R7", "REV-S3"], "B0194": ["REV-R4"],
    "B0196": ["REV-R2"], "B0197": ["REV-R2"],
    "B0199": ["REV-R3", "REV-R4", "REV-R6", "REV-R7", "REV-R8"],
    "B0200": ["REV-R3", "REV-R4", "REV-R7", "REV-S3"],
    "B0201": ["REV-R3", "REV-R4", "REV-R8", "REV-S3"],
    "B0203": ["REV-R2"], "B0204": ["REV-R2"],
}

for i in range(124, 157):
    BLOCK_ITEMS[f"B{i:04d}"] = ["REV-R2"]


ITEM_META = {
    "REV-R1": ("压缩双语摘要并只保留问题、方法、关键发现、边界和贡献", "must_fix", "major", "双语摘要"),
    "REV-R2": ("将作业号、诊断史和版本流水移出主文", "must_fix", "major", "结果与数据声明"),
    "REV-R3": ("依据近期工作收紧并重建原创贡献", "must_fix", "major", "引言与相关工作"),
    "REV-R4": ("校准性能可移植性构念并采用双平台软件栈案例研究定位", "must_fix", "major", "标题、摘要、RQ、讨论与结论"),
    "REV-R5": ("完成资金、CRediT和AI声明", "must_fix", "minor", "稿末声明"),
    "REV-R6": ("以四格对称映射和相同候选预算处理SpMV baseline", "must_fix", "major", "SpMV方法、结果与讨论"),
    "REV-R7": ("以独立作业为统计单位并分解计时边界", "must_fix", "major", "统计方法与性能结果"),
    "REV-R8": ("用冻结holdout验证规则并报告regret", "must_fix", "major", "规则、讨论与结论"),
    "REV-R9": ("操作化Production gate", "must_fix", "minor", "工程接入门禁"),
    "REV-S1": ("增加集中式correctness contract", "should_fix", "minor", "正确性方法"),
    "REV-S2": ("区分2019 Triton-C与当前Python/JIT/AMD支持", "should_fix", "minor", "背景与相关工作"),
    "REV-S3": ("用可逆干预校准RSF机制表述", "should_fix", "minor", "摘要、RSF结果与结论"),
    "REV-S4": ("首次展开缩写并补最低成本验证", "consider", "minor", "首次缩写与表10"),
}

DECLINED_TARGETS = {
    "REV-R5": ["B0208"], "REV-R9": ["B0080"],
    "REV-S2": ["B0017"], "REV-S4": ["B0016"],
}


def build_authority():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    all_targets = {item_id: [] for item_id in ITEM_META}
    for block_id, ids in BLOCK_ITEMS.items():
        operation = "delete_block" if block_id in DELETIONS else "replace_block"
        for item_id in ids:
            all_targets[item_id].append({"block_id": block_id, "allowed_operations": [operation]})
    for item_id, blocks in DECLINED_TARGETS.items():
        all_targets[item_id] = [{"block_id": block, "allowed_operations": ["replace_block"]} for block in blocks]
    for item_id in all_targets:
        all_targets[item_id].sort(key=lambda row: row["block_id"])

    items = []
    for ordinal, (item_id, (description, obligation, severity, section)) in enumerate(ITEM_META.items(), 1):
        items.append({
            "id": item_id,
            "source_refs": [{"seat": "EIC", "channel": "editorial", "ordinal": ordinal, "subclaim_ordinal": 0}],
            "description": description,
            "reviewer": "Integrated revision roadmap",
            "severity": severity,
            "evidence_anchor": {"anchor_type": "text", "locator": f"integrated_revision_roadmap.md::{item_id[4:]}", "quote": description},
            "confidence": 5,
            "competence_basis": "Integrated editorial and research-quality review",
            "obligation_class": obligation,
            "cost_scope": {"kind": "section", "locator": section},
            "consequence_if_unaddressed": {"code": "claim_scope_unsupported", "target": {"kind": "section", "locator": section}},
            "target_section": section,
            "suggested_action": description,
            "consensus_level": "CONSENSUS-4",
            "verification_criteria": f"修订稿在{section}中完成该义务且不越过证据边界",
            "proposed_targets": all_targets[item_id],
        })
    roadmap = {
        "schema_version": "revision-roadmap/1.0", "revision_round": 1,
        "base_draft_sha256": sha(BASE), "block_manifest_sha256": sha(MANIFEST),
        "items": items, "total_items": len(items),
        "obligation_counts": {"must_fix": 9, "should_fix": 3, "consider": 1},
        "editorial_decision": "Major Revision",
        "consensus_summary": "中心主张强于设计；本轮用新实验补强比较、统计、holdout、timing与RSF机制，同时收紧性能可移植性构念。",
        "dissenting_opinions": [],
    }
    roadmap_path = OUT / "revision-roadmap-round1.json"
    write_json(roadmap_path, roadmap)

    claim_surface = {
        "schema_version": "claim-surface-manifest/1.0", "revision_round": 1,
        "roadmap_sha256": sha(roadmap_path), "base_draft_sha256": sha(BASE),
        "claim_intent_sources": [], "surfaces": [],
    }
    claim_path = OUT / "claim-surface-manifest-round1.json"
    write_json(claim_path, claim_surface)

    event_id = "AUTHOR-EVENT-2026-10-16-confirm"
    choices = []
    selected = {"REV-R1", "REV-R2", "REV-R3", "REV-R4", "REV-R6", "REV-R7", "REV-R8", "REV-S1", "REV-S3"}
    for item_id in ITEM_META:
        if item_id in selected:
            choices.append({"item_id": item_id, "author_event_id": event_id, "author_triage": "will_address",
                            "authorized_targets": all_targets[item_id], "claim_strength_authorizations": []})
        else:
            choices.append({"item_id": item_id, "author_event_id": event_id, "author_triage": "wont_address",
                            "author_reason": "作者确认本轮暂不修改该项。",
                            "authorized_targets": [], "claim_strength_authorizations": []})
    author_input = {
        "schema_version": "author-adjudication-input/1.0",
        "author_events": [{"event_id": event_id, "source": "explicit_session_user_message", "actor_role": "author",
                           "input_sha256": hashlib.sha256("确认".encode()).hexdigest()}],
        "display_order": {"mode": "source_traceability", "item_ids": list(ITEM_META), "author_event_id": event_id},
        "author_adjudications": choices, "collateral_authorizations": [],
    }
    write_json(OUT / "author-adjudication-input-round1.json", author_input)
    print(json.dumps({"status": "AUTHORITY_INPUTS_WRITTEN", "roadmap": str(roadmap_path), "claim_surface": str(claim_path),
                      "base_hash": manifest["base_draft_hash"], "operations": len(REPLACEMENTS) + len(DELETIONS)}, ensure_ascii=False))


def build_patch():
    roadmap_path = OUT / "revision-roadmap-round1.json"
    claim_path = OUT / "claim-surface-manifest-round1.json"
    adjudication_path = OUT / "author-adjudication-round1.json"
    adjudication = json.loads(adjudication_path.read_text())
    manifest = json.loads(MANIFEST.read_text())
    hashes = {row["block_id"]: row["old_hash"] for row in manifest["blocks"]}
    ops = []
    for block_id in sorted(set(REPLACEMENTS) | DELETIONS):
        common = {"block_id": block_id, "old_hash": hashes[block_id], "roadmap_item_ids": BLOCK_ITEMS[block_id],
                  "claim_strength_changes": [], "collateral_authorization_ids": []}
        if block_id in DELETIONS:
            ops.append({"op": "delete_block", **common})
        else:
            ops.append({"op": "replace_block", **common, "new_text": REPLACEMENTS[block_id]})
    projection = {k: adjudication[k] for k in ("author_events", "display_order", "author_adjudications", "collateral_authorizations")}
    patch = {
        "patch_format_version": "1.1", "authorization_context": "review_roadmap", "revision_round": 1,
        "base_draft_hash": manifest["base_draft_hash"], "roadmap_sha256": sha(roadmap_path),
        "author_adjudication_sha256": sha(adjudication_path),
        "author_decision_digest": hashlib.sha256(jcs(projection)).hexdigest(),
        "claim_surface_manifest_sha256": sha(claim_path), "ops": ops,
        "emitted_by": "academic-paper revision mode / draft_writer",
    }
    write_json(OUT / "revision-patch-round1.json", patch)
    print(json.dumps({"status": "PATCH_WRITTEN", "operations": len(ops)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("authority", "patch"))
    args = parser.parse_args()
    build_authority() if args.mode == "authority" else build_patch()


if __name__ == "__main__":
    main()
