# RSF-C3 可逆因果消融（revision v2）

## 当前状态

`PASS_NEGATIVE_CAUSAL_RESULT`。冻结协议为 `docs/protocol/rsf_c3_causal_ablation_v2.json`。作业 925770 完成固定实际迭代回放；925877 完成函数 lookup、普通算术占位与恢复；925991 完成独立 PMC。全部版本通过各自的独立/共享 FP64 reference 和预登记的 ±10% 可逆性门槛。预登记的“移除函数路径后跨后端差距缩小”没有出现，因此 transcendental presence 不能作为当前 RSF 差距的因果解释。

失败作业 925767 的 HIP 部分通过，但 Triton 动态导入遗漏 `tolerance.py` 路径而 `FAILED/1`；证据保留在 `results/raw/bw1000/revision-v1-rsf-c3-fixed-replay-925767/`。修复仅补模块目录并清除编译警告，未改变 kernel 语义。

## 固定实际迭代回放

- 成功作业：925770，`COMPLETED/0`
- 原始证据：`results/raw/bw1000/revision-v1-rsf-c3-fixed-replay-925770/`
- 结构化结果：`results/bw1000/revision_v1/ablation/rsf-c3-fixed-replay-paired-v2.json`
- 规模：n=65537；warmup=10；repeat=30；同 stream device event；persistent inputs/outputs
- correctness：HIP 相对独立 CPU FP64 reference 最大绝对误差 `1.60e-16`，零失败；Triton fixed replay 与原始输出五字段逐项一致。

| variant | HIP median μs | Triton median μs | Triton/HIP |
|---|---:|---:|---:|
| original pre | 62.80 | 226.88 | 3.61 |
| fixed actual-iteration replay | 57.12 | 224.08 | 3.92 |
| original restore | 57.28 | 225.44 | 3.94 |

HIP 的首个 original 序列比恢复序列慢约 8.8%，仍在预登记门槛内；因此机制判断以 fixed 与紧邻的 restore 为主。固定迭代后，HIP 和 Triton 分别为 57.12 和 224.08 μs，与恢复后的 57.28 和 225.44 μs 几乎相同。去除动态收敛 predicate 和迭代数分歧没有缩小跨后端差距。

## 阶段结论

该阶段排除“实际迭代次数不同”以及“动态收敛判断本身”作为主要解释。下一阶段按冻结协议直接检验 transcendental 路径。

## 函数路径干预与恢复

作业 925877 使用固定实际迭代轨迹，依次替换不变 coefficient（`log+exp`）、`sqrt`、asinh 等效路径、全部超越函数和普通算术占位。lookup 表只包含九个登记 case 的冻结轨迹值；所有物理语义 lookup 版本与原始 fixed replay 的最大绝对差异不超过 `2.78e-16`。普通算术占位使用单独的标量 FP64 reference，不与物理 RSF 输出混作 correctness。

| variant | HIP μs | Triton μs | Triton/HIP |
|---|---:|---:|---:|
| fixed original pre | 60.48 | 235.44 | 3.89 |
| lookup coefficient (`log+exp`) | 60.00 | 233.84 | 3.90 |
| lookup `sqrt` | 58.08 | 233.52 | 4.02 |
| lookup asinh path | 39.52 | 194.80 | 4.93 |
| lookup all transcendentals | 34.08 | 192.16 | 5.64 |
| arithmetic placeholder | 32.80 | 181.84 | 5.54 |
| fixed original restore | 59.84 | 240.56 | 4.02 |

HIP 和 Triton 的恢复比值分别为 `0.989` 和 `1.022`，满足可逆性门槛。移除 asinh 或全部超越函数确实加速两个后端，但 HIP 的收益更大，跨后端差距反而扩大。该结果否定“超越函数是否存在”这一单因素解释；剩余候选包括同一循环表达式的后端 lowering、谓词/选择指令、除法、指令调度和代码体积。

结构化结果位于 `results/bw1000/revision_v1/ablation/rsf-c3-function-path-paired-v2.json`。失败的首次 lookup 作业 925778 因 Triton 普通全局常量不能在 JIT kernel 中访问而失败，原始证据完整保留；修复只把 `STEPS=61` 改为内核字面 constexpr。

## 专用 PMC

独立作业 925991 使用相同源码采集 type-3 PMC，profiler 时间不参与 speedup。原始 fixed replay 中 Triton/HIP `SQ_INSTS_VALU` 比为 `2.34`；lookup coefficient、sqrt、asinh、全部函数和普通算术占位分别为 `2.38`、`2.32`、`2.21`、`2.31` 和 `2.64`。即使所有超越函数被 lookup 或普通算术替代，Triton 仍执行约 2.3–2.6 倍的 VALU 指令。

这支持更窄的机制结论：差距与 Triton/HCU 对固定上限循环及其逐 lane 控制流/算术的 lowering 相关，而不是由某一个超越函数单独造成。PMC 是方向性计数，不能独立区分谓词化、重复计算和指令选择；不得把 profiler dispatch 时间用于正式倍率。

原始 PMC：`results/raw/bw1000/revision-v1-rsf-c3-lookup-pmc-925991/`；结构化结果：`results/bw1000/revision_v1/ablation/rsf-c3-lookup-pmc-v2.json`。

## 东方冻结协议复测

东方作业 22427075 `COMPLETED/0`，输入 TSV SHA256 与 BW1000 相同，干预源码和冻结协议 SHA256 相同。东方使用 Triton 2.1/DTK 24.04 与 native DTK 26.04，因此只比较趋势，不把绝对倍率与 BW1000 混作硬件差异。

| variant | 东方 HIP μs | 东方 Triton μs | Triton/HIP |
|---|---:|---:|---:|
| fixed original pre | 105.44 | 492.68 | 4.67 |
| lookup coefficient | 102.72 | 483.33 | 4.71 |
| lookup `sqrt` | 98.88 | 469.18 | 4.75 |
| lookup asinh path | 67.20 | 360.89 | 5.37 |
| lookup all transcendentals | 54.72 | 307.94 | 5.63 |
| arithmetic placeholder | 52.00 | 320.34 | 6.16 |
| fixed original restore | 105.60 | 484.48 | 4.59 |

HIP/Triton 恢复比均通过 ±10% 门槛。东方复现了 BW1000 的关键方向：移除函数路径后两端都加速，但跨后端差距不缩小、反而扩大。由此，负面因果结论具有第二平台栈证据；绝对倍率仍限定于各自平台内。

东方原始证据：`results/raw/dongfang/revision-v1-rsf-c3-lookup-22427075/`；结构化结果：`results/dongfang/revision_v1/ablation/rsf-c3-function-path-paired-v2.json`。

## `log` 与 `exp` 单独隔离

BW1000 作业 927685 在不变 coefficient 子表达式上分别执行 `lookup_log` 和 `lookup_exp_retain_log`。后者仍计算并写出 runtime `log`，因此只移除 `exp`。两端输出与标量 FP64 reference 完全一致，恢复比 HIP `0.9994`、Triton `1.0042`。

| coefficient variant | HIP batch μs | Triton batch μs |
|---|---:|---:|
| original pre | 5.846 | 70.366 |
| lookup `log` | 5.365 | 70.793 |
| lookup `exp`, retain `log` | 5.499 | 70.394 |
| original restore | 5.842 | 70.662 |

HIP 对两个函数替换都有小幅响应；Triton 的固定调用/代码路径成本掩盖了单个不变函数的差异。结合完整 C3 的 coefficient lookup 几乎不改变跨后端倍率，可排除循环外单次 `log` 或 `exp` 为主要原因。该子表达式结果只用于分离函数，不与完整 C3 latency 相加。

原始证据：`results/raw/bw1000/revision-v1-rsf-coefficient-isolation-927685/`；结构化结果：`results/bw1000/revision_v1/ablation/rsf-coefficient-isolation-paired-v1.json`。
