# Timing 边界分解（revision v2）

## 当前状态

`PASS`。BW1000 Triton 作业 925680、通用 HIP 作业 925708 和 RSF-C3 HIP 作业 925715 均 `COMPLETED/0`，所有登记 case 通过 correctness，并记录首次调用、预热后 host submit-to-sync、单次 device event、连续 batch 单次值和 graph replay。独立 profiler 作业 925723 `COMPLETED/0`，回传逐 dispatch 的 kernel start/end 时间线。正式 latency 与 profiler 结果保持分离。

失败作业 925673 因动态导入 RSF probe 时未登记同目录 `tolerance.py` 导入路径而 `FAILED/1`；失败日志保留在 `results/raw/bw1000/revision-v1-timing-v2-triton-925673/`。修复只改变模块搜索路径，925680 使用修复后源码重新执行全部 case。

## 证据

- 探针：`BW1000_test/generic_microbench/timing_decomposition_triton_v2.py`
- 作业：`BW1000_test/generic_microbench/run_timing_decomposition_triton_v2.slurm`
- 原始证据：`results/raw/bw1000/revision-v1-timing-v2-triton-925680/`
- HIP 原始证据：`results/raw/bw1000/revision-v1-timing-v2-hip-925708/`、`results/raw/bw1000/revision-v1-timing-v2-rsf-hip-925715/`
- HIP 结构化结果：`results/bw1000/revision_v1/timing/timing-decomposition-hip-v2.json`、`results/bw1000/revision_v1/timing/timing-decomposition-rsf-hip-v2.json`
- profiler 原始证据：`results/raw/bw1000/revision-v1-timing-v2-profiler-925723/`
- profiler 时间线：`results/bw1000/revision_v1/timing/hip-generic-kernel-timeline-v2.json`、`hip-rsf-kernel-timeline-v2.json`、`triton-kernel-timeline-v2.json`
- result SHA256：`8686f5a3765031f9cbbb4704378ccfeebb38fdb68eec0491024af664711b9c68`
- RSF-C3 HIP result SHA256：`d787c4d1fae27b276870efb3bea5bc557418db1ecda842f38a645cb6f020b01f`
- 平台：BW1000，PyTorch 2.10.0 DTK build，Triton 3.5.0。

## 结果

单位均为 μs；host、event、batch 和 graph 是不同计时边界，不互相替代。

| case | first call incl. JIT | post-JIT host | single event | batch=128 / call | graph replay event |
|---|---:|---:|---:|---:|---:|
| empty | 2,882,812 | 32.32 | 32.80 | 11.44 | 17.92 |
| stencil small | 839,030 | 55.86 | 55.04 | 28.21 | 17.92 |
| interior stencil large | 426,428 | 107.16 | 97.92 | 58.03 | 64.80 |
| row reduction | 745,369 | 282.06 | 257.60 | 221.58 | 227.84 |
| SpMV short row (`dw8192`) | 749,307 | 74.78 | 73.04 | 46.06 | 15.68 |
| SpMV long tail (`orani678`) | 538,985 | 94.17 | 84.56 | 46.81 | 35.52 |
| RSF C3, n=65537 | 992,425 | 236.19 | 228.48 | 112.48 | 116.32 |

表中为各重复的中位数；P5/P95 和逐次样本保存在原始 JSON。

### HIP 对称计时

| case | first call AOT cold | post-AOT host | single event | batch=128 / call | graph replay event |
|---|---:|---:|---:|---:|---:|
| empty | 26,800 | 13.31 | 7.36 | 2.00 | 11.76 |
| stencil small | 116.71 | 15.26 | 6.56 | 2.64 | 11.84 |
| interior stencil large | 105.64 | 70.36 | 59.84 | 57.88 | 65.12 |
| row reduction | 366.29 | 250.44 | 223.68 | 222.59 | 228.48 |
| SpMV short row (`dw8192`) | 111.45 | 17.71 | 7.04 | 5.10 | 11.84 |
| SpMV long tail (`orani678`) | 83.60 | 54.57 | 43.52 | 31.43 | 48.00 |
| RSF C3, n=65537 | 11,583.63 | 71.44 | 60.16 | 56.43 | 64.32 |

RSF-C3 HIP 对独立 CPU FP64 reference 的最大绝对误差为 `2.22e-16`，零失败。首次 AOT 调用包含运行时冷启动，不与 Triton JIT 首次调用作编译复杂度比较。

### 对称边界上的 Triton/HIP 比值

| case | single event | batch=128 / call | graph replay event |
|---|---:|---:|---:|
| empty | 4.46 | 5.72 | 1.52 |
| stencil small | 8.39 | 10.67 | 1.51 |
| interior stencil large | 1.64 | 1.00 | 1.00 |
| row reduction | 1.15 | 1.00 | 1.00 |
| SpMV short row (`dw8192`) | 10.37 | 9.03 | 1.32 |
| SpMV long tail (`orani678`) | 1.94 | 1.49 | 0.74 |
| RSF C3, n=65537 | 3.80 | 1.99 | 1.81 |

比值大于 1 表示 Triton 较慢。规则的大规模 stencil 和 reduction 在 batch/graph 边界上基本持平；小 kernel 和短行 SpMV 对调用协议高度敏感。RSF-C3 在 batch=128 后仍约慢 `1.99×`，所以其主要差距不能只归因于单次 Python 提交开销，但在可逆消融完成前仍不能归因为 transcendental lowering。

### 独立 profiler kernel timeline

作业 925723 使用 `hipprof --hip-trace --stats`，SQLite 原始数据库保留每次 dispatch 的 `BeginNs`、`EndNs` 和 `DurationNs`。下表为 profiler 下的 kernel-duration 中位数，仅用于调用边界交叉核查，不用于正式 speedup。

| case | HIP kernel μs | Triton kernel μs | Triton/HIP |
|---|---:|---:|---:|
| empty | 0.64 | 0.64 | 1.00 |
| stencil small | 2.24 | 2.56 | 1.14 |
| interior stencil large | 57.12 | 62.88 | 1.10 |
| row reduction | 221.92 | 242.08 | 1.09 |
| SpMV short row (`dw8192`) | 4.72 | 8.32 | 1.76 |
| SpMV long tail (`orani678`) | 31.44 | 28.64 | 0.91 |
| RSF C3, n=65537 | 56.32 | 182.72 | 3.24 |

两个 stencil 和两个 SpMV 共用 kernel 名称，但各自形成无重叠的快/慢 duration cluster；表中按预登记的小/大、短行/长尾次序对应低/高簇。结构化 JSON 保留 1104 个 HIP 通用、184 个 HIP RSF 和 2279 个 Triton dispatch 的完整起止时间。Triton 时间线还包含 correctness 路径产生的 PyTorch 辅助 kernel；表中只选取登记的 Triton kernel 名称。

时间线进一步收紧了结论：空 kernel 的纯 dispatch 时间两端相同，而单次 event 为 32.80 对 7.36 μs，表明小算子的 event 差异主要属于调用/事件路径；大 stencil 和 reduction 的纯 kernel 差异约 9%–10%；RSF-C3 的纯 kernel 差距仍约 3.24×，需要 E5 可逆消融解释。

## 解释边界

1. `empty` 是进程内第一个 Triton kernel，其首次调用包含 runtime 初始化和 JIT；后续 case 的首次调用主要反映各自 specialization/JIT。不同 case 的 first-call 数字不能直接排序为编译复杂度。
2. device event 是同一 stream 的设备可见区间，不自动等于纯 ISA 执行时间。
3. batch=128 和 graph replay 表明固定提交/事件成分可以被摊薄，但二者协议不同。例如 short-row SpMV graph event 低于 batch event，不应把差值直接归因于编译器。
4. graph capture 在本作业七个 case 均成功；这只证明当前环境和这些固定地址调用可捕获，不代表完整 SeisSol production graph 已验证。
5. profiler 会扰动执行并改变调用调度，因此 profiler duration 只用于边界与机制核查，不替换无 profiler 的正式 latency。
6. HIP 与 Triton 现已具有对称的 host/event/batch/graph 边界和逐 dispatch 起止时间；E4 的七层计时分解闭合。
