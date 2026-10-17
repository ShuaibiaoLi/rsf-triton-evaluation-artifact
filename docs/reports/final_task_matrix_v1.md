# 六项阶段任务验收矩阵（v1）

| 任务 | 状态 | 证据 | 结论 |
|---|---|---|---|
| 1. 嵌入式 RSF 桥接 | `PASS` | 作业 875389/875406/875426/875437 | HIP 指针、stream、RSF 表达式和 face×padded layout 均可传入 Triton |
| 2. TPV101 最小生产 smoke | `PARTIAL` | 官方 HIP TPV101 作业 866206；Triton bridge 独立 smoke | 官方路径通过；Triton 尚未替换真实 LTSRateAndState 调用点 |
| 3. 应用级性能对照 | `PARTIAL` | 统一 C3/native、bridge steady-state 98.899 us | 有独立 kernel 和 bridge 数据；无完整 SeisSol Triton speedup |
| 4. PMC/Roofline | `PASS` | 作业 873152/873843/874792 | 完成硬件行为、经验 FP64 roof、带宽和 Roofline |
| 5. 论文整合 | `PASS` | `manuscript_cn_consolidated_v1.md` | 已区分 kernel、bridge、生产接入和未测试结论 |
| 6. 最终可复现性验收 | `PASS_WITH_SCOPE` | 源码、Slurm、原始日志、环境和设备信息 | BW1000 单卡范围内可复现；多卡/第二设备未纳入范围 |

## 审计意见后的增量

| 增量任务 | 状态 | 证据 | 边界 |
|---|---|---|---|
| event/host/batch 计时审计 | `PASS_WITH_SCOPE` | 作业 875921；`timing_boundary_audit_v1.md` | 尚无 graph replay/rocprof timeline |
| HIP/Triton API/kernel trace | `PASS_WITH_SCOPE` | 作业 875950；`extended_patterns_and_trace_v1.md` | native/Triton 调用集合尚未完全一一匹配 |
| Triton ISA/HSACO metadata | `PARTIAL` | 作业 872998；gfx936 metadata，spill=0 | 缺少同规模 HIP HSACO metadata |
| 多维 stencil | `PASS_WITH_SCOPE` | 作业 875970/875993，2D 1024×1024，Triton 70.96 μs、HIP 30.559 μs、比值2.32 | event 对照，不等于纯 ISA 分解 |
| 不均匀行长 SpMV-like | `PASS_WITH_SCOPE` | 作业 875970/875993，nnz=667264，Triton 83.84 μs、HIP 50.238 μs、比值1.67 | 仍需真实矩阵分布扩展 |
| SeisSol LTS 风格 bridge 重复验证 | `PASS_WITH_SCOPE` | 作业 875980，bad=0，99.911 μs | 仍非真实 `LTSRateAndState` 调用点 |

## 当前最终边界

本研究已经完成“国产 DCU 上科学计算算子的 Triton 性能可移植性、优化方法与迁移边界”的单卡实证闭环。完整 SeisSol TPV101 Triton 生产替换仍是 `NOT_TESTED`，不能用独立 bridge 或微基准 speedup 替代。该限制已作为论文结果的一部分明确记录。
