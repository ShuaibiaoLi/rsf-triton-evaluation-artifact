# SeisSol 热点清单与性能可移植性阶段交付

日期：2026-09-28。研究对象：SeisSol v1.3.2，BW1000/DTK 26.04，官方 TPV101 预计算算例。

## 1. 热点清单

| 路径 | SeisSol 计时项/组件 | 计算特征 | 适合的后端 | 当前证据 |
|---|---|---|---|---|
| 波场传播 | `computeLocalIntegration` | DG 局部积分、批量小矩阵/GEMM、代码生成 | 官方 HIP/AdaptiveCpp、SYCL；Triton 仅适合局部规则 kernel | BW1000 TPV101 内置计时：7.13288 s |
| 波场传播 | `computeNeighboringIntegration` | 邻居积分、非规则数据访问、MPI/边界交换耦合 | 官方 native 优先；Triton 需先解决数据布局与调度边界 | BW1000 TPV101 内置计时：7.17273 s |
| 动态破裂/RSF | `computeDynamicRupture` | 摩擦定律、Newton、状态更新、非线性分支 | HIP/AdaptiveCpp native；Triton 可作为语义 kernel/局部实验 | BW1000 TPV101 内置计时：7.18198 s；C3 native/Triton 同卡微基准 |
| 点源 | `computePointSources` | 点源批量更新 | 当前 TPV101 无点源样本 | `0 s`，不可作为性能结论 |
| I/O | NetCDF/HDF5、receiver/fault writer | I/O 与 CPU/MPI 控制流 | 不适合 Triton | TPV101 日志记录 writer 时间 |
| 编排 | MPI、LTS、mesh/PUML、输出调度 | 控制流、通信和不规则拓扑 | MPI/native | 官方 executable smoke 已通过 |

## 2. 两条代表性路径

### 路径 A：波场传播/代码生成

`computeLocalIntegration` 与 `computeNeighboringIntegration` 合计 14.30561 s，占官方 TPV101 `Time spent in compute kernels` 21.4876 s 的约 66.6%。该路径覆盖 SeisSol 的 DG 局部/邻居积分、生成代码、GPU offloading 和非规则数据访问，是评估官方 HIP/AdaptiveCpp 迁移价值的主路径。当前没有把 Triton 强行用于 MPI/邻居拓扑；后续只对局部规则 kernel 做受控原型。

### 路径 B：动态破裂/RSF

`computeDynamicRupture` 为 7.18198 s，占 compute kernels 的约 33.4%。CPU 与 BW1000 的既有 FL=3/FL=0 端到端对照显示 Aging RSF 约占 Dynamic Rupture 时间的 47.6%（CPU）和 27.0%（BW1000），因此 RSF 是真实热点，但不是完整 SeisSol 的全部性能问题。C3 公式提取微基准显示 HIP native 比 Triton 快约 8.4–9.3 倍；该结果用于性能边界分析，不支持 Triton 加速主张。

## 3. 统一对照矩阵

| 后端/层级 | 正确性 | 性能 | 计时边界 | 工程成本证据 | 状态 |
|---|---|---|---|---|---|
| CPU SeisSol | TPV101 smoke PASS；FL=3/FL=0 对照 | Dynamic Rupture 与 RSF 占比已记录 | 端到端官方算例 | 已有 CPU 构建和运行记录，尚未量化开发时间 | ACCEPTED baseline |
| 官方 HIP/AdaptiveCpp SeisSol | TPV101 GPU smoke PASS；识别 `GPU: true`、`Device API: HIP` | 波场/动态破裂内置计时已记录 | 完整 2 s TPV101 | `gfx936` 兼容补丁、Lua EASI、MPI/NetCDF/HDF5 依赖记录齐全 | ACCEPTED application baseline |
| Triton RSF 语义适配器 | C0–C6 PASS；适配器作业执行并记录元数据 | C3 同卡 193–538 µs；native 22–58 µs | kernel 同步计时 | 适配器源码与 kernel hash 已记录；开发工时尚未量化 | ACCEPTED semantic boundary, NOT production path |
| Triton 波场路径 | 尚未实现 | 尚未实现 | — | — | NOT_TESTED |

## 4. Profiling 状态

本阶段完成了两层 profiling：

1. **应用层 profiling：PASS。** 官方 TPV101 内置 LoopStatistics 已记录 local integration、neighboring integration、dynamic rupture、总 kernel 时间、FLOP/s 和 I/O writer 时间。
2. **硬件计数器 profiling：NOT_ACCEPTED。** 既有 `rocprof` probe 的应用执行成功，但 DTK 后处理仍受 `rocminfo` 硬编码路径影响，未形成可审计的 occupancy、带宽和指令计数 CSV。

因此，本阶段可以交付热点排序和应用级时间分解，不能宣称硬件计数器归因。

## 5. 阶段结论

SeisSol 的主要研究对象已从单一 RSF 扩展为两条路径：波场传播/代码生成是主要应用级热点，动态破裂/RSF 是非线性语义和 Triton 适配案例。当前最可靠的论文结论是：BW1000 上官方 HIP/AdaptiveCpp SeisSol 路径可运行；Triton 可以复现 RSF 语义，但在当前 C3 公式微基准中明显慢于 native；Triton 生产接入和波场路径尚未完成。完整端到端 Triton 加速仍不成立，论文主线应是迁移边界与性能可移植性。
