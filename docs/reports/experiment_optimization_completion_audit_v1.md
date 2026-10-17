# 实验优化方案完成度审计

审计对象：`revisions/roadmap_v1/experiment_optimization_plan.md`。本审计按原计划逐项核验证据，不以已有结果反向缩小完成标准。

## 最小充分新增实验包

| 计划项 | 状态 | 权威证据 |
|---|---|---|
| SpMV 四格对称映射 | PASS | BW1000 924088、924300、924336_[0-4]；`spmv_four_cell_revision_v1.md` |
| 五类代表案例独立作业复测 | PASS | SpMV 924336_[0-4]；non-SpMV 925740_[0-4]；两个 job-level 聚合 JSON |
| 冻结规则后的矩阵 holdout | PASS_WITH_NEGATIVE_GENERALIZATION | 924488；HIP regret 17.81%，Triton regret 2.80%，规则未回改 |
| 六类代表点 timing 分解 | PASS | 925680、925708、925715、925723；first/host/event/batch/graph/profiler/empty 边界齐全 |
| RSF 固定迭代与 transcendental 可逆消融 | PASS_NEGATIVE_CAUSAL_RESULT | 925770、925877、925991、927685；函数移除没有缩小跨后端差距 |

最小充分包在 BW1000 已闭合。负面结果属于有效实验结果，不改写为正向机制证据。

## E1–E8 要求

| ID | 状态 | 审计结论 |
|---|---|---|
| E1 对称映射与公平 baseline | PASS | 四格均有独立 CPU FP64 reference、等候选预算和候选账本；论文只能比较登记映射或同映射后端差异。 |
| E2 独立作业层复测 | PASS | 核心案例均至少 5 个独立 Slurm 任务；统计单位是任务内中位数。 |
| E3 tuning/holdout 分离 | PASS_WITH_LIMITATION | 本轮严格隔离且规则未回改；`sme3Da` 曾用于更早探索，不是 pristine holdout。 |
| E4 timing 边界 | PASS | 七层边界闭合；profiler 作业与正式 latency 分离。 |
| E5 RSF 因果消融 | PASS_NEGATIVE_CAUSAL_RESULT | 固定迭代、log、exp、sqrt、asinh、全部 lookup、普通算术占位、恢复和 PMC 均完成；预登记因果假设未成立。 |
| E6 correctness contract | PASS | 集中 JSON 契约、独立 reference、容差、非有限/不收敛策略和边界集已登记并执行。 |
| E7 对称调优账本 | PASS | 96 个候选、correctness 过滤、样本、tie-break、冻结时间和源码 SHA256 均保留。 |
| E8 平台/软件栈记录 | PASS | BW1000 每个正式目录含 environment/source SHA/raw；东方 RSF 共同子集及 SpMV tuning/formal/holdout 均已回传，含环境、源码 SHA、raw、账本和失败记录。 |

## 条件性 E9–E10

- E9 production-path precursor 是条件性新增项，不属于最小充分包。完整 MPI/LTS/生产替换仍为 `NOT_TESTED`，不得声称 production integration。
- E10 经验 Roofline 已有历史实验资产，但本轮修订没有新增 ceiling。当前 RSF 和不规则 SpMV 机制结论来自 timing/PMC/干预，不依赖基础 Roofline 单独归因。

## 失败证据完整性

已确认保留至少以下本轮失败或无效运行：925673（Triton import）、925698（首次调用语义无效）、925767（fixed replay import）、925778（Triton constexpr）、东方 22427195（Bash `GROUPS` 特殊变量）。它们没有被成功作业覆盖或删除。

## 东方共同子集闭环

东方 RSF 作业 22427075 完成共同消融；SpMV tuning 22427200、formal 22431674 和冻结后 holdout 22431696 均为 `COMPLETED/0`。远端账本与 WSL 独立重放在忽略路径字段后语义相等。holdout 的 HIP/Triton regret 分别为 0.19% 和 12.63%，规则没有回改。详细证据见 `docs/reports/dongfang_spmv_revision_v1.md`。

## 总体审计结论

原计划的必须项 E1–E8 和“最小充分新增实验包”均有本轮权威证据并已闭环。E9–E10 是计划明确标注的条件性项目：当前不声称 production integration，也不新增依赖 Roofline 的机制归因，因此不构成本轮完成阻塞。所有负面结果、局限与失败运行均继续保留。
