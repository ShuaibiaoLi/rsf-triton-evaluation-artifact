# RSF-C3 静态循环上界干预（revision v3）

## 判定与适用范围

`PASS_BOUNDED_CAUSAL_RESULT`：在 BW1000 `gfx936`、九个冻结 Aging-RSF 输入、`n=65537` 的 fixed replay 中，同时把 HIP 与 Triton 的**静态循环上界**由 60 改为 22，Triton/HIP 任务级中位比值由 3.714 降为 2.900，五个独立作业方向一致。该干预与 v2 的“固定实际迭代轨迹”不同：v2 保留 60 轮静态循环，v3 才改变编译期上界。22 来自九例已知实际迭代次数 `8,22,6,4,4,6,6,1,6` 的最大值，是 oracle；不能作为未知输入或 SeisSol 生产路径的通用上界。

## 门禁与正式计时

本地独立 CPU FP64 检查在 `N=1,9,65,257,4099,65537` 上确认两上界的五字段输出相同，并以错误的 8 轮上界构造非零差异。首次 BW1000 smoke 作业 `935980` 中，HIP 正确性通过，但 `N=1` 的微秒级恢复检查未过 ±10% 门槛；原始失败记录保留，不作性能结论。修订后的 smoke 作业 `935988` 对六种规模的 HIP/Triton 五字段均通过 correctness；正式恢复判定移至 `n=65537` 的稳态计时。HIP 在正式规模相对独立标量 reference 的最大绝对误差为 `1.60e-16`；Triton 两变体与原始 GPU 输出逐字段一致。

正式数组作业 `936008_[0-4]` 对应实际作业 `936009`、`936012`、`936013`、`936017`、`936008`，全部 `COMPLETED/0`。每个独立任务交替选择 HIP-first/Triton-first，分别对 pre60、bound22、restore60 做 30 次无 profiler device-event 测量并取任务内中位数；两端的恢复/pre 均满足 ±10%。表中每行是一个独立作业单位，不把 30 次调用当作独立重复。

| 任务 | 顺序 | HIP 60/22 (μs) | Triton 60/22 (μs) | Triton/HIP 60→22 |
|---:|---|---:|---:|---:|
| 0 | HIP-first | 62.56 / 53.60 | 218.72 / 154.72 | 3.496→2.887 |
| 1 | Triton-first | 63.20 / 53.84 | 218.48 / 156.16 | 3.457→2.900 |
| 2 | HIP-first | 57.36 / 53.68 | 224.96 / 159.28 | 3.922→2.967 |
| 3 | Triton-first | 59.92 / 53.60 | 222.56 / 160.64 | 3.714→2.997 |
| 4 | HIP-first | 58.08 / 53.60 | 218.88 / 155.12 | 3.769→2.894 |

任务内 22/60 比值的五任务中位数，HIP 为 `0.895`（描述性 P5—P95 `0.853—0.933`），Triton 为 `0.709`（`0.708—0.720`）；跨端 Triton/HIP 比值从 `3.714`（`3.465—3.891`）降至 `2.900`（`2.888—2.991`）。这是冻结输入与作业级描述统计，不是形式显著性或等效性检验。

## 独立 PMC 与机制边界

独立剖析作业 `936030` `COMPLETED/0`，源码和输入哈希与正式计时一致，profiler 时间未用于倍率。`SQ_INSTS_VALU` 从 60 到 22：HIP `3,812,013→3,695,163`（0.969），Triton `8,610,000→3,351,750`（0.389）。各自后端的 VMEM 读写计数、寄存器报告和 grid/workgroup 在两个上界间保持一致。Triton 在这次干预中的 VALU 降幅更大，与性能比值缩小方向一致。

这一成对干预支持：在登记案例及当前编译栈下，60 轮静态上界参与形成 Triton fixed replay 的额外指令量和相对延迟。它**不能**把效应唯一归因于 inactive-lane 算术、`tl.where`、循环展开、分支 lowering 或指令选择；编译期上界改变可能同时影响多种生成代码属性。22 轮后 Triton/HIP 仍为约 2.9，也不能声称差距已经消除。v2 的实际迭代轨迹与函数路径负结果仍成立，因为它们没有测试这个静态上界变量。下一步若要进一步区分机制，应在相同冻结输入上做 22 轮变体的 ISA/编译产物对照和受控控制流改写；若要讨论可部署优化，则必须为未知输入建立安全的动态上界策略，并重新通过正确性与生产门禁。

## 可审计证据

- 正式任务级汇总：`results/bw1000/revision_v2/ablation/rsf-c3-loop-bound-independent-jobs-v3.json`；原始目录：`results/raw/bw1000/revision-v2-rsf-c3-loop-bound-formal-{936009,936012,936013,936017,936008}/`。
- PMC 汇总：`results/bw1000/revision_v2/ablation/rsf-c3-loop-bound-pmc-v3.json`；原始目录：`results/raw/bw1000/revision-v2-rsf-c3-loop-bound-pmc-936030/`。
- 通过的 smoke：`results/raw/bw1000/revision-v2-rsf-c3-loop-bound-smoke-935988/`；保留的失败：`results/raw/bw1000/revision-v2-rsf-c3-loop-bound-smoke-935980/`。
- 本地与设备脚本：`scripts/validate_rsf_loop_bound_v3.py`、`BW1000_test/generic_microbench/rsf_c3_loop_bound_{native,triton}_v3.*`、`BW1000_test/generic_microbench/run_rsf_c3_loop_bound{,_pmc}_v3.slurm`。每个原始目录保存环境、源码哈希、日志及原始样本；正式与 PMC 的关键源码哈希一致。
- 只读复核：在工作区根目录运行 `python3 scripts/verify_rsf_c3_loop_bound_v3.py`，重新从五个任务的 30 次原始样本计算中位数与比值，检查恢复门槛、源码哈希、任务身份和独立 PMC 字段；当前输出为 `PASS`。此复核不替代设备重跑。

这份 v3 证据不属于旧 `v1.0.1`，随 `v1.0.2` 纳入公开证据包；包内 `SHA256SUMS` 与只读复核脚本提供完整性和数值检查。完整 SeisSol Triton 生产替换仍为 `NOT_TESTED`。
