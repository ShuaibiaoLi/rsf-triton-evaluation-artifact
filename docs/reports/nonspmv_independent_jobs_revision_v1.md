# 非 SpMV 独立作业层配对复测

## 状态

`PASS`。BW1000 Slurm 数组 `925740_[0-4]` 的五个独立任务均为 `COMPLETED/0`。偶数任务采用 HIP→Triton，奇数任务采用 Triton→HIP；每个任务独立经历 AOT/JIT 冷态、correctness、warmup 和稳态测量。三个登记 case 在两端均通过 correctness。

## 证据与统计单位

- 作业脚本：`BW1000_test/generic_microbench/run_nonspmv_independent_jobs_v1.slurm`
- 原始证据：`results/raw/bw1000/revision-v1-nonspmv-paired-925740/`
- 聚合器：`scripts/aggregate_nonspmv_independent_jobs.py`
- 结构化聚合：`results/bw1000/revision_v1/formal/nonspmv-independent-jobs-v1.json`
- 统计单位：每个独立 Slurm 任务内 batch=128 的 per-call 中位数；kernel call 不作为独立重复。

## 结果

比值定义为 Triton/HIP，小于 1 表示 Triton 较快。

| case | 五任务比值中位数 | P5–P95 | bootstrap median 95% CI | 方向是否一致 |
|---|---:|---:|---:|---|
| interior stencil large | 1.0021 | 0.9965–1.0035 | 0.9952–1.0037 | 否，实质持平 |
| row reduction | 0.9963 | 0.9952–0.9964 | 0.9952–0.9964 | 是，但效应不足 0.5% |
| RSF C3, n=65537 | 1.9836 | 1.9500–2.0405 | 1.9469–2.0464 | 是，Triton 较慢 |

interior stencil 的区间跨越 1，不能宣称任何后端获胜。row reduction 的方向虽在五个任务中一致，但效应只有约 0.37%，应解释为工程等价而不是有意义的加速。RSF-C3 的五个任务均接近 2×，与 E4 batch 边界结果一致，确认该差距不是单作业偶然波动。

## E2 结论

结合 SpMV 数组 `924336_[0-4]`，计划中的五类代表点已经具有独立作业层证据：`apache2`、`dw8192`、`orani678`、row reduction、interior stencil 和 RSF-C3。核心方向中，SpMV 登记最佳映射的差异与 RSF-C3 差距稳定；规则的大规模 stencil/reduction 则应报告为约等价。首次调用和单次 event 仍由 E4 单独报告，不与本报告的稳态 batch 统计混合。
