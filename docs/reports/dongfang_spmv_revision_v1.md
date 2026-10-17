# 东方 SpMV 四格映射与冻结 holdout（revision v1）

## 结论状态

`PASS_WITH_SCOPE`。东方平台已按与 BW1000 相同的 revision-v1 语义和候选预算完成：96 个 tuning 候选、三个代表矩阵的冻结四格运行，以及规则冻结后的 `sme3Da` 48 候选 holdout。全部候选均通过独立 CPU FP64 reference。该平台结果用于验证结构趋势和软件栈迁移边界，不与 BW1000 做绝对延迟比较，也不构成 BW1000 性能结论的替代证据。

## 可追溯证据

- tuning 修复作业：22427200，`COMPLETED/0`；96/96 候选 correctness `PASS`。
- tuning 原始证据：`results/raw/dongfang/revision-v1-spmv4-tuning-22427200/`。
- WSL 独立重放：`results/dongfang/revision_v1/tuning/tuning-ledger-local-replay.json`。
- frozen rule：`results/dongfang/revision_v1/tuning/frozen_rule_v1.json`，SHA256 `5be85ef87a4bbc9bcf809392d9e67880b7cf8fa32de74905966ebf5a5a6faa98`。
- 正式代表作业：22431674，`COMPLETED/0`；12/12 记录 correctness `PASS`。
- 正式 WSL 重放：`results/dongfang/revision_v1/formal/spmv-four-cell-formal-local-replay.json`。
- holdout 作业：22431696，`COMPLETED/0`；48/48 候选 correctness `PASS`。
- holdout 原始证据：`results/raw/dongfang/revision-v1-spmv4-holdout-22431696/`。
- holdout WSL 重放及评估：`results/dongfang/revision_v1/holdout/sme3Da-ledger-local-replay.json`、`sme3Da-evaluation-v1.json`。

远端账本与 WSL 独立重放账本在忽略机器相关的 `source` 路径后语义相等。失败作业 22427195（Bash 特殊变量 `GROUPS`）和 22431672（内联 Python 解析失败）继续保留，不纳入性能结论。

## 东方独立冻结配置

| backend | mapping | block | warps / groups | tuning normalized geometric mean |
|---|---|---:|---:|---:|
| HIP | cooperative-row | 256 | 2 | 1.003 |
| HIP | one-thread-per-row | 128 | 8 | 3.211 |
| Triton | block-row | 128 | 1 | 1.012 |
| Triton | row-program | 256 | 1 | 1.056 |

配置只根据东方 tuning set `bcsstk17`、`poisson3Da` 独立冻结，没有复用 BW1000 最优参数。两个 tuning 矩阵不足以建立结构阈值，因此该规则仍是常数默认配置。

## 三个代表矩阵

作业 22431674 是一次冻结配置的单作业复测，数值用于共同子集趋势检查，不具备独立作业层统计推断资格。

| matrix | HIP cooperative/thread | Triton block/row | row-level Triton/HIP | cooperative Triton/HIP | best Triton/best HIP |
|---|---:|---:|---:|---:|---:|
| `dw8192` | 1.288 | 1.012 | 8.721 | 6.850 | 8.721 |
| `apache2` | 0.974 | 0.528 | 3.601 | 1.952 | 1.952 |
| `orani678` | 0.0667 | 1.382 | 0.311 | 6.443 | 4.662 |

数值均为“前者延迟/后者延迟”，小于 1 表示分子更快。`orani678` 的 row-level 比值为 0.311，但加入 cooperative mapping 后 best/best 比值为 4.662，再次说明旧反转主要受映射不对称影响。东方上三个代表矩阵的候选集合内 best HIP 均快于 best Triton；该趋势与 BW1000 一致，但倍率不能跨平台直接比较。

## 冻结规则 holdout

规则冻结后，22431696 对 `sme3Da` 一次性运行完整 48 候选；评估没有回改 frozen rule。

| backend | frozen prediction | holdout oracle | top-1 | regret |
|---|---|---|---:|---:|
| HIP | cooperative, block 256, groups 2, 55.521 us | cooperative, block 256, groups 1, 55.414 us | no | 0.19% |
| Triton | block-row, block 128, warps 1, 195.443 us | row-program, block 128, warps 1, 173.522 us | no | 12.63% |

HIP 未命中 group，但预测性能与 oracle 几乎等价；Triton 未命中 mapping，产生 12.63% regret。因此东方结果支持“冻结流程可执行且 HIP 常数默认在该输入上稳健”，但不支持把 Triton 常数默认写成可靠的样本外规则。`sme3Da` 曾出现在早期探索中，仍只能称 revision holdout，不称 pristine holdout。

## 与 BW1000 的证据边界

- 两个平台共同支持：映射选择会改变 SpMV 胜负，简单 row-to-row 反转不足以证明后端整体优势。
- 两个平台 holdout 都未命中两端全部 top-1，两个 tuning 矩阵形成的常数默认不能上升为普适结构规则。
- BW1000 是目标平台，正式性能和统计结论仍以 BW1000 实测为准；东方只提供第二软件栈的趋势复现。
