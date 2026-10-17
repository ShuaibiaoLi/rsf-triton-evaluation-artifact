# 论文评审整改与公开包完成度审计 v1

审计日期：2026-10-18。审计依据为当前 WSL 权威工作区、评审文件、回传的 BW1000/东方实验记录以及本地冻结公开包。`PASS` 只表示用户目标中相应交付已经由当前证据支持，不扩大论文结论。

## 论文范围与三项贡献

| 要求 | 状态 | 权威证据 |
|---|---|---|
| 保留 SeisSol 官方 RSF 语义 | PASS | 主稿 3.2、3.5、5；FP64 reference、golden vectors、C0—C6 契约与 source audit |
| 保留独立内核 | PASS | `BW1000_test/generic_microbench/` 正式源码、任务级结果与 correctness gate |
| 保留 TPV101 应用背景 | PASS | 主稿第5节、图6、官方 HIP 2 s TPV101 日志与 source data |
| 舍弃完整 SeisSol 生产接入 | PASS | 新标题/RQ3/贡献/结论均不以生产接入为贡献；生产路径保持 `NOT_TESTED` |
| 公平映射、计时语义、可逆归因三主线 | PASS | 主稿标题、RQ1—RQ3、三项贡献、4.1—4.4、结论相互对齐 |

## 十一项优化方案

逐项适合性、实施状态和边界见 `reviewer_revision_action_matrix_v1.md`。其中：

- 1、2、4—10 已执行；规则算子采用诚实的描述性作业级分析，不事后制造等效性门槛。
- 3 已形成可上传的本地冻结包；作者已批准代码采用 MIT、数据/图件/文档采用 CC BY 4.0。
- 11 经评估不属于修复中央推理的最小必要项；现有 RSF C2/C3、C4—C6、函数隔离和 Newton case-mix 已覆盖本文所需非线性结构，新增无关内核被明确延期。

## 公开证据包

目录：`release/rsf-triton-artifact-v1/`。

已验证：

1. 任务级与聚合数据、失败记录、环境、协议、内核源码、Slurm 入口、图件和 source data 均在包内。
2. `verify_artifact.py` 可复核两个平台的 SpMV 反转、非 SpMV 作业级统计、RSF 负面消融、empty-kernel profiler 数值及 TPV101 三个百分比。
3. `reproduce_aggregate_stats.py` 可重建 18 项任务级聚合。
4. `figure_source_manifest.json` 强制 Figure 1—7 唯一映射；旧图号重复 source data 未进入公开包。
5. 全量 `SHA256SUMS` 验证通过；公开包内绘图脚本已在隔离临时目录成功运行。

## 外部发布状态

版权所有者已批准代码采用 MIT、数据/图件/文档采用 CC BY 4.0。发布包包含正式许可文件及相应 SPDX 语义。下一步按包内 `UPLOAD.md` 创建 GitHub 仓库和冻结 `v1.0.0` release；仓库 URL、release commit/tag 和 DOI 只能在实际创建后写入论文。
