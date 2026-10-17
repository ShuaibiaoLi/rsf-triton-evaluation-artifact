# 补充材料索引（公开归档版）

本索引指定本稿关键结论的可审计文件。证据包公开于 https://github.com/ShuaibiaoLi/rsf-triton-evaluation-artifact ，冻结版本为 `v1.0.1`。代码采用 MIT License，数据、图件与文档采用 CC BY 4.0。GitHub 公开仓库不是匿名审稿入口；尚未取得 DOI。

| 材料 | 用途 | 冻结文件 |
|---|---|---|
| SM1 | 作业级输入与描述性统计 | `results/bw1000/revision_v1/formal/nonspmv-independent-jobs-v1.json`；每个 case 给出五个独立 Slurm array task 的 HIP/Triton 任务内中位数、顺序和原始目录 |
| SM2 | SpMV 对称候选账本 | `results/bw1000/revision_v1/tuning/tuning-ledger-local-replay.json`；96 候选、2 个 tuning 矩阵、每 cell 12 个候选与 selection metric |
| SM3 | 冻结规则与 tie-break | `results/bw1000/revision_v1/tuning/frozen_rule_v1.json`；冻结时间、协议 SHA256、账本 SHA256、normalized-geomean 选择规则和 tie-break |
| SM4 | 正式四格作业级结果 | `results/bw1000/revision_v1/formal/spmv-four-cell-independent-jobs-v1.json`；五个独立任务、P5/P95 与原始任务目录 |
| SM5 | 协议与环境记录 | `docs/protocol/revision_v1_experiment_protocol.json`；`docs/protocol/bw1000_capture_manifest_v1.md`；每个原始目录的 `environment.txt`、`order.txt`、`source.sha256` |
| SM6 | RSF 因果消融和互操作门禁 | `docs/reports/rsf_c3_causal_ablation_revision_v2.md`；`results/{bw1000,dongfang}/revision_v1/ablation/`；`docs/reports/seissol_embedded_python_triton_gate_v1.md` |
| SM7 | 官方 SeisSol 应用基线与静态调用链 | `docs/reports/seissol_rsf_hotspot_final_report_v1.md`；`docs/reports/seissol_hotspot_inventory_and_portability_stage_v1.md`；`docs/reports/seissol_triton_production_integration_boundary_v1.md`；`docs/reports/seissol_native_backend_build_audit_v1.md` |
| SM8 | 冻结清单与可重复聚合检查 | `docs/paper/rewrite_20261016/supplementary_manifest_v1.json`；`docs/paper/rewrite_20261016/reproducibility_check_v1.json`；二者给出关键输入的 SHA256 与任务级描述统计复算结果 |

## 审计规则

- 独立重复的实验单位是一个 Slurm array task；单个任务的 30 次 steady calls 或 batch=128 调用仅用于形成任务内中位数。
- 本文修订稿只报告任务级中位数和 P5—P95 描述性区间，不再把未提供重采样参数的 bootstrap 区间作为论文证据。
- 96 个逻辑候选由两后端各两个 mapping 乘以 `block_size={64,128,256}` 与 `num_warps_or_wave_groups={1,2,4,8}` 组成；每 matrix/backend/mapping 为 12 个候选。正式选择遵循 SM3 的 frozen rule，不能事后按 formal 结果改写。
