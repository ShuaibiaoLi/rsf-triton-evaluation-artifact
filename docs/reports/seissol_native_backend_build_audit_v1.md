# BW1000 SeisSol 官方 HIP 后端构建与 smoke 审计

日期：2026-09-27（中国标准时间）  
源码：SeisSol v1.3.2-git-v2，revision `20334d02af9c539a9cf7cc4cd03c3e2622104aea`  
目标：在 BW1000/DTK 26.04 上构建官方 SeisSol HIP/AdaptiveCpp 路径，并完成官方 TPV101 最小 smoke。

## 构建结果

构建在节点 `f11r1n17`、分区 `hx1hdnormal`、单卡 `dcu:1` 上通过。关键作业如下：

| 作业 | 内容 | 结果 |
|---:|---|---|
| 866153 | 首次 `gfx936` 兼容补丁构建 | 失败；随后暴露 NetCDF include 参数缺失 |
| 866158 | 补充 NetCDF include | 失败；AdaptiveCpp 目标缺少 `mpi.h` |
| 866173 | 补充 MPI include/link 参数 | PASS，退出码 0 |
| 866200 | 带 Lua 的 EASI 依赖 | PASS，退出码 0 |
| 866201 | 链接 Lua EASI 后重构建 SeisSol | PASS，退出码 0 |
| 866192 | 无 Lua EASI 的 TPV101 smoke | 失败；`!FunctionMap` 解析错误 |
| 866206 | Lua EASI 的 TPV101 smoke | PASS，退出码 0 |

构建生成的可执行文件为：

`SeisSol_Release_dgfx936_hip_6_elastic`

构建日志显示最终目标 `SeisSol-bin`、`libSeisSol_offloading...so` 和 device library 均已链接。原始 stdout/stderr 位于远端 `seissol-rsf-hotspot/evidence/`，已回传至本目录 `BW1000_test/seissol-source/evidence/866173/` 和 `866206/`。

## gfx936 适配边界

上游 GemmForge 0.0.207 的架构描述表没有 `gfx936`。本研究使用
`BW1000_test/seissol-source/gfx936-gemmforge-compat-v1.patch` 增加最小架构映射，以允许 SeisSol 代码生成继续执行。该映射复用 `gfx90a` 的代码生成参数，仅是兼容描述，不声称 BW1000 硬件等同于 gfx90a，也不应称为未经修改的上游构建。

因此本文后续应使用以下准确表述：

> 基于官方 SeisSol v1.3.2 HIP/AdaptiveCpp 后端，叠加可审计的 BW1000 `gfx936` GemmForge 兼容补丁。

## Smoke 证据

作业 `866206` 的设备记录：

- 主机：`f11r1n17`
- 卡系列：`BW`
- rocm-smi Unique ID：`TBH49801050301`
- SeisSol：`DEVICE_BACKEND = hip`、`DEVICE_ARCH = gfx936`
- 运行时：`GPU: true`、`Device API: HIP`、设备名 `BW`
- 网格：19,183 cells、3,454 vertices
- 算例：官方预计算 TPV101，2 s 模拟
- `computeDynamicRupture` 总时间：7.18198 s
- 总模拟时间：30.9352 s
- 计算结束：`Simulation done`、`SeisSol done. Goodbye.`
- smoke 状态：`PASS`

可执行文件 SHA256：
`2d2669115aa77d3801d600f960afd5ae6ec3d21da007d0b4073ed3f8b100c415`

## 性能结论边界

此结果证明官方 SeisSol HIP 后端已能在 BW1000 上构建并运行官方 TPV101，完成了 Build Gate 和应用 smoke gate；它尚不提供 Triton 与官方 SeisSol 后端的最终加速比。当前 Triton kernel 仍是独立 C3/C4 探针，尚未替换 SeisSol 生产执行路径，因此把 TPV101 的 30.9352 s 与独立 Triton C3 延迟直接相除是不成立的。

下一项工作应在相同算例/输入和相同计时边界下，先建立 SeisSol 官方 RSF 动力破裂 kernel 的可比 native 微基准，或完成 Triton C ABI/设备 kernel 接入；在此之前，最终 speedup 保持 `PENDING`。
