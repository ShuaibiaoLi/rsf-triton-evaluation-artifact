# SeisSol Aging RSF 热点最终判定报告

日期：2026-09-24  
算例：官方 `SeisSol/precomputed-seissol/tpv101`  
研究对象：开源 SeisSol，不依赖课题自有算法库

## 1. 算例与语义

官方 TPV101 配置明确使用：

```text
FL = 3  # Rate-and-State Aging Law
RS_f0 = 0.6
RS_sr0 = 1d-6
RS_b = 0.012
RS_iniSlipRate1 = 1d-12
RS_iniSlipRate2 = 0.0
rs_sl0 = 0.02
```

网格为官方 `tpv5-20k.h5`，约 20k 规模；smoke 版本只将 `EndTime` 从 2.0 s 改为 0.01 s，未修改 RSF 参数、材料、网格和时间步进语义。

## 2. 正确性与可运行性

| 平台 | 配置 | 结果 |
|---|---|---|
| CPU | `build-cpu/SeisSol_Release_dnoarch_6_elastic` | PASS |
| BW1000 | `SeisSol_Release_dgfx936_hip_6_elastic` | PASS |
| CPU FL=0 对照 | 同一网格/步数 | PASS |
| BW1000 FL=0 对照 | 同一网格/步数 | PASS |

两种平台日志均出现 `Initialize Friction Model`、`Starting simulation` 和正常结束；BW1000 的第一次失败是输出目录已存在导致的 XDMF writer 冲突，不是 RSF 或 HIP kernel 失败。换用新输出前缀后运行通过。

## 3. 对照测量

### CPU，OMP_NUM_THREADS=1

| 配置 | Dynamic Rupture 总时间 |
|---|---:|
| FL=3 Aging RSF | 1.24865 s |
| FL=0 | 0.654214 s |
| 差值（RSF 近似贡献） | 0.594436 s |

RSF 近似占 FL=3 Dynamic Rupture 时间：

```text
0.594436 / 1.24865 = 47.6%
```

### BW1000 gfx936 HIP

| 配置 | Dynamic Rupture 总时间 | 总计算 kernel 时间 |
|---|---:|---:|
| FL=3 Aging RSF | 0.292790 s | 0.765462 s |
| FL=0 | 0.213606 s | 0.756627 s |
| 差值（RSF 近似贡献） | 0.079184 s | — |

RSF 近似占 FL=3 Dynamic Rupture 时间：

```text
0.079184 / 0.292790 = 27.0%
```

由于 FL=3 与 FL=0 只改变摩擦律选择，差值可作为 RSF 路径的端到端近似开销；它不是单个函数的精确硬件计时，后续 Triton 论文实验仍应增加 kernel-level instrumentation。

## 4. 最终判定

### 热点判定：GO

Aging RSF 在官方 SeisSol TPV101 真实计算路径中是显著热点：

- CPU 上约占 Dynamic Rupture 时间的 47.6%；
- BW1000 上约占 Dynamic Rupture 时间的 27.0%；
- 不是 Proxy 推断，而是完整 SeisSol 算例的 FL=3/FL=0 对照。

因此，RSF-Triton 具备继续研究和形成独立论文的实证基础。

### 论文范围

推荐冻结为：

> Performance Characterization and Triton Optimization of Aging Rate-and-State Friction in SeisSol Dynamic Rupture Simulations

下一阶段重点优化：

1. Aging state update；
2. friction evaluation 中的 `log/exp/asinh` 等 FP64 超越函数；
3. Newton/slip-rate inversion；
4. 分支和非收敛路径；
5. CPU/BW1000 端到端替换前后的数值一致性与性能收益。

## 5. 限制

- 当前是 0.01 s smoke case，不是完整 2.0 s 生产运行；
- FL=3 与 FL=0 差值是 RSF 路径近似开销，不等价于硬件 profiler 的单 kernel 精确时间；
- 尚未将 Triton kernel 接回 SeisSol；
- 尚未完成完整 2.0 s、多规模和多线程扩展实验。

因此最终结论是：**RSF 热点判定已通过，论文路线 GO；性能优化和 SeisSol 接入进入下一阶段。**

