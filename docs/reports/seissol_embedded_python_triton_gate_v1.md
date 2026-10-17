# SeisSol 嵌入 Python/Triton 接入门禁（v1）

## 实验结果

在 BW1000 Slurm、DTK 26.04、`gfx936` 上完成了三个逐级探针：

| 作业 | 内容 | 结果 |
|---|---|---|
| 875389 | C++ 嵌入 Python，传入 HIP 设备指针，Triton raw pointer copy/add，回读验证 | PASS，`bad=0` |
| 875406 | 在 C++ 创建的非默认 HIP stream 上，通过 `torch.cuda.ExternalStream` 启动 Triton kernel | PASS，`stream_interop=PASS` |
| 875426 | C++ 嵌入 Python，12 个 FP64 RSF 输入/输出设备指针，Triton RSF 表达式 kernel | PASS，`bad=0` |
| 875437 | 3 个 fault face × 16 个 padded points，20 次预热后稳定调用 | PASS，`bad=0`，steady mean 98.899 us |

## 解决的问题

该门禁证明不需要从 HSACO 外部重建 DTK/HCU launcher ABI；SeisSol 风格的 C++ 主程序可以保留设备分配和 stream，将原始设备指针传入嵌入式 Python，复用 Triton HCU 已验证的 launcher。非默认 stream 也可以通过 `torch.cuda.ExternalStream` 复用，避免隐式切换到另一个队列。

## 接入边界

该结果已覆盖 SeisSol 风格的 face/padded layout 和稳定调用开销，但仍不是完整 SeisSol TPV101 生产替换。完整替换仍需把 `LTSRateAndState` 的真实 auxiliary memory、状态变量迭代、MPI/LTS 生命周期接入同一模块，并完成端到端 correctness。当前生产接入状态更新为 `LAYOUT_BRIDGE_PASS / FULL_APPLICATION_PENDING`。

## 源码对齐审计

SeisSol 的 `misc::NumPaddedPoints` 在 `DynamicRupture/Misc.h` 中由 `leadDim<init::QInterpolated>()` 编译期确定，并非固定常数；官方 GPU `BaseFrictionSolver::evaluate` 使用 SYCL queue，通过 `runtime.syncToSycl`/`syncFromSycl` 管理队列边界。当前 bridge 已验证 HIP 指针和 HIP stream，但还没有从 SeisSol 的 SYCL queue 获得底层 HIP stream，也没有取得真实 `Layer::var(..., AllocationPlace::Device)` 的所有字段。因此真实生产 hook 仍需 SeisSol 源码级改造，不能由独立 bridge 结果替代。

## 可复现实验材料

- `BW1000_test/seissol-source/embedded_pointer_probe_v1.py`
- `BW1000_test/seissol-source/embed_python_triton_pointer_v1.cpp`
- `BW1000_test/seissol-source/embedded_rsf_pointer_probe_v1.py`
- `BW1000_test/seissol-source/embed_python_rsf_pointer_v1.cpp`
- 对应 `run_embed_*_v1.slurm` 和远端 `evidence/embed-{triton,rsf}-<jobid>.*`
