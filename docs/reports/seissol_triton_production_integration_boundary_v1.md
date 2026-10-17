# SeisSol 生产接入边界审计（v1）

日期：2026-10-01  
平台：BW1000 Slurm，SeisSol v1.3.2-git-v2 源码

## 结论

官方动态破裂 GPU 路径不是一个可由 Python 调用替换的独立函数，而是嵌入 SeisSol 时间推进、局部时间步和设备队列管理中的 SYCL/C++ 模板实现。当前没有形成“把 Triton Python kernel 直接替换官方 RSF kernel”的生产闭环，因此生产接入状态保持 `NOT_TESTED`，不能把独立 Triton 微基准与官方 SeisSol 端到端时间直接比较。

## 源码证据

1. `GpuImpl/BaseFrictionSolver.h::evaluate` 负责完整设备侧流程：调用 `runtime.syncToSycl(&queue)`，复制 LTS 数据，在同一个 `sycl::queue` 上提交预计算、初始应力、摩擦更新、输出和能量计算 kernel，最后调用 `runtime.syncFromSycl(&queue)`。
2. `GpuImpl/RateAndState.h::updateFrictionAndSlip` 依次执行初始变量计算、状态变量迭代、流体压力、法向应力、滑移率和牵引更新；`updateStateVariableIterative` 在每个状态更新次数内提交多个依赖同一队列的 kernel，并在 kernel 内执行 Newton 反演。
3. `GpuImpl/AgingLaw.h::updateStateVariable` 直接访问 `stateVariableBuffer`、`localSlipRate` 和 `stateVarReference` 等设备指针，并使用 SYCL 数学函数。数据采用 `real (*)[misc::NumPaddedPoints]` 的按面、按填充点布局，而不是独立微基准的扁平数组接口。
4. `Factory.cpp` 为 Aging RSF 同时构造 CPU `friction_law::AgingLaw` 和 GPU `friction_law_gpu::AgingLaw`，生产路径通过工厂和 SeisSol 生命周期选择后端，不存在现成的 Triton C ABI 插槽。

## 接入门禁

若要实施生产接入，至少必须同时解决：

- **指针与布局**：从 SeisSol 的设备分配取得稳定的 USM/device pointer，并将填充面布局映射为 Triton 可接受的 stride/shape；
- **队列与同步**：让 Triton/HIP launcher 使用 SeisSol 当前队列或明确的跨 API event 依赖，不能在每个时间步隐式同步；
- **生命周期**：覆盖 auxiliary memory 分配、LTS layer 切换、析构和异常路径；
- **构建边界**：选择预编译 HSACO/C ABI launcher 或嵌入 Python/JIT。后者会引入解释器、缓存和进程生命周期依赖，不适合作为正式生产路径；前者需要稳定的 Triton 离线编译产物、ABI、目标架构和构建系统改造；
- **正确性**：在官方 TPV101/RSF 时间推进中验证状态变量、牵引、滑移率、输出和 MPI/多队列同步。

## 研究决策

当前最小安全结论是：独立 Triton kernel 已完成语义和硬件行为研究，但生产接入仍被“SYCL 队列/USM 生命周期 + C++ 模板数据布局 + Triton 离线 ABI”三重边界阻断。下一阶段优先完成接入接口设计与静态数据布局原型；在没有稳定 C ABI 和端到端 correctness 之前，不投入高风险的完整 SeisSol 替换，也不发布生产 speedup。论文将该结果作为“迁移边界”证据，而不是把未完成的接入描述为已实现。

