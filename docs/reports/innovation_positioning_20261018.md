# 创新点与下一轮实验方向复核（2026-10-18）

## 判断

当前论文更适合作为**可审计的对照评测与机制边界研究**，不宜包装成新 SpMV 算法、新 Triton 编译优化器或 SeisSol 生产后端。最有辨识度的实测结果是：在同一卡、同一输入、预登记候选空间内，`orani678` 从简单映射比较的 Triton/HIP < 1 变为两端 best-vs-best > 1；七层计时阻止把调用路径差异写成纯设备内核差异；RSF 可逆干预区分了实际迭代轨迹、函数路径与静态循环上界，前两者不足以单独解释差距，后者在九个已知案例上使差距收窄。这些是**受测平台栈与注册合同内**的观察，不是全硬件或全矩阵结论。

## 近邻核对与修正

| 来源 | 已有覆盖 | 与本文的可辩护差异 | 处理 |
|---|---|---|---|
| Xing 等，2026，DOI `10.1007/s11227-026-08634-y`，出版社摘要 | 海光 DCU 上 CSR SpMV 的 wavefront 映射、访存和长短行调度 | 本文不是更快的 CSR 方法，而是在 Triton/HIP 两端控制候选预算、正确性、冻结选择和计时边界后检查排序 | 加入主稿 1、2.2、6.4 节与参考文献；不得声称 DCU SpMV 尚无先例 |
| Guo 等，2024，DOI `10.1007/s11227-024-06234-2`，出版社开放获取版 | DCU 上的结构感知 SpGEMM 优化 | 算子与研究问题不同；只作国产平台稀疏优化背景 | 加入主稿，明确 SpGEMM 不等于 SpMV |
| Chen 与 Xia，2026，DOI `10.1145/3774895.3812198`；Ewart 与 Araya-Polo，2026，arXiv:2609.10368 | Triton/HPC 算子和多厂商 stencil 比较 | 已经覆盖“高层 DSL 对 HPC 的性能依赖于结构”的大命题；本文只能主张具体公平合同和反转证据 | 保留现稿限制，不宣称 Triton/HPC 评测先例缺失 |
| Li 等，2026，arXiv:2607.04454；Ramos 与 Lowery，2026，arXiv:2605.03208 | correctness-performance gap；HIP/Triton 内核隔离与复现 | 本文应用分级门禁与可逆 RSF 消融不同；两篇均不能替代真实 SeisSol 生产运行 | 将 Kerncap 列为预印本近邻，不把隔离内核等同应用集成 |

检索范围：2026-10-18 使用英文关键词 `Triton HIP scientific computing SpMV AMD`、`GPU rate state friction SeisSol` 和中文关键词 `国产 DCU Triton HIP 稀疏矩阵` 等，核对出版社、Crossref、arXiv 原始页面。CNKI/万方未取得可审计的完整覆盖；“没有检索到完全同构论文”不是新颖性的证明。Xing 文全文需要机构访问，此处仅依据出版社摘要比较其公开陈述，不推定其全部实验细节。

## 已完成实验：静态循环上界，而非重复排序布局

进一步审计发现前期已有两项相关 BW1000 负结果：作业 `874107` 仅改变 `max_iterations` 的均匀、交错和分组布局，Triton 延迟差约 1.4%；作业 `874220` 用九个真实输入案例构造同质、分块、交错布局，真实迭代次数最大 22，三种布局延迟差约 1.3%。原始 JSON 在 `BW1000_test/generic_microbench/evidence/874220/result.json`，解释见 `docs/reports/rsf_newton_real_case_mix_v1.md`。它们**没有**缩短 fixed replay 的循环上限，也没有 HIP 配对，不能直接否定新假设；但已说明“再做一次排序”不是高优先级。此处纠正本报告初版把分组排在首位的建议。

现有 BW1000 fixed replay 为 HIP 57.12 μs、Triton 224.08 μs；lookup 全部超越函数后为 HIP 34.08 μs、Triton 192.16 μs，且 `SQ_INSTS_VALU` 比仍约 2.3。Triton 的 `fixed_replay_kernel` 在 60 轮每轮都形成 `friction`、`derivative`、`residual` 与 `candidate`，然后以 `tl.where` 选择；HIP 同语义实现则在 `k >= actual` 时 `continue`。这两份源代码提示无效轮次的实际设备指令可能不同，但源码结构不能证明编译后执行了哪些指令。

更小的单因素干预是保持九个登记案例、输入顺序、线程映射和全部数学表达式不变，**同时将 HIP 与 Triton fixed replay 的循环上限从 60 改为 22**。九例的实际迭代次数为 `8,22,6,4,4,6,6,1,6`；22 只对该冻结输入集合充足，不能作为通用 RSF 上限。独立 CPU FP64 检查 `scripts/validate_rsf_loop_bound_v3.py` 已在 `N=1,9,65,257,4099,65537` 上确认 60 与 22 的五字段输出一致，并确认错误的 8 轮上限会在 22 次案例上产生不同数值。该检查是本地语义门禁，不是 BW1000 GPU correctness 或性能证据。

预先划定的解释边界是：比值变化本身不足以区分循环 trip count、谓词化、编译器专门化和寄存器压力；且已知九例的 22 轮 oracle 不能推广到未知输入。不得把现有 `874220` 的单次布局测试误当五作业统计或 HIP/Triton 配对证据。

执行门禁：

1. **Inspect / Plan**：冻结 v2 fixed replay 源码和旧结果哈希，新增 v3 文件而不覆盖历史实验；记录 input TSV、九例实际次数、两个后端的编译参数及 60/22 两个变体。
2. **Local Check**：运行 `scripts/validate_rsf_loop_bound_v3.py`；使用独立标量 reference 检查五字段，错误的 8 轮设置作为负对照。WSL 不产生 BW1000 性能结论。
3. **Deploy Dry-run / Apply / Build / Smoke**：按工作区远程规则单向部署；BW1000 先构建并小规模 smoke，设备侧 correctness 覆盖 `N=1,9,65,257,4099,65537` 与不收敛案例。任何失败保留原始日志。
4. **Profiling / Fetch Evidence**：通过门禁后在独立 Slurm 作业中交替 HIP-first/Triton-first，至少五个作业单位；无 profiler 的 device-event 报告任务级中位数与描述性 P5–P95。PMC/ISA 单独采集，不把 profiler 时间用于 speedup。
5. **判定**：每端 original pre→22→original restore 的恢复值必须满足预登记 ±10%；失败即不作性能机制结论。所有环境、源码哈希、输入、原始样本和失败记录回传 WSL。

## 实测结果与下一步判断

BW1000 smoke `935988` 在六种规模上通过 HIP/Triton 五字段 correctness；首次 `935980` 因 `N=1` 微秒级恢复波动未过 ±10%，失败日志保留。正式数组作业 `936008_[0-4]` 的五个独立任务全部通过恢复门槛。静态上界 60→22 后，HIP 与 Triton 的任务级延迟比中位数分别为 0.895 和 0.709；Triton/HIP 中位比值 3.714→2.900，五个任务方向一致。独立 PMC 作业 `936030` 显示 `SQ_INSTS_VALU`：HIP 3,812,013→3,695,163，Triton 8,610,000→3,351,750。详细任务表、源码哈希和失败记录见 `docs/reports/rsf_c3_loop_bound_revision_v3.md`。

因此，固定 60 轮静态上界是受测 fixed replay 多余指令量与相对延迟的一部分，而非唯一根因；v2 “固定实际迭代仍未缩小差距”并不与之矛盾，因为 v2 未改变编译期上界。下一轮优先比较 22 轮变体的编译产物，并用受控控制流改写检验谓词/分支解释，而非再做一次输入排序。22 轮仍有约 2.9 倍相对差距，且是 oracle，不可写成普适优化。v3 不属于旧 `v1.0.1`，随 `v1.0.2` 纳入公开证据包；完整 SeisSol Triton 生产替换仍为 `NOT_TESTED`。
