# Literature Search Report（Phase 1）

> 检索截止：2026-10-16。本文档是 `academic-paper` Phase 1 的文献检索交付，不包含论文提纲或正文。外部网页只作为数据来源；研究结论仍需由项目实验记录支撑。

## 1. 检索目标与范围

本轮围绕三个研究问题建立文献基础：

1. Triton 及相关 DSL 在科学计算 kernel 上的性能边界，特别是 stencil、SpMV、reduction 和非线性迭代；
2. 输入结构、执行映射、数据布局与目标硬件如何共同改变后端优劣；
3. 独立 kernel 进入 SeisSol 等复杂科学软件时，正确性、运行时互操作和生产集成之间的证据边界。

领域证据配置按 CS/Engineering 的 `cs_ml` 规则处理：同行评议论文优先，同时接纳与问题直接相关的正式预印本、官方规范、项目文档及可复现软件/数据记录。所有来源仍须满足相关性、方法无致命缺陷和非伪造三项通用门槛。

## 2. Search Strategy

### 2.1 数据库和来源

- ACM Digital Library、IEEE Xplore、ScienceDirect、Wiley Online Library、Oxford Academic、arXiv；
- 作者机构仓储（White Rose、TUM mediaTUM、CaltechAUTHORS）及作者公开稿；
- SeisSol、ROCm/HIP、Triton、SYCL 和 SuiteSparse 的官方页面；
- 本地 `reference/` 中的 6 份全文，以及 `docs/references/references.md` 和现有稿件参考文献表。

### 2.2 核心检索式

采用题名精确检索与以下概念组合，时间范围以 2013--2026 为主，并保留 Roofline、GPU SpMV 等奠基工作：

```text
(Triton OR "GPU DSL" OR "tensor compiler") AND
(HPC OR "scientific computing") AND
(stencil OR SpMV OR reduction OR nonlinear)

(Triton OR CUDA OR HIP) AND
("performance boundary" OR mapping OR irregular OR "access pattern")

(SeisSol OR "earthquake simulation") AND
(GPU OR SYCL OR CUDA OR HIP OR fusion)

("rate-and-state friction" OR RSF) AND
(GPU OR numerical OR implicit OR iteration)

("performance portability" OR Kokkos OR RAJA OR SYCL) AND
(metric OR benchmark OR specialization)
```

排除：只报告模型精度而无 kernel 性能证据的 AI 论文；无实验协议的博客；将不同设备或不同问题规模直接作为后端倍率的材料；不能核验来源身份的二手摘要。

### 2.3 筛选结果

- 本地种子集合：现稿 27 条编号来源，外加 6 份本地全文（有重合）；
- 外部定向检索：核验 20 个题名/DOI/官方页面候选；
- 全文或官方摘要级评估：22 项；
- 建议进入重写稿的核心来源：18 项，其中 14 项为论文/正式预印本，4 项为官方规范、软件或数据文档。

这是面向论文重写的定向邻近工作检索，不应表述为系统综述，也不能据此作“首次”“唯一”之类穷尽性新颖性声明。

## 3. 元数据与引用核验

| 来源 | 核验结果 | 重写时处理 |
|---|---|---|
| Tillet et al., Triton, 2019 | DOI `10.1145/3315508.3329973`；MAPL 2019，10--19 | 保留；用于语言设计原点，不用于支持国产 DCU 性能结论 |
| Williams et al., Roofline, 2009 | DOI `10.1145/1498765.1498785`；CACM 52(4), 65--76 | 保留；明确本文 ceiling 是实测/经验 ceiling 时再用于解释 |
| Pennycook et al., 2016 | arXiv `1611.07409` 和 PMBS 公开稿可核验；检索未确认 Crossref DOI | 不补造 DOI；若正文不计算其指标，只用来约束“性能可移植性”措辞 |
| Edwards et al., Kokkos, 2014 | DOI `10.1016/j.jpdc.2014.07.003`；JPDC 74(12), 3202--3216 | 保留 |
| Beckingsale et al., RAJA, 2019 | DOI `10.1109/P3HPC49587.2019.00012`；71--81 | 保留 |
| Ragan-Kelley et al., Halide, 2013 | DOI `10.1145/2499370.2462176` | 保留 |
| Luporini et al., Devito, 2020 | DOI `10.1145/3374916`；TOMS 46(1), Article 6, 28 pp. | 将“1--28”改为 Article 6/28 pages 更准确 |
| Bell & Garland, GPU SpMV, 2009 | DOI `10.1145/1654059.1654078` | 保留；用于结构与映射背景 |
| Davis & Hu, SuiteSparse, 2011 | DOI `10.1145/2049662.2049663`；TOMS 38(1) | 保留；矩阵网页继续作为数据记录分别引用 |
| Dorozhinskii et al., SeisSol fused GEMMs, 2024 | DOI `10.1002/cpe.8037`；C&CPE 36(12), e8037 | 本地全文和出版信息一致 |
| Chen & Xia, 2026 | DOI `10.1145/3774895.3812198`；ICS Workshops '26, 60--64，CC BY 4.0 | 核心邻近工作；作者全名为 Yixian Chen、Chunwei Xia |
| Li et al., 2026 | arXiv `2607.04454`，当前检索仅确认预印本 | 标为 preprint，不能写成同行评议论文 |
| Ewart & Araya-Polo, 2026 | arXiv `2609.10368`，当前检索仅确认预印本 | 标为 preprint；其多厂商 stencil 是重要邻近工作 |
| Sun & Zhang, 2025 | DOI `10.1093/gji/ggaf085`；GJI 241, 1029--1041 | 补全现有清单缺失 DOI 和末页 |
| Liu et al. | DOI `10.1016/j.mechmat.2025.105540`；Mechanics of Materials 212, 105540；2025 在线发表/2026 卷期存在年份口径差异 | 参考文献按目标格式采用出版社卷期年份，并可注明 available online 2025；避免两种年份混用 |
| Wei & Chen | DOI `10.1093/gji/ggag175`；GJI 246(1), article ggag175 | 本地 PDF 是“submitted to JGR”稿，引用必须以 OUP Version of Record 为准 |

### 需立即修正的风险

1. `reference/05_...pdf` 的抬头是 *manuscript submitted to Journal of Geophysical Research: Solid Earth*，不能把该 PDF 本身当作 GJI 版本的版式或页码证据；科学内容可比对，书目信息用 OUP 页面。
2. Liu et al. 同时出现 2025 在线年与 2026 卷期年。全文引用风格需选定一种规则并统一，不能在正文和参考文献中交替。
3. Pennycook et al. 的本地条目没有可靠 DOI。保留 arXiv/PMBS 公开稿链接，除非后续从 IEEE/Crossref 得到明确记录。
4. 官方文档须写访问日期并固定版本或 commit。`latest` 页面只能支持当前文档状态，不能自动证明实验时的软件行为。

## 4. Annotated Bibliography

### A. Triton、DSL 与性能评测

**Tillet, Kung, & Cox (2019), “Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations.”** 会议论文，DOI `10.1145/3315508.3329973`。提出以 tile 为中心、由编译器承担低层调度的 Triton 编程模型。适合支持 Triton 的设计动机及其最初面向规则张量计算的语境；不能支持本文在 AMD/国产 DCU 上的具体表现。

**Chen & Xia (2026), “From Naive CUDA to Triton: A Systematic Evaluation of AI-Era HPC Operator Development.”** ICS Workshops 论文，DOI `10.1145/3774895.3812198`。以 stencil、SpMV、GEMM 和 N-body 构成规则性×计算强度矩阵，在 RTX 3090 上发现 Triton 对规则计算密集任务较强，对非规则与 all-to-all 模式明显退化。它是本研究最邻近的算子级对照，但只有 NVIDIA 单平台、四个算子和较短篇幅；本研究应明确增加了双 DCU 平台、真实矩阵结构干预、RSF 非线性状态迭代和应用迁移门禁。

**Ewart & Araya-Polo (2026), “Stencil Computation at the Intersection of AI and HPC.”** arXiv:2609.10368。比较 TinyTC 与 PyTorch/Triton 的高阶 stencil，覆盖 Intel、NVIDIA 和 AMD GPU，并用 Roofline/存储层次解释。它削弱了“缺少多厂商 Triton stencil 研究”的泛化表述；本稿的新意应落在同源 Triton/HIP、国产 DCU、边界/非整除规模和跨算子结构规律。

**Li, Rathnasuriya, & Yang (2026), “Correct but Slow: An Empirical Study of the GPU Kernel Evaluation Gap in Modern Domain-Specific Languages.”** arXiv:2607.04454。研究 Triton/TileLang kernel 通过正确性测试却不具备替换质量的问题，提出 library-relative efficiency 与 Roofline utilization 的互补筛查。它直接支持“Correctness Gate 不等于 Performance Gate”，但其对象是 A100/GH200 上的深度学习算子，不能替代本文的科学计算及运行时互操作证据。

**Ragan-Kelley et al. (2013), Halide.** PLDI 论文，DOI `10.1145/2499370.2462176`。通过算法与调度分离说明高层 DSL 仍需目标相关调度。适合作为“映射是性能变量”的概念先例。

**Luporini et al. (2020), Devito.** ACM TOMS 46(1), Article 6，DOI `10.1145/3374916`。展示从符号方程经多级 IR 到优化 stencil 代码的路线及地震成像算例。用于连接科学 DSL、自动优化与地球物理计算，不宜据此宣称 Triton 具备相同的领域语义或优化覆盖。

### B. 性能模型与可移植性

**Williams, Waterman, & Patterson (2009), Roofline.** CACM 52(4), 65--76，DOI `10.1145/1498765.1498785`。提供算术强度、带宽和计算 ceiling 的解释框架。本文需要同时报告实测 ceiling 的来源和模型失配，避免仅凭点位给出单因果结论。

**Pennycook, Sewall, & Lee (2016), “A Metric for Performance Portability.”** PMBS 公开稿/arXiv:1611.07409。定义依赖明确平台集合和效率基准的性能可移植性度量。由于当前两平台软件栈不对齐、原生基准也未必代表各自最佳实现，本稿宜称“双平台结构趋势/实证”，不宜计算或暗示严格性能可移植性。

**Edwards, Trott, & Sunderland (2014), Kokkos.** JPDC 74(12), 3202--3216，DOI `10.1016/j.jpdc.2014.07.003`。指出 CPU/GPU 所需数据布局和访问模式可能冲突，并通过执行空间与布局抽象实现专门化。用于说明可移植源码不消除结构和目标相关优化。

**Beckingsale et al. (2019), RAJA.** P3HPC, 71--81，DOI `10.1109/P3HPC49587.2019.00012`。提供面向大型科学应用的可移植并行抽象案例。它适合作为框架类相关工作；本文并未提出等价的通用编程框架。

### C. 稀疏结构与后端映射

**Bell & Garland (2009), “Implementing Sparse Matrix-Vector Multiplication on Throughput-Oriented Processors.”** SC'09，DOI `10.1145/1654059.1654078`。系统讨论稀疏格式、线程映射和矩阵结构对 GPU SpMV 的影响，是解释短规则行、重尾行长与负载均衡差异的基础来源。本文应将自己的七矩阵结果表述为 Triton/HIP 映射下的新平台证据，而非首次发现 SpMV 结构敏感性。

**Davis & Hu (2011), “The University of Florida Sparse Matrix Collection.”** ACM TOMS 38(1)，DOI `10.1145/2049662.2049663`。定义并记录 SuiteSparse/UF collection 的来源与元数据。用于数据集引用；每张矩阵仍应保留具体 collection 页面和访问日期。

### D. SeisSol GPU 与应用集成

**Dorozhinskii, Gadeschi, & Bader (2024), “Fused GEMMs towards an efficient GPU implementation of the ADER-DG method in SeisSol.”** C&CPE 36(12), e8037，DOI `10.1002/cpe.8037`。通过 ChainForge 融合连续 batched GEMM、减少中间数据移动，并把生成器接入 YATeTo/SeisSol。它证明 SeisSol 中 proxy、代码生成与生产集成可以形成完整证据链，同时提醒本文不能把独立 RSF kernel 结果写成完整应用收益。

**Dorozhinskii & Bader (2023), “Combining CUDA and SYCL for Accelerating Earthquake Simulations in SeisSol.”** PASC 2023 演示文稿，TUM 机构仓储。材料区分 wave propagation 的 CUDA/HIP/SYCL 与 dynamic rupture 的 SYCL 路径，可用于历史架构背景；演示文稿证据等级低于正式论文和当前源码，所有现状描述需与目标 checkout 核验。

**SeisSol official documentation and source repository.** GPU、Dynamic Rupture、build/parameter 文档和实际源码是调用链、支持范围、参数语义、布局与 ABI 的权威运行时资料。引用时应固定版本/commit 和访问日期；文档不能证明某个 kernel 是最大热点，热点必须来自本项目 profiling。

### E. RSF 数值语义与 GPU 地震模拟

**Liu, Bhattacharya, & Lapusta, “Learning a potential formulation for rate-and-state friction.”** Mechanics of Materials 212, 105540，DOI `10.1016/j.mechmat.2025.105540`。指出经验 RSF 的历史依赖和缺乏势/变分形式使隐式求解困难且计算昂贵，并探索神经势函数替代。它支持 RSF 不是普通 elementwise 算子的数值背景；不能证明 SeisSol Aging-RSF kernel 在本文设备上的具体瓶颈。

**Wei & Chen (2026), “Diversity and transition of rupture styles governed by rate-and-state friction.”** GJI 246(1), ggag175，DOI `10.1093/gji/ggag175`。展示 RSF 参数、初始条件与成核方式对破裂形态的影响。适合支撑物理参数必须有依据和真实状态分布的重要性，不直接支持 kernel 性能主张。

**Sun & Zhang (2025), “A new finite-difference method for earthquake cycles accelerated by GPU and multigrid method.”** GJI 241, 1029--1041，DOI `10.1093/gji/ggaf085`。使用 GPU CUDA 与几何多重网格加速二维地震周期模拟，并指出大规模线性系统求解是其主要瓶颈。它说明含 RSF 的应用热点可能位于更高层求解过程，因此反对未经 profiling 将 RSF 局部更新称为全应用主热点。

## 5. Literature Matrix

| 来源 | Triton/DSL | 结构与映射 | 双/多平台 | 应用集成 | RSF/地震 | 证据类型 |
|---|---:|---:|---:|---:|---:|---|
| Tillet et al. 2019 | 主 | 辅 |  |  |  | 语言/编译器论文 |
| Chen & Xia 2026 | 主 | 主 |  |  |  | 4-kernel 实验 |
| Ewart & Araya-Polo 2026 | 主 | 辅 | 主 |  | 辅 | 多 GPU stencil 预印本 |
| Li et al. 2026 | 主 | 辅 | 辅 | 辅 |  | DSL 评测预印本 |
| Halide 2013 | 主 | 主 | 辅 |  |  | DSL 论文 |
| Devito 2020 | 主 | 主 | 辅 | 辅 | 辅 | 科学 DSL/地震成像 |
| Roofline 2009 |  | 辅 | 辅 |  |  | 性能模型 |
| Pennycook et al. 2016 |  |  | 主 |  |  | 度量方法 |
| Kokkos 2014 | 辅 | 主 | 主 | 主 |  | 框架/mini-app |
| RAJA 2019 | 辅 | 辅 | 主 | 主 |  | 框架/应用 |
| Bell & Garland 2009 |  | 主 | 辅 |  |  | GPU SpMV |
| Davis & Hu 2011 |  | 主 |  |  |  | 数据集论文 |
| Dorozhinskii et al. 2024 | 辅 | 主 | 主 | 主 | 主 | SeisSol 正式论文 |
| Dorozhinskii & Bader 2023 |  | 辅 | 主 | 主 | 主 | 公开演示文稿 |
| Liu et al. |  |  |  | 辅 | 主 | RSF 数值方法 |
| Wei & Chen 2026 |  |  |  |  | 主 | RSF 物理模拟 |
| Sun & Zhang 2025 |  | 辅 |  | 主 | 主 | GPU 地震周期模拟 |

## 6. 研究缺口与可防守的新颖性边界

### 6.1 已有工作已经覆盖的内容

- Triton 在规则/不规则 HPC kernel 上表现不同，已有 Chen & Xia 的直接结果；
- 多厂商 Triton stencil 已有 Ewart & Araya-Polo 的 2026 预印本；
- “正确但很慢”的 DSL kernel 评估缺口已有 Li et al. 的直接论述；
- SpMV 对矩阵结构、格式和线程映射敏感是成熟结论；
- SeisSol GPU kernel fusion 和异构后端已存在正式工作与工程路径。

因此，重写稿不能声称首次研究 Triton for HPC、首次发现 irregular kernel 退化、首次提出 correctness/performance 分离、首次发现 SpMV 结构敏感，或首次将 GPU 用于 RSF 地震模拟。

### 6.2 本项目最可防守的组合缺口

在本轮核验到的邻近来源中，尚未发现一项工作同时具备以下组合：

1. 在两个国产/非 NVIDIA DCU 软件栈上，以同一套 FP64 科学 kernel 比较 Triton 与同平台 HIP；
2. 从规则 stencil、reduction、真实矩阵 SpMV 延伸到含超越函数、状态与非线性迭代的 Aging-RSF；
3. 用固定 nnz、行长分布、局部性、program 映射、边界和迭代消融解释“后端胜负翻转”；
4. 把数值正确性、性能、外部指针/stream/布局互操作和完整生产应用替换分成独立门禁，并公开负结果与编译失败。

这是一项**检索范围内的组合缺口**，不是无条件“世界首次”。建议贡献表述为：“本文在统一双平台实验协议下，系统刻画结构、映射与运行时边界共同决定 Triton/HIP 选择的实证规律，并以 SeisSol Aging-RSF 展示从独立 kernel 到应用集成的分级证据。”

### 6.3 尚需补强的文献缺口

- AMD/ROCm Triton 在 HPC、尤其 FP64 special functions 和稀疏计算上的同行评议研究仍少；应继续检索 ROCm/Triton 编译 lowering 与 AMD GPU kernel 研究。
- “国产 DCU”相关论文与设备架构公开资料不足。论文若披露设备型号，应优先加入厂商正式架构/编程指南；若不能披露，需清楚限定可复现性。
- GPU kernel 与大型 MPI/LTS 应用生命周期集成的通用研究尚未充分覆盖；可补充 scientific application portability、embedded Python/C ABI/stream interoperability 的正式工作。
- RSF 原始 Aging law 与 SeisSol 采用语义的物理来源应从 SeisSol 源码引用链继续回溯到 Dieterich/Ruina 及 TPV/SEAS benchmark，而非只引最新应用论文。

## 7. Coverage Distribution Advisory

`DISTRIBUTIONAL_SKEW_ADVISORY`

- Dimension: time distribution
- Concentration: 2019--2026 = 13/17（76%）
- Advisory: 这是由 Triton 的出现时间和当前工具链问题导致的合理集中，但物理与稀疏算法背景不能只依赖近期论文。
- Search response: 已保留 Roofline 2009、GPU SpMV 2009、Halide 2013、Kokkos 2014 和性能可移植性 2016；下一轮补 Dieterich/Ruina 基础来源。

`DISTRIBUTIONAL_SKEW_ADVISORY`

- Dimension: methodological distribution
- Concentration: computational/benchmark studies = 14/17（82%）
- Advisory: 与本文系统实验型研究问题一致，不构成缺陷；但生产集成结论需要源码、官方文档与版本化工程证据补足。
- Search response: 保留 SeisSol 官方文档、源码 commit、SuiteSparse 数据记录和本地实验 manifest 作为非论文证据层。

## 8. 推荐进入重写稿的引用优先级

**必须引用：** Triton 2019；Chen & Xia 2026；Ewart & Araya-Polo 2026；Li et al. 2026；Roofline；Bell & Garland；Davis & Hu；Dorozhinskii et al. 2024；SeisSol 官方文档/源码；Liu et al.；Sun & Zhang。

**用于边界和对照：** Pennycook；Kokkos；RAJA；Halide；Devito；Dorozhinskii & Bader；Wei & Chen。

**引用纪律：** 两篇 2026 arXiv 工作必须标注 preprint；项目自己的性能数值只引本地实验材料，不用外部文献代证；每个“未发现”或“组合缺口”必须附检索截止日和检索范围。

## 9. 已核验入口

- Triton 2019: https://doi.org/10.1145/3315508.3329973
- Chen & Xia 2026: https://doi.org/10.1145/3774895.3812198
- Ewart & Araya-Polo 2026: https://arxiv.org/abs/2609.10368
- Li et al. 2026: https://arxiv.org/abs/2607.04454
- Roofline: https://doi.org/10.1145/1498765.1498785
- Kokkos: https://doi.org/10.1016/j.jpdc.2014.07.003
- RAJA: https://doi.org/10.1109/P3HPC49587.2019.00012
- Halide: https://doi.org/10.1145/2499370.2462176
- Devito: https://doi.org/10.1145/3374916
- GPU SpMV: https://doi.org/10.1145/1654059.1654078
- SuiteSparse collection paper: https://doi.org/10.1145/2049662.2049663
- SeisSol fused GEMMs: https://doi.org/10.1002/cpe.8037
- RSF potential formulation: https://doi.org/10.1016/j.mechmat.2025.105540
- RSF rupture styles: https://doi.org/10.1093/gji/ggag175
- GPU earthquake-cycle method: https://doi.org/10.1093/gji/ggaf085

