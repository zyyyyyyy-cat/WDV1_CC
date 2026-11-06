# 2025–2026 跨模态行人重识别可迁移机制调研

> **目标**：为 UAD / WHU-MARS（RGB–NIR–TIR × ground–aerial 任意场景 ReID）寻找 2025–2026 年顶会顶刊、有开源代码、能直接提升检索指标的可迁移机制。
>
> **调研日期**：2026-10-29  
> **检索范围**：CVPR / ICCV / ECCV / NeurIPS / ICLR / AAAI / TPAMI / TIP / TMM / TCSVT / Pattern Recognition / ACM MM 2025–2026；辅以 2024 年但机制仍未被 UAD 采用的基础工作。  
> **检索方式**：WebSearch + arXiv/OpenReview/CVF/出版社页面全文抓取 + GitHub API 逐仓库核验（是否只有 README、是否有 train/test、最近更新时间）。WebFetch 在当前环境被域名校验拦截，改用 curl 实际抓取页面并解析，链接状态以 2026-10-29 的 HTTP/API 返回为准。  
> **证据分级**：
> - **A｜已读全文**：抓取并阅读了 arXiv HTML / CVF 全文中的方法、损失、实验表。
> - **B｜仅摘要/README**：只核实了摘要、论文页、官方 README 或部分表格。
> - **C｜仅检索结果**：仅从搜索结果/二手笔记获得机制描述，未打开原文。
>
> **重要约束**：本文所有数值均来自论文或官方 README 的原始表述；无法核实的数值明确写“未提取”。预期收益均为**机制推论**，不是已验证的 UAD 实验结果。

---

## 0. 一页结论

1. **UAD 最值得补的不是“再加一个对齐损失”，而是三个结构性缺口**：① 分类器/BN 对三模态完全共享，TIR 分类边界被 RGB/NIR 主导；② ProCA 的“每身份每模态一个均值中心”把 ground/aerial、昼夜、相机混成一个点，且 K=4 的 batch 中心噪声大；③ GPD 原型是零初始化 + 固定 EMA + 硬标签 CE，缺少置信度/软目标/每模态结构。
2. **2025–2026 年最直接可用的机制集中在四类**：模态解耦（MDReID、WSL）、模态路由参数高效微调（ReID5o）、相机/平台感知的原型校准与可靠记忆（CLNS、PoSeD、SPRED）、直接排序优化与查询感知重排（SupRank、BIT）。空-地 ReID 的 ViSA / HiHR / SeCap 提供了“平台/视角条件化”和“局部图融合”的证据，但大多绑定 CLIP 文本或较重结构，迁移优先级低于前四类。
3. **最便宜、最先做的两个实验**：测试时按 camera ID 做特征中心化/缩放（MayI、DART3）+ 每模态独立分类头（WSL、MDReID）。两者都不改变 UAD 的四项损失主体，可单独 A/B，且直接对应 TIR 与 ground/aerial 两个已定位瓶颈。
4. **对 mAP 最直接的是 SupRank 的 listwise AP 代理损失**：UAD 的 WRT 用未归一化欧氏距离、只做 batch 内 pairwise，而评测是 L2 归一化后的全 gallery 排序；SupRank 把“batch AP 与全局 AP 的分解差距”显式补偿，和 UAD 的 mAP 目标一致。代价是需要 memory queue 或较大 batch。
5. **不要优先迁移**：CLIP 文本提示整套框架（HiHR、SeCap、MP-ReID/Uni-Prompt）需要文本编码器与两阶段训练，UAD 的 ImageNet ViT-B 没有对应接口；隐私/联邦（PPA、L2RW）与 UAD 主指标无关；生成式热红外增强有身份幻觉风险，除非有身份保持约束。

---

## 1. UAD 的现状与可插入点（阅读代码后的定位）

以下事实来自 `D:\CODE\Doing\WHU_base\WDV1_CC\WHU\01_UAD_WHU_Analysis.md` 与本地代码，作为后文“插入位置”的依据。

| 组件 | 当前实现 | 结构性限制 | 可插入位置 |
|---|---|---|---|
| Backbone | 共享 ViT-B/16，`torch.cat(list(x), dim=0)` 沿 batch 拼接三模态 | 无模态专属参数、无模态 token、无平台条件 | `model/backbones/vit_pytorch.py`、`model/make_model.py::build_transformer` |
| 分类 | 共享 `Linear(768, C)` + 共享 `BNNeck` | TIR 分类边界被 RGB/NIR 样本主导 | `make_model.py` 初始化/forward；`loss/make_loss.py` |
| WRT | BN 前未归一化特征上的加权三元组 + SoftMargin | 训练几何（模长敏感）与推理（L2 方向）不一致 | `loss/wrt_loss.py`、`loss/make_loss.py` |
| ProCA | 每身份每模态 K=4 特征均值 → 模态中心 → 身份中心（stop-grad） | 单中心混合平台/视角；弱模态等权；中心是 batch 统计量 | `processor/processor.py::PCA` |
| GPD | 训练身份原型 bank，零初始化，硬 CE，温度 0.03，EMA 新中心权重 0.2 | 无置信度、无软目标、无每模态/每平台原型、bank 不保存 | `processor/processor.py::PrototypeBank`、`GPD` |
| 推理 | BN 前 768D CLS，L2 归一化，平方欧氏排序，无 rerank | 无 TTA、无 camera 校准、无 query expansion、无 mINP | `processor/processor.py::do_inference`、`utils/metrics.py` |

**论文结果给出的优先级**（来自 UAD 论文 Table 6/8）：TIR→TIR 20.98 Rank-1 / 14.00 mAP，远低于 RGB→RGB 40.47/26.83；TIR 相关跨模态多在 8–9 Rank-1；GPD 单独 +2.0 mAP/+4.8 Rank-1，ProCA 单独仅 +0.4/+0.6 mAP。因此本调研把“原型/记忆可靠性”和“TIR 表征”排在“再加一个中心对齐项”之前。

---

## 2. 论文卡片

### 2.1 任意场景 / 多光谱（RGB–NIR–TIR）

#### P01｜MDReID: Modality-Decoupled Learning for Any-to-Any Multi-Modal Object Re-Identification
- **会议/年份**：NeurIPS 2025（Spotlight）
- **论文**：https://arxiv.org/abs/2510.23301
- **代码**：https://github.com/stone96123/MDReID （已核验：含 `train_net.py`、`test_net.py`、configs、CLIP ViT-B-16 权重说明）
- **证据等级**：A｜已读全文

**核心机制**：在 ViT 的 patch 序列前为每个模态同时 prepend 两个可学习 token——`I_sp^M`（模态特定）和 `I_sh^M`（模态共享）——而不是只用一个 CLS；三个模态得到 6 个槽位，推理时按“当前可用的模态”构造定长向量 `v_full` 和二值 mask。训练时用 ROL（Representation Orthogonality Loss）把 6×6 相似度矩阵拉向预设目标：特定–特定同模态为 1、共享–共享跨模态全为 1、特定–共享为 0；再用 KDL 约束共享/特定分支的知识差异。分类端使用 **每模态独立分类器**，而不是一个共享分类器。监督对象始终是训练身份 PID，不引入额外标注。

**关键增益**：RGBNT201（RGB/NIR/TIR 行人）82.1 mAP / 85.2 Rank-1，比 TOP-ReID 的 72.3/76.6 高 +9.8 mAP/+8.6 R1；RGBNT100 85.3/95.6；MSVR310 51.0/68.9。消融显示 MDL（共享/特定解耦）平均 +11.5 mAP/+11.1 R1；**每模态分类器相对单分类器平均 +13.4 mAP/+13.7 R1**。

**对 UAD 的可迁移点**：这是与 WHU-MARS 最同构的工作（三光谱、any-to-any）。最值得先拿走的不是完整 6 槽结构，而是“每模态分类器 + 共享/特定分离”的思想：UAD 的 TIR 分类头可以不再与 RGB/NIR 争抢共享权重。完整 token 解耦可作为第二阶段。
**预期收益/风险**：每模态分类器预期 +1–3 mAP，TIR→TIR 更明显；完整解耦预期 +2–5 mAP，但输出维度从 768 变 1536/4608，推理接口、ProCA 中心、GPD 原型都要改。CLIP 预训练与 UAD 的 ImageNet ViT 不同，不能直接复制权重。

#### P02｜ReID5o: Achieving Omni Multi-modal Person Re-identification in a Single Model
- **会议/年份**：NeurIPS 2025
- **论文**：https://arxiv.org/abs/2506.09385
- **代码**：https://github.com/Zplusdragon/ReID5o_ORBench （已核验：含 `ReID5o/` 代码目录）
- **证据等级**：A｜已读全文

**核心机制**：五种模态各有独立 tokenizer，统一送入一个 CLIP 多模态编码器；在编码器每个线性层 `W` 旁并行插入**模态特定的低秩矩阵** `ΔW^mod = B^mod A^mod`，由 tokenizer 给出的二值模态控制信号 `c^mod` 决定是否激活，即 `y = Wx + c^mod B^mod A^mod x`。这样共享 `W` 学模态不变特征，低秩专家学模态特定残差。多模态查询再经过 feature mixture（MSA + 1 层 Transformer + MLP）融合任意模态组合。损失为 SDM + identity classification。

**关键增益**：ORBench 单模态检索 mAP 58.09（CLIP 基线 48.73）；四模态查询 mAP 86.35（CLIP 79.15）。多专家路由在 r=4 时只增加 2.36M 参数 / 3.64G FLOPs，却比无路由的共享 tokenizer 提升 +5.29 mAP；r=16/32 反而下降。直接用 5 个独立编码器（UE）参数巨大且更差。

**对 UAD 的可迁移点**：把 ViT-B 中每个 `nn.Linear` 替换为“共享基座 + 3 个 rank-4 模态 LoRA”，用 UAD 已有的模态标签（batch 布局：前 B 是 RGB、中 B 是 NIR、后 B 是 TIR）路由。这是**在不改变 CE/WRT/ProCA/GPD 定义的前提下增加模态容量**的最小改动。
**预期收益/风险**：预期 +1–3 mAP，TIR 收益最大；新增约 2–3M 参数，训练成本可控。风险是 UAD 只有 500/1000 个训练身份，LoRA 可能过拟合 TIR；建议 rank=4、只加在 q/k/v 与 MLP 的第二个线性层，并对 LoRA 用更低 LR。

#### P03｜Multi-modal Multi-platform Person Re-Identification: Benchmark and Method（MP-ReID / Uni-Prompt ReID）
- **会议/年份**：ICCV 2025
- **论文**：https://arxiv.org/abs/2503.17096
- **代码/数据**：https://mp-reid.github.io/ （已核验：项目页提供论文与数据集入口；**方法代码未找到开源**）
- **证据等级**：A｜已读全文（方法/表格）；代码仅项目页

**核心机制**：基于 CLIP 的可学习文本提示被拆成三段：Specific-ReID prompt（身份相关）、Modality-aware prompt（模态）、Platform-aware prompt（UAV/ground）。一个轻量 visual-enhanced network 把图像特征映射成 context 偏移 `σ`，加到三段 prompt 上，使提示随输入图像自适应。训练分两阶段：第一阶段冻结模态/平台 prompt，用 CLIP-ReID 的 i2t/t2i 损失学 ReID prompt；第二阶段冻结 ReID prompt，用模态对比损失和平台对比损失分别学模态/平台 prompt。监督是身份标签 + 模态/平台伪标签（由数据源给出）。

**关键增益**：在 MP-ReID 的 12 个设置上，完整 Uni-Prompt 平均 Rank-1 64.73 / mAP 58.45，Base 为 55.51/47.98；最难的“跨模态 + 跨平台”类别从 28.40/23.08 提升到 43.16/40.26。平台 prompt 是该类别提升的主要来源；模态 prompt 在纯跨模态类别 +6.23 R1/+7.09 mAP。

**对 UAD 的可迁移点**：UAD 的场景定义是 modality × platform，但代码没有显式平台条件。**平台感知 prompt 的思想可以降级实现为“平台 token/平台 embedding/平台特定 BN”**，不需要 CLIP 文本编码器：把 ground/aerial 作为可学习 embedding 加到 CLS 或每层 Adapter 的偏置上。
**预期收益/风险**：若只加平台 token，预期 +0.5–1.5 mAP；若加平台特定分类器/原型，收益更大但参数增加。风险是 UAD 的 camera ID 与平台标签并非完全等价，需要先用相机高度/文件名建立 platform 映射；MP-ReID 方法代码不可用，不能直接复现。

#### P04｜Weakly Supervised Visible-Infrared Person Re-Identification via Heterogeneous Expert Collaborative Consistency Learning
- **会议/年份**：ICCV 2025
- **论文**：https://arxiv.org/abs/2507.12942
- **代码**：https://github.com/KongLingqi2333/WSL-VIReID （已核验：含 `WSL_ReID/` 训练代码）
- **证据等级**：A｜已读全文

**核心机制**：为可见光和红外各建一个身份分类器 `W_v`、`W_r`，作为“异构专家”；编码器前几层模态不共享、后几层共享，专家用 CE + WRT 在**各自模态内**训练。然后让每个专家去预测另一模态所有样本的身份，用 Count Priority Selection 得到跨模态身份对应矩阵 `M^{v→r}`；对应关系再反过来监督编码器和共享分类器的协同一致性学习。整个方法只使用单模态身份标签，不需要成对跨模态标签。

**关键增益**：SYSU-MM01 All-Search 70.4 Rank-1 / 66.6 mAP / 52.7 mINP；LLCM V→IR 55.3/58.7、IR→V 47.3/53.3。基线仅模态内训练时 SYSU 为 47.8/47.2，说明“每模态专家 + 跨模态关系”本身贡献巨大。

**对 UAD 的可迁移点**：UAD 有完整三模态标签，不需要弱监督对应估计；但**每模态分类器 + 模态内 WRT** 可以直接采用。UAD 的 WRT 已经在全 batch 上做，可改为“模态内 WRT + 全局 WRT”两项，避免 TIR 正样本被 RGB 正样本淹没。
**预期收益/风险**：每模态分类器 + 模态内 WRT 预期 +1–2 mAP；风险是 TIR 类内样本少，模态内 WRT 的正样本池更小，需要保持 K=4 或增大 TIR 采样。

---

### 2.2 相机/平台感知的原型、记忆与可靠性

#### P05｜CLNS: Camera-aware Label Noise Suppression for Unsupervised Visible-Infrared Person Re-Identification
- **会议/年份**：Pattern Recognition 2026（在线发表；ScienceDirect）
- **论文**：https://www.sciencedirect.com/science/article/abs/pii/S0031320326008381
- **代码**：https://github.com/AeroVILab-AHU/CLNS （已核验：含 `train_1.py`、`test_1.py`、`cam_associate_loss.py`、预训练权重链接）
- **证据等级**：B｜已读摘要 + README + 结果表；未逐页读全文

**核心机制**：粗到细的无监督 VI-ReID 噪声抑制流水线。Camera-aware Prototype Calibration（CPC）利用**跨相机一致性**修正聚类结构错误，构造可靠原型；Optimal Transport Prototype Matching（OTPM）在全局做跨模态原型匹配；实例级 Neighbor-guided Camera-domain Learning（NCL）用邻域软标签稠密化特征分布；Noise-aware Memory Updating（NMU）根据样本/原型可靠性更新记忆，避免错误累积。整个方法不使用任何真实标签。

**关键增益**：SYSU-MM01 All-Search 69.49 Rank-1 / 65.64 mAP / 51.2 mINP；RegDB V→IR 94.79/91.83、IR→V 95.33/91.79；LLCM V→IR 54.6/57.8、IR→V 46.9/52.6。README 报告在三个数据集上均超过当时 SOTA。

**对 UAD 的可迁移点**：UAD 有 camera ID，且 ProCA 只按模态聚合、不按相机/平台聚合。可把 ProCA 的“每身份每模态中心”升级为“每身份每模态每相机（或每平台）中心”，再用跨相机一致性做校准，最后 OT 匹配到身份全局原型。NMU 的可靠性权重可直接用于 `PrototypeBank.update`，让 TIR/航拍噪声样本少改原型。
**预期收益/风险**：预期 +1–2 mAP，尤其是 A→G/G→A 和 TIR 跨相机；风险是每相机样本更少（K 可能只有 1–2），中心估计方差大，需要向全局中心收缩（shrinkage）而不是硬分组。OT 在 3 模态 × 相机组上的成本需要控制（可用 Sinkhorn 迭代 20–30 次）。

#### P06｜PoSeD: Prototype Matters — Modality-unified Prototype Self-distillation for Unsupervised Visible-infrared Person Re-identification
- **会议/年份**：ACM Multimedia 2026
- **论文**：https://arxiv.org/abs/2609.11514
- **代码**：https://github.com/Terminator8758/PoSeD （已核验：仓库存在，但**仅有 README，代码未释放**，README 写 “Code will be available soon”）
- **证据等级**：A｜已读全文；代码不可用

**核心机制**：把可见光和红外原型拼成一个**统一原型记忆** `P`，用跨模态聚类得到的统一伪标签做多正样本对比：一个样本对所有同伪标签原型（同模态 + 跨模态）都是正样本，而不是只对比跨模态原型。第二个分支把“原型到原型的相似度”作为教师，蒸馏到“实例到原型”的在线相似度，并用实例自适应融合系数抑制聚类噪声。整个框架在一个对比损失里同时优化模态内和模态间关系。

**关键增益**：无监督 SYSU-MM01 66.0 Rank-1 / 62.6 mAP；RegDB 95.3/89.6。与 BMIL 框架结合后 SYSU 67.5/64.7、RegDB 94.7/89.3。论文强调统一原型对比比“只做跨模态对比”更稳，原型自蒸馏能纠正 OT 硬匹配噪声。

**对 UAD 的可迁移点**：UAD 的 GPD 正是“实例–原型”硬 CE，可改为：① 原型 bank 作为教师，输出软分布；② 对同一身份的所有模态/平台原型做多正样本对比；③ 用置信度（与原型相似度 margin）加权蒸馏。这不需要 PoSeD 的代码，机制可以在 `PrototypeBank` 与 `GPD` 内重写。
**预期收益/风险**：预期 +0.5–1.5 mAP；GPD 是 UAD 主要增益来源，升级它的性价比高于新增 ProCA 项。风险是教师=学生同源，容易确认偏差；需要 EMA 教师或历史原型快照，并保留硬 CE 作为下限。

#### P07｜SPRED: Self-Reinforcing Prototype Evolution with Dual-Knowledge Cooperation for Semi-Supervised Lifelong Person Re-Identification
- **会议/年份**：ICCV 2025
- **论文**：https://arxiv.org/abs/2507.01884
- **代码**：https://github.com/zhoujiahuan1991/ICCV2025-SPRED （已核验：含 `continual_train.py`、`train_transfer.py`、configs、数据处理）
- **证据等级**：A｜已读全文

**核心机制**：使用**可学习身份原型**建模当前阶段身份分布；对无标签样本取最近的两个原型 `p_A`、`p_B`，用二者分类得分 `s_A = exp(<f,p_A>)/(exp(<f,p_A>)+exp(<f,p_B>))` 判断置信度，模糊样本丢弃，生成高质量伪标签。另一半是 Dual-Knowledge Cooperation：用当前模型和上一阶段历史模型分别聚类，取聚类一致性高的伪标签，抑制噪声。原型结构还用于知识蒸馏，缓解遗忘。

**关键增益**：在 12 个数据集的 Semi-LReID 基准上，相对 LSTKC：50% 标签率 Seen-Avg mAP/R1 +5.7/+4.2，20% 时 +12.2/+12.1，10% 时 +15.5/+17.3。全监督 100% 标签率下也与 DKP 竞争，SPRED† 在 Seen/UnSeen 上继续领先。

**对 UAD 的可迁移点**：UAD 是单阶段全监督，不需要 lifelong；可迁移的是“**top-2 原型 margin 作为置信度**”和“**历史原型快照做一致性过滤**”。把 GPD 的每个样本损失按 margin 加权，低置信度样本（尤其 TIR 早期）降权；bank 保留历史副本做双知识一致性。
**预期收益/风险**：预期 +0.3–1.0 mAP；实现简单，几乎不增加计算。风险是早期原型不可靠，margin 过滤会进一步减少 TIR 有效样本；建议 warm-up 后启用，并保留最低权重 0.2。

#### P08｜Instruct-ReID++: Towards Universal Purpose Instruction-Guided Person Re-identification
- **会议/年份**：arXiv 2405.17790，2025 年修订（未核实正式期刊）
- **论文**：https://arxiv.org/abs/2405.17790
- **代码**：https://github.com/hwz-zju/Instruct-ReID （已核验：仓库 198 stars，含训练/评测）
- **证据等级**：A｜已读全文

**核心机制**：统一“指令 + 图像”的 ReID 框架；IRM 用 adaptive triplet loss 同时约束身份和指令对齐；IRM++ 改为只用图像编码器提 gallery 特征，并引入**两个 memory bank**（身份级、指令级）做对比学习，每次迭代提供多个负样本，而不是 triplet 的单个负样本。任务特定/任务无关两种评测都在 OmniReID++ 上训练。

**关键增益**：任务无关评测下，LLCM 可见–红外 +4.2 mAP，CUHK-PEDES +2.5 mAP；任务特定评测下 LLCM +4.3 mAP；共 10 个测试集。论文明确报告 memory bank 对比学习优于 triplet。

**对 UAD 的可迁移点**：UAD 的 WRT 是 batch 内 pairwise，负样本数受 batch 限制；GPD 虽全局但只在原型层。可以在 WRT 旁加一个**特征 memory queue**（每身份/每模态保存若干归一化特征），让 WRT 或 Sup-AP 看到跨 batch 的难负样本。对 TIR 尤其重要，因为 TIR 样本少、batch 内负样本多样性不足。
**预期收益/风险**：预期 +0.5–1.5 mAP；风险是 queue 中的旧特征与当前 backbone 不同步，需要动量更新或较短 queue（如 1–2 个 epoch）。

---

### 2.3 空-地 / 视角感知 ReID

#### P09｜ViSA: View-Aware Semantic Alignment for Aerial-Ground Person Re-Identification
- **会议/年份**：CVPR 2026（Poster）
- **论文**：https://arxiv.org/abs/2605.18192
- **代码**：https://github.com/Cat-Zero/ViSA （已核验：含 `fastreid/`、`tools/train_net.py`、CARGO config）
- **证据等级**：A｜已读全文

**核心机制**：不再强行把 ground/aerial 压成同一表示，而是显式分离 view-invariant 与 view-specific。ETGM 用一组 MoE 专家 token，通过 cross-attention/self-attention/FFN 生成 `Q_inv`、`Q_spe` 两类语义查询；DLFM 用 top-k 稀疏图 + GCN 在 patch 上做局部区域对齐，并由查询引导聚合。损失是每样本 CE + triplet；没有额外的对抗或重建项。

**关键增益**：CARGO ALL 70.51 Rank-1 / 65.46 mAP，比此前最好方法 +1.92 R1/+5.27 mAP；A↔G 71.28/69.00。消融：只加 DLFM 就把 ALL 从 61.54/53.54 提到 67.31/62.86；去掉 ETGM 后 A↔G Rank-1 从 71.28 降到 68.09。

**对 UAD 的可迁移点**：UAD 的 ground/aerial 差异与 CARGO 类似，而且 UAD 只取 CLS、丢弃 128 个 patch token。**DLFM 的 top-k 图 + GCN 局部融合**可以直接加在 ViT 最后几层 patch token 上，提升航拍小目标与遮挡；ETGM 的 view-aware 查询可作为平台条件化的轻量版本。
**预期收益/风险**：局部图融合预期 +0.5–1.5 mAP，尤其 A→G；风险是 UAD 训练集较小（500/1000 ID），MoE 专家容易过拟合；建议先只加 DLFM，冻结专家数量为 1，再做 view token。

#### P10｜HiHR: Hierarchical Hyperbolic Representation for Aerial-Ground Person Re-Identification
- **会议/年份**：ECCV 2026
- **论文**：https://arxiv.org/abs/2607.09186
- **代码**：https://github.com/YangQiWei3/HiHR （已核验：含 `fastreid/`、`train.sh`、configs）
- **证据等级**：A｜已读全文

**核心机制**：从预训练视觉–文本编码器提多粒度特征；TMF 用文本特征作 query、多层 CLS 作 key/value 做 cross-attention，并用同一 attention 聚合多层 patch token，得到 view-agnostic 与 view-aware 两个融合特征。HHL 把二者映射到 Lorentz 双曲空间，构造“粗层（身份可分 + 跨视角一致）/细层（视角特定判别）”的父子结构，用 entailment cone 正则约束层级。推理时把两个双曲特征对数映射回欧氏空间、缩放后拼接。

**关键增益**：CARGO ALL 67.52 mAP / 74.36 R1，baseline（CLIP ViT-B/16 + CE + triplet）63.61/68.23；A→G 70.53/75.53，比最强基线 +1.53 mAP/+4.25 R1。AG-ReID v1 A→G 79.21/86.06。

**对 UAD 的可迁移点**：可迁移的是“**粗层身份 + 细层视角**”的双层监督，而不是双曲空间本身。可在 UAD 中把 ProCA 的身份中心当“粗层”、把平台/相机中心当“细层”，用父子包含关系约束（平台中心应靠近身份中心但保留偏移），无需文本编码器。
**预期收益/风险**：完整 HiHR 迁移成本高（CLIP 文本、双曲运算、双特征拼接），预期收益不足以抵消工程量；仅“粗/细双层原型”值得借鉴。风险是双曲曲率/数值稳定性，UAD 小数据上难调。

#### P11｜SeCap: Self-Calibrating and Adaptive Prompts for Cross-view Person Re-Identification in Aerial-Ground Networks
- **会议/年份**：CVPR 2025（Highlight）
- **论文**：https://arxiv.org/abs/2503.06965
- **代码**：https://github.com/wangshining681/SeCap-AGPReID （已核验：含 `fastreid/`、`train.sh`、`test.sh`、LAGPeR 数据处理）
- **证据等级**：A｜已读全文

**核心机制**：编码–解码结构。编码器 VDT 在每层把 CLS 解耦成 `Cls - View` 的 view-invariant 部分和 view-specific 部分；解码器 PRM 用 view-invariant 特征通过 cross-attention 重新校准一组可学习 prompt（`P_re = FFN(SA(CA(Prompt, X_inv))) + Prompt`）；LFRM 用校准后的 prompt 解码局部特征。最终输出 `[X_inv, X_local]`。

**关键增益**：LAGPeR A→G 41.79 R1/30.37 mAP（VDT 40.15/28.97），G→A 35.26/33.42（VDT 33.55/31.98），G→A+G 24.39/19.24；G2APS-ReID A→G 75.31/58.57（VDT 73.05/56.23）。在 AG-ReID.v1 A→G 也取得最好结果。值得注意的是 CLIP-ReID 在该任务上反而大幅下降（A→G 24.40/17.60），说明空-地任务不能直接套 CLIP-ReID。

**对 UAD 的可迁移点**：把“视角解耦 + prompt 重校准”降级为**平台条件化 Adapter**：用 ground/aerial 标签生成一组 prompt，cross-attend 到当前 CLS，再经 FFN 产生通道偏置，加到 ViT 中间层。LFRM 的局部特征解码可复用 ViSA 的图融合。
**预期收益/风险**：平台 prompt 预期 +0.5–1.0 mAP；完整 VDT 改造风险较高，且 SeCap 的 baseline 是 ViT（非 CLIP），与 UAD 更接近，但仍需重训。UAD 的 camera ID 与平台标签需要先对齐。

#### P12｜BIT: Matching-based Bi-directional Interaction Transformation Network for Visible-Infrared Person Re-Identification
- **会议/年份**：CVPR 2026
- **论文**：https://arxiv.org/abs/2603.14243
- **代码**：https://github.com/Xuan266/BIT （已核验：仓库存在，但**目前只有 README，没有训练/评测代码**）
- **证据等级**：A｜已读全文；代码不可用

**核心机制**：共享 ViT-B/16 编码 patch 特征后，BCI 在一个 batch 内构造所有 visible–IR 图像对（B² 对），做双向 cross-attention（visible→IR、IR→visible）并堆叠 T=3 个 BCI block，每层用 Aggregation Contrastive Loss 保证身份一致。QA Scoring 在推理时计算双向 patch 相似度矩阵、Top-k 过滤、互为 top-k 的 patch 对应，再用置信度感知 MLP 聚合为最终相似度 Ψ，替代全局 CLS 点积。

**关键增益**：SYSU-MM01 All-Search 80.53 Rank-1 / 79.76 mAP（比 DiVE 79.07/74.96 明显高）；LLCM V→I 67.9/69.8、I→V 59.0/65.3；RegDB 平均也报告 SOTA（具体数值未提取）。不依赖 rerank 即达到该水平。

**对 UAD 的可迁移点**：**QA Scoring 可以只做推理期重排**：保留 UAD 全局特征做初排，取 top-20 候选，再用 ViT 的 patch token 做 query-aware 双向 top-k 匹配，融合分数。UAD 的 batch 本身是三模态 tuple，也可在训练时构造 RGB–TIR、NIR–TIR 对做 BCI。
**预期收益/风险**：推理期 patch 重排预期 +1–3 mAP，航拍小目标收益最大；代价是 N_q × top-k 的 patch 矩阵，需限制候选数。训练期 BCI 风险高（B² 对、三模态 6 个方向），且代码未释放，不建议作为第一优先。

---

### 2.4 度量学习、排序与重排

#### P13｜Optimization of Rank Losses for Image Retrieval（SupRank / ROADMAP）
- **会议/年份**：IEEE TPAMI 2025
- **论文**：https://arxiv.org/abs/2309.08250
- **代码**：https://github.com/elias-ramzi/SupRank （已核验：官方 PyTorch 实现，含 SOP/iNat/CUB 配置）
- **证据等级**：A｜已读全文

**核心机制**：针对 rank loss 的两个病态：不可微与不可分解。SupRank 用一个可微代理“上界”替代离散 rank 算子，对 outlier 更鲁棒；ROADMAP 增加一个 decomposability gap 损失 `L_DG*`，显式补偿“batch 内 AP 估计”与“全训练集 AP”之间的差距。方法同时适用于 AP、R@k，以及层级 AP/NDCG；训练时用全 batch 的排序列表，而不是 pairwise 三元组。

**关键增益**：SOP 81.9 R1 / 55.7 mAP@R，Smooth-AP 为 80.9/54.3；iNaturalist 71.8 R1 / 29.5 mAP@R，Smooth-AP 为 67.3/26.5。与 XBM 相比，iNat 上 +12.5 R1/+11 mAP。batch 越小，`L_DG*` 的相对收益越大（Fig. 6）。

**对 UAD 的可迁移点**：UAD 的 WRT 是未归一化欧氏距离上的加权三元组，和推理的 L2 余弦排序不一致；Sup-AP 可以直接作用在**归一化后的 BN 前特征**上，优化 mAP 本身。UAD 的 mAP 只有 11.0，排序质量比 Rank-1 更差，listwise 损失正好针对这一点。
**预期收益/风险**：预期 +0.5–2 mAP；风险是 UAD 每卡 batch 只有 64 个 tuple（192 张图），全局 AP 估计噪声大，需要 memory queue 或 `L_DG*`；与 GPD 的全局 CE 有部分功能重叠，建议先替换 WRT 而不是叠加，权重从 0.5 开始 warm-up。

#### P14｜Cheb-GR: Rethinking K-nearest Neighbor Search in Re-ranking for Person Re-identification
- **会议/年份**：CVPR 2025
- **论文**：https://openaccess.thecvf.com/content/CVPR2025/html/Yang_Cheb-GR_Rethinking_K-nearest_Neighbor_Search_in_Re-ranking_for_Person_Re-identification_CVPR_2025_paper.html
- **代码**：**未找到开源**（CVF 页面与 GitHub 搜索均无官方仓库）
- **证据等级**：C｜仅检索结果/摘要；方法细节未逐页核对

**核心机制**：用 Chebyshev 定理自适应选择邻居，替代 k-reciprocal re-ranking 中昂贵的 kNN 搜索；在选出的邻居子图上做图卷积， refine 图像特征后再算距离。论文给出基于 Chebyshev 不等式的理论分析，强调在保持排序质量的同时显著降低计算量。它是纯后处理，不改训练。

**关键增益**：Market-1501 上 TransReID + Cheb-GR mAP 93.2，QE 91.9、LBR 90.8、GCR 92.5，接近 KR/ECN 的 94.0；运行 0.9 s，AGW+GCR 为 43.3 s，约为 k-reciprocal 的 1/50。MSMT17 未在该文报告。

**对 UAD 的可迁移点**：UAD 默认 `RE_RANKING=False`，而跨模态 gallery 的邻居图很容易被模态/相机主导；直接套 k-reciprocal 可能伤害 TIR。Cheb-GR 的自适应邻居选择值得作为**相机感知重排**的骨架：邻居只在同模态/同平台内选，再做跨模态融合。
**预期收益/风险**：若邻居选择正确，预期 +1–3 mAP；风险是无代码、跨模态邻居不可靠、同相机过滤后图更稀疏。建议先用 CA-Jaccard（有 CVPR 2024 开源实现）做 sanity check，再决定是否复现 Cheb-GR。

#### P15｜DM-Adapter: Domain-Aware Mixture-of-Adapters for Text-Based Person Retrieval
- **会议/年份**：AAAI 2025
- **论文**：https://arxiv.org/abs/2503.04144
- **代码**：https://github.com/Liu-Yating/DM-Adapter （已核验：含 `train.py`、`test.py`、model/processor）
- **证据等级**：B｜已读摘要/README + 结果检索；未逐页读全文

**核心机制**：冻结 CLIP 主干，在 Transformer 中插入稀疏 MoE Adapter（默认 6 个专家、top-2 路由、reduction=8），由一个 Domain-Aware Router 根据域特征给专家加权，而不是只按 token 路由。只训练约 16M 参数（对比 IRRA 195M），损失为 SDM + auxiliary。它证明了“域/模态感知的 Adapter 路由”比单一 Adapter 更能吸收域差异。

**关键增益**：CUHK-PEDES Rank-1 72.17（比 CSKT +2.47 R1，mAP +2.16）；ICFG-PEDES 62.64 R1 / 36.50 mAP；RSTPReid 60.00/47.37。参数效率显著优于全量微调。

**对 UAD 的可迁移点**：如果 UAD 后续要冻结 backbone（例如为了多数据集/持续学习），DM-Adapter 的“域感知路由”可直接替换 ReID5o 的“模态硬路由”：用 modality + platform 的联合 embedding 作为 router 输入，选择 top-2 专家。若继续全量微调，Adapter 的收益主要是正则化/防过拟合，而非参数效率。
**预期收益/风险**：全量微调下预期 +0–1 mAP；冻结 backbone 下可能 +1–2 mAP 但绝对指标会下降。风险是 UAD 数据量小，6 专家 top-2 容易路由崩溃；建议 3 专家 top-1。

---

### 2.5 测试时自适应、相机偏差与后处理

#### P16｜DART³: Leveraging Distance for Test Time Adaptation in Person Re-Identification
- **会议/年份**：arXiv 2505.18337（2025，预印本；未核实正式接收）
- **论文**：https://arxiv.org/abs/2505.18337
- **代码**：**未找到开源**（GitHub 搜索无官方仓库）
- **证据等级**：A｜已读全文

**核心机制**：针对 ReID 的 camera bias，在测试时学习两组外部 scale-and-shift 参数：query 侧可学习，gallery 侧初始化后冻结，防止退化为平凡解；同时适应源模型的 BN 参数。适应目标不是分类熵，而是**最近邻距离**：用 query 到 gallery 的 top-k 距离构造 softmax 分布，最小化其熵，因为最近邻距离与预测错误相关。无需源数据、无需改架构。

**关键增益**：MSMT17（5 相机训练）TransReID 从 34.4 mAP/41.3 R1 提升到 35.5/43.8；LITE 版 36.2/44.4。CLIP-ReID 从 42.1/50.0 提升到 46.0/55.3；Duke-MTMC 66.7/77.8→69.0/79.6。TENT（分类熵）在该任务上经常下降，说明 ReID 不能直接套分类 TTA。

**对 UAD 的可迁移点**：UAD 的评测是 transductive（query+gallery 同时可见），且 camera ID 可用。可以在 `do_inference` 后、`metrics.eval_func` 前，对每个 camera 的特征做可学习的中心/尺度校准，用 top-k 最近邻熵作为无监督目标；LITE 版只有 10k–50k 参数，适合 UAD 的小 gallery。
**预期收益/风险**：预期 +1–3 mAP，尤其是 ground/aerial 跨相机；风险是相机样本少时统计不稳、可能在已见相机上过拟合。建议只校准 gallery 侧统计，query 侧用固定的全局统计，或对每相机做 shrinkage。

#### P17｜Exploring the Camera Bias of Person Re-identification
- **会议/年份**：ICLR 2025（Spotlight）
- **论文**：https://arxiv.org/abs/2502.10195
- **代码**：**未找到开源**（搜索结果无官方仓库）
- **证据等级**：A｜已读全文

**核心机制**：系统分析 ReID 模型在 unseen domain 上的 camera bias：同相机样本的余弦距离被系统性拉近，导致跨相机检索失败。作者重新审视**特征归一化**（per-camera centering / scaling）作为去偏手段，解释其为何有效，并证明对低层图像属性、身体角度等细粒度偏差也适用。无监督学习部分则提出丢弃“单相机簇”等简单训练策略，减少 camera-biased pseudo label。

**关键增益**：MSMT17 unseen 相机上，camera-specific centering/scaling 把 mAP 从 27.2 提到 34.7（Fig. 4 读数）；无监督 CC 加上训练策略后 mAP +19.3。论文强调这是 test-time post-processing，适用于多种模型与 benchmark。

**对 UAD 的可迁移点**：这是**最便宜**的迁移：UAD 已有 camera ID，且评测排除同相机 gallery。可在 `utils/metrics.py` 里对 query/gallery 特征按 camera 做中心化/标准化，再算余弦距离。它不改变训练，也不依赖代码。
**预期收益/风险**：预期 +1–3 mAP；风险是 UAD 每个 camera 的样本数差异大（航拍/地面、昼夜），小样本相机统计不可靠，需限制最小样本数或向全局均值收缩。

#### P18｜Heterogeneous Test-Time Training for Multi-Modal Person Re-identification（HTT）
- **会议/年份**：AAAI 2024
- **论文**：https://ojs.aaai.org/index.php/AAAI/article/view/28398
- **代码**：https://github.com/ziwang1121/HTT （已核验：含 `test_time_train.py`、`loss/multi_modal_id_margin_loss.py`、RGBNT 配置）
- **证据等级**：B｜已读摘要 + README + 结果检索；未逐页读全文

**核心机制**：训练期除了 CE/triplet/3M，还提出 Cross-identity Inter-modal（CIM）损失，约束**不同身份**的跨模态特征距离；测试期做 Multi-modal Test-time Training（MTT），利用异构模态之间的关系构造自监督任务，在推理前用无标签测试数据微调网络。它把“多模态关系”同时用于训练监督和测试自适应。

**关键增益**：RGBNT201 从 69.0 mAP/70.0 R1（无 MTT）提升到 71.1/73.4（+2.1 mAP/+3.4 R1）；另在 Market1501-MM、RGBN300、RGBNT100 上验证。

**对 UAD 的可迁移点**：UAD 的测试集也是 fixed gallery，可在推理前用 HTT 的 MTT 思路做**少量 epoch 的无标签自适应**：用 RGB/NIR/TIR 三模态之间的循环一致性/互为最近邻作为自监督，微调 BN 或最后几层。CIM 损失也可加入训练，约束“不同身份的跨模态样本”不要过近。
**预期收益/风险**：预期 +1–2 mAP；风险是测试时微调可能过拟合 gallery 的相机/身份分布，且 UAD 的 gallery 很大（9 万张），需要限制步数/样本。建议只在固定评测协议下做，不用于在线部署。

#### P19｜SVIP: Learning Source-free Domain Adaptation for Visible-Infrared Person Re-Identification
- **会议/年份**：NeurIPS 2025
- **论文**：https://openreview.net/forum?id=I7y7MgsEgO （OpenReview 页面有浏览器校验，未抓取全文）
- **代码**：https://github.com/LYXRhythm/SVIP （已核验：含 `train_regdb.py`、`test_regdb.py`、`make_corrupt.py`、预训练模型说明）
- **证据等级**：B｜仅摘要 + README；具体数值未提取

**核心机制**：源数据不可用时的 VI-ReID 域适应。SGCL 用**冻结源模型的特征空间**作为参考，在目标域无标签数据上做对比学习，保留源模型的判别结构；RTL 学习一个残差映射，把目标特征表示为“源特征 + 残差”，从而在适应目标域的同时不破坏源知识；SCCA 对可见/红外做互为结构约束，选出可靠跨模态对，完成无源标签的模态对齐。

**关键增益**：摘要声称在标准 VI-ReID 基准上显著优于现有 source-free 方法；具体 SYSU-MM01/RegDB/LLCM 数值未在可访问材料中提取，需查 OpenReview 全文/补充材料。

**对 UAD 的可迁移点**：如果目标是把 WHU-MARS 模型部署到新校园/新相机（无源数据、无标签），SVIP 的“残差映射 + 冻结源参考”比直接微调更安全。对当前 benchmark 提分不是首选。
**预期收益/风险**：跨域部署预期 +2–5 mAP；风险是仍需要目标域无标签数据与源模型，且 ResNet-50 实现与 UAD 的 ViT 不同，需要改 backbone。不作为第一阶段实验。

---

### 2.6 热红外 / 低光增强与频域方法

#### P20｜Spectral Enhancement and Pseudo-Anchor Guidance for Infrared-Visible Person Re-Identification（SEPG-Net）
- **会议/年份**：ICASSP 2025
- **论文**：https://arxiv.org/abs/2412.19111
- **代码**：https://github.com/1024AILab/ReID-SEPG （已核验：含 `code/` 目录）
- **证据等级**：A｜已读全文

**核心机制**：输入级“光谱增强”：把 RGB 图像转灰度 `Y_grey`，再取 RGB 的 FFT 相位分量 `X_pha` 做 IFFT，二者相加得到 `Y_seg`，复制成 3 通道作为“伪红外”输入；与原红外图像一起送入前 3 层共享权重的双流 ResNet-50。损失端提出 PABA：为每个模态学习伪锚点，用同身份异模态特征拉近、异身份推远，做细粒度双向聚合。它不使用 GAN，避免生成式身份幻觉。

**关键增益**：SYSU-MM01 All-Search baseline 69.6 R1/66.8 mAP；+SE 后 72.3/69.4；+SE+CC 73.8/69.7；+SE+CC+PABA 76.4/71.3。即光谱增强 +2.7 R1/+2.6 mAP，PABA 相对 cross-center 再 +2.6/+1.6。

**对 UAD 的可迁移点**：这是**最适合 UAD 输入管线**的增强：只在 RGB 分支生成伪 TIR 视图（灰度 + 相位），不改变 NIR/TIR 样本，不引入生成模型。PABA 的思想可并入 ProCA：伪锚点替代/补充模态中心，做细粒度对齐。
**预期收益/风险**：仅加伪 TIR 视图预期 +0.5–1.5 mAP（TIR 相关方向更明显）；风险是伪 TIR 与真实 TIR 的传感器噪声/尺度不同，可能引入新的域偏移；建议作为辅助视图（与原图共享标签）而非替换原图，权重 0.3–0.5。

#### P21｜Discovering Multi-Frequency Embedding for Visible-Infrared Person Re-identification（MFENet）
- **会议/年份**：IEEE TCSVT 2025
- **论文**：https://ieeexplore.ieee.org/document/11175191
- **代码**：https://github.com/GuHY777/MFENet-VIReID （已核验：含 `main.py`、`model/`、`losses/`、evaluation）
- **证据等级**：B｜仅摘要 + README；SYSU-MM01 数值未提取

**核心机制**：在频域做多频分解。HiLo-FM 用频域滤波提低频、用轻量多尺度卷积提高频细节，再 attention 融合；FADE 对多频分量加权、用多分支学习多样化特征。损失端提出 Cross-Modality Soft Retrieval（CMSR），优先跨模态一致性而非模态内一致性；另有频域多样性约束。

**关键增益**：LLCM IR→VIS 61.06 Rank-1 / 67.75 mAP（检索结果口径，未从论文原表逐项核对）；SYSU-MM01 与 RegDB 的具体数值未提取。属于 2025 年 VI-ReID 的强 baseline 之一。

**对 UAD 的可迁移点**：可把 HiLo-FM 作为 ViT patch embedding 的并行分支（低频捕获轮廓、高频捕获纹理），或仅把 FADE 的多频加权作为特征增强。CMSR 可替代 ProCA 的等权中心对齐，让跨模态一致性优先。
**预期收益/风险**：预期 +0.5–1.5 mAP；风险是 ResNet 双流结构与 ViT 不匹配，需重新设计；频域分支会增加显存。建议先实现 CMSR 损失，再加频域分支。

#### P22｜MFEN: Multi-Frequency Expert Network for Visible-Infrared Person Re-ID
- **会议/年份**：CVPR 2026（Highlight）
- **论文**：https://arxiv.org/abs/2606.12051
- **代码**：**未找到开源**（论文无 GitHub 链接，GitHub 搜索无官方仓库）
- **证据等级**：A｜已读全文；代码不可用

**核心机制**：为不同频段构造多个频率调制专家，每个专家处理一个频带；一个 gating 函数按样本动态融合专家输出，使每个样本自适应选择最相关的频域线索。训练端提出 Random Frequency Augmentation（RFA）：在两模态图像的振幅谱之间交换低频分量，模拟另一模态的光照/颜色，同时保留相位（结构）；Frequency Auxiliary Optimization（FAO）在空间域和频域同时加损失，互为镜像约束。

**关键增益**：SYSU-MM01 80.93 Rank-1 / 76.56 mAP（比 DSSF³ +1.81 R1/+1.29 mAP）；RegDB 平均 94.48/90.16（+3.30 R1/+4.48 mAP）；LLCM V→I 67.9/69.8、I→V 59.0/65.3。

**对 UAD 的可迁移点**：**RFA 是零架构改动的数据增强**：对 RGB/TIR 图像的振幅谱交换低频，保留相位，可显著缓解光照/颜色差异且不破坏结构。即使不实现频率专家，也建议先在 `datasets/make_dataloader.py` 加 RFA 做消融。FAO 可作为辅助损失加在 ViT 特征上。
**预期收益/风险**：RFA 预期 +0.5–1.5 mAP（TIR/夜间更明显）；风险是低频交换可能改变人体轮廓/背景，需限制交换比例（如 10–30% 低频）并做身份保持检查。代码未释放，需自行实现 FFT 增强。

---

### 2.7 训练策略（权重平均 / EMA / LLRD / 增强）

#### P23｜Model Stock: All We Need Is Just a Few Fine-Tuned Models
- **会议/年份**：ECCV 2024（Oral）
- **论文**：https://arxiv.org/abs/2403.19522
- **代码**：https://github.com/naver-ai/model-stock （已核验：含 notebook、评估脚本、权重）
- **证据等级**：B｜已读方法段 + README；未逐表核对数值

**核心机制**：把预训练权重 `w_0` 当作锚点，只用两个不同随机种子/超参微调的权重 `w_1,w_2`，在每层做加权平均：权重由 `w_1,w_2` 之间的夹角决定，夹角大时更靠近预训练权重，夹角小时更靠近两个微调模型的中心。它近似“微调权重空间的中心”，不需要训练大量模型，也不增加推理成本。

**关键增益**：论文报告在 ImageNet 及 5 个 distribution-shift benchmark 上超过 Model Soup，且只需 2 个微调模型（具体数值未提取）。属于通用视觉分类证据，不是 ReID 检索证据。

**对 UAD 的可迁移点**：UAD 训练 120 epoch，最后阶段可以保存 2–3 个不同种子/不同模态采样顺序的 checkpoint，做 layer-wise Model Stock 或简单 uniform soup；也可对 EMA 权重与最终权重做插值。**这是零推理成本的稳健性提升**，适合在确定网络结构后做。
**预期收益/风险**：预期 +0.1–0.5 mAP（ReID 上未验证）；风险是分类头/BN 统计需要对齐，且不同初始化模型不能直接平均；UAD 的 prototype bank 不随权重保存，soup 后需要重新初始化/重算。**mixup/cutmix 不建议用于 ReID**：混合身份会破坏 CE/WRT/ProCA 的身份语义；跨模态更适合 RFA/低频交换这类“保持身份标签”的增强。

---

## 3. 横向对比表

| 编号 | 论文 | 场景 | 机制关键词 | 代码 | 证据 | 对 UAD 直接度 |
|---|---|---|---|---|---|---|
| P01 | MDReID | RGB/NIR/TIR | 共享/特定 token、每模态分类器、正交损失 | 可用 | A | ★★★★★ |
| P02 | ReID5o | 任意多模态 | 模态路由 LoRA 专家、特征混合 | 可用 | A | ★★★★☆ |
| P03 | MP-ReID/Uni-Prompt | 多模态+多平台 | 模态/平台文本 prompt、视觉增强 | 未开源 | A | ★★★☆☆ |
| P04 | WSL-VIReID | VI-ReID | 每模态专家分类器、跨模态关系 | 可用 | A | ★★★★☆ |
| P05 | CLNS | US-VI-ReID | 相机感知原型校准、OT、可靠记忆 | 可用 | B | ★★★★★ |
| P06 | PoSeD | US-VI-ReID | 统一原型记忆、原型自蒸馏 | 仅 README | A | ★★★★☆ |
| P07 | SPRED | 半监督终身 | 可学习原型、top-2 margin、双知识 | 可用 | A | ★★★☆☆ |
| P08 | Instruct-ReID++ | 通用 ReID | memory bank 对比、adaptive triplet | 可用 | A | ★★★☆☆ |
| P09 | ViSA | 空-地 | view-aware MoE、局部图融合 | 可用 | A | ★★★★☆ |
| P10 | HiHR | 空-地 | 多粒度文本融合、双曲层级 | 可用 | A | ★★☆☆☆ |
| P11 | SeCap | 空-地 | 视角解耦、prompt 重校准 | 可用 | A | ★★★☆☆ |
| P12 | BIT | VI-ReID | B² 双向交互、query-aware patch 打分 | 仅 README | A | ★★★★☆ |
| P13 | SupRank | 通用检索 | listwise AP 代理、分解差距损失 | 可用 | A | ★★★★☆ |
| P14 | Cheb-GR | 通用 ReID | Chebyshev 邻居选择、图重排 | 未开源 | C | ★★☆☆☆ |
| P15 | DM-Adapter | 文本 ReID | 域感知 MoE Adapter | 可用 | B | ★★☆☆☆ |
| P16 | DART³ | 通用 ReID TTA | 相机 scale/shift、距离熵 | 未开源 | A | ★★★★☆ |
| P17 | MayI | 通用 ReID 偏差 | 相机特征归一化、USL 策略 | 未开源 | A | ★★★★★ |
| P18 | HTT | 多模态 ReID TTT | 多模态测试时训练、CIM | 可用 | B | ★★★☆☆ |
| P19 | SVIP | VI-ReID SFDA | 源自由残差迁移、结构一致性 | 可用 | B | ★★☆☆☆ |
| P20 | SEPG-Net | VI-ReID | 灰度+相位伪红外、PABA | 可用 | A | ★★★★☆ |
| P21 | MFENet | VI-ReID | 高低频分解、CMSR | 可用 | B | ★★★☆☆ |
| P22 | MFEN | VI-ReID | 多频专家、RFA、FAO | 未开源 | A | ★★★★☆ |
| P23 | Model Stock | 通用分类 | 两层权重平均、预训练锚点 | 可用 | B | ★★☆☆☆ |

---

## 4. 最值得迁移的 8 个机制（按预期价值排序）

排序综合四个维度：**对 WHU-MARS 已定位瓶颈的直接性**、**论文证据强度**、**与 UAD 现有四项损失的兼容性**、**实现成本/风险**。收益均为预期，需实验验证。

### 第 1｜相机/平台感知的原型校准 + 可靠性加权记忆
- **来源**：CLNS（P05）为主，PoSeD（P06）、SPRED（P07）补充软目标与置信度。
- **插入位置**：
  - `processor/processor.py::PCA`：把“每身份每模态一个中心”改为“每身份每模态每相机（或每平台）中心 + 向全局中心 shrinkage”；用跨相机一致性（同一身份在不同相机下的中心应可互查）剔除错误中心。
  - `PrototypeBank.update`：把固定 EMA 系数 0.2 改为按样本/中心置信度加权；增加历史原型副本（用于 SPRED 式一致性过滤）。
  - `GPD`：保留硬 CE，同时增加“原型→原型相似度”软蒸馏（PoSeD），温度从 0.03 起调，避免过尖。
- **需要的监督**：PID（已有）+ camera ID（已解析）+ 可选的 ground/aerial 平台标签（可由 camera ID/文件名映射）。不需要新标注。
- **与 UAD 损失的交互**：
  - CE：不变；更干净的原型会让 GPD 的负类 logits 更可靠。
  - WRT：不变；但 WRT 的 batch 内正样本仍会跨相机，和新的相机中心约束方向一致。
  - ProCA：从“单中心”变成“多中心 + 一致性校准”，直接提升 ProCA 中心质量；注意 ProCA 权重仅 0.01，新增项不要抢梯度。
  - GPD：这是 GPD 的内部升级，不新增分支；需要防止“原型自蒸馏”与硬 CE 双重计数，建议软损失权重 ≤0.3。
- **预期收益**：+0.5–2.0 mAP，TIR 与 A→G/G→A 更明显。
- **风险**：每相机只有 K=1–4 个样本时中心方差大；OT/校准可能把噪声当结构。建议先做“每模态每平台”（2 组）而不是每相机（>10 组）。

### 第 2｜每模态分类器 + 模态特定 BN/BNNeck
- **来源**：MDReID（P01）、WSL-VIReID（P04）。
- **插入位置**：
  - `model/make_model.py`：`self.classifier` 改为 `nn.ModuleList([Linear(768,C) for _ in range(3)])`；`BNNeck` 改为 3 个 `BatchNorm1d(768)`（或至少 3 个分类器、共享 BN）。
  - forward 中按 `[RGB;NIR;TIR]` 的 batch 布局给每个子 batch 选对应头；返回的 logits 仍拼成 `[3B,C]`，`make_loss.py` 的 target 重复逻辑不变。
- **需要的监督**：PID（已有），无需模态标签额外标注（布局已知）。
- **与 UAD 损失的交互**：
  - CE：从“一个分类边界”变成“三个边界”，TIR 不再被 RGB/NIR 的类中心挤压；这是最直接的作用。
  - WRT：不变；可进一步把 WRT 拆成“模态内 WRT + 全局 WRT”，避免 TIR 正样本被 RGB 正样本淹没。
  - ProCA：每模态中心的语义更纯，ProCA 的“模态中心→身份中心”更合理。
  - GPD：原型仍按身份共享；如果后续要做每模态原型，这一层是前置条件。
- **预期收益**：+1–3 mAP，TIR→TIR 和 TIR→RGB 最明显。
- **风险**：TIR 类别样本少，独立分类头容易欠拟合；需要 TIR 重采样/类别平衡，或对 TIR 头使用更低 weight decay。共享 BN 会削弱收益，建议至少分类器独立。

### 第 3｜测试时相机特征校准 + query expansion
- **来源**：MayI（P17）、DART³（P16）；query expansion/re-ranking 可接 Cheb-GR（P14）或现有 k-reciprocal。
- **插入位置**：
  - `processor/processor.py::do_inference` 之后、`utils/metrics.py::eval_func` 之前：按 camera ID 计算 gallery 特征的均值/方差，对 query/gallery 做 per-camera centering/scaling；或学习 query 侧 scale/shift + 冻结 gallery 侧（DART³ LITE）。
  - `utils/metrics.py`：在距离矩阵上做 query expansion（top-k 邻居特征平均）或相机感知重排；先作为独立开关，默认关闭。
- **需要的监督**：无标签；只用 camera ID 与 gallery 特征。属于 transductive 后处理。
- **与 UAD 损失的交互**：完全正交，不改 CE/WRT/ProCA/GPD，也不改训练。它只改变推理特征分布，因此最适合先做“无训练成本”的 A/B。
- **预期收益**：+1–3 mAP，跨相机/ground-aerial 更明显；对 Rank-1 的收益可能小于 mAP。
- **风险**：相机样本少时统计不可靠；如果 query 与 gallery 的相机分布差异大，校准可能把 query 推离正确邻居。建议：只在 gallery 侧做校准、对样本数 <50 的相机回退到全局统计、用验证集选“是否启用”。

### 第 4｜模态路由 LoRA 专家（参数高效模态容量）
- **来源**：ReID5o（P02）；DM-Adapter（P15）提供域感知路由的替代方案。
- **插入位置**：
  - `model/backbones/vit_pytorch.py`：把每个 block 的 `qkv`/`proj`/`fc2` 的 `nn.Linear` 包成 `ModalityLoRALinear`：`y = Wx + c_m B_m A_m x`，`r=4`，`B_m` 零初始化。
  - `model/make_model.py`：构造 `modality_id` 张量（0/1/2，长度 3B），传入 backbone forward；不需要改数据管线。
- **需要的监督**：PID（已有）；模态标签来自 batch 布局。
- **与 UAD 损失的交互**：
  - CE：LoRA 增加模态特定容量，TIR 分类头（若同时做第 2 项）受益最大。
  - WRT：在更强的模态特征上计算，正负样本更可分；注意 LoRA 初期输出为 0，WRT 不会突变。
  - ProCA：模态中心更分散/更可辨，ProCA 的“拉近”才有意义。
  - GPD：原型质量提升；但 LoRA 参数不参与原型更新，bank 仍按特征 EMA。
- **预期收益**：+1–3 mAP，TIR 最明显；参数 +2–3M。
- **风险**：小数据上 LoRA 过拟合；建议只加在 attention 的 qkv 与 proj，rank=4，LoRA LR 为 base 的 0.1–0.2；若 TIR 样本过少，可共享 NIR/TIR 的 LoRA。

### 第 5｜频域增强 / RFA + 频率辅助损失
- **来源**：MFEN（P22）的 RFA/FAO、SEPG-Net（P20）的伪红外、MFENet（P21）的 HiLo-FM。
- **插入位置**：
  - `datasets/make_dataloader.py` 的 train transform：对 RGB/TIR 成对样本（同一 tuple）交换振幅谱低频分量（RFA），保留相位；或对 RGB 生成“灰度+相位”伪 TIR 作为额外视图（SEPG）。
  - `model/make_model.py`：可选加一个轻量频域分支（FFT→低/高频滤波→Conv），与 CLS 特征拼接或门控融合。
  - `loss/make_loss.py`：可选 FAO/CMSR 辅助损失，权重 ≤0.1。
- **需要的监督**：PID（已有）；RFA 用同一 tuple 的跨模态图像，不需要同步帧（UAD tuple 本身不要求帧同步）。
- **与 UAD 损失的交互**：
  - CE/WRT：增强后的图像仍用原 PID，监督不变；RFA 相当于“保持身份的光照归一化”。
  - ProCA：跨模态特征更接近，模态中心方差下降，ProCA 更容易。
  - GPD：原型更稳定；注意增强后的特征不要直接进 bank（用原图特征更新）。
- **预期收益**：仅 RFA/伪 TIR 预期 +0.5–1.5 mAP；加频域分支 +1–2 mAP。
- **风险**：改变低频可能改变人体轮廓/背景；生成式伪 TIR 有身份幻觉。建议 RFA 只交换 10–30% 低频、保留相位，并在 TIR→TIR 上验证不掉点。

### 第 6｜直接 AP / listwise 排序损失
- **来源**：SupRank/ROADMAP（P13）。
- **插入位置**：
  - `loss/make_loss.py`：新增 `SupAPLoss`，输入是 L2 归一化后的 BN 前特征 `global_feat` 与 PID；输出与 WRT 加权组合（先替换 WRT，再尝试 0.5×WRT + 0.5×Sup-AP）。
  - `processor/processor.py`：如果 batch 太小，增加一个特征 memory queue（按 PID 存归一化特征），在 queue+batch 上算 AP 代理。
- **需要的监督**：PID（已有），batch 内 listwise 标签。
- **与 UAD 损失的交互**：
  - CE：不变；Sup-AP 只作用于检索特征。
  - WRT：功能重叠，建议先替换；WRT 的未归一化欧氏几何与 Sup-AP 的余弦几何不一致，叠加可能互相拉扯。
  - ProCA：Sup-AP 让同类特征更聚、异类更散，ProCA 的中心对齐更容易；但两者都在“拉近正样本”，需控制总权重。
  - GPD：GPD 是全局原型分类，Sup-AP 是 listwise 排序，二者互补；可保留 GPD，Sup-AP 权重 0.5–1.0。
- **预期收益**：+0.5–2 mAP；mAP 收益可能大于 Rank-1。
- **风险**：UAD 每卡 192 张图、16 个身份，batch AP 估计噪声大；必须用 `L_DG*` 或 memory queue；SupRank 代码是为 ResNet/检索设计的，需要适配 ViT 特征与 AMP。

### 第 7｜Query-aware patch 匹配重排
- **来源**：BIT（P12）的 QA Scoring；ViSA（P09）的局部图融合可做训练期版本。
- **插入位置**：推理期，`processor/processor.py::do_inference` 或 `utils/metrics.py`：
  1. 用 UAD 全局特征取 top-K（如 20）候选；
  2. 从 `vit_pytorch.py` 暴露最后一层 patch token（需改 forward 返回）；
  3. 对每个 query–候选对计算双向 patch 相似度、top-k 过滤、互为 top-k 对应，MLP/加权聚合为 `Ψ`；
  4. 最终分数 = `α·cos(global) + (1-α)·Ψ`，α 在验证集选。
- **需要的监督**：推理期无监督；若训练 BCI，则用现有 tuple 的 PID 构造正负对，不需要新标注。
- **与 UAD 损失的交互**：推理期正交；训练期 BCI 会改变 backbone 特征，进而影响 CE/WRT/ProCA/GPD，风险较高，建议先只做推理重排。
- **预期收益**：+1–3 mAP，航拍小目标/遮挡更明显；Rank-1 也可能提升。
- **风险**：计算量大（N_q × K × 128² patch 相似度）；跨模态 patch 对应可能不可靠，需限制 K 和 top-k（如 k=5）；BIT 代码未释放。

### 第 8｜模态共享/特定 token 解耦 + 正交约束
- **来源**：MDReID（P01）；与第 2 项同源，但结构改动更大。
- **插入位置**：
  - `model/backbones/vit_pytorch.py`：在 patch embedding 前 prepend 两个 token（specific/shared），输出 `[shared, specific, patches]`；`forward_features` 返回两个 CLS。
  - `model/make_model.py`：分类器接 shared+specific（或各自分类），推理时可按需只取 shared 或拼接；ProCA/GPD 用 shared 特征。
  - `loss/make_loss.py`：ROL（6×6 相似度目标）+ KDL；权重从 0.1 起。
- **需要的监督**：PID + 模态标签（batch 布局）。
- **与 UAD 损失的交互**：
  - CE：每模态分类器 + 共享/特定分离，TIR 分类边界独立。
  - WRT：可在 shared 特征上做跨模态 WRT、在 specific 上做模态内 WRT。
  - ProCA：specific 分支承担模态差异，shared 分支更适合做身份中心，ProCA 可以只作用在 shared 上。
  - GPD：原型用 shared 特征，减少模态噪声；specific 不进 bank。
- **预期收益**：+2–5 mAP（若实现正确）；但工程量大、推理维度变化。
- **风险**：6 槽特征维度、mask、分类器/原型/评测全链路都要改；UAD 小数据上可能过拟合。建议在第 1–7 项稳定后再做。

---

## 5. 机制 × UAD 损失交互矩阵

| 机制 | CE | WRT | ProCA | GPD | 主要风险 |
|---|---|---|---|---|---|
| 每模态分类器/BN | **增强/拆分** | 正交（可拆模态内 WRT） | 中心更纯 | 原型更干净 | TIR 头欠拟合 |
| 相机/平台原型校准 | 正交 | 正交 | **直接增强** | **直接增强** | 每相机样本少 |
| 可靠记忆/软蒸馏 | 正交 | 正交 | 稳定中心 | **替代硬 CE 部分** | 确认偏差 |
| 模态 LoRA 专家 | 增强容量 | 特征更强 | 中心更可分 | 原型更好 | 过拟合/参数 |
| 频域增强/RFA | 正交 | 正样本更近 | 中心方差下降 | 原型更稳 | 身份/轮廓破坏 |
| Sup-AP | 正交 | **替代/竞争** | 正样本更聚 | 互补 | batch 小、尺度 |
| patch 重排 | 正交 | 正交 | 正交 | 正交 | 计算量/对应噪声 |
| token 解耦 | **拆分** | 可分层 | **只作用 shared** | **只作用 shared** | 工程量大 |
| 测试时相机校准 | 正交 | 正交 | 正交 | 正交 | transductive 过拟合 |

---

## 6. 不建议优先迁移 / 需要警惕的机制

1. **完整 CLIP 文本提示框架**（HiHR、SeCap、MP-ReID/Uni-Prompt）：UAD 使用 ImageNet ViT-B/16，没有文本编码器；两阶段 prompt 训练会改变优化路径，且 UAD 的瓶颈是 TIR 表征而非“缺少语义先验”。可借用“平台条件化”思想，但不要整套搬。
2. **双曲空间/层级表示**（HiHR）：理论优雅，但在 500–1000 ID 的 UAD 上曲率/数值稳定性难调，收益不确定；粗/细双层原型的思想更实用。
3. **隐私/联邦/匿名化**（PPA、L2RW）：与 UAD benchmark 的主指标无关；除非未来要做跨机构部署。
4. **生成式热红外/低光增强**（GAN/diffusion）：容易产生身份幻觉，尤其在 TIR 细节本已稀缺时；若要用，必须加身份保持损失/关键点约束，并只在训练增强中使用。
5. **mixup/cutmix**：会混合身份标签，破坏 CE/WRT/ProCA 的身份语义；跨模态应优先“保持标签的光照/频域增强”（RFA、灰度+相位），而不是样本混合。
6. **无代码/仅 README 的论文**（BIT、PoSeD、MP-ReID、MFEN、Cheb-GR、DART3、MayI）：机制可借鉴，但复现成本高；在排期上应放在有代码的方法之后。
7. **直接把 CLIP-ReID 套到空-地**：SeCap 论文中 CLIP-ReID 在 LAGPeR A→G 仅 24.40/17.60，远低于 ViT 的 38.67/27.25；UAD 若换 CLIP 初始化需要重新验证，不能假设 CLIP 一定更好。

---

## 7. 建议的实验落地顺序（从便宜到贵）

**阶段 A（1–2 天，零/低训练成本）**
1. 在 `utils/metrics.py` 加 per-camera 特征中心化/缩放开关，做 A/B（MayI）。  
2. 在 `do_inference` 加 flip TTA（水平翻转特征平均）与 query expansion；注意 UAD 已 L2 归一化，flip 可能改变航拍视角，先在小 gallery 验证。  
3. 统计 UAD 每个 camera 的样本数、平台映射、TIR 占比，确认哪些相机可做校准。

**阶段 B（1–2 周，训练改动小）**
4. 每模态分类器 + 每模态 BNNeck（第 2 项），单独 A/B。  
5. GPD 置信度加权 + 历史原型快照（SPRED/PoSeD 简化版），单独 A/B。  
6. 加 RFA（低频振幅交换）或伪 TIR 视图（SEPG），单独 A/B。

**阶段 C（2–4 周，结构改动）**
7. 模态 LoRA 专家（第 4 项）+ 每模态分类器组合。  
8. Sup-AP 替换 WRT（第 6 项），先小权重、后全权重。  
9. 相机/平台原型校准 + OT（第 1 项）与 GPD 联合训练。

**阶段 D（高成本/高风险）**
10. patch 重排（第 7 项）作为推理插件，先离线评估 top-K 候选。  
11. token 解耦（第 8 项）或 ViSA 局部图融合（P09）。  
12. 测试时训练（HTT/DART3）作为部署选项，不作为 benchmark 主结果。

**评测纪律**：所有改动需同时报告 AS-ReID、3×3 模态对、A→G/G→A、mINP（需先补 `utils/metrics.py` 的 mINP 实现）；至少 2–3 个 seed，避免把 +0.1 mAP 当稳定收益。TIR→TIR 是最敏感的单项指标，应作为早期 gate。

---

## 8. 链接与代码核验日志（2026-10-29）

| 论文 | 论文页 | 代码页 | 核验结果 |
|---|---|---|---|
| MDReID | arxiv.org/abs/2510.23301 | github.com/stone96123/MDReID | 200；含 train/test/configs |
| ReID5o | arxiv.org/abs/2506.09385 | github.com/Zplusdragon/ReID5o_ORBench | 200；含 `ReID5o/` |
| MP-ReID | arxiv.org/abs/2503.17096 | mp-reid.github.io | 项目页 200；**方法代码未找到** |
| WSL-VIReID | arxiv.org/abs/2507.12942 | github.com/KongLingqi2333/WSL-VIReID | 200；含 `WSL_ReID/` |
| CLNS | sciencedirect.com/.../S0031320326008381 | github.com/AeroVILab-AHU/CLNS | 200；含 train/test/权重 |
| PoSeD | arxiv.org/abs/2609.11514 | github.com/Terminator8758/PoSeD | 200；**仅 README，代码未释放** |
| SPRED | arxiv.org/abs/2507.01884 | github.com/zhoujiahuan1991/ICCV2025-SPRED | 200；含 continual_train |
| Instruct-ReID++ | arxiv.org/abs/2405.17790 | github.com/hwz-zju/Instruct-ReID | 200；含训练/评测 |
| ViSA | arxiv.org/abs/2605.18192 | github.com/Cat-Zero/ViSA | 200；含 fastreid/tools |
| HiHR | arxiv.org/abs/2607.09186 | github.com/YangQiWei3/HiHR | 200；含 fastreid/train.sh |
| SeCap | arxiv.org/abs/2503.06965 | github.com/wangshining681/SeCap-AGPReID | 200；含 fastreid/train.sh |
| BIT | arxiv.org/abs/2603.14243 | github.com/Xuan266/BIT | 仓库 200；**仅 README，代码未释放** |
| SupRank | arxiv.org/abs/2309.08250 | github.com/elias-ramzi/SupRank | 200；官方实现 |
| Cheb-GR | CVF CVPR2025 | — | **未找到开源** |
| DM-Adapter | arxiv.org/abs/2503.04144 | github.com/Liu-Yating/DM-Adapter | 200；含 train/test |
| DART³ | arxiv.org/abs/2505.18337 | — | **未找到开源** |
| MayI | arxiv.org/abs/2502.10195 | — | **未找到开源** |
| HTT | AAAI 2024 | github.com/ziwang1121/HTT | 200；含 TTT 脚本 |
| SVIP | openreview.net/forum?id=I7y7MgsEgO | github.com/LYXRhythm/SVIP | 200；含 train/test |
| SEPG-Net | arxiv.org/abs/2412.19111 | github.com/1024AILab/ReID-SEPG | 200；含 `code/` |
| MFENet | ieeexplore.ieee.org/document/11175191 | github.com/GuHY777/MFENet-VIReID | 200；含 model/losses |
| MFEN | arxiv.org/abs/2606.12051 | — | **未找到开源** |
| Model Stock | arxiv.org/abs/2403.19522 | github.com/naver-ai/model-stock | 200；notebook/权重 |

> 说明：WebFetch 因域名校验无法使用，以上状态由 curl 直接请求 arXiv/GitHub API/出版社页面得到；GitHub “200 + 仅有 README” 被明确标注为“代码未释放”，不视为可用开源。OpenReview 的 SVIP 页面有浏览器校验，未能抓取全文，具体数值未核实。

---

## 9. 关键参考（按首字母/主题）

- CLNS: Camera-aware Label Noise Suppression for Unsupervised VI-ReID. *Pattern Recognition*, 2026. https://www.sciencedirect.com/science/article/abs/pii/S0031320326008381
- DART³: Bhattacharya et al. *DART³: Leveraging Distance for Test Time Adaptation in Person Re-Identification*. arXiv:2505.18337, 2025.
- HiHR: Yang & Zhang. *Hierarchical Hyperbolic Representation for Aerial-Ground Person Re-Identification*. ECCV 2026. arXiv:2607.09186.
- HTT: Wang, Huang, Zheng, He. *Heterogeneous Test-Time Training for Multi-Modal Person Re-identification*. AAAI 2024.
- MDReID: Feng et al. *Modality-Decoupled Learning for Any-to-Any Multi-Modal Object Re-Identification*. NeurIPS 2025. arXiv:2510.23301.
- MFEN: *Multi-Frequency Expert Network for Visible-Infrared Person Re-ID*. CVPR 2026 (Highlight). arXiv:2606.12051.
- MFENet: Gu et al. *Discovering Multi-Frequency Embedding for Visible-Infrared Person Re-identification*. IEEE TCSVT 2025.
- Model Stock: Jang, Yun, Han. *Model Stock: All We Need Is Just a Few Fine-Tuned Models*. ECCV 2024 (Oral). arXiv:2403.19522.
- MP-ReID/Uni-Prompt: Ha et al. *Multi-modal Multi-platform Person Re-Identification: Benchmark and Method*. ICCV 2025. arXiv:2503.17096.
- PoSeD: Wang & Gong. *Prototype Matters: Modality-unified Prototype Self-distillation for Unsupervised VI-ReID*. ACM MM 2026. arXiv:2609.11514.
- ReID5o: Zu et al. *ReID5o: Achieving Omni Multi-modal Person Re-identification in a Single Model*. NeurIPS 2025. arXiv:2506.09385.
- SeCap: Wang et al. *Self-Calibrating and Adaptive Prompts for Cross-view Person Re-Identification in Aerial-Ground Networks*. CVPR 2025. arXiv:2503.06965.
- SEPG-Net: Ge et al. *Spectral Enhancement and Pseudo-Anchor Guidance for Infrared-Visible Person Re-Identification*. ICASSP 2025. arXiv:2412.19111.
- SPRED: Xu et al. *Self-Reinforcing Prototype Evolution with Dual-Knowledge Cooperation for Semi-Supervised Lifelong Person Re-Identification*. ICCV 2025. arXiv:2507.01884.
- SupRank: Araujo et al. *Optimization of Rank Losses for Image Retrieval*. IEEE TPAMI 2025. arXiv:2309.08250.
- SVIP: Li et al. *Learning Source-free Domain Adaptation for Visible-Infrared Person Re-Identification*. NeurIPS 2025.
- ViSA: Zhang et al. *View-Aware Semantic Alignment for Aerial-Ground Person Re-Identification*. CVPR 2026. arXiv:2605.18192.
- WSL-VIReID: Zhang et al. *Weakly Supervised Visible-Infrared Person Re-Identification via Heterogeneous Expert Collaborative Consistency Learning*. ICCV 2025. arXiv:2507.12942.
- 徐等. *UAD/WHU-MARS 原论文与代码分析*. 本地文件 `WHU/01_UAD_WHU_Analysis.md`, 2026-10-25.
