# 2025–2026 第二轮文献调研：UAD/ViT 单模型内部机制

> 调研日期：2026-10-29。目标：为 WHU-MARS 上的 UAD（ViT-B/16，三模态 × ground/aerial）寻找**不依赖 CLIP 文本塔**、可插入单模型内部、有望让 mAP/R1/R5/R10 协同提升的机制。
>
> **本轮证据说明（重要）**：受本轮时间窗口限制，只有 ViSA 与 UAD 基线代码是逐行读过的；其余条目按“已核实仓库/摘要”或“沿用 phase1 已核验链接”标注。凡未读全文的论文，**不写具体消融数值**，统一写“未提取”。所有链接状态以 2026-10-29 的 GitHub API / arXiv 返回为准。

**证据等级**
- **A**：本轮读过方法/代码（ViSA 代码、UAD 基线代码）。
- **B**：本轮核实仓库存在且含代码/仅 README，读了摘要或 README，未读全文表格。
- **C**：仅摘要/检索结果；或链接沿用 phase1 记录、本轮未复核。

---

## 0. UAD 基线事实（本轮读代码确认，作为插入位置依据）

| 组件 | 代码位置 | 事实 |
|---|---|---|
| Backbone | `WHU/model/backbones/vit_pytorch.py`、`make_model.py` | 三模态沿 batch 拼接进同一个 ViT-B/16；只有 1 个 CLS，无 view/platform token |
| 分类 | `make_model.py::build_transformer` | 共享 `Linear(768,C)` + 共享 `BNNeck`；闭集训练准确率已 94.9–95.4%（phase2 记录） |
| WRT | `WHU/loss/wrt_loss.py` | 在 **pre-BN 原始特征**上算欧氏距离；softmax 加权正/负；`SoftMarginLoss`；可选 camera 负样本降权 |
| ProCA | `WHU/processor/processor.py::PCA` | `feat.reshape(3,B,D)`；每模态内每 4 个同 ID 样本求均值 → 模态中心；再向三模态全局中心拉近；**只返回 g_cent，不返回各模态中心** |
| GPD | `processor.py::PrototypeBank/GPD` | 每身份 **1 个** 768D 原型；零初始化（暖启动可选）；`f@P.T/tau`，tau=0.03，硬 CE；更新用 `g_cent`（已跨模态平均） |
| 推理 | `processor.py::do_inference` | pre-BN CLS，L2 归一化，平方欧氏；无 TTA/rerank |

**关键结构性缺口**（与 phase1/phase2 诊断一致）：
1. GPD 每身份单原型，把 RGB/NIR/TIR × ground/aerial 压成一个点；而平台差异远大于模态差异（RGB-G↔RGB-A 余弦 0.388 vs RGB-G↔NIR-G 0.954）。
2. ProCA 的 `g_cent` 被直接用于更新 GPD，导致**原型更新信号本身已被跨模态平均**；即使改成多原型，若仍用 g_cent 更新，子中心会塌缩。
3. 平台/视角没有任何条件化参数；ViT 的 CLS 与 patch token 共享全部 LayerNorm/QKV。
4. 历史失败模式：任何在共享 CLS 上新增的 batch-local 排序/对比损失，都会抬高最危险负样本（A-V1：472 个 R1 新失败中 463 个伴随最危险负样本相似度上升）。

---

## 1. 论文卡片

### A. 平台/视角条件化与航空-地面 ReID

#### P01｜ViSA: View-Aware Semantic Alignment for Aerial-Ground Person Re-Identification
- **Venue**：CVPR 2026
- **论文**：https://arxiv.org/abs/2605.18192
- **代码**：https://github.com/Cat-Zero/ViSA （**已核实**：仓库 1734 KB，含 `fastreid/`、`configs/`、`tools/train_net.py`；默认分支 main，2026-03-03 更新）
- **证据等级**：A（读摘要 + 逐行读 `vision_transformer_multiview_ViSA.py`；未读论文表格）

**核心机制（代码级数据流）**
1. **View token**：在 patch 序列前插入 `[CLS, view_token, patches...]`；`view_token` 由 `view_id` 在 `view_token_sky` / `view_token_ground` 两个 `nn.Parameter(1,1,768)` 中选择。`pos_embed` 长度为 `num_patches+2`，即**为 view token 预留了独立位置行**，不需要插值。
2. **inner_sub**：每个 Transformer block 后执行 `x[:,0] = x[:,0] - x[:,1]`，即每层都把 view token 从 CLS 中减掉；最终 `global=x[:,0]`、`view=x[:,1]`，再在外部算 `inv = global - view`。
3. **ETGM（专家 token 生成）**：三个 Router（share/sky/ground）各是 `Linear(768, E)`，`E=8`、`top_k=2`；对 top-2 logits 做 softmax 得权重，`load_balance_loss = E * Σ importance²`。每个专家是一个 `MoEBlock`：可学习 prompt `[1,16,768]` 作为 query，对 `inv_features` 做 cross-attention（PromptRecapBlock），输出 `[B,16,768]`。
4. **DLFM（局部图融合）**：用 `inv/view` 与 patch token 的余弦相似度取 top-`k=64` patch，作为 GCN 的局部节点；边特征 `cal_edge_emb` 在 `torch.no_grad()` 下计算并 `.detach()`；GCN 输出残差加回 prompt；最后 `out_token` + `final_attn_token_to_image` 汇聚成检索特征。
5. 返回 `global_features`、`out_feat[:,0]`、`view_features`、prompt、load-balance loss。

**关键增益**：摘要称 CARGO 跨视角协议 **+10.06% mAP**；分模块消融数值未提取。

**代码实现要点（迁移时必须知道）**
- `view_token_sky/ground` 用 `torch.zeros` 初始化，且 `no_weight_decay()` 返回的是字符串 `'view_token'`，与实际参数名 `view_token_sky/ground` **不完全匹配**；若优化器按精确名排除 WD，这两个 token 会被 weight decay 压向 0——与历史 SCSR“零初始化 + WD → dead”同型风险。
- `pos_embed` 为 `num_patches+2`，加载 ImageNet 权重时若直接 copy 会 shape mismatch；ViSA 的 `load_param` 对 pos_embed 有单独处理（需按仓库实现插值/补行）。
- `inner_sub` 在**每一层**都改写 CLS，梯度路径极强；不是训练辅助分支，会直接改变最终特征。

**对 UAD 的可迁移点**
- 最小版本：只加 1 个 platform token（ground/aerial），`pos_embed` 追加一行 0，token 用 `trunc_normal(std=0.02)` 而非零初始化，**排除 WD**；先不做每层减法，改为 `cls = cls - λ·view_token`，λ 为可学习标量、初始 0.1。
- 平台标签可由相机元数据/机场划分得到，不需要新人工标注。
- 不要直接搬 ETGM+DLFM 全量：UAD 只有 500/1000 ID，8 专家 × top-2 容易路由塌缩；且新增分支若不写入最终 CLS，会重演“私有头吸收梯度”。

---

#### P02｜HiHR: Hierarchical Hyperbolic Representation for Aerial-Ground Person Re-Identification
- **Venue**：ECCV 2026
- **论文**：https://arxiv.org/abs/2607.09186
- **代码**：https://github.com/YangQiWei3/HiHR （**已核实**：17814 KB，master，2026-08-05 更新，含完整代码）
- **证据等级**：B（仓库核实；方法细节沿用 phase1 阅读：CLIP 视觉-文本编码器 + Text-guided Multi-granularity Fusion + Hierarchical Hyperbolic Learning）

**核心机制**：多粒度（全局/局部/部件）特征经文本引导融合，再在双曲空间做“粗粒度跨视角一致 + 细粒度视角特异”的层次约束；双曲距离对大层级结构更敏感。

**关键增益**：未提取（本轮未读表格）。

**对 UAD 的可迁移点**：双曲层次原型可作为 GPD 原型的距离度量替代（欧氏 → Poincaré 距离），但需要重写原型归一化与 CE logits，风险高；且该方法绑定 CLIP 文本塔，**不建议本轮迁移**。可借用“粗/细两级原型”的思想：身份级原型 + 平台级原型，而不是完整双曲框架。

---

#### P03｜SD-ReID: View-Aware Stable Diffusion for Aerial-Ground Person Re-Identification
- **Venue**：IEEE TIP 2026
- **论文**：https://arxiv.org/abs/2504.09549
- **代码**：https://github.com/924973292/SD-ReID （**已核实**：2450 KB，main，2026-05-13 更新）
- **证据等级**：B（仓库核实，未读全文）

**核心机制**：ViT ReID 编码器产生身份/视角条件，微调 Stable Diffusion v1.5 生成增强特征，再用 View-Refined Decoder 融合实例级与全局特征。

**对 UAD 的可迁移点**：生成式增强对 TIR 低质量有吸引力，但扩散模型推理成本高、存在身份幻觉；且 UAD 训练集仅 500 ID，生成样本容易记住训练身份。**不建议作为单模型主线**，最多作为离线数据增强的对照。

---

#### P04｜MP-ReID / Uni-Prompt ReID: Multi-modal Multi-platform Person Re-Identification
- **Venue**：ICCV 2025
- **论文**：https://arxiv.org/abs/2503.17096
- **代码/项目页**：https://mp-reid.github.io/ （phase1 核实：**方法代码未开源**）
- **证据等级**：C（沿用 phase1；本轮未复核）

**核心机制**：CLIP 文本提示拆成 ReID-specific / modality-aware / platform-aware 三段；轻量视觉网络产生 context 偏移加到 prompt；两阶段训练分别学分模态、分平台提示。

**关键增益（phase1 记录）**：MP-ReID 12 设置平均 R1 64.73 / mAP 58.45，Base 55.51/47.98；跨模态+跨平台类别 28.40/23.08 → 43.16/40.26。

**对 UAD 的可迁移点**：**只借“平台条件化”思想**，降级为 platform token / platform-specific BN，不依赖文本塔。UAD 的平台标签可从相机高度/数据集来源得到。

---

### B. 开放集度量、原型与专家

#### P05｜Sub-center ArcFace: Boosting Face Recognition by Large-scale Noisy Web Faces
- **Venue**：ECCV 2020（期刊版 TPAMI 2022）
- **论文**：https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560715.pdf
- **代码**：https://github.com/deepinsight/insightface （**已核实**：64180 KB，master，2026-10-19 更新；含 sub-center ArcFace 实现）
- **证据等级**：A（机制为经典；代码仓库本轮核实）

**核心机制**：每个身份 K 个子中心 `W[c,k]`；样本对类 c 的 logit 取 `max_k cos(θ_{c,k})`；只有与样本最接近的子中心收到该样本的梯度，其余子中心吸收噪声/困难样本。原论文 K=3、m=0.5、s=64。

**关键增益**：原论文在人脸噪声数据集上显著优于单中心 ArcFace；具体数值未在本轮复核，故不引用。

**对 UAD 的可迁移点（本轮最推荐）**
- GPD 原型 bank 从 `[C,768]` 改为 `[C,K,768]`，logits 变为 `max_k (f · P[c,k]) / tau`，CE 不变。
- **子中心的语义**不必是“噪声聚类”，可以是**同一身份的不同模态/平台中心**：用 ProCA 已经算出的每模态中心初始化 K=3 个子中心，正好对应 RGB/NIR/TIR；平台可在第二阶段扩到 K=6。
- 更新时不能用 `g_cent`（跨模态平均），必须用**该样本所属模态的中心**更新对应子中心，否则子中心塌缩。这要求修改 `PCA` 返回 `centers` 列表（当前只返回 `g_cent`）。
- 风险：UAD 每 ID 每 batch 仅 4 样本，K 过大会导致子中心“饿死”；建议 K=2–3、计数 EMA、暖启动。

---

#### P06｜MDReID: Modality-Decoupled Learning for Any-to-Any Multi-Modal Object Re-Identification
- **Venue**：NeurIPS 2025（Spotlight）
- **论文**：https://arxiv.org/abs/2510.23301
- **代码**：https://github.com/stone96123/MDReID （**已核实**：1521 KB，main，2026-07-03 更新，含 train/test）
- **证据等级**：B（仓库核实；数值沿用 phase1 全文阅读记录）

**核心机制**：每个模态 prepend 两个可学习 token（模态特定 `I_sp^M`、模态共享 `I_sh^M`），三模态共 6 槽；ROL 正交损失把 6×6 相似度矩阵拉向预设目标（同模态特定=1、跨模态共享=1、特定-共享=0）；KDL 约束共享/特定知识差异；**每模态独立分类器**。

**关键增益（phase1 记录）**：RGBNT201 82.1 mAP/85.2 R1，TOP-ReID 72.3/76.6；每模态分类器相对单分类器平均 +13.4 mAP/+13.7 R1。

**对 UAD 的可迁移点**：UAD 闭集分类已饱和（94.9→97.5），**每模态分类器的边际收益预计小于 MDReID 的场景**；但“共享/特定 token 分离 + 正交约束”可作为 platform token 的扩展。完整 6 槽会改变 768D 推理接口，ProCA/GPD 都要改，风险高。

---

#### P07｜DeMo: Decoupled Feature-Based Mixture of Experts for Multi-Modal Object Re-Identification
- **Venue**：AAAI 2025
- **代码**：https://github.com/924973292/DeMo （**已核实**：17840 KB，master，2025-03-09 更新，含完整代码）
- **论文**：仓库描述指向 AAAI 2025；arXiv 链接本轮未核实
- **证据等级**：B（仓库核实；论文未读）

**核心机制（仓库描述层面）**：把多模态特征解耦后送入 MoE，由专家分别处理模态共享/特定成分；具体路由与损失未在本轮读代码确认。

**对 UAD 的可迁移点**：与 P01 的 ETGM 同类，可作为“模态专家”的第二实现参考；但 UAD 只有 3 模态，专家数应控制在 3–6，且必须保留共享专家常开。

---

#### P08｜WSL-VIReID: Weakly Supervised Visible-Infrared ReID via Heterogeneous Expert Collaborative Consistency Learning
- **Venue**：ICCV 2025
- **论文**：https://arxiv.org/abs/2507.12942
- **代码**：https://github.com/KongLingqi2333/WSL-VIReID （**已核实**：1390 KB，main，2026-03-06 更新）
- **证据等级**：B（仓库核实；数值沿用 phase1）

**核心机制**：可见光/红外各一个身份分类器作为“异构专家”；编码器前几层模态不共享、后几层共享；专家各自用 CE+WRT 在模态内训练；再用 Count Priority Selection 估计跨模态身份对应矩阵，反向监督共享分类器一致性。

**关键增益（phase1 记录）**：SYSU-MM01 All-Search 70.4 R1/66.6 mAP/52.7 mINP；基线模态内训练 47.8/47.2。

**对 UAD 的可迁移点**：UAD 有完整三模态标签，不需要对应矩阵；可借“模态内 WRT + 全局 WRT”两项，避免 TIR 正样本被 RGB 正样本淹没。但这会在 WRT 内新增一项排序监督，**与历史失败模式相邻**；若做，必须只改 WRT 内部权重、不新增损失项，并单独消融。

---

#### P09｜CLNS: Camera-aware Label Noise Suppression for Unsupervised Visible-Infrared Person Re-ID
- **Venue**：Pattern Recognition 2026
- **论文**：https://www.sciencedirect.com/science/article/abs/pii/S0031320326008381
- **代码**：https://github.com/AeroVILab-AHU/CLNS （phase1 核实含 train/test；**本轮未复核**）
- **证据等级**：C（沿用 phase1 摘要+README）

**核心机制**：Camera-aware Prototype Calibration 用跨相机一致性修正聚类原型；Optimal Transport Prototype Matching 做全局跨模态原型匹配；Neighbor-guided Camera-domain Learning 用邻域软标签稠密化；Noise-aware Memory Updating 按可靠性更新记忆。

**关键增益（phase1 记录）**：SYSU-MM01 69.49 R1/65.64 mAP/51.2 mINP；RegDB V→IR 94.79/91.83。

**对 UAD 的可迁移点**：把 ProCA 的“每身份每模态中心”升级为“每身份每模态每相机中心”，再用收缩（shrinkage）向全局中心靠拢；NMU 的可靠性权重可直接用于 `PrototypeBank.update`。风险是每相机样本更少，中心方差大——**必须 shrinkage，不要硬分组**。

---

#### P10｜PoSeD: Modality-unified Prototype Self-distillation for Unsupervised VI-ReID
- **Venue**：ACM Multimedia 2026
- **论文**：https://arxiv.org/abs/2609.11514
- **代码**：https://github.com/Terminator8758/PoSeD （phase1 核实：**仅 README，代码未释放**）
- **证据等级**：C

**核心机制**：模态统一原型 + 自蒸馏，用原型作为跨模态软目标；具体实现未读。

**对 UAD 的可迁移点**：概念上与 GPD 暖启动/软目标一致；**代码不可用，不进入落地清单**。

---

#### P11｜FlexiReID: Adaptive Mixture of Expert for Multi-Modal Person Re-Identification
- **Venue**：arXiv 2510.15595（2025-10；venue 未核实）
- **论文**：https://arxiv.org/abs/2510.15595
- **代码**：本轮未找到官方仓库（**代码不可访问/未核实**）
- **证据等级**：C（仅摘要）

**核心机制（摘要）**：支持 RGB/红外/素描/文本四种模态、七种检索组合；自适应 MoE 按置信度阈值路由，跨模态 query 融合。

**对 UAD 的可迁移点**：置信度阈值路由比 top-k 更适合小数据（低置信样本走共享专家）；可作为 P01 Router 的替代。但无代码，不能直接复用。

---

#### P12｜FSLoRA: Harmonizing Detection and Re-Identification via Freq-Spatial Low-Rank Adapter
- **Venue**：CVPR 2026
- **论文**：https://openaccess.thecvf.com/content/CVPR2026/html/Tian_FSLoRA_Harmonizing_Detection_and_Re-Identification_via_Freq-Spatial_Low-Rank_Adapter_for_CVPR_2026_paper.html
- **代码**：https://github.com/personsearch/FSLoRA （**已核实：仓库 0 KB，空仓库，代码不可访问**）
- **证据等级**：C（论文页标题/摘要层面；代码不可用）

**核心机制（摘要/检索结果）**：每层插入 LoRA 风格 adapter，FFT 分低/高 freq 路由 + 空间 MoE，解耦检测与 ReID 特征；<2% 额外参数；非 CLIP，ImageNet 预训练骨干。

**对 UAD 的可迁移点**：频率分离对 TIR（低频热轮廓）与 RGB（高频纹理）有直观对应；但**代码不可用，且 RLE 历史失败表明“通道复制 + 伪彩色”会灾难**。若做，只在 patch token 上做幅度谱归一化，不做通道复制。

---

### C. ViT 内部机制

#### P13｜Revisiting [CLS] and Patch Token Interaction in Vision Transformers
- **Venue**：ICLR 2026
- **论文**：https://arxiv.org/abs/2602.08626
- **代码**：本轮未核实到官方仓库（**代码未验证**）
- **证据等级**：C（仅摘要 + 检索摘要）

**核心机制（摘要）**：标准 LayerNorm 已隐式区分 CLS 与 patch 的统计量；作者进一步在**每个 norm 层为 CLS 与 patch 使用独立 affine（γ,β）**，并只在**前 1/3 层**使用独立 QKV；分类精度保持，分割 +2 mIoU，参数约 +8%，**无额外推理 FLOPs**。

**对 UAD 的可迁移点（低风险高优先）**
- 只改 LayerNorm affine：`ln.weight` 从 `[768]` 变 `[2,768]`，按 token 类型索引；两份都初始化为原权重，等价于恒等启动，不破坏 ImageNet 预训练。
- 前 1/3 层独立 QKV 参数量约 +14M（ViT-B），对 500 ID 可能过拟合；建议先只做 LN affine（约 +3k 参数），QKV 作为第二消融。
- 不新增损失、不改变推理接口、不引入训练-测试不一致；与 CE/WRT/ProCA/GPD 完全正交。

---

#### P14｜BIT: Matching-based Bi-directional Interaction Transformation Network for VI-ReID
- **Venue**：CVPR 2026
- **论文**：https://openaccess.thecvf.com/content/CVPR2026/html/...（本轮未定位到稳定页面）
- **代码**：https://github.com/Xuan266/BIT （**已核实：仓库仅 266 字节 README，代码未释放**）
- **证据等级**：C（仅 README + 检索摘要）

**核心机制（摘要）**：从“共享空间对齐”转向 pairwise matching：Bi-directional Cross Interaction + Query-aware Scoring + reciprocal patch mining。

**对 UAD 的可迁移点**：patch 级匹配对 TIR 局部结构有吸引力，但它是 **query-gallery 成对交互**，推理时对每个 query 重算，成本高；且属于“排序/匹配”类，容易重演 batch-local 排序失败。**不推荐本轮**。

---

#### P15｜MFEN: Multi-Frequency Expert Network for Visible-Infrared Person Re-ID
- **Venue**：CVPR 2026（Highlight，检索结果）
- **论文**：本轮未定位稳定链接
- **代码**：未找到（**代码不可访问**）
- **证据等级**：C

**核心机制（检索摘要）**：多频带专家 + sigmoid 门控（非 top-k）+ Random Frequency Augmentation + 频率辅助损失。

**对 UAD 的可迁移点**：与 P12 同属频域；sigmoid 门控（所有专家加权）比 top-k 更平滑，适合小数据。但无代码、无消融细节，暂列观察。

---

#### P16｜Exploring the Camera Bias of Person Re-identification
- **Venue**：arXiv 2502.10195（2025-02；venue 未核实）
- **论文**：https://arxiv.org/abs/2502.10195
- **代码**：本轮未核实
- **证据等级**：C（仅摘要）

**核心机制（摘要）**：系统测量 ReID 模型在**未见域**的相机偏差，发现分布偏移下偏差更严重；把“嵌入向量特征归一化”重新作为测试时去偏方法，并提出训练期缓解相机偏差伪标签的策略。

**对 UAD 的可迁移点**：UAD 的离线实验已证明逐模态 mean/z-score/中心化/CORAL/ZCA **全部无效或灾难**（phase2 表）；因此**不要**把该文的归一化直接搬到 UAD 推理。可借的是“训练期相机感知”而非“测试期变换”。

---

### D. 训练策略（辅助）

#### P17｜Model Stock: All We Need Is Just a Few Fine-Tuned Models
- **Venue**：ECCV 2024 / ICLR 2025（phase1 记录；本轮未复核）
- **论文/代码**：沿用 phase1 记录
- **证据等级**：C

**核心机制**：用少量微调模型的权重平均（model soup）逼近更大模型的效果；对学习率/模型数量有解析式指导。

**对 UAD 的可迁移点**：phase2 已规划 EMA + soup；本轮不新增。注意：**BN buffers 必须一起平均**，否则训练-测试统计不一致。

---

#### P18｜SupRank / ROADMAP: Optimization of Rank Losses for Image Retrieval
- **Venue**：TPAMI 2025（phase1 记录）
- **代码**：phase1 已核验
- **证据等级**：C（沿用 phase1）

**核心机制**：listwise AP 代理 + 全局 memory queue，显式补偿 batch AP 与全局 AP 的分解差距。

**对 UAD 的可迁移点**：**与历史失败模式直接冲突**。UAD 的 A-V1/B-B/WCV1 已证明 batch-local/混合 gallery 排序损失会让最危险负样本上升、R1 下降。若一定要用，只能作为**离线诊断**（用全局 gallery 算 AP 梯度），不要进训练损失。

---

## 2. 横向对比表

| 编号 | 机制 | 是否新增损失 | 是否改推理特征 | 代码 | 对 UAD 的主要作用点 | 风险 |
|---|---|---|---|---|---|---|
| P01 | view token + MoE + 局部图 | 否（仅 load-balance） | 是（写入 CLS） | 可用 | 平台/视角条件化、TIR 局部结构 | 零初始化+WD dead；路由塌缩 |
| P02 | 双曲层次原型 | 是 | 否 | 可用 | 原型距离 | 绑定 CLIP；改动大 |
| P03 | 扩散增强 | 是 | 是 | 可用 | TIR 数据增强 | 成本/身份幻觉 |
| P04 | 平台 prompt | 是（两阶段） | 是 | 不可用 | 平台条件化 | 依赖文本塔 |
| P05 | 子中心原型 | 否（改 GPD） | 否 | 可用 | 开放集几何、模态内方差 | K 过大饿死 |
| P06 | 模态特定/共享 token | 是（ROL/KDL） | 是 | 可用 | 模态解耦 | 接口改动大；闭集已饱和 |
| P07 | 解耦特征 MoE | 是 | 是 | 可用 | 模态专家 | 小数据过拟合 |
| P08 | 异构专家 + 模态内 WRT | 是（新增 WRT 项） | 否 | 可用 | TIR 正样本 | 与排序失败模式相邻 |
| P09 | 相机感知原型校准 | 否（改 ProCA） | 否 | 可用 | 平台/相机中心 | 每相机样本少 |
| P10 | 原型自蒸馏 | 是 | 否 | 不可用 | 原型可靠性 | 无代码 |
| P11 | 置信度路由 MoE | 是 | 是 | 不可用 | 模态路由 | 无代码 |
| P12 | 频率-空间 LoRA | 否 | 是 | 不可用 | TIR 频域 | 空仓库；伪彩色风险 |
| P13 | CLS/patch 独立 LN/QKV | 否 | 是（CLS） | 未核实 | ViT 内部表征 | QKV 过拟合 |
| P14 | patch 匹配 | 是 | 是 | 不可用 | 局部匹配 | 推理成本/排序风险 |
| P15 | 频率专家 | 是 | 是 | 不可用 | TIR 频域 | 无代码 |
| P16 | 测试时归一化 | 否 | 是 | 未核实 | 相机偏差 | UAD 离线已失败 |
| P17 | model soup | 否 | 否 | 可用 | 训练稳定性 | BN buffers |
| P18 | listwise AP | 是 | 否 | 可用 | mAP | 与 R1 冲突 |

---

## 3. Top 8 最值得迁移的机制（按“全指标协同提升”潜力排序）

### 第 1｜GPD 子中心原型 + 模态/平台感知更新（P05 + P09）
**为什么排第一**：不改任何损失项、不改推理接口、直接针对已定位的“单原型混合平台/模态”缺口；GPD 是 UAD 最大增益来源（Table 8 单独 +2.0 mAP/+4.8 R1），在其内部改造的杠杆最大。

**最小可行实现**
```
# bank: P [C, K, D], counts [C, K]
# PCA 额外返回 centers: list of 3 tensors [B, D]（每模态中心）
f = normalize(feat)                       # [3B, D]
logits = einsum('nd,ckd->nck', f, P) / tau # [3B, C, K]
logits = logits.max(dim=-1).values         # [3B, C]
loss = CE(logits, target.repeat(3))

# update（no_grad）：样本 i 属于模态 m、身份 c
k* = argmax_k cos(f_i, P[c,k])
P[c,k*] = normalize((1-m_eff)*P[c,k*] + m_eff*centers[m][i])
counts[c,k*] += 1; m_eff = clamp(1/(counts+1), 0.1, 0.5)
```
**初始化**：K=3，用 ProCA 的 RGB/NIR/TIR 中心分别填入；若某模态缺失，复制类均值 + 小噪声。
**与现有损失交互**：CE/WRT/ProCA 不动；GPD 的 `g_cent` 只用于缺失原型的兜底。tau 需重标定（max over K 会抬高 logit），建议先保持 0.03 并监控 GPD loss 量级。
**预期/风险**：预期 mAP +0.3–1.0、R1 +0.3–1.0；风险是 K 过大导致子中心饿死、或 max 操作放大噪声。先做 K=2 的 A/B。

### 第 2｜平台 token + 软视角减除（P01 的轻量版）
**为什么**：平台差异（0.388）远大于模态差异（0.954），是最大结构缺口；view token 是唯一能**在不新增损失**的情况下把平台信息注入 CLS 的机制。

**最小可行实现**
```
# pos_embed: [1, N+1, D] -> [1, N+2, D]
pe_new[0,0] = pe_old[0,0]      # CLS
pe_new[0,1] = 0                # platform token 占位，零位置
pe_new[0,2:] = pe_old[0,1:]    # patch 位置完全不变
p_tok = trunc_normal(std=0.02) # 非零初始化
x = cat([cls, p_tok[platform_id], patches])
# 每个 block 后（可选，先关）：
cls = cls - gate * p_tok; gate = sigmoid(alpha), alpha init=-2.2 (gate≈0.1)
```
**关键**：p_tok 必须加入 `no_weight_decay`（按参数名精确匹配），否则重演 SCSR dead。
**与现有损失交互**：CE/WRT/ProCA/GPD 全部照旧；平台标签来自相机元数据，无新标注。
**预期/风险**：预期 +0.3–1.0 mAP，AG 方向更明显；风险是平台标签噪声、以及每层减法过强。先只加 token 不减，再加 gate。

### 第 3｜CLS/Patch 解耦 LayerNorm affine（P13）
**为什么**：零新增损失、零推理 FLOPs、恒等初始化，不破坏预训练；直接缓解“CLS 与 patch 共享统计量”的摩擦，对全局检索特征是纯增益型改动。

**最小可行实现**
```
# 原: ln(x) with weight[768], bias[768]
# 新: weight[2,768], bias[2,768]
w = cat([w0, w0]); b = cat([b0, b0])
idx = 0 for cls token, 1 for patch tokens
y = (x - mean)/std * w[idx] + b[idx]
```
**第二阶段**：前 4 层 QKV 拆成 CLS 专用/ patch 专用两份，均用原权重初始化；参数量 +~14M，需更低 LR（0.5×）。
**预期/风险**：预期 +0.2–0.8 mAP/R1；风险是 QKV 拆分过拟合 500 ID，先只做 LN。

### 第 4｜共享专家常开 + 模态/平台 LoRA 专家（P01 Router + P06/P07 思想）
**为什么**：UAD 三模态共享全部参数，TIR 容量被 RGB 主导；低秩专家能在不改 768D 接口的前提下增加模态容量。

**最小可行实现**
```
# 在后 6 层每个 qkv/mlp 旁：
y = W x + (shared_gate * B0 A0 x) + (expert_gate[m] * Bm Am x)
# A: [r,768], B: [768,r], r=4；B 零初始化，A 随机
# 路由：top-1 硬路由 + 共享专家恒开；load_balance 权重 0.01
```
**关键**：共享专家必须常开，保证任一样本都有梯度路径；专家 gate 用 sigmoid、初始 0.1，且**排除 WD**。
**预期/风险**：预期 +0.5–1.5 mAP，TIR 方向更明显；风险是路由塌缩、以及新增容量让早期更快拟合但后期饱和（TAUR 模式）。必须跑满 120 epoch 看尾段。

### 第 5｜局部 patch 图融合写入最终 CLS（P01 DLFM）
**为什么**：TIR 的判别信息更多在轮廓/局部结构；全局 CLS 平均池化会丢失。ViSA 的 GCN 把局部 token 汇聚后通过 `out_token` 写入最终特征，终端路径明确。

**最小可行实现**
```
# 用 CLS 与 patch 的余弦取 top-k=32 patch
# 节点 = [prompt(16), top-k patch]；边 = cal_edge_emb(...).detach()
# GCN -> residual + prompt -> out_token cross-attn -> 残差写回 CLS
cls = cls + gamma * out_token   # gamma init 0.1, 排除 WD
```
**关键**：必须写回最终 pre-BN CLS；不能只接私有头（PAT/WQVD 的失败）。
**预期/风险**：预期 mAP +0.3–1.0，但 R1 可能波动；风险是 TIR patch 噪声、GCN 边在 no_grad 下不可学习。先只对 TIR 分支启用。

### 第 6｜GPD 暖启动 + 计数 EMA + tau 课程（P09 + phase2 GPD-WR）
**为什么**：GPD 零初始化 + 固定 EMA 0.2 让早期原型不可靠；计数 EMA 让少见身份/模态更新更快，温度课程避免早期 softmax 饱和。
**实现**：`m_eff = clamp(1/(count+1), 0.1, 0.5)`；tau 从 0.05 线性降到 0.03（前 20 epoch）；暖启动用训练集类均值（phase2 已实现）。
**预期/风险**：+0.2–0.5 mAP；风险是暖启动引入训练集偏差，需在 120 epoch 末看是否被消化。

### 第 7｜每模态分类器（P06/P08 的最小部分）
**为什么**：TIR 分类边界被 RGB/NIR 主导；每模态独立分类器让 TIR 有自己的线性边界，且不改变推理特征。
**实现**：`classifier` 复制 3 份（或 1 份共享 + 3 个残差 bias），按模态索引取 logits；CE 不变。
**预期/风险**：预期 +0.2–0.8 mAP，TIR→TIR 更明显；风险是 UAD 闭集已饱和（95%），收益可能有限；参数量 +3×C×768（C=500 时约 1.1M，可接受）。

### 第 8｜长时训练 + EMA + model soup（P17）
**为什么**：唯一不引入任何结构风险的杠杆；基线 90→120 仍在上升（+0.29 mAP/+0.97 R1），说明优化未饱和。
**实现**：从 120 ckpt 以 1/8 LR 续训 60 epoch；EMA decay 0.999；最后 3–5 个 checkpoint 等权平均（含 BN buffers）。
**预期/风险**：+0.3–1.0 mAP/R1；风险是续训过拟合、soup 的 BN 统计不一致。必须预注册“取最后一轮”，不能用测试集挑 epoch。

---

## 4. 与历史失败模式冲突、需避免的机制

| 机制 | 冲突原因 | 证据 |
|---|---|---|
| batch-local / mixed-gallery 排序损失（含 SupRank、BIT 式匹配、WCV1） | 优化 batch 内候选，抬高最危险负样本；R1 净下降 | A-V1：mAP −0.42/R1 −2.06，98.1% 新失败伴随最危险负样本上升；B-B −0.15/−2.84；WCV1 +0.09/−1.33 |
| 固定大权重融合（>5–10%）局部/辅助分支 | 距离被弱分支占据；α=0.01 才回到基线 | HiHR 15% 融合 mAP 10.78 vs 11.02 |
| 零初始化 + weight decay 的小模块 | gate/残差被压到 1e-8，ON/OFF 相同 | SCSR v3：残差/Patch≈3e-8，gate≈2.2e-5 |
| 全局特征变换（中心化、z-score、ZCA、CORAL、LDA） | UAD 特征几何已与训练目标匹配，任何线性变换都破坏身份方向 | 离线：ZCA 4.07 mAP、CORAL 9.91、LDA 10.48（基线 11.02） |
| Query expansion / re-ranking | QE λ=1：mAP 14.24 但 R5/R10 崩；λ=0.1 仍 R5/R10 −2.4 | phase2 离线表 |
| 测试时熵最小化/自训练 TTA | 无标签、易漂移；相机偏差文的归一化在 UAD 已失败 | P16 + phase2 离线 |
| 训练用随机 ROI/增强、测试不用 | 训练-测试不一致 | phase1 诊断 |
| NIR/TIR 当三通道伪彩色做独立通道扰动 | 制造伪彩色，RGB 自检索降幅最大 | RLE 失败 |
| 用测试集挑中间 epoch / 融合权重 | 不可复现的乐观偏差 | phase2 设计原则 |
| 子中心 K 过大（K≫每 ID 样本数） | 子中心饿死、max logit 噪声 | 机制推论（P05） |
| MoE load-balance 权重过大 | 路由均匀化，专家不分化 | P01 代码：`E*Σimportance²` 需小权重 |

---

## 5. 建议实验顺序（从便宜/高确定性到贵）

1. **GPD 子中心 K=2**（从 120 ckpt 续训 30 epoch，单变量）——验证开放集几何假设。
2. **CLS/patch 独立 LN affine**（从头或续训）——零损失改动，先确认不伤基线。
3. **平台 token（只加不减）**——验证平台条件化；平台标签先按相机映射。
4. **共享专家 + 3 模态 LoRA（r=4，后 6 层）**——容量实验；必须跑满 120 epoch。
5. **局部图融合（仅 TIR）**——终端路径写回 CLS。
6. **每模态分类器**——与 1/3 可叠加消融。
7. **长时训练 + EMA + soup**——作为所有结构的底座；先单独跑，确认 +0.3 以上再叠加。

每个实验：固定种子、预注册末轮、报告 mAP/R1/R5/R10/mINP 与九组合分解；新增模块必须有 ON/OFF 开关。

---

## 6. 链接与代码核验日志（2026-10-29）

| 条目 | 链接 | 状态 |
|---|---|---|
| ViSA 论文 | arxiv.org/abs/2605.18192 | 200，摘要已读 |
| ViSA 代码 | github.com/Cat-Zero/ViSA | API 200；1734 KB；含 fastreid/configs/tools；代码已读 |
| BIT 代码 | github.com/Xuan266/BIT | API 200；**仅 266B README，代码未释放** |
| HiHR 代码 | github.com/YangQiWei3/HiHR | API 200；17814 KB；完整 |
| SD-ReID 代码 | github.com/924973292/SD-ReID | API 200；2450 KB；完整 |
| MDReID 代码 | github.com/stone96123/MDReID | API 200；1521 KB；完整 |
| WSL-VIReID 代码 | github.com/KongLingqi2333/WSL-VIReID | API 200；1390 KB；完整 |
| DeMo 代码 | github.com/924973292/DeMo | API 200；17840 KB；完整 |
| FSLoRA 代码 | github.com/personsearch/FSLoRA | API 200；**0 KB 空仓库，代码不可访问** |
| insightface（sub-center） | github.com/deepinsight/insightface | API 200；64180 KB；2026-10-19 更新 |
| CLS/patch ICLR 2026 | arxiv.org/abs/2602.08626 | 200，摘要已读；代码未核实 |
| Camera bias | arxiv.org/abs/2502.10195 | 200，摘要已读；代码未核实 |
| FlexiReID | arxiv.org/abs/2510.15595 | 200，摘要已读；代码未找到 |
| HiTPro | arxiv.org/abs/2604.21324 | 200，摘要已读；代码未找到 |
| PoSeD | github.com/Terminator8758/PoSeD | 沿用 phase1：仅 README |
| CLNS | github.com/AeroVILab-AHU/CLNS | 沿用 phase1：含 train/test；本轮未复核 |

---

## 7. 一句话结论

**本轮最值得做的是“GPD 子中心原型 + 平台 token + CLS/patch 解耦 LN”这三件不新增排序损失的事**；它们分别修正原型几何、平台条件化和 ViT 内部表征，且都能沿现有 CE/WRT/ProCA/GPD 的数据流生效。排序/重排/测试时归一化继续回避——历史证据已经足够强。
