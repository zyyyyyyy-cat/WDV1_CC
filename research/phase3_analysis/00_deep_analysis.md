# UAD/ViT 历史实验、指标现象与机制深度分析

> 范围：UAD/ViT 基线（含最新 WDV1-A0）。**不包含 CLIP/WC 实验。**
> 目标：从代码机制与指标现象出发，重新理解 UAD 的瓶颈，筛出下一轮唯一核心机制。
> 证据：`02_Previous_Experiments.md`、`05_Results.csv`、两组历史代码审计、最新 A-V1/B-B/WCV1 审计、`feature_statistics.csv`、WDV1-A0 日志。
> 标记：[C] 代码/日志事实；[R] 实验/指标支持；[H] 机制假设。

---

## 0. 一页结论

1. **A0（终段 LR 再训 30 epoch）无效**：指标从 epoch 5 起平，epoch 30 还略降；损失也平。模型在 120 epoch 已收敛。90→120 的增益来自 cosine LR 衰减，不是“多走 min-LR 步数”。**训练 recipe 不是当前瓶颈。**
2. **历史方法不是“没执行”**：大量模块确实更新、梯度非零，但终端路径断开、数值 dead/饱和、训练-测试候选不一致、梯度预算失控、单卡→DDP 不迁移，导致指标不升或退化。
3. **DDP 基线上没有任何方法实现 mAP/R1 双升**；单卡只有 KGL 浅层 adapter 和 PAT E1a 实现 6–7/9 模态对双升，且 PAT E1a 迁到 DDP 后失败。**单卡 broad 正信号不能直接外推 DDP。**
4. **TIR 是“表征弱”，不只是“中心偏”**：TIR-G 的 hardest negative 最高、hardest positive 偏低、margin 最差；只做原型/中心/分数校准无法创造 TIR 身份细节。需要改善 TIR/NIR **特征质量** 的机制。
5. **平台差异 > 模态差异**（质心余弦 0.39 vs 0.93），UAD 的 ProCA 只按模态聚合、GPD 单原型混合平台；平台是最大的未建模结构。
6. **GPD 是唯一被证明有效的全局机制**（+2.0 mAP/+4.8 R1），但单原型混合模态/平台；batch-local 排序/对比损失（A-V1/B-B/WCV1/TAUR/PAT）全部失败或只抬 R5/R10。
7. 当前最值得认真比较的两条核心机制：
   - **C1：场景感知全局原型**（identity×modality×platform 原型，正样本 max(场景,身份)，负样本最难场景）——全局覆盖、全指标潜力高，但替换 GPD 有风险、无直接实验。
   - **C2：场景条件化浅层 adapter**（L1–4 按 modality×platform 低秩残差，深层共享）——有 KGL 6/9 pair 双升的直接正证据，但单卡、幅度小、DDP 未验证。
8. 不要再用：batch-local 排序/对比损失、固定大权重局部融合、强输入扰动、全局后处理特征变换、零初始化+WD 的小模块、训练-测试不一致的随机 ROI/同相机邻居。

---

## 1. 证据基线与方法

- 基线 H014：mAP 11.0234 / R1 29.7112 / R5 46.3232 / R10 54.5511 / mINP 0.6595。[R]
- 论文目标 +1.0pp：1000 12.0/30.5/47.3/55.8；GD 15.5/32.9/46.0/53.5；2337 10.6/26.7/41.9/49.2。
- 31 个 UAD/ViT 正式实验（1000 AS，相对同模式 baseline）。
- 代码审计覆盖 HiHR、ViSA、MulConv/SPM、ViT-L、SCSR、SFTC、ICRE、KGL、M-SpecGene、TAUR、PAT、PEFT、ReID5o、WQVD、RLE、A-V1、B-B、WCV1。
- 最新 WDV1-A0：120 checkpoint，恒 6.4e-5，30 epoch；最终 11.03/29.51/46.42/54.47。

---

## 2. 历史实验机制总表

| 实验 | 实际机制 | 最终路径/梯度 | 指标现象（vs 同模式） | 根因 | 可保留信号 |
|---|---|---|---|---|---|
| HiHR H1/H2 | L5/8/12 CLS → Parent/Child 查询 → 1792D 拼接；固定权重 | 三支都接主损失；WRT 标定放大 46× | mAP +0.42、R1 −1.14；3/9 pair 双升 | 固定融合 15% 过强；Stage 预算混杂；R1 边界被稀释 | mAP/mINP 正；小 α=0.01 回基线 |
| ViSA+MD+MP | shared CLS + modality token；ETGM/DLFM 局部图；1536D 拼接 | fusion WRT detach shared；local 固定 20% 距离 | mAP +0.28、R1 −0.34；1/9 pair 双升 | 80% CLS 未直接适配最终距离；local 弱信息进入距离 | TIR→RGB 小升 |
| MulConv / SPM | FFN 并行多尺度卷积；SPM 区域 GeM 10752D | 主损失直接监督拼接特征 | MulConv −0.05/−0.38；SPM −0.47/−1.47 | 区域语义不对齐、GeM 负值截断、维数膨胀 | MulConv 3/9 pair 双升 |
| ViT-L | ViT-L/16；batch/LR 同时减半 | 原四损失 | mAP +0.42、R1 −0.50 | 容量与优化混杂；每卡 96 图 | NIR→RGB +1.14/+1.22 |
| SCSR v3/v4.1 | TIR Haar 高频条件调制；v4 改 Block7 attention 前 | 只改 TIR patch，经后续层到 CLS | 基本持平；ON/OFF 近零 | v3 数值 dead（残差 3e-8）；v4 m_rms 1.66% | ON/OFF 证据方法可复用 |
| SFTC | CNN local K/V 与 ViT K/V 联合 softmax | 经 Patch 到 CLS；local 质量 4–14% | mAP −1.00、R1 −3.12；0/9 双升 | 联合 softmax 改变原路由；ON>OFF 但 OFF 已低于 baseline | 即时补偿≠整体收益 |
| ICRE | L6/L9 patch 路由 → L9 残差写回 | gamma 0.01→0.436，残差 25–28% | mAP −2.73、R1 −6.90；0/9 | 全 Patch 强平滑；校准放大 89.8× | 初始小残差≠全程小 |
| KGL adapter | L1–3 每 block 后按模态 64 维瓶颈残差（含 CLS） | 后续 9 层共享；主损失训练 | mAP +0.15、R1 +0.21；6/9 双升 | 单卡、容量小、无特殊 LR/WD；TIR→RGB R1 仍降 | **最广 pair 正信号；值得强化并测 DDP** |
| KGL HNSM | CLS detach 挖难关系；Patch 关系头 BCE | 关系头不参与推理 | mAP +0.18、R1 −0.05；7/9 双升 | 候选相机规则与评测不一致；私有头 | pair 覆盖广，头部收益弱 |
| M-SpecGene | 多光谱预训练 Encoder 初始化 | 全主干微调 | mAP −4.91、R1 −13.02；0/9 | pos_embed 轴错配（cos 0.818）+ MAE/分辨率差异 | 严格 key 检查/位置插值经验 |
| TAUR | L6/L9 TIR 有界 adapter + 四方向 rank loss | rank 梯度只回 adapter（1.5–2.1%） | mAP −0.45、R1 −0.31；R5 +0.85/R10 +0.53 | 梯度子空间过窄；有界残差饱和；sampler 混杂 | R5/R10 正；TIR→TIR R1 +0.75 |
| PAT E1a single | Part token + CSL/PSD | 只用 CLS 推理；Part 与 CLS 断开 | mAP +0.31、R1 +0.37；6/9 双升 | 单卡有效，DDP 失败（+0.02/−0.92） | 单卡 broad 正；不可外推 DDP |
| PAT E1b/E2–E4 | 随机 ROI / Part-ID / Part-GPD | 同上 | 大多退化；E3 最差 | CSL 饱和、同相机邻居 92.7%、目标张力 | E4 在 NIR→TIR R1 +1.17 |
| PEFT DPA/TCA | FFN 并行 patch 瓶颈 / CLS adapter | 末层 DPA 只写 Patch、无 patch loss | 全部退化；DPA 末层空路径 | 终端路径断开；全参数微调下冗余 | TCA 在 RGB→TIR R1 +1.08 |
| ReID5o FFN experts | 末 2 层 FFN 按模态 rank4 专家 | 全 token 含 CLS；路由一致 | −0.05/−0.17；0/9 双升 | 共享 FFN 全可训练使专家冗余；ProCA/GPD 拉回共享中心 | TIR→TIR 小升 |
| WQVD A1/A2 | L11 局部残差 / L12 条件通道门控 | A1 主损失监督 z=c+0.05u；local CE 训 u | A1 +0.15/−0.58；A2 0/−0.71 | 局部 CE 不保证跨模态稳定；固定小残差仍改 top-1 | A1 同权重融合 +0.25/+0.92；A2 4/9 双升 |
| RLE E1–E4 | MRLE/RRLE 通道/区域扰动 | 输入级 | −1.27～−1.43 / −3.91～−4.50；0/9 | 增强过强、伪彩色、欠拟合 | 无；模态分解定位可复用 |
| A-V1 | MP 模态原型 + Pair/Mixed + 软困难采样 | 新增损失占 20–34.5%，无预算 | −0.42/−2.06；3/9 双升 | 候选 3% 身份；正例占 softmax 36–44%；负例变强 | MP/局部 pair 有部分 TIR 收益 |
| B-B | Query-risk 负身份分层 ranking 替换 WRT | BB 占总损失 39–41% | −0.15/−2.84；0/9 | 3% 负身份覆盖；固定权重 85.8；softplus 饱和 | batch top-1 改善但不迁移全库 |
| WCV1 | Mixed-gallery rank loss；正例组均值；rho 0.10 | 候选 detach，仅 anchor 回传 | +0.09/−1.33；4/9 双升 | 正例吸引主导（cos 0.86）；12.6% 覆盖；rho 不约束方向 | mINP +0.064；TIR→RGB mAP 小升 |
| WDV1-A0 | 120 checkpoint，恒 6.4e-5，30 epoch | 原四损失 | +0.01/−0.20/+0.10/−0.08 | 模型已收敛；低 LR 只漂移 top-1 | 无；否定“更多 min-LR 步数” |

---

## 3. 指标现象：量化与统一解释

### 3.1 ΔmAP 与 ΔR1 高度正相关

- 31 个实验：Pearson corr(ΔmAP, ΔR1) = **0.968**。
- DDP 11 个：3 个 mAP↑R1↓，7 个双降，1 个持平；**0 个双升**。
- 单卡 20 个：2 个双升（PAT E1a +0.31/+0.37；KGL adapter +0.15/+0.21），1 个 mAP↑R1↓，17 个双降。

**解释**：大多数模块是“整体变差”，mAP 与 R1 一起掉；mAP↑R1↓ 是少数固定融合/多分支方法的特征。真正稀缺的是**双升**，尤其 DDP 双升为零。

### 3.2 Pair 级覆盖：单卡有广覆盖，DDP 没有

9 个模态对中“mAP 与 R1 同时 > +0.05”的对数：

| 方法 | 模式 | 双升/9 | mAP 升/9 | R1 升/9 | 双降/9 | 平均 ΔmAP/ΔR1 |
|---|---|---:|---:|---:|---:|---|
| KGL HNSM | single | **7** | 9 | 7 | 0 | +0.28/+0.26 |
| KGL adapter | single | **6** | 8 | 7 | 0 | +0.25/+0.27 |
| PAT E1a single | single | **6** | 8 | 6 | 0 | +0.41/+0.31 |
| WQVD A2 | single | 4 | 6 | 4 | 0 | +0.10/+0.06 |
| HiHR H1 | DDP | 3 | 5 | 3 | 4 | +0.01/−0.45 |
| ViSA seed1234 | DDP | 1 | 2 | 4 | 3 | −0.21/−0.37 |
| 其余 DDP | DDP | 0–1 | 0–1 | 0–3 | 6–9 | 负 |

**关键**：KGL/PAT 的广覆盖正信号都在单卡；PAT E1a 迁到 DDP 后 0/9 双升、−0.43/−0.64。**单卡 broad 正信号不能直接外推 DDP。**

### 3.3 TIR 五对 vs 非 TIR 四对

DDP 方法中：

- 没有任何方法让 TIR 五对的 mAP 与 R1 同时提升。
- HiHR 是唯一 TIR5 mAP 微正（+0.04/+0.07），但 TIR5 R1 负（−0.21/−0.11），非 TIR R1 更负（−0.76/−0.50）。
- KGL/PAT 单卡 TIR5 mAP 正，但 R1 幅度小（KGL HNSM +0.23，KGL adapter +0.04）。
- 其余 DDP 方法 TIR5 双负。

### 3.4 每个模态对的最佳方法

| 方向 | 最佳 ΔmAP | 最佳 ΔR1 |
|---|---|---|
| RGB→RGB | KGL adapter +0.78/+0.89（single） | SCSR V3 +0.94（DDP，mAP −0.27） |
| RGB→NIR | PAT E1a +0.53/+0.24（single） | HiHR H2 +0.42（DDP，mAP +0.19） |
| RGB→TIR | PEFT TCA +0.28/+1.08（single） | PEFT TCA +1.08 |
| NIR→RGB | PAT E1a +0.77/+1.27（single） | PAT E1a +1.27 |
| NIR→NIR | PAT E1a +0.53/+0.28（single） | KGL adapter +0.70（single） |
| NIR→TIR | PAT E1a +0.23/−0.04（single） | PAT E4 +1.17（single，mAP +0.15） |
| TIR→RGB | HiHR H2 +0.19/+0.33（DDP） | HiHR H1 +0.38（DDP） |
| TIR→NIR | PAT E2 +0.20（single）；HiHR H2 +0.13/+0.57（DDP） | PAT E1a DDP +1.27（mAP −0.23） |
| TIR→TIR | PAT E1a +0.54/+1.03（single） | PAT E1a +1.03 |

**观察**：不同 TIR 方向的最佳机制不同（part、adapter、frequency、ranking），说明 TIR 需要 **按场景条件化的统一机制**，而不是单一局部模块。HiHR 的 DDP TIR→RGB/TIR→NIR 小升，但代价是非 TIR R1 大降；如果能“只在 TIR/跨平台启用”，可能保留收益。

### 3.5 特征质量：TIR 是“表征弱”，不只是“中心偏”

Ground 同场景同模态（`feature_statistics.csv`）：

| 场景 | mean pos | hardest pos | hardest neg | margin |
|---|---:|---:|---:|---:|
| RGB-G | 0.383 | 0.204 | 0.537 | −0.333 |
| NIR-G | 0.326 | 0.142 | 0.530 | −0.389 |
| TIR-G | 0.323 | 0.151 | 0.554 | −0.404 |

- TIR 的 hardest negative 最高、hardest positive 偏低、margin 最差。
- 这不是简单的“模态中心偏移”；TIR 类内可分性本身弱。
- **只做原型/中心/分数校准，无法创造 TIR 身份细节；需要改善 TIR 特征质量的机制。**

### 3.6 平台差异大于模态差异

- RGB-G↔RGB-A 质心余弦 0.388；RGB-G↔NIR-G 0.954；RGB-G↔TIR-G 0.927。
- UAD 的 ProCA 只按模态聚合，不按平台；GPD 单原型混合平台。
- **平台是最大的未建模结构；任何场景条件化机制都应把 platform 纳入。**

### 3.7 损失-指标脱耦

DDP 实验最终损失与指标（`epoch_metrics.csv`）：

| 实验 | CE | WRT | PCA | GPD | mAP | R1 |
|---|---:|---:|---:|---:|---:|---:|
| HiHR H1 | 0.172 | 0.090 | 0.192 | 0.069 | 11.44 | 28.57 |
| ViT-L | 0.116 | 0.053 | 0.135 | 0.032 | 11.44 | 29.21 |
| ViSA | 0.107 | 0.111 | 0.197 | 0.082 | 11.30 | 29.37 |
| SCSR V3 | 0.105 | 0.115 | 0.201 | 0.082 | 11.05 | 29.76 |
| baseline | 0.106 | 0.116 | 0.201 | 0.082 | 11.02 | 29.71 |
| TAUR | 0.071 | 0.083 | 0.203 | 0.053 | 10.57 | 29.40 |
| SFTC | 0.129 | 0.119 | 0.204 | 0.098 | 10.02 | 26.59 |
| ICRE | 0.114 | 0.093 | 0.167 | 0.061 | 8.29 | 22.81 |

- **ICRE 的 CE/WRT/PCA/GPD 几乎最低，但 mAP/R1 最差**；TAUR 的损失也低于 baseline，但 mAP 更低。
- 损失下降不等于检索几何变好；GPD 损失大小不是检索质量的代理。
- 这解释了 A0：继续降低损失/更多步数不会自动改善指标；需要改变**几何/目标**。

### 3.8 WDV1-A0 专项

| epoch | mAP | R1 | R5 | R10 |
|---:|---:|---:|---:|---:|
| 5 | 11.03 | 29.68 | 46.35 | 54.68 |
| 10 | 11.03 | 29.73 | 46.32 | 54.49 |
| 15 | 11.04 | 29.70 | 46.46 | 54.54 |
| 20 | 11.03 | 29.65 | 46.48 | 54.54 |
| 25 | 11.03 | 29.56 | 46.45 | 54.52 |
| 30 | 11.03 | 29.51 | 46.42 | 54.47 |

- 损失：CE 0.109→0.115、WRT 0.117→0.118、PCA 0.204→0.203、GPD 0.083→0.093；没有继续下降。
- 训练 acc 0.98；GPD warm start 500/500、92,133 样本；bank 初始化正常。
- 结论：**不是 warm-start 失败，也不是 EMA 没开；是模型在 120 epoch 已经收敛，低 LR 只产生 top-1 漂移。**
- 推论：A2（200 epoch 从头）也不能简单外推为有效；更长 schedule 只有改变 LR 轨迹才可能有效，且需要单独验证。**训练 recipe 不应再作为主线。**

---

## 4. 机制族深挖

### 4.1 多层/局部/多分支

- HiHR/ViSA/PAT/MulConv/SPM 都在“增加局部/多层信息”，但结果要么 mAP↑R1↓，要么全面退化。
- 共同问题：固定融合权重（15–20%）把距离分配给弱分支；局部特征没有独立强检索证据/跨模态稳定性；末端分类器/距离尺度同时改变；单卡 vs DDP 差异巨大。
- 可保留：HiHR 的 mAP/mINP 正、PAT 单卡的 6/9 pair 双升、ViSA 的 TIR→RGB 小升。说明“局部/多层信息”不是无用，而是**配比、监督对象、训练模式**没有解决。

### 4.2 跨层/跨来源融合

- SFTC 的 local K/V 进入原 softmax，改变原注意力路由；ON>OFF 但 OFF 已低于 baseline。
- ICRE 的残差从 1% 涨到 25–28%，全 Patch 强平滑；9 宫格全降。
- 可保留：SFTC 的“即时补偿”证据、ICRE 的“初始安全≠全程安全”教训。
- 机制上，跨层融合要满足：**小残差、可学习门控、按场景条件化、终端路径直达 CLS**。固定大残差/全 Patch 改写风险高。

### 4.3 模态适配/容量

- KGL 浅层 adapter 是唯一 6–7/9 pair 双升的机制；幅度小，但方向正确。
- TAUR 的 TIR adapter 有界、梯度子空间窄，早期升后期饱和。
- ReID5o 末层 expert 近基线；共享 FFN 全训练使专家冗余。
- ViT-L 容量提升 mAP↑R1↓；配置混杂。
- 可保留：**浅层、低秩、按模态/平台条件化的残差适配** 是最有 pair 覆盖证据的方向；需要 DDP 验证与强度提升。

### 4.4 排序/关系/辅助监督

- TAUR/HNSM/PAT/AV1/BB/WCV1 都尝试更接近检索的监督，但：batch 候选覆盖 3%–12.6%；正例吸收/正例预算错误；负例被推近；无梯度预算/衰减，新增损失后期占比 20%–44%；私有头/关系头不参与推理。
- 可保留：TAUR 的 R5/R10、HNSM/PAT 的 pair 覆盖；但**不能再用 batch-local 排序损失作为主机制**。
- 开放问题：**全局原型级**的 margin/ranking 是否可行？GPD 已证明全局原型有效；在原型上做 margin/层次结构，可能避开 batch-local 问题。

### 4.5 原型/中心/采样

- GPD 是主要增益；ProCA 小。
- 单原型混合模态/平台；TIR/平台弱。
- 采样器对 TIR 重复采样；1000 基本平衡，2337 TIR 少。
- 可保留/可升级：场景/层次原型、原型可靠性/计数、原型 margin、相机/平台平衡采样；但不要再加 batch-local 正负样本项。

### 4.6 输入增强

- RLE 四版全面退化；RGB 自检索降幅最大；E2 关 TIR 仍退化。
- SP-MAE 未完成（epoch 61：8.91/25.36），不能判死。
- 可保留：模态分解定位、输入保真条件；**强扰动/伪彩色不可取**；温和的 TIR 强度标准化仍可单变量验证。

### 4.7 训练 recipe

- A0 无效；H018 显示 LR 配置差异可 −2.67/−6.57；基线 90→120 的增益来自 cosine 衰减。
- EMA/soup/TTA 未在 A0 评测；但 A0 final 无增益，EMA 可能略好，不能作为主线。
- 结论：**训练 recipe 不是当前瓶颈；不要再在 recipe 上打转。**

---

## 5. 可保留信号清单

1. **KGL 浅层模态 adapter**：6/9 pair 双升、8/9 mAP 升、0 双降；单卡 +0.15/+0.21/+0.42/+0.03。方向：浅层、低秩、模态条件化。**需要 DDP 验证与强度提升。**
2. **PAT E1a 单卡**：6/9 pair 双升、+0.31/+0.37/+0.62/−0.06；但 DDP 失败。说明局部/Part 信息在单卡有效，DDP 训练动力学不同。
3. **HiHR DDP TIR→RGB/TIR→NIR**：+0.14/+0.38、+0.13/+0.57；代价是非 TIR R1 大降。说明多层信息对 TIR 跨模态有用，固定融合是问题。
4. **GPD 全局原型**：+2.0 mAP/+4.8 R1；UAD 的核心增益；升级它最有机会影响全指标。
5. **TAUR R5/R10**：+0.85/+0.53；说明定向排序能改善覆盖，但伤 mAP/R1。
6. **WQVD A1 同权重融合**：+0.25/+0.92 即时收益；说明局部有信息，但共同训练代价未解决。
7. **A2/EMA/soup/TTA**：未验证；A0 无效，但 EMA/TTA 仍可作为低成本 post-hoc，不应作为主线。

---

## 6. 候选机制矩阵（开放讨论，不锁定）

| 候选 | 机制 | 证据 | 全指标潜力 | 风险 | 优先级 |
|---|---|---|---|---|---|
| C1 场景感知全局原型（SAPD） | identity×modality×platform 原型；正样本 max(场景,身份)，负样本最难场景 | GPD 全局有效；TIR/平台是瓶颈；单原型混合场景 | 高：R1/mAP/R5/R10/TIR/AG 都可覆盖 | 替换 GPD 有风险；场景过分离；6× 计算 | P0 |
| C2 场景条件化浅层 adapter（SCSA） | L1–4 按 modality×platform 低秩残差；共享深层 | KGL 6/9 pair 双升；TAUR 饱和；单卡未迁 DDP | 中高：改善 TIR/NIR 特征质量；保留 RGB | 单卡信号可能不迁 DDP；容量/强度难调 | P0/P1 |
| C3 全局原型 margin / sub-center | 在 GPD 原型上加 cosine margin / 多子中心 | GPD 有效；ArcFace 强；无 UAD 直接证据 | 中高：R1/mAP；可能伤 R5/R10 | margin 过强；子中心塌缩 | P1 |
| C4 场景条件化多层融合 | L6/L9/L12 CLS 按场景门控；小残差 | HiHR TIR 跨模态正；固定融合伤 R1 | 中：TIR/AG；RGB 风险 | 历史融合失败；门控/终端路径 | P1/P2 |
| C5 每模态分类器/BNNeck | 3 分类头/BNNeck | 闭集 gap 3.5pp；MDReID 大增益（不同数据） | 低中：TIR 分类边界 | 闭集饱和；推理不变 | P2 |
| C6 全局 memory queue / 难负 | 跨 batch 特征/原型队列 | 文献；A-V1/B-B/WCV1 gather 失败 | 中：全局难负 | 覆盖/梯度预算；历史负面 | P2 |
| C7 EMA teacher 自蒸馏 | EMA teacher 软目标/一致性 | 训练稳定通用；UAD 无证据 | 低中：泛化 | 计算；与 GPD 交互 | P2/P3 |
| C8 更强 vision-only 预训练 | DINOv2/MAE ViT-B 初始化 | CLIP 视觉 mAP +1（但排除）；M-SpecGene/ViT-L 失败 | 高：整体表征 | 换初始化/位置编码；工程大 | P1（备选） |
| C9 相机/平台平衡采样 | 采样分布对齐场景 | 平台 gap 大；TIR 重复 | 低中：TIR/AG | 非模型机制；可能伤强模态 | P2 |
| C10 输入标准化/温和 TIR 增强 | per-image/模态标准化 | TIR 特征弱；RLE 强扰动失败 | 低中：TIR | 可能去掉热对比度线索 | P2 |

**当前判断（待文献/红队修正）**：
- 若只选一个核心，**C2 有最直接的 broad pair 证据**（KGL 6–7/9 双升），但单卡、幅度小；**C1 有最高天花板**（全局、全指标），但无直接实验、替换 GPD 风险大。
- C3 可作为 C1 的内部加强（margin/sub-center），不单独成核心。
- C4/C5/C9/C10 是 C1/C2 失败后的定向补充。
- C8 是“换更强 ViT 初始化”的备选，不是 UAD 内部机制；若允许，可能比小模块更有效。

---

## 7. 待文献与红队回答

1. 2025–2026 是否有 **scene/domain-aware prototype**、**hierarchical prototype**、**prototype margin** 的开源代码与消融？是否支持 C1？
2. 是否有 **modality/platform-conditioned shallow adapter / FiLM / conditional BN** 在跨模态/跨域 ReID 的强证据？是否支持 C2？
3. 是否有 **vision-only stronger ViT initialization**（DINOv2/MAE）在 ReID 的可靠增益？是否值得作为 C8？
4. 是否存在 **global ranking/AP proxy** 能避开 batch-local 覆盖问题？
5. 是否有 **TIR-specific representation** 的新机制（非强扰动）？
6. 哪个候选最可能在 **DDP 120 epoch** 上同时抬 mAP/R1/R5/R10？需要红队排序。

---

## 8. 下一步

- 合并 01/02/03 三份 agent 报告（历史机制、指标现象、2025–2026 文献）；
- 对 C1/C2/C3 做一次红队审理（机制、数据流、梯度流、实现风险、指标覆盖）；
- 只选一个核心，写唯一配置，训练 120 epoch DDP；
- 训练后按指标现象复盘，再调整。

> 本文件不锁定最终方案；C1/C2 只是当前证据下的两个首选假设。
