# UAD/ViT 基线历史实验机制复盘（含 WDV1-A0）

> 范围：只含 UAD/ViT 基线实验（H001–H048 中与 UAD 相关者 + WAV1/WBV1/WCV1 + WDV1-A0）。CLIP/WC 实验不进入本报告。
> 性质：只读分析，未修改代码、未训练。证据分级沿用审计档案：**[代码事实]** 可在源码行核对；**[实验支持]** 来自日志/CSV/诊断；**[机制假设]** 由代码+现象推出、未经单变量证明。
> 关键路径：基线 `D:/CODE/Doing/WHU_base/WDV1_CC/WHU`；历史审计 `research/phase1_history/*.md`；轨迹 `research/phase1_logs/log_trajectory_report.md`、`epoch_metrics.csv`；指标 `D:/CODE/Doing/WHU_Code/WHU_Research_Notes/05_Results.csv`；A0 日志 `D:/CODE/Doing/WHU_base/WDV1_CC/logs/train_log_20261029_1556.txt`。

---

## 1. 证据基线与方法

### 1.1 基线链路与度量（所有结论的参照系）

```
imgs=[RGB,NIR,TIR] -> cat -> ViT-B/16(256x128, 16x8=128 patch+CLS) -> final LN -> CLS[768]
  -> bottleneck(BN1d) -> classifier -> CE                     (make_model.py:67-80)
  -> f0(pre-BN CLS) -> WRT(欧氏) / ProCA(0.01) / GPD(tau=.03)  (processor.py:279-296)
检索：pre-BN CLS 的 L2 归一化欧氏（metrics.py:462-465），同相机 gallery 全部排除（metrics.py:164）
```

- 总损失 `CE + 1.0*WRT + 0.01*ProCA + 1.0*GPD`，单次 `backward`，全局 `clip_grad_norm_(1.0)`（`processor.py:298-301`）；GPD bank 在 step 后 EMA 更新（`:309`）。
- 训练 batch：PKM，256 个三元组=768 图；WRT 只在**本卡** 192 行内计算；训练身份 500。
- 基线 H014（DDP 256/0.032）：**11.02 / 29.71 / 46.32 / 54.55**；单卡 H019（64/0.008）：10.14/27.81/43.97/52.80；论文 11.0/29.5/46.3/54.8。
- 总体 mINP 正式日志不打印；离线同权重评测为 **0.66**（`phase1_protocol/offline_calibration_all.json`, mode=raw）。

### 1.2 必须先固定的四个数值事实（决定所有机制解释）

| 事实 | 数值 | 来源 | 含义 |
|---|---|---|---|
| 九对 mAP 均值 vs 总体 mAP | 12.29 vs 11.02 | 05_Results R0159–R0168 | 总体 mAP ≈ 0.90×九对均值，近似可加（每对权重 1/9） |
| 九对 R1 均值 vs 总体 R1 | 19.12 vs 29.71 | 同上 | 总体 R1 是九对“并集”，重叠率 F=0.131；单对 R1 +1pp ≈ 总体 +0.28–0.29pp |
| 1 个 query 的分辨率 | 1/6405 = 0.0156pp（R1/mAP 计数） | 6405 query | ±0.2pp R1 ≈ ±13 个 query |
| 重复运行噪声 | 同配方重跑 H029：−0.09/−0.77；ViSA 两 seed 差 0.24/1.13 | 05_Results；H005/H006 | **单 seed 无法分辨 <0.5pp 的模块效应** |

另有：同一 UAD 特征用 raw 欧氏 vs L2 余弦检索，R1 相同但 **top-1 只有 65.8% 一致**（`final_docs/01_diagnosis_synthesis.md` §1.2）——top-1 是一个大量近 tie 的阈值统计。

### 1.3 训练/评测协议差异（所有实验共用，非某个模块的问题）

- 训练 WRT/ProCA/GPD **不排除同相机**（`wrt_loss.py:46-47` 仅按 PID）；评测排除**全部**同相机 gallery（`metrics.py:164`）。同相机异身份负样本被训练、却不被评测；同相机同身份正样本被拉近、却永不考。
- 新增损失若只在本卡 batch 内取负例，则与全 gallery（≈87k gallery、500 身份）不是同一统计量。TAUR 的 rank 是唯一显式排除同相机的（`taur_rank_loss.py:25`），但候选仅本卡 ~49 个 TIR 负样本。
- 评测是 `Validation Results`（测试集每 5/10 epoch），不存在独立验证集；"选中间 epoch"不可用。

---

## 2. 单实验机制表

> 行内数值为 AS-ReID 总体 mAP/R1（相对同模式基线 pp）。8 列按任务书要求；细节见第 3 节。

| 实验 | 机制（新增/修改、位置、作用对象） | 最终路径 | 梯度 | 指标现象 | 根因 | 可保留信号 | 重做假设 |
|---|---|---|---|---|---|---|---|
| **HiHR** H001/H003 | 读 L5/8/12 全序列；TMF 2 query×3 层 CLS 交叉注意力，patch 用 3 个跨头平均标量加权；`F=[√.8u0,√.133up,√.067uc]`；parent/child CE 0.1/0.025+同模态 triplet 0.1+HHL 0.1；两阶段 | 1792 维 F 直接检索，接收全部四项损失；无主路径 detach | 三分量都收 CE/WRT/ProCA/GPD；新参 6.0M，LR×1；WRT 用 9.78F（正/负比不一致 1.45×，每样本梯度×46） | 同预算 **+0.42/−1.14**；同 Stage2 轮次 +1.54/+1.58；9/9 mAP 升；同模态 R1 −1.54/−0.70/−1.55 且 mINP +0.53/+0.32/+0.36；HHL 无增益 | 固定 20% 混合压缩 top-1 margin；Stage2 重 warmup 先破坏几何；WRT 标定把辅助项放大 | Stage1 同 60ep **+0.82/+1.00**；AG A→G +0.56、G→A +0.75/+1.71；9/9 mAP | Stage1-only 120ep；η 离线扫描/自适应门控；WRT 按梯度而非中位数标定 |
| **ViSA+MD+MP** H005/H006 | specific token 进共享序列(129→130)；ETGM 8 专家 Top-2；DLFM top-40 patch 图注意力；`z=[u_s,0.5u_l]/√1.25`；local CE 0.1、specific WRT 0.05、fusion WRT 0.5 | z(1536) 检索；**z 上无 CE**，唯一 fusion WRT 对 shared detach；80% 分量未为 z 优化 | 136M 新参、2×LR、1-D 参带 WD；三处 detach（shared/specific 池、选中 patch、fusion） | +0.28/−0.34（seed1234）；+0.04/−1.47（seed2025）；9/9 mAP 升；同模态 R1 降、mINP 升 | 固定 20% + 最终描述子缺直接 CE + 优化预算（参数 2.6×、2×LR） | 9/9 mAP；seed 方差估计；AG G→A +0.41/+0.78 | z 上加 CE、shared 不 detach、去 specific token 做 token-only 对照 |
| **MulConv / SPM** H012/H007 | 12 层 FFN 并行多尺度卷积（s=0.2，上投影零初始化）；SPM 末层 token 14 区域 GeM 拼 10752 维、全监督 | MulConv 只改 patch→后续 CLS；SPM 直接改检索描述子与分类器/原型维数 | 新参全进 optimizer，base LR/WD；无 detach | −0.05/−0.38；SPM −0.47/−1.47；SPM RGB→RGB mAP −1.29 | 全参数微调下局部卷积冗余；SPM 分量范数不等权+GeM 非负截断+分类器空间改变 | MulConv NIR→RGB +0.08/+0.23 | 冻结主干再测 adapter；SPM 逐分量 L2+学习权重 |
| **ViT-L** H021 | 24 层 1024 维替换 ViT-B；batch/LR 同时变（128/0.016） | 同 UAD 链路，维度随 in_planes | 全参微调；无新损失 | **+0.42/−0.50**；NIR→RGB +1.14/+1.22/+0.33；RR R1 −1.22 | 容量与优化混杂；容量增益偏跨模态 | NIR→RGB mAP/R1/mINP 三项同升（全历史唯一方向） | 匹配 batch/LR 的 ViT-B 对照；只把大主干用于跨模态 |
| **SCSR v3/v4.1** H022/H023 | TIR patch 高频条件调制：v3 在 L6 后（零初始化、平滑有界残差）；v4.1 在 Block7 attention 前（U std .5、output std .02、WD0） | 只改 TIR patch，CLS 保留；经后续层影响 CLS | 无新损失；v3 参数 base LR+WD1e-4；v4 WD0 | +0.03/+0.05；+0.00/−0.17；ON/OFF 相同（v3）/−0.0001pp（v4） | v3 残差/patch≈3e-8（数值 dead，零初始化+小梯度+WD）；v4 仅单层单模态 1.7% RMS | v4 的“非零初始化+WD0+监控 residual RMS”避免 dead | 无（除非扩大到多层/多模态并加跨模态对齐） |
| **SFTC v2.1** H024 | Block7 attention 拼接 128 个 CNN local K/V（与原 129 key 同一 softmax，257）；bias=−2 前 5 轮；无新损失 | 只经 block7 attention 改写 f0；local 不进入后续 token 序列 | 新参 LR×3，1-D 参 WD0；**clip 频率 1.0**，新分支与主干共享预算 | **−1.00/−3.12**；ON−OFF=+0.92/+1.36；OFF−baseline=−1.92/−4.48；patch_only≈ON、cls_only≈OFF | local 是竞争性 K/V（初始质量≈11.8%），改变预训练路由；联合训练代价>模块收益 | 收益全走 Patch 路径（CLS local 质量仅 4–7%）；ON/OFF 分解方法 | 加性 gating 零初始化；或只在 patch 侧加证据并直接监督聚合 |
| **ICRE/CLEE** H025 | L9 后对全部 128 patch 做 128×128 低秩（256）混合，`patches+gamma*delta`；router 2-way 零初始化；校准 out_proj×89.8 | 只改 patch，CLS 经 L10–12 间接；`patches6` 无 detach（梯度直达 L1–6） | 无新损失；gamma 标量自由，训练后 0.01→0.436，残差比 25–28% | **−2.73/−6.90**；9/9 降；RR −5.85/−8.66 | 强平滑算子破坏 patch 身份细节；gamma 无上限；校准同时放大分支内部梯度 90× | 无（仅“监控残差比/gamma 上限”教训） | 冻结主干+gamma 硬上限；或 per-token gate+残差范数约束 |
| **KGL adapter** H026 / HNSM H028 | L1–L3 后每模态零初始化 768→64→768 残差（含 CLS 与 patch）；HNSM 训练-only 关系头（patch mean，BCE 0.5，ep10 起 ramp） | adapter 进最终 CLS（后续 9 层传播）；关系头不参与推理 | 无 LR/WD 特例；`up` 零初始化→down 首步零梯度（两阶段启动） | **+0.15/+0.21/+0.42/+0.03**（历史唯一四项同向）；TIR→RGB R1 −0.52、TIR→TIR mINP −0.03；HNSM +0.18/−0.05 | 幅度小（0.88M/模态、3/12 层、零初始化）；早期模态分离无对齐项 | 四项同向小升；AG A→G R1 +1.63 | 2–3 seed 复现；adapter 附近加跨模态对齐/正交约束、bottleneck 加大 |
| **M-SpecGene** H030 | 只换多光谱 MAE encoder 初始化（150/150 严格加载），位置编码双线性到 16×8 | 同 UAD 链路 | 全参微调；pos_embed 可训练且带 WD | **5.23/14.79**；RGB→RGB −12.14/−18.41；早期 PCA≈0、GPD≈ln500 | 源 pos 是 W-major 固定 sincos，代码按 H-major 插值：cos 0.818、相对误差 55.8%；叠加域/任务差异 | 无 | 修正 pos 转换后重训；先做线性探测 |
| **TAUR v2** H031 | TIR patch 两个 ρ=0.1 有界 adapter（L6/L9 后，只改 TIR）；rank loss 四方向、**排除同相机**，梯度只回 adapter；CameraPKMSampler；WRT_STABLE | 只经 TIR patch 影响 f0；rank 不进标量损失 | rank 对 adapter 梯度仅为 UAD 的 1.5–2.1%，对主干=0；adapter LR×3 | **−0.45/−0.31**，R5 +0.85、R10 +0.53；9/9 mAP 降；T→T R1 +0.33 | 代理目标梯度太小+残差 ≤10%；采样器与 WRT_STABLE 未控制 | 唯一同相机对齐协议；TIR 内 R1 小升；R5/R10 升 | matched control（同 sampler+原 WRT）；rank 放到最终描述子/主干 |
| **PAT** H032–H037 | 3 个 part token + 固定 8×8 区域双向 mask（part↔region patch）；CSL（未归一化点积、τ=.1，logit≈992）；PSD 替换全局 CE；part 头训练-only | 推理只取 global CLS；part 信息经 region patch→后续 CLS 间接 | CSL 对 part token 可微但 top-k/邻居 detach；part 头私有 | E1a single +0.31/+0.37；E1a DDP +0.02/−0.92；E1b 9.31/25.34；E2 9.97；E3 9.07；E4 10.03 | CSL 饱和（loss≈0）；PSD 邻居 96.4% 同 ID/92.7% 同相机；E1b 随机 ROI 在 eval 仍生效；part 不进推理 | single E1a 小升（不迁移到 DDP）；PSD 替换 CE 的接口 | PSD-only、跨相机邻居池、CSL 归一化+温度校准 |
| **PEFT DPA/TCA** H038–H040 | DPA 每层 patch adapter；TCA 每层 CLS tuning + 12 层共享 adapter | DPA 只改 patch（**L12 空路径、梯度恒零**）；TCA 直接改 CLS | 无特例 LR/WD | DPA −0.09/−0.32；TCA −0.39/−2.11；组合 −0.26/−1.53；TCA RR R1 −3.56 | 全参微调下 adapter 冗余；直接改 CLS 破坏预训练几何；L12 DPA 死支路 | 无 | 冻结主干+adapter；或只对 patch 加、不与 CLS 相加 |
| **ReID5o** H041 | 最后 2 层 FFN 的 rank-4 模态 LoRA（B 零初始化），共享权重仍全训 | 经 MLP 进 CLS | 全参 LR/WD；未选专家零梯度 | **−0.05/−0.17**；T→T +0.08/+0.14；N→R R1 −0.75 | 全参微调下容量冗余；ProCA/GPD 与模态特化张力 | TIR 内小升；模态专用低秩路由正确 | 冻结共享 FFN 只训专家；专家只作用于 patch |
| **WQVD A1/A2** H046/H047 | A1 `z=c+0.05u`（u 从 L11 top-16 patch，2 query+关系注意力）+局部 CE 0.05；A2 `z=c*(1+0.2(2m−1))`（mask 由 L12 patch+detach cls） | z 即检索特征；主损失只监督 z | `cls.detach()` 只阻断条件读取；局部 CE 经 u 回主干 | A1 +0.15/−0.58；A2 0.00/−0.71（R5 +0.81）；**A1-CLS 比 UAD −1.50 R1，融合补回 +0.92**；u 单用 5.51/17.27；A1 mINP 0.713 最高 | 主损失只监督 z，c 与 u 共同适配使 c 退化；u 只补回 60% | 同权重 ON>OFF +0.25/+0.92（γ=0 精确复现 baseline）；mINP 升 | 保护 c（对 c 加 CE / branch dropout）；γ 自适应；u 只做推理期重排 |
| **RLE E1–E4** H042–H045 | 训练输入增强：RGB 先 MRLE 再 RRLE；NIR/TIR 独立通道 RRLE；E2 关 TIR、E3/E4 shared/safe | 无结构改动 | 无新参数/损失 | **−1.27…−1.43 / −3.9…−4.5**；RGB→RGB mAP −7.5…−7.9、R1 −10…−11；E2 仍降 | `convert('RGB')` 后灰度复制 3 通道，独立通道增强制造伪彩色；RGB 双重增强；身份细节擦除+欠拟合 | 无 | 先只关 RandomErasing；单变量 RGB MRLE；通道一致增强 |
| **A-V1** (WAV1) | MP 原型分类（7 方向）+ Pair(6 向)/Mixed(3 源) 对称对比（0.3uni+0.7set）；相机 K-group 重排+困难交换；融合 0.1MP+0.075Pair+0.025Mixed | 损失直接作用 pre-BN CLS；原型 bank 为 buffer | 无新参；无梯度预算；新增标量占比 11%→34.5% | **−0.42/−2.06**；472 新失败/340 修复；463/472 新失败 max-neg 上升 | 候选仅 15 个负身份（3%）；分母含正例（正例占 softmax 质量 36–44%）；MP 无相机掩码且主导 | 工程（step_ok 门控、manifest）；困难参考冻结思路 | 全局负例队列+末期衰减；正负分离 logsumexp |
| **B-B** (WBV1) | 替换 WRT 的 query-risk pairwise ranking：正例预算 50% 均匀+50% 给最容易正例；负身份/图像各 50% 困难加权；固定权重 85.807 | 作用 pre-BN CLS | 权重由 detach 相似度算；无新参；固定权重全程占标量 ~40% | **−0.15/−2.84**；batch 内 top_error 94.6%→0.75%、gap −0.12→+0.32；α 末期 0.012 | 候选 15 负身份（3%）；batch 目标解决但全库不可迁移；softplus 对 gap>0.15 饱和 | 诊断指标（top_error/gap/budget） | 重标定+末期衰减；加全局队列；只在 near-tie 上作用 |
| **WCV1** (WCV1) | mixed-gallery rank：gather 768 候选/63 负身份、负例 logsumexp、正例模态组均值、候选 detach、feature 级梯度预算 ρ=0.10 | 作用 pre-BN CLS（显式捕获 raw CLS） | 无新参；梯度比恒 0.10 但 cos(aux,base) 仅 0.30–0.33；不衰减 | **+0.09/−1.33**；mINP 0.724 vs 0.660；73.5% 新失败为纯负例变强；batch 内可分离/全库错 4.95%→2.45% | 候选 12.6% 身份、危险图像 1.0%；正例吸引主导（pos/neg 梯度范数 1.88、cos 0.86/0.23）；不衰减 | gather/dedup/预算探针/coverage 诊断 | 末期把 ρ 衰减到 0；全局图像级负例队列；对称梯度 |
| **WDV1-A0** | 从 120 checkpoint 全量加载（含 BN/分类器），constant LR 6.4e-5（=cosine floor 0.002×0.032）、30ep、TTA flip | 同一 pre-BN CLS，损失不变 | 全参 SGD+WD1e-4；无新参 | **11.03/29.51/46.42/54.47**；mAP 每次评估恒 11.03；R1 29.68→29.51；T→T 20.66→20.89 | ∫LR=0.0019，仅为基线尾段（ep90–120）的 1/30；训练目标已饱和（CE 0.10、acc 0.98）；指标分辨率 1query=0.0156pp | checkpoint 复现（loss 与 120ep 一致）；TTA flip ≤0.05pp（中性）；mAP 稳定 vs R1 噪声的对照 | 用 1e-3 量级 warm restart 重做并报告 ∫LR |

---

## 3. 分组深挖

### 3.1 多层/局部证据（HiHR、ViSA、PAT、MulConv、ReID5o、KGL adapter）

**共同正向路径**：局部/多层 token 经 attention 或残差进入后续层，最终仍由同一个 pre-BN CLS 输出。**没有一个是“旁支不参与推理”**——HiHR/ViSA 的新分量直接进入检索向量，PAT 的 part 只经 patch→CLS 间接，KGL adapter 直接改 CLS。

**共同梯度事实**：除 ViSA 的 shared/local detach 外，新分支都收到主损失；因此“模块只受弱辅助监督”是错误描述（HiHR 的 F 三分量都收 CE/WRT/ProCA/GPD，`visual_hihr.py:96-105`）。

**指标同构**：HiHR 与 ViSA 的九宫格都表现为 **9/9 mAP 升 + 三个同模态方向 R1 降 + 同模态 mINP 升**：

```
HiHR ΔmAP 全 ≥ +0.17；ΔR1: RR −1.54, NN −0.70, TT −1.55；ΔmINP: +0.53/+0.32/+0.36
ViSA ΔmAP 全 ≥ +0.18；ΔR1: RR −1.07, NN −0.37, TT −0.05；ΔmINP: +0.27/+0.26/+0.17
```

两者的模块与监督差异很大，但最终余弦都可精确分解为固定加权和：`cos = 0.8cos0 + 0.2cos1`（HiHR 为 0.8/0.133/0.067）。固定权重在训练与推理全程强制分配距离预算，无法在“局部不可靠的样本”上自动降权。

**为什么强模态伤 R1、弱模态有时升**：同模态 gallery 里的同身份近邻是最强竞争者；强模态 CLS 已含足量细粒度信息，额外分量只增加分数离散度，压缩 top-1 margin（RR R1：HiHR −1.54、ViSA −1.07、PAT −2.34）。弱模态（TIR query）CLS 本身就弱，额外证据能抬高首位：HiHR TIR→RGB R1 **+1.41**、T→N 0.00、N→T +1.12；PAT T→N +0.84、N→T +0.75。这与“回归均值”一致：模块把强方向拉低、把弱方向拉高，净效应取决于同模态 R1 权重。

**PAT 的特殊失败**：`vit_pytorch.py:339-384` 的 mask 让 part 只连自己区域 patch；`make_model.py:107-114` 推理只取 `x[:,0]`。part→patch→下一层 CLS 的路径存在，但 L12 的 part 对同层 CLS 无直接路径。E1b 的 `random_part_roi` 在 eval/内存初始化仍生效（`:422-425` 没有 `self.training` 门控），是确定的训练-测试不一致。E1a 的 CSL logits≈992、loss≈0（`supervision_summary.json`：p1_csl_loss −5.2e-8），实际起作用的是 PSD，而 PSD 邻居 92.7% 同相机。

### 3.2 跨层/来源融合（SFTC、ICRE、HiHR）

- SFTC：local K/V 与原 K/V **共用一个 softmax**（`vit_pytorch.py:129-135`），bias=−2 只把初始 local 质量压到 11.8%，不是恒等启动。OFF 时 softmax 从 257 重新归一化到 129 key，所以 **OFF 也不是 baseline**。分解：`−1.00 = +0.92（ON−OFF） + (−1.92)（OFF−baseline）`，模块只回收联合训练代价的约一半/三分之一。
- ICRE：全 patch 低秩混合是“平滑算子”而非小残差（gamma 0.436、残差 25–28%、router 仍 50/50），且对 L1–6 有直接梯度旁路；校准放大 out_proj 90× 同时放大分支步长。这是全历史最严重的单模块退化之一（−2.73/−6.90）。
- HiHR：跨层读出本身受主干全部四项损失监督，梯度路径干净；问题在“固定 20%”与“Stage2 重 warmup”，不在“读不出跨层信息”。

### 3.3 模态适配（KGL、ReID5o、SCSR、TAUR、WQVD A2）

- KGL：唯一四项同向（+0.15/+0.21/+0.42/+0.03），但幅度落在重复噪声内，且 TIR→RGB R1 −0.52、TIR→TIR mINP −0.03——早期模态分离会把 TIR 的模态风格而非身份放大。
- ReID5o：最后两层 FFN 的 rank-4 模态 LoRA，TIR→TIR +0.08/+0.14，但 NIR→RGB R1 −0.75；全参微调下专家是低秩偏置，与 ProCA/GPD 的共享身份空间张力明显。
- SCSR：v3 是“零初始化+小梯度+WD”形成的数值 dead（残差/patch 3e-8）；v4.1 避免了 dead（1.7% RMS），但作用面只有单层单模态，ON/OFF 差 0.0001pp。
- TAUR：唯一正确对齐“评测排除同相机”的新损失，但梯度只有 UAD 对 adapter 的 1.5–2.1%、残差 ≤10%，且同一次实验改了采样器与 WRT_STABLE，无 matched control。TIR→TIR R1 +0.33 与 R5/R10 +0.85/+0.53 不能归因于 rank loss。
- WQVD A2：有界乘性通道门控（scale∈[0.8,1.2]）只重加权已有通道，mAP 持平、R1 −0.71、R5 +0.81：这是“只改分数分布、不新增信息”的典型结果。

### 3.4 排序/关系监督（PAT CSL/PSD、KGL HNSM、TAUR rank、A-V1、B-B、WCV1）

这是本轮最重要的机制族，因为最新三组都在这里失败。

**候选覆盖（[代码事实]）**：

| 方法 | 每 Query 负身份 | 覆盖率 | 危险身份出现率 | 危险图像出现率 |
|---|---:|---:|---:|---:|
| A-V1（本卡） | 15 | 3.0% | — | — |
| B-B（本卡） | 15 | 3.0% | — | — |
| WCV1（4 卡 gather） | 63 | 12.6% | 11.9% | 1.0% |

评测的最大负例来自 500 身份、≈87k gallery。**89–99% 的 step 根本看不到真正的 top-1 竞争者**（WCV1 诊断 `coverage_summary.json`）。

**目标几何（[代码事实]）**：
- A-V1：`uni = logsumexp(valid) − mean(s_pos)`，`set = logsumexp(valid) − logsumexp(pos)`，分母含正例；τ=0.1 下正例吸收约 36–44% 的 softmax 质量，负例压制被稀释（`av1.py:84-87`）。
- B-B：`a` 的 50% 预算给 **最容易正例**（最大相似度，`bb_ranking_loss.py:56,64-66`），只在“最安全正例都输”时救 top-1；负身份 50% 困难加权，末轮最危险负身份拿 19.4% 身份预算；softplus 在 gap>0.15 后饱和。
- WCV1：负例 logsumexp（干净）、正例模态组均值、**候选 detach**（只 anchor 端回传）、ρ=0.10 只约束范数不约束方向（cos 0.30–0.33）、不衰减。

**后果（[实验支持]）**：
- A-V1：472 个 R1 新失败 / 340 修复；463/472（98.1%）新失败伴随 max-neg 上升；全体负对均值反而下降（同模态 0.0211→−0.0019）但 P99 上升（0.3525→0.3675）。→ 不是“全体负样本被抬高”，而是**尾部/最危险负例被抬高**。
- B-B：batch 内 top-1 已解决（top_error 94.6%→0.75%），全库 R1 仍 −2.84；mAP 几乎持平（−0.15）。教科书式“候选集不可迁移”。
- WCV1：mAP +0.09、mINP +0.064、R1 −1.33；324 新失败中 238（73.5%）纯负例变强、仅 13（4.0%）纯正例变弱；batch 内可分离但全库仍错的比例 4.95%→2.45%。**报告 `report.md:189` 把“正例吸引主导梯度”误写成“失败以正例变弱为主”，与自身 `ranking/summary.json` 矛盾**——后续不应按该结论优先改正例监督。

**[机制假设] 为什么最危险负例会被抬高**：可见负身份被显式推开后，损失对**未见身份**没有任何约束；在 768 维单位球上，降低可见负例相似度最容易的方向往往是与“平台/相机/外观流形”共享的方向，这会同时抬高对大量未见候选的相似度。正例吸引（WC 的 cos 0.86）进一步把簇压紧（mINP↑），但 max over 500 身份的负例增长快于 max over 正例，top-1 margin 收缩。A-V1 的“均值降、P99 升”是这一机制的直接签名。

### 3.5 PEFT/容量（ViT-L、PEFT、ReID5o、KGL）

- 全参数微调下，额外 adapter/专家是**冗余**：DPA −0.09/−0.32、TCA −0.39/−2.11、ReID5o −0.05/−0.17、KGL +0.15/+0.21（噪声内）。PEFT 的最后一层 DPA 是确定空路径（L12 patch 残差无消费者、无 patch 损失→梯度恒零）。
- 真正扩大容量（ViT-L）得 +0.42 mAP/−0.50 R1，但 batch/LR 同时变；其 NIR→RGB +1.14/+1.22/+0.33 是全历史唯一“mAP/R1/mINP 三项同升”的方向。
- 结论：容量不是主要瓶颈；**“加容量”与“加约束”必须分开验证**。

### 3.6 输入增强（RLE、SPM）

RLE 的根因是实现级：`datasets/bases.py:8-16` 把所有图 `convert('RGB')`，NIR/TIR 是三通道复制灰度图；独立通道 RRLE 给三通道乘不同因子→伪彩色，等于制造捷径。E2 关 TIR 仍降 1.27/4.23，说明 RGB 的 MRLE+RRLE 双重增强是主因。SPM 的根因是几何：14 个区域 GeM（负值 clamp）与未同尺度处理的 CLS 直接拼接，10752 维的分量范数隐式决定权重。

### 3.7 原型/中心（内建 GPD/ProCA、A-V1 MP、HiHR HHL）

- GPD 是 UAD 的主要增益来源（论文 Table 8：单独 +2.0 mAP/+4.8 R1），因此任何新增项都在扰动一个已经很强的全局原型几何。
- A-V1 的 MP 是新增项里唯一有 500 身份全局覆盖的（占新增标量 58.6%），但**无相机掩码**，优化的是“全相机混合的身份原型”，与评测的跨相机图像排序不一致。
- HiHR HHL 开/关轨迹几乎重合（mAP/R1 完全相同），说明双曲层次监督不是主因；两阶段/描述子才是。

### 3.8 采样（TAUR、A-V1、PKM）

- TAUR `CAMERA_PRIORITY=True`→`CameraPKMSampler`：同身份 4 实例优先跨相机，117 步/epoch（基线 114）；改变了 WRT/ProCA/GPD 的输入分布，**没有 matched control**。这是 TAUR 归因的最大混淆。
- A-V1 的相机 K-group 重排实际几乎无效（cameras_before≈2.015→after≈2.031），却改变 ProCA 组内中心；困难交换每轮改 35% 组。多因素同时变，无法归因。
- PKM 对少样本模态重复采样（`sampler.py:109-120`）；1000 基本平衡，2337 的 TIR 重复率更高，1000 的 TIR 结论不能直接外推。

---

## 4. 跨实验规律

### 4.1 规律一：mAP 可恢复，R1 是脆弱的路径相关量

| 证据 | 数值 | 说明 |
|---|---|---|
| HiHR Stage1→Stage2 切换 | 10.72/27.99 → 9.83/26.76 | 重 warmup 立即 −0.89/−1.23 |
| 之后 | ep40 mAP 11.05（同预算超基线），R1 28.17；ep60 11.44/28.57 | mAP 恢复并超过，R1 始终未回到 29.71 |
| HiHR 九宫格 | 9/9 mAP 升；同模态 R1 全降、mINP 全升 | 平均排序改善、头部 margin 收缩 |
| A0（≈0 LR） | mAP 6 次评估恒 11.03；R1 29.68→29.51 | mAP 是 6405 个 AP 的平均（稳），R1 是阈值统计（漂） |
| PEFT TCA / WQVD A2 / ReID5o | mAP −0.39/0.00/−0.05；R1 −2.11/−0.71/−0.17 | 小改动的 R1 代价不成比例 |

**这是全历史最稳定的规律**：任何改变优化路径的干预（新损失、新模块、LR 重启、继续训练）都先损失 R1；mAP 可以恢复，R1 往往不能。

### 4.2 规律二：mAP↑R1↓ 是 max 统计 vs 平均统计的通用响应

R1 由 `max_pos > max_neg` 决定；mAP 由全体正例的排序决定。对 28k 个负样本取 max 是极值统计：任何增加分数离散度、但不专门抬高 top-1 margin 的扰动，都会把 max-neg 抬得比平均负例多。证据：
- A-V1：负对均值 −0.02，P99 +0.015，每 query max-neg +0.059（新失败组）。
- 固定 20% 拼接（HiHR/ViSA）在数学上就是把一个新分量的方差加进 cosine。
- 离线 raw 欧氏 vs 余弦：R1 相同但 top-1 仅 65.8% 一致 → 大量 query 的 top-1 是近 tie，任何扰动都会翻转其中一部分。

**因此“mAP 升”不能作为机制有效的证据；“R1 不掉”才是。**

### 4.3 规律三：另一簇“双降”来自路径/输入破坏，不是监督对象

ICRE（25–28% patch 重写）、SFTC（257-key 共享 softmax）、RLE（伪彩色）、M-SpecGene（pos 轴错配）、MulConv/SPM（区域拼接几何）——它们的 CE 在早期也高于基线（SFTC ep10 CE 2.64 vs 2.15；ICRE 2.41；RLE 3.6–3.8），是共同训练代价/欠拟合，而非“后期过拟合”。日志事实：所有方法最后 30 epoch 的绝对 mAP/R1 斜率仍为正，没有“损失降、指标绝对降”的组合。

### 4.4 规律四：排序损失的可迁移性由候选覆盖决定

B-B 在 batch 内把 top-1 错误从 94.6% 压到 0.75%，全库 R1 仍 −2.84；WCV1 把“batch 内可分离但全库错”从 4.95% 降到 2.45%，全库 R1 仍 −1.33。只要训练候选只有 3%–13% 的负身份，**batch 内的成功不能外推**。

### 4.5 规律五：名义权重 ≠ 梯度份额

| 方法 | 名义权重 | 实际 |
|---|---|---|
| TAUR rank | λ=0.1 | adapter 上仅为 UAD 的 1.5–2.1%，主干 0 |
| B-B | 85.807（一次性标定） | 标量占比 38.8%→40.8%（bb_raw 降 46× 仍不衰减） |
| WCV1 | ρ=0.10 | 梯度比恒 0.10，但标量占比 44.5%，cos 仅 0.30 |
| SFTC | — | clip 频率 1.0，新分支与主干共享预算 |
| H018（只改 LR 0.032→0.008） | — | **−2.67/−6.57，大于几乎所有模块** |

### 4.6 规律六：单 seed 无法判定 <0.5pp

H029 同配方重跑 −0.09/−0.77；ViSA seed1234 vs seed2025 差 0.24/1.13。按此尺度：KGL +0.15/+0.21、PAT single +0.31/+0.37、SCSR +0.03/+0.05、ReID5o −0.05/−0.17、TAUR −0.45/−0.31、ViSA −0.34、PAT DDP −0.92、WCV1 −1.33、HiHR −1.14 全部落在 1–2 倍噪声内；**只有 A-V1 (−2.06)、B-B (−2.84)、ICRE/SFTC/RLE/M-SpecGene 明确超出**。
方向一致性可以部分弥补幅度不足（7 个 mAP↑R1↓ 案例同向），但单个方法的 ±0.3pp 不能当结论。

### 4.7 规律七：训练/评测协议错配是公共背景

- 训练 WRT 把同相机异身份当负样本、评测删除；同相机同身份正样本被拉近、评测不考。
- 总体 R1 是九对并集（29.71 vs 19.12），单对 R1 的总体杠杆只有 ≈0.28pp；总体 mAP 近似九对 1/9 加权。
- 因此“改一个 pair”几乎不可能让总体 mAP +1pp（需单对 +9pp），但可能让总体 R1 +1pp（需单对 +3.2–3.8pp）。

---

## 5. 可保留机制清单（按证据强度排序）

1. **pre-BN CLS 单一描述子 + UAD 四项损失**（强）。唯一复现论文（11.02/29.71 vs 11.0/29.5）；A0 从 checkpoint 继续训练时 loss/指标与 120ep 完全一致（CE 0.099 vs 0.11、mAP 11.03 vs 11.02），证明加载与评测链路正确。
2. **"mAP / R1 / mINP 三联 + 九宫格 + 同模态/跨模态拆分"的诊断协议**（强）。它把“平均排序改善、头部收缩”与“路径损坏”区分开；没有它，HiHR/ViSA 会被误判为成功。
3. **同权重 ON/OFF + 与独立 baseline 的三方报告**（强）。SFTC 的 ON−OFF=+0.92/+1.36 与 OFF−baseline=−1.92/−4.48 是唯一能定量“模块收益 vs 联合训练代价”的格式。
4. **WQVD A1 的局部证据在推理期真实存在**（中强）。γ=0 精确复现 baseline（10.14/27.81），γ=0.05 得 +0.25/+0.92；u 单用仅 5.51/17.27。→ 局部 patch 信息是互补的，问题是联合训练把 c 弄坏（A1-CLS −1.50 R1）。
5. **HiHR Stage1 的同预算双升 +0.82/+1.00**（中）。历史唯一同 60ep 双升；但含 TMF、CosineClassifier、无 ProCA/GPD、不同 WRT 尺度四重混杂，不能直接归因，必须做 2×2 拆解。
6. **平台轴（AG）的正向信号**（中）。HiHR A→G mAP +0.56、G→A +0.75/+1.71；ViSA G→A +0.41/+0.78；而平台差异（场景质心 cos 0.39–0.45）远大于模态差异（0.93–0.95）。→ 局部/层次分量在“外观差距大”的方向有帮助。
7. **KGL adapter 的四项同向小升**（弱）。+0.15/+0.21/+0.42/+0.03，方向一致且 TIR 方向退化；落在噪声内，需 2–3 seed。
8. **ViT-L 的 NIR→RGB 三项同升**（弱-中）。+1.14/+1.22/+0.33，但 batch/LR 混杂。
9. **TAUR 的 R5/R10 +0.85/+0.53 与同相机对齐协议**（弱）。不能归因于 rank（梯度 2%），但采样器/WRT_STABLE 是廉价可复用件，需要 matched control。
10. **工程实践**（强，作为实践而非方法）：A-V1 的 `step_ok` 门控 Bank、RNG/manifest/hash、采样 ledger；WCV1 的 gather 形状检查 + sample-key 去重 + finite 守护 + feature 级梯度预算探针 + `batch_separated_full_wrong_rate`；SCSR v4 的“非零初始化 + WD0 + residual RMS 监控”避免 dead；HiHR 的“WRT 需按梯度而非距离中位数标定”。

**已证不可保留**：固定 20% 拼接、新增 pairwise/listwise ranking 损失（在 batch 候选上）、未归一化点积+低温度、零初始化+WD 的小模块、直接改 CLS 的 per-layer adapter、全 param 微调下的额外 adapter/专家、对灰度复制图做独立通道增强、把局部头作为训练-only 私有头。

---

## 6. 需要进一步验证的假设（可检验、按优先级）

1. **R1 是尖峰而非平台（最高优先）**。A0 只用了基线尾段 1/30 的 ∫LR，不能证明饱和。用同一 checkpoint、warm restart LR ∈ {5e-4, 1e-3, 2e-3} 各 20–30ep，报告 ∫LR 与逐 epoch R1。预测：若 R1 单调下降而 mAP 可升，则“任何再优化都伤 R1”成立，整个历史的 mAP↑R1↓ 有统一解释；若 R1 上升，则“延长训练”仍是未用杠杆。
2. **压缩/离散度假说**。对 H014、HiHR、ViSA、WQVD-A1、WCV1 计算：每 query top-1 margin 分布、与 top-1 差距 <δ 的候选数、身份质心两两余弦、负相似度 P99。预测：mAP↑R1↓ 的方法应表现为质心余弦↑、近 tie 密度↑、P99↑。
3. **候选覆盖是排序损失的必要条件**。对任何新排序损失，训练期每 N 步抽样 query 计算全库 max-neg；若 batch max-neg 降而全库 max-neg 升，直接否证该损失。可作为 120ep 前的 gate。
4. **同相机对齐是否有效**。WDV1 代码已支持 `WRT_CAMERA` / `WRT_CAM_NEG_WEIGHT`（A0 均为关）。跑 `WRT_CAMERA=True, cam_neg_weight∈{0,0.25,1}` 三组，测跨相机 R1。预测：若训练/评测错配重要，跨相机 R1 改善；否则无变化。
5. **保护主描述子能否保住局部收益**。WQVD A1 加 branch dropout（每步以 p 置 γ=0）或对 c 加 CE，目标是让 A1-CLS 回到 10.14/27.81 同时保留 fusion +0.92。预测：若退化来自 c/u 共同适配，则可同时拿到 +0.9 R1。
6. **HiHR Stage1 的 2×2 拆解**：{TMF on/off}×{CosineClassifier on/off}，均 120ep、均无 ProCA/GPD 早期。预测：若 +0.82/+1.00 主要来自分类器/单位化几何，则“UAD + cosine classifier”这一 3 行改动就值得单独验证。
7. **平台门控融合**。用 HiHR/ViSA 的已训练权重离线扫 η∈{0,0.01,0.05,0.1,0.15,0.2}，并分 AG 两方向；若 R1 随 η 单调降而 mAP 在 η≈0.05–0.1 达峰，则“样本自适应小权重门控”是有根据的修正，而不是再赌固定权重。
8. **PSD-only**（不要 part token）：把 PAT 的软邻居目标直接加在最终 CLS 上，邻居池改为跨相机。预测：若 PSD 是 PAT 的有用部分，单加 PSD 应优于 E1a；若否，说明“局部 token”本身无贡献。
9. **TAUR 的 matched control**：同 CameraPKMSampler + 原 WRT、无 rank、无 adapter，才能判断 R5/R10 +0.85/+0.53 来自采样器还是排序项。
10. **A0 的 TTA flip 已实测中性**（≤0.05pp），应从“未尝试杠杆”清单中移除；同理，per-modality mean/z-score 后处理已实测无效/有害（11.03/29.65、10.59/26.84），不应再作为方向。

---

## 7. 证据路径

**代码（本机）**
- 基线：`WDV1_CC/WHU/model/make_model.py:67-80`（forward）、`:70-72`（finetune 全量加载）；`model/backbones/vit_pytorch.py:317-341`（CLS/final LN）；`processor/processor.py:279-309`（四项损失/backward/clip/bank）、`:443-445`（TTA flip）；`loss/wrt_loss.py:39-81`（WRT，仅 PID 分正负）；`loss/make_loss.py:30-37`；`utils/metrics.py:164`（同相机排除）、`:191-192`（mINP）、`:462-465`（L2）；`datasets/sampler.py:94-120`（少数模态重复）。
- WDV1-A0：`configs/LRA_UAD_A0.yml`（LR 6.4e-5、constant、TTA_FLIP True）、`solver/make_optimizer.py`（SGD+WD）、`solver/scheduler_factory.py:7-9`、`config/defaults.py:184-187`（LR_MIN_FACTOR=0.002）。
- HiHR：`WHU_Base/WHU_HiHR/WHU/model/visual_hihr.py:46-59`（TMF）、`:96-105`（F 拼接）、`loss/visual_hihr.py:37`（parent detach）、`processor/processor.py:232-277`。
- ViSA：`WHU_Base/WHU_ViSA+MD+MP/WHU/model/visa_adapt.py:71-154`；`processor/processor.py:236-255`。
- SFTC：`WHU_Base/WHU_Fusion+MSCMNet+SDS+CoMer_SFTC/WHU/model/backbones/vit_pytorch.py:122-142`；`model/backbones/sftc_local.py:67-173`。
- ICRE：审计镜像 `.../WHU_ICRE/model/backbones/clee.py:41-68`；`vit_pytorch.py:346-358`。
- TAUR：`TAUR/WHU/model/backbones/thermal_adapter.py:21-34`；`loss/taur_rank_loss.py:16-66`；`processor/taur_grad.py`；`datasets/sampler_camera.py`。
- KGL：`WK/WHU/model/backbones/vit_pytorch.py:241-257,380-395`；`model/make_model.py:107-184`。
- PAT：`WP/WHU/model/backbones/vit_pytorch.py:339-384,422-425`；`model/make_model.py:107-114`；`processor/processor.py:160-206`。
- PEFT：`WB/WHU/model/backbones/vit_pytorch.py:109-146,177-241`。
- ReID5o：`WO/WHU/model/backbones/modality_experts.py:13-50`；`vit_pytorch.py:321-329`。
- WQVD：`WQVD/WHU/model/modules/a1_local.py:28-55`；`a2_modulation.py:37-54`；`model/make_model.py:106-145`；`processor/processor.py:232-235`。
- RLE：`WR/WHU/datasets/rle.py:14-120`；`datasets/bases.py:8-16`；`datasets/make_dataloader.py:41-60`。
- A-V1/B-B/WCV1：`WAV1/WHU/loss/av1.py:47-115`；`WBV1/WHU/loss/bb_ranking_loss.py:32-116`；`WCV1/WHU/loss/mixed_rank.py:39-142`；`WCV1/WHU/processor/processor.py:221-284`。

**日志/诊断**
- A0：`WDV1_CC/logs/train_log_20261029_1556.txt`（eval 块 L169-173、L230-234、L291-295、L352-356、L413-417、L474-478；末轮 loss L440-441；GPD warm start L112）。
- 轨迹：`research/phase1_logs/log_trajectory_report.md` §3–§6（H014 逐 epoch、TAUR ep10 +1.44/+4.81、PAT R1 ep70 转负、HiHR 同预算表 §4.3）；`epoch_metrics.csv`。
- 指标：`05_Results.csv`（R0159 H014、R0160-0168 九对、R0169-0170 AG；各方法 train_final 行）。
- 审计：`research/phase1_history/audit_group1_hihr_visa_sftc_icre_taur.md`、`audit_group2_kgl_pat_wqvd_scsr_rle_peft_reid5o_mspec.md`、`audit_latest_av1_bb_wcv1.md`。
- 协议/离线：`research/phase1_protocol/protocol_target_analysis.md` §2、§4；`offline_calibration_all.json`（raw 11.02/29.71/0.66）。
- WQVD A1 诊断：`WQVD/logs/A1_vs_UAD_gamma_analysis/feature_comparison.csv`（经审计引用：UAD-CLS 10.1395/27.8064/0.6033；A1-CLS 10.0352/26.3076/0.7128；A1-Fusion 10.2882/27.2287/0.6902；A1-Local 5.5141/17.2678/0.2666）。
- PAT 诊断：`WP/WHU_D/analysis/retrieval/retrieval_summary.json`（482/423/净 −59）、`analysis/supervision/supervision_summary.json`（CSL≈0、PSD 96.4%/92.7%）。
- WC 诊断：`WCV1/logs/diagnosis_120/report.md`、`ranking/summary.json`、`coverage/coverage_summary.json`、`probes/gradient_pairs.csv`。
- SFTC 诊断：`SFTC/logs/SFTC_120_diagnostics/`（on/off/patch_only/cls_only）。
- 历史档案：`02_Previous_Experiments.md` 2.4/2.5（MulConv/SPM、ViT-L）、2.10（H029 重复）、其余各节。
