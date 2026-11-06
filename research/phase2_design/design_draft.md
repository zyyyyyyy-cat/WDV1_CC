# LRA-UAD 方案草稿（待红队审理）

> 目标：在 WHU-MARS-1000 / 1000GD / 2337 上，相对论文 UAD (Ours) 的 mAP、Rank-1、Rank-5、Rank-10（主）与 mINP（次）均取得 ≥ +1.0pp；至少不低于论文值。
> 基线代码：D:\CODE\Doing\WHU_base\WDV1_CC\WHU。
> 本文是 Phase 2 草稿，不是最终交付文档。

---

## 0. 一句话结论

历史 17 组方法、最新 A-V1/B-B/WCV1、以及本轮 6 类免训练后处理实验共同表明：**UAD 的瓶颈不在“再加一个损失/模块”，而在训练动力学与开放集度量几何**。因此本方案不再新增排序/对比损失，而是沿三条轴做低风险、可消融的改进：

1. **延长并稳定优化**：120→180 epoch 继续训练（或 200 epoch 从头训练）+ 分层学习率衰减（LLRD）+ EMA + model soup。
2. **修正训练-检索几何错配**：Camera-Aware WRT（同相机负样本降权）+ Normalized WRT（原始/归一化特征混合距离）。
3. **提升全局原型可靠性**：GPD 原型暖启动（RGB 类均值）+ 计数感知 EMA + 温度课程。
4. **推理期低风险增益**：水平翻转 TTA；可选置信度门控 QE（默认关闭）。

Plan B（若单模型仍不达标）：UAD 与已训练 CLIP-ReID 的特征/分数互补集成，或 CLIP-ReID 继续训练 + EMA/TTA。

---

## 1. 证据摘要（决定方案形态）

### 1.1 基线精确复现

离线用 `WHU_Diagnosis/WHU_1000/features_bb7907c940c6.npz` 精确复现 H014：

| 指标 | 值 |
|---|---:|
| mAP | 11.0234 |
| Rank-1 | 29.7112 |
| Rank-5 | 46.3232 |
| Rank-10 | 54.5511 |
| mINP | 0.6595 |

与 `logs/1000/1000_test_log_20260912_1037.txt` 完全一致，确认离线实验台可靠。

### 1.2 目标缺口（相对论文 +1.0pp）

| 数据集/协议 | mAP 缺口 | R1 缺口 | R5 缺口 | R10 缺口 |
|---|---:|---:|---:|---:|
| 1000 AS | +0.98 | +0.79 | +0.98 | +1.25 |
| 1000GD AS | +1.10 | +1.29 | 已达标 | 已达标 |
| 2337 AS | +1.00 | +0.97 | +1.00 | +0.81 |
| 1000 AG A→G | +1.04 | +1.27 | — | +0.78 |
| 1000 AG G→A | +1.00 | +1.00 | — | +0.80 |

九组合中最大缺口：NIR→NIR R1 +2.83、TIR→RGB R1 +2.64、NIR→TIR R1 +2.41、NIR→RGB R1 +1.99。TIR 相关五对合计占 1000 总体 mAP 权重 55.6%。

### 1.3 历史失败模式（17 组 + 最新 3 组）

- 没有任何方法在 mAP/R1/R5/R10 上同时超过论文；最好 mAP +0.42pp（HiHR/ViT-L），最好 R1 +0.26pp（SCSR V3），最好 R5 +0.85pp（TAUR），最好 R10 +0.53pp（TAUR）。
- 反复出现 **mAP↑ / R1↓**：HiHR、ViSA、ViT-L、PAT、WQVD A1、WCV1。
- 反复出现 **R5/R10↑ / mAP,R1↓**：TAUR。
- 最新 A-V1：mAP −0.42、R1 −2.06；472 个 Rank-1 新失败中 463 个（98.1%）伴随最危险负样本相似度上升。
- 最新 WCV1：mAP +0.09、R1 −1.33；精心实现的 mixed-gallery rank loss 仍失败。
- 最新 B-B：mAP −0.15、R1 −2.84；替换 WRT 的 query-risk ranking 失败。
- 结论：**在 UAD 的 CLS 上继续叠加排序/对比/模态原型损失，会系统性抬高最危险负样本，伤害 top-1。**

### 1.4 日志动力学

- 所有实验末期绝对指标仍在上升，没有“后期绝对下降”。
- 基线 90→120：mAP +0.29、R1 +0.97；尾段斜率 mAP +0.011、R1 +0.0325 pp/epoch。
- 失败方法多在 ep10–20 领先，随后饱和；基线继续改善并反超。
- 损失分量（CE/WRT/PCA/GPD）单调下降，没有后期爆炸/原型崩塌证据。
- 结论：**延长训练是最高确定性的未尝试杠杆。**

### 1.5 免训练后处理实验（本轮新增）

在精确复现的离线评测台上测试（全部基于同一 H014 特征）：

| 方法 | mAP | R1 | R5 | R10 | 结论 |
|---|---:|---:|---:|---:|---|
| 基线 | 11.02 | 29.71 | 46.32 | 54.55 | — |
| 逐模态分数 mean 校准 | 11.03 | 29.65 | 46.04 | 54.54 | 无效 |
| 逐模态 z-score | 10.59 | 26.84 | 43.34 | 51.80 | 严重退化 |
| 逐模态中心化 | 11.06 | 29.48 | 46.09 | 54.58 | 近无效 |
| 逐模态 ZCA 白化 | 4.07 | 20.98 | 34.47 | 41.94 | 灾难 |
| 模态 CORAL | 9.91 | 26.15 | 43.01 | 50.79 | 灾难 |
| LDA（shrink0.3） | 10.48 | 26.98 | 43.72 | 51.69 | 退化 |
| QE k=10, λ=1 | 14.24 | 29.96 | 38.94 | 45.17 | mAP 强、R5/R10 崩 |
| QE k=10, λ=0.1 | 11.88 | 29.93 | 43.93 | 52.15 | mAP +0.86、R5/R10 −2.4 |

结论：**任何全局特征变换都不能改善 UAD 特征；QE 只能作为 mAP 的可选增强，不能作为全指标方案。**

### 1.6 特征几何

- 平台差异 >> 模态差异：RGB-G↔RGB-A 质心余弦 0.388，RGB-G↔NIR-G 0.954，RGB-G↔TIR-G 0.927。
- 闭集分类已饱和：共享分类器训练准确率 94.9–95.4%，每模态分类器 97.5–98.4%，最近类中心 99.9%。**瓶颈是开放集度量，不是分类边界。**
- 特征范数几乎恒定（均值≈10，std≈0.38），raw 欧氏与 L2 余弦的 R1 相同，但 top-1 仅 65.8% 一致 → 顶部边界极脆弱。
- 最危险负样本相似度均值约 0.47–0.54，而最难正样本约 0.09–0.24；TIR 正样本最弱（RGB-G 0.204、NIR-G 0.141、TIR-G 0.088）。

### 1.7 CLIP 证据

CLIP-ReID + ProCA + GPD（WC 实验）：
- seed1234：mAP 12.01 / R1 29.18 / R5 46.18 / R10 54.24。
- seed42：12.10 / 29.37 / 45.51 / 54.21。
- seed2026：12.13 / 28.56 / 46.31 / 54.07。

CLIP 在跨模态/NIR→NIR 上大幅优于 UAD（例如 NIR→NIR mAP +1.23/R1 +2.25，TIR→RGB mAP +1.22/R1 +1.87），但 TIR→TIR 与 RGB→RGB R1 下降。**CLIP 是 mAP 强杠杆，且与 UAD 互补；但单模型 R1/R5/R10 未达 +1。**

---

## 2. 设计原则

1. **不新增排序/对比/混合 gallery 损失**；保留 UAD 的 CE + WRT + 0.01 ProCA + GPD 结构与权重。
2. **不改变推理特征维度**；仍是 768 维 pre-BN CLS，L2 归一化。
3. **所有改动必须沿现有四条损失的数据流/梯度流生效**；不引入“训练用、推理不用”的分支。
4. **低容量、可关闭、可消融**；每个组件有独立配置开关与预期指标。
5. **优先训练动力学与几何对齐**，而不是结构堆叠。
6. **分阶段验证**：先跑最便宜/最高确定性的 LHF+EMA+TTA，再逐步加入 CA-WRT/NWRT/GPD-WR。
7. **不以测试集挑选中间 epoch**；正式结果取配置末轮完整评测。

---

## 3. 方案 LRA-UAD

### 3.1 总览

```
输入三模态 tuple
  → shared ViT-B/16（原 UAD，不变）
  → pre-BN CLS f (768)
      ├─ CE：BNNeck → classifier（原 UAD）
      ├─ WRT：CA-WRT + NWRT（修改现有损失）
      ├─ ProCA：原实现（不变）
      └─ GPD：GPD-WR（暖启动 + 计数 EMA + 温度课程）
  → 推理：flip TTA（可选 QE）→ L2 → 欧氏
```

训练总损失：

```
L = L_CE + L_CA-WRT + 0.01 L_ProCA + L_GPD
```

不新增损失项；WRT 与 GPD 是现有项的内部改进。

### 3.2 组件 A：Long-Horizon Fine-tuning（LHF）

**目的**：利用基线 120 epoch 仍在上升的动力学，继续优化开放集几何。

**推荐实验 A1（首选，便宜）**：
- 从 `logs/UAD_1000_DDP/transformer_120.pth` 继续训练 60 epoch。
- DDP：`BASE_LR=0.004`（1/8 原 LR），cosine 到 `0.0001`，warmup 1 epoch。
- 单卡：`BASE_LR=0.001`，cosine 到 `0.000025`，warmup 1 epoch。
- 其他配置不变；`MAX_EPOCHS=60`，`CHECKPOINT_PERIOD=10`，`EVAL_PERIOD=10`。
- 目的：在低 LR 下继续降低 WRT/GPD，提升 R1/R5/R10。

**备选 A2（若 A1 饱和或下降）**：
- 从头训练 200 epoch，cosine + warmup 5；其他不变。
- 成本 1.67×；用于验证“更长 schedule”是否本身有效。

**实现**：
- `MODEL.PRETRAIN_CHOICE='finetune'`，`MODEL.FINETUNE_PATH=<checkpoint>`；`build_transformer` 加载完整 state dict（含 classifier/BNNeck）。
- 训练开始时 `START_EPOCH=0`，scheduler 按 60 epoch 重新 cosine；不恢复 optimizer。
- 保留原 120 epoch 结果作为对照。

**预期**：mAP +0.3–1.0、R1 +0.5–1.5；若 A1 在 20–30 epoch 后指标下降，停止并回退。

### 3.3 组件 B：Layer-wise LR Decay（LLRD）

**目的**：保护 ImageNet 预训练的低层特征，避免统一 LR 在后期破坏表示。

**参数组**（`decay=0.75`，可配 0.8）：
- `base.blocks.i`：`lr = base_lr * decay^(11-i)`。
- `base.patch_embed`：`lr = base_lr * decay^12`。
- `base.cls_token`、`base.pos_embed`：`lr = base_lr * 0.1`。
- `base.norm`：`lr = base_lr * 0.5`。
- `classifier`、`bottleneck`：`lr = base_lr`（bias 仍 ×2）。
- 其余参数：`base_lr`。
- Weight decay 保持 1e-4；bias 1e-4。

**数据流/梯度流**：只影响 optimizer 的 LR，不改变前向/反向；所有参数仍接收原损失梯度。

**风险**：早期层 LR 过低可能减慢适应；在 A1 低 LR 微调中风险较小。A2 从头训练时 decay 建议 0.8。

### 3.4 组件 C：EMA + Model Soup

**EMA**：
- 复制模型为 `ema_model`；每个 optimizer step 后：
  `ema_p = decay * ema_p + (1-decay) * p`，`decay=0.999`（可配 0.9995）。
- BN running stats：直接复制当前模型 buffers（或 EMA，推荐 EMA buffers）。
- 训练末期保存 `transformer_ema.pth`；测试脚本可加载 final 或 EMA。
- 默认报告 EMA；final 作为消融。

**Model Soup**：
- `CHECKPOINT_PERIOD=10`；在训练后 40% 的 epoch 保存。
- `tools/model_soup.py` 对最后 3–5 个 checkpoint 做等权平均（含 BN buffers）。
- 测试 soup；默认报告 soup 或 EMA 中预注册的一个，避免测试集择优。

**预期**：+0.2–0.5pp，主要改善 R1/R5/R10 的稳定性。

### 3.5 组件 D：Camera-Aware WRT（CA-WRT）

**动机**：评测排除所有同相机 gallery；当前 WRT 把同相机异身份当普通负样本、同相机同身份当普通正样本，训练目标与评测不一致。

**实现**：
- processor 构造 `camids_cat = torch.cat(camids, dim=0)`（顺序与 `[RGB;NIR;TIR]` 一致）。
- WRT 接收 `camids`，构造：
  - `same_cam = camids[:,None] == camids[None,:]`
  - `pos_mask = is_pos * (~same_cam + same_cam * w_pos)`
  - `neg_mask = is_neg * (~same_cam + same_cam * w_neg)`
  - 对角线置 0（排除自匹配）。
  - `w_pos=0.5`、`w_neg=0.25`（默认；可配 0/0.25/0.5/1）。
- 若某行 `pos_mask` 或 `neg_mask` 全零，回退到原 `is_pos`/`is_neg`（保证有梯度）。
- `softmax_weights` 仍按掩码归一化；距离与 SoftMarginLoss 不变。

**梯度流**：camids 是布尔掩码，不接收梯度；WRT 对特征的梯度经 `dist_ap`/`dist_an` 回到共享 ViT。**不新增参数。**

**预期**：+0.1–0.4pp，主要改善跨相机 R1；若退化，设 `w_neg=1,w_pos=1` 回退。

**消融**：`WRT_CAMERA_WEIGHTS=(1,1)` 为原版；`(0.5,0.25)` 为默认；`(0,0)` 为硬排除。

### 3.6 组件 E：Normalized WRT（NWRT）

**动机**：WRT 用未归一化特征欧氏距离，推理用 L2 余弦；raw 与 normalized 的 top-1 仅 65.8% 一致。由于特征范数稳定（std 0.38），完全替换风险不大，但仍采用混合。

**实现**：
- 对 WRT 输入 `inputs` 计算：
  - `u = normalize(inputs, dim=1)`
  - `s = inputs.norm(dim=1, keepdim=True).mean().detach()`（保持尺度）
  - `inputs_mix = (1-β) * inputs + β * (u * s)`
- `β` 从 0 线性升到 `β_target=0.5`，在前 5 epoch 完成；之后固定。
- 距离仍在 `inputs_mix` 上计算欧氏距离；SoftMarginLoss 不变。
- 训练/推理一致性提高，但不破坏原始特征尺度。

**梯度流**：β 路径可微；`s` detach；梯度经 raw 与 normalized 两支回到同一特征，等价于对特征方向和模长分别加权。

**预期**：+0.1–0.3pp，主要改善 R1；若退化，β_target=0 回退。

### 3.7 组件 F：GPD-WR（暖启动 + 计数 EMA + 温度课程）

**F1 暖启动**：
- 训练前（或 fine-tune 前）用 `train_loader_normal`（RGB 训练集，确定性 transform）跑一遍 `model.eval() + no_grad()`。
- 对每张 RGB 训练图取 pre-BN CLS，L2，按 PID 求均值，再 L2，写入 `PrototypeBank.prototypes`。
- 只初始化共享身份原型；不改变 GPD 公式、温度、权重。
- 目的：避免零初始化导致早期未初始化类 logits 为 0、负类集合不完整。
- 若没有 RGB loader，用 `train_loader` 的三模态 batch 累积；优先 RGB（最强模态）。
- 该 pass 不更新模型、不消耗训练 RNG（保存/恢复 RNG 状态）。

**F2 计数感知 EMA**：
- bank 增加 `counts[C]` buffer。
- 每次更新：`m_eff = min(m_max, max(m_min, 1/(count+1)))`，`m_min=0.2`、`m_max=1.0`；`P <- normalize((1-m_eff)P + m_eff*new)`；`count += 1`。
- 新身份第一次直接赋值（m_eff=1），之后逐渐降到 0.2。
- DDP：all_gather 后各 rank 同步更新 counts/prototypes。
- 目的：稀有身份（TIR/航拍）更快跟上，常见身份更稳定。

**F3 温度课程**：
- `tau(epoch) = 0.03 + (0.05-0.03) * max(0, 1 - epoch/20)`。
- 前 20 epoch 用较高温度（软目标），之后 0.03；避免早期原型不可靠时过度尖锐。
- 可配 `GPD_TAU_INIT=0.05`、`GPD_TAU_FINAL=0.03`、`GPD_TAU_WARM=20`。

**梯度流**：暖启动/EMA/counts 均为 no_grad/buffer；GPD 对当前特征的梯度仍经归一化特征回传；原型不接收梯度。

**预期**：+0.1–0.4pp；若退化，关闭 F1/F2/F3 逐项回退。

### 3.8 组件 G：Flip TTA（FTTA）

**实现**：
- `do_inference` 中每张图：
  - `f1 = model(img, mode=mode)`
  - `f2 = model(torch.flip(img, dims=[3]), mode=mode)`
  - `f = normalize(f1) + normalize(f2)`，再 L2。
- 评测器仍做 L2；距离不变。
- 只影响推理，不改变训练/权重。
- 默认开启；提供 `TEST.TTA_FLIP=False` 对照。

**预期**：+0.2–0.5pp；成本 2× 前向。

### 3.9 可选组件 H：置信度门控 QE（CG-QE）

- 离线证据：λ=0.1 时 mAP +0.86、R1 +0.22，但 R5/R10 −2.4。
- 设计：仅对 top1-top2 margin ≥ τ 的 query 做 QE（这些 query 更可能已正确，QE 用于改善 AP 而不破坏 top-1）；其余保持原特征。
- 参数：`k=10, α=3, λ=0.1, τ` 由训练集/验证集选择；默认关闭。
- 需要进一步离线验证（本轮未完成门控实验）。在确认能同时保住 R5/R10 前，不进入主方案。

### 3.10 可选组件 I：Per-Modality Classifier（PMC）

- `classifier` 改为 3 个 `Linear(768,C)`，按模态路由；共享 BNNeck。
- CE 用对应模态头；WRT/ProCA/GPD 不变。
- 训练/推理仍用 pre-BN CLS；推理不加载分类头。
- 预期 +0.1–0.5pp；闭集已饱和，收益可能有限；作为消融。

### 3.11 可选组件 J：TIR 输入标准化（MSIN）

- 对 TIR（可选 NIR）在 ToTensor 后、Normalize 前做 per-image 标准化：
  `x = (x - x.mean()) / (x.std() + 1e-6)`，再按原 mean/std 归一化。
- 目的：缓解热红外对比度/强度分布差异；不改变身份结构。
- 预期 +0.2–0.8pp on TIR pairs；风险低，作为 TIR 消融。

---

## 4. 训练/推理全流程

### 4.1 训练流程（A1 fine-tune）

1. 构建数据/模型（原 UAD）；加载 `transformer_120.pth` 完整权重。
2. 初始化 GPD bank：RGB 训练集 no-grad pass → 类均值。
3. 初始化 EMA 模型（复制当前权重）。
4. 构造 optimizer：LLRD 参数组；SGD momentum 0.9；WD 1e-4；bias ×2。
5. 60 epoch cosine：LR 0.004→0.0001（DDP），warmup 1；每 10 epoch 评估/保存。
6. 每个 iteration：
   - 前向 `[RGB;NIR;TIR]`，得到 `cls_score/global_feat/feat`。
   - CE（BNNeck）、CA-WRT+NWRT（pre-BN，β ramp）、ProCA 0.01、GPD（tau schedule）。
   - backward、unscale、clip 1.0、step。
   - 更新 GPD bank（count-aware EMA）；更新 EMA 模型。
7. 训练结束保存 final、EMA；可选 model soup。

### 4.2 推理流程

1. 加载 final/EMA/soup。
2. 每张图：原图 + 水平翻转，前向，L2 后平均，再 L2。
3. 可选 CG-QE（默认关闭）。
4. 原评测器：全 gallery、排除同相机、平方欧氏、mAP/CMC/mINP/九组合/AG。

---

## 5. 分阶段实验与判定

| 阶段 | 配置 | 目的 | 通过标准 | 失败回退 |
|---|---|---|---|---|
| S0 | 基线 H014 复测 + flip TTA | 验证 TTA | 任一主指标 ≥ +0.2pp 且无指标下降 | 关闭 TTA |
| S1 | A1 LHF（60 epoch, LR 0.004） | 验证延长训练 | 1000 mAP/R1 均 ≥ +0.5pp | 降 LR/缩 epoch；或 A2 |
| S2 | S1 + LLRD + EMA | 稳定与增益 | 1000 四项 ≥ +0.3pp vs S1 | 关闭 LLRD/EMA |
| S3 | S2 + CA-WRT + NWRT | 几何对齐 | R1 提升且 mAP 不降 | 权重回退到 1/β=0 |
| S4 | S3 + GPD-WR | 原型可靠性 | mAP/R1 ≥ +0.2pp | 逐项关闭 |
| S5 | S4 + soup + TTA | 推理稳定 | 四项 ≥ 目标 | 回退 |
| S6 | 可选 PMC/MSIN/CG-QE | 补 TIR/R5/R10 | 目标指标提升且无回退 | 关闭 |
| S7 | 2337 重训 / 1000GD 同权重 | 跨数据集 | 2337 四项 ≥ +1pp | 回退 1000 方案 |

**注意**：每阶段只改一组变量；用同一 seed 1234；报告 final/EMA/soup 的预注册选择；不以测试集挑中间 epoch。

---

## 6. 预期、风险与争议点

### 6.1 预期

- LHF + LLRD + EMA/soup：+0.5–1.5pp（最确定）。
- CA-WRT + NWRT：+0.1–0.5pp（方向正确，但幅度不确定）。
- GPD-WR：+0.1–0.4pp。
- FTTA：+0.2–0.5pp。
- 合计现实区间：+0.9–2.5pp；能否九组合全覆盖取决于 TIR/NIR→NIR。

### 6.2 风险

- **LHF 可能过拟合**：低 LR + EMA/soup 缓解；若 20–30 epoch 后下降，停止。
- **CA-WRT 可能有害**：同相机负样本也能教相机不变性；用软权重（0.25），并做消融。
- **NWRT 可能破坏 WRT**：β ramp + 0.5 混合；消融。
- **GPD 暖启动可能偏 RGB**：只初始化共享原型；count-aware EMA 让 TIR 后续修正；消融。
- **TIR→TIR 仍可能是瓶颈**：需要 MSIN/PMC 或两阶段 TIR adapter；本方案先验证核心。
- **+1pp 全指标非常激进**：若 S5 后仍差 0.3–0.5pp，进入 Plan B。

### 6.3 争议点

- 是否允许继续训练/更长 schedule？论文 120 epoch，但目标是对比论文数值，延长训练是合法的方法改进；需在报告中披露。
- 是否允许 TTA/QE？TTA 是推理增强，需单独报告；QE 默认关闭，避免协议争议。
- EMA/soup 是否算“模型改进”？属于训练/推理策略，需在方法中明确。
- 单模型 vs 双模型集成：Plan B 需明确标注“双模型互补集成”，不冒充单模型。

---

## 7. Plan B：CLIP 互补集成 / CLIP 继续训练

### 7.1 证据

CLIP-ReID + ProCA + GPD：mAP 12.01–12.13（已达 mAP 目标），R1 29.2–29.4，R5 46.2，R10 54.1–54.2；与 UAD 在不同模态对互补。

### 7.2 B1：特征/分数集成

- 在服务器用 WC/CLIP 的 `test_whu_mars.py` 提取 UAD 与 CLIP 的 query/gallery 特征（同一 manifest）。
- 离线评测：
  - 分数融合 `S = (1-α)S_UAD + αS_CLIP`（α∈[0,1]）。
  - 特征拼接 `[normalize(f_UAD), normalize(f_CLIP)]`，L2。
  - 逐查询置信度加权。
- 预期：mAP 12.5+、R1 30.5+、R5 47.3+、R10 55.8+；若成立，直接满足目标（但为双模型）。
- 需明确报告双模型成本与公平性。

### 7.3 B2：CLIP-ReID 继续训练 + EMA/TTA

- 对 CLIP-ReID 的 stage2 继续低 LR 训练 40–60 epoch；加 EMA/TTA。
- 目标：在 mAP 12.0 基础上补 R1/R5/R10 +1pp。
- 风险：CLIP 的两阶段/Adam 配置与 UAD 不同，需单独调 LR。

---

## 8. 代码修改地图（草案）

| 文件 | 修改 |
|---|---|
| `config/defaults.py` | 新增 LLRD/EMA/FINETUNE/WRT_CAM/NWRT/GPD_WR/TTA/QE 配置 |
| `configs/LRA_UAD.yml` | 新配置，默认关闭所有新组件，逐项开启 |
| `solver/make_optimizer.py` | LLRD 参数组 |
| `loss/wrt_loss.py` | CA-WRT 掩码 + NWRT 混合距离 |
| `loss/make_loss.py` | 传递 camids/epoch；WRT 配置 |
| `processor/processor.py` | camids_cat；GPD 暖启动/count EMA/tau；EMA 模型；TTA；保存 |
| `model/make_model.py` | finetune 加载；可选 PMC |
| `train.py` | finetune 入口；EMA 初始化；配置 |
| `tools/model_soup.py` | 权重平均 |
| `tools/ensemble_eval.py` | UAD+CLIP 特征/分数集成（Plan B） |
| `utils/metrics.py` | 可选 CG-QE；保持原评测 |

---

## 9. 待红队审理的问题

1. LHF 从 120 checkpoint 继续训练，是否比 200 epoch 从头训练更可靠？LR/epoch 如何选？
2. CA-WRT 的软权重是否应硬排除同相机负样本？是否会把相机信息从特征中移除，反而伤害跨相机？
3. NWRT 混合是否值得？raw/normalized top1 65.8% 一致，但 R1 相同，可能只是 near-tie 重排。
4. GPD 暖启动用 RGB 类均值是否会偏置 TIR？是否应三模态分别初始化再平均？
5. EMA/soup 在 ReID 小数据上是否稳定？decay 0.999 是否过高？
6. 是否有更高效的 TIR 组件（MSIN/PMC/两阶段 adapter）应进入核心？
7. Plan B 双模型集成是否应作为主线，而不是备选？
8. 目标 +1pp 九组合是否现实？若核心只达到 +0.5–0.8，如何调整方案？
