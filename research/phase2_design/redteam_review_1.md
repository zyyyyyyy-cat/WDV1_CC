# 红队评审 #1：LRA-UAD 方案

> 评审日期：2026-10-29  
> 评审对象：`D:\CODE\Doing\WHU_base\WDV1_CC\research\phase2_design\design_draft.md`  
> 评审方式：只读静态分析；未修改代码、配置或方案草稿。  
> 证据标签：**[代码事实]** 当前源码可直接核对；**[实验支持]** 日志/诊断/离线结果；**[机制假设]** 由代码与现象推出、未经单因素对照；**[协议事实]** 评测协议或目标口径。

---

## 0. 评审范围、口径与必须先解决的问题

### 0.1 已核对材料

- 方案草稿、`protocol_target_analysis.md`、`target_table.csv`；
- `log_trajectory_report.md`、两组历史代码审计；
- `literature_review_2025_2026.md`；
- A-V1 诊断与代码修改记录、B-B 实现说明、WCV1 实现说明与 120 epoch 诊断；
- 基线代码：`processor/processor.py`、`loss/wrt_loss.py`、`loss/make_loss.py`、`model/make_model.py`、`utils/metrics.py`、`datasets/sampler.py`、`sampler_ddp.py`、`make_dataloader.py`、`solver/*`、`config/defaults.py`、`configs/UAD.yml`。

### 0.2 目标口径存在歧义，必须先冻结

`target_table.csv` 实际包含四类目标：

- AS-ReID 12 个 cell（1000/1000GD/2337 × mAP/R1/R5/R10）；
- AG-ReID 6 个 cell（仅 1000，A→G/G→A × mAP/R1/R10）；
- VI-ReID 6 个 cell（1000，NIR→RGB/TIR→RGB × R1/mAP/mINP）；
- 3×3 模态对 27 个 cell（1000，R1/mAP/mINP）。

草稿把上述全部写进“缺口”分析，但“全面 +1.0pp”的验收集没有定义。这是评审中最重要、也最容易被忽视的问题：**如果“全面”指 12 个 AS cell，方案还有讨论空间；如果指 12+6+6+27 共 51 个 cell，历史上没有任何方法接近，目标在统计上几乎不可达。** [协议事实]

建议把验收集分成三层，并在实验前冻结：

| 层级 | cell | 说明 |
|---|---|---|
| Tier A（必须） | AS 12 个 | 1000 / 1000GD / 2337 × mAP/R1/R5/R10 |
| Tier B（应达） | AG 6 个 + VI 6 个 | AG 只在 1000；VI 的 mINP 是论文正式报告口径 |
| Tier C（尽量） | 3×3 27 个 | 其中 2 个 cell 已达标（RGB→RGB R1、1000GD R5/R10） |

另外三点口径必须先写清：

1. **1000/2337 的总体 AS mINP 没有论文目标**：论文只在 VI/3×3 报告 mINP，代码 `compute()` 总体分支也未取回 `mINP`。草稿的“mINP 次”只能对 VI/3×3 pair 成立，不能对 AS 总体声称“论文 +1”。[代码事实][协议事实]
2. **1000GD 的 R5/R10 只有 +0.14/+0.06pp 的余量**：草稿把 GD 的 R5/R10 写为“已达标”，但这是零余量；任何 R5/R10 的 0.1–0.2pp 回退都会使 GD 目标失败。GD 的 R5/R10 应作为**护栏指标**，不是已完成的免费指标。[实验支持]
3. **1000GD 的 daytime 未在代码中实现**：`whu_mars.py` 的过滤只有 `pid 1000–1999` 与 `camid<=5`，没有 day/night 字段；本地 GD 是否严格等于论文 GD 仍未核实。GD 目标存在协议偏差风险。[代码事实][协议事实]

### 0.3 一个必须先承认的统计事实

6405 个 query 时，Rank-1 的 1pp 只有约 **64 个净 query**。A-V1 的 Rank-1 转变是 472 个新失败 / 340 个新成功（净 −132 ≈ −2.06pp）；PAT DDP 是 482 / 423（净 −59 ≈ −0.92pp）。也就是说，**±1–2pp 完全落在近 tie 翻转噪声的量级内**。历史最好 R1 增益只有 +0.26pp（SCSR V3），而一次失败实验的近 tie 翻转可以有 2pp。单 seed、在测试集上选 epoch 的“+1pp”很可能是一次幸运翻转，不是可复现的方法收益。[实验支持/机制假设]

因此，草稿的单 seed、单一末轮、“不以测试集挑 epoch”原则是正确的，但还不足以支撑 +1pp 的结论；至少需要 2 个 seed、paired bootstrap 或 per-query 配对检验，并报告翻转数量。

---

## 1. 总体判断

### 1.1 核心诊断成立，但方案的重心放错了

草稿的核心判断——**不要继续叠加排序/对比/混合 gallery 损失；问题在训练动力学与开放集度量几何**——与 17 组历史方法、A-V1、B-B、WCV1 的证据一致，应保留。具体证据：

- 历史无任何方法在 mAP/R1/R5/R10 上同时超过论文；最好 mAP +0.42 伴随 R1 −1.14（HiHR），最好 R1 +0.26 伴随 mAP +0.03（SCSR V3）；TAUR 为 mAP −0.45/R1 −0.31，仅 R5/R10 +0.85/+0.53。[实验支持]
- A-V1 的 472 个 Rank-1 新失败中 463 个（98.1%）伴随最危险负样本相似度上升；全体 query 的最危险负样本相似度平均 +0.039，明显大于最佳正样本的 +0.029，margin 平均下降 0.010。WCV1 的 mixed-gallery rank loss 仍为 mAP +0.09/R1 −1.33。B-B 为 −0.15/−2.84。新增排序/负样本压力确实会抬高最危险负样本。[实验支持]
- 基线 90→120 仍在上升，没有后期绝对下降；尾段斜率 mAP +0.011、R1 +0.033 pp/epoch；损失分量单调下降。延长训练是尚未认真尝试的杠杆。[实验支持]

### 1.2 但“mAP 才是真正的约束”，而 LHF 是 R1/R10 杠杆

草稿把 LHF 的预期写成 mAP +0.3–1.0、R1 +0.5–1.5，但基线尾段的斜率是高度不对称的：

| 指标（1000） | 基线 90→120 | 尾段斜率（pp/epoch） | 目标缺口 | 按尾段斜率所需 epoch | 再训 60 epoch 的外推 |
|---|---:|---:|---:|---:|---:|
| mAP | 10.73→11.02（+0.29） | 0.0097 | +0.98 | ≈101 | +0.58 |
| Rank-1 | 28.74→29.71（+0.97） | 0.0323 | +0.79 | ≈24 | +1.94 |
| Rank-5 | 45.76→46.32（+0.56） | 0.0187 | +0.98 | ≈52 | +1.12 |
| Rank-10 | 53.26→54.55（+1.29） | 0.0430 | +1.25 | ≈29 | +2.58 |

[实验支持 + 机制假设：斜率为日志事实，外推是线性假设]

这张表给出三个结论：

1. **LHF 最可能先把 R1/R10 推过目标，而 mAP/R5 仍差 0.4–0.5pp。** 草稿的“LHF + LLRD + EMA/soup + CA-WRT + GPD + TTA 合计 +0.9–2.5pp”把各组件的最佳情形相加了；实际 mAP 的合计中心值更可能在 +0.5–0.8，而不是 +1.0。
2. **1000GD 的 mAP 缺口 +1.10 是最大的 AS mAP 缺口，不是“已经更容易”。** GD 的基线 mAP 更高（14.40），但目标也更高（15.50）；按同样的尾段斜率也要 100+ epoch。草稿把 GD 当成“只需 mAP/R1”的顺带验证，低估了 mAP 的绝对难度。
3. **2337 的 mAP +1.00 与 R5 +1.00 同样在尾段斜率下需要约 100/53 epoch。** 2337 独立重训，若沿用 120 epoch，LHF 不实施就几乎不可能达标。

**因此，方案的最大战略问题是：它把最高确定性的杠杆押在 LHF，但 LHF 主要改善 R1/R10；目标真正卡住的是 mAP/R5。** 要达到 mAP +1，需要改变特征几何（CLIP 融合、相机校准、或真正改善 AP/尾部覆盖的方法），而不是再训练 60 epoch。草稿的 §6.1 预期“合计现实区间 +0.9–2.5pp”对 mAP 偏乐观。

### 1.3 历史效应量与目标的关系

| 方法 | ΔmAP | ΔR1 | 备注 |
|---|---:|---:|---|
| HiHR H1 Stage2 | +0.42 | −1.14 | 历史最好 mAP |
| ViSA+MD+MP | +0.28 | −0.34 | 9 宫格 mAP 全升、R1 降 |
| TAUR v2 | −0.45 | −0.31 | R5 +0.85/R10 +0.53 |
| SCSR V3 | +0.03 | +0.05 | 历史最好 R1 附近 |
| KGL adapter | +0.15 | +0.21 | 四项小正，未协同 |
| A-V1 | −0.42 | −2.06 | 98.1% 新失败伴随危险负样本上升 |
| WCV1 | +0.09 | −1.33 | mINP 普遍上升 |
| B-B | −0.15 | −2.84 | 替换 WRT 失败 |
| CLIP-ReID+ProCA+GPD | +0.99～+1.11 | −0.34～−1.15 | mAP 强、R1 弱 |

[实验支持]

**没有任何单一方法同时取得 mAP 与 R1 的 +1pp；历史模块效应普遍在 ±0.5pp，最好的“双正”也只有 KGL adapter 的 +0.15/+0.21。** 目标超出历史模块效应一个量级。唯一已经显示出“mAP +1”的是 CLIP，但它的 R1 比 UAD 低 0.3–1.1pp。这说明目标最可能由“UAD 的 R1 + CLIP 的 mAP”这种互补组合实现，而不是由 UAD 上再叠一个模块实现。这是 Plan B 应升级为共同主线的核心依据。

### 1.4 概率判断（主观区间）

以下概率不是统计推导，而是基于历史效应量、尾段斜率、seed/近 tie 噪声、目标 cell 数的判断。

| 方案 | 1000 AS 四项全 +1 | Tier A（AS 12 cell）全 +1 | 含 AG/VI/3×3（Tier C）全 +1 |
|---|---:|---:|---:|
| 草稿原样 | 10–20% | 5–10% | <5% |
| 修订版单模型（LHF + 相机采样 + 免费校准 + soup/TTA + 2 seed） | 25–35% | 10–15% | <10% |
| Plan B 融合（1000 AS） | 35–45% | 15–25%（GD 同权重） | 10–15%（需 2337/AG 验证） |
| Plan B 融合 + CLIP 2337 重训 | 35–45% | 15–25% | 15–20% |

核心判断：**草稿原样达到 Tier A 的概率约 5–10%；先做免费校准和融合、再修 LHF，可把 Tier A 提到 10–15%；若只要求 1000 AS 四项，修订单模型 25–35%、融合 35–45%。** 如果“全面”包括 AG/VI/3×3，建议明确降级为“尽量达”，否则方案会在一个历史上无人接近的目标上空转。

---

## 2. 逐组件评审表

| 组件 | 机制（正向/反向） | 证据 | 主要风险 | 建议 | 优先级 |
|---|---|---|---|---|---|
| **LHF** | 延长同一 CE/WRT/ProCA/GPD 的优化；直接改变全局 L2 几何，不只是近 tie | 90→120 mAP +0.29/R1 +0.97；尾段斜率 0.0097/0.0323；无绝对下降；H018（LR 0.008）−2.67/−6.57 说明 LR 主导 | 重启 LR 是终段 LR 的 62 倍；finetune 加载路径未实现；GPD bank 冷启动；mAP 增长慢；单 seed/选 epoch | **保留但改**：先恒低 LR 30 epoch，再 200 epoch 从头；先修加载与 bank | P0 |
| **LLRD** | 只改 optimizer 分组 LR；前向/反向不变 | 本项目无 LLRD 证据；H018 说明 LR 是最大混杂；文献仅通用 | 低层适应变慢；cosine floor 是全局的；分组多；在 A1 低 LR 中几乎无意义；与 LHF 抢“低 LR”解释 | **删除/降级**：仅 A2 可选，decay 0.85–0.9，不作为核心 | P2 |
| **EMA/Soup** | 权重空间平均；改变全局几何；推理用 pre-BN CLS，BN buffer/分类头不影响检索 | Model Stock/Model Soup 通用；UAD 无证据；EMA 0.999 窗口约 9 epoch | 对单调上升轨迹会滞后；soup 可能不如 final；测试集选 final/EMA/soup；DDP eval 必须复制到各 rank | **保留为廉价 post-hoc**：0.999，last 3，预注册 final 为主 | P1 |
| **CA-WRT** | 对同相机正/负 pair 加权；只改损失掩码 | WRT 无 camera mask（`wrt_loss.py:46-47`）；评测排除全部同相机（`metrics.py:164`）；TAUR 的 camera sampler 与 WRT_STABLE 混杂 | **w_pos=0.5 会压低跨模态同相机正样本**；同相机负样本占比未知；softmax 重归一；本卡 192 图 batch | **改**：只压负样本（w_pos=1,w_neg=0.25）；或先做 camera sampler；先统计 | P2 |
| **NWRT** | 径向归一化混合；近 tie | 特征范数 10±0.38；raw/normalized R1 完全相同（29.71）；top-1 仅 65.8% 一致 | β=0.5 任意；改变 WRT softmax 尺度；无收益证据；与 CA-WRT 双重改 WRT | **删除**；若只想训练/推理一致，单测 β=1 | P3 |
| **GPD-WR F1** | bank 不在 checkpoint；暖启动身份原型 | [代码事实] 只存 `model.state_dict()`；bank 零初始化；`train_loader_normal` 仅 RGB | RGB-only 偏置；counts=0 首次更新覆盖暖启动；DDP 未广播 | **保留但改**：三模态暖启动 + counts=1 + rank0 广播；仅 finetune 必需 | P1 |
| **GPD-WR F2** | 计数感知 EMA；“稀有身份更快” | counts 按身份更新；PKM 按图像数分组；g_cent 已是三模态中心；与 F1 冲突 | 不是 modality-aware；TIR 少但 RGB 多的身份不受益；首步覆盖暖启动；几乎是 no-op | **删除**或仅作低优先消融 | P3 |
| **GPD-WR F3** | tau 0.05→0.03 | 原实现 tau=0.03；无早期软化证据；GPD loss 单调下降 | 早期削弱 GPD；finetune 无必要 | **仅消融，默认关** | P3 |
| **FTTA** | 推理期翻转平均；全局 | 训练已有 p=0.5 flip；项目无 TTA 证据；成本 2× | TIR/航拍视角不对称；协议披露；可能掉 R1 | **免费测 1000，预注册**；全四项 ≥+0.15 才保留 | P1 诊断 |
| **CG-QE** | transductive 查询扩展；mAP 强、R5/R10 崩 | QE λ0.1：mAP +0.86、R5/R10 −2.4；门控未测 | τ 在测试集选；被门控的 query R1/R5/R10 本来已正确，不能再升；协议公平性 | **默认关**；只作 mAP 上限诊断 | P3 |
| **PMC** | 3 个分类头；CE 梯度；推理不变 | 闭集 CE 已饱和（训练 acc 94.9–95.4%）；MDReID 的 +13.4 mAP 来自不同数据/基线 | TIR 头欠拟合；共享 BN；预期小；不能加推理容量 | 可选单变量消融，不作核心 | P2 |
| **MSIN** | TIR 输入逐图标准化 | 无项目证据；RLE 的输入增强失败是警示 | 去掉热对比度身份线索；改变 ImageNet 预训练域 | 可选 TIR 消融；仅 TIR 仍为瓶颈时 | P2/P3 |
| **Plan B** | 双模型分数/特征融合，或 CLIP 长训 | CLIP mAP 12.0–12.13、R1 29.2–29.4、R5 46.2、R10 54.1–54.2；CLIP 在 NIR→NIR/TIR→RGB 更好，UAD 在 TIR→TIR/RGB→RGB R1 更好 | 无融合实验；R1 目标高于两模型；2337/AG 未验；双模型公平性/成本 | **立即离线融合**；若过 1000 AS 则共主线；CLIP 长训并行 | P0 诊断 / P1 |

---

## 3. 关键组件详评

### 3.1 LHF：方向对，但草稿的具体实现有三处硬伤

#### 3.1.1 机制是否成立

成立，但幅度不对称。延长同一四项损失的训练，直接改变共享 ViT 的 pre-BN CLS 几何，因此不是“近 tie 重排”，而是全局排序变化；基线尾段 mAP/R1 仍在上升，说明优化没有饱和。[实验支持]

但 LHF 的 mAP 边际收益只有 R1 的约 1/3（0.0097 vs 0.0323 pp/epoch）。**目标 mAP +0.98 需要约 100 epoch 的尾段斜率，或一个不同的几何机制；60 epoch 的 LHF 大概率只能给 mAP +0.4–0.6。** 草稿把 LHF 的 mAP 上界写成 +1.0，是把“最乐观外推”当成了预期。[机制假设]

#### 3.1.2 硬伤一：`PRETRAIN_CHOICE='finetune'` 在基线代码中没有实现

**[代码事实]** `model/make_model.py:54-56` 只处理 `pretrain_choice == 'imagenet'`；`load_param_finetune`（`:91-95`）虽然存在，但没有被任何代码调用；`load_param`（`:83-89`）显式跳过 `classifier`。因此：

- 按草稿的 `MODEL.PRETRAIN_CHOICE='finetune'` 直接跑，模型**不会加载 `transformer_120.pth`**，而是从随机初始化开始（或至少分类器/BNNeck 是随机）；
- 即使手动调用 `load_param`，`classifier`/`bottleneck` 也不会恢复，CE 会从高损失开始，先破坏已训练好的 pre-BN 特征。

**修复**：新增 `finetune` 分支，加载完整 state dict（去掉可能的 `module.` 前缀），包含 `classifier`/`bottleneck`；在 step 0、optimizer 更新前，用同一固定 batch 比较原 H014 与 finetune 模型的 `global_feat`，必须逐元素一致；记录 checkpoint hash。没有这一步，A1 的所有结果都不可解释。

#### 3.1.3 硬伤二：LR 重启幅度过大，且 cosine floor 不是 0.0001

**[代码事实]** `solver/scheduler_factory.py:11-16` 的 `lr_min = 0.002 * BASE_LR`；DDP 基线 `BASE_LR=0.032` 的终段 LR 是 **6.4e-5**。草稿 A1 的 `BASE_LR=0.004` 是终段 LR 的 **62 倍**，且 cosine 会再降到 `0.002*0.004=8e-6`，不是草稿写的 0.0001。`solver/cosine_lr.py:67-94` 对每个 group 用各自的 `base_values` 做 cosine，但 `lr_min` 是全局单值；当 LLRD 把某组压到低于 `lr_min` 时，该组 LR 会在训练中**反向上升**（decay 0.75、base 0.004 时尚未触发；decay 0.5 会触发）。[代码事实]

62 倍的 LR 重启相当于第二段 cosine 周期。HiHR 的 Stage1→Stage2 切换（LR 从 0.032 重新 warmup）已经显示：切换本身可使 mAP 从 10.72 降到 9.83、R1 从 27.99 降到 26.76；Stage2 的 R1 增益（+0.58）远小于基线后半程（+2.72）。**在 120 epoch 的已训练解上再给 62 倍 LR，风险与 HiHR Stage2 同类。** [实验支持/机制假设]

**修复**：
- 首选 **A0**：从 H014 以**恒定的终段 LR**（DDP 6.4e-5；single 1.6e-5）继续 30 epoch，其他完全不变；每 5 epoch 评估；末轮预注册。这是对“尾段仍在上升”最直接的检验，成本约 0.25×120 epoch。
- 若 A0 通过，再做 **A2**：200 epoch 从头训练，`BASE_LR` 不变，warmup 5，cosine `t_initial=200`，`lr_min=0.002*BASE_LR`；这是“更长 schedule”的干净检验，而不是“重启”。
- 只有在 A0/A2 都失败、且确认需要第二段时，才做 A1 重启；此时峰值 LR ≤5e-4（DDP），warmup 3–5 epoch，且不要声称 floor=1e-4。

#### 3.1.4 硬伤三：GPD bank 不在 checkpoint，冷启动会破坏加载后的几何

**[代码事实]** `processor.py:156-161` 构造 `PrototypeBank`，`:225-227` 在 optimizer step 后更新；`torch.save` 只保存 `model.state_dict()`（`:266-274`），bank 不是模型的一部分。H014 结束时 GPD loss 已降到 0.08，但 finetune 从零 bank 重启，GPD 的 logits 初始接近“目标类 + 其余零”，损失会从 ~6 开始，前几步的梯度可能把已训练的特征拉走。[代码事实/机制假设]

**修复**：F1 暖启动是必需的，但要满足四点：

1. **三模态**，不是 RGB-only：用 `model.eval()` + `no_grad` 跑一遍 train 的三模态图像（或每个 PID 每模态若干张），按 PID 求均值再 L2；`train_loader_normal` 目前只有 RGB（`make_dataloader.py:129-138`），需要额外构造 NIR/TIR normal loader，或接受 RGB + 一个轻量三模态 batch 的折中；
2. **counts 初始化为 1**（或暖启动原型视为已更新一次），否则 F2 的 `m_eff=1/(0+1)=1` 会在第一次更新时把暖启动原型完全覆盖；
3. **rank0 计算、广播到所有 rank**，否则各 rank 的 bank 不同，GPD loss 与梯度不一致；
4. **验证**：记录 step 0 的 GPD loss、bank 非零覆盖率、原型与 `g_cent` 的余弦；若 GPD loss 仍从 >5 开始，暖启动没有生效。

另一个更简单的替代：finetune 的第 1 个 epoch 关闭 GPD（或权重 0），让 bank 自然填充；这不增加额外 pass，但会浪费 1/60 的预算。两种都可，关键是不要“零 bank + 全 GPD 权重 + 高 LR”同时发生。

#### 3.1.5 LHF 的风险与判定

- **过拟合/退化**：低 LR 延长训练不等于不会过拟合；必须看训练/测试 gap、GPD loss、危险负样本相似度，而不只是 mAP。
- **DDP eval**：训练期每 10 epoch 的验证把模态分给不同 rank，最后 gather 到 rank0。若验证的是 EMA 模型，**每个 rank 必须持有同一份 EMA 权重**；只在 rank0 更新 EMA、其他 rank 仍用 final 模型，会把两个模型的模态特征拼接在一起，结果无效。[代码事实/实现风险]
- **判定**：A0 的通过标准应是 mAP ≥ +0.25、R1 ≥ +0.4、R5/R10 ≥ +0.2，且没有任何指标下降；若 30 epoch 内 mAP 不升，停止，不要用“R1 升了”掩盖 mAP 的失败。

### 3.2 CA-WRT：负样本侧方向对，但草稿的正样本权重用反了

#### 3.2.1 机制

WRT 只按 PID 定义正负（`wrt_loss.py:46-47`），同相机异身份是负样本、同相机同身份是正样本；评测 `metrics.py:164` 排除**全部**同相机 gallery。训练/评测错配是真实的。[代码事实]

草稿的 `pos_mask = is_pos * (~same_cam + same_cam*w_pos)`，`w_pos=0.5`：这会把**同相机的同身份正样本**降权。问题在于，UAD 的 batch 是 `[RGB;NIR;TIR]` 三模态拼接，同一 PID 的跨模态正样本很可能同相机（若数据是同一相机节点的三传感器同步采集；需实测）。**跨模态同相机正样本是 TIR/NIR 对齐的主要信号；把它降权 0.5，可能削弱跨模态对齐，正好伤害 TIR 五对。** [代码事实/机制假设]

负样本侧 `w_neg=0.25` 的方向更合理：同相机异身份在评测中永远不考，降低其权重可以把容量让给跨相机负样本。但同相机负样本也可能是“最危险负样本”（同背景/同光照），完全降权不一定好；需要先统计。

#### 3.2.2 证据

- 没有直接证据表明 CA-WRT 在 WHU 上有效；TAUR 的 rank loss 显式排除同相机，但审计显示 rank 梯度只占 adapter 的 1.5–2.1%，且同时改了 `CameraPKMSampler` 和 `WRT_STABLE`，无法归因。[实验支持/审计事实]
- KGL/PAT/ViSA 等历史方法均未显式处理同相机，也没有出现“同相机负样本是主因”的证据；A-V1 的失败主要来自危险负样本相似度上升，而不是同相机。[实验支持]

#### 3.2.3 建议

1. **先统计**：用数据集元数据 + PKM 采样器枚举若干 epoch 的 batch，统计每个 query 的 WRT 正/负 pair 中同相机的比例；再在 H014 模型上计算“每 query 最难负样本是否同相机”的比例。如果同相机负样本占比 <5%，CA-WRT 不值得做。[免训练诊断]
2. **若做，先只改负样本**：`w_pos=1.0`、`w_neg=0.25`（或 0 硬排除）；保留原 WRT 的正样本定义；对角线是否置 0 要与原版对照（原 WRT 的对角线是正样本，距离 0，权重很低；置 0 是额外改动）。
3. **更优先的替代：Camera-Aware PKM 采样**。TAUR 的 `CameraPKMSampler` 让同一 PID 的 K 个实例优先进跨相机，属于**正样本侧**对齐评测，且不改变损失形式；它比 CA-WRT 更接近“训练分布 = 评测分布”。但必须配套 matched UAD control（同 sampler + 原 WRT），因为 TAUR 的 sampler 改动没有对照日志。[实验支持/审计事实]
4. 不要把 CA-WRT 与 NWRT、camera sampler 同时上；三者在改同一个 WRT 几何，无法归因。

### 3.3 NWRT：没有证据的 near-tie 重排，应删除

- 特征范数几乎恒定（均值 ≈10，std ≈0.38），raw 欧氏与 L2 余弦的 **R1 完全相同**，top-1 仅 65.8% 一致；这说明差异主要是 near-tie 的顺序翻转，而不是系统性的几何差异。[实验支持]
- 草稿的 `inputs_mix = (1-β)inputs + β(u*s)`，其中 `s` 是标量均值范数；在范数几乎恒定时，这基本只是把模长拉到共同尺度，β=0.5 没有理论依据。它还会改变 WRT 中 softmax 权重的距离尺度，带来一个难以解释的 schedule 混杂。
- **结论：删除。** 如果目标是训练/推理一致，做一个 β=1 的完整归一化 WRT 单变量消融即可；预期约为 0，不应进入主方案。[机制假设/实验支持]

### 3.4 GPD-WR：只保留 F1，F2 的前提不成立，F3 没有证据

#### F2 计数感知 EMA 的问题

草稿的动机是“稀有身份（TIR/航拍）更快跟上”。但：[代码事实/机制假设]

- `PrototypeBank.update` 的 `counts` 是按**身份**更新的，不是按模态；bank 的原型是 `g_cent`（三模态中心的均值），本身就是三模态的；
- PKM 采样按每个 PID 的**最大模态图像数**分组（`sampler.py:94-125`），图像多的身份出现更多 batch，counts 更大；图像少的身份 counts 小，确实会获得更大的 `m_eff`。但这解决的是“图像少的身份”，不是“TIR 少的身份”；一个 TIR 少但 RGB 多的身份，counts 仍然很大，F2 不会让它更快；
- 所有身份在第一个 epoch 内都会被采样到（500 IDs / 16 per batch），counts 在前几步就从 0 变 1、2、3……；`m_min=0.2` 使 4 次更新后所有常见身份都回到 0.2，稀有身份的优势只持续几个 step；
- 与 F1 直接冲突：暖启动后 counts=0，首次更新 `m_eff=1`，会把暖启动原型完全覆盖。

**结论：删除 F2**，或把它重新设计为“每模态原型 bank”（`[3,C,D]` + 每模态 counts），但那是更大的结构改动，且与“不新增分支”的原则冲突；不建议。

#### F3 温度课程

原实现固定 tau=0.03，GPD loss 在全部历史实验里单调下降，没有早期过尖/崩塌的证据；把 tau 在前 20 epoch 从 0.05 降到 0.03，会让早期 GPD 更软、更弱。在 finetune 场景（原型已暖启动）没有理由再软化。**保留 0.03；F3 只作低优先消融。** [实验支持/机制假设]

#### F1 的定位

F1 对 finetune 是必需的（bank 不在 checkpoint）；对 200 epoch 从头训练，冷启动在第一个 epoch 内会自然修正，F1 的增益较小。不要把 F1 当成 +0.1–0.4 的普适增益；它的主要价值是**避免 A1 的冷启动破坏**。

### 3.5 FTTA / CG-QE / PMC / MSIN

- **FTTA**：推理期翻转平均，直接改变全局排序，不是 near-tie 专用；但项目内没有证据，且 TIR/航拍视角不对称，翻转可能改变身份线索。它是**免费可测**的（只需一次额外前向），应作为 S0 的预注册诊断；不要把它写进 +1pp 的预期预算。通过标准应是四项都不下降且至少一项 +0.2，而不是“任一指标 +0.2”。[机制假设]
- **CG-QE**：离线证据是 mAP +0.86、R5/R10 −2.4（λ=0.1）。草稿的门控假设“高 margin query 更可能已正确，QE 只改善 AP 不伤 top-1”在逻辑上只对“已正确的 query”成立；对这些 query，QE 不可能提高 R1/R5/R10，只能提高 mAP。因此门控 QE 是一个 **mAP-only** 工具，不能帮助主指标；而且 τ 在测试集上选会引入选择偏差。**默认关闭；只在 mAP 单独落后、且 R5/R10 有护栏时，作为最后的诊断。** [机制假设/实验支持]
- **PMC**：CE 已饱和（训练 acc 94.9–95.4%），PMC 改变的是 CE 经 BN 回传的梯度；在饱和区，这个梯度很小。MDReID 的 +13.4 mAP 来自 RGBNT201 这类更难、基线更低的数据，不能直接外推。**作为单变量消融可以，但不应作为核心 +1 杠杆。** 若做，建议同时试“3 分类头 + 共享 BN”和“3 分类头 + 3 BNNeck”，并报告 TIR→TIR/TIR→RGB。[实验支持/机制假设]
- **MSIN**：TIR 逐图标准化是低风险输入改动，可能帮助 TIR 的强度/对比度差异；但它会去掉热对比度这一身份线索，且改变 ImageNet 预训练域。RLE 的失败说明输入级改动要非常小心。**仅在 TIR 仍是瓶颈、且 LHF/校准/fusion 都不够时做单变量消融；不要默认开启。** [机制假设]

### 3.6 Plan B：应升级为共同主线，但必须分“1000 AS”和“2337/AG”两件事看

#### 3.6.1 融合的机制与证据

- CLIP-ReID + ProCA + GPD 的三个 seed：mAP 12.01/12.10/12.13，R1 29.18/29.37/28.56，R5 46.18/45.51/46.31，R10 54.24/54.21/54.07。**mAP 已经超过 1000 的目标 12.0**；R1 比目标 30.5 低 1.1–1.9，R5 低约 1.0–1.8，R10 低约 1.5–1.7。[实验支持]
- CLIP 在 NIR→NIR（mAP +1.23/R1 +2.25）、TIR→RGB（mAP +1.22/R1 +1.87）上明显优于 UAD；UAD 在 TIR→TIR、RGB→RGB 的 R1 上更好。两者互补。[实验支持]
- 对 L2 特征，分数融合 `α cos_UAD + (1-α) cos_CLIP` 与特征拼接（归一化后）在数学上等价于加权余弦；不需要训练，只需特征提取与逐 query 排序。RRF（reciprocal rank fusion）对分数尺度更稳健。[机制事实]

#### 3.6.2 预期、成本、公平性

- **预期**：如果两模型的 top-1 错误集重叠约 60–75%，融合可把 R1 从 29.2–29.7 抬到 30.5–31.5，mAP 从 12.0–12.1 抬到 12.4–13.0。草稿的“mAP 12.5+/R1 30.5+”是**可能**的，但不是必然；R1 目标高于两模型各自，融合必须依靠互补性。[机制假设]
- **成本**：1000/GD 只需一次特征提取（93k gallery × 2 模型）和离线排序，成本远低于一次训练；2337 需要 CLIP 在 2337 上重训（两阶段），成本高且不确定；AG 需要 CLIP 的 AG 特征，未知。
- **公平性**：双模型、双倍参数/推理；不能冒充单模型。报告时必须同时给 UAD 单模型、CLIP 单模型、融合三者，并明确“双模型互补集成”。[方法]
- **实现风险**：manifest 对齐（用 WCV1 的稳定文件键/hash 检查）；同相机过滤必须在融合后的全 gallery 上做；α 不能在测试集上扫（预注册 0.5 或 RRF）；CLIP 的特征协议（pre-BN/维度/归一化）必须与 UAD 一致；两个模型的分工（UAD 强 R1、CLIP 强 mAP）要分协议报告。[代码事实/方法]

#### 3.6.3 CLIP 继续训练是否更可行

- **是，但要看目标**：CLIP 已经在 mAP 目标上；它需要的是 R1/R5/R10 +1.1/+1.0/+1.6。如果 CLIP 的 120 epoch 轨迹也像 UAD 一样尾段仍在上升，40–60 epoch 低 LR 延长 + EMA/TTA 可能给出 +0.5–1.2 R1，mAP 再 +0.3–0.6；这有望达到 1000 AS 目标，且是**单模型**，没有双模型公平性问题。[机制假设]
- **风险**：CLIP 的 R1 基线比 UAD 低 0.3–1.1pp，需要补的更多；CLIP 的两阶段/Adam 配置与 UAD 不同，LR/epoch 需单独调；2337 要重训；AG 上 CLIP-ReID 可能变差（SeCap 论文的 LAGPeR 上 CLIP-ReID 远低于 ViT，虽然 WHU-MARS 不同）。[文献/机制假设]
- **建议**：如果 GPU 允许，B2（CLIP 长训）与 B1（融合）并行；B1 是 1000 AS 数值目标概率最高的路径，B2 是更干净的单模型备选。不要在未做 B1/B2 的情况下，把全部预算押在 UAD 的小模块上。

---

## 4. 必须修改的问题（按严重度）

### S0（阻断，必须先修，否则实验无效）

**S0-1. finetune 加载路径未实现。**  
[代码事实] `make_model.py:54-56` 只处理 `imagenet`；`load_param_finetune` 未被调用；`load_param` 跳过 `classifier`。草稿 A1 直接设 `PRETRAIN_CHOICE='finetune'` 会从随机权重开始或丢失分类器/BNNeck。  
**修复**：新增完整 state-dict 加载分支；step 0 特征与原 H014 逐元素一致；log hash；把该项作为 A1 的“启动自检”。

**S0-2. LHF 的 LR/epoch 设定与基线调度不匹配。**  
[代码事实/计算] 终段 LR 6.4e-5（DDP）；草稿 0.004 是 62×；cosine floor 是 8e-6，不是 1e-4。  
**修复**：先 A0 恒低 LR 30 epoch；再 A2 200 epoch 从头；A1 重启仅在 A0/A2 失败后，峰值 ≤5e-4、warmup 3–5。

**S0-3. GPD bank 冷启动 + F2 覆盖暖启动。**  
[代码事实] bank 不在 checkpoint；零初始化；F2 counts=0 首次更新 `m_eff=1` 会覆盖 F1。  
**修复**：三模态暖启动 + counts=1 + rank0 广播 + step-0 GPD loss 检查；删除 F2。

**S0-4. 验收集与目标口径未定义。**  
[协议事实] 12 AS / 6 AG / 6 VI / 27 个 3×3 的“全面”不可同日而语；AS 总体 mINP 没有论文目标；GD R5/R10 只有 0.14/0.06 余量。  
**修复**：冻结 Tier A/B/C；Tier A 为必须，B/C 为尽量；mINP 只对 VI/3×3；GD R5/R10 作为护栏。

### S1（高，直接影响结论有效性）

**S1-1. mAP 是绑定约束，而方案是 R1 重的。**  
[实验支持/计算] LHF 的尾段 mAP 斜率只有 R1 的 1/3；60 epoch 外推 mAP +0.58、R1 +1.94；目标 mAP +0.98。  
**修复**：增加 mAP 专项：先做免费 per-camera/per-platform 校准与 UAD+CLIP 融合；把 mAP 作为 S1/S2 的主 gate，而不是只看 R1。

**S1-2. CA-WRT 的 w_pos=0.5 方向错误。**  
[机制假设/代码事实] 同相机正样本包含跨模态正样本；降权会削弱 TIR/NIR 对齐。  
**修复**：w_pos=1.0、w_neg=0.25（或 0）；先统计同相机 pair 比例与最难负样本的相机身份；优先做 camera sampler 的 matched control。

**S1-3. NWRT 没有证据。**  
[实验支持] raw/normalized R1 完全相同；65.8% top-1 一致；β=0.5 任意。  
**修复**：删除；若做一致性消融，β=1 单测，预期 ~0。

**S1-4. 实验捆绑，违反草稿自己的“单变量”原则。**  
S2（LLRD+EMA）、S3（CA-WRT+NWRT）、S5（soup+TTA）都是一次改多个变量；失败时无法归因，成功时也无法写方法。  
**修复**：一个 stage 只改一个变量；或者做 2×2 因子设计，但不要“打包后再拆”。

**S1-5. 无验证集、单 seed、测试集选 epoch。**  
[实验支持/统计] 1pp R1 = 64 个 query；A-V1 的翻转量是 ±2pp；单 seed +1pp 不可信。  
**修复**：至少 2 个 seed；paired bootstrap（按 query 重采样）；预注册末轮；报告翻转数；如果只能在测试集上看，明确标注“探索性”，不要把最佳 epoch 当正式结果。

**S1-6. GD R5/R10 的零余量。**  
[协议事实] 46.14 vs 46.0、53.56 vs 53.5；任何 0.1–0.2 回退就失败。  
**修复**：每个 stage 都跑 GD；把 GD R5/R10 作为否决项；不要用 GD 调参。

### S2（中，影响实现质量/解释）

**S2-1. QE 的 τ 在测试集选，且门控 QE 只能升 mAP。**  
**修复**：默认关；若用，预注册 τ 或在不重叠的 split 上选；主结果不含 QE；报告 R5/R10。

**S2-2. EMA/soup 的 DDP eval 与 checkpoint 细节。**  
[代码事实] 训练期验证把模态分给不同 rank；EMA 必须在所有 rank 上一致；`CHECKPOINT_PERIOD=120` 使基线只有 ep120，soup 需要改成 10。  
**修复**：EMA 在每个 rank 更新或 eval 前广播；测试脚本显式加载 final/EMA/soup；预注册报告哪一个；保留 final 作为主结果。

**S2-3. 1000GD daytime 与 mINP 公式未核实。**  
[代码事实/协议事实] `whu_mars.py:75,90` 只有 pid/camid；`metrics.py:192` 的 mINP 是 `num_rel/(last+1)`，与 AGW 是否一致未核实。  
**修复**：manifest 审计 GD 的昼夜构成；用 AGW 官方实现核对 mINP；在核实前把 GD/mINP 目标降级为参考。

**S2-4. 288×144 / 16×9 的 test-time 分辨率不可直接跑。**  
[代码事实] `vit_pytorch.py:317-341` 直接加固定 `pos_embed`，没有 runtime 插值；`build_transformer` 用 `SIZE_TRAIN` 构建。  
**修复**：如果要做分辨率诊断，需在构建时用 288×144 并让 `load_param` 插值；否则不要把它写成“免费一步”。鉴于论文 Table 5 与 Table 6 的 UAD 数值完全相同，分辨率大概率不是主因，优先级低。

**S2-5. LLRD 的全局 cosine floor。**  
[代码事实] `cosine_lr.py:67-94` 的 `lr_min` 是全局的；低 LR 组可能被 floor 反向抬升。  
**修复**：若做 LLRD，decay 0.85–0.9，并检查每个 group 的 LR 曲线；不要用 0.5；A1 中不要用 LLRD。

---

## 5. 修订版方案

### 5.1 组件组合（建议）

**核心（按顺序）**

1. **免费诊断（S0）**：per-camera/per-platform 校准、per-query 模态分数归一化、flip TTA、UAD+CLIP 融合、同相机 pair 统计、GD manifest。
2. **LHF**：A0 恒低 LR 30 epoch → A2 200 epoch 从头；不默认 LLRD；不默认 EMA，EMA/soup 作为 post-hoc。
3. **GPD F1**：仅 finetune 必需；三模态、counts=1、rank0 广播。
4. **Camera-aware PKM 采样**：作为正样本侧对齐；必须有 matched UAD control。
5. **CA-WRT**：只有在 S2a（camera sampler）失败、且同相机负样本统计支持时，做“仅负样本”版本。
6. **post-hoc**：soup/EMA、flip TTA；每个都要单变量、预注册。
7. **可选单变量**：PMC 或 MSIN（二选一，只在 TIR 仍为瓶颈时）。
8. **Plan B**：UAD+CLIP 融合先做；CLIP 长训并行；2337 单独重训。

**删除**

- NWRT；GPD F2/F3；CG-QE（主结果）；TIR adapter/LoRA；memory queue；hierarchical ProCA；per-camera 训练原型（ProCA 权重只有 0.01，且 K=4 组内相机切分后样本太少）；输入分辨率 288×144（除非 VI 目标单独成为必须）。

**理由**

- 历史 17 组 + A-V1/B-B/WCV1 已经证明：新增排序/对比/混合 gallery 损失会抬高最危险负样本，R1 下降；TIR adapter/LoRA 类容量改动的历史结果均未超过基线；memory queue 的动机（WCV1 的 danger identity present rate 只有 11.9%、danger image 1.0%）不足以支撑新增分支，而且 A-V1 显示增加负样本压力是反向的。[实验支持]
- ProCA 权重只有 0.01；把它改成层次/相机中心，收益上界很小，风险却大（每相机 K=1–2 个样本）。[代码事实/机制假设]

### 5.2 参数建议

| 项 | 建议 | 说明 |
|---|---|---|
| A0 恒定 LR | DDP 6.4e-5；single 1.6e-5 | 等于各自 120 epoch cosine 的终段 LR；30 epoch；eval every 5 |
| A2 200 epoch | `BASE_LR` 0.032 DDP / 0.008 single；warmup 5；cosine 200；`lr_min=0.002*BASE_LR` | 干净的长 schedule 检验 |
| A1 重启（备选） | DDP 5e-4 / single 1.25e-4；warmup 3–5；cosine 60 | 不要 0.004；不要声称 floor=1e-4 |
| LLRD | 仅 A2 可选，decay 0.85–0.9；`patch_embed` 与 `blocks.0` 同一量级；`pos_embed`/`cls_token` 0.1–0.2；`norm` 0.5–1.0 | 先检查每组 LR 曲线；不做 0.75 |
| EMA | decay 0.999；每 step 更新；初始化自加载权重；所有 rank 一致 | 0.9995 窗口约 17 epoch，滞后太大；0.999 约 9 epoch |
| Soup | 最后 3 个 checkpoint（10 epoch 间隔）；等权；预注册 | 同时报告 final 与 soup；不要只报 soup |
| TTA | flip；`normalize(f1)+normalize(f2)` 再 L2；先 1000 | 2× 成本；四项全 ≥+0.15 才保留 |
| Camera sampler | TAUR 的 `CameraPKMSampler`；K=4；matched UAD control | 检查 steps/epoch（117 vs 114）；不要与 CA-WRT 同开 |
| CA-WRT | w_pos=1.0；w_neg=0.25（或 0）；对角线按原版保留或单测 | 只在统计支持时做；日志记录 pair 比例 |
| GPD warm start | 三模态、每 PID 均值、L2；counts=1；rank0 广播 | step-0 GPD loss 必须显著低于 5 |
| GPD tau | 0.03（原版） | F3 只作消融 |
| PMC | 3 个 `Linear(768,C)`，共享 BN；CE 路由；推理不变 | 单变量 120 epoch |
| MSIN | TIR per-image standardization；p=1 | 单变量 120 epoch；只看 TIR pair + overall |
| Fusion | 先 `α=0.5` 余弦融合 + RRF；α sweep 只作诊断；不拿测试集选 α | 双模型；报告三者 |
| Seeds | 至少 2 个；报告 mean 与范围；paired bootstrap | 1pp = 64 query；单 seed 不够 |

### 5.3 实验顺序与判定（修订版 S0–S7）

| 阶段 | 配置 | 目的 | 通过标准 | 失败动作 |
|---|---|---|---|---|
| S0a | H014 复测 + per-query 导出 + paired bootstrap | 建立噪声基线 | 复现 11.0234/29.7112 等 | 先修评测 |
| S0b | per-camera/per-platform 校准（gallery-only、both、λ=0.1/0.3/0.5、shrinkage） | 免费 mAP 杠杆 | 四项全 ≥+0.15 且 R5/R10 不降 | 关闭该变换 |
| S0c | per-query 模态分数归一化 | 混合 gallery 竞争 | 同上 | 关闭 |
| S0d | flip TTA（1000） | 免费推理增益 | 四项全 ≥+0.15 | 关闭 |
| S0e | UAD+CLIP 融合（α=0.5/RRF） | Plan B 上限 | 1000 AS 四项 ≥ 目标 | 保留为单模型路线，记录差距 |
| S0f | 同相机 pair 统计、GD manifest、mINP 公式 | 决定 CA-WRT/GD 口径 | 统计可解释 | 暂缓 CA-WRT/GD 结论 |
| S1a | A0 恒低 LR 30 epoch | 验证“尾段仍升” | mAP ≥+0.25、R1 ≥+0.4、R5/R10 ≥+0.2，无指标下降 | 停止 LHF；转 Plan B |
| S1b | A2 200 epoch 从头 | 干净长 schedule | mAP ≥+0.5、R1 ≥+0.8 vs H014 | 若 mAP <+0.5，Plan B 升主线 |
| S2a | camera sampler + matched control | 正样本侧对齐 | R1 ≥+0.2 且 mAP 不降 | 关闭 |
| S2b | CA-WRT 仅负样本 | 负样本侧对齐 | R1 ≥+0.2 vs S2a 且 mAP 不降 | 关闭 |
| S2c | per-camera 校准应用到最佳模型 | mAP 杠杆 | 四项全 ≥+0.15 | 关闭 |
| S3a | PMC 或 MSIN（二选一） | TIR 覆盖 | TIR pair mAP/R1 ≥+0.3 且总体不降 | 关闭；不做另一个 |
| S4a | soup/EMA | 稳定性 | 四项全 ≥+0.1 vs final | 保留 final |
| S4b | TTA + 校准组合 | 推理组合 | 四项全 ≥+0.15 | 回退 |
| S5 | 1000GD 同权重；2337 重训；AG 评测 | 跨数据集 | Tier A 全达标；GD R5/R10 不低于目标 | 不达标则记录，不调 GD/AG 测试集 |
| S6 | Plan B 融合/CLIP 长训 | 数值目标 | 1000 AS 四项 ≥ 目标 | 作为双模型交付 |
| S7 | 2 seed 复现最佳配置 | 稳健性 | 2 seed 均值仍 ≥ 目标，或报告区间 | 降级为“单次达到” |

**合并/跳过**：LLRD 不单独成 stage；NWRT 删除；F2/F3 删除；CG-QE 不进入主线；TIR adapter/memory queue/hierarchical center 不进入排期；分辨率只作为 VI 专项诊断，不作为主线。

**注意**：S0b/S0c/S0d 都是推理期变换，必须在**同一份 H014 特征**上做，且用 same-camera 排除后的全 gallery；不要在每个 pair 上分别调 λ（会过拟合 9 个 cell）。预注册一个 λ 或一个很小的集合，报告所有结果。

### 5.4 判定纪律

1. **主 gate 是 mAP 和 R5，不只是 R1。** 每个 stage 都报告四项 + GD R5/R10；任何一项下降超过 0.1pp 都要解释。
2. **1pp 的显著性**：用 paired bootstrap 对 query 重采样，报告 95% CI；若 +0.8pp 的 CI 跨 0，不能称为达标。
3. **至少 2 seed**：如果只有 1 seed，把结果写成“单次运行”，不要写“方法提升”。
4. **预注册报告对象**：final / EMA / soup / TTA 四者中选一个作为主结果，其余作消融；不能看完测试集再选。
5. **不调 GD/AG 的测试集**：GD 是同权重子集，只作为护栏；AG 只在 1000 上评测，不能拿 AG 结果反调 1000 配方。

---

## 6. 需要补充的诊断/验证（免训练优先）

| # | 诊断 | 回答什么问题 | 成本 | 优先级 |
|---|---|---|---|---|
| 1 | per-camera 特征中心化/缩放（gallery-only / both；λ=0.1/0.3/0.5；shrinkage；min-count） | 是否存在免费的 mAP 杠杆；相机偏差是否可去除 | 0 训练，离线 | P0 |
| 2 | per-platform（ground/aerial）中心化/缩放 | AG 与跨平台检索是否有免费增益 | 0 训练 | P0 |
| 3 | per-query 模态分数归一化（top-k 统计后跨模态重排） | 混合 gallery 的模态竞争能否缓解 | 0 训练 | P0 |
| 4 | UAD+CLIP 分数/特征/RRF 融合；α sweep 仅作诊断 | Plan B 上限；1000 AS 能否达标 | 特征提取 | P0 |
| 5 | flip TTA（1000） | TTA 是否真有 +0.2；对 TIR/航拍是否伤害 | 2× 前向 | P0 |
| 6 | 同相机正/负 pair 比例；每 query 最难负样本的相机身份 | CA-WRT 是否值得做；w_pos 应不应降 | 元数据 + 固定 batch 前向 | P0 |
| 7 | LHF 中间 checkpoint（ep130/140/150/160/180）的 rank transitions、s+/s−、危险负样本相似度 | LHF 是全局改善还是 near-tie 翻转；是否重复 A-V1 的失败模式 | 特征提取 | P0 |
| 8 | GPD bank 覆盖率、零原型比例、原型范数、与 g_cent 余弦、counts 分布 | 暖启动是否生效；F2 是否有意义 | 训练内记录 | P1 |
| 9 | 分组梯度范数（base blocks / patch_embed / clasifier / 新分支）与 clip 频率 | LLRD/CA-WRT 是否只是在改优化预算；新分支是否挤占梯度 | 训练内记录 | P1 |
| 10 | 每 query 的最佳分数按 gallery 模态分解；错误 top-1 的来源模态 | 40% 的跨模态冲突能否被 per-query 归一化/fusion 修复 | 离线 | P1 |
| 11 | QE 按 margin 分位的 mAP/R5/R10 | 门控 QE 是否有安全区；τ 应如何定 | 离线 | P2 |
| 12 | GD manifest 昼夜构成；mINP 与 AGW 公式核对 | GD/mINP 目标是否合法 | 数据审计 | P1 |
| 13 | 288×144（pos_embed 插值）同权重 VI 指标 | Table 5 的差距是否来自分辨率 | 代码改动 + 推理 | P2 |
| 14 | 2 seed / paired bootstrap | +1pp 是否超出噪声 | 训练 + 统计 | P0（最终） |

**最重要的三个**：1（per-camera 校准）、4（UAD+CLIP 融合）、7（LHF checkpoint 的 rank transition / 危险负样本）。它们分别回答“mAP 有没有免费杠杆”“有没有互补上限”“LHF 是否在重复历史失败”。

---

## 7. Plan B 评估

### 7.1 B1：UAD + CLIP 融合

**应升级为共同主线，但只对 1000 AS 有高概率。** 理由：

- CLIP 已经在 mAP 目标上（12.0–12.13），UAD 在 R1/R5/R10 上更强；两者的互补性有 per-pair 证据（CLIP 的 NIR→NIR/TIR→RGB 更好，UAD 的 TIR→TIR/RGB→RGB R1 更好）。[实验支持]
- 融合不需要训练；1000/GD 的成本只是特征提取和离线排序。它可以在 1–2 天内回答“上限在哪里”，比再跑一个 120 epoch 的小模块便宜得多。[机制事实]
- 但 R1 目标 30.5 高于两模型各自；融合必须带来 +0.8–1.3 的互补增益。若两者的错误重叠高，融合可能只到 30.0–30.3，仍差 0.2–0.5。不要承诺“12.5+/30.5+”，先测。[机制假设]

**公平性/成本**：双模型、双倍参数/推理；必须标注“双模型互补集成”，同时报告 UAD/CLIP 单模型。2337 需要 CLIP 在 2337 上重训；AG 需要 CLIP 的 AG 特征。若不能接受双模型，B1 只能作为诊断/上限，不作为最终交付。

**实现风险**：manifest 对齐、同相机过滤在融合后做、α 不在测试集上扫、CLIP 特征协议与 UAD 一致、存储/显存。用 RRF 或固定 α=0.5 避免选择偏差。

### 7.2 B2：CLIP 继续训练 + EMA/TTA

**比草稿的定位更可行，但仍有不确定性。** CLIP 的 mAP 已经在目标上，缺的是 R1/R5/R10；如果 CLIP 的 120 epoch 轨迹也像 UAD 一样尾段上升，低 LR 延长 40–60 epoch + EMA/TTA 可能补上 +0.5–1.2 R1，mAP 再 +0.3–0.6。它是单模型，公平性最好；但 CLIP 的 R1 基线比 UAD 低，需要补的更多，且 2337/AG 未验证。[机制假设]

**建议**：B1 先做（便宜、直接回答 1000 AS）；B2 在 GPU 允许时并行（干净的单模型备选）。若 B1 达到 1000 AS 目标，把 B1 作为 1000/GD 的数值交付，同时继续 B2/单模型为 2337 和“方法”叙事。不要把两者混为一个“方法”。

### 7.3 结论

- **Plan B 不应只是“如果单模型差 0.3–0.5 再说”的备选。** 由于 mAP 是绑定约束、CLIP 已在 mAP 目标上，Plan B 是当前证据下最接近 1000 AS 目标的路径；应在 S0 就做离线融合，而不是等 S5。
- **但 Plan B 不能自动解决 Tier A 的 2337 和 AG。** 2337 需要 CLIP 重训（或 UAD LHF 200 epoch）；AG 需要单独评估；如果 Tier A 包括 AG/VI/3×3，Plan B 的概率仍只有 10–15%。
- **不要用 Plan B 掩盖单模型的失败**：报告必须分开“UAD 单模型”“CLIP 单模型”“双模型融合”，并明确验收对象。

---

## 8. 最终建议（10 条）

1. **保留核心原则**：不新增排序/对比/混合 gallery 损失；历史 17 组 + A-V1/B-B/WCV1 已证明这条路会抬高最危险负样本。
2. **先冻结验收集**：Tier A = AS 12 cell；Tier B = AG/VI；Tier C = 3×3 27 cell。若“全面”= Tier C，明确这是“尽量达”，否则方案不可证伪。
3. **承认 mAP 是绑定约束**：LHF 的尾段 mAP 斜率只有 R1 的 1/3；60 epoch 更可能给 mAP +0.4–0.6、R1 +1.2–1.9。不要把 LHF 的 mAP 上界写成 +1.0。
4. **反转 A1/A2**：先 A0 恒低 LR 30 epoch；再 A2 200 epoch 从头；A1 的 0.004 重启是 62× 终段 LR，风险与 HiHR Stage2 同类。
5. **修 finetune 加载与 GPD bank**：`PRETRAIN_CHOICE='finetune'` 当前未实现；`load_param` 跳过 classifier；bank 不在 checkpoint。先做 step-0 特征一致性 + 三模态暖启动，否则 A1 结果无效。
6. **删除 NWRT、GPD F2/F3、CG-QE 主结果、TIR adapter/LoRA、memory queue、hierarchical ProCA**；CA-WRT 改为“仅负样本”或先做 camera sampler；LLRD 降为 A2 可选（0.85–0.9）。
7. **先做免费诊断**：per-camera/per-platform 校准、per-query 模态归一化、flip TTA、UAD+CLIP 融合、同相机 pair 统计。它们比再训一个小模块更可能给出 mAP 增益，且成本低。
8. **Plan B 升级为共同主线**：立即离线融合；若 1000 AS 四项达标，作为 1000/GD 的数值交付；CLIP 长训并行作为单模型备选；2337/AG 单独验证；明确双模型公平性。
9. **统计纪律**：2 seed + paired bootstrap；一个 stage 一个变量；预注册 final/EMA/soup；GD R5/R10 作为护栏；不以测试集挑 epoch；不在 GD/AG 测试集上调参。
10. **概率结论**：草稿原样 Tier A 约 5–10%，1000 AS 四项约 10–20%；修订单模型 Tier A 约 10–15%，1000 AS 约 25–35%；融合的 1000 AS 约 35–45%，Tier A 约 15–25%。目标本身在历史效应量之上，建议把“1000 AS 达标”作为第一交付，“2337/AG/3×3 全面”作为第二交付，并提前决定是否接受双模型。

---

## 附：关键证据路径

- 方案：`D:\CODE\Doing\WHU_base\WDV1_CC\research\phase2_design\design_draft.md`
- 协议：`D:\CODE\Doing\WHU_base\WDV1_CC\research\phase1_protocol\protocol_target_analysis.md`、`target_table.csv`
- 轨迹：`D:\CODE\Doing\WHU_base\WDV1_CC\research\phase1_logs\log_trajectory_report.md`
- 审计：`D:\CODE\Doing\WHU_base\WDV1_CC\research\phase1_history\audit_group1_hihr_visa_sftc_icre_taur.md`、`audit_group2_kgl_pat_wqvd_scsr_rle_peft_reid5o_mspec.md`
- 文献：`D:\CODE\Doing\WHU_base\WDV1_CC\research\phase1_literature\literature_review_2025_2026.md`
- A-V1：`D:\CODE\Doing\WHU_base\WAV1\AV1_DIAGNOSIS_ANALYSIS.md`、`CODE_CHANGE_RECORD.md`
- B-B：`D:\CODE\Doing\WHU_base\WBV1\WHU\BB_IMPLEMENTATION.md`
- WCV1：`D:\CODE\Doing\WHU_base\WCV1\WHU\WCV1_IMPLEMENTATION.md`、`D:\CODE\Doing\WHU_base\WCV1\logs\diagnosis_120\report.md`
- 基线代码：`D:\CODE\Doing\WHU_base\WDV1_CC\WHU\processor\processor.py`、`loss\wrt_loss.py`、`loss\make_loss.py`、`model\make_model.py`、`utils\metrics.py`、`datasets\sampler.py`、`datasets\sampler_ddp.py`、`datasets\make_dataloader.py`、`solver\scheduler_factory.py`、`solver\cosine_lr.py`、`configs\UAD.yml`
