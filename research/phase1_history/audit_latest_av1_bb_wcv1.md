# 最新三组实验代码机制审计：A-V1 / B-B / WCV1

> 审计日期：2026-10-29
> 性质：只读静态审计 + 既有日志/诊断产物的二次核对；未修改任何代码，未重训。
> 基线：`D:/CODE/Doing/WHU_base/WDV1_CC/WHU`（UAD，DDP，mAP 11.0234 / R1 29.7112，见 `WCV1/logs/diagnosis_120/report.md:20` 与 `WAV1/AV1_DIAGNOSIS_ANALYSIS.md:27-31`）。

路径简写：

| 简写 | 绝对根路径 |
|---|---|
| `A/` | `D:/CODE/Doing/WHU_base/WAV1/` |
| `B/` | `D:/CODE/Doing/WHU_base/WBV1/` |
| `C/` | `D:/CODE/Doing/WHU_base/WCV1/` |
| `U/` | `D:/CODE/Doing/WHU_base/WDV1_CC/WHU/` |

证据等级：
- **[代码事实]**：直接由源码行号确认。
- **[实验支持]**：由训练日志、检索日志、诊断 CSV/JSON 直接读出。
- **[机制假设]**：代码 + 数据现象支持的推断，未做单变量消融，不能当作已证实因果。

---

## 0. 结论摘要

| 实验 | 末轮 mAP / R1 | 相对 UAD | 最关键的 3 条机制结论 |
|---|---:|---|---|
| A-V1 | 10.6032 / 27.6503 | mAP −0.42 pp，R1 −2.06 pp | ① Pair/Mixed 只在单卡 16-PID batch 内取负例（15/499≈3% 身份覆盖），MP 虽覆盖 500 个原型但只是原型级分类；② 新损失无梯度预算/衰减，标量占比由第 7 轮约 11% 升到第 120 轮约 34.5%，MP 独占约 20%；③ 诊断显示 463/472 个 R1 新失败的最危险负样本相似度上升，而全体负对均值反而下降，说明优化的是 batch 平均而非全库最大值。 |
| B-B | 10.87 / 26.87 | mAP −0.15 pp，R1 −2.84 pp | ① query-risk/负身份分层把负例预算压到“batch 内最危险身份/图像”（实测最危险负身份平均拿约 19.4% 的身份预算；单个最危险图像对的理论上限约 10%，取决于身份内 softmax 集中度），同卡只有 15 个负身份≈3% 全库身份；② batch 内 top-1 被彻底解决（top_error 94.6%→0.75%，gap −0.12→+0.32），但全库 R1 低于基线 2.84 pp，属于候选集不可迁移而非优化失败；③ 固定权重 85.807006 来自 ImageNet 上的一次性 BB0 标定，全程不衰减；B-B 标量始终占总损失约 39–41%，而自适应 α 在末期已降到 0.012（风险分支对 99% Query 关闭）。 |
| WCV1 | 11.11 / 28.38 | mAP +0.09 pp，R1 −1.33 pp | ① 正例吸引主导辅助梯度：feature 级正/负梯度范数比约 1.9×，aux 与正例方向 cos≈0.86、与负例 cos≈0.23；候选被 detach，只有 anchor 端回传，负侧压制进一步变弱；② 4 卡 gather 后只有 64 个身份/768 个候选，危险负身份曝光率约 11.9%，危险图像约 1.0%；batch 内“分离正确但全库错误”率从 4.95% 降到 2.46%，全库 R1 仍下降；③ rho=0.10 只约束梯度范数、不约束方向（grad_cosine 0.30–0.33），且不衰减；标量占比从第 11 轮约 4% 升到第 120 轮约 44.5%（feature 梯度比仍是 10%）。 |

三组共同的失败骨架：**新损失优化的是“PKM batch 内、平均/软排序”的目标，而 R1 由“全 gallery、每 Query 最大值”决定；mAP 对全体正例的平均更宽容，因此出现 mAP 小升/持平而 R1 下降。**

---

## 1. 统一调用链

### 1.1 A-V1

```text
train.py
 └─ make_dataloader
     ├─ PKMSampler_DDP → AV1Sampler(A/WHU/datasets/av1_sampler.py)
     │    ├─ 物化原 PKM 实际 batch（:36-45）
     │    ├─ PID/模态 K-group 相机重排（:48-71）
     │    └─ epoch≥11 的跨 batch PID-group 软困难交换（:74-110）
     └─ full_train_loaders（确定性 val transform，A/WHU/datasets/make_dataloader.py:145-150）
 └─ processor.do_train(A/WHU/processor/processor.py:93)
     ├─ 模型前向：`cls_score, global_feat, feat = model(imgs, target, camids)`（:251-252）
     ├─ 原损失 loss_fn（CE + WRT + ProCA + GPD）
     ├─ MP：modality_prototype_loss(global_feat, target, mp_bank)（:283-291）
     ├─ Pair/Mixed：image_contrast_loss(global_feat, target, camids, sample_ids)（:287-288）
     ├─ loss += ramp·(0.1·MP + 0.1·(0.75·Pair+0.25·Mixed))（:289-291）
     ├─ backward → unscale → clip(1.0) → step（:293-298）
     ├─ 成功步才更新 GPD/MP Bank（:299-314）
     ├─ epoch10：extract_centers + hard_matrix → hard_reference_epoch010.pt（:361-384）
     └─ epoch11 起把固定 hardness 交给 AV1Sampler（:218-222）
```

### 1.2 B-B

```text
train.py
 └─ make_dataloader（PKM，无 gather）
 └─ processor.do_train(B/WHU/processor/processor.py:91)
     ├─ 模型前向（:190-191）
     ├─ camera_rep = torch.cat(camids)（:188）
     ├─ make_loss.loss_func → BBRankingLoss(feat, target_rep, camera_rep)（B/WHU/loss/make_loss.py:60-74）
     ├─ 原 PCA/GPD 仍作用于 global_feat（:201-224）
     ├─ backward → unscale → clip(1.0) → step（:226-230）
     └─ proto_bank.update（:232-234，无 AMP 成功步门控）
```

### 1.3 WCV1

```text
train.py
 └─ make_dataloader
     ├─ ImageDataset(return_metadata=True) 产出 4 元组 (imgs, pid, camids, sample_keys)
     └─ PKMSampler_DDP
 └─ processor.do_train(C/WHU/processor/processor.py:106)
     ├─ 模型前向（:221-222）
     ├─ 用 model_meta.wcv1_raw_feature 覆盖 global_feat（:223-227）
     │  └─ 该节点在 model/make_model.py:71-75 捕获，是 BN 之前的 LayerNorm 后 CLS
     ├─ 原损失 loss_fn（CE + WRT + ProCA + GPD）作用于 raw feature（:228-255）
     ├─ epoch>10：mixed_rank_loss（:257-268）
     ├─ budgeted_auxiliary_loss（:269-274）
     ├─ scaler.scale(loss).backward → clip(1.0) → step（:276-280）
     └─ proto_bank.update（:282-284，无 AMP 成功步门控）
```

---

## 2. 新损失的精确数据流

### 2.1 共用特征

| 项目 | A-V1 | B-B | WCV1 |
|---|---|---|---|
| 监督特征 | `global_feat`：ViT 最后 LayerNorm 后、BNNeck 前 768D CLS（`A/WHU/model/make_model.py:68-77`） | 同左（`B/WHU/model/make_model.py` 同结构） | 同左，且显式取 `wcv1_raw_feature`（`C/WHU/model/make_model.py:71-75`） |
| 推理特征 | `NECK_FEAT=before`，同一 pre-BN CLS（A configs/A1.yml:61） | 同左（B configs/BB0.yml:55，BB1.yml:55） | 同左（C configs/WCV1.yml:53） |
| 归一化 | `F.normalize(features.float(), dim=1)` 后 reshape（A/WHU/loss/av1.py:49,92） | `F.normalize(features.float(), p=2, dim=1, eps=1e-12)`（B/WHU/loss/bb_ranking_loss.py:40） | `F.normalize(features.float(), dim=1)`（C/WHU/loss/mixed_rank.py:40） |
| FP32/AMP | 损失算术在 `autocast(enabled=False)`，但输入是 AMP 产生的 FP16 特征再 `.float()`（A/WHU/processor/processor.py:284-288） | 同左（B/WHU/loss/bb_ranking_loss.py:37-40） | 同左（C/WHU/loss/mixed_rank.py:39-40） |
| 推理/训练特征是否一致 | 是 | 是 | 是 |

说明：三组都做到了“训练监督特征 = 最终检索特征”，没有额外可学习分支，也没有把 BN 后特征用于新损失。FP32 只保证损失算术与 logits 的精度，ViT 输出在 AMP 下仍是 FP16，`.float()` 不会恢复已损失的低位精度；这是 AMP 常规行为，不是本组独有 bug。

### 2.2 候选集合、正负样本、掩码

| 项目 | A-V1 | B-B | WCV1 |
|---|---|---|---|
| 单卡特征量 | 3×64=192 行（日志 `actual images per GPU=192`） | 3×64=192 行 | 3×64=192 行 anchor |
| 候选池（损失内） | Pair：源 64 × 目标 64；Mixed：源 64 × 3×64=192（`av1.py:100-110`） | 全部 192×192（`bb_ranking_loss.py:41-45`），**不跨卡** | anchor 192 × 全局 gather 768（`mixed_rank.py:42-53`） |
| 负身份数 | 本卡 15 个（16 PID batch）→ 15/499≈3.0% | 本卡 15 个→3.0% | 全局 63 个→63/499≈12.6% |
| 正样本定义 | 同 PID、不同 camera、不同 sample_id（`candidate_mask`，`av1.py:68-77`） | 同 PID、不同 camera（`bb_ranking_loss.py:42-45`） | 同 PID、不同 camera、不同 sample key（`mixed_rank.py:68-71`） |
| 负样本定义 | 不同 PID、不同 camera、不同 sample_id | 不同 PID、不同 camera | 不同 PID、不同 camera、不同 sample key |
| 同相机掩码 | 有（Pair 和 Mixed 都有） | 有 | 有 |
| 自匹配/重复图 | 有：sample_id = 模态偏移 + 原始索引，Mixed 排除同 ID；Pair 跨模态 ID 天然不同（`A/WHU/datasets/bases.py:88-107`） | **没有显式 sample key**；依赖“自身与自身同相机”来排除自匹配；同图重复 occurrence 也因同相机被排除 | 有：按 `(modality, path)` 构造 key，跨 rank 去重保留首个，同 key 排除（`C/WHU/datasets/bases.py:79-101`，`mixed_rank.py:55-66`） |
| 平台掩码 | 无（训练只按 camera 不等式；评测的 Ground/Aerial 只用于分组诊断） | 无 | 无 |
| 缺失正/负样本 | 行无效、跳过；不会进 logsumexp（`av1.py:78-87`） | `valid=(pos_counts>0)&negative.any(1)`（`:46-47`） | `valid=positive.any(1)&negative.any(1)`（`mixed_rank.py:72`） |

关键点：

1. **[代码事实]** A-V1 的 Pair 和 Mixed 都只用当前 rank 的 192 个特征，**没有跨卡 gather**；MP 的原型 Bank 用 `all_reduce` 合并，但合并的是“身份原型”，不是图像候选（`A/WHU/loss/av1.py:32-36`，`A/WHU/processor/processor.py:313-314`）。
2. **[代码事实]** B-B 明确注释“All tensors here are local to the rank; no prototype bank or feature gather is used”（`B/WHU/loss/bb_ranking_loss.py:68-70`）。
3. **[代码事实]** WCV1 是唯一做跨卡候选 gather 的方法：`all_gather` 特征与 metadata，先检查各 rank 形状一致，再做 sample-key 去重（`C/WHU/loss/mixed_rank.py:42-53`）；anchor 仍是本卡 192 行。
4. **[推导]** PKM 全局 batch 为 256 个“图像三元组”，即 64 个训练身份；每卡 64 个三元组 = 16 个身份。因此 A/B 的每 Query 负身份数约 15（3.0% of 499），C gather 后约 63（12.6% of 499）。C 的诊断实测危险身份曝光率 11.9%，与 63/499 一致（`C/logs/diagnosis_120/report.md:141-152`）。

### 2.3 损失公式与权重

#### A-V1

- MP：六个跨模态方向 `softmax(u_query·P_target/0.1)` 等权平均 + `0.1×TIR→TIR`；Bank `[3,500,768]` FP32 buffer + valid mask，无梯度（`av1.py:47-65`）。
- Pair/Mixed：`S = u_q u_g^T / 0.1`；
  - `uni = logsumexp_all_valid - mean_positive_logit`；
  - `set = logsumexp_all_valid - logsumexp_positive`；
  - `L_img = 0.3·uni + 0.7·set`（`av1.py:84-87`）；
  - `L_pair = mean(6 个方向)`，`L_mixed = mean(3 个源模态)`，`L_img_total = 0.75·L_pair + 0.25·L_mixed`（`av1.py:113-115`）。
- 总新增：`ramp·(0.1·MP + 0.1·(0.75·Pair+0.25·Mixed))`，`ramp=min(epoch/5,1)`（`processor.py:289-291`）。
- 实际标量（末轮 Iter100，`A/logs/train_log_20261026_1121.txt:1454-1458`）：
  - MP `0.9706×0.1=0.0971`；
  - Pair `0.6101×0.075=0.0458`；
  - Mixed `0.9158×0.025=0.0229`；
  - 新增合计 `0.1657`；原损失合计约 `0.126+0.103+0.01×0.161+0.084=0.3146`；
  - **新增占总标量约 34.5%，其中 MP 20.2%、Pair 9.5%、Mixed 4.8%**。
  - 第 7 轮对照（`:155-157`）：新增 `0.715` / 总 `6.443`≈11.1%。即固定权重 + 只升不降的 ramp，使新增损失在后期相对权重升至约 3 倍。

#### B-B

- `a_qp = (1-α_q)/|P_q| + α_q·1[p=p_q*]`；`p_q*` 取 **最大相似度的正样本（最容易正例）**（`bb_ranking_loss.py:56,64-66`）。
- `α_q = α_max·sigmoid(-gap/0.05)`，`gap = max_pos - max_neg`，`α_max=0.5`，只在 `valid` 行生效（`:58-63`）。
- 负身份：`c = 0.5·uniform(J_q) + 0.5·softmax(每个负身份的最大相似度 / 0.1)`（`:70-80`）。
- 负身份内图像：`d = 0.5·uniform(K_qj) + 0.5·softmax(图像相似度 / 0.1)`（`:81-86`）。
- 最终：`b = Σ_j c_qj·d_qn|j`，`loss = Σ_{p,n} a_qp·b_qn·0.05·softplus((0.05+s_qn-s_qp)/0.05)`（`:91-101`）。
- 实际权重：CLI 覆盖 `MODEL.BB_LOSS_WEIGHT=85.807006`（`B/logs/train_log_20261028_1111.txt:2`），而仓库内 `BB0.yml:6`、`BB1.yml:6` 仍是 `0.0`；`make_loss.py:40-41` 对 `<=0` 直接抛错，所以**两份 yml 不能原样启动**。
- 实际标量：
  - 第 1 轮（`:106`）：BB 加权 `6.9937`；原损失 `6.212+0.01×0.453+4.832=11.05`；BB 占 `38.8%`。
  - 第 120 轮（`:1337-1338`）：BB 加权 `0.1557`；原损失 `0.137+0.01×0.160+0.087=0.2256`；BB 占 `40.8%`。
  - `bb_raw` 从 `0.0821` 降到 `0.0018`（约 46×），但固定权重使 BB 标量份额几乎不变。
- 分层集中度：末轮 `negative_identity_budget_max_mean=0.1935`（`:1338`），即最危险负身份拿约 19.4% 的身份预算。身份内再按 `d = 0.5/K + 0.5·softmax` 分配，单个 (身份, 图像) 对的 `b` 预算上限约为 `c_max×(0.5+0.5/K)`；若身份内 softmax 完全集中到一张图（K 为该身份候选图数，约 12），上限约 `0.19×0.54≈10%`（未再乘 softplus 饱和因子）。

#### WCV1

- 候选：本地 anchor `z`（可导）× gather 后的候选 `candidate=z.detach()`（不可导），`logits=z·candidate^T/0.07`（`mixed_rank.py:40-41,67`）。
- `v = logsumexp(负例 logits)`；`terms = softplus(v - l + margin/τ)`，`margin=0`（`:84-85`）。
- 正例按“目标模态组”取均值：每个模态组内 `1/pc`，再对模态组数取均值（`:86-92`）。
- 每行归一化：`world·ell.sum()/(active·global_counts[m])`，`active=3`（`:93-97`）。
- 预算：`weight = ramp·min(1.0, ρ·sqrt(e0/ea))`，`ρ=0.10`，`e0=||∂base/∂F||²`，`ea=||∂aux/∂F||²`（`mixed_rank.py:125-141`）；`ramp=(epoch-10)/10`（`processor.py:269`）。
- 实际标量/梯度：
  - 第 11 轮（`C/logs/train_log_20261028_2339.txt:293`）：`aux_loss=4.877`，`weight=0.0342`，`grad_ratio=0.01`，`grad_cosine=0.464`，`saturation=0.907`；
  - 第 120 轮（`:1418-1419`）：`aux_loss=1.027`，`weight=0.2233`，`grad_ratio=0.10`，`grad_cosine=0.324`，`saturation=0.070`；
  - 末轮标量：aux `0.2233×1.027=0.2293`，原损失 `0.115+0.085+0.01×0.158+0.084=0.2856`，aux 占 **44.5%**；但 feature 级梯度比例由预算强制为 **10%**。标量份额与梯度份额不是一回事，报告和日志必须分开读。
  - 梯度探针（`C/logs/diagnosis_120/probes/gradient_pairs.csv`，16 组中位数）：
    - WCV1 feature_all：`positive` 梯度范数 `0.0958`，`negative` `0.0510`（比值 1.88）；
    - `aux` 与 `positive` 的 cos `0.860`，与 `negative` 的 cos `0.234`；
    - last_block_parameter：aux/base 范数比 `1.012`（未乘 weight），cos `0.154`；negative/positive 参数梯度范数比约 `1.08`（negative 0.2747 / positive 0.2540），cos `-0.827`。

---

## 3. 反向梯度流

| 项目 | A-V1 | B-B | WCV1 |
|---|---|---|---|
| 回传共享 ViT 的项 | CE、WRT、ProCA、GPD、MP、Pair、Mixed 全部 | CE、BB、ProCA、GPD 全部 | CE、WRT、ProCA、GPD、mixed-rank 全部 |
| Query/Gallery 两端 | Pair/Mixed 两端都不 detach；同一图像可作为 Query 和其它 Query 的 Gallery 累加梯度（`av1.py:73-87,101-109`；文档 `A/CODE_CHANGE_RECORD.md:57`） | 只有 pair loss 的 `similarities` 可导；`a,b,α,gap` 来自 `similarities.detach()`（`bb_ranking_loss.py:52-87`） | **候选 detach，仅 anchor 端可导**（`mixed_rank.py:41,67`）；每个特征只通过自己那一行获得 aux 梯度 |
| Bank | MP Bank 是 buffer，无梯度；GPD Bank 无梯度；PCA `g_cent` detach | GPD Bank 无梯度；PCA `g_cent` detach | GPD Bank 无梯度；PCA `g_cent` detach |
| 跨卡 | MP Bank 的原型/计数 `all_reduce`；**图像特征不跨卡** | **不跨卡** | **候选特征/元数据 all_gather**；anchor 不跨卡 |
| 梯度预算 | 无；固定 λ | 无；固定 85.807 | 有；feature 级 ρ=0.10，`max_weight=1.0` |
| 裁剪 | `clip_grad_norm_(model.parameters(), 1.0)`（`processor.py:295`） | 同左（`:228`） | 同左（`:278`） |
| 新增可训练参数 | 无；Bank 是 buffer，不进入 optimizer | 无；`BBRankingLoss` 无参数 | 无；无新参数 |
| AMP/成功步一致 | 有 `step_ok` 门控；跨 rank MIN/MAX 检查（`:299-308`）；Bank 只在成功步更新（`:310-314`） | **没有 `step_ok` 门控**：`scaler.step/update` 后无条件 `proto_bank.update`（`:226-234`） | **没有 `step_ok` 门控**：同 `:276-284` |

结论：

1. **[代码事实]** 三组的新损失都没有引入新可训练参数；MP/GPD Bank 是 buffer，`optimizer` 不会更新它们。所谓“新增参数是否进入 optimizer”在本轮三组均为“不适用（无新增参数）”。
2. **[代码事实]** A-V1 的对称 Pair/Mixed 与 WC 的 anchor-only 是两种完全不同的梯度几何：A-V1 同一张图既被拉/推，也作为别人的 gallery 被推；WC 的候选完全 detach，repulsion 只发生在 anchor 一侧。
3. **[代码事实]** B-B 的权重 `a,b` 用 detach 的相似度计算，pairwise softplus 仍可导；这与“只在权重上 stop-gradient”一致，不是梯度断流。
4. **[机制假设]** WC 的 anchor-only + 正例组均值，使每步梯度更接近“把 anchor 拉向正例均值”，而不是“对称地把正负两侧分开”；这与探针的正例主导方向一致。

---

## 4. 训练–测试一致性

### 4.1 一致的部分

- **[代码事实]** 三组最终检索特征都是 pre-BN CLS，且损失也作用在同一 pre-BN CLS：A `model.forward` 返回的 `global_feat`、B 同、C `wcv1_raw_feature`。`NECK_FEAT=before` + `FEAT_NORM=yes` 在 A/B/C 配置中一致。
- **[代码事实]** 同相机排除规则与 `U/WHU/utils/metrics.py` 一致：评测对每个 Query 删除所有同 camera 的 Gallery（`metrics.py:161-164`），训练三组都用 `camera_q != camera_g`。
- **[代码事实]** 正/负样本都按 PID 划分，不引入额外代理分类作为最终排序；MP/GPD 只是训练期正则，不参与推理。
- **[代码事实]** 自匹配：A 用 sample_id，C 用 sample key；B 依赖“自图同相机”排除。未发现直接的 self-positive 泄漏。重复图像：A 的 sample_id 在同模态内唯一；C 用 path key 去重；B 的同图重复也因同相机被排除。

### 4.2 不一致的核心：候选集合

| 维度 | 训练 | 评测 | 后果 |
|---|---|---|---|
| 身份数 | A/B 每卡 16 PID（15 负身份）；C gather 后 64 PID（63 负身份） | 500 个测试 PID，93,609 张 Gallery | A/B 仅覆盖约 3% 的负身份，C 约 12.6% |
| 图像数 | A/B 192；C 192 anchor vs 768 candidate | 6,405 Query × 93,609 Gallery | 每 Query 的负数从 15–63 增至数万 |
| 正例数 | B 日志 mean 5.6；C 日志 mean 6–9；A 未直接记录，按 4 张/模态×2 目标模态减同相机估计约 5–8 | 每身份 Gallery 数十张（同相机排除后） | 训练只是对全库正例分布的小样本估计 |
| 负例定义 | 同 camera 全部排除，包括同 PID 与不同 PID | 同 camera 全部排除 | 定义一致 |
| 候选更新 | A/B 不跨卡；C 只 gather 候选，不 gather anchor 梯度 | 全库一次性排序 | 优化目标不是全库 AP/R1 的无偏估计 |

**[代码事实]** 三组都没有“跨 batch/全库负例记忆”用于图像级对比：A 的 MP Bank 和 GPD Bank 是身份原型分类器，不是以真实 Gallery 图像为负例的队列；B 没有 Bank；C 的候选只到本步 768 行。

**[实验支持]** WC 的免重训诊断直接量化了这个缺口（`C/logs/diagnosis_120/report.md:133-160`）：

- UAD / WCV1 的 `danger_identity_present_rate` 均为 `0.1195 / 0.1190`，危险图像仅 `0.0103 / 0.0104`；
- `batch_separated_full_wrong_rate` 从 `0.04948` 降到 `0.02446`：batch 内能分开，但全库仍错的比例只降了一半，仍有约 2.45%；
- 这解释了“batch 内 top-1 改善、全库 R1 不改善”的同时存在。

### 4.3 是否泄漏

- **[代码事实]** 未发现同相机正例泄漏：三组的 positive 都带 `camera_q != camera_g`。
- **[代码事实]** 未发现自匹配泄漏：A/C 显式 sample ID/key；B 的自匹配因同 camera 被排除。
- **[代码事实/数据假设]** 若同一物理图像在同一模态内以不同 camera ID 重复入库，A/B 的 camera-only/ID-based 规则可能无法把它识别为同一图；C 的 path key 可识别同路径重复。WHU-MARS 的正常数据结构下未观察到这种重复，但这属于数据前提，不是代码保证。
- **[代码事实]** A 的 Pair 跨模态方向中 sample_id 永远不会相等（不同模态偏移不同），因此“同 sample ID 排除”对 Pair 实际不起作用；它只在 Mixed 的同模态源（source=gallery 中同模态块）起作用。这不是 bug，但说明 Pair 的“自匹配排除”完全依赖 camera，而不是 sample ID。

---

## 5. 逐方法失败机制

### 5.1 A-V1

#### 5.1.1 实际权重与有效 Query 数

- **[实验支持]** 末轮每个 rank 每步：MP 7 个方向各 64 个有效 Query（`A/logs/train_log_20261026_1121.txt:1458`）；Pair 六方向 53–57/64；Mixed 三源各 60/64。全局约 ×4 rank。不是“有效 Query 过少”。
- **[实验支持]** 标量权重：MP 0.1、Pair 0.075、Mixed 0.025；末轮贡献 0.097/0.046/0.023，新增合计 0.166，占总损失 0.480 的 34.5%（第 7 轮仅 11.1%）。
- **[代码事实]** MP 是新损失里唯一有“500 个身份 × 3 模态”全局负类的项；Pair/Mixed 只覆盖本卡 15 个负身份。MP 末轮占新增标量的 58.6%，即**主导的新增项是全局原型分类，而不是图像级 pair/mixed**。

#### 5.1.2 最危险负样本为什么上升

- **[实验支持]** `A/AV1_DIAGNOSIS_ANALYSIS.md:67-79`：
  - 472 个 R1 新失败中，463 个（98.1%）最危险负样本相似度上升；267 个（56.6%）是“正样本未下降、负样本上升”的纯负侧失败，196 个两侧共同恶化，9 个纯正侧；
  - 新失败组平均 `Δs+=+0.00575`、`Δs−=+0.05859`、`Δmargin=−0.05284`；margin 由 +0.02259 变为 −0.03026；
  - 全体 Query 平均 `Δs+=+0.02909`、`Δs−=+0.03911`；
  - **全体有效负图像对的近似均值反而下降**（同模态 0.0211→−0.0019，跨模态 0.0200→−0.0030），但 99 分位上升（同模态 0.3525→0.3675，跨模态 0.3325→0.3425）。
- **[代码事实]** Pair/Mixed 的 `uni`/`set` 分母都是 `valid` 全集，**正样本也在分母里**（`av1.py:84-86`）。在 τ=0.1 下，用诊断给出的平均相似度粗算：
  - Pair：每个 Query 只有目标模态的约 3–4 个正例（K=4 张目标模态图减去同相机），负例约 60；跨模态正例均值约 0.2655，负例约 0.02；`exp(0.2655/0.1)≈14.2`，`exp(0.02/0.1)≈1.22`；正例质量约 `3.5×14.2≈50`，负例约 `60×1.22≈73`，正例占约 36–44% 的 softmax 质量；
  - Mixed：正例约 8–10 个（同模态约 2–3 个，均值约 0.3493，`exp(3.493)≈32.9`；跨模态约 6 个，均值约 0.2655），负例约 180 个；正例质量约 `2.5×32.9+6×14.2≈167`，负例约 `180×1.22≈220`，正例仍占约 43%。
  - 这只是用诊断均值/候选规模做的量级估算，不是逐样本精确值；但它说明：**分母中正例吸收了约 36–44% 的概率质量，负例的“压制”梯度比 negatives-only 分母小。**
- **[代码事实]** MP 无相机掩码（`av1.py:47-65`），Bank 聚合一个身份在一整个模态下的所有图像；Pair/Mixed 有相机掩码。也就是说，占新增损失 58.6% 的 MP 项优化的是“所有相机混合的身份原型分类”，不是评测所用的“跨相机图像排序”。
- **[代码事实/实验支持]** 相机多样性重排几乎无效：日志 `cameras_before≈2.0`、`cameras_after≈2.02–2.05`（`A/logs/train_log_20261026_1121.txt:110,227,1459`），每轮最多约 +0.03 个相机/K-group。Epoch 1–10 无 swap；epoch 11 起每轮约 645/1824–1840 组（约 35%）做困难交换（`:227,1459`）。
- **[代码事实]** 困难矩阵固定在 epoch10（`processor.py:361-384`），之后 110 个 epoch 不更新；它反映的是 epoch10 的身份困难度，不是收敛期的全库困难度。
- **[机制假设 1：候选集不可迁移]** 新损失在 15 个负身份内把 batch 排名做好（有效 Query 44–64/64），但评测的最大负例来自 500 个身份的 93,609 张图。对 batch 内的“平均负例”下压，不能控制全库最大负例；诊断中均值下降、P99 上升正是这种“平均改善、尾部变差”的几何。
- **[机制假设 2：正例吸收分母 + 无全局图像级负例]** Pair/Mixed 的 logsumexp 分母被高相似正例占据，负例梯度份额被压缩；MP 虽有全局身份覆盖但只是原型均值，不能约束“某个错误身份的某张图”成为全库 top-1。因此最危险负例仍有上升空间。
- **[机制假设 3：采样器混淆]** 相机重排实际几乎没改变 K-group 相机数，却改变了 ProCA 的组内中心构成（文档 `A/CODE_CHANGE_RECORD.md:63` 也承认会改变 ProCA 中心）；困难交换又改变 PID 共现。A-V1 同时改了损失、采样器、ProCA 分组三个因素，无法把退化归给单一损失。

#### 5.1.3 为什么 R1 降幅大于 mAP

- **[实验支持]** R1 29.7112→27.6503（−2.06 pp），mAP 11.0234→10.6032（−0.42 pp）。
- **[实验支持]** 472 个新失败中 354 个仍在前五、420 个仍在前十，只有 52 个连 Top-10 也丢失（`A/AV1_DIAGNOSIS_ANALYSIS.md:47-49`）。因此 R1 的二值阈值下降很大，AP 的损失被“仍在前列的正例”和 340 个新成功部分抵消（AP 贡献分解 `:53-62`：新失败 −0.347、新成功 +0.190、持续正确 −0.199、持续错误 −0.064）。
- 这是典型的“顶部次序互换”：多个正确图像仍在，AP 只小幅变化；但第一名被错误负例抢走，R1 严格下降。

---

### 5.2 B-B

#### 5.2.1 实际权重与 batch 内效果

- **[实验支持]** 末轮 `valid_query_ratio=0.9931`，`positive_count_mean=5.725`，`negative_identity_count_mean=14.979`，`valid_query_count=190.68/192`（`B/logs/train_log_20261028_1111.txt:1338`）。有效 Query 充足。
- **[实验支持]** batch 内 top-1 已被解决：
  - `top_gap_mean` 从 `-0.1225` 升到 `+0.3212`；
  - `top_error_ratio` 从 `0.9456` 降到 `0.0075`；
  - `bb_raw` 从 `0.0821` 降到 `0.0018`；
  - `alpha_mean` 从 `0.4231` 降到 `0.0116`，`alpha_near_max_ratio` 从 `0.5150` 降到 `0.0005`。
  - 含义：训练后期 99% 的 Query 在 batch 内已经“最差正例 > 最好负例”，**query-risk 正样本预算几乎关闭**（`alpha` 只对约 1% 的困难 Query 有效）。
- **[实验支持]** 负身份分层确实在收紧：`negative_identity_budget_max_mean` 从 `0.1185` 升到 `0.1935`（`:106,1338`），最危险负身份拿到的身份预算约从 11.9% 升到 19.4%。身份内再 50% 给最难图像，所以梯度进一步集中到“最危险身份的最危险图像”。
- **[代码事实]** 候选池只有本卡 192 行、15 个负身份，**没有跨卡 gather**（`bb_ranking_loss.py:68-70`）；每个 Query 的负身份覆盖为 15/499≈3.0%。
- **[代码事实]** 正样本风险预算给的是 `best_pos_index`，即**最大相似度、最容易的正样本**（`:56,64-66`），不是最远/最难正样本。设计意图是“当最安全的正例都输给负例时，优先救 top-1”；但在 mAP 语境下，它不会改善那些“最容易正例仍赢、但 hardest positive 排名很差”的 Query。

#### 5.2.2 “是否把梯度集中在最危险负样本，从而抬高全体负样本相似度？”

- **[代码事实]** 是“集中”，但不是“抬高全体”：
  - `c` 的 50% 质量按身份最危险相似度分配，`d` 的 50% 质量按身份内最危险图像分配；实测最危险负身份平均拿约 19.4% 身份预算，单个最危险图像对的 `b` 预算上限约 10%（取决于身份内 softmax 集中度）；
  - 其余约 90% 预算仍由 uniform 和 softmax 尾部覆盖，但 `b` 的 softmax 温度 0.1 会让高相似负例获得大部分质量；
  - 另外 softplus 在误差 > ~3τ（τ=0.05，即余弦 gap >0.15）后梯度近似饱和，**简单负例几乎不再被显式约束**。
- **[实验支持]** B-B 只在自己的 batch 目标上可验证：top_error 0.75%、gap +0.32；**没有 B-B 的逐 Query 全库诊断**（本目录只有训练日志，没有类似 WCV1 的 ranking/coverage/probes 输出），所以“全体负样本相似度上升”这一说法不能从现有产物直接证实。
- **[机制假设]** 更准确的说法是“**只压低 batch 内最危险的少数负例，未压低全库其余负例**”。在 15 个负身份上把最危险对压下去后，未加权/未见过的全库负例由其它损失和特征几何决定；R1 由全库最大负例决定，因此 batch 内 top-1 的改善不能迁移。若全库负例相似度整体上移，更可能是“特征几何被少数困难对牵引”的副产物，而不是损失显式抬高负例。
- **[实验支持]** 同一 PKM 采样器的候选覆盖率可直接解释“batch 内解决、全库不解决”：WC 诊断的 `danger_identity_present_rate≈11.9%` 是 64 身份 batch 的数字；A/B 的每卡 batch 只有 16 身份，危险负身份在**本卡候选**中的概率约 15/499≈3.0%。B-B 的 15 个负身份里出现全库真正 top-1 竞争者的概率极低。

#### 5.2.3 固定大权重与自适应关闭

- **[代码事实]** `BB_LOSS_WEIGHT=85.807006` 是一次性标定值；标定脚本用 `pos_mode='uniform'` 的 BB0，以 `WRT_fixed` 梯度中位数 / BB0 梯度中位数求系数（`B/WHU/calibrate_bb.py:40-47,99-113`）。实际跑的是 BB1 `adaptive`，但沿用同一系数（`:113` 注释明确“same measured coefficient for BB0 and BB1”）。
- **[实验支持]** BB 标量占比从第 1 轮 38.8% 到第 120 轮 40.8%，基本不变；而 `bb_raw` 下降了 46×。这说明固定系数让一个“几乎已经满足”的 ranking 项持续占总标量的约 40%，其梯度虽然可能随 loss 缩小，但**没有任何衰减/停止条件**。
- **[实验支持]** 自适应 α 在末期几乎关闭（0.0116，near-max 0.0005）；因此后期 B-B 实际退化为“uniform 正例预算 + 困难负身份/图像加权”的 pairwise ranking，query-risk 机制只对约 1% 的 Query 生效。
- **[机制假设]** 早期（epoch1）B-B 占 38.8% 标量、top_error 94.6%、α 平均 0.423，且特征还是 ImageNet 初始化；大权重在早期强烈塑形，可能是 BB 曲线启动慢（epoch10 仅 7.83/21.50，epoch40 才 10.13/26.43）的原因之一。后期它又在已饱和的 batch 尾部持续施加梯度，收益很小但可能改变全局几何。

#### 5.2.4 为什么 R1 降而 mAP 基本持平

- **[实验支持]** R1 29.7112→26.87（−2.84），mAP 11.0234→10.87（−0.15）。mAP 几乎持平，R1 明显下降。
- **[实验支持]** 分协议（`B/logs/train_log_20261028_1111.txt:1345-1352`）：
  - 同模态 R1 全降：RGB→RGB 38.13（UAD 42.48）、NIR→NIR 26.74（27.96）、TIR→TIR 19.53（20.75）；
  - mAP 有升有降：NIR→RGB 15.82（UAD 15.66）、TIR→RGB 6.12（5.85）、NIR→NIR 17.12（16.83）上升；RGB→RGB 25.49（26.53）、RGB→NIR 15.82（16.07）下降。
  - 这符合“平均 pair 排序（mAP）基本守住，但 top-1 边界（R1）被全库困难负例击穿”。
- **[机制假设]** B-B 的软排名目标是 pairwise softplus，平均后会奖励“大多数 pair 的间隔”；`a,b` 的困难加权又只覆盖 batch 内 15 个负身份。全库 R1 需要压住每个 Query 的最大负例，而该负例大多不在 batch 内，故 R1 先掉，mAP 因大量中等难度负例仍被正确排序而基本不动。

---

### 5.3 WCV1

#### 5.3.1 负池、正例组均值、梯度预算

- **[代码事实] 负池**：`v = logsumexp` 只在负例上（`:84`），所以负例概率质量恒为 1，这一点比 A-V1 的“正例在分母”更干净；但候选是 `z.detach()`（`:41`），**候选端不回传**，repulsion 只发生在 anchor 端。
- **[代码事实] 正例组均值**：每个 anchor 的正例先按目标模态分组，组内 `1/pc`，再对组数平均（`:86-92`）；这使每个 anchor 的正例吸引力总量约为 1/组数×组数=1，而每个 anchor 的负例总排斥量也为 1；但因为正例方向一致、负例方向互相抵消，feature 级正例梯度范数约为负例的 1.9×（探针中位数），且 aux 与正例 cos 0.86、与负例 cos 0.23。
- **[代码事实] 梯度预算**：`ρ=0.10` 只限制 `||g_aux||/||g_base||`（feature 级），不限制方向；`grad_ratio` 日志在 ramp 完成后恒为 0.10，`grad_cosine` 仅 0.30–0.33（`C/logs/train_log_20261028_2339.txt:293,1419`）。
- **[代码事实]** `active=3` 的除法（`:97`）让 aux 标量再乘 1/3；预算探针会把这个因子补偿掉，因此 feature 梯度比仍是 0.1，但**报告的 aux_loss 标量比 nominal 小**，造成 44.5% 的标量份额与 10% 的梯度份额并存。
- **[代码事实]** 预算每步用两次 `torch.autograd.grad(..., retain_graph=True)` 再加一次最终 `backward()`（`:125-126,276`），训练成本约翻倍；这是工程代价，不是数值错误。

#### 5.3.2 为什么 mAP 小升而 R1 下降

- **[实验支持]** 总体：mAP +0.09 pp，R1 −1.33 pp，mINP +0.0644 pp（`C/logs/diagnosis_120/report.md:18-20`）。
- **[实验支持]** 分协议（`:21-32`）：跨模态 mAP 多数上升（RGB→NIR +0.22、RGB→TIR +0.23、NIR→RGB +0.29、NIR→TIR +0.26、TIR→RGB +0.36、TIR→NIR +0.26），R1 多数下降；同模态 RGB→RGB mAP 基本不变但 R1 −2.25。
- **[实验支持]** 324 个 R1 新失败 / 239 个修复；新失败分类（`C/logs/diagnosis_120/ranking/summary.json` `overall.new_failure_categories`）：
  - `negative_strengthened = 238`（73.5%）
  - `both_adverse = 60`（18.5%）
  - `positive_weakened = 13`（4.0%）
  - `neither_on_uad_extrema = 13`
  - 即**大多数 R1 新失败是“负例变强”，不是“正例变弱”**。
- **[实验支持]** batch 内可分离性改善但全库不够：
  - `batch_separated_full_wrong_rate` 4.948%→2.446%；
  - `danger_identity_present_rate` 11.95%→11.90%（几乎不变）；
  - `danger_image_present_rate` 1.025%→1.042%。
- **[机制假设 1：AP 与 R1 的关注点不同]** 正例组均值 + uniform 正例预算会改善“多个正例的平均排名”（mAP）和“最差正例的尾部排名”（mINP），但不会专门压低“全库最大负例”。R1 由后者决定，所以 mAP/mINP 升、R1 降。
- **[机制假设 2：候选集太小]** 768 个候选只含 63 个负身份；真正危险的身份/图像只有约 11.9%/1.0% 的概率进入本步候选。batch 内分离正确但全库错误的样本仍占 2.45%，说明改善集中在本步可见的 63 个身份。
- **[机制假设 3：正例吸引主导 + 候选 detach]** anchor-only 梯度与正例组均值使 aux 的方向更接近“拉近正例”，负侧主要靠 logsumexp 的软权重；探针正例/负例范数比 1.88，方向 cos 0.86/0.23。对 AP/R1 的瓶颈——全库最大负例——没有足够的定向压制。
- **[机制假设 4：rho 不约束方向且不衰减]** `grad_ratio=0.1` 是恒定值，`ramp` 只从 0 升到 1，之后不再下降；第 90–120 轮 mAP/R1 仍在缓慢变化，说明 10% 的 off-axis 梯度在收敛期持续施加影响。若改为末期衰减到 0，至少可以保护最后阶段的 top-1 边界（待验证）。

#### 5.3.3 报告与数据的一处矛盾

- **[文档–数据矛盾]** `C/logs/diagnosis_120/report.md:189` 写“新增失败以最佳正例变弱为主，且正例吸引占辅助主导：优先修订正例监督分配”。
- **[实验支持]** 同一目录 `ranking/summary.json` 的 `overall.new_failure_categories` 明确是 `negative_strengthened=238`、`positive_weakened=13`、`both_adverse=60`；即 **73.5% 的新失败是纯负例变强，纯正例变弱只有 4.0%**。报告把“辅助梯度中正例吸引占主导”误写成了“失败以正例变弱为主”。这是报告结论层面的错误，后续决策不应据此优先修改正例监督；应先处理全库负例覆盖/负侧压制。
- 另：报告承认 coverage 是“eval 缓存的训练池覆盖代理，不是历史训练回放”（`:11`），所以 danger exposure 只能作为机制证据，不能当作逐步因果。

---

## 6. R1 与 mAP 的系统性分解

| 方法 | R1 | mAP | 新失败/修复 | 关键转移/诊断 | 解释 |
|---|---:|---:|---|---|---|
| A-V1 | 27.65 (−2.06) | 10.60 (−0.42) | 472 / 340 | 463/472 新失败 max-neg 上升；354/472 仍在前五；AP 贡献 −0.347/+0.190/−0.199/−0.064 | 顶部次序互换；正例大多仍在前列，AP 损失被修复抵消；R1 阈值损失更直接 |
| B-B | 26.87 (−2.84) | 10.87 (−0.15) | 未做逐 Query 诊断 | batch top_error 94.6%→0.75%，gap +0.32；mAP 几乎持平 | batch 内 hard-negative 目标已收敛；全库负例覆盖不足，R1 掉、mAP 守住 |
| WCV1 | 28.38 (−1.33) | 11.11 (+0.09) | 324 / 239 | neg-strengthened 238、both 60、pos-only 13；mINP +0.064；danger id 11.9% | 正例平均/尾部改善（mAP/mINP），全库最大负例未改善（R1） |

**共同结构**：

1. R1 是“每个 Query 的最大负例 vs 最佳正例”的阈值；mAP 对所有正例的 precision 取平均，对少数 top-1 翻转不敏感；mINP 对最后一个正例的位置更敏感，容易与 mAP 同向。
2. 三组训练损失的正例权重/分母都偏向“多个正例的平均”（A 的 uniform 0.3、B 的 0.5/|P|、C 的组均值），而不是“只保护最佳正例的 top-1 边界”。
3. 训练负例覆盖 15–63 个身份，评测最大负例来自 500 个身份、93,609 张图；训练目标的“平均负例”与评测的“最大负例”不是同一个统计量。
4. 因此出现：mAP 小升/持平、mINP 改善、R1 明显下降。

---

## 7. 确定性缺陷、空路径、饱和、数值 dead、权重裁剪、有效 Query

### 7.1 已确认的缺陷/风险（按严重度）

1. **[代码事实][高] B-B 与 WCV1 的 GPD Bank 没有 AMP 成功步门控。**
   - `B/WHU/processor/processor.py:226-234`：`scaler.step(); scaler.update(); if use_aux_modules: proto_bank.update(...)`，没有 `step_ok`。
   - `C/WHU/processor/processor.py:276-284`：同样无条件更新。
   - A-V1 已修复：`A/WHU/processor/processor.py:299-314` 用 `old_scale/new_scale` 判定成功步，且跨 rank MIN/MAX 校验。
   - 影响：AMP overflow 跳步时，参数没更新但 Bank 前进，训练状态与记忆不一致。日志中 A-V1 有 8/13696 次 skip；B/C 未记录 skip 数，无法量化但机制存在。
2. **[代码事实][高] 仓库配置与实际运行不一致，且 B-B 的 yml 不能原样启动。**
   - `B/configs/BB0.yml:6`、`BB1.yml:6` 的 `BB_LOSS_WEIGHT=0.0`；`B/WHU/loss/make_loss.py:40-41` 对 `<=0` 抛错。实际运行通过 CLI 传 `85.807006`（`B/logs/train_log_20261028_1111.txt:2`）。
   - `C/configs/WCV1.yml:37-38` 是 `BASE_LR=0.008, IMS_PER_BATCH=64`；实际运行是 `256 / 0.032`（`C/logs/train_log_20261028_2339.txt:2`）。
   - 影响：复现实验若只读 yml 会得到错误配置或直接报错；标定系数/实际 batch/LR 没有写回配置。
3. **[代码事实][中] A-V1 的相机多样性重排实际上几乎无效。**
   - `A/WHU/datasets/av1_sampler.py:48-71` 只在同一 PID 的现有 K 次 occurrence 内重排，不能创造新的图像/相机；日志 `cameras_before≈2.015`、`after≈2.031`（`A/logs/train_log_20261026_1121.txt:110,227,1459`）。
   - 但它仍改变了 ProCA 的 K-group 组合（文档 `A/CODE_CHANGE_RECORD.md:63` 承认），因此是多因素混淆，而不是“无害的采样改进”。
4. **[代码事实][中] A-V1 MP 与 Pair/Mixed 的相机规则不一致。**
   - MP 无 camera mask（`A/WHU/loss/av1.py:47-65`），Bank 混合所有相机；Pair/Mixed 排除同相机（`:68-77`）。
   - MP 占新增标量 58.6%，却优化与评测不同的“全相机身份原型”目标。
5. **[代码事实][中] A-V1 MP Bank 跨 rank 聚合按 rank 等权，而不是按样本数。**
   - `av1.py:22-36`：本地中心先归一化、`counts[local]=1`，再 all_reduce。若同一 PID 在不同 rank 上样本数不同，聚合不是全局样本均值。PKM 下每个 rank 通常每 PID 一个 occurrence，偏差很小，但属于实现近似。
6. **[代码事实][中] A-V1 的 `mp_bank.update` 受 `use_aux_modules` 而不是 `av1` 门控。**
   - `processor.py:310-314`：只有 `use_aux_modules and step_ok` 才更新 MP Bank。若 `LOSS_TYPE='base'` 但 `AV1.ENABLED=True`，MP 损失会用 bootstrap Bank 训练，却永不 EMA 更新（latent path）。
7. **[代码事实][中] A-V1 缺少设计文档要求的梯度范数诊断。**
   - `A/idea.md:603-611` 明确要求在 epoch1/5 记录 `L_base`、`λP·L_MP`、`λC·0.75·L_pair`、`λC·0.25·L_mixed` 对 `global_feat` 的梯度范数；实际日志只有 loss 和有效 Query 数，没有梯度预算/范数，导致无法量化 MP 的主导程度。
8. **[代码事实][中] WCV1 候选 detach 导致非对称梯度。**
   - `mixed_rank.py:41,67`：候选完全断流，只有 anchor 端回传；与 A-V1 的对称 Pair/Mixed 不同。若目标是 in-batch 对称对比，这个实现会削弱负侧 repulsion。
9. **[代码事实][低] WCV1 的 `active=3` 除法与预算探针相互补偿。**
   - `mixed_rank.py:94-97` 除以 `active`；`budgeted_auxiliary_loss` 用实际 aux loss 的梯度做归一化，因此 feature 梯度比仍为 0.1。不是错误，但使 aux 的标量份额（44.5%）与梯度份额（10%）严重脱节，容易误读日志。
10. **[代码事实][低] WCV1 的预算探针每步多做两次 `autograd.grad`，训练成本约翻倍。**
    - `mixed_rank.py:125-126`；不是正确性问题，但会限制可扩展性。
11. **[代码事实][低] WCV1 的 `grad_ratio`/`weight` 没有末期衰减。**
    - `processor.py:269` ramp 只升；`mixed_rank.py:135-138` 的 weight 只由 ramp 和 rho 决定。收敛期仍在施加 10% off-axis 梯度。
12. **[代码事实][低] B-B 没有 sample key/重复图去重，依赖同相机排除自匹配。**
    - `bb_ranking_loss.py:42-45`；在 WHU-MARS 正常数据下无泄漏，但若同图以不同 camera ID 重复入库，可能被当作正样本/负样本。
13. **[代码事实][低] A-V1 的 sample_id 对 Pair 跨模态方向不起自匹配作用。**
    - `bases.py:88-107` 的 offset 按模态不同，所以 Pair 跨模态的 `q_id != g_id` 永远为真；Pair 的自匹配排除完全依赖 camera。Mixed 的同模态块才真正用到 sample_id。不是漏排（自图同相机），但文档“干净地使用 sample ID”在 Pair 上名不副实。

### 7.2 未发现的缺陷

- **空路径**：三组都没有在本次运行中触发全空损失方向。A-V1 每方向 Pair 44–64、Mixed 52–64、MP 64；B-B valid 190/192；C valid_fraction 0.95–1.0。
- **数值 NaN/dead**：A-V1 `finite` 正常、AMP skip 8/13696；C `finite=1` 全程（日志 `finite=1`）；B-B 无 NaN，`bb_raw` 最低 0.0018 但仍非零。没有“权重被裁剪到 0”：A/B 固定权重，C 的 weight 0.18–0.25，`max_weight=1.0` 从未触发上限。
- **有效 Query 过少**：否。A/B/C 的 valid query 比例分别为约 69–100%（A 的方向级）、99.3%（B）、95–100%（C）。真正的稀缺是**负身份覆盖**（3%/3%/12.6%）和**正例数**（B 5.7，C 6–9），不是 Query 数。
- **饱和**：
  - WCV1 第 11 轮 `saturation=0.907`（90.7% 正例对已远超 batch 最危险负例，主要因为当时特征弱、最危险负例还不近），第 120 轮 0.050–0.132；不是“损失饱和死掉”，但正例梯度只集中在靠近边界的少数对。
  - B-B `top_error_ratio` 末轮 0.75%、`bb_raw` 0.0018，pairwise 软 hinge 在 batch 内已接近饱和；`alpha` 0.0116，risk 分支基本关闭。
  - A-V1 MP 0.97、Pair 0.61、Mixed 0.92，均未饱和到零。
- **梯度裁剪**：三组都调用 `clip_grad_norm_(model.parameters(), 1.0)`；日志没有记录裁剪前后的梯度范数，无法判断是否经常触发。若有裁剪，全局同比例缩放不改变 A/B 的固定权重比，也不改变 C 的 0.1 预算比；但会改变绝对步长。属于**未监控项**，不是已确认缺陷。

---

## 8. 与文档描述的差异

| 文档 | 文档说法 | 代码/数据 | 差异性质 |
|---|---|---|---|
| `A/CODE_CHANGE_RECORD.md:63` | 相机多样性 K-group 重排“优先让一个 K-group 覆盖更多相机” | 日志 `cameras_before≈2.015→after≈2.031`（`A/logs/train_log_20261026_1121.txt:110,227,1459`） | 实现存在，但效果近似 no-op；文档高估了收益 |
| `A/idea.md:603-611` | 训练前/早期必须做梯度范数诊断 | 日志只有 loss 与有效 Query 数；`A/AV1_DIAGNOSIS_ANALYSIS.md:137` 也承认没有中间权重/梯度 | 设计要求的诊断未执行，导致 A-V1 无法量化 MP/Pair/Mixed 的梯度占比 |
| `A/CODE_CHANGE_RECORD.md:32` | 总损失固定为 `0.1 MP + 0.1(0.75 Pair+0.25 Mixed)`，ramp 到 1 | 与代码一致（`processor.py:289-291`） | 一致；但“ramp 只升不降”使后期新增占标量 34.5%，文档未讨论 |
| `B/WHU/configs/BB0.yml:6`、`BB1.yml:6` | `BB_LOSS_WEIGHT: 0.0` | 实际 CLI `85.807006`；`make_loss.py:40-41` 对 0 抛错 | 配置不可直接运行；实际系数只在日志里 |
| `B/WHU/calibrate_bb.py:113` | 用 BB0 标定同一系数给 BB0/BB1 | 实际跑 BB1 adaptive，未重新标定 | 方法上按文档执行，但 adaptive 的梯度尺度随 α 改变，系数未必匹配；无 BB1 标定记录 |
| `B/WHU/idea.md:1758` | λ_B 一次性标定后固定 | 代码与运行一致；BB 标量始终约 40% | 一致；风险是“固定不衰减”未在文档中评估 |
| `C/WCV1_IMPLEMENTATION.md` | “所有候选 detach、跨卡 gather、按稳定文件键保留首个、全同相机排除、负实例 logsumexp、正例 gallery 模态组均值、全局有效 anchor 模态均值” | 代码一致（`mixed_rank.py:40-107`） | 一致 |
| `C/logs/diagnosis_120/report.md:189` | “新增失败以最佳正例变弱为主……优先修订正例监督分配” | `ranking/summary.json`：`negative_strengthened=238`、`positive_weakened=13`、`both=60` | **报告结论与自身数据矛盾**：纯负例变强占 73.5%，纯正例变弱仅 4.0% |
| `C/configs/WCV1.yml:37-38` | `BASE_LR 0.008`、`IMS_PER_BATCH 64` | 实际 `0.032`、`256`（`train_log_20261028_2339.txt:2`） | 配置文件不是实际运行配置；实际配置只在 CLI/日志 |
| `C/WHU/WCV1_FINAL_CHECK.md:5` | 显式保留内部 raw CLS，使预算包含 CE 路径 | 代码一致（`make_model.py:71-75`，`processor.py:223-227`） | 一致；但没有提到候选 detach 的非对称性 |
| `C/logs/diagnosis_120/report.md:11-13` | coverage 是 eval 缓存代理，不是历史回放；终态梯度只说明当前模型 | 与现状一致；报告中已声明边界 | 诚实边界；不应把 danger exposure 当逐步因果 |

---

## 9. 设计规则：不应再做 / 可以保留

### 9.1 不应再做（Do not repeat）

1. **不要用单卡/单 batch 的 15–63 个负身份去直接优化全库 R1。**
   - A/B 3.0%、C 12.6% 的身份覆盖，评测是 500 身份/93,609 图。若目标是 R1，必须有全局/跨 batch 图像级负例记忆（queue、memory bank、gallery cache）或至少跨 rank gather；否则只能改善 batch 内排序。
2. **不要在一次实验里同时改损失、采样器、ProCA 分组和权重。**
   - A-V1 同时引入 MP、Pair/Mixed、相机重排、困难交换；相机重排又几乎无效果，却改变 ProCA 中心。没有单变量消融，任何指标变化都不能归因。
3. **不要用一次性 ImageNet 标定的固定大权重贯穿全程。**
   - B-B 85.807006 从 epoch1 到 120 不衰减，BB 标量始终约 40%；一旦 batch 排序饱和，它仍在用大系数作用于极少数困难对。应使用 ramp + 末期衰减，或每 N 轮重新标定相对梯度。
4. **不要只约束辅助损失的梯度范数，不约束方向。**
   - C 的 rho=0.10 把 feature 梯度比固定在 0.1，但 cos(aux,base) 仅 0.30–0.33；方向偏离 AP/R1 时，10% 的 off-axis 梯度持续 100+ epoch 仍能改变 top-1。预算应同时报告 cos，并在末期把 rho 衰减到 0。
5. **不要把 query-risk 预算给“最容易/最相似的正样本”并期望修复 mAP。**
   - B-B 的 α 质量给 `best_pos`（max 相似度）；它只在“最安全正例都输”时救 top-1，不改善“最远正例”导致的 AP 损失。应把 top-1 风险预算与 AP/困难正例预算分开。
6. **不要 detach 候选却声称对称 in-batch 对比。**
   - C 的候选 detach 使负侧只有 anchor 端受力；若要有对称 repulsion，应允许候选回传，或明确使用 memory bank 的 query-side-only 语义。
7. **不要让正例留在对比分母里并期待强负例压制。**
   - A 的 Pair/Mixed 分母包含正例；在 τ=0.1、正例相似度 0.27–0.35 时，正例吸收约一半以上 softmax 质量，负例梯度份额被压缩。若要强负例压制，应在分母中分离正负，或使用 negatives-only logsumexp。
8. **不要在 AMP skip 步更新 Bank。**
   - B/C 目前无条件更新；A 已证明可低成本修复。统一使用 `step_ok` 门控。
9. **不要让配置与实际运行脱节。**
   - B 的 yml 权重为 0、C 的 yml batch/LR 与实际不一致；标定系数、实际 CLI override、数据路径都要写回/归档，否则不可复现。
10. **不要用聚合 mAP 作为唯一判据。**
    - 三组都出现“mAP 小升/持平、R1 下降”；必须同时报告 R1、mINP、逐协议、per-query 新失败分类（negative-strengthened vs positive-weakened）。
11. **不要在没有全库 per-query 诊断时宣称机制。**
    - B-B 没有 ranking/coverage 诊断，“负例相似度上升”无法证实；A 有诊断但无中间权重；C 有终态诊断但无历史回放（报告已声明）。机制结论必须带证据等级。
12. **不要假设“相机多样性重排”一定有效。**
    - K=4 且每 PID 相机数有限时，A 的 after 仅比 before 高约 0.02–0.03；先测量再决定是否保留。

### 9.2 可以保留（Keep）

1. **pre-BN CLS 作为唯一训练/推理特征。** 三组都没有额外可学习分支，推理接口不变；这是最干净的工程基础。
2. **同相机排除 + 自匹配排除。** A/C 的 sample ID/key 设计正确；与 evaluator 的 camera 规则一致（`U/WHU/utils/metrics.py:161-164`）。
3. **Bank 的 buffer/no-grad/不进 optimizer。** MP/GPD Bank 不接收梯度、不进入 optimizer；只做 EMA；这是正确的“记忆不是参数”实现。
4. **A-V1 的 AMP 成功步门控、RNG 保存、manifest/artifact hash、resume 校验、采样 ledger 断言。** 这些是高质量的复现工程，应保留。
5. **C 的 gather 形状检查 + sample-key 去重 + metadata 一致性检查。** `mixed_rank.py:42-66` 在分布式正确性上是可靠的；保留。
6. **C 的 finite 检查与非 finite 自动停用。** `budgeted_auxiliary_loss` 在非有限时退回 base，不会 poison 图；保留。
7. **C 的 feature 级梯度预算诊断。** 虽然不保证 AP 对齐，但它能量化“辅助项到底有多强”；保留并补充方向 cos 与正/负分解。
8. **A/C 的 start epoch / ramp。** A ramp 5 epoch、C start 10 + ramp 10 都避免了早期强干扰；保留，并增加末期 decay。
9. **按方向/模态记录有效 Query 与 loss。** A 的方向级 counts、C 的 `valid_mod0/1/2`、B 的 modality ratios 都是可复用的覆盖诊断；保留。
10. **A 的固定 epoch-10 困难参考 + 全量确定性提取。** 思路正确（避免在线刷新导致不稳定）；但需要评估“是否过期”，可在 epoch 40/80 重新生成并对比。
11. **C 的 batch-separated-full-wrong 指标。** 这是区分“batch 内可分离”和“全库可检索”的好指标；应扩展为训练期/中期诊断。
12. **C 的 sample key 含模态目录。** 同图跨模态不同 key、同图同模态同 key，语义正确；保留。

---

## 10. 最小验证建议（不属于已证实因果）

1. **A-V1 单变量消融**：固定 UAD 配置，分别只开 MP、只开 Pair/Mixed、只开相机重排、只开困难交换；在每个变体上跑同一 Query 清单的 `max-neg` / `danger identity present` 诊断。若“只开 MP”就复现 R1 下降，优先修 MP 的相机掩码/权重。
2. **B-B 重新标定 + 衰减**：在 epoch 20/60/100 测量 BB0/BB1 的 feature 梯度范数，重设系数；加末期 `λ_B→0` 或 α/负身份温度退火；比较 R1/mAP。
3. **WCV1 负例覆盖增强**：保持现有 target，增加跨 batch 图像级负例队列（真实训练图，key 去重），或至少在末期提高 gather/候选数；同时把 rho 在最后 10–20 epoch 衰减到 0，观察 R1 是否回升。
4. **统一 per-query 诊断**：对 B-B/A-V1 补 `new_failure_categories`（negative-strengthened / positive-weakened / both）和 99 分位负相似度；否则无法验证“负例变强”假设。
5. **全库危险负例训练期探针**：每 N 步抽样 Query，计算其全库 max negative；若训练期该值持续上升，而 batch max negative 下降，则“候选集不可迁移”机制被直接支持。

---

## 11. 证据索引

### 代码

- A-V1 损失：`A/WHU/loss/av1.py:47-65`（MP）、`:68-77`（mask）、`:73-87`（uni/set）、`:90-115`（Pair/Mixed）。
- A-V1 采样：`A/WHU/datasets/av1_sampler.py:35-119`；相机重排 `:48-71`；困难交换 `:74-110`。
- A-V1 训练/状态：`A/WHU/processor/processor.py:169-222`（启用/恢复/参考加载）、`:283-314`（损失/反向/成功步）、`:361-384`（epoch10 参考）、`:386-410`（checkpoint）。
- A-V1 特征提取/困难矩阵：`A/WHU/processor/av1_state.py:57-102`。
- A-V1 数据集 sample_id：`A/WHU/datasets/bases.py:88-107`。
- B-B 损失：`B/WHU/loss/bb_ranking_loss.py:32-116`；权重 detach `:52-87`；pair loss `:91-100`。
- B-B 组装/归一化：`B/WHU/loss/make_loss.py:28-44,53-79`。
- B-B 训练：`B/WHU/processor/processor.py:188-234`。
- B-B 标定：`B/WHU/calibrate_bb.py:40-113`。
- C 损失：`C/WHU/loss/mixed_rank.py:19-109`；detach/gather `:39-53`；去重 `:55-66`；损失 `:67-101`；预算 `:112-142`。
- C 训练：`C/WHU/processor/processor.py:178-189`（配置）、`:221-284`（raw feature/aux/反向/Bank）。
- C 特征捕获：`C/WHU/model/make_model.py:67-77`。
- C sample key：`C/WHU/datasets/bases.py:79-108`。
- 评测同相机规则：`U/WHU/utils/metrics.py:136-164`；平台掩码 `:259-305`。

### 日志/诊断

- A-V1 训练日志：`A/logs/train_log_20261026_1121.txt:106`（早期）、`:110,227,1459`（采样统计）、`:1454-1459`（末轮组件/Query 数）、`:1493-1494`（末轮指标）。
- A-V1 诊断：`A/AV1_DIAGNOSIS_ANALYSIS.md:27-31`（总体）、`:47-62`（Rank/AP 分解）、`:67-79`（s+/s−/分布）、`:89-100`（九方向）、`:110-121`（平台/相机）。
- B-B 训练日志：`:2`（CLI 覆盖）、`:106`（第 1 轮）、`:1337-1338`（第 120 轮）、`:1345-1352`（九方向）、`:1369-1370`（A→G/G→A）、`:1372-1373`（末轮指标）。
- C 训练日志：`:2`（CLI 覆盖）、`:293`（第 11 轮 aux）、`:1418-1419`（第 120 轮 aux）、`:1453-1454`（末轮指标）。
- C 诊断报告：`C/logs/diagnosis_120/report.md:18-32`（总体/九方向）、`:133-160`（覆盖/梯度）、`:163-185`（探针）、`:187-194`（建议及矛盾）。
- C 排名 summary：`C/logs/diagnosis_120/ranking/summary.json`（`overall.new_failure_categories` 等）。
- C 梯度探针：`C/logs/diagnosis_120/probes/gradient_pairs.csv`（16 组中位数：WCV1 feature_all positive/negative `0.0958/0.0510`，aux-positive cos `0.860`，aux-negative cos `0.234`；last_block aux/base `1.012`，cos `0.154`）。
- C 覆盖：`C/logs/diagnosis_120/coverage/coverage_summary.json`、`wc*v1_candidate_exposure.csv`、`*_positive_relations.csv`。

---

## 12. 一句话总结

- **A-V1**：在 15 个 batch 负身份上做对称多正例对比 + 全局原型分类，新增项后期占标量 34.5%，但没有全局图像级负例记忆、且分母被正例吸收；诊断显示全库最大负例上升而负对均值下降，R1 掉 2.06 pp，mAP 掉 0.42 pp。
- **B-B**：在 15 个 batch 负身份上把负例预算集中到最危险负身份（实测约 19.4%，单对上限约 10%），batch 内 top-1 从 94.6% 错误降到 0.75%，但 85.8 固定权重全程不让位、α 末期关闭；全库 R1 仍掉 2.84 pp，mAP 基本持平，是“候选集不可迁移”的教科书案例。
- **WCV1**：4 卡 gather 到 768 候选/63 负身份，正例组均值 + anchor-only + rho=0.10 使 mAP/mINP 小升、R1 掉 1.33 pp；73.5% 的 R1 新失败是纯负例变强，而报告却写成“正例变弱为主”，后续不应按该错误结论优先改正例监督。
