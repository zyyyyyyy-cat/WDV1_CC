# 第一组历史实验机制审计：HiHR / ViSA+MD+MP / SFTC / ICRE / TAUR

审计任务：用静态代码与既有日志核对 `02_Previous_Experiments.md` 第 2.2 / 2.3 / 2.7 / 2.8 / 2.12 节，并补充档案没有说清的机制。本次**只读**，未修改任何项目代码，未训练、未重新推理。

证据等级（沿用 02 的约定并细分）：

- **[代码事实]**：可直接在当前源码某行核对。
- **[实验支持]**：来自训练日志 / 诊断 CSV / `wrt_calibration.json` 的具体数值。
- **[机制假设]**：由代码事实+实验现象推出的解释，未经单因素对照证明。
- **[静态计算]**：本次用 CPU 版 PyTorch 实例化模块得到的参数量，不涉及权重与数据。

---

## 0. 路径映射与公共基线

### 0.1 本机实际代码根

用户给出的“实验目录”在本机的顶层名与服务器不同，实际读取路径如下。所有行号均以这些文件为准。

| 方法 | 本机代码根（绝对路径） | 备注 |
|---|---|---|
| Baseline UAD | `D:\CODE\Doing\WHU_base\WDV1_CC\WHU` | 与档案使用的 `WHU_DDP` 代码逐文件一致，仅 README/`01_...md` 不同 |
| HiHR | `D:\CODE\Doing\WHU_base\WHU_Base\WHU_HiHR\WHU` | `WHU_Base\WHU_HiHR\HiHR` 是官方仓库，不是本次适配代码 |
| ViSA+MD+MP | `D:\CODE\Doing\WHU_base\WHU_Base\WHU_ViSA+MD+MP\WHU` | 原始仓库 `ViSA/ MDReID/ MPReID/` 与 WHU 适配代码并列 |
| SFTC | `D:\CODE\Doing\WHU_base\WHU_Base\WHU_Fusion+MSCMNet+SDS+CoMer_SFTC\WHU` | `D:\CODE\Doing\WHU_base\SFTC\WHU` 为同代码副本，仅 README/配置名不同 |
| ICRE | `D:\CODE\Doing\WHU_Code\WHU_Research_Archive\2026-10-24_Historical_Audit\_audit_evidence\WHU_base\WHU_ICRE` | 本机 `WHU_base` 下**没有** ICRE 源码，只有审计镜像；镜像与档案描述一致，含训练日志 |
| TAUR | `D:\CODE\Doing\WHU_base\TAUR\WHU` | `MDReID/ MOS/ UPCL/ WSL/` 为参考仓库 |

日志位置：

- HiHR：`...\_audit_evidence\WHU_base\HiHR\logs\{HiHR_H2_stage1,HiHR_H1_stage2,HiHR_H2_stage2}\` 与 `logs\wrt_calibration.json`
- ViSA：`...\_audit_evidence\WHU_base\ViSA+MD+MP\logs\{UAD_ViSA_1000_DDP,UAD_ViSA_1000_DDP_SEED2025}\`
- SFTC：`...\_audit_evidence\WHU_base\WHU_Fusion+MSCMNet+SDS+CoMer_SFTC\logs\SFTC_1000_DDP\`；诊断 CSV 在 `D:\CODE\Doing\WHU_base\SFTC\logs\SFTC_120_diagnostics\`
- ICRE：`...\_audit_evidence\WHU_base\WHU_ICRE\logs\UAD_ICRE_1000_DDP\train_log_20260918_0127.txt`
- TAUR：`...\_audit_evidence\WHU_base\WHU_MD+MOS+UPCL+WSL_TAUR\logs\TAUR_1000_DDP_v2\train_log_20261012_2327.txt`
- Baseline：`...\_audit_evidence\WHU_base\WHU_DDP\logs\UAD_1000_DDP\train_log_20260911_2145.txt`

### 0.2 Baseline 调用链

```
train.py:128  make_dataloader
train.py:130  make_model
train.py:132  make_loss
train.py:134  make_optimizer
train.py:136  create_scheduler
train.py:138  do_train
```

- 模型：`WDV1_CC\WHU\model\make_model.py:28` `build_transformer`；`:49-52` 构造 `vit_base_in` 并加载 ImageNet；`:60-65` classifier / bottleneck；`:67-80` forward。
- 主干：`WDV1_CC\WHU\model\backbones\vit_pytorch.py:151-154` Block 两个残差流；`:317-341` `forward_features`；`:336-337` 12 层循环；`:339` 最终 LayerNorm；`:341` 返回 `x[:,0]`。
- 训练：`WDV1_CC\WHU\processor\processor.py:176-231` 单步；`:188` 前向；`:190` ID+WRT；`:194-213` ProCA/GPD；`:214-217` 加权求和；`:219-223` backward/clip/step；`:227` 原型更新。
- 损失：`WDV1_CC\WHU\loss\make_loss.py:30-37`；WRT：`loss\wrt_loss.py:39-65`，欧氏距离 `pdist_torch` `:67-81`；ProCA/GPD：`processor.py:13-30`、`:33-71`、`:74-88`。
- 采样：`datasets\sampler.py:69-143`；collate：`datasets\make_dataloader.py:17-23`。
- 评估：`utils\metrics.py:327` `R1_mAP_eval`；`:361-385` 不同 mode 合并；`:462-465` L2；`:480-488` 总体；`:136-203` `eval_func`；同相机过滤在 `:164`。

### 0.3 Baseline 前向/反向（后续所有图以此为基准）

```
imgs = [RGB[B], NIR[B], TIR[B]]
  └─ torch.cat -> X[3B,3,256,128]                  make_model.py:69
       │
       ├─ patch_embed (Conv16x16) -> [3B,128,768]
       ├─ cat(cls_token) -> [3B,129,768] + pos_embed
       ├─ blocks 1..12 (attn + mlp, 两个残差流)     vit_pytorch.py:151-154
       ├─ final LayerNorm
       └─ f0 = x[:,0]  (pre-BN CLS, 3B x 768)       vit_pytorch.py:341
             ├─ bottleneck(BN1d) -> feat            make_model.py:71
             │     └─ classifier -> cls_score        make_model.py:72
             ├─ CE(cls_score, target_rep)
             ├─ WRT(f0, target_rep)                 (欧氏距离)
             ├─ ProCA(f0)                           (模态中心对齐)
             └─ GPD(f0, prototype bank)             (余弦 + tau=0.03)
```

反向：单一 `loss` 标量 -> `backward`，所有参数共享一次全局 `clip_grad_norm_(max_norm=1.0)`。

### 0.4 Baseline 的损失/归一化顺序

| 损失 | 输入张量 | BN 前/后 | 归一化 | 度量 |
|---|---|---|---|---|
| CE | `cls_score = classifier(BN(f0))` | BN 后 | 分类器权重线性 | logits |
| WRT | `f0` | BN 前 | 无；直接欧氏距离 | 欧氏 `(sum((a-b)^2)).sqrt()` |
| ProCA | `f0` | BN 前 | 每个模态先 L2，再组内均值+L2；`g_cent` detach | 平方欧氏 |
| GPD | `f0` | BN 前 | `f` 与 prototype 都 L2 | 余弦 logits / 0.03 |
| 检索 | `f0` | BN 前 | L2（`FEAT_NORM='yes'`） | 欧氏 ≡ 余弦 |

总损失 = `1.0*CE + 1.0*WRT + 0.01*ProCA + 1.0*GPD`（`configs/UAD.yml`；`processor.py:214-217`）。

### 0.5 Baseline 评估协议的关键事实

1. 总体 AS：3 个模态的 query/gallery 全部拼接后统一检索（`metrics.py:361-385`），不是分模态平均。
2. 同相机过滤：`remove = (g_camids[order] == q_camid)`（`metrics.py:164`），即**排除全部同相机 Gallery，包括异身份**；这与 Market1501 的 `(pid==pid & cam==cam)` 不同。
3. 训练 WRT **不排除同相机**：`is_pos/is_neg` 只由 pid 决定（`loss\wrt_loss.py:46-47`）。所以 WRT 的“最难负样本”可能来自评估时会被删除的同相机样本。
4. 训练 batch：`PKM`，每索引元组含 3 模态；`IMS_PER_BATCH=256` 是索引数，每卡 64 索引 = 192 图；全局 768 图。WRT 只在**本卡** 192 图上计算；DDP 只同步梯度与 GPD 原型（`processor.py:41-71`）。
5. 训练身份数：日志统一打印 `identities: 500`（1000 数据集只有 500 个训练身份）；GPD 原型库大小 500。
6. 论文/档案的“1000”是数据集/评测口径，不是训练身份数。

### 0.6 Baseline 关键数值（H014，DDP 256/0.032）

总体 11.02 / 29.71 / 46.32 / 54.55。Epoch 60 中间点：9.90 / 26.99 / 43.64 / 52.24（用于分析 HiHR 两段式）。

九宫格（Rank-1 / mAP / mINP）：

```
RR 42.48/26.53/5.06   R→N 27.17/16.07/2.23   R→T 7.49/5.61/1.09
N→R 24.54/15.66/2.82  N→N 27.96/16.83/2.40   N→T 6.37/5.25/1.04
T→R 7.40/5.85/1.30    T→N 7.92/5.25/1.00     T→T 20.75/13.52/2.28
```

---

## 1. HiHR（H001/H002/H003）

### 1.1 调用链与新增文件

```
train.py:128 make_dataloader
train.py:130 make_model -> model/make_model.py:103 make_model
                            └─ make_model.py:105-112 若 HIHR.ENABLED:
                                 VisualHiHR(build_transformer.base, ...)
                                 可选 INIT_CHECKPOINT 严格加载
train.py:132 make_loss（未改）
train.py:134 make_optimizer -> solver/make_optimizer.py:23-25 新参数可乘 NEW_LR_FACTOR
train.py:138 do_train -> processor/processor.py
```

新增/改写文件：

- `model/visual_hihr.py`（TMF、VisualHiHR、CosineClassifier）
- `loss/visual_hihr.py`（同模态 triplet、Lorentz HHL、辅助损失）
- `model/backbones/vit_pytorch.py:317-345`（`return_layers` 读取第 5/8/12 层输出）
- `processor/processor.py:232-277`（F、WRT scale、辅助损失、ProCA/GPD）
- `config/defaults.py:207-225`（HIHR 超参数）
- `tools/calibrate_wrt.py`（日志 `wrt_calibration.json`）

### 1.2 插入位置（精确到张量）

- **读取层**：`vit_pytorch.py:336-344`，在 block 1..12 的循环中把第 5/8/12 层**完整序列**（CLS+Patch，129×768）存下；返回 `x[:,0]`（最终 LayerNorm 后 CLS）与 `[selected[5], selected[8], selected[12]]`。
  - 注意：第 12 层 token 是 **final LayerNorm 之前**的输出；`u0` 是 final LayerNorm 之后的 CLS。
- **TMF**：`visual_hihr.py:46-59`。
  - `norms` 三个 LayerNorm；共享 `projection(768→512)` 作用于包括 Patch 在内的全部 token（`:49`）。
  - Parent Query = 共享参数 `shared_query`；Child Query = `qp + child_offset + modality_embedding(m)`（`:51-53`）。
  - 4 头 Cross-Attention：Q 为 2 个 query，K/V 为**3 层 CLS**（`:54`）。`need_weights=True` 默认对头取平均，得到 `a[b,2,3]`。
  - Patch 融合：`einsum('brl,blnd->brnd', a, z[:,:,1:])`（`:55`）。**每个 query 只有 3 个标量权重，且 128 个 Patch 共用同一组 3 个权重**；不是逐 Patch、不是逐 head。
  - `cat([t, patches])` 后经过**共享的** 1 层 `TransformerEncoderLayer`（`:56-57`）；取 token0，`output_norm`，L2。
  - Transformer 的 `out_proj/linear2` 初始化乘 1e-3（`:39-43`），近似恒等启动。
- **CLS 侧**：`f0` 原本的 ViT 路径完全保留；`u0 = normalize(f0)`（`:101`）。
- **拼接**：`F = [sqrt(1-eta) u0, sqrt(2 eta/3) up, sqrt(eta/3) uc]`，`eta=0.2`（`:102-103`），`||F||=1`。平方权重为 0.8 / 0.1333 / 0.0667。
- **模态路由**：仅 Child Query 的 `modality_embedding(m)`；`m` 在训练时按 batch 顺序由 0/1/2 生成（`:92-93`），评估时按 `mode` 映射（`:98`）。TMF 对所有模态读同一组 token、共享全部权重；除 modality embedding 外没有模态掩码或独立分支。
- **作用对象**：同时使用 CLS 和 Patch。CLS 走 3 层 CLS 的 attention；Patch 走 3 层 Patch 的 3 标量加权；两者在同一个 Transformer 中（batch 维变成 2B）。

### 1.3 正向数据流

```
X[3B,3,256,128]
  │
  ├─ ViT blocks 1..12 ─┬─ block5  out Z5[3B,129,768] ─┐
  │                     ├─ block8  out Z8[3B,129,768] ─┤ (pre final-LN)
  │                     ├─ block12 out Z12[3B,129,768]─┤
  │                     └─ final LN -> f0[3B,768]      │
  │                                                    ▼
  │                                  TMF: LN(768) -> Linear 768→512 (shared)
  │                                       z[3B,3,129,512]
  │                                       │
  │                    q_parent[3B,512], q_child[3B,512] (shared+offset+modal emb)
  │                                       │
  │                  MHA(q, CLS_{5,8,12}) -> t[3B,2,512], a[3B,2,3]
  │                  patches = a · Patch_{5,8,12}[3B,2,128,512]
  │                  cat(t.unsqueeze(2), patches) -> [3B,2,129,512]
  │                  shared 1-layer Transformer -> token0 -> LN -> L2
  │                                       │
  │                                  up[3B,512], uc[3B,512]
  ▼
F[3B,1792] = concat(sqrt(.8) u0, sqrt(.1333) up, sqrt(.0667) uc)   ||F||=1
  ├─ CE: CosineClassifier(1792->500, scale=16) on F
  ├─ WRT: on (9.780072309592285 * F)
  ├─ ProCA: on F
  └─ GPD: on F (already unit)
检索（mode>0）：直接返回 F；不经过 bottleneck/classifier       visual_hihr.py:104-105
```

### 1.4 反向梯度流

```
loss = 1.0*CE(F) + 1.0*WRT(9.78F) + 0.01*ProCA(F) + GPD(F)
              │            │             │            │
              ▼            ▼             ▼            ▼
        CosineClassifier   └──────► F ◄──┴────────────┘
                                     │
            ┌────────────────────────┼───────────────────────────┐
            ▼                        ▼                           ▼
   sqrt(.8)*dL/dF0        sqrt(.1333)*dL/dup            sqrt(.0667)*dL/duc
            │                        │                           │
            │                TMF transformer/attn/proj/LN          │
            │                        │                           │
            │                dL/dZ5, dL/dZ8, dL/dZ12              │
            ▼                        ▼                           ▼
        final LN ───────────► ViT blocks 1..12 ◄───────────────┘
                                     │
                              patch_embed / pos_embed etc.

辅助:
  parent_score CE(w=0.1)   -> up -> TMF -> backbone
  child_score CE(w=0.025)  -> uc -> TMF -> backbone
  intra-modality triplet(w=0.1, uc) -> uc -> TMF -> backbone
  HHL: parent DETACHED anchors; child live -> uc -> backbone
```

关键点：

- **没有 detach 切断主损失**。F 的三个分量都接收 CE/WRT/ProCA/GPD；Parent/Child 不是弱监督旁支。
- **唯一的 detach** 在 `loss/visual_hihr.py:37` 的 `p = parent.detach()`，只阻止 HHL 更新父锚；Child 仍回传到 TMF 和整个 ViT。
- HHL 的锚点按身份×模态求均值，必须三模态齐全才算 `complete`（`:40-49`）；日志 H2 末期 `hhl_valid=1.0`。
- `optimizer_step_if_finite`（`:15-33`）在裁剪后检查梯度范数有限性；DDP 所有 rank 一致 step/skip。
- 新参数全部在 optimizer 中；`HIHR.NEW_LR_FACTOR=1.0`（默认），所以**没有特殊学习率**；bias 仍乘 `BIAS_LR_FACTOR=2`；WD 1e-4（classifier 权重也是 1e-4，但 CosineClassifier 的权重在 forward 中被 L2 归一化，WD 只缩尺度、不改变方向）。

### 1.5 损失与归一化顺序（HiHR）

| 项 | 输入 | BN 前/后 | 归一化 | 权重 |
|---|---|---|---|---|
| CE | F[3B,1792] | 无 BN | `CosineClassifier`：权重 L2，feature 不重新归一化（F 已单位） | 1.0 |
| WRT | 9.78007 × F | 无 BN | 无 normalize_feature；欧氏距离 | 1.0 |
| ProCA | F | 无 BN | 内部每模态 L2；中心 L2；全局中心 detach | 0.01 |
| GPD | F | 无 BN | `f = normalize(F) = F`；prototype L2；tau 0.03 | 1.0 |
| Parent CE | up | 无 | CosineClassifier scale 16 | 0.1 |
| Child CE | uc | 无 | CosineClassifier scale 16 | 0.025 |
| 同模态 Triplet | uc | 无 | `torch.cdist` | 0.1 |
| HHL | up.detach 锚点, uc | 无 | Lorentz，FP32，autocast off | 0.1 × min(1, epoch/5) |

相加顺序：`loss = CE + WRT`（`make_loss`）→ `+ 0.1*parent + (0.025*child + 0.1*triplet) + lambda_hhl*HHL`（`processor.py:240-244`）→ `+ 0.01*ProCA + GPD`（`:274-277`）。数学上等价于求和。

**WRT scale**：`processor.py:236` 用 `cfg.HIHR.WRT_SCALE * F`；两次训练日志都覆盖为 `9.780072309592285`（stage1 与 H1/H2 stage2 都是）。`wrt_calibration.json` 记录的规则是正/负样本距离中位数比值的几何平均；正样本 ratio 8.127、负样本 ratio 11.769、disagreement 1.448。见 1.9 的解释。

### 1.6 优化器与两阶段

- `make_optimizer.py:23-25`：`HIHR.ENABLED and not key.startswith('base.')` 时 `lr *= NEW_LR_FACTOR`（默认 1.0，等于无效果）。
- Stage1：`configs/HiHR_stage1.yml`，60 轮，`LOSS_TYPE=base`，因此 `use_aux_modules=False`，**没有 ProCA/GPD，也没有 HHL**；但 parent/child CE 与 triplet 从第 1 轮就开启。
- Stage2：`HiHR_H1_stage2.yml` / `HiHR_H2_stage2.yml`，60 轮，`LOSS_TYPE=base+pca+gpd`；`INIT_CHECKPOINT` 严格加载 Stage1 全量权重（`visual_hihr.py:110-113`），然后由 `train.py` **重新创建 optimizer 与 scheduler**，LR 从 0.032 重新 warmup+cosine。
- H1 与 H2 都加载 `HiHR_H2_stage1/transformer_60.pth`（日志第 2 行可证），因此 H1/H2 是同一 Stage1 上的 HHL 开/关对照。

### 1.7 与评估协议的一致性

- 检索特征：1792 维 F，L2 后欧氏（`metrics.py:462-465`）。
- 同相机过滤：与 baseline 相同，排除全部同相机。
- 训练 WRT：**不排除同相机**；同模态 triplet 也不排除同相机。
- WRT 的距离尺度：F 是单位向量，最大欧氏距离 2；baseline CLS 的 norm 约 27.7（final LN 后每维≈1），最大距离约 39。乘 9.78 后最大 19.6，仍然小于 baseline 的 39。
- 训练目标与评估度量：F 上的 WRT 是“9.78×欧氏”；评估是 L2 后的余弦。由于 F 单位，欧氏与余弦单调等价，但 WRT 的 softmax 权重 `exp(±d)` **对尺度敏感**，9.78 只匹配中位数，不匹配分布形状。

### 1.8 结果核对（与 02 的数值一致）

| 编号 | AS mAP / R1 / R5 / R10 | 相对 H014 |
|---|---|---|
| H002 Stage1 (60 ep) | 10.72 / 27.99 / 44.62 / 52.72 | -0.30 / -1.72 / -1.70 / -1.83 |
| H001 H1 Stage2 | 11.44 / 28.57 / 45.82 / 54.52 | +0.42 / -1.14 / -0.50 / -0.03 |
| H003 H2 Stage2 (HHL) | 11.44 / 28.57 / 45.92 / 54.27 | +0.42 / -1.14 / -0.40 / -0.28 |

H001 九宫格相对 baseline 的 (ΔmAP / ΔR1)：

```
RR +0.68/-1.54   R→N +0.89/+0.18   R→T +0.17/+0.14
N→R +0.72/-0.79  N→N +0.52/-0.70   N→T +0.49/+1.12
T→R +0.29/+1.41  T→N +0.43/0.00    T→T +0.54/-1.55
```

[实验支持] **9 个方向的 mAP 全部上升；三个同模态方向的 R1 全部下降（RR -1.54、NN -0.70、TT -1.55），且这三个方向的 mINP 全部上升（+0.53/+0.32/+0.36）。** 这与 02 的“代表性模态变化”一致，02 给出了 TIR 的数字，但未指出“9 宫格 mAP 全升、同模态 R1 全降、mINP 全升”这一更完整的模式。

[实验支持] Baseline 在 epoch60→120 提升 +1.12 mAP / +2.72 R1；HiHR Stage1→Stage2 只提升 +0.72 mAP / +0.58 R1。HiHR 的第二段是重新 warmup、LR 从 0.032 重走 cosine 的 60 轮，而 baseline 的第二段是同一 cosine 曲线继续下降的低 LR 精修；两者后半程的 R1 收益差异很大（+0.58 vs +2.72）。

### 1.9 失败机制：为什么 mAP 升而 R1 降

**[代码事实 1] 固定权重、单位化、拼接，余弦可精确分解。**
```
F = [a1 u1, a2 u2, a3 u3],  a1²=0.8, a2²=0.1333, a3²=0.0667, Σai²=1
||F||=||F'||=1
cos(F,F') = Σ ai² cos(ui,ui') = 0.8 cos0 + 0.1333 cosp + 0.0667 cosc
```
所以最终相似度是**三个分量余弦的固定加权平均**，CLS 的权重只有 0.8；新增分量合计固定占 20%。

**[代码事实 2] 三个分量都接收全部主损失。** `processor.py:236-240` 把 WRT 作用在 F 上，CE 作用在 F 上，ProCA/GPD 作用在 F 上；梯度按 sqrt(0.8/0.1333/0.0667) 分别回传到 f0/up/uc。因此不能把 parent/child 说成“只受弱辅助监督”；它们的辅助 CE 只是额外项。

**[代码事实 3] WRT 的尺度标定并不可靠。**
`wrt_calibration.json` 的参考是**已训练的 UAD H014 权重**，而被标定的 HiHR 是**随机新模块 + ImageNet 主干**（json 的 `comparison_caveat` 自己写明）。正/负样本的 ratio 分别是 8.13/11.77，相差 1.448 倍，单个标量 gamma 不可能同时匹配两者；选几何平均 9.78 意味着相对参考模型，正样本被缩放得较少、负样本被缩放得较多，WRT 的 margin 分布整体偏移。校准前后每个样本的 WRT 梯度均值从 0.00137 变成 0.0627（约 46 倍，json 的 `gradient_F_per_sample_mean`），即 **WRT 在总损失中的有效权重被显著放大**，且该放大不是按 CE 的梯度尺度做的。

**[实验支持 1] 同模态 mINP 全升、R1 全降、9 宫格 mAP 全升。** 这与“新增分量让更多正样本（尤其是中后位）靠近、但没有让最近正样本相对最难负样本更可分”一致；指标本身不直接给出样本间距离变化，因此这只是相容证据。

**[机制假设 A] 固定 20% 混合的“稀释/平滑”效应。** 对每个 query，R1 只由最近正样本与最难负样本的余弦差决定。若新增分量在大量正样本上有正贡献（mAP 升），但在最难的那一对上贡献接近或更大，则 `0.8*cos0 + 0.2*cos1` 的 margin 相对纯 CLS 会被压缩并加入噪声，R1 下降；mAP 对全体正样本的排序更宽容，因而上升。mINP 上升（最后相关正样本位置变好）与“尾部正样本改善”一致。该假设能统一解释 9 宫格 mAP 全升 + 同模态 R1 全降 + mINP 全升，但没有 ON/OFF 或单分量检索证据，不能证明唯一因果。

**[机制假设 B] 两段式优化与分类器变化。** 同一 epoch 比较：HiHR Stage1（60 轮，无 ProCA/GPD）为 10.72/27.99，baseline epoch 60 为 9.90/26.99，即 **+0.82 mAP / +1.00 R1**；Stage1 并不差。问题出在后半程：Stage2 只增加 **+0.72 mAP / +0.58 R1**，而 baseline epoch60→120 增加 **+1.12 mAP / +2.72 R1**。也就是说，最终 R1 的 -1.14pp 是“HiHR 的第二段 60 轮收益远小于 baseline 的第二段 60 轮”造成的，而不是 Stage1 本身落后。Stage2 重新创建 optimizer/scheduler、LR 从 0.032 重新 warmup，和 baseline 后半程的低 LR 精修不是同一优化过程；两段式调度、ProCA/GPD、辅助损失、余弦分类器同时变化。因此“层次模块提升 mAP”应降级为“在 1792 维统一描述子 + 余弦分类器 + 两段式调度 + ProCA/GPD 的联合配置下，最终 mAP 略升、R1 下降”，不能归因给 TMF 本身。HHL 没有额外收益（H1 vs H2：mAP/R1 完全相同，R5 +0.10、R10 -0.25），支持“双曲层次监督不是主因”。该条有较强实验支持，但仍非单因素。

**[机制假设 C] TMF 的跨层融合粒度过粗。** `a` 是 4 头平均后的 3 个标量（`:54` 默认 `average_attn_weights=True`），128 个 Patch 共用同一组层权重；Layer 5/8/12 的 Patch 被压成同一个全局 3 层混合。“跨层 Patch 证据选择”实际上没有逐 Patch/逐 head 的选择能力，新增信息主要是 3 个 CLS 的加权组合。这是 [代码事实]，它对“表达能力不足”是支持，但不能单独证明 R1 下降。

### 1.10 可复用教训（HiHR）

1. 把新增分量拼进唯一检索特征、并让它接收全部主损失，是比“只在旁支加辅助损失”更干净的监督设计；但**固定范数权重**会在训练/推理全程强制分配距离预算，无法在局部不可靠时自动降权。
2. mAP/mINP/R1 必须一起看：同模态 mINP 升 + R1 降是“尾部正样本改善、头部不变差/变差”的可观测模式，不能只用 mAP 宣布成功。
3. 距离标定不能用“中位数匹配”代替梯度匹配；同一标量无法同时匹配正/负样本的尺度比，必须在训练后用同权重 ON/OFF 或单分量检索复核。
4. 两段式训练要单独报告 Stage1 与 baseline 同 epoch 的对照；否则“两段式 + 额外损失 + 重启 LR”与“模块”无法分离。

### 1.11 与 02 档案的差异

- 02 的数值、模块描述、损失权重、HHL 的 detach、`hhl_valid=1`、两段式结构均**核对正确**。
- 02 未指出：`wrt_calibration.json` 的参考是已训练 UAD、被标定对象是 fresh HiHR；正/负 ratio 不一致 1.45 倍；WRT 每样本梯度被放大约 46 倍。只写了“校准已做”，容易让读者以为尺度中性。
- 02 未指出：TMF 的 Patch 融合只有 3 个跨头平均标量，128 Patch 共用；这会显著限制“跨层 Patch 融合”的实际表达力。
- 02 未指出：三个分量都接收主损失；若读者把“辅助 Parent CE 0.1 / Child CE 0.025”理解为“parent/child 只受弱监督”，会误判梯度。
- 02 的 Stage1/Stage2 与 baseline 同 epoch 对比缺失；加上该对比后，“mAP 提升来自层次模块”这一归因应进一步弱化。
- 02 的“多分支单位拼接……改善更多正样本平均位置却削弱最近邻边界”是合理假设，本次补上了余弦可精确分解的公式与 9 宫格证据，但仍无 ON/OFF，不能升级为已证因果。

---

## 2. ViSA + MDReID + 模态 Prompt（H005/H006）

### 2.1 调用链与新增文件

```
train.py:128 make_dataloader（datasets 与 baseline 完全相同）
train.py:130 make_model -> model/make_model.py:86 forward
train.py:134 make_optimizer -> solver/make_optimizer.py:20-27（新参数 2×LR）
train.py:136 create_scheduler -> solver/scheduler_factory.py:18-22（新组 warmup/floor 同步乘 2）
train.py:138 do_train -> processor/processor.py:202-261
```

新增/改写：

- `model/visa_adapt.py`（ETGM、DLFM、make_retrieval_feature）
- `model/backbones/vit_pytorch.py:243-265, 357-381`（specific token 与 `patch11`）
- `model/make_model.py:39-127`（adapt_mode、specific、ETGM/DLFM、最终 z）
- `processor/processor.py:236-255`（specific WRT、local CE、fusion WRT、balance）
- `solver/make_optimizer.py:20-27`、`solver/cosine_lr.py`、`solver/scheduler_factory.py:18-22`

### 2.2 插入位置

- **specific token**：`vit_pytorch.py:364-366`。从 3 个可学习 token 中按 `modality_ids` 选出每张图的一个，然后 `cat(cls_token, selected_specific, patches)`，形成长度 130 的序列（CLS + specific + 128 patch），**参与全部 12 层 self-attention**；位置编码使用独立的 `specific_pos_embed`（初始 0，`:265`、`:368`）。
  - 这不是“旁支”：每层 attention 的 key/value 集合多了一个 token，原 CLS 的注意力分配也随之改变。`patch11 = x[:,2:,:]` 是第 11 层输出、第 12 层之前（`:379`）。
  - shared = `x[:,0]`，specific = `x[:,1]`，都在 final LayerNorm 之后（`:380-381`）。
- **ETGM**：`visa_adapt.py:71-91`。三套池：`shared`、`rgb`、`ir`；`rgb` 用于 `modality_ids==0`，`ir` 用于 `modality_ids!=0`（NIR 与 TIR 共享一套 IR 池）。每池 8 专家、Top-2、每专家 16 query、12 头（默认）。
  - `shared` 池输入 `shared_feature.detach()`（`:80`）；gate/专家参数仍然训练。
  - `specific` 池输入 `specific_feature[ids]`，**不 detach**（`:87`）。
  - `balance = n_experts * mean_batch(softmax(gate))²`（`:59-60`），与模态数无关的负载均衡项。
- **DLFM**：`visa_adapt.py:119-146`。
  - 对 shared/specific 各取与 CLS（或 specific）余弦最高的 40 个 L11 Patch（`:131-136`）；**选中的 Patch 值与索引计算都 detach**。
  - 每张图的 2×16 个 query 与 40 个 Patch 形成 query 中心子图；邻接矩阵在 `no_grad` + FP32 下由节点余弦构造（`:105-113`），`relu` 后加单位阵，D^-1/2 归一化。
  - GCN：`(A @ (nodes @ W))[:,:,0]`，tanh（`:114-116`）；**只有节点值和邻接 detach，query 与 GCN 权重保留梯度**。
  - shared/specific 各自加残差后，与 `out_token` 拼成 33 个 token，经 PostAttention，取 token0 为 local（`:143-146`）。
- **最终检索**：`make_retrieval_feature` `:149-154`：
  `z = cat(normalize(shared), 0.5*normalize(local)) / sqrt(1.25)`；`cos(z,z') = 0.8 cos_s + 0.2 cos_l`，`||z||=1`。
- **模态路由**：specific token 的索引 + ETGM 的 rgb/ir 二路分派；DLFM 与 local 对所有模态共享。没有平台路由。

### 2.3 正向数据流

```
X[3B]  modal_ids[3B]
  │
  ├─ patch_embed -> P[3B,128,768]
  ├─ cat(CLS, specific[m], P) -> [3B,130,768]      specific_pos_embed 初始0
  ├─ blocks 1..11 -> x
  │    patch11 = x[:,2:,:]   (L11 Patch, 128 x 768)
  ├─ block 12 -> final LayerNorm -> shared x[:,0], specific x[:,1]
  │
  ├─ shared ──┬─ bottleneck+classifier -> CE/WRT/ProCA/GPD (原 UAD)
  │           ├─ ETGM.shared(shared.detach()) -> Q_s[3B,16,768]
  │           └─ DLFM 选择 shared 的 top-40 patch (detach)
  │
  ├─ specific ─┬─ ETGM.rgb/ir(specific) -> Q_p[3B,16,768]   (保留梯度)
  │            └─ DLFM 选择 specific 的 top-40 patch (detach)
  │
  ├─ local = PostAttention(cat(out_token, Q_s+res, Q_p+res))[:,0]
  │    ├─ local_classifier(BN(local)) -> local CE 0.1
  │    └─ (不参与主 CE/WRT/ProCA/GPD)
  └─ z = [normalize(shared), 0.5 normalize(local)]/sqrt(1.25)  (1536)
       └─ fusion WRT 0.5 (shared 分量 detach)
检索（mode>0）：返回 z                                          make_model.py:120-127
```

### 2.4 反向梯度流

```
主损失: CE(BN(shared)) + WRT(shared) + 0.01 ProCA(shared) + GPD(shared)
        └─► shared ─► 全 ViT（blocks 1..12、specific token 也经 attention 参与）
        └─► specific 通过 shared CLS 的 self-attention 间接受主损失影响（另有自己的 specific WRT）

新增:
  specific WRT(0.05, raw specific, 逐模态) ─► specific token/层 1..12/ViT
  local CE(0.1, BN(local)) ─► local_classifier/BN/PostAttention/out_token/GCN/query
                              ─► specific query ─► specific token/ViT
                              ─► shared query 路径在 shared_feature.detach() 处被切断
  fusion WRT(0.5, z) ─► z 的 local 部分 ─► local 路径
                     ─► z 的 shared 部分被 make_retrieval_feature(detach_global_for_loss=True) 切断
  balance(1e-4) ─► gate
  DLFM 选中的 Patch 值与邻接 detach ─► local 损失不能改写其读取的 L11 Patch 内容
```

三个 detach 点（[代码事实]）：

1. `visa_adapt.py:80` `shared_feature.detach()`：Shared ETGM 不把梯度回传给 CLS；但专家自己的参数仍训练。
2. `visa_adapt.py:102-103` 与 `:136`：节点值/邻接/选中 Patch detach；局部损失不能直接修改 L11 Patch，但能训练 Query、GCN、out_token，并经 specific query 影响 specific token 与骨干。
3. `processor.py:246-247` + `visa_adapt.py:152-153`：fusion WRT 对 shared 分量 detach；它只能调整 local（及其上游 specific），**不能直接调整 CLS**。

因此 02 的说法“fusion WRT 切断 shared 直接梯度，却仍经 specific 和共享 ViT 间接影响主干”是准确的；应补一句：**主损失也从不作用于 z，z 上只有 WRT，而且 z 的 80% 分量被 detach**。

### 2.5 损失与归一化顺序（ViSA）

| 项 | 输入 | BN | 归一化 | 权重 |
|---|---|---|---|---|
| CE | `classifier(BN(shared))` | BN 后 | 线性 | 1.0 |
| WRT | shared | BN 前 | 无 | 1.0 |
| ProCA | shared | BN 前 | 模态 L2 + 中心 | 0.01 |
| GPD | shared | BN 前 | L2 + 余弦/tau | 1.0 |
| specific WRT | 每个模态切片 raw specific | BN 前 | 无；单独对每个模态做 WRT | 0.05 × ramp |
| local CE | `local_classifier(BN(local))` | BN 后 | 线性 | 0.1 × ramp |
| fusion WRT | z（单位） | 无 | L2（构造时） | 0.5 × ramp |
| balance | ETGM gate | - | - | 1e-4 × ramp |

`loss = base + ProCA/GPD`（`:209-236`）`+ ramp * (specific + local + fusion + balance)`（`:250-255`）；`ramp = min(1, epoch/5)`（`:249`），**第 1 轮就是 0.2**，不是从 0 开始。

注意三套距离尺度：

- shared WRT：raw shared（final LayerNorm 后，norm 约 27.7）；
- specific WRT：raw specific（同样 norm 约 27.7）；
- fusion WRT：单位 z（最大欧氏距离 2）。
WRT 的 softmax 权重对尺度敏感，因此三项的“表面权重”0.05/0.5 不等于梯度比例（idea.md 第 361 行也提醒了这一点）。

### 2.6 优化器与参数规模

- 新参数识别：`make_optimizer.py:20-27`：`base.specific_tokens.*`、`base.specific_pos_embed`、`etgm.*`、`dlfm.*`、`local_bottleneck.*`、`local_classifier.*` → LR ×2。
- scheduler：`scheduler_factory.py:18-22` 把新组的 `lr_min` 与 warmup 起点也乘 2，保持比例；`solver/cosine_lr.py` 增加 list 支持。
- WD：新参数仍走默认 `WEIGHT_DECAY=1e-4`，包括 1-D 的 `specific_tokens`、`specific_pos_embed`、`out_token`、norm 参数；这一点和 SFTC 的处理不同。
- **[静态计算]** ETGM 127,918,104 + DLFM 8,269,056 = **136,187,160** 新参数；base ViT 85,746,432；再加 local BN/classifier 约 0.39M。总参数约为原来的 2.6 倍，新模块单独就超过主干。每个专家只被 Top-2 路由到约 1/4 的样本，存在大量低更新专家。

### 2.7 与评估协议的一致性

- 检索 z（1536）L2 后欧氏；同相机过滤与 baseline 相同。
- z 的余弦是 0.8 shared + 0.2 local；shared 来自被 specific token 修改过的 token 序列，所以与 baseline 的 CLS 不是同一个函数。
- 训练新增损失不排除同相机：specific WRT 逐模态、fusion WRT 都与 baseline 一样按 pid 分正负。
- fusion WRT 的候选集是本卡 batch，不是全 Gallery；其 positives 包括同相机样本，而评估会删除同相机。

### 2.8 结果核对

| 编号 | AS mAP / R1 / R5 / R10 | 相对 H014 |
|---|---|---|
| H005 seed1234 | 11.30 / 29.37 / 45.87 / 54.47 | +0.28 / -0.34 / -0.45 / -0.08 |
| H006 seed2025 | 11.06 / 28.24 / 45.43 / 54.10 | +0.04 / -1.47 / -0.89 / -0.45 |

H005 九宫格 (ΔmAP / ΔR1)：

```
RR +0.45/-1.07   R→N +0.34/+0.04   R→T +0.31/-0.37
N→R +0.42/-0.61  N→N +0.18/-0.37   N→T +0.28/+0.84
T→R +0.23/+0.52  T→N +0.26/+0.04   T→T +0.25/-0.05
```

[实验支持] **9 个方向 mAP 全部上升**；R1 在 RR/NR/NN/TT 等方向下降，在 T→R、T→N、N→T 上升。三个同模态方向的 mINP 全升（+0.27/+0.26/+0.17）。这与 HiHR 的模式几乎同构：mAP 全升 + 同模态 R1 降 + 同模态 mINP 升。02 只给了 TIR→RGB 与 RR 两个例子，未指出 9 宫格全升。

### 2.9 失败机制

**[代码事实 1] 最终 z 是固定 80/20 余弦混合；z 上只有 fusion WRT，且 shared 被 detach。**
主 CE/WRT/ProCA/GPD 全部作用在 shared；z 上唯一损失是 fusion WRT，而它的 shared 分量 detached。因此：
- z 的 80% 部分从未被直接优化以适配 z；
- local 另有自己的 CE（0.1，作用在 local 自身空间），但没有任何损失把 shared 与 local 作为联合 z 做 CE；
- local 在 fusion WRT 中被要求“在 shared 固定的情况下补偿排序”。

**[代码事实 2] specific token 进入共享序列。** 每层 attention 的 key 数从 129 变 130；它的直接身份监督只有逐模态 WRT 0.05，此外通过 ETGM specific 池/local 分支间接接收 local CE 与 fusion WRT 的梯度。它使 shared CLS 的计算路径与 baseline 不同；这不是一个可插拔的旁支。specific token 的梯度会改写全部 12 层（包括 patch_embed 之后的所有 block）。

**[代码事实 3] local 读取的 Patch 值被 detach。** local 只能通过 query/GCN/out_token 适配；它不能直接改写 L11 Patch 的内容。由于 local 的输入是 L11 Patch 的固定快照，local 的可学习自由度主要在于“读哪些 Patch、怎么聚合”，而不是“Patch 内容”。

**[代码事实 4] 参数量 136M、2× LR、全参数共享 clip 1.0。** 新模块梯度与主干梯度共同竞争同一个裁剪预算；新模块的 1-D 参数还带 WD。没有梯度范数日志，无法量化竞争，但这是一个明确的结构风险。

**[实验支持]** H005/H006 均 R1 下降，mAP 增益从 +0.28 缩到 +0.04（seed 变化），说明 mAP 收益不稳定；T→R 的 R1 上升（+0.52）与 RR 的 R1 下降（-1.07）方向相反。

**[机制假设]** 与 HiHR 相同的“固定 20% 稀释”机制：z 的余弦是 0.8/0.2 加权；新增 local 在大量正样本上为正（mAP 全升），但它对最难负样本的区分力不足，导致头部 margin 收缩（同模态 R1 降）。ViSA 的 local 监督比 HiHR 的 parent/child 更弱（只有 fusion WRT 且 shared detach），因此这个解释更自然；但 local 的 Patch detach 与 specific token 的结构扰动也是并行候选原因，无法用现有证据分离。

**[机制假设 2] specific token 的模态特化与 AS 检索目标冲突。** specific WRT 是“模态内”排序（`processor.py:241-242`），它鼓励 specific token 保留模态特有信息；而最终 z 的 80% 来自 shared。specific token 通过 attention 影响 shared，可能把模态特有噪声带入共享表示，在强模态（RGB）上最明显（RR R1 -1.07）。这只是假设，需 T0（token_only）或 specific-off 对照。

### 2.10 可复用教训（ViSA）

1. 若最终检索是 `concat(shared, local)`，必须让**最终描述子本身**接受至少一个直接 CE/WRT，并让 shared 也参与该损失的梯度；否则“局部补偿全局”只是单向的。
2. 固定 20% 权重在训练和推理都生效；在 local 未证明具备强 top-1 判别力前，这个分配会系统性地压缩 CLS margin。
3. 额外 token 进入主干会改变预训练 CLS 的注意力分配，不能把它当成“只在旁边加一个分支”；要做 token-only/T0 对照。
4. 参数量增加 2.6 倍、2× LR、共享 clip，必须报告梯度范数/裁剪频率；当前没有这些日志，无法排除优化预算被新模块挤占。

### 2.11 与 02 档案的差异

- 02 的模块、detach、权重、1536 维公式、seed 数值均**核对正确**。
- 02 未指出 9 宫格 mAP 全升、同模态 mINP 全升、同模态 R1 降的完整模式；只举了两对。
- 02 未指出 z 上**没有 CE**、shared 在 fusion WRT 中 detach、且 z 的 80% 从未被为 z 直接优化。这比“fusion WRT 切断 shared 直接梯度”更具体，是解释 R1 下降的关键不对称。
- 02 未指出 specific token 改变了每层 attention 的 key 集合（129→130），从而 shared CLS 的预训练路径本身被改写。
- 02 未给出 ViSA 新增 136M 参数、2× LR、1-D 参数也带 WD 的事实；这些使“模块本身没学好”与“优化被拖累”无法区分。
- 02 的“T0 token-only 有配置但未发现完成运行”核对正确（`configs/UAD_T0.yml` 存在）；因此没有 token-only 消融可用。

---

## 3. SFTC v2.1（H024）

### 3.1 调用链与新增文件

```
train.py:128 make_dataloader（新增 metadata? 否；仅 TAUR 改）
train.py:130 make_model -> model/make_model.py:38-78 构造 SFTCSource 并挂到 base
train.py:134 make_optimizer -> solver/make_optimizer.py:14-19
train.py:138 do_train -> processor/processor.py:181-255
```

新增/改写：

- `model/backbones/sftc_local.py`（SFTCLocalEncoder、SFTCSource）
- `model/backbones/vit_pytorch.py:122-142`（Attention 接受 local_context）、`:331-333`、`:354-355`（Block 7 注入）
- `model/make_model.py:71-78`（挂载）、`:80-93`（模态 id 映射）、`:95-108`（forward）
- `processor/processor.py:181-186`（set_epoch）、`:200-203`、`:236-249`
- `solver/make_optimizer.py:14-19`
- `utils/sftc_diagnostics.py`、`utils/sftc_inference_probe.py`、`utils/sftc_probe_metrics.py`（诊断）

### 3.2 插入位置

- `BLOCK_INDEX=6`（0-based）→ **第 7 个 Transformer Block 的 Attention**（`vit_pytorch.py:354-355`，`make_model.py:72-74`）。
- 不是残差流里加 token，而是**在该层 attention 的 logits 上拼接一个新来源**：
  - 原 Q/K/V：`q,k,v = qkv(norm1(x))`，N=129（CLS+128 Patch）。
  - 局部来源：同一张图 → `SFTCLocalEncoder` → 128 个 local token（16×8）→ `local_k/local_v` 两个线性层；得到 `K_l/V_l`（128 个 key/value）。
  - `attn = cat([orig_logits, local_logits + source_bias], dim=-1).softmax(dim=-1)`（`:130-135`），然后 `v = cat([orig_v, local_v])`。**一个 softmax 同时归一化 129+128=257 个 key**。
- 模态路由：`source_bias` 形状 `(3, 12)`（模态×head），按 `modal_ids` 取行（`sftc_local.py:125, 171-173`）；`modal_gamma/beta` 形状 `(3,32)` 只作用于 stem（`:72-73, 109-111`）。CNN 主干、K/V 投影、其余模块在三模态间共享。
- 局部 token **不进入 token 序列**，只在该层作为额外 K/V；之后第 8–12 层只看到被该层 attention 改写后的 129 个 token。
- 初始化：`local_k/local_v` 从第 7 层预训练 `qkv` 的 K/V 切片复制（`:137-142`）；CNN 采用 Kaiming 初始化。
- `source_bias`：前 5 个 epoch 固定为 −2（forward 用常量 `bias_init`，`sftc_local.py:171-172`），backward 后 `suppress_bias_gradient` 把它的 grad 清零并删除 optimizer 状态（`:152-159`，`processor.py:239`）。第 6 轮起可学习。

### 3.3 正向数据流

```
imgs[3B]  modal_ids[3B] (folder->sensor 0/1/2)
  │
  ├─ stem: Conv3x3 s2 -> 32ch -> GroupNorm -> (x*gamma[m]+beta[m]) -> GELU
  ├─ stage1: DWConv s2 -> 64ch -> stage2: DWConv s2 -> 128ch
  ├─ f4[64] / f8[128] -> blur 两次/一次 -> concat 192 -> 1x1 256 -> 1x1 768
  ├─ flatten 16x8 -> 128 tokens + 0.1 * 2D-sincos pos -> LN
  ├─ local_k/local_v (768->768)  => K_l,V_l [3B,12,128,64]
  └─ source_bias[m, head]  (3 x 12)
        │
ViT:
  patch_embed -> cls+patch [3B,129,768]
  blocks 1..6 正常
  block 7: q,k,v (129) ; cat logits with local logits + bias -> softmax(257)
           output = attn @ cat([v, V_l]) -> proj -> residual -> mlp
  blocks 8..12 正常
  final LN -> f0 (pre-BN CLS)
  └─ BN -> classifier -> CE; f0 -> WRT/PCA/GPD
检索：f0（768，pre-BN CLS）                                  make_model.py:100-108
```

### 3.4 反向梯度流

```
CE(BN(f0)) + WRT(f0) + 0.01 ProCA(f0) + GPD(f0)
   └─► f0 ─► final LN ─► blocks 12..8 ─► block7 attn ─┬─► 原 qkv/proj/MLP
                                                       ├─► local_k/local_v ─► local CNN ─► stem gamma/beta
                                                       └─► source_bias（前5轮 grad 清零；之后可学）
无 local 身份损失；局部来源的唯一梯度来自最终 CLS 的 UAD 四项损失。
```

- [代码事实] 局部 CNN/KV 的梯度**只**来自最终 CLS 的 CE/WRT/ProCA/GPD；没有 local CE、没有对 128 local token 的直接监督。
- [代码事实] `source_bias` 在前 5 轮不是通过 detach 固定，而是 forward 用常量、backward 后清零 grad；因此它不接收梯度，但也不会产生“零梯度节点”影响 DDP。
- 新参数：`base.sftc.*` 全部进入 optimizer，LR ×3；1-D 参数（source_bias、modal_gamma/beta、norm 的 weight/bias）WD=0，2-D conv/linear WD=1e-4（`make_optimizer.py:14-19`）。
- 全部参数共享一次 `clip_grad_norm_(max_norm=1.0)`（`processor.py:241`）。

### 3.5 损失与归一化顺序（SFTC）

与 baseline 完全相同：`CE(BN(f0)) + WRT(f0) + 0.01 ProCA(f0) + GPD(f0)`；没有新增损失、没有权重改写、没有归一化改写。局部来源只通过 attention 改变 f0。

### 3.6 与评估协议的一致性

- 检索 f0（768）L2 后欧氏；同相机过滤与 baseline 相同。
- 训练 WRT 仍不排除同相机；没有新损失。
- 评估时 `sftc_enabled=cfg.SFTC.TEST_LOCAL_ENABLED=True`（`processor.py:408`）；同权重诊断可关闭，但训练路径禁止关闭（`vit_pytorch.py:334-335` 在 training 且关闭时直接报错）。
- 模态 id 映射：训练用 `training_modal_ids`（`make_model.py:80-85`），评估用 `evaluation_modal_ids` 把数据集 1/2/3 映射到 sensor 0/1/2（`:87-93`）。

### 3.7 结果核对与诊断证据

| 运行 | AS mAP / R1 / R5 / R10 |
|---|---|
| 正式 H024（= ON） | 10.02 / 26.59 / 43.67 / 51.82 |
| 同权重 OFF | 9.1008 / 25.2303 / 41.405 / 50.133 |
| 同权重 patch_only | 10.0205 / 26.6667 / 44.044 / 51.850 |
| 同权重 cls_only | 9.1934 / 25.3552 / 41.686 / 50.367 |
| 独立 UAD H014 | 11.02 / 29.71 / 46.32 / 54.55 |

- [实验支持] ON - OFF = +0.9215 mAP / +1.3583 R1。
- [实验支持] patch_only ≈ ON（差 0.002 mAP / -0.078 R1），cls_only ≈ OFF（+0.093/+0.125）。**局部来源的即时收益几乎全部经过 Patch query 路径，CLS query 直接使用 local 的收益很小。**
- [实验支持] 从 `attention_source_statistics.csv` 计算的平均 local 注意力质量（epoch 120）：
  - CLS query：NIR 4.38%、RGB 5.74%、TIR 7.25%；
  - Patch query：NIR 10.80%、RGB 13.01%、TIR 14.26%。
  若所有 logits 相同，bias=-2 时 128 个 local key 的初始质量为 `128e^-2/(129+128e^-2)=11.8%`。也就是说：**Patch query 基本维持初始的 ~11–14%，CLS query 反而降到 4–7%，模型没有学会让 CLS 更依赖局部来源**。
- [实验支持] `04_local_feature_retrieval.csv`：CNN fused 特征独立检索仅 0.3152 / 1.0929（mAP/R1），local value 0.2784 / 0.7806。CNN 不是独立可检索的身份描述子。
- [实验支持] 九宫格 mAP 全部下降（RR -1.48、R→N -1.46、R→T -0.46、N→R -1.45、N→N -1.43、N→T -0.31、T→R -0.43、T→N -0.27、T→T -1.68）。02 的“9 个模态 mAP 全部下降”正确。
- [实验支持] epoch 120 的日志：`grad(CNN/KV)={'cnn':0.51,'kv':0.29}`，`global_norm=2.46`，`clip=0.406`；`SFTC clipping ... clip_frequency: 1.0`。也就是说**每个记录的 step 都被全局裁剪**，新分支与主干共享一个裁剪预算。

### 3.8 失败机制：ON>OFF 但整体仍弱于独立 UAD

**[代码事实 1] 局部来源是“竞争性 K/V”，不是加性残差。**
一个 softmax 同时归一化 129 个原 key 与 128 个 local key（`vit_pytorch.py:133-135`）。只要 local 分到概率质量，原 129 个 key 的总质量就下降。即便 bias=-2，局部质量也不是 0；初始约 11.8%。因此 block 7 的预训练 attention 路由从第一步起就被改变。

**[代码事实 2] 训练后的 OFF 不是 baseline。**
OFF 只把 local 候选从 softmax 中移除（`sftc_inference_probe.py:85-92`），但 block 7 的 q/k/v 权重是在 257 个 key 的归一化下训练的。OFF 时 softmax 重新归一化到 129 个 key，得到的 attention 与 baseline 的 attention 不同。因此：
```
总效应 = (ON - baseline) = (ON - OFF) + (OFF - baseline)
       = 模块在共同权重上的即时收益 + 联合训练造成的权重漂移代价
```
本实验：`-1.00 = +0.92 + (-1.92)`（mAP），`-3.12 = +1.36 + (-4.48)`（R1）。**模块只回收了联合训练代价的约一半（mAP）/ 约三分之一（R1）。**

**[实验支持 3] 路径诊断。**
patch_only=10.02/26.67 ≈ ON；cls_only=9.19/25.36 ≈ OFF；注意力质量 CLS 4–7%、Patch 11–14%。这说明局部来源主要通过 Patch token 的 attention 改写、再经第 8–12 层影响 CLS；CLS 的直接 local 读取几乎可以忽略。

**[机制假设 1] 优化路径/注意力竞争。**
局部 K/V 是随机初始化的 CNN，前 5 轮 bias 只把它的总质量压到 ~12%，并未关闭；后续可学习后也没有显著增加（Patch 10–14%）。在这段时间里，block 7 的 pretrained q/k 被优化去适应一个含有噪声的新 key 集合；最终权重虽然让 ON 利用 local（ON>OFF），但原 CLS 表征已经被联合训练改变。这是 02 的“优化路径”解释，代码与诊断支持其前提，但无法用单因素证明。

**[机制假设 2] 共享裁剪预算与 3× LR。**
新参数 LR 3 倍、1-D 参数 WD 0；全局 clip 1.0 且每个记录 step 都触发。局部 CNN/KV 的梯度范数 ~0.5/0.3、全局 ~2.5，新分支占用了可观的裁剪预算；在裁剪后，主干的步长被统一缩小。若没有新分支，主干可用同样的 1.0 预算走更远。该假设可以用“相同 clip、关闭新分支的独立训练”检验，但本次没有该对照。

**[机制假设 3] 局部来源的信息量不足。**
CNN fused 特征独立检索 mAP 0.32、R1 1.09，远低于 CLS；它只有在与 CLS 的 attention 上下文一起时才有用（ON>OFF）。因此模块的“即时收益”有限，而联合训练的改变是全局的；当模块无法补偿时，整体退化。这是与 02 的“补偿部分共同训练损失”一致的更具体版本。

**[结论边界]** 不能把 H024 的 -1.00/-3.12 直接归因于“联合 softmax 竞争”或“CNN 噪声”；02 也承认无法唯一归因。但可以确证：**ON>OFF 是同一组权重上的反事实，不是方法相对 baseline 的增量；OFF 本身也是被联合训练改变过的模型。**

### 3.9 可复用教训（SFTC）

1. 新来源进入 softmax 会改变原来源的归一化质量；“负 bias”不是恒等启动，−2 仍有 ~12% 质量。若要恒等启动，需要让 local logits 在初始时远低于原 logits（或使用加性 gating 并零初始化），而不是只加常数 bias。
2. 必须在同一 checkpoint 上同时报告 ON、OFF 与**独立 baseline**；ON>OFF 不能替代“方法 > baseline”。本实验的代价/收益分解（模块回收约一半）应成为标准报告格式。
3. 路径诊断（cls_only/patch_only）比单一 ON/OFF 更有解释力：它指出收益主要来自 Patch 路径，后续设计应把局部监督放在 Patch 侧或直接监督聚合。
4. 新分支与主干共享 clip 时，应记录裁剪频率与各参数组的梯度范数；本实验的 clip_frequency=1.0 说明新分支可能占用预算，但没有分组数据。

### 3.10 与 02 档案的差异

- 02 的模块、Block 7、129×257、初始化、bias、ON/OFF、path、local retrieval 数值均**核对正确**。
- 02 说“新参数 LR 倍率 3，WD 仍 1e-4”不完全正确：**1-D 参数（source_bias、modal_gamma/beta、norm）WD=0**，只有 2-D conv/linear 是 1e-4（`make_optimizer.py:14-19`）。
- 02 未量化初始 local 质量（约 11.8%），也未给出 epoch 120 的 CLS/Patch local 质量（4–7% vs 11–14%）；这两组数是“局部来源主要走 Patch 路径”的直接证据。
- 02 未指出 OFF 会同时把 block 7 的 softmax 从 257 key 重新归一化到 129 key；因此 OFF 不是“同一个模型的纯减法”，ON/OFF 差值会同时包含“local 信息”和“训练时归一化失配”两部分。
- 02 未指出 clip_frequency=1.0 与 3× LR 新分支对共享裁剪预算的潜在影响；这属于未排除的优化混淆。

---

## 4. ICRE / CLEE（H025）

### 4.1 调用链与新增文件

```
train.py:128 make_dataloader
train.py:130 make_model -> model/make_model.py:67-72 若 CLEE.ENABLED:
                             base.enable_clee(...)
train.py:134 make_optimizer（`.clee.` 参数 base LR）
train.py:138 do_train -> processor/processor.py:122 calibrate_clee（DDP 包装前）
                          :189-191 统计开关
                          :236-241 backward/clip
```

新增/改写：

- `model/backbones/clee.py`（CrossLayerEvidenceEnhancement）
- `model/backbones/vit_pytorch.py:297-305`（enable_clee）、`:346-358`（L6/L9 注入）
- `model/make_model.py:67-72`
- `processor/clee_runtime.py`（校准、统计）
- `solver/make_optimizer.py:22-25`（`.clee.` LR、gamma WD）
- `configs/UAD_ICRE.yml`、`config/defaults.py:44-51`

注意：本机 `D:\CODE\Doing\WHU_base\WHU_ICRE` 不存在；以上行号来自审计镜像 `...\_audit_evidence\WHU_base\WHU_ICRE`。正式路径按档案为 `/project/code/WHU_base/WHU_ICRE`。

### 4.2 插入位置

- `vit_pytorch.py:346-354`：在 block 循环中，block 6 之后存 `patches6 = x[:,1:]`；block 9 之后执行
  `patches9 = clee(patches6, x[:,1:])`，然后 `x = cat(x[:,:1], patches9)`。
- **只改写 Patch token**；第 9 层 CLS 没有被 CLEE 直接改写。最终 CLS 只通过 block 10–12 的 attention 间接受到 patches9 的影响。
- CLEE 内部（`clee.py:41-68`）：
  - `norm6/proj6`、`norm9/proj9`：768→256；
  - router `Linear(512→64)→GELU→Linear(64→2)`，逐 token 的 2-way softmax；**末层零初始化**（`:38-39`）→ 初始 0.5/0.5；
  - `source = w6*z6 + w9*z9`；
  - Q 来自 z9，K/V 来自 source，各自 Linear+LN，4 头、每头 64；
  - FP32 attention（`autocast` 关闭），softmax 在 128 个 Patch token 上；
  - `delta = out_proj(attended)`（256→768）；
  - `return patches9 + gamma * delta`（`:68`），`gamma` 标量初始 0.01。
- 模态路由：**没有**。所有模态共享同一模块、同一组参数、同一 gamma；没有 modality embedding 或掩码。
- 无 detach：`patches6` 与 `patches9` 都是主干残差流的活张量；CLEE 的梯度可以回传到 block 1–9 以及 patch_embed。`patches6` 是一条直接的“早期层旁路”。

### 4.3 正向数据流

```
X[3B]
  │
  ├─ ViT blocks 1..6 ─► x6[3B,129,768]
  │                     ├─ CLS 继续
  │                     └─ patches6 = x6[:,1:]  [3B,128,768] ─────┐
  ├─ blocks 7..9 ─► x9[3B,129,768]                                 │
  │                     ├─ CLS 继续                                  ▼
  │                     └─ patches9 ────────────────► CLEE(p6,p9)
  │                                                       │
  │                        z6=proj6(LN6(p6)), z9=proj9(LN9(p9))  (256)
  │                        w = softmax(router(cat(z6,z9)))  (逐 token 2-way)
  │                        source = w6*z6 + w9*z9
  │                        q=LN(q_proj(z9)); k=LN(k_proj(source)); v=LN(v_proj(source))
  │                        A = softmax(QK^T/sqrt(64))  (128 x 128, 4 heads)
  │                        delta = out_proj(A V)  (768)
  │                        patches9' = patches9 + gamma * delta
  │                                                       │
  ├─ x9' = cat(CLS9, patches9') ◄──────────────────────────┘
  ├─ blocks 10..12 ─► final LN ─► f0 (768)
  └─ BN -> CE; f0 -> WRT/ProCA/GPD
检索：f0（pre-BN CLS 768）
```

### 4.4 反向梯度流

```
CE/WRT/ProCA/GPD ─► f0 ─► final LN ─► blocks 12,11,10 ─► x9' ─┬─► CLS9
                                                               └─► patches9' = patches9 + gamma*delta
                                                                     ├─► patches9 -> blocks 1..9 -> patch_embed
                                                                     └─► gamma * delta
                                                                           ├─ gamma（标量）
                                                                           └─ out_proj <- A <- Q/K/V <- proj6/proj9/router <- patches6/patches9
```

- [代码事实] 没有 detach/no_grad/stop-gradient（CLEE 内唯一的 `no_grad` 是统计 `last_stats`，`:63-67`）。
- [代码事实] gamma 的梯度非零（日志 `CLEE gradients`）；`out_proj` 权重梯度非零；整个模块进入 optimizer。
- [代码事实] `.clee.` 参数统一使用 base LR（0.032），不享受 bias 2×；`gamma` WD=0，其余 WD=1e-4（`make_optimizer.py:22-25`）。
- [代码事实-校准] `clee_runtime.py:77-81`：在 8 个身份 × 3 模态 = **24 张图**上计算 `median(gamma*||delta||/||patch||)`，然后把 `out_proj.weight/bias` 乘以 `0.01/median=89.799`。这会：
  - 把初始残差比压到 1%；
  - 同时把 `delta` 及其上游（`attended`、q/k/v、router、proj6/proj9）的梯度放大约 90 倍（因为 `dL/dattended = gamma * W_out^T dL/dp_out`）；`gamma` 的梯度 `dL/dgamma = <dL/dp_out, delta>` 也随 delta 放大；只有 `out_proj` 自身的权重梯度 `dL/dW_out = dL/ddelta ⊗ attended` 不受该缩放影响；
  - 校准集只有 24 张图，且在 fresh ImageNet 主干上测量。

### 4.5 损失与归一化顺序（ICRE）

与 baseline 完全相同：`CE(BN(f0)) + WRT(f0) + 0.01*ProCA(f0) + GPD(f0)`。没有新增损失。CLEE 只改变 f0 的生成路径。

### 4.6 与评估协议的一致性

- 检索 f0（768）L2 后欧氏；同相机过滤与 baseline 相同。
- 训练 WRT 仍不排除同相机；CLEE 没有自己的损失/候选集。
- 所有模态同一模块、同一路由（无路由），所以不存在“只改善 TIR”的条件路径。

### 4.7 结果核对

| 项 | 数值 |
|---|---|
| AS mAP / R1 / R5 / R10 | 8.29 / 22.81 / 38.69 / 47.42 |
| 相对 H014 | -2.73 / -6.90 / -7.63 / -7.13 |
| 校准 median / scale | 0.000111 / 89.799 |
| 末期 gamma | 0.4364 |
| 末期 residual_ratio (RGB/IR/TIR) | 0.2615 / 0.2502 / 0.2598 |
| 末期 layer6 router 均值 (RGB/IR/TIR) | 0.4918 / 0.4816 / 0.5220 |
| gamma grad | -1.7e-2 ～ +1.9e-3（非零） |
| out_proj.weight grad norm | ~4.4e-3（非零） |

九宫格 ΔmAP / ΔR1：

```
RR -5.85/-8.66   R→N -3.93/-6.19   R→T -1.11/-1.40
N→R -3.75/-5.06  N→N -4.48/-7.40   N→T -0.81/-0.66
T→R -1.20/-1.31  T→N -0.74/-0.52   T→T -3.42/-6.00
```

[实验支持] 9 个方向全部下降；RGB→RGB（-5.85 mAP / -8.66 R1）和 NIR→NIR（-4.48/-7.40）退化最大；TIR→TIR 也 -3.42/-6.00。02 的“RGB→RGB、NIR→NIR、TIR→TIR 分别为 -5.85/-4.48/-3.42”正确。

### 4.8 失败机制：明确参与、梯度非零却大幅退化

**[代码事实 1] 这是一个“全 Patch 重写”算子，不是小残差。**
CLEE 在每一层 9 之后对**全部 128 个 Patch** 做 128×128 的全自注意力混合，输出 `delta` 加到每个 Patch。`delta` 的每个 token 是 128 个 `v` 的凸组合，`v` 又是 `source` 的线性投影；因此 delta 是一个**低秩（256 维）、跨 token 高度相关**的全局信号。它会把 Patch 特征“平滑”到一个共享子空间。`gamma=0.436`、`residual_ratio≈0.25` 意味着最终每个 Patch 有约 25% 的范数来自这个全局混合。

**[代码事实 2] 初始化安全只约束起点。**
校准把初始残差压到 1%，但 gamma 是自由标量、out_proj 被放大 90 倍；训练后 gamma 增大 43.6 倍，残差比到 25–28%。02 说的“可学融合没有一直保持初始 1% 扰动”正确；应补上：**校准放大 out_proj 同时放大了该分支的内部梯度**，所以“1% 起点”伴随的是“更大的分支步长”。

**[代码事实 3] Router 没有学出选择。**
router 末层零初始化，训练后三个模态的 layer6 权重均值仍在 0.48–0.53。也就是说，“跨层证据路由”平均而言仍接近 50/50；模块更接近“固定平均 L6/L9 + 可学习投影 + 全 Patch 混合”，而不是“按 token 选择层”。02 说“router 接近均分不证明两层真互补”，可以更强：**本实现里 router 基本没有分化**。

**[代码事实 4] 早期层也被改写。**
`patches6` 没有 detach，CLEE 的梯度直接回传到 block 1–6。虽然模块物理位置在 block 9，它同时对早期特征施加梯度；这是与“只做深层增强”不同的机制。

**[机制假设 A] 深 Patch 平滑破坏身份几何。**
block 10–12 的 CLS 依赖 Patch 作为 key/value；当 Patch 被 25% 的全局低秩信号改写后，不同身份之间的 Patch 差异被压缩，CLS 可用的细节减少。RGB/NIR 退化最大，与“预训练特征越强，被平滑的损失越大”一致。该假设与所有同模态退化相容，但缺 ON/OFF（无法在同一权重上关闭 CLEE）。

**[机制假设 B] gamma 只能全局调节，无法按模态/样本抑制。**
单个标量 gamma 同时控制所有模态、所有 Patch、所有层；一旦某些样本需要小扰动，模型无法局部关闭。与 HiHR 的固定权重类似，这是“不可自适应”的结构风险。

**[机制假设 C] 校准集过小导致初始 1% 只对 24 张图成立。**
24 张图的 median 不能代表 500 个身份、7 个相机、3 个模态的分布；初始残差在某些样本上可能远大于 1%。这是 [机制假设]，但代码事实（8 个索引 × 3 模态）明确。

**[结论边界]** 02 的“增强通路强而非未启用”有日志支持；但“非零梯度/25% 残差”只能证明模块在施加影响，不能证明影响方向是坏的。要证明因果，需要同权重 ON/OFF（把 gamma 设 0 但保留 CLEE 的旁路/权重）或不同 gamma 的训练对照，本实验没有。

### 4.9 可复用教训（ICRE）

1. “初始化 1%”不是安全保证；必须同时监控训练过程中的残差比、gamma、router 分布，并在训练中设置硬上限（如 clamp gamma 或残差范数），否则自由标量会放大到 25%。
2. 对 Patch 做全 token 混合的模块，应评估其是否压缩了 Patch 多样性（例如 Patch 余弦分布、有效秩）；只看最终 mAP 不够。
3. Router 零初始化后若长期保持 ~50/50，应判定“路由未学习”，不能把结构名称当作机制事实。
4. 对早期层有旁路梯度的模块，不能只按物理插入层（L9）解释其影响；要列出所有梯度路径。

### 4.10 与 02 档案的差异

- 02 的结构、gamma、残差比、router、梯度、九宫格数值均**核对正确**。
- 02 未指出：CLS 在第 9 层没有被改写，最终 CLS 完全通过 block 10–12 间接受影响；也没有指出 CLEE 对 block 1–6 有直接梯度旁路。
- 02 未指出：router 训练后仍接近 0.5（本实现没有学出逐 token 层选择），因此“token 级路由”更接近“固定平均”。
- 02 未指出：校准把 out_proj 放大 89.8 倍，同时放大了分支内部梯度；校准集仅 24 张图。02 只说了“校准到 1%”。
- 02 的“已被最终检索监督组织过的深层 Patch 被未经同等预训练的融合注意力改写”中，“未经同等预训练”容易读成“未训练”；实际是随机初始化后联合训练。建议改为“随机初始化、在 500 身份上从头学习”。
- 02 未提供 ON/OFF 或 gamma=0 诊断；因此“深层 Patch 被改写导致退化”仍是机制假设，不应写成已证事实。

---

## 5. TAUR v2（H031）

### 5.1 调用链与新增文件

```
train.py:128 make_dataloader（CAMERA_PRIORITY=True 时改用 CameraPKMSampler）
train.py:130 make_model -> model/make_model.py:126 enable_thermal_adapters
train.py:134 make_optimizer -> solver/make_optimizer.py:20-27（adapter LR ×3）
train.py:138 do_train -> processor/processor.py:194-293
```

新增/改写：

- `model/backbones/thermal_adapter.py`
- `model/backbones/vit_pytorch.py:299-310`、`:329-340`、`:354-357`
- `loss/taur_rank_loss.py`
- `processor/taur_grad.py`
- `datasets/sampler_camera.py`、`datasets/make_dataloader.py:26-34, 100-127`、`datasets/bases.py:71-108`
- `loss/wrt_loss.py`（stable 版本）、`loss/make_loss.py:19-20`
- `configs/TAUR.yml`

### 5.2 插入位置

- `vit_pytorch.py:354-357`：在 block 循环中，layer 6 和 layer 9 之后（1-based），若 `str(layer) in thermal_adapters`：
  `x = adapter(x, modal_id)`。
- `thermal_adapter.py:21-34`：
  - `thermal = modality_ids == 3`（数据集 1=RGB、2=NIR、3=TIR）；
  - 只取 `tokens[thermal,1:]`（Patch，不含 CLS）；
  - `LN → 768→128 → GELU → 128→768`；
  - `h_norm = ||patches||`（每图），`u_norm = ||residual||`（每图），`scale = rho*h_norm/(h_norm+u_norm+eps)`（`:30`）；
  - `updated = patches + (scale*residual).float()`，写回 `result[thermal,1:]`；CLS 与 RGB/NIR 行不变（`:33`）。
  - 因此**每个样本、每一层的残差范数严格小于 `rho*||patch||`（rho=0.1）**；两层最多两段 10%。
- **模态路由**：硬编码 `modality_ids==3`，只改 TIR Patch；不存在共享的模态条件分支。TIR 之外的模态完全不受 adapter 影响。

### 5.3 正向数据流

```
imgs[3B]  modids[3B] (1/2/3)
  │
  ├─ ViT patch_embed + cls+pos
  ├─ blocks 1..6
  ├─ if modal==3: TIR patches' = TIR patches + bounded(LN->128->GELU->768)   (rho 0.1)
  ├─ blocks 7..9
  ├─ if modal==3: TIR patches'' = ... (second adapter)
  ├─ blocks 10..12 -> final LN -> f0 (768, pre-BN CLS)
  │
  ├─ CE(BN(f0))
  ├─ WRT(f0)                      (stable=True)
  ├─ ProCA(f0) / GPD(f0)
  └─ TAUR rank loss (separate graph): cosine on normalize(f0), 4 directions,
       negatives: same local batch, exclude same camera and same sample
检索：f0（768，pre-BN CLS）      make_model.py:131-145
```

### 5.4 反向梯度流

```
UAD 四项:
  loss = 1.0*CE + 1.0*WRT + 0.01*ProCA + 1.0*GPD
    └─► f0 ─► final LN ─► blocks 12..10 ─► patches'' ─► adapter9 ─► patches' ─► adapter6 ─► blocks 1..6
                                         └─► CLS 始终在残差流中，不直接经过 adapter
    └─► adapter 参数（通过 TIR Patch 路径）

TAUR rank:
  rank_loss = 0.25*(T2T+T2ALL+R2T+N2T)  on global_feat (pre-BN CLS, 本卡 3B)
    └─ torch.autograd.grad(rank_loss*lambda, adapter_parameters)   (taur_grad.py:7-9)
    └─ merge 到 .grad                                               (taur_grad.py:58-66)
    └─ 主干/CLS 无 rank 梯度（按设计）

DDP:
  no_sync 包裹 forward+backward+autograd.grad（processor.py:220-288）
  synchronize_gradients(model) 手工 all-reduce mean（taur_grad.py:68-100）
  scaler.unscale_ -> clip_grad_norm_(all, 1.0) -> step
```

关键事实：

- [代码事实] rank 梯度只到 adapter，**不到主干**；这是硬性的梯度路由，不是 detach 的副作用。
- [代码事实] rank loss 不加入打印的 loss 标量；它只以梯度合并的方式参与优化（`processor.py:281-288`）。
- [代码事实] `WRT_STABLE=True` 会改变**原 WRT** 的数值：`stable` 分支用 `-inf` mask + softmax，替代原实现的 `exp(dist-max)*mask/(Z+1e-6)`（`wrt_loss.py:6-13`）；这是对所有参数的训练目标改动，不只是新分支。
- [代码事实] 新参数 LR ×3（`make_optimizer.py:20-27`）；bias 的 `lr` 被 adapter 分支覆盖为 `BASE_LR*3`（不再乘 `BIAS_LR_FACTOR=2`）；WD 仍 1e-4。日志：Adapter 参数 **398,080**。

### 5.5 损失与归一化顺序（TAUR）

| 项 | 输入 | BN | 归一化 | 度量 | 权重 |
|---|---|---|---|---|---|
| CE | BN(f0) | 后 | 线性 | logits | 1.0 |
| WRT | f0 | 前 | 无 | 欧氏；stable softmax | 1.0 |
| ProCA | f0 | 前 | 模态 L2 | 平方欧氏 | 0.01 |
| GPD | f0 | 前 | L2 | 余弦/tau | 1.0 |
| rank | f0 | 前 | L2 | 余弦（点积） | λ(t): 0→0.1 (epoch 5→20) × 0.25/方向 |

rank 细节（`taur_rank_loss.py`）：

- `similarity = normalize(f) @ normalize(f)^T`（`:23-24`）；
- `eligible = (cid_i != cid_j) & (sample_id_i != sample_id_j)`（`:25-26`）——**排除同相机与同一张原图**；
- 四个方向：`T2T (3,3)`、`T2ALL (3,None)`、`R2T (1,3)`、`N2T (2,3)`（`:28-30`）；
- `candidates = eligible & gallery`；`pos = candidates & same_pid`；`neg = candidates & ~same_pid`；
- 每个 query：`neg_lse = logsumexp(s_neg/tau)`，`pair = tau*softplus(neg_lse + (margin - s_pos)/tau)`，对正样本**取平均**（`:41-46`）；
- 缺方向的 loss 置 0，仍按固定 0.25 求和（`:60`），**不重新分配权重**；
- `rank_weight`：epoch≤5 为 0；之后线性升到 `RANK_WEIGHT=0.1`，在 epoch 20 达到满值（`:62-66`）。

### 5.6 与评估协议的一致性

这是五组中**唯一**让新增损失在训练时显式排除同相机的：

- [代码事实] rank loss 的 `eligible` 排除同相机（`taur_rank_loss.py:25`），与评估的 `remove = g_camid == q_camid` 一致；
- 但原 WRT/ProCA/GPD 仍不排除同相机；
- rank 的 gallery 只是本卡 192 图里的一个模态子集（例如 T2T 平均 ~49 个负样本），不是 93,609 的全 Gallery；Positives 平均 ~2 个；
- 评估用全 Gallery 的余弦排序；rank 用本卡 batch 的余弦排序。候选分布差异巨大。

[实验支持] 末期日志：T2T valid_q=6632/7488、neg/q=49.23、pos/q=1.99；T2ALL valid_q=7372/7488、neg/q=147.54、pos/q=5.94；`positive_cameras/q≈1.36`。四方向都有大量有效 query，不是“空损失”。

### 5.7 结果核对

| 项 | 数值 |
|---|---|
| AS mAP / R1 / R5 / R10 | 10.57 / 29.40 / 47.17 / 55.08 |
| 相对 H014 | -0.45 / -0.31 / +0.85 / +0.53 |
| TIR→TIR (R1/mAP/mINP) | 21.08 / 13.48 / 2.15（baseline 20.75/13.52/2.28） |
| TIR→RGB (R1/mAP/mINP) | 7.45 / 5.39 / 0.97（baseline 7.40/5.85/1.30） |
| RGB→RGB (R1/mAP) | 41.22 / 25.29（baseline 42.48/26.53） |

九宫格 ΔmAP / ΔR1：

```
RR -1.24/-1.26   R→N -0.81/-0.66   R→T -0.33/-0.14
N→R -0.58/0.00   N→N -0.13/+0.05   N→T -0.21/+0.84
T→R -0.46/+0.05  T→N -0.24/-0.71   T→T -0.04/+0.33
```

[实验支持] **9 个方向 mAP 全部下降；R1 有升有降；总体 R5/R10 上升。** 02 只列了 TIR→TIR、TIR→RGB、RGB→RGB，未指出 9 宫格 mAP 全降；这是“R5/R10 升但 mAP 降”的重要证据：更多正样本进入前 10，但整体 AP（正样本排序质量）变差。

### 5.8 失败机制：新增损失非零但指标分化

**[代码事实 1] rank 梯度只占 adapter 总梯度约 1.5–2.1%，余弦 0.29–0.35。**
日志 `TAUR mean Adapter gradient contribution before clipping`（epoch 120 三次记录）：

```
uad_adapter_norm 0.00232, rank_adapter_norm 4.91e-05, ratio 0.02117, cosine 0.292
uad_adapter_norm 0.00307, rank_adapter_norm 4.72e-05, ratio 0.01538, cosine 0.333
uad_adapter_norm 0.00289, rank_adapter_norm 4.53e-05, ratio 0.01570, cosine 0.351
```

也就是说，即使 λ 已达 0.1，**新损失对 adapter 的直接梯度只有 UAD 四项的 ~2%**；对主干是 0（按设计）。因此“rank loss 非零”远不等于“它能改变模型”；这是比 02 的“proxy 与全 Gallery 不一致”更直接的代码/日志事实。02 完全未引用这一日志。

**[代码事实 2] 采样器被改变，且 02 未提。**
`TAUR.yml` 中 `DATALOADER.CAMERA_PRIORITY: True`；`make_dataloader.py:100-127` 在 DDP 下改用 `CameraPKMSampler`（`sampler_camera.py`）。该采样器：
- 对每个身份/模态，优先从**不同相机**抽 K 个样本（若可用）；
- 全局 plan 由所有 rank 独立按同一 seed 生成，再按 `rank*local_batch` 切片；
- epoch 步数由“最大完整批”决定；日志为 **117 步/epoch**，baseline 是 114 步（多 2.6% 更新）；
- 每个 rank 仍 16 身份 × 4 实例，但同身份的 4 个实例更可能来自不同相机，WRT/ProCA/GPD 的输入分布随之改变。

这不是新损失，但会改变**所有原有损失**。`TAUR/idea.md:433` 自己写了“由于采样器改动可能单独影响性能，正式消融应包含使用相同采样策略的 UAD 对照”；`configs/UAD_TAUR_control.yml` 存在，但本机与审计镜像都**没有该 control 的日志**。因此 H031 的 -0.45/-0.31 不能归因于 adapter/rank loss。

**[代码事实 3] WRT_STABLE 改了原 WRT。**
`wrt_loss.py:6-13` 新增 `stable` 分支；`make_loss.py:19-20` 对所有训练启用。它替换了原来的 `exp(dist-max)*mask/(Z+1e-6)` 与 `mask==0` 的处理；在 PKM 每行都有正/负样本时数值差异可能不大，但它是对**原损失**的改动，不能忽略。

**[实验支持 4] 指标模式。**
- 9 宫格 mAP 全降、R5/R10 升：更多正样本进入前 10，但 AP 下降，说明正样本在列表中的“顺序/位置”变差，而不是“完全没找到”。
- TIR→TIR R1 +0.33、mAP -0.04、mINP -0.13：首位略升，但最后相关正样本变差。
- RGB→RGB mAP -1.24：不是“只影响 TIR”的模式；adapter 只改 TIR，RGB→RGB 的变化只能来自共享主干（UAD 损失 + 采样器）或 TIR 训练对共享权重的间接影响。

**[机制假设 A] 直接原因：rank loss 梯度太小。**
λ=0.1、rank/UAD≈0.02，且 rank 对主干为 0。即使 rank 的代理完全正确，它也只能在 adapter 的 398k 参数上施加约 2% 的额外方向修正；而 adapter 的残差被 rho=0.1 限制。要在 9 个方向上改变 mAP，幅度不足。更可能的结果是“几乎不变”或“被 UAD 梯度淹没”，与 mAP -0.45 的小幅下降相容（本组没有同配置重复种子，不能量化噪声）；R5/R10 的 +0.85/+0.53 更可能来自采样器/训练轨迹，而非 rank。

**[机制假设 B] 代理目标与 AP 的差异。**
rank 用每 query 的正样本平均 + 本卡 ~49 个负样本的 logsumexp；评估用全 Gallery 的 AP。两者对“最难负样本”和“正样本顺序”的权重不同：把更多正样本推进前 10 可能提高 R5/R10，但若正样本内部顺序变差或新增了更难负样本，mAP 下降。该解释与 02 的“proxy 不等于 AP”一致，但在 rank 梯度仅 2% 的前提下，它只能解释很小一部分。

**[机制假设 C] 相机优先采样改变原损失难度。**
把 K 个实例从“可能同相机”改为“尽量跨相机”，会让 WRT/ProCA/GPD 的正样本更难、负样本更接近评估分布；这通常能改善泛化，但也可能改变 top-1 的尖锐度。R5/R10 升、mAP/R1 降与“召回变宽、排序变钝”相容。由于没有 matched control，无法定量。

**[结论边界]** 02 的“限制 rank 梯度到 Adapter 缩短了可改进子空间”是一个合理方向，但现有日志显示更直接的事实是：**rank 梯度只有 UAD 的 ~2%，且主干为 0**；再加上 sampler/WRT-stable 两个未控制的改动，H031 的指标分化不能归因于 rank loss。02 对“四方向有效”核对正确，但缺少梯度占比与采样器两项关键证据。

### 5.9 可复用教训（TAUR）

1. 新增损失的“非零”必须升级为“相对原损失的梯度占比 + 对主干的梯度”；本实验的 2% 说明 λ=0.1 只是名义权重。
2. 只把梯度限制在 adapter 时，adapter 的可改进子空间（398k、残差 ≤10%）与目标（全 Gallery AP）之间差距过大；若想影响最终 CLS，必须让新目标至少部分回传到主干，或直接监督最终描述子。
3. 改采样器/原损失必须配套 matched control；本实验配置了 control 但未跑，导致无法归因。
4. 代理损失应报告候选集大小（本例 T2T ~49 负样本）与全 Gallery 的差距；否则“全负样本”会被误读。

### 5.10 与 02 档案的差异

- 02 的结构、四方向、margin/temp、0.25/方向、warmup/ramp、autograd 路由、DDP 同步、LR 3×、WRT_STABLE 均**核对正确**；数值也正确（含 TIR→TIR、TIR→RGB）。
- 02 **完全遗漏 `CAMERA_PRIORITY=True` / `CameraPKMSampler`**（`datasets/make_dataloader.py:100-127`、`datasets/sampler_camera.py`），也未提步数 117 vs 114。该改动会影响四个原损失，是 H031 归因的主要混淆。
- 02 未引用日志中的 `TAUR mean Adapter gradient contribution`：rank/UAD ≈ 1.5–2.1%、cosine ≈ 0.29–0.35。该事实直接削弱“rank loss 导致指标分化”的解释。
- 02 说“新增损失非零但指标分化”时，把原因主要放在 proxy/全 Gallery 不一致与梯度限制；应补充“即便在 adapter 上，rank 梯度也只有 UAD 的 ~2%”这一更直接的量级事实。
- 02 的“限制 rank 梯度只更 Adapter 保护了 shared 参数免受直接附加梯度，却缩短了可改进的子空间”应标为机制假设；目前没有 matched control 或 ON/OFF 证明。
- 02 未指出 WRT_STABLE 作用于所有参数（不仅是新分支），因此 H031 的 baseline 不是严格的“原 WRT”。

---

## 6. 跨实验综合

### 6.1 五组方法的“最终检索特征 / 新损失 / 归一化”总表

| 方法 | 最终检索张量 | 维度 | 最终特征是否被新损失直接监督 | 新损失 | 新参数（静态计算） | 新参数 LR | 训练是否排除同相机 |
|---|---|---:|---|---|---:|---|---|
| Baseline | `global_feat`（pre-BN CLS） | 768 | - | 无 | - | - | 否 |
| HiHR | `F` = 3 段单位拼接 | 1792 | 是：CE/WRT/ProCA/GPD 全在 F | parent/child CE、triplet、HHL | TMF 4.61M + 3 个分类器 1.41M ≈ 6.02M | 1.0× | 否（WRT/triplet） |
| ViSA | `z` = [u_s, 0.5 u_l]/√1.25 | 1536 | 只被 fusion WRT；CE/ProCA/GPD 只在 shared | specific WRT、local CE、fusion WRT、balance | ETGM 127.9M + DLFM 8.27M + local head ≈ 136.6M | 2.0× | 否（specific/fusion WRT） |
| SFTC | `f0`（pre-BN CLS） | 768 | 是（仅原四项） | 无 | CNN 0.26M + local KV 1.18M ≈ 1.44M | 3.0× | 否 |
| ICRE | `f0`（pre-BN CLS） | 768 | 是（仅原四项） | 无 | CLEE 0.826M | 1.0× | 否 |
| TAUR | `f0`（pre-BN CLS） | 768 | 是（仅原四项） | rank（梯度只到 adapter） | 2 × 0.199M = 0.398M | 3.0× | 原 WRT 否；rank 是 |

注：参数计数为本次用 `torch` CPU 实例化模块的静态结果；不含 baseline 已删除的 BN/classifier（HiHR 中它们不在图里）。

### 6.2 为什么 mAP 升但 R1 降（HiHR、ViSA）

**共同代码事实：固定权重、单位化、拼接。**

```
HiHR: cos(F,F') = 0.8 cos0 + 0.1333 cosp + 0.0667 cosc
ViSA: cos(z,z') = 0.8 coss + 0.2 cosl
```

两者的新增分量合计固定占 20% 的平方权重，且都不随样本/模态自适应。

**共同实验支持：9 宫格 mAP 全升 + 同模态 mINP 全升 + 同模态 R1 降。**

```
HiHR ΔmAP 全部 >= +0.17；ΔR1: RR -1.54, NN -0.70, TT -1.55；ΔmINP: RR +0.53, NN +0.32, TT +0.36
ViSA ΔmAP 全部 >= +0.18；ΔR1: RR -1.07, NN -0.37, TT -0.05；ΔmINP: RR +0.27, NN +0.26, TT +0.17
```

两个方法的模块/监督差异很大（HiHR 的 parent/child 接收主损失；ViSA 的 local 只受 CE 0.1 + fusion WRT 0.5，且 shared detach），但指标模式几乎同构。**最简约的共同因子是“固定 20% 的余弦混合”，不是某个具体监督项。**

**机制假设（数学上的一阶解释）：**
对每个 query，R1 由“最近正样本 vs 最难负样本”的余弦差决定，mAP 由全部正样本的排序决定。混合后：
- 若新增分量在大量正样本上有正贡献，则许多正样本上移 → mAP/mINP 升；
- 若新增分量在“最难的那一对”上贡献接近（或对负样本贡献也大），则 `0.8*cos0 + 0.2*cos1` 的 margin 相对 `cos0` 被压缩并加入噪声 → R1 降。
- 新增分量如果更像“通用/平滑”方向（对所有候选都有正相似度），就正好产生“平均排序改善、头部 margin 变差”的模式。

**为什么不能只归因于“混合”：**
- HiHR 还改了分类器（Cosine scale 16 vs BN+Linear）、两段式调度、WRT 尺度；ViSA 还加了 specific token、136M 参数、2× LR。
- 两者都没有 T0/ON-OFF，无法把“混合”与“优化路径”分开。
- 因此结论应写成：**固定 20% 混合是唯一与两个方法都相容的共同结构原因；现有证据不能排除分类器/两段式/额外 token/参数量等并行原因。** 02 的“相似度分配给三部分，改善平均位置但削弱最近邻边界”方向正确；本审计补上了公式、9 宫格与 mINP 证据，并把因果强度限定为“相容，未证明”。

**与 TAUR 的对比：** TAUR 没有新增拼接描述子，而是只改 TIR Patch；它的模式是“9 宫格 mAP 全降 + R5/R10 升”，与 HiHR/ViSA 的“mAP 全升 + R1 降”不同。这反过来说明“mAP/R1 分化”不是所有模块的普遍结果，而是与最终描述子的构造方式有关。

### 6.3 为什么 SFTC 在共同权重上 ON>OFF，但整体仍弱于独立 UAD

代码级可证的分解：

```
ON  - baseline = -1.00 mAP / -3.12 R1
OFF - baseline = -1.92 mAP / -4.48 R1
ON  - OFF      = +0.92 mAP / +1.36 R1
```

- [代码事实] ON/OFF 是**同一个 checkpoint**、同一组主干权重，仅推理时是否加入 local K/V（`sftc_inference_probe.py:8, 85-92`）。
- [代码事实] OFF 时 block 7 的 softmax 从 257 key 重新归一化到 129 key；训练时的 q/k 是在 257 key 下优化的，因此 OFF 不是 baseline。
- [代码事实] 局部来源是竞争性 K/V（同一 softmax），初始 local 质量在 bias=-2 时约 11.8%，train 后 Patch query 10.8–14.3%、CLS query 4.4–7.3%。
- [实验支持] patch_only ≈ ON、cls_only ≈ OFF：收益主要经 Patch 路径再传播到 CLS。
- [结论] “模块有效”与“方法优于 baseline”是两个命题。H024 只证明了前者在共同权重上的局部成立；联合训练把主干改变了 1.92/4.48，模块只回收了 0.92/1.36。**不能用 ON>OFF 抵消 ON<baseline。**

02 的结论方向正确；本审计新增了“OFF 重新归一化 + local 质量分布 + 代价/收益分解”三项可核证据。

### 6.4 为什么 ICRE 明确参与、梯度非零却大幅退化

- [代码事实] 全 Patch 的 128×128 自注意力混合，delta 低秩（256）且跨 token 相关；gamma 自由标量；无 detach；对 L6 有直接梯度旁路。
- [实验支持] gamma 0.01→0.436，残差比 1%→25–28%，router 仍 ~50/50，梯度非零；9 宫格全降，RGB/NIR 同模态最惨。
- [机制假设] 25% 的全局平滑改写破坏了 Patch 的身份细节，进而破坏 CLS 的 key/value；gamma 不能按模态/样本局部关闭；out_proj 被放大 89.8× 也放大了分支内部步长。
- [边界] 没有 ON/OFF 或 gamma=0 诊断；“明确参与”只能否定“模块未启用”，不能证明“退化由该模块造成”。

### 6.5 为什么 TAUR 新增损失非零但指标分化

- [代码事实] rank 梯度只到 adapter；日志 rank/UAD ≈ 1.5–2.1%，cosine ≈ 0.3；对主干为 0。
- [代码事实] 候选集是本卡 ~49 个 TIR 负样本（T2T），不是 93k Gallery；正样本按 query 平均，不是 AP。
- [代码事实] CameraPKMSampler（同相机实例被替换为跨相机）与 WRT_STABLE 同时改变了原训练目标；无 matched control 日志。
- [实验支持] 9 宫格 mAP 全降、R5/R10 升、R1 混合；TIR→TIR R1 +0.33 但 mAP/mINP 降。
- [机制假设] 名义 λ=0.1 被 2% 的梯度占比抵消；指标变化更可能来自采样器/训练轨迹；proxy 与全 Gallery 的差异只能解释很小的部分。02 把原因主要放在 proxy 不一致与梯度限制；更直接的事实是“梯度太小 + 采样器未控制”。

### 6.6 协议一致性对照

| 方法 | 训练正/负是否排除同相机 | 最终检索是否与训练度量一致 | 备注 |
|---|---|---|---|
| Baseline | 否 | 近似：WRT 欧氏 vs 评估余弦；GPD 余弦；CE BN+线性 | 训练/评估本来就不完全一致 |
| HiHR | 否 | WRT 在 9.78F 上欧氏（单位 F，与余弦单调）；PCA/GPD 余弦；CE cosine | 尺度 9.78 只匹配中位数 |
| ViSA | 否 | specific WRT 在 raw specific；fusion WRT 在单位 z；主损失在 shared | z 的 80% 分量被 detach，无 CE on z |
| SFTC | 否 | 与 baseline 相同 | 无新损失 |
| ICRE | 否 | 与 baseline 相同 | 无新损失 |
| TAUR | 原 WRT 否；rank 是 | rank 用余弦，最接近评估；原 WRT 仍是欧氏 | rank 候选仅本卡 ~49 负样本 |

**关键结论：** 5 组中只有 TAUR 的新损失显式遵守评估的同相机排除规则；但它的候选集仍是本卡 batch，不是全 Gallery。其余 4 组的新损失/主损失都允许同相机正负样本，与评估的“排除全部同相机”不一致。这一点 02 在 1.3 只说评估侧，未在每组中对照。

---

## 7. 对 `02_Previous_Experiments.md` 的勘误/补充清单

| 章节 | 性质 | 内容 |
|---|---|---|
| 2.2.4 HiHR | 遗漏（重要） | WRT 标定参考是**已训练 UAD**，被标定是 **fresh HiHR**；json 自述两者训练状态不同。正/负 ratio 8.13 vs 11.77（相差 1.45×），单标量无法同时匹配；WRT 每样本梯度从 0.00137 变到 0.0627。 |
| 2.2.4 HiHR | 遗漏 | TMF 的 Patch 融合只有 3 个“4 头平均”的标量，128 Patch 共用；不是逐 Patch/逐 head。 |
| 2.2.6 HiHR | 遗漏 | 9 宫格 mAP 全升；三个同模态 R1 全降、mINP 全升。02 只给了 TIR 的例子。 |
| 2.2.4 HiHR | 可能误读 | `idea.md` 的“Parent/Child 各自仅接受弱辅助监督”在代码中不成立为“只受弱监督”：F 的主 CE/WRT/ProCA/GPD 全部回传到 up/uc。02 的 2.2.4 虽写了主损失作用于 F，但未点明这一点。 |
| 2.2.7 HiHR | 可加强 | Stage1（60 轮，无 ProCA/GPD）相对 baseline epoch60 是 +0.82/+1.00；Stage2 的 R1 增益 +0.58 远小于 baseline 后半程 +2.72。mAP 增益在 Stage2 才出现，不能归因于 TMF 本身。 |
| 2.3.4 ViSA | 遗漏（重要） | 最终 z 上**没有 CE**；唯一看 z 的 fusion WRT 把 shared 分量 detach。80% 的 CLS 从未被直接优化以适配 z；这是比“切断 shared 直接梯度”更强的结构不对称。 |
| 2.3.4 ViSA | 遗漏 | specific token 进入共享序列，每层 key 129→130；shared CLS 的预训练计算路径本身被改写，不是旁支。 |
| 2.3.4/2.3.5 ViSA | 遗漏 | ETGM 127.9M + DLFM 8.27M ≈ 136.2M 新参数（约 2.6× 总参数），2× LR，1-D 参数也带 WD；共享 clip 1.0。02 未提。 |
| 2.3.6 ViSA | 遗漏 | 9 宫格 mAP 全升；同模态 mINP 全升；同模态 R1 降。与 HiHR 同构。 |
| 2.7.4 SFTC | 错误（细节） | “WD 仍 1e-4”不正确：1-D 参数（source_bias、modal_gamma/beta、norm）WD=0；只有 2-D conv/linear 是 1e-4（`make_optimizer.py:14-19`）。 |
| 2.7.4/2.7.6 SFTC | 遗漏 | 初始 local 注意力质量约 11.8%（bias=-2，128 local vs 129 原 key）；epoch 120 时 CLS query 4.4–7.3%、Patch query 10.8–14.3%。Patch 路径收益占主导有直接注意力证据。 |
| 2.7.6 SFTC | 遗漏 | OFF 会同时把 block 7 的 softmax 从 257 重新归一化到 129 key；OFF 不是“同一模型减 local”。 |
| 2.7.9 SFTC | 可加强 | 代价/收益分解：ON-baseline -1.00/-3.12；OFF-baseline -1.92/-4.48；ON-OFF +0.92/+1.36。模块只回收约一半/三分之一。 |
| 2.8.4 ICRE | 遗漏 | only Patch 被改写；L9 CLS 不变，最终 CLS 只经 L10–12 间接受影响；`patches6` 无 detach，梯度可直达 block 1–6。 |
| 2.8.6 ICRE | 遗漏 | 校准 out_proj 乘 89.799，同时把分支内部梯度放大约 90×；校准仅 24 张图（8 身份×3 模态）。 |
| 2.8.6 ICRE | 可加强 | router 训练后仍 ~0.48–0.53，路由没有学出分化；模块更接近“固定平均 L6/L9 + 投影”。02 只说“接近均分不证明互补”，未说“路由未学习”。 |
| 2.8.7 ICRE | 过度推断风险 | “被未经同等预训练的融合注意力改写”容易读成“未训练”；实际是随机初始化后联合训练。建议改为“随机初始化、在 500 身份上从头学习”。 |
| 2.12.4/2.12.5 TAUR | 遗漏（重要） | `DATALOADER.CAMERA_PRIORITY=True` → `CameraPKMSampler`；K 实例优先跨相机；117 steps/epoch vs baseline 114。02 完全未提，且 idea 自己要求 matched control，但 control 未跑（只有 config）。 |
| 2.12.4 TAUR | 遗漏（重要） | 日志 `TAUR mean Adapter gradient contribution`：rank/UAD ≈ 1.5–2.1%，cosine ≈ 0.29–0.35。rank 对主干为 0。02 未引用。 |
| 2.12.6 TAUR | 遗漏 | 9 宫格 mAP 全降；R5/R10 升；R1 混合。02 只举三对。 |
| 2.12.6 TAUR | 过度推断 | 把指标分化主要归因于“proxy 与全 Gallery 不一致 + 梯度限制”；更直接的事实是 rank 梯度只有 2% + 采样器未控制。应将“梯度限制缩短子空间”标为假设。 |
| 2.12.4 TAUR | 可加强 | WRT_STABLE 作用于所有参数（原 WRT），不仅新分支；因此 H031 的 baseline 不是严格的原 WRT。02 提了该工程偏差，但未说明其作用范围。 |
| 1.3/全局 | 遗漏 | 训练 WRT/ProCA/GPD 都不排除同相机，评估排除全部同相机；只有 TAUR 的新 rank loss 对齐评估，且候选仍是本卡 batch。 |
| 1.3/全局 | 未说明 | 所有 1000 实验的**训练身份数是 500**（日志 `identities: 500`）；“1000”是数据集/评测口径。02 未在每组配置中说明。 |

上述“遗漏”不是要求 02 重写全部内容；建议在 02 的相应章节加一行“机制补充”，并在 phase2 的设计中把未控制项写进消融计划。

---

## 8. 可复用教训与 phase2 检查清单

### 8.1 关于“最终描述子”

1. **谁是最终检索特征，必须唯一且明确。** 5 组最终特征分别是 F(1792)、z(1536)、f0(768)、f0(768)、f0(768)。只有 final tensor 上的损失才决定检索；旁支损失再多，也不能替代最终描述子上的直接监督。
2. **固定权重拼接会改变余弦几何。** 若 `cos = (1-w) cos0 + w cos1`，w 固定则 CLS margin 必被压缩 `w` 比例（一阶），除非新增分量在最难对上提供更强分离。使用固定 w 时，必须报告“单 CLS / 单 local / 混合”三组检索，以及 ON/OFF。
3. **拼接前各分量是否 L2、最终是否 L2，必须写清。** HiHR/ViSA 都先单位化再拼、再整体单位化；这等价于余弦加权平均。若分量未单位化，范数会隐式决定权重，风险更大。
4. **最终描述子上的 CE 不能省。** ViSA 的 z 没有 CE；这使 80% 的 shared 无法为 z 的 top-1 直接负责。

### 8.2 关于梯度

1. 每个新损失都应报告：作用张量、是否 detach、对主干/最终描述子的梯度占比、对新增参数的梯度占比。TAUR 的 2% 与 SFTC 的“仅 CLS 监督”是两类不同的“名义参与”。
2. “限制梯度到 Adapter”同时限制了保护与可改进空间；如果没有对最终描述子的梯度路径，新目标不可能直接改变检索排序。
3. 初始化安全（1%/0.01）只约束起点；必须监控训练中的残差比例、gamma、router 分布，并设硬上限。
4. 全局 clip 与多参数组 LR 会改变相对步长；新增参数组应报告裁剪频率与组梯度范数，不能只报告“总 clip=1”。

### 8.3 关于评估协议

1. 评估排除**全部同相机** Gallery；训练 WRT 不排除。任何“训练/检索一致”的声明都必须逐项核对正负样本构造，而不是只看特征归一化。
2. 全 Gallery AP 与“本卡 batch 代理排序”不是同一目标；代理损失要报告候选集大小、正样本数、是否包含同相机。
3. mAP/R1/R5/R10/mINP 必须一起报告。本组出现了三种不同模式：
   - HiHR/ViSA：mAP↑、同模态 mINP↑、同模态 R1↓；
   - SFTC：全指标↓，但同权重 ON>OFF；
   - TAUR：9 宫格 mAP↓、R5/R10↑、R1 混合。
   只看 mAP 或只看 R1 都会误判。
4. 同权重 ON/OFF 是“反事实”，不是“相对 baseline 的增量”。标准报告应同时给 baseline、ON、OFF，并做代价/收益分解。

### 8.4 phase2 必做的控制实验（针对本组暴露的问题）

1. 任何“拼接新描述子”的方法：T0（只有 CLS）、T1（CLS+辅助但权重 0）、full（实际权重）三组；或同权重 ON/OFF。
2. 任何“新增损失”：报告该损失对最终描述子/主干/新参数的梯度占比；若为 0 或 <5%，不要把指标变化归因于它。
3. 任何“改 sampler / 改原 loss”的方法：必须配套同 sampler + 原 loss 的 matched UAD control。
4. 任何“校准”：校准集规模、参考模型状态（trained vs fresh）、校准后梯度倍数都要记录；不要只用中位数匹配。
5. 所有实验：记录每 epoch 的 loss 分量、梯度范数（分组）、clip 频率、以及最终描述子/辅助描述子的单独检索结果。

---

## 9. 本次审计的结论摘要（每实验 2 条）

- **HiHR**：新增的 20% 拼接分量（parent/child）接收全部主损失，固定权重使 `cos(F)=0.8cos0+0.133cosp+0.067cosc`；9 宫格 mAP 全升、同模态 mINP 全升、同模态 R1 全降，支持“平均正样本改善、头部 margin 收缩”。同时，WRT 标定用已训练 UAD 对 fresh HiHR，正/负 ratio 相差 1.45×，单标量 9.78 并不中性，且 WRT 每样本梯度被放大约 46×。
- **ViSA**：最终 z 只被 fusion WRT 监督，且 shared 分量 detach，80% 的 CLS 从未为 z 的 top-1 直接优化；z 的 20% local 分量导致与 HiHR 同构的“9 宫格 mAP 全升 + 同模态 R1 降 + mINP 升”。此外 specific token 改写共享序列（129→130 key），ETGM/DLFM 新增约 136M 参数、2× LR。
- **SFTC**：局部来源与原始 K/V 共享一个 softmax（129+128），bias=-2 仅把初始 local 质量压到约 11.8%，并未构成恒等启动；ON/OFF=+0.92/+1.36，但 OFF 本身已比 baseline 低 -1.92/-4.48，模块只回收联合训练代价的约一半/三分之一，ON>OFF 不能证明方法优于 UAD。
- **ICRE**：CLEE 在 L9 之后对全部 Patch 做 128×128 低秩全局混合，gamma 从 0.01 涨到 0.436、残差比 25–28%，router 仍 ~50/50；模块确实参与且梯度非零，但它是一个强平滑算子而非小残差，9 宫格全降、RGB/NIR 同模态退化最大。
- **TAUR**：rank loss 的梯度只到 TIR adapter，且日志显示仅占 adapter 上 UAD 梯度的 ~1.5–2.1%（cosine ~0.3），对主干为 0；同时 `CAMERA_PRIORITY` 采样器与 `WRT_STABLE` 改动了原训练，且没有 matched control。因此“新增损失非零但指标分化”更可能是小梯度 + 采样器/原损失改动，而不是 proxy 可解释的全部差异。

---

### 附：本次读取的关键证据行号索引

- Baseline：`WDV1_CC\WHU\model\make_model.py:67-80`；`model\backbones\vit_pytorch.py:151-154, 317-341`；`processor\processor.py:176-231`；`loss\wrt_loss.py:39-81`；`utils\metrics.py:136-203, 327-488`；`datasets\sampler.py:69-143`。
- HiHR：`WHU_Base\WHU_HiHR\WHU\model\visual_hihr.py:20-59, 62-108`；`loss\visual_hihr.py:12-83`；`processor\processor.py:175-176, 232-284`；`model\backbones\vit_pytorch.py:317-345`；`solver\make_optimizer.py:23-25`；`configs\HiHR_stage1.yml`、`HiHR_H2_stage2.yml`。
- ViSA：`WHU_Base\WHU_ViSA+MD+MP\WHU\model\visa_adapt.py:71-154`；`model\make_model.py:86-127`；`model\backbones\vit_pytorch.py:357-381`；`processor\processor.py:202-255`；`solver\make_optimizer.py:20-27`；`solver\scheduler_factory.py:18-22`。
- SFTC：`WHU_Base\WHU_Fusion+MSCMNet+SDS+CoMer_SFTC\WHU\model\backbones\sftc_local.py:67-173`；`model\backbones\vit_pytorch.py:122-142, 331-355`；`model\make_model.py:71-108`；`processor\processor.py:181-249`；`solver\make_optimizer.py:14-19`；诊断 CSV 见 `D:\CODE\Doing\WHU_base\SFTC\logs\SFTC_120_diagnostics\`。
- ICRE：`...\_audit_evidence\WHU_base\WHU_ICRE\model\backbones\clee.py:7-68`；`model\backbones\vit_pytorch.py:297-305, 346-358`；`model\make_model.py:67-72`；`processor\clee_runtime.py:42-109`；`processor\processor.py:122, 189-191, 236-241`；`solver\make_optimizer.py:22-25`。
- TAUR：`TAUR\WHU\model\backbones\thermal_adapter.py:21-34`；`model\backbones\vit_pytorch.py:299-310, 329-340, 354-357`；`loss\taur_rank_loss.py:16-66`；`processor\taur_grad.py:7-100`；`processor\processor.py:194-293`；`datasets\sampler_camera.py`；`datasets\make_dataloader.py:26-34, 100-127`；`solver\make_optimizer.py:20-27`；`configs\TAUR.yml`、`UAD_TAUR_control.yml`。
- 日志/诊断：见 0.1 的目录列表；核心数值见正文 1.8、2.8、3.7、4.7、5.7。
