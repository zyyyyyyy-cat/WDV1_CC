# Group 2 机制审计：KGL / PAT / WQVD / SCSR / RLE / PEFT / ReID5o / M-SpecGene

审计日期：2026-10-29  
审计方式：静态代码核对 + 本机已有诊断文件/日志复核；未运行训练、未修改代码。  
证据分级：

- **[代码事实]**：直接由当前代码的调用链、张量形状、梯度路径或配置得到。
- **[实验支持]**：来自本机日志/诊断 CSV/JSON，或 02 档案所引服务器日志中的已报告数值。会标注“本机”或“02/服务器”。
- **[机制假设]**：由代码事实和实验结果推断，不能当作已证因果。

---

## 0. 路径表与基线机制

### 0.1 路径缩写

| 缩写 | 绝对路径 |
|---|---|
| `BASE` | `D:/CODE/Doing/WHU_base/WDV1_CC/WHU` |
| `KGL` | `D:/CODE/Doing/WHU_base/WK/WHU` |
| `PAT_S` | `D:/CODE/Doing/WHU_base/WP/WHU` |
| `PAT_D` | `D:/CODE/Doing/WHU_base/WP/WHU_D` |
| `WQVD` | `D:/CODE/Doing/WHU_base/WQVD/WHU` |
| `SCSR3` | `D:/CODE/Doing/WHU_base/SCSR/WHUv3` |
| `SCSR4` | `D:/CODE/Doing/WHU_base/SCSR/WHUv4` |
| `RLE` | `D:/CODE/Doing/WHU_base/WR/WHU` |
| `PEFT` | `D:/CODE/Doing/WHU_base/WB/WHU` |
| `REID` | `D:/CODE/Doing/WHU_base/WO/WHU` |
| `MSPEC` | `D:/CODE/Doing/WHU_base/WM/WHU` |
| `MSPEC_SRC` | `D:/CODE/Doing/WHU_base/WM/M-SpecGene-main` |
| `ARCH02` | `D:/CODE/Doing/WHU_Code/WHU_Research_Notes/02_Previous_Experiments.md` |

### 0.2 基线正向与反向

**[代码事实] 输入与主干**

- `BASE/datasets/make_dataloader.py:17-23`：训练 collate 把每个模态堆成 `[B,3,256,128]`，返回 `imgs=[M 个张量]`、`pids=[B]`、`camids=[M 个 [B]]`。
- `BASE/model/make_model.py:67-73`：`x=torch.cat(list(x),dim=0)` → `[M*B,3,256,128]`；`base(x)` 返回 CLS `[M*B,768]`；`bottleneck` 为 BN1d；`classifier` 为无 bias 线性层。
- `BASE/model/backbones/vit_pytorch.py:317-341`：patch embed 后 token 数 16×8=128；`cls_token` 拼在最前 → `[B,129,768]`；12 个 Block；最后 `norm`；返回 `x[:,0]`，即 `[B,768]`。
- 推理 `BASE/model/make_model.py:74-80`：`NECK_FEAT='before'` 时返回 **pre-BN CLS**；测试阶段 `BASE/utils/metrics.py:462-465` 再做 L2 归一化。最终检索描述子不是 BN 后的 `feat`。

**[代码事实] 损失与顺序**

- `BASE/loss/make_loss.py:30-37`：`ID_LOSS=CE(score,target)`；`TRI_LOSS=WRT(feat,target)`；总损失 = `ID_LOSS_WEIGHT*ID + TRIPLET_LOSS_WEIGHT*TRI`。`score` 来自 `BN(global_feat)`，`feat` 是 pre-BN `global_feat`。
- `BASE/processor/processor.py:195-220`：先算 CE+WRT；再算 ProCA（`PCA`，0.01）与 GPD（无权重，系数 1）；`loss += ...`；随后 `scaler.scale(loss).backward()`（219行）。
- `BASE/processor/processor.py:225-227`：反向与 optimizer step 之后，在 `no_grad` 内更新 GPD 原型 bank（EMA 0.2，`GPD` tau=0.03）。
- `BASE/processor/processor.py:298-302`：验证逐模态 loader，`eval_model(img, mode=mode)`，只取返回张量。

**[代码事实] 评测**

- `BASE/utils/metrics.py:99` 与 `:164`：每个 query 排除所有同相机 gallery（`g_camids == q_camid`），这是 WHU/SYSU 协议，不是 Market1501 的“同 ID 同相机”。
- `BASE/utils/metrics.py:491-565`：九组模态 pair；`:569-613`：A→G / G→A；`:615-625`：详细 domain 协议。
- 所有第二组实验均为 `NECK_FEAT='before'`、`FEAT_NORM='yes'`，最终比较的是 **pre-BN CLS 的 L2 归一化欧氏/内积排序**。

---

## 1. KGL：浅层模态 Adapter + HNSM

### 1.1 调用链与插入位置

- Adapter 类：`KGL/model/backbones/vit_pytorch.py:241-257`，`LN→Linear(768→64)→GELU→Linear(64→768)`，`up` 权重和 bias 零初始化（253-254）。
- 模块注册：`KGL/model/backbones/vit_pytorch.py:315-338`。`modality_adapters` 形状为 `[adapter_depth][num_modalities]`；每个 adapter 独立。`self.apply(_init_weights)` 之后再次把每个 `up` 清零（335-338），确保不是被通用初始化破坏的“伪零初始化”。
- 插入点：`KGL/model/backbones/vit_pytorch.py:380-389`，在 **每个 block 完整输出之后**、`block_index < adapter_depth`（默认 3，即 L1-L3）时执行：
  - `adapter_delta = Σ_m mask(modal_id==m) * adapter_m(x)`
  - `x = x + adapter_delta`
  - `x` 是 `[B,129,768]`，包含 CLS 和全部 patch；**Adapter 同时作用于 CLS 与 Patch**，不是 patch-only。
- 训练模态 id：`KGL/model/make_model.py:192`，`torch.arange(M).repeat_interleave(B)`；输入按 modality-major `cat`，两者一致。
- 推理模态 id：`KGL/model/make_model.py:212-216`，`mode-1` 填满整个 batch；验证 loader 顺序与 `DATASETS.MODALITIES` 一致，因此路由正确。
- HNSM 关系头：`KGL/model/make_model.py:29-43`，`LayerNorm→Linear(768→256)→GELU→Linear(256→1)`，输入是 `abs(features_a-features_b)`。
- HNSM 挖掘：`KGL/model/make_model.py:107-184`。`global_feat` reshape 成 `[M,B,768]` 后 L2 normalize 作为 **detached 挖掘描述子**；最终 block 的 patch tokens 做 mean pool（103-105），reshape 成 `[M,B,768]`；正样本按 PID 采样，负样本 hard/random；支持 NIR-NIR、TIR-TIR、NIR-RGB、TIR-RGB。
- HNSM 训练激活：`KGL/processor/processor.py:160-170`，`epoch > START_EPOCH(10)` 后 ramp；`hnsm_scale = min(1,(epoch-10)/10)`；`hnsm_active` 在 `processor.py:186` 传入模型。

### 1.2 正向数据流与最终描述子

**[代码事实]**

- Adapter 路径：`x` 经过 L1-L3 后叠加模态残差；L4-L12 仍共享。最终 `norm(x)[:,0]` 进入 `bottleneck` 与 `classifier`；检索返回 pre-BN CLS。
- 因此 Adapter **确实进入最终 CLS**：L1-L3 的 CLS 与 patch 都被改写，后续 9 个 block 会继续传播。
- 没有“最后一层空路径”：adapter 在第 3 个 block 后结束，后面还有 9 个 block。
- HNSM 的正向：训练时 `base(..., return_tokens=hnsm_active)` 返回 `(CLS, patch_tokens)`（`vit_pytorch.py:393-395`）；`patch_tokens` 是最终 `norm` 之后的 `x[:,1:]`。关系头输出只用于训练，**推理不返回、不拼接**。
- HNSM 的最终检索描述子仍是 CLS，不含关系头输出。

### 1.3 反向梯度流与 optimizer

**[代码事实]**

- Adapter 参数全部 `requires_grad=True`；`KGL/solver/make_optimizer.py:5-23` 没有对 adapter 做特殊 LR/WD：它们和主干一样用 `BASE_LR=0.008`、`WD=1e-4`；名字含 `bias` 的 adapter bias 走 `BIAS_LR_FACTOR=2`（因为该文件未排除 adapter）。
- Adapter `up` 零初始化导致 **第一步对 `down` 的梯度为零**（`∂L/∂down` 依赖 `up`），只有 `up` 先获得梯度；之后 `down` 才被激活。不是永久死路，但启动慢。
- HNSM 的挖掘相似度在 `torch.no_grad()` 内（`make_model.py:136-161`），`positive_index/negative_index` 不可微；但 `pooled_tokens` 本身 **没有 detach**，`relation_head` 的 BCE 梯度会回传到 anchor patch、正/负 patch，再经最终 block 与共享主干回传。
- 关系头私有参数会吸收一部分关系任务；CLS 只通过共享主干间接受到影响。`GlobalRelationHead` 不参与推理，因此不存在“关系头直接改善检索头”的路径。
- HNSM 的 BCE 在 `KGL/processor/processor.py:222-227`：`loss += HNSM.WEIGHT * hnsm_scale * BCEWithLogits`，权重 0.5，10 轮后线性升到 0.5。
- HNSM 不与 Adapter 同时开启：`configs/UAD_adapter.yml` 中 `HNSM.ENABLED=False`；`configs/UAD_hnsm.yml` 中 `ADAPTER.ENABLED=False`。

### 1.4 损失与归一化顺序

- 主损失完全沿用 UAD：CE(BN(CLS)) + WRT(pre-BN CLS) + 0.01 ProCA + GPD；Adapter/HNSM 不改变这四项。
- HNSM 关系特征是 **未归一化** 的 patch mean；关系头内部先 `LayerNorm` 再分类。挖掘描述子是归一化 CLS，二者尺度不同，但关系头有 LN。
- 检索归一化仍由 evaluator 在最后做 L2。

### 1.5 训练-测试一致性

- Adapter：训练用 `modal_ids = arange(M).repeat_interleave(B)`；推理用 `mode-1`。两者与 modality-major 输入顺序一致，路径相同。**[代码事实]**
- HNSM：训练时 `return_tokens=True` 只影响返回值，不改变 transformer 计算；推理时 `return_tokens=False`，CLS 路径相同。关系头不参与推理。**[代码事实]**
- 没有 train/test 路由错位。

### 1.6 为什么幅度小、TIR 方向仍退化

- **[代码事实]** 只插 3/12 层，bottleneck 64，残差零初始化；每模态独立残差约 0.88M 参数（3 blocks × 3 modalities × 约 98k），相对 86M 主干很小。
- **[代码事实]** Adapter 作用于所有 token（含 CLS），且每个模态有独立残差；这是显式的 **早期模态分离**，没有共享/对齐项。唯一跨模态约束来自最终 WRT 与共享 batch。
- **[实验支持/02]** H026 总体 mAP/R1/R5/R10 = +0.15/+0.21/+0.42/+0.03pp，四项同向但幅度小；TIR→RGB R1 7.12→6.60，TIR→TIR mINP 2.10→2.07。说明总体小升不以弱模态改善为代价以外的方式出现——目标 TIR 方向仍退化。
- **[机制假设]** 早期模态特定残差会把 TIR 的低层纹理/背景差异放大；后 9 层共享 block 只能部分吸收。TIR 本身身份线索弱，专门化残差可能更偏向模态风格而非身份，导致 TIR→RGB 与 TIR→TIR mINP 下降。
- **[机制假设]** 四项总体小升也可能包含单 seed 波动；没有重复 seed、没有总体 mINP 正式日志，不足以证明稳定机制。

### 1.7 可复用教训

- 浅层 adapter 的输出尺度小、位置浅，天然只能做“小幅条件化”；若目标是跨模态对齐，应把跨模态约束放在 adapter 附近或给 adapter 一个对齐/正交约束，而不是只靠最终 WRT。
- 零初始化 `up` 会带来“先学 up、后学 down”的两阶段启动；如果容量小，120 轮内可能只学到很小偏移。
- 评估“弱模态改善”必须同时看 TIR→RGB 与 TIR→TIR，不能只看总体四项。

### 1.8 与 02 的差异

- 02 的 2.9.4 说 Adapter “按真实模态路由到 64 维瓶颈残差”，但没有说明 **它同时作用于 CLS 和 Patch**；这是机制解释中应补的关键事实。
- 02 没有记录 adapter 的 LR/WD 没有特例（base LR 0.008、WD 1e-4，bias 走 2×LR）；这影响“为什么幅度小”的解释。
- 02 对 HNSM 的描述基本准确；但可补充：关系头输入的 patch mean 未归一化，关系头内部 LN；关系头私有参数会吸收任务，CLS 只经共享主干间接获益。
- 02 说“HNSM mAP+0.18、R1−0.05”与本机无法独立复核（本机无 KGL 训练日志），只能作为 02/服务器证据。

---

## 2. PAT：Part Token、CSL、PSD、Part-ID/WRT/ProCA、PGPD

### 2.1 调用链与插入位置

- Part token：`PAT_S/model/backbones/vit_pytorch.py:270-277`，3 个 `part_tokens` 与 `part_pos_embed`；在 `forward_features` 中拼成 `[CLS, part0, part1, part2, patches]`（399-406）。
- Part 初始化：`PAT_S/model/backbones/vit_pytorch.py:475-480`，`load_param` 把预训练 `cls_token` 复制到三个 part token，把 `pos_embed[:, :1]` 复制到 `part_pos_embed`。
- Part 注意力 mask：`PAT_S/model/backbones/vit_pytorch.py:339-384`：
  - global 流：CLS + 全部 patch 全连接；
  - 每个 part 只连接自己固定的 8×8 重叠区域 patch（起点 0、4、8，patch 网格 16×8）；
  - `region_indices` 包含 part token 与其区域 patch，mask 是 **双向** 的：part 可以读 patch，patch 也可以读 part；
  - `cls_part_connect=True` 时额外打开 CLS↔part 直连（360-363）。
- E1b 的随机 ROI：`PAT_S/model/backbones/vit_pytorch.py:422-425`，当 `part_aware and random_part_roi` 时每次 forward 调 `_make_part_attention_mask(random_roi=True)`。**这里没有 `self.training` 判断**，因此 E1b 的验证/测试和 CSL memory 初始化也使用随机 ROI。这是明确的训练-测试一致性问题。
- 推理：`PAT_S/model/make_model.py:107-114`，`part_aware` 时只取 `backbone_output[0]`，即 global CLS；part token 不拼进最终描述子。
- Part 头：`PAT_S/model/make_model.py:74-86`，3 个独立 `part_bottlenecks` + `part_classifiers`，只用于 E2/E3 的 Part-ID。
- CSL memory：`PAT_S/processor/processor.py:100-189`；初始化 `PAT_S/processor/processor.py:298-328`，用全部训练图像、`model.eval()`、`extract_part_features` 建 3 个 sample-level memory。
- PSD：`PAT_S/processor/processor.py:192-206`，用 CSL 的 top-k 邻居 PID 构造软标签，监督 global classifier logits。
- Part losses：`PAT_S/processor/processor.py:404-430`；PGPD：`PAT_S/processor/processor.py:432-450`。

### 2.2 正向数据流与最终描述子

**[代码事实]**

- Part token 从输入贯穿 12 个 block；每个 block 的 patch token 可以读同区域 part token（因为 mask 双向），因此 part 信息可以沿 “part→region patch→下一层 CLS” 路径影响 CLS。
- 最后一层：block 12 的 CLS 输出来自 block 11 的 token；block 12 的 part token 只能影响 block 12 的 patch，不能在同层影响已经算出的 CLS。因此 **block 12 的 part token 对最终 CLS 没有同层直接路径**；但 block 1-11 的 part token 可通过下一层 patch 间接影响 CLS。
- E1a（`CLS_PART_CONNECT=False`, `RANDOM_PART_ROI=False`）：固定 ROI，part→patch→后续 CLS 路径稳定存在。
- E1b（`CLS_PART_CONNECT=True`, `RANDOM_PART_ROI=True`）：额外 CLS↔part 直连；但随机 ROI 在 eval 仍生效，导致同一测试图像每次 forward 的 part 区域不同。
- 最终检索：global CLS，`NECK_FEAT='before'`，pre-BN L2；part token 与 part classifier **不进入推理描述子**。

### 2.3 反向梯度流与 optimizer

**[代码事实]**

- CSL memory 更新在 `@torch.no_grad()` 的 `PartMemory.update` 中，`current_features = part_features.detach()`（`processor.py:130`），memory 不回传梯度。
- 但 CSL forward 的 `query = part_features[:,part]` 带梯度；`logits = similarity / temperature`（160-166）；`numerator/denominator` 对 query 可微，因此 CSL 梯度进入 part token 与共享主干。
- CSL 的 `top_indices` 与 `neighbor_labels` 是 `detach()` 的（184-188），不直接反传。
- PSD 的 `soft_targets` 来自邻居标签，只有 `cls_score` 参与 `log_probs`，梯度进入 global classifier/BN/主干。
- E1b 的随机 ROI 由 Python `random.sample` 生成，索引不可微；selected patch values 仍可微。
- Optimizer：`PAT_S/solver/make_optimizer.py` 无任何 part/CSL 参数特例；全部 base LR 0.008、WD 1e-4，bias 2×LR。Part BN bias 冻结（`make_model.py:82-84`）。
- DDP：`PAT_D/processor/processor.py:125-162`，每个 rank 对当前 batch 的 `part_features/indices/labels` 做 `all_gather`，然后在本地 memory 上执行同一次 update；memory 初始化每个 rank 都跑全训练集（非分布式 loader），因此各 rank memory 一致。
- DDP 的 CSL 仍以本 rank query 计算 top-k/损失；梯度经 DDP 平均。

### 2.4 损失、归一化顺序与 02 关系

- E1a/E1b：`L = L_WRT + 0.01 L_ProCA + L_GPD + L_CSL + L_PSD`。当 `PSD_REPLACE_ID_CE=True` 时，`global_base_objective` 返回 **仅 WRT**（`processor.py:221-223`），普通 global CE 被移除；PSD 内部 = `0.5*soft CE + 0.5*hard CE`（`processor.py:192-206`）。所以不是“原 CE + PSD”，而是“WRT + PSD（含 hard CE）+ CSL + ProCA + GPD”。
- E2：Part-ID 0.5 + Part-WRT 0.5 + Part-ProCA 0.01 + 原 UAD；无 CSL/PSD。
- E3：E2 + CSL 0.5，`CSL_STRICT_CROSS_ID=True`；无 PSD。
- E4：CSL 0.5 + PGPD 替换 GPD；无 Part-ID/WRT/ProCA；`PGPD_GAMMA=1.0`。
- CSL 的 `similarity = query @ memory.t()` **没有 L2 归一化**（`processor.py:160-166`）；part feature 的范数约 10（本机 DDP supervision 诊断：part norm 中位数约 9.7-10.8），top-1 点积可达约 100，除以 `CSL_TAU=0.1` 后 logit 约 1000。**softmax 极度尖锐**。
- PSD 的邻居来自这个尖锐 top-k；本机 DDP 诊断显示：top-1 邻居同 ID 比例 96.41%，同模态 82.14%，同平台 96.58%，同相机 92.74%；错误 ID 质量 mean 2.44%，effective 1.22%。也就是说，PSD 主要在强化“同相机、同平台、同模态的简单近邻”，而不是跨模态/跨平台身份。

### 2.5 训练-测试一致性

- E1a：固定 ROI；训练/验证/测试同一路径；memory 只在训练更新；推理只用 CLS。**[代码事实]**
- E1b：`random_part_roi=True` 在 `forward_features` 中 **没有 training gate**（`vit_pytorch.py:422-425`）。因此：
  - 验证/测试每个 batch 使用新的随机子 ROI；
  - CSL memory 初始化时 `model.eval()` 但仍是随机 ROI；
  - 同一图像重复测试会得到不同特征。
  这是 E1b 结果不可复现、训练-测试分布不一致的确定实现问题。**[代码事实]**
- DDP E1a：`PAT_D` 的 CSL all_gather 正确；但 `configs/E1a_PAT_Paper.yml` 文件内 `DIST_TRAIN: False`、`BASE_LR: 0.008`、`IMS_PER_BATCH: 64`，实际四卡运行通过命令行覆盖为 `DIST_TRAIN True`、`IMS_PER_BATCH 256`、`BASE_LR 0.032`（`PAT_D/README.md`）。配置文件和实际运行不一致，复现时容易误用。**[代码事实]**

### 2.6 为什么单卡 E1a 小升、DDP 反而 R1 降

**[实验支持/本机 DDP 诊断]**

- `PAT_D/analysis/retrieval/retrieval_summary.json`：DDP UAD vs DDP PAT E1a 的 Rank-1 转移：
  - both correct 1421；
  - UAD-only correct 482；
  - PAT-only correct 423；
  - both wrong 4079；
  - 净变化 = 423−482 = −59 queries ≈ **−0.92pp**（与 02 的 R1−0.92 一致）。
  - 平均 margin change = −0.00249；说明 PAT 主要改变 **近邻/近 tie 的排序**，不是整体表征崩溃。
- `PAT_D/analysis/supervision/supervision_summary.json`：最终 checkpoint 的 CSL top-1 logit 平均约 992，`p1_csl_loss` 平均约 −5.2e-8（数值上 0）。这意味着 CSL 在训练后期 **已经饱和**：top-k 的 logsumexp 与全体的 logsumexp 几乎相等，CSL 梯度接近 0。
- 同文件：PSD 的 top-1 邻居同 ID 96.4%、同相机 92.7%；错误 ID effective mass 仅 1.22%。PSD 更像“同相机近邻的标签平滑 CE”，而不是跨模态身份蒸馏。
- 02 报告单卡 E1a：mAP 10.45/R1 28.18，比单卡 UAD +0.31/+0.37；DDP E1a：11.04/28.79，比 DDP UAD +0.02/−0.92。单卡增益本身很小。

**[机制假设]**

1. **CSL 饱和 + 邻居偏差**：未归一化点积/温度 0.1 使 CSL logits 约 1000，softmax 近 one-hot，CSL 损失和梯度在后期消失；真正起作用的是 PSD，而 PSD 的邻居 92.7% 同相机、96.4% 同 ID，主要强化简单近邻。它不能提供跨模态/跨平台的身份对齐，因此对总体 R1 没有稳定增益。
2. **DDP 的优化与 memory 更新节奏不同**：DDP 全局 batch 256、LR 0.032，每步 CSL memory 用 4 个 rank 的 all_gather batch 更新（相当于每步 4× 数据），memory EMA 相对参数更新的节奏与单卡不同；同时 DDP baseline 本身比单卡高约 +1.9 R1，E1a 的微小单卡增益在更强的 DDP baseline 上没有复现。单卡 +0.37 不能外推到 DDP。
3. **近 tie 重排**：转移矩阵显示 PAT 使 482 个 UAD-correct 变错、423 个变对，净 −59；这更像“改变边界样本”而不是系统性提升。小增益/小退化都应绑定 seed、训练模式和 near-tie 统计，而不是归因于某个模块的普适有效性。

### 2.7 E2-E4 的机制

- E2：Part-ID/WRT/ProCA 直接监督局部 token；part token 通过 attention 间接影响 CLS。总体下降，说明局部监督与全局身份目标存在容量/优化竞争，或局部区域本身不稳定。
- E3：`CSL_STRICT_CROSS_ID=True`，CSL 显式把 **不同 ID** 的局部近邻从分母中保留并拉近（因为 numerator 是 top-k 相似度，loss 最小化会提高 top-k 相似度，同时压低全体）；这与 Part-ID/WRT/ProCA 的身份区分目标直接冲突。E3 最差（02：9.07/24.12）与这一代码结构一致。**[代码事实 + 机制假设]**
- E4：CSL 仍跨 ID，PGPD 用局部邻居的异 ID 频次提高对应原型负 logit；比 E3 好但仍低于 baseline。局部近邻既用于“拉近”又用于“推远”，目标张力明显。

### 2.8 可复用教训

- CSL 这类“未归一化点积 + 小温度”的对比损失，必须先检查 logit 尺度；否则损失会在早期饱和，模块名存实亡。
- 自蒸馏的邻居必须做相机/模态/平台分层统计；本机诊断已经显示 92.7% 同相机，不能用“局部近邻”直接外推跨模态判别。
- 随机 ROI 必须显式 gate `self.training`，且 memory 初始化要用确定性 mask；否则 E1b 的验证和 memory 都被随机化。
- 单卡最优不能直接迁移到 DDP；DDP 的 batch/LR/memory 更新节奏会改变近邻统计。

### 2.9 与 02 的差异

- 02 的 2.13.4 说 E1b “随机子 ROI”，但 **没有指出随机 ROI 在 eval/test 和 memory 初始化仍然生效**；这是 02 的明显遗漏，也是 E1b 结果解释必须补的训练-测试一致性问题。
- 02 把 CSL 当作活跃的局部监督；本机 DDP 诊断显示 CSL 在最终 checkpoint 已饱和到约 0，实际起作用的是 PSD，且 PSD 邻居严重同相机/同身份。02 的“PSD 可能传错身份外观近邻”方向正确，但没有给出“错误质量仅 1.22%、同相机 92.7%”的量化事实。
- 02 说 DDP E1a 通过 all_gather 维护 memory，正确；但没有指出 DDP 每次 update 覆盖 4× batch、以及全局 batch 256/LR 0.032 与单卡 64/0.008 的差异。
- 02 对 E3/E4 “跨 ID 局部近邻形成吸引”判断正确；可补充 CSL loss 的方向：最小化 `logsumexp(all)−logsumexp(topk)` 会提高 top-k 相似度，strict_cross_id 下就是显式拉近异 ID 近邻。

---

## 3. WQVD A1 / A2：局部残差与条件通道调制

### 3.1 调用链与插入位置

**A1（`FEATURE_AUG.TYPE='a1'`）**

- 分支类：`WQVD/model/modules/a1_local.py:8-55`。
- 取特征：`WQVD/model/backbones/vit_pytorch.py:319-354`，`return_tokens='l11'` 时：
  - `p11` 在 `i==10`（第 11 个 block）后捕获，**未经过最终 LN**；
  - `cls` 仍然走完 12 个 block + 最终 LN；
  - 返回 `(cls, p11)`。
- 融合：`WQVD/model/make_model.py:106-113`，A1 调用 `feature_aug(cls, patches)`，返回 `z, u`；`global_feat=z`。
- A1 内部：
  - patch `LN→Linear(768→192)`（17-18）；
  - query = 2 个可学 token + `query_proj(cls.detach())`（33-34）；
  - FP32 cosine top-16、温度 0.1、softmax 权重（36-40）；
  - 选中 patch 经共享 4-head self-attention，按权重求和（42-49）；
  - `u = output_norm(output_proj([h1;h2]))`，`output_norm` 无 affine（50）；
  - `z = cls + GAMMA(0.05) * u`（51）。
- 局部头：`WQVD/model/make_model.py:75-83`，`local_bottleneck=BN1d(768)` + `local_classifier=Linear(768,num_classes,bias=False)`；`local_feat=u`（135）。
- 局部 CE：`WQVD/processor/processor.py:232-235`，`local_ce = loss_fn.id_criterion(local_score, target_rep)`，权重 0.05。

**A2（`FEATURE_AUG.TYPE='a2'`）**

- 分支类：`WQVD/model/modules/a2_modulation.py:7-54`。
- 取特征：`WQVD/model/backbones/vit_pytorch.py:352-353`，`return_tokens='l12'` 返回最终 LN 后的 `(cls, patches)`。
- A2 内部：
  - patch `Linear(768→192)`（40）；
  - 4 个可学 query，self-attn + 残差（41-43）；
  - cross-attn 读 patch，FFN + 残差，得到 `cond`（44-46）；
  - `m0 = mask_token + cls_proj(cls.detach())`（47）；
  - cross-attn 读 `cond`，mask head 输出 768 维，sigmoid（48-50）；
  - `z = cls * (1 + GAMMA(0.2)*(2mask-1))`（51），scale 有界 [0.8,1.2]；
  - mask head 最后一层 `std=0.005`、bias=0（34-35），初始 mask≈0.5，scale≈1。
- A2 无局部 CE；只使用原 UAD 损失。

### 3.2 正向数据流与最终描述子

**[代码事实]**

- A1：`z=c+0.05u` 直接作为 `global_feat`；主 CE 用 `BN(z)`，WRT/ProCA/GPD 用 `z`；推理用 `z`。`u` 也经独立局部 BN/分类器得到 `local_score`，但只在训练使用。最终检索描述子是 **pre-BN z**。
- A2：`z=c*scale` 直接作为 `global_feat`；主 CE 用 `BN(z)`，WRT/ProCA/GPD 用 `z`；推理用 `z`。最终检索描述子是 **pre-BN z**。
- 两者都不是空路径；A1 的 u、A2 的 mask 都进入最终 z。
- A1 的 patch 来自 L11、CLS 来自 L12，存在 **深度不一致**；A2 的 patch 与 CLS 都来自 L12。

### 3.3 反向梯度流与 optimizer

**[代码事实]**

- A1：
  - `query_proj(cls.detach())` 阻断 CLS→query 条件的直接梯度（33）；
  - 但 patch values、top-k 权重、relation attention、`u`、`z` 都可微；主损失梯度直接回传 `cls` 和 `u`；
  - 局部 CE 只作用于 `u`，但 `u` 同时是 `z` 的一部分，因此局部 CE 也经 `u` 影响主干与融合路径；
  - `local_bottleneck`/`local_classifier` 只在训练使用，但其梯度会影响主干。
- A2：
  - `cls.detach()` 用于 `m0` 条件（47），阻断 CLS→mask 条件的直接梯度；
  - `z=cls*scale` 对 `cls` 的梯度是 `scale`，对 mask/cond/patch 的梯度经乘法路径回传；
  - mask head 的最终层非零初始化，保证上游初始有梯度。
- Optimizer：`WQVD/solver/make_optimizer.py` 无分支特例；A1/A2 分支参数、局部 BN、局部分类器全部 `BASE_LR=0.008`、`WD=1e-4`，bias 2×LR；局部 BN bias 冻结（`make_model.py:79-81`）。没有独立 LR/WD。

### 3.4 损失与归一化顺序

- A1：`L = CE(BN(z)) + WRT(z) + 0.01 ProCA(z) + GPD(z) + 0.05 CE(BN_local(u))`。主 CE 的 `id_criterion` 被暴露到 `loss_fn` 上（`WQVD/loss/make_loss.py:37`）。
- A2：`L = CE(BN(z)) + WRT(z) + 0.01 ProCA(z) + GPD(z)`。
- 所有损失在同一个 autocast 块内求和后一次性 backward；没有单独 detach 的 z 副本；没有额外的特征归一化，最终由 evaluator 做 L2。

### 3.5 训练-测试一致性

- A1/A2 训练与推理都走 `_extract_feature`；A1 推理也计算局部 `u`（虽然 local head 不返回），A2 推理计算 mask；路径一致。**[代码事实]**
- 无模态 id 路由；三个模态共享同一分支。
- `diagnostic_features` 仅 eval 用，不改变训练。

### 3.6 为什么 A1 同权重融合 ON>OFF，但整体仍弱于独立 UAD

**[实验支持/本机 `WQVD/logs/A1_vs_UAD_gamma_analysis/feature_comparison.csv`]**

总体 AS-ReID：

| 特征 | mAP | R1 | mINP |
|---|---:|---:|---:|
| UAD-CLS（独立训练） | 10.1395 | 27.8064 | 0.6033 |
| A1-CLS（同 checkpoint，gamma=0） | 10.0352 | 26.3076 | 0.7128 |
| A1-Fusion（gamma=0.05） | 10.2882 | 27.2287 | 0.6902 |
| A1-Local（u 单用） | 5.5141 | 17.2678 | 0.2666 |

- A1 训练把 CLS 的 R1 从 27.81 降到 26.31（−1.50pp），mAP 降 0.10pp，但 mINP 升 0.11；
- 同权重把 `0.05u` 加回去（A1-Fusion）相对 A1-CLS：mAP +0.253pp、R1 +0.921pp、mINP −0.023pp；
- 但 A1-Fusion 仍比独立 UAD：mAP +0.149pp、R1 −0.578pp、mINP +0.087pp；
- `u` 单用只有 5.51 mAP/17.27 R1，远弱于 CLS。

**[代码事实 + 机制假设]**

1. **主损失只监督 z，不监督 c**：`z=c+0.05u`，CE/WRT/ProCA/GPD 都可以通过 u 满足目标；没有任何项要求 c 单独保持判别性。局部 CE 又显式把 u 训练成身份可分。因此 c 和 u **共同适应**，c 单独检索能力下降（A1-CLS −1.50 R1），u 成为补偿项。
2. **融合能恢复一部分，但不能恢复全部**：u 携带局部身份信息（虽然单用很弱），加回后 R1 +0.92；但 u 来自 L11 patch、与 L12 CLS 深度不一致，且局部线索可能偏同相机/同模态外观，无法完全替代独立 UAD 的 CLS。净效果仍 −0.58 R1。
3. **mINP 与 R1 的分离**：A1-CLS 的 mINP 反而最高（0.7128），说明训练后的 CLS 正样本更紧凑，但 top-1 边界更差；融合降低 mINP 但提高 R1。这再次说明“聚类更紧”不等于“首位更准”。
4. **gamma 不是简单调大就好**：本机诊断 gamma=0.1 时 R1 25.95、mAP 9.73，低于 gamma=0.05；说明 u 与 c 的尺度/方向并不天然对齐，不能靠加大残差解决。

### 3.7 A2 为什么接近基线但不提升

- **[代码事实]** A2 只对现有 CLS 做有界通道乘性重标定（[0.8,1.2]），不引入新的 patch/局部内容；mask 由 L12 patch + detached CLS 条件生成。
- **[代码事实]** mask 初始≈0.5、scale≈1；mask head 1.26M 参数，全部 base LR/WD；没有局部 CE 或额外正则。
- **[机制假设]** 乘性门控只能重新加权已有通道，不能创造新的可分信息；如果 CLS 中某些通道稳定地夹带相机/模态噪声，门控可能有用，但优化目标只有身份 CE/WRT/ProCA/GPD，容易过拟合训练身份/相机。A2 的 mAP 持平、R1 −0.71、R5/R10 小升，符合“轻微重排、无新增信息”的解释。
- **[机制假设]** mask 条件使用 `cls.detach()`，mask 不直接接收“当前 CLS 应该如何改变”的梯度；它只能从最终损失间接学习，条件路径的优化信号较弱。

### 3.8 可复用教训

- 残差/门控分支必须显式保护主描述子：例如对 c 加一个 identity-preserving loss，或在训练时随机 dropout 分支，避免 c 与 u 共同退化。
- 评估残差方法要同时报告 ON/OFF（同权重）和独立训练 baseline；A1 的“融合 ON>OFF”只说明 u 有即时信息，不说明整体训练优于 UAD。
- 条件使用 `detach` 时，要区分“阻断条件梯度”和“阻断主路径梯度”；A1/A2 都只阻断了条件读取，主路径仍可回传。

### 3.9 与 02 的差异

- 02 的 2.16.4 对 A1/A2 的结构描述基本准确；但 2.16.6/2.16.7 应补充本机表的 **A1-CLS R1 −1.50pp** 与 **A1-Local 仅 17.27 R1**。02 说“A1 融合对该主干即时 +0.2531/+0.9212pp”，正确；但没有强调“c 被训练坏了 1.50pp，u 只补回 0.92pp”的定量关系。
- 02 说 A1 `CLS.detach` 只阻断 Query 条件，正确；可补充局部 CE 作用于 u，而 u 同时进入 z，所以局部 CE 不是完全“旁路”的。
- 02 说 A2 “mask 末层非零 std0.005”，正确；可补充初始 mask≈0.5、scale≈1，并非从恒等 mask 开始，而是从近似恒等 scale 开始。

---

## 4. SCSR v3 / v4.1：TIR 高频条件调制

### 4.1 调用链与插入位置

**v3**

- 模块：`SCSR3/model/scsr.py:59-138`。
- 结构：`LN(768)→Linear(768→192)`；patch reshape 成 16×8；Haar DWT 得到 LL/LH/HL/HH；条件网络 `Conv(4*192→64)→GELU→DepthwiseConv(64→64,3×3)→GELU→Conv(64→3*192)`；`gates=0.5*tanh(q)`；只改高频：`change=IDWT(0, gate_h*LH, gate_w*HL, gate_hh*HH)`；`delta=Linear(192→768)`；`smooth_bounded_residual` 上限约 0.2||P||；`updated=patches+residual`。
- 零初始化：`condition[-1]` 权重/bias 零初始化（103-104）；`up` std 0.02（98）。因此初始 `q=0`、`change=0`、`delta=0`、`residual=0`。
- 路由：`SCSR3/model/backbones/vit_pytorch.py:324-331`：
  - `indices = (modal_ids==target_id).nonzero()`；
  - `selected=tokens.index_select(0,indices)`；
  - `updated=cat([selected[:, :1], self.scsr(selected[:,1:])], dim=1)`，**保留该行 CLS**；
  - `tokens.index_copy(0,indices,updated)`，out-of-place，梯度正常。
- 插入点：`SCSR3/model/backbones/vit_pytorch.py:384-392`，`block_number == INSERT_AFTER_BLOCK(6)`，即 L6 后；后面还有 L7-L12 六个 block。
- 训练 modal id：`SCSR3/model/make_model.py:81-97`，按 modality-major 生成 1..M；推理 modal id：`SCSR3/processor/processor.py:338-339` 从 val loader 的 `modid` 传入（1 基）。
- `target_id = modalities.index('Thermal')+1 = 3`（`make_model.py:42-48`）。

**v4.1**

- 模块：`SCSR4/model/scsr_v41.py:44-133`。
- 插入点：`SCSR4/model/backbones/vit_pytorch.py:157-172` 的 `AttentionInputBlock.forward`：
  - `z=self.norm1(x)`；
  - `updated=modulator(z,modal_ids)`；
  - `attention=self.attn(updated)`；
  - `x=x+drop_path(attention)`；
  - `x=x+drop_path(mlp(norm2(x)))`。
  - 即：**Block 7 的 norm1 之后、Attention 之前**，只改 Attention 输入；原 residual 流 `x` 不变。
- 模块内部：`SCSR4/model/scsr_v41.py:83-133`：
  - `tokens` 形状断言 `[B,129,768]`；
  - `mask = modal_ids==target_id`；
  - `original=tokens[mask,1:]`，只取 patch，**CLS 不改**；
  - FP32 下 reshape `[B,768,16,8]`，Haar DWT；
  - `LL` 16 维、三个高频各 4 维、3 路能量 → 31 通道；`Conv(31→32,3×3)→GELU→Conv(32→12,1×1)` → `v[B,3,4,8,4]`；
  - `U[3,768,4]`；`t=einsum('jcr,bjrhw->bjchw',U,v)`；`m=0.2*tanh(t)`；
  - `updated_high=(1+m_j)*high_j`；IDWT；写回；
  - `result=tokens.clone(); result[mask,1:]=updated`，**RGB/NIR 与 CLS 逐元素不变**。
- 参数：40,032（`scsr_v41.py:81` 断言）；`U` std 0.5、`output` std 0.02（72-75），因此 **v4.1 不是零初始化恒等**，初始 `t` 非零、`m` 非零。
- Optimizer：`SCSR4/solver/make_optimizer.py:6-30`，SCSR 参数单独 group，`SCSR_WEIGHT_DECAY=0.0`、`SCSR_LR_MULTIPLIER=1.0`。

### 4.2 正向数据流与最终描述子

- v3：L6 后只改 TIR 的 patch；CLS 在该层保留；L7-L12 共享，TIR patch 的变化通过 attention 影响后续 CLS。最终描述子是 final LN 后的 CLS（pre-BN）。**不是空路径**，有 6 个后续 block。
- v4：Block 7 的 attention 输入被改；CLS 在 Block 7 的 attention 中会读到被改的 patch，因此同层就能影响 CLS；后面还有 5 个 block。最终描述子是 final LN 后的 CLS。**不是空路径**。
- 两版都只改 TIR 行；RGB/NIR 的 token 在路由/写回处逐元素不变（v3 `index_copy` 只写选中行；v4 `result=tokens.clone()` 只写 `mask,1:`）。

### 4.3 反向梯度流与 optimizer

- v3：
  - `condition[-1]` 零初始化使初始 `q=0`，但 `tanh'(0)=1`，`∂L/∂condition` 初始非零；`up` 初始梯度为零（change=0），随后才启动。
  - 主损失（CE/WRT/ProCA/GPD）经最终 CLS 回传到 L7-L12、再到 SCSR 的 patch 残差；CLS 在该层保留，但 CLS 仍通过后续 block 的 attention 接收 patch 残差梯度。
  - 所有 SCSR 参数在 optimizer 中，`SCSR3/solver/make_optimizer.py:19-27` 强制 `lr=BASE_LR`、`wd=WEIGHT_DECAY`（**WD=1e-4**，不是 0）；bias 也强制 base LR，避免 2×。
- v4：
  - 主损失经 Block 7 attention 的 Q/K/V 和后续 block 回传到 `U/output/conv`；CLS 在 Block 7 被 attention 影响，因此梯度路径比 v3 更直接。
  - SCSR 参数单独 group，`WD=0`、`LR=1.0×BASE_LR`；不参与 bias 2×LR 规则。
- 两版都没有新增损失；没有 detach 主残差（只对诊断统计 detach）。

### 4.4 损失与归一化顺序

- 无新增损失；原 UAD 四项不变。
- v3 的 `smooth_bounded_residual` 在 FP32 内计算，最后 cast 回 patch dtype（`scsr.py:128-129`）；`p_norm` 来自原 patch，`delta_norm` 来自 delta；径向梯度保留。
- v4 的 `m` 乘在高频系数上，等价于有界乘性残差；IDWT 后 cast 回原 dtype；没有额外 LayerNorm/scale。
- 最终检索仍是 pre-BN CLS 的 L2。

### 4.5 训练-测试一致性

- v3/v4 训练始终开启模块；`scsr_runtime_enabled` 只允许在 `model.eval()` 时关闭（`SCSR3 vit:355-356`、`SCSR4 vit:370-371`），训练中关闭会报错。
- 训练 modal id 由 modality 顺序生成；推理从 val loader 的 `modid` 传入；`target_id` 一致。**路由正确**。
- v3/v4 的 ON/OFF 同权重诊断是 eval-only，不影响训练路径。

### 4.6 为什么同权重 ON/OFF 几乎无差

**[实验支持/本机诊断]**

- v3：
  - `SCSR/logsv3/SCSR_1000_diagnostics/scsr_on_off_metrics.csv`：总体 ON vs OFF，R1/R5/R10 完全相同（0.29758/0.46183/0.54551），mAP 差约 3.9e-8（绝对，约 0.000004pp），mINP 差约 1e-9。
  - `scsr_internal_activation.csv`：TIR patch RMS 平均约 8.11；SCSR residual RMS 平均约 **2.59e-7**；`r_effective` 平均约 **3.33e-8**；actual change mean abs 约 1.93e-7；gate abs mean 约 2.23e-5。
  - 也就是说，v3 的残差相对 patch 约 **3e-8**，模块在数值上已经塌缩到近零。
- v4：
  - `SCSR/logsv4/mechanisms_fp32/v41_on_off_metrics.csv`：总体 ON/ON vs OFF/OFF：mAP 0.11023510 vs 0.11023621（−1.1e-6 绝对，约 −0.00011pp）；R1 0.2953942 vs 0.2952381（+0.0156pp）。
  - `v41_initial_vs_trained_gate.csv`：训练后 `m_rms` 平均约 0.0166，`m_max_abs` 平均约 0.119；三路高频 increment relative 平均约 1.46%-1.78%。即 v4 的调制约 1.7% RMS、最大约 12%，比 v3 大很多，但仍很小。
- **[代码事实]** v3 的零初始化 + `up` 初始零梯度 + WD 1e-4 共同形成“近零固定点”：`q` 虽可学，但主损失对残差的梯度很小（诊断 batch 1 的 `residual` 梯度 RMS 约 1.96e-7，activation RMS 约 7.6），WD 持续把参数拉向 0，120 轮后仍近零。
- **[代码事实]** v4 用非零初始化的 `U`/`output` 和 WD 0 避免了“零初始化陷阱”，但只改 TIR patch 的一个层、幅度 1.7% RMS；后面共享 block 会把变化摊薄，RGB/NIR 完全不变，因此总体指标几乎不动。
- **[实验支持]** ON/OFF 的差异只体现在极少数 near-tie query 上；v3 的 R1/R5/R10 完全相同，v4 的 R1 +0.0156pp 属于排序边界翻转，不是表征改善。
- **[机制假设]** v3 的“弱通路”不是“幅度小但仍有作用”，而是 **数值上近乎 dead**；v4 则是“小但非零”，两者不能用同一个“弱有效通路”解释。02 的 2.6.7 用“弱有效通路”描述 v3 偏轻。

### 4.7 可复用教训

- 零初始化输出 + 小梯度 + WD 很容易形成“模块存在但永远学不起来”的陷阱；应在训练日志里监控 `residual_norm/activation_norm`，并考虑对模块 WD=0 或给残差一个 warmup/尺度。
- 同权重 ON/OFF 是判断“推理时模块是否有即时贡献”的必要诊断；但即使 ON/OFF 有差，也不能直接推出“重训后一定更好”，因为主干已在该模块开启的轨迹上优化。
- 单层、单模态、小幅度的高频残差很容易被后续共享层吸收；若目标是 TIR 结构补偿，需要更强的条件/更长的作用链，或显式的跨模态对齐约束。

### 4.8 与 02 的差异

- 02 的 2.6.6/2.6.7 对 ON/OFF 数值的判断正确（v3 完全相同、v4 约 −0.000111pp/+0.0156pp）；但没有给出 v3 的 **residual/patch ≈ 3e-8、gate ≈ 2.2e-5**，因此“弱有效通路”不足以描述“数值上 dead”。
- 02 说 v4.1 “40,032 参数、独立 WD0、LR 倍率 1”，正确；但没有强调 v4.1 **不是零初始化恒等**（U std 0.5、output std 0.02，初始调制非零），而 v3 是零初始化。这是两版机制对比的关键。
- 02 说 v3 的 SCSR 参数采用主干 LR，但没有明确 v3 的 WD 仍是 1e-4（v4 才是 WD0）；这是“为什么 v3 塌缩、v4 不塌缩”的重要差异。
- 02 对 v3/v4 的 insertion 描述正确；可补充 v4 的 modulation 在 Block 7 attention 前，CLS 同层即可通过 attention 受影响，而 v3 要经过后续 block。

---

## 5. RLE E1-E4：训练期增强

### 5.1 调用链与插入位置

- 增强模块：`RLE/datasets/rle.py:14-120`。
- `moderate_rle`（14-19）：RGB 三通道按 `Beta(0.3,0.3)` 权重归一化后混合，复制成 3 通道；相当于把 RGB 变成灰度化/低饱和。
- `radical_rle`（22-78）：
  - 以 0.5 概率触发；
  - 循环最多 `max_attempts` 次；每次随机面积 `U(0.02,0.4)*H*W`、长宽比 `U(0.3,3.33)`；
  - 独立通道：每通道 `Beta(0.4,0.4)` 采样因子，除以该区域通道最大值后乘到区域；
  - 共享通道：一个因子乘到三通道；
  - `safe` 时把因子限制在 [0.5,2] 和 `1/max`；
  - 当 `mask.min()<0.1` 或达到 `max_effective_regions` 停止；
  - 零最大通道跳过。
- 模态路由：`ModalityRLE`（82-104）：
  - RGB：先 0.5 概率 `moderate_rle`，再 `radical_rle`（默认独立通道、0.5 概率）；
  - `IR`（NIR）：`radical_rle` 默认独立通道；
  - `Thermal`：
    - E1：`radical_rle` 默认独立通道；
    - E2：直接返回，不加新 RLE；
    - E3：`radical_rle(shared=True)`；
    - E4：`radical_rle(shared=True,safe=True,sh=0.2,max_attempts=20,max_effective_regions=2)`。
- 流水线：`RLE/datasets/make_dataloader.py:41-60`，`Resize→Flip→Pad→RandomCrop→ToTensor→RLE→Normalize→RandomErasing`；RLE 在 Normalize 前、原 RandomErasing 前。
- 图像读取：`RLE/datasets/bases.py:8-16`，所有图像 `convert('RGB')`；NIR/TIR 原始单通道被复制成三个相同通道。

### 5.2 正向数据流与最终描述子

- 无模型结构改动；只改变训练输入。最终检索仍是原 CLS。
- 训练时同一 identity 的三个模态各自独立做 RLE（RGB 还多做 MRLE）；标签不变，但跨模态外观对应关系被独立扰动。
- 推理/测试使用干净的 `val_transforms`（`make_dataloader.py:50-54`），无 RLE。

### 5.3 反向梯度流 / 损失 / optimizer

- 无新增损失、无新增参数； optimizer 不变。
- 增强只改变输入，梯度通过被增强的图像回传；没有 detach。
- 训练-测试只有“分布差异”，没有结构/路由不一致。

### 5.4 为什么四版都大幅退化，尤其 RGB 自检索

**[实验支持/02]**

- H042-H045 总体 mAP 8.71-8.87、R1 23.31-23.90，对单卡四项均明显下降。
- E1-E4 的 RGB→RGB mAP 下降 7.50-7.87pp、R1 下降 10.07-11.15pp；TIR→TIR mAP 仅下降 0.22-0.60pp。
- E2 完全关闭 TIR 新 RLE 仍显著下降，说明不能把全部问题归咎于 TIR 扰动。

**[代码事实 + 机制假设]**

1. **RGB 承受双重增强**：RGB 先 0.5 概率 MRLE（通道混合，破坏颜色身份），再 0.5 概率 RRLE（局部乘性大幅改变亮度/颜色）；RGB 自检索退化最大与此一致。
2. **NIR/TIR 是复制成 3 通道的灰度图**（`bases.py:16` `convert('RGB')`），独立通道 RRLE 会给三通道乘不同因子，把原本灰度一致的 NIR/TIR 变成 **不一致的伪彩色/纹理**；这不是真实模态变化，可能诱导模型学习通道差捷径，破坏身份。E3/E4 只对 TIR 用 shared 因子，NIR 仍是独立通道；E2 只关闭 TIR，NIR 和 RGB 仍被扰动。
3. **RRLE 的强度很大**：最多 100 个区域、每次面积 2%-40%、Beta(0.4,0.4) 常取近 0 或近 1，区域可被强烈压暗或提亮；叠加原 RandomErasing（p=0.5），航拍低分辨率人物的身份细节可能被擦除。E4 限制到 2 个区域/20% 面积仍失败，说明不只是“过度擦除”，还有 RGB 颜色破坏和 NIR 伪彩色。
4. **身份标签不变不等于身份可见性不变**：增强后的图像仍标同一 PID，但可能已丢失可辨身份；跨模态同一索引的三图被独立扰动，进一步破坏对应关系。
5. **[机制假设]** E3/E4 的 TIR 保护只保护了 TIR 的共享通道一致性，没有保护 RGB 的 MRLE、也没有保护 NIR 的独立通道；因此“保护 TIR”不足以恢复总体指标。

### 5.5 可复用教训

- 对 IR/TIR 做彩色图像增强前，必须先检查数据加载是否把灰度复制成 3 通道；独立通道增强会制造伪彩色。
- 增强强度要有身份保真检查（例如增强后的人体检测/身份可辨性、或与原图的结构相似度），不能只看覆盖度。
- 多模态同一索引的图像应使用一致/配对的空间变换（同一 crop/flip），RLE 的随机区域也应考虑跨模态一致性，否则会破坏跨模态对应。
- 负结果应保留，但下一步应做“只关原 RandomErasing”“只开 RGB MRLE”“只开 NIR”等单变量对照，而不是继续调 TIR 参数。

### 5.6 与 02 的差异

- 02 的 2.17 对流水线、E1-E4 参数、RGB 主损害的描述正确；但没有指出 **NIR/TIR 是 `convert('RGB')` 后的三通道复制图，独立通道 RRLE 会产生伪彩色**。这是解释“E2 仍退化、E3/E4 保护 TIR 无效”的关键遗漏。
- 02 说“RGB/NIR 强扰动、与原擦除叠加和身份细节损失都是候选解释”，方向正确；可补充 RGB 的 MRLE+RRLE 双重增强是 RGB 自检索退化最大的直接结构原因。
- 02 说 E4 同时改多个参数，正确；本机代码也确认 E4 同时改 shared/safe/sh/attempts/regions。

---

## 6. PEFT-BoA：DPA / TCA / 组合

### 6.1 调用链与插入位置

- DPA：`PEFT/model/backbones/vit_pytorch.py:109-126`，`Linear(768→96)→ReLU→Linear(96→768)`；权重 std 0.001，bias 0。
- TCA：`PEFT/model/backbones/vit_pytorch.py:128-146`，共享 `ClassAdapter`：`Linear(768→1536)→GELU→Dropout→Linear(1536→768)`；权重 std 0.001，bias 0。
- Block：`PEFT/model/backbones/vit_pytorch.py:177-241`：
  - `attn_input=norm1(x)`；TCA 时 `attn_input=[CLS+tuning_mhsa, patches]`（210-213）；
  - `x=x+drop_path(attn(attn_input))`（214）；
  - `ffn_input=norm2(x)`；TCA 时 `ffn_input=[CLS+tuning_ffn, patches]`（217-220）；
  - `adapter_cls=shared_tca(x[:, :1])`（TCA）；`adapter_patch=dpa(x[:,1:])`（DPA）（221-222）；
  - `adapter_output=cat([adapter_cls,adapter_patch])`（223）；
  - `ffn_output=mlp(ffn_input)`（224）；
  - `x=x+drop_path(ffn_output)+adapter_output`（240）。
- 注册：`PEFT/model/backbones/vit_pytorch.py:372-391`，12 个 block 都按 `use_dpa/use_tca` 实例化；TCA 共享 adapter 在 `self.tca`；`self.apply(_init_weights)` 后调用 `reset_adapter_parameters` 把 DPA/TCA 权重设 std 0.001、tuning 置零（388-391）。
- 配置：`configs/UAD_DPA.yml`、`UAD_TCA.yml`、`UAD_DPA_TCA.yml`；单卡 64/0.008，120 轮。

### 6.2 正向数据流与最终描述子

- DPA：每个 block 的 patch 输出叠加 DPA 残差；该残差会影响下一层 attention，进而影响下一层 CLS。
- TCA：每个 block 的 CLS 在 norm1/norm2 前叠加 per-layer tuning，且在 block 输出叠加共享 TCA 对 CLS 的残差；CLS 被直接修形。
- 最终：`PEFT/model/backbones/vit_pytorch.py:438-451`，12 个 block 后 `norm(x)`，返回 `x[:,0]`；检索 pre-BN CLS。
- **最后一层 DPA 空路径**：
  - block 12 中 `adapter_patch=dpa(x[:,1:])`，`adapter_output` 加回 patch；随后循环结束；
  - `x=self.norm(x)` 是逐 token 的 LayerNorm；`return x[:,0]` 只取 CLS；
  - block 12 的 CLS 已在 attention/FFN 中算好，不依赖 block 12 的 DPA patch 残差；
  - 没有 patch-level loss，因此 **block 12 的 DPA 参数梯度为零，且不影响检索**。这是确定的空路径。
  - TCA 在 block 12 仍影响 CLS（tuning_mhsa/ffn + shared TCA 加在 CLS 上），没有空路径。
  - DPA+TCA 组合中，block 12 的 DPA 支路仍为空；TCA 支路有效。

### 6.3 反向梯度流 / optimizer

- DPA/TCA 参数全部 `requires_grad=True`；`PEFT/solver/make_optimizer.py` 无特例，base LR 0.008、WD 1e-4，bias 2×LR。
- DPA block 1-11：梯度经后续 attention 回传；block 12：零梯度（空路径）。
- TCA：共享 adapter 在每个 block 都收到梯度（12 层梯度相加），per-layer tuning 各自收到梯度；最后一层也有效。
- 诊断残差统计是 `.detach()`，不参与损失（228-238）。

### 6.4 损失与归一化

- 无新增损失；原 CE/WRT/ProCA/GPD 不变。
- Adapter 残差直接加在 block 输出，不经过 drop_path；原 FFN 仍走 drop_path。
- 最终 LN 后再取 CLS；检索 L2 在 evaluator。

### 6.5 训练-测试一致性

- DPA/TCA 在训练/推理走同一 forward；无 modality 路由。
- block 12 DPA 在训练/推理都不影响 CLS；区别只是训练时它参与计算但不产生梯度。
- TCA 训练/推理一致。

### 6.6 为什么 DPA 近中性、TCA 明显退化

- **[代码事实]** DPA 只改 patch，且最后一层空路径；前 11 层的 patch 残差要经过后续 attention 才能影响 CLS，作用被逐层稀释。
- **[代码事实]** TCA 直接改 CLS，且在 12 层都改；共享 adapter 的同一组权重收到所有层的梯度，等价于一个跨层共享的 CLS 低秩偏置；per-layer tuning 再给每层 CLS 加偏移。它直接改变最终 CLS 的统计和类间几何。
- **[实验支持/02]** DPA：mAP −0.09/R1 −0.32，R5/R10 +0.07/+0.17；TCA：−0.39/−2.11；组合：−0.26/−1.53。DPA 近中性，TCA 明显退化，组合介于两者之间。TCA 的 RGB→RGB R1 下降 3.56pp，说明直接 CLS 修形对强模态伤害大。
- **[机制假设]** 在 **全参数微调** 条件下，主干本身已有足够自由度；额外 DPA/TCA 分支是冗余的。TCA 直接扰动 CLS 的预训练几何，破坏 BN/分类器已适应的方向；DPA 只间接影响 CLS，因此伤害小但也难有增量。
- **[机制假设]** 最后一层 DPA 空路径不是 DPA 退化的原因（它只是无效参数）；DPA 的轻微退化来自前 11 层 patch 残差改变了共享注意力的输入分布，而全参数主干不需要这个额外分支。TCA 的退化主要来自直接 CLS 修形，不能归因于空路径。

### 6.7 可复用教训

- 任何“每层都加 adapter”的设计，必须逐层追踪输出是否有终端消费者；最后一层 patch-only 残差在 CLS-only 检索下是死支路。
- 全参数微调时，额外 adapter 的收益假设不成立；它更像“扰动项”而非“容量扩展”。若要验证 adapter，应至少在冻结主干或部分冻结的设置下做对照。
- 直接改 CLS 的分支（TCA）风险显著高于 patch 分支；需要更强的 identity-preserving 约束或更小的初始化/残差尺度。

### 6.8 与 02 的差异

- 02 的 2.14.7 已正确指出最后一层 DPA 空路径；本审计补充：该支路不仅“不影响推理”，而且在训练中 **梯度为零**（没有 patch loss），是彻底死支路。
- 02 说 TCA “直接改 CLS 的统计与共享类间几何”，正确；可补充 TCA 共享 adapter 在 12 层共享权重、梯度跨层相加，per-layer tuning 进一步直接加在 CLS 上。
- 02 说“组合强于 TCA 但不及 DPA 和 Baseline”，正确；本审计强调这不是“互补成功”，而是 DPA 的近中性稀释了 TCA 的伤害。

---

## 7. ReID5o：末层模态 FFN 低秩专家

### 7.1 调用链与插入位置

- 专家线性：`REID/model/backbones/modality_experts.py:13-50`。继承 `nn.Linear`，复用原 `weight/bias`（23-24）；每个模态一组 `lora_A[rank,in]`、`lora_B[out,rank]`（28-35）；`scaling=alpha/rank`（27）；`lora_A` Kaiming、`lora_B` 零初始化（36-37）。
- 前向：`x[B,N,dim]`；按 `modality_ids` 选每组 A/B（46-47）；`bmm(x,A^T)`、`bmm(...,B^T)`；`F.linear(x,weight,bias)+scaling*residual`（48-50）。**所有 token（含 CLS）使用同一图像模态的专家**。
- 安装：`REID/model/backbones/vit_pytorch.py:321-329`，`for block in self.blocks[-layers:]`，把 `block.mlp.fc1/fc2` 替换为 `ModalityExpertLinear`；默认 `layers=2`（block 11/12，0 基 10/11），`rank=4`、`alpha=4.0`。
- Mlp/Block 传递 modality_ids：`vit_pytorch.py:101-109`、`:154-157`。
- 训练模态 id：`REID/model/make_model.py:91-107`，`arange(M).repeat_interleave(B)`；推理：`make_model.py:108-115`，`mode-1` 或显式 `modality_ids`。
- 配置：`configs/UAD_FFN_Experts_DDP4.yml`；实际训练日志显示 `MODEL.DIST_TRAIN True`、`SOLVER.IMS_PER_BATCH 256`、`SOLVER.BASE_LR 0.032`、`OUTPUT_DIR logs/UAD_FFN_Experts_1000_DDP4_lr0032`（`REID/logs/1000——MER.txt:1-4`）。**注意：当前配置文件写的是 BASE_LR 0.008，README 也解释为 0.008；实际运行通过命令行覆盖为 0.032，配置文件已与实际运行不一致。**

### 7.2 正向数据流与最终描述子

- 专家只改最后 2 个 block 的 FFN；共享 `weight/bias` 仍在，且 **完全可训练**（不是冻结 LoRA）。
- 由于 B 零初始化，初始专家输出 = 原 FFN；训练后专家给 FFN 加 rank-4 模态特定增量。
- 所有 token（含 CLS）都走同一图像的专家；block 12 的 CLS 也经过专家，因此 **不是空路径**。
- 最终检索仍是 final LN 后的 CLS（pre-BN）。

### 7.3 反向梯度流 / optimizer

- 专家参数全部在 optimizer 中；`REID/solver/make_optimizer.py` 无特例，base LR/WD（实际运行 0.032/1e-4，bias 2×LR）。
- `lora_B` 零初始化：第一步 `∂L/∂A=0`（因为 residual 依赖 B），`B` 先获得梯度；之后 A 才启动。与 KGL adapter 类似的两阶段启动。
- `torch.stack(lora_A)[modality_ids]` 让所有专家都在计算图中；未选中的专家得到零梯度（测试 `tests/test_modality_experts.py` 验证），不是 None；DDP `find_unused_parameters=True` 可处理。
- 共享 FFN 权重也收到梯度；专家不会独占模态信息。

### 7.4 损失与归一化

- 无新增损失；原 CE/WRT/ProCA/GPD 不变。
- 专家残差加在 `F.linear` 输出后，随后 GELU/dropout；没有额外归一化。
- 最终检索 L2。

### 7.5 训练-测试一致性

- 训练用 modality-major id；推理用 `mode-1`；测试 `tests/test_modality_experts.py` 验证 train/inference 特征一致（在注入非零 B 后）。
- 所有模态的专家都在 state_dict 中；`load_param` 对专家 checkpoint 严格检查（`make_model.py:125-143`）。
- Ground/aerial 同模态使用同一个专家；路由正确。

### 7.6 为什么接近基线但不提升

**[实验支持/本机日志]**

- DDP UAD baseline：overall mAP 11.02/R1 29.71/R5 46.32/R10 54.55（`REID/logs/1000.txt` 末尾）。
- ReID5o 专家：mAP 10.97/R1 29.54/R5 46.53/R10 54.40（`REID/logs/1000——MER.txt` 末尾）；即 −0.05/−0.17/+0.21/−0.15，与 02 一致。
- 模态 pair：
  - TIR→TIR：baseline 13.52 mAP/20.75 R1/2.28 mINP；专家 13.60/20.89/2.30，+0.08/+0.14/+0.02。
  - NIR→RGB：baseline R1 24.54；专家 23.79，−0.75。
- 说明专家带来 **模态内小收益**，但跨模态 NIR→RGB 变差，总体被抵消。

**[代码事实 + 机制假设]**

1. **全参数微调下容量冗余**：共享 FFN 已经可训练，rank-4 低秩专家只是给最后 2 层加一个很小的模态特定增量；主干本身可以表达模态特化，专家没有提供“新信息”，只是额外的低秩偏置。
2. **ProCA/GPD 与专家目标相反**：ProCA 把三模态中心拉向全局中心，GPD 用共享原型做身份区分；这两个损失鼓励共享身份空间，而模态专家鼓励模态特定变换。二者存在张力，专家学到的更多是模态风格而非跨模态互补。
3. **零初始化 B 的两阶段启动**：A 初始零梯度，B 先学；120 轮内专家可能只学到小幅修正。rank 4 也很窄。
4. **TIR 内小升、跨模态下降**：专家可以拟合 TIR 的模态内统计，但不能保证跨模态对应；NIR→RGB R1 −0.75 与“模态内特化不等于跨模态身份互补”一致。
5. **实际 LR 0.032**：与 DDP baseline 同 LR/同 batch（日志确认），因此这不是 LR 不公平导致的；专家本身的增量确实很小。

### 7.7 可复用教训

- 在共享主干已全参数微调时，低秩模态专家很容易变成冗余；要验证专家价值，应做“冻结共享 FFN、只训专家”或“专家 rank 扫描”的对照，否则无法区分“容量不足”和“容量冗余”。
- 模态专家最好只作用于 patch 或加一个跨模态共享约束，避免 CLS 被模态特定偏置直接拉走。
- 零初始化 B 的 LoRA 需要足够长的训练或 A 的预热；短训练下可能只学到很小增量。
- 评估要同时看模态内与跨模态；TIR→TIR 小升不能代表总体。

### 7.8 与 02 的差异

- 02 的 2.15.4/2.15.6 对结构、rank、alpha、零初始化、路由、结果的描述基本准确；本审计用本机日志复核了 TIR→TIR +0.08/+0.14、NIR→RGB −0.75。
- 02 的 2.15.5 写 Base LR 0.032，与实际日志一致；但当前 `configs/UAD_FFN_Experts_DDP4.yml` 写 0.008，README 也主张 0.008。这是 **代码快照与实际运行不一致**，复现时必须以日志/命令行覆盖为准；不是 02 的数值错误。
- 02 说“共享原线性权重保留可训练”，正确；可补充“这使专家在全参数微调下容量冗余”的机制解释。

---

## 8. M-SpecGene：Encoder 初始化迁移

### 8.1 调用链与加载

- 转换/加载：`MSPEC/model/backbones/mspecgene.py`：
  - `expected_encoder_keys`（31-40）：150 个 key（cls_token、pos_embed、patch_embed 2 个、12×12 blocks、norm 2 个）。
  - 映射（80-126）：支持 MMPretrain `layers.i.ln1/attn.qkv/ffn.layers.0.0/...` 和 det-transform 名称。
  - 位置编码 resize（163-198）：`patch_positions=pos_embed[:,1:].reshape(1, source_h, source_w, D)`，`F.interpolate(..., size=target_grid=(16,8), mode=bilinear, align_corners=False)`，再 flatten 回 token 维；CLS 位置原样保留。
  - 严格检查（214-275）：missing/unexpected/shape 全检查；150/150 必须齐全。
  - 加载（328-350）：`target_state[key].copy_(value)`，覆盖 UAD backbone 的 encoder。
- `MSPEC/model/make_model.py:55-59`：`pretrain_choice=='mspecgene'` 时调 `load_mspecgene_backbone`；之后 classifier/bottleneck 重新初始化（63-68）。
- 配置：`MSPEC/configs/UAD_MSpecGene.yml`，单卡 64/0.008，120 轮；`PRETRAIN_PATH='./pretrained/M-SpecGene_VIT-B_UAD_256x128.pth'`。
- 本机转换报告：`MSPEC/pretrained/M-SpecGene_VIT-B_UAD_256x128.pth.json` 显示 source=`M-SpecGene_VIT-B_seg_transform.pth`，加载 150/150，ratio 1.0，source pos `[1,197,768]`，target `[1,129,768]`，target grid `[16,8]`，interpolation bilinear，无 ignored tensors。
- 本机加载 converted checkpoint：150 tensors，全部 finite；`cls_token` std 0.439；`pos_embed` std 0.530、per-token norm 约 19.6；patch_embed std 0.0469、bias std 0.538；block0 norm1 weight mean 0.179；block0 qkv std 0.0609；block11 fc2 std 0.095。没有 NaN/Inf。

### 8.2 正向数据流与最终描述子

- 只换 encoder 初始化；backbone 结构、UAD 损失、BN/classifier、检索描述子都不变。
- 最终检索仍是 final LN 后 CLS（pre-BN）。
- 全主干微调，pos_embed 在 UAD 中是 **可训练参数**（ImageNet 路径也是）；但源 M-SpecGene 的 pos_embed 是 **固定 2D sincos，`requires_grad=False`**（`MSPEC_SRC/.../selfsup/mae.py:96`，`init_weights` 构建 sincos `110-114`）。

### 8.3 关键代码事实：位置编码转换缺陷

**[代码事实]**

- M-SpecGene 的 `build_2d_sincos_position_embedding`（`MSPEC_SRC/pretrain/mmpretrain-main_rgbt/mmpretrain/models/utils/position_encoding.py:123-182`）使用：
  - `grid_w=arange(w)`、`grid_h=arange(h)`；
  - `torch_meshgrid=torch.meshgrid(indexing='ij')`（17-19）；
  - `out_w=grid_w.flatten()*omega`、`out_h=grid_h.flatten()*omega`；
  - 返回 `[sin(w),cos(w),sin(h),cos(h)]`。
  - 因此 token 顺序是 **W-major（w 慢、h 快）**，不是 UAD patch token 的 H-major（h 慢、w 快）。
- 本机验证：`M-SpecGene_VIT-B_seg_transform.pth` 的 `pos_embed[0,1:]` 与上述 W-major sincos 14×14 的余弦为 1.000、相对误差 0.000（逐 token），确认它是标准 sincos 且为 W-major。
- `mspecgene.py:188-198` 的 `resize_position_embedding` 把 `pos_embed[:,1:]` reshape 成 `(1, source_h=14, source_w=14, D)`，即 **按 H-major** 解释；`F.interpolate(size=target_grid=(16,8))` 又按 `(H=16,W=8)` 解释目标。由于源表实际是 W-major，这会把 **w/h 轴和 token 顺序同时错配**。
- 本机量化：
  - 转换后的 `pos_embed` 与“正确的 W-major 14×14 → (w=8,h=16) 插值并按 UAD H-major 重排”的忠实版本：cos 平均 **0.818**（min 0.666），相对误差平均 **55.8%**（max 81.8%）。
  - 与标准 H-major 16×8 sincos 相比：cos 平均 **0.811**（min 0.657），相对误差平均 **58.7%**（max 82.7%）。
  - 作为对照：若按源 W-major 顺序做“忠实”插值（14×14→(w=8,h=16) 再重排到 UAD H-major），与标准 16×8 sincos 的余弦为 0.908、相对误差 40.5%；当前代码的实际结果更差（0.811/58.7%）。这里的 40.5% 仍包含“源 sincos 的转置约定与目标标准 sincos 不同”的成分，但当前代码额外把轴/顺序错配，误差更大。
- 结论：**位置编码不是忠实迁移，而是被轴交换/重排破坏的 sincos 表**。这不是“源域不适配”的抽象判断，而是可定位的转换缺陷。**[代码事实]**

### 8.4 其他域差异

- **[代码事实]** 源 M-SpecGene 是 MAE 自监督预训练（`MSPEC_SRC/.../configs/mae/mae_vit-base-p16_8xb512-amp-coslr-500e_in1k_siam.py`），`mask_ratio=0.90`，训练数据 RGBT550K，`RandomResizedCrop scale=224`（1:1），`SelfSupDataPreprocessor mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375]`。
- UAD：256×128（2:1）、`PIXEL_MEAN/STD=[0.5]*3`、身份 ReID 的 CE/WRT/ProCA/GPD。源模型的输入宽高比、归一化（等效对比度）、任务目标都不同。
- **[代码事实]** 源 pos_embed 是固定 sincos；UAD 把它复制到可训练 `pos_embed`，并对其施加 WD 1e-4（`make_optimizer` 无 no_weight_decay 特例）。源权重是在固定 sincos 下训练的，UAD 却允许它漂移；这进一步增加初始不匹配。
- **[代码事实]** 源 per-token pos norm 约 19.6；ImageNet 预训练 UAD 的 learned pos per-token norm 约 4.0（本机 `WK/WHU/jx_vit_base_p16_224-80ecf9dd.pth`：pos_embed norm 55.86/197≈3.98）。源位置信号强约 5×，attention 权重对位置编码的依赖可能更强。

### 8.5 为什么灾难性下降

**[实验支持/02]**

- H030：mAP 5.23/R1 14.79；对单卡 UAD 10.14/27.81 为 −4.91/−13.02pp。
- RGB→RGB mAP/R1 下降 12.14/18.41pp，比 TIR→TIR 的 5.18/8.57pp 更大；所有模态都退化，不是只损 TIR。

**[代码事实 + 机制假设]**

1. **位置编码错配（最强代码级候选）**：转换后的 pos 与忠实 sincos 的平均余弦仅 0.81、相对误差约 59%；源 attention 权重在固定 sincos 下训练，UAD 却从被交换/重排的表开始。虽然 UAD pos_embed 可训练，但 120 轮、LR 0.008、WD 1e-4 下未必能完全恢复；初始 attention 模式被破坏，可能把模型推入差的局部解。
2. **自监督 MAE 特征几何**：源是 90% 掩码重建/多光谱匹配，不是身份判别；其 CLS 几何不保证适合 ReID 的欧氏/内积排序。全参数微调可以适配，但 120 轮从 MAE 初始化开始，可能不如 ImageNet 监督初始化。
3. **输入域差异**：源 224×224 1:1、caffe 归一化；UAD 256×128 2:1、0.5/0.5 归一化；位置编码/感受野/对比度都不一致，进一步放大位置编码错配。
4. **不能把所有下降都归因于转换 bug**：没有用“修正后的 pos 转换”重训的对照，不能证明 bug 单独造成 13pp；但现有证据足以推翻 02 的“不能直接认定转换 bug”——至少存在确定的 pos 转换缺陷，且它是最可疑的首因。
5. **[机制假设]** 源模型可能在训练中依赖固定的 2D sincos 绝对位置；UAD 的 2:1 非方形网格需要不同的频率拉伸。当前代码既没有按源 W-major 顺序重排，也没有重新生成适合 16×8 的 sincos，导致 patch token 的“空间身份”从一开始就是错的。

### 8.6 可复用教训

- 迁移任何 MAE/ViT 权重前，必须检查源 `pos_embed` 是 **learned** 还是 **fixed sincos**；两者的插值/重排规则完全不同。
- 对 sincos，优先“按目标网格重新生成”或“严格按源 token 顺序重排”，不要直接套用 learned pos 的 H-major 双线性插值。
- 转换报告应记录源 pos 的类型、token 顺序、per-token norm、插值后的余弦/相对误差；当前报告只记录 shape，不足以发现这类错误。
- 迁移后先做 **线性探测/零样本检索** 或至少前几个 epoch 的 loss/accuracy 对照；灾难性下降应在正式 120 轮前被诊断。
- 源/目标 aspect ratio、归一化、任务目标（重建 vs 身份）都应作为迁移条件记录，不能只说“多光谱预训练更接近”。

### 8.7 与 02 的差异（重点）

- 02 的 2.11.4 说“CLS 位置保留、Patch 位置双线性到 16×8”，并认为“严格加载支持并非设计上允许大面积静默漏加载”，但 **没有检查源 pos_embed 是固定 sincos、且是 W-major 顺序**。本审计发现转换后的 pos 与忠实版本平均余弦仅 0.818，说明存在确定的轴/顺序错配。
- 02 的 2.11.7 说“不能直接认定转换 bug”；本审计认为应改为：**至少 pos_embed 转换存在可复现的缺陷；它是否是唯一主因仍需修正后重训验证，但不能再把“转换无 bug”作为前提。**
- 02 对 150/150 严格加载、classifier/BN 重初始化、全主干微调的描述正确；可补充源 MAE 90% mask、224 1:1、caffe 归一化、固定 sincos 与 UAD 2:1/0.5 归一化的差异。
- 02 说“RGB 同模态下降大于 TIR，表明损害不限于新光谱”，本机/02 数值支持；这与“位置编码/输入域整体错配”一致，而不是“只影响 TIR 的光谱先验”。

---

## 9. 跨实验综合

### 9.1 插入位置与最终描述子的关系

| 实验 | 插入位置 | 是否进入最终 CLS | 最后一层空路径 | 最终检索张量 |
|---|---|---|---|---|
| KGL Adapter | L1-L3 后，全 token | 是（后续 9 层传播） | 否 | pre-BN CLS |
| KGL HNSM | 训练-only 关系头 | 仅经共享主干间接 | 关系头不参与推理 | pre-BN CLS |
| PAT part | 输入到 L12，part↔patch attention | 是（part→patch→后续 CLS） | L12 part 对同层 CLS 无直接路径 | pre-BN CLS |
| WQVD A1 | L11 patch + L12 CLS，z=c+0.05u | 是（z） | 否 | pre-BN z |
| WQVD A2 | L12 patch + CLS，z=c*scale | 是（z） | 否 | pre-BN z |
| SCSR v3 | L6 后 TIR patch | 是（L7-L12 传播） | 否 | pre-BN CLS |
| SCSR v4 | Block7 norm1 后 attention 前 | 是（同层 attention + 后续） | 否 | pre-BN CLS |
| RLE | 输入增强 | 无结构改动 | 无 | pre-BN CLS |
| PEFT DPA | 每层 FFN 后 patch | L1-L11 是；L12 否 | **是（L12 DPA）** | pre-BN CLS |
| PEFT TCA | 每层 CLS tuning + 共享 adapter | 是 | 否 | pre-BN CLS |
| ReID5o | 最后 2 层 FFN fc1/fc2，全 token | 是 | 否 | pre-BN CLS |
| M-SpecGene | 只换 encoder 初始化 | 是 | 否 | pre-BN CLS |

### 9.2 损失与最终度量的关系

- 所有实验最终都用 **pre-BN CLS 的 L2 欧氏/内积**；训练中的 CE 用 BN(CLS)，WRT/ProCA/GPD 用 pre-BN CLS。新增模块若只改变训练路径、但不改变 pre-BN CLS，最终可能无感（SCSR v3 就是极端例子）。
- 新增损失若作用于私有头（HNSM 关系头、PAT part 头、WQVD local 头），必须检查它是否只训练私有头而不影响最终 CLS。HNSM/Part/WQVD 都有共享主干回传，但强度不同。
- 新增损失若与主损失目标冲突（PAT E3 的 strict cross-ID CSL、ReID5o 的模态专家 vs ProCA/GPD），即使代码路径正确，也可能退化。

### 9.3 “训练-测试一致”不等于“有效”

- KGL Adapter、WQVD A1/A2、SCSR v4、PEFT、ReID5o 都是训练/推理同路径；但效果从小升到退化不等。路径正确只是必要条件，不充分。
- SCSR v3 的 ON/OFF 证明“同权重推理时模块几乎不贡献”；PAT E1b 的随机 ROI 则证明“训练-测试不一致”会直接破坏可复现性。

### 9.4 证据结构

- **最强代码级负结果**：PEFT L12 DPA 空路径；SCSR v3 残差 3e-8；PAT E1b 随机 ROI 在 eval 生效；M-SpecGene pos 转换轴错配；RLE 对复制灰度图做独立通道增强。
- **最强实验级负结果**：ReID5o 总体 −0.05/−0.17 但 TIR 内 +0.08/+0.14；WQVD A1-CLS 比 UAD −1.50 R1、融合只补回 +0.92；PAT DDP 净 −59 Rank-1、CSL 饱和；M-SpecGene −4.91/−13.02。
- **最值得保留的小正结果**：KGL Adapter 四项总体小升（但 TIR 方向退化、单 seed）；PAT 单卡 E1a 小升（DDP 不复现）；SCSR v3/v4 的即时差异几乎为零（不应作为成功）。

---

## 10. 对 02 档案的修正/补充清单

1. **KGL 2.9.4**：补充 Adapter 同时作用于 CLS 与 Patch；补充无特殊 LR/WD、bias 2×LR；补充 `up` 零初始化导致的两阶段启动。02 对 HNSM 的描述基本准确。
2. **PAT 2.13.4/2.13.7**：补充 E1b 的 `random_part_roi` 在 eval/test 和 CSL memory 初始化中仍生效，这是训练-测试一致性问题。补充 CSL 未归一化点积 + 温度 0.1 导致 logit≈1000、最终 CSL loss≈0；补充 PSD 邻居 96.4% 同 ID、92.7% 同相机、错误 ID effective mass 1.22%；补充 DDP Rank-1 转移 482/423、净 −59。02 对 E1a/E1b 公式、PSD 替换 CE、E3/E4 strict cross-ID 的描述正确。
3. **WQVD 2.16**：补充本机 A1-CLS R1 −1.50pp、A1-Local R1 17.27、A1-Fusion 只补回 +0.92；补充局部 CE 作用于 u，而 u 同时进入 z；补充 A2 mask 初始≈0.5、scale≈1。02 的结构描述正确。
4. **SCSR 2.6**：补充 v3 residual/patch≈3e-8、gate≈2.2e-5，模块数值上近乎 dead；补充 v4 非零初始化（U std 0.5/output std 0.02）、m_rms≈1.66%、最大 11.9%；补充 v3 WD=1e-4、v4 WD=0；补充 v4 在 Block7 attention 前、同层可影响 CLS。02 的 ON/OFF 数值正确，但“弱有效通路”对 v3 偏轻。
5. **RLE 2.17**：补充 `read_image` 把所有图像转 3 通道，NIR/TIR 是复制灰度；独立通道 RRLE 产生伪彩色；E3/E4 只共享 TIR，NIR 仍独立；E2 仍保留 RGB/NIR 扰动。02 的流水线与 RGB 主损害描述正确。
6. **PEFT 2.14**：补充 L12 DPA 参数梯度为零（彻底死支路），不只是“不影响推理”；补充 TCA 共享 adapter 在 12 层共享权重、每层 tuning 直接加 CLS；补充全参数微调下 adapter 冗余。02 已正确指出空路径。
7. **ReID5o 2.15**：补充实际运行 LR 0.032（日志确认）与当前配置 0.008 不一致；补充专家 rank4/alpha4、184,320 参数、最后 2 层 fc1/fc2、B 零初始化、A Kaiming、全 token；补充共享 FFN 全可训练导致的容量冗余。02 结果数值基本正确。
8. **M-SpecGene 2.11**：**最重要修正**。补充源 pos_embed 是固定 2D sincos、W-major 顺序；当前 `resize_position_embedding` 按 H-major 解释并插值到 (16,8)，转换后 pos 与忠实版本平均余弦 0.818、相对误差 55.8%，与标准 16×8 sincos 平均余弦 0.811、相对误差 58.7%。因此至少存在确定的 pos 转换缺陷；02 的“不能直接认定转换 bug”应改为“存在可复现的位置编码转换缺陷，是否是唯一主因需修正后重训验证”。补充源 MAE 90% mask、224 1:1、caffe 归一化与 UAD 2:1/0.5 归一化的差异。

---

## 11. 附：关键文件/行号索引

### 基线

- `BASE/model/make_model.py:60-80`：classifier/BN/forward。
- `BASE/model/backbones/vit_pytorch.py:317-345`：CLS/12 blocks/final norm。
- `BASE/processor/processor.py:187-227`：四项损失、backward、GPD 更新。
- `BASE/processor/processor.py:298-312`：逐模态验证。
- `BASE/loss/make_loss.py:30-37`：CE+WRT。
- `BASE/utils/metrics.py:99,164`：同相机排除；`:462-465` L2；`:491-565` 九组；`:569-613` AG。

### KGL

- `KGL/model/backbones/vit_pytorch.py:241-257`：Adapter；`:315-338` 注册/零初始化；`:380-395` L1-L3 路由；`:397-402` forward。
- `KGL/model/make_model.py:29-43`：关系头；`:99-101` 构造；`:107-184` 挖掘；`:186-221` 训练/推理。
- `KGL/processor/processor.py:160-170` HNSM ramp；`:220-230` BCE。
- `KGL/solver/make_optimizer.py:5-23` 无特例。
- `KGL/configs/UAD_adapter.yml`、`UAD_hnsm.yml`。

### PAT

- `PAT_S/model/backbones/vit_pytorch.py:270-277` part token；`:339-384` mask；`:399-433` forward；`:422-425` 随机 ROI eval。
- `PAT_S/model/make_model.py:74-121` part 头/forward。
- `PAT_S/processor/processor.py:100-189` PartMemory；`:130` detach；`:160-166` 未归一化 logits；`:192-206` PSD；`:221-223` CE 替换；`:345-450` 训练。
- `PAT_D/processor/processor.py:125-162` all_gather。
- `PAT_D/analysis/retrieval/retrieval_summary.json`、`analysis/supervision/supervision_summary.json`（本机诊断）。
- `PAT_S/configs/E1a_PAT_Paper.yml` 等。

### WQVD

- `WQVD/model/backbones/vit_pytorch.py:319-354` L11/L12 返回。
- `WQVD/model/modules/a1_local.py:28-55`；`a2_modulation.py:37-54`。
- `WQVD/model/make_model.py:106-145` z/局部头/forward。
- `WQVD/processor/processor.py:232-235` 局部 CE。
- `WQVD/logs/A1_vs_UAD_gamma_analysis/feature_comparison.csv`（本机诊断）。

### SCSR

- `SCSR3/model/scsr.py:106-138`；`model/backbones/vit_pytorch.py:324-331,384-392`；`model/make_model.py:81-104`；`solver/make_optimizer.py:19-27`。
- `SCSR4/model/scsr_v41.py:83-133`；`model/backbones/vit_pytorch.py:157-172,349-384`；`model/make_model.py:70-94`；`solver/make_optimizer.py:6-30`。
- `SCSR/logsv3/SCSR_1000_diagnostics/scsr_on_off_metrics.csv`、`scsr_internal_activation.csv`；`SCSR/logsv4/mechanisms_fp32/v41_on_off_metrics.csv`、`v41_initial_vs_trained_gate.csv`。

### RLE

- `RLE/datasets/rle.py:14-120`；`datasets/make_dataloader.py:41-60`；`datasets/bases.py:8-16`（convert RGB）；`configs/UAD.yml`。

### PEFT

- `PEFT/model/backbones/vit_pytorch.py:109-146` DPA/TCA；`:177-241` Block；`:372-391` 注册/初始化；`:413-451` forward（`:447-451` 最后 norm/CLS）。
- `PEFT/configs/UAD_DPA.yml`、`UAD_TCA.yml`、`UAD_DPA_TCA.yml`。

### ReID5o

- `REID/model/backbones/modality_experts.py:13-50`。
- `REID/model/backbones/vit_pytorch.py:101-109,154-157,321-362`。
- `REID/model/make_model.py:91-122`。
- `REID/configs/UAD_FFN_Experts_DDP4.yml`（BASE_LR 0.008 与日志 0.032 不一致）。
- `REID/logs/1000.txt`（UAD baseline）、`REID/logs/1000——MER.txt`（专家）末尾。

### M-SpecGene

- `MSPEC/model/backbones/mspecgene.py:31-40,80-126,163-198,214-275,328-350`。
- `MSPEC/model/make_model.py:55-59`。
- `MSPEC/pretrained/M-SpecGene_VIT-B_UAD_256x128.pth.json`（本机转换报告）。
- `MSPEC_SRC/pretrain/mmpretrain-main_rgbt/mmpretrain/models/utils/position_encoding.py:123-182`（sincos W-major）。
- `MSPEC_SRC/pretrain/mmpretrain-main_rgbt/mmpretrain/models/selfsup/mae.py:96,110-114,195-210`（固定 sincos）。
- `MSPEC_SRC/pretrain/mmpretrain-main_rgbt/configs/_base_/datasets/rgbt550k_bs512_mae_base.py`（224/归一化）。
- `MSPEC/configs/UAD_MSpecGene.yml`。

---

## 12. 总体结论（按证据等级）

- **[代码事实]** 第二组多数模块都“确实接入了最终 CLS”，但接入位置、是否被后续层稀释、是否有私有头吸收任务，差异很大；PEFT L12 DPA 是唯一确定的“空路径”，SCSR v3 是“路径存在但数值近零”的极端。
- **[实验支持]** 小正结果（KGL Adapter、PAT 单卡 E1a）幅度小、单 seed、且不迁移到 DDP；大多数模块在总体指标上无稳定收益。
- **[实验支持]** 大负结果都有明确的结构性解释：RLE 的颜色/伪彩色破坏、ReID5o 的容量冗余、PEFT TCA 的直接 CLS 扰动、M-SpecGene 的位置编码/域错配。
- **[机制假设]** 这些失败不能简单归因于“模块没用”；更常见的是“模块与现有全参数微调/共享身份空间/最终 CLS 度量不匹配”。后续复用应优先做同权重 ON/OFF、逐层路径追踪、邻居分层统计和初始化/位置编码的忠实性检查。
