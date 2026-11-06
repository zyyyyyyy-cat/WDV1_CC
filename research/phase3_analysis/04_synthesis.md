# UAD/ViT 深度分析最终综合（Phase 3）

> 范围：UAD/ViT 单模型，含 WDV1-A0；不含 CLIP/WC、不做跨模型融合。
> 依据：`01_historical_uad_mechanisms.md`、`02_metric_phenomena.md`、`03_literature_2025_2026_round2.md`、`05_Results.csv`、A-V1/WCV1 逐 query 诊断、A0 日志。
> 本文是最终综合分析；`00_deep_analysis.md` 是早期底稿，其中 A0“已饱和”、平台优先等结论已被本文修正。

---

## 0. 本轮修正的八个关键认识

1. **A0 是“空对照”，不是“训练 recipe 到极限”。**
   - A0：LR=6.4e-5 恒定 30 epoch，LR 积分 = 0.00192；基线 120 epoch 的分段 LR 积分 ≈1.392。A0 只给了基线 **0.14%** 的优化预算；等价预算需要约 890 epoch。
   - 同样 30 epoch，基线尾段（LR 3.2e-4）是 +0.29 mAP/+0.97 R1；A0 是 0.00/−0.17。
   - 所以 A0 只能说明“在 cosine floor 上再走 30 epoch 什么都不会发生”，**不能否定 A1（5e-4 重启）或 A2（200 epoch 从头）**。训练 recipe 仍是未充分验证的备选，但不是当前主攻方向。

2. **健康模块区间里，mAP 与 R1 几乎正交。**
   - 全量 45 个实验 corr(ΔmAP,ΔR1)=0.992，是被 RLE/ICRE/M-SpecGene 等坍塌样本撑出的假相关。
   - 剔除 |ΔmAP|>0.6 的坍塌样本后（n=32），corr(ΔmAP,ΔR1)=**0.379**；corr(ΔmAP,ΔmINP)=**0.808**；corr(ΔR1,ΔmINP)=**−0.084**。
   - 指标分成两个因子：**因子 I = {mAP, mINP}（覆盖/深度）**；**因子 II = {R1,R5,R10}（边界）**，其中 R5/R10 是 R1 的稳定投影（r=0.90），R1 是最噪声的端点。

3. **mAP 由头部 query 决定，R1 由全体 query 决定。**
   - Top-10% query 贡献 39.9% 的 AP，Top-20% 贡献 61.9%，后 50% 只贡献 9.2%。
   - R1 每个 query 等权；1pp = 64 个 query 净翻转。
   - UAD 中位 query 的 margin = −0.0475；只有 29.7% query margin>0，正好等于 R1=29.71%。
   - 任何“只改善好 query 的第 2–100 名”的改动抬 mAP 不动 R1；任何“只挪边界 query”的改动动 R1 不动 mAP。

4. **“mAP↑R1↓”的机制是 max 统计 vs 平均统计，不是正样本变差。**
   - A-V1：472 个 R1 新失败中 267 个（56.6%）正样本相似度反而上升；463/472（98.1%）伴随最危险负样本上升。
   - 全体负样本均值 ↓0.023，但每 query best-negative ↑0.039、best-positive ↑0.029 → margin −0.010。
   - HiHR/ViSA 的固定 20% 拼接在数学上就是把新分量的方差加进 cosine；分数离散度↑ → max over 87k negative 抬得比平均负例多 → top-1 margin 收缩。
   - **“mAP 升”不能作为机制有效的证据；“R1 不掉”才是。**

5. **平台不是主要 mAP 瓶颈，模态才是。**
   - 协议难度：`same_modality_same_platform` mAP 20.08；`platform_only` 21.89（≈无损）；`modality_only` 9.80（−10.3pp）。
   - 最危险负样本：71.4% 同平台、67.7% 同模态、0% 同相机 → top-1 竞争主要来自同域近邻，不是平台。
   - 早前“平台差异 0.388 vs 模态 0.954，所以平台优先”的判断只说明特征空间质心距离，不说明检索难度；**必须收回平台优先的结论**。平台 token 降为第二/第三候选。

6. **排序损失的可迁移性由候选覆盖决定。**
   - A-V1/B-B 每卡只有 15 个负身份（3.0%）；WCV1 4 卡 gather 后 63 个（12.6%）；危险身份出现率 11.9%、危险图像 1.0%。
   - B-B 把 batch 内 top-1 错误 94.6%→0.75%，全库 R1 仍 −2.84；WCV1 把“batch 内可分离但全库错” 4.95%→2.45%，全库 R1 仍 −1.33。
   - **batch 内排序成功不能外推全库 R1；不要再做主机制。**

7. **名义权重 ≠ 梯度份额；单 seed 不能判定 <0.5pp。**
   - TAUR rank λ=0.1，对 adapter 梯度仅 UAD 的 1.5–2.1%，对主干 0；B-B 固定权重 85.8 全程占标量 40%；WCV1 标量 44.5%、cos 仅 0.30、不衰减；H018 只改 LR 就 −2.67/−6.57。
   - 同配方重跑 H029 −0.09/−0.77；ViSA 两 seed 差 0.24/1.13。按此尺度，KGL +0.15/+0.21、PAT single +0.31/+0.37、SCSR +0.03/+0.05 等都在 1–2 倍噪声内；只有 A-V1/B-B/ICRE/SFTC/RLE/M-SpecGene 明确超出。
   - 后续任何 <0.5pp 的结论必须 2–3 seed；<1.1pp R1 的变化不能号称机制有效。

8. **A0 的附带价值是噪声带，不是方法结论。**
   - 同模型再训 30 epoch：mAP ±0.01、R1 ≤0.2；评测确定性为 0；seed 方差 mAP 0.24/R1 1.13。
   - H015 同权重只换 1000GD 子集，就能造出 +3.38 mAP/+1.90 R1/−0.18 R5/−0.99 R10——**协议/子集效应比绝大多数模块大一个量级**。

> 附一条核对：`02_metric_phenomena.md` 称 overall 走 `eval_func_top_k`（top_k=50000）、九组合走 `eval_func`，两者是不同函数。核对 A0 日志与 `config/defaults.py`：`TOP_K_EVAL=0`，overall 与九组合都走 `eval_func`，差异只是 gallery 规模（93609 vs 31203）。该条不采纳；其余指标结构结论不受影响。

---

## 1. 指标结构：两个因子

### 1.1 因子划分（|ΔmAP|≤0.6，n=32）

| | ΔmAP | ΔR1 | ΔR5 | ΔR10 | ΔmINP |
|---|---:|---:|---:|---:|---:|
| ΔmAP | 1 | 0.379 | 0.412 | — | **0.808** |
| ΔR1 | | 1 | — | — | **−0.084** |
| ΔR5 | | | 1 | **0.902** | — |
| ΔmINP | | | | | 1 |

- **因子 I（覆盖/深度）**：mAP、mINP。mAP 是 105 个正样本的排名积分；mINP 是最后一个正样本的深度。
- **因子 II（边界）**：R1、R5、R10。R1 是 max over 87k negatives 的阈值统计；R5/R10 是较稳的投影。
- 因此“全指标 +1”必须同时改善两个因子；只改一个因子的机制最多做到“一升一平/一降”。

### 1.2 关键量化

- 正样本中位数 105（min 18，max 666）；正样本越多，R1/mAP 越高（20–49: 22.96/8.92；≥100: 34.05/12.27）。
- 首个正样本排名中位数 7，p90 243，p99 2708；所以 R5=46.32、R10=54.55。
- 最后正样本名次中位数 30,518/81,009（≈37% 深度）；overall mINP=0.6595%。
- 近 tie：|margin|<0.02 占 16.5% query，其中 39.9% 会翻转 top-1；全部 812 个翻转中 88% 来自 |margin|<0.05。
- 结论：**R1 的 1pp 不是 64 个独立判断，而是 0.05 宽度薄壳里的净漂移**；任何扰动都能翻转一部分。

### 1.3 总体 vs 九组合

- ΔmAP(overall) ≈ 0.908 × mean₉(ΔmAP)；ΔR1(overall) ≈ 1.51 × mean₉(ΔR1)。
- 相对变化一致（斜率 1.02/0.98）；绝对差只是操作点不同（11.02 vs 12.29；29.71 vs 19.12）。
- 总体 R1 29.71 vs 同模态 pair R1 30.40：**混合 gallery 是 −0.7pp 的竞争惩罚，不是并集红利**。
- 单对 R1 +1pp 只换总体 +0.28–0.29pp；单对 mAP +1pp 换总体 +0.11pp。**单对策略几乎不可能让总体 mAP +1pp（需单对 +9pp），但可能让总体 R1 +1pp（需单对 +3.2–3.8pp）。**

---

## 2. 历史机制：七条规律

### 2.1 mAP 可恢复，R1 是脆弱的路径相关量

- HiHR Stage1→Stage2 重 warmup：10.72/27.99 → 9.83/26.76（−0.89/−1.23）；之后 mAP 反超，R1 再未回到 29.71。
- A0：mAP 6 次评估恒 11.03；R1 29.68→29.51。
- PEFT TCA/WQVD A2/ReID5o：mAP −0.39/0.00/−0.05，R1 −2.11/−0.71/−0.17。
- **任何改变优化路径的干预都先损失 R1；mAP 可恢复，R1 往往不能。**

### 2.2 mAP↑R1↓ 的签名：同模态 R1 降 + 同模态 mINP 升

- HiHR：9/9 mAP 升；RR/NN/TT R1 −1.54/−0.70/−1.55；同模态 mINP +0.53/+0.32/+0.36。
- ViSA：9/9 mAP 升；RR/NN/TT R1 −1.07/−0.37/−0.05；mINP +0.27/+0.26/+0.17。
- PAT-DDP：ΔR1_same −1.40 vs cross +0.13。
- 机制：强模态 CLS 已含足量细粒度信息，额外分量只增加分数离散度；弱模态（TIR query）本身弱，额外证据能抬首位（HiHR TIR→RGB R1 +1.41）。
- **固定 20% 拼接**在训练/推理全程强制分配距离预算，无法在局部不可靠样本上自动降权。

### 2.3 双降簇：路径/输入破坏，不是监督对象错

- ICRE：全 patch 低秩平滑，gamma 0.01→0.436、残差 25–28%；−2.73/−6.90。
- SFTC：local K/V 与原 K/V 共享 257-key softmax；−1.00/−3.12；ON−OFF=+0.92/+1.36，但 OFF−baseline=−1.92/−4.48。
- RLE：`convert('RGB')` 后 NIR/TIR 是三通道复制灰度图，独立通道 RRLE 制造伪彩色；E2 关 TIR 仍降 1.27/4.23。
- M-SpecGene：pos_embed 轴错配（cos 0.818），全模态灾难下降。
- MulConv/SPM：区域 GeM 负值截断、10752 维分量范数隐式加权。
- 日志：所有方法最后 30 epoch 绝对斜率仍为正，没有“损失降、指标绝对降”的后期过拟合。

### 2.4 排序损失：候选覆盖是硬约束

- A-V1：15 负身份（3%）；分母含正例，正例占 softmax 36–44%；新增标量 11%→34.5%；−0.42/−2.06。
- B-B：15 负身份（3%）；固定权重 85.8 全程占标量 40%；batch top-1 94.6%→0.75%，全库 R1 −2.84。
- WCV1：63 负身份（12.6%）；正例吸引主导（pos/neg 梯度范数 1.88，cos 0.86/0.23）；ρ=0.10 只约束范数不约束方向，不衰减；+0.09/−1.33。
- **batch 内成功不能外推；不要再做主机制。**

### 2.5 名义权重 ≠ 梯度份额

- TAUR rank λ=0.1，对 adapter 仅 UAD 的 1.5–2.1%，主干 0。
- B-B 固定权重 85.8，标量占比 38.8%→40.8%，bb_raw 降 46× 仍不衰减。
- WCV1 标量占比 44.5%，cos 0.30。
- H018 只改 LR 0.032→0.008：−2.67/−6.57，大于几乎所有模块。
- 任何新损失必须报告：对主干的梯度范数、与主损失的 cos、占总损失比例、是否衰减。

### 2.6 单 seed 不能判定 <0.5pp

- 同配方重跑 H029 −0.09/−0.77；ViSA 两 seed 差 0.24/1.13。
- KGL +0.15/+0.21、PAT single +0.31/+0.37、SCSR +0.03/+0.05 都在噪声内。
- 只有 A-V1（−2.06）、B-B（−2.84）、ICRE/SFTC/RLE/M-SpecGene 明确超出。
- **后续必须 2–3 seed；报告 mean/range；1pp=64 query，别用单次 +0.3 宣布成功。**

### 2.7 训练/评测协议错配是公共背景

- WRT 只按 PID 分正负，训练同相机异身份负样本；评测排除全部同相机。
- 总体 R1 是九对并集；单对杠杆 0.28pp。
- 1000GD 是同权重子集，不是独立训练；GD R5/R10 只有 0.14/0.06 余量，是护栏。
- 1000GD daytime 未实现；mINP 只在 VI/3×3 有论文目标。

---

## 3. WDV1-A0：空对照的定量结论

| 项 | 值 |
|---|---|
| 初始化 | UAD-120 `transformer_120.pth` |
| LR | 6.4e-5 constant（峰 0.032 的 1/500） |
| 30 epoch LR 积分 | 0.00192 |
| 基线 120 epoch LR 积分 | ≈1.392 |
| 比例 | **0.14%** |
| 等价预算 | 约 890 epoch |
| 结果 | mAP 11.03 恒定；R1 29.68→29.51（−0.17） |
| 同 30 epoch 基线尾段（LR 3.2e-4） | +0.29/+0.97 |

**结论**：
- 不是收敛失败（CE 0.10、acc 0.98，与 120ep 一致）。
- 不是过拟合（损失不升、mAP 不动）。
- 不是“训练 recipe 到极限”的证明；只是一次**欠功效的空对照**。
- 价值：给出“同模型再训”的噪声带 ±0.01 mAP/±0.2 R1；并说明在 floor 上继续走不会有效。
- **若要验证训练 recipe，需要 A1（5e-4 重启）或 A2（200 epoch 从头）；但 seed 噪声 1.13pp R1，<1pp 的收益不可检测。训练 recipe 不是当前主攻方向，但不应被 A0 判死。**

---

## 4. 模态 vs 平台：纠正此前的平台优先

- 特征空间质心：RGB-G↔RGB-A 0.388；RGB-G↔NIR-G 0.954；RGB-G↔TIR-G 0.927。
- 但检索协议难度：
  - `same_modality_same_platform` mAP 20.08；
  - `platform_only` mAP 21.89（≈无损）；
  - `modality_only` mAP 9.80（−10.3pp）；
  - `modality_and_platform` 9.84。
- 最危险负样本：71.4% 同平台、67.7% 同模态、0% 同相机。
- **结论：平台差异在特征空间很大，但模型已经能处理平台；跨模态才是 mAP 的主要损失来源。平台 token 不是第一优先级；模态结构才是。**
- 这也解释了为什么“场景=模态×平台”不应直接做成 K=6 场景原型：会把有限的子中心容量浪费在平台维度，而平台对 mAP 几乎无损。

---

## 5. 2025–2026 文献：Top 8 与代码状态

| 排名 | 机制 | 是否加损失 | 是否改推理 | 代码 | 对 UAD 的作用 |
|---|---|---|---|---|---|
| 1 | **GPD 子中心原型**（sub-center ArcFace；K=3 用 ProCA 模态中心初始化/更新） | 否（改 GPD） | 否 | insightface 已核实 | 直接升级主增益来源；模态内方差；不改接口 |
| 2 | 平台 token + 软视角减除（ViSA 轻量版） | 否 | 是 | ViSA 已核实 | 平台条件化；但平台非主瓶颈 |
| 3 | CLS/Patch 独立 LayerNorm affine（ICLR 2026） | 否 | 是 | 未核实 | 零 FLOPs、恒等初始化；低风险 |
| 4 | 共享专家 + 3 模态 LoRA（r=4，后 6 层） | 否（load-balance 小项） | 是 | ViSA/ReID5o | 模态容量；路由塌缩/后期饱和风险 |
| 5 | 局部 patch 图融合写回最终 CLS（ViSA DLFM） | 否 | 是 | ViSA | TIR 局部结构；复杂/噪声 |
| 6 | GPD 暖启动 + 计数 EMA + tau 课程 | 否 | 否 | 本仓库已实现 | 早期原型可靠性；+0.2–0.5 |
| 7 | 每模态分类器（MDReID/WSL 最小部分） | 否 | 否 | MDReID/WSL | TIR 分类边界；闭集已 95% |
| 8 | 长时训练 + EMA + model soup | 否 | 否 | 通用 | 训练稳定性；seed 噪声内 |

**文献 Top 1 的精确实现（P05 + P09）**：
```
P: [C, K, D], counts: [C, K], K=3
f = normalize(feat)                       # [3B,D]
logits = einsum('nd,ckd->nck', f, P) / tau
logits = logits.max(dim=-1).values        # [3B,C]
loss = CE(logits, target.repeat(3))
# 更新：样本 i 属于模态 m、身份 c；用该模态中心 centers[m][i] 更新，
# 不能用 g_cent（跨模态平均会塌缩子中心）
k* = argmax_k cos(f_i, P[c,k])
P[c,k*] = normalize((1-m_eff)*P[c,k*] + m_eff*centers[m][i])
counts[c,k*] += 1; m_eff = clamp(1/(counts+1), 0.1, 0.5)
```
- 初始化：K=3，用 ProCA 的 RGB/NIR/TIR 中心分别填入；缺失模态复制类均值。
- 与现有损失：CE/WRT/ProCA 不动；GPD 的 `g_cent` 只做缺失原型兜底。
- tau：max over K 会抬高 logit；先保持 0.03，监控 GPD loss 量级。
- 风险：K 过大子中心饿死；建议 K=3 或 K=2；计数 EMA、暖启动。
- 代码状态：insightface 已核实含 sub-center ArcFace；ViSA/MDReID/DeMo/HiHR/SD-ReID/WSL 有完整代码；BIT/FSLoRA/PoSeD 仅 README/空仓库。

---

## 6. 唯一核心候选：GPD-SC（Sub-Center Global Prototype Discrimination）

### 6.1 为什么是它

- **GPD 是 UAD 唯一被证明有效的全局机制**（+2.0 mAP/+4.8 R1）；ProCA 只 +0.4/+0.6。
- **模态是主瓶颈**：跨模态 mAP 20.1→9.8；TIR 特征质量弱；单原型把 RGB/NIR/TIR 平均成一个点。
- **不新增损失、不改推理接口**：只把 `P [C,D]` 改为 `[C,K,D]`，logits 取 max_k；总损失结构不变。
- **全局覆盖**：所有训练身份都参与，不是 batch-local 3%–12.6%。
- **直接针对两个指标因子**：子中心保留模态内多峰（改善 mAP/mINP 覆盖），max over K 提供更硬的类间竞争（改善 R1 边界）。
- **文献有强先例**：sub-center ArcFace 在人脸开放集度量上有效；代码已核实。

### 6.2 机制

- Bank `P ∈ R^{C×K×768}`，K=3（RGB/NIR/TIR），`counts ∈ R^{C×K}`。
- 特征 `u = L2(f)`：
  $$
  s_i(c)=\max_{k=1}^{K}\frac{u_i^\top P_{c,k}}{\tau},\quad
  \mathcal L_{GPD\text{-}SC}=-\frac1N\sum_i\log\frac{\exp(s_i(y_i))}{\sum_c\exp(s_i(c))}
  $$
- 初始化：用训练集每 `(pid, modality)` 的类均值填入对应子中心；`counts=1`；缺失模态复制身份均值。
- 更新：每个样本按自己的模态 `m` 与身份 `c`，用该模态中心 `centers[m][i]` 更新对应子中心（或 argmax 子中心；见 6.4）；EMA `m_eff=clamp(1/(count+1),0.1,0.5)`；DDP all-gather 后更新。
- 推理：不变，仍用 pre-BN CLS 的 L2 欧氏；不加载 bank。
- 训练：120 epoch 从头 DDP，其余完全沿用 UAD 配方；第一轮不加 EMA/TTA/CA-WRT/LLRD/margin，只改 GPD。

### 6.3 正向/反向

```
f [3B,768] -> u=L2(f)
  -> logits[n,c,k] = u_n · P[c,k] / tau
  -> s[n,c] = max_k logits
  -> CE(s, target_rep)
  -> backward: 只有 f 收梯度；P 是 buffer
  -> optimizer step 后：按样本模态/身份 EMA 更新 P；counts++
```

- 梯度：max 操作把梯度路由到每个身份最相似的那个子中心（logit 最大者）；bank 不接收梯度；f→ViT。
- 归一化：u 与 P 都 L2；tau=0.03；不改变 CE/WRT/ProCA 的顺序。
- 参数量：C×K×768；C=500/K=3 → 1.15M floats（4.6MB）；C=1000/K=3 → 9.2MB；可接受。
- 计算：logits [3B,C,K]，比原 GPD 多 K 倍；K=3；3B=768、C=500 → 1.15M logits；可接受。
- DDP：all-gather 每模态中心/特征；更新一致。

### 6.4 需要红队确认的两个实现选择

1. **子中心分配**：
   - A. 强制按模态索引：样本 m 更新 `P[c,m]`；语义稳定，不会饿死；但 max 正样本可能选择其他模态中心。
   - B. 按 argmax 子中心：标准 sub-center；数据驱动；但可能把 TIR 特征写进 RGB 子中心，导致模态语义漂移。
   - 建议：**A 为主（模态标签已知，稳定）；B 作为消融**。若 A 的 max 正样本经常选错模态，再改 B。
2. **更新目标**：
   - 用 `centers[m][i]`（ProCA 模态中心，K=4 样本均值）而不是单样本特征；降低噪声。
   - 不能用 `g_cent`（跨模态平均），否则子中心塌缩。
   - 需要修改 `PCA` 返回 `centers` 列表（当前只返回 `g_cent`）。

### 6.5 预期与风险

- 预期：mAP +0.3–1.0、R1 +0.3–1.0、R5/R10 +0.2–0.8、mINP 不降；TIR/NIR 方向更明显。
- 风险：
  - K=3 子中心饿死/塌缩；监控每子中心 counts 与 usage。
  - max over K 抬高 logit，tau=0.03 可能过尖；先保持，监控 GPD loss。
  - GPD 是主增益，替换有风险；第一轮单变量、120 epoch、报告逐 epoch 尾段。
  - 单 seed 无法分辨 <0.5pp；第一轮看方向与量级，第二轮 2–3 seed。
- 失败回退：K=1（原 GPD）、K=2、更新用 argmax、tau=0.04.

### 6.6 与历史失败模式的区别

- 不是新增损失：总损失不变，不引入梯度预算竞争。
- 不是 batch-local：所有 C 个身份参与，覆盖 100%。
- 不是固定融合：不改特征维度/推理；不增加距离分量。
- 不是零初始化+WD：暖启动 + 计数 EMA；子中心有明确模态语义。
- 不是每相机原型：K=3 模态，不是稀疏相机组。

---

## 7. 备选机制（第二/第三优先，不是本轮主线）

| 机制 | 为什么不是第一 | 何时启用 |
|---|---|---|
| 平台 token + 软视角减除 | 平台对 mAP 几乎无损；AG 只有 ~1pp 缺口 | GPD-SC 后 AG 仍差，再做 |
| CLS/Patch 独立 LayerNorm affine | 零成本但预期 +0.2–0.8；单独不足以 +1 | GPD-SC 通过后作为低风险叠加 |
| 共享专家 + 3 模态 LoRA | 容量/路由风险；历史上后期饱和 | GPD-SC 后 TIR 仍弱 |
| 局部 patch 图融合写回 CLS | 复杂；历史局部融合多失败 | 仅 TIR 专项，最后 |
| 每模态分类器 | 闭集已 95%；收益可能小 | 与 GPD-SC 叠加消融 |
| 长时训练 + EMA + soup | A0 空对照；A1/A2 未验证；seed 噪声 | 作为底座对照，不作为主机制 |
| 更强 vision-only 初始化（DINOv2/MAE） | 换初始化/位置编码；工程大；历史 M-SpecGene/ViT-L 失败 | 若 GPD-SC 不足且允许换初始化 |

---

## 8. 下一步实验设计（只跑一个核心）

1. **实现 GPD-SC K=3**：bank `[C,3,D]`、max_k logits、PCA 返回模态中心、按模态更新、暖启动、counts。
2. **配置**：`configs/SCGPD_UAD.yml`，120 epoch 从头 DDP，batch 256/LR 0.032，其余 UAD 原配方；不开 EMA/TTA/CA-WRT/LLRD。
3. **Gate**：
   - 1000 AS：mAP ≥ +0.3、R1 ≥ +0.3、R5/R10 ≥ +0.2，mINP 不降；无任何指标下降超过 0.1。
   - 若 R1 下降 >0.3 或 mAP 不升，停止；先分析子中心 usage/模态分配。
   - GD R5/R10 不低于 46.0/53.5。
4. **Ablation（同一代码开关）**：
   - K=1（原 GPD）、K=2、K=3。
   - 更新目标：模态中心 vs 单样本；分配：强制模态 vs argmax。
   - 暖启动 on/off；tau 0.03/0.04。
5. **第二轮**：2–3 seed；报告 mean/range；paired bootstrap。
6. **若通过**：再加 CLS/Patch LN 或平台 token（单变量），最后 2337。

---

## 9. 一句话总结

**UAD 的主要瓶颈是模态，不是平台；mAP/mINP 是覆盖因子，R1/R5/R10 是边界因子；A0 是空对照，不是训练到极限。唯一主候选是把 GPD 的单原型升级为 K=3 模态子中心（max_k logits、ProCA 模态中心初始化/更新、零新损失、零推理改动），先跑 120 epoch 单变量，再看指标决定下一步。**
