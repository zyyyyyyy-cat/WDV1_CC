# Phase 1：评测协议核对与 +1.0pp 目标分析

> 分析日期：2026-10-29  
> 范围：只做静态分析，不修改代码。对照文件为本地 `WHU` 代码、`WHU` 测试日志、`05_Results.csv`、`02_Previous_Experiments.md`、`00_Research_Overview.md` 与论文 PDF。  
> 输出：本文件 + 同目录 `target_table.csv`（126 行，UTF-8-BOM，空值表示论文或本地未报告该指标）。  
> 证据标签：**[P]** 论文事实，**[C]** 代码事实，**[R]** 结果/日志事实，**[D]** 文档/CSV 事实，**[I]** 推算或机制推论，**[U]** 未决项。

---

## 0. 结论摘要

1. **同相机排除规则与论文 Eq.(2) 一致，但比 Market1501 更严格**：代码排除的是所有与 query 同 `camid` 的 gallery，而不是仅同 `pid+camid`。[C] metrics.py L99–L100、L164；[P] Eq.(2) `c ≠ c_q`。因此同相机的异身份负样本也被排除，而同相机同身份正样本在训练中仍然存在，构成训练—评测目标错配（见第 6 节）。
2. **九组合是 3×3 的 query 模态 × gallery 模态**，只按模态切分、不按平台切分；每个组合重新构造候选 gallery 并用同一个 `eval_func` 评测。[C] metrics.py L491–L565。它**不是**论文/代码中另一套 36 条“场景对”协议；后者只在 `TEST.DETAILED_DOMAIN_EVAL=True` 时运行，而该开关默认 False，现有日志也没有这些行。[C] metrics.py L215–L256、L615；defaults.py L199。
3. **AG-ReID 用 camid 划分**：加载时 `camid -= 1`，因此原始 c1–c5 → 0–4 为 ground，c6–c7 → 5–6 为 aerial；代码用 `camid < 5` / `>= 5` 切分，所有模态都保留。[C] whu_mars.py L92；metrics.py L569–L613。与论文“5 个地面相机节点 + 2 个 UAV 平台”一致。[P] Sec. 3。
4. **1000-GD 的代码条件只实现了“ground”，没有实现“daytime”**：过滤条件是 `pid∈[1000,1999]` 且原始 `camid<=5`；文件名正则只解析 `pid` 和 `camid`，没有任何 day/night 字段。[C] whu_mars.py L66、L75、L89–L90。[P] Table 3 caption 明确写的是 “ground and daytime queries and galleries only”。所以本地 1000-GD 是否严格等于论文 GD，取决于数据 split 是否已经把夜间地面图排除；**代码本身没有做这件事**。[U]
5. **mINP 只在模态对/VI 结果中报告，总体 AS-ReID 不报告**：`eval_func` 会计算 `all_INP`，但 `compute()` 在总体分支调用时没有取回 mINP；只有在 3×3 pair 分支显式 `return_minp=True`。[C] metrics.py L191–L199、L305、L487–L488。论文也只对 VI-ReID 报告 mINP（Table 5/6），Table 3/4 没有。[P] Sec. 5.1。因此 **AS/AG 的总体 mINP 没有“论文值 +1.0pp”的合法目标**。
6. **本地基线几乎复现论文 Table 3**：H014(1000) 11.02/29.71/46.32/54.55 vs 论文 11.0/29.5/46.3/54.8；H015(1000GD) 14.40/31.61/46.14/53.56 vs 14.5/31.9/45.0/52.5；H020(2337) 9.60/25.73/40.90/48.39 vs 9.6/25.7/40.9/48.2。[D] 05_Results R0159/R0171/R0229；[P] Table 3。按“论文 +1.0pp”口径，多数 AS 指标只差约 0.8–1.3pp；**1000GD 的 R5/R10 已经超过目标**（分别高 0.14/0.06pp）。[I]
7. **总体 mAP 近似可加，R1 不可加**：Σ AP 是逐 query 对全 gallery 的 AP 平均，所以单个模态对的 mAP 提升对总体的贡献约等于其 query 份额 × 该 gallery 模态的相关样本份额；1000/1000GD 下每对约 1/9。R1 是逐 query 的 top-1 命中率，是跨 gallery 模态的“并集”，不能把九个 pair R1 平均成总体 R1：H014 的 pair R1 均值只有 19.12%，总体 R1 是 29.71%。[I]
8. **杠杆排序（综合 mAP 权重、R1 净新增、当前难度）**：TIR 相关五对（TIR→RGB/NIR/TIR、RGB→TIR、NIR→TIR）是最大结构性杠杆，占 1000/1000GD 总体 mAP 权重的 **55.6%**、占 2337 的 **49.7%**；其次是 NIR→NIR（同模态最大 R1 缺口 2.83pp）；再次是 RGB↔NIR 跨模态；RGB→RGB 的 R1 已达标，只对 mAP/mINP 有约 1.3pp 缺口。[I]
9. **要 overall +1.0pp mAP，均匀提升九对时每对约 +1.0pp；只提升一个 query 模态的三对时每对约 +3.0pp；只提升单对时需要约 +9.0pp（1000/1000GD）或 +7.9–11.9pp（2337），后者不现实**。要 overall +1.0pp R1，单对硬模态约需 +3.2–3.8pp，均匀九对约需 +1.0–1.5pp；TIR 相关五对平均约 +1.2–2.0pp。[I] 详见第 4 节。
10. **跨数据集可迁移性**：1000GD 与 1000 共用 H014 权重，只是筛选 query/gallery（H015），所以 1000 的改进原则上直接传递；且 GD 的 R5/R10 已达标，只需 mAP/R1。2337 是独立训练（H020），身份数翻倍、TIR 更少，必须重新训练验证；最可能跨数据集成立的是 TIR 表征/对齐、相机感知负样本、原型可靠性这类与类别数无关的改动，而不是依赖 500 类/1000 类的超参。[I]

---

## 1. 资料与口径

### 1.1 证据来源

| 类型 | 路径 | 用途 |
|---|---|---|
| 代码 | `D:\CODE\Doing\WHU_base\WDV1_CC\WHU\utils\metrics.py` | 协议、mINP、九组合、AG |
| 代码 | `D:\CODE\Doing\WHU_base\WDV1_CC\WHU\datasets\whu_mars.py` | 1000-GD 过滤、camid 解析 |
| 代码 | `D:\CODE\Doing\WHU_base\WDV1_CC\WHU\datasets\make_dataloader.py` | query/gallery 拼接 |
| 代码 | `D:\CODE\Doing\WHU_base\WDV1_CC\WHU\datasets\sampler.py` | PKM 采样与少数模态重复 |
| 代码 | `D:\CODE\Doing\WHU_base\WDV1_CC\WHU\loss\wrt_loss.py` | WRT 正负样本定义 |
| 代码 | `D:\CODE\Doing\WHU_base\WDV1_CC\WHU\processor\processor.py` | 推理/评测入口 |
| 配置 | `D:\CODE\Doing\WHU_base\WDV1_CC\WHU\configs\UAD.yml`、`config\defaults.py` | 输入尺寸、TOP_K、详细域评测开关 |
| 日志 | `D:\CODE\Doing\WHU_Code\WHU\logs\1000\1000_test_log_20260912_1037.txt` | H014 推理，query 数、九组合、AG、总体 |
| 日志 | `D:\CODE\Doing\WHU_Code\WHU\logs\1000GD\1000GD_test_log_20260912_1045.txt` | H015 推理 |
| 日志 | `D:\CODE\Doing\WHU_Code\WHU\logs\2337\2337_test_log_20261025_1505.txt` | H020 推理 |
| 日志 | `D:\CODE\Doing\WHU_Code\WHU\logs\1000\1000_train_log_20260911_2145.txt` | H014 最终训练评测 |
| 日志 | `D:\CODE\Doing\WHU_Code\WHU\logs\2337\2337_train_log_20260912_1833.txt` | H020 最终训练评测 |
| CSV | `D:\CODE\Doing\WHU_Code\WHU_Research_Notes\05_Results.csv` | R0001–R0014 论文；R0159–R0170 H014；R0171–R0180 H015；R0229–R0240 H020 |
| 文档 | `D:\CODE\Doing\WHU_Code\WHU_Research_Notes\02_Previous_Experiments.md`、`00_Research_Overview.md` | 配置、边界、历史结论 |
| 论文 | `D:\CODE\Doing\WHU_Code\WHU.pdf` | Table 1–6、Sec. 3.3、Eq.(2)、Sec. 5.1 |

### 1.2 目标口径

- 论文值 **[P]**：论文 Table 3/4/5/6 的 UAD (Ours)。
- 本地复现值 **[R]**：H014（1000，DDP 4×）、H015（1000GD，加载 H014 权重）、H020（2337，DDP 4×）。
- 目标 **[I]**：`target = paper + 1.0pp`。这是用户口径，不是论文口径。
- 缺口：`paper_gap = paper - baseline`（负值表示本地高于论文）；`baseline_to_target = target - baseline`（负值表示本地已经超过目标）。`target_table.csv` 中两者均为有符号 pp。
- 论文没有报告的指标，`paper/target/paper_gap/baseline_to_target` 留空，不伪造目标。

---

## 2. 评测协议逐项核对

### 2.1 同相机 gallery 排除规则

**代码事实 [C]**

- `utils/metrics.py` L90：`valid_gallery = (g_pids == q_pid) & (g_camids != q_camid)`，用于计算 `num_rel`。
- L99：`remove = (g_camids[order] == q_camid)`；L100 `keep = np.invert(remove)`。上一行 L98 的 Market1501 版本 `(g_pids == q_pid) & (g_camids == q_camid)` 被注释掉。
- L163–L165 的 `eval_func` 同样使用 `g_camids == q_camid`。
- `eval_func_top_k` L90–L100 也是同一规则，但当前 `TEST.TOP_K_EVAL=0`，走全排序 `eval_func`。[C] defaults.py L193；日志 L152 显示 chunk 全排序。

**与论文一致性 [P]**

论文 Sec. 3.3 与 Eq.(2)：

```
g_q = {(x,y,s,c) ∈ G | c ≠ c_q}
```

即排除**所有与 query 相机相同的 gallery**，不是只排除同身份同相机。代码完全一致。

**重要后果 [I]**

1. 同相机、异身份的 gallery 负样本在评测中被移除；训练 WRT 却把它们当负样本（见 6.1）。这会让模型花容量去分离评测根本不考的 pair。
2. 同相机、同身份的正样本在训练中存在（同一相机多张图），但评测永远不考；模型可能利用相机背景/光照等捷径。
3. 由于相机节点是三光谱的，排除同 camid 会同时移除该相机下 RGB/NIR/TIR 三个模态的 gallery；这与论文“same camera”一致，但意味着跨模态同相机样本也不参与排序。
4. 对 AG-ReID，query 与 gallery 的相机集合本来就互斥（ground vs aerial），所以同相机排除在 AG 中实际不起作用。

**有效 query 数 [C]**

- `eval_func` 在 `orig_cmc` 全 0 时 `continue`（L170–L172），分母是 `num_valid_q`，不是全部 query（L197–L199）。
- `eval_func_top_k` 也有同样跳过逻辑（L92–L104）。
- 正常推理日志只打印 `query 2135` 等，不打印每个协议的有效 query 数；`_count_valid_queries`（L263–L265）只在 `evaluate_domain_protocols` 中使用，而详细域评测默认关闭（defaults.py L199），H014/H015/H020 日志中没有 `Category ... Query->Gallery` 行。[R]
- 因此总体和九组合的分母可能不同；两者不能严格互相还原。[I]

### 2.2 九组合的构造

**代码事实 [C]**

`R1_mAP_eval.compute()` L491–L534：

1. 取 `unique_q_mod_ids` 与 `unique_g_mod_ids`，默认都是 `[1,2,3]`。
2. 对每个 `(q_mod, g_mod)`：
   - `query_indices_for_pair = np.where(q_mids == q_mod)`
   - `gallery_indices_for_pair = np.where(g_mids == g_mod)`
   - 只在对应子集上重新计算距离矩阵 `compute_indices_chunked(qf_pair, gf_pair, top_k=0, ...)`
   - 调用 `eval_func(..., return_minp=True)`。
3. 打印 3×3 表（L553–L565），并在 L543–L551 单独抽取 `(2,1)` 与 `(3,1)` 作为 Table 5 的 NIR→RGB、TIR→RGB。

**关键性质**

- 九组合**只按 modality 切分**，不按 platform 切分；每个组合内部同时包含 ground 和 aerial（1000/2337）。这与 Table 6 “modality-pair evaluation of the single UAD model”一致。[P]
- 每个组合的 gallery 是该模态的全部 gallery，不是“混合 gallery 中再过滤”。因此：
  - 同一 query 在 pair 评测里只面对 1/3 的 gallery；在总体评测里面对全部 3 个模态的 gallery。
  - pair mAP/R1 与总体 mAP/R1 **不是可加分解**；不能把九个 pair 指标平均成总体。[I]
- 九组合的 query/gallery 数量、有效 query 数没有写入正常日志；只有 pair 的 R1/mAP/mINP。[R]
- 另一套 `_domain_protocol_definitions()`（L215–L256）定义了 36 条 `same_modality_same_platform`、`modality_only_ground`、`platform_only`、`modality_and_platform_*` 协议，并在 `evaluate_domain_protocols()` 中按 `_platform_mask` 切分。它不是用户所说的“九组合”，也没有在 H014/H015/H020 中运行。[C]

### 2.3 AG-ReID 的 camid 划分

**代码事实 [C]**

- `whu_mars.py` L88–L95：文件名正则 `r'(\d+)_c(\d+)'` 解析 `pid, camid`；L92 `camid -= 1` 后写入 dataset。
- `metrics.py` L569–L577：
  - `ground_q = np.where(q_camids < 5)`
  - `aerial_q = np.where(q_camids >= 5)`
  - gallery 同样切分。
- 只要求 ground/aerial 两侧非空（L583–L584），否则打印 skipped。
- 使用全部模态，不在 AG 内再按模态过滤；`eval_func` 仍做同相机排除，但因 q/g 相机集合互斥，实际不删样本。

**与数据语义的对应 [P][I]**

论文 Sec. 3：5 个 ground camera nodes + 2 个 UAV 平台。若原始相机编号为 c1–c5（ground）、c6–c7（aerial），减 1 后正好是 0–4 / 5–6，代码正确。若数据实际是 0-based 或相机编号顺序不同，AG 会被切错；代码没有显式 platform 字段，只有 camid 阈值。[U]

**日志验证 [R]**

- H014 1000 log L178–L180：A→G mAP 11.46/R1 31.63/R10 58.52；G→A 13.30/30.12/54.50。
- H020 2337 log L180–L181：A→G 9.51/26.29/49.47；G→A 10.92/27.64/50.89。
- H015 1000GD log L178：`AG-ReID evaluation skipped: both ground and aerial query/gallery samples are required.` 这与 1000GD 只保留 ground（camid<=5）一致。

**与论文 Table 4 的关系 [P]**

论文 Table 4 明确是 WHU-MARS-1000 上的 AG-ReID；因此：
- 1000 的 AG 有论文目标；
- 2337 的 AG 没有论文目标（本地 H020 值只能作为迁移参考）；
- 1000GD 的 AG 不适用（只有 ground）。

### 2.4 1000-GD 过滤条件

**代码事实 [C]**

`datasets/whu_mars.py`：

- L66：`pattern = re.compile(r'(\d+)_c(\d+)')`，只解析 pid 和 camid。
- L75 与 L90：`if apply_filter and (pid < 1000 or pid >= 2000 or camid > 5): continue`
- L89 注释：`# WHU-MARS-1000-GD: filter ground + daytime only`
- L36–L38：`apply_filter` 只传给 query/gallery，train 始终 `apply_filter=False`。
- L92：通过过滤后才 `camid -= 1`。

**它实际做了什么**

- 保留测试身份 `pid 1000–1999`；
- 保留原始相机 `camid 1–5`（ground）；
- **没有任何 day/night 解析、时间戳过滤、目录名过滤或元数据读取**。

**与论文的差异 [P][U]**

论文 Table 3 caption：“WHU-MARS-1000-GD evaluates the WHU-MARS-1000-trained model using ground and daytime queries and galleries only.” 代码只实现了 ground，daytime 未显式实现。是否等价取决于：
1. 数据 split 是否已经把夜间地面图排除在 query/test 目录之外；
2. 或者 ground 相机是否只在白天采集（论文写采集时间约 09:00–21:00，覆盖昼夜，因此不能假定 ground 全是白天）。

**可用数量证据 [R][I]**

- 1000GD query：每模态 399，总 1197；1000 query：每模态 2135，总 6405。399/2135 = 18.7%。[R] 1000GD log L148–L150；1000 log L148–L150。
- 若只按 ground 比例估算：论文 Table 1 的 ground 每模态约 46,212 / 总每模态约 61,974 = 74.5%，ground-only query 应接近 1,590（按比例采样假设），远大于 399。[I]
- 因此 399 这个数要么说明 query 采样在相机/平台上并不按图像比例，要么说明 split/过滤还隐含了 daytime 限制。**仅凭代码无法判定**。[U]
- 本地 1000GD 基线 14.40/31.61 vs 论文 14.5/31.9，差距仅 0.1/0.3pp，说明子集难度大致接近；但这不能证明 day/night 条件严格一致。[R][I]

**结论**：不能把“`pid 1000–1999, camid<=5`”直接等同于论文的 ground+daytime。应在数据清单层面核对 query/test 中地面图的昼夜构成；否则 1000GD 的目标比较存在协议偏差风险。

### 2.5 mINP 的定义与实现

**代码定义 [C]**

`utils/metrics.py` L190–L192：

```python
last_relevant = np.nonzero(orig_cmc)[0][-1]
all_INP.append(num_rel / (last_relevant + 1.0))
```

其中：
- `orig_cmc` 是排除同相机 gallery 后、按距离排序的 0/1 匹配序列；
- `num_rel` 是有效 gallery 中相关样本总数；
- `last_relevant` 是最后一个相关样本在有效 gallery 中的 0-based 下标；
- 因此 `INP = num_rel / rank_last`（rank_last 为 1-based 的最后相关排名），即“最后一个相关位置的精度”。

L199：`mINP = np.mean(all_INP)`，对有效 query 平均。

**论文口径 [P]**

- Sec. 5.1：“we additionally report mean Inverse Negative Precision (mINP) [47] for VI-ReID only.”
- Table 5（NIR→RGB、TIR→RGB）和 Table 6（3×3）报告 mINP。
- Table 3/4（AS/AG）不报告 mINP。

**定义风险 [U]**

论文引用 [47]（AGW）。文献中 mINP 存在两种常见写法：
- `num_rel / rank_last`（“最后一个相关位置的精度”，本地实现，项目 01 文档 L462 也这样描述）；
- `1 - (N - rank_last)/N = rank_last/N`（以 gallery 大小 N 归一化的 INP）。

本地实现属于第一种；它是否与 AGW 官方实现完全一致，需要看 [47] 的代码/补充材料。  
本地与论文的 mINP 数值非常接近（例如 1000 上 local/paper：RGB→RGB 5.06/5.36，NIR→NIR 2.40/2.64，TIR→TIR 2.28/2.35，NIR→RGB 2.82/3.10，TIR→RGB 1.30/1.36），且同目录的离线评测脚本 `offline_eval.py` L63 也使用 `inp = n_rel / float(last)`，说明项目内部口径一致、且很可能与论文相同。[I]  
但“很可能”不等于“已核实”：在把 mINP 当作正式目标前，仍建议用 AGW 官方实现或论文补充材料确认。若公式不同，`paper+1.0pp` 的 mINP 目标不可比。

**总体 AS-ReID 是否报告 mINP？**

- `compute()` 总体分支 L480–L488：
  - `top_k` 开启时调用 `eval_func_top_k`，该函数只返回 `cmc, mAP`，根本不计算 mINP [C] L71–L134；
  - `top_k=0`（当前配置）时调用 `eval_func`，但只接收 `cmc, mAP`，没有传 `return_minp=True`；`eval_func` 内部算出的 `all_INP` 被丢弃 [C] L487–L488。
- `evaluate_domain_protocols` 的 `overall_official` 行把 `mINP` 写成 `""` [C] L276–L278。
- 推理日志只打印 `mAP`、`Rank-1/5/10`，没有总体 mINP [R] 1000 log L182–L185；1000GD L180–L183；2337 L183–L186。
- 因此：**当前日志没有总体 AS-ReID mINP；论文也没有 AS mINP。** 若研究目标包含“总体 mINP +1pp”，必须先补报并冻结本地基线，不能使用“论文+1”。

**可用的本地补充 [D]（非正式日志，需确认来源）**

同目录的 `offline_eval.py`（L63 `inp = n_rel / float(last)`）与 `offline_calibration_raw.json` 使用同一公式对 H014 的 6,405 个 query 做了离线总体评测，`mode="raw"` 给出：

```
R1 0.29711, R5 0.46323, R10 0.54551, mAP 0.11023, mINP 0.006595 (=0.66%)
```

其中 R1/R5/R10/mAP 与 H014 日志完全一致，说明该离线结果可复现日志；总体 AS mINP 约 **0.66%**，显著低于各 pair mINP（2–5%），符合“总体 gallery 更大、最后一个相关样本更靠后”的预期。[I] 若要把总体 mINP 纳入目标，可先以这个 0.66% 作为基线，但应把该文件视为离线诊断，而不是正式训练日志。

### 2.6 query/gallery 数量

**已知 query 数 [R]**

| 数据集 | RGB | NIR | TIR | 合计 | 日志 |
|---|---:|---:|---:|---:|---|
| 1000 | 2,135 | 2,135 | 2,135 | 6,405 | 1000 log L148–L150 |
| 1000GD | 399 | 399 | 399 | 1,197 | 1000GD log L148–L150 |
| 2337 | 5,254 | 5,275 | 4,491 | 15,020 | 2337 log L149–L151 |

**gallery 数：日志未打印，以下为推算 [I]**

`make_dataloader.py` L116–L127 把 `dataset.query[modality] + dataset.gallery[modality]` 拼成一个 loader，`num_query = len(query[modality])`。代码没有打印 dataset 统计；`verbose` 在推理入口没有产生统计行。[C][R]

推算依据：论文 Table 2 中 1000 的 train 92,313 + test pool 93,609 = 185,922（Table 1 总数）；2337 的 217,504 + 217,116 = 434,620。两者都恰好等于数据集总图像数，说明 Table 2 的 “gallery images” 是包含 query 的 test pool，而不是去掉 query 后的 gallery。因此：

```
gallery = test_pool - query
```

- **1000**：1000 是平衡三模态子集，总 185,922 / 3 = 61,974 每模态；test pool 93,609 / 3 = 31,203 每模态；gallery = 31,203 − 2,135 = **29,068 每模态**，总计 **87,204**。[I]
- **1000GD**：日志没有 gallery 数。若只按 ground 比例 74.5% 估算，gallery ≈ 0.745 × 31,203 − 399 ≈ **22,847 每模态**（总约 68.5k）；若 daytime 进一步筛掉样本，则更低（可能 15k–23k）。这个区间足够做权重分析，因为三模态在 1000 中平衡，相对权重仍接近 1/3。[I][U]
- **2337**：按论文 Table 1 的模态总量 155,892 / 155,655 / 123,073（占比 35.9% / 35.8% / 28.3%）把 test pool 217,116 分摊，再减 query：
  - RGB ≈ 77,915 − 5,254 = **72,661**
  - NIR ≈ 77,804 − 5,275 = **72,529**
  - TIR ≈ 61,397 − 4,491 = **56,906**
  - 总 gallery ≈ **202,096**。[I]
  - 这是按总图像比例分摊的近似；实际 split 可能不同。TIR gallery 明显少于 RGB/NIR（约 −21.6%），与论文 Table 1 的 TIR 稀缺一致。

**query 份额 [I]**

| 数据集 | RGB | NIR | TIR |
|---|---:|---:|---:|
| 1000 | 33.33% | 33.33% | 33.33% |
| 1000GD | 33.33% | 33.33% | 33.33% |
| 2337 | 34.98% | 35.12% | 29.90% |

---

## 3. 精确目标表

### 3.1 AS-ReID（论文 Table 3）

| 数据集 | 指标 | 论文 | 本地基线 | 目标(论文+1) | paper−baseline | baseline→target |
|---|---|---:|---:|---:|---:|---:|
| 1000GD | mAP | 14.5 | 14.40 | 15.5 | +0.10 | **+1.10** |
| 1000GD | Rank-1 | 31.9 | 31.61 | 32.9 | +0.29 | **+1.29** |
| 1000GD | Rank-5 | 45.0 | 46.14 | 46.0 | −1.14 | **−0.14（已达标）** |
| 1000GD | Rank-10 | 52.5 | 53.56 | 53.5 | −1.06 | **−0.06（已达标）** |
| 1000GD | mINP | — | — | — | — | 无合法目标 |
| 1000 | mAP | 11.0 | 11.02 | 12.0 | −0.02 | **+0.98** |
| 1000 | Rank-1 | 29.5 | 29.71 | 30.5 | −0.21 | **+0.79** |
| 1000 | Rank-5 | 46.3 | 46.32 | 47.3 | −0.02 | **+0.98** |
| 1000 | Rank-10 | 54.8 | 54.55 | 55.8 | +0.25 | **+1.25** |
| 1000 | mINP | — | — | — | — | 无合法目标 |
| 2337 | mAP | 9.6 | 9.60 | 10.6 | 0.00 | **+1.00** |
| 2337 | Rank-1 | 25.7 | 25.73 | 26.7 | −0.03 | **+0.97** |
| 2337 | Rank-5 | 40.9 | 40.90 | 41.9 | 0.00 | **+1.00** |
| 2337 | Rank-10 | 48.2 | 48.39 | 49.2 | −0.19 | **+0.81** |
| 2337 | mINP | — | — | — | — | 无合法目标 |

来源：[P] Table 3；[D] 05_Results R0001–R0003、R0159、R0171、R0229；[R] H014 train log L1121–L1125，H015 test log L179–L183，H020 train log L1841–L1845。

**观察**

- 本地基线相对论文的误差都在 ±0.3pp 内，只有 1000GD R5/R10 比论文高 1pp 以上（−1.14/−1.06 paper_gap）。
- 目标缺口最大的是 1000 R10（1.25pp）、1000GD R1（1.29pp）、1000GD mAP（1.10pp）；最小的是 1000 R1（0.79pp）。
- 1000GD R5/R10 已经达标，不应再作为优化目标；1000GD 只需关注 mAP/R1。

### 3.2 AG-ReID（论文 Table 4，仅 1000）

| 方向 | 指标 | 论文 | 本地 H014 | 目标 | paper−baseline | baseline→target |
|---|---|---:|---:|---:|---:|---:|
| A→G | mAP | 11.5 | 11.46 | 12.5 | +0.04 | **+1.04** |
| A→G | Rank-1 | 31.9 | 31.63 | 32.9 | +0.27 | **+1.27** |
| A→G | Rank-10 | 58.3 | 58.52 | 59.3 | −0.22 | **+0.78** |
| G→A | mAP | 13.3 | 13.30 | 14.3 | 0.00 | **+1.00** |
| G→A | Rank-1 | 30.5 | 30.12 | 31.5 | +0.38 | **+1.38** |
| G→A | Rank-10 | 54.7 | 54.50 | 55.7 | +0.20 | **+1.20** |

来源：[P] Table 4；[D] R0004–R0005、R0169–R0170；[R] 1000 log L178–L180。

**2337 AG 本地值（无论文目标，仅迁移参考）**

| 方向 | mAP | Rank-1 | Rank-10 | 若拿 1000 论文+1 作参照的缺口 (mAP/R1/R10) |
|---|---:|---:|---:|---:|
| A→G | 9.51 | 26.29 | 49.47 | 2.99 / 6.61 / 9.83 |
| G→A | 10.92 | 27.64 | 50.89 | 3.38 / 3.86 / 4.81 |

来源：[D] R0239–R0240；[R] 2337 log L180–L181。  
**注意**：这不是合法的 2337 目标；论文 Table 4 只在 1000 上训练/评测。列在这里是为了说明 2337 跨平台难度远高于 1000。

### 3.3 VI-ReID（论文 Table 5，仅 1000）

| 方向 | 指标 | 论文 | 本地 H014 | 目标 | paper−baseline | baseline→target |
|---|---|---:|---:|---:|---:|---:|
| NIR→RGB | Rank-1 | 25.53 | 24.54 | 26.53 | +0.99 | **+1.99** |
| NIR→RGB | mAP | 16.58 | 15.66 | 17.58 | +0.92 | **+1.92** |
| NIR→RGB | mINP | 3.10 | 2.82 | 4.10 | +0.28 | **+1.28** |
| TIR→RGB | Rank-1 | 9.04 | 7.40 | 10.04 | +1.64 | **+2.64** |
| TIR→RGB | mAP | 6.32 | 5.85 | 7.32 | +0.47 | **+1.47** |
| TIR→RGB | mINP | 1.36 | 1.30 | 2.36 | +0.06 | **+1.06** |

来源：[P] Table 5；[D] R0009/R0012、R0163/R0166；[R] 1000 log L154–L162（pair 值），L164–L166（VI 表）。

**观察**

- 本地 VI 目标缺口比 AS 更大：NIR→RGB R1 差 1.99pp，TIR→RGB R1 差 2.64pp。
- 这可能是 256×128 vs 论文 144×288 的输入差异，也可能是训练随机性；见 6.3。
- 论文 Table 5 的 UAD 数值与 Table 6 的对应 cell 完全相同（NIR→RGB 25.53/16.58/3.10；TIR→RGB 9.04/6.32/1.36），说明论文很可能没有为 UAD 单独跑一个 288×144 的 VI 模型，或至少 UAD 行复用了 Table 6 的数字。这对“分辨率差异导致本地偏低”的解释是一个反证。[P][I]

**1000GD / 2337 的 VI 本地值（无论文目标）**

| 数据集 | 方向 | R1 | mAP | mINP |
|---|---|---:|---:|---:|
| 1000GD | NIR→RGB | 22.56 | 19.65 | 5.79 |
| 1000GD | TIR→RGB | 7.62 | 8.60 | 3.39 |
| 2337 | NIR→RGB | 22.02 | 13.31 | 1.85 |
| 2337 | TIR→RGB | 6.48 | 4.55 | 0.76 |

来源：[D] R0175/R0178、R0233/R0236；[R] H015 log L157/L160，2337 log L158/L161。

### 3.4 3×3 模态对（论文 Table 6，仅 1000）

格式：**论文 → 本地 H014 → 目标（论文+1）**；括号内为 `baseline→target`，负值表示已达标。

| Query→Gallery | Rank-1 | mAP | mINP |
|---|---|---|---|
| RGB→RGB | 40.47 → 42.48 → 41.47（**−1.01，已达标**） | 26.83 → 26.53 → 27.83（+1.30） | 5.36 → 5.06 → 6.36（+1.30） |
| RGB→NIR | 26.56 → 27.17 → 27.56（+0.39） | 16.52 → 16.07 → 17.52（+1.45） | 2.43 → 2.23 → 3.43（+1.20） |
| RGB→TIR | 8.10 → 7.49 → 9.10（+1.61） | 6.00 → 5.61 → 7.00（+1.39） | 1.11 → 1.09 → 2.11（+1.02） |
| NIR→RGB | 25.53 → 24.54 → 26.53（+1.99） | 16.58 → 15.66 → 17.58（+1.92） | 3.10 → 2.82 → 4.10（+1.28） |
| NIR→NIR | 29.79 → 27.96 → 30.79（**+2.83**） | 17.93 → 16.83 → 18.93（**+2.10**） | 2.64 → 2.40 → 3.64（+1.24） |
| NIR→TIR | 7.78 → 6.37 → 8.78（+2.41） | 5.87 → 5.25 → 6.87（+1.62） | 1.11 → 1.04 → 2.11（+1.07） |
| TIR→RGB | 9.04 → 7.40 → 10.04（**+2.64**） | 6.32 → 5.85 → 7.32（+1.47） | 1.36 → 1.30 → 2.36（+1.06） |
| TIR→NIR | 8.57 → 7.92 → 9.57（+1.65） | 5.69 → 5.25 → 6.69（+1.44） | 1.02 → 1.00 → 2.02（+1.02） |
| TIR→TIR | 20.98 → 20.75 → 21.98（+1.23） | 14.00 → 13.52 → 15.00（+1.48） | 2.35 → 2.28 → 3.35（+1.07） |

来源：[P] Table 6；[D] R0006–R0014、R0160–R0168；[R] 1000 log L154–L162。

**观察**

- 27 个 cell 中只有 **RGB→RGB Rank-1** 已达标（本地 42.48 > 目标 41.47）；其余 26 个都需提升。
- 同模态中 **NIR→NIR 缺口最大**（R1 +2.83，mAP +2.10）；TIR→TIR 缺口反而较小（R1 +1.23，mAP +1.48），因为论文 TIR→TIR 本身较低。
- 跨模态中 **TIR→RGB R1 +2.64**、**NIR→TIR R1 +2.41**、**NIR→RGB R1 +1.99** 最大；这些是弱模态交叉方向。
- mINP 缺口普遍在 +1.0–1.3pp，比较均匀；没有哪个 pair 的 mINP 已达标。
- 1000GD 与 2337 的同九对本地值见 `target_table.csv` 中 `(local only)` 行；它们没有论文目标，不能与 1000 的目标直接比较。

### 3.5 目标缺口汇总

- AS-ReID：12 个 paper-target cell，其中 **2 个已达标**（1000GD R5/R10）；其余 10 个在 0.79–1.29pp。
- AG-ReID：6 个 cell，全部需 0.78–1.38pp。
- VI-ReID：6 个 cell，全部需 1.06–2.64pp；最大是 TIR→RGB R1 2.64。
- 3×3：27 个 cell，**1 个已达标**（RGB→RGB R1）；其余 26 个需 0.39–2.83pp。
- 总体上，**AS 的缺口最小（≈1pp），VI/3×3 的 R1 缺口最大（≈1.6–2.8pp）**。若目标是“所有协议都 +1”，主要难点在 VI/3×3 的 R1 和 mINP，而不是 AS 的总体 mAP。

---

## 4. 综合指标的加权贡献与杠杆

### 4.1 总体 AS-ReID 的聚合方式

**代码事实 [C]**

- 每个模态的 val loader = query + gallery，前三/前 `num_query` 个为 query（`make_dataloader.py` L116–L127；`split_all` metrics.py L361–L382）。
- 三个模态的 query/gallery 被拼接成一个全局集合（metrics.py L425–L460），总体距离矩阵是 3 模态 query × 3 模态 gallery。
- 总体 mAP = 所有有效 query 的 AP 平均（metrics.py L198）；总体 R1 = 有效 query 中 top-1 正确的比例（L196–L197）。
- 因此：
  - **mAP**：逐 query AP 的宏平均，不是九个 pair mAP 的简单平均；
  - **R1**：逐 query 命中率，是跨 gallery 模态的并集；
  - **mINP**：总体未报告（见 2.5）。

**数值证据 [I]**

H014 1000：

- 九个 pair mAP 的等权平均 = (26.53+16.07+5.61+15.66+16.83+5.25+5.85+5.25+13.52)/9 = **12.29%**；总体 mAP = **11.02%**。
- 九个 pair R1 的等权平均 = **19.12%**；总体 R1 = **29.71%**。

解释：
- pair mAP 只面对 1/3 gallery，负样本少，因此普遍高于总体；总体 mAP 约是 pair 平均的 0.90 倍。不能把 pair mAP 平均当作总体。
- 总体 R1 远高于 pair R1 平均，因为一个 query 的 top-1 可以来自任一 gallery 模态；pair R1 的“并集”抬高了总体。因此 **R1 的边际收益是次线性的，取决于“该 query 是否已经通过其他模态命中”**。

### 4.2 模态对的 mAP 权重

对总体 mAP 的一阶近似：

```
∂mAP_overall / ∂mAP_pair(qmod,gmod)
    ≈ [N_q(qmod) / N_q(total)] × [R_g(gmod) / R_g(total)]
```

其中 `R_g` 是该 gallery 模态中与 query 同身份的相关样本数（同相机排除后）。1000 三模态平衡，`R_g` 份额≈1/3；2337 按 gallery 估计份额 35.9%/35.8%/28.3%。

| 数据集 | RGB→RGB | RGB→NIR | RGB→TIR | NIR→RGB | NIR→NIR | NIR→TIR | TIR→RGB | TIR→NIR | TIR→TIR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 0.1111 | 0.1111 | 0.1111 | 0.1111 | 0.1111 | 0.1111 | 0.1111 | 0.1111 | 0.1111 |
| 1000GD | 0.1111 | 0.1111 | 0.1111 | 0.1111 | 0.1111 | 0.1111 | 0.1111 | 0.1111 | 0.1111 |
| 2337 | 0.1257 | 0.1255 | 0.0986 | 0.1262 | 0.1260 | 0.0990 | 0.1074 | 0.1072 | 0.0843 |

**结构性结论 [I]**

- 1000/1000GD：任意单对的 mAP 权重相同（1/9），但 **TIR 相关五对合计 5/9 = 55.6%**；非 TIR 四对合计 4/9 = 44.4%。
- 2337：TIR 相关五对合计 **49.7%**，非 TIR 四对合计 **50.3%**；单对权重最低的是 TIR→TIR（0.0843），最高的是 NIR→RGB（0.1262）。
- 注意：“TIR 相关五对”不是两对，而是 TIR 作为 query 的 3 对 + TIR 作为 gallery 的 3 对 − TIR→TIR 重复 1 对 = 5 对。TIR 在 query 和 gallery 两个维度都出现，所以它的结构权重高于其图像占比。

### 4.3 R1 的边际贡献

R1 不是 AP 求和，不能直接用权重相乘。定义：

- `p_pair`：某 pair 的 R1；
- `u`：总体 R1；
- `F = (u - mean(p_pair)) / (100 - mean(p_pair))`：在所有 pair 上平均而言，一个“在该 pair 错”的 query 仍能通过其他 gallery 模态命中的比例（重叠率）。

由本地基线计算：

| 数据集 | 九个 pair R1 均值 | 总体 R1 | 重叠率 F | 净新增率 1−F |
|---|---:|---:|---:|---:|
| 1000 | 19.12 | 29.71 | 0.131 | **0.869** |
| 1000GD | 19.44 | 31.61 | 0.151 | **0.849** |
| 2337 | 16.49 | 25.73 | 0.111 | **0.889** |

一阶近似：

```
单个 pair R1 提升 δ  →  总体 R1 提升 ≈ q_share(qmod) × (1-F) × δ
```

- 1000/1000GD：q_share=1/3，1−F≈0.85–0.87 → 单对每 +1pp，总体约 +0.28–0.29pp。
- 2337：RGB/NIR query q_share≈0.35，TIR≈0.299 → 单对每 +1pp，总体约 +0.27–0.31pp。

但这是“只改一对”的下界式估计；如果同一 query 模态的三对同时提升，且新命中的 query 有重叠，group 增益会小于 3 倍。因此第 4.4 节给区间而不是单点。

### 4.4 “overall +1.0pp”的条件表

以下阈值均为 **每个 pair 需要提升的 pp**，目标为总体 +1.0pp；若目标缺口是 1.1/1.3pp，按比例放大。

#### 1000 与 1000GD（query 三模态各 1/3，pair mAP 权重各 1/9）

| 改进策略 | 需要的 pair mAP 提升 | 需要的 pair R1 提升 | 评价 |
|---|---:|---:|---|
| 九对均匀提升 | **+1.0pp/对** | **+1.0–1.5pp/对** | 最现实；mAP 近似可加，R1 依赖并集重叠 |
| 一个 query 模态的三对（如 TIR→RGB/NIR/TIR） | **+3.0pp/对** | **+1.2–3.5pp/对** | mAP 只需 3 对各 3pp；R1 若三对同向，区间较宽 |
| 单个 pair | **+9.0pp** | **+3.2–3.5pp** | mAP 不现实；R1 因只影响 1/3 query 且净新增高，反而可行 |
| TIR 相关五对（TIR→RGB/NIR/TIR、RGB→TIR、NIR→TIR） | **+1.8pp/对（平均）** | **+1.2–2.0pp/对（平均）** | 结构权重 55.6%；最大杠杆 |
| 非 TIR 四对（RGB/NIR 之间） | **+2.25pp/对（平均）** | **+1.3–2.0pp/对（平均）** | 权重 44.4%；强模态更容易提升 |

#### 2337（query 份额 RGB 34.98%、NIR 35.12%、TIR 29.90%；pair 权重见 4.2）

| 改进策略 | 需要的 pair mAP 提升 | 需要的 pair R1 提升 | 评价 |
|---|---:|---:|---|
| 九对均匀提升 | **+1.0pp/对** | **+1.0–1.5pp/对** | 与 1000 相同的总量逻辑 |
| 一个 query 模态的三对 | RGB **+2.86**、NIR **+2.85**、TIR **+3.34** | RGB **+1.1–3.2**、NIR **+1.1–3.2**、TIR **+1.25–3.76** | TIR query 份额小，mAP 需要更高 |
| 单个 pair | **+7.9–11.9pp**（取决于 pair） | **+3.2–3.8pp** | TIR→TIR 最差（11.9），NIR→RGB 最好（7.9） |
| TIR 相关五对 | **+2.0pp/对（平均）** | **+1.1–2.0pp/对（平均）** | 结构权重 49.7% |
| 非 TIR 四对 | **+2.0pp/对（平均）** | **+1.2–2.0pp/对（平均）** | 权重 50.3% |

**读法**

- 若研究方案只能改一个模块、且该模块只显著提升一个 pair，那么它**几乎不可能**让总体 mAP +1pp（需要 +8–12pp 的单对 mAP）；但可能让总体 R1 +1pp（单对 R1 +3–4pp）。
- 若模块影响一个 query 模态的三对（例如 TIR 表征），mAP 需要每对 +3pp（1000/1000GD）或 +2.9–3.3pp（2337）；这是“中等难度”。
- 若模块影响 TIR 相关五对，mAP 平均只需 +1.8–2.0pp；这是最可行的结构路径。
- 若目标是“mAP 和 R1 同时 +1”，最稳妥的是均匀提升九对约 +1–1.5pp，或 TIR 五对约 +2pp。单对策略只能满足 R1，不能满足 mAP。

### 4.5 杠杆排序

综合 **mAP 权重、R1 净新增、当前难度、可达成性**：

| 优先级 | 模态对/簇 | mAP 权重（1000 / 2337） | 当前 R1（1000） | 目标缺口（R1/mAP） | 杠杆判断 |
|---|---|---:|---:|---:|---|
| P0 | **TIR 相关五对整体** | 55.6% / 49.7% | 6.4–20.8 | 1.2–2.6 / 1.4–1.6 | 结构权重最大；TIR 弱模态，提升空间大；一次 TIR 表征改进可同时抬五个 pair |
| P1 | **TIR→TIR** | 11.1% / 8.4% | 20.75 | +1.23 / +1.48 | 同模态 TIR 是瓶颈核心；R1 缺口相对小，但 mAP/mINP 仍差 |
| P1 | **TIR→RGB / RGB→TIR** | 各 11.1% / 9.9–10.7% | 7.40 / 7.49 | +2.64 / +1.61；mAP +1.47 / +1.39 | 跨模态最弱方向；R1 净新增高，单 pp 的总体 R1 杠杆大 |
| P2 | **NIR→NIR** | 11.1% / 12.6% | 27.96 | **+2.83 / +2.10** | 同模态最大缺口；强模态，通常比 TIR 容易提升 |
| P2 | **NIR→RGB / RGB→NIR** | 各 11.1% / 12.5–12.6% | 24.54 / 27.17 | +1.99 / +0.39（R1）；mAP +1.92 / +1.45 | 强模态交叉；RGB→NIR R1 已接近目标，重点在 mAP |
| P3 | **RGB→RGB** | 11.1% / 12.6% | 42.48 | R1 已达标；mAP +1.30 / mINP +1.30 | 只对 mAP/mINP 有贡献；R1 无需再优化 |
| P3 | **TIR→NIR / NIR→TIR** | 各 11.1% / 9.9–10.7% | 7.92 / 6.37 | +1.65 / +2.41（R1）；mAP +1.44 / +1.62 | 属于 TIR 簇；单独看 mAP 权重最低的是 2337 的 TIR→TIR，但 TIR 簇整体仍关键 |

**结论**：
- 按“总体 mAP 的每 pp 贡献”，1000 所有 pair 相同；2337 的 RGB/NIR 相关 pair 略高（0.126 vs TIR→TIR 0.084）。  
- 按“总体 R1 的每 pp 贡献”，硬 TIR 交叉 pair 最高（净新增率高），RGB→RGB 最低（大量 query 已通过其他模态命中）。  
- 按“一次改进覆盖的 pair 数”，**TIR 表征/对齐**是最大的结构杠杆：它同时影响 TIR query 的三对和 TIR gallery 的三对（去重后五对），合计权重 55.6%（1000/1000GD）/49.7%（2337）。  
- 因此推荐的组合是：**TIR 簇（P0/P1）+ NIR→NIR（P2）+ NIR↔RGB 的 mAP（P2）**；RGB→RGB 只作为 mAP/mINP 的补充，不作为 R1 目标。

---

## 5. 2337 与 1000GD 的可迁移性

### 5.1 1000GD 不是独立训练

- H015 加载 H014 的 1000 权重，只在 query/gallery 上应用 `FILTER_GD=True`；训练仍是 1000 的 500 个训练身份。[D] 02_Previous_Experiments 2.1；[R] 1000GD log L1（`TEST.WEIGHT logs/UAD_1000_DDP/transformer_120.pth`）。
- 因此 1000GD 是 **同一模型的子集重排**，不是跨数据集泛化测试；1000 的模型改进原则上会直接反映到 GD，除非改进对 ground/daytime 子集有偏。
- 1000GD 的基线比 1000 高（mAP 14.40 vs 11.02；R1 31.61 vs 29.71），主要因为 gallery 更小、负样本更少以及去掉了 aerial/night 困难样本；mAP 的差距（+3.38）明显大于 R1（+1.90），说明 mAP 对 gallery 规模很敏感。[I]
- 目标上，GD 的 R5/R10 已经超过“论文+1”；只需 mAP +1.10、R1 +1.29。由于 GD 的 mAP 基线更高、gallery 更小，1pp mAP 的绝对难度可能略低于 1000。[I]
- 但 GD 的 pair 结构与 1000 不同：GD 只在 ground 子集内重排，TIR→RGB 的 R1 几乎没变（7.62 vs 7.40），说明 GD 的“变容易”主要来自 RGB/NIR 同模态和跨模态 mAP，而不是 TIR 交叉。**TIR 仍是 GD 的瓶颈**，这与 1000 的 TIR 结论一致。[R]

**迁移建议**：先在 1000 上把 AS mAP/R1 做到 +1，再用同一权重跑 GD；如果 GD 的 mAP/R1 没跟上，检查改进是否只作用于 aerial/night（GD 中没有），或只作用于 RGB/NIR 强模态。GD 不需要单独训练；也不要把 GD 当成第二个独立验证集来调参。

### 5.2 2337 是独立训练

- H020 在 2337 上独立训练 120 epoch，DDP 4×，batch 256/LR 0.032，seed 1234。[D] 02_Previous_Experiments 2.1.5；[R] 2337 train log。
- 数据差异：训练身份 1,000（1000 为 500）、测试身份 1,337（1000 为 500）、TIR 图像少约 21%、query 15,020（1000 为 6,405）。[P] Table 1/2。
- 基线 9.60/25.73/40.90/48.39，目标 10.6/26.7/41.9/49.2；mAP/R1 缺口约 1pp，和 1000 的绝对缺口相近，但相对缺口更大（mAP +10.4%，R1 +3.8%；1000 为 +9.1%/+2.7%）。[I]
- 2337 的 AG/3×3 本地值明显低于 1000（例如 A→G R1 26.29 vs 31.63；TIR→TIR R1 18.83 vs 20.75），说明更大的身份空间和 TIR 稀缺使跨平台/跨模态更难。[R]

**最可能跨数据集成立的改动**

1. **TIR 表征/对齐**：1000 和 2337 的 TIR 相关 pair 都是弱项；2337 的 TIR 更少，改进的边际收益可能更大。TIR→TIR、TIR→RGB、RGB→TIR 的本地缺口在 2337 上更大（相对 1000 论文目标约 2.3–2.8pp mAP、2.9–3.6pp R1）。[I]
2. **相机感知负样本/训练—评测对齐**：当前 WRT 把所有异身份当负样本，包括评测会排除的同相机异身份；这是一个与数据集规模无关的目标错配。若改成相机感知的负样本 mask 或相机平衡采样，理论上可同时作用于 1000/2337；但需要实验验证同相机负样本是否真的有害。[I]
3. **原型/中心可靠性**：GPD bank 在 2337 上有 1,000 类（1000 为 500 类），原型更多、初始化更慢；对原型归一化、EMA 或冷启动的改进可能在 2337 上收益更大，但依赖类数的超参（温度、momentum、epoch）需要重新调。[I]
4. **模态均衡采样**：2337 的 TIR 更少，PKM 的重复采样更强；改进 sampler（减少重复、按可用样本自适应）可能在 2337 上更明显，但对平衡的 1000 影响较小。[I]
5. **与类别数无关的损失几何/归一化**：例如训练距离与推理归一化的一致性、特征范数约束；这些不依赖 500/1,000 类，迁移性较好。

**可能不迁移的改动**

- 只在 1000 的 500 个测试身份上提 mAP、但 R1 下降的模块（历史 HiHR、ViT-L、ViSA 等）；它们本身没有实现协同提升，跨数据集更难。
- 依赖 500 类 GPD bank 动态或 1000 数据 batch 组成的超参；到 2337 需要重新调。
- 依赖 aerial/night 样本比例的改动；1000GD 没有 aerial，2337 的 day/night 比例也不同。
- 只在 256×128 上有效的局部/分辨率相关模块；如果 2337 仍用 256×128，这部分可迁移，但 VI 的 288×144 目标不能直接套用。

### 5.3 迁移优先级

1. 先补齐协议：总体 mINP、1000GD day/night 清单、有效 query 计数；否则目标不可验证。
2. 在 1000 上做 TIR 簇 + NIR→NIR；用同一权重直接跑 1000GD（无需重训），看 GD mAP/R1 是否同步。
3. 任何声称 +1 的改动，必须在 2337 上按 H020 recipe 重训一次，报告 AS + AG；不要用 1000 的结果推断 2337。
4. 1000GD 只作为“同一模型的子集稳健性”检查，不作为独立数据集调参；2337 才是真正的跨数据集验证。

---

## 6. 协议层面的风险与陷阱

### 6.1 训练使用同相机负样本，评测排除同相机：是否存在错配？

**代码事实 [C]**

- `loss/wrt_loss.py` L46–L48：
  ```python
  is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
  is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()
  ```
  只按 PID 判断正负，**没有 camera mask**；因此同相机、异身份的样本是负样本，同相机、同身份的不同图片是正样本。
- CE 和 GPD 也是全局身份分类/原型对比，不区分相机。[C] 01 文档 Sec. 5。
- 评测 `eval_func` 排除所有同相机 gallery（L164），包括同相机异身份负样本和同相机同身份正样本。

**是否存在错配？**

是，存在两层错配 [I]：

1. **负样本错配**：训练花容量分离同相机异身份 pair，但评测把这些 pair 从 gallery 中删掉。模型可能学到“同相机内的细粒度区分”，而这在评测中不被奖励；更糟的是，如果同相机异身份样本在训练中占主导，模型可能把相机/背景线索当作身份线索，损害跨相机泛化。
2. **正样本错配**：同相机同身份的正样本在训练中被拉近，但评测永远不考同相机正样本；模型可能利用同相机的背景、光照、姿态连续性等捷径。这些捷径在跨相机 query 上可能失效。

**但不应夸大** [I]：

- 同相机负样本也能提供有用的判别信号；完全去掉可能降低训练难度。
- WRT 是 per-GPU batch 内的关系（DDP 下不是全局），同相机 pair 在 batch 中是否足够多取决于采样；实际影响需要统计每个 batch 的同相机负样本比例。
- 这是一个“目标分布错配”，不是“协议 bug”；论文本身也用同样的训练/评测组合，并且复现数值接近，说明它不是当前 1pp 缺口的主因，但可能是跨相机 R1 的潜在瓶颈。

**可验证的实验（不修改现有代码，仅作为研究建议）**：统计每个 batch 中同相机正/负 pair 的比例；做 camera-aware negative mask 或 camera-balanced sampler 的对照，观察跨相机 R1/mAP 与同相机 pair 准确率的变化。

### 6.2 采样器对少样本模态重复采样，对 TIR 的影响

**代码事实 [C]**

`datasets/sampler.py` PKMSampler：

- L94–L102：对每个 PID，`max_num = max_m(count_m)`，向下取整到 `NUM_INSTANCE` 的倍数；`pid_max_count[pid] = max_num`。
- L109–L120：
  - `need = pid_max_count[pid]`；
  - 若某模态样本数 `cur >= need`，无放回采样 `need` 个；
  - 若 `cur < need`，`k, r = divmod(need, cur)`，把该模态列表重复 `k` 次，再随机补 `r` 个；
  - 结果每个模态都被对齐到 `need`，少数模态被重复。
- 一个 sampler index 是一个三模态 tuple；`IMS_PER_BATCH` 是 tuple 数，每 tuple 3 张图。[C] 01 文档 Sec. 4.1；02 文档 1.3。

**对 TIR 的影响 [I]**

- 1000 三模态基本平衡，重复较少；2337 的 TIR 总量少约 21%，且某些 PID 的 TIR 可能远少于 RGB/NIR，因此 TIR 的重复倍率更高。
- 正面：保证每个 batch 都有三模态，避免 TIR 缺失；让 TIR 参与每次 ProCA/GPD 中心计算。
- 负面：
  - TIR 的有效样本多样性下降，模型可能记住重复的 TIR 图像，过拟合；
  - TIR 的实际训练频率高于其自然占比，可能让 TIR 指标“看起来”提升，但不代表对未见 TIR 的泛化；
  - 2337 与 1000 的 TIR 重复率不同，所以 1000 上的 TIR 改进幅度不能直接外推到 2337；
  - `cur=0` 时 `divmod(need, 0)` 会除零；当前训练身份可能都覆盖三模态，但代码没有保护。[C]
- 需要统计：每个 PID 的 TIR 重复倍率、每个 epoch 中 TIR 唯一图像数/总采样数；目前日志没有这些统计。[U]

### 6.3 论文 Table 5 的 288×144 与本地 256×128

**事实**

- 论文 task-specific settings：VI-ReID 输入 144×288，其他任务 128×256；并明确“All images are resized to 144×288”用于 VI。[P] Sec. 5.1。
- 本地 `configs/UAD.yml` L16–L17：`SIZE_TRAIN: [256, 128]`、`SIZE_TEST: [256, 128]`（H×W），即 256 高 × 128 宽；AS/AG 与论文的 128×256（宽×高）一致，VI 不一致。[C][P]
- ViT patch 16：本地 256×128 → 16×8 = 128 patch + CLS = 129 token；288×144 → 18×9 = 162 patch + CLS = 163 token。位置编码需要插值，token 数和宽高比都不同。[I]

**对目标比较的影响 [I][U]**

- 本地 Table 5 的 NIR→RGB/TIR→RGB 数值来自同一个 256×128 的九组合运行，不是单独的 288×144 VI 运行；论文 Table 5 的 UAD 数值与 Table 6 的对应 cell 完全一致，说明论文 UAD 行可能也复用了 Table 6（或至少没有体现 288×144 的差异）。因此：
  - 如果论文确实用了 288×144，本地 256×128 的 VI 基线偏低 1–2pp 是合理的，目标缺口有一部分来自分辨率，不应全部归因于方法；
  - 如果论文 UAD 行实际来自 Table 6 的 256×128 运行，那么分辨率差异不是主因，本地缺口更可能是训练随机性或实现细节。
- 建议：在不重训的前提下，用 H014 权重做一次 test-time 288×144 推理（如果模型位置编码支持插值），对比同一权重的 VI 指标；这能把“分辨率”与“方法”分离。若不支持，至少记录该差异，不把 Table 5 当作严格可比目标。
- AS/AG 的目标不受此影响（本地 256×128 = 论文 128×256）。

### 6.4 mINP 缺失对目标口径的影响

- 论文只对 VI/3×3 报告 mINP；AS/AG 没有 mINP。[P] Sec. 5.1。
- 本地总体 AS/AG 也不报告 mINP（2.5）。因此：
  - **不能为 AS/AG 的总体 mINP 设定“论文+1”目标**；如果研究目标要求 mINP，必须先建立本地基线（例如 H014 的总体 mINP），再定义“baseline+1”的次级目标，并明确它不是论文口径。
  - 3×3/VI 的 mINP 有论文目标，但本地实现公式与 AGW 标准 mINP 可能不同（2.5），需要先核对公式。
- 风险：如果为了“全指标 +1”而优化 mINP，可能会与 mAP/R1 冲突（mINP 更关注最后一个相关样本的位置，可能牺牲 top-1）。论文没有给 AS/AG mINP 目标，也没有证明 mINP 与 mAP/R1 同向，因此 mINP 应作为次要/诊断指标，而不是主目标。

### 6.5 其他协议陷阱

1. **1000GD 的 daytime 未显式实现**（2.4）：这是最大的单点协议风险。若本地 GD 含夜间地面图，则基线比论文更難，目标缺口可能被高估或低估；需要数据清单核对。
2. **AG 依赖 camid 顺序与 1-based 编号**（2.3）：代码没有 platform 字段；若相机编号不是 c1–c5 ground / c6–c7 aerial，AG 会切错。日志能跑出非空结果，说明当前数据至少满足“有 <5 和 >=5 的相机”，但不能证明语义正确。
3. **九组合不含平台**（2.2）：Table 6 是 all-scenario 的 modality pair；不要把它和 AG 的 ground/aerial 结果混在一起。1000GD 的九组合只在 ground 内，不能与 1000 的九组合直接比较“平台影响”。
4. **有效 query 数未记录**（2.1）：总体/九组合/AG 的分母可能不同；pair 指标与总体指标不能严格还原。若要精确加权，需要打开 `DETAILED_DOMAIN_EVAL` 或增加有效 query 日志（这里只建议，不修改代码）。
5. **query/gallery 可能有重复文件**：论文说 query 从 gallery pool 抽 1–3 张/身份/相机；本地代码分别读 `query/` 与 `test/`，不做 hash 去重。若 query 图像仍留在 test/，同相机排除会删掉它（因为 camid 相同），所以 AS 不会因自匹配虚高；但若存在跨相机复制的同一图像，仍可能虚高。AG 的 q/g 平台互斥，不受同相机保护；需要数据清单核对。[U]
6. **论文值单次运行，无 seed 方差**：本地 H014 与论文 Table 3 的差异在 ±0.3pp 内，但论文没有报告多 seed 均值/方差；+1pp 的目标可能接近某些指标的运行噪声。历史模块的 mAP 增益常在 +0.1–0.5pp（Table 8 的 ProCA +0.1 mAP），因此 +1pp 是明显高于常见模块效应的目标，但仍建议多 seed 验证。[P][D]
7. **训练 LR 语义**：论文写 “base LR 0.008 per GPU”，本地 README 四卡用 0.032（4× 线性放大），H014 日志 `BASE_LR: 0.008` 但实际命令行可能覆盖为 0.032；02 文档记录 H014 为 0.032。由于 H014 与论文数值几乎一致，本文把 H014 当作有效复现；但若要做严格的超参归因，需以实际命令为准。[D][R]
8. **mINP 的 top-k 分支缺失**：若未来启用 `TOP_K_EVAL>0`，`eval_func_top_k` 完全不计算 mINP，mINP 目标会再次缺失；当前 TOP_K_EVAL=0，不受影响，但改配置时要注意。[C]
9. **top-k 截断的 AP 近似**：`eval_func_top_k` 用完整 gallery 的 `num_rel` 作 AP 分母，却只对 top-k 结果求和（L90–L125），属于截断近似；当前 `TOP_K_EVAL=0` 不受影响，但若为加速启用 top-k，mAP 会偏低，不能与论文/历史全 gallery 数值直接比较。[C]

---

## 7. 建议的评测与目标跟踪清单（不修改代码）

1. **先补总体 mINP**：在总体 `eval_func` 调用处取回 `mINP`（`return_minp=True`），并在日志/CSV 中报告；同时记录有效 query 数。否则 AS/AG 的 mINP 目标无法验证。
2. **核对 1000GD 的 day/night**：从数据 manifest 统计 query/test 中 ground 图像的昼夜分布，确认 399/模态的构成；若本地 GD 含夜间图，明确标注“本地 GD ≠ 论文 GD”。
3. **核对 mINP 公式**：用 AGW 官方实现或论文补充材料确认 `mINP = rank_last/N` 还是 `num_rel/rank_last`；在确认前，mINP 只作参考。
4. **VI 分辨率对照**：用 H014 权重做一次 288×144 test-time 推理，比较同一权重的 NIR→RGB/TIR→RGB；若差异主要来自分辨率，Table 5 的目标应降级为“参考”。
5. **目标表使用有符号缺口**：`baseline_to_target ≤ 0` 表示已达标（1000GD R5/R10、3×3 RGB→RGB R1）；不要把这些 cell 再算作“需要提升”。
6. **按杠杆排实验**：优先 TIR 簇（五对）和 NIR→NIR；RGB→RGB 只补 mAP/mINP。若要 overall mAP +1，按“九对各 +1”或“一个 query 模态三对各 +3”设计；不要期望单对 mAP +9。
7. **1000GD 用同一权重直接重跑**；2337 必须重训。任何 +1 结论至少要在 1000 和 2337 上各验证一次，1000GD 作为子集稳健性检查。
8. **多 seed**：至少 2–3 个 seed 估计 +1pp 是否超出噪声；论文本身没有方差，历史模块效应也多在 ±0.5pp。

---

## 8. 来源索引

### 8.1 代码

| 文件 | 行 | 内容 |
|---|---|---|
| `WHU\utils\metrics.py` | 90 | `valid_gallery = (g_pids == q_pid) & (g_camids != q_camid)` |
| 同上 | 99–100 | 同相机排除（sysu 规则） |
| 同上 | 163–165 | `eval_func` 同相机排除 |
| 同上 | 191–192 | mINP = `num_rel/(last_relevant+1)` |
| 同上 | 196–199 | CMC/mAP/mINP 平均 |
| 同上 | 201–203 | `return_minp` |
| 同上 | 215–256 | 36 条域协议定义 |
| 同上 | 259–260 | platform mask `camid<5`/`>=5` |
| 同上 | 268–324 | 详细域评测（含 overall mINP 空值） |
| 同上 | 480–488 | 总体评测，未取回 mINP |
| 同上 | 491–565 | 九组合 3×3 |
| 同上 | 543–551 | Table 5 (NIR/TIR→RGB) |
| 同上 | 569–613 | AG-ReID |
| `WHU\datasets\whu_mars.py` | 66 | 文件名正则 `(\d+)_c(\d+)` |
| 同上 | 75、90 | `pid 1000–1999, camid<=5` 过滤 |
| 同上 | 89 | 注释 “ground + daytime only” |
| 同上 | 92 | `camid -= 1` |
| 同上 | 36–38 | train 不做过滤，query/gallery 过滤 |
| `WHU\datasets\make_dataloader.py` | 116–127 | query+gallery 拼接、num_query |
| `WHU\datasets\sampler.py` | 94–120 | PKM 少数模态重复采样 |
| `WHU\loss\wrt_loss.py` | 46–48 | 仅按 PID 定义正负样本 |
| `WHU\processor\processor.py` | 328–340 | evaluator 构造（top_k=cfg.TEST.TOP_K_EVAL） |
| 同上 | 348–365 | 推理与结果打印 |
| `WHU\config\defaults.py` | 193 | `TOP_K_EVAL=0` |
| 同上 | 199 | `DETAILED_DOMAIN_EVAL=False` |
| `WHU\configs\UAD.yml` | 16–17 | 输入 256×128 |
| 同上 | 53–54 | NECK_FEAT before、FEAT_NORM yes |

### 8.2 日志

| 日志 | 行 | 内容 |
|---|---|---|
| `logs\1000\1000_test_log_20260912_1037.txt` | 148–150 | query 2135/模态，总 6405 |
| 同上 | 154–162 | 九组合 R1/mAP/mINP |
| 同上 | 164–176 | Table 5/6 打印 |
| 同上 | 178–180 | AG A→G / G→A |
| 同上 | 182–185 | 总体 mAP/R1/R5/R10 |
| `logs\1000GD\1000GD_test_log_20260912_1045.txt` | 148–150 | query 399/模态，总 1197 |
| 同上 | 154–162 | 九组合 |
| 同上 | 178 | AG skipped |
| 同上 | 180–183 | 总体 |
| `logs\2337\2337_test_log_20261025_1505.txt` | 149–151 | query 5254/5275/4491，总 15020 |
| 同上 | 155–163 | 九组合 |
| 同上 | 180–181 | AG |
| 同上 | 183–186 | 总体 |
| `logs\1000\1000_train_log_20260911_2145.txt` | 1094–1102 | H014 九组合 |
| 同上 | 1119–1120 | H014 AG |
| 同上 | 1121–1125 | H014 总体 |
| `logs\2337\2337_train_log_20260912_1833.txt` | 1814–1822 | H020 九组合 |
| 同上 | 1839–1840 | H020 AG |
| 同上 | 1841–1845 | H020 总体 |
| `logs\1000GD\1000GD_test_log_20260912_1045.txt` | 179–183 | H015 总体 |

### 8.3 CSV 行

- 论文：R0001（1000GD AS）、R0002（1000 AS）、R0003（2337 AS）、R0004–R0005（AG）、R0006–R0014（3×3/VI）。
- H014：R0159（AS）、R0160–R0168（3×3）、R0169–R0170（AG）。
- H015：R0171（AS）、R0172–R0180（3×3）。
- H020：R0229（AS）、R0230–R0238（3×3）、R0239–R0240（AG）。

### 8.4 论文

- Sec. 3.3、Eq.(2)：同相机排除。
- Table 1/2：数据集统计、query/gallery 数量。
- Table 3：AS-ReID（1000GD/1000/2337）。
- Table 4：AG-ReID（1000）。
- Table 5：VI-ReID（1000，144×288）。
- Table 6：3×3 modality pair。
- Sec. 5.1：mINP 只用于 VI-ReID。

---

## 9. 未决项

1. **1000GD 的 day/night 字段与 399 query 的构成**：需要数据 manifest；当前代码没有 day/night 解析。[U]
2. **mINP 公式是否与 AGW [47] 一致**：需要 AGW 官方实现/补充材料核对。[U]
3. **1000GD 与 2337 的精确 gallery 数**：日志未打印；本文用论文 Table 1/2 和平衡假设推算，误差可能影响权重的小数点，但不改变“TIR 五对≈50–56%”的结构结论。[U]
4. **AG 的 camid 语义**：需要确认原始 c1–c5/c6–c7 与 ground/aerial 的对应；代码只靠阈值。[U]
5. **有效 query 数**：未记录；总体/九组合分母可能不同，本文的权重和阈值是一阶近似。[U]
6. **多 seed 方差**：论文与本地正式结果均为单次运行；+1pp 的显著性未知。[U]
7. **query/test 是否有跨相机重复文件**：需要 hash 清单；同相机排除只能防止同相机自匹配。[U]

---

## 10. 一句话总结

本地评测协议在“同相机排除、九组合、AG 划分”上与论文基本一致，但 **1000GD 的 daytime 未在代码中实现、总体 AS/AG 的 mINP 未报告、mINP 公式与标准 AGW 可能不同、训练仍使用评测会排除的同相机负样本**。按论文+1.0pp 口径，AS 的缺口约 0.8–1.3pp，VI/3×3 的 R1 缺口约 1.6–2.8pp；要达到 overall +1pp，最现实的路径是 **TIR 相关五对平均 +1.8–2.0pp + NIR→NIR/跨 RGB-NIR 的 mAP 提升**，而不是追求单个 pair 的 +9pp mAP。1000GD 可用同一权重直接重跑；2337 必须独立重训验证。完整数值见 `target_table.csv`。
