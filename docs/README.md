# LRA-UAD 交付说明

> **最新分析以 `research/phase3_analysis/04_synthesis.md` 为准**：A0 是空对照，平台不是主瓶颈，当前唯一主候选是 GPD-SC（K=3 模态子中心原型）。本目录的 01/02 是 Phase 2 旧版记录。

## 交付物

| 文件 | 内容 |
|---|---|
| `docs/01_方案设计说明.md` | 诊断依据、方案全貌、组件设计、数据流/梯度流、参数、实验矩阵、预期与风险 |
| `docs/02_代码修改方案.md` | 逐文件代码修改、归一化顺序、论文-代码差异、运行命令、自检与限制 |
| `WHU/configs/LRA_UAD_A0.yml` | 首选：从 H014 120 epoch 以终段 LR 继续 30 epoch |
| `WHU/configs/LRA_UAD_A2.yml` | 备选：200 epoch 从头训练 |
| `WHU/configs/LRA_UAD_full.yml` | A0 + GPD 暖启动 + CA-WRT + EMA + flip TTA（组合运行） |
| `WHU/tools/*.py` | 加载自检、model soup、特征导出、UAD+CLIP 融合、相机 pair 统计、paired bootstrap |

## 建议执行顺序

1. **加载自检**（服务器）：
   ```bash
   cd WHU
   python tools/check_checkpoint_load.py \
     --config_file configs/LRA_UAD_A0.yml \
     --checkpoint /path/to/UAD_1000_DDP/transformer_120.pth \
     --num_classes 500
   ```
   必须 PASS。

2. **S0 免费诊断**：
   - `python tools/camera_pair_stats.py --config_file configs/UAD.yml --batches 20`
   - 用 H014 权重跑一次 `test.py`，`TEST.TTA_FLIP True`，看四项是否都不降、至少一项 +0.2。
   - 用 `tools/dump_features.py` 导出 UAD 特征；用 WC/CLIP 导出 CLIP 特征；用 `tools/fuse_features.py` 做 α=0.5 / RRF 融合，先回答 1000 AS 的上限。

3. **S1a A0（首选）**：
   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py \
     --config_file configs/LRA_UAD_A0.yml \
     MODEL.DIST_TRAIN True \
     MODEL.PRETRAIN_PATH /path/to/UAD_1000_DDP/transformer_120.pth \
     DATASETS.ROOT_DIR /path/to/data \
     SOLVER.IMS_PER_BATCH 256 SOLVER.BASE_LR 0.000064 \
     OUTPUT_DIR logs/LRA_UAD_A0_DDP
   ```
   Gate：mAP ≥ +0.25、R1 ≥ +0.4、R5/R10 ≥ +0.2，且无指标下降；GD R5/R10 不低于 46.0/53.5。

4. **按单变量依次加**：GPD 暖启动（A0 默认已开）→ CA-WRT（先统计）→ EMA → soup → TTA。
   不要一次开多个组件；每次只改一个变量。

5. **A2**：若 A0 的 mAP 仍差，跑 `configs/LRA_UAD_A2.yml`（200 epoch 从头）。
   Gate：mAP ≥ +0.5、R1 ≥ +0.8。
   A0/A2 都失败后，才考虑 `configs/LRA_UAD_A1.yml`（第二段 cosine，峰值 5e-4，60 epoch）。

6. **1000GD / 2337 / AG**：同一 1000 权重跑 GD；2337 独立重训；AG 在同一次评测中输出。

7. **Plan B**：若单模型 mAP 仍差 0.3–0.5，用 UAD+CLIP 融合作为 1000 AS 的数值交付；明确披露双模型。

## 重要提醒

- 不要再用新增排序/对比/混合 gallery 损失；历史和最新实验已反复证明会抬高最危险负样本。
- 不要用测试集挑 epoch / EMA / soup / 融合 α；预注册。
- 1pp Rank-1 ≈ 64 个 query；至少 2 seed + paired bootstrap。
- 1000GD R5/R10 只有 0.14/0.06pp 余量，作为护栏。
- 1000GD daytime 未在代码中实现；先做 manifest 审计。
- AS 总体 mINP 没有论文目标；mINP 只对 VI/3×3 设目标。
