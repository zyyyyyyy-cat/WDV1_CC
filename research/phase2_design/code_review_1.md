# 代码审查 #1（自审）

> 说明：原计划由独立 agent 审查；两个独立审查 agent 在本环境均卡住未返回，已停止。本报告是主线程的自审记录，不是独立第三方审查。审查对象：`WDV1_CC/WHU` 的 LRA-UAD 修改。

## 1. 结论

- **P0：0 个**。
- **P1：0 个**（在修复 `make_loss` 的 triplet 分支、`check_checkpoint_load` 的 wrapper、`dump_features` 的 PRETRAIN 依赖、`fuse_features` 空子集、`ConstantLRScheduler` 的 bias LR 之后）。
- 代码可进入服务器实验；但 DDP 暖启动、TTA、CA-WRT 的真实 GPU 行为尚未实测。

## 2. 已验证项

| 项 | 结果 |
|---|---|
| `python -m compileall -q config configs datasets loss model processor solver utils tools train.py test.py` | PASS |
| 4 个新 YAML 配置 merge/freeze | PASS |
| 真实 `transformer_120.pth` finetune 加载 | 158/158 tensors，missing/unexpected/shape mismatch 均 0，逐元素最大差 0.000e+00 |
| CPU 模型前向（三模态，mode=0） | `cls_score [6,10]`、`global_feat [6,768]`、`feat [6,768]` |
| WRT camera mask + norm blend backward | loss 有限，`feat.grad.norm()≈0.6` |
| `make_loss` WRT 分支 + camids/epoch | loss/ID/tri 有限，backward 正常 |
| PrototypeBank update/initialize | 非零原型数正确 |
| GPD warm start（3 模态假 loader） | 3/3 身份，counts=1，原型非零 |
| ModelEma update | EMA delta 有限（~3.7e-9） |
| LLRD optimizer groups | 154 groups，min/max LR 正常 |
| ConstantLRScheduler | 所有 group LR 固定 6.4e-5 |
| model_soup（合成 2 checkpoint） | 浮点平均正确，int 取第一个 |
| fuse_features（合成 npz，score/RRF） | 运行、输出 JSON |
| paired_bootstrap（合成 report） | 运行、输出 CI |

## 3. 逐项结论

### 3.1 原 UAD 默认路径

所有新开关默认关闭：
- `WRT_CAMERA=False` → `pos_mask=is_pos`、`neg_mask=is_neg`，WRT 与原实现一致。
- `WRT_NORM_BETA=0` → 不进入 norm blend。
- `GPD_WARMSTART=False` → 不跑暖启动。
- `GPD_COUNT_EMA=False` → `PrototypeBank.update` 走固定 `m=0.2` 分支。
- `GPD_TAU_INIT=FINAL=0.03`、`WARM=0` → tau=0.03。
- `EMA=False` → 不创建 EMA。
- `TTA_FLIP=False` → `do_inference` 只跑一次前向。
- `LLRD_DECAY=0` → `_llrd_scale` 返回 1.0。
- `SCHEDULER='cosine'`、`LR_MIN_FACTOR=0.002` → 原 cosine floor。
- `do_train` 新增 `train_normal_loaders` 只被暖启动使用；默认不迭代。
- `make_dataloader` 多返回 3 个确定性 loader；默认不影响训练 batch。

**结论**：默认路径与基线逐损失/逐优化器一致；未发现数值路径改动。

### 3.2 finetune 加载

- 在 classifier/bottleneck 创建后加载；包含 backbone + BNNeck + classifier。
- 支持 `state_dict`/`model` wrapper、`module.` 前缀、shape mismatch 跳过并打印。
- 真实 `transformer_120.pth`：158/158，0 差异。
- 自检工具 `tools/check_checkpoint_load.py` 可复现 PASS。

### 3.3 DataLoader/do_train 调用

- `make_dataloader` 返回 `train_loader, train_normal_loaders, val_loaders, ...`。
- `train.py` 同步改名并传入 `do_train`。
- `test.py` 第二返回值改名 `train_normal_loaders`，不使用；不报错。
- `tools/dump_features.py` 同步 unpack。
- `do_train` 位置参数与 `train.py` 调用一致。

### 3.4 DDP 暖启动

- 所有 rank 都进入 `if cfg.MODEL.GPD_WARMSTART` 分支（同一 config）。
- rank0 计算；其他 rank 先到 `dist.barrier()` 等待；rank0 完成后所有 rank 一起 `broadcast`；无 rank 提前进入训练。
- `prototypes`/`counts` 均为 float32，NCCL broadcast 兼容。
- 非 DDP 路径直接计算。
- **未实测**：4 卡真实数据、NCCL timeout、DataLoader worker 与 rank0 的交互。

### 3.5 GPD 暖启动

- `model.eval()` + `@torch.no_grad()`；不更新模型/BN。
- `batch[1]` 是 `val_collate_fn` 的 pid tuple；`torch.as_tensor` 正确。
- `model(img, mode=mode)` 返回 pre-BN CLS（`NECK_FEAT='before'`）；`feat.float()` + L2。
- `index_add_` 维度正确；`counts=1` 防止 count EMA 首次覆盖（默认 count EMA 关闭）。
- 结束时 `model.train()`。
- **注意**：类别中心按样本数加权；1000 三模态平衡，2337 TIR 较少时会偏 RGB/NIR。这是有意为之的先验，不是 bug。

### 3.6 CA-WRT

- `camids_cat = cat(camids)` 与 `[RGB;NIR;TIR]` 特征顺序一致。
- `pos_mask=is_pos`（正样本不变）；`neg_mask` 同相机负样本乘 0.25；全零行回退 `is_neg`。
- 对角线保持原 WRT 行为（正样本、距离≈0、权重低）。
- mask 不接收梯度；不会 NaN。
- **未实测**：真实 batch 的同相机负样本比例与指标影响；先跑 `camera_pair_stats.py`。

### 3.7 EMA

- `ModelEma` 浮点 EMA、整数 buffer 复制；`eval()` 副本。
- 每 step 更新；所有 rank 各自维护；rank0 保存 `transformer_ema.pth`。
- 键与训练模型一致；`test.py` 的 `load_param` 会跳过 classifier，但保留 bottleneck/backbone；检索特征不依赖 classifier。
- **残余风险**：EMA 额外显存约 344MB/GPU；A0 默认开启，若 OOM 关闭。

### 3.8 TTA

- `torch.flip(img, dims=[3])` 对 `[B,3,H,W]` 翻转宽度；正确。
- 每支 `F.normalize` 后相加；evaluator 再 L2；与文档一致。
- **未实测**：真实 TIR/航拍的翻转收益；按 gate 四项全 ≥+0.15 才保留。

### 3.9 工具

- `check_checkpoint_load`：cfg merge/freeze 正常；真实 checkpoint PASS。
- `model_soup`：keys/dtype 检查；合成测试 PASS。
- `dump_features`：覆盖 `PRETRAIN_CHOICE='none'`，不依赖 ImageNet 路径；q/g 切分按 `num_query`；键 `<modality>/<basename>`。
- `fuse_features`：按 keys 对齐；同相机排除；空子集返回 NaN 而不崩；score/RRF/concat 三种模式；合成测试通过。
- `camera_pair_stats`：metadata-only，对角线排除；同相机正/负/跨模态正统计。
- `paired_bootstrap`：per-query 配对；合成测试通过。

## 4. 残余风险（不构成 P0/P1，但需服务器确认）

1. **DDP 暖启动未实测**：NCCL broadcast、barrier、DataLoader worker、rank0 单卡前向时长。
2. **A0 真实指标未验证**：本机无 GPU/数据；A0 是否能把 mAP 推过 +0.25 未知。
3. **TTA 未实测**：可能对 TIR/航拍有副作用。
4. **CA-WRT 效果未知**：先统计同相机负样本比例；若 <5%，不建议开。
5. **EMA 显存**：A0 默认开启；OOM 时关闭。
6. **1000GD daytime / mINP 公式**：协议未核实；GD R5/R10 作为护栏。
7. **CLIP 特征导出**：本仓库只提供 UAD 侧和融合工具；CLIP 侧需在 WC/CLIP 代码中加保存分支。
8. **PMC/MSIN 未落地**：仅文档设计，未改代码。

## 5. 建议的最小服务器自检

```bash
cd WHU
python -m compileall -q config configs datasets loss model processor solver utils tools train.py test.py
python tools/check_checkpoint_load.py \
  --config_file configs/LRA_UAD_A0.yml \
  --checkpoint /path/to/UAD_1000_DDP/transformer_120.pth \
  --num_classes 500
# 期望：158/158，0 差异，PASS

# 1 个真实 batch 的 smoke（建议临时加打印或用一个最小脚本）：
# - loss 有限
# - GPD loss 明显低于 5
# - bank 非零覆盖率接近 500/500
# - DDP 各 rank 的 bank 一致
```

## 6. 结论

**代码可以进入服务器实验。** 先跑 A0；A0 通过后再逐个加 GPD 暖启动（默认已开）、CA-WRT、EMA、TTA。若 A0 的 mAP 仍差 0.3–0.5，直接做 UAD+CLIP 融合。
