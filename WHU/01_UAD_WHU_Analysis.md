# 01_UAD_WHU_Analysis

> **文档定位：** WHU-MARS / UAD 原论文与官方开源实现的长期研究底稿。本文只分析原任务、原方法、代码执行机制、实验现象、已存在的局限和潜在研究切入点；**不提出具体改进网络、模块拼接方案或代码修改计划**。
>
> **版本：** v1.0  
> **分析日期：** 2026-10-25  
> **论文来源：** 用户上传的 `WHU(4).pdf`，CVPR 2026 主论文，共 11 页；[CVF 官方页面](https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_WHU-MARS_A_Multispectral_Aerial-Ground_Benchmark_Towards_Any-Scenario_Person_Re-Identification_CVPR_2026_paper.html)。  
> **指导来源：** 用户上传的 `AGENTS(1).md` 与 `Research_Guidelines(2).md`。  
> **官方代码快照：** [`msm8976/WHU-MARS`](https://github.com/msm8976/WHU-MARS)，分支 `main`，读取时最新提交为 [`3e2a07314119402586c76096e87d40420894ad30`](https://github.com/msm8976/WHU-MARS/commit/3e2a07314119402586c76096e87d40420894ad30)（提交日期 2026-08-05）。

------------------------------------------------------------------------

## 0. 如何使用本文档

### 0.1 证据等级

为避免把合理猜测写成确定事实，全文采用以下标记：

- **\[P\] 论文事实：** 来自论文正文、公式、图或表。
- **\[C\] 代码事实：** 来自公开仓库当前代码、配置或运行入口。
- **\[R\] 结果证据：** 来自论文报告的定量结果或消融结果。
- **\[I\] 机制推论：** 由论文事实、代码路径或结果支持的解释，但仍不等于因果已被单独实验证实。
- **\[U\] 未决项：** 现有资料不足以确定，需要数据清单、实际运行日志、梯度统计或补充材料才能确认。

### 0.2 阅读顺序

- **快速恢复：** 先读第 1 节的核心结论、第 2 节任务与数据、第 4–6 节数据流和训练机制。
- **设计损失或模块前：** 重点读第 5 节损失与梯度、第 7 节推理评测、第 11–12 节薄弱点和切入点。
- **复现实验前：** 重点读第 6 节训练配置、第 9 节图表结果、第 10 节论文—代码核对。
- **阅读具体代码时：** 使用第 13 节的代码索引直接跳转到对应文件与函数，避免重新从头读一遍整个仓库。

### 0.3 分析边界

本次已阅读上传主论文 11 页的正文、公式、全部图表和参考文献页，并检查官方仓库的核心训练、模型、数据、损失、优化器、调度器和评测文件。指导文件提及的 `00_Research_Overview.md` 与 `07_Current_Workspace.md` 未随本次任务提供，当前环境也无法访问用户的本地项目目录，因此没有读取这两个文件。没有访问用户的本地 `D:\CODE\Doing\WHU_Code\WHU`、服务器 `/project/code/WHU_Code/WHU_DDP`、实际数据目录、模型权重或训练日志，因此**本文描述的是公开官方仓库快照，不自动等于用户当前本地/服务器代码的实际状态**。官方补充材料链接在当前环境返回 403，未能单独核对；凡依赖补充材料或实际数据文件才能确定的内容均标为未决。

------------------------------------------------------------------------

## 1. 一页核心结论

1.  **UAD 是共享单分支 ViT-B/16，而不是三模态分支融合网络。** RGB、NIR/IR、TIR/Thermal 图像分别通过同一套 backbone；代码中的 `torch.cat(list(x), dim=0)` 是沿 batch 维拼接，不是沿特征维融合。模型没有模态专属 backbone、模态 token 或显式跨模态注意力交互。\[C\]
2.  **最终检索描述子是 768 维的 ViT 最后一层 CLS 特征。** 训练时分类使用 BNNeck 后特征，WRT、ProCA、GPD 使用 BN 前全局特征；推理按 `TEST.NECK_FEAT='before'` 取 BN 前特征，再做 L2 归一化。\[C\]
3.  **UAD 的核心是两个附加训练约束，不是增加可学习特征分支。** ProCA 用 batch 内同一身份的模态中心进行对齐；GPD 用跨 iteration 的身份原型记忆进行全局分类式对比。原型记忆是 buffer，不通过反向传播更新。\[P\]\[C\]
4.  **四项损失作用于不同路径：** CE 通过 BNNeck 和分类器监督身份分类；WRT 在 BN 前特征上做 batch 内加权三元组约束；ProCA 在归一化模态中心上对齐；GPD 在归一化特征与归一化全局原型之间做交叉熵。四项梯度最终在共享 backbone 汇合。\[C\]
5.  **论文消融表显示，GPD 是主要增益来源，ProCA 的单独收益较小。** 在 WHU-MARS-1000 的 Table 8 中，baseline 为 8.9 mAP / 23.9 Rank-1，ProCA 单独为 9.3 / 24.5，GPD 单独为 10.9 / 28.7，二者合用为 11.0 / 29.5。\[R\]
6.  **TIR 是最清晰的性能瓶颈。** Table 6 中 TIR→TIR 为 20.98 Rank-1 / 14.00 mAP，低于 RGB→RGB 的 40.47 / 26.83 与 NIR→NIR 的 29.79 / 17.93；TIR→RGB、TIR→NIR 也只有 9.04、8.57 Rank-1。由此不能把 TIR 问题简单归结为跨模态对齐不足，TIR 的模态内表征也明显较弱。\[R\]
7.  **场景空间是“模态 × 平台”，但实现没有显式建模平台/相机。** 配置关闭 SIE camera/view；模型 forward 不向 backbone 传入 camera 或 modality 标签；ProCA 只按模态求中心，不按 ground/aerial 或 camera 求中心。\[C\]
8.  **论文、默认配置和 README 推荐命令之间存在需要保留的差异。** 公开 DDP 命令将 `IMS_PER_BATCH` 从 YAML 的 64 覆盖为 256、将 `BASE_LR` 从 0.008 覆盖为 0.032；自定义 PKM sampler 中一个 sampler index 实际代表一个包含三模态图像的 tuple，因此必须按采样器语义理解 batch。\[C\]
9.  **公开代码并不能完整复现论文报告的所有评测输出。** 评测器计算 CMC 与 mAP、并做 3×3 模态对评估，但当前实现没有 mINP 计算；1000-GD 的筛选代码在数据集类中被注释；默认评测也没有显式 ground↔aerial 协议分流。\[C\]
10. **最值得继续研究的不是“随便加一个模块”，而是几个已被证据定位的机制缺口：** TIR 模态内表征、模态与平台因素未分离、ProCA 中心可靠性和局部多模态结构、训练损失几何与推理距离不完全一致、prototype bank 的动态行为、采样中的模态重采样，以及评测协议和报告完整性。这些是研究问题，不是本文预先确定的改进方案。\[I\]

后续研究目标按指导文件记录为：在 WHU-MARS-1000、1000-GD、2337 上，mAP、Rank-1、Rank-5、Rank-10、mINP 均争取较论文 UAD (Ours) 至少提升 1%，且底线不低于论文对应值。这里不擅自把“1%”解释成相对百分比或百分点；后续正式结果表应明确口径。还要注意论文 Table 3/4 未报告 AS-ReID/AG-ReID 的 mINP，因此这些协议的 mINP 需要先建立一致的评测实现与可比基准，不能假装论文已提供对应数值。

------------------------------------------------------------------------

## 2. 任务定义、数据集结构与评测目标

### 2.1 Figure 1：AS-ReID 到底比传统 ReID 多了什么

**论文位置：** PDF 第 1 页，Figure 1；正文第 1–2 页。

Figure 1(a) 将传统 RGB ReID、Aerial-Ground ReID、Visible-Infrared ReID、Multi-Modality ReID 与 AS-ReID 放在一起对比。前几类通常预先限定模态或视角组合；AS-ReID 则将一个场景定义为 `(modality, viewpoint)` 组合，场景集合为：

$$
\mathcal{S}=\mathcal{M}\times\mathcal{V}
$$

在 WHU-MARS 中，主要场景维度是 RGB/NIR/TIR 与 ground/aerial 的组合，即最多六种模态—平台场景。Figure 1(b) 强调的是**任意场景 query 对混合场景 gallery 的检索**，而不是某一对固定模态之间的检索。

这一定义对分析 UAD 很重要：模型不仅需要把 RGB 与红外映射到相近空间，还需要同时承受平台高度、视角、人体尺度、相机节点、时间、天气和季节变化。若只看一个固定 query-gallery 对，可能掩盖全场景检索的弱点。

### 2.2 Table 1：WHU-MARS 的数据覆盖优势与不平衡

**论文位置：** PDF 第 3 页，Table 1。

WHU-MARS 的结构性特点如下：

- 数据在校园环境采集，使用 2 套 UAV 平台与 5 个 ground camera nodes；UAV 搭载 DJI H20T，飞行高度约 20–50 m，地面相机高度约 1.5 m。采集跨越 7 月至次年 1 月、当地时间约 09:00–21:00，共 13 个 session、超过 38 小时视频。
- 同一身份覆盖 RGB、NIR、TIR，并同时包含 ground 与 aerial 观察。
- 数据覆盖昼夜、多个季节和天气，且包含不同传感器及成像条件。
- WHU-MARS-2337 的各模态数量并不均衡。按 Table 1 的旧版论文统计，RGB 总量约为 155,892，NIR 约为 155,655，TIR 约为 123,073；TIR 比 RGB/NIR 少约两成。
- WHU-MARS-1000 是更平衡的子集，论文将其描述为包含 61,974 个帧同步三光谱 triplet；WHU-MARS-2337 则保留更完整但非完全同步的图像集合。

**机制含义：** 1000 与 2337 不只是“样本量不同”。1000 的模态分布更平衡、且有更多同步三光谱样本；2337 的 TIR 数量较少，跨模态样本对应关系也不总是帧同步。二者同时改变了采样分布、身份数量、场景覆盖和负样本难度。因此，2337 指标下降不能只归因于“gallery 更大”。\[I\]

### 2.3 Table 2：训练身份与测试身份是分开的

**论文位置：** PDF 第 4 页，Table 2。

| 数据版本      | Train IDs | Train images | Gallery IDs | Gallery images | Query images |
|---------------|----------:|-------------:|------------:|---------------:|-------------:|
| WHU-MARS-2337 |     1,000 |      217,504 |       1,337 |        217,116 |       15,020 |
| WHU-MARS-1000 |       500 |       92,313 |         500 |         93,609 |        6,405 |

Query 数量按模态分别为：2337 的 RGB/NIR/TIR 为 5,254/5,275/4,491；1000 的三模态各为 2,135。论文说明 query 是从相应 gallery 图像池中按身份和相机抽取的 1–3 张图像。因此，表中 query 计数应理解为评测 query 子集，不应直接与训练、gallery 数相加当作独立唯一图像总数。

**重要含义：** 训练分类器和 GPD 原型库使用的是训练身份（1000 版本为 500 类，2337 版本为 1000 类），评测身份是未参与训练的新身份。因而 GPD 的训练身份原型并不是测试身份的原型；它通过训练身份上的全局分离约束来学习可泛化的嵌入空间。

### 2.4 Figure 2：图像异质性不止是模态变化

**论文位置：** PDF 第 4 页，Figure 2。

Figure 2 将 ground/aerial 两行样本与 viewpoint、modality、day/night、weather、season 等变化并列展示。它传达的关键点是：同一身份的变化来源是多因素叠加的，而不是单独的 RGB↔IR 转换。

这对 UAD 的局限分析有直接影响：如果同一身份在不同模态中的样本同时来自不同平台、相机、时间或天气，那么“模态差异”可能与“平台/成像条件差异”共变。当前论文未给出每个模态 × 平台 × 昼夜 × 季节的完整交叉计数，因此无法仅凭总体结果断言某一因素独立造成了多少性能下降。\[U\]

### 2.5 评测协议：同相机 gallery 排除

**论文位置：** PDF 第 5 页，公式 (2)；代码位置：`CVPR26_UAD/utils/metrics.py` 的 `eval_func` / `eval_func_top_k`。

论文将 query 的有效 gallery 定义为：

$$
\mathcal{G}_q=\{(x,y,s,c)\in\mathcal{G}\mid c\neq c_q\}
$$

也就是说，排除的是**所有与 query 相机 ID 相同的 gallery 图像**，不只是同身份同相机图像。公开评测代码同样按 `g_camids != q_camid` 过滤，和论文公式一致。

此协议有两个需要长期保留的注意点：

1.  结果会依赖文件名中的 camera ID 是否跨模态、跨平台保持一致且具有正确语义。公开数据类从文件名正则 `r'(\d+)_c(\d+)'` 解析 PID/camera，并将 camera ID 减 1；评测时不额外验证相机 ID 的物理含义。
2.  论文描述 query 是从 gallery 图像池中抽取，而公开代码分别读取 `query/` 与 `test/` 文件夹并将前者作为 query、后者作为 gallery；代码不做图像哈希去重。实际目录中是否存在相同文件的复制版本，需要用拿到的合法数据清单核对，不能仅凭论文文字或 loader 假定。\[U\]

------------------------------------------------------------------------

## 3. UAD 的整体结构：单分支骨干 + 两个训练期机制

### 3.1 Figure 3：图中真正发生的交互

**论文位置：** PDF 第 5 页，Figure 3；代码位置：`CVPR26_UAD/model/make_model.py`、`CVPR26_UAD/processor/processor.py`。

Figure 3 左侧的多场景输入经由一个统一 ReID backbone。论文强调“不使用场景特定分支或 pairwise alignment objective”时，应理解为不为每对模态/平台设置独立分支或专用对齐损失；基础 WRT 本身仍然会计算 batch 内正负样本距离。后续的 ProCA 与 GPD 并非两个独立特征提取分支：

- **ProCA：** 把同一身份、同一模态的多个特征汇聚成模态中心，再把各模态中心拉向身份中心。
- **GPD：** 使用身份中心更新 prototype memory；每个训练样本与所有训练身份原型进行对比式分类。
- **Shared Classifier：** 共享身份分类器用于常规 CE 监督。

图中的颜色编码身份、形状编码模态；中心和原型是训练约束的中间对象。图示不意味着图像层面进行了跨模态融合，也不意味着推理时需要把 RGB/NIR/TIR 三张图一起输入。

### 3.2 模型骨干与维度

**代码：** [`model/backbones/vit_pytorch.py`](https://github.com/msm8976/WHU-MARS/blob/3e2a07314119402586c76096e87d40420894ad30/CVPR26_UAD/model/backbones/vit_pytorch.py)，`TransReID`、`PatchEmbed`、`forward_features`、`vit_base_in`；[`model/make_model.py`](https://github.com/msm8976/WHU-MARS/blob/3e2a07314119402586c76096e87d40420894ad30/CVPR26_UAD/model/make_model.py)，`build_transformer`。

- Backbone：ViT-Base，embedding dimension 768，12 个 Transformer block，12 个 attention heads，MLP ratio 4。
- 输入尺寸：训练/测试配置为 `[256,128]`，即高度 256、宽度 128；论文以宽 × 高写作 128×256。
- Patch size 与 stride：16×16。因而每张图得到 `16 × 8 = 128` 个 patch token，再加 1 个 CLS token，总 token 数为 129。
- Patch embedding 是卷积投影；每个 block 使用 LayerNorm、Multi-Head Self-Attention、MLP 与 residual connection。Drop path rate 默认配置为 0.1，dropout 和 attention dropout 为 0。
- 最终 backbone 输出最后一层 LayerNorm 后的 CLS token，维度为 768；没有使用 patch token 的平均池化，也没有在最终描述子中拼接局部 token。
- 由于预训练权重来自 224×224 的 ImageNet ViT-B/16，预训练位置编码需要从原先 197 个 token（196 patches + CLS）插值到当前 129 个 token（128 patches + CLS）。

### 3.3 预训练与初始化

配置 `configs/UAD.yml` 使用：

- `PRETRAIN_CHOICE: imagenet`
- `PRETRAIN_PATH: jx_vit_base_p16_224-80ecf9dd.pth`
- backbone 通过 `load_param` 载入兼容的 ImageNet 权重；分类 head、distillation、pre-logits 等不兼容的 head 权重会跳过。
- backbone 中的线性层、LayerNorm、CLS token、位置编码有各自初始化逻辑；载入 ImageNet 权重后，兼容的 backbone 参数被预训练参数覆盖。
- UAD 的身份分类器为 `Linear(768, num_classes, bias=False)`，分类器权重用标准差 0.001 的正态分布初始化。
- BNNeck 是 `BatchNorm1d(768)`，gamma 初始化为 1、beta 为 0，beta 的梯度被关闭。
- 当前配置关闭 `SIE_CAMERA` 与 `SIE_VIEW`，因此实际网络不添加 camera/viewpoint 的可学习偏置；模型 forward 也没有把 `camids` 或 modality label 传入 backbone。

**核心判断：** 预训练模型提供的是通用 RGB 图像特征初始化；NIR/TIR 通过同一套 3 通道输入接口和同一套参数学习。UAD 的模态适应主要依靠数据、身份监督、WRT、ProCA 和 GPD，而不是显式的模态专用参数或输入条件化机制。\[C\]

### 3.4 输入处理与数据增强

**代码：** `CVPR26_UAD/datasets/make_dataloader.py`、`datasets/bases.py`。

训练增强顺序为：

1.  Resize 到 256×128；
2.  RandomHorizontalFlip，概率 0.5；
3.  Pad 10 像素；
4.  RandomCrop 回 256×128；
5.  ToTensor；
6.  Normalize，mean/std 均为 `[0.5,0.5,0.5]`；
7.  Random Erasing，概率 0.5，pixel mode，最多一个 erase block。

验证/测试只 Resize、ToTensor、Normalize，不做随机增强。图像读取使用 PIL `convert('RGB')`，统一成 3 通道接口。每张 tuple 中的三张图分别调用 transform，随机翻转、裁剪与擦除并不是跨模态同步采样；这与该方法不要求帧同步配对的设置相容，但也意味着训练增强本身不会保留精确的像素级对应关系。

论文第 6 页写“padding、random resized cropping、horizontal flipping、random erasing”；公开代码实际是 Resize + Pad + RandomCrop，并没有调用 `RandomResizedCrop`。这可能是论文表述较宽泛，也可能是文字与公开实现的细节差异；复现时应以实际运行代码为准，并记录版本。\[P\]\[C\]

------------------------------------------------------------------------

## 4. 从 sampler 到 CLS 特征：完整正向数据流

### 4.1 PKM sampler 的真实 batch 语义

**代码：** `CVPR26_UAD/datasets/sampler.py::PKMSampler`、`sampler_ddp.py::PKMSampler_DDP`、`make_dataloader.py::train_collate_fn`。

UAD 使用 `P × M × K` 采样，公开配置为 `M=3` 个模态、`K=4` 个实例/身份/模态，论文设置 `P=16` 个身份。采样器的一个 index 不是一张普通图片，而是一个 tuple：该 tuple 同时包含一个 RGB、一个 IR/NIR、一个 Thermal/TIR 图像索引，且三者具有相同 PID，但不要求来自同步帧。

- 每个身份在每个模态中抽取 4 张图像；
- 同一身份的 4 个 tuple 连续放入 batch；
- 一个 tuple 在 collate 后变成三张模态图像，分别进入三个 modality batch；
- 因此 `IMS_PER_BATCH=64` 的实际含义是每个进程 64 个 tuple 样本，即每模态 64 张、总计 192 张原始图像，而不是总共 64 张图像。

DDP README 推荐命令是 4 卡、`IMS_PER_BATCH=256`。`PKMSampler_DDP` 先按 world size 将 tuple batch 切分到每卡；每卡得到 64 个 tuple，也就是 16 个身份 × 每身份 4 个 tuple × 每 tuple 3 张图像 = 192 张原始图像/卡，与论文的“per-GPU batch size 192”一致。

### 4.2 数据从 loader 进入模型

在默认三模态顺序 `['RGB','IR','Thermal']` 下，单进程每个 batch 形成：

- `imgs[0]`: `[B, 3, 256, 128]`，RGB；
- `imgs[1]`: `[B, 3, 256, 128]`，IR/NIR；
- `imgs[2]`: `[B, 3, 256, 128]`，Thermal/TIR；
- `target`: `[B]`，每个 tuple 的身份标签，按身份分组且每组 K=4 个标签相同；
- `camids`: 三个模态各自的 camera ID 列表，但训练模型 forward 不使用它们。

`build_transformer.forward(mode=0)` 执行 `torch.cat(list(x), dim=0)`，将三个模态沿 batch 维拼接：

$$
X=[X_{RGB};X_{NIR};X_{TIR}]\in\mathbb{R}^{3B\times3\times256\times128}
$$

这只是为了用同一个模型批量处理三模态，不是图像融合或特征通道拼接。

### 4.3 Backbone 输出和分类路径

每张输入图像经过 patch embedding、CLS token、位置编码、12 层 Transformer 与最终 LayerNorm，输出：

$$
f=f_\theta(x)\in\mathbb{R}^{768}
$$

对一个三模态 batch，`global_feat` 的形状为 `[3B,768]`。模型随后计算：

$$
z=\mathrm{BN}(f),\qquad s=W_{cls}z
$$

其中 `z` 为 BNNeck 特征，`s` 为对训练身份类别的分类 logits，形状为 `[3B,C_train]`。`C_train` 是训练身份数量，而不是数据集名称中的总身份数：WHU-MARS-1000 为 500 类，WHU-MARS-2337 为 1000 类。

### 4.4 一个容易误读的点：没有显式特征融合

UAD 的特征交互发生在**损失约束和中心/原型计算层面**：

- 三模态特征经过同一个 encoder；
- CE 与 WRT 在每张图像的特征上计算；
- ProCA 将同一身份、同一模态的 K 个特征聚合成中心，再在模态中心之间对齐；
- GPD 将图像特征与身份原型库比较。

没有代码把三个模态的 CLS 特征拼成 2304 维向量，也没有在推理时要求同时提供多模态图像。每次 query 独立通过共享模型，输出单个 768 维描述子。

------------------------------------------------------------------------

## 5. 损失函数逐项拆解：优化什么、梯度流向哪里

训练目标为：

$$
\mathcal{L}_{UAD}=\mathcal{L}_{CE}+\mathcal{L}_{WRT}+0.01\,\mathcal{L}_{ProCA}+1.0\,\mathcal{L}_{GPD}
$$

这里 `λ_GPD=1.0` 在代码中不是单独配置项，而是通过 `loss += loss_gpd` 隐式实现；ProCA 权重由 `MODEL.PCA_LOSS_WEIGHT=0.01` 控制。\[P\]\[C\]

### 5.1 CE：BNNeck 后的训练身份分类

代码路径：`loss/make_loss.py::loss_func`、`model/make_model.py::build_transformer.forward`、`loss/softmax_loss.py`。

当前 `IF_LABELSMOOTH='off'`，所以实际使用普通 cross entropy，而不是 label-smoothed CE：

$$
\mathcal{L}_{CE}=\mathrm{CE}(W_{cls}\,\mathrm{BN}(f_i),y_i)
$$

`target` 按模态重复三次，与 `[RGB batch; NIR batch; TIR batch]` 的特征顺序一致。CE 的梯度流向：

- 分类器权重；
- BNNeck 的可训练参数；
- 共享 ViT backbone。

CE 不直接更新 ProCA 中心或 GPD 原型库。分类器学习的是训练身份的闭集类别边界；评测身份不属于这些训练类别，因此 CE 的价值在于塑造身份判别特征，而不是在测试时直接输出测试身份类别。

### 5.2 WRT：batch 内加权三元组约束

代码路径：`loss/wrt_loss.py::TripletLoss_WRT`、`softmax_weights`、`pdist_torch`。

WRT 在 **BN 前的 `global_feat=f`** 上计算欧氏距离矩阵，不使用 BNNeck 特征，也不调用 `normalize_feature=True`。对每个 anchor：

- 该 batch 中每个身份有 K×M=12 个同身份图像实例（含 anchor 自身）；因此正样本池覆盖同身份的模态内和跨模态样本，并非只对某一对模态做专门对齐；
- 正样本集合是 batch 内相同 PID 的样本（代码的正样本 mask 包含 anchor 自身；距离计算对平方距离设置了最小值，故自身距离是近似 0 的数值项）；
- 负样本集合是 batch 内不同 PID 的样本；
- 正样本权重对较大距离赋予更高权重，强调远正样本；
- 负样本权重对较小距离赋予更高权重，强调近负样本；
- 用 `SoftMarginLoss(d_an - d_ap, target=1)` 约束加权负距离大于加权正距离，没有显式固定 margin。

可写成机制示意：

$$
\mathcal{L}_{WRT}\approx\mathrm{softplus}(d_{ap}^{soft}-d_{an}^{soft})
$$

其中 `d_ap^soft` 和 `d_an^soft` 是对正/负样本距离加权后的值。WRT 梯度经原始 CLS 特征回到共享 backbone；分类器和 BNNeck 不接收这项损失的直接梯度。

**关键几何差异：** WRT 使用未归一化的特征欧氏距离，因此对特征向量的模长也敏感；但最终推理会对特征做 L2 归一化，只保留方向所决定的排序。这构成训练度量与最终检索度量之间的潜在不完全一致。\[C\]\[I\]

### 5.3 ProCA：先模态内聚合，再模态间对齐

论文公式 (4)–(6)，代码路径：`processor/processor.py::PCA`。代码里的 `PCA` 指 Progressive Center Alignment，不是主成分分析。

设一个身份 y 在模态 m 中的 K 个原始特征为 $f_{m,k}^{y}$。实现首先逐样本归一化，再对 K 个特征求平均并重新归一化：

$$
\tilde{\mu}_{y,m}=\frac{1}{K}\sum_{k=1}^{K}\mathrm{norm}(f_{m,k}^{y}),\quad
\mu_{y,m}=\mathrm{norm}(\tilde{\mu}_{y,m})
$$

再对 M 个模态中心求平均，得到身份中心：

$$
\tilde{\mu}_{y}=\frac{1}{M}\sum_{m=1}^{M}\mu_{y,m},\quad
\mu_y=\mathrm{norm}(\tilde{\mu}_{y})
$$

代码明确对 `g_cent` 使用 `.detach()`，因此身份中心是 stop-gradient anchor。这里的“Progressive”体现为“样本特征 → 模态中心 → 身份中心”的聚合层次；公开代码每个 iteration 都计算这一路径，并没有额外的 epoch 级渐进调度。ProCA 损失为：

$$
\mathcal{L}_{ProCA}=\frac{1}{P}\sum_y\frac{1}{M}\sum_m\|\mu_{y,m}-\mu_y\|_2^2
$$

实现中的 `group_size=NUM_INSTANCE=4`，并依赖 sampler 保证 `target` 中每个身份连续出现 K 次。它把同一身份每个模态的中心向共享身份中心拉近。

**梯度细节：** 梯度通过每个模态中心及其组成的归一化特征回传到 backbone；不会通过身份中心 `g_cent` 反向传播。身份中心在当前 batch 内被计算，但只作为固定 anchor 使用。ProCA 不包含额外的可学习参数，也没有独立的 ProCA optimizer。

**它直接约束的对象不是每一张图像到中心的距离，而是模态中心之间的距离。** 单张样本通过中心均值间接收到梯度；少数离群样本可能被均值稀释，也可能因进入中心而影响整个 K 样本组。\[C\]\[I\]

### 5.4 GPD：跨 batch 的身份原型记忆与全局区分

论文公式 (7)–(9)，代码路径：`processor/processor.py::PrototypeBank`、`GPD`。

原型库为每个**训练身份**保存一个 768 维向量，初始化为全零 buffer。每个训练 batch 中：

1.  固定当前 prototype bank，计算本 batch 的 GPD loss；
2.  对每张特征做 L2 归一化；
3.  将归一化特征与全部训练身份原型计算点积，并除以温度 $\tau=0.03$；
4.  用训练 PID 作为类别标签计算交叉熵；
5.  完成 optimizer step 后，再用当前 batch 的身份中心更新原型库。

GPD 目标可写为：

$$
\mathcal{L}_{GPD}^{(i)}=-\log\frac{\exp(\mathrm{norm}(f_i)^\top p_{y_i}/\tau)}{\sum_{y\in\mathcal{Y}_{train}}\exp(\mathrm{norm}(f_i)^\top p_y/\tau)}
$$

原型更新的论文写法是：

$$
p_y\leftarrow\mathrm{norm}(\alpha p_y+(1-\alpha)\mu_y),\quad \alpha=0.8
$$

代码配置 `GPD_MOMENTUM=0.2`，但实现写成 `norm((1-m)*old + m*new_center)`。所以这里的代码 `m=0.2` 是**新中心权重**，旧原型权重是 0.8，与论文的 $\alpha=0.8$ 在数值行为上对应，并非冲突。

代码对尚未初始化的当前 batch 身份原型，用当前 batch 的 `g_cent` 临时替换零向量，再归一化用于本轮 logits。其余尚未初始化身份仍是零原型，对应 logits 为 0。随着训练进行，prototype bank 逐渐被填充。

**梯度边界：** GPD 的梯度流向当前特征和 backbone；`P` 是 buffer 的 clone，原型不接收梯度。prototype bank 在 `torch.no_grad()` 下通过 EMA 更新。DDP 时先 all-gather 各 rank 的身份中心，再在每个 rank 上按身份聚合并更新，从而让各 rank 的原型库保持同步。注意：GPD 的原型更新本身依赖 `PCA` 函数产生的 `g_cent`，所以“GPD-only”通常表示不加 ProCA 对齐损失，并不表示完全不计算模态中心/身份中心。

### 5.5 四项损失的总梯度流

``` text
image batch (RGB / NIR / TIR)
             │
             ▼
       shared ViT-B/16
             │
      raw CLS feature f (768D)
        ┌────┼───────────────┬────────────────┐
        │    │               │                │
        ▼    ▼               ▼                ▼
      BN(f) WRT pairwise   ProCA centers    normalize(f)
        │    distances        │                │
        ▼    │                ▼                ▼
   classifier│           center alignment   GPD vs bank
        │    │                │                │
        ▼    ▼                ▼                ▼
       L_CE  L_WRT       0.01 * L_ProCA       L_GPD
        └────┴───────────────┴────────────────┘
                         │
                         ▼
              summed gradient to shared ViT
```

- CE 更新 classifier、BNNeck 和 backbone。
- WRT 更新 backbone。
- ProCA 更新 backbone；identity center stop-gradient。
- GPD 更新 backbone；prototype bank 不经梯度更新，而是在 optimizer step 后手动 EMA 更新。
- BNNeck 与分类器只受 CE 直接监督。

### 5.6 训练实现中的额外细节

- `CenterLoss` 在 `make_loss.py` 中被实例化，`optimizer_center` 也被创建并在每个 iteration 调用 `zero_grad()`，但训练目标中没有加入 CenterLoss，且没有调用 `optimizer_center.step()`。所以它不是 UAD 实际训练的损失项，ProCA 才是论文所述的中心机制。\[C\]
- GPD 原型更新发生在 `scaler.step(optimizer)` / `scaler.update()` 之后，但代码没有检查 AMP 是否因非有限梯度跳过了 optimizer step；若发生跳步，prototype bank 仍会用当前前向得到的中心更新。这是实现事实，实际发生频率及影响尚无日志证据。\[C\]\[U\]
- prototype bank 没有纳入 model optimizer，也不随 `model.state_dict()` 一起保存。当前代码只保存模型 state dict，没有完整保存 bank、optimizer、scheduler、GradScaler 和随机状态。因此它适合一次性完整训练，但从 checkpoint 精确恢复 UAD 训练状态并未在公开入口中实现。\[C\]

------------------------------------------------------------------------

## 6. 优化器、学习率、训练阶段与随机性

### 6.1 公开配置与论文描述

| 项目                        | 论文描述                                | `configs/UAD.yml` 默认值 | README 四卡命令覆盖值                  |
|-----------------------------|-----------------------------------------|--------------------------|----------------------------------------|
| Backbone                    | ImageNet 预训练 ViT-Base                | `vit_base_in`            | 不变                                   |
| 输入尺寸                    | 128×256                                 | `[256,128]`（H×W）       | 不变                                   |
| Epochs                      | 120                                     | 120                      | 不变                                   |
| Optimizer                   | SGD                                     | SGD                      | 不变                                   |
| Momentum                    | 论文未在主文单独强调 optimizer momentum | 0.9                      | 不变                                   |
| Base LR                     | 0.008 per GPU 的论文表述                | 0.008                    | 0.032                                  |
| Warm-up                     | 5 epochs linear                         | 5 epochs                 | 不变                                   |
| Scheduler                   | cosine decay                            | cosine                   | 不变                                   |
| `IMS_PER_BATCH`             | 每卡 192 张原始图像                     | 64 个 tuple / 单进程     | 256 个 tuple 全局，4 卡时每卡 64 tuple |
| `NUM_INSTANCE`              | K=4                                     | 4                        | 不变                                   |
| ProCA weight                | 0.01                                    | 0.01                     | 不变                                   |
| GPD temperature             | 0.03                                    | 代码固定为 0.03          | 不变                                   |
| Prototype old-memory weight | 0.8                                     | 新中心权重配置为 0.2     | 不变                                   |

README 的官方四卡命令：

``` bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=16666 train.py \
  --config_file configs/UAD.yml MODEL.DIST_TRAIN True SOLVER.IMS_PER_BATCH 256 SOLVER.BASE_LR 0.032
```

**不能把 YAML 默认值当作论文四卡训练的完整配置。** YAML 默认的 `IMS_PER_BATCH=64` 更接近单卡运行；如果不覆盖就直接用四卡 DDP，采样器会把它分成每卡 16 个 tuple（48 张原始图像），每卡约 4 个身份，明显不是论文的 P=16、每卡 192 张图像设置。四卡推荐命令把它覆盖为 256。由于 sampler 的 index 是一个三模态 tuple，这个 256 不等于全局只有 256 张原始图片，而是 256 个 tuple，即每个模态合计 256 张、三模态合计 768 张原始图像；四卡每卡处理 64 个 tuple / 192 张图像。`BASE_LR=0.032` 与论文中的 0.008-per-GPU 表述可理解为四卡线性放大设置，但比较实验时必须记录实际命令和每卡/全局 batch 语义。

### 6.2 优化器与学习率组

代码路径：`solver/make_optimizer.py`、`solver/scheduler_factory.py`、`solver/cosine_lr.py`。

- SGD，momentum=0.9。
- Weight decay 与 bias weight decay 均为 1e-4。
- 参数名包含 `bias` 时，学习率乘以 `BIAS_LR_FACTOR=2`；分类器没有 bias，`LARGE_FC_LR=False`，所以不会额外给 classifier 设置两倍 LR。
- scheduler 使用 120 epoch cosine，`lr_min=0.002 × base_lr`，warm-up 初始值为 `0.01 × base_lr`，warm-up 5 epoch。
- `do_train` 在每个 epoch 开始调用 `scheduler.step(epoch)`，epoch 从 1 开始。因此初始化设置的 `0.01 × base_lr` 并不是一个完整训练 epoch 的实际 LR；第 1 个 epoch 已进入线性 warm-up 的第 1 个步点，约为 base LR 的 20.8%，第 5 个 epoch 到达 base LR。复现时若要逐项对齐 LR 曲线，应看实际日志，而不只看配置中的 `WARMUP_EPOCHS=5`。\[C\]
- 每次反向传播后先 `scaler.unscale_(optimizer)`，再做 `clip_grad_norm_(model.parameters(), max_norm=1.0)`，随后 `scaler.step` 与 `scaler.update`。因此 clip 的对象是四项损失合并后的模型总梯度，而不是分别对每个损失单独裁剪。配置默认每 10 个 epoch 做一次评估、每 120 个 epoch 保存一次模型，常规完整训练只在结束时保存最终模型；这也进一步说明当前 checkpoint 不是可完整恢复训练状态的快照。

### 6.3 随机种子与可复现性

`train.py::set_seed` 将 torch、CUDA、NumPy、Python `random` 设为 seed 1234；同时设 `torch.backends.cudnn.deterministic=True` 和 `benchmark=True`。这表明代码有固定随机种子的意图，但这两个 cuDNN 设置同时开启并不等于严格的逐 bit 确定性保证。数据采样、worker 随机增强、CUDA 算子和 DDP 环境仍应在正式复现时记录。

------------------------------------------------------------------------

## 7. 推理描述子与评测代码实际做了什么

### 7.1 推理特征

**代码：** `model/make_model.py::build_transformer.forward`、`processor/processor.py::do_inference`、`utils/metrics.py::R1_mAP_eval.compute`。

当 `mode != 0` 时，模型只执行 backbone 与 BNNeck；若 `TEST.NECK_FEAT='before'`，返回 `global_feat=f`，而不是 BN 后特征。官方 UAD 配置设为：

- `NECK_FEAT='before'`；
- `FEAT_NORM='yes'`；
- `RE_RANKING=False`。

评测器再对 query/gallery 特征做 L2 normalization，并用平方欧氏距离排序。对单位归一化特征而言，平方欧氏距离与 cosine similarity 具有相同排序关系。因此 UAD 最终用于检索的是**L2 归一化后的 768 维 CLS 描述子**。

### 7.2 指标的含义与评测代码的输出范围

- **Rank-k/CMC：** 对每个 query，前 k 个 gallery 中至少出现一个正确身份匹配的比例；Rank-1 只关注第一名，Rank-5/10 允许更靠后的首个正确匹配。

- **mAP：** 对每个 query 计算平均精度，再对 query 求平均，既考虑正确匹配的数量，也考虑它们在排序中的位置。

- **mINP：** 每个 query 的 INP 通常以“相关 gallery 样本数 / 最后一个相关样本的排名”衡量，再对 query 求平均；它更关注最难的、排在最后的相关样本是否被检索到。

- loader 为 RGB、IR、Thermal 分别构建 query+gallery 序列；评测器将每种模态的前 `num_query` 张作为 query，其余作为 gallery，再将三模态合并为全场景检索。

- 主 AS-ReID 评测排除同 camera ID gallery，计算 CMC（Rank-1/5/10 等）和 mAP。

- 同一评测器还对 3×3 query-modality / gallery-modality 组合单独计算 CMC 与 mAP。

- 默认不做 re-ranking。YAML 注释里出现的 `TOP_K_EVAL=50000` 并未启用；`config/defaults.py` 的 `TOP_K_EVAL=0` 会让当前配置走全 gallery 排序分支。

- 当前公开 `utils/metrics.py` **没有 mINP 计算**，返回值也只有 CMC、mAP 和模态对结果。因此论文 Table 5、Table 6 的 mINP 数值不能直接由当前公开评测函数复现，需要额外实现或使用其他评测脚本；本文不假设用户本地版本没有做过这项扩展。

- 评测器不显式计算 ground→aerial / aerial→ground 的平台对协议。`whu_mars.py` 没有返回独立 platform label，默认 loader 只按模态拆分。因此 Table 4 的 AG-ReID 结果不能仅靠默认的 3 模态评测入口直接得到，可能需要独立的子集数据组织或额外协议代码；当前公开主入口未显示这部分完整实现。

------------------------------------------------------------------------

## 8. 论文所有图的逐图分析

### Figure 1（PDF p.1）— 任务动机

图中把传统 ReID、跨平台、可见—红外、多模态与 AS-ReID 对照。AS-ReID 的核心不是新增某一种传感器，而是把任意 query 与混合 gallery 统一到单模型检索中。它要求模型在训练中同时处理模态和平台差异，不能只依赖固定方向的配对映射。

### Figure 2（PDF p.4）— 数据变化来源

图中以 ground/aerial 两行展示三光谱样本，并用 viewpoint、modality、day/night、weather、season 五类因素说明变化来源。它支持“身份变化具有多因素叠加”的任务描述，但不能单独证明哪一个因素是当前模型性能下降的主因；需要分组指标或受控数据分析才能分解。

### Figure 3（PDF p.5）— UAD 总体机制

图中显示共享 backbone、ProCA 的模态中心与身份中心，以及 GPD 的原型记忆。图示与代码一致的部分是：共享特征提取器、中心级对齐、全局身份原型和共享分类监督。需要避免的误读是：图中没有三模态 feature concatenation、独立专家或模态专属分支；中心和原型是训练期的损失/记忆对象，推理不需要原型库。

### Figure 4（PDF p.8）— 类内/类间 cosine 分布

图例将绿色标为类内 pair、蓝色标为类间 pair。按图中虚线均值读取：

- Baseline：类内均值约 0.591，类间均值约 0.430，差值 0.1617；
- UAD：类内均值约 0.286，类间均值约 0.020，差值 0.2654。

因此图 4 直接支持的是：UAD 增大了类内与类间相似度均值之间的间隔，且数值变化主要表现为类间相似度下降更多。**图中类内平均 cosine similarity 本身也下降了**，所以不能只凭“均值差扩大”就说类内相似度绝对变高。更稳妥的结论是整体分布的区分间隔扩大，具体是类内收紧、类间推远、还是两者同时发生，需要看分布方差/重叠区域与样本级结果。论文图没有提供这些更细的统计量。\[P\]\[I\]

------------------------------------------------------------------------

## 9. 论文所有表的逐表分析

### Table 1（PDF p.3）— 数据集比较

WHU-MARS 的独特性在于 RGB/NIR/TIR 与 ground/aerial 的交叉覆盖，并有 1000 平衡子集与 2337 全量子集。2337 中 TIR 数量显著低于 RGB/NIR，为采样与模态学习带来不平衡。该表主要支持数据集覆盖范围，不直接证明 UAD 的算法优势。

### Table 2（PDF p.4）— 两个数据版本的 split

1000 版本为 500 个训练身份与 500 个测试身份；2337 版本为 1000 个训练身份与 1337 个测试身份。训练身份与评测身份分离意味着训练分类器和 GPD 原型库必须泛化到新身份。2337 的训练图像数在 TIR 上更少，采样器可能需要更多重采样以维持 batch 内模态平衡。

### Table 3（PDF p.7）— AS-ReID 主结果

UAD (Ours) 的结果为：

| 评测设置         |  mAP | Rank-1 | Rank-5 | Rank-10 |
|------------------|-----:|-------:|-------:|--------:|
| WHU-MARS-1000-GD | 14.5 |   31.9 |   45.0 |    52.5 |
| WHU-MARS-1000    | 11.0 |   29.5 |   46.3 |    54.8 |
| WHU-MARS-2337    |  9.6 |   25.7 |   40.9 |    48.2 |

从 1000 到 2337，mAP 下降 1.4 个百分点，Rank-1 下降 3.8 个百分点，Rank-5 下降 5.4 个百分点，Rank-10 下降 6.6 个百分点。该下降与更大的身份空间、更复杂的场景覆盖和模态不平衡一致，但由于数据规模、训练身份数与模态分布同时变化，不能只将下降归因于 gallery hard negatives。\[R\]\[I\]

1000-GD 相比完整 1000 的 mAP 与 Rank-1 更高，但 Rank-5/10 反而更低（45.0 vs 46.3；52.5 vs 54.8）。这说明不同子集下首个正确匹配与后续正确匹配的排序结构可能不同；“子集更简单”不意味着所有指标都必须单调上升。

Table 3 的公开对比方法来自各自开源实现，论文也说明除特定任务尺寸外，其余设置沿用对应实现。UAD 在 Table 3 的三个设置中均取得表内最佳的 mAP、Rank-1、Rank-5、Rank-10；参数量为 85.7M，也略低于表中其他 Transformer-based 方法。该结果支持其统一模型的竞争力，但 Table 3 的对比方法来自各自开源实现，因此更适合看整体 benchmark 表现，而不是所有方法在完全统一训练细节下的严格受控因果比较。

### Table 4（PDF p.7）— Aerial-Ground 方向不对称

| 方向          |  mAP | Rank-1 | Rank-10 |
|---------------|-----:|-------:|--------:|
| Aerial→Ground | 11.5 |   31.9 |    58.3 |
| Ground→Aerial | 13.3 |   30.5 |    54.7 |

Aerial→Ground 的 Rank-1 更高，但 mAP 和 Rank-10 更低；Ground→Aerial 则相反。说明两个方向的首个命中与整体相关样本排序并不一致，平台变化不是一个可假定对称的映射。后续研究若只看 Rank-1，可能漏掉相关样本整体排序的退化。

### Table 5（PDF p.7）— NIR/TIR→RGB

UAD 在 NIR→RGB 上为 25.53 Rank-1 / 16.58 mAP / 3.10 mINP，并非表中最强：PMT 的 mAP 为 18.16、Rank-1 为 27.12，DEEN 的 Rank-1 为 27.63。UAD 在 TIR→RGB 上达到 9.04 / 6.32 / 1.36，高于表中其他方法的对应值。

这说明统一模型与 ProCA/GPD 对 TIR→RGB 有一定帮助，但 NIR→RGB 上仍有专用方法更强。由于这些专用方法通常训练独立的两模态模型，而 UAD 一次训练覆盖三模态，比较应同时考虑任务覆盖与性能，不应只按一个指标判断统一模型是否“更好”。

### Table 6（PDF p.8）— 3×3 模态检索

| Query→Gallery | Rank-1 |   mAP | mINP |
|---------------|-------:|------:|-----:|
| RGB→RGB       |  40.47 | 26.83 | 5.36 |
| RGB→NIR       |  26.56 | 16.52 | 2.43 |
| RGB→TIR       |   8.10 |  6.00 | 1.11 |
| NIR→RGB       |  25.53 | 16.58 | 3.10 |
| NIR→NIR       |  29.79 | 17.93 | 2.64 |
| NIR→TIR       |   7.78 |  5.87 | 1.11 |
| TIR→RGB       |   9.04 |  6.32 | 1.36 |
| TIR→NIR       |   8.57 |  5.69 | 1.02 |
| TIR→TIR       |  20.98 | 14.00 | 2.35 |

这里有两个重要事实：

1.  TIR→TIR 本身弱于 RGB→RGB 和 NIR→NIR，说明 TIR 问题包含模态内表征不足，而非仅仅是跨模态对齐失败。
2.  TIR 作为 gallery 模态时，RGB→TIR 与 NIR→TIR 的 Rank-1 仅为 8.10 和 7.78；TIR 作为 query 时，TIR→RGB/NIR 也只有 9.04/8.57。低值在多个方向同时出现，说明 TIR 相关瓶颈具有双向性。

这些结果不能直接分离 TIR 传感器质量、图像尺度、样本数量、昼夜分布和身份外观变化各自的贡献，但它们足以把 TIR 相关问题列为首要研究对象。

### Table 7（PDF p.8）— 配对 MM-ReID 对比

Table 7 的设置使用同步三光谱 triplet，属于可利用配对样本的 MM-ReID 评测；最佳 DeMo 达到 44.8 mAP / 56.2 Rank-1。该表未列出 UAD (Ours)，所以它主要说明 WHU-MARS 上的配对 MM-ReID 仍然具有挑战性，并不能作为 UAD 在该表协议下的直接性能证据。不能把 Table 7 的最高值与 Table 3 的 AS-ReID 结果直接比较，因为协议和 gallery 结构不同。

### Table 8（PDF p.8）— UAD 消融

| Lbase | ProCA | GPD |  mAP | Rank-1 | Rank-5 | Rank-10 |
|-------|-------|-----|-----:|-------:|-------:|--------:|
| ✓     |       |     |  8.9 |   23.9 |   41.4 |    50.5 |
| ✓     | ✓     |     |  9.3 |   24.5 |   42.0 |    50.7 |
| ✓     |       | ✓   | 10.9 |   28.7 |   45.9 |    54.2 |
| ✓     | ✓     | ✓   | 11.0 |   29.5 |   46.3 |    54.8 |

相对 baseline：

- ProCA 单独：mAP +0.4、Rank-1 +0.6、Rank-5 +0.6、Rank-10 +0.2 个百分点；
- GPD 单独：mAP +2.0、Rank-1 +4.8、Rank-5 +4.5、Rank-10 +3.7 个百分点；
- 二者合用：mAP +2.1、Rank-1 +5.6、Rank-5 +4.9、Rank-10 +4.3 个百分点；
- 在 GPD 上再加 ProCA：mAP +0.1、Rank-1 +0.8、Rank-5 +0.4、Rank-10 +0.6 个百分点。

因此，Table 8 的主要证据是 GPD 损失带来的性能增益明显大于 ProCA 对齐损失；ProCA 在 GPD 之上有额外增益，但 mAP 增量很小。论文将两者的协同解释为 ProCA 有助于稳定 prototype memory，但表格没有直接报告 prototype 的方差、漂移或稳定性指标，所以它证明的是性能协同，不是对“原型稳定性提高”的独立测量。解释消融时还要注意公开 `do_train` 的实现：只要 `LOSS_TYPE != 'base'`，训练循环就会计算 PCA 和 GPD 两项，并用 PCA 生成的 `g_cent` 更新原型库；是否把某项损失加进总目标才由字符串包含 `pca` / `gpd` 控制。因此 `base+gpd` 消融仍使用 ProCA 的中心构造作为 prototype 更新来源，只是没有加入 ProCA 对齐损失；`base+pca` 则会计算但不加入 GPD loss。不能把 Table 8 理解成“GPD 完全不依赖任何中心构造”。论文没有报告多随机种子均值、标准差或置信区间，所以不能仅凭 +0.1 mAP 判断 ProCA 的 mAP 增益在不同运行中是否稳定。\[R\]\[U\]

------------------------------------------------------------------------

## 10. 论文与公开代码的一致性审计

| 项目                     | 论文/公开描述                                    | 公开代码实际情况                                                                         | 分析结论                                   |
|--------------------------|--------------------------------------------------|------------------------------------------------------------------------------------------|--------------------------------------------|
| 共享模型                 | 单模型、单 backbone                              | `build_transformer` 共享 ViT；模态沿 batch 维拼接                                        | 一致                                       |
| 特征维度                 | ViT-Base 表征                                    | CLS 768D                                                                                 | 一致                                       |
| ProCA                    | 模态中心→身份中心，identity center stop-gradient | `PCA` 归一化、分组均值、中心归一化、`detach()`                                           | 核心机制一致                               |
| GPD                      | 全局原型、温度 0.03、EMA                         | `GPD` + `PrototypeBank`；新中心系数 0.2、旧原型系数 0.8                                  | 机制一致，momentum 参数命名容易误读        |
| 损失权重                 | ProCA 0.01、GPD 1.0                              | PCA 乘 0.01，GPD 直接加和                                                                | 一致                                       |
| batch size               | 论文每卡 192 张图像                              | README 四卡命令每卡 64 tuple × 3 模态图像                                                | 一致，但配置字段名称容易误读               |
| 学习率                   | 论文 0.008 per GPU                               | YAML 默认 0.008；README 四卡命令覆盖为 0.032                                             | 需按实际命令理解；不是一个可忽略的配置差异 |
| 数据增强                 | 论文写 random resized cropping                   | 代码为 Resize + Pad + RandomCrop                                                         | 文字与实现细节不完全一致                   |
| 数据集 1000 图像总数     | 上传论文 Table 1/2 对应 185,922                  | 当前 repo main 的 README 和 `whu_mars.py` docstring 已改为 185,742，ground 每模态 46,152 | 版本/统计冲突，需以实际数据清单为准        |
| 1000-GD                  | 论文报告 ground+daytime 子集结果                 | `whu_mars.py` 中 PID/camera 筛选示例被注释；默认 loader 未实现 day/night 字段筛选        | 默认入口不能保证复现 Table 3 的 1000-GD    |
| AG-ReID Table 4          | 报告 A→G 与 G→A                                  | 默认 loader/evaluator 没有显式 platform label 或平台对分流                               | 公开默认入口不足以直接复现该协议           |
| mINP                     | Table 5/6 报告 mINP                              | 当前 `utils/metrics.py` 只计算 CMC、mAP 和模态对 CMC/mAP                                 | 评测输出不完整，需要其他脚本或扩展         |
| CenterLoss               | 主方法为 ProCA/GPD                               | `CenterLoss` 被实例化但未加入总损失，optimizer 未 step                                   | CenterLoss 不属于实际 UAD 训练目标         |
| camera/view conditioning | 任务包含平台/视角差异                            | `SIE_CAMERA=False`、`SIE_VIEW=False`；forward 不传 camera/view/modality 标签             | 当前方法没有显式条件化相机/平台/模态       |
| Prototype bank 保存      | 论文描述迭代更新记忆                             | bank 是 processor 内部 buffer，模型 checkpoint 不包含它                                  | 完整续训状态未被保存                       |

### 10.1 数据统计冲突应如何理解

论文 PDF Table 1 报 WHU-MARS-1000 为 185,922 张图像，ground 每模态 46,212；Table 2 的 train 92,313 + gallery 93,609 也等于 185,922。官方仓库当前 main 的最后提交（2026-08-05，commit message 为 “Correct dataset statistics”）将 README 和 `whu_mars.py` docstring 改为 185,742、ground 每模态 46,152；差值为 180 张图像，即每个 ground 模态少 60 张。

这是真实存在的**论文版本与当前仓库元数据不一致**。当前环境没有获得许可数据包及逐文件清单，无法判定这是否对应数据版本修订、统计口径变化或其他原因。不能静默选择一个数字覆盖另一个；本地复现时应记录数据包版本，并从实际文件计数和 split manifest 计算数量。

### 10.2 其他实现细节的影响范围

- `whu_mars.py` 将文件名中的第四个字段设为 modality index；继承的 `get_imagedata_info` 将该字段统计为 `trackid/view`，所以 `view_num` 实际上更接近模态数而不是物理视角数。但当前 `SIE_VIEW=False`，这个统计不会影响当前 backbone forward。
- `TransReID.forward_features` 支持 camera/view embedding，但模型构建后实际 forward 没有传入这些标签；由于当前 SIE 关闭，现有配置可运行。若只把开关打开而不调整调用路径，不能假设条件 embedding 会自动正确工作。
- `MODEL.NO_MARGIN=True` 是配置项，但当前选择的 `TripletLoss_WRT` 使用 `SoftMarginLoss`，并不读取这个开关；它不是 WRT 是否启用固定 margin 的实际控制项。
- `do_train` 以 `loss_type != 'base'` 决定是否进入 auxiliary 分支，并在该分支中总是计算 PCA 与 GPD；具体损失是否加入总目标，再由字符串是否包含 `pca` 或 `gpd` 决定。因此 `base+gpd` 的原型仍由 `PCA` 计算出的 `g_cent` 更新，`base+pca` 也会做无损失贡献的 GPD 前向和原型更新。这不改变该消融对“是否加入 ProCA loss/GPD loss”的定义，但解释结果时必须保留这一实现依赖。
- 当前 `eval_func_top_k` 在启用 top-k 截断时会用完整 gallery 的相关样本数作 AP 分母，但只对 top-k 结果求和；这属于截断近似评测。当前 UAD YAML 未启用 `TOP_K_EVAL`，默认走全排序分支，因此不影响当前默认结果。

------------------------------------------------------------------------

## 11. UAD 已有的优势与结构性薄弱点

本节只识别原模型已有的能力和限制，不给出具体改进方案。

### 11.1 已有优势

1.  **统一性强：** 单一共享 backbone，不需要为每对模态/平台训练独立模型。
2.  **无需帧级配对：** ProCA 按身份和模态聚合，不依赖同步 RGB/NIR/TIR 帧；适用于 2337 中并非完全同步的样本。
3.  **中心级约束与全局原型约束互补：** ProCA 主要控制同身份的跨模态中心差异，GPD 提供训练身份集合上的全局类别分离。
4.  **推理简单：** 只需要单张图像和 768D CLS 描述子，prototype bank 仅用于训练。
5.  **参数成本低：** ProCA 与 GPD 没有额外可学习的 backbone 分支；原型库作为 buffer，不会把模型变成多分支网络。
6.  **消融有明确证据：** Table 8 显示两个机制都能改善 baseline，GPD 的收益尤其明显。

### 11.2 薄弱点 A：TIR 的模态内表征和跨模态对应都弱

**证据：** Table 6 中 TIR→TIR 仅 20.98 Rank-1 / 14.00 mAP，远低于 RGB→RGB 和 NIR→NIR；TIR 与 RGB/NIR 的跨模态 Rank-1 多在 8–9 左右。\[R\]

**机制解释：** 这同时指向两类问题：TIR 本身可能缺乏足够稳定、可区分的身份细节；即使 TIR 特征在自身模态内有一定结构，它与 RGB/NIR 的统一空间仍可能有较大偏差。不能假定只强化跨模态对齐就能解决 TIR，因为模态内检索本身已经明显落后。\[I\]

**尚缺证据：** 当前论文没有按 TIR 图像尺度、昼夜、天气、平台、相机或身份分组报告结果，无法把低性能唯一归因于传感器、样本量、图像质量或网络结构。\[U\]

### 11.3 薄弱点 B：场景定义包含平台，但训练机制主要显式处理模态

**代码事实：** camera/view SIE 关闭；模型 forward 不使用 camera IDs 或 modality IDs；ProCA 只产生每个身份的模态中心；sampler 保障每身份/模态的样本数，但不显式平衡 ground/aerial、camera、时间或天气。\[C\]

**机制风险：** 每个模态中心会混合该身份在多个平台、相机和拍摄条件下的特征。若同一模态内部存在多峰分布（例如 ground 与 aerial 的视角差异），单个均值中心可能无法表示这些子分布；相反，强行将它们压到单中心也可能抹去某些有用的视角互补信息。\[I\]

**证据边界：** Table 4 显示 A→G 与 G→A 的 mAP、Rank-1 不对称，支持平台方向差异确实存在；但表格不能证明差异一定来自“缺少平台 embedding”，因为数据分布、query/gallery 数量和难例构成也会影响结果。

### 11.4 薄弱点 C：ProCA 的中心平均可能丢失身份内的多样性

**代码事实：** 每个身份、每个模态只有 K=4 个特征参与当前 batch center；逐样本归一化后求均值并再次归一化；三个模态中心等权平均成身份 anchor；anchor stop-gradient。\[C\]

**潜在风险：**

- K=4 的中心是当前 mini-batch 采样的统计量，不是该身份的全数据中心；采样视角变化会导致中心波动。
- 同一身份可能具有不同平台/相机/尺度子分布，平均后这些差异被压缩成一个方向。
- 三模态等权处理隐含了模态中心同等可靠的假设，但 2337 的 TIR 数据较少且结果更弱。
- ProCA 直接惩罚的是中心之间的距离，不是所有样本对身份中心的距离；个别离群样本仍可能在中心约束下被弱化。

**结果证据：** Table 8 中 ProCA 单独只带来 +0.4 mAP / +0.6 Rank-1；在 GPD 上增加 ProCA，mAP 仅 +0.1，但 Rank-1 +0.8。说明 ProCA 有增益，但在当前实现与设置下不是主要收益来源。\[R\]

**尚不能断言：** 不能据此认定“中心对齐方向无效”；目前只证明当前 ProCA 配置的边际收益较小，不能否定中心对齐的其他形式或作用位置。\[U\]

### 11.5 薄弱点 D：GPD 原型库存在初始化、滞后与状态管理问题

**代码事实：** 原型初始为零；当前 batch 未初始化的正类临时使用 `g_cent`；所有未初始化的其他类别仍是零向量；已初始化原型用旧权重 0.8、新中心权重 0.2 更新；原型不反向传播、不保存在 model checkpoint 中。\[C\]

**潜在风险：**

- 训练早期的全局负类原型集合不完整，零原型产生零 logits，不能提供与真实身份原型相同的负类方向信息；随着训练推进，GPD 的有效负类集合会逐渐变化。
- EMA 有助于稳定原型，但也会滞后于持续更新的 backbone；温度 0.03 使归一化点积 logits 的尺度较大，优化敏感性值得关注。
- 若训练中断后只加载模型 checkpoint 并重新初始化 prototype bank，就无法恢复此前的原型状态；公开代码没有完整续训状态保存机制。
- AMP 若跳过模型参数更新，当前代码仍会更新 bank；实际是否发生、发生频率及其影响没有被报告。

这些属于实现中真实存在的状态机制与潜在风险，不代表已证明它们导致论文结果下降。

### 11.6 薄弱点 E：训练目标与最终检索几何并不完全一致

UAD 同时使用：

- BNNeck 后 logits 的 CE；
- BN 前、未归一化特征的 WRT 欧氏距离；
- 归一化特征中心上的 ProCA；
- 归一化特征与归一化原型上的 GPD；
- 推理时 L2 归一化后的 BN 前 CLS 特征。

这意味着 WRT 对特征模长敏感，而最终排序主要依赖特征方向；CE 的分类边界作用于 BNNeck 空间，而推理不使用 BNNeck 输出。ProCA/GPD 的几何更接近归一化检索空间，WRT/CE 则带来不同的优化信号。

这种设计并非必然错误：BNNeck 分离分类与检索特征、WRT 使用原始特征也是 ReID 常见做法。但在 WHU 这种模态差异大的任务中，各损失对方向、模长、身份内聚合和身份间分离的影响可能不同，存在优化竞争的可能。公开论文没有提供各损失的梯度范数、梯度夹角或模态分组梯度统计，因此目前只能将其列为合理研究问题，不能断言某项损失正在造成梯度冲突。\[I\]\[U\]

### 11.7 薄弱点 F：全局 CLS 单描述子可能压缩细粒度身份线索

**代码事实：** ViT 有 128 个 patch token，但 backbone 最终只返回最后一层 CLS token；推理不拼接 patch、局部 part 或其他层特征。\[C\]

**潜在风险：** 航拍图像中的人体可能只占较少像素，NIR/TIR 中可见的局部纹理也可能较弱。所有身份信息被压缩到一个 768D 全局 token，可能限制局部衣物结构、肢体轮廓或小尺度特征在最终描述子中的保留程度。\[I\]

**证据边界：** 论文没有做 token/part 级消融，也没有按人体像素高度或局部可见性报告结果，所以这是结构层面的潜在限制，而不是已由表格证实的主要性能原因。\[U\]

### 11.8 薄弱点 G：模态均衡采样可能带来重复采样偏差

**代码事实：** 对每个身份，sampler 将各模态抽样数量对齐到该身份模态中的最大数量（按 K 对齐）；较少的模态会通过有放回扩展/重复索引补足。2337 的 TIR 总量显著少于 RGB/NIR。\[C\]\[P\]

**潜在风险：** 这种做法保证每个 batch 都有相同数量的三模态身份样本，但可能增加低样本模态重复出现的频率，降低有效样本多样性。它解决了 batch 内模态缺失问题，却不等同于解决了模态数据分布不平衡问题。

尤其在 2337 中，TIR 的数量少且检索指标弱，值得区分“均衡 sampler 带来的监督公平性”与“重复样本造成的过拟合/有效多样性下降”。目前没有按 PID、模态统计重采样倍率，也没有比较原始样本频率和实际训练频率，故影响程度未确定。\[I\]\[U\]

### 11.9 薄弱点 H：身份级全局原型不显式刻画模态/平台可靠性

GPD 为每个训练身份维护一个 prototype，而不是按身份 × 模态或身份 × 平台维护多个原型。其优势是把各场景压进统一身份方向，参数开销小；风险是同一身份在 RGB/NIR/TIR 或 ground/aerial 中的分布差异只能通过单个 prototype 间接表达。

结合 Table 6 的 TIR 弱项和 Table 4 的方向不对称，可以提出研究问题：单个身份原型是否足以表示异质性较强的身份分布？原型更新是否应被视为对可靠中心的平滑估计，还是会被弱模态/少数平台的噪声拉偏？当前结果没有直接回答这些问题。\[I\]

### 11.10 薄弱点 I：闭集训练身份监督与开放身份检索之间存在泛化落差

CE 与 GPD 都基于训练身份类别；测试 query/gallery 身份不在训练类别中。它们必须通过学习可迁移的身份判别几何来帮助新身份检索，而不是直接分类测试身份。

因此，训练身份上的分类/原型分离改善并不必然等价于测试身份的 mAP、mINP 和完整排序改善。论文 Table 3 从 1000 扩展到 2337 时指标下降，说明更大的身份与场景覆盖仍有难度，但不能仅凭这组跨版本比较判定泛化差距的具体来源。\[I\]

### 11.11 薄弱点 J：损失权重与优化冲突缺少直接诊断证据

总损失系数为 CE=1、WRT=1、ProCA=0.01、GPD=1。不同损失的数值尺度不同，不能简单按系数大小判断梯度贡献；例如 GPD 的温度 0.03 会改变 logits 与梯度尺度，而最终总梯度还会经过全局 norm clipping=1.0。

论文报告损失消融，但没有报告：

- 各项损失对 backbone 的梯度范数；
- 各项梯度之间的方向相似度/冲突程度；
- 不同模态的梯度贡献差异；
- 梯度裁剪触发频率；
- AMP skip step 与 prototype bank 更新的统计。

因此，优化冲突是有代码依据的研究切入点，但当前不能宣称“GPD 梯度压制了 ProCA”或“温度太低导致训练不稳定”。\[U\]

------------------------------------------------------------------------

## 12. 潜在研究切入点清单（只定义问题，不给具体方案）

以下按“证据强度 × 对任务的重要性”排序，目的是帮助未来研究快速定位问题，而不是预先决定使用哪种模块或损失。

| 优先级 | 切入点                         | 已有证据                                                                        | 关键待回答问题                                                                     | 当前证据边界                                                |
|--------|--------------------------------|---------------------------------------------------------------------------------|------------------------------------------------------------------------------------|-------------------------------------------------------------|
| P0     | TIR 的模态内表征与跨模态检索   | Table 6 的 TIR→TIR、TIR→RGB/NIR 均弱；Table 5 TIR→RGB 仍只有个位数 Rank-1       | 性能瓶颈主要来自 TIR 的身份信息不足、模态偏移、样本数量/质量，还是三者共同作用？   | 论文未做 TIR 图像质量、尺度或场景分层分析                   |
| P0     | 训练与推理度量的一致性         | WRT 用未归一化 BN 前特征；推理用 L2 归一化特征；CE 使用 BNNeck 空间             | WRT 对特征模长的优化是否与最终方向排序一致？各损失对最终检索描述子的实际贡献如何？ | 缺少特征范数、梯度与排序变化诊断                            |
| P0     | ProCA 的真实边际贡献与中心质量 | Table 8 中 ProCA 单独增益小，GPD 是主要增益；ProCA 中心基于每模态 K=4 均值      | 中心约束的收益是否受视角混合、弱模态噪声、K 值和 batch 采样波动限制？              | 只有主消融，没有中心质量或多 seed 分析                      |
| P1     | 模态与平台因素的分离           | 任务是 modality × viewpoint；代码关闭 camera/view SIE，ProCA 只按 modality 聚合 | 跨平台差异有多少被模态中心间接吸收，有多少仍作为未建模变化存在？                   | Table 4 方向不对称，但没有完整的 modality×platform 分层结果 |
| P1     | 模态平衡与重复采样的代价       | 2337 TIR 数量更少；sampler 对少样本模态重复采样以匹配每 PID 的最大数量          | batch 内平衡是否提升监督覆盖，是否同时降低 TIR 的有效样本多样性？                  | 未统计每 PID 的重复倍率、有效样本数和采样分布               |
| P1     | Prototype bank 的动态行为      | 原型零初始化、未初始化类 logits 为 0、EMA 更新、checkpoint 不保存 bank          | 原型初始化速度、EMA 滞后和原型覆盖率如何影响 GPD 的训练早期与后期？                | 没有原型覆盖率、原型漂移、resume 或 AMP skip 统计           |
| P1     | 全局 CLS 对局部身份线索的压缩  | 128 patch tokens 最终只取 CLS，所有模态共用同一全局描述子                       | 航拍小目标与 TIR 局部纹理的身份信息是否在 CLS 聚合中被削弱？                       | 没有 token/part 消融或人体尺度分层结果                      |
| P1     | 闭集训练到新身份检索的泛化     | train/test PID 分离；CE 与 GPD 都监督训练身份                                   | 训练身份的全局区分是否能稳定迁移到未见身份，尤其是 TIR/跨平台组合？                | 没有按训练身份与测试身份分布的专门泛化诊断                  |
| P2     | 各损失的梯度相互作用           | 四项损失共享 backbone；全局梯度裁剪 1.0                                         | 哪些损失对哪些模态/层产生互补或冲突梯度？                                          | 没有梯度范数、夹角或层级梯度统计                            |
| P2     | 评测协议完整性与复现可追溯性   | mINP 未在公开评测器实现；1000-GD 筛选被注释；AG 平台协议不在默认入口            | 不同版本、协议与脚本是否真正一致，指标是否可完整复现？                             | 需核对用户本地代码、数据目录与日志                          |
| P2     | 统计稳定性                     | Table 8 的 ProCA-on-GPD mAP 增益仅 0.1 pp；论文未报告多 seed 方差               | 小幅收益是否超出训练随机性，哪些指标稳定改善？                                     | 无多 seed 结果或置信区间                                    |

### 12.1 研究切入点的解释边界

- **“代码没有某个显式机制”不等于模型完全没有学到相应不变性。** 共享 backbone、数据增强、身份分类、WRT、ProCA 和 GPD 都可能间接学习平台/模态不变性；缺少显式条件化只是结构事实。
- **“TIR 指标低”不等于已确定由某个模块造成。** 它证明了性能瓶颈位置，但原因仍需区分数据、表示和优化因素。
- **“ProCA 单独增益小”不等于中心学习方向无价值。** 它只说明当前 ProCA 形式与当前训练配置的边际收益较小。
- **“增加某项训练监督”不保证最终描述子保留该信息。** UAD 本身的 BNNeck/原始 CLS/归一化 CLS 路径就说明：监督在哪个张量上发生，与推理最终使用哪个张量，是必须分别追踪的问题。

------------------------------------------------------------------------

## 13. 官方代码索引：后续阅读从哪里开始

所有链接固定到本次读取的官方提交 `3e2a07314119402586c76096e87d40420894ad30`，避免仓库未来更新后链接悄悄指向不同实现。

| 研究问题                             | 文件/函数                                                                                                                                                                                                                                                                                        | 应关注内容                                                                   |
|--------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| 运行参数和损失开关                   | [`configs/UAD.yml`](https://github.com/msm8976/WHU-MARS/blob/3e2a07314119402586c76096e87d40420894ad30/CVPR26_UAD/configs/UAD.yml)                                                                                                                                                                | PCA 权重、GPD momentum、batch、LR、epoch、输入归一化、test feature           |
| 默认配置与 seed                      | [`config/defaults.py`](https://github.com/msm8976/WHU-MARS/blob/3e2a07314119402586c76096e87d40420894ad30/CVPR26_UAD/config/defaults.py)                                                                                                                                                          | 默认 seed、optimizer、SIE、TOP_K_EVAL 等                                     |
| ViT patch/CLS/位置编码               | [`model/backbones/vit_pytorch.py`](https://github.com/msm8976/WHU-MARS/blob/3e2a07314119402586c76096e87d40420894ad30/CVPR26_UAD/model/backbones/vit_pytorch.py)                                                                                                                                  | `PatchEmbed`、`TransReID.forward_features`、`load_param`、`resize_pos_embed` |
| BNNeck、classifier、特征输出         | [`model/make_model.py`](https://github.com/msm8976/WHU-MARS/blob/3e2a07314119402586c76096e87d40420894ad30/CVPR26_UAD/model/make_model.py)                                                                                                                                                        | `build_transformer.__init__`、`forward`、`load_param`                        |
| 多模态采样                           | [`datasets/sampler.py`](https://github.com/msm8976/WHU-MARS/blob/3e2a07314119402586c76096e87d40420894ad30/CVPR26_UAD/datasets/sampler.py)、[`sampler_ddp.py`](https://github.com/msm8976/WHU-MARS/blob/3e2a07314119402586c76096e87d40420894ad30/CVPR26_UAD/datasets/sampler_ddp.py)              | PID 分组、K=4、tuple index、DDP 分块、重复采样                               |
| transforms 与 batch 组装             | [`datasets/make_dataloader.py`](https://github.com/msm8976/WHU-MARS/blob/3e2a07314119402586c76096e87d40420894ad30/CVPR26_UAD/datasets/make_dataloader.py)、[`datasets/bases.py`](https://github.com/msm8976/WHU-MARS/blob/3e2a07314119402586c76096e87d40420894ad30/CVPR26_UAD/datasets/bases.py) | transforms、collate、图像读取、三模态 batch 的形状                           |
| PID/camera/modality 解析             | [`datasets/whu_mars.py`](https://github.com/msm8976/WHU-MARS/blob/3e2a07314119402586c76096e87d40420894ad30/CVPR26_UAD/datasets/whu_mars.py)                                                                                                                                                      | 文件名解析、PID relabel、camid 减 1、1000-GD 注释筛选                        |
| WRT 细节                             | [`loss/wrt_loss.py`](https://github.com/msm8976/WHU-MARS/blob/3e2a07314119402586c76096e87d40420894ad30/CVPR26_UAD/loss/wrt_loss.py)                                                                                                                                                              | 正负样本权重、距离矩阵、SoftMarginLoss                                       |
| CE 与基础损失组合                    | [`loss/make_loss.py`](https://github.com/msm8976/WHU-MARS/blob/3e2a07314119402586c76096e87d40420894ad30/CVPR26_UAD/loss/make_loss.py)、[`softmax_loss.py`](https://github.com/msm8976/WHU-MARS/blob/3e2a07314119402586c76096e87d40420894ad30/CVPR26_UAD/loss/softmax_loss.py)                    | label smoothing 开关、CE/WRT 的输入特征                                      |
| ProCA、prototype bank、GPD、训练循环 | [`processor/processor.py`](https://github.com/msm8976/WHU-MARS/blob/3e2a07314119402586c76096e87d40420894ad30/CVPR26_UAD/processor/processor.py)                                                                                                                                                  | `PCA`、`PrototypeBank.update`、`GPD`、`do_train`、`do_inference`             |
| optimizer 参数组                     | [`solver/make_optimizer.py`](https://github.com/msm8976/WHU-MARS/blob/3e2a07314119402586c76096e87d40420894ad30/CVPR26_UAD/solver/make_optimizer.py)                                                                                                                                              | SGD、bias LR、weight decay、未使用的 center optimizer                        |
| LR warm-up/cosine                    | [`solver/scheduler_factory.py`](https://github.com/msm8976/WHU-MARS/blob/3e2a07314119402586c76096e87d40420894ad30/CVPR26_UAD/solver/scheduler_factory.py)、[`cosine_lr.py`](https://github.com/msm8976/WHU-MARS/blob/3e2a07314119402586c76096e87d40420894ad30/CVPR26_UAD/solver/cosine_lr.py)    | warm-up 初值、epoch 起点、cosine min LR                                      |
| 主训练入口                           | [`train.py`](https://github.com/msm8976/WHU-MARS/blob/3e2a07314119402586c76096e87d40420894ad30/CVPR26_UAD/train.py)                                                                                                                                                                              | seed、DDP 初始化、配置覆盖、调用链                                           |
| 评测与 3×3 模态对                    | [`utils/metrics.py`](https://github.com/msm8976/WHU-MARS/blob/3e2a07314119402586c76096e87d40420894ad30/CVPR26_UAD/utils/metrics.py)                                                                                                                                                              | camera filter、CMC/mAP、feature norm、modality-pair evaluation、mINP 缺失    |
| 官方训练命令                         | [`CVPR26_UAD/README.md`](https://github.com/msm8976/WHU-MARS/blob/3e2a07314119402586c76096e87d40420894ad30/CVPR26_UAD/README.md)                                                                                                                                                                 | 四卡 batch/LR override 与默认 YAML 的区别                                    |

------------------------------------------------------------------------

## 14. 后续复用时的核对清单

以下是阅读和分析的核对顺序，不是实验方案：

1.  **先确认实际代码版本。** 官方 `main`、本地 `WHU`、服务器 `WHU_DDP` 是否一致；记录 commit 或代码目录快照。
2.  **先确认数据版本和 split。** 用实际文件数/manifest 核对 1000、1000-GD、2337；不要直接把论文统计、README 统计和本地数据视为相同。
3.  **从 sampler 反推 batch 语义。** 明确 `IMS_PER_BATCH` 是 tuple 数还是原始图片数，记录 world size、每卡 tuple、每卡三模态图像总数、P/M/K。
4.  **明确模型训练和推理各自返回什么张量。** 记录 CE 的 BNNeck 特征、WRT/ProCA/GPD 的输入、最终 `NECK_FEAT` 和 `FEAT_NORM`。
5.  **逐项检查损失的实际调用。** 只看配置名不够；要确认损失进入 `loss` 总和、梯度回到哪些模块、是否有 detach/no_grad、是否真正执行 optimizer step。
6.  **单独追踪 prototype bank。** 记录初始化、首次有效更新、EMA 系数、DDP 同步、AMP skip 时更新行为，以及 checkpoint 是否保存 bank。
7.  **评测协议与指标逐项对齐。** 同 camera filter、是否 rerank、全 gallery 还是 top-k、是否计算 mINP、是否有 platform-pair 子集、query/gallery 是否有重复文件。
8.  **解释指标时不只看 Rank-1。** 同时看 mAP、Rank-5/10、mINP、3×3 模态对和 A→G/G→A 方向差异；第一名命中改善不等于整体相关样本排序改善。
9.  **区分证据和假设。** 代码事实可以直接陈述；性能原因需要由消融、分组结果、特征分析或梯度证据支持；缺少证据的内容保留为待研究问题。
10. **不要重复确认已稳定的公共机制。** 本文作为公共 UAD 机制底稿；后续实验记录应重点保存相对 baseline 的真实代码差异、独特机制、结果和复盘，而不是复制本文全部内容。

------------------------------------------------------------------------

## 15. 当前未决项与证据缺口

1.  **用户当前本地/服务器代码版本未检查。** 本文只核对官方公开仓库快照；用户的 `WHU_DDP` 可能有复现改动，需对照 Git diff/文件版本后才能断言实际训练行为完全一致。
2.  **实际 WHU-MARS 数据文件未访问。** 无法核实最新 1000 版本统计究竟应为 185,742 还是论文中的 185,922，也无法检查 query 与 test 文件是否有重复图像。
3.  **官方补充材料未能读取。** CVF supplementary 链接在当前环境返回 403；设备规格、详细标注规则、额外数据统计未在本文中扩展分析。
4.  **没有模型权重和训练日志。** 无法统计各损失梯度范数、原型库覆盖率、特征模长、AMP skip step、每身份重采样倍率、各层 token 贡献或多随机种子方差。
5.  **公开评测器缺少 mINP 和平台对评测。** 若本地版本已经补充这些功能，需要以本地实现为准；本文不把公开仓库缺失直接外推为用户版本缺失。
6.  **论文表格只提供总体结果。** 缺少按相机、平台、昼夜、季节、天气、人体像素尺度、TIR 图像质量分层的结果，当前不能定量拆分每一种域偏移的贡献。

------------------------------------------------------------------------

## 16. 最终归纳

UAD 的核心设计可以概括为：**共享 ViT-B/16 学习单图像 CLS 描述子；CE 与 WRT 提供基础身份分类和 batch 内度量监督；ProCA 通过模态中心减小同身份跨模态中心差异；GPD 通过训练身份 prototype memory 增加全局身份分离；推理只保留 L2 归一化的 BN 前 768D CLS 特征。**

原论文结果证明这套机制能显著超过其内部 baseline，并在 AS-ReID、AG-ReID 和部分 VI-ReID 方向上取得竞争力；消融尤其表明 GPD 是主要性能增益来源。但结果也清楚显示：TIR 的模态内与跨模态检索仍然很弱，2337 上性能明显下降，ProCA 的边际增益有限，平台方向结果不对称。

从模型自身看，最有研究价值的缺口集中在：**弱模态表征、模态与平台因素的区分、中心统计量的可靠性、归一化检索空间与训练损失几何的关系、全局原型的动态行为、采样中的模态重复，以及评测协议完整性**。这些问题都能从论文图表或代码路径找到明确依据，但其各自对最终指标的因果贡献仍需通过有针对性的证据来区分。

本文不预设某种新模块一定有效，也不把任何一个潜在切入点直接写成改进方案。后续研究应以本文为公共机制底稿，先确认实际代码/数据版本，再围绕真实瓶颈提出具体假设，避免重复阅读和重复推理。
