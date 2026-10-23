# P5 区域读出增量与空间机制检验报告

**状态：COMPLETE。任务 A、任务 B 均完成；新增训练 0，新外部图像推理 0。**
完成时间：2026-10-16T14:23:27+08:00。执行窗口：2026-10-16T14:06:55+08:00 至 2026-10-16T16:06:55+08:00。
本轮唯一新增目录：`${R52_PATH_99FA50DA40CD}`。

## 一、中文结论：事实与推断分开

### 已观察到的事实

- 任务 A 直接以只读 mmap 复用 P3/P4 各自保存的 10,000 次配对抽样指标，没有重抽样、重跑外部模型或调用旧阶段总入口。数组轴顺序由阶段源码、常量、身份清单、完成记录及原 R_CLIN 区间复现共同核对。
- 追加比较固定为 R_IMG−M0、R_IMG−XATTN、R_IMG−B2S，两队列各自分析；所有输入条件、预测头和 AP/AUC 结果完整保留。
- 六项重点 AP 的 99.166667% 个体参考区间中，跨越或包含 0 的数量为 6/6。该结果不能改写原 P3/P4 的预设主要比较。
- 任务 B 使用原 6 个 R_IMG/R_CLIN selected-best，在相同的 1,513 张 HAM validation 上完成 NATIVE、SHIFT_HALF、PERMUTE_FIXED。所有条件保留，未依据任务 A 的区间方向决定是否执行机制检验。
- 空间干预确实改变了模型输出：softmax MEL 概率的逐 run 平均绝对变化范围为 0.000184 至 0.011150。但四项三 seed 平均 NATIVE−干预 softmax AP 差值中，仅 1/4 为正，并非所有干预都降低 AP。
- NATIVE 包装与原完整 forward 的前 8 图最大绝对误差为 0；identity 排列误差为 0。各区域质量最大相对误差 3.34709711e-07，低于 1e-5；valid 标记逐批一致。模型参数及全部缓冲区前后状态指纹一致。

### 解释边界

任务 A 是看过前阶段结果后指定的追加比较。六比较近似 Bonferroni 参考只处理本轮指定六项重点 AP 的多重比较，不消除此前模型、方法或比较的选择影响，不能称为原预设确认性主要终点。
任务 B 只描述固定模型在既有验证集上的空间依赖性。平移或排列导致的读出分布变化，不等于独立定位精度、病灶因果解释或临床相关性；结果也不是新的外部性能验证。验证集已参与原选模，因此不添加确认性显著性标签。
本次观察支持“这些固定模型的输出会响应区域与视觉特征的空间重排”，但不支持把本检验写成“原生区域对应在各模型和两种干预下均提供稳定的 AP 优势”。

## 二、任务 A：追加区域读出比较

点估计来自各阶段 PER_RUN_METRICS.csv。先计算每个 training seed 的 R_IMG−参照差值，再求三个 seed 的均值与样本 SD(ddof=1)。每次 bootstrap 同样先配对求差再平均，而非相减两条区间端点。
SIIM 使用原患者聚类抽样，PCG64 seed20261016、患者字符串字典序；Derm7pt 使用原病例/同图簇抽样，PCG64 seed20261017、数值 case 代表排序，不称为已核实患者级区间。区间条件于三个固定训练运行，训练 seed 变异另列。
数组维度均为 (10000,3,5,4,2)：draw、seed、model、input/head、metric。seed 顺序 42/2024/3407；model 顺序 M0/R_IMG/R_CLIN/XATTN/B2S；metric 顺序 AUC/AP。unit0/1 为 E_REAL（SIIM）或 E_AVAILABLE（Derm7pt）的 softmax_MEL/endpoint_MEL；unit2/3 为 E_NULL 对应双头。
完整区间表：REGION_CONTRASTS.csv；逐 seed 差值：REGION_CONTRASTS_PER_SEED.csv。重点实际输入 softmax MEL AP 如下：

| 队列 | 追加比较 | AP 差值均值 | seed 差值 SD | 95% 区间 | 六比较参考区间（99.166667%） |
|---|---|---:|---:|---|---|
| SIIM | R_IMG_minus_M0 | 0.012253 | 0.007095 | [0.002628, 0.021377] | [-0.001282, 0.025074] |
| SIIM | R_IMG_minus_XATTN | 0.005809 | 0.004633 | [-0.004737, 0.014826] | [-0.008732, 0.017881] |
| SIIM | R_IMG_minus_B2S | 0.001976 | 0.015273 | [-0.007931, 0.011466] | [-0.011856, 0.014555] |
| Derm7pt | R_IMG_minus_M0 | 0.014553 | 0.069363 | [-0.009505, 0.038220] | [-0.017130, 0.046003] |
| Derm7pt | R_IMG_minus_XATTN | 0.001566 | 0.040911 | [-0.021781, 0.025292] | [-0.029994, 0.034956] |
| Derm7pt | R_IMG_minus_B2S | 0.002158 | 0.070530 | [-0.022518, 0.027855] | [-0.030109, 0.036520] |

六比较参考覆盖率 = 1−0.05/6；分位点 = 0.05/(2×6) 与 1−0.05/(2×6)。AUC、E_NULL 及另一预测头均保留 95% 描述区间；未据结果选择头或条件。

### 实际可用输入 softmax MEL AUC

| 队列 | 追加比较 | AUC 差值均值 | seed 差值 SD | 95% 区间 |
|---|---|---:|---:|---|
| SIIM | R_IMG_minus_M0 | 0.017850 | 0.009074 | [0.008444, 0.027091] |
| SIIM | R_IMG_minus_XATTN | 0.016224 | 0.026425 | [0.006834, 0.025358] |
| SIIM | R_IMG_minus_B2S | 0.015526 | 0.032646 | [0.005264, 0.025721] |
| Derm7pt | R_IMG_minus_M0 | 0.009962 | 0.037868 | [-0.002091, 0.021934] |
| Derm7pt | R_IMG_minus_XATTN | 0.011574 | 0.026036 | [-0.002418, 0.025365] |
| Derm7pt | R_IMG_minus_B2S | 0.008569 | 0.033691 | [-0.005951, 0.023094] |

### 原 R_CLIN 主要比较并列引用，不替换

| 原阶段队列 | 原主要 AP 比较 | 原差值 | 原 97.5% 区间 |
|---|---|---:|---|
| SIIM | R_CLIN_minus_M0 | 0.016882 | [0.006884, 0.027192] |
| SIIM | R_CLIN_minus_XATTN | 0.010438 | [-0.001618, 0.021344] |
| Derm7pt | R_CLIN_minus_M0 | -0.006887 | [-0.030510, 0.015152] |
| Derm7pt | R_CLIN_minus_XATTN | -0.019874 | [-0.045113, 0.005532] |

ORIGINAL_RCLIN_COMPARISONS.csv 引用完整原比较。M0 始终是当前 FIM FULL；B2S 是同协议简化系统参照，不是历史 B2、clean_B2 或单因素消融。未拼接队列、未平均病例概率构造集成，也未跨队列相减绝对 AP。
Brier/NLL/ECE 与固定阈值表现只整理原始表：ORIGINAL_CALIBRATION_METRICS.csv、ORIGINAL_FIXED_THRESHOLD_PERFORMANCE.csv；其三 seed 描述汇总为 ORIGINAL_DESCRIPTIVE_SUMMARY.csv。Brier/NLL/ECE 越低越好；相同开发政策下 sensitivity/specificity/PPV/NPV 越高越好，但不同模型的既有数值阈值可能不同。未重拟合、重选阈值或改变原外部测试。

## 三、任务 B：验证集固定空间干预

严格加载 P3 清单中的 6 个原 best，保持网络结构和权重。RGB 短边256、中心裁剪224、ImageNet归一化；FP32、eval/inference_mode、TF32关闭、batch8、workers2。临床输入始终是原 NATIVE 22维编码，标签及图像 ID 不进入模型。
干预发生在 region_tokens 的原 resize/sigmoid/replicate-padding 3×3 软形态操作之后。interior/boundary/context 三张完整权重图共同循环平移 (H//2,W//2)，或共同按固定空间索引 gather；图像、fused视觉特征、原全局/注意力池化、临床向量和区域通道顺序不变。
固定排列规则：对确切 image_id 的 UTF-8 SHA-256 取前8字节，以 little-endian 无符号整数得到 uid；PCG64(SeedSequence([20261018,uid])) 产生 H×W 的 permutation。同图在全部模型和 seed 下使用同一规则，没有尝试其他位移、排列种子或最不利扰动。
每批只计算一次骨干、fused特征、预测 mask logits、原池化和临床 embedding；然后重新计算区域 tokens、alpha、残差、原 CMGF 与双头。alpha 随 token 自然响应，没有锁为原值；没有磁盘高维特征缓存。
推理耗时 132.7 秒。6×1513×3 条件全部完成；双头指标36行、三 seed 描述汇总120行。NATIVE 与 P3 已保存验证预测另做只读复现核对，未重新推理原阶段。

### Softmax MEL 与七分类结果

| 模型 | 条件 | AP 均值 ± SD | AUC 均值 ± SD | 七分类 macro-F1 均值 ± SD |
|---|---|---:|---:|---:|
| R_IMG | NATIVE | 0.698820 ± 0.024303 | 0.926861 ± 0.002888 | 0.784284 ± 0.005980 |
| R_IMG | SHIFT_HALF | 0.701787 ± 0.019890 | 0.926356 ± 0.003779 | 0.781256 ± 0.007217 |
| R_IMG | PERMUTE_FIXED | 0.700341 ± 0.019918 | 0.924209 ± 0.006071 | 0.778334 ± 0.010214 |
| R_CLIN | NATIVE | 0.703386 ± 0.026216 | 0.923895 ± 0.004812 | 0.790420 ± 0.008158 |
| R_CLIN | SHIFT_HALF | 0.701153 ± 0.026615 | 0.925808 ± 0.006045 | 0.792234 ± 0.005056 |
| R_CLIN | PERMUTE_FIXED | 0.705913 ± 0.022646 | 0.926449 ± 0.005653 | 0.788029 ± 0.005567 |

### NATIVE−干预差值，描述性而非确认性检验

| 模型 | 对比 | Softmax AP 差值 ± SD | 专用头 AP 差值 ± SD | 七分类 macro-F1 差值 ± SD |
|---|---|---:|---:|---:|
| R_IMG | NATIVE_minus_SHIFT_HALF | -0.002968 ± 0.005672 | -0.001599 ± 0.003147 | 0.003028 ± 0.002083 |
| R_IMG | NATIVE_minus_PERMUTE_FIXED | -0.001522 ± 0.005089 | -0.003711 ± 0.010046 | 0.005949 ± 0.004829 |
| R_CLIN | NATIVE_minus_SHIFT_HALF | 0.002233 ± 0.001378 | 0.005716 ± 0.011151 | -0.001814 ± 0.007769 |
| R_CLIN | NATIVE_minus_PERMUTE_FIXED | -0.002527 ± 0.005332 | 0.005219 ± 0.010897 | 0.002391 ± 0.010719 |

双头 AP/AUC、binary Brier/NLL/ECE 和七分类 macro-F1 的逐 run 结果见 VALIDATION_SPATIAL_INTERVENTION.csv。SPATIAL_INTERVENTION_SUMMARY.csv 包括各条件与两个配对差值的均值、样本 SD；逐 seed 差值另列。AP/AUC/F1 的正 NATIVE−干预差值有利于 NATIVE；Brier/NLL/ECE 的负差值才有利于 NATIVE。
REGION_SPATIAL_DIAGNOSTICS.csv 记录区域面积、valid比例、alpha、残差系数和质量误差；MODEL_OUTPUT_CHANGES.csv 记录概率/logit变化及七分类argmax变化比例；DIAGNOSTIC_THREE_SEED_SUMMARY.csv 给对应三 seed 汇总。区域顺序0/1/2分别为interior/boundary/context；region_weights是alpha，region_area是质量/(H×W)，region_valid是有效比例，residual_coefficient是有界残差系数。

## 四、可用于论文的发现与不可延伸的结论

可直接陈述的事实：本轮追加了不依赖新外部推理的 R_IMG 直接比较，并以完全固定模型在既有 HAM 验证集上检验了区域权重与视觉特征的空间对应；完整呈现三个训练 seed、两个头、全部预定条件。
关于增量：应逐队列引用上表的配对效应及区间。不能仅凭正点估计或某个未经调整的区间，把 R_IMG 写成普遍优于所有参照；也不能把六比较参考区间写成消除了事后选择偏差。
关于机制：任何输出变化只能说明该空间对应操作影响了固定模型的读出；任何指标下降或上升都应按实测值陈述。它不能独立证明病灶定位正确、临床因果机制成立，或把验证集诊断升级为新的外部有效性证据。没有生成独立定位 Dice、显著性标签或新主要终点。

## 五、实际文件与完整性

- P3只读来源：`${R52_PATH_2E90ACAFEFFB}`。
- P4只读来源：`${R52_PATH_12DFAC0CCEF0}`。
- 新输出目录：`${R52_PATH_99FA50DA40CD}`。
- P5_FROZEN_PROTOCOL.json 冻结追加比较、空间操作、窗口、模型及原始输入路径；P5_CODE_FREEZE.json 记录执行代码指纹。
- SIIM_BOOTSTRAP_REUSE_CHECK.json、Derm7pt_BOOTSTRAP_REUSE_CHECK.json 包含源文件行号、轴顺序、SHA-256、完成记录及原区间复现。
- FIXED_MODEL_IDENTITIES.json 与各 MODEL_IDENTITY/TECHNICAL_CHECK/INFERENCE_SUMMARY 文件给出 best 身份、实际导入路径及逐模型状态不变记录。
- FINAL_INTEGRITY.json 记录独立结果核对；PACKAGE_CONTENTS_SHA256.csv 位于ZIP内；ZIP_VERIFICATION.json 记录最终压缩包读取与成员哈希检查。
- 报告与最终 P5_PROGRESS.json 一同收入结果ZIP，且与目录中的对应文件逐字节一致。
- ZIP不含权重、原始图像、逐例预测、完整临床资料或凭据；私有预测只保留在本轮 private_predictions 中。
- 原稿、P1–P4、共享源码/环境和其他队列保持不变；没有云端调用、新训练、best重选、外部新推理、阈值选择或校准拟合。

本轮交付后停止，不自动启动新候选或下一轮训练。
