# B2_REPORT

生成时间：2026-10-12T05:31:36.711410+08:00。状态：**COMPLETE**；最终输出复核：**PASS**。

“阶段B2”是本轮任务名；表中“模型B2”指历史CMGF模型，二者不混用。

## 1. 实际执行范围

- 六个历史固定模型：B2/B6，各 training seeds 42、2024、3407，原 split seed42。没有选择性省略任何运行或预测头。
- SIIM：33,126 图、2,056 名患者、584 张阳性图；各模型均产生 E_NULL/E_REAL 两种输入，共12个外部条件单元。
- HAM：原 validation 1,513 图、internal-test 1,520 图，六个模型各以 NATIVE 临床输入推理，共12个内部条件单元；未推理训练集。
- 保存18个逐例预测文件，共415,710行；配对条件保存在同一 SIIM 文件。文件、患者映射、22维临床向量与 bootstrap 数组只在本地 `local_only/`，不交回ZIP。
- 已实际执行固定权重加载、GPU推理、CPU统计、开发集阈值选择，以及不改变预测的校准截距/斜率点值评价。没有训练、微调、云连接、恢复队列、依赖安装、预训练权重下载或 C/D 任务；没有重跑B1。
- 原论文、原图、原代码、原配置、原数据和原结果保持不变。`FINAL_QC.json` 重核所用源文件及六个权重 SHA；这是本任务所需输入范围，不是全项目新审计。
- 这是历史模型在已使用过的SIIM队列上的新输入条件评价，不是新的未见队列验证。此前辅助100-ID名单未恢复的问题没有被B2解决；本轮未重查或重建该名单。

## 2. 权重、设备与技术门槛

设备：NVIDIA GeForce RTX 4060 Laptop GPU；Python `3.12.14`，torch `2.8.0+cu128`，torchvision `0.23.0+cu128`，timm `1.0.27`，CUDA runtime `12.8`。环境完整记录见 `ENVIRONMENT.json`、`STATISTICAL_ENVIRONMENT.json`、`FINAL_DEVICE_QUERY.json`。

固定 FP32、TF32关闭、无autocast、eval、inference_mode；构造时只关闭 pretrained 自动下载，随后完整 strict=True 加载固定 state_dict。六个历史checkpoint内容 SHA 均与导航记录相符，没有缺失层、随机补层或键前缀转换。受限加载为 weights_only=True，仅明确允许 PyTorch 的 TorchVersion 类型，未启用任意pickle。

首次严格配置检查发现 checkpoint 比 sidecar 多一个派生字段 `data.metadata.metadata_dim=22`。已核对为唯一数据配置差异，比较时只在内存中补入该字段后逐项等同；没有更改原配置或权重。首次停止记录留在输出目录，随后六项技术检查全部通过，此项不是未解决的科学差异。

RGB → 短边256 → 中心裁剪224 → ImageNet normalize；沿实际 `build_transforms`，Resize 为 BILINEAR、antialias=True。完整变换 repr 在 `technical/*.json`。临床张量显式传入，未借用缺省 metadata 分支。

| 模型 | training seed | checkpoint记录epoch | 选定batch | 特征复用最大logit差 | 小批量端到端图/秒 |
|---|---:|---:|---:|---:|---:|
| B2 | 42 | 41 | 8 | 0 | 86.46 |
| B2 | 2024 | 31 | 32 | 0 | 135.54 |
| B2 | 3407 | 35 | 32 | 0 | 131.44 |
| B6 | 42 | 22 | 8 | 0 | 114.20 |
| B6 | 2024 | 32 | 32 | 0 | 129.10 |
| B6 | 3407 | 18 | 32 | 0 | 131.44 |

固定原validation字典序前96图做技术检查，不用外部标签选批量。六个模型两条件两头的完整forward与视觉特征复用均通过 atol1e-5/rtol1e-4 核验；最大实际差为0。batch8/16/32仅按可用内存与技术耗时选择，两条件共用批量。这些吞吐仅用于本轮运行窗口检查，不是新增论文效率基准。

推理前整轮估计 1.26 小时，未超过4小时门槛。实际全量推理 42.61 分钟；10,000次 bootstrap 2.14 分钟；从协议冻结至本报告约 51.03 分钟（包含检查、实现与整理，不等同纯GPU时间）。每次使用都检查图像SHA并解码，共216,954个图像使用记录，未跳过失败图像。

## 3. 预先冻结的真实临床输入

`metadata_mapping.json` 与 `B2_FROZEN_PROTOCOL.json` 在正式 SIIM 推理前保存；`ANALYSIS_CODE_FREEZE.json` 在任何性能统计前固定脚本和模型副本的SHA。没有看AP再挑映射、seed或预测头。

- age_approx → age/100：33,058条有限值，68条原缺失/不可解析/非有限值；有限年龄超出0–120的数量为0，未裁剪或目标队列填均值。
- sex：33,061条原词表匹配，65条原missing/unknown。
- anatomical site：13,400条直接匹配，16,845条按预设 `torso → trunk` 同义映射；2,354条不支持的非空值保留原encoder语义（unknown one-hot但missing flag=0）；527条真实missing/unknown令flag=1。
- 不支持值是 head/neck、palms/soles、oral/genital；未猜成更细部位。逐类别计数和最终token见 `METADATA_TOKEN_MAPPING.csv`。
- 22维词表、年龄缩放和三个missing flags固定。E_NULL是年龄0、sex/site unknown one-hot、三个flags全1，不是全零向量。
- 本次 E_REAL 映射是新指定的输入评价政策，不是历史SIIM已经使用这些字段的证明。patient_id、target、图像ID、诊断未进入模型输入。

## 4. 两项主要比较

共同头为七分类softmax的MEL索引4（`prob_mel`），AP是非插值 average_precision，不是PR曲线梯形积分。差值统一定义 **E_REAL−E_NULL**。以下均为三个固定运行的指标差均值，不是把三模型概率集成为一个模型。

| 模型 | NULL AP均值 | REAL AP均值 | AP均值差 | 95%患者聚类区间 | 97.5%单项区间（两主要比较） | seed差值样本SD |
|---|---:|---:|---:|---|---|---:|
| B2 | 0.129588 | 0.128601 | -0.000986 | [-0.008854, +0.005681] | [-0.009882, +0.006617] | 0.005390 |
| B6 | 0.124047 | 0.132784 | +0.008736 | [+0.003315, +0.014443] | [+0.002574, +0.015199] | 0.003686 |

- B2 的共同头 AP 均值差为 -0.000986，97.5% 区间包含0。
- B6 的共同头 AP 均值差为 +0.008736，97.5% 区间全在0以上。

全部10,000次有效患者聚类配对抽样使用 PCG64 seed20261013；按原patient_id字符串字典序抽2,056个患者、有放回，以患者重复次数作为其全部图像的权重；每次抽样在模型、seed、条件、head间共享。单类别抽样重抽，共尝试 10000 次、排除 0 次。两个主要AP比较的97.5%单项percentile区间是预设Bonferroni近似处理；没有新增主要比较或P值。

区间只反映患者抽样、条件于这三个固定训练运行；seed SD单列，未重抽训练seed。不据此宣称训练稳定性、AP等效、临床获益或跨协议模型优越。包含0不等于证明无效应。Brier越低越好；AP/AUC越高越好，所有差值仍保留相同REAL−NULL方向。

## 5. 完整支持性结果

| 模型 | score | 指标 | NULL三运行均值 | REAL三运行均值 | 均值差 | 95%区间 |
|---|---|---|---:|---:|---:|---|
| B2 | prob_mel | roc_auc | 0.764782 | 0.772113 | +0.007330 | [-0.000640, +0.015070] |
| B2 | prob_mel | brier | 0.018606 | 0.022083 | +0.003476 | [+0.002882, +0.004097] |
| B2 | prob_mel_endpoint | roc_auc | 0.765434 | 0.772535 | +0.007100 | [+0.000559, +0.013680] |
| B2 | prob_mel_endpoint | average_precision | 0.132244 | 0.129158 | -0.003086 | [-0.010122, +0.003209] |
| B2 | prob_mel_endpoint | brier | 0.018672 | 0.022431 | +0.003760 | [+0.003137, +0.004406] |
| B6 | prob_mel | roc_auc | 0.787292 | 0.794012 | +0.006721 | [+0.000173, +0.013272] |
| B6 | prob_mel | brier | 0.018611 | 0.020444 | +0.001833 | [+0.001354, +0.002297] |
| B6 | prob_mel_endpoint | roc_auc | 0.790273 | 0.796631 | +0.006358 | [+0.001134, +0.011602] |
| B6 | prob_mel_endpoint | average_precision | 0.127255 | 0.133929 | +0.006674 | [+0.001912, +0.011379] |
| B6 | prob_mel_endpoint | brier | 0.018545 | 0.020731 | +0.002186 | [+0.001689, +0.002684] |

- `B2_METRICS.csv`：48行，全部模型/seed/队列/条件/两个头的AP、AUC、Brier、binary NLL、15等宽ECE及校准点值。
- `THREE_RUN_CONDITION_SUMMARY.csv`：40行，条件均值、样本SD(ddof=1)、最小与最大值；专用 sigmoid melanoma head 始终单列。
- `PAIRED_POINT_DIFFERENCES.csv`：60行全部差值；`PAIRED_PATIENT_CLUSTER_CI.csv`：36行AP/AUC/Brier逐运行区间；`FIXED_THREE_RUN_MEAN_CI.csv`：12行固定三运行均值差区间。
- NLL/logit只用固定epsilon1e-7；AP/AUC用原概率。校准为无penalty binomial logit截距+斜率的单次评价，不生成或使用校准后概率，不bootstrap拟合。状态计数：`{"converged_unpenalized_binomial_MLE_evaluation_only": 48}`。NLL、ECE、校准参数只给点值，未额外扩大统计范围。
- `ROC_COORDINATES.csv`、`PR_COORDINATES.csv`含24个SIIM score向量的全坐标（drop_intermediate=False），PR横轴recall、纵轴precision；PR最后一点无对应threshold，ROC首点threshold=inf是标准端点。没有制作/替换论文图片。
- `RELIABILITY_BINS.csv`：720行，15等宽bins；最后一bin包含p=1，空bin均值/事件率留空，不填0。

## 6. 开发集阈值及迁移

每模型/seed/score在本轮HAM validation NATIVE上选择满足sensitivity>=0.90（主政策）或>=0.95（次政策）的最大specificity阈值，同分取最大数值阈值，p>=threshold为阳性。`DEVELOPMENT_THRESHOLDS.csv`保存24个阈值及验证计数；本轮实际范围为 [1.8361688e-15, 0.0016369474]。

同一阈值用于本模型/score的HAM internal-test、SIIM E_NULL与E_REAL，没有在SIIM或test重新选择，没有调用all6旧阈值。`THRESHOLD_CONFUSION_COUNTS.csv`96行完整保留TP/FN/FP/TN、敏感度、特异度、PPV、NPV及阳性率；分母为0者留空。

`THRESHOLD_ABSOLUTE_CI.csv`（96行）、`THRESHOLD_PAIRED_CI.csv`（48行）和`THRESHOLD_FIXED_THREE_RUN_CI.csv`（16行）使用上述同一患者抽样，条件于已选开发阈值。它们不包含重新开发阈值的不确定性。24个阈值通过独立穷举复核，96组计数独立核对一致。

## 7. 历史对应及来源

旧NULL只用于12项概率差诊断：`HISTORICAL_NULL_DIAGNOSTIC.csv`。本轮NULL与历史保存概率的最大绝对差范围为 [0.00065281987, 0.0021190643]；数值差可能包含历史/本次执行精度、库版本、批量等环境差异，本轮没有用该诊断归因或调参。主成对效应始终是本轮REAL减本轮NULL，绝不用旧NULL减新REAL。

- `historical_encoder_reference`: `${R52_PATH_56265031EB7F}`; SHA-256 `db399b81dbed64eb56130f27f1936089808872260a50fd345ce909f1ed1685e6`.
- `SIIM_original_mapping`: `${R52_PATH_83C520ED9A14}`; SHA-256 `a1c7c97c59e5f49232dfe3b6f6817efaaee5dbf9bcb4ce584c1889e641be5c4e`.
- `original_HAM_val`: `${R52_PATH_A7EDEF14667C}`; SHA-256 `b1e4982d61df694d6acc64a5c024f9423a110d87489a2731c5e361f183caf4f3`.
- `original_HAM_test`: `${R52_PATH_1E005046B878}`; SHA-256 `e0805420739e57e029e95ac399b449ba9b91cd1d91226fc9033f18c3cda5cd48`.

其余六权重、六原配置、十二历史HAM身份参照及恢复图像manifest的完整绝对路径与SHA见 `INPUT_MANIFEST.csv`。`original_configs/`是原配置逐字节副本。`MODEL_SOURCE_MANIFEST.csv`给原始实现路径和复制SHA；只有models包initializer新建为空以隔离无关注册器，模型实现本身原样复制。`METHOD_CODE_REFERENCES.csv`/`CODE_LINE_INDEX.csv`定位实际函数及行号：`load`严格加载、`forward_conditions`成对输入、`encode_metadata`原编码、`threshold`开发阈值、`point`指标/校准、`bootstrap`固定患者抽样。

## 8. 后续研究安全交接与缺项

`${R52_PATH_D5D08D4FE43D}` SHA `95c99504904d41e39f1466c36e7cd3a44c2aa9d7221eacf3d265a9229acf01ab` 与B1留存一致。原包含逐例预测/临床成员sidecar，所以**未整包复制**；`Existing_Research/`仅按白名单交回现成README、run_index、25份配置、协议与聚合metrics等 58 个源文件，逐文件SHA可追溯。原包不变，没有评估Q系或运行75条件。

该现成索引含25项候选：24项记录为本地最终完成证据，Q110 seed2024只有队列完成线索而最终文件未同步。Q000 seed3407的36轮暂停任务不冒充完成。T1/B1/Concat的逐例缺项按原README保留，不补造。更详细的复用边界见 `NEXT_TRAINING_DECISION.md`。

B2规定的六模型成对推理、统计和阈值部分均无未解决阻塞。后续研究缺项不阻塞本轮B2，也不代表许可启动训练。手稿英文及投稿短文仍由ChatGPT处理，R3.1原稿未改。到此停止，不进入C/D。
