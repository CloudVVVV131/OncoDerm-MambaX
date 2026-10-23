# ORIGINALCORE_DERM7PT_REPORT

生成时间：2026-10-17T18:49:11+08:00。状态：**COMPLETE**。最终限定数值/来源复核：**PASS**。

## 1. 执行范围与评价身份

- 本次仅评价原稿 B2/B6 各 training seed 42、2024、3407，共六个原始 selected-best。新增训练0；不换best、不微调、不增加seed、不集成。

- 完成六模型 × 两输入 × 两预测头，共24个评分向量；每个向量均为同一纳入病例集。六项原预定运行及两头全部保留。

- 本任务是在查看 P1–P6/P4结果后，为补齐原稿模型覆盖而追加的评价，不是最初预注册的确认性试验。不撤销、替换或重算 P3/P4 原主要比较。

- 本次没有加载 M0/FIM FULL、B2S、区域候选或clean模型，没有重跑B1/B2、P1–P6，没有连接云端、训练、拟合校准器或重选阈值；原论文、环境、共享代码与历史结果不变。

- 用户于本轮集中确认：**六个原模型及其开发过程均未主动使用 Derm7pt**，覆盖训练、辅助训练、开发选型、调参和checkpoint选择。原话与询问范围保存在 `USAGE_HISTORY_CONFIRMATION.json`，不是由下载时间推断，也不担保未知跨数据集重复不存在。

## 2. 六模型身份与执行条件

| 标签 | 完整原run ID | 原选定epoch | 权重字节数 | SHA-256 |
| --- | --- | --- | --- | --- |
| ORIG_B2_s42 | oncoderm_mambax_b2_metadata_cmgf_seed42_20260706_234933 | 41 | 360781827 | 7d3b060e46add18b7df02014e1df9869133ddca2dfd8d5fafb36feeb725bd16b |
| ORIG_B2_s2024 | oncoderm_mambax_b2_metadata_cmgf_seed2024_20260708_160522 | 31 | 360781891 | bd9300f6fed9f9cb71796cf25d8bc5468a1003a169be137d7a95e96bd8c4181a |
| ORIG_B2_s3407 | oncoderm_mambax_b2_metadata_cmgf_seed3407_20260708_230453 | 35 | 360781891 | c9173f33ff787eb5533a7abfb0987dd6577618a15f6f51fff847d5d06beaabf5 |
| ORIG_B6_s42 | oncoderm_mambax_b6_full_seed42_20260707_004712 | 22 | 401499163 | 525302d30405b75229c428d5b1a8d1f282de0e0d25ac98ef13c84fd91b5f923c |
| ORIG_B6_s2024 | oncoderm_mambax_b6_full_seed2024_20260708_163051 | 32 | 401499163 | 4772ae5d3ef772c58438ce35915326f10a01d8fcd416999e6f274ecfd005abfa |
| ORIG_B6_s3407 | oncoderm_mambax_b6_full_seed3407_20260708_232735 | 18 | 401499163 | d27998a562ccafeed44bf731674dec95943b3276db80bd1239af645a8c006ef9 |

权重共同父目录：`${R52_PATH_DD938EDE818E}`。原文件名均为 `checkpoints/best_val_macro_f1.pth`；原 `config.yaml` 和逐字节副本、完整路径在 `FIXED_MODEL_IDENTITIES.json`。

六文件大小及完整SHA均与任务书、B2实际清单、技术记录匹配。使用weights_only=True，只显式允许TorchVersion；构造时唯一覆盖pretrained=False以禁止下载，随后strict=True恢复全部模型参数。没有缺失键、随机补层或任意pickle。checkpoint epoch与B2记录一致。数据配置仅按B2已经确认的派生metadata_dim=22对齐，其余原值一致。

设备：NVIDIA GeForce RTX 4060 Laptop GPU；torch 2.8.0+cu128，torchvision 0.23.0+cu128，timm 1.0.27，CUDA runtime 12.8。FP32、eval/inference_mode，TF32关闭，batch8/workers2。

实际导入模型：`${R52_PATH_C43AB3AA73CB}`；SHA `7e737a004f69053821628f84b3da9c44a4e4a208a386fa7322c5c7ff9331a0c5`。模型实现与B2实际副本逐字节一致。`MODEL_SOURCE_MANIFEST.csv`给出原源码、B2副本和本轮导入路径；`CODE_LINE_INDEX.csv`给实际函数行号。

原B6配置使用已存checkpoint对应的axial Conv1d scan实现，未改成官方mamba_ssm或新的区域读出；原B2关闭CSSA/LPAH。完整配置和后端元信息见各运行 `*_TECHNICAL.json`。此处只记录实际实现，不重新定义原方法。

RGB → 短边256、BILINEAR、antialias=True → 中心裁剪224 → ImageNet normalize。每模型固定前8张分别核对完整forward与两条件特征复用，atol1e-5/rtol1e-4，六项均通过、最大logit差为0；随后只共享视觉特征，两种临床输入分别经过原CMGF和两个头。

主评分为七分类softmax MEL索引4 (`prob_mel`)，专用sigmoid melanoma头 (`prob_mel_endpoint`) 为次要。没有按成绩选头，没有HAM七类强行重映射、macro-F1或新阈值搜索。

## 3. 病例、临床输入与数据关系

- 源表1011行；纳入1011个derm病例，melanoma 252，non-melanoma 759，患病率24.925816%；排除0。全部20种原诊断分布见 `DIAGNOSIS_DISTRIBUTION.csv`。non-melanoma不等于全部良性。

- 只用原表derm引用；保留FCL/FCl及扩展名大小写，未搜索替代路径。9条clinic与derm同路径记录按derm定义各计一次，不将物理照片数当病例数。

- 元数据SHA：`471199fb86bcc97cd1521bcc4255bb3f9c563e2ed0ad3ff836392d81a3ed9d3b`。复用P4病例/同图簇绑定与原像素指纹，本轮重新核对实际derm文件SHA及行对应；未重新解压或全包审计。

- B2原22维词表、顺序、age/100、unknown/missing语义与P4逐项一致，并对每个原模型用原encoder重建临床张量验证精确一致。E_AVAILABLE仅含实际sex/location；全部age=0、age_missing=1。head neck/buttocks映射unknown one-hot，但site_missing=0。E_NULL为原结构化全缺失向量，不是全零。

- diagnosis、7点检查、management、elevation、difficulty、notes、路径和病例ID不进入模型；真值掩膜不进入分类。映射见 `CLINICAL_MAPPING.json`。

- 原稿HAM split42训练6982、验证1513成员/文件指纹与P4所用split42一致。仅在这些明确名单和原始恢复图像清单范围内核对ID、文件SHA、已有RGB像素SHA，未发现明确训练/开发同图匹配。没有按模型分数排除病例。

- **覆盖限制集中记录：**原B6辅助100成员仍未知，本轮未追查、重建或声称其无重叠；新FIM辅助100不能替代历史名单。没有可靠患者/跨病例病灶关联，也未新做近重复检索，因此不宣称全部跨数据集关系均已排除。

- 重采样单位为case_num及已确认同图簇，代表case_num数值升序；本集1011簇。case_id没有被当患者ID，区间不是已核实的患者级区间。

## 4. E_AVAILABLE共同头表现

AP为非插值average_precision，不是PR梯形面积。每run先算指标，再求三run均值及样本SD(ddof=1)；不拼接预测、不平均概率。

| 模型 | 指标 | 三run均值 | seed样本SD | 95%病例簇区间 |
| --- | --- | --- | --- | --- |
| ORIG_B2 | roc_auc | 0.847065 | 0.003720 | [0.820541, 0.872244] |
| ORIG_B2 | average_precision | 0.677181 | 0.010797 | [0.619263, 0.734404] |
| ORIG_B6 | roc_auc | 0.849202 | 0.001651 | [0.823898, 0.873154] |
| ORIG_B6 | average_precision | 0.668175 | 0.009103 | [0.610124, 0.726352] |

## 5. 两项重点输入效应及支持性分析

统一方向：E_AVAILABLE−E_NULL。以下完整保留两头AP/AUC；97.5%仅用于共同头的两个AP输入比较，是近似Bonferroni参考，不能消除本次追加评价的选择背景。

| 模型 | 头 | 指标 | 配对差均值 | 配对差seed SD | 置信度 | 病例簇区间 |
| --- | --- | --- | --- | --- | --- | --- |
| ORIG_B2 | prob_mel | roc_auc | 0.002784 | 0.001690 | 95% | [0.000661, 0.005067] |
| ORIG_B2 | prob_mel | average_precision | 0.000635 | 0.003885 | 95% | [-0.005528, 0.007233] |
| ORIG_B2 | prob_mel | average_precision | 0.000635 | 0.003885 | 97.5% | [-0.006542, 0.008020] |
| ORIG_B2 | prob_mel_endpoint | roc_auc | 0.002478 | 0.000990 | 95% | [0.000718, 0.004309] |
| ORIG_B2 | prob_mel_endpoint | average_precision | 0.000772 | 0.002752 | 95% | [-0.004591, 0.006425] |
| ORIG_B6 | prob_mel | roc_auc | 0.005481 | 0.005101 | 95% | [0.002651, 0.008445] |
| ORIG_B6 | prob_mel | average_precision | 0.007012 | 0.008378 | 95% | [0.000508, 0.013445] |
| ORIG_B6 | prob_mel | average_precision | 0.007012 | 0.008378 | 97.5% | [-0.000479, 0.014327] |
| ORIG_B6 | prob_mel_endpoint | roc_auc | 0.002588 | 0.002398 | 95% | [0.000494, 0.004791] |
| ORIG_B6 | prob_mel_endpoint | average_precision | 0.001539 | 0.003870 | 95% | [-0.003118, 0.007262] |

共同头B2的AP输入效应95%/97.5%区间均包含0；B6的95%区间下限为正，但97.5%区间包含0。不能把未通过两比较参考区间的结果写成已经确认的AP增益，也不能把包含0写成等效或无效。AUC为支持性结果，不替换AP重点比较。

## 6. 原B6相对原B2完整配置比较

统一方向：ORIG_B6−ORIG_B2；全部为支持性完整配置比较，不是单因素LPAH消融。共同头E_AVAILABLE AP差均值为负、AUC差均值为正，两者95%区间均跨0；不据此宣称任一模型全面优越或等效。

| 输入 | 头 | 指标 | 配对差均值 | 配对差seed SD | 95%病例簇区间 |
| --- | --- | --- | --- | --- | --- |
| E_AVAILABLE | prob_mel | roc_auc | 0.002137 | 0.002076 | [-0.010982, 0.015235] |
| E_AVAILABLE | prob_mel | average_precision | -0.009006 | 0.019563 | [-0.032613, 0.014423] |
| E_AVAILABLE | prob_mel_endpoint | roc_auc | -0.005065 | 0.005014 | [-0.018210, 0.007914] |
| E_AVAILABLE | prob_mel_endpoint | average_precision | -0.021192 | 0.022551 | [-0.043835, 0.002845] |
| E_NULL | prob_mel | roc_auc | -0.000559 | 0.005700 | [-0.014160, 0.012950] |
| E_NULL | prob_mel | average_precision | -0.015383 | 0.018114 | [-0.039553, 0.009545] |
| E_NULL | prob_mel_endpoint | roc_auc | -0.005175 | 0.006620 | [-0.018331, 0.007821] |
| E_NULL | prob_mel_endpoint | average_precision | -0.021959 | 0.024384 | [-0.044989, 0.002385] |

## 7. 完整逐run指标

Brier使用未裁剪概率；仅binary NLL用epsilon1e-7；ECE为15等宽bin、末bin包含1、空bin均值为空。Brier/NLL/ECE仅描述，无新校准拟合或置信区间。

| 模型 | seed | 输入 | 头 | AP | AUC | Brier | NLL | ECE15 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ORIG_B2 | 42 | E_AVAILABLE | prob_mel | 0.688420 | 0.851196 | 0.185353 | 2.116095 | 0.189628 |
| ORIG_B2 | 42 | E_AVAILABLE | prob_mel_endpoint | 0.684875 | 0.849243 | 0.181050 | 2.040445 | 0.182798 |
| ORIG_B2 | 42 | E_NULL | prob_mel | 0.684531 | 0.846514 | 0.199336 | 2.350501 | 0.202052 |
| ORIG_B2 | 42 | E_NULL | prob_mel_endpoint | 0.682832 | 0.845630 | 0.199090 | 2.338444 | 0.202882 |
| ORIG_B2 | 2024 | E_AVAILABLE | prob_mel | 0.676233 | 0.846017 | 0.191459 | 2.143627 | 0.195170 |
| ORIG_B2 | 2024 | E_AVAILABLE | prob_mel_endpoint | 0.677057 | 0.843570 | 0.182368 | 2.020719 | 0.185557 |
| ORIG_B2 | 2024 | E_NULL | prob_mel | 0.679900 | 0.844574 | 0.186586 | 2.077374 | 0.188838 |
| ORIG_B2 | 2024 | E_NULL | prob_mel_endpoint | 0.679442 | 0.841782 | 0.183567 | 2.007240 | 0.186923 |
| ORIG_B2 | 3407 | E_AVAILABLE | prob_mel | 0.666889 | 0.843981 | 0.179026 | 2.118104 | 0.180834 |
| ORIG_B2 | 3407 | E_AVAILABLE | prob_mel_endpoint | 0.666347 | 0.844812 | 0.179179 | 2.076819 | 0.178176 |
| ORIG_B2 | 3407 | E_NULL | prob_mel | 0.665207 | 0.841753 | 0.183075 | 2.143153 | 0.182420 |
| ORIG_B2 | 3407 | E_NULL | prob_mel_endpoint | 0.663687 | 0.842778 | 0.182538 | 2.101655 | 0.183899 |
| ORIG_B6 | 42 | E_AVAILABLE | prob_mel | 0.660980 | 0.851068 | 0.163991 | 1.467821 | 0.160241 |
| ORIG_B6 | 42 | E_AVAILABLE | prob_mel_endpoint | 0.653203 | 0.843335 | 0.165418 | 1.369265 | 0.158807 |
| ORIG_B6 | 42 | E_NULL | prob_mel | 0.662863 | 0.849839 | 0.171620 | 1.545949 | 0.169210 |
| ORIG_B6 | 42 | E_NULL | prob_mel_endpoint | 0.656023 | 0.843212 | 0.170628 | 1.408726 | 0.166600 |
| ORIG_B6 | 2024 | E_AVAILABLE | prob_mel | 0.665137 | 0.848608 | 0.183427 | 2.141429 | 0.185041 |
| ORIG_B6 | 2024 | E_AVAILABLE | prob_mel_endpoint | 0.640461 | 0.833966 | 0.185073 | 2.145564 | 0.184838 |
| ORIG_B6 | 2024 | E_NULL | prob_mel | 0.650382 | 0.837472 | 0.180953 | 2.124220 | 0.181711 |
| ORIG_B6 | 2024 | E_NULL | prob_mel_endpoint | 0.635890 | 0.829054 | 0.183301 | 2.134012 | 0.183347 |
| ORIG_B6 | 3407 | E_AVAILABLE | prob_mel | 0.678408 | 0.847931 | 0.174200 | 1.456210 | 0.169833 |
| ORIG_B6 | 3407 | E_AVAILABLE | prob_mel_endpoint | 0.671039 | 0.845128 | 0.169311 | 1.257914 | 0.164335 |
| ORIG_B6 | 3407 | E_NULL | prob_mel | 0.670243 | 0.843853 | 0.186223 | 1.717304 | 0.184512 |
| ORIG_B6 | 3407 | E_NULL | prob_mel_endpoint | 0.668172 | 0.842399 | 0.182706 | 1.555488 | 0.181922 |

三run均值/SD40行在 `THREE_RUN_SUMMARY.csv`；24行逐run在 `PER_RUN_METRICS.csv`；360行可靠性bin在 `RELIABILITY_BINS.csv`；逐seed差值在 `PER_SEED_DIFFERENCES.csv`。未定义值不会填0。

## 8. Bootstrap与固定阈值

共10000次有效抽样、尝试10000次、单类重抽0次；NumPy PCG64 seed20261021。同一簇抽样在六模型、训练seed、两条件和两头间共享，簇重复次数赋给簇内图像；先每run计算AP/AUC、同seed差值，再平均三个seed。percentile区间、quantile method=linear。训练seed不重采样，seed变化另列样本SD。所有区间条件于这六个固定模型。

阈值唯一来源：`${R52_PATH_DDC319274460}`。`DEVELOPMENT_THRESHOLDS_ORIGINAL.csv`为逐字节副本；24条来自各原run/seed/head自己的HAM validation NATIVE 90%/95%敏感度政策。源表没有run_id列，因此经B2已执行PREDICTION_MANIFEST的唯一family/seed及checkpoint SHA绑定到完整run_id，见 `THRESHOLD_IDENTITY_JOIN.csv`。

同一原阈值以p>=t应用两种Derm输入，共48条。`THRESHOLD_COUNTS.csv`保留TP/FN/FP/TN、敏感度、特异度、PPV/NPV及未定义原因；`THRESHOLD_SUMMARY.csv`给三run描述均值/样本SD。未使用P3/P4新模型阈值、未再推理HAM、未在Derm7pt选阈值。

## 9. 跨队列与图源

`ORIGINAL_CORE_CROSS_COHORT_SUMMARY.csv`仅并列原B2/B6在SIIM及Derm7pt的两输入、双头、五指标三run均值和SD。SIIM取B2已完成FP32聚合值，未重推理、未混入旧AMP结果。SIIM E_REAL有实际年龄，Derm7pt E_AVAILABLE所有年龄缺失；两者患病率和输入覆盖不同，不相减绝对AP，不拼接队列。

`ROC_COORDINATES.csv`、`PR_COORDINATES.csv`含全部24个评分向量的完整无平滑坐标。ROC首点threshold=inf，PR最后一点threshold为空，均为标准端点。未制作或修改论文图。

## 10. 复核、时间与交付

限定最终复核PASS：六权重与所有登记来源指纹未变；24行指标及均值SD核对；48条阈值计数独立核对；完整ROC积分/PR阶梯AP与点值一致；前三个既有bootstrap抽样用sklearn权重定义交叉核对；全部22条区间与保存数组再核对。没有新增推理或统计比较。

执行窗口从2026-10-17T18:04:50+08:00至2026-10-17T20:04:50+08:00。截至本报告约44.36分钟，包含准备、使用史确认、代码接入和交付；六模型逐项加载/小批量/推理耗时合计67.33秒，bootstrap实际12.35秒。首模型估时低于窗口；无需超时恢复。具体时间见日志与各运行JSON。

源码与结果均在本轮新目录；逐例预测、病例临床绑定和bootstrap断点仅在本地 `private_predictions/`，不进入ZIP。返回ZIP按显式白名单打包，含本报告、最终进度、冻结协议、必要代码、配置/指纹、全部聚合及曲线；不含权重、原图、meta.csv、逐例预测/临床资料、凭据或.env。

报告状态只代表本任务全部授权步骤完成，不代表文章结论或投稿接收判断。本任务到此停止，原稿英文与R5整合交由ChatGPT。


## 来源定位索引

- 模型：`FIXED_MODEL_IDENTITIES.json`的models逐项信息；冻结协议 `models`；技术文件 `source_path`、`source_sha256`、`epoch`、`strict`。

- 原临床编码：`model_code/lesionmamba/data/datasets.py::encode_metadata`；原预处理：`data/transforms.py::build_transforms`。

- 加载及成对forward：`core_infer.py::load/full/visual/head/infer_one`；病例编码：`core_data.py::DermData`。

- 纯指标来自P4 `p4_metric_core.py::GroupedMetric/ece/binary`，AST提取到 `core_metrics.py`，无P4总入口导入。新增统计只执行 `core_statistics.py::points/bootstrap`。

- 源文件完整绝对路径、SHA、大小见 `ORIGINALCORE_PROTOCOL.json::evidence`；函数实际行号见 `CODE_LINE_INDEX.csv`。
