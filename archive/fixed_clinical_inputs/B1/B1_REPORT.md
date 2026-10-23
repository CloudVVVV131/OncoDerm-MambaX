# B1_REPORT：既有预测的正式补充统计分析

状态：COMPLETE。完成日期：2026-10-11（Asia/Shanghai）。仅执行 B1，执行完毕停止。

## 1. 中文摘要

- 7 个直接模型、9 个重复运行均完成指定分析。16 个预测文件经原始清单精确定位及哈希核对；3 个 AUG/STAB 同 run 同哈希副本去重后为 13 个唯一运行。
- 共 33,126 张图像、2,056 名患者、584 个阳性图像。按 ID 精确连接，未按行号拼接、未删样本、未筛选分数或预测头。
- R3.1 原值核对 46/46 通过四位小数容差 5.1e-5：Table 1 点估计 14 项、Table S13 专用头 18 项、Table 3 均值/样本 SD 12 项、Table 2 差值 2 项。
- 10,000 次有效患者聚类配对 bootstrap 完成，尝试 10,000 次，单类拒绝 0 次；计算耗时 212.189 秒。没有缩减重复次数。
- 固定 seed42 的共同头 AP：B6 低于 B2、高于 EfficientNet-B0，两项主要比较的 97.5% 区间均不含 0。三次固定运行平均 B6−B2 AP 的 95% 区间包含 0；这是不同估计对象，不能混为一项结论。
- 两种预测头分别汇总；7 个模型完整 ROC/PR 坐标已交付。未生成或覆盖任何论文图片。

## 2. 输入、事前冻结与科学口径

唯一手稿目录：`${R52_PATH_520E50114650}`。

已读取任务入口：
- `${R52_PATH_CE54EE539476}`
- `${R52_PATH_03B1A6BFBC99}`

历史预测定位依据：包内 `Reference_Evidence/Stage_A/PREDICTION_INVENTORY.csv`、`SOURCE_MANIFEST.csv` 及 `EVIDENCE/AUG_direct_baseline_manifest.json`。具体每个输入绝对路径、大小、修改时间及 SHA-256 见 `INPUT_MANIFEST.csv`；每个运行与 score 对应见指标 CSV。

患者映射：`${R52_PATH_83C520ED9A14}`，只读取 `image_name`、`patient_id`、`target`。这些字段不进入模型，本次没有模型运行。

在 2026-10-11 17:32:24 +08:00 冻结分析协议、独立脚本、输入清单和环境，17:33:26 开始产生新性能值。此为已查看过的 SIIM 队列上的本次补充分析事前冻结，不是历史 7 月研究预注册，也不是新患者外部验证。

- 原协议 SHA-256：`f9544644203bfecd0efe488749a48ef687c660fb8d5c152de9003c3f3e569446`
- 本次冻结协议 SHA-256：`d046582c2b064059211f65e61c9dc79157beba9b4a60ab0d807c221e9db15fd3`
- 独立分析脚本 SHA-256：`e5a290ffc05cfa9da2e709b0148db3a4e10075d34c9ef7acf8125b4b78d57244`

共同评分为七分类 MEL 概率 `prob_mel`；专用 melanoma 头 `prob_mel_endpoint` 仅在明确单列的重复运行结果中使用。AP 是 `sklearn.metrics.average_precision_score` 的 recall 增量加权 precision，不是 PR 梯形积分。AUC 参考 `roc_auc_score`。

## 3. 七模型原点估计（split42 / training seed42 / prob_mel）

| 模型 | ROC-AUC | AP |
| --- | --- | --- |
| B1 | 0.709511 | 0.084490 |
| B2 | 0.798544 | 0.153338 |
| B4 | 0.765966 | 0.113483 |
| B6 | 0.781036 | 0.124919 |
| EfficientNet-B0 | 0.768164 | 0.091419 |
| Swin-Tiny | 0.720752 | 0.080100 |
| DermaMamba port | 0.747256 | 0.100808 |

以上均与 R3.1 Table 1 对应；未用后续新训练替换任何一行。完整未四舍五入值见 `DIRECT_POINT_METRICS.csv`。

## 4. 患者聚类配对区间

NumPy PCG64 seed=20261011；patient_id 按原始字符串字典序排序。每次从 2,056 名患者有放回抽取 2,056 次，患者抽取次数赋给其全部图像。同一次抽样用于所有 17 个评分向量。指标仍为图像级加权指标，没有先对患者内概率求 max/mean。两类缺失时应沿同一 RNG 流重抽，本次未发生。

差值统一为“左模型减右模型”。95% 双侧 percentile 使用 2.5%/97.5% 分位数，NumPy quantile method=linear。所有预定比较均报告，无 p 值。

| 比较 | 指标 | 左侧点值 | 右侧点值 | 差值 | 95% CI |
| --- | --- | --- | --- | --- | --- |
| B6 − B2 | roc_auc | 0.781036 | 0.798544 | -0.017508 | [-0.032262, -0.003221] |
| B6 − B2 | average_precision | 0.124919 | 0.153338 | -0.028419 | [-0.047337, -0.010375] |
| B6 − EfficientNet-B0 | roc_auc | 0.781036 | 0.768164 | 0.012872 | [-0.008952, 0.034369] |
| B6 − EfficientNet-B0 | average_precision | 0.124919 | 0.091419 | 0.033500 | [0.013554, 0.055086] |
| B6 − DermaMamba port | roc_auc | 0.781036 | 0.747256 | 0.033780 | [0.011768, 0.056628] |
| B6 − DermaMamba port | average_precision | 0.124919 | 0.100808 | 0.024111 | [0.002694, 0.046992] |
| B2 − EfficientNet-B0 | roc_auc | 0.798544 | 0.768164 | 0.030380 | [0.009828, 0.050930] |
| B2 − EfficientNet-B0 | average_precision | 0.153338 | 0.091419 | 0.061919 | [0.040351, 0.086562] |

两个主要 AP 比较另外报告 1.25%/98.75% 分位区间（97.5% 单项覆盖，近似 Bonferroni 两比较家族控制）：

| 主要 AP 比较 | 差值 | 97.5% CI |
| --- | --- | --- |
| B6 − B2 | -0.028419 | [-0.050651, -0.007696] |
| B6 − EfficientNet-B0 | 0.033500 | [0.011035, 0.058496] |

本次 CI 是新的补充分析结果；旧 Table 2 的点差已核对，但不要求不同 RNG 生成的 CI 端点逐位相同。患者抽样区间条件于已固定模型/运行，不能推广为整个训练算法的不确定性或新队列性能保证。

## 5. 两种头的三次运行汇总

各模型 training seeds=42/2024/3407，split42。均值与样本 SD（ddof=1）按三个独立 run 计算，不拼接预测求一个总 AUC，不逐 run 取较好预测头。

| 模型 | score | 指标 | 均值 | 样本 SD |
| --- | --- | --- | --- | --- |
| B2 | prob_mel | roc_auc | 0.764781 | 0.042368 |
| B2 | prob_mel | average_precision | 0.129601 | 0.020830 |
| B2 | prob_mel_endpoint | roc_auc | 0.765432 | 0.042131 |
| B2 | prob_mel_endpoint | average_precision | 0.132207 | 0.018994 |
| B4 | prob_mel | roc_auc | 0.724551 | 0.040441 |
| B4 | prob_mel | average_precision | 0.095857 | 0.017776 |
| B4 | prob_mel_endpoint | roc_auc | 0.722938 | 0.037681 |
| B4 | prob_mel_endpoint | average_precision | 0.096930 | 0.019165 |
| B6 | prob_mel | roc_auc | 0.787288 | 0.007743 |
| B6 | prob_mel | average_precision | 0.124034 | 0.000915 |
| B6 | prob_mel_endpoint | roc_auc | 0.790270 | 0.004413 |
| B6 | prob_mel_endpoint | average_precision | 0.127257 | 0.004423 |

共同头三次运行汇总是本次新增分析；专用头与 R3.1 Table 3/S13 一致。18 行完整逐 run 双头指标在 `REPEATED_RUN_METRICS.csv`。

次要分析：每次共用患者重采样，分别计算三个 seed 的 B6−B2 差值，再取三个差值的平均：

| score | 指标 | 平均差值 | 95% CI |
| --- | --- | --- | --- |
| prob_mel | roc_auc | 0.022507 | [0.012389, 0.032650] |
| prob_mel | average_precision | -0.005568 | [-0.017012, 0.004776] |
| prob_mel_endpoint | roc_auc | 0.024837 | [0.014368, 0.035430] |
| prob_mel_endpoint | average_precision | -0.004950 | [-0.016307, 0.005677] |

这些区间全部标记 `conditional_on_fixed_three_runs`。没有对训练 seed 总体方差进行推断，没有生成“稳定性显著性”结论。三次运行平均 AUC 差区间为正，AP 差区间含 0；不据此宣称所有 score/seed 下 B6 都优于 B2。

## 6. 正确性、曲线及完整性

- 23 项小型测试全部通过，包含 ties、常数分数、非均匀整数权重、零权重、显式重复行、患者重复及单类未定义处理。
- 17 个真实 score 向量 × 3 组独立验证权重，共 51 次与 sklearn `sample_weight` 对照；最大绝对误差 1.11e-16，低于 1e-10。验证流 seed20261012 独立于正式 bootstrap。
- 前 100 次正式 bootstrap 耗时约 2.112 秒，预计总耗时 211.2 秒；100 次计入最终 10,000 次，未重置 RNG。
- 完整 ROC 坐标 228,429 行，PR 坐标 228,429 行，均含 7 模型且 `drop_intermediate=False`。PR 的 x=recall、y=precision；threshold 仅作为曲线坐标附属值，不代表选定操作阈值。ROC 起点 threshold=inf、PR 末端 threshold 为空，均为库的标准输出。
- 各曲线端点、单调性、AP 的 recall 增量和 ROC 面积均通过检查。没有平滑、抽稀、重画论文或替换现有 Figure 2。
- 最终读取本次保存的 10,000×17×2 个逐次指标并独立核对所有区间表，最大分位数差 0；没有重新抽样。
- 最终原文件保护检查：26 项冻结输入、14 项清点来源、193 项解压原包资产、4 项冻结分析文件全部 SHA 一致；输入与清点来源的已记录修改时间保持不变。范围内检查不等于全项目内容审计。

## 7. 后续已有运行复用清单（仅导航）

找到现成 `${R52_PATH_D5D08D4FE43D}` 及同名 README，仅读其 index 和静态 config，不读取逐例预测或重新评价后续模型。

25 个候选运行：按已有完成报告，24 个完成、1 个 `B27_Q110_seed2024` 最终文件未同步本机。核心 22 个中有 21 个最终 75 条件结果，另有 3 个 T1 参照。未把 exit0 等同于本机最终 best 已核验。暂停的 Q000 seed3407 未计入完成运行，也未恢复。

分类：24 个“数据/协议不同须另行决定”、1 个“信息不足”；本次没有判定任何候选可直接替换 R3.1 行。既有后续训练为 HAM 验证证据、FP32/TF32关闭、80轮、effective batch64、macro-F1-first；与新方案的建议 batch48、AUC-first 及辅助数据/模型定义不相同。具体结构、数据/辅助清单导航、基础权重来源、seed、历史 checkpoint 选择、完成状态及外部暴露边界见 `REUSE_AND_GAP_REGISTER.csv`。

这是可复用候选清点，不是要求把已完成模型重训。后续是否采用另一条研究主线需作者决定；不得把这些验证结果当作已完成的新外部结果。原 100-ID 辅助名单追查未重新开展。

## 8. B2 准备情况与未完成项

B1 本身没有未完成或阻塞项。B2 仅清点：6 个原 B2/B6 checkpoint 的路径及大小存在且与历史清单一致，6 个配置 SHA 一致；未读/未加载权重，不声明其内容或严格加载兼容性已通过。本机已存在后续图像恢复报告，详细范围见 `B2_READINESS.md`。

E_REAL 映射冻结、严格加载和成对推理、校准/阈值及设备批准均尚未在本轮执行。B2/C/D 没有启动，也没有连接远端或恢复任何队列。

## 9. 实际新增操作、复现与交付

仅进行：ZIP 完整性验证与新目录解压、静态 OOXML 表读取、已有预测的本地 CPU 指标和区间分析、完整曲线坐标输出、有限来源清点、源文件 hash 复核、报告和打包。没有训练、推理、权重加载、校准拟合、操作阈值选择、模型测速、下载数据、安装升级依赖或修改旧资产。pilot 耗时为本次统计计算计时，不是模型 profiling。

环境：Python 3.13.9；NumPy 2.3.5，pandas 2.3.3，SciPy 1.16.3，scikit-learn 1.7.2，PyYAML 6.0.3；单 worker、BLAS 单线程，付费算力费用 0。完整环境和时刻在 `ENVIRONMENT.json`、`EXECUTION.log`。

交付路径：
- `${R52_PATH_0C9E194539E0}`
- `${R52_PATH_4572453C9C70}`
- `${R52_PATH_C3CC6AC309DF}`

ZIP 只含报告、协议/清单、聚合指标/差值/区间、曲线坐标、测试、独立脚本和环境记录，不含原始患者/临床表、逐例预测、模型权重、图像、凭据或 .env。逐次 bootstrap 指标及 RNG 断点留在本机 `local_only/`，不入包。复现说明见 `README_REPRODUCTION.md`。

停止边界：B1 完成。论文英文与投稿短文交由 ChatGPT 处理；本轮没有进入 B2/C/D。
