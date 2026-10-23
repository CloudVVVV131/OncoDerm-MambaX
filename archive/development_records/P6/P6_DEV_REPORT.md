# P6 开发报告

## 实际状态：COMPLETE

新增训练仅 FIM_TKEEP_REGION_s42、FIM_TKEEP_GLOBAL_s42。训练与selected-best验证完成 2/2。
M0/R_IMG/R_CLIN/XATTN仅引用已有seed42 HAM validation/NATIVE汇总，不重训、不重新推理。

## 全部开发结果

| 模型 | 选定epoch | softmax MEL AP | AUC | macro-F1 | MEL Brier | MEL NLL | 七分类NLL |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 | 34 | 0.691352 | 0.924716 | 0.808075 | 0.063789 | 0.696434 | 1.274512 |
| R_IMG | 23 | 0.724575 | 0.930194 | 0.786216 | 0.083885 | 0.553716 | 1.023576 |
| R_CLIN | 26 | 0.732862 | 0.925643 | 0.790629 | 0.056275 | 0.603149 | 1.094280 |
| XATTN | 13 | 0.690365 | 0.924547 | 0.790911 | 0.062864 | 0.319045 | 0.605801 |
| TKEEP_REGION | 18 | 0.679162 | 0.925428 | 0.772749 | 0.063622 | 0.346011 | 0.722322 |
| TKEEP_GLOBAL | 25 | 0.699883 | 0.916980 | 0.793241 | 0.071395 | 0.553724 | 1.079575 |

- TKEEP_REGION − TKEEP_GLOBAL: AP差值 -0.020721。
- TKEEP_REGION − R_IMG: AP差值 -0.045413。
- TKEEP_REGION − M0: AP差值 -0.012190。
- TKEEP_GLOBAL − R_IMG: AP差值 -0.024691。
- TKEEP_GLOBAL − M0: AP差值 +0.008532。

结果未同时满足REGION相对GLOBAL、旧R_IMG和M0均有AP改善这一方向。不能将其表述为已经验证的区域内容增益；按任务C类处置保留全部结果并停止，具体差值及代价见上述表格和配套CSV。

主要开发对比为 REGION−GLOBAL 的softmax MEL AP。两个头全部AP/AUC、unclipped Brier、binary NLL、15等宽ECE见
`ALL_VALIDATION_BINARY_METRICS.csv`；七分类macro-F1、sum-over-7 Brier、NLL及top-label ECE见
`ALL_VALIDATION_MULTICLASS_METRICS.csv`。`DESCRIPTIVE_DEVELOPMENT_DIFFERENCES.csv`保留全部预设描述性差值。
NLL epsilon=1e-7；AP为average_precision_score，不是梯形PR面积。验证病例1513（MEL164）。
原参考值原样只读引用，记录来源路径、SHA与原best身份，不混用test、外部成绩或不同头。

## 实现、初始化和训练

P6-v2四个槽按g/内部/边缘/外围顺序，各自LayerNorm(768)→Linear(768,128,bias)→GELU，concat后
LayerNorm(512)→Linear(512,768,no bias)。h与原g按g+0.5*tanh(eta)*(h−g)融合；eta初始0，Linear保持默认初始化。
REGION使用原预测软掩膜region_tokens；GLOBAL三个区域槽均用同一fused feature的全局空间均值。
原region函数的detach、插值、replicate-padding 3×3形态操作、1e-4有效质量阈值及无效区域占位保持不变。
无旧alpha/score/type embedding实例，无临床读出门控；临床编码、原gate/CMGF、双头只在读出后调用一次。

共享原模型从原训练前initial_state_s42.pt严格载入；新模块fork_rng seed6042。两个条件state_dict键、形状、数值完全相同，
保存共同P6训练前状态及两个初始化身份；技术测试状态不参与正式训练。未加载旧best/last微调。
身份详见`P6_INITIALIZATION.json`、两个`*_INITIALIZATION.json`及`CUDA_TECHNICAL_CHECK_s42.json`。

HAM split42，training seed42，6982 train/1513 validation及冻结辅助100成员，22维原临床编码。
分类顺序42+epoch、逐图增强42+epoch*1000003+index*1009、辅助PCG64 SeedSequence([42,epoch,19])、loader42017。
224输入；effective48/aux32/micro8/workers2；AdamW 3e-5/3e-4、wd.05、warmup5/cosine80、最多80轮、patience15、clip1。
CE beta.999、有效smoothing0、整有效批次target-weight分母；BCE.5、consistency.05、mask Dice+BCE.1、sparse.0005、TV.002。
AMP训练及原溢出回退/BN与RNG恢复，FP32验证、TF32关闭。轻量spawn worker隔离，未改变共享软件或系统设置。
选best仅validation macro-F1，其次softmax MEL AUC，完全相同取更早；AP记录但不用于选模。

## 检查与诊断

小检查覆盖eta0恒等、非零eta时REGION响应区域内容而GLOBAL输出不随mask变化、区域mask权重detach、
原辅助mask与视觉梯度保持、一次backbone/fuse/mask/clinical、有效48+32更新和CE归一化。
逐轮诊断及selected-best诊断记录四槽token范数/投影范数、区域有效率、eta、实际残差范数/原pool范数。
模块梯度在每轮首/末有效更新采样，记录裁剪前L2；eta0首步其他读出参数梯度为0是该残差初始化的数学结果，非冻结。
没有alpha字段，不将其写成0；GLOBAL的区域面积/有效率仅为预测mask诊断，不表示分类槽实际使用这些权重。
scalar coefficient不等于分支贡献百分比；辅助Dice只为训练优化诊断，不是独立定位验证。

## 时间、断点与边界

主动窗口 2026-10-16T14:58:36.909633+08:00 至 2026-10-16T20:58:36.909633+08:00。
- FIM_TKEEP_REGION_s42: TRAINING_COMPLETE, 完成 33 轮, best=18, 训练及逐轮验证 69.9 分钟。
- FIM_TKEEP_GLOBAL_s42: TRAINING_COMPLETE, 完成 40 轮, best=25, 训练及逐轮验证 84.9 分钟。

第1轮完整断点写入后，默认GBK读取含中文路径的UTF-8进度JSON发生异常。仅为三个read_text调用补充UTF-8参数，从epoch2/next_batch0的完整optimizer/scheduler/scaler/RNG断点恢复；已完成第1轮未重跑，6小时截止时间未延长。保留原始代码冻结记录、修复差异与恢复证据。

逐轮与窗口结束保存model、optimizer、scheduler、scaler、Python/NumPy/Torch CPU/CUDA RNG、epoch/next_batch、
global step、历史与部分epoch累计信息，恢复按确定性数据顺序继续。未完成模型不会冒充完成。
本轮没有HAM test/SIIM/Derm7pt模型推理、bootstrap、阈值选择、校准拟合、其他seed、远端/云资源或论文修改。
此前外部队列的使用史不因P6重置；P3/P4原主要比较及P1–P5结果不变。

## 判断范围与交付

这是反复使用开发集上的单seed筛查，没有独立性能确认、显著性、泛化或因果结论。
A类是两模型均改善且接近，提示容量/形式信号；B类需REGION相对GLOBAL和旧读出有有用改善，仍只是后续重复的候选信号；
C类为无改善或代价明显。任务未预先给出“有用/接近”的数值门槛，报告保留全量指标供作者判断，不事后制定显著性标准。
不因成绩改结构、挑选轮次、增加候选或seed。

ZIP包含任务、协议、代码及差异、配置、初始化指纹、日志、全部验证汇总、诊断、本报告和最终P6_PROGRESS.json。
明确排除权重/断点、原图/掩膜、辅助成员逐例表、逐例预测、完整临床资料和凭据。
完成ZIP CRC及逐文件SHA核验后停止，不自动扩种子或进行外部评价。
