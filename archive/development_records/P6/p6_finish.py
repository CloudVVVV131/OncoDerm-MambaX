"""P6 validation-only evaluation, descriptive comparison, allowlisted delivery."""
from datetime import datetime
from pathlib import Path
import csv
import hashlib
import time
import zipfile
from p6_common import OUT, PKG, WINDOW, VARIANTS, read, dump, sha, table


def evaluate_run(condition):
    import gc
    import numpy as np
    import torch
    from sklearn.metrics import f1_score
    from p6_metric_core import binary, ece
    import p6_training as t
    assert condition in VARIANTS
    root = OUT / f'{condition}_s42'
    status = read(root / 'status.json')
    assert status['status'] == 'TRAINING_COMPLETE'
    if (root / 'VALIDATION_RESULT.json').exists():
        assert read(root / 'VALIDATION_RESULT.json')['best_sha256'] == sha(root / 'best.pt')
        return
    if time.time() >= t.DEADLINE:
        t.progress('PARTIAL', f'{condition}_s42 validation pending')
        return
    t.seed_all(42)
    state = torch.load(root / 'best.pt', map_location='cpu', weights_only=False)
    assert state['epoch'] == status['best_epoch']
    cfg = state['config']['original_config']
    model = t.fresh_model(cfg)
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    expected, epoch = state['selection_validation'], state['epoch']
    del state
    val = t.HAMDataset(cfg, 'val')
    assert len(val) == 1513
    probability, endpoint, targets = [], [], []
    diagnostics = {}
    started = time.perf_counter()
    with torch.inference_mode():
        for images, metadata, labels, _ in t.loader(val, list(range(len(val))), batch_size=8):
            output = model(images.cuda(non_blocking=True), metadata.cuda(non_blocking=True))
            probability.append(output['logits'].float().softmax(1).cpu().numpy())
            endpoint.append(output['mel_logits'].float().sigmoid().cpu().numpy())
            targets.append(labels.numpy())
            t.merge_diagnostics(diagnostics, t.readout_diagnostics(output, 'validation'))
    prob = np.concatenate(probability).astype(np.float64)
    end = np.concatenate(endpoint).astype(np.float64)
    y7 = np.concatenate(targets)
    y = (y7 == 4).astype(np.int64)
    assert len(y) == 1513 and y.sum() == 164
    assert np.isfinite(prob).all() and np.isfinite(end).all()
    assert np.allclose(prob.sum(1), 1, atol=1e-6)
    multi = {'macro_f1': float(f1_score(y7, prob.argmax(1), labels=list(range(7)), average='macro')),
        'multiclass_brier_sum7': float(np.mean(np.sum((prob - np.eye(7)[y7])**2, axis=1))),
        'multiclass_nll': float(np.mean(-np.log(np.clip(prob[np.arange(len(y7)), y7], 1e-7, 1)))),
        'top_label_ece15': float(ece((prob.argmax(1) == y7).astype(np.int64), prob.max(1))[0])}
    heads = []
    for head, score in [('softmax_MEL', prob[:, 4]), ('endpoint_MEL', end)]:
        metrics, _ = binary(y, score)
        heads.append({'head': head, **metrics})
    for actual, reference in [(multi['macro_f1'], expected['macro_f1']),
                              (heads[0]['roc_auc'], expected['mel_auc']),
                              (heads[0]['average_precision'], expected['mel_ap'])]:
        assert abs(actual - reference) < 1e-7, 'Selected-best validation mismatch'
    result = {'status': 'COMPLETE', 'run': root.name, 'condition': condition,
        'cohort': 'HAM_val', 'input': 'NATIVE', 'training_seed': 42,
        'selected_epoch': epoch, 'best_path': str(root / 'best.pt'), 'best_sha256': sha(root / 'best.pt'),
        'strict_loading': True, 'dtype': 'FP32', 'TF32': False,
        'selection_metrics_reproduced': True, 'multiclass': multi, 'binary': heads,
        'eta_raw': float(model.readout.eta.detach()),
        'readout_diagnostics': t.summarize_diagnostics(diagnostics),
        'evaluation_seconds': time.perf_counter() - started,
        'per_example_predictions_saved': False,
        'probability_array_fingerprint': hashlib.sha256(prob.tobytes()).hexdigest(),
        'endpoint_array_fingerprint': hashlib.sha256(end.tobytes()).hexdigest()}
    dump(root / 'VALIDATION_RESULT.json', result)
    t.log(f'SELECTED_BEST_VALIDATED {root.name} epoch={epoch} AP={heads[0]["average_precision"]:.8f}')
    del model, val, probability, endpoint, targets, prob, end, y7, y
    gc.collect()
    torch.cuda.empty_cache()


def package():
    archive = OUT / 'P6_DEV_RESULTS.zip'
    assert not archive.exists(), 'Preserve existing delivery'
    references = []
    path = OUT / 'READ_ONLY_VALIDATION_REFERENCES.csv'
    if path.exists():
        with path.open(encoding='utf-8', newline='') as stream:
            references = list(csv.DictReader(stream))
    identities = read(OUT / 'VALIDATION_REFERENCE_IDENTITIES.json') if (OUT / 'VALIDATION_REFERENCE_IDENTITIES.json').exists() else []
    epochs = {r['model']: r['best_epoch'] for r in identities}
    binary_rows, multi_rows, diag_rows, runs = [], [], [], []
    for row in references:
        row['selected_epoch'] = epochs[row['model']]
        (multi_rows if row['head'] == 'seven_class' else binary_rows).append(row)
    for variant in VARIANTS:
        root = OUT / f'{variant}_s42'
        status = read(root / 'status.json') if (root / 'status.json').exists() else {'status': 'NOT_STARTED', 'completed_epochs': 0}
        run = {'run': root.name, **status, 'validation_complete': False}
        if (root / 'effective_config.json').exists():
            cfg = read(root / 'effective_config.json')
            run.update(parameter_count=cfg['parameter_count'], readout_parameter_count=cfg['readout_parameter_count'])
        if (root / 'training_history.csv').exists():
            with (root / 'training_history.csv').open(encoding='utf-8', newline='') as stream:
                history = list(csv.DictReader(stream))
            run['sum_epoch_training_validation_seconds'] = sum(float(h['epoch_seconds']) for h in history)
            best = max(history, key=lambda r: (float(r['val_macro_f1']), float(r['val_mel_auc']), -int(r['epoch'])))
            assert int(best['epoch']) == status['best_epoch']
            run['selection_rule_verified_from_full_history'] = True
        result_path = root / 'VALIDATION_RESULT.json'
        if result_path.exists():
            result = read(result_path)
            assert result['best_sha256'] == sha(root / 'best.pt')
            run.update(validation_complete=True, selected_epoch=result['selected_epoch'], best_sha256=result['best_sha256'])
            common = {'model': variant.replace('FIM_', ''), 'run': root.name, 'training_seed': 42,
                'selected_epoch': result['selected_epoch'], 'cohort': 'HAM_val', 'input': 'NATIVE',
                'origin': 'NEW_P6_SELECTED_BEST_VALIDATION'}
            binary_rows += [{**common, **b} for b in result['binary']]
            multi_rows.append({**common, 'head': 'seven_class', **result['multiclass']})
            diag_rows.append({**common, 'eta_raw': result['eta_raw'], **result['readout_diagnostics']})
        runs.append(run)
    completed = sum(r['status'] == 'TRAINING_COMPLETE' and r['validation_complete'] for r in runs)
    errors = sorted(OUT.glob('*ERROR*.json'))
    state = 'COMPLETE' if completed == 2 else ('BLOCKED' if errors else 'PARTIAL')
    differences = []
    for head in ['softmax_MEL', 'endpoint_MEL', 'seven_class']:
        data = multi_rows if head == 'seven_class' else binary_rows
        values = {r['model']: r for r in data if r['head'] == head}
        metrics = ['macro_f1', 'multiclass_brier_sum7', 'multiclass_nll', 'top_label_ece15'] if head == 'seven_class' else [
            'average_precision', 'roc_auc', 'brier', 'binary_nll', 'ece15']
        for left, right in [('TKEEP_REGION', 'TKEEP_GLOBAL'), ('TKEEP_REGION', 'R_IMG'),
                            ('TKEEP_REGION', 'M0'), ('TKEEP_GLOBAL', 'R_IMG'), ('TKEEP_GLOBAL', 'M0')]:
            if left in values and right in values:
                for metric in metrics:
                    differences.append({'left': left, 'right': right, 'head': head, 'metric': metric,
                        'left_value': float(values[left][metric]), 'right_value': float(values[right][metric]),
                        'difference_left_minus_right': float(values[left][metric])-float(values[right][metric]),
                        'inference_scope': 'single_seed_reused_development_validation_no_CI_no_significance'})
    table(OUT / 'ALL_VALIDATION_BINARY_METRICS.csv', binary_rows)
    table(OUT / 'ALL_VALIDATION_MULTICLASS_METRICS.csv', multi_rows)
    table(OUT / 'DESCRIPTIVE_DEVELOPMENT_DIFFERENCES.csv', differences)
    table(OUT / 'SELECTED_BEST_READOUT_DIAGNOSTICS.csv', diag_rows)
    table(OUT / 'RUN_SUMMARY.csv', runs)
    integrity = []
    if (OUT / 'P6_CODE_FREEZE.json').exists():
        for item in read(OUT / 'P6_CODE_FREEZE.json'):
            assert sha(item['path']) == item['sha256'], item['path']
            integrity.append({**item, 'unchanged': True})
    if (OUT / 'SOURCE_COPY_IDENTITIES.json').exists():
        for item in read(OUT / 'SOURCE_COPY_IDENTITIES.json'):
            assert sha(item['source']) == sha(item['copy']) == item['sha256']
    dump(OUT / 'FINAL_INTEGRITY.json', {'status': 'PASSED', 'code': integrity,
        'original_copied_sources_unchanged': True, 'source_models_not_modified': True})
    final_progress = {'status': state, 'updated_at': datetime.now().astimezone().isoformat(), 'window': WINDOW,
        'planned_new_training_runs': 2, 'training_completed': sum(r['status'] == 'TRAINING_COMPLETE' for r in runs),
        'selected_best_validation_completed': completed, 'runs': runs,
        'new_external_inference': 0, 'other_training_seeds': 0, 'bootstrap': False,
        'calibrator_fitting': False, 'threshold_selection': False, 'manuscript_modified': False,
        'next_action': 'STOP; return deliverables, no automatic expansion',
        'report': str(OUT / 'P6_DEV_REPORT.md'), 'results_zip': str(archive),
        'packaging_status': 'VERIFIED', 'verification': str(OUT / 'ZIP_VERIFICATION.json')}
    dump(OUT / 'P6_PROGRESS.json', final_progress)
    def get(model, head, metric):
        data = multi_rows if head == 'seven_class' else binary_rows
        found = next((r for r in data if r['model'] == model and r['head'] == head), None)
        return f'{float(found[metric]):.6f}' if found is not None else 'N/A'
    model_names = ['M0', 'R_IMG', 'R_CLIN', 'XATTN', 'TKEEP_REGION', 'TKEEP_GLOBAL']
    row_lines = []
    for name in model_names:
        src = next((r for r in multi_rows if r['model'] == name), {})
        row_lines.append('| ' + ' | '.join([name, str(src.get('selected_epoch', 'N/A')),
            get(name, 'softmax_MEL', 'average_precision'), get(name, 'softmax_MEL', 'roc_auc'),
            get(name, 'seven_class', 'macro_f1'), get(name, 'softmax_MEL', 'brier'),
            get(name, 'softmax_MEL', 'binary_nll'), get(name, 'seven_class', 'multiclass_nll')]) + ' |')
    timing = '\n'.join(f"- {r['run']}: {r['status']}, 完成 {r['completed_epochs']} 轮, best={r.get('best_epoch')}, "
                     f"训练及逐轮验证 {r.get('sum_epoch_training_validation_seconds',0)/60:.1f} 分钟。" for r in runs)
    focus = [r for r in differences if r['head'] == 'softmax_MEL' and r['metric'] == 'average_precision']
    contrasts = '\n'.join(f"- {r['left']} − {r['right']}: AP差值 {r['difference_left_minus_right']:+.6f}。" for r in focus)
    recovery_text = ''
    if (OUT / 'RUNTIME_RECOVERY.json').exists():
        recovery_text = ('第1轮完整断点写入后，默认GBK读取含中文路径的UTF-8进度JSON发生异常。'
            '仅为三个read_text调用补充UTF-8参数，从epoch2/next_batch0的完整optimizer/scheduler/scaler/RNG断点恢复；'
            '已完成第1轮未重跑，6小时截止时间未延长。保留原始代码冻结记录、修复差异与恢复证据。')
    conclusion = '两个正式模型未全部完成；不把部分状态写成完成，不据此作性能结论。'
    if completed == 2:
        d = {(r['left'],r['right']):r['difference_left_minus_right'] for r in focus}
        if d[('TKEEP_REGION','TKEEP_GLOBAL')] > 0 and d[('TKEEP_REGION','R_IMG')] > 0 and d[('TKEEP_REGION','M0')] > 0:
            conclusion = 'REGION在这一次开发运行中的AP方向同时高于GLOBAL、旧R_IMG和M0，属于任务B类方向的开发信号；是否达到有用幅度需结合下表概率误差和macro-F1判断，不能等同于稳定区域增量。'
        elif d[('TKEEP_REGION','R_IMG')] > 0 and d[('TKEEP_GLOBAL','R_IMG')] > 0:
            conclusion = '两个新读出的AP均高于旧R_IMG，存在读出容量/形式的开发信号；没有预设“接近”或“有用”的数值阈值，不能自动断言区域内容优势。必须同时查看REGION−GLOBAL和概率误差代价。'
        else:
            conclusion = '结果未同时满足REGION相对GLOBAL、旧R_IMG和M0均有AP改善这一方向。不能将其表述为已经验证的区域内容增益；按任务C类处置保留全部结果并停止，具体差值及代价见上述表格和配套CSV。'
    report = f'''# P6 开发报告

## 实际状态：{state}

新增训练仅 FIM_TKEEP_REGION_s42、FIM_TKEEP_GLOBAL_s42。训练与selected-best验证完成 {completed}/2。
M0/R_IMG/R_CLIN/XATTN仅引用已有seed42 HAM validation/NATIVE汇总，不重训、不重新推理。

## 全部开发结果

| 模型 | 选定epoch | softmax MEL AP | AUC | macro-F1 | MEL Brier | MEL NLL | 七分类NLL |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(row_lines)}

{contrasts}

{conclusion}

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

主动窗口 {WINDOW['started_at']} 至 {WINDOW['deadline']}。
{timing}

{recovery_text}

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
'''
    (OUT / 'P6_DEV_REPORT.md').write_text(report, encoding='utf-8')
    names = ['TASK_PROMPT.txt', 'WINDOW.json', 'P6_FROZEN_PROTOCOL.json', 'P6_CODE_FREEZE.json',
             'SOURCE_COPY_IDENTITIES.json', 'TRAINER_REUSE_CHECK.json', 'DATA_CHECK.json',
             'P6_INITIALIZATION.json', 'RUNTIME.json', 'CUDA_TECHNICAL_CHECK_s42.json',
             'VALIDATION_REFERENCE_IDENTITIES.json', 'READ_ONLY_VALIDATION_REFERENCES.csv',
             'ALL_VALIDATION_BINARY_METRICS.csv', 'ALL_VALIDATION_MULTICLASS_METRICS.csv',
             'DESCRIPTIVE_DEVELOPMENT_DIFFERENCES.csv', 'SELECTED_BEST_READOUT_DIAGNOSTICS.csv',
             'RUN_SUMMARY.csv', 'P6_DEV_REPORT.md', 'P6_PROGRESS.json', 'FINAL_INTEGRITY.json', 'training.log',
             'RUNTIME_RECOVERY.json', 'REPORT_TEXT_CORRECTION.json']
    files = [OUT/n for n in names if (OUT/n).exists()]
    files += list(OUT.glob('*.py')) + list(OUT.glob('*.diff')) + list(OUT.glob('*_s42_INITIALIZATION.json')) + errors
    files += list(PKG.rglob('*.py')) + [PKG / 'Reference_Only/B6_seed42_original_config.yaml']
    files += list((OUT / 'encoding_recovery_20261016_151600').glob('*.py'))
    files += list((OUT / 'encoding_recovery_20261016_151600').glob('*.json'))
    for variant in VARIANTS:
        root = OUT / f'{variant}_s42'
        files += [root/n for n in ['effective_config.json', 'status.json', 'training_history.csv',
            'readout_diagnostics.csv', 'TWO_EPOCH_PROGRESS.json', 'VALIDATION_RESULT.json'] if (root/n).exists()]
    assert len(files) == len(set(files))
    assert all(p.suffix.lower() not in ['.pt', '.pth', '.npy', '.npz', '.jpg', '.png', '.env'] for p in files)
    assert not any('fim_mask_train100' in p.name for p in files)
    manifest = [{'path': p.relative_to(OUT).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files]
    dump(OUT / 'PACKAGE_CONTENTS.json', manifest)
    files.append(OUT / 'PACKAGE_CONTENTS.json')
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for path in files:
            z.write(path, path.relative_to(OUT))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for item in manifest:
            assert hashlib.sha256(z.read(item['path'])).hexdigest() == item['sha256']
        assert z.read('P6_PROGRESS.json') == (OUT / 'P6_PROGRESS.json').read_bytes()
        assert z.read('P6_DEV_REPORT.md') == (OUT / 'P6_DEV_REPORT.md').read_bytes()
    dump(OUT / 'ZIP_VERIFICATION.json', {'status': 'PASSED', 'CRC': True, 'member_SHA256': True,
        'report_and_final_progress_byte_identical': True, 'path': str(archive), 'bytes': archive.stat().st_size,
        'members': len(files), 'sha256': sha(archive)})
    print(f'P6_DELIVERY {state} {archive}', flush=True)


if __name__ == '__main__':
    package()
