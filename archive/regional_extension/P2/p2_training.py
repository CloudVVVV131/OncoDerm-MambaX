"""P2 frozen-v1 seed extension: train/validation/aux100, never test or SIIM."""
from pathlib import Path
import os
import sys
import json
import math
import time
import random
import shutil
import argparse
import traceback
from datetime import datetime

OUT = Path(__file__).resolve().parent
PKG = OUT / 'FIM_RegionClinical_Dev_Start'
ROOT = Path(__import__('os').environ['R52_PATH_EA6F443AA882'])
ARCHIVE = ROOT / 'oncoderm_bioengineering_full_results_20260817_110459/project'
REV = ROOT / 'project/revision'
WINDOW = json.loads((OUT / 'WINDOW.json').read_text(encoding='utf-8-sig'))
OLD = Path(WINDOW['source_seed42'])
THREE = Path(WINDOW['source_three_seed'])
P1 = Path(WINDOW['source_P1'])
DEADLINE = datetime.fromisoformat(WINDOW['deadline']).timestamp()
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['TORCH_HOME'] = str(OUT / 'cache/torch')
os.environ['HF_HOME'] = str(OUT / 'cache/huggingface')
os.environ['HF_HUB_OFFLINE'] = '1'
sys.dont_write_bytecode = True
sys.path.insert(0, str(PKG / 'Reference_Only/model_code'))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import average_precision_score
from lesionmamba.models.oncoderm_mambax import OncoDermMambaX
from lesionmamba.data.datasets import encode_metadata, LABEL_MAP
from lesionmamba.data.transforms import build_transforms, IMAGENET_MEAN, IMAGENET_STD
from p2_model import P2Model

VARIANTS = ['R_IMG', 'R_CLIN', 'XATTN']
SEEDS = [2024, 3407]
RUNS = [(s, v) for s in SEEDS for v in VARIANTS]


def dump(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def save(path, obj):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    torch.save(obj, temp)
    os.replace(temp, path)


def log(message):
    line = f'{datetime.now().astimezone().isoformat()} {message}'
    print(line, flush=True)
    with (OUT / 'training.log').open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(s):
    random.setstate(s['python'])
    np.random.set_state(s['numpy'])
    torch.set_rng_state(s['torch'])
    torch.cuda.set_rng_state_all(s['cuda'])


def config(training_seed):
    assert training_seed in SEEDS, 'P2 authorizes only new training seeds2024/3407'
    cfg = yaml.safe_load((PKG / 'Reference_Only/B6_seed42_original_config.yaml').read_text())
    cfg['model']['pretrained'] = False
    cfg['project']['seed'] = training_seed
    cfg['training_seed'] = training_seed
    cfg['split_seed'] = 42
    return cfg


class HAMDataset(Dataset):
    def __init__(self, cfg, split):
        assert split in ['train', 'val'], 'P2 forbids test and external dataset roles'
        self.training_seed = cfg['training_seed']
        self.frame = pd.read_csv(ARCHIVE / f'splits/ham10000_seed42_{split}.csv')
        self.transform = build_transforms(cfg, split)
        self.train = split == 'train'
        self.metadata = torch.stack([encode_metadata(r, cfg['data']['metadata']) for _, r in self.frame.iterrows()])
        self.labels = self.frame.dx.map(LABEL_MAP).to_numpy(dtype=np.int64)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, key):
        epoch, index = key if isinstance(key, tuple) else (0, key)
        if self.train:
            # Stateless per-image augmentation is independent of aux RNG and worker scheduling.
            seed = self.training_seed + epoch * 1000003 + index * 1009
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed % (2**32))
        row = self.frame.iloc[index]
        with Image.open(REV / 'data/HAM10000/images' / f'{row.image_id}.jpg') as im:
            image = self.transform(im.convert('RGB'))
        return image, self.metadata[index], int(self.labels[index]), index


def worker_init(_):
    torch.set_num_threads(1)


def loader(dataset, keys, workers=2, batch_size=48):
    generator = torch.Generator().manual_seed(dataset.training_seed * 1000 + 17)
    return DataLoader(dataset, sampler=keys, batch_size=batch_size, num_workers=workers,
                      pin_memory=True, prefetch_factor=1, worker_init_fn=worker_init,
                      generator=generator, persistent_workers=False)


def load_aux():
    frame = pd.read_csv(OUT / 'fim_mask_train100.csv')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    images, masks = [], []
    for r in frame.itertuples():
        with Image.open(r.image_path) as im:
            images.append(transform(im.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)))
        with Image.open(r.mask_path) as im:
            masks.append(torch.from_numpy((np.array(im.convert('L').resize((224, 224), Image.Resampling.NEAREST)) > 127).astype(np.float32))[None])
    images, masks = torch.stack(images).pin_memory(), torch.stack(masks).pin_memory()
    assert images.shape == (100, 3, 224, 224) and masks.shape == (100, 1, 224, 224)
    assert masks.flatten(1).sum(1).min() > 0
    return images, masks


def optimizer_for(model):
    groups = [{'params': [p for n, p in model.named_parameters() if n.startswith('backbone.')], 'lr': 3e-5},
              {'params': [p for n, p in model.named_parameters() if not n.startswith('backbone.')], 'lr': 3e-4}]
    optimizer = torch.optim.AdamW(groups, lr=3e-4, weight_decay=.05)
    def factor(epoch):
        if epoch < 5:
            return (epoch + 1) / 5
        return .5 * (1 + torch.cos(torch.tensor((epoch - 5) / 75 * torch.pi))).item()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    return optimizer, scheduler


def class_weights(labels):
    counts = torch.bincount(torch.tensor(labels), minlength=7).float()
    weights = (1 - .999) / (1 - .999**counts).clamp_min(1e-12)
    weights = weights / weights.sum() * 7
    return weights.cuda(), ((len(labels) - counts[4]) / counts[4]).cuda()


def dice_bce(logits, target):
    logits = F.interpolate(logits.float(), size=target.shape[-2:], mode='bilinear', align_corners=False)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none').mean((1, 2, 3))
    pred = logits.sigmoid()
    intersection = (pred * target).sum((1, 2, 3))
    dice = 1 - (2 * intersection + 1) / (pred.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return (bce + dice).mean()


def observation(value):
    values = value.detach().float().flatten()
    return {'n': values.numel(), 'sum': float(values.sum()), 'sum_sq': float(values.square().sum()), 'max': float(values.max())}


def merge_diagnostics(total, current):
    for key, value in current.items():
        previous = total.setdefault(key, {'n': 0, 'sum': 0., 'sum_sq': 0., 'max': -float('inf')})
        for field in ['n', 'sum', 'sum_sq']:
            previous[field] += value[field]
        previous['max'] = max(previous['max'], value['max'])


def readout_diagnostics(out, scope):
    current = {}
    diag = out['readout_diagnostics']
    for name in ['region_weights', 'region_area', 'region_valid']:
        if name in diag:
            for i, region in enumerate(['interior', 'boundary_band', 'outer_context']):
                current[f'{scope}_{name}_{region}'] = observation(diag[name][:, i])
    current[f'{scope}_residual_coefficient'] = observation(diag['residual_coefficient'].expand(len(out['logits'])))
    return current


def summarize_diagnostics(total):
    result = {}
    for key, v in total.items():
        mean = v['sum'] / v['n']
        result[key + '_mean'] = mean
        result[key + '_sd_within_samples'] = math.sqrt(max(0., v['sum_sq'] / v['n'] - mean * mean))
        result[key + '_max'] = v['max']
    return result


def update_once(model, optimizer, scaler, batch, aux, aux_ids, weights, pos_weight, micro, mask_weight):
    optimizer.zero_grad(set_to_none=True)
    images, metadata, labels, _ = batch
    labels_gpu = labels.cuda(non_blocking=True)
    denominator = weights[labels_gpu].sum()
    n = len(labels)
    losses = []
    diagnostics = {}
    for start in range(0, n, micro):
        end = min(n, start + micro)
        y = labels_gpu[start:end]
        with torch.autocast('cuda', dtype=torch.float16):
            out = model(images[start:end].cuda(non_blocking=True), metadata[start:end].cuda(non_blocking=True))
            ce = F.cross_entropy(out['logits'].float(), y, weight=weights, reduction='sum') / denominator
            mel = F.binary_cross_entropy_with_logits(out['mel_logits'].float(), (y == 4).float(), pos_weight=pos_weight)
            consistency = F.mse_loss(out['mel_logits'].float().sigmoid(), out['logits'].float().softmax(1)[:, 4])
            attn = out['attention'].float()
            tv = (attn[:, :, 1:, :] - attn[:, :, :-1, :]).abs().mean() + (attn[:, :, :, 1:] - attn[:, :, :, :-1]).abs().mean()
            loss = ce + (end - start) / n * (.5 * mel + .05 * consistency + .0005 * attn.mean() + .002 * tv)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite classification loss')
        scaler.scale(loss).backward()
        losses.append(loss.detach())
        merge_diagnostics(diagnostics, readout_diagnostics(out, 'classification'))
    for start in range(0, 32, micro):
        ids = aux_ids[start:start + micro]
        x = aux[0][ids].cuda(non_blocking=True)
        mask = aux[1][ids].cuda(non_blocking=True)
        # Both conditions forward the same auxiliary images in training mode.
        with torch.set_grad_enabled(mask_weight > 0), torch.autocast('cuda', dtype=torch.float16):
            out = model(x, torch.zeros((len(ids), 22), device='cuda'))
            loss = dice_bce(out['mask_logits'], mask) * (len(ids) / 32) * mask_weight
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite auxiliary loss')
        if mask_weight:
            scaler.scale(loss).backward()
        losses.append(loss.detach())
        merge_diagnostics(diagnostics, readout_diagnostics(out, 'auxiliary'))
        with torch.no_grad():
            mask_prob = F.interpolate(out['mask_logits'].detach().float(), size=mask.shape[-2:], mode='bilinear', align_corners=False).sigmoid()
            aux_dice = (2 * (mask_prob * mask).sum((1, 2, 3)) + 1) / (mask_prob.sum((1, 2, 3)) + mask.sum((1, 2, 3)) + 1)
            merge_diagnostics(diagnostics, {'train_auxiliary_soft_dice': observation(aux_dice)})
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(norm):
        previous_scale = scaler.get_scale()
        scaler.update(new_scale=previous_scale * .5)
        optimizer.zero_grad(set_to_none=True)
        log(f'AMP_BACKOFF scale={previous_scale}->{scaler.get_scale()}; retry same update')
        return None
    mask_gradient = any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in model.mask_head.parameters())
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    if scaler.get_scale() < scale_before:
        raise RuntimeError('AMP skipped optimizer update')
    merge_diagnostics(diagnostics, {'gradient_norm_before_clip': observation(norm)})
    return float(torch.stack(losses).sum()), float(norm), mask_gradient, diagnostics


def update(model, optimizer, scaler, batch, aux, aux_ids, weights, pos_weight, micro, mask_weight):
    before_rng = rng_state()
    before_buffers = {name: value.clone() for name, value in model.named_buffers()}
    for attempt in range(9):
        result = update_once(model, optimizer, scaler, batch, aux, aux_ids, weights, pos_weight, micro, mask_weight)
        if result is not None:
            return result
        # Retry consumes neither a new data update nor extra BN/dropout exposure.
        with torch.no_grad():
            for name, value in model.named_buffers():
                value.copy_(before_buffers[name])
        restore_rng(before_rng)
    raise RuntimeError('Nonfinite gradients persist after eight AMP scale reductions')


@torch.inference_mode()
def validate(model, dataset, micro):
    assert dataset.train is False and len(dataset) == 1513
    model.eval()
    ps, ys = [], []
    diagnostics = {}
    for images, metadata, labels, _ in loader(dataset, list(range(len(dataset))), batch_size=micro):
        out = model(images.cuda(non_blocking=True), metadata.cuda(non_blocking=True))
        ps.append(out['logits'].softmax(1).cpu().numpy())
        ys.append(labels.numpy())
        merge_diagnostics(diagnostics, readout_diagnostics(out, 'validation'))
    p, y = np.concatenate(ps), np.concatenate(ys)
    assert np.isfinite(p).all()
    return {'macro_f1': float(f1_score(y, p.argmax(1), average='macro', labels=list(range(7)))),
            'mel_auc': float(roc_auc_score(y == 4, p[:, 4])),
            'mel_ap': float(average_precision_score(y == 4, p[:, 4])),
            'diagnostics': summarize_diagnostics(diagnostics)}


def fresh_model(cfg):
    training_seed = cfg['training_seed']
    assert training_seed in SEEDS
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        initial = torch.load(THREE / f'initial_state_s{training_seed}.pt', map_location='cpu', weights_only=True)
    assert initial['seed'] == cfg['training_seed']
    model = P2Model(cfg, cfg['p1_variant'], initial['model'])
    del initial
    return model.cuda()


def progress(status, current=None):
    runs = []
    for training_seed, variant in RUNS:
        path = OUT / f'{variant}_s{training_seed}/status.json'
        r = json.loads(path.read_text()) if path.exists() else {'status': 'NOT_STARTED', 'completed_epochs': 0}
        runs.append({'run': f'{variant}_s{training_seed}', **r})
    dump(OUT / 'P2_PROGRESS.json', {'status': status, 'updated_at': datetime.now().astimezone().isoformat(),
         'planned_new_runs': 6, 'reused_P1_runs': 3, 'readonly_M0_runs': 3,
         'training_completed': sum(r['status'] == 'TRAINING_COMPLETE' for r in runs),
         'M0': 'current FIM FULL three seeds HAM_val/NATIVE summaries only', 'current': current,
         'deadline': datetime.fromtimestamp(DEADLINE).astimezone().isoformat(), 'runs': runs,
         'forbidden_eval_roles': ['HAM_test', 'SIIM', 'new_external'], 'automatic_next_phase': False})


def smoke(cfg, train, aux):
    from p2_checks import check_models
    return check_models(cfg, train, aux)


def run(condition, cfg, train, val, aux, micro):
    training_seed = cfg['training_seed']
    assert condition in VARIANTS and training_seed in SEEDS
    cfg = {**cfg, 'p1_variant': condition}
    run_name = f'{condition}_s{training_seed}'
    run_dir = OUT / run_name
    run_dir.mkdir(exist_ok=True)
    if (run_dir / 'status.json').exists() and json.loads((run_dir / 'status.json').read_text())['status'] == 'TRAINING_COMPLETE':
        return True
    seed_all(training_seed)
    model = fresh_model(cfg)
    optimizer, scheduler = optimizer_for(model)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
    weights, pos = class_weights(train.labels)
    mask_weight = .1
    effective = {'original_config': cfg, 'condition': condition, 'lambda_mask_aux': mask_weight, 'seed': training_seed,
        'training_seed': training_seed, 'split_seed': 42, 'run_name': run_name,
        'classification_order_seed': 'training_seed + epoch (zero-based)', 'dataloader_generator_seed': training_seed * 1000 + 17,
        'classification_effective_batch': 48, 'auxiliary_effective_batch': 32, 'microbatch': micro, 'workers': 2,
        'precision_train': 'FP16 AMP, FP32 parameters/loss reductions', 'precision_eval': 'FP32', 'TF32': False,
        'amp_overflow_policy': 'halve scale; restore pre-update BN buffers and RNG; retry same effective batch (up to 8 retries), no skipped data updates',
        'max_epochs': 80, 'patience': 15, 'warmup_epochs': 5, 'selection': 'val macro-F1, then softmax MEL AUC, earliest exact tie',
        'label_smoothing_effective': 0., 'class_weights': weights.cpu().tolist(), 'bce_pos_weight': float(pos),
        'aux_metadata': [0.] * 22, 'aux_geometry': 'shared direct 224 square resize; image bilinear, mask nearest',
        'aux_sampling': '32 distinct fixed-set members per update, PCG64 SeedSequence([training_seed, epoch, 19]); matched conditions',
        'classification_aug_seed': 'training_seed + epoch*1000003 + dataset_index*1009',
        'initialization': json.loads((OUT / f'P2_INITIALIZATION_s{training_seed}.json').read_text(encoding='utf-8')),
        'mask_manifest_sha256': json.loads((OUT / 'DATA_CHECK.json').read_text(encoding='utf-8'))['selected_manifest_sha256'],
        'source_paths': {'model': str(PKG / 'Reference_Only/model_code/lesionmamba/models/oncoderm_mambax.py'),
                         'candidate': str(PKG / 'region_readout_reference.py'), 'integration': str(OUT / 'p2_model.py')},
        'new_module_seed': training_seed + 1000, 'new_module_initial_hashes': model.initial_readout_hashes,
        'readout_rng_isolated': model.readout_rng_isolated,
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'readout_parameter_count': sum(p.numel() for p in model.readout.parameters()),
        'allowed_evaluation': ['HAM_validation'], 'AP_used_for_selection': False}
    if (run_dir / 'effective_config.json').exists():
        assert json.loads((run_dir / 'effective_config.json').read_text(encoding='utf-8')) == effective, 'Resume configuration changed'
    else:
        dump(run_dir / 'effective_config.json', effective)
    epoch, next_batch, history, best_pair, best_epoch, global_step = 0, 0, [], (-1., -1.), -1, 0
    epoch_loss, epoch_samples, elapsed_prior = 0., 0, 0.
    epoch_diagnostics, diagnostic_history = {}, []
    if (run_dir / 'last.pt').exists():
        state = torch.load(run_dir / 'last.pt', map_location='cpu', weights_only=False)
        assert state['config'] == effective, 'Checkpoint seed/configuration mismatch'
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        epoch, next_batch, history = state['epoch'], state['next_batch'], state['history']
        best_pair, best_epoch, global_step = state['best_pair'], state['best_epoch'], state['global_step']
        epoch_loss, epoch_samples, elapsed_prior = state['epoch_loss'], state['epoch_samples'], state['elapsed_prior']
        epoch_diagnostics, diagnostic_history = state['epoch_diagnostics'], state['diagnostic_history']
        restore_rng(state['rng'])
        del state
    steps = math.ceil(len(train) / 48)
    run_started = time.time()
    log(f'START {run_name} epoch={epoch + 1} next_batch={next_batch} micro={micro} effective=48+32')

    def checkpoint(next_epoch, batch, loss_sum, sample_count, elapsed, status):
        save(run_dir / 'last.pt', {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'rng': rng_state(),
             'epoch': next_epoch, 'next_batch': batch, 'history': history, 'best_pair': best_pair,
             'best_epoch': best_epoch, 'global_step': global_step, 'epoch_loss': loss_sum,
             'epoch_samples': sample_count, 'elapsed_prior': elapsed, 'config': effective,
             'epoch_diagnostics': epoch_diagnostics, 'diagnostic_history': diagnostic_history})
        dump(run_dir / 'status.json', {'status': status, 'completed_epochs': len(history),
             'best_epoch': best_epoch + 1 if best_epoch >= 0 else None, 'next_epoch': next_epoch + 1,
             'next_batch': batch, 'global_step': global_step, 'last_checkpoint': str(run_dir / 'last.pt'),
             'elapsed_this_invocation_seconds': time.time() - run_started})
        progress('RUNNING' if status == 'RUNNING' else status, run_name)

    for current in range(epoch, 80):
        epoch_start = time.time()
        model.train()
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(training_seed + current)).tolist()
        aux_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([training_seed, current, 19])))
        aux_batches = [aux_rng.choice(100, 32, replace=False) for _ in range(steps)]
        first_batch = next_batch if current == epoch else 0
        keys = [(current, i) for i in order[first_batch * 48:]]
        for batch_id, batch in enumerate(loader(train, keys), start=first_batch):
            loss, norm, _, diagnostics = update(model, optimizer, scaler, batch, aux, aux_batches[batch_id], weights, pos, micro, mask_weight)
            merge_diagnostics(epoch_diagnostics, diagnostics)
            global_step += 1
            epoch_loss += loss * len(batch[2])
            epoch_samples += len(batch[2])
            if batch_id % 20 == 0:
                log(f'{run_name} epoch={current + 1}/80 update={batch_id + 1}/{steps} loss={loss:.6f}')
                dump(run_dir / 'status.json', {'status': 'RUNNING', 'completed_epochs': len(history),
                     'epoch': current + 1, 'batch': batch_id + 1, 'global_step': global_step,
                     'best_epoch': best_epoch + 1 if best_epoch >= 0 else None})
                progress('RUNNING', run_name)
            if time.time() >= DEADLINE or shutil.disk_usage(OUT).free < 2 * 1024**3 or (OUT / 'STOP_REQUESTED').exists():
                checkpoint(current, batch_id + 1, epoch_loss, epoch_samples, elapsed_prior + time.time() - epoch_start, 'PARTIAL')
                log(f'PARTIAL {run_name}: deadline, low disk, or local stop marker; checkpoint saved')
                return False
        validation = validate(model, val, micro)
        pair = (validation['macro_f1'], validation['mel_auc'])
        improved = pair > best_pair
        if improved:
            best_pair, best_epoch = pair, current
            save(run_dir / 'best.pt', {'model': model.state_dict(), 'config': effective, 'epoch': current + 1,
                 'selection_validation': validation})
        scheduler.step()
        seconds = elapsed_prior + time.time() - epoch_start
        history.append({'epoch': current + 1, 'train_loss': epoch_loss / epoch_samples,
             'val_macro_f1': pair[0], 'val_mel_auc': pair[1], 'val_mel_ap': validation['mel_ap'], 'best': improved,
             'epoch_seconds': seconds, 'global_step': global_step,
             'backbone_lr_next': optimizer.param_groups[0]['lr']})
        diagnostic_history.append({'epoch': current + 1, 'eta_raw': float(model.readout.eta.detach()),
             **summarize_diagnostics(epoch_diagnostics), **validation['diagnostics']})
        pd.DataFrame(history).to_csv(run_dir / 'training_history.csv', index=False)
        pd.DataFrame(diagnostic_history).to_csv(run_dir / 'readout_diagnostics.csv', index=False)
        complete = current == 79 or current - best_epoch >= 15
        epoch_diagnostics = {}
        checkpoint(current + 1, 0, 0., 0, 0., 'TRAINING_COMPLETE' if complete else 'RUNNING')
        log(f'EPOCH {run_name} {current + 1}/80 seconds={seconds:.1f} val_macro_f1={pair[0]:.6f} val_mel_auc={pair[1]:.6f} best={improved}')
        epoch_loss, epoch_samples, elapsed_prior, next_batch = 0., 0, 0., 0
        if current == 1:
            estimate = {'run': run_name, 'completed_epochs': 2,
                'epoch_seconds': [h['epoch_seconds'] for h in history],
                'elapsed_training_seconds': sum(h['epoch_seconds'] for h in history),
                'estimated_remaining_to_max80_seconds': np.mean([h['epoch_seconds'] for h in history]) * 78,
                'early_stopping_can_reduce_this': True, 'actual_validation': history[-1]}
            dump(run_dir / 'TWO_EPOCH_PROGRESS.json', estimate)
            log('TWO_EPOCH_PROGRESS ' + json.dumps(estimate))
        if complete:
            log(f'TRAINING_COMPLETE {run_name} completed_epochs={current + 1} best_epoch={best_epoch + 1}')
            return True
        if time.time() >= DEADLINE:
            checkpoint(current + 1, 0, 0., 0, 0., 'PARTIAL')
            return False
    return True


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if time.time() >= DEADLINE:
        progress('PARTIAL')
        return
    from p2_checks import check_inputs
    check_inputs()
    aux = load_aux()
    progress('TECHNICAL_SMOKE')
    for training_seed in SEEDS:
        cfg = config(training_seed)
        train, val = HAMDataset(cfg, 'train'), HAMDataset(cfg, 'val')
        smoke_path = OUT / f'CUDA_TECHNICAL_CHECK_s{training_seed}.json'
        if smoke_path.exists():
            smoke_result = json.loads(smoke_path.read_text(encoding='utf-8'))
            assert smoke_result['status'] == 'PASSED' and smoke_result['microbatch'] == 8
            assert smoke_result['training_seed'] == training_seed
            micro = 8
        else:
            micro = smoke(cfg, train, aux)
        log(f'TECHNICAL_CHECK_PASSED; formal models reset to original pre-training seed{training_seed} state')
        for condition in VARIANTS:
            if time.time() >= DEADLINE or (OUT / 'STOP_REQUESTED').exists():
                progress('PARTIAL', f'{condition}_s{training_seed}')
                return
            if not run(condition, cfg, train, val, aux, micro):
                return
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            from p2_finish import evaluate_run
            evaluate_run(training_seed, condition)
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    progress('ALL_SIX_COMPLETE_SUMMARY_PENDING')


if __name__ == '__main__':
    sys.modules['p2_training'] = sys.modules[__name__]
    try:
        main()
    except Exception:
        error = traceback.format_exc()
        log(error)
        dump(OUT / f'TRAINING_ERROR_{datetime.now():%Y%m%d_%H%M%S}.json', {'error': error, 'at': datetime.now().astimezone().isoformat()})
        progress('BLOCKED')
        raise
