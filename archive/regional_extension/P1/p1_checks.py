"""Narrow P1 input and integration checks, with no held-out evaluation."""
import csv
import ctypes
import gc
import hashlib
import inspect
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import p1_training as t
from p1_model import P1Model


def resource_snapshot(stage):
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (n, ctypes.c_ulonglong) for n in ['total_physical', 'available_physical', 'total_page_file',
            'available_page_file', 'total_virtual', 'available_virtual', 'available_extended_virtual']]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    row = {'stage': stage, 'time': time.time(), **{name: getattr(status, name) for name, _ in status._fields_}}
    if torch.cuda.is_initialized():
        row.update(cuda_free_bytes=torch.cuda.mem_get_info()[0], cuda_allocated_bytes=torch.cuda.memory_allocated(),
                   cuda_reserved_bytes=torch.cuda.memory_reserved())
    with (t.OUT / 'RESOURCE_LOG.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(row) + '\n')
    t.log(f"RESOURCE {stage} physical_free_GiB={status.available_physical / 1024**3:.2f} commit_headroom_GiB={status.available_page_file / 1024**3:.2f}")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def check_inputs():
    assert torch.cuda.is_available(), 'CUDA is required'
    expected = {
        t.OLD / 'initial_state_s42.pt': 'cb1dc95021bb234e039a1819ab1aca3519c175fc0dc9555b1963c5914d0b8fcc',
        t.PKG / 'Reference_Only/model_code/lesionmamba/models/oncoderm_mambax.py': '7e737a004f69053821628f84b3da9c44a4e4a208a386fa7322c5c7ff9331a0c5',
        t.ARCHIVE / 'splits/ham10000_seed42_train.csv': 'f30b240e75191f37d765cefd5f67b352cd89c0c543e0a5e2623204f3ac833816',
        t.ARCHIVE / 'splits/ham10000_seed42_val.csv': 'b1e4982d61df694d6acc64a5c024f9423a110d87489a2731c5e361f183caf4f3',
        t.OUT / 'fim_mask_train100.csv': 'ef507cb90b4955b2a9f7dabbdc532a1bf9176899c72c46f646231f795c648a13',
    }
    evidence = []
    for path, digest in expected.items():
        assert path.is_file(), str(path)
        actual = sha(path)
        assert actual == digest, f'Hash mismatch: {path}'
        evidence.append({'path': str(path), 'sha256': actual, 'bytes': path.stat().st_size})
    frames = [pd.read_csv(t.ARCHIVE / f'splits/ham10000_seed42_{s}.csv') for s in ['train', 'val']]
    assert [len(f) for f in frames] == [6982, 1513]
    for frame in frames:
        assert frame.image_id.is_unique
        assert all((t.REV / 'data/HAM10000/images' / f'{i}.jpg').is_file() for i in frame.image_id)
    assert not set(frames[0].image_id) & set(frames[1].image_id)
    assert not set(frames[0].lesion_id.dropna()) & set(frames[1].lesion_id.dropna())
    aux = pd.read_csv(t.OUT / 'fim_mask_train100.csv')
    assert len(aux) == 100 and aux.image_id.is_unique
    assert not set(aux.image_id) & set(frames[1].image_id)
    for row in aux.itertuples():
        assert sha(row.image_path) == row.image_sha256, f'Aux image changed: row {row.selection_order}'
        assert sha(row.mask_path) == row.mask_sha256, f'Aux mask changed: row {row.selection_order}'
    t.dump(t.OUT / 'DATA_CHECK.json', {'status': 'PASSED', 'HAM_train': 6982, 'HAM_val': 1513, 'aux': 100,
        'selected_manifest_sha256': expected[t.OUT / 'fim_mask_train100.csv'],
        'aux_original_image_and_mask_hashes_matched': 100, 'train_val_image_and_lesion_disjoint': True,
        'frozen_members_reused_not_reselected': True,
        'previous_aux_eligibility_rule': str(t.OLD / 'MASK_SELECTION_RULE.md'),
        'no_new_test_or_external_reads': True, 'input_files': evidence})
    init = json.loads((t.OLD / 'INITIALIZATION.json').read_text(encoding='utf-8'))
    assert init['initial_state_sha256'] == expected[t.OLD / 'initial_state_s42.pt']
    t.dump(t.OUT / 'P1_INITIALIZATION.json', {'base': init, 'training_seed': 42, 'new_module_seed': 1042,
        'source_loaded': str(t.OLD / 'initial_state_s42.pt'), 'strict_base_load': True,
        'trained_weights_used_for_initialization': False, 'new_module_rng_isolated': True})
    extracted = {}
    for name in ['METRICS.csv', 'INTERNAL_MULTICLASS_METRICS.csv']:
        with (t.OLD / name).open(encoding='utf-8-sig', newline='') as stream:
            rows = [r for r in csv.DictReader(stream) if r['condition'] == 'FULL' and r['cohort'] == 'HAM_val'
                    and r.get('input', 'NATIVE') == 'NATIVE']
        assert len(rows) == (2 if name == 'METRICS.csv' else 1)
        extracted[name] = {'source': str(t.OLD / name), 'selected_rows_only': rows}
    extracted['best_identity'] = json.loads((t.OLD / 'FULL_EVALUATION_MODEL.json').read_text(encoding='utf-8'))
    extracted['status'] = json.loads((t.OLD / 'FIM_LPAH_FULL_s42/status.json').read_text(encoding='utf-8'))
    t.dump(t.OUT / 'M0_VALIDATION_REFERENCE.json', extracted)
    t.log('INPUT_CHECK_PASSED original initial state, train/val, fixed aux100; M0 validation records extracted')


def check_models(cfg, train, aux):
    assert torch.cuda.is_available()
    generator = torch.Generator().manual_seed(42)
    logits = torch.randn(23, 7, generator=generator, requires_grad=True)
    labels = torch.arange(23) % 7
    weights = torch.arange(1, 8).float()
    expected = F.cross_entropy(logits, labels, weight=weights)
    actual = sum(F.cross_entropy(logits[i:i+8], labels[i:i+8], weight=weights, reduction='sum')
                 for i in range(0, 23, 8)) / weights[labels].sum()
    assert torch.allclose(expected, actual, atol=1e-7)
    refused = []
    for role in ['test', 'SIIM', 'external']:
        try:
            t.HAMDataset(cfg, role)
        except AssertionError:
            refused.append(role)
    assert len(refused) == 3
    resource_snapshot('before_active_loader_check')
    active_loader = t.loader(train, [(0, i) for i in range(96)])
    active_iterator = iter(active_loader)
    batch = next(active_iterator)
    resource_snapshot('two_workers_live')
    x, meta = batch[0][:2].cuda(), batch[1][:2].cuda()
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        initial = torch.load(t.OLD / 'initial_state_s42.pt', map_location='cpu', weights_only=True)
    t.seed_all(42)
    base = t.OncoDermMambaX(cfg)
    base.load_state_dict(initial['model'], strict=True)
    base = base.cuda().eval()
    with torch.no_grad():
        expected_outputs = {k: v.cpu() for k, v in base(x, meta).items()}
    del base
    gc.collect()
    torch.cuda.empty_cache()
    records, init_hashes = [], {}
    class_w, pos = t.class_weights(train.labels)
    for variant in t.VARIANTS:
        t.seed_all(42)
        model = P1Model(cfg, variant, initial['model']).cuda().eval()
        init_hashes[variant] = model.initial_readout_hashes
        calls = {'backbone': 0, 'fuse': 0, 'mask_head': 0, 'metadata_encoder': 0}
        def counter(key):
            def hook(*_):
                calls[key] += 1
            return hook
        hooks = [getattr(model, key).register_forward_hook(counter(key)) for key in calls]
        with torch.no_grad():
            output = model(x, meta)
        for hook in hooks:
            hook.remove()
        assert all(v == 1 for v in calls.values()), calls
        differences = {k: float((output[k].cpu() - v).abs().max()) for k, v in expected_outputs.items()}
        assert all(torch.allclose(output[k].cpu(), v, atol=1e-6, rtol=1e-5) for k, v in expected_outputs.items())
        with torch.no_grad():
            model.readout.eta.fill_(.2)
        output = model(x, meta)
        output['features'].retain_grad()
        output['logits'].square().mean().backward()
        assert float(output['features'].grad.abs().sum()) > 0
        assert all(p.grad is None or not bool(p.grad.abs().sum()) for p in model.mask_head.parameters())
        assert any(p.grad is not None and bool(p.grad.abs().sum()) for p in model.readout.parameters())
        assert any(p.grad is not None and bool(p.grad.abs().sum()) for p in model.metadata_encoder.parameters())
        model.zero_grad(set_to_none=True)
        mask_output = model(aux[0][:2].cuda(), torch.zeros(2, 22, device='cuda'))
        t.dice_bce(mask_output['mask_logits'], aux[1][:2].cuda()).backward()
        assert any(p.grad is not None and bool(p.grad.abs().sum()) for p in model.mask_head.parameters())
        with torch.no_grad():
            Fmap = output['features'].detach()
            e0 = model._pool(Fmap, output['attention'].detach())
            h = model.metadata_encoder(meta)
            if variant != 'XATTN':
                _, da = model.readout(Fmap, output['mask_logits'].detach(), e0, h)
                _, db = model.readout(Fmap, output['mask_logits'].detach(), e0, h + torch.linspace(-1, 1, 768, device='cuda'))
                alpha_change = float((da['region_weights'] - db['region_weights']).abs().max())
                assert alpha_change == 0 if variant == 'R_IMG' else alpha_change > 0
            else:
                alpha_change = None
            original_logits = model(x, meta)['logits']
            missing_logits = model(x, torch.zeros_like(meta))['logits']
            assert not torch.equal(original_logits, missing_logits), 'Original CMGF must remain live'
        t.dump(t.OUT / f'{variant}_INITIALIZATION_CHECK.json', {
            'base_key_mapping': model.base_key_mapping, 'all_base_tensors_equal_before_new_readout': True,
            'new_parameter_hashes': model.initial_readout_hashes, 'new_parameter_seed': 1042,
            'model_source': inspect.getfile(t.OncoDermMambaX), 'integration_source': inspect.getfile(P1Model),
            'readout_source': inspect.getfile(type(model.readout)),
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'readout_parameter_count': sum(p.numel() for p in model.readout.parameters())})
        del model, output, mask_output, Fmap, e0, h
        gc.collect()
        torch.cuda.empty_cache()
        t.seed_all(42)
        model = t.fresh_model({**cfg, 'p1_variant': variant}).train()
        optimizer, scheduler = t.optimizer_for(model)
        scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
        resource_snapshot(variant + '_before_complete_update_workers_live')
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        loss, norm, mask_grad, diagnostics = t.update(model, optimizer, scaler, batch, aux, np.arange(32), class_w, pos, 8, .1)
        torch.cuda.synchronize()
        assert mask_grad and np.isfinite(loss) and np.isfinite(norm)
        resource_snapshot(variant + '_after_complete_update_workers_live')
        records.append({'variant': variant, 'eta0_max_abs_differences': differences, 'single_forward_calls': calls,
            'classification_mask_gradient_blocked': True, 'auxiliary_mask_gradient_live': True,
            'classification_feature_and_metadata_gradients_live': True, 'CMGF_metadata_response': True,
            'region_alpha_metadata_change': alpha_change, 'AMP_update_loss': loss, 'AMP_update_norm': norm,
            'AMP_scale_after': scaler.get_scale(), 'update_seconds': time.perf_counter() - started,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'diagnostics': t.summarize_diagnostics(diagnostics)})
        t.log(f'CUDA_TECHNICAL_PASSED {variant}: FP32 identity, gradient isolation, AMP effective48+aux32')
        del model, optimizer, scheduler, scaler
        gc.collect()
        torch.cuda.empty_cache()
    assert init_hashes['R_IMG'] == init_hashes['R_CLIN']
    del active_iterator, active_loader
    gc.collect()
    t.dump(t.OUT / 'CUDA_TECHNICAL_CHECK.json', {'status': 'PASSED', 'microbatch': 8, 'workers': 2,
        'weighted_CE_tail_batch_equivalence': True, 'forbidden_loaders_refused': refused,
        'two_DataLoader_workers_kept_alive_during_CUDA_updates': True,
        'region_candidates_initial_readout_equal': True, 'formal_training_resets': True,
        'cuda_device': torch.cuda.get_device_name(), 'torch_version': str(torch.__version__), 'records': records})
    return 8
