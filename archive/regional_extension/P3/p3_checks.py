"""Scoped input, loss branch and complete-state replay checks; no held-out access."""
import ast
import gc
import hashlib
import json
from pathlib import Path
import torch
import p3_training as t
from p3_initialization import sha


def check_inputs():
    assert torch.cuda.is_available()
    p2 = Path(t.WINDOW['source_P2'])
    delivery = Path(t.WINDOW['source_P2_delivery'])
    assert json.loads((delivery / 'P2_PROGRESS.json').read_text())['status'] == 'COMPLETE'
    sources = []
    for path in (t.PKG / 'model_code').rglob('*.py'):
        original = p2 / 'FIM_RegionClinical_Dev_Start/Reference_Only/model_code' / path.relative_to(t.PKG / 'model_code')
        assert sha(path) == sha(original)
        sources.append({'local': str(path), 'original': str(original), 'sha256': sha(path)})
    a = ast.parse((p2 / 'p2_training.py').read_text())
    b = ast.parse((t.OUT / 'p3_training.py').read_text())
    unchanged = ['optimizer_for', 'class_weights', 'loader', 'seed_all', 'rng_state', 'restore_rng', 'update']
    for name in unchanged:
        assert ast.dump(next(n for n in a.body if getattr(n, 'name', '') == name)) == ast.dump(next(n for n in b.body if getattr(n, 'name', '') == name)), name
    w0 = ast.parse((p2 / 'p2_worker_data.py').read_text())
    w1 = ast.parse((t.OUT / 'p3_worker_data.py').read_text())
    c0 = next(n for n in w0.body if getattr(n, 'name', '') == 'HAMDataset')
    c1 = next(n for n in w1.body if getattr(n, 'name', '') == 'HAMDataset')
    for name in ['__getitem__', '__len__']:
        assert ast.dump(next(n for n in c0.body if n.name == name)) == ast.dump(next(n for n in c1.body if n.name == name))
    t.dump(t.OUT / 'INPUT_CHECK.json', {'status': 'PASSED', 'GPU': torch.cuda.get_device_name(),
        'source_files': sources, 'unchanged_function_AST': unchanged,
        'worker_image_transform_seed_formula_AST_unchanged': True,
        'no_auxiliary_reader_in_B2S': not hasattr(t, 'load_aux')})


def model_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def check_run(cfg, train, val):
    seed = cfg['training_seed']
    assert len(train) == 6982 and len(val) == 1513
    assert train.metadata.shape == (6982, 22)
    assert not set(train.frame.image_id) & set(val.frame.image_id)
    assert not set(train.frame.lesion_id.dropna()) & set(val.frame.lesion_id.dropna())
    t.seed_all(seed)
    model = t.fresh_model(cfg)
    assert not model.use_cssa and not model.use_prev_fusion and not model.use_mask_aux
    assert model.use_metadata and model.use_mel_head and model.metadata_dim == 22
    optimizer, scheduler = t.optimizer_for(model)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
    weights, pos = t.class_weights(train.labels)
    ids = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed)).tolist()[:48]
    batches = list(t.loader(train, [(0, i) for i in ids]))
    assert len(batches) == 1
    batch = batches[0]
    reference = train[(0, ids[0])]
    assert torch.equal(batch[0][0], reference[0]) and torch.equal(batch[1][0], reference[1])
    t.seed_all(seed)
    count = {'mask_head': 0, 'prev_proj': 0}
    handles = []
    for name in count:
        def hook(module, args, output, name=name):
            count[name] += 1
        handles.append(getattr(model, name).register_forward_hook(hook))
    model.train()
    result = t.update(model, optimizer, scaler, batch, None, None, weights, pos, 8, 0.)
    assert result is not None and not result[2] and count == {'mask_head': 0, 'prev_proj': 0}
    grad = {name: any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in getattr(model, name).parameters())
            for name in ['backbone', 'fuse', 'attention_from_feature', 'metadata_encoder', 'metadata_gate', 'head', 'mel_head']}
    assert all(grad.values()), grad
    for h in handles:
        h.remove()
    # Verify full-state replay on the same scoped effective batch, not a formal epoch.
    path = t.OUT / f'technical_resume_s{seed}.pt'
    t.save(path, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'rng': t.rng_state(),
        'epoch': 0, 'next_batch': 1, 'training_seed': seed})
    t.update(model, optimizer, scaler, batch, None, None, weights, pos, 8, 0.)
    expected = model_hash(model)
    state = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
    scaler.load_state_dict(state['scaler']); t.restore_rng(state['rng'])
    del state
    t.update(model, optimizer, scaler, batch, None, None, weights, pos, 8, 0.)
    assert expected == model_hash(model), 'Complete checkpoint replay mismatch'
    t.dump(t.OUT / f'B2S_TECHNICAL_CHECK_s{seed}.json', {'status': 'PASSED',
        'training_seed': seed, 'microbatch': 8, 'effective_batch': 48, 'workers': 2,
        'HAM_train': 6982, 'HAM_val': 1513, 'worker_first_image_exact': True,
        'unused_module_forward_counts': count, 'active_classification_gradient_branches': grad,
        'strict_weight_loading': True, 'complete_optimizer_scheduler_scaler_rng_replay_exact': True,
        'formal_training_reinitializes_from_original_initial_state': True,
        'source': str(t.PKG / 'model_code/lesionmamba/models/oncoderm_mambax.py')})
    del model, optimizer, scheduler, scaler, batch, batches, reference, weights, pos
    gc.collect(); torch.cuda.empty_cache()
