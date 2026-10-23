"""Small P6 technical gates; test states never become formal initialization."""
import gc
import inspect
from pathlib import Path
import time
from p6_common import OUT, PKG, OLD, P1, read, dump, sha


def check_inputs():
    import numpy as np
    import torch
    import p6_training as t
    from p6_model import P6Model, state_sha
    assert torch.cuda.is_available(), 'Local CUDA GPU unavailable'
    cfg = t.config(42)
    train, val = t.HAMDataset(cfg, 'train'), t.HAMDataset(cfg, 'val')
    assert len(train) == 6982 and len(val) == 1513
    assert train.metadata.shape == (6982, 22) and val.metadata.shape == (1513, 22)
    assert not set(train.frame.image_id) & set(val.frame.image_id)
    assert not set(train.frame.lesion_id) & set(val.frame.lesion_id)
    for dataset in [train, val]:
        assert set(dataset.labels).issubset(set(range(7)))
        assert torch.isfinite(dataset.metadata).all()
        assert all((t.REV / 'data/HAM10000/images' / f'{name}.jpg').is_file() for name in dataset.frame.image_id)
    for forbidden in ['test', 'SIIM', 'Derm7pt']:
        try:
            t.HAMDataset(cfg, forbidden)
        except AssertionError:
            pass
        else:
            raise AssertionError('Forbidden dataset role admitted')
    t.seed_all(42)
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        original = torch.load(OLD / 'initial_state_s42.pt', map_location='cpu', weights_only=True)
    assert original['seed'] == 42
    initial_record = read(OLD / 'INITIALIZATION.json')
    assert sha(OLD / 'initial_state_s42.pt') == initial_record['initial_state_sha256']
    before = t.rng_state()
    region = P6Model(cfg, 'region', original['model'])
    region_rng = t.rng_state()
    region_hash = state_sha(region.state_dict())
    region_new = {k: v.clone() for k, v in region.readout.state_dict().items()}
    t.restore_rng(before)
    control = P6Model(cfg, 'global', original['model'])
    assert torch.equal(region_rng['torch'], torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(region_rng['cuda'], torch.cuda.get_rng_state_all()))
    assert region.state_dict().keys() == control.state_dict().keys()
    assert all(torch.equal(v, control.state_dict()[k]) for k, v in region.state_dict().items())
    assert all(torch.equal(v, control.readout.state_dict()[k]) for k, v in region_new.items())
    assert all(torch.equal(v, region.state_dict()[k]) for k, v in original['model'].items())
    assert all(torch.equal(v, control.state_dict()[k]) for k, v in original['model'].items())
    assert not any(any(word in k for word in ['alpha', 'score', 'region_type']) for k in region.readout.state_dict())
    initial_path = OUT / 'initial_state_p6_s42.pt'
    if initial_path.exists():
        saved = torch.load(initial_path, map_location='cpu', weights_only=True)
        assert state_sha(saved['model']) == region_hash == saved['state_sha256']
        del saved
    else:
        t.save(initial_path, {'model': region.state_dict(), 'training_seed': 42,
                             'new_module_seed': 6042, 'state_sha256': region_hash})
    identity = {'status': 'PASSED', 'source_initialization': initial_record,
        'shared_initial_state': str(initial_path), 'shared_initial_state_sha256': sha(initial_path),
        'state_tensor_sha256': region_hash, 'new_readout_tensor_sha256': state_sha(region.readout.state_dict()),
        'new_readout_seed': 6042, 'shared_base_exact': True, 'all_new_parameters_exact': True,
        'state_dict_keys_shapes_values_identical': True, 'original_model_tensor_count': len(original['model']),
        'new_readout_hashes': region.initial_readout_hashes,
        'parameter_count': sum(p.numel() for p in region.parameters()),
        'readout_parameter_count': sum(p.numel() for p in region.readout.parameters()),
        'readout_rng_restored': True, 'model_source': inspect.getfile(type(region)),
        'base_source': inspect.getfile(type(region).__bases__[0]),
        'worker_source': inspect.getfile(t.HAMDataset)}
    dump(OUT / 'P6_INITIALIZATION.json', identity)
    for mode in ['REGION', 'GLOBAL']:
        dump(OUT / f'FIM_TKEEP_{mode}_s42_INITIALIZATION.json', {**identity, 'mode': mode.lower()})
    dump(OUT / 'RUNTIME.json', {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
        'GPU': torch.cuda.get_device_name(), 'device_total_bytes': torch.cuda.get_device_properties(0).total_memory,
        'TF32': False, 'workers': 2, 'spawn_entry_stdlib_only': True,
        'train_count': len(train), 'validation_count': len(val), 'validation_positive': int(np.sum(val.labels == 4)),
        'actual_model_source': identity['model_source'], 'actual_base_source': identity['base_source']})
    del region, control, original, region_new, train, val
    gc.collect()
    torch.cuda.empty_cache()


def gradient_norm(module):
    import torch
    values = [p.grad.detach().float().norm() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(values).norm()) if values else None


def check_models(cfg, train, aux):
    import numpy as np
    import torch
    import torch.nn.functional as F
    import p6_training as t
    from p6_model import TokenPreservingReadout
    began = time.perf_counter()
    t.seed_all(42)
    tests = {}
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(6042)
        r = TokenPreservingReadout('region').cuda()
        g = TokenPreservingReadout('global').cuda()
        g.load_state_dict(r.state_dict(), strict=True)
        feature = torch.randn(2, 768, 14, 14, device='cuda', requires_grad=True)
        mask = torch.randn(2, 1, 14, 14, device='cuda', requires_grad=True)
        original_pool = feature.mean((-2, -1))
        zero, _ = r(feature, mask, original_pool)
        assert torch.equal(zero, original_pool)
        with torch.no_grad():
            r.eta.fill_(.2)
            g.eta.fill_(.2)
        changed_mask = mask.roll(7, -1)
        original, diag = r(feature, mask, original_pool)
        changed, _ = r(feature, changed_mask, original_pool)
        global_a, global_diag = g(feature, mask, original_pool)
        global_b, _ = g(feature, changed_mask, original_pool)
        regional_response = float((original - changed).abs().max())
        assert regional_response > 1e-7 and torch.equal(global_a, global_b)
        feature_grad, mask_grad = torch.autograd.grad(original.square().sum(), (feature, mask), allow_unused=True)
        assert feature_grad is not None and torch.isfinite(feature_grad).all() and feature_grad.abs().sum() > 0
        assert mask_grad is None
        assert original.shape == (2, 768) and torch.isfinite(original).all()
        tests['synthetic'] = {'eta0_exact_pool_identity': True, 'nonzero_eta_region_mask_response_max': regional_response,
            'global_mask_invariant_output': True, 'detached_mask_weights': True, 'feature_gradient_live': True,
            'state_keys_shapes_match': True, 'nonfinite': False}
        del r, g, feature, mask, original_pool, zero, original, changed, global_a, global_b, feature_grad, diag, global_diag
    keys = [(0, i) for i in range(48)]
    iterator = iter(t.loader(train, keys))
    batch = next(iterator)
    reference = t.ReferenceHAMDataset(cfg, 'train')
    assert torch.equal(reference.metadata, train.metadata)
    for j in range(4):
        item = reference[keys[j]]
        assert torch.equal(item[0], batch[0][j]) and torch.equal(item[1], batch[1][j]) and item[2] == batch[2][j]
    tests['worker_isolation'] = {'two_workers_used': True, 'metadata_exact': True,
        'first_four_augmented_images_exact_to_original_dataset': True,
        'dataset_source': str(Path(t.HAMDataset.__module__.replace('.', '/')))}
    del reference
    for mode in ['region', 'global']:
        t.seed_all(42)
        model = t.fresh_model({**cfg, 'readout_mode': mode})
        model.eval()
        calls = {key: 0 for key in ['backbone', 'fuse', 'mask_head', 'metadata_encoder', 'metadata_gate']}
        handles = []
        for key in calls:
            def hook(module, inputs, output, name=key):
                calls[name] += 1
            handles.append(getattr(model, key).register_forward_hook(hook))
        images, metadata = batch[0][:8].cuda(), batch[1][:8].cuda()
        with torch.inference_mode():
            output = model(images, metadata)
            assert output['logits'].shape == (8, 7) and output['mel_logits'].shape == (8,)
            assert torch.isfinite(output['logits']).all()
        for handle in handles:
            handle.remove()
        assert all(value == 1 for value in calls.values()), calls
        # Nonzero eta is a disposable technical state, never formal initialization.
        model.readout.eta.data.fill_(.2)
        model.zero_grad(set_to_none=True)
        output = model(images[:2], metadata[:2])
        output['logits'].square().mean().backward()
        assert all(p.grad is None or not p.grad.any() for p in model.mask_head.parameters())
        slot_gradients = {name: float(p.grad.float().norm()) if p.grad is not None else None
                          for name, p in model.readout.named_parameters()}
        assert all(value is not None and value > 0 for value in slot_gradients.values())
        model.zero_grad(set_to_none=True)
        output = model(aux[0][:2].cuda(), torch.zeros(2, 22, device='cuda'))
        t.dice_bce(output['mask_logits'], aux[1][:2].cuda()).backward()
        assert gradient_norm(model.mask_head) > 0 and gradient_norm(model.backbone) > 0
        tests[mode] = {'one_call_per_batch': calls, 'readout_parameter_gradients_nonzero_eta': slot_gradients,
            'classification_mask_direct_gradient_absent': True,
            'auxiliary_mask_gradient_norm': gradient_norm(model.mask_head),
            'auxiliary_backbone_gradient_norm': gradient_norm(model.backbone)}
        del model, output, images, metadata
        gc.collect()
        torch.cuda.empty_cache()
    t.seed_all(42)
    model = t.fresh_model({**cfg, 'readout_mode': 'region'})
    model.train()
    model.record_module_gradients = True
    optimizer, scheduler = t.optimizer_for(model)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
    weights, pos = t.class_weights(train.labels)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([42, 0, 19])))
    aux_ids = rng.choice(100, 32, replace=False)
    result = t.update(model, optimizer, scaler, batch, aux, aux_ids, weights, pos, 8, .1)
    assert result is not None and result[2]
    fake_logits = torch.randn(48, 7, device='cuda')
    labels = batch[2].cuda()
    denominator = weights[labels].sum()
    full = F.cross_entropy(fake_logits, labels, weight=weights)
    micro = sum(F.cross_entropy(fake_logits[i:i+8], labels[i:i+8], weight=weights, reduction='sum') / denominator
                for i in range(0, 48, 8))
    assert abs(float(full - micro)) < 1e-6
    tests['effective_update'] = {'classification': 48, 'auxiliary': 32, 'micro': 8, 'loss': result[0],
        'gradient_norm': result[1], 'mask_gradient_live': result[2], 'CE_micro_full_error': abs(float(full-micro)),
        'diagnostics': t.summarize_diagnostics(result[3]), 'technical_state_discarded': True}
    del iterator, batch, model, optimizer, scheduler, scaler, result, fake_logits, labels, full, micro
    gc.collect()
    torch.cuda.empty_cache()
    dump(OUT / 'CUDA_TECHNICAL_CHECK_s42.json', {'status': 'PASSED', 'training_seed': 42,
         'microbatch': 8, 'elapsed_seconds': time.perf_counter()-began, 'tests': tests,
         'no_formal_training_state_reused_from_checks': True})
    return 8
