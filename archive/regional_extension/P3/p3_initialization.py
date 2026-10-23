"""Explicit same-name/shape transfer from the corresponding FIM pre-training state."""
import hashlib
import json
from pathlib import Path
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def build_initial(cfg):
    import p3_training as t
    seed = cfg['training_seed']
    root = t.OLD if seed == 42 else t.THREE
    record = root / ('INITIALIZATION.json' if seed == 42 else f'INITIALIZATION_s{seed}.json')
    identity = json.loads(record.read_text(encoding='utf-8-sig'))
    path = root / f'initial_state_s{seed}.pt'
    assert sha(path) == identity['initial_state_sha256']
    assert identity['seed'] == seed and identity['status'] == 'READY'
    assert identity['snapshot_revision'] == 'aa096f03029c7f0ec052013f64c819b34f8ad790'
    assert identity['public_weight_sha256'] == 'a1aefa409b513cf209b085424eb3efffe4e4a9f511491bc2c12ea35209e6bb95'
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        initial = torch.load(path, map_location='cpu', weights_only=True)
    assert initial['seed'] == seed
    t.seed_all(seed)
    model = t.OncoDermMambaX(cfg)
    target, source = model.state_dict(), initial['model']
    # All shared keys are declared by original module identity; only fuse input differs.
    prefixes = ('backbone.', 'prev_proj.', 'fuse.', 'mask_head.', 'attention_from_feature.',
                'metadata_encoder.', 'metadata_gate.', 'head.', 'mel_head.')
    assert all(k.startswith(prefixes) for k in target)
    copied, retained = [], []
    for key in target:
        assert key in source, key
        if target[key].shape == source[key].shape:
            target[key] = source[key]
            copied.append(key)
        else:
            assert key == 'fuse.net.0.weight', key
            retained.append({'key': key, 'B2S_shape': list(target[key].shape),
                             'FULL_shape': list(source[key].shape),
                             'policy': 'retain freshly seed-initialized B2S tensor, no slicing'})
    assert len(retained) == 1
    model.load_state_dict(target, strict=True)
    for k in copied:
        assert torch.equal(model.state_dict()[k], source[k])
    absent = sorted(set(source) - set(target))
    assert all(k.startswith(('cssa_last.', 'cssa_prev.')) for k in absent)
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    result = {'training_seed': seed, 'original_initial_state': str(path),
        'original_sha256': identity['initial_state_sha256'], 'original_initialization_record': str(record),
        'original_record_sha256': sha(record), 'base_revision': identity['snapshot_revision'],
        'base_sha256': identity['public_weight_sha256'], 'copied_shared_keys': copied,
        'shape_mismatch_retained_fresh': retained, 'FULL_only_absent_CSSA_keys': absent,
        'strict_complete_state_load': True, 'B2S_initial_tensor_sha256': h.hexdigest(),
        'unused_mask_head_and_prev_proj_retained': True, 'initialization_seed': seed,
        'training_rng_reset_to_seed_after_initialization': True}
    output = t.OUT / f'B2S_INITIALIZATION_s{seed}.json'
    if output.exists():
        assert json.loads(output.read_text()) == result
    else:
        t.dump(output, result)
    del initial, source, target
    t.seed_all(seed)
    return model
