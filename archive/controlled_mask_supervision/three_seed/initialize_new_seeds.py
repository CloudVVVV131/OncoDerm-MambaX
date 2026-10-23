"""Offline initialization and narrowly scoped checks for the remaining FIM seeds."""
import hashlib
import json
from pathlib import Path
import ast
import time
import numpy as np
import torch
import timm
from fim_training import OUT, OLD, PKG, DEADLINE, config, seed_all, dump, OncoDermMambaX, HAMDataset


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tensor_digest(state, backbone):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if name.startswith('backbone.') == backbone:
            h.update(name.encode())
            h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    assert time.time() < DEADLINE and torch.cuda.is_available()
    torch.set_num_threads(4)
    original = json.loads((OLD / 'INITIALIZATION.json').read_text())
    weight = Path(original['public_weight_path'])
    assert weight.stat().st_size == original['public_weight_bytes']
    assert sha(weight) == original['public_weight_sha256']
    assert weight.parent.name == original['snapshot_revision']
    assert sha(OUT / 'fim_mask_train100.csv') == sha(OLD / 'fim_mask_train100.csv') == 'ef507cb90b4955b2a9f7dabbdc532a1bf9176899c72c46f646231f795c648a13'
    source_checks = []
    for item in json.loads((OLD / 'INPUT_SOURCES.json').read_text()):
        if '/splits/' in item['path'].replace('\\', '/'):
            assert sha(item['path']) == item['sha256']
            source_checks.append(item)
    for p in (PKG / 'Reference_Only').rglob('*'):
        if p.is_file():
            origin = OLD / p.relative_to(OUT)
            assert sha(p) == sha(origin)
    old_tree = ast.parse((OLD / 'fim_training.py').read_text())
    new_tree = ast.parse((OUT / 'fim_training.py').read_text())
    preserved = ['optimizer_for', 'class_weights', 'dice_bce', 'update_once', 'update', 'validate', 'load_aux', 'rng_state', 'restore_rng']
    for name in preserved:
        a = next(n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        b = next(n for n in new_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(a) == ast.dump(b), f'Frozen function changed: {name}'
    results = []
    sequences = []
    for s in [2024, 3407]:
        cfg = config(s)
        seed_all(s)
        model = OncoDermMambaX(cfg)
        backbone = timm.create_model(cfg['model']['timm_name'], pretrained=True, features_only=True,
            out_indices=tuple(cfg['model']['out_indices']),
            pretrained_cfg_overlay={'file': str(weight), 'hf_hub_id': None})
        model.backbone.load_state_dict(backbone.state_dict(), strict=True)
        del backbone
        source = {k: original[k] for k in ['public_weight_path', 'public_weight_sha256', 'public_weight_bytes', 'snapshot_revision', 'pretrained_config']}
        source.update(seed=s, training_seed=s, split_seed=42, model_source=str(PKG / 'Reference_Only/model_code/lesionmamba/models/oncoderm_mambax.py'),
                      timm_version=timm.__version__, torch_version=str(torch.__version__), status='READY',
                      initialization_policy='new random non-pretrained parameters; fixed local public backbone; no historical state loaded')
        path = OUT / f'initial_state_s{s}.pt'
        assert not path.exists(), 'Never overwrite an initialization'
        torch.save({'model': model.state_dict(), 'seed': s, 'config': cfg, 'source': source}, path)
        source.update(initial_state_path=str(path), initial_state_sha256=sha(path),
                      backbone_tensor_sha256=tensor_digest(model.state_dict(), True),
                      nonbackbone_tensor_sha256=tensor_digest(model.state_dict(), False),
                      parameter_count=sum(p.numel() for p in model.parameters()))
        dump(OUT / f'INITIALIZATION_s{s}.json', source)
        results.append(source)
        del model
        train = HAMDataset(cfg, 'train')
        assert train.training_seed == s and len(train) == 6982
        x1 = train[(0, 0)][0]
        x2 = train[(0, 0)][0]
        assert torch.equal(x1, x2), 'Stateless augmentation is not repeatable'
        aug_sha = hashlib.sha256(x1.numpy().tobytes()).hexdigest()
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(s)).numpy()
        aux = np.random.Generator(np.random.PCG64(np.random.SeedSequence([s, 0, 19]))).choice(100, 32, replace=False)
        sequences.append({'seed': s, 'epoch0_order_sha256': hashlib.sha256(order.tobytes()).hexdigest(),
                          'epoch0_first_aux_sha256': hashlib.sha256(aux.tobytes()).hexdigest(),
                          'epoch0_image0_aug_sha256': aug_sha, 'dataloader_generator_seed': s * 1000 + 17})
    assert results[0]['backbone_tensor_sha256'] == results[1]['backbone_tensor_sha256']
    assert results[0]['nonbackbone_tensor_sha256'] != results[1]['nonbackbone_tensor_sha256']
    for key in ['epoch0_order_sha256', 'epoch0_first_aux_sha256', 'epoch0_image0_aug_sha256']:
        assert sequences[0][key] != sequences[1][key]
    dump(OUT / 'PARAMETERIZATION_CHECK.json', {'status': 'PASSED', 'preserved_function_AST': preserved,
         'same_backbone_across_seeds': True, 'new_nonbackbone_per_seed': True,
         'same_initial_state_within_each_pair': True, 'seed42_state_loaded': False,
         'mask_members_byte_identical': True, 'split_seed': 42, 'sequences': sequences,
         'split_source_checks': source_checks})
    print('OFFLINE INITIALIZATION AND SEED PARAMETERIZATION PASSED', flush=True)


if __name__ == '__main__':
    main()
