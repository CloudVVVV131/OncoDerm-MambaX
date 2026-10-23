"""Resume frozen P2; isolate worker imports, preserving dataset and training behavior."""
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import ast
import json
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    import gc
    import hashlib
    import numpy as np
    import torch
    import p2_training as t
    import p2_worker_data as w
    from p2_prepare import verify_freeze
    verify_freeze()
    root = Path(__file__).resolve().parent
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    before = t.rng_state()
    original = ast.parse((root / 'p2_training.py').read_text(encoding='utf-8'))
    isolated = ast.parse((root / 'p2_worker_data.py').read_text(encoding='utf-8'))
    for name in ['HAMDataset', 'worker_init']:
        a = next(n for n in original.body if getattr(n, 'name', '') == name)
        b = next(n for n in isolated.body if getattr(n, 'name', '') == name)
        assert ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False), name
    assert w.ARCHIVE == t.ARCHIVE and w.REV == t.REV
    assert w.encode_metadata is t.encode_metadata and w.build_transforms is t.build_transforms
    original_class, original_worker = t.HAMDataset, t.worker_init
    checks = []
    for split in ['train', 'val']:
        old = original_class(t.config(3407), split)
        new = w.HAMDataset(t.config(3407), split)
        assert old.frame.equals(new.frame) and torch.equal(old.metadata, new.metadata)
        assert np.array_equal(old.labels, new.labels)
        keys = [(8, i) for i in [0, 1, 7, 47, 88, 100, 501, 1000]] if split == 'train' else list(range(8))
        expected = [old[key] for key in keys]
        for key, reference in zip(keys, expected):
            actual = new[key]
            assert all(torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b
                       for a, b in zip(actual, reference))
        t.worker_init = w.worker_init
        # The same loader still uses two workers and the same sampler/generator settings.
        batches = list(t.loader(new, keys, batch_size=8))
        actual = batches[0]
        assert len(batches) == 1
        assert torch.equal(actual[0], torch.stack([r[0] for r in expected]))
        assert torch.equal(actual[1], torch.stack([r[1] for r in expected]))
        assert actual[2].tolist() == [r[2] for r in expected]
        assert actual[3].tolist() == [r[3] for r in expected]
        checks.append({'split': split, 'eight_images_exact_tensor_equality': True,
                       'workers': 2, 'metadata_and_labels_exact': True})
        del old, new, expected, actual, batches
        gc.collect()
    t.restore_rng(before)
    check = {'status': 'PASSED', 'at': datetime.now().astimezone().isoformat(),
        'dataset_and_worker_init_AST_identical': True, 'checks': checks,
        'scientific_code_and_original_files_unchanged': True,
        'change': 'Dataset class and worker_init import from a lightweight module; spawned main imports stdlib only.',
        'original_dataset_source': str(root / 'p2_training.py'),
        'worker_dataset_source': str(root / 'p2_worker_data.py'),
        'inference_or_training_in_check': False, 'rng_restored_after_check': True}
    t.dump(root / f'WORKER_ISOLATION_CHECK_{stamp}.json', check)
    t.log('WORKER_ISOLATION_PASSED: exact dataset/worker AST; identical direct and two-worker tensors')
    if args.check_only:
        t.worker_init = original_worker
        print(json.dumps(check))
        return
    t.HAMDataset, t.worker_init = w.HAMDataset, w.worker_init
    started = datetime.now().astimezone()
    deadline = started + timedelta(hours=12)
    checkpoint = root / 'R_IMG_s3407/last.pt'
    h = hashlib.sha256()
    with checkpoint.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    authorization = {'started_at': started.isoformat(), 'deadline': deadline.isoformat(),
        'user_instruction': 'Continue until all authorized P2 runs complete',
        'original_deadline': t.WINDOW['deadline'], 'original_WINDOW_preserved': True,
        'scope': 'Only incomplete seed3407 R_IMG/R_CLIN/XATTN; seed2024 and seed42 unchanged',
        'checkpoint': str(checkpoint), 'checkpoint_sha256': h.hexdigest(),
        'checkpoint_bytes': checkpoint.stat().st_size,
        'entrypoint': str(Path(__file__).resolve()),
        'dataset_worker_module': str(root / 'p2_worker_data.py'),
        'model_optimizer_scheduler_loss_loader_parameters_unchanged': True}
    t.dump(root / f'CONTINUATION_AUTHORIZATION_{stamp}.json', authorization)
    t.DEADLINE = deadline.timestamp()
    failed = False
    try:
        t.main()
    except Exception:
        failed = True
        error = traceback.format_exc()
        t.log(error)
        t.dump(root / f'TRAINING_ERROR_{datetime.now():%Y%m%d_%H%M%S}.json',
               {'error': error, 'at': datetime.now().astimezone().isoformat(),
                'attempt_started_at': started.isoformat()})
        t.progress('BLOCKED')
    t.dump(root / f'ATTEMPT_RESULT_{datetime.now():%Y%m%d_%H%M%S}.json',
        {'started_at': started.isoformat(), 'finished_at': datetime.now().astimezone().isoformat(),
         'exception_raised': failed, 'continuation_authorization': str(root / f'CONTINUATION_AUTHORIZATION_{stamp}.json'),
         'delivery_pending': True, 'prior_delivery_overwritten': False})
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
