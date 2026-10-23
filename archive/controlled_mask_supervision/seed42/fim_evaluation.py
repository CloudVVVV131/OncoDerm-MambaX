"""FP32 inference for the two newly trained seed42 models, after both bests lock.
Frozen mapping: B2 prepare_b2.py 94-124 and metadata_mapping.json.
No fitting, model selection, historical inference, or training is performed here.
"""
from pathlib import Path
import json
import hashlib
import time
import gc
import sys
import os
from datetime import datetime

OUT = Path(__file__).resolve().parent
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
from fim_training import (PKG, ROOT, REV, DEADLINE, config, dump, log, loader,
                          HAMDataset, OncoDermMambaX, encode_metadata, build_transforms)
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

PRIVATE = OUT / 'private_predictions'
NULL = torch.tensor([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1], dtype=torch.float32)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def normalize(v):
    if pd.isna(v):
        return 'unknown'
    return ' '.join(str(v).strip().lower().split()) or 'unknown'


class SIIMDataset(Dataset):
    def __init__(self, cfg):
        source = ROOT / '_figure_source_freeze_20260819/ISIC_2020_Training_GroundTruth_v2.csv'
        assert sha(source) == 'a1c7c97c59e5f49232dfe3b6f6817efaaee5dbf9bcb4ce584c1889e641be5c4e'
        self.frame = pd.read_csv(source, usecols=['image_name','patient_id','target','age_approx','sex','anatom_site_general_challenge'], dtype={'image_name': str, 'patient_id': str})
        self.frame = self.frame.sort_values('image_name', kind='stable').reset_index(drop=True)
        assert len(self.frame) == 33126 and self.frame.patient_id.nunique() == 2056
        assert self.frame.image_name.is_unique and self.frame.patient_id.notna().all() and self.frame.target.sum() == 584
        mapped = pd.DataFrame(index=self.frame.index)
        age = pd.to_numeric(self.frame.age_approx, errors='coerce')
        mapped['age'] = age.where(np.isfinite(age), np.nan)
        mapped['sex'] = self.frame.sex.map(normalize)
        mapped['localization'] = self.frame.anatom_site_general_challenge.map(normalize).replace({'torso': 'trunk'})
        self.metadata = torch.stack([encode_metadata(r, cfg['data']['metadata']) for _, r in mapped.iterrows()])
        assert self.metadata.shape == (33126, 22) and torch.isfinite(self.metadata).all()
        tensor_path = PRIVATE / 'SIIM_E_REAL.npy'
        if not tensor_path.exists():
            with tensor_path.open('xb') as f:
                np.save(f, self.metadata.numpy(), allow_pickle=False)
        mapping = json.loads((PKG / 'Reference_Only/metadata_mapping.json').read_text())
        assert sha(tensor_path) == mapping['private_tensor_sha256'], 'Clinical tensor differs from frozen B2'
        self.transform = build_transforms(cfg, 'eval')
        self.labels = self.frame.target.to_numpy(dtype=np.int64)
        self.frame[['image_name', 'patient_id', 'target']].to_csv(PRIVATE / 'SIIM_alignment.csv', index=False)
        dump(OUT / 'EVALUATION_METADATA_CHECK.json', {'status': 'PASSED', 'source_path': str(source),
             'source_sha256': sha(source), 'clinical_tensor_sha256': sha(tensor_path),
             'matches_frozen_B2_tensor': True, 'images': 33126, 'patients': 2056, 'positive_images': 584,
             'E_NULL': NULL.tolist(), 'labels_and_patient_ids_not_model_inputs': True})

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        r = self.frame.iloc[i]
        with Image.open(REV / 'data/SIIM2020/images' / f'{r.image_name}.jpg') as im:
            x = self.transform(im.convert('RGB'))
        return x, self.metadata[i], int(self.labels[i]), i


def stopped():
    return time.time() >= DEADLINE or (OUT / 'STOP_REQUESTED').exists()


@torch.inference_mode()
def infer(model, dataset, condition, cohort, micro):
    modes = ['E_REAL', 'E_NULL'] if cohort == 'SIIM' else ['NATIVE']
    stem = f'{condition}_{cohort}'
    state_path = PRIVATE / f'{stem}_status.json'
    logits_path = PRIVATE / f'{stem}_logits.npy'
    score_path = PRIVATE / f'{stem}_probabilities.npy'
    start = 0
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['status'] == 'COMPLETE':
            return True
        start = state['next_index']
    shape = (len(dataset), len(modes), 8)
    if logits_path.exists():
        logits = np.lib.format.open_memmap(logits_path, mode='r+')
        probs = np.lib.format.open_memmap(score_path, mode='r+')
        assert logits.shape == probs.shape == shape
    else:
        logits = np.lib.format.open_memmap(logits_path, mode='w+', dtype='float32', shape=shape)
        probs = np.lib.format.open_memmap(score_path, mode='w+', dtype='float32', shape=shape)
    begin = time.time()
    for images, metadata, _, index in loader(dataset, list(range(start, len(dataset))), batch_size=micro):
        x = images.cuda(non_blocking=True)
        for c, mode in enumerate(modes):
            meta = NULL.cuda().expand(len(x), -1) if mode == 'E_NULL' else metadata.cuda(non_blocking=True)
            out = model(x, meta)
            values = torch.cat([out['logits'], out['mel_logits'][:, None]], 1)
            scores = torch.cat([out['logits'].softmax(1), out['mel_logits'].sigmoid()[:, None]], 1)
            assert torch.isfinite(values).all() and torch.isfinite(scores).all()
            logits[index.numpy(), c] = values.cpu().numpy()
            probs[index.numpy(), c] = scores.cpu().numpy()
        last = int(index[-1]) + 1
        if last % 800 == 0 or last == len(dataset) or stopped():
            logits.flush()
            probs.flush()
            dump(state_path, {'status': 'COMPLETE' if last == len(dataset) else 'PARTIAL',
                 'next_index': last, 'total': len(dataset), 'modes': modes, 'condition': condition,
                 'cohort': cohort, 'elapsed_current_invocation_seconds': time.time() - begin,
                 'updated_at': datetime.now().astimezone().isoformat()})
            log(f'EVALUATION {stem} {last}/{len(dataset)}')
        if stopped() and last < len(dataset):
            return False
    dump(PRIVATE / f'{stem}_hashes.json', {'logits_sha256': sha(logits_path), 'probabilities_sha256': sha(score_path)})
    return True


def main():
    PRIVATE.mkdir(exist_ok=True)
    for condition in ['FULL', 'ZERO']:
        state = json.loads((OUT / f'FIM_LPAH_{condition}_s42/status.json').read_text())
        assert state['status'] == 'TRAINING_COMPLETE', 'Both bests must be locked before evaluation'
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cfg = config()
    datasets = {'HAM_val': HAMDataset(cfg, 'val'), 'HAM_test': HAMDataset(cfg, 'test'), 'SIIM': SIIMDataset(cfg)}
    micro = json.loads((OUT / 'SMOKE_TEST.json').read_text())['microbatch']
    for cohort in ['HAM_val', 'HAM_test']:
        ds = datasets[cohort]
        ds.frame[['image_id', 'lesion_id', 'dx']].to_csv(PRIVATE / f'{cohort}_alignment.csv', index=False)
    for condition in ['FULL', 'ZERO']:
        path = OUT / f'FIM_LPAH_{condition}_s42/best.pt'
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            ckpt = torch.load(path, weights_only=True, map_location='cpu')
        model = OncoDermMambaX(cfg)
        model.load_state_dict(ckpt['model'], strict=True)
        model.requires_grad_(False).eval().cuda()
        assert all(not m.training for m in model.modules())
        dump(OUT / f'{condition}_EVALUATION_MODEL.json', {'path': str(path), 'sha256': sha(path),
             'best_epoch': ckpt['epoch'], 'strict_loading': True, 'dtype': 'FP32', 'TF32': False,
             'model_source': str(PKG / 'Reference_Only/model_code/lesionmamba/models/oncoderm_mambax.py')})
        del ckpt
        for cohort, ds in datasets.items():
            if stopped() or not infer(model, ds, condition, cohort, micro):
                dump(OUT / 'EVALUATION_STATUS.json', {'status': 'PARTIAL', 'current': f'{condition}_{cohort}'})
                return
        del model
        gc.collect()
        torch.cuda.empty_cache()
    dump(OUT / 'EVALUATION_STATUS.json', {'status': 'COMPLETE', 'models': 2, 'cohorts': ['HAM_val','HAM_test','SIIM'],
         'precision': 'FP32', 'external_conditions': ['E_REAL','E_NULL'], 'scores': ['softmax_MEL','endpoint_MEL']})


if __name__ == '__main__':
    main()
