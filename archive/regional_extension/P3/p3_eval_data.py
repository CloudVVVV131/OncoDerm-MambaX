"""P3-authorized deterministic evaluation dataset; lightweight spawned worker imports."""
import sys
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from p3_common import OUT, SOURCE, ROOT, REV, ARCHIVE, PRIVATE, sha, dump, read
sys.path.insert(0, str(SOURCE / 'model_code'))
from lesionmamba.data.datasets import encode_metadata, LABEL_MAP
from lesionmamba.data.transforms import build_transforms

NULL = torch.tensor([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1], dtype=torch.float32)


def normalize(v):
    if pd.isna(v):
        return 'unknown'
    return ' '.join(str(v).strip().lower().split()) or 'unknown'


class EvaluationData(Dataset):
    def __init__(self, cfg, cohort):
        assert cohort in ['HAM_val', 'HAM_test', 'SIIM']
        assert (OUT / 'P3_FROZEN_PROTOCOL.json').exists(), 'All bests and analysis must freeze before evaluation'
        self.cohort = cohort
        self.training_seed = cfg.get('training_seed', cfg['project']['seed'])
        self.transform = build_transforms(cfg, 'eval')
        PRIVATE.mkdir(exist_ok=True)
        if cohort == 'SIIM':
            source = ROOT / '_figure_source_freeze_20260819/ISIC_2020_Training_GroundTruth_v2.csv'
            mapping = read(SOURCE / 'metadata_mapping.json')
            assert sha(source) == mapping['source_hash']
            frame = pd.read_csv(source, usecols=['image_name','patient_id','target','age_approx','sex','anatom_site_general_challenge'],
                                dtype={'image_name': str, 'patient_id': str})
            frame = frame.sort_values('image_name', kind='stable').reset_index(drop=True)
            assert len(frame) == 33126 and frame.patient_id.nunique() == 2056 and frame.target.sum() == 584
            assert frame.image_name.is_unique and frame.patient_id.notna().all()
            mapped = pd.DataFrame(index=frame.index)
            age = pd.to_numeric(frame.age_approx, errors='coerce')
            mapped['age'] = age.where(np.isfinite(age), np.nan)
            mapped['sex'] = frame.sex.map(normalize)
            mapped['localization'] = frame.anatom_site_general_challenge.map(normalize).replace({'torso': 'trunk'})
            self.metadata = torch.stack([encode_metadata(r, cfg['data']['metadata']) for _, r in mapped.iterrows()])
            path = PRIVATE / 'SIIM_E_REAL.npy'
            if not path.exists():
                with path.open('xb') as f:
                    np.save(f, self.metadata.numpy(), allow_pickle=False)
            assert sha(path) == mapping['private_tensor_sha256']
            assert np.array_equal(self.metadata.numpy(), np.load(path, allow_pickle=False)), 'Clinical encoding changed across fixed models'
            assert NULL.tolist() == mapping['E_NULL']
            self.labels = frame.target.to_numpy(dtype=np.int64)
            self.ids = frame.image_name.tolist()
            self.image_dir = REV / 'data/SIIM2020/images'
            alignment = frame[['image_name', 'patient_id', 'target']]
            self.expected_count = 33126
        else:
            split = cohort.split('_')[1]
            source = ARCHIVE / f'splits/ham10000_seed42_{split}.csv'
            frame = pd.read_csv(source)
            self.metadata = torch.stack([encode_metadata(r, cfg['data']['metadata']) for _, r in frame.iterrows()])
            self.labels = frame.dx.map(LABEL_MAP).to_numpy(dtype=np.int64)
            self.ids = frame.image_id.tolist()
            self.image_dir = REV / 'data/HAM10000/images'
            alignment = frame[['image_id','lesion_id','dx']]
            self.expected_count = 1513 if split == 'val' else 1520
        assert len(self.ids) == self.expected_count and len(set(self.ids)) == len(self.ids)
        assert self.metadata.shape == (self.expected_count,22) and torch.isfinite(self.metadata).all()
        alignment_path = PRIVATE / f'{cohort}_alignment.csv'
        if alignment_path.exists():
            pd.testing.assert_frame_equal(pd.read_csv(alignment_path), alignment, check_dtype=False)
        else:
            alignment.to_csv(alignment_path, index=False)
        identity = {'cohort': cohort, 'images': len(self.ids), 'source': str(source),
            'source_sha256': sha(source), 'alignment_sha256': sha(alignment_path),
            'transform': 'RGB, Resize(short=256,bilinear), CenterCrop224, ToTensor, ImageNet mean/std',
            'clinical_dimension': 22, 'model_inputs_exclude_ids_patient_target_diagnosis': True}
        if cohort == 'SIIM':
            identity.update({'patients':2056, 'positive_images':584,
                'E_REAL_sha256': sha(PRIVATE / 'SIIM_E_REAL.npy'), 'E_NULL': NULL.tolist()})
        dump(OUT / f'{cohort}_INPUT_IDENTITY.json', identity)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        with Image.open(self.image_dir / f'{self.ids[index]}.jpg') as im:
            x = self.transform(im.convert('RGB'))
        return x, self.metadata[index], int(self.labels[index]), index


def worker_init(_):
    torch.set_num_threads(1)
