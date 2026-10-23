"""Identical HAM dataset methods, without model/metric imports in spawned workers."""
from pathlib import Path
import sys
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

OUT = Path(__file__).resolve().parent
PKG = OUT / 'source'
ROOT = Path(__import__('os').environ['R52_PATH_EA6F443AA882'])
ARCHIVE = ROOT / 'oncoderm_bioengineering_full_results_20260817_110459/project'
REV = ROOT / 'project/revision'
sys.dont_write_bytecode = True
sys.path.insert(0, str(PKG / 'model_code'))
from lesionmamba.data.datasets import encode_metadata, LABEL_MAP
from lesionmamba.data.transforms import build_transforms


class HAMDataset(Dataset):
    def __init__(self, cfg, split):
        assert split in ['train', 'val'], 'Training loader forbids held-out data'
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
