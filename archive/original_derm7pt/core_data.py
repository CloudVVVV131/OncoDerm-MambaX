"""Lightweight Derm7pt-only Windows DataLoader worker."""
import sys
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from core_common import OUT, DATA, PRIVATE, read, table, sha
sys.path.insert(0,str(OUT/'model_code'))
from lesionmamba.data.datasets import encode_metadata
from lesionmamba.data.transforms import build_transforms

class DermData(Dataset):
    def __init__(self,cfg):
        self.rows=table(PRIVATE/'COHORT_ALIGNMENT.csv')
        self.metadata=torch.from_numpy(np.load(PRIVATE/'E_AVAILABLE.npy',allow_pickle=False))
        self.transform=build_transforms(cfg,'eval')
        binding=read(OUT/'COHORT_BINDING.json')
        assert sha(PRIVATE/'COHORT_ALIGNMENT.csv')==binding['alignment_sha256']
        assert sha(PRIVATE/'E_AVAILABLE.npy')==binding['clinical_sha256']
        mapping=read(OUT/'CLINICAL_MAPPING.json')
        regenerated=[]
        for r in self.rows:
            loc=r['location'] if r['location'] in mapping['unsupported_nonempty'] else mapping['location_mapping'][r['location']]
            regenerated.append(encode_metadata(pd.Series(dict(age=np.nan,sex=r['sex'],localization=loc)),cfg['data']['metadata']))
        assert torch.equal(torch.stack(regenerated),self.metadata)
        self.null=torch.tensor(mapping['E_NULL'],dtype=torch.float32)
        original_null=encode_metadata(pd.Series(dict(age=np.nan,sex='unknown',localization='unknown')),cfg['data']['metadata'])
        assert torch.equal(original_null,self.null)
        assert torch.equal(self.metadata[:,19:],torch.tensor([1.,0.,0.]).expand(len(self),-1))
        assert self.transform.transforms[0].antialias is True
        assert self.transform.transforms[0].interpolation.value=='bilinear'

    def __len__(self): return len(self.rows)

    def __getitem__(self,i):
        row=self.rows[i]
        with Image.open(DATA/'images'/row['derm']) as image:
            x=self.transform(image.convert('RGB'))
        return x,self.metadata[i],i

def worker_init(_):
    torch.set_num_threads(1)
