"""Frozen Derm7pt-only input adapter; lightweight Windows worker imports."""
import csv
import sys
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from p4_common import OUT, SOURCE, DATA, PRIVATE, read, sha, dump
sys.path.insert(0, str(SOURCE / 'model_code'))
from lesionmamba.data.datasets import encode_metadata
from lesionmamba.data.transforms import build_transforms

NULL = torch.tensor([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1], dtype=torch.float32)

class EvaluationData(Dataset):
    def __init__(self, cfg):
        protocol = read(OUT/'P4_FROZEN_PROTOCOL.json')
        binding = protocol['cohort']
        self.training_seed = cfg['training_seed']
        self.transform = build_transforms(cfg, 'eval')
        with (PRIVATE/'COHORT_ALIGNMENT.csv').open(encoding='utf-8', newline='') as f:
            self.rows = list(csv.DictReader(f))
        self.metadata = torch.from_numpy(np.load(PRIVATE/'E_AVAILABLE.npy', allow_pickle=False))
        self.labels = np.load(PRIVATE/'labels.npy', allow_pickle=False)
        assert len(self.rows) == binding['included_rows']
        assert self.metadata.shape == (len(self),22) and torch.isfinite(self.metadata).all()
        assert sha(PRIVATE/'COHORT_ALIGNMENT.csv') == binding['alignment_sha256']
        assert sha(PRIVATE/'E_AVAILABLE.npy') == binding['clinical_tensor_sha256']
        assert sha(PRIVATE/'labels.npy') == binding['labels_sha256']
        mapping = protocol['clinical_mapping']
        assert NULL.tolist() == mapping['E_NULL']
        # Preserve nonempty unsupported source sites through the original encoder:
        # its category fallback is unknown, while its missing indicator remains zero.
        regenerated=[]
        for row in self.rows:
            loc = row['location'] if row['location'] in mapping['unsupported_nonempty'] else mapping['location_mapping'][row['location']]
            regenerated.append(encode_metadata(pd.Series({'age':np.nan,'sex':row['sex'],'localization':loc}), cfg['data']['metadata']))
        assert torch.equal(torch.stack(regenerated), self.metadata)
        missing = encode_metadata(pd.Series({'age':np.nan,'sex':'unknown','localization':'unknown'}),cfg['data']['metadata'])
        assert torch.equal(missing,NULL)
        assert torch.equal(self.metadata[:,19:],torch.tensor([1.,0.,0.]).expand(len(self),-1))
        dump(OUT/'CLINICAL_ENCODER_CHECK.json', {'status':'PASSED','rows':len(self),
            'matches_original_encoder':True,'unsupported_nonempty_site_missing':0,
            'all_age_missing':True,'E_NULL_matches_original_encoder':True,
            'transform':repr(self.transform),'dataset_source':str(__file__),
            'no_diagnosis_or_identifiers_passed_to_model':True})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        with Image.open(DATA/'images'/self.rows[index]['derm']) as im:
            image=self.transform(im.convert('RGB'))
        return image,self.metadata[index],index

def worker_init(_):
    torch.set_num_threads(1)
