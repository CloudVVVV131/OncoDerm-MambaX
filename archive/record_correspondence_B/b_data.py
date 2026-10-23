"""Lightweight dataset module: no historical drivers imported."""
from b_common import OUT, PRIVATE, rows, sha
import sys
import torch
from PIL import Image
sys.path.insert(0,str(OUT/"model_code"))
from lesionmamba.data.transforms import build_transforms

class Images(torch.utils.data.Dataset):
    def __init__(self,cohort,cfg,indices):
        self.records=rows(PRIVATE/cohort/"images.csv")
        self.indices=indices
        self.transform=build_transforms(cfg,"eval")
        assert self.transform.transforms[0].antialias is True
        assert self.transform.transforms[0].interpolation.value=="bilinear"

    def __len__(self):
        return len(self.indices)

    def __getitem__(self,i):
        idx=int(self.indices[i])
        r=self.records[idx]
        assert sha(r["path"])==r["sha256"],"Image fingerprint mismatch"
        with Image.open(r["path"]) as im:
            x=self.transform(im.convert("RGB"))
        return x,idx

def worker_init(_):
    torch.set_num_threads(1)
