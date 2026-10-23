"""HAM validation only; original clinical encoding and image preprocessing."""
import sys
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from p5_common import *
sys.path.insert(0,str(SOURCE/'model_code'))
from lesionmamba.data.datasets import encode_metadata,LABEL_MAP
from lesionmamba.data.transforms import build_transforms

class ValidationData(Dataset):
    def __init__(self,cfg):
        protocol=read(OUT/'P5_FROZEN_PROTOCOL.json')['task_B']
        assert protocol['cohort']=='HAM_val' and protocol['images']==1513
        item=protocol['ham_validation_csv']; assert sha(item['path'])==item['sha256']
        frame=pd.read_csv(item['path']); assert len(frame)==1513 and frame.image_id.is_unique
        old=P3/'private_predictions/HAM_val_alignment.csv'
        assert sha(old)==read(P3/'HAM_val_INPUT_IDENTITY.json')['alignment_sha256']
        pd.testing.assert_frame_equal(frame[['image_id','lesion_id','dx']],pd.read_csv(old),check_dtype=False)
        self.ids=frame.image_id.tolist(); self.labels=frame.dx.map(LABEL_MAP).to_numpy(dtype=np.int64)
        self.metadata=torch.stack([encode_metadata(r,cfg['data']['metadata']) for _,r in frame.iterrows()])
        assert self.metadata.shape==(1513,22) and torch.isfinite(self.metadata).all()
        self.training_seed=cfg['training_seed']; self.image_root=Path(protocol['image_root'])
        self.transform=build_transforms(cfg,'eval')
        assert cfg['data']['input_size']==224 and cfg['data']['resize_short_side']==256
        assert all((self.image_root/f'{i}.jpg').is_file() for i in self.ids)
        p=PRIVATE/'validation_labels.npy'
        if p.exists(): assert np.array_equal(np.load(p,allow_pickle=False),self.labels)
        else:np.save(p,self.labels,allow_pickle=False)
        identity={'source':item,'images':1513,'image_root':str(self.image_root),
            'ordered_image_ids_sha256':hashlib.sha256('\n'.join(self.ids).encode()).hexdigest(),
            'clinical_tensor_sha256':hashlib.sha256(self.metadata.numpy().tobytes()).hexdigest(),
            'labels_sha256':sha(p),'transform':repr(self.transform),'clinical_encoding':'unchanged P3 encoder',
            'inference_allowed_only_on':'HAM_val','labels_and_IDs_not_model_inputs':True}
        dest=OUT/'HAM_VALIDATION_INPUT_IDENTITY.json'
        if dest.exists():assert read(dest)==identity
        else:dump(dest,identity)

    def __len__(self):return len(self.ids)
    def __getitem__(self,i):
        with Image.open(self.image_root/f'{self.ids[i]}.jpg') as im:x=self.transform(im.convert('RGB'))
        return x,self.metadata[i],self.ids[i],i

def worker_init(_):torch.set_num_threads(1)
