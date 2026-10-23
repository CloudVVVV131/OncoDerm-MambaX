from pathlib import Path
import os
import sys
import json
import random
import hashlib

OUT = Path(__file__).resolve().parent
os.environ['HF_HOME'] = str(OUT / 'cache/huggingface')
os.environ['TORCH_HOME'] = str(OUT / 'cache/torch')
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '30'
os.environ['HF_HUB_ETAG_TIMEOUT'] = '15'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
PKG = OUT / 'start_package/FIM_Focused_Training_Start'
sys.path.insert(0, str(PKG / 'Reference_Only/model_code'))

import numpy as np
import torch
import timm
import yaml
from huggingface_hub import hf_hub_download
from timm.models import get_pretrained_cfg
from lesionmamba.models.oncoderm_mambax import OncoDermMambaX
import lesionmamba.models.oncoderm_mambax as model_module


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert json.loads((OUT / 'DATA_CHECK.json').read_text())['status'] == 'READY'
    assert torch.cuda.is_available()
    cfg = yaml.safe_load((PKG / 'Reference_Only/original_configs/B6_seed42.yaml').read_text())
    pcfg = get_pretrained_cfg(cfg['model']['timm_name']).to_dict()
    source = {'pretrained_config': pcfg, 'timm_version': timm.__version__, 'torch_version': torch.__version__,
              'model_source': model_module.__file__, 'status': 'ACQUIRING_PUBLIC_INITIALIZATION'}
    (OUT / 'INITIALIZATION.json').write_text(json.dumps(source, indent=2), encoding='utf-8')
    print('Public source:', pcfg['hf_hub_id'], flush=True)
    path = hf_hub_download(repo_id=pcfg['hf_hub_id'], filename='model.safetensors',
                           cache_dir=OUT / 'cache/huggingface/hub', token=False)
    source.update(public_weight_path=str(path), public_weight_sha256=sha(path), public_weight_bytes=Path(path).stat().st_size,
                  download_source=f"https://huggingface.co/{pcfg['hf_hub_id']}/resolve/main/model.safetensors",
                  snapshot_revision=Path(path).parent.name)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    cfg['model']['pretrained'] = False
    model = OncoDermMambaX(cfg)
    backbone = timm.create_model(cfg['model']['timm_name'], pretrained=True, features_only=True,
                                 out_indices=tuple(cfg['model']['out_indices']),
                                 pretrained_cfg_overlay={'file': path, 'hf_hub_id': None})
    model.backbone.load_state_dict(backbone.state_dict(), strict=True)
    del backbone
    assert all(p.requires_grad for p in model.parameters())
    target = OUT / 'initial_state_s42.pt'
    assert not target.exists()
    torch.save({'model': model.state_dict(), 'seed': 42, 'config': cfg, 'source': source}, target)
    source.update(status='READY', initial_state_path=str(target), initial_state_sha256=sha(target),
                  parameter_count=sum(p.numel() for p in model.parameters()), seed=42)
    (OUT / 'INITIALIZATION.json').write_text(json.dumps(source, indent=2), encoding='utf-8')
    print(json.dumps(source, indent=2), flush=True)


if __name__ == '__main__':
    main()
