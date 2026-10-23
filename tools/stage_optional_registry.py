"""Copy one preserved Python package to a new workspace and apply an explicit overlay.

No project modules are imported. No models, data, tensors or experiments are run.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    dest=a.destination.resolve()
    if not a.destination.is_absolute() or dest==root or root in dest.parents or dest in root.parents:
        p.error('Use an absolute new workspace outside the release tree.')
    if dest.exists(): p.error('Destination exists; refusing overwrite.')
    source=root/'archive/original_framework/august_snapshot/lesionmamba'
    overlay=root/'adapters/august_registry/registry.py'
    identity=json.loads((root/'adapters/august_registry/OVERLAY_IDENTITY.json').read_text(encoding='utf-8'))
    digest=lambda b:hashlib.sha256(b).hexdigest()
    if digest((root/identity['source']).read_bytes())!=identity['source_sha256'] or digest(overlay.read_bytes())!=identity['overlay_sha256']:
        p.error('Registry identity mismatch.')
    files=list(source.rglob('*'))
    if any(f.is_symlink() for f in files): p.error('Symlinks are not allowed in source.')
    if any(f.is_file() and f.suffix!='.py' for f in files): p.error('Only this source-only Python package is permitted.')
    record={'mode':'dry-run' if a.dry_run else 'source-workspace-only','destination':str(dest),
            'original_registry_sha256':identity['source_sha256'],'overlay_sha256':identity['overlay_sha256'],
            'excluded_comparator':'DermaMamba','research_executed':False,'models_loaded':0}
    if not a.dry_run:
        # copytree refuses an existing target, including races after the check.
        dest.mkdir(parents=True,exist_ok=False)
        shutil.copytree(source,dest/'lesionmamba')
        shutil.copyfile(overlay,dest/'lesionmamba/models/registry.py')
        (dest/'RELEASE_OVERLAY_RECEIPT.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,indent=2))


if __name__=='__main__': main()
