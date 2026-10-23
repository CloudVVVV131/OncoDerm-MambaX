from pathlib import Path
import csv
import hashlib
import json
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image

OUT = Path(__file__).resolve().parent
ROOT = Path(__import__('os').environ['R52_PATH_EA6F443AA882'])
REV = ROOT / 'project/revision'
ARCHIVE = ROOT / 'oncoderm_bioengineering_full_results_20260817_110459/project'
PKG = OUT / 'start_package/FIM_Focused_Training_Start'
SOURCES = []


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path, **kwargs):
    path = Path(path)
    SOURCES.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)})
    return pd.read_csv(path, **kwargs)


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    assert shutil.disk_usage(OUT).free >= 12 * 1024**3
    model = PKG / 'Reference_Only/model_code/lesionmamba/models/oncoderm_mambax.py'
    assert sha(model) == '7e737a004f69053821628f84b3da9c44a4e4a208a386fa7322c5c7ff9331a0c5'
    splits = {k: read(ARCHIVE / f'splits/ham10000_seed42_{k}.csv') for k in ['train', 'val', 'test']}
    assert {k: len(v) for k, v in splits.items()} == {'train': 6982, 'val': 1513, 'test': 1520}
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        assert not set(splits[a].image_id) & set(splits[b].image_id)
        assert not set(splits[a].lesion_id.dropna()) & set(splits[b].lesion_id.dropna())
    ham = read(REV / 'manifests/HAM10000_recovered_images.csv')
    siim = read(REV / 'manifests/SIIM2020_recovered_images.csv')
    candidates = read(REV / 'manifests/localization200_restored.csv')
    relations = read(REV / 'manifests/data_manifest.csv', usecols=['image_id', 'lesion_id', 'patient_id'])
    near = read(REV / 'reports/round2/localization_near_duplicate_refined.csv')
    assert not candidates.image_id.duplicated().any()
    assert set(ham.image_id) == set(pd.concat(splits.values()).image_id)
    assert len(ham) == 10015 and len(siim) == 33126
    missing_images = [str(p) for p in pd.concat([ham.path, siim.path]) if not Path(p).is_file()]
    if missing_images:
        dump('missing_classification_images.json', missing_images)
        raise RuntimeError(f'{len(missing_images)} missing classification images')
    protected_ids = set(splits['val'].image_id) | set(splits['test'].image_id) | set(siim.image_id)
    protected = pd.concat([ham[ham.image_id.isin(protected_ids)], siim])
    protected_files = set(protected.sha256)
    protected_pixels = set(protected.pixel_sha256)
    protected_rel = relations[relations.image_id.isin(protected_ids)]
    protected_lesions = set(protected_rel.lesion_id.dropna())
    protected_patients = set(protected_rel.patient_id.dropna())
    rows = []
    for row in candidates.to_dict('records'):
        image_path, mask_path = Path(row['image_path']), Path(row['mask_path'])
        errors = []
        if not image_path.is_file() or not mask_path.is_file():
            errors.append('missing_image_or_mask')
        else:
            image_sha, mask_sha = sha(image_path), sha(mask_path)
            with Image.open(image_path) as im:
                rgb = im.convert('RGB')
                dims = im.size
                pixel_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
            with Image.open(mask_path) as im:
                mask_dims = im.size
                mask_array = np.asarray(im.convert('L'))
                foreground = int((mask_array > 127).sum())
            if dims != mask_dims:
                errors.append('image_mask_size_mismatch')
            if foreground == 0:
                errors.append('empty_mask')
            if (image_sha, mask_sha, pixel_sha) != (row['image_sha256'], row['mask_sha256'], row['pixel_sha256']):
                errors.append('manifest_hash_mismatch')
            row.update(width=dims[0], height=dims[1], mask_width=mask_dims[0], mask_height=mask_dims[1], foreground_pixels=foreground)
        rel = relations[relations.image_id == row['image_id']]
        reasons = []
        if row['image_id'] in protected_ids:
            reasons.append('protected_image_id')
        if row['image_sha256'] in protected_files:
            reasons.append('protected_file_sha256')
        if row['pixel_sha256'] in protected_pixels:
            reasons.append('protected_rgb_pixel_sha256')
        if set(rel.lesion_id.dropna()) & protected_lesions:
            reasons.append('protected_known_lesion')
        if set(rel.patient_id.dropna()) & protected_patients:
            reasons.append('protected_known_patient')
        # Existing perceptual candidates are not declarations of shared identity.
        exact = near[(near.localization_id == row['image_id']) & near.classification_id.isin(protected_ids) & near.same_pixel_sha256.eq(True)]
        if len(exact):
            reasons.append('existing_exact_pixel_relation')
        row.update(known_lesion_ids=';'.join(sorted(map(str, rel.lesion_id.dropna().unique()))),
                   known_patient_ids=';'.join(sorted(map(str, rel.patient_id.dropna().unique()))),
                   allowed_ham_train_id_overlap=row['image_id'] in set(splits['train'].image_id),
                   errors=';'.join(errors), exclusion_reasons=';'.join(reasons))
        rows.append(row)
    checked = pd.DataFrame(rows)
    checked.to_csv(OUT / 'candidate_pair_checks_private.csv', index=False)
    if checked.errors.ne('').any():
        dump('DATA_CHECK.json', {'status': 'BLOCKED', 'data_error_rows': checked.loc[checked.errors.ne(''), ['image_id', 'errors']].to_dict('records')})
        raise RuntimeError('Real image/mask data errors; see DATA_CHECK.json')
    eligible = checked[checked.exclusion_reasons.eq('')].sort_values('image_id', kind='stable')
    summary = {'status': 'READY' if len(eligible) >= 100 else 'BLOCKED', 'candidate_count': len(checked),
               'eligible_count': len(eligible), 'excluded_count': len(checked) - len(eligible),
               'selection_seed': 20261012, 'generator': 'numpy.random.Generator(PCG64)',
               'pixel_hash_definition': 'SHA256 of PIL convert RGB .tobytes(), unresized',
               'classification_rows': {k: len(v) for k, v in splits.items()},
               'existing_perceptual_candidates': len(near), 'existing_max_ssim': float(near.ssim_gray_128.max()),
               'limitation': 'No image-ID or exact-hash match is not proof of patient/lesion independence; unknown identifiers remain unknown.',
               'model_source_path': str(model), 'model_source_sha256': sha(model),
               'completed_at': datetime.now().astimezone().isoformat()}
    if len(eligible) < 100:
        dump('DATA_CHECK.json', summary)
        dump('INPUT_SOURCES.json', SOURCES)
        raise RuntimeError(f'Only {len(eligible)} eligible auxiliary pairs; require 100')
    pick = np.random.Generator(np.random.PCG64(20261012)).choice(len(eligible), size=100, replace=False)
    selected = eligible.iloc[pick].copy()
    selected.insert(0, 'selection_order', np.arange(100))
    selected.to_csv(OUT / 'fim_mask_train100.csv', index=False)
    summary['selected_count'] = 100
    summary['selected_manifest_sha256'] = sha(OUT / 'fim_mask_train100.csv')
    summary['selected_known_lesion_count'] = int(selected.known_lesion_ids.ne('').sum())
    summary['selected_known_patient_count'] = int(selected.known_patient_ids.ne('').sum())
    summary['selected_allowed_ham_train_id_overlap'] = int(selected.allowed_ham_train_id_overlap.sum())
    summary['old_localization200_overlap'] = 100
    dump('DATA_CHECK.json', summary)
    dump('INPUT_SOURCES.json', SOURCES)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
