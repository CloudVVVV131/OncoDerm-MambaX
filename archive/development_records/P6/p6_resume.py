"""Resume the saved P6 epoch after a local JSON decoding correction."""
import os
import ast
import difflib
from pathlib import Path
from datetime import datetime
import traceback


def main():
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    from p6_common import OUT, read, dump, sha
    backup = OUT / 'encoding_recovery_20261016_151600'
    for item in read(backup / 'P6_CODE_FREEZE.json'):
        if Path(item['path']).name not in ['p6_training.py', 'p6_finish.py']:
            assert sha(item['path']) == item['sha256'], item['path']
    old_text = (backup / 'p6_training.py').read_text(encoding='utf-8')
    new_text = (OUT / 'p6_training.py').read_text(encoding='utf-8')
    class NormalizeEncoding(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'read_text':
                node.keywords = [k for k in node.keywords if k.arg != 'encoding']
            return node
    assert ast.dump(NormalizeEncoding().visit(ast.parse(old_text)), include_attributes=False) == ast.dump(
        NormalizeEncoding().visit(ast.parse(new_text)), include_attributes=False)
    diff = ''.join(difflib.unified_diff(old_text.splitlines(True), new_text.splitlines(True),
        fromfile='before_utf8_recovery', tofile='p6_training.py'))
    (OUT / 'UTF8_RECOVERY.diff').write_text(diff, encoding='utf-8')
    import gc
    import torch
    checkpoint = OUT / 'FIM_TKEEP_REGION_s42/last.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    required = ['model','optimizer','scheduler','scaler','rng','epoch','next_batch','history',
                'best_pair','best_epoch','global_step','epoch_loss','epoch_samples','config']
    assert all(k in saved for k in required)
    assert saved['epoch'] == 1 and saved['next_batch'] == 0 and len(saved['history']) == 1
    assert saved['global_step'] == 146
    record = {'at': datetime.now().astimezone().isoformat(), 'reason': 'UTF-8 JSON read with Windows default GBK',
        'checkpoint': str(checkpoint), 'checkpoint_sha256_before_resume': sha(checkpoint),
        'completed_epochs': 1, 'resume_epoch': 2, 'next_batch': 0,
        'complete_checkpoint_fields': required, 'scientific_trainer_AST_identical_after_ignoring_encoding_keywords': True,
        'deadline_unchanged': read(OUT / 'WINDOW.json')['deadline'],
        'completed_epoch_not_repeated': True, 'new_run_restart': False,
        'original_freeze': str(backup / 'P6_CODE_FREEZE.json'),
        'shared_source_or_environment_changes': False}
    dump(OUT / 'RUNTIME_RECOVERY.json', record)
    del saved
    gc.collect()
    from p6_prepare import freeze
    freeze()
    import p6_training as t
    try:
        t.main()
    except Exception:
        error = traceback.format_exc()
        t.log(error)
        dump(OUT / 'RESUME_ERROR.json', {'error': error})
        t.progress('BLOCKED')
    from p6_finish import package
    package()


if __name__ == '__main__':
    main()
