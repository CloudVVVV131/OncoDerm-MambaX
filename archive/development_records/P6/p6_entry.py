"""Spawn-safe P6 entry; no torch/model/metric imports at module scope."""
import os
from pathlib import Path
import traceback


def main():
    root = Path(__file__).resolve().parent
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    from p6_common import dump
    import p6_training as t
    from p6_prepare import prepare, freeze
    try:
        prepare()
        freeze()
        t.main()
    except Exception:
        error = traceback.format_exc()
        t.log(error)
        dump(root / 'EXECUTION_ERROR.json', {'error': error})
        t.progress('BLOCKED')
    from p6_finish import package
    package()


if __name__ == '__main__':
    main()
