"""Stdlib-only launcher; spawned workers never import training or statistics."""
import os
import traceback
from datetime import datetime

def main():
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[key]='2'
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    from p4_common import OUT, dump, update_progress
    try:
        from p4_evaluate import main as evaluate
        evaluate()
    except Exception:
        dump(OUT/f'EVALUATION_ERROR_{datetime.now():%Y%m%d_%H%M%S}.json',{'error':traceback.format_exc()})
        update_progress(status='BLOCKED',evaluation_status='BLOCKED')
        raise

if __name__=='__main__':
    main()
