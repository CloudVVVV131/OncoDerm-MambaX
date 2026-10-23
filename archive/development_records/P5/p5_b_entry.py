"""Windows light-worker launcher; no old-stage entrypoint imports."""
import os
import traceback
from datetime import datetime

def main():
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='2'
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    from p5_common import OUT,dump,progress
    try:
        from p5_task_b import main as evaluate
        evaluate()
    except Exception:
        dump(OUT/f'B_ERROR_{datetime.now():%Y%m%d_%H%M%S}.json',{'error':traceback.format_exc()})
        progress(status='BLOCKED',task_B='BLOCKED');raise

if __name__=='__main__':main()
