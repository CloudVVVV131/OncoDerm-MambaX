"""Stdlib-only evaluation entrypoint, preserving Windows worker isolation."""
import traceback
from datetime import datetime


def main():
    from p3_common import OUT,dump,update_progress
    try:
        from p3_prepare_evaluation import freeze
        freeze()
        from p3_evaluate import main as evaluate
        evaluate()
    except Exception:
        dump(OUT/f'EVALUATION_ERROR_{datetime.now():%Y%m%d_%H%M%S}.json',{'error':traceback.format_exc()})
        update_progress(status='BLOCKED',evaluation_status='BLOCKED')
        raise


if __name__=='__main__':
    main()
