"""Stdlib-only entrypoint: workers never import the trainer or model libraries."""
import traceback
from datetime import datetime


def main():
    import p3_training as t
    try:
        t.main()
    except Exception:
        error = traceback.format_exc()
        t.log(error)
        t.dump(t.OUT / f'TRAINING_ERROR_{datetime.now():%Y%m%d_%H%M%S}.json', {'error': error})
        t.progress('BLOCKED')
        raise


if __name__ == '__main__':
    main()
