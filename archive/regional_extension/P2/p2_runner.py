"""Sequential local P2 worker; package actual complete/partial/blocked state."""
from datetime import datetime
import traceback
import p2_training as t
from p2_finish import package


if __name__ == '__main__':
    failed = False
    try:
        t.main()
    except Exception:
        failed = True
        error = traceback.format_exc()
        t.log(error)
        t.dump(t.OUT / f'TRAINING_ERROR_{datetime.now():%Y%m%d_%H%M%S}.json',
            {'error': error, 'at': datetime.now().astimezone().isoformat()})
        t.progress('BLOCKED')
    package()
    if failed:
        raise SystemExit(1)
