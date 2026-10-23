"""Minimal spawn-safe entry; model code imported in the parent process only."""
if __name__=='__main__':
    from core_infer import main
    from core_common import progress
    try: main()
    except Exception as exc:
        progress(status='BLOCKED',inference='BLOCKED',error=repr(exc))
        raise
