from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


MANIFEST_MEMBER = "project/COLD_START_ARTIFACT_MANIFEST.json"


def hash_stream(stream: object) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    while block := stream.read(8 * 1024 * 1024):
        size += len(block)
        digest.update(block)
    return size, digest.hexdigest()


def safe_member_name(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and bool(path.parts)
        and path.parts[0] == "project"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    archive_path = Path(args.archive).resolve()
    errors: list[str] = []
    checks: list[dict[str, object]] = []
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        unsafe = [
            member.name for member in members if not safe_member_name(member.name)
        ]
        checks.append(
            {
                "name": "all_archive_paths_safe",
                "passed": not unsafe,
                "detail": unsafe[:20],
            }
        )
        if unsafe:
            errors.append("all_archive_paths_safe")
        by_name = {member.name: member for member in members}
        manifest_member = by_name.get(MANIFEST_MEMBER)
        if manifest_member is None or not manifest_member.isfile():
            raise SystemExit(f"Archive lacks {MANIFEST_MEMBER}")
        manifest_stream = archive.extractfile(manifest_member)
        if manifest_stream is None:
            raise SystemExit("Could not read cold-start manifest from archive")
        manifest = json.load(manifest_stream)
        expected_files = manifest.get("files", [])
        checks.append(
            {
                "name": "manifest_inventory_present",
                "passed": isinstance(expected_files, list) and bool(expected_files),
                "detail": len(expected_files)
                if isinstance(expected_files, list)
                else None,
            }
        )
        if not isinstance(expected_files, list) or not expected_files:
            errors.append("manifest_inventory_present")
            expected_files = []

        expected_member_names = {MANIFEST_MEMBER}
        for entry in expected_files:
            relative = str(entry["path"])
            member_name = f"project/{relative}"
            expected_member_names.add(member_name)
            member = by_name.get(member_name)
            if member is None or not member.isfile():
                errors.append(f"missing_member:{relative}")
                continue
            stream = archive.extractfile(member)
            if stream is None:
                errors.append(f"unreadable_member:{relative}")
                continue
            observed_size, observed_hash = hash_stream(stream)
            if observed_size != int(entry["bytes"]):
                errors.append(f"size_mismatch:{relative}")
            if observed_hash != str(entry["sha256"]).lower():
                errors.append(f"sha256_mismatch:{relative}")

        unexpected_files = sorted(
            member.name
            for member in members
            if member.isfile() and member.name not in expected_member_names
        )
        if unexpected_files:
            errors.append("unexpected_archive_files")
        checks.append(
            {
                "name": "archive_matches_manifest_exactly",
                "passed": not unexpected_files
                and not any(
                    error.startswith(
                        (
                            "missing_member:",
                            "unreadable_member:",
                            "size_mismatch:",
                            "sha256_mismatch:",
                        )
                    )
                    for error in errors
                ),
                "detail": {
                    "unexpected_files": unexpected_files[:20],
                    "expected_files": len(expected_files),
                },
            }
        )

        public_raw_files = [
            member.name
            for member in members
            if member.isfile() and member.name.startswith("project/data/raw/")
        ]
        if public_raw_files:
            errors.append("public_raw_data_in_upload_archive")
        checks.append(
            {
                "name": "public_raw_data_excluded",
                "passed": not public_raw_files,
                "detail": public_raw_files[:20],
            }
        )

    with archive_path.open("rb") as archive_stream:
        archive_sha256 = hash_stream(archive_stream)[1]
    payload = {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": archive_sha256,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "errors": errors,
    }
    output = Path(args.output) if args.output else Path(f"{archive_path}.qa.json")
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"cold_start_archive status={payload['status']} bytes={payload['archive_bytes']} "
        f"errors={len(errors)} qa={output}"
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
