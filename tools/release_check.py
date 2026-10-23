#!/usr/bin/env python3
"""Stdlib-only release integrity and text path binding. Never executes research code."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys

TOKEN = re.compile(r"\$\{(R52_PATH_[A-Z0-9_]+)\}")
KEY = re.compile(r"[a-z][a-z0-9_]*\Z")
MANIFEST = "RELEASE_MANIFEST.csv"
TEXT_SUFFIXES = {".json", ".yaml", ".yml", ".toml"}
MAX_TEXT_BYTES = 4 * 1024 * 1024


class CheckError(Exception):
    pass


def ensure(condition, message):
    if not condition:
        raise CheckError(message)


def inside(path, root):
    return path == root or root in path.parents


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def reject_links(path):
    for part in (path, *path.parents):
        ensure(not part.is_symlink(), "Symlink is unsupported: " + str(part))
        if hasattr(part, "is_junction"):
            ensure(not part.is_junction(), "Junction is unsupported: " + str(part))


def release_root(value):
    raw = Path(value).absolute()
    reject_links(raw)
    result = raw.resolve(strict=True)
    ensure(result.is_dir(), "Release root must be a directory")
    return result


def release_file(root, relative):
    rel = PurePosixPath(relative)
    ensure(relative and "\\" not in relative and ":" not in relative,
           "Use a release-relative POSIX path")
    ensure(not rel.is_absolute() and all(p not in {"", ".", ".."} for p in rel.parts),
           "Unsafe release-relative path: " + relative)
    candidate = root.joinpath(*rel.parts)
    reject_links(candidate)
    actual = candidate.resolve(strict=True)
    ensure(inside(actual, root) and actual.is_file(), "Not a release file: " + relative)
    return actual


def read_json(path):
    ensure(path.stat().st_size <= MAX_TEXT_BYTES, "JSON input too large")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify(root):
    """Verify the release itself, not any external scientific input or result."""
    manifest = release_file(root, MANIFEST)
    seen = set()
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        ensure(set(reader.fieldnames or []) == {"path", "size_bytes", "sha256"},
               "Manifest requires exactly path,size_bytes,sha256 columns")
        for row in reader:
            name = row["path"]
            ensure(name != MANIFEST, "Manifest must not hash itself")
            ensure(name not in seen, "Duplicate manifest path: " + name)
            seen.add(name)
            item = release_file(root, name)
            ensure(re.fullmatch(r"[a-f0-9]{64}", row["sha256"] or ""), "Invalid SHA: " + name)
            ensure(item.stat().st_size == int(row["size_bytes"]), "Size mismatch: " + name)
            ensure(file_sha(item) == row["sha256"], "SHA mismatch: " + name)
    actual = set()
    for item in root.rglob("*"):
        reject_links(item)
        if item.is_file() and item != manifest:
            actual.add(item.relative_to(root).as_posix())
    ensure(actual == seen, "Manifest membership mismatch: unlisted=" + str(sorted(actual - seen))
           + "; missing=" + str(sorted(seen - actual)))
    return {"status": "PASS", "operation": "verify", "verified_files": len(seen),
            "manifest_sha256": file_sha(manifest), "research_execution": "NOT_RUN"}


def load_schema(root, name):
    path = release_file(root, name)
    data = read_json(path)
    ensure(set(data) == {"schema_version", "paths"} and data["schema_version"] == 1,
           "Expected schema_version=1 and paths")
    ensure(isinstance(data["paths"], dict) and data["paths"], "paths must be a nonempty object")
    tokens = set()
    for key, spec in data["paths"].items():
        ensure(KEY.fullmatch(key), "Invalid logical key: " + key)
        ensure(isinstance(spec, dict) and set(spec) == {"token", "kind", "description"},
               "Schema entry requires token,kind,description: " + key)
        ensure(re.fullmatch(r"R52_PATH_[A-Z0-9_]+", spec["token"]), "Invalid token: " + key)
        ensure(spec["token"] not in tokens, "Repeated token")
        tokens.add(spec["token"])
        ensure(spec["kind"] in {"file", "directory"}, "Unsupported path kind: " + key)
        ensure(isinstance(spec["description"], str) and spec["description"], "Missing description")
    return data["paths"], file_sha(path)


def plan(root, schema_name, bindings_name, template_name, workspace_name=None):
    specs, schema_sha = load_schema(root, schema_name)
    binding_path = Path(bindings_name).absolute()
    reject_links(binding_path)
    binding_path = binding_path.resolve(strict=True)
    ensure(not inside(binding_path, root), "Keep private bindings outside the public release")
    binding = read_json(binding_path)
    ensure(set(binding) == {"paths"} and isinstance(binding["paths"], dict), "Bindings require paths object")
    supplied = binding["paths"]
    ensure(set(supplied) == set(specs), "Binding keys differ: unknown=" + str(sorted(set(supplied) - set(specs)))
           + "; missing=" + str(sorted(set(specs) - set(supplied))))
    paths = {}
    for key, value in supplied.items():
        ensure(isinstance(value, str) and value and "R52_PATH_" not in value and "${" not in value,
               "Binding must be a resolved path: " + key)
        raw = Path(value)
        ensure(raw.is_absolute(), "Binding must be absolute on this host: " + key)
        reject_links(raw)
        target = raw.resolve(strict=True)
        ensure(not inside(target, root), "Input must be outside release: " + key)
        ensure(target.is_file() if specs[key]["kind"] == "file" else target.is_dir(),
               "Input kind mismatch: " + key)
        paths[key] = target
    template = release_file(root, template_name)
    ensure(template.suffix.lower() in TEXT_SUFFIXES, "Only JSON/YAML/TOML text templates supported")
    ensure(template.name != "MATERIALIZATION_RECEIPT.json", "Reserved output name")
    ensure(template.stat().st_size <= MAX_TEXT_BYTES, "Text template too large")
    source = template.read_bytes()
    text = source.decode("utf-8-sig")
    used = set(TOKEN.findall(text))
    token_keys = {spec["token"]: key for key, spec in specs.items()}
    ensure(used <= set(token_keys), "Unbound template tokens: " + str(sorted(used - set(token_keys))))
    # Use forward slashes, safe in JSON strings and YAML/TOML paths. Reject characters
    # that would require changing a template's quoting; never reserialize the config.
    replacements = {}
    for token in used:
        path_text = paths[token_keys[token]].as_posix()
        ensure(not any(char in path_text for char in ('"', "'", "\n", "\r", "\t", "\\", "#", "{", "}", "[", "]")),
               "Path requires unsafe text escaping; use a simple local alias: " + token_keys[token])
        replacements[token] = path_text
    rendered_text = TOKEN.sub(lambda match: replacements[match.group(1)], text)
    ensure("R52_PATH_" not in rendered_text and "${" not in rendered_text,
           "Unresolved or unsupported token remains")
    if template.suffix.lower() == ".json":
        json.loads(text)
        json.loads(rendered_text)
    # YAML/TOML are deliberately text-only: no third-party parser or research import.
    rendered = (b"\xef\xbb\xbf" if source.startswith(b"\xef\xbb\xbf") else b"") + rendered_text.encode("utf-8")
    workspace = None
    if workspace_name:
        raw_workspace = Path(workspace_name).absolute()
        reject_links(raw_workspace)
        workspace = raw_workspace.resolve()
        ensure(not inside(workspace, root) and not inside(root, workspace), "Workspace must be separate from release")
        ensure(not workspace.exists(), "Workspace already exists; overwrite refused")
        ensure(workspace.parent.is_dir(), "Workspace parent must exist")
        for key, value in paths.items():
            ensure(not inside(value, workspace) and not inside(workspace, value),
                   "Workspace overlaps input: " + key)
        ensure(not inside(binding_path, workspace), "Workspace overlaps bindings")
    report = {"status": "PASS", "operation": "check-plan", "schema_sha256": schema_sha,
              "template": template_name, "template_sha256": sha(source),
              "rendered_config_sha256": sha(rendered), "binding_sha256": file_sha(binding_path),
              "logical_inputs": [{"key": key, "kind": specs[key]["kind"], "exists": True} for key in sorted(paths)],
              "replaced_tokens": sorted(used), "change_scope": "path tokens only; every other source byte preserved",
              "external_scientific_identities": "NOT_VALIDATED; historical checks must remain enabled",
              "research_execution": "NOT_RUN", "external_data_loaded": False,
              "yaml_toml_semantic_parsing": "NOT_IMPLEMENTED; text-only",
              "source_template_read_only": True}
    return report, workspace, template.name, rendered


def run(args):
    root = release_root(args.release_root)
    if args.command == "verify":
        return verify(root)
    report, workspace, name, rendered = plan(root, args.schema, args.bindings,
                                             args.template, getattr(args, "workspace", None))
    if args.command == "materialize":
        ensure(workspace is not None, "Workspace required")
        workspace.mkdir()  # Exclusive creation: no overwrite or reuse.
        with (workspace / name).open("xb") as handle:
            handle.write(rendered)
        report["operation"] = "materialize"
        report["output_name"] = name
        report["execution_permission"] = "NONE; this is a text preparation receipt"
        with (workspace / "MATERIALIZATION_RECEIPT.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("verify", "check-plan", "materialize"):
        child = sub.add_parser(command)
        child.add_argument("--release-root", required=True)
        if command != "verify":
            child.add_argument("--schema", default="logical_paths.example.json",
                               help="Release-relative schema JSON")
            child.add_argument("--bindings", required=True, help="Private local bindings outside release")
            child.add_argument("--template", required=True, help="Release-relative JSON/YAML/TOML file")
        if command == "materialize":
            child.add_argument("--workspace", required=True, help="New directory outside release and inputs")
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (CheckError, OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc), "research_execution": "NOT_RUN"}))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
