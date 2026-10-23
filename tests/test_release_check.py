"""Finite synthetic packaging checks; no research imports or computations."""
import argparse
import ast
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import types


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--archived-template", help="Optional existing INPUT_SOURCES.json text-only fixture; never executes it")
    args = parser.parse_args()
    report_path = Path(args.report).absolute()
    if report_path.exists():
        raise SystemExit("Refusing to overwrite a test report")
    wrapper = Path(__file__).resolve().parents[1] / "tools" / "release_check.py"
    started = time.monotonic()
    scratch = Path(tempfile.mkdtemp(prefix="r52_release_synthetic_"))
    release = scratch / "release"
    inputs = scratch / "synthetic_inputs"
    release.mkdir()
    inputs.mkdir()
    for name in ("train.csv", "val.csv"):
        (inputs / name).touch()
    (inputs / "empty_image_directory").mkdir()
    (release / "schema.json").write_text(json.dumps({"schema_version": 1, "paths": {
        "ham_train_members": {"token": "R52_PATH_TEST_TRAIN", "kind": "file", "description": "Synthetic only"},
        "image_directory": {"token": "R52_PATH_TEST_IMAGES", "kind": "directory", "description": "Empty synthetic only"}
    }}), encoding="utf-8")
    template_data = b'{\r\n  "source": "${R52_PATH_TEST_TRAIN}",\r\n  "images": "${R52_PATH_TEST_IMAGES}",\r\n  "weight_identity": "UNCHANGED_FAKE_HASH",\r\n  "epochs": 80, "authorized": false\r\n}\r\n'
    (release / "config.json").write_bytes(template_data)
    (release / "config.yaml").write_text('source: "${R52_PATH_TEST_TRAIN}"\nepochs: 80\n', encoding="utf-8")
    (release / "config.toml").write_text('source = "${R52_PATH_TEST_TRAIN}"\nepochs = 80\n', encoding="utf-8")
    (release / "unresolved.json").write_text('{"x":"${R52_PATH_UNKNOWN}"}', encoding="utf-8")
    binding = {"paths": {"ham_train_members": str(inputs / "train.csv"),
                           "image_directory": str(inputs / "empty_image_directory")}}
    bindings = scratch / "bindings.json"
    bindings.write_text(json.dumps(binding), encoding="utf-8")
    bad_bindings = scratch / "unknown_bindings.json"
    bad_bindings.write_text(json.dumps({"paths": dict(binding["paths"], unknown="bad")}), encoding="utf-8")
    archived_info = None
    if args.archived_template:
        archived_source = Path(args.archived_template).resolve(strict=True)
        assert archived_source.name == "INPUT_SOURCES.json"
        assert archived_source.stat().st_size < 65536
        actual_text = archived_source.read_bytes()
        entries = json.loads(actual_text)
        known_inputs = {
            "R52_PATH_E37E406F66F5": "ham_train_members",
            "R52_PATH_A7EDEF14667C": "ham_validation_members",
            "R52_PATH_1E005046B878": "ham_test_members",
            "R52_PATH_332D999CFE2E": "ham_recovered_image_index",
            "R52_PATH_3691B63AD4FB": "siim_recovered_image_index",
            "R52_PATH_8177255F37A8": "historical_localization200_index",
            "R52_PATH_944ABBADAEF1": "historical_data_manifest",
            "R52_PATH_3598D7B71D40": "historical_localization_duplicate_record"
        }
        assert len(entries) == len(known_inputs)
        assert {entry["path"][2:-1] for entry in entries} == set(known_inputs)
        archived_schema = {"schema_version": 1, "paths": {}}
        archived_bindings = {"paths": {}}
        for token, key in known_inputs.items():
            archived_schema["paths"][key] = {"token": token, "kind": "file", "description": "Synthetic placeholder for archived text only"}
            placeholder = inputs / (key + ".csv")
            placeholder.touch()
            archived_bindings["paths"][key] = str(placeholder)
        (release / "archived_schema.json").write_text(json.dumps(archived_schema), encoding="utf-8")
        (release / "INPUT_SOURCES.json").write_bytes(actual_text)
        (scratch / "archived_bindings.json").write_text(json.dumps(archived_bindings), encoding="utf-8")
        archived_info = {"archive_relative_path": "archive/controlled_mask_supervision/seed42/INPUT_SOURCES.json",
                         "template_sha256": hashlib.sha256(actual_text).hexdigest(), "input_entries": len(entries),
                         "real_scientific_inputs_checked": False, "purpose": "Actual archived token syntax and exact nonpath preservation only"}
    with (release / "RELEASE_MANIFEST.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "size_bytes", "sha256"])
        writer.writeheader()
        for path in sorted(release.iterdir()):
            if path.name != "RELEASE_MANIFEST.csv":
                writer.writerow({"path": path.name, "size_bytes": path.stat().st_size, "sha256": digest(path)})
    bad_release = scratch / "tampered_release"
    shutil.copytree(release, bad_release)
    (bad_release / "config.json").write_bytes(template_data.replace(b"80", b"79"))
    before = {x.relative_to(release).as_posix(): digest(x) for x in release.rglob("*") if x.is_file()}
    forbidden_modules = {"torch", "numpy", "timm", "torchvision", "tensorflow", "lesionmamba", "ceu"}
    allowed_imports = {"argparse", "csv", "hashlib", "json", "os", "pathlib", "re", "sys"}
    source = wrapper.read_text(encoding="utf-8")
    tree = ast.parse(source)
    actual_imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            actual_imports.update(x.name.split(".")[0] for x in node.names)
        elif isinstance(node, ast.ImportFrom):
            actual_imports.add((node.module or "").split(".")[0])
    assert actual_imports <= allowed_imports
    blocked_events = []
    wrapper_active = [False]
    current_inputs_read = []
    guard_probes = [0]
    source_write_attempts = []

    def audit(event, values):
        if not wrapper_active[0]:
            return
        denied = (event.startswith("socket.") or event.startswith("subprocess.")
                  or event in {"os.system", "os.fork", "os.posix_spawn", "os.spawn", "pty.spawn"})
        if denied:
            blocked_events.append(event)
            raise RuntimeError("Audit denied side effect: " + event)
        if event == "import" and str(values[0]).split(".")[0] in forbidden_modules:
            blocked_events.append("import:" + str(values[0]))
            raise RuntimeError("Research module import denied")
        if event == "open" and values and isinstance(values[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(values[0])).absolute()
            if path.suffix.lower() in {".pt", ".pth", ".safetensors", ".npz", ".npy", ".pkl"}:
                blocked_events.append("tensor-open")
                raise RuntimeError("Tensor/serialized model input denied")
            mode = values[1]
            flags = values[2] if len(values) > 2 and isinstance(values[2], int) else 0
            writing = (isinstance(mode, str) and any(x in mode for x in "wax+")) or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            protected = path == release or release in path.parents or path == inputs or inputs in path.parents
            if protected and writing:
                source_write_attempts.append(str(path.relative_to(scratch)))
                raise RuntimeError("Synthetic source/input write denied")
            if path == inputs or inputs in path.parents:
                current_inputs_read.append(path.name)
        if event in {"os.mkdir", "os.remove", "os.rmdir", "os.rename"}:
            for raw in values[:2]:
                if isinstance(raw, (str, bytes, os.PathLike)):
                    path = Path(os.fsdecode(raw)).absolute()
                    if path == release or release in path.parents or path == inputs or inputs in path.parents:
                        source_write_attempts.append(str(path.relative_to(scratch)))
                        raise RuntimeError("Source/input filesystem mutation denied")

    sys.addaudithook(audit)
    wrapper_active[0] = True
    # Prove the guard is active without making network/process/tensor calls.
    for event, values in (("socket.__new__", (None, 0, 0, 0)),
                          ("subprocess.Popen", ("not-started", [], None, None)),
                          ("open", (str(scratch / "not-read.pt"), "rb", 0))):
        try:
            sys.audit(event, *values)
        except RuntimeError:
            guard_probes[0] += 1
    probe_events = list(blocked_events)
    blocked_events.clear()
    namespace = {"__name__": "release_wrapper_under_test", "__file__": str(wrapper)}
    exec(compile(source, str(wrapper), "exec"), namespace)
    wrapper_main = namespace["main"]
    tests = []

    def call(name, argv, expected):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = wrapper_main(argv)
        data = json.loads(out.getvalue())
        assert code == expected, (name, code, data)
        tests.append({"name": name, "exit_code": code, "expected_exit_code": expected,
                      "status": "PASS", "operation_result": data["status"]})
        return data

    base = ["--release-root", str(release), "--schema", "schema.json", "--bindings", str(bindings)]
    call("manifest_integrity", ["verify", "--release-root", str(release)], 0)
    call("tampered_manifest_rejected", ["verify", "--release-root", str(bad_release)], 2)
    call("resolved_paths_stat_only", ["check-plan", *base, "--template", "config.json"], 0)
    call("yaml_text_only", ["check-plan", *base, "--template", "config.yaml"], 0)
    call("toml_text_only", ["check-plan", *base, "--template", "config.toml"], 0)
    workspace = scratch / "new_workspace"
    materialized = call("materialize_new_workspace", ["materialize", *base, "--template", "config.json", "--workspace", str(workspace)], 0)
    call("overwrite_rejected", ["materialize", *base, "--template", "config.json", "--workspace", str(workspace)], 2)
    call("workspace_in_release_rejected", ["materialize", *base, "--template", "config.json", "--workspace", str(release / "forbidden")], 2)
    call("workspace_in_input_rejected", ["materialize", *base, "--template", "config.json", "--workspace", str(inputs / "empty_image_directory" / "forbidden")], 2)
    call("unresolved_token_rejected", ["check-plan", *base, "--template", "unresolved.json"], 2)
    call("path_traversal_rejected", ["check-plan", *base, "--template", "../bindings.json"], 2)
    call("unknown_binding_key_rejected", ["check-plan", "--release-root", str(release), "--schema", "schema.json", "--bindings", str(bad_bindings), "--template", "config.json"], 2)
    expected = template_data.replace(b"${R52_PATH_TEST_TRAIN}", (inputs / "train.csv").as_posix().encode()).replace(b"${R52_PATH_TEST_IMAGES}", (inputs / "empty_image_directory").as_posix().encode())
    assert (workspace / "config.json").read_bytes() == expected
    assert materialized["template_sha256"] == hashlib.sha256(template_data).hexdigest()
    if archived_info:
        archived_workspace = scratch / "archived_workspace"
        call("actual_archived_template_materialization", ["materialize", "--release-root", str(release),
             "--schema", "archived_schema.json", "--bindings", str(scratch / "archived_bindings.json"),
             "--template", "INPUT_SOURCES.json", "--workspace", str(archived_workspace)], 0)
        materialized_entries = json.loads((archived_workspace / "INPUT_SOURCES.json").read_bytes())
        assert all({k: v for k, v in a.items() if k != "path"} == {k: v for k, v in b.items() if k != "path"}
                   for a, b in zip(entries, materialized_entries))
        actual_expected = actual_text
        for token, key in known_inputs.items():
            actual_expected = actual_expected.replace(("${" + token + "}").encode(), Path(archived_bindings["paths"][key]).as_posix().encode())
        assert (archived_workspace / "INPUT_SOURCES.json").read_bytes() == actual_expected
        archived_info["original_bytes_and_sha_fields_preserved"] = True
        archived_info["nonpath_bytes_preserved"] = True
    after = {x.relative_to(release).as_posix(): digest(x) for x in release.rglob("*") if x.is_file()}
    assert before == after
    assert not blocked_events and not source_write_attempts and not current_inputs_read
    assert not forbidden_modules.intersection(sys.modules)
    wrapper_active[0] = False
    receipt = {"status": "PASS", "scope": "synthetic packaging only; no research reproduction",
               "wrapper_sha256": digest(wrapper), "tests": tests, "test_count": len(tests),
               "import_roots": sorted(actual_imports), "stdlib_only_imports": True,
               "audit_guard_probe_count": guard_probes[0], "audit_probe_events": probe_events,
               "wrapper_forbidden_side_effect_attempts": blocked_events,
               "source_or_input_write_attempts": source_write_attempts,
               "external_input_contents_read": current_inputs_read,
               "source_tree_identical": before == after, "nonpath_bytes_preserved": True,
               "research_modules_imported": [], "network_calls": 0, "processes_started": 0,
               "tensor_files_loaded": 0, "elapsed_seconds": time.monotonic() - started,
               "real_inputs_validated": False, "end_to_end_reproduction": "NOT_RUN"}
    receipt["actual_archived_template_check"] = archived_info
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    # Only remove this script's verified synthetic temporary tree, never release inputs.
    assert scratch.parent == Path(tempfile.gettempdir()).resolve()
    assert scratch.name.startswith("r52_release_synthetic_")
    shutil.rmtree(scratch)
    print(json.dumps({"status": "PASS", "tests": len(tests), "seconds": receipt["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
