"""Materialize declared path tokens in a NEW configuration copy; never run code."""
import argparse
import os
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.source.suffix not in {".json", ".yaml", ".yml", ".toml", ".txt", ".md", ".csv"}:
        parser.error("Only text configurations/records are accepted, never Python source")
    source = args.source.read_text(encoding="utf-8-sig")
    variables = sorted(set(re.findall(r"\$\{(R52_PATH_[A-F0-9]+)\}", source)))
    missing = [key for key in variables if not os.environ.get(key)]
    if missing:
        parser.error("Unset path variables: " + ", ".join(missing))
    for key in variables:
        value = os.environ[key]
        # Forward slashes remain valid Windows paths and avoid JSON escape errors.
        value = value.replace("\\", "/")
        if any(char in value for char in ['"', "\n", "\r"]):
            parser.error("Path value contains unsafe configuration characters")
        source = source.replace("${" + key + "}", value)
    if args.destination.resolve() == args.source.resolve():
        parser.error("Destination must differ from source")
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with args.destination.open("x", encoding="utf-8") as stream:
        stream.write(source)
    print("New text copy written; no study script was imported or executed.")


if __name__ == "__main__":
    main()
