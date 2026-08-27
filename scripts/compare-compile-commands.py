#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compares compile_commands.json between GN and CMake builds."""

import argparse
import json
from pathlib import Path
import re
import signal
import sys

# Ignore SIGPIPE on Linux (this happens when piping the output to head/tail)
if hasattr(signal, "SIGPIPE"):
  signal.signal(signal.SIGPIPE, signal.SIG_DFL)

DEFAULT_DIRS = ["src", "bench", "test", "models", "eval", "tools", "litert"]
UNSET = "<unset>"
EXIT_SUCCESS, EXIT_DIFF, EXIT_ERR = 0, 1, 255

DEFINE_RE = re.compile(
    r'(?:^|\s)(?:"[-/]D((?:\\.|[^"\\])*)"|\'[-/]D((?:\\.|[^\'\\])*)\'|[-/]D((?:\\'
    r" |[^ \t\r\n])+))"
)


def find_repo_root(start: Path) -> Path:
  """Finds repo root by checking ancestors for markers."""
  p = start.resolve()
  p = p.parent if p.is_file() else p
  for parent in [p, *p.parents]:
    if (parent / "include" / "xnnpack.h").exists() or (
        parent / "CMakeLists.txt"
    ).exists():
      return parent
  for parent in [p, *p.parents]:
    if (parent / ".git").exists() or (parent / "BUILD.gn").exists():
      return parent
  return p


def parse_define(d: str) -> tuple[str, str]:
  """Parses a define token into (name, value)."""
  name, sep, val = d.partition("=")
  if not sep:
    return name, "1"
  if val.startswith(('"', "'")) and val.endswith(('"', "'")) and len(val) >= 2:
    val = val[1:-1]
  return name, val.replace('\\"', '"')


def extract_defines(
    cmd: str | list[str],
    include_re: re.Pattern[str],
    ignore_res: list[re.Pattern[str]],
) -> dict[str, str]:
  """Extracts define flags from a command string or argument list."""
  raw = (
      [t[2:] for t in cmd if t.startswith(("-D", "/D")) and len(t) > 2]
      if isinstance(cmd, list)
      else [m[0] or m[1] or m[2] for m in DEFINE_RE.findall(cmd)]
  )
  defs = {}
  for d in raw:
    name, val = parse_define(d)
    if any(r.search(name) for r in ignore_res):
      continue
    if include_re.search(name):
      defs[name] = val
  return defs


def is_known_source(rel_posix: str, include_dirs: tuple[str, ...]) -> bool:
  return rel_posix.startswith(include_dirs)


def load_commands(
    path: Path,
    repo_root: Path,
    include_re: re.Pattern[str],
    ignore_res: list[re.Pattern[str]],
    include_dirs: tuple[str, ...],
) -> dict[str, dict[str, str]]:
  """Loads compile_commands.json and returns file -> defines mapping."""
  if not path.is_file():
    sys.stderr.write(f"Error: File not found: '{path}'\n")
    sys.exit(EXIT_ERR)
  try:
    entries = json.loads(path.read_text(encoding="utf-8"))
  except Exception as e:
    sys.stderr.write(f"Error reading '{path}': {e}\n")
    sys.exit(EXIT_ERR)

  if not isinstance(entries, list):
    sys.stderr.write(f"Error: '{path}' is not a valid JSON list.\n")
    sys.exit(EXIT_ERR)

  result: dict[str, dict[str, str]] = {}
  for e in entries:
    if not isinstance(e, dict) or not (f := e.get("file")):
      continue
    abs_p = (Path(e.get("directory", repo_root)) / f).resolve()
    try:
      rel_p = abs_p.relative_to(repo_root).as_posix()
    except ValueError:
      rel_p = abs_p.as_posix()

    if is_known_source(rel_p, include_dirs) and rel_p not in result:
      cmd = e.get("command") or e.get("arguments", [])
      result[rel_p] = extract_defines(cmd, include_re, ignore_res)
  return result


def compare(
    data1: dict[str, dict[str, str]],
    data2: dict[str, dict[str, str]],
    pedantic: bool,
):
  """Compares two compile command maps."""
  files1, files2 = set(data1), set(data2)
  only1, only2 = sorted(files1 - files2), sorted(files2 - files1)
  diffs, match_count = [], 0

  for f in sorted(files1 & files2):
    d1, d2 = data1[f], data2[f]
    file_diffs = []
    for k in sorted(set(d1) | set(d2)):
      v1, v2 = d1.get(k), d2.get(k)
      if v1 == v2 or (not pedantic and {v1, v2} <= {"0", None}):
        continue
      file_diffs.append((k, v1 or UNSET, v2 or UNSET))
    if file_diffs:
      diffs.append((f, file_diffs))
    else:
      match_count += 1
  return only1, only2, diffs, match_count


def print_report(
    label1: str,
    label2: str,
    path1: Path,
    path2: Path,
    only1: list[str],
    only2: list[str],
    diffs: list[tuple[str, list[tuple[str, str, str]]]],
    match_count: int,
    total1: int,
    total2: int,
) -> None:
  """Prints text comparison summary."""
  print(
      f"Comparing compile commands:\n  {label1}: {path1} ({total1} files)\n "
      f" {label2}: {path2} ({total2} files)\n{'=' * 80}"
  )
  if only1:
    print(
        f"\nMissing in {label2} (only in {label1}) [{len(only1)} files]:\n"
        + "\n".join(f"  - {f}" for f in only1)
    )
  if only2:
    print(
        f"\nMissing in {label1} (only in {label2}) [{len(only2)} files]:\n"
        + "\n".join(f"  - {f}" for f in only2)
    )
  if diffs:
    print(f"\nDefine differences [{len(diffs)} files]:")
    for f, f_diffs in diffs:
      print(f"  {f}:")
      for d, v1, v2 in f_diffs:
        print(f"    {d}: {label1}={v1}, {label2}={v2}")

  print(f"\n{'=' * 80}\nSummary:")
  print(
      f"  {label1} files analyzed: {total1}\n  {label2} files analyzed:"
      f" {total2}"
  )
  print(
      f"  Files only in {label1}: {len(only1)}\n  Files only in {label2}:"
      f" {len(only2)}"
  )
  print(
      f"  Files with define differences: {len(diffs)}\n  Identically matching"
      f" files: {match_count}"
  )
  print(
      "\n** SUCCESS: All files and defines match identically **"
      if not (only1 or only2 or diffs)
      else "\n** DIFFERENCES DETECTED **"
  )


def output_jsonl(
    label1: str,
    label2: str,
    path1: Path,
    path2: Path,
    only1: list[str],
    only2: list[str],
    diffs: list[tuple[str, list[tuple[str, str, str]]]],
    match_count: int,
    total1: int,
    total2: int,
) -> None:
  """Outputs JSON Lines (JSONL) formatted comparison result."""
  print(
      json.dumps({
          "type": "summary",
          "label1": label1,
          "label2": label2,
          "path1": str(path1),
          "path2": str(path2),
          "total_files_1": total1,
          "total_files_2": total2,
          "only_in_1_count": len(only1),
          "only_in_2_count": len(only2),
          "files_with_define_differences_count": len(diffs),
          "matching_files_count": match_count,
          "has_differences": bool(only1 or only2 or diffs),
      })
  )
  for f in only1:
    print(json.dumps({"type": "only_in_1", "file": f}))
  for f in only2:
    print(json.dumps({"type": "only_in_2", "file": f}))
  for f, f_diffs in diffs:
    print(
        json.dumps({
            "type": "define_difference",
            "file": f,
            "differences": [
                {"define": d, "val1": v1, "val2": v2} for d, v1, v2 in f_diffs
            ],
        })
    )


def main() -> None:
  parser = argparse.ArgumentParser(
      "compare-compile-commands",
      description="Compares compile_commands.json between GN and CMake builds.",
  )
  parser.add_argument(
      "compile_commands_1",
      type=Path,
      help="Path to first compile_commands.json",
  )
  parser.add_argument(
      "compile_commands_2",
      type=Path,
      help="Path to second compile_commands.json",
  )
  parser.add_argument(
      "--include-define",
      default="^XNN_",
      help="Regex pattern of define macro(s) to include (default: '^XNN_')",
  )
  parser.add_argument(
      "--ignore-define",
      action="append",
      default=[],
      help="Regex pattern of define macro(s) to ignore (can be repeated)",
  )
  parser.add_argument(
      "--include-directories",
      action="append",
      help=(
          "Directory prefix to include (can be repeated or comma-separated,"
          " default: src,bench,test,models,eval,tools,litert)"
      ),
  )
  parser.add_argument(
      "--pedantic", action="store_true", help="Treat <unset> as distinct from 0"
  )
  parser.add_argument("--repo-root", type=Path, help="Explicit repository root")
  parser.add_argument(
      "--jsonl", action="store_true", help="Output in JSON Lines (JSONL) format"
  )

  args = parser.parse_args()
  p1, p2 = args.compile_commands_1, args.compile_commands_2
  repo_root = args.repo_root or find_repo_root(p1)
  include_re = re.compile(args.include_define)
  ignore_res = [re.compile(pat) for pat in args.ignore_define]
  raw_dirs = (
      [
          d.strip()
          for item in args.include_directories
          for d in item.split(",")
          if d.strip()
      ]
      if args.include_directories
      else DEFAULT_DIRS
  )
  include_dirs = tuple(
      d.strip("/") + "/" if d.strip("/.") else "" for d in raw_dirs
  )

  b1, b2 = p1.name, p2.name
  label1, label2 = (f"{b1} (first)", f"{b2} (second)") if b1 == b2 else (b1, b2)

  data1 = load_commands(p1, repo_root, include_re, ignore_res, include_dirs)
  data2 = load_commands(p2, repo_root, include_re, ignore_res, include_dirs)
  only1, only2, diffs, match_count = compare(data1, data2, args.pedantic)

  if args.jsonl:
    output_jsonl(
        label1,
        label2,
        p1,
        p2,
        only1,
        only2,
        diffs,
        match_count,
        len(data1),
        len(data2),
    )
  else:
    print_report(
        label1,
        label2,
        p1,
        p2,
        only1,
        only2,
        diffs,
        match_count,
        len(data1),
        len(data2),
    )

  sys.exit(EXIT_DIFF if (only1 or only2 or diffs) else EXIT_SUCCESS)


if __name__ == "__main__":
  main()
