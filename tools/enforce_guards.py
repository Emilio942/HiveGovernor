#!/usr/bin/env python3
"""
Guard Enforcer (T02)

Checks a unified diff against global rules:
- GR-1: Max changed lines (default 200) and max changed files (default 3)
- GR-2: Changes must be within guarded regions delimited by:
  >>> BEGIN:AI_EDIT
  >>> END:AI_EDIT

Usage:
  - Read diff from stdin:    git diff | python tools/enforce_guards.py
  - From a file:             python tools/enforce_guards.py --diff-file path.diff
  - Custom limits:           python tools/enforce_guards.py --max-lines 120 --max-files 2

The script exits non‑zero on violations and prints a short summary.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

BEGIN_MARK = ">>> BEGIN:AI_EDIT"
END_MARK = ">>> END:AI_EDIT"


@dataclass
class EnforceResult:
    ok: bool
    changed_files: int
    changed_lines: int
    errors: List[str]


def _parse_unified_diff(diff_text: str) -> Dict[str, List[List[str]]]:
    """Parse unified diff into mapping file->list of hunks (each hunk as list of lines).

    A very lightweight parser sufficient for enforcement checks.
    """
    files: Dict[str, List[List[str]]] = {}
    current_file: Optional[str] = None
    current_hunk: Optional[List[str]] = None

    lines = diff_text.splitlines()
    for line in lines:
        if line.startswith("+++ "):
            # format: +++ b/path
            path = line[4:].strip()
            if path.startswith("b/"):
                path = path[2:]
            current_file = path
            files.setdefault(current_file, [])
            current_hunk = None
        elif line.startswith("@@ ") and current_file is not None:
            # start of a new hunk
            current_hunk = []
            files[current_file].append(current_hunk)
        elif current_hunk is not None and (line.startswith(" ") or line.startswith("+") or line.startswith("-")):
            current_hunk.append(line)
        # ignore other headers like diff --git, --- a/...
    return files


def check_diff_text(diff_text: str, max_lines: int = 200, max_files: int = 3) -> EnforceResult:
    files = _parse_unified_diff(diff_text)

    changed_files = 0
    changed_lines = 0
    errors: List[str] = []

    for path, hunks in files.items():
        file_has_change = False
        for hunk in hunks:
            in_guard = False
            for raw in hunk:
                if not raw:
                    continue
                prefix, content = raw[0], raw[1:]
                text = content.strip()

                # Track guard region state off of context or changed lines
                if BEGIN_MARK in text:
                    in_guard = True
                    continue
                if END_MARK in text:
                    in_guard = False
                    continue

                if prefix == "+":
                    # Count only non-marker additions
                    if text not in (BEGIN_MARK, END_MARK):
                        changed_lines += 1
                        file_has_change = True
                        if not in_guard:
                            errors.append(f"Change outside guard: {path} -> '{content.strip()[:60]}'")
                elif prefix == "-":
                    # Deletions count toward size gate; allow outside-guard detection via context markers
                    if text not in (BEGIN_MARK, END_MARK):
                        changed_lines += 1
                        file_has_change = True
                        if not in_guard:
                            # Heuristic: if we are not within a guard by context, flag deletion as well
                            errors.append(f"Deletion outside guard: {path} -> '{content.strip()[:60]}'")
                else:
                    # context line: update guard state already handled above
                    pass
        if file_has_change:
            changed_files += 1

    # Size gates
    if changed_files > max_files:
        errors.append(f"Too many files changed: {changed_files} > {max_files}")
    if changed_lines > max_lines:
        errors.append(f"Too many lines changed: {changed_lines} > {max_lines}")

    ok = len(errors) == 0
    return EnforceResult(ok=ok, changed_files=changed_files, changed_lines=changed_lines, errors=errors)


def _read_diff_from_args(args: argparse.Namespace) -> str:
    if args.diff_file:
        return open(args.diff_file, "r", encoding="utf-8").read()
    # read from stdin
    if not sys.stdin.isatty():
        return sys.stdin.read()
    # Fall back to empty
    return ""


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Guard Enforcer for AI_EDIT regions and size gates")
    parser.add_argument("--diff-file", help="Path to a unified diff file", default=None)
    parser.add_argument("--max-lines", type=int, default=200)
    parser.add_argument("--max-files", type=int, default=3)
    args = parser.parse_args(argv)

    diff_text = _read_diff_from_args(args)
    res = check_diff_text(diff_text, max_lines=args.max_lines, max_files=args.max_files)

    if res.ok:
        print(f"OK: {res.changed_files} file(s), {res.changed_lines} line(s) within policy.")
        return 0
    print("FAIL:")
    for e in res.errors:
        print(f" - {e}")
    print(f"Summary: {res.changed_files} file(s), {res.changed_lines} line(s)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
