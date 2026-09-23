"""Branch tests for the Phase 8 holdout gate (``prereg_approved``).

Calls the gate function directly on temporary copies of the
pre-registration. It NEVER invokes the evaluation (``main``); importing the
module runs no evaluation code.

The DRAFT case uses the exact Status line that was in effect when the 2025
run executed before approval (see docs/PHASE8_PREREG.md, Section 7). That is
the regression test for the substring bug.

Usage::

    python -m scripts.test_phase8_gate
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from scripts.phase8_holdout_2025 import APPROVED_MARKER, PREREG, prereg_approved

DRAFT_LINE = ("**Status:** DRAFT — awaiting project-owner approval. "
              "`scripts/phase8_holdout_2025.py --season 2025` refuses to run until this line "
              "reads `**Status:** APPROVED`.")


def with_status(body: str, status_line: str) -> str:
    lines = body.splitlines()
    idx = next(i for i, ln in enumerate(lines) if ln.startswith("**Status:**"))
    lines[idx] = status_line
    return "\n".join(lines) + "\n"


def main() -> int:
    body = PREREG.read_text()
    cases = [
        # (name, file content or None for "missing", expected)
        ("file missing", None, False),
        ("DRAFT (exact line in effect at the 13:54:45 run)", with_status(body, DRAFT_LINE), False),
        ("APPROVED (exact marker line)", with_status(body, APPROVED_MARKER), True),
        ("APPROVED with surrounding whitespace", with_status(body, f"  {APPROVED_MARKER}  "), True),
        ("current file: APPROVED (retroactive -- ...)", body, False),
        ("marker only inside another sentence", with_status(body, f"Do not set {APPROVED_MARKER} yet."), False),
    ]
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for i, (name, content, expected) in enumerate(cases):
            path = Path(tmp) / f"prereg_{i}.md"
            if content is not None:
                path.write_text(content)
            got = prereg_approved(path)
            old = path.exists() and APPROVED_MARKER in path.read_text()  # the buggy substring check
            passed = got == expected
            ok &= passed
            print(f"{'PASS' if passed else 'FAIL'}  {name:<52} gate={got!s:<5} expected={expected!s:<5} "
                  f"(old substring check would have said {old})")
    print("\nall gate branches behave as specified" if ok else "\n!! gate test FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
