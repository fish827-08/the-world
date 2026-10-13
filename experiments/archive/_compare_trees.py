"""Compare two trees: file inventory, size, quick hash equality."""
import hashlib
import os
import sys
from pathlib import Path

MAIN = Path(r"C:\Users\圣羽\Desktop\temp\tempCode\the-world\digital-life-sphere")
REV = Path(r"C:\Users\圣羽\Desktop\temp\tempCode\the-world\_review")
EXCLUDE = {".venv", "target", "__pycache__", ".git", "results"}


def collect(root: Path):
    out = {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in EXCLUDE for part in p.parts):
            continue
        rel = p.relative_to(root).as_posix()
        data = p.read_bytes()
        out[rel] = (len(data), hashlib.md5(data).hexdigest())
    return out


def main():
    m, r = collect(MAIN), collect(REV)
    rows = []
    for rel in sorted(set(m) | set(r)):
        if rel not in m:
            rows.append((rel, -1, r[rel][0], "ONLY-REV"))
        elif rel not in r:
            rows.append((rel, m[rel][0], -1, "ONLY-MAIN"))
        elif m[rel] == r[rel]:
            rows.append((rel, m[rel][0], r[rel][0], "same"))
        else:
            rows.append((rel, m[rel][0], r[rel][0], "CHANGED"))
    print(f"{'file':48s} {'main':>8s} {'rev':>8s}  state")
    print("-" * 80)
    for rel, ms, rs, st in rows:
        if st != "same":
            print(f"{rel:48s} {ms:>8d} {rs:>8d}  {st}")
    n_same = sum(1 for r in rows if r[3] == "same")
    print(f"\ntotal files: {len(rows)} | identical: {n_same} | "
          f"changed: {sum(1 for r in rows if r[3]=='CHANGED')} | "
          f"only-rev: {sum(1 for r in rows if r[3]=='ONLY-REV')} | "
          f"only-main: {sum(1 for r in rows if r[3]=='ONLY-MAIN')}")


if __name__ == "__main__":
    sys.exit(main())