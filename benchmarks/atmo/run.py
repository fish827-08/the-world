"""AV benchmark 运行器。用法（在仓库根/worktree 根）：

  python -m benchmarks.atmo.run --suite held_suarez --run-dir <run_dir> [--run-dir-b <run2>] [--out <json>]
  python -m benchmarks.atmo.audit atmo_core atmo
退出码：判定 PASS=0；FAIL/INCOMPLETE=1（含格式/输入契约错误）。
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from benchmarks.atmo import suites


def _fmt(x):
    if isinstance(x, float):
        return f"{x:.6g}"
    if isinstance(x, dict):
        return json.dumps(x, ensure_ascii=False)
    return str(x)


def print_report(report):
    print(f"suite={report['suite']}  verdict={report['verdict']}  "
          f"snaps={report['n_snaps']}  days={report['days']}")
    print(f"{'id':<20}{'status':<10}{'measured':<34}{'limits':<24}{'unit':<12}mandatory")
    for r in report["rows"]:
        lim = f"[{r['limit_lo']}, {r['limit_hi']}]" \
            if r["limit_lo"] is not None or r["limit_hi"] is not None else "-"
        print(f"{r['id']:<20}{r['status']:<10}{_fmt(r['measured']):<34}{lim:<24}"
              f"{r['unit']:<12}{'Y' if r['mandatory'] else 'N'}"
              + (f"  # {r['note']}" if r["note"] else ""))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="av-bench")
    ap.add_argument("--suite", required=True, choices=sorted(suites.SUITES))
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--run-dir-b", default=None,
                    help="同配置同 seed 第二次运行目录（CONS-DET 判定需要）")
    ap.add_argument("--out", default=None, help="JSON 报告输出路径（默认 _rerun_logs/）")
    a = ap.parse_args(argv)
    try:
        report = suites.evaluate(suites.SUITES[a.suite], a.run_dir, a.run_dir_b)
    except ValueError as e:
        print(f"INPUT_CONTRACT_ERROR: {e}")
        return 1
    print_report(report)
    out = Path(a.out) if a.out else Path("_rerun_logs") / (
        f"av-{a.suite}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"report -> {out}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
