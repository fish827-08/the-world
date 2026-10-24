"""ExperimentResult 的持久化（模块四：experiment result persistence）。

输出约定（每个实验一个目录，名字 = 实验名）：
- manifest.json   实验元数据 + 完整配置 + 汇总（JSON）
- generations.csv 每代一行观测点（宽表，含 trait 分布/多样性/谱系列）
- generations.json 同数据 JSON 形式（与 CSV 互为镜像）

外加批量运行的 __survey__.json / __survey__.md 汇总。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from observatory.experiment import ExperimentResult


def _sample_rows(result: ExperimentResult) -> list[dict]:
    return [s.flatten() for s in result.samples]


def save_experiment(result: ExperimentResult, out_dir: str | Path) -> Path:
    """保存单个实验到目录，返回目录路径。"""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)

    manifest = {
        "name": result.run.spec.name,
        "group": result.run.spec.group,
        "description": result.run.spec.description,
        "seed": result.run.seed,
        "overrides": result.run.spec.overrides,
        "config": result.config.to_dict(),
        "finished_normally": result.finished_normally,
        "ended_reason": result.ended_reason,
        "extinct": result.extinct,
        "total_ticks": result.total_ticks,
        "final_population": result.final_population,
        "duration_s": round(result.duration_s, 2),
        "totals": result.totals,
        "summary": result.summary,
        "n_samples": len(result.samples),
    }
    (d / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    rows = _sample_rows(result)
    if rows:
        fieldnames = list(rows[0].keys())
        with (d / "generations.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        (d / "generations.json").write_text(
            json.dumps(rows, ensure_ascii=False), encoding="utf-8"
        )
    else:
        # 无样本（极端：从未触发采样）——留空文件并注明
        (d / "generations.csv").write_text(
            "# no samples\n", encoding="utf-8"
        )
        (d / "generations.json").write_text("[]", encoding="utf-8")
    return d


def load_manifest(out_dir: str | Path) -> dict:
    return json.loads((Path(out_dir) / "manifest.json").read_text(encoding="utf-8"))


def load_generations(out_dir: str | Path) -> list[dict]:
    text = (Path(out_dir) / "generations.csv").read_text(encoding="utf-8")
    if text.startswith("#"):
        return []
    with (Path(out_dir) / "generations.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_survey(results: dict[str, ExperimentResult], out_dir: str | Path) -> Path:
    """批量保存全部实验 + 汇总 json/md，返回根目录。"""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    for name, res in results.items():
        saved[name] = str(save_experiment(res, root / name))
    (root / "__survey__.json").write_text(
        json.dumps({"saved": saved}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return root


def survey_summary_markdown(results: dict[str, ExperimentResult]) -> str:
    """把批量结果展成一份人类可读的摘要（进 __survey__.md）。"""
    lines = ["# Evolution Observatory — Survey 摘要", ""]
    header = (
        "| 实验 | 组 | 世代达标 | 达成世代 | tick | 最终种群 | 灭绝 | 耗时(s) | 多样性 start→end |"
    )
    lines.append(header)
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name, r in results.items():
        sm = r.summary
        div = sm.get("diversity", {})
        gens = sm.get("generations_reached", 0)
        status = "是" if r.finished_normally else r.ended_reason
        lines.append(
            f"| {name} | {r.run.spec.group} | {status} | {gens} | "
            f"{r.total_ticks} | {r.final_population} | {'是' if r.extinct else ''} | "
            f"{r.duration_s:.0f} | {div.get('start', '-')} → {div.get('end', '-')} |"
        )
    lines.append("")
    lines.append("### 性状漂变（start → end，均值）")
    lines.append("")
    for name, r in results.items():
        drift = r.summary.get("trait_drift", {})
        if not drift:
            continue
        bits = ", ".join(
            f"{t}={v['start']:.3f}→{v['end']:.3f}" for t, v in drift.items()
        )
        lines.append(f"- **{name}**: {bits}")
    lines.append("")
    return "\n".join(lines)


def save_survey_markdown(results: dict[str, ExperimentResult], out_dir: str | Path) -> Path:
    p = Path(out_dir) / "__survey__.md"
    p.write_text(survey_summary_markdown(results), encoding="utf-8")
    return p