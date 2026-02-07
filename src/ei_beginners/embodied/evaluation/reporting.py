"""Report output helpers."""

from __future__ import annotations

import json
from pathlib import Path

from ei_beginners.embodied.models import EvaluationReport


def save_report(report: EvaluationReport, report_dir: str | Path) -> tuple[Path, Path]:
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)

    json_path = report_path / f"{report.identifier}_report.json"
    md_path = report_path / f"{report.identifier}_report.md"

    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md_path.write_text(_to_markdown(report), encoding="utf-8")

    return json_path, md_path


def _to_markdown(report: EvaluationReport) -> str:
    metrics = report.metrics
    return "\n".join(
        [
            f"# Evaluation Report: {report.identifier}",
            "",
            "## Summary",
            report.summary,
            "",
            "## Metrics",
            f"- Executability: {metrics.executability}",
            f"- Goal Satisfaction: {metrics.goal_satisfaction}",
            f"- Partial Success: {metrics.partial_success}",
            f"- Success: {report.success}",
            "",
            "## Error Taxonomy",
            *(f"- {name}: {count}" for name, count in sorted(metrics.error_taxonomy.items())),
            "",
            "## Details",
            f"```json\n{json.dumps(report.details, indent=2, ensure_ascii=False)}\n```",
            "",
        ]
    )
