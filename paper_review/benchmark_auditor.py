"""Benchmark methodology audit — detect confounds and methodological issues.

Based on findings from docs/07_higher_order_benchmark_audit.md which uncovered
a missing-task-identifier confound in the higher-order benchmark.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Pattern


@dataclass
class AuditFinding:
    """A single benchmark methodology finding.

    Attributes:
        kind: Type of finding (task_id, metric_def, baseline_fairness, etc.).
        description: Human-readable description.
        line_number: Line in the source where the issue was found.
        severity: 'error', 'warning', 'info', or 'suggestion'.
        recommendation: Suggested fix.
    """

    kind: str
    description: str
    line_number: int = 0
    severity: str = "info"
    recommendation: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "description": self.description,
            "line_number": self.line_number,
            "severity": self.severity,
            "recommendation": self.recommendation,
        }


@dataclass
class AuditReport:
    """Complete benchmark audit report.

    Attributes:
        paper_source: Path to the reviewed paper.
        findings: List of findings.
        summary: Short text summary.
    """

    paper_source: str
    findings: List[AuditFinding] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "paper_source": self.paper_source,
            "finding_count": len(self.findings),
            "findings": [f.to_dict() for f in self.findings],
            "summary": self.summary,
        }


# Checks to run on a LaTeX paper source

TASK_ID_PATTERN = re.compile(
    r"(?i)(?:task|dataset|benchmark)\s*(?:identifier|token|id|label|embedding|prefix)"
)

METRIC_DEFINITION_PATTERN = re.compile(
    r"(?i)(?:accuracy|precision|recall|f1\s*score|perplexity|bpb|bits.per.byte|loss)"
)

BASELINE_FAIRNESS_PATTERNS: List[Tuple[str, Pattern[str], str]] = [
    ("matched_params", re.compile(r"(?i)(?:matched|comparable|same|equal)\s*(?:parameter|size|count|capacity)"),
     "Check: are baselines compared at matched parameter counts?"),
    ("matched_compute", re.compile(r"(?i)(?:matched|comparable|same|equal)\s*(?:compute|flops|training\s*time|epoch)"),
     "Check: are baselines compared at matched compute budgets?"),
    ("multiple_seeds", re.compile(r"(?i)(?:seed|trial|run|repetition)\s*(?:[0-9]|multiple|\d)"),
     "Check: are results reported across multiple seeds?"),
    ("significance", re.compile(r"(?i)(?:significant|confidence\s*interval|std|standard\s*deviation|variance|p[\s-]value)"),
     "Check: is statistical significance reported?"),
]

EVAL_PROTOCOL_PATTERNS: List[Tuple[str, Pattern[str], str]] = [
    ("split_defined", re.compile(r"(?i)(?:train|validation|test|dev|eval|held.out)\s*(?:split|set|data)"),
     "Check: are dataset splits clearly defined?"),
    ("metric_defined", re.compile(r"(?i)(?:metric|measure|evaluate|score)\s*(?:is|was|defined)"),
     "Check: are evaluation metrics defined before results are presented?"),
    ("implementation_baseline", re.compile(r"(?i)(?:our\s*implementation|our\s*reproduction|reimplemented|reproduced|own\s*implementation)"),
     "Check: are baseline results from the same implementation pipeline?"),
]

CONFOUND_PATTERNS: List[Tuple[str, Pattern[str], str]] = [
    ("no_task_id", re.compile(
        r"(?i)(?:mixed|combined|joint|multi.task)\s*(?:training|curriculum|task)"
    ), "Check: when mixing tasks, are task identifiers present in the input?"),
    ("cherry_picking", re.compile(
        r"(?i)(?:best\s*of|selected|we\s*report\s*the\s*best|tuned\s*separately)"
    ), "Check: are results cherry-picked or from a single best run?"),
    ("data_leakage", re.compile(
        r"(?i)(?:pre.train|pre.trained|trained\s*on\s*test|test\s*set\s*appears)"
    ), "Check: is there potential data leakage?"),
]


def audit_benchmarks(source_path: Path, verbose: bool = False) -> AuditReport:
    """Run benchmark methodology checks on a LaTeX paper.

    Args:
        source_path: Path to the .tex file.
        verbose: Enable verbose logging.

    Returns:
        AuditReport with findings.
    """
    if not source_path.exists():
        return AuditReport(
            paper_source=str(source_path),
            findings=[AuditFinding("error", f"File not found: {source_path}", severity="error")],
            summary="Paper source not found.",
        )

    text = source_path.read_text(encoding="utf-8")
    lines = text.split("\n")
    report = AuditReport(paper_source=str(source_path))

    # Check 1: Task identifiers in multi-task benchmarks
    if CONFOUND_PATTERNS[0][1].search(text):
        task_id_definitions = TASK_ID_PATTERN.findall(text)
        if not task_id_definitions:
            report.findings.append(AuditFinding(
                kind="task_id",
                description=(
                    "Paper mentions multi-task training but no explicit task "
                    "identifier/token pattern found. Missing task IDs was the "
                    "key confound discovered in the higher-order benchmark audit "
                    "(see docs/07_higher_order_benchmark_audit.md)."
                ),
                severity="error",
                recommendation=(
                    "Add explicit task identifier tokens to benchmark inputs "
                    "when mixing tasks during training or evaluation."
                ),
            ))
        else:
            report.findings.append(AuditFinding(
                kind="task_id",
                description="Multi-task benchmark with task identifiers found — good.",
                severity="info",
            ))

    # Check 2: Metric definitions
    metrics_found = METRIC_DEFINITION_PATTERN.findall(text)
    if not metrics_found:
        report.findings.append(AuditFinding(
            kind="metric_def",
            description="No metric keywords (accuracy, perplexity, BPB, etc.) found.",
            severity="warning",
            recommendation="Define all evaluation metrics explicitly.",
        ))
    else:
        unique_metrics = set(m.lower() for m in metrics_found)
        report.findings.append(AuditFinding(
            kind="metric_def",
            description=f"Metrics mentioned: {', '.join(sorted(unique_metrics))}.",
            severity="info",
        ))

    # Check 3: Baseline fairness
    for check_id, pattern, description in BASELINE_FAIRNESS_PATTERNS:
        if pattern.search(text):
            report.findings.append(AuditFinding(
                kind=check_id,
                description=f"{description} — keyword found.",
                severity="info",
            ))
        else:
            report.findings.append(AuditFinding(
                kind=check_id,
                description=f"{description} — no mention found.",
                severity="warning" if check_id in ("matched_params", "matched_compute") else "info",
                recommendation=(
                    "Report whether baseline comparisons are matched on parameters, "
                    "compute, or both."
                ) if check_id in ("matched_params", "matched_compute") else "",
            ))

    # Check 4: Evaluation protocol
    for check_id, pattern, description in EVAL_PROTOCOL_PATTERNS:
        if pattern.search(text):
            report.findings.append(AuditFinding(
                kind=check_id,
                description=f"{description} — found.",
                severity="info",
            ))
        else:
            report.findings.append(AuditFinding(
                kind=check_id,
                description=f"{description} — no mention.",
                severity="info",
            ))

    # Check 5: Other confounds
    for check_id, pattern, description in CONFOUND_PATTERNS[1:]:
        if pattern.search(text):
            report.findings.append(AuditFinding(
                kind=check_id,
                description=f"{description} — keyword pattern found.",
                severity="warning",
            ))

    # Generate summary
    errors = sum(1 for f in report.findings if f.severity == "error")
    warnings = sum(1 for f in report.findings if f.severity == "warning")
    report.summary = (
        f"Benchmark audit complete: {errors} errors, {warnings} warnings, "
        f"{len(report.findings)} total findings."
    )

    return report
