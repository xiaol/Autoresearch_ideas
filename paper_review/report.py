"""Structured review report generation (JSON + Markdown)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from paper_review.benchmark_auditor import AuditFinding, AuditReport
from paper_review.claim_extractor import Claim
from paper_review.config import ReviewConfig, ReviewLevel, LEVEL_NAMES
from paper_review.experiment_validator import Mismatch, ValidationResult
from paper_review.math_checker import MathReport, MathStatement


@dataclass
class ReviewReport:
    """Complete review report for a paper.

    Attributes:
        paper_source: Path to the reviewed paper.
        review_level: The level at which review was run.
        timestamp: ISO-format timestamp.
        config: The configuration used.
        section_structure: Section structure info.
        claims: Extracted claims.
        validation: Claim vs experiment cross-reference result.
        audit: Benchmark audit result.
        math_report: Math checker report.
        summary: Human-readable summary.
    """

    paper_source: str
    review_level: int
    timestamp: str = ""
    config: Dict[str, object] = field(default_factory=dict)
    section_structure: Dict[str, object] = field(default_factory=dict)
    claims: List[Dict[str, object]] = field(default_factory=list)
    validation: Optional[Dict[str, object]] = None
    audit: Optional[Dict[str, object]] = None
    math_report: Optional[Dict[str, object]] = None
    summary: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "paper_source": self.paper_source,
            "review_level": self.review_level,
            "level_name": LEVEL_NAMES.get(ReviewLevel(self.review_level), "Unknown"),
            "timestamp": self.timestamp or datetime.utcnow().isoformat(),
            "config": self.config,
            "section_structure": self.section_structure,
            "claims": self.claims,
            "validation": self.validation,
            "audit": self.audit,
            "math_report": self.math_report,
            "summary": self.summary,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_markdown(self) -> str:
        """Render the report as human-readable Markdown."""
        lines: List[str] = []
        info = self.to_dict()

        lines.append(f"# Paper Review Report")
        lines.append("")
        lines.append(f"- **Paper:** `{self.paper_source}`")
        lines.append(f"- **Level:** {info['level_name']} (Level {self.review_level})")
        lines.append(f"- **Timestamp:** {self.timestamp or datetime.utcnow().isoformat()}")
        lines.append("")

        # Section structure
        if self.section_structure:
            lines.append("## Section Structure")
            lines.append("")
            ss = self.section_structure
            lines.append(f"- Sections found: {', '.join(ss.get('sections_found', []))}")
            lines.append(f"- Equations: {ss.get('equation_environments', 0)}")
            lines.append(f"- Tables: {ss.get('table_environments', 0)}")
            lines.append(f"- Figures: {ss.get('figure_environments', 0)}")
            lines.append(f"- Citations: {ss.get('citation_count', 0)}")
            lines.append("")

        # Claims
        if self.claims:
            lines.append(f"## Claims Found ({len(self.claims)})")
            lines.append("")
            for i, claim in enumerate(self.claims, 1):
                lines.append(f"{i}. **Line {claim['line_number']}"
                             f"** [{claim['kind']}] `{claim.get('value', '')}`")
                lines.append(f"   _{claim['text'][:100]}..._")
            lines.append("")

        # Validation
        if self.validation:
            lines.append("## Claim Validation")
            lines.append("")
            lines.append(f"- Validated: {self.validation.get('validated_claims', 0)}")
            lines.append(f"- Total: {self.validation.get('total_claims', 0)}")
            lines.append(f"- Mismatches: {self.validation.get('mismatch_count', 0)}")
            if self.validation.get("mismatches"):
                for m in self.validation["mismatches"]:
                    lines.append(f"  - ⚠ Line {m['claim_line']}: "
                                 f"expected `{m['expected_value']}`, "
                                 f"found `{m['found_value']}`")
            lines.append("")

        # Audit
        if self.audit:
            lines.append("## Benchmark Audit")
            lines.append("")
            lines.append(f"- {self.audit.get('summary', '')}")
            if self.audit.get("findings"):
                for finding in self.audit["findings"]:
                    icon = {"error": "❌", "warning": "⚠", "info": "ℹ",
                            "suggestion": "💡"}.get(finding["severity"], "•")
                    lines.append(f"  - {icon} {finding['description']}")
            lines.append("")

        # Math
        if self.math_report:
            lines.append("## Mathematical Content")
            lines.append("")
            lines.append(f"- {self.math_report.get('summary', '')}")
            if self.math_report.get("unresolved_references"):
                lines.append(f"- Unresolved refs: {', '.join(self.math_report['unresolved_references'][:10])}")
            lines.append("")

        # Summary
        if self.summary:
            lines.append("## Summary")
            lines.append("")
            lines.append(self.summary)
            lines.append("")

        return "\n".join(lines)


def write_report(report: ReviewReport, output_path: Path, fmt: str = "json") -> Path:
    """Write a review report to disk.

    Args:
        report: The report to write.
        output_path: Where to write.
        fmt: 'json' or 'md'.

    Returns:
        The path the report was written to.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "md":
        output_path.write_text(report.to_markdown(), encoding="utf-8")
    else:
        output_path.write_text(report.to_json(), encoding="utf-8")

    return output_path
