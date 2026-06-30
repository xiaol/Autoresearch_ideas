"""Cross-reference paper claims against actual experiment results.

Reads experiment result files from the results/ directory and compares
extracted claims against the measured numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from paper_review.claim_extractor import Claim


@dataclass
class Mismatch:
    """A discrepancy between a paper claim and an experiment result.

    Attributes:
        claim: The extracted claim from the paper.
        expected_value: The value stated in the paper.
        found_value: The actual value from experiment results.
        result_source: Path to the result file.
        severity: 'error', 'warning', or 'info'.
        description: Human-readable explanation.
    """

    claim: Claim
    expected_value: str
    found_value: Optional[str]
    result_source: str
    severity: str = "warning"
    description: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "claim_text": self.claim.text,
            "claim_line": self.claim.line_number,
            "expected_value": self.expected_value,
            "found_value": self.found_value,
            "result_source": self.result_source,
            "severity": self.severity,
            "description": self.description,
        }


@dataclass
class ValidationResult:
    """Overall validation result for a paper.

    Attributes:
        paper_source: Path to the reviewed paper.
        mismatches: List of claim-vs-result mismatches found.
        validated_claims: Number of claims that matched.
        total_claims: Total number of claims checked.
        summary: Short text summary.
    """

    paper_source: str
    mismatches: List[Mismatch] = field(default_factory=list)
    validated_claims: int = 0
    total_claims: int = 0
    summary: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "paper_source": self.paper_source,
            "validated_claims": self.validated_claims,
            "total_claims": self.total_claims,
            "mismatch_count": len(self.mismatches),
            "mismatches": [m.to_dict() for m in self.mismatches],
            "summary": self.summary,
        }


def discover_result_files(results_root: Path) -> List[Path]:
    """Find all JSON result files under the results directory."""
    if not results_root.exists():
        return []
    return sorted(results_root.rglob("*.json"))


def read_result_summaries(results_root: Path) -> Dict[str, Any]:
    """Read all result summary files into a structured dict."""
    summaries: Dict[str, Any] = {}
    for json_path in discover_result_files(results_root):
        if "summary" not in json_path.stem and json_path.name != "summary.json":
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            # Key by relative path from results_root
            rel = json_path.relative_to(results_root)
            summaries[str(rel)] = data
        except (json.JSONDecodeError, ValueError):
            continue
    return summaries


def get_result_value(results: Dict[str, Any], metric: str) -> Optional[str]:
    """Extract a metric value from nested result dicts.

    Searches common keys like val_bpb, val_loss, val_perplexity, eval_accuracy,
    tokens_per_second, average_score.
    """
    if isinstance(results, dict):
        # Direct match
        if metric in results:
            return _format_value(results[metric])
        # Nested search
        for key, value in results.items():
            if isinstance(value, dict):
                found = get_result_value(value, metric)
                if found is not None:
                    return found
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                for item in value:
                    if isinstance(item, dict) and metric in item:
                        return _format_value(item[metric])
    return None


def _format_value(val: Any) -> Optional[str]:
    """Format a value for comparison."""
    if isinstance(val, (int, float)):
        return f"{val:.4f}" if isinstance(val, float) else str(val)
    return str(val) if val is not None else None


def _extract_numeric_value(text: str) -> Optional[str]:
    """Extract a numeric value string (first float/int) from text."""
    import re

    match = re.search(r"(\d+\.\d+|\d+)", text)
    return match.group(1) if match else None


def validate_claims(
    claims: List[Claim],
    results_root: Path,
    verbose: bool = False,
) -> ValidationResult:
    """Cross-reference a list of claims against experiment results.

    Args:
        claims: Claims extracted from a paper.
        results_root: Root of the results/ directory.
        verbose: Enable verbose logging.

    Returns:
        ValidationResult with mismatches.
    """
    result = ValidationResult(
        paper_source=claims[0].source_file if claims else "",
        total_claims=len(claims),
    )

    summaries = read_result_summaries(results_root)
    if not summaries:
        result.summary = (
            "No experiment result files found. Run experiments first "
            "to enable claim cross-referencing."
        )
        return result

    for claim in claims:
        numeric_value = _extract_numeric_value(claim.text)
        if numeric_value is None:
            continue

        # Try common metrics
        found = None
        for metric in ("val_bpb", "val_loss", "val_perplexity", "eval_accuracy",
                       "tokens_per_second", "average_score", "accuracy"):
            found = get_result_value(summaries, metric)
            if found is not None:
                break

        if found is not None and numeric_value != _extract_numeric_value(found):
            result.mismatches.append(
                Mismatch(
                    claim=claim,
                    expected_value=numeric_value,
                    found_value=found,
                    result_source=", ".join(summaries.keys()),
                    severity="warning",
                    description=(
                        f"Paper states '{numeric_value}' but experiment results "
                        f"show '{found}' in metric {metric}."
                    ),
                )
            )
        else:
            result.validated_claims += 1

    if result.mismatches:
        result.summary = (
            f"Found {len(result.mismatches)} mismatches "
            f"out of {result.total_claims} claims checked."
        )
    else:
        result.summary = (
            f"All {result.total_claims} claims matched experiment results "
            f"or could not be verified (no mismatches found)."
        )

    return result
