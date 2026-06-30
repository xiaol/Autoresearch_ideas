"""Core review orchestration — multi-pass inference scaling engine."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from paper_review.benchmark_auditor import audit_benchmarks
from paper_review.claim_extractor import extract_claims, extract_section_structure
from paper_review.config import ReviewConfig, ReviewLevel
from paper_review.experiment_validator import validate_claims
from paper_review.math_checker import check_math
from paper_review.report import ReviewReport


def run_review(
    config: ReviewConfig,
    verbose: bool = False,
) -> ReviewReport:
    """Run a multi-pass paper review at the configured level.

    Each level builds on the previous one:
    - Level 1: Surface scan (section structure, counts)
    - Level 2: Claim extraction + cross-referencing
    - Level 3: Deep audit (benchmark methodology)
    - Level 4: Full verification (math check + adversarial)

    Args:
        config: Review configuration (level, paths, etc.).
        verbose: Enable verbose logging.

    Returns:
        ReviewReport with all findings at the requested level.
    """
    paper_path = config.paper_path
    if paper_path is None:
        raise ValueError("paper_path must be set in the config")

    if not paper_path.exists():
        raise FileNotFoundError(f"Paper not found: {paper_path}")

    paper_path = paper_path.resolve()

    # Normalize results/papers roots relative to the repo root
    results_root = _resolve_relative_path(config.results_root, paper_path)
    papers_root = _resolve_relative_path(config.papers_root, paper_path)

    if verbose:
        _log(f"Reviewing: {paper_path}")
        _log(f"Level: {config.level} ({config.level.name})")
        _log(f"Results root: {results_root}")
        _log(f"Papers root: {papers_root}")

    report = ReviewReport(
        paper_source=str(paper_path),
        review_level=config.level,
        config={
            "level": config.level,
            "results_root": str(results_root),
            "papers_root": str(papers_root),
            "verbose": config.verbose,
            "claim_patterns": config.claim_patterns,
            "benchmark_patterns": config.benchmark_patterns,
        },
    )

    # --- Level 1: Surface scan ---
    if verbose:
        _log("Running Level 1: Surface scan...")
    section_info = extract_section_structure(paper_path)
    report.section_structure = section_info

    # --- Level 2: Claim extraction + validation ---
    if config.level >= ReviewLevel.CLAIM_CHECK:
        if verbose:
            _log("Running Level 2: Claim extraction & fact-checking...")

        claims = extract_claims(
            paper_path,
            extra_patterns=config.claim_patterns,
            verbose=verbose,
        )
        report.claims = [c.to_dict() for c in claims]

        validation = validate_claims(
            claims,
            results_root=results_root,
            verbose=verbose,
        )
        report.validation = validation.to_dict()

    # --- Level 3: Deep audit ---
    if config.level >= ReviewLevel.DEEP_AUDIT:
        if verbose:
            _log("Running Level 3: Deep audit...")

        audit = audit_benchmarks(
            paper_path,
            verbose=verbose,
        )
        report.audit = audit.to_dict()

    # --- Level 4: Full verification ---
    if config.level >= ReviewLevel.FULL_VERIFICATION:
        if verbose:
            _log("Running Level 4: Full verification...")

        math = check_math(
            paper_path,
            verbose=verbose,
        )
        report.math_report = math.to_dict()

    # --- Generate summary ---
    summary_parts: List[str] = []
    summary_parts.append(f"Level {config.level} review of {paper_path.name}.")

    if report.claims:
        summary_parts.append(
            f"Found {len(report.claims)} claims."
        )

    if report.validation:
        v = report.validation
        mismatches = v.get("mismatch_count", 0)
        validated = v.get("validated_claims", 0)
        total = v.get("total_claims", 0)
        summary_parts.append(
            f"Validated {validated}/{total} claims; "
            f"{mismatches} mismatches."
        )

    if report.audit:
        a = report.audit
        errors = sum(1 for f in a.get("findings", []) if f.get("severity") == "error")
        warnings = sum(1 for f in a.get("findings", []) if f.get("severity") == "warning")
        summary_parts.append(
            f"Benchmark audit: {errors} errors, {warnings} warnings."
        )

    if report.math_report:
        m = report.math_report
        summary_parts.append(
            f"Math: {m.get('equation_count', 0)} equations, "
            f"{m.get('theorem_count', 0)} theorems, "
            f"{m.get('proof_count', 0)} proofs."
        )

    report.summary = " ".join(summary_parts)

    return report


def discover_papers(papers_root: Path) -> List[Dict[str, object]]:
    """Discover all paper packages under a root directory.

    Returns:
        List of dicts with 'name', 'path', 'has_main_tex' keys.
    """
    papers_root = Path(papers_root)
    if not papers_root.exists():
        return []

    found: List[Dict[str, object]] = []
    for entry in sorted(papers_root.iterdir()):
        if entry.is_dir():
            tex_files = list(entry.rglob("*.tex"))
            found.append({
                "name": entry.name,
                "path": str(entry),
                "main_tex": str(tex_files[0]) if tex_files else None,
                "tex_file_count": len(tex_files),
            })
    return found


def _resolve_relative_path(path: Path, relative_to: Path) -> Path:
    """Resolve a path relative to a reference file's parent if not absolute."""
    if path.is_absolute():
        return path.resolve()
    # Try relative to the paper's location, then cwd
    candidates = [
        relative_to.parent / path,
        Path.cwd() / path,
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return candidates[0].resolve()


def _log(msg: str) -> None:
    """Simple logger."""
    print(f"[paper_review] {msg}")
