"""Review level configuration — maps PAT's 4-level taxonomy to review passes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import List, Optional


class ReviewLevel(IntEnum):
    """Progressive review levels matching PAT's taxonomy of AI-human collaboration.

    Level 1 — AI as formatting checker (surface scan).
    Level 2 — AI-assisted fact-checking (claim extraction + validation).
    Level 3 — AI-augmented review (deep audit, methodology checking).
    Level 4 — AI-led verification (adversarial testing, full re-run).
    """

    SURFACE_SCAN = 1
    CLAIM_CHECK = 2
    DEEP_AUDIT = 3
    FULL_VERIFICATION = 4


LEVEL_NAMES = {
    ReviewLevel.SURFACE_SCAN: "Surface Scan",
    ReviewLevel.CLAIM_CHECK: "Claim Extraction & Fact-Checking",
    ReviewLevel.DEEP_AUDIT: "Deep Audit & Methodology Review",
    ReviewLevel.FULL_VERIFICATION: "Full Verification & Adversarial Testing",
}

LEVEL_DESCRIPTIONS = {
    ReviewLevel.SURFACE_SCAN: (
        "Check paper structure, section completeness, equation/table/figure references. "
        "Analogous to PAT Level 1 — AI as formatting checker."
    ),
    ReviewLevel.CLAIM_CHECK: (
        "Extract all numerical claims, performance comparisons, and benchmark results "
        "from the LaTeX source. Cross-reference against experiment outputs. "
        "Analogous to PAT Level 2 — AI-assisted fact-checking."
    ),
    ReviewLevel.DEEP_AUDIT: (
        "Audit benchmark methodology, check for confounds (task identifiers, metric "
        "definitions, fairness of comparisons), verify statistical validity. "
        "Analogous to PAT Level 3 — AI-augmented review."
    ),
    ReviewLevel.FULL_VERIFICATION: (
        "Adversarial claim testing, re-run key experiments, comprehensive verification "
        "of all results. Analogous to PAT Level 4 — AI-led verification."
    ),
}


@dataclass
class ReviewConfig:
    """Configuration for a paper review run.

    Attributes:
        level: Review depth level (1-4).
        paper_path: Path to the main LaTeX file to review.
        results_root: Root directory for experiment results (for cross-referencing).
        papers_root: Root directory for paper packages.
        output_path: Where to write the review report (JSON).
        verbose: Enable verbose logging.
        claim_patterns: Custom claim regex patterns (merged with defaults).
        benchmark_patterns: Custom benchmark audit patterns (merged with defaults).
    """

    level: ReviewLevel = ReviewLevel.SURFACE_SCAN
    paper_path: Optional[Path] = None
    results_root: Path = Path("results")
    papers_root: Path = Path("papers")
    output_path: Optional[Path] = None
    verbose: bool = False
    claim_patterns: List[str] = field(default_factory=list)
    benchmark_patterns: List[str] = field(default_factory=list)

    @property
    def resolved_output_path(self) -> Path:
        if self.output_path is not None:
            return self.output_path
        stem = "paper_review_report"
        if self.paper_path is not None:
            stem = f"{self.paper_path.stem}_review"
        return Path(f"{stem}.json")

    def run_all_checks(self) -> bool:
        """Whether to run all checks from Level 1 up to the configured level."""
        return True


def load_config(path: Optional[Path] = None, **overrides) -> ReviewConfig:
    """Load a review config, optionally from a JSON file with overrides."""
    cfg = ReviewConfig()
    if path is not None and path.exists():
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        for key, value in data.items():
            if hasattr(cfg, key):
                if key == "level":
                    setattr(cfg, key, ReviewLevel(value))
                else:
                    setattr(cfg, key, value)
    for key, value in overrides.items():
        if hasattr(cfg, key) and value is not None:
            if key == "level":
                setattr(cfg, key, ReviewLevel(value))
            else:
                setattr(cfg, key, value)
    return cfg
