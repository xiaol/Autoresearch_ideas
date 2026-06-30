"""Extract numerical claims, comparisons, and results from LaTeX paper sources."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Tuple


@dataclass
class Claim:
    """A single extracted claim from a paper.

    Attributes:
        kind: Type of claim (numerical, comparison, benchmark, parameter, math).
        text: The matched sentence or fragment containing the claim.
        line_number: Line number in the source file.
        source_file: Path to the source file.
        value: Extracted numeric value(s), if applicable.
        context: Surrounding context (lines before/after).
        confidence: Heuristic confidence score (0.0-1.0).
    """

    kind: str
    text: str
    line_number: int
    source_file: str
    value: Optional[str] = None
    context: str = ""
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "text": self.text.strip(),
            "line_number": self.line_number,
            "source_file": self.source_file,
            "value": self.value,
            "context": self.context.strip(),
            "confidence": self.confidence,
        }


# --- Claim detection patterns ---

NUMERICAL_PATTERNS: List[Tuple[str, str]] = [
    # "achieves X" / "reaches X" / "obtains X"
    (r"(?i)(achieves?|reaches?|obtains?|gets?|yields?|attains?)\s+([\d]+\.[\d]+|[\d]+)", "numerical"),
    # "X% improvement" / "X percent"
    (r"(?i)([\d]+\.[\d]+|[\d]+)\s*[%%]\s*(improvement|better|reduction|gain|lift)", "numerical"),
    # "accuracy A" / "perplexity P" / "BPB B" / "score S"
    (r"(?i)(accuracy|perplexity|bpb|score|precision|recall|f1|bleu|rouge)\s+((?:of\s+)?[\d]+\.[\d]+|[\d]+)", "numerical"),
    # "X vs Y" / "X versus Y" comparisons
    (r"(?i)([\d]+\.[\d]+|[\d]+)\s*(?:vs\.?|versus|compared to|vs)\s*([\d]+\.[\d]+|[\d]+)", "comparison"),
    # "improves by Z%" / "improves by Z points"
    (r"(?i)improves?\s+by\s+([\d]+\.[\d]+|[\d]+)\s*(?:[%%]|points?)", "numerical"),
    # Parameter counts: "X parameters" / "X params" / "X B parameters"
    (r"(?i)([\d,]+(?:\.\d+)?)\s*(?:[MBKmbk]?\s*)?(?:parameters?|params?)", "parameter"),
    # Benchmark scores: "achieved X on Y"
    (r"(?i)([\d]+\.[\d]+|[\d]+)[%%]?\s+(?:on|for)\s+([a-z0-9_-]+)", "benchmark"),
]

SECTION_PATTERNS: List[Tuple[str, Pattern[str]]] = [
    ("method", re.compile(r"\\section\*?\{(?:Method|Approach|Model|Architecture)\}", re.IGNORECASE)),
    ("results", re.compile(r"\\section\*?\{(?:Results|Experiments|Evaluation|Benchmarks)\}", re.IGNORECASE)),
    ("conclusion", re.compile(r"\\section\*?\{(?:Conclusion|Discussion|Summary)\}", re.IGNORECASE)),
    ("introduction", re.compile(r"\\section\*?\{(?:Introduction)\}", re.IGNORECASE)),
]

EQUATION_RE = re.compile(r"\\begin\{(equation|align|multline|gather)\*?\}")
TABLE_RE = re.compile(r"\\begin\{table\*?\}")
FIGURE_RE = re.compile(r"\\begin\{figure\*?\}")
CITE_RE = re.compile(r"\\cite\{[^}]+\}")
REF_RE = re.compile(r"\\ref\{[^}]+\}")
LABEL_RE = re.compile(r"\\label\{[^}]+\}")


def extract_claims(
    source_path: Path,
    extra_patterns: Optional[List[str]] = None,
    verbose: bool = False,
) -> List[Claim]:
    """Extract claims from a LaTeX file.

    Args:
        source_path: Path to the .tex file.
        extra_patterns: Additional regex patterns to match.
        verbose: If True, print progress.

    Returns:
        List of extracted Claim objects.
    """
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    lines = source_path.read_text(encoding="utf-8").split("\n")
    claims: List[Claim] = []

    # Compile patterns
    patterns: List[Tuple[Pattern[str], str]] = [
        (re.compile(pat), kind) for pat, kind in NUMERICAL_PATTERNS
    ]
    if extra_patterns:
        for pat in extra_patterns:
            patterns.append((re.compile(pat), "custom"))

    current_section = "preamble"
    for line_idx, line in enumerate(lines, start=1):
        stripped = line.strip()

        # Track section boundaries
        for sec_name, sec_re in SECTION_PATTERNS:
            if sec_re.search(stripped):
                current_section = sec_name
                break

        # Skip comments and empty lines
        if stripped.startswith("%") or not stripped:
            continue

        # Check each pattern
        for compiled, kind in patterns:
            for match in compiled.finditer(stripped):
                context_start = max(0, line_idx - 3)
                context_end = min(len(lines), line_idx + 2)
                context_lines = lines[context_start:context_end]

                claim = Claim(
                    kind=kind,
                    text=stripped[:200],
                    line_number=line_idx,
                    source_file=str(source_path),
                    value=match.group(0)[:100],
                    context="\n".join(context_lines),
                    confidence=_estimate_confidence(kind, current_section),
                )
                claims.append(claim)

    return _deduplicate(claims)


def _estimate_confidence(kind: str, section: str) -> float:
    """Estimate confidence based on claim type and where it appears."""
    base = 0.5
    if kind == "comparison":
        base = 0.6
    elif kind == "parameter":
        base = 0.7
    elif kind == "benchmark":
        base = 0.6
    if section == "results":
        base = min(1.0, base + 0.2)
    elif section in ("method", "introduction"):
        base = min(1.0, base + 0.1)
    elif section == "conclusion":
        base = max(0.1, base - 0.1)
    return round(base, 2)


def _deduplicate(claims: List[Claim]) -> List[Claim]:
    """Remove near-duplicate claims on the same line."""
    seen: set[Tuple[int, str, str]] = set()
    deduped: List[Claim] = []
    for claim in claims:
        key = (claim.line_number, claim.kind, claim.value or "")
        if key not in seen:
            seen.add(key)
            deduped.append(claim)
    return deduped


def find_tex_files(paper_dir: Path) -> List[Path]:
    """Find the main .tex file(s) in a paper directory."""
    tex_files = sorted(paper_dir.rglob("*.tex"))
    # Prefer main.tex, then any top-level .tex, then any .tex
    main = [f for f in tex_files if f.name == "main.tex"]
    if main:
        return main[:1]
    top_level = [f for f in tex_files if f.parent == paper_dir]
    if top_level:
        return top_level[:1]
    return tex_files[:1]


def extract_section_structure(source_path: Path) -> Dict[str, object]:
    """Extract the section/equation/figure/table structure of a paper."""
    if not source_path.exists():
        return {"error": "File not found", "path": str(source_path)}

    text = source_path.read_text(encoding="utf-8")

    sections = []
    for sec_name, sec_re in SECTION_PATTERNS:
        matches = sec_re.findall(text)
        if matches:
            sections.append(sec_name)

    return {
        "source_file": str(source_path),
        "sections_found": sections,
        "equation_environments": len(EQUATION_RE.findall(text)),
        "table_environments": len(TABLE_RE.findall(text)),
        "figure_environments": len(FIGURE_RE.findall(text)),
        "citation_count": len(CITE_RE.findall(text)),
        "reference_count": len(REF_RE.findall(text)),
    }
