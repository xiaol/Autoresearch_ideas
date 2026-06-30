"""Mathematical statement checker — identify equations, derivations, and claims."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class MathStatement:
    """A mathematical statement found in the paper.

    Attributes:
        kind: Type (equation, derivation, theorem, lemma, proof, assumption).
        environment: LaTeX environment (equation, align, theorem, proof, etc.).
        content: The mathematical content (truncated).
        label: The LaTeX label, if any.
        line_number: Line number in the source.
        source_file: Path to the source file.
    """

    kind: str
    environment: str
    content: str
    label: str
    line_number: int
    source_file: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "environment": self.environment,
            "content": self.content[:200],
            "label": self.label,
            "line_number": self.line_number,
            "source_file": self.source_file,
        }


@dataclass
class MathReport:
    """Complete math-check report.

    Attributes:
        paper_source: Path to the reviewed paper.
        statements: List of identified mathematical statements.
        equation_count: Number of display equations.
        theorem_count: Number of theorem-like environments.
        proof_count: Number of proof environments.
        unresolved_references: Equation labels referenced but not defined.
        summary: Short text summary.
    """

    paper_source: str
    statements: List[MathStatement] = field(default_factory=list)
    equation_count: int = 0
    theorem_count: int = 0
    proof_count: int = 0
    unresolved_references: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "paper_source": self.paper_source,
            "equation_count": self.equation_count,
            "theorem_count": self.theorem_count,
            "proof_count": self.proof_count,
            "statement_count": len(self.statements),
            "unresolved_references": self.unresolved_references,
            "statements": [s.to_dict() for s in self.statements],
            "summary": self.summary,
        }


# LaTeX environments considered mathematical
MATH_ENVIRONMENTS = [
    "equation", "equation*", "align", "align*", "multline", "multline*",
    "gather", "gather*", "split", "array",
]

THEOREM_ENVIRONMENTS = [
    "theorem", "lemma", "corollary", "proposition", "conjecture",
    "definition", "assumption", "hypothesis", "claim", "remark",
    "example", "note", "observation",
]

PROOF_ENVIRONMENTS = [
    "proof", "pf",
]

_INLINE_MATH = re.compile(r"\$[^$]+\$")
_LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
_REF_RE = re.compile(r"\\ref\{([^}]+)\}")


def check_math(source_path: Path, verbose: bool = False) -> MathReport:
    """Analyze mathematical content in a LaTeX paper.

    Args:
        source_path: Path to the .tex file.
        verbose: Enable verbose logging.

    Returns:
        MathReport with identified statements and statistics.
    """
    if not source_path.exists():
        return MathReport(
            paper_source=str(source_path),
            summary=f"File not found: {source_path}",
        )

    text = source_path.read_text(encoding="utf-8")
    lines = text.split("\n")
    report = MathReport(paper_source=str(source_path))

    # Collect all labels and references
    all_labels = set()
    for label_match in _LABEL_RE.finditer(text):
        all_labels.add(label_match.group(1))
    all_refs = set()
    for ref_match in _REF_RE.finditer(text):
        all_refs.add(ref_match.group(1))

    # Track theorem-like environments
    theorem_open = None

    for line_idx, line in enumerate(lines, start=1):
        stripped = line.strip()

        # Display math environments
        for env in MATH_ENVIRONMENTS:
            if f"\\begin{{{env}}}" in stripped:
                label = ""
                # Look ahead for label
                for j in range(line_idx, min(line_idx + 5, len(lines) + 1)):
                    label_match = _LABEL_RE.search(lines[j - 1])
                    if label_match:
                        label = label_match.group(1)
                        break
                report.statements.append(MathStatement(
                    kind="equation",
                    environment=env,
                    content=stripped[:200],
                    label=label,
                    line_number=line_idx,
                    source_file=str(source_path),
                ))
                report.equation_count += 1

        # Theorem-like environments
        for env in THEOREM_ENVIRONMENTS:
            if f"\\begin{{{env}}}" in stripped:
                theorem_open = env
                label = ""
                for j in range(line_idx, min(line_idx + 5, len(lines) + 1)):
                    label_match = _LABEL_RE.search(lines[j - 1])
                    if label_match:
                        label = label_match.group(1)
                        break
                report.statements.append(MathStatement(
                    kind=env,
                    environment=env,
                    content=stripped[:200],
                    label=label,
                    line_number=line_idx,
                    source_file=str(source_path),
                ))
                report.theorem_count += 1
            if f"\\end{{{env}}}" in stripped:
                theorem_open = None

        # Proof environments
        for env in PROOF_ENVIRONMENTS:
            if f"\\begin{{{env}}}" in stripped or f"\\begin{{{env}}}" in stripped:
                report.statements.append(MathStatement(
                    kind="proof",
                    environment=env,
                    content=stripped[:200],
                    label="",
                    line_number=line_idx,
                    source_file=str(source_path),
                ))
                report.proof_count += 1

    # Find equation labels that are referenced but not defined
    eq_labels = {
        s.label for s in report.statements
        if s.kind == "equation" and s.label
    }
    report.unresolved_references = sorted(
        ref for ref in all_refs
        if ref not in all_labels and not any(
            ref == lab for lab in eq_labels
        )
    )

    report.summary = (
        f"Found {report.equation_count} equations, {report.theorem_count} "
        f"theorem-like environments, {report.proof_count} proofs. "
        f"{len(report.unresolved_references)} potentially unresolved references."
    )

    return report
