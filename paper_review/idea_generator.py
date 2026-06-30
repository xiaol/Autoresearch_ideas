"""Moonshine-style research idea generator from existing review outputs.

Generates structured research ideas by applying four strategies to the
Level 1-4 review results, following the paradigm from arXiv:2606.10806
where AI conjectures research directions from existing work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class IdeaConfig:
    """Configuration for research idea generation.

    Attributes:
        max_ideas: Maximum number of ideas to generate (default 5).
        idea_types: Categories of ideas to allow (default: all four strategies).
        creativity: Creativity / exploration factor (0.0 = conservative, 1.0 = speculative).
    """

    max_ideas: int = 5
    idea_types: List[str] = field(
        default_factory=lambda: ["gap", "methodology", "domain_translation", "ablation"]
    )
    creativity: float = 0.5


@dataclass
class Idea:
    """A single structured research idea.

    Attributes:
        hypothesis: The research hypothesis / conjecture.
        rationale: Why this direction is promising (evidence from review).
        suggested_experiment: Concrete experimental design.
        category: Which strategy generated this idea.
        confidence: Estimated likelihood of impact (0.0-1.0).
    """

    hypothesis: str
    rationale: str
    suggested_experiment: str
    category: str
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, object]:
        return {
            "hypothesis": self.hypothesis,
            "rationale": self.rationale,
            "suggested_experiment": self.suggested_experiment,
            "category": self.category,
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------------------
# Strategy 1: Gap-based ideas
# ---------------------------------------------------------------------------

def _gap_based_ideas(
    claims: List[Dict[str, object]],
    validation: Optional[Dict[str, object]],
) -> List[Idea]:
    """Generate ideas from discrepancies between claims and experiments."""
    ideas: List[Idea] = []

    # Use validation mismatches
    if validation:
        mismatches = validation.get("mismatches", [])
        for mismatch in mismatches:
            claim_text = mismatch.get("claim_text", "").strip()
            expected = mismatch.get("expected_value", "?")
            found = mismatch.get("found_value", "?")
            desc = mismatch.get("description", "")
            claim_line = mismatch.get("claim_line", 0)

            if not claim_text:
                continue

            # Truncate long claim text
            if len(claim_text) > 120:
                claim_text = claim_text[:117] + "..."

            hypothesis = (
                f"The gap between the stated value ({expected}) and the observed "
                f"value ({found}) for the claim on line {claim_line} is caused by "
                f"an unaccounted confound in the experimental setup."
            )
            rationale = (
                f"Validation mismatch detected: {desc or claim_text}. "
                f"The experiment yielded {found} instead of the claimed {expected}, "
                f"suggesting that one or more uncontrolled factors affect the outcome."
            )
            suggested_experiment = (
                f"Run a controlled ablation study varying potential confounds "
                f"(e.g., initialization seed, learning rate schedule, batch size, "
                f"data ordering) while holding the core method fixed. For each "
                f"factor, report the distribution of outcomes across 5+ random seeds "
                f"to determine which factor bridges the gap."
            )
            ideas.append(Idea(
                hypothesis=hypothesis,
                rationale=rationale,
                suggested_experiment=suggested_experiment,
                category="gap",
                confidence=0.6,
            ))
            if len(ideas) >= 2:
                break

    # Use low-confidence claims
    if len(ideas) < 2:
        low_conf_claims = [c for c in claims if c.get("confidence", 1.0) < 0.6]
        for claim in low_conf_claims:
            text = claim.get("text", "").strip()
            kind = claim.get("kind", "unknown")
            value = claim.get("value", "")
            line = claim.get("line_number", 0)

            if len(text) > 120:
                text = text[:117] + "..."

            hypothesis = (
                f"The {kind} claim on line {line} (value: {value}) has low "
                f"confidence in the original paper and may not replicate under "
                f"controlled conditions."
            )
            rationale = (
                f"Claim confidence was estimated at {claim.get('confidence', 0.0):.2f}, "
                f"below the 0.6 threshold. The claim appears in the paper body but "
                f"lacks clear supporting evidence or is ambiguous: \"{text}\"."
            )
            suggested_experiment = (
                f"Design a pre-registered replication study that isolates the exact "
                f"experimental conditions of this claim. Run 10 independent trials "
                f"with the same hyperparameters and report mean / std. Compare against "
                f"the paper's stated {value}."
            )
            ideas.append(Idea(
                hypothesis=hypothesis,
                rationale=rationale,
                suggested_experiment=suggested_experiment,
                category="gap",
                confidence=0.65,
            ))
            if len(ideas) >= 2:
                break

    return ideas


# ---------------------------------------------------------------------------
# Strategy 2: Methodology ideas
# ---------------------------------------------------------------------------

def _methodology_ideas(audit: Optional[Dict[str, object]]) -> List[Idea]:
    """Generate ideas from methodology / audit findings."""
    ideas: List[Idea] = []

    if not audit:
        return ideas

    findings = audit.get("findings", [])
    # Focus on errors and warnings
    issues = [f for f in findings if f.get("severity") in ("error", "warning")]

    for finding in issues:
        kind = finding.get("kind", "unknown")
        description = finding.get("description", "")
        recommendation = finding.get("recommendation", "")

        if not description:
            continue

        hypothesis = (
            f"The absence of proper {kind} methodology systematically distorts "
            f"benchmark rankings and may invalidate reported improvements."
        )
        rationale = (
            f"Audit finding ({finding.get('severity')}): {description}. "
            f"This methodological gap affects how results should be interpreted "
            f"and compared across papers in this sub-field."
        )
        suggested_experiment = (
            f"Conduct a systematic meta-study: collect 10+ recent papers in this "
            f"area and re-evaluate their reported improvements after controlling for "
            f"{kind}. Use a standardized evaluation protocol that enforces "
            f"{recommendation if recommendation else 'best practices in this area'}. "
            f"Report how many claimed improvements persist after correction."
        )
        ideas.append(Idea(
            hypothesis=hypothesis,
            rationale=rationale,
            suggested_experiment=suggested_experiment,
            category="methodology",
            confidence=0.7 if finding.get("severity") == "error" else 0.55,
        ))
        if len(ideas) >= 2:
            break

    # If no issues found, suggest a methodology audit extension
    if not issues:
        hypothesis = (
            "Standardized methodology checklists for this sub-field can reduce "
            "inconsistencies in reported results across papers."
        )
        rationale = (
            "No methodology issues were flagged in this audit, but cross-paper "
            "methodology audits are rare in ML literature. A formal taxonomy of "
            "common benchmark confounds would benefit the entire field."
        )
        suggested_experiment = (
            "Survey 20 recent papers from top venues (NeurIPS, ICML, ICLR) in "
            "this area. For each, apply the audit framework from this review and "
            "catalog the frequency of each methodology issue. Publish a meta-analysis "
            "with recommended best-practice guidelines."
        )
        ideas.append(Idea(
            hypothesis=hypothesis,
            rationale=rationale,
            suggested_experiment=suggested_experiment,
            category="methodology",
            confidence=0.5,
        ))

    return ideas


# ---------------------------------------------------------------------------
# Strategy 3: Domain translation ideas
# ---------------------------------------------------------------------------

def _domain_translation_ideas(
    math_report: Optional[Dict[str, object]],
    section_structure: Dict[str, object],
) -> List[Idea]:
    """Generate cross-domain conjecture ideas, following Moonshine's approach."""
    ideas: List[Idea] = []

    if not math_report:
        return ideas

    eq_count = math_report.get("equation_count", 0)
    theorem_count = math_report.get("theorem_count", 0)
    statements = math_report.get("statements", [])

    if eq_count < 2 and theorem_count < 1:
        return ideas  # Not enough mathematical content for domain translation

    # Identify technical domains from section structure
    sections = section_structure.get("sections_found", [])
    known_domains = {
        "method": "representation learning architectures",
        "results": "empirical evaluation methodology",
        "introduction": "the problem formulation in this paper",
        "conclusion": "the theoretical framework proposed here",
    }

    target_domains = [
        "computer vision (image classification, detection, segmentation)",
        "natural language processing (text classification, generation, translation)",
        "reinforcement learning (policy optimization, reward modeling)",
        "graph neural networks (node classification, link prediction)",
        "time-series forecasting (financial, weather, biological signals)",
        "scientific computing (PDE solving, molecular dynamics)",
        "audio processing (speech recognition, music generation)",
        "multi-modal learning (vision-language, audio-visual)",
        "federated learning (privacy-preserving distributed training)",
        "meta-learning (few-shot adaptation, hyperparameter optimization)",
    ]

    # Determine source technique
    technique = "the theoretical framework"
    for sec in sections:
        if sec in known_domains:
            technique = known_domains[sec]
            break

    # Generate one domain-translation idea
    hypothesis = (
        f"The {technique} used in this paper can be translated to "
        f"{target_domains[0]}, yielding competitive or superior performance "
        f"compared to domain-specific specialized approaches."
    )
    rationale = (
        f"The paper contains {eq_count} equations and {theorem_count} theorems, "
        f"indicating a formalizable technique that may generalize beyond the "
        f"original application domain. Following Moonshine's cross-domain "
        f"conjecture approach (arXiv:2606.10806), we hypothesize that the "
        f"core mathematical insight transfers to structurally similar problems."
    )
    suggested_experiment = (
        f"Implement the paper's core technique in {target_domains[0]}. "
        f"Select 3 standard benchmarks in that domain and compare against "
        f"domain-specialized baselines under matched compute budgets. "
        f"Report adaptation difficulty (hyperparameter tuning cost), "
        f"relative performance delta, and qualitative analysis of when "
        f"the transfer succeeds or fails."
    )
    ideas.append(Idea(
        hypothesis=hypothesis,
        rationale=rationale,
        suggested_experiment=suggested_experiment,
        category="domain_translation",
        confidence=0.5,
    ))

    # Second idea if there are theorems or many equations
    if theorem_count > 0 or eq_count > 5:
        hypothesis2 = (
            f"The theoretical results (theorems, lemmas) in this paper can be "
            f"translated into algorithmic improvements for {target_domains[3]} "
            f"that go beyond the paper's original experimental setting."
        )
        rationale2 = (
            f"With {theorem_count} theorem-like environments reported, the paper "
            f"offers provable guarantees that may yield guaranteed improvements "
            f"when adapted to {target_domains[3]}, where similar theoretical "
            f"frameworks are under-explored."
        )
        suggested_experiment2 = (
            f"Derive a variant of the paper's theoretical results specialized for "
            f"{target_domains[3]}. Implement and benchmark on 3 standard graph "
            f"benchmarks (e.g., ogbn-arxiv, Cora, PubMed). Compare against GCN, "
            f"GAT, and GraphTransformer baselines. Measure whether theoretical "
            f"guarantees translate to empirical gains."
        )
        ideas.append(Idea(
            hypothesis=hypothesis2,
            rationale=rationale2,
            suggested_experiment=suggested_experiment2,
            category="domain_translation",
            confidence=0.55,
        ))

    return ideas


# ---------------------------------------------------------------------------
# Strategy 4: Ablation ideas
# ---------------------------------------------------------------------------

def _ablation_ideas(claims: List[Dict[str, object]]) -> List[Idea]:
    """Generate ablation study ideas from comparison and parameter claims."""
    ideas: List[Idea] = []

    # Look for comparison claims
    comparison_claims = [c for c in claims if c.get("kind") == "comparison"]
    parameter_claims = [c for c in claims if c.get("kind") == "parameter"]
    benchmark_claims = [c for c in claims if c.get("kind") == "benchmark"]

    # Generate from comparison claims
    for claim in comparison_claims[:2]:
        text = claim.get("text", "").strip()
        if len(text) > 100:
            text = text[:97] + "..."

        hypothesis = (
            f"The reported comparison is confounded by unablated differences in "
            f"training setup (e.g., learning rate, optimizer, regularization) "
            f"between the compared methods."
        )
        rationale = (
            f"Comparison claim: \"{text}\". Without ablating individual factors, "
            f"it is unclear which component drives the performance difference."
        )
        suggested_experiment = (
            f"Design a factorial ablation study: start from the baseline method, "
            f"then add each component of the proposed method one at a time. "
            f"Report performance after each addition. Additionally, vary "
            f"hyperparameters (learning rate, batch size, weight decay) for each "
            f"ablation level to ensure differences are not due to suboptimal tuning "
            f"of the baseline. Use 5 random seeds per condition."
        )
        ideas.append(Idea(
            hypothesis=hypothesis,
            rationale=rationale,
            suggested_experiment=suggested_experiment,
            category="ablation",
            confidence=0.65,
        ))
        if len(ideas) >= 2:
            break

    # Generate from parameter claims
    if len(ideas) < 2 and parameter_claims:
        claim = parameter_claims[0]
        text = claim.get("text", "").strip()
        value = claim.get("value", "")
        if len(text) > 100:
            text = text[:97] + "..."
        line = claim.get("line_number", 0)

        hypothesis = (
            f"The parameter count (line {line}, value: {value}) is not the "
            f"sole determinant of performance; the architectural design within "
            f"a fixed parameter budget matters more."
        )
        rationale = (
            f"Parameter claim: \"{text}\". Comparing methods at a single parameter "
            f"count conflates architecture quality with scale. A controlled study "
            f"varying parameter count while keeping architecture fixed is needed."
        )
        suggested_experiment = (
            f"Vary the parameter count from 0.5x to 4x the reported value in "
            f"log-uniform steps (6 levels). For each level, train 3 different "
            f"architectural variants. Plot performance vs parameter count and "
            f"compute the Pareto frontier. Determine whether the claimed "
            f"advantage persists across scales or only holds at the specific "
            f"parameter count reported."
        )
        ideas.append(Idea(
            hypothesis=hypothesis,
            rationale=rationale,
            suggested_experiment=suggested_experiment,
            category="ablation",
            confidence=0.7,
        ))

    # If no comparison/parameter claims, generate from benchmark claims
    if not ideas and benchmark_claims:
        claim = benchmark_claims[0]
        text = claim.get("text", "").strip()
        if len(text) > 100:
            text = text[:97] + "..."

        hypothesis = (
            f"The reported benchmark performance is sensitive to evaluation "
            f"details (metric computation, preprocessing, prompt formatting) "
            f"that are not ablated in standard comparisons."
        )
        rationale = (
            f"Benchmark claim: \"{text}\". Benchmark evaluations are sensitive "
            f"to evaluation protocol details that are rarely studied systematically."
        )
        suggested_experiment = (
            f"Run a protocol ablation: for the reported benchmark, vary each "
            f"evaluation detail independently (prompt template, few-shot example "
            f"selection, metric computation variant, normalization). Report the "
            f"distribution of scores across protocol variants and identify which "
            f"choices shift rankings between methods."
        )
        ideas.append(Idea(
            hypothesis=hypothesis,
            rationale=rationale,
            suggested_experiment=suggested_experiment,
            category="ablation",
            confidence=0.6,
        ))

    return ideas


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_ideas(
    report: object,
    config: object,
) -> List[Idea]:
    """Generate structured research ideas from review report outputs.

    Applies four complementary strategies to produce 3-5 specific, actionable
    research directions grounded in the report's findings.

    Args:
        report: A ReviewReport instance whose dict fields (claims, validation,
            audit, math_report, section_structure) are already populated.
        config: A ReviewConfig instance (used for level checks, though at
            Level 5 all prior levels have already run).

    Returns:
        List of Idea objects sorted by confidence descending.
    """
    # Access report fields as dicts (they are already converted by Level 4)
    claims: List[Dict[str, object]] = getattr(report, "claims", []) or []
    validation: Optional[Dict[str, object]] = getattr(report, "validation", None)
    audit: Optional[Dict[str, object]] = getattr(report, "audit", None)
    math_report: Optional[Dict[str, object]] = getattr(report, "math_report", None)
    section_structure: Dict[str, object] = getattr(report, "section_structure", {}) or {}

    all_ideas: List[Idea] = []

    # Strategy 1: Gap-based ideas
    all_ideas.extend(_gap_based_ideas(claims, validation))

    # Strategy 2: Methodology ideas
    all_ideas.extend(_methodology_ideas(audit))

    # Strategy 3: Domain translation ideas
    all_ideas.extend(_domain_translation_ideas(math_report, section_structure))

    # Strategy 4: Ablation ideas
    all_ideas.extend(_ablation_ideas(claims))

    # Sort by confidence descending and cap at reasonable total
    all_ideas.sort(key=lambda idea: idea.confidence, reverse=True)

    # Target 3-5 ideas total; take up to config.max_ideas if available
    max_ideas = getattr(config, "level", 5) * 1  # Use level as crude cap
    try:
        max_ideas_cfg = getattr(config, "idea_config", None)
        if max_ideas_cfg is not None:
            max_ideas = max_ideas_cfg.max_ideas
    except AttributeError:
        pass

    return all_ideas[:max(3, min(max_ideas, len(all_ideas)))]
