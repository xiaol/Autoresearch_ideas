#!/usr/bin/env python3

from __future__ import annotations

import html
import re
import textwrap
from pathlib import Path


BOOK_ROOT = Path(__file__).resolve().parents[1]
COURSE_ROOT = BOOK_ROOT / "course"
BOOK_COVER_ROOT = BOOK_ROOT / "src" / "assets" / "course-covers"
BOOK_PROCESS_ROOT = BOOK_ROOT / "src" / "assets" / "course-process"
BOOK_EVIDENCE_ROOT = BOOK_ROOT / "src" / "assets" / "course-evidence"


PART_PALETTES = {
    "Part 1. Foundations and Learning Method": {
        "bg_a": "#0F172A",
        "bg_b": "#1D4ED8",
        "accent": "#F59E0B",
        "muted": "#93C5FD",
        "pill": "#1E3A8A",
        "surface": "#0B1220",
        "surface_alt": "#111C33",
    },
    "Part 2. Classical Models and Evaluation": {
        "bg_a": "#111827",
        "bg_b": "#0F766E",
        "accent": "#F97316",
        "muted": "#99F6E4",
        "pill": "#134E4A",
        "surface": "#0F172A",
        "surface_alt": "#10262A",
    },
    "Part 3. Training and Modern Modeling": {
        "bg_a": "#1F2937",
        "bg_b": "#7C3AED",
        "accent": "#22C55E",
        "muted": "#DDD6FE",
        "pill": "#4C1D95",
        "surface": "#111827",
        "surface_alt": "#261B52",
    },
    "Part 4. Retrieval, Ranking, and Systems": {
        "bg_a": "#0F172A",
        "bg_b": "#0369A1",
        "accent": "#FB7185",
        "muted": "#BAE6FD",
        "pill": "#0C4A6E",
        "surface": "#101827",
        "surface_alt": "#0F2C45",
    },
    "Part 5. Responsibility and Growth": {
        "bg_a": "#111827",
        "bg_b": "#166534",
        "accent": "#FACC15",
        "muted": "#BBF7D0",
        "pill": "#14532D",
        "surface": "#111827",
        "surface_alt": "#183B28",
    },
}


CHAPTERS = [
    {
        "slug": "chapter-01-harness-engineering",
        "number": "01",
        "title": "Learn Machine Learning Through Harness Engineering",
        "part": "Part 1. Foundations and Learning Method",
        "hook": "Build reusable learning systems around AI tools.",
    },
    {
        "slug": "chapter-02-math-without-losing-courage",
        "number": "02",
        "title": "Math Without Losing Courage",
        "part": "Part 1. Foundations and Learning Method",
        "hook": "Translate formulas into plain engineering language.",
    },
    {
        "slug": "chapter-03-problem-framing",
        "number": "03",
        "title": "Data, Labels, and Problem Framing",
        "part": "Part 1. Foundations and Learning Method",
        "hook": "Turn vague ML requests into decision-ready tasks.",
    },
    {
        "slug": "chapter-04-first-models",
        "number": "04",
        "title": "First Models: Linear Models and Nearest Neighbors",
        "part": "Part 2. Classical Models and Evaluation",
        "hook": "Earn complexity with honest first baselines.",
    },
    {
        "slug": "chapter-05-tabular-models",
        "number": "05",
        "title": "Trees, Ensembles, and Strong Baselines",
        "part": "Part 2. Classical Models and Evaluation",
        "hook": "Compare tabular model families without hype.",
    },
    {
        "slug": "chapter-06-evaluation-and-experiment-design",
        "number": "06",
        "title": "Evaluation, Error Analysis, and Experiment Design",
        "part": "Part 2. Classical Models and Evaluation",
        "hook": "Trust results by inspecting errors, slices, and thresholds.",
    },
    {
        "slug": "chapter-07-optimization-and-representation",
        "number": "07",
        "title": "Optimization and Representation Learning",
        "part": "Part 3. Training and Modern Modeling",
        "hook": "Debug training and understand learned representations.",
    },
    {
        "slug": "chapter-08-neural-networks-in-practice",
        "number": "08",
        "title": "Neural Networks in Practice",
        "part": "Part 3. Training and Modern Modeling",
        "hook": "Structure neural experiments so others can rerun them.",
    },
    {
        "slug": "chapter-09-sequence-models-and-transformers",
        "number": "09",
        "title": "Sequence Models, Attention, and Transformers",
        "part": "Part 3. Training and Modern Modeling",
        "hook": "Read transformer systems as engineering choices, not magic.",
    },
    {
        "slug": "chapter-10-adaptation-and-foundation-models",
        "number": "10",
        "title": "Transfer Learning, Fine-Tuning, and Foundation Models",
        "part": "Part 3. Training and Modern Modeling",
        "hook": "Choose prompting, retrieval, or fine-tuning on purpose.",
    },
    {
        "slug": "chapter-11-retrieval-and-embeddings",
        "number": "11",
        "title": "Unsupervised Learning, Embeddings, and Retrieval",
        "part": "Part 4. Retrieval, Ranking, and Systems",
        "hook": "Inspect embeddings and retrieval with real evidence.",
    },
    {
        "slug": "chapter-12-ranking-and-decision-systems",
        "number": "12",
        "title": "Recommendation, Ranking, and Decision Systems",
        "part": "Part 4. Retrieval, Ranking, and Systems",
        "hook": "Treat ranking as a feedback loop, not a score table.",
    },
    {
        "slug": "chapter-13-data-engineering-for-ml",
        "number": "13",
        "title": "Data Engineering for ML Teams",
        "part": "Part 4. Retrieval, Ranking, and Systems",
        "hook": "Fix data pipelines before blaming the model.",
    },
    {
        "slug": "chapter-14-mlops-and-launch",
        "number": "14",
        "title": "Training, Serving, and MLOps",
        "part": "Part 4. Retrieval, Ranking, and Systems",
        "hook": "Launch models with monitoring, rollback, and cost control.",
    },
    {
        "slug": "chapter-15-risk-and-responsibility",
        "number": "15",
        "title": "Responsible AI, Safety, and Human Feedback",
        "part": "Part 5. Responsibility and Growth",
        "hook": "Make safety and responsibility part of normal engineering.",
    },
    {
        "slug": "chapter-16-professional-growth",
        "number": "16",
        "title": "From Learner to Professional ML Engineer",
        "part": "Part 5. Responsibility and Growth",
        "hook": "Build a career through visible artifacts and judgment.",
    },
]


REAL_OUTPUT_PATHS = {
    "chapter-04-first-models": BOOK_ROOT
    / "examples"
    / "delivery-time-prediction"
    / "artifacts"
    / "run-output.txt",
    "chapter-11-retrieval-and-embeddings": BOOK_ROOT
    / "examples"
    / "knowledge-search-retrieval"
    / "artifacts"
    / "run-output.txt",
}


def wrap_lines(text: str, width: int) -> list[str]:
    return textwrap.wrap(
        text,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )


def title_lines(title: str) -> list[str]:
    lines = wrap_lines(title, width=28)
    if len(lines) <= 3:
        return lines
    merged = lines[:2]
    merged.append(" ".join(lines[2:]))
    return merged


def hook_lines(hook: str) -> list[str]:
    return wrap_lines(hook, width=46)


def svg_text_block(
    x: int,
    y: int,
    lines: list[str],
    font_size: int,
    fill: str,
    line_height: int,
    font_weight: str = "400",
    anchor: str = "start",
) -> str:
    escaped = [html.escape(line) for line in lines if line]
    if not escaped:
        return ""

    parts = [
        (
            f'<text x="{x}" y="{y}" fill="{fill}" font-size="{font_size}" '
            f'font-weight="{font_weight}" text-anchor="{anchor}" '
            f'font-family="Arial, sans-serif">'
        )
    ]
    for index, line in enumerate(escaped):
        dy = 0 if index == 0 else line_height
        parts.append(f'<tspan x="{x}" dy="{dy}">{line}</tspan>')
    parts.append("</text>")
    return "".join(parts)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_outline(path: Path) -> dict[str, object]:
    metadata: dict[str, str] = {}
    flow_steps: list[str] = []
    in_flow = False

    for raw_line in read_text(path).splitlines():
        line = raw_line.strip()
        if line == "## Teaching Flow":
            in_flow = True
            continue
        if in_flow and line.startswith("## "):
            in_flow = False
        if not in_flow and line.startswith("- ") and ":" in line:
            key, value = line[2:].split(":", 1)
            metadata[key.strip()] = value.strip()
        if in_flow:
            match = re.match(r"\d+\.\s+(.*)", line)
            if match:
                flow_steps.append(match.group(1).strip())

    metadata["Teaching Flow"] = flow_steps
    return metadata


def parse_markdown_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current_heading: str | None = None
    current_lines: list[str] = []

    for raw_line in text.splitlines():
        if raw_line.startswith("## "):
            if current_heading is not None:
                sections[current_heading] = "\n".join(current_lines).strip()
            current_heading = raw_line[3:].strip()
            current_lines = []
            continue
        if current_heading is not None:
            current_lines.append(raw_line)

    if current_heading is not None:
        sections[current_heading] = "\n".join(current_lines).strip()

    return sections


def extract_code_blocks(text: str, language: str) -> list[str]:
    pattern = rf"```{language}\n(.*?)```"
    return [match.strip() for match in re.findall(pattern, text, flags=re.S)]


def parse_demo_commands(path: Path) -> dict[str, object]:
    text = read_text(path)
    sections = parse_markdown_sections(text)
    bash_blocks = extract_code_blocks(text, "bash")
    text_blocks = extract_code_blocks(text, "text")

    return {
        "sections": sections,
        "bash": bash_blocks[0] if bash_blocks else "",
        "prompt": text_blocks[0] if text_blocks else "",
    }


def summarize_section(section_text: str, width: int, max_lines: int) -> list[str]:
    if not section_text:
        return []

    lines: list[str] = []
    for raw_line in section_text.splitlines():
        cleaned = raw_line.strip()
        if not cleaned:
            continue
        cleaned = cleaned.lstrip("-").strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        wrapped = wrap_lines(cleaned, width=width) or [cleaned]
        lines.extend(wrapped)
        if len(lines) >= max_lines:
            break

    return lines[:max_lines]


def derive_demo_input(demo_meta: dict[str, object]) -> tuple[str, list[str]]:
    bash_block = str(demo_meta.get("bash", "")).strip()
    prompt_block = str(demo_meta.get("prompt", "")).strip()

    if bash_block:
        lines: list[str] = []
        for raw_line in bash_block.splitlines():
            cleaned = raw_line.strip()
            if not cleaned:
                continue
            prefixed = cleaned if cleaned.startswith("$") else f"$ {cleaned}"
            lines.extend(wrap_lines(prefixed, width=48) or [prefixed])
        return "Demo Command", lines[:8]

    if prompt_block:
        flattened = re.sub(r"\s+", " ", prompt_block)
        lines = wrap_lines(flattened, width=54)
        return "Skill Prompt", lines[:8]

    sections = demo_meta.get("sections", {})
    if isinstance(sections, dict):
        for heading in (
            "Recommended On-Screen Demo",
            "On-Screen Artifact",
            "Optional Supporting Demo",
        ):
            if heading in sections:
                return heading, summarize_section(str(sections[heading]), width=52, max_lines=8)

    return "Demo Input", ["Prepare the lesson artifact pack before recording."]


def derive_evidence_lines(slug: str, outline_meta: dict[str, object], demo_meta: dict[str, object]) -> tuple[str, list[str]]:
    real_output_path = REAL_OUTPUT_PATHS.get(slug)
    if real_output_path and real_output_path.exists():
        lines = [
            line.rstrip()
            for line in read_text(real_output_path).splitlines()
            if line.strip()
        ]
        return "Observed Output", lines[:8]

    artifact = str(outline_meta.get("On-screen artifact", ""))
    takeaway = str(outline_meta.get("Viewer takeaway", ""))
    sections = demo_meta.get("sections", {})

    lines = [
        f"Artifact focus: {artifact}",
        f"Case: {outline_meta.get('Anchor case', '')}",
        f"Judgment: {takeaway}",
    ]

    if isinstance(sections, dict):
        section_text = str(
            sections.get("On-Screen Artifact")
            or sections.get("Recommended On-Screen Demo")
            or sections.get("Optional Supporting Demo")
            or ""
        )
        for line in summarize_section(section_text, width=60, max_lines=3):
            lines.append(line)

    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(wrap_lines(line, width=58) or [line])

    return "Artifact Capture", wrapped[:8]


def render_card(chapter: dict[str, str]) -> str:
    palette = PART_PALETTES[chapter["part"]]
    title = [html.escape(line) for line in title_lines(chapter["title"])]
    hook = [html.escape(line) for line in hook_lines(chapter["hook"])]
    part = html.escape(chapter["part"])
    chapter_label = f"Chapter {chapter['number']}"

    pill_width = max(360, min(760, 44 + len(part) * 11))

    title_nodes = []
    for index, line in enumerate(title):
        y = 342 + index * 92
        title_nodes.append(
            f'<text x="148" y="{y}" fill="#F8FAFC" font-size="74" '
            f'font-weight="700" font-family="Arial, sans-serif">{line}</text>'
        )

    hook_nodes = []
    for index, line in enumerate(hook):
        y = 650 + index * 44
        hook_nodes.append(
            f'<text x="148" y="{y}" fill="{palette["muted"]}" font-size="32" '
            f'font-family="Arial, sans-serif">{line}</text>'
        )

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900" fill="none">
  <defs>
    <linearGradient id="bg" x1="120" y1="40" x2="1480" y2="860" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="{palette["bg_a"]}"/>
      <stop offset="100%" stop-color="{palette["bg_b"]}"/>
    </linearGradient>
    <radialGradient id="haloA" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(1320 180) rotate(90) scale(340 340)">
      <stop offset="0%" stop-color="{palette["accent"]}" stop-opacity="0.34"/>
      <stop offset="100%" stop-color="{palette["accent"]}" stop-opacity="0"/>
    </radialGradient>
    <radialGradient id="haloB" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(1420 760) rotate(90) scale(320 320)">
      <stop offset="0%" stop-color="#FFFFFF" stop-opacity="0.14"/>
      <stop offset="100%" stop-color="#FFFFFF" stop-opacity="0"/>
    </radialGradient>
    <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
      <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#FFFFFF" stroke-opacity="0.06" stroke-width="1"/>
    </pattern>
  </defs>
  <rect width="1600" height="900" fill="url(#bg)"/>
  <rect width="1600" height="900" fill="url(#grid)"/>
  <circle cx="1320" cy="180" r="340" fill="url(#haloA)"/>
  <circle cx="1420" cy="760" r="320" fill="url(#haloB)"/>
  <rect x="88" y="82" width="1424" height="736" rx="42" fill="#020617" fill-opacity="0.28" stroke="#FFFFFF" stroke-opacity="0.16" stroke-width="2"/>
  <rect x="112" y="108" width="14" height="684" rx="7" fill="{palette["accent"]}"/>
  <rect x="1328" y="112" width="116" height="116" rx="22" fill="#FFFFFF" fill-opacity="0.05" stroke="#FFFFFF" stroke-opacity="0.12"/>
  <rect x="1280" y="160" width="212" height="212" rx="34" fill="#FFFFFF" fill-opacity="0.04" stroke="#FFFFFF" stroke-opacity="0.10"/>
  <text x="148" y="156" fill="#E2E8F0" font-size="26" letter-spacing="4" font-family="Arial, sans-serif">MACHINE LEARNING FULL COURSE BY AI</text>
  <rect x="148" y="196" width="{pill_width}" height="54" rx="27" fill="{palette["pill"]}" fill-opacity="0.92" stroke="#FFFFFF" stroke-opacity="0.14"/>
  <text x="178" y="231" fill="#F8FAFC" font-size="25" font-family="Arial, sans-serif">{part}</text>
  <text x="1290" y="214" fill="{palette["muted"]}" font-size="34" text-anchor="end" font-family="Arial, sans-serif">{chapter_label}</text>
  <text x="1448" y="792" fill="#FFFFFF" fill-opacity="0.08" font-size="160" text-anchor="end" font-weight="700" font-family="Arial, sans-serif">{chapter["number"]}</text>
  {"".join(title_nodes)}
  {"".join(hook_nodes)}
  <text x="148" y="770" fill="#F8FAFC" fill-opacity="0.82" font-size="24" letter-spacing="2" font-family="Arial, sans-serif">FROM MANUSCRIPT TO VIDEO LESSON</text>
  <text x="148" y="804" fill="#CBD5E1" fill-opacity="0.72" font-size="20" font-family="Arial, sans-serif">Use as chapter opener, thumbnail base, or editing placeholder.</text>
</svg>
'''


def render_process_figure(chapter: dict[str, str], outline_meta: dict[str, object]) -> str:
    palette = PART_PALETTES[chapter["part"]]
    flow_steps = outline_meta.get("Teaching Flow", [])
    if not isinstance(flow_steps, list):
        flow_steps = []

    step_boxes: list[str] = []
    connector_nodes: list[str] = []
    start_y = 182
    box_height = 106
    gap = 18

    for index, step in enumerate(flow_steps[:5]):
        y = start_y + index * (box_height + gap)
        box_fill = palette["surface"] if index % 2 == 0 else palette["surface_alt"]
        step_boxes.append(
            f'<rect x="96" y="{y}" width="850" height="{box_height}" rx="24" fill="{box_fill}" stroke="#FFFFFF" stroke-opacity="0.10"/>'
        )
        step_boxes.append(
            f'<circle cx="146" cy="{y + 53}" r="22" fill="{palette["accent"]}" fill-opacity="0.92"/>'
        )
        step_boxes.append(
            svg_text_block(
                146,
                y + 61,
                [str(index + 1)],
                font_size=24,
                fill="#08111F",
                line_height=24,
                font_weight="700",
                anchor="middle",
            )
        )
        step_boxes.append(
            svg_text_block(
                188,
                y + 42,
                wrap_lines(str(step), width=43),
                font_size=28,
                fill="#F8FAFC",
                line_height=34,
                font_weight="600",
            )
        )
        if index < min(4, len(flow_steps) - 1):
            connector_y = y + box_height
            connector_nodes.append(
                f'<path d="M 522 {connector_y + 2} C 522 {connector_y + 16}, 522 {connector_y + 24}, 522 {connector_y + 28}" stroke="{palette["accent"]}" stroke-width="4" stroke-linecap="round"/>'
            )
            connector_nodes.append(
                f'<path d="M 522 {connector_y + 28} L 514 {connector_y + 18} L 530 {connector_y + 18} Z" fill="{palette["accent"]}"/>'
            )

    title = svg_text_block(
        96,
        92,
        title_lines(chapter["title"]),
        font_size=44,
        fill="#F8FAFC",
        line_height=50,
        font_weight="700",
    )
    subtitle = svg_text_block(
        96,
        146,
        wrap_lines(str(outline_meta.get("Primary visual", "")), width=52),
        font_size=22,
        fill=palette["muted"],
        line_height=28,
        font_weight="500",
    )
    case_block = svg_text_block(
        1036,
        212,
        wrap_lines(f'Case: {outline_meta.get("Anchor case", "")}', width=27),
        font_size=26,
        fill="#F8FAFC",
        line_height=32,
        font_weight="600",
    )
    skill_block = svg_text_block(
        1036,
        394,
        wrap_lines(f'Skill: {outline_meta.get("Main skill or harness", "")}', width=27),
        font_size=24,
        fill="#F8FAFC",
        line_height=30,
        font_weight="600",
    )
    takeaway_block = svg_text_block(
        1036,
        592,
        wrap_lines(f'Takeaway: {outline_meta.get("Viewer takeaway", "")}', width=27),
        font_size=24,
        fill="#F8FAFC",
        line_height=30,
        font_weight="600",
    )

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900" fill="none">
  <defs>
    <linearGradient id="bg" x1="120" y1="40" x2="1480" y2="860" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="{palette["bg_a"]}"/>
      <stop offset="100%" stop-color="{palette["bg_b"]}"/>
    </linearGradient>
    <pattern id="grid" width="36" height="36" patternUnits="userSpaceOnUse">
      <path d="M 36 0 L 0 0 0 36" fill="none" stroke="#FFFFFF" stroke-opacity="0.05" stroke-width="1"/>
    </pattern>
  </defs>
  <rect width="1600" height="900" fill="url(#bg)"/>
  <rect width="1600" height="900" fill="url(#grid)"/>
  <rect x="62" y="54" width="1476" height="792" rx="36" fill="#020617" fill-opacity="0.22" stroke="#FFFFFF" stroke-opacity="0.10"/>
  <rect x="70" y="62" width="12" height="776" rx="6" fill="{palette["accent"]}"/>
  {title}
  {subtitle}
  <text x="1388" y="108" fill="{palette["muted"]}" font-size="30" text-anchor="end" font-family="Arial, sans-serif">Chapter {chapter["number"]}</text>
  <rect x="1000" y="156" width="500" height="170" rx="26" fill="{palette["surface"]}" stroke="#FFFFFF" stroke-opacity="0.10"/>
  <text x="1036" y="190" fill="{palette["muted"]}" font-size="18" letter-spacing="2" font-family="Arial, sans-serif">ANCHOR CASE</text>
  {case_block}
  <rect x="1000" y="348" width="500" height="170" rx="26" fill="{palette["surface_alt"]}" stroke="#FFFFFF" stroke-opacity="0.10"/>
  <text x="1036" y="382" fill="{palette["muted"]}" font-size="18" letter-spacing="2" font-family="Arial, sans-serif">MAIN SKILL OR HARNESS</text>
  {skill_block}
  <rect x="1000" y="540" width="500" height="238" rx="26" fill="{palette["surface"]}" stroke="#FFFFFF" stroke-opacity="0.10"/>
  <text x="1036" y="574" fill="{palette["muted"]}" font-size="18" letter-spacing="2" font-family="Arial, sans-serif">VIEWER TAKEAWAY</text>
  {takeaway_block}
  {"".join(step_boxes)}
  {"".join(connector_nodes)}
</svg>
'''


def render_evidence_panel(
    chapter: dict[str, str],
    outline_meta: dict[str, object],
    demo_meta: dict[str, object],
) -> str:
    palette = PART_PALETTES[chapter["part"]]
    input_label, input_lines = derive_demo_input(demo_meta)
    evidence_label, evidence_lines = derive_evidence_lines(chapter["slug"], outline_meta, demo_meta)

    artifact_lines = wrap_lines(
        f'Artifact: {outline_meta.get("On-screen artifact", "")}',
        width=28,
    )
    skill_lines = wrap_lines(
        f'Skill: {outline_meta.get("Main skill or harness", "")}',
        width=28,
    )
    case_lines = wrap_lines(
        f'Case: {outline_meta.get("Anchor case", "")}',
        width=28,
    )
    takeaway_lines = wrap_lines(
        f'Goal: {outline_meta.get("Viewer takeaway", "")}',
        width=28,
    )

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900" fill="none">
  <defs>
    <linearGradient id="bg" x1="120" y1="40" x2="1480" y2="860" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="{palette["bg_a"]}"/>
      <stop offset="100%" stop-color="{palette["bg_b"]}"/>
    </linearGradient>
    <pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse">
      <path d="M 32 0 L 0 0 0 32" fill="none" stroke="#FFFFFF" stroke-opacity="0.04" stroke-width="1"/>
    </pattern>
  </defs>
  <rect width="1600" height="900" fill="url(#bg)"/>
  <rect width="1600" height="900" fill="url(#grid)"/>
  <rect x="56" y="52" width="1488" height="796" rx="36" fill="#020617" fill-opacity="0.24" stroke="#FFFFFF" stroke-opacity="0.10"/>
  <rect x="74" y="70" width="418" height="760" rx="28" fill="{palette["surface"]}" stroke="#FFFFFF" stroke-opacity="0.10"/>
  <rect x="522" y="70" width="972" height="332" rx="28" fill="{palette["surface_alt"]}" stroke="#FFFFFF" stroke-opacity="0.10"/>
  <rect x="522" y="430" width="972" height="400" rx="28" fill="{palette["surface"]}" stroke="#FFFFFF" stroke-opacity="0.10"/>
  <text x="106" y="118" fill="#F8FAFC" font-size="42" font-weight="700" font-family="Arial, sans-serif">{html.escape(chapter["title"])}</text>
  <text x="106" y="156" fill="{palette["muted"]}" font-size="22" font-family="Arial, sans-serif">Chapter {chapter["number"]} evidence panel</text>
  <text x="106" y="212" fill="{palette["muted"]}" font-size="18" letter-spacing="2" font-family="Arial, sans-serif">ANCHOR CASE</text>
  {svg_text_block(106, 246, case_lines, 24, "#F8FAFC", 30, "600")}
  <text x="106" y="372" fill="{palette["muted"]}" font-size="18" letter-spacing="2" font-family="Arial, sans-serif">ARTIFACT FOCUS</text>
  {svg_text_block(106, 406, artifact_lines, 24, "#F8FAFC", 30, "600")}
  <text x="106" y="532" fill="{palette["muted"]}" font-size="18" letter-spacing="2" font-family="Arial, sans-serif">MAIN SKILL</text>
  {svg_text_block(106, 566, skill_lines, 22, "#F8FAFC", 28, "600")}
  <text x="106" y="676" fill="{palette["muted"]}" font-size="18" letter-spacing="2" font-family="Arial, sans-serif">LESSON GOAL</text>
  {svg_text_block(106, 710, takeaway_lines, 22, "#F8FAFC", 28, "600")}
  <text x="558" y="114" fill="{palette["muted"]}" font-size="18" letter-spacing="2" font-family="Arial, sans-serif">{html.escape(input_label.upper())}</text>
  <rect x="558" y="136" width="900" height="240" rx="20" fill="#020617" fill-opacity="0.44" stroke="#FFFFFF" stroke-opacity="0.08"/>
  {svg_text_block(586, 182, input_lines, 26, "#E2E8F0", 30, "500")}
  <text x="558" y="474" fill="{palette["muted"]}" font-size="18" letter-spacing="2" font-family="Arial, sans-serif">{html.escape(evidence_label.upper())}</text>
  <rect x="558" y="496" width="900" height="304" rx="20" fill="#020617" fill-opacity="0.44" stroke="#FFFFFF" stroke-opacity="0.08"/>
  {svg_text_block(586, 542, evidence_lines, 26, "#F8FAFC", 30, "500")}
</svg>
'''


def render_assets_readme(chapter: dict[str, str], outline_meta: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Chapter Assets",
            "",
            f"Generated asset pack for **{chapter['title']}**.",
            "",
            "Files:",
            "",
            "- `title-card.svg`",
            "- `process-figure.svg`",
            "- `evidence-panel.svg`",
            "",
            f"- Anchor case: {outline_meta.get('Anchor case', '')}",
            f"- Main skill or harness: {outline_meta.get('Main skill or harness', '')}",
            f"- On-screen artifact: {outline_meta.get('On-screen artifact', '')}",
            "",
            "Regenerate from the book root:",
            "",
            "```bash",
            "python3 scripts/generate_course_title_cards.py",
            "```",
            "",
        ]
    )


def write_if_changed(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def main() -> None:
    for chapter in CHAPTERS:
        outline_path = COURSE_ROOT / chapter["slug"] / "lesson-outline.md"
        demo_path = COURSE_ROOT / chapter["slug"] / "demo-commands.md"

        outline_meta = parse_outline(outline_path)
        demo_meta = parse_demo_commands(demo_path)

        title_svg = render_card(chapter)
        process_svg = render_process_figure(chapter, outline_meta)
        evidence_svg = render_evidence_panel(chapter, outline_meta, demo_meta)
        assets_readme = render_assets_readme(chapter, outline_meta)

        course_asset_root = COURSE_ROOT / chapter["slug"] / "assets"

        targets = {
            course_asset_root / "title-card.svg": title_svg,
            course_asset_root / "process-figure.svg": process_svg,
            course_asset_root / "evidence-panel.svg": evidence_svg,
            course_asset_root / "README.md": assets_readme,
            BOOK_COVER_ROOT / f'{chapter["slug"]}.svg': title_svg,
            BOOK_PROCESS_ROOT / f'{chapter["slug"]}-process.svg': process_svg,
            BOOK_EVIDENCE_ROOT / f'{chapter["slug"]}-evidence.svg': evidence_svg,
        }

        for path, content in targets.items():
            write_if_changed(path, content)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
