#!/usr/bin/env python3
"""CLI entry point for the PAT-style paper review tool.

Usage:
    python -m paper_review.cli papers/associative_state_universal_transformers/main.tex
    python -m paper_review.cli papers/recurrent_ffn/main.tex --level 2
    python -m paper_review.cli --list-papers
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from paper_review.config import ReviewConfig, ReviewLevel, load_config
from paper_review.core import discover_papers, run_review
from paper_review.report import write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PAT-style automated paper review tool for Autoresearch_ideas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m paper_review.cli papers/associative_state_universal_transformers/main.tex\n"
            "  python -m paper_review.cli papers/recurrent_ffn/main.tex --level 2\n"
            "  python -m paper_review.cli --list-papers\n"
            "  python -m paper_review.cli --list-papers --papers-root papers\n"
        ),
    )

    parser.add_argument(
        "paper_path",
        nargs="?",
        type=str,
        default=None,
        help="Path to the main .tex file to review.",
    )

    parser.add_argument(
        "--level",
        "-l",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=1,
        help="Review depth level (1=surface, 2=claims, 3=audit, 4=full, 5=ideas). Default: 1.",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (default: <paper_stem>_review.json).",
    )

    parser.add_argument(
        "--results-root",
        type=str,
        default="results",
        help="Root directory for experiment results. Default: results/.",
    )

    parser.add_argument(
        "--papers-root",
        type=str,
        default="papers",
        help="Root directory for paper packages. Default: papers/.",
    )

    parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["json", "md"],
        default="json",
        help="Output format. Default: json.",
    )

    parser.add_argument(
        "--list-papers",
        action="store_true",
        help="List all discoverable paper packages and exit.",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON config file for custom review settings.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Resolve paths
    papers_root = Path(args.papers_root).resolve()
    results_root = Path(args.results_root).resolve()

    # --list-papers mode
    if args.list_papers:
        papers = discover_papers(papers_root)
        if not papers:
            print(f"No paper packages found under {papers_root}.")
            return 0

        print(f"Discovered {len(papers)} paper packages under {papers_root}:\n")
        for p in papers:
            main_tex = p["main_tex"] or "(no .tex found)"
            print(f"  {p['name']}/")
            print(f"    main: {main_tex}")
            print(f"    .tex files: {p['tex_file_count']}")
        return 0

    # Review mode
    if args.paper_path is None:
        parser.print_help()
        return 1

    paper_path = Path(args.paper_path)
    if not paper_path.exists():
        print(f"Error: paper not found: {paper_path}", file=sys.stderr)
        return 1

    # Build configuration
    config = load_config(
        path=Path(args.config) if args.config else None,
        level=args.level,
        paper_path=paper_path,
        results_root=results_root,
        papers_root=papers_root,
        output_path=Path(args.output) if args.output else None,
        verbose=args.verbose,
    )

    if args.verbose:
        print(f"[paper_review] Reviewing: {paper_path}")
        print(f"[paper_review] Level: {config.level} ({config.level.name})")
        print(f"[paper_review] Output format: {args.format}")

    # Run review
    try:
        report = run_review(config, verbose=args.verbose)
    except Exception as e:
        print(f"Error during review: {e}", file=sys.stderr)
        return 1

    # Write output
    output_path = config.resolved_output_path
    write_report(report, output_path, fmt=args.format)

    # Also print summary to stdout
    print(f"\n--- Review Complete ---")
    print(report.summary)
    print(f"Full report written to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
