# Machine Learning Full Course by AI

This directory is a standalone book project for a machine learning course centered on harness engineering.

Working subtitle:

`From First Principles to Production Systems in the Harness Era`

The core idea is simple: machine learning should no longer be taught as a one-way sequence of lectures and notebooks. In the future of learning, students will build and use skills, eval loops, project memory, and reusable harnesses that make both study and engineering work more reliable. This book is designed around that future.

## Project Files

- `OUTLINE.md`: the full-book structure and teaching thesis
- `BOOK_SUMMARY.md`: compressed memory of the book so future writing sessions stay coherent
- `glossary.md`: shared terminology
- `bibliography.md`: canonical references and source spine
- `src/`: the mdBook manuscript source
- `src/how-to-use-reader-skills.md`: the reader-facing workflow for actually using the local skills

## Build Notes

If `mdbook` is installed:

```bash
mdbook build
```

For a full review build:

```bash
./scripts/build-book.sh
```

This produces:

- the HTML book in `book/`
- a review PDF in `dist/machine-learning-full-course-by-ai.pdf`

## Intended Reader Journey

The manuscript is written for four overlapping audiences:

1. Beginners who need structure, intuition, and confidence.
2. Builders who want to ship projects quickly.
3. Engineers who need production habits, experimentation discipline, and system thinking.
4. Specialists who want a launchpad toward research, infra, safety, or product leadership.

## Editorial Direction

- Language: English
- Tone: rigorous, practical, optimistic
- Pedagogy: concept, code, critique, reflection, rebuild
- Differentiator: explicit use of Codex-style skills and harness engineering as part of the learning method

## Current Status

- Full manuscript draft complete
- Reader skill usage guide added to the manuscript
- mdBook HTML build working
- PDF export working for review builds
