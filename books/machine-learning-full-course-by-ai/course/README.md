# Machine Learning Full Course Video Production

This folder turns the book manuscript into a chapter-by-chapter course production workspace.

## Goal

The final outcome is one complete YouTube course built from the book, not a few disconnected videos.

The book remains the master curriculum.

This `course/` folder is the production layer for:

- lesson design
- voice narration
- scene planning
- demo planning
- visual asset collection

## Course Structure

### Part 1. Foundations and Learning Method

1. Chapter 1: Learn Machine Learning Through Harness Engineering
2. Chapter 2: Math Without Losing Courage
3. Chapter 3: Data, Labels, and Problem Framing

### Part 2. Classical Models and Evaluation

4. Chapter 4: First Models: Linear Models and Nearest Neighbors
5. Chapter 5: Trees, Ensembles, and Strong Baselines
6. Chapter 6: Evaluation, Error Analysis, and Experiment Design

### Part 3. Training and Modern Modeling

7. Chapter 7: Optimization and Representation Learning
8. Chapter 8: Neural Networks in Practice
9. Chapter 9: Sequence Models, Attention, and Transformers
10. Chapter 10: Transfer Learning, Fine-Tuning, and Foundation Models

### Part 4. Retrieval, Ranking, and Systems

11. Chapter 11: Unsupervised Learning, Embeddings, and Retrieval
12. Chapter 12: Recommendation, Ranking, and Decision Systems
13. Chapter 13: Data Engineering for ML Teams
14. Chapter 14: Training, Serving, and MLOps

### Part 5. Responsibility and Growth

15. Chapter 15: Responsible AI, Safety, and Human Feedback
16. Chapter 16: From Learner to Professional ML Engineer

## Folder Convention

Each chapter folder contains:

- `lesson-outline.md`
- `voiceover.md`
- `scene-cards.md`
- `demo-commands.md`
- `assets/README.md`

## Production Docs

Top-level course planning docs live here:

- `production-checklist.md`
- `youtube-release-plan.md`

## Generated Visual Assets

Each chapter now has a generated title card at:

- `chapter-xx-.../assets/title-card.svg`

The same covers are mirrored into the book source at:

- `src/assets/course-covers/`

To regenerate them after renaming chapters or restyling the course:

```bash
cd /Users/xiaol/x/PaperX/books/machine-learning-full-course-by-ai
python3 scripts/generate_course_title_cards.py
```

## Recording Standard

Each lesson should try to include:

1. one real case
2. one visible figure
3. one artifact, command, or output
4. one reusable skill or harness
5. one closing takeaway

## Current Runnable Cases

These are already executable in the repo:

- `examples/delivery-time-prediction/`
- `examples/knowledge-search-retrieval/`

Use:

```bash
cd /Users/xiaol/x/PaperX/books/machine-learning-full-course-by-ai
./scripts/run-example-cases.sh
```

## Pilot Lesson Template

The first fully prepared pilot lesson package is:

- `chapter-04-first-models/`

It now includes:

- `README.md`
- `shot-list.md`
- `recording-checklist.md`

Use Chapter 4 as the reference template for upgrading the remaining chapters from scaffold to recording-ready lesson packs.

## Good Narration Rule

Do not read the book word for word.

Use the chapter as the source document, then record the lesson as:

problem -> process -> result -> judgment
