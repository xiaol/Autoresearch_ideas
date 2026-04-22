# Shot List

## Shot 1. Hook

- Duration: 0:00 to 0:35
- Visual: `assets/title-card.svg`
- On-screen text: "Why not jump straight to a deep model?"
- Narration goal: create tension around premature complexity

## Shot 2. Baseline Ladder

- Duration: 0:35 to 1:45
- Visual: `assets/process-figure.svg`
- On-screen focus: naive baseline -> interpretable baseline -> richer model
- Narration goal: explain why each rung answers a different engineering question

## Shot 3. Case Framing

- Duration: 1:45 to 2:35
- Visual: delivery brief and chapter process figure
- Source: `/Users/xiaol/x/PaperX/books/machine-learning-full-course-by-ai/examples/delivery-time-prediction/artifacts/brief.md`
- Narration goal: define the decision, prediction moment, and constraints

## Shot 4. Command Demo

- Duration: 2:35 to 3:30
- Visual: terminal capture
- Command:

```bash
cd /Users/xiaol/x/PaperX/books/machine-learning-full-course-by-ai
python3 examples/delivery-time-prediction/scripts/run_baseline.py
```

- Narration goal: show that the lesson is grounded in a real runnable case

## Shot 5. Result Readout

- Duration: 3:30 to 5:10
- Visual: `assets/evidence-panel.svg`
- On-screen focus:
  - naive mean baseline MAE: `9.12`
  - linear regression MAE: `1.94`
- Narration goal: explain what the improvement means and what it does not prove

## Shot 6. Weight Interpretation

- Duration: 5:10 to 6:30
- Visual: zoomed terminal output or evidence panel
- On-screen focus:
  - `distance_km: 11.706`
  - `weather_score: 3.495`
  - `is_rush_hour: 7.014`
- Narration goal: connect learned weights to operational intuition

## Shot 7. Skill Application

- Duration: 6:30 to 7:40
- Visual: Baseline Builder prompt on screen
- Prompt:

```text
Use $ml-baseline-builder to review whether the linear baseline is meaningfully better than the naive baseline in this delivery-time example.
```

- Narration goal: shift from result reading to reusable engineering judgment

## Shot 8. Closing Takeaway

- Duration: 7:40 to 8:20
- Visual: title card or simple takeaway slide
- On-screen text: "Complexity should earn its place."
- Narration goal: leave the viewer with one durable standard
