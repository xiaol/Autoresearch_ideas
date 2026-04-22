# Chapter 4 Example Video

This folder now supports rendering a local example video for Chapter 4.

## Output

The renderer produces:

- `video-output/chapter-04-example-video.mp4`
- `video-output/chapter-04-thumbnail.png`
- `video-output/video-manifest.json`
- per-scene slide images
- per-scene narration audio

The refined renderer produces:

- `video-output-v2/chapter-04-v2-minimax-female-subtitled.mp4`
- `video-output-v2/chapter-04-v2-subtitles.srt`
- `video-output-v2/chapter-04-v2-thumbnail.png`
- `video-output-v2/video-manifest.json`
- per-scene MiniMax narration audio

## How It Works

The example video is a narrated slide-style lesson built from:

- the chapter title card
- the generated process figure
- the generated evidence panel
- additional generated slides for the case brief, weight interpretation, and closing takeaway

## Render Command

```bash
cd /Users/xiaol/x/PaperX/books/machine-learning-full-course-by-ai/course/chapter-04-first-models
python3 render-video.py
```

Refined MiniMax/subtitled version:

```bash
cd /Users/xiaol/x/PaperX/books/machine-learning-full-course-by-ai/course/chapter-04-first-models
python3 render-video-v2.py
```

## Notes

- The narration uses macOS `say` with voice `Alex`.
- The rendered video is an example lesson video, not a final polished YouTube cut.
- It is meant to prove the pipeline and give you a concrete artifact to review.
- After review, we can improve pacing, subtitles, background music, or replace TTS with your own narration.
- The refined v2 renderer requests MiniMax `English_LovelyGirl` by default and falls back to macOS `Samantha` only if MiniMax generation fails.
