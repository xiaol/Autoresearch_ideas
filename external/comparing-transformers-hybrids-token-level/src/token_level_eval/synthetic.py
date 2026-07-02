from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from token_level_eval.common import parse_dtype, set_seed


FILLER_WORDS = (
    "during",
    "the",
    "quiet",
    "meeting",
    "people",
    "checked",
    "several",
    "notes",
    "before",
    "the",
    "report",
    "continued",
    "with",
    "ordinary",
    "details",
    "about",
    "the",
    "room",
    "and",
    "schedule",
)

PRONOUN_PAIRS = [
    ("Liam", "he", "violinist", "Naomi", "she", "pilot"),
    ("Noah", "he", "chemist", "Emma", "she", "archivist"),
    ("Ethan", "he", "designer", "Olivia", "she", "doctor"),
    ("Mason", "he", "teacher", "Ava", "she", "engineer"),
]

ENTITY_PAIRS = [
    ("Julia", "orange notebook", "Sofia", "green folder"),
    ("Maya", "silver key", "Nora", "blue ticket"),
    ("Iris", "red scarf", "Clara", "yellow map"),
    ("Elena", "glass cup", "Diana", "wooden box"),
]

CLOSURES = [
    ("<header>", "</header>"),
    ("<section>", "</section>"),
    ("<article>", "</article>"),
    ("[", "]"),
    ("(", ")"),
    ("{", "}"),
]


@dataclass(frozen=True)
class ProbeExample:
    family: str
    distance: int
    prefix: str
    positive: str
    negative: str | None = None


def filler(distance: int, rng: random.Random) -> str:
    return " ".join(rng.choice(FILLER_WORDS) for _ in range(distance))


def generate_pronoun(distance: int, rng: random.Random) -> ProbeExample:
    male, male_pronoun, male_role, female, female_pronoun, female_role = rng.choice(PRONOUN_PAIRS)
    query_male = rng.random() < 0.5
    role = male_role if query_male else female_role
    correct = male_pronoun if query_male else female_pronoun
    distractor = female_pronoun if query_male else male_pronoun
    prefix = (
        f"{male} is the {male_role}. {female} is the {female_role}. "
        f"{filler(distance, rng)} The {role} reviewed the report, and"
    )
    return ProbeExample("pronoun_memory", distance, prefix, " " + correct, " " + distractor)


def generate_entity(distance: int, rng: random.Random) -> ProbeExample:
    name_a, object_a, name_b, object_b = rng.choice(ENTITY_PAIRS)
    query_a = rng.random() < 0.5
    queried_object = object_a if query_a else object_b
    correct = name_a if query_a else name_b
    distractor = name_b if query_a else name_a
    prefix = (
        f"{name_a} carried the {object_a}. {name_b} carried the {object_b}. "
        f"{filler(distance, rng)} Q: Who carried the {queried_object}? "
        f"(A) {name_a} (B) {name_b} Answer:"
    )
    return ProbeExample("entity_tracking", distance, prefix, " " + correct, " " + distractor)


def generate_closure(distance: int, rng: random.Random) -> ProbeExample:
    opener, closer = rng.choice(CLOSURES)
    prefix = f"{opener}\n  counter += 1\n  {filler(distance, rng)}\n"
    return ProbeExample("structural_closure", distance, prefix, closer, None)


def generate_examples(distances: list[int], num_examples: int, seed: int) -> list[ProbeExample]:
    rng = random.Random(seed)
    examples: list[ProbeExample] = []
    generators = [generate_pronoun, generate_entity, generate_closure]
    for distance in distances:
        for _ in range(num_examples):
            for generator in generators:
                examples.append(generator(distance, rng))
    return examples


def score_candidate_logprob(model: Any, tokenizer: Any, prefix: str, candidate: str) -> float:
    import torch
    import torch.nn.functional as F

    from token_level_eval.scoring import infer_input_device

    full_text = prefix + candidate
    with torch.inference_mode():
        encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True, return_tensors="pt")
        input_ids = encoded["input_ids"]
        offsets = encoded.get("offset_mapping")
        if offsets is None:
            prefix_len = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
            candidate_positions = list(range(prefix_len, input_ids.size(1)))
        else:
            offset_pairs = offsets[0].tolist()
            candidate_positions = [idx for idx, (start, end) in enumerate(offset_pairs) if end > len(prefix) and idx > 0]
        if not candidate_positions:
            return float("-inf")

        device = infer_input_device(model)
        input_ids = input_ids.to(device)
        outputs = model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits.float()
        total_logprob = 0.0
        for pos in candidate_positions:
            label = input_ids[0, pos]
            log_probs = F.log_softmax(logits[0, pos - 1], dim=-1)
            total_logprob += float(log_probs[label].detach().cpu())
        return total_logprob


def evaluate_model(
    model_name: str,
    alias: str,
    examples: list[ProbeExample],
    *,
    tokenizer_name: str | None,
    model_cfg,
) -> list[dict[str, Any]]:
    from token_level_eval.scoring import load_causal_lm, load_tokenizer, unload_model

    tokenizer = load_tokenizer(
        tokenizer_name or model_name,
        trust_remote_code=model_cfg.trust_remote_code,
        local_files_only=model_cfg.local_files_only,
    )
    model = load_causal_lm(model_name, model_cfg)
    rows = evaluate_loaded_model(model, tokenizer, alias, examples)
    unload_model(model)
    return rows


def evaluate_loaded_model(
    model: Any,
    tokenizer: Any,
    alias: str,
    examples: list[ProbeExample],
    *,
    reset_fn: Any | None = None,
) -> list[dict[str, Any]]:
    from tqdm.auto import tqdm

    rows: list[dict[str, Any]] = []
    for example_id, example in enumerate(tqdm(examples, desc=alias)):
        if reset_fn is not None:
            reset_fn()
        positive_logprob = score_candidate_logprob(model, tokenizer, example.prefix, example.positive)
        if example.negative is not None:
            if reset_fn is not None:
                reset_fn()
            negative_logprob = score_candidate_logprob(model, tokenizer, example.prefix, example.negative)
            margin = positive_logprob - negative_logprob
            rows.append(
                {
                    "model": alias,
                    "family": example.family,
                    "distance": example.distance,
                    "example_id": example_id,
                    "positive": example.positive,
                    "negative": example.negative,
                    "positive_logprob": positive_logprob,
                    "negative_logprob": negative_logprob,
                    "margin": margin,
                    "accuracy": float(margin > 0.0),
                    "nll": None,
                }
            )
        else:
            rows.append(
                {
                    "model": alias,
                    "family": example.family,
                    "distance": example.distance,
                    "example_id": example_id,
                    "positive": example.positive,
                    "negative": None,
                    "positive_logprob": positive_logprob,
                    "negative_logprob": None,
                    "margin": None,
                    "accuracy": None,
                    "nll": -positive_logprob,
                }
            )
    return rows


def summarize_probe_rows(rows: list[dict[str, Any]]):
    import pandas as pd

    df = pd.DataFrame(rows)
    grouped = df.groupby(["model", "family", "distance"], dropna=False)
    return grouped.agg(
        count=("example_id", "size"),
        accuracy=("accuracy", "mean"),
        margin=("margin", "mean"),
        nll=("nll", "mean"),
    ).reset_index()


def parse_model_specs(values: list[str]) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for value in values:
        if "=" in value:
            alias, model = value.split("=", 1)
        else:
            model = value
            alias = Path(value).name
        specs.append((alias, model))
    return specs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run controlled token-level synthetic probes.")
    parser.add_argument("--models", nargs="+", required=True, help="Model specs alias=HF_MODEL_OR_PATH.")
    parser.add_argument("--tokenizer", default=None, help="Shared tokenizer path. Defaults to each model path.")
    parser.add_argument("--distances", nargs="+", type=int, default=[32, 64, 128, 256, 512, 1024])
    parser.add_argument("--num-examples", type=int, default=100, help="Examples per family per distance.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--no-trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from token_level_eval.scoring import ModelLoadConfig

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = generate_examples(args.distances, args.num_examples, args.seed)
    model_cfg = ModelLoadConfig(
        dtype=args.dtype,
        device=args.device,
        device_map=args.device_map,
        trust_remote_code=not args.no_trust_remote_code,
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
    )

    all_rows: list[dict[str, Any]] = []
    for alias, model_name in parse_model_specs(args.models):
        all_rows.extend(
            evaluate_model(
                model_name,
                alias,
                examples,
                tokenizer_name=args.tokenizer,
                model_cfg=model_cfg,
            )
        )

    rows_path = output_dir / "synthetic_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize_probe_rows(all_rows)
    summary.to_csv(output_dir / "synthetic_summary.csv", index=False)
    metadata = {
        "models": args.models,
        "distances": args.distances,
        "num_examples_per_family_per_distance": args.num_examples,
        "seed": args.seed,
        "dtype": str(parse_dtype(args.dtype)),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote synthetic probe rows to {rows_path}")
    print(f"Wrote summary to {output_dir / 'synthetic_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
