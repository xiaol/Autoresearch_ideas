from __future__ import annotations

import argparse
import csv
import json
import math
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import torch

from auto_research_llm_ideas.experiments.data import BYTE_BOS, BYTE_EOS, BYTE_PAD, default_device, set_seed
from auto_research_llm_ideas.experiments.train_lm import load_language_model_checkpoint

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - optional dependency path
    AutoModelForCausalLM = None
    AutoTokenizer = None


RULER_NOTES = [
    "This is a supported core subset of official RULER-style synthetic tasks, focused on self-contained tasks.",
    "The task names and scoring rules match official RULER conventions for niah, variable tracking, common-word extraction, and freq-word extraction.",
    "QA tasks and essay-based haystacks are intentionally excluded here because they depend on external corpora/download steps.",
]

NOISE_SENTENCE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."

WORD_BANK = [
    "amber", "atlas", "aurora", "basil", "cedar", "comet", "copper", "coral", "delta", "ember",
    "falcon", "fern", "frost", "glimmer", "harbor", "hazel", "indigo", "iris", "jade", "juniper",
    "lagoon", "lantern", "maple", "meadow", "meteor", "mist", "onyx", "opal", "orbit", "orchid",
    "pearl", "pine", "plume", "prairie", "quartz", "raven", "reef", "river", "sage", "scarlet",
    "shadow", "signal", "silver", "solstice", "spruce", "starling", "stone", "storm", "summit", "sunset",
    "thunder", "timber", "topaz", "valley", "velvet", "violet", "willow", "winter", "zephyr", "zenith",
]

RULER_TASKS: Dict[str, Dict[str, object]] = {
    "niah_single_1": {
        "task": "niah",
        "tokens_to_generate": 32,
        "template": (
            "Some special magic {type_needle_v} are hidden within the following text. "
            "Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n"
            "{context}\n"
            "What are all the special magic {type_needle_v} for {query} mentioned in the provided text?"
        ),
        "answer_prefix": " The special magic {type_needle_v} for {query} mentioned in the provided text are",
    },
    "vt": {
        "task": "variable_tracking",
        "tokens_to_generate": 24,
        "template": (
            "Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n"
            "{context}\n"
            "Question: Find all variables that are assigned the value {query} in the text above."
        ),
        "answer_prefix": (
            " Answer: According to the chain(s) of variable assignment in the text above, "
            "{num_v} variables are assigned the value {query}, they are: "
        ),
    },
    "cwe": {
        "task": "common_words_extraction",
        "tokens_to_generate": 48,
        "template": (
            "Below is a numbered list of words. In these words, some appear more often than others. "
            "Memorize the ones that appear most often.\n"
            "{context}\n"
            "Question: What are the 10 most common words in the above list?"
        ),
        "answer_prefix": " Answer: The top 10 words that appear most often in the list are:",
    },
    "fwe": {
        "task": "freq_words_extraction",
        "tokens_to_generate": 32,
        "template": (
            "Read the following coded text and track the frequency of each coded word. Find the three most "
            "frequently appeared coded words. {context}\n"
            "Question: Do not provide any explanation. Please ignore the dots '....'. "
            "What are the three most frequently appeared words in the above coded text?"
        ),
        "answer_prefix": " Answer: According to the coded text above, the three most frequently appeared words are:",
    },
}


def string_match_part(preds: Sequence[str], refs: Sequence[Sequence[str]]) -> float:
    score = sum(max(1.0 if reference.lower() in pred.lower() else 0.0 for reference in ref) for pred, ref in zip(preds, refs))
    return round(100.0 * score / max(len(preds), 1), 2)


def string_match_all(preds: Sequence[str], refs: Sequence[Sequence[str]]) -> float:
    score = 0.0
    for pred, ref in zip(preds, refs):
        score += sum(1.0 if reference.lower() in pred.lower() else 0.0 for reference in ref) / max(len(ref), 1)
    return round(100.0 * score / max(len(preds), 1), 2)


RULER_METRICS = {
    "niah": string_match_all,
    "variable_tracking": string_match_all,
    "common_words_extraction": string_match_all,
    "freq_words_extraction": string_match_all,
    "qa": string_match_part,
}


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def normalize_prediction(text: str) -> str:
    return "".join(character if character.isprintable() else " " for character in text).strip()


def make_random_identifier(rng: random.Random, length: int = 5) -> str:
    return "".join(rng.choice(string.ascii_uppercase) for _ in range(length))


def make_random_number(rng: random.Random, digits: int = 7) -> str:
    lower_bound = 10 ** (digits - 1)
    upper_bound = (10**digits) - 1
    return str(rng.randint(lower_bound, upper_bound))


def make_coded_word(rng: random.Random, length: int = 6) -> str:
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(length))


class ByteTextTokenizer:
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        payload = list(text.encode("utf-8", errors="ignore"))
        if add_special_tokens:
            return [BYTE_BOS, *payload]
        return payload

    def decode(self, token_ids: Sequence[int]) -> str:
        payload = bytes(token_id for token_id in token_ids if 0 <= token_id < 256)
        return payload.decode("utf-8", errors="ignore")

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))


class BaseGenerator:
    def count_tokens(self, text: str) -> int:
        raise NotImplementedError

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        raise NotImplementedError

    @property
    def max_context_length(self) -> int:
        raise NotImplementedError

    @property
    def descriptor(self) -> str:
        raise NotImplementedError


class LocalCheckpointGenerator(BaseGenerator):
    def __init__(self, checkpoint_path: str | Path, device: torch.device | None = None) -> None:
        self.device = device or default_device()
        self.model, self.payload = load_language_model_checkpoint(checkpoint_path, device=self.device)
        self.tokenizer = ByteTextTokenizer()
        self._descriptor = f"checkpoint:{Path(checkpoint_path).name}"

    @property
    def max_context_length(self) -> int:
        return int(self.payload["model_config"]["seq_len"])

    @property
    def descriptor(self) -> str:
        return self._descriptor

    def count_tokens(self, text: str) -> int:
        return self.tokenizer.count_tokens(text)

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        x = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        generated: List[int] = []
        self.model.eval()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                if x.size(1) > self.max_context_length:
                    x = x[:, -self.max_context_length :]
                logits, _ = self.model(x)
                next_token = int(torch.argmax(logits[:, -1, :], dim=-1).item())
                if next_token in {BYTE_EOS, BYTE_PAD, BYTE_BOS}:
                    break
                generated.append(next_token)
                x = torch.cat(
                    [x, torch.tensor([[next_token]], dtype=torch.long, device=self.device)],
                    dim=1,
                )

        return self.tokenizer.decode(generated)


class HuggingFaceGenerator(BaseGenerator):
    def __init__(self, model_name_or_path: str, device: torch.device | None = None) -> None:
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise RuntimeError("transformers is required for HuggingFaceGenerator")

        self.device = device or default_device()

        try:
            import lm_engine.hf_models  # noqa: F401
        except Exception:
            pass

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        max_positions = getattr(self.model.config, "max_position_embeddings", None)
        self._max_context_length = int(max_positions) if max_positions is not None else 4096
        self._descriptor = f"hf:{model_name_or_path}"

    @property
    def max_context_length(self) -> int:
        return self._max_context_length

    @property
    def descriptor(self) -> str:
        return self._descriptor

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=True))

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generated = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        continuation = generated[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(continuation, skip_special_tokens=True)


def format_reference_list(values: Sequence[str]) -> str:
    return ", ".join(values)


def build_niah_sample(task_name: str, seq_budget: int, generator: BaseGenerator, sample_seed: int) -> Dict[str, object]:
    task = RULER_TASKS[task_name]
    rng = random.Random(sample_seed)
    key = f"{rng.choice(WORD_BANK)}-{rng.choice(WORD_BANK)}"
    value = make_random_number(rng)
    needle_sentence = f"One of the special magic numbers for {key} is: {value}."
    template = task["template"]
    answer_prefix = task["answer_prefix"]

    def build_with_units(num_units: int) -> str:
        haystack = "\n".join([NOISE_SENTENCE for _ in range(num_units)])
        context_lines = haystack.split("\n")
        insertion_index = min(len(context_lines), max(0, len(context_lines) // 2))
        context_lines.insert(insertion_index, needle_sentence)
        context = "\n".join(context_lines)
        prompt = template.format(type_needle_v="number", context=context, query=key)
        return prompt + answer_prefix.format(type_needle_v="number", query=key)

    prompt = fit_prompt_to_budget(
        build_prompt=build_with_units,
        token_counter=generator.count_tokens,
        seq_budget=seq_budget,
        max_new_tokens=int(task["tokens_to_generate"]),
    )
    input_text = prompt[: -len(answer_prefix.format(type_needle_v="number", query=key))]
    return {
        "task_name": task_name,
        "task_type": task["task"],
        "input": input_text,
        "answer_prefix": answer_prefix.format(type_needle_v="number", query=key),
        "outputs": [value],
    }


def build_variable_tracking_sample(task_name: str, seq_budget: int, generator: BaseGenerator, sample_seed: int) -> Dict[str, object]:
    task = RULER_TASKS[task_name]
    rng = random.Random(sample_seed)
    variables = [make_random_identifier(rng) for _ in range(5)]
    base_value = make_random_number(rng, digits=5)
    chain = [f"VAR {variables[0]} = {base_value}"]
    chain.extend(f"VAR {variables[index + 1]} = VAR {variables[index]}" for index in range(4))

    def build_with_units(num_units: int) -> str:
        sentences = [NOISE_SENTENCE for _ in range(num_units)]
        positions = sorted(rng.sample(range(max(len(sentences), len(chain))), len(chain)))
        offset = 0
        for position, statement in zip(positions, chain):
            sentences.insert(min(position + offset, len(sentences)), statement)
            offset += 1
        context = "\n".join(sentences)
        prompt = task["template"].format(context=context, query=base_value)
        prefix = task["answer_prefix"].format(num_v=len(variables), query=base_value)
        return prompt + prefix

    prompt = fit_prompt_to_budget(
        build_prompt=build_with_units,
        token_counter=generator.count_tokens,
        seq_budget=seq_budget,
        max_new_tokens=int(task["tokens_to_generate"]),
    )
    prefix = task["answer_prefix"].format(num_v=len(variables), query=base_value)
    return {
        "task_name": task_name,
        "task_type": task["task"],
        "input": prompt[: -len(prefix)],
        "answer_prefix": prefix,
        "outputs": variables,
    }


def build_common_words_sample(task_name: str, seq_budget: int, generator: BaseGenerator, sample_seed: int) -> Dict[str, object]:
    task = RULER_TASKS[task_name]
    rng = random.Random(sample_seed)
    common_words = rng.sample(WORD_BANK, 10)
    uncommon_words = [f"{rng.choice(WORD_BANK)}_{index}" for index in range(40)]

    def build_with_units(num_units: int) -> str:
        pool = common_words * 8 + uncommon_words[: max(5, num_units)] * 1
        rng_local = random.Random(sample_seed + num_units)
        rng_local.shuffle(pool)
        context = " ".join(f"{index + 1}. {word}" for index, word in enumerate(pool))
        prompt = task["template"].format(context=context)
        return prompt + task["answer_prefix"]

    prompt = fit_prompt_to_budget(
        build_prompt=build_with_units,
        token_counter=generator.count_tokens,
        seq_budget=seq_budget,
        max_new_tokens=int(task["tokens_to_generate"]),
    )
    return {
        "task_name": task_name,
        "task_type": task["task"],
        "input": prompt[: -len(task["answer_prefix"])],
        "answer_prefix": task["answer_prefix"],
        "outputs": common_words,
    }


def build_freq_words_sample(task_name: str, seq_budget: int, generator: BaseGenerator, sample_seed: int) -> Dict[str, object]:
    task = RULER_TASKS[task_name]
    rng = random.Random(sample_seed)
    vocab = ["..."] + [make_coded_word(rng) for _ in range(32)]
    top_three = vocab[1:4]

    def build_with_units(num_units: int) -> str:
        counts = []
        total_weight = sum(1.0 / ((rank + 1) ** 2.0) for rank in range(len(vocab)))
        for rank, word in enumerate(vocab):
            expected = max(1, int((num_units * (1.0 / ((rank + 1) ** 2.0))) / total_weight))
            counts.extend([word] * expected)
        rng_local = random.Random(sample_seed + num_units)
        rng_local.shuffle(counts)
        context = " ".join(counts)
        prompt = task["template"].format(context=context)
        return prompt + task["answer_prefix"]

    prompt = fit_prompt_to_budget(
        build_prompt=build_with_units,
        token_counter=generator.count_tokens,
        seq_budget=seq_budget,
        max_new_tokens=int(task["tokens_to_generate"]),
    )
    return {
        "task_name": task_name,
        "task_type": task["task"],
        "input": prompt[: -len(task["answer_prefix"])],
        "answer_prefix": task["answer_prefix"],
        "outputs": top_three,
    }


TASK_BUILDERS = {
    "niah_single_1": build_niah_sample,
    "vt": build_variable_tracking_sample,
    "cwe": build_common_words_sample,
    "fwe": build_freq_words_sample,
}


def fit_prompt_to_budget(
    *,
    build_prompt: Callable[[int], str],
    token_counter: Callable[[str], int],
    seq_budget: int,
    max_new_tokens: int,
) -> str:
    low = 1
    high = max(8, seq_budget)
    best_prompt = build_prompt(low)

    while token_counter(build_prompt(high)) + max_new_tokens <= seq_budget and high < seq_budget * 8:
        best_prompt = build_prompt(high)
        high *= 2

    while low <= high:
        mid = (low + high) // 2
        prompt = build_prompt(mid)
        total_tokens = token_counter(prompt) + max_new_tokens
        if total_tokens <= seq_budget:
            best_prompt = prompt
            low = mid + 1
        else:
            high = mid - 1

    return best_prompt


def evaluate_ruler_core(
    *,
    generator: BaseGenerator,
    tasks: Sequence[str],
    num_samples: int,
    max_seq_length: int | None,
    output_dir: str | Path,
    seed: int = 17,
) -> Dict[str, object]:
    set_seed(seed)
    task_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    task_metrics: List[Dict[str, object]] = []

    budget = max_seq_length or generator.max_context_length
    budget = min(budget, generator.max_context_length)

    for task_name in tasks:
        if task_name not in TASK_BUILDERS:
            raise ValueError(f"Unsupported RULER core task: {task_name}")

        task_examples = []
        preds = []
        refs = []
        for sample_index in range(num_samples):
            example = TASK_BUILDERS[task_name](task_name, budget, generator, seed * 1000 + sample_index)
            prompt = example["input"] + example["answer_prefix"]
            prediction = normalize_prediction(
                generator.generate(prompt=prompt, max_new_tokens=int(RULER_TASKS[task_name]["tokens_to_generate"]))
            )
            task_examples.append({**example, "pred": prediction})
            preds.append(prediction)
            refs.append(example["outputs"])

        metric_name = RULER_METRICS[RULER_TASKS[task_name]["task"]]
        score = metric_name(preds, refs)
        nulls = sum(1 for pred in preds if len(pred.strip()) == 0)

        task_metrics.append(
            {
                "task": task_name,
                "task_type": RULER_TASKS[task_name]["task"],
                "num_samples": num_samples,
                "seq_budget": budget,
                "score": score,
                "null_predictions": f"{nulls}/{num_samples}",
                "model": generator.descriptor,
            }
        )

        for row_index, row in enumerate(task_examples):
            prediction_rows.append(
                {
                    "task": task_name,
                    "index": row_index,
                    "input": row["input"],
                    "answer_prefix": row["answer_prefix"],
                    "outputs": row["outputs"],
                    "pred": row["pred"],
                }
            )

    average_score = sum(metric["score"] for metric in task_metrics) / max(len(task_metrics), 1)
    summary = {
        "notes": RULER_NOTES,
        "model": generator.descriptor,
        "seq_budget": budget,
        "num_samples": num_samples,
        "tasks": list(tasks),
        "average_score": average_score,
        "task_metrics": task_metrics,
    }

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "ruler_core_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_root / "ruler_core_predictions.json").write_text(json.dumps(prediction_rows, indent=2), encoding="utf-8")
    write_csv(output_root / "ruler_core_metrics.csv", task_metrics)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a supported core subset of RULER-style long-context tasks.")
    parser.add_argument("--output-dir", default="auto_research_llm_ideas/results/ruler_core")
    parser.add_argument("--tasks", nargs="+", default=["niah_single_1", "vt", "cwe", "fwe"])
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--hf-model", type=str, default=None)
    args = parser.parse_args()

    if (args.checkpoint is None) == (args.hf_model is None):
        raise ValueError("Pass exactly one of --checkpoint or --hf-model")

    if args.checkpoint is not None:
        generator = LocalCheckpointGenerator(args.checkpoint)
    else:
        generator = HuggingFaceGenerator(args.hf_model)

    summary = evaluate_ruler_core(
        generator=generator,
        tasks=args.tasks,
        num_samples=args.num_samples,
        max_seq_length=args.max_seq_length,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
