#!/usr/bin/env python3

"""Convert a Hugging Face RWKV7 repo to an MLX directory without HF model remote code.

This is a workaround for repos like `fla-hub/rwkv7-0.1B-g1` where:
1. `mlx_lm` can load the RWKV7 weights natively, but
2. stock `mlx_lm.convert` fails because tokenizer/config loading goes through
   Hugging Face `auto_map` remote code that imports `fla`.

The script keeps the original tokenizer assets, writes MLX weights/config, and
removes model-side `auto_map` from the saved `config.json` so `mlx_lm generate`
does not try to import the HF RWKV modeling code again.
"""

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from mlx_lm.utils import load_model, save_config, save_model


TOKENIZER_FILES = [
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "rwkv_vocab_v20230424.txt",
]

EXTRA_COPY_GLOBS = [
    "*.py",
    "generation_config.json",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a HF RWKV7 repo to MLX format without HF model remote code."
    )
    parser.add_argument(
        "--hf-path",
        required=True,
        help="Local model directory or Hugging Face repo id.",
    )
    parser.add_argument(
        "--mlx-path",
        required=True,
        help="Output directory for the converted MLX model.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face revision.",
    )
    parser.add_argument(
        "--keep-config-auto-map",
        action="store_true",
        help="Keep `auto_map` in the saved config.json. Default removes it.",
    )
    return parser.parse_args()


def resolve_source(hf_path: str, revision: str | None) -> Path:
    src = Path(hf_path)
    if src.exists():
        return src.resolve()

    allow_patterns = [
        "*.json",
        "*.py",
        "*.txt",
        "model*.safetensors",
        "*.jinja",
    ]
    return Path(
        snapshot_download(
            hf_path,
            revision=revision,
            allow_patterns=allow_patterns,
        )
    )


def load_json(path: Path) -> dict:
    with open(path, "r") as handle:
        return json.load(handle)


def copy_if_exists(src: Path, dst: Path):
    if src.exists():
        shutil.copy(src, dst)


def main():
    args = parse_args()

    src = resolve_source(args.hf_path, args.revision)
    dst = Path(args.mlx_path).resolve()
    if dst.exists():
        raise ValueError(f"Output path already exists: {dst}")

    config = load_json(src / "config.json")
    model_type = config.get("model_type")
    if model_type != "rwkv7":
        raise ValueError(
            f"Expected config.json model_type 'rwkv7', found {model_type!r} at {src}"
        )

    print(f"[INFO] Source: {src}")
    print("[INFO] Loading RWKV7 weights with mlx-lm native model support")
    model, config = load_model(src, lazy=True)

    dst.mkdir(parents=True)
    (dst / "__init__.py").write_text("", encoding="utf-8")

    print("[INFO] Saving MLX weights")
    save_model(dst, model, donate_model=True)

    if not args.keep_config_auto_map:
        config.pop("auto_map", None)
    print("[INFO] Saving config.json")
    save_config(config, dst / "config.json")

    print("[INFO] Copying tokenizer and helper files")
    for file_name in TOKENIZER_FILES:
        copy_if_exists(src / file_name, dst)
    for pattern in EXTRA_COPY_GLOBS:
        for path in src.glob(pattern):
            shutil.copy(path, dst)

    print(f"[INFO] Done: {dst}")
    if args.keep_config_auto_map:
        print(
            "[INFO] config.json kept `auto_map`; mlx-lm may still try to import HF model remote code."
        )
    else:
        print(
            "[INFO] Removed model-side `auto_map` from config.json; use "
            "`mlx_lm generate --trust-remote-code` so HF can still load the custom tokenizer."
        )


if __name__ == "__main__":
    main()
