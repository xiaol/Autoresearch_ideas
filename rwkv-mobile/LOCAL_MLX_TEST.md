# Local MLX Test Notes

This clone was tested locally under:

- workspace: `/Users/xiaol/x/PaperX/auto_research_llm_ideas/rwkv-mobile`
- host: macOS Apple Silicon environment
- goal: verify that the `mlx` backend builds and can load a tiny RWKV checkpoint

## What worked

- The repository cloned successfully.
- A minimal MLX-only CMake configure succeeded.
- The examples built successfully with `ENABLE_MLX_BACKEND=ON`.
- A tiny local RWKV `.pth` checkpoint was converted to safetensors.

## Configure command

```bash
/Users/xiaol/x/PaperX/.localdeps/cmake/data/bin/cmake \
  -S auto_research_llm_ideas/rwkv-mobile \
  -B auto_research_llm_ideas/rwkv-mobile/build-mlx \
  -DENABLE_MLX_BACKEND=ON \
  -DENABLE_WEBRWKV_BACKEND=OFF \
  -DENABLE_NCNN_BACKEND=OFF \
  -DENABLE_LLAMACPP_BACKEND=OFF \
  -DENABLE_MNN_BACKEND=OFF \
  -DENABLE_QNN_BACKEND=OFF \
  -DENABLE_MTK_NP7_BACKEND=OFF \
  -DENABLE_COREML_BACKEND=OFF \
  -DENABLE_SERVER=OFF \
  -DBUILD_EXAMPLES=ON \
  -DCMAKE_BUILD_TYPE=Release
```

## Build command

```bash
/Users/xiaol/x/PaperX/.localdeps/cmake/data/bin/cmake \
  --build auto_research_llm_ideas/rwkv-mobile/build-mlx -j 4
```

## Conversion command used

```bash
python auto_research_llm_ideas/rwkv-mobile/converter/convert_rwkv_to_safetensors.py \
  --input /Users/xiaol/x/PaperX/Attention-Residuals/RWKV-LM/RWKV-v4neo/math_demo/rwkv-200.pth \
  --output /Users/xiaol/x/PaperX/auto_research_llm_ideas/rwkv-mobile/tmp/rwkv-200.st
```

## Runtime test command

```bash
./examples/simple_benchmark \
  /Users/xiaol/x/PaperX/auto_research_llm_ideas/rwkv-mobile/tmp/rwkv-200.st \
  mlx
```

## Result

The first failure was during MLX Metal initialization before model evaluation:

```text
NSRangeException: *** -[__NSArray0 objectAtIndex:]: index 0 beyond bounds for empty array
...
mlx::core::metal::Device::Device()
...
mlx_initialize
...
```

After fixing metallib staging, MLX initialization succeeded, but model loading
failed with:

```text
The file “config.json” couldn’t be opened.
... /rwkv-200.st/config.json
... Not a directory
```

## Interpretation

- The MLX backend is compiled and linked correctly.
- The initial metallib issue was a packaging bug.
- After that was fixed, the next failure showed that this MLX backend expects a
  model directory containing `config.json`, not a single `.st` file.
- Symbol inspection of `libMLXModelFFI.a` suggests support for models like
  Llama, Gemma, Qwen2, and Phi3, but not obvious RWKV symbols.
- Therefore the current MLX backend in this repo should not be assumed to load a
  raw RWKV checkpoint directly.

## Best next steps

- Retry the same binary in a less restricted local shell outside the current app sandbox.
- If the MLX runtime initializes there, continue with:
  - `examples/test_clear_states`
  - `examples/test_chat_save_state`
  - `examples/test_chat_load_state`
- If MLX still fails outside the sandbox, test a known-good MLX model artifact
  expected by the bundled `libMLXModelFFI.a`, since the exact accepted model
  format is not documented in this repo.

## HF RWKV7 Workaround

For Hugging Face RWKV7 repos that use remote code such as
`fla-hub/rwkv7-0.1B-g1`, stock `mlx_lm convert` fails in this local venv
because HF config/tokenizer loading imports `fla`. A local workaround script is
available:

```bash
/Users/xiaol/x/PaperX/auto_research_llm_ideas/mlx_lm_test/.venv/bin/python \
  converter/convert_hf_rwkv7_to_mlx.py \
  --hf-path fla-hub/rwkv7-0.1B-g1 \
  --mlx-path tmp/rwkv7-0.1B-g1-mlx
```

The converted directory can then be tested with:

```bash
/Users/xiaol/x/PaperX/auto_research_llm_ideas/mlx_lm_test/.venv/bin/python \
  -m mlx_lm generate \
  --model tmp/rwkv7-0.1B-g1-mlx \
  --prompt "Hello" \
  --max-tokens 16 \
  --temp 0.0 \
  --verbose True \
  --trust-remote-code
```

Validated locally with:

- `fla-hub/rwkv7-0.1B-g1`
- `fla-hub/rwkv7-0.4B-g1`

The `0.4B` model converted to `tmp/rwkv7-0.4B-g1-mlx-script` and produced a
successful `mlx_lm generate` smoke test with output starting:

```text
Hello! How can I assist you today?
```
