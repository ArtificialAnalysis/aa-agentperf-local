# Recipes

A recipe is everything `managed-run` needs to serve one model on one kind of
hardware: the pinned files, the runtime, the launch settings, and the memory
floor. Every YAML file in this folder is one recipe. The tool loads all of
them, and the TUI lists them.

```text
recipes/<model>/<hardware>/<profile_id>.yaml
```

- `<model>` is the base model, such as `qwen38-27b`.
- `<hardware>` is where the recipe was built for: `any` for portable GGUF
  recipes that run on CUDA, ROCm, or Metal; `nvidia-cuda` for any NVIDIA GPU
  with FP4 support; or a device, such as `rtx-5090`, `dgx-spark`,
  `strix-halo`, or `m5-pro`.
- The file name is the recipe's `profile_id`. You pass it to `--profile-id`.

A recipe holds only what the run uses: its name, the files to download, the
devices it may run on, the memory shape for the memory check, and the server
launch settings. The server serves the model as `<profile_id>-<nonce>`. Leave
out an optional field, such as `vllm` or `moe_runner_backend`, when it does not
apply.

The TUI names a recipe from two fields and its folder:

- `model_name` is the base model, such as `Qwen3.8 27B`. Every recipe in one
  model folder uses the same name.
- `quantization` is the weight format the files use, such as `Q4_K_M`,
  `UD-Q4_K_M`, or `NVFP4`.
- The hardware it was built for is its hardware folder, such as `rtx-5090` or
  `any`; a recipe does not repeat it. The TUI shows a label for the folders the
  catalog ships, set in [`labels.py`](../agentperf_local/tui/labels.py), and
  the folder name for any other.

The speed-up it shows, such as MTP or DFlash, comes from `speculation_policy`.

## Run a recipe

Check what your machine can run before you download anything:

```console
uv run agentperf-local deployment-options --profile-id gemma4-12b-it-q4-0
```

It prints the detected accelerator, the installed frameworks, the recipe's
memory floor, and whether the recipe fits. Then run it:

```console
uv run agentperf-local managed-run \
  --profile-id gemma4-12b-it-q4-0 \
  --framework llama-cpp \
  --output-dir results/gemma4-12b
```

- **llama.cpp** recipes do not pin a build. Put a `llama-server` on `PATH`.
- **SGLang** and **vLLM** recipes pin one exact release in
  `runtime_versions`. Run them where `import sglang` or `import vllm` works.

To try recipes from another folder, pass `--recipes PATH`. The TUI marks
them as not from the Artificial Analysis catalog.

## Add a recipe

1. Copy the recipe closest to yours into `recipes/<model>/<hardware>/`, and
   name the file after the new `profile_id`.
2. Set `as_of` to today's date, in quotes: `as_of: '2026-09-27'`.
3. Pin every file. This prints the size and SHA-256 of each file for the
   `artifacts` list:

   ```console
   uv run scripts/build_model_manifest.py \
     --repository owner/name --revision <40-character commit> --include model.gguf
   ```

4. Set the `memory` shape. `scripts/estimate_kv_budget.py` computes the KV
   cache from the model config. The launcher computes the memory floor from
   this shape, so measure `runtime_overhead_bytes` on real hardware.
5. Run the tests. `test_bundled_catalog_matches_its_pinned_digest` fails and
   prints the new digest. Paste it into `BUNDLED_RECIPES_DIGEST` in
   `agentperf_local/deployment/catalog.py`.

   ```console
   uv run pytest tests/test_model_catalog.py tests/test_model_candidates.py
   ```

6. Run it once with `managed-run` on the hardware. Put the result summary
   and your hardware in the pull request.

[`recipe-v2.schema.json`](../docs/schemas/recipe-v2.schema.json) describes
every field. The loader in
[`catalog.py`](../agentperf_local/deployment/catalog.py) is stricter than the
schema. Its error messages name the field that is wrong.

## Caveats

- **Runtime pins are exact.** SGLang 0.5.17 served the FP4 checkpoints,
  passed every readiness check, and generated nonsense. Only the
  qualification probes caught it. The launcher refuses any release other than
  the pinned one.
- **Unified memory needs more than the floor.** On a unified-memory device
  the operating system and the loader share the same pool.
  `gpt-oss-120b-mxfp4` does not load on a 119 GiB GB10.
- **Metal will not give one process all host memory.** A Mac that passes the
  memory check can still fail to load a recipe near its floor.
- **Quantizations differ between devices.** Q4_K_M, Q4_0, NVFP4, and MXFP4
  are four-bit formats, but they are not identical weights. Compare runs of
  the same recipe.
