# Contributing

Most contributions are new recipes. This guide sets what a recipe pull request
must contain before a review starts. For code changes, follow
[`AGENTS.md`](AGENTS.md) and run its local CI checks before you push.

[`recipes/README.md`](recipes/README.md#add-a-recipe) tells you how to write a
recipe file. This guide tells you what to prove about it.

## Recipe pull requests

A recipe pull request has four parts. The
[pull request template](.github/pull_request_template.md) has a section for
each one.

1. A recipe file in the established format.
2. Performance numbers from the full `agentperf-default-v1` replay.
3. The exact software you ran, so another person can get the same numbers.
4. An accuracy check, if the weights come from a source that is not trusted
   or are below 4 bits.

Put one model in a pull request. Recipes for the same model on different
hardware can go together.

## 1. Format

Copy the recipe nearest to yours and change only what differs. The loader
rejects most format errors, but these rules are for reviewers:

- **Path.** `recipes/<model>/<hardware>/<profile_id>.yaml`. `<hardware>` is
  `any`, `nvidia-cuda`, or a device folder that already exists. Ask in the
  pull request before you add a new device folder.
- **`profile_id`.** Lowercase, `<model>-<quantization>[-<speculation>][-<device>]`,
  for example `qwen36-27b-nvfp4-mtp-dgx-spark`. Recipes in `any/` and
  `nvidia-cuda/` have no device suffix. Use the device suffix that the other
  recipes in that folder use.
- **`display_name`.** `<Model> <size> <quantization> [<speculation>], <device> <backend or framework>`,
  for example `Qwen3.6 27B NVFP4 MTP, DGX Spark vLLM`. A portable GGUF recipe
  is `<Model> <size> <quantization> GGUF`.
- **Key order.** Keep the key order of the recipe you copied. Leave out an
  optional field that does not apply. Do not set it to its default.
- **Pins.** `hf_revision` and every `source_revision` are full 40-character
  commits. Every artifact has its `sha256` and `size_bytes` from
  `scripts/build_model_manifest.py`. SGLang and vLLM recipes pin one exact
  release in `runtime_versions`.
- **Downloadable.** Every artifact must download from its pinned repository
  and revision on a clean machine. A file that you converted or quantized
  yourself must be on Hugging Face before the review.
- **Comments.** Add a YAML comment to each `memory` value that is not read
  directly from the model config. Say how you measured it.
- **Digest.** Update `BUNDLED_RECIPES_DIGEST`, as `recipes/README.md` step 5
  shows.

## 2. Performance

Run the recipe with `managed-run` on the hardware it names. Use the default
settings: the full `agentperf-default-v1` replay (8 tasks, 168 turns), the
`exact` output policy, and the 65,536-token context. Do not use `aa-mini-v1`,
`--context-tokens`, or an attached server (`run`). Their numbers are not
comparable.

```console
uv run agentperf-local managed-run \
  --profile-id <profile_id> \
  --framework <llama-cpp | sglang | vllm> \
  --output-dir results/<profile_id>
```

The run is valid for review only when all of these are true:

- `summary.json` shows `"success": true` and empty `failed_turn_ids`,
  `length_finished_turn_ids`, and `short_output_warning_turn_ids`.
- `qualification.json` shows `"passed": true` (5 of 5 probes).
- `measurement.json` shows `producer.source_state` as `"clean"`. Run from a
  commit, not from a tree with local changes.

Report one row for each machine, with these values from `summary.json`:

| Field in the table | Source in `summary.json` |
| --- | --- |
| Output tok/s | `output_tokens_per_second` |
| End-to-end tok/s | `end_to_end_output_tokens_per_second` |
| TTFT p50 / p95 | `latency_distributions_ms.time_to_first_token` |
| E2E latency p50 / p95 | `latency_distributions_ms.e2e` |
| Replay time | `measured_duration_ms` |

If the recipe uses speculative decoding, also give the draft acceptance rate
from `deployment.log`. Then run the same model target-only on the same machine,
and report that row too. Reviewers use it to see if the draft helps.

Paste `summary.json` and `qualification.json` into the collapsed blocks in the
template. Do not paste `deployment.json`: it holds local paths.

## 3. Reproduction

Give enough for another person to get the same software. Take the first two
values from the run output, not from memory.

- **agentperf-local:** `producer.source_revision` from `measurement.json`.
- **Runtime:** `deployment.runtime.version` and
  `deployment.runtime.executable_sha256` from `deployment.json`.
- **llama.cpp:** the repository URL and full commit, and the build command
  with its CMake flags. If the commit is not on `ggml-org/llama.cpp` master,
  say which branch or pull request it is from.
- **vLLM or SGLang:** the Docker image by digest
  (`vllm/vllm-openai@sha256:…`), or the exact `pip install` command if you did
  not use Docker. The release must match `runtime_versions`.
- **Machine:** the device and its memory, the OS and kernel, and the driver
  with its CUDA, ROCm, or Metal version.
- **Converted files:** for each artifact you converted or quantized, the
  command and the commit of the tool that made it.

## 4. Accuracy

Weights from these Hugging Face sources are trusted. No accuracy check is
needed for them at 4 bits or more:

- `nvidia`
- `RadixArk`
- `unsloth`
- The model creator's own organization, such as `Qwen`, `google`, `openai`,
  or `inclusionAI`.

Any other source of target weights needs an accuracy check from you. This
applies to `hf_repository`, and to every artifact whose `source_repository`
holds target weights. It also applies to weights that you quantized yourself.
Draft models for speculative decoding do not need a check: the target model
verifies every draft token, so a bad draft lowers the acceptance rate, not the
accuracy.

Target weights below 4 bits, such as Q3_K_M, IQ3_XXS, Q2_K, or any 2-bit or
ternary format, need an accuracy check from Artificial Analysis. This applies
to every source, trusted or not. A maintainer runs the steps below before the
pull request can merge. If the weights are also from a source that is not
trusted, do your own check too.

To check accuracy:

1. Pick a reference: the same model, from a trusted source, at the same or a
   higher bit width. For weights below 4 bits, the reference is 4 bits or
   more.
2. Serve each model with the launch command in `deployment.json`
   (`deployment.command`). Use the recipe's thinking setting and the model
   card's sampling settings.
3. Run GPQA Diamond 4 times on each model with the same harness, pinned by
   commit. Report the mean and the 4 scores for each.
4. The recipe passes when its mean is no more than 3 points below the
   reference mean.

Attach the harness output for every run as a zip file. Give the harness
repository, its commit, and the full command.

## Review

A maintainer runs the recipe on matching hardware when one is available. A
recipe that is not complete gets a comment that names the missing part, and no
other review until that part is there.
