# DGX Spark recipes

The catalog carries the eight text-generation recipes that NVIDIA's
`local-ai-inference-recipes` commit `12b25c664d8bea9a819f6e6e33d59267dd7528df`
lists for one DGX Spark. Each managed launch keeps the recipe's model-specific
vLLM flags, sets the served context to 65,536 tokens, and allows one active
sequence.

| Profile | NVIDIA recipe | Runtime |
| --- | --- | --- |
| `qwen38-27b-nvfp4-dgx-spark` | Qwen3.8 27B, DFlash | vLLM 0.28.0 |
| `gemma4-26b-a4b-nvfp4-dgx-spark` | Gemma 4 26B A4B, external MTP | vLLM 0.23.1 |
| `gemma4-31b-it-nvfp4-dgx-spark` | Gemma 4 31B, external draft | vLLM 0.23.1 |
| `nemotron3-super-120b-a12b-nvfp4-mtp-dgx-spark` | Nemotron 3 Super 120B A12B, MTP | vLLM 0.24.0 |
| `nemotron35-lightning-30b-a3b-nvfp4-dspark-dgx-spark` | Nemotron 3.5 Lightning 30B A3B, DSpark | vLLM 0.28.1 |
| `qwen35-122b-a10b-nvfp4-mtp-dgx-spark` | Qwen3.5 122B A10B, MTP | vLLM 0.28.0 |
| `qwen36-27b-nvfp4-mtp-dgx-spark` | Qwen3.6 27B, MTP | vLLM 0.28.0 |
| `qwen36-35b-a3b-nvfp4-mtp-dgx-spark` | Qwen3.6 35B A3B, MTP | vLLM 0.28.0 |

DiffusionGemma is left out. It generates images and does not serve the
OpenAI-compatible text API that AgentPerf measures.

Run each profile with `--framework vllm` in an environment where
`import vllm` works and the version matches the table.

## Status

We have not yet run these vLLM profiles on a DGX Spark. The catalog therefore:

- keeps every admission state `unqualified`;
- leaves measured token counts empty; and
- uses a provisional 100 GiB memory floor for seven profiles. The Qwen3.8 27B
  profile keeps its earlier memory model (32.5 GiB).

The external draft models (DFlash, the Gemma assistant models, and DSpark) are
named by repository only. vLLM fetches them at launch, and the catalog does
not pin their revision.

Open validation work:

- Launch each profile at 65,536 tokens with one active sequence.
- Run raw-completion probes at 16k, 32k, 48k, and 60k input tokens.
- Record startup allocation and peak unified memory. Replace the 100 GiB
  floors with measured formulas.
- Run the endpoint qualification probes before changing any admission state.

## Qwen3.8 Flash-Next

The catalog has no DGX Spark recipe for Qwen3.8 Flash-Next yet. No vLLM
release can serve it on one Spark: v0.30.0 reads the model's n-gram (PLE)
table only in FP8, and the FP8 checkpoint (123.6 GiB) does not fit. The
checkpoints that fit store the table in NVFP4.

The public
[MiaAI-Lab reference](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)
(vLLM's `qwen38-flash-next` development image plus nine patched files) does
serve it, and replayed in 1,016 s. It is not in the catalog because recipes
must run on a stock framework release.

## GB10 findings from SGLang profiles

These findings come from the SGLang 0.5.18 weights profiles on a GB10
(SM121, 119 GiB unified memory). They were taken at a 131,072-token context.
The recipes now launch at 65,536 tokens.

- **Fused-expert kernel.** SGLang picks its mixture-of-experts kernel from
  the device. On SM121 it picks a TensorRT-LLM kernel for NVFP4, loads the
  weights, and then fails the first forward pass. The
  `gemma4-26b-a4b-nvfp4` profile pins `flashinfer_cutlass` to avoid this.
- **gpt-oss-120b does not load.** Its floor is 79.8 GiB and its weights are
  60.8 GiB. The operating system kills the server part way through the
  weight read. Lowering SGLang's static memory fraction does not help, so the
  loader's working set is what does not fit. On a unified-memory device,
  treat a recipe's floor as necessary but not sufficient.
- **Shared memory is not free memory.** Another process can hold most of the
  pool. A Spark that already serves one model can hold 100 of its 119 GiB.
