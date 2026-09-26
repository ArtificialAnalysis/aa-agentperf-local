# Models

The bundled catalog,
[`agentperf_local/data/model-candidates-v2.json`](../agentperf_local/data/model-candidates-v2.json),
has 29 managed profiles. That file is the source of truth. This page lists
what it contains.

Every profile:

- pins each file by Hugging Face revision, size, and SHA-256;
- serves the 65,536-token benchmark context with one active sequence; and
- has the admission state `unqualified`. No profile has passed the full
  qualification yet.

Check what your machine can run before you download anything:

```console
uv run agentperf-local deployment-options --profile-id gemma4-12b-it-q4-0
```

It prints the detected accelerator, the installed frameworks, the profile's
memory floor, and whether the profile fits. Then run it:

```console
uv run agentperf-local managed-run \
  --profile-id gemma4-12b-it-q4-0 \
  --framework llama-cpp \
  --output-dir results/gemma4-12b
```

The memory column below is the floor at the full 65,536-token context, in GiB.
The launcher recomputes it for a reduced `--context-tokens` value. The recipe
evidence pages report measurements taken at 131,072 tokens, before the
benchmark context moved to 65,536.

## Portable GGUF profiles

These run on any supported accelerator with a llama.cpp build for its backend.

| Profile | Model | Runtime | Speculation | Devices | Memory |
| --- | --- | --- | --- | --- | ---: |
| `gemma4-12b-it-q4-0` | Google Gemma 4 12B IT QAT Q4_0 | llama.cpp | none | CUDA, ROCm, Metal | 9.3 |
| `gemma4-26b-a4b-q4-0` | Google Gemma 4 26B A4B IT QAT Q4_0 | llama.cpp | none | CUDA, ROCm, Metal | 16.4 |
| `qwen38-27b-q4-k-m` | Qwen3.8 27B Q4_K_M | llama.cpp | none | CUDA, ROCm, Metal | 21.6 |

## RTX 5090 profiles

These come from NVIDIA's llama.cpp recipes for GeForce RTX. See
[RTX 5090 recipes](recipes/rtx-5090.md) for long-context results.

| Profile | Model | Runtime | Speculation | Devices | Memory |
| --- | --- | --- | --- | --- | ---: |
| `qwen38-27b-q4-k-m-mtp` | Qwen3.8 27B Q4_K_M | llama.cpp | MTP | CUDA | 22.6 |
| `gemma4-26b-a4b-q4-k-m-mtp-rtx5090` | Gemma 4 26B A4B Q4_K_M | llama.cpp | MTP, external draft | CUDA | 20.5 |
| `muse-glimmer-30b-q4-k-m-dflash-rtx5090` | Muse Glimmer 30B Q4_K_M | llama.cpp | DFlash, external draft | CUDA | 20.3 |
| `nemotron35-lightning-30b-a3b-q4-k-m-dflash-rtx5090` | Nemotron 3.5 Lightning 30B A3B Q4_K_M | llama.cpp | DFlash, external draft | CUDA | 27.9 |
| `qwen35-9b-q4-k-m-mtp-rtx5090` | Qwen3.5 9B Q4_K_M | llama.cpp | MTP | CUDA | 11.3 |
| `qwen36-27b-q4-k-m-mtp-rtx5090` | Qwen3.6 27B Q4_K_M | llama.cpp | MTP | CUDA | 22.8 |
| `qwen36-35b-a3b-q4-k-m-mtp-rtx5090` | Qwen3.6 35B A3B Q4_K_M | llama.cpp | MTP | CUDA | 24.5 |

## Strix Halo profiles

These reuse the RTX 5090 files with AMD-specific settings. See
[Strix Halo recipes](recipes/strix-halo.md).

| Profile | Model | Runtime | Speculation | Devices | Memory |
| --- | --- | --- | --- | --- | ---: |
| `gemma4-26b-a4b-q4-k-m-mtp-strix-halo` | Gemma 4 26B A4B Q4_K_M | llama.cpp ROCm | MTP, external draft | AMD | 20.5 |
| `muse-glimmer-30b-q4-k-m-dflash-strix-halo` | Muse Glimmer 30B Q4_K_M | llama.cpp ROCm | DFlash, external draft | AMD | 20.3 |
| `nemotron35-lightning-30b-a3b-q4-k-m-dflash-strix-halo` | Nemotron 3.5 Lightning 30B A3B Q4_K_M | llama.cpp ROCm | DFlash, external draft | AMD | 27.9 |
| `qwen35-9b-q4-k-m-mtp-strix-halo` | Qwen3.5 9B Q4_K_M | llama.cpp ROCm | MTP | AMD | 11.3 |
| `qwen36-27b-q4-k-m-mtp-strix-halo` | Qwen3.6 27B Q4_K_M | llama.cpp ROCm | MTP | AMD | 22.8 |
| `qwen36-35b-a3b-q4-k-m-mtp-strix-halo` | Qwen3.6 35B A3B Q4_K_M | llama.cpp Vulkan | MTP | AMD | 24.5 |
| `qwen38-27b-q4-k-m-mtp-strix-halo` | Qwen3.8 27B Q4_K_M | llama.cpp ROCm | MTP | AMD | 22.6 |
| `qwen38-flash-next-iq4-nl-mtp-strix-halo` | Qwen3.8 Flash-Next IQ4_NL | llama.cpp ROCm, pwilkin fork | MTP, external draft | AMD | 85.5 |

## M5 Pro profile

See [M5 Pro recipe](recipes/m5-pro.md).

| Profile | Model | Runtime | Speculation | Devices | Memory |
| --- | --- | --- | --- | --- | ---: |
| `qwen38-27b-q4-k-m-mtp-m5-pro` | Qwen3.8 27B Q4_K_M | llama.cpp Metal | MTP | Apple silicon | 22.9 |

## SGLang weights profiles

These serve NVFP4 or MXFP4 safetensors on CUDA. Run them where `import sglang`
works, with `--framework sglang`.

| Profile | Model | Runtime | Speculation | Devices | Memory |
| --- | --- | --- | --- | --- | ---: |
| `gemma4-26b-a4b-nvfp4` | Google Gemma 4 26B A4B NVFP4, NVIDIA export | SGLang 0.5.18 | none | CUDA | 29.3 |
| `gpt-oss-120b-mxfp4` | OpenAI gpt-oss-120b MXFP4 | SGLang 0.5.18 | none | CUDA | 76.2 |

## DGX Spark profiles

These reproduce NVIDIA's DGX Spark vLLM recipes. Run them where `import vllm`
works, with `--framework vllm`. See [DGX Spark recipes](recipes/dgx-spark.md).

| Profile | Model | Runtime | Speculation | Devices | Memory |
| --- | --- | --- | --- | --- | ---: |
| `qwen38-27b-nvfp4-dgx-spark` | Qwen3.8 27B NVFP4, RadixArk export | vLLM 0.28.0 | DFlash | CUDA | 32.5 |
| `gemma4-26b-a4b-nvfp4-dgx-spark` | Google Gemma 4 26B A4B NVFP4 | vLLM 0.23.1 | external draft | CUDA | 100.0 |
| `gemma4-31b-it-nvfp4-dgx-spark` | Google Gemma 4 31B IT NVFP4 | vLLM 0.23.1 | external draft | CUDA | 100.0 |
| `nemotron3-super-120b-a12b-nvfp4-mtp-dgx-spark` | NVIDIA Nemotron 3 Super 120B A12B NVFP4 | vLLM 0.24.0 | MTP | CUDA | 100.0 |
| `nemotron35-lightning-30b-a3b-nvfp4-dspark-dgx-spark` | NVIDIA Nemotron 3.5 Lightning 30B A3B NVFP4 | vLLM 0.28.1 | DSpark | CUDA | 100.0 |
| `qwen35-122b-a10b-nvfp4-mtp-dgx-spark` | Qwen3.5 122B A10B NVFP4 | vLLM 0.28.0 | MTP | CUDA | 100.0 |
| `qwen36-27b-nvfp4-mtp-dgx-spark` | Qwen3.6 27B NVFP4 | vLLM 0.28.0 | MTP | CUDA | 100.0 |
| `qwen36-35b-a3b-nvfp4-mtp-dgx-spark` | Qwen3.6 35B A3B NVFP4 | vLLM 0.28.0 | MTP | CUDA | 100.0 |

## Evidence levels

Each profile records how far its recipe has been checked on each device.

| Level | Meaning | Profiles |
| --- | --- | --- |
| `official-gguf-backend` | The model publisher ships the GGUF for a standard llama.cpp backend. | Gemma 4 Q4_0 |
| `upstream-gguf-conversion` | A third-party GGUF conversion on a standard backend. | `qwen38-27b-q4-k-m` |
| `vendor-sglang-recipe` | The vendor publishes an SGLang recipe for the checkpoint. | SGLang weights profiles |
| `vendor-hardware-listed-recipe-confirm` | NVIDIA lists the recipe for the device. We have not run it yet. | Seven DGX Spark profiles |
| `upstream-end-to-end` | The upstream recipe reports an end-to-end run. | `qwen38-27b-nvfp4-dgx-spark` |
| `local-end-to-end` | We ran the long-context probes on the device. | RTX 5090 profiles |
| `local-inference-screen` | We ran short inference screens on the device. | Strix Halo and M5 Pro profiles |

## Caveats

- **Runtime pins are exact.** A weights profile names the one runtime release
  it was checked against, and the launcher refuses any other. SGLang 0.5.17
  served these FP4 checkpoints, passed every readiness check, and generated
  nonsense. Only the qualification probes caught it. The same checkpoints are
  correct under 0.5.18.
- **Unified memory needs more than the floor.** The floor is measured on a
  device with its own memory. On a unified-memory device the operating system
  and the loader share the same pool. `gpt-oss-120b-mxfp4` (79.8 GiB floor at
  131,072 tokens) does not load on a 119 GiB unified-memory GB10.
- **Metal will not give one process all host memory.** A Mac that passes the
  memory check can still fail to load a profile near its floor.
- **The FP4 profiles need NVIDIA hardware** with FP4 kernel support.
- **Quantizations differ between devices.** Q4_K_M, Q4_0, NVFP4, and MXFP4
  are four-bit formats, but they are not identical weights. Compare runs of
  the same profile.
- **The Qwen3.8 27B NVFP4 export is third-party.** Neither Qwen nor NVIDIA
  publishes one. The RadixArk export's provenance is on its model card.
- **Tool-call parsers can fail probes.** Under SGLang, Qwen3.8 27B NVFP4 returns
  only the first of two parallel tool calls with every Qwen tool-call parser.
  With no parser the raw text holds both calls. The probe reports what an
  OpenAI-compatible client receives.
- **llama.cpp profiles do not pin a build.** The recipe pages name the build we
  tested (b11026). Put a matching `llama-server` on `PATH`.
