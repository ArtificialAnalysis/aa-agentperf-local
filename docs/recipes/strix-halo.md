# Strix Halo recipes

The bundled catalog has eight `-strix-halo` profiles. Seven of them reuse the
exact target and draft files of the matching RTX 5090 profile. Their settings
are the fastest configuration seen in a short screen. They are not a proven
optimum, and none of them has run the full eight-task replay.

The eighth, `qwen38-flash-next-iq4-nl-mtp-strix-halo`, needs a llama.cpp fork.
See [Qwen3.8 Flash-Next](#qwen38-flash-next). It has run the full replay as a
managed run.

All eight keep the admission state `unqualified`.

The measurements on this page were taken at a 131,072-token context. The
recipes now launch at 65,536 tokens.

## Test machine

- AMD Ryzen AI Max+ 395 with Radeon 8060S (`gfx1151`), 16 cores and 32 threads.
- 128 GB RAM. Linux reports about 125 GiB usable. No swap.
- GTT allocation limit of 94.2 GiB. Vulkan reports 96,966 MiB including the
  framebuffer allocation.
- ROCm 7.14 and Mesa Vulkan 25.0.7.
- The platform profile, CPU governor, and CPU energy preference were already
  set to `performance`. The GPU performance level was `auto`. We did not change
  them.

Both backends use upstream llama.cpp b11026 (commit `b49650adb`):

| Archive | SHA-256 |
| --- | --- |
| `llama-b11026-bin-ubuntu-rocm-10.0-x64.tar.gz` | `9b0efc311eca415b7edc5d0af516754799e485e53eb68f57aa621a1e6a4ad1a7` |
| `llama-b11026-bin-ubuntu-vulkan-x64.tar.gz` | `1b40310bf4d47c2c84853ebb4ccaf4dcbd992596cd1c2f610be6a0532a874708` |

The ROCm archive name says ROCm 10.0, but `ldd` shows that it loads the host's
ROCm 7.14 libraries. Report the runtime as ROCm 7.14.

This release rejects `--no-mmap`. Use `--load-mode none` for the non-mmap path.

## Selected profiles

| Profile | Backend | Ubatch | Speculation | Target backend sampling | Three-request E2E |
| --- | --- | ---: | --- | --- | ---: |
| `gemma4-26b-a4b-q4-k-m-mtp-strix-halo` | ROCm | 512 | MTP 3 | on | 11.16 s |
| `muse-glimmer-30b-q4-k-m-dflash-strix-halo` | ROCm | 2048 | DFlash 15 | off | 34.36 s |
| `nemotron35-lightning-30b-a3b-q4-k-m-dflash-strix-halo` | ROCm | 2048 | DFlash 7 | off | 14.21 s |
| `qwen35-9b-q4-k-m-mtp-strix-halo` | ROCm | 2048 | MTP 2 | off | 14.17 s |
| `qwen36-27b-q4-k-m-mtp-strix-halo` | ROCm | 512 | MTP 3 | off | 34.15 s |
| `qwen36-35b-a3b-q4-k-m-mtp-strix-halo` | Vulkan | 512 | MTP 2 | on | 10.89 s |
| `qwen38-27b-q4-k-m-mtp-strix-halo` | ROCm | 512 | MTP 2 | on | 41.06 s |

Three-request E2E is the sum of three uncached requests with 256 output tokens
each. It includes HTTP, prefill, and generation. It excludes startup, warm-up,
and shutdown. It is not a replay time.

Every profile also uses a 131,072-token context, one slot, full GPU offload,
flash attention, `--load-mode none`, `--fit off`, batch 2048, 16 CPU threads,
and `--cache-ram 0`. External drafts are fully offloaded and use draft backend
sampling. The launcher pins `--device ROCm0` or `--device Vulkan0`.

To run one, put the matching ROCm or Vulkan b11026 build on `PATH` as
`llama-server`. The catalog does not download or switch runtime builds. The
Vulkan profile needs a host with one GPU and refuses `--device N`, because HIP
device visibility does not control Vulkan device order.

## Qwen3.8 Flash-Next

`qwen38-flash-next-iq4-nl-mtp-strix-halo` serves Qwen3.8 Flash-Next (177B
parameters) with native MTP speculation at depth 3.

Upstream llama.cpp supports this model, but not its MTP draft head. The recipe
therefore needs the public `strix-halo` branch of
[pwilkin/llama.cpp](https://github.com/pwilkin/llama.cpp). Build it with the
fork's own installer, which pins every commit:

```console
bash <(curl -fsSL https://raw.githubusercontent.com/pwilkin/strix-halo/main/install-flash-next.sh) --model-dir ~/flash-next-models
mkdir -p ~/flash-next-bin
ln -s ~/.local/bin/llama-server-strix-halo ~/flash-next-bin/llama-server
PATH=~/flash-next-bin:$PATH uv run agentperf-local managed-run \
  --profile-id qwen38-flash-next-iq4-nl-mtp-strix-halo --framework llama-cpp --output-dir results/flash-next
```

The installer builds pwilkin/llama.cpp `b0f31f5` against pwilkin/rocm-systems
`7dda3ac` (a ROCr and HIP runtime) in your home directory. It needs the build
packages it lists, but no root access for the build itself. The
`llama-server-strix-halo` wrapper sets the runtime library path and the ROCm
environment the fork needs. Put the wrapper on `PATH` as `llama-server`. A
plain `llama-server` binary from the fork runs its generic paths only.

The catalog downloads its own verified copy of the weights. The installer's
copy in `--model-dir` is only needed if you want to run the fork's own
launcher.

The weights are the fork author's IQ4_NL export,
[ilintar/qwen3.8-flash-next-gguf-strix-halo](https://huggingface.co/ilintar/qwen3.8-flash-next-gguf-strix-halo),
plus a shared-embedding MTP draft head. The launch uses the fork launcher's
settings:

- a batch and micro-batch of 16,384 tokens;
- F16 KV cache, flash attention, and `--fit off`;
- `--load-mode none --lazy-mode on-direct`.

The lazy mode reads the 28.8 GB per-layer-embedding table from disk on demand,
so it is never resident. The recipe's memory floor leaves those bytes out
(`lazy_read_bytes`). Without the lazy mode the model does not fit the 94.2 GiB
GPU pool.

### Result

One managed full replay on the test machine above, 2026-09-25:

| Metric | Value |
| --- | ---: |
| End-to-end replay | 883.2 s |
| Turns | 168 of 168 |
| Qualification probes | passed |
| Output tokens per second | 41.8 |
| Time to first token, p50 | 1.06 s |
| Uncached prompt tokens | 192,069 |

The 14.10 GB of runtime overhead in the memory floor is measured: GPU memory
in use after a full replay, minus the resident weights, KV cache, and recurrent
state. The catalog rounds it up to 16 GB.

## Evidence

Qwen3.5 9B is the only profile with long-context evidence:

- ROCm MTP depth 2 processed uncached prompts of 32k, 64k, 96k, and 120k input
  tokens. The 120k request took 223.57 s.
- Retrieval probes passed at 8k and about 120k input tokens. The 120k probe
  took 249.58 s and generated 958 tokens.
- All five endpoint qualification probes passed.
- Without speculation it decoded about 37 tokens/s on short requests. With MTP
  depth 2 it decoded 62 to 67 tokens/s.

The other six profiles have short-request evidence only. A configured 131k
context does not prove that they complete long-context inference or tool use.

The screen covered 61 short configurations (183 requests) with 33 min 49 s of
request time. Capacity and retrieval requests took another 21 min 58 s.

Evidence files:

- [Screening timings](../recipe-evidence/strix-halo-m5-pro-screening.json)
- [Qwen3.5 9B qualification](../recipe-evidence/qwen35-9b-amd-qualification.json).
  It carries the source RTX profile ID. The AMD profile uses the same files and
  settings.

## Caveats

- Some early screens ran while stopped containers still held GPU memory. Treat
  these numbers as functional evidence, not isolated peak memory or repeated
  best performance.
- The catalog keeps the RTX 5090 memory budgets for these files. They are
  planning estimates, not measured AMD peaks.
- Q4_K_M, ROCmFP4, MLX four-bit, and NVFP4 are the same bit-width class. They
  are not identical weights.
- The Nemotron external draft is NVFP4, not Q4_K_M.
- A first retrieval probe capped output at 256 tokens. At 120k input the model
  was still reasoning when the cap hit, so that result is inconclusive. Later
  probes allowed 1,024 tokens.

## Screened but not selected

These ROCmFP4 checkpoints need the [ROCmFPX fork](https://github.com/ROCmFPX/ROCmFPX).
They are excluded until clean repeated runs exist.

| Base model | Checkpoint | Revision | Difference |
| --- | --- | --- | --- |
| Qwen3.8 27B | [pugant/Qwen3.8-27B-MTP-Q4_0_ROCMFP4_STRIX_LEAN](https://huggingface.co/pugant/Qwen3.8-27B-MTP-Q4_0_ROCMFP4_STRIX_LEAN) | `1f98e4da8ef4af6af126e1ebbf1704e15d5d170a` | Mixed four-bit, 4.34 effective bits per weight, integrated MTP |
| Qwen3.6 35B A3B | [plunderstruck/Qwen3.6-35B-A3B-MTP-ROCmFP4-GGUF](https://huggingface.co/plunderstruck/Qwen3.6-35B-A3B-MTP-ROCmFP4-GGUF) | `fe537448c6eb259e6591fea8b9ed320a64ef4856` | ROCmFP4 body, F16 embeddings, Q6_K output, F32 routers, integrated MTP |

[llama.cpp issue 27306](https://github.com/ggml-org/llama.cpp/issues/27306)
reports long-prefill Vulkan MTP failures on `gfx1151` with an older build.
Short-prompt success does not prove that long contexts work.
