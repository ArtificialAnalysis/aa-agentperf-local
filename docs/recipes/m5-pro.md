# M5 Pro recipe

The bundled profile `qwen38-27b-q4-k-m-mtp-m5-pro` serves Qwen3.8 27B Q4_K_M
with llama.cpp Metal and MTP depth 4. It passed a functional check and the five
endpoint probes. It has not run long-context inference or the full replay, so
its admission state stays `unqualified`.

The measurements on this page were taken at a 131,072-token context. The
recipe now launches at 65,536 tokens.

## Test machine

- Apple M5 Pro, 18 CPU cores, 20 GPU cores, Metal 4.
- 64 GiB unified memory. llama.cpp reports 53,084 MiB of Metal-addressable memory.
- No GPU memory limit or fan overrides.
- High Power mode on battery, not AC power.
- System swap was nonzero and stable during the sweep.

## Recipe

- Runtime: llama.cpp b11026 (`b49650adb31f2e49a0d76113aeb1792134fd8413`).
  `llama-b11026-bin-macos-arm64.tar.gz` has SHA-256
  `dbbfc7bd866a2594fea3bb10b56c5b205fffdaba23695f89da053b76e9a456d2`.
- Model: `unsloth/Qwen3.8-27B-GGUF` at revision
  `f1bfb127c64f7072bdd2cad55f258b9c8b2910fe`, file `Qwen3.8-27B-Q4_K_M.gguf`
  (17,106,775,008 bytes, SHA-256
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`). This is
  the same target file as the RTX 5090 and Strix Halo profiles.
- The embedded MTP head needs no draft file.
- One slot, 131,072-token context, full GPU offload, flash attention, mmap
  loading, batch 2048, ubatch 512, six CPU threads, no prompt cache, and
  `--spec-type draft-mtp --spec-draft-n-max 4`.
- The launcher selects `MTL0` and turns on draft backend sampling. Target
  backend sampling stays off.

Put the b11026 Metal build on `PATH` as `llama-server` before you choose the
profile.

## Depth selection

| MTP depth | Three-request E2E | Whole functional test | Endpoint probes |
| --- | ---: | ---: | --- |
| 1 | 46.30 s | 67.08 s | 5/5 passed |
| 2 | 49.35 s | 74.48 s | 5/5 passed |
| 3 | 46.46 s | 64.83 s | 5/5 passed |
| 4 | 46.08 s | 63.48 s | 5/5 passed |

Three-request E2E sums three uncached requests of 256 output tokens each. The
whole-test time adds startup, warm-up, the five probes, and shutdown. It
excludes downloads. Depths 1, 3, and 4 are within 1%, so depth 4 is not a
proven winner.

At depth 4 the server generated 17.3 to 21.0 tokens/s. All 66 layers ran on
Metal. The runtime reported 25,981 MiB of Metal allocation at shutdown. That is
an allocation report, not a peak-memory guarantee.

The catalog memory plan keeps the RTX recipe's attention shape and runtime
reserve. It rounds the depth-4 recurrent state (748.12 MiB) up to 749 MiB. This
is a planning budget, not a measured long-context peak.

Evidence files:

- [Screening timings](../recipe-evidence/strix-halo-m5-pro-screening.json)
- [Five-probe qualification](../recipe-evidence/qwen38-27b-m5-qualification.json).
  It carries the source profile ID. The Mac profile uses the same files and
  settings.

## Alternatives screened

An earlier screen compared other Qwen3.8 27B four-bit checkpoints. Different
chat templates change token counts (1,973 input tokens in the GGUF against
1,931 in MLX for the same request), so these are not exact-token comparisons.

| Candidate | Revision | Notes |
| --- | --- | --- |
| [ggml-org GGUF](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF) | `0669b98607d47046c7c2b3f801011d54a08cfccf` | Q4_K_M, but a different file from the Unsloth target. Target-only decode: 13.5 to 13.8 tokens/s. |
| [MTPLX Optimized Speed](https://huggingface.co/Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed) | `4dda207f2ffdaa264b2512ebf81249b933abc008` | MLX affine four-bit with eight-bit exceptions. MTP depth 3 decoded 32 to 36 tokens/s and passed the five probes. |
| [MTPLX Bare Speed](https://huggingface.co/Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed) | `b59d7002368575f08a58ba0d26686b53a9c162d6` | Excluded: another model server was running during its screen. |

The MLX runs used [MTPLX](https://github.com/youssofal/MTPLX) 2.11.3, MLX 0.32.2,
and mlx-lm 0.31.3. MTPLX states that its four-bit verify kernels are validated
on distribution and argmax, not bit-identical to stock MLX. Some MLX samples
overlapped the startup of an unrelated server, so they need a repeat before
anyone treats them as isolated results. The catalog uses llama.cpp Metal for
the next validation stage.

## Models that do not fit

Qwen3.5 122B ROCmFP4 (60.7 GiB), gpt-oss-120b MXFP4 GGUF (59.0 GiB), and
Nemotron 3 Super Q4_0 (66.1 GiB) exceed the Metal budget before any KV cache.
CPU or SSD expert streaming would not be a GPU-resident result, and lower-bit
files would change the quantization class.
