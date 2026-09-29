<p align="center">
  <a href="https://artificialanalysis.ai">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/ArtificialAnalysis/aa-agentperf-local/main/docs/assets/aa-logo-dark.svg">
      <img alt="Artificial Analysis" src="https://raw.githubusercontent.com/ArtificialAnalysis/aa-agentperf-local/main/docs/assets/aa-logo-light.svg" width="240">
    </picture>
  </a>
</p>

<h1 align="center">agentperf-local</h1>

<p align="center">
  Benchmark how fast your machine serves an AI agent.<br>
  An open-source tool from <a href="https://artificialanalysis.ai">Artificial Analysis</a>.
</p>

<p align="center">
  <a href="https://github.com/ArtificialAnalysis/aa-agentperf-local/blob/main/LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-8842FD"></a>
  <img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-8842FD">
</p>

`agentperf-local` measures how fast your machine serves an AI agent. It replays
recorded agent conversations against an OpenAI-compatible model server and
reports throughput and latency. Each request carries the full conversation so
far, as a real agent's would. It measures speed without measuring output quality.

It can benchmark a server you already run, or it can download a pinned model,
start the server, and benchmark it for you.

## Quick start

You need [`uv`](https://docs.astral.sh/uv/). It fetches Python 3.12 if you do
not have it. Docker and Rust are optional. macOS, Linux, and Windows are
supported. On Windows, managed runs need an NVIDIA GPU and llama.cpp; vLLM and
SGLang run only on Linux.

You also need either a model server that you already run with an
OpenAI-compatible API, such as llama.cpp, LM Studio, vLLM, SGLang, or Ollama,
or a serving framework that the tool can start. The bundled recipes use
llama.cpp (`llama-server` on your `PATH`), vLLM or SGLang on NVIDIA GPUs, or
Splash on Apple Silicon, at the exact version each recipe pins. The tool does
not install frameworks.
`agentperf-local deployment-options --profile-id <id>` shows which frameworks
can serve a recipe on this machine.

```console
uv tool install agentperf-local
agentperf-local
```

To try it without installing, run `uvx agentperf-local`. `pipx install
agentperf-local` also works.

To work from a source checkout instead:

```console
git clone https://github.com/ArtificialAnalysis/aa-agentperf-local.git
cd aa-agentperf-local
uv sync
uv run agentperf-local
```

The examples below use `uv run agentperf-local` from a checkout. With an
installed tool, drop the `uv run`.

The TUI walks you through choosing a model, setting up, and running. Arrow keys
move, Enter continues, Escape goes back, `?` opens help, and `q` quits. To
prefill settings with flags, use `agentperf-local tui`. See
[TEXTUAL_TUI.md](https://github.com/ArtificialAnalysis/aa-agentperf-local/blob/main/docs/TEXTUAL_TUI.md).

To run without the TUI, pick a catalog profile and a framework you have
installed:

```console
uv run agentperf-local managed-run \
  --profile-id gemma4-12b-it-q4-0 \
  --framework llama-cpp \
  --output-dir results/gemma4-12b
```

`managed-run` downloads the pinned model into the Hugging Face cache, checks
every file's SHA-256, starts the server on localhost, runs the replay, and
stops the server. Every recipe it can run is a YAML file in
[`recipes/`](https://github.com/ArtificialAnalysis/aa-agentperf-local/tree/main/recipes), by model then hardware.

The `qwen38-27b-splash-dflash` recipe serves on [Splash](https://github.com/incoai/splash),
Inco AI's Metal engine for Apple Silicon (M3 or newer, macOS 26.4 or later).
Install it with `brew install incoai/tap/splash`. The recipe pins Splash 1.0.2
and the `incoai/Qwen3.8-27B-Splash` package by commit. `managed-run` downloads
and verifies the package itself, then starts Splash's server on that snapshot,
so Splash never resolves a newer revision. Splash drops `ignore_eos`, so this
recipe runs only under the `recorded` output policy, and its KV cache is always
eight-bit. Both make its results not comparable with `exact` runs:

```console
uv run agentperf-local managed-run \
  --profile-id qwen38-27b-splash-dflash \
  --framework splash \
  --output-token-policy recorded \
  --output-dir results/qwen38-splash
```

## The default run

The default replay is `agentperf-default-v1`: eight recorded agent tasks with
168 model turns. It uses the `exact` output policy, which makes every turn
generate its recorded number of tokens. This is the comparable run.

`--replay aa-mini-v1` selects a six-turn synthetic replay. It is a quick
install check. Its results are not comparable.

## Context requirement

A full run needs a 65,536-token context at batch size 1. Every catalog profile
launches at that context. An attached server that serves more, such as
131,072 tokens, also counts as full.

The default replay's largest turn needs about 58,000 tokens, so it only runs
at the full 65,536. If your device cannot fit that, you can still check your
setup with `aa-mini-v1`, which needs 8,192:

```console
uv run agentperf-local managed-run \
  --profile-id gemma4-12b-it-q4-0 \
  --framework llama-cpp \
  --replay aa-mini-v1 \
  --context-tokens 8192 \
  --output-dir results/gemma4-12b-mini
```

In the TUI, start with `uv run agentperf-local tui --replay aa-mini-v1` and
choose a smaller context on the Setup screen. A run below 65,536 tokens is
marked `reduced: true` and is not comparable with full-context results.

For an attached server, the tool reads the context at the start of the run:
`meta.n_ctx` from llama.cpp, `max_model_len` from vLLM and SGLang, or
`maximum_context_tokens` from the Splash status endpoint. A server that
reports none of them is recorded as not comparable.

## Attached servers

Point `run` at a server you already run:

```console
uv run agentperf-local run \
  --base-url http://127.0.0.1:8080/v1 \
  --model served-model \
  --output-dir results/my-server
```

Use your server's base URL and the model name it reports:

| Server | Typical base URL |
| --- | --- |
| llama.cpp | `http://127.0.0.1:8080/v1` |
| LM Studio | `http://127.0.0.1:1234/v1` |
| vLLM | `http://127.0.0.1:8000/v1` |
| SGLang | `http://127.0.0.1:30000/v1` |

The `exact` policy sends `ignore_eos`, which is not part of the OpenAI API.
Before the replay, `run` and `managed-run` check that the server honors it. If
it does not, they stop and name the flag to change. The TUI switches to the
`recorded` policy instead and says so.

**Ollama cannot honor `ignore_eos`.** The tool detects Ollama before the run
and warns you. The run then uses the `recorded` policy, which lets the model
stop on its own and reports end-to-end latency as a normalized estimate. That
result is not directly comparable with `exact` runs.

To send an API key, put it in an environment variable and pass the variable's
name with `--api-key-env`. No flag takes a literal key.

## Real tool calling

By default the replay skips tool time between turns. `run --tool-mode fixed_delay`
sleeps after each turn for the tool time recorded there. `run --tool-mode live`
runs the recorded shell commands in Docker containers, so tool time is real.
This mode is opt-in and needs Docker and prebuilt images. The build scripts
are in the source checkout. They are bash scripts, so on Windows run them from
Git Bash or WSL:

```console
scripts/install-swebench-validation.sh   # arm64 hosts only, once
scripts/build-default-containers.sh

uv run agentperf-local run \
  --base-url http://127.0.0.1:8080/v1 \
  --model served-model \
  --output-dir results/live-tools \
  --tool-mode live
```

Security caveats:

- The recorded commands are untrusted input. Docker reduces the risk but is not
  a security boundary.
- Access to the Docker daemon is equivalent to root on most hosts.
- Containers have no network by default, but a replay manifest can ask for one.
  Pass `--live-network none` to force no network.
- The task workspace is mounted read-write. It lives under the output directory
  unless you pass `--live-workspace-root`.
- The scripts pull or build third-party images and clone the SWE-bench harness
  from GitHub.


## Rust client (experimental)

Python is the default client. An optional Rust client records timings outside
Python. It needs a [Rust toolchain](https://rustup.rs):

```console
uv sync --extra rust
uv run agentperf-local tui --client rust
```

With an installed tool, use `uv tool install 'agentperf-local[rust]'`
instead. A plain `uv sync` removes the extension again. Both clients produce the same
metrics.

## Submitting results

Submitting is optional and nothing is uploaded unless you ask.
`prepare-submission` builds the submission file, `submit` sends it, and
`submission-status` reads it back. A run on a server you started needs
`run --attached-server FILE`. See [SUBMITTING.md](https://github.com/ArtificialAnalysis/aa-agentperf-local/blob/main/docs/SUBMITTING.md) for
details on what is sent to Artificial Analysis.

## Commands

| Command | Purpose |
| --- | --- |
| `tui` | Open the guided full-screen app. |
| `run` | Replay a workload against a server you run. |
| `managed-run` | Download a catalog model, then serve and benchmark it. |
| `deployment-options` | Show which frameworks can serve one catalog profile on this machine (default `gemma4-12b-it-q4-0`, or `--profile-id`). |
| `doctor` | Show local hardware facts without identifiers. |
| `convert` | Convert an agent recording into a replay manifest. |
| `prepare-submission` | Build a submission file without uploading it. |
| `submit` | Send a prepared submission to Artificial Analysis. |
| `submission-status` | Read a submission's status. |

Run `uv run agentperf-local <command> --help` for every option.
`python -m agentperf_local` is the same program.

## Results

A run writes a new directory:

```text
results/my-server/
├── turns.jsonl
├── tasks.json
├── tools.json
├── failures.json
├── summary.json
└── measurement.json
```

`summary.json` holds the headline numbers: output tokens per second and median
and p95 first-token and turn times. It is written last, so a crashed run has
no summary. A managed run also writes the server log and deployment record.
[FORMATS.md](https://github.com/ArtificialAnalysis/aa-agentperf-local/blob/main/docs/FORMATS.md) describes each file.

The results are private. They can contain paths, model labels, and errors.
Starting a run sends the recorded prompts to the model server, so a remote URL
sends them off your machine.

## Development

```console
make lint        # ruff check, ruff format --check, ty check
make test        # pytest
make test-rust   # Rust core tests
```

CI runs these checks and the Python and Rust client equivalence tests. See
[AGENTS.md](https://github.com/ArtificialAnalysis/aa-agentperf-local/blob/main/AGENTS.md) for the full list and the code conventions.

## Documentation

- [Recipes](https://github.com/ArtificialAnalysis/aa-agentperf-local/tree/main/recipes): managed-run recipes and instructions for adding more.
- [Textual TUI](https://github.com/ArtificialAnalysis/aa-agentperf-local/blob/main/docs/TEXTUAL_TUI.md): options, keys, and screens.
- [Architecture](https://github.com/ArtificialAnalysis/aa-agentperf-local/blob/main/docs/ARCHITECTURE.md): measurement rules, evidence boundary,
  and package layout.
- [Data formats](https://github.com/ArtificialAnalysis/aa-agentperf-local/blob/main/docs/FORMATS.md): inputs, outputs, and schemas.
- [Submitting](https://github.com/ArtificialAnalysis/aa-agentperf-local/blob/main/docs/SUBMITTING.md): what is uploaded and how it is used.

agentperf-local is built and maintained by [Artificial Analysis](https://artificialanalysis.ai).
Code is licensed under [Apache-2.0](https://github.com/ArtificialAnalysis/aa-agentperf-local/blob/main/LICENSE). The Artificial Analysis name and logo
are not covered by the code license.
