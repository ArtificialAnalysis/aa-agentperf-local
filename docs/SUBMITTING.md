# Submitting results

Submitting is optional. No command uploads anything unless you run `submit` or
tick **Submit this run to Artificial Analysis** in the TUI.

## Commands

Build the submission file from a finished run. The only network request asks
GitHub for the full commit of the serving framework's build:

```console
uv run agentperf-local prepare-submission results/qwen38-27b \
  --output results/qwen38-27b-submission.json
```

Send it. The command checks the file, prints a privacy notice, and asks you to
confirm. `--yes` confirms without a prompt:

```console
uv run agentperf-local submit results/qwen38-27b-submission.json
```

It prints a submission ID. Read its status later:

```console
uv run agentperf-local submission-status sub_...
```

The service keys a submission on its `run_id`. Sending the same file again
returns the same submission, so you can retry an interrupted upload with the
same command. A different body for the same run is refused with
`idempotency_conflict`. A failed or canceled upload in the TUI keeps the file
and shows the `submit` command that retries it.

A token is optional. Set `AGENTPERF_SUBMIT_TOKEN`, or name another variable with
`--token-env`. Without a token the upload is anonymous. The client refuses to
send a token over plain `http` to a non-local host.

## Runs on your own server

A run against a server you started needs a description of that server. Write it
as YAML and pass it to `run` with `--attached-server`:

```yaml
model_release_slug: qwen3-8-27b          # the Artificial Analysis model release
hf_repository: unsloth/Qwen3.8-27B-GGUF
hf_revision: f1bfb127c64f7072bdd2cad55f258b9c8b2910fe
framework: llama-cpp                     # llama-cpp, vllm, sglang, or splash
framework_version: 'version: 0.3.0 (build 10621, commit c1d0e7a00)'
accelerator_backend: metal               # cuda, metal, rocm, vulkan, sycl, or xpu
server_launch_command: >-
  llama-server -m /models/Qwen3.8-27B-Q4_K_M.gguf --ctx-size 65536 --api-key sk-...
```

```console
uv run agentperf-local run --replay agentperf-default-v1 \
  --base-url http://127.0.0.1:8080/v1 --model qwen3.8-27b \
  --attached-server server.yaml --output-dir results/my-server
```

- The server must run on this computer, at a loopback address.
- Paste the launch command as you ran it. The client replaces local paths with
  placeholders such as `$LOCAL_DIR_1` and secret values with placeholders such
  as `$API_KEY`, and it drops secret variables such as `HF_TOKEN`.
- Give `framework_commit` (short or full) or `framework_container_reference`
  (`image@sha256:...`) when the version text names neither a commit nor a
  release.
- On a machine with more than one accelerator, pass `--device` too.

The TUI submits managed runs only. Submit a run on your own server from the
command line.

## What is sent

`prepare-submission` writes one JSON file, the exact body `submit` sends. It
matches the service's pinned spec,
[`submission-openapi.json`](../agentperf_local/data/submission-openapi.json):

| Part | Contents |
| --- | --- |
| `client` | Client version and source commit. A release build or a clean git checkout only. |
| `benchmark` | Workload digest, and the context the run asked for and the server served. |
| `hardware` | Operating system, CPU, memory, and the one accelerator as its driver reports it. |
| `deployment` | Model release, weights, framework build, backend, and the redacted launch command. A managed run adds its recipe. |
| `policy` | Client backend, sampling, output-token policy, cache isolation, and tool replay mode. |
| `run` | Wall time and the time the progress display took. |
| `turns` | Per-turn timings and token counts, by position only. |
| `qualification` | The pass or fail of each synthetic protocol probe. |
| `power` | NVIDIA power telemetry for the measured phase. Required on NVIDIA. |

The service derives every total, distribution, speed, and cache rate from the
turns. It keeps the whole file in private storage indefinitely, and it may
publish aggregate results and per-turn timings.

The file never contains prompts, responses, tool arguments, credentials, local
paths, hostnames, serial numbers, or endpoint URLs. Timings and task shape can
still fingerprint a run. Review the file before you send it.

## What can be submitted

`prepare-submission` refuses a run the service would refuse, and names the
reason:

- Every turn must succeed, with at least two output tokens.
- The server must report a context of at least the context the run asked for.
- The run must use the `exact` or `recorded` output policy, the default
  16,384-token fallback, no output-token margin, and cache isolation.
- A managed run must report cached and uncached prompt tokens on every turn.
- A run on NVIDIA must record power, so do not pass `--no-power`.
- The client must be a release or a clean git checkout.

Ollama runs use the `recorded` output policy. Their end-to-end latency is a
normalized estimate and is not directly comparable with `exact` runs. A failed
qualification probe does not block a submission; the service records it.
