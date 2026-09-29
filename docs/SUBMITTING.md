# Submitting results

Submitting is optional. No command uploads anything unless you run `submit` or
tick **Submit this run to Artificial Analysis** in the TUI.

## Commands

Build a bundle from a finished run. This step makes no network request:

```console
uv run agentperf-local prepare-submission results/gemma4-12b \
  --output-dir results/gemma4-12b-submission
```

Send it. The command validates the bundle, prints a privacy notice, and asks
you to confirm. `--yes` confirms without a prompt:

```console
uv run agentperf-local submit results/gemma4-12b-submission
```

It prints a submission ID. Read its status later:

```console
uv run agentperf-local submission-status sub_...
```

Sending the same bundle again returns the same submission, so you can retry an
interrupted upload with the same command. A failed or canceled upload in the
TUI keeps the bundle and shows the `submit` command that retries it.

A token is optional. Set `AGENTPERF_SUBMIT_TOKEN`, or name another variable with
`--token-env`. Without a token the upload is anonymous. The client refuses to
send a token over plain `http` to a non-local host.

## What is uploaded

`prepare-submission` writes four files:

| File | Published? | Contents |
| --- | --- | --- |
| `aggregate.json` | May be published | Aggregate metrics, benchmark labels, client version, and a coarse hardware class for managed runs. |
| `sanitized-turn-evidence.json` | May be published | Per-turn timings and token counts by position only. |
| `private-audit.json` | Never published | The detailed hardware snapshot, the deployment summary, the endpoint probe results, and the power summary. Deleted within 180 days. |
| `bundle-manifest.json` | No | Sizes and SHA-256 digests that bind the other three files. |

The bundle never contains prompts, responses, tool arguments, credentials,
local paths, hostnames, serial numbers, or endpoint URLs. Timings and task shape
can still fingerprint a run. Review the files before you send them.

[FORMATS.md](FORMATS.md#local-public-aggregate) describes each file in detail.

## What can be submitted

- A run must have no failed turns. A turn that produced less than half its
  recorded output length also blocks the bundle.
- Runs with `--tool-mode live` cannot be submitted.
- Attached-server runs publish no hardware, because the client machine is not
  the server.
- Runs below the 65,536-token benchmark context are marked `reduced: true`
  and are kept apart from full-context results.
- Ollama runs use the `recorded` output policy. Their end-to-end latency is a
  normalized estimate and is not directly comparable with `exact` runs. The
  model may also stop early, which can trip the short-output rule above.

## Trust tiers

Every submission starts as `community-self-reported`. The client is open
source, so nothing it reports can prove which GPU, model, or runtime served a
run.

A `verified` tier is planned but not live. When it arrives, the service, not the
client, will assign it. Planned checks include a clean, recent release commit, a
managed deployment from the bundled catalog, passing endpoint probes, and
plausible hardware numbers. Failed checks will not reject a run; it stays
self-reported.
