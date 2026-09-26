# Data formats

`agentperf-local` uses versioned JSON and JSONL at its boundaries. Python
dataclasses implement the runtime contracts and are the code that runs. The
closed JSON Schemas under [`docs/schemas`](schemas) are the normative reference
for shareable and supporting artifacts; they are enforced in tests, not loaded
at run time, and the Python validators enforce some rules the schemas do not.

## Replay input

`convert` accepts an agent recording or workload manifest and creates a replay
directory:

```text
workload/
├── manifest.json
└── traces/
    └── TASK_ID.jsonl
```

The manifest contains `version`, `mode`, `name`, `source`, and `tasks`. Every task
binds a unique `task_id` to a relative trace path, names its `source_recording`,
and records expected model-call, tool-call, and tool-delay totals. Optional fields
describe the workload family, adapter, live tool environment, and each task's
`required_context_tokens`.

Trace paths cannot escape the manifest directory. Before inference, the runner
checks every trace row, task identifier, and declared call count.

Each non-empty trace line is one model turn in replay order. The required fields
are:

- `version`;
- `turn_id`, `task_id`, `conversation_id`, and `conversation_idx`;
- OpenAI-compatible `messages`; and
- `simulated_tool_delay_ms_after`.

Tool definitions and recorded tool calls default to empty lists. Optional fields
carry output-token targets, recorded token counts and duration, and source
provenance. Counts, indexes, and durations must be non-negative.

`target_output_tokens` is a reference output length used to normalize timing.
`max_output_tokens` is an optional positive request cap, further limited by the
run's configured maximum. An explicit trace cap applies under all output-token
policies; the output-token margin applies only when deriving a cap from a recorded
target. Without an explicit cap, existing target-based and fixed policies apply.
The default `exact` policy requires a target or recorded completion count on every
turn. Synthetic traces can supply authored targets, as the bundled mini replay does.
Under `recorded` or `fixed`, traces without a reference output length can omit the
target and set only the cap. Their normalized timing uses the observed output length
and equals the raw timing; unmeasurable decode windows still raise a warning.

A trace may include a Docker tool environment, but it is ignored unless the user
explicitly selects live tool mode. Recorded shell actions are untrusted input; see
[the architecture guide](ARCHITECTURE.md#replay-controls) before enabling it.

A recorded SWE-bench image names one architecture, and an emulated container makes
tool timings meaningless. Live tool mode therefore reads the Docker daemon's
architecture at the start of a run and selects the matching published image. A
digest-pinned image and an image named with `--live-tool-image` are both left
exactly as given. Every live image must already be in the local Docker store, and
that is checked before the first request, so a missing image stops the run instead
of discarding replayed inference. Pull or build the images first: a pull started by
the first container would otherwise land inside the measured window.

## Private run output

`run` writes five artifacts to a new directory:

| File | Contents |
| --- | --- |
| `turns.jsonl` | Ordered per-turn timing, tokens, provider-reported input-cache accounting, normalization, response shape, and tool results. |
| `tasks.json` | Per-task totals and failed turn IDs. |
| `tools.json` | Tool totals grouped by delay source, tool name, and command. |
| `failures.json` | Request and tool errors keyed by task and turn. |
| `summary.json` | Run identity, effective policy, totals, `output_tokens_per_second`, latency distributions, warnings, and artifact paths. |

Every envelope has `version: 1` and a `kind` discriminator. Durations use
milliseconds unless named otherwise. Missing measurements are JSON `null`.

The writer encodes the complete set before writing. It creates and syncs the four
data artifacts, then creates and syncs `summary.json` last. The summary is the local
completion marker, not a signature. Writers refuse existing output files and
symlinked output directories.

`turns.jsonl` remains in manifest and trace order. A successful turn completed the
request, passed the post-close F0 transport check, and had no configured tool-replay
error. This does not mean that the model solved the source task.

Observed output length prefers server usage and falls back to local tokenization.
Normalization scales the first-to-last visible-token interval to the recorded
target while preserving non-generation latency. Raw values are always retained.
A short-output warning is recorded when observed output is less than half of a
positive target.

The API key and its environment-variable name are never serialized. Results can
still reveal endpoint and model labels, paths, cache namespaces, tool names,
exceptions, and run settings. Treat the complete directory as private.

`managed-run` and the managed TUI path write more private companions around the
replay. Every one of them carries the run identifier that `measurement.json`
minted, so a companion from another attempt can never be attached to this one:

| File | Written | Contents |
| --- | --- | --- |
| `deployment.log` | before inference | Mode-0600 child-server log used for GPU startup evidence and diagnostics. |
| `deployment.json` | before inference | A deployment identifier and creation time, the exact Hub revision, artifact path, size and SHA-256, launch command and its digest, runtime fingerprint, GPU startup policy, and detected hardware. |
| `measurement.json` | before inference | The run identifier, pre-run suite, managed artifact or attached alias digest, runtime, endpoint alias, client source, local hardware binding, the served-context observation, and the SHA-256 of the `deployment.json` bytes. |
| `qualification.json` | before inference | The public synthetic protocol probes, run against the ready server and bound to the run identifier. |
| `telemetry.jsonl` | around inference | Normalized NVIDIA samples from the child collector. Absent on hosts without `nvidia-smi`. |
| `power.json` | after inference | The measured phase reduced to energy, coverage, and validity, plus how the collection ended. |

The managed controller owns the serving child, verifies the exact model alias and
requested GPU backend, runs the probes, starts the collector, runs the same
five-artifact replay writer, stops the collector, and stops the child. These
companions remain private and are outside the five-file durable commit set.
`summary.json` also carries the run identifier, and the packaging gates require
it to match the binding.

## Benchmark binding

`run`, `managed-run`, and the TUI collect a hardware snapshot and create
`measurement.json` before inference, binding the expected identity — and the
served-context observation from the pre-replay probe — to that attempt.

These hashes establish local byte consistency. They do not prove that an endpoint
loaded the declared bytes or ran on the detected accelerator.

## Local public aggregate

`prepare-submission` converts a valid bound result into the `public-minimal-v1`
aggregate. The typed producer
allowlists aggregate metrics, execution-bound device-class fields for managed runs,
benchmark labels, producer provenance, and digests that bind the aggregate to local
evidence. Attached runs publish `hardware: null` because the client snapshot does
not describe the remote server. The aggregate does not copy prompts, responses, tool
content, endpoints, filesystem paths, stable device identifiers, or raw diagnostics.

The envelope contains two different hashes:

- `payload_digest` is SHA-256 over the compact, sorted-key bytes of the `payload`
  object, prefixed with `sha256:`.
- The bundle manifest's exact-file SHA-256 covers the emitted envelope bytes,
  including whitespace and the trailing newline.

The packager reopens and cross-checks the complete private artifact set. It rejects
request or tool failures, incomplete timing and token evidence, short outputs,
capped generations, action-count mismatches, a summary whose run identifier
differs from the binding, and inconsistent aggregates. Observers run between
turns and their time is excluded from the measured duration; a run stays
packageable while that excluded time is at most one percent of the measured
duration, and the aggregate states it as `observer_duration_ms` so a receiver
can apply a stricter rule. A valid aggregate is still self-reported evidence, not
proof of hardware or model identity.

The sanitized turn evidence in the bundle is ordinal-only. It excludes source
identifiers and generated content, and a receiver can derive its aggregate timing,
token, pacing, and action metrics. Timing and task shape can still fingerprint the
underlying trajectory.

`prepare-submission` writes a new directory containing:

```text
submission/
├── aggregate.json                  # public-minimal-v1, may be published
├── sanitized-turn-evidence.json    # may be published
├── private-audit.json              # AA-private, never published
└── bundle-manifest.json            # written last, digests the other three
```

`private-audit.json` carries what the service needs to classify a submission
but must not publish: the full hardware snapshot (including GPU power limit,
clock caps, and CPU base frequency), the path-free deployment summary, the
runtime qualification report, and the reduced power summary. It contains no
prompts, responses, paths, hostnames, endpoint URLs, or credentials, and it
states its own bounded retention. Only a managed run writes a qualification
report; an attached run's audit carries none.

The bundle manifest binds the exact length, SHA-256, media type, schema, role,
privacy profile, and run identifier of the three payload files. It is written
last. `submit` reopens the directory, verifies exact files and
cross-file bindings, checks every payload object against the closed field
contracts, and recomputes row-derived aggregates. It requires a managed run's
private audit to reproduce the public hardware profile. An attached run must omit
that public profile. Every bundle must name the same run and aggregate throughout.

`prepare-submission` performs no network request. `submit` is the only command
that sends bytes: it validates the bundle as above, prints the private-audit notice, asks for a yes,
and sends the four files in one request. Re-sending the same bundle returns the
same submission, so an interrupted upload is retried by running `submit` again.
`submission-status` reads back the status, tier, and reason codes.

## Supporting schemas

| Schema | Produced or consumed by |
| --- | --- |
| [`model-candidates-v2`](schemas/model-candidates-v2.schema.json) | The bundled unsigned candidate and managed-deployment catalog |
| [`private-audit-v1`](schemas/private-audit-v1.schema.json) | The AA-private audit file inside a bundle |
| [`private-nvidia-telemetry-v2`](schemas/private-nvidia-telemetry-v2.schema.json) | The NVIDIA collector; carries the run identifier and records glitched lines as all-missing samples |
| [`public-submission-v2`](schemas/public-submission-v2.schema.json) | `prepare-submission` and aggregate bundle validation; carries the run identifier and the observer time the run excluded |
| [`runtime-qualification-v1`](schemas/runtime-qualification-v1.schema.json) | Superseded by v2; kept only because the checked-in MLX evidence uses it |
| [`runtime-qualification-v2`](schemas/runtime-qualification-v2.schema.json) | Managed runs; adds the run identifier |
| [`sanitized-turn-evidence-v1`](schemas/sanitized-turn-evidence-v1.schema.json) | `prepare-submission` |
| [`submission-bundle-v2`](schemas/submission-bundle-v2.schema.json) | `prepare-submission` and `submit`; binds the private audit and the run identifier |

Schema validation checks structure, types, bounds, enums, and closed objects.
Semantic validators still enforce relationships such as exact turn order,
recomputed totals, file hashes, and cross-file bindings.

Changing a metric definition, privacy meaning, canonicalization rule, or required
field requires a new schema and profile. Superseded schemas are removed until a
released version has produced artifacts that need them.
