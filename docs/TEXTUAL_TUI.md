# Textual TUI

`agentperf-local` includes a keyboard-first terminal interface for replaying an
AgentPerf dataset against an OpenAI-compatible model server. Results are written
locally. Nothing is uploaded unless the user ticks Submit on the confirm screen;
see [Submitting a run](#submitting-a-run).

## Launch

From a source checkout:

```console
uv run agentperf-local
```

After installing the package, run `agentperf-local` directly.
`python -m agentperf_local` is the same command. With no command, it opens
the TUI with default settings. To prefill settings with flags, use the `tui`
command, as shown below.

The TUI uses the Python client unless `--client rust` or Advanced options →
Client selects the experimental Rust client. Rust needs the `rust` extra
(`uv sync --locked --extra rust`); without it, preflight stops and says so.

Configuration can be prefilled without starting a run:

```console
uv run agentperf-local tui \
  --output-dir results/demo \
  --base-url http://127.0.0.1:30000/v1 \
  --model served-model \
  --api-key-env PRIVATE_TOKEN
```

| Option | Meaning |
| --- | --- |
| `--replay agentperf-default-v1` | Select the default bundled replay: the full AgentPerf replay with 8 tasks and 168 turns. |
| `--replay aa-mini-v1` | Select the quick install check: six synthetic tool-use turns designed for an 8,192-token context. Its results are not comparable. |
| `--manifest PATH` | Use a custom converted replay manifest. |
| `--output-dir PATH` | Set the results folder. Each run writes one fresh `run-<timestamp>` subdirectory inside it, so the same folder serves every run. |
| `--base-url URL` | Set the OpenAI-compatible API base URL. Plain http is fine for local and LAN servers; a non-local URL combined with an API key requires HTTPS so the key never travels in cleartext. |
| `--model NAME` | Set the model name sent to the server. |
| `--api-key-env NAME` | Name an environment variable containing the API key. |
| `--client python\|rust` | Select the recorded streaming client. Default `python`; `rust` is an experimental client for high-concurrency benchmarking and needs the `rust` extra. |
| `--recipes PATH` | Load recipes from another folder with the same layout. |
| `--cache-root PATH` | Set the model cache used by managed models. Defaults to the standard Hugging Face hub cache, honoring `HF_HUB_CACHE` and `HF_HOME`. |
| `--port PORT` | Set the owned localhost port used by a managed server. |
| `--startup-timeout-seconds N` | Bound managed server startup. |
| `--device N` | Preselect the detected accelerator a managed model uses. |
| `--submit-token-env NAME` | Name the optional environment variable holding an Artificial Analysis submit token. Without one, the upload is anonymous. Default `AGENTPERF_SUBMIT_TOKEN`. |
| `--submit-base-url URL` | Point the upload at another submission service, for testing. |

## Keyboard flow

The primary path is:

```text
Choose a model & config → Setup → Confirm → Running → Result
```

**Use existing server** on the welcome screen goes directly to Setup. Back returns
to the route you came from, and keeps your field values. Ticking the run
confirmation for a managed run keeps focus on it. For your own server, focus moves
to the optional submission checkbox once the server answers its check. Down from
the submission checkbox reaches **Run benchmark**.

- Up and Down choose an option.
- Left and Right move between controls, including the welcome choices and model details.
- Up and Down also move between closed dropdowns, fields, and buttons.
- Enter opens a dropdown, selects a model, or reviews setup from a text field.
- Space checks the run confirmation.
- Escape goes back. During a run, a first Escape arms cancellation and a second
  Escape within three seconds cancels; on the Start screen it exits.
- `l` swaps the activity log for the owned server's own log during a run this
  app started, and back. It does nothing for a server you run yourself.
- `d` toggles extra run charts. In a narrow terminal it switches between the
  activity log and metrics; it never starts or cancels a run.
- `?` opens the help page: keys, the commands that start llama.cpp, Ollama,
  LM Studio, and vLLM, and what the evidence means.
- Press `?` again to close help. `F1` remains available while typing. Returning from
  help or privacy restores the field, cursor, and scroll position you left.
- `Ctrl+P` opens the privacy page.
- `q` quits from an idle screen; during a run it arms cancellation like Escape,
  and only `Ctrl+C` cancels immediately. While a cancellation is still in
  flight, a second `q` force-quits; a managed server may then be left running.

Plain `q`, `l`, `d`, and `?` remain editable characters while an input field has focus.
`Ctrl+C` cancels an active run or exits an idle TUI.

The interface supports terminals down to 48 columns by 16 rows. Compact layouts
stack the model list and details and keep all primary actions reachable by keyboard.
Primary actions stay at the bottom while page content scrolls. Setup shows the
selected model and editable server settings; managed server addresses are set
automatically. Expand **Advanced options** to change the measured client.
Fields keep their labels and values on one line, even in compact terminals.
Right from the model list reaches a separate scrollable detail pane, so long
compatibility notes and download details remain readable.
Left and Right keep their normal cursor behavior inside text fields. Tab remains
an optional shortcut; every control is reachable with arrows. Welcome choices
include their descriptions in the clickable area, and controls highlight on hover.
During model preparation, download bytes and a progress bar replace the live
activity spinner. The top progress bar appears when the replay starts.

Routine headings and summaries use neutral text. Purple marks the main action
and keyboard focus; green marks a successful result, and warning colors mark
problems that need attention. The footer shows relevant shortcuts without
repeating the focused button. Help contains the less frequent privacy shortcut.

## Included choices

The model screen lists every recipe in [`recipes/`](../recipes), plus an
`Other model or server` entry. The list has two parts. **This computer** holds
the recipes for this computer's GPU family, then `Other model or server`.
After a gap, **Other hardware** holds the recipes for other GPU families,
greyed out. Each part is a table with the column headings **Model/Quant** (the
model, then each recipe's weight format), **Spec decode** (the speculative
decoding method, or a dash for none), and **Built for** (the hardware the recipe
was built for). Recipes sit under their model's name. A colored mark gives each
recipe's standing:

| Mark | Standing |
| --- | --- |
| Green `●` | Runs on this computer. |
| Amber `▲` | Needs setup first (install a framework, or memory could not be checked), or fits only at a reduced context the replay allows. |
| Red `✗` | Too large for this computer's memory at any context the replay allows. |
| Grey `·` | Made for other hardware. |

Within a part, models follow in name order, and a model's recipes go best
standing first. Each recipe is checked at its full context, whatever context
setup last held, against the selected replay's minimum. With several
accelerators and no device chosen yet, a recipe shows its best standing across
them. The detail pane leads with the standing and, when a recipe cannot start,
the reason; for llama.cpp it also names the backend a recipe pins, such as
Vulkan. Recipes loaded with `--recipes`
also carry a note that they are not from Artificial Analysis.

Every catalog candidate is evaluated through the same selection surface. A
candidate with a complete deployment recipe is intersected with the detected
CUDA, ROCm, or Apple Metal platform, the accelerator families declared by that
recipe, its framework list, installed executables, and its minimum memory. Only
installed launch candidates appear in the selector. A blocked candidate remains
visible and shows whether the problem is platform support, memory, or a missing
runtime. Backend capability remains pending until startup verification. Every
catalog candidate is a managed choice, so a server the user already runs is
reached through the `Other model or server` entry.

On a machine with more than one detected accelerator, the managed setup screen
adds a Device selector; the run cannot start until one is chosen, and the owned
server process is pinned to that device so the private records describe only it.

The managed setup screen also has a Context selector. It offers the full
65,536-token benchmark context plus a fixed ladder of reduced choices (32,768 /
16,384 / 8,192 tokens), each labelled with the accelerator memory it needs. A
reduced choice appears only when the selected replay's floor allows it, so the
default replay offers the full context alone. The default is the full context
when it fits the selected device, otherwise the largest reduced option that
fits, so a small device gets a working default instead of a dead end. A reduced choice keeps a persistent orange
notice through the confirm, run, and result screens. A reduced run is not
comparable with full-context results. It can still be submitted; its requested
and observed context stay explicit so the service keeps it separate. Attached servers show no picker;
their served context is observed at run time instead.

The default replay is the full `agentperf-default-v1`. The quick-check
`aa-mini-v1` replay is one synthetic library-assistant task with six model
turns and five tool calls, sized for an 8,192-token context. Its results are
not comparable. To run it on a managed model at that size, choose 8,192 tokens
in Setup.

## Submitting a run

The confirm screen groups a second checkbox and a compact upload summary in one
panel: "Submit this run to Artificial Analysis." It is offered for every ready
managed or attached-endpoint run. A token is optional; an upload without one is
anonymous and self-reported. The summary separates what may be published, what
stays private for 180 days, what is never sent, and how failed verification
checks are handled. When a token is set, the app also asks the service whether
this client commit is inside the allowlist window and adds a short eligibility
note. That note is advisory; the service decides.

A managed run with Submit ticked probes the server's agent protocol before the
replay and samples GPU power around it on NVIDIA hosts. Failed probes are still
uploaded with `passed: false`, so the service can classify the run as
self-reported. An attached run records
a self-reported benchmark binding before inference. When either run finishes,
the result page adds an upload stage: the bundle is prepared beside the run
folder, sent in one request with a progress bar, and the submission identifier
and status are shown with the command that reads them back later. The bar tracks
bytes sent; 100% does not mean the service has confirmed the submission. A spinner
stays visible beside New run and says “Waiting for confirmation…” until the
service responds. While submission is in flight, New run is disabled and `q`
and Escape wait; `Ctrl+C` cancels the upload
and quits. A failed or cancelled upload keeps the bundle on disk and shows the
exact `submit` command that retries it.

## Network and evidence boundary

Setup and local preflight do not contact the model server or Hugging Face.
Preflight validates local files and output paths, checks client availability and
that a named API-key environment variable is set, collects a minimal hardware
summary, and classifies the server URL as local or remote. When a check fails,
the readiness page reports the failure without showing private values. Some
checks name the specific field; others report a generic message that lists the
fields to review. With a submit token configured, an advisory revision check can
contact Artificial Analysis without blocking the run.

For a server you run yourself, the app sends one `GET /v1/models` request after
you tick the consent box, and the same request again immediately before the
replay starts. This request carries the API key when one is set and no prompt
content. It confirms the server answers, whether it lists your model name, and
the context length it reports for that model. A server that does not answer or
returns an HTTP error stops the run before any prompt is sent. The context the
server reports is bound into the run's evidence the same way the `run` command
binds it.

The same check sends `GET /api/version` and `GET /api/tags` to detect Ollama.
Ollama cannot honour `ignore_eos`, so the app warns before you press Run and
uses the `recorded` output policy. End-to-end latency is then a normalized estimate and is not
directly comparable with `exact` runs. For other servers, the app probes
`ignore_eos` before the replay and falls back to `recorded` the same way if the
server drops it.

After explicit confirmation, a managed run serves the artifact from the shared
Hugging Face cache or downloads it there from the pinned revision. The cache
lookup is scoped to the pinned commit hash, and any file the run serves —
cached or freshly downloaded — must match the catalog's pinned size and
SHA-256. The run then starts an isolated
localhost process with a fresh alias, waits for that alias, checks GPU startup
evidence, invokes the normal replay runner, and stops the complete owned process
group on success, failure, or cancellation.

Starting a run sends replay prompts and tool definitions to the configured model
server. A remote URL sends that content off the local machine. The API-key value
is used only when execution starts.

For an attached endpoint, detected hardware describes only the computer running
the client; model bytes, runtime bytes, process locality, and GPU binding remain
unverified. Managed runs bind an exact artifact and runtime and save backend
startup evidence. This local evidence still does not independently attest the
physical accelerator. Remote runs report service-path timing, including network
latency.

The run screen has a phase heading, a progress bar with task and turn counts
and the elapsed time, a context line naming the size of the request now going
out and the share of the served context window it fills, and a status line
with the last turn's first-token time, decode speed, and total time. The size
is the one the recording counted, because nothing has tokenized the prompt
yet, so the line words it as approximate. The line is words only, so it never
reads as a second progress bar; a server that never reported a context length
gets no share, because there is no honest denominator for one.
Below that, an activity log on the left records one line per step — setup
checked, server answered or model file verified, server ready, GPU startup
verified, replay loaded — and one line per turn with its decode speed and
duration. The last-turn status above the log carries first-token latency.
A single spinning live line names the step in progress. A metrics column on
the right shows the run's tokens per second so far and a decode-speed range chart.
Press `d` to reveal the first-token range chart and per-turn trend. A managed run also shows the
model download progress in GiB and can swap the activity log for the server's
own log with `l`. Below 100 columns the run screen shows the activity log by
default, because its metrics column is wider than other side panels; `d`
switches to the metrics, and the context line shortens. Terminals shorter than
28 rows omit the trend, and shorter than 21 rows omit the first-token chart.

The result screen leads with the outcome, tokens per second, median response
times, and the folder containing the reports. **Result details** reveals p90
timings, per-turn decode percentiles, and three final range charts.

A range chart is a one-line box plot of one per-turn metric, labelled under
the line with its lowest and highest values (min and max) and the median. The
whiskers reach the min and max, the box spans the middle half of turns (p25 to
p75), and the lit cell is the median. The plot is a sketch rather than to scale:
it centres on the median on a log scale, so its width shows how far turns stray
from typical, and a whisker past the edge ends in an arrow. Decode speed reaches
2× either side of the median and the two times reach 10×, because latency grows
with each task's context; one metric always uses the same scale, so runs compare
by eye. The result screen stacks its three charts.

No run or result screen renders
prompts, responses, tool arguments, server URLs, or secrets. The server log
pane shows the owned server's own output, which names only its loopback
address.

## Local outputs

The Setup screen's results folder is a stable parent. Each run writes into its
own fresh `run-<timestamp>` subdirectory of that folder, so an earlier run's
files are never replaced or mixed with a later run's. A completed run writes
the existing private reports there:

- `turns.jsonl`;
- `tasks.json`;
- `tools.json`;
- `failures.json`; and
- `summary.json`.

A managed TUI run additionally writes `deployment.log`, `deployment.json`, and
`measurement.json`. These preserve the private server log, exact launch and
runtime fingerprint, artifact digest, detected hardware, and pre-inference
measurement binding.

Cancellation before result finalization discards the active replay, removes the
measurement binding, and removes the run folder when it is empty. A cancelled or
failed managed run keeps `deployment.log`, `deployment.json`, and
`measurement.json` in its own run folder as diagnostics; the next attempt names
a fresh run folder, so those leftovers never block it. Cancellation is disabled
during the short durable-save phase so a committed report set is not
interrupted.

## Maintainer notes

[`tui/app.py`](../agentperf_local/tui/app.py) owns the screens and keyboard
navigation. [`tui/controller.py`](../agentperf_local/tui/controller.py) adapts
the shared runner and managed pipeline. Benchmark rules stay in the runner, not
in widgets.

The runner reports progress only at turn boundaries, and observer time is
excluded from the run's duration. The stream loop does no UI work.
[`tui/widgets.py`](../agentperf_local/tui/widgets.py) holds the activity log,
charts, and the small ASCII kitty in the run header. Widget repaints keep their
size, so no tick forces a layout pass.

## Verification

Run the TUI tests:

```console
uv run pytest tests/test_textual_app.py -q
```

Render deterministic screens as SVG at the minimum supported size. The default
size is 118 by 36. Check both after a layout change:

```console
uv run scripts/render_textual_screens.py \
  --output-dir results/tui-screens \
  --width 48 \
  --height 16
```

## Current limits

The TUI does not install serving frameworks or decide leaderboard eligibility.
It also does not run repeated scored suites or generate a public leaderboard
result; a submitted run is classified by the service, not by the app. Power is
sampled on NVIDIA hosts only. A managed model can start only when the detected
host matches its recipe. Live tool mode is available only through `run`.
