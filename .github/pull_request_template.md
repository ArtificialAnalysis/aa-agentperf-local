<!--
Not a recipe? Delete this template and describe your change.

Recipe pull requests: read CONTRIBUTING.md first. Fill in every section.
Delete the Accuracy section only if every target weight comes from nvidia,
RadixArk, unsloth, or the model creator, at 4 bits or more. Weights below 4
bits need a check by Artificial Analysis whatever the source: keep the
section, and write "Needs AA verification" in it.
-->

## Recipe

| Recipe file | Model source (`hf_repository`) | Framework | Tested on |
| --- | --- | --- | --- |
| `recipes/<model>/<hardware>/<profile_id>.yaml` | `owner/name` | llama.cpp / SGLang / vLLM | <!-- device, memory --> |

<!-- Anything a reviewer should be suspicious of: untuned settings, memory values you estimated, known failures. -->

## Performance

Full `agentperf-default-v1` replay, `managed-run`, 65,536 context, `exact` output policy.

| Machine | Recipe | Output tok/s | End-to-end tok/s | TTFT p50 / p95 (ms) | E2E latency p50 / p95 (ms) | Replay time (s) | Acceptance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| | | | | | | | |

<!-- Speculative recipes: add a target-only row for the same machine. Acceptance is from deployment.log. -->

<details>
<summary><code>summary.json</code></summary>

```json

```

</details>

<details>
<summary><code>qualification.json</code></summary>

```json

```

</details>

## Reproduce

```console
uv run agentperf-local managed-run --profile-id <profile_id> --framework <framework> --output-dir results/<profile_id>
```

- **agentperf-local commit** (`measurement.json` → `producer.source_revision`):
- **Runtime version and SHA-256** (`deployment.json` → `deployment.runtime`):
- **llama.cpp** repository, commit, and build command — or **vLLM / SGLang** image digest or `pip install` command:
- **Machine** (device, memory, OS and kernel, driver and CUDA / ROCm / Metal version):
- **Converted files** (command and tool commit), if any:

## Accuracy

<!-- Required when target weights come from a source that is not trusted. See CONTRIBUTING.md, section 4.
     Below 4 bits: write "Needs AA verification". A maintainer runs the check before merge. -->

- **Reference model** (`owner/name@revision`):
- **Harness** (repository, commit, command):

| Model | Run 1 | Run 2 | Run 3 | Run 4 | Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| This recipe | | | | | |
| Reference | | | | | |

<!-- Attach the harness output for every run as a zip file. -->

## Checklist

- [ ] The recipe follows the format rules in `CONTRIBUTING.md`, and every artifact downloads on a clean machine.
- [ ] `BUNDLED_RECIPES_DIGEST` is updated, and `uv run pytest tests/test_model_catalog.py tests/test_model_candidates.py` passes.
- [ ] The run has 0 failed turns, 5 of 5 probes passed, and `source_state` is `clean`.
- [ ] Speculative recipe: a target-only row is in the table. (Delete this line if not speculative.)
- [ ] Untrusted weights: the accuracy check passes. (Delete this line if the weights are trusted.)
- [ ] Below 4 bits: the Accuracy section says "Needs AA verification". (Delete this line if 4 bits or more.)
