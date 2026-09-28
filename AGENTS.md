### Languages
- python
- rust

### Code Quality

- Type casting is prohibited and banned. You are a disciplined engineer not a wizard. The only documented exception is `__getattr__`-based lazy proxies, where the type system cannot see through attribute forwarding; in those cases `cast()` is preferred over `# type: ignore` and must carry an inline comment referencing this carve-out.
- Same constraints apply to the use of "as any" and "any" types. This is not allowed.
- Avoid magic numbers, everything in the code should be self-explanatory and avoid unnecessary complexity that requires consulting the outside world.
- Write docstrings in simple English: short sentences, active voice, one idea per sentence.
- When working in Python ensure you treat type hints as first-class citizens and use Pydantic models (`BaseModel`; declare `class X(BaseModel, frozen=True)` for immutable values, and put other config in the class line too) instead of dictionaries and key-value pairs data structures. Put invariants in `@model_validator(mode="after")`. Copy a model with `replace_fields`, never `model_copy(update=...)`, which skips validation. Show errors to people through `error_text`. Read JSON into a model with `read_record` (raw bytes) or `read_object` (decoded dict): strict JSON mode, unknown keys rejected unless the format says readers skip them. Keep a hand-written reader only when the JSON shape is not the model's field list. Hold secrets in `SecretStr`. Models are for data: values that are validated, stored, or sent. Objects that do work (observers, controllers, collectors, anything holding a process, stream, widget, event, or callable) and test fakes stay plain dataclasses, so they need no `arbitrary_types_allowed`. Stdlib dataclasses also remain for `RawRead` (built inside the stream loop) and Textual `Message` subclasses.

### Testing

Tests are liabilities that must earn their place:

- Prefer one end-to-end test through the module's public surface over many per-function tests that monkeypatch internals.
- Prefer parameterized pytests over individual test functions when testing multiple cases over public surfaces.
- Keep test code boring.

### Local CI Checks

Before pushing a PR branch, run the same checks as `.github/workflows/ci.yml`:

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run ty check --python-platform win32
uv run pytest
cargo test --locked -p agentperf-local-rustcore --no-default-features
uv sync --locked --extra rust
uv run pytest
uv run pytest tests/test_streaming.py -q
```

CI also runs `actionlint` on the workflow files. The `cargo` and `--extra rust` steps need a
Rust toolchain. Run `uv sync --locked` afterwards to return to the default
Python-only environment.

### Docstrings

Module docstrings are a short index of the public surface. Decision rationale lives on the function, class, or call site it explains — never accumulated as narrative in the module header.

## Conventions

- **Commits**: imperative, `feat:`/`fix:`/`chore:`-style prefixes. Do NOT add
  a Claude/AI co-author trailer.
- **Entry points**: argparse `main() -> int` per executable module. One
  console script, `agentperf-local`, is equivalent to `python -m agentperf_local`;
  run everything else via `uv run <path>` or `python -m <module>`.
- **Style**: ruff, line length 120, lint set `E,F,I,UP`; py3.12 syntax
  (`X | None`, builtin generics); `orjson` for JSONL I/O; `pathlib`.
- **Imports**: a package `__init__.py` holds only its docstring. Import every
  name from the module that defines it, so one name has one import path. A name
  used outside its own module is public; a leading underscore means it is not.
- **Tests**: flat `tests/test_<module>.py`; `asyncio_mode = "auto"` (no
  asyncio markers); shared helpers imported as `from tests.<helper> import …`;
  fixtures under `tests/fixtures/`. Tests must not hit the network — SSE
  behavior is covered by the localhost server in `tests/localhost_sse.py`.
- **The streaming path is sacred**: never add per-chunk Python work (parsing,
  tokenizing, logging) inside the stream loop. Defer to post-close decode or
  phase-end aggregation. Equivalence tests pin all client impls to identical
  metrics — run them after touching any client.
