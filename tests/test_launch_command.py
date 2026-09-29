"""Check that a shareable launch command holds no local path and no secret."""

import pytest

from agentperf_local.deployment.launch_command import LocalPath, redact_launch_command, render_launch_command


@pytest.mark.parametrize(
    ("typed", "shared"),
    [
        (
            "HF_TOKEN=hf_x CUDA_VISIBLE_DEVICES=0 /opt/bin/llama-server -m /home/me/models/q.gguf --api-key sk-1",
            'CUDA_VISIBLE_DEVICES=0 llama-server -m "$LOCAL_DIR_1"/q.gguf --api-key "$API_KEY"',
        ),
        (
            "vllm serve ~/models/qwen --api-key=sk-1 --download-dir=/data/hf --max-num-batched-tokens 8192",
            'vllm serve "$LOCAL_DIR_1"/qwen --api-key="$API_KEY" --download-dir="$LOCAL_DIR_2"/hf '
            "--max-num-batched-tokens 8192",
        ),
        (
            'llama-server -m "$MODEL_DIR/my model.gguf" --hf-token hf_x',
            'llama-server -m "$MODEL_DIR"\'/my model.gguf\' --hf-token "$HF_TOKEN"',
        ),
        (
            "sglang serve --model-path /m/a --tokenizer-path /m/b",
            'sglang serve --model-path "$LOCAL_DIR_1"/a --tokenizer-path "$LOCAL_DIR_1"/b',
        ),
        (
            "docker run -e HF_TOKEN=hf_x --env=HUGGING_FACE_HUB_TOKEN=hf_y image@sha256:abc serve",
            'docker run -e HF_TOKEN="$HF_TOKEN" --env=HUGGING_FACE_HUB_TOKEN="$HUGGING_FACE_HUB_TOKEN" '
            "image@sha256:abc serve",
        ),
        ("env HF_TOKEN=hf_x llama-server -hft hf_y", 'env HF_TOKEN="$HF_TOKEN" llama-server -hft "$HF_TOKEN"'),
        (
            "llama-server -m C:/Users/me/q.gguf --host gpu.example.com --port 8080",
            'llama-server -m "$LOCAL_DIR_1"/q.gguf --host "$HOST" --port 8080',
        ),
    ],
    ids=[
        "env-and-flag-secrets",
        "equals-forms",
        "placeholders-stay-expandable",
        "one-directory-one-name",
        "container-environment-secrets",
        "secrets-after-the-first-word",
        "windows-path-and-remote-host",
    ],
)
def test_a_typed_command_loses_its_paths_and_secrets_and_stays_stable(typed: str, shared: str) -> None:
    assert redact_launch_command(typed) == shared
    # Redacting again changes nothing, so a recorded command can be checked by redacting it.
    assert redact_launch_command(shared) == shared


def test_known_paths_become_named_placeholders() -> None:
    command = render_launch_command(
        ("/usr/bin/python3", "-m", "vllm", "serve", "/cache/snapshots/abc", "--port", "8080"),
        (("VLLM_CACHE", "/cache/vllm"),),
        (
            LocalPath(path="/usr/bin/python3", placeholder="$PYTHON"),
            LocalPath(path="/cache/snapshots/abc", placeholder="$MODEL_DIR"),
        ),
    )

    assert command == 'VLLM_CACHE="$LOCAL_DIR_1"/vllm "$PYTHON" -m vllm serve "$MODEL_DIR" --port 8080'


@pytest.mark.parametrize(
    ("typed", "message"),
    [
        ("", "must name the server executable"),
        ("HF_TOKEN=x", "must name the server executable"),
        ("llama-server 'unclosed", "not a valid shell command"),
        ("vllm serve x --revision hf_AbCdEfGhIjKlMnOpQrStUv123", "looks like an access token"),
        ("""vllm serve x --speculative-config '{"model": "/home/me/draft"}'""", "local path inside an argument"),
        ("llama-server -m C:\\Users\\me\\q.gguf", "forward slashes"),
    ],
    ids=["empty", "environment-only", "bad-quoting", "token-shape", "path-inside-json", "windows-backslashes"],
)
def test_a_command_the_rules_cannot_make_safe_is_refused(typed: str, message: str) -> None:
    with pytest.raises(ValueError, match=message) as refusal:
        redact_launch_command(typed)

    # The refusal never repeats what it refused.
    assert "AbCdEf" not in str(refusal.value) and "/home/me" not in str(refusal.value)
