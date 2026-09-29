"""Build one valid managed submission body for tests that exercise the upload path."""

from agentperf_local.common.units import BYTES_PER_GIB
from agentperf_local.deployment.qualification import QUALIFICATION_PROBE_IDS
from agentperf_local.submission.contract import (
    Accelerator,
    Benchmark,
    CappedOutputPolicy,
    Client,
    Hardware,
    ManagedDeployment,
    Power,
    Qualification,
    QualificationOutcome,
    Run,
    SubmissionRequest,
    Turn,
)
from agentperf_local.submission.notice import PRIVACY_NOTICE_VERSION

SAMPLE_RUN_ID = "8f5b2f2e-4c3a-4d6e-9b1a-2c3d4e5f6a7b"


def sample_submission(*, wall_duration_ms: float = 1_000.0) -> SubmissionRequest:
    """Return a managed NVIDIA run of one turn; change wall_duration_ms to change its content."""
    return SubmissionRequest(
        run_id=SAMPLE_RUN_ID,
        privacy_notice_version=PRIVACY_NOTICE_VERSION,
        client=Client(name="agentperf-local", version="0.3.0", source_revision="1" * 40, source_state="release"),
        benchmark=Benchmark(
            workload_digest=f"sha256:{'2' * 64}",
            workload_context_tokens=65_536,
            context_tokens=65_536,
            observed_context_tokens=65_536,
        ),
        hardware=Hardware(
            platform_family="linux",
            architecture="x86_64",
            memory_bytes=64 * BYTES_PER_GIB,
            accelerator=Accelerator(
                vendor="nvidia",
                product="NVIDIA GeForce RTX 5090",
                memory_bytes=32 * BYTES_PER_GIB,
                memory_is_unified=False,
                driver_version="580.95.05",
            ),
        ),
        deployment=ManagedDeployment(
            deployment_mode="managed",
            model_release_slug="qwen3-8-27b",
            hf_repository="unsloth/Qwen3.8-27B-GGUF",
            hf_revision="3" * 40,
            framework="llama-cpp",
            framework_version="version: 6890 (c1d0e7a00)",
            framework_commit="c1d0e7a004015f23bc0233470b747b596f29b264",
            server_launch_command='llama-server --model "$MODEL_DIR"/model.gguf',
            accelerator_backend="cuda",
            profile_id="qwen38-27b-q4-k-m-mtp",
            model_artifact_digest=f"sha256:{'4' * 64}",
            recipe="profile_id: qwen38-27b-q4-k-m-mtp\n",
            model_size_bytes=17 * BYTES_PER_GIB,
        ),
        policy=CappedOutputPolicy(
            client_backend="python",
            transport_policy_id="direct-sse-no-retry-v1",
            cache_isolation_enabled=True,
            cache_isolation_mode="run_namespace_prefix",
            output_token_policy="exact",
            output_token_fallback=16_384,
            output_token_margin=0,
        ),
        run=Run(wall_duration_ms=wall_duration_ms, observer_duration_ms=0.0),
        turns=(
            Turn(
                turn_ordinal=0,
                task_ordinal=0,
                turn_in_task=0,
                finish_reason="length",
                response_chunks=4,
                replayed_pacing_ms=0.0,
                e2e_latency_ms=500.0,
                time_to_first_token_ms=100.0,
                generation_ms=400.0,
                server_prompt_tokens=100,
                server_output_tokens=50,
                target_output_tokens=50,
                cached_input_tokens=60,
                uncached_input_tokens=40,
            ),
        ),
        qualification=Qualification(
            synthetic_pack_id="aa-runtime-synthetic-v1",
            outcomes=tuple(
                QualificationOutcome(probe_id=probe_id, passed=True, failure_codes=())
                for probe_id in QUALIFICATION_PROBE_IDS
            ),
        ),
        power=Power(
            collector_id="aa-nvidia-smi-v1",
            sampled_power_energy_valid=True,
            power_coverage=1.0,
            power_integration_coverage=1.0,
            energy_joules=200.0,
        ),
    )
