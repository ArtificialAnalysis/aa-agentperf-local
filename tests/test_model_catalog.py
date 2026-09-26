"""Exercise the typed pilot model catalog boundary."""

from dataclasses import replace
from pathlib import Path

import orjson
import pytest

from agentperf_local.common.json_types import JsonObject, normalize_json_object
from agentperf_local.deployment.catalog import (
    BUNDLED_MODEL_CATALOG_DIGEST,
    BUNDLED_MODEL_CATALOG_PATH,
    MAX_MODEL_CATALOG_BYTES,
    DeploymentFramework,
    ModelCandidate,
    ModelCatalog,
    ModelDeployment,
    load_model_catalog,
)
from agentperf_local.deployment.context_policy import derived_minimum_memory_bytes

CATALOG_PATH = BUNDLED_MODEL_CATALOG_PATH


def test_bundled_catalog_matches_its_pinned_digest() -> None:
    assert load_model_catalog(BUNDLED_MODEL_CATALOG_PATH).file_digest == BUNDLED_MODEL_CATALOG_DIGEST


def _named(catalog: ModelCatalog, profile_id: str) -> ModelCandidate:
    """Return one catalog model by name, so tests never depend on catalog order."""
    return next(model for model in catalog.models if model.profile_id == profile_id)


def _deployment(catalog: ModelCatalog, profile_id: str) -> ModelDeployment:
    """Return one model's managed recipe, which every bundled candidate carries."""
    deployment = _named(catalog, profile_id).deployment
    assert deployment is not None
    return deployment


def _catalog_object() -> JsonObject:
    return normalize_json_object(orjson.loads(CATALOG_PATH.read_bytes()))


def _model(data: JsonObject, index: int) -> JsonObject:
    models = data.get("models")
    assert isinstance(models, list)
    model = models[index]
    assert isinstance(model, dict)
    return model


def _device(data: JsonObject, model_index: int, device_index: int) -> JsonObject:
    evidence = _model(data, model_index).get("device_evidence")
    assert isinstance(evidence, list)
    device = evidence[device_index]
    assert isinstance(device, dict)
    return device


def test_loads_pilot_candidates_as_typed_records_in_source_order() -> None:
    """Expose the exact pilot intent without promoting its trust state."""
    catalog = load_model_catalog(CATALOG_PATH)

    assert catalog.status == "pilot-candidates-not-release"
    assert catalog.as_of == "2026-09-21"
    # Portable GGUF recipes lead the catalog, so the default choice runs anywhere.
    assert [model.profile_id for model in catalog.models] == [
        "gemma4-12b-it-q4-0",
        "gemma4-26b-a4b-q4-0",
        "qwen38-27b-q4-k-m",
        "qwen38-27b-q4-k-m-mtp",
        "qwen38-27b-nvfp4-dgx-spark",
        "gemma4-26b-a4b-nvfp4-dgx-spark",
        "gemma4-31b-it-nvfp4-dgx-spark",
        "nemotron3-super-120b-a12b-nvfp4-mtp-dgx-spark",
        "nemotron35-lightning-30b-a3b-nvfp4-dspark-dgx-spark",
        "qwen35-122b-a10b-nvfp4-mtp-dgx-spark",
        "qwen36-27b-nvfp4-mtp-dgx-spark",
        "qwen36-35b-a3b-nvfp4-mtp-dgx-spark",
        "gemma4-26b-a4b-nvfp4",
        "gpt-oss-120b-mxfp4",
        "gemma4-26b-a4b-q4-k-m-mtp-rtx5090",
        "muse-glimmer-30b-q4-k-m-dflash-rtx5090",
        "nemotron35-lightning-30b-a3b-q4-k-m-dflash-rtx5090",
        "qwen35-9b-q4-k-m-mtp-rtx5090",
        "qwen36-27b-q4-k-m-mtp-rtx5090",
        "qwen36-35b-a3b-q4-k-m-mtp-rtx5090",
        "gemma4-26b-a4b-q4-k-m-mtp-strix-halo",
        "muse-glimmer-30b-q4-k-m-dflash-strix-halo",
        "nemotron35-lightning-30b-a3b-q4-k-m-dflash-strix-halo",
        "qwen35-9b-q4-k-m-mtp-strix-halo",
        "qwen36-27b-q4-k-m-mtp-strix-halo",
        "qwen36-35b-a3b-q4-k-m-mtp-strix-halo",
        "qwen38-27b-q4-k-m-mtp-strix-halo",
        "qwen38-flash-next-iq4-nl-mtp-strix-halo",
        "qwen38-27b-q4-k-m-mtp-m5-pro",
    ]
    assert [model.hf_revision for model in catalog.models] == [
        "29d097773436b69ff9feafd636ab4cf873786537",
        "d1c082be9cf3c8a514acf63b8761f4b41935842e",
        "f1bfb127c64f7072bdd2cad55f258b9c8b2910fe",
        "f1bfb127c64f7072bdd2cad55f258b9c8b2910fe",
        "319f741cce68d7914884900c138a1fbb70a42f30",
        "a19cfe00be84568a6867111c9a68c9c44fdcffe6",
        "4135a98a9b728a548947683219633b25682223ac",
        "ff433f5493e25d631c9f12b5d55c674229923d02",
        "bee7596271d1495f6992ae224aefde4410e816b8",
        "98915d837c4e7c87ac8296d02e89de19b3207e6d",
        "0893e1606ff3d5f97a441f405d5fc541a6bdf404",
        "1355db6a052410cfd62085d94b58866fd0f2c3c5",
        "a19cfe00be84568a6867111c9a68c9c44fdcffe6",
        "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
        "c099eb48e663fd284577b04978a94ffccb261841",
        "70bf1b61ac09f91b24d39038091b41c582bc5d7a",
        "f2d3fe3694501008786e81e5f20360cbf715496a",
        "9716a636ee4bddc3fed678220b7a33dd2a4160ae",
        "5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace",
        "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d",
        "c099eb48e663fd284577b04978a94ffccb261841",
        "70bf1b61ac09f91b24d39038091b41c582bc5d7a",
        "f2d3fe3694501008786e81e5f20360cbf715496a",
        "9716a636ee4bddc3fed678220b7a33dd2a4160ae",
        "5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace",
        "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d",
        "f1bfb127c64f7072bdd2cad55f258b9c8b2910fe",
        "ba5b0d696d6997d82fcc55ba3f8e6128db6e0311",
        "f1bfb127c64f7072bdd2cad55f258b9c8b2910fe",
    ]
    assert all(
        evidence.aa_admission_state == "unqualified" for model in catalog.models for evidence in model.device_evidence
    )
    assert all(model.artifact_manifest_status == "complete-file-sha256-pinned" for model in catalog.models)
    # The DGX Spark NVFP4 recipe is a safetensors weights repository served by vLLM on CUDA.
    qwen_weights = _named(catalog, "qwen38-27b-nvfp4-dgx-spark")
    assert [evidence.device_id for evidence in qwen_weights.device_evidence] == ["nvidia-cuda"]
    assert qwen_weights.deployment is not None
    assert qwen_weights.deployment.artifact_kind == "safetensors-repository"
    assert qwen_weights.deployment.frameworks == ("vllm",)
    assert qwen_weights.deployment.runtime_version_for("vllm") == "0.28.0"
    assert qwen_weights.deployment.vllm is not None
    dgx_spark = [model for model in catalog.models if model.profile_id.endswith("dgx-spark")]
    assert len(dgx_spark) == 8
    for candidate in dgx_spark:
        assert candidate.deployment is not None
        assert candidate.deployment.context_tokens == 65536
        assert candidate.deployment.frameworks == ("vllm",)
        assert candidate.deployment.vllm is not None
    # SGLang profiles share their qualified release. llama.cpp publishes no comparable
    # release version to pin.
    weights = [
        model.deployment
        for model in catalog.models
        if model.deployment is not None and "sglang" in model.deployment.frameworks
    ]
    assert {deployment.runtime_version_for("sglang") for deployment in weights} == {"0.5.18"}
    assert _deployment(catalog, "gemma4-12b-it-q4-0").runtime_versions == ()
    gemma = _named(catalog, "gemma4-12b-it-q4-0")
    assert gemma.deployment is not None
    assert gemma.deployment.artifact_kind == "gguf-single-file"
    assert gemma.deployment.artifacts[0].sha256 == ("93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b")
    assert gemma.deployment.frameworks == ("llama-cpp",)
    assert [evidence.device_id for evidence in gemma.device_evidence] == [
        "nvidia-cuda",
        "amd-rocm",
        "apple-silicon",
    ]
    rtx_profile_ids = (
        "gemma4-26b-a4b-q4-k-m-mtp-rtx5090",
        "muse-glimmer-30b-q4-k-m-dflash-rtx5090",
        "nemotron35-lightning-30b-a3b-q4-k-m-dflash-rtx5090",
        "qwen35-9b-q4-k-m-mtp-rtx5090",
        "qwen36-27b-q4-k-m-mtp-rtx5090",
        "qwen36-35b-a3b-q4-k-m-mtp-rtx5090",
        "qwen38-27b-q4-k-m-mtp",
    )
    for profile_id in rtx_profile_ids:
        candidate = _named(catalog, profile_id)
        assert candidate.deployment is not None
        assert candidate.deployment.context_tokens == 65536
        assert candidate.deployment.llama_cpp is not None
        assert [
            (evidence.tested_input_tokens, evidence.tested_output_tokens, evidence.tested_concurrency)
            for evidence in candidate.device_evidence
        ] == [(120000, 1, 1)]
    nemotron = _deployment(catalog, "nemotron35-lightning-30b-a3b-q4-k-m-dflash-rtx5090")
    assert nemotron.artifact_kind == "gguf-file-set"
    draft = next(artifact for artifact in nemotron.artifacts if artifact.filename.startswith("dflash-"))
    assert draft.source_repository == "apolo13x/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-DFlash-GGUF"
    assert draft.source_revision == "3051796f1bcf60ac44a27c2f79c52a2a2b3e2b37"
    flash_next = _deployment(catalog, "qwen38-flash-next-iq4-nl-mtp-strix-halo")
    assert flash_next.llama_cpp is not None
    assert flash_next.llama_cpp.lazy_mode == "on-direct"
    # The lazily read per-layer-embedding table stays on disk, so the floor is below the download size.
    assert flash_next.minimum_memory_bytes < flash_next.artifact_size_bytes
    assert derived_minimum_memory_bytes(flash_next, 65536) == flash_next.minimum_memory_bytes


def test_managed_candidate_can_limit_hardware_and_framework_compatibility() -> None:
    catalog = load_model_catalog(CATALOG_PATH)
    gemma = _named(catalog, "gemma4-12b-it-q4-0")
    assert gemma.deployment is not None

    limited = replace(gemma, device_evidence=(gemma.device_evidence[0],))

    assert limited.deployment is not None
    assert limited.deployment.frameworks == ("llama-cpp",)
    assert tuple(evidence.device_id for evidence in limited.device_evidence) == ("nvidia-cuda",)


@pytest.mark.parametrize(
    ("source_profile", "screened_profile"),
    (
        ("gemma4-26b-a4b-q4-k-m-mtp-rtx5090", "gemma4-26b-a4b-q4-k-m-mtp-strix-halo"),
        ("muse-glimmer-30b-q4-k-m-dflash-rtx5090", "muse-glimmer-30b-q4-k-m-dflash-strix-halo"),
        ("nemotron35-lightning-30b-a3b-q4-k-m-dflash-rtx5090", "nemotron35-lightning-30b-a3b-q4-k-m-dflash-strix-halo"),
        ("qwen35-9b-q4-k-m-mtp-rtx5090", "qwen35-9b-q4-k-m-mtp-strix-halo"),
        ("qwen36-27b-q4-k-m-mtp-rtx5090", "qwen36-27b-q4-k-m-mtp-strix-halo"),
        ("qwen36-35b-a3b-q4-k-m-mtp-rtx5090", "qwen36-35b-a3b-q4-k-m-mtp-strix-halo"),
        ("qwen38-27b-q4-k-m-mtp", "qwen38-27b-q4-k-m-mtp-strix-halo"),
        ("qwen38-27b-q4-k-m-mtp", "qwen38-27b-q4-k-m-mtp-m5-pro"),
    ),
)
def test_screened_profiles_preserve_artifacts_without_promoting_evidence(
    source_profile: str, screened_profile: str
) -> None:
    catalog = load_model_catalog(CATALOG_PATH)
    source = _named(catalog, source_profile)
    candidate = _named(catalog, screened_profile)
    assert candidate.hf_repository == source.hf_repository
    assert candidate.hf_revision == source.hf_revision
    assert source.deployment is not None and candidate.deployment is not None
    deployment = candidate.deployment
    assert deployment.artifacts == source.deployment.artifacts
    assert derived_minimum_memory_bytes(deployment, deployment.context_tokens) == deployment.minimum_memory_bytes
    assert deployment.llama_cpp is not None
    assert deployment.llama_cpp.flash_attention and deployment.llama_cpp.disable_fit
    assert deployment.llama_cpp.cache_ram_mib == 0
    assert deployment.llama_cpp.draft_backend_sampling
    assert len(candidate.device_evidence) == 1
    evidence = candidate.device_evidence[0]
    assert evidence.evidence_level == "local-inference-screen"
    assert evidence.aa_admission_state == "unqualified"
    assert evidence.tested_output_tokens == (1 if screened_profile.startswith("qwen35-9b") else 256)


@pytest.mark.parametrize("case", ("empty", "wrong-runtime", "out-of-order", "duplicate"))
def test_managed_frameworks_must_be_a_unique_canonical_subset(case: str) -> None:
    catalog = load_model_catalog(CATALOG_PATH)
    gemma = _named(catalog, "gemma4-12b-it-q4-0")
    deployment = gemma.deployment
    assert isinstance(deployment, ModelDeployment)

    frameworks: tuple[DeploymentFramework, ...]
    if case == "empty":
        frameworks = ()
    elif case == "wrong-runtime":
        frameworks = ("sglang",)
    elif case == "out-of-order":
        frameworks = ("sglang", "llama-cpp")
    else:
        frameworks = ("llama-cpp", "llama-cpp")

    with pytest.raises(ValueError, match="at least one framework|llama.cpp|frameworks must"):
        replace(deployment, frameworks=frameworks)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("unexpected-envelope-field", "catalog has invalid fields"),
        ("missing-envelope-field", "missing as_of"),
        ("wrong-version", "catalog.version must be 2"),
        ("release-status", "catalog.status must be pilot-candidates-not-release"),
        ("unknown-model-field", "unexpected verified"),
        ("duplicate-profile", "profile_id values must be unique"),
        ("device-order", "canonical subset of CUDA, ROCm, and Apple Silicon"),
        ("device-architecture", "device architecture does not match"),
        ("partial-test-shape", "must be all present or all null"),
        ("multiline-display-name", "display_name must be short printable text"),
        ("llama-backend-device", "backend must match the recipe's device evidence"),
        ("llama-threads-zero", "threads must be a positive integer"),
        ("llama-negative-cache", "cache_ram_mib must be non-negative"),
        ("llama-lazy-read-without-lazy-mode", "only a llama.cpp recipe with a lazy_mode can read artifact bytes"),
    ],
)
def test_rejects_malformed_or_promoted_catalog_data(tmp_path: Path, case: str, message: str) -> None:
    """Fail closed before malformed JSON can become UI model choices."""
    data = _catalog_object()
    if case == "unexpected-envelope-field":
        data["authority"] = "aa-signed"
    elif case == "missing-envelope-field":
        del data["as_of"]
    elif case == "wrong-version":
        data["version"] = 3
    elif case == "release-status":
        data["status"] = "release"
    elif case == "unknown-model-field":
        _model(data, 0)["verified"] = True
    elif case == "duplicate-profile":
        _model(data, 1)["profile_id"] = _model(data, 0)["profile_id"]
    elif case == "device-order":
        evidence = _model(data, 0).get("device_evidence")
        assert isinstance(evidence, list)
        evidence.reverse()
    elif case == "device-architecture":
        _device(data, 0, 0)["architecture"] = "rocm"
    elif case == "partial-test-shape":
        # The Qwen NVFP4 DGX Spark recipe is the one entry that pins a complete tested shape.
        _device(data, 4, 0)["tested_output_tokens"] = None
    elif case == "multiline-display-name":
        _model(data, 0)["display_name"] = "spoofed heading\nAA VERIFIED"
    elif case.startswith("llama-"):
        deployment = _model(data, -1).get("deployment")
        assert isinstance(deployment, dict)
        launch = deployment.get("llama_cpp")
        assert isinstance(launch, dict)
        if case == "llama-backend-device":
            launch["backend"] = "rocm"
        elif case == "llama-threads-zero":
            launch["threads"] = 0
        elif case == "llama-lazy-read-without-lazy-mode":
            memory = deployment.get("memory")
            assert isinstance(memory, dict)
            memory["lazy_read_bytes"] = 1
        else:
            launch["cache_ram_mib"] = -1
    catalog_path = tmp_path / f"{case}.json"
    catalog_path.write_bytes(orjson.dumps(data))

    with pytest.raises(ValueError, match=message):
        load_model_catalog(catalog_path)


@pytest.mark.parametrize("encoded", [b"not json", b"[]"])
def test_rejects_invalid_json_and_non_object_roots(tmp_path: Path, encoded: bytes) -> None:
    """Reject input before catalog records are exposed."""
    path = tmp_path / "catalog.json"
    path.write_bytes(encoded)

    with pytest.raises(ValueError):
        load_model_catalog(path)


def test_rejects_symbolic_linked_catalog(tmp_path: Path) -> None:
    """Do not follow a catalog path that can change after selection."""
    target = tmp_path / "catalog.json"
    target.write_bytes(CATALOG_PATH.read_bytes())
    link = tmp_path / "catalog-link.json"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="symbolic link"):
        load_model_catalog(link)


def test_rejects_oversized_catalog_before_parsing(tmp_path: Path) -> None:
    """Bound memory use for a user-supplied catalog path."""
    path = tmp_path / "oversized-catalog.json"
    path.write_bytes(b" " * (MAX_MODEL_CATALOG_BYTES + 1))

    with pytest.raises(ValueError, match="must not exceed"):
        load_model_catalog(path)
