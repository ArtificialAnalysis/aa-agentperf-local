"""Exercise the recipe folder boundary."""

import re
import shutil
from pathlib import Path

import pytest
import yaml

from agentperf_local.common.json_types import JsonObject, normalize_json_object
from agentperf_local.common.models import replace_fields
from agentperf_local.deployment.catalog import (
    BUNDLED_RECIPES_DIGEST,
    BUNDLED_RECIPES_ROOT,
    MAX_RECIPE_BYTES,
    DeploymentFramework,
    ModelCandidate,
    ModelCatalog,
    ModelDeployment,
    load_model_catalog,
)

CATALOG_PATH = BUNDLED_RECIPES_ROOT
GEMMA_RECIPE = Path("gemma-4-12b", "any", "gemma4-12b-it-q4-0.yaml")
METAL_RECIPE = Path("qwen3-8-27b", "m5-pro", "qwen38-27b-q4-k-m-mtp-m5-pro.yaml")
SPLASH_RECIPE = Path("qwen3-8-27b", "m5-pro", "qwen38-27b-splash-dflash.yaml")


def test_bundled_catalog_matches_its_pinned_digest() -> None:
    digest = load_model_catalog(BUNDLED_RECIPES_ROOT).digest
    assert digest == BUNDLED_RECIPES_DIGEST, f"recipes changed; set BUNDLED_RECIPES_DIGEST to {digest}"


def _named(catalog: ModelCatalog, profile_id: str) -> ModelCandidate:
    """Return one catalog model by name, so tests never depend on catalog order."""
    return next(model for model in catalog.models if model.profile_id == profile_id)


def _read(root: Path, recipe: Path) -> JsonObject:
    return normalize_json_object(yaml.safe_load((root / recipe).read_bytes()))


def _write(root: Path, recipe: Path, data: JsonObject) -> None:
    (root / recipe).write_text(yaml.safe_dump(data, sort_keys=False))


def test_loads_recipes_as_typed_records_in_path_order() -> None:
    """Expose the exact pilot intent without promoting its trust state."""
    catalog = load_model_catalog(CATALOG_PATH)

    assert catalog.as_of == "2026-09-30"
    # Recipes load in recipes/<model_release_slug>/<hardware>/ order, and "any" sorts first within a model.
    assert [(model.profile_id, model.hf_revision) for model in catalog.models] == [
        ("gemma4-12b-it-q4-0", "29d097773436b69ff9feafd636ab4cf873786537"),
        ("gemma4-26b-a4b-q4-0", "d1c082be9cf3c8a514acf63b8761f4b41935842e"),
        ("gemma4-26b-a4b-nvfp4-dgx-spark", "a19cfe00be84568a6867111c9a68c9c44fdcffe6"),
        ("gemma4-26b-a4b-nvfp4", "a19cfe00be84568a6867111c9a68c9c44fdcffe6"),
        ("gemma4-26b-a4b-q4-k-m-mtp-rtx5090", "c099eb48e663fd284577b04978a94ffccb261841"),
        ("gemma4-26b-a4b-q4-k-m-mtp-strix-halo", "c099eb48e663fd284577b04978a94ffccb261841"),
        ("gemma4-31b-it-nvfp4-dgx-spark", "4135a98a9b728a548947683219633b25682223ac"),
        ("gpt-oss-120b-mxfp4", "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"),
        ("ling3-flash-q4-k-m", "29edce5130491e638f3e5ebec91fa0a5ed4f780e"),
        ("ling3-flash-q4-k-m-dspark-dgx-spark", "29edce5130491e638f3e5ebec91fa0a5ed4f780e"),
        ("ling3-flash-q4-k-m-dspark-strix-halo", "29edce5130491e638f3e5ebec91fa0a5ed4f780e"),
        ("muse-glimmer-30b-q4-k-m-dflash-rtx5090", "70bf1b61ac09f91b24d39038091b41c582bc5d7a"),
        ("muse-glimmer-30b-q4-k-m-dflash-strix-halo", "70bf1b61ac09f91b24d39038091b41c582bc5d7a"),
        ("nemotron35-lightning-30b-a3b-nvfp4-dspark-dgx-spark", "bee7596271d1495f6992ae224aefde4410e816b8"),
        ("nemotron35-lightning-30b-a3b-q4-k-m-dflash-rtx5090", "f2d3fe3694501008786e81e5f20360cbf715496a"),
        ("nemotron35-lightning-30b-a3b-q4-k-m-dflash-strix-halo", "f2d3fe3694501008786e81e5f20360cbf715496a"),
        ("nemotron3-super-120b-a12b-nvfp4-mtp-dgx-spark", "ff433f5493e25d631c9f12b5d55c674229923d02"),
        ("qwen35-122b-a10b-nvfp4-mtp-dgx-spark", "98915d837c4e7c87ac8296d02e89de19b3207e6d"),
        ("qwen35-9b-nvfp4-mtp-dgx-spark", "97aef92393f126bf649f310cd40861be8dad3279"),
        ("qwen35-9b-q4-k-m-mtp-m5-pro", "9716a636ee4bddc3fed678220b7a33dd2a4160ae"),
        ("qwen35-9b-q4-k-m-mtp-rtx5090", "9716a636ee4bddc3fed678220b7a33dd2a4160ae"),
        ("qwen35-9b-q4-k-m-mtp-strix-halo", "9716a636ee4bddc3fed678220b7a33dd2a4160ae"),
        ("qwen36-27b-nvfp4-mtp-dgx-spark", "0893e1606ff3d5f97a441f405d5fc541a6bdf404"),
        ("qwen36-27b-q4-k-m-mtp-rtx5090", "5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace"),
        ("qwen36-27b-q4-k-m-mtp-strix-halo", "5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace"),
        ("qwen36-35b-a3b-nvfp4-mtp-dgx-spark", "1355db6a052410cfd62085d94b58866fd0f2c3c5"),
        ("qwen36-35b-a3b-q4-k-m-mtp-m5-pro", "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d"),
        ("qwen36-35b-a3b-q4-k-m-mtp-rtx5090", "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d"),
        ("qwen36-35b-a3b-q4-k-m-mtp-strix-halo", "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d"),
        ("qwen38-27b-q4-k-m", "f1bfb127c64f7072bdd2cad55f258b9c8b2910fe"),
        ("qwen38-27b-nvfp4-dgx-spark", "319f741cce68d7914884900c138a1fbb70a42f30"),
        ("qwen38-27b-q4-k-m-mtp-m5-pro", "f1bfb127c64f7072bdd2cad55f258b9c8b2910fe"),
        ("qwen38-27b-splash-dflash", "9d27070b71f7142c6b6025f03ac011d70a73cb48"),
        ("qwen38-27b-q4-k-m-mtp", "f1bfb127c64f7072bdd2cad55f258b9c8b2910fe"),
        ("qwen38-27b-q4-k-m-mtp-strix-halo", "f1bfb127c64f7072bdd2cad55f258b9c8b2910fe"),
    ]
    # The DGX Spark NVFP4 recipe is a safetensors weights repository served by vLLM on CUDA.
    qwen_weights = _named(catalog, "qwen38-27b-nvfp4-dgx-spark")
    assert qwen_weights.devices == ("nvidia-cuda",)
    assert qwen_weights.deployment.artifact_kind == "safetensors-repository"
    assert qwen_weights.deployment.frameworks == ("vllm",)
    assert qwen_weights.deployment.runtime_version_for("vllm") == "0.28.0"
    assert qwen_weights.deployment.vllm is not None
    dgx_spark = [model for model in catalog.models if model.profile_id.endswith("dgx-spark")]
    assert len(dgx_spark) == 10
    for candidate in dgx_spark:
        assert candidate.deployment.context_tokens == 65536
    # Ling 3.0 flash is the one DGX Spark recipe that llama.cpp serves; the rest run vLLM.
    assert [model.profile_id for model in dgx_spark if model.deployment.vllm is None] == [
        "ling3-flash-q4-k-m-dspark-dgx-spark"
    ]
    # SGLang profiles share their qualified release. llama.cpp publishes no comparable
    # release version to pin.
    weights = [model.deployment for model in catalog.models if "sglang" in model.deployment.frameworks]
    assert {deployment.runtime_version_for("sglang") for deployment in weights} == {"0.5.18"}
    assert _named(catalog, "gemma4-12b-it-q4-0").deployment.runtime_versions == ()
    gemma = _named(catalog, "gemma4-12b-it-q4-0")
    assert gemma.deployment.artifact_kind == "gguf-single-file"
    assert gemma.deployment.artifacts[0].sha256 == ("93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b")
    assert gemma.deployment.frameworks == ("llama-cpp",)
    assert gemma.devices == ("nvidia-cuda", "amd-rocm", "apple-silicon")
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
        assert candidate.devices == ("nvidia-cuda",)
        assert candidate.deployment.llama_cpp is not None
    nemotron = _named(catalog, "nemotron35-lightning-30b-a3b-q4-k-m-dflash-rtx5090").deployment
    assert nemotron.artifact_kind == "gguf-file-set"
    draft = next(artifact for artifact in nemotron.artifacts if artifact.filename.startswith("dflash-"))
    assert draft.source_repository == "apolo13x/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-DFlash-GGUF"
    assert draft.source_revision == "3051796f1bcf60ac44a27c2f79c52a2a2b3e2b37"
    # Ling's target-only recipe drafts nothing; its DSpark recipes cap the checkpoints that copy the draft KV cache.
    ling = _named(catalog, "ling3-flash-q4-k-m").deployment.llama_cpp
    assert ling is not None
    assert ling.speculative_tokens is None
    ling_dspark = _named(catalog, "ling3-flash-q4-k-m-dspark-dgx-spark").deployment.llama_cpp
    assert ling_dspark is not None
    assert ling_dspark.context_checkpoints == 2
    # Splash serves its own packed package on Apple Silicon and pins the release it was run on.
    splash = _named(catalog, "qwen38-27b-splash-dflash")
    assert splash.devices == ("apple-silicon",)
    assert splash.deployment.artifact_kind == "splash-package"
    assert splash.deployment.frameworks == ("splash",)
    assert splash.deployment.runtime_version_for("splash") == "1.0.2"


def test_managed_candidate_can_limit_hardware_and_framework_compatibility() -> None:
    catalog = load_model_catalog(CATALOG_PATH)
    gemma = _named(catalog, "gemma4-12b-it-q4-0")

    limited = replace_fields(gemma, devices=("nvidia-cuda",))

    assert limited.deployment.frameworks == ("llama-cpp",)
    assert limited.devices == ("nvidia-cuda",)


@pytest.mark.parametrize(
    ("source_profile", "screened_profile"),
    (
        ("gemma4-26b-a4b-q4-k-m-mtp-rtx5090", "gemma4-26b-a4b-q4-k-m-mtp-strix-halo"),
        ("muse-glimmer-30b-q4-k-m-dflash-rtx5090", "muse-glimmer-30b-q4-k-m-dflash-strix-halo"),
        ("nemotron35-lightning-30b-a3b-q4-k-m-dflash-rtx5090", "nemotron35-lightning-30b-a3b-q4-k-m-dflash-strix-halo"),
        ("qwen35-9b-q4-k-m-mtp-rtx5090", "qwen35-9b-q4-k-m-mtp-strix-halo"),
        ("qwen35-9b-q4-k-m-mtp-rtx5090", "qwen35-9b-q4-k-m-mtp-m5-pro"),
        ("qwen36-27b-q4-k-m-mtp-rtx5090", "qwen36-27b-q4-k-m-mtp-strix-halo"),
        ("qwen36-35b-a3b-q4-k-m-mtp-rtx5090", "qwen36-35b-a3b-q4-k-m-mtp-strix-halo"),
        ("qwen36-35b-a3b-q4-k-m-mtp-rtx5090", "qwen36-35b-a3b-q4-k-m-mtp-m5-pro"),
        ("qwen38-27b-q4-k-m-mtp", "qwen38-27b-q4-k-m-mtp-strix-halo"),
        ("qwen38-27b-q4-k-m-mtp", "qwen38-27b-q4-k-m-mtp-m5-pro"),
    ),
)
def test_device_recipes_reuse_the_source_artifacts(source_profile: str, screened_profile: str) -> None:
    catalog = load_model_catalog(CATALOG_PATH)
    source = _named(catalog, source_profile)
    candidate = _named(catalog, screened_profile)
    assert candidate.hf_repository == source.hf_repository
    assert candidate.hf_revision == source.hf_revision
    deployment = candidate.deployment
    assert deployment.artifacts == source.deployment.artifacts
    assert deployment.llama_cpp is not None
    assert deployment.llama_cpp.flash_attention and deployment.llama_cpp.disable_fit
    assert deployment.llama_cpp.cache_ram_mib == 0
    assert deployment.llama_cpp.draft_backend_sampling
    assert len(candidate.devices) == 1


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
        replace_fields(deployment, frameworks=frameworks)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("unknown-model-field", f"{GEMMA_RECIPE.as_posix()}: verified: Extra inputs are not permitted"),
        ("missing-as-of", f"{GEMMA_RECIPE.as_posix()}: as_of: Field required"),
        ("malformed-as-of", "as_of must be an ISO date"),
        ("unquoted-as-of", "quote dates"),
        ("misnamed-file", "must be named after its profile_id"),
        ("slug-folder-mismatch", "must sit in the folder of its model_release_slug gemma-4-31b"),
        ("malformed-slug", "model_release_slug must be letters and digits joined by single dots or hyphens"),
        ("model-alias", f"{GEMMA_RECIPE.as_posix()}: deployment.model_alias: Extra inputs are not permitted"),
        ("stray-file", "recipes must be .yaml files"),
        ("flat-recipe", "may only hold folders"),
        ("device-order", "canonical subset of nvidia-cuda, amd-rocm, and apple-silicon"),
        (
            "unknown-device",
            f"{GEMMA_RECIPE.as_posix()}: devices.0: Input should be 'nvidia-cuda', 'amd-rocm' or 'apple-silicon'",
        ),
        ("reduced-context", "context_tokens must be 65536"),
        ("multiline-model-name", "model_name must be short printable text"),
        ("model-name-mismatch", "every recipe in gemma-4-12b must share the model_name Gemma 4 12B"),
        ("hardware-field", f"{GEMMA_RECIPE.as_posix()}: hardware: Extra inputs are not permitted"),
        ("same-shown-name", "would show the same name; give one a variant"),
        ("llama-backend-device", "backend must match the recipe's devices"),
        (
            "llama-threads-zero",
            f"{METAL_RECIPE.as_posix()}: deployment.llama_cpp.threads: Input should be greater than 0",
        ),
        ("llama-negative-cache", "cache_ram_mib must be non-negative"),
        ("llama-lazy-read-without-lazy-mode", "only a llama.cpp recipe with a lazy_mode can read artifact bytes"),
        ("llama-speculative-without-depth", "a speculative llama.cpp recipe must set speculative_tokens"),
        ("llama-target-only-with-depth", "a target-only llama.cpp recipe must not set speculative_tokens"),
        ("splash-other-runtime", "a Splash package is served by Splash alone"),
        ("splash-missing-manifest", "a Splash package recipe must pin manifest.json"),
        ("splash-missing-draft", "a Splash package recipe must pin files under draft/"),
    ],
)
def test_rejects_malformed_or_promoted_recipes(tmp_path: Path, case: str, message: str) -> None:
    """Fail closed before a malformed recipe can become a UI model choice."""
    root = tmp_path / "recipes"
    shutil.copytree(CATALOG_PATH, root)
    gemma = _read(root, GEMMA_RECIPE)
    if case == "unknown-model-field":
        gemma["verified"] = True
    elif case == "missing-as-of":
        del gemma["as_of"]
    elif case == "malformed-as-of":
        gemma["as_of"] = "2026-9-21"
    elif case == "unquoted-as-of":
        (root / GEMMA_RECIPE).write_text((root / GEMMA_RECIPE).read_text().replace("'2026-09-21'", "2026-09-21"))
    elif case == "misnamed-file":
        (root / GEMMA_RECIPE).rename(root / GEMMA_RECIPE.with_name("gemma.yaml"))
    elif case == "slug-folder-mismatch":
        gemma["model_release_slug"] = "gemma-4-31b"
    elif case == "malformed-slug":
        gemma["model_release_slug"] = "gemma--4"
    elif case == "model-alias":
        deployment = gemma.get("deployment")
        assert isinstance(deployment, dict)
        deployment["model_alias"] = "gemma"
    elif case == "stray-file":
        (root / GEMMA_RECIPE).with_suffix(".yml").write_text("")
    elif case == "flat-recipe":
        shutil.copy(root / GEMMA_RECIPE, root / GEMMA_RECIPE.name)
    elif case == "device-order":
        devices = gemma.get("devices")
        assert isinstance(devices, list)
        devices.reverse()
    elif case == "unknown-device":
        gemma["devices"] = ["rtx-5090"]
    elif case == "reduced-context":
        deployment = gemma.get("deployment")
        assert isinstance(deployment, dict)
        deployment["context_tokens"] = 32768
    elif case == "multiline-model-name":
        gemma["model_name"] = "spoofed heading\nAA VERIFIED"
    elif case == "same-shown-name":
        gemma["profile_id"] = "gemma4-12b-it-q4-0-twin"
        _write(root, GEMMA_RECIPE.with_name("gemma4-12b-it-q4-0-twin.yaml"), gemma)
        gemma = _read(root, GEMMA_RECIPE)
    elif case == "hardware-field":
        gemma["hardware"] = "rtx-5090"
    elif case == "model-name-mismatch":
        shutil.copytree(root / GEMMA_RECIPE.parent, root / "gemma-4-12b" / "nvidia-cuda")
        sibling = root / "gemma-4-12b" / "nvidia-cuda" / GEMMA_RECIPE.name
        sibling.rename(sibling.with_name("gemma4-12b-sibling.yaml"))
        gemma["profile_id"] = "gemma4-12b-sibling"
        gemma["model_name"] = "Gemma Four"
        _write(root, Path("gemma-4-12b", "nvidia-cuda", "gemma4-12b-sibling.yaml"), gemma)
        gemma = _read(root, GEMMA_RECIPE)
    elif case.startswith("llama-"):
        metal = _read(root, METAL_RECIPE)
        deployment = metal.get("deployment")
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
        elif case == "llama-speculative-without-depth":
            del launch["speculative_tokens"]
        elif case == "llama-target-only-with-depth":
            metal["speculation_policy"] = "disabled-target-only-baseline"
        else:
            launch["cache_ram_mib"] = -1
        _write(root, METAL_RECIPE, metal)
    elif case.startswith("splash-"):
        splash = _read(root, SPLASH_RECIPE)
        deployment = splash.get("deployment")
        assert isinstance(deployment, dict)
        artifacts = deployment.get("artifacts")
        assert isinstance(artifacts, list)
        if case == "splash-other-runtime":
            deployment["frameworks"] = ["llama-cpp"]
            del deployment["runtime_versions"]
        else:
            dropped = "manifest.json" if case == "splash-missing-manifest" else "draft/"
            deployment["artifacts"] = [
                artifact
                for artifact in artifacts
                if not (isinstance(artifact, dict) and str(artifact.get("filename")).startswith(dropped))
            ]
        _write(root, SPLASH_RECIPE, splash)
    if case not in ("unquoted-as-of", "misnamed-file", "stray-file", "flat-recipe"):
        _write(root, GEMMA_RECIPE, gemma)

    with pytest.raises(ValueError, match=re.escape(message)):
        load_model_catalog(root)


@pytest.mark.parametrize(("encoded", "message"), [(b"key: [unclosed", "invalid recipe YAML"), (b"- a list", "mapping")])
def test_rejects_invalid_yaml_and_non_mapping_recipes(tmp_path: Path, encoded: bytes, message: str) -> None:
    """Reject input before catalog records are exposed."""
    root = tmp_path / "recipes"
    shutil.copytree(CATALOG_PATH, root)
    (root / GEMMA_RECIPE).write_bytes(encoded)

    with pytest.raises(ValueError, match=message):
        load_model_catalog(root)


def test_rejects_symbolic_linked_recipe(tmp_path: Path) -> None:
    """Do not follow a recipe path that can change after selection."""
    root = tmp_path / "recipes"
    shutil.copytree(CATALOG_PATH, root)
    target = tmp_path / "outside.yaml"
    (root / GEMMA_RECIPE).rename(target)
    (root / GEMMA_RECIPE).symlink_to(target)

    with pytest.raises(ValueError, match="must be a regular file"):
        load_model_catalog(root)


def test_rejects_oversized_recipe_before_parsing(tmp_path: Path) -> None:
    """Bound memory use for a user-supplied recipe folder."""
    root = tmp_path / "recipes"
    shutil.copytree(CATALOG_PATH, root)
    (root / GEMMA_RECIPE).write_bytes(b" " * (MAX_RECIPE_BYTES + 1))

    with pytest.raises(ValueError, match="outside the accepted range"):
        load_model_catalog(root)
