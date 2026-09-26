#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "${PYTHON+x}" ]]; then
  PYTHON_BIN="$PYTHON"
elif [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${ROOT}/.venv/bin/python"
else
  PYTHON_BIN="$(command -v python3 || true)"
fi

detect_default_arch() {
  case "$(uname -m)" in
    arm64|aarch64) echo arm64 ;;
    *) echo x86_64 ;;
  esac
}

ARCH="${AGENTPERF_LOCAL_IMAGE_ARCH:-$(detect_default_arch)}"
MANIFEST=""
SWEBENCH_INSTANCES="agentperf_local/data/replays/agentperf-default-v1/swebench-instances.jsonl"
PINCHBENCH_IMAGE="${PINCHBENCH_IMAGE:-agentperf-pinchbench:latest}"
SWEBENCH_NAMESPACE="${SWEBENCH_NAMESPACE:-docker.io/swebench}"
SWEBENCH_TAG="${SWEBENCH_TAG:-latest}"
MAX_WORKERS="${MAX_WORKERS:-2}"
FORCE_REBUILD=0
SKIP_PINCHBENCH=0
SKIP_SWEBENCH=0
LIST_ONLY=0

usage() {
  cat <<'EOF'
Usage: scripts/build-default-containers.sh [options]

Build or fetch Docker images used by the bundled agentperf-default-v1 replay.

Options:
  --arch ARCH                 Image architecture segment: arm64 or x86_64. Defaults to the host architecture.
  --manifest FILE             Default-compatible replay manifest.
  --pinchbench-image IMAGE    PinchBench image tag to build. Default: agentperf-pinchbench:latest.
  --swebench-instances FILE   Bundled SWE-bench instance metadata JSONL.
  --swebench-namespace NAME   SWE-bench image namespace. Default: docker.io/swebench.
  --swebench-tag TAG          SWE-bench image tag. Default: latest.
  --max-workers N             Parallel SWE-bench image builds. Default: 2.
  --force-rebuild             Rebuild images even when Docker already has them.
  --skip-pinchbench           Do not build the PinchBench image.
  --skip-swebench             Do not build or pull SWE-bench task images.
  --list-only                 Print target image names without building.
  -h, --help                  Show this help.

For arm64 SWE-bench image builds, run scripts/install-swebench-validation.sh first.
EOF
}

need_value() {
  if [[ $# -lt 2 ]]; then
    echo "Missing value for $1" >&2
    usage >&2
    exit 2
  fi
}

require_positive_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$name must be a positive integer, got: $value" >&2
    exit 2
  fi
}

assert_image_arch() {
  local image="$1"
  local actual_arch
  actual_arch="$(docker image inspect "$image" --format '{{.Architecture}}')"
  if [[ "$actual_arch" != "$DOCKER_ARCH" ]]; then
    echo "Image $image has architecture $actual_arch, expected $DOCKER_ARCH" >&2
    exit 1
  fi
  echo "Verified $image architecture: $actual_arch"
}

require_python() {
  if [[ -z "$PYTHON_BIN" ]]; then
    echo "Python is required to read the default manifest. Set PYTHON=/path/to/python or create the project venv." >&2
    exit 1
  fi
}

# The single reader of the manifest's SWE-bench image names, for listing, pulling and verifying.
swebench_images() {
  require_python
  "$PYTHON_BIN" - "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
for task in manifest["tasks"]:
    if task.get("adapter") == "swebench":
        print(task["tool_environment"]["image"])
PY
}

verify_swebench_image_arch() {
  swebench_images | while IFS= read -r image; do assert_image_arch "$image"; done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch) need_value "$@"; ARCH="$2"; shift 2 ;;
    --manifest) need_value "$@"; MANIFEST="$2"; shift 2 ;;
    --pinchbench-image) need_value "$@"; PINCHBENCH_IMAGE="$2"; shift 2 ;;
    --swebench-instances) need_value "$@"; SWEBENCH_INSTANCES="$2"; shift 2 ;;
    --swebench-namespace) need_value "$@"; SWEBENCH_NAMESPACE="$2"; shift 2 ;;
    --swebench-tag) need_value "$@"; SWEBENCH_TAG="$2"; shift 2 ;;
    --max-workers) need_value "$@"; MAX_WORKERS="$2"; shift 2 ;;
    --force-rebuild) FORCE_REBUILD=1; shift ;;
    --skip-pinchbench) SKIP_PINCHBENCH=1; shift ;;
    --skip-swebench) SKIP_SWEBENCH=1; shift ;;
    --list-only) LIST_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$ARCH" != "arm64" && "$ARCH" != "x86_64" ]]; then
  echo "--arch must be arm64 or x86_64, got: $ARCH" >&2
  exit 2
fi
DOCKER_ARCH="${ARCH/x86_64/amd64}"
PINCHBENCH_PLATFORM="${PINCHBENCH_PLATFORM:-linux/${DOCKER_ARCH}}"
MANIFEST="${MANIFEST:-agentperf_local/data/replays/agentperf-default-v1/manifest-${ARCH}.json}"
if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST" >&2
  exit 2
fi
if [[ ! -f "$SWEBENCH_INSTANCES" ]]; then
  echo "SWE-bench instance metadata not found: $SWEBENCH_INSTANCES" >&2
  exit 2
fi
require_positive_integer "--max-workers" "$MAX_WORKERS"

if [[ "$LIST_ONLY" == "1" ]]; then
  echo "PinchBench: $PINCHBENCH_IMAGE platform=$PINCHBENCH_PLATFORM"
  swebench_images | sed 's/^/SWE-bench: /'
  exit 0
fi

if [[ "$SKIP_PINCHBENCH" != "1" ]]; then
  docker build --platform "$PINCHBENCH_PLATFORM" \
    -t "$PINCHBENCH_IMAGE" \
    -f docker/pinchbench/Dockerfile docker/pinchbench
  assert_image_arch "$PINCHBENCH_IMAGE"
fi

if [[ "$SKIP_SWEBENCH" == "1" ]]; then
  exit 0
fi

if [[ "$ARCH" == "x86_64" ]]; then
  while IFS= read -r image; do
    echo "SWE-bench: $image"
    docker pull "$image"
  done < <(swebench_images)
  verify_swebench_image_arch
  exit 0
fi

require_python

"$PYTHON_BIN" - \
  "$MANIFEST" \
  "$SWEBENCH_INSTANCES" \
  "$SWEBENCH_NAMESPACE" \
  "$SWEBENCH_TAG" \
  "$MAX_WORKERS" \
  "$FORCE_REBUILD" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path.cwd()
sys.path.insert(0, str(root / ".external" / "SWE-bench"))

try:
    import docker
    import yaml
    from swebench.harness.docker_build import build_instance_images
    from swebench.harness.test_spec.test_spec import make_test_spec
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing optional SWE-bench build dependency. "
        "Run scripts/install-swebench-validation.sh first. "
        f"Missing module: {exc.name}"
    ) from exc

manifest_path = Path(sys.argv[1])
instances_path = Path(sys.argv[2])
namespace = sys.argv[3]
tag = sys.argv[4]
max_workers = int(sys.argv[5])
force_rebuild = sys.argv[6] == "1"

conda_skip_packages = {
    "_libgcc_mutex",
    "_openmp_mutex",
    "libgcc-ng",
    "libgomp",
    "libstdcxx-ng",
}


def relax_conda_package(dep):
    if not isinstance(dep, str):
        return dep
    name = dep.split("=", 1)[0]
    if name in conda_skip_packages or name.startswith("ld_impl_linux-"):
        return None
    parts = dep.split("=")
    if len(parts) >= 2:
        return "=".join(parts[:2])
    return dep


def relax_arm64_conda_environment(command: str) -> str:
    match = re.search(
        r"cat <<'(?P<marker>[^']+)' > /root/environment\.yml\n"
        r"(?P<body>.*?)"
        r"\n(?P=marker)",
        command,
        flags=re.DOTALL,
    )
    if match is None:
        return command

    environment = yaml.safe_load(match.group("body"))
    channels = environment.get("channels") or []
    environment["channels"] = list(dict.fromkeys(["conda-forge", *channels, "defaults"]))
    dependencies = []
    for dep in environment.get("dependencies", []):
        relaxed = relax_conda_package(dep)
        if relaxed is not None:
            dependencies.append(relaxed)
    environment["dependencies"] = dependencies
    body = yaml.safe_dump(environment, sort_keys=False)
    return command[: match.start("body")] + body.rstrip() + command[match.end("body") :]


def relax_spec_for_arm64(spec):
    spec.env_script_list = [relax_arm64_conda_environment(command) for command in spec.env_script_list]
    return spec


instances = {}
for line in instances_path.read_text().splitlines():
    if line.strip():
        instance = json.loads(line)
        instances[instance["instance_id"]] = instance

manifest = json.loads(manifest_path.read_text())
specs = []
for task in manifest["tasks"]:
    if task.get("adapter") != "swebench":
        continue
    task_id = task["task_id"]
    try:
        instance = instances[task_id]
    except KeyError as exc:
        raise SystemExit(f"Missing SWE-bench instance metadata for {task_id}") from exc
    specs.append(
        relax_spec_for_arm64(
            make_test_spec(
                instance,
                namespace=namespace,
                instance_image_tag=tag,
                env_image_tag=tag,
                arch="arm64",
            )
        )
    )

if not specs:
    raise SystemExit("Manifest contains no SWE-bench tasks.")

print("SWE-bench arm64 instance images:")
for spec in specs:
    print(f"  {spec.instance_image_key}")

client = docker.from_env()
successful, failed = build_instance_images(
    client=client,
    dataset=specs,
    force_rebuild=force_rebuild,
    max_workers=max_workers,
    namespace=namespace,
    tag=tag,
    env_image_tag=tag,
)
if failed:
    raise SystemExit(f"{len(failed)} SWE-bench image build(s) failed: {failed}")
print(f"Successfully built or found {len(successful)} SWE-bench image(s).")
PY

verify_swebench_image_arch
