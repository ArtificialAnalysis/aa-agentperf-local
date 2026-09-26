#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="${ROOT_DIR}/.external"
SWEBENCH_DIR="${EXTERNAL_DIR}/SWE-bench"
PYTHON_BIN="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"
SWEBENCH_REPO="${SWEBENCH_REPO:-https://github.com/SWE-bench/SWE-bench.git}"
SWEBENCH_REF="${SWEBENCH_REF:-cd37836ffec01d01a0d699a80a039d84ff2cebfe}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  if [[ -z "${PYTHON+x}" && -x "$(command -v uv)" ]]; then
    (cd "${ROOT_DIR}" && uv sync)
  fi
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  echo "Set PYTHON=/path/to/python, install uv, or create the project venv first." >&2
  exit 1
fi

mkdir -p "${EXTERNAL_DIR}"
if [[ ! -d "${SWEBENCH_DIR}/.git" ]]; then
  git clone "${SWEBENCH_REPO}" "${SWEBENCH_DIR}"
else
  git -C "${SWEBENCH_DIR}" fetch origin
fi

git -C "${SWEBENCH_DIR}" checkout "${SWEBENCH_REF}"
if "${PYTHON_BIN}" -m pip --version >/dev/null 2>&1; then
  "${PYTHON_BIN}" -m pip install -e "${SWEBENCH_DIR}" docker PyYAML
elif command -v uv >/dev/null 2>&1; then
  uv pip install --python "${PYTHON_BIN}" -e "${SWEBENCH_DIR}" docker PyYAML
else
  echo "pip is unavailable for ${PYTHON_BIN}, and uv was not found on PATH." >&2
  exit 1
fi
"${PYTHON_BIN}" - <<'PY'
import docker  # noqa: F401
import yaml  # noqa: F401
import swebench.harness.docker_build  # noqa: F401
import swebench.harness.run_evaluation  # noqa: F401
import swebench.harness.test_spec.test_spec  # noqa: F401

print("SWE-bench harness import ok")
PY
