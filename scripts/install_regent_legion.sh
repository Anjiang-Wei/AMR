#!/usr/bin/env bash
set -euo pipefail

# Install the Legion/Regent revision used by Artifact.pdf with
# language/scripts/setup_env.py:
# c61071541218747e35767317f6f89b83f374f264.
#
# Usage:
#   scripts/install_regent_legion.sh [install-prefix]
#
# Defaults to ./deps/legion-artifact. The script writes
# ./deps/regent-artifact-env.sh, which can be sourced before compiling AMR.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEGION_COMMIT="${LEGION_COMMIT:-c61071541218747e35767317f6f89b83f374f264}"
PREFIX="${1:-${ROOT_DIR}/deps/legion-artifact}"
LEGION_DIR="${PREFIX}/legion"
DEPS_DIR="${PREFIX}/deps"
SCRATCH_DIR="${PREFIX}/scratch"
ENV_FILE="${ROOT_DIR}/deps/regent-artifact-env.sh"

mkdir -p "${PREFIX}" "${DEPS_DIR}" "${SCRATCH_DIR}" "$(dirname "${ENV_FILE}")"

if [[ ! -d "${LEGION_DIR}/.git" ]]; then
  git clone https://github.com/StanfordLegion/legion.git "${LEGION_DIR}"
else
  git -C "${LEGION_DIR}" fetch --tags origin
fi

git -C "${LEGION_DIR}" checkout "${LEGION_COMMIT}"

export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
export USE_GASNET="${USE_GASNET:-0}"
export GPU_ARCH="${GPU_ARCH:-pascal}"

if [[ -n "${CUDA_HOME:-}" ]]; then
  export CUDA="${CUDA_HOME}"
fi

if [[ "${ENABLE_CUDA:-1}" == "1" ]]; then
  export USE_CUDA=1
  if [[ -z "${CUDA:-}" && -n "$(command -v nvcc || true)" ]]; then
    export CUDA="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
  fi
fi

python3 "${LEGION_DIR}/language/scripts/setup_env.py" \
  --prefix "${DEPS_DIR}" \
  --scratch "${SCRATCH_DIR}" \
  -j "${JOBS:-16}"

cat > "${ENV_FILE}" <<EOF
export LEGION_DIR="${LEGION_DIR}"
export LG_RT_DIR="${LEGION_DIR}/runtime"
export PATH="${LEGION_DIR}/language:\${PATH}"
export LD_LIBRARY_PATH="${LEGION_DIR}/bindings/regent:\${LD_LIBRARY_PATH:-}"
EOF

echo "Installed Legion/Regent at ${LEGION_DIR}"
echo "Pinned commit: ${LEGION_COMMIT}"
echo "To use it: source ${ENV_FILE}"
