#!/usr/bin/env bash
set -euo pipefail

# Reproduce the AMR paper's __demand(__cuda) performance comparison on the
# Riemann local-upsample benchmark. Both variants are compiled with the same
# 2560 x 2560 input used for the GPU experiment.
#
# Usage:
#   source deps/regent-artifact-env.sh
#   scripts/reproduce_cuda_demand_perf.sh
#
# Useful overrides:
#   GPU_ARCH=pascal        # pascal for Tesla P100, volta, ampere, ...
#   RUNS=3                 # repetitions per variant
#   PROFILE=1              # emit Legion profiler logs under src/profile
#   CSIZE=16384            # MB of CPU system memory for large 2560 x 2560 regions
#   FBMEM=8192 ZCMEM=512   # MB per GPU for framebuffer/zero-copy memory

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${ROOT_DIR}/src"
BUILD_DIR="${SRC_DIR}/build"
PROFILE_DIR="${SRC_DIR}/profile"

REGENT="${REGENT:-$(command -v regent.py || true)}"
GPU_ARCH="${GPU_ARCH:-pascal}"
RUNS="${RUNS:-3}"
LOOP_CNT="${LOOP_CNT:-1}"
DT="${DT:-1e-6}"
STRIDE="${STRIDE:-1}"
CSIZE="${CSIZE:-16384}"
FBMEM="${FBMEM:-8192}"
ZCMEM="${ZCMEM:-512}"
PROFILE="${PROFILE:-0}"

CPU_TARGET="bench_riemann_cpu"
GPU_TARGET="bench_riemann_cuda"
INPUT_FILE="${SRC_DIR}/input_riemann_GPUbig.rg"
INPUT_LINK="${SRC_DIR}/input.rg"

if [[ -z "${REGENT}" ]]; then
  echo "ERROR: regent.py not found. Run scripts/install_regent_legion.sh and source deps/regent-artifact-env.sh." >&2
  exit 1
fi

if [[ ! -f "${INPUT_FILE}" ]]; then
  echo "ERROR: missing ${INPUT_FILE}" >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}" "${PROFILE_DIR}"

ORIG_INPUT=""
ORIG_WAS_LINK=0
ORIG_EXISTED=0
if [[ -e "${INPUT_LINK}" || -L "${INPUT_LINK}" ]]; then
  ORIG_EXISTED=1
  if [[ -L "${INPUT_LINK}" ]]; then
    ORIG_WAS_LINK=1
    ORIG_INPUT="$(readlink "${INPUT_LINK}")"
  else
    ORIG_INPUT="${INPUT_LINK}.bench-backup.$$"
    mv "${INPUT_LINK}" "${ORIG_INPUT}"
  fi
fi

restore_input() {
  rm -f "${INPUT_LINK}"
  if [[ "${ORIG_EXISTED}" == "1" ]]; then
    if [[ "${ORIG_WAS_LINK}" == "1" ]]; then
      ln -s "${ORIG_INPUT}" "${INPUT_LINK}"
    else
      mv "${ORIG_INPUT}" "${INPUT_LINK}"
    fi
  fi
}
trap restore_input EXIT

rm -f "${INPUT_LINK}"
ln -s "${INPUT_FILE}" "${INPUT_LINK}"

echo "Using Regent: ${REGENT}"
if [[ -n "${LEGION_DIR:-}" ]]; then
  echo "Using Legion: ${LEGION_DIR}"
  git -C "${LEGION_DIR}" rev-parse HEAD || true
  export LD_LIBRARY_PATH="${LEGION_DIR}/bindings/regent:${LD_LIBRARY_PATH:-}"
fi
echo "Input: ${INPUT_FILE}"

echo "Compiling CPU baseline without __demand(__cuda) on ssprk3Stage..."
(
  cd "${BUILD_DIR}"
  AMR_LOOP_CNT="${LOOP_CNT}" AMR_TIME_STEP="${DT}" AMR_STRIDE="${STRIDE}" \
    OBJNAME="${CPU_TARGET}" "${REGENT}" "${SRC_DIR}/test_riemann_local_upsample.rg" -fflow 0
)

echo "Compiling CUDA variant with __demand(__cuda) on ssprk3Stage..."
(
  cd "${BUILD_DIR}"
  AMR_LOOP_CNT="${LOOP_CNT}" AMR_TIME_STEP="${DT}" AMR_STRIDE="${STRIDE}" \
    OBJNAME="${GPU_TARGET}" "${REGENT}" "${SRC_DIR}/test_riemann_local_upsample_gpu0.rg" \
    -fflow 0 -fopenmp 0 -fcuda 1 -fcuda-offline 1 -findex-launch 1 -ffuture 0 -fgpu-arch "${GPU_ARCH}"
)

run_one() {
  local label="$1"
  local exe="$2"
  shift 2
  local run_flags=("$@")
  local stdout_file="${PROFILE_DIR}/${label}.stdout"
  : > "${stdout_file}"

  for i in $(seq 1 "${RUNS}"); do
    local prof_flags=()
    if [[ "${PROFILE}" == "1" ]]; then
      prof_flags=(-lg:prof 1 -lg:prof_logfile "${PROFILE_DIR}/${label}_${i}_%.gz")
    fi

    echo "Running ${label}, repetition ${i}/${RUNS}..."
    (
      cd "${SRC_DIR}"
      "${BUILD_DIR}/${exe}" "${LOOP_CNT}" "${DT}" "${STRIDE}" \
        "${run_flags[@]}" "${prof_flags[@]}"
    ) | tee -a "${stdout_file}"
  done
}

run_one cpu "${CPU_TARGET}" -ll:cpu 1 -ll:gpu 0 -ll:util 4 -ll:csize "${CSIZE}"
run_one cuda "${GPU_TARGET}" -ll:cpu 1 -ll:gpu 1 -ll:util 4 -ll:csize "${CSIZE}" -ll:fsize "${FBMEM}" -ll:zsize "${ZCMEM}"

python3 - "${PROFILE_DIR}/cpu.stdout" "${PROFILE_DIR}/cuda.stdout" <<'PY'
import re
import statistics
import sys

def times(path):
    vals = []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.search(r"Time taken:\s*([0-9.eE+-]+)\s*seconds", line)
        if m:
            vals.append(float(m.group(1)))
    return vals

cpu = times(sys.argv[1])
gpu = times(sys.argv[2])
if not cpu or not gpu:
    raise SystemExit("Could not find 'Time taken:' lines in benchmark output.")

cpu_med = statistics.median(cpu)
gpu_med = statistics.median(gpu)
speedup = cpu_med / gpu_med if gpu_med else float("inf")

print("")
print("Summary")
print(f"CPU times (s):  {', '.join(f'{x:.6f}' for x in cpu)}")
print(f"CUDA times (s): {', '.join(f'{x:.6f}' for x in gpu)}")
print(f"Median CPU:     {cpu_med:.6f} s")
print(f"Median CUDA:    {gpu_med:.6f} s")
print(f"Speedup:        {speedup:.2f}x")
PY

if [[ "${PROFILE}" == "1" ]]; then
  if [[ -n "${LEGION_DIR:-}" && -x "${LEGION_DIR}/tools/legion_prof.py" ]]; then
    for label in cpu cuda; do
      logs=("${PROFILE_DIR}/${label}_"*.gz)
      if [[ -e "${logs[0]}" ]]; then
        echo ""
        echo "Generating Legion Prof statistics for ${label}..."
        python3 "${LEGION_DIR}/tools/legion_prof.py" -s -f \
          -o "${PROFILE_DIR}/legion_prof_${label}" \
          "${logs[@]}" | tee "${PROFILE_DIR}/${label}_legion_prof_stats.txt"
        echo ""
        echo "ssprk3Stage lines for ${label}:"
        grep -i "ssprk3Stage" "${PROFILE_DIR}/${label}_legion_prof_stats.txt" || true
      fi
    done
  fi

  cat <<EOF

Legion profiler logs were written to ${PROFILE_DIR}.
To regenerate task-level statistics and timelines:
  python3 "\${LEGION_DIR}/tools/legion_prof.py" -s -f -o ${PROFILE_DIR}/legion_prof_cpu ${PROFILE_DIR}/cpu_*.gz
  python3 "\${LEGION_DIR}/tools/legion_prof.py" -s -f -o ${PROFILE_DIR}/legion_prof_cuda ${PROFILE_DIR}/cuda_*.gz
EOF
fi
