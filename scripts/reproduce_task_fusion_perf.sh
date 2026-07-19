#!/usr/bin/env bash
set -euo pipefail

# Reproduce the AMR paper's task-fusion performance comparison.
#
# The baseline is CPU-only with every active __inline annotation removed from
# a generated source tree. The optimized variant is CPU-only with the inline
# annotations present in the current source tree.
#
# Usage:
#   source deps/regent-artifact-env.sh
#   scripts/reproduce_task_fusion_perf.sh
#
# Useful overrides:
#   RUNS=3             # repetitions per variant
#   LOOP_CNT=4         # solver launches used for the timed loop
#   PATCH_SIZE=36      # recommended starting point for the CPU-only claim
#   BASE_PATCHES=6     # base patches in each direction
#   LEVEL_MAX=2        # maximum refinement level
#   CPUS=4             # CPU processors used by Legion
#   UTIL=4             # utility processors
#   CSIZE=4096         # MB of CPU system memory
#   PROFILE=1          # emit Legion profiler logs and summary stats

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${ROOT_DIR}/src"
BUILD_DIR="${SRC_DIR}/build"
PROFILE_DIR="${SRC_DIR}/profile/task_fusion"
GEN_ROOT="${BUILD_DIR}/task_fusion_generated"
FUSED_SRC_DIR="${GEN_ROOT}/fused"
UNFUSED_SRC_DIR="${GEN_ROOT}/unfused"

REGENT="${REGENT:-$(command -v regent.py || true)}"
RUNS="${RUNS:-3}"
LOOP_CNT="${LOOP_CNT:-4}"
DT="${DT:-1e-6}"
STRIDE="${STRIDE:-1}"
PATCH_SIZE="${PATCH_SIZE:-36}"
BASE_PATCHES="${BASE_PATCHES:-6}"
LEVEL_MAX="${LEVEL_MAX:-2}"
GHOSTS="${GHOSTS:-4}"
CPUS="${CPUS:-4}"
UTIL="${UTIL:-4}"
CSIZE="${CSIZE:-4096}"
PROFILE="${PROFILE:-0}"

FUSED_TARGET="bench_riemann_task_fused"
UNFUSED_TARGET="bench_riemann_task_unfused"

if [[ -z "${REGENT}" ]]; then
  echo "ERROR: regent.py not found. Run scripts/install_regent_legion.sh and source deps/regent-artifact-env.sh." >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}" "${PROFILE_DIR}" "${GEN_ROOT}"

if [[ -n "${LEGION_DIR:-}" ]]; then
  export LD_LIBRARY_PATH="${LEGION_DIR}/bindings/regent:${LD_LIBRARY_PATH:-}"
fi

python3 - "${SRC_DIR}" "${FUSED_SRC_DIR}" "${UNFUSED_SRC_DIR}" "${PATCH_SIZE}" "${BASE_PATCHES}" "${LEVEL_MAX}" "${GHOSTS}" <<'PY'
import pathlib
import re
import shutil
import sys

src = pathlib.Path(sys.argv[1])
fused = pathlib.Path(sys.argv[2])
unfused = pathlib.Path(sys.argv[3])
patch_size = int(sys.argv[4])
base_patches = int(sys.argv[5])
level_max = int(sys.argv[6])
ghosts = int(sys.argv[7])

for root in (fused, unfused):
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

for path in src.glob("*.rg"):
    text = path.read_text()
    for root in (fused, unfused):
        (root / path.name).write_text(text)

input_text = f'''import "regent"

local config = {{}}

config.eos       = {{}}
config.transport = {{}}

config.num_base_patches_i = {base_patches}
config.num_base_patches_j = {base_patches}
config.patch_size         = {patch_size}
config.level_max          = {level_max}
config.num_ghosts         = {ghosts}

config.numerics_modules = "numerics"

config.domain_length_x =  0.6
config.domain_length_y =  0.6
config.domain_shift_x  = -0.3
config.domain_shift_y  = -0.3

config.eos.Rg    = 1.0
config.eos.gamma = 1.4

config.transport.T_ref    = 1.0
config.transport.mu_ref   = 0.0
config.transport.visc_exp = 0.76
config.transport.Pr       = 0.7

return config
'''

for root in (fused, unfused):
    (root / "input.rg").write_text(input_text)

for path in unfused.glob("*.rg"):
    text = path.read_text()
    text = re.sub(r"^([ \t]*)local\s+__demand\s*\(\s*__leaf\s*,\s*__inline\s*\)[ \t]*$", r"\1local", text, flags=re.MULTILINE)
    text = re.sub(r"^([ \t]*)local\s+__demand\s*\(\s*__inline\s*,\s*__leaf\s*\)[ \t]*$", r"\1local", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*__demand\s*\(\s*__leaf\s*,\s*__inline\s*\)[ \t]*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*__demand\s*\(\s*__inline\s*,\s*__leaf\s*\)[ \t]*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*__demand\s*\(\s*__inline\s*\)[ \t]*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"local\s+__demand\s*\(\s*__inline\s*\)\s+task", "local task", text)
    path.write_text(text)

active_inline = []
for path in unfused.glob("*.rg"):
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        stripped = line.strip()
        if "__inline" in stripped and "--" not in stripped.split("__inline", 1)[0]:
            active_inline.append(f"{path.name}:{lineno}:{stripped}")

if active_inline:
    raise SystemExit("Active __inline annotations remain in unfused tree:\n" + "\n".join(active_inline[:50]))
PY

echo "Using Regent: ${REGENT}"
if [[ -n "${LEGION_DIR:-}" ]]; then
  echo "Using Legion: ${LEGION_DIR}"
  git -C "${LEGION_DIR}" rev-parse HEAD || true
fi
echo "Generated fused source: ${FUSED_SRC_DIR}"
echo "Generated unfused source: ${UNFUSED_SRC_DIR}"
echo "Input: base patches ${BASE_PATCHES}x${BASE_PATCHES}, patch size ${PATCH_SIZE}, level max ${LEVEL_MAX}"
echo "Benchmark loop count: ${LOOP_CNT} solver launches"
echo "Runtime: -ll:cpu ${CPUS} -ll:gpu 0 -ll:util ${UTIL} -ll:csize ${CSIZE}"

echo "Compiling fused CPU variant with active __inline annotations..."
(
  cd "${FUSED_SRC_DIR}"
  AMR_LOOP_CNT="${LOOP_CNT}" AMR_TIME_STEP="${DT}" AMR_STRIDE="${STRIDE}" \
    OBJNAME="${BUILD_DIR}/${FUSED_TARGET}" "${REGENT}" "test_riemann_local_upsample.rg" -fflow 0
)

echo "Compiling unfused CPU baseline with all active __inline annotations removed..."
(
  cd "${UNFUSED_SRC_DIR}"
  AMR_LOOP_CNT="${LOOP_CNT}" AMR_TIME_STEP="${DT}" AMR_STRIDE="${STRIDE}" \
    OBJNAME="${BUILD_DIR}/${UNFUSED_TARGET}" "${REGENT}" "test_riemann_local_upsample.rg" -fflow 0
)

run_one() {
  local label="$1"
  local exe="$2"
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
      "${BUILD_DIR}/${exe}" \
        -ll:cpu "${CPUS}" -ll:gpu 0 -ll:util "${UTIL}" -ll:csize "${CSIZE}" \
        "${prof_flags[@]}"
    ) | tee -a "${stdout_file}"
  done
}

run_one unfused "${UNFUSED_TARGET}"
run_one fused "${FUSED_TARGET}"

python3 - "${PROFILE_DIR}/unfused.stdout" "${PROFILE_DIR}/fused.stdout" <<'PY'
import re
import statistics
import sys

def times(path):
    vals = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.search(r"Time taken:\s*([0-9.eE+-]+)\s*seconds", line)
            if m:
                vals.append(float(m.group(1)))
    return vals

unfused = times(sys.argv[1])
fused = times(sys.argv[2])
if not unfused or not fused:
    raise SystemExit("Could not find 'Time taken:' lines in benchmark output.")

unfused_med = statistics.median(unfused)
fused_med = statistics.median(fused)
speedup = unfused_med / fused_med if fused_med else float("inf")

print("")
print("Summary")
print(f"Unfused CPU times (s): {', '.join(f'{x:.6f}' for x in unfused)}")
print(f"Fused CPU times (s):   {', '.join(f'{x:.6f}' for x in fused)}")
print(f"Median unfused:        {unfused_med:.6f} s")
print(f"Median fused:          {fused_med:.6f} s")
print(f"Task-fusion speedup:   {speedup:.2f}x")
PY

if [[ "${PROFILE}" == "1" ]]; then
  if [[ -n "${LEGION_DIR:-}" && -f "${LEGION_DIR}/tools/legion_prof.py" ]]; then
    for label in unfused fused; do
      logs=("${PROFILE_DIR}/${label}_"*.gz)
      if [[ -e "${logs[0]}" ]]; then
        echo ""
        echo "Generating Legion Prof statistics for ${label}..."
        python3 "${LEGION_DIR}/tools/legion_prof.py" -s -f \
          -o "${PROFILE_DIR}/legion_prof_${label}" \
          "${logs[@]}" | tee "${PROFILE_DIR}/${label}_legion_prof_stats.txt"
        echo ""
        echo "Representative task lines for ${label}:"
        grep -E "solver\\.main|solver\\.calcRHSLaunch|solver\\.calcRHSLeaf|solver\\.calcGradVelColl|ssprk3Stage|Meta-Task Statistics" \
          "${PROFILE_DIR}/${label}_legion_prof_stats.txt" || true
      fi
    done
  fi
fi
