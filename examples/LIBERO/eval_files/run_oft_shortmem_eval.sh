#!/bin/bash
set -euo pipefail

STARVLA_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/data/zhl/Retrieval/starVLA_artifacts}
LIBERO_HOME=${LIBERO_HOME:-/data/zhl/openpi/third_party/libero}
LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-${STARVLA_DIR}/playground/libero_config}
STARVLA_PYTHON=${STARVLA_PYTHON:-$(command -v python)}
if [ -z "${LIBERO_PYTHON:-}" ]; then
  if [ -x /data/zhl/openpi/.conda-envs/openpi-pi05/bin/python ]; then
    LIBERO_PYTHON=/data/zhl/openpi/.conda-envs/openpi-pi05/bin/python
  else
    LIBERO_PYTHON=${STARVLA_PYTHON}
  fi
fi
GPU_ID=${GPU_ID:-0}
PORT=${PORT:-6694}
HOST=${HOST:-127.0.0.1}
NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-50}
WAIT_SECONDS=${WAIT_SECONDS:-20}
CKPT=${CKPT:-${STARVLA_DIR}/playground/Checkpoints/qwenoft_shortmem_libero_all/checkpoints/steps_13000_pytorch_model.pt}

if [ "$#" -gt 0 ]; then
  TASK_SUITES=("$@")
else
  TASK_SUITES=(libero_spatial libero_object libero_goal libero_10)
fi

mkdir -p "${STARVLA_DIR}/playground" "${STARVLA_DIR}/playground/Datasets"
ln -sfn "${ARTIFACT_ROOT}/results" "${STARVLA_DIR}/playground/Checkpoints"
ln -sfn "${ARTIFACT_ROOT}/models" "${STARVLA_DIR}/playground/Pretrained_models"
ln -sfn "${ARTIFACT_ROOT}/datasets/libero" "${STARVLA_DIR}/playground/Datasets/LEROBOT_LIBERO_DATA"

mkdir -p "${LIBERO_CONFIG_PATH}"
cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
assets: ${LIBERO_HOME}/libero/libero/assets
benchmark_root: ${LIBERO_HOME}/libero/libero
bddl_files: ${LIBERO_HOME}/libero/libero/bddl_files
datasets: ${STARVLA_DIR}/playground/Datasets
init_states: ${LIBERO_HOME}/libero/libero/init_files
EOF

export LIBERO_HOME
export LIBERO_CONFIG_PATH
export PYTHONPATH="${LIBERO_HOME}:${STARVLA_DIR}:${PYTHONPATH:-}"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}

if [ ! -f "${CKPT}" ]; then
  echo "[ERROR] checkpoint not found: ${CKPT}" >&2
  exit 1
fi

model_root=$(echo "${CKPT}" | awk -F'/checkpoints/' '{print $1}')
folder_name=$(echo "${CKPT}" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
server_log="${model_root}/logs/server_${folder_name}.log"
mkdir -p "${model_root}/logs"

cleanup() {
  if [ -n "${server_pid:-}" ] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

cd "${STARVLA_DIR}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${STARVLA_PYTHON}" deployment/model_server/server_policy.py \
  --ckpt_path "${CKPT}" \
  --port "${PORT}" \
  --use_bf16 > "${server_log}" 2>&1 &
server_pid=$!

sleep "${WAIT_SECONDS}"
if ! kill -0 "${server_pid}" 2>/dev/null; then
  echo "[ERROR] policy server exited early, see ${server_log}" >&2
  exit 1
fi

for task_suite_name in "${TASK_SUITES[@]}"; do
  video_out_path="${model_root}/videos/${task_suite_name}/${folder_name}"
  log_path="${model_root}/logs/${task_suite_name}"
  mkdir -p "${video_out_path}" "${log_path}"

  "${LIBERO_PYTHON}" ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path "${CKPT}" \
    --args.host "${HOST}" \
    --args.port "${PORT}" \
    --args.task-suite-name "${task_suite_name}" \
    --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
    --args.video-out-path "${video_out_path}" \
    2>&1 | tee "${log_path}/${folder_name}.log"
done
