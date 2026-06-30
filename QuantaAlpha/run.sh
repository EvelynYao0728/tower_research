#!/bin/bash
# QuantaAlpha main experiment runner
#
# Usage:
#   ./run.sh "initial direction"                    # default experiment
#   ./run.sh "initial direction" "suffix"           # with factor library suffix
#   CONFIG=configs/experiment.yaml ./run.sh "direction"
#   EXPERIMENT_ID=exp_20260601_093507 RESUME_PATH=./data/results/workspace_.../original_00_01/__session__/0/1_factor_construct STEP_N=3 ./run.sh "direction" "suffix"
#
# Examples:
#   ./run.sh "price-volume factor mining"
#   ./run.sh "momentum reversal factors" "exp_momentum"

# =============================================================================
# Locate project root
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# =============================================================================
# Load .env configuration
# =============================================================================
if [ -f "${SCRIPT_DIR}/.env" ]; then
    set -a
    source "${SCRIPT_DIR}/.env"
    set +a
else
    echo "Error: .env file not found"
    echo "Please run: cp configs/.env.example .env"
    exit 1
fi

# 强制覆盖 shell / conda 里可能残留的 QUANTALPHA_FACTOR_EVAL_WORKERS（如 =40）
export QUANTALPHA_FACTOR_EVAL_WORKERS="${QUANTALPHA_FACTOR_EVAL_WORKERS:-4}"

if [ -n "${QUANTALPHA_LLM_STUB:-}" ]; then
    echo "QUANTALPHA_LLM_STUB=${QUANTALPHA_LLM_STUB} (offline stub LLM)"
fi

# =============================================================================
# Python environment: prefer project .venv, then QUANTALPHA_VENV, then conda
# Set USE_PROJECT_VENV=0 to skip .venv and use conda only.
# Offline / CI (no LLM API): export QUANTALPHA_LLM_STUB=1 before running.
# =============================================================================
USE_PROJECT_VENV="${USE_PROJECT_VENV:-1}"
if [ "${USE_PROJECT_VENV}" = "1" ] && [ -f "${SCRIPT_DIR}/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/.venv/bin/activate"
    echo "Using project venv: ${SCRIPT_DIR}/.venv"
elif [ -n "${QUANTALPHA_VENV:-}" ] && [ -f "${QUANTALPHA_VENV}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${QUANTALPHA_VENV}/bin/activate"
    echo "Using QUANTALPHA_VENV: ${QUANTALPHA_VENV}"
else
    eval "$(conda shell.bash hook)" 2>/dev/null
    conda activate "${CONDA_ENV_NAME:-quantaalpha}" 2>/dev/null

    if [ $? -ne 0 ]; then
        source activate "${CONDA_ENV_NAME:-quantaalpha}" 2>/dev/null
    fi
fi

if ! command -v quantaalpha &> /dev/null; then
    echo "Error: quantaalpha command not found. Please install: pip install -e ."
    exit 1
fi

echo "Python: $(python --version)"
echo "QuantaAlpha: $(which quantaalpha)"
echo ""

# =============================================================================
# Experiment isolation
# =============================================================================
CONFIG_PATH="${CONFIG_PATH:-configs/experiment.yaml}"
if [ -n "${CONFIG:-}" ]; then
    CONFIG_PATH="${CONFIG}"
fi

if [ -z "${EXPERIMENT_ID}" ]; then
    EXPERIMENT_ID="exp_$(date +%Y%m%d_%H%M%S)"
fi
export EXPERIMENT_ID

RESULTS_BASE="${DATA_RESULTS_DIR:-./data/results}"

if [ "${EXPERIMENT_ID}" != "shared" ]; then
    export WORKSPACE_PATH="${RESULTS_BASE}/workspace_${EXPERIMENT_ID}"
    export PICKLE_CACHE_FOLDER_PATH_STR="${RESULTS_BASE}/pickle_cache_${EXPERIMENT_ID}"
    mkdir -p "${WORKSPACE_PATH}" "${PICKLE_CACHE_FOLDER_PATH_STR}"
    echo "Experiment ID: ${EXPERIMENT_ID}"
    echo "Workspace: ${WORKSPACE_PATH}"
fi

# =============================================================================
# Validate private minute feature roots (see quantaalpha.data.private_catalog)
# =============================================================================
LEGACY_ROOT="${QUANTALPHA_LEGACY_PANEL_ROOT:-/home/yzyao.25/research/data/simple_factors}"
PER_FEAT_ROOT="${QUANTALPHA_PER_FEATURE_ROOT:-/home/yzyao.25/research/data/0511simple_factors}"
for p in "${LEGACY_ROOT}" "${PER_FEAT_ROOT}"; do
    if [ ! -e "${p}" ]; then
        echo "Error: data path does not exist: ${p}"
        echo "Set QUANTALPHA_LEGACY_PANEL_ROOT / QUANTALPHA_PER_FEATURE_ROOT if needed."
        exit 1
    fi
done
echo "Private feature roots OK: ${LEGACY_ROOT}"
echo "                         : ${PER_FEAT_ROOT}"

# =============================================================================
# Parse arguments and run
# =============================================================================
DIRECTION="$1"
LIBRARY_SUFFIX="$2"

if [ -n "${LIBRARY_SUFFIX}" ]; then
    export FACTOR_LIBRARY_SUFFIX="${LIBRARY_SUFFIX}"
fi

echo ""
echo "Starting experiment..."
echo "Config: ${CONFIG_PATH}"
echo "Data: ${LEGACY_ROOT} + ${PER_FEAT_ROOT}"
echo "Results: ${RESULTS_BASE}"
echo "----------------------------------------"

if [ -n "${RESUME_PATH}" ]; then
    if [ -z "${STEP_N}" ]; then
        echo "Error: RESUME_PATH requires STEP_N (remaining workflow steps to run)"
        exit 1
    fi
    if [ ! -f "${RESUME_PATH}" ]; then
        echo "Error: session checkpoint not found: ${RESUME_PATH}"
        exit 1
    fi
    echo "Resume session: ${RESUME_PATH}"
    echo "Remaining steps: ${STEP_N}"
    quantaalpha mine --direction "${DIRECTION}" --path "${RESUME_PATH}" --step_n "${STEP_N}" --config_path "${CONFIG_PATH}"
elif [ -n "${STEP_N}" ]; then
    quantaalpha mine --direction "${DIRECTION}" --step_n "${STEP_N}" --config_path "${CONFIG_PATH}"
else
    quantaalpha mine --direction "${DIRECTION}" --config_path "${CONFIG_PATH}"
fi
