#!/usr/bin/env bash
#
# run_seoul.sh — launch a Seoul air-pollution β-VAE scenario-generation experiment.
#
# It drives the real pipeline in ../src (via test.py) and takes care of the one manual
# step of the original workflow: selecting the pollutant by placing its CSV where the
# dataset definition (SEOUL_dataset.json) expects it.
#
# Usage:
#   ./THESIS_SEOUL/run_seoul.sh <POLLUTANT> <NODES> [SEED_BEGIN] [SEED_END]
#
#   POLLUTANT : PM25 | PM10 | NO2 | O3 | SO2
#   NODES     : 16 | 25
#   SEED_BEGIN: first random seed (default 0)
#   SEED_END  : last random seed, exclusive (default SEED_BEGIN+1)
#
# Examples:
#   ./THESIS_SEOUL/run_seoul.sh PM25 16 0 10
#   ./THESIS_SEOUL/run_seoul.sh O3   25
#
set -euo pipefail

# --- locate the repository root (parent of this script's folder) ---------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- arguments -----------------------------------------------------------------------
POLLUTANT="${1:-}"
NODES="${2:-}"
SEED_B="${3:-0}"
SEED_E="${4:-$((SEED_B + 1))}"

usage() {
    echo "Usage: $0 <PM25|PM10|NO2|O3|SO2> <16|25> [SEED_BEGIN] [SEED_END]" >&2
    exit 1
}

[[ -z "$POLLUTANT" || -z "$NODES" ]] && usage

case "$POLLUTANT" in PM25|PM10|NO2|O3|SO2) ;; *) echo "ERROR: unknown pollutant '$POLLUTANT'" >&2; usage ;; esac
case "$NODES"     in 16|25) ;;              *) echo "ERROR: nodes must be 16 or 25" >&2;        usage ;; esac

# --- map experiment id ---------------------------------------------------------------
NUM_CASE="bvae_0${NODES}_seoul"

# --- resolve the dataset files -------------------------------------------------------
# NOTE: all Seoul CSVs (both n16 and n25) live in the SEOUL_SMALLGRAPH_16 folder.
DATA_DIR="$REPO_ROOT/data/neuroCorrelation_dataset/KOREA_SEOUL/SEOUL_SMALLGRAPH/SEOUL_SMALLGRAPH_16"

# The NO2 file has a special suffix (Wasserstein instability, n25 only).
if [[ "$POLLUTANT" == "NO2" ]]; then
    SRC_CSV="$DATA_DIR/SEOUL_GRAPH_n${NODES}_2019_data_NO2_wasserstein_problem.csv"
else
    SRC_CSV="$DATA_DIR/SEOUL_GRAPH_n${NODES}_2019_data_${POLLUTANT}.csv"
fi

# Generic filename that SEOUL_dataset.json reads for time-slot "B".
GENERIC_CSV="$DATA_DIR/SEOUL_GRAPH_n${NODES}_2019_data.csv"

if [[ ! -f "$SRC_CSV" ]]; then
    echo "ERROR: pollutant dataset not found:" >&2
    echo "       $SRC_CSV" >&2
    echo "       (this pollutant/nodes combination is not available)" >&2
    exit 1
fi

# Back up an existing generic file once, then install the chosen pollutant.
if [[ -f "$GENERIC_CSV" && ! -f "$GENERIC_CSV.orig" ]]; then
    cp "$GENERIC_CSV" "$GENERIC_CSV.orig"
fi
cp "$SRC_CSV" "$GENERIC_CSV"

# --- experiment naming ---------------------------------------------------------------
SUFFIX="SEOUL_${NODES}_${POLLUTANT}"
MAIN_FOLDER="THESIS_SEOUL_${POLLUTANT}_${NODES}"

echo "|----------------------------------------"
echo "| Seoul scenario generation"
echo "|   pollutant : $POLLUTANT"
echo "|   nodes     : $NODES   (num_case=$NUM_CASE)"
echo "|   seeds     : [$SEED_B, $SEED_E)"
echo "|   dataset   : $(basename "$SRC_CSV")"
echo "|   output    : data/neuroCorrelation_experiments/$MAIN_FOLDER"
echo "|----------------------------------------"

# --- run the real pipeline from the repo root ----------------------------------------
cd "$REPO_ROOT"
python test.py --exp neuroD \
    --num_case "$NUM_CASE" \
    --experiment_name_suffix "$SUFFIX" \
    --main_folder "$MAIN_FOLDER" \
    --repeation_b "$SEED_B" \
    --repeation_e "$SEED_E" \
    --optimization no \
    --load_model no \
    --train_models yes \
    --time_slot B
