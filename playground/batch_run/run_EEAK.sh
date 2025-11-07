#!/bin/bash

# Save the caller's args, clear them for the sourced script
ARGS=("$@")
set --


# Restore the caller's args
set -- "${ARGS[@]}"

# --- Parse arguments ---
PLOTS_ONLY=false
if [[ "$1" == "--plots-only" ]]; then
  PLOTS_ONLY=true
  shift
fi

AK_RADIUS=${1:-0.6}

source /cvmfs/fcc.cern.ch/sw/latest/setup.sh
set -e

# Usage:
#  sh run_AK.sh --plots-only 0.8 (or skip the --plots-only)


# --- Environment variables ---
export INPUT_DIR=/fs/ddn/sdf/group/atlas/d/gregork/fastsim/jetbenchmarks/IDEA_20251105
export FOLDER_NAME=ISR_EEAK${AK_RADIUS//./}  # Replace dot with nothing for folder name
export JET_ALGO=EEAK
export AK_RADIUS
export HISTOGRAMS_FOLDER_NAME=Histograms_ECM240_20251105

# --- Main execution ---
if [ "$PLOTS_ONLY" = false ]; then
  echo "Running full histmaker..."
  fccanalysis run histmaker_jetE_filter_GT.py

else
  echo "Skipping fccanalysis run... (plots-only mode)"
fi

# Always run final plotting step
python3 simple_histograms.py
python3 resolution_plots.py
fccanalysis plots plots_jetE_alljets.py

