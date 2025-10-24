#!/bin/bash
source /cvmfs/fcc.cern.ch/sw/latest/setup.sh

# Usage:
#  sh run_AK.sh --plots-only 0.8 (or skip the --plots-only)

set -e

# --- Parse arguments ---
PLOTS_ONLY=false
if [[ "$1" == "--plots-only" ]]; then
  PLOTS_ONLY=true
  shift   # remove the flag so $1,$2,... move down
fi

# --- Read optional radius argument ---
AK_RADIUS=${1:-0.6}

# --- Environment variables ---
export INPUT_DIR=/fs/ddn/sdf/group/atlas/d/gregork/fastsim/jetbenchmarks/22102025/ISR_ecm240
export FOLDER_NAME=GenJetDurhamFastJet_ISR_AK${AK_RADIUS//./}  # Replace dot with nothing for folder name
export JET_ALGO=AK
export AK_RADIUS
export HISTOGRAMS_FOLDER_NAME=Histograms_ECM240

# --- Main execution ---
if [ "$PLOTS_ONLY" = false ]; then
  echo "Running full histmaker..."
  fccanalysis run histmaker_jetE.py

else
  echo "Skipping fccanalysis run... (plots-only mode)"
fi

# Always run final plotting step
python3 simple_histograms.py
python3 resolution_plots.py
fccanalysis plots plots_jetE_alljets.py


