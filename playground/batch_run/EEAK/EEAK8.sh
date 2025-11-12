source /cvmfs/fcc.cern.ch/sw/latest/setup.sh
set -e

# --- Environment variables ---
export INPUT_DIR=/fs/ddn/sdf/group/atlas/d/gregork/fastsim/jetbenchmarks/IDEA_20251105
export FOLDER_NAME=ISR_EEAK8
export JET_ALGO=EEAK
export AK_RADIUS=0.8
export HISTOGRAMS_FOLDER_NAME=Histograms_ECM240_20251105

#fccanalysis run histmaker_jetE_filter_GT.py
python3 simple_histograms.py
python3 resolution_plots.py
fccanalysis plots plots_jetE_alljets_Compare_AK.py


