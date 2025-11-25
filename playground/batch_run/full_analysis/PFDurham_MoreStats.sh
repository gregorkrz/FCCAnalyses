source /cvmfs/fcc.cern.ch/sw/latest/setup.sh
set -e

export INPUT_DIR=/fs/ddn/sdf/group/atlas/d/gregork/fastsim/jetbenchmarks/IDEA_20251114
export FOLDER_NAME=PFDurham_ISR_FullyMatched
export JET_ALGO=Durham
export HISTOGRAMS_FOLDER_NAME=Histograms_ECM240_20251114_6M
export JET_MATCHING_RADIUS=1.0
export KEEP_ONLY_FULLY_MATCHED_EVENTS=1

fccanalysis run --n-threads 16  histmaker_jetE_filter_GT.py
python3 simple_histograms.py
python3 resolution_plots.py
fccanalysis plots plots_jetE_alljets_Compare_AK.py
