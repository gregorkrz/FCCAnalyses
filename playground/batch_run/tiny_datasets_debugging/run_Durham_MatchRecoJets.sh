# This script can be used to quickly test the analysis on super small datasets (~500 events)
source /cvmfs/fcc.cern.ch/sw/latest/setup.sh
set -e

export INPUT_DIR=/fs/ddn/sdf/group/atlas/d/gregork/fastsim/jetbenchmarks/Tiny_IDEA_20251105/
export FOLDER_NAME=PFDurham_ISR_MatchRecoJets
export JET_ALGO=Durham
export HISTOGRAMS_FOLDER_NAME=Histograms_20251112_Debug
export JET_MATCHING_RADIUS=1.0
export KEEP_ONLY_FULLY_MATCHED_EVENTS=1
export MATCH_RECO_JETS=1

fccanalysis run histmaker_jetE_filter_GT.py
python3 simple_histograms.py
python3 resolution_plots.py
fccanalysis plots plots_jetE_alljets_Compare_AK.py
