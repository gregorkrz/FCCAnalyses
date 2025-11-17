# This script can be used to quickly test the analysis on super small datasets (~500 events)
source /cvmfs/fcc.cern.ch/sw/latest/setup.sh
set -e

export INPUT_DIR=/fs/ddn/sdf/group/atlas/d/gregork/fastsim/jetbenchmarks/Tiny_IDEA_20251105_NoMuonInCalo/
export FOLDER_NAME=CaloJetDurham_ISR
export JET_ALGO=CaloJetDurham
export HISTOGRAMS_FOLDER_NAME=Histograms_Tiny_IDEA_20251105_MuonDebug

fccanalysis run histmaker_jetE_filter_GT.py
python3 simple_histograms.py
python3 resolution_plots.py
fccanalysis plots plots_jetE_alljets_Compare_AK.py
