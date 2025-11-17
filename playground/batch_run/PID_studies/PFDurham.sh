source /cvmfs/fcc.cern.ch/sw/latest/setup.sh
set -e

export INPUT_DIR=/fs/ddn/sdf/group/atlas/d/gregork/fastsim/jetbenchmarks/Tiny_IDEA_20251105_NoMuonInCalo/
export FOLDER_NAME=CaloJetDurham_ISR
export JET_ALGO=CaloJetDurham
export HISTOGRAMS_FOLDER_NAME=Histograms_Tiny_IDEA_20251105_MuonDebug


fccanalysis run PID_composition_histograms.py
fccanalysis plots PID_composition_histograms_plots.py
