source /cvmfs/fcc.cern.ch/sw/latest/setup.sh
set -e

export INPUT_DIR=/fs/ddn/sdf/group/atlas/d/gregork/fastsim/jetbenchmarks/IDEA_20251114
export FOLDER_NAME=CaloJetDurham_ISR_NoFilter
export JET_ALGO=CaloJetDurham
export HISTOGRAMS_FOLDER_NAME=Histograms_ECM240_20251114_6M_Fix0312_ME
export JET_MATCHING_RADIUS=1.0

fccanalysis run --n-threads 16  histmaker_jetE_filter_GT.py
python3 simple_histograms.py
python3 resolution_plots.py
fccanalysis plots plots_jetE_alljets_Compare_AK.py
