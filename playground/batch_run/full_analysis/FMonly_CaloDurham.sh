source /cvmfs/fcc.cern.ch/sw/latest/setup.sh
set -e

export INPUT_DIR=/fs/ddn/sdf/group/atlas/d/gregork/fastsim/jetbenchmarks/IDEA_20251105
export FOLDER_NAME=CaloJetDurham_ISR_FullyMatched_MatchRecoJets
export JET_ALGO=CaloJetDurham
export HISTOGRAMS_FOLDER_NAME=Histograms_ECM240_20251105_MatchR10
export JET_MATCHING_RADIUS=1.0
export KEEP_ONLY_FULLY_MATCHED_EVENTS=1
export MATCH_RECO_JETS=1

fccanalysis run --n-threads 16  histmaker_jetE_filter_GT.py
python3 simple_histograms.py
python3 resolution_plots.py
fccanalysis plots plots_jetE_alljets_Compare_AK.py

