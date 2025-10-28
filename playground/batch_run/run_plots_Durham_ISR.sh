source /cvmfs/fcc.cern.ch/sw/latest/setup.sh
set -e

export INPUT_DIR=/fs/ddn/sdf/group/atlas/d/gregork/fastsim/jetbenchmarks/22102025/ISR_ecm240
export FOLDER_NAME=GenJetDurhamFastJet_ISR_Durham
export JET_ALGO=Durham
export HISTOGRAMS_FOLDER_NAME=Histograms_ECM240

#fccanalysis run histmaker_jetE.py
python3 simple_histograms.py
python3 resolution_plots.py
fccanalysis plots plots_jetE_alljets.py --legend-x-min 115

