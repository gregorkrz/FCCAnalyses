set -e

source /cvmfs/fcc.cern.ch/sw/latest/setup.sh
export INPUT_DIR=/fs/ddn/sdf/group/atlas/d/gregork/fastsim/jetbenchmarks/22102025/ISR_ecm240
export FOLDER_NAME=GenJetDurhamFastJet_ISR

fccanalysis run histmaker_jetE.py
python3 simple_histograms.py
python3 resolution_plots.py
fccanalysis plots plots_jetE_alljets.py
