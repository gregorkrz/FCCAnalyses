

source /cvmfs/fcc.cern.ch/sw/latest/setup.sh
set -e

export INPUT_DIR=/fs/ddn/sdf/group/atlas/d/gregork/fastsim/jetbenchmarks/22102025/NoISR_ecm240
export FOLDER_NAME=GenJetDurhamFastJet_NoISR_AK10
export JET_ALGO=AK
export AK_RADIUS=1.0 # rather get the radius as the first argument to the script. How to?

fccanalysis run histmaker_jetE.py
python3 simple_histograms.py
python3 resolution_plots.py
fccanalysis plots plots_jetE_alljets.py
