source /cvmfs/fcc.cern.ch/sw/latest/setup.sh
set -e

export INPUT_DIR=/fs/ddn/sdf/group/atlas/d/gregork/fastsim/jetbenchmarks/20251028
export FOLDER_NAME=PFDurham_AK14_ISR
export JET_ALGO=AK
export AK_RADIUS=1.4 # Rather get the radius as the first argument to the script. How to?
export HISTOGRAMS_FOLDER_NAME=Histograms_ECM240

#fccanalysis run histmaker_jetE.py
python3 simple_histograms.py
python3 resolution_plots.py
fccanalysis plots plots_jetE_alljets.py


