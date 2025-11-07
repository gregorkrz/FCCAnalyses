source /cvmfs/fcc.cern.ch/sw/latest/setup.sh
set -e

export INPUT_DIR=/sdf/home/g/gregork/idea_fullsim/fast_sim/outputs
export FOLDER_NAME=PFDurham_ISR_v2_Tiny
export JET_ALGO=Durham
export HISTOGRAMS_FOLDER_NAME=Histograms_ECM240

fccanalysis run histmaker_jetE.py
python3 simple_histograms.py
python3 resolution_plots.py
fccanalysis plots plots_jetE_alljets.py
