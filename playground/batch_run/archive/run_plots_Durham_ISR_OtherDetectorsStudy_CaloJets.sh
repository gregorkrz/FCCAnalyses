source /cvmfs/fcc.cern.ch/sw/latest/setup.sh
set -e

export INPUT_DIR=/fs/ddn/sdf/group/atlas/d/gregork/fastsim/jetbenchmarks/other_detectors_20251103
export FOLDER_NAME=CaloJetsDurham_ISR_OtherDetectors
export JET_ALGO=CaloJetDurham
export HISTOGRAMS_FOLDER_NAME=Histograms_ECM240_OtherDetectors

#fccanalysis run histmaker_jetE.py
#python3 simple_histograms.py
python3 resolution_plots.py
#fccanalysis plots plots_jetE_alljets.py
#fccanalysis plots --legend-x-min 0.30 plots_jetE_alljets_Compare_AK.py
