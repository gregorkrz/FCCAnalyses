source /cvmfs/fcc.cern.ch/sw/latest/setup.sh
set -e

export INPUT_DIR=/fs/ddn/sdf/group/atlas/d/gregork/fastsim/jetbenchmarks/IDEA_20251105
#export FOLDER_NAME=p8_ee_ZH_vvgg_ecm240
export JET_ALGO=Durham
export HISTOGRAMS_FOLDER_NAME=Histograms_ECM240_20251105/Jet_Algorithm_Comparison

#python3 simple_histograms.py
#python3 resolution_plots.py
#fccanalysis plots plots_jetE_alljets_Compare_AK.py


export FOLDER_NAME=p8_ee_ZH_vvbb_ecm240
python3 simple_histograms.py
python3 resolution_plots.py
fccanalysis plots plots_jetE_alljets_Compare_AK.py


export FOLDER_NAME=p8_ee_ZH_6jet_HF_ecm240
python3 simple_histograms.py
python3 resolution_plots.py
fccanalysis plots plots_jetE_alljets_Compare_AK.py


export FOLDER_NAME=p8_ee_ZH_6jet_LF_ecm240
python3 simple_histograms.py
python3 resolution_plots.py
fccanalysis plots plots_jetE_alljets_Compare_AK.py

