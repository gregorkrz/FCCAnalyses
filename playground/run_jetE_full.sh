source /cvmfs/fcc.cern.ch/sw/latest/setup.sh
fccanalysis run histmaker_jetE.py
python3 simple_histograms.py
fccanalysis plots plots_jetE_alljets.py
