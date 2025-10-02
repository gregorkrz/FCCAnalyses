
source /cvmfs/sw.hsf.org/key4hep/setup.sh -r 2024-03-10
git clone git@github.com:gregorkrz/FCCAnalyses.git
cd FCCAnalyses
source ./setup.sh
fccanalysis build -j 8

