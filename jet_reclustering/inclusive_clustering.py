# source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh  # Need to do the nightly due to some compatibility issues!
# fccanalysis run inclusive_clustering.py --nevents 10000
import os, copy # tagging

#Mandatory: List of processes

processList = {
    "p8_ee_WW_ecm365_fullhad": {'fraction': 1},
    "p8_ee_ZH_qqbb_ecm365": {'fraction': 1},
    "p8_ee_ZH_llbb_ecm365": {'fraction': 1},
}

from addons.FastJet.jetClusteringHelper import (
    ExclusiveJetClusteringHelper,
)
from jetClusteringHelper import InclusiveJetClusteringHelper

Rs = [1, 3, 5, 7, 9]
# Define the input dir (optional)
# inputDir    = "outputs/FCCee/higgs/mH-recoil/mumu/stage1"
inputDir = "../../idea_fullsim/fast_sim/outputs"

# Optional: output directory, default is local running directory
outputDir = "../../idea_fullsim/fast_sim/outputs_jet_reclustering"



class RDFanalysis():
    #__________________________________________________________
    #Mandatory: analysers function to define the analysers to process, please make sure you return the last dataframe, in this example it is df2
    def analysers(df):
        collections = {
            "GenParticles": "Particle",
            "PFParticles": "ReconstructedParticles",
            "PFTracks": "EFlowTrack",
            "PFPhotons": "EFlowPhoton",
            "PFNeutralHadrons": "EFlowNeutralHadron",
            "TrackState": "EFlowTrack_1",
            "TrackerHits": "TrackerHits",
            "CalorimeterHits": "CalorimeterHits",
            "dNdx": "EFlowTrack_2",
            "PathLength": "EFlowTrack_L",
            "Bz": "magFieldBz",
        }
        collections_noleps = copy.deepcopy(collections)

        collections_noleps["PFParticles"] = "ReconstructedParticles"
        df1 = df
        AK = {}
        AK_gen = {}
        for radius in Rs:
            AK[radius] = InclusiveJetClusteringHelper(collections_noleps["PFParticles"], radius / 10, 1, "AK" + str(radius))
            AK_gen[radius] = InclusiveJetClusteringHelper(collections_noleps["GenParticles"], radius / 10, 1, "AKgen" + str(radius), "MCParticle")
            df1 = AK[radius].define(df1)
            df1 = AK_gen[radius].define(df1)
        for radius in Rs:
            # define AKX_jetE, AKX_jetPhi, AKX_jetEta. similar for AKgenX
            df1 = df1.Define("AK" + str(radius) + "_jetE", "JetClusteringUtils::get_jet_energies(" + AK[radius].jets + ")")
            df1 = df1.Define("AK" + str(radius) + "_jetPhi", "JetClusteringUtils::get_jet_phis(" + AK[radius].jets + ")")
            df1 = df1.Define("AK" + str(radius) + "_jetEta", "JetClusteringUtils::get_jet_etas(" + AK[radius].jets + ")")
            df1 = df1.Define("AKgen" + str(radius) + "_jetE", "JetClusteringUtils::get_jet_energies(" + AK_gen[radius].jets + ")")
            df1 = df1.Define("AKgen" + str(radius) + "_jetPhi", "JetClusteringUtils::get_jet_phis(" + AK_gen[radius].jets + ")")
            df1 = df1.Define("AKgen" + str(radius) + "_jetEta", "JetClusteringUtils::get_jet_etas(" + AK_gen[radius].jets + ")")

        # Save all the jets into the jets_AKX collection
        # print all the columns in df1

        return df1
    #__________________________________________________________
    #Mandatory: output function, please make sure you return the branchlist as a python list
    def output():
        cols = ["AK3", "AK4", "AK5"]
        return cols

