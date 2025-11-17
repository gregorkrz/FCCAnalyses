# This file filters the dataframe such that the weird events where WW don't decay don't contribute to the end results
# (this was resulting in a weird delta function peak at zero for GT partons invariant mass)
# List of processes (mandatory)

import os

assert "INPUT_DIR" in os.environ # To make sure we are taking the right input dir and folder name
assert "FOLDER_NAME" in os.environ
assert "HISTOGRAMS_FOLDER_NAME" in os.environ # Default: Histograms_ECM240

inputDir = os.environ.get("INPUT_DIR")
print("Using input dir:", inputDir)
print("Using folder name:", os.environ.get("FOLDER_NAME"))
print("Using histograms folder name: ", os.environ["HISTOGRAMS_FOLDER_NAME"])

frac = 1

processList = {
    #'p8_ee_WW_ecm365_fullhad': {'fraction': 1},
    ############## SINGLE HIGGS PROCESSES ##############
    # 6 jets
    #"p8_ee_ZH_6jet_ecm240": {'fraction': frac},
    #"p8_ee_ZH_6jet_HF_ecm240": {'fraction': frac},
    #"p8_ee_ZH_6jet_LF_ecm240": {'fraction': frac},

    # 4 jets
    #"p8_ee_ZH_qqbb_ecm240": {'fraction': frac},
    "p8_ee_ZH_bbbb_ecm240": {'fraction': frac},
    #"p8_ee_ZH_bbgg_ecm240": {'fraction': frac},
    #"p8_ee_ZH_qqgg_ecm240": {'fraction': frac},

    # 2 jets
    #"p8_ee_ZH_vvgg_ecm240": {'fraction': frac},
    #"p8_ee_ZH_vvqq_ecm240": {'fraction': frac},
    #"p8_ee_ZH_vvbb_ecm240": {'fraction': frac},
}
# Link to the dictionary that contains all the cross section informations etc... (mandatory)
procDict = "FCCee_procDict_winter2023_IDEA.json"

# additional/custom C++ functions, defined in header files (optional)
includePaths = ["functions.h", "utils.h"]

outputDir = "../../idea_fullsim/fast_sim/{}/{}".format(os.environ["HISTOGRAMS_FOLDER_NAME"], os.environ.get("FOLDER_NAME"))


def build_graph(df, dataset):
    #results = []
    print("############## Doing dataset:", dataset, "##############")
    df = df.Define("weight", "1.0")
    weightsum = df.Sum("weight")
    df = df.Define("gen_muon_energies", "FCCAnalyses::ZHfunctions::get_muon_energies(Particle)")
    df = df.Define("reco_muon_energies", "FCCAnalyses::ZHfunctions::get_reco_muon_energies(ReconstructedParticles)")
    muon_energies = df.AsNumpy(["gen_muon_energies"])["gen_muon_energies"]
    print("muon energies", len(muon_energies), muon_energies[:5])
    print("reco muon energies", len(df.AsNumpy(["reco_muon_energies"])["reco_muon_energies"]), df.AsNumpy(["reco_muon_energies"])["reco_muon_energies"][:5])
    hist_muon_energies = df.Histo1D(("gen_muon_energies", "Generated muon energies;Muon energy [GeV];Events", 100, 0, 120), "gen_muon_energies")
    hist_reco_energies = df.Histo1D(("reco_muon_energies", "Reconstructed muon energies;Muon energy [GeV];Events", 100, 0, 120), "reco_muon_energies")
    return [hist_muon_energies, hist_reco_energies], weightsum
