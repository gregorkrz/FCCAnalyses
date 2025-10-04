# fccanalysis run histmaker_jetE_AK.py --nevents 10000

# list of processes (mandatory)
processList = {
    # 'p8_ee_ZZ_ecm240':{'fraction':1},
    'p8_ee_WW_ecm365_fullhad_AK': {'fraction': 1},
    #"p8_ee_ZH_qqbb_ecm365": {'fraction': 1},
    #"p8_ee_ZH_llbb_ecm365": {'fraction': 1},
    # 'wzp6_ee_mumuH_ecm240':{'fraction':1},
    #'p8_ee_WW_mumu_ecm240': {'fraction': 1, 'crossSection': 0.25792},
    #'p8_ee_ZZ_mumubb_ecm240': {'fraction': 1, 'crossSection': 2 * 1.35899 * 0.034 * 0.152},
    #'p8_ee_ZH_Zmumu_ecm240': {'fraction': 1, 'crossSection': 0.201868 * 0.034},
}

# Production tag when running over EDM4Hep centrally produced events, this points to the yaml files for getting sample statistics (mandatory)
# prodTag     = "FCCee/winter2023/IDEA/"

# Link to the dictionary that contains all the cross section informations etc... (mandatory)
procDict = "FCCee_procDict_winter2023_IDEA.json"

# additional/custom C++ functions, defined in header files (optional)
includePaths = ["../playground/functions.h"]

# Define the input dir (optional)
# inputDir    = "outputs/FCCee/higgs/mH-recoil/mumu/stage1"
inputDir = "../../idea_fullsim/fast_sim/outputs"

# Optional: output directory, default is local running directory
outputDir = "../../idea_fullsim/fast_sim/histograms"

# optional: ncpus, default is 4, -1 uses all cores available
nCPUS = -1

# scale the histograms with the cross-section and integrated luminosity
doScale = False
intLumi = 5000000  # 5 /ab

# Define some binning for various histograms
bins_count_jets = (5, 0, 5)

Rs = [2, 4, 5, 8, 10, 15, 20, 30]

# build_graph function that contains the analysis logic, cuts and histograms (mandatory)
def build_graph(df, dataset):
    #results = []
    df = df.Define("weight", "1.0")
    weightsum = df.Sum("weight")
    #df = df.Define("n_jets", "Jet.size()")
    # compute energy of hardest jet over energy of hardest genjet
    results = []
    for r in Rs:
        # list columns in df and print them
        print("Columns in df for R={}: {}".format(r, df.GetColumnNames()))

        df = df.Define(f"deltaR_matching_{r}", f"FCCAnalyses::ZHfunctions::get_reco_truth_jet_mapping(RecoJetInclusiveAK{r}, GenJetInclusiveAK{r})")
        df = df.Define(f"matching_processing_{r}", f"FCCAnalyses::ZHfunctions::get_energy_ratios_for_matched_jets(deltaR_matching_{r}, RecoJetInclusiveAK{r}, GenJetInclusiveAK{r})")
        df = df.Define(f"jet_E_reco_over_true_{r}", f"matching_processing_{r}.first")
        df = df.Define(f"E_unmatched_{r}", f"matching_processing_{r}.second")
        #df = df.Define("h_E_ratio_{}".format(r), f"FCCAnalyses::ZHfunctions::get_histo_E_ratio(jet_E_reco_over_true_{r})".format(r))
        #df = df.Define("E_of_unmatched_reco_jets_{}".format(r),
        #                  f"FCCAnalyses::ZHfunctions::get_histo_E_unmatched(E_unmatched_{r})".format(r))
        h_ratio = df.Histo1D((f"h_ratio_{r}", "E_reco/E_true;E_reco / E_true;Events", 100, 0, 2), f"jet_E_reco_over_true_{r}")
        h_unmatched = df.Histo1D((f"h_unmatched_{r}",
                                  "E of unmatched reco jets;E_reco;Events", 100, 0, 300),
                                  f"E_unmatched_{r}")
        results.append(h_ratio)
        results.append(h_unmatched)
    #h_fancy = df.Histo1D(("h_fancy", "E_reco/E_true (fancy matching);E_reco / E_true;Events", 150, 0.4, 1.2), "ratio_jet_energies_fancy")
    #h_Ejet = df.Histo1D(("h_E_all_jets", "E of jet;E_reco;Events", 100, 0, 300), "JetDurhamN4.energy")
    #h_Egenjet = df.Histo1D(("h_E_all_genjets", "E of genjet;E_gen;Events", 100, 0, 300), "GenJetDurhamN4.energy")
    # count -1s  in ratio_jet_energies_fancy
    # print size of ratio_jet_energies_fancy
    #df = df.Define("ratio_jet_energies_fancy_E0", "ratio_jet_energies_fancy[0]")
    #h_fancy1 = df.Histo1D(("h_fancy_E0", "E_reco/E_true (fancy matching);E_reco / E_true;Events", 150, 0.4, 1.2), "ratio_jet_energies_fancy_E0")
    #h_unmatched_reco_jets = df.Histo1D(("h_unmatched_reco_jets", "E of unmatched reco jets;E_reco;Events", 100, 0, 300), "E_of_unmatched_reco_jets")
    #results = [h_fancy, h_fancy1, h_unmatched_reco_jets, h_Ejet, h_Egenjet]
    #for i in range(4):
    #    df = df.Define("jet_E{}".format(i), "ratio_jet_energies[{}]".format(i))
    #    h_E = df.Histo1D(("h_E{}".format(i), "E_reco/E_true;E_reco / E_true;Events", 50, 0.8, 1.2), "jet_E{}".format(i))
    #    results.append(h_E)
    return results, weightsum

