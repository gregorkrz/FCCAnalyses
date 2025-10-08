# fccanalysis run histmaker_jetE.py --nevents 10000
# source /cvmfs/fcc.cern.ch/sw/latest/setup.sh
# list of processes (mandatory)

processList = {
    # 'p8_ee_ZZ_ecm240':{'fraction':1},
    'p8_ee_WW_ecm365_fullhad': {'fraction': 1},
    "p8_ee_ZH_qqbb_ecm365": {'fraction': 1},
    "p8_ee_ZH_llbb_ecm365": {'fraction': 1},
    "p8_ee_ZH_6jet_ecm365": {'fraction': 1},
    "p8_ee_ZH_vvbb_ecm365": {'fraction': 1},
    # 'wzp6_ee_mumuH_ecm240':{'fraction':1},
    #'p8_ee_WW_mumu_ecm240': {'fraction': 1, 'crossSection': 0.25792},
    #'p8_ee_ZZ_mumubb_ecm240': {'fraction': 1, 'crossSection': 2 * 1.35899 * 0.034 * 0.152},
    #'p8_ee_ZH_Zmumu_ecm240': {'fraction': 1, 'crossSection': 0.201868 * 0.034},
}
bins = [0, 50, 100, 150, 200]

# Production tag when running over EDM4Hep centrally produced events, this points to the yaml files for getting sample statistics (mandatory)
# prodTag     = "FCCee/winter2023/IDEA/"

# Link to the dictionary that contains all the cross section informations etc... (mandatory)
procDict = "FCCee_procDict_winter2023_IDEA.json"

# additional/custom C++ functions, defined in header files (optional)
includePaths = ["functions.h"]

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

# build_graph function that contains the analysis logic, cuts and histograms (mandatory)
def build_graph(df, dataset):
    #results = []
    df = df.Define("weight", "1.0")
    weightsum = df.Sum("weight")
    #df = df.Define("n_jets", "Jet.size()")
    # compute energy of hardest jet over energy of harderst genjet
    df = df.Define("jet_energies", "FCCAnalyses::ZHfunctions::sort_jet_energies(JetDurhamN4)")
    df = df.Define("genjet_energies", "FCCAnalyses::ZHfunctions::sort_jet_energies(GenJetDurhamN4)")
    df = df.Define("ratio_jet_energies", "FCCAnalyses::ZHfunctions::elementwise_divide(jet_energies, genjet_energies)")
    df = df.Define("fancy_matching", "FCCAnalyses::ZHfunctions::get_reco_truth_jet_mapping(JetDurhamN4, GenJetDurhamN4, 0.4)")
    df = df.Define("matching_processing", "FCCAnalyses::ZHfunctions::get_energy_ratios_for_matched_jets(fancy_matching, JetDurhamN4, GenJetDurhamN4)")
    df = df.Define("ratio_jet_energies_fancy", "std::get<0>(matching_processing)")
    df = df.Define("E_of_unmatched_reco_jets", "std::get<1>(matching_processing)")
    df = df.Define("genjet_energies_matched", "std::get<2>(matching_processing)")
    # Bin the ratio_jet_energies_fancy according to genjet_energies (bins [0, 50, 100, 150, 200])
    histograms = []
    for i in range(len(bins) - 1):
        df = df.Define("binned_E_reco_over_true_{}_{}".format(bins[i], bins[i+1]), "FCCAnalyses::ZHfunctions::filter_number_by_bin(ratio_jet_energies_fancy, genjet_energies_matched, {}, {})".format(bins[i], bins[i + 1]))
        hh = df.Histo1D(("binned_E_reco_over_true_{}_{}".format(bins[i], bins[i+1]), "Ereco/Etrue;Ereco/Etrue;Events", 150, 0.4, 1.2), "binned_E_reco_over_true_{}_{}".format(bins[i], bins[i+1]))
        histograms.append(hh)
    h_fancy = df.Histo1D(("h_fancy", "E_reco/E_true (fancy matching);E_reco / E_true;Events", 200, 0.4, 2.0), "ratio_jet_energies_fancy")
    h_Ejet = df.Histo1D(("h_E_all_jets", "E of jet;E_reco;Events", 100, 0, 300), "JetDurhamN4.energy")
    h_Egenjet = df.Histo1D(("h_E_all_genjets", "E of genjet;E_gen;Events", 100, 0, 300), "GenJetDurhamN4.energy")
    # count -1s  in ratio_jet_energies_fancy
    # print size of ratio_jet_energies_fancy
    df = df.Define("ratio_jet_energies_fancy_E0", "ratio_jet_energies_fancy[0]")
    h_fancy1 = df.Histo1D(("h_fancy_E0", "E_reco/E_true (fancy matching);E_reco / E_true;Events", 150, 0.4, 1.2), "ratio_jet_energies_fancy_E0")
    h_unmatched_reco_jets = df.Histo1D(("h_unmatched_reco_jets", "E of unmatched reco jets;E_reco;Events", 100, 0, 300), "E_of_unmatched_reco_jets")
    results = [h_fancy, h_fancy1, h_unmatched_reco_jets, h_Ejet, h_Egenjet]
    for i in range(4):
        df = df.Define("jet_E{}".format(i), "ratio_jet_energies[{}]".format(i))
        h_E = df.Histo1D(("h_E{}".format(i), "E_reco/E_true;E_reco / E_true;Events", 50, 0.8, 1.2), "jet_E{}".format(i))
        results.append(h_E)
    return results + histograms, weightsum
