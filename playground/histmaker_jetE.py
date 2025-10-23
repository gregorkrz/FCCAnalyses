# export FOLDER_NAME=GenJetDurhamFastJet_NoISR
# export INPUT_DIR=/fs/ddn/sdf/group/atlas/d/gregork/fastsim/jetbenchmarks/22102025/ISR_ecm240 fccanalysis run histmaker_jetE.py
# source /cvmfs/fcc.cern.ch/sw/latest/setup.sh

# list of processes (mandatory)
from truth_matching import get_Higgs_mass_with_truth_matching
from jet_helper import get_jet_vars
import os

assert "INPUT_DIR" in os.environ # To make sure we are taking the right input dir and folder name
assert "FOLDER_NAME" in os.environ


inputDir = os.environ.get("INPUT_DIR")
print("Using input dir:", inputDir)
print("Using folder name:", os.environ.get("FOLDER_NAME"))

frac = 1

processList = {
    #'p8_ee_WW_ecm365_fullhad': {'fraction': 1},
    ############## SINGLE HIGGS PROCESSES ######################
    "p8_ee_ZH_qqbb_ecm240": {'fraction': frac},
    "p8_ee_ZH_6jet_ecm240": {'fraction': frac},
    "p8_ee_ZH_vvbb_ecm240": {'fraction': frac},
    "p8_ee_ZH_bbbb_ecm240": {'fraction': frac},
    "p8_ee_ZH_vvgg_ecm240": {'fraction': frac},
    #############################################################
    # 'wzp6_ee_mumuH_ecm240':{'fraction':1},
    #'p8_ee_WW_mumu_ecm240': {'fraction': 1, 'crossSection': 0.25792},
    #'p8_ee_ZZ_mumubb_ecm240': {'fraction': 1, 'crossSection': 2 * 1.35899 * 0.034 * 0.152},
    #'p8_ee_ZH_Zmumu_ecm240': {'fraction': 1, 'crossSection': 0.201868 * 0.034},
}

nJets_processList = {
    "p8_ee_ZH_qqbb_ecm240": 4,
    "p8_ee_ZH_6jet_ecm240": 6,
    "p8_ee_ZH_vvbb_ecm240": 2,
    "p8_ee_ZH_bbbb_ecm240": 4,
    "p8_ee_ZH_vvgg_ecm240": 2,
}

#def get_files(procname):
#    prefix = "/fs/ddn/sdf/group/atlas/d/gregork/fastsim/jetbenchmarks/"
#    files = []
#    for i in range(1, 6, 1):
#        files.append(prefix + procname + ".root")
#for proc in processList:
#    processList[proc]['files'] = get_files(proc)

bins = [0, 50, 75, 100, 125,  150, 175, 200]
bins_eta = [-5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 5]

# Production tag when running over EDM4Hep centrally produced events, this points to the yaml files for getting sample statistics (mandatory)
# prodTag     = "FCCee/winter2023/IDEA/"

# Link to the dictionary that contains all the cross section informations etc... (mandatory)
procDict = "FCCee_procDict_winter2023_IDEA.json"

# additional/custom C++ functions, defined in header files (optional)
includePaths = ["functions.h", "utils.h"]

# Define the input dir (optional)
# inputDir    = "outputs/FCCee/higgs/mH-recoil/mumu/stage1"
#inputDir = "../../idea_fullsim/fast_sim/outputs"

# Optional: output directory, default is local running directory

########### Different output dirs ######################

outputDir = "../../idea_fullsim/fast_sim/Histograms_ECM240/{}".format(os.environ.get("FOLDER_NAME"))

#outputDir = "../../idea_fullsim/fast_sim/histograms_view/GenJetEEKtFastJet"
#######################################################

GenJetVariable = "GenJetFastJet" # set to GenJetDurhamN4 to use the delphes genjets (they are a bit funny)
RecoJetVariable = "RecoJetFastJet"

#outputDir = "../../idea_fullsim/fast_sim/histograms"

# optional: ncpus, default is 4, -1 uses all cores available
nCPUS = -1

# Scale the histograms with the cross-section and integrated luminosity
doScale = False
intLumi = 5000000  # 5 /ab

# Define some binning for various histograms
bins_count_jets = (5, 0, 5)

def point_format(number):
    return str(number).replace(".", "p")
def neg_format(number):
    # put n5 for -5
    if number < 0:
        return point_format("n{}".format(abs(number)))
    else:
        return point_format(number)

# build_graph function that contains the analysis logic, cuts and histograms (mandatory)

def build_graph(df, dataset):
    #results = []
    df = df.Define("weight", "1.0")
    weightsum = df.Sum("weight")
    #df = df.Define("n_jets", "Jet.size()")
    # Compute energy of hardest jet over energy of hardest genjet
    df = df.Define("stable_gen_particles", "FCCAnalyses::ZHfunctions::stable_particles(Particle, true)")

    # For Durham:
    if os.environ.get("JET_ALGO", "durham").lower() == "durham":
        df = get_jet_vars(df, "stable_gen_particles", N_durham=nJets_processList[dataset], name="FastJet_jets")
        df = get_jet_vars(df, "ReconstructedParticles", N_durham=nJets_processList[dataset], name="FastJet_jets_reco")
    elif os.environ.get("JET_ALGO", "durham").lower() == "ak":
        ak_radius = float(os.environ.get("AK_RADIUS", "0.4"))
        df = get_jet_vars(df, "stable_gen_particles", AK_radius=ak_radius, name="FastJet_jets")
        df = get_jet_vars(df, "ReconstructedParticles", AK_radius=ak_radius, name="FastJet_jets_reco")
    else:
        raise ValueError("Unknown JET_ALGO: {}".format(os.environ.get("JET_ALGO")))
    # For eeKT:
    #df = get_jet_vars(df, "stable_gen_particles", ee_pt_cutoff=0, name="FastJet_jets")
    #df = get_jet_vars(df, "ReconstructedParticles", ee_pt_cutoff=0, name="FastJet_jets_reco")
    first_k = nJets_processList[dataset]
    df = df.Define("GenJetFastJet", "FCCAnalyses::ZHfunctions::fastjet_to_vec_rp_jet(FastJet_jets, {})".format(first_k))
    df = df.Define("RecoJetFastJet", "FCCAnalyses::ZHfunctions::fastjet_to_vec_rp_jet(FastJet_jets_reco, {})".format(first_k))
    #print("recojet fastjet:", df.AsNumpy([RecoJetVariable])[RecoJetVariable])
    df = df.Define("jet_energies", "FCCAnalyses::ZHfunctions::sort_jet_energies({})".format(RecoJetVariable))
    df = df.Define("genjet_energies", "FCCAnalyses::ZHfunctions::sort_jet_energies({})".format(GenJetVariable))
    #df = df.Define("ratio_jet_energies", "FCCAnalyses::ZHfunctions::elementwise_divide(jet_energies, genjet_energies)")
    df = df.Define("fancy_matching", "FCCAnalyses::ZHfunctions::get_reco_truth_jet_mapping_greedy({}, {}, 1.0, false)".format(RecoJetVariable, GenJetVariable))
    df = df.Define("njets", "{}.size()".format(RecoJetVariable))
    df = df.Define("ngenjets", "{}.size()".format(GenJetVariable))

    # Will be different for each process with e+e- kt algorithm
    hist_njets = df.Histo1D(("h_njets", "Number of reconstructed jets;N_jets;Events", 10, 0, 10), "njets")
    hist_ngenjets = df.Histo1D(("h_ngenjets", "Number of generated jets;N_genjets;Events", 10, 0, 10), "ngenjets")

    df = df.Define("distance_between_genjets", "FCCAnalyses::ZHfunctions::get_jet_distances({})".format(GenJetVariable))
    df = df.Define("distance_between_recojets", "FCCAnalyses::ZHfunctions::get_jet_distances({})".format(RecoJetVariable))
    df = df.Define("min_distance_between_genjets", "FCCAnalyses::ZHfunctions::min(FCCAnalyses::ZHfunctions::get_jet_distances({}))".format(format(GenJetVariable)))
    df = df.Define("min_distance_between_recojets", "FCCAnalyses::ZHfunctions::min(FCCAnalyses::ZHfunctions::get_jet_distances({}))".format(RecoJetVariable))
    hist_dist_jets_gen = df.Histo1D(("h_dist_jets_gen", "Distance between gen jets;#DeltaR(jet_i, jet_j);Events", 100, 0, 5), "distance_between_genjets")
    hist_dist_jets_reco = df.Histo1D(("h_dist_jets_reco", "Distance between reco jets;#DeltaR(jet_i, jet_j);Events", 100, 0, 5), "distance_between_recojets")
    hist_min_dist_jets_gen = df.Histo1D(("h_min_dist_jets_gen", "Min distance between gen jets;min #DeltaR(jet_i, jet_j);Events", 100, 0, 5), "min_distance_between_genjets")
    hist_min_dist_jets_reco = df.Histo1D(("h_min_dist_jets_reco", "Min distance between reco jets;min #DeltaR(jet_i, jet_j);Events", 100, 0, 5), "min_distance_between_recojets")
    df = df.Define("matched_genjet_E_and_all_genjet_E", "FCCAnalyses::ZHfunctions::matched_genjet_E_and_all_genjet_E(fancy_matching, {})".format(GenJetVariable))
    df = df.Define("matched_genjet_energies", "std::get<0>(matched_genjet_E_and_all_genjet_E)")
    df = df.Define("all_genjet_energies", "std::get<1>(matched_genjet_E_and_all_genjet_E)")
    hist_genjet_all_energies = df.Histo1D(("h_genjet_all_energies", "E of all gen jets;E_gen;Events", 10, 0, 200), "all_genjet_energies")
    hist_genjet_matched_energies = df.Histo1D(("h_genjet_matched_energies", "E of matched gen jets;E_gen;Events", 10, 0, 200), "matched_genjet_energies")
    df = df.Define("matching_processing", "FCCAnalyses::ZHfunctions::get_energy_ratios_for_matched_jets(fancy_matching, {}, {})".format(RecoJetVariable, GenJetVariable))
    df = df.Define("ratio_jet_energies_fancy", "std::get<0>(matching_processing)")
    df = df.Define("E_of_unmatched_reco_jets", "std::get<1>(matching_processing)")
    df = df.Define("num_unmatched_reco_jets", "E_of_unmatched_reco_jets.size()")
    df = df.Define("genjet_energies_matched", "std::get<2>(matching_processing)")
    df = df.Define("genjet_etas_matched", "std::get<3>(matching_processing)")
    df = df.Define("num_matched_reco_jets", "genjet_energies_matched.size()")
    # Bin the ratio_jet_energies_fancy according to genjet_energies (bins [0, 50, 100, 150, 200])
    histograms = [hist_genjet_all_energies, hist_genjet_matched_energies, hist_dist_jets_gen, hist_dist_jets_reco, hist_min_dist_jets_reco, hist_min_dist_jets_gen]
    for i in range(len(bins) - 1):
        df = df.Define("binned_E_reco_over_true_{}_{}".format(bins[i], bins[i+1]), "FCCAnalyses::ZHfunctions::filter_number_by_bin(ratio_jet_energies_fancy, genjet_energies_matched, {}, {})".format(bins[i], bins[i + 1]))
        hh = df.Histo1D(("binned_E_reco_over_true_{}_{}".format(bins[i], bins[i+1]), "Ereco/Etrue;Ereco/Etrue;Events", 300, 0.8, 1.2), "binned_E_reco_over_true_{}_{}".format(bins[i], bins[i+1]))
        histograms.append(hh)
    df = df.Define("jet_etas", "FCCAnalyses::ZHfunctions::get_jet_eta({})".format(RecoJetVariable))
    df = df.Define("genjet_etas", "FCCAnalyses::ZHfunctions::get_jet_eta({})".format(GenJetVariable))
    h_eta = df.Histo1D(("h_eta", "eta of reco jets;eta;Events", 100, -5, 5), "jet_etas")
    h_eta_gen = df.Histo1D(("h_eta_gen", "eta of gen jets;eta;Events", 100, -5, 5), "genjet_etas")
    # For each energy and eta bin, count the number of unmatched reco jets over the number of gen jets in that bin, and save these variables in the output file too
    for i in range(len(bins_eta) - 1):
        df = df.Define("binned_E_reco_over_true_eta_{}_{}".format(neg_format(bins_eta[i]), neg_format(bins_eta[i+1])), "FCCAnalyses::ZHfunctions::filter_number_by_bin(ratio_jet_energies_fancy, genjet_etas_matched, {}, {})".format(bins_eta[i], bins_eta[i + 1]))
        hh = df.Histo1D(("binned_E_reco_over_true_eta_{}_{}".format(neg_format(bins_eta[i]), neg_format(bins_eta[i+1])), "Ereco/Etrue;Ereco/Etrue;Events", 300, 0.8, 1.2), "binned_E_reco_over_true_eta_{}_{}".format(neg_format(bins_eta[i]), neg_format(bins_eta[i+1])))
        histograms.append(hh)
    h_fancy = df.Histo1D(("h_fancy", "E_reco/E_true (fancy matching);E_reco / E_true;Events", 300, 0.5, 1.5), "ratio_jet_energies_fancy")
    # make a histogram of jet energies

    h_Ejet = df.Histo1D(("h_E_all_reco_jets", "E of reco jet;E_reco;Events", 100, 0, 300), "jet_energies")
    h_Egenjet = df.Histo1D(("h_E_all_gen_jets", "E of gen jet;E_gen;Events", 100, 0, 300), "genjet_energies")

    #h_Ejet = df.Histo1D(("h_E_all_jets", "E of jet;E_reco;Events", 100, 0, 300), "JetDurhamN4.energy")
    #h_Egenjet = df.Histo1D(("h_E_all_genjets", "E of genjet;E_gen;Events", 100, 0, 300), "genjet_ene".format(GenJetVariable))
    ### Invariant mass plots ###
    #df = df.Define("invariant_mass_genjets", "FCCAnalyses::ZHfunctions::invariant_mass(GenJetDurhamN4)")
    #df = df.Define("invariant_mass_recojets", "FCCAnalyses::ZHfunctions::invariant_mass(JetDurhamN4)")
    # Should be the same as the mass from particles in exclusive jets case, just a double-check
    #df = df.Define("invariant_mass_gen_particles", "FCCAnalyses::ZHfunctions::invariant_mass(FCCAnalyses::ZHfunctions::stable_particles(Particle))")
    #df = df.Define("invariant_mass_reco_particles", "FCCAnalyses::ZHfunctions::invariant_mass(ReconstructedParticles)")
    #h_mass = [
    #    df.Histo1D(("h_invariant_mass_genjets", "Invariant mass of gen jets;M_genjets;Events", 100, 0, 250), "invariant_mass_genjets"),
    #    df.Histo1D(("h_invariant_mass_recojets", "Invariant mass of reco jets;M_recojets;Events", 100, 0, 250), "invariant_mass_recojets"),
    #    df.Histo1D(("h_invariant_mass_gen_particles", "Invariant mass of gen particles;M_genparticles;Events", 100, 0, 250), "invariant_mass_gen_particles"),
    #    df.Histo1D(("h_invariant_mass_reco_particles", "Invariant mass of reco particles;M_recoparticles;Events", 100, 0, 250), "invariant_mass_reco_particles")
    #]
    # count -1s  in ratio_jet_energies_fancy
    # print size of ratio_jet_energies_fancy
    df = df.Define("ratio_jet_energies_fancy_E0", "ratio_jet_energies_fancy[0]")
    h_fancy1 = df.Histo1D(("h_fancy_E0", "E_reco/E_true (fancy matching);E_reco / E_true;Events", 150, 0.4, 1.2), "ratio_jet_energies_fancy_E0")
    h_unmatched_reco_jets = df.Histo1D(("h_unmatched_reco_jets", "E of unmatched reco jets;E_reco;Events", 100, 0, 300), "E_of_unmatched_reco_jets")
    results = [h_fancy, h_fancy1, h_unmatched_reco_jets, hist_njets, hist_ngenjets] #+ h_mass
    #for i in range(4):
    #    #df = df.Define("jet_E{}".format(i), "ratio_jet_energies[{}]".format(i))
    #    #h_E = df.Histo1D(("h_E{}".format(i), "E_reco/E_true;E_reco / E_true;Events", 50, 0.8, 1.2), "jet_E{}".format(i))
    #    #results.append(h_E)
    df = get_Higgs_mass_with_truth_matching(df, genjets_field=GenJetVariable, recojets_field=RecoJetVariable)

    df = df.Define("inv_mass_all_gen_particles", "FCCAnalyses::ZHfunctions::invariant_mass(stable_gen_particles);")
    h_mH_all_stable_part = df.Histo1D(("h_mH_all_stable_part", "Invariant mass of all particles; Minv; Events", 100, 0, 250), "inv_mass_all_gen_particles")
    h_mH_reco = df.Histo1D(("h_mH_reco", "Higgs mass from reco jets;M_H (reco jets);Events", 100, 0, 250), "inv_mass_reco")
    h_mH_gen = df.Histo1D(("h_mH_gen", "Higgs mass from gen jets;M_H (gen jets);Events", 100, 0, 250), "inv_mass_gen")
    h_mH_gen_all = df.Histo1D(("h_mH_gen_all", "Higgs mass from all gen jets;M_H (all gen jets);Events", 100, 0, 250), "inv_mass_gen_all")
    h_mH_reco_all = df.Histo1D(("h_mH_reco_all", "Higgs mass from all reco jets;M_H (all reco jets);Events", 100, 0, 250), "inv_mass_reco_all")
    results = results + [h_mH_reco, h_mH_gen, h_mH_gen_all, h_mH_reco_all, h_mH_all_stable_part, h_Ejet, h_Egenjet]
    return results + histograms + [h_eta, h_eta_gen], weightsum

