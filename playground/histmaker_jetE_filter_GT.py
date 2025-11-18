# This file filters the dataframe such that the weird events where WW don't decay don't contribute to the end results
# (this was resulting in a weird delta function peak at zero for GT partons invariant mass)
# export FOLDER_NAME=GenJetDurhamFastJet_NoISR
# export INPUT_DIR=/fs/ddn/sdf/group/atlas/d/gregork/fastsim/jetbenchmarks/22102025/ISR_ecm240 fccanalysis run histmaker_jetE.py
# source /cvmfs/fcc.cern.ch/sw/latest/setup.sh

# List of processes (mandatory)

from truth_matching import get_Higgs_mass_with_truth_matching
from jet_helper import get_jet_vars
import os

assert "INPUT_DIR" in os.environ # To make sure we are taking the right input dir and folder name
assert "FOLDER_NAME" in os.environ
assert "HISTOGRAMS_FOLDER_NAME" in os.environ # Default: Histograms_ECM240

matching_radius = os.environ.get("JET_MATCHING_RADIUS", 0.3)

inputDir = os.environ.get("INPUT_DIR")
print("Using input dir:", inputDir)
print("Using folder name:", os.environ.get("FOLDER_NAME"))
print("Using histograms folder name: ", os.environ["HISTOGRAMS_FOLDER_NAME"])

frac = 1

processList = {
    #'p8_ee_WW_ecm365_fullhad': {'fraction': 1},
    ############## SINGLE HIGGS PROCESSES ##############
    # 4 jets
    "p8_ee_ZH_qqbb_ecm240": {'fraction': frac},
    "p8_ee_ZH_bbbb_ecm240": {'fraction': frac},
    "p8_ee_ZH_bbgg_ecm240": {'fraction': frac},
    "p8_ee_ZH_qqgg_ecm240": {'fraction': frac},

    # 2 jets
    "p8_ee_ZH_vvgg_ecm240": {'fraction': frac},
    "p8_ee_ZH_vvqq_ecm240": {'fraction': frac},
    "p8_ee_ZH_vvbb_ecm240": {'fraction': frac},

    # 6 jets
    "p8_ee_ZH_6jet_ecm240": {'fraction': frac},
    "p8_ee_ZH_6jet_HF_ecm240": {'fraction': frac},
    "p8_ee_ZH_6jet_LF_ecm240": {'fraction': frac},

    ##############2 jets: other detectors study #################
    #"p8_ee_ZH_vvgg_ecm240_CEPC":  {'fraction': frac},
    #"p8_ee_ZH_vvgg_ecm240_CLD":  {'fraction': frac},
    #"p8_ee_ZH_vvqq_ecm240_CEPC":  {'fraction': frac},
    #"p8_ee_ZH_vvqq_ecm240_CLD":  {'fraction': frac},
    #"p8_ee_ZH_vvbb_ecm240_CEPC":  {'fraction': frac},
    #"p8_ee_ZH_vvbb_ecm240_CLD":  {'fraction': frac},
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
    "p8_ee_ZH_vvqq_ecm240": 2,

    #"p8_ee_ZH_vvgg_ecm240_CEPC": 2,
    #"p8_ee_ZH_vvgg_ecm240_CLD": 2,
    #"p8_ee_ZH_vvqq_ecm240_CEPC": 2,
    #"p8_ee_ZH_vvqq_ecm240_CLD": 2,
    #"p8_ee_ZH_vvbb_ecm240_CEPC": 2,
    #"p8_ee_ZH_vvbb_ecm240_CLD": 2,

    "p8_ee_ZH_6jet_HF_ecm240": 6,
    "p8_ee_ZH_6jet_LF_ecm240": 6,
    "p8_ee_ZH_bbgg_ecm240": 4,
    "p8_ee_ZH_qqgg_ecm240": 4,
}

nJets_from_H_process_list = {
    "p8_ee_ZH_qqbb_ecm240": 2,
    "p8_ee_ZH_6jet_ecm240": 4,
    "p8_ee_ZH_vvbb_ecm240": 2,
    "p8_ee_ZH_bbbb_ecm240": 2,
    "p8_ee_ZH_vvgg_ecm240": 2,
    "p8_ee_ZH_vvqq_ecm240": 2,
    #"p8_ee_ZH_vvgg_ecm240_CEPC": 2,
    #"p8_ee_ZH_vvgg_ecm240_CLD": 2,
    #"p8_ee_ZH_vvqq_ecm240_CEPC": 2,
    #"p8_ee_ZH_vvqq_ecm240_CLD": 2,
    #"p8_ee_ZH_vvbb_ecm240_CEPC": 2,
    #"p8_ee_ZH_vvbb_ecm240_CLD": 2,
    #"p8_ee_ZH_vvbb_ecm240_IDEA": 2,
    "p8_ee_ZH_6jet_HF_ecm240": 4,
    "p8_ee_ZH_6jet_LF_ecm240": 4,
    "p8_ee_ZH_bbgg_ecm240": 2,
    "p8_ee_ZH_qqgg_ecm240": 2,
}


#def get_files(procname):
#    prefix = "/fs/ddn/sdf/group/atlas/d/gregork/fastsim/jetbenchmarks/"
#    files = []
#    for i in range(1, 6, 1):
#        files.append(prefix + procname + ".root")
#for proc in processList:
#    processList[proc]['files'] = get_files(proc)
#bins = [0, 25, 50, 75, 100, 125, 150, 175, 200]
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
bins_eta = [-5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 5]
bins_costheta = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]

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

outputDir = "../../idea_fullsim/fast_sim/{}/{}".format(os.environ["HISTOGRAMS_FOLDER_NAME"], os.environ.get("FOLDER_NAME"))

Fully_Matched_Only = os.environ.get("KEEP_ONLY_FULLY_MATCHED_EVENTS", "0").lower() == "1"
if Fully_Matched_Only:
    print("Keeping only fully matched events in the output histograms! Efficiency will be 1.")

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
    print("############## Doing dataset:", dataset, "##############")
    df = df.Define("MC_part_idx", "FCCAnalyses::ZHfunctions::get_MC_quark_index_for_Higgs(Particle, _Particle_daughters.index, false)")
    mcpart_idx_display = df.AsNumpy(["MC_part_idx"])["MC_part_idx"]
    print("MC part. idx", len(mcpart_idx_display), mcpart_idx_display[:5])
    print("Filtering df , current size: ", len(mcpart_idx_display))
    df = df.Filter("MC_part_idx.size() == {}".format(nJets_from_H_process_list[dataset]))
    print("After filtering, len=", len(df.AsNumpy(["MC_part_idx"])["MC_part_idx"]))
    df = df.Define("weight", "1.0")
    weightsum = df.Sum("weight")
    df = df.Define("calo_hit_energy", "CalorimeterHits.energy")
    hist_calo_hist_E = df.Histo1D(("h_calo_hit_energy", "Calo hit energy;E_calo_hit;Events", 100, 0, 3),
                                  "calo_hit_energy")
    che = df.AsNumpy(["calo_hit_energy"])["calo_hit_energy"]
    print("Calo hit energy array size:", len(che))
    #print("Calo hit energies:", che)
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
    elif os.environ.get("JET_ALGO", "durham").lower() == "eeak":
        ak_radius = float(os.environ.get("AK_RADIUS", "0.4"))
        df = get_jet_vars(df, "stable_gen_particles", AK_radius=ak_radius, name="FastJet_jets", is_ee_AK=True)
        df = get_jet_vars(df, "ReconstructedParticles", AK_radius=ak_radius, name="FastJet_jets_reco", is_ee_AK=True)
    elif os.environ.get("JET_ALGO", "durham").lower() == "calojetdurham":
        df = get_jet_vars(df, "stable_gen_particles", N_durham=nJets_processList[dataset], name="FastJet_jets")
        df = df.Define("FastJet_jets_reco", "CaloJetDurham")
    else:
        raise ValueError("Unknown JET_ALGO: {}".format(os.environ.get("JET_ALGO")))
    # For eeKT:
    #df = get_jet_vars(df, "stable_gen_particles", ee_pt_cutoff=0, name="FastJet_jets")
    #df = get_jet_vars(df, "ReconstructedParticles", ee_pt_cutoff=0, name="FastJet_jets_reco")
    first_k = nJets_processList[dataset]
    df = df.Define("GenJetFastJet", "FCCAnalyses::ZHfunctions::fastjet_to_vec_rp_jet(FastJet_jets, {})".format(first_k))
    if not os.environ.get("JET_ALGO", "durham").lower() == "calojetdurham":
        df = df.Define("RecoJetFastJet", "FCCAnalyses::ZHfunctions::fastjet_to_vec_rp_jet(FastJet_jets_reco, {})".format(first_k))
        # store the neutral and charged components of the jets
        df = df.Define("RecoJetFastJetNC", "FCCAnalyses::ZHfunctions::fastjet_to_vec_rp_jet_split_based_on_charge(FastJet_jets_reco, ReconstructedParticles, {})".format(first_k))
        df = df.Define("GenJetFastJetNC", "FCCAnalyses::ZHfunctions::fastjet_to_vec_rp_jet_split_based_on_charge(FastJet_jets, stable_gen_particles, {})".format(first_k))
    else:
        # For Durham
        df = df.Define("RecoJetFastJet", "FastJet_jets_reco")
    #print("recojet fastjet:", df.AsNumpy([RecoJetVariable])[RecoJetVariable])
    df = df.Define("jet_energies", "FCCAnalyses::ZHfunctions::sort_jet_energies({})".format(RecoJetVariable))
    df = df.Define("genjet_energies", "FCCAnalyses::ZHfunctions::sort_jet_energies({})".format(GenJetVariable))
    #df = df.Define("ratio_jet_energies", "FCCAnalyses::ZHfunctions::elementwise_divide(jet_energies, genjet_energies)")
    df = df.Define("fancy_matching", "FCCAnalyses::ZHfunctions::get_reco_truth_jet_mapping_greedy({}, {}, {}, false)".format(RecoJetVariable, GenJetVariable, matching_radius))
    df = df.Define("njets", "{}.size()".format(RecoJetVariable))
    df = df.Define("ngenjets", "{}.size()".format(GenJetVariable))
    df = df.Define("distance_between_genjets", "FCCAnalyses::ZHfunctions::get_jet_distances({})".format(GenJetVariable))
    df = df.Define("distance_between_recojets", "FCCAnalyses::ZHfunctions::get_jet_distances({})".format(RecoJetVariable))
    df = df.Define("min_distance_between_genjets", "FCCAnalyses::ZHfunctions::min(FCCAnalyses::ZHfunctions::get_jet_distances({}))".format(format(GenJetVariable)))
    df = df.Define("min_distance_between_recojets", "FCCAnalyses::ZHfunctions::min(FCCAnalyses::ZHfunctions::get_jet_distances({}))".format(RecoJetVariable))
    # Will be different for each process with e+e- kt algorithm
    hist_njets = df.Histo1D(("h_njets", "Number of reconstructed jets;N_jets;Events", 10, 0, 10), "njets")
    hist_ngenjets = df.Histo1D(("h_ngenjets", "Number of generated jets;N_genjets;Events", 10, 0, 10), "ngenjets")
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
    if not os.environ.get("JET_ALGO", "durham").lower() == "calojetdurham":
        # Use the RecoJetVariable+NC and GenJetVariable+NC for getting the charged and neutral components of the jets
        # First element of the pairs: neutral (N), second: charged (C)
        df = df.Define("matching_processing_Neutral_part", "FCCAnalyses::ZHfunctions::get_energy_ratios_for_matched_jets(fancy_matching, RecoJetFastJetNC.first, GenJetFastJetNC.first)")
        df = df.Define("ratio_jet_energies_fancy_Neutral_part", "std::get<0>(matching_processing_Neutral_part)")
        df = df.Define("matching_processing_Charged_part", "FCCAnalyses::ZHfunctions::get_energy_ratios_for_matched_jets(fancy_matching, RecoJetFastJetNC.second, GenJetFastJetNC.second)")
        df = df.Define("ratio_jet_energies_fancy_Charged_part", "std::get<0>(matching_processing_Charged_part)")
        df = df.Define("genjet_Neutral_energies_matched", "std::get<2>(matching_processing_Neutral_part)")
        df = df.Define("genjet_Charged_energies_matched", "std::get<2>(matching_processing_Charged_part)")

        # print some of ratio_jet_energies_fancy_Neutral_part, ratio_jet_energies_fancy_Neutral_part
        rjefnpc = df.AsNumpy(["ratio_jet_energies_fancy_Neutral_part"])["ratio_jet_energies_fancy_Neutral_part"]
        rjefcpc = df.AsNumpy(["ratio_jet_energies_fancy_Charged_part"])["ratio_jet_energies_fancy_Charged_part"]
        print("Neutral:", rjefnpc[:5])
        print("Charged:", rjefcpc[:5])

        df = df.Define("jet_energy_Neutral_Reco", "FCCAnalyses::ZHfunctions::sort_jet_energies(RecoJetFastJetNC.first)")
        df = df.Define("jet_energy_Charged_Reco", "FCCAnalyses::ZHfunctions::sort_jet_energies(RecoJetFastJetNC.second)")
        df = df.Define("genjet_energy_Neutral", "FCCAnalyses::ZHfunctions::sort_jet_energies(GenJetFastJetNC.first)")
        df = df.Define("genjet_energy_Charged", "FCCAnalyses::ZHfunctions::sort_jet_energies(GenJetFastJetNC.second)")
        print("Neutral reco jet energies:", df.AsNumpy(["jet_energy_Neutral_Reco"])["jet_energy_Neutral_Reco"][:5])
        print("Charged reco jet energies:", df.AsNumpy(["jet_energy_Charged_Reco"])["jet_energy_Charged_Reco"][:5])
        print("Neutral gen jet energies:", df.AsNumpy(["genjet_energy_Neutral"])["genjet_energy_Neutral"][:5])
        print("Charged gen jet energies:", df.AsNumpy(["genjet_energy_Charged"])["genjet_energy_Charged"][:5])

    df = df.Define("ratio_jet_energies_fancy", "std::get<0>(matching_processing)")
    df = df.Define("E_of_unmatched_reco_jets", "std::get<1>(matching_processing)")
    df = df.Define("num_unmatched_reco_jets", "E_of_unmatched_reco_jets.size()")
    df = df.Define("genjet_energies_matched", "std::get<2>(matching_processing)")
    df = df.Define("genjet_etas_matched", "std::get<3>(matching_processing)")
    df = df.Define("genjet_costhetas_matched", "FCCAnalyses::Utils::get_costheta_from_eta(genjet_etas_matched)")
    df = df.Define("num_matched_reco_jets", "genjet_energies_matched.size()")
    # Bin the ratio_jet_energies_fancy according to genjet_energies (bins [0, 50, 100, 150, 200])
    histograms = [hist_genjet_all_energies, hist_genjet_matched_energies, hist_dist_jets_gen, hist_dist_jets_reco, hist_min_dist_jets_reco, hist_min_dist_jets_gen]
    for i in range(len(bins) - 1):
        df = df.Define("binned_E_reco_over_true_{}_{}".format(bins[i], bins[i+1]), "FCCAnalyses::ZHfunctions::filter_number_by_bin(ratio_jet_energies_fancy, genjet_energies_matched, {}, {})".format(bins[i], bins[i + 1]))
        hh = df.Histo1D(("binned_E_reco_over_true_{}_{}".format(bins[i], bins[i+1]), "Ereco/Etrue;Ereco/Etrue;Events", 1000, 0, 2.0), "binned_E_reco_over_true_{}_{}".format(bins[i], bins[i+1]))
        histograms.append(hh)
        hh1 = df.Histo1D(
            ("higher_res_binned_E_reco_over_true_{}_{}".format(bins[i], bins[i + 1]), "Ereco/Etrue;Ereco/Etrue;Events", 5000, 0,
             2.0), "binned_E_reco_over_true_{}_{}".format(bins[i], bins[i + 1])) # For some smaller experiments
        histograms.append(hh1)
        hh1 = df.Histo1D(
            ("even_higher_res_binned_E_reco_over_true_{}_{}".format(bins[i], bins[i + 1]), "Ereco/Etrue;Ereco/Etrue;Events",
             20000, 0,
             2.0), "binned_E_reco_over_true_{}_{}".format(bins[i], bins[i + 1]))  # For some smaller experiments
        histograms.append(hh1)
        # do this for charged and neutral parts too
        if not os.environ.get("JET_ALGO", "durham").lower() == "calojetdurham":
            # Neutral
            df = df.Define("binned_E_Neutral_reco_over_true_{}_{}".format(bins[i], bins[i + 1]),
                           "FCCAnalyses::ZHfunctions::filter_number_by_bin(ratio_jet_energies_fancy_Neutral_part, genjet_Neutral_energies_matched, {}, {})".format(
                               bins[i], bins[i + 1]))
            hh = df.Histo1D(
                ("binned_E_Neutral_reco_over_true_{}_{}".format(bins[i], bins[i + 1]), "Ereco/Etrue;Ereco/Etrue;Events", 1000,
                 0, 2.0), "binned_E_Neutral_reco_over_true_{}_{}".format(bins[i], bins[i + 1]))
            histograms.append(hh)
            hh1 = df.Histo1D(
                ("higher_res_binned_E_Neutral_reco_over_true_{}_{}".format(bins[i], bins[i + 1]), "Ereco/Etrue;Ereco/Etrue;Events",
                 5000,
                 0, 2.0), "binned_E_Neutral_reco_over_true_{}_{}".format(bins[i], bins[i + 1]))
            histograms.append(hh1)
            # Charged
            df = df.Define("binned_E_Charged_reco_over_true_{}_{}".format(bins[i], bins[i + 1]),
                           "FCCAnalyses::ZHfunctions::filter_number_by_bin(ratio_jet_energies_fancy_Charged_part, genjet_Charged_energies_matched, {}, {})".format(
                               bins[i], bins[i + 1]))
            hh = df.Histo1D(
                ("binned_E_Charged_reco_over_true_{}_{}".format(bins[i], bins[i + 1]), "Ereco/Etrue;Ereco/Etrue;Events",
                 1000,
                 0, 2.0), "binned_E_Charged_reco_over_true_{}_{}".format(bins[i], bins[i + 1]))
            histograms.append(hh)
            hh1 = df.Histo1D(
                ("higher_res_binned_E_Charged_reco_over_true_{}_{}".format(bins[i], bins[i + 1]),
                 "Ereco/Etrue;Ereco/Etrue;Events", 5000, 0, 2.0), "binned_E_Charged_reco_over_true_{}_{}".format(bins[i], bins[i + 1]))
            histograms.append(hh1)
    df = df.Define("jet_etas", "FCCAnalyses::ZHfunctions::get_jet_eta({})".format(RecoJetVariable))
    df = df.Define("genjet_etas", "FCCAnalyses::ZHfunctions::get_jet_eta({})".format(GenJetVariable))
    h_eta = df.Histo1D(("h_eta", "eta of reco jets;eta;Events", 100, -5, 5), "jet_etas")
    h_eta_gen = df.Histo1D(("h_eta_gen", "eta of gen jets;eta;Events", 100, -5, 5), "genjet_etas")
    # For each energy and eta bin, count the number of unmatched reco jets over the number of gen jets in that bin, and save these variables in the output file too
    for i in range(len(bins_eta) - 1):
        df = df.Define("binned_E_reco_over_true_eta_{}_{}".format(neg_format(bins_eta[i]), neg_format(bins_eta[i+1])), "FCCAnalyses::ZHfunctions::filter_number_by_bin(ratio_jet_energies_fancy, genjet_etas_matched, {}, {})".format(bins_eta[i], bins_eta[i + 1]))
        hh = df.Histo1D(("binned_E_reco_over_true_eta_{}_{}".format(neg_format(bins_eta[i]), neg_format(bins_eta[i+1])), "Ereco/Etrue;Ereco/Etrue;Events",  1000, 0, 2.0), "binned_E_reco_over_true_eta_{}_{}".format(neg_format(bins_eta[i]), neg_format(bins_eta[i+1])))
        histograms.append(hh)
    for i in range(len(bins_costheta) - 1):
        df = df.Define("binned_E_reco_over_true_costheta_{}_{}".format(neg_format(bins_costheta[i]), neg_format(bins_costheta[i+1])), "FCCAnalyses::ZHfunctions::filter_number_by_bin(ratio_jet_energies_fancy, genjet_costhetas_matched, {}, {})".format(bins_costheta[i], bins_costheta[i + 1]))
        hh = df.Histo1D(("binned_E_reco_over_true_costheta_{}_{}".format(neg_format(bins_costheta[i]), neg_format(bins_costheta[i+1])), "Ereco/Etrue;Ereco/Etrue;Events",  1000, 0, 2.0), "binned_E_reco_over_true_costheta_{}_{}".format(neg_format(bins_costheta[i]), neg_format(bins_costheta[i+1])))
        histograms.append(hh)
    h_fancy = df.Histo1D(("h_fancy", "E_reco/E_true (fancy matching);E_reco / E_true;Events", 300, 0.5, 1.5), "ratio_jet_energies_fancy")
    df = df.Define("ratio_jet_energies_fancy_higheta", "FCCAnalyses::ZHfunctions::cut_by_quantity(ratio_jet_energies_fancy, genjet_etas_matched, -999999, -0.9)")
    h_fancy_higheta = df.Histo1D(("h_fancy_higheta", "E_reco/E_true (fancy matching, |eta|>2);E_reco / E_true;Events", 300, 0.5, 1.5), "ratio_jet_energies_fancy_higheta")
    # Make a histogram of jet energies
    h_Ejet = df.Histo1D(("h_E_all_reco_jets", "E of reco jet;E_reco;Events", 100, 0, 300), "jet_energies")
    h_Egenjet = df.Histo1D(("h_E_all_gen_jets", "E of gen jet;E_gen;Events", 100, 0, 300), "genjet_energies")
    df = df.Define("ratio_jet_energies_fancy_E0", "ratio_jet_energies_fancy[0]")
    h_fancy1 = df.Histo1D(("h_fancy_E0", "E_reco/E_true (fancy matching);E_reco / E_true;Events", 150, 0.4, 1.2), "ratio_jet_energies_fancy_E0")
    h_unmatched_reco_jets = df.Histo1D(("h_unmatched_reco_jets", "E of unmatched reco jets;E_reco;Events", 100, 0, 300), "E_of_unmatched_reco_jets")
    results = [h_fancy, h_fancy1, h_unmatched_reco_jets, hist_njets, hist_ngenjets, h_fancy_higheta] #+ h_mass
    df = get_Higgs_mass_with_truth_matching(df, genjets_field=GenJetVariable, recojets_field=RecoJetVariable,
                                            define_mc_quark_idx=False, expected_num_jets=nJets_from_H_process_list[dataset],
                                            matching_radius=matching_radius)
    #print("MC part idx", df.AsNumpy(["MC_part_idx"])["MC_part_idx"][:5])
    ##################################################################################################################
    df = df.Define("matching_reco_with_partons", "FCCAnalyses::ZHfunctions::get_reco_truth_jet_mapping_greedy({}, {}, {}, false)".format(RecoJetVariable, "MC_part_asjets", matching_radius))
    print("Matching reco with partons:", df.AsNumpy(["matching_reco_with_partons"])["matching_reco_with_partons"][:5])
    df = df.Define("matching_proc_with_partons",
                   "FCCAnalyses::ZHfunctions::get_energy_ratios_for_matched_jets(matching_reco_with_partons, {}, {})".format(RecoJetVariable, "MC_part_asjets"))
    df = df.Define("ratio_jet_energies_matching_with_partons", "std::get<0>(matching_proc_with_partons)")
    df = df.Define("E_of_unmatched_reco_jets_with_partons", "std::get<1>(matching_proc_with_partons)")
    print("ratio_jet_energies_matching_with_partons:", df.AsNumpy(["ratio_jet_energies_matching_with_partons"])["ratio_jet_energies_matching_with_partons"][:5])
    h_ratio_matching_with_partons = df.Histo1D(("h_ratio_matching_with_partons", "E_reco/E_parton;E_reco / E_parton;Events", 300, 0.5, 1.5), "ratio_jet_energies_matching_with_partons")
    df = df.Define("inv_mass_all_gen_particles", "FCCAnalyses::ZHfunctions::invariant_mass(stable_gen_particles);")
    h_mH_all_stable_part = df.Histo1D(("h_mH_all_stable_part", "Invariant mass of all particles; Minv; Events", 100, 0, 250), "inv_mass_all_gen_particles")
    h_mH_reco = df.Histo1D(("h_mH_reco", "Higgs mass from reco jets;M_H (reco jets);Events", 500, 0, 250), "inv_mass_reco")
    h_mH_gen = df.Histo1D(("h_mH_gen", "Higgs mass from gen jets;M_H (gen jets);Events", 500, 0, 250), "inv_mass_gen")
    h_mH_reco_core = df.Histo1D(("h_mH_reco_core", "Higgs mass from reco jets;M_H (reco jets);Events", 300, 75, 150), "inv_mass_reco")
    h_mH_gen_core = df.Histo1D(("h_mH_gen_core", "Higgs mass from gen jets;M_H (gen jets);Events", 300, 75, 150), "inv_mass_gen")
    h_mH_gen_all = df.Histo1D(("h_mH_gen_all", "Higgs mass from all gen jets;M_H (all gen jets);Events", 500, 0, 250), "inv_mass_gen_all")
    h_mH_reco_all = df.Histo1D(("h_mH_reco_all", "Higgs mass from all reco jets;M_H (all reco jets);Events", 500, 0, 250), "inv_mass_reco_all")
    results = results + [h_mH_reco, h_mH_gen, h_mH_gen_all, h_mH_reco_all, h_mH_all_stable_part, h_Ejet, h_Egenjet,
                         hist_calo_hist_E, h_mH_reco_core, h_mH_gen_core]
    #### More mass histograms: for inv_mass_stable_gt_particles_from_higgs, inv_mass_reco_particles_matched_from_higgs, inv_mass_MC_part
    h_mH_stable_gt_particles = df.Histo1D(("h_mH_stable_gt_particles", "Higgs mass from stable gt particles;M_H (stable gt particles);Events", 500, 0, 250), "inv_mass_stable_gt_particles_from_higgs")
    h_mH_reco_particles_matched = df.Histo1D(("h_mH_reco_particles_matched", "Higgs mass from reco particles matched;M_H (reco particles matched);Events", 500, 0, 250), "inv_mass_reco_particles_matched_from_higgs")
    h_mH_MC_part = df.Histo1D(("h_mH_MC_part", "Higgs mass from initial MC part.;M_H (MC part);Events", 500, 0, 250), "inv_mass_MC_part")
    results = results + [h_mH_stable_gt_particles, h_mH_reco_particles_matched, h_mH_MC_part]
    return results + histograms + [h_eta, h_eta_gen, h_ratio_matching_with_partons], weightsum
