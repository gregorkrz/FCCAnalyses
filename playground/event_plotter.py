# fccanalysis run event_plotter.py

# source /cvmfs/fcc.cern.ch/sw/latest/setup.sh
# source /cvmfs/sw.hsf.org/key4hep/setup.sh
# source  /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh

import pickle
import os
from event_displays import Vec_RP, Event
from truth_matching import get_Higgs_mass_with_truth_matching
from jet_helper import get_jet_vars, get_jet_vars_from_genjet_matching
#inputDir = "/fs/ddn/sdf/group/atlas/d/gregork/fastsim/jetbenchmarks/"

### TEMPORARILY ###
#inputDir = "/fs/ddn/sdf/group/atlas/d/gregork/fastsim/jetbenchmarks/20251028_only1root"

'''
PLOT IDX options:

1: the slice of the E_reco/E_true around .75 and E_genjet > 75 GeV
2: events with unmatched reco-to-gen jets
3: events with invariant mass of the quarks < 100 GeV (What's happening?))
4: events with invariant mass 123-127 GeV (Higgs mass window)
5: events with invariant mass < 10 GeV
6: all events
7: events with mH_reco around 90-95 GeV
'''


if False:
    # Check the weird 2nd small peak at around 95 GeV in Higgs mass reco distribution in vvqq (light-flavour) samples
    PLOT_IDX = 7
    inputDir = "/fs/ddn/sdf/group/atlas/d/gregork/fastsim/jetbenchmarks/Tiny_IDEA_20251105/"
    outputDir = "../../idea_fullsim/fast_sim/Histograms_20251112_Debug/EventDisplays_Durham_" + str(PLOT_IDX)
    processList = {
        "p8_ee_ZH_vvqq_ecm240": {'fraction': 1},
    }

if True:
    # Investigate issues with the differences between the 'ideal' jet matching and the actual reco clustering
    PLOT_IDX = 1
    inputDir = "/fs/ddn/sdf/group/atlas/d/gregork/fastsim/jetbenchmarks/Tiny_IDEA_20251105/"
    os.environ["MATCH_RECO_JETS"] = "1"
    outputDir = "../../idea_fullsim/fast_sim/Hist19_Fix0112/MatchRecoJets_EventDisplays_Durham_" + str(PLOT_IDX)
    processList = {
        #"p8_ee_ZH_6jet_LF_ecm240": {'fraction': 1},
        "p8_ee_ZH_vvqq_ecm240": {'fraction': 1},
        #"p8_ee_ZH_bbbb_ecm240": {'fraction': 1},
    }

#gf = "GenJetDurhamN4"
#rf = "JetDurhamN4"
gf = "GenJetFastJet"
rf = "RecoJetFastJet"

nJets_processList = {
    "p8_ee_ZH_qqbb_ecm240": 4,
    "p8_ee_ZH_6jet_ecm240": 6,
    "p8_ee_ZH_vvbb_ecm240": 2,
    "p8_ee_ZH_bbbb_ecm240": 4,
    "p8_ee_ZH_vvgg_ecm240": 2,
    "p8_ee_ZH_vvqq_ecm240": 2,
    "p8_ee_ZH_6jet_LF_ecm240": 6
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

# Production tag when running over EDM4Hep centrally produced events, this points to the yaml files for getting sample statistics (mandatory)
# prodTag     = "FCCee/winter2023/IDEA/"

# Link to the dictionary that contains all the cross section informations etc... (mandatory)
procDict = "FCCee_procDict_winter2023_IDEA.json"

global_event_idx = {}
# additional/custom C++ functions, defined in header files (optional)
includePaths = ["functions.h", "utils.h"]

# Define the input dir (optional)
# inputDir    = "outputs/FCCee/higgs/mH-recoil/mumu/stage1"
#inputDir = "../../idea_fullsim/fast_sim/outputs"


def plot_filter(E_reco_over_true, n_unmatched, inv_mass_Higgs, E_genjet, idx=1):
    if idx == 1:
        return True
    print("E genjet", E_genjet)
    if idx == 6:
        return True
    if idx == 5:
        if inv_mass_Higgs < 10.0:
            return True
        else:
            return False
    if idx == 3:
        if inv_mass_Higgs < 100.0:
            return True
        else:
            return False
    if idx == 4:
        if (inv_mass_Higgs > 123.0) and (inv_mass_Higgs < 127.0):
            return True
        else:
            return False
    if idx == 2 and n_unmatched > 0:
        return True
    elif idx == 1:
        for i, E in enumerate(E_reco_over_true):
            if (E > 0.73) and (E < 0.77) and (E_genjet[i] > 75.0):
                return True
    elif idx == 7:
        if (inv_mass_Higgs > 90.0) and (inv_mass_Higgs < 95.0):
            return True
        else:
            return False
    return False


def build_graph(df, dataset):
    global global_event_idx # Is this thread-safe??
    df = df.Define("weight", "1.0")
    weightsum = df.Sum("weight")
    df = df.Define("MC_part_idx",
                   "FCCAnalyses::ZHfunctions::get_MC_quark_index_for_Higgs(Particle, _Particle_daughters.index, false)")
    mcpart_idx_display = df.AsNumpy(["MC_part_idx"])["MC_part_idx"]
    df = df.Define("_RPEtaFilter", "FCCAnalyses::ZHfunctions::filter_reco_particles(ReconstructedParticles)")
    df = df.Define("ReconstructedParticlesEtaFilter", "_RPEtaFilter.first")
    df = df.Define("ReconstructedParticlesToEtaFilterRPIndex", "_RPEtaFilter.second")
    print("MC part. idx", len(mcpart_idx_display), mcpart_idx_display[:5])
    print("Filtering df , current size: ", len(mcpart_idx_display))
    df = df.Filter("MC_part_idx.size() == {}".format(nJets_from_H_process_list[dataset]))
    df = df.Define("MC_quark_index", "FCCAnalyses::ZHfunctions::get_MC_quark_index_for_Higgs(Particle, _Particle_daughters.index, false);")
    df = df.Define("reco_mc_links",
                   "FCCAnalyses::ZHfunctions::getRP2MC_index(_RecoMCLink_from.index, _RecoMCLink_to.index, ReconstructedParticlesEtaFilter, Particle, ReconstructedParticlesToEtaFilterRPIndex)")
    df = df.Define("mc2rp", "reco_mc_links.second")
    df = df.Define("GT_jets", "FCCAnalyses::ZHfunctions::get_GT_jets_from_initial_particles(Particle, MC_quark_index);")

    df = df.Define("_serialized_evt", "FCCAnalyses::Utils::serialize_event(ReconstructedParticlesEtaFilter);")
    df = df.Define("_serialized_calo_jets", "FCCAnalyses::Utils::serialize_event(CaloJetDurham);")
    df = df.Define("stable_gen_part_neutrinoFilter", "FCCAnalyses::ZHfunctions::stable_particles(Particle, true).first")
    df = df.Define("stable_gen_particles_idx", "FCCAnalyses::ZHfunctions::stable_particles(Particle, true).second")
    # Durham jets
    df = get_jet_vars(df, "stable_gen_part_neutrinoFilter", N_durham=nJets_processList[dataset], name="FastJet_jets")
    if os.environ.get("MATCH_RECO_JETS", "0") == "1":
        # If set to 1, it will use reco jets that are made out of the particles matched to the gen jet constituents
        print("No reco clustering - simply match gen jet constituents to corresponding reco particles")
        df = get_jet_vars_from_genjet_matching(df, name="FastJet_jets_reco", genjet_name="FastJet_jets")
    else:
        df = get_jet_vars(df, "ReconstructedParticlesEtaFilter", N_durham=nJets_processList[dataset], name="FastJet_jets_reco")
        df = get_jet_vars_from_genjet_matching(df, name="FastJet_jets_reco_IdealMatching", genjet_name="FastJet_jets")
        df = df.Define("_reco_particle_to_jet_mapping_idealMatching",
                       "FCCAnalyses::ZHfunctions::get_reco_particle_jet_mapping(ReconstructedParticlesEtaFilter.size(), FastJet_jets_reco_IdealMatching);")
    df = df.Define("_reco_particle_to_jet_mapping", "FCCAnalyses::ZHfunctions::get_reco_particle_jet_mapping(ReconstructedParticlesEtaFilter.size(), FastJet_jets_reco);")
    if os.environ.get("MATCH_RECO_JETS", "0") != "1":
        # get both _reco_particle_to_jet_mapping and _reco_particle_to_jet_mapping_idealMatching as list
        reco_to_jet = df.AsNumpy(["_reco_particle_to_jet_mapping"])["_reco_particle_to_jet_mapping"]
        reco_to_jet_ideal = df.AsNumpy(["_reco_particle_to_jet_mapping_idealMatching"])["_reco_particle_to_jet_mapping_idealMatching"]
        reco_to_jet = list([list(item) for item in reco_to_jet])
        reco_to_jet_ideal = list([list(item) for item in reco_to_jet_ideal])
        print("Reco particle to jet:", reco_to_jet[:4])
        print("Reco particle to jet ideal:", reco_to_jet_ideal[:4])

        pickle.dump(
            {"reco2jet": reco_to_jet, "reco2jet_ideal": reco_to_jet_ideal},
                open(os.path.join(outputDir, "reco_particle_to_jet_mapping_{}.pkl".format(dataset)), "wb")
        )
    df = df.Define("GenJetFastJet", "FCCAnalyses::ZHfunctions::fastjet_to_vec_rp_jet(FastJet_jets, {})".format(
        nJets_processList[dataset]))
    df = df.Define("RecoJetFastJet", "FCCAnalyses::ZHfunctions::fastjet_to_vec_rp_jet(FastJet_jets_reco, {})".format(
        nJets_processList[dataset]))
    df = df.Define("jet_energies", "FCCAnalyses::ZHfunctions::sort_jet_energies({})".format(rf))
    df = df.Define("genjet_energies", "FCCAnalyses::ZHfunctions::sort_jet_energies({})".format(gf))
    df = df.Define("fancy_matching",
                   f"FCCAnalyses::ZHfunctions::get_reco_truth_jet_mapping_greedy({rf}, {gf}, 1.0)")
    df = df.Define("distance_between_genjets", f"FCCAnalyses::ZHfunctions::get_jet_distances({gf})")
    df = df.Define("distance_between_recojets", f"FCCAnalyses::ZHfunctions::get_jet_distances({rf})")
    df = df.Define("min_distance_between_genjets",
                   f"FCCAnalyses::ZHfunctions::min(FCCAnalyses::ZHfunctions::get_jet_distances({gf}))")
    df = df.Define("min_distance_between_recojets",
                   f"FCCAnalyses::ZHfunctions::min(FCCAnalyses::ZHfunctions::get_jet_distances({rf}))")
    df = df.Define("matched_genjet_E_and_all_genjet_E",
                   f"FCCAnalyses::ZHfunctions::matched_genjet_E_and_all_genjet_E(fancy_matching, {gf})")
    df = df.Define("matched_genjet_energies", "std::get<0>(matched_genjet_E_and_all_genjet_E)")
    df = df.Define("matching_filter", "(matched_genjet_energies.size() == genjet_energies.size()) && (genjet_energies.size() == {})".format(nJets_processList[dataset]))
    df = df.Define("all_genjet_energies", "std::get<1>(matched_genjet_E_and_all_genjet_E)")
    # fancy matching to numpy print first two events
    matching_idx = df.AsNumpy(["fancy_matching"])["fancy_matching"]
    print("Matching event 0: ", list(matching_idx[0]))
    df = df.Define("matching_processing",
                   f"FCCAnalyses::ZHfunctions::get_energy_ratios_for_matched_jets(fancy_matching, {rf}, {gf})")
    df = df.Define("ratio_jet_energies_fancy", "std::get<0>(matching_processing)")
    df = df.Define("E_of_unmatched_reco_jets", "std::get<1>(matching_processing)")
    df = df.Define("num_unmatched_reco_jets", "E_of_unmatched_reco_jets.size()")
    df = df.Define("E_of_matched_gen_jets", "std::get<2>(matching_processing)")
    l = df.AsNumpy(["ratio_jet_energies_fancy"])["ratio_jet_energies_fancy"]
    print("Energy ratios for event 0:", l[0])

    E_matched_jets = df.AsNumpy(["E_of_matched_gen_jets"])["E_of_matched_gen_jets"]
    n_unmatched = df.AsNumpy(["num_unmatched_reco_jets"])["num_unmatched_reco_jets"]
    l = list([list(item) for item in l])
    # Save this to pickle
    E_matched_jets = list([list(item) for item in E_matched_jets])
    matching_filter = df.AsNumpy(["matching_filter"])["matching_filter"]
    matching_filter = list(matching_filter)
    pickle.dump(E_matched_jets, open(os.path.join(outputDir, "E_matched_jets_{}.pkl".format(dataset)), "wb"))
    pickle.dump(l, open(os.path.join(outputDir, "reco_over_gen_jet_energy_ratios_{}.pkl".format(dataset)), "wb"))
    pickle.dump(matching_filter, open(os.path.join(outputDir, "matching_filter_{}.pkl".format(dataset)), "wb"))
    n_unmatched = list(n_unmatched)
    print("Number of unmatched reco jets per event: ", n_unmatched)
    df = df.Define("_stable_gen_particle_to_jet_mapping", "FCCAnalyses::ZHfunctions::get_reco_particle_jet_mapping(stable_gen_part_neutrinoFilter.size(), FastJet_jets);")
    # AK6 jets
    #df = get_jet_vars(df, "stable_gen_part_neutrinoFilter", name="FastJet_jets", is_ee_AK=True, AK_radius=0.6)
    #df = get_jet_vars(df, "ReconstructedParticles", name="FastJet_jets_reco", is_ee_AK=True, AK_radius=0.6)

    df = df.Define("_serialized_evt_gen", "FCCAnalyses::Utils::serialize_event(stable_gen_part_neutrinoFilter);")
    #df = df.Define("")  # JET CLUSTERING HERE
    #df = get_jet_vars(df, "ReconstructedParticles", N_durham=2)
    df = df.Define("_serialized_jets", "FCCAnalyses::Utils::serialize_event(RecoJetFastJet);")
    df = df.Define("_serialized_evt_eta", "std::get<0>(_serialized_evt);")
    df = df.Define("_serialized_evt_phi", "std::get<1>(_serialized_evt);")
    df = df.Define("_serialized_evt_pt", "std::get<2>(_serialized_evt);")
    df = df.Define("_serialized_evt_PDG", "std::get<3>(_serialized_evt);")
    df = df.Define("_serialized_evt_mass", "std::get<4>(_serialized_evt);")
    df = df.Define("_serialized_calojets_eta", "std::get<0>(_serialized_calo_jets);")
    df = df.Define("_serialized_calojets_phi", "std::get<1>(_serialized_calo_jets);")
    df = df.Define("_serialized_calojets_pt", "std::get<2>(_serialized_calo_jets);")
    df = df.Define("_serialized_evt_gen_eta", "std::get<0>(_serialized_evt_gen);")
    df = df.Define("_serialized_evt_gen_phi", "std::get<1>(_serialized_evt_gen);")
    df = df.Define("_serialized_evt_gen_pt", "std::get<2>(_serialized_evt_gen);")
    df = df.Define("_serialized_evt_gen_PDG", "std::get<3>(_serialized_evt_gen);")
    df = df.Define("_serialized_evt_gen_mass", "std::get<4>(_serialized_evt_gen);")
    df = df.Define("_serialized_jets_eta", "std::get<0>(_serialized_jets);")
    df = df.Define("_serialized_jets_phi", "std::get<1>(_serialized_jets);")
    df = df.Define("_serialized_jets_pt", "std::get<2>(_serialized_jets);")
    df = df.Define("_serialized_jets_m", "std::get<4>(_serialized_jets);")
    df = df.Define("_serialized_initial_partons", "FCCAnalyses::Utils::serialize_event(GT_jets);")
    df = df.Define("_serialized_initial_partons_eta", "std::get<0>(_serialized_initial_partons);")
    df = df.Define("_serialized_initial_partons_phi", "std::get<1>(_serialized_initial_partons);")
    df = df.Define("_serialized_initial_partons_pt", "std::get<2>(_serialized_initial_partons);")
    df = df.Define("_serialized_genjets", "FCCAnalyses::Utils::serialize_event(GenJetFastJet);")
    df = df.Define("_serialized_genjets_eta", "std::get<0>(_serialized_genjets);")
    df = df.Define("_serialized_genjets_phi", "std::get<1>(_serialized_genjets);")
    df = df.Define("_serialized_genjets_pt", "std::get<2>(_serialized_genjets);")
    df = df.Define("_calohits_as_vec_rp", "FCCAnalyses::Utils::convert_calohits_to_vec_rp(CalorimeterHits);")
    df = df.Define("_calohits_serialized", "FCCAnalyses::Utils::serialize_event(_calohits_as_vec_rp);")
    df = df.Define("_calohits_eta", "std::get<0>(_calohits_serialized);")
    df = df.Define("_calohits_phi", "std::get<1>(_calohits_serialized);")
    df = df.Define("_calohits_pt", "std::get<2>(_calohits_serialized);")
    # Also get the truth particles (quarks etc.)
    df = get_Higgs_mass_with_truth_matching(df, genjets_field=gf, recojets_field=rf, expected_num_jets=nJets_from_H_process_list[dataset])
    df = df.Define("MCparts", "FCCAnalyses::Utils::serialize_event(MC_part_asjets)")
    df = df.Define("MCparts_eta", "std::get<0>(MCparts);")
    df = df.Define("MCparts_phi", "std::get<1>(MCparts);")
    df = df.Define("MCparts_pt", "std::get<2>(MCparts);")
    df = df.Define("inv_mass_all_gen_particles", "FCCAnalyses::ZHfunctions::invariant_mass(stable_gen_part_neutrinoFilter);")
    tonumpy = df.AsNumpy(["_serialized_evt_eta", "_serialized_evt_phi", "_serialized_evt_pt", "_serialized_evt_mass",
                          "_serialized_jets_eta", "_serialized_jets_phi", "_serialized_jets_pt", "_serialized_initial_partons_eta",
                          "_serialized_initial_partons_phi", "_serialized_initial_partons_pt", "_serialized_genjets_eta",
                          "_serialized_genjets_phi", "_serialized_genjets_pt", "_serialized_evt_gen_eta",
                          "_serialized_evt_gen_phi", "_serialized_evt_gen_pt", "MCparts_eta", "MCparts_phi", "MCparts_pt",
                          "_serialized_evt_gen_PDG", "_serialized_evt_gen_mass",  "_calohits_eta", "_calohits_phi", "_calohits_pt", "_serialized_jets_m",
                          "MC_quark_index", "_serialized_calojets_eta", "_serialized_calojets_phi",
                          "_serialized_calojets_pt", "fancy_matching", "HardP_to_GenJet_mapping", "HardP_to_RecoJet_mapping",
                          "_serialized_evt_PDG", "_reco_particle_to_jet_mapping", "_stable_gen_particle_to_jet_mapping"])
    tonumpy = {key: list([list(x) for x in tonumpy[key]]) for key in tonumpy}
    #inv_mass_gen_all = list(df.AsNumpy(["inv_mass_gen_all"])["inv_mass_gen_all"])
    inv_mass_all_gen_p = list(df.AsNumpy(["inv_mass_all_gen_particles"])["inv_mass_all_gen_particles"])
    inv_mass_reco_higgs = list(df.AsNumpy(["inv_mass_reco"])["inv_mass_reco"])
    inv_mass_gen_higgs = list(df.AsNumpy(["inv_mass_gen"])["inv_mass_gen"])
    mcPart_idx = list(df.AsNumpy(["MC_quark_index"])["MC_quark_index"])
    matchings = {
        "GenJet-RecoJet": list(df.AsNumpy(["fancy_matching"])["fancy_matching"]),
        "HardP-GenJet": list(df.AsNumpy(["HardP_to_GenJet_mapping"])["HardP_to_GenJet_mapping"]),
        "HardP-RecoJet": list(df.AsNumpy(["HardP_to_RecoJet_mapping"])["HardP_to_RecoJet_mapping"]),
        "PartonPT": list(df.AsNumpy(["_serialized_initial_partons_pt"])["_serialized_initial_partons_pt"]),
        "GenJetPT": list(df.AsNumpy(["_serialized_genjets_pt"])["_serialized_genjets_pt"]),
        "RecoJetPT": list(df.AsNumpy(["_serialized_jets_pt"])["_serialized_jets_pt"]),
    }
    # print _serialized_evt_mass, _serialized_evt_gen_mass, and the PDG IDs for the first three events
    print("_serialized_evt_mass (first 3 events):", [tonumpy["_serialized_evt_mass"][i] for i in range(3)])
    print("_serialized_evt_PDG (first 3 events):", [tonumpy["_serialized_evt_PDG"][i] for i in range(3)])
    print("_serialized_evt_gen_mass (first 3 events):", [tonumpy["_serialized_evt_gen_mass"][i] for i in range(3)])
    print("_serialized_evt_gen_PDG (first 3 events):", [tonumpy["_serialized_evt_gen_PDG"][i] for i in range(3)])
    gen_jets_inv_mass = []
    gen_p_inv_mass = []
    #for event_idx in range(len(l)):
    #    gen_jets_inv_mass.append(inv_mass_gen_all[event_idx])
    #    gen_p_inv_mass.append(inv_mass_all_gen_p[event_idx])
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    # print the correlation between gen_jets_inv_mass and gen_p_inv_mass
    ax[0].scatter(gen_jets_inv_mass, gen_p_inv_mass, alpha=0.5)
    import numpy as np
    bins = np.linspace(0, 200, 50)
    ax[1].hist(gen_jets_inv_mass, bins=bins, histtype="step", label="gen jets inv mass")
    ax[1].hist(gen_p_inv_mass, bins=bins, histtype="step", label="all gen particles inv mass")
    ax[1].legend()
    ax[0].set_xlabel("mass of gen jets")
    ax[0].set_ylabel("mass of all gen particles")
    fig.tight_layout()
    # check what proportion of them are the same
    fig.savefig(os.path.join(outputDir, "genjets_vs_genparticles_mass_{}.pdf".format(dataset)))
    print("---------------------------------------------------")
    #print("Inv mass gen all (first 5 entries):", inv_mass_gen_all[:5])
    for event_idx in range(len(l)):
        ##assert n_unmatched[event_idx]  ==  len([x for x in l[event_idx] if x < 0]), "n_unmatched does not match the length of unmatched jets!:" + str(n_unmatched[event_idx]) + " vs " + str(len([x for x in l[event_idx] if x < 0]))
        if plot_filter(l[event_idx], n_unmatched[event_idx], inv_mass_reco_higgs[event_idx], E_matched_jets[event_idx] , idx=PLOT_IDX):
            if global_event_idx.get(dataset, 0) > 5:
                return [], weightsum # Just plot the first X events
            #event_idx = # I want an event idx that is unique per dataset, even with multiple input root files. How do I do this?
            #print("Plotting event idx ", global_event_idx.get(dataset, 0), " from dataset ", dataset)
            eta, phi, pt = tonumpy["_serialized_evt_eta"][event_idx], tonumpy["_serialized_evt_phi"][event_idx], tonumpy["_serialized_evt_pt"][event_idx]
            particle_to_jet = tonumpy["_reco_particle_to_jet_mapping"][event_idx]
            stable_gen_particle_to_jet = tonumpy["_stable_gen_particle_to_jet_mapping"][event_idx]
            vec_rp = Vec_RP(eta=eta, phi=phi, pt=pt, pdg=tonumpy["_serialized_evt_PDG"][event_idx], jets=particle_to_jet)
            etamc, phimc, ptmc = tonumpy["_serialized_evt_gen_eta"][event_idx], tonumpy["_serialized_evt_gen_phi"][event_idx], tonumpy["_serialized_evt_gen_pt"][event_idx]
            vec_mc = Vec_RP(eta=etamc, phi=phimc, pt=ptmc, pdg=tonumpy["_serialized_evt_gen_PDG"][event_idx], jets=stable_gen_particle_to_jet) #, txt=[str(pdg) for pdg in tonumpy["_serialized_evt_gen_PDG"][event_idx]])
            jets_eta, jets_phi, jets_pt = tonumpy["_serialized_jets_eta"][event_idx], tonumpy["_serialized_jets_phi"][event_idx], tonumpy["_serialized_jets_pt"][event_idx]
            jets_m = tonumpy["_serialized_jets_m"][event_idx]
            print("jets_m", jets_m)
            #jets_text = l[event_idx]
            jets_text = [f"pt={round(jets_pt[i], 2)}e={round(jets_eta[i], 2)}ph={round(jets_phi[i], 2)}m={round(jets_m[i], 2)}" for i in range(len(jets_eta))]
            vec_jets = Vec_RP(eta=jets_eta, phi=jets_phi, pt=jets_pt, m=jets_m)#, txt=jets_text)
            gt_eta, gt_phi, gt_pt = tonumpy["_serialized_initial_partons_eta"][event_idx], tonumpy["_serialized_initial_partons_phi"][event_idx], tonumpy["_serialized_initial_partons_pt"][event_idx]
            vec_gt = Vec_RP(eta=gt_eta, phi=gt_phi, pt=gt_pt)
            genjets_eta, genjets_phi, genjets_pt = tonumpy["_serialized_genjets_eta"][event_idx], tonumpy["_serialized_genjets_phi"][event_idx], tonumpy["_serialized_genjets_pt"][event_idx]
            vec_genjets = Vec_RP(eta=genjets_eta, phi=genjets_phi, pt=genjets_pt)
            mcpart_eta, mcpart_phi, mcpart_pt = tonumpy["MCparts_eta"][event_idx], tonumpy["MCparts_phi"][event_idx], tonumpy["MCparts_pt"][event_idx]
            vec_mcparts = Vec_RP(eta=mcpart_eta, phi=mcpart_phi, pt=mcpart_pt)
            #print("Length of initial partons: ", len(mcpart_eta), "Length of MC parton idx: ", len(mcPart_idx[event_idx]))
            calohits_eta, calohits_phi = tonumpy["_calohits_eta"][event_idx], tonumpy["_calohits_phi"][event_idx]
            calohits_pt = np.ones(len(calohits_eta)) * 5  # Dummy pt for calohits (for some reason energy is not being stored)
            vec_calohits = Vec_RP(eta=calohits_eta, phi=calohits_phi, pt=calohits_pt)
            vec_calojets = Vec_RP(eta=tonumpy["_serialized_calojets_eta"][event_idx], phi=tonumpy["_serialized_calojets_phi"][event_idx],
                                  pt=tonumpy["_serialized_calojets_pt"][event_idx])
            #print("Vec calohits: eta : ", calohits_eta[:5], " phi: ", calohits_phi[:5], " pt: ", calohits_pt[:5])
            #gj_fccanalysis_eta, gj_fccanalysis_phi, gj_fccanalysis_pt = tonumpy["fj_eta"][event_idx], tonumpy["fj_phi"][event_idx], tonumpy["fj_pt"][event_idx]
            #vec_genjets_fccanalysis = Vec_RP(eta=gj_fccanalysis_eta, phi=gj_fccanalysis_phi, pt=gj_fccanalysis_pt)
            event = Event(vec_rp=vec_rp, vec_mc=vec_mc, additional_collections={
                "RecoJets": vec_jets, "InitialPartons": vec_mcparts, "GenJets": vec_genjets,
                "Status1GenParticles": vec_mc, "CaloHits": vec_calohits, "CaloJets": vec_calojets}, #"GenJetsFCCAnalysis": vec_genjets_fccanalysis}
                additional_text_dict={key: val[event_idx] for key, val in matchings.items()}
            )
            fig, ax = event.display()
            ax[0, 0].set_title("{}, event {}, len(in.part.)={}, mHreco={} mHgen={} mcPart={}".format(dataset, global_event_idx.get(dataset, 0), len(mcpart_eta), round(inv_mass_reco_higgs[event_idx], 2), round(inv_mass_gen_higgs[event_idx], 2), mcPart_idx[event_idx]))
            fig.tight_layout()
            if not os.path.exists(outputDir):
                os.makedirs(outputDir)
            fig.savefig(os.path.join(outputDir, "event_{}_{}.png".format(dataset, global_event_idx.get(dataset, 0))))
            # Close the figure
            fig.clf()
            del fig
            plotly_fig = event.plot_eta_phi()
            plotly_fig.write_html(os.path.join(outputDir, "event_{}_{}.html".format(dataset, global_event_idx.get(dataset, 0))))
            global_event_idx[dataset] = global_event_idx.get(dataset, 0) + 1
    return [], weightsum
