# fccanalysis run event_plotter.py
# source /cvmfs/fcc.cern.ch/sw/latest/setup.sh
# source /cvmfs/sw.hsf.org/key4hep/setup.sh
# source  /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh
import os
from event_displays import Vec_RP, Event
from truth_matching import get_Higgs_mass_with_truth_matching
from jet_helper import get_jet_vars


inputDir = "/fs/ddn/sdf/group/atlas/d/gregork/fastsim/jetbenchmarks/"

processList = {
    #'p8_ee_WW_ecm365_fullhad': {'fraction': 0.01},
    #"p8_ee_ZH_qqbb_ecm365": {'fraction': 0.01},
    #"p8_ee_ZH_6jet_ecm365": {'fraction': 0.01},
    "p8_ee_ZH_vvbb_ecm365": {"fraction": 0.0001},
    #"p8_ee_ZH_bbbb_ecm365": {'fraction': 0.01},
    #"p8_ee_ZH_vvgg_ecm365": {'fraction': 0.01},
    #"p8_ee_ZH_qqbb_ecm365": {'fraction': 1},
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


'''
PLOT IDX options:
1: the slice of the E_reco/E_true around 0.9
2: events with unmatched reco-to-gen jets

'''

PLOT_IDX = 2

# Optional: output directory, default is local running directory
outputDir = "../../idea_fullsim/fast_sim/event_displays_vvbb_" + str(PLOT_IDX)
#outputDir = "../../idea_fullsim/fast_sim/histograms"



def plot_filter(E_reco_over_true, n_unmatched, idx=1):
    # TODO: implement different plotting filter idxs
    if idx == 2 and n_unmatched > 0:
        return True
    elif idx == 1:
        for E in E_reco_over_true:
            if (E > 0.87) and (E < 0.93):
                return True
    return False


def build_graph(df, dataset):
    global global_event_idx # is this thread-safe??
    df = df.Define("weight", "1.0")
    weightsum = df.Sum("weight")
    df = df.Define("MC_quark_index", "FCCAnalyses::ZHfunctions::get_MC_quark_index(Particle);")
    df = df.Define("GT_jets", "FCCAnalyses::ZHfunctions::get_GT_jets_from_initial_particles(Particle, MC_quark_index);")
    df = df.Define("jet_energies", "FCCAnalyses::ZHfunctions::sort_jet_energies(JetDurhamN4)")
    df = df.Define("genjet_energies", "FCCAnalyses::ZHfunctions::sort_jet_energies(GenJetDurhamN4)")
    df = df.Define("ratio_jet_energies", "FCCAnalyses::ZHfunctions::elementwise_divide(jet_energies, genjet_energies)")
    df = df.Define("fancy_matching",
                   "FCCAnalyses::ZHfunctions::get_reco_truth_jet_mapping_greedy(JetDurhamN4, GenJetDurhamN4, 1.0)")
    df = df.Define("distance_between_genjets", "FCCAnalyses::ZHfunctions::get_jet_distances(GenJetDurhamN4)")
    df = df.Define("distance_between_recojets", "FCCAnalyses::ZHfunctions::get_jet_distances(JetDurhamN4)")
    df = df.Define("min_distance_between_genjets",
                   "FCCAnalyses::ZHfunctions::min(FCCAnalyses::ZHfunctions::get_jet_distances(GenJetDurhamN4))")
    df = df.Define("min_distance_between_recojets",
                   "FCCAnalyses::ZHfunctions::min(FCCAnalyses::ZHfunctions::get_jet_distances(JetDurhamN4))")
    df = df.Define("matched_genjet_E_and_all_genjet_E",
                   "FCCAnalyses::ZHfunctions::matched_genjet_E_and_all_genjet_E(fancy_matching, GenJetDurhamN4)")
    df = df.Define("matched_genjet_energies", "std::get<0>(matched_genjet_E_and_all_genjet_E)")
    df = df.Define("all_genjet_energies", "std::get<1>(matched_genjet_E_and_all_genjet_E)")
    df = df.Define("matching_processing",
                   "FCCAnalyses::ZHfunctions::get_energy_ratios_for_matched_jets(fancy_matching, JetDurhamN4, GenJetDurhamN4)")
    df = df.Define("ratio_jet_energies_fancy", "std::get<0>(matching_processing)")
    df = df.Define("E_of_unmatched_reco_jets", "std::get<1>(matching_processing)")
    df = df.Define("num_unmatched_reco_jets", "E_of_unmatched_reco_jets.size()")
    l = df.AsNumpy(["ratio_jet_energies_fancy"])["ratio_jet_energies_fancy"]
    n_unmatched = df.AsNumpy(["num_unmatched_reco_jets"])["num_unmatched_reco_jets"]
    l = list([list(item) for item in l])
    n_unmatched = list(n_unmatched)
    print("Number of unmatched reco jets per event: ", n_unmatched)
    df = df.Define("_serialized_evt", "FCCAnalyses::Utils::serialize_event(ReconstructedParticles);")
    df = df.Define("stable_gen_part_neutrinoFilter", "FCCAnalyses::ZHfunctions::stable_particles(Particle, true)")
    df = df.Define("_serialized_evt_gen", "FCCAnalyses::Utils::serialize_event(stable_gen_part_neutrinoFilter);")
    #df = df.Define("")  # JET CLUSTERING HERE
    df = get_jet_vars(df, "ReconstructedParticles", N_durham=2)
    df = df.Define("_serialized_jets", "FCCAnalyses::Utils::serialize_event(JetDurhamN4);")
    df = df.Define("_serialized_evt_eta", "std::get<0>(_serialized_evt);")
    df = df.Define("_serialized_evt_phi", "std::get<1>(_serialized_evt);")
    df = df.Define("_serialized_evt_pt", "std::get<2>(_serialized_evt);")
    df = df.Define("_serialized_evt_gen_eta", "std::get<0>(_serialized_evt_gen);")
    df = df.Define("_serialized_evt_gen_phi", "std::get<1>(_serialized_evt_gen);")
    df = df.Define("_serialized_evt_gen_pt", "std::get<2>(_serialized_evt_gen);")
    df = df.Define("_serialized_evt_gen_PDG", "std::get<3>(_serialized_evt_gen);")
    df = df.Define("_serialized_jets_eta", "std::get<0>(_serialized_jets);")
    df = df.Define("_serialized_jets_phi", "std::get<1>(_serialized_jets);")
    df = df.Define("_serialized_jets_pt", "std::get<2>(_serialized_jets);")
    df = df.Define("_serialized_initial_partons", "FCCAnalyses::Utils::serialize_event(GT_jets);")
    df = df.Define("_serialized_initial_partons_eta", "std::get<0>(_serialized_initial_partons);")
    df = df.Define("_serialized_initial_partons_phi", "std::get<1>(_serialized_initial_partons);")
    df = df.Define("_serialized_initial_partons_pt", "std::get<2>(_serialized_initial_partons);")
    df = df.Define("_serialized_genjets", "FCCAnalyses::Utils::serialize_event(GenJetDurhamN4);")
    df = df.Define("_serialized_genjets_eta", "std::get<0>(_serialized_genjets);")
    df = df.Define("_serialized_genjets_phi", "std::get<1>(_serialized_genjets);")
    df = df.Define("_serialized_genjets_pt", "std::get<2>(_serialized_genjets);")
    # Also get the truth particles (quarks etc)
    df = get_Higgs_mass_with_truth_matching(df)
    df = df.Define("MCparts", "FCCAnalyses::Utils::serialize_event(MC_part_asjets)")
    df = df.Define("MCparts_eta", "std::get<0>(MCparts);")
    df = df.Define("MCparts_phi", "std::get<1>(MCparts);")
    df = df.Define("MCparts_pt", "std::get<2>(MCparts);")

    tonumpy = df.AsNumpy(["_serialized_evt_eta", "_serialized_evt_phi", "_serialized_evt_pt", "_serialized_jets_eta",
                          "_serialized_jets_phi", "_serialized_jets_pt", "_serialized_initial_partons_eta",
                          "_serialized_initial_partons_phi", "_serialized_initial_partons_pt", "_serialized_genjets_eta",
                          "_serialized_genjets_phi", "_serialized_genjets_pt", "_serialized_evt_gen_eta",
                          "_serialized_evt_gen_phi", "_serialized_evt_gen_pt", "MCparts_eta", "MCparts_phi", "MCparts_pt",
                          "_serialized_evt_gen_PDG", "fj_eta", "fj_phi", "fj_pt"])
    tonumpy = {key: list([list(x) for x in tonumpy[key]]) for key in tonumpy}
    for event_idx in range(len(l)):
        ##assert n_unmatched[event_idx] == len([x for x in l[event_idx] if x < 0]), "n_unmatched does not match the length of unmatched jets!:" + str(n_unmatched[event_idx]) + " vs " + str(len([x for x in l[event_idx] if x < 0]))
        if plot_filter(l[event_idx], n_unmatched[event_idx], idx=PLOT_IDX):
            if global_event_idx.get(dataset, 0) > 10:
                return [], weightsum # Just plot 10 events... #
            #event_idx = # I want an event idx that is unique per dataset, even with multiple input root files. How do I do this?
            #print("Plotting event idx ", global_event_idx.get(dataset, 0), " from dataset ", dataset)
            eta, phi, pt = tonumpy["_serialized_evt_eta"][event_idx], tonumpy["_serialized_evt_phi"][event_idx], tonumpy["_serialized_evt_pt"][event_idx]
            vec_rp = Vec_RP(eta=eta, phi=phi, pt=pt)
            etamc, phimc, ptmc = tonumpy["_serialized_evt_gen_eta"][event_idx], tonumpy["_serialized_evt_gen_phi"][event_idx], tonumpy["_serialized_evt_gen_pt"][event_idx]
            vec_mc = Vec_RP(eta=etamc, phi=phimc, pt=ptmc)#, txt=[str(pdg) for pdg in tonumpy["_serialized_evt_gen_PDG"][event_idx]])
            jets_eta, jets_phi, jets_pt = tonumpy["_serialized_jets_eta"][event_idx], tonumpy["_serialized_jets_phi"][event_idx], tonumpy["_serialized_jets_pt"][event_idx]
            jets_text = l[event_idx]
            vec_jets = Vec_RP(eta=jets_eta, phi=jets_phi, pt=jets_pt, txt=[str(round(x, 2)) for x in jets_text])
            gt_eta, gt_phi, gt_pt = tonumpy["_serialized_initial_partons_eta"][event_idx], tonumpy["_serialized_initial_partons_phi"][event_idx], tonumpy["_serialized_initial_partons_pt"][event_idx]
            vec_gt = Vec_RP(eta=gt_eta, phi=gt_phi, pt=gt_pt)
            genjets_eta, genjets_phi, genjets_pt = tonumpy["_serialized_genjets_eta"][event_idx], tonumpy["_serialized_genjets_phi"][event_idx], tonumpy["_serialized_genjets_pt"][event_idx]
            vec_genjets = Vec_RP(eta=genjets_eta, phi=genjets_phi, pt=genjets_pt)
            mcpart_eta, mcpart_phi, mcpart_pt = tonumpy["MCparts_eta"][event_idx], tonumpy["MCparts_phi"][event_idx], tonumpy["MCparts_pt"][event_idx]
            vec_mcparts = Vec_RP(eta=mcpart_eta, phi=mcpart_phi, pt=mcpart_pt)
            print("Length of initial partons: ", len(mcpart_eta))
            gj_fccanalysis_eta, gj_fccanalysis_phi, gj_fccanalysis_pt = tonumpy["fj_eta"][event_idx], tonumpy["fj_phi"][event_idx], tonumpy["fj_pt"][event_idx]
            vec_genjets_fccanalysis = Vec_RP(eta=gj_fccanalysis_eta, phi=gj_fccanalysis_phi, pt=gj_fccanalysis_pt)

            event = Event(vec_rp=vec_rp, additional_collections={
                "RecoJets": vec_jets, "InitialPartons": vec_mcparts, "GenJets": vec_genjets,
                "Status1GenParticles": vec_mc, "GenJetsFCCAnalysis": vec_genjets_fccanalysis}
            )
            fig, ax = event.display()
            ax[0].set_title("{}, event {}, len(in.part.)={}".format(dataset, global_event_idx.get(dataset, 0), len(mcpart_eta)))
            fig.tight_layout()
            if not os.path.exists(outputDir):
                os.makedirs(outputDir)
            fig.savefig(os.path.join(outputDir, "event_{}_{}.png".format(dataset, global_event_idx.get(dataset, 0))))
            # Close the figure
            fig.clf()
            del fig
            global_event_idx[dataset] = global_event_idx.get(dataset, 0) + 1
    return [], weightsum

