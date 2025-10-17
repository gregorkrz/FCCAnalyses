# fccanalysis run playground.py
# source /cvmfs/fcc.cern.ch/sw/latest/setup.sh
# source /cvmfs/sw.hsf.org/key4hep/setup.sh
# source  /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh


inputDir = "/fs/ddn/sdf/group/atlas/d/gregork/fastsim/jetbenchmarks/"

processList = {
    'p8_ee_WW_ecm365_fullhad': {'fraction': 1},
    "p8_ee_ZH_qqbb_ecm365": {'fraction': 1},
    "p8_ee_ZH_6jet_ecm365": {'fraction': 1},
    "p8_ee_ZH_vvbb_ecm365": {'fraction': 1},

    ## Not used below
    # 'wzp6_ee_mumuH_ecm240':{'fraction':1},
    # "p8_ee_ZH_llbb_ecm365": {'fraction': 1},
    #'p8_ee_WW_mumu_ecm240': {'fraction': 1, 'crossSection': 0.25792},
    #'p8_ee_ZZ_mumubb_ecm240': {'fraction': 1, 'crossSection': 2 * 1.35899 * 0.034 * 0.152},
    #'p8_ee_ZH_Zmumu_ecm240': {'fraction': 1, 'crossSection': 0.201868 * 0.034},
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
includePaths = ["functions.h"]

# Define the input dir (optional)
# inputDir    = "outputs/FCCee/higgs/mH-recoil/mumu/stage1"
#inputDir = "../../idea_fullsim/fast_sim/outputs"

# Optional: output directory, default is local running directory
outputDir = "../../idea_fullsim/fast_sim/histograms/ideal_clustering"
#outputDir = "../../idea_fullsim/fast_sim/histograms"

# optional: ncpus, default is 4, -1 uses all cores available
nCPUS = -1

# scale the histograms with the cross-section and integrated luminosity
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
    df = df.Define("weight", "1.0")
    weightsum = df.Sum("weight")
    df = df.Define("MC_quark_index", "FCCAnalyses::ZHfunctions::get_MC_quark_index(Particle);")
    df = df.Define("GT_labels", "FCCAnalyses::ZHfunctions::getGTLabels(MC_quark_index, Particle, _Particle_daughters.index);")
    print("GT labels:", df.AsNumpy(["GT_labels"])["GT_labels"])
    df = df.Define("MClinks", "FCCAnalyses::ZHfunctions::getRP2MC_index(_RecoMCLink_from.index, _RecoMCLink_to.index, ReconstructedParticles, Particle)")
    df = df.Define("mc2rp", "MClinks.second")
    df = df.Define("IdealClusteringRecoJetsLabels", "FCCAnalyses::ZHfunctions::convertMCJetLabelsIntoRecoJetLabels(GT_labels, mc2rp)")
    print("IdealClusteringRecoJetsLabels:", df.AsNumpy(["IdealClusteringRecoJetsLabels"])["IdealClusteringRecoJetsLabels"])
    df = df.Define("IdealClusteringRecoJets", "FCCAnalyses::ZHfunctions::get_jets_from_recojetlabels(IdealClusteringRecoJetsLabels, ReconstructedParticles)")
    #print("IdealClusteringRecoJets:", df.AsNumpy(["IdealClusteringRecoJets"])["IdealClusteringRecoJets"])
    #df = df.Define("njets_ideal_clustering", "IdealClusteringRecoJets.size()")
    #df = df.Define("Ejets_ideal_clustering", "IdealClusteringRecoJets.energy")
    #hist = df.Histo1D(("h_ejets", "E of jets (ideal clustering);E_{jets};Entries", 5, 0, 5), "Ejets_ideal_clustering")
    #df = df.Define("njets", "JetDurhamN4.size()")
    #df = df.Define("njets", "IdealClusteringRecoJets.size()")
    df = df.Define("njets", "IdealClusteringRecoJets.size()")
    hist = df.Histo1D(("h_njets_ideal_clustering", "N of jets (ideal clustering);N_{jets};Entries", 5, 0, 5), "njets")
    #if plot_filter(df.AsNumpy(["ratio_jet_energies_fancy"])["ratio_jet_energies_fancy"]):
    #    # event passed the filter. now plot it
    #    df = df.Define("_serialized_filt_event", "FCCAnalyses::ZHfunctions::serialize_event(ReconstructedParticles)")
    #    df = df.Define("_serialized_filt_event_jets", "FCCAnalyses::ZHfunctions::serialize_event(Particle)")
    return [hist], weightsum

