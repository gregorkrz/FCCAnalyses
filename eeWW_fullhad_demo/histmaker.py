# fccanalysis run histmaker_jetE.py

# list of processes (mandatory)
processList = {
    # 'p8_ee_ZZ_ecm240':{'fraction':1},
     'p8_ee_WW_ecm365_fullhad': {'fraction': 1},
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
includePaths = ["functions.h"]

# Define the input dir (optional)
# inputDir    = "outputs/FCCee/higgs/mH-recoil/mumu/stage1"
inputDir = "../../idea_fullsim/fast_sim/outputs"

# Optional: output directory, default is local running directory
outputDir = "../../idea_fullsim/fast_sim/histograms/p8_ee_WW_ecm365_fullhad"

# optional: ncpus, default is 4, -1 uses all cores available
nCPUS = -1

# scale the histograms with the cross-section and integrated luminosity
doScale = False
intLumi = 5000000  # 5 /ab

# Define some binning for various histograms
bins_count_particle = (50, 0, 350)

# build_graph function that contains the analysis logic, cuts and histograms (mandatory)
def build_graph(df, dataset):
    #results = []
    df = df.Define("weight", "1.0")
    weightsum = df.Sum("weight")
    df = df.Define("n_truth", "Particle.size()")
    df = df.Define("n_reco", "ReconstructedParticles.size()")
    #results.append(df.Histo1D(("n_reco", "Number of ReconstructedParticles;N_{reco}", *bins_count_particle), "n_reco", "weight"))
    #results.append(df.Histo1D(("n_truth", "Number of Truth Particles;N_{truth}", *bins_count_particle), "n_truth", "weight"))
    h_truth = df.Histo1D(("h_truth", "Number of truth particles per event;N;Events", *bins_count_particle), "n_truth")
    h_reco = df.Histo1D(("h_reco", "Number of reconstructed particles per event;N;Events", *bins_count_particle), "n_reco")
    return [h_reco, h_truth], weightsum

