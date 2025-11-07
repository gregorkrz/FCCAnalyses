# Use this script to run comparison of different algorithms but the same physics process.
# Move the files around using utils/reorganize_files.py first to create the right folder structure.

# FOLDER_NAME= HISTOGRAMS_FOLDER_NAME= fccanalysis plots plots_jetE_alljets_Compare_AK.py


import ROOT
import os


#assert "INPUT_DIR" in os.environ # To make sure we are taking the right input dir and folder name
#assert "FOLDER_NAME" in os.environ

#FOLDER_NAME in [p8_ee_ZH_6jet_ecm240, p8_ee_ZH_bbbb_ecm240, p8_ee_ZH_qqbb_ecm240, p8_ee_ZH_vvbb_ecm240, p8_ee_ZH_vvgg_ecm240]

assert "FOLDER_NAME" in os.environ
assert "HISTOGRAMS_FOLDER_NAME" in os.environ # default: Histograms_ECM240_AK_organized_NoISR


intLumi        = 1.
intLumiLabel   = ""
#ana_tex = os.environ["FOLDER_NAME"]
ana_tex = ""

delphesVersion = '3.4.2'
energy         = 240.0
collider       = 'FCC-ee'
formats        = ['png', 'pdf']

outdir         = '../../idea_fullsim/fast_sim/{}/{}'.format(os.environ["HISTOGRAMS_FOLDER_NAME"], os.environ["FOLDER_NAME"])
inputDir       = '../../idea_fullsim/fast_sim/{}/{}'.format(os.environ["HISTOGRAMS_FOLDER_NAME"], os.environ["FOLDER_NAME"])

plotStatUnc    = True

colors = {}
color_presets = [ROOT.kRed, ROOT.kBlue+1, ROOT.kCyan+2, ROOT.kMagenta, ROOT.kOrange+7, ROOT.kGray+1, ROOT.kGreen, ROOT.kOrange,
                 ROOT.kViolet, ROOT.kAzure+1, ROOT.kPink+1, ROOT.kTeal+1]

procs = {"signal": {}, "backgrounds": {}}

legend = {}

i = 0

for file in sorted(os.listdir(inputDir)):
    if file.endswith(".root"):
        proc_name = file.replace(".root", "")
        procs["signal"][proc_name] = [proc_name]
        legend[proc_name] = proc_name
        colors[proc_name] = color_presets[i]
        i += 1

print("Procs:", procs, "Legend:", legend)

hists = {}

hists["h_fancy"] = {
    "output":   "jet_E_deltaR_matching",
    "logy":     False,
    "stack":    False,
    "ymin": 0,
    "xtitle":   "E_reco/E_true (deltaR matching)",
    "ytitle":   "Events",
}

hists["h_unmatched_reco_jets"] = {
    "output":   "E_of_unmatched_reco_jets",
    "logy":     False,
    "stack":    False,
    #"rebin":    100,
    #"xmin":     120,
    #"xmax":     140,
    ##"ymin":     0,
    "xtitle":   "E of unmatched reco jets",
    "ytitle":   "Events",
}

hists["h_unmatched_reco_jets"] = {
    "output":   "E_of_unmatched_reco_jets",
    "logy":     False,
    "stack":    False,
    #"rebin":    100,
    #"xmin":     120,
    #"xmax":     140,
    ##"ymin":     0,
    "xtitle":   "E of unmatched reco jets",
    "ytitle":   "Events",
}

'''hists["h_E_all_jets"] = {
    "output":   "E_of_all_reco_jets",
    "logy":     False,
    "stack":    False,
    #"ymax": 15000,
    #"rebin":    100,
    #"xmin":     120,
    #"xmax":     140,
    ##"ymin":     0,
    "xtitle":   "E of reco jets",
    "ytitle":   "Events",
}'''


'''hists["h_E_all_genjets"] = {
    "output":   "E_of_all_gen_jets",
    "logy":     False,
    "stack":    False,
    #"rebin":    100,
    #"xmin":     120,
    #"xmax":     140,
    ##"ymin":     0,
    "xtitle":   "E of gen jets",
    "ytitle":   "Events",
}'''


hists["h_eta"] = {
    "output":   "h_eta",
    "logy":     False,
    "stack":    False,
    #"rebin":    100,
    #"xmin":     120,
    #"xmax":     140,
    ##"ymin":     0,
    "xtitle":   "eta of reco jets",
    "ytitle":   "Events",
}

hists["h_eta_gen"] = {
    "output":   "h_eta_gen",
    "logy":     False,
    "stack":    False,
    #"rebin":    100,
    #"xmin":     120,
    #"xmax":     140,
    ##"ymin":     0,
    "xtitle":   "eta of gen jets",
    "ytitle":   "Events",
}


hists["h_dist_jets_gen"] = {
    "output": "h_dist_jets_gen",
    "logy": False,
    "stack": False,
    "xtitle": "DeltaR between gen jets",
    "ytitle": "Pairs of gen jets",
}


hists["h_min_dist_jets_gen"] = {
    "output": "h_min_dist_jets_gen",
    "logy": False,
    "stack": False,
    "xtitle": "Min. DeltaR between gen jets",
    "ytitle": "Events",
}

hists["h_min_dist_jets_reco"] = {
    "output": "h_min_dist_jets_reco",
    "logy": False,
    "stack": False,
    "xtitle": "Min. DeltaR between reco jets",
    "ytitle": "Events",
}

hists["h_mH_reco"] = {
    "output": "h_mH_reco",
    "logy": False,
    "stack": False,
    "ymax": 20000,
    "xtitle": "Reconstructed Higgs mass (all matched jets)",
    "ytitle": "Events",
    # Normalize to 1
}

hists["h_mH_gen"] = {
    "output": "h_mH_gen",
    "logy": False,
    "stack": False,
    "ymax": 120000,
    "xtitle": " Higgs mass from gen particles (all matched jets)",
    "ytitle": "Events",
}



hists["h_njets"] = {
    "output":   "h_njets",
    "logy":     False,
    "stack":    False,
    #"rebin":    100,
    #"xmin":     120,
    #"xmax":     140,
    ##"ymin":     0,
    "xtitle":   "Number of reco jets",
    "ytitle":   "Events",
}

hists["h_ngenjets"] = {
    "output":   "h_ngenjets",
    "logy":     False,
    "stack":    False,
    #"rebin":    100,
    #"xmin":     120,
    #"xmax":     140,
    ##"ymin":     0,
    "xtitle":   "Number of gen jets",
    "ytitle":   "Events",
}

hists["h_mH_gen_all"] = {
    "output": "h_mH_gen_all",
    "logy": False,
    "stack": False,
    "xtitle": " Higgs mass from gen particles (all gen jets)",
    "ytitle": "Events",

}

hists["h_mH_reco_all"] = {
    "output": "h_mH_reco_all",
    "logy": False,
    "stack": False,
    "xtitle": "Reconstructed Higgs mass (all reco jets)",
    "ytitle": "Events",
}

hists["h_mH_all_stable_part"] = {
    "output": "h_mH_all_stable_part",
    "logy": False,
    "stack": False,
    "xtitle": "invariant mass of all particles",
    "ytitle": "Events"
}

hists["h_E_all_reco_jets"] = {
    "output": "h_E_all_reco_jets",
    "logy": False,
    "stack": False,
    "xtitle": "reco jet E",
    "ytitle": "Events"

}

hists["h_E_all_gen_jets"] = {
    "output": "h_E_all_genjets",
    "logy": False,
    "stack": False,
    "xtitle": "gen jet E",
    "ytitle": "Events"
}
'''
hists["h_calo_hit_energy"] = {
    "output": "h_calo_hit_energy",
    "logy": False,
    "stack": False,
    "xtitle": "Calo hit energy",
    "ytitle": "Hits"
}
'''
hists["h_mH_reco_core"] = {
    "output": "h_mH_reco_core",
    "logy": True,
    "stack": False,
    "ymax": 1e4,
    "xtitle": "mH (reco) [GeV]",
    "ytitle": "Events",
}

hists["h_mH_gen_core"] = {
    "output": "h_mH_gen_core",
    "logy": True,
    "stack": False,
    #"ymax": 150000, # Can I set this to auto?
    "xtitle": "mH (gen) [GeV]",
    "ytitle": "Events",
}

hists["h_mH_stable_gt_particles"] = {
    "output": "h_mH_stable_gt_particles",
    "logy": True,
    "stack": False,
    "xtitle": "Higgs mass from stable GT particles",
    "ytitle": "Events"
}

hists["h_mH_reco_particles_matched"] = {
    "output": "h_mH_reco_particles_matched",
    "logy": True,
    "stack": False,
    "xtitle": "Higgs mass from reco particles matched from Higgs",
    "ytitle": "Events"
}

hists["h_mH_MC_part"] =  {
    "output": "h_mH_MC_part",
    "logy": True,
    "stack": False,
    "xtitle": "Higgs mass from MC partons",
    "ytitle": "Events"
}

'''
hists["h_frac_E_charged"] = {
    "output": "h_frac_E_charged",
    "logy": False,
    "xtitle": "Fraction of reco. E in event by charged",
    "ytitle": "Events"
}

hists["h_E_charged"] = {
    "output": "h_E_charged",
    "logy": False,
    "xtitle": "E of charged PFCand",
    "ytitle": "Number of PFCands"
}

hists["h_E_reco_over_true_Charged"] = {
    "output":   "h_E_reco_over_true_Charged",
    "logy":     False,
    "stack":    False,
    "xtitle":   "E_reco/E_true (Charged PFCands)",
    "ytitle":   "Number of PFCands",
    "xmin": 0.9,
    "xmax": 1.1,
    "ymax": 5e6
}
'''
