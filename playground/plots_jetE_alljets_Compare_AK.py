# fccanalysis plots plots_jetE_alljets_Compare_AK.py
import ROOT
import os


#assert "INPUT_DIR" in os.environ # To make sure we are taking the right input dir and folder name
#assert "FOLDER_NAME" in os.environ

#FOLDER_NAME in [p8_ee_ZH_6jet_ecm240, p8_ee_ZH_bbbb_ecm240, p8_ee_ZH_qqbb_ecm240, p8_ee_ZH_vvbb_ecm240, p8_ee_ZH_vvgg_ecm240]

assert "FOLDER_NAME" in os.environ
assert "HISTOGRAMS_FOLDER_NAME" in os.environ # default: Histograms_ECM240_AK_organized_NoISR



intLumi        = 1.
intLumiLabel   = ""
#ana_tex        = 'All matched jets using deltaR matching'
ana_tex = os.environ["FOLDER_NAME"]
delphesVersion = '3.4.2'
energy         = 240.0
collider       = 'FCC-ee'
formats        = ['png', 'pdf']

outdir         = '../../idea_fullsim/fast_sim/{}/{}'.format(os.environ["HISTOGRAMS_FOLDER_NAME"], os.environ["FOLDER_NAME"])
inputDir       = '../../idea_fullsim/fast_sim/{}/{}'.format(os.environ["HISTOGRAMS_FOLDER_NAME"], os.environ["FOLDER_NAME"])

plotStatUnc    = True

colors = {}
color_presets = [ROOT.kRed, ROOT.kBlue+1, ROOT.kCyan+2, ROOT.kMagenta, ROOT.kOrange+7, ROOT.kGray+1]
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
    "ymax": 120000,
    "xtitle": "Reconstructed Higgs mass (all matched jets)",
    "ytitle": "Events",
}

hists["h_mH_gen"] = {
    "output": "h_mH_gen",
    "logy": False,
    "stack": False,
    "ymax": 120000,
    "xtitle": " Higgs mass from gen particles (all matched jets)",
    "ytitle": "Events",
}

