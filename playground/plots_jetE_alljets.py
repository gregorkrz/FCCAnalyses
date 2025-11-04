#fccsw tutorial
# fccanalysis plots plots_jetE_alljets.py

import ROOT
import os

assert "INPUT_DIR" in os.environ # To make sure we are taking the right input dir and folder name
assert "FOLDER_NAME" in os.environ

# Global parameters
intLumi        = 1.
intLumiLabel   = ""
#ana_tex        = 'All matched jets using deltaR matching'
ana_tex=""
delphesVersion = '3.4.2'
energy         = 240.0
collider       = 'FCC-ee'
formats        = ['png','pdf']

#outdir         = '../../idea_fullsim/fast_sim/histograms_view/comparison_multiple_jets_allJets_greedyMatching'
#inputDir       = '../../idea_fullsim/fast_sim/histograms/greedy_matching'

outdir         = '../../idea_fullsim/fast_sim/Histograms_ECM240/{}'.format(os.environ["FOLDER_NAME"])
inputDir       = '../../idea_fullsim/fast_sim/Histograms_ECM240/{}'.format(os.environ["FOLDER_NAME"])

plotStatUnc    = True

colors = {}
colors['WW'] = ROOT.kRed
colors['ZH'] = ROOT.kBlue+1
#colors["ZHll"] = ROOT.kGreen+2
colors["ZHvv"] = ROOT.kCyan+2
colors["ZH6jet"] = ROOT.kMagenta
colors["ZHbbbb"] = ROOT.kOrange+7
colors["ZHvvgg"] = ROOT.kGray+1

#procs = {}
#procs['signal'] = {'ZH':['wzp6_ee_mumuH_ecm240']}
#procs['backgrounds'] =  {'WW':['p8_ee_WW_ecm240'], 'ZZ':['p8_ee_ZZ_ecm240']}

procs = {}

procs["signal"] = {
    "ZH": ["p8_ee_ZH_qqbb_ecm240"],
    "ZH6jet": ["p8_ee_ZH_6jet_ecm240"],
    "ZHvv": ["p8_ee_ZH_vvbb_ecm240"],
    "ZHbbbb": ["p8_ee_ZH_bbbb_ecm240"],
    "ZHvvgg": ["p8_ee_ZH_vvgg_ecm240"],
}

procs["backgrounds"] = {}

legend = {}
legend['WW'] = "ee->WW->qqqq"
legend["ZH"] = "ee->ZH->qqbb"
#legend["ZHll"] = "ee->ZH->llbb"
legend["ZH6jet"] = "ee->Z(qq)H(WW->qqqq)"
legend["ZHvv"] = "ee->ZH->vvbb"
legend["ZHbbbb"] = "ee->ZH->bbbb"
legend["ZHvvgg"] = "ee->ZH->vvgg"
#legend['n_truth'] = 'N Truth Particles'

hists = {}

hists["h_fancy"] = {
    "output":   "jet_E_deltaR_matching",
    "logy":     False,
    "stack":    False,
    "ymin": 0,
    #"rebin":    100,
    #"xmin":     120,
    #"xmax":     140,
    ##"ymin":     0,
    #"ymax":     10000,
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
    "ymax": 150000,
    "xtitle": "Reconstructed Higgs mass (all matched jets)",
    "ytitle": "Events",
}

hists["h_mH_gen"] = {
    "output": "h_mH_gen",
    "logy": False,
    "stack": False,
    "ymax": 150000,
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
    "ymax": 1e5,
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
