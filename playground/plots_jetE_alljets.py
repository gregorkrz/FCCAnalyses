#fccsw tutorial
# fccanalysis plots plots_jetE_alljets.py

import ROOT

# global parameters
intLumi        = 1.
intLumiLabel   = ""
#ana_tex        = 'All matched jets using deltaR matching'
ana_tex=""
delphesVersion = '3.4.2'
energy         = 240.0
collider       = 'FCC-ee'
formats        = ['png','pdf']

outdir         = '../../idea_fullsim/fast_sim/histograms_view/comparison_multiple_jets_allJets'
inputDir       = '../../idea_fullsim/fast_sim/histograms'

plotStatUnc    = True

colors = {}
colors['WW'] = ROOT.kRed
colors['ZH'] = ROOT.kBlue+1
colors["ZHll"] = ROOT.kGreen+2
colors["ZHvv"] = ROOT.kCyan+2
colors["ZH6jet"] = ROOT.kMagenta


#procs = {}
#procs['signal'] = {'ZH':['wzp6_ee_mumuH_ecm240']}
#procs['backgrounds'] =  {'WW':['p8_ee_WW_ecm240'], 'ZZ':['p8_ee_ZZ_ecm240']}

procs = {}
procs["signal"] = {
    "WW": ["p8_ee_WW_ecm365_fullhad"],
    "ZH": ["p8_ee_ZH_qqbb_ecm365"],
    "ZHll": ["p8_ee_ZH_llbb_ecm365"],
    "ZH6jet": ["p8_ee_ZH_6jet_ecm365"],
    "ZHvv": ["p8_ee_ZH_vvbb_ecm365"],
}

procs["backgrounds"] = {}
legend = {}
legend['WW'] = "ee->WW->qqqq"
legend["ZH"] = "ee->ZH->qqbb"
legend["ZHll"] = "ee->ZH->llbb"
legend["ZH6jet"] = "ee->Z(qq)H(WW->qqqq)"
legend["ZHvv"] = "ee->ZH->vvbb"
#legend['n_truth'] = 'N Truth Particles'

hists = {}

hists["h_fancy"] = {
    "output":   "jet_E_deltaR_matching",
    "logy":     False,
    "stack":    False,
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

hists["h_E_all_jets"] = {
    "output":   "E_of_all_reco_jets",
    "logy":     False,
    "stack":    False,
    #"rebin":    100,
    #"xmin":     120,
    #"xmax":     140,
    ##"ymin":     0,
    "xtitle":   "E of reco jets",
    "ytitle":   "Events",
}

hists["h_E_all_genjets"] = {
    "output":   "E_of_all_gen_jets",
    "logy":     False,
    "stack":    False,
    #"rebin":    100,
    #"xmin":     120,
    #"xmax":     140,
    ##"ymin":     0,
    "xtitle":   "E of gen jets",
    "ytitle":   "Events",
}
