#fccsw tutorial
# fccanalysis plots plots_jetE.py

import ROOT

# global parameters
intLumi        = 1.
intLumiLabel   = ""
ana_tex        = 'Highest-energy jets'
delphesVersion = '3.4.2'
energy         = 240.0
collider       = 'FCC-ee'
formats        = ['png','pdf']

outdir         = '../../idea_fullsim/fast_sim/histograms_view/p8_ee_WW_ecm365_fullhad'
inputDir       = '../../idea_fullsim/fast_sim/histograms'

plotStatUnc    = True

colors = {}
colors['WW'] = ROOT.kRed
colors['ZH'] = ROOT.kBlue+1




#procs = {}
#procs['signal'] = {'ZH':['wzp6_ee_mumuH_ecm240']}
#procs['backgrounds'] =  {'WW':['p8_ee_WW_ecm240'], 'ZZ':['p8_ee_ZZ_ecm240']}

procs = {}
procs["signal"] = {"WW": ["p8_ee_WW_ecm365_fullhad"], "ZH": ["p8_ee_ZH_qqbb_ecm365"]}

procs["backgrounds"] = {}
legend = {}
legend['WW'] = 'ee->WW->qqqq'
legend["ZH"] = "ee->ZH->qqbb"

#legend['n_truth'] = 'N Truth Particles'

hists = {}


hists["h_E0"] = {
    "output":   "E",
    "logy":     True,
    "stack":    False,
    #"rebin":    100,
    #"xmin":     120,
    #"xmax":     140,
    ##"ymin":     0,
    #"ymax":     2500,
    "xtitle":   "E_reco/E_true (Durham N=4 jets, naive matching)",
    "ytitle":   "Events",
}


hists["h_fancy_E0"] = {
    "output":   "jet_E_deltaR_matching",
    "logy":     True,
    "stack":    False,
    #"rebin":    100,
    #"xmin":     120,
    #"xmax":     140,
    ##"ymin":     0,
    #"ymax":     2500,
    "xtitle":   "E_reco/E_true (deltaR matching)",
    "ytitle":   "Events",
}

