#fccsw tutorial
# fccanalysis plots plots_jetE.py

import ROOT

# global parameters
intLumi        = 1.
intLumiLabel   = ""
ana_tex        = 'e+e- -> WW --> qqqq'
delphesVersion = '3.4.2'
energy         = 240.0
collider       = 'FCC-ee'
formats        = ['png','pdf']

outdir         = '../../idea_fullsim/fast_sim/histograms_view/p8_ee_WW_ecm365_fullhad'
inputDir       = '../../idea_fullsim/fast_sim/histograms/p8_ee_WW_ecm365_fullhad'

plotStatUnc    = True

colors = {}
colors['n_reco'] = ROOT.kRed
#colors['n_truth'] = ROOT.kBlue+1

#procs = {}
#procs['signal'] = {'ZH':['wzp6_ee_mumuH_ecm240']}
#procs['backgrounds'] =  {'WW':['p8_ee_WW_ecm240'], 'ZZ':['p8_ee_ZZ_ecm240']}

procs = {}
procs["signal"] = {"n_reco": ["ee_WW_ecm240_samples"]}

procs["backgrounds"] = {}
legend = {}
legend['n_reco'] = 'N PFCands'
#legend['n_truth'] = 'N Truth Particles'

hists = {}

hists["n_reco"] = {
    "output":   "n_reco",
    "logy":     False,
    "stack":    False,
    #"rebin":    100,
    #"xmin":     120,
    #"xmax":     140,
    ##"ymin":     0,
    #"ymax":     2500,
    "xtitle":   "N",
    "ytitle":   "Events",
}
