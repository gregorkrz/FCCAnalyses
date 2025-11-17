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

hists["gen_muon_energies"] = {
    "output":   "gen_muon_energies",
    "logy":     False,
    "stack":    False,
    "ymin": 0,
    "xtitle":   "E_reco/E_true (deltaR matching)",
    "ytitle":   "Events",
}
