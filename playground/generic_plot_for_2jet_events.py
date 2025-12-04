from Colors import PROCESS_COLORS, HUMAN_READABLE_PROCESS_NAMES, LINE_STYLES
# Dictionary should map a folder name to a human-readable method name
import matplotlib.pyplot as plt
import numpy as np
import pickle
from ROOT import TFile

bins_E = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def root_file_get_hist_and_edges(root_file, hist_name, rebin_factor=1):
    # assume root file is a text, open it
    root_file_obj = TFile.Open(root_file, "READ")
    h = root_file_obj.Get(hist_name)
    if not h:
        raise Exception(f"Warning: histogram {hist_name} not found")
    nb = h.GetNbinsX()
    edges = np.array([h.GetXaxis().GetBinLowEdge(1)] +
                     [h.GetXaxis().GetBinUpEdge(i) for i in range(1, nb + 1)])
    y = np.array([h.GetBinContent(i) for i in range(1, nb + 1)], dtype=float)
    assert len(edges) == len(y) + 1
    return y, edges

method_dict = {
    "PFDurham_ISR_FullyMatched_MatchRecoJets": "PFJets + Ideal Matching",
    "PFDurham_ISR_FullyMatched": "PFJets",
    "CaloJetDurham_ISR_FullyMatched": "CaloJets"
}

processes_twojet = [
    "p8_ee_ZH_vvbb_ecm240",
    "p8_ee_ZH_vvgg_ecm240",
    "p8_ee_ZH_vvqq_ecm240"
]

histograms_folder = "Histograms_ECM240_20251114_6M_Fix0112"


fig, ax = plt.subplots(figsize=(8, 6))