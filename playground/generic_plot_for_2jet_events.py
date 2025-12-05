from Colors import PROCESS_COLORS, HUMAN_READABLE_PROCESS_NAMES, LINE_STYLES
# Dictionary should map a folder name to a human-readable method name
import matplotlib.pyplot as plt
import numpy as np
import pickle
from ROOT import TFile

bins_E = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

def root_file_get_hist_and_edges(root_file, hist_name, rebin_factor=1):
    # assume root file is a text, open it
    print("Reading histogram", hist_name, "from", root_file)
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
    #"PFDurham_ISR_FullyMatched_MatchRecoJets": "PFJets + Ideal Matching",
    "PFDurham_ISR_FullyMatched": "After applying filter",
    "PFDurham_ISR_NoFilter": "Before applying filter",
    #"CaloJetDurham_ISR_FullyMatched": "CaloJets"
}

processes_twojet = [
    "p8_ee_ZH_vvbb_ecm240",
    "p8_ee_ZH_vvgg_ecm240",
    "p8_ee_ZH_vvqq_ecm240"
]

histograms_folder = "Histograms_ECM240_20251114_6M_Fix0312_ME"

fig, ax = plt.subplots(3, 2, figsize=(12, 12))
# Do the two jet processes, and for each one plot gen on the left (h_mH_all_stable_part, h_mH_all_stable_part_BeforeFiltering) and reco on the right (hist_calo_hist_E, h_inv_mass_all_reco_particles_BeforeFiltering)
line_styles = {
    "PFDurham_ISR_FullyMatched": "-",
    "PFDurham_ISR_NoFilter": "--",
    "CaloJetDurham_ISR_FullyMatched": ":",
    "PFDurham_ISR_FullyMatched_MatchRecoJets": "--"
}


for i, process in enumerate(processes_twojet):
    for j, (method_folder, method_name) in enumerate(method_dict.items()):
        root_file = f"../../idea_fullsim/fast_sim/{histograms_folder}/{method_folder}/{process}.root"
        def plot_hist(hist_name_gen, hist_name_reco, label):
            y_gen, edges_gen = root_file_get_hist_and_edges(root_file, hist_name_gen)
            y_reco, edges_reco = root_file_get_hist_and_edges(root_file, hist_name_reco)
            bin_centers_gen = 0.5 * (edges_gen[:-1] + edges_gen[1:])
            bin_centers_reco = 0.5 * (edges_reco[:-1] + edges_reco[1:])
            ax[i, 0].step(bin_centers_gen, y_gen, where='mid',
                          label=label,
                          #color=PROCESS_COLORS[process],
                          linestyle=line_styles[method_folder], alpha=0.7)
            ax[i, 1].step(bin_centers_reco, y_reco, where='mid',
                          label=label,
                          #color=PROCESS_COLORS[process],
                          linestyle=line_styles[method_folder], alpha=0.7)
        plot_hist("h_mH_all_stable_part_BeforeFiltering", "h_inv_mass_all_reco_particles_BeforeFiltering", method_name)
        #plot_hist("h_mH_all_stable_part", "hist_calo_hist_E", "After applying filter")
    ax[i, 0].set_title(f"{HUMAN_READABLE_PROCESS_NAMES[process]} - MC Gen Particles")
    ax[i, 1].set_title(f"{HUMAN_READABLE_PROCESS_NAMES[process]} - Reconstructed Particles")
    ax[i, 0].set_xlabel("Invariant Mass (GeV)")
    ax[i, 1].set_xlabel("Invariant Mass (GeV)")
    ax[i, 0].set_ylabel("Events")
    ax[i, 1].set_ylabel("Events")
    ax[i, 0].legend()
    ax[i, 1].legend()
    ax[i, 0].set_xlim([100, 150])
    ax[i, 1].set_xlim([100, 150])

fig.tight_layout()
fig.savefig(f"../../idea_fullsim/fast_sim/{histograms_folder}/2jet_invariant_mass_comparison.pdf")
#fig.show()
