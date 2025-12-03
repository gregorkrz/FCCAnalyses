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

process_for_detailed_bins_plots = [
    "p8_ee_ZH_vvbb_ecm240",
    "p8_ee_ZH_vvgg_ecm240",
    "p8_ee_ZH_vvqq_ecm240"
]

histograms_folder = "Histograms_ECM240_20251114_6M_Fix0112"


# Resolution plots, PF vs PF+Ideal matching
fig, ax = plt.subplots(10, figsize=(4, 19))
fig_mH, ax_mH = plt.subplots(2, 2, figsize=(8, 6))
for method in ["PFDurham_ISR_FullyMatched", "PFDurham_ISR_FullyMatched_MatchRecoJets"]:
    method_color = {
        "PFDurham_ISR_FullyMatched": "blue",
        "PFDurham_ISR_FullyMatched_MatchRecoJets": "orange"
    }
    method_linestyle = { # For the m_H plots
        "PFDurham_ISR_FullyMatched": "-",
        "PFDurham_ISR_FullyMatched_MatchRecoJets": "--"
    }
    f = pickle.load(open(f"../../idea_fullsim/fast_sim/{histograms_folder}/{method}/energy_fit_params_per_process.pkl", "rb"))
    f_mH = pickle.load(open(f"../../idea_fullsim/fast_sim/{histograms_folder}/{method}/Higgs_mass_histograms_data.pkl", "rb"))
    for i, process in enumerate(PROCESS_COLORS.keys()):
        if process not in f:
            continue
        color = PROCESS_COLORS[process]
        label = HUMAN_READABLE_PROCESS_NAMES[process]
        linestyle = "--"
        cp =  f[process]["std68_higherRes"]
        xs, ys = cp[2], cp[3]
        x_pts, y_pts = cp[4], cp[5]
        ax[i].plot(xs, ys, label=f"{method_dict[method]} A={round(cp[0][0], 2)} B={round(cp[0][1], 2)}", color=method_color[method], linestyle=linestyle)
        ax[i].plot(x_pts, y_pts, 'x', color=method_color[method], markersize=4)
        ax[i].set_ylabel("$\sigma_E / E_{true}$")
        ax[i].set_title(label + r" ($\frac{A}{\sqrt{E}}$ ⊕ $B$)")
        Higgs_x, Higgs_y = f_mH[process]["x_vals_reco"], f_mH[process]["y_vals_reco"]
        if LINE_STYLES[process] == ":":
            ax_mH[0, 0].plot(Higgs_x, Higgs_y, label=f"{label} ({method_dict[method]})", color=PROCESS_COLORS[process], linestyle=method_linestyle[method])
            ax_mH[0, 1].plot(Higgs_x, Higgs_y, label=f"{label} ({method_dict[method]})", color=PROCESS_COLORS[process], linestyle=method_linestyle[method])
            ax_mH[0, 0].set_title(r"H → Light-flavour jets")
        elif LINE_STYLES[process] == "-":
            ax_mH[1, 0].plot(Higgs_x, Higgs_y, label=f"{label} ({method_dict[method]})", color=PROCESS_COLORS[process], linestyle=method_linestyle[method])
            ax_mH[1, 1].plot(Higgs_x, Higgs_y, label=f"{label} ({method_dict[method]})", color=PROCESS_COLORS[process], linestyle=method_linestyle[method])
            ax_mH[1, 0].set_title(r"H → Light-flavour and b-jets")
for i in range(len(ax)):
    ax[i].legend(fontsize=6.5)
    ax[i].grid()
for i in range(len(ax_mH)):
    for j in range(len(ax_mH[i])):
        ax_mH[i, j].set_xlabel("$m_H$ (reco.) [GeV]")
        ax_mH[i, j].set_xlabel("$m_H$ (reco.) [GeV]")
        ax_mH[i, j].grid()
    ax_mH[i, 1].set_xlim(100, 140)
    ax_mH[i, 0].legend(title="l ∈ {u, d, s}; q ∈ {u, d, s, c, b}", fontsize=7.5, title_fontsize=8)
fig.tight_layout()
fig_path = f"../../idea_fullsim/fast_sim/{histograms_folder}/JER_comparison_PF_and_ideal_matching.pdf"

fig_mH.tight_layout()
fig_mH_path = f"../../idea_fullsim/fast_sim/{histograms_folder}/Higgs_mass_comparison_PF_and_ideal_matching.pdf"
print("Saving figure to", fig_mH_path)
print("Saving figure to", fig_path)
fig.savefig(fig_path)
fig_mH.savefig(fig_mH_path)

####################################
# Resolution plots, PF vs PF+Ideal matching
fig, ax = plt.subplots(10, figsize=(4, 19))
fig_mH, ax_mH = plt.subplots(2, 2, figsize=(8, 6))

for prefix in ["neutral", "charged", "photons", "higherRes"]:
    figs_processes, axs_processes = {}, {}
    histogram_limits = [[0, 2], [0.3, 1.4], [0.75, 1.15], [0.95, 1.05]]
    for method in ["PFDurham_ISR_FullyMatched",
                   "CaloJetDurham_ISR_FullyMatched",
                   "PFDurham_ISR_FullyMatched_MatchRecoJets",]:
        method_color = {
            "PFDurham_ISR_FullyMatched": "blue",
            "CaloJetDurham_ISR_FullyMatched": "green",
            "PFDurham_ISR_FullyMatched_MatchRecoJets": "orange"
        }
        method_linestyle = { # For the m_H plots
            "PFDurham_ISR_FullyMatched": "-",
            "CaloJetDurham_ISR_FullyMatched": "--",
            "PFDurham_ISR_FullyMatched_MatchRecoJets": ":"
        }
        f = pickle.load(open(f"../../idea_fullsim/fast_sim/{histograms_folder}/{method}/energy_fit_params_per_process.pkl", "rb"))
        f_mH = pickle.load(open(f"../../idea_fullsim/fast_sim/{histograms_folder}/{method}/Higgs_mass_histograms_data.pkl", "rb"))
        for i, process in enumerate(PROCESS_COLORS.keys()):
            if process not in f:
                continue
            color = PROCESS_COLORS[process]
            label = HUMAN_READABLE_PROCESS_NAMES[process]
            if process in process_for_detailed_bins_plots:
                # Make detailed binning plots for these processes
                if method == "CaloJetDurham_ISR_FullyMatched":
                    binning_metadata = f[process]["std68_higherRes"]
                else:
                    binning_metadata = f[process]["std68_" + prefix]
                hist_names = binning_metadata[7]
                if process not in figs_processes:
                    n_bins = len(binning_metadata[6])

                    figs_processes[process], axs_processes[process] = plt.subplots(n_bins, 4, figsize=(15, 4 * n_bins))
                for b in range(n_bins):
                    low_point, high_point, mpv_point = binning_metadata[6][b]
                    y_hist, edges = root_file_get_hist_and_edges(
                        f"../../idea_fullsim/fast_sim/{histograms_folder}/{method}/{process}.root",
                        hist_names[b])
                    # Now plot this in each column of bin b.
                    # Each plot has different x axis limits (historam_limits)
                    for col in range(len(histogram_limits)):
                        mask = (edges[:-1] >= histogram_limits[col][0]) & (edges[1:] <= histogram_limits[col][1])
                        # Do a step histogram using mask
                        axs_processes[process][b, col].step(edges[:-1][mask], y_hist[mask], where='post',
                                                            color=method_color[method], label=f"{method_dict[method]}")
                        # Also plot low, high (--) and MPV (|) lines vertically, if they fall within the histogram limits
                        # These vertical lines shouldn't have any legend entries and should be in the method_color
                        if histogram_limits[col][0] <= low_point <= histogram_limits[col][1]:
                            axs_processes[process][b, col].axvline(low_point, color=method_color[method], linestyle='--')
                        if histogram_limits[col][0] <= high_point <= histogram_limits[col][1]:
                            axs_processes[process][b, col].axvline(high_point, color=method_color[method], linestyle='--')
                        if histogram_limits[col][0] <= mpv_point <= histogram_limits[col][1]:
                            axs_processes[process][b, col].axvline(mpv_point, color=method_color[method], linestyle=':')
                        axs_processes[process][b, col].set_title(f"[{bins_E[b]}, {bins_E[b+1]}] GeV")
                        axs_processes[process][b, col].legend()
                        axs_processes[process][b, col].grid(True)
            linestyle = "--"
            if method == "CaloJetDurham_ISR_FullyMatched":
                cp = f[process]["std68_higherRes"]
            else:
                print(f[process].keys())
                cp = f[process]["std68_" + prefix]
            xs, ys = cp[2], cp[3]
            x_pts, y_pts = cp[4], cp[5]
            if prefix == "higherRes":
                ax[i].plot(xs, ys, label=f"{method_dict[method]} A={round(cp[0][0], 2)} B={round(cp[0][1], 2)}", color=method_color[method], linestyle=linestyle)
                ax[i].plot(x_pts, y_pts, 'x', color=method_color[method], markersize=4)
                ax[i].set_ylabel("$\sigma_E / E_{true}$")
                ax[i].set_title(label + r" ($\frac{A}{\sqrt{E}}$ ⊕ $B$)")
            Higgs_x, Higgs_y = f_mH[process]["x_vals_reco"], f_mH[process]["y_vals_reco"]
            if prefix == "higherRes":
                if LINE_STYLES[process] == ":":
                    ax_mH[0, 0].plot(Higgs_x, Higgs_y, label=f"{label} ({method_dict[method]})", color=PROCESS_COLORS[process], linestyle=method_linestyle[method])
                    ax_mH[0, 1].plot(Higgs_x, Higgs_y, label=f"{label} ({method_dict[method]})", color=PROCESS_COLORS[process], linestyle=method_linestyle[method])
                    ax_mH[0, 0].set_title(r"H → Light-flavour jets")
                elif LINE_STYLES[process] == "-":
                    ax_mH[1, 0].plot(Higgs_x, Higgs_y, label=f"{label} ({method_dict[method]})", color=PROCESS_COLORS[process], linestyle=method_linestyle[method])
                    ax_mH[1, 1].plot(Higgs_x, Higgs_y, label=f"{label} ({method_dict[method]})", color=PROCESS_COLORS[process], linestyle=method_linestyle[method])
                    ax_mH[1, 0].set_title(r"H → Light-flavour and b-jets")
    for process in figs_processes:
        figs_processes[process].tight_layout()
        fig_process_path = f"../../idea_fullsim/fast_sim/{histograms_folder}/Detailed_JER_histograms_{process}{prefix}.pdf"
        print("Saving figure to", fig_process_path)
        figs_processes[process].savefig(fig_process_path)

for i in range(len(ax_mH)):
    for j in range(len(ax_mH[i])):
        ax_mH[i, j].set_xlabel("$m_H$ (reco.) [GeV]")
        ax_mH[i, j].set_xlabel("$m_H$ (reco.) [GeV]")
        ax_mH[i, j].grid()
    ax_mH[i, 1].set_xlim(100, 140)
    ax_mH[i, 0].legend(title="l ∈ {u, d, s}; q ∈ {u, d, s, c, b}", fontsize=7.5, title_fontsize=8)
for i in range(len(ax)):
    ax[i].legend(fontsize=6.5)
    ax[i].grid()
fig.tight_layout()
fig_path = f"../../idea_fullsim/fast_sim/{histograms_folder}/JER_comparison_PF_and_CaloJets.pdf"
print("Saving figure to", fig_path)
fig.savefig(fig_path)
fig_mH.tight_layout()
fig_mH_path = f"../../idea_fullsim/fast_sim/{histograms_folder}/Higgs_mass_comparison_PF_and_CaloJets.pdf"
print("Saving figure to", fig_mH_path)

fig_mH.savefig(fig_mH_path)

for prefix in ["neutral", "charged", "photons"]:
    ###########################################
    # Resolution plots, calo jets vs neutral part of PF jets
    fig, ax = plt.subplots(10, figsize=(4, 19))
    for method in ["PFDurham_ISR_FullyMatched", "PFDurham_ISR_FullyMatched_MatchRecoJets", "CaloJetDurham_ISR_FullyMatched"]:
        method_color = {
            "PFDurham_ISR_FullyMatched": "blue",
            "CaloJetDurham_ISR_FullyMatched": "green",
            "PFDurham_ISR_FullyMatched_MatchRecoJets": "orange"
        }
        method_linestyle = { # For the $m_H$ plots
            "PFDurham_ISR_FullyMatched": "-",
            "CaloJetDurham_ISR_FullyMatched": "--",
            "PFDurham_ISR_FullyMatched_MatchRecoJets": ":"
        }
        method_text = {
            "PFDurham_ISR_FullyMatched": "PF jets, {} part".format(prefix),
            "CaloJetDurham_ISR_FullyMatched": "Calo jets",
            "PFDurham_ISR_FullyMatched_MatchRecoJets": "PF jets, {} part + ideal matching".format(prefix)
        }
        f = pickle.load(open(f"../../idea_fullsim/fast_sim/{histograms_folder}/{method}/energy_fit_params_per_process.pkl", "rb"))
        f_mH = pickle.load(open(f"../../idea_fullsim/fast_sim/{histograms_folder}/{method}/Higgs_mass_histograms_data.pkl", "rb"))
        for i, process in enumerate(PROCESS_COLORS.keys()):
            if process not in f:
                continue
            color = PROCESS_COLORS[process]
            label = HUMAN_READABLE_PROCESS_NAMES[process]
            linestyle = "--"
            if method == "CaloJetDurham_ISR_FullyMatched":
                if "std68_higherRes" in f[process]:
                    cp = f[process]["std68_higherRes"]
                else:
                    cp = f[process]["std68"]
            else:
                cp = f[process]["std68_" + prefix]
            xs, ys = cp[2], cp[3]
            x_pts, y_pts = cp[4], cp[5]
            ax[i].plot(xs, ys, label=f"{method_text[method]} A={round(cp[0][0], 2)} B={round(cp[0][1], 2)}", color=method_color[method], linestyle=linestyle)
            ax[i].plot(x_pts, y_pts, 'x', color=method_color[method], markersize=4)
            ax[i].set_ylabel("$\sigma_E / E_{true}$")
            ax[i].set_title(label + r" ($\frac{A}{\sqrt{E}}$ ⊕ $B$)")
            Higgs_x, Higgs_y = f_mH[process]["x_vals_reco"], f_mH[process]["y_vals_reco"]
    for i in range(len(ax)):
        ax[i].legend(fontsize=6.5)
        ax[i].grid()
    fig.tight_layout()
    fig_path = f"../../idea_fullsim/fast_sim/{histograms_folder}/JER_comparison_{prefix}_PF_vs_CaloJets.pdf"
    print("Saving figure to", fig_path)
    fig.savefig(fig_path)
