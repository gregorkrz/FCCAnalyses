from Colors import PROCESS_COLORS, HUMAN_READABLE_PROCESS_NAMES, LINE_STYLES
# Dictionary should map a folder name to a human-readable method name
import matplotlib.pyplot as plt
import numpy as np
import pickle

method_dict = {
    "PFDurham_ISR_FullyMatched_MatchRecoJets": "PFJets + Ideal Matching",
    "PFDurham_ISR_FullyMatched": "PFJets",
    "CaloJetDurham_ISR_FullyMatched": "CaloJets"
}

histograms_folder = "Histograms_ECM240_20251114_6M"

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
for method in ["PFDurham_ISR_FullyMatched", "CaloJetDurham_ISR_FullyMatched"]:
    method_color = {
        "PFDurham_ISR_FullyMatched": "blue",
        "CaloJetDurham_ISR_FullyMatched": "green"
    }
    method_linestyle = { # For the m_H plots
        "PFDurham_ISR_FullyMatched": "-",
        "CaloJetDurham_ISR_FullyMatched": "--"
    }
    f = pickle.load(open(f"../../idea_fullsim/fast_sim/{histograms_folder}/{method}/energy_fit_params_per_process.pkl", "rb"))
    f_mH = pickle.load(open(f"../../idea_fullsim/fast_sim/{histograms_folder}/{method}/Higgs_mass_histograms_data.pkl", "rb"))
    for i, process in enumerate(PROCESS_COLORS.keys()):
        if process not in f:
            continue
        color = PROCESS_COLORS[process]
        label = HUMAN_READABLE_PROCESS_NAMES[process]
        linestyle = "--"
        if "std68_higherRes" in f[process]:
            cp =  f[process]["std68_higherRes"]
        else:
            cp =  f[process]["std68"]
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
fig_path = f"../../idea_fullsim/fast_sim/{histograms_folder}/JER_comparison_PF_and_CaloJets.pdf"

fig_mH.tight_layout()
fig_mH_path = f"../../idea_fullsim/fast_sim/{histograms_folder}/Higgs_mass_comparison_PF_and_CaloJets.pdf"
print("Saving figure to", fig_mH_path)
print("Saving figure to", fig_path)
fig.savefig(fig_path)
fig_mH.savefig(fig_mH_path)


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
        "PFDurham_ISR_FullyMatched": "PF jets, neutral part",
        "CaloJetDurham_ISR_FullyMatched": "Calo jets",
        "PFDurham_ISR_FullyMatched_MatchRecoJets": "PF jets, neutral part + ideal matching"
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
            cp = f[process]["std68_neutral"]
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
fig_path = f"../../idea_fullsim/fast_sim/{histograms_folder}/JER_comparison_Neutral_PF_vs_CaloJets.pdf"
print("Saving figure to", fig_path)
fig.savefig(fig_path)

