# Plotting without the FCCsw
# python3 plotting.py

import ROOT
import numpy as np
import matplotlib.pyplot as plt
ROOT.gROOT.SetBatch(True)  # don’t pop up ROOT canvases

# open your file
f = ROOT.TFile.Open("../../idea_fullsim/fast_sim/histograms/p8_ee_WW_ecm365_fullhad_AK.root")
# Get list of columns and print their titles

print("---------")

# Mapping of histogram name → legend label
hists = {
    "h_ratio_2":  "AK2",
    "h_ratio_4":  "AK4",
    "h_ratio_5":  "AK5",
    "h_ratio_8":  "AK8",
    "h_ratio_10": "AK10",
    "h_ratio_15": "AK15",
    "h_ratio_20": "AK20",
    "h_ratio_30": "AK30"
}

hists_unmatched = {
    "h_unmatched_2":  "AK2",
    "h_unmatched_4":  "AK4",
    "h_unmatched_5":  "AK5",
    "h_unmatched_8":  "AK8",
    "h_unmatched_10": "AK10",
    "h_unmatched_15": "AK15",
    "h_unmatched_20": "AK20",
    "h_unmatched_30": "AK30"

}
def hist1_to_np(h):
    nb = h.GetNbinsX()
    edges = np.array([h.GetXaxis().GetBinLowEdge(1)] +
                     [h.GetXaxis().GetBinUpEdge(i) for i in range(1, nb+1)])
    y = np.array([h.GetBinContent(i) for i in range(1, nb+1)], dtype=float)
    return edges, y

plt.figure(figsize=(8,6))

for name, label in hists.items():
    h = f.Get(name)
    if not h:
        print(f"Warning: histogram {name} not found")
        continue
    edges, y = hist1_to_np(h)
    plt.step(edges, np.r_[y, y[-1]], where="post", label=label)


plt.xlabel("x")
plt.ylabel("Entries")
plt.title("Comparison of ratios")
plt.legend()
plt.grid(True, alpha=0.3)
# log y scale
plt.yscale("log")
plt.tight_layout()
plt.savefig("../../idea_fullsim/fast_sim/histograms_view/p8_ee_WW_ecm365_fullhad_AK.pdf")





def plot_histogram(prefix, Rs, output_filename, xlabel):
    plt.clf()
    plt.figure(figsize=(8, 6))
    his = {}
    for r in Rs:
        his[f"{prefix}_{r}"] = f"AK{r}"
    for name, label in his.items():
        h = f.Get(name)
        if not h:
            print(f"Warning: histogram {name} not found")
            continue
        edges, y = hist1_to_np(h)
        plt.step(edges, np.r_[y, y[-1]], where="post", label=label)
    plt.xlabel(xlabel)
    plt.ylabel("Events")
    plt.legend()
    plt.grid(True, alpha=0.3)
    # log y scale
    #plt.yscale("log")
    plt.tight_layout()
    plt.savefig("../../idea_fullsim/fast_sim/histograms_view/" + output_filename)

plot_histogram("h_njets", [2,4,5,8,10,15,20,30], "p8_ee_WW_ecm365_fullhad_AK_njets.pdf", "Number of jets")
plot_histogram("h_ngenjets", [2,4,5,8,10,15,20,30], "p8_ee_WW_ecm365_fullhad_AK_ngenjets.pdf", "Number of gen jets")
