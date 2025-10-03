import ROOT
import numpy as np
import matplotlib.pyplot as plt

ROOT.gROOT.SetBatch(True)  # don’t pop up ROOT canvases

# open your file
f = ROOT.TFile.Open("myfile.root")

# mapping of histogram name → legend label
hists = {
    "h_ratio_2":  "AK2",
    "h_ratio_4":  "AK4",
    "h_ratio_5":  "AK5",
    "h_ratio_8":  "AK8",
    "h_ratio_10": "AK10",
    "h_ratio_15": "AK15",
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
plt.tight_layout()
plt.show()

f.Close()
