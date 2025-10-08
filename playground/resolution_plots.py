import numpy as np
import ROOT
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

###########################################################################################
bins = [0, 50, 100, 150, 200]
input_file = "histograms_jetE_deltaR_matching.root"
f = ROOT.TFile.Open("../../idea_fullsim/fast_sim/histograms/p8_ee_ZH_qqbb_ecm365.root")
############################################################################################

def get_std68(theHist, bin_edges, percentage=0.683, epsilon=0.005):
    # theHist, bin_edges = np.histogram(data_for_hist, bins=bins, density=True)
    wmin = 0.2
    wmax = 1.0
    weight = 0.0
    points = []
    sums = []
    # Fill list of bin centers and the integral up to those point
    for i in range(len(bin_edges) - 1):
        weight += theHist[i] * (bin_edges[i + 1] - bin_edges[i])
        points.append([(bin_edges[i + 1] + bin_edges[i]) / 2, weight])
        sums.append(weight)
    low = wmin
    high = wmax
    width = 100
    for i in range(len(points)):
        for j in range(i, len(points)):
            wy = points[j][1] - points[i][1]
            if abs(wy - percentage) < epsilon:
                wx = points[j][0] - points[i][0]
                if wx < width:
                    low = points[i][0]
                    high = points[j][0]
                    width = wx
    return 0.5 * (high - low), low, high

def root_file_get_hist_and_edges(root_file, hist_name):
    h = root_file.Get(hist_name)
    if not h:
        print(f"Warning: histogram {hist_name} not found")
        return None, None
    nb = h.GetNbinsX()

    edges = np.array([h.GetXaxis().GetBinLowEdge(1)] +
                     [h.GetXaxis().GetBinUpEdge(i) for i in range(1, nb+1)])
    y = np.array([h.GetBinContent(i) for i in range(1, nb+1)], dtype=float)
    assert len(edges) == len(y) + 1
    return y, edges
bin_mid_points = []
sigmaEoverE = []
for i in range(len(bins) - 1):
    hist_name = f"binned_E_reco_over_true_{bins[i]}_{bins[i+1]}"
    y, edges = root_file_get_hist_and_edges(f, hist_name)
    if y is None:
        print(f"Skipping bin [{bins[i]}, {bins[i+1]}] due to missing histogram")
        continue
    print("y:", y, "edges:", edges)
    std68, low, high = get_std68(y, edges, percentage=0.683, epsilon=0.005)
    bin_mid = 0.5 * (bins[i] + bins[i + 1])
    bin_mid_points.append(bin_mid)
    sigmaEoverE.append(std68)
    print(f"Bin [{bins[i]}, {bins[i+1]}]: std68 = {std68:.4f}, low = {low:.4f}, high = {high:.4f}")

fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(bin_mid_points, sigmaEoverE, color='blue', label='Data points')
ax.set_xlabel('Jet True Energy [GeV]')
ax.set_ylabel(r'$\sigma_E / E$')
ax.set_title('Jet Energy Resolution vs Jet Energy')
ax.grid(True, alpha=0.3)
fig.savefig("jet_energy_resolution_data_points.pdf")

