import numpy as np
import ROOT
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from copy import copy
import argparse
import os

assert "INPUT_DIR" in os.environ # To make sure we are taking the right input dir and folder name
assert "FOLDER_NAME" in os.environ

###########################################################################################
# add --folder argument

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default="../../idea_fullsim/fast_sim/Histograms_ECM240/{}".format(os.environ["FOLDER_NAME"]))
parser.add_argument("--output", type=str, default=os.environ["FOLDER_NAME"])
args = parser.parse_args()

###########################################################################################
# python3 resolution_plots.py --folder ../../idea_fullsim/fast_sim/histograms/greedy_matching --output comparison_multiple_jets_allJets_greedyMatching

dir = "../../idea_fullsim/fast_sim/Histograms_ECM240/{}".format(args.output)
# make dir if it doesn't exist
os.makedirs(dir, exist_ok=True)
print("Saving to directory:", dir)

def point_format(number):
    return str(number).replace(".", "p")
def neg_format(number):
    # put n5 for -5
    if number < 0:
        return point_format("n{}".format(abs(number)))
    else:
        return point_format(number)

processList = {
    "p8_ee_ZH_qqbb_ecm240": {'fraction': 1},
    "p8_ee_ZH_6jet_ecm240": {'fraction': 1},
    "p8_ee_ZH_vvbb_ecm240": {'fraction': 1},
    "p8_ee_ZH_bbbb_ecm240": {'fraction': 1},
    "p8_ee_ZH_vvgg_ecm240": {'fraction': 1},
}

########################################################################################################

binsE = [0, 50, 75, 100]
bins_eta = [-5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 5]

def get_result_for_process(procname, bins=binsE, suffix=""):
    f = ROOT.TFile.Open(os.path.join(args.folder, "{}.root".format(procname)))
    fig_hist, ax_hist = plt.subplots(figsize=(5, 5))
    def get_std68(theHist, bin_edges, percentage=0.683, epsilon=0.001):
        # theHist, bin_edges = np.histogram(data_for_hist, bins=bins, density=True)
        s =  np.sum(theHist * np.diff(bin_edges))
        if s != 0:
            theHist /= s  # normalize the histogram to 1
        wmin = 0.2
        wmax = 1.0
        weight = 0.0
        points = []
        sums = []
        am = np.argmax(theHist)
        MPV = 0.5 * (bin_edges[am] + bin_edges[am + 1])
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
        if low == 0.2 and high == 1.0:
            # Didn't fit well, try mean and stdev
            # compute the stdev from the histogram
            std68 = 0.0
            #print(theHist)
            print("Fitting didnt work")
            mean = np.sum([(0.5 * (bin_edges[i] + bin_edges[i + 1])) * theHist[i] * (bin_edges[i + 1] - bin_edges[i]) for i in range(len(theHist))])
            print("MEAN", mean)
            std68 = np.sqrt(np.sum([((0.5 * (bin_edges[i] + bin_edges[i + 1])) - mean) ** 2 * theHist[i] * (bin_edges[i + 1] - bin_edges[i]) for i in range(len(theHist))]))
            print("STD68", std68)
            return std68, mean - std68, mean + std68, MPV
        return 0.5 * (high - low), low, high, MPV

    def root_file_get_hist_and_edges(root_file, hist_name):
        # Print available root column names
        print("Available histograms:", [key.GetName() for key in root_file.GetListOfKeys()])
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
    responses = []
    bins_to_histograms = {}
    for i in range(len(bins) - 1):
        hist_name = f"binned_E_reco_over_true_{suffix}{neg_format(bins[i])}_{neg_format(bins[i+1])}"
        y, edges = root_file_get_hist_and_edges(f, hist_name)
        if y is None:
            print(f"Skipping bin [{bins[i]}, {bins[i+1]}] due to missing histogram")
            continue
        #print("y:", y, "edges:", edges)
        # plot the current bin histogram using y and edges (NORMALIZED)
        #ax_hist.step(edges[:-1], y, where="post", label=f"[{bins[i]}, {bins[i+1]}] GeV")
        bin_widths = np.diff(edges)
        area = np.sum(y * bin_widths)
        if area != 0:
            y_normalized = y / area
        else:
            y_normalized = y
        ax_hist.step(edges[:-1], y_normalized, where="post", label=f"[{bins[i]}, {bins[i + 1]}] GeV")
        bins_to_histograms[i] = [y_normalized, edges]
        yc = copy(y)
        std68, low, high, MPV = get_std68(y, edges, percentage=0.683, epsilon=0.001)
        bin_mid = 0.5 * (bins[i] + bins[i + 1])
        bin_mid_points.append(bin_mid)
        sigmaEoverE.append(std68 / MPV)
        responses.append(MPV)
        print(f"Bin [{bins[i]}, {bins[i+1]}]: std68 = {std68:.4f}, low = {low:.4f}, high = {high:.4f}, MPV={MPV},N={np.sum(yc)}")
    ax_hist.legend()
    ax_hist.set_xlabel(r'$E_{reco} / E_{true}$')
    ax_hist.set_ylabel('Entries')
    return bin_mid_points, sigmaEoverE, fig_hist, responses, bins_to_histograms


fig, ax = plt.subplots(2, 1, figsize=(8, 6))
bin_to_histograms_storage = {}
for process in sorted(list(processList.keys())):
    bin_mid_points, sigmaEoverE, fig_histograms, resp, bin_to_histograms = get_result_for_process(process)
    bin_to_histograms_storage[process] = bin_to_histograms
    fig_histograms.tight_layout()
    fig_histograms.savefig(
        "../../idea_fullsim/fast_sim/Histograms_ECM240/{}/bins_{}.pdf".format(os.environ["FOLDER_NAME"], process)
    )
    ax[0].plot(bin_mid_points, sigmaEoverE, ".--", label=process)
    ax[1].plot(bin_mid_points, resp, ".--", label=process)
ax[0].legend()
ax[0].set_xlabel('Jet True Energy [GeV]')
ax[0].set_ylabel(r'$\sigma_E / E$')
ax[0].set_title('Jet Energy Resolution vs Jet Energy')
ax[0].grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("../../idea_fullsim/fast_sim/Histograms_ECM240/{}/jet_energy_resolution_data_points.pdf".format(args.output))


### Plot each bin on a separate plot, but different processes on same plot
fig, ax = plt.subplots(len(binsE) - 1, 1, figsize=(6, 4 * (len(binsE) - 1)), sharex=True)
for i in range(len(binsE) - 1):
    for process in sorted(list(processList.keys())):
        y_normalized, edges = bin_to_histograms_storage[process][i]
        # plot on ax[i]
        bin_widths = np.diff(edges)

        ax[i].step(edges[:-1], y_normalized, where="post", label=process)
    ax[i].set_title(f'Bin [{binsE[i]}, {binsE[i + 1]}] GeV')
    ax[i].set_ylabel('Entries')
    ax[i].legend()
    ax[i].set_xlim([0.95, 1.05])
ax[-1].set_xlabel(r'$E_{reco} / E_{true}$')

fig.tight_layout()
fig.savefig("../../idea_fullsim/fast_sim/Histograms_ECM240/{}/jet_energy_bins_comparison.pdf".format(args.output))

fig, ax = plt.subplots(2, 1, figsize=(8, 6))
for process in sorted(list(processList.keys())):
    bin_mid_points, sigmaEoverE, fig_histograms, resp, _ = get_result_for_process(process, bins=bins_eta, suffix="eta_")
    fig_histograms.tight_layout()
    fig_histograms.savefig(
        "../../idea_fullsim/fast_sim/Histograms_ECM240/{}/bins_eta_{}.pdf".format(args.output, process)
    )
    ax[0].plot(bin_mid_points, sigmaEoverE, ".--", label=process)
    ax[1].plot(bin_mid_points, resp, ".--", label=process)


ax[0].legend()
ax[0].set_xlabel('Jet Eta [GeV]')
ax[0].set_ylabel(r'$\sigma_E / E$')
ax[0].set_title('Jet Energy Resolution vs Jet Energy')
ax[0].grid(True, alpha=0.3)
ax[1].grid(True, alpha=0.3)
ax[1].set_title("Energy Response vs Energy")
ax[1].set_xlabel('Jet Eta [GeV]')
ax[1].set_ylabel("$\sigma_E / E$")
fig.tight_layout()
fig.savefig("../../idea_fullsim/fast_sim/Histograms_ECM240/{}/jet_Eta_resolution_data_points.pdf".format(args.output))

