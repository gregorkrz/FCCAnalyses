import numpy as np
import ROOT
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from copy import copy
import argparse
import os

assert "FOLDER_NAME" in os.environ
histograms_folder = os.environ.get("HISTOGRAMS_FOLDER_NAME", "Histograms_ECM240")
###########################################################################################
# add --folder argument

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default="../../idea_fullsim/fast_sim/{}/{}".format(histograms_folder, os.environ["FOLDER_NAME"]))
parser.add_argument("--output", type=str, default=os.environ["FOLDER_NAME"])
args = parser.parse_args()

###########################################################################################
# python3 resolution_plots.py --folder ../../idea_fullsim/fast_sim/histograms/greedy_matching --output comparison_multiple_jets_allJets_greedyMatching

dir = "../../idea_fullsim/fast_sim/{}/{}".format(histograms_folder, args.output)
# Make dir if it doesn't exist

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

processList = {}

for file in os.listdir(args.folder):
    if file.endswith(".root"):
        proc_name = file.replace(".root", "")
        processList[proc_name] = {'fraction': 1}

########################################################################################################

binsE = [0, 40, 50, 60, 70, 80, 90, 100]
bins_eta = [-5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 5]

# Define the Double-Sided Crystal Ball function
def double_crystal_ball(x, mu, sigma, alphaL, nL, alphaR, nR, norm):
    """Double-sided Crystal Ball PDF."""
    t = (x - mu) / sigma
    result = np.zeros_like(t)

    # Left side
    maskL = t < -alphaL
    result[maskL] = norm * (
        (nL / abs(alphaL)) ** nL
        * np.exp(-0.5 * alphaL**2)
        / (nL / abs(alphaL) - abs(alphaL) - t[maskL]) ** nL
    )

    # Gaussian core
    maskC = (~maskL) & (t < alphaR)
    result[maskC] = norm * np.exp(-0.5 * t[maskC] ** 2)

    # Right side
    maskR = t >= alphaR
    result[maskR] = norm * (
        (nR / abs(alphaR)) ** nR
        * np.exp(-0.5 * alphaR**2)
        / (nR / abs(alphaR) - abs(alphaR) + t[maskR]) ** nR
    )

    return result


def get_result_for_process(procname, bins=binsE, suffix="", sigma_method="std68"):
    # Sigma methods: std68, RMS, interquantile_range
    f = ROOT.TFile.Open(os.path.join(args.folder, "{}.root".format(procname)))
    fig_hist, ax_hist = plt.subplots(figsize=(5, 5))
    def get_std68(theHist, bin_edges, percentage=0.683, epsilon=0.001):
        # theHist, bin_edges = np.histogram(data_for_hist, bins=bins, density=True)
        s =  np.sum(theHist * np.diff(bin_edges))
        if s != 0:
            theHist /= s  # normalize the histogram to 1
        wmin = 0.2
        wmax = 1.8
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
        if low == 0.2 and high == 1.8:
            # Didn't fit well, try mean and stdev
            # Compute the stdev from the histogram
            std68 = 0.0
            #print(theHist)
            print("Fitting didnt work")
            mean = np.sum([(0.5 * (bin_edges[i] + bin_edges[i + 1])) * theHist[i] * (bin_edges[i + 1] - bin_edges[i]) for i in range(len(theHist))])
            print("MEAN", mean)
            std68 = np.sqrt(np.sum([((0.5 * (bin_edges[i] + bin_edges[i + 1])) - mean) ** 2 * theHist[i] * (bin_edges[i + 1] - bin_edges[i]) for i in range(len(theHist))]))
            print("STD68", std68)
            return std68, mean - std68, mean + std68, MPV
        return 0.5 * (high - low), low, high, MPV

    def root_file_get_hist_and_edges(root_file, hist_name, rebin_factor=1):
        # Rebin_factor: if 1, no rebinning, if 2, combine every 2 bins, etc.
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
        if rebin_factor > 1:
            # rebin y and edges
            n_bins_rebinned = len(y) // rebin_factor
            y_rebinned = np.array([np.sum(y[i*rebin_factor:(i+1)*rebin_factor]) for i in range(n_bins_rebinned)])
            edges_rebinned = np.array([edges[i*rebin_factor] for i in range(n_bins_rebinned)] +
                                     [edges[n_bins_rebinned * rebin_factor]])
            return y_rebinned, edges_rebinned
        return y, edges
    bin_mid_points = []
    lo_hi_MPV = []
    sigmaEoverE = []
    responses = []
    bins_to_histograms = {}
    def is_twojet_proc(procname):
        return "vvbb" in procname or "vvqq" in procname or "vvgg" in procname
    for i in range(len(bins) - 1):
        hist_name = f"binned_E_reco_over_true_{suffix}{neg_format(bins[i])}_{neg_format(bins[i+1])}"
        rf = 1
        #if bins[i] == 0 and bins[i+1] == 25 and is_twojet_proc(procname):
        #    rf = 2  # to reduce statistical fluctuations when stats are low
        y, edges = root_file_get_hist_and_edges(f, hist_name, rebin_factor=rf)
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
        if sigma_method == "std68":
            std68, low, high, MPV = get_std68(y, edges, percentage=0.683, epsilon=0.001)
        elif sigma_method == "RMS":
            MPV = 0.5 * (edges[np.argmax(y)] + edges[np.argmax(y) + 1])
            mean = np.sum([(0.5 * (edges[i] + edges[i + 1])) * yc[i] * (edges[i + 1] - edges[i]) for i in range(len(yc))]) / np.sum([yc[i] * (edges[i + 1] - edges[i]) for i in range(len(yc))])
            RMS = np.sqrt(np.sum([((0.5 * (edges[i] + edges[i + 1])) - mean) ** 2 * yc[i] * (edges[i + 1] - edges[i]) for i in range(len(yc))]) / np.sum([yc[i] * (edges[i + 1] - edges[i]) for i in range(len(yc))]))
            std68  = RMS
            low = mean - std68
            high = mean + std68
        elif sigma_method == "interquantile_range":
            MPV = 0.5 * (edges[np.argmax(y)] + edges[np.argmax(y) + 1])
            # compute the cumulative distribution
            s =  np.sum(yc * np.diff(edges))
            if s != 0:
                yc = yc / s  # normalize the histogram to 1
            cumulative = np.cumsum(yc * np.diff(edges))
            # find the 15.85% and 84.15% quantiles
            low_idx = np.searchsorted(cumulative, 0.1585)
            high_idx = np.searchsorted(cumulative, 0.8415)
            low = edges[low_idx]
            high = edges[high_idx]
            std68 = 0.5 * (high - low)
        elif sigma_method == "DSCB":
            centers = 0.5 * (edges[1:] + edges[:-1])
            MPV_guess = 0.5 * (edges[np.argmax(y)] + edges[np.argmax(y) + 1])
            sigma_guess = np.std(np.repeat(centers, yc.astype(int))) if np.sum(yc) > 0 else 1.0
            p0 = [MPV_guess, sigma_guess, 1.5, 3.0, 1.5, 3.0, max(yc)]
            try:
                popt, _ = curve_fit(double_crystal_ball, centers, yc, p0=p0, maxfev=10000)
                mu, sigma, alphaL, nL, alphaR, nR, norm = popt
                MPV = mu
                std68 = sigma
                low, high = mu - sigma, mu + sigma
            except RuntimeError:
                print("⚠️ DSCB fit failed; reverting to RMS.")
                return get_result_for_process(y, edges, sigma_method="RMS")
        else:
            raise ValueError(f"Unknown sigma method: {sigma_method}")
        lo_hi_MPV.append([low, high, MPV])
        bin_mid = 0.5 * (bins[i] + bins[i + 1])
        bin_mid_points.append(bin_mid)
        sigmaEoverE.append(std68 / MPV)
        responses.append(MPV)
        print(f"Bin [{bins[i]}, {bins[i+1]}]: {method} = {std68:.4f}, low = {low:.4f}, high = {high:.4f}, MPV={MPV},N={np.sum(yc)}")
    ax_hist.legend()
    ax_hist.set_xlabel(r'$E_{reco} / E_{true}$')
    ax_hist.set_ylabel('Entries')
    return bin_mid_points, sigmaEoverE, fig_hist, responses, bins_to_histograms, lo_hi_MPV

bin_to_histograms_storage = {}
method_low_high_mid_point_storage = {}

for method in ["std68", "RMS", "interquantile_range"]:
    print("-----------------------------------------------------------")
    print("Using sigma method:", method)
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    for process in sorted(list(processList.keys())):
        bin_mid_points, sigmaEoverE, fig_histograms, resp, bin_to_histograms, mpv_lo_hi = get_result_for_process(process, sigma_method=method)
        if process not in method_low_high_mid_point_storage:
            method_low_high_mid_point_storage[process] = {}
        method_low_high_mid_point_storage[process][method] = mpv_lo_hi
        if method == "std68":
            bin_to_histograms_storage[process] = bin_to_histograms
            fig_histograms.tight_layout()
            fig_histograms.savefig(
                "../../idea_fullsim/fast_sim/{}/{}/bins_{}_{}.pdf".format(histograms_folder, os.environ["FOLDER_NAME"], process, method)
            )
        ax[0].plot(bin_mid_points, sigmaEoverE, ".--", label=process)
        ax[1].plot(bin_mid_points, resp, ".--", label=process)
    ax[0].legend()
    ax[0].set_xlabel('Jet True Energy [GeV]')
    ax[0].set_ylabel(r'$\sigma_E / E$')
    ax[0].set_title('Jet Energy Resolution vs Jet Energy')
    ax[0].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("../../idea_fullsim/fast_sim/{}/{}/jet_energy_resolution_{}.pdf".format(histograms_folder, args.output, method))

method_to_color = {"std68": "blue", "RMS": "orange", "interquantile_range": "green", "DSCB": "red"}
### Plot each bin on a separate plot, but different processes on same plot
fig, ax = plt.subplots(len(binsE) - 1, 1, figsize=(6, 4 * (len(binsE) - 1)), sharex=True)
fig_bins, ax_bins = plt.subplots(len(binsE) - 1, 1, figsize=(6, 4 * (len(binsE) - 1)), sharex=True)

for i in range(len(binsE) - 1):
    for process in sorted(list(processList.keys())):
        y_normalized, edges = bin_to_histograms_storage[process][i]
        # plot on ax[i]
        bin_widths = np.diff(edges)
        ax[i].step(edges[:-1], y_normalized, where="post", label=process)
        ax_bins[i].step(edges[:-1], y_normalized, where="post", label=process)
        for method in method_low_high_mid_point_storage[process]:
            lo, hi, mpv = method_low_high_mid_point_storage[process][method][i]
            # plot vertical lines at lo, hi and mppv using method_to_color
            for _ax in [ax[i], ax_bins[i]]:
                _ax.axvline(lo, color=method_to_color[method], linestyle="--", alpha=0.8)
                _ax.axvline(hi, color=method_to_color[method], linestyle="--", alpha=0.8)
                _ax.axvline(mpv, color=method_to_color[method], linestyle="-", alpha=0.8)
    ax[i].set_title(f'Bin [{binsE[i]}, {binsE[i + 1]}] GeV')
    ax[i].set_ylabel('Entries')
    ax[i].legend()
    ax[i].set_xlim([0.95, 1.05])
    ax_bins[i].set_title(f'Bin [{binsE[i]}, {binsE[i + 1]}] GeV')
    ax_bins[i].set_ylabel('Entries')
    ax_bins[i].legend()
    ax_bins[i].set_yscale("log")

ax[-1].set_xlabel(r'$E_{reco} / E_{true}$')
ax_bins[-1].set_xlabel(r'$E_{reco} / E_{true}$')

fig.tight_layout()
fig.savefig("../../idea_fullsim/fast_sim/{}/{}/jet_energy_bins_comparison.pdf".format(histograms_folder, args.output))

fig_bins.tight_layout()
fig_bins.savefig("../../idea_fullsim/fast_sim/{}/{}/jet_energy_bins_comparison_full_axis.pdf".format(histograms_folder, args.output))

for method in ["std68", "RMS", "interquantile_range"]:
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    for process in sorted(list(processList.keys())):
        bin_mid_points, sigmaEoverE, fig_histograms, resp, _, _ = get_result_for_process(process, bins=bins_eta,
                                                                                         suffix="eta_",
                                                                                         sigma_method=method)
        if method == "std68":
            fig_histograms.tight_layout()
            fig_histograms.savefig(
            "../../idea_fullsim/fast_sim/{}/{}/bins_eta_{}.pdf".format(histograms_folder, args.output, process)
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
    fig.savefig("../../idea_fullsim/fast_sim/{}/{}/jet_Eta_resolution_data_points_{}.pdf".format(histograms_folder, args.output, method))

