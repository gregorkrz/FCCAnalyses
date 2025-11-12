# For each root file in the direct inputDir, open the root histogram and read the 'h_fancy' histogram. the legend entry should be the root file name. plot it on the same mpl canvas and please normalize it to 1!
import os
import ROOT
import matplotlib.pyplot as plt
import numpy as np
from Colors import PROCESS_COLORS, HUMAN_READABLE_PROCESS_NAMES, LINE_STYLES

assert "INPUT_DIR" in os.environ # To make sure we are taking the right input dir and folder name
assert "FOLDER_NAME" in os.environ
assert "HISTOGRAMS_FOLDER_NAME" in os.environ

import matplotlib
matplotlib.rcParams.update({
    #'font.sans-serif': "Arial",
    'font.family': "sans-serif", # Ensure Matplotlib uses the sans-serif family
    #"mathtext.fontset": "stix", # serif math, similar to LaTeX Times
    #"mathtext.default": "it",   # math variables italic by default
    "font.size": 12
})

inputDir = "../../idea_fullsim/fast_sim/{}/{}".format(os.environ["HISTOGRAMS_FOLDER_NAME"], os.environ["FOLDER_NAME"])
# Get all ROOT files in the directory
root_files = [f for f in os.listdir(inputDir) if f.endswith(".root")]
# remove "p8_ee_ZH_llbb_ecm365".root if it exists
if "p8_ee_ZH_llbb_ecm365.root" in root_files:
    root_files.remove("p8_ee_ZH_llbb_ecm365.root") # Quick fix we dont need that file for now

plt.figure(figsize=(8, 6))
for fname in sorted(root_files):
    file_path = os.path.join(inputDir, fname)
    f = ROOT.TFile.Open(file_path)
    if not f or f.IsZombie():
        print(f"Could not open {fname}")
        continue
    hist = f.Get("h_fancy")
    if not hist:
        print(f"No 'h_fancy' histogram in {fname}")
        f.Close()
        continue
    # Convert histogram to numpy arrays
    n_bins = hist.GetNbinsX()
    x_vals = np.array([hist.GetBinCenter(i) for i in range(1, n_bins + 1)])
    y_vals = np.array([hist.GetBinContent(i) for i in range(1, n_bins + 1)])
    # Normalize
    integral = np.sum(y_vals)
    print("Integral of histogram in {}: {}".format(fname, integral))
    if integral > 0:
        y_vals = y_vals / integral
    else:
        print(f"Warning: {fname} histogram integral = 0")
    # Plot
    label = os.path.splitext(fname)[0]
    #plt.plot(x_vals, y_vals, "x", label=HUMAN_READABLE_PROCESS_NAMES[label], color=PROCESS_COLORS[label])
    plt.plot(x_vals, y_vals, linestyle=LINE_STYLES[label], color=PROCESS_COLORS[label])
    f.Close()
plt.xlabel("$E_{reco} / E_{true}")
plt.xlim([0.9, 1.1])
plt.ylabel("Normalized Entries")
#plt.title()
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../../idea_fullsim/fast_sim/{}/{}/norm_E_over_true_overlaid.pdf".format(os.environ["HISTOGRAMS_FOLDER_NAME"], os.environ["FOLDER_NAME"]))

# Make the same plot but with a thin line, only for nu nu q q and with xlim from -0.6 to +0.6
fig, ax = plt.subplots(2, 1, figsize=(5, 10))
for fname in root_files:
    file_path = os.path.join(inputDir, fname)
    f = ROOT.TFile.Open(file_path)
    if not f or f.IsZombie():
        print(f"Could not open {fname}")
        continue
    hist = f.Get("h_fancy")
    if not hist:
        print(f"No 'h_fancy' histogram in {fname}")
        f.Close()
        continue
    # Convert histogram to numpy arrays
    n_bins = hist.GetNbinsX()
    x_vals = np.array([hist.GetBinCenter(i) for i in range(1, n_bins + 1)])
    y_vals = np.array([hist.GetBinContent(i) for i in range(1, n_bins + 1)])
    # Normalize
    integral = np.sum(y_vals)
    print("Integral of histogram in {}: {}".format(fname, integral))
    if integral > 0:
        y_vals = y_vals / integral
    else:
        print(f"Warning: {fname} histogram integral = 0")
    # Plot
    label = os.path.splitext(fname)[0]
    ax[0].plot(x_vals, y_vals, label=label, linewidth=1)
    ax[1].plot(x_vals, y_vals, label=label, linewidth=1)
    # Now make a Gaussian fit from x_vals 0.8 to 1.2. Plot it in that range as well and put sigma and mean in the legend
    # You can initialize parameters with stdev and mean
    # Do it here:
    # Fit a Gaussian to the 0.8–1.2 range and plot it
    mask = (x_vals >= 0.8) & (x_vals <= 1.2)
    x_fit = x_vals[mask]
    y_fit = y_vals[mask]
    if x_fit.size >= 3 and np.sum(y_fit) > 0:
        # Moment-based initial guesses (use y as weights)
        w = y_fit
        mu0 = np.average(x_fit, weights=w)
        var0 = np.average((x_fit - mu0) ** 2, weights=w)
        sigma0 = np.sqrt(max(var0, 1e-12))
        A0 = y_fit.max()
        def gauss(x, A, mu, sigma):
            # clip sigma to avoid division by zero during fitting/plotting
            s = np.clip(sigma, 1e-12, None)
            return A * np.exp(-0.5 * ((x - mu) / s) ** 2)
        # Try a nonlinear least-squares fit; fall back to moment estimates if it fails
        try:
            from scipy.optimize import curve_fit
            popt, _ = curve_fit(
                gauss, x_fit, y_fit,
                p0=[A0, mu0, sigma0],
                bounds=([0.0, 0.8, 1e-6], [np.inf, 1.2, np.inf]),
                maxfev=10000,
            )
            A, mu, sigma = popt
        except Exception as _e:
            # Fallback: use moment estimates without optimization
            A, mu, sigma = A0, mu0, sigma0
        # Plot the fitted Gaussian over the fit window
        x_dense = np.linspace(0.9, 1.1, 200)
        ax[0].plot(
            x_dense,
            gauss(x_dense, A, mu, sigma),
            linestyle="--",
            linewidth=1,
            label=f"{label} fit μ={mu:.3f}, σ={sigma:.3f}",
        )
        ax[1].plot(
            x_dense,
            gauss(x_dense, A, mu, sigma),
            linestyle="--",
            linewidth=1,
            label=f"{label} fit μ={mu:.3f}, σ={sigma:.3f}",
        )
    else:
        print(f"Not enough points in [0.8, 1.2] for {fname} to fit.")
    #############################################################################
    ### Plot on the same plot the histogram h_ratio_matching_with_partons     ###
    #############################################################################
    '''hist_ratio = f.Get("h_ratio_matching_with_partons")
    if not hist_ratio:
        print(f"No 'h_ratio_matching_with_partons' histogram in {fname}")
        f.Close()
        continue
    # Convert histogram to numpy arrays
    n_bins_ratio = hist_ratio.GetNbinsX()
    x_vals_ratio = np.array([hist_ratio.GetBinCenter(i) for i in range(1, n_bins_ratio + 1)])
    y_vals_ratio = np.array([hist_ratio.GetBinContent(i) for i in range(1, n_bins_ratio + 1)])
    # Normalize
    integral_ratio = np.sum(y_vals_ratio)
    print("Integral of histogram ratio in {}: {}".format(fname, integral_ratio))
    if integral_ratio > 0:
        y_vals_ratio = y_vals_ratio / integral_ratio
    else:
        print(f"Warning: {fname} histogram ratio integral = 0")
    # Plot
    ax.plot(x_vals_ratio, y_vals_ratio, label=label + " (matched to partons)", linestyle='dashed', linewidth=1)'''
    #############
    # Print on the plot the text 'eta < -0.9'
    #ax.text(0.95, 0.9 - 0.1 * root_files.index(fname), "eta < -0.9".format(label), transform=ax.transAxes)
    ax[0].set_xlabel("E_reco / E_true")
    ax[0].set_ylabel("Normalized Entries")
    ax[0].set_title("Comparison of E_reco/E_true histograms (ee->ZH->nu nu g g)")
    ax[0].legend()
    ax[0].grid(True)
    ax[1].grid(True)
    ax[1].set_xlim([0.85, 1.15])
    f.Close()

fig.tight_layout()
fig.savefig("../../idea_fullsim/fast_sim/{}/{}/norm_E_over_true_overlaid_vvgg.pdf".format(os.environ["HISTOGRAMS_FOLDER_NAME"], os.environ["FOLDER_NAME"]))
#plt.show()

# also plot a log y version
plt.yscale("log")
plt.ylim(1e-5, 1)
plt.xlim([0.5, 1.5])
plt.savefig("../../idea_fullsim/fast_sim/{}/{}/norm_E_over_true_overlaid_logy.pdf".format(os.environ["HISTOGRAMS_FOLDER_NAME"], os.environ["FOLDER_NAME"]))
#plt.show()

# There are two histograms: h_genjet_all_energies and h_genjet_matched_energies. Make a plot with the ratio (so basically efficiency) of matched over all vs energy
plt.figure(figsize=(6, 6))
for fname in sorted(root_files):
    file_path = os.path.join(inputDir, fname)
    f = ROOT.TFile.Open(file_path)
    if not f or f.IsZombie():
        print(f"Could not open {fname}")
        continue
    hist_all = f.Get("h_genjet_all_energies")
    hist_matched = f.Get("h_genjet_matched_energies")
    if not hist_all or not hist_matched:
        print(f"No required histograms in {fname}")
        f.Close()
        continue
    # Convert histograms to numpy arrays
    n_bins = hist_all.GetNbinsX()
    x_vals = np.array([hist_all.GetBinCenter(i) for i in range(1, n_bins + 1)])
    # Remove the xvals larger than 175
    #filt = (x_vals <= 100) & (x_vals >= 20)
    filt = np.array([hist_all.GetBinContent(i) for i in range(1, n_bins + 1)]) > 1000 # Cut out the low statistics bins
    filt = filt & (x_vals <= 100)
    x_vals = x_vals[filt]
    y_all = np.array([hist_all.GetBinContent(i) for i in range(1, n_bins + 1)])[filt]
    y_matched = np.array([hist_matched.GetBinContent(i) for i in range(1, n_bins + 1)])[filt]
    # Calculate ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.true_divide(y_matched, y_all)
        ratio[~np.isfinite(ratio)] = 0  # set inf and NaN to 0
    # Plot
    label = os.path.splitext(fname)[0]
    plt.plot(x_vals, ratio, LINE_STYLES[label], label=HUMAN_READABLE_PROCESS_NAMES[label], color=PROCESS_COLORS[label])
    plt.plot(x_vals, ratio, "x", color=PROCESS_COLORS[label])
    f.Close()

plt.xlabel("$E_{true}$ [GeV]")
plt.ylabel("Matching Efficiency (Matched/ All)")
#plt.ylim([0.80, 1.001])
plt.title("Jet Matching Efficiency vs. Energy")
plt.legend(title="l ∈ {u, d, s}; q ∈ {u, d, s, c, b}", fontsize=10, title_fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.savefig("../../idea_fullsim/fast_sim/{}/{}/matching_efficiency_vs_energy.pdf".format(os.environ["HISTOGRAMS_FOLDER_NAME"], os.environ["FOLDER_NAME"]))
plt.clf()

## Produce the Higgs mass plots similar to the ones in the fccanalysis plots file but easier to manipulate
fig, ax = plt.subplots(len(root_files), 2,  figsize=(8, 2.7 * len(root_files)))
fig_mH_all, ax_mH_all = plt.subplots(1, 2, figsize=(8, 6)) # plot mH reco of all root files on the same plot, left side fully and right side zoomed into the peak

figlog, axlog = plt.subplots(len(root_files), 2, figsize=(8, 2.7 * len(root_files)))

if len(root_files) == 1:
    ax = np.array([ax])
    axlog = np.array([axlog])

# Higgs mass histogram
for i, fname in enumerate(sorted(root_files)):
    file_path = os.path.join(inputDir, fname)
    f = ROOT.TFile.Open(file_path)
    if not f or f.IsZombie():
        print(f"Could not open {fname}")
        continue
    hist_gen = f.Get("h_mH_gen")
    hist_reco = f.Get("h_mH_reco")
    hist_gt = f.Get("h_mH_stable_gt_particles") # for the resolution with ideal clustering
    hist_gt_recomatched = f.Get("h_mH_reco_particles_matched")
    if not hist_gen or not hist_reco:
        print(f"No 'h_mH_gen'/'h_mH_reco' histogram in {fname}")
        f.Close()
        continue
    n_bins = hist_gen.GetNbinsX()
    x_vals_gen = np.array([hist_gen.GetBinCenter(i) for i in range(1, n_bins + 1)])
    y_vals_gen = np.array([hist_gen.GetBinContent(i) for i in range(1, n_bins + 1)])

    nbins_gt = hist_gt.GetNbinsX()
    x_vals_gt = np.array([hist_gt.GetBinCenter(i) for i in range(1, nbins_gt + 1)])
    y_vals_gt = np.array([hist_gt.GetBinContent(i) for i in range(1, nbins_gt + 1)])

    nbins_gt_reco_matched = hist_gt_recomatched.GetNbinsX()
    x_vals_gt_recomatched = np.array([hist_gt_recomatched.GetBinCenter(i) for i in range(1, nbins_gt_reco_matched + 1)])
    y_vals_gt_recomatched = np.array([hist_gt_recomatched.GetBinContent(i) for i in range(1, nbins_gt_reco_matched + 1)])
    step_size_gt_recomatched = x_vals_gt_recomatched[1] - x_vals_gt_recomatched[0]
    integral_gt_recomatched = np.sum(y_vals_gt_recomatched)
    print("Integral of reco-GT matched histogram in {}: {}".format(fname, integral_gt_recomatched))
    if integral_gt_recomatched > 0:
        y_vals_gt_recomatched = y_vals_gt_recomatched / integral_gt_recomatched / step_size_gt_recomatched
    else:
        print(f"Warning: {fname} reco-GT matched histogram integral = 0")
    # Normalize
    integral = np.sum(y_vals_gen)
    step_size_gen = x_vals_gen[1] - x_vals_gen[0]
    print("Integral of histogram in {}: {}".format(fname, integral))
    if integral > 0:
        y_vals_gen = y_vals_gen / integral / step_size_gen
    else:
        print(f"Warning: {fname} histogram integral = 0")
    print("Sum of y vals now", np.sum(y_vals_gen))
    integral_gt = np.sum(y_vals_gt)
    step_size_gt = x_vals_gt[1] - x_vals_gt[0]
    print("Integral of GT histogram in {}: {}".format(fname, integral_gt))
    if integral_gt > 0:
        y_vals_gt = y_vals_gt / integral_gt / step_size_gt
    else:
        print(f"Warning: {fname} GT histogram integral = 0")
    # Plot
    label = os.path.splitext(fname)[0]
    # Plot the histograms with bins for reco (full lines) and gen(dashed lines)
    n_bins = hist_reco.GetNbinsX()
    x_vals_reco = np.array([hist_reco.GetBinCenter(i) for i in range(1, n_bins + 1)])
    y_vals_reco = np.array([hist_reco.GetBinContent(i) for i in range(1, n_bins + 1)])
    integral_reco = np.sum(y_vals_reco)
    step_size_reco = x_vals_reco[1] - x_vals_reco[0]
    print("Integral of histogram in {}: {}".format(fname, integral_reco))
    if integral_reco > 0:
        y_vals_reco = y_vals_reco / integral_reco
        y_vals_reco = y_vals_reco / step_size_reco  # Normalize to bin width
    else:
        print(f"Warning: {fname} histogram integral = 0")
    #ax_mH_all[0].step(x_vals_reco, y_vals_reco, where='mid', label=label, linestyle='solid')
    #ax_mH_all[1].step(x_vals_reco, y_vals_reco, where='mid', label=label, linestyle='solid')
    ax_mH_all[0].step(x_vals_reco, y_vals_reco, where='mid', color=PROCESS_COLORS[label], linestyle=LINE_STYLES[label], label=HUMAN_READABLE_PROCESS_NAMES[label])
    ax_mH_all[1].step(x_vals_reco, y_vals_reco, where='mid', color=PROCESS_COLORS[label], linestyle=LINE_STYLES[label], label=HUMAN_READABLE_PROCESS_NAMES[label])
    #ax.plot(x_vals_reco, y_vals_reco, label=label + " (reco)", linestyle='solid')
    #ax.plot(x_vals_gen, y_vals_gen, label=label + " (gen)", linestyle='dashed')
    for k in range(2):
        ax[i, k].step(x_vals_reco, y_vals_reco, where='mid', color="blue", label="reco")
        ax[i, k].step(x_vals_gen, y_vals_gen, where='mid', color="orange", label="gen")
        #ax[i].step(x_vals_gt, y_vals_gt, where='mid', color="green", label="GT", linestyle='dotted')
        axlog[i, k].step(x_vals_reco, y_vals_reco, where='mid', color="blue", label="reco")
        axlog[i, k].step(x_vals_gen, y_vals_gen, where='mid', color="orange", label="gen")
        axlog[i, k].step(x_vals_gt, y_vals_gt, where='mid', color="green", label="GT")
        axlog[i, k].step(x_vals_gt_recomatched, y_vals_gt_recomatched, where='mid', color="red", label="reco-GT matched")
        axlog[i, k].legend()
        axlog[i, k].set_ylim([1e-3, 1])
        ax[i, k].legend()
        axlog[i, k].set_title(label)
        axlog[i, k].set_yscale("log")
        axlog[i, k].set_xlabel("$m_H$ [GeV]")
        axlog[i, k].set_ylabel("Events (norm.)")
        ax[i, k].set_title(label)
    ax[i, 1].set_xlim([115, 135])
    axlog[i, 1].set_xlim([115, 135])

#ax.legend()
#ax.set_yscale("log")

p = "../../idea_fullsim/fast_sim/{}/{}/Higgs_mass_reco_vs_gen.pdf".format(os.environ["HISTOGRAMS_FOLDER_NAME"], os.environ["FOLDER_NAME"])
plog = "../../idea_fullsim/fast_sim/{}/{}/log_Higgs_mass_reco_vs_gen.pdf".format(os.environ["HISTOGRAMS_FOLDER_NAME"], os.environ["FOLDER_NAME"])
phiggs = "../../idea_fullsim/fast_sim/{}/{}/Higgs_mass_reco_overlaid_mH_reco_normalized.pdf".format(os.environ["HISTOGRAMS_FOLDER_NAME"], os.environ["FOLDER_NAME"])


ax_mH_all[0].set_xlabel("$m_H$ [GeV]")
ax_mH_all[1].set_xlabel("$m_H$ [GeV]")
ax_mH_all[0].set_ylabel("Normalized Events")
ax_mH_all[1].set_ylabel("Normalized Events")
ax_mH_all[1].set_xlim([115, 135])
ax_mH_all[0].grid()
ax_mH_all[1].grid()
ax_mH_all[0].legend(title="l ∈ {u, d, s}; q ∈ {u, d, s, c, b}", fontsize=12, title_fontsize=10)
#ax_mH_all[1].legend()

fig_mH_all.tight_layout()
fig.tight_layout()
fig.savefig(p)
figlog.tight_layout()
figlog.savefig(plog)
fig_mH_all.savefig(phiggs)
print("Saving to", p, plog, phiggs)


# Make a plot of h_mH_all_stable_part
fig2, ax2 = plt.subplots(len(root_files), 1, figsize=(8, 2.7 * len(root_files)))
if len(root_files) == 1:
    ax2 = np.array([ax2])
for i, fname in enumerate(root_files):
    file_path = os.path.join(inputDir, fname)
    f = ROOT.TFile.Open(file_path)
    if not f or f.IsZombie():
        print(f"Could not open {fname}")
        continue
    hist_gen = f.Get("h_mH_all_stable_part")
    if not hist_gen:
        print(f"No 'h_mH_all_stable_part' histogram in {fname}")
        f.Close()
        continue
    n_bins = hist_gen.GetNbinsX()
    x_vals_gen = np.array([hist_gen.GetBinCenter(i) for i in range(1, n_bins + 1)])
    y_vals_gen = np.array([hist_gen.GetBinContent(i) for i in range(1, n_bins + 1)])
    # Normalize
    integral = np.sum(y_vals_gen)
    step_size_gen = x_vals_gen[1] - x_vals_gen[0]
    print("Integral of histogram in {}: {}".format(fname, integral))
    if integral > 0:
        y_vals_gen = y_vals_gen / integral / step_size_gen
    else:
        print(f"Warning: {fname} histogram integral = 0")
    # Plot
    label = os.path.splitext(fname)[0]
    ax2[i].step(x_vals_gen, y_vals_gen, where='mid', label=label)
    ax2[i].set_title(label)
    ax2[i].set_xlabel("Inv Mass all gen particles [GeV]")
    ax2[i].set_ylabel("Normalized Entries / GeV")
    ax2[i].legend()
    ax2[i].grid()
    f.Close()
fig2.tight_layout()
fig2.savefig("../../idea_fullsim/fast_sim/{}/{}/inv_mass_all_gen_particles_normalized.pdf".format(os.environ["HISTOGRAMS_FOLDER_NAME"], os.environ["FOLDER_NAME"]))
plt.clf()
#plt.show()

# Make a histogram of h_E_reco_over_true_Charged. First, zoomed in from 0.9 to 1.1, then, log y from 0 to 2 (full range of the histogram)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
for fname in root_files:
    file_path = os.path.join(inputDir, fname)
    f = ROOT.TFile.Open(file_path)
    if not f or f.IsZombie():
        print(f"Could not open {fname}")
        continue
    hist = f.Get("h_E_reco_over_true_Charged")
    if not hist:
        print(f"No 'h_E_reco_over_true_Charged' histogram in {fname}")
        f.Close()
        continue
    n_bins = hist.GetNbinsX()
    x_vals = np.array([hist.GetBinCenter(i) for i in range(1, n_bins + 1)])
    y_vals = np.array([hist.GetBinContent(i) for i in range(1, n_bins + 1)])
    # Normalize
    integral = np.sum(y_vals)
    step_size = x_vals[1] - x_vals[0]
    print("Integral of histogram in {}: {}".format(fname, integral))
    if integral > 0:
        y_vals = y_vals / integral / step_size
    else:
        print(f"Warning: {fname} histogram integral = 0")
    label = os.path.splitext(fname)[0]
    ax[0].step(x_vals, y_vals, where='mid', label=label)
    ax[1].step(x_vals, y_vals, where='mid', label=label)
    f.Close()

ax[0].set_xlabel("E_reco / E_true (Charged)")
ax[0].set_xlim([0.9, 1.1])
ax[0].legend()
ax[1].set_xlabel("E_reco / E_true (Charged)")
ax[1].set_yscale("log")
ax[1].legend()
ax[0].set_ylabel("Normalized Entries / bin width")
fig.tight_layout()
fig.savefig("../../idea_fullsim/fast_sim/{}/{}/E_reco_over_true_Charged.pdf".format(os.environ["HISTOGRAMS_FOLDER_NAME"], os.environ["FOLDER_NAME"]))
plt.clf()
