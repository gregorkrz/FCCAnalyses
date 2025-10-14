
inputDir = "../../idea_fullsim/fast_sim/histograms"
# for each root file in the direct inputDir, open the root histogram and read the 'h_fancy' histogram. the legend entry should be the root file name. plot it on the same mpl canvas and please normalize it to 1!
import os
import ROOT
import matplotlib.pyplot as plt
import numpy as np

inputDir = "../../idea_fullsim/fast_sim/histograms"

# Get all ROOT files in the directory
root_files = [f for f in os.listdir(inputDir) if f.endswith(".root")]
# remove "p8_ee_ZH_llbb_ecm365".root if it exists
if "p8_ee_ZH_llbb_ecm365.root" in root_files:
    root_files.remove("p8_ee_ZH_llbb_ecm365.root")
plt.figure(figsize=(8, 6))

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
    if integral > 0:
        y_vals = y_vals / integral
    else:
        print(f"Warning: {fname} histogram integral = 0")

    # Plot
    label = os.path.splitext(fname)[0]
    plt.plot(x_vals, y_vals, label=label)

    f.Close()

plt.xlabel("X")
plt.ylabel("Normalized Entries")
plt.title("Comparison of h_fancy histograms")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../../idea_fullsim/fast_sim/histograms_view/comparison_multiple_jets_allJets/norm_E_over_true_overlaid.pdf")
#plt.show()

# also plot a log y version
plt.yscale("log")
plt.ylim(1e-5, 1)
plt.savefig("../../idea_fullsim/fast_sim/histograms_view/comparison_multiple_jets_allJets/norm_E_over_true_overlaid_logy.pdf")
#plt.show()

# There are two histograms: h_genjet_all_energies and h_genjet_matched_energies. Make a plot with the ratio (so basically efficiency) of matched over all vs energy
plt.figure(figsize=(8, 6))
for fname in root_files:
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
    y_all = np.array([hist_all.GetBinContent(i) for i in range(1, n_bins + 1)])
    y_matched = np.array([hist_matched.GetBinContent(i) for i in range(1, n_bins + 1)])

    # Calculate ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.true_divide(y_matched, y_all)
        ratio[~np.isfinite(ratio)] = 0  # set inf and NaN to 0

    # Plot
    label = os.path.splitext(fname)[0]
    plt.plot(x_vals, ratio, label=label)

    f.Close()
plt.xlabel("Gen Jet Energy [GeV]")
plt.ylabel("Matching Efficiency (Matched / All)")
plt.title("Gen Jet Matching Efficiency vs Energy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../../idea_fullsim/fast_sim/histograms_view/comparison_multiple_jets_allJets/matching_efficiency_vs_energy.pdf")

