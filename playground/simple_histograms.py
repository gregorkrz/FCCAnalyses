
# For each root file in the direct inputDir, open the root histogram and read the 'h_fancy' histogram. the legend entry should be the root file name. plot it on the same mpl canvas and please normalize it to 1!
import os
import ROOT
import matplotlib.pyplot as plt
import numpy as np

assert "INPUT_DIR" in os.environ # To make sure we are taking the right input dir and folder name
assert "FOLDER_NAME" in os.environ

inputDir = "../../idea_fullsim/fast_sim/Histograms_ECM240/{}".format(os.environ["FOLDER_NAME"])

# Get all ROOT files in the directory
root_files = [f for f in os.listdir(inputDir) if f.endswith(".root")]
# remove "p8_ee_ZH_llbb_ecm365".root if it exists
if "p8_ee_ZH_llbb_ecm365.root" in root_files:
    root_files.remove("p8_ee_ZH_llbb_ecm365.root") # quick fix we dont need that file for now

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
    print("Integral of histogram in {}: {}".format(fname, integral))
    if integral > 0:
        y_vals = y_vals / integral
    else:
        print(f"Warning: {fname} histogram integral = 0")
    # Plot
    label = os.path.splitext(fname)[0]
    plt.plot(x_vals, y_vals, label=label)
    f.Close()

plt.xlabel("E_reco / E_true")
plt.xlim([0.9, 1.1])
plt.ylabel("Normalized Entries")
plt.title("Comparison of h_fancy histograms")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../../idea_fullsim/fast_sim/Histograms_ECM240/{}/norm_E_over_true_overlaid.pdf".format(os.environ["FOLDER_NAME"]))
#plt.show()

# also plot a log y version
plt.yscale("log")
plt.ylim(1e-5, 1)
plt.xlim([0.5, 1.5])
plt.savefig("../../idea_fullsim/fast_sim/Histograms_ECM240/{}/norm_E_over_true_overlaid_logy.pdf".format(os.environ["FOLDER_NAME"]))
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
    # remove the xvals larger than 175
    filt = x_vals <= 100
    x_vals = x_vals[filt]
    y_all = np.array([hist_all.GetBinContent(i) for i in range(1, n_bins + 1)])[filt]
    y_matched = np.array([hist_matched.GetBinContent(i) for i in range(1, n_bins + 1)])[filt]
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
plt.ylim([0.95, 1.02])
plt.title("Gen Jet Matching Efficiency vs Energy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../../idea_fullsim/fast_sim/Histograms_ECM240/{}/matching_efficiency_vs_energy.pdf".format(os.environ["FOLDER_NAME"]))
plt.clf()



## Produce the Higgs mass plots similar to the ones in the fccanalysis plots file but easier to manipulate
fig, ax = plt.subplots(figsize=(6, 6))
# Higgs mass histogram

for fname in root_files:
    file_path = os.path.join(inputDir, fname)
    f = ROOT.TFile.Open(file_path)
    if not f or f.IsZombie():
        print(f"Could not open {fname}")
        continue
    hist_gen = f.Get("h_mH_gen")
    hist_reco = f.Get("h_mH_reco")
    if not hist_gen or not hist_reco:
        print(f"No 'h_mH_gen'/'h_mH_reco' histogram in {fname}")
        f.Close()
        continue
    n_bins = hist_gen.GetNbinsX()
    x_vals_gen = np.array([hist_gen.GetBinCenter(i) for i in range(1, n_bins + 1)])
    y_vals_gen = np.array([hist_gen.GetBinContent(i) for i in range(1, n_bins + 1)])
    # Normalize
    integral = np.sum(y_vals_gen)
    print("Integral of histogram in {}: {}".format(fname, integral))
    if integral > 0:
        y_vals = y_vals_gen / integral
    else:
        print(f"Warning: {fname} histogram integral = 0")
    # Plot
    label = os.path.splitext(fname)[0]
    # plot the histograms with bins for reco (full lines) and gen(dashed lines)
    n_bins = hist_reco.GetNbinsX()
    x_vals_reco = np.array([hist_reco.GetBinCenter(i) for i in range(1, n_bins + 1)])
    y_vals_reco = np.array([hist_reco.GetBinContent(i) for i in range(1, n_bins + 1)])
    integral_reco = np.sum(y_vals_reco)
    print("Integral of histogram in {}: {}".format(fname, integral_reco))
    if integral_reco > 0:
        y_vals_reco = y_vals_reco / integral_reco
    else:
        print(f"Warning: {fname} histogram integral = 0")
    ax.plot(x_vals_reco, y_vals_reco, label=label + " (reco)", linestyle='solid')
    ax.plot(x_vals_gen, y_vals_gen, label=label + " (gen)", linestyle='dashed')
p = "../../idea_fullsim/fast_sim/Histograms_ECM240/{}/higgs_mass_reco_vs_gen.pdf".format(os.environ["FOLDER_NAME"])
fig.savefig(p)
print("saving to", p)
