# Print the number of events that pass selection criteria. The total number of events is stored in sumOfWeights in the root file.
# For the number of passed events, take the histogram h_mH_stable_gt_particles, and look at the total number of events (sum of all bins)
import ROOT
import os
import numpy as np

from Colors import PROCESS_COLORS, HUMAN_READABLE_PROCESS_NAMES, LINE_STYLES
import pickle

#base_dir = "../../idea_fullsim/fast_sim/Histograms_ECM240_20251114_6M"

base_dir = "../../idea_fullsim/fast_sim/Histograms_20251112_Debug"

for folder_name in os.listdir(base_dir):
    if not os.path.isdir(os.path.join(base_dir, folder_name)):
        continue
    print("###############################################")
    print(f"    Processing folder: {folder_name}")
    print("###############################################")
    inputDir = os.path.join(base_dir, folder_name)
    # Get all ROOT files in the directory
    pkl_files = [f for f in os.listdir(inputDir) if f.endswith(".pkl") and f.startswith("basic_stats_")]
    for fname in sorted(pkl_files):
        file_path = os.path.join(inputDir, fname)
        # open the pickle file
        with open(file_path, "rb") as f:
            stats = pickle.load(f)
        name = fname.replace("basic_stats_", "").replace(".pkl", "")
        number_of_events_total = stats["before_filtering"]
        integral = stats["after_filtering"]
        print(f"{HUMAN_READABLE_PROCESS_NAMES.get(name, name)}: N total: {number_of_events_total}, N passed: {integral}, Pass rate: {integral / number_of_events_total:.4f}")
