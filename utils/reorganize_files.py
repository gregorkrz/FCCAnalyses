import os
import shutil

# Example dictionary mapping folder names to new names

#### FOR AK
folder_map = {
    "GenJetDurhamFastJet_ISR": "Durham",
    "GenJetDurhamFastJet_ISR_AK": "AK4",
    "GenJetDurhamFastJet_ISR_AK10": "AK10",
    "GenJetDurhamFastJet_ISR_AK14": "AK14",
    "GenJetDurhamFastJet_ISR_AK6": "AK6",

    #"GenJetDurhamFastJet_NoISR": "Durham",
    #"GenJetDurhamFastJet_NoISR_AK": "AK4",
    #"GenJetDurhamFastJet_NoISR_AK10": "AK10",
    #"GenJetDurhamFastJet_NoISR_AK14": "AK14",
    #"GenJetDurhamFastJet_NoISR_AK6": "AK6"

}


folder_map = {
    "PFDurham_AK4_ISR": "AK4_PF",
    "PFDurham_AK6_ISR": "AK6_PF",
    "PFDurham_AK8_ISR": "AK8_PF",
    "PFDurham_AK10_ISR": "AK10_PF",
    "PFDurham_AK12_ISR": "AK12_PF",
    "PFDurham_ISR": "Durham_PF",
    "CaloJetDurham_ISR": "Durham_Calo"
}
'''
folder_map = {
    "CaloJetDurham_ISR": "CaloJets_Durham",
    "PFDurham_ISR": "PFJets_Durham"
}'''

# GenJetDurhamFastJet_ISR     GenJetDurhamFastJet_ISR_AK10  GenJetDurhamFastJet_ISR_AK6  GenJetDurhamFastJet_NoISR_AK
# GenJetDurhamFastJet_NoISR_AK14  GenJetDurhamFastJet_NoISR_AK8
# GenJetDurhamFastJet_ISR_AK  GenJetDurhamFastJet_ISR_AK14  GenJetDurhamFastJet_NoISR
# GenJetDurhamFastJet_NoISR_AK10  GenJetDurhamFastJet_NoISR_AK6


# Base directory (optional, use '.' if script is in the same place)
base_dir = "/sdf/home/g/gregork/idea_fullsim/fast_sim/Histograms_ECM240"
target_dir = "/sdf/home/g/gregork/idea_fullsim/fast_sim/Histograms_ECM240/Calo_vs_PF_and_AK_organized"

# Loop through each folder and its corresponding new name
for folder_name, new_name in folder_map.items():
    folder_path = os.path.join(base_dir, folder_name)
    # Find all .root files in this folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".root"):
            src_path = os.path.join(folder_path, filename)
            # Create destination folder based on filename (without extension)
            file_base = os.path.splitext(filename)[0]
            dest_dir = os.path.join(target_dir, file_base)
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, f"{new_name}.root")
            shutil.copy2(src_path, dest_path)
            print(f"Copied {src_path} → {dest_path}")
