# Code to produce the plots with N jets vs resolution

import matplotlib.pyplot as plt
import pickle

methods = {
    "PFDurham_ISR": "PF Durham",
    "CaloJetDurham_ISR": "Calo Durham"
}

line_styles = {
    "PFDurham_ISR": "-",
    "CaloJetDurham_ISR": ":"
}

processes = {
    "ZH → b-jets": {
        2: "p8_ee_ZH_vvbb_ecm240",
        4: "p8_ee_ZH_bbbb_ecm240",
        6: "p8_ee_ZH_6jet_HF_ecm240"
    },
    "ZH → light jets": {
        2: "p8_ee_ZH_vvqq_ecm240",
        4: "p8_ee_ZH_qqgg_ecm240",
        6: "p8_ee_ZH_6jet_LF_ecm240"
    }
}

nJets_processList = {
    "p8_ee_ZH_qqbb_ecm240": 4,
    "p8_ee_ZH_6jet_ecm240": 6,
    "p8_ee_ZH_vvbb_ecm240": 2,
    "p8_ee_ZH_bbbb_ecm240": 4,
    "p8_ee_ZH_vvgg_ecm240": 2,
    "p8_ee_ZH_vvqq_ecm240": 2,
    "p8_ee_ZH_6jet_HF_ecm240": 6,
    "p8_ee_ZH_6jet_LF_ecm240": 6,
    "p8_ee_ZH_bbgg_ecm240": 4,
    "p8_ee_ZH_qqgg_ecm240": 4,
}

main_dir = "../../idea_fullsim/fast_sim/{hist_folder_name}/{method_name}/energy_fit_params_per_process.pkl"
hist_folder_name = "Histograms_ECM240_20251105"

fig, ax = plt.subplots(1, 1, figsize=(6, 3))

for method in methods:
    method_name = method
    file_path = main_dir.format(hist_folder_name=hist_folder_name, method_name=method_name)
    f = pickle.load(open(file_path, "rb"))
    for process in processes:
        njets = sorted(processes[process].keys())
        stochastic_terms = []
        for njet in njets:
            process_name = processes[process][njet]
            stochastic_terms.append(
                f[process_name]["gaussian_fit"][0][0]
            )
        ax.plot(
            njets,
            stochastic_terms,
            label=f"{methods[method]} - {process}",
            linestyle=line_styles[method],
            marker="o",
            color="C{}".format(list(processes.keys()).index(process))
    )


ax.legend()
ax.set_xlabel("Number of jets in event")
ax.set_ylabel("Stochastic term (A)")
fig.tight_layout()

fig.savefig("../../idea_fullsim/fast_sim/{hist_folder_name}/Njets_vs_stochastic_term.pdf".format(hist_folder_name=hist_folder_name))
